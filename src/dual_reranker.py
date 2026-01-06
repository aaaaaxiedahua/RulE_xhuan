import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch.utils.data import DataLoader

from layers import MLP


@dataclass
class DualRerankConfig:
    k: int = 256
    neighbors: int = 16
    beta: float = 0.5
    dim: int = 128
    lt_hops: int = 1
    lh_hops: int = 1
    lr: float = 1e-3
    steps: int = 2000
    batch_size: int = 16
    neg_num: int = 32
    eval_every: int = 1000
    log_every: int = 200
    base: str = "kge"  # reserved; grounding-replacement uses KGE base


def _signed_relation_embedding(relation_embedding: nn.Embedding, rel_ids: torch.Tensor, num_relations: int) -> torch.Tensor:
    rel_ids = rel_ids.long()
    flag = torch.pow(-1, rel_ids // num_relations).unsqueeze(-1)
    base = rel_ids % num_relations
    return relation_embedding(base) * flag


class DualContextReranker(nn.Module):
    def __init__(self, graph, entity_embedding: nn.Embedding, relation_embedding: nn.Embedding, num_relations: int, dim: int = 128):
        super().__init__()
        self.graph = graph
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding
        self.num_relations = int(num_relations)

        ent_dim = int(entity_embedding.embedding_dim)
        rel_dim = int(relation_embedding.embedding_dim)
        self.dim = int(dim)

        self.edge_mlp = MLP(ent_dim + rel_dim + rel_dim, [self.dim, self.dim])
        self.ctx_mlp = MLP(ent_dim + self.dim + rel_dim, [self.dim, self.dim])
        self.score_mlp = MLP(self.dim + self.dim + rel_dim + 1, [self.dim, 1])

        self.null_edge_ctx = nn.Parameter(torch.zeros(self.dim))

    def _encode_nodes(self, node_ids: torch.Tensor, query_r: torch.Tensor, neighbors: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          ctx: [B, dim]
          has_edges: [B] bool
        """
        device = node_ids.device
        B = int(node_ids.size(0))

        node_ids_cpu = node_ids.detach().to("cpu")
        seed_pos_cpu, nbr_cpu, rel_cpu = self.graph.sample_out_edges(node_ids_cpu, neighbors=neighbors)

        q_emb = _signed_relation_embedding(self.relation_embedding, query_r, self.num_relations)
        node_emb = self.entity_embedding(node_ids)

        if seed_pos_cpu.numel() == 0:
            edge_ctx = self.null_edge_ctx.unsqueeze(0).expand(B, -1)
            has_edges = torch.zeros((B,), dtype=torch.bool, device=device)
        else:
            seed_pos = seed_pos_cpu.to(device)
            nbr = nbr_cpu.to(device)
            rel = rel_cpu.to(device)

            nbr_emb = self.entity_embedding(nbr)
            rel_emb = _signed_relation_embedding(self.relation_embedding, rel, self.num_relations)
            q_edge = q_emb[seed_pos]

            edge_in = torch.cat([nbr_emb, rel_emb, q_edge], dim=-1)
            edge_msg = self.edge_mlp(edge_in)

            edge_sum = scatter(edge_msg, seed_pos, dim=0, dim_size=B, reduce="sum")
            edge_cnt = scatter(torch.ones((edge_msg.size(0),), device=device), seed_pos, dim=0, dim_size=B, reduce="sum")
            has_edges = edge_cnt > 0
            edge_cnt = edge_cnt.clamp(min=1).unsqueeze(-1)
            edge_ctx = edge_sum / edge_cnt
            if (~has_edges).any():
                edge_ctx = torch.where(has_edges.unsqueeze(-1), edge_ctx, self.null_edge_ctx.unsqueeze(0).expand(B, -1))

        ctx_in = torch.cat([node_emb, edge_ctx, q_emb], dim=-1)
        ctx = self.ctx_mlp(ctx_in)
        return ctx, has_edges

    def forward(
        self,
        all_h: torch.Tensor,
        all_r: torch.Tensor,
        cand_t: torch.Tensor,
        base_scores: Optional[torch.Tensor] = None,
        neighbors: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          all_h: [B]
          all_r: [B]
          cand_t: [B, K]
          base_scores: [B, K] optional (e.g., kge scores for candidates)
        Returns:
          delta: [B, K]
          mask_h: [B] bool (head has sampled edges)
          mask_t: [B, K] bool (tail has sampled edges)
        """
        device = all_h.device
        B, K = int(cand_t.size(0)), int(cand_t.size(1))

        ctx_h, mask_h = self._encode_nodes(all_h, all_r, neighbors=neighbors)

        flat_t = cand_t.reshape(-1)
        flat_r = all_r.unsqueeze(1).expand(-1, K).reshape(-1)
        ctx_t, mask_t_flat = self._encode_nodes(flat_t, flat_r, neighbors=neighbors)
        ctx_t = ctx_t.view(B, K, -1)
        mask_t = mask_t_flat.view(B, K)

        q_emb = _signed_relation_embedding(self.relation_embedding, all_r, self.num_relations).unsqueeze(1).expand(-1, K, -1)
        ctx_h = ctx_h.unsqueeze(1).expand(-1, K, -1)
        if base_scores is None:
            base_scores = torch.zeros((B, K), device=device)
        score_in = torch.cat([ctx_h, ctx_t, q_emb, base_scores.unsqueeze(-1)], dim=-1)
        delta = self.score_mlp(score_in).squeeze(-1)
        return delta, mask_h, mask_t


class DualRerankTrainer:
    def __init__(self, model, graph, device, config: DualRerankConfig, save_path: str):
        self.model = model
        self.graph = graph
        self.device = device
        self.config = config
        self.save_path = save_path

    def _get_base_logits(self, all_h: torch.Tensor, all_r: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        kge = self.model.compute_g_KGE(all_h, all_r)
        if self.config.base == "kge":
            return kge.detach(), kge
        rule_logits, _, _ = self.model(all_h, all_r, None)
        base = (rule_logits + float(alpha) * kge).detach()
        return base, kge

    def train(self, alpha: float = 3.0, valid_set=None, num_worker: int = 0) -> None:
        cfg = self.config
        reranker: DualContextReranker = self.model.dual_reranker
        reranker.train()

        for p in self.model.entity_embedding.parameters():
            p.requires_grad = False
        for p in self.model.relation_embedding.parameters():
            p.requires_grad = False

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, reranker.parameters()), lr=float(cfg.lr))

        facts = self.graph.ground_train_facts
        n = len(facts)
        hr2o = self.graph.hr2o
        nentity = self.graph.entity_size

        loss_ema = None
        best_valid_mrr = None
        best_state = None
        stats_pos_in_topk = 0
        stats_random_negs = 0
        stats_filtered_true = 0
        stats_samples = 0
        for step in range(1, int(cfg.steps) + 1):
            idx = torch.randint(0, n, (int(cfg.batch_size),))
            batch = [facts[i] for i in idx.tolist()]
            all_h = torch.tensor([h for h, _, _ in batch], device=self.device, dtype=torch.long)
            all_r = torch.tensor([r for _, r, _ in batch], device=self.device, dtype=torch.long)
            all_t = torch.tensor([t for _, _, t in batch], device=self.device, dtype=torch.long)

            base_all, kge_all = self._get_base_logits(all_h, all_r, alpha=alpha)
            K = int(cfg.k)
            topk_idx = torch.topk(kge_all, k=K, dim=1).indices.detach().to("cpu")

            cand = []
            labels = []
            for b in range(int(cfg.batch_size)):
                h = int(all_h[b].item())
                r = int(all_r[b].item())
                t_pos = int(all_t[b].item())
                key = self.graph.encode_hr(h, r)
                true_tails = set(hr2o.get(key, []))

                negs = []
                topk_list = topk_idx[b].tolist()
                stats_pos_in_topk += int(t_pos in topk_list)
                for t in topk_list:
                    if t == t_pos or (t in true_tails):
                        stats_filtered_true += int(t != t_pos and (t in true_tails))
                        continue
                    negs.append(int(t))
                    if len(negs) >= K - 1:
                        break
                while len(negs) < K - 1:
                    t = int(torch.randint(0, nentity, (1,)).item())
                    if t == t_pos or (t in true_tails):
                        continue
                    negs.append(t)
                    stats_random_negs += 1

                c = [t_pos] + negs[: K - 1]
                perm = torch.randperm(K)
                c_tensor = torch.tensor(c, dtype=torch.long)[perm]
                label = int((perm == 0).nonzero(as_tuple=False).item())

                cand.append(c_tensor)
                labels.append(label)

            cand_t = torch.stack(cand, dim=0).to(self.device)  # [B, K]
            labels = torch.tensor(labels, device=self.device, dtype=torch.long)

            base_cand = base_all.gather(1, cand_t)
            kge_cand = kge_all.gather(1, cand_t).detach()

            delta, _, mask_t = reranker(all_h, all_r, cand_t, base_scores=kge_cand, neighbors=int(cfg.neighbors))
            delta = torch.tanh(delta)
            active = mask_t.float()
            mean = (delta * active).sum(dim=1, keepdim=True) / active.sum(dim=1, keepdim=True).clamp(min=1)
            delta = (delta - mean) * active

            logits = base_cand + float(cfg.beta) * delta
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ema = float(loss.item()) if loss_ema is None else (0.95 * loss_ema + 0.05 * float(loss.item()))
            stats_samples += int(cfg.batch_size)
            if int(cfg.log_every) > 0 and step % int(cfg.log_every) == 0:
                pos_in_topk_rate = stats_pos_in_topk / max(stats_samples, 1)
                rand_neg_per_sample = stats_random_negs / max(stats_samples, 1)
                filtered_true_per_sample = stats_filtered_true / max(stats_samples, 1)
                logging.info(
                    "DualRerank train step=%d loss=%.6f ema=%.6f K=%d posInTopK=%.3f randNeg=%.2f filteredTrue=%.2f",
                    step,
                    float(loss.item()),
                    float(loss_ema),
                    int(K),
                    float(pos_in_topk_rate),
                    float(rand_neg_per_sample),
                    float(filtered_true_per_sample),
                )

            eval_every = int(getattr(cfg, "eval_every", 0) or 0)
            if valid_set is not None and eval_every > 0 and step % eval_every == 0:
                valid_mrr = self.evaluate_dataset(valid_set, split="valid", num_worker=num_worker, expectation=True)
                if best_valid_mrr is None or valid_mrr > best_valid_mrr:
                    best_valid_mrr = float(valid_mrr)
                    best_state = {k: v.detach().cpu().clone() for k, v in reranker.state_dict().items()}
                    logging.info("DualRerank best valid MRR=%.6f at step=%d", best_valid_mrr, step)

        if best_state is not None:
            reranker.load_state_dict(best_state, strict=True)
        self.save()

    def save(self) -> None:
        path = os.path.join(self.save_path, "dual_reranker.pt")
        torch.save({"dual_reranker": self.model.dual_reranker.state_dict(), "config": self.config.__dict__}, path)
        logging.info("Saved dual reranker to %s", path)

    def load_if_exists(self) -> bool:
        path = os.path.join(self.save_path, "dual_reranker.pt")
        if not os.path.exists(path):
            return False
        state = torch.load(path, map_location=self.device)
        self.model.dual_reranker.load_state_dict(state["dual_reranker"], strict=True)
        logging.info("Loaded dual reranker from %s", path)
        return True

    @torch.no_grad()
    def evaluate_dataset(self, dataset, split: str, num_worker: int = 0, expectation: bool = True) -> float:
        cfg = self.config
        model = self.model
        reranker: DualContextReranker = model.dual_reranker
        model.eval()
        reranker.eval()

        dataloader = DataLoader(dataset, batch_size=1, num_workers=num_worker)
        ranks = []
        log_calls = 0
        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0).to(self.device)
            all_r = all_r.squeeze(0).to(self.device)
            all_t = all_t.squeeze(0).to(self.device)
            flag = flag.squeeze(0).to(self.device)
            B = int(all_h.numel())

            kge_score = model.compute_g_KGE(all_h, all_r)
            logits_base = kge_score

            K = min(int(cfg.k), int(kge_score.size(1)))
            cand_t = torch.topk(kge_score, k=K, dim=1).indices
            base_cand = kge_score.gather(1, cand_t).detach()
            delta, mask_h, mask_t = reranker(all_h, all_r, cand_t, base_scores=base_cand, neighbors=int(cfg.neighbors))
            delta = torch.tanh(delta)
            active = mask_t.float()
            mean = (delta * active).sum(dim=1, keepdim=True) / active.sum(dim=1, keepdim=True).clamp(min=1)
            delta = (delta - mean) * active
            logits = logits_base.clone()
            logits.scatter_add_(1, cand_t, float(cfg.beta) * delta)

            for i in range(B):
                t = int(all_t[i].item())
                val = logits[i, t]
                fi = flag[i]
                L = (logits[i][fi] > val).sum().item() + 1
                H = (logits[i][fi] >= val).sum().item() + 2
                ranks.append((int(all_h[i].item()), int(all_r[i].item()), t, int(L), int(H)))

            log_calls += 1
            log_every = int(cfg.log_every)
            if log_every > 0 and log_calls % log_every == 0:
                in_c_list = []
                before_list = []
                after_list = []
                for i in range(B):
                    t = int(all_t[i].item())
                    fi = flag[i]
                    in_c_list.append(float((cand_t[i] == t).any().item()))
                    before_list.append(float((logits_base[i][fi] > logits_base[i, t]).sum().item() + 1))
                    after_list.append(float((logits[i][fi] > logits[i, t]).sum().item() + 1))
                in_c = sum(in_c_list) / max(len(in_c_list), 1)
                before = sum(before_list) / max(len(before_list), 1)
                after = sum(after_list) / max(len(after_list), 1)
                delta_abs = float(delta.abs().mean().item())
                delta_max = float(delta.abs().max().item())
                logging.info(
                    "DualRerank split=%s inCandRate=%.3f rankBaseAvg=%.2f rankFinalAvg=%.2f maskHRate=%.3f candK=%d | deltaAbs=%.4f deltaMax=%.4f",
                    split,
                    float(in_c),
                    float(before),
                    float(after),
                    float(mask_h.float().mean().item()),
                    int(K),
                    delta_abs,
                    delta_max,
                )

        ranks = torch.tensor(ranks, dtype=torch.long, device=self.device)
        query2LH = {(int(h), int(r), int(t)): (int(L), int(H)) for h, r, t, L, H in ranks.data.cpu().numpy().tolist()}

        hit1 = hit3 = hit10 = mr = mrr = 0.0
        for (L, H) in query2LH.values():
            if expectation:
                for rank in range(L, H):
                    if rank <= 1:
                        hit1 += 1.0 / (H - L)
                    if rank <= 3:
                        hit3 += 1.0 / (H - L)
                    if rank <= 10:
                        hit10 += 1.0 / (H - L)
                    mr += rank / (H - L)
                    mrr += 1.0 / rank / (H - L)
            else:
                rank = H - 1
                if rank <= 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 10:
                    hit10 += 1
                mr += rank
                mrr += 1.0 / rank

        denom = max(len(ranks), 1)
        hit1 /= denom
        hit3 /= denom
        hit10 /= denom
        mr /= denom
        mrr /= denom

        logging.info(">>>>> DualRerank: Evaluating on %s", split)
        logging.info("Data : %d", len(query2LH))
        logging.info("Hit1 : %.6f", hit1)
        logging.info("Hit3 : %.6f", hit3)
        logging.info("Hit10: %.6f", hit10)
        logging.info("MR   : %.6f", mr)
        logging.info("MRR  : %.6f", mrr)
        return float(mrr)
