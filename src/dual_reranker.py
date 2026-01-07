import os
import logging
from dataclasses import dataclass
from collections import Counter
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
    beta: float = 0.5
    dim: int = 128
    layers: int = 2
    node_topk: int = 128
    rule_gamma: float = 0.0
    rule_eps: float = 1e-3
    use_rule_weight: bool = True
    lr: float = 1e-3
    steps: int = 2000
    batch_size: int = 16
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

        # DualContextReranker is kept for backward experimentation; not used in the dual_rerank path by default.
        ctx_h, mask_h = self._encode_nodes(all_h, all_r, neighbors=16)

        flat_t = cand_t.reshape(-1)
        flat_r = all_r.unsqueeze(1).expand(-1, K).reshape(-1)
        ctx_t, mask_t_flat = self._encode_nodes(flat_t, flat_r, neighbors=16)
        ctx_t = ctx_t.view(B, K, -1)
        mask_t = mask_t_flat.view(B, K)

        q_emb = _signed_relation_embedding(self.relation_embedding, all_r, self.num_relations).unsqueeze(1).expand(-1, K, -1)
        ctx_h = ctx_h.unsqueeze(1).expand(-1, K, -1)
        if base_scores is None:
            base_scores = torch.zeros((B, K), device=device)
        score_in = torch.cat([ctx_h, ctx_t, q_emb, base_scores.unsqueeze(-1)], dim=-1)
        delta = self.score_mlp(score_in).squeeze(-1)
        return delta, mask_h, mask_t


class HeadGNNReranker(nn.Module):
    """
    Head-centered multi-hop message passing over a sampled subgraph from head.
    Scores only candidate tails; if a candidate is not reached in the sampled subgraph, delta=0 for that candidate.
    """

    def __init__(
        self,
        graph,
        entity_embedding: nn.Embedding,
        relation_embedding: nn.Embedding,
        num_relations: int,
        dim: int = 128,
        layers: int = 2,
        node_topk: int = 128,
        rule_gamma: float = 0.0,
        rule_eps: float = 1e-3,
        use_rule_weight: bool = True,
    ):
        super().__init__()
        self.graph = graph
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding
        self.num_relations = int(num_relations)
        self.dim = int(dim)
        self.layers = int(layers)
        self.node_topk = int(node_topk)
        self.rule_gamma = float(rule_gamma)
        self.rule_eps = float(rule_eps)
        self.use_rule_weight = bool(use_rule_weight)

        ent_dim = int(entity_embedding.embedding_dim)
        rel_dim = int(relation_embedding.embedding_dim)
        self.ent_proj = nn.Linear(ent_dim, self.dim)
        self.layer_emb = nn.Embedding(max(self.layers, 1) + 1, rel_dim)
        self.msg_mlp = MLP(2 * self.dim + 3 * rel_dim, [self.dim, self.dim])
        self.gru = nn.GRUCell(self.dim, self.dim)
        self.select_mlp = MLP(self.dim + rel_dim, [self.dim, 1])
        self.score_mlp = MLP(self.dim + rel_dim + 1, [self.dim, 1])
        self.start = nn.Parameter(torch.zeros(self.dim))

        self.relation2rules = None
        self._rule_weight = None  # CPU float tensor after sigmoid
        self._prior_cache = {}
        self.last_stats = None

    def set_rules(self, relation2rules, rules_weight_emb: Optional[torch.Tensor] = None) -> None:
        self.relation2rules = relation2rules
        self._prior_cache = {}
        self._rule_weight = None
        if rules_weight_emb is None or not self.use_rule_weight:
            return
        try:
            w = torch.sigmoid(rules_weight_emb.detach().to("cpu").float()).view(-1)
        except Exception:
            w = None
        self._rule_weight = w

    def _rule_prior_tensor(self, query_r: int, hop: int) -> Optional[torch.Tensor]:
        if self.relation2rules is None:
            return None
        if query_r < 0 or query_r >= len(self.relation2rules):
            return None
        key = (int(query_r), int(hop))
        if key in self._prior_cache:
            return self._prior_cache[key]

        rules = self.relation2rules[int(query_r)]
        if not rules:
            self._prior_cache[key] = None
            return None

        prior = torch.zeros((self.num_relations * 2,), dtype=torch.float)
        for rule_id, (_, body) in rules:
            pos = int(hop) - 1
            if pos < 0 or pos >= len(body):
                continue
            rel = int(body[pos])
            if rel < 0 or rel >= int(prior.numel()):
                continue
            w = 1.0
            if self._rule_weight is not None and int(rule_id) < int(self._rule_weight.numel()):
                w = float(self._rule_weight[int(rule_id)].item())
            if w <= 0:
                continue
            prior[rel] += float(w)

        if float(prior.sum().item()) <= 0:
            self._prior_cache[key] = None
            return None
        prior = prior / prior.sum()
        self._prior_cache[key] = prior
        return prior

    def _get_out_edges_cpu(self, node_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.graph.build_neighbor_index()
        start = int(self.graph._head_ptr[int(node_id)].item())
        end = int(self.graph._head_ptr[int(node_id) + 1].item())
        if end <= start:
            empty = torch.empty((0,), dtype=torch.long)
            return empty, empty
        return self.graph._edge_tail_sorted[start:end], self.graph._edge_rel_sorted[start:end]

    def _sample_edges_cpu(self, node_id: int, k: int, rel_prior: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("_sample_edges_cpu removed: this reranker no longer performs per-node neighbor sampling.")

    def forward(
        self,
        all_h: torch.Tensor,
        all_r: torch.Tensor,
        cand_t: torch.Tensor,
        base_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = all_h.device
        B, K = int(cand_t.size(0)), int(cand_t.size(1))
        if base_scores is None:
            base_scores = torch.zeros((B, K), device=device)

        q_emb = _signed_relation_embedding(self.relation_embedding, all_r, self.num_relations)

        deltas = torch.zeros((B, K), device=device)
        mask_t = torch.zeros((B, K), dtype=torch.bool, device=device)
        mask_h = torch.zeros((B,), dtype=torch.bool, device=device)
        stats = {
            "visited_nodes": 0.0,
            "sampled_edges": 0.0,
            "prior_edges": 0.0,
            "prior_nonempty": 0.0,
        }

        for i in range(B):
            h = int(all_h[i].item())
            r = int(all_r[i].item())
            cand_cpu = cand_t[i].detach().to("cpu")
            delta_i, mask_i, head_ok, one_stats = self._score_one(h, r, cand_cpu, q_emb[i], base_scores[i])
            deltas[i] = delta_i
            mask_t[i] = mask_i
            mask_h[i] = bool(head_ok)
            stats["visited_nodes"] += float(one_stats.get("visited_nodes", 0.0))
            stats["sampled_edges"] += float(one_stats.get("sampled_edges", 0.0))
            stats["prior_edges"] += float(one_stats.get("prior_edges", 0.0))
            stats["prior_nonempty"] += float(one_stats.get("prior_nonempty", 0.0))

        denom = max(B, 1)
        self.last_stats = {
            "visited_nodes": float(stats["visited_nodes"] / denom),
            "sampled_edges": float(stats["sampled_edges"] / denom),
            "prior_edges": float(stats["prior_edges"] / denom),
            "prior_nonempty": float(stats["prior_nonempty"] / denom),
        }

        return deltas, mask_h, mask_t

    def _score_one(
        self,
        h: int,
        r: int,
        cand_t_cpu: torch.Tensor,
        q_emb: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, dict]:
        device = q_emb.device
        K = int(cand_t_cpu.numel())

        nodes = [int(h)]
        mapping = {int(h): 0}
        visited = {int(h)}
        frontier = [int(h)]

        h_emb = self.entity_embedding(torch.tensor([int(h)], device=device))
        x = self.ent_proj(h_emb).view(1, -1) + self.start.view(1, -1)

        head_ok = False
        sampled_edges_total = 0
        prior_edges_total = 0
        prior_nonempty = 0

        for layer in range(1, int(self.layers) + 1):
            max_nodes = min(int(self.graph.entity_size), 1 + int(self.layers) * max(int(self.node_topk), 0))
            if not frontier or len(nodes) >= int(max_nodes):
                break

            rel_prior = self._rule_prior_tensor(int(r), int(layer))
            prior_nonempty += float(rel_prior is not None)

            edges = []
            rel_hist = Counter()
            for u in frontier:
                tails, rels = self._get_out_edges_cpu(int(u))
                if int(tails.numel()) == 0:
                    continue
                if layer == 1 and int(u) == int(h):
                    head_ok = True
                for v, rr in zip(tails.tolist(), rels.tolist()):
                    edges.append((int(u), int(rr), int(v)))
                    rel_hist[int(rr)] += 1

            sampled_edges_total += int(len(edges))
            if rel_prior is not None and len(edges) > 0:
                for _, rr, _ in edges:
                    prior_edges_total += int(float(rel_prior[int(rr)].item()) > 0.0)

            if not edges:
                frontier = []
                continue

            # Optional rule bias weight on messages (AdaProp style expands full neighbors; we bias the message, not sampling).
            edge_w = None
            if rel_prior is not None and float(self.rule_gamma) > 0:
                edge_w = (1.0 - float(self.rule_gamma)) + float(self.rule_gamma) * rel_prior

            new_nodes = []
            new_pos = {}
            edge_src_local = []
            edge_rel = []
            edge_dst_pos = []
            edge_weight = []
            for u, rr, v in edges:
                if v in visited:
                    continue
                pos = new_pos.get(v)
                if pos is None:
                    pos = len(new_nodes)
                    new_pos[v] = pos
                    new_nodes.append(v)
                edge_src_local.append(mapping[int(u)])
                edge_rel.append(int(rr))
                edge_dst_pos.append(int(pos))
                if edge_w is None:
                    edge_weight.append(1.0)
                else:
                    edge_weight.append(float(edge_w[int(rr)].item()) + float(self.rule_eps))

            if not new_nodes:
                frontier = []
                continue

            remaining = int(max_nodes) - int(len(nodes))
            if remaining <= 0:
                break

            # Compute hidden for all newly discovered nodes (before layer topk filtering).
            src_idx = torch.tensor(edge_src_local, dtype=torch.long, device=device)
            dst_pos_t = torch.tensor(edge_dst_pos, dtype=torch.long, device=device)
            rel_ids = torch.tensor(edge_rel, dtype=torch.long, device=device)

            new_ids = torch.tensor(new_nodes, dtype=torch.long, device=device)
            dst_init = self.ent_proj(self.entity_embedding(new_ids))

            rel_emb = _signed_relation_embedding(self.relation_embedding, rel_ids, self.num_relations)
            layer_e = self.layer_emb(torch.tensor([int(layer)], device=device)).expand(rel_emb.size(0), -1)
            q_e = q_emb.unsqueeze(0).expand(rel_emb.size(0), -1)

            msg_in = torch.cat([x[src_idx], rel_emb, q_e, layer_e, dst_init[dst_pos_t]], dim=-1)
            msg = self.msg_mlp(msg_in)
            if edge_weight:
                ew = torch.tensor(edge_weight, device=device, dtype=msg.dtype).unsqueeze(-1)
                msg = msg * ew
            agg = scatter(msg, dst_pos_t, dim=0, dim_size=int(new_ids.size(0)), reduce="sum")
            new_hidden = self.gru(agg, dst_init)

            # AdaProp-style: select topk NEW nodes only; no duplicates across layers.
            k_keep = min(int(self.node_topk), int(new_hidden.size(0)), int(remaining))
            if k_keep <= 0:
                break

            sel_in = torch.cat([new_hidden, q_emb.unsqueeze(0).expand(new_hidden.size(0), -1)], dim=-1)
            sel = self.select_mlp(sel_in).squeeze(-1)
            top_idx = torch.topk(sel, k=int(k_keep), dim=0).indices.detach().to("cpu")

            selected_nodes = [int(new_nodes[j]) for j in top_idx.tolist()]
            selected_hidden = new_hidden[top_idx.to(device)]

            for nid in selected_nodes:
                visited.add(int(nid))
                mapping[int(nid)] = len(nodes)
                nodes.append(int(nid))

            x = torch.cat([x, selected_hidden], dim=0)
            frontier = selected_nodes

        idx = torch.zeros((K,), dtype=torch.long, device=device)
        mask = torch.zeros((K,), dtype=torch.bool, device=device)
        for j, t in enumerate(cand_t_cpu.tolist()):
            lj = mapping.get(int(t), None)
            if lj is not None:
                idx[j] = int(lj)
                mask[j] = True

        xt = x[idx]
        qk = q_emb.unsqueeze(0).expand(K, -1)
        score_in = torch.cat([xt, qk, base_scores.unsqueeze(-1)], dim=-1)
        delta = self.score_mlp(score_in).squeeze(-1)
        delta = delta * mask.float()
        stats = {
            "visited_nodes": float(len(nodes)),
            "sampled_edges": float(sampled_edges_total),
            "prior_edges": float(prior_edges_total),
            "prior_nonempty": float(prior_nonempty),
        }
        return delta, mask, bool(head_ok), stats


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
        reranker: nn.Module = self.model.dual_reranker
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

            delta, _, mask_t = reranker(all_h, all_r, cand_t, base_scores=kge_cand)
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
                pos_visited = mask_t.gather(1, labels.view(-1, 1)).float().mean().item()
                cand_visited = mask_t.float().mean().item()
                pos_in_topk_rate = stats_pos_in_topk / max(stats_samples, 1)
                rand_neg_per_sample = stats_random_negs / max(stats_samples, 1)
                filtered_true_per_sample = stats_filtered_true / max(stats_samples, 1)
                extra = ""
                if hasattr(reranker, "last_stats") and isinstance(getattr(reranker, "last_stats"), dict):
                    ls = reranker.last_stats
                    extra = " | visited=%.1f edges=%.1f priorEdges=%.1f priorNonEmpty=%.2f" % (
                        float(ls.get("visited_nodes", 0.0)),
                        float(ls.get("sampled_edges", 0.0)),
                        float(ls.get("prior_edges", 0.0)),
                        float(ls.get("prior_nonempty", 0.0)),
                    )
                logging.info(
                    "DualRerank train step=%d loss=%.6f ema=%.6f K=%d posInTopK=%.3f posVisited=%.3f candVisited=%.3f randNeg=%.2f filteredTrue=%.2f%s",
                    step,
                    float(loss.item()),
                    float(loss_ema),
                    int(K),
                    float(pos_in_topk_rate),
                    float(pos_visited),
                    float(cand_visited),
                    float(rand_neg_per_sample),
                    float(filtered_true_per_sample),
                    extra,
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
        ranks_base = []
        ranks_final = []
        improved = 0
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
            delta, mask_h, mask_t = reranker(all_h, all_r, cand_t, base_scores=base_cand)
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
                val0 = logits_base[i, t]
                L0 = (logits_base[i][fi] > val0).sum().item() + 1
                H0 = (logits_base[i][fi] >= val0).sum().item() + 2
                L1 = (logits[i][fi] > val).sum().item() + 1
                H1 = (logits[i][fi] >= val).sum().item() + 2
                ranks_base.append((int(all_h[i].item()), int(all_r[i].item()), t, int(L0), int(H0)))
                ranks_final.append((int(all_h[i].item()), int(all_r[i].item()), t, int(L1), int(H1)))
                if (L1 + H1) < (L0 + H0):
                    improved += 1

            log_calls += 1
            log_every = int(cfg.log_every)
            if log_every > 0 and log_calls % log_every == 0:
                in_c_list = []
                pos_visited_list = []
                before_list = []
                after_list = []
                for i in range(B):
                    t = int(all_t[i].item())
                    fi = flag[i]
                    hit = (cand_t[i] == t)
                    in_c = bool(hit.any().item())
                    in_c_list.append(float(in_c))
                    pos_visited_list.append(float((mask_t[i][hit].any().item()) if in_c else 0.0))
                    before_list.append(float((logits_base[i][fi] > logits_base[i, t]).sum().item() + 1))
                    after_list.append(float((logits[i][fi] > logits[i, t]).sum().item() + 1))
                in_c = sum(in_c_list) / max(len(in_c_list), 1)
                pos_v = sum(pos_visited_list) / max(len(pos_visited_list), 1)
                before = sum(before_list) / max(len(before_list), 1)
                after = sum(after_list) / max(len(after_list), 1)
                delta_abs = float(delta.abs().mean().item())
                delta_max = float(delta.abs().max().item())
                cand_v = float(mask_t.float().mean().item())
                extra = ""
                if hasattr(reranker, "last_stats") and isinstance(getattr(reranker, "last_stats"), dict):
                    ls = reranker.last_stats
                    extra = " | visited=%.1f edges=%.1f priorEdges=%.1f priorNonEmpty=%.2f" % (
                        float(ls.get("visited_nodes", 0.0)),
                        float(ls.get("sampled_edges", 0.0)),
                        float(ls.get("prior_edges", 0.0)),
                        float(ls.get("prior_nonempty", 0.0)),
                    )
                logging.info(
                    "DualRerank split=%s inCandRate=%.3f posVisited=%.3f candVisited=%.3f rankBaseAvg=%.2f rankFinalAvg=%.2f candK=%d | deltaAbs=%.4f deltaMax=%.4f%s",
                    split,
                    float(in_c),
                    float(pos_v),
                    float(cand_v),
                    float(before),
                    float(after),
                    int(K),
                    delta_abs,
                    delta_max,
                    extra,
                )

        query2LH_base = {(int(h), int(r), int(t)): (int(L), int(H)) for h, r, t, L, H in ranks_base}
        query2LH_final = {(int(h), int(r), int(t)): (int(L), int(H)) for h, r, t, L, H in ranks_final}

        def _metrics(query2lh):
            hit1 = hit3 = hit10 = mr = mrr = 0.0
            for (L, H) in query2lh.values():
                if expectation:
                    for rank in range(L, H):
                        p = 1.0 / (H - L)
                        if rank <= 1:
                            hit1 += p
                        if rank <= 3:
                            hit3 += p
                        if rank <= 10:
                            hit10 += p
                        mr += rank * p
                        mrr += (1.0 / rank) * p
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

            denom = max(len(query2lh), 1)
            return (hit1 / denom, hit3 / denom, hit10 / denom, mr / denom, mrr / denom)

        b1, b3, b10, bmr, bmrr = _metrics(query2LH_base)
        f1, f3, f10, fmr, fmrr = _metrics(query2LH_final)

        denom = max(len(query2LH_final), 1)
        imp_rate = float(improved) / float(denom)

        logging.info(">>>>> DualRerank: Evaluating on %s", split)
        logging.info("Data : %d | improved(midRank): %.3f", len(query2LH_final), imp_rate)
        logging.info("Base : Hit1 %.6f | Hit3 %.6f | Hit10 %.6f | MR %.6f | MRR %.6f", b1, b3, b10, bmr, bmrr)
        logging.info("Final: Hit1 %.6f | Hit3 %.6f | Hit10 %.6f | MR %.6f | MRR %.6f", f1, f3, f10, fmr, fmrr)
        logging.info("Delta: Hit1 %+0.6f | Hit3 %+0.6f | Hit10 %+0.6f | MR %+0.6f | MRR %+0.6f", f1 - b1, f3 - b3, f10 - b10, fmr - bmr, fmrr - bmrr)
        return float(fmrr)
