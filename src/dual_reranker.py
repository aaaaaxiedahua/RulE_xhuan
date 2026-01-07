import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch.utils.data import DataLoader

from layers import MLP

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


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
    tau: float = 1.0
    edge_topk: int = -1
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


class AdaPropGNNLayer(nn.Module):
    """
    Minimal port of AdaProp's GNNLayer:
    - edge attention conditioned on query relation
    - optional edge topk
    - node topk sampling for newly discovered nodes (straight-through estimator)
    Relation id space: [0, 2R) plus idd at 2R.
    """

    def __init__(
        self,
        dim: int,
        attn_dim: int,
        n_rel_base: int,
        n_ent: int,
        n_node_topk: int = -1,
        n_edge_topk: int = -1,
        tau: float = 1.0,
        act=None,
    ):
        super().__init__()
        self.n_rel_base = int(n_rel_base)
        self.n_ent = int(n_ent)
        self.dim = int(dim)
        self.attn_dim = int(attn_dim)
        self.n_node_topk = int(n_node_topk)
        self.n_edge_topk = int(n_edge_topk)
        self.tau = float(tau)
        self.act = act if act is not None else (lambda x: x)

        self.rela_embed = nn.Embedding(2 * self.n_rel_base + 1, self.dim)
        self.Ws_attn = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.Wr_attn = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(self.dim, self.attn_dim)
        self.w_alpha = nn.Linear(self.attn_dim, 1)
        self.W_h = nn.Linear(self.dim, self.dim, bias=False)
        self.W_samp = nn.Linear(self.dim, 1, bias=False)

        self._softmax = None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.training and self.tau > 0:
            self._softmax = lambda x: F.gumbel_softmax(x, tau=float(self.tau), hard=False)
        else:
            self._softmax = lambda x: F.softmax(x, dim=1)
        return self

    def forward(
        self,
        q_rel: torch.Tensor,  # [B]
        hidden: torch.Tensor,  # [N_prev, dim]
        edges: torch.Tensor,  # [E, 6] (batch, head, rel, tail, head_idx, tail_idx)
        nodes: torch.Tensor,  # [N, 2] (batch, node)
        old_nodes_new_idx: torch.Tensor,  # [N_prev]
        batch_size: int,
        edge_prior_logit: Optional[torch.Tensor] = None,  # [E]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = nodes.device
        if self._softmax is None:
            self._softmax = lambda x: F.softmax(x, dim=1)

        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]

        hs = hidden[sub]
        hr = self.rela_embed(rel)
        batch_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[batch_idx]

        n_node = int(nodes.size(0))
        message = hs + hr

        alpha_logit = self.w_alpha(F.relu(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))).squeeze(-1)
        if edge_prior_logit is not None:
            alpha_logit = alpha_logit + edge_prior_logit.to(device=device, dtype=alpha_logit.dtype)

        if self.n_edge_topk > 0:
            edge_prob = F.gumbel_softmax(alpha_logit, tau=1.0, hard=False)
            topk_index = torch.argsort(edge_prob, descending=True)[: int(self.n_edge_topk)]
            edge_prob_hard = torch.zeros_like(alpha_logit)
            edge_prob_hard[topk_index] = 1.0
            alpha_logit = alpha_logit * (edge_prob_hard - edge_prob.detach() + edge_prob)

        alpha = torch.sigmoid(alpha_logit).unsqueeze(-1)
        message = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce="sum")
        hidden_new = self.act(self.W_h(message_agg)).clone()

        if self.n_node_topk <= 0:
            return hidden_new, nodes, torch.ones((n_node,), dtype=torch.bool, device=device)

        tmp_diff = torch.ones((n_node,), device=device)
        tmp_diff[old_nodes_new_idx] = 0
        bool_diff = tmp_diff.bool()
        diff_node = nodes[bool_diff]

        # No new nodes to sample from: keep everything (all nodes are "old").
        if diff_node.numel() == 0:
            return hidden_new, nodes, torch.ones((n_node,), dtype=torch.bool, device=device)

        diff_logit = self.W_samp(hidden_new[bool_diff]).squeeze(-1)

        node_scores = torch.full((int(batch_size), self.n_ent), float("-inf"), device=device)
        node_scores[diff_node[:, 0], diff_node[:, 1]] = diff_logit
        # Some batches may have no diff nodes (row stays all -inf); make them uniform to avoid NaNs.
        empty_row = torch.isinf(node_scores).all(dim=1)
        if empty_row.any():
            node_scores[empty_row] = 0.0
        node_scores = self._softmax(node_scores)

        k = min(int(self.n_node_topk), int(self.n_ent))
        topk_index = torch.topk(node_scores, k, dim=1).indices.reshape(-1)
        topk_batchidx = torch.arange(int(batch_size), device=device).repeat(k, 1).T.reshape(-1)
        batch_topk_nodes = torch.zeros((int(batch_size), self.n_ent), device=device)
        batch_topk_nodes[topk_batchidx, topk_index] = 1.0

        bool_sampled_diff = batch_topk_nodes[diff_node[:, 0], diff_node[:, 1]].bool()
        bool_keep = ~bool_diff
        bool_keep[bool_diff] = bool_sampled_diff

        diff_prob_hard = batch_topk_nodes[diff_node[:, 0], diff_node[:, 1]]
        diff_prob = node_scores[diff_node[:, 0], diff_node[:, 1]]
        hidden_new[bool_diff] = hidden_new[bool_diff] * (diff_prob_hard - diff_prob.detach() + diff_prob).unsqueeze(-1)

        new_nodes = nodes[bool_keep]
        hidden_new = hidden_new[bool_keep]
        return hidden_new, new_nodes, bool_keep


class HeadGNNReranker(nn.Module):
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
        tau: float = 1.0,
        edge_topk: int = -1,
    ):
        super().__init__()
        self.graph = graph
        self.entity_embedding = entity_embedding  # interface compatibility; not used by AdaProp scaffold
        self.relation_embedding = relation_embedding  # interface compatibility; not used by AdaProp scaffold
        self.num_relations = int(num_relations)  # base relation count (R)
        self.dim = int(dim)
        self.layers = int(layers)
        self.node_topk = int(node_topk)
        self.rule_gamma = float(rule_gamma)
        self.rule_eps = float(rule_eps)
        self.use_rule_weight = bool(use_rule_weight)
        self.tau = float(tau)
        self.edge_topk = int(edge_topk)

        self.attn_dim = max(self.dim // 2, 8)
        self.dropout = nn.Dropout(0.1)
        self.gate = nn.GRU(self.dim, self.dim)
        self.W_final = nn.Linear(self.dim, 1, bias=False)
        nn.init.zeros_(self.W_final.weight)

        n_ent = int(self.graph.entity_size)
        self.gnn_layers = nn.ModuleList(
            [
                AdaPropGNNLayer(
                    dim=self.dim,
                    attn_dim=self.attn_dim,
                    n_rel_base=self.num_relations,
                    n_ent=n_ent,
                    n_node_topk=self.node_topk,
                    n_edge_topk=self.edge_topk,
                    tau=self.tau,
                    act=lambda x: x,
                )
                for _ in range(max(self.layers, 1))
            ]
        )

        self.relation2rules = None
        self._rule_weight = None  # CPU float tensor after sigmoid
        self._prior_cache: Dict[Tuple[int, int], Optional[torch.Tensor]] = {}
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

    def _rule_prior_tensor(self, query_r: int, layer: int) -> Optional[torch.Tensor]:
        if self.relation2rules is None:
            return None
        if query_r < 0 or query_r >= len(self.relation2rules):
            return None
        key = (int(query_r), int(layer))
        if key in self._prior_cache:
            return self._prior_cache[key]

        rules = self.relation2rules[int(query_r)]
        if not rules:
            self._prior_cache[key] = None
            return None

        prior = torch.zeros((self.num_relations * 2,), dtype=torch.float)
        for rule_id, (_, body) in rules:
            pos = int(layer) - 1
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

    def _get_neighbors(self, nodes: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes_cpu = nodes.detach().to("cpu").long()
        idd_rel = 2 * int(self.num_relations)

        sampled_edges = []
        for b, u in nodes_cpu.tolist():
            tails, rels = self._get_out_edges_cpu(int(u))
            for v, rr in zip(tails.tolist(), rels.tolist()):
                sampled_edges.append((int(b), int(u), int(rr), int(v)))
            sampled_edges.append((int(b), int(u), int(idd_rel), int(u)))

        if not sampled_edges:
            sampled_edges = [(int(b), int(u), int(idd_rel), int(u)) for b, u in nodes_cpu.tolist()]

        sampled_edges = torch.tensor(sampled_edges, dtype=torch.long)

        _, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
        edges6 = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], dim=1)

        mask = edges6[:, 2] == int(idd_rel)
        old_nodes_new_idx = tail_index[mask].sort()[0]

        device = nodes.device
        return tail_nodes.to(device), edges6.to(device), old_nodes_new_idx.to(device)

    def _edge_prior_logit(self, q_rel_edge: torch.Tensor, layer: int, rel_ids: torch.Tensor) -> torch.Tensor:
        device = rel_ids.device
        if self.relation2rules is None or self.rule_gamma <= 0:
            return torch.zeros((int(rel_ids.numel()),), device=device, dtype=torch.float)

        idd_rel = 2 * int(self.num_relations)
        q_cpu = q_rel_edge.detach().to("cpu").long()
        rel_cpu = rel_ids.detach().to("cpu").long()
        out = torch.zeros((int(rel_cpu.numel()),), dtype=torch.float)

        for qr in torch.unique(q_cpu).tolist():
            prior = self._rule_prior_tensor(int(qr), int(layer))
            if prior is None:
                continue
            idx = (q_cpu == int(qr)) & (rel_cpu != int(idd_rel))
            if not idx.any():
                continue
            p = prior[rel_cpu[idx]].float().clamp(min=0)
            out[idx] = torch.log(p + float(self.rule_eps))

        return (float(self.rule_gamma) * out).to(device)

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

        n_ent = int(self.graph.entity_size)
        q_sub = all_h.long()
        q_rel = all_r.long()

        h0 = torch.zeros((1, B, self.dim), device=device)
        nodes = torch.cat([torch.arange(B, device=device).unsqueeze(1), q_sub.unsqueeze(1)], dim=1)  # [B,2]
        hidden = torch.zeros((B, self.dim), device=device)

        edges_total = 0.0
        prior_edge_total = 0.0
        prior_edge_nonzero = 0.0
        prior_nonempty = 0.0
        visited_mask = torch.zeros((B, n_ent), dtype=torch.bool, device=device)

        for layer_idx in range(int(self.layers)):
            nodes, edges, old_nodes_new_idx = self._get_neighbors(nodes, batch_size=B)
            n_node = int(nodes.size(0))
            edges_total += float(edges.size(0))
            prior_edge_total += float(edges.size(0))

            edge_batch = edges[:, 0]
            rel_ids = edges[:, 2]
            q_rel_edge = q_rel[edge_batch]
            prior_nonempty += float(any(self._rule_prior_tensor(int(qr), int(layer_idx + 1)) is not None for qr in torch.unique(q_rel_edge).tolist()))
            edge_prior_logit = self._edge_prior_logit(q_rel_edge, layer=int(layer_idx + 1), rel_ids=rel_ids)
            prior_edge_nonzero += float((edge_prior_logit != 0).sum().item())

            hidden, nodes, sampled_nodes_mask = self.gnn_layers[layer_idx](
                q_rel=q_rel,
                hidden=hidden,
                edges=edges,
                nodes=nodes,
                old_nodes_new_idx=old_nodes_new_idx,
                batch_size=B,
                edge_prior_logit=edge_prior_logit,
            )

            h0 = torch.zeros((1, n_node, self.dim), device=device).index_copy_(1, old_nodes_new_idx, h0)
            h0 = h0[0, sampled_nodes_mask, :].unsqueeze(0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((B, n_ent), device=device)
        scores_all[nodes[:, 0], nodes[:, 1]] = scores
        visited_mask[nodes[:, 0], nodes[:, 1]] = True

        deltas = scores_all.gather(1, cand_t)
        mask_t = visited_mask.gather(1, cand_t)
        mask_h = torch.ones((B,), dtype=torch.bool, device=device)

        self.last_stats = {
            "visited_nodes": float(visited_mask.float().sum(dim=1).mean().item()),
            "sampled_edges": float(edges_total / max(B * max(int(self.layers), 1), 1)),
            "prior_edges": float(prior_edge_nonzero / max(B * max(int(self.layers), 1), 1)),
            "prior_nonempty": float(prior_nonempty / max(int(self.layers), 1)),
        }
        return deltas, mask_h, mask_t


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
                # evaluate_dataset() switches model/ reranker to eval(); restore training mode for backprop.
                reranker.train()

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
        iterator = dataloader
        if tqdm is not None:
            iterator = tqdm(dataloader, desc=f"DualRerank eval {split}", leave=False)
        ranks_base = []
        ranks_final = []
        improved = 0
        log_calls = 0
        for batch in iterator:
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
