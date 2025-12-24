import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from typing import Optional


class RuleContextEncoder(nn.Module):
    def __init__(self, rule_vec_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(rule_vec_dim, out_dim, bias=False)

    def forward(self, query_rel: torch.LongTensor, relation2rules, rules_weight_emb: torch.Tensor) -> torch.Tensor:
        device = query_rel.device
        batch_size = query_rel.size(0)
        out = torch.zeros(batch_size, self.proj.out_features, device=device)
        for i in range(batch_size):
            r = int(query_rel[i].item())
            if r < 0 or r >= len(relation2rules) or len(relation2rules[r]) == 0:
                continue
            rule_indices = [idx for idx, _ in relation2rules[r]]
            rule_indices = torch.as_tensor(rule_indices, dtype=torch.long, device=device)
            vec = rules_weight_emb[rule_indices]
            out[i] = self.proj(vec).mean(0)
        return out


class IncrementalNeighborSampler:
    def __init__(self, triples, n_ent: int, n_rel: int, device: torch.device):
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.self_loop_rel = 2 * n_rel
        self.device = device

        heads = torch.as_tensor([h for h, _, _ in triples], dtype=torch.long)
        rels = torch.as_tensor([r for _, r, _ in triples], dtype=torch.long)
        tails = torch.as_tensor([t for _, _, t in triples], dtype=torch.long)

        order = torch.argsort(heads)
        self.heads = heads[order]
        self.rels = rels[order]
        self.tails = tails[order]

        counts = torch.bincount(self.heads, minlength=n_ent)
        indptr = torch.zeros(n_ent + 1, dtype=torch.long)
        indptr[1:] = torch.cumsum(counts, dim=0)
        self.indptr = indptr

    def get_neighbors(self, nodes: torch.LongTensor, batch_size: int):
        nodes = nodes.to("cpu")
        edges = []
        for b, u in nodes.tolist():
            start = int(self.indptr[u].item())
            end = int(self.indptr[u + 1].item())
            if end > start:
                rel = self.rels[start:end].tolist()
                tail = self.tails[start:end].tolist()
                for r, v in zip(rel, tail):
                    edges.append((b, u, r, v))
            edges.append((b, u, self.self_loop_rel, u))

        edges = torch.as_tensor(edges, dtype=torch.long, device=self.device)
        head_nodes, head_index = torch.unique(edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
        edges = torch.cat([edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], dim=1)

        mask = edges[:, 2] == self.self_loop_rel
        old_nodes_new_idx = tail_index[mask].sort()[0]
        return tail_nodes, edges, old_nodes_new_idx


class TopKPropagationLayer(nn.Module):
    def __init__(self, hidden_dim: int, attn_dim: int, n_rel: int, tau: float = 0.0, act="relu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_rel = n_rel
        self.tau = tau
        self.self_loop_rel = 2 * n_rel

        acts = {"relu": nn.ReLU(), "tanh": torch.tanh, "idd": lambda x: x}
        self.act = acts.get(act, nn.ReLU())

        self.rela_embed = nn.Embedding(2 * n_rel + 1, hidden_dim)
        self.Ws_attn = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.Wq_attn = nn.Linear(hidden_dim, attn_dim)
        self.Wc_attn = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_samp = nn.Linear(hidden_dim, 1, bias=False)

    def _per_batch_topk(self, batch_idx: torch.LongTensor, logits: torch.Tensor, k: int, batch_size: int):
        selected = torch.zeros_like(logits, dtype=torch.bool)
        for b in range(batch_size):
            mask = batch_idx == b
            if mask.sum().item() == 0:
                continue
            b_logits = logits[mask]
            kk = min(k, b_logits.numel())
            topk_local = torch.topk(b_logits, kk, dim=0).indices
            global_idx = torch.nonzero(mask, as_tuple=True)[0][topk_local]
            selected[global_idx] = True
        return selected

    def forward(self, q_rel: torch.LongTensor, rule_ctx: torch.Tensor, hidden: torch.Tensor, edges: torch.LongTensor,
                nodes: torch.LongTensor, old_nodes_new_idx: torch.LongTensor, batch_size: int, n_node_topk: int):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        r_idx = edges[:, 0]

        hs = hidden[sub]
        hr = self.rela_embed(rel)
        h_qr = self.rela_embed(q_rel)[r_idx]
        h_c = rule_ctx[r_idx]

        message = hs + hr
        alpha = self.w_alpha(torch.relu(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wq_attn(h_qr) + self.Wc_attn(h_c))).squeeze(-1)
        alpha = torch.sigmoid(alpha).unsqueeze(-1)

        message = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=nodes.size(0), reduce="sum")
        hidden_new = self.act(self.W_h(message_agg))
        hidden_new = hidden_new.clone()

        if n_node_topk <= 0:
            return hidden_new, nodes, torch.ones(nodes.size(0), dtype=torch.bool, device=nodes.device)

        tmp_diff = torch.ones(nodes.size(0), dtype=torch.bool, device=nodes.device)
        tmp_diff[old_nodes_new_idx] = False
        diff_mask = tmp_diff
        same_mask = ~diff_mask
        diff_nodes = nodes[diff_mask]

        if diff_nodes.size(0) == 0:
            return hidden_new[same_mask], nodes[same_mask], same_mask

        diff_logits = self.W_samp(hidden_new[diff_mask]).squeeze(-1)
        if self.training and self.tau and self.tau > 0:
            g = -torch.log(-torch.log(torch.rand_like(diff_logits).clamp_min_(1e-12)).clamp_min_(1e-12))
            diff_logits_g = (diff_logits + g) / self.tau
            diff_prob = torch.zeros_like(diff_logits_g)
            for b in range(batch_size):
                m = diff_nodes[:, 0] == b
                if m.sum().item() == 0:
                    continue
                diff_prob[m] = F.softmax(diff_logits_g[m], dim=0)
        else:
            diff_prob = None

        selected_diff = self._per_batch_topk(diff_nodes[:, 0], diff_logits, n_node_topk, batch_size)

        if diff_prob is not None:
            hidden_new[diff_mask] = hidden_new[diff_mask] * (selected_diff.float() - diff_prob.detach() + diff_prob).unsqueeze(-1)

        same_mask[diff_mask] = selected_diff
        new_nodes = nodes[same_mask]
        new_hidden = hidden_new[same_mask]
        return new_hidden, new_nodes, same_mask


class TopKReasoner(nn.Module):
    def __init__(self, n_ent: int, n_rel: int, hidden_dim: int = 64, attn_dim: int = 8, n_layer: int = 5,
                 n_node_topk: int = 200, tau: float = 0.0, dropout: float = 0.1, act: str = "relu",
                 use_rule_semantic: bool = True, rule_vec_dim: Optional[int] = None):
        super().__init__()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.n_node_topk = n_node_topk
        self.use_rule_semantic = use_rule_semantic

        self.layers = nn.ModuleList([
            TopKPropagationLayer(hidden_dim, attn_dim, n_rel, tau=tau, act=act) for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, 1, bias=False)

        if self.use_rule_semantic:
            if rule_vec_dim is None:
                raise ValueError("rule_vec_dim is required when use_rule_semantic=True")
            self.rule_ctx = RuleContextEncoder(rule_vec_dim, hidden_dim)
        else:
            self.rule_ctx = None

        self.use_kge = False
        self.kge_alpha = 1.0

    def set_kge_fusion(self, enabled: bool, alpha: float = 1.0):
        self.use_kge = bool(enabled)
        self.kge_alpha = float(alpha)

    def forward(self, subs, rels, sampler: IncrementalNeighborSampler, relation2rules=None, rules_weight_emb=None,
                kge_score_fn=None):
        device = next(self.parameters()).device
        q_sub = torch.as_tensor(subs, dtype=torch.long, device=device)
        q_rel = torch.as_tensor(rels, dtype=torch.long, device=device)
        batch_size = q_sub.size(0)

        if self.use_rule_semantic and relation2rules is not None and rules_weight_emb is not None:
            rule_ctx = self.rule_ctx(q_rel, relation2rules, rules_weight_emb.to(device))
        else:
            rule_ctx = torch.zeros(batch_size, self.hidden_dim, device=device)

        nodes = torch.stack([torch.arange(batch_size, device=device), q_sub], dim=1)
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)

        for layer in self.layers:
            nodes, edges, old_nodes_new_idx = sampler.get_neighbors(nodes, batch_size=batch_size)
            hidden, nodes, keep_mask = layer(
                q_rel=q_rel,
                rule_ctx=rule_ctx,
                hidden=hidden,
                edges=edges,
                nodes=nodes,
                old_nodes_new_idx=old_nodes_new_idx.to(nodes.device),
                batch_size=batch_size,
                n_node_topk=self.n_node_topk,
            )

            n_node = nodes.size(0)
            h0_aligned = torch.zeros(1, n_node, self.hidden_dim, device=device).index_copy_(1, old_nodes_new_idx.to(device), h0)
            h0_aligned = h0_aligned[:, keep_mask, :]
            hidden = self.dropout(hidden)
            hidden, h0 = self.gru(hidden.unsqueeze(0), h0_aligned)
            hidden = hidden.squeeze(0)

        scores = self.readout(hidden).squeeze(-1)
        scores_all = torch.zeros((batch_size, self.n_ent), device=device)
        scores_all[nodes[:, 0], nodes[:, 1]] = scores

        if self.use_kge and kge_score_fn is not None:
            kge_scores = kge_score_fn(q_sub, q_rel, nodes)
            scores_all[nodes[:, 0], nodes[:, 1]] = scores_all[nodes[:, 0], nodes[:, 1]] + self.kge_alpha * kge_scores

        return scores_all
