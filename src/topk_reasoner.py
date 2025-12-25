import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from typing import Optional


class RuleContextEncoder(nn.Module):
    def __init__(self, num_rules: int, out_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_rules, out_dim)

    def forward(self, query_rel: torch.LongTensor, relation2rules) -> torch.Tensor:
        device = query_rel.device
        batch_size = query_rel.size(0)
        out = torch.zeros(batch_size, self.embed.embedding_dim, device=device)
        for i in range(batch_size):
            r = int(query_rel[i].item())
            if r < 0 or r >= len(relation2rules) or len(relation2rules[r]) == 0:
                continue
            rule_ids = [rule_id for rule_id, _ in relation2rules[r]]
            rule_ids = torch.as_tensor(rule_ids, dtype=torch.long, device=device)
            out[i] = self.embed(rule_ids).mean(0)
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

    def get_neighbors(self, nodes: torch.LongTensor, batch_size: int, forbidden_triples: Optional[torch.LongTensor] = None):
        nodes = nodes.to("cpu")
        forbidden_triples_cpu = None
        if forbidden_triples is not None:
            if forbidden_triples.dim() != 2 or forbidden_triples.size(1) != 3:
                raise ValueError("forbidden_triples must have shape [batch_size, 3] = (h, r, t)")
            forbidden_triples_cpu = forbidden_triples.to("cpu")
        edges = []
        for b, u in nodes.tolist():
            fh = fr = ft = None
            if forbidden_triples_cpu is not None and 0 <= b < forbidden_triples_cpu.size(0):
                fh, fr, ft = forbidden_triples_cpu[b].tolist()
            start = int(self.indptr[u].item())
            end = int(self.indptr[u + 1].item())
            if end > start:
                rel = self.rels[start:end].tolist()
                tail = self.tails[start:end].tolist()
                for r, v in zip(rel, tail):
                    if fh is not None and u == fh and r == fr and v == ft:
                        continue
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
    def __init__(self, hidden_dim: int, attn_dim: int, n_rel: int, n_ent: int, tau: float = 0.0, act="relu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_rel = n_rel
        self.n_ent = n_ent
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

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        if self.training and self.tau and self.tau > 0:
            self.softmax = lambda x: F.gumbel_softmax(x, tau=self.tau, hard=False)
        else:
            self.softmax = lambda x: F.softmax(x, dim=1)
        for module in self.children():
            module.train(mode)
        return self

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
        alpha = self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wq_attn(h_qr) + self.Wc_attn(h_c)))
        alpha = torch.sigmoid(alpha)

        message = alpha * message
        n_node = nodes.size(0)
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce="sum")
        hidden_new = self.act(self.W_h(message_agg))
        hidden_new = hidden_new.clone()

        if n_node_topk <= 0:
            return hidden_new, nodes, torch.ones(nodes.size(0), dtype=torch.bool, device=nodes.device)

        tmp_diff_node_idx = torch.ones(n_node, device=nodes.device)
        tmp_diff_node_idx[old_nodes_new_idx] = 0
        bool_diff_node_idx = tmp_diff_node_idx.bool()
        diff_node = nodes[bool_diff_node_idx]

        if diff_node.numel() == 0:
            keep_mask = ~bool_diff_node_idx
            return hidden_new[keep_mask], nodes[keep_mask], keep_mask

        diff_node_logit = self.W_samp(hidden_new[bool_diff_node_idx]).squeeze(-1)
        node_scores = torch.full((batch_size, self.n_ent), float("-inf"), device=nodes.device)
        node_scores[diff_node[:, 0], diff_node[:, 1]] = diff_node_logit

        all_inf = torch.isneginf(node_scores).all(dim=1)
        if all_inf.any():
            node_scores[all_inf, 0] = 0.0

        node_scores = self.softmax(node_scores)

        topk = min(n_node_topk, self.n_ent)
        topk_index = torch.topk(node_scores, topk, dim=1).indices.reshape(-1)
        topk_batchidx = torch.arange(batch_size, device=nodes.device).repeat(topk, 1).T.reshape(-1)
        batch_topk_nodes = torch.zeros((batch_size, self.n_ent), device=nodes.device)
        batch_topk_nodes[topk_batchidx, topk_index] = 1

        bool_sampled_diff_nodes_idx = batch_topk_nodes[diff_node[:, 0], diff_node[:, 1]].bool()
        bool_same_node_idx = ~bool_diff_node_idx
        bool_same_node_idx[bool_diff_node_idx] = bool_sampled_diff_nodes_idx

        diff_node_prob_hard = batch_topk_nodes[diff_node[:, 0], diff_node[:, 1]]
        diff_node_prob = node_scores[diff_node[:, 0], diff_node[:, 1]]
        hidden_new[bool_diff_node_idx] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob).unsqueeze(-1)

        new_nodes = nodes[bool_same_node_idx]
        new_hidden = hidden_new[bool_same_node_idx]
        return new_hidden, new_nodes, bool_same_node_idx


class TopKReasoner(nn.Module):
    def __init__(self, n_ent: int, n_rel: int, hidden_dim: int = 64, attn_dim: int = 8, n_layer: int = 5,
                 n_node_topk: int = 200, tau: float = 0.0, dropout: float = 0.1, act: str = "relu",
                 use_rule_semantic: bool = True, num_rules: Optional[int] = None):
        super().__init__()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.n_node_topk = n_node_topk
        self.use_rule_semantic = use_rule_semantic

        self.layers = nn.ModuleList(
            [TopKPropagationLayer(hidden_dim, attn_dim, n_rel, n_ent, tau=tau, act=act) for _ in range(n_layer)]
        )
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, 1, bias=False)

        if self.use_rule_semantic:
            if num_rules is None:
                raise ValueError("num_rules is required when use_rule_semantic=True")
            self.rule_ctx = RuleContextEncoder(num_rules, hidden_dim)
        else:
            self.rule_ctx = None

        self.use_kge = False
        self.kge_alpha = 1.0

    def set_kge_fusion(self, enabled: bool, alpha: float = 1.0):
        self.use_kge = bool(enabled)
        self.kge_alpha = float(alpha)

    def forward(
        self,
        subs,
        rels,
        sampler: IncrementalNeighborSampler,
        relation2rules=None,
        kge_score_fn=None,
        forbidden_tails: Optional[torch.LongTensor] = None,
    ):
        device = next(self.parameters()).device
        q_sub = torch.as_tensor(subs, dtype=torch.long, device=device)
        q_rel = torch.as_tensor(rels, dtype=torch.long, device=device)
        batch_size = q_sub.size(0)

        forbidden_triples = None
        if forbidden_tails is not None:
            forbidden_tails = torch.as_tensor(forbidden_tails, dtype=torch.long, device=device)
            if forbidden_tails.dim() != 1 or forbidden_tails.size(0) != batch_size:
                raise ValueError("forbidden_tails must have shape [batch_size]")
            forbidden_triples = torch.stack([q_sub, q_rel, forbidden_tails], dim=1)

        if self.use_rule_semantic and relation2rules is not None:
            rule_ctx = self.rule_ctx(q_rel, relation2rules)
        else:
            rule_ctx = torch.zeros(batch_size, self.hidden_dim, device=device)

        nodes = torch.stack([torch.arange(batch_size, device=device), q_sub], dim=1)
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)

        for layer in self.layers:
            nodes_full, edges, old_nodes_new_idx = sampler.get_neighbors(
                nodes, batch_size=batch_size, forbidden_triples=forbidden_triples
            )
            hidden, nodes, keep_mask = layer(
                q_rel=q_rel,
                rule_ctx=rule_ctx,
                hidden=hidden,
                edges=edges,
                nodes=nodes_full,
                old_nodes_new_idx=old_nodes_new_idx.to(nodes_full.device),
                batch_size=batch_size,
                n_node_topk=self.n_node_topk,
            )

            if keep_mask.dim() != 1 or keep_mask.size(0) != nodes_full.size(0):
                raise ValueError(
                    f"keep_mask shape {tuple(keep_mask.shape)} must match nodes_full shape {tuple(nodes_full.shape)}"
                )

            keep_mask = keep_mask.to(device=device)
            old_nodes_new_idx = old_nodes_new_idx.to(device=device)
            n_node_full = int(keep_mask.size(0))
            h0_aligned_full = torch.zeros(1, n_node_full, self.hidden_dim, device=device).index_copy_(1, old_nodes_new_idx, h0)
            h0_aligned = h0_aligned_full[:, keep_mask, :]
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
