import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from .complex_ops import (
    as_complex2,
    complex_dist_l2,
    complex_modulus_mean,
    complex_mul,
    complex_normalize_per_dim,
    complex_unit_from_phase,
)
from .encoder import HoGCNRelationEncoder
from .rules import Rule


class ProjeRulE(nn.Module):
    """
    ProjeRulE (RotatE version): Hypothesis–Verification–Gating.

    Base score (data):       S_Base = -|| h ∘ r_target - t ||
    Evidence (encoder):      v_ctx  = Encoder(h, G_h, r_target) in complex (not unit-normalized)
    Rule resonance:          w_k = Softmax( Re(<v_ctx, T_rho_k>) / tau_rule )
    Rule aggregation:        T_agg = Σ_k w_k T_rho_k
    Decoupling:
        alpha = sigmoid(mean_modulus(T_agg) * beta1 + beta2)
        T_hat = per-dimension normalize(T_agg)
    Adaptive fusion:
        Score = S_Base + lambda_rule * [ alpha*S_Logic + (1-alpha)*S_Implicit ] + bias_t
    """

    def __init__(
        self,
        graph,
        dim: int = 200,
        lambda_rule: float = 1.0,
        tau_rule: float = 1.0,
        tau_edge: float = 1.0,
        encoder_hops: int = 2,
        encoder_max_2hop_paths: int = 5000,
        encoder_max_edges_per_node: int = 50,
        encoder_limit_paths: bool = True,
        beta1: float = 10.0,
        beta2: float = -5.0,
        init_phase_scale: float = math.pi,
    ):
        super().__init__()
        self.graph = graph
        self.num_entities = graph.entity_size
        self.num_relations = graph.relation_size
        self.dim = dim

        self.lambda_rule = float(lambda_rule)
        self.tau_rule = float(tau_rule)
        self.tau_edge = float(tau_edge)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)

        self.entity_embedding = nn.Embedding(self.num_entities, self.dim * 2)
        self.relation_phase = nn.Embedding(self.num_relations, self.dim)
        nn.init.uniform_(self.entity_embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.relation_phase.weight, a=-init_phase_scale, b=init_phase_scale)

        self.bias = nn.Parameter(torch.zeros(self.num_entities))

        self.encoder = HoGCNRelationEncoder(
            graph,
            num_relations=self.num_relations,
            num_hops=int(encoder_hops),
            max_paths_per_hop=int(encoder_max_2hop_paths),
            max_edges_per_node=int(encoder_max_edges_per_node),
            limit_paths=bool(encoder_limit_paths),
        )
        self.rules_by_head: Dict[int, List[Rule]] = defaultdict(list)

    def set_rules(self, rules: Sequence[Sequence[int]]) -> None:
        """
        Each rule should be [rule_id, head_rel, body_rel_1, body_rel_2, ...].
        """
        self.rules_by_head.clear()
        for rule in rules:
            if len(rule) < 3:
                continue
            head = int(rule[1])
            body = torch.tensor(list(rule[2:]), dtype=torch.long)
            self.rules_by_head[head].append(Rule(head_rel=head, body_rels=body))

    def _rel_phase_full(self, rel_ids: torch.Tensor) -> torch.Tensor:
        base = rel_ids % self.num_relations
        sign = torch.where(rel_ids < self.num_relations, 1.0, -1.0).to(self.relation_phase.weight.dtype)
        return self.relation_phase(base) * sign.unsqueeze(-1)

    def _rel_rot_full(self, rel_ids: torch.Tensor) -> torch.Tensor:
        return complex_unit_from_phase(self._rel_phase_full(rel_ids))

    def _entities_complex(self, entity_ids: torch.Tensor) -> torch.Tensor:
        return as_complex2(self.entity_embedding(entity_ids), self.dim)

    def _compile_rule_ops(self, head_rel: int, device: torch.device) -> torch.Tensor:
        rules = self.rules_by_head.get(int(head_rel), [])
        if not rules:
            return torch.empty(0, self.dim, 2, device=device)

        body_lists = [r.body_rels.to(device) for r in rules]
        max_len = max(int(b.numel()) for b in body_lists)
        padded = []
        mask = []
        for b in body_lists:
            if b.numel() < max_len:
                pad = torch.full((max_len - b.numel(),), 0, dtype=torch.long, device=device)
                b2 = torch.cat([b, pad], dim=0)
                m2 = torch.cat([torch.ones_like(b, dtype=torch.bool), torch.zeros_like(pad, dtype=torch.bool)], dim=0)
            else:
                b2 = b
                m2 = torch.ones_like(b, dtype=torch.bool)
            padded.append(b2)
            mask.append(m2)
        body = torch.stack(padded, dim=0)  # (K, L)
        body_mask = torch.stack(mask, dim=0)  # (K, L)

        body_phase = self._rel_phase_full(body.reshape(-1)).view(body.size(0), body.size(1), self.dim)
        body_phase = body_phase * body_mask.unsqueeze(-1).to(body_phase.dtype)
        rule_phase = body_phase.sum(dim=1)  # (K, dim)
        return complex_unit_from_phase(rule_phase)  # (K, dim, 2)

    def _rule_resonance(self, v_ctx: torch.Tensor, rel_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = v_ctx.device
        B = v_ctx.size(0)

        alpha = torch.zeros(B, device=device)
        T_hat = torch.zeros(B, self.dim, 2, device=device)
        T_hat[..., 0] = 1.0  # identity rotation

        for rel_val in rel_target.unique():
            rel_int = int(rel_val.item())
            idx = torch.nonzero(rel_target == rel_val, as_tuple=True)[0]
            ops = self._compile_rule_ops(rel_int, device=device)  # (K, dim, 2)
            if ops.numel() == 0:
                alpha[idx] = 0.0
                continue

            v = v_ctx.index_select(0, idx)  # (b, dim, 2)
            s = (v[:, None, :, 0] * ops[None, :, :, 0] + v[:, None, :, 1] * ops[None, :, :, 1]).mean(dim=-1)  # (b, K)
            w = torch.softmax(s / max(self.tau_rule, 1e-6), dim=1)  # (b, K)

            T_agg = torch.einsum("bk,kdc->bdc", w, ops)  # (b, dim, 2)
            c = complex_modulus_mean(T_agg)  # (b,)
            a_val = torch.sigmoid(c * self.beta1 + self.beta2)  # (b,)

            alpha[idx] = a_val
            T_hat[idx] = complex_normalize_per_dim(T_agg)

        return alpha, T_hat

    def score_all_tails(
        self,
        heads: torch.Tensor,  # (B,)
        rel_target: torch.Tensor,  # (B,)
        chunk_size: int = 4096,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = heads.device

        h = self._entities_complex(heads)  # (B, dim, 2)
        r_base = self._rel_rot_full(rel_target)  # (B, dim, 2)

        v_ctx = self.encoder(heads, rel_target, self.relation_phase.weight, tau_edge=self.tau_edge)  # (B, dim, 2)
        alpha, T_hat = self._rule_resonance(v_ctx, rel_target)  # (B,), (B, dim, 2)
        alpha = alpha.view(-1, 1)  # (B, 1)

        h_base = complex_mul(h, r_base)
        h_logic = complex_mul(h, T_hat)
        h_imp = complex_mul(h, v_ctx)  # shrinkage allowed

        all_t = as_complex2(self.entity_embedding.weight.to(device), self.dim)  # (N, dim, 2)
        scores_parts = []
        for start in range(0, self.num_entities, chunk_size):
            end = min(start + chunk_size, self.num_entities)
            t = all_t[start:end]  # (C, dim, 2)

            s_base = -complex_dist_l2(h_base.unsqueeze(1), t.unsqueeze(0))  # (B, C)
            s_logic = -complex_dist_l2(h_logic.unsqueeze(1), t.unsqueeze(0))  # (B, C)
            s_imp = -complex_dist_l2(h_imp.unsqueeze(1), t.unsqueeze(0))  # (B, C)

            score = s_base + self.lambda_rule * (alpha * s_logic + (1.0 - alpha) * s_imp)
            score = score + self.bias[start:end].view(1, -1)
            scores_parts.append(score)

        scores = torch.cat(scores_parts, dim=1)
        return scores, {"alpha": alpha.squeeze(1), "v_ctx": v_ctx}
