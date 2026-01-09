from typing import List

import torch
import torch.nn as nn

from .complex_ops import complex_unit_from_phase


class HoGCNRelationEncoder(nn.Module):
    """
    A lightweight, HoGCN/HoGRN-inspired (weight-free) relation encoder.

    Given a query (h, r_target, ?), it aggregates outgoing relation "rotations"
    around h with an attention conditioned on r_target, producing a complex vector v_ctx.

    - Evidence vector v_ctx is *not* per-dimension normalized: its modulus encodes certainty
      via cancellation (shrinkage) when neighborhood relations conflict.
    """

    def __init__(self, graph, num_relations: int):
        super().__init__()
        self.num_entities = graph.entity_size
        self.num_relations = num_relations
        self._out_rels: List[torch.Tensor] = self._build_out_rel_index(graph)

    @staticmethod
    def _build_out_rel_index(graph) -> List[torch.Tensor]:
        out_lists: List[List[int]] = [[] for _ in range(graph.entity_size)]
        for h, r, _t in graph.ground_train_facts:
            out_lists[h].append(r)
        return [torch.tensor(v, dtype=torch.long) for v in out_lists]

    @staticmethod
    def rel_phase_full(rel_phase: torch.Tensor, rel_ids: torch.Tensor, num_relations: int) -> torch.Tensor:
        """
        Map relation ids in [0, 2*num_relations) to signed phases in R^d.
        rel_phase: (num_relations, d)
        rel_ids: (...,)
        returns: (..., d)
        """
        base = rel_ids % num_relations
        sign = torch.where(rel_ids < num_relations, 1.0, -1.0).to(rel_phase.dtype)
        return rel_phase.index_select(0, base) * sign.unsqueeze(-1)

    def forward(
        self,
        heads: torch.Tensor,  # (B,)
        rel_target: torch.Tensor,  # (B,)
        rel_phase: torch.Tensor,  # (num_rel, d)
        tau_edge: float = 1.0,
    ) -> torch.Tensor:
        """
        Returns v_ctx in complex form: (B, d, 2).
        """
        device = heads.device
        rel_target_phase = self.rel_phase_full(rel_phase, rel_target, self.num_relations)
        rel_target_rot = complex_unit_from_phase(rel_target_phase)  # (B, d, 2)

        v_all = []
        for i in range(heads.size(0)):
            h = int(heads[i].item())
            rels = self._out_rels[h].to(device)
            if rels.numel() == 0:
                v_all.append(torch.zeros_like(rel_target_rot[i]))
                continue

            rels_phase = self.rel_phase_full(rel_phase, rels, self.num_relations)  # (E, d)
            rels_rot = complex_unit_from_phase(rels_phase)  # (E, d, 2)

            # Similarity of rotations: sum_d cos(Δθ_d) = <u_e, u_q> per dim summed over dims.
            s = (rels_rot[..., 0] * rel_target_rot[i, :, 0] + rels_rot[..., 1] * rel_target_rot[i, :, 1]).sum(dim=-1)
            s = s / max(float(tau_edge), 1e-6)
            a = torch.softmax(s, dim=0)  # (E,)
            v = (a.view(-1, 1, 1) * rels_rot).sum(dim=0)  # (d, 2)
            v_all.append(v)

        return torch.stack(v_all, dim=0)

