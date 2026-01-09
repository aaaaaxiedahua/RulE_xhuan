from typing import List, Tuple

import torch
import torch.nn as nn

from .complex_ops import complex_unit_from_phase
from .complex_ops import complex_mul


class HoGCNRelationEncoder(nn.Module):
    """
    A lightweight, HoGCN/HoGRN-inspired (weight-free) relation encoder.

    Given a query (h, r_target, ?), it aggregates outgoing relation "rotations"
    around h with an attention conditioned on r_target, producing a complex vector v_ctx.

    - Evidence vector v_ctx is *not* per-dimension normalized: its modulus encodes certainty
      via cancellation (shrinkage) when neighborhood relations conflict.
    """

    def __init__(
        self,
        graph,
        num_relations: int,
        num_hops: int = 2,
        max_paths_per_hop: int = 5000,
        max_edges_per_node: int = 50,
        limit_paths: bool = True,
    ):
        super().__init__()
        self.num_entities = graph.entity_size
        self.num_relations = num_relations
        if num_hops < 1:
            raise ValueError(f"num_hops must be >= 1, got {num_hops}")
        self.num_hops = int(num_hops)
        self.max_paths_per_hop = int(max_paths_per_hop)
        self.max_edges_per_node = int(max_edges_per_node)
        self.limit_paths = bool(limit_paths)
        self._out_edges: List[Tuple[torch.Tensor, torch.Tensor]] = self._build_out_edge_index(graph)

    @staticmethod
    def _build_out_edge_index(graph) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        out_dst: List[List[int]] = [[] for _ in range(graph.entity_size)]
        out_rel: List[List[int]] = [[] for _ in range(graph.entity_size)]
        for h, r, _t in graph.ground_train_facts:
            out_dst[h].append(int(_t))
            out_rel[h].append(int(r))

        out = []
        for h in range(graph.entity_size):
            dst = torch.tensor(out_dst[h], dtype=torch.long)
            rel = torch.tensor(out_rel[h], dtype=torch.long)
            out.append((dst, rel))
        return out

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
            dst1, rel1 = self._out_edges[h]
            rel1 = rel1.to(device)
            dst1 = dst1.to(device)
            if rel1.numel() == 0:
                v_all.append(torch.zeros_like(rel_target_rot[i]))
                continue

            # Initialize hop-1 path operators as outgoing edge relations.
            rel1_phase = self.rel_phase_full(rel_phase, rel1, self.num_relations)  # (E, d)
            ops = complex_unit_from_phase(rel1_phase)  # (E, d, 2)
            nodes = dst1  # (E,)

            hop_vectors = []
            for hop in range(1, self.num_hops + 1):
                # Aggregate evidence from current hop operators.
                s = (ops[..., 0] * rel_target_rot[i, :, 0] + ops[..., 1] * rel_target_rot[i, :, 1]).sum(dim=-1)
                s = s / max(float(tau_edge), 1e-6)
                a = torch.softmax(s, dim=0)
                hop_vectors.append((a.view(-1, 1, 1) * ops).sum(dim=0))

                if hop == self.num_hops:
                    break

                # Expand to next hop: compose current path-operator with outgoing edge relations.
                composed_ops = []
                composed_nodes = []

                # Downsample current paths before expansion if too many.
                if self.limit_paths and nodes.numel() > self.max_paths_per_hop:
                    perm = torch.randperm(nodes.numel(), device=device)[: self.max_paths_per_hop]
                    nodes = nodes.index_select(0, perm)
                    ops = ops.index_select(0, perm)

                for j in range(nodes.numel()):
                    mid = int(nodes[j].item())
                    dst2, rel2 = self._out_edges[mid]
                    rel2 = rel2.to(device)
                    dst2 = dst2.to(device)
                    if rel2.numel() == 0:
                        continue

                    # Cap fanout per node to avoid explosion.
                    if self.limit_paths and rel2.numel() > self.max_edges_per_node:
                        perm2 = torch.randperm(rel2.numel(), device=device)[: self.max_edges_per_node]
                        rel2 = rel2.index_select(0, perm2)
                        dst2 = dst2.index_select(0, perm2)

                    rel2_phase = self.rel_phase_full(rel_phase, rel2, self.num_relations)  # (E2, d)
                    rel2_rot = complex_unit_from_phase(rel2_phase)  # (E2, d, 2)

                    composed_ops.append(complex_mul(ops[j].unsqueeze(0), rel2_rot))  # (E2, d, 2)
                    composed_nodes.append(dst2)

                    if self.limit_paths and sum(int(x.size(0)) for x in composed_nodes) >= self.max_paths_per_hop:
                        break

                if not composed_ops:
                    # No further expansion possible.
                    break

                ops = torch.cat(composed_ops, dim=0)
                nodes = torch.cat(composed_nodes, dim=0)

                # Hard cap after expansion.
                if self.limit_paths and nodes.numel() > self.max_paths_per_hop:
                    perm3 = torch.randperm(nodes.numel(), device=device)[: self.max_paths_per_hop]
                    nodes = nodes.index_select(0, perm3)
                    ops = ops.index_select(0, perm3)

            v_all.append(torch.stack(hop_vectors, dim=0).mean(dim=0))

        return torch.stack(v_all, dim=0)
