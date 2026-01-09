from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Rule:
    head_rel: int
    body_rels: torch.Tensor  # (L,)

