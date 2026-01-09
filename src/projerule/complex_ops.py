import torch


def as_complex2(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Convert a flat (..., 2*dim) tensor into (..., dim, 2) where the last axis is (re, im).
    """
    return x.view(*x.shape[:-1], dim, 2)


def complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication for (..., dim, 2) tensors.
    """
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]
    re = ar * br - ai * bi
    im = ar * bi + ai * br
    return torch.stack([re, im], dim=-1)


def complex_unit_from_phase(phase: torch.Tensor) -> torch.Tensor:
    """
    Convert a phase tensor (..., dim) to a unit complex vector (..., dim, 2) = (cos, sin).
    """
    return torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)


def complex_dist_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    L2 distance between complex vectors a and b.
    a: (..., dim, 2)
    b: (..., dim, 2)
    Returns: (...,) as sum over dim of per-dimension complex magnitude.
    """
    diff = a - b
    per_dim = torch.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2 + 1e-12)
    return per_dim.sum(dim=-1)


def complex_normalize_per_dim(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize each complex dimension to unit modulus.
    z: (..., dim, 2)
    """
    mod = torch.sqrt(z[..., 0] ** 2 + z[..., 1] ** 2 + eps)
    return z / mod.unsqueeze(-1)


def complex_modulus_mean(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Mean modulus across dimensions.
    z: (..., dim, 2)
    returns: (...,)
    """
    mod = torch.sqrt(z[..., 0] ** 2 + z[..., 1] ** 2 + eps)
    return mod.mean(dim=-1)

