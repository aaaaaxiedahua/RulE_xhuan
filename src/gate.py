import torch
from torch import nn


class CalibrationGate(nn.Module):
    def __init__(
        self,
        hidden_dim,
        use_stats=True,
        mlp_hidden_dim=256,
        dropout=0.1,
        alpha_min=0.0,
        alpha_max=3.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_stats = bool(use_stats)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

        input_dim = self.hidden_dim * 3
        if self.use_stats:
            input_dim += 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, int(mlp_hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_hidden_dim), 1),
        )

    def forward(self, h_emb, r_emb, margin_ratio=None, entropy=None):
        pieces = [h_emb, r_emb]
        if self.use_stats:
            if margin_ratio is None or entropy is None:
                raise ValueError("margin_ratio and entropy must be provided when use_stats=True")
            if margin_ratio.dim() == 1:
                margin_ratio = margin_ratio.unsqueeze(-1)
            if entropy.dim() == 1:
                entropy = entropy.unsqueeze(-1)
            pieces.extend([margin_ratio, entropy])

        x = torch.cat(pieces, dim=-1)
        z = self.net(x).squeeze(-1)
        alpha_unit = torch.sigmoid(z)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_unit
        return alpha

    def extra_repr(self):
        return (
            f"hidden_dim={self.hidden_dim}, use_stats={self.use_stats}, "
            f"alpha_min={self.alpha_min}, alpha_max={self.alpha_max}"
        )
