import torch
from torch import nn


class CalibrationGate(nn.Module):
    def __init__(
        self,
        feature_dim=6,
        hidden_dim=32,
        dropout=0.3,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, features):
        """
        Args:
            features: [batch, feature_dim]

        Returns:
            lambda_mix: [batch] in (0, 1)
        """
        if features.dim() != 2 or features.size(-1) != self.feature_dim:
            raise ValueError(
                f"features must be [batch, {self.feature_dim}], got {tuple(features.size())}"
            )
        z = self.net(features).squeeze(-1)
        return torch.sigmoid(z)

    def extra_repr(self):
        return f"feature_dim={self.feature_dim}, hidden_dim={self.hidden_dim}"
