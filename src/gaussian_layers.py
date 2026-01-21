"""
Gaussian-RulE: Gaussian Embedding Layers
高斯嵌入操作层

核心操作：
1. KLDivergence - KL散度计算
2. GaussianTransform - 高斯变换
3. ProductOfGaussians - 高斯乘积（规则融合）
4. KLRegularization - KL正则化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergence(nn.Module):
    """
    计算两个高斯分布的 KL 散度

    D_KL(P || Q) = 0.5 * [
        Σ(σ²_p / σ²_q) +
        Σ((μ_q - μ_p)² / σ²_q) +
        Σ(log(σ²_q / σ²_p)) - d
    ]

    处处可微！即使两个分布相距很远，梯度也存在
    """
    def __init__(self, epsilon=1e-8):
        super(KLDivergence, self).__init__()
        self.epsilon = epsilon

    def forward(self, p, q):
        """
        Args:
            p: (mu_p, logvar_p)
                - mu_p: [..., d] 均值
                - logvar_p: [..., d] log方差
            q: (mu_q, logvar_q)

        Returns:
            kl: [...] KL散度（标量）
        """
        mu_p, logvar_p = p
        mu_q, logvar_q = q

        # 转换为方差
        var_p = torch.exp(logvar_p)
        var_q = torch.exp(logvar_q)

        # KL散度的四项（文档公式）
        term1 = var_p / (var_q + self.epsilon)  # 宽窄比
        term2 = (mu_p - mu_q) ** 2 / (var_q + self.epsilon)  # 中心距离（关键！）
        term3 = logvar_q - logvar_p  # log比
        term4 = -mu_p.size(-1)  # -d

        # 求和
        kl = 0.5 * (term1.sum(-1) + term2.sum(-1) + term3.sum(-1) + term4)

        return kl

    def forward_broadcast(self, p, q_all):
        """
        广播版本：计算一个分布与所有分布的KL散度

        Args:
            p: (mu_p, logvar_p)
                - mu_p: [batch, d]
                - logvar_p: [batch, d]
            q_all: (mu_q_all, logvar_q_all)
                - mu_q_all: [N, d] 所有实体的分布
                - logvar_q_all: [N, d]

        Returns:
            kl: [batch, N] 每个batch元素与所有实体的KL散度
        """
        mu_p, logvar_p = p
        mu_q_all, logvar_q_all = q_all

        # 扩维以便广播
        mu_p = mu_p.unsqueeze(1)  # [batch, 1, d]
        logvar_p = logvar_p.unsqueeze(1)  # [batch, 1, d]

        mu_q_all = mu_q_all.unsqueeze(0)  # [1, N, d]
        logvar_q_all = logvar_q_all.unsqueeze(0)  # [1, N, d]

        var_p = torch.exp(logvar_p)
        var_q_all = torch.exp(logvar_q_all)

        # KL散度计算（广播）
        term1 = var_p / (var_q_all + self.epsilon)
        term2 = (mu_p - mu_q_all) ** 2 / (var_q_all + self.epsilon)
        term3 = logvar_q_all - logvar_p
        term4 = -mu_p.size(-1)

        kl = 0.5 * (term1.sum(-1) + term2.sum(-1) + term3.sum(-1) + term4)
        # [batch, N]

        return kl


class GaussianTransform(nn.Module):
    """
    通过关系变换高斯分布

    变换规则（文档公式）：
        μ' = μ + μ_r
        log(σ'²) = log(σ² + σ_r²) = logaddexp(log_σ², log_σ_r²)
    """
    def __init__(self):
        super(GaussianTransform, self).__init__()

    def forward(self, gaussian, relation_mean, relation_logvar):
        """
        Args:
            gaussian: (mu, logvar)
            relation_mean: [..., d] 关系均值
            relation_logvar: [..., d] 关系log方差

        Returns:
            gaussian_out: (mu_out, logvar_out)
        """
        mu, logvar = gaussian

        # 均值相加（平移）
        mu_out = mu + relation_mean

        # 方差累加（在log域用logaddexp）
        logvar_out = torch.logaddexp(logvar, relation_logvar)

        return (mu_out, logvar_out)


class ProductOfGaussians(nn.Module):
    """
    多个高斯分布的乘积（交集）

    文档中的解析解：
        Σ_inter = (Σ_1^-1 + Σ_2^-1)^-1
        μ_inter = Σ_inter (Σ_1^-1 μ_1 + Σ_2^-1 μ_2)

    妙处：自动加权平均！方差小的（确定性高的）权重大
    """
    def __init__(self, epsilon=1e-8):
        super(ProductOfGaussians, self).__init__()
        self.epsilon = epsilon

    def forward(self, gaussians):
        """
        Args:
            gaussians: [(mu_1, logvar_1), (mu_2, logvar_2), ...]

        Returns:
            gaussian_inter: (mu_inter, logvar_inter)
        """
        if len(gaussians) == 0:
            raise ValueError("Cannot compute product of empty gaussian list")

        if len(gaussians) == 1:
            return gaussians[0]

        # 堆叠所有高斯分布
        means = torch.stack([g[0] for g in gaussians])  # [num_gaussians, d]
        logvars = torch.stack([g[1] for g in gaussians])  # [num_gaussians, d]

        vars = torch.exp(logvars)
        precisions = 1.0 / (vars + self.epsilon)  # Σ^-1

        # 新精度（文档公式）
        precision_inter = precisions.sum(dim=0)  # [d]
        var_inter = 1.0 / (precision_inter + self.epsilon)

        # 新均值（加权平均，权重由精度决定）
        weighted_means = (means * precisions).sum(dim=0)  # [d]
        mu_inter = var_inter * weighted_means

        logvar_inter = torch.log(var_inter + self.epsilon)

        return (mu_inter, logvar_inter)


class KLRegularization(nn.Module):
    """
    KL正则化 - 防止方差退化

    目标：保持方差在合理范围
    Loss = λ * |log(σ²) - target|

    避免：
    - 方差太小 → 分布退化为点（过拟合）
    - 方差太大 → 分布过于模糊（欠拟合）
    """
    def __init__(self, lambda_kl=0.001, target_logvar=0.0):
        super(KLRegularization, self).__init__()
        self.lambda_kl = lambda_kl
        self.target_logvar = target_logvar

    def forward(self, logvars):
        """
        Args:
            logvars: [num_entities, d] 所有实体的log方差

        Returns:
            loss: 标量正则化损失
        """
        # 计算平均log方差（每个实体在所有维度的平均）
        mean_logvar = logvars.mean(-1)  # [num_entities]

        # 与目标的偏差
        deviation = mean_logvar - self.target_logvar

        # L1损失
        loss = self.lambda_kl * torch.abs(deviation).mean()

        return loss
