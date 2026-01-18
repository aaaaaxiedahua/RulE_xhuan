"""
Box-RulE: Box Embedding Layers
盒嵌入相关的操作层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VolumeRegularization(nn.Module):
    """
    体积正则化模块 - 防止盒子坍塌

    核心思想：在Log域操作，保持盒子宽度在合理范围
    """
    def __init__(self, lambda_vol=0.001, target_log_vol=0.0, epsilon=1e-8):
        super(VolumeRegularization, self).__init__()
        self.lambda_vol = lambda_vol
        self.target_log_vol = target_log_vol
        self.epsilon = epsilon

    def forward(self, widths):
        """
        Args:
            widths: (batch_size, hidden_dim) 或 (num_entities, hidden_dim)
        Returns:
            loss: 标量，体积正则化损失
        """
        # 计算log体积
        log_widths = torch.log(widths + self.epsilon)
        log_vol = torch.sum(log_widths, dim=-1)  # (batch_size,)

        # 计算与目标的偏差
        deviation = log_vol - self.target_log_vol

        # L1损失
        loss = self.lambda_vol * torch.abs(deviation).mean()

        return loss


class BoxTransform(nn.Module):
    """
    盒子变换模块

    通过关系对盒子进行平移和缩放变换
    """
    def __init__(self):
        super(BoxTransform, self).__init__()

    def forward(self, box, relation_trans, relation_scale):
        """
        Args:
            box: (center, width)
                - center: (batch_size, hidden_dim)
                - width: (batch_size, hidden_dim)
            relation_trans: (batch_size, hidden_dim) 平移向量
            relation_scale: (batch_size, hidden_dim) 缩放参数（原始值）

        Returns:
            box_out: (center_out, width_out)
        """
        center, width = box

        # 平移中心
        center_out = center + relation_trans

        # 缩放宽度（Softplus保证非负）
        scale_factor = F.softplus(relation_scale)
        width_out = width * scale_factor

        return (center_out, width_out)


class IntersectionVolume(nn.Module):
    """
    计算两个盒子的交集体积

    核心：在Log域计算，数值稳定
    """
    def __init__(self, epsilon=1e-8):
        super(IntersectionVolume, self).__init__()
        self.epsilon = epsilon

    def forward(self, box_a, box_b):
        """
        Args:
            box_a: (center_a, width_a)
            box_b: (center_b, width_b)

        Returns:
            volume: 交集体积
        """
        center_a, width_a = box_a
        center_b, width_b = box_b

        # 计算边界
        min_a = center_a - width_a
        max_a = center_a + width_a
        min_b = center_b - width_b
        max_b = center_b + width_b

        # 计算交集边界
        min_inter = torch.max(min_a, min_b)
        max_inter = torch.min(max_a, max_b)

        # 计算交集宽度（负数变0）
        width_inter = torch.clamp(max_inter - min_inter, min=0.0)

        # 在Log域计算体积
        log_width_inter = torch.log(width_inter + self.epsilon)
        log_vol = torch.sum(log_width_inter, dim=-1)

        # 转回原始域
        volume = torch.exp(log_vol)

        return volume


class BoxVolume(nn.Module):
    """
    计算盒子的体积
    """
    def __init__(self, epsilon=1e-8):
        super(BoxVolume, self).__init__()
        self.epsilon = epsilon

    def forward(self, box):
        """
        Args:
            box: (center, width)

        Returns:
            volume: 盒子体积
        """
        center, width = box

        # 在Log域计算
        log_width = torch.log(2 * width + self.epsilon)
        log_vol = torch.sum(log_width, dim=-1)

        volume = torch.exp(log_vol)

        return volume
