"""
Rule-GNN 辅助层

包含 Rule-GNN 需要的辅助函数和层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max


def scatter_softmax(src, index, dim=0, dim_size=None):
    """
    对 scatter 的元素做 softmax

    Args:
        src: 源张量 [num_elements]
        index: 索引 [num_elements]
        dim: 聚合维度
        dim_size: 输出大小

    Returns:
        softmax后的张量 [num_elements]
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1

    # 先找到每个index的最大值（数值稳定性）
    max_value_per_index = scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
    max_value = max_value_per_index[index]

    # exp(src - max)
    exp_src = torch.exp(src - max_value)

    # 计算每个index的sum
    sum_per_index = scatter_add(exp_src, index, dim=dim, dim_size=dim_size)
    sum_value = sum_per_index[index]

    # softmax
    return exp_src / (sum_value + 1e-16)


class AttentionAggregation(nn.Module):
    """
    注意力聚合层

    用于聚合多个规则的信息
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weight = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: [num_items, hidden_dim]

        Returns:
            aggregated: [hidden_dim]
        """
        # 计算注意力权重
        attn_scores = self.attention_weight(x)  # [num_items, 1]
        attn_weights = F.softmax(attn_scores, dim=0)  # [num_items, 1]

        # 加权求和
        aggregated = (x * attn_weights).sum(dim=0)  # [hidden_dim]

        return aggregated


class EdgeTypeFilter(nn.Module):
    """
    边类型过滤器

    用于根据规则体过滤边
    """
    def __init__(self):
        super().__init__()

    def forward(self, edge_index, edge_type, target_relation):
        """
        过滤出指定关系类型的边

        Args:
            edge_index: [2, num_edges]
            edge_type: [num_edges]
            target_relation: 目标关系ID (int)

        Returns:
            filtered_edge_index: [2, num_filtered_edges]
            filtered_indices: 过滤后的边索引
        """
        mask = (edge_type == target_relation)
        filtered_edge_index = edge_index[:, mask]
        filtered_indices = mask.nonzero(as_tuple=True)[0]

        return filtered_edge_index, filtered_indices


class RuleMatchingLayer(nn.Module):
    """
    规则匹配层

    计算当前边与规则的匹配度
    """
    def __init__(self, num_relations, rule_dim):
        super().__init__()
        self.num_relations = num_relations
        self.rule_dim = rule_dim

        # 关系嵌入到规则空间的映射
        self.relation_to_rule = nn.Linear(rule_dim, rule_dim)

    def forward(self, edge_relation, rule_embedding, rule_body, current_layer):
        """
        计算边关系与规则的匹配度

        Args:
            edge_relation: 边的关系ID [num_edges]
            rule_embedding: 规则嵌入 [hidden_dim]
            rule_body: 规则体（关系序列） [rule_length]
            current_layer: 当前层索引

        Returns:
            match_scores: 匹配得分 [num_edges]
        """
        # 如果当前层超过规则长度，返回0
        if current_layer >= len(rule_body):
            return torch.zeros(edge_relation.size(0), device=edge_relation.device)

        # 获取规则体中当前层应该有的关系
        expected_relation = rule_body[current_layer]

        # 简单匹配：如果关系一致，得分为1，否则为0.1
        match_scores = torch.where(
            edge_relation == expected_relation,
            torch.ones_like(edge_relation, dtype=torch.float),
            torch.ones_like(edge_relation, dtype=torch.float) * 0.1
        )

        return match_scores


def create_relation_mask(edge_type, allowed_relations):
    """
    创建关系掩码

    Args:
        edge_type: [num_edges]
        allowed_relations: 允许的关系列表

    Returns:
        mask: [num_edges] 布尔型掩码
    """
    mask = torch.zeros_like(edge_type, dtype=torch.bool)
    for rel in allowed_relations:
        mask = mask | (edge_type == rel)
    return mask


def get_rule_body_relations(rules):
    """
    从规则列表中提取所有规则体的关系

    Args:
        rules: 规则列表，每个规则是字典 {'id': int, 'head': int, 'body': [int]}

    Returns:
        all_body_relations: 所有规则体中出现的关系ID集合
    """
    all_relations = set()
    for rule in rules:
        all_relations.update(rule['body'])
    return list(all_relations)
