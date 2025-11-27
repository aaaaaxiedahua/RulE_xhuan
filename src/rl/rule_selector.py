"""
RuleSelectorAgent - 规则选择器

高层 Agent，使用 UCB + ε-greedy 策略为查询选择 top-K 最相关规则
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import math


class RuleSelectorAgent(nn.Module):
    """
    规则选择器 Agent

    使用 UCB (Upper Confidence Bound) + ε-greedy 策略选择规则

    Args:
        entity_dim: 实体嵌入维度
        rel_dim: 关系嵌入维度
        rule_dim: 规则嵌入维度
        num_rules: 规则总数
        hidden_dim: 隐藏层维度
        ucb_c: UCB 探索系数
    """

    def __init__(self, entity_dim, rel_dim, rule_dim, num_rules, hidden_dim=128, ucb_c=1.0):
        super(RuleSelectorAgent, self).__init__()

        self.entity_dim = entity_dim
        self.rel_dim = rel_dim
        self.rule_dim = rule_dim
        self.num_rules = num_rules
        self.hidden_dim = hidden_dim
        self.ucb_c = ucb_c

        # 1. 查询编码器：(entity, relation) → query representation
        query_input_dim = entity_dim + rel_dim
        self.query_encoder = nn.Sequential(
            nn.Linear(query_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 2. 规则-查询匹配网络
        # 输入: [query_emb(hidden_dim) + rule_emb(rule_dim)]
        # 输出: 匹配分数 (标量)
        self.rule_query_matcher = nn.Sequential(
            nn.Linear(hidden_dim + rule_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 3. UCB 统计（用于探索-利用平衡）
        self.rule_counts = defaultdict(int)       # 每条规则被选次数
        self.rule_rewards = defaultdict(float)    # 每条规则的累积奖励
        self.total_selections = 0                 # 总选择次数

    def forward(self, query_entity, query_relation, rule_embeddings,
                epsilon=0.1, top_k=5, deterministic=False):
        """
        选择 top-K 规则

        Args:
            query_entity: 查询实体嵌入 [entity_dim]
            query_relation: 查询关系嵌入 [rel_dim]
            rule_embeddings: 所有规则的嵌入 [num_rules, rule_dim]
            epsilon: ε-greedy 探索率
            top_k: 选择规则数量
            deterministic: 是否确定性选择（测试时用）

        Returns:
            selected_rules: 选中的规则 ID [top_k]
            selection_probs: 选择概率 [top_k]
        """
        device = query_entity.device

        # 1. 编码查询 (entity + relation)
        query_repr = torch.cat([query_entity, query_relation], dim=-1)  # [entity_dim + rel_dim]
        query_emb = self.query_encoder(query_repr)  # [hidden_dim]

        # 2. 计算每条规则的匹配分数
        rule_scores = []
        for rule_id in range(self.num_rules):
            rule_emb = rule_embeddings[rule_id]  # [rule_dim]
            combined = torch.cat([query_emb, rule_emb], dim=-1)  # [hidden_dim + rule_dim]
            score = self.rule_query_matcher(combined)  # [1]
            rule_scores.append(score.squeeze())

        rule_scores = torch.stack(rule_scores)  # [num_rules]

        # 3. 计算 UCB 分数（exploration bonus）
        if not deterministic:
            ucb_scores = torch.zeros_like(rule_scores)
            for rule_id in range(self.num_rules):
                # 平均奖励（利用项）
                avg_reward = self.rule_rewards[rule_id] / (self.rule_counts[rule_id] + 1e-9)

                # UCB bonus（探索项）: c * sqrt(log(N) / n_i)
                if self.total_selections > 0:
                    ucb_bonus = self.ucb_c * math.sqrt(
                        math.log(self.total_selections + 1) / (self.rule_counts[rule_id] + 1)
                    )
                else:
                    ucb_bonus = 1.0  # 初始时给予较大探索奖励

                ucb_scores[rule_id] = rule_scores[rule_id] + ucb_bonus
        else:
            # 测试时不使用 UCB，直接用匹配分数
            ucb_scores = rule_scores

        # 4. ε-greedy 选择
        if not deterministic and np.random.random() < epsilon:
            # 探索：随机选择
            selected_rules = torch.randperm(self.num_rules, device=device)[:top_k]
        else:
            # 利用：选择 UCB 分数最高的
            _, selected_rules = torch.topk(ucb_scores, k=min(top_k, self.num_rules))

        # 5. 计算选择概率（用于策略梯度更新）
        selected_scores = rule_scores[selected_rules]
        selection_probs = F.softmax(selected_scores, dim=0)

        return selected_rules, selection_probs

    def update_statistics(self, rule_id, reward):
        """
        更新 UCB 统计信息

        Args:
            rule_id: 规则 ID
            reward: 获得的奖励
        """
        self.rule_counts[rule_id] += 1
        self.rule_rewards[rule_id] += reward
        self.total_selections += 1

    def reset_statistics(self):
        """重置 UCB 统计（新 epoch 开始时）"""
        self.rule_counts = defaultdict(int)
        self.rule_rewards = defaultdict(float)
        self.total_selections = 0

    def get_statistics(self):
        """
        获取统计信息（用于分析）

        Returns:
            stats: 字典，包含规则使用统计
        """
        stats = {
            'total_selections': self.total_selections,
            'unique_rules_used': len(self.rule_counts),
            'top_rules': sorted(
                [(rule_id, count, self.rule_rewards[rule_id] / (count + 1e-9))
                 for rule_id, count in self.rule_counts.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]  # Top 10 最常用规则
        }
        return stats
