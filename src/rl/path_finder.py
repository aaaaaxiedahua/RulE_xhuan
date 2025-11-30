"""
PathFinderAgent - 路径搜索器

低层 Agent，使用 REINFORCE with Baseline (Actor-Critic) 在 KG 上搜索路径
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PathFinderAgent(nn.Module):
    """
    路径搜索 Agent

    使用 Actor-Critic 架构：
    - Actor (策略网络): 选择下一步要走的关系
    - Critic (价值网络): 估计当前状态的价值，用作 baseline 减小方差

    Args:
        state_dim: 状态编码维度（来自 StateEncoder）
        action_dim: 动作空间大小（关系数量）
        hidden_dim: 隐藏层维度
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PathFinderAgent, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 1. 策略网络 π(a|s) - Actor
        # 输入: 状态编码 [state_dim]
        # 输出: 动作 logits [action_dim]
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )

        # 2. 价值网络 V(s) - Critic
        # 输入: 状态编码 [state_dim]
        # 输出: 状态价值 [1]
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def select_action(self, state, action_mask=None, deterministic=False):
        """
        选择动作

        Args:
            state: 状态编码 [state_dim] 或 [batch, state_dim]
            action_mask: 有效动作掩码 [action_dim] 或 [batch, action_dim]
                        True 表示该动作可用，False 表示不可用
            deterministic: 是否确定性选择（测试时用）

        Returns:
            action: 选择的动作 ID (标量或 [batch])
            log_prob: 动作的 log 概率 (标量或 [batch])
            value: 状态价值估计 (标量或 [batch])
        """
        # 1. 策略网络输出 logits
        logits = self.policy_net(state)  # [action_dim] 或 [batch, action_dim]

        # 2. 应用动作掩码（只允许有效动作）
        if action_mask is not None:
            if not action_mask.any():
                action_mask = torch.ones_like(action_mask, dtype=torch.bool)
            # 将无效动作的 logit 设为极小值
            logits = logits.masked_fill(~action_mask, -1e9)

        # 3. Softmax 得到概率分布
        action_probs = F.softmax(logits, dim=-1)
        if torch.isnan(action_probs).any():
            action_probs = torch.nan_to_num(action_probs, nan=0.0, posinf=0.0, neginf=0.0)
            uniform = 1.0 / action_probs.size(-1)
            action_probs = action_probs + uniform
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

        # 4. 选择动作
        if deterministic:
            # 测试时：选择概率最大的动作
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1) + 1e-10)
        else:
            # 训练时：从分布中采样
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # 5. 价值网络估计状态价值
        value = self.value_net(state).squeeze(-1)  # [1] → 标量 或 [batch, 1] → [batch]

        return action, log_prob, value

    def evaluate_actions(self, states, actions):
        """
        评估给定状态-动作对（用于训练）

        Args:
            states: 状态序列 [batch, state_dim]
            actions: 动作序列 [batch]

        Returns:
            log_probs: 动作的 log 概率 [batch]
            values: 状态价值 [batch]
            entropy: 策略熵 [batch]
        """
        # 1. 策略网络输出
        logits = self.policy_net(states)  # [batch, action_dim]
        action_probs = F.softmax(logits, dim=-1)

        # 2. 计算 log 概率
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)

        # 3. 计算熵（用于鼓励探索）
        entropy = dist.entropy()

        # 4. 价值网络输出
        values = self.value_net(states).squeeze(-1)  # [batch]

        return log_probs, values, entropy

    def get_value(self, state):
        """
        获取状态价值（用于计算优势函数）

        Args:
            state: 状态编码 [state_dim] 或 [batch, state_dim]

        Returns:
            value: 状态价值 (标量或 [batch])
        """
        return self.value_net(state).squeeze(-1)
