"""
StateEncoder - 状态编码器

编码当前推理状态，包括：
- 当前实体位置
- 查询关系（目标）
- 规则上下文（指导信息）
- 路径历史（避免循环）
"""

import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """
    状态编码器

    将当前推理状态编码为固定维度的向量，用于 PathFinder 决策

    Args:
        entity_dim: 实体嵌入维度 (hidden_dim * 2，因为是复数)
        rel_dim: 关系嵌入维度 (hidden_dim)
        rule_dim: 规则嵌入维度 (mlp_rule_dim)
        history_dim: 历史编码维度
        state_dim: 输出状态维度
    """

    def __init__(self, entity_dim, rel_dim, rule_dim, history_dim=128, state_dim=128):
        super(StateEncoder, self).__init__()

        self.entity_dim = entity_dim
        self.rel_dim = rel_dim
        self.rule_dim = rule_dim
        self.history_dim = history_dim
        self.state_dim = state_dim

        # 1. 实体编码器
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_dim, state_dim),
            nn.ReLU()
        )

        # 2. 关系编码器
        self.relation_encoder = nn.Sequential(
            nn.Linear(rel_dim, state_dim),
            nn.ReLU()
        )

        # 3. 规则上下文编码器 (LSTM)
        # 输入: 每条规则的嵌入 [num_selected_rules, rule_dim]
        # 输出: 规则上下文表示 [state_dim]
        self.rule_lstm = nn.LSTM(
            input_size=rule_dim,
            hidden_size=state_dim,
            num_layers=1,
            batch_first=True
        )

        # 4. 历史路径编码器 (GRU)
        # 输入: 每步的 (entity, relation) [num_steps, entity_dim + rel_dim]
        # 输出: 路径历史表示 [history_dim]
        self.history_gru = nn.GRU(
            input_size=entity_dim + rel_dim,
            hidden_size=history_dim,
            num_layers=1,
            batch_first=True
        )

        # 5. 状态融合层
        # 输入: [entity(128) + relation(128) + rule(128) + history(128)] = 512
        # 输出: [state_dim]
        fusion_input_dim = state_dim * 3 + history_dim
        self.state_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, state_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(state_dim * 2, state_dim),
            nn.LayerNorm(state_dim)
        )

    def forward(self, current_entity, query_rel, rule_context=None, path_history=None):
        """
        编码当前状态

        Args:
            current_entity: 当前实体嵌入 [entity_dim] 或 [batch, entity_dim]
            query_rel: 查询关系嵌入 [rel_dim] 或 [batch, rel_dim]
            rule_context: 选中规则的嵌入 [num_rules, rule_dim] 或 [batch, num_rules, rule_dim]
                         如果为 None，使用零向量
            path_history: 路径历史 [num_steps, entity_dim+rel_dim] 或 [batch, num_steps, entity_dim+rel_dim]
                         如果为 None，使用零向量

        Returns:
            state_emb: 状态编码 [state_dim] 或 [batch, state_dim]
        """
        # 检查输入维度，添加 batch 维度（如果需要）
        if current_entity.dim() == 1:
            current_entity = current_entity.unsqueeze(0)
            query_rel = query_rel.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = current_entity.size(0)

        # 1. 编码当前实体
        h_entity = self.entity_encoder(current_entity)  # [batch, state_dim]

        # 2. 编码查询关系
        h_rel = self.relation_encoder(query_rel)  # [batch, state_dim]

        # 3. 编码规则上下文
        if rule_context is not None and rule_context.numel() > 0:
            # 如果 rule_context 是 2D，添加 batch 维度
            if rule_context.dim() == 2:
                rule_context = rule_context.unsqueeze(0).expand(batch_size, -1, -1)

            # LSTM 编码规则序列
            _, (h_rule, _) = self.rule_lstm(rule_context)
            h_rule = h_rule.squeeze(0)  # [batch, state_dim]
        else:
            # 没有规则上下文，使用零向量
            h_rule = torch.zeros(batch_size, self.state_dim, device=current_entity.device)

        # 4. 编码路径历史
        if path_history is not None and path_history.numel() > 0:
            # 如果 path_history 是 2D，添加 batch 维度
            if path_history.dim() == 2:
                path_history = path_history.unsqueeze(0).expand(batch_size, -1, -1)

            # GRU 编码路径序列
            _, h_history = self.history_gru(path_history)
            h_history = h_history.squeeze(0)  # [batch, history_dim]
        else:
            # 没有路径历史，使用零向量
            h_history = torch.zeros(batch_size, self.history_dim, device=current_entity.device)

        # 5. 拼接所有特征
        state = torch.cat([h_entity, h_rel, h_rule, h_history], dim=-1)  # [batch, fusion_input_dim]

        # 6. 融合得到最终状态表示
        state_emb = self.state_fusion(state)  # [batch, state_dim]

        # 移除 batch 维度（如果输入是单个样本）
        if squeeze_output:
            state_emb = state_emb.squeeze(0)

        return state_emb
