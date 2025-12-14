"""
RulE-SSRL策略网络模块

本模块实现了规则引导的策略网络，结合以下组件：
1. LSTM用于路径历史编码（借鉴SSRL）
2. MLP用于状态表示（借鉴SSRL）
3. 规则注意力机制（核心创新）
4. 基于规则引导的动作评分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from collections import defaultdict


class RuleGuidedPolicyNetwork(nn.Module):
    """
    规则引导的策略网络

    将神经网络学习与逻辑规则引导相结合，用于知识图谱推理。

    架构：
        - LSTM：编码路径历史
        - MLP：计算状态表示
        - 规则注意力：动态学习规则权重
        - 动作评分器：基础分数 + 规则加成
    """

    def __init__(self, entity_dim, relation_dim, rule_dim, hidden_dim,
                 num_layers=1, dropout=0.1):
        """
        初始化策略网络

        参数：
            entity_dim: 实体嵌入维度（来自RulE）
            relation_dim: 关系嵌入维度（来自RulE）
            rule_dim: 规则嵌入维度（来自RulE）
            hidden_dim: LSTM和MLP的隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super(RuleGuidedPolicyNetwork, self).__init__()

        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.rule_dim = rule_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 动作维度：关系 + 实体
        self.action_dim = relation_dim + entity_dim

        # LSTM用于路径历史编码（借鉴SSRL设计）
        self.path_encoder = nn.LSTM(
            input_size=self.action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # MLP用于状态表示（借鉴SSRL设计）
        # 输入：当前实体 + 路径历史 + 查询关系
        input_dim = entity_dim + hidden_dim + relation_dim
        self.W1 = nn.Linear(input_dim, self.action_dim)
        self.W2 = nn.Linear(self.action_dim, self.action_dim)
        self.dropout = nn.Dropout(dropout)

        # 规则注意力网络（核心创新）
        self.rule_attention = nn.Sequential(
            nn.Linear(hidden_dim + relation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # 动作基础评分器
        self.action_base_scorer = nn.Linear(self.action_dim, 1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier初始化"""
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.action_base_scorer.weight)

        for name, param in self.path_encoder.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, current_entity, query_relation, path_history,
                action_space, rule_model):
        """
        前向传播，计算动作概率分布

        参数：
            current_entity: 当前实体ID (int)
            query_relation: 查询关系ID (int)
            path_history: 到目前为止的路径，格式为(关系, 实体)元组列表
            action_space: 可用动作列表，格式为(关系, 实体)元组列表
            rule_model: RulE模型实例（用于访问嵌入和规则）

        返回：
            action_probs: [len(action_space)] 动作概率分布
        """
        device = next(self.parameters()).device

        # 获取嵌入
        current_entity_emb = rule_model.get_entity_embedding_by_id(current_entity).squeeze(0)
        query_relation_emb = rule_model.get_relation_embedding_by_id(query_relation).squeeze(0)

        # 用LSTM编码路径历史
        if len(path_history) == 0:
            # 初始化为零向量
            H = torch.zeros(self.hidden_dim, device=device)
        else:
            # 获取路径的动作嵌入
            action_embs = []
            for rel, ent in path_history:
                rel_emb = rule_model.get_relation_embedding_by_id(rel).squeeze(0)
                ent_emb = rule_model.get_entity_embedding_by_id(ent).squeeze(0)
                action_emb = torch.cat([rel_emb, ent_emb], dim=-1)
                action_embs.append(action_emb)

            # 堆叠并通过LSTM
            action_seq = torch.stack(action_embs, dim=0).unsqueeze(0)  # [1, seq_len, action_dim]
            _, (h_n, _) = self.path_encoder(action_seq)
            H = h_n[-1, 0, :]  # [hidden_dim]

        # 计算状态表示（借鉴SSRL）
        X = torch.cat([current_entity_emb, H, query_relation_emb], dim=-1)  # [input_dim]
        X = self.W1(X)
        X = F.relu(X)
        X = self.dropout(X)
        X = self.W2(X)
        state_repr = self.dropout(X)  # [action_dim]

        # 获取适用的规则
        applicable_rules = rule_model.get_rules_for_relation(query_relation)

        # 计算动作得分
        action_scores = []

        for relation, next_entity in action_space:
            # 获取动作嵌入
            rel_emb = rule_model.get_relation_embedding_by_id(relation).squeeze(0)
            ent_emb = rule_model.get_entity_embedding_by_id(next_entity).squeeze(0)
            action_emb = torch.cat([rel_emb, ent_emb], dim=-1)

            # 基础得分（神经网络学习）
            base_score = torch.dot(state_repr, action_emb)

            # 规则引导加成（核心创新）
            rule_bonus = 0.0

            if len(applicable_rules) > 0:
                # 计算规则注意力权重
                rule_weights = self._compute_rule_attention_weights(
                    H, query_relation_emb, applicable_rules, device
                )

                # 检查此动作是否匹配任何规则的建议
                current_step = len(path_history)
                for (rule_id, rule_emb, rule_body), weight in zip(applicable_rules, rule_weights):
                    if current_step < len(rule_body):
                        suggested_relation = rule_body[current_step]
                        if relation == suggested_relation:
                            rule_bonus += weight.item()

            # 最终得分 = 基础分 + 规则加成
            final_score = base_score + rule_bonus
            action_scores.append(final_score)

        # 转换为概率
        action_scores = torch.stack(action_scores)
        action_probs = F.softmax(action_scores, dim=0)

        return action_probs

    def _compute_rule_attention_weights(self, state_hidden, query_relation_emb,
                                       applicable_rules, device):
        """
        计算适用规则的注意力权重

        核心创新：动态学习哪些规则值得信任

        参数：
            state_hidden: [hidden_dim] LSTM隐藏状态
            query_relation_emb: [relation_dim] 查询关系嵌入
            applicable_rules: (rule_id, rule_emb, rule_body)列表
            device: torch设备

        返回：
            weights: [num_rules] 注意力权重（softmax归一化）
        """
        if len(applicable_rules) == 0:
            return []

        # 拼接状态和查询关系
        state_query = torch.cat([state_hidden, query_relation_emb], dim=-1)  # [hidden_dim + relation_dim]

        # 计算注意力得分
        attention_scores = []
        for rule_id, rule_emb, rule_body in applicable_rules:
            # 目前使用状态+查询来计算注意力
            # 未来可以结合rule_emb
            score = self.rule_attention(state_query)  # [1]
            attention_scores.append(score)

        attention_scores = torch.cat(attention_scores)  # [num_rules]
        weights = F.softmax(attention_scores, dim=0)  # [num_rules]

        return weights

    def rollout(self, start_entity, query_relation, graph, model, max_steps=3):
        """
        执行多步rollout来采样一条路径

        参数：
            start_entity: 起始实体ID
            query_relation: 查询关系ID
            graph: KnowledgeGraph实例
            model: RulE模型实例
            max_steps: 最大步数

        返回：
            path: (关系, 实体)元组列表
            final_entity: 到达的最终实体ID
        """
        current = start_entity
        path = []

        for step in range(max_steps):
            # 获取动作空间（当前实体的邻居）
            action_space = self._get_action_space(current, graph)

            if len(action_space) == 0:
                # 死路，返回当前位置
                break

            # 计算动作概率
            with torch.no_grad():
                action_probs = self.forward(
                    current_entity=current,
                    query_relation=query_relation,
                    path_history=path,
                    action_space=action_space,
                    rule_model=model
                )

            # 采样动作
            action_idx = torch.multinomial(action_probs, 1).item()
            relation, next_entity = action_space[action_idx]

            # 更新路径和当前位置
            path.append((relation, next_entity))
            current = next_entity

        return path, current

    def _get_action_space(self, entity, graph):
        """
        获取实体的可用动作（出边）

        参数：
            entity: 实体ID
            graph: KnowledgeGraph实例

        返回：
            action_space: (关系, 目标实体)元组列表
        """
        action_space = []

        # 使用KnowledgeGraph的hr2o字典获取邻居
        # hr2o存储 (h, r) -> [t1, t2, ...] 的映射
        for relation in range(graph.relation_size * 2):  # 包括正向和逆向关系
            hr_index = graph.encode_hr(entity, relation)
            if hr_index in graph.hr2o:
                targets = graph.hr2o[hr_index]
                for target in targets:
                    action_space.append((relation, target))

        return action_space

    def initialize_path(self, init_action, device):
        """
        用虚拟起始动作初始化路径（兼容SSRL设计）

        参数：
            init_action: (关系, 实体)元组
            device: torch设备

        返回：
            (h, c): 初始LSTM隐藏状态和细胞状态
        """
        batch_size = 1  # 单条路径
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return h, c


class PolicyNetworkTrainingHelper:
    """
    策略网络训练辅助类

    提供以下功能：
    - 计算规则监督损失
    - 批处理
    - 动作空间管理
    """

    @staticmethod
    def compute_rule_supervised_loss(policy_network, query_batch, graph, model, device):
        """
        计算一批查询的规则监督损失

        参数：
            policy_network: RuleGuidedPolicyNetwork实例
            query_batch: [(h, r, t), ...] 查询三元组列表
            graph: KnowledgeGraph实例
            model: RulE模型实例
            device: torch设备

        返回：
            loss: 标量张量
        """
        total_loss = 0.0
        valid_queries = 0

        for h, r, t in query_batch:
            h, r, t = h.item(), r.item(), t.item()

            # 获取适用的规则
            applicable_rules = model.get_rules_for_relation(r)

            if len(applicable_rules) == 0:
                # 此关系没有规则，跳过
                continue

            # 模拟路径探索
            current = h
            path_history = []
            step_loss = 0.0

            for step in range(3):  # 最多3步
                # 获取动作空间
                action_space = policy_network._get_action_space(current, graph)

                if len(action_space) == 0:
                    break

                # 从策略网络获取动作概率
                action_probs = policy_network.forward(
                    current_entity=current,
                    query_relation=r,
                    path_history=path_history,
                    action_space=action_space,
                    rule_model=model
                )

                # 创建标签向量（规则监督）
                label_vector = torch.zeros(len(action_space), device=device)
                num_suggested = 0

                for i, (relation, next_entity) in enumerate(action_space):
                    # 检查此动作是否被任何规则建议
                    for rule_id, rule_emb, rule_body in applicable_rules:
                        if step < len(rule_body):
                            suggested_relation = rule_body[step]
                            if relation == suggested_relation:
                                label_vector[i] = 1.0
                                num_suggested += 1

                # 归一化标签向量
                if num_suggested > 0:
                    label_vector = label_vector / label_vector.sum()

                    # BCE损失
                    loss = F.binary_cross_entropy(action_probs, label_vector)
                    step_loss += loss

                # 采取概率最高的动作继续
                action_idx = torch.argmax(action_probs).item()
                relation, next_entity = action_space[action_idx]
                path_history.append((relation, next_entity))
                current = next_entity

                # 如果到达目标则提前停止
                if current == t:
                    break

            if step_loss > 0:
                total_loss += step_loss
                valid_queries += 1

        if valid_queries == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / valid_queries
