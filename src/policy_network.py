"""
RulE-SSRL策略网络模块（重构版）

按照SSRL架构重构，采用向量化批处理模式，保留规则引导的核心创新。

主要改进：
1. 预计算的固定大小动作空间（max_num_actions=200）
2. 向量化的动作评分计算
3. 批处理的规则匹配机制
4. GPU友好的实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RuleGuidedPolicyNetwork(nn.Module):
    """
    规则引导的策略网络（向量化批处理版本）

    架构参考SSRL，但添加规则引导机制作为核心创新。
    """

    def __init__(self, entity_dim, relation_dim, rule_dim, hidden_dim,
                 max_num_actions=200, num_layers=1, dropout=0.1,
                 rule_bonus_coef=0.1, rule_bonus_default=0.05):
        """
        初始化策略网络

        参数:
            entity_dim: 实体嵌入维度
            relation_dim: 关系嵌入维度
            rule_dim: 规则嵌入维度
            hidden_dim: LSTM隐藏层维度
            max_num_actions: 最大动作数（固定大小，参考SSRL）
            num_layers: LSTM层数
            dropout: Dropout比率
            rule_bonus_coef: 推理阶段规则加成系数（默认0.1）
            rule_bonus_default: 推理阶段默认规则加成（默认0.05）
        """
        super(RuleGuidedPolicyNetwork, self).__init__()

        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.rule_dim = rule_dim
        self.hidden_dim = hidden_dim
        self.max_num_actions = max_num_actions
        self.num_layers = num_layers

        # 推理阶段规则加成参数（从配置文件读取）
        self.rule_bonus_coef = rule_bonus_coef
        self.rule_bonus_default = rule_bonus_default

        # 动作嵌入维度：关系 + 实体
        self.action_dim = relation_dim + entity_dim

        # LSTM用于路径历史编码（参考SSRL设计）
        self.path_encoder = nn.LSTM(
            input_size=self.action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # MLP用于状态表示（参考SSRL的policy_MLP）
        # 输入：当前实体 + 路径历史 + 查询关系
        input_dim = entity_dim + hidden_dim + relation_dim
        self.W1 = nn.Linear(input_dim, self.action_dim)
        self.W2 = nn.Linear(self.action_dim, self.action_dim)
        self.dropout = nn.Dropout(dropout)

        # RulE-SSRL核心创新：规则引导层
        # 为每个关系预计算规则匹配矩阵
        # 规则注意力网络：学习规则权重
        self.rule_attention = nn.Sequential(
            nn.Linear(hidden_dim + relation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier初始化"""
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

        for name, param in self.path_encoder.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def get_init_state(self, batch_size, device):
        """
        获取初始LSTM状态

        参数:
            batch_size: 批大小
            device: 设备

        返回:
            (h0, c0): 初始隐藏状态和细胞状态
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    def step(self, next_relations, next_entities, lstm_state, prev_relation,
             query_embedding, current_entities, rule_model, current_step=0,
             query_relation_ids=None):
        """
        执行一步策略网络前向传播（向量化批处理版本，参考SSRL agent.step）

        参数:
            next_relations: [batch_size, max_num_actions] 可用关系
            next_entities: [batch_size, max_num_actions] 可用目标实体
            lstm_state: (h, c) LSTM状态，每个形状为 [num_layers, batch_size, hidden_dim]
            prev_relation: [batch_size] 上一步选择的关系
            query_embedding: [batch_size, relation_dim] 查询关系嵌入
            current_entities: [batch_size] 当前实体ID
            rule_model: RulE模型实例（用于获取嵌入和规则信息）
            current_step: 当前步数（用于规则匹配）
            query_relation_ids: [batch_size] 查询关系ID（用于规则匹配）

        返回:
            logits: [batch_size, max_num_actions] 动作log概率
            new_lstm_state: (h, c) 新的LSTM状态
            action_idx: [batch_size] 采样的动作索引
        """
        device = next_relations.device
        batch_size = next_relations.size(0)

        # 1. 编码上一步动作：关系嵌入
        # prev_relation shape: [batch_size]
        prev_relation_emb = rule_model.get_relation_embedding_by_id(prev_relation)  # [batch_size, relation_dim]

        # 当前实体嵌入
        current_entity_emb = rule_model.get_entity_embedding_by_id(current_entities)  # [batch_size, entity_dim]

        # 上一步动作嵌入 = 关系 + 实体
        prev_action_emb = torch.cat([prev_relation_emb, current_entity_emb], dim=-1)  # [batch_size, action_dim]

        # 2. LSTM更新：输入上一步动作，更新路径历史状态
        # LSTM期望输入 [batch_size, seq_len=1, action_dim]
        lstm_input = prev_action_emb.unsqueeze(1)  # [batch_size, 1, action_dim]
        lstm_output, new_lstm_state = self.path_encoder(lstm_input, lstm_state)
        H = lstm_output.squeeze(1)  # [batch_size, hidden_dim]

        # 3. 计算状态表示（参考SSRL的MLP）
        # 拼接：当前实体 + 路径历史 + 查询关系
        state_input = torch.cat([current_entity_emb, H, query_embedding], dim=-1)  # [batch_size, input_dim]
        X = self.W1(state_input)
        X = F.relu(X)
        X = self.dropout(X)
        X = self.W2(X)
        state_repr = self.dropout(X)  # [batch_size, action_dim]

        # 4. 获取候选动作嵌入（向量化）
        # next_relations: [batch_size, max_num_actions]
        # next_entities: [batch_size, max_num_actions]

        # 展平索引以批量获取嵌入
        flat_relations = next_relations.view(-1)  # [batch_size * max_num_actions]
        flat_entities = next_entities.view(-1)

        # 获取嵌入
        relation_embs = rule_model.get_relation_embedding_by_id(flat_relations)  # [B*A, relation_dim]
        entity_embs = rule_model.get_entity_embedding_by_id(flat_entities)  # [B*A, entity_dim]

        # 拼接并重塑
        action_embs = torch.cat([relation_embs, entity_embs], dim=-1)  # [B*A, action_dim]
        action_embs = action_embs.view(batch_size, self.max_num_actions, self.action_dim)  # [B, A, action_dim]

        # 5. 计算基础动作得分（点积）
        # state_repr: [batch_size, action_dim]
        # action_embs: [batch_size, max_num_actions, action_dim]
        base_scores = torch.bmm(
            action_embs,
            state_repr.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, max_num_actions]

        # 6. RulE-SSRL核心创新：添加规则引导加成
        # 计算规则加成（向量化）
        rule_bonus = self._compute_rule_bonus_vectorized(
            next_relations, H, query_embedding, rule_model, current_step, device,
            query_relation_ids=query_relation_ids
        )  # [batch_size, max_num_actions]

        # 7. 组合得分
        final_scores = base_scores + rule_bonus  # [batch_size, max_num_actions]

        # 8. 处理PAD动作：创建mask
        # PAD实体的标记（假设ePAD是实体数量）
        ePAD = rule_model.graph.ePAD
        valid_mask = (next_entities != ePAD).float()  # [batch_size, max_num_actions]

        # 对PAD位置设置为极小值
        final_scores = final_scores.masked_fill(valid_mask == 0, -1e10)

        # 9. 计算log概率
        logits = F.log_softmax(final_scores, dim=1)  # [batch_size, max_num_actions]

        # 10. 采样动作
        probs = torch.exp(logits)
        action_idx = torch.multinomial(probs, 1).squeeze(1)  # [batch_size]

        return logits, new_lstm_state, action_idx

    def _compute_rule_bonus_vectorized(self, next_relations, state_hidden, query_relation_emb,
                                       rule_model, current_step, device,
                                       query_relation_ids=None):
        """
        向量化计算规则引导加成（核心创新）

        根据规则体中当前步骤期望的关系，给匹配的动作加成。
        改进：使用预训练的规则质量权重 (rules_weight_emb) 进行加权。

        参数:
            next_relations: [batch_size, max_num_actions] 候选关系
            state_hidden: [batch_size, hidden_dim] LSTM隐藏状态
            query_relation_emb: [batch_size, relation_dim] 查询关系嵌入
            rule_model: RulE模型实例
            current_step: 当前步数
            device: torch设备
            query_relation_ids: [batch_size] 查询关系ID（用于规则匹配）

        返回:
            rule_bonus: [batch_size, max_num_actions] 规则加成得分
        """
        batch_size = next_relations.size(0)
        rule_bonus = torch.zeros(batch_size, self.max_num_actions, device=device)

        # 计算规则注意力权重（批处理）
        # 拼接状态和查询关系
        state_query = torch.cat([state_hidden, query_relation_emb], dim=-1)  # [batch_size, hidden_dim + relation_dim]
        attention_scores = self.rule_attention(state_query)  # [batch_size, 1]
        rule_weight = torch.sigmoid(attention_scores)  # [batch_size, 1]

        # 如果没有提供query_relation_ids，退化为简化版本
        if query_relation_ids is None:
            for b in range(batch_size):
                rule_bonus[b, :] = rule_weight[b, 0] * 0.1
            return rule_bonus

        # 检查是否有预计算的规则质量权重
        has_rule_quality = hasattr(rule_model, 'rules_weight_emb') and rule_model.rules_weight_emb is not None

        # ⚡ 性能优化：预先转换为numpy避免重复.item()调用
        if isinstance(query_relation_ids, torch.Tensor):
            query_relation_ids_np = query_relation_ids.cpu().numpy()
        else:
            query_relation_ids_np = query_relation_ids

        # ⚡ 性能优化：预先计算所有规则的质量（避免循环中重复计算）
        if has_rule_quality:
            rules_quality_cache = torch.norm(rule_model.rules_weight_emb, dim=1).clamp(min=0.1).cpu().numpy()
        else:
            rules_quality_cache = None

        # 规则匹配打分：检查候选动作的关系是否匹配规则体当前步期望的关系
        for b in range(batch_size):
            query_rel = query_relation_ids_np[b]  # ⚡ 直接从numpy读取，无需.item()

            # 获取该查询关系对应的所有规则
            if not hasattr(rule_model, 'relation2rules') or query_rel >= len(rule_model.relation2rules):
                # 没有规则，使用默认加成
                rule_bonus[b, :] = rule_weight[b, 0] * 0.1
                continue

            rules = rule_model.relation2rules[query_rel]

            if len(rules) == 0:
                # 没有规则，使用默认加成
                rule_bonus[b, :] = rule_weight[b, 0] * 0.1
                continue

            # 遍历每条规则，计算加权匹配加成
            weighted_match = torch.zeros(self.max_num_actions, device=device)
            total_rule_quality = 0.0

            for rule_id, (r_head, r_body) in rules:
                # r_body 是规则体的关系列表，如 [属于, 适用症状]
                if current_step < len(r_body):
                    expected_rel = r_body[current_step]  # 当前步期望的关系

                    # ===== 改进：获取规则质量权重 =====
                    if rules_quality_cache is not None:
                        # ⚡ 从预计算的缓存中读取，避免重复计算和.item()
                        rule_quality = float(rules_quality_cache[rule_id])
                    else:
                        rule_quality = 1.0  # 退化为原来的等权重

                    # 检查哪些候选动作的关系匹配期望关系
                    match_mask = (next_relations[b] == expected_rel).float()
                    weighted_match += rule_quality * match_mask
                    total_rule_quality += rule_quality

            if total_rule_quality > 0:
                # 归一化匹配次数，乘以规则权重
                # 匹配的动作获得更高加成
                normalized_match = weighted_match / total_rule_quality
                rule_bonus[b, :] = rule_weight[b, 0] * normalized_match * self.rule_bonus_coef  # 使用配置参数
            else:
                # 没有适用的规则（current_step超出规则长度），使用默认加成
                rule_bonus[b, :] = rule_weight[b, 0] * self.rule_bonus_default  # 使用配置参数

        return rule_bonus

    def rollout(self, start_entity, query_relation, graph, model, max_steps=3):
        """
        执行多步rollout来采样一条路径（用于推理阶段）

        参数:
            start_entity: 起始实体ID
            query_relation: 查询关系ID
            graph: KnowledgeGraph实例
            model: RulE模型实例
            max_steps: 最大步数

        返回:
            path: (关系, 实体)元组列表
            final_entity: 到达的最终实体ID
        """
        device = next(self.parameters()).device

        # 初始化
        current_entity = start_entity
        path = []

        # LSTM状态
        lstm_state = self.get_init_state(1, device)

        # 初始关系
        prev_relation = torch.tensor([graph.rPAD], dtype=torch.long, device=device)

        # 查询嵌入和ID
        query_relation_tensor = torch.tensor([query_relation], dtype=torch.long, device=device)
        query_embedding = model.get_relation_embedding_by_id(query_relation_tensor)
        query_relation_ids = query_relation_tensor  # 用于规则匹配

        for step in range(max_steps):
            # 获取动作空间
            next_actions = graph.array_store[current_entity:current_entity+1, :, :].copy()  # [1, max_actions, 2]

            next_entities_np = next_actions[:, :, 0]
            next_relations_np = next_actions[:, :, 1]

            # 转换为tensor
            next_entities = torch.from_numpy(next_entities_np).long().to(device)
            next_relations = torch.from_numpy(next_relations_np).long().to(device)
            current_entities_tensor = torch.tensor([current_entity], dtype=torch.long, device=device)

            # 检查是否有有效动作
            valid_mask = (next_entities != graph.ePAD).any(dim=1)
            if not valid_mask.item():
                # 没有有效动作，停止
                break

            # 前向传播（传入query_relation_ids用于规则匹配）
            logits, lstm_state, action_idx = self.step(
                next_relations=next_relations,
                next_entities=next_entities,
                lstm_state=lstm_state,
                prev_relation=prev_relation,
                query_embedding=query_embedding,
                current_entities=current_entities_tensor,
                rule_model=model,
                current_step=step,
                query_relation_ids=query_relation_ids
            )

            # 提取采样的动作
            action_idx_val = action_idx.item()
            chosen_relation = next_relations[0, action_idx_val].item()
            chosen_entity = next_entities[0, action_idx_val].item()

            # 检查是否为PAD
            if chosen_entity == graph.ePAD:
                break

            # 更新路径和当前位置
            path.append((chosen_relation, chosen_entity))
            current_entity = chosen_entity
            prev_relation = torch.tensor([chosen_relation], dtype=torch.long, device=device)

        return path, current_entity


class PolicyNetworkTrainingHelper:
    """
    策略网络训练辅助类（重构版）

    提供向量化的规则监督损失计算。
    """

    @staticmethod
    def compute_rule_supervised_loss(policy_network, query_batch, graph, model, device,
                                    path_length=3, num_rollouts=1, lambda_rule=0.3):
        """
        计算一批查询的规则监督损失（目标导向版本）

        核心思想：reward = target + λ * rule
        改进：目标实体为主要监督（满分1.0），规则匹配为辅助监督（λ×质量）

        参数:
            policy_network: RuleGuidedPolicyNetwork实例
            query_batch: [batch_size, 3] 查询三元组 (h, r, t)
            graph: KnowledgeGraph实例
            model: RulE模型实例
            device: torch设备
            path_length: 路径长度
            num_rollouts: 每个查询的rollout数量
            lambda_rule: 规则权重系数（从配置文件读取，默认0.3）

        返回:
            loss: 标量张量
        """
        batch_size = query_batch.size(0)

        # 提取查询
        start_entities = query_batch[:, 0].cpu().numpy()  # [batch_size]
        query_relations = query_batch[:, 1].cpu().numpy()
        target_entities = query_batch[:, 2].cpu().numpy()

        # 扩展为多个rollout
        if num_rollouts > 1:
            start_entities = np.repeat(start_entities, num_rollouts)
            query_relations = np.repeat(query_relations, num_rollouts)
            target_entities = np.repeat(target_entities, num_rollouts)
            expanded_batch_size = batch_size * num_rollouts
        else:
            expanded_batch_size = batch_size

        # 初始化状态
        current_entities = start_entities.copy()
        lstm_state = policy_network.get_init_state(expanded_batch_size, device)

        # 初始关系（DUMMY_START）
        prev_relation = torch.full((expanded_batch_size,), graph.rPAD, dtype=torch.long, device=device)

        # 查询嵌入和ID
        query_relation_tensor = torch.from_numpy(query_relations).long().to(device)
        query_embedding = model.get_relation_embedding_by_id(query_relation_tensor)

        # 检查是否有预计算的规则质量权重
        has_rule_quality = hasattr(model, 'rules_weight_emb') and model.rules_weight_emb is not None

        total_loss = 0.0
        valid_steps = 0

        # 模拟路径探索
        for step in range(path_length):
            # 获取动作空间（使用预计算的array_store）
            next_actions = graph.array_store[current_entities, :, :].copy()  # [B, max_actions, 2]

            next_entities_np = next_actions[:, :, 0]
            next_relations_np = next_actions[:, :, 1]

            # 转换为tensor
            next_entities = torch.from_numpy(next_entities_np).long().to(device)
            next_relations = torch.from_numpy(next_relations_np).long().to(device)
            current_entities_tensor = torch.from_numpy(current_entities).long().to(device)

            # 前向传播（传入query_relation_ids用于规则匹配打分）
            logits, lstm_state, action_idx = policy_network.step(
                next_relations=next_relations,
                next_entities=next_entities,
                lstm_state=lstm_state,
                prev_relation=prev_relation,
                query_embedding=query_embedding,
                current_entities=current_entities_tensor,
                rule_model=model,
                current_step=step,
                query_relation_ids=query_relation_tensor
            )

            # ========== 改进：目标导向监督（目标主导 + 规则辅助） ==========
            # 核心思想：reward = target + λ * rule
            # 目标实体奖励（满分1.0）+ 规则匹配奖励（0.3×质量）

            # Part 1: 计算目标实体奖励（直接奖励）
            target_entity_reward = (next_entities == torch.from_numpy(target_entities).unsqueeze(1).to(device)).float()
            # target_entity_reward: [batch, max_actions], 值为 {0, 1}

            # Part 2: 计算规则匹配奖励（规则打分）
            rule_match_reward = torch.zeros(expanded_batch_size, policy_network.max_num_actions, device=device)
            has_rule_supervision = False

            for b in range(expanded_batch_size):
                query_rel = query_relations[b]

                # 获取该查询关系对应的所有规则
                if not hasattr(model, 'relation2rules') or query_rel >= len(model.relation2rules):
                    continue

                rules = model.relation2rules[query_rel]
                if len(rules) == 0:
                    continue

                # 遍历每条规则，找出当前步期望的关系（使用规则质量加权）
                for rule_id, (r_head, r_body) in rules:
                    # r_body 是规则体的关系列表，如 [属于, 适用症状]
                    if step < len(r_body):
                        expected_rel = r_body[step]  # 当前步期望的关系

                        # 获取规则质量权重
                        if has_rule_quality:
                            rule_emb = model.rules_weight_emb[rule_id]  # [hidden_dim]
                            rule_quality = torch.norm(rule_emb).item()
                            rule_quality = max(rule_quality, 0.1)  # 避免为0
                        else:
                            rule_quality = 1.0  # 退化为原来的等权重

                        # 找到关系匹配的动作，累加规则质量
                        match_mask = (next_relations[b] == expected_rel).float()
                        rule_match_reward[b] += rule_quality * match_mask  # 加权累加
                        if match_mask.sum() > 0:
                            has_rule_supervision = True

            # Part 3: 组合奖励（目标主导 + 规则辅助）
            # 使用传入的 lambda_rule 参数（从配置文件读取）
            total_reward = target_entity_reward + lambda_rule * rule_match_reward
            # 说明：
            # - 找到目标 → 奖励 = 1.0（满分保底）
            # - 匹配规则 → 奖励 = lambda_rule × rule_quality（额外加成）
            # - 目标+规则 → 奖励 = 1.0 + lambda_rule×quality（最佳情况）

            # Part 4: 过滤无效动作并归一化
            valid_mask = (next_entities != graph.ePAD).float()
            total_reward = total_reward * valid_mask

            reward_sum = total_reward.sum(dim=1, keepdim=True)

            if reward_sum.sum() > 0:
                # 有奖励信号：归一化为概率分布
                valid_rows = (reward_sum > 0).float()
                reward_dist = total_reward / (reward_sum + 1e-10)

                # 计算KL散度损失
                step_loss = F.kl_div(logits, reward_dist, reduction='none')
                step_loss = (step_loss.sum(dim=1) * valid_rows.squeeze()).sum() / (valid_rows.sum() + 1e-10)

                total_loss += step_loss
                valid_steps += 1

            # 执行动作
            chosen_entities = next_entities[torch.arange(expanded_batch_size, device=device), action_idx]
            chosen_relations = next_relations[torch.arange(expanded_batch_size, device=device), action_idx]

            current_entities = chosen_entities.cpu().numpy()
            prev_relation = chosen_relations

        if valid_steps == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss / valid_steps

    @staticmethod
    def compute_policy_gradient_loss_with_rule_shaping(
        policy_network, query_batch, graph, model, device,
        path_length=3, num_rollouts=5, lambda_rule=0.3, gamma=0.99,
        baseline='avg_reward_normalized', entropy_weight=0.01
    ):
        """
        方案3：Policy Gradient + 规则塑形奖励

        核心思想：
        1. 主要奖励：只在最后检查是否到达目标（1或0）
        2. 塑形奖励：每步遵循规则的额外奖励（lambda×质量）
        3. 累积折扣：反向传播奖励到每一步
        4. Baseline：减小方差
        5. Entropy：鼓励探索

        参数:
            policy_network: RuleGuidedPolicyNetwork实例
            query_batch: [batch_size, 3] 查询三元组 (h, r, t)
            graph: KnowledgeGraph实例
            model: RulE模型实例
            device: torch设备
            path_length: 路径长度
            num_rollouts: 每个查询的rollout数量（至少5，用于baseline）
            lambda_rule: 规则塑形系数（0.1-0.3）
            gamma: 折扣因子（0.95-0.99）
            baseline: 'n/a', 'avg_reward', 'avg_reward_normalized'
            entropy_weight: 熵正则化权重（0.01-0.05）

        返回:
            loss: 标量张量
            metrics: 统计字典
        """
        import torch.nn.functional as F

        batch_size = query_batch.size(0)

        # 提取查询
        start_entities = query_batch[:, 0].cpu().numpy()
        query_relations = query_batch[:, 1].cpu().numpy()
        target_entities = query_batch[:, 2].cpu().numpy()

        # 扩展为多个rollout
        if num_rollouts > 1:
            start_entities = np.repeat(start_entities, num_rollouts)
            query_relations = np.repeat(query_relations, num_rollouts)
            target_entities = np.repeat(target_entities, num_rollouts)
            expanded_batch_size = batch_size * num_rollouts
        else:
            expanded_batch_size = batch_size

        # ========== Step 1: 执行Rollout ==========
        import logging
        logging.info('[DEBUG] 开始Rollout, batch_size={}, num_rollouts={}, path_length={}'.format(
            batch_size, num_rollouts, path_length))

        current_entities = start_entities.copy()
        lstm_state = policy_network.get_init_state(expanded_batch_size, device)
        prev_relation = torch.full((expanded_batch_size,), graph.rPAD, dtype=torch.long, device=device)

        query_relation_tensor = torch.from_numpy(query_relations).long().to(device)
        query_embedding = model.get_relation_embedding_by_id(query_relation_tensor)

        has_rule_quality = hasattr(model, 'rules_weight_emb') and model.rules_weight_emb is not None

        # 存储路径信息
        log_action_probs = []  # 每步的log概率
        entropies = []          # 每步的熵
        shaping_rewards = []    # 每步的规则塑形奖励

        for step in range(path_length):
            logging.info('[DEBUG] Step {}/{}'.format(step+1, path_length))
            # 获取动作空间
            logging.info('[DEBUG] 开始获取动作空间...')
            next_actions = graph.array_store[current_entities, :, :].copy()
            logging.info('[DEBUG] 动作空间获取完成, shape={}'.format(next_actions.shape))
            next_entities_np = next_actions[:, :, 0]
            next_relations_np = next_actions[:, :, 1]

            next_entities = torch.from_numpy(next_entities_np).long().to(device)
            next_relations = torch.from_numpy(next_relations_np).long().to(device)
            current_entities_tensor = torch.from_numpy(current_entities).long().to(device)

            # 前向传播
            logits, lstm_state, action_idx = policy_network.step(
                next_relations=next_relations,
                next_entities=next_entities,
                lstm_state=lstm_state,
                prev_relation=prev_relation,
                query_embedding=query_embedding,
                current_entities=current_entities_tensor,
                rule_model=model,
                current_step=step,
                query_relation_ids=query_relation_tensor
            )

            # 记录采样动作的log概率
            action_log_prob = logits[torch.arange(expanded_batch_size, device=device), action_idx]
            log_action_probs.append(action_log_prob)

            # 计算熵（鼓励探索）
            probs = torch.exp(logits)
            entropy = -(probs * logits).sum(dim=-1)  # [expanded_batch_size]
            entropies.append(entropy)

            # ========== 计算规则塑形奖励（当前步）==========
            logging.info('[DEBUG] 开始计算规则塑形奖励...')
            step_shaping_reward = torch.zeros(expanded_batch_size, device=device)
            chosen_relations = next_relations[torch.arange(expanded_batch_size, device=device), action_idx]

            # ⚡ 性能优化：预先计算规则质量缓存
            if has_rule_quality:
                rules_quality_cache = torch.norm(model.rules_weight_emb, dim=1).clamp(min=0.1).cpu().numpy()
            else:
                rules_quality_cache = None

            # ⚡ 性能优化：转换为numpy避免重复.item()
            chosen_relations_np = chosen_relations.cpu().numpy()

            for b in range(expanded_batch_size):
                if b % 100 == 0 and b > 0:
                    logging.info('[DEBUG] 规则匹配进度: {}/{}'.format(b, expanded_batch_size))

                query_rel = query_relations[b]  # query_relations已经是numpy数组

                if not hasattr(model, 'relation2rules') or query_rel >= len(model.relation2rules):
                    continue

                rules = model.relation2rules[query_rel]
                if len(rules) == 0:
                    continue

                chosen_rel = chosen_relations_np[b]  # ⚡ 直接从numpy读取

                # 检查当前步的关系是否匹配规则
                for rule_id, (r_head, r_body) in rules:
                    if step < len(r_body):
                        expected_rel = r_body[step]

                        if chosen_rel == expected_rel:
                            # 匹配！给予塑形奖励
                            if rules_quality_cache is not None:
                                # ⚡ 从缓存读取，避免重复计算
                                rule_quality = float(rules_quality_cache[rule_id])
                            else:
                                rule_quality = 1.0

                            step_shaping_reward[b] += lambda_rule * rule_quality
                            break  # 匹配一条规则即可

            logging.info('[DEBUG] 规则塑形奖励计算完成')
            shaping_rewards.append(step_shaping_reward)

            # 执行动作
            chosen_entities = next_entities[torch.arange(expanded_batch_size, device=device), action_idx]
            current_entities = chosen_entities.cpu().numpy()
            prev_relation = chosen_relations

        # ========== Step 2: 计算主要奖励（最终目标）==========
        final_entities = torch.from_numpy(current_entities).long().to(device)
        target_entities_tensor = torch.from_numpy(target_entities).long().to(device)

        primary_reward = (final_entities == target_entities_tensor).float()  # {0, 1}

        # ========== Step 3: 计算累积折扣奖励 ==========
        cumulative_rewards = [torch.zeros(expanded_batch_size, device=device) for _ in range(path_length)]
        cumulative_rewards[-1] = primary_reward + shaping_rewards[-1]

        # 反向累积（从最后一步向前）
        for t in range(path_length - 2, -1, -1):
            cumulative_rewards[t] = shaping_rewards[t] + gamma * cumulative_rewards[t + 1]

        # 转换为张量 [expanded_batch_size, path_length]
        cumulative_rewards_tensor = torch.stack(cumulative_rewards, dim=1)

        # ========== Step 4: Baseline稳定化（减小方差）==========
        if baseline != 'n/a' and num_rollouts > 1:
            # 重塑为 [batch_size, num_rollouts, path_length]
            cumulative_3d = cumulative_rewards_tensor.view(batch_size, num_rollouts, path_length)

            if baseline == 'avg_reward':
                # 减去每个查询的平均奖励
                baseline_vals = cumulative_3d.mean(dim=1, keepdim=True)
                cumulative_3d = cumulative_3d - baseline_vals

            elif baseline == 'avg_reward_normalized':
                # 减均值 + 归一化标准差
                mean = cumulative_3d.mean(dim=1, keepdim=True)
                std = cumulative_3d.std(dim=1, keepdim=True) + 1e-8
                cumulative_3d = (cumulative_3d - mean) / std

            # 重新展平
            cumulative_rewards_tensor = cumulative_3d.view(expanded_batch_size, path_length)

        # ========== Step 5: 计算Policy Gradient损失 ==========
        pg_loss = 0.0
        for t in range(path_length):
            pg_loss += -(log_action_probs[t] * cumulative_rewards_tensor[:, t]).mean()

        # ========== Step 6: Entropy正则化（鼓励探索）==========
        entropy_loss = torch.stack(entropies).mean()

        # 总损失 = PG损失 - 熵奖励
        total_loss = pg_loss - entropy_weight * entropy_loss

        # ========== 统计信息 ==========
        metrics = {
            'reward_avg': primary_reward.mean().item(),
            'success_rate': (primary_reward > 0.5).float().mean().item(),
            'entropy': entropy_loss.item(),
            'pg_loss': pg_loss.item(),
            'avg_cumulative_reward': cumulative_rewards_tensor[:, 0].mean().item(),
        }

        return total_loss, metrics
