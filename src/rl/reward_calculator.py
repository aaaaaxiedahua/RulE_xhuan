"""
RewardCalculator - 奖励计算器

计算路径的奖励，包括：
1. 最终奖励：是否到达目标实体
2. 规则一致性奖励：路径是否符合规则
3. 接近目标奖励：是否在嵌入空间中接近目标
"""

import torch
import torch.nn.functional as F


class RewardCalculator:
    """
    奖励计算器

    奖励公式：
    R = R_final + α * (R_rule + (1 - R_final) * R_closer)

    Args:
        rule_model: 预训练的 RulE 模型（用于获取嵌入和规则信息）
        alpha: 中间奖励权重
    """

    def __init__(self, rule_model, alpha=0.1):
        self.rule_model = rule_model
        self.alpha = alpha

        # 预处理规则：构建规则体 → 规则头的映射
        self._build_rule_index()

    def _build_rule_index(self):
        """
        构建规则索引，用于快速查找匹配的规则

        规则格式: [rule_id, rule_head, rel_1, rel_2, ...]
        """
        self.rule_body_to_head = {}  # {(rel_1, rel_2, ...): [(rule_id, rule_head), ...]}

        for rule_id, rule in enumerate(self.rule_model.rules):
            if len(rule) < 3:  # 至少需要 [rule_id, rule_head, rel_1]
                continue

            rule_head = rule[1]
            rule_body = tuple(rule[2:])  # (rel_1, rel_2, ...)

            if rule_body not in self.rule_body_to_head:
                self.rule_body_to_head[rule_body] = []

            self.rule_body_to_head[rule_body].append((rule_id, rule_head))

    def compute_reward(self, trajectory, target_entity, query_relation):
        """
        计算轨迹的总奖励

        Args:
            trajectory: 路径 [(entity_0, None), (entity_1, rel_0), (entity_2, rel_1), ...]
            target_entity: 目标实体 ID
            query_relation: 查询关系 ID

        Returns:
            total_reward: 总奖励 (标量)
            reward_breakdown: 奖励分解字典（用于分析）
        """
        rewards = {}

        # ===== 1. 最终奖励（最重要） =====
        final_entity = trajectory[-1][0]
        rewards['final'] = 1.0 if final_entity == target_entity else 0.0

        # ===== 2. 规则一致性奖励（中间） =====
        # 提取路径中的关系序列
        path_relations = tuple([step[1] for step in trajectory[1:] if step[1] is not None])

        # 查找匹配的规则
        matched_rules = self._find_matching_rules(path_relations, query_relation)

        if matched_rules:
            # 使用规则置信度作为奖励（这里简化为固定值，实际可以从规则挖掘中获取）
            # 如果有多条匹配规则，取最大置信度
            rewards['rule_consistency'] = 0.5  # 简化版本，实际应该从规则置信度获取
        else:
            rewards['rule_consistency'] = 0.0

        # ===== 3. 接近目标奖励（中间，仅失败时启用） =====
        if rewards['final'] == 0.0:
            # 只有在没到达目标时才计算接近奖励
            getting_closer_reward = self._compute_getting_closer_reward(trajectory, target_entity)
            rewards['getting_closer'] = getting_closer_reward
        else:
            rewards['getting_closer'] = 0.0

        # ===== 加权求和 =====
        total_reward = (
            rewards['final'] +
            self.alpha * (
                rewards['rule_consistency'] +
                (1 - rewards['final']) * rewards['getting_closer']
            )
        )

        return total_reward, rewards

    def _find_matching_rules(self, path_relations, query_relation):
        """
        查找与路径匹配的规则

        Args:
            path_relations: 路径关系序列 (rel_1, rel_2, ...)
            query_relation: 查询关系

        Returns:
            matched_rules: 匹配的规则列表 [(rule_id, rule_head), ...]
        """
        matched_rules = []

        # 检查完整路径是否匹配某条规则
        if path_relations in self.rule_body_to_head:
            for rule_id, rule_head in self.rule_body_to_head[path_relations]:
                # 检查规则头是否匹配查询关系
                if rule_head == query_relation:
                    matched_rules.append((rule_id, rule_head))

        # 也可以检查路径的子序列（部分匹配）
        # 这里为了简化，只检查完整匹配

        return matched_rules

    def _compute_getting_closer_reward(self, trajectory, target_entity):
        """
        计算接近目标的奖励

        衡量路径是否在嵌入空间中逐步接近目标实体

        Args:
            trajectory: 路径 [(entity_0, None), (entity_1, rel_0), ...]
            target_entity: 目标实体 ID

        Returns:
            getting_closer_reward: 归一化的接近奖励 [0, 1]
        """
        if len(trajectory) < 2:
            return 0.0

        # 获取目标实体嵌入
        target_emb = self.rule_model.entity_embedding.weight[target_entity]

        # 计算起点到目标的距离
        start_entity = trajectory[0][0]
        start_emb = self.rule_model.entity_embedding.weight[start_entity]
        dist_start = self._embedding_distance(start_emb, target_emb)

        # 计算每一步的改进
        getting_closer_sum = 0.0
        for i in range(1, len(trajectory)):
            curr_entity = trajectory[i][0]
            prev_entity = trajectory[i-1][0]

            curr_emb = self.rule_model.entity_embedding.weight[curr_entity]
            prev_emb = self.rule_model.entity_embedding.weight[prev_entity]

            dist_curr = self._embedding_distance(curr_emb, target_emb)
            dist_prev = self._embedding_distance(prev_emb, target_emb)

            # 如果距离减小，给予正奖励
            improvement = dist_prev - dist_curr
            getting_closer_sum += max(0.0, improvement.item())

        # 归一化到 [0, 1]
        if dist_start.item() > 1e-9:
            normalized_reward = min(1.0, getting_closer_sum / dist_start.item())
        else:
            normalized_reward = 0.0

        return normalized_reward

    def _embedding_distance(self, emb1, emb2):
        """
        计算两个嵌入之间的距离

        Args:
            emb1: 嵌入1 [dim]
            emb2: 嵌入2 [dim]

        Returns:
            distance: 欧氏距离 (标量)
        """
        return torch.norm(emb1 - emb2, p=2)
