"""
KGReasoningEnv - 知识图谱推理环境

提供强化学习的环境接口，支持：
- 状态转移（沿着 KG 边移动）
- 动作掩码（只允许有效动作）
- 奖励计算
"""

import torch
import numpy as np


class KGReasoningEnv:
    """
    知识图谱推理环境

    提供标准的 RL 环境接口：reset(), step(), get_action_mask()

    Args:
        graph: KnowledgeGraph 对象
        rule_model: 预训练的 RulE 模型
        state_encoder: StateEncoder 实例
        reward_calculator: RewardCalculator 实例
        max_steps: 每个 episode 的最大步数
    """

    def __init__(self, graph, rule_model, state_encoder, reward_calculator, max_steps=5):
        self.graph = graph
        self.rule_model = rule_model
        self.state_encoder = state_encoder
        self.reward_calculator = reward_calculator
        self.max_steps = max_steps

        # Episode 状态
        self.query_head = None
        self.query_rel = None
        self.query_tail = None
        self.current_entity = None
        self.selected_rules = None
        self.path_history = []
        self.trajectory = []
        self.step_count = 0

        # 构建邻接表（加速查询）
        self._build_adjacency()

    def _build_adjacency(self):
        """
        构建邻接表：entity → [(relation, neighbor_entity), ...]

        用于快速查询某个实体的所有出边
        """
        self.adjacency = {}  # {entity_id: [(relation, neighbor), ...]}

        for h, r, t in self.graph.train_facts:
            if h not in self.adjacency:
                self.adjacency[h] = []
            self.adjacency[h].append((r, t))

            # 添加逆关系
            inv_r = r + self.graph.relation_size
            if t not in self.adjacency:
                self.adjacency[t] = []
            self.adjacency[t].append((inv_r, h))

    def reset(self, query, selected_rules):
        """
        重置环境，开始新的 episode

        Args:
            query: (head, relation, tail) 查询三元组
            selected_rules: 选中的规则 ID 列表 [top_k]

        Returns:
            state: 初始状态编码 [state_dim]
        """
        self.query_head = query[0]
        self.query_rel = query[1]
        self.query_tail = query[2]
        self.selected_rules = selected_rules

        # 初始化当前位置
        self.current_entity = self.query_head

        # 初始化轨迹
        self.path_history = []
        self.trajectory = [(self.current_entity, None)]

        # 步数
        self.step_count = 0

        # 编码初始状态
        state = self._encode_state()

        return state

    def step(self, action):
        """
        执行动作，转移到下一个状态

        Args:
            action: 选择的关系 ID (int 或 tensor)

        Returns:
            next_state: 下一个状态编码 [state_dim]
            reward: 奖励 (float)
            done: 是否结束 (bool)
            info: 额外信息 (dict)
        """
        # 转换 action 为 int
        if isinstance(action, torch.Tensor):
            action = action.item()

        # 1. 执行动作：沿着关系边移动
        neighbors = self._get_neighbors(self.current_entity, action)

        if len(neighbors) == 0:
            # 死胡同：没有可走的边
            next_state = self._encode_state()
            reward = -0.1  # 小惩罚
            done = True
            info = {'reason': 'dead_end', 'success': False}
            return next_state, reward, done, info

        # 2. 随机选择一个邻居（如果有多个）
        next_entity = np.random.choice(neighbors)

        # 3. 更新路径历史
        current_entity_emb = self.rule_model.entity_embedding.weight[self.current_entity]
        action_rel_emb = self.rule_model.relation_embedding.weight[action]
        self.path_history.append(
            torch.cat([current_entity_emb, action_rel_emb], dim=-1)
        )

        # 4. 更新轨迹
        self.trajectory.append((next_entity, action))

        # 5. 更新位置
        self.current_entity = next_entity
        self.step_count += 1

        # 6. 编码新状态
        next_state = self._encode_state()

        # 7. 判断是否结束
        done = (self.step_count >= self.max_steps) or (next_entity == self.query_tail)

        # 8. 计算奖励
        if done:
            reward, breakdown = self.reward_calculator.compute_reward(
                self.trajectory,
                self.query_tail,
                self.query_rel
            )
            info = {
                'reason': 'reached_target' if next_entity == self.query_tail else 'max_steps',
                'success': next_entity == self.query_tail,
                'reward_breakdown': breakdown,
                'path_length': len(self.trajectory) - 1
            }
        else:
            # 中间步不给奖励（延迟奖励）
            reward = 0.0
            info = {'reason': 'ongoing', 'success': False}

        return next_state, reward, done, info

    def get_action_mask(self):
        """
        获取有效动作掩码

        只允许满足以下条件的动作：
        1. 当前实体有该关系的出边（KG 约束）
        2. 该关系出现在选中规则的规则体中（规则约束）

        Returns:
            mask: [num_relations] bool tensor，True 表示该动作可用
        """
        device = self.rule_model.entity_embedding.weight.device

        # 1. 获取当前实体的所有出边关系
        outgoing_rels = set()
        if self.current_entity in self.adjacency:
            for rel, _ in self.adjacency[self.current_entity]:
                outgoing_rels.add(rel)

        # 2. 获取选中规则体中的关系
        rule_rels = set()
        for rule_id in self.selected_rules:
            rule_id_int = rule_id.item() if isinstance(rule_id, torch.Tensor) else rule_id
            if rule_id_int < len(self.rule_model.rules):
                rule = self.rule_model.rules[rule_id_int]
                # 规则格式: [rule_id, rule_head, rel_1, rel_2, ...]
                if len(rule) > 2:
                    rule_rels.update(rule[2:])

        # 3. 计算交集：KG 邻居 ∩ 规则关系
        valid_rels = outgoing_rels & rule_rels

        # 4. 生成 mask
        mask = torch.zeros(self.graph.relation_size * 2, dtype=torch.bool, device=device)
        for rel in valid_rels:
            if rel < self.graph.relation_size * 2:
                mask[rel] = True

        # 如果没有有效动作，允许所有出边（避免卡死）
        if mask.sum().item() == 0 and len(outgoing_rels) > 0:
            for rel in outgoing_rels:
                if rel < self.graph.relation_size * 2:
                    mask[rel] = True

        return mask

    def _encode_state(self):
        """
        编码当前状态

        Returns:
            state: 状态编码 [state_dim]
        """
        # 当前实体嵌入
        current_entity_emb = self.rule_model.entity_embedding.weight[self.current_entity]

        # 查询关系嵌入
        query_rel_emb = self.rule_model.relation_embedding.weight[self.query_rel]

        # 规则上下文：选中规则的嵌入
        if self.selected_rules is not None and len(self.selected_rules) > 0:
            rule_context = self.rule_model.rule_emb[self.selected_rules]  # [top_k, rule_dim]
        else:
            rule_context = None

        # 路径历史
        if len(self.path_history) > 0:
            path_history = torch.stack(self.path_history)  # [num_steps, entity_dim+rel_dim]
        else:
            path_history = None

        # 使用 StateEncoder 编码
        state = self.state_encoder(
            current_entity=current_entity_emb,
            query_rel=query_rel_emb,
            rule_context=rule_context,
            path_history=path_history
        )

        return state

    def _get_neighbors(self, entity, relation):
        """
        获取实体沿着某个关系的邻居

        Args:
            entity: 实体 ID
            relation: 关系 ID

        Returns:
            neighbors: 邻居实体 ID 列表
        """
        neighbors = []
        if entity in self.adjacency:
            for rel, neighbor in self.adjacency[entity]:
                if rel == relation:
                    neighbors.append(neighbor)
        return neighbors

    def get_current_state(self):
        """
        获取当前状态信息（用于调试）

        Returns:
            state_info: 状态信息字典
        """
        return {
            'current_entity': self.current_entity,
            'query': (self.query_head, self.query_rel, self.query_tail),
            'step_count': self.step_count,
            'trajectory': self.trajectory,
            'selected_rules': self.selected_rules
        }
