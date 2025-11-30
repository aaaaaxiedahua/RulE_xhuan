"""
KGReasoningEnv - 知识图谱推理环境

提供强化学习的环境接口，支持：
- 状态转移（沿着 KG 边移动）
- 动作掩码（只允许有效动作）
- 奖励计算
"""

import torch
import numpy as np
import logging


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
        top_epsilon: 每步保留的候选动作数量（剪枝），<=0 表示不剪枝
    """

    def __init__(self, graph, rule_model, state_encoder, reward_calculator, max_steps=5, top_epsilon=32):
        self.graph = graph
        self.rule_model = rule_model
        self.state_encoder = state_encoder
        self.reward_calculator = reward_calculator
        self.max_steps = max_steps
        self.top_epsilon = top_epsilon if top_epsilon is not None else -1

        # Episode 状态
        self.query_head = None
        self.query_rel = None
        self.query_tail = None
        self.current_entity = None
        self.path_history = []
        self.trajectory = []
        self.step_count = 0

        # 构建邻接表（加速查询）
        self._build_adjacency()
        self._build_rule_whitelist()

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

    def _build_rule_whitelist(self):
        """
        构建查询关系到规则体关系的映射，用于动作掩码
        """
        self.rule_whitelist = {}
        rules = getattr(self.rule_model, 'rules', [])
        max_relation_id = self.graph.relation_size * 2
        for rule in rules:
            if len(rule) <= 2:
                continue
            head_rel = rule[1] % self.graph.relation_size
            body_rels = [
                rel for rel in rule[2:]
                if rel < max_relation_id
            ]
            if not body_rels:
                continue
            if head_rel not in self.rule_whitelist:
                self.rule_whitelist[head_rel] = set()
            self.rule_whitelist[head_rel].update(body_rels)

    def reset(self, query):
        """
        重置环境，开始新的 episode

        Args:
            query: (head, relation, tail) 查询三元组

        Returns:
            state: 初始状态编码 [state_dim]
        """
        self.query_head = query[0]
        self.query_rel = query[1]
        self.query_tail = query[2]

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
        base_rel_idx = action % self.graph.relation_size
        # Inverse relations are encoded by offsetting the ID by relation_size; reuse
        # the base embedding but flip its sign to keep direction information.
        relation_flag = -1.0 if action >= self.graph.relation_size else 1.0
        action_rel_emb = self.rule_model.relation_embedding.weight[base_rel_idx] * relation_flag
        if torch.isnan(action_rel_emb).any() or torch.isinf(action_rel_emb).any():
            logging.error('Invalid relation embedding encountered (action=%d)', action)
            raise ValueError('action relation embedding contains NaN/Inf')
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
        if torch.isnan(next_state).any() or torch.isinf(next_state).any():
            logging.error('Invalid encoded state encountered during step; trajectory=%s', self.trajectory)
            raise ValueError('state encoder output contains NaN/Inf')

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

        # 2. 获取规则白名单
        base_query_rel = self.query_rel % self.graph.relation_size
        rule_rels = self.rule_whitelist.get(base_query_rel)

        if rule_rels:
            valid_rels = outgoing_rels & rule_rels
        else:
            valid_rels = outgoing_rels

        if len(valid_rels) == 0:
            valid_rels = outgoing_rels

        valid_rels = self._prune_by_top_epsilon(valid_rels)

        # 3. 生成 mask
        max_rel = self.graph.relation_size * 2
        mask = torch.zeros(max_rel, dtype=torch.bool, device=device)
        for rel in valid_rels:
            if rel < max_rel:
                mask[rel] = True

        # 如果没有有效动作，允许所有出边（避免卡死）
        if mask.sum().item() == 0 and len(outgoing_rels) > 0:
            for rel in outgoing_rels:
                if rel < max_rel:
                    mask[rel] = True

        # 如果仍没有动作（孤立节点），允许所有关系，防止策略网崩溃
        if mask.sum().item() == 0:
            mask[:] = True

        return mask

    def _prune_by_top_epsilon(self, relations):
        """
        根据预训练嵌入得分，从候选动作中选出 top-epsilon 个
        """
        if (
            self.top_epsilon is None
            or self.top_epsilon <= 0
            or len(relations) <= self.top_epsilon
        ):
            return relations

        rel_list = list(relations)
        device = self.rule_model.relation_embedding.weight.device
        base_ids = torch.tensor(
            [rel % self.graph.relation_size for rel in rel_list],
            device=device,
            dtype=torch.long
        )
        rel_embs = self.rule_model.relation_embedding.weight[base_ids]
        direction = torch.tensor(
            [-1.0 if rel >= self.graph.relation_size else 1.0 for rel in rel_list],
            device=device
        )
        rel_embs = rel_embs * direction.unsqueeze(-1)
        query_emb = self.rule_model.relation_embedding.weight[self.query_rel % self.graph.relation_size]
        scores = rel_embs @ query_emb
        top_k = min(self.top_epsilon, len(rel_list))
        top_indices = torch.topk(scores, k=top_k).indices.tolist()
        return {rel_list[idx] for idx in top_indices}

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
            'trajectory': self.trajectory
        }
