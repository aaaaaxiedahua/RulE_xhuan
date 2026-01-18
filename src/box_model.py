"""
Box-RulE: Core Model Implementation
核心模型实现 - 第1部分：初始化和基础函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from box_layers import BoxTransform, IntersectionVolume, BoxVolume, VolumeRegularization
from torch.nn.utils.rnn import pad_sequence


class BoxRulE(nn.Module):
    """
    Box-RulE模型

    核心创新：
    1. 盒嵌入表示（点 → 盒子）
    2. 动态规则置信度 w_i(h)
    3. 几何逻辑推理
    """

    def __init__(self, graph, args):
        super(BoxRulE, self).__init__()

        self.graph = graph
        self.device = args.device if hasattr(args, 'device') else torch.device('cpu')
        self.num_entities = graph.entity_size
        self.num_relations = graph.relation_size
        self.padding_index = graph.relation_size

        self.hidden_dim = args.hidden_dim
        self.epsilon = args.epsilon if hasattr(args, 'epsilon') else 1e-8

        # ===== 盒嵌入参数 =====
        # 实体中心
        self.entity_center_emb = nn.Embedding(self.num_entities, self.hidden_dim)
        nn.init.uniform_(
            self.entity_center_emb.weight,
            a=-1.0,
            b=1.0
        )

        # 实体宽度
        self.entity_width_emb = nn.Embedding(self.num_entities, self.hidden_dim)
        init_width = args.init_width if hasattr(args, 'init_width') else 0.5
        nn.init.constant_(
            self.entity_width_emb.weight,
            init_width
        )

        # 关系平移（支持双向关系）
        self.relation_trans_emb = nn.Embedding(self.num_relations * 2 + 1, self.hidden_dim,
                                                padding_idx=self.padding_index)
        nn.init.uniform_(
            self.relation_trans_emb.weight,
            a=-0.5,
            b=0.5
        )

        # 关系缩放（支持双向关系）
        self.relation_scale_emb = nn.Embedding(self.num_relations * 2 + 1, self.hidden_dim,
                                                padding_idx=self.padding_index)
        nn.init.constant_(
            self.relation_scale_emb.weight,
            0.0
        )

        # ===== Margin参数 =====
        self.gamma_fact = nn.Parameter(
            torch.Tensor([args.gamma_fact]),
            requires_grad=False
        )

        self.gamma_rule = nn.Parameter(
            torch.Tensor([args.gamma_rule]),
            requires_grad=False
        )

        # ===== 盒子操作模块 =====
        self.box_transform = BoxTransform()
        self.intersection_volume = IntersectionVolume(epsilon=self.epsilon)
        self.box_volume = BoxVolume(epsilon=self.epsilon)
        self.volume_regularization = VolumeRegularization(
            lambda_vol=args.lambda_vol if hasattr(args, 'lambda_vol') else 0.001,
            target_log_vol=0.0,
            epsilon=self.epsilon
        )

        # ===== 规则相关 =====
        self.relation2rules = None
        self.rule_features = None
        self.rule_masks = None

        logging.info(f'BoxRulE Model initialized with hidden_dim={self.hidden_dim}')

    def set_rules(self, rules):
        """
        设置规则
        Args:
            rules: 规则列表，每个规则格式为 [rule_id, rule_head, body_1, body_2, ...]
        """
        logging.info(f'Setting {len(rules)} rules')
        self.num_rules = len(rules)
        self.max_length = max([len(rule[2:]) for rule in rules])

        # 构建relation到rules的映射
        self.relation2rules = [[] for _ in range(self.num_relations * 2)]
        for rule in rules:
            relation = rule[1]
            self.relation2rules[relation].append([rule[0], (rule[1], rule[2:])])

        # 格式化规则
        self.rule_features = []
        rule_masks = []
        for rule in rules:
            rule_ = rule + [self.padding_index for _ in range(self.max_length - len(rule[2:]))]
            self.rule_features.append(rule_)
            rule_mask = torch.ones(len(rule) - 2).bool()
            rule_masks.append(rule_mask)

        self.rule_masks = pad_sequence(rule_masks, batch_first=True, padding_value=False)
        self.rule_features = torch.tensor(self.rule_features, dtype=torch.long)

    def transform(self, box, relation_id):
        """
        通过关系变换盒子
        Args:
            box: (center, width)
            relation_id: 关系ID
        Returns:
            box_out: (center_out, width_out)
        """
        center, width = box
        trans = self.relation_trans_emb(relation_id)
        scale = self.relation_scale_emb(relation_id)
        return self.box_transform((center, width), trans, scale)

    def compute_KGE(self, sample, mode='single'):
        """
        计算KGE分数（基于盒嵌入）
        Args:
            sample: 三元组样本
            mode: 'single', 'head-batch', 'tail-batch'
        Returns:
            score: KGE分数
        """
        # Debug: 打印mode类型和值
        if not isinstance(mode, str):
            raise TypeError(f'mode must be str, got {type(mode)}: {mode}')

        if mode == 'single':
            head = sample[:, 0]
            relation = sample[:, 1]
            tail = sample[:, 2]

            center_h = self.entity_center_emb(head).unsqueeze(1)
            width_h = self.entity_width_emb(head).unsqueeze(1)
            center_t = self.entity_center_emb(tail).unsqueeze(1)
            width_t = self.entity_width_emb(tail).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            center_h = self.entity_center_emb(head_part[:, 0]).unsqueeze(1)
            width_h = self.entity_width_emb(head_part[:, 0]).unsqueeze(1)
            relation = head_part[:, 1]

            center_t = self.entity_center_emb(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            width_t = self.entity_width_emb(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            center_h = self.entity_center_emb(head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            width_h = self.entity_width_emb(head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            relation = tail_part[:, 1]

            center_t = self.entity_center_emb(tail_part[:, 2]).unsqueeze(1)
            width_t = self.entity_width_emb(tail_part[:, 2]).unsqueeze(1)

        else:
            raise ValueError(f'Mode {mode} not supported in compute_KGE')

        # 通过关系变换头实体盒子
        trans = self.relation_trans_emb(relation).unsqueeze(1)
        scale = self.relation_scale_emb(relation).unsqueeze(1)
        box_pred = self.box_transform((center_h, width_h), trans, scale)

        # 计算交集体积
        vol_inter = self.intersection_volume(box_pred, (center_t, width_t))

        # 计算分数
        score = self.gamma_fact.item() - (self.gamma_fact.item() - vol_inter)

        return score

    def compute_ruleE(self, sample, mode='single'):
        """
        计算规则分数（训练阶段，与原RulE兼容）
        Args:
            sample: 规则样本
            mode: 'single' 或 'batch'
        Returns:
            score: 规则分数
        """
        if mode == 'single':
            rule, mask = sample
            rule = rule.unsqueeze(1)
        elif mode == 'batch':
            pos_part, mask, neg_idx, neg_part = sample
            batch_size, negative_sample_size = neg_idx.size(0), neg_idx.size(1)
            pos_part = pos_part.unsqueeze(dim=1).repeat(1, negative_sample_size, 1)
            neg_idx = neg_idx.unsqueeze(dim=2) + 1
            neg_part = neg_part.unsqueeze(dim=2)
            rule = pos_part.scatter(2, neg_idx, neg_part)

        # 获取规则体和规则头
        rule_body = rule[:, :, 2:]
        rule_head = rule[:, :, 1]

        # 处理规则体关系
        relations_flag = torch.pow(-1, rule_body // self.num_relations).unsqueeze(-1)
        rule_body_relations = rule_body % self.num_relations
        rule_body_relations = torch.where(
            rule_body == self.num_relations * 2,
            self.padding_index,
            rule_body_relations
        )

        # 嵌入规则体
        body_emb = self.relation_trans_emb(rule_body_relations) * relations_flag
        cal_mask = mask.unsqueeze(1).unsqueeze(-1)
        body_sum = (body_emb * cal_mask).sum(-2)

        # 嵌入规则头
        head_relations_flag = torch.pow(-1, rule_head // self.num_relations).unsqueeze(-1)
        head_emb = self.relation_trans_emb(rule_head % self.num_relations) * head_relations_flag

        # 计算距离
        distance = torch.norm(body_sum - head_emb, p=2, dim=-1)
        score = self.gamma_rule.item() - distance

        return score

    def compute_dynamic_confidence(self, h, rule_head, rule_body):
        """
        计算动态置信度 w_i(h) - 方案2核心
        Args:
            h: 头实体ID
            rule_head: 规则头关系
            rule_body: 规则体关系列表
        Returns:
            w_i: 动态置信度 [0, 1]
        """
        # 获取头实体盒子
        center_h = self.entity_center_emb(h)
        width_h = self.entity_width_emb(h)
        box_h = (center_h, width_h)

        # 计算Box_Body（规则体推导）
        box_current = box_h
        for relation in rule_body:
            if relation == self.padding_index:
                break
            box_current = self.transform(box_current, relation)
        box_body = box_current

        # 计算Box_Head（规则头推导）
        box_head = self.transform(box_h, rule_head)

        # 计算包含率
        vol_inter = self.intersection_volume(box_body, box_head)
        vol_body = self.box_volume(box_body)
        w_i = vol_inter / (vol_body + self.epsilon)

        return w_i

    def forward(self, all_h, all_r, edges_to_remove=None):
        """
        推理阶段的前向传播（使用动态置信度w_i）
        Args:
            all_h: 头实体ID
            all_r: 目标关系ID
            edges_to_remove: 需要过滤的边
        Returns:
            score: 规则增强分数
            mask: 有效候选mask
        """
        query_r = all_r[0].item()
        device = all_r.device

        # 如果没有规则，返回零分数
        if self.relation2rules is None or len(self.relation2rules[query_r]) == 0:
            mask = torch.ones(all_h.size(0), self.num_entities, device=device).bool()
            score = torch.zeros(all_h.size(0), self.num_entities, device=device)
            return score, mask

        # 初始化分数
        total_score = torch.zeros(all_h.size(0), self.num_entities, device=device)

        # 对每条规则计算分数
        for rule_idx, (rule_head, rule_body) in self.relation2rules[query_r]:
            # 对batch中的每个头实体
            for i, h in enumerate(all_h):
                # 计算动态置信度 w_i(h)
                w_i = self.compute_dynamic_confidence(h, torch.tensor(rule_head), torch.tensor(rule_body))

                # 如果w_i太小，跳过这条规则
                if w_i < 0.01:
                    continue

                # 计算Box_Body
                center_h = self.entity_center_emb(h)
                width_h = self.entity_width_emb(h)
                box_current = (center_h, width_h)

                for relation in rule_body:
                    if relation == self.padding_index:
                        break
                    box_current = self.transform(box_current, torch.tensor(relation))

                box_body = box_current

                # 对所有候选实体计算grounding得分
                center_all = self.entity_center_emb.weight
                width_all = self.entity_width_emb.weight
                box_all = (center_all, width_all)

                # 计算交集体积
                v_rule = self.intersection_volume(box_body, box_all)

                # 融合得分
                total_score[i] += w_i * v_rule

        mask = torch.ones_like(total_score).bool()
        return total_score, mask
