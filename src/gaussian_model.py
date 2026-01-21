"""
Gaussian-RulE: Core Model Implementation
核心模型实现 - 高斯分布嵌入的规则增强知识图谱推理

创新点：
1. 实体表示为高斯分布 N(μ, Σ)
2. 动态置信度 w_i(h) 基于 KL 散度
3. 规则推理通过概率分布变换
4. 处处可微，无梯度消失问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from gaussian_layers import KLDivergence, GaussianTransform, ProductOfGaussians, KLRegularization
from torch.nn.utils.rnn import pad_sequence


class GaussianRulE(nn.Module):
    """
    高斯分布嵌入的规则增强知识图谱推理模型
    """

    def __init__(self, graph, args):
        super(GaussianRulE, self).__init__()

        self.graph = graph
        self.device = args.device if hasattr(args, 'device') else torch.device('cpu')
        self.num_entities = graph.entity_size
        self.num_relations = graph.relation_size
        self.padding_index = graph.relation_size

        self.hidden_dim = args.hidden_dim
        self.epsilon = args.epsilon if hasattr(args, 'epsilon') else 1e-8

        # ===== 实体嵌入（高斯分布）=====
        # 均值 μ
        self.entity_mean_emb = nn.Embedding(self.num_entities, self.hidden_dim)
        nn.init.uniform_(
            self.entity_mean_emb.weight,
            a=-1.0,
            b=1.0
        )

        # log方差 log(σ²)
        self.entity_logvar_emb = nn.Embedding(self.num_entities, self.hidden_dim)
        init_logvar = args.init_logvar if hasattr(args, 'init_logvar') else -1.0
        nn.init.constant_(
            self.entity_logvar_emb.weight,
            init_logvar  # log(0.37) ≈ -1.0
        )

        # ===== 关系嵌入（高斯分布，支持双向）=====
        # 均值
        self.relation_mean_emb = nn.Embedding(
            self.num_relations * 2 + 1,
            self.hidden_dim,
            padding_idx=self.padding_index
        )
        nn.init.uniform_(
            self.relation_mean_emb.weight,
            a=-0.5,
            b=0.5
        )

        # log方差
        self.relation_logvar_emb = nn.Embedding(
            self.num_relations * 2 + 1,
            self.hidden_dim,
            padding_idx=self.padding_index
        )
        nn.init.constant_(
            self.relation_logvar_emb.weight,
            -2.0  # log(0.14) ≈ -2.0，关系的不确定性较小
        )

        # ===== Margin 参数 =====
        self.gamma_fact = nn.Parameter(
            torch.Tensor([args.gamma_fact]),
            requires_grad=False
        )

        self.gamma_rule = nn.Parameter(
            torch.Tensor([args.gamma_rule]),
            requires_grad=False
        )

        # ===== 高斯操作模块 =====
        self.kl_divergence = KLDivergence(epsilon=self.epsilon)
        self.gaussian_transform = GaussianTransform()
        self.product_of_gaussians = ProductOfGaussians(epsilon=self.epsilon)
        self.kl_regularization = KLRegularization(
            lambda_kl=args.lambda_kl if hasattr(args, 'lambda_kl') else 0.001,
            target_logvar=args.target_logvar if hasattr(args, 'target_logvar') else 0.0
        )

        # ===== 规则相关 =====
        self.relation2rules = None
        self.rule_features = None
        self.rule_masks = None

        logging.info(f'GaussianRulE Model initialized with hidden_dim={self.hidden_dim}')

    def set_rules(self, rules):
        """
        设置规则
        Args:
            rules: 规则列表，每个规则格式为 [rule_id, rule_head, body_1, body_2, ...]
        """
        logging.info(f'Setting {len(rules)} rules')
        self.num_rules = len(rules)
        self.max_length = max([len(rule[2:]) for rule in rules])

        # 构建 relation → rules 的映射
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

    def transform(self, gaussian, relation_id):
        """
        通过关系变换高斯分布

        Args:
            gaussian: (mu, logvar)
            relation_id: 关系ID

        Returns:
            gaussian_out: (mu_out, logvar_out)
        """
        mu, logvar = gaussian
        mu_r = self.relation_mean_emb(relation_id)
        logvar_r = self.relation_logvar_emb(relation_id)

        return self.gaussian_transform(gaussian, mu_r, logvar_r)

    def compute_KGE(self, sample, mode='single'):
        """
        计算 KGE 分数（基于高斯嵌入）

        Args:
            sample: 三元组样本
            mode: 'single', 'head-batch', 'tail-batch'

        Returns:
            score: KGE 分数（越大越好）
        """
        # 检查 mode 类型
        if not isinstance(mode, str):
            raise TypeError(f'mode must be str, got {type(mode)}: {mode}')

        if mode == 'single':
            head = sample[:, 0]
            relation = sample[:, 1]
            tail = sample[:, 2]

            mu_h = self.entity_mean_emb(head).unsqueeze(1)
            logvar_h = self.entity_logvar_emb(head).unsqueeze(1)
            mu_t = self.entity_mean_emb(tail).unsqueeze(1)
            logvar_t = self.entity_logvar_emb(tail).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            mu_h = self.entity_mean_emb(head_part[:, 0]).unsqueeze(1)
            logvar_h = self.entity_logvar_emb(head_part[:, 0]).unsqueeze(1)
            relation = head_part[:, 1]

            mu_t = self.entity_mean_emb(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            logvar_t = self.entity_logvar_emb(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            mu_h = self.entity_mean_emb(head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            logvar_h = self.entity_logvar_emb(head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            relation = tail_part[:, 1]

            mu_t = self.entity_mean_emb(tail_part[:, 2]).unsqueeze(1)
            logvar_t = self.entity_logvar_emb(tail_part[:, 2]).unsqueeze(1)

        else:
            raise ValueError(f'Mode {mode} not supported in compute_KGE')

        # 通过关系变换头实体
        mu_r = self.relation_mean_emb(relation).unsqueeze(1)
        logvar_r = self.relation_logvar_emb(relation).unsqueeze(1)

        mu_pred = mu_h + mu_r
        logvar_pred = torch.logaddexp(logvar_h, logvar_r)

        # 计算 KL 散度
        kl = self.kl_divergence((mu_pred, logvar_pred), (mu_t, logvar_t))

        # 分数 = gamma - KL（KL 越小分数越高）
        score = self.gamma_fact - kl

        return score

    def compute_ruleE(self, sample, mode='single'):
        """
        计算规则分数（训练阶段）

        Args:
            sample: 规则样本
            mode: 'single' 或 'batch'

        Returns:
            score: 规则分数（越大越好）
        """
        if mode == 'single':
            rule, mask = sample
            rule = rule.unsqueeze(1)  # [batch, 1, rule_length]
        elif mode == 'batch':
            pos_part, mask, neg_idx, neg_part = sample
            batch_size, negative_sample_size = neg_idx.size(0), neg_idx.size(1)

            # 扩展正样本
            pos_part = pos_part.unsqueeze(dim=1).repeat(1, negative_sample_size, 1)

            # 在指定位置替换为负样本
            neg_idx = neg_idx.unsqueeze(dim=2) + 2  # +2 因为前面有 rule_id 和 head
            neg_part = neg_part.unsqueeze(dim=2)
            rule = pos_part.scatter(2, neg_idx, neg_part)
        else:
            raise ValueError(f'Mode {mode} not supported')

        # 获取规则体和规则头
        rule_body = rule[:, :, 2:]  # [batch, num_samples, body_length]
        rule_head = rule[:, :, 1]   # [batch, num_samples]

        # 处理双向关系
        relations_flag = torch.pow(-1, rule_body // self.num_relations).unsqueeze(-1)
        rule_body_relations = rule_body % self.num_relations
        rule_body_relations = torch.where(
            rule_body == self.num_relations * 2,
            self.padding_index,
            rule_body_relations
        )

        # 嵌入规则体（累加均值）
        body_mean = self.relation_mean_emb(rule_body_relations) * relations_flag
        body_logvar = self.relation_logvar_emb(rule_body_relations)

        # 应用 mask
        cal_mask = mask.unsqueeze(1).unsqueeze(-1)  # [batch, 1, body_length, 1]

        # 累加均值
        body_mean_sum = (body_mean * cal_mask).sum(-2)  # [batch, num_samples, hidden_dim]

        # 累加方差（在log域）
        body_logvar_masked = torch.where(
            cal_mask.squeeze(-1),
            body_logvar,
            torch.full_like(body_logvar, -float('inf'))
        )
        # 使用 logsumexp 来累加（log域的加法）
        body_logvar_sum = torch.logsumexp(body_logvar_masked, dim=-2)

        # 嵌入规则头
        head_relations_flag = torch.pow(-1, rule_head // self.num_relations).unsqueeze(-1)
        head_mean = self.relation_mean_emb(rule_head % self.num_relations) * head_relations_flag
        head_logvar = self.relation_logvar_emb(rule_head % self.num_relations)

        # 计算 KL 散度
        kl = self.kl_divergence((body_mean_sum, body_logvar_sum), (head_mean, head_logvar))

        # 分数 = gamma - KL
        score = self.gamma_rule - kl

        return score

    def compute_dynamic_confidence(self, h, rule_head, rule_body):
        """
        【核心创新】计算动态置信度 w_i(h)

        问题：这条规则对实体 h 是否适用？

        思路：
            1. 从 h 出发，沿规则体推导 → Gaussian_Body
            2. 从 h 出发，沿规则头推导 → Gaussian_Head
            3. 计算两个分布的相似度 → w_i

        数学：
            w_i = exp(-KL(Body || Head))

        Args:
            h: 头实体ID
            rule_head: 规则头关系
            rule_body: 规则体关系列表

        Returns:
            w_i: 动态置信度 [0, 1]
        """
        # 获取头实体的分布
        mu_h = self.entity_mean_emb(h)
        logvar_h = self.entity_logvar_emb(h)
        gaussian_h = (mu_h, logvar_h)

        # 计算 Gaussian_Body（沿规则体变换）
        gaussian_body = gaussian_h
        for relation in rule_body:
            if relation == self.padding_index:
                break
            gaussian_body = self.transform(gaussian_body, relation)

        # 计算 Gaussian_Head（沿规则头变换）
        gaussian_head = self.transform(gaussian_h, rule_head)

        # 计算 KL 散度
        kl = self.kl_divergence(gaussian_body, gaussian_head)

        # 转换为置信度 [0, 1]
        w_i = torch.exp(-kl)

        return w_i

    def forward(self, all_h, all_r, edges_to_remove=None):
        """
        推理阶段的前向传播（使用动态置信度）

        Args:
            all_h: 头实体ID [batch]
            all_r: 目标关系ID [batch]
            edges_to_remove: 需要过滤的边

        Returns:
            score: [batch, num_entities] 规则增强分数
            mask: [batch, num_entities] 有效候选mask
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

        # 获取所有候选实体的分布
        mu_all = self.entity_mean_emb.weight  # [N, d]
        logvar_all = self.entity_logvar_emb.weight
        gaussian_all = (mu_all, logvar_all)

        # 对每条规则计算分数
        for rule_idx, (rule_head, rule_body) in self.relation2rules[query_r]:
            # 对 batch 中的每个头实体
            for i, h in enumerate(all_h):
                # 计算动态置信度 w_i(h)
                w_i = self.compute_dynamic_confidence(
                    h,
                    torch.tensor(rule_head, device=device),
                    torch.tensor(rule_body, device=device)
                )

                # 如果 w_i 太小，跳过（加速）
                if w_i < 0.01:
                    continue

                # 计算 Gaussian_Body
                mu_h = self.entity_mean_emb(h)
                logvar_h = self.entity_logvar_emb(h)
                gaussian_body = (mu_h, logvar_h)

                for relation in rule_body:
                    if relation == self.padding_index:
                        break
                    gaussian_body = self.transform(
                        gaussian_body,
                        torch.tensor(relation, device=device)
                    )

                # 计算 grounding 得分（对所有候选实体）
                kl_all = self.kl_divergence.forward_broadcast(
                    (gaussian_body[0].unsqueeze(0), gaussian_body[1].unsqueeze(0)),
                    gaussian_all
                )  # [1, N]

                # 转换为分数（归一化）
                v_rule = torch.exp(-kl_all.squeeze(0) / self.gamma_rule)  # [N]

                # 融合得分
                total_score[i] += w_i * v_rule

        mask = torch.ones_like(total_score).bool()
        return total_score, mask
