"""
Box-RulE: Trainer Implementation - Part 1
训练器实现 - 第1部分：WarmupTrainer（几何预热）
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import os


def kge_collate_fn(batch):
    """
    自定义collate函数，保持mode为字符串
    """
    positive_samples = []
    negative_samples = []
    subsampling_weights = []
    mode = batch[0][3]  # 所有样本的mode相同，取第一个

    for item in batch:
        positive_samples.append(item[0])
        negative_samples.append(item[1])
        subsampling_weights.append(item[2])

    positive_samples = torch.stack(positive_samples, dim=0)
    negative_samples = torch.stack(negative_samples, dim=0)
    subsampling_weights = torch.stack(subsampling_weights, dim=0)

    return positive_samples, negative_samples, subsampling_weights, mode


class WarmupTrainer:
    """
    几何预热训练器

    目标：
    1. 初始化盒嵌入
    2. 防止盒子坍塌
    3. 只训练KGE，不加载规则
    """

    def __init__(self, model, graph, train_dataset, args):
        self.model = model
        self.graph = graph
        self.args = args
        self.device = args.device if hasattr(args, 'device') else torch.device('cpu')

        # 数据加载器
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.cpu_num if hasattr(args, 'cpu_num') else 0,
            collate_fn=kge_collate_fn
        )

        # 优化器
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate
        )

        # 移动模型到设备
        if self.device.type == 'cuda':
            self.model = self.model.cuda(self.device)

        logging.info('WarmupTrainer initialized')

    def train_step(self, batch):
        """
        单步训练
        """
        self.model.train()
        self.optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = batch

        if self.device.type == 'cuda':
            positive_sample = positive_sample.cuda(self.device)
            negative_sample = negative_sample.cuda(self.device)
            subsampling_weight = subsampling_weight.cuda(self.device)

        # 计算KGE分数
        negative_score = self.model.compute_KGE((positive_sample, negative_sample), mode)
        positive_score = self.model.compute_KGE(positive_sample, mode='single')

        # KGE损失
        negative_score = (F.softmax(negative_score * self.args.adversarial_temperature, dim=1).detach()
                         * F.logsigmoid(-negative_score)).sum(dim=1)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_loss = -(subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_loss = -(subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        loss_kge = (positive_loss + negative_loss) / 2

        # 体积正则化
        widths = self.model.entity_width_emb.weight
        loss_vol = self.model.volume_regularization(widths)

        # 总损失
        loss_total = loss_kge + loss_vol

        # DEBUG: 记录反向传播前的参数
        param_before = self.model.entity_center_emb.weight.data[0, 0].item()

        loss_total.backward()

        # DEBUG: 检查梯度
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))

        self.optimizer.step()

        # DEBUG: 记录参数更新后的值
        param_after = self.model.entity_center_emb.weight.data[0, 0].item()
        param_change = abs(param_after - param_before)

        return {
            'loss_kge': loss_kge.item(),
            'loss_vol': loss_vol.item(),
            'loss_total': loss_total.item(),
            'grad_norm': grad_norm.item(),
            'param_change': param_change
        }

    def train(self, max_steps):
        """
        训练主循环
        """
        warmup_valid_steps = getattr(self.args, 'warmup_valid_steps', 500)
        logging.info(f'Starting Warmup Training for {max_steps} steps')
        logging.info(f'Validation every {warmup_valid_steps} steps')

        step = 0
        best_mrr = 0.0
        training_logs = []

        while step < max_steps:
            for batch in self.train_dataloader:
                if step >= max_steps:
                    break

                log = self.train_step(batch)
                training_logs.append(log)

                # 每100步输出平均损失
                if step % 100 == 0 and step > 0:
                    avg_loss_kge = sum([l['loss_kge'] for l in training_logs]) / len(training_logs)
                    avg_loss_vol = sum([l['loss_vol'] for l in training_logs]) / len(training_logs)
                    avg_loss_total = sum([l['loss_total'] for l in training_logs]) / len(training_logs)
                    avg_grad_norm = sum([l['grad_norm'] for l in training_logs]) / len(training_logs)
                    avg_param_change = sum([l['param_change'] for l in training_logs]) / len(training_logs)

                    logging.info(f'[Warmup] Step {step}/{max_steps}: '
                               f'loss_kge={avg_loss_kge:.4f}, '
                               f'loss_vol={avg_loss_vol:.6f}, '
                               f'loss_total={avg_loss_total:.4f}, '
                               f'grad_norm={avg_grad_norm:.4f}, '
                               f'param_change={avg_param_change:.8f}')
                    training_logs = []

                # 定期验证
                if step % warmup_valid_steps == 0 and step > 0:
                    logging.info(f'--- Validation at step {step} ---')
                    mrr = self.evaluate(split='valid')

                    if mrr > best_mrr:
                        best_mrr = mrr
                        logging.info(f'*** New best MRR: {best_mrr:.4f} ***')

                        # 保存最优模型
                        best_checkpoint = os.path.join(self.args.save_path, 'warmup_best_model.pt')
                        os.makedirs(self.args.save_path, exist_ok=True)
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'step': step,
                            'best_mrr': best_mrr
                        }, best_checkpoint)
                        logging.info(f'Best model saved at step {step}')

                    self.model.train()

                step += 1

        logging.info('Warmup Training completed')
        logging.info(f'Best validation MRR: {best_mrr:.4f}')

        return best_mrr



    @torch.no_grad()
    def evaluate(self, split='valid'):
        """
        评估模型性能（仅使用KGE）
        """
        logging.info(f'Evaluating on {split} set...')

        from data import EvalDataset
        eval_dataset = EvalDataset(self.graph, split=split, batch_size=1)
        dataloader = DataLoader(eval_dataset, batch_size=1, num_workers=0)

        self.model.eval()

        # 收集所有预测结果
        concat_logits = []
        concat_all_h = []
        concat_all_r = []
        concat_all_t = []
        concat_flag = []

        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            flag = flag.squeeze(0)

            if self.device.type == 'cuda':
                all_h = all_h.cuda(self.device)
                all_r = all_r.cuda(self.device)
                all_t = all_t.cuda(self.device)
                flag = flag.cuda(self.device)

            # 计算KGE分数
            batch_size = all_h.size(0)
            num_entities = self.graph.entity_size

            # 对每个查询计算所有候选实体的分数
            logits = torch.zeros(batch_size, num_entities, device=self.device)
            for i in range(batch_size):
                h = all_h[i:i+1].expand(num_entities)
                r = all_r[i:i+1].expand(num_entities)
                t = torch.arange(num_entities, device=self.device)

                sample = torch.stack([h, r, t], dim=1)
                scores = self.model.compute_KGE(sample, mode='single')
                logits[i] = scores.squeeze()

            concat_logits.append(logits)
            concat_all_h.append(all_h)
            concat_all_r.append(all_r)
            concat_all_t.append(all_t)
            concat_flag.append(flag)

        # 合并所有batch的结果
        concat_logits = torch.cat(concat_logits, dim=0)
        concat_all_h = torch.cat(concat_all_h, dim=0)
        concat_all_r = torch.cat(concat_all_r, dim=0)
        concat_all_t = torch.cat(concat_all_t, dim=0)
        concat_flag = torch.cat(concat_flag, dim=0)

        # 计算排名
        ranks = []
        for k in range(concat_all_t.size(0)):
            t = concat_all_t[k]
            val = concat_logits[k, t]

            # 过滤掉训练集中的三元组
            L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
            H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
            ranks.append([L, H])

        # 计算指标（使用期望排名）
        hit1, hit3, hit10, mr, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        for (L, H) in ranks:
            for rank in range(L, H):
                if rank <= 1:
                    hit1 += 1.0 / (H - L)
                if rank <= 3:
                    hit3 += 1.0 / (H - L)
                if rank <= 10:
                    hit10 += 1.0 / (H - L)
                mr += rank / (H - L)
                mrr += 1.0 / rank / (H - L)

        hit1 /= len(ranks)
        hit3 /= len(ranks)
        hit10 /= len(ranks)
        mr /= len(ranks)
        mrr /= len(ranks)

        logging.info(f'Results on {split} set:')
        logging.info(f'  Hits@1:  {hit1:.4f}')
        logging.info(f'  Hits@3:  {hit3:.4f}')
        logging.info(f'  Hits@10: {hit10:.4f}')
        logging.info(f'  MR:      {mr:.2f}')
        logging.info(f'  MRR:     {mrr:.4f}')

        return mrr


class JointTrainer:
    """
    联合训练器

    目标：
    1. 训练KGE + 规则
    2. 学习动态置信度w_i(h)
    3. 规则负采样与原RulE相同
    """

    def __init__(self, model, graph, train_dataset, rule_dataset, args):
        self.model = model
        self.graph = graph
        self.args = args
        self.device = args.device if hasattr(args, 'device') else torch.device('cpu')

        # 数据加载器
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.cpu_num if hasattr(args, 'cpu_num') else 0,
            collate_fn=kge_collate_fn
        )

        self.rule_dataloader = DataLoader(
            rule_dataset,
            batch_size=args.rule_batch_size if hasattr(args, 'rule_batch_size') else 128,
            shuffle=True,
            num_workers=args.cpu_num if hasattr(args, 'cpu_num') else 0
        )

        # 优化器
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate
        )

        if self.device.type == 'cuda':
            self.model = self.model.cuda(self.device)

        logging.info('JointTrainer initialized')

    def train_step(self, batch_triple, batch_rule):
        """
        单步联合训练
        """
        self.model.train()
        self.optimizer.zero_grad()

        # ===== KGE分支 =====
        positive_sample, negative_sample, subsampling_weight, mode = batch_triple

        if self.device.type == 'cuda':
            positive_sample = positive_sample.cuda(self.device)
            negative_sample = negative_sample.cuda(self.device)
            subsampling_weight = subsampling_weight.cuda(self.device)

        # 计算KGE分数
        negative_score = self.model.compute_KGE((positive_sample, negative_sample), mode)
        positive_score = self.model.compute_KGE(positive_sample, mode='single')

        # KGE损失
        negative_score = (F.softmax(negative_score * self.args.adversarial_temperature, dim=1).detach()
                         * F.logsigmoid(-negative_score)).sum(dim=1)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_loss_kge = -(subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_loss_kge = -(subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        loss_kge = (positive_loss_kge + negative_loss_kge) / 2

        # ===== Rule分支 =====
        positive_rule, negative_idx, negative_rule, mode_rule, rule_mask = batch_rule

        if self.device.type == 'cuda':
            positive_rule = positive_rule.cuda(self.device)
            negative_idx = negative_idx.cuda(self.device)
            negative_rule = negative_rule.cuda(self.device)
            rule_mask = rule_mask.cuda(self.device)

        # 计算Rule分数
        negative_rule_score = self.model.compute_ruleE((positive_rule, rule_mask, negative_idx, negative_rule), mode=mode_rule)
        positive_rule_score = self.model.compute_ruleE((positive_rule, rule_mask))

        # Rule损失
        negative_rule_score = (F.softmax(negative_rule_score * self.args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_rule_score)).sum(dim=1)
        positive_rule_score = F.logsigmoid(positive_rule_score)

        positive_loss_rule = -positive_rule_score.mean()
        negative_loss_rule = -negative_rule_score.mean()
        loss_rule = (positive_loss_rule + negative_loss_rule) / 2

        # ===== 体积正则化 =====
        widths = self.model.entity_width_emb.weight
        loss_vol = self.model.volume_regularization(widths)

        # ===== 总损失 =====
        weight_rule = self.args.weight_rule if hasattr(self.args, 'weight_rule') else 1.0
        loss_total = loss_kge + weight_rule * loss_rule + loss_vol

        loss_total.backward()
        self.optimizer.step()

        return {
            'loss_kge': loss_kge.item(),
            'loss_rule': loss_rule.item(),
            'loss_vol': loss_vol.item(),
            'loss_total': loss_total.item()
        }

    def train(self, max_steps):
        """
        训练主循环
        """
        joint_valid_steps = getattr(self.args, 'joint_valid_steps', 500)
        logging.info(f'Starting Joint Training for {max_steps} steps')
        logging.info(f'Validation every {joint_valid_steps} steps')

        step = 0
        best_mrr = 0.0
        training_logs = []

        # 创建迭代器
        triple_iter = iter(self.train_dataloader)
        rule_iter = iter(self.rule_dataloader)

        while step < max_steps:
            try:
                batch_triple = next(triple_iter)
            except StopIteration:
                triple_iter = iter(self.train_dataloader)
                batch_triple = next(triple_iter)

            try:
                batch_rule = next(rule_iter)
            except StopIteration:
                rule_iter = iter(self.rule_dataloader)
                batch_rule = next(rule_iter)

            log = self.train_step(batch_triple, batch_rule)
            training_logs.append(log)

            # 每100步输出平均损失
            if step % 100 == 0 and step > 0:
                avg_loss_kge = sum([l['loss_kge'] for l in training_logs]) / len(training_logs)
                avg_loss_rule = sum([l['loss_rule'] for l in training_logs]) / len(training_logs)
                avg_loss_vol = sum([l['loss_vol'] for l in training_logs]) / len(training_logs)
                avg_loss_total = sum([l['loss_total'] for l in training_logs]) / len(training_logs)

                logging.info(f'[Joint] Step {step}/{max_steps}: '
                           f'loss_kge={avg_loss_kge:.4f}, '
                           f'loss_rule={avg_loss_rule:.4f}, '
                           f'loss_vol={avg_loss_vol:.6f}, '
                           f'loss_total={avg_loss_total:.4f}')
                training_logs = []

            # 定期验证
            if step % joint_valid_steps == 0 and step > 0:
                logging.info(f'--- Validation at step {step} ---')
                mrr = self.evaluate(split='valid')

                if mrr > best_mrr:
                    best_mrr = mrr
                    logging.info(f'*** New best MRR: {best_mrr:.4f} ***')

                    # 保存最优模型
                    best_checkpoint = os.path.join(self.args.save_path, 'joint_best_model.pt')
                    os.makedirs(self.args.save_path, exist_ok=True)
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'step': step,
                        'best_mrr': best_mrr
                    }, best_checkpoint)
                    logging.info(f'Best model saved at step {step}')

                self.model.train()

            step += 1

        logging.info('Joint Training completed')
        logging.info(f'Best validation MRR: {best_mrr:.4f}')

        return best_mrr

    @torch.no_grad()
    def evaluate(self, split='valid'):
        """
        评估模型性能（使用KGE，暂不使用规则推理）
        """
        logging.info(f'Evaluating on {split} set...')

        from data import EvalDataset
        eval_dataset = EvalDataset(self.graph, split=split, batch_size=1)
        dataloader = DataLoader(eval_dataset, batch_size=1, num_workers=0)

        self.model.eval()

        # 收集所有预测结果
        concat_logits = []
        concat_all_h = []
        concat_all_r = []
        concat_all_t = []
        concat_flag = []

        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            flag = flag.squeeze(0)

            if self.device.type == 'cuda':
                all_h = all_h.cuda(self.device)
                all_r = all_r.cuda(self.device)
                all_t = all_t.cuda(self.device)
                flag = flag.cuda(self.device)

            # 计算KGE分数
            batch_size = all_h.size(0)
            num_entities = self.graph.entity_size

            logits = torch.zeros(batch_size, num_entities, device=self.device)
            for i in range(batch_size):
                h = all_h[i:i+1].expand(num_entities)
                r = all_r[i:i+1].expand(num_entities)
                t = torch.arange(num_entities, device=self.device)

                sample = torch.stack([h, r, t], dim=1)
                scores = self.model.compute_KGE(sample, mode='single')
                logits[i] = scores.squeeze()

            concat_logits.append(logits)
            concat_all_h.append(all_h)
            concat_all_r.append(all_r)
            concat_all_t.append(all_t)
            concat_flag.append(flag)

        # 合并所有batch的结果
        concat_logits = torch.cat(concat_logits, dim=0)
        concat_all_h = torch.cat(concat_all_h, dim=0)
        concat_all_r = torch.cat(concat_all_r, dim=0)
        concat_all_t = torch.cat(concat_all_t, dim=0)
        concat_flag = torch.cat(concat_flag, dim=0)

        # 计算排名
        ranks = []
        for k in range(concat_all_t.size(0)):
            t = concat_all_t[k]
            val = concat_logits[k, t]

            L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
            H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
            ranks.append([L, H])

        # 计算指标
        hit1, hit3, hit10, mr, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        for (L, H) in ranks:
            for rank in range(L, H):
                if rank <= 1:
                    hit1 += 1.0 / (H - L)
                if rank <= 3:
                    hit3 += 1.0 / (H - L)
                if rank <= 10:
                    hit10 += 1.0 / (H - L)
                mr += rank / (H - L)
                mrr += 1.0 / rank / (H - L)

        hit1 /= len(ranks)
        hit3 /= len(ranks)
        hit10 /= len(ranks)
        mr /= len(ranks)
        mrr /= len(ranks)

        logging.info(f'Results on {split} set:')
        logging.info(f'  Hits@1:  {hit1:.4f}')
        logging.info(f'  Hits@3:  {hit3:.4f}')
        logging.info(f'  Hits@10: {hit10:.4f}')
        logging.info(f'  MR:      {mr:.2f}')
        logging.info(f'  MRR:     {mrr:.4f}')

        return mrr
