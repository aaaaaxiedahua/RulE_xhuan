"""
Rule-GNN 训练器

负责训练 Rule-GNN 模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import tqdm
import os

from rule_gnn_model import RuleGNN
from data import KnowledgeGraph, ValidDataset, TestDataset


class RuleGNNTrainer:
    """
    Rule-GNN 训练器

    从预训练的 RulE 模型加载嵌入，训练 GNN 层
    """

    def __init__(self, model, graph, train_dataset, valid_dataset, test_dataset,
                 device='cuda', logger=None):
        """
        初始化训练器

        Args:
            model: RuleGNN 模型
            graph: KnowledgeGraph 对象
            train_dataset: 训练数据集
            valid_dataset: 验证数据集
            test_dataset: 测试数据集
            device: 设备 ('cuda' 或 'cpu')
            logger: 日志记录器
        """
        self.model = model.to(device)
        self.graph = graph
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        # 构建 PyG 格式的图
        self.edge_index, self.edge_type = self._build_pyg_graph()
        self.edge_index = self.edge_index.to(device)
        self.edge_type = self.edge_type.to(device)

    def _build_pyg_graph(self):
        """
        将 KnowledgeGraph 转换为 PyTorch Geometric 格式

        Returns:
            edge_index: [2, num_edges] 边索引
            edge_type: [num_edges] 边类型
        """
        all_edges = []
        all_types = []

        # 收集所有边（包括训练集、验证集、测试集）
        for split in ['train', 'valid', 'test']:
            data = getattr(self.graph, f'{split}_facts')
            for h, r, t in data:
                all_edges.append([h, t])
                all_types.append(r)

                # 添加逆边
                all_edges.append([t, h])
                all_types.append(r + self.graph.relation_size)

        edge_index = torch.tensor(all_edges, dtype=torch.long).t()
        edge_type = torch.tensor(all_types, dtype=torch.long)

        return edge_index, edge_type

    def load_pretrained_from_rule(self, checkpoint_path):
        """
        从预训练的 RulE 模型加载嵌入

        Args:
            checkpoint_path: RulE checkpoint 路径
        """
        self.logger.info(f"Loading pretrained embeddings from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 提取嵌入
        embeddings_dict = {
            'entity_embedding': checkpoint['entity_embedding.weight'],
            'relation_embedding': checkpoint['relation_embedding.weight'],
            'rule_emb': checkpoint['rule_emb.weight']
        }

        # 加载到模型
        self.model.load_pretrained_embeddings(embeddings_dict)

        self.logger.info("Pretrained embeddings loaded successfully")

    def get_active_rules(self, query_relations):
        """
        获取查询关系对应的激活规则

        Args:
            query_relations: 查询关系ID列表 [batch_size]

        Returns:
            rule_ids: 激活的规则ID列表
        """
        # 收集所有相关规则（去重）
        active_rules = set()

        for r in query_relations:
            r_item = r.item() if torch.is_tensor(r) else r
            if r_item in self.graph.relation2rules:
                for rule in self.graph.relation2rules[r_item]:
                    rule_id = rule[0]  # rule = [rule_id, head, body...]
                    active_rules.add(rule_id)

        return torch.tensor(list(active_rules), dtype=torch.long, device=self.device)

    def train(self, args):
        """
        训练 Rule-GNN 模型

        Args:
            args: 训练参数
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Rule-GNN Training")
        self.logger.info("=" * 80)

        # 优化器
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.g_lr
        )

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # 训练循环
        best_mrr = 0.0
        patience_counter = 0
        max_patience = 10

        # 使用 rule_gnn_num_iters，默认值为 50
        num_epochs = getattr(args, 'rule_gnn_num_iters', 50)
        valid_every = getattr(args, 'rule_gnn_valid_every', 5)

        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # 训练一个 epoch
            train_loss = self.train_epoch(optimizer, args)

            self.logger.info(f"Train Loss: {train_loss:.4f}")

            # 验证
            if (epoch + 1) % valid_every == 0 or epoch == num_epochs - 1:
                valid_metrics = self.evaluate(self.valid_dataset, split='valid')

                self.logger.info(f"Valid MRR: {valid_metrics['MRR']:.4f}")
                self.logger.info(f"Valid Hits@1: {valid_metrics['HITS@1']:.4f}")
                self.logger.info(f"Valid Hits@3: {valid_metrics['HITS@3']:.4f}")
                self.logger.info(f"Valid Hits@10: {valid_metrics['HITS@10']:.4f}")

                # 学习率调度
                scheduler.step(valid_metrics['MRR'])

                # 保存最佳模型
                if valid_metrics['MRR'] > best_mrr:
                    best_mrr = valid_metrics['MRR']
                    patience_counter = 0

                    save_path = os.path.join(args.save_path, 'rule_gnn_best.pt')
                    self.save_checkpoint(save_path)
                    self.logger.info(f"Saved best model to {save_path}")
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= max_patience:
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        # 最终测试
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Final Test Evaluation")
        self.logger.info("=" * 80)

        # 加载最佳模型
        best_checkpoint_path = os.path.join(args.save_path, 'rule_gnn_best.pt')
        self.load_checkpoint(best_checkpoint_path)

        test_metrics = self.evaluate(self.test_dataset, split='test')

        self.logger.info(f"Test MRR: {test_metrics['MRR']:.4f}")
        self.logger.info(f"Test Hits@1: {test_metrics['HITS@1']:.4f}")
        self.logger.info(f"Test Hits@3: {test_metrics['HITS@3']:.4f}")
        self.logger.info(f"Test Hits@10: {test_metrics['HITS@10']:.4f}")
        self.logger.info(f"Test MR: {test_metrics['MR']:.2f}")

        return test_metrics

    def train_epoch(self, optimizer, args):
        """
        训练一个 epoch

        Args:
            optimizer: 优化器
            args: 训练参数

        Returns:
            avg_loss: 平均损失
        """
        self.model.train()

        # TrainDataset 已经内部分好 batch，每个 item 就是一个 batch
        # 所以 DataLoader 的 batch_size 设为 1
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4
        )

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training")

        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= args.batch_per_epoch:
                break

            # 解包批次数据
            # TrainDataset 返回: (all_h, all_r, all_t, target, edges_to_remove)
            all_h, all_r, all_t, target, edges_to_remove = batch

            # 因为 DataLoader batch_size=1，需要 squeeze 掉第一个维度
            all_h = all_h.squeeze(0).to(self.device)
            all_r = all_r.squeeze(0).to(self.device)
            all_t = all_t.squeeze(0).to(self.device)
            target = target.squeeze(0).to(self.device)
            edges_to_remove = edges_to_remove.squeeze(0).to(self.device)

            # 构建 pos_samples: [batch_size, 3] (h, r, t)
            pos_samples = torch.stack([all_h, all_r, all_t], dim=1)

            # 获取查询（h, r）
            queries = pos_samples[:, :2]  # [batch_size, 2]
            batch_size = queries.size(0)

            # 获取激活的规则
            rule_ids = self.get_active_rules(queries[:, 1])

            if len(rule_ids) == 0:
                # 没有规则，跳过
                continue

            # 对所有实体打分
            scores = self.model(queries, self.edge_index, self.edge_type,
                               rule_ids, candidates=None)
            # [batch_size, num_entities]

            # 计算损失（二元交叉熵 + 标签平滑）
            # target 是多热标签 [batch_size, num_entities]
            if args.smoothing > 0:
                # 标签平滑
                smooth_target = target * (1.0 - args.smoothing) + args.smoothing / self.graph.entity_size
                loss = nn.BCEWithLogitsLoss()(scores, smooth_target)
            else:
                loss = nn.BCEWithLogitsLoss()(scores, target)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 定期打印
            if (batch_idx + 1) % args.print_every == 0:
                avg_loss = total_loss / num_batches
                self.logger.info(f"Batch {batch_idx + 1}, Avg Loss: {avg_loss:.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def evaluate(self, dataset, split='valid'):
        """
        评估模型

        Args:
            dataset: 评估数据集
            split: 数据集划分 ('valid' 或 'test')

        Returns:
            metrics: 评估指标字典
        """
        self.model.eval()

        eval_loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            collate_fn=dataset.collate_fn
        )

        ranks = []
        reciprocal_ranks = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Evaluating {split}"):
                pos_samples, filter_mask = batch

                # 转移到设备
                pos_samples = pos_samples.to(self.device)  # [batch_size, 3]
                filter_mask = filter_mask.to(self.device)  # [batch_size, num_entities]

                # 查询
                queries = pos_samples[:, :2]  # [batch_size, 2]
                true_tails = pos_samples[:, 2]  # [batch_size]

                # 获取激活的规则
                rule_ids = self.get_active_rules(queries[:, 1])

                if len(rule_ids) == 0:
                    # 没有规则，使用所有规则
                    rule_ids = torch.arange(len(self.graph.rules), dtype=torch.long, device=self.device)

                # 对所有实体打分
                scores = self.model(queries, self.edge_index, self.edge_type,
                                   rule_ids, candidates=None)
                # [batch_size, num_entities]

                # 过滤已知三元组
                scores = scores.masked_fill(filter_mask.bool(), -1e9)

                # 计算排名
                batch_size = scores.size(0)
                for i in range(batch_size):
                    true_tail = true_tails[i].item()
                    true_score = scores[i, true_tail].item()

                    # 排名 = 比真实尾实体得分高的实体数 + 1
                    rank = (scores[i] > true_score).sum().item() + 1

                    ranks.append(rank)
                    reciprocal_ranks.append(1.0 / rank)

        # 计算指标
        ranks = np.array(ranks)
        reciprocal_ranks = np.array(reciprocal_ranks)

        metrics = {
            'MRR': np.mean(reciprocal_ranks),
            'MR': np.mean(ranks),
            'HITS@1': np.mean(ranks <= 1),
            'HITS@3': np.mean(ranks <= 3),
            'HITS@10': np.mean(ranks <= 10)
        }

        return metrics

    def save_checkpoint(self, path):
        """
        保存检查点

        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        """
        加载检查点

        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Loaded checkpoint from {path}")
