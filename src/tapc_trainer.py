"""
TAPC-RulE: Critic Trainer Module
Critic训练器模块 - 负责训练路径判别器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
from tqdm import tqdm

from tapc_critic import PathCritic, TypeAwareEmbedding, construct_path_sequence
from tapc_dataset import PathCriticDataset


class CriticTrainer:
    """
    路径判别器训练器
    负责训练PathCritic模型
    """

    def __init__(self, graph, entity_to_type, entity_emb_layer, relation_emb_layer,
                 hidden_dim=256, num_layers=1, dropout=0.1, lambda_weight=0.3,
                 proj_dim=512, lr=0.001, device='cuda'):
        """
        初始化训练器

        参数:
            graph: KnowledgeGraph实例
            entity_to_type: Dict[int, int] 实体到类型的映射
            entity_emb_layer: nn.Embedding 实体嵌入层
            relation_emb_layer: nn.Embedding 关系嵌入层
            hidden_dim: Critic隐藏层维度
            num_layers: GRU层数
            dropout: Dropout概率
            lambda_weight: 类型融合权重
            proj_dim: 投影维度（用于统一实体和关系嵌入维度）
            lr: 学习率
            device: 设备
        """
        self.graph = graph
        self.entity_to_type = entity_to_type
        self.entity_emb_layer = entity_emb_layer
        self.relation_emb_layer = relation_emb_layer
        self.device = device

        # 获取嵌入维度
        entity_dim = entity_emb_layer.weight.shape[1]
        relation_dim = relation_emb_layer.weight.shape[1]
        num_clusters = max(entity_to_type.values()) + 1

        # 初始化类型感知嵌入（使用entity_dim）
        self.type_aware_emb = TypeAwareEmbedding(
            num_clusters=num_clusters,
            embedding_dim=entity_dim,
            lambda_weight=lambda_weight
        ).to(device)

        # 初始化PathCritic（使用特征投影）
        self.critic = PathCritic(
            entity_dim=entity_dim,
            relation_dim=relation_dim,
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        # 优化器
        self.optimizer = optim.Adam(
            list(self.critic.parameters()) + list(self.type_aware_emb.parameters()),
            lr=lr
        )

        # 损失函数
        self.criterion = nn.BCELoss()

        logging.info(f"CriticTrainer初始化完成:")
        logging.info(f"  - 实体嵌入维度: {entity_dim}")
        logging.info(f"  - 关系嵌入维度: {relation_dim}")
        logging.info(f"  - 投影维度: {proj_dim}")
        logging.info(f"  - 隐藏层维度: {hidden_dim}")
        logging.info(f"  - GRU层数: {num_layers}")
        logging.info(f"  - Dropout: {dropout}")
        logging.info(f"  - 类型数量: {num_clusters}")
        logging.info(f"  - Lambda权重: {lambda_weight}")
        logging.info(f"  - 学习率: {lr}")

    def train_epoch(self, dataloader, epoch):
        """
        训练一个epoch

        参数:
            dataloader: DataLoader实例
            epoch: 当前epoch数

        返回:
            avg_loss: 平均损失
            accuracy: 准确率
        """
        self.critic.train()
        self.type_aware_emb.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (paths, labels) in enumerate(pbar):
            # 构造路径序列
            batch_sequences = []

            for path in paths:
                # path: (e0, r1, e1, r2, e2, ...) - 通用路径格式
                sequence = construct_path_sequence(
                    path,
                    self.entity_emb_layer,
                    self.relation_emb_layer,
                    self.type_aware_emb,
                    self.entity_to_type,
                    self.device,
                    entity_proj=self.critic.entity_proj,
                    relation_proj=self.critic.relation_proj
                )
                batch_sequences.append(sequence)

            # 堆叠成batch: [batch_size, seq_len, embedding_dim]
            batch_sequences = torch.stack(batch_sequences, dim=0)

            # 标签转为tensor
            labels = torch.tensor(labels, dtype=torch.float32, device=self.device).unsqueeze(1)

            # 前向传播
            predictions = self.critic(batch_sequences)

            # 计算损失
            loss = self.criterion(predictions, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            predicted = (predictions > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, dataloader):
        """
        验证模型

        参数:
            dataloader: DataLoader实例

        返回:
            avg_loss: 平均损失
            accuracy: 准确率
        """
        self.critic.eval()
        self.type_aware_emb.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for paths, labels in dataloader:
                # 构造路径序列
                batch_sequences = []

                for path in paths:
                    sequence = construct_path_sequence(
                        path,
                        self.entity_emb_layer,
                        self.relation_emb_layer,
                        self.type_aware_emb,
                        self.entity_to_type,
                        self.device,
                        entity_proj=self.critic.entity_proj,
                        relation_proj=self.critic.relation_proj
                    )
                    batch_sequences.append(sequence)

                # 堆叠成batch
                batch_sequences = torch.stack(batch_sequences, dim=0)
                labels = torch.tensor(labels, dtype=torch.float32, device=self.device).unsqueeze(1)

                # 前向传播
                predictions = self.critic(batch_sequences)

                # 计算损失
                loss = self.criterion(predictions, labels)

                # 统计
                total_loss += loss.item()
                predicted = (predictions > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        return avg_loss, accuracy

    def save_checkpoint(self, save_path, epoch, best_acc):
        """
        保存模型检查点

        参数:
            save_path: 保存路径
            epoch: 当前epoch
            best_acc: 最佳准确率
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'critic_state_dict': self.critic.state_dict(),
            'type_aware_emb_state_dict': self.type_aware_emb.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': best_acc
        }

        torch.save(checkpoint, save_path)
        logging.info(f"检查点已保存到: {save_path}")

    def load_checkpoint(self, load_path):
        """
        加载模型检查点

        参数:
            load_path: 加载路径

        返回:
            epoch: 训练的epoch数
            best_acc: 最佳准确率
        """
        checkpoint = torch.load(load_path, map_location=self.device)

        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.type_aware_emb.load_state_dict(checkpoint['type_aware_emb_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

        logging.info(f"检查点已加载: epoch={epoch}, best_acc={best_acc:.4f}")

        return epoch, best_acc

    def train(self, train_dataset, val_dataset=None, num_epochs=10,
              batch_size=32, save_dir='./checkpoints'):
        """
        完整的训练流程

        参数:
            train_dataset: 训练数据集
            val_dataset: 验证数据集（可选）
            num_epochs: 训练轮数
            batch_size: 批次大小
            save_dir: 保存目录

        返回:
            best_acc: 最佳准确率
        """
        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )

        best_acc = 0.0
        best_epoch = 0

        logging.info("=" * 60)
        logging.info("开始训练PathCritic")
        logging.info("=" * 60)
        logging.info(f"训练参数:")
        logging.info(f"  - 训练样本数: {len(train_dataset)}")
        logging.info(f"  - Batch size: {batch_size}")
        logging.info(f"  - Epoch数: {num_epochs}")
        logging.info(f"  - 总batch数/epoch: {len(train_loader)}")
        if val_dataset is not None:
            logging.info(f"  - 验证样本数: {len(val_dataset)}")
        logging.info("=" * 60)

        for epoch in range(1, num_epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            logging.info(f"Epoch {epoch}/{num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # 验证
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                logging.info(f"Epoch {epoch}/{num_epochs} - "
                            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                # 保存最佳模型
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch
                    save_path = os.path.join(save_dir, 'best_critic.pth')
                    self.save_checkpoint(save_path, epoch, best_acc)
                    logging.info(f"新的最佳模型！Acc: {best_acc:.4f}")

            # 定期保存检查点
            if epoch % 5 == 0:
                save_path = os.path.join(save_dir, f'critic_epoch_{epoch}.pth')
                self.save_checkpoint(save_path, epoch, best_acc)

        logging.info("=" * 60)
        logging.info(f"训练完成！")
        logging.info(f"  - 最佳准确率: {best_acc:.4f}")
        logging.info(f"  - 最佳Epoch: {best_epoch}")
        logging.info("=" * 60)

        return best_acc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("CriticTrainer模块已创建")
    print("此模块需要配合其他TAPC模块使用")
