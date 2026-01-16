"""
TAPC-RulE: Path Critic Module
路径判别器模块 - 基于Bi-GRU的路径质量判别
"""

import torch
import torch.nn as nn
import logging


class PathCritic(nn.Module):
    """
    路径判别器
    使用Bi-GRU编码路径序列，判断路径的语义合理性
    """

    def __init__(self, input_dim, hidden_dim=256, num_layers=1, dropout=0.1):
        """
        初始化路径判别器

        参数:
            input_dim: 输入维度（实体嵌入维度，通常是hidden_dim*2）
            hidden_dim: GRU隐藏层维度
            num_layers: GRU层数
            dropout: Dropout概率
        """
        super(PathCritic, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bi-GRU编码器
        self.bi_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # MLP打分器
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 类型嵌入（后续挂载）
        self.type_embeddings = None

        logging.info(f"PathCritic初始化: input_dim={input_dim}, "
                    f"hidden_dim={hidden_dim}, num_layers={num_layers}")

    def forward(self, path_sequences):
        """
        前向传播

        参数:
            path_sequences: [batch_size, seq_len, input_dim]
                          交错的实体和关系嵌入序列

        返回:
            scores: [batch_size, 1] 路径置信度分数 (0-1之间)
        """
        # Bi-GRU编码
        # output: [batch_size, seq_len, hidden_dim*2]
        # h_n: [num_layers*2, batch_size, hidden_dim]
        output, h_n = self.bi_gru(path_sequences)

        # Max-Pooling: 捕捉最显著的特征（可能是异常信号）
        # [batch_size, seq_len, hidden_dim*2] -> [batch_size, hidden_dim*2]
        pooled, _ = torch.max(output, dim=1)

        # MLP打分
        scores = self.scorer(pooled)  # [batch_size, 1]

        return scores


class TypeAwareEmbedding(nn.Module):
    """
    类型感知的嵌入融合
    将实体嵌入与类型嵌入融合
    """

    def __init__(self, num_clusters, embedding_dim, lambda_weight=0.3):
        """
        初始化类型感知嵌入

        参数:
            num_clusters: 类型数量
            embedding_dim: 嵌入维度
            lambda_weight: 类型融合权重
        """
        super(TypeAwareEmbedding, self).__init__()

        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        self.lambda_weight = lambda_weight

        # 可学习的类型嵌入矩阵
        self.type_embeddings = nn.Embedding(num_clusters, embedding_dim)
        nn.init.xavier_uniform_(self.type_embeddings.weight)

        logging.info(f"TypeAwareEmbedding初始化: num_clusters={num_clusters}, "
                    f"lambda={lambda_weight}")

    def forward(self, entity_ids, entity_emb_layer, entity_to_type):
        """
        融合实体嵌入和类型嵌入

        参数:
            entity_ids: [batch_size] 实体ID
            entity_emb_layer: nn.Embedding 实体嵌入层
            entity_to_type: Dict[int, int] 实体到类型的映射

        返回:
            fused_emb: [batch_size, embedding_dim] 融合后的嵌入
        """
        # 获取原始实体嵌入
        e_kg = entity_emb_layer(entity_ids)  # [batch_size, embedding_dim]

        # 获取类型ID
        type_ids = torch.tensor(
            [entity_to_type[eid.item()] for eid in entity_ids],
            dtype=torch.long,
            device=entity_ids.device
        )

        # 获取类型嵌入
        e_type = self.type_embeddings(type_ids)  # [batch_size, embedding_dim]

        # 融合: x_e = e_KG + λ * e_Type
        fused_emb = e_kg + self.lambda_weight * e_type

        return fused_emb


def construct_path_sequence(path, entity_emb_layer, relation_emb_layer,
                            type_aware_emb, entity_to_type, device):
    """
    构造路径的输入序列

    参数:
        path: Tuple (e0, r1, e1, r2, e2) 路径元组
        entity_emb_layer: 实体嵌入层
        relation_emb_layer: 关系嵌入层
        type_aware_emb: TypeAwareEmbedding实例
        entity_to_type: 实体到类型的映射
        device: 设备

    返回:
        sequence: [seq_len, embedding_dim] 交错序列
    """
    e0, r1, e1, r2, e2 = path

    # 转换为tensor
    e0_tensor = torch.tensor([e0], dtype=torch.long, device=device)
    e1_tensor = torch.tensor([e1], dtype=torch.long, device=device)
    e2_tensor = torch.tensor([e2], dtype=torch.long, device=device)
    r1_tensor = torch.tensor([r1], dtype=torch.long, device=device)
    r2_tensor = torch.tensor([r2], dtype=torch.long, device=device)

    # 融合实体嵌入（类型注入）
    x_e0 = type_aware_emb(e0_tensor, entity_emb_layer, entity_to_type)
    x_e1 = type_aware_emb(e1_tensor, entity_emb_layer, entity_to_type)
    x_e2 = type_aware_emb(e2_tensor, entity_emb_layer, entity_to_type)

    # 获取关系嵌入
    x_r1 = relation_emb_layer(r1_tensor)
    x_r2 = relation_emb_layer(r2_tensor)

    # 构造交错序列: [e0, r1, e1, r2, e2]
    sequence = torch.cat([x_e0, x_r1, x_e1, x_r2, x_e2], dim=0)
    # 形状: [5, embedding_dim]

    return sequence


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    print("PathCritic模块测试")

    # 测试PathCritic
    input_dim = 1000
    batch_size = 32
    seq_len = 5

    critic = PathCritic(input_dim=input_dim, hidden_dim=256)
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    output = critic(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")

    assert output.shape == (batch_size, 1), "输出形状错误"
    assert (output >= 0).all() and (output <= 1).all(), "输出范围错误"

    print("PathCritic测试通过！")
