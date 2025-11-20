"""
Rule-GNN 核心模型

用 GNN 的消息传递机制替换 RulE 的路径枚举
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from rule_gnn_layers import scatter_softmax, AttentionAggregation


class RuleAwareGraphConv(nn.Module):
    """
    规则感知的图卷积层

    核心创新：
    1. 注意力权重由规则嵌入调控
    2. 只有符合规则的边才有高注意力
    3. 消息聚合时自动过滤无关边
    """
    def __init__(self, in_dim, out_dim, num_relations, num_rules, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations

        # 关系特定的变换矩阵（类似 R-GCN）
        self.W_r = nn.Parameter(torch.Tensor(num_relations, in_dim, out_dim))

        # 注意力机制的参数
        self.W_q = nn.Linear(in_dim, out_dim)  # Query: 目标节点
        self.W_k = nn.Linear(in_dim * 3, out_dim)  # Key: [源节点; 关系; 规则]

        # 规则嵌入（将从预训练的RulE加载）
        self.rule_embedding = nn.Embedding(num_rules, in_dim)

        # 偏置
        self.bias = nn.Parameter(torch.Tensor(out_dim))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer Norm
        self.layer_norm = nn.LayerNorm(out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_r)
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.rule_embedding.weight)

    def forward(self, x, edge_index, edge_type, rule_ids, return_attention=False):
        """
        前向传播

        Args:
            x: 节点特征 [num_nodes, in_dim]
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
            rule_ids: 当前激活的规则ID [num_active_rules]
            return_attention: 是否返回注意力权重

        Returns:
            out: 更新后的节点特征 [num_nodes, out_dim]
            attention_weights: (可选) 注意力权重
        """
        src, dst = edge_index  # [num_edges]
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # 获取规则嵌入
        h_R = self.rule_embedding(rule_ids)  # [num_active_rules, in_dim]

        # 初始化累加器（边计算边聚合，节省内存）
        combined_messages = torch.zeros(num_edges, self.out_dim, device=x.device)
        combined_attention = torch.zeros(num_edges, device=x.device) if return_attention else None
        num_rules = len(rule_ids)

        if num_rules == 0:
            # 没有规则，返回零向量
            out = torch.zeros(num_nodes, self.out_dim, device=x.device)
            if return_attention:
                return out, None
            else:
                return out

        for rule_idx, rule_id in enumerate(rule_ids):
            h_r_single = h_R[rule_idx]  # [in_dim]

            # === 计算规则感知的注意力权重 ===

            # Query: 目标节点特征
            query = self.W_q(x[dst])  # [num_edges, out_dim]

            # 获取关系嵌入（简化：使用 W_r 的平均作为关系表示）
            relation_emb = torch.zeros(num_edges, self.in_dim, device=x.device)
            for r in range(self.num_relations):
                mask = (edge_type == r)
                if mask.sum() > 0:
                    relation_emb[mask] = self.W_r[r].mean(dim=-1)

            # 扩展规则嵌入到所有边
            rule_emb_expanded = h_r_single.unsqueeze(0).expand(num_edges, -1)

            # Key: 拼接 [源节点; 关系嵌入; 规则嵌入]
            key_input = torch.cat([
                x[src],              # 源节点特征
                relation_emb,        # 关系嵌入
                rule_emb_expanded    # 规则嵌入
            ], dim=-1)  # [num_edges, in_dim * 3]

            key = self.W_k(key_input)  # [num_edges, out_dim]

            # 计算注意力分数
            attn_scores = (query * key).sum(dim=-1) / (self.out_dim ** 0.5)
            # [num_edges]

            # Softmax归一化（针对每个目标节点）
            attn_weights = scatter_softmax(attn_scores, dst, dim=0, dim_size=num_nodes)
            # [num_edges]

            # === 计算消息 ===

            # 对每个关系类型分别计算消息
            messages = torch.zeros(num_edges, self.out_dim, device=x.device)

            for r in range(self.num_relations):
                mask = (edge_type == r)
                if mask.sum() > 0:
                    # m_ij = α_ij * W_r * h_j
                    msg = torch.matmul(x[src[mask]], self.W_r[r])  # [num_edges_r, out_dim]
                    msg = msg * attn_weights[mask].unsqueeze(-1)  # 加权
                    messages[mask] = msg

            # 累加到累加器（而不是保存到列表）
            combined_messages += messages
            if return_attention:
                combined_attention += attn_weights

        # === 聚合所有规则的消息 ===

        # 取平均
        combined_messages /= num_rules
        if return_attention:
            combined_attention /= num_rules

        # 聚合到目标节点
        out = scatter_add(combined_messages, dst, dim=0, dim_size=num_nodes)
        # [num_nodes, out_dim]

        # 添加偏置
        out = out + self.bias

        # Layer Normalization
        out = self.layer_norm(out)

        # ReLU激活
        out = F.relu(out)

        # Dropout
        out = self.dropout(out)

        if return_attention:
            return out, combined_attention
        else:
            return out


class RuleGNN(nn.Module):
    """
    完整的 Rule-GNN 模型

    用 GNN 消息传播替换 RulE 的路径枚举
    """
    def __init__(self, num_entities, num_relations, num_rules,
                 hidden_dim, num_layers, dropout=0.1):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_rules = num_rules
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # 实体嵌入（将从预训练的RulE加载）
        self.entity_embedding = nn.Embedding(num_entities, hidden_dim)

        # 关系嵌入（将从预训练的RulE加载）
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)

        # Rule-Aware GNN 层
        self.conv_layers = nn.ModuleList([
            RuleAwareGraphConv(hidden_dim, hidden_dim, num_relations, num_rules, dropout)
            for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 预测层
        self.score_func = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # 用于保存注意力权重（可解释性）
        self.attention_weights = []

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def load_pretrained_embeddings(self, embeddings_dict):
        """
        从预训练的 RulE 模型加载嵌入

        Args:
            embeddings_dict: 包含 'entity_embedding', 'relation_embedding', 'rule_emb' 的字典
        """
        print("Loading pretrained embeddings from RulE...")

        # 加载实体嵌入
        if 'entity_embedding' in embeddings_dict:
            entity_emb = embeddings_dict['entity_embedding']
            # RulE的实体嵌入是复数表示（real+imag），取实部
            if entity_emb.size(1) == self.hidden_dim * 2:
                entity_emb = entity_emb[:, :self.hidden_dim]  # 只取实部
            self.entity_embedding.weight.data.copy_(entity_emb)
            print(f"  Loaded entity embeddings: {entity_emb.shape}")

        # 加载关系嵌入
        if 'relation_embedding' in embeddings_dict:
            relation_emb = embeddings_dict['relation_embedding']
            # RulE 只有 num_relations + 1 个关系嵌入（包括 padding）
            # Rule-GNN 有 num_relations * 2 个（正向 + 逆向）
            # 逆关系嵌入 = -1 * 正向关系嵌入
            num_original = self.num_relations // 2  # 原始关系数量

            # 正向关系：直接复制
            self.relation_embedding.weight.data[:num_original].copy_(relation_emb[:num_original])

            # 逆向关系：复制并取负
            self.relation_embedding.weight.data[num_original:].copy_(-relation_emb[:num_original])

            print(f"  Loaded relation embeddings: {relation_emb.shape} -> [{self.num_relations}, {self.hidden_dim}]")
            print(f"    Forward relations: 0-{num_original-1}")
            print(f"    Inverse relations: {num_original}-{self.num_relations-1} (negated)")

        # 加载规则嵌入（到每个GNN层）
        if 'rule_emb' in embeddings_dict:
            rule_emb = embeddings_dict['rule_emb']
            for layer in self.conv_layers:
                layer.rule_embedding.weight.data.copy_(rule_emb)
            print(f"  Loaded rule embeddings: {rule_emb.shape}")

    def forward(self, queries, edge_index, edge_type, rule_ids,
                candidates=None, return_attention=False):
        """
        前向传播

        Args:
            queries: 查询三元组 (h, r) [batch_size, 2]
            edge_index: 全图的边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
            rule_ids: 当前查询相关的规则ID列表 [num_active_rules]
            candidates: 候选尾实体（如果为None，则对所有实体打分）
            return_attention: 是否返回注意力权重

        Returns:
            scores: 预测分数 [batch_size, num_candidates or num_entities]
            attention_weights: (可选) 注意力权重
        """
        # 初始化节点特征
        h = self.entity_embedding.weight  # [num_entities, hidden_dim]

        # 多层传播（规则长度）
        self.attention_weights = []

        for layer_idx, conv in enumerate(self.conv_layers):
            if return_attention:
                h, attn = conv(h, edge_index, edge_type, rule_ids, return_attention=True)
                self.attention_weights.append(attn)
            else:
                h = conv(h, edge_index, edge_type, rule_ids, return_attention=False)

        # 提取查询头实体的表示
        batch_size = queries.size(0)
        h_heads = h[queries[:, 0]]  # [batch_size, hidden_dim]

        # 计算得分
        if candidates is None:
            # 对所有实体打分
            h_tails = h  # [num_entities, hidden_dim]

            # 广播计算
            h_heads_exp = h_heads.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            h_tails_exp = h_tails.unsqueeze(0)  # [1, num_entities, hidden_dim]

            # 拼接并通过MLP
            combined = torch.cat([
                h_heads_exp.expand(-1, self.num_entities, -1),
                h_tails_exp.expand(batch_size, -1, -1)
            ], dim=-1)  # [batch_size, num_entities, hidden_dim*2]

            scores = self.score_func(combined).squeeze(-1)  # [batch_size, num_entities]
        else:
            # 只对候选实体打分
            h_tails = h[candidates]  # [batch_size, num_candidates, hidden_dim]

            combined = torch.cat([
                h_heads.unsqueeze(1).expand(-1, candidates.size(1), -1),
                h_tails
            ], dim=-1)

            scores = self.score_func(combined).squeeze(-1)  # [batch_size, num_candidates]

        if return_attention:
            return scores, self.attention_weights
        else:
            return scores

    def compute_rule_loss(self, rule_data, gamma_rule=5.0):
        """
        计算规则一致性损失（类似RulE的规则损失）

        Args:
            rule_data: 规则数据 [(rule_id, body_relations, head_relation), ...]
            gamma_rule: Margin参数

        Returns:
            rule_loss: 规则损失
        """
        if len(rule_data) == 0:
            return torch.tensor(0.0, device=self.entity_embedding.weight.device)

        rule_loss = 0.0

        for rule_id, body_rels, head_rel in rule_data:
            # 获取规则嵌入（从第一层）
            h_R = self.conv_layers[0].rule_embedding(torch.tensor([rule_id], device=self.entity_embedding.weight.device))

            # 获取关系嵌入
            h_body = self.relation_embedding(torch.tensor(body_rels, device=self.entity_embedding.weight.device))
            h_head = self.relation_embedding(torch.tensor([head_rel], device=self.entity_embedding.weight.device))

            # 组合规则体（简单求和）
            h_body_sum = h_body.sum(dim=0, keepdim=True)  # [1, hidden_dim]

            # 期望: h_body_sum + h_R ≈ h_head
            distance = torch.norm(h_body_sum + h_R - h_head, p=2)

            # 使用margin-based loss
            rule_loss += F.relu(distance - gamma_rule)

        return rule_loss / len(rule_data)
