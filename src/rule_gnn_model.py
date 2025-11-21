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
    规则感知的图卷积层（稀疏化实现）

    核心创新：
    1. 注意力权重由规则嵌入调控
    2. 只有符合规则的边才有高注意力
    3. 消息聚合时自动过滤无关边

    稀疏化优化：
    1. 预构建关系到边的索引映射，避免重复计算mask
    2. Query计算移到规则循环外，只计算一次
    3. 按关系分块计算，每次只处理~113条边而非全部10432条
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

        # 稀疏索引映射（由set_graph初始化）
        self.relation2edges = None   # dict[int -> Tensor]: 关系r的边索引
        self.relation2src = None     # dict[int -> Tensor]: 关系r的源节点
        self.relation2dst = None     # dict[int -> Tensor]: 关系r的目标节点
        self.graph_initialized = False

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_r)
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.rule_embedding.weight)

    def set_graph(self, edge_index, edge_type, device):
        """
        预构建关系到边的索引映射（只需调用一次）

        这是稀疏化的关键：预先按关系类型分组存储边索引，
        避免在forward中重复计算mask操作。

        Args:
            edge_index: [2, num_edges] 边索引
            edge_type: [num_edges] 边类型
            device: 计算设备

        内存占用分析（以UMLS为例）：
            92关系 × 113边 × 8bytes × 3(edges/src/dst) = 249KB
            vs 原稠密实现的79MB，节省99.7%
        """
        self.relation2edges = {}
        self.relation2src = {}
        self.relation2dst = {}

        for r in range(self.num_relations):
            mask = (edge_type == r)
            if mask.sum() > 0:
                edge_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
                self.relation2edges[r] = edge_indices.to(device)
                self.relation2src[r] = edge_index[0][mask].to(device)
                self.relation2dst[r] = edge_index[1][mask].to(device)

        self.graph_initialized = True

        # 打印统计信息
        total_edges = sum(len(v) for v in self.relation2edges.values())
        avg_edges = total_edges / len(self.relation2edges) if self.relation2edges else 0
        print(f"  [RuleAwareGraphConv] 稀疏索引构建完成:")
        print(f"    - 关系数: {len(self.relation2edges)}")
        print(f"    - 总边数: {total_edges}")
        print(f"    - 平均每关系边数: {avg_edges:.1f}")

    def forward(self, x, edge_index, edge_type, rule_ids, return_attention=False):
        """
        稀疏化的前向传播

        核心优化：
        1. Query计算移到规则循环外，只计算1次（节省98%计算）
        2. 按关系分块计算，每次只处理~113条边（节省99%内存）
        3. 使用预构建的稀疏索引，避免重复mask计算

        Args:
            x: 节点特征 [num_nodes, in_dim]
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
            rule_ids: 当前激活的规则ID [num_active_rules]
            return_attention: 是否返回注意力权重

        Returns:
            out: 更新后的节点特征 [num_nodes, out_dim]
            attention_weights: (可选) 注意力权重

        内存对比（UMLS数据集）：
            稠密实现: ~24GB (OOM)
            稀疏实现: ~160MB (可运行)
        """
        src, dst = edge_index  # [num_edges]
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        device = x.device

        # 获取规则嵌入
        h_R = self.rule_embedding(rule_ids)  # [num_active_rules, in_dim]
        num_rules = len(rule_ids)

        if num_rules == 0:
            # 没有规则，返回零向量
            out = torch.zeros(num_nodes, self.out_dim, device=device)
            if return_attention:
                return out, None
            else:
                return out

        # ========== 优化1: Query只计算1次（移到规则循环外）==========
        # 原实现：在每个规则循环内计算，50次 × 79MB = 3.95GB
        # 优化后：只计算1次，79MB
        query_all = self.W_q(x[dst])  # [num_edges, out_dim]

        # 初始化累加器
        combined_messages = torch.zeros(num_edges, self.out_dim, device=device)
        combined_attention = torch.zeros(num_edges, device=device) if return_attention else None

        # ========== 优化2: 按关系分块计算（稀疏化核心）==========
        # 确保稀疏索引已初始化
        if not self.graph_initialized or self.relation2edges is None:
            raise RuntimeError("稀疏索引未初始化，请先调用 set_graph() 方法")

        for rule_idx in range(num_rules):
            h_rule = h_R[rule_idx]  # [in_dim]

            # 按关系分块处理
            for r in self.relation2edges.keys():
                # 获取当前关系的稀疏索引（预构建，无需计算mask）
                edge_indices_r = self.relation2edges[r]  # [num_edges_r] ~113
                src_r = self.relation2src[r]             # [num_edges_r]
                dst_r = self.relation2dst[r]             # [num_edges_r]
                num_edges_r = src_r.size(0)

                if num_edges_r == 0:
                    continue

                # Query: 从预计算结果中索引（不分配新内存）
                query_r = query_all[edge_indices_r]  # [num_edges_r, out_dim]

                # 源节点特征
                h_src_r = x[src_r]  # [num_edges_r, in_dim]

                # 关系嵌入（使用W_r的平均作为关系表示）
                h_rel_r = self.W_r[r].mean(dim=-1)  # [in_dim]

                # Key: 构建小矩阵（核心内存节省点）
                # 原实现：[10432, 6000] = 237MB
                # 稀疏实现：[~113, 6000] = 2.6MB
                key_input_r = torch.cat([
                    h_src_r,                                          # [num_edges_r, in_dim]
                    h_rel_r.unsqueeze(0).expand(num_edges_r, -1),    # [num_edges_r, in_dim]
                    h_rule.unsqueeze(0).expand(num_edges_r, -1)      # [num_edges_r, in_dim]
                ], dim=-1)  # [num_edges_r, in_dim * 3]

                key_r = self.W_k(key_input_r)  # [num_edges_r, out_dim]

                # 注意力分数
                attn_scores_r = (query_r * key_r).sum(dim=-1) / (self.out_dim ** 0.5)
                # [num_edges_r]

                # Softmax（稀疏版本，只对当前关系的边）
                attn_weights_r = scatter_softmax(attn_scores_r, dst_r, dim=0, dim_size=num_nodes)
                # [num_edges_r]

                # 消息计算
                # 原实现：[10432, 2000] = 79MB
                # 稀疏实现：[~113, 2000] = 0.86MB
                msg_r = torch.matmul(h_src_r, self.W_r[r])  # [num_edges_r, out_dim]
                msg_r = msg_r * attn_weights_r.unsqueeze(-1)  # 加权

                # 稀疏累加到对应边位置
                combined_messages[edge_indices_r] += msg_r

                if return_attention:
                    combined_attention[edge_indices_r] += attn_weights_r

        # ========== 聚合所有规则的消息 ==========

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
