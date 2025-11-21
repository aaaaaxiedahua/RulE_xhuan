# Rule-GNN 稀疏化改造方案

## 1. 问题背景

### 1.1 当前问题

Rule-GNN在训练时出现CUDA OOM错误：

```
RuntimeError: CUDA error: out of memory
File "rule_gnn_model.py", line 118, in forward
    attn_scores = (query * key).sum(dim=-1) / (self.out_dim ** 0.5)
```

**关键发现**：
- 预训练阶段（RulE模型）：正常运行，无OOM
- GNN训练阶段（Rule-GNN模型）：第1个batch就OOM

### 1.2 内存分析

| 阶段 | 模型 | 内存峰值 | 结果 |
|------|------|---------|------|
| 预训练 | RulE (稀疏grounding) | ~5 GB | ✓ 正常 |
| GNN训练 | Rule-GNN (稠密矩阵) | ~24 GB | ✗ OOM |

### 1.3 根本原因

Rule-GNN使用**稠密矩阵**存储边特征，而原RulE使用**稀疏邻接表**。

---

## 2. 稠密 vs 稀疏对比分析

### 2.1 原RulE的稀疏实现

```python
# data.py - 原RulE的数据结构
self.relation2adjacency = [[[], []] for k in range(self.relation_size*2)]
# 每个关系单独存储: [[dst_indices], [src_indices]]

# 传播时只处理实际存在的边
def propagate(self, x, relation, edges_to_remove=None):
    node_in = self.relation2adjacency[relation][0][1]   # 稀疏索引
    node_out = self.relation2adjacency[relation][0][0]
    message = x[node_in]  # 只取有边的节点
    x = scatter(message, node_out, ...)  # 稀疏聚合
    return x
```

**特点**：
- 按关系类型分组存储边
- 只处理实际存在的边
- 使用`scatter`进行稀疏聚合

### 2.2 当前Rule-GNN的稠密实现

```python
# rule_gnn_model.py - 当前实现
def forward(self, x, edge_index, edge_type, rule_ids):
    # 为所有边分配全零矩阵
    relation_emb = torch.zeros(num_edges, self.in_dim, device=x.device)
    # [10432, 2000] = 79 MB

    for r in range(self.num_relations):  # 92个关系
        mask = (edge_type == r)
        if mask.sum() > 0:
            relation_emb[mask] = self.W_r[r].mean(dim=-1)
```

**问题**：
- 预先分配`[num_edges, hidden_dim]`的全零矩阵
- 循环中用mask选择性填充
- 每个规则循环都重复分配

### 2.3 内存占用对比

以UMLS数据集为例（num_edges=10432, hidden_dim=2000, num_relations=92, num_rules=50）：

| 矩阵 | 稠密实现 | 稀疏实现 | 节省比例 |
|------|---------|---------|---------|
| `relation_emb` | 79MB × 50 = 3.95GB | 0 (不存储) | 100% |
| `query` | 79MB × 50 = 3.95GB | 79MB × 1 = 79MB | 98% |
| `key_input` | 237MB × 50 = 11.85GB | 2.6MB × 1 = 2.6MB | 99.98% |
| `messages` | 79MB × 50 = 3.95GB | 0.86MB × 1 = 0.86MB | 99.98% |
| **总计** | **~24 GB** | **~160 MB** | **99.3%** |

---

## 3. GNN领域的稀疏矩阵方法

### 3.1 常用稀疏格式

#### 3.1.1 COO格式 (Coordinate Format)

```python
# PyTorch Geometric 使用的格式
edge_index = torch.tensor([[0, 1, 2],    # src nodes
                           [1, 2, 0]])    # dst nodes
# 表示边: 0→1, 1→2, 2→0
```

**优点**：
- 构造简单，易于理解
- 适合动态图（边的增删）
- PyG、DGL等框架的标准格式

**缺点**：
- 随机访问效率低
- 不适合矩阵运算

#### 3.1.2 CSR格式 (Compressed Sparse Row)

```python
# scipy.sparse 和 torch.sparse_csr_tensor 使用
row_ptr = [0, 2, 3, 4]      # 每行的起始位置
col_idx = [1, 2, 0, 1]      # 列索引
values = [1, 1, 1, 1]       # 值

# 表示邻接矩阵:
# [[0, 1, 1],
#  [1, 0, 0],
#  [0, 1, 0]]
```

**优点**：
- 行切片高效 O(1)
- 稀疏矩阵乘法高效
- 内存紧凑

**缺点**：
- 列切片低效
- 构造后修改代价大

#### 3.1.3 邻接表格式 (Adjacency List)

```python
# 原RulE使用的格式
relation2adjacency[r] = [[dst_nodes], [src_nodes]]

# 按关系类型分组
relation2adjacency[0] = [[1, 2], [0, 0]]  # 关系0的边: 0→1, 0→2
relation2adjacency[1] = [[0], [2]]         # 关系1的边: 2→0
```

**优点**：
- 按关系类型快速索引
- 适合异构图/知识图谱
- 内存占用与边数成正比

**缺点**：
- 需要额外维护索引结构

### 3.2 主流GNN框架的稀疏实现

#### 3.2.1 PyTorch Geometric (PyG)

```python
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

class GCNConv(MessagePassing):
    def forward(self, x, edge_index):
        # edge_index: [2, num_edges] (COO格式)
        row, col = edge_index

        # 消息传递 (稀疏)
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        # x_j: 源节点特征 [num_edges, hidden_dim]
        return x_j

    def aggregate(self, inputs, index):
        # 稀疏聚合
        return scatter_add(inputs, index, dim=0)
```

**关键函数**：
- `scatter_add`: 稀疏加法聚合
- `scatter_mean`: 稀疏平均聚合
- `scatter_max`: 稀疏最大值聚合
- `scatter_softmax`: 稀疏softmax

#### 3.2.2 Deep Graph Library (DGL)

```python
import dgl
import dgl.function as fn

class GCNLayer(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # 消息传递 (稀疏)
            g.update_all(
                message_func=fn.copy_u('h', 'm'),
                reduce_func=fn.sum('m', 'h_new')
            )
            return g.ndata['h_new']
```

**关键函数**：
- `fn.copy_u`: 复制源节点特征
- `fn.sum/mean/max`: 聚合函数
- `g.update_all`: 批量消息传递

#### 3.2.3 PyTorch Sparse

```python
import torch

# 创建稀疏张量 (COO格式)
indices = torch.tensor([[0, 1, 2],
                        [1, 2, 0]])
values = torch.tensor([1.0, 2.0, 3.0])
sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(3, 3))

# 稀疏矩阵乘法
result = torch.sparse.mm(sparse_tensor, dense_matrix)
```

**限制**：
- 不支持所有操作（如`torch.cat`）
- 梯度支持有限
- 某些操作需要`coalesce()`

### 3.3 R-GCN的稀疏实现

R-GCN是处理多关系图的标准方法，与Rule-GNN场景相似：

```python
# PyG的R-GCN实现
from torch_geometric.nn import RGCNConv

class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations):
        self.weight = Parameter(torch.Tensor(num_relations, in_channels, out_channels))

    def forward(self, x, edge_index, edge_type):
        # 按关系类型分组处理 (稀疏)
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)

        for r in range(self.num_relations):
            mask = (edge_type == r)
            if mask.sum() == 0:
                continue

            edge_index_r = edge_index[:, mask]
            # 只处理当前关系的边
            out += self.propagate(edge_index_r, x=x, weight=self.weight[r])

        return out
```

**关键思想**：按关系类型分块处理，而不是一次性处理所有边

### 3.4 CompGCN的稀疏实现

CompGCN是另一种处理多关系图的方法：

```python
class CompGCNConv(MessagePassing):
    def forward(self, x, edge_index, edge_type, rel_embed):
        # 获取关系嵌入
        rel = rel_embed[edge_type]  # [num_edges, hidden_dim]

        # 组合节点和关系嵌入 (稀疏)
        row, col = edge_index
        x_j = x[col]  # 源节点特征

        # 消息 = 节点特征 ⊙ 关系嵌入
        msg = x_j * rel  # [num_edges, hidden_dim]

        # 稀疏聚合
        out = scatter_add(msg, row, dim=0, dim_size=x.size(0))
        return out
```

---

## 4. 稀疏化改造方案

### 4.1 改造思路

将Rule-GNN从"全边稠密计算"改为"按关系分块稀疏计算"，借鉴R-GCN的实现方式。

```
┌─────────────────────────────────────────────────────────┐
│                     当前稠密实现                         │
├─────────────────────────────────────────────────────────┤
│ for rule in rules:                    # 50次           │
│     relation_emb = zeros(num_edges, hidden_dim)  # 79MB│
│     for r in relations:               # 92次           │
│         mask = (edge_type == r)                        │
│         relation_emb[mask] = W_r[r].mean()             │
│     key_input = cat([...])            # 237MB          │
│     ...                                                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                     稀疏化实现                           │
├─────────────────────────────────────────────────────────┤
│ query = W_q(x[dst])                   # 79MB (只算1次) │
│ relation2edges = precompute_indices() # 83KB           │
│                                                        │
│ for rule in rules:                    # 50次           │
│     for r in relations:               # 92次           │
│         edges_r = relation2edges[r]   # ~113条边       │
│         key_input_r = cat([...])      # 2.6MB          │
│         msg_r = compute_message(...)                   │
│         combined_messages[edges_r] += msg_r            │
└─────────────────────────────────────────────────────────┘
```

### 4.2 数据结构改造

#### 4.2.1 新增：关系到边的索引映射

```python
class RuleAwareGraphConv(nn.Module):
    def __init__(self, ...):
        # ... 原有参数 ...

        # 新增：存储关系到边的映射
        self.relation2edges = None   # dict[int -> Tensor]
        self.relation2src = None     # dict[int -> Tensor]
        self.relation2dst = None     # dict[int -> Tensor]

    def set_graph(self, edge_index, edge_type, device):
        """
        预构建关系到边的索引映射（只调用1次）

        Args:
            edge_index: [2, num_edges]
            edge_type: [num_edges]
            device: 计算设备
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
```

#### 4.2.2 内存占用分析

```
relation2edges: 92关系 × 113索引 × 8 bytes = 83 KB
relation2src:   92关系 × 113索引 × 8 bytes = 83 KB
relation2dst:   92关系 × 113索引 × 8 bytes = 83 KB
────────────────────────────────────────────────────
总计: 249 KB (vs 原稠密矩阵 79 MB)
```

### 4.3 前向传播改造

#### 4.3.1 完整的稀疏化forward实现

```python
def forward(self, x, edge_index, edge_type, rule_ids, return_attention=False):
    """
    稀疏化的前向传播

    核心改动:
    1. query提到规则循环外，只计算1次
    2. 按关系分块计算，避免全边稠密矩阵
    3. 使用scatter_add进行稀疏累加
    """
    src, dst = edge_index
    num_nodes = x.size(0)
    num_edges = edge_index.size(1)
    device = x.device

    # ========== 优化1: query只计算1次 ==========
    query_all = self.W_q(x[dst])  # [num_edges, hidden_dim]

    # 初始化累加器
    combined_messages = torch.zeros(num_edges, self.out_dim, device=device)

    # 获取规则嵌入
    h_R = self.rule_embedding(rule_ids)  # [num_rules, hidden_dim]
    num_rules = len(rule_ids)

    if num_rules == 0:
        out = torch.zeros(num_nodes, self.out_dim, device=device)
        return (out, None) if return_attention else out

    # ========== 优化2: 按关系分块计算 ==========
    for rule_idx in range(num_rules):
        h_rule = h_R[rule_idx]  # [hidden_dim]

        for r in range(self.num_relations):
            if r not in self.relation2edges:
                continue

            # 获取当前关系的边（稀疏索引）
            edge_indices_r = self.relation2edges[r]
            src_r = self.relation2src[r]
            dst_r = self.relation2dst[r]
            num_edges_r = src_r.size(0)

            # Query: 从预计算结果中索引
            query_r = query_all[edge_indices_r]  # [num_edges_r, hidden_dim]

            # Key: 只为当前关系的边构建
            h_src_r = x[src_r]  # [num_edges_r, hidden_dim]
            h_rel_r = self.W_r[r].mean(dim=-1)  # [hidden_dim]

            # 拼接Key（小矩阵）
            key_input_r = torch.cat([
                h_src_r,
                h_rel_r.unsqueeze(0).expand(num_edges_r, -1),
                h_rule.unsqueeze(0).expand(num_edges_r, -1)
            ], dim=-1)  # [num_edges_r, hidden_dim*3]

            key_r = self.W_k(key_input_r)  # [num_edges_r, hidden_dim]

            # 注意力分数
            attn_scores_r = (query_r * key_r).sum(dim=-1) / (self.out_dim ** 0.5)
            # [num_edges_r]

            # Softmax（稀疏）
            attn_weights_r = scatter_softmax(attn_scores_r, dst_r, dim=0, dim_size=num_nodes)
            # [num_edges_r]

            # 消息计算
            msg_r = torch.matmul(h_src_r, self.W_r[r])  # [num_edges_r, hidden_dim]
            msg_r = msg_r * attn_weights_r.unsqueeze(-1)

            # ========== 优化3: 稀疏累加 ==========
            combined_messages[edge_indices_r] += msg_r

    # 平均
    combined_messages /= num_rules

    # 聚合到节点
    out = scatter_add(combined_messages, dst, dim=0, dim_size=num_nodes)
    # [num_nodes, hidden_dim]

    # 后续处理
    out = out + self.bias
    out = self.layer_norm(out)
    out = F.relu(out)
    out = self.dropout(out)

    return out
```

### 4.4 Trainer改造

```python
class RuleGNNTrainer:
    def __init__(self, model, graph, ...):
        # ... 原有代码 ...

        # 构建图结构
        self.edge_index, self.edge_type = self._build_pyg_graph()
        self.edge_index = self.edge_index.to(device)
        self.edge_type = self.edge_type.to(device)

        # 新增：为每个GNN层预构建索引
        for conv_layer in self.model.conv_layers:
            conv_layer.set_graph(self.edge_index, self.edge_type, device)
```

---

## 5. 数学等价性证明

### 5.1 公式对比

#### 稠密实现的计算

$$
\mathbf{key\_input} = \text{concat}([\mathbf{h}_{\text{src}[0:N]}, \mathbf{h}_{\text{rel}[0:N]}, \mathbf{h}_{\text{rule}[0:N]}])
$$

$$
\mathbf{k} = \mathbf{W}_k \cdot \mathbf{key\_input}
$$

其中 $N = \text{num\_edges} = 10432$

#### 稀疏实现的计算

对于每个关系 $r$，设其边集为 $E_r$，边数为 $n_r$（平均~113）：

$$
\mathbf{key\_input}_r = \text{concat}([\mathbf{h}_{\text{src}[E_r]}, \mathbf{h}_{r}, \mathbf{h}_{\text{rule}}])
$$

$$
\mathbf{k}_r = \mathbf{W}_k \cdot \mathbf{key\_input}_r
$$

最终结果：
$$
\mathbf{k} = \text{concat}([\mathbf{k}_{r_1}, \mathbf{k}_{r_2}, ..., \mathbf{k}_{r_{92}}])
$$

### 5.2 等价性

由于：
- 每条边只属于一个关系类型
- 线性变换 $\mathbf{W}_k$ 对所有边共享
- concat操作满足结合律

因此：
$$
\mathbf{k}_{\text{稠密}} = \mathbf{k}_{\text{稀疏}}
$$

**结论：数学公式完全等价，只是计算顺序不同**

---

## 6. 修改清单

### 6.1 文件修改

| 文件 | 修改内容 | 代码行数 |
|------|---------|---------|
| `rule_gnn_model.py` | `RuleAwareGraphConv.forward()` 重构 | ~60行 |
| `rule_gnn_model.py` | 新增 `set_graph()` 方法 | ~20行 |
| `rule_gnn_trainer.py` | 调用 `set_graph()` 初始化 | ~5行 |

**总修改量**：约85行

### 6.2 不需要修改的部分

- ✓ 预训练代码 (`trainer.py`, `model.py`)
- ✓ 数据加载代码 (`data.py`)
- ✓ 配置文件 (`config/*.json`)
- ✓ `RuleGNN` 类的其他方法
- ✓ 评估代码

---

## 7. 验证方案

### 7.1 正确性验证

```python
def test_sparse_equivalence():
    """验证稀疏实现与稠密实现结果一致"""

    # 使用小数据集
    num_nodes, num_edges = 100, 500
    hidden_dim = 64
    num_relations = 10
    num_rules = 5

    # 创建测试数据
    x = torch.randn(num_nodes, hidden_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))
    rule_ids = torch.arange(num_rules)

    # 稠密实现
    conv_dense = RuleAwareGraphConvDense(hidden_dim, hidden_dim, num_relations, num_rules)
    out_dense = conv_dense(x, edge_index, edge_type, rule_ids)

    # 稀疏实现
    conv_sparse = RuleAwareGraphConvSparse(hidden_dim, hidden_dim, num_relations, num_rules)
    conv_sparse.load_state_dict(conv_dense.state_dict())
    conv_sparse.set_graph(edge_index, edge_type, x.device)
    out_sparse = conv_sparse(x, edge_index, edge_type, rule_ids)

    # 验证
    assert torch.allclose(out_dense, out_sparse, atol=1e-5)
    print("✓ 稀疏实现与稠密实现结果一致!")
```

### 7.2 内存验证

```python
def test_memory_usage():
    """验证内存使用减少"""
    import torch

    torch.cuda.reset_peak_memory_stats()

    # 运行稀疏实现
    out = model(queries, edge_index, edge_type, rule_ids)

    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"峰值内存: {peak_memory:.2f} GB")

    # 预期: < 1 GB (vs 原来的 24 GB)
```

### 7.3 性能验证

```python
def test_training_convergence():
    """验证训练收敛性"""

    # 在UMLS数据集上训练
    # 对比稀疏实现和稠密实现的:
    # - MRR
    # - Hits@1, Hits@3, Hits@10
    # - 收敛速度

    # 预期: 指标应该相同或非常接近
```

---

## 8. 风险评估

| 风险项 | 级别 | 缓解措施 |
|-------|------|---------|
| 计算结果变化 | 低 | 数学等价，可对比验证 |
| Softmax数值稳定性 | 中 | 使用scatter_softmax，内置数值稳定处理 |
| 空关系处理 | 低 | 循环中跳过空关系 |
| 边界情况 | 中 | 处理num_rules=0的情况 |
| 梯度计算 | 低 | PyTorch自动微分支持scatter操作 |

---

## 9. 预期效果

### 9.1 内存

| 指标 | 稠密实现 | 稀疏实现 | 改善 |
|------|---------|---------|------|
| 峰值内存 | ~24 GB | ~160 MB | 99.3% ↓ |
| 单层GNN | ~12 GB | ~80 MB | 99.3% ↓ |
| 3层GNN | ~36 GB | ~240 MB | 99.3% ↓ |

### 9.2 能否运行

| GPU | 稠密实现 | 稀疏实现 |
|-----|---------|---------|
| 24GB (RTX 3090) | ✗ OOM | ✓ 可运行 |
| 16GB (RTX 4080) | ✗ OOM | ✓ 可运行 |
| 8GB (RTX 3070) | ✗ OOM | ✓ 可运行 |

### 9.3 训练指标

预期与稠密实现完全相同（数学等价）

---

## 10. 实施计划

### 阶段1: 基础改造（1-2小时）

1. 新增 `set_graph()` 方法
2. 修改 `forward()` 中的query计算
3. 实现按关系分块计算

### 阶段2: 验证测试（1小时）

1. 正确性测试
2. 内存测试
3. 小规模训练测试

### 阶段3: 完整训练（视需要）

1. UMLS数据集完整训练
2. 性能对比
3. 调参优化

---

## 附录A: 依赖库

```python
# 必需
torch >= 1.9.0
torch_scatter >= 2.0.9  # scatter_add, scatter_softmax

# 可选（如果使用PyG）
torch_geometric >= 2.0.0
```

## 附录B: 参考实现

- PyTorch Geometric RGCNConv: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.RGCNConv
- DGL RelGraphConv: https://docs.dgl.ai/api/python/nn.pytorch.html#relgraphconv
- CompGCN: https://github.com/malllabiisc/CompGCN
