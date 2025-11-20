# Rule-GNN 完整训练步骤指南

## 📚 目录

1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [配置文件说明](#3-配置文件说明)
4. [训练流程详解](#4-训练流程详解)
5. [Rule-GNN 核心算法步骤](#5-rule-gnn-核心算法步骤)
6. [输出文件说明](#6-输出文件说明)
7. [常见问题排查](#7-常见问题排查)
8. [进阶使用](#8-进阶使用)

---

## 1. 环境准备

### 1.1 创建 Python 环境

```bash
# 创建 conda 环境
conda create -n RulE python=3.8.0
conda activate RulE
```

### 1.2 安装依赖

```bash
cd /path/to/RulE-master
pip install -r requirements.txt
```

**核心依赖**:
- `torch>=1.10.0` - PyTorch 深度学习框架
- `torch-geometric==2.0.4` - 图神经网络框架
- `torch-scatter` - scatter 操作（PyG 依赖）
- `numpy` - 数值计算
- `tqdm` - 进度条显示

### 1.3 验证安装

```bash
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__)"
```

**预期输出**:
```
PyTorch: 1.10.0+cu113
PyG: 2.0.4
```

---

## 2. 数据准备

### 2.1 数据集结构

Rule-GNN 需要以下数据文件（以 UMLS 为例）:

```
data/umls/
├── entities.dict          # 实体字典
├── relations.dict         # 关系字典
├── train.txt             # 训练三元组
├── valid.txt             # 验证三元组
├── test.txt              # 测试三元组
└── mined_rules.txt       # 挖掘的逻辑规则
```

### 2.2 数据格式说明

#### `entities.dict`
格式: `<entity_id>\t<entity_name>`

```
0	umls:C0000005
1	umls:C0000039
2	umls:C0000052
...
```

#### `relations.dict`
格式: `<relation_id>\t<relation_name>`

```
0	umls:treats
1	umls:diagnoses
2	umls:causes
...
```

#### `train.txt / valid.txt / test.txt`
格式: `<head_name>\t<relation_name>\t<tail_name>`

```
umls:C0000005	umls:treats	umls:C0000039
umls:C0000052	umls:diagnoses	umls:C0000084
...
```

**注意**: 使用实体/关系名称，不是 ID！

#### `mined_rules.txt`
格式: `<rule_head_id> <rule_body_id_1> <rule_body_id_2> ...`

```
3 1 2
5 2 4
7 3 1 2
...
```

**示例**: `3 1 2` 表示规则 "r1 ∧ r2 → r3"

### 2.3 数据集下载

**UMLS** (医学知识图谱):
- 实体数: 135
- 关系数: 46
- 训练三元组: 5,216
- 规则数: ~600

**FB15k-237** (常识知识图谱):
- 实体数: 14,541
- 关系数: 237
- 训练三元组: 272,115
- 规则数: ~2,000

**WN18RR** (词汇知识图谱):
- 实体数: 40,943
- 关系数: 11
- 训练三元组: 86,835
- 规则数: ~500

---

## 3. 配置文件说明

### 3.1 配置文件位置

```bash
config/umls_rule_gnn_config.json
```

### 3.2 核心参数解析

#### 📁 基础配置

```json
{
    "dataset": "umls",
    "data_path": "../data/umls",
    "rule_file": "../data/umls/mined_rules.txt",
    "save_path": "umls"
}
```

- `dataset`: 数据集名称（用于日志）
- `data_path`: 数据集目录路径
- `rule_file`: 规则文件路径
- `save_path`: 输出保存路径（相对于 `src/` 目录）

#### ⚙️ 设备配置

```json
{
    "cuda": true,
    "cpu_num": 10,
    "seed": 800
}
```

- `cuda`: 是否使用 GPU（true/false）
- `cpu_num`: CPU 线程数（用于数据加载）
- `seed`: 随机种子（保证可复现）

#### 🧠 模型配置

```json
{
    "hidden_dim": 2000,
    "p_norm": 2
}
```

- `hidden_dim`: 嵌入维度（越大表达能力越强，但占用更多内存）
- `p_norm`: 距离度量的范数（RulE 使用，通常为 2）

#### 🔄 RulE 预训练参数

```json
{
    "batch_size": 256,
    "negative_sample_size": 512,
    "rule_batch_size": 256,
    "rule_negative_size": 128,
    "gamma_fact": 6,
    "gamma_rule": 8,
    "learning_rate": 0.0001,
    "max_steps": 30000
}
```

**批次设置**:
- `batch_size`: 三元组批大小
- `negative_sample_size`: 每个正三元组对应的负样本数
- `rule_batch_size`: 规则批大小
- `rule_negative_size`: 每个正规则对应的负样本数

**损失函数**:
- `gamma_fact`: 三元组损失的 margin（越大越严格）
- `gamma_rule`: 规则损失的 margin
- `weight_rule`: 规则损失权重（默认 1.0）

**优化器**:
- `learning_rate`: 学习率
- `max_steps`: 最大训练步数
- `warm_up_steps`: 学习率 warm-up 步数（默认 max_steps/2）

**负采样策略**:
- `negative_adversarial_sampling`: 是否使用对抗性负采样
- `adversarial_temperature`: 对抗性采样温度（越小越难）

#### 🌐 Rule-GNN 参数

```json
{
    "smoothing": 0.2,
    "batch_per_epoch": 1000000,
    "print_every": 10,
    "g_batch_size": 16,
    "g_lr": 0.0001,
    "dropout": 0.1,
    "rule_gnn_num_iters": 50,
    "rule_gnn_valid_every": 5
}
```

**训练控制**:
- `smoothing`: 标签平滑系数（防止过拟合）
- `batch_per_epoch`: 每个 epoch 最大批次数
- `print_every`: 每 N 个批次打印日志

**优化器**:
- `g_batch_size`: Rule-GNN 批大小
- `g_lr`: Rule-GNN 学习率
- `dropout`: Dropout 率

**训练循环**:
- `rule_gnn_num_iters`: Rule-GNN 训练 epoch 数
- `rule_gnn_valid_every`: 每 N 个 epoch 验证一次

### 3.3 参数调优建议

#### 💻 GPU 内存受限 (< 8GB)

```json
{
    "hidden_dim": 1000,
    "batch_size": 128,
    "negative_sample_size": 256,
    "g_batch_size": 8
}
```

#### ⚡ 快速测试（减少训练时间）

```json
{
    "max_steps": 1000,
    "rule_gnn_num_iters": 5,
    "valid_steps": 100
}
```

#### 🏆 追求最佳性能

```json
{
    "hidden_dim": 2000,
    "max_steps": 50000,
    "rule_gnn_num_iters": 100,
    "dropout": 0.15
}
```

---

## 4. 训练流程详解

### 4.1 完整训练（从零开始）

#### 步骤 1: 进入源码目录

```bash
cd /path/to/RulE-master/src
```

#### 步骤 2: 启动训练

```bash
python main_rule_gnn.py --init ../config/umls_rule_gnn_config.json
```

#### 步骤 3: 训练过程监控

**阶段 1: 加载数据**

```
================================================================================
Phase 1: Loading Data
================================================================================
Entities: 135
Relations: 46
Train triples: 5216
Valid triples: 652
Test triples: 661
Number of rules: 587
Max rule length: 3
```

**阶段 2: RulE 预训练**

```
================================================================================
Phase 2: RulE Pre-training (RotatE + Rule Embeddings)
================================================================================
Starting RulE pre-training...

Step 100/30000 | Loss: 2.345 | Fact Loss: 1.234 | Rule Loss: 1.111
Step 200/30000 | Loss: 2.123 | Fact Loss: 1.112 | Rule Loss: 1.011
...
Step 1000/30000 | Valid MRR: 0.523 | Hits@10: 0.721
...
Step 30000/30000 | Valid MRR: 0.867 | Hits@10: 0.943

RulE pre-training completed!
Evaluating RulE pre-training results...
Valid MRR: 0.867 | Hits@1: 0.792 | Hits@3: 0.904 | Hits@10: 0.943
Test MRR: 0.859 | Hits@1: 0.783 | Hits@3: 0.897 | Hits@10: 0.938
```

**阶段 3: 导出嵌入**

```
================================================================================
Phase 3: Exporting Embeddings for Rule-GNN
================================================================================
Exported entity embeddings: torch.Size([135, 4000])
Exported relation embeddings: torch.Size([46, 2000])
Exported rule embeddings: torch.Size([587, 2000])
```

**注意**: `entity_embedding` 维度是 `hidden_dim * 2`（复数嵌入）

**阶段 4: Rule-GNN 训练**

```
================================================================================
Phase 4: Rule-GNN Training (replaces Grounding)
================================================================================
GNN layers (= max rule length): 3
Rule-GNN parameters: 24,567,890

Loading pretrained embeddings into Rule-GNN...
Starting Rule-GNN training...

Epoch 1/50
Training: 100%|████████████| 326/326 [00:45<00:00,  7.2it/s, loss=1.234]
Train Loss: 1.2345

Epoch 5/50
Valid MRR: 0.912 | Hits@1: 0.856 | Hits@3: 0.943 | Hits@10: 0.978
Saved best model to umls/rule_gnn_best.pt

...

Epoch 50/50
Valid MRR: 0.941 | Hits@1: 0.897 | Hits@3: 0.967 | Hits@10: 0.989
```

**阶段 5: 最终测试**

```
================================================================================
Final Test Evaluation
================================================================================
Test MRR: 0.938 | Hits@1: 0.893 | Hits@3: 0.964 | Hits@10: 0.987
Test MR: 1.82
```

### 4.2 跳过预训练（使用已有 checkpoint）

#### 前提条件

确保存在 RulE 预训练 checkpoint:

```bash
ls -lh umls/rule_checkpoint
# 应该看到文件存在
```

#### 启动训练

```bash
python main_rule_gnn.py --init ../config/umls_rule_gnn_config.json --skip_pretrain
```

**训练流程**:
```
阶段 1: 加载数据
阶段 2: 跳过 RulE 预训练（从 checkpoint 加载）
阶段 3: 导出嵌入
阶段 4: Rule-GNN 训练
阶段 5: 保存结果
```

### 4.3 训练时间估算

#### UMLS 数据集（135 实体，5K 三元组）

| 阶段 | GPU (V100) | CPU (10核) |
|-----|-----------|-----------|
| RulE 预训练 (30K steps) | ~30 分钟 | ~3 小时 |
| Rule-GNN 训练 (50 epochs) | ~15 分钟 | ~1.5 小时 |
| **总计** | ~45 分钟 | ~4.5 小时 |

#### FB15k-237 数据集（14K 实体，272K 三元组）

| 阶段 | GPU (V100) | CPU (10核) |
|-----|-----------|-----------|
| RulE 预训练 (50K steps) | ~4 小时 | ~2 天 |
| Rule-GNN 训练 (50 epochs) | ~2 小时 | ~8 小时 |
| **总计** | ~6 小时 | ~2.5 天 |

---

## 5. Rule-GNN 核心算法步骤

### 5.1 设计模式说明

Rule-GNN 采用 **全实体打分模式（Full Ranking）**，与 RulE 原始 Grounding 阶段一致：

| 特性 | Rule-GNN（Grounding 模式） | KGE 负采样模式 |
|------|--------------------------|--------------|
| 打分实体数 | 所有实体（如 135） | 1 + neg_size（如 129） |
| 标签类型 | 多热标签（多个正确答案） | 单热（第 0 个是正样本） |
| 损失函数 | BCEWithLogitsLoss | CrossEntropyLoss |
| 适用场景 | 小图、多答案查询 | 大图、单答案查询 |

### 5.2 训练阶段完整步骤

#### 输入数据

```
- queries: 查询 (h, r) [batch_size, 2]
- target: 多热标签 [batch_size, num_entities]
  - target[i][j] = 1 表示实体 j 是查询 i 的正确答案
- edge_index: 图的边索引 [2, num_edges]
- edge_type: 边的类型 [num_edges]
```

#### 步骤 1: 初始化节点特征

```python
# 从预训练的 RulE 加载实体嵌入
h = entity_embedding.weight  # [num_entities, hidden_dim]
```

**说明**：
- 实体嵌入来自 RulE 预训练阶段
- 这是所有实体的初始表示

#### 步骤 2: 获取激活规则

```python
# 根据查询关系，找到相关的规则
active_rules = set()
for r in query_relations:
    if r in relation2rules:
        for rule in relation2rules[r]:
            active_rules.add(rule_id)

rule_ids = list(active_rules)  # [num_active_rules]
```

**示例**：
- 查询关系 `treats`（治疗）
- 相关规则：
  - 规则 1: `diagnoses ∧ treats → treats`
  - 规则 2: `causes ∧ treats → treats`
- `rule_ids = [1, 2]`

#### 步骤 3: GNN 消息传递（多层）

```python
for layer_idx in range(num_layers):
    h = gnn_layer(h, edge_index, edge_type, rule_ids)
```

**每层 GNN 做的事情**：

##### 3.1 计算规则感知的注意力权重

```python
for rule_id in rule_ids:
    # 获取规则嵌入
    h_rule = rule_embedding[rule_id]  # [hidden_dim]

    # Query: 目标节点特征
    query = W_q(h[dst])  # [num_edges, hidden_dim]

    # Key: 拼接 [源节点; 关系嵌入; 规则嵌入]
    key_input = concat([h[src], relation_emb, rule_emb])  # [num_edges, hidden_dim*3]
    key = W_k(key_input)  # [num_edges, hidden_dim]

    # 注意力分数
    attn_scores = (query * key).sum(dim=-1) / sqrt(hidden_dim)
    attn_weights = softmax(attn_scores, per_dst_node)  # [num_edges]
```

**关键点**：
- 注意力权重由**规则嵌入调控**
- 符合规则的边获得更高的注意力

##### 3.2 计算消息

```python
# 对每条边计算消息
for edge_idx in range(num_edges):
    src_node = edge_index[0, edge_idx]
    rel_type = edge_type[edge_idx]

    # 消息 = 关系变换矩阵 * 源节点特征 * 注意力权重
    msg = W_r[rel_type] @ h[src_node] * attn_weights[edge_idx]
```

##### 3.3 聚合消息到目标节点

```python
# 使用 scatter_add 聚合
h_new = scatter_add(messages, dst, dim=0)  # [num_entities, hidden_dim]

# 添加偏置 + LayerNorm + ReLU + Dropout
h_new = h_new + bias
h_new = layer_norm(h_new)
h_new = relu(h_new)
h_new = dropout(h_new)
```

##### 3.4 多规则聚合

```python
# 平均所有规则的消息
h_final = mean([h_rule1, h_rule2, ...])  # [num_entities, hidden_dim]
```

**图解**：

```
     Layer 1                 Layer 2                 Layer 3
        ↓                       ↓                       ↓
    [Entity Emb]    →    [Updated Emb]    →    [Final Emb]
        ↓                       ↓                       ↓
   规则1 + 规则2            规则1 + 规则2           规则1 + 规则2
   消息传递                 消息传递                消息传递
```

#### 步骤 4: 对所有实体打分

```python
# 提取查询头实体的表示
h_heads = h[queries[:, 0]]  # [batch_size, hidden_dim]

# 所有实体的表示
h_tails = h  # [num_entities, hidden_dim]

# 拼接并通过 MLP 打分
# h_heads: [batch_size, 1, hidden_dim] 扩展
# h_tails: [1, num_entities, hidden_dim] 扩展
combined = concat([h_heads, h_tails], dim=-1)  # [batch_size, num_entities, hidden_dim*2]

scores = MLP(combined).squeeze(-1)  # [batch_size, num_entities]
```

**说明**：
- 对每个查询，计算所有实体作为尾实体的得分
- 输出维度：`[batch_size, num_entities]`

#### 步骤 5: 计算损失

```python
# target: 多热标签 [batch_size, num_entities]
# 标签平滑
if smoothing > 0:
    smooth_target = target * (1.0 - smoothing) + smoothing / num_entities
    loss = BCEWithLogitsLoss(scores, smooth_target)
else:
    loss = BCEWithLogitsLoss(scores, target)
```

**多热标签示例**：
```
查询: (实体5, 关系3, ?)
正确答案: 实体 10, 12, 15

target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, ...]
                                      ↑     ↑        ↑
                                     10    12       15
```

#### 步骤 6: 反向传播

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 5.3 评估阶段步骤

#### 步骤 1-4: 与训练相同

获取所有实体的得分 `scores: [batch_size, num_entities]`

#### 步骤 5: 过滤已知三元组

```python
# filter_mask: 标记所有已知三元组（训练+验证+测试）
# 将已知答案的得分设为负无穷，避免它们影响排名
scores = scores.masked_fill(filter_mask, -1e9)
```

#### 步骤 6: 计算排名

```python
for i in range(batch_size):
    true_tail = true_tails[i]
    true_score = scores[i, true_tail]

    # 排名 = 比真实答案得分高的实体数 + 1
    rank = (scores[i] > true_score).sum() + 1

    ranks.append(rank)
    reciprocal_ranks.append(1.0 / rank)
```

#### 步骤 7: 计算指标

```python
metrics = {
    'MRR': mean(reciprocal_ranks),      # 平均倒数排名
    'MR': mean(ranks),                   # 平均排名
    'HITS@1': mean(ranks <= 1),          # Top-1 准确率
    'HITS@3': mean(ranks <= 3),          # Top-3 准确率
    'HITS@10': mean(ranks <= 10)         # Top-10 准确率
}
```

### 5.4 与原始 RulE Grounding 的对比

| 步骤 | RulE Grounding | Rule-GNN |
|------|---------------|----------|
| **规则处理** | 显式枚举每条规则的 grounding 路径 | 隐式通过 GNN 消息传递 |
| **路径遍历** | r1 → r2 → r3 按顺序遍历 | 多层 GNN，每层聚合所有关系 |
| **计算方式** | 稀疏矩阵乘法（逐关系） | 注意力加权消息传递 |
| **规则权重** | MLP 学习每条规则的权重 | 注意力机制自动调控 |
| **可并行性** | 规则之间顺序处理 | 所有规则并行计算 |

### 5.5 为什么用 Grounding 模式而不是负采样？

1. **数据兼容**：`TrainDataset` 返回多热标签 `target`，天然支持 Grounding 模式

2. **多答案问题**：
   - 知识图谱中 `(h, r, ?)` 可能有多个正确答案
   - 例如："北京的大学" → 清华、北大、人大...
   - 负采样只能处理单答案，全实体打分能处理多答案

3. **与 RulE 一致**：保持训练方式一致，便于对比

4. **计算可行**：UMLS 只有 135 个实体，全打分计算量可接受

### 5.6 完整代码流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    Rule-GNN 训练流程                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. 数据加载                                                  │
│    - TrainDataset 返回: (all_h, all_r, all_t, target, edges) │
│    - target: 多热标签 [batch_size, num_entities]             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 获取激活规则                                              │
│    - 根据查询关系找相关规则                                   │
│    - rule_ids = [rule1, rule2, ...]                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. GNN 消息传递（重复 num_layers 次）                         │
│    ┌─────────────────────────────────────────────────────┐  │
│    │ For each rule:                                      │  │
│    │   - 计算规则感知的注意力权重                          │  │
│    │   - 消息 = W_r * h[src] * attention                 │  │
│    │   - 聚合到目标节点: h[dst] = sum(messages)          │  │
│    └─────────────────────────────────────────────────────┘  │
│    - 多规则取平均                                           │
│    - LayerNorm + ReLU + Dropout                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 全实体打分                                                │
│    - h_heads = h[queries[:, 0]]                             │
│    - scores = MLP(concat(h_heads, h_all))                   │
│    - 输出: [batch_size, num_entities]                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. 计算损失                                                  │
│    - 标签平滑: smooth_target = target * 0.8 + 0.2/N         │
│    - loss = BCEWithLogitsLoss(scores, smooth_target)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. 反向传播                                                  │
│    - optimizer.zero_grad()                                  │
│    - loss.backward()                                        │
│    - optimizer.step()                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 输出文件说明

### 6.1 输出目录结构

训练完成后，输出目录结构如下:

```
{save_path}/  (例如 src/umls/)
├── config.json                # 训练配置备份
├── run.log                    # 完整训练日志
├── rule_checkpoint            # RulE 预训练模型
├── rule_gnn_best.pt           # Rule-GNN 最佳模型
└── rule_gnn_results.json      # 测试结果
```

### 5.2 文件详解

#### `config.json`

保存的训练配置（JSON 格式），用于复现实验:

```json
{
    "dataset": "umls",
    "data_path": "../data/umls",
    "hidden_dim": 2000,
    "max_steps": 30000,
    "rule_gnn_num_iters": 50,
    ...
}
```

#### `run.log`

完整的训练日志（文本格式）:

```
2024-11-19 10:00:00,123 - INFO - ================================================================================
2024-11-19 10:00:00,124 - INFO - Rule-GNN Training
2024-11-19 10:00:00,125 - INFO - ================================================================================
2024-11-19 10:00:01,456 - INFO - Phase 1: Loading Data
...
```

#### `rule_checkpoint`

RulE 预训练模型（PyTorch checkpoint）:

```python
checkpoint = torch.load('rule_checkpoint')
# 包含:
# - 'model': 模型 state_dict
# - 'entity_embedding.weight': 实体嵌入
# - 'relation_embedding.weight': 关系嵌入
# - 'rule_emb.weight': 规则嵌入
```

#### `rule_gnn_best.pt`

Rule-GNN 最佳模型（PyTorch checkpoint）:

```python
checkpoint = torch.load('rule_gnn_best.pt')
# 包含:
# - 'model_state_dict': Rule-GNN 模型 state_dict
```

#### `rule_gnn_results.json`

测试集结果（JSON 格式）:

```json
{
    "dataset": "umls",
    "hidden_dim": 2000,
    "num_layers": 3,
    "test_metrics": {
        "MRR": 0.938,
        "MR": 1.82,
        "HITS@1": 0.893,
        "HITS@3": 0.964,
        "HITS@10": 0.987
    },
    "timestamp": "2024-11-19 12:30:45"
}
```

### 5.3 使用训练好的模型

#### 加载 Rule-GNN 模型

```python
import torch
from rule_gnn_model import RuleGNN

# 创建模型
model = RuleGNN(
    num_entities=135,
    num_relations=46*2,
    num_rules=587,
    hidden_dim=2000,
    num_layers=3,
    dropout=0.1
)

# 加载 checkpoint
checkpoint = torch.load('umls/rule_gnn_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

#### 进行推理

```python
# 准备查询: (head, relation)
queries = torch.tensor([[10, 5]])  # head=10, relation=5

# 准备图数据
edge_index = ...  # [2, num_edges]
edge_type = ...   # [num_edges]

# 激活的规则
rule_ids = torch.tensor([0, 1, 2, 10, 15])

# 前向传播
with torch.no_grad():
    scores = model(queries, edge_index, edge_type, rule_ids)
    # scores: [1, num_entities]

    # 获取 top-10 预测
    top10_scores, top10_entities = torch.topk(scores[0], k=10)

    print("Top-10 预测:")
    for i, (entity, score) in enumerate(zip(top10_entities, top10_scores)):
        print(f"{i+1}. Entity {entity.item()}: {score.item():.4f}")
```

---

## 6. 常见问题排查

### 6.1 CUDA 内存不足

**错误信息**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**解决方案**:

1. 减小批大小:
```json
{
    "batch_size": 128,        // 原 256
    "g_batch_size": 8         // 原 16
}
```

2. 减小嵌入维度:
```json
{
    "hidden_dim": 1000        // 原 2000
}
```

3. 使用 CPU 训练:
```json
{
    "cuda": false
}
```

### 6.2 找不到 rule_checkpoint

**错误信息**:
```
ERROR - RulE checkpoint not found: umls/rule_checkpoint
ERROR - Please run without --skip_pretrain first
```

**解决方案**:

1. 首次训练不要使用 `--skip_pretrain`:
```bash
python main_rule_gnn.py --init ../config/umls_rule_gnn_config.json
```

2. 或者修改配置文件中的 `save_path` 指向已有 checkpoint 目录

### 6.3 PyG 安装失败

**错误信息**:
```
ERROR: Could not find a version that satisfies the requirement torch-geometric
```

**解决方案**:

按照 PyG 官方文档安装:

```bash
# 1. 确认 PyTorch 版本
python -c "import torch; print(torch.__version__)"
# 例如: 1.10.0+cu113

# 2. 安装对应版本的 PyG
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```

### 6.4 训练卡住不动

**现象**: 训练进度条长时间不更新

**排查步骤**:

1. 检查数据加载器:
```json
{
    "cpu_num": 4  // 减少数据加载线程数
}
```

2. 检查是否死锁:
```bash
# 查看 GPU 使用情况
nvidia-smi

# 查看进程
ps aux | grep python
```

3. 添加调试日志:
```python
# 在 rule_gnn_trainer.py 的训练循环中添加
print(f"Batch {batch_idx}, Loss: {loss.item()}")
```

### 6.5 验证指标异常低

**现象**: Valid MRR < 0.1

**可能原因**:

1. **规则文件错误**: 检查 `mined_rules.txt` 格式
2. **数据泄漏**: 确认训练/验证/测试集划分正确
3. **学习率过大**: 降低 `g_lr`
```json
{
    "g_lr": 0.00005  // 原 0.0001
}
```

### 6.6 OOM (Out of Memory) 在 CPU 上

**错误信息**:
```
Killed
```

**解决方案**:

1. 减小批大小:
```json
{
    "g_batch_size": 4
}
```

2. 减少规则数量（在数据预处理阶段过滤低置信度规则）

---

## 7. 进阶使用

### 7.1 多数据集训练

#### 创建新配置文件

```bash
cp config/umls_rule_gnn_config.json config/fb15k237_rule_gnn_config.json
```

#### 修改配置

```json
{
    "dataset": "fb15k237",
    "data_path": "../data/fb15k237",
    "rule_file": "../data/fb15k237/mined_rules.txt",
    "save_path": "fb15k237_output",

    "hidden_dim": 1000,
    "max_steps": 50000,
    "smoothing": 0.5
}
```

#### 启动训练

```bash
python main_rule_gnn.py --init ../config/fb15k237_rule_gnn_config.json
```

### 7.2 超参数网格搜索

创建搜索脚本 `grid_search.sh`:

```bash
#!/bin/bash

for hidden_dim in 500 1000 2000; do
    for dropout in 0.1 0.2 0.3; do
        for lr in 0.0001 0.00005; do
            save_path="grid_search/h${hidden_dim}_d${dropout}_lr${lr}"

            # 修改配置文件
            cat config/umls_rule_gnn_config.json | \
                jq ".hidden_dim = $hidden_dim | .dropout = $dropout | .g_lr = $lr | .save_path = \"$save_path\"" \
                > config/temp_config.json

            # 训练
            python src/main_rule_gnn.py --init config/temp_config.json
        done
    done
done

# 找出最佳结果
python scripts/find_best_model.py grid_search/
```

### 7.3 可视化训练过程

#### 使用 TensorBoard

在 `rule_gnn_trainer.py` 中添加:

```python
from torch.utils.tensorboard import SummaryWriter

class RuleGNNTrainer:
    def __init__(self, ...):
        ...
        self.writer = SummaryWriter(log_dir=os.path.join(args.save_path, 'tensorboard'))

    def train_epoch(self, ...):
        ...
        self.writer.add_scalar('Loss/train', avg_loss, epoch)

    def evaluate(self, ...):
        ...
        self.writer.add_scalar('MRR/valid', metrics['MRR'], epoch)
```

启动 TensorBoard:

```bash
tensorboard --logdir=umls/tensorboard
# 访问 http://localhost:6006
```

### 7.4 模型集成（Ensemble）

```python
import torch
from rule_gnn_model import RuleGNN

# 加载多个模型
models = []
for i in range(5):
    model = RuleGNN(...)
    checkpoint = torch.load(f'ensemble/model_{i}/rule_gnn_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    models.append(model)

# 集成推理
def ensemble_predict(queries, edge_index, edge_type, rule_ids):
    all_scores = []

    with torch.no_grad():
        for model in models:
            scores = model(queries, edge_index, edge_type, rule_ids)
            all_scores.append(scores)

    # 平均分数
    ensemble_scores = torch.stack(all_scores).mean(dim=0)
    return ensemble_scores
```

### 7.5 导出嵌入用于下游任务

```python
import torch
import numpy as np

# 加载模型
model = RuleGNN(...)
checkpoint = torch.load('umls/rule_gnn_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# 导出实体嵌入
entity_embeddings = model.entity_embedding.weight.data.cpu().numpy()
np.save('entity_embeddings.npy', entity_embeddings)

# 导出关系嵌入
relation_embeddings = model.relation_embedding.weight.data.cpu().numpy()
np.save('relation_embeddings.npy', relation_embeddings)

# 用于下游任务（如实体分类、聚类等）
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10)
entity_clusters = kmeans.fit_predict(entity_embeddings)
```

---

## 8. 性能优化技巧

### 8.1 混合精度训练（FP16）

```python
# 在 rule_gnn_trainer.py 中添加
from torch.cuda.amp import autocast, GradScaler

class RuleGNNTrainer:
    def __init__(self, ...):
        ...
        self.scaler = GradScaler()

    def train_epoch(self, optimizer, args):
        ...
        for batch in train_loader:
            with autocast():
                scores = self.model(...)
                loss = criterion(scores, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
```

**效果**: 训练速度提升 2-3 倍，内存占用减半

### 8.2 梯度累积

```python
accumulation_steps = 4

for batch_idx, batch in enumerate(train_loader):
    loss = ...
    loss = loss / accumulation_steps
    loss.backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**效果**: 等效于更大的批大小，不增加内存占用

### 8.3 数据预加载

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=8,       # 增加数据加载线程
    pin_memory=True,     # 加速 CPU->GPU 传输
    prefetch_factor=2    # 预加载批次数
)
```

---

## 9. 实验复现清单

### ✅ 复现 UMLS 数据集结果

- [ ] 环境安装完成（Python 3.8 + PyTorch + PyG）
- [ ] 数据集下载并放置在 `data/umls/`
- [ ] 配置文件检查（`config/umls_rule_gnn_config.json`）
- [ ] 启动训练: `python main_rule_gnn.py --init ../config/umls_rule_gnn_config.json`
- [ ] 训练完成，验证指标 MRR > 0.93
- [ ] 测试集结果保存在 `rule_gnn_results.json`

### 📊 预期结果对比

| 模型 | Valid MRR | Test MRR | Test Hits@10 |
|-----|-----------|----------|--------------|
| RulE | 0.867 | 0.859 | 0.938 |
| **Rule-GNN** | **0.941** | **0.938** | **0.987** |
| 提升 | +7.4% | +7.9% | +4.9% |

---

## 10. 总结

### 关键要点

1. **数据准备**: 确保数据格式正确（实体/关系字典 + 三元组 + 规则）
2. **配置调优**: 根据硬件资源调整 `hidden_dim`, `batch_size`, `g_batch_size`
3. **训练监控**: 观察日志中的 Valid MRR，确保模型收敛
4. **结果验证**: 对比测试集指标，确认性能提升

### 下一步

- 📖 阅读 [Rule-GNN代码详解.md](Rule-GNN代码详解.md) 理解实现细节
- 🔬 尝试在其他数据集（FB15k-237, WN18RR）上训练
- 🚀 探索超参数调优和模型改进

---

**文档版本**: v1.0
**更新时间**: 2024-11-19
**维护者**: Rule-GNN Team
