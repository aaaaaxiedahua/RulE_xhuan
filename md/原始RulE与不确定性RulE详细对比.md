# 原始RulE与不确定性RulE详细对比文档

本文档详细对比原始RulE模型和引入不确定性建模的新RulE模型，包括架构、步骤、公式和参数。

---

## 目录
1. [原始RulE模型详解](#原始rule模型详解)
2. [不确定性RulE模型详解](#不确定性rule模型详解)
3. [对比总结](#对比总结)

---

# 原始RulE模型详解

## 一、模型架构

### 1.1 核心组件

```python
class RulE(nn.Module):
    def __init__(self):
        # 组件1: 实体嵌入
        self.entity_embedding = nn.Parameter(
            torch.randn(num_entities, hidden_dim * 2)  # 复数表示：实部+虚部
        )

        # 组件2: 关系嵌入
        self.relation_embedding = nn.Parameter(
            torch.randn(num_relations, hidden_dim)  # 相位表示
        )

        # 组件3: 规则嵌入
        self.rule_emb = nn.Parameter(
            torch.randn(num_rules, rule_dim)
        )

        # 组件4: MLP规则特征（Grounding阶段训练）
        self.mlp_feature = nn.Parameter(
            torch.randn(num_rules, mlp_rule_dim)
        )

        # 组件5: 评分MLP（Grounding阶段训练）
        self.score_model = MLP(mlp_rule_dim, [hidden_size, 1])
```

### 1.2 参数表

| 参数名 | 维度 | 说明 | 训练阶段 |
|--------|------|------|----------|
| `entity_embedding` | `[num_entities, hidden_dim*2]` | 实体的复数嵌入 | 预训练 |
| `relation_embedding` | `[num_relations, hidden_dim]` | 关系的相位嵌入 | 预训练 |
| `rule_emb` | `[num_rules, rule_dim]` | 规则嵌入 | 预训练 |
| `mlp_feature` | `[num_rules, mlp_rule_dim]` | 规则MLP特征 | Grounding |
| `score_model` | MLP网络 | 候选实体打分网络 | Grounding |

---

## 二、预训练阶段

### 2.1 目标

联合学习实体嵌入、关系嵌入和规则嵌入。

### 2.2 步骤流程

#### Step 1: 知识图谱嵌入损失 (KGE Loss)

**输入**：三元组 `(h, r, t)`

**公式**：
```
1. 获取嵌入：
   h_emb = entity_embedding[h]        # [hidden_dim*2]
   r_emb = relation_embedding[r]      # [hidden_dim]
   t_emb = entity_embedding[t]        # [hidden_dim*2]

2. 转换关系为复数（RotatE）：
   r_phase = r_emb                    # 相位
   r_re = cos(r_phase)                # 实部
   r_im = sin(r_phase)                # 虚部

   h_re, h_im = split(h_emb)          # 分离头实体的实部虚部
   t_re, t_im = split(t_emb)

3. 复数旋转：
   score_re = h_re * r_re - h_im * r_im - t_re
   score_im = h_re * r_im + h_im * r_re - t_im

4. RotatE距离：
   d = ||score_re||² + ||score_im||²  # L2距离
   score = γ_fact - d                  # γ_fact是margin

5. 对抗负采样：
   neg_samples = sample_negative(h, r, t)

   for each neg (h', r, t') or (h, r, t'):
       score_neg = γ_fact - d_neg

   # 负样本权重（对抗）
   p_neg = softmax(α * score_neg)     # α是对抗温度

6. KGE损失：
   loss_kge = -log(σ(score_pos)) - Σ p_neg * log(σ(-score_neg))
```

**参数**：
- `γ_fact`: margin (典型值: 6.0)
- `α`: 对抗温度 (典型值: 0.5)
- `negative_sample_size`: 负样本数 (典型值: 256)

---

#### Step 2: 规则嵌入损失 (Rule Loss)

**输入**：规则 `[rule_id, r_head, r_body_1, r_body_2, ...]`

**公式**：
```
1. 获取规则组成部分：
   rule = [rule_id, r_head, r1, r2, ..., rL]

   r_head_emb = relation_embedding[r_head]     # [hidden_dim]
   r_body_embs = [relation_embedding[ri] for ri in [r1, r2, ..., rL]]
   R_i = rule_emb[rule_id]                     # [rule_dim]

2. 聚合规则体的关系嵌入：
   r_body_sum = Σ r_body_embs[j]              # [hidden_dim]
                j=1..L

3. 计算规则距离：
   d_rule = ||r_body_sum + R_i - r_head_emb||  # L2范数

4. 规则置信度（关键）：
   w_i = γ_rule - d_rule                       # 标量

   注意：这是一个确定性的值！

5. 负采样：
   # 随机替换规则体中的某个关系
   neg_rule = [rule_id, r_head, r1, r_neg, r3, ...]
   d_rule_neg = ||r_body_sum_neg + R_i - r_head_emb||
   w_neg = γ_rule - d_rule_neg

6. 规则损失：
   loss_rule = -log(σ(w_i)) - Σ log(σ(-w_neg))
```

**参数**：
- `γ_rule`: margin (典型值: 6.0)
- `rule_negative_size`: 负样本数 (典型值: 128)

---

#### Step 3: 总损失

**公式**：
```
loss_total = loss_kge + λ_rule * loss_rule
```

**参数**：
- `λ_rule`: 规则损失权重 (典型值: 1.0)

---

### 2.3 预训练输出

保存到 `checkpoint`:
```python
{
    'entity_embedding': [num_entities, hidden_dim*2],
    'relation_embedding': [num_relations, hidden_dim],
    'rule_emb': [num_rules, rule_dim]
}
```

---

## 三、Grounding阶段

### 3.1 目标

学习如何将规则grounding到知识图谱上，并对候选实体打分。

### 3.2 冻结参数

```python
# 冻结预训练的嵌入
entity_embedding.requires_grad = False
relation_embedding.requires_grad = False
rule_emb.requires_grad = False
```

### 3.3 训练参数

```python
# 只训练这些
mlp_feature.requires_grad = True
score_model.requires_grad = True
```

### 3.4 步骤流程

#### Step 1: 检索适用规则

**输入**：查询 `(h, r, ?)`

**操作**：
```python
applicable_rules = relation2rules[r]
# 返回所有头部是r的规则
# 例如: [rule_15, rule_42, rule_88]
```

---

#### Step 2: 规则Grounding

**对每个规则进行图传播**：

**公式**：
```
规则: r1 ∧ r2 ∧ ... ∧ rL → r_head

1. 初始化：
   current_entities = {h}              # 从查询头实体开始
   current_scores = {h: 1.0}           # 初始分数

2. 第一跳（沿着r1）：
   next_entities = {}
   for each entity e in current_entities:
       for each neighbor n connected by r1:
           next_entities[n] += current_scores[e]

   current_entities = next_entities

3. 第二跳（沿着r2）：
   next_entities = {}
   for each entity e in current_entities:
       for each neighbor n connected by r2:
           next_entities[n] += current_scores[e]

   current_entities = next_entities

4. ... 重复L跳

5. 输出：
   grounding_counts = current_entities  # [num_entities]
   # grounding_counts[e] = 从h经过r1∧r2∧...∧rL到达e的路径数
```

**代码实现**：
```python
grounding_counts = graph.grounding(
    h, r_head, [r1, r2, ..., rL],
    edges_to_remove=training_edges  # 移除训练边防止作弊
)
# 返回: [num_entities] 张量
```

---

#### Step 3: 计算规则置信度

**这里使用预训练学到的置信度**：

**公式**：
```
1. 获取规则嵌入：
   R_i = rule_emb[rule_id]
   r_body_embs = [relation_embedding[rj] for rj in [r1, r2, ..., rL]]
   r_head_emb = relation_embedding[r_head]

2. 计算规则距离：
   r_body_sum = Σ r_body_embs[j]
   d_rule = ||r_body_sum + R_i - r_head_emb||

3. 规则置信度：
   w_i = γ_rule - d_rule               # 标量，确定性
```

**关键点**：
- w_i 是一个固定的标量
- 无法表达不确定性

---

#### Step 4: 构建软多热编码

**公式**：
```
1. 初始化候选实体特征：
   candidate_features = zeros([num_entities, mlp_rule_dim])

2. 对每个grounded的规则：
   for each rule_i with grounding_counts_i:
       # 获取规则的MLP特征
       rule_feature_i = mlp_feature[rule_i]  # [mlp_rule_dim]

       # 对每个候选实体
       for each entity e:
           if grounding_counts_i[e] > 0:
               # 累加特征
               candidate_features[e] += w_i * grounding_counts_i[e] * rule_feature_i

公式形式：
   feature[e] = Σ (w_i × count_i[e] × mlp_feature[i])
                i∈grounded_rules
```

**关键点**：
- w_i 作为规则的权重
- 路径数 count_i[e] 表示该规则有多少条路径到达实体e
- mlp_feature[i] 是规则的可学习特征向量

---

#### Step 5: 聚合与打分

**公式**：
```
1. 聚合规则特征：
   # 平均所有规则的贡献
   aggregated_features = Σ feature[e] / num_grounded_rules

2. 层归一化：
   aggregated_features = LayerNorm(aggregated_features)

3. 激活：
   aggregated_features = ReLU(aggregated_features)

4. MLP打分：
   score[e] = score_model(aggregated_features[e])
   # score_model是1-2层的MLP
```

---

#### Step 6: 计算损失

**公式**：
```
1. 获取真实答案：
   labels = query_tails                # [batch_size]

2. 标签平滑交叉熵：
   # 标签平滑：防止过拟合
   smooth_labels = (1 - ε) * one_hot(labels) + ε / num_entities

   loss = CrossEntropy(scores, smooth_labels)
```

**参数**：
- `ε`: 平滑系数 (典型值: 0.2)

---

### 3.5 Grounding输出

保存到 `grounding.pt`:
```python
{
    'mlp_feature': [num_rules, mlp_rule_dim],
    'score_model.state_dict': {...}
}
```

---

## 四、推理阶段

### 4.1 完整流程

**输入**：查询 `(h, r, ?)`

**步骤**：

```
1. 检索适用规则：
   applicable_rules = relation2rules[r]

2. 对每个规则进行grounding：
   for each rule in applicable_rules:
       grounding_counts = graph.grounding(h, rule.body)

       # 计算规则置信度
       w_i = γ_rule - d_rule           # 使用预训练的

       # 累加特征
       for each entity e:
           if grounding_counts[e] > 0:
               features[e] += w_i * grounding_counts[e] * mlp_feature[rule.id]

3. 聚合与打分：
   aggregated = LayerNorm(ReLU(features))
   scores = score_model(aggregated)    # [num_entities]

4. 【可选】结合KGE分数：
   kge_scores = compute_rotate_score(h, r, all_entities)
   final_scores = scores + α * kge_scores

---

### 4.2 KGE分数结合详解

#### 4.2.1 结合公式

**完整打分公式** (src/trainer.py:669):
```python
final_scores = rule_scores + α * kge_scores
```

其中:
- `rule_scores`: 规则推理分数 [batch_size, num_entities]
- `kge_scores`: RotatE嵌入分数 [batch_size, num_entities]
- `α`: KGE权重超参数 (典型值: 2.0-5.0)

#### 4.2.2 RotatE分数计算

**输入**: 查询 `(h, r, ?)`

**步骤**:
```
1. 准备候选实体:
   all_candidates = [0, 1, 2, ..., num_entities-1]  # 所有实体

2. 获取嵌入:
   h_emb = entity_embedding[h]                      # [hidden_dim*2] 复数
   r_emb = relation_embedding[r]                    # [hidden_dim] 相位
   t_embs = entity_embedding[all_candidates]        # [num_entities, hidden_dim*2]

3. 分离复数实部和虚部:
   h_re, h_im = h_emb[:hidden_dim], h_emb[hidden_dim:]
   t_re, t_im = t_embs[:, :hidden_dim], t_embs[:, hidden_dim:]

4. 将关系转换为复数(旋转):
   phase = r_emb / (embedding_range / π)
   r_re = cos(phase)                                # [hidden_dim]
   r_im = sin(phase)                                # [hidden_dim]

5. 复数乘法 (h ∘ r):
   # (h_re + i*h_im) × (r_re + i*r_im)
   score_re = h_re * r_re - h_im * r_im             # [hidden_dim]
   score_im = h_re * r_im + h_im * r_re             # [hidden_dim]

6. 与尾实体计算距离:
   diff_re = score_re - t_re                        # [num_entities, hidden_dim]
   diff_im = score_im - t_im                        # [num_entities, hidden_dim]

7. L2范数距离:
   dist = sqrt(diff_re² + diff_im²)                 # [num_entities, hidden_dim]
   dist = sum(dist, dim=-1)                         # [num_entities] 求和

8. 转换为分数 (距离越小分数越高):
   kge_scores = γ_fact - dist                       # [num_entities]
```

#### 4.2.3 结合示例

**场景**: 查询 `(drug_aspirin, treats_symptom, ?)`

假设有3个候选症状:
```
候选1: symptom_chest_pain
候选2: symptom_headache
候选3: symptom_nausea
```

**规则推理分数** (来自grounding):
```
rule_scores = [
    0.87,   # symptom_chest_pain (规则强支持)
    -5.0,   # symptom_headache (无规则支持)
    -5.0,   # symptom_nausea (无规则支持)
]
```

**KGE分数** (来自RotatE):
```
# 计算每个候选的嵌入距离
dist_chest_pain = 5.28    → kge_score = 6.0 - 5.28 = 0.72
dist_headache = 5.85      → kge_score = 6.0 - 5.85 = 0.15
dist_nausea = 5.77        → kge_score = 6.0 - 5.77 = 0.23

kge_scores = [0.72, 0.15, 0.23]
```

**最终分数** (α=3.0):
```
final_scores = rule_scores + 3.0 * kge_scores

symptom_chest_pain:  0.87 + 3.0 * 0.72 = 3.03  ✅ 最高
symptom_headache:   -5.0  + 3.0 * 0.15 = -4.55
symptom_nausea:     -5.0  + 3.0 * 0.23 = -4.31
```

**分析**:
- **chest_pain**: 规则和KGE都支持 → 高分
- **headache/nausea**: 规则不支持，KGE也弱 → 低分
- **KGE的作用**: 为规则未覆盖的实体提供兜底评分

#### 4.2.4 α参数影响

**不同α值的效果**:
```
α=0.0 (纯规则):
  chest_pain: 0.87,  headache: -5.0,  nausea: -5.0
  → 规则未覆盖的实体完全没机会

α=1.0 (小权重KGE):
  chest_pain: 1.59,  headache: -4.85,  nausea: -4.77
  → KGE提供微弱补充

α=3.0 (平衡点,推荐):
  chest_pain: 3.03,  headache: -4.55,  nausea: -4.31
  → 规则为主,KGE提供有效补充

α=5.0 (高权重KGE):
  chest_pain: 4.47,  headache: -4.25,  nausea: -3.85
  → KGE主导,规则作用被弱化
```

**最佳实践**:
- UMLS数据集: α = 3.0~4.0 (规则丰富)
- FB15k-237: α = 2.0~3.0 (规则稀疏,需更多KGE)
- 一般推荐: α = 3.0

---

### 4.3 Grounding机制详细示例

#### 4.3.1 简单2-hop规则示例

**知识图谱**:
```
实体: Alice, Bob, Carol, David, Eve
关系: father, mother, brother

三元组:
  (Alice, father, Bob)
  (Alice, mother, Carol)
  (Bob, father, David)
  (Bob, brother, Eve)
```

**规则**: `father(X, Y) ∧ father(Y, Z) → grandfather(X, Z)`

**查询**: `(Alice, grandfather, ?)`

**Grounding过程**:

```
步骤0: 初始化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
我们用一个向量记录"当前在哪些实体上"
初始状态: 在Alice上

实体:    Alice   Bob   Carol   David   Eve
向量x:   [ 1  ,  0  ,   0   ,   0   ,  0  ]
         ↑ 表示"当前在Alice这个位置"


步骤1: 第1跳传播 (沿father关系)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
规则要求: 第一步走father关系

知识图谱中的father边:
  - (Alice, father, Bob)    ✓ 起点是Alice
  - (Bob, father, David)    ✗ 起点是Bob,但Bob当前值为0,所以不传播

操作: 看哪些实体当前有"消息"(值>0)
  - Alice的值是1 → 沿着Alice的father边传播
  - Alice --father--> Bob
  - 所以Bob收到消息1

传播后状态:
实体:    Alice   Bob   Carol   David   Eve
向量x:   [ 0  ,  1  ,   0   ,   0   ,  0  ]
                ↑ Bob收到了Alice传来的消息


步骤2: 第2跳传播 (再沿father关系)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
规则要求: 第二步还是走father关系

知识图谱中的father边:
  - (Alice, father, Bob)    ✗ 起点Alice当前值为0,不传播
  - (Bob, father, David)    ✓ 起点是Bob,Bob当前值为1

操作: 看哪些实体当前有"消息"(值>0)
  - Bob的值是1 → 沿着Bob的father边传播
  - Bob --father--> David
  - 所以David收到消息1

传播后状态:
实体:    Alice   Bob   Carol   David   Eve
向量x:   [ 0  ,  0  ,   0   ,   1   ,  0  ]
                              ↑ David收到了Bob传来的消息


最终结果:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
grounding_count = [0, 0, 0, 1, 0]
                              ↑
                       David被到达1次

解释:
存在1条路径满足规则 father ∧ father:
  Alice --father--> Bob --father--> David

所以David的grounding count = 1
```

**代码实现** (src/data.py:410-421):
```python
grounding_count = graph.grounding(
    h=torch.tensor([0]),      # Alice
    r_head=grandfather,       # 查询关系
    rule=[father, father],    # 规则体: father∧father
    edges_to_remove=None
)
# 返回: tensor([0, 0, 0, 1, 0])
```

#### 4.3.2 多路径示例

**扩展知识图谱**:
```
在之前的基础上新增:
  (Alice, father, Tom)   # Alice有两个孩子: Bob和Tom
  (Tom, father, David)   # Tom和Bob都有一个孩子David
```

**完整的知识图谱**:
```
  Alice
   ├─father→ Bob ──father→ David
   └─father→ Tom ──father→ David
```

**Grounding过程**:

```
步骤0: 初始化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
实体:    Alice   Bob   Carol   David   Eve   Tom
向量x:   [ 1  ,  0  ,   0   ,   0   ,  0  ,  0 ]
         ↑ 从Alice开始


步骤1: 第1跳传播 (沿father关系)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
知识图谱中所有的father边:
  - (Alice, father, Bob)    ✓ Alice值为1
  - (Alice, father, Tom)    ✓ Alice值为1
  - (Bob, father, David)    ✗ Bob值为0,不传播
  - (Tom, father, David)    ✗ Tom值为0,不传播

操作:
  - Alice(值=1) 沿着2条father边传播:
    • Alice --father--> Bob   (Bob收到1)
    • Alice --father--> Tom   (Tom收到1)

传播后:
实体:    Alice   Bob   Carol   David   Eve   Tom
向量x:   [ 0  ,  1  ,   0   ,   0   ,  0  ,  1 ]
                ↑                             ↑
              Bob和Tom各自收到消息


步骤2: 第2跳传播 (再沿father关系)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
知识图谱中所有的father边:
  - (Alice, father, Bob)    ✗ Alice值为0,不传播
  - (Alice, father, Tom)    ✗ Alice值为0,不传播
  - (Bob, father, David)    ✓ Bob值为1
  - (Tom, father, David)    ✓ Tom值为1

操作:
  - Bob(值=1) 沿着father边传播:
    • Bob --father--> David   (David收到1)
  - Tom(值=1) 沿着father边传播:
    • Tom --father--> David   (David再收到1)

传播后:
实体:    Alice   Bob   Carol   David   Eve   Tom
向量x:   [ 0  ,  0  ,   0   ,   2   ,  0  ,  0 ]
                              ↑
                     David收到2次消息! (1+1=2)


最终结果:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
grounding_count = [0, 0, 0, 2, 0, 0]
                              ↑
                       David被到达2次

解释:
存在2条路径满足规则 father ∧ father:
  路径1: Alice --father--> Bob --father--> David
  路径2: Alice --father--> Tom --father--> David

所以David的grounding count = 2 (更高置信度!)
```

**关键洞察**:
- Grounding count表示**路径数量**
- 多条路径 → 更高的置信度
- 这是一种**软匹配**,而非0/1的硬匹配

#### 4.3.3 Grounding在打分中的应用

**完整流程** (src/model.py:337-409):

假设查询 `(Alice, grandfather, ?)`，有2条规则:

```python
# Rule 1: father ∧ father → grandfather
grounding_count_1 = [0, 0, 0, 2, 0, 0]  # David有2条路径
rule_confidence_1 = 0.85                # 规则1的置信度
mlp_feature_1 = [0.2, 0.5, 0.3, ...]   # 规则1的100维MLP特征

# Rule 2: father ∧ brother → grandfather (假设存在)
grounding_count_2 = [0, 0, 0, 0, 1, 0]  # Eve有1条路径
rule_confidence_2 = 0.62                # 规则2的置信度
mlp_feature_2 = [0.1, 0.3, 0.4, ...]

# 对David的特征聚合:
feature[David] = rule_confidence_1 × grounding_count_1[David] × mlp_feature_1
               + rule_confidence_2 × grounding_count_2[David] × mlp_feature_2

               = 0.85 × 2 × [0.2, 0.5, 0.3, ...]
               + 0.62 × 0 × [0.1, 0.3, 0.4, ...]

               = 1.7 × [0.2, 0.5, 0.3, ...]
               = [0.34, 0.85, 0.51, ...]

# 对Eve的特征聚合:
feature[Eve] = 0.85 × 0 × mlp_feature_1
             + 0.62 × 1 × mlp_feature_2

             = 0.62 × [0.1, 0.3, 0.4, ...]
             = [0.062, 0.186, 0.248, ...]

# MLP打分:
score[David] = MLP(LayerNorm(ReLU([0.34, 0.85, ...]))) = 0.92 ✅
score[Eve] = MLP(LayerNorm(ReLU([0.062, 0.186, ...]))) = 0.35

# David得分更高因为:
# 1. 有2条路径支持 (vs Eve的1条)
# 2. 规则置信度更高 (0.85 vs 0.62)
```

#### 4.3.4 边移除机制

**问题**: 训练时如果不移除训练边,模型会作弊

**示例**:
```
训练三元组: (Alice, grandfather, David)

如果不移除边,grounding会直接走:
  Alice --grandfather--> David  (训练边)

这样模型学到的只是记忆训练集,而非规则推理!
```

**解决方案** (src/data.py:434-446):
```python
# 训练时指定要移除的边
edges_to_remove = [(Alice, grandfather, David)]

propagate(x, relation=grandfather, edges_to_remove):
    # 在传播时将这些边的消息置零
    message[edges_to_remove] = 0

# 这样模型被迫通过规则路径推理:
# Alice → father → Bob → father → David
# 而不能直接走训练边
```

---

5. 过滤与排序：
   # 过滤训练/验证/测试中已知的三元组
   final_scores[known_triplets] = -inf

   # 排序
   ranked_entities = argsort(final_scores, descending=True)

6. 输出Top-K
```

---

## 五、原始RulE的关键特点

### 5.1 规则置信度

**计算方式**：
```
w_i = γ_rule - ||r_body_sum + R_i - r_head||
```

**特点**：
- ✅ 简单直接
- ✅ 可微分
- ❌ 确定性标量，无法表达不确定性
- ❌ 无法区分"置信度=0.7因为质量中等"和"置信度=0.7但样本太少不确定"

### 5.2 训练流程

```
预训练: 联合学习 entity/relation/rule embeddings
   ↓
Grounding: 冻结embeddings，学习MLP参数
   ↓
推理: 使用确定性置信度w_i加权规则
```

---

---

# 不确定性RulE模型详解

## 一、模型架构

### 1.1 核心组件

```python
class UncertaintyRulE(nn.Module):
    def __init__(self):
        # 【继承自原始RulE】
        # 组件1: 实体嵌入
        self.entity_embedding = nn.Parameter(
            torch.randn(num_entities, hidden_dim * 2)
        )

        # 组件2: 关系嵌入
        self.relation_embedding = nn.Parameter(
            torch.randn(num_relations, hidden_dim)
        )

        # 组件3: 规则嵌入
        self.rule_emb = nn.Parameter(
            torch.randn(num_rules, rule_dim)
        )

        # 组件4: MLP规则特征（Grounding阶段训练）
        self.mlp_feature = nn.Parameter(
            torch.randn(num_rules, mlp_rule_dim)
        )

        # 组件5: 评分MLP（Grounding阶段训练）
        self.score_model = MLP(mlp_rule_dim, [hidden_size, 1])


        # 【新增】不确定性建模组件
        # 组件6: 置信度均值网络
        self.mu_network = nn.Sequential(
            nn.Linear(rule_dim + hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 组件7: 置信度方差网络
        self.logvar_network = nn.Sequential(
            nn.Linear(rule_dim + hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
```

### 1.2 参数表

| 参数名 | 维度 | 说明 | 训练阶段 | 新增 |
|--------|------|------|----------|------|
| `entity_embedding` | `[num_entities, hidden_dim*2]` | 实体的复数嵌入 | 预训练 | ❌ |
| `relation_embedding` | `[num_relations, hidden_dim]` | 关系的相位嵌入 | 预训练 | ❌ |
| `rule_emb` | `[num_rules, rule_dim]` | 规则嵌入 | 预训练 | ❌ |
| `mlp_feature` | `[num_rules, mlp_rule_dim]` | 规则MLP特征 | Grounding | ❌ |
| `score_model` | MLP网络 | 候选实体打分网络 | Grounding | ❌ |
| **`mu_network`** | **3层MLP** | **置信度均值网络** | **预训练** | **✅** |
| **`logvar_network`** | **3层MLP** | **置信度对数方差网络** | **预训练** | **✅** |

**新增参数量**：
- mu_network: `(rule_dim + hidden_dim) × 256 + 256 × 128 + 128 × 1` ≈ 100K
- logvar_network: 同上 ≈ 100K
- **总计**: ≈ 200K 参数

---

## 二、预训练阶段

### 2.1 目标

联合学习实体嵌入、关系嵌入、规则嵌入，**以及规则置信度的不确定性分布**。

### 2.2 步骤流程

#### Step 1: 知识图谱嵌入损失 (KGE Loss)

**与原始RulE完全相同**，详见上文。

**公式**：
```
loss_kge = -log(σ(score_pos)) - Σ p_neg * log(σ(-score_neg))
```

---

#### Step 2: 规则嵌入损失 (Rule Loss) - 【核心改动】

**输入**：规则 `[rule_id, r_head, r_body_1, r_body_2, ...]`

**公式**：
```
1. 获取规则组成部分（与原始相同）：
   rule = [rule_id, r_head, r1, r2, ..., rL]

   r_head_emb = relation_embedding[r_head]     # [hidden_dim]
   r_body_embs = [relation_embedding[ri] for ri in [r1, r2, ..., rL]]
   R_i = rule_emb[rule_id]                     # [rule_dim]

2. 聚合规则体的关系嵌入（与原始相同）：
   r_body_sum = Σ r_body_embs[j]              # [hidden_dim]
                j=1..L

3. 【新增】构建输入特征：
   # 拼接规则嵌入和聚合的关系嵌入
   features = concat([R_i, r_body_sum])        # [rule_dim + hidden_dim]

4. 【新增】计算不确定性置信度分布：
   # 均值网络
   μ_i = mu_network(features)                  # [1] → 标量

   # 对数方差网络（输出log(σ²)保证σ²>0）
   log_σ²_i = logvar_network(features)         # [1] → 标量

   # 转换为标准差
   σ_i = exp(0.5 × log_σ²_i) = exp(log_σ_i) = √(σ²_i)

   注意：现在规则置信度是一个分布 w_i ~ N(μ_i, σ_i²)

5. 【新增】重参数化采样（Reparameterization Trick）：
   # 为了让采样过程可微分
   ε ~ N(0, 1)                                 # 标准正态分布
   w_i = μ_i + σ_i × ε                         # 采样的置信度

   # 训练时采样多次取平均
   w_samples = [μ_i + σ_i × ε_k for k in 1..K]
   w_i = mean(w_samples)                       # 平均K次采样

6. 计算规则距离（与原始相同）：
   d_rule = ||r_body_sum + R_i - r_head_emb||  # L2范数

7. 【修改】规则损失加权：
   # 原始: loss = -log(σ(γ_rule - d_rule))
   # 新版: 使用采样的置信度加权

   score_rule = sigmoid(w_i) × (γ_rule - d_rule)

   # 负采样
   score_rule_neg = sigmoid(w_neg) × (γ_rule - d_rule_neg)

   loss_rule = -log(σ(score_rule)) - Σ log(σ(-score_rule_neg))
```

**关键改动**：
- 原始：`w_i = γ_rule - d_rule` (确定性标量)
- 新版：`w_i ~ N(μ_i, σ_i²)` (概率分布)
- 训练：使用重参数化采样，取平均
- 效果：置信度低的规则对损失贡献小

---

#### Step 3: 不确定性正则化损失 【新增】

**目的**：约束学到的μ和σ有合理的性质。

##### Sub-step 3.1: KL散度正则化

**目的**：防止方差退化为0，保持一定的不确定性。

**公式**：
```
KL散度（变分推断标准项）：
   KL(N(μ_i, σ_i²) || N(0, 1)) = 0.5 × Σ (μ_i² + σ_i² - log(σ_i²) - 1)
                                  i∈all_rules

解释：
   - 如果 μ_i→0, σ_i→1: KL→0 (接近先验)
   - 如果 μ_i很大或σ_i→0: KL很大 (惩罚)
```

**效果**：
- 阻止 σ_i → 0 (方差塌缩)
- 鼓励模型保持适当的不确定性

##### Sub-step 3.2: 支持数驱动的方差约束

**目的**：训练样本少的规则应该有更高的不确定性。

**公式**：
```
1. 预先统计每个规则的支持数：
   support_count[i] = 规则i在训练集中有多少条支持路径

   例如:
   - Rule 15: support_count = 1000
   - Rule 42: support_count = 10

2. 计算目标方差：
   target_σ_i = λ_0 / (1 + log(support_count[i] + 1))

   λ_0 是超参数（典型值：1.0）

   例如:
   - Rule 15: target_σ = 1.0 / (1 + log(1001)) ≈ 0.14  (低不确定性)
   - Rule 42: target_σ = 1.0 / (1 + log(11)) ≈ 0.42    (高不确定性)

3. 方差匹配损失：
   loss_sigma = Σ (σ_i - target_σ_i)²
                i∈all_rules
```

**效果**：
- 自动让数据少的规则有高 σ
- 自动让数据多的规则有低 σ

##### Sub-step 3.3: 总不确定性损失

**公式**：
```
loss_uncertainty = β_kl × loss_kl + β_sigma × loss_sigma

其中:
   loss_kl = 0.5 × Σ (μ_i² + σ_i² - log(σ_i²) - 1)
   loss_sigma = Σ (σ_i - target_σ_i)²
```

**参数**：
- `β_kl`: KL散度权重 (典型值: 0.001)
- `β_sigma`: 方差匹配权重 (典型值: 0.1)
- `λ_0`: 支持数依赖系数 (典型值: 1.0)

---

#### Step 4: 总损失

**公式**：
```
loss_total = loss_kge + λ_rule × loss_rule + λ_uncertainty × loss_uncertainty
```

**参数**：
- `λ_rule`: 规则损失权重 (典型值: 1.0)
- `λ_uncertainty`: 不确定性损失权重 (典型值: 0.01)

---

### 2.3 预训练输出

保存到 `checkpoint`:
```python
{
    'entity_embedding': [num_entities, hidden_dim*2],
    'relation_embedding': [num_relations, hidden_dim],
    'rule_emb': [num_rules, rule_dim],

    # 【新增】
    'mu_network.state_dict': {...},
    'logvar_network.state_dict': {...}
}
```

---

## 三、Grounding阶段

### 3.1 目标

与原始RulE相同：学习如何将规则grounding到知识图谱上，并对候选实体打分。

### 3.2 冻结参数

```python
# 冻结预训练的嵌入
entity_embedding.requires_grad = False
relation_embedding.requires_grad = False
rule_emb.requires_grad = False

# 【新增】冻结不确定性网络
mu_network.requires_grad = False
logvar_network.requires_grad = False
```

### 3.3 训练参数

```python
# 只训练这些（与原始相同）
mlp_feature.requires_grad = True
score_model.requires_grad = True
```

### 3.4 步骤流程

#### Step 1: 检索适用规则

**与原始RulE完全相同**。

```python
applicable_rules = relation2rules[r]
```

---

#### Step 2: 规则Grounding

**与原始RulE完全相同**。

```python
grounding_counts = graph.grounding(
    h, r_head, [r1, r2, ..., rL],
    edges_to_remove=training_edges
)
```

---

#### Step 3: 计算规则置信度 【核心改动】

**原始方式**：
```python
w_i = γ_rule - d_rule  # 确定性标量
```

**新方式**：
```python
1. 构建输入特征：
   R_i = rule_emb[rule_id]
   r_body_sum = Σ relation_embedding[rj]
   features = concat([R_i, r_body_sum])

2. 使用不确定性模型计算分布参数：
   μ_i = mu_network(features)          # 置信度均值
   log_σ²_i = logvar_network(features)  # 对数方差
   σ_i = exp(0.5 × log_σ²_i)           # 标准差

3. 训练时采样，推理时用均值：
   if training:
       # 采样多次取平均
       w_samples = [μ_i + σ_i × ε_k for k in 1..K]
       w_i = mean(w_samples)
   else:
       # 推理时直接用均值
       w_i = μ_i
```

**关键点**：
- 训练时：采样提供正则化效果（类似Dropout）
- 推理时：用均值保证确定性预测
- σ_i 提供可解释性信息

---

#### Step 4: 构建软多热编码 【修改】

**公式**：
```
1. 初始化候选实体特征：
   candidate_features = zeros([num_entities, mlp_rule_dim])

2. 对每个grounded的规则：
   for each rule_i with grounding_counts_i:
       # 【修改】使用不确定性模型计算的置信度
       μ_i, σ_i = uncertainty_model(rule_i, ...)

       if training:
           w_i = mean([μ_i + σ_i × ε_k for k in 1..K])
       else:
           w_i = μ_i

       # 获取规则的MLP特征
       rule_feature_i = mlp_feature[rule_i]

       # 对每个候选实体累加特征
       for each entity e:
           if grounding_counts_i[e] > 0:
               candidate_features[e] += w_i × grounding_counts_i[e] × rule_feature_i

公式形式：
   feature[e] = Σ (w_i × count_i[e] × mlp_feature[i])
                i∈grounded_rules

   其中 w_i 来自不确定性模型，而非固定公式
```

**与原始RulE的区别**：
- 原始：`w_i = γ_rule - d_rule` (固定公式)
- 新版：`w_i = μ_i` (神经网络学习)

---

#### Step 5: 聚合与打分

**与原始RulE完全相同**。

```python
aggregated_features = LayerNorm(ReLU(candidate_features))
scores = score_model(aggregated_features)
```

---

#### Step 6: 计算损失

**与原始RulE完全相同**。

```python
loss = CrossEntropy(scores, smooth_labels)
```

---

### 3.5 Grounding输出

**与原始RulE相同**：

保存到 `grounding.pt`:
```python
{
    'mlp_feature': [num_rules, mlp_rule_dim],
    'score_model.state_dict': {...}
}
```

---

## 四、推理阶段

### 4.1 完整流程

**输入**：查询 `(h, r, ?)`

**步骤**：

```
1. 检索适用规则（与原始相同）：
   applicable_rules = relation2rules[r]

2. 对每个规则进行grounding（与原始相同）：
   for each rule in applicable_rules:
       grounding_counts = graph.grounding(h, rule.body)

       # 【修改】计算规则置信度
       # 原始: w_i = γ_rule - d_rule
       # 新版: 使用不确定性模型

       R_i = rule_emb[rule.id]
       r_body_sum = Σ relation_embedding[rj]
       features = concat([R_i, r_body_sum])

       μ_i = mu_network(features)
       log_σ²_i = logvar_network(features)
       σ_i = exp(0.5 × log_σ²_i)

       w_i = μ_i  # 推理时直接用均值

       # 累加特征
       for each entity e:
           if grounding_counts[e] > 0:
               features[e] += w_i × grounding_counts[e] × mlp_feature[rule.id]

3. 聚合与打分（与原始相同）：
   aggregated = LayerNorm(ReLU(features))
   scores = score_model(aggregated)

4. 【可选】结合KGE分数（与原始相同）：
   kge_scores = compute_rotate_score(h, r, all_entities)
   final_scores = scores + α × kge_scores

5. 过滤与排序（与原始相同）：
   final_scores[known_triplets] = -inf
   ranked_entities = argsort(final_scores, descending=True)

6. 【新增】输出可解释性信息：
   for each grounded_rule:
       print(f"Rule {rule.id}:")
       print(f"  置信度: μ={μ:.3f}, σ={σ:.3f}")
       if σ < 0.2:
           print(f"  高质量规则")
       elif σ < 0.5:
           print(f"  中等质量规则")
       else:
           print(f"  低质量规则（不确定性高）")
```

---

## 五、不确定性RulE的关键特点

### 5.1 规则置信度

**原始方式**：
```
w_i = γ_rule - ||r_body_sum + R_i - r_head||  # 确定性标量
```

**新方式**：
```
w_i ~ N(μ_i, σ_i²)  # 概率分布

其中:
   μ_i = mu_network([R_i, r_body_sum])
   σ_i = exp(0.5 × logvar_network([R_i, r_body_sum]))
```

**优势**：
- ✅ 表达不确定性：σ_i 越大表示越不确定
- ✅ 自动降权低质量规则：σ高的规则影响小
- ✅ 提供可解释性：可以看到每个规则的置信度和不确定性
- ✅ 自适应：根据训练数据量自动调整σ

### 5.2 训练流程

```
预训练: 联合学习 entity/relation/rule embeddings + 不确定性分布
   ↓
   【新增】正则化：KL散度 + 支持数驱动的方差约束
   ↓
Grounding: 冻结embeddings和不确定性网络，学习MLP参数
   ↓
   【新增】训练时采样置信度，提供正则化
   ↓
推理: 使用μ作为置信度，σ提供可解释性
```

---

---

# 对比总结

## 一、核心差异对比表

| 维度 | 原始RulE | 不确定性RulE |
|------|----------|--------------|
| **规则置信度表示** | 标量 w_i | 分布 w_i ~ N(μ_i, σ_i²) |
| **置信度计算** | w_i = γ_rule - d_rule | μ_i, σ_i = 双MLP网络 |
| **训练时采样** | 无 | 重参数化采样 |
| **推理时使用** | w_i | μ_i |
| **损失函数** | kge_loss + rule_loss | kge_loss + rule_loss + uncertainty_loss |
| **正则化** | 无特殊正则化 | KL散度 + 支持数约束 |
| **参数量** | N | N + 200K (双MLP) |
| **可解释性** | 仅置信度值 | 置信度 + 不确定性 |
| **计算成本** | 1x | 训练: 2-3x, 推理: 1.1x |

---

## 二、公式对比

### 2.1 规则置信度计算

#### 原始RulE:
```
w_i = γ_rule - ||Σ r_body_emb[j] + R_i - r_head_emb||
```

#### 不确定性RulE:
```
features = concat([R_i, Σ r_body_emb[j]])

μ_i = mu_network(features)
     = Linear(ReLU(Linear(ReLU(Linear(features)))))

log_σ²_i = logvar_network(features)
          = Linear(ReLU(Linear(ReLU(Linear(features)))))

σ_i = exp(0.5 × log_σ²_i)

训练时: w_i = μ_i + σ_i × ε, ε ~ N(0,1)
推理时: w_i = μ_i
```

---

### 2.2 损失函数

#### 原始RulE:
```
loss_total = loss_kge + λ_rule × loss_rule

其中:
   loss_rule = -log(σ(γ_rule - d_rule)) - Σ log(σ(-(γ_rule - d_rule_neg)))
```

#### 不确定性RulE:
```
loss_total = loss_kge + λ_rule × loss_rule + λ_uncertainty × loss_uncertainty

其中:
   loss_rule 使用采样的 w_i

   loss_uncertainty = β_kl × loss_kl + β_sigma × loss_sigma

   loss_kl = 0.5 × Σ (μ_i² + σ_i² - log(σ_i²) - 1)

   loss_sigma = Σ (σ_i - λ_0/(1+log(support_count_i+1)))²
```

---

### 2.3 软多热编码构建

#### 原始RulE:
```
feature[e] = Σ (w_i × count_i[e] × mlp_feature[i])
             i∈grounded_rules

其中: w_i = γ_rule - d_rule (确定性)
```

#### 不确定性RulE:
```
feature[e] = Σ (w_i × count_i[e] × mlp_feature[i])
             i∈grounded_rules

其中: w_i = μ_i (来自神经网络)
     训练时: w_i 采样自 N(μ_i, σ_i²)
```

---

## 三、数据流对比图

### 原始RulE:

```
预训练阶段:
   规则 → [r_body, R_i, r_head] → d_rule → w_i → rule_loss
                                              ↓
                                          loss_total
                                              ↓
                                         反向传播

Grounding阶段:
   Query(h,r,?) → 检索规则 → Grounding
                              ↓
                        grounding_counts
                              ↓
                      计算 w_i (固定公式)
                              ↓
                   feature = Σ w_i × count × mlp_feature
                              ↓
                           MLP打分
                              ↓
                          交叉熵损失
```

### 不确定性RulE:

```
预训练阶段:
   规则 → [R_i, r_body_sum] → concat → features
                                          ↓
                              ┌───────────┴───────────┐
                              ↓                       ↓
                         mu_network            logvar_network
                              ↓                       ↓
                             μ_i                    σ_i
                              ↓                       ↓
                         【重参数化采样】
                              ↓
                         w_i = μ_i + σ_i × ε
                              ↓
                          rule_loss
                              ↓
                      【新增】uncertainty_loss
                         (KL + 支持数约束)
                              ↓
                          loss_total
                              ↓
                         反向传播

Grounding阶段:
   Query(h,r,?) → 检索规则 → Grounding
                              ↓
                        grounding_counts
                              ↓
                      【修改】计算 μ_i, σ_i
                         (使用训练好的网络)
                              ↓
                      训练时: w_i ~ N(μ_i, σ_i²)
                      推理时: w_i = μ_i
                              ↓
                   feature = Σ w_i × count × mlp_feature
                              ↓
                           MLP打分
                              ↓
                          交叉熵损失
```

---

## 四、具体例子对比

### 场景：预测 (Albert_Einstein, nationality, ?)

假设有两个规则：
- Rule 15: `born_in(x,y) ∧ city_of(y,z) → nationality(x,z)` (1000个训练样本)
- Rule 42: `works_in(x,y) ∧ located_in(y,z) → nationality(x,z)` (10个训练样本)

#### 原始RulE:

```
Rule 15:
   d_rule = 2.5
   w_15 = 6.0 - 2.5 = 3.5

Rule 42:
   d_rule = 3.0
   w_42 = 6.0 - 3.0 = 3.0

候选实体 Germany:
   grounding_counts_15[Germany] = 2
   grounding_counts_42[Germany] = 0

   feature[Germany] = 3.5 × 2 × mlp_feature[15]
                    = 7.0 × [0.2, 0.5, ..., 0.3]

候选实体 USA:
   grounding_counts_15[USA] = 0
   grounding_counts_42[USA] = 1

   feature[USA] = 3.0 × 1 × mlp_feature[42]
                = 3.0 × [0.1, 0.3, ..., 0.4]

问题：Rule 42只有10个训练样本，但置信度3.0看起来很高，可能导致过拟合！
```

#### 不确定性RulE:

```
Rule 15 (1000个训练样本):
   features = concat([R_15, r_body_sum])
   μ_15 = mu_network(features) = 0.850
   σ_15 = exp(0.5 × logvar_network(features)) = 0.120

   解释: σ小 → 高可信度

Rule 42 (10个训练样本):
   features = concat([R_42, r_body_sum])
   μ_42 = mu_network(features) = 0.650
   σ_42 = exp(0.5 × logvar_network(features)) = 0.450

   解释: σ大 → 低可信度（自动识别数据不足）

候选实体 Germany:
   grounding_counts_15[Germany] = 2
   grounding_counts_42[Germany] = 0

   w_15 = μ_15 = 0.850
   feature[Germany] = 0.850 × 2 × mlp_feature[15]
                    = 1.7 × [0.2, 0.5, ..., 0.3]

候选实体 USA:
   grounding_counts_15[USA] = 0
   grounding_counts_42[USA] = 1

   w_42 = μ_42 = 0.650
   feature[USA] = 0.650 × 1 × mlp_feature[42]
                = 0.650 × [0.1, 0.3, ..., 0.4]

优势：Rule 42因为σ大，模型自动降低了它的置信度(μ=0.650)，
     避免过度依赖数据不足的规则！
```

---

## 五、训练时采样的作用

### 原始RulE (无采样):

```
每次前向传播:
   w_i = 3.5 (固定)
   feature = 3.5 × count × mlp_feature

梯度总是相同的 → 可能过拟合
```

### 不确定性RulE (采样):

```
第1次前向传播:
   ε_1 = -0.3
   w_i = 0.85 + 0.12 × (-0.3) = 0.814
   feature = 0.814 × count × mlp_feature

第2次前向传播:
   ε_2 = 0.8
   w_i = 0.85 + 0.12 × 0.8 = 0.946
   feature = 0.946 × count × mlp_feature

第3次前向传播:
   ε_3 = -0.1
   w_i = 0.85 + 0.12 × (-0.1) = 0.838
   feature = 0.838 × count × mlp_feature

梯度每次都不同 → 提供正则化，类似Dropout效果
```

---

## 六、超参数对比

### 原始RulE:

| 超参数 | 典型值 | 说明 |
|--------|--------|------|
| `hidden_dim` | 500-2000 | 嵌入维度 |
| `γ_fact` | 6.0 | 三元组margin |
| `γ_rule` | 6.0 | 规则margin |
| `λ_rule` | 1.0 | 规则损失权重 |
| `learning_rate` | 0.0001 | 学习率 |
| `batch_size` | 256 | 批大小 |

### 不确定性RulE:

| 超参数 | 典型值 | 说明 |
|--------|--------|------|
| `hidden_dim` | 500-2000 | 嵌入维度 |
| `γ_fact` | 6.0 | 三元组margin |
| `γ_rule` | 6.0 | 规则margin (可能不再需要) |
| `λ_rule` | 1.0 | 规则损失权重 |
| **`λ_uncertainty`** | **0.01** | **不确定性损失权重** |
| **`β_kl`** | **0.001** | **KL散度权重** |
| **`β_sigma`** | **0.1** | **方差匹配权重** |
| **`λ_0`** | **1.0** | **支持数系数** |
| **`num_samples`** | **5** | **训练时采样次数** |
| `learning_rate` | 0.0001 | 学习率 |
| `batch_size` | 256 | 批大小 |

---

## 七、优缺点总结

### 原始RulE:

**优点**：
- ✅ 简单直观
- ✅ 计算高效
- ✅ 实现简单

**缺点**：
- ❌ 无法表达不确定性
- ❌ 对数据不足的规则过度自信
- ❌ 对噪声规则不敏感
- ❌ 缺乏可解释性

### 不确定性RulE:

**优点**：
- ✅ 表达规则质量的不确定性
- ✅ 自动降权低质量规则
- ✅ 提供可解释性(μ和σ)
- ✅ 训练时采样提供正则化
- ✅ 理论基础扎实(变分推断)

**缺点**：
- ❌ 新增约200K参数
- ❌ 训练时间增加2-3倍
- ❌ 需要调更多超参数
- ❌ 实现复杂度增加

---

## 八、适用场景

### 原始RulE适用:
- 数据集质量高，规则都很可靠
- 计算资源有限
- 不需要校准的置信度
- 快速原型开发

### 不确定性RulE适用:
- 规则质量参差不齐(如AMIE自动挖掘)
- 训练数据不足
- 需要可解释性
- 需要校准的置信度(医疗、金融等应用)
- 有足够的计算资源

---

## 九、预期性能提升

基于文档预测：

| 数据集 | 原始MRR | 预期提升 | 目标MRR | 提升原因 |
|--------|---------|----------|---------|----------|
| FB15k-237 | 0.362 | +1.5% | 0.367 | 规则质量差异大 |
| WN18RR | 0.519 | +2.0% | 0.529 | 中等规模，噪声规则多 |
| UMLS | 0.867 | +2.5% | 0.889 | 小数据集，部分规则样本少 |
| Kinship | 0.736 | +3.0% | 0.758 | 小数据集，罕见规则多 |

**提升来源**：
1. 噪声规则过滤 (+1%)
2. 低频规则保护 (+1%)
3. 隐式集成效果 (+1%)

---

## 十、实现建议

### 阶段1: 复现原始RulE
- 确保在基准数据集上达到论文结果
- 理解代码结构

### 阶段2: 实现不确定性模块
- 添加 `mu_network` 和 `logvar_network`
- 实现重参数化采样
- 添加不确定性损失

### 阶段3: 集成训练
- 修改预训练循环
- 修改Grounding循环
- 调试训练稳定性

### 阶段4: 实验验证
- 对比实验
- 消融实验
- 可解释性分析

---

**文档结束**
