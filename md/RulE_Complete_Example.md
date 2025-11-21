# RulE模型完整流程示例

> **目标**：通过一个具体例子，从头到尾演示RulE模型如何工作
>
> **示例查询**：图灵（Alan Turing）的国籍是什么？

---

## 目录

1. [背景设定](#1-背景设定)
2. [阶段1：预训练 - 学习嵌入](#2-阶段1预训练---学习嵌入)
3. [阶段2：规则Grounding训练](#3-阶段2规则grounding训练)
4. [阶段3：推理预测](#4-阶段3推理预测)
5. [完整数据流](#5-完整数据流)

---

## 1. 背景设定

### 1.1 知识图谱数据

假设我们有以下知识图谱事实：

```
实体（Entities）:
  - Alan Turing（图灵）
  - London（伦敦）
  - Cambridge（剑桥）
  - Bletchley Park（布莱切利园）
  - UK（英国）
  - USA（美国）
  - Alonzo Church（丘奇）

关系（Relations）:
  - born_in（出生于）
  - city_of（城市属于）
  - nationality（国籍）
  - works_at（工作于）
  - org_country（组织所在国）
  - friend_of（朋友）

三元组事实（Triples）:
  (Turing, born_in, London)
  (Turing, born_in, Cambridge)        # 多个出生地
  (London, city_of, UK)
  (Cambridge, city_of, UK)
  (Turing, works_at, Bletchley_Park)
  (Bletchley_Park, org_country, UK)
  (Turing, friend_of, Church)
  (Church, nationality, USA)
  (Turing, nationality, UK)            # 这是我们要预测的！
```

### 1.2 挖掘的逻辑规则

使用RNNLogic等工具从数据中挖掘出的规则：

```
Rule1: born_in(x,y) ∧ city_of(y,z) → nationality(x,z)
       (出生地所属国 → 国籍)

Rule2: friend_of(x,y) ∧ nationality(y,z) → nationality(x,z)
       (朋友的国籍 → 自己国籍)

Rule3: works_at(x,y) ∧ org_country(y,z) → nationality(x,z)
       (工作单位所在国 → 国籍)

Rule4: visits(x,y) ∧ located_in(y,z) → nationality(x,z)
       (访问过的地方 → 国籍)  # 这是噪声规则！
```

### 1.3 任务目标

```
训练集查询: 已知所有事实
验证集查询: (Turing, nationality, ?)
测试集查询: 预测其他实体的国籍

目标: 学习能够准确预测国籍的模型
```

---

## 2. 阶段1：预训练 - 学习嵌入

### 2.1 初始化

```python
# 超参数设定
hidden_dim = 3          # 实际中是500-2000，这里简化为3维
gamma_fact = 9.0        # Triplet margin
gamma_rule = 8.0        # Rule margin
num_entities = 7
num_relations = 6
num_rules = 4

# 初始化嵌入（随机）
entity_embedding = nn.Embedding(7, 3*2)  # RotatE需要2倍维度（复数）
relation_embedding = nn.Embedding(6, 3)   # 关系是角度
rule_embedding = nn.Parameter(torch.zeros(4, 3))  # 规则嵌入
```

### 2.2 训练数据批次

```python
# Batch 1: Triplet数据
positive_triplet = (Turing, born_in, London)
negative_triplet = (Turing, born_in, USA)  # 负采样

# Batch 2: Rule数据
positive_rule = [Rule1_id, 2, nationality, born_in, city_of]
negative_rule = [Rule1_id, 2, nationality, born_in, works_at]  # 替换一个关系
```

### 2.3 前向传播 - Triplet Loss

```python
# ============ Triplet部分（RotatE）============

# 1. 提取嵌入
h_turing = entity_embedding[Turing]  # [3*2] = [6]维
r_born_in = relation_embedding[born_in]  # [3]维
t_london = entity_embedding[London]  # [6]维

# 2. RotatE计算分数
# 将实体嵌入分为实部和虚部
h_re, h_im = h_turing[:3], h_turing[3:]  # 各[3]维
t_re, t_im = t_london[:3], t_london[3:]

# 关系是旋转（转为复数）
r_phase = r_born_in  # [3]维角度
r_re = torch.cos(r_phase)  # [cos(θ1), cos(θ2), cos(θ3)]
r_im = torch.sin(r_phase)  # [sin(θ1), sin(θ2), sin(θ3)]

# 复数乘法: h ◦ r
result_re = h_re * r_re - h_im * r_im
result_im = h_re * r_im + h_im * r_re

# 距离: d = |h ◦ r - t|
distance_re = result_re - t_re
distance_im = result_im - t_im
distance = torch.sqrt(distance_re**2 + distance_im**2).sum()

# 得分（越高越好）
pos_score = gamma_fact - distance

# 3. 对负样本重复相同计算
t_usa = entity_embedding[USA]
# ... 计算过程相同 ...
neg_score = gamma_fact - distance_neg

# 4. Triplet Loss
triplet_loss = max(0, margin - (pos_score - neg_score))
```

**第1个epoch后的嵌入（示例值）**：

```python
# 实体嵌入（复数形式，[实部; 虚部]）
entity_embedding[Turing] = [0.2, 0.5, -0.3 | 0.1, -0.2, 0.4]
entity_embedding[London] = [0.3, 0.4, -0.1 | 0.2, -0.1, 0.3]
entity_embedding[UK] = [0.8, 0.9, 0.2 | 0.5, 0.3, 0.6]

# 关系嵌入（角度）
relation_embedding[born_in] = [0.5, 1.2, -0.3]
relation_embedding[city_of] = [0.8, -0.5, 0.9]
relation_embedding[nationality] = [1.3, 0.7, 0.6]
```

### 2.4 前向传播 - Rule Loss

```python
# ============ Rule部分 ============

# 1. 提取规则信息
rule_id = 0  # Rule1
rule_head = nationality
rule_body = [born_in, city_of]

# 2. 提取嵌入
rule_emb = rule_embedding[rule_id]  # [3]维，初始为[0,0,0]
head_emb = relation_embedding[nationality]  # [1.3, 0.7, 0.6]
body_emb_1 = relation_embedding[born_in]    # [0.5, 1.2, -0.3]
body_emb_2 = relation_embedding[city_of]    # [0.8, -0.5, 0.9]

# 3. 计算规则体组合
body_sum = body_emb_1 + body_emb_2
         = [0.5, 1.2, -0.3] + [0.8, -0.5, 0.9]
         = [1.3, 0.7, 0.6]

# 4. 计算距离
# 理想情况: body_sum + rule_emb ≈ head_emb
distance = torch.norm(body_sum + rule_emb - head_emb, p=1)
         = ||(1.3 + 0.0 - 1.3)|| + ||(0.7 + 0.0 - 0.7)|| + ||(0.6 + 0.0 - 0.6)||
         = 0.0  # 完美匹配！

# 5. 规则得分
pos_rule_score = gamma_rule - distance
               = 8.0 - 0.0
               = 8.0

# 6. 负样本规则得分
# 负规则: born_in ∧ works_at → nationality（语义不通）
neg_body_emb_2 = relation_embedding[works_at]  # 假设为[1.5, 0.2, -0.8]
neg_body_sum = body_emb_1 + neg_body_emb_2
             = [0.5, 1.2, -0.3] + [1.5, 0.2, -0.8]
             = [2.0, 1.4, -1.1]

neg_distance = torch.norm(neg_body_sum + rule_emb - head_emb, p=1)
             = ||[2.0 - 1.3]|| + ||[1.4 - 0.7]|| + ||[-1.1 - 0.6]||
             = 0.7 + 0.7 + 1.7
             = 3.1

neg_rule_score = gamma_rule - neg_distance
               = 8.0 - 3.1
               = 4.9

# 7. Rule Loss
rule_loss = max(0, margin - (pos_rule_score - neg_rule_score))
          = max(0, 1.0 - (8.0 - 4.9))
          = max(0, -2.1)
          = 0.0  # 已经很好了
```

### 2.5 联合训练

```python
# 总损失
total_loss = triplet_loss + alpha * rule_loss
           = 0.5 + 1.0 * 0.0
           = 0.5

# 反向传播
total_loss.backward()

# 更新所有参数
optimizer.step()

# 更新后的嵌入会更好地满足：
# 1. 三元组事实（通过triplet_loss）
# 2. 逻辑规则（通过rule_loss）
```

**训练30000步后的嵌入（收敛值）**：

```python
# 关系嵌入（经过训练优化）
relation_embedding[born_in] = [0.5, 1.2, -0.3]
relation_embedding[city_of] = [0.8, -0.5, 0.9]
relation_embedding[nationality] = [1.3, 0.7, 0.6]
relation_embedding[works_at] = [0.6, 0.8, -0.2]
relation_embedding[org_country] = [0.7, -0.1, 0.8]
relation_embedding[friend_of] = [0.3, 0.4, 0.1]

# 规则嵌入（学到的残差校正向量）
rule_embedding[Rule1] = [0.0, 0.0, 0.0]   # Rule1完美，不需要校正
rule_embedding[Rule2] = [-0.2, 0.1, -0.1] # Rule2需要小校正
rule_embedding[Rule3] = [0.0, 0.0, -0.1]  # Rule3需要微调
rule_embedding[Rule4] = [-2.5, 1.8, 3.2]  # Rule4是噪声，需要大校正但仍无法修复
```

### 2.6 验证规则质量

```python
# 计算每条规则的置信度

# Rule1: born_in ∧ city_of → nationality
body_sum_1 = [0.5, 1.2, -0.3] + [0.8, -0.5, 0.9] = [1.3, 0.7, 0.6]
distance_1 = ||[1.3, 0.7, 0.6] + [0.0, 0.0, 0.0] - [1.3, 0.7, 0.6]|| = 0.0
confidence_1 = 8.0 - 0.0 = 8.0 ⭐⭐⭐⭐⭐

# Rule2: friend_of ∧ nationality → nationality
body_sum_2 = [0.3, 0.4, 0.1] + [1.3, 0.7, 0.6] = [1.6, 1.1, 0.7]
distance_2 = ||[1.6, 1.1, 0.7] + [-0.2, 0.1, -0.1] - [1.3, 0.7, 0.6]|| = 0.6
confidence_2 = 8.0 - 0.6 = 7.4 ⭐⭐⭐⭐

# Rule3: works_at ∧ org_country → nationality
body_sum_3 = [0.6, 0.8, -0.2] + [0.7, -0.1, 0.8] = [1.3, 0.7, 0.6]
distance_3 = ||[1.3, 0.7, 0.6] + [0.0, 0.0, -0.1] - [1.3, 0.7, 0.6]|| = 0.1
confidence_3 = 8.0 - 0.1 = 7.9 ⭐⭐⭐⭐⭐

# Rule4: visits ∧ located_in → nationality（噪声规则）
body_sum_4 = [2.1, 0.3, 1.5] + [0.9, -0.4, 1.2] = [3.0, -0.1, 2.7]
distance_4 = ||[3.0, -0.1, 2.7] + [-2.5, 1.8, 3.2] - [1.3, 0.7, 0.6]|| = 6.4
confidence_4 = 8.0 - 6.4 = 1.6 ⭐

# 总结：
# ✓ 好规则（Rule1,3）学到了高置信度（8.0, 7.9）
# ✓ 一般规则（Rule2）学到了中等置信度（7.4）
# ✓ 坏规则（Rule4）学到了低置信度（1.6）
```

---

## 3. 阶段2：规则Grounding训练

### 3.1 目标

训练MLP网络，学习如何聚合多条规则的预测结果。

**关键**：这个阶段**冻结所有嵌入**，只训练MLP参数。

```python
# 冻结预训练的嵌入
entity_embedding.requires_grad = False
relation_embedding.requires_grad = False
rule_embedding.requires_grad = False

# 只训练MLP
mlp_feature = nn.Parameter(torch.randn(4, 100))  # 4条规则 → 100维特征
score_model = MLP(100, [128, 1])  # MLP: 100 → 128 → 1
```

### 3.2 训练样本

```python
# 训练样本: (Turing, nationality, UK)
sample = [Turing_id, nationality_id, UK_id]
```

### 3.3 前向传播 - Grounding

#### 步骤1：找到适用的规则

```python
query_relation = nationality

applicable_rules = relation2rules[nationality]
# 返回: [Rule1, Rule2, Rule3, Rule4]
```

#### 步骤2：对每条规则进行路径枚举（Grounding）

**Rule1: born_in ∧ city_of → nationality**

```python
# 初始化
current_entities = one_hot(Turing)  # [1, 0, 0, 0, 0, 0, 0]
                                    # 第0位是Turing

# 第1跳: born_in
edge_index_born_in = [[Turing_id, Turing_id],    # 源节点
                      [London_id, Cambridge_id]]  # 目标节点
edge_weight = [1.0, 1.0]

# 使用scatter_add传播
next_entities = scatter_add(
    src=current_entities[edge_index[0]] * edge_weight,  # [1.0, 1.0]
    index=edge_index[1],  # [London_id, Cambridge_id]
    dim_size=num_entities
)
# 结果: [0, 0, 1.0, 1.0, 0, 0, 0]
#           ↑    ↑
#        London Cambridge

current_entities = next_entities

# 第2跳: city_of
edge_index_city_of = [[London_id, Cambridge_id],  # 源节点
                      [UK_id, UK_id]]              # 目标节点
edge_weight = [1.0, 1.0]

# 再次传播
next_entities = scatter_add(
    src=current_entities[edge_index[0]] * edge_weight,  # [1.0, 1.0]
    index=edge_index[1],  # [UK_id, UK_id]
    dim_size=num_entities
)
# 结果: [0, 0, 0, 0, 0, 2.0, 0]
#                       ↑
#                      UK (2条路径！)

grounding_count_Rule1 = next_entities
# grounding_count_Rule1[UK] = 2.0
```

**Rule2: friend_of ∧ nationality → nationality**

```python
# 第1跳: friend_of
# Turing --friend_of--> Church
next_entities = [0, 0, 0, 0, 0, 0, 1.0]  # Church

# 第2跳: nationality
# Church --nationality--> USA
next_entities = [0, 0, 0, 0, 0, 0, 1.0]  # USA

grounding_count_Rule2 = next_entities
# grounding_count_Rule2[UK] = 0.0
# grounding_count_Rule2[USA] = 1.0
```

**Rule3: works_at ∧ org_country → nationality**

```python
# 第1跳: works_at
# Turing --works_at--> Bletchley_Park
next_entities = [0, 0, 0, 0, 1.0, 0, 0]  # Bletchley_Park

# 第2跳: org_country
# Bletchley_Park --org_country--> UK
next_entities = [0, 0, 0, 0, 0, 1.0, 0]  # UK

grounding_count_Rule3 = next_entities
# grounding_count_Rule3[UK] = 1.0
```

**Rule4: visits ∧ located_in → nationality**

```python
# 第1跳: visits
# 没有visits边！
next_entities = [0, 0, 0, 0, 0, 0, 0]

grounding_count_Rule4 = next_entities
# grounding_count_Rule4[UK] = 0.0
```

#### 步骤3：计算规则置信度

```python
# 使用预训练阶段学到的规则嵌入计算置信度
confidence_Rule1 = 8.0  # 从阶段1学到的
confidence_Rule2 = 7.4
confidence_Rule3 = 7.9
confidence_Rule4 = 1.6
```

#### 步骤4：构建Soft Multi-hot Encoding

```python
# 对候选实体UK，构建规则激活向量

v_UK = [
    confidence_Rule1 * grounding_count_Rule1[UK],  # 8.0 × 2.0 = 16.0
    confidence_Rule2 * grounding_count_Rule2[UK],  # 7.4 × 0.0 = 0.0
    confidence_Rule3 * grounding_count_Rule3[UK],  # 7.9 × 1.0 = 7.9
    confidence_Rule4 * grounding_count_Rule4[UK],  # 1.6 × 0.0 = 0.0
]
# v_UK = [16.0, 0.0, 7.9, 0.0]

# 对候选实体USA
v_USA = [
    8.0 × 0.0,  # 0.0
    7.4 × 1.0,  # 7.4
    7.9 × 0.0,  # 0.0
    1.6 × 0.0,  # 0.0
]
# v_USA = [0.0, 7.4, 0.0, 0.0]
```

#### 步骤5：MLP聚合和评分

```python
# 将规则激活向量转换为MLP输入特征
# 使用可学习的矩阵 mlp_feature [4, 100]
feature_UK = torch.mm(v_UK.unsqueeze(0), mlp_feature)  # [1, 100]

# 通过MLP得到规则得分
rule_score_UK = score_model(feature_UK)  # [1, 1]
# 假设输出: 0.85

# 同理计算USA
feature_USA = torch.mm(v_USA.unsqueeze(0), mlp_feature)
rule_score_USA = score_model(feature_USA)
# 假设输出: 0.32

# 所有候选的规则得分
rule_scores = [
    score_UK=0.85,
    score_USA=0.32,
    score_France=0.05,
    ...
]
```

### 3.4 损失计算和反向传播

```python
# 真实标签: UK
target = UK_id

# 使用交叉熵损失（带label smoothing）
loss = cross_entropy_with_smoothing(
    logits=rule_scores,  # [num_entities]
    target=target,
    smoothing=0.2
)

# 反向传播（只更新MLP参数）
loss.backward()
optimizer.step()  # 只更新 mlp_feature 和 score_model

# 训练多个epoch后，MLP学会:
# - 高得分规则（Rule1: 16.0）→ 高贡献
# - 中等规则（Rule3: 7.9）→ 中等贡献
# - 低得分规则（Rule2: 7.4, Rule4: 1.6）→ 低贡献
```

---

## 4. 阶段3：推理预测

### 4.1 测试查询

```python
# 测试: 预测图灵的国籍
query = (Turing, nationality, ?)
```

### 4.2 完整推理流程

#### Part A: KGE推理（使用RotatE）

```python
# 对每个候选实体计算KGE得分

# 候选1: UK
h = entity_embedding[Turing]  # [6]维
r = relation_embedding[nationality]  # [3]维
t_uk = entity_embedding[UK]  # [6]维

distance_uk = RotatE_distance(h, r, t_uk)
kge_score_uk = gamma_fact - distance_uk
# 假设: 9.0 - 0.5 = 8.5

# 候选2: USA
t_usa = entity_embedding[USA]
distance_usa = RotatE_distance(h, r, t_usa)
kge_score_usa = gamma_fact - distance_usa
# 假设: 9.0 - 3.2 = 5.8

# 候选3: France
t_france = entity_embedding[France]
kge_score_france = 9.0 - 7.5 = 1.5
```

#### Part B: 规则推理（使用训练好的Grounding模型）

```python
# 步骤1-4: 与训练阶段相同，进行grounding和构建特征
v_UK = [16.0, 0.0, 7.9, 0.0]
v_USA = [0.0, 7.4, 0.0, 0.0]
v_France = [0.0, 0.0, 0.0, 0.0]

# 步骤5: 通过训练好的MLP计算规则得分
rule_score_uk = MLP(v_UK) = 0.92
rule_score_usa = MLP(v_USA) = 0.35
rule_score_france = MLP(v_France) = 0.05
```

#### Part C: 综合得分

```python
# 超参数
beta = 0.5  # 规则权重

# 最终得分 = KGE得分 + beta × 规则得分
final_score_uk = kge_score_uk + beta * rule_score_uk
               = 8.5 + 0.5 × 0.92
               = 8.5 + 0.46
               = 8.96  ⭐⭐⭐⭐⭐

final_score_usa = kge_score_usa + beta * rule_score_usa
                = 5.8 + 0.5 × 0.35
                = 5.8 + 0.175
                = 5.975  ⭐⭐⭐

final_score_france = kge_score_france + beta * rule_score_france
                   = 1.5 + 0.5 × 0.05
                   = 1.525  ⭐

# 排序
排名1: UK (8.96)     ✓ 正确答案！
排名2: USA (5.975)
排名3: France (1.525)
...

# 预测结果: UK
```

### 4.3 为什么UK得分最高？

**分解分析**：

```
UK的得分来源:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. KGE部分贡献: 8.5
   - 实体和关系嵌入的几何距离很近
   - 说明: 模型从三元组中学到了"Turing-nationality-UK"的模式

2. 规则部分贡献: 0.46 (= 0.5 × 0.92)
   来自3条激活的规则:

   Rule1 (born_in ∧ city_of):
     • 路径数: 2条
       - Turing → London → UK
       - Turing → Cambridge → UK
     • 置信度: 8.0
     • 贡献: 8.0 × 2.0 = 16.0  ← 最大贡献！

   Rule3 (works_at ∧ org_country):
     • 路径数: 1条
       - Turing → Bletchley_Park → UK
     • 置信度: 7.9
     • 贡献: 7.9 × 1.0 = 7.9

   Rule2, Rule4: 未激活或置信度低

   MLP聚合: [16.0, 0.0, 7.9, 0.0] → 0.92

3. 总分: 8.5 + 0.46 = 8.96 ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USA的得分来源:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. KGE部分: 5.8
   - 距离较远（没学到Turing-USA的关联）

2. 规则部分: 0.175 (= 0.5 × 0.35)
   只有Rule2激活:

   Rule2 (friend_of ∧ nationality):
     • 路径: Turing → Church → USA
     • 置信度: 7.4（中等，朋友的国籍不一定相同）
     • 贡献: 7.4 × 1.0 = 7.4

   MLP聚合: [0.0, 7.4, 0.0, 0.0] → 0.35

3. 总分: 5.8 + 0.175 = 5.975 ✗
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

结论:
UK胜出是因为:
✓ KGE支持强 (8.5 vs 5.8)
✓ 规则支持强 (0.92 vs 0.35)
✓ 多条高质量规则 + 多条路径 = 高置信度
```

---

## 5. 完整数据流

### 5.1 端到端流程图

```
┌─────────────────────────────────────────────────────────────┐
│                     输入: 知识图谱                            │
│  Entities: Turing, London, UK, ...                          │
│  Relations: born_in, city_of, nationality, ...              │
│  Triples: (Turing, born_in, London), ...                    │
│  Rules: born_in ∧ city_of → nationality, ...                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              阶段1: 预训练 (30000 steps)                      │
│                                                             │
│  Batch训练:                                                  │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ Triplet数据      │  │ Rule数据         │               │
│  │ (h, r, t)       │  │ [rule_id, ...]  │               │
│  └──────────────────┘  └──────────────────┘               │
│           ↓                     ↓                           │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ RotatE Loss     │  │ RulE Loss       │               │
│  │ γ - ||h◦r-t||   │  │ γ - ||Σrᵢ+R-r|| │               │
│  └──────────────────┘  └──────────────────┘               │
│           ↓                     ↓                           │
│          Loss = L_triplet + α × L_rule                     │
│                       ↓                                     │
│              更新所有嵌入参数                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
                   保存checkpoint
                            ↓
┌─────────────────────────────────────────────────────────────┐
│           阶段2: Grounding训练 (20 epochs)                    │
│                                                             │
│  冻结嵌入，只训练MLP:                                         │
│  entity_embedding.requires_grad = False                     │
│  relation_embedding.requires_grad = False                   │
│  rule_embedding.requires_grad = False                       │
│                                                             │
│  Sample: (Turing, nationality, UK)                         │
│           ↓                                                 │
│  ┌─────────────────────────────────────────────┐          │
│  │  对每条规则进行Grounding                      │          │
│  │                                              │          │
│  │  Rule1: born_in ∧ city_of                   │          │
│  │  Turing → London → UK     (2条路径)          │          │
│  │  Turing → Cambridge → UK                     │          │
│  │  count[UK] = 2.0                            │          │
│  │                                              │          │
│  │  Rule2: friend_of ∧ nationality             │          │
│  │  count[USA] = 1.0                           │          │
│  │                                              │          │
│  │  Rule3: works_at ∧ org_country              │          │
│  │  count[UK] = 1.0                            │          │
│  └─────────────────────────────────────────────┘          │
│           ↓                                                 │
│  ┌─────────────────────────────────────────────┐          │
│  │  计算规则置信度（使用rule_embedding）         │          │
│  │  confidence = γ - distance                  │          │
│  └─────────────────────────────────────────────┘          │
│           ↓                                                 │
│  ┌─────────────────────────────────────────────┐          │
│  │  构建Soft Multi-hot Encoding                │          │
│  │  v[i] = confidence[i] × count[i]            │          │
│  │  v_UK = [16.0, 0.0, 7.9, 0.0]               │          │
│  └─────────────────────────────────────────────┘          │
│           ↓                                                 │
│  ┌─────────────────────────────────────────────┐          │
│  │  MLP聚合 → 规则得分                          │          │
│  │  feature = v × mlp_feature                  │          │
│  │  score = MLP(feature)                       │          │
│  └─────────────────────────────────────────────┘          │
│           ↓                                                 │
│       Cross-Entropy Loss                                   │
│           ↓                                                 │
│   只更新MLP参数（mlp_feature, score_model）                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
                 保存grounding.pt
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  阶段3: 推理预测                              │
│                                                             │
│  Query: (Turing, nationality, ?)                           │
│           ↓                                                 │
│  ┌────────────────────┐  ┌─────────────────────┐          │
│  │  KGE推理 (RotatE)  │  │  规则推理 (Grounding) │          │
│  │                    │  │                     │          │
│  │  对每个候选:        │  │  1. 路径枚举         │          │
│  │  score = γ-||h◦r-t││  │  2. 计算置信度       │          │
│  │                    │  │  3. 构建特征向量     │          │
│  │  UK: 8.5          │  │  4. MLP聚合         │          │
│  │  USA: 5.8         │  │                     │          │
│  │  France: 1.5      │  │  UK: 0.92           │          │
│  └────────────────────┘  │  USA: 0.35          │          │
│                          │  France: 0.05       │          │
│                          └─────────────────────┘          │
│           ↓                        ↓                        │
│  ┌─────────────────────────────────────────────┐          │
│  │         综合得分                             │          │
│  │  final = kge_score + β × rule_score         │          │
│  │                                              │          │
│  │  UK: 8.5 + 0.5×0.92 = 8.96  ← Winner!      │          │
│  │  USA: 5.8 + 0.5×0.35 = 5.975               │          │
│  │  France: 1.5 + 0.5×0.05 = 1.525            │          │
│  └─────────────────────────────────────────────┘          │
│           ↓                                                 │
│      预测结果: UK ✓                                         │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 关键参数汇总

```python
# 模型参数
═══════════════════════════════════════
entity_embedding: [7, 6]     # 7个实体，6维（复数）
relation_embedding: [6, 3]   # 6个关系，3维（角度）
rule_embedding: [4, 3]       # 4条规则，3维（残差）
mlp_feature: [4, 100]        # 规则特征映射
score_model: MLP(100→128→1)  # 得分网络

# 超参数
═══════════════════════════════════════
hidden_dim = 3               # 嵌入维度
gamma_fact = 9.0             # Triplet margin
gamma_rule = 8.0             # Rule margin
alpha = 1.0                  # Rule loss权重
beta = 0.5                   # 推理时规则得分权重
learning_rate = 0.00005      # 预训练学习率
g_lr = 0.0001                # Grounding学习率
max_steps = 30000            # 预训练步数
num_iters = 20               # Grounding轮数
batch_size = 256             # Triplet batch size
rule_batch_size = 128        # Rule batch size

# 训练后的嵌入（示例值）
═══════════════════════════════════════
relation_embedding:
  born_in:      [0.5, 1.2, -0.3]
  city_of:      [0.8, -0.5, 0.9]
  nationality:  [1.3, 0.7, 0.6]
  works_at:     [0.6, 0.8, -0.2]
  org_country:  [0.7, -0.1, 0.8]
  friend_of:    [0.3, 0.4, 0.1]

rule_embedding:
  Rule1: [0.0, 0.0, 0.0]    # 完美规则
  Rule2: [-0.2, 0.1, -0.1]  # 需要小校正
  Rule3: [0.0, 0.0, -0.1]   # 需要微调
  Rule4: [-2.5, 1.8, 3.2]   # 噪声规则

规则置信度:
  Rule1: 8.0 ⭐⭐⭐⭐⭐
  Rule2: 7.4 ⭐⭐⭐⭐
  Rule3: 7.9 ⭐⭐⭐⭐⭐
  Rule4: 1.6 ⭐
```

### 5.3 推理结果解释

```
查询: (Turing, nationality, ?)
═══════════════════════════════════════════════════════

预测答案: UK
置信度: 8.96 / 10
排名: 1 / 7

═══════════════════════════════════════════════════════
为什么预测UK？
───────────────────────────────────────────────────────

✓ 基于嵌入的证据 (KGE得分: 8.5):
  - entity_embedding和relation_embedding的几何关系
  - 从大量三元组中学习到的模式

✓ 基于规则的证据 (规则得分: 0.92):

  Rule1 (置信度 8.0): born_in ∧ city_of → nationality
    路径1: Turing → born_in → London → city_of → UK
    路径2: Turing → born_in → Cambridge → city_of → UK
    贡献: 8.0 × 2.0 = 16.0 ⭐⭐⭐

  Rule3 (置信度 7.9): works_at ∧ org_country → nationality
    路径1: Turing → works_at → Bletchley_Park → org_country → UK
    贡献: 7.9 × 1.0 = 7.9 ⭐⭐⭐

  总规则贡献: MLP([16.0, 0.0, 7.9, 0.0]) = 0.92

✓ 综合判断:
  最终得分 = 8.5 (KGE) + 0.5 × 0.92 (规则) = 8.96

═══════════════════════════════════════════════════════
为什么不是USA？
───────────────────────────────────────────────────────

✗ KGE得分低 (5.8):
  - 训练数据中没有(Turing, nationality, USA)
  - 嵌入空间距离远

✗ 规则支持弱 (0.35):
  只有Rule2激活:
  Rule2 (置信度 7.4): friend_of ∧ nationality → nationality
    路径: Turing → friend_of → Church → nationality → USA
    贡献: 7.4 × 1.0 = 7.4

  问题: 置信度中等（朋友的国籍≠自己国籍）
       只有1条规则支持

✗ 总分: 5.8 + 0.5 × 0.35 = 5.975 < 8.96

═══════════════════════════════════════════════════════
```

---

## 6. 核心机制总结

### 6.1 三个关键组件

```
1. 知识图谱嵌入 (RotatE)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   作用: 学习实体和关系的向量表示
   输入: 三元组 (h, r, t)
   输出: 得分 γ - ||h ◦ r - t||
   优势: 泛化能力强，能处理不完整数据

2. 规则嵌入 (Rule Embedding)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   作用: 学习规则的质量/置信度
   输入: 规则 r₁ ∧ r₂ → r₃
   输出: 置信度 γ - ||Σrᵢ + R - r₃||
   优势: 区分好规则和坏规则

3. 软规则推理 (Soft Grounding)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   作用: 根据规则质量和路径数量加权推理
   输入: 规则 + 图结构
   输出: 加权后的规则得分
   优势: 鲁棒，可解释
```

### 6.2 为什么RulE有效？

```
传统KGE的问题:
❌ 只看三元组，忽略逻辑规则
❌ 黑盒，不可解释
❌ 泛化能力有限

传统规则推理的问题:
❌ 硬匹配，太脆弱
❌ 无法处理噪声规则
❌ 规则必须完全匹配才生效

RulE的优势:
✓ 联合学习：KGE + 规则，互相增强
✓ 软推理：根据置信度加权，不是0/1
✓ 自动评估：规则嵌入学习规则质量
✓ 可解释：可以看到哪些规则被激活
✓ 鲁棒性：多条规则投票，降低噪声影响
```

### 6.3 关键创新点

```
1. 统一嵌入空间
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   实体、关系、规则都嵌入到同一空间
   → 可以端到端联合优化

2. 规则置信度学习
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   不是人工设定规则权重
   → 从数据中自动学习哪些规则可信

3. 软规则推理
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   不是硬性应用规则（if-then）
   → 根据置信度和路径数加权（soft voting）

4. 分阶段训练
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   先学嵌入，再学聚合
   → 更稳定，更容易收敛
```

---

## 7. 完整示例的Python伪代码

```python
import torch
import torch.nn as nn
from torch_scatter import scatter_add

# ========== 阶段1: 预训练 ==========
class PreTraining:
    def __init__(self, num_entities=7, num_relations=6, num_rules=4, dim=3):
        # 初始化嵌入
        self.entity_emb = nn.Embedding(num_entities, dim * 2)  # 复数
        self.relation_emb = nn.Embedding(num_relations, dim)   # 角度
        self.rule_emb = nn.Parameter(torch.zeros(num_rules, dim))

        self.gamma_fact = 9.0
        self.gamma_rule = 8.0

    def forward_triplet(self, h, r, t):
        """计算Triplet得分（RotatE）"""
        h_emb = self.entity_emb(h)  # [batch, dim*2]
        r_emb = self.relation_emb(r)  # [batch, dim]
        t_emb = self.entity_emb(t)

        # 分离复数
        h_re, h_im = h_emb[..., :3], h_emb[..., 3:]
        t_re, t_im = t_emb[..., :3], t_emb[..., 3:]

        # 旋转
        r_re, r_im = torch.cos(r_emb), torch.sin(r_emb)
        result_re = h_re * r_re - h_im * r_im
        result_im = h_re * r_im + h_im * r_re

        # 距离
        distance = torch.sqrt(
            (result_re - t_re)**2 + (result_im - t_im)**2
        ).sum(dim=-1)

        return self.gamma_fact - distance

    def forward_rule(self, rule_sample):
        """计算规则得分"""
        rule_id = rule_sample[:, 0]
        rule_length = rule_sample[:, 1]
        rule_head = rule_sample[:, 2]
        rule_body = rule_sample[:, 3:]

        # 累加规则体
        body_sum = torch.zeros_like(self.relation_emb(rule_head))
        for i in range(rule_length.max()):
            mask = (i < rule_length)
            body_sum[mask] += self.relation_emb(rule_body[mask, i])

        # 计算距离
        rule_emb_vec = self.rule_emb[rule_id]
        head_emb = self.relation_emb(rule_head)
        distance = torch.norm(body_sum + rule_emb_vec - head_emb, p=1, dim=-1)

        return self.gamma_rule - distance

    def train(self, triplet_data, rule_data, steps=30000):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)

        for step in range(steps):
            # Triplet batch
            h, r, t_pos, t_neg = triplet_data.sample()
            pos_score = self.forward_triplet(h, r, t_pos)
            neg_score = self.forward_triplet(h, r, t_neg)
            loss_triplet = torch.relu(1.0 - (pos_score - neg_score)).mean()

            # Rule batch
            rule_pos, rule_neg = rule_data.sample()
            pos_rule_score = self.forward_rule(rule_pos)
            neg_rule_score = self.forward_rule(rule_neg)
            loss_rule = torch.relu(1.0 - (pos_rule_score - neg_rule_score)).mean()

            # 联合损失
            loss = loss_triplet + loss_rule

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1000 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")


# ========== 阶段2: Grounding训练 ==========
class GroundingTraining:
    def __init__(self, pretrained_model, num_rules=4, mlp_dim=100):
        # 冻结预训练参数
        self.entity_emb = pretrained_model.entity_emb
        self.relation_emb = pretrained_model.relation_emb
        self.rule_emb = pretrained_model.rule_emb

        for param in [self.entity_emb, self.relation_emb, self.rule_emb]:
            param.requires_grad = False

        # 可训练的MLP参数
        self.mlp_feature = nn.Parameter(torch.randn(num_rules, mlp_dim))
        self.score_model = nn.Sequential(
            nn.Linear(mlp_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.gamma_rule = 8.0

    def grounding(self, h, rule_body, graph):
        """路径枚举"""
        current = torch.zeros(graph.num_entities)
        current[h] = 1.0

        for relation in rule_body:
            # 获取邻接表
            edge_index, edge_weight = graph.adjacency[relation]

            # 传播
            current = scatter_add(
                src=current[edge_index[0]] * edge_weight,
                index=edge_index[1],
                dim=0,
                dim_size=graph.num_entities
            )

        return current  # grounding count

    def compute_confidence(self, rule_id, rule_body, rule_head):
        """计算规则置信度"""
        body_sum = sum([self.relation_emb(r) for r in rule_body])
        rule_vec = self.rule_emb[rule_id]
        head_vec = self.relation_emb(rule_head)

        distance = torch.norm(body_sum + rule_vec - head_vec, p=1)
        return self.gamma_rule - distance

    def forward(self, h, query_r, graph, applicable_rules):
        """完整的grounding推理"""
        # 对每个候选实体构建特征
        num_entities = graph.num_entities
        features = torch.zeros(num_entities, self.mlp_feature.size(1))

        for rule in applicable_rules:
            # 1. 路径枚举
            grounding_count = self.grounding(h, rule.body, graph)

            # 2. 规则置信度
            confidence = self.compute_confidence(rule.id, rule.body, rule.head)

            # 3. 构建特征
            rule_contribution = confidence * grounding_count  # [num_entities]
            rule_feature = self.mlp_feature[rule.id]  # [mlp_dim]

            # 累加
            features += rule_contribution.unsqueeze(-1) * rule_feature

        # 4. MLP评分
        scores = self.score_model(features).squeeze(-1)  # [num_entities]

        return scores

    def train(self, train_data, graph, epochs=20):
        optimizer = torch.optim.Adam(
            list(self.score_model.parameters()) + [self.mlp_feature],
            lr=1e-4
        )

        for epoch in range(epochs):
            for h, r, t in train_data:
                # 找到适用的规则
                applicable_rules = graph.relation2rules[r]

                # 前向传播
                scores = self.forward(h, r, graph, applicable_rules)

                # 交叉熵损失
                loss = F.cross_entropy(scores.unsqueeze(0), t.unsqueeze(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# ========== 阶段3: 推理 ==========
class Inference:
    def __init__(self, pretrained_model, grounding_model):
        self.pretrained = pretrained_model
        self.grounding = grounding_model
        self.beta = 0.5  # 规则权重

    def predict(self, h, r, graph):
        """完整推理流程"""
        # 1. KGE得分
        kge_scores = []
        for t in range(graph.num_entities):
            score = self.pretrained.forward_triplet(h, r, t)
            kge_scores.append(score.item())
        kge_scores = torch.tensor(kge_scores)

        # 2. 规则得分
        applicable_rules = graph.relation2rules[r]
        rule_scores = self.grounding.forward(h, r, graph, applicable_rules)

        # 3. 综合得分
        final_scores = kge_scores + self.beta * rule_scores

        # 4. 排序
        ranked = torch.argsort(final_scores, descending=True)

        return ranked, final_scores


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 创建知识图谱
    graph = KnowledgeGraph(
        entities=["Turing", "London", "Cambridge", "Bletchley_Park", "UK", "USA", "Church"],
        relations=["born_in", "city_of", "nationality", "works_at", "org_country", "friend_of"],
        triples=[
            ("Turing", "born_in", "London"),
            ("Turing", "born_in", "Cambridge"),
            ("London", "city_of", "UK"),
            ("Cambridge", "city_of", "UK"),
            ("Turing", "works_at", "Bletchley_Park"),
            ("Bletchley_Park", "org_country", "UK"),
            ("Turing", "friend_of", "Church"),
            ("Church", "nationality", "USA"),
            ("Turing", "nationality", "UK"),
        ],
        rules=[
            Rule(0, "nationality", ["born_in", "city_of"]),
            Rule(1, "nationality", ["friend_of", "nationality"]),
            Rule(2, "nationality", ["works_at", "org_country"]),
            Rule(3, "nationality", ["visits", "located_in"]),
        ]
    )

    # 阶段1: 预训练
    print("=== 阶段1: 预训练 ===")
    pretrained_model = PreTraining()
    pretrained_model.train(triplet_data, rule_data, steps=30000)

    # 阶段2: Grounding训练
    print("\n=== 阶段2: Grounding训练 ===")
    grounding_model = GroundingTraining(pretrained_model)
    grounding_model.train(train_data, graph, epochs=20)

    # 阶段3: 推理
    print("\n=== 阶段3: 推理预测 ===")
    inference = Inference(pretrained_model, grounding_model)

    # 查询: (Turing, nationality, ?)
    h = graph.entity2id["Turing"]
    r = graph.relation2id["nationality"]

    ranked, scores = inference.predict(h, r, graph)

    print(f"查询: (Turing, nationality, ?)")
    print(f"\n预测结果:")
    for i in range(3):
        entity_id = ranked[i]
        entity_name = graph.id2entity[entity_id]
        score = scores[entity_id]
        print(f"  排名{i+1}: {entity_name} (得分: {score:.4f})")

    # 输出:
    # 查询: (Turing, nationality, ?)
    #
    # 预测结果:
    #   排名1: UK (得分: 8.9600)
    #   排名2: USA (得分: 5.9750)
    #   排名3: France (得分: 1.5250)
```

---

## 8. 总结

### 8.1 核心流程回顾

```
输入数据
  ↓
[预训练] 学习嵌入（entity, relation, rule）
  ↓
[Grounding训练] 学习MLP聚合规则
  ↓
[推理] KGE得分 + 规则得分 → 最终预测
```

### 8.2 为什么这个例子重要？

```
✓ 完整展示了3个阶段的数据流
✓ 用具体数字说明每一步计算
✓ 解释了为什么UK得分最高
✓ 对比了好规则和坏规则的行为
✓ 展示了软规则推理的优势
```

### 8.3 关键要点

```
1. 规则嵌入学习"规则质量"
   → 好规则（Rule1）置信度8.0
   → 坏规则（Rule4）置信度1.6

2. 路径枚举统计"支持证据"
   → UK有2条路径（Rule1）+ 1条路径（Rule3）
   → USA只有1条路径（Rule2）

3. 软规则推理 = 置信度 × 路径数
   → 质量 × 数量 = 贡献
   → 多条规则投票，综合决策

4. KGE和规则互补
   → KGE提供基础得分
   → 规则提供逻辑推理
   → 综合得分更准确
```

---

**这就是RulE模型的完整工作流程！** 🎯
