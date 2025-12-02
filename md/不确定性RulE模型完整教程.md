# 不确定性RulE模型完整教程

**Uncertainty-aware Rule Embedding for Knowledge Graph Reasoning**

---

## 📋 目录

1. [模型概述](#1-模型概述)
2. [核心创新](#2-核心创新)
3. [数学公式详解](#3-数学公式详解)
4. [完整算法流程](#4-完整算法流程)
5. [参数配置指南](#5-参数配置指南)
6. [完整示例](#6-完整示例)
7. [实现细节](#7-实现细节)
8. [常见问题](#8-常见问题)

---

## 1. 模型概述

### 1.1 背景

**原始RulE的问题**：
```
规则置信度计算: w_i = γ_rule - ||r_body_sum + R_i - r_head||

问题:
❌ 确定性标量，无法表达不确定性
❌ 数据不足的规则和数据充足的规则得到相同类型的置信度
❌ 无法区分"置信度0.7因为质量中等"和"置信度0.7但样本太少不确定"
```

**不确定性RulE的解决方案**：
```
规则置信度建模为概率分布: w_i ~ N(μ_i, σ_i²)

优势:
✅ μ_i: 置信度均值（质量）
✅ σ_i: 置信度不确定性（数据量/可靠性）
✅ 自动降权低质量/低数据量的规则
✅ 提供可解释性
```

### 1.2 模型架构图

```
┌─────────────────────────────────────────────────────────────┐
│                   不确定性RulE架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  输入: 规则 [rule_id, r_head, r_body_1, r_body_2, ...]      │
│                           ↓                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  规则嵌入模块 (继承自原始RulE)                       │    │
│  │                                                       │    │
│  │  R_i = rule_emb[rule_id]           [rule_dim]       │    │
│  │  r_body_sum = Σ relation_emb[r_j]  [hidden_dim]     │    │
│  │  r_head_emb = relation_emb[r_head] [hidden_dim]     │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↓                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  【新增】不确定性建模模块                            │    │
│  │                                                       │    │
│  │  features = concat([R_i, r_body_sum])                │    │
│  │            [rule_dim + hidden_dim]                   │    │
│  │                                                       │    │
│  │         ┌──────────────┐     ┌──────────────┐       │    │
│  │         │  μ_network   │     │logvar_network│       │    │
│  │         │  3层MLP      │     │  3层MLP      │       │    │
│  │         └──────────────┘     └──────────────┘       │    │
│  │               ↓                      ↓               │    │
│  │             μ_i                   log(σ²_i)          │    │
│  │           [1]标量                [1]标量            │    │
│  │                                                       │    │
│  │  σ_i = exp(0.5 × log(σ²_i))                         │    │
│  │                                                       │    │
│  │  规则置信度分布: w_i ~ N(μ_i, σ_i²)                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↓                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  重参数化采样 (训练时)                               │    │
│  │                                                       │    │
│  │  for k in 1..num_samples:                            │    │
│  │      ε_k ~ N(0, 1)                                   │    │
│  │      w_k = μ_i + σ_i × ε_k                           │    │
│  │                                                       │    │
│  │  w_i = mean([w_1, w_2, ..., w_num_samples])         │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↓                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  损失函数                                             │    │
│  │                                                       │    │
│  │  loss = loss_kge                                     │    │
│  │       + λ_rule × loss_rule(w_i)                      │    │
│  │       + λ_uncertainty × (β_kl×loss_kl + β_σ×loss_σ)  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 核心创新

### 2.1 不确定性建模

#### 创新点1: 概率分布表示

**原始RulE**:
```python
w_i = γ_rule - ||r_body_sum + R_i - r_head_emb||  # 标量
```

**不确定性RulE**:
```python
w_i ~ N(μ_i, σ_i²)  # 概率分布

其中:
  μ_i = mu_network([R_i, r_body_sum])      # 置信度均值
  σ_i = exp(0.5 × logvar_network([R_i, r_body_sum]))  # 标准差
```

#### 创新点2: 重参数化采样

**问题**: 如何从分布中采样且保持可微分？

**解决**: Reparameterization Trick (VAE同款技术)
```python
# 不可微分的采样:
w_i = sample(N(μ_i, σ_i²))  ✗ 无法反向传播

# 重参数化采样:
ε ~ N(0, 1)                 # 固定分布采样
w_i = μ_i + σ_i × ε         ✓ 可微分!

# 梯度流:
∂w_i/∂μ_i = 1               # 可以计算
∂w_i/∂σ_i = ε               # 可以计算
```

#### 创新点3: 不确定性正则化

**KL散度约束**:
```python
KL(N(μ_i, σ_i²) || N(0, 1)) = 0.5 × (μ_i² + σ_i² - log(σ_i²) - 1)

作用:
- 防止σ_i → 0 (方差塌缩)
- 防止μ_i → ∞ (过度自信)
- 保持适度不确定性
```

**支持数驱动约束**:
```python
target_σ_i = λ_0 / (1 + log(support_count_i + 1))

loss_σ = (σ_i - target_σ_i)²

示例:
- support_count = 1000 → target_σ = 0.14  (低不确定性)
- support_count = 10   → target_σ = 0.42  (高不确定性)
```

---

## 3. 数学公式详解

### 3.1 符号表

| 符号 | 维度 | 含义 |
|------|------|------|
| `R_i` | `[rule_dim]` | 规则i的嵌入向量 |
| `r_j` | `[hidden_dim]` | 关系j的嵌入向量 |
| `r_body` | `[L, hidden_dim]` | 规则体中L个关系的嵌入 |
| `r_head` | `[hidden_dim]` | 规则头关系的嵌入 |
| `μ_i` | `标量` | 规则i置信度的均值 |
| `σ_i` | `标量` | 规则i置信度的标准差 |
| `σ²_i` | `标量` | 规则i置信度的方差 |
| `w_i` | `标量` | 规则i的置信度（采样值） |
| `ε` | `标量` | 标准正态分布采样 N(0,1) |
| `γ_rule` | `标量` | 规则margin超参数 |
| `λ_0` | `标量` | 支持数依赖系数 |

### 3.2 前向传播公式

#### 步骤1: 规则表示

给定规则: `rule_i = [rule_id, r_head, r_1, r_2, ..., r_L]`

```
1. 获取规则嵌入:
   R_i = rule_emb[rule_id]                    ∈ ℝ^{rule_dim}

2. 获取关系嵌入:
   r_head_emb = relation_emb[r_head]          ∈ ℝ^{hidden_dim}
   r_body_embs = [relation_emb[r_j]]_{j=1}^L  ∈ ℝ^{L × hidden_dim}

3. 聚合规则体:
   r_body_sum = Σ_{j=1}^L r_body_embs[j]      ∈ ℝ^{hidden_dim}
```

#### 步骤2: 构建输入特征

```
features = [R_i ; r_body_sum]                 ∈ ℝ^{rule_dim + hidden_dim}

其中 [; ] 表示拼接操作
```

#### 步骤3: 计算分布参数

**均值网络**:
```
μ_i = μ_network(features)
    = W_3 · ReLU(W_2 · ReLU(W_1 · features + b_1) + b_2) + b_3

    其中:
    W_1 ∈ ℝ^{256 × (rule_dim + hidden_dim)}
    W_2 ∈ ℝ^{128 × 256}
    W_3 ∈ ℝ^{1 × 128}
```

**方差网络**:
```
log(σ²_i) = logvar_network(features)
          = V_3 · ReLU(V_2 · ReLU(V_1 · features + c_1) + c_2) + c_3

σ_i = exp(0.5 × log(σ²_i)) = √(σ²_i)

注: 输出log(σ²)而非σ保证σ²>0
```

#### 步骤4: 重参数化采样

**训练时** (采样num_samples次):
```
for k = 1 to num_samples:
    ε_k ~ N(0, 1)                    # 标准正态采样
    w_k = μ_i + σ_i × ε_k            # 重参数化

w_i = (1/num_samples) × Σ_{k=1}^{num_samples} w_k    # 平均
```

**推理时** (不采样):
```
w_i = μ_i                            # 直接用均值
```

#### 步骤5: 规则距离计算

```
d_rule = ||r_body_sum + R_i - r_head_emb||_2

这是原始RulE的规则距离公式（保持不变）
```

### 3.3 损失函数

#### 总损失

```
L_total = L_kge + λ_rule × L_rule + λ_uncertainty × L_uncertainty
```

#### 3.3.1 KGE损失 (与原始RulE相同)

```
L_kge = -log σ(score_pos) - Σ_{i=1}^N p_i × log σ(-score_neg_i)

其中:
  score_pos = γ_fact - ||h ∘ r - t||          # RotatE正样本分数
  score_neg = γ_fact - ||h' ∘ r - t'||        # 负样本分数
  p_i = softmax(α × score_neg_i)              # 对抗权重
  σ(x) = 1/(1 + e^{-x})                       # sigmoid函数
```

#### 3.3.2 规则损失 (修改)

**原始RulE**:
```
L_rule = -log σ(γ_rule - d_rule) - Σ log σ(-(γ_rule - d_rule_neg))
```

**不确定性RulE**:
```
score_rule = σ(w_i) × (γ_rule - d_rule)

L_rule = -log σ(score_rule) - Σ log σ(-score_rule_neg)

其中:
  w_i 是采样的置信度（不是固定公式）
  σ(w_i) ∈ [0, 1] 作为置信度的归一化
```

**关键改动**: 使用学习到的置信度 `w_i` 而非固定公式

#### 3.3.3 不确定性损失 (新增)

```
L_uncertainty = β_kl × L_kl + β_σ × L_σ
```

**KL散度损失**:
```
L_kl = 0.5 × Σ_{i=1}^{num_rules} (μ_i² + σ_i² - log(σ_i²) - 1)

推导 (KL散度公式):
KL(N(μ, σ²) || N(0, 1)) = ∫ N(μ,σ²) log[N(μ,σ²)/N(0,1)] dx
                         = 0.5 × (μ² + σ² - log(σ²) - 1)
```

**方差匹配损失**:
```
L_σ = Σ_{i=1}^{num_rules} (σ_i - σ_target_i)²

其中:
  σ_target_i = λ_0 / (1 + log(support_count_i + 1))

  support_count_i = 规则i在训练集中的支持路径数
```

### 3.4 梯度计算

#### 对μ的梯度

```
∂L/∂μ_i = ∂L/∂w_i × ∂w_i/∂μ_i
        = ∂L/∂w_i × 1
        = ∂L/∂w_i

直接传播!
```

#### 对σ的梯度

```
∂L/∂σ_i = ∂L/∂w_i × ∂w_i/∂σ_i
        = ∂L/∂w_i × ε

通过采样的ε传播
```

#### 对log(σ²)的梯度

```
∂L/∂log(σ²_i) = ∂L/∂σ_i × ∂σ_i/∂log(σ²_i)
               = ∂L/∂σ_i × 0.5 × σ_i

链式法则
```

---

## 4. 完整算法流程

### 4.1 预训练阶段

```python
算法1: 不确定性RulE预训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

输入:
  - KG三元组: {(h, r, t)}
  - 规则集合: {rule_i}
  - 超参数: γ_fact, γ_rule, λ_rule, λ_uncertainty, β_kl, β_σ, λ_0

输出:
  - 实体嵌入: entity_emb
  - 关系嵌入: relation_emb
  - 规则嵌入: rule_emb
  - 不确定性网络: μ_network, logvar_network

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 初始化:
   entity_emb ~ Uniform(-ε_fact, ε_fact)
   relation_emb ~ Uniform(-ε_fact, ε_fact)
   rule_emb ~ Kaiming
   μ_network ~ Kaiming
   logvar_network ~ Kaiming

2. 预计算支持数:
   for each rule_i:
       support_count[i] = count_support_paths(rule_i, training_set)
       σ_target[i] = λ_0 / (1 + log(support_count[i] + 1))

3. For epoch = 1 to max_epochs:

   3.1 采样三元组batch: batch_triplets

   3.2 计算KGE损失:
       L_kge = compute_rotate_loss(batch_triplets)

   3.3 采样规则batch: batch_rules

   3.4 For each rule_i in batch_rules:

       a) 获取规则表示:
          R_i = rule_emb[rule_i.id]
          r_body_sum = Σ relation_emb[r_j]
          r_head_emb = relation_emb[rule_i.head]

       b) 构建特征:
          features = concat([R_i, r_body_sum])

       c) 计算分布参数:
          μ_i = μ_network(features)
          log_σ²_i = logvar_network(features)
          σ_i = exp(0.5 × log_σ²_i)

       d) 重参数化采样:
          w_samples = []
          for k = 1 to num_samples:
              ε_k ~ N(0, 1)
              w_k = μ_i + σ_i × ε_k
              w_samples.append(w_k)
          w_i = mean(w_samples)

       e) 计算规则距离:
          d_rule = ||r_body_sum + R_i - r_head_emb||

       f) 规则损失:
          score_rule = sigmoid(w_i) × (γ_rule - d_rule)
          L_rule += -log(sigmoid(score_rule))

   3.5 计算不确定性损失:

       a) KL散度损失:
          L_kl = 0.5 × Σ(μ_i² + σ_i² - log(σ_i²) - 1)

       b) 方差匹配损失:
          L_σ = Σ(σ_i - σ_target[i])²

       c) 总不确定性损失:
          L_uncertainty = β_kl × L_kl + β_σ × L_σ

   3.6 总损失:
       L_total = L_kge + λ_rule × L_rule + λ_uncertainty × L_uncertainty

   3.7 反向传播并更新:
       optimizer.zero_grad()
       L_total.backward()
       optimizer.step()

   3.8 验证集评估:
       if epoch % eval_interval == 0:
           MRR = evaluate(valid_set)
           if MRR > best_MRR:
               save_checkpoint()

4. 返回训练好的模型
```

### 4.2 Grounding阶段

```python
算法2: Grounding训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

输入:
  - 预训练模型: entity_emb, relation_emb, rule_emb, μ_network, logvar_network
  - KG三元组: {(h, r, t)}
  - 超参数: g_lr, g_batch_size, num_iters, smoothing

输出:
  - MLP特征: mlp_feature
  - 评分网络: score_model

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 冻结预训练参数:
   entity_emb.requires_grad = False
   relation_emb.requires_grad = False
   rule_emb.requires_grad = False
   μ_network.requires_grad = False
   logvar_network.requires_grad = False

2. 初始化可训练参数:
   mlp_feature ~ Kaiming  [num_rules, mlp_rule_dim]
   score_model ~ Kaiming  MLP(mlp_rule_dim → 1)

3. 预计算规则置信度:
   for each rule_i:
       features = concat([rule_emb[i], r_body_sum[i]])
       μ_i = μ_network(features)
       σ_i = exp(0.5 × logvar_network(features))
       # 保存μ_i用于推理

4. For iter = 1 to num_iters:

   4.1 For each batch in train_loader:

       a) 解包batch:
          all_h, all_r, all_t, edges_to_remove = batch

       b) 检索适用规则:
          query_r = all_r[0]
          applicable_rules = relation2rules[query_r]

       c) 初始化特征张量:
          features = zeros([batch_size, num_entities, mlp_rule_dim])

       d) For each rule in applicable_rules:

          i. Grounding:
             counts = graph.grounding(all_h, rule.body, edges_to_remove)
             # counts: [batch_size, num_entities]

          ii. 获取规则置信度(使用预计算的μ):
              w_i = μ[rule.id]  # 推理时不采样

          iii. 累加特征:
               rule_feat = mlp_feature[rule.id]
               for each entity e:
                   if counts[e] > 0:
                       features[e] += w_i × counts[e] × rule_feat

       e) 聚合与归一化:
          features = ReLU(LayerNorm(features))

       f) MLP打分:
          scores = score_model(features)  # [batch_size, num_entities]

       g) 计算损失(标签平滑交叉熵):
          loss = CrossEntropy(scores, all_t, smoothing=smoothing)

       h) 反向传播:
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

   4.2 验证:
       if iter % eval_interval == 0:
           MRR = evaluate(valid_set)

5. 保存grounding模型
```

### 4.3 推理阶段

```python
算法3: 推理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

输入:
  - 查询: (h, r, ?)
  - 完整模型: 所有参数

输出:
  - 候选实体排名
  - (可选) 规则置信度和不确定性信息

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 检索适用规则:
   applicable_rules = relation2rules[r]

2. 初始化:
   features = zeros([num_entities, mlp_rule_dim])
   rule_info = []  # 用于可解释性

3. For each rule in applicable_rules:

   3.1 Grounding:
       counts = graph.grounding(h, rule.body, edges_to_remove=None)

   3.2 计算规则置信度:
       features_i = concat([rule_emb[rule.id], r_body_sum])
       μ_i = μ_network(features_i)
       σ_i = exp(0.5 × logvar_network(features_i))

       w_i = μ_i  # 推理时直接用均值

   3.3 累加特征:
       for each entity e:
           if counts[e] > 0:
               features[e] += w_i × counts[e] × mlp_feature[rule.id]

   3.4 保存规则信息(可选):
       rule_info.append({
           'rule_id': rule.id,
           'confidence_mean': μ_i,
           'confidence_std': σ_i,
           'quality': 'high' if σ_i < 0.2 else 'medium' if σ_i < 0.5 else 'low'
       })

4. 聚合与打分:
   features = ReLU(LayerNorm(features))
   rule_scores = score_model(features)  # [num_entities]

5. (可选) 结合KGE分数:
   kge_scores = compute_rotate_score(h, r, all_entities)
   final_scores = rule_scores + α × kge_scores

6. 过滤已知三元组:
   final_scores[known_triplets] = -inf

7. 排序:
   ranked_entities = argsort(final_scores, descending=True)

8. 返回:
   return ranked_entities, rule_info
```

---

## 5. 参数配置指南

### 5.1 超参数表

#### 5.1.1 原始RulE参数 (保持不变)

| 参数名 | 符号 | 典型值 | 数据集差异 | 说明 |
|--------|------|--------|------------|------|
| `hidden_dim` | - | 500-2000 | UMLS:2000, FB15k:500 | 嵌入维度 |
| `gamma_fact` | γ_fact | 6.0 | 固定 | 三元组margin |
| `gamma_rule` | γ_rule | 5.0-8.0 | 可调 | 规则margin |
| `learning_rate` | - | 0.0001 | 固定 | 预训练学习率 |
| `batch_size` | - | 256 | 固定 | 三元组批大小 |
| `negative_sample_size` | N | 256-512 | 大数据集用512 | 负样本数 |
| `adversarial_temperature` | α | 0.5 | 固定 | 对抗温度 |

#### 5.1.2 不确定性新增参数

| 参数名 | 符号 | 典型值 | 调优范围 | 说明 |
|--------|------|--------|----------|------|
| **`num_samples`** | K | **5** | 3-10 | 训练时采样次数 |
| **`lambda_uncertainty`** | λ_uncertainty | **0.01** | 0.001-0.1 | 不确定性损失权重 |
| **`beta_kl`** | β_kl | **0.001** | 0.0001-0.01 | KL散度权重 |
| **`beta_sigma`** | β_σ | **0.1** | 0.01-1.0 | 方差匹配权重 |
| **`lambda_0`** | λ_0 | **1.0** | 0.5-2.0 | 支持数依赖系数 |
| `mlp_hidden_dims` | - | [256, 128] | 固定 | MLP隐藏层维度 |

### 5.2 数据集特定配置

#### 5.2.1 UMLS (小数据集)

```json
{
  "dataset": "umls",
  "num_entities": 135,
  "num_relations": 46,
  "num_rules": 18400,

  "hidden_dim": 2000,
  "gamma_fact": 6.0,
  "gamma_rule": 6.0,
  "learning_rate": 0.0001,
  "batch_size": 256,

  "num_samples": 5,
  "lambda_uncertainty": 0.01,
  "beta_kl": 0.001,
  "beta_sigma": 0.1,
  "lambda_0": 1.0,

  "mlp_rule_dim": 100,
  "g_lr": 0.0001,
  "g_batch_size": 16,
  "smoothing": 0.2
}
```

#### 5.2.2 FB15k-237 (大数据集)

```json
{
  "dataset": "fb15k237",
  "num_entities": 14541,
  "num_relations": 237,
  "num_rules": 131883,

  "hidden_dim": 500,
  "gamma_fact": 6.0,
  "gamma_rule": 6.0,
  "learning_rate": 0.00005,
  "batch_size": 256,
  "negative_sample_size": 512,

  "num_samples": 3,
  "lambda_uncertainty": 0.005,
  "beta_kl": 0.0005,
  "beta_sigma": 0.05,
  "lambda_0": 1.0,

  "mlp_rule_dim": 100,
  "g_lr": 0.0001,
  "g_batch_size": 16,
  "smoothing": 0.1
}
```

### 5.3 参数调优建议

#### 5.3.1 调优优先级

```
高优先级 (影响大):
1. num_samples      → 影响训练稳定性
2. lambda_uncertainty → 控制不确定性建模强度
3. beta_sigma       → 匹配支持数约束

中优先级 (微调):
4. beta_kl          → 防止方差塌缩
5. lambda_0         → 控制支持数敏感度

低优先级 (通常不变):
6. mlp_hidden_dims  → 网络架构
```

#### 5.3.2 诊断指南

**症状1: 训练不稳定，loss震荡**
```
可能原因: num_samples太小
解决方案: 增加num_samples (5 → 10)
```

**症状2: σ_i全部接近0**
```
可能原因: beta_kl太大或beta_sigma太大
解决方案: 降低beta_kl (0.001 → 0.0001)
```

**症状3: 所有规则σ都很大**
```
可能原因: beta_kl太小
解决方案: 增加beta_kl (0.001 → 0.01)
```

**症状4: 低支持数规则σ没有变大**
```
可能原因: beta_sigma太小或lambda_0太小
解决方案: 增加beta_sigma (0.1 → 0.5) 或 lambda_0 (1.0 → 2.0)
```

**症状5: 性能没有提升**
```
可能原因: lambda_uncertainty太小，不确定性建模被忽略
解决方案: 增加lambda_uncertainty (0.01 → 0.05)
```

---

## 6. 完整示例

### 6.1 场景设置

**知识图谱**: 医学领域UMLS

**实体**:
```
- drug_aspirin (阿司匹林)
- drug_warfarin (华法林)
- disease_heart_attack (心脏病)
- disease_stroke (中风)
- disease_headache (头痛)
- symptom_chest_pain (胸痛)
```

**关系**:
```
- treats (治疗)
- causes (引起)
- has_symptom (有症状)
```

**知识图谱三元组**:
```
(drug_aspirin, treats, disease_heart_attack)     # 1000次出现
(drug_aspirin, treats, disease_stroke)           # 800次出现
(drug_aspirin, treats, disease_headache)         # 50次出现
(disease_heart_attack, has_symptom, symptom_chest_pain)
(disease_stroke, causes, symptom_chest_pain)
```

**规则** (AMIE挖掘):
```
Rule 1: treats(X,Y) ∧ has_symptom(Y,Z) → treats_symptom(X,Z)
        支持数: 1000条路径

Rule 2: treats(X,Y) ∧ causes(Y,Z) → treats_symptom(X,Z)
        支持数: 50条路径
```

**查询**: `(drug_aspirin, treats_symptom, ?)`

---

### 6.2 预训练阶段示例

#### 6.2.1 Rule 1的不确定性建模

```python
# Step 1: 获取规则表示
rule_1 = [15, treats_symptom, treats, has_symptom]
R_1 = rule_emb[15]              # [200] 假设rule_dim=200

# Step 2: 聚合规则体
r_treats = relation_emb[treats]         # [500] 假设hidden_dim=500
r_has_symptom = relation_emb[has_symptom]
r_body_sum = r_treats + r_has_symptom   # [500]

# Step 3: 构建特征
features_1 = concat([R_1, r_body_sum])  # [700] = 200 + 500

# Step 4: 计算分布参数
μ_1 = μ_network(features_1)
    = Linear_3(ReLU(Linear_2(ReLU(Linear_1(features_1)))))
    = Linear_3(ReLU(Linear_2(ReLU([700×256]·features_1))))
    = ... (前向传播)
    = 0.850  # 标量

log_σ²_1 = logvar_network(features_1)
         = ... (类似结构)
         = -4.32

σ_1 = exp(0.5 × (-4.32)) = exp(-2.16) = 0.115

# Rule 1的分布:
w_1 ~ N(0.850, 0.115²)
```

**解释**:
- μ_1 = 0.850: 规则1的置信度均值较高
- σ_1 = 0.115: 标准差小，因为支持数1000很大
- 质量评估: **高质量规则** (μ高且σ低)

#### 6.2.2 Rule 2的不确定性建模

```python
# Rule 2
rule_2 = [42, treats_symptom, treats, causes]
R_2 = rule_emb[42]

r_treats = relation_emb[treats]
r_causes = relation_emb[causes]
r_body_sum = r_treats + r_causes

features_2 = concat([R_2, r_body_sum])  # [700]

# 计算分布参数
μ_2 = μ_network(features_2) = 0.650
log_σ²_2 = logvar_network(features_2) = -1.65
σ_2 = exp(0.5 × (-1.65)) = 0.434

# Rule 2的分布:
w_2 ~ N(0.650, 0.434²)
```

**解释**:
- μ_2 = 0.650: 置信度均值中等
- σ_2 = 0.434: 标准差大，因为支持数只有50
- 质量评估: **中低质量规则** (μ中等且σ高)

#### 6.2.3 训练时采样示例

**Rule 1 (高质量规则)**:
```python
num_samples = 5
w_samples_1 = []

for k in range(5):
    ε_k = torch.randn(1)  # 采样标准正态
    w_k = 0.850 + 0.115 × ε_k
    w_samples_1.append(w_k)

# 5次采样结果:
ε_1 = -0.52  →  w_1 = 0.850 + 0.115×(-0.52) = 0.790
ε_2 = +1.03  →  w_2 = 0.850 + 0.115×1.03    = 0.968
ε_3 = -0.18  →  w_3 = 0.850 + 0.115×(-0.18) = 0.829
ε_4 = +0.65  →  w_4 = 0.850 + 0.115×0.65    = 0.925
ε_5 = -0.91  →  w_5 = 0.850 + 0.115×(-0.91) = 0.745

# 平均:
w_1_avg = (0.790 + 0.968 + 0.829 + 0.925 + 0.745) / 5 = 0.851

# 接近μ_1 = 0.850 ✓
```

**Rule 2 (中低质量规则)**:
```python
# 5次采样结果:
ε_1 = -0.52  →  w_1 = 0.650 + 0.434×(-0.52) = 0.424
ε_2 = +1.03  →  w_2 = 0.650 + 0.434×1.03    = 1.097
ε_3 = -0.18  →  w_3 = 0.650 + 0.434×(-0.18) = 0.572
ε_4 = +0.65  →  w_4 = 0.650 + 0.434×0.65    = 0.932
ε_5 = -0.91  →  w_5 = 0.650 + 0.434×(-0.91) = 0.255

# 平均:
w_2_avg = (0.424 + 1.097 + 0.572 + 0.932 + 0.255) / 5 = 0.656

# 方差大，但平均接近μ_2 = 0.650 ✓
```

**关键对比**:
- Rule 1采样: [0.790, 0.968, 0.829, 0.925, 0.745] **方差小**
- Rule 2采样: [0.424, 1.097, 0.572, 0.932, 0.255] **方差大**
- 训练效果: Rule 2的不稳定性自然降低其影响

#### 6.2.4 不确定性损失计算

**KL散度损失**:
```python
# Rule 1
μ_1 = 0.850, σ_1 = 0.115
L_kl_1 = 0.5 × (0.850² + 0.115² - log(0.115²) - 1)
       = 0.5 × (0.7225 + 0.0132 - (-4.32) - 1)
       = 0.5 × 4.0557
       = 2.028

# Rule 2
μ_2 = 0.650, σ_2 = 0.434
L_kl_2 = 0.5 × (0.650² + 0.434² - log(0.434²) - 1)
       = 0.5 × (0.4225 + 0.1883 - (-1.65) - 1)
       = 0.5 × 1.2608
       = 0.630

# 总KL损失
L_kl = L_kl_1 + L_kl_2 = 2.028 + 0.630 = 2.658
```

**方差匹配损失**:
```python
# 目标方差计算
λ_0 = 1.0

# Rule 1: support_count = 1000
σ_target_1 = 1.0 / (1 + log(1001)) = 1.0 / 7.91 = 0.126

# Rule 2: support_count = 50
σ_target_2 = 1.0 / (1 + log(51)) = 1.0 / 4.94 = 0.202

# 方差匹配损失
L_σ_1 = (0.115 - 0.126)² = 0.000121
L_σ_2 = (0.434 - 0.202)² = 0.053824

L_σ = L_σ_1 + L_σ_2 = 0.053945
```

**分析**:
- Rule 1: σ=0.115接近σ_target=0.126 ✓ 损失小
- Rule 2: σ=0.434远大于σ_target=0.202 ✗ 需要调整

**总不确定性损失**:
```python
β_kl = 0.001
β_σ = 0.1

L_uncertainty = β_kl × L_kl + β_σ × L_σ
              = 0.001 × 2.658 + 0.1 × 0.053945
              = 0.002658 + 0.005395
              = 0.008053
```

---

### 6.3 Grounding阶段示例

#### 6.3.1 查询处理

**查询**: `(drug_aspirin, treats_symptom, ?)`

**Step 1: 检索适用规则**
```python
query_r = treats_symptom
applicable_rules = relation2rules[treats_symptom]
# 返回: [Rule 1, Rule 2]
```

**Step 2: Rule 1 Grounding**

```python
# Rule 1: treats ∧ has_symptom → treats_symptom

初始化:
h = drug_aspirin
x = [1, 0, 0, 0, 0, 0]  # [aspirin, warfarin, heart_attack, stroke, headache, chest_pain]
     ↑

第1跳 (treats):
知识图谱中的treats边:
  - (aspirin, treats, heart_attack)  ✓
  - (aspirin, treats, stroke)        ✓
  - (aspirin, treats, headache)      ✓

传播:
x = [0, 0, 1, 1, 1, 0]
         ↑  ↑  ↑
   heart_attack, stroke, headache都收到消息

第2跳 (has_symptom):
知识图谱中的has_symptom边:
  - (heart_attack, has_symptom, chest_pain)  ✓ heart_attack值为1

传播:
x = [0, 0, 0, 0, 0, 1]
                  ↑
            chest_pain收到1次消息

grounding_count_1 = [0, 0, 0, 0, 0, 1]
                                   ↑
                            chest_pain: 1条路径
```

**Step 3: Rule 2 Grounding**

```python
# Rule 2: treats ∧ causes → treats_symptom

初始化:
x = [1, 0, 0, 0, 0, 0]

第1跳 (treats):
x = [0, 0, 1, 1, 1, 0]  # 同上

第2跳 (causes):
知识图谱中的causes边:
  - (stroke, causes, chest_pain)  ✓ stroke值为1

传播:
x = [0, 0, 0, 0, 0, 1]
                  ↑
            chest_pain收到1次消息

grounding_count_2 = [0, 0, 0, 0, 0, 1]
                                   ↑
                            chest_pain: 1条路径
```

#### 6.3.2 特征聚合

```python
# 对chest_pain的特征聚合

# Rule 1贡献:
w_1 = μ_1 = 0.850  # 推理时用均值
mlp_feature_1 = [0.21, 0.35, 0.42, ..., 0.18]  # 100维

contribution_1 = 0.850 × 1 × mlp_feature_1
               = [0.179, 0.298, 0.357, ..., 0.153]

# Rule 2贡献:
w_2 = μ_2 = 0.650
mlp_feature_2 = [0.12, 0.28, 0.35, ..., 0.22]

contribution_2 = 0.650 × 1 × mlp_feature_2
               = [0.078, 0.182, 0.228, ..., 0.143]

# 总特征:
feature[chest_pain] = contribution_1 + contribution_2
                    = [0.179+0.078, 0.298+0.182, ...]
                    = [0.257, 0.480, 0.585, ..., 0.296]

# 对比: 如果只有Rule 1 (原始RulE可能过度依赖低质量规则)
# 不确定性RulE自动降低了Rule 2的权重(0.650 vs 可能的0.9+)
```

#### 6.3.3 MLP打分

```python
# 归一化和激活
feature_norm = LayerNorm([0.257, 0.480, ...])
             = [0.245, 0.468, ...]  # 归一化后

feature_relu = ReLU(feature_norm)
             = [0.245, 0.468, ...]  # 都是正数，不变

# MLP打分 (假设score_model是2层MLP)
hidden = ReLU(W1 · feature_relu + b1)
       = ReLU([128×100] · [0.245, ...] + b1)
       = [0.52, 0.31, ..., 0.68]  # 128维隐层

score[chest_pain] = W2 · hidden + b2
                  = [1×128] · [0.52, ...] + b2
                  = 0.87  # 高分! ✓
```

#### 6.3.4 其他候选实体

```python
# headache (无grounding)
grounding_count_1[headache] = 0
grounding_count_2[headache] = 0

feature[headache] = zeros([100])  # 没有规则支持

score[headache] = score_model(zeros([100])) + bias
                = 0.0 + bias
                = -5.0  # 低分

# warfarin (不同药物)
# 假设也有一些规则grounding
score[warfarin] = 0.32  # 中等分数
```

#### 6.3.5 结合KGE分数

```python
# RotatE计算 (α=3.0)
kge_score[chest_pain] = compute_rotate_score(aspirin, treats_symptom, chest_pain)
                      = 0.72

kge_score[headache] = 0.15
kge_score[warfarin] = 0.28

# 最终分数
final[chest_pain] = 0.87 + 3.0 × 0.72 = 3.03  ✅ 最高!
final[headache] = -5.0 + 3.0 × 0.15 = -4.55
final[warfarin] = 0.32 + 3.0 × 0.28 = 1.16

# 排序
ranked = [chest_pain, warfarin, headache, ...]
```

---

### 6.4 可解释性输出

```python
# 对预测结果提供解释

Prediction: (drug_aspirin, treats_symptom, symptom_chest_pain)

Evidence:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
规则支持:

1. Rule 1: treats(X,Y) ∧ has_symptom(Y,Z) → treats_symptom(X,Z)
   置信度: μ = 0.850, σ = 0.115
   质量: ★★★★★ 高质量 (σ < 0.2)
   支持路径: 1条
     - aspirin → treats → heart_attack → has_symptom → chest_pain

   贡献度: 0.850 × 1 = 0.850

2. Rule 2: treats(X,Y) ∧ causes(Y,Z) → treats_symptom(X,Z)
   置信度: μ = 0.650, σ = 0.434
   质量: ★★☆☆☆ 中低质量 (σ > 0.4)
   支持路径: 1条
     - aspirin → treats → stroke → causes → chest_pain

   贡献度: 0.650 × 1 = 0.650

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
分数分解:

规则推理分数: 0.87
KGE分数: 0.72
最终分数: 0.87 + 3.0 × 0.72 = 3.03

排名: 1 / 6 (候选实体总数)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
不确定性分析:

Rule 1:
  - 高置信度 (μ=0.850)
  - 低不确定性 (σ=0.115)
  → 可靠规则，强支持

Rule 2:
  - 中等置信度 (μ=0.650)
  - 高不确定性 (σ=0.434)
  → 数据不足，弱支持

总体置信度: 高 (主要依赖Rule 1)
```

---

## 7. 实现细节

### 7.1 网络初始化

```python
def init_networks(rule_dim, hidden_dim):
    """
    初始化不确定性网络
    """
    # 输入维度
    input_dim = rule_dim + hidden_dim

    # μ网络
    mu_network = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

    # logvar网络
    logvar_network = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

    # Kaiming初始化
    for net in [mu_network, logvar_network]:
        for layer in net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), mode='fan_in')
                nn.init.zeros_(layer.bias)

    return mu_network, logvar_network
```

### 7.2 重参数化采样

```python
def reparameterize_sample(mu, logvar, num_samples=5):
    """
    重参数化采样

    Args:
        mu: [batch_size, 1] 均值
        logvar: [batch_size, 1] 对数方差
        num_samples: 采样次数

    Returns:
        w: [batch_size, 1] 采样平均值
    """
    # 计算标准差
    std = torch.exp(0.5 * logvar)  # σ = exp(0.5 × log(σ²))

    # 多次采样
    samples = []
    for _ in range(num_samples):
        # 从标准正态分布采样
        eps = torch.randn_like(mu)

        # 重参数化: w = μ + σ × ε
        w = mu + std * eps
        samples.append(w)

    # 平均
    w_avg = torch.stack(samples, dim=0).mean(dim=0)

    return w_avg
```

### 7.3 不确定性损失

```python
def compute_uncertainty_loss(mu, logvar, support_counts,
                             beta_kl=0.001, beta_sigma=0.1, lambda_0=1.0):
    """
    计算不确定性损失

    Args:
        mu: [num_rules, 1] 均值
        logvar: [num_rules, 1] 对数方差
        support_counts: [num_rules] 支持数
        beta_kl: KL散度权重
        beta_sigma: 方差匹配权重
        lambda_0: 支持数系数

    Returns:
        loss_uncertainty: 总不确定性损失
    """
    # 1. KL散度损失
    # KL(N(μ,σ²) || N(0,1)) = 0.5 × (μ² + σ² - log(σ²) - 1)
    kl_loss = 0.5 * torch.sum(
        mu.pow(2) + logvar.exp() - logvar - 1
    )

    # 2. 计算目标方差
    target_std = lambda_0 / (1 + torch.log(support_counts.float() + 1))
    target_logvar = 2 * torch.log(target_std)  # log(σ²) = 2×log(σ)

    # 3. 方差匹配损失
    sigma_loss = torch.sum((logvar - target_logvar).pow(2))

    # 4. 总损失
    loss_uncertainty = beta_kl * kl_loss + beta_sigma * sigma_loss

    return loss_uncertainty, {
        'kl_loss': kl_loss.item(),
        'sigma_loss': sigma_loss.item()
    }
```

### 7.4 完整训练循环

```python
def train_epoch(model, train_loader, optimizer, args):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    total_kge_loss = 0
    total_rule_loss = 0
    total_uncertainty_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        # 1. KGE损失
        triplet_batch = batch['triplets']
        kge_loss = model.compute_kge_loss(triplet_batch)

        # 2. 规则损失 (带不确定性采样)
        rule_batch = batch['rules']

        # 2.1 获取规则表示
        R_i = model.rule_emb[rule_batch[:, 0]]
        r_body_sum = aggregate_rule_body(model, rule_batch[:, 2:])

        # 2.2 构建特征
        features = torch.cat([R_i, r_body_sum], dim=-1)

        # 2.3 计算分布参数
        mu = model.mu_network(features)
        logvar = model.logvar_network(features)

        # 2.4 重参数化采样
        w = reparameterize_sample(mu, logvar, num_samples=args.num_samples)

        # 2.5 计算规则损失
        rule_loss = compute_rule_loss(model, rule_batch, w)

        # 3. 不确定性损失
        uncertainty_loss, _ = compute_uncertainty_loss(
            mu, logvar,
            support_counts=model.support_counts[rule_batch[:, 0]],
            beta_kl=args.beta_kl,
            beta_sigma=args.beta_sigma,
            lambda_0=args.lambda_0
        )

        # 4. 总损失
        loss = (kge_loss +
                args.lambda_rule * rule_loss +
                args.lambda_uncertainty * uncertainty_loss)

        # 5. 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 6. 统计
        total_loss += loss.item()
        total_kge_loss += kge_loss.item()
        total_rule_loss += rule_loss.item()
        total_uncertainty_loss += uncertainty_loss.item()

        if batch_idx % args.log_interval == 0:
            print(f'Batch {batch_idx}: '
                  f'Loss={loss.item():.4f}, '
                  f'KGE={kge_loss.item():.4f}, '
                  f'Rule={rule_loss.item():.4f}, '
                  f'Unc={uncertainty_loss.item():.4f}')

    return {
        'total_loss': total_loss / len(train_loader),
        'kge_loss': total_kge_loss / len(train_loader),
        'rule_loss': total_rule_loss / len(train_loader),
        'uncertainty_loss': total_uncertainty_loss / len(train_loader)
    }
```

---

## 8. 常见问题

### 8.1 理论问题

**Q1: 为什么要用重参数化采样，不能直接采样吗？**

A: 直接采样不可微分:
```python
# 不可微分:
w = torch.normal(mu, std)  # 梯度无法传播到mu和std

# 可微分:
eps = torch.randn_like(mu)  # 固定分布，不依赖参数
w = mu + std * eps          # 梯度可以传播
```

**Q2: KL散度损失的作用是什么？**

A: 防止两个退化情况:
1. σ → 0: 方差塌缩，模型过度自信
2. μ → ∞: 均值爆炸，训练不稳定

KL散度鼓励 μ ≈ 0, σ ≈ 1 (接近先验)

**Q3: 为什么推理时不采样？**

A: 推理时需要确定性预测:
- 训练时: 采样提供正则化
- 推理时: 用均值μ保证可复现性

### 8.2 实现问题

**Q4: num_samples应该设置多少？**

A:
- 快速原型: 1-3
- 推荐值: 5
- 大模型: 10+
- 权衡: 采样越多越稳定，但计算越慢

**Q5: 如何预计算support_counts？**

```python
def compute_support_counts(rules, train_triplets, graph):
    """
    统计每个规则的支持数
    """
    support_counts = torch.zeros(len(rules))

    for i, rule in enumerate(rules):
        rule_head = rule[1]
        rule_body = rule[2:]

        # 找到所有符合规则头的三元组
        matching_triplets = [t for t in train_triplets if t[1] == rule_head]

        count = 0
        for h, r, t in matching_triplets:
            # 检查是否存在路径
            grounding = graph.grounding(h, rule_head, rule_body, None)
            if grounding[t] > 0:
                count += grounding[t].item()

        support_counts[i] = count

    return support_counts
```

**Q6: 训练不稳定怎么办？**

排查清单:
1. 检查num_samples (增加到10)
2. 降低学习率 (0.0001 → 0.00005)
3. 增加梯度裁剪 (max_norm=1.0 → 0.5)
4. 降低lambda_uncertainty (0.01 → 0.001)

### 8.3 性能问题

**Q7: 不确定性RulE比原始RulE慢多少？**

A:
- 预训练: 2-3倍 (多了2个MLP和采样)
- Grounding: 1.1倍 (只是推理时多一次前向传播)
- 推理: 几乎相同

**Q8: 如何加速训练？**

1. 减少num_samples (5 → 3)
2. 使用混合精度训练 (FP16)
3. 减小MLP层数 ([256,128] → [128])
4. 批量计算不确定性参数

---

## 附录A: 完整参数列表

### A.1 模型参数

| 参数类别 | 参数名 | 形状 | 说明 |
|----------|--------|------|------|
| **嵌入** | entity_emb | [num_entities, hidden_dim×2] | 实体嵌入 |
| | relation_emb | [num_relations, hidden_dim] | 关系嵌入 |
| | rule_emb | [num_rules, rule_dim] | 规则嵌入 |
| **不确定性** | μ_network.W1 | [256, rule_dim+hidden_dim] | 均值网络第1层 |
| | μ_network.W2 | [128, 256] | 均值网络第2层 |
| | μ_network.W3 | [1, 128] | 均值网络第3层 |
| | logvar_network.V1 | [256, rule_dim+hidden_dim] | 方差网络第1层 |
| | logvar_network.V2 | [128, 256] | 方差网络第2层 |
| | logvar_network.V3 | [1, 128] | 方差网络第3层 |
| **Grounding** | mlp_feature | [num_rules, mlp_rule_dim] | 规则MLP特征 |
| | score_model | MLP网络 | 评分网络 |

**总参数量估算** (以UMLS为例):
```
原始RulE: ~10M
不确定性RulE: ~10.2M (+200K)
增加: 2%
```

### A.2 超参数速查表

```python
# 预训练
hidden_dim = 500-2000
gamma_fact = 6.0
gamma_rule = 5.0-8.0
learning_rate = 0.0001
batch_size = 256
max_steps = 15000-30000

# 不确定性
num_samples = 5
lambda_uncertainty = 0.01
beta_kl = 0.001
beta_sigma = 0.1
lambda_0 = 1.0

# Grounding
mlp_rule_dim = 100
g_lr = 0.0001
g_batch_size = 16
num_iters = 20
smoothing = 0.2

# 推理
alpha = 3.0  # KGE权重
```

---

**文档版本**: v1.0
**创建日期**: 2025-01-12
**作者**: 不确定性RulE项目组
**适用代码版本**: RulE v1.0 + 不确定性扩展
