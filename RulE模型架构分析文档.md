# RulE模型架构分析文档

## 目录
1. [模型概述](#模型概述)
2. [整体架构图](#整体架构图)
3. [核心模块详解](#核心模块详解)
4. [数学公式](#数学公式)
5. [训练流程](#训练流程)

---

## 模型概述

RulE（Rule-Enhanced Knowledge Graph Embedding）是一个神经符号推理框架，将实体、关系和逻辑规则统一嵌入到同一个向量空间中。模型分为三个主要阶段：

1. **预训练阶段（Pre-training）**：联合学习实体、关系和规则的嵌入表示
2. **规则传播阶段（Grounding）**：在知识图谱上执行逻辑规则传播，识别可应用的规则实例
3. **推理阶段（Inference）**：结合规则推理和知识图嵌入进行预测

---

## 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          RulE 模型架构                               │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  第一阶段：预训练（Pre-training）                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  输入：三元组 (h, r, t) + 规则 (r_head, r_body)                      │
│     ↓                                 ↓                              │
│  ┌──────────────┐              ┌──────────────┐                     │
│  │ RotatE 模块   │              │  RulE 模块   │                     │
│  │  (KGE打分)   │              │  (规则打分)   │                     │
│  └──────────────┘              └──────────────┘                     │
│     ↓                                 ↓                              │
│  Entity Embedding (E)           Rule Embedding (R)                  │
│  Relation Embedding (Φ)         + 不确定性建模                      │
│     ↓                                 ↓                              │
│  Loss_fact = L_pos + L_neg      Loss_rule = L_pos + L_neg           │
│     ↓                                 ↓                              │
│  └─────────────────┬─────────────────┘                              │
│                    ↓                                                 │
│  L_pre = Loss_fact + λ_rule·Loss_rule + λ_unc·L_uncertainty        │
│                    ↓                                                 │
│          优化 → 保存最佳检查点                                        │
└──────────────────────────────────────────────────────────────────────┘

                           ↓ 加载预训练权重

┌──────────────────────────────────────────────────────────────────────┐
│  第二阶段：规则传播训练（Grounding Training）                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  冻结参数：Entity Emb, Relation Emb, Rule Emb, Uncertainty Networks │
│  可训练参数：MLP Feature, Score Model, FuncToNodeSum                │
│                                                                      │
│  【预计算阶段（可选）】                                               │
│  如果启用软传播：                                                     │
│    ┌──────────────────────────────────────────────┐                │
│    │ build_soft_pred_edges():                     │                │
│    │ - 为每个关系计算top-B KGE预测边              │                │
│    │ - 权重 = η·sigmoid(KGE_score/temp)          │                │
│    │ - 缓存到内存，供后续传播使用                  │                │
│    └──────────────────────────────────────────────┘                │
│                                                                      │
│  输入：查询 (h, r, ?)                                                │
│     ↓                                                                │
│  ┌────────────────────────────────────────────────────────┐         │
│  │  1. 规则检索：获取关系 r 对应的所有规则                │         │
│  │     relation2rules[r] → [(rule_id, (r_head, r_body))]  │         │
│  └────────────────────────────────────────────────────────┘         │
│     ↓                                                                │
│  ┌────────────────────────────────────────────────────────┐         │
│  │  2. 图传播（Grounding）：多跳路径遍历                  │         │
│  │                                                         │         │
│  │     【硬传播】graph.propagate() - 仅使用观测边          │         │
│  │          或                                              │         │
│  │     【软传播】soft_grounding_count():                   │         │
│  │       逐跳执行：                                          │         │
│  │       ├─ 观测边传播：x_real = A_r·x                     │         │
│  │       ├─ 死胡同检测：dead = (x_real.sum()==0)          │         │
│  │       ├─ 预测边传播：x_pred = A^soft_r·x （如需要）     │         │
│  │       ├─ 混合：x = x_real + mask·x_pred                │         │
│  │       └─ 束搜索：保留top-B候选（可选）                   │         │
│  │                                                         │         │
│  │     从头实体 h 出发，按规则体 r_body 传播                │         │
│  │     返回 count tensor：每个候选实体的路径数量            │         │
│  └────────────────────────────────────────────────────────┘         │
│     ↓                                                                │
│  ┌────────────────────────────────────────────────────────┐         │
│  │  3. 规则聚合（Rule Aggregation）                       │         │
│  │                                                         │         │
│  │  输入：                                                 │         │
│  │    - rule_count [num_rules, num_candidates]            │         │
│  │    - rule_emb, mlp_feature (带置信度 μ 调制)           │         │
│  │    - (h, r) 查询嵌入                                   │         │
│  │                                                         │         │
│  │  ┌─────────────────────────────────────────────┐      │         │
│  │  │ 【可选增强1】Query-Conditioned Attention     │      │         │
│  │  │ (如果启用 use_query_attention=True)          │      │         │
│  │  ├─────────────────────────────────────────────┤      │         │
│  │  │ 1. 编码查询：                                │      │         │
│  │  │    query_vec = MLP([E(h) || Φ(r)])          │      │         │
│  │  │                                              │      │         │
│  │  │ 2. 计算每条规则的attention权重：             │      │         │
│  │  │    attn_i = sigmoid(MLP([query_vec||R_i]))  │      │         │
│  │  │                                              │      │         │
│  │  │ 3. 调制grounding count：                     │      │         │
│  │  │    count'_i = attn_i × count_i              │      │         │
│  │  │                                              │      │         │
│  │  │ 目的：不同查询关注不同规则                   │      │         │
│  │  └─────────────────────────────────────────────┘      │         │
│  │                       ↓                                │         │
│  │  ┌─────────────────────────────────────────────┐      │         │
│  │  │ 【可选增强2】Hierarchical Aggregation        │      │         │
│  │  │ (如果启用 use_hierarchical_agg=True)         │      │         │
│  │  ├─────────────────────────────────────────────┤      │         │
│  │  │ 1. 根据规则置信度 μ 分组：                   │      │         │
│  │  │    q_i = normalize(μ_i) ∈ [0,1]             │      │         │
│  │  │    G_low    = {i | q_i < 0.4}               │      │         │
│  │  │    G_medium = {i | 0.4 ≤ q_i < 0.7}         │      │         │
│  │  │    G_high   = {i | q_i ≥ 0.7}               │      │         │
│  │  │                                              │      │         │
│  │  │ 2. 每组独立聚合：                            │      │         │
│  │  │    feature_low  = FuncToNodeSum(G_low)      │      │         │
│  │  │    feature_med  = FuncToNodeSum(G_medium)   │      │         │
│  │  │    feature_high = FuncToNodeSum(G_high)     │      │         │
│  │  │                                              │      │         │
│  │  │ 3. 计算层间权重（基于查询）：                │      │         │
│  │  │    w = softmax(MLP_gate([E(h) || Φ(r)]))    │      │         │
│  │  │                                              │      │         │
│  │  │ 4. 加权求和：                                │      │         │
│  │  │    feature = Σ_g w_g × feature_g            │      │         │
│  │  │                                              │      │         │
│  │  │ 目的：高/低质量规则差异化处理                │      │         │
│  │  └─────────────────────────────────────────────┘      │         │
│  │                       ↓                                │         │
│  │  ┌─────────────────────────────────────────────┐      │         │
│  │  │ 【基础聚合】FuncToNodeSum (总是执行)         │      │         │
│  │  ├─────────────────────────────────────────────┤      │         │
│  │  │ 1. 加权求和：                                │      │         │
│  │  │    feature = Σ count_i × mlp_feature_i      │      │         │
│  │  │ 2. Layer Norm + ReLU                        │      │         │
│  │  │ 3. 平均：feature.mean(规则维度)              │      │         │
│  │  └─────────────────────────────────────────────┘      │         │
│  │                                                         │         │
│  │  输出：每个候选实体的聚合特征向量                        │         │
│  └────────────────────────────────────────────────────────┘         │
│     ↓                                                                │
│  ┌────────────────────────────────────────────────────────┐         │
│  │  4. 评分（Scoring）                                    │         │
│  │     MLP Score Model → 候选实体得分                     │         │
│  └────────────────────────────────────────────────────────┘         │
│     ↓                                                                │
│  Loss = CrossEntropy(scores, target) + Label Smoothing              │
│     ↓                                                                │
│  优化 → 保存最佳模型                                                 │
└──────────────────────────────────────────────────────────────────────┘

                           ↓ 加载最佳模型

┌──────────────────────────────────────────────────────────────────────┐
│  第三阶段：推理（Inference）                                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  输入：查询 (h, r, ?)                                                │
│     ↓                                                                │
│  ┌──────────────────┐              ┌──────────────────┐            │
│  │  规则推理得分     │              │   KGE 得分       │            │
│  │  (Grounding)     │              │   (RotatE)       │            │
│  └──────────────────┘              └──────────────────┘            │
│     ↓                                      ↓                        │
│     └──────────────┬───────────────────────┘                        │
│                    ↓                                                 │
│  最终得分 = Score_rule + α · Score_KGE （可选融合）                 │
│                    ↓                                                 │
│          排序 → 返回Top-K候选实体                                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 核心模块详解

RulE模型包含9个核心模块：
1. 嵌入层（Embedding Layers）
2. RotatE模块（知识图嵌入）
3. RulE模块（规则嵌入）
4. 不确定性建模（Uncertainty Modeling）
5. 图传播模块（Grounding）
6. 规则聚合模块（Rule Aggregation）
7. 评分模块（Scoring）
8. 软传播模块（Soft Grounding）
9. 数据结构模块

### 模块关系图

```
                    ┌─────────────────────┐
                    │   数据结构模块 (9)   │
                    │  - KnowledgeGraph   │
                    │  - 规则表示          │
                    └──────────┬──────────┘
                               │ 提供数据
           ┌───────────────────┴───────────────────┐
           ↓                                       ↓
    ┌────────────┐                          ┌────────────┐
    │ 嵌入层 (1)  │                          │ 图传播 (5) │
    │- Entity Emb│◄────┐                    │(硬传播)    │
    │- Relation  │     │                    └─────┬──────┘
    │- Rule Emb  │     │ 使用                     │
    └─────┬──────┘     │                          │
          │            │                    ┌─────┴──────┐
          ├────────────┼───────────────────►│ 软传播 (8) │
          │            │                    │- KGE预测边 │
          ↓            │                    │- 死胡同检测 │
    ┌────────────┐     │                    │- 束搜索    │
    │ RotatE (2) │     │                    └─────┬──────┘
    │(KGE打分)   │     │                          │ grounding count
    └─────┬──────┘     │                          ↓
          │            │                    ┌──────────────────────┐
          │ KGE得分    │                    │   规则聚合 (6)        │
          │            │                    ├──────────────────────┤
          ↓            │              ┌────►│【可选】Query Attn    │
    ┌────────────┐     │              │     │ - 查询编码           │
    │  RulE (3)  │─────┘              │     │ - 计算attention      │
    │(规则打分)   │                    │     │ - 调制count          │
    └─────┬──────┘                    │     ├──────────────────────┤
          │                            │     │【可选】Hierarchical  │
          ↓                            │     │ - 规则质量分组       │
    ┌────────────┐                    │     │ - 分组聚合           │
    │不确定性(4) │────────────────────┤     │ - Gate网络           │
    │- μ Network │ 规则置信度 μ       │     │ - 加权融合           │
    │- σ Network │                    │     ├──────────────────────┤
    │- 重参数化  │                    └────►│【基础】FuncToNodeSum │
    └────────────┘                          │ - 加权求和           │
                                            │ - LayerNorm + ReLU   │
                                            └──────────┬───────────┘
                                                       │ 聚合特征
                                                       ↓
                                                 ┌────────────┐
                                                 │ 评分模块(7)│
                                                 │  MLP打分   │
                                                 └─────┬──────┘
                                                       │
                                                       ↓
                                                  最终得分


预训练阶段：使用模块 1-4
Grounding阶段：使用模块 1, 5/8, 6, 7, 9 (6中可启用Query Attn和Hierarchical)
推理阶段：使用模块 1-2, 5/8, 6, 7, 9
```

---

### 1. 嵌入层（Embedding Layers）

#### 1.1 实体嵌入（Entity Embedding）
- **功能**：将每个实体映射到高维向量空间
- **维度**：`num_entities × (hidden_dim × 2)`
- **说明**：使用复数表示（实部+虚部各占 hidden_dim），用于RotatE旋转操作
- **初始化**：均匀分布 `[-embedding_range_fact, +embedding_range_fact]`

#### 1.2 关系嵌入（Relation Embedding）
- **功能**：将每个关系映射到向量空间，表示实体间的语义关系
- **维度**：`num_relations × hidden_dim`
- **说明**：
  - 关系嵌入表示旋转角度（相位）
  - 支持逆关系：关系 r 和逆关系 r_inv 通过符号区分
  - 存在padding索引用于规则体的填充
- **初始化**：均匀分布 `[-embedding_range_fact, +embedding_range_fact]`

#### 1.3 规则嵌入（Rule Embedding）
- **功能**：为每条逻辑规则学习独立的向量表示
- **维度**：`num_rules × hidden_dim`
- **说明**：
  - 规则表示为 `rule_head ← rule_body`，例如 `r3 ← r1 ∧ r2`
  - 规则嵌入作为偏移量，补充规则体关系的语义
- **初始化**：Kaiming均匀初始化

---

### 2. RotatE模块（知识图嵌入）

#### 2.1 功能
计算三元组 (h, r, t) 的合理性得分，基于旋转假设：`t ≈ h ∘ r`

#### 2.2 核心思想
- 在复数空间中，关系表示旋转操作
- 头实体通过关系旋转后应接近尾实体

#### 2.3 处理流程
1. 将实体嵌入分解为实部和虚部
2. 将关系嵌入转换为复数旋转（通过相位表示）
3. 执行复数乘法（旋转操作）
4. 计算旋转后的头实体与尾实体的距离
5. 使用gamma作为margin，距离越小得分越高

---

### 3. RulE模块（规则嵌入）

#### 3.1 功能
计算逻辑规则的合理性得分，衡量规则头与规则体的语义一致性

#### 3.2 核心思想
规则头的关系嵌入应该接近规则体关系嵌入之和加上规则特定的偏移量

#### 3.3 处理流程
1. 获取规则体中所有关系的嵌入
2. 对规则体关系嵌入求和（考虑逆关系的符号）
3. 加上规则特定的嵌入向量（偏移量）
4. 计算与规则头关系嵌入的距离
5. 使用gamma_rule作为margin进行打分

---

### 4. 不确定性建模（Uncertainty Modeling）

#### 4.1 功能
为每条规则学习置信度分布，量化规则的可靠性和不确定性

#### 4.2 核心组件
- **μ Network**：预测规则置信度的均值
- **logσ² Network**：预测规则置信度的方差（log尺度）
- **Support Counts**：规则的支持度统计（先验知识）

#### 4.3 处理流程
1. 构造规则特征：连接规则嵌入和规则体关系嵌入之和
2. 通过MLP网络预测 μ（均值）和 logσ²（对数方差）
3. 重参数化采样：从 N(μ, σ²) 中采样权重 w
4. 使用 σ(w) 作为规则置信度权重，调制规则得分

#### 4.4 损失函数组成
- **KL散度损失**：约束分布接近标准正态分布
- **方差匹配损失**：根据support counts调整方差大小
  - 支持度高的规则应有较小方差（高置信度）
  - 支持度低的规则应有较大方差（低置信度）

---

### 5. 图传播模块（Grounding）

#### 5.1 功能
在知识图谱上执行规则体的多跳传播，识别满足规则的候选实体

#### 5.2 核心思想
从查询头实体出发，按照规则体指定的关系序列进行图遍历，统计到达每个实体的路径数量

#### 5.3 处理流程
1. 初始化：从头实体 h 开始（one-hot向量）
2. 对规则体中的每个关系：
   - 通过邻接矩阵进行消息传递
   - 累积到达每个实体的路径计数
3. 移除训练边：防止模型记忆训练集
4. 返回 count tensor：每个候选实体的grounding路径数量

#### 5.4 可选：软传播（Soft Grounding）
- **动机**：处理知识图谱不完整性
- **方法**：
  - 使用KGE预测的边补充观测边
  - 在死胡同（dead-end）或稀疏路径时启用
  - 预测边权重 = η · sigmoid(KGE_score / temp)
- **参数**：
  - `topB`：每个关系保留的top-K预测边
  - `η`：预测边的权重系数
  - `temp`：温度参数，控制置信度平滑
  - `beam`：束搜索大小，限制每步保留的候选实体数

---

### 6. 规则聚合模块（Rule Aggregation）

#### 6.1 FuncToNodeSum（基础聚合）
- **功能**：将多条规则的贡献聚合到候选实体
- **输入**：
  - `rule_count`：grounding计数矩阵 `[num_rules, num_candidates]`
  - `rule_emb`：规则嵌入权重
  - `mlp_feature`：规则的MLP特征向量
- **处理流程**：
  1. 将grounding count作为注意力权重
  2. 对规则特征进行加权求和：`count × mlp_feature`
  3. Layer Normalization
  4. ReLU激活
  5. 对规则维度取平均，得到候选实体特征

#### 6.2 Query-Conditioned Attention（可选增强）
- **功能**：根据查询 (h, r) 动态调整规则重要性
- **核心思想**：不同查询应关注不同规则

##### 完整处理流程
```
输入：
  - h_emb: [batch, hidden_dim*2]  头实体嵌入
  - r_emb: [batch, hidden_dim]    查询关系嵌入
  - rule_embs: [num_rules, hidden_dim]  规则嵌入
  - rule_count: [num_rules, num_candidates]  grounding计数

流程：
  1. 编码查询向量
     query = concat([h_emb, r_emb])  # [batch, hidden_dim*3]
     query_vec = MLP_query_encoder(query)  # [batch, attention_hidden_dim]

  2. 扩展维度以匹配规则数量
     query_exp = query_vec.unsqueeze(1)  # [batch, 1, attention_hidden_dim]
     query_exp = query_exp.expand(batch, num_rules, -)  # [batch, num_rules, attention_hidden_dim]

     rule_exp = rule_embs.unsqueeze(0)  # [1, num_rules, hidden_dim]
     rule_exp = rule_exp.expand(batch, -, -)  # [batch, num_rules, hidden_dim]

  3. 计算注意力权重（每条规则独立打分）
     attn_input = concat([query_exp, rule_exp], dim=-1)
                # [batch, num_rules, attention_hidden_dim + hidden_dim]

     attention_scores = sigmoid(MLP_attention_net(attn_input))
                      # [batch, num_rules, 1]  ∈ (0, 1)

  4. 归一化（稳定训练）
     attention_scores = attention_scores / (attention_scores.mean(dim=1) + 1e-9)

  5. 调制grounding count
     对每个候选实体c，找到其对应的查询索引q
     adjusted_count[i, c] = attention_scores[q, i] × rule_count[i, c]

输出：
  - adjusted_count: 经过query-specific调制的grounding count
```

##### 关键设计
- **非竞争性打分**：使用sigmoid而非softmax，规则之间不互斥
- **Query-specific**：真正做到针对不同查询动态选择规则
- **训练稳定性**：通过归一化避免attention过大/过小

#### 6.3 Hierarchical Rule Aggregation（可选增强）
- **功能**：根据规则质量分层聚合
- **核心思想**：高质量规则和低质量规则应分开处理，避免低质量规则污染结果

##### 完整处理流程
```
输入：
  - rule_mu: [num_rules, 1]  预计算的规则置信度
  - rule_count: [num_rules, num_candidates]
  - mlp_feature: [num_rules, mlp_rule_dim]
  - h_emb, r_emb: 查询嵌入
  - quality_thresholds: 质量分组阈值，如 [0.4, 0.7]

流程：
  1. 规则质量归一化
     mu_min = rule_mu.min()
     mu_max = rule_mu.max()
     q_i = (rule_mu[i] - mu_min) / (mu_max - mu_min + 1e-9)  # ∈ [0, 1]

  2. 规则分组（示例：3组）
     G_low    = {i | q_i < 0.4}        # 低质量规则
     G_medium = {i | 0.4 ≤ q_i < 0.7}  # 中质量规则
     G_high   = {i | q_i ≥ 0.7}        # 高质量规则

     使用 bucketize 操作高效分组

  3. 每组独立聚合
     for g in [G_low, G_medium, G_high]:
       if len(g) == 0:
         continue

       # 提取该组的规则数据
       g_rule_count = rule_count[g]      # [|g|, num_candidates]
       g_rule_emb = rule_emb[g]          # [|g|, hidden_dim]
       g_mlp_feature = mlp_feature[g]    # [|g|, mlp_rule_dim]

       # 聚合（使用FuncToNodeSum）
       feature_g = FuncToNodeSum(g_rule_count, g_rule_emb, g_mlp_feature)
                 # [num_candidates, mlp_rule_dim]

       # 保存到对应组
       group_features[g] = feature_g

  4. 计算层间gate权重（基于查询）
     query_vec = concat([h_emb, r_emb])  # [batch, hidden_dim*3]
     group_logits = MLP_hierarchical_gate(query_vec)  # [batch, num_groups]
     group_weights = softmax(group_logits, dim=-1)    # [batch, num_groups]

     # 对候选实体维度扩展
     group_weights_for_candidates = group_weights[candidate_query_idx]
                                  # [num_candidates, num_groups]

  5. 加权融合
     final_feature = zeros([num_candidates, mlp_rule_dim])

     for g in range(num_groups):
       final_feature += group_weights_for_candidates[:, g].unsqueeze(-1)
                      × group_features[g]

输出：
  - final_feature: [num_candidates, mlp_rule_dim]
```

##### 关键设计
- **质量感知**：根据规则置信度 μ 自动分组，无需人工标注
- **分层处理**：每组规则独立聚合，避免相互干扰
- **Query-aware Gate**：不同查询对高/低质量规则的依赖程度不同
- **动态调整**：通过softmax保证权重和为1，自适应平衡各层贡献

##### 三种聚合模式对比
```
┌─────────────────────┬───────────────┬────────────────┬──────────────────┐
│     聚合模式         │ 是否分组       │ 是否Query感知  │   适用场景        │
├─────────────────────┼───────────────┼────────────────┼──────────────────┤
│ 基础 FuncToNodeSum  │ 否            │ 否             │ 规则质量相近      │
│ Query Attention     │ 否            │ 是             │ 查询差异大        │
│ Hierarchical Agg    │ 是(按质量)    │ 是             │ 规则质量差异大    │
└─────────────────────┴───────────────┴────────────────┴──────────────────┘
```

##### 可以同时启用两种增强吗？
**可以！** 模块6的完整执行顺序：
```
1. Query Attention调制 count (如果启用)
   → count' = attention × count

2. Hierarchical分组聚合 (如果启用)
   → 使用调制后的 count' 进行分组聚合

3. 基础 FuncToNodeSum (如果不启用Hierarchical)
   → 直接聚合 count' 或 count
```

---

### 7. 评分模块（Scoring）

#### 7.1 MLP Score Model
- **功能**：将聚合后的规则特征映射到标量得分
- **架构**：
  - 大数据集（FB15k-237, WN18RR, YAGO）：`MLP(100, [128, 1])` - 两层
  - 小数据集（UMLS, Kinship, Family）：`MLP(100, [1])` - 单层
- **输出**：每个候选实体的规则推理得分

#### 7.2 融合策略（可选）
在推理阶段，可以将规则得分与KGE得分融合：
- **公式**：`Final_Score = Score_rule + α × Score_KGE`
- **参数**：α 控制KGE的贡献权重（典型值：2.0-5.0）

---

### 8. 软传播模块（Soft Grounding Module）

软传播模块是RulE模型处理知识图谱不完整性的创新机制，通过KGE预测的虚拟边补充观测边，提升规则传播的鲁棒性。

#### 8.1 模块动机
在知识图谱中：
- 观测边往往不完整（存在缺失边）
- 规则传播可能遇到"死胡同"（dead-end）：无法继续传播
- 稀疏连接导致候选实体覆盖不足

软传播通过引入KGE预测的"软边"来解决这些问题。

#### 8.2 核心组件

##### 8.2.1 预测边预计算（`build_soft_pred_edges`）
- **功能**：为每个关系预计算top-K最可能的边
- **时机**：在grounding训练开始前或推理前执行（一次性预计算）
- **过程**：
  1. 对每个关系 r ∈ [0, 2×num_relations)：
     - 计算所有可能的 (h, r, t) 三元组的KGE得分
     - 对每个头实体 h，选择得分最高的top-B个尾实体
     - 计算软边权重：`weight = η · sigmoid(KGE_score / temp)`
     - 存储为稀疏边列表：`(node_out, node_in, weight)`
  2. 缓存结果到 `_soft_pred_edges`，避免重复计算
- **适用场景**：小规模图谱（如Kinship、UMLS），实体数量不超过数千

##### 8.2.2 预测边传播（`_pred_propagate`）
- **功能**：沿着预计算的软边执行消息传递
- **输入**：
  - `x`：当前状态向量 `[num_entities, batch, 1]`
  - `relation`：要传播的关系ID
  - `edges_to_remove`：需要屏蔽的观测边索引
  - `query_h`：查询头实体（用于屏蔽训练边）
- **处理流程**：
  1. 获取关系 r 的预计算软边 `(node_out, node_in, weight)`
  2. 构造消息：`message = x[node_in] × weight`
  3. 如果指定了 `edges_to_remove`，屏蔽相应的软边
  4. 使用 `scatter` 操作聚合消息到目标节点
  5. 返回传播结果 `[num_entities, batch, 1]`

##### 8.2.3 束搜索剪枝（`_apply_beam`）
- **功能**：限制每步传播中保留的候选实体数量
- **目的**：控制计算复杂度，避免候选爆炸
- **处理流程**：
  1. 对每个查询，选择状态向量中值最大的top-B个实体
  2. 将其他实体的状态值置零
  3. 返回剪枝后的状态向量
- **参数**：`beam_size`（典型值：50-200，0表示不剪枝）

##### 8.2.4 软传播主流程（`soft_grounding_count`）
- **功能**：执行混合传播策略（观测边+预测边）
- **输入**：
  - `all_h`：查询头实体
  - `query_r`：查询关系
  - `rule_body`：规则体关系序列
  - `edges_to_remove`：训练边索引
  - `stats`：统计信息收集器（可选）
- **处理流程**：

  ```
  对规则体的每个关系 r_k：
    1. 观测边传播：
       x_real = graph.propagate(x, r_k, edges_to_remove)

    2. 死胡同检测：
       dead = (x_real.sum() == 0)  # 无法继续传播的查询

    3. 稀疏路径检测（可选）：
       sparse = (x_real中非零元素数量 < kmin)

    4. 决定是否使用预测边：
       - 模式1（默认）：仅在死胡同时使用
         use_pred = dead
       - 模式2：在稀疏路径时使用
         use_pred = sparse
       - 模式3：总是使用
         use_pred = True

    5. 预测边传播（如果需要）：
       if use_pred:
         x_pred = _pred_propagate(x, r_k, edges_to_remove)
         x = x_real + x_pred * use_pred_mask
       else:
         x = x_real

    6. 束搜索剪枝（可选）：
       x = _apply_beam(x, beam_size)

  返回最终grounding计数
  ```

#### 8.3 关键参数

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `soft_topb` | 0 | 每个关系保留的top-K预测边（0=关闭） |
| `soft_eta` | 0.3 | 预测边权重系数（0-1之间） |
| `soft_temp` | 1.0 | 温度参数，控制置信度平滑 |
| `soft_beam` | 0 | 束搜索大小（0=不剪枝） |
| `soft_only_on_deadend` | True | 仅在死胡同使用预测边 |
| `soft_real_kmin` | 0 | 最小路径数阈值（触发稀疏检测） |
| `soft_log_steps` | 0 | 统计日志频率（0=不记录） |

#### 8.4 统计监控
当 `soft_log_steps > 0` 时，系统会定期记录：
- **deadend_rate**：死胡同查询比例
- **sparse_rate**：稀疏路径查询比例
- **pred_used_rate**：实际使用预测边的查询比例
- **pred/real_mass**：预测边贡献 vs 观测边贡献的质量比

#### 8.5 使用场景
- **适合**：
  - 小规模知识图谱（实体数 < 10K）
  - 稀疏图谱（连接度低）
  - 规则体较长（容易遇到死胡同）
- **不适合**：
  - 大规模图谱（预计算开销大）
  - 密集图谱（观测边已足够）

#### 8.6 与硬传播的对比
| 特性 | 硬传播 | 软传播 |
|------|--------|--------|
| 使用边 | 仅观测边 | 观测边+预测边 |
| 完整性处理 | 无 | 补充缺失边 |
| 计算开销 | 低 | 中（预计算）|
| 鲁棒性 | 低 | 高 |
| 适用图谱 | 任意 | 小规模图谱 |

---

### 9. 数据结构模块

#### 9.1 KnowledgeGraph
- **功能**：知识图谱的核心数据结构
- **关键属性**：
  - `relation2adjacency`：邻接列表，用于图传播
  - `hr2o`：训练集的 (h,r)→t 映射
  - `hr2oo`：训练+验证集的映射
  - `hr2ooo`：训练+验证+测试集的映射（用于过滤评估）
- **关键方法**：
  - `grounding()`：执行多跳规则传播（硬传播）
  - `propagate()`：单跳消息传递

#### 9.2 规则表示
- **格式**：`[rule_id, rule_head, relation_1, relation_2, ..., relation_n]`
- **示例**：规则 `r1 ∧ r2 → r3` 表示为 `[id, r3, r1, r2]`
- **逆关系处理**：
  - 正向关系 r 的ID为 r
  - 逆关系的ID为 r + relation_size

---

## 数学公式

### 1. 预训练阶段公式

#### 1.1 RotatE得分函数
对于三元组 (h, r, t)：

```
h, t ∈ C^d  （复数空间）
r ∈ R^d   （实数空间，表示旋转相位）

相位转换：
r_complex = exp(i·θ_r) = cos(θ_r) + i·sin(θ_r)
其中 θ_r = r / (embedding_range / π)

RotatE 距离：
d(h, r, t) = ||h ∘ r_complex - t||

RotatE 得分：
Score_KGE(h, r, t) = γ_fact - ||h ∘ r_complex - t||_2
```

其中：
- `∘` 表示复数乘法（Hadamard积）
- `|| · ||_2` 表示L2范数
- `γ_fact` 是可学习的margin参数

#### 1.2 RulE得分函数
对于规则 `rule_head ← r_1 ∧ r_2 ∧ ... ∧ r_n`：

```
规则体嵌入求和：
r_body_sum = Σ_{i=1}^n Φ(r_i) · flag_i

其中 flag_i = {+1, 正向关系
              {-1, 逆关系

RulE 距离：
d(rule) = ||r_body_sum + R_rule - Φ(rule_head)||_p

RulE 得分（带不确定性）：
w_rule ~ N(μ_rule, σ²_rule)
w_rule_sig = sigmoid(w_rule)
Score_RulE(rule) = w_rule_sig · (γ_rule - d(rule))
```

其中：
- `R_rule` 是规则特定的嵌入向量
- `Φ(r)` 是关系 r 的嵌入
- `|| · ||_p` 是Lp范数（通常p=2）
- `γ_rule` 是规则的margin参数
- `w_rule_sig` 是规则置信度权重

#### 1.3 不确定性建模
对于每条规则：

```
规则特征：
features_i = [R_i || r_body_sum_i]  （连接操作）

均值和方差网络：
μ_i = MLP_μ(features_i)
log(σ²_i) = MLP_logvar(features_i)

重参数化采样：
ε ~ N(0, 1)
w_i = μ_i + σ_i · ε

KL散度损失：
L_KL = 0.5 · Σ_i (μ²_i + σ²_i - log(σ²_i) - 1)

目标方差（基于支持度）：
σ²_target,i = (λ_0 / (1 + log(support_count_i + 1)))²

方差匹配损失：
L_sigma = Σ_i (log(σ²_i) - log(σ²_target,i))²

不确定性总损失：
L_uncertainty = β_KL · L_KL + β_sigma · L_sigma
```

#### 1.4 预训练损失函数
```
负采样损失（对抗采样）：
p(h'|h,r,t) ∝ exp(α · Score_KGE(h', r, t))  （头实体负采样）
p(t'|h,r,t) ∝ exp(α · Score_KGE(h, r, t'))  （尾实体负采样）

三元组损失：
L_fact = - (1/N) Σ [
    log σ(Score_KGE(h, r, t))
    + E_{h'~p} log σ(-Score_KGE(h', r, t))
    + E_{t'~p} log σ(-Score_KGE(h, r, t'))
]

规则损失：
L_rule = - (1/M) Σ [
    log σ(Score_RulE(rule))
    + E_{rule'~p} log σ(-Score_RulE(rule'))
]

总损失：
L_pretrain = L_fact + λ_rule · L_rule + λ_unc · L_uncertainty + λ_reg · L_reg

正则化项：
L_reg = (1/N) Σ (||h||² + ||t||²)
```

其中：
- `σ(·)` 是sigmoid函数
- `α` 是对抗采样温度
- `λ_rule`, `λ_unc`, `λ_reg` 是损失权重超参数

---

### 2. 规则传播阶段公式

#### 2.1 图传播（Grounding）
给定查询 (h, r, ?) 和规则 `r ← r_1 ∧ r_2 ∧ ... ∧ r_n`：

```
初始化：
x^(0) = one_hot(h) ∈ R^|E|

逐跳传播（硬传播）：
x^(k) = A_{r_k} · x^(k-1)

其中 A_{r_k} 是关系 r_k 的邻接矩阵：
A_{r_k}[i,j] = 1  如果存在边 (j, r_k, i)
             = 0  否则

最终grounding计数：
count(e) = x^(n)[e]  （到达实体 e 的路径数量）
```

#### 2.2 软传播（Soft Grounding）
软传播是处理知识图谱不完整性的关键机制。

##### 预测边权重计算
```
对每个关系 r 和头实体 h：
  1. 计算KGE得分：
     scores = Score_KGE(h, r, t)  对所有 t ∈ E

  2. 选择top-B候选：
     top_B_tails = topk(scores, B)

  3. 计算软权重：
     w(h, r, t) = η · sigmoid(Score_KGE(h, r, t) / temp)

     其中：
     - η ∈ (0, 1)：预测边权重系数
     - temp > 0：温度参数，控制置信度分布的锐度

  4. 构造软邻接矩阵：
     A^soft_{r}[t, h] = w(h, r, t)  对 t ∈ top_B_tails
                      = 0           其他
```

##### 混合传播策略
```
对规则体的每一跳 k：
  1. 观测边传播（硬传播）：
     x^(k)_real = A_{r_k} · x^(k-1)

  2. 死胡同检测：
     dead(e) = 1{Σ_i x^(k)_real[i,e] = 0}  # 无路径到达

  3. 稀疏路径检测（可选）：
     sparse(e) = 1{|{i : x^(k)_real[i,e] > 0}| < k_min}  # 路径数不足

  4. 预测边传播（软传播）：
     x^(k)_pred = A^soft_{r_k} · x^(k-1)

  5. 选择性混合：
     模式A（仅死胡同）：
       use_pred(e) = dead(e)

     模式B（稀疏路径）：
       use_pred(e) = sparse(e)

     模式C（总是使用）：
       use_pred(e) = 1

     最终结果：
       x^(k)[e] = x^(k)_real[e] + use_pred(e) · x^(k)_pred[e]

  6. 束搜索剪枝（可选）：
     对每个查询 e，保留 x^(k)[·, e] 中最大的 B_beam 个实体：
       x^(k)_pruned[i, e] = x^(k)[i, e]  如果 i ∈ top_B_beam(x^(k)[·, e])
                          = 0            其他
```

##### 训练边屏蔽
在训练阶段，需要屏蔽训练边防止信息泄露：
```
对于查询 (h, r, t_true)：
  1. 在观测边传播中移除 (h, r, t_true)
  2. 在预测边传播中也要移除 (h, r, t_true)

  message[t, h] = 0  如果 t = t_true 且头实体 = h
```

##### 性能统计
```
deadend_rate = (Σ dead查询数) / (总传播步数)
sparse_rate = (Σ sparse查询数) / (总传播步数)
pred_used_rate = (Σ 实际使用预测边的步数) / (总传播步数)
mass_ratio = (Σ 预测边贡献的路径质量) / (Σ 观测边贡献的路径质量)
```

#### 2.3 规则聚合
对于候选实体集合 C：

```
基础聚合（FuncToNodeSum）：
对于候选实体 e ∈ C：

加权特征：
feature_e = Σ_{i∈R} count_i(e) · MLP_feature_i

其中：
- R 是查询关系 r 对应的规则集合
- count_i(e) 是规则 i 到达实体 e 的路径数
- MLP_feature_i 是规则 i 的MLP特征向量

归一化和激活：
feature_e = ReLU(LayerNorm(feature_e))
```

#### 2.4 Query Attention增强（可选）
```
查询编码：
query_vec = MLP_query([E(h) || Φ(r)])

注意力计算：
attention_i = sigmoid(MLP_attn([query_vec || R_i]))

调整后的grounding count：
count'_i(e) = attention_i · count_i(e)
```

#### 2.5 Hierarchical Aggregation增强（可选）
```
规则质量分数（基于预计算的 μ）：
q_i = (μ_i - min(μ)) / (max(μ) - min(μ))  ∈ [0, 1]

规则分组：
G_0 = {i | q_i < threshold_1}       （低质量）
G_1 = {i | threshold_1 ≤ q_i < threshold_2}  （中质量）
G_2 = {i | q_i ≥ threshold_2}       （高质量）

层内聚合（对每组独立）：
feature^(g)_e = FuncToNodeSum({count_i(e) | i ∈ G_g})

层间gate（基于查询）：
w = softmax(MLP_gate([E(h) || Φ(r)]))  ∈ R^G

最终特征：
feature_e = Σ_{g=0}^{G-1} w_g · feature^(g)_e
```

#### 2.6 评分和损失
```
规则推理得分：
Score_rule(h, r, e) = MLP_score(feature_e) + bias_e

标签平滑：
target = (1 - ε) · one_hot(t) + ε · uniform(|C|)

交叉熵损失：
L_ground = - Σ_{e∈C} target(e) · log softmax(Score_rule(h, r, e))

注意：只在有grounding的query上计算损失（mask.sum() > 0）
```

---

### 3. 推理阶段公式

#### 3.1 规则推理得分
```
Score_rule(h, r, e) = 如上述2.6节计算
```

#### 3.2 KGE得分
```
Score_KGE(h, r, e) = γ_fact - ||E(h) ∘ Φ_complex(r) - E(e)||_2
```

#### 3.3 融合得分（可选）
```
Score_final(h, r, e) = Score_rule(h, r, e) + α · Score_KGE(h, r, e)
```

其中 α 是融合权重超参数（典型值：2.0-5.0）

#### 3.4 排序和评估
```
候选实体排序：
rank(e) 基于 Score_final(h, r, e) 降序排列

评估指标：
MRR = (1/|Q|) Σ_{q∈Q} 1/rank(t_q)
Hits@k = (1/|Q|) Σ_{q∈Q} 𝟙[rank(t_q) ≤ k]
MR = (1/|Q|) Σ_{q∈Q} rank(t_q)

过滤设置：
在排序时移除所有真实三元组（训练+验证+测试集）
```

---

## 训练流程

### 阶段一：预训练（Pre-training）

#### 输入数据
- 知识图谱三元组：`{(h, r, t)}`
- 逻辑规则：`{(rule_head, rule_body)}`

#### 可训练参数
- 实体嵌入 `E`
- 关系嵌入 `Φ`
- 规则嵌入 `R`
- 不确定性网络 `MLP_μ`, `MLP_logvar`

#### 训练步骤
1. 初始化所有嵌入和网络参数
2. 对每个训练批次：
   - 采样正负三元组样本
   - 采样正负规则样本
   - 计算不确定性权重 w ~ N(μ, σ²)
   - 计算RotatE得分（带对抗负采样）
   - 计算RulE得分（带权重调制和对抗负采样）
   - 计算不确定性损失
   - 反向传播，更新参数
3. 定期在验证集评估（仅使用KGE得分）
4. 保存验证集MRR最高的检查点

#### 学习率调度
- 初始学习率：`learning_rate`
- 在 `warm_up_steps`（通常是max_steps的一半）后：
  - 学习率 ÷ 10
  - 如果设置了 `disable_adv`，关闭对抗采样（温度设为0）

#### 输出
- 预训练检查点：`save_path/checkpoint`
- 嵌入文件：`entity_embedding.npy`, `relation_embedding.npy`, `rule_embedding.npy`

---

### 阶段二：规则传播训练（Grounding Training）

#### 输入
- 加载预训练的实体、关系、规则嵌入
- 知识图谱和规则结构

#### 冻结参数
- 实体嵌入 `E` ✗
- 关系嵌入 `Φ` ✗
- 规则嵌入 `R` ✗
- 不确定性网络 `MLP_μ`, `MLP_logvar` ✗

#### 可训练参数
- MLP规则特征 `mlp_feature` ✓
- 评分模型 `score_model` ✓
- 规则聚合模块 `FuncToNodeSum` ✓
- 可选：`query_attention`, `hierarchical_gate` ✓
- 实体偏置 `bias` ✓

#### 预计算
1. 计算规则权重嵌入（用于快速前向传播）
2. 预计算规则置信度 μ（用于grounding阶段加权）
3. 如果启用软传播，预计算KGE预测边

#### 训练步骤
1. 按关系分组训练数据（同一批次内的query关系相同）
2. 对每个训练批次：
   - 执行规则传播（grounding）
   - 移除训练边，防止信息泄露
   - 聚合规则特征（应用可选增强）
   - MLP打分
   - 计算交叉熵损失（带标签平滑）
   - 反向传播，更新可训练参数
3. 每个epoch结束后，在验证集评估（仅使用规则得分）
4. 保存验证集MRR最高的模型

#### 标签平滑
```
smoothing = 0.2-0.5
target = smoothing * uniform_distribution + (1 - smoothing) * one_hot(true_entity)
```

#### 输出
- Grounding检查点：`save_path/grounding.pt`
- Grounding规则特征：`g_rule_embedding.npy`

---

### 阶段三：评估（Evaluation）

#### 评估模式
1. **规则推理模式**（`evaluate`）：
   - 仅使用规则传播得分
   - 评估规则推理能力

2. **融合模式**（`evaluate_t`）：
   - 规则得分 + α × KGE得分
   - 评估综合推理性能

#### 评估指标
- **MRR（Mean Reciprocal Rank）**：平均倒数排名
- **Hits@k**：正确答案在Top-k中的比例（k=1,3,10）
- **MR（Mean Rank）**：平均排名

#### 过滤设置（Filtered Setting）
- 排序时排除所有已知的真实三元组
- 使用 `hr2ooo` 映射（包含训练、验证、测试集）

#### 候选集统计
- **零候选率**：没有任何规则grounding的查询比例
- **预言机覆盖率**：真实答案在候选集中的比例
- **候选集大小**：平均、P50、P90、P99

---

## 模型特点总结

### 优势
1. **神经符号融合**：结合KGE的泛化能力和逻辑规则的可解释性
2. **联合学习**：实体、关系、规则在统一空间中共同训练
3. **不确定性建模**：量化规则可靠性，自适应调整规则贡献
4. **可扩展性**：支持多种增强策略（Query Attention, Hierarchical Aggregation, Soft Grounding）
5. **图不完整性处理**：软传播机制补充缺失边，提升规则覆盖率
6. **鲁棒性**：通过软传播和束搜索，在稀疏图谱上保持推理能力
7. **灵活性**：硬传播和软传播可根据数据集特性选择使用

### 关键创新
1. **规则嵌入**：为逻辑规则学习向量表示
2. **图传播**：通过多跳遍历实例化规则
3. **置信度加权**：基于统计支持度的概率建模
4. **动态规则选择**：Query-specific attention机制
5. **分层聚合**：根据规则质量差异化处理
6. **软传播机制**：
   - KGE预测边作为虚拟边补充观测边
   - 死胡同检测和自适应启用预测边
   - 束搜索剪枝控制计算复杂度
   - 训练边屏蔽防止信息泄露

---

## 超参数说明

### 预训练超参数
| 参数名 | 典型值 | 说明 |
|--------|--------|------|
| hidden_dim | 500-2000 | 嵌入维度（大数据集用小值） |
| gamma_fact | 6.0 | KGE margin |
| gamma_rule | 5.0-8.0 | 规则 margin |
| learning_rate | 0.00005-0.0001 | 学习率 |
| max_steps | 15000-30000 | 训练步数 |
| weight_rule | 1.0 | 规则损失权重 |
| adversarial_temperature | 0.25-0.5 | 对抗采样温度（0为关闭） |
| batch_size | 256 | 三元组批大小 |
| negative_sample_size | 256-512 | 负样本数 |
| rule_batch_size | 128-256 | 规则批大小 |
| rule_negative_size | 64-128 | 规则负样本数 |
| lambda_uncertainty | 0.001 | 不确定性损失权重 |
| beta_kl | 0.001 | KL散度损失系数 |
| beta_sigma | 0.01 | 方差匹配损失系数 |
| lambda_0 | 1.0 | 基础方差系数 |
| num_samples | 5 | 重参数化采样次数 |

### Grounding超参数
| 参数名 | 典型值 | 说明 |
|--------|--------|------|
| mlp_rule_dim | 100 | MLP规则特征维度 |
| alpha | 2.0-5.0 | KGE融合权重 |
| smoothing | 0.2-0.5 | 标签平滑系数 |
| g_lr | 0.0001 | Grounding学习率 |
| g_batch_size | 16 | Grounding批大小 |
| num_iters | 20 | 训练epoch数 |
| batch_per_epoch | 1000000 | 每epoch最大批数 |
| print_every | 10-1000 | 日志打印频率 |

### 可选增强超参数
| 参数名 | 典型值 | 说明 |
|--------|--------|------|
| use_query_attention | false | 是否启用Query Attention |
| attention_hidden_dim | 64 | Attention隐藏维度 |
| use_hierarchical_agg | false | 是否启用分层聚合 |
| quality_thresholds | [0.4, 0.7] | 质量分组阈值 |
| use_soft_grounding | false | 是否启用软传播 |
| soft_topb | 50 | 保留top-K预测边 |
| soft_eta | 0.3 | 预测边权重系数 |
| soft_temp | 1.0 | 软传播温度 |
| soft_beam | 0 | 束搜索大小（0为关闭） |
| soft_only_on_deadend | true | 仅在死胡同使用预测边 |

---

## 文件结构

```
src/
├── model.py           # RulE模型定义
│   ├── RulE类         # 主模型
│   ├── QueryConditionedAttention  # Query Attention模块
│   └── 辅助方法       # RotatE, RulE打分, 图传播
├── trainer.py         # 训练器
│   ├── PreTrainer    # 预训练阶段
│   └── GroundTrainer # Grounding阶段
├── layers.py          # 神经网络层
│   ├── MLP           # 多层感知机
│   └── FuncToNodeSum # 规则聚合模块
├── data.py            # 数据加载
│   ├── KnowledgeGraph      # 知识图谱结构
│   ├── RuleDataset         # 规则数据集
│   ├── KGETrainDataset     # KGE训练集
│   └── TrainDataset/ValidDataset/TestDataset
└── main.py            # 主入口
```

---

## 引用
如需了解更多细节，请参考：
- 论文原文：RulE: Neural-Symbolic Knowledge Graph Reasoning
- 代码仓库：RulE-master
- 详细文档：CLAUDE.md

---

**文档版本**：v1.0
**更新日期**：2025-12-23
**作者**：根据代码分析自动生成
