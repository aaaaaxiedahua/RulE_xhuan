# RulE模型详细技术分析

## 目录
1. [模型概述](#模型概述)
2. [核心组件与数学公式](#核心组件与数学公式)
3. [训练流程详解](#训练流程详解)
4. [推理与评估](#推理与评估)
5. [实现细节](#实现细节)

---

## 模型概述

RulE (Rule Embedding) 是一个神经符号知识图谱推理框架，将实体、关系和逻辑规则统一嵌入到同一个向量空间中。模型通过三个阶段完成知识图谱补全任务：

- **阶段一：预训练 (Pre-training)** - 联合学习实体嵌入、关系嵌入和规则嵌入
- **阶段二：规则落地 (Grounding)** - 在知识图谱上实例化抽象规则
- **阶段三：推理 (Inference)** - 结合规则和嵌入进行预测

---

## 核心组件与数学公式

### 1. 实体和关系嵌入 (RotatE)

#### 1.1 嵌入表示

RulE使用RotatE作为知识图谱嵌入基础：

- **实体嵌入**: $\mathbf{e}_h, \mathbf{e}_t \in \mathbb{C}^{d}$ （复数空间，维度为 $d$）
- **关系嵌入**: $\mathbf{r} \in \mathbb{R}^{d}$，表示为相位角

在实现中，复数嵌入通过实部和虚部分开存储：
- 实体嵌入维度：`num_entities × (hidden_dim * 2)`
- 关系嵌入维度：`num_relations × hidden_dim`

```python
# 代码位置: src/model.py:62-80
self.entity_embedding = torch.nn.Embedding(self.num_entities, self.hidden_dim * 2)
self.relation_embedding = torch.nn.Embedding(self.num_relations + 1, self.hidden_dim)
```

#### 1.2 RotatE评分函数

给定三元组 $(h, r, t)$，RotatE将关系建模为复数空间中的旋转：

$$
\mathbf{e}_h \circ \mathbf{r} \approx \mathbf{e}_t
$$

其中 $\circ$ 表示复数的Hadamard乘积（逐元素乘法）。

**距离函数**：
$$
d_r(h,t) = \|\mathbf{e}_h \circ \mathbf{r} - \mathbf{e}_t\|
$$

**评分函数**：
$$
f_r(h,t) = \gamma_{fact} - \|\mathbf{e}_h \circ \mathbf{r} - \mathbf{e}_t\|_2
$$

其中 $\gamma_{fact}$ 是固定的margin参数（通常为6）。

#### 1.3 复数旋转的实现

关系被转换为相位角：
$$
\theta_r = \frac{\mathbf{r}}{\gamma_{fact}/\pi} \cdot \pi
$$

复数旋转通过欧拉公式实现：
$$
\mathbf{r}_{complex} = \cos(\theta_r) + i \cdot \sin(\theta_r)
$$

**实部和虚部的计算**：
$$
\begin{aligned}
\text{re}_{score} &= \text{re}_h \cdot \cos(\theta_r) - \text{im}_h \cdot \sin(\theta_r) - \text{re}_t \\
\text{im}_{score} &= \text{re}_h \cdot \sin(\theta_r) + \text{im}_h \cdot \cos(\theta_r) - \text{im}_t
\end{aligned}
$$

**最终分数**：
$$
score = \gamma_{fact} - \sqrt{\text{re}_{score}^2 + \text{im}_{score}^2}
$$

```python
# 代码位置: src/model.py:236-266
def RotatE(self, head, relation, tail, mode='tail-batch'):
    re_head, im_head = torch.chunk(head, 2, dim=2)
    re_tail, im_tail = torch.chunk(tail, 2, dim=2)

    # 转换为相位角
    phase_relation = relation/(self.embedding_range_fact.item()/self.pi)
    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)

    # 复数乘法
    re_score = re_head * re_relation - im_head * im_relation - re_tail
    im_score = re_head * im_relation + im_head * re_relation - im_tail

    # 计算欧氏距离
    score = torch.stack([re_score, im_score], dim=0)
    score = score.norm(dim=0)
    score = self.gamma_fact.item() - score.sum(dim=2)

    return score
```

---

### 2. 规则嵌入 (Rule Embedding)

#### 2.1 规则表示

逻辑规则的形式：
$$
r_{b_1} \land r_{b_2} \land \cdots \land r_{b_n} \rightarrow r_h
$$

其中：
- $r_h$ 是规则头（head relation）
- $r_{b_1}, \ldots, r_{b_n}$ 是规则体（body relations）

规则在模型中的表示为：
$$
\text{Rule} = [\text{rule\_id}, r_h, r_{b_1}, r_{b_2}, \ldots, r_{b_n}]
$$

#### 2.2 规则嵌入空间

每个规则有两个嵌入：
- **规则嵌入** $\mathbf{e}_{rule} \in \mathbb{R}^d$：规则的抽象表示
- **MLP特征** $\mathbf{f}_{rule} \in \mathbb{R}^{d_{mlp}}$：用于grounding阶段的可学习特征

```python
# 代码位置: src/model.py:147-152
self.mlp_feature = nn.Parameter(torch.zeros(self.num_rules, self.mlp_rule_dim))
self.rule_emb = torch.nn.Embedding(self.num_rules, self.rule_dim)
```

#### 2.3 规则嵌入的评分函数

规则的嵌入通过聚合规则体的关系嵌入得到：

**规则体聚合**：
$$
\mathbf{v}_{body} = \sum_{i=1}^{n} \text{sign}(r_{b_i}) \cdot \mathbf{e}_{r_{b_i}}
$$

其中：
$$
\text{sign}(r) = \begin{cases}
+1 & \text{if } r < N_{rel} \text{ (正向关系)} \\
-1 & \text{if } r \geq N_{rel} \text{ (逆向关系)}
\end{cases}
$$

**规则完整表示**：
$$
\mathbf{v}_{rule} = \mathbf{v}_{body} + \mathbf{e}_{rule}
$$

**规则评分函数**：
$$
f_{rule} = \gamma_{rule} - \|\mathbf{v}_{rule} - \text{sign}(r_h) \cdot \mathbf{e}_{r_h}\|_p
$$

其中：
- $\gamma_{rule}$ 是规则的margin参数（通常为5-8）
- $p$ 是范数类型（通常$p=2$）

```python
# 代码位置: src/model.py:270-301
def add_ruleE(self, rules, mask):
    # 提取规则体
    inputs = rules[:,:,2:]

    # 处理正向/逆向关系
    relations_flag = torch.pow(-1, inputs // self.num_relations).unsqueeze(-1)
    inputs_com = inputs % self.num_relations

    # 获取关系嵌入并应用符号
    embedding = self.relation_embedding(inputs_com) * relations_flag

    # 规则嵌入
    rule_embedding = self.rule_emb(rules[:,:,0])

    # 规则头嵌入
    embedding_r = self.relation_embedding(rules[:,:,1] % self.num_relations)
    relations_flag = torch.pow(-1, rules[:,:,1] // self.num_relations).unsqueeze(-1)
    embedding_r *= relations_flag

    # 聚合规则体
    rule_body = embedding * mask.unsqueeze(1).unsqueeze(-1)
    outputs = rule_body.sum(-2) + rule_embedding

    # 计算距离
    dist = self.gamma_rule.item() - torch.norm(outputs - embedding_r, p=self.p, dim=-1)

    return dist, rule_embedding
```

---

### 3. 规则落地 (Rule Grounding)

#### 3.1 图传播算法

给定查询 $(h, r, ?)$，规则落地通过多跳传播找到满足规则体的实体：

对于规则 $r_{b_1} \land r_{b_2} \rightarrow r_h$：

**初始化**：
$$
\mathbf{x}^{(0)} = \text{one\_hot}(h) \in \{0,1\}^{|E|}
$$

**第一跳传播** (沿着 $r_{b_1}$)：
$$
\mathbf{x}^{(1)}_t = \sum_{h': (h', r_{b_1}, t) \in KG} \mathbf{x}^{(0)}_{h'}
$$

**第二跳传播** (沿着 $r_{b_2}$)：
$$
\mathbf{x}^{(2)}_t = \sum_{h': (h', r_{b_2}, t) \in KG} \mathbf{x}^{(1)}_{h'}
$$

**落地计数**：
$$
\text{count}(t, rule, h) = \mathbf{x}^{(n)}_t
$$

表示有多少条从 $h$ 出发、遵循规则体到达 $t$ 的路径。

```python
# 代码位置: src/data.py:410-421
def grounding(self, h, r, rule, edges_to_remove):
    # 初始化为one-hot向量
    x = torch.nn.functional.one_hot(h, self.entity_size).transpose(0, 1).unsqueeze(-1)

    # 逐跳传播
    for r_body in rule:
        if r_body == r:
            x = self.propagate(x, r_body, edges_to_remove)
        else:
            x = self.propagate(x, r_body, None)

    return x.squeeze(-1).transpose(0, 1)
```

#### 3.2 传播操作

使用消息传递神经网络 (MPNN) 框架：

$$
\mathbf{x}^{(l+1)}_v = \text{AGGREGATE}\left(\{\mathbf{x}^{(l)}_u : (u, r, v) \in KG\}\right)
$$

使用`torch_scatter`实现高效聚合：

```python
# 代码位置: src/data.py:423-447
def propagate(self, x, relation, edges_to_remove=None):
    node_in = self.relation2adjacency[relation][0][1]  # 头实体
    node_out = self.relation2adjacency[relation][0][0] # 尾实体

    # 收集消息
    message = x[node_in]

    if edges_to_remove is None:
        # 聚合到目标节点
        x = scatter(message, node_out, dim=0, dim_size=x.size(0))
    else:
        # 移除指定边
        message = message.view(-1, D)
        message[edges_to_remove] = 0
        message = message.view(E, B, D)
        x = scatter(message, node_out, dim=0, dim_size=x.size(0))

    return x
```

#### 3.3 规则聚合与评分

对于查询 $(h, r, ?)$，所有头为 $r$ 的规则参与评分：

**步骤1：收集所有相关规则的落地计数**
$$
C \in \mathbb{R}^{|R_r| \times |E|}
$$
其中 $R_r = \{\text{rule} : \text{rule.head} = r\}$，$C_{ij}$ 表示规则 $i$ 对实体 $j$ 的落地计数。

**步骤2：提取候选实体集合**
$$
\mathcal{C} = \{e : \exists \text{rule}, C_{\text{rule}, e} > 0\}
$$

**步骤3：规则特征加权聚合**

使用FuncToNodeSum层聚合规则特征：

$$
\mathbf{h}_e = \text{LayerNorm}\left(\text{ReLU}\left(\sum_{\text{rule} \in R_r} C_{\text{rule}, e} \cdot \mathbf{W} \cdot \mathbf{f}_{rule}\right)\right)
$$

然后取平均：
$$
\mathbf{h}_e = \frac{1}{|R_r|} \sum_{\text{rule} \in R_r} \mathbf{h}_{e, \text{rule}}
$$

```python
# 代码位置: src/layers.py:62-73
def forward(self, A_fn, x_f, mlp_rule_feature):
    # A_fn: 落地计数矩阵 [num_candidates, num_rules]
    # mlp_rule_feature: 规则特征 [num_rules, mlp_rule_dim]

    weight = torch.transpose(A_fn, 0, 1).unsqueeze(-1)
    message = x_f.unsqueeze(0)

    feature = torch.transpose((message * weight), 1, 2)
    weighted_features = torch.matmul(feature, mlp_rule_feature)
    weighted_features_norm = self.layer_norm(weighted_features)
    weighted_features_relu = torch.relu(weighted_features_norm)
    output = weighted_features_relu.mean(1)

    return output
```

**步骤4：MLP评分**
$$
s(e | h, r) = \text{MLP}(\mathbf{h}_e) + b_e
$$

其中 $b_e$ 是实体偏置项。

```python
# 代码位置: src/model.py:337-409
def forward(self, all_h, all_r, edges_to_remove):
    query_r = all_r[0].item()

    # 对所有相关规则进行落地
    rule_index = list()
    rule_count = list()
    mask = torch.zeros(all_h.size(0), self.graph.entity_size, device=device)

    for index, (r_head, r_body) in self.relation2rules[query_r]:
        count = self.graph.grounding(all_h, r_head, r_body, edges_to_remove).float()
        mask += count
        rule_index.append(index)
        rule_count.append(count)

    # 提取候选集
    candidate_set = torch.nonzero(mask.view(-1), as_tuple=True)[0]

    # 聚合规则特征
    output = self.rule_to_entity(rule_count, rule_emb, mlp_feature)

    # MLP评分
    output = self.score_model(feature).squeeze(-1)

    # 添加偏置
    score = score + self.bias.unsqueeze(0)

    return score, mask
```

---

## 训练流程详解

### 阶段一：预训练 (Pre-training)

#### 目标

联合学习三类嵌入：
1. 实体嵌入 $\mathbf{E} = \{\mathbf{e}_1, \ldots, \mathbf{e}_{|E|}\}$
2. 关系嵌入 $\mathbf{R} = \{\mathbf{r}_1, \ldots, \mathbf{r}_{|R|}\}$
3. 规则嵌入 $\mathbf{Rule} = \{\mathbf{e}_{rule_1}, \ldots, \mathbf{e}_{rule_{|Rules|}}\}$

#### 损失函数

**总损失**：
$$
\mathcal{L} = \mathcal{L}_{fact} + \lambda_{rule} \cdot \mathcal{L}_{rule} + \lambda_{reg} \cdot \mathcal{L}_{reg}
$$

其中：
- $\lambda_{rule}$：规则损失权重（config中的`weight_rule`，通常为1）
- $\lambda_{reg}$：正则化系数（config中的`regularization`，通常为0）

#### 1. 三元组损失 $\mathcal{L}_{fact}$

使用负采样的对数sigmoid损失：

**正样本损失**：
$$
\mathcal{L}_{fact}^{+} = -\frac{1}{B} \sum_{(h,r,t) \in \mathcal{B}} w_{(h,r)} \cdot \log \sigma(f_r(h,t))
$$

**负样本损失（带自对抗采样）**：
$$
\mathcal{L}_{fact}^{-} = -\frac{1}{B} \sum_{(h,r,t) \in \mathcal{B}} w_{(h,r)} \sum_{t' \in \mathcal{N}(h,r,t)} p(t'|h,r,t) \cdot \log \sigma(-f_r(h,t'))
$$

其中自对抗采样的概率分布：
$$
p(t'|h,r,t) = \frac{\exp(\alpha \cdot f_r(h,t'))}{\sum_{t'' \in \mathcal{N}(h,r,t)} \exp(\alpha \cdot f_r(h,t''))}
$$

- $\alpha$：对抗温度（config中的`adversarial_temperature`，通常为0.5）
- $w_{(h,r)}$：子采样权重（类似word2vec）

**组合**：
$$
\mathcal{L}_{fact} = \frac{\mathcal{L}_{fact}^{+} + \mathcal{L}_{fact}^{-}}{2}
$$

```python
# 代码位置: src/trainer.py:153-188
# 负样本
negative_fact_score = (F.softmax(negative_fact_score * args.adversarial_temperature, dim=1).detach()
                    * F.logsigmoid(-negative_fact_score)).sum(dim=1)

# 正样本
positive_fact_score = F.logsigmoid(positive_fact_score).squeeze(dim=1)

# 加权平均
positive_fact_loss = -(subsampling_weight * positive_fact_score).sum() / subsampling_weight.sum()
negative_fact_loss = -(subsampling_weight * negative_fact_score).sum() / subsampling_weight.sum()

loss_fact = (positive_fact_loss + negative_fact_loss) / 2
```

#### 2. 规则损失 $\mathcal{L}_{rule}$

与三元组损失类似，但针对规则：

**正样本损失**：
$$
\mathcal{L}_{rule}^{+} = -\frac{1}{B_{rule}} \sum_{\text{rule} \in \mathcal{B}_{rule}} \log \sigma(f_{rule})
$$

**负样本损失**：
规则的负采样通过随机替换规则体中的某个关系实现。

$$
\mathcal{L}_{rule}^{-} = -\frac{1}{B_{rule}} \sum_{\text{rule} \in \mathcal{B}_{rule}} \sum_{\text{rule}' \in \mathcal{N}(\text{rule})} p(\text{rule}'|\text{rule}) \cdot \log \sigma(-f_{rule'})
$$

**组合**：
$$
\mathcal{L}_{rule} = \frac{\mathcal{L}_{rule}^{+} + \mathcal{L}_{rule}^{-}}{2}
$$

```python
# 代码位置: src/trainer.py:159-185
# 负样本
negative_rule_score = (F.softmax(negative_rule_score * args.adversarial_temperature, dim=1).detach()
                    * F.logsigmoid(-negative_rule_score)).sum(dim=1)

# 正样本
positive_rule_score = F.logsigmoid(positive_rule_score)

positive_rule_loss = -positive_rule_score.mean() * args.weight_rule
negative_rule_loss = -negative_rule_score.mean() * args.weight_rule

loss_rule = (positive_rule_loss + negative_rule_loss) / 2
```

#### 3. 正则化损失 $\mathcal{L}_{reg}$

L2正则化防止过拟合：
$$
\mathcal{L}_{reg} = \frac{1}{B} \sum_{(h,r,t) \in \mathcal{B}} \left(\|\mathbf{e}_h\|_2^2 + \|\mathbf{e}_t\|_2^2\right)
$$

#### 训练策略

1. **学习率调度**：
   - 初始学习率：`learning_rate`（0.00005-0.0001）
   - Warm-up步数后学习率除以10

2. **对抗温度调度**：
   - Warm-up阶段使用较高温度（如0.5）
   - 后期可以降低或关闭（`disable_adv=True`）

3. **验证与保存**：
   - 每`valid_steps`步在验证集上评估
   - 保存最佳MRR的检查点

```python
# 代码位置: src/trainer.py:88-123
for step in range(0, args.max_steps + 1):
    log = self.train_step(optimizer, self.triplets_iterator, self.rules_iterator, args)

    if step >= warm_up_steps:
        current_learning_rate = current_learning_rate / 10
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=current_learning_rate
        )
        warm_up_steps = warm_up_steps * 3
        if args.disable_adv:
            args.adversarial_temperature = 0

    if step % args.valid_steps == 0:
        mrr = self.evaluate("valid", self.expectation)
        if mrr > best_mrr:
            save_model(self.model, optimizer, args)
            best_mrr = mrr
```

---

### 阶段二：规则落地训练 (Grounding Training)

#### 目标

固定预训练的嵌入，只训练MLP参数以学习如何聚合规则落地结果。

#### 参数冻结

```python
# 代码位置: src/trainer.py:390-393
self.model.entity_embedding.weight.requires_grad = False
self.model.relation_embedding.weight.requires_grad = False
self.model.rule_emb.weight.requires_grad = False
```

#### 损失函数

使用交叉熵损失与标签平滑：

**标签平滑**：
$$
\tilde{y}_e = (1 - \epsilon) \cdot y_e + \epsilon \cdot \mathbb{1}(e = t)
$$

其中：
- $y_e \in \{0, 1\}$：原始标签（所有真实尾实体为1）
- $\epsilon$：平滑系数（config中的`smoothing`，通常为0.2-0.5）
- $\mathbb{1}(e = t)$：当前查询的真实尾实体

**交叉熵损失**：
$$
\mathcal{L}_{ground} = -\frac{\sum_{e \in \mathcal{C}} \tilde{y}_e \cdot \log p(e|h,r)}{\sum_{e \in \mathcal{C}} \tilde{y}_e}
$$

其中：
$$
p(e|h,r) = \frac{\exp(s(e|h,r))}{\sum_{e' \in \mathcal{C}} \exp(s(e'|h,r))}
$$

```python
# 代码位置: src/trainer.py:486-503
# 标签平滑
target_t = torch.nn.functional.one_hot(all_t, self.train_set.graph.entity_size)
target = target * smoothing + target_t * (1 - smoothing)

# 前向传播
grounding_rule_score, mask = model(all_h, all_r, edges_to_remove)

if mask.sum().item() != 0:
    # 计算log概率
    rule_logits = (torch.softmax(grounding_rule_score, dim=1) + 1e-8).log()

    # 交叉熵损失
    loss = -(rule_logits[mask] * target[mask]).sum() / torch.clamp(target[mask].sum(), min=1)
    loss.backward()
    optimizer.step()
```

#### 训练策略

1. **规则权重预计算**：
   在grounding训练前，一次性计算所有规则的嵌入评分

   ```python
   # 代码位置: src/model.py:414-431
   def eval_compute_rule_weight(self, device):
       for rules, rules_mask in zip(rule_batches, rule_mask_batches):
           rule_weight_emb = self.add_ruleE_g(rules.unsqueeze(1), rules_mask).squeeze(1)
           rules_weight_emb.append(rule_weight_emb)

       self.rules_weight_emb = torch.cat(rules_weight_emb)
   ```

2. **边移除策略**：
   训练时移除查询三元组 $(h,r,t)$ 以防止trivial解

3. **迭代训练**：
   通常训练20个epoch（`num_iters=20`）

---

## 推理与评估

### 评估模式

#### 1. 纯规则推理

仅使用规则落地的分数：
$$
\text{score}(t|h,r) = s(t|h,r)
$$

```python
# 代码位置: src/trainer.py:522-630
def evaluate(self, split, alpha=3.0, expectation=True):
    logits, mask = model(all_h, all_r, None)
    # 不使用KGE分数
```

#### 2. 混合推理

结合KGE和规则分数：
$$
\text{score}(t|h,r) = s_{rule}(t|h,r) + \alpha \cdot s_{KGE}(t|h,r)
$$

其中 $\alpha$ 是可调节权重（config中的`alpha`，通常为2-5）。

```python
# 代码位置: src/trainer.py:634-742
def evaluate_t(self, split, alpha=3.0, expectation=True):
    logits, mask = model(all_h, all_r, None)
    kge_score = model.compute_g_KGE(all_h, all_r)
    logits = logits + alpha * kge_score
```

### 评估指标

#### 排名计算

对于查询 $(h,r,t)$，计算所有候选实体的分数并排序。

**Filtered设置**：排除所有已知的真实三元组
$$
\text{Rank}(t|h,r) = 1 + |\{e : \text{score}(e|h,r) > \text{score}(t|h,r) \land (h,r,e) \notin KG\}|
$$

#### 期望排名（Expectation Ranking）

当存在tie（相同分数）时，使用期望排名：

对于分数为 $s$ 的三元组，其排名区间为 $[L, H)$：
- $L$：严格大于 $s$ 的实体数量 + 1
- $H$：大于等于 $s$ 的实体数量 + 2

**期望排名**：
$$
\text{ExpRank} = \frac{L + (H-1)}{2} = \frac{L + H - 1}{2}
$$

在计算指标时，对所有可能排名求期望：
$$
\text{MRR} = \frac{1}{H - L} \sum_{r=L}^{H-1} \frac{1}{r}
$$

```python
# 代码位置: src/trainer.py:273-310
for h, r, t, L, H in ranks.data.cpu().numpy().tolist():
    query2LH[(h, r, t)] = (L, H)

for (L, H) in query2LH.values():
    if expectation:
        for rank in range(L, H):
            if rank <= 1:
                hit1 += 1.0 / (H - L)
            if rank <= 3:
                hit3 += 1.0 / (H - L)
            if rank <= 10:
                hit10 += 1.0 / (H - L)
            mr += rank / (H - L)
            mrr += 1.0 / rank / (H - L)
```

#### 评估指标定义

1. **MRR (Mean Reciprocal Rank)**：
   $$
   \text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{Rank}(q)}
   $$

2. **Hits@k**：排名在前k的比例
   $$
   \text{Hits@k} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{1}(\text{Rank}(q) \leq k)
   $$

3. **MR (Mean Rank)**：
   $$
   \text{MR} = \frac{1}{|Q|} \sum_{q \in Q} \text{Rank}(q)
   $$

---

## 实现细节

### 1. 关系的双向表示

为了处理逆关系，关系数量翻倍：
- 正向关系：$r \in [0, N_{rel})$
- 逆向关系：$r + N_{rel} \in [N_{rel}, 2 \cdot N_{rel})$

**符号函数**：
$$
\text{sign}(r) = (-1)^{\lfloor r / N_{rel} \rfloor}
$$

实现：
```python
relations_flag = torch.pow(-1, inputs // self.num_relations)
inputs_com = inputs % self.num_relations
embedding = self.relation_embedding(inputs_com) * relations_flag.unsqueeze(-1)
```

### 2. 嵌入初始化

#### 实体和关系嵌入

使用均匀分布初始化：
$$
\mathbf{e}, \mathbf{r} \sim \mathcal{U}(-\epsilon_{fact}, \epsilon_{fact})
$$

其中：
$$
\epsilon_{fact} = \frac{\gamma_{fact} + 2.0}{d}
$$

```python
# 代码位置: src/model.py:52-69
self.embedding_range_fact = nn.Parameter(
    torch.Tensor([(self.gamma_fact.item() + self.epsilon) / hidden_dim]),
    requires_grad=False
)

nn.init.uniform_(
    tensor=self.entity_embedding.weight,
    a=-self.embedding_range_fact.item(),
    b=self.embedding_range_fact.item()
)
```

#### 规则嵌入和MLP特征

使用Kaiming初始化：
```python
# 代码位置: src/model.py:147-152
nn.init.kaiming_uniform_(self.mlp_feature, a=math.sqrt(5), mode="fan_in")
nn.init.kaiming_uniform_(self.rule_emb.weight, a=math.sqrt(5), mode="fan_in")
```

### 3. 批处理策略

#### 预训练阶段

使用双向迭代器交替采样head-batch和tail-batch：
```python
# 代码位置: src/trainer.py:81-82
self.triplets_iterator = BidirectionalOneShotIterator(
    triplets_dataloader_head, triplets_dataloader_tail
)
```

#### Grounding阶段

按关系分组批次，确保同一batch中的查询使用相同关系：
```python
# 代码位置: src/data.py:460-470
for r, instances in enumerate(self.r2instances):
    for k in range(0, len(instances), self.batch_size):
        self.batches.append(instances[start:end])
random.shuffle(self.batches)
```

### 4. 高效的图传播

使用`torch_scatter`库实现高效的稀疏矩阵操作：

```python
from torch_scatter import scatter

# 聚合操作: 将消息聚合到目标节点
x = scatter(message, node_out, dim=0, dim_size=x.size(0))
```

等价于密集矩阵操作：
$$
\mathbf{X}^{(l+1)} = \mathbf{A}_r \mathbf{X}^{(l)}
$$

但在稀疏图上更加高效。

### 5. 数值稳定性技巧

#### Log-Softmax技巧

在grounding训练中：
```python
rule_logits = (torch.softmax(grounding_rule_score, dim=1) + 1e-8).log()
```

添加小常数避免`log(0)`。

#### Padding处理

规则长度不同，使用padding和mask：
```python
# 代码位置: src/data.py:62-68
positive_sample = pad_sequence(
    [torch.LongTensor(_[0][0]) for _ in data],
    batch_first=True,
    padding_value=data[0][0][-1]
)
rule_mask = pad_sequence([_[4] for _ in data], batch_first=True, padding_value=False)
```

---

## 完整训练流程总结

### 主函数流程

```
main.py:
├── 1. 加载知识图谱
│   └── KnowledgeGraph(data_path)
│       ├── 读取 entities.dict, relations.dict
│       ├── 读取 train.txt, valid.txt, test.txt
│       └── 构建邻接表和索引
│
├── 2. 创建数据集
│   ├── TrainDataset (用于grounding)
│   ├── ValidDataset, TestDataset (用于评估)
│   └── RuleDataset (用于预训练)
│
├── 3. 初始化RulE模型
│   ├── 实体嵌入 (num_entities × hidden_dim*2)
│   ├── 关系嵌入 (num_relations × hidden_dim)
│   └── 读取规则并创建规则嵌入
│
├── 4. 预训练阶段 (PreTrainer)
│   ├── for step in range(max_steps):
│   │   ├── 采样三元组batch
│   │   ├── 采样规则batch
│   │   ├── 计算 L_fact 和 L_rule
│   │   ├── 反向传播更新参数
│   │   └── 定期验证并保存最佳模型
│   │
│   └── 加载最佳检查点
│
├── 5. 规则落地训练 (GroundTrainer)
│   ├── 冻结 entity_emb, relation_emb, rule_emb
│   ├── 预计算规则权重
│   │
│   └── for iter in range(num_iters):
│       ├── for batch in train_dataloader:
│       │   ├── 规则落地：grounding(h, r, rule_body)
│       │   ├── 聚合规则特征
│       │   ├── MLP评分
│       │   ├── 计算交叉熵损失
│       │   └── 更新MLP参数
│       │
│       └── 验证并保存最佳模型
│
└── 6. 最终评估
    ├── 加载最佳grounding模型
    ├── 在valid和test集上评估
    └── 输出 MRR, Hits@1/3/10, MR
```

### 超参数配置示例

以UMLS数据集为例：

```json
{
    "hidden_dim": 2000,           // 嵌入维度
    "gamma_fact": 6,              // 三元组margin
    "gamma_rule": 8,              // 规则margin
    "learning_rate": 0.0001,      // 学习率
    "max_steps": 30000,           // 预训练步数
    "batch_size": 256,            // 三元组batch大小
    "rule_batch_size": 256,       // 规则batch大小
    "negative_sample_size": 512,  // 负采样数量
    "adversarial_temperature": 0.25,  // 自对抗温度
    "weight_rule": 1,             // 规则损失权重

    "mlp_rule_dim": 100,          // MLP特征维度
    "alpha": 2.0,                 // KGE和规则融合权重
    "smoothing": 0.2,             // 标签平滑
    "g_lr": 0.0001,               // Grounding学习率
    "num_iters": 20               // Grounding迭代次数
}
```

---

## 关键创新点总结

1. **统一嵌入空间**：
   - 将逻辑规则和知识图谱嵌入到同一向量空间
   - 规则嵌入通过聚合规则体的关系嵌入得到

2. **规则落地机制**：
   - 使用图神经网络的消息传递进行多跳推理
   - 高效计算规则实例化的路径数量

3. **两阶段训练**：
   - 预训练学习语义嵌入
   - Grounding训练学习规则聚合策略

4. **自对抗负采样**：
   - 动态调整负样本权重
   - 加速训练收敛

5. **混合推理**：
   - 结合符号推理（规则）和向量推理（嵌入）
   - 可调节的融合权重

---

## 文件结构与代码对应

```
src/
├── main.py              # 主训练流程
│   └── main():89-201    # 完整训练pipeline
│
├── model.py             # RulE模型定义
│   ├── __init__:10-93   # 初始化嵌入
│   ├── RotatE:236-266   # RotatE评分函数
│   ├── add_ruleE:270-301       # 规则嵌入评分
│   ├── forward:337-409         # 规则落地推理
│   └── eval_compute_rule_weight:414-431  # 预计算规则权重
│
├── trainer.py           # 训练器
│   ├── PreTrainer
│   │   ├── train:35-124        # 预训练主循环
│   │   ├── train_step:126-224  # 单步训练（L_fact + L_rule）
│   │   └── evaluate:226-326    # KGE评估
│   │
│   └── GroundTrainer
│       ├── train:387-458       # Grounding训练主循环
│       ├── train_step:463-520  # 单步训练（L_ground）
│       ├── evaluate:522-630    # 纯规则评估
│       └── evaluate_t:634-742  # 混合评估
│
├── data.py              # 数据处理
│   ├── KnowledgeGraph:222-447
│   │   ├── __init__:223-382    # 加载数据
│   │   ├── grounding:410-421   # 规则落地
│   │   └── propagate:423-447   # 图传播
│   │
│   ├── RuleDataset:11-68       # 规则数据集
│   ├── KGETrainDataset:71-177  # 三元组数据集
│   └── TrainDataset:449-493    # Grounding数据集
│
├── layers.py            # 神经网络层
│   ├── MLP:7-49                # 多层感知机
│   └── FuncToNodeSum:51-73     # 规则聚合层
│
└── utils.py             # 工具函数
    ├── load_config:12-28       # 加载配置
    ├── save_model:43-72        # 保存模型
    └── set_logger:79-93        # 日志设置
```

---

## 总结

RulE模型通过以下方式实现知识图谱推理：

1. **预训练**：使用RotatE学习实体和关系的向量表示，同时学习规则的抽象嵌入
2. **规则落地**：通过图传播算法在知识图谱上实例化抽象规则
3. **推理**：结合规则落地结果和嵌入信息进行预测

模型的核心优势在于：
- 统一的嵌入空间同时表示符号规则和向量嵌入
- 高效的图传播算法实现规则落地
- 灵活的两阶段训练策略
- 可解释的推理过程（通过规则落地）

---

## 训练流程三个阶段详解

在 `src/main.py` 中，训练流程通过三个注释标记清晰地划分为三个阶段：

### 阶段1: `# For pre-training` (第133-156行)

**目的**: 预训练实体、关系和规则的嵌入

**主要操作**:
```python
# For pre-training

pre_trainer = PreTrainer(
    graph=graph,
    model=RulE_model,
    valid_set=valid_set,
    test_set=test_set,
    ruleset=ruleset,
    expectation=True,
    device = device,
    num_worker=args.cpu_num
)

# 如果取消注释这两行,会加载之前的预训练checkpoint继续训练
# checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
# RulE_model.load_state_dict(checkpoint['model'])

# 如果取消注释这两行,会在训练前评估初始模型(通常不需要)
# valid_mrr = pre_trainer.evaluate('valid', expectation=True)
# test_mrr = pre_trainer.evaluate('test', expectation=True)

# 如果取消注释这一行,会执行预训练(首次训练时需要)
# pre_trainer.train(args)
```

**训练内容**:
- 使用 RotatE 联合训练实体嵌入、关系嵌入和规则嵌入
- 优化三元组损失 $\mathcal{L}_{fact}$ 和规则损失 $\mathcal{L}_{rule}$
- 训练步数: `max_steps` (通常为30000步)

**输出文件**:
- `checkpoint` - 最佳预训练模型参数
- `entity_embedding.npy` - 实体嵌入
- `relation_embedding.npy` - 关系嵌入
- `rule_embedding.npy` - 规则嵌入

**使用场景**:
- **首次训练**: 取消注释第155行 `pre_trainer.train(args)`,执行完整预训练
- **继续预训练**: 取消注释第148-149行加载checkpoint + 第155行继续训练
- **跳过预训练**: 全部注释(已有预训练模型时),直接进入阶段2

---

### 阶段2: `# load rule embedding and KGE embedding` (第162-171行)

**目的**: 加载预训练好的嵌入,评估预训练效果,准备进入grounding阶段

**主要操作**:
```python
# load rule embedding and KGE embedding

# 加载预训练的最佳模型
checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
RulE_model.load_state_dict(checkpoint['model'])

logging.info('Test the results of pre-training')

# 评估预训练模型的性能
valid_mrr = pre_trainer.evaluate('valid', expectation=True)
test_mrr = pre_trainer.evaluate('test', expectation=True)
```

**功能说明**:
- 从 `checkpoint` 文件加载预训练阶段学到的所有参数
- 在验证集和测试集上评估预训练模型的质量
- 输出评估指标: MRR, Hits@1/3/10, MR

**为什么需要这个阶段**:
1. **验证预训练质量**: 确保预训练达到预期效果
2. **加载最佳模型**: 预训练期间保存的是验证集上最好的模型,而不是最后一步的模型
3. **准备frozen参数**: 接下来grounding阶段会冻结这些参数

**与阶段1的区别**:
- **阶段1 (第148-149行)**: 在预训练**之前**加载checkpoint,用于继续未完成的预训练
- **阶段2 (第164-165行)**: 在预训练**之后**加载checkpoint,用于获取最佳预训练结果

---

### 阶段3: `# RulE_model.add_param()` (第173-194行)

**目的**: Grounding训练 - 学习如何聚合规则落地结果

**主要操作**:
```python
# RulE_model.add_param()
# ↑ 这是历史遗留注释,原本用于添加grounding参数
# 现在这些参数在模型初始化时就创建了,所以这行被注释掉

# 如果有之前的grounding checkpoint,加载它(继续grounding训练)
checkpoint = torch.load(os.path.join(args.save_path, 'grounding.pt'))
RulE_model.load_state_dict(checkpoint['model'])

# 创建Grounding训练器
ground_trainer = GroundTrainer(
    model=RulE_model,
    args = args,
    train_set=train_set,
    valid_set=valid_set,
    test_set=test_set,
    test_kge_set = test_kge_set,
    device=device,
    num_worker=args.cpu_num
)

# 可选: grounding训练前的评估(通常不需要)
# valid_mrr = ground_trainer.evaluate('valid', expectation=True)
# test_mrr = ground_trainer.evaluate('test', expectation=True)

# 执行grounding训练
ground_trainer.train(args)
```

**训练内容**:
- **冻结参数**: entity_embedding, relation_embedding, rule_emb (不再更新)
- **训练参数**: mlp_feature, score_model (MLP聚合网络)
- **训练方法**:
  - 规则落地: 通过图传播在KG上实例化规则
  - 特征聚合: 使用MLP聚合规则落地的计数
  - 损失函数: 交叉熵损失 + 标签平滑
- **训练轮数**: `num_iters` (通常为20轮)

**输出文件**:
- `grounding.pt` - Grounding阶段模型参数
- `g_rule_embedding.npy` - Grounding后的规则MLP特征

**关于 `add_param()` 注释**:
- 这是早期版本的遗留代码标记
- 原本用于在grounding前动态添加MLP参数
- 现在这些参数在 `model.py` 的 `__init__` 中就已创建,不需要额外添加
- 保留这个注释是为了标记"从这里开始进入grounding阶段"

---

### 三个阶段的对比总结

| 阶段 | 注释标记 | 主要操作 | 可训练参数 | 输出 | 典型配置 |
|------|---------|---------|-----------|------|---------|
| **阶段1** | `# For pre-training` | 联合训练嵌入 | entity_emb<br>relation_emb<br>rule_emb | checkpoint<br>xxx_emb.npy | 首次训练时取消注释155行 |
| **阶段2** | `# load rule embedding...` | 加载并评估预训练模型 | 无(仅评估) | 评估指标 | 始终执行(第164-171行) |
| **阶段3** | `# RulE_model.add_param()` | Grounding训练 | mlp_feature<br>score_model<br>(冻结其他) | grounding.pt<br>g_rule_emb.npy | 始终执行(第178-194行) |

---

### 典型训练场景

#### 场景1: 首次完整训练

```python
# 阶段1: 取消注释预训练
pre_trainer.train(args)  # 第155行

# 阶段2: 加载并评估(保持默认)
checkpoint = torch.load(...)  # 第164-165行
pre_trainer.evaluate(...)     # 第170-171行

# 阶段3: Grounding训练(保持默认)
ground_trainer.train(args)    # 第194行
```

#### 场景2: 已有预训练,只训练grounding

```python
# 阶段1: 全部注释(跳过预训练)
# pre_trainer.train(args)

# 阶段2: 加载预训练模型(保持默认)
checkpoint = torch.load(...)
pre_trainer.evaluate(...)

# 阶段3: Grounding训练(保持默认)
ground_trainer.train(args)
```

#### 场景3: 继续未完成的grounding训练

```python
# 阶段1: 全部注释(跳过预训练)

# 阶段2: 加载预训练模型(保持默认)

# 阶段3: 取消注释加载grounding checkpoint
checkpoint = torch.load('grounding.pt')  # 第175-176行
RulE_model.load_state_dict(...)
ground_trainer.train(args)
# 注意: 仍会从第1轮训练到第20轮,会重复已完成的轮次
```

---

### 代码执行流程图

```
main.py 执行流程:
│
├─【阶段1准备】第133-146行
│  └─ 创建 PreTrainer(仅创建对象,不训练)
│
├─【阶段1可选】第148-156行 (全部被注释)
│  ├─ 第148-149行: 加载之前的checkpoint (用于继续预训练)
│  ├─ 第152-153行: 训练前评估(通常不需要)
│  └─ 第155行: 执行预训练 ← 首次训练时需要取消注释
│
├─【阶段2必须】第162-171行 ✅
│  ├─ 第164-165行: 加载预训练最佳模型
│  └─ 第170-171行: 评估预训练效果
│      └─ 输出: MRR=0.802 (你日志中看到的)
│
└─【阶段3必须】第173-194行 ✅
   ├─ 第175-176行: 可选加载grounding checkpoint
   ├─ 第178-187行: 创建 GroundTrainer
   ├─ 第189-190行: 可选grounding前评估
   └─ 第194行: 执行grounding训练(20轮)
```

---

这三个阶段清晰地体现了RulE的设计理念:
1. **先学习语义** (预训练) - 理解实体、关系、规则的含义
2. **验证语义质量** (评估) - 确保学到了有用的表示
3. **学习推理** (grounding) - 在固定语义的基础上学习如何使用规则进行推理
