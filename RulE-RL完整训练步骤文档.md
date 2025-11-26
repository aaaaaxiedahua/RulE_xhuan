# RulE-RL 完整训练步骤文档

本文档详细描述RulE-RL的每一个训练步骤，包括输入、输出、数学公式和使用的技术。

---

## 目录

1. [训练流程总览](#一训练流程总览)
2. [Phase 1: 加载预训练模型](#二phase-1-加载预训练模型)
3. [Phase 2: 初始化RL组件](#三phase-2-初始化rl组件)
4. [Phase 3: Episode训练循环](#四phase-3-episode训练循环)
5. [Phase 4: 模型更新](#五phase-4-模型更新)
6. [Phase 5: 评估与保存](#六phase-5-评估与保存)
7. [完整算法伪代码](#七完整算法伪代码)
8. [技术栈总结](#八技术栈总结)

---

## 一、训练流程总览

### 1.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RulE-RL 完整训练流程                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │ Phase 1      │──▶│ Phase 2      │──▶│ Phase 3      │                │
│  │ 加载预训练    │   │ 初始化RL组件 │   │ Episode训练  │                │
│  └──────────────┘   └──────────────┘   └──────┬───────┘                │
│                                                │                         │
│                                                ▼                         │
│                                         ┌──────────────┐                │
│                                         │ Phase 4      │                │
│                                         │ 模型更新     │                │
│                                         └──────┬───────┘                │
│                                                │                         │
│                                                ▼                         │
│                                         ┌──────────────┐                │
│                                         │ Phase 5      │                │
│                                         │ 评估与保存   │                │
│                                         └──────────────┘                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 RulE-RL 完整配置与参数说明

---

#### A. RulE-RL 配置文件 `config/umls_rule_rl_config.json`

```json
{
    "comment": "=== 预训练部分 ===",
    "data_path": "../data/umls",
    "rule_file": "../data/umls/mined_rules.txt",
    "pretrain_checkpoint": "../outputs/umls/checkpoint",
    "cuda": true,
    "cpu_num": 10,
    "seed": 800,
    "hidden_dim": 2000,
    "mlp_rule_dim": 100,
    "gamma_fact": 6,
    "gamma_rule": 8,
    "p_norm": 2,

    "comment": "=== RL部分 ===",
    "save_path": "../outputs/umls",
    "state_dim": 128,
    "history_dim": 128,
    "policy_hidden_dim": 256,
    "value_hidden_dim": 256,
    "top_k_rules": 5,
    "max_steps": 5,
    "gamma": 0.99,
    "epsilon_start": 0.5,
    "epsilon_end": 0.05,
    "ucb_c": 1.0,
    "alpha": 0.1,
    "beta": 0.05,
    "lr_policy": 0.001,
    "lr_value": 0.001,
    "lr_selector": 0.0001,
    "num_epochs": 100,
    "log_interval": 100,
    "eval_interval": 5,
    "save_interval": 10,
    "grad_clip": 1.0
}
```

---

#### B. 参数详细说明

| 参数名 | 值 | 类型 | 说明 |
|-------|-----|------|------|
| **数据与路径** | | | |
| `data_path` | `../data/umls` | 共用 | UMLS数据集路径 |
| `rule_file` | `../data/umls/mined_rules.txt` | 共用 | 挖掘的规则文件路径 |
| `pretrain_checkpoint` | `../outputs/umls/checkpoint` | 预训练 | 预训练RulE模型路径（冻结参数）|
| `save_path` | `../outputs/rule_rl` | RL | RulE-RL保存路径（只保存RL组件）|
| **设备配置** | | | |
| `cuda` | `true` | 共用 | 是否使用GPU |
| `cpu_num` | `10` | 共用 | CPU线程数 |
| `seed` | `800` | 共用 | 随机种子 |
| **预训练嵌入维度** | | | |
| `hidden_dim` | `2000` | 预训练 | 关系嵌入维度 (决定entity_dim=4000) |
| `mlp_rule_dim` | `100` | 预训练 | 规则特征维度 |
| `gamma_fact` | `6` | 预训练 | 三元组margin (RotatE距离计算) |
| `gamma_rule` | `8` | 预训练 | 规则margin (RL奖励计算规则置信度) |
| `p_norm` | `2` | 预训练 | 距离范数类型 (L2范数) |
| **RL模型架构** | | | |
| `state_dim` | `128` | RL | StateEncoder输出的状态向量维度 |
| `history_dim` | `128` | RL | GRU编码历史路径的隐藏层维度 |
| `policy_hidden_dim` | `256` | RL | PathFinder策略网络隐藏层维度 |
| `value_hidden_dim` | `256` | RL | PathFinder价值网络隐藏层维度 |
| **RL超参数** | | | |
| `top_k_rules` | `5` | RL | RuleSelector每次选择的规则数量 |
| `max_steps` | `5` | RL | 单个episode的最大步数 |
| `gamma` | `0.99` | RL | 折扣因子 (计算累积回报) |
| `epsilon_start` | `0.5` | RL | ε-greedy初始探索率 |
| `epsilon_end` | `0.05` | RL | ε-greedy最终探索率 |
| `ucb_c` | `1.0` | RL | UCB探索系数 (平衡exploration/exploitation) |
| **奖励参数** | | | |
| `alpha` | `0.1` | RL | 中间奖励权重 (rule_consistency, getting_closer) |
| `beta` | `0.0` | RL | (已停用) 旧版惩罚权重，仅保留占位 |
| **学习率** | | | |
| `lr_policy` | `0.001` | RL | PathFinder策略网络学习率 |
| `lr_value` | `0.001` | RL | PathFinder价值网络学习率 |
| `lr_selector` | `0.0001` | RL | RuleSelector网络学习率 |
| **训练控制** | | | |
| `num_epochs` | `100` | RL | RL训练总轮数 |
| `log_interval` | `100` | RL | 每多少个episode打印一次日志 |
| `eval_interval` | `5` | RL | 每多少个epoch在验证集上评估一次 |
| `save_interval` | `10` | RL | 每多少个epoch保存一次检查点 |
| `grad_clip` | `1.0` | RL | 梯度裁剪的最大范数 (防止梯度爆炸) |

**为什么需要两个路径**:
- `pretrain_checkpoint`: 加载预训练嵌入（entity_embedding, relation_embedding, rule_emb），**这些参数会被冻结**
- `save_path`: 保存RL组件（StateEncoder, RuleSelector, PathFinder），**不包含冻结的预训练参数**
- 两个路径指向同一个文件夹 `../outputs/umls/`，便于统一管理同一数据集的所有模型文件

**总奖励包含什么**:

最新奖励只包含三部分：终止二元奖励 `R_final_bin`、规则一致性 `R_rule`、以及仅在失败时启用的归一化接近奖励 `R_closer_norm`。

| 奖励类型 | 权重 | 取值范围 | 作用 |
|---------|------|---------|------|
| 最终奖励 `R_final_bin` | 1.0 | {0, 1} | 命中目标 = 1，失败 = 0 |
| 规则一致性 `R_rule` | α=0.1 | [0, 1.0] | 路径符合规则模式，利用规则置信度 |
| 接近目标 `R_closer_norm` | α=0.1 | [0, 1.0] | 失败时才启用的归一化接近奖励 |

**已移除 / 停用的组件**:
- ~~探索多样性 `R_diversity`~~: 贡献 < 1%，且可能与最短路径目标冲突
- ~~长度惩罚 `P_length`~~: 始终为0 (episode在max_steps时自动终止)
- ~~死胡同惩罚 `P_dead`、循环惩罚 `P_loop`~~: 最新版本不再使用

**公式**:
```
R_total = R_final_bin
        + 0.1 × (R_rule + (1 - R_final_bin) × R_closer_norm)
```

> `R_closer_norm` 通过将原始距离缩短量除以起点距离 (或max_steps) 并截断到[0,1] 来得到，成功时 `R_final_bin = 1` 使其自动失效；失败时它提供最多0.1的补偿信号。

**示例**:
```
查询: (aspirin, treats, headache)
路径: aspirin → pain → headache (2步，成功)

奖励计算:
  R_final_bin = 1        (命中目标)
  R_rule = 0.85          (路径匹配规则: treats ∧ relieves)
  R_closer_norm = 0      (成功 → 关闭)

  R_total = 1 + 0.1×(0.85 + 0)
          = 1.085

失败示例:
  R_final_bin = 0
  R_rule = 0.72
  R_closer_norm = 0.6    (归一化接近奖励)
  R_total = 0 + 0.1×(0.72 + 0.6)
          = 0.132
```

**文件夹结构**:
```
outputs/umls/
├── checkpoint          # 预训练RulE模型（加载用，冻结）
├── grounding.pt        # Grounding阶段模型（可选）
├── rule_rl_checkpoint  # RulE-RL的RL组件（训练保存）
└── config.json         # 配置文件备份
```

---

#### C. 派生维度 (从配置自动计算)

| 维度名 | 计算方式 | 值 (UMLS) | 说明 |
|--------|---------|----------|------|
| `entity_dim` | `hidden_dim × 2` | 4000 | 实体嵌入维度 (RotatE复数表示) |
| `rel_dim` | `hidden_dim` | 2000 | 关系嵌入维度 |
| `rule_dim` | `mlp_rule_dim` | 100 | 规则嵌入维度 |
| `query_dim` | `entity_dim + rel_dim` | 6000 | 查询向量维度 (实体+关系拼接) |
| `num_relations` | 从数据集读取 | 46 | 动作空间大小 (含逆关系) |
| `num_entities` | 从数据集读取 | 135 | 实体总数 |
| `num_rules` | 从规则文件读取 | 18400 | 规则总数 |

---

#### D. 实际张量维度

| 张量 | 维度 | 来源 |
|------|------|------|
| `entity_embedding` | [135, 4000] | 预训练checkpoint |
| `relation_embedding` | [46, 2000] | 预训练checkpoint |
| `rule_emb` | [18400, 100] | 预训练checkpoint |
| 查询表示 | [6000] | entity_emb + rel_emb拼接 |
| 状态编码 | [128] | StateEncoder输出 |
| 策略logits | [46] | PolicyNetwork输出 |
| 状态价值 | [1] | ValueNetwork输出 |
| UCB得分 | [18400] | RuleSelector输出 |

---

---

## 二、Phase 1: 加载预训练模型

### Step 1.1: 加载知识图谱

**输入**:
- `data_path`: 数据集路径 (如 `../data/umls`)

**输出**:
- `graph`: KnowledgeGraph对象

**数据结构**:
```python
# 输入文件
entities.dict:    entity_id \t entity_name
relations.dict:   relation_id \t relation_name
train.txt:        head \t relation \t tail
valid.txt:        head \t relation \t tail
test.txt:         head \t relation \t tail
mined_rules.txt:  rule_head rule_body_1 rule_body_2 ...

# 输出数据结构
graph = {
    'num_entities': 135,           # UMLS实体数
    'num_relations': 46,           # UMLS关系数 (含逆关系)
    'adjacency_list': {            # 邻接表
        entity_id: {
            relation_id: [neighbor_ids]
        }
    },
    'hr2t': {                      # (head, rel) -> tails 映射
        (head, rel): {tail_ids}
    },
    'triplets': [(h, r, t), ...]   # 所有三元组
}
```

**使用技术**:
- 邻接表 (Adjacency List): O(1)邻居查询
- 字典 (Dict/HashMap): O(1)键值查询

---

### Step 1.2: 加载预训练RulE模型

**输入**:
- `checkpoint_path`: 预训练模型路径
- `graph`: 知识图谱对象

**输出**:
- `rule_model`: 加载了预训练权重的RulE模型

**代码**:
```python
# 创建模型
rule_model = RulE(
    graph=graph,
    hidden_dim=200,        # 关系嵌入维度
    p_norm=2,              # L2范数
    gamma_fact=6,          # 三元组margin
    gamma_rule=5           # 规则margin
)

# 加载预训练权重
checkpoint = torch.load(checkpoint_path)
rule_model.load_state_dict(checkpoint['model'])
```

**加载的权重**:

| 参数名 | 维度 | 说明 |
|--------|------|------|
| `entity_embedding` | [num_entities, hidden_dim×2] | 实体嵌入 (复数) |
| `relation_embedding` | [num_relations, hidden_dim] | 关系嵌入 (相位) |
| `rule_emb` | [num_rules, rule_dim] | 规则嵌入 |

**使用技术**:
- RotatE嵌入: 复数空间中的旋转操作
- PyTorch状态字典: 模型参数序列化

---

### Step 1.3: 冻结预训练参数

**输入**:
- `rule_model`: 预训练模型

**输出**:
- 冻结参数后的模型 (不参与梯度计算)

**代码**:
```python
# 冻结实体嵌入
for param in rule_model.entity_embedding.parameters():
    param.requires_grad = False

# 冻结关系嵌入
for param in rule_model.relation_embedding.parameters():
    param.requires_grad = False

# 冻结规则嵌入
rule_model.rule_emb.requires_grad = False
```

**冻结后效果**:

| 参数 | 冻结前 | 冻结后 |
|------|--------|--------|
| `entity_embedding` | 可训练 | 只读特征 |
| `relation_embedding` | 可训练 | 只读特征 |
| `rule_emb` | 可训练 | 只读特征 |

**使用技术**:
- 迁移学习 (Transfer Learning): 复用预训练知识
- 参数冻结 (Parameter Freezing): requires_grad=False

---

## 三、Phase 2: 初始化RL组件

### Step 2.1: 初始化StateEncoder

**输入**:
- `entity_dim`: 实体嵌入维度 (4000, 从预训练读取)
- `rel_dim`: 关系嵌入维度 (2000, 从预训练读取)
- `rule_dim`: 规则特征维度 (100, 从预训练读取)
- `history_dim`: 历史编码维度 (128, RL自定义)

**输出**:
- `state_encoder`: StateEncoder模块

**网络结构** (UMLS数据集):
```
StateEncoder
├── entity_encoder: Linear(4000 → 128)
├── relation_encoder: Linear(2000 → 128)
├── rule_encoder: LSTM(100 → 128)
├── history_encoder: GRU(6000 → 128)   # entity_dim + rel_dim
└── state_fusion: MLP(512 → 256 → 128)
```

**数学公式**:
```
h_entity = ReLU(W_e · current_entity + b_e)       # [128] ← current_entity: [4000]
h_rel = ReLU(W_r · query_relation + b_r)          # [128] ← query_relation: [2000]
h_rule = LSTM(rule_context)[-1]                   # [128] ← rule_context: [K, 100]
h_history = GRU(path_history)[-1]                 # [128] ← path_history: [T, 6000]

state = MLP(concat[h_entity, h_rel, h_rule, h_history])  # [128]
```

**使用技术**:
- MLP (多层感知机): 特征变换
- LSTM (长短期记忆): 序列建模 (规则上下文)
- GRU (门控循环单元): 序列建模 (路径历史)
- 特征融合 (Feature Fusion): 多模态信息整合

---

### Step 2.2: 初始化RuleSelectorAgent (高层Agent)

**输入**:
- `query_dim`: 查询嵌入维度 (6000 = 4000 + 2000)
- `rule_dim`: 规则嵌入维度 (100, 从预训练读取)
- `num_rules`: 规则总数 (UMLS: 18400)

**输出**:
- `rule_selector`: RuleSelectorAgent模块

**网络结构** (UMLS数据集):
```
RuleSelectorAgent
├── query_encoder: MLP(6000 → 256 → 128)
├── rule_query_matcher: MLP(228 → 128 → 1)  # 128 + 100
└── UCB统计:
    ├── rule_counts: Dict[rule_id → int]
    ├── rule_rewards: Dict[rule_id → float]
    └── total_selections: int
```

**数学公式**:

1. **查询编码**:
```
query_repr = concat[entity_emb, relation_emb]     # [6000] = [4000] + [2000]
query_emb = MLP_query(query_repr)                  # [128]
```

2. **神经匹配得分**:
```
combined_i = concat[query_emb, rule_emb_i]        # [228] = [128] + [100]
neural_score_i = MLP_matcher(combined_i)          # [1]
```

3. **UCB得分**:
```
UCB(i) = neural_score(i) + c × √(2 × ln(N) / n_i)

其中:
- c = 1.0 (探索系数)
- N = total_selections (总选择次数)
- n_i = rule_counts[i] (规则i被选次数)
```

4. **规则选择**:
```
if random() < ε:
    selected = random_sample(rules, K)    # 探索
else:
    selected = top_k(UCB_scores, K)       # 利用
```

**使用技术**:
- Contextual Bandit: 上下文感知的规则选择
- UCB (Upper Confidence Bound): 探索-利用平衡
- ε-greedy: 随机探索策略

---

### Step 2.3: 初始化PathFinderAgent (低层Agent)

**输入**:
- `state_dim`: 状态维度 (128)
- `action_dim`: 动作空间大小 (num_relations)

**输出**:
- `path_finder`: PathFinderAgent模块

**网络结构**:
```
PathFinderAgent
├── policy_net: MLP(128 → 256 → 256 → num_relations)
└── value_net: MLP(128 → 256 → 1)
```

**数学公式**:

1. **策略网络 (Actor)**:
```
logits = Policy_net(state)                        # [num_relations]
logits_masked = logits.masked_fill(~mask, -∞)     # 应用动作掩码
π(a|s) = softmax(logits_masked)                   # 动作概率分布
```

2. **价值网络 (Critic)**:
```
V(s) = Value_net(state)                           # 状态价值估计
```

3. **动作选择**:
```
训练时: a ~ Categorical(π(·|s))                   # 采样
测试时: a = argmax π(·|s)                         # 贪心
```

4. **对数概率**:
```
log π(a|s) = log(π(a|s))                          # 用于策略梯度
```

**使用技术**:
- Actor-Critic架构: 策略网络 + 价值网络
- REINFORCE with Baseline: 减小方差的策略梯度
- Categorical分布: 离散动作采样

---

### Step 2.4: 初始化KGReasoningEnv (环境)

**输入**:
- `graph`: 知识图谱
- `rule_model`: 预训练RulE模型
- `state_encoder`: 状态编码器
- `max_steps`: 最大步数 (5)

**输出**:
- `env`: KGReasoningEnv环境

**环境接口**:
```python
class KGReasoningEnv:
    def reset(query) -> state:
        """重置环境,返回初始状态"""
        pass

    def step(action, selected_rules) -> (next_state, reward, done, info):
        """执行动作,返回转移结果"""
        pass

    def get_action_mask(selected_rules) -> mask:
        """获取有效动作掩码"""
        pass
```

**状态转移公式**:
```
s_0 = (head_entity, query_relation, [], [])       # 初始状态

s_t+1 = (
    next_entity,           # 新位置
    query_relation,        # 查询目标 (不变)
    selected_rules,        # 选中规则 (不变)
    path_history + [(current_entity, action)]  # 更新历史
)
```

**动作掩码计算**:
```
valid_actions = outgoing_rels(current_entity) ∩ rule_body_rels(selected_rules)
mask[i] = True  if i ∈ valid_actions else False
```

**使用技术**:
- OpenAI Gym接口: 标准RL环境规范
- 动作掩码 (Action Masking): 约束无效动作
- 邻接表查询: O(1)邻居获取

---

### Step 2.5: 初始化RewardCalculator (奖励计算器)

**输入**:
- `rule_model`: 预训练模型 (用于嵌入距离计算)
- `alpha`: 中间奖励权重 (0.1)

**输出**:
- `reward_calculator`: RewardCalculator模块

**奖励公式** (简化版):

**总奖励**:
```
R_total = R_final_bin + α × (R_rule + (1 - R_final_bin) × R_closer_norm)
```

**1. 最终奖励 (R_final_bin)**:
```
R_final_bin = {
    1,  if final_entity == target
    0,  otherwise
}
```

**2. 规则一致性奖励 (R_rule)**:
```
R_rule = max{ conf(rule) | rule.body == path_relations }

其中规则置信度:
conf(rule) = (γ_rule - ||body_sum + rule_emb - head_emb||₂) / γ_rule
```

**3. 接近目标奖励 (R_closer_norm)**:
```
raw = Σ max(0, dist(e_{t-1}, target) - dist(e_t, target))
R_closer_norm = min(1, raw / (dist(e_0, target) + ε))
```
> 仅当 `R_final_bin = 0` 时启用：`α × (1 - R_final_bin) × R_closer_norm`

**已移除**:
- ~~探索多样性奖励 (R_diversity)~~: 贡献 < 1%，与最短路径目标冲突
- ~~长度/死胡同/循环惩罚~~: 最新方案中不再使用

**使用技术**:
- 奖励塑形 (Reward Shaping): 稠密奖励信号
- 符号-神经混合: 结合规则嵌入和实体嵌入
- 嵌入空间距离: L2范数作为相似度度量

---

### Step 2.6: 初始化优化器

**输入**:
- `path_finder`: PathFinder Agent
- `rule_selector`: RuleSelector Agent
- 学习率配置

**输出**:
- 三个优化器

**代码**:
```python
# 策略网络优化器
policy_optimizer = Adam(
    path_finder.policy_net.parameters(),
    lr=1e-3
)

# 价值网络优化器
value_optimizer = Adam(
    path_finder.value_net.parameters(),
    lr=1e-3
)

# 规则选择器优化器
rule_selector_optimizer = Adam(
    rule_selector.parameters(),
    lr=1e-4
)
```

**使用技术**:
- Adam优化器: 自适应学习率
- 分离优化: 不同组件使用不同学习率

---

## 四、Phase 3: Episode训练循环

### Step 3.1: 采样训练查询

**输入**:
- `train_triplets`: 训练集三元组

**输出**:
- `query`: (head, relation, tail) 单个查询

**数据流**:
```
train_triplets: [(h₁,r₁,t₁), (h₂,r₂,t₂), ...]
    ↓ 按顺序或随机采样
query: (head, relation, tail)
```

---

### Step 3.2: 高层Agent选择规则

**输入**:
- `head`: 头实体ID
- `relation`: 查询关系ID
- `epsilon`: 当前探索率

**输出**:
- `selected_rules`: Top-K规则ID列表
- `selection_probs`: 选择概率

**详细步骤**:

```
Step 3.2.1: 获取嵌入
─────────────────────
entity_emb = rule_model.entity_embedding[head]     # [4000]
rel_emb = rule_model.relation_embedding[relation]  # [2000]
rule_embeddings = rule_model.rule_emb              # [18400, 100]

Step 3.2.2: 编码查询
─────────────────────
query_repr = concat[entity_emb, rel_emb]           # [6000]
query_emb = query_encoder(query_repr)              # [128]

Step 3.2.3: 计算神经匹配得分
─────────────────────────────
for i in range(num_rules):
    combined = concat[query_emb, rule_embeddings[i]]  # [228]
    neural_scores[i] = rule_query_matcher(combined)   # [1]

Step 3.2.4: 计算UCB得分
────────────────────────
for i in range(num_rules):
    avg_reward = rule_rewards[i] / (rule_counts[i] + 1)
    ucb_bonus = sqrt(2 * log(total_selections + 1) / (rule_counts[i] + 1))
    ucb_scores[i] = neural_scores[i] + ucb_bonus

Step 3.2.5: ε-greedy选择
─────────────────────────
if random() < epsilon:
    selected_rules = random.sample(candidate_rules, K)
else:
    selected_rules = topk(ucb_scores, K)

Step 3.2.6: 计算选择概率
─────────────────────────
selection_probs = softmax(neural_scores[selected_rules])
```

---

### Step 3.3: 环境初始化

**输入**:
- `query`: (head, relation, tail)

**输出**:
- `state`: 初始状态编码

**详细步骤**:

```
Step 3.3.1: 重置环境状态
─────────────────────────
env.current_entity = head
env.query_relation = relation
env.target_entity = tail
env.trajectory = [(head, None)]
env.path_history = []
env.step_count = 0

Step 3.3.2: 编码初始状态
─────────────────────────
state = state_encoder(
    current_entity = entity_embedding[head],               # [4000]
    query_relation = relation_embedding[relation],         # [2000]
    rule_context = [],                                     # 空
    path_history = []                                      # 空
)
# state: [128]
```

---

### Step 3.4: Episode循环 (核心)

**输入**:
- `state`: 当前状态
- `selected_rules`: 选中的规则
- `env`: 环境

**输出**:
- `episode_data`: 轨迹数据

**详细步骤** (每个时间步):

```
┌─────────────────────────────────────────────────────────────┐
│                    Episode 单步循环                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  while not done:                                             │
│                                                              │
│    Step 3.4.1: 获取动作掩码                                  │
│    ─────────────────────────                                 │
│    outgoing = graph.get_outgoing_relations(current_entity)  │
│    rule_rels = union(rule.body for rule in selected_rules)  │
│    valid_actions = outgoing ∩ rule_rels                     │
│    mask = [True if i in valid_actions else False]           │
│                                                              │
│    Step 3.4.2: 策略网络选择动作                              │
│    ───────────────────────────                               │
│    logits = policy_net(state)                               │
│    logits[~mask] = -inf                                     │
│    probs = softmax(logits)                                  │
│    action = sample(Categorical(probs))                      │
│    log_prob = log(probs[action])                            │
│                                                              │
│    Step 3.4.3: 环境执行动作                                  │
│    ─────────────────────────                                 │
│    neighbors = graph.get_neighbors(current_entity, action)  │
│    next_entity = random.choice(neighbors)                   │
│    current_entity = next_entity                             │
│    step_count += 1                                          │
│                                                              │
│    Step 3.4.4: 更新轨迹                                      │
│    ────────────────────                                      │
│    trajectory.append((next_entity, action))                 │
│    path_history.append((entity_emb, action_emb))            │
│                                                              │
│    Step 3.4.5: 编码新状态                                    │
│    ────────────────────                                      │
│    next_state = state_encoder(                              │
│        current_entity = entity_embedding[next_entity],      │
│        query_relation = relation_embedding[query_rel],      │
│        rule_context = rule_embeddings[selected_rules],      │
│        path_history = path_history                          │
│    )                                                         │
│                                                              │
│    Step 3.4.6: 判断终止                                      │
│    ────────────────────                                      │
│    done = (next_entity == target) or (step_count >= max_steps) │
│                                                              │
│    Step 3.4.7: 记录数据                                      │
│    ────────────────────                                      │
│    episode_data.states.append(state)                        │
│    episode_data.actions.append(action)                      │
│    episode_data.log_probs.append(log_prob)                  │
│                                                              │
│    Step 3.4.8: 状态转移                                      │
│    ────────────────────                                      │
│    state = next_state                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### Step 3.5: 计算Episode奖励

**输入**:
- `trajectory`: Episode轨迹
- `target_entity`: 目标实体

**输出**:
- `total_reward`: 总奖励
- `reward_breakdown`: 奖励分解

**详细步骤** (简化版):

```
Step 3.5.1: 计算最终奖励（二元）
──────────────────────────────
final_entity = trajectory[-1][0]
R_final_bin = 1.0 if final_entity == target_entity else 0.0

Step 3.5.2: 计算规则一致性奖励
───────────────────────────────
path_rels = [step[1] for step in trajectory if step[1] is not None]
matched_rules = [rule for rule in rules if rule.body == path_rels]
R_rule = max(confidence(rule) for rule in matched_rules) if matched_rules else 0.0

Step 3.5.3: 计算接近目标奖励（归一化，仅失败时启用）
────────────────────────────────────────────
raw = 0
for t in range(1, len(trajectory)):
    dist_prev = ||entity_emb[trajectory[t-1][0]] - entity_emb[target]||₂
    dist_curr = ||entity_emb[trajectory[t][0]] - entity_emb[target]||₂
    raw += max(0, dist_prev - dist_curr)

dist_start = ||entity_emb[trajectory[0][0]] - entity_emb[target]||₂ + ε
R_closer_norm = min(1, raw / dist_start)

Step 3.5.4: 加权求和
─────────────────────
α = 0.1
total_reward = R_final_bin + α * (R_rule + (1 - R_final_bin) * R_closer_norm)
```

---

## 五、Phase 4: 模型更新

### Step 4.1: 计算折扣回报

**输入**:
- `rewards`: Episode各步奖励 (通常只有最后一步有奖励)
- `gamma`: 折扣因子 (0.99)

**输出**:
- `returns`: 各步折扣回报

**公式**:
```
G_t = r_t + γ × r_{t+1} + γ² × r_{t+2} + ... + γ^{T-t} × r_T

递推计算:
G_T = r_T
G_{T-1} = r_{T-1} + γ × G_T
...
G_0 = r_0 + γ × G_1
```

**代码**:
```python
returns = []
G = 0
for r in reversed(rewards):
    G = r + gamma * G
    returns.insert(0, G)
returns = torch.tensor(returns)
```

---

### Step 4.2: 计算优势函数

**输入**:
- `returns`: 折扣回报
- `states`: Episode状态序列

**输出**:
- `advantages`: 优势函数值

**公式**:
```
A_t = G_t - V(s_t)

其中:
- G_t: 实际折扣回报
- V(s_t): 价值网络估计的状态价值
```

**代码**:
```python
# 计算状态价值
states = torch.stack(episode_data['states'])
values = value_net(states).squeeze()  # [T]

# 计算优势函数
advantages = returns - values.detach()

# 标准化 (可选,减小方差)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

---

### Step 4.3: 更新PathFinder (低层Agent)

**输入**:
- `log_probs`: 动作对数概率
- `advantages`: 优势函数
- `returns`: 折扣回报
- `values`: 状态价值估计

**输出**:
- 更新后的策略网络和价值网络

**策略梯度公式 (REINFORCE with Baseline)**:
```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) × A(s,a)]

Policy Loss = -1/T × Σ log π(a_t|s_t) × A_t
```

**价值网络损失 (MSE)**:
```
Value Loss = 1/T × Σ (V(s_t) - G_t)²
```

**代码**:
```python
# Step 4.3.1: 策略损失
log_probs = torch.stack(episode_data['log_probs'])
policy_loss = -(log_probs * advantages).mean()

# Step 4.3.2: 价值损失
value_loss = F.mse_loss(values, returns)

# Step 4.3.3: 更新策略网络
policy_optimizer.zero_grad()
policy_loss.backward()
torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
policy_optimizer.step()

# Step 4.3.4: 更新价值网络
value_optimizer.zero_grad()
value_loss.backward()
torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
value_optimizer.step()
```

**使用技术**:
- REINFORCE算法: 基础策略梯度
- Baseline (价值网络): 减小方差
- 梯度裁剪 (Gradient Clipping): 稳定训练

---

### Step 4.4: 更新RuleSelector (高层Agent)

**输入**:
- `selected_rules`: 选中的规则
- `selection_probs`: 选择概率
- `episode_reward`: Episode总奖励

**输出**:
- 更新后的规则选择器

**策略梯度公式**:
```
∇_φ J(φ) = ∇_φ Σ log P(rule_i) × R_episode

Rule Selector Loss = -Σ log P(rule_i) × R_episode
```

**UCB统计更新**:
```
for rule_id in selected_rules:
    rule_counts[rule_id] += 1
    rule_rewards[rule_id] += episode_reward
total_selections += len(selected_rules)
```

**代码**:
```python
# Step 4.4.1: 策略梯度更新
rule_selector_loss = -torch.sum(torch.log(selection_probs + 1e-10)) * episode_reward

rule_selector_optimizer.zero_grad()
rule_selector_loss.backward()
rule_selector_optimizer.step()

# Step 4.4.2: UCB统计更新
for rule_id in selected_rules:
    rule_selector.rule_counts[rule_id] += 1
    rule_selector.rule_rewards[rule_id] += episode_reward
rule_selector.total_selections += len(selected_rules)
```

**使用技术**:
- 策略梯度: 端到端优化
- UCB更新: 在线学习统计

---

### Step 4.5: 更新探索率

**输入**:
- `epoch`: 当前epoch
- `epsilon_start`: 初始探索率
- `epsilon_end`: 最终探索率
- `decay_epochs`: 衰减epochs数

**输出**:
- `epsilon`: 当前探索率

**公式**:
```
epsilon = max(epsilon_end, epsilon_start - epoch × (epsilon_start - epsilon_end) / decay_epochs)
```

**代码**:
```python
epsilon = max(0.05, 0.5 - epoch * 0.01)
```

---

## 六、Phase 5: 评估与保存

### Step 5.1: 评估流程

**输入**:
- `test_queries`: 测试查询列表
- 训练好的模型

**输出**:
- `metrics`: 评估指标 (MRR, Hits@1/3/10)

**评估公式**:
```
MRR = 1/|Q| × Σ 1/rank_q

Hits@K = 1/|Q| × Σ 1(rank_q ≤ K)
```

**详细步骤**:
```
for query in test_queries:
    head, relation, tail = query

    # Step 5.1.1: 选择规则 (不探索)
    selected_rules = rule_selector(head, relation, epsilon=0.0)

    # Step 5.1.2: 对所有候选实体打分
    scores = []
    for candidate in range(num_entities):
        score = run_episode(head, relation, candidate, deterministic=True)
        scores.append(score)

    # Step 5.1.3: 过滤已知三元组
    for known_tail in hr2ooo[(head, relation)]:
        if known_tail != tail:
            scores[known_tail] = -inf

    # Step 5.1.4: 计算排名
    sorted_indices = argsort(scores, descending=True)
    rank = (sorted_indices == tail).nonzero() + 1
    ranks.append(rank)

# Step 5.1.5: 计算指标
MRR = mean(1.0 / ranks)
Hits@1 = mean(ranks <= 1)
Hits@3 = mean(ranks <= 3)
Hits@10 = mean(ranks <= 10)
```

---

### Step 5.2: 保存检查点

**输入**:
- 所有模型组件
- 训练状态

**输出**:
- 检查点文件

**代码**:
```python
torch.save({
    # 模型参数
    'state_encoder': state_encoder.state_dict(),
    'rule_selector': rule_selector.state_dict(),
    'path_finder': path_finder.state_dict(),

    # 优化器状态
    'policy_optimizer': policy_optimizer.state_dict(),
    'value_optimizer': value_optimizer.state_dict(),
    'rule_selector_optimizer': rule_selector_optimizer.state_dict(),

    # UCB统计
    'rule_counts': dict(rule_selector.rule_counts),
    'rule_rewards': dict(rule_selector.rule_rewards),
    'total_selections': rule_selector.total_selections,

    # 训练状态
    'epoch': epoch,
    'epsilon': epsilon,
    'best_mrr': best_mrr,

    # 配置
    'args': args
}, checkpoint_path)
```

---

## 七、完整算法伪代码

```python
# ═══════════════════════════════════════════════════════════════════
#                     RulE-RL 完整训练算法
# ═══════════════════════════════════════════════════════════════════

def train_rule_rl(args):
    """RulE-RL主训练函数"""

    # ═══════════ Phase 1: 加载预训练模型 ═══════════
    graph = load_knowledge_graph(args.data_path)
    rule_model = load_pretrained_rule(args.checkpoint_path)
    freeze_parameters(rule_model)

    # ═══════════ Phase 2: 初始化RL组件 ═══════════
    # 动态读取预训练维度 (UMLS: entity=4000, rel=2000, rule=100)
    entity_dim = rule_model.entity_embedding.weight.shape[1]
    rel_dim = rule_model.relation_embedding.weight.shape[1]
    rule_dim = rule_model.rule_emb.shape[1]

    state_encoder = StateEncoder(
        entity_dim=entity_dim,      # 4000 (从预训练读取)
        rel_dim=rel_dim,            # 2000 (从预训练读取)
        rule_dim=rule_dim,          # 100 (从预训练读取)
        history_dim=128             # RL自定义
    )
    rule_selector = RuleSelectorAgent(
        query_dim=entity_dim + rel_dim,  # 6000
        rule_dim=rule_dim                # 100
    )
    path_finder = PathFinderAgent(state_dim=128, action_dim=num_relations)
    env = KGReasoningEnv(graph, rule_model, state_encoder, max_steps=5)
    reward_calculator = RewardCalculator(rule_model, alpha=0.1, beta=0.05)

    # 优化器
    policy_optimizer = Adam(path_finder.policy_net.parameters(), lr=1e-3)
    value_optimizer = Adam(path_finder.value_net.parameters(), lr=1e-3)
    rule_selector_optimizer = Adam(rule_selector.parameters(), lr=1e-4)

    # ═══════════ Phase 3-4: 训练循环 ═══════════
    for epoch in range(args.num_epochs):
        epsilon = max(0.05, 0.5 - epoch * 0.01)  # 探索率衰减

        for query in train_queries:
            head, relation, tail = query

            # ─────── Step 3.2: 高层Agent选择规则 ───────
            entity_emb = rule_model.entity_embedding[head]
            rel_emb = rule_model.relation_embedding[relation]
            selected_rules, selection_probs = rule_selector(
                entity_emb, rel_emb, rule_model.rule_emb,
                epsilon=epsilon, top_k=5
            )

            # ─────── Step 3.3: 环境初始化 ───────
            state = env.reset(query)
            episode_data = {'states': [], 'actions': [], 'log_probs': [], 'rewards': []}
            done = False

            # ─────── Step 3.4: Episode循环 ───────
            while not done:
                # 获取动作掩码
                action_mask = env.get_action_mask(selected_rules)

                # 策略网络选择动作
                action, log_prob = path_finder.select_action(state, action_mask)

                # 环境执行动作
                next_state, reward, done, info = env.step(action, selected_rules)

                # 记录数据
                episode_data['states'].append(state)
                episode_data['actions'].append(action)
                episode_data['log_probs'].append(log_prob)
                episode_data['rewards'].append(reward)

                state = next_state

            # ─────── Step 3.5: 计算Episode奖励 ───────
            total_reward, breakdown = reward_calculator.compute_reward(
                env.trajectory, tail
            )
            episode_data['rewards'][-1] = total_reward  # 最后一步给总奖励

            # ─────── Step 4.1-4.2: 计算回报和优势 ───────
            returns = compute_returns(episode_data['rewards'], gamma=0.99)
            states = torch.stack(episode_data['states'])
            values = path_finder.value_net(states).squeeze()
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ─────── Step 4.3: 更新PathFinder ───────
            log_probs = torch.stack(episode_data['log_probs'])
            policy_loss = -(log_probs * advantages).mean()
            value_loss = F.mse_loss(values, returns)

            policy_optimizer.zero_grad()
            policy_loss.backward()
            clip_grad_norm_(path_finder.policy_net.parameters(), 1.0)
            policy_optimizer.step()

            value_optimizer.zero_grad()
            value_loss.backward()
            clip_grad_norm_(path_finder.value_net.parameters(), 1.0)
            value_optimizer.step()

            # ─────── Step 4.4: 更新RuleSelector ───────
            rule_selector_loss = -torch.sum(
                torch.log(selection_probs + 1e-10)
            ) * total_reward

            rule_selector_optimizer.zero_grad()
            rule_selector_loss.backward()
            rule_selector_optimizer.step()

            rule_selector.update_ucb_statistics(selected_rules, total_reward)

        # ═══════════ Phase 5: 评估与保存 ═══════════
        if (epoch + 1) % args.eval_interval == 0:
            metrics = evaluate(valid_queries)
            print(f"Epoch {epoch+1}: MRR={metrics['mrr']:.4f}")

            if metrics['mrr'] > best_mrr:
                best_mrr = metrics['mrr']
                save_checkpoint(f"{args.save_path}/best_checkpoint.pt")

    return best_mrr
```

---

## 八、技术栈总结

### 8.1 使用的深度学习技术

| 技术 | 组件 | 作用 |
|------|------|------|
| **MLP** | 所有Agent | 特征变换、得分计算 |
| **LSTM** | StateEncoder | 规则序列编码 |
| **GRU** | StateEncoder | 路径历史编码 |
| **Embedding** | RulE模型 | 实体/关系/规则表示 |
| **Softmax** | PathFinder | 动作概率分布 |
| **Adam** | 所有优化器 | 参数优化 |

### 8.2 使用的强化学习技术

| 技术 | 组件 | 作用 |
|------|------|------|
| **REINFORCE** | PathFinder | 策略梯度更新 |
| **Baseline (Value Network)** | PathFinder | 减小方差 |
| **Actor-Critic** | PathFinder | 策略+价值网络 |
| **UCB** | RuleSelector | 探索-利用平衡 |
| **ε-greedy** | RuleSelector | 随机探索 |
| **Action Masking** | KGReasoningEnv | 约束无效动作 |
| **Reward Shaping** | RewardCalculator | 稠密奖励信号 |

### 8.3 使用的数据结构

| 数据结构 | 用途 | 时间复杂度 |
|----------|------|-----------|
| **邻接表** | 图存储、邻居查询 | O(1) |
| **字典** | 规则映射、UCB统计 | O(1) |
| **COO稀疏矩阵** | 规则grounding | O(edges) |
| **反向索引** | 过滤、存在性检查 | O(1) |

### 8.4 关键公式汇总

| 公式名 | 数学表达式 |
|--------|-----------|
| **UCB得分** | $UCB(i) = \hat{Q}(i) + c \sqrt{\frac{2\ln N}{n_i}}$ |
| **折扣回报** | $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$ |
| **优势函数** | $A_t = G_t - V(s_t)$ |
| **策略梯度** | $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)]$ |
| **总奖励** | $R = R_{final\_bin} + \alpha \big(R_{rule} + (1 - R_{final\_bin}) R_{closer\_norm}\big)$ |
| **规则置信度** | $conf(r) = \frac{\gamma_{rule} - \|body\_sum + rule\_emb - head\_emb\|_2}{\gamma_{rule}}$ |

---

## 九、训练日志示例

```
═══════════════════════════════════════════════════════════════════
                    RulE-RL Training Log
═══════════════════════════════════════════════════════════════════

[Config]
  Dataset: UMLS
  Num Entities: 135
  Num Relations: 46
  Num Rules: 18,400
  Hidden Dim: 200
  Top-K Rules: 5
  Max Steps: 5
  Gamma: 0.99
  Epsilon: 0.5 → 0.05

═══════════════════════════════════════════════════════════════════
Epoch 1/100 | Epsilon: 0.500
───────────────────────────────────────────────────────────────────
Query 100/5216: (aspirin, treats, headache)
  Selected Rules: [234, 567, 1023, 89, 456]
  Path: aspirin --treats--> pain --relieves--> headache ✓
  Reward Breakdown:
    - Final: 1.0
    - Rule Consistency: 0.085
    - Getting Closer: 0.052
    - Total: 1.137
  Policy Loss: 0.234, Value Loss: 0.156

Query 200/5216: Avg Reward=0.524, Avg Length=3.2

Epoch 1 Summary:
  Avg Reward: 0.412
  Avg Length: 3.5
  Policy Loss: 0.456
  Success Rate: 35.2%

═══════════════════════════════════════════════════════════════════
Epoch 5/100 | Epsilon: 0.450
───────────────────────────────────────────────────────────────────
Validation Results:
  MRR: 0.456
  Hits@1: 0.312
  Hits@3: 0.523
  Hits@10: 0.672

Checkpoint saved to: ../outputs/rule_rl/checkpoint_epoch_5.pt

═══════════════════════════════════════════════════════════════════
Epoch 50/100 | Epsilon: 0.050
───────────────────────────────────────────────────────────────────
Epoch 50 Summary:
  Avg Reward: 0.823
  Avg Length: 2.1
  Success Rate: 72.3%

Validation Results:
  MRR: 0.789
  Hits@1: 0.698
  Hits@3: 0.845
  Hits@10: 0.923

New Best MRR! Saving best model...

═══════════════════════════════════════════════════════════════════
Training Complete!
───────────────────────────────────────────────────────────────────
Total Training Time: 4.3 hours
Best Validation MRR: 0.912

Final Test Results:
  MRR: 0.912
  MR: 2.34
  Hits@1: 0.834
  Hits@3: 0.921
  Hits@10: 0.967

Model saved to: ../outputs/rule_rl/final_model.pt
═══════════════════════════════════════════════════════════════════
```

---

**文档版本**: v1.2
**更新日期**: 2024年11月22日
**更新说明**:
- v1.2: 移除探索多样性(R_diversity)和长度惩罚(P_length)奖励组件,简化奖励函数
- v1.1: 添加完整的预训练参数分析和RulE-RL参数对照表 (基于UMLS数据集)
- v1.0: 初始版本，完整训练流程

**作者**: RulE-RL项目组
