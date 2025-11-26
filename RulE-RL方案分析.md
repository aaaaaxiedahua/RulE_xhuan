# RulE-RL 方案分析文档

本文档分析RulE-RL强化学习创新方案的流程、需要修改的原模型部分、以及新模型的模块组成。

---

## 一、整体流程概述

### 1.1 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                    RulE-RL 训练流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. 加载预训练RulE模型                                        │
│     ↓                                                        │
│  2. 冻结RulE模型参数（entity_embedding, relation_embedding,   │
│     rule_emb）                                               │
│     ↓                                                        │
│  3. 初始化RL组件                                              │
│     - 高层Agent (Rule Selector)                              │
│     - 低层Agent (Path Finder)                                │
│     - 环境 (KGReasoningEnv)                                  │
│     - 奖励计算器 (RewardCalculator)                          │
│     ↓                                                        │
│  4. Episode循环训练                                           │
│     a. 输入查询 (h, r, ?)                                    │
│     b. 高层Agent选择Top-K规则                                 │
│     c. 低层Agent在KG中寻找路径                                │
│     d. 计算奖励并更新两个Agent                                │
│     ↓                                                        │
│  5. 保存检查点                                                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 推理流程

```
查询 (h, r, ?)
    ↓
高层Agent选择Top-K规则 (ε=0, 不探索)
    ↓
对每个候选实体:
    ↓
低层Agent执行路径查找 (deterministic=True)
    ↓
计算路径得分
    ↓
排序得到最终预测
```

### 1.3 与原RulE流程对比

| 阶段 | 原RulE | RulE-RL |
|------|--------|---------|
| **预训练** | 训练entity/relation/rule嵌入 | 复用RulE预训练结果 |
| **规则选择** | 使用所有相关规则 | RL动态选择Top-K |
| **路径探索** | BFS枚举所有路径 | RL策略指导探索 |
| **评分** | MLP聚合规则得分 | Episode累积奖励 |

---

## 二、需要替换/修改的原模型部分

### 2.1 需要保留（冻结）的部分

以下是从原RulE模型中复用的组件：

| 组件 | 文件位置 | 用途 |
|------|----------|------|
| `entity_embedding` | src/model.py:20 | 提供实体表示 |
| `relation_embedding` | src/model.py:21 | 提供关系表示 |
| `rule_emb` | src/model.py:22 | 提供规则表示 |
| `relation2rules` | src/model.py | 关系到规则的映射 |

### 2.2 需要替换的核心逻辑

#### 2.2.1 规则应用逻辑（最重要的替换）

**原代码位置**: `src/model.py:337-409` (`forward` 方法)

**原逻辑**:
```python
# 原RulE：使用所有规则
for rule in self.relation2rules[query_r]:
    grounding_count = graph.grounding(h, rule)
    # 所有规则一视同仁处理
```

**替换为**:
```python
# RulE-RL：RL选择规则
selected_rules = rule_selector(query_entity, query_rel, rule_embeddings, top_k=5)
# 只处理选中的规则
for rule_id in selected_rules:
    # 路径查找由PathFinder Agent执行
```

#### 2.2.2 路径探索逻辑

**原代码位置**: `src/data.py:410-421` (`grounding` 方法)

**原逻辑**:
```python
def grounding(self, h, r, rule_body, edges_to_remove):
    # BFS枚举所有可能路径
    for rel in rule_body:
        h = self.propagate(h, rel, ...)
    return grounding_count
```

**替换为**:
```python
# RL引导的路径查找
while not done:
    action = path_finder.select_action(state, action_mask)
    next_state, reward, done = env.step(action)
```

#### 2.2.3 评分机制

**原代码位置**: `src/model.py:380-390`

**原逻辑**:
```python
# 使用FuncToNodeSum和MLP聚合
aggregated = FuncToNodeSum(grounding_counts, rule_features)
score = self.score_model(aggregated)
```

**替换为**:
```python
# 使用Episode累积奖励作为得分
episode_score = sum(rewards)
```

### 2.3 需要修改的训练流程

**原代码位置**: `src/trainer.py`

| 原组件 | 替换/修改 |
|--------|----------|
| `GroundTrainer` (src/trainer.py:369-760) | 替换为 `RuleRLTrainer` |
| `train_step()` 的交叉熵损失 | 替换为 REINFORCE 策略梯度 |
| `evaluate()` / `evaluate_t()` | 修改为RL based评估 |

### 2.4 可以删除的部分

以下原RulE组件在RulE-RL中不再需要：

| 组件 | 位置 | 原因 |
|------|------|------|
| `mlp_feature` | src/model.py:25 | 被PathFinder的Policy网络替代 |
| `score_model` | src/model.py:32-37 | 被RL奖励机制替代 |
| `FuncToNodeSum` | src/layers.py:60-71 | 被RL策略网络替代 |
| `bias` | src/model.py:56 | 不再需要 |

---

## 三、新模型模块组成

### 3.1 模块架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     RulE-RL 新增模块                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  1. RuleSelectorAgent (高层Agent)                    │    │
│  │     - QueryEncoder                                   │    │
│  │     - RuleQueryMatcher                              │    │
│  │     - UCB统计模块                                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  2. PathFinderAgent (低层Agent)                      │    │
│  │     - PolicyNetwork                                  │    │
│  │     - ValueNetwork (baseline)                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  3. StateEncoder                                     │    │
│  │     - EntityEncoder                                  │    │
│  │     - RelationEncoder                                │    │
│  │     - RuleEncoder (LSTM)                            │    │
│  │     - HistoryEncoder (GRU)                          │    │
│  │     - StateFusion                                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  4. KGReasoningEnv (环境)                            │    │
│  │     - reset() / step()                              │    │
│  │     - get_action_mask()                             │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  5. RewardCalculator (奖励计算)                      │    │
│  │     - final_reward                                   │    │
│  │     - rule_consistency_reward                        │    │
│  │     - getting_closer_reward                          │    │
│  │     - diversity_reward                               │    │
│  │     - penalties (dead_end, loop, length)            │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  6. RuleRLTrainer (训练器)                           │    │
│  │     - train_episode()                                │    │
│  │     - train()                                        │    │
│  │     - evaluate()                                     │    │
│  │     - save_checkpoint()                              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 各模块详细说明

#### 3.2.1 RuleSelectorAgent（高层Agent）

**功能**: 根据查询动态选择最相关的Top-K条规则

**核心组件**:
- `query_encoder`: MLP，将(h, r)映射到上下文向量
- `rule_query_matcher`: MLP，计算规则-查询匹配得分
- UCB统计: 记录每条规则的选择次数和累积奖励

**算法**: Contextual Bandit + UCB

**输入/输出**:
```python
输入: query_entity_emb, query_rel_emb, rule_embeddings
输出: selected_rules (Top-K规则ID), selection_probs
```

#### 3.2.2 PathFinderAgent（低层Agent）

**功能**: 在知识图谱中寻找从头实体到目标的路径

**核心组件**:
- `policy_net`: MLP，状态→动作概率
- `value_net`: MLP，状态→状态价值（用于baseline）

**算法**: REINFORCE with baseline

**输入/输出**:
```python
输入: state, action_mask
输出: action (关系ID), log_prob
```

#### 3.2.3 StateEncoder

**功能**: 将当前状态编码为向量表示

**核心组件**:
- `entity_encoder`: 线性层，编码当前实体
- `relation_encoder`: 线性层，编码查询关系
- `rule_encoder`: LSTM，编码激活规则的上下文
- `history_encoder`: GRU，编码历史路径
- `state_fusion`: MLP，融合所有编码

**状态表示**:
```python
state = [
    h_entity,      # 当前实体嵌入
    h_query_rel,   # 查询关系嵌入
    h_rule,        # 规则上下文
    h_history      # 历史路径编码
]
```

#### 3.2.4 KGReasoningEnv（环境）

**功能**: 模拟知识图谱推理环境

**核心方法**:
- `reset(query)`: 重置环境，返回初始状态
- `step(action, selected_rules)`: 执行动作，返回(next_state, reward, done, info)
- `get_action_mask(selected_rules)`: 获取有效动作掩码

**状态转移**:
```
当前实体 --[选择关系]--> 下一个实体
```

#### 3.2.5 RewardCalculator（奖励计算器）

**功能**: 计算轨迹的总奖励

**奖励组成**:

| 奖励类型 | 权重 | 说明 |
|----------|------|------|
| R_final_bin | 1.0 | 命中目标=1，否则=0 |
| R_rule | α=0.1 | 路径与规则的一致性 |
| R_closer_norm | α=0.1 | 仅在失败时启用的归一化接近奖励 |

**奖励公式**:
```python
total_reward = R_final_bin + α × (R_rule + (1 - R_final_bin) × R_closer_norm)
```

#### 3.2.6 RuleRLTrainer（训练器）

**功能**: 完整的RulE-RL训练流程

**核心方法**:
- `train_episode(query, epsilon)`: 训练单个episode
- `train(train_queries, num_epochs)`: 完整训练循环
- `evaluate(test_queries)`: 评估模型
- `save_checkpoint(path)`: 保存检查点

---

## 四、代码文件结构建议

```
src/
├── main.py                 # 主入口（需修改）
├── model.py                # 原RulE模型（保留，用于加载预训练）
├── data.py                 # 数据处理（保留）
├── trainer.py              # 原训练器（保留）
├── layers.py               # 原网络层（保留）
├── utils.py                # 工具函数（保留）
│
├── rl/                     # 新增：RL模块目录
│   ├── __init__.py
│   ├── agents.py           # RuleSelectorAgent, PathFinderAgent
│   ├── env.py              # KGReasoningEnv
│   ├── reward.py           # RewardCalculator
│   ├── encoder.py          # StateEncoder
│   └── trainer.py          # RuleRLTrainer
│
└── main_rl.py              # 新增：RulE-RL主入口
```

---

## 五、关键技术点

### 5.1 层次化RL设计

**为什么需要层次化**:
- 规则选择和路径查找是两个不同粒度的决策
- 分离可以降低动作空间复杂度
- 高层Agent学习"选择什么规则"，低层Agent学习"怎么走"

**协作机制**:
```
高层Agent的奖励 = 低层Agent的最终成功率
低层Agent的状态 = f(当前位置, 高层Agent选择的规则)
```

### 5.2 奖励塑形

**为什么需要奖励塑形**:
- 稀疏奖励（只有最终奖励）导致学习困难
- 中间奖励提供学习信号引导

**关键设计**:
- `rule_consistency`: 利用RulE的规则嵌入计算
- `getting_closer`: 利用RotatE的实体嵌入计算
- 结合了符号推理和神经推理的优势

### 5.3 动作掩码机制

**目的**: 确保Agent只选择有效的动作

**实现**:
```python
# 有效动作 = 当前实体的出边 ∩ 规则体中的关系
valid_actions = set(outgoing_rels) & set(rule_body_rels)
```

### 5.4 UCB探索策略

**为什么使用UCB**:
- ε-greedy过于简单，可能探索不充分
- UCB在探索和利用之间有理论保证

**UCB公式**:
```python
score = Q̂(rule) + sqrt(2 * log(N_total) / N_rule)
```

---

## 六、与原模型的数据流对比

### 6.1 原RulE数据流

```
输入: (h, r, ?)
    ↓
获取所有相关规则: relation2rules[r]
    ↓
对每条规则执行grounding (BFS)
    ↓
聚合规则特征: FuncToNodeSum
    ↓
MLP评分: score_model
    ↓
输出: 每个候选实体的得分
```

### 6.2 RulE-RL数据流

```
输入: (h, r, ?)
    ↓
高层Agent选择Top-K规则
    ↓
初始化环境: env.reset()
    ↓
循环:
  - 编码状态: StateEncoder
  - 低层Agent选择动作
  - 环境执行: env.step()
  - 计算奖励: RewardCalculator
    ↓
输出: Episode累积奖励作为得分
```

---

## 七、超参数配置

### 7.1 新增超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rl_lr` | 1e-3 | PathFinder学习率 |
| `rule_selector_lr` | 1e-4 | RuleSelector学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `max_steps` | 5 | Episode最大步数 |
| `top_k_rules` | 5 | 选择的规则数量 |
| `epsilon_start` | 0.5 | 初始探索率 |
| `epsilon_end` | 0.05 | 最终探索率 |

### 7.2 配置文件示例

```json
{
    "data_path": "../data/umls",
    "rule_checkpoint": "../outputs/rule/checkpoint",
    "hidden_dim": 200,

    "rl_lr": 0.001,
    "rule_selector_lr": 0.0001,
    "gamma": 0.99,
    "max_steps": 5,
    "top_k_rules": 5,

    "num_epochs": 100,
    "log_interval": 100,
    "eval_interval": 5,
    "save_interval": 10,

    "save_path": "../outputs/rule_rl"
}
```

---

## 八、实施建议

### 8.1 实施优先级

1. **Phase 1**: 实现基础框架
   - StateEncoder
   - KGReasoningEnv
   - RewardCalculator（简化版，只有final_reward）

2. **Phase 2**: 实现RL Agents
   - PathFinderAgent + REINFORCE
   - RuleSelectorAgent（先用随机选择）

3. **Phase 3**: 完善训练流程
   - RuleRLTrainer
   - 完整的RewardCalculator
   - UCB策略

4. **Phase 4**: 优化和扩展
   - 课程学习
   - 超参数调优
   - 消融实验

### 8.2 可能的挑战

| 挑战 | 解决方案 |
|------|----------|
| RL训练不稳定 | 使用baseline、梯度裁剪、学习率调度 |
| 探索效率低 | UCB策略、课程学习 |
| 评估速度慢 | 批量评估、早停机制 |
| 奖励稀疏 | 奖励塑形、中间奖励 |

---

## 九、预期效果

### 9.1 性能提升

| 数据集 | 原RulE MRR | 预期RulE-RL MRR | 提升 |
|--------|------------|-----------------|------|
| UMLS | 0.867 | 0.912 | +5.2% |
| Kinship | 0.736 | 0.785 | +6.7% |
| FB15k-237 | 0.362 | 0.390 | +7.7% |

### 9.2 效率提升

- 推理速度: 2.0x加速
- 规则使用率: 35% (vs 100%)
- 路径枚举数: 12% (vs 100%)

---

**文档版本**: v1.0
**创建日期**: 2024年11月
**作者**: RulE-RL项目分析
