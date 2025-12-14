# RulE-SSRL: 融合自监督强化学习的知识图谱推理新模型方案

## 一、概述

本文档提出一种将**自监督强化学习(SSRL)**与**RulE神经符号推理框架**深度融合的创新方案。该方案结合了RulE的规则嵌入与图谱嵌入联合学习能力，以及SSRL在大动作空间下的高效路径探索能力，旨在构建一个更强大的知识图谱推理系统。

### 1.1 背景与动机

| 方法 | 优势 | 局限性 |
|------|------|--------|
| **RulE** | 神经符号融合、规则可解释性强、支持复杂规则推理 | 规则grounding计算开销大、依赖预挖掘规则质量 |
| **SSRL** | 路径探索能力强、处理大动作空间、可解释推理路径 | 稀疏奖励问题、收敛速度慢 |

**融合动机**：将SSRL的路径探索与预热能力用于增强RulE的规则grounding过程，同时利用RulE的规则嵌入为SSRL提供更好的先验知识引导。

### 1.2 核心创新点

本方案通过**规则引导的策略网络**实现RulE与SSRL的深度融合，核心创新在于规则嵌入的使用方式：

| 维度 | 传统方法 | RulE-SSRL融合方法 | 创新点 |
|------|---------|------------------|--------|
| **嵌入学习** | RulE: 独立预训练嵌入<br/>SSRL: 独立预训练嵌入 | **联合训练**：同时优化 entity_emb, relation_emb, rule_emb 和 policy_network<br/>损失函数: L = L_KGE + L_Rule + L_Policy | ✓ 嵌入与策略协同优化<br/>✓ 规则嵌入直接服务于路径探索 |
| **规则使用** | RulE: Grounding机械传播<br/>SSRL: 无规则 | **规则注意力机制**：策略网络动态计算应该参考哪条规则<br/>• 规则嵌入 → 注意力权重<br/>• 规则建议的动作获得得分加成 | ✓ 规则作为"软建议"而非硬约束<br/>✓ 神经网络学习何时采纳规则 |
| **动作选择** | RulE: 按规则传播（确定性）<br/>SSRL: 策略网络（无结构先验） | **规则引导的策略**：<br/>action_score = base_score + rule_bonus<br/>• base_score: 策略网络学习<br/>• rule_bonus: 符合规则则加分 | ✓ 结合灵活性和结构性<br/>✓ 策略可以突破规则限制 |
| **训练方式** | RulE: 预训练→Grounding<br/>SSRL: BFS标签→SL→RL | **分阶段训练**：<br/>• Phase 1: 预训练嵌入<br/>• Phase 1.5: 预计算规则质量<br/>• Phase 2: 策略网络训练（规则质量加权） | ✓ 简化训练流程<br/>✓ 规则嵌入质量指导策略学习 |
| **推理方式** | RulE: Grounding传播<br/>SSRL: 多次采样 | **策略网络多次采样**：<br/>• K次路径探索<br/>• 规则嵌入提供探索指引<br/>• 聚合K条路径结果 | ✓ 去掉Grounding模块<br/>✓ 统一的推理接口 |

**关键优势**：
- **不是叠加**：不是"先RulE预训练，再SSRL训练"，而是分阶段优化
- **规则嵌入有用武之地**：直接在策略网络中作为"建议者"参与决策
- **规则质量加权**：利用预训练的 `rules_weight_emb` 区分高/低质量规则，高质量规则获得更大引导权重
- **架构简洁**：三阶段训练（预训练→策略网络训练→推理）

---

## 二、核心架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RulE-SSRL 融合架构（简化版）                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                  Phase 1: 预训练（嵌入学习）                       │   │
│  │                                                                   │   │
│  │  ┌─────────────────────────────────────────────────────────────┐ │   │
│  │  │           嵌入学习 (Entity, Relation, Rule)                  │ │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │ │   │
│  │  │  │ KGE (RotatE) │  │ Rule嵌入学习  │  │  嵌入共享     │      │ │   │
│  │  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │ │   │
│  │  │         └──────────────────┴──────────────────┘              │ │   │
│  │  │                           ▼                                  │ │   │
│  │  │              entity_emb, relation_emb, rule_emb              │ │   │
│  │  │                                                              │ │   │
│  │  │  训练: L_total = L_KGE + L_Rule                             │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                  │                                      │
│                                  ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │        Phase 1.5: 预计算规则质量权重 ⭐ (新增)                    │   │
│  │                                                                   │   │
│  │  调用: model.eval_compute_rule_weight(device)                    │   │
│  │                                                                   │   │
│  │  计算: rules_weight_emb [num_rules, hidden_dim]                  │   │
│  │        rule_quality = norm(rules_weight_emb[rule_id])            │   │
│  │                                                                   │   │
│  │  含义: 范数越大 → 规则体与规则头匹配越好 → 规则质量越高            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                  │                                      │
│                                  ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              Phase 2: 策略网络训练（规则质量加权）                 │   │
│  │                                                                   │   │
│  │  Step 1: 冻结预训练嵌入                                           │   │
│  │  ├─ entity_emb.requires_grad_(False)                             │   │
│  │  ├─ relation_emb.requires_grad_(False)                           │   │
│  │  └─ rule_emb.requires_grad_(False)                               │   │
│  │                                                                   │   │
│  │  Step 2: 策略网络训练（使用规则质量加权）⭐                        │   │
│  │  ┌─────────────────────────────────────────────────────────────┐ │   │
│  │  │  输入: state = [current_entity, path_history, query_rel]   │ │   │
│  │  │                                                             │ │   │
│  │  │  规则打分 (改进):                                            │ │   │
│  │  │  ├─ 获取规则质量: quality = norm(rules_weight_emb[rule_id]) │ │   │
│  │  │  ├─ 加权匹配: weighted_match += quality × match_mask        │ │   │
│  │  │  └─ rule_bonus = weighted_match / total_quality             │ │   │
│  │  │                                                             │ │   │
│  │  │  规则监督 (改进):                                            │ │   │
│  │  │  ├─ 高质量规则: 监督信号权重大                               │ │   │
│  │  │  └─ 低质量规则: 监督信号权重小                               │ │   │
│  │  │                                                             │ │   │
│  │  │  动作得分: action_score = base_score + rule_bonus           │ │   │
│  │  │  训练损失: L_Policy = KL_div(logits, target_dist)          │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                  │                                      │
│                                  ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Phase 3: 推理阶段                              │   │
│  │                                                                   │   │
│  │  对查询 (h, r_q, ?)：                                             │   │
│  │                                                                   │   │
│  │  Step 1: 多次路径采样                                             │   │
│  │  ├─ 用策略网络从h出发探索K条路径（如K=10）                        │   │
│  │  ├─ 规则嵌入在探索中通过质量加权提供建议                          │   │
│  │  └─ 每条路径记录终点实体                                         │   │
│  │                                                                   │   │
│  │  Step 2: 聚合路径结果                                             │   │
│  │  ├─ 统计每个实体被访问的次数                                      │   │
│  │  ├─ 方法A: 简单投票 score = count/K                              │   │
│  │  ├─ 方法B: 加权投票（考虑路径质量）                               │   │
│  │  └─ 方法C: 集成KGE score（可选）                                 │   │
│  │                                                                   │   │
│  │  Step 3: 输出答案排序                                             │   │
│  │  └─ 按score排序，返回top-K候选实体                               │   │
│  │                                                                   │   │
│  │  优势：                                                           │   │
│  │  • 统一接口：只用策略网络，无需Grounding                         │   │
│  │  • 可扩展：调整K平衡精度和速度                                    │   │
│  │  • 灵活：可结合多种聚合策略                                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

核心改进：
✓ 去掉独立的Grounding阶段
✓ 嵌入训练和策略训练分阶段进行（预训练→策略训练）
✓ 规则嵌入直接在策略网络中使用（提供动作建议）
✓ ⭐ 规则质量加权：利用 rules_weight_emb 区分高/低质量规则
✓ 推理阶段明确：策略网络多次采样+聚合
```

### 2.2 模块详解

#### 2.2.1 规则引导的策略网络 (Rule-Guided Policy Network)

**核心创新**：策略网络通过**规则注意力机制**，让规则嵌入直接参与动作选择，规则作为"软建议"指导路径探索。

**关键设计思想**：
- 传统RL策略：独立学习，缺乏结构先验
- 传统RulE：机械应用规则，缺乏灵活性
- **新方法**：策略网络学习"何时采纳哪条规则的建议"

```python
class RuleGuidedPolicyNetwork(nn.Module):
    def __init__(self, entity_dim, relation_dim, rule_dim, hidden_dim):
        super().__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.rule_dim = rule_dim
        self.hidden_dim = hidden_dim

        # LSTM编码路径历史
        self.lstm = nn.LSTM(
            input_size=relation_dim + entity_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # 规则注意力网络（核心创新1）
        self.rule_attention = nn.Sequential(
            nn.Linear(hidden_dim + relation_dim + rule_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 动作基础评分网络
        self.action_base_scorer = nn.Linear(
            hidden_dim + relation_dim + entity_dim, 1
        )

    def compute_rule_attention_weights(self, state, query_relation, applicable_rules):
        """
        计算应该参考哪条规则（核心创新）

        输入：
        - state: 当前状态（LSTM输出）
        - query_relation: 查询关系嵌入
        - applicable_rules: [(rule_id, rule_emb, rule_body), ...]

        输出：
        - rule_weights: [num_rules] 每条规则的注意力权重

        思想：神经网络学习在当前状态下应该参考哪条规则
        """
        rule_scores = []

        for rule_id, rule_emb, rule_body in applicable_rules:
            # 融合当前状态、查询关系、规则嵌入
            combined = torch.cat([state, query_relation, rule_emb], dim=-1)

            # 神经网络打分：这条规则对当前状态有多重要？
            score = self.rule_attention(combined)
            rule_scores.append(score)

        if len(rule_scores) == 0:
            return torch.tensor([])

        # Softmax归一化为注意力权重
        rule_scores = torch.cat(rule_scores, dim=0)
        rule_weights = F.softmax(rule_scores, dim=0)

        return rule_weights

    def get_rule_suggestions(self, applicable_rules, path_history):
        """
        获取规则建议的下一步动作

        根据当前路径历史，判断每条规则建议走哪个关系
        """
        suggestions = []

        for rule_id, rule_emb, rule_body in applicable_rules:
            # 当前已经走了几步
            step = len(path_history)

            # 规则建议：走rule_body[step]这个关系
            if step < len(rule_body):
                suggested_relation = rule_body[step]
            else:
                suggested_relation = None  # 规则已经走完

            suggestions.append(suggested_relation)

        return suggestions

    def forward(self, current_entity, query_relation, path_history,
                action_space, rule_model):
        """
        前向传播：规则引导的动作选择

        核心流程：
        1. LSTM编码路径历史
        2. 计算规则注意力权重（学习参考哪条规则）
        3. 获取规则建议的动作
        4. 为每个候选动作打分：基础分 + 规则加成
        5. 输出动作概率分布

        Args:
            current_entity: 当前实体
            query_relation: 查询关系
            path_history: 路径历史 [(r_0, e_1), (r_1, e_2), ...]
            action_space: 可选动作 [(r, e), ...]
            rule_model: 规则模型（提供规则嵌入和规则列表）

        Returns:
            action_probs: 动作概率分布 π(a|s, rules)
        """
        # Step 1: LSTM编码路径历史
        if len(path_history) > 0:
            path_emb = self.encode_path(path_history)
            h_t, _ = self.lstm(path_emb.unsqueeze(0))
            state = h_t.squeeze(0)
        else:
            state = torch.zeros(self.hidden_dim)

        # Step 2: 获取适用于查询关系的规则
        applicable_rules = rule_model.get_rules_for_relation(query_relation)
        # [(rule_id, rule_emb, rule_body), ...]

        # Step 3: 计算规则注意力权重（核心！）
        rule_weights = self.compute_rule_attention_weights(
            state,
            query_relation,
            applicable_rules
        )

        # Step 4: 获取规则建议的动作
        rule_suggestions = self.get_rule_suggestions(
            applicable_rules,
            path_history
        )

        # Step 5: 为每个候选动作打分
        action_scores = {}

        for relation, next_entity in action_space:
            # 5.1 基础分数（策略网络自己学的）
            action_feature = torch.cat([
                state,
                self.relation_emb[relation],
                self.entity_emb[next_entity]
            ])
            base_score = self.action_base_scorer(action_feature).item()

            # 5.2 规则加成（核心创新2）
            rule_bonus = 0.0

            for i, suggested_relation in enumerate(rule_suggestions):
                if relation == suggested_relation:
                    # 这个动作符合第i条规则的建议
                    # 加上该规则的注意力权重
                    rule_bonus += rule_weights[i].item()

            # 5.3 最终得分 = 基础分 + 规则加成
            action_scores[(relation, next_entity)] = base_score + rule_bonus

        # Step 6: Softmax归一化为概率分布
        scores_tensor = torch.tensor(list(action_scores.values()))
        action_probs = F.softmax(scores_tensor, dim=0)

        return action_probs, action_scores

    def encode_path(self, path_history):
        """编码路径历史"""
        path_embeddings = []
        for relation, entity in path_history:
            emb = torch.cat([
                self.relation_emb[relation],
                self.entity_emb[entity]
            ])
            path_embeddings.append(emb)
        return torch.stack(path_embeddings)
```

**关键创新点**：

1. **规则注意力机制**：
   ```
   不是固定地应用所有规则，而是动态学习：
   "在当前状态下，应该参考哪条规则？"

   当前状态(state)包含：
   • current_entity: 当前在图上的哪个实体
   • path_history: 已经走过的路径 [(r0,e1), (r1,e2), ...]
   • query_relation: 查询的目标关系

   LSTM将这些信息编码为向量 h_t（状态表示）

   规则注意力计算：
   rule_weight[i] = Attention(h_t, query_relation, rule_emb[i])

   含义：根据当前位置、已走路径和查询目标，
         动态决定每条规则的重要性

   例子：
   • 在路径开始（Tom_Brady），参考规则建议走plays_for
   • 在路径中间（Buccaneers），参考规则建议走head_coach
   • 如果偏离规则路径（Tampa），降低该规则的权重
   ```

2. **规则作为软建议**：
   ```
   action_score = base_score + Σ rule_weight[i] * match(action, rule[i])

   • 如果动作符合规则建议 → 得分增加
   • 增加多少取决于规则的注意力权重
   • 策略可以选择不采纳规则建议（保留灵活性）
   ```

3. **端到端可学习**：
   ```
   整个过程可微分，可以通过梯度下降训练：
   • 学习规则注意力权重（何时参考规则）
   • 学习动作基础评分（策略自己的判断）
   • 学习何时采纳/忽略规则（平衡结构与灵活性）
   ```

#### 2.2.2 混合奖励函数 (Hybrid Reward Function) - 规则引导奖励塑形

**重要区分：动作打分 vs 奖励函数**

在RulE-SSRL中，有两个容易混淆的"规则引导"机制：

| 维度 | 动作打分（2.2.1节） | 混合奖励函数（本节） |
|------|-------------------|-------------------|
| **公式** | `action_score = base_score + rule_bonus` | `R_final = R_term + λ·R_rule` |
| **作用** | 策略网络选择动作时的打分机制 | 强化学习训练时的反馈信号 |
| **位置** | 在策略网络的`forward()`中 | 在RL训练的`compute_returns()`中 |
| **使用阶段** | Phase 1联合训练 + Phase 2 RL微调 + Phase 3推理 | **仅Phase 2 RL微调** |
| **输出** | 动作概率分布 π(a\|s) | 标量奖励值（用于计算return） |
| **训练方式** | 梯度反向传播（可微分） | Policy Gradient（REINFORCE） |

**示例说明区别**：

```python
# ========== 动作打分（在策略网络forward中）==========
# 用途：选择下一步走哪个动作
def forward(self, current_entity, query_relation, path_history, action_space):
    # 对每个候选动作打分
    for relation, next_entity in action_space:
        # 神经网络学习的基础分数
        base_score = self.action_base_scorer(...)  # 例如：0.6

        # 规则建议的加成
        rule_bonus = 0.0
        if relation in rule_suggested_relations:
            rule_bonus = rule_weights[i]  # 例如：0.3

        # 最终得分 = 基础分 + 规则加成
        action_scores[action] = base_score + rule_bonus  # 0.6 + 0.3 = 0.9

    # Softmax转换为概率
    action_probs = softmax(action_scores)  # [0.1, 0.7, 0.2]
    return action_probs  # 用于选择动作

# ========== 混合奖励（在RL训练的回报计算中）==========
# 用途：评价已经执行的动作好不好（反馈信号）
def compute_reward(current_entity, target_entity, path, query_relation):
    # 终止奖励（到达目标）
    R_term = 1.0 if current_entity == target_entity else 0.0  # 0或1

    # 规则奖励（路径符合规则）
    R_rule = compute_rule_match(path, query_relation)  # 例如：0.5

    # 混合奖励
    R_final = R_term + 0.1 * R_rule  # 0 + 0.1×0.5 = 0.05

    return R_final  # 用于计算return，更新策略

# 关键区别：
# 1. action_score用于"前向"：选择动作（策略网络输出）
# 2. R_final用于"后向"：评价动作（RL反馈信号）
```

**使用时机**：

```
┌─────────────────────────────────────────────────────────────┐
│                    训练流程对比                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: 联合训练                                           │
│  ├─ 策略网络使用: action_score = base_score + rule_bonus  ✓│
│  ├─ 训练方式: 规则监督（类似监督学习）                       │
│  └─ 混合奖励函数: 不使用                                  ✗ │
│                                                             │
│  Phase 2: RL微调（可选）                                     │
│  ├─ 策略网络使用: action_score = base_score + rule_bonus  ✓│
│  ├─ 训练方式: Policy Gradient                               │
│  └─ 混合奖励函数: R_final = R_term + λ·R_rule            ✓ │
│                                                             │
│  Phase 3: 推理                                              │
│  ├─ 策略网络使用: action_score = base_score + rule_bonus  ✓│
│  └─ 混合奖励函数: 不使用                                  ✗ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**核心创新**：设计混合奖励函数缓解RL的稀疏奖励问题，鼓励代理探索符合逻辑规则的路径。

**奖励函数公式**：
```
R_final(s_t, a_t, s_{t+1}) = R_term(s_{t+1}) + λ · R_rule(s_t, a_t, s_{t+1})
```

其中：
- **R_term**: 终止奖励（稀疏），当且仅当到达目标实体时为1，否则为0
- **R_rule**: 规则奖励（密集），在每一步t到t+1，如果路径片段 (e_t, r_t, e_{t+1}) 支持与查询关系 r_q 相关的高置信度规则，奖励值与RulE计算的规则置信度 **w** 成正比
- **λ**: 平衡超参数，控制规则引导的强度

```python
class HybridRewardFunction(nn.Module):
    def __init__(self, rule_model, lambda_rule=0.1):
        super().__init__()
        self.rule_model = rule_model
        self.lambda_rule = lambda_rule  # 规则奖励权重

    def compute_reward(self, current_entity, target_entity, query_relation, path):
        """
        计算混合奖励: R_final = R_term + λ * R_rule

        Args:
            current_entity: 当前到达的实体 e_{t+1}
            target_entity: 目标实体
            query_relation: 查询关系 r_q
            path: 已走过的路径 [(e_0, r_0, e_1), (e_1, r_1, e_2), ...]
        Returns:
            reward: 混合奖励值
        """
        # 1. 终止奖励 (稀疏)
        R_term = 1.0 if current_entity == target_entity else 0.0

        # 2. 规则奖励 (密集)
        R_rule = self._compute_rule_reward(path, query_relation)

        # 3. 混合奖励
        R_final = R_term + self.lambda_rule * R_rule

        return R_final

    def _compute_rule_reward(self, path, query_relation):
        """
        计算规则奖励 - 核心创新

        核心思想：
        - 如果当前路径片段 (e_t, r_t, e_{t+1}) 是某条规则R的body的一部分
        - 且规则R的head是查询关系 r_q
        - 则奖励值与规则置信度 w_R 成正比

        实现：
        1. 获取所有与 r_q 相关的规则
        2. 对每条规则R，计算当前路径与规则body的匹配度
        3. 奖励 = Σ w_R * match_score(path, R.body)
        """
        # 获取适用规则
        applicable_rules = self.rule_model.get_rules_for_relation(query_relation)

        if len(applicable_rules) == 0:
            return 0.0

        total_rule_reward = 0.0

        for rule_id, (r_head, r_body) in applicable_rules:
            # 计算规则置信度 w_R (从RulE模型学习得到)
            w_R = self.rule_model.get_rule_confidence(rule_id, query_relation)

            # 计算路径与规则body的匹配分数
            match_score = self._compute_path_rule_match(path, r_body)

            # 规则奖励与置信度成正比
            total_rule_reward += w_R * match_score

        return total_rule_reward

    def _compute_path_rule_match(self, path, rule_body):
        """
        计算路径与规则body的匹配程度

        匹配逻辑：
        - 路径关系序列: [r_0, r_1, ..., r_t]
        - 规则body: [r_body_0, r_body_1, ..., r_body_m]
        - 如果路径是规则body的前缀，匹配分数更高

        例如：
        - path = [plays_for, head_coach]
        - rule_body = [plays_for, head_coach, located_in]
        - match_score = 2/3 = 0.67 (前两个关系匹配)
        """
        path_relations = [r for (e_from, r, e_to) in path]

        if len(path_relations) == 0 or len(rule_body) == 0:
            return 0.0

        # 计算最长公共前缀
        match_count = 0
        max_len = min(len(path_relations), len(rule_body))

        for i in range(max_len):
            if path_relations[i] == rule_body[i]:
                match_count += 1
            else:
                break  # 前缀匹配中断

        # 归一化匹配分数
        match_score = match_count / len(rule_body)

        return match_score
```

**奖励函数设计优势**：
1. **缓解稀疏奖励**：每步都有密集的规则奖励信号，不必等到终点才获得反馈
2. **逻辑引导探索**：符合规则的路径获得更高奖励，减少无效探索
3. **软约束机制**：规则奖励与置信度成正比，避免过度依赖低质量规则

---

## 三、训练流程详解

### 3.1 两阶段训练流程（简化版）

```
┌────────────────────────────────────────────────────────────────────┐
│                        训练流程总览                                  │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Phase 1: 联合训练 (Joint Training)                                │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │                                                           │    │
│   │  同时优化：                                                │    │
│   │  • entity_embedding                                       │    │
│   │  • relation_embedding                                     │    │
│   │  • rule_embedding                                         │    │
│   │  • policy_network (使用规则嵌入)                          │    │
│   │                                                           │    │
│   │  训练数据：                                                │    │
│   │  • 三元组 (h, r, t) → L_KGE                               │    │
│   │  • 规则 (r_head, r_body) → L_Rule                         │    │
│   │  • 查询 (h, r_q, t) → L_Policy                            │    │
│   │                                                           │    │
│   │  联合损失：                                                │    │
│   │  L_total = w1·L_KGE + w2·L_Rule + w3·L_Policy             │    │
│   │                                                           │    │
│   └──────────────────────────────────────────────────────────┘    │
│                           │                                        │
│                           ▼                                        │
│   Phase 2: RL微调 (RL Fine-tuning, 可选)                           │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │                                                           │    │
│   │  冻结嵌入层，微调策略网络：                                │    │
│   │  • 策略梯度更新                                            │    │
│   │  • 混合奖励: R = R_term + λ·R_rule                        │    │
│   │  • Loss = -E[R_t * log π(a_t|s_t)]                        │    │
│   │                                                           │    │
│   └──────────────────────────────────────────────────────────┘    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**关键改进**：
- ✓ 去掉独立的BFS标签生成阶段
- ✓ 去掉独立的SL预训练阶段
- ✓ 去掉Grounding阶段
- ✓ 嵌入和策略联合优化，端到端训练

### 3.2 Phase 1: 联合训练详解

**重要说明：什么是"联合训练"？**

联合训练不是"先训练嵌入，再训练策略"，而是**同时训练所有参数**：

```python
# ❌ 错误理解（顺序训练）
# Step 1: 先训练嵌入
for epoch in range(100):
    loss_emb = compute_kge_loss() + compute_rule_loss()
    loss_emb.backward()

# Step 2: 冻结嵌入，再训练策略
freeze(entity_emb, relation_emb, rule_emb)
for epoch in range(100):
    loss_policy = compute_policy_loss()
    loss_policy.backward()

# ✓ 正确理解（联合训练）
# 一个训练循环，同时更新所有参数
for epoch in range(100):
    # 在同一个forward pass中计算所有损失
    L_kge = compute_kge_loss()       # entity_emb, relation_emb参与计算
    L_rule = compute_rule_loss()      # rule_emb参与计算
    L_policy = compute_policy_loss()  # policy_network + 所有嵌入参与计算

    # 总损失
    total_loss = L_kge + L_rule + L_policy

    # 一次backward，梯度传播到所有参数
    total_loss.backward()  # 同时更新entity_emb, relation_emb, rule_emb, policy_network
    optimizer.step()       # 所有参数一起更新
```

**联合训练的优势**：
1. **嵌入直接服务于策略**：entity_emb和relation_emb的学习受到policy_network的反馈
2. **避免信息损失**：不存在"嵌入学完了，策略再适应"的gap
3. **端到端优化**：整个系统作为一个整体进行优化

#### 3.2.1 联合损失函数

```python
class JointTrainingLoss(nn.Module):
    def __init__(self, weight_kge=1.0, weight_rule=1.0, weight_policy=0.5):
        super().__init__()
        self.weight_kge = weight_kge
        self.weight_rule = weight_rule
        self.weight_policy = weight_policy

    def forward(self, batch_triplets, batch_rules, batch_queries, model):
        """
        联合训练损失：同时优化嵌入和策略

        Args:
            batch_triplets: [(h, r, t), ...] 三元组
            batch_rules: [(rule_id, r_head, r_body), ...] 规则
            batch_queries: [(h, r_q, t), ...] 查询
            model: RulE-SSRL模型

        Returns:
            total_loss, loss_dict
        """

        # ==== Loss 1: KGE损失 ====
        # 学习实体和关系嵌入，使得 score(h,r,t) 高
        kge_scores = model.compute_kge(batch_triplets)
        L_kge = self.compute_kge_loss(kge_scores)

        # ==== Loss 2: 规则嵌入损失 ====
        # 学习规则嵌入，使得 rule_emb + body_emb ≈ head_emb
        rule_scores = model.compute_rule_embedding_loss(batch_rules)
        L_rule = self.compute_rule_loss(rule_scores)

        # ==== Loss 3: 策略损失（核心！）====
        # 训练策略网络，使其能找到从h到t的路径
        L_policy = self.compute_policy_loss(batch_queries, model)

        # ==== 总损失 ====
        total_loss = (self.weight_kge * L_kge +
                     self.weight_rule * L_rule +
                     self.weight_policy * L_policy)

        return total_loss, {
            'L_kge': L_kge.item(),
            'L_rule': L_rule.item(),
            'L_policy': L_policy.item()
        }

    def compute_policy_loss(self, batch_queries, model):
        """
        策略损失：多种方式可选

        ⭐ 重要说明：本方案支持三种策略损失计算方式

        方式A: 规则监督（推荐，核心创新）
        - 用规则作为"软标签"指导策略学习
        - 规则建议的动作应该有更高的概率
        - 适用于有预挖掘规则的场景

        方式B: 自监督（备选）
        - 让策略网络尝试从h到达t
        - 用到达成功率作为监督信号
        - 适用于规则质量不高的场景

        方式C: 路径标签监督（如果有预标注路径）
        - 使用人工标注或BFS生成的完整路径标签
        - 论文SSRL使用的是BFS生成state-label pairs的变体
        - 适用于有高质量路径标注的场景

        对比论文：
        - 论文SSRL在SL阶段使用BFS生成的标签训练（类似方式C的变体）
        - 本文档的RulE-SSRL方案在联合训练阶段推荐使用方式A（规则监督）
        """
        total_loss = 0.0

        for h, r_q, t in batch_queries:
            # 方式A: 规则监督（核心创新！）
            # 用规则作为"软标签"指导策略学习
            loss = self.rule_supervised_policy_loss(h, r_q, t, model)

            # 方式B: 自监督（备选）
            # loss = self.self_supervised_policy_loss(h, r_q, t, model)

            total_loss += loss

        return total_loss / len(batch_queries)

    def rule_supervised_policy_loss(self, h, r_q, t, model):
        """
        规则监督的策略损失（核心创新）

        思想：
        1. 获取查询关系r_q的相关规则
        2. 规则建议的动作应该有更高的概率
        3. 用规则建议作为"软标签"训练策略

        ⚠️ 重要：负样本对抗策略

        当前实现（仅正样本）存在问题：
        - 只对规则建议的动作计算loss
        - 负样本（非规则建议的动作）未被显式抑制
        - 导致在大动作空间数据集（如FB15K-237）上效果不佳

        改进方案：需要添加负样本对抗，有三种实现方式
        """
        # 获取规则建议的路径
        applicable_rules = model.get_rules_for_relation(r_q)

        # 从h开始，按规则探索
        current = h
        path_history = []
        total_loss = 0.0

        for step in range(model.max_path_length):
            # 获取可选动作
            action_space = model.graph.get_neighbors(current)

            # 策略网络输出
            action_probs, _ = model.policy_network(
                current, r_q, path_history, action_space, model
            )

            # 规则建议的动作（软标签）
            rule_suggested_actions = []
            for rule_id, rule_emb, rule_body in applicable_rules:
                if step < len(rule_body):
                    rule_suggested_actions.append(rule_body[step])

            # ========== 负样本对抗方案（三选一）==========
            # 选择其中一种方案实现

            # 方案A：完整BCE（推荐，与论文SSRL一致）
            # 不需要采样，直接用动作空间中的所有动作
            loss = self._compute_loss_with_full_bce(
                action_space, action_probs, rule_suggested_actions
            )

            # 方案B：过滤负采样（类似KGE的filtered setting）
            # 从动作空间中采样负样本，过滤掉导致目标的动作
            # loss = self._compute_loss_with_filtered_sampling(
            #     action_space, action_probs, rule_suggested_actions,
            #     t, model, negative_sample_size=32
            # )

            # 方案C：随机负采样（类似Rule训练）
            # 纯随机采样负样本，不过滤
            # loss = self._compute_loss_with_random_sampling(
            #     action_space, action_probs, rule_suggested_actions,
            #     negative_sample_size=32
            # )

            total_loss += loss

            # 执行动作（选概率最高的）
            action_idx = torch.argmax(action_probs).item()
            relation, next_entity = action_space[action_idx]

            path_history.append((relation, next_entity))
            current = next_entity

            # 如果到达目标，结束
            if current == t:
                break

        return total_loss

    def _compute_loss_with_full_bce(self, action_space, action_probs, rule_suggested_actions):
        """
        方案A：完整BCE（Binary Cross Entropy）

        优势：
        - ✅ 与论文SSRL完全一致（论文公式6）
        - ✅ 实现最简单（3行代码）
        - ✅ 所有动作都有监督信号（正样本+负样本）
        - ✅ 不需要负采样逻辑

        劣势：
        - ❌ 在大动作空间时计算开销较大（但通常可接受）

        核心思想：
        - 动作空间本身就包含了所有候选动作（通常5-20个）
        - 不需要采样，直接将非正样本作为负样本
        - 正样本：规则建议的动作，label=1
        - 负样本：其他动作，label=0
        - 使用完整BCE计算loss

        显存分析：
        - 动作空间大小：通常5-50个
        - action_probs: [num_actions] float32
        - label_vector: [num_actions] float32
        - 显存占用：num_actions × 4 bytes × 2 ≈ 400 bytes（可忽略）
        - 不会爆显存
        """
        # 构建标签向量
        label_vector = torch.zeros(len(action_space))

        for i, (relation, next_entity) in enumerate(action_space):
            if relation in rule_suggested_actions:
                # 正样本：label=1
                label_vector[i] = 1.0 / len(rule_suggested_actions)
            # else: 负样本，label=0（已初始化为0）

        # 归一化（使标签和为1）
        if label_vector.sum() > 0:
            label_vector = label_vector / label_vector.sum()

        # 完整BCE损失
        # BCE = -Σ [y_i * log(p_i) + (1-y_i) * log(1-p_i)]
        #       ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^
        #       正样本项            负样本项
        loss = F.binary_cross_entropy(action_probs, label_vector)

        return loss

    def _compute_loss_with_filtered_sampling(self, action_space, action_probs,
                                            rule_suggested_actions, target_entity,
                                            model, negative_sample_size=32):
        """
        方案B：过滤负采样（类似KGE的filtered setting）

        优势：
        - ✅ 控制负样本数量，避免大动作空间的计算开销
        - ✅ 过滤真实路径，避免错误惩罚
        - ✅ 灵活：可调整负样本数量

        劣势：
        - ❌ 实现复杂，需要判断"哪些动作导致目标"
        - ❌ 需要额外存储或计算路径信息

        核心思想：
        - 从动作空间中采样固定数量的负样本
        - 过滤掉可能导致目标实体的动作（避免误伤好路径）
        - 类似KGE训练的filtered negative sampling

        类比KGE训练：
        - KGE：从14,505个实体中采样256个，过滤掉真实三元组
        - Policy：从20个动作中采样32个，过滤掉导致目标的动作
        """
        # Step 1: 找出正样本索引
        positive_indices = []
        for i, (relation, next_entity) in enumerate(action_space):
            if relation in rule_suggested_actions:
                positive_indices.append(i)

        # Step 2: 找出需要过滤的动作（导致目标的动作）
        # 简化版本：只过滤直接连接到目标的动作
        true_actions = []
        for i, (relation, next_entity) in enumerate(action_space):
            if next_entity == target_entity:
                true_actions.append(i)
            # 更复杂版本：检查是否存在路径 next_entity -> ... -> target_entity
            # 需要BFS或其他路径搜索算法

        # Step 3: 采样负样本（排除正样本和true_actions）
        all_indices = set(range(len(action_space)))
        exclude_indices = set(positive_indices + true_actions)
        candidate_negative_indices = list(all_indices - exclude_indices)

        if len(candidate_negative_indices) == 0:
            # 无负样本可用，回退到只计算正样本
            positive_loss = 0.0
            for i in positive_indices:
                positive_loss += -torch.log(action_probs[i] + 1e-10)
            return positive_loss / len(positive_indices)

        # 随机采样
        num_neg_samples = min(negative_sample_size, len(candidate_negative_indices))
        negative_indices = random.sample(candidate_negative_indices, num_neg_samples)

        # Step 4: 计算BCE损失（只对正样本和采样的负样本）
        label_vector = torch.zeros(len(action_space))
        for i in positive_indices:
            label_vector[i] = 1.0 / len(positive_indices)

        # 只对选中的动作计算loss
        selected_indices = positive_indices + negative_indices
        selected_probs = action_probs[selected_indices]
        selected_labels = label_vector[selected_indices]

        loss = F.binary_cross_entropy(selected_probs, selected_labels)

        return loss

    def _compute_loss_with_random_sampling(self, action_space, action_probs,
                                          rule_suggested_actions, negative_sample_size=32):
        """
        方案C：随机负采样（类似Rule训练）

        优势：
        - ✅ 实现极简单
        - ✅ 控制负样本数量
        - ✅ 计算开销低

        劣势：
        - ❌ 可能采样到"好的"动作并错误惩罚
        - ❌ 在动作空间很大时，随机采样可能miss重要的负样本

        核心思想：
        - 从动作空间中纯随机采样负样本
        - 不过滤任何动作
        - 类似Rule训练的负采样方式

        类比Rule训练：
        - Rule：随机替换规则body中的关系，不过滤
        - Policy：随机采样动作空间中的负样本，不过滤
        """
        # Step 1: 找出正样本索引
        positive_indices = []
        for i, (relation, next_entity) in enumerate(action_space):
            if relation in rule_suggested_actions:
                positive_indices.append(i)

        # Step 2: 纯随机采样负样本（只排除正样本）
        all_indices = set(range(len(action_space)))
        exclude_positive = all_indices - set(positive_indices)

        if len(exclude_positive) == 0:
            # 无负样本，只计算正样本
            positive_loss = 0.0
            for i in positive_indices:
                positive_loss += -torch.log(action_probs[i] + 1e-10)
            return positive_loss / len(positive_indices)

        # 随机采样，不过滤
        num_neg_samples = min(negative_sample_size, len(exclude_positive))
        negative_indices = random.sample(list(exclude_positive), num_neg_samples)

        # Step 3: 计算BCE损失
        label_vector = torch.zeros(len(action_space))
        for i in positive_indices:
            label_vector[i] = 1.0 / len(positive_indices)

        selected_indices = positive_indices + negative_indices
        selected_probs = action_probs[selected_indices]
        selected_labels = label_vector[selected_indices]

        loss = F.binary_cross_entropy(selected_probs, selected_labels)

        return loss

    def self_supervised_policy_loss(self, h, r_q, t, model):
        """
        自监督策略损失（备选方案）

        思想：
        让策略网络尝试从h到达t，用到达成功率作为监督信号
        """
        # 多次采样
        num_rollouts = 5
        success_count = 0

        for _ in range(num_rollouts):
            path = model.policy_network.rollout(h, r_q, max_steps=3)
            if path[-1][1] == t:  # 最后一个实体是否是目标
                success_count += 1

        # 成功率越高越好
        success_rate = success_count / num_rollouts
        loss = -torch.log(torch.tensor(success_rate + 1e-10))

        return loss
```

#### 3.2.2 负采样方案的显存与性能分析

**关键问题：方案A（完整BCE）会不会爆显存？**

答案：**不会**。下面详细分析。

---

**一、显存占用对比分析**

```python
# ========== 假设条件 ==========
# 数据集：FB15K-237（最大动作空间）
# 实体数：14,505
# 平均每个实体的邻居数：19.74
# 最坏情况动作空间大小：100个邻居

# ========== 方案A：完整BCE ==========
# 单个query、单个step的显存占用

# 1. 动作空间大小（最坏情况）
num_actions = 100  # 邻居数

# 2. 策略网络计算
action_probs = policy_network(...)  # [100] float32
# 显存：100 × 4 bytes = 400 bytes

# 3. 标签向量
label_vector = torch.zeros(100)  # [100] float32
# 显存：100 × 4 bytes = 400 bytes

# 4. BCE计算
loss = F.binary_cross_entropy(action_probs, label_vector)  # 标量
# 显存：4 bytes

# 5. 梯度（反向传播时）
action_probs.grad  # [100] float32
label_vector.grad  # [100] float32（不需要梯度）
# 显存：100 × 4 bytes = 400 bytes

# 总计（单个query、单个step）：
# 400 + 400 + 4 + 400 = 1,204 bytes ≈ 1.2 KB

# ========== 完整训练iteration ==========
# 假设：batch_queries = 32个query
#      每个query走3步
#      每步平均20个动作

total_memory_per_iteration = 32 × 3 × 20 × 4 × 3  # query × steps × actions × 4bytes × 3张量
                           = 23,040 bytes
                           ≈ 23 KB

# ========== 对比：KGE训练的显存占用 ==========
# KGE负采样（data.py实现）
batch_size = 512
negative_sample_size = 256
entity_dim = 500

# 正样本嵌入
head_emb = entity_embedding(batch_triplets[:, 0])  # [512, 500] float32
rel_emb = relation_embedding(batch_triplets[:, 1])  # [512, 250] float32
tail_emb = entity_embedding(batch_triplets[:, 2])  # [512, 500] float32

# 负样本嵌入
neg_tail_emb = entity_embedding(neg_tails)  # [512, 256, 500] float32

# 显存占用：
kge_memory = 512 × 500 × 4 + 512 × 250 × 4 + 512 × 500 × 4 + 512 × 256 × 500 × 4
           = 1,024,000 + 512,000 + 1,024,000 + 262,144,000
           = 264,704,000 bytes
           ≈ 252 MB

# ========== 对比结论 ==========
# Policy训练（方案A）：23 KB / iteration
# KGE训练：           252 MB / iteration
# Policy只占KGE的 0.009%
```

---

**二、为什么方案A不会爆显存？**

**核心原因**：动作空间本身就很小

| 数据集 | 平均度数 | 最大度数（估计） | 方案A显存 |
|--------|---------|----------------|----------|
| **WN18RR** | 2.19 | ~10 | ~1 KB |
| **NELL-995** | 4.07 | ~30 | ~3 KB |
| **FB15K-237** | 19.74 | ~100 | ~10 KB |
| **UMLS** | 6.8 | ~50 | ~5 KB |

**对比KGE为什么需要负采样？**

```
KGE训练：
├─ 候选空间：14,505个实体（全图）
├─ 如果不采样：14,505 × 500 × 4 = 29 MB（仅一个batch的一个三元组）
└─ 必须采样：256个负样本 = 512 KB（可接受）

Policy训练：
├─ 候选空间：20个动作（当前节点邻居）
├─ 不需要采样：20 × 4 × 3 = 240 bytes（可忽略）
└─ 结论：直接用所有动作，无需采样
```

---

**三、极端情况分析**

**最坏情况：遇到超高度数节点（100+邻居）**

```python
# 假设最坏情况
num_actions = 200  # 极端高度数节点
batch_queries = 32
max_steps = 3

# 方案A显存
worst_case_memory = 32 × 3 × 200 × 4 × 3
                  = 230,400 bytes
                  ≈ 225 KB

# 仍然可忽略（现代GPU通常8GB+显存）
```

**如果仍然担心，可以使用动作空间剪枝**：

```python
def rule_supervised_policy_loss_with_pruning(self, h, r_q, t, model):
    max_action_space_size = 50  # 剪枝阈值

    for step in range(model.max_path_length):
        action_space = model.graph.get_neighbors(current)

        # 如果动作空间太大，剪枝
        if len(action_space) > max_action_space_size:
            # 优先保留规则建议的动作
            action_space = self.prune_action_space(
                action_space,
                applicable_rules,
                step,
                top_k=max_action_space_size
            )

        # 后续与方案A完全一致
        loss = self._compute_loss_with_full_bce(
            action_space, action_probs, rule_suggested_actions
        )
```

---

**四、三种方案的显存对比**

| 方案 | 单个query显存 | 是否需要采样 | 显存风险 | 推荐度 |
|------|-------------|------------|---------|--------|
| **方案A：完整BCE** | ~1-10 KB | ❌ 不需要 | ⭐ 极低 | ⭐⭐⭐⭐⭐ |
| **方案B：过滤负采样** | ~1-3 KB | ✅ 需要（固定32个） | ⭐ 极低 | ⭐⭐⭐ |
| **方案C：随机负采样** | ~1-3 KB | ✅ 需要（固定32个） | ⭐ 极低 | ⭐⭐ |

**结论**：
- 方案A虽然用所有动作，但显存占用极低（~1-10 KB）
- 远低于KGE训练（~252 MB）
- 不会爆显存，可以放心使用

---

**五、实际训练的显存占用估算**

```python
# ========== 完整训练显存峰值估算 ==========

# 1. 模型参数（常驻显存）
model_params = {
    'entity_embedding': 14505 × 500 × 4 = 29 MB,
    'relation_embedding': 237 × 250 × 4 = 0.24 MB,
    'rule_embedding': 1000 × 250 × 4 = 1 MB,
    'policy_network': ~2M params × 4 = 8 MB,
}
total_model = 29 + 0.24 + 1 + 8 = 38.24 MB

# 2. 梯度（常驻显存，与模型参数1:1）
gradients = 38.24 MB

# 3. 优化器状态（Adam：2倍参数）
optimizer_state = 38.24 × 2 = 76.48 MB

# 4. 训练batch（临时显存）
training_batch = {
    'KGE_batch': 252 MB,          # 最大头
    'Rule_batch': 1 MB,
    'Policy_batch': 0.023 MB,      # 方案A
}
total_batch = 252 + 1 + 0.023 = 253 MB

# 5. 中间激活值（估计）
activations = ~50 MB

# ========== 峰值显存总计 ==========
peak_memory = model + gradients + optimizer + batch + activations
            = 38.24 + 38.24 + 76.48 + 253 + 50
            = 455.96 MB
            ≈ 456 MB

# ========== 结论 ==========
# 即使使用方案A（完整BCE），显存占用仍然很小
# Policy部分只占 0.023 MB / 456 MB = 0.005%
# 可以在任何现代GPU上训练（8GB显存足够）
```

---

**六、优化建议**

如果仍然想进一步降低显存（虽然没必要）：

**优化1：动作空间剪枝**
```python
# 限制最大动作空间为50
if len(action_space) > 50:
    action_space = prune_to_top_k(action_space, k=50)
```

**优化2：使用方案C（固定32个负样本）**
```python
# 显存从10 KB降到3 KB
# 但牺牲了训练效果
```

**优化3：减少query_batch_size**
```python
# 从32降到16
# 但训练速度变慢
```

**推荐**：
- ✅ 直接使用方案A，无需优化
- ✅ 显存占用可忽略（< 1% 总显存）
- ❌ 不要过度优化，反而增加复杂度

---

**七、实验验证**

可以在实际训练中监控显存：

```python
import torch

# 训练前
torch.cuda.reset_peak_memory_stats()

# 训练一个epoch
for batch in dataloader:
    loss = compute_loss(batch)
    loss.backward()
    optimizer.step()

# 检查峰值显存
peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
print(f"Peak memory: {peak_memory:.2f} MB")

# 预期：~400-500 MB（FB15K-237）
# Policy部分占比 < 1%
```

---

**总结**：

1. **方案A（完整BCE）不会爆显存**
   - 单个query只占1-10 KB
   - 比KGE训练小1万倍

2. **为什么不需要负采样？**
   - 动作空间本身就小（5-20个）
   - 不像KGE需要从14,505个实体采样

3. **推荐方案**：
   - 🥇 直接使用方案A（完整BCE）
   - 显存占用可忽略
   - 无需负采样优化

#### 3.2.3 训练循环详解

**关键问题：如何训练策略网络？**

策略网络训练的特殊性：
- KGE损失和Rule损失：可以batch并行计算
- Policy损失：需要"走路径"，逐个query处理

**重要说明："逐个处理"是什么意思？**

**先回答关键问题：32个查询从哪来？**

```python
# ========== 数据准备（训练开始前）==========
# 从训练集准备三种数据源

# 1. 三元组数据集（用于KGE损失）
triplet_dataset = [
    (Tom_Brady, plays_for, Buccaneers),
    (Buccaneers, located_in, Tampa),
    (Patrick_Mahomes, plays_for, Chiefs),
    ...  # 共50,000个三元组
]

# 2. 规则数据集（用于Rule损失）
rule_dataset = [
    (rule_0, coached_by, [plays_for, head_coach]),
    (rule_1, plays_in_league, [plays_for, belongs_to]),
    ...  # 共2,000条规则
]

# 3. 查询数据集（用于Policy损失）
# 重要：这些query也是从训练三元组中来的！
query_dataset = triplet_dataset  # 就是三元组！
# 或者：query_dataset = triplet_dataset[:10000]  # 可以只用一部分

# ========== 每个训练iteration的采样 ==========
# 每个iteration从三个数据源独立采样batch

# 配置参数（可调整）
triplet_batch_size = 512  # KGE用的batch size
rule_batch_size = 256     # Rule用的batch size
query_batch_size = 32     # Policy用的batch size（不是固定的！）

# 采样
batch_triplets = random.sample(triplet_dataset, triplet_batch_size)
# 采样512个三元组：
# [
#     (Tom_Brady, plays_for, Buccaneers),
#     (Buccaneers, located_in, Tampa),
#     ...  # 共512个
# ]

batch_rules = random.sample(rule_dataset, rule_batch_size)
# 采样256条规则：
# [
#     (rule_0, coached_by, [plays_for, head_coach]),
#     (rule_1, plays_in_league, [plays_for, belongs_to]),
#     ...  # 共256条
# ]

batch_queries = random.sample(query_dataset, query_batch_size)
# 采样32个查询（实际上也是三元组）：
# [
#     (Tom_Brady, coached_by, Todd_Bowles),      ← 这个可能在batch_triplets里
#     (Patrick_Mahomes, plays_in_league, NFL),   ← 也可能不在
#     ...  # 共32个
# ]

# 重要说明：
# 1. batch_queries和batch_triplets可能有重叠，但通常不同
# 2. 三个batch是独立采样的
# 3. query_batch_size=32只是示例，可以调整（如16, 64等）
```

**关键点**：
- **32不是固定的**：可以配置（query_batch_size参数）
- **queries的来源**：从训练三元组中采样，和batch_triplets是同一个数据源
- **独立采样**：batch_queries和batch_triplets可能不同

**数据流示意图**：

```
训练数据准备：
┌─────────────────────────────────────────────────────────┐
│              原始训练数据                                 │
│  train.txt: 50,000个三元组                               │
│  mined_rules.txt: 2,000条规则                           │
└─────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          创建三个数据加载器                               │
├─────────────────────────────────────────────────────────┤
│  1. Triplet DataLoader                                  │
│     └─ 用于KGE损失，batch_size=512                       │
│                                                         │
│  2. Rule DataLoader                                     │
│     └─ 用于Rule损失，batch_size=256                      │
│                                                         │
│  3. Query DataLoader (关键！)                           │
│     └─ 用于Policy损失，batch_size=32                     │
│     └─ 数据源：也是train.txt的三元组                      │
│     └─ 为什么叫"查询"？因为训练策略网络找路径             │
└─────────────────────────────────────────────────────────┘
                     │
                     ▼
         每个训练iteration独立采样
┌─────────────────────────────────────────────────────────┐
│  ⭐ 重要：三种数据都使用随机采样（与RulE原模型一致）        │
├─────────────────────────────────────────────────────────┤
│  Iteration 1:                                           │
│  ├─ batch_triplets: [512个三元组]  ← 随机采样           │
│  ├─ batch_rules: [256条规则]       ← 随机采样           │
│  └─ batch_queries: [32个查询]      ← 随机采样           │
│                                                         │
│  Iteration 2:                                           │
│  ├─ batch_triplets: [另外512个三元组]  ← 随机采样       │
│  ├─ batch_rules: [另外256条规则]       ← 随机采样       │
│  └─ batch_queries: [另外32个查询]      ← 随机采样       │
│  ...                                                    │
│                                                         │
│  说明：                                                  │
│  • 三个数据源每个iteration独立随机采样                   │
│  • batch_queries和batch_triplets可能有重叠但通常不同    │
│  • 论文SSRL也使用随机采样策略                            │
│  • RulE原模型在预训练阶段也使用随机采样                  │
└─────────────────────────────────────────────────────────┘
```

**代码实现示例**：

```python
# ========== 数据加载器创建（训练开始前）==========
from torch.utils.data import DataLoader

# 1. KGE数据加载器
triplet_dataset = KGETrainDataset(train_triplets)  # 50,000个三元组
triplet_loader = DataLoader(
    triplet_dataset,
    batch_size=512,  # 可配置
    shuffle=True
)

# 2. Rule数据加载器
rule_dataset = RuleDataset(mined_rules)  # 2,000条规则
rule_loader = DataLoader(
    rule_dataset,
    batch_size=256,  # 可配置
    shuffle=True
)

# 3. Query数据加载器（关键！）
# 数据源就是训练三元组
query_dataset = QueryDataset(train_triplets)  # 也是50,000个三元组
query_loader = DataLoader(
    query_dataset,
    batch_size=32,   # 可配置（不是固定的32！）
    shuffle=True
)

# ========== 训练循环 ==========
for epoch in range(num_epochs):
    # 三个迭代器可能长度不同，需要同步
    # 方法1：zip最短的
    for (batch_triplets, batch_rules, batch_queries) in zip(
        triplet_loader,   # 50000/512 = 98个batch
        rule_loader,      # 2000/256 = 8个batch（会重复采样）
        query_loader      # 50000/32 = 1563个batch
    ):
        # batch_triplets: [512, 3]
        # batch_rules: [256, max_rule_len]
        # batch_queries: [32, 3]

        # 计算三种损失
        L_kge = compute_kge_loss(batch_triplets)
        L_rule = compute_rule_loss(batch_rules)
        L_policy = compute_policy_loss(batch_queries)  # 这32个query要for循环处理

        # 总损失
        total_loss = L_kge + L_rule + 0.5 * L_policy
        total_loss.backward()
        optimizer.step()
```

**关于batch_queries和batch_triplets的关系**：

```python
# 示例说明
batch_triplets = [
    (Tom_Brady, plays_for, Buccaneers),      # ID 1
    (LeBron_James, plays_for, Lakers),       # ID 2
    (Buccaneers, located_in, Tampa),         # ID 3
    ...  # 共512个
]

batch_queries = [
    (Tom_Brady, coached_by, Todd_Bowles),    # ID 5001，不在batch_triplets里
    (LeBron_James, plays_for, Lakers),       # ID 2，恰好在batch_triplets里
    (Brady, won, Super_Bowl),                # ID 8888，不在batch_triplets里
    ...  # 共32个
]

# 关系：
# 1. 两者都来自train.txt（同一数据源）
# 2. 但每个iteration独立采样，所以可能有重叠但通常不同
# 3. 数量不同（512 vs 32）
# 4. batch_queries中的三元组，用于训练策略网络"走路径"
```

# ========== 三种损失的计算方式对比 ==========

# 1️⃣ KGE损失：batch并行（矩阵运算）
def compute_kge_loss_batch(model, batch_triplets):
    """
    一次性计算整个batch的KGE损失
    """
    # 所有三元组一起处理
    heads = model.entity_embedding(batch_triplets[:, 0])    # [512, 500]
    relations = model.relation_embedding(batch_triplets[:, 1])  # [512, 250]
    tails = model.entity_embedding(batch_triplets[:, 2])    # [512, 500]

    # 矩阵运算，一次计算所有得分
    scores = model.score_function(heads, relations, tails)  # [512]
    loss = margin_loss(scores)

    return loss  # 一次返回整个batch的loss

# 2️⃣ Rule损失：batch并行（矩阵运算）
def compute_rule_loss_batch(model, batch_rules):
    """
    一次性计算整个batch的Rule损失
    """
    rule_ids = batch_rules[:, 0]  # [256]
    rule_heads = batch_rules[:, 1]  # [256]
    rule_bodies = batch_rules[:, 2:]  # [256, max_len]

    # 所有规则一起处理
    rule_embs = model.rule_embedding(rule_ids)  # [256, 250]
    body_embs = model.relation_embedding(rule_bodies)  # [256, 3, 250]
    head_embs = model.relation_embedding(rule_heads)  # [256, 250]

    # 矩阵运算
    scores = compute_rule_scores(rule_embs, body_embs, head_embs)  # [256]
    loss = margin_loss(scores)

    return loss  # 一次返回整个batch的loss

# 3️⃣ Policy损失：逐个处理（for循环）
def compute_policy_loss_batch(model, batch_queries):
    """
    ❌ 不能像KGE那样batch并行！
    ✅ 需要for循环逐个处理每个query
    """
    total_policy_loss = 0.0

    # 关键：用for循环逐个处理
    for h, r_q, t in batch_queries:  # 循环32次
        # 每个query需要"走路径"
        policy_loss_single = train_policy_single_query(model, h, r_q, t)
        total_policy_loss += policy_loss_single

    # 平均
    avg_policy_loss = total_policy_loss / len(batch_queries)

    return avg_policy_loss

# ========== 为什么Policy损失不能batch并行？ ==========

def train_policy_single_query(model, h, r_q, t):
    """
    为什么需要逐个处理？因为"走路径"是动态、sequential的过程
    """
    current = h  # Tom_Brady
    path_history = []
    total_loss = 0.0

    # 走3步
    for step in range(3):
        # ❌ 问题1：每个query在每一步的动作空间不同
        action_space = model.graph.get_neighbors(current)
        # Tom_Brady: [(plays_for, Buccaneers), (born_in, USA), ...]  # 5个邻居
        # Patrick_Mahomes: [(plays_for, Chiefs), (won, Super_Bowl), ...]  # 8个邻居
        # 不同大小的动作空间很难batch处理（需要复杂的padding和mask）

        # ❌ 问题2：每个query的路径长度可能不同
        # 有的query 2步就到达目标，有的需要3步
        # 动态停止很难用矩阵运算表示

        # 策略网络计算
        action_probs = model.policy_network(
            current, r_q, path_history, action_space, model
        )  # [num_neighbors]

        # 计算损失（基于规则建议）
        loss = compute_action_loss(action_probs, action_space, r_q, model)
        total_loss += loss

        # 选择动作，移动
        action_idx = torch.argmax(action_probs)
        relation, next_entity = action_space[action_idx]

        path_history.append((relation, next_entity))
        current = next_entity  # 移动到下一个实体

        # ❌ 问题3：动态停止条件
        if current == t:
            break  # 提前到达，停止

    return total_loss

# ========== 完整训练流程 ==========
```

**图示说明**：

```
训练Iteration的数据流：

┌─────────────────────────────────────────────────────────────┐
│            一个训练iteration                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1️⃣ KGE Loss（batch并行，一次性处理）                         │
│     batch_triplets = [512个三元组]                           │
│     ↓                                                        │
│     [矩阵运算] → L_kge (单个标量)                             │
│     耗时：~10ms                                              │
│                                                             │
│  2️⃣ Rule Loss（batch并行，一次性处理）                        │
│     batch_rules = [256条规则]                                │
│     ↓                                                        │
│     [矩阵运算] → L_rule (单个标量)                            │
│     耗时：~5ms                                               │
│                                                             │
│  3️⃣ Policy Loss（逐个处理，for循环）                          │
│     batch_queries = [32个查询]                               │
│     ↓                                                        │
│     for query in batch_queries:  # 循环32次                 │
│         ├─ query_1: Tom_Brady → ... → Todd_Bowles (3步)     │
│         │   耗时：~2ms                                       │
│         ├─ query_2: Patrick_Mahomes → ... → NFL (2步)       │
│         │   耗时：~1.5ms                                     │
│         └─ ...                                              │
│     ↓                                                        │
│     L_policy = sum(losses) / 32 (单个标量)                   │
│     总耗时：~50ms (32个query累加)                            │
│                                                             │
│  总损失 = L_kge + L_rule + 0.5 * L_policy                   │
│  ↓                                                          │
│  total_loss.backward() → optimizer.step()                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**总结**：
1. **batch_queries有32个query**（不是只有1个）
2. **但需要用for循环逐个处理**（不能矩阵并行）
3. **为什么？** 因为：
   - 每个query的动作空间大小不同
   - 每个query的路径长度不同
   - "走路径"是sequential过程
4. **性能影响**：Policy损失计算较慢（~50ms），但通过较小的权重（0.5）来平衡

```python
def joint_training(model, train_data, num_epochs, device):
    """
    联合训练：嵌入 + 策略

    训练数据组织：
    train_data包含三种数据：
    • triplets: [(h, r, t), ...] 用于KGE loss
    • rules: [(rule_id, r_head, r_body), ...] 用于Rule loss
    • queries: [(h, r_q, t), ...] 用于Policy loss
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = JointTrainingLoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # 准备批次数据
        for batch_idx, (triplets, rules, queries) in enumerate(train_data):
            triplets = triplets.to(device)  # [batch_size, 3]
            rules = rules.to(device)        # [batch_size, max_rule_len]
            queries = queries.to(device)     # [batch_size, 3]

            # ========== 分别计算三种损失 ==========

            # 1. KGE损失（可以并行batch计算）
            L_kge = compute_kge_loss_batch(model, triplets)

            # 2. Rule损失（可以并行batch计算）
            L_rule = compute_rule_loss_batch(model, rules)

            # 3. Policy损失（需要逐个query处理）
            L_policy = 0.0
            for h, r_q, t in queries:
                # 从h出发，用规则引导策略走路径
                policy_loss_single = train_policy_single_query(
                    model, h, r_q, t
                )
                L_policy += policy_loss_single

            L_policy /= len(queries)  # 平均

            # 总损失
            total_loss = (1.0 * L_kge + 1.0 * L_rule + 0.5 * L_policy)

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            # 日志
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  L_kge: {L_kge.item():.4f}")
                print(f"  L_rule: {L_rule.item():.4f}")
                print(f"  L_policy: {L_policy.item():.4f}")

        print(f"Epoch {epoch} finished. Avg loss: {epoch_loss/len(train_data):.4f}")

        # 验证
        if epoch % 5 == 0:
            validate(model, valid_data, device)


def train_policy_single_query(model, h, r_q, t):
    """
    训练单个query的策略

    核心：用规则作为"软监督"

    步骤：
    1. 获取r_q的相关规则
    2. 从h开始，策略网络选择动作
    3. 如果选择的动作符合规则建议 → 奖励
    4. 如果不符合 → 惩罚
    """
    # 获取规则建议
    applicable_rules = model.get_rules_for_relation(r_q)

    current = h
    path_history = []
    total_loss = 0.0

    for step in range(3):  # 最多走3步
        # 获取当前可选动作
        action_space = model.graph.get_neighbors(current)
        # [(r1, e1), (r2, e2), ...]

        # 策略网络输出动作概率
        action_probs = model.policy_network(
            current, r_q, path_history, action_space, model
        )

        # 规则建议哪些动作？
        rule_suggested_relations = []
        for rule_id, rule_emb, rule_body in applicable_rules:
            if step < len(rule_body):
                rule_suggested_relations.append(rule_body[step])

        # 计算损失：鼓励选择规则建议的动作
        for i, (relation, next_entity) in enumerate(action_space):
            if relation in rule_suggested_relations:
                # 这个动作符合规则，应该有高概率
                # 交叉熵损失
                loss = -torch.log(action_probs[i] + 1e-10)
                total_loss += loss

        # 执行动作（选概率最高的）
        action_idx = torch.argmax(action_probs)
        relation, next_entity = action_space[action_idx]

        path_history.append((relation, next_entity))
        current = next_entity

        if current == t:
            break  # 到达目标

    return total_loss
```

**训练特点总结**：
- **KGE/Rule损失**：batch并行，效率高
- **Policy损失**：逐个query，需要走路径，较慢
- **平衡策略**：可以用较小的policy loss权重（如0.5）

### 3.3 Phase 2: RL微调（可选）

```python
def rl_finetuning(model, train_queries, num_epochs, device):
    """
    RL微调：冻结嵌入，用Policy Gradient微调策略网络
    """
    # 冻结嵌入层
    for param in model.entity_embedding.parameters():
        param.requires_grad = False
    for param in model.relation_embedding.parameters():
        param.requires_grad = False
    for param in model.rule_embedding.parameters():
        param.requires_grad = False

    # 只优化策略网络
    optimizer = torch.optim.Adam(model.policy_network.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for h, r_q, t in train_queries:
            # 采样一个episode
            trajectory = model.policy_network.rollout(h, r_q, max_steps=3)

            # 计算回报（使用混合奖励）
            returns = compute_returns(trajectory, t, r_q, model)

            # Policy Gradient更新
            loss = policy_gradient_loss(trajectory, returns)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("RL fine-tuning完成")
```

### 3.3.1 维度对齐与内存管理

**问题1：嵌入维度能对应上吗？**

**答案**：可以对应，通过神经网络层自动对齐维度。

在RulE-SSRL中，不同组件的嵌入维度可以不同：

```python
# 维度配置示例（基于RulE原始设计）
entity_dim = 500          # 实体嵌入（RotatE用复数，实际是250维×2）
relation_dim = 250        # 关系嵌入
rule_dim = 250            # 规则嵌入
policy_hidden_dim = 256   # 策略网络LSTM隐藏层

# 为什么可以不同？
# 答：通过线性层自动对齐
```

**策略网络中的维度对齐**：

```python
class RuleGuidedPolicyNetwork(nn.Module):
    def __init__(self, entity_dim, relation_dim, rule_dim, hidden_dim):
        super().__init__()
        self.entity_dim = entity_dim      # 500
        self.relation_dim = relation_dim  # 250
        self.rule_dim = rule_dim          # 250
        self.hidden_dim = hidden_dim      # 256

        # LSTM自动处理输入维度对齐
        self.lstm = nn.LSTM(
            input_size=relation_dim + entity_dim,  # 250 + 500 = 750
            hidden_size=hidden_dim,                 # 输出256
            batch_first=True
        )

        # 规则注意力：自动对齐不同维度
        self.rule_attention = nn.Sequential(
            nn.Linear(
                hidden_dim + relation_dim + rule_dim,  # 256 + 250 + 250 = 756
                hidden_dim                             # 输出256
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 最终输出标量得分
        )

        # 动作评分：拼接后再线性变换
        self.action_base_scorer = nn.Linear(
            hidden_dim + relation_dim + entity_dim,  # 256 + 250 + 500 = 1006
            1  # 输出一个得分
        )

    def forward(self, current_entity, query_relation, path_history, ...):
        # 获取不同维度的嵌入
        entity_emb = self.entity_emb[current_entity]  # [500]
        relation_emb = self.relation_emb[query_relation]  # [250]
        rule_emb = self.rule_emb[rule_id]  # [250]

        # 直接拼接，线性层自动处理
        combined = torch.cat([entity_emb, relation_emb, rule_emb], dim=-1)
        # combined.shape = [500 + 250 + 250] = [1000]

        # 通过线性层降维
        output = self.some_linear_layer(combined)  # [1000] → [256]
```

**关键点**：
- **不需要手动对齐**：PyTorch的`nn.Linear`层会自动学习如何组合不同维度
- **灵活配置**：可以根据数据集调整各个维度（大数据集用更大维度）
- **无信息损失**：线性层通过训练学习最佳的维度映射

**问题2：训练会数据爆炸吗？**

**答案**：不会，因为设计了内存控制策略。

```python
# 内存占用分析

# ========== 组件1：KGE损失（batch并行）==========
batch_size = 512  # 可控制
triplet_batch = torch.tensor(triplets[:batch_size])  # [512, 3]

# 正样本嵌入
head_emb = entity_embedding(triplet_batch[:, 0])  # [512, 500]
rel_emb = relation_embedding(triplet_batch[:, 1])  # [512, 250]
tail_emb = entity_embedding(triplet_batch[:, 2])  # [512, 500]

# 负采样
negative_samples = 128  # 可控制
neg_tails = sample_negatives(batch_size, negative_samples)  # [512, 128]
neg_tail_emb = entity_embedding(neg_tails)  # [512, 128, 500]

# 内存占用：512 × (500 + 250 + 500) + 512 × 128 × 500
#          ≈ 0.64M + 32.8M = 33.44M 参数
#          × 4 bytes (float32) = 133.76 MB （可接受）

# ========== 组件2：Rule损失（batch并行）==========
rule_batch_size = 256  # 规则数量有限，batch小一些
rule_batch = rules[:rule_batch_size]  # [256, max_rule_length]

rule_emb = rule_embedding(rule_batch[:, 0])  # [256, 250]
body_emb = relation_embedding(rule_batch[:, 2:])  # [256, 3, 250]

# 内存占用：256 × 250 + 256 × 3 × 250
#          = 64K + 192K = 256K 参数
#          × 4 bytes = 1 MB （很小）

# ========== 组件3：Policy损失（逐个处理，避免爆炸）==========
# 关键：不batch处理query，而是逐个处理
for h, r_q, t in queries:  # 一次只处理一个query
    current = h
    path_history = []  # 当前路径历史

    for step in range(max_path_length):  # max_path_length=3
        # 获取邻居
        neighbors = graph.get_neighbors(current)  # 假设平均50个邻居

        # 策略网络计算
        state = lstm(path_history)  # [1, 256]
        action_scores = []

        for relation, next_entity in neighbors:  # 最多50次循环
            rel_emb = relation_embedding(relation)  # [250]
            ent_emb = entity_embedding(next_entity)  # [500]
            combined = torch.cat([state, rel_emb, ent_emb])  # [1006]
            score = action_scorer(combined)  # [1]
            action_scores.append(score)

        # 内存占用：1 × 256 + 50 × (250 + 500 + 1006)
        #          = 256 + 50 × 1756 = 87,800 参数
        #          × 4 bytes = 351 KB （每个query）

        # 选择动作，移动到下一个实体
        current = next_entity

# 为什么不会爆炸？
# 1. 每次只处理一个query
# 2. 路径长度限制（max=3）
# 3. 不accumulate梯度跨query
# 4. 动作空间可以剪枝（如限制top-50邻居）

# ========== 总内存估算（训练时）==========
# 模型参数：
# - entity_embedding: 14,505 entities × 500 = 7.25M
# - relation_embedding: 237 relations × 250 = 0.06M
# - rule_embedding: ~1000 rules × 250 = 0.25M
# - policy_network: ~2M parameters
# 总计：≈ 9.56M parameters × 4 bytes = 38.24 MB （模型本身）

# 训练batch内存：
# - KGE batch: ~134 MB
# - Rule batch: ~1 MB
# - Policy (单个query): ~0.35 MB
# 总计：≈ 135 MB （每个训练step）

# 峰值内存：模型(38 MB) + 梯度(38 MB) + 训练batch(135 MB)
#          ≈ 211 MB （非常小！）

# 实际GPU占用会稍高（缓存、中间变量等），但远小于现代GPU容量（8GB+）
```

**内存控制策略总结**：

1. **KGE/Rule批处理**：
   - 使用batch并行，可控制batch_size
   - 负采样数量可调（默认128）

2. **Policy逐个处理**：
   - 每次只处理一个query，不batch
   - 路径长度限制（max_path_length=3）
   - 动作空间剪枝（如top-50邻居）

3. **梯度累积**（如果需要更大effective batch）：
   ```python
   accumulation_steps = 4
   for i, (triplets, rules, queries) in enumerate(dataloader):
       loss = compute_loss(triplets, rules, queries)
       loss = loss / accumulation_steps
       loss.backward()

       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

4. **动作空间剪枝**：
   ```python
   # 如果邻居太多（如>100），剪枝
   neighbors = graph.get_neighbors(current)
   if len(neighbors) > 50:
       # 基于规则或嵌入相似度筛选top-50
       neighbors = prune_actions(neighbors, top_k=50)
   ```

**结论**：
- ✓ 维度可以不同，通过线性层自动对齐
- ✓ 内存可控，不会爆炸
- ✓ 可以在普通GPU上训练（8GB显存足够）

### 3.4 推理阶段详解

**核心问题：**
1. 如何从h出发选出K条路径？
2. 为什么有A/B/C三种聚合方法？

#### 3.4.1 路径采样过程（问题1）

```python
def sample_one_path(model, h, r_q, max_steps=3, sampling_strategy='stochastic'):
    """
    从h出发，用策略网络采样一条路径

    参数说明：
    - h: 起始实体
    - r_q: 查询关系
    - max_steps: 最多走几步
    - sampling_strategy: 采样策略
        • 'greedy': 贪心，每步选概率最高的动作
        • 'stochastic': 随机采样，按概率分布采样
        • 'beam': Beam Search，保留top-K候选
    """
    current = h
    path = []
    path_history = []

    for step in range(max_steps):
        # Step 1: 获取当前节点的邻居（可选动作）
        action_space = model.graph.get_neighbors(current)
        # 返回: [(relation, next_entity), ...]
        # 例如: [(plays_for, Buccaneers), (born_in, USA), ...]

        if len(action_space) == 0:
            break  # 没有出边，停止

        # Step 2: 策略网络输出动作概率
        with torch.no_grad():
            action_probs = model.policy_network(
                current_entity=current,
                query_relation=r_q,
                path_history=path_history,
                action_space=action_space,
                rule_model=model
            )
            # 返回: [0.7, 0.2, 0.1] 每个动作的概率

        # Step 3: 根据采样策略选择动作
        # ⭐ 重要：采样策略在训练和推理阶段都会使用

        if sampling_strategy == 'greedy':
            # 贪心：选概率最高的
            # 用途：快速测试、确定性推理
            action_idx = torch.argmax(action_probs).item()

        elif sampling_strategy == 'stochastic':
            # 随机采样：按概率分布采样（推荐！）
            # 用途：训练阶段（必须用）+ 推理阶段（推荐用）
            action_idx = torch.multinomial(action_probs, 1).item()
            # 例如：
            # probs = [0.7, 0.2, 0.1]
            # 70%概率选索引0，20%选索引1，10%选索引2
            #
            # 为什么推荐：
            # - 多样性好，能探索不同路径
            # - 训练时必须用（论文公式5）
            # - 推理时效果最好（覆盖更全面）

        elif sampling_strategy == 'beam':
            # Beam Search需要维护多条候选路径
            # 用途：高精度要求场景
            # 暂略，较复杂

        # Step 4: 执行选择的动作
        relation, next_entity = action_space[action_idx]

        # 记录路径
        path.append((current, relation, next_entity))
        path_history.append((relation, next_entity))

        # 移动到下一个节点
        current = next_entity

    return path, current  # 返回路径和终点实体


def inference(model, query, num_samples=10, sampling_strategy='stochastic'):
    """
    推理：采样K条路径并聚合

    核心思想：
    - 多次采样能覆盖不同路径
    - 高频访问的实体更可能是正确答案
    """
    h, r_q = query
    model.eval()

    # ========== Step 1: 采样K条路径 ==========
    all_paths = []
    entity_visit_count = defaultdict(int)
    entity_visit_scores = defaultdict(float)

    for i in range(num_samples):
        # 采样一条路径
        path, final_entity = sample_one_path(
            model, h, r_q,
            max_steps=3,
            sampling_strategy=sampling_strategy
        )

        all_paths.append(path)

        # 统计终点实体
        entity_visit_count[final_entity] += 1

        # 可选：记录路径得分（用于方法B）
        path_score = compute_path_score(path, r_q, model)
        entity_visit_scores[final_entity] += path_score

    # ========== Step 2: 聚合路径结果 ==========
    # 有三种聚合方法，下面详细说明

    entity_scores = aggregate_paths(
        all_paths,
        entity_visit_count,
        entity_visit_scores,
        num_samples,
        method='A'  # 选择聚合方法
    )

    return entity_scores


def compute_path_score(path, r_q, model):
    """
    计算路径质量得分（用于聚合方法B）

    路径越"好"，得分越高
    可以考虑：
    - 路径长度（短路径更好？）
    - 规则匹配度（符合规则的路径更好）
    - 动作概率乘积（高概率路径更好）
    """
    # 方法1: 路径上动作概率的乘积
    prob_product = 1.0
    for step, (e_from, r, e_to) in enumerate(path):
        # 重新计算这步的动作概率
        action_space = model.graph.get_neighbors(e_from)
        action_probs = model.policy_network(...)
        idx = find_action_index((r, e_to), action_space)
        prob_product *= action_probs[idx]

    return prob_product

    # 方法2: 规则匹配度
    # match_score = compute_rule_match(path, r_q, model)
    # return match_score
```

**采样策略对比**：

| 策略 | 优点 | 缺点 | 适用场景 | 使用阶段 |
|------|------|------|---------|---------|
| **Greedy**<br/>（贪心） | • 确定性，可复现<br/>• 每次都选最优动作 | • 缺乏多样性<br/>• 多次采样可能得到相同路径 | 小规模测试、快速验证 | 仅推理阶段可选 |
| **Stochastic**<br/>（随机采样）| • 多样性好<br/>• 能探索不同路径<br/>• 覆盖更全面 | • 有噪声<br/>• 需要更多采样次数 | **推荐！** 正式推理 | ⭐ 训练阶段必须用<br/>⭐ 推理阶段推荐用 |
| **Beam Search** | • 平衡确定性和多样性<br/>• 保留多个候选 | • 实现复杂<br/>• 计算开销大 | 高精度要求场景 | 仅推理阶段可选 |

**重要说明 - 如何判断使用哪种策略**：

```python
# ========== 训练阶段（SL + RL）==========
# 固定使用 stochastic 策略（论文公式5）
at ∼ Categorical(πt)  # 必须按概率分布采样

# 原因：
# 1. 保证探索多样性
# 2. 梯度反向传播需要随机性
# 3. 防止过拟合到单一路径

# ========== 推理阶段 ==========
# 可以选择策略，推荐 stochastic

# 配置示例：
inference_config = {
    'sampling_strategy': 'stochastic',  # 推荐！
    'num_samples': 10,                  # 采样10条路径
}

# 论文SSRL的做法：
# - 训练：stochastic（论文第4页公式5）
# - 推理：stochastic（论文实验部分，多次采样聚合）
```

**与论文对比**：

论文《Knowledge Graph Reasoning with Self-supervised Reinforcement Learning》的采样策略：
- **SL阶段**：agent按策略网络采样动作（stochastic），如果动作标签为1则执行，否则停留
- **RL阶段**：按策略网络采样动作（stochastic），用于policy gradient更新
- **推理阶段**：论文未明确说明，但从beam search提及推测也使用stochastic多次采样

#### 3.4.2 路径聚合方法（问题2）

**为什么有A/B/C三种方法？不同应用场景需要不同聚合策略。**

```python
def aggregate_paths(all_paths, visit_count, visit_scores, num_samples,
                   method='A', model=None, query=None):
    """
    聚合K条路径的结果

    三种方法对比：
    A: 简单投票（频率统计）- 最简单
    B: 加权投票（考虑路径质量）- 更精细
    C: 集成KGE（结合知识图谱嵌入）- 最全面
    """

    if method == 'A':
        # ========== 方法A: 简单投票 ==========
        """
        思想：哪个实体被访问次数多，得分就高

        例子：
        - 10次采样
        - Todd_Bowles被访问8次 → score = 8/10 = 0.8
        - Tampa被访问1次 → score = 1/10 = 0.1

        优点：简单快速，无需额外计算
        缺点：不考虑路径质量（好路径坏路径同等对待）
        """
        entity_scores = {
            entity: count / num_samples
            for entity, count in visit_count.items()
        }

    elif method == 'B':
        # ========== 方法B: 加权投票 ==========
        """
        思想：考虑路径质量，好路径的贡献更大

        例子：
        - Path 1到Todd_Bowles：质量0.9 → 贡献0.9
        - Path 2到Todd_Bowles：质量0.6 → 贡献0.6
        - Todd_Bowles总得分 = (0.9 + 0.6) / 10 = 0.15

        路径质量如何衡量？
        • 动作概率乘积：π(a1) × π(a2) × π(a3)
        • 规则匹配度：路径符合规则的程度
        • 路径长度：短路径可能更直接

        优点：更精细，考虑路径质量
        缺点：需要额外计算，稍慢
        """
        entity_scores = {}

        for entity in visit_count.keys():
            # 找到所有到达这个实体的路径
            paths_to_entity = [
                path for path in all_paths
                if path[-1][2] == entity  # 终点是entity
            ]

            # 计算加权得分
            weighted_score = 0.0
            for path in paths_to_entity:
                # 路径质量
                quality = compute_path_score(path, query[1], model)
                weighted_score += quality

            # 归一化
            entity_scores[entity] = weighted_score / num_samples

    elif method == 'C':
        # ========== 方法C: 集成KGE ==========
        """
        思想：结合策略网络和KGE，优势互补

        策略网络：善于路径推理，但可能miss一些直接连接
        KGE嵌入：善于判断实体关系相似度

        集成：
        final_score = α × path_score + (1-α) × kge_score

        例子：
        - Todd_Bowles
          • path_score = 0.8（路径频繁访问）
          • kge_score = 0.7（嵌入相似度）
          • final = 0.6×0.8 + 0.4×0.7 = 0.76

        优点：最全面，结合两种推理方式
        缺点：需要额外KGE计算，最慢
        """
        h, r_q = query

        # 方法A的路径得分
        path_scores = {
            entity: count / num_samples
            for entity, count in visit_count.items()
        }

        # KGE得分（所有实体）
        kge_scores = model.compute_kge_scores(h, r_q)
        # 返回: {entity_id: kge_score, ...}

        # 集成
        alpha = 0.6  # 路径得分权重
        entity_scores = {}

        all_entities = set(path_scores.keys()) | set(kge_scores.keys())
        for entity in all_entities:
            path_score = path_scores.get(entity, 0.0)
            kge_score = kge_scores.get(entity, 0.0)

            entity_scores[entity] = (
                alpha * path_score + (1 - alpha) * kge_score
            )

    return entity_scores
```

**三种方法对比**：

| 方法 | 计算复杂度 | 精度 | 适用场景 | 推荐度 |
|------|----------|------|---------|--------|
| **A: 简单投票** | 低（只统计频率） | 中 | • 快速推理<br/>• 资源受限场景 | ⭐⭐⭐ |
| **B: 加权投票** | 中（需计算路径质量） | 高 | • 需要更高精度<br/>• 路径质量差异大时 | ⭐⭐⭐⭐ |
| **C: 集成KGE** | 高（需要KGE计算） | 最高 | • 追求最佳性能<br/>• 计算资源充足 | ⭐⭐⭐⭐⭐ |

**重要说明：三种方法是互斥的**

推理时，你需要**选择其中一种**聚合方法，而不是三种一起用：

```python
# ❌ 错误用法：不要三种方法混合使用
entity_scores_A = aggregate_paths(..., method='A')
entity_scores_B = aggregate_paths(..., method='B')
entity_scores_C = aggregate_paths(..., method='C')
final_scores = (entity_scores_A + entity_scores_B + entity_scores_C) / 3

# ✓ 正确用法：选择一种方法
entity_scores = aggregate_paths(..., method='B')  # 只用方法B

# 配置文件中设置
config = {
    'aggregation_method': 'B',  # 'A', 'B', 或 'C'
    'num_samples': 10
}
```

**如何选择方法**：
- **开发阶段**：用方法A快速验证模型是否有效
- **调优阶段**：尝试方法B看是否能提升性能
- **最终实验**：用方法C追求最佳结果（论文报告）

**推荐策略**：
- 开发/调试阶段：用方法A（快速迭代）
- 正式实验：用方法B或C（追求性能）
- 如果KGE很强：用方法C
- 如果只想用策略网络：用方法B

```python
# 完整推理代码
def inference(model, query, num_samples=10, device='cuda'):
    """
    推理：用策略网络多次采样+聚合

    Args:
        model: 训练好的RulE-SSRL模型
        query: (h, r_q, ?) 查询
        num_samples: 采样路径数量（K）
        device: 设备

    Returns:
        entity_scores: {entity_id: score} 每个实体的得分
    """
    h, r_q = query
    model.eval()

    # Step 1: 多次采样路径
    all_paths = []
    entity_visit_count = defaultdict(int)

    for _ in range(num_samples):
        with torch.no_grad():
            # 策略网络探索一条路径（随机采样）
            path, final_entity = sample_one_path(
                model, h, r_q,
                max_steps=3,
                sampling_strategy='stochastic'  # 推荐随机采样
            )

            all_paths.append(path)
            entity_visit_count[final_entity] += 1

    # Step 2: 聚合路径结果（选择方法）
    entity_scores = aggregate_paths(
        all_paths,
        entity_visit_count,
        num_samples,
        method='A'  # 可选：A/B/C
    )

    return entity_scores


def evaluate(model, test_queries, device='cuda'):
    """
    评估模型性能
    """
    metrics = {'hits@1': 0, 'hits@3': 0, 'hits@10': 0, 'mrr': 0}
    num_queries = len(test_queries)

    for h, r_q, t_true in test_queries:
        # 推理
        entity_scores = inference(model, (h, r_q), num_samples=10)

        # 排序
        sorted_entities = sorted(
            entity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 计算指标
        for rank, (entity, score) in enumerate(sorted_entities, 1):
            if entity == t_true:
                if rank <= 1:
                    metrics['hits@1'] += 1
                if rank <= 3:
                    metrics['hits@3'] += 1
                if rank <= 10:
                    metrics['hits@10'] += 1
                metrics['mrr'] += 1.0 / rank
                break

    # 归一化
    for key in metrics:
        metrics[key] /= num_queries

    return metrics
```

**推理优势**：
- ✓ 统一接口：只用策略网络，不需要Grounding模块
- ✓ 可扩展：可以调整采样次数平衡精度和速度
- ✓ 灵活：可以结合KGE、加权等多种策略

---

## 四、实验设计与预期效果

### 4.1 实验设置

```python
# 数据集配置
DATASETS = {
    'FB15K-237': {'entities': 14505, 'relations': 237, 'degree': 19.74},
    'WN18RR': {'entities': 40945, 'relations': 11, 'degree': 2.19},
    'NELL-995': {'entities': 75492, 'relations': 200, 'degree': 4.07},
    'UMLS': {'entities': 135, 'relations': 46, 'degree': 6.8}
}

# 模型配置
CONFIG = {
    'hidden_dim': 500,
    'gamma_fact': 6.0,
    'gamma_rule': 5.0,
    'mlp_rule_dim': 100,
    'policy_hidden_dim': 256,
    'max_path_length': 3,

    # SSRL特定参数
    'sl_epochs': 5,
    'rl_epochs': 10,
    'sl_lr': 1e-3,
    'rl_lr': 1e-4,
    'reward_shaping_weight': 0.1,
    'distillation_weight': 0.05
}
```

### 4.2 预期实验结果

| 数据集 | 方法 | Hits@1 | Hits@3 | Hits@10 | MRR |
|--------|------|--------|--------|---------|-----|
| FB15K-237 | RulE (baseline) | 24.5 | 35.8 | 48.2 | 32.1 |
| FB15K-237 | SSRL-MINERVA | 22.3 | 34.5 | 47.6 | 30.5 |
| FB15K-237 | **RulE-SSRL (ours)** | **26.8** | **38.2** | **51.5** | **34.8** |
| | | | | | |
| NELL-995 | RulE (baseline) | 68.2 | 78.5 | 83.1 | 74.5 |
| NELL-995 | SSRL-MINERVA | 71.4 | 80.5 | 83.5 | 76.2 |
| NELL-995 | **RulE-SSRL (ours)** | **73.8** | **82.3** | **86.2** | **78.9** |

### 4.3 消融实验设计

```
┌─────────────────────────────────────────────────────────────┐
│                      消融实验设计                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  实验1: SL预训练的影响                                        │
│  ├── w/o SL: 直接RL训练                                      │
│  ├── w/ random init: 随机初始化后SL                          │
│  └── w/ rule-weighted SL: 规则加权SL (完整方法)              │
│                                                             │
│  实验2: 奖励塑形的影响                                        │
│  ├── Terminal only: 仅终端奖励                               │
│  ├── + Distance reward: 加入距离奖励                         │
│  └── + Rule reward: 加入规则对齐奖励 (完整方法)              │
│                                                             │
│  实验3: 知识蒸馏的影响                                        │
│  ├── w/o distillation: 无蒸馏                                │
│  ├── Forward only: 仅前向蒸馏                                │
│  └── Bidirectional: 双向蒸馏 (完整方法)                      │
│                                                             │
│  实验4: 动作空间剪枝的影响                                    │
│  ├── No pruning: 完整动作空间                                │
│  ├── Random pruning: 随机剪枝                                │
│  └── Rule-guided pruning: 规则引导剪枝 (完整方法)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 五、实现路线图

### 5.1 开发阶段

```
Phase 1: 基础框架搭建 (Week 1-2)
├── 实现PolicyNetwork类
├── 实现BFS标签生成
├── 集成到现有RulE框架
└── 单元测试

Phase 2: 自监督预训练 (Week 3-4)
├── 实现JointPretrainingLoss
├── 实现规则加权标签生成
├── 调试联合训练流程
└── 在UMLS数据集验证

Phase 3: 强化学习集成 (Week 5-6)
├── 实现RuleGuidedReward
├── 实现RL训练循环
├── 实现层次化动作剪枝
└── 调试RL阶段

Phase 4: 知识蒸馏与优化 (Week 7-8)
├── 实现双向蒸馏机制
├── 实现Grounding阶段集成
├── 超参数调优
└── 全数据集实验

Phase 5: 实验与论文撰写 (Week 9-10)
├── 完整实验运行
├── 消融实验
├── 结果分析
└── 论文撰写
```

### 5.2 关键代码文件结构

```
src/
├── model.py              # 原RulE模型 (保留)
├── model_ssrl.py         # 新增: RulE-SSRL模型
│   ├── PolicyNetwork
│   ├── RuleGuidedReward
│   └── RulESSRL (主模型)
├── trainer.py            # 原训练器 (保留)
├── trainer_ssrl.py       # 新增: SSRL训练器
│   ├── SLPreTrainer
│   ├── RLTrainer
│   └── JointGroundTrainer
├── data.py               # 原数据加载 (保留)
├── data_ssrl.py          # 新增: SSRL数据处理
│   ├── BFSLabelGenerator
│   └── PathDataset
├── layers.py             # 原网络层 (保留)
├── layers_ssrl.py        # 新增: SSRL网络层
│   ├── HierarchicalActionPruning
│   └── BidirectionalDistillation
├── main.py               # 原入口 (保留)
└── main_ssrl.py          # 新增: SSRL入口
```

---

## 六、总结

### 6.1 方法总结

RulE-SSRL通过**规则引导的策略网络**实现RulE与SSRL的深度融合：

1. **三阶段训练**: 预训练（嵌入学习）→ 预计算规则质量 → 策略网络训练（规则质量加权）
2. **规则质量加权**: 利用预训练的 `rules_weight_emb` 区分高/低质量规则
3. **统一推理**: 去掉Grounding模块，统一用策略网络推理

**核心创新**：
- 规则嵌入不再只用于Grounding，而是直接在策略网络中作为"软建议"
- 规则质量加权：高质量规则获得更大引导权重，噪声规则影响被降低
- 不是组件叠加，而是真正的融合：`action_score = base_score + rule_bonus`
- 架构简洁：三阶段训练流程

### 6.2 预期贡献

- **理论贡献**: 提出规则引导的策略网络架构，规则作为软约束而非硬约束
- **方法贡献**: 规则注意力机制、规则监督的策略学习、联合训练框架
- **实验贡献**: 预期在多个KG数据集上达到或超越单独使用RulE或SSRL的性能

### 6.3 关键优势

| 维度 | RulE-SSRL | 纯RulE | 纯SSRL |
|------|----------|--------|--------|
| **推理效率** | 中（多次采样） | 高（一次传播） | 低（多次采样） |
| **推理覆盖** | 高（采样+规则引导） | 高（规则完整覆盖） | 中（受策略限制） |
| **灵活性** | 高（可突破规则） | 低（受规则限制） | 高（自由探索） |
| **可解释性** | 高（路径+规则） | 高（规则权重） | 高（路径） |
| **训练复杂度** | 中（三阶段） | 中（两阶段） | 高（BFS+SL+RL） |

### 6.4 最新改进：规则嵌入质量加权

**问题发现**：原实现中，规则打分和规则监督只使用规则的符号结构（关系ID序列），没有利用预训练学到的规则嵌入向量，导致所有规则被同等对待。

**改进方案**：利用预训练阶段计算的 `rules_weight_emb` 对不同质量的规则进行加权。

#### 6.4.1 规则质量权重来源

RulE预训练时通过 `add_ruleE_g` 计算每条规则的质量嵌入：

```python
# model.py: eval_compute_rule_weight()
# 预计算所有规则的质量嵌入
outputs = rule_body.sum(-2) + rule_embedding
dist = gamma_rule/hidden_dim - (outputs - embedding_r)^p
self.rules_weight_emb = ...  # [num_rules, hidden_dim]
```

**含义**：
- `rules_weight_emb[rule_id]` 是一个 `[hidden_dim]` 维向量
- 向量范数越大 → 规则体与规则头匹配越好 → 规则质量越高

#### 6.4.2 改进实现

**1. 策略网络训练前预计算规则质量**

```python
# trainer.py: PolicyTrainer.train()
# Step 1.5: 预计算规则质量权重
self.model.eval_compute_rule_weight(self.device)
```

**2. 规则打分中使用质量加权**

```python
# policy_network.py: _compute_rule_bonus_vectorized()
for rule_id, (r_head, r_body) in rules:
    # 获取规则质量权重
    rule_emb = rule_model.rules_weight_emb[rule_id]
    rule_quality = torch.norm(rule_emb).item()
    rule_quality = max(rule_quality, 0.1)  # 避免为0

    # 加权匹配
    match_mask = (next_relations[b] == expected_rel).float()
    weighted_match += rule_quality * match_mask
```

**3. 规则监督中使用质量加权**

```python
# policy_network.py: compute_rule_supervised_loss()
for rule_id, (r_head, r_body) in rules:
    # 获取规则质量权重
    rule_emb = model.rules_weight_emb[rule_id]
    rule_quality = torch.norm(rule_emb).item()

    # 加权监督信号
    match_mask = (next_relations[b] == expected_rel).float()
    target_mask[b] += rule_quality * match_mask  # 加权累加
```

#### 6.4.3 效果对比

| 改进前 | 改进后 |
|-------|-------|
| 规则1 (高质量): 权重=1 | 规则1 (高质量): 权重=0.85 |
| 规则2 (中质量): 权重=1 | 规则2 (中质量): 权重=0.52 |
| 规则3 (低质量): 权重=1 | 规则3 (低质量): 权重=0.18 |
| target_dist: [0.33, 0.33, 0.33] | target_dist: [0.55, 0.33, 0.12] |

**核心优势**：
- **高质量规则引导更强**：预训练学到的好规则获得更大权重
- **噪声规则影响降低**：低质量规则的监督信号被削弱
- **无需额外计算**：直接复用预训练已有的 `rules_weight_emb`

#### 6.4.4 完整训练流程

```
Phase 1: 预训练
├── 学习 entity_emb, relation_emb, rule_emb
└── 规则嵌入学习规则语义
    ↓
Phase 2: 策略网络训练
├── Step 1: 冻结预训练嵌入
├── Step 1.5: 预计算规则质量 (eval_compute_rule_weight)  ← 新增
├── Step 2: 训练循环
│   ├── 规则打分: 用 rules_weight_emb 加权  ← 改进
│   └── 规则监督: 用 rules_weight_emb 加权  ← 改进
└── Step 3: 保存最佳模型
    ↓
Phase 3: 推理
└── 策略网络多次采样 + 聚合
```

### 6.5 未来工作

1. 探索不同的规则监督策略（当前用规则作为软标签）
2. 研究规则注意力的可解释性分析
3. 扩展到更复杂的规则形式（带变量、时序规则）
4. 引入图神经网络增强实体表示
5. 研究few-shot/zero-shot场景下的泛化能力
