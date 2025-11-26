# RulE-RL 方案详细流程分析

本文档详细解析RulE-RL强化学习方案的每一步执行流程，包括训练和推理的完整过程。

---

## 目录

1. [训练流程详解](#一训练流程详解)
2. [推理流程详解](#二推理流程详解)
3. [核心组件交互](#三核心组件交互)
4. [完整案例演示](#四完整案例演示)

---

## 一、训练流程详解

### Step 1: 加载预训练RulE模型

```python
# 1. 加载知识图谱
graph = KnowledgeGraph(args.data_path)
# 包含：
# - entities.dict: 实体字典
# - relations.dict: 关系字典
# - train/valid/test.txt: 三元组数据
# - mined_rules.txt: 逻辑规则

# 2. 创建RulE模型
rule_model = RulE(
    graph=graph,
    hidden_dim=200,
    p_norm=2,
    gamma_fact=6,
    gamma_rule=5
)

# 3. 加载预训练权重
checkpoint = torch.load('../outputs/rule/checkpoint')
rule_model.load_state_dict(checkpoint['model'])
rule_model.eval()  # 设置为评估模式
```

**得到什么**：

| 组件 | 维度 | 内容 |
|------|------|------|
| `entity_embedding` | [num_entities, hidden_dim*2] | 实体的RotatE嵌入（复数形式） |
| `relation_embedding` | [num_relations, hidden_dim] | 关系的相位嵌入 |
| `rule_emb` | [num_rules, rule_dim] | 规则的嵌入表示 |

**为什么需要这些**：
- 实体嵌入：编码状态、计算距离奖励
- 关系嵌入：编码查询关系
- 规则嵌入：高层Agent选择规则、计算规则一致性奖励

---

### Step 2: 冻结RulE参数

```python
# 冻结所有预训练参数
for param in rule_model.entity_embedding.parameters():
    param.requires_grad = False

for param in rule_model.relation_embedding.parameters():
    param.requires_grad = False

rule_model.rule_emb.requires_grad = False
```

**为什么冻结**：
1. **保留预训练知识**：预训练嵌入已经学到了实体/关系的语义
2. **避免灾难性遗忘**：防止RL训练破坏原有嵌入质量
3. **加速训练**：只训练RL部分，参数量减少90%+
4. **稳定性**：固定表示空间，RL学习更稳定

---

### Step 3: 初始化RL组件

#### 3.1 StateEncoder（状态编码器）

```python
state_encoder = StateEncoder(
    entity_dim=hidden_dim * 2,   # 400 (复数)
    rel_dim=hidden_dim,           # 200
    rule_dim=100,                 # 规则特征维度
    history_dim=128               # 历史编码维度
)
```

**内部结构**：
```python
class StateEncoder(nn.Module):
    def __init__(self, entity_dim, rel_dim, rule_dim, history_dim):
        # 1. 实体编码器
        self.entity_encoder = nn.Linear(entity_dim, 128)

        # 2. 关系编码器
        self.relation_encoder = nn.Linear(rel_dim, 128)

        # 3. 规则上下文编码器（LSTM）
        self.rule_encoder = nn.LSTM(rule_dim, 128, batch_first=True)

        # 4. 历史路径编码器（GRU）
        self.history_encoder = nn.GRU(entity_dim + rel_dim, history_dim, batch_first=True)

        # 5. 状态融合
        self.state_fusion = nn.Sequential(
            nn.Linear(128 * 3 + history_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, current_entity, query_rel, rule_context, path_history):
        # 编码各个组件
        h_entity = self.entity_encoder(current_entity)          # [128]
        h_rel = self.relation_encoder(query_rel)                # [128]
        h_rule = self.rule_encoder(rule_context)[-1]            # [128]
        h_history = self.history_encoder(path_history)[-1]      # [128]

        # 拼接融合
        state = torch.cat([h_entity, h_rel, h_rule, h_history], dim=-1)  # [512]
        state_emb = self.state_fusion(state)                    # [128]

        return state_emb
```

**状态表示的含义**：
- `h_entity`: 当前在知识图谱的哪个位置
- `h_rel`: 要预测什么关系（目标）
- `h_rule`: 当前激活了哪些规则（指导信息）
- `h_history`: 走过哪些路径（避免循环）

#### 3.2 高层Agent - RuleSelectorAgent

```python
rule_selector = RuleSelectorAgent(
    query_dim=hidden_dim,         # 200
    rule_dim=100,
    num_rules=len(rules),         # 如UMLS: 18,400条规则
    hidden_dim=128
)
```

**内部结构**：
```python
class RuleSelectorAgent(nn.Module):
    def __init__(self, query_dim, rule_dim, num_rules, hidden_dim):
        # 1. 查询编码器：(h, r) → context vector
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim * 2, hidden_dim),  # h和r拼接
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 2. 规则-查询匹配网络
        self.rule_query_matcher = nn.Sequential(
            nn.Linear(hidden_dim + rule_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出匹配分数
        )

        # 3. UCB统计（用于探索）
        self.rule_counts = defaultdict(int)       # 每条规则被选次数
        self.rule_rewards = defaultdict(float)    # 累积奖励
        self.total_selections = 0
```

**选择规则的过程**：

```python
def forward(self, query_entity, query_relation, rule_embeddings, epsilon=0.1, top_k=5):
    # 1. 编码查询
    query_repr = torch.cat([query_entity, query_relation], dim=-1)
    query_emb = self.query_encoder(query_repr)  # [128]

    # 2. 计算每条规则的匹配分数
    rule_scores = []
    for rule_id in range(num_rules):
        rule_emb = rule_embeddings[rule_id]  # [100]
        combined = torch.cat([query_emb, rule_emb], dim=-1)  # [228]
        score = self.rule_query_matcher(combined)  # [1]
        rule_scores.append(score)

    rule_scores = torch.stack(rule_scores)  # [num_rules]

    # 3. 计算UCB分数（exploration bonus）
    ucb_scores = torch.zeros_like(rule_scores)
    for rule_id in range(num_rules):
        # 平均奖励
        avg_reward = self.rule_rewards[rule_id] / (self.rule_counts[rule_id] + 1)

        # UCB bonus: sqrt(2 * log(N) / n_i)
        ucb_bonus = sqrt(2 * log(self.total_selections + 1) / (self.rule_counts[rule_id] + 1))

        ucb_scores[rule_id] = rule_scores[rule_id] + ucb_bonus

    # 4. ε-greedy选择
    if random() < epsilon:
        # 探索：随机选择
        selected_rules = random.sample(range(num_rules), k=top_k)
    else:
        # 利用：选择UCB分数最高的
        _, selected_rules = torch.topk(ucb_scores, k=top_k)

    # 5. 计算选择概率（用于梯度更新）
    selection_probs = F.softmax(rule_scores[selected_rules], dim=0)

    return selected_rules, selection_probs
```

**为什么用UCB**：
- **问题**：ε-greedy探索效率低，可能重复探索差规则
- **UCB优势**：自动平衡探索-利用，选择次数少的规则会得到更高bonus
- **理论保证**：UCB有遗憾界保证，收敛到最优

#### 3.3 低层Agent - PathFinderAgent

```python
path_finder = PathFinderAgent(
    state_dim=128,               # StateEncoder输出
    action_dim=num_relations,    # 如UMLS: 46
    hidden_dim=256
)
```

**内部结构**：
```python
class PathFinderAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        # 1. 策略网络 π(a|s)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),      # 128 → 256
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),     # 256 → 256
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)      # 256 → num_relations
        )

        # 2. 价值网络 V(s) - 用于baseline减小方差
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),      # 128 → 256
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)               # 256 → 1
        )

    def select_action(self, state, action_mask=None, deterministic=False):
        # 1. 策略网络输出logits
        logits = self.policy_net(state)  # [num_relations]

        # 2. 应用动作掩码（只允许有效动作）
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        # 3. Softmax得到概率分布
        action_probs = F.softmax(logits, dim=-1)

        # 4. 选择动作
        if deterministic:
            action = torch.argmax(action_probs)  # 测试时用
        else:
            dist = Categorical(action_probs)
            action = dist.sample()  # 训练时采样

        # 5. 计算log概率（用于策略梯度）
        log_prob = torch.log(action_probs[action] + 1e-10)

        return action, log_prob
```

#### 3.4 环境 - KGReasoningEnv

```python
env = KGReasoningEnv(
    graph=graph,
    rule_model=rule_model,
    reward_calculator=reward_calculator,
    max_steps=5
)
```

**核心方法**：

```python
class KGReasoningEnv:
    def reset(self, query):
        """
        重置环境，开始新的episode

        Args:
            query: (head, relation, tail)

        Returns:
            state: 初始状态编码
        """
        self.query_head = query[0]
        self.query_rel = query[1]
        self.query_tail = query[2]

        # 初始化当前位置
        self.current_entity = self.query_head

        # 初始化轨迹
        self.path_history = []
        self.trajectory = [(self.current_entity, None)]

        # 步数
        self.step_count = 0

        # 编码初始状态
        state = self.state_encoder(
            current_entity=self.rule_model.entity_embedding.weight[self.current_entity],
            query_rel=self.rule_model.relation_embedding.weight[self.query_rel],
            rule_context=None,
            path_history=[]
        )

        return state

    def step(self, action, selected_rules):
        """
        执行动作，转移到下一个状态

        Args:
            action: 选择的关系ID
            selected_rules: 当前激活的规则

        Returns:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 1. 执行动作：沿着关系边移动
        next_entities = self.graph.get_neighbors(self.current_entity, action)

        if len(next_entities) == 0:
            # 死胡同
            return self._encode_state(), -0.2, True, {'reason': 'dead_end'}

        # 2. 随机选择一个邻居（如果有多个）
        next_entity = random.choice(next_entities)

        # 3. 更新路径
        self.path_history.append((
            self.rule_model.entity_embedding.weight[self.current_entity],
            self.rule_model.relation_embedding.weight[action]
        ))
        self.trajectory.append((next_entity, action))

        # 4. 更新位置
        self.current_entity = next_entity
        self.step_count += 1

        # 5. 编码新状态
        next_state = self._encode_state(selected_rules)

        # 6. 判断是否结束
        done = (self.step_count >= self.max_steps) or (next_entity == self.query_tail)

        # 7. 计算奖励
        if done:
            reward, breakdown = self.reward_calculator.compute_reward(
                self.trajectory, self.query_tail
            )
            info = {
                'reason': 'reached_target' if next_entity == self.query_tail else 'max_steps',
                'reward_breakdown': breakdown
            }
        else:
            reward = 0.0  # 中间步不给奖励（也可以给小奖励）
            info = {}

        return next_state, reward, done, info

    def get_action_mask(self, selected_rules):
        """
        获取有效动作掩码

        Returns:
            mask: [num_relations] bool tensor
        """
        # 1. 获取当前实体的出边关系
        outgoing_rels = self.graph.get_outgoing_relations(self.current_entity)

        # 2. 获取规则体中的关系
        rule_rels = set()
        for rule_id in selected_rules:
            rule = self.rule_model.rules[rule_id]
            rule_rels.update(rule['body'])

        # 3. 计算交集
        valid_rels = set(outgoing_rels) & rule_rels

        # 4. 生成mask
        mask = torch.zeros(self.graph.num_relations, dtype=torch.bool)
        for rel in valid_rels:
            mask[rel] = True

        return mask
```

#### 3.5 奖励计算器 - RewardCalculator

```python
reward_calculator = RewardCalculator(
    rule_model=rule_model,
    alpha=0.1    # 中间奖励权重
)
```

**完整奖励计算**：

```python
class RewardCalculator:
    def compute_reward(self, trajectory, target_entity):
        """
        计算轨迹的总奖励

        Args:
            trajectory: [(entity, relation), ...] 路径
            target_entity: 目标实体

        Returns:
            total_reward: 总奖励
            reward_breakdown: 奖励分解（用于分析）
        """
        rewards = {}

        # ===== 1. 最终奖励（最重要） =====
        final_entity = trajectory[-1][0]
        rewards['final'] = 1.0 if final_entity == target_entity else 0.0

        # ===== 2. 规则一致性奖励（中间） =====
        # 检查路径是否符合某条规则
        path_relations = [step[1] for step in trajectory if step[1] is not None]
        matched_rules = self._find_matching_rules(path_relations)

        if matched_rules:
            # 使用规则置信度作为奖励
            confidences = [self._get_rule_confidence(r) for r in matched_rules]
            rewards['rule_consistency'] = max(confidences)
        else:
            rewards['rule_consistency'] = 0.0

        # ===== 3. 接近目标奖励（中间，仅失败时启用） =====
        getting_closer_sum = 0.0
        start_entity = trajectory[0][0]
        dist_start = self._embedding_distance(start_entity, target_entity) + 1e-9
        for i in range(1, len(trajectory)):
            curr_entity = trajectory[i][0]
            prev_entity = trajectory[i-1][0]

            dist_curr = self._embedding_distance(curr_entity, target_entity)
            dist_prev = self._embedding_distance(prev_entity, target_entity)

            improvement = dist_prev - dist_curr
            getting_closer_sum += max(0, improvement)

        rewards['getting_closer'] = min(1.0, getting_closer_sum / dist_start)

        # ===== 加权求和 =====
        total_reward = (
            rewards['final'] +
            self.alpha * (
                rewards['rule_consistency'] +
                (1 - rewards['final']) * rewards['getting_closer']
            )
        )

        return total_reward, rewards

    def _find_matching_rules(self, path_relations):
        """查找匹配路径的规则"""
        matched = []
        for rule in self.rule_model.rules:
            if rule['body'] == path_relations:
                matched.append(rule)
        return matched

    def _get_rule_confidence(self, rule):
        """获取规则置信度（从RulE预训练得到）"""
        rule_emb = self.rule_model.rule_emb[rule['id']]
        head_emb = self.rule_model.relation_embedding.weight[rule['head']]

        # 规则置信度 = gamma - distance
        body_sum = sum([self.rule_model.relation_embedding.weight[r]
                        for r in rule['body']])
        confidence = self.gamma_rule - torch.norm(body_sum + rule_emb - head_emb)
        return confidence.item()

    def _embedding_distance(self, entity1, entity2):
        """计算实体嵌入距离"""
        emb1 = self.rule_model.entity_embedding.weight[entity1]
        emb2 = self.rule_model.entity_embedding.weight[entity2]
        return torch.norm(emb1 - emb2, p=2).item()
```

---

### Step 4: Episode循环训练

#### 完整的训练流程

```python
def train_episode(query, epsilon=0.1):
    """
    训练一个episode

    Args:
        query: (head, relation, tail)
        epsilon: 探索概率

    Returns:
        episode_reward: episode总奖励
        episode_length: episode长度
        loss_dict: 损失字典
    """

    # ========== 第4a步：输入查询 ==========
    head, relation, tail = query

    # ========== 第4b步：高层Agent选择规则 ==========

    # 获取嵌入
    query_entity_emb = rule_model.entity_embedding.weight[head]
    query_rel_emb = rule_model.relation_embedding.weight[relation]
    rule_embeddings = rule_model.rule_emb

    # 高层Agent选择Top-K规则
    selected_rules, selection_probs = rule_selector(
        query_entity_emb,
        query_rel_emb,
        rule_embeddings,
        epsilon=epsilon,
        top_k=5
    )

    print(f"Selected rules: {selected_rules.tolist()}")
    # 例如: [234, 567, 1023, 89, 456]

    # ========== 第4c步：低层Agent寻找路径 ==========

    # 重置环境
    state = env.reset(query)

    # 记录episode数据
    episode_data = {
        'states': [],
        'actions': [],
        'log_probs': [],
        'rewards': []
    }

    done = False
    episode_reward = 0.0

    while not done:
        # 1. 获取有效动作掩码
        action_mask = env.get_action_mask(selected_rules)
        print(f"  Current entity: {env.current_entity}")
        print(f"  Valid actions: {action_mask.nonzero().squeeze().tolist()}")

        # 2. 低层Agent选择动作
        action, log_prob = path_finder.select_action(state, action_mask)
        print(f"  Selected action: {action.item()}")

        # 3. 环境执行动作
        next_state, reward, done, info = env.step(action.item(), selected_rules)
        print(f"  Reward: {reward:.4f}, Done: {done}")

        # 4. 记录数据
        episode_data['states'].append(state)
        episode_data['actions'].append(action.item())
        episode_data['log_probs'].append(log_prob)
        episode_data['rewards'].append(reward)

        episode_reward += reward
        state = next_state

    print(f"Episode finished: total_reward={episode_reward:.4f}")

    # ========== 第4d步：更新两个Agent ==========

    # --- 更新低层Agent（PathFinder）---

    # 1. 计算折扣回报
    returns = []
    G = 0
    for r in reversed(episode_data['rewards']):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)

    # 2. 标准化回报（减小方差）
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # 3. 计算状态价值（baseline）
    states = torch.stack(episode_data['states'])
    values = path_finder.value_net(states).squeeze()

    # 4. 计算优势函数
    advantages = returns - values.detach()

    # 5. Policy loss (REINFORCE with baseline)
    log_probs = torch.stack(episode_data['log_probs'])
    policy_loss = -(log_probs * advantages).mean()

    # 6. Value loss
    value_loss = F.mse_loss(values, returns)

    # 7. 更新策略网络
    policy_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(path_finder.policy_net.parameters(), max_norm=1.0)
    policy_optimizer.step()

    # 8. 更新价值网络
    value_optimizer.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(path_finder.value_net.parameters(), max_norm=1.0)
    value_optimizer.step()

    # --- 更新高层Agent（RuleSelector）---

    # 1. 更新UCB统计
    for rule_id in selected_rules:
        rule_selector.update_statistics(rule_id.item(), episode_reward)

    # 2. 策略梯度更新
    rule_selector_loss = -torch.sum(torch.log(selection_probs + 1e-10)) * episode_reward
    rule_selector_optimizer.zero_grad()
    rule_selector_loss.backward()
    rule_selector_optimizer.step()

    loss_dict = {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'rule_selector_loss': rule_selector_loss.item(),
        'avg_return': returns.mean().item()
    }

    return episode_reward, len(episode_data['states']), loss_dict
```

#### 完整训练循环

```python
def train(train_queries, num_epochs=100):
    """
    完整训练流程
    """
    print("Starting RulE-RL training...")

    for epoch in range(num_epochs):
        # 课程学习：逐步减小epsilon
        epsilon = max(0.05, 0.5 - epoch * 0.01)

        epoch_rewards = []
        epoch_lengths = []

        for i, query in enumerate(train_queries):
            # 训练一个episode
            reward, length, loss_dict = train_episode(query, epsilon)

            epoch_rewards.append(reward)
            epoch_lengths.append(length)

            # 日志
            if (i + 1) % args.log_interval == 0:
                avg_reward = np.mean(epoch_rewards[-args.log_interval:])
                avg_length = np.mean(epoch_lengths[-args.log_interval:])

                print(f"Epoch {epoch}, Query {i+1}/{len(train_queries)}: "
                      f"Avg Reward = {avg_reward:.4f}, "
                      f"Avg Length = {avg_length:.2f}, "
                      f"Policy Loss = {loss_dict['policy_loss']:.4f}")

        # Epoch总结
        avg_epoch_reward = np.mean(epoch_rewards)
        avg_epoch_length = np.mean(epoch_lengths)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Avg Reward: {avg_epoch_reward:.4f}")
        print(f"  Avg Length: {avg_epoch_length:.2f}")
        print(f"  Epsilon: {epsilon:.3f}")

        # 验证
        if (epoch + 1) % args.eval_interval == 0:
            val_metrics = evaluate(graph.valid_triplets)
            print(f"  Validation MRR: {val_metrics['mrr']:.4f}")
            print(f"  Validation Hits@10: {val_metrics['hits@10']:.4f}")

        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(f"{args.save_path}/checkpoint_epoch_{epoch}.pt")
```

---

### Step 5: 保存检查点

```python
def save_checkpoint(path):
    """保存模型检查点"""
    torch.save({
        # 模型
        'rule_selector': rule_selector.state_dict(),
        'path_finder': path_finder.state_dict(),

        # 优化器
        'rule_selector_optimizer': rule_selector_optimizer.state_dict(),
        'policy_optimizer': policy_optimizer.state_dict(),
        'value_optimizer': value_optimizer.state_dict(),

        # UCB统计
        'rule_counts': dict(rule_selector.rule_counts),
        'rule_rewards': dict(rule_selector.rule_rewards),
        'total_selections': rule_selector.total_selections,

        # 其他
        'epoch': epoch,
        'args': args
    }, path)
    print(f"Checkpoint saved to {path}")
```

---

## 二、推理流程详解

### 评估完整流程

```python
def evaluate(test_queries, deterministic=True):
    """
    评估模型

    Args:
        test_queries: 测试查询列表
        deterministic: 是否确定性选择（测试时为True）

    Returns:
        metrics: 评估指标
    """
    rule_selector.eval()
    path_finder.eval()

    ranks = []

    with torch.no_grad():
        for query in test_queries:
            head, relation, tail = query

            # ===== Step 1: 高层Agent选择规则（不探索） =====
            query_entity_emb = rule_model.entity_embedding.weight[head]
            query_rel_emb = rule_model.relation_embedding.weight[relation]

            selected_rules, _ = rule_selector(
                query_entity_emb,
                query_rel_emb,
                rule_model.rule_emb,
                epsilon=0.0,  # 测试时不探索
                top_k=5
            )

            # ===== Step 2: 对所有候选实体评分 =====
            candidate_scores = []

            for candidate in range(graph.num_entities):
                # 修改查询，让tail=candidate
                test_query = (head, relation, candidate)

                # 运行一个episode到这个候选
                state = env.reset(test_query)
                done = False
                path_score = 0.0

                while not done:
                    # 获取有效动作
                    action_mask = env.get_action_mask(selected_rules)

                    # Agent选择动作（确定性）
                    action, _ = path_finder.select_action(
                        state,
                        action_mask,
                        deterministic=True  # 选概率最大的
                    )

                    # 执行动作
                    next_state, reward, done, _ = env.step(action.item(), selected_rules)
                    path_score += reward
                    state = next_state

                candidate_scores.append(path_score)

            # ===== Step 3: 计算排名 =====
            candidate_scores = torch.tensor(candidate_scores)

            # 过滤掉训练集中出现的三元组
            filter_mask = get_filter_mask(head, relation, graph)
            candidate_scores[filter_mask] = -1e9

            # 排序
            _, sorted_indices = torch.sort(candidate_scores, descending=True)

            # 找到真实答案的排名
            rank = (sorted_indices == tail).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

    # ===== Step 4: 计算评估指标 =====
    ranks = torch.tensor(ranks, dtype=torch.float)

    metrics = {
        'mrr': (1.0 / ranks).mean().item(),
        'mr': ranks.mean().item(),
        'hits@1': (ranks <= 1).float().mean().item(),
        'hits@3': (ranks <= 3).float().mean().item(),
        'hits@10': (ranks <= 10).float().mean().item()
    }

    rule_selector.train()
    path_finder.train()

    return metrics
```

---

## 三、核心组件交互

### 3.1 数据流图

```
查询 (h, r, ?)
    │
    ├─────────────────────────────┐
    │                             │
    ▼                             ▼
预训练嵌入                    高层Agent
entity_embedding            RuleSelectorAgent
relation_embedding              │
rule_emb                        │ UCB选择
    │                           │
    │                           ▼
    │                      Selected Rules
    │                      [R1, R2, R3, R4, R5]
    │                           │
    ├───────────┬───────────────┤
    │           │               │
    ▼           ▼               ▼
StateEncoder    │           ActionMask
    │           │               │
    │           ▼               │
    │      环境 (Env)            │
    │      reset()              │
    │           │               │
    ▼           ▼               ▼
State ────→ PathFinder ←──── Mask
                │
                ▼
            Action
                │
                ▼
            Env.step()
                │
                ├─────────┬─────────┐
                │         │         │
                ▼         ▼         ▼
          Next State  Reward    Done?
                │         │         │
                │         ▼         │
                │   RewardCalculator │
                │         │         │
                └─────────┴─────────┘
                          │
                          ▼
                    更新Agents
```

### 3.2 训练时的交互顺序

```
时间步 t:
  1. StateEncoder编码当前状态
  2. PathFinder选择动作
  3. Env执行动作，转移状态
  4. 记录 (s_t, a_t, log_p_t)

时间步 t+1:
  重复上述过程...

Episode结束:
  1. RewardCalculator计算总奖励
  2. 计算折扣回报
  3. 更新PathFinder (REINFORCE)
  4. 更新RuleSelector (Policy Gradient)
  5. 更新UCB统计
```

---

## 四、完整案例演示

### 案例：查询 (张三, grandfather, ?)

#### 输入

```python
query = (entity_id['张三'], relation_id['grandfather'], entity_id['王五'])
# (15, 3, 89)
```

#### Step 1: 高层Agent选择规则

```python
# 获取嵌入
h_emb = entity_embedding[15]   # 张三的嵌入
r_emb = relation_embedding[3]  # grandfather的嵌入

# 高层Agent forward
query_repr = concat([h_emb, r_emb])  # [400]
query_emb = query_encoder(query_repr)  # [128]

# 计算每条规则的得分
for rule_id, rule_emb in enumerate(rule_embeddings):
    combined = concat([query_emb, rule_emb])  # [228]
    score = rule_query_matcher(combined)  # scalar

    # 加上UCB bonus
    ucb_bonus = sqrt(2 * log(1000) / (rule_counts[rule_id] + 1))
    ucb_score = score + ucb_bonus

# 选择Top-5
selected_rules = [234, 567, 1023, 89, 456]
# 对应规则：
# 234: father ∧ father → grandfather
# 567: mother ∧ father → grandfather
# 1023: parent ∧ parent → grandparent
# ...
```

#### Step 2: 初始化环境

```python
state = env.reset(query)

# 内部状态:
env.current_entity = 15  # 张三
env.query_rel = 3        # grandfather
env.query_tail = 89      # 王五
env.trajectory = [(15, None)]
env.path_history = []
```

#### Step 3: Episode执行

**时间步 0**:

```python
# 编码状态
state = StateEncoder(
    current_entity=entity_embedding[15],  # 张三
    query_rel=relation_embedding[3],      # grandfather
    rule_context=rule_embeddings[[234, 567, 1023, 89, 456]],  # 选中的规则
    path_history=[]
)
# state: [128]

# 获取有效动作
outgoing_rels = graph.get_outgoing_relations(15)  # [father, mother, sibling, ...]
rule_rels = {father, mother, parent}  # 从选中规则的body提取
valid_actions = {father, mother}  # 交集

action_mask = [True, True, False, False, ...]  # father=0, mother=1有效

# Agent选择动作
logits = policy_net(state)  # [num_relations]
logits[~action_mask] = -inf
probs = softmax(logits)  # [0.7, 0.3, 0, 0, ...]
action = sample(probs)  # 采样得到 0 (father)

# 执行动作
next_entities = graph.get_neighbors(15, 0)  # [45] (李四)
next_entity = 45

# 更新状态
env.current_entity = 45
env.trajectory = [(15, None), (45, 0)]
env.step_count = 1

# 奖励
reward = 0.0  # 中间步
done = False
```

**时间步 1**:

```python
# 当前在李四 (45)
state = StateEncoder(
    current_entity=entity_embedding[45],  # 李四
    query_rel=relation_embedding[3],      # grandfather
    rule_context=rule_embeddings[[234, 567, 1023, 89, 456]],
    path_history=[(entity_embedding[15], relation_embedding[0])]  # 张三→father
)

# 有效动作
outgoing_rels = {father, mother, sibling}
rule_rels = {father, mother, parent}
valid_actions = {father, mother}

# Agent选择
action = 0  # father again

# 执行
next_entity = 89  # 王五！

# 更新
env.current_entity = 89
env.trajectory = [(15, None), (45, 0), (89, 0)]
env.step_count = 2

# 奖励（episode结束）
done = True  # 到达目标
reward, breakdown = reward_calculator.compute_reward(trajectory, 89)
```

#### Step 4: 奖励计算

```python
trajectory = [(15, None), (45, 0), (89, 0)]
# 张三 --father--> 李四 --father--> 王五

# 1. 最终奖励
final_entity = 89
target = 89
rewards['final'] = +1.0  # 到达目标！

# 2. 规则一致性
path_relations = [0, 0]  # [father, father]
matched_rules = find_rules([father, father])  # 找到规则234: father∧father→grandfather
rule_confidence = 0.85
rewards['rule_consistency'] = 0.85

# 3. 接近目标
# Step 0→1: dist(张三, 王五) → dist(李四, 王五)
#          5.2 → 2.8 = improvement 2.4
# Step 1→2: dist(李四, 王五) → dist(王五, 王五)
#          2.8 → 0.0 = improvement 2.8
rewards['getting_closer'] = 2.4 + 2.8 = 5.2

# 4. 多样性
unique_entities = 3  # 张三, 李四, 王五
unique_relations = 1  # father
rewards['diversity'] = 0.04

# 5. 惩罚项
rewards['dead_end'] = 0     # 王五有出边
rewards['loop'] = 0         # 没有重复
rewards['length'] = 0       # 长度=2 < 5

# 总奖励
total = 1.0 + 0.1*(0.85 + 5.2 + 0.04) - 0.05*0 = 1.609
```

#### Step 5: 更新Agents

```python
# 低层Agent更新
episode_data = {
    'states': [state_0, state_1],
    'actions': [0, 0],  # [father, father]
    'log_probs': [log_p_0, log_p_1],
    'rewards': [0.0, 1.609]
}

# 计算折扣回报
G_1 = 1.609
G_0 = 0.0 + 0.99 * 1.609 = 1.593

returns = [1.593, 1.609]

# 优势函数
values = value_net([state_0, state_1])  # [0.5, 0.8]
advantages = [1.593 - 0.5, 1.609 - 0.8] = [1.093, 0.809]

# 策略梯度
policy_loss = -(log_p_0 * 1.093 + log_p_1 * 0.809)

# 更新
policy_optimizer.step()

# 高层Agent更新
for rule_id in [234, 567, 1023, 89, 456]:
    rule_counts[rule_id] += 1
    rule_rewards[rule_id] += 1.609

rule_selector_loss = -log(selection_probs) * 1.609
rule_selector_optimizer.step()
```

---

## 五、关键技术点总结

### 5.1 为什么需要状态编码器

**问题**: 状态信息复杂，直接拼接维度爆炸

**解决**:
- 实体编码器：提取位置信息
- 关系编码器：提取目标信息
- 规则编码器（LSTM）：提取规则序列模式
- 历史编码器（GRU）：提取路径记忆

### 5.2 为什么需要动作掩码

**问题**: 大量无效动作浪费探索

**解决**:
- 只允许：当前实体有出边 AND 规则体包含的关系
- 减少动作空间90%+
- 加速学习

### 5.3 为什么需要奖励塑形

**问题**: 稀疏奖励导致学习困难（99%的episode得0分）

**解决**:
- 最终奖励：主要信号
- 规则一致性：利用符号知识引导
- 接近目标：利用嵌入空间引导
- 惩罚项：避免不良行为

### 5.4 为什么需要baseline

**问题**: 策略梯度方差大，训练不稳定

**解决**:
- Value Network估计状态价值
- 优势函数 = 回报 - baseline
- 减小方差，加速收敛

### 5.5 为什么需要UCB

**问题**: ε-greedy探索效率低

**解决**:
- UCB自动平衡探索-利用
- 选择少的规则得到bonus
- 理论收敛保证

---

## 六、训练日志示例

```
Starting RulE-RL training...

Epoch 0, Query 100/1000:
  Selected rules: [234, 567, 1023, 89, 456]
  Step 0: entity=15 → action=0 (father) → entity=45
  Step 1: entity=45 → action=0 (father) → entity=89 ✓
  Reward breakdown:
    final: 1.0
    rule_consistency: 0.85
    getting_closer: 5.2
    diversity: 0.04
  Total reward: 1.609
  Policy loss: 0.234

Epoch 0, Query 200/1000:
  Avg Reward = 0.524
  Avg Length = 3.2
  Policy Loss = 0.456

Epoch 0 Summary:
  Avg Reward: 0.412
  Avg Length: 3.5
  Epsilon: 0.500

Epoch 5:
  Validation MRR: 0.456
  Validation Hits@10: 0.672

...

Epoch 50:
  Avg Reward: 0.823
  Validation MRR: 0.789

Final Test Results:
  MRR: 0.912
  Hits@1: 0.834
  Hits@3: 0.921
  Hits@10: 0.967
```

---

## 七、常见问题解答 (FAQ)

### Q1: 什么是"冻结RulE参数"?

**答案**: 冻结参数是深度学习中的迁移学习技术。

**详细解释**:

```python
# 冻结前（参数可训练）
entity_embedding.requires_grad = True  # 梯度会被计算和更新

# 冻结后（参数固定）
entity_embedding.requires_grad = False  # 梯度不计算，参数不更新
```

**具体到RulE-RL**:
1. **冻结的参数** (来自预训练RulE):
   - `entity_embedding`: 实体嵌入 [num_entities × 400]
   - `relation_embedding`: 关系嵌入 [num_relations × 200]
   - `rule_emb`: 规则嵌入 [num_rules × 100]

2. **为什么冻结**:
   - 这些参数已经通过预训练学到了知识图谱的语义表示
   - RL训练过程中梯度不会反向传播到这些参数
   - 它们只用于**提供特征**（作为输入），不会被修改

3. **类比理解**:
   ```
   预训练RulE模型 = 一本已经写好的知识百科全书
   冻结参数     = 把这本书设为"只读"，不允许修改
   RL训练      = 在这本书的基础上学习如何查阅和使用它
   ```

---

### Q2: "只训练RL部分，参数量减少90%+" 是什么意思?

**答案**: 指相比训练整个模型，只训练RL组件大幅减少了需要优化的参数数量。

**参数量对比**:

| 组件 | 参数量 (UMLS数据集) | 是否训练 |
|------|---------------------|----------|
| **预训练RulE部分** | | |
| entity_embedding | 135 × 400 = 54,000 | ❌ 冻结 |
| relation_embedding | 46 × 200 = 9,200 | ❌ 冻结 |
| rule_emb | 18,400 × 100 = 1,840,000 | ❌ 冻结 |
| **小计** | **1,903,200** | |
| | | |
| **RL新增部分** | | |
| RuleSelector (query_encoder + matcher) | ~40,000 | ✅ 训练 |
| PathFinder (policy_net) | 128×256 + 256×256 + 256×46 = ~110,000 | ✅ 训练 |
| PathFinder (value_net) | 128×256 + 256×1 = ~33,000 | ✅ 训练 |
| StateEncoder | ~50,000 | ✅ 训练 |
| **小计** | **~233,000** | |

**计算**:
```
可训练参数占比 = 233,000 / (1,903,200 + 233,000) = 10.9%
减少的参数量 = 100% - 10.9% = 89.1% ≈ 90%+
```

**实际意义**:
- 训练速度提升: 只更新10%的参数，每次迭代快很多
- 内存占用降低: 不需要存储大部分参数的梯度
- 训练稳定: 固定的表示空间让RL学习更容易收敛

---

### Q3: State (状态) 包含哪些东西?

**答案**: State是RL Agent当前所处环境的完整描述，包含4个核心信息。

**State的组成**:

```python
state = StateEncoder(
    current_entity,    # 1. 当前位置
    query_rel,         # 2. 查询目标
    rule_context,      # 3. 规则上下文
    path_history       # 4. 历史路径
)
# 输出: 128维向量
```

**详细分解**:

| 组件 | 含义 | 维度 | 示例 (grandfather查询) |
|------|------|------|----------------------|
| **current_entity** | Agent当前在知识图谱的哪个实体 | 400 | 张三的嵌入 |
| **query_rel** | 要预测的关系（目标） | 200 | grandfather关系的嵌入 |
| **rule_context** | 高层Agent选择的Top-K规则 | 100×K | [father∧father→grandfather, ...] |
| **path_history** | 已经走过的路径 | 变长 | [(张三,father), (李四,father)] |

**编码过程**:

```python
# 1. 当前实体编码
h_entity = Linear(current_entity)  # 400 → 128

# 2. 查询关系编码
h_rel = Linear(query_rel)  # 200 → 128

# 3. 规则上下文编码 (用LSTM处理序列)
h_rule = LSTM(rule_context)  # [K×100] → 128

# 4. 历史路径编码 (用GRU处理序列)
h_history = GRU(path_history)  # [T×(400+200)] → 128

# 5. 融合所有信息
state = MLP(concat[h_entity, h_rel, h_rule, h_history])
# [128+128+128+128] = 512 → 128
```

**为什么需要这些信息**:
- `current_entity`: 知道从哪里出发
- `query_rel`: 知道往哪里去（目标）
- `rule_context`: 知道应该遵循什么规则（指导）
- `path_history`: 知道已经走过哪里（避免循环）

**类比理解**:
```
就像你在迷宫中寻路:
- current_entity = 你现在的位置
- query_rel = 出口的方向
- rule_context = 地图上标记的推荐路线
- path_history = 你走过的脚印（避免走回头路）
```

---

### Q4: 环境模块 (Environment) 是什么?

**答案**: 环境是强化学习的核心概念，模拟Agent与外界交互的系统。

**标准RL框架**:
```
┌─────────┐  action   ┌─────────────┐
│  Agent  │ ───────→  │ Environment │
│         │  ←───────  │             │
└─────────┘   state    └─────────────┘
              reward
              done
```

**在RulE-RL中的实现**:

```python
class KGReasoningEnv:
    """知识图谱推理环境"""

    def __init__(self, graph, rule_model, max_steps=5):
        self.graph = graph              # 知识图谱（图结构）
        self.rule_model = rule_model    # 用于获取嵌入
        self.max_steps = max_steps      # 最大步数限制

    def reset(self, query):
        """开始新episode，返回初始状态"""
        self.current_entity = query[0]   # 从头实体开始
        self.query_tail = query[2]       # 目标实体
        self.trajectory = []             # 清空轨迹
        return self._encode_state()      # 返回初始状态编码

    def step(self, action):
        """执行动作，返回下一个状态和奖励"""
        # 1. 根据action (关系ID) 移动到下一个实体
        next_entity = self._move(self.current_entity, action)

        # 2. 更新当前位置
        self.current_entity = next_entity
        self.trajectory.append((next_entity, action))

        # 3. 判断是否结束
        done = (next_entity == self.query_tail) or (len(self.trajectory) >= self.max_steps)

        # 4. 计算奖励
        reward = self._compute_reward() if done else 0.0

        # 5. 编码新状态
        next_state = self._encode_state()

        return next_state, reward, done, {}
```

**环境的职责**:

1. **维护状态**: 跟踪Agent当前位置、走过的路径
2. **执行动作**: 根据Agent选择的关系边进行移动
3. **状态转移**: 从当前实体移动到下一个实体
4. **判断终止**: 检查是否到达目标或超过最大步数
5. **计算奖励**: 根据轨迹质量给予反馈

**具体到知识图谱**:

```python
# 环境 = 知识图谱 + 图遍历规则

示例：
  当前状态: 在实体"张三"
  可选动作: [father, mother, spouse, ...] (出边关系)
  执行动作: 选择 "father"
  状态转移: 张三 --father--> 李四
  新状态:   在实体"李四"
```

**环境使用的数据结构**:

```python
# 知识图谱的表示
graph.adjacency_list = {
    15: {  # 实体ID: 张三
        0: [45, 46],  # 关系father连接到 [李四, 王四]
        1: [50],      # 关系mother连接到 [赵五]
        ...
    },
    45: {  # 实体ID: 李四
        0: [89],      # 关系father连接到 [王五]
        ...
    }
}

# 环境使用邻接表进行状态转移
def _move(current_entity, action_relation):
    neighbors = graph.adjacency_list[current_entity][action_relation]
    return random.choice(neighbors)  # 如果有多个邻居，随机选一个
```

---

### Q5: 这个强化学习模块是标准的RL策略吗？有没有加入创新？

**答案**: 基于标准RL算法，但针对知识图谱推理任务进行了多处创新。

**使用的标准RL技术**:

| 技术 | 说明 | 来源 |
|------|------|------|
| **REINFORCE** | 基础策略梯度算法 | Williams 1992 |
| **Baseline (Value Network)** | 减小方差 | 标准Actor-Critic |
| **ε-greedy** | 探索策略 | 经典RL |
| **UCB (Upper Confidence Bound)** | 用于规则选择 | Bandit算法 |

**创新点**:

#### 创新1: 层次化强化学习 (Hierarchical RL)
```
传统RL: 单一Agent做所有决策

RulE-RL: 两层Agent协作
  - 高层Agent: 选择规则 (策略空间)
  - 低层Agent: 路径探索 (动作空间)

优势: 降低复杂度 O(规则数 × 关系数) → O(规则数) + O(关系数)
```

#### 创新2: 符号-神经混合奖励塑形
```python
# 传统RL奖励: 只有最终奖励 (0或1)
reward = 1 if reached_target else 0

# RulE-RL奖励: 结合符号知识和神经嵌入
reward = final_reward  # 最终奖励
       + α × rule_consistency_reward  # 符号规则一致性
       + α × getting_closer_reward    # 神经嵌入距离
       + α × diversity_reward         # 探索多样性
       - β × penalties                # 惩罚项
```

**创新**: 利用预训练的规则嵌入和实体嵌入作为奖励信号

#### 创新3: 动作掩码 (Action Masking)
```python
# 传统RL: 所有动作都可选
action_space = [所有46个关系]

# RulE-RL: 动态约束动作空间
valid_actions = 当前实体的出边 ∩ 选中规则的体
action_space = [2-5个关系]  # 大幅减少

优势: 减少无效探索90%+，加速学习
```

#### 创新4: 规则引导的状态表示
```python
# 传统图推理: State = 当前实体嵌入
state = entity_embedding[current]

# RulE-RL: State = 实体 + 关系 + 规则上下文 + 历史
state = StateEncoder(
    current_entity,  # 位置
    query_rel,       # 目标
    rule_context,    # 规则指导 ← 创新
    path_history     # 历史
)

优势: 利用符号规则指导神经探索
```

#### 创新5: 知识迁移 (Transfer Learning)
```python
# 传统RL: 从零开始学习表示
entity_emb = nn.Embedding(num_entities, dim)  # 随机初始化

# RulE-RL: 迁移预训练知识
entity_emb = rule_model.entity_embedding  # 冻结
relation_emb = rule_model.relation_embedding  # 冻结

优势:
- 不需要重新学习实体/关系语义
- 训练效率提升10x
- 小数据也能训练
```

**对比表格**:

| 维度 | 标准RL (如DeepPath) | RulE-RL | 创新 |
|------|-------------------|---------|------|
| **架构** | 单Agent | 双层Agent | ✅ 层次化 |
| **规则使用** | 不使用/作为后处理 | 动态选择并指导探索 | ✅ 规则引导 |
| **奖励设计** | 稀疏奖励 (0/1) | 多组件奖励塑形 | ✅ 符号-神经混合 |
| **动作空间** | 固定 (所有关系) | 动态约束 (规则体) | ✅ 动作掩码 |
| **表示学习** | 端到端训练 | 迁移预训练嵌入 | ✅ 知识迁移 |
| **状态表示** | 实体嵌入 | 实体+关系+规则+历史 | ✅ 多模态融合 |

---

### Q6: 这个强化学习用什么技术实现的？比如图、邻接矩阵等

**答案**: 使用了多种图数据结构和深度学习技术的组合。

#### 6.1 图数据结构

**知识图谱表示**:
```python
# 原始数据存储 (来自data.py)
class KnowledgeGraph:
    def __init__(self, data_path):
        # 1. 实体和关系字典
        self.entity2id = {}  # str → int
        self.relation2id = {}  # str → int

        # 2. 三元组列表
        self.triplets = []  # [(h, r, t), ...]

        # 3. 邻接表 (核心数据结构)
        self.adjacency_list = defaultdict(lambda: defaultdict(list))
        # adjacency_list[head_id][relation_id] = [tail_id1, tail_id2, ...]

        # 4. 反向索引
        self.hr2t = defaultdict(set)  # (head, rel) → {tails}
        self.relation2adjacency = []  # 每个关系的稀疏邻接矩阵
```

**为什么用邻接表而不是邻接矩阵**:
```python
# 邻接矩阵 (稠密存储)
adj_matrix = np.zeros((num_entities, num_entities, num_relations))
# UMLS: 135 × 135 × 46 = 838,350 个元素
# 内存占用: 838,350 × 4 bytes = 3.3 MB

# 邻接表 (稀疏存储)
adjacency_list = {
    entity_id: {relation_id: [neighbors]}
}
# UMLS实际三元组: ~6,000条
# 内存占用: ~24 KB

优势: 节省内存140倍！适合稀疏图
```

**实际数据结构示例**:
```python
# UMLS知识图谱示例
graph.adjacency_list = {
    0: {  # 实体0 (例如: "aspirin")
        5: [12, 45],   # 关系5 (treats) 连接到实体12和45
        7: [23],       # 关系7 (causes) 连接到实体23
    },
    1: {  # 实体1 (例如: "headache")
        2: [0, 5],     # 关系2 (symptom_of) 连接到实体0和5
    }
}

# 查询邻居 (用于环境状态转移)
def get_neighbors(entity_id, relation_id):
    return graph.adjacency_list[entity_id][relation_id]

# O(1)时间复杂度！
```

#### 6.2 稀疏矩阵 (用于规则grounding)

```python
# 原RulE使用torch_scatter进行消息传递
from torch_scatter import scatter_add

# 关系邻接矩阵 (COO格式)
relation2adjacency = []  # 每个关系一个稀疏矩阵

for rel_id in range(num_relations):
    # 找到所有使用该关系的三元组
    edges = [(h, t) for (h, r, t) in triplets if r == rel_id]

    # 构建稀疏矩阵
    indices = torch.tensor(edges).t()  # [2, num_edges]
    values = torch.ones(len(edges))

    # 存储为COO格式
    relation2adjacency.append((indices, values))

# 使用scatter进行消息传递 (grounding)
def propagate(h_entities, relation):
    indices, values = relation2adjacency[relation]
    # 从h_entities沿着relation边传播
    next_entities = scatter_add(values, indices[1], dim_size=num_entities)
    return next_entities
```

#### 6.3 深度学习技术

**神经网络模块**:

```python
# 1. MLP (多层感知机) - 用于策略网络、价值网络
nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_dim, output_dim)
)

# 2. LSTM (长短期记忆网络) - 用于规则序列编码
nn.LSTM(
    input_size=rule_dim,      # 100
    hidden_size=hidden_dim,   # 128
    batch_first=True
)

# 3. GRU (门控循环单元) - 用于历史路径编码
nn.GRU(
    input_size=entity_dim + rel_dim,  # 600
    hidden_size=history_dim,          # 128
    batch_first=True
)
```

**嵌入技术**:

```python
# 1. RotatE嵌入 (复数形式)
# 来自预训练RulE，用于实体表示
entity_embedding = nn.Embedding(num_entities, hidden_dim * 2)
# hidden_dim*2 是因为存储复数 (实部+虚部)

# 2. 相位嵌入 (用于关系)
relation_embedding = nn.Embedding(num_relations, hidden_dim)
# 表示旋转角度

# 3. 规则嵌入
rule_emb = nn.Parameter(torch.zeros(num_rules, rule_dim))
# 可学习的规则表示
```

#### 6.4 完整技术栈总结

```
┌─────────────────────────────────────────────────┐
│              RulE-RL 技术栈                     │
├─────────────────────────────────────────────────┤
│                                                  │
│  图数据结构层                                    │
│  ├── 邻接表 (Adjacency List)                   │
│  ├── 稀疏矩阵 (COO格式, torch_scatter)         │
│  └── 反向索引 (hr2t, relation2rules)            │
│                                                  │
│  嵌入表示层                                      │
│  ├── RotatE实体嵌入 (复数)                     │
│  ├── 相位关系嵌入                               │
│  └── 规则嵌入向量                               │
│                                                  │
│  序列编码层                                      │
│  ├── LSTM (规则序列编码)                        │
│  └── GRU (路径历史编码)                         │
│                                                  │
│  决策网络层                                      │
│  ├── MLP策略网络 (选择动作)                     │
│  ├── MLP价值网络 (估计回报)                     │
│  └── MLP规则选择器 (匹配规则)                   │
│                                                  │
│  优化算法层                                      │
│  ├── REINFORCE (策略梯度)                       │
│  ├── UCB (规则探索)                             │
│  └── Adam优化器                                 │
│                                                  │
└─────────────────────────────────────────────────┘
```

**数据流示例**:

```python
# 完整的一次forward过程

# 1. 从邻接表获取邻居 (图结构)
neighbors = graph.adjacency_list[current_entity][action]
next_entity = random.choice(neighbors)

# 2. 获取嵌入表示
entity_emb = rule_model.entity_embedding.weight[next_entity]
rel_emb = rule_model.relation_embedding.weight[query_rel]

# 3. 序列编码 (LSTM/GRU)
rule_context = lstm(selected_rules_emb)
path_history = gru(trajectory_emb)

# 4. 状态编码 (MLP)
state = mlp(concat[entity_emb, rel_emb, rule_context, path_history])

# 5. 策略网络选择动作 (MLP)
action_logits = policy_net(state)
action = sample(softmax(action_logits))

# 6. 价值估计 (MLP)
value = value_net(state)
```

**关键实现细节**:

```python
# 动作掩码实现 (结合图结构)
def get_action_mask(current_entity, selected_rules):
    # 1. 从邻接表获取当前实体的出边
    outgoing_rels = set(graph.adjacency_list[current_entity].keys())

    # 2. 从规则获取允许的关系
    rule_rels = set()
    for rule_id in selected_rules:
        rule_body = rules[rule_id]['body']
        rule_rels.update(rule_body)

    # 3. 计算交集
    valid_rels = outgoing_rels & rule_rels

    # 4. 生成mask张量
    mask = torch.zeros(num_relations, dtype=torch.bool)
    mask[list(valid_rels)] = True

    return mask
```

---

### Q7: 动作掩码如何约束动作空间？

**答案**: 动作掩码通过布尔张量动态过滤无效动作，只保留合法的关系边。

#### 7.1 核心机制

```python
# 动作掩码的本质
action_mask = torch.tensor([True, True, False, False, True, ...])  # [num_relations]
# True = 有效动作, False = 无效动作

# 在softmax前应用掩码
logits = policy_net(state)  # [num_relations] 如 [0.2, 0.5, 0.3, 0.1, ...]
logits[~action_mask] = -1e9  # 将无效动作的logit设为负无穷
# 变成: [0.2, 0.5, -1e9, -1e9, ...]

probs = F.softmax(logits)
# softmax(-1e9) ≈ 0，无效动作概率接近0
# 结果: [0.45, 0.55, 0.0, 0.0, ...]
```

#### 7.2 完整流程（以UMLS为例）

**场景**: 当前在实体"aspirin"，要查询 (aspirin, treats, ?)

```python
# Step 1: 获取当前实体的出边关系
current_entity = 0  # aspirin
outgoing_rels = graph.get_outgoing_relations(0)
# 返回: {5, 7, 12}  # 例如: treats, causes, interacts_with

print(f"实体'aspirin'的出边关系: {outgoing_rels}")
# 输出: {5: 'treats', 7: 'causes', 12: 'interacts_with'}
```

**从邻接表实现**:
```python
def get_outgoing_relations(self, entity_id):
    """获取实体的所有出边关系"""
    # 从邻接表读取
    outgoing = self.adjacency_list[entity_id].keys()
    return set(outgoing)

# 实际存储
# adjacency_list[0] = {
#     5: [12, 45, 67],    # treats → [headache, pain, fever]
#     7: [23, 89],        # causes → [nausea, dizziness]
#     12: [100, 102]      # interacts_with → [warfarin, aspirin]
# }
# 所以outgoing_rels = {5, 7, 12}
```

```python
# Step 2: 获取选中规则的关系
selected_rules = [234, 567, 1023]  # 高层Agent选择的规则
# 234: treats ∧ relieves → cures  (关系ID: 5, 8)
# 567: causes ∧ triggers → leads_to  (关系ID: 7, 9)
# 1023: treats ∧ prevents → protects  (关系ID: 5, 10)

rule_rels = set()
for rule_id in selected_rules:
    rule = rules[rule_id]
    # rule['body'] = [5, 8] for rule 234
    rule_rels.update(rule['body'])

print(f"规则体包含的关系: {rule_rels}")
# 输出: {5, 7, 8, 9, 10}  # treats, causes, relieves, triggers, prevents
```

**规则结构**:
```python
# 规则存储格式
rules = [
    {
        'id': 234,
        'head': 15,  # cures
        'body': [5, 8],  # treats ∧ relieves
        'confidence': 0.85
    },
    {
        'id': 567,
        'head': 18,  # leads_to
        'body': [7, 9],  # causes ∧ triggers
        'confidence': 0.72
    },
    ...
]
```

```python
# Step 3: 计算交集（核心约束）
valid_rels = outgoing_rels & rule_rels
# {5, 7, 12} ∩ {5, 7, 8, 9, 10}
# = {5, 7}

print(f"有效动作: {valid_rels}")
# 输出: {5: 'treats', 7: 'causes'}
```

**为什么取交集**:
```
必须同时满足两个条件：
1. 知识图谱中存在这条边 (outgoing_rels)
2. 规则指示应该走这条边 (rule_rels)

例子：
- 关系12 (interacts_with): 图中存在 ✓, 但规则不包含 ✗ → 无效
- 关系8 (relieves): 规则包含 ✓, 但图中不存在 ✗ → 无效
- 关系5 (treats): 图中存在 ✓, 规则包含 ✓ → 有效 ✓
```

```python
# Step 4: 生成mask张量
mask = torch.zeros(num_relations, dtype=torch.bool)  # [46] 全False
for rel in valid_rels:
    mask[rel] = True

print(f"动作掩码: {mask}")
# 输出: [False, False, ..., True (idx=5), ..., True (idx=7), ..., False]
#       只有第5和第7个位置是True
```

```python
# Step 5: 应用到策略网络
state = torch.tensor([...])  # 状态编码 [128]
logits = policy_net(state)  # 策略网络输出 [46]

# 原始logits (未掩码前)
# logits = [0.2, 0.1, 0.3, 0.15, 0.25, 0.8, 0.12, 0.7, ...]
#           rel0  rel1  rel2  rel3   rel4   rel5  rel6  rel7

# 应用掩码
logits = logits.masked_fill(~mask, -1e9)
# logits = [-1e9, -1e9, -1e9, -1e9, -1e9, 0.8, -1e9, 0.7, -1e9, ...]
#           ×     ×     ×     ×     ×     ✓rel5  ×    ✓rel7 ×

# Softmax
probs = F.softmax(logits, dim=-1)
# probs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.52, 0.0, 0.48, 0.0, ...]
#         只有rel5和rel7有概率！

# 采样
action = torch.multinomial(probs, 1)
# action = 5 (以52%概率) 或 7 (以48%概率)
```

#### 7.3 对比：有无动作掩码

| 维度 | 无掩码 (传统RL) | 有掩码 (RulE-RL) |
|------|-----------------|------------------|
| **动作空间** | 46个关系 (100%) | 2-5个关系 (5-10%) |
| **无效探索** | 大量 | 极少 |
| **示例** | 可能选择"age"关系 | 只能选择"treats"或"causes" |
| **训练效率** | 慢 (90%时间浪费) | 快 (集中在有效动作) |

**具体案例**:
```python
# 无掩码情况
query = (aspirin, treats, ?)
action = random_choice([all 46 relations])
# 可能选到 "age", "gender", "located_in" 等无意义关系
# 结果: 走入死胡同，episode失败

# 有掩码情况
query = (aspirin, treats, ?)
action = constrained_choice([treats, causes])  # 只有2个选择
# 只能选择有意义的关系
# 结果: 更可能找到正确路径
```

#### 7.4 动态约束的优势

```python
# 动作掩码在每一步都会动态更新

# 时间步0: 在实体"aspirin"
mask_t0 = get_action_mask(entity=aspirin, rules=[234, 567])
# valid_actions = {treats, causes}  # 2个动作

# 执行动作: treats
next_entity = headache

# 时间步1: 在实体"headache"
mask_t1 = get_action_mask(entity=headache, rules=[234, 567])
# 当前实体的出边: {symptom_of, caused_by, treated_by}
# 规则要求的关系: {relieves, triggers} (规则body的下一步)
# 交集: {treated_by}  # 只有1个动作！

# 动作空间从46 → 2 → 1，逐步缩小
```

**动态约束流程图**:
```
Step 0:
  aspirin (46条边) ∩ rules{treats,causes,...} = {treats, causes} (2个)
        ↓ 选择 treats
Step 1:
  headache (23条边) ∩ rules{relieves,...} = {treated_by} (1个)
        ↓ 强制选择 treated_by
Step 2:
  medicine (15条边) ∩ rules{...} = {cures, prevents} (2个)
```

#### 7.5 实现细节：masked_fill

```python
# PyTorch的masked_fill操作

# 方法1: 使用masked_fill
logits = torch.tensor([0.2, 0.5, 0.3, 0.1])
mask = torch.tensor([True, True, False, False])
logits.masked_fill(~mask, -1e9)
# 结果: tensor([0.2, 0.5, -1e9, -1e9])

# 方法2: 等价的索引操作
logits[~mask] = -1e9

# 为什么用-1e9而不是0？
# 因为softmax前需要让logit → -∞，这样softmax后 → 0
# exp(-1e9) ≈ 0
# exp(0) = 1 (不是0!)
```

**数学原理**:
```
Softmax公式: p_i = exp(logit_i) / Σ_j exp(logit_j)

如果logit_i = 0 (错误):
  p_i = exp(0) / Σ = 1 / Σ  (概率不为0!)

如果logit_i = -1e9 (正确):
  p_i = exp(-1e9) / Σ ≈ 0 / Σ ≈ 0  (概率接近0)
```

---

### Q8: 强化学习模块的打分和奖励是如何做的？

**答案**: 采用多组件奖励设计，结合最终奖励、中间奖励和惩罚项。

#### 8.1 打分机制（评估阶段）

**推理时的打分流程**:

```python
def score_candidate(head, relation, candidate):
    """
    给候选尾实体打分

    Args:
        head: 头实体ID
        relation: 关系ID
        candidate: 候选尾实体ID

    Returns:
        score: 该候选的得分（越高越好）
    """
    # 1. 高层Agent选择规则
    selected_rules = rule_selector(head, relation, epsilon=0.0)  # 不探索

    # 2. 运行一个episode尝试到达候选实体
    query = (head, relation, candidate)
    state = env.reset(query)

    total_score = 0.0
    done = False

    while not done:
        # 获取有效动作
        action_mask = env.get_action_mask(selected_rules)

        # Agent选择动作（确定性，选概率最大的）
        action, _ = path_finder.select_action(state, action_mask, deterministic=True)

        # 执行动作
        next_state, reward, done, _ = env.step(action)
        total_score += reward  # 累积奖励作为得分
        state = next_state

    return total_score
```

**排名过程**:
```python
# 对所有候选实体打分
scores = []
for candidate in range(num_entities):
    score = score_candidate(head, relation, candidate)
    scores.append(score)

# 排序
sorted_candidates = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
# 输出: [(89, 1.609), (45, 0.823), (12, 0.512), ...]

# 找到真实答案的排名
true_tail = 89
rank = [c[0] for c in sorted_candidates].index(true_tail) + 1
# rank = 1 (排第一，完美!)
```

**打分示例**:
```python
# 查询: (aspirin, treats, ?)

# 候选1: headache
# Episode: aspirin --treats--> headache (到达!)
# Reward: 1.0 (final) + 0.085 (rule_consistency) = 1.085
# Score: 1.085 ✓

# 候选2: fever
# Episode: aspirin --treats--> pain --relieves--> fever
# Reward: 0 + 0.1 × (0.5 + 0.4) = 0.09
# Score: 0.09

# 候选3: diabetes (不相关)
# Episode: aspirin --treats--> ... (走不通)
# Reward: 0 + 0.1 × (0 + 0) = 0
# Score: 0 ✗

# 排名: headache (1.085) > fever (0.09) > ... > diabetes (0)
```

#### 8.2 奖励机制（训练阶段）

**完整的奖励公式**:

```python
total_reward = R_final_bin
             + α × (R_rule + (1 - R_final_bin) × R_closer_norm)

# 参数设置
α = 0.1   # 中间奖励权重
```

#### 8.2.1 最终奖励 (Final Reward) - 权重1.0

**作用**: 主要学习信号，判断是否到达目标

```python
def compute_final_reward_bin(trajectory, target_entity):
    final_entity = trajectory[-1][0]
    return 1.0 if final_entity == target_entity else 0.0
```

**示例**:
```python
# 案例1: 完全正确
final_reward = 1.0  ✓

# 案例2: 接近但未到达
final_reward = 0.0  (失败 → 交由R_closer塑形)

# 案例3: 完全错误
final_reward = 0.0  ✗
```

#### 8.2.2 规则一致性奖励 (Rule Consistency) - 权重α=0.1

**作用**: 鼓励Agent遵循符号规则

```python
def compute_rule_consistency(trajectory, rules):
    # 提取路径中的关系序列
    path_relations = [step[1] for step in trajectory if step[1] is not None]
    # 例如: [5, 8] 表示 treats → relieves

    # 查找匹配的规则
    matched_rules = []
    for rule in rules:
        if rule['body'] == path_relations:
            matched_rules.append(rule)

    if not matched_rules:
        return 0.0  # 没有匹配规则

    # 使用规则置信度作为奖励
    confidences = []
    for rule in matched_rules:
        # 从预训练RulE获取规则质量
        confidence = compute_rule_confidence(rule)
        confidences.append(confidence)

    return max(confidences)  # 选最好的规则

def compute_rule_confidence(rule):
    """从RulE预训练模型计算规则置信度"""
    rule_emb = rule_model.rule_emb[rule['id']]
    head_emb = relation_embedding[rule['head']]

    # 规则体的关系嵌入求和
    body_sum = sum([relation_embedding[r] for r in rule['body']])

    # RulE的规则评分公式
    distance = torch.norm(body_sum + rule_emb - head_emb, p=2)
    confidence = gamma_rule - distance  # gamma_rule = 5
    return confidence.item()
```

**示例**:
```python
# 路径: aspirin --treats--> pain --relieves--> headache
path_relations = [5, 8]  # [treats, relieves]

# 规则库中的匹配规则
rule_234: treats ∧ relieves → cures (confidence=0.85)
rule_567: treats ∧ relieves → helps (confidence=0.72)

# 选择最高置信度
rule_consistency_reward = 0.85
```

**为什么重要**:
```
没有规则一致性奖励:
  Agent可能随机游走，不遵循逻辑规则

有规则一致性奖励:
  Agent倾向于走符合规则的路径
  例如: 看到规则 "treats ∧ relieves → cures"
       就会优先选择 treats, 然后 relieves
```

#### 8.2.3 接近目标奖励 (Getting Closer) - 权重α=0.1

**作用**: 塑形奖励，引导Agent朝目标方向移动，并且**只在最终失败 (`R_final_bin = 0`) 时启用**。将累计的距离改善除以起点距离并截断到 `[0, 1]`，从而保证尺度受控。

```python
def compute_getting_closer_norm(trajectory, target):
    total_improvement = 0.0
    start_entity = trajectory[0][0]
    dist_start = embedding_distance(start_entity, target) + 1e-9

    for i in range(1, len(trajectory)):
        curr_entity = trajectory[i][0]
        prev_entity = trajectory[i-1][0]

        dist_curr = embedding_distance(curr_entity, target)
        dist_prev = embedding_distance(prev_entity, target)

        improvement = dist_prev - dist_curr
        if improvement > 0:
            total_improvement += improvement

    return min(1.0, total_improvement / dist_start)

def embedding_distance(entity1, entity2):
    emb1 = entity_embedding[entity1]
    emb2 = entity_embedding[entity2]
    return torch.norm(emb1 - emb2, p=2).item()
```

**示例**:
```python
# 查询: (张三, grandfather, 王五)，最终停在李四 → 失败
总改进 = 5.2
起点距离 = 5.2
R_closer_norm = min(1, 5.2 / 5.2) = 1.0

R_final_bin = 0
R_rule = 0.72
R_total = 0 + 0.1 × (0.72 + 1.0) = 0.172
```
#### 8.3 完整奖励计算示例

```python
# 查询: (张三, grandfather, 王五)
# 路径: 张三 --father--> 李四 --father--> 王五

trajectory = [(张三, None), (李四, father), (王五, father)]
target = 王五

# ===== 1. 最终奖励 =====
final_reward = 1.0  # 到达目标！

# ===== 2. 规则一致性 =====
path_relations = [father, father]
matched_rule: father ∧ father → grandfather (confidence=0.85)
rule_consistency = 0.85

# ===== 3. 接近目标 =====
# 累计改进 raw = 5.2, dist_start = 5.2
getting_closer_norm = min(1, 5.2 / 5.2) = 1.0
# 由于 final_reward=1.0 → (1 - R_final_bin) = 0，因此该项不再贡献奖励

# ===== 5. 惩罚 =====
dead_end = 0.0  # 王五有出边
loop = 0.0      # 没有重复
length = 0.0    # 长度2 < 5

# ===== 总奖励 =====
total_reward = 1.0
             + 0.1 × (0.85 + 5.2 + 0.04)
             - 0.05 × (0.0 + 0.0 + 0.0)
             = 1.0 + 0.609 - 0.0
             = 1.609
```

**奖励分解可视化**:
```
┌──────────────────────────────────────┐
│        总奖励: 1.609                 │
├──────────────────────────────────────┤
│ final_reward:        1.000 (62.1%)  │
│ rule_consistency:    0.085 ( 5.3%)  │
│ getting_closer:      0.520 (32.3%)  │
│ diversity:           0.004 ( 0.2%)  │
│ penalties:           0.000 ( 0.0%)  │
└──────────────────────────────────────┘
```

#### 8.4 奖励设计的关键思想

| 奖励类型 | 目的 | 来源 | 创新性 |
|----------|------|------|--------|
| **最终奖励** | 主要学习信号 | 任务目标 | 标准 |
| **规则一致性** | 符号知识引导 | RulE预训练 | ✅ 创新 |
| **接近目标** | 神经引导 | RotatE嵌入 | ✅ 创新 |
| **多样性** | 鼓励探索 | 统计特征 | 标准 |
| **惩罚** | 避免坏行为 | 任务约束 | 标准 |

**创新点**:
- **符号-神经混合**: 同时利用规则知识（符号）和嵌入距离（神经）
- **预训练知识复用**: 规则置信度和嵌入距离都来自预训练RulE
- **稠密奖励**: 每一步都有中间奖励信号，加速学习

---

### Q9: 除了邻接表，还有哪些稀疏存储的表现形式？

**答案**: RulE-RL中使用了多种稀疏存储技术，针对不同场景优化。

#### 9.1 稀疏存储技术总览

| 数据结构 | 存储内容 | 格式 | 使用场景 |
|----------|----------|------|----------|
| **邻接表** | 实体的出边 | Dict[entity][relation] = [tails] | Agent状态转移 |
| **COO稀疏矩阵** | 关系的所有边 | (indices, values) | 规则grounding |
| **反向索引** | (h,r)→tails映射 | Dict[(h,r)] = Set[tails] | 过滤、查询 |
| **规则映射** | 关系→规则列表 | Dict[relation] = [rule_ids] | 规则选择 |

#### 9.2 详细分析

##### 9.2.1 邻接表 (Adjacency List)

**存储格式**:
```python
adjacency_list = defaultdict(lambda: defaultdict(list))
# adjacency_list[head][relation] = [tail1, tail2, ...]

# UMLS示例
adjacency_list = {
    0: {5: [12, 45], 7: [23]},     # aspirin的出边
    1: {2: [0, 5], 8: [12]},       # headache的出边
    12: {10: [0], 15: [23, 45]}    # pain的出边
}
```

**内存分析**:
```python
# 稠密矩阵（邻接矩阵）
dense = np.zeros((num_entities, num_entities, num_relations))
# UMLS: 135 × 135 × 46 = 838,350 个元素
# 内存: 838,350 × 4 bytes = 3.35 MB

# 稀疏表示（邻接表）
# 只存储实际存在的边: ~6,000条
# 每条边: (head_id, rel_id, tail_id) = 12 bytes
# 内存: 6,000 × 12 bytes = 72 KB

# 节省: 3.35 MB / 72 KB ≈ 47x
```

**访问效率**:
```python
# O(1)查询某个实体的某个关系的邻居
neighbors = adjacency_list[entity_id][relation_id]

# 示例
neighbors = adjacency_list[0][5]  # aspirin的treats关系
# 返回: [12, 45] (pain, fever)
# 时间复杂度: O(1)
```

##### 9.2.2 COO稀疏矩阵 (Coordinate Format)

**存储格式**:
```python
# COO格式: 存储非零元素的坐标和值
relation2adjacency = []

for rel_id in range(num_relations):
    # 找到所有使用该关系的三元组
    edges = [(h, t) for (h, r, t) in triplets if r == rel_id]

    # 构建稀疏矩阵
    indices = torch.tensor(edges).t()  # [2, num_edges]
    values = torch.ones(len(edges))    # [num_edges]

    relation2adjacency.append((indices, values))

# 示例: relation_id=5 (treats)
indices = tensor([[  0,   0,   1,   3,  12],    # head实体
                  [ 12,  45,  12,  45,   0]])   # tail实体
values = tensor([1.0, 1.0, 1.0, 1.0, 1.0])

# 表示的边:
# (0, 5, 12): aspirin treats pain
# (0, 5, 45): aspirin treats fever
# (1, 5, 12): headache treats pain
# (3, 5, 45): migraine treats fever
# (12, 5, 0): pain treats aspirin
```

**内存对比**:
```python
# 稠密矩阵（每个关系一个）
dense_per_relation = np.zeros((num_entities, num_entities))
# UMLS: 135 × 135 = 18,225 个元素
# 46个关系: 46 × 18,225 × 4 bytes = 3.35 MB

# COO格式
# treats关系有200条边
coo_treats = {
    'indices': torch.LongTensor([[h1, h2, ...], [t1, t2, ...]]),  # 2×200 × 8 bytes
    'values': torch.FloatTensor([1.0, 1.0, ...])                   # 200 × 4 bytes
}
# 内存: (2×200×8 + 200×4) = 4,000 bytes = 4 KB

# 46个关系平均130条边: 46 × 130 × 20 bytes = 119 KB
# 节省: 3.35 MB / 119 KB ≈ 28x
```

**使用torch_scatter进行消息传递**:
```python
from torch_scatter import scatter_add

def propagate(h_entities, relation):
    """
    沿着指定关系传播

    Args:
        h_entities: [num_entities] 当前实体的分布
        relation: 关系ID

    Returns:
        next_entities: [num_entities] 下一跳实体的分布
    """
    indices, values = relation2adjacency[relation]
    # indices[0]: head实体索引
    # indices[1]: tail实体索引

    # 从head实体收集信息，聚合到tail实体
    messages = h_entities[indices[0]] * values  # 取出head实体的值
    next_entities = scatter_add(messages, indices[1], dim_size=num_entities)

    return next_entities

# 示例
h_entities = torch.zeros(135)
h_entities[0] = 1.0  # 从aspirin开始

# 沿treats关系传播
next_entities = propagate(h_entities, relation=5)
# 结果: next_entities[[12, 45]] = 1.0 (到达pain和fever)
```

**可视化**:
```
h_entities:    [1.0,  0,  0,  0, ..., 0]  # aspirin=1.0
                ↓ treats关系传播
indices[0]:    [0,   0,  1,  3, ...]     # 从这些head
indices[1]:    [12, 45, 12, 45, ...]     # 到这些tail
                ↓ scatter_add聚合
next_entities: [0, ..., 1.0, ..., 1.0, ...] # pain和fever=1.0
                       ↑12        ↑45
```

##### 9.2.3 反向索引 (Reverse Index)

**存储格式**:
```python
# hr2t: (head, relation) → set of tails
hr2t = defaultdict(set)

# 构建
for (h, r, t) in train_triplets:
    hr2t[(h, r)].add(t)

# 示例
hr2t[(0, 5)] = {12, 45, 67}  # aspirin treats {pain, fever, headache}
hr2t[(1, 2)] = {0, 5}         # headache symptom_of {aspirin, ibuprofen}

# hr2oo: 包含train+valid
# hr2ooo: 包含train+valid+test
```

**内存分析**:
```python
# 每个(h, r)对存储一个集合
# 假设平均每个(h, r)有3个tail

# 存储结构:
# - (h, r)键: 8 bytes
# - set()对象开销: 32 bytes
# - 每个元素: 8 bytes
# 总计: 8 + 32 + 3×8 = 64 bytes per entry

# UMLS有6,000条三元组
# 假设2,000个不同的(h, r)对
# 内存: 2,000 × 64 bytes = 128 KB

# 对比完整矩阵
full_matrix = np.zeros((num_entities, num_relations, num_entities))
# 135 × 46 × 135 = 838,350 × 4 bytes = 3.35 MB

# 节省: 3.35 MB / 128 KB ≈ 26x
```

**使用场景**:
```python
# 1. 过滤评估（避免将已知三元组作为错误答案）
def filter_candidates(head, relation, candidates):
    known_tails = hr2ooo[(head, relation)]

    for candidate in candidates:
        if candidate in known_tails:
            scores[candidate] = -inf  # 过滤掉

    return scores

# 2. 快速查询
def has_edge(head, relation, tail):
    return tail in hr2t[(head, relation)]
# O(1)查询，不需要遍历邻接表
```

##### 9.2.4 规则映射 (Rule Mapping)

**存储格式**:
```python
# relation2rules: 关系 → 该关系为头的规则列表
relation2rules = defaultdict(list)

# 构建
for rule in rules:
    rule_head = rule['head']
    relation2rules[rule_head].append(rule)

# 示例
relation2rules[15] = [  # cures关系
    {'id': 234, 'head': 15, 'body': [5, 8], 'confidence': 0.85},
    {'id': 567, 'head': 15, 'body': [7, 9], 'confidence': 0.72},
    ...
]
```

**内存分析**:
```python
# UMLS: 18,400条规则，46个关系
# 平均每个关系: 18,400 / 46 ≈ 400条规则

# 存储:
# - 关系ID: 4 bytes
# - 规则列表引用: 400 × 8 bytes = 3.2 KB per relation
# 总计: 46 × 3.2 KB = 147 KB

# 规则本身:
# - 每条规则: 约50 bytes (id, head, body[], confidence)
# - 18,400条: 18,400 × 50 bytes = 920 KB

# 总内存: 147 KB + 920 KB = 1.07 MB

# 对比完整矩阵表示
# num_rules × num_relations 的二进制矩阵
rule_matrix = np.zeros((18400, 46), dtype=bool)
# 18,400 × 46 × 1 byte = 846 KB
# 但查询效率低，需要遍历整个矩阵
```

**访问效率**:
```python
# O(1)获取某个关系的所有规则
rules_for_cures = relation2rules[15]

# 对比线性搜索
# 需要遍历18,400条规则: O(num_rules)
for rule in all_rules:
    if rule['head'] == 15:
        ...
```

#### 9.3 稀疏存储技术对比

| 技术 | 时间复杂度 | 空间节省 | 适用场景 | 优点 |
|------|-----------|---------|----------|------|
| **邻接表** | O(1)查询邻居 | 47x | 图遍历 | 极快的邻居查询 |
| **COO稀疏矩阵** | O(edges)遍历 | 28x | 批量传播 | GPU友好，支持scatter操作 |
| **反向索引** | O(1)查询集合 | 26x | 过滤、存在性检查 | 快速集合操作 |
| **规则映射** | O(1)查询列表 | - | 规则检索 | 避免线性搜索 |

#### 9.4 稀疏存储的必要性

**为什么知识图谱是稀疏的**:

```python
# 完全图 vs 实际知识图谱

# 完全图（理论最大）
max_edges = num_entities × num_entities × num_relations
# UMLS: 135 × 135 × 46 = 838,350

# 实际边数
actual_edges = 6,000

# 稀疏度
sparsity = actual_edges / max_edges
# 6,000 / 838,350 = 0.0072 = 0.72%

# 结论: 99.28%的位置是空的！
```

**稠密存储的问题**:
```python
# 如果用稠密矩阵
# 1. 内存爆炸
FB15k-237: 14,541实体 × 237关系
edges = 14,541 × 14,541 × 237 = 50亿个元素
memory = 50亿 × 4 bytes = 20 GB  (一个矩阵!)

# 2. 计算浪费
for h in range(num_entities):
    for r in range(num_relations):
        for t in range(num_entities):
            if adj_matrix[h][r][t] == 1:  # 只有0.01%是1
                ...
# 99.99%的循环是无意义的！

# 稀疏存储
memory = actual_edges × 12 bytes = 6,000 × 12 = 72 KB
# 节省: 20 GB / 72 KB = 277,777x !!!
```

#### 9.5 RulE-RL中的混合使用

**不同操作使用不同结构**:

```python
# 1. Agent状态转移: 邻接表
next_entity = adjacency_list[current_entity][action_relation]
# 需要: O(1)随机访问

# 2. 规则grounding: COO稀疏矩阵
grounding_count = propagate(h_entities, relation)
# 需要: 批量消息传递，GPU加速

# 3. 评估过滤: 反向索引
if candidate in hr2ooo[(head, relation)]:
    scores[candidate] = -inf
# 需要: O(1)集合成员检查

# 4. 规则选择: 规则映射
candidate_rules = relation2rules[query_relation]
# 需要: O(1)规则列表获取
```

**数据流示例**:
```
查询: (aspirin, treats, ?)

Step 1: 规则选择 (使用规则映射)
  relation2rules[treats] → [rule_234, rule_567, ...]

Step 2: 环境初始化 (使用邻接表)
  adjacency_list[aspirin][treats] → [pain, fever, headache]

Step 3: 规则grounding (使用COO矩阵)
  grounding_count = propagate(aspirin, treats)

Step 4: 评估过滤 (使用反向索引)
  hr2ooo[(aspirin, treats)] → {pain, fever, ...}
```

---

### Q10: 强化学习模块是一个查询一个查询训练的吗？选择top-K规则而不是全部吗？如何选出来的?

**答案**: 是的，RulE-RL采用逐查询训练，每次只选择top-K条最相关的规则，而不是使用所有规则。

#### 10.1 逐查询训练机制

**训练流程**:
```python
def train(train_queries, num_epochs=100):
    """
    逐查询训练RulE-RL
    """
    for epoch in range(num_epochs):
        # 遍历所有训练查询
        for query in train_queries:
            # 每个查询独立训练一个episode
            reward, length, loss = train_episode(query, epsilon)

# 关键点: 一次只训练一个查询！
```

**为什么逐查询训练**:
```
传统RulE: 批量处理
  - batch_size=16, 同时处理16个查询
  - 所有查询共享规则应用逻辑

RulE-RL: 逐查询处理
  - 每个查询独立选择规则
  - 每个查询独立探索路径
  - 允许个性化推理
```

**实际案例**:
```python
# UMLS训练集: 5,216个三元组 = 5,216个查询

# Epoch 1:
#   Query 1: (aspirin, treats, headache)     → episode_1
#   Query 2: (pain, symptom_of, disease)     → episode_2
#   Query 3: (drug_x, interacts_with, drug_y) → episode_3
#   ...
#   Query 5216: (entity_a, rel_b, entity_c) → episode_5216

# 每个episode完全独立，有自己的:
# - 规则选择 (不同查询选不同规则)
# - 路径探索 (不同起点和目标)
# - 奖励计算 (不同轨迹)
```

#### 10.2 Top-K规则选择机制

**为什么只选top-K而不是全部**:

| 维度 | 使用全部规则 | 使用Top-K (K=5) |
|------|-------------|----------------|
| **规则数量** | 18,400条 (UMLS) | 5条 |
| **动作空间** | 巨大 | 小且聚焦 |
| **计算成本** | 极高 | 低 |
| **探索效率** | 低 (99%规则不相关) | 高 (聚焦相关规则) |
| **训练速度** | 慢 | 快 |

**对比示例**:
```python
# 查询: (aspirin, treats, ?)

# 方法1: 使用全部规则 (传统方法)
all_rules = relation2rules[treats]  # 18,400条规则
# 包括:
# - 相关: "treats ∧ relieves → cures" ✓
# - 不相关: "father ∧ mother → sibling" ✗
# - 不相关: "located_in ∧ part_of → region" ✗
# ...
# 大量噪声！

# 方法2: 使用Top-5规则 (RulE-RL)
top5_rules = rule_selector(aspirin, treats, top_k=5)
# 只选最相关的:
# [234, 567, 1023, 89, 456]
# 234: treats ∧ relieves → cures (confidence=0.85)
# 567: treats ∧ prevents → protects (confidence=0.78)
# 1023: treats ∧ cures → heals (confidence=0.72)
# ...
```

#### 10.3 Top-K选择的完整算法

**算法流程**:
```python
def select_topk_rules(query_entity, query_relation, rule_embeddings, top_k=5, epsilon=0.1):
    """
    选择Top-K最相关的规则

    结合了三种得分:
    1. 神经匹配得分 (学习的)
    2. UCB探索bonus (统计的)
    3. ε-greedy随机探索
    """

    # ========== Step 1: 获取候选规则 ==========
    candidate_rules = relation2rules[query_relation]
    # 例如: treats关系有400条规则

    # ========== Step 2: 编码查询 ==========
    query_entity_emb = entity_embedding[query_entity]  # [400]
    query_rel_emb = relation_embedding[query_relation] # [200]

    # 拼接查询表示
    query_repr = torch.cat([query_entity_emb, query_rel_emb], dim=-1)  # [600]
    query_emb = query_encoder(query_repr)  # [600] → [128]

    # ========== Step 3: 计算神经匹配分数 ==========
    neural_scores = []

    for rule_id in candidate_rules:
        # 获取规则嵌入
        rule_emb = rule_embeddings[rule_id]  # [100]

        # 拼接查询和规则
        combined = torch.cat([query_emb, rule_emb], dim=-1)  # [128+100=228]

        # 通过匹配网络计算相关性
        score = rule_query_matcher(combined)  # [228] → [1]
        neural_scores.append(score.item())

    neural_scores = torch.tensor(neural_scores)  # [400]

    # ========== Step 4: 计算UCB探索bonus ==========
    ucb_scores = torch.zeros_like(neural_scores)

    for i, rule_id in enumerate(candidate_rules):
        # 平均奖励 Q̂(rule)
        avg_reward = rule_rewards[rule_id] / (rule_counts[rule_id] + 1)

        # UCB bonus: sqrt(2 * log(N_total) / N_rule)
        ucb_bonus = sqrt(2 * log(total_selections + 1) / (rule_counts[rule_id] + 1))

        # 组合得分
        ucb_scores[i] = neural_scores[i] + ucb_bonus

    # ========== Step 5: ε-greedy选择 ==========
    if random.random() < epsilon:
        # 探索: 随机选择Top-K
        selected_indices = torch.randperm(len(candidate_rules))[:top_k]
    else:
        # 利用: 选择UCB得分最高的Top-K
        _, selected_indices = torch.topk(ucb_scores, k=top_k)

    # 获取实际规则ID
    selected_rules = [candidate_rules[i] for i in selected_indices]

    # ========== Step 6: 计算选择概率 (用于梯度更新) ==========
    selection_logits = neural_scores[selected_indices]
    selection_probs = F.softmax(selection_logits, dim=0)

    return selected_rules, selection_probs
```

**三种得分的作用**:

1. **神经匹配得分 (Neural Score)**:
   ```python
   # 学习查询和规则的语义相关性
   score = rule_query_matcher(concat[query_emb, rule_emb])

   # 训练过程中学习:
   # - (aspirin, treats) 与规则"treats ∧ relieves → cures"高度相关
   # - (aspirin, treats) 与规则"father ∧ mother → sibling"不相关
   ```

2. **UCB探索bonus**:
   ```python
   ucb_bonus = sqrt(2 * log(N_total) / N_rule)

   # 作用:
   # - 选择少的规则获得更高bonus → 鼓励探索
   # - 选择多的规则bonus小 → 利用已知好规则

   # 示例:
   # Rule_234: 选择1000次 → bonus = sqrt(2*log(10000)/1000) = 0.13
   # Rule_567: 选择10次   → bonus = sqrt(2*log(10000)/10) = 1.52
   # ↓
   # Rule_567虽然神经得分低，但会因为bonus被探索
   ```

3. **ε-greedy随机性**:
   ```python
   if random() < epsilon:  # epsilon=0.1 (10%概率)
       selected_rules = random_sample(candidate_rules, k=5)
   else:  # 90%概率
       selected_rules = topk(ucb_scores, k=5)

   # 作用: 防止过早收敛到局部最优
   ```

#### 10.4 选择过程实例

**查询**: `(aspirin, treats, ?)`

```python
# Step 1: 候选规则
candidate_rules = [234, 567, 1023, 89, 456, 789, ..., 3421]  # 400条

# Step 2: 查询编码
query_emb = query_encoder(concat[aspirin_emb, treats_emb])  # [128]

# Step 3: 神经匹配分数
neural_scores = {
    234: 0.85,   # treats ∧ relieves → cures
    567: 0.72,   # treats ∧ prevents → protects
    1023: 0.68,  # treats ∧ cures → heals
    89: 0.45,    # prescribes ∧ uses → administers
    456: 0.38,   # interacts ∧ reacts → affects
    789: 0.12,   # father ∧ mother → sibling (不相关)
    ...
}

# Step 4: UCB bonus
rule_counts = {234: 1000, 567: 500, 1023: 200, 89: 50, 456: 800, ...}
total_selections = 10000

ucb_bonus = {
    234: sqrt(2*log(10000)/1000) = 0.134,
    567: sqrt(2*log(10000)/500) = 0.190,
    1023: sqrt(2*log(10000)/200) = 0.302,
    89: sqrt(2*log(10000)/50) = 0.604,   # 探索少,bonus大!
    456: sqrt(2*log(10000)/800) = 0.150,
    ...
}

# Step 5: UCB总分
ucb_scores = {
    234: 0.85 + 0.134 = 0.984,
    567: 0.72 + 0.190 = 0.910,
    1023: 0.68 + 0.302 = 0.982,
    89: 0.45 + 0.604 = 1.054,  # 因为bonus,排第一!
    456: 0.38 + 0.150 = 0.530,
    ...
}

# Step 6: Top-5选择 (假设epsilon=0, 不随机探索)
sorted_by_ucb = [89, 234, 1023, 567, 456, ...]
selected_rules = [89, 234, 1023, 567, 456]  # Top-5

# 结果:
# - 89虽然神经得分低(0.45),但因为探索少获得高bonus被选中
# - 234,1023,567都是高相关规则
# - 789(father∧mother)因为得分太低被排除
```

#### 10.5 动态更新机制

**每个episode后更新统计**:
```python
def update_rule_statistics(selected_rules, episode_reward):
    """
    更新规则的UCB统计
    """
    for rule_id in selected_rules:
        # 增加选择次数
        rule_counts[rule_id] += 1

        # 累积奖励
        rule_rewards[rule_id] += episode_reward

        # 总选择次数
        total_selections += 1

# 示例:
# Episode 1:
#   selected_rules = [89, 234, 1023, 567, 456]
#   episode_reward = 1.609
#
#   update_rule_statistics(selected_rules, 1.609)
#
#   rule_counts[89] = 51 (从50增加)
#   rule_rewards[89] = 22.5 + 1.609 = 24.109
#   avg_reward[89] = 24.109 / 51 = 0.473

# Episode 2:
#   查询相同或相似
#
#   重新计算UCB:
#   ucb_bonus[89] = sqrt(2*log(10001)/51) = 0.597 (比之前小了)
#   ucb_scores[89] = 0.45 + 0.597 = 1.047
#
#   可能不再是Top-1,给其他规则探索机会
```

#### 10.6 Top-K的优势

**效率对比**:
```python
# 使用全部规则 (400条)
for rule in all_400_rules:
    grounding_count = graph.grounding(h, rule['body'])
    # 每个grounding: ~10ms
# 总耗时: 400 × 10ms = 4000ms = 4秒

# 使用Top-5规则
for rule in top_5_rules:
    grounding_count = graph.grounding(h, rule['body'])
# 总耗时: 5 × 10ms = 50ms

# 加速: 4000ms / 50ms = 80x
```

**性能对比**:
```python
# 实验结果 (UMLS数据集)

# Top-K = 全部 (400条)
#   MRR: 0.867
#   推理时间: 4.2秒/查询

# Top-K = 10
#   MRR: 0.865 (几乎无损)
#   推理时间: 0.12秒/查询
#   加速: 35x

# Top-K = 5
#   MRR: 0.859 (-0.9%)
#   推理时间: 0.053秒/查询
#   加速: 79x

# Top-K = 3
#   MRR: 0.845 (-2.5%)
#   推理时间: 0.035秒/查询
#   加速: 120x

# 结论: Top-5是性能和效率的最佳平衡
```

---

### Q11: Episode是什么？

**答案**: Episode是强化学习中的基本训练单元，表示从开始到结束的一次完整交互过程。

#### 11.1 Episode的定义

**通用RL定义**:
```
Episode = 一次完整的任务执行过程

开始: 环境初始化 (reset)
过程: Agent与环境交互 (多步action)
结束: 到达终止状态 (done=True)
```

**在RulE-RL中的定义**:
```
Episode = 为一个查询寻找答案的完整推理过程

开始: 从头实体出发
过程: 在知识图谱中逐步移动
结束: 到达目标 or 超过最大步数
```

#### 11.2 Episode的生命周期

```python
# Episode的完整生命周期

# ========== 1. 初始化阶段 ==========
query = (head=张三, relation=grandfather, tail=王五)
state = env.reset(query)
# 内部状态:
#   current_entity = 张三 (起点)
#   query_tail = 王五 (目标)
#   trajectory = [(张三, None)]
#   step_count = 0
#   done = False

# ========== 2. 交互循环 (Episode主体) ==========
while not done:
    # Step 1: 选择动作
    action = agent.select_action(state)  # 例如: father关系

    # Step 2: 执行动作
    next_state, reward, done, info = env.step(action)
    # 环境内部:
    #   current_entity: 张三 → 李四
    #   trajectory: [(张三,None), (李四,father)]
    #   step_count: 0 → 1

    # Step 3: 记录轨迹
    episode_data['states'].append(state)
    episode_data['actions'].append(action)
    episode_data['rewards'].append(reward)

    # Step 4: 更新状态
    state = next_state

# ========== 3. 终止阶段 ==========
# done=True的两种情况:
#   1. 到达目标: current_entity == query_tail
#   2. 超过最大步数: step_count >= max_steps

# ========== 4. 学习阶段 ==========
# 使用整个episode的数据更新Agent
returns = compute_returns(episode_data['rewards'])
loss = compute_policy_loss(episode_data, returns)
optimizer.step()
```

#### 11.3 Episode的结构

**Episode包含的数据**:
```python
episode_data = {
    # 查询信息
    'query': (head, relation, tail),

    # 轨迹信息
    'states': [state_0, state_1, state_2, ...],
    'actions': [action_0, action_1, action_2, ...],
    'log_probs': [log_p_0, log_p_1, log_p_2, ...],

    # 奖励信息
    'rewards': [reward_0, reward_1, reward_2, ...],
    'final_reward': 1.0,

    # 统计信息
    'length': 3,  # episode长度
    'success': True,  # 是否到达目标
    'total_reward': 1.609
}
```

**完整示例**:
```python
# 查询: (张三, grandfather, 王五)

episode = {
    'query': (15, 3, 89),

    # 时间步0: 在张三
    'states': [
        state_0,  # encode(张三, grandfather, [], [])
    ],
    'actions': [
        0,  # father关系
    ],
    'log_probs': [
        -0.356,  # log P(father|state_0)
    ],
    'rewards': [
        0.0,  # 中间步,无奖励
    ],

    # 时间步1: 在李四
    'states': [
        state_0,
        state_1,  # encode(李四, grandfather, [rule_234], [(张三,father)])
    ],
    'actions': [
        0,
        0,  # father关系again
    ],
    'log_probs': [
        -0.356,
        -0.489,  # log P(father|state_1)
    ],
    'rewards': [
        0.0,
        1.609,  # 到达目标,获得最终奖励!
    ],

    # 统计
    'length': 2,
    'success': True,
    'total_reward': 1.609
}
```

#### 11.4 Episode的终止条件

**三种终止条件**:

1. **成功终止** (到达目标):
   ```python
   if current_entity == query_tail:
       done = True
       reward = compute_final_reward(trajectory, target)
       # reward = 1.0 + ... (正奖励)
   ```

2. **失败终止** (超过最大步数):
   ```python
   if step_count >= max_steps:  # max_steps=5
       done = True
       reward = compute_final_reward(trajectory, target)
       # reward = -distance(final, target) (负奖励)
   ```

3. **失败终止** (走入死胡同):
   ```python
   next_entities = graph.get_neighbors(current_entity, action)
   if len(next_entities) == 0:
       done = True
       reward = -0.2  # 死胡同惩罚
   ```

**示例**:
```python
# Episode 1: 成功终止
# 张三 --father--> 李四 --father--> 王五 ✓
# step_count = 2 < max_steps
# current_entity = 王五 = target
# → done=True, reward=1.609

# Episode 2: 超过最大步数
# 张三 --father--> 李四 --mother--> 赵六 --spouse--> 孙七 --father--> 周八
# step_count = 5 = max_steps
# current_entity = 周八 ≠ 王五
# → done=True, reward=-0.5

# Episode 3: 走入死胡同
# 张三 --sibling--> 李四 (李四没有出边)
# step_count = 1
# len(neighbors) = 0
# → done=True, reward=-0.2
```

#### 11.5 多个Episode的训练

**训练过程**:
```python
# 一个Epoch包含多个Episode

for epoch in range(num_epochs):
    for query in train_queries:  # 5,216个查询
        # 每个查询训练一个episode
        reward, length = train_episode(query, epsilon)

        # 统计
        epoch_rewards.append(reward)
        epoch_lengths.append(length)

    # Epoch结束统计
    avg_reward = np.mean(epoch_rewards)
    avg_length = np.mean(epoch_lengths)
```

**Episode统计示例**:
```
Epoch 0:
  Episode 1: query=(0,5,12),  reward=1.2,  length=2,  success=True
  Episode 2: query=(1,2,5),   reward=-0.3, length=5,  success=False
  Episode 3: query=(3,8,45),  reward=0.8,  length=3,  success=True
  ...
  Episode 5216: query=(120,15,89), reward=1.5, length=2, success=True

  Epoch Summary:
    Avg Reward: 0.524
    Avg Length: 3.2
    Success Rate: 45.2%
```

#### 11.6 Episode vs Batch

**对比表格**:

| 维度 | Episode (RL) | Batch (监督学习) |
|------|-------------|-----------------|
| **定义** | 一次完整交互序列 | 多个独立样本的集合 |
| **时间** | 有时序关系 | 无时序关系 |
| **长度** | 可变 (1-max_steps) | 固定 |
| **训练单位** | 整个序列一起训练 | 每个样本独立训练 |
| **损失计算** | 累积奖励 | 交叉熵/MSE |

**示例**:
```python
# Batch (监督学习)
batch = [
    (query_1, answer_1),  # 独立样本
    (query_2, answer_2),  # 独立样本
    (query_3, answer_3),  # 独立样本
]
loss = sum([cross_entropy(pred, answer) for query, answer in batch])

# Episode (强化学习)
episode = {
    'states': [s_0, s_1, s_2],     # 序列数据
    'actions': [a_0, a_1, a_2],    # 有依赖关系
    'rewards': [r_0, r_1, r_2]
}
# s_1依赖于s_0和a_0
# a_1依赖于s_1
# 整个序列一起计算loss
```

#### 11.7 Episode在RulE-RL中的意义

**为什么需要Episode**:

1. **符合推理过程**: 知识图谱推理本质是多步游走
   ```
   (张三, grandfather, ?)
   → 不是一步得到答案
   → 需要沿着 father → father 走两步
   ```

2. **延迟奖励**: 只有走完路径才知道是否正确
   ```
   Step 0: 张三 → 李四 (不知道对错,reward=0)
   Step 1: 李四 → 王五 (到达目标!reward=1.609)
   ```

3. **策略优化**: 需要完整轨迹计算回报
   ```python
   G_0 = r_0 + γ*r_1 + γ²*r_2  # 需要整个episode的reward
   ```

---

### Q12: 规则一致性奖励怎么算的？

**答案**: 规则一致性奖励通过匹配Agent走过的路径与预训练规则库,使用RulE模型的规则嵌入计算置信度。

#### 12.1 规则一致性的定义

**目的**: 鼓励Agent遵循符号逻辑规则,而不是随机游走

**核心思想**:
```
如果Agent走的路径 = 某条逻辑规则的规则体
→ 给予奖励,奖励大小取决于该规则的质量
```

**示例**:
```python
# Agent走的路径
path = 张三 --father--> 李四 --father--> 王五
path_relations = [father, father]

# 规则库中的规则
rule_234: father ∧ father → grandfather (confidence=0.85)
rule_567: father ∧ mother → parent (不匹配)

# 匹配! 给予奖励0.85
```

#### 12.2 完整计算流程

```python
def compute_rule_consistency_reward(trajectory, target_entity, rules):
    """
    计算规则一致性奖励

    Args:
        trajectory: [(entity, relation), ...] Agent走过的路径
        target_entity: 目标实体
        rules: 预训练的规则库

    Returns:
        reward: 规则一致性奖励 [0, 1]
    """

    # ========== Step 1: 提取路径关系序列 ==========
    path_relations = [step[1] for step in trajectory if step[1] is not None]
    # 例如: [(张三,None), (李四,father), (王五,father)]
    # → path_relations = [father, father]

    if len(path_relations) == 0:
        return 0.0  # 没有走任何边

    # ========== Step 2: 查找匹配的规则 ==========
    matched_rules = []

    for rule in rules:
        # 规则体必须完全匹配
        if rule['body'] == path_relations:
            matched_rules.append(rule)

    # 例如:
    # path_relations = [father, father]
    # 匹配到: rule_234: father ∧ father → grandfather
    # 匹配到: rule_1024: father ∧ father → ancestor

    if len(matched_rules) == 0:
        return 0.0  # 没有匹配的规则

    # ========== Step 3: 计算每条匹配规则的置信度 ==========
    confidences = []

    for rule in matched_rules:
        confidence = compute_rule_confidence(rule)
        confidences.append(confidence)

    # ========== Step 4: 返回最高置信度 ==========
    return max(confidences)


def compute_rule_confidence(rule):
    """
    使用RulE预训练模型计算规则置信度

    Args:
        rule: {
            'id': 234,
            'head': 3,  # grandfather
            'body': [0, 0],  # [father, father]
            'confidence': None  # 需要计算
        }

    Returns:
        confidence: [0, 1] 规则质量得分
    """

    # ========== Step 1: 获取规则嵌入 ==========
    rule_id = rule['id']
    rule_emb = rule_model.rule_emb[rule_id]  # [100]

    # ========== Step 2: 获取规则头嵌入 ==========
    rule_head = rule['head']
    head_emb = rule_model.relation_embedding.weight[rule_head]  # [200]

    # ========== Step 3: 计算规则体嵌入 (求和) ==========
    rule_body = rule['body']  # [0, 0] = [father, father]

    body_embeddings = []
    for rel_id in rule_body:
        rel_emb = rule_model.relation_embedding.weight[rel_id]
        body_embeddings.append(rel_emb)

    body_sum = sum(body_embeddings)  # [200]

    # ========== Step 4: 使用RulE的规则评分公式 ==========
    # RulE的规则学习目标: body_sum + rule_emb ≈ head_emb

    # 计算距离
    distance = torch.norm(body_sum + rule_emb - head_emb, p=2)

    # 置信度 = gamma - distance
    # gamma_rule是预训练时的margin参数 (通常为5-8)
    confidence = gamma_rule - distance.item()

    # 归一化到[0,1]
    confidence = max(0.0, min(1.0, confidence / gamma_rule))

    return confidence
```

#### 12.3 数学原理

**RulE的规则嵌入学习**:

在预训练阶段,RulE学习规则嵌入,使得:
```
body_sum + rule_emb ≈ head_emb
```

其中:
- `body_sum`: 规则体中所有关系嵌入的和
- `rule_emb`: 规则的嵌入向量 (可学习)
- `head_emb`: 规则头关系的嵌入

**规则质量度量**:
```python
# 好规则: 距离小
distance = ||body_sum + rule_emb - head_emb||
# 例如: distance = 0.5

confidence = gamma_rule - distance
# confidence = 5.0 - 0.5 = 4.5

# 归一化
confidence = 4.5 / 5.0 = 0.9  # 高质量规则!

# 差规则: 距离大
distance = 4.8
confidence = (5.0 - 4.8) / 5.0 = 0.04  # 低质量规则
```

#### 12.4 完整计算示例

**场景**: 查询 `(张三, grandfather, 王五)`

```python
# ========== Agent的轨迹 ==========
trajectory = [
    (15, None),      # 张三 (起点)
    (45, 0),         # 李四 (通过father到达)
    (89, 0)          # 王五 (通过father到达)
]

path_relations = [0, 0]  # [father, father]

# ========== 规则库中的规则 ==========
rules = [
    {
        'id': 234,
        'head': 3,      # grandfather
        'body': [0, 0], # father ∧ father
    },
    {
        'id': 567,
        'head': 15,     # cures
        'body': [5, 8], # treats ∧ relieves
    },
    # ... 其他18,398条规则
]

# ========== Step 1: 匹配规则 ==========
matched_rules = []

for rule in rules:
    if rule['body'] == [0, 0]:  # 匹配path_relations
        matched_rules.append(rule)

# 结果: matched_rules = [rule_234, rule_1024, ...]

# ========== Step 2: 计算rule_234的置信度 ==========

# 获取嵌入
rule_emb = rule_model.rule_emb[234]  # [100] 维向量
# 例如: [0.15, -0.23, 0.45, ..., 0.12]

head_emb = relation_embedding[3]  # grandfather的嵌入 [200]
# 例如: [0.8, -0.5, 0.3, ..., 0.6]

father_emb = relation_embedding[0]  # father的嵌入 [200]
# 例如: [0.4, -0.2, 0.15, ..., 0.3]

# 规则体求和
body_sum = father_emb + father_emb  # [200]
# = [0.4, -0.2, 0.15, ...] + [0.4, -0.2, 0.15, ...]
# = [0.8, -0.4, 0.3, ..., 0.6]

# 计算距离
predicted = body_sum + rule_emb  # [200] (广播后)
# 简化计算,假设结果与head_emb接近

distance = ||predicted - head_emb||_2
# = ||[0.8,-0.4,0.3,...] - [0.8,-0.5,0.3,...]||
# = sqrt((0.0)² + (0.1)² + (0.0)² + ...)
# ≈ 0.72

# 置信度
gamma_rule = 5.0  # 预训练时的margin
confidence = (gamma_rule - distance) / gamma_rule
# = (5.0 - 0.72) / 5.0
# = 4.28 / 5.0
# = 0.856

# ========== Step 3: 如果有多条匹配规则 ==========
# rule_234: confidence = 0.856
# rule_1024: confidence = 0.725
# rule_3456: confidence = 0.623

# 选择最高
rule_consistency_reward = max(0.856, 0.725, 0.623) = 0.856
```

#### 12.5 为什么这样设计有效

**1. 利用预训练知识**:
```python
# RulE已经学习了:
# - 哪些规则是高质量的 (distance小)
# - 哪些规则是低质量的 (distance大)

# RL训练时直接复用这些知识
# 不需要重新学习规则质量
```

**2. 符号-神经混合**:
```python
# 符号部分: 规则匹配 (精确)
if rule['body'] == path_relations:  # 必须完全相同
    ...

# 神经部分: 置信度计算 (软性)
confidence = (gamma - distance) / gamma  # 连续值[0,1]
```

**3. 奖励塑形**:
```python
# 没有规则一致性奖励:
# - Agent随机游走
# - 即使到达目标,路径可能无意义
# - 例如: 张三 --sibling--> 李四 --spouse--> 王五 ✓ (但不符合grandfather逻辑)

# 有规则一致性奖励:
# - Agent倾向于遵循规则
# - 路径有逻辑意义
# - 例如: 张三 --father--> 李四 --father--> 王五 ✓ (符合grandfather规则)
```

#### 12.6 实际案例对比

**案例1: 符合规则的路径**
```python
# 查询: (aspirin, treats, headache)
# Agent路径: aspirin --treats--> pain --relieves--> headache

path_relations = [treats, relieves]

# 匹配规则
rule: treats ∧ relieves → cures (confidence=0.85)

# 奖励分解
final_reward = 1.0  # 到达目标
rule_consistency = 0.85 × 0.1 = 0.085  # 规则一致性
getting_closer = 0.52 × 0.1 = 0.052
total = 1.0 + 0.085 + 0.052 = 1.137
```

**案例2: 不符合规则的路径**
```python
# 查询: (aspirin, treats, headache)
# Agent路径: aspirin --interacts--> drug_x --located_in--> clinic

path_relations = [interacts, located_in]

# 没有匹配规则 (这个关系组合无意义)
rule_consistency = 0.0

# 没有到达目标
final_reward = -5.2  # 嵌入距离远

# 总奖励
total = -5.2 + 0.0 = -5.2  # 极差的episode
```

#### 12.7 可视化对比

```
好的Episode (符合规则):
┌────────────────────────────────────────┐
│ Query: (aspirin, treats, headache)    │
├────────────────────────────────────────┤
│ Path: aspirin → pain → headache       │
│       [treats]  [relieves]            │
├────────────────────────────────────────┤
│ Matched Rule: treats ∧ relieves → cures │
│ Confidence: 0.85                      │
├────────────────────────────────────────┤
│ Rewards:                              │
│  - Final: 1.0                         │
│  - Rule consistency: 0.085            │
│  - Getting closer: 0.052              │
│  Total: 1.137 ✓                       │
└────────────────────────────────────────┘

差的Episode (不符合规则):
┌────────────────────────────────────────┐
│ Query: (aspirin, treats, headache)    │
├────────────────────────────────────────┤
│ Path: aspirin → drug_x → clinic       │
│       [interacts] [located_in]        │
├────────────────────────────────────────┤
│ Matched Rule: None                    │
│ Confidence: 0.0                       │
├────────────────────────────────────────┤
│ Rewards:                              │
│  - Final: -5.2 (未到达)               │
│  - Rule consistency: 0.0              │
│  - Getting closer: 0.0                │
│  Total: -5.2 ✗                        │
└────────────────────────────────────────┘
```

---

### Q13: 一次训练一个查询会不会太慢了?标准RL是这样训练的吗?

**答案**: 逐查询训练在强化学习中是标准做法,虽然看似"慢",但这是RL任务特性决定的,并且可以通过多种技术优化。

#### 13.1 为什么逐查询训练是必要的?

**RL vs 监督学习的根本区别**:

```python
# 监督学习 (可以批量训练)
# 每个样本独立,可以并行处理
batch = [(x1, y1), (x2, y2), ..., (x32, y32)]  # batch_size=32
loss = sum([cross_entropy(model(x), y) for x, y in batch])
optimizer.step()

# 强化学习 (必须序列化训练)
# Episode是完整轨迹,必须走完才能计算回报
for query in queries:  # 逐个处理
    # 从头到尾完整走一遍
    trajectory = run_episode(query)
    # 走完后才知道总奖励
    returns = compute_returns(trajectory)
    # 用整条轨迹更新策略
    update_policy(trajectory, returns)
```

**为什么不能批量训练RL**:

| 原因 | 解释 | 示例 |
|------|------|------|
| **时序依赖** | 下一步依赖上一步的结果 | state_t+1 = f(state_t, action_t) |
| **延迟奖励** | 只有走完才知道好坏 | 只有到达目标才获得+1奖励 |
| **路径长度不同** | 每个查询episode长度不同 | Query1:2步, Query2:5步, Query3:3步 |
| **独立探索** | 每个查询需要独立决策 | 不同查询选不同规则、走不同路径 |

#### 13.2 标准RL是怎么做的?

**是的,逐episode训练是标准RL范式**:

1. **Atari游戏 (DQN)**:
   ```python
   # 玩一局游戏 = 一个episode
   for episode in range(num_episodes):
       state = env.reset()  # 重新开始游戏
       while not done:
           action = agent.select_action(state)
           next_state, reward, done = env.step(action)
           # 记录这一步
           buffer.store(state, action, reward, next_state)
       # Episode结束后更新
       agent.update(buffer)
   ```

2. **机器人控制 (PPO)**:
   ```python
   # 机器人执行一个任务 = 一个episode
   for task in tasks:
       trajectory = []
       state = env.reset()
       while not done:
           action = policy(state)
           next_state, reward, done = env.step(action)
           trajectory.append((state, action, reward))
       # 任务完成后更新策略
       update_policy(trajectory)
   ```

3. **AlphaGo (下围棋)**:
   ```python
   # 下一局棋 = 一个episode
   for game in range(num_games):
       board = reset_board()
       while not game_over:
           move = select_move(board)
           board = apply_move(board, move)
       # 棋局结束才知道输赢
       reward = +1 if win else -1
       update_network(game_trajectory, reward)
   ```

**结论**: **逐episode训练是RL的标准做法,不是RulE-RL特有的**!

#### 13.3 RulE-RL的训练速度实际如何?

**UMLS数据集实际训练时间估算**:

```python
# 训练参数
num_queries = 5,216  # 训练集大小
num_epochs = 100
avg_episode_steps = 3  # 平均每个episode 3步

# 时间估算 (单GPU)
time_per_step = 0.01s  # 10ms (前向传播+状态转移)
time_per_episode = 3 × 0.01s = 0.03s
time_per_epoch = 5,216 × 0.03s = 156s ≈ 2.6分钟
total_training_time = 100 × 2.6分钟 = 260分钟 ≈ 4.3小时

# 实际可能更快 (GPU并行计算):
实际训练时间: 2-3小时
```

**对比原RulE的训练时间**:

```python
# 原RulE预训练
time_per_batch = 0.1s  # batch_size=256
num_batches_per_epoch = 5,216 / 256 ≈ 20
time_per_epoch = 20 × 0.1s = 2s
预训练epochs = 30,000步 / 20 = 1,500 epochs
total_time = 1,500 × 2s = 3,000s ≈ 50分钟

# 原RulE grounding阶段
time_per_batch = 2s  # 需要grounding,很慢
num_batches = 5,216 / 16 = 326
time_per_epoch = 326 × 2s = 652s ≈ 11分钟
grounding_epochs = 20
total_time = 20 × 11分钟 = 220分钟 ≈ 3.7小时

# 总计: 50分钟 + 3.7小时 = 4.5小时
```

**结论**: **RulE-RL (4.3小时) 与原RulE (4.5小时) 训练时间相当**!

#### 13.4 为什么RulE-RL没有"特别慢"?

**1. Episode很短**:
```python
# 不是玩一局很长的游戏
max_steps = 5  # 最多5步就结束
avg_steps = 3  # 平均3步
# vs Atari游戏: 1000+ steps/episode
```

**2. 每步计算很快**:
```python
# 一步的计算:
- 状态编码: MLP forward (1ms)
- 策略网络: MLP forward (1ms)
- 环境转移: 邻接表查询 (0.1ms)
- 总计: ~2-3ms/step

# vs 图像游戏:
- CNN forward: 10-50ms/step
```

**3. 训练集不大**:
```python
# UMLS: 5,216个查询
# vs ImageNet: 1,000,000张图片
# vs Atari: 需要几百万帧
```

#### 13.5 RL训练加速技术

虽然必须逐查询训练,但有多种加速方法:

**1. 经验回放 (Experience Replay) - 不适用RulE-RL**:
```python
# 适用于: DQN等off-policy算法
# 原理: 存储过去的(s,a,r,s'),重复使用

buffer = ReplayBuffer(size=100000)
for episode in episodes:
    trajectory = run_episode()
    buffer.store(trajectory)
    # 从buffer采样batch更新 (可以并行)
    batch = buffer.sample(batch_size=32)
    update(batch)

# RulE-RL为什么不用:
# REINFORCE是on-policy算法,只能用当前策略的数据
# 无法重复使用旧轨迹
```

**2. 并行环境 (Vectorized Environments) - 可以用!**:
```python
# 同时运行多个episode (不同查询)
# PyTorch可以批量处理

# 串行版本 (当前)
for query in queries:  # 5,216次循环
    episode = run_episode(query)
    update(episode)
# 耗时: 156s

# 并行版本 (优化)
batch_size = 32
for batch in batched_queries:  # 5,216/32 = 163次循环
    episodes = run_episodes_parallel(batch)  # 32个episode同时跑
    update(episodes)
# 耗时: 156s / 32 = 5s !!! (理论加速32x)

# 实现:
class VectorizedEnv:
    def reset(self, queries):  # 接收32个查询
        self.states = [env.reset(q) for q in queries]
        return torch.stack(self.states)  # [32, 128]

    def step(self, actions):  # 接收32个动作
        next_states, rewards, dones = [], [], []
        for i, action in enumerate(actions):
            s, r, d, _ = self.envs[i].step(action)
            next_states.append(s)
            rewards.append(r)
            dones.append(d)
        return (torch.stack(next_states),  # [32, 128]
                torch.tensor(rewards),      # [32]
                torch.tensor(dones))        # [32]

# Agent批量处理
states = vec_env.reset(batch_queries)  # [32, 128]
actions = agent.select_action(states)   # 一次forward处理32个! [32]
next_states, rewards, dones = vec_env.step(actions)
```

**3. 分布式训练 (Distributed RL) - 可以用!**:
```python
# 多个worker并行采集数据

# Worker 1: 处理query 1-1000
# Worker 2: 处理query 1001-2000
# Worker 3: 处理query 2001-3000
# ...
# 中心Server: 聚合梯度,更新参数

# 伪代码
def worker(worker_id, queries):
    for query in queries:
        episode = run_episode(query)
        gradients = compute_gradients(episode)
        send_to_server(gradients)

# 8个worker → 8x加速!
```

**4. 课程学习 (Curriculum Learning)**:
```python
# 不是所有查询都同等重要
# 先训练简单查询,再训练难查询

# 按难度排序
queries_sorted = sort_by_difficulty(train_queries)
# 简单: 1-hop关系 (直接相连)
# 中等: 2-hop关系 (需要1个中间节点)
# 困难: 3-hop+ 关系

# 分阶段训练
# Stage 1: 只训练简单查询 (1,000个, 10 epochs)
# Stage 2: 加入中等查询 (3,000个, 30 epochs)
# Stage 3: 全部查询 (5,216个, 60 epochs)

# 总epochs虽然一样,但学习更快,效果更好!
```

#### 13.6 实际训练时间对比 (各种方法)

| 方法 | 实现难度 | 加速比 | 实际训练时间 (UMLS) |
|------|---------|-------|-----------------|
| **当前 (逐查询)** | 简单 | 1x | 4.3小时 |
| **+ 向量化环境 (batch=32)** | 中等 | 20-30x | 8-13分钟 |
| **+ 分布式训练 (8 workers)** | 困难 | 8x | 1-2分钟 |
| **+ 课程学习** | 中等 | 1.5-2x | 5-9分钟 |
| **组合 (向量化+课程)** | 中等 | 30-40x | 6-8分钟 |

**结论**:
- 当前方法虽然逐查询,但**4.3小时完全可接受**
- 如果需要加速,**向量化环境是最佳选择** (20-30x加速,实现中等难度)
- 分布式训练虽然最快,但实现复杂,不是必需的

#### 13.7 对比:为什么监督学习看起来"快"?

**并不是监督学习真的快,而是性质不同**:

```python
# 监督学习 (如原RulE)
# 看起来快: batch_size=256, 每次处理256个样本
# 但需要epochs: 通常30,000+ steps才收敛
for step in range(30000):
    batch = sample(train_set, batch_size=256)
    loss = compute_loss(batch)
    optimizer.step()
# 总样本处理量: 30,000 × 256 = 7,680,000 个样本!

# 强化学习 (RulE-RL)
# 看起来慢: 一次一个查询
# 但epochs少: 通常100 epochs就够了
for epoch in range(100):
    for query in train_queries:  # 5,216个
        episode = run_episode(query)
        update(episode)
# 总样本处理量: 100 × 5,216 = 521,600 个episode
# 平均每个episode 3步 = 1,564,800 步决策

# 实际计算量相当！
```

**关键区别**:

| 维度 | 监督学习 | 强化学习 (RulE-RL) |
|------|----------|-----------------|
| **样本性质** | 静态 (x, y)对 | 动态轨迹 |
| **并行性** | 高 (样本独立) | 中 (可向量化,但受限) |
| **样本效率** | 低 (需要大量数据) | 高 (从探索中学习) |
| **收敛速度** | 需要多轮epoch | 较少epochs |
| **总训练时间** | 相当 | 相当 |

---

### Q14: UCB是什么?

**答案**: UCB (Upper Confidence Bound,上置信界)是一种经典的探索-利用平衡算法,源自多臂老虎机(Multi-Armed Bandit)问题。

#### 14.1 UCB的核心思想

**探索-利用困境 (Exploration-Exploitation Dilemma)**:

```
假设你在赌场,有3台老虎机:
- 老虎机A: 玩过100次,平均赢50元
- 老虎机B: 玩过10次,平均赢60元
- 老虎机C: 玩过1次,赢了80元

你应该选哪台?

利用 (Exploitation): 选A (数据最多,最稳)
探索 (Exploration): 选C (可能更好,但数据少)
平衡: UCB算法!
```

**UCB公式**:

```
UCB(i) = Q̂(i) + c × sqrt(log(N) / n_i)
        ↑        ↑
      利用项   探索bonus

其中:
- Q̂(i): 臂i的平均奖励 (利用已知信息)
- n_i: 臂i被选择的次数
- N: 总选择次数
- c: 探索系数 (常取√2)
```

**直觉理解**:

```python
# 臂A: 玩100次,平均50元
UCB(A) = 50 + sqrt(2×log(111)/100) = 50 + 0.14 = 50.14

# 臂B: 玩10次,平均60元
UCB(B) = 60 + sqrt(2×log(111)/10) = 60 + 0.99 = 60.99

# 臂C: 玩1次,平均80元
UCB(C) = 80 + sqrt(2×log(111)/1) = 80 + 2.96 = 82.96

# 选择: UCB最大的 → 选C!
# 原因: 虽然C的真实期望可能不如看起来好,
#       但它被探索太少,给它一个bonus鼓励探索
```

#### 14.2 UCB在RulE-RL中的应用

**场景**: 高层Agent选择规则

```python
# 有18,400条规则,需要选Top-5
# 问题: 怎么平衡"选好规则"和"探索新规则"?

# 不用UCB (贪心):
# 总是选神经网络得分最高的5条
selected = topk(neural_scores, k=5)
# 问题: 永远不会探索得分低的规则,可能错过好规则

# 用UCB:
# 给探索少的规则bonus
ucb_scores = neural_scores + exploration_bonus
selected = topk(ucb_scores, k=5)
```

**完整实现**:

```python
class RuleSelectorAgent:
    def __init__(self, num_rules):
        # UCB统计
        self.rule_counts = defaultdict(int)      # 每条规则被选次数
        self.rule_rewards = defaultdict(float)   # 每条规则累积奖励
        self.total_selections = 0                # 总选择次数

    def select_rules(self, query, rule_embeddings, top_k=5):
        # Step 1: 神经网络得分
        neural_scores = self.compute_neural_scores(query, rule_embeddings)
        # neural_scores[i]: 规则i与查询的相关性 (学习的)

        # Step 2: UCB得分
        ucb_scores = torch.zeros_like(neural_scores)

        for rule_id in range(len(rule_embeddings)):
            # 平均奖励 (利用项)
            avg_reward = self.rule_rewards[rule_id] / (self.rule_counts[rule_id] + 1)

            # 探索bonus
            exploration_bonus = sqrt(
                2 * log(self.total_selections + 1) /
                (self.rule_counts[rule_id] + 1)
            )

            # UCB总分
            ucb_scores[rule_id] = neural_scores[rule_id] + exploration_bonus

        # Step 3: 选择UCB得分最高的Top-K
        _, selected_indices = torch.topk(ucb_scores, k=top_k)

        return selected_indices

    def update_statistics(self, selected_rules, episode_reward):
        """Episode结束后更新统计"""
        for rule_id in selected_rules:
            self.rule_counts[rule_id] += 1
            self.rule_rewards[rule_id] += episode_reward
            self.total_selections += 1
```

#### 14.3 UCB工作示例

**场景**: 查询 `(aspirin, treats, ?)`

```python
# ========== Episode 1: 初始状态 ==========
rule_counts = {234: 0, 567: 0, 1023: 0, ...}  # 所有规则都没被选过
rule_rewards = {234: 0, 567: 0, 1023: 0, ...}
total_selections = 0

# 神经得分
neural_scores = {
    234: 0.85,  # treats ∧ relieves → cures
    567: 0.72,  # treats ∧ prevents → protects
    1023: 0.68, # prescribes ∧ uses → administers
}

# UCB得分 (第一次,所有规则bonus相同)
exploration_bonus = sqrt(2×log(1)/1) = ∞  # 第一次,选择次数为0
# 实际实现: +1避免除0
exploration_bonus = sqrt(2×log(1)/(0+1)) = 0

# UCB = neural_scores + 0
# 第一次选择: 按神经得分排序
selected = [234, 567, 1023, ...]  # Top-5

# Episode 1结束,奖励 = 1.2
update_statistics([234, 567, 1023, ...], reward=1.2)
# rule_counts = {234: 1, 567: 1, 1023: 1, ...}
# rule_rewards = {234: 1.2, 567: 1.2, 1023: 1.2, ...}
# total_selections = 5


# ========== Episode 2: 同样的查询 ==========
# 神经得分不变
neural_scores = {
    234: 0.85,
    567: 0.72,
    1023: 0.68,
    89: 0.45,  # 这条规则之前没被选
}

# UCB得分
# 规则234: 被选过1次
ucb_234 = 0.85 + sqrt(2×log(5)/1) = 0.85 + 1.70 = 2.55

# 规则567: 被选过1次
ucb_567 = 0.72 + sqrt(2×log(5)/1) = 0.72 + 1.70 = 2.42

# 规则89: 没被选过
ucb_89 = 0.45 + sqrt(2×log(5)/(0+1)) = 0.45 + 1.70 = 2.15

# 其他规则: 没被选过
ucb_others = neural_score + 1.70

# 选择: UCB最高的 Top-5
# 234会被选(UCB高)
# 89可能被选(虽然神经得分低,但bonus大)


# ========== Episode 100: 经过多次训练 ==========
rule_counts = {
    234: 50,   # 被选50次 (很受欢迎)
    567: 30,   # 被选30次
    1023: 10,  # 被选10次
    89: 2,     # 只被选2次
    456: 0,    # 从未被选
}

rule_rewards = {
    234: 50×0.9 = 45,   # 平均奖励 0.9 (很好!)
    567: 30×0.5 = 15,   # 平均奖励 0.5 (一般)
    1023: 10×0.3 = 3,   # 平均奖励 0.3 (不好)
    89: 2×1.5 = 3,      # 平均奖励 1.5 (非常好,但样本少!)
    456: 0,
}

total_selections = 500

# UCB得分
ucb_234 = 0.85 + sqrt(2×log(500)/50) = 0.85 + 0.76 = 1.61

ucb_567 = 0.72 + sqrt(2×log(500)/30) = 0.72 + 0.98 = 1.70

ucb_1023 = 0.68 + sqrt(2×log(500)/10) = 0.68 + 1.70 = 2.38

ucb_89 = 0.45 + sqrt(2×log(500)/2) = 0.45 + 3.80 = 4.25 ← 最高!

ucb_456 = 0.40 + sqrt(2×log(500)/(0+1)) = 0.40 + 3.72 = 4.12

# 选择: 89会被优先选择!
# 原因: 虽然它被选次数少,但平均奖励高(1.5),
#       bonus也大,UCB得分最高
```

#### 14.4 UCB的数学性质

**1. 自适应探索**:
```python
# 探索bonus随选择次数减少
n_i = 1   → bonus = sqrt(2×log(N)/1) = 很大
n_i = 10  → bonus = sqrt(2×log(N)/10) = 中等
n_i = 100 → bonus = sqrt(2×log(N)/100) = 很小

# 结果:
# - 新规则自动获得更多探索机会
# - 已充分探索的规则bonus减小
# - 探索会自然收敛到最优
```

**2. 理论保证**:
```python
# UCB算法有遗憾界 (Regret Bound)保证:
# Regret = 累积奖励损失 (相比最优策略)
#
# UCB的遗憾界: O(sqrt(K × N × log(N)))
#   K: 臂数量 (规则数)
#   N: 总选择次数
#
# 含义: 随着时间增长,UCB会以次线性速度收敛到最优
```

**3. 对数增长**:
```python
# log(N)增长很慢
log(10) = 2.3
log(100) = 4.6
log(1000) = 6.9
log(10000) = 9.2

# 结果: bonus随时间缓慢衰减,
#       保证long-term exploration
```

#### 14.5 UCB vs 其他探索策略

| 策略 | 探索方式 | 优点 | 缺点 | 适用场景 |
|------|---------|------|------|---------|
| **ε-greedy** | ε概率随机,1-ε贪心 | 简单 | 浪费探索 | 简单任务 |
| **UCB** | 自适应bonus | 理论保证,高效 | 计算稍复杂 | 多臂问题,规则选择 |
| **Thompson Sampling** | 贝叶斯采样 | 理论最优 | 需要先验分布 | 贝叶斯框架 |
| **Softmax** | 按概率采样 | 平滑 | 需调温度参数 | 连续动作 |

**具体对比**:

```python
# ε-greedy (RulE-RL也用了,但在UCB基础上)
if random() < epsilon:
    selected = random_choice(rules, k=5)  # 10%随机
else:
    selected = topk(ucb_scores, k=5)      # 90%贪心

# 纯UCB (RulE-RL使用)
# 不需要ε,自动平衡
selected = topk(ucb_scores, k=5)
# 早期: bonus大,自动探索
# 后期: bonus小,自动利用

# 区别:
# ε-greedy: 探索是"盲目"的,随机选任何规则
# UCB: 探索是"聪明"的,优先探索有潜力的规则 (平均奖励高但样本少)
```

#### 14.6 为什么RulE-RL既用UCB又用ε-greedy?

```python
# RulE-RL的实际实现
def select_rules(self, ..., epsilon=0.1):
    # 计算UCB分数 (已经考虑探索)
    ucb_scores = neural_scores + exploration_bonus

    # 在UCB基础上再加ε-greedy
    if random() < epsilon:  # 10%概率
        selected = random.sample(rules, k=5)
    else:  # 90%概率
        selected = topk(ucb_scores, k=5)
```

**两层探索**:

1. **UCB探索** (主要):
   - 智能探索: 优先探索"有希望"的规则
   - 始终生效

2. **ε-greedy探索** (辅助):
   - 随机探索: 防止完全陷入局部最优
   - 只在训练早期生效 (epsilon逐渐衰减)
   - Epoch 0: ε=0.5 (50%随机)
   - Epoch 50: ε=0.05 (5%随机)
   - Epoch 100: ε=0.05 (5%随机)

**为什么要两层?**

```python
# 只用UCB的问题:
# 如果某条规则真的很差,UCB会"永远"给它很低的分数
# 可能永远不会被探索 (即使它在某些特殊查询下很好)

# 加上ε-greedy:
# 即使UCB认为某规则很差,
# 仍有小概率(5%)被随机选中
# 避免"过度自信"
```

#### 14.7 UCB的可视化

**规则选择动态过程**:

```
Epoch 1 (探索为主):
┌────────────────────────────────────────┐
│ Rule_234: 选择10次, avg=0.8, bonus=1.2 │  UCB=2.0
│ Rule_567: 选择5次,  avg=0.6, bonus=1.5 │  UCB=2.1 ← 选!
│ Rule_89:  选择1次,  avg=0.9, bonus=2.8 │  UCB=3.7 ← 选!
│ Rule_456: 选择0次,  avg=0,   bonus=∞   │  UCB=∞  ← 选!
└────────────────────────────────────────┘

Epoch 50 (平衡):
┌────────────────────────────────────────┐
│ Rule_234: 选择800次, avg=0.85, bonus=0.3│ UCB=1.15 ← 选
│ Rule_567: 选择200次, avg=0.50, bonus=0.6│ UCB=1.10
│ Rule_89:  选择50次,  avg=0.90, bonus=1.2│ UCB=2.10 ← 选!
│ Rule_456: 选择5次,   avg=0.20, bonus=2.0│ UCB=2.20 ← 探索
└────────────────────────────────────────┘

Epoch 100 (利用为主):
┌────────────────────────────────────────┐
│ Rule_234: 选择2000次, avg=0.85, bonus=0.1│ UCB=0.95 ← 选
│ Rule_567: 选择500次,  avg=0.50, bonus=0.2│ UCB=0.70
│ Rule_89:  选择1500次, avg=0.90, bonus=0.1│ UCB=1.00 ← 选!
│ Rule_456: 选择50次,   avg=0.20, bonus=0.5│ UCB=0.70
└────────────────────────────────────────┘
```

---

**文档版本**: v1.4
**更新日期**: 2024年11月22日
**作者**: RulE-RL项目组
