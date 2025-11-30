# RulE 单层强化学习改造方案

本文面向需要将现有 RulE-RL 层次化结构压缩为单层 Actor-Critic 的场景，说明整体思路、具体改动点、训练流程，并给出训练时间/显存控制建议和可参考的标准 RL 代码框架。

## 1. 目标与整体思路
- **保留** 预训练阶段：继续用 `PreTrainer` 学习实体/关系/规则嵌入，让规则在奖励阶段仍然具备语义。
- **删除** 高层 `RuleSelectorAgent`：只保留一个 PathFinder 式的 actor-critic，直接在剪枝后的动作空间上决策。
- **规则作用方式**：不再“先选规则”，而是通过动作剪枝与奖励塑形影响单层 Agent。
- **关注点**：减小动作空间（Top‑ε 剪枝 + 图邻接掩码）、限制 episode 步数、关闭不必要的 autograd 开销，从而控制训练时间与显存。

## 2. 模块改动清单
1. **配置文件**
   - 删除 `top_k_rules`、`epsilon_start`、`epsilon_end`、`ucb_c`、`lr_selector` 等只给高层 Agent 使用的字段。
   - 新增或保留的关键参数：`rl_max_steps`、`num_epochs`、`grad_clip`、`top_epsilon` (用于剪枝)、`debug_train_query_limit`。

2. **主入口 `src/main.py`**
   - 取消 `RuleSelectorAgent` 相关初始化与优化器，只留 `StateEncoder`、`PathFinderAgent`、`KGReasoningEnv`、`RewardCalculator`。
   - `KGReasoningEnv.reset(query)` 只接收 `(h, r, t)`，无需 `selected_rules`。

3. **状态编码 `StateEncoder`**
   - 删去 `rule_context` 分支，或改成“查询关系的规则统计” (例如平均规则嵌入) 以保持状态维度。
   - 仍保留当前实体、查询关系、历史路径（GRU）三个部分，输出统一的 `state_dim`。

4. **环境与动作掩码 `KGReasoningEnv`**
   - `_build_adjacency` 不变，`get_action_mask` 改为：
     ```python
     outgoing = adjacency[current_entity]
     candidate = outgoing
     if rule_whitelist[query_rel]:
         candidate = outgoing ∩ rule_whitelist[query_rel]
     if use_top_epsilon:
         candidate = top_epsilon_by_kge(candidate, epsilon=top_epsilon)
     ```
   - 其中 `rule_whitelist` 可在初始化时按“查询关系 → 规则体关系集合”构建，保证规则仍对动作空间起过滤作用。

5. **Trainer `RulERLTrainer`**
   - `train_episode` 只负责 PathFinder 的轨迹采样，不再返回 `selector_loss`。
   - 优化器与梯度更新仅包含 `policy_optimizer`、`value_optimizer`。
   - 日志中可增加剪枝后动作数量、Top‑ε 统计等信息，便于诊断训练时间。

6. **奖励 `RewardCalculator`**
   - 保持不变：`R_total = R_final + α * (R_rule + (1 - R_final) * R_closer)`。单层模式下规则一致性奖励的重要性更高，可根据实验调节 `α`。

## 3. 单层训练流程
1. **预训练阶段**
   - `PreTrainer.train(args)` → 保存 checkpoint、规则嵌入、实体/关系嵌入。
2. **RL 初始化**
   - 加载 checkpoint → 冻结 `RulE_model` → 初始化环境、单层 Agent、优化器。
3. **Episode 采样**
   - 对每个训练查询：
     1. `state = env.reset(query)`
     2. 循环 ≤ `rl_max_steps`：
        - `mask = env.get_action_mask()`（图邻接 ∩ 规则白名单，再 加 Top‑ε 剪枝）
        - `action, log_prob, value = path_finder.select_action(state, mask)`
        - `next_state, reward, done, info = env.step(action)`
        - 记录 `(state, action, log_prob, value, reward)`，`state = next_state`
4. **回报与优势**
   - `returns = discount(rewards, gamma)`，`values = value_net(states)`，`advantages = (returns - values.detach())` 并标准化。
5. **更新 Actor-Critic**
   - `policy_loss = -(log_probs * advantages).mean()` → 反向传播 → 梯度裁剪 → `policy_optimizer.step()`
   - `value_loss = mse(values, returns)` → 反向传播 → `value_optimizer.step()`
6. **监控与保存**
   - 每 `log_interval` 步打印平均奖励、成功率、剪枝后动作数。
   - 每 `eval_interval` epoch 跑验证集，以 `deterministic=True` 评估并保存最佳 checkpoint。

## 4. 训练时间与显存控制
| 问题 | 对策 |
|------|------|
| 动作空间过大导致每步前向慢 | 启用 Top‑ε 剪枝 (`epsilon≈20~40`)，同时 `rl_max_steps` 控制在 3~5；对大图可引入 `debug_train_query_limit` 做小数据 sanity check。 |
| Episode 堆叠导致显存占用 | `rl_max_steps` 小于 6 时，`states/log_probs` 等缓存量极低；确保 `torch.autograd.set_detect_anomaly(False)`，并按 episode 级别释放局部列表。 |
| 训练集太大导致 epoch 很慢 | 采样部分查询（如每 epoch 随机抽 `N` 条），或将 `num_epochs` 调低并在每个 epoch 内随机打乱查询顺序。 |
| 邻接表占用 CPU 内存 | 若数据集极大，可换用稀疏存储（CSR）或动态邻接查询；也可把 `adjacency` 构建为只含出边实体的字典，避免复制所有关系。 |

## 5. 参考的标准 Actor-Critic 代码
```python
class SingleLayerTrainer:
    def train_episode(self, query):
        states, actions, log_probs, values, rewards = [], [], [], [], []
        state = self.env.reset(query)
        done = False
        while not done:
            mask = self.env.get_action_mask()
            action, log_prob, value = self.policy.select_action(state, mask)
            next_state, reward, done, _ = self.env.step(action)
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            state = next_state

        returns = self._discount(rewards, self.args.gamma)
        advantages = (returns - torch.stack(values).detach())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(torch.stack(log_probs) * advantages).mean()
        value_loss = F.mse_loss(torch.stack(values), returns)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.grad_clip)
        self.policy_optim.step()

        self.value_optim.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), self.args.grad_clip)
        self.value_optim.step()
```
上面是标准 on-policy Actor-Critic 的核心逻辑。结合 RulE，只需把 `env.reset/query/state_encoder` 换成对应实现，并在 `get_action_mask` 中执行规则白名单 + Top‑ε 剪枝即可。

## 6. 备选增强点
1. **规则特征拼接**：可把查询关系对应的规则嵌入平均值拼进状态向量，增加规则语义。
2. **混合奖励**：对 `R_rule` 增加温度或自适应权重，让规则在单层结构中发挥更强的约束作用。
3. **经验采样**：如果要进一步缩短训练时间，可选用 replay buffer + off-policy 算法（如 DQN/Soft Actor-Critic），但需额外改写 Reward/Env 接口。

以上方案即可在保证规则信息仍被利用的前提下，将 RulE-RL 改造成单层强化学习模式，并对训练时间与显存做出针对性的优化。
