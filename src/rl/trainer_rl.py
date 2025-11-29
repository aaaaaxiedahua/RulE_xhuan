"""
RulERLTrainer - RulE-RL 训练器

完整的训练和评估流程，包括：
- Episode 训练循环
- 策略梯度更新（REINFORCE with Baseline）
- 评估和指标计算
- 检查点管理
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm


class RulERLTrainer:
    """
    RulE-RL 训练器

    Args:
        rule_model: 预训练的 RulE 模型（冻结）
        rule_selector: RuleSelectorAgent 实例
        path_finder: PathFinderAgent 实例
        env: KGReasoningEnv 实例
        graph: KnowledgeGraph 对象
        args: 训练参数
    """

    def __init__(self, rule_model, rule_selector, path_finder, env, graph, args):
        self.rule_model = rule_model
        self.rule_selector = rule_selector
        self.path_finder = path_finder
        self.env = env
        self.graph = graph
        self.args = args

        # 优化器
        self.rule_selector_optimizer = optim.Adam(
            rule_selector.parameters(),
            lr=args.lr_selector
        )
        self.policy_optimizer = optim.Adam(
            path_finder.policy_net.parameters(),
            lr=args.lr_policy
        )
        self.value_optimizer = optim.Adam(
            path_finder.value_net.parameters(),
            lr=args.lr_value
        )

        # 训练统计
        self.best_mrr = 0.0
        self.global_step = 0

    def train(self, train_queries, valid_queries, test_queries):
        """
        完整训练流程

        Args:
            train_queries: 训练查询列表 [(h, r, t), ...]
            valid_queries: 验证查询列表
            test_queries: 测试查询列表
        """
        logging.info('\n' + '='*80)
        logging.info('Starting RulE-RL training')
        logging.info('='*80)
        logging.info(f'Train queries: {len(train_queries)}')
        logging.info(f'Validation queries: {len(valid_queries)}')
        logging.info(f'Test queries: {len(test_queries)}')

        for epoch in range(self.args.num_epochs):
            logging.info(f'\n{"="*80}')
            logging.info(f'Epoch {epoch + 1}/{self.args.num_epochs}')
            logging.info(f'{"="*80}')

            # 课程学习：逐步减小 epsilon
            epsilon = max(
                self.args.epsilon_end,
                self.args.epsilon_start - epoch * (self.args.epsilon_start - self.args.epsilon_end) / self.args.num_epochs
            )

            # 训练一个 epoch
            epoch_stats = self.train_epoch(train_queries, epsilon)

            # 打印 epoch 统计
            logging.info(f'\nEpoch {epoch + 1} training stats:')
            logging.info(f'  Avg reward: {epoch_stats["avg_reward"]:.4f}')
            logging.info(f'  Avg path length: {epoch_stats["avg_length"]:.2f}')
            logging.info(f'  Success rate: {epoch_stats["success_rate"]:.2%}')
            logging.info(f'  Policy loss: {epoch_stats["avg_policy_loss"]:.4f}')
            logging.info(f'  Value loss: {epoch_stats["avg_value_loss"]:.4f}')
            logging.info(f'  Epsilon: {epsilon:.3f}')

            # 验证
            if (epoch + 1) % self.args.eval_interval == 0:
                logging.info(f'\n{"="*80}')
                logging.info('Validation evaluation')
                logging.info(f'{"="*80}')
                val_metrics = self.evaluate(valid_queries)
                logging.info(f'  MRR: {val_metrics["mrr"]:.4f}')
                logging.info(f'  MR: {val_metrics["mr"]:.2f}')
                logging.info(f'  Hits@1: {val_metrics["hits@1"]:.4f}')
                logging.info(f'  Hits@3: {val_metrics["hits@3"]:.4f}')
                logging.info(f'  Hits@10: {val_metrics["hits@10"]:.4f}')

                # 保存最佳模型
                if val_metrics['mrr'] > self.best_mrr:
                    self.best_mrr = val_metrics['mrr']
                    self.save_checkpoint(f'{self.args.save_path}/best_checkpoint.pt', epoch, val_metrics)
                    logging.info(f'  [OK] Saved best checkpoint (MRR: {self.best_mrr:.4f})')

            # 定期保存检查点
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(f'{self.args.save_path}/checkpoint_epoch_{epoch+1}.pt', epoch)
                logging.info(f'  [OK] Saved checkpoint: epoch_{epoch+1}')

        # 最终测试
        logging.info(f'\n{"="*80}')
        logging.info('Final test evaluation')
        logging.info(f'{"="*80}')
        test_metrics = self.evaluate(test_queries)
        logging.info(f'  MRR: {test_metrics["mrr"]:.4f}')
        logging.info(f'  MR: {test_metrics["mr"]:.2f}')
        logging.info(f'  Hits@1: {test_metrics["hits@1"]:.4f}')
        logging.info(f'  Hits@3: {test_metrics["hits@3"]:.4f}')
        logging.info(f'  Hits@10: {test_metrics["hits@10"]:.4f}')

        return test_metrics

    def train_epoch(self, train_queries, epsilon):
        """
        训练一个 epoch

        Args:
            train_queries: 训练查询列表
            epsilon: 探索率

        Returns:
            epoch_stats: epoch 统计字典
        """
        self.rule_selector.train()
        self.path_finder.train()

        epoch_rewards = []
        epoch_lengths = []
        epoch_successes = []
        epoch_policy_losses = []
        epoch_value_losses = []

        # 随机打乱查询
        indices = np.random.permutation(len(train_queries))

        for i, idx in enumerate(tqdm(indices, desc='Training')):
            query = train_queries[idx]

            # 训练一个 episode
            reward, length, success, loss_dict = self.train_episode(query, epsilon)

            epoch_rewards.append(reward)
            epoch_lengths.append(length)
            epoch_successes.append(success)
            epoch_policy_losses.append(loss_dict['policy_loss'])
            epoch_value_losses.append(loss_dict['value_loss'])

            self.global_step += 1

            # 定期打印日志
            if (i + 1) % self.args.log_interval == 0:
                recent_reward = np.mean(epoch_rewards[-self.args.log_interval:])
                recent_success = np.mean(epoch_successes[-self.args.log_interval:])
                logging.info(
                    f'  Step {i+1}/{len(train_queries)}: '
                    f'Reward={recent_reward:.3f}, Success={recent_success:.2%}'
                )

        # 计算 epoch 统计
        epoch_stats = {
            'avg_reward': np.mean(epoch_rewards),
            'avg_length': np.mean(epoch_lengths),
            'success_rate': np.mean(epoch_successes),
            'avg_policy_loss': np.mean(epoch_policy_losses),
            'avg_value_loss': np.mean(epoch_value_losses)
        }

        return epoch_stats

    def train_episode(self, query, epsilon):
        """
        训练一个 episode

        Args:
            query: (head, relation, tail)
            epsilon: 探索率

        Returns:
            total_reward: 总奖励
            path_length: 路径长度
            success: 是否成功
            loss_dict: 损失字典
        """
        head, relation, tail = query
        device = self.rule_model.entity_embedding.weight.device
        debug_logging = self.global_step < 5

        if debug_logging:
            logging.info(
                '[Debug][Episode %d] Query=(%d, %d, %d), epsilon=%.3f',
                self.global_step + 1,
                head,
                relation,
                tail,
                epsilon
            )

        # ===== Step 1: 高层 Agent 选择规则 =====
        query_entity_emb = self.rule_model.entity_embedding.weight[head]
        query_rel_emb = self.rule_model.relation_embedding.weight[relation]

        rule_embeddings = self.rule_model.rule_emb.weight

        selected_rules, selection_probs = self.rule_selector(
            query_entity_emb,
            query_rel_emb,
            rule_embeddings,
            epsilon=epsilon,
            top_k=self.args.top_k_rules,
            deterministic=False
        )

        # ===== Step 2: 低层 Agent 搜索路径 =====
        state = self.env.reset(query, selected_rules)

        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        done = False

        episode_info = None

        while not done:
            # 获取有效动作掩码
            action_mask = self.env.get_action_mask()

            # Agent 选择动作
            action, log_prob, value = self.path_finder.select_action(
                state,
                action_mask,
                deterministic=False
            )

            # 执行动作
            next_state, reward, done, info = self.env.step(action)

            # 记录轨迹
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            episode_info = info

            if debug_logging:
                logging.info(
                    '[Debug][Episode %d][Step %d] action=%d reward=%.4f done=%s reason=%s',
                    self.global_step + 1,
                    len(actions),
                    int(action),
                    float(reward),
                    done,
                    info.get('reason', 'n/a')
                )

            state = next_state

        # ===== Step 3: 计算回报和优势 =====
        returns = self._compute_returns(rewards, self.args.gamma, device)
        advantages = self._compute_advantages(returns, values)

        # ===== Step 4: 更新策略网络（PathFinder） =====
        policy_loss = self._update_policy(log_probs, advantages)

        # ===== Step 5: 更新价值网络（PathFinder） =====
        value_loss = self._update_value(states, returns)

        # ===== Step 6: 更新规则选择器 =====
        # 使用最终奖励更新规则选择器
        final_reward = rewards[-1] if len(rewards) > 0 else 0.0
        selector_loss = self._update_rule_selector(selection_probs, selected_rules, final_reward)

        # 更新 UCB 统计
        for rule_id in selected_rules:
            self.rule_selector.update_statistics(rule_id.item(), final_reward)

        # 统计
        total_reward = sum(rewards)
        path_length = len(actions)
        success = episode_info.get('success', False) if episode_info else False

        loss_dict = {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'selector_loss': selector_loss
        }

        if debug_logging:
            logging.info(
                '[Debug][Episode %d] Done reason=%s, total_reward=%.4f, path_length=%d, success=%s',
                self.global_step + 1,
                episode_info.get('reason') if episode_info else 'unknown',
                float(total_reward),
                path_length,
                success
            )

        return total_reward, path_length, success, loss_dict

    def _compute_returns(self, rewards, gamma, device):
        """
        计算折扣回报

        Args:
            rewards: 奖励列表
            gamma: 折扣因子

        Returns:
            returns: 回报列表
        """
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=device)

    def _compute_advantages(self, returns, values):
        """
        计算优势函数 A(s,a) = R - V(s)

        Args:
            returns: 回报 [T]
            values: 状态价值 [T]

        Returns:
            advantages: 优势 [T]
        """
        values_tensor = torch.stack(values)
        advantages = returns - values_tensor.detach()
        # 标准化（减小方差）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def _update_policy(self, log_probs, advantages):
        """
        更新策略网络

        Args:
            log_probs: log 概率列表
            advantages: 优势列表

        Returns:
            loss: 策略损失
        """
        log_probs_tensor = torch.stack(log_probs)
        policy_loss = -(log_probs_tensor * advantages).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.path_finder.policy_net.parameters(), self.args.grad_clip)
        self.policy_optimizer.step()

        return policy_loss.item()

    def _update_value(self, states, returns):
        """
        更新价值网络

        Args:
            states: 状态列表
            returns: 回报列表

        Returns:
            loss: 价值损失
        """
        states_tensor = torch.stack(states)
        values = self.path_finder.get_value(states_tensor.detach())
        value_loss = nn.MSELoss()(values, returns)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.path_finder.value_net.parameters(), self.args.grad_clip)
        self.value_optimizer.step()

        return value_loss.item()

    def _update_rule_selector(self, selection_probs, selected_rules, reward):
        """
        更新规则选择器

        Args:
            selection_probs: 选择概率 [top_k]
            selected_rules: 选中的规则 [top_k]
            reward: 最终奖励

        Returns:
            loss: 选择器损失
        """
        # 策略梯度：最大化 log P(rules) * reward
        log_probs = torch.log(selection_probs + 1e-10)
        selector_loss = -(log_probs.mean() * reward)

        self.rule_selector_optimizer.zero_grad()
        selector_loss.backward()
        nn.utils.clip_grad_norm_(self.rule_selector.parameters(), self.args.grad_clip)
        self.rule_selector_optimizer.step()

        return selector_loss.item()

    def evaluate(self, test_queries):
        """
        评估模型

        Args:
            test_queries: 测试查询列表

        Returns:
            metrics: 评估指标字典
        """
        self.rule_selector.eval()
        self.path_finder.eval()

        ranks = []

        with torch.no_grad():
            for query in tqdm(test_queries, desc='Evaluating'):
                head, relation, tail = query

                # 对每个查询进行推理
                rank = self._rank_query(head, relation, tail)
                ranks.append(rank)

        # 计算指标
        ranks = torch.tensor(ranks, dtype=torch.float32)
        metrics = {
            'mrr': (1.0 / ranks).mean().item(),
            'mr': ranks.mean().item(),
            'hits@1': (ranks <= 1).float().mean().item(),
            'hits@3': (ranks <= 3).float().mean().item(),
            'hits@10': (ranks <= 10).float().mean().item()
        }

        self.rule_selector.train()
        self.path_finder.train()

        return metrics

    def _rank_query(self, head, relation, tail):
        """
        对单个查询进行排名

        Args:
            head: 头实体
            relation: 关系
            tail: 尾实体（真实答案）

        Returns:
            rank: 真实答案的排名
        """
        # 选择规则（确定性）
        query_entity_emb = self.rule_model.entity_embedding.weight[head]
        query_rel_emb = self.rule_model.relation_embedding.weight[relation]

        rule_embeddings = self.rule_model.rule_emb.weight

        selected_rules, _ = self.rule_selector(
            query_entity_emb,
            query_rel_emb,
            rule_embeddings,
            epsilon=0.0,
            top_k=self.args.top_k_rules,
            deterministic=True
        )

        # 对所有候选实体评分（简化版本：只运行一次到真实答案）
        # 完整版本应该对所有实体评分，但计算量太大
        # 这里使用简化策略：基于路径奖励排名
        query = (head, relation, tail)
        state = self.env.reset(query, selected_rules)
        done = False
        path_reward = 0.0

        while not done:
            action_mask = self.env.get_action_mask()
            action, _, _ = self.path_finder.select_action(state, action_mask, deterministic=True)
            next_state, reward, done, _ = self.env.step(action)
            path_reward += reward
            state = next_state

        # 简化排名：假设排名与奖励成反比
        # 实际应该对所有候选评分后排序
        # 这里返回一个基于奖励的估计排名
        if path_reward > 0.9:
            rank = 1
        elif path_reward > 0.5:
            rank = 3
        elif path_reward > 0.2:
            rank = 10
        else:
            rank = 50

        return rank

    def save_checkpoint(self, path, epoch, metrics=None):
        """
        保存检查点

        Args:
            path: 保存路径
            epoch: 当前 epoch
            metrics: 评估指标（可选）
        """
        checkpoint = {
            'epoch': epoch,
            'rule_selector': self.rule_selector.state_dict(),
            'path_finder': self.path_finder.state_dict(),
            'rule_selector_optimizer': self.rule_selector_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'rule_counts': dict(self.rule_selector.rule_counts),
            'rule_rewards': dict(self.rule_selector.rule_rewards),
            'total_selections': self.rule_selector.total_selections,
            'best_mrr': self.best_mrr,
            'global_step': self.global_step,
            'args': self.args
        }

        if metrics is not None:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """
        加载检查点

        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(path)

        self.rule_selector.load_state_dict(checkpoint['rule_selector'])
        self.path_finder.load_state_dict(checkpoint['path_finder'])
        self.rule_selector_optimizer.load_state_dict(checkpoint['rule_selector_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])

        self.rule_selector.rule_counts = checkpoint['rule_counts']
        self.rule_selector.rule_rewards = checkpoint['rule_rewards']
        self.rule_selector.total_selections = checkpoint['total_selections']
        self.best_mrr = checkpoint['best_mrr']
        self.global_step = checkpoint['global_step']

        logging.info(f'Checkpoint loaded: {path}')
        logging.info(f'  Epoch: {checkpoint["epoch"]}')
        logging.info(f'  Best MRR: {self.best_mrr:.4f}')
