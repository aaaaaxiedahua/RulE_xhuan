"""
RulE-RL 主训练脚本

该脚本实现了RulE-RL框架，通过强化学习增强RulE：
1. 高层规则选择器Agent (UCB + ε-greedy)
2. 低层路径搜索器Agent (REINFORCE with Baseline)

训练流程：
- Phase 1: 加载预训练RulE模型（冻结参数）
- Phase 2: 初始化RL组件
- Phase 3-4: Episode训练循环与RL更新
- Phase 5: 评估与保存
"""

import logging
import os
import argparse
import torch
import numpy as np
from datetime import datetime

# 导入原始RulE组件
from data import KnowledgeGraph, TrainDataset, ValidDataset, TestDataset, RuleDataset
from model import RulE
from utils import load_config, save_config, set_logger, set_seed
from trainer import PreTrainer  # 导入预训练器

# 导入RL组件
from rl.state_encoder import StateEncoder
from rl.rule_selector import RuleSelectorAgent
from rl.path_finder import PathFinderAgent
from rl.kg_env import KGReasoningEnv
from rl.reward_calculator import RewardCalculator
from rl.trainer_rl import RulERLTrainer


def parse_args(args=None):
    """解析RulE-RL的命令行参数"""
    parser = argparse.ArgumentParser(
        description='RulE-RL: 基于强化学习的规则增强知识图谱推理',
        usage='main_rl.py [<args>] [-h | --help]'
    )

    # 配置文件
    parser.add_argument('-init', '--init_checkpoint_config',
                       default="../config/umls_rule_rl_config.json",
                       type=str,
                       help='RulE-RL配置JSON文件路径')

    # 数据路径
    parser.add_argument('--data_path', type=str, help='数据集路径')
    parser.add_argument('--rule_file', type=str, help='挖掘的规则文件路径')
    parser.add_argument('--pretrain_checkpoint', type=str,
                       help='预训练RulE检查点路径（将被冻结）')
    parser.add_argument('--save_path', type=str,
                       help='RulE-RL检查点保存路径（仅RL组件）')

    # 设备配置
    parser.add_argument('--cuda', action='store_true', default=False, help='使用GPU')
    parser.add_argument('--cpu_num', default=10, type=int, help='CPU线程数')
    parser.add_argument('--seed', default=800, type=int, help='随机种子')

    # 预训练嵌入维度（冻结）
    parser.add_argument('--hidden_dim', default=2000, type=int,
                       help='关系嵌入维度（来自预训练）')
    parser.add_argument('--mlp_rule_dim', default=100, type=int,
                       help='规则嵌入维度（来自预训练）')
    parser.add_argument('--gamma_fact', default=6, type=float,
                       help='三元组margin（来自预训练）')
    parser.add_argument('--gamma_rule', default=8, type=float,
                       help='规则margin（来自预训练）')
    parser.add_argument('--p_norm', default=2, type=int,
                       help='距离范数类型')

    # RL模型架构
    parser.add_argument('--state_dim', default=128, type=int,
                       help='StateEncoder输出维度')
    parser.add_argument('--history_dim', default=128, type=int,
                       help='GRU编码路径历史的隐藏层维度')
    parser.add_argument('--policy_hidden_dim', default=256, type=int,
                       help='策略网络隐藏层维度')
    parser.add_argument('--value_hidden_dim', default=256, type=int,
                       help='价值网络隐藏层维度')

    # RL超参数
    parser.add_argument('--top_k_rules', default=5, type=int,
                       help='每次查询选择的规则数量')
    parser.add_argument('--max_steps', default=5, type=int,
                       help='每个episode的最大步数')
    parser.add_argument('--gamma', default=0.99, type=float,
                       help='折扣因子')
    parser.add_argument('--epsilon_start', default=0.5, type=float,
                       help='初始探索率')
    parser.add_argument('--epsilon_end', default=0.05, type=float,
                       help='最终探索率')
    parser.add_argument('--ucb_c', default=1.0, type=float,
                       help='UCB探索系数')

    # 奖励参数
    parser.add_argument('--alpha', default=0.1, type=float,
                       help='中间奖励权重')
    parser.add_argument('--beta', default=0.05, type=float,
                       help='惩罚权重（已弃用）')

    # 优化参数
    parser.add_argument('--lr_policy', default=0.001, type=float,
                       help='策略网络学习率')
    parser.add_argument('--lr_value', default=0.001, type=float,
                       help='价值网络学习率')
    parser.add_argument('--lr_selector', default=0.0001, type=float,
                       help='规则选择器学习率')
    parser.add_argument('--grad_clip', default=1.0, type=float,
                       help='梯度裁剪最大范数')

    # 训练控制
    parser.add_argument('--num_epochs', default=100, type=int,
                       help='训练轮数')
    parser.add_argument('--log_interval', default=100, type=int,
                       help='每N个episode打印日志')
    parser.add_argument('--eval_interval', default=5, type=int,
                       help='每N个epoch评估一次')
    parser.add_argument('--save_interval', default=10, type=int,
                       help='每N个epoch保存检查点')

    # 预训练参数（如果需要自动预训练）
    parser.add_argument('--auto_pretrain', action='store_true', default=True,
                       help='如果预训练检查点不存在，自动进行预训练')
    parser.add_argument('--learning_rate', default=0.00005, type=float,
                       help='预训练学习率')
    parser.add_argument('--pretrain_max_steps', default=30000, type=int,
                       help='预训练最大步数')
    parser.add_argument('--warm_up_steps', default=15000, type=int,
                       help='预训练学习率预热步数')
    parser.add_argument('--batch_size', default=256, type=int,
                       help='预训练三元组批次大小')
    parser.add_argument('--negative_sample_size', default=512, type=int,
                       help='预训练负样本数')
    parser.add_argument('--rule_batch_size', default=128, type=int,
                       help='预训练规则批次大小')
    parser.add_argument('--rule_negative_size', default=64, type=int,
                       help='预训练规则负样本数')
    parser.add_argument('--weight_rule', default=1.0, type=float,
                       help='规则损失权重')
    parser.add_argument('--adversarial_temperature', default=0.5, type=float,
                       help='对抗采样温度')
    parser.add_argument('--test_batch_size', default=16, type=int,
                       help='测试批次大小')

    return parser.parse_args(args)


def main():
    """RulE-RL主训练函数"""

    # 解析参数
    args = parse_args()

    # 从JSON加载配置（如果提供）
    if args.init_checkpoint_config:
        logging.info(f'从 {args.init_checkpoint_config} 加载配置')
        args = load_config(args.init_checkpoint_config)
        args = args[0]

    # 设置保存目录
    if args.save_path is None:
        args.save_path = os.path.join('../outputs',
                                      f'rule_rl_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 保存配置
    save_config(args)

    # 设置日志和随机种子
    set_logger(args.save_path)
    set_seed(args.seed)

    logging.info('='*80)
    logging.info('RulE-RL 训练')
    logging.info('='*80)

    # ========================================================================
    # Phase 1: 加载预训练RulE模型
    # ========================================================================
    logging.info('\n' + '='*80)
    logging.info('Phase 1: 加载预训练RulE模型')
    logging.info('='*80)

    # 加载知识图谱
    logging.info(f'从 {args.data_path} 加载知识图谱')
    graph = KnowledgeGraph(args.data_path)

    logging.info(f'  - 实体数: {graph.entity_size}')
    logging.info(f'  - 关系数: {graph.relation_size}')
    logging.info(f'  - 训练三元组: {len(graph.train_facts)}')
    logging.info(f'  - 验证三元组: {len(graph.valid_facts)}')
    logging.info(f'  - 测试三元组: {len(graph.test_facts)}')

    # 加载数据集
    logging.info('初始化数据集...')
    train_set = TrainDataset(graph, batch_size=16)
    valid_set = ValidDataset(graph, batch_size=16)
    test_set = TestDataset(graph, batch_size=16)
    ruleset = RuleDataset(graph.relation_size, args.rule_file, negative_sample_size=64)

    rules = [rule[0] for rule in ruleset.rules]
    logging.info(f'  - 从 {args.rule_file} 加载了 {len(rules)} 条规则')

    # 设置设备
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f'使用GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        logging.info('使用CPU')

    # 初始化RulE模型
    logging.info('初始化RulE模型...')
    rule_model = RulE(
        graph=graph,
        p_norm=args.p_norm,
        mlp_rule_dim=args.mlp_rule_dim,
        gamma_fact=args.gamma_fact,
        gamma_rule=args.gamma_rule,
        hidden_dim=args.hidden_dim,
        device=device,
        data_path=args.data_path
    )
    rule_model.set_rules(rules)

    # 检查预训练检查点是否存在
    logging.info(f'检查预训练检查点: {args.pretrain_checkpoint}')

    if not os.path.exists(args.pretrain_checkpoint):
        if args.auto_pretrain:
            # 自动进行预训练
            logging.info('='*80)
            logging.info('预训练检查点不存在，开始自动预训练...')
            logging.info('='*80)

            # 创建预训练检查点目录
            pretrain_dir = os.path.dirname(args.pretrain_checkpoint)
            if not os.path.exists(pretrain_dir):
                os.makedirs(pretrain_dir)
                logging.info(f'创建预训练检查点目录: {pretrain_dir}')

            # 初始化预训练器
            from data import KGETrainDataset
            pre_train_set = KGETrainDataset(
                graph,
                batch_size=args.batch_size,
                negative_sample_size=args.negative_sample_size
            )

            pre_trainer = PreTrainer(
                model=rule_model,
                graph=graph,
                train_set=pre_train_set,
                valid_set=valid_set,
                test_set=test_set,
                ruleset=ruleset,
                device=device
            )

            logging.info(f'开始预训练 (pretrain_max_steps={args.pretrain_max_steps})...')

            # 临时保存原始 max_steps，使用预训练的 max_steps
            original_max_steps = args.max_steps
            args.max_steps = args.pretrain_max_steps

            pre_trainer.train(args)

            # 恢复 RL 的 max_steps
            args.max_steps = original_max_steps

            logging.info('='*80)
            logging.info(f'预训练完成，检查点已保存到: {args.pretrain_checkpoint}')
            logging.info('='*80)
        else:
            logging.error(f'预训练检查点未找到: {args.pretrain_checkpoint}')
            logging.error('请设置 --auto_pretrain 或先使用 main.py 训练RulE模型')
            raise FileNotFoundError(f'检查点未找到: {args.pretrain_checkpoint}')

    # 加载预训练检查点
    logging.info(f'从 {args.pretrain_checkpoint} 加载预训练检查点')
    checkpoint = torch.load(args.pretrain_checkpoint, map_location=device)
    rule_model.load_state_dict(checkpoint['model'])
    rule_model.to(device)

    logging.info('预训练模型加载成功')
    logging.info(f'  - 实体嵌入形状: {rule_model.entity_embedding.weight.shape}')
    logging.info(f'  - 关系嵌入形状: {rule_model.relation_embedding.weight.shape}')
    logging.info(f'  - 规则嵌入形状: {rule_model.rule_emb.shape}')

    # 冻结预训练参数
    logging.info('冻结预训练参数...')
    for param in rule_model.entity_embedding.parameters():
        param.requires_grad = False
    for param in rule_model.relation_embedding.parameters():
        param.requires_grad = False
    rule_model.rule_emb.requires_grad = False

    # 统计冻结参数和可训练参数
    frozen_params = sum(p.numel() for p in rule_model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in rule_model.parameters() if p.requires_grad)
    logging.info(f'  - 冻结参数: {frozen_params:,}')
    logging.info(f'  - 可训练参数: {trainable_params:,}')

    # ========================================================================
    # Phase 2: 初始化RL组件
    # ========================================================================
    logging.info('\n' + '='*80)
    logging.info('Phase 2: 初始化RL组件')
    logging.info('='*80)

    # 从预训练模型获取维度
    entity_dim = rule_model.entity_embedding.weight.shape[1]  # 应为 hidden_dim * 2
    rel_dim = rule_model.relation_embedding.weight.shape[1]   # 应为 hidden_dim
    rule_dim = rule_model.rule_emb.shape[1]                   # 应为 mlp_rule_dim
    num_entities = graph.entity_size
    num_relations = graph.relation_size
    num_rules = len(rules)

    logging.info(f'维度信息:')
    logging.info(f'  - 实体维度: {entity_dim} (hidden_dim × 2，用于复数嵌入)')
    logging.info(f'  - 关系维度: {rel_dim}')
    logging.info(f'  - 规则维度: {rule_dim}')
    logging.info(f'  - 查询维度: {entity_dim + rel_dim}')
    logging.info(f'  - 实体数量: {num_entities}')
    logging.info(f'  - 关系数量: {num_relations}')
    logging.info(f'  - 规则数量: {num_rules}')

    # 初始化RL组件
    logging.info('\n正在初始化RL组件...')

    # 1. StateEncoder - 状态编码器
    logging.info('  [1/6] 初始化 StateEncoder')
    state_encoder = StateEncoder(
        entity_dim=entity_dim,
        rel_dim=rel_dim,
        rule_dim=rule_dim,
        history_dim=args.history_dim,
        state_dim=args.state_dim
    ).to(device)
    logging.info(f'    ✓ StateEncoder 初始化完成 (state_dim={args.state_dim})')

    # 2. RuleSelectorAgent - 规则选择器
    logging.info('  [2/6] 初始化 RuleSelectorAgent')
    rule_selector = RuleSelectorAgent(
        entity_dim=entity_dim,
        rel_dim=rel_dim,
        rule_dim=rule_dim,
        num_rules=num_rules,
        hidden_dim=args.state_dim,
        ucb_c=args.ucb_c
    ).to(device)
    logging.info(f'    ✓ RuleSelectorAgent 初始化完成 (num_rules={num_rules})')

    # 3. PathFinderAgent - 路径搜索器
    logging.info('  [3/6] 初始化 PathFinderAgent')
    path_finder = PathFinderAgent(
        state_dim=args.state_dim,
        action_dim=num_relations * 2,  # 包括逆关系
        hidden_dim=args.policy_hidden_dim
    ).to(device)
    logging.info(f'    ✓ PathFinderAgent 初始化完成 (action_dim={num_relations * 2})')

    # 4. RewardCalculator - 奖励计算器
    logging.info('  [4/6] 初始化 RewardCalculator')
    reward_calculator = RewardCalculator(
        rule_model=rule_model,
        alpha=args.alpha
    )
    logging.info(f'    ✓ RewardCalculator 初始化完成 (alpha={args.alpha})')

    # 5. KGReasoningEnv - 知识图谱推理环境
    logging.info('  [5/6] 初始化 KGReasoningEnv')
    env = KGReasoningEnv(
        graph=graph,
        rule_model=rule_model,
        state_encoder=state_encoder,
        reward_calculator=reward_calculator,
        max_steps=args.max_steps
    )
    logging.info(f'    ✓ KGReasoningEnv 初始化完成 (max_steps={args.max_steps})')

    # 6. RulERLTrainer - RulE-RL训练器
    logging.info('  [6/6] 初始化 RulERLTrainer')
    trainer = RulERLTrainer(
        rule_model=rule_model,
        rule_selector=rule_selector,
        path_finder=path_finder,
        env=env,
        graph=graph,
        args=args
    )
    logging.info('    ✓ RulERLTrainer 初始化完成')

    logging.info('\n' + '='*80)
    logging.info('所有RL组件初始化成功!')
    logging.info('='*80)

    # 统计可训练参数
    total_params = sum(p.numel() for p in rule_selector.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in path_finder.parameters() if p.requires_grad)
    logging.info(f'\nRL组件可训练参数总数: {total_params:,}')

    # ========================================================================
    # Phase 3-4: 训练循环
    # ========================================================================
    logging.info('\n' + '='*80)
    logging.info('Phase 3-4: RL训练')
    logging.info('='*80)

    # 准备训练数据
    train_queries = [(h, r, t) for h, r, t in graph.train_facts]
    valid_queries = [(h, r, t) for h, r, t in graph.valid_facts]
    test_queries = [(h, r, t) for h, r, t in graph.test_facts]

    # 开始训练
    test_metrics = trainer.train(train_queries, valid_queries, test_queries)

    # ========================================================================
    # Phase 5: 最终评估
    # ========================================================================
    logging.info('\n' + '='*80)
    logging.info('Phase 5: 训练完成')
    logging.info('='*80)
    logging.info(f'最终测试结果:')
    logging.info(f'  MRR: {test_metrics["mrr"]:.4f}')
    logging.info(f'  MR: {test_metrics["mr"]:.2f}')
    logging.info(f'  Hits@1: {test_metrics["hits@1"]:.4f}')
    logging.info(f'  Hits@3: {test_metrics["hits@3"]:.4f}')
    logging.info(f'  Hits@10: {test_metrics["hits@10"]:.4f}')

    logging.info('\n' + '='*80)
    logging.info('RulE-RL训练成功完成!')
    logging.info('='*80)


if __name__ == '__main__':
    main()
