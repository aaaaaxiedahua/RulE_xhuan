"""
Rule-GNN 训练脚本

Rule-GNN 用 GNN 消息传递替换 RulE 的 Grounding 阶段

训练流程:
1. RulE 预训练 (RotatE + 规则嵌入) - 可选
2. 导出嵌入 (entity, relation, rule)
3. Rule-GNN 训练 (GNN 替代 Grounding)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import torch
import numpy as np
import random

from data import KnowledgeGraph, RuleDataset, TrainDataset, ValidDataset, TestDataset
from model import RulE
from trainer import PreTrainer
from rule_gnn_model import RuleGNN
from rule_gnn_trainer import RuleGNNTrainer
from utils import load_config, save_config, set_logger, set_seed


def parse_args(args=None):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Rule-GNN Training',
        usage='main_rule_gnn.py [<args>] [-h | --help]'
    )

    parser.add_argument("--local_rank", type=int, default=0)

    # data path
    parser.add_argument('--data_path', default="../data/umls", type=str, help='dataset path')
    parser.add_argument('--rule_file', default="../data/umls/mined_rules.txt", type=str)
    parser.add_argument('--dataset', default='umls', type=str, help='dataset name')

    # device
    parser.add_argument('--cuda', action='store_true', default=False, help='use GPU')
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('--seed', default=800, type=int, help='seed')

    # pre train process (KGE + rulE)
    parser.add_argument('-b', '--batch_size', default=256, type=int)
    parser.add_argument('-n', '--negative_sample_size', default=512, type=int)
    parser.add_argument('--rule_batch_size', default=256, type=int, help='rule batch size')
    parser.add_argument('--rule_negative_size', default=128, type=int)

    parser.add_argument('-d', '--hidden_dim', default=2000, type=int)
    parser.add_argument('-g_f', '--gamma_fact', default=6, type=float, help='the triplet margin')
    parser.add_argument('-g_r', '--gamma_rule', default=8, type=float, help='the rule margin')
    parser.add_argument('--disable_adv', action='store_true', default=False, help='disable the adversarial negative sampling')
    parser.add_argument('--negative_adversarial_sampling', default=True, type=bool)
    parser.add_argument('-a', '--adversarial_temperature', default=0.25, type=float)

    parser.add_argument('--uni_weight', action='store_true', default=False,
                       help='Otherwise use subsampling weighting like in word2vec')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    parser.add_argument('--save_checkpoint_steps', default=10, type=int)
    parser.add_argument('--valid_steps', default=1000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--weight_rule', type=float, default=1)
    parser.add_argument('-reg', '--regularization', default=0, type=float)
    parser.add_argument('--max_steps', default=30000, type=int)
    parser.add_argument('--p_norm', default=2, type=int)

    # save path
    parser.add_argument('-init', '--init', default=None, type=str, help='Path to config file (JSON)')
    parser.add_argument('-save', '--save_path', default=None, type=str)

    # grounding training process (RulE uses this)
    parser.add_argument('--mlp_rule_dim', default=100, type=int)
    parser.add_argument('--alpha', default=2.0, type=float, help='weight the KGE score')

    # Rule-GNN specific parameters
    parser.add_argument('--smoothing', default=0.2, type=float)
    parser.add_argument('--batch_per_epoch', default=1000000, type=int)
    parser.add_argument('--print_every', default=10, type=int)
    parser.add_argument('--g_batch_size', default=16, type=int)
    parser.add_argument('--g_lr', default=0.0001, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--rule_gnn_num_iters', default=50, type=int)
    parser.add_argument('--rule_gnn_valid_every', default=5, type=int)
    parser.add_argument('--skip_pretrain', action='store_true', help='Skip RulE pretraining (load from checkpoint)')

    return parser.parse_args(args)


def main():
    """主函数"""
    args = parse_args()

    # 保存命令行参数（在加载配置前）
    skip_pretrain = args.skip_pretrain if hasattr(args, 'skip_pretrain') else False

    # read the given config
    if args.init:
        args = load_config(args.init)
        args = args[0]

    # 恢复命令行参数
    args.skip_pretrain = skip_pretrain

    # 设置随机种子
    set_seed(args.seed)

    # 设置保存路径
    if not hasattr(args, 'save_path') or args.save_path is None:
        args.save_path = os.path.join('../outputs', datetime.now().strftime('%Y%m-%d%H-%M%S'))

    os.makedirs(args.save_path, exist_ok=True)

    # 保存配置
    save_config(args)

    # 设置日志
    set_logger(args.save_path)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Rule-GNN Training")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset if hasattr(args, 'dataset') else args.data_path}")
    logger.info(f"Save Path: {args.save_path}")

    # 设置设备
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # ======================================================================
    # 阶段 1: 加载数据
    # ======================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Phase 1: Loading Data")
    logger.info("=" * 80)

    graph = KnowledgeGraph(args.data_path)
    logger.info(f"Entities: {graph.entity_size}")
    logger.info(f"Relations: {graph.relation_size}")
    logger.info(f"Train triples: {len(graph.train_facts)}")
    logger.info(f"Valid triples: {len(graph.valid_facts)}")
    logger.info(f"Test triples: {len(graph.test_facts)}")

    # 加载规则
    ruleset = RuleDataset(graph.relation_size, args.rule_file, args.rule_negative_size)
    rules = [rule[0] for rule in ruleset.rules]
    logger.info(f"Number of rules: {len(ruleset)}")

    # 创建数据集
    train_set = TrainDataset(graph, args.g_batch_size)
    valid_set = ValidDataset(graph, args.g_batch_size)
    test_set = TestDataset(graph, args.g_batch_size)

    # ======================================================================
    # 阶段 2: RulE 预训练 (可选)
    # ======================================================================
    rule_checkpoint_path = os.path.join(args.save_path, 'rule_checkpoint')

    if not args.skip_pretrain:
        logger.info("\n" + "=" * 80)
        logger.info("Phase 2: RulE Pre-training (RotatE + Rule Embeddings)")
        logger.info("=" * 80)

        # 创建 RulE 模型
        rule_model = RulE(
            graph=graph,
            p_norm=args.p_norm,
            mlp_rule_dim=args.mlp_rule_dim,
            gamma_fact=args.gamma_fact,
            gamma_rule=args.gamma_rule,
            hidden_dim=args.hidden_dim,
            device=device,
            dataset=args.data_path
        )
        rule_model.set_rules(rules)

        # 创建预训练器
        pre_trainer = PreTrainer(
            graph=graph,
            model=rule_model,
            valid_set=valid_set,
            test_set=test_set,
            ruleset=ruleset,
            expectation=True,
            device=device,
            num_worker=args.cpu_num
        )

        # 执行预训练
        logger.info("Starting RulE pre-training...")
        pre_trainer.train(args)

        logger.info('RulE pre-training completed!')

        # 加载最佳 checkpoint 并评估
        checkpoint = torch.load(rule_checkpoint_path)
        rule_model.load_state_dict(checkpoint['model'])

        logger.info('Evaluating RulE pre-training results...')
        valid_mrr = pre_trainer.evaluate('valid', expectation=True)
        test_mrr = pre_trainer.evaluate('test', expectation=True)

    else:
        logger.info("\n" + "=" * 80)
        logger.info("Phase 2: Skipping RulE Pre-training (loading from checkpoint)")
        logger.info("=" * 80)

        # 创建 RulE 模型并加载 checkpoint
        rule_model = RulE(
            graph=graph,
            p_norm=args.p_norm,
            mlp_rule_dim=args.mlp_rule_dim,
            gamma_fact=args.gamma_fact,
            gamma_rule=args.gamma_rule,
            hidden_dim=args.hidden_dim,
            device=device,
            dataset=args.data_path
        )
        rule_model.set_rules(rules)

        if os.path.exists(rule_checkpoint_path):
            checkpoint = torch.load(rule_checkpoint_path)
            rule_model.load_state_dict(checkpoint['model'])
            logger.info(f"Loaded RulE checkpoint from {rule_checkpoint_path}")
        else:
            logger.error(f"RulE checkpoint not found: {rule_checkpoint_path}")
            logger.error("Please run without --skip_pretrain first")
            return

    # ======================================================================
    # 阶段 3: 导出嵌入
    # ======================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Phase 3: Exporting Embeddings for Rule-GNN")
    logger.info("=" * 80)

    embeddings_dict = rule_model.export_embeddings()
    logger.info(f"Exported entity embeddings: {embeddings_dict['entity_embedding'].shape}")
    logger.info(f"Exported relation embeddings: {embeddings_dict['relation_embedding'].shape}")
    logger.info(f"Exported rule embeddings: {embeddings_dict['rule_emb'].shape}")

    # 获取规则体最大长度（用于设置 GNN 层数）
    max_body_len = rule_model.max_length
    logger.info(f"Max rule length: {max_body_len}")

    # ======================================================================
    # 阶段 4: Rule-GNN 训练 (替代 Grounding)
    # ======================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Phase 4: Rule-GNN Training (replaces Grounding)")
    logger.info("=" * 80)

    # GNN 层数 = 规则最大长度
    num_layers = max_body_len
    logger.info(f"GNN layers (= max rule length): {num_layers}")

    # 创建 Rule-GNN 模型
    rule_gnn_model = RuleGNN(
        num_entities=graph.entity_size,
        num_relations=graph.relation_size * 2,  # 包括逆关系
        num_rules=len(ruleset),
        hidden_dim=args.hidden_dim,
        num_layers=num_layers,
        dropout=args.dropout if hasattr(args, 'dropout') else 0.1
    )

    logger.info(f"Rule-GNN parameters: {sum(p.numel() for p in rule_gnn_model.parameters()):,}")

    # 创建 Rule-GNN 训练器
    rule_gnn_trainer = RuleGNNTrainer(
        model=rule_gnn_model,
        graph=graph,
        train_dataset=train_set,
        valid_dataset=valid_set,
        test_dataset=test_set,
        device=device,
        logger=logger
    )

    # 加载预训练嵌入
    logger.info("Loading pretrained embeddings into Rule-GNN...")
    rule_gnn_model.load_pretrained_embeddings(embeddings_dict)

    # 训练 Rule-GNN
    logger.info("Starting Rule-GNN training...")
    test_metrics = rule_gnn_trainer.train(args)

    # ======================================================================
    # 阶段 5: 保存结果
    # ======================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Final Results")
    logger.info("=" * 80)

    results = {
        'dataset': args.dataset if hasattr(args, 'dataset') else args.data_path,
        'hidden_dim': args.hidden_dim,
        'num_layers': num_layers,
        'test_metrics': test_metrics,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    results_file = os.path.join(args.save_path, 'rule_gnn_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")

    logger.info("\n" + "=" * 80)
    logger.info("Training Completed!")
    logger.info("=" * 80)
    logger.info(f"RulE checkpoint: {rule_checkpoint_path}")
    logger.info(f"Rule-GNN checkpoint: {os.path.join(args.save_path, 'rule_gnn_best.pt')}")
    logger.info(f"Results: {results_file}")


if __name__ == '__main__':
    main()
