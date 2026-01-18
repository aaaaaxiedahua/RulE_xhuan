"""
Box-RulE: Main Training Script
主训练脚本 - 两阶段训练流程
"""

import argparse
import logging
import torch
import os
import sys

from graph import Graph
from data import KGETrainDataset, RuleDataset, TestDataset
from box_model import BoxRulE
from box_trainer import WarmupTrainer, JointTrainer


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Box-RulE Training')

    # ===== 数据参数 =====
    parser.add_argument('--data_path', type=str, default='../data/FB15k-237',
                        help='数据集路径')
    parser.add_argument('--rule_path', type=str, default='../data/FB15k-237/rules',
                        help='规则文件路径')

    # ===== 模型参数 =====
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='嵌入维度')
    parser.add_argument('--init_width', type=float, default=0.5,
                        help='盒子初始宽度')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='数值稳定性参数')

    # ===== 训练参数 =====
    parser.add_argument('--batch_size', type=int, default=512,
                        help='KGE批次大小')
    parser.add_argument('--rule_batch_size', type=int, default=128,
                        help='规则批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--negative_sample_size', type=int, default=128,
                        help='负样本数量')

    # ===== 阶段参数 =====
    parser.add_argument('--warmup_steps', type=int, default=10000,
                        help='预热阶段步数')
    parser.add_argument('--joint_steps', type=int, default=20000,
                        help='联合训练步数')

    # ===== 损失权重 =====
    parser.add_argument('--gamma_fact', type=float, default=12.0,
                        help='KGE margin')
    parser.add_argument('--gamma_rule', type=float, default=12.0,
                        help='Rule margin')
    parser.add_argument('--lambda_vol', type=float, default=0.001,
                        help='体积正则化权重')
    parser.add_argument('--weight_rule', type=float, default=1.0,
                        help='规则损失权重')
    parser.add_argument('--adversarial_temperature', type=float, default=1.0,
                        help='对抗温度参数')

    # ===== 其他参数 =====
    parser.add_argument('--cpu_num', type=int, default=4,
                        help='CPU工作线程数')
    parser.add_argument('--save_path', type=str, default='../checkpoints',
                        help='模型保存路径')
    parser.add_argument('--log_steps', type=int, default=100,
                        help='日志输出间隔')
    parser.add_argument('--valid_steps', type=int, default=1000,
                        help='验证间隔')
    parser.add_argument('--cuda', action='store_true',
                        help='使用GPU')

    args = parser.parse_args()

    # 设置设备
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    return args


def setup_logging():
    """
    配置日志
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_data(args):
    """
    加载数据
    """
    logging.info('Loading data...')

    # 加载知识图谱
    graph = Graph(args.data_path)

    # 加载训练集
    train_dataset = KGETrainDataset(
        triples=graph.train_facts,
        nentity=graph.entity_size,
        nrelation=graph.relation_size,
        negative_sample_size=args.negative_sample_size,
        mode='head-batch'
    )

    # 加载规则集
    rule_dataset = RuleDataset(
        rule_path=args.rule_path,
        nrelation=graph.relation_size,
        negative_sample_size=args.negative_sample_size
    )

    # 加载验证集和测试集
    valid_dataset = TestDataset(graph, 'valid')
    test_dataset = TestDataset(graph, 'test')

    logging.info(f'Entities: {graph.entity_size}')
    logging.info(f'Relations: {graph.relation_size}')
    logging.info(f'Train triples: {len(graph.train_facts)}')
    logging.info(f'Rules: {len(rule_dataset)}')

    return graph, train_dataset, rule_dataset, valid_dataset, test_dataset


def main():
    """
    主函数 - 两阶段训练流程
    """
    # 解析参数
    args = parse_args()
    setup_logging()

    logging.info('='*50)
    logging.info('Box-RulE Training')
    logging.info('='*50)

    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)

    # 加载数据
    graph, train_dataset, rule_dataset, valid_dataset, test_dataset = load_data(args)

    # 初始化模型
    logging.info('Initializing Box-RulE model...')
    model = BoxRulE(graph, args)

    # 加载规则
    logging.info('Loading rules into model...')
    rules = rule_dataset.get_all_rules()
    model.set_rules(rules)

    # ===== 阶段1: Warmup训练 (几何预热) =====
    logging.info('')
    logging.info('='*50)
    logging.info('Stage 1: Warmup Training (Geometric Initialization)')
    logging.info('='*50)

    warmup_trainer = WarmupTrainer(
        model=model,
        graph=graph,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        args=args
    )

    warmup_trainer.train(max_steps=args.warmup_steps)

    # 加载最优模型
    logging.info('')
    logging.info('Loading best Warmup model...')
    best_warmup_path = os.path.join(args.save_path, 'warmup_best_model.pt')
    if os.path.exists(best_warmup_path):
        checkpoint = torch.load(best_warmup_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f'Loaded best model from step {checkpoint["step"]} with MRR {checkpoint["best_mrr"]:.4f}')
    else:
        logging.info('No best model found, using final model')

    # 评估Warmup阶段效果（使用最优模型）
    logging.info('')
    logging.info('Evaluating Warmup Stage on validation set...')
    warmup_valid_mrr = warmup_trainer.evaluate(split='valid')
    logging.info(f'Warmup Valid MRR: {warmup_valid_mrr:.4f}')

    # 在测试集上测试
    logging.info('')
    logging.info('Evaluating Warmup Stage on test set...')
    warmup_test_mrr = warmup_trainer.evaluate(split='test')
    logging.info(f'Warmup Test MRR: {warmup_test_mrr:.4f}')

    # 保存Warmup模型
    warmup_checkpoint = os.path.join(args.save_path, 'warmup_final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args,
        'warmup_valid_mrr': warmup_valid_mrr,
        'warmup_test_mrr': warmup_test_mrr
    }, warmup_checkpoint)
    logging.info(f'Warmup final model saved to {warmup_checkpoint}')

    # ===== 阶段2: Joint训练 (KGE + 规则联合训练) =====
    logging.info('')
    logging.info('='*50)
    logging.info('Stage 2: Joint Training (KGE + Rules)')
    logging.info('='*50)

    joint_trainer = JointTrainer(
        model=model,
        graph=graph,
        train_dataset=train_dataset,
        rule_dataset=rule_dataset,
        valid_dataset=valid_dataset,
        args=args
    )

    joint_trainer.train(max_steps=args.joint_steps)

    # 加载最优模型
    logging.info('')
    logging.info('Loading best Joint model...')
    best_joint_path = os.path.join(args.save_path, 'joint_best_model.pt')
    if os.path.exists(best_joint_path):
        checkpoint = torch.load(best_joint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f'Loaded best model from step {checkpoint["step"]} with MRR {checkpoint["best_mrr"]:.4f}')
    else:
        logging.info('No best model found, using final model')

    # 评估Joint阶段效果（使用最优模型）
    logging.info('')
    logging.info('Evaluating Joint Stage on validation set...')
    joint_valid_mrr = joint_trainer.evaluate(split='valid')
    logging.info(f'Joint Valid MRR: {joint_valid_mrr:.4f}')

    # 在测试集上测试
    logging.info('')
    logging.info('Evaluating Joint Stage on test set...')
    joint_test_mrr = joint_trainer.evaluate(split='test')
    logging.info(f'Joint Test MRR: {joint_test_mrr:.4f}')

    # 保存最终模型
    final_checkpoint = os.path.join(args.save_path, 'joint_final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args,
        'joint_valid_mrr': joint_valid_mrr,
        'joint_test_mrr': joint_test_mrr
    }, final_checkpoint)
    logging.info(f'Joint final model saved to {final_checkpoint}')

    # ===== 训练完成 =====
    logging.info('')
    logging.info('='*50)
    logging.info('Training Completed!')
    logging.info('='*50)
    logging.info(f'Warmup steps: {args.warmup_steps}')
    logging.info(f'Joint steps: {args.joint_steps}')
    logging.info(f'Total steps: {args.warmup_steps + args.joint_steps}')
    logging.info(f'Model saved to: {args.save_path}')


if __name__ == '__main__':
    main()
