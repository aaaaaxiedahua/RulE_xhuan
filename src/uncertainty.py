import torch
import argparse
import logging
from data import KnowledgeGraph, RuleDataset


def compute_support_counts(data_path, rule_file):
    """
    计算每条规则的支持路径数

    使用精确的 grounding 方法计算实际支持路径数

    Args:
        data_path: 数据集路径
        rule_file: 规则文件路径

    Returns:
        support_counts: [num_rules] tensor，每条规则的支持路径数
    """
    logging.info(f'Loading knowledge graph from {data_path}')
    graph = KnowledgeGraph(data_path)

    logging.info(f'Loading rules from {rule_file}')
    ruleset = RuleDataset(graph.relation_size, rule_file, negative_sample_size=0)

    support_counts = []

    logging.info(f'Computing support counts for {len(ruleset.rules)} rules using exact grounding...')
    logging.info(f'This may take a while (a few minutes to hours depending on dataset size)...')

    for rule_idx, (rule, _) in enumerate(ruleset.rules):
        rule_id, rule_head, rule_body = rule[0], rule[1], rule[2:]

        # ===== 精确方法：使用 grounding 计算实际路径数 =====
        count = 0.0

        # 遍历所有实体作为头实体
        for h in range(graph.entity_size):
            # 使用 grounding 计算从 h 出发沿着规则体能到达的实体
            h_tensor = torch.tensor([h], dtype=torch.long)

            # 调用知识图谱的 grounding 方法
            # grounding(h, r_head, r_body, edges_to_remove=None)
            # 返回: [1, num_entities] 的张量，表示每个实体被到达的路径数
            paths = graph.grounding(h_tensor, rule_head, rule_body, edges_to_remove=None)

            # 累加所有路径数
            count += paths.sum().item()

        # 至少为1，避免0计数（对于从未被支持的规则）
        count = max(count, 1.0)
        support_counts.append(count)

        if (rule_idx + 1) % 10 == 0:
            logging.info(f'  Processed {rule_idx + 1}/{len(ruleset.rules)} rules (current: {count:.0f} paths)')

    # ===== 近似方法（已注释）：使用关系频率几何平均 =====
    # for rule_idx, (rule, _) in enumerate(ruleset.rules):
    #     rule_id, rule_head, rule_body = rule[0], rule[1], rule[2:]
    #
    #     # 计算规则体中每个关系的频率
    #     count = 1.0
    #     for rel in rule_body:
    #         # 处理反向关系：rel可能是 r 或 r+relation_size
    #         rel_id = rel % graph.relation_size
    #
    #         # 统计该关系在训练集中出现的次数
    #         freq = 0
    #         # graph.train_facts 格式: [(h, r, t), ...]
    #         for h, r, t in graph.train_facts:
    #             if r == rel:  # 匹配正向或反向关系
    #                 freq += 1
    #
    #         # 至少为1，避免0计数
    #         freq = max(freq, 1)
    #         count *= freq
    #
    #     # 使用几何平均
    #     if len(rule_body) > 0:
    #         count = count ** (1.0 / len(rule_body))
    #     else:
    #         count = 1.0
    #
    #     support_counts.append(count)
    #
    #     if (rule_idx + 1) % 100 == 0:
    #         logging.info(f'  Processed {rule_idx + 1}/{len(ruleset.rules)} rules')

    support_counts_tensor = torch.tensor(support_counts, dtype=torch.float32)

    logging.info(f'Support counts computed:')
    logging.info(f'  Min: {support_counts_tensor.min().item():.2f}')
    logging.info(f'  Max: {support_counts_tensor.max().item():.2f}')
    logging.info(f'  Mean: {support_counts_tensor.mean().item():.2f}')
    logging.info(f'  Std: {support_counts_tensor.std().item():.2f}')

    return support_counts_tensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute support counts for rules')
    parser.add_argument('--data_path', required=True, help='Path to dataset directory')
    parser.add_argument('--rule_file', required=True, help='Path to mined rules file')
    parser.add_argument('--output', default='support_counts.pt', help='Output file name')
    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 计算support counts
    counts = compute_support_counts(args.data_path, args.rule_file)

    # 保存到数据集目录
    import os
    output_path = os.path.join(args.data_path, args.output)
    torch.save(counts, output_path)

    logging.info(f'Support counts saved to {output_path}')
    logging.info(f'Total rules: {len(counts)}')
