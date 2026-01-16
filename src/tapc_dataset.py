"""
TAPC-RulE: Critic Dataset Module
Critic数据集模块 - 构造路径判别训练数据
"""

import torch
from torch.utils.data import Dataset
import random
import logging
from collections import defaultdict


class PathCriticDataset(Dataset):
    """
    路径判别器训练数据集
    从图谱中采样真实路径作为正样本，生成类型感知的负样本
    """

    def __init__(self, graph, entity_to_type, rules, num_samples=100000,
                 neg_ratio=1.0, type_aware_sampling=True, max_path_length=3):
        """
        初始化数据集

        参数:
            graph: KnowledgeGraph实例
            entity_to_type: Dict[int, int] 实体到类型的映射
            rules: List[Tuple] 规则列表，格式: [(rule_id, r_head, r_body), ...]
            num_samples: 采样的正样本数量
            neg_ratio: 负样本比例（每个正样本对应的负样本数）
            type_aware_sampling: 是否使用类型感知负采样
            max_path_length: 最大路径长度（跳数）
        """
        self.graph = graph
        self.entity_to_type = entity_to_type
        self.rules = rules
        self.num_samples = num_samples
        self.neg_ratio = neg_ratio
        self.type_aware_sampling = type_aware_sampling
        self.max_path_length = max_path_length

        logging.info(f"初始化PathCriticDataset: num_samples={num_samples}, "
                    f"neg_ratio={neg_ratio}, type_aware={type_aware_sampling}")
        logging.info(f"规则数量: {len(rules)}")

        # 构建类型到实体的反向索引
        self.type_to_entities = self._build_type_index()

        # 构建关系到规则的映射
        self.relation2rules = self._build_rule_index()

        # 生成样本
        self.samples = self._generate_samples()

        logging.info(f"数据集构造完成，总样本数={len(self.samples)}")

    def _build_type_index(self):
        """构建类型到实体列表的反向索引"""
        type_to_entities = defaultdict(list)
        for entity_id, type_id in self.entity_to_type.items():
            type_to_entities[type_id].append(entity_id)

        logging.info(f"类型索引构建完成，共{len(type_to_entities)}个类型")
        return type_to_entities

    def _build_rule_index(self):
        """构建关系到规则的映射"""
        relation2rules = defaultdict(list)
        for rule in self.rules:
            rule_id = rule[0]
            r_head = rule[1]
            r_body = rule[2:]
            relation2rules[r_head].append((rule_id, r_head, r_body))

        logging.info(f"规则索引构建完成，共{len(relation2rules)}个关系有对应规则")
        return relation2rules

    def _sample_positive_paths(self):
        """
        基于规则grounding采样正样本路径
        从训练集三元组出发，使用规则grounding找到能连接到真实尾实体的路径
        最多采样num_samples个，如果遍历完所有三元组不够就用实际数量

        返回:
            paths: List[Tuple] 路径列表
        """
        paths = []
        train_facts = self.graph.ground_train_facts

        logging.info(f"开始从{len(train_facts)}个训练三元组中基于规则grounding采样路径...")
        logging.info(f"目标采样数量: {self.num_samples}")

        # 遍历训练集三元组
        for idx, (h, r, t) in enumerate(train_facts):
            # 获取该关系对应的规则
            if r not in self.relation2rules:
                continue

            # 对每个规则进行grounding
            for rule_id, r_head, r_body in self.relation2rules[r]:
                try:
                    # 使用grounding_with_paths获取路径
                    h_tensor = torch.tensor([h], dtype=torch.long)
                    rule_paths, count = self.graph.grounding_with_paths(
                        h_tensor, r_head, r_body, edges_to_remove=None
                    )

                    # 筛选：只保留能连接到真实尾实体t的路径（通用判断）
                    for path in rule_paths:
                        # 路径格式：(e0, r1, e1, r2, e2, ..., rN, eN)
                        # 尾实体总是最后一个元素
                        if len(path) > 0 and path[0] == h and path[-1] == t:
                            paths.append(path)

                        # 达到目标数量就停止
                        if len(paths) >= self.num_samples:
                            break

                except Exception as e:
                    logging.warning(f"Grounding失败 (h={h}, r={r}, rule={rule_id}): {e}")
                    continue

                # 达到目标数量就停止
                if len(paths) >= self.num_samples:
                    break

            # 达到目标数量就停止
            if len(paths) >= self.num_samples:
                break

            # 进度显示
            if (idx + 1) % 1000 == 0:
                logging.info(f"  已处理 {idx + 1}/{len(train_facts)} 个三元组，采样到 {len(paths)} 条路径")

        logging.info(f"正样本采样完成，共{len(paths)}条路径（目标: {self.num_samples}）")

        # 统计路径长度分布（通用统计）
        from collections import Counter
        path_lengths = [len(p) for p in paths]
        length_counter = Counter(path_lengths)

        logging.info(f"路径长度分布:")
        for length in sorted(length_counter.keys()):
            hop_num = (length - 1) // 2  # 计算hop数：(长度-1)/2
            count = length_counter[length]
            logging.info(f"  - {hop_num}-hop路径（长度{length}）: {count}条")

        return paths

    def _generate_negative_path(self, positive_path):
        """
        生成负样本路径（替换中间节点）- 通用版本，支持任意长度路径

        参数:
            positive_path: 正样本路径
                格式: (e0, r1, e1, r2, e2, ..., rN, eN)
                奇数位置是实体，偶数位置是关系

        返回:
            negative_path: 负样本路径（替换中间节点）
        """
        path_len = len(positive_path)

        # 找到所有实体的位置（偶数索引：0, 2, 4, ...）
        entity_positions = [i for i in range(0, path_len, 2)]

        # 排除首尾实体，只保留中间实体
        middle_entity_positions = entity_positions[1:-1]

        if len(middle_entity_positions) == 0:
            # 没有中间实体（只有1-hop路径：e0, r1, e1），无法构造负样本
            # 这种情况下，随机替换尾实体
            neg_path = list(positive_path)
            neg_path[-1] = self._sample_negative_entity(positive_path[-1])
            return tuple(neg_path)

        # 随机选择一个中间实体位置
        pos = random.choice(middle_entity_positions)

        # 替换该位置的实体
        neg_path = list(positive_path)
        neg_path[pos] = self._sample_negative_entity(positive_path[pos])

        return tuple(neg_path)

    def _sample_negative_entity(self, original_entity):
        """
        采样负样本实体

        参数:
            original_entity: 原始实体ID

        返回:
            negative_entity: 负样本实体ID
        """
        if self.type_aware_sampling:
            # 类型感知负采样：替换为同类型的其他实体
            type_id = self.entity_to_type.get(original_entity, 0)
            candidates = self.type_to_entities[type_id]

            # 过滤掉原实体
            candidates = [e for e in candidates if e != original_entity]

            if len(candidates) == 0:
                # 如果没有同类型的其他实体，随机采样
                neg_entity = random.randint(0, self.graph.entity_size - 1)
                while neg_entity == original_entity:
                    neg_entity = random.randint(0, self.graph.entity_size - 1)
            else:
                neg_entity = random.choice(candidates)
        else:
            # 随机负采样
            neg_entity = random.randint(0, self.graph.entity_size - 1)
            while neg_entity == original_entity:  # 确保不同
                neg_entity = random.randint(0, self.graph.entity_size - 1)

        return neg_entity

    def _generate_samples(self):
        """生成所有训练样本（正样本+负样本）"""
        samples = []

        # 采样正样本
        positive_paths = self._sample_positive_paths()
        num_positive = len(positive_paths)

        logging.info(f"开始生成负样本，neg_ratio={self.neg_ratio}")

        # 为每个正样本生成负样本
        for pos_path in positive_paths:
            # 添加正样本
            samples.append((pos_path, 1))

            # 生成负样本
            num_neg = int(self.neg_ratio)
            for _ in range(num_neg):
                neg_path = self._generate_negative_path(pos_path)
                samples.append((neg_path, 0))

        num_negative = len(samples) - num_positive

        logging.info(f"负样本生成完成，共{num_negative}条负样本")
        logging.info(f"数据集统计:")
        logging.info(f"  - 正样本: {num_positive}条")
        logging.info(f"  - 负样本: {num_negative}条")
        logging.info(f"  - 总样本: {len(samples)}条")
        logging.info(f"  - 正负比例: 1:{self.neg_ratio}")

        # 打乱样本顺序
        random.shuffle(samples)
        logging.info(f"样本顺序已打乱")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回一个样本

        返回:
            path: Tuple (e0, r1, e1, r2, e2)
            label: int (0或1)
        """
        path, label = self.samples[idx]
        return path, label


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("PathCriticDataset模块已创建")
    print("此模块需要配合KnowledgeGraph使用")
