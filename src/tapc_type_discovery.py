"""
TAPC-RulE: Type Discovery Module
类型发现模块 - 通过K-Means聚类自动发现实体类型
"""

import torch
import numpy as np
import json
import logging
from sklearn.cluster import KMeans
from collections import defaultdict
import os


class TypeDiscovery:
    """
    类型发现器
    使用K-Means聚类将实体划分为不同的语义类型
    """

    def __init__(self, num_clusters=50, random_state=42):
        """
        初始化类型发现器

        参数:
            num_clusters: 聚类数量（类型数量）
            random_state: 随机种子
        """
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.kmeans = None
        self.entity_to_type = {}
        self.cluster_centers = None

        logging.info(f"初始化TypeDiscovery，聚类数量K={num_clusters}")

    def fit(self, entity_embeddings):
        """
        执行K-Means聚类

        参数:
            entity_embeddings: [num_entities, embedding_dim] numpy数组

        返回:
            entity_to_type: Dict[int, int] 实体ID到类型ID的映射
        """
        logging.info(f"开始K-Means聚类，实体数量={entity_embeddings.shape[0]}, "
                    f"嵌入维度={entity_embeddings.shape[1]}")

        # 执行K-Means聚类
        self.kmeans = KMeans(
            n_clusters=self.num_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300,
            verbose=1
        )

        cluster_ids = self.kmeans.fit_predict(entity_embeddings)

        # 构建实体到类型的映射
        self.entity_to_type = {
            entity_id: int(cluster_id)
            for entity_id, cluster_id in enumerate(cluster_ids)
        }

        # 保存聚类中心
        self.cluster_centers = self.kmeans.cluster_centers_

        # 统计每个类型的实体数量
        type_counts = defaultdict(int)
        for type_id in cluster_ids:
            type_counts[type_id] += 1

        logging.info(f"聚类完成！生成{self.num_clusters}个类型")
        logging.info(f"类型分布统计:")

        # 计算统计信息
        counts_list = list(type_counts.values())
        avg_count = np.mean(counts_list)
        max_count = max(counts_list)
        min_count = min(counts_list)

        logging.info(f"  - 平均每类实体数: {avg_count:.2f}")
        logging.info(f"  - 最大类实体数: {max_count}")
        logging.info(f"  - 最小类实体数: {min_count}")
        logging.info(f"前10个类型的实体分布:")
        for type_id in sorted(type_counts.keys())[:10]:
            logging.info(f"  Type {type_id}: {type_counts[type_id]} 个实体")

        return self.entity_to_type

    def save(self, save_path):
        """
        保存类型映射和聚类中心

        参数:
            save_path: 保存目录
        """
        os.makedirs(save_path, exist_ok=True)

        # 保存entity_to_type映射
        mapping_file = os.path.join(save_path, 'entity_to_type.json')
        with open(mapping_file, 'w') as f:
            json.dump(self.entity_to_type, f)
        logging.info(f"类型映射已保存到: {mapping_file}")

        # 保存聚类中心
        centers_file = os.path.join(save_path, 'cluster_centers.npy')
        np.save(centers_file, self.cluster_centers)
        logging.info(f"聚类中心已保存到: {centers_file}")

        # 保存元信息
        meta_file = os.path.join(save_path, 'type_discovery_meta.json')
        meta_info = {
            'num_clusters': self.num_clusters,
            'num_entities': len(self.entity_to_type),
            'embedding_dim': self.cluster_centers.shape[1]
        }
        with open(meta_file, 'w') as f:
            json.dump(meta_info, f, indent=2)
        logging.info(f"元信息已保存到: {meta_file}")

    def load(self, save_path):
        """
        加载类型映射和聚类中心

        参数:
            save_path: 保存目录
        """
        # 加载entity_to_type映射
        mapping_file = os.path.join(save_path, 'entity_to_type.json')
        with open(mapping_file, 'r') as f:
            # JSON的key是字符串，需要转换为int
            loaded_mapping = json.load(f)
            self.entity_to_type = {int(k): v for k, v in loaded_mapping.items()}
        logging.info(f"类型映射已加载，共{len(self.entity_to_type)}个实体")

        # 加载聚类中心
        centers_file = os.path.join(save_path, 'cluster_centers.npy')
        self.cluster_centers = np.load(centers_file)
        logging.info(f"聚类中心已加载，形状={self.cluster_centers.shape}")

        return self.entity_to_type


def extract_entity_embeddings(rule_model):
    """
    从训练好的RulE模型中提取实体嵌入

    参数:
        rule_model: RulE模型实例

    返回:
        entity_embeddings: [num_entities, embedding_dim] numpy数组
    """
    logging.info("从RulE模型提取实体嵌入...")

    # 提取实体嵌入矩阵
    entity_emb = rule_model.entity_embedding.weight.data

    # 转换为numpy数组
    if entity_emb.is_cuda:
        entity_emb = entity_emb.cpu()
    entity_embeddings = entity_emb.numpy()

    logging.info(f"提取完成，形状={entity_embeddings.shape}")

    return entity_embeddings


def analyze_type_distribution(entity_to_type, entity_names=None):
    """
    分析类型分布（可选的可视化工具）

    参数:
        entity_to_type: 实体到类型的映射
        entity_names: 实体ID到名称的映射（可选）
    """
    from collections import Counter

    type_counts = Counter(entity_to_type.values())

    print("\n=== 类型分布分析 ===")
    print(f"总类型数: {len(type_counts)}")
    print(f"总实体数: {len(entity_to_type)}")
    print(f"\n各类型实体数量（Top 10）:")

    for type_id, count in type_counts.most_common(10):
        print(f"  Type {type_id}: {count} 个实体")

        # 如果提供了实体名称，显示该类型的示例实体
        if entity_names:
            entities_in_type = [
                entity_names.get(eid, f"Entity_{eid}")
                for eid, tid in entity_to_type.items()
                if tid == type_id
            ][:5]  # 显示前5个
            print(f"    示例: {', '.join(entities_in_type)}")

    print("\n" + "="*50)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    print("TypeDiscovery模块测试")
    print("此模块需要配合RulE模型使用")
    print("请在main.py中调用此模块")
