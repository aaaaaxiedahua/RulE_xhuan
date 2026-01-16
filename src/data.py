import torch
from torch.utils.data import Dataset
from torch_scatter import scatter
import numpy as np
import os
import random
from easydict import EasyDict
from torch.nn.utils.rnn import pad_sequence


class RuleDataset(Dataset):
    def __init__(self, num_relations, input, negative_sample_size, mode='batch'):
        self.rules = list()
        self.num_relations = num_relations
        # self.ending_idx = num_relations
        self.padding_idx = num_relations * 2 
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        idx = 0

        self.rules = []
        with open(input, 'r') as fi:
            for line in fi:
                rule = line.strip().split()
                if len(rule) <= 1:
                    continue
                rule = [idx] + [int(_) for _ in rule]
                idx += 1
                formatted_rule = [rule, self.padding_idx]
                self.rules.append(formatted_rule)
            
    
    def __len__(self):
        return len(self.rules)

    def __getitem__(self, idx):
        positive_sample = self.rules[idx]

        # negative samping
        # negative_idx = np.zeros(self.negative_sample_size)
        negative_idx = np.random.randint(len(positive_sample[0])-1 ,size=self.negative_sample_size)
        
        negative_sample = np.random.randint(self.num_relations, size=self.negative_sample_size)  
        negative_idx = torch.LongTensor(negative_idx)
        negative_sample = torch.LongTensor(negative_sample)

        # # set rule mask to get the last element of during RNN
        # rule_mask = torch.zeros(len(positive_sample[0])-2).bool()
        
        # rule_mask[-1] = True

        # # set rule mask to get the last element of during RotatE3D
        rule_mask = torch.ones(len(positive_sample[0])-2).bool()
        # rule_mask[-1] = False

        return positive_sample, negative_idx, negative_sample, self.mode, rule_mask  
        

    @staticmethod
    def collate_fn(data):

        positive_sample = pad_sequence([torch.LongTensor(_[0][0]) for _ in data], batch_first=True, padding_value=data[0][0][-1])
        negative_idx = torch.stack([_[1] for _ in data], dim=0)
        negative_sample = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        rule_mask = pad_sequence([_[4] for _ in data], batch_first=True, padding_value=False)
        
        return positive_sample, negative_idx, negative_sample, mode, rule_mask


class KGETrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail


class RuleETestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode

class KnowledgeGraph(object):
    def __init__(self, data_path):
        self.data_path = data_path

        self.entity2id = dict()
        self.relation2id = dict()
        self.id2entity = dict()
        self.id2relation = dict()

        with open(os.path.join(data_path, 'entities.dict')) as fi:
            for line in fi:
                id, entity = line.strip().split('\t')
                self.entity2id[entity] = int(id)
                self.id2entity[int(id)] = entity

        with open(os.path.join(data_path, 'relations.dict')) as fi:
            for line in fi:
                id, relation = line.strip().split('\t')
                self.relation2id[relation] = int(id)
                self.id2relation[int(id)] = relation

        self.entity_size = len(self.entity2id)
        self.relation_size = len(self.relation2id)
        
        
        self.train_facts = list()
        self.ground_train_facts = list()
        self.valid_facts = list()
        self.test_facts = list()
        self.hr2o = dict()          # only contain training set
        self.hr2oo = dict()         # contain training and valid set
        self.hr2ooo = dict()        # contain training, valid and test set
        self.relation2adjacency = [[[], []] for k in range(self.relation_size*2)]
        self.relation2ht2index = [dict() for k in range(self.relation_size*2)]
        self.relation2outdegree = [[0 for i in range(self.entity_size)] for k in range(self.relation_size*2)]

        with open(os.path.join(data_path, "train.txt")) as fi:
            for line in fi:
                h, r, t = line.strip().split('\t')
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                self.train_facts.append((h, r, t))
                self.ground_train_facts.append((h,r,t))
                # self.ground_train_facts.append((t, r + self.relation_size, t))
                
                hr_index = self.encode_hr(h, r)
                if hr_index not in self.hr2o:
                    self.hr2o[hr_index] = list()
                self.hr2o[hr_index].append(t)

                if hr_index not in self.hr2oo:
                    self.hr2oo[hr_index] = list()
                self.hr2oo[hr_index].append(t)

                if hr_index not in self.hr2ooo:
                    self.hr2ooo[hr_index] = list()
                self.hr2ooo[hr_index].append(t)

            
                self.relation2adjacency[r][0].append(t)
                self.relation2adjacency[r][1].append(h)

                ht_index = self.encode_ht(h, t)
                assert ht_index not in self.relation2ht2index[r]
                index = len(self.relation2ht2index[r])
                self.relation2ht2index[r][ht_index] = index

                self.relation2outdegree[r][t] += 1

                # get inverse facts
                h, r, t = line.strip().split('\t')
                h, r, t = self.entity2id[t], self.relation2id[r] + self.relation_size, self.entity2id[h]
                self.ground_train_facts.append((h,r,t))

                hr_index = self.encode_hr(h, r)
                if hr_index not in self.hr2o:
                    self.hr2o[hr_index] = list()
                self.hr2o[hr_index].append(t)

                if hr_index not in self.hr2oo:
                    self.hr2oo[hr_index] = list()
                self.hr2oo[hr_index].append(t)

                if hr_index not in self.hr2ooo:
                    self.hr2ooo[hr_index] = list()
                self.hr2ooo[hr_index].append(t)

            
                self.relation2adjacency[r][0].append(t)
                self.relation2adjacency[r][1].append(h)

                ht_index = self.encode_ht(h, t)
                assert ht_index not in self.relation2ht2index[r]
                index = len(self.relation2ht2index[r])
                self.relation2ht2index[r][ht_index] = index

                self.relation2outdegree[r][t] += 1

        with open(os.path.join(data_path, "valid.txt")) as fi:
            for line in fi:
                h, r, t = line.strip().split('\t')
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                self.valid_facts.append((h, r, t))

                hr_index = self.encode_hr(h, r)

                if hr_index not in self.hr2oo:
                    self.hr2oo[hr_index] = list()
                self.hr2oo[hr_index].append(t)

                if hr_index not in self.hr2ooo:
                    self.hr2ooo[hr_index] = list()
                self.hr2ooo[hr_index].append(t)

                h, r, t = line.strip().split('\t')
                h, r, t = self.entity2id[t], self.relation2id[r] + self.relation_size, self.entity2id[h]
                self.valid_facts.append((h, r, t))

                hr_index = self.encode_hr(h, r)

                if hr_index not in self.hr2oo:
                    self.hr2oo[hr_index] = list()
                self.hr2oo[hr_index].append(t)

                if hr_index not in self.hr2ooo:
                    self.hr2ooo[hr_index] = list()
                self.hr2ooo[hr_index].append(t)

                

        with open(os.path.join(data_path, "test.txt")) as fi:
            for line in fi:
                h, r, t = line.strip().split('\t')
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                self.test_facts.append((h, r, t))

                hr_index = self.encode_hr(h, r)

                if hr_index not in self.hr2ooo:
                    self.hr2ooo[hr_index] = list()
                self.hr2ooo[hr_index].append(t)

                h, r, t = line.strip().split('\t')
                h, r, t = self.entity2id[t], self.relation2id[r]+self.relation_size, self.entity2id[h]
                self.test_facts.append((h, r, t))

                hr_index = self.encode_hr(h, r)

                if hr_index not in self.hr2ooo:
                    self.hr2ooo[hr_index] = list()
                self.hr2ooo[hr_index].append(t)



        for r in range(self.relation_size * 2):
            index = torch.LongTensor(self.relation2adjacency[r])
            value = torch.ones(index.size(1))
            self.relation2adjacency[r] = [index, value]

            self.relation2outdegree[r] = torch.LongTensor(self.relation2outdegree[r])

        print("Data loading | DONE!")

    def encode_hr(self, h, r):
        return r * self.entity_size + h

    def decode_hr(self, index):
        h, r = index % self.entity_size, index // self.entity_size
        return h, r

    def encode_ht(self, h, t):
        return t * self.entity_size + h

    def decode_ht(self, index):
        h, t = index % self.entity_size, index // self.entity_size
        return h, t

    def get_updated_adjacency(self, r, edges_to_remove):
        if edges_to_remove == None:
            return None
        index = self.relation2sparse[r][0]
        value = self.relation2sparse[r][1]
        mask = (index.unsqueeze(1) == edges_to_remove.unsqueeze(-1))
        mask = mask.all(dim=0).any(dim=0)
        mask = ~mask
        index = index[:, mask]
        value = value[mask]
        return [index, value]

    def grounding(self, h, r, rule, edges_to_remove):
        device = h.device
        with torch.no_grad():
            x = torch.nn.functional.one_hot(h, self.entity_size).transpose(0, 1).unsqueeze(-1)
            if device.type == "cuda":
                x = x.cuda(device)
            for r_body in rule:
                if r_body == r:
                    x = self.propagate(x, r_body, edges_to_remove)
                else:
                    x = self.propagate(x, r_body, None)
        return x.squeeze(-1).transpose(0, 1)

    def propagate(self, x, relation, edges_to_remove=None):
        device = x.device
        node_in = self.relation2adjacency[relation][0][1] # h
        node_out = self.relation2adjacency[relation][0][0] # t
        if device.type == "cuda":
            node_in = node_in.cuda(device)
            node_out = node_out.cuda(device)

        message = x[node_in]
        E, B, D = message.size()

        if edges_to_remove == None:
            x = scatter(message, node_out, dim=0, dim_size=x.size(0))
        else:
            # message: edge * batch * dim
            message = message.view(-1, D)
            bias = torch.arange(B)
            if device.type == "cuda":
                bias = bias.cuda(device)
            edges_to_remove = edges_to_remove * B + bias
            message[edges_to_remove] = 0
            message = message.view(E, B, D)
            x = scatter(message, node_out, dim=0, dim_size=x.size(0))

        return x

    def get_neighbors_with_relations(self, entity_id):
        """
        获取实体的所有邻居及对应的关系
        用于路径采样

        参数:
            entity_id: 实体ID

        返回:
            neighbors: List[(relation_id, neighbor_id)]
        """
        neighbors = []

        # 遍历所有关系
        for r in range(self.relation_size * 2):
            # 获取该关系的邻接表
            adjacency = self.relation2adjacency[r]
            node_in = adjacency[0][1]  # 头实体列表
            node_out = adjacency[0][0]  # 尾实体列表

            # 查找entity_id作为头实体的边
            if isinstance(node_in, torch.Tensor):
                node_in = node_in.cpu().numpy()
                node_out = node_out.cpu().numpy()

            indices = np.where(node_in == entity_id)[0]
            for idx in indices:
                neighbor = int(node_out[idx])
                neighbors.append((r, neighbor))

        return neighbors

    def grounding_with_paths(self, h, r, rule, edges_to_remove):
        """
        扩展的grounding方法，返回具体路径实例（通用版本，支持任意长度规则）

        参数:
            h: [batch_size] 头实体tensor
            r: 目标关系ID
            rule: 规则体（关系序列）
            edges_to_remove: 需要移除的边

        返回:
            paths: List[Tuple] 路径列表，格式: (e0, r1, e1, r2, e2, ..., rN, eN)
            count: [batch_size, num_entities] 计数矩阵（保持兼容）
        """
        # 首先调用原始grounding获取计数矩阵
        count = self.grounding(h, r, rule, edges_to_remove)

        # 使用通用方法搜索路径（支持任意长度规则）
        paths = self._grounding_generic(h, rule, edges_to_remove)

        return paths, count

    def _grounding_generic(self, h, rule, edges_to_remove):
        """
        通用的路径搜索方法，支持任意长度的规则
        使用递归方式搜索路径

        参数:
            h: [batch_size] 头实体tensor
            rule: 规则体（关系序列）[r1, r2, ..., rN]
            edges_to_remove: 需要移除的边

        返回:
            paths: List[Tuple] 路径列表，格式: (e0, r1, e1, r2, e2, ..., rN, eN)
        """
        paths = []

        # 将tensor转为列表
        if isinstance(h, torch.Tensor):
            h_list = h.cpu().numpy().tolist()
        else:
            h_list = [h] if not isinstance(h, list) else h

        # 对每个头实体进行路径搜索
        for h_entity in h_list:
            # 从头实体开始递归搜索
            current_paths = self._search_paths_recursive(h_entity, rule, 0, [h_entity])
            paths.extend(current_paths)

        return paths

    def _search_paths_recursive(self, current_entity, rule, rule_index, current_path):
        """
        递归搜索路径

        参数:
            current_entity: 当前实体
            rule: 规则体（关系序列）
            rule_index: 当前处理到规则的第几个关系
            current_path: 当前路径（已经走过的实体和关系）

        返回:
            paths: 从current_entity出发，按照rule剩余部分能到达的所有路径
        """
        # 递归终止条件：规则已经全部匹配完
        if rule_index >= len(rule):
            return [tuple(current_path)]

        paths = []
        current_relation = rule[rule_index]

        # 获取当前实体的所有邻居
        neighbors = self.get_neighbors_with_relations(current_entity)

        for rel, next_entity in neighbors:
            # 检查关系是否匹配
            if rel != current_relation:
                continue

            # 避免回到起始实体（避免自环）
            if next_entity == current_path[0]:
                continue

            # 构造新路径：添加关系和下一个实体
            new_path = current_path + [current_relation, next_entity]

            # 递归搜索下一跳
            sub_paths = self._search_paths_recursive(next_entity, rule, rule_index + 1, new_path)
            paths.extend(sub_paths)

        return paths

    def _grounding_2hop(self, h, rule, edges_to_remove):
        """
        搜索2跳路径: h -> e1 -> e2

        参数:
            h: [batch_size] 头实体tensor
            rule: [r1, r2] 规则体
            edges_to_remove: 需要移除的边

        返回:
            paths: List[Tuple] 格式: [(h, r1, e1, r2, e2), ...]
        """
        paths = []
        r1, r2 = rule[0], rule[1]

        # 将tensor转为numpy以便处理
        if isinstance(h, torch.Tensor):
            h_list = h.cpu().numpy().tolist()
        else:
            h_list = [h] if not isinstance(h, list) else h

        # 对batch中的每个头实体进行路径搜索
        for h_entity in h_list:
            # 第一跳: h -> e1
            neighbors_1 = self.get_neighbors_with_relations(h_entity)

            for rel_1, e1 in neighbors_1:
                # 检查关系是否匹配
                if rel_1 != r1:
                    continue

                # 第二跳: e1 -> e2
                neighbors_2 = self.get_neighbors_with_relations(e1)

                for rel_2, e2 in neighbors_2:
                    # 检查关系是否匹配
                    if rel_2 != r2:
                        continue

                    # 避免自环
                    if e2 == h_entity:
                        continue

                    # 构造路径
                    path = (h_entity, r1, e1, r2, e2)
                    paths.append(path)

        return paths

    def _grounding_3hop(self, h, rule, edges_to_remove):
        """
        搜索3跳路径: h -> e1 -> e2 -> e3

        参数:
            h: [batch_size] 头实体tensor
            rule: [r1, r2, r3] 规则体
            edges_to_remove: 需要移除的边

        返回:
            paths: List[Tuple] 格式: [(h, r1, e1, r2, e2, r3, e3), ...]
        """
        paths = []
        r1, r2, r3 = rule[0], rule[1], rule[2]

        # 将tensor转为numpy以便处理
        if isinstance(h, torch.Tensor):
            h_list = h.cpu().numpy().tolist()
        else:
            h_list = [h] if not isinstance(h, list) else h

        # 对batch中的每个头实体进行路径搜索
        for h_entity in h_list:
            # 第一跳: h -> e1
            neighbors_1 = self.get_neighbors_with_relations(h_entity)

            for rel_1, e1 in neighbors_1:
                if rel_1 != r1:
                    continue

                # 第二跳: e1 -> e2
                neighbors_2 = self.get_neighbors_with_relations(e1)

                for rel_2, e2 in neighbors_2:
                    if rel_2 != r2:
                        continue

                    # 第三跳: e2 -> e3
                    neighbors_3 = self.get_neighbors_with_relations(e2)

                    for rel_3, e3 in neighbors_3:
                        if rel_3 != r3:
                            continue

                        # 避免自环
                        if e3 == h_entity:
                            continue

                        # 构造路径
                        path = (h_entity, r1, e1, r2, e2, r3, e3)
                        paths.append(path)

        return paths

class TrainDataset(Dataset):
    def __init__(self, graph, batch_size):
        self.graph = graph
        self.batch_size = batch_size

        self.r2instances = [[] for r in range(self.graph.relation_size * 2)]
        for h, r, t in self.graph.ground_train_facts:
            self.r2instances[r].append((h, r, t))

        self.make_batches()

    def make_batches(self):
        for r in range(self.graph.relation_size * 2):
            random.shuffle(self.r2instances[r])

        self.batches = list()
        for r, instances in enumerate(self.r2instances):
            for k in range(0, len(instances), self.batch_size):
                start = k
                end = min(k + self.batch_size, len(instances))
                self.batches.append(instances[start:end])
        random.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        data = self.batches[idx]

        all_h = torch.LongTensor([_[0] for _ in data])
        all_r = torch.LongTensor([_[1] for _ in data])
        all_t = torch.LongTensor([_[2] for _ in data])
        target = torch.zeros(len(data), self.graph.entity_size)
        edges_to_remove = []
        for k, (h, r, t) in enumerate(data):
            hr_index = self.graph.encode_hr(h, r)
            t_index = torch.LongTensor(self.graph.hr2o[hr_index])
            target[k][t_index] = 1

            ht_index = self.graph.encode_ht(h, t)
            edge = self.graph.relation2ht2index[r][ht_index]
            edges_to_remove.append(edge)
        edges_to_remove = torch.LongTensor(edges_to_remove)

        return all_h, all_r, all_t, target, edges_to_remove

class ValidDataset(Dataset):
    def __init__(self, graph, batch_size):
        self.graph = graph
        self.batch_size = batch_size

        facts = self.graph.valid_facts

        r2instances = [[] for r in range(self.graph.relation_size * 2)]
        for h, r, t in facts:
            r2instances[r].append((h, r, t))

        self.batches = list()
        for r, instances in enumerate(r2instances):
            random.shuffle(instances)
            for k in range(0, len(instances), self.batch_size):
                start = k
                end = min(k + self.batch_size, len(instances))
                self.batches.append(instances[start:end])

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        data = self.batches[idx]

        all_h = torch.LongTensor([_[0] for _ in data])
        all_r = torch.LongTensor([_[1] for _ in data])
        all_t = torch.LongTensor([_[2] for _ in data])

        mask = torch.ones(len(data), self.graph.entity_size).bool()
        for k, (h, r, t) in enumerate(data):
            hr_index = self.graph.encode_hr(h, r)
            t_index = torch.LongTensor(self.graph.hr2ooo[hr_index])
            mask[k][t_index] = 0

        return all_h, all_r, all_t, mask

class TestDataset(Dataset):
    def __init__(self, graph, batch_size):
        self.graph = graph
        self.batch_size = batch_size

        facts = self.graph.test_facts

        r2instances = [[] for r in range(self.graph.relation_size * 2)]
        for h, r, t in facts:
            r2instances[r].append((h, r, t))

        self.batches = list()
        for r, instances in enumerate(r2instances):
            random.shuffle(instances)
            for k in range(0, len(instances), self.batch_size):
                start = k
                end = min(k + self.batch_size, len(instances))
                self.batches.append(instances[start:end])

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        data = self.batches[idx]

        all_h = torch.LongTensor([_[0] for _ in data])
        all_r = torch.LongTensor([_[1] for _ in data])
        all_t = torch.LongTensor([_[2] for _ in data])

        mask = torch.ones(len(data), self.graph.entity_size).bool()
        for k, (h, r, t) in enumerate(data):
            hr_index = self.graph.encode_hr(h, r)
            t_index = torch.LongTensor(self.graph.hr2ooo[hr_index])
            mask[k][t_index] = 0

        return all_h, all_r, all_t, mask

def Iterator(dataloader):
    while True:
        for data in dataloader:
            yield data

class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data

