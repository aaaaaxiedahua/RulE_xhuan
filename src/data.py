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

    def __getstate__(self):
        """pickle 时排除稀疏张量，解决 Windows 多进程 DataLoader 序列化问题"""
        state = self.__dict__.copy()
        state.pop('relation2sparse', None)
        state.pop('use_sparse', None)
        return state

    def build_sparse_adjacency(self, device=None):
        """
        预构建每个关系的稀疏邻接矩阵，可选一次性搬到 GPU。
        调用时机：在 main.py 中创建 graph 后、确定 device 后调用一次。
        """
        self.relation2sparse = [None] * (self.relation_size * 2)
        self.use_sparse = True

        for r in range(self.relation_size * 2):
            # relation2adjacency[r][0] 是 [2, num_edges]: [targets, sources]
            index = self.relation2adjacency[r][0]
            value = self.relation2adjacency[r][1]

            # 构建 sparse COO: A[target, source] = 1，表示边 source → target
            adj = torch.sparse_coo_tensor(
                index, value,
                size=(self.entity_size, self.entity_size)
            ).coalesce()

            if device is not None and device.type == "cuda":
                adj = adj.cuda(device)

            self.relation2sparse[r] = adj

        print("Sparse adjacency matrices built | DONE!")

    def grounding(self, h, r, rule, edges_to_remove):
        """
        规则 grounding：从头实体 h 出发，沿规则体传播，计算到达每个实体的次数。
        如果已调用 build_sparse_adjacency()，则使用稀疏模式；否则使用原密集模式。
        """
        device = h.device

        # 检查是否启用稀疏模式
        if hasattr(self, 'use_sparse') and self.use_sparse:
            return self._grounding_sparse(h, r, rule, edges_to_remove)
        else:
            return self._grounding_dense(h, r, rule, edges_to_remove)

    def _grounding_dense(self, h, r, rule, edges_to_remove):
        """原始密集模式 grounding（保持不变）"""
        device = h.device
        with torch.no_grad():
            x = torch.nn.functional.one_hot(h, self.entity_size).transpose(0, 1).unsqueeze(-1)
            if device.type == "cuda":
                x = x.cuda(device)
            for r_body in rule:
                if r_body == r:
                    x = self._propagate_dense(x, r_body, edges_to_remove)
                else:
                    x = self._propagate_dense(x, r_body, None)
        return x.squeeze(-1).transpose(0, 1)

    def _grounding_sparse(self, h, r, rule, edges_to_remove):
        """稀疏模式 grounding：使用稀疏矩阵乘法，减少显存占用"""
        device = h.device
        batch = h.size(0)

        with torch.no_grad():
            # 初始化：[entity_size, batch] 稀疏表示的起点
            x = torch.zeros(self.entity_size, batch, device=device)
            x[h, torch.arange(batch, device=device)] = 1.0

            for r_body in rule:
                if r_body == r:
                    x = self._propagate_sparse(x, r_body, edges_to_remove)
                else:
                    x = self._propagate_sparse(x, r_body, None)

        return x.transpose(0, 1)  # [batch, entity_size]

    def _propagate_dense(self, x, relation, edges_to_remove=None):
        """原始密集模式 propagate（保持不变）"""
        device = x.device
        node_in = self.relation2adjacency[relation][0][1]  # h (source)
        node_out = self.relation2adjacency[relation][0][0]  # t (target)
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

    def _propagate_sparse(self, x, relation, edges_to_remove=None):
        """
        稀疏模式 propagate：使用 torch.sparse.mm 替代 gather + scatter。

        Args:
            x: [entity_size, batch] 密集张量，当前实体状态
            relation: 关系 ID
            edges_to_remove: [batch] 每个 batch 样本要删除的边索引（可选）

        Returns:
            x_new: [entity_size, batch] 传播后的实体状态
        """
        adj = self.relation2sparse[relation]  # 已在 GPU 上的稀疏矩阵

        # 核心操作：稀疏矩阵 × 密集矩阵 → 密集矩阵
        # adj: [entity_size, entity_size] sparse, x: [entity_size, batch] dense
        x_new = torch.sparse.mm(adj, x)  # [entity_size, batch]

        if edges_to_remove is not None:
            # 修正被删除边的贡献
            # edges_to_remove[i] 是第 i 个 batch 样本要删除的边在邻接列表中的索引
            indices = adj.indices()  # [2, num_edges]: [targets, sources]
            node_out = indices[0]    # target 节点
            node_in = indices[1]     # source 节点

            batch = x.size(1)
            batch_range = torch.arange(batch, device=x.device)

            # 获取被删边的源节点和目标节点
            src = node_in[edges_to_remove]   # [batch] 被删边的源节点
            dst = node_out[edges_to_remove]  # [batch] 被删边的目标节点

            # 减去被删边的贡献：x_new[dst[i], i] -= x[src[i], i]
            removed_val = x[src, batch_range]  # [batch]
            x_new[dst, batch_range] = x_new[dst, batch_range] - removed_val

        return x_new

    # 保留旧的 propagate 作为兼容接口
    def propagate(self, x, relation, edges_to_remove=None):
        """兼容接口：根据是否启用稀疏模式选择实现"""
        if hasattr(self, 'use_sparse') and self.use_sparse:
            # 稀疏模式需要 2D 输入，处理 3D 输入的情况
            if x.dim() == 3:
                x_2d = x.squeeze(-1)
                result = self._propagate_sparse(x_2d, relation, edges_to_remove)
                return result.unsqueeze(-1)
            else:
                return self._propagate_sparse(x, relation, edges_to_remove)
        else:
            return self._propagate_dense(x, relation, edges_to_remove)

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

class EvalDataset(Dataset):
    """
    评估数据集 - 支持选择valid或test数据
    """
    def __init__(self, graph, split='valid', batch_size=1):
        self.graph = graph
        self.batch_size = batch_size
        self.split = split

        # 根据split选择数据
        if split == 'valid':
            facts = self.graph.valid_facts
        elif split == 'test':
            facts = self.graph.test_facts
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'valid' or 'test'")

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

