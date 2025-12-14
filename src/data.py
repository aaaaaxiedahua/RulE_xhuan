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
    def __init__(self, data_path, max_num_actions=200):
        """
        初始化知识图谱

        参数:
            data_path: 数据路径
            max_num_actions: 每个实体的最大动作数（用于策略网络，参考SSRL设计）
        """
        self.data_path = data_path
        self.max_num_actions = max_num_actions

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

        # RulE-SSRL: 预计算动作空间数组（参考SSRL grapher设计）
        # 用于策略网络的高效动作空间检索
        self.ePAD = self.entity_size  # 实体padding索引
        self.rPAD = self.relation_size * 2  # 关系padding索引
        self.array_store = None  # 将在读取训练数据后初始化
        
        
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

        # RulE-SSRL: 构建预计算的动作空间数组（参考SSRL grapher设计）
        print("Building action space array store...")
        self._build_array_store()
        print("Action space array store built!")

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

    def _build_array_store(self):
        """
        预计算动作空间数组（参考SSRL grapher.py设计）

        构建一个形状为 [entity_size, max_num_actions, 2] 的numpy数组，
        其中每个实体存储其所有可用动作：
        - array_store[entity, action_idx, 0] = 目标实体
        - array_store[entity, action_idx, 1] = 关系
        - 不足max_num_actions的部分用PAD填充
        - 超过max_num_actions的部分被截断
        """
        # 初始化数组，用PAD填充
        self.array_store = np.ones((self.entity_size, self.max_num_actions, 2), dtype=np.int32)
        self.array_store[:, :, 0] *= self.ePAD
        self.array_store[:, :, 1] *= self.rPAD

        # 首先为每个实体添加NO_OP动作（自环）
        for entity in range(self.entity_size):
            self.array_store[entity, 0, 0] = entity
            self.array_store[entity, 0, 1] = self.rPAD  # NO_OP用rPAD表示

        # 然后从hr2o字典填充实际的动作
        action_counts = [1] * self.entity_size  # 每个实体已经有1个NO_OP动作

        for hr_index, tails in self.hr2o.items():
            h, r = self.decode_hr(hr_index)

            for t in tails:
                if action_counts[h] >= self.max_num_actions:
                    break  # 达到最大动作数，截断

                self.array_store[h, action_counts[h], 0] = t
                self.array_store[h, action_counts[h], 1] = r
                action_counts[h] += 1

    def get_num_actions(self, entity):
        """
        获取实体的有效动作数量（不包括PAD）

        参数:
            entity: 实体ID

        返回:
            动作数量
        """
        return np.where(self.array_store[entity, :, 0] != self.ePAD)[0].shape[0]

    def return_next_actions(self, current_entities, start_entities, query_relations,
                           answers, all_correct_answers, last_step, num_rollouts):
        """
        返回批量实体的下一步可用动作（参考SSRL environment.py设计）

        参数:
            current_entities: 当前实体数组 [batch_size * num_rollouts]
            start_entities: 起始实体数组（用于避免查询边）
            query_relations: 查询关系数组
            answers: 目标答案数组
            all_correct_answers: 所有正确答案列表（用于最后一步过滤）
            last_step: 是否是最后一步
            num_rollouts: 每个查询的rollout数量

        返回:
            ret: [batch_size * num_rollouts, max_num_actions, 2] 动作数组
                 ret[:, :, 0] = 目标实体
                 ret[:, :, 1] = 关系
        """
        ret = self.array_store[current_entities, :, :].copy()

        # 对于每个实体，检查是否需要移除查询边或过滤答案
        for i in range(current_entities.shape[0]):
            # 如果仍在起点，移除查询边（避免直接走到答案）
            if current_entities[i] == start_entities[i]:
                relations = ret[i, :, 1]
                entities = ret[i, :, 0]
                mask = np.logical_and(relations == query_relations[i], entities == answers[i])
                ret[i, :, 0][mask] = self.ePAD
                ret[i, :, 1][mask] = self.rPAD

            # 如果是最后一步，过滤掉其他正确答案（只保留当前目标答案）
            if last_step:
                entities = ret[i, :, 0]
                relations = ret[i, :, 1]

                correct_e2 = answers[i]
                for j in range(entities.shape[0]):
                    if entities[j] in all_correct_answers[int(i / num_rollouts)] and entities[j] != correct_e2:
                        entities[j] = self.ePAD
                        relations[j] = self.rPAD

        return ret

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


# ========== RulE-SSRL: 策略训练的查询数据集 ==========
class QueryDataset(Dataset):
    """
    策略网络训练的查询数据集

    提供(h, r, t)三元组作为策略网络的监督信号。
    每个三元组作为一个查询，策略网络需要学习从h导航到t，
    使用关系r作为引导。

    数据来源：知识图谱的训练三元组
    """

    def __init__(self, train_triplets):
        """
        初始化查询数据集

        参数：
            train_triplets: 训练集中的(h, r, t)元组列表
        """
        self.queries = train_triplets

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        """
        获取单个查询

        返回：
            (h, r, t): 查询三元组
                h - 头实体ID
                r - 查询关系ID
                t - 目标实体ID（ground truth）
        """
        return self.queries[idx]

    @staticmethod
    def collate_fn(data):
        """
        DataLoader的批整理函数

        参数：
            data: (h, r, t)元组列表

        返回：
            batch: [batch_size, 3]张量
        """
        batch = torch.stack([torch.LongTensor(triplet) for triplet in data], dim=0)
        return batch

