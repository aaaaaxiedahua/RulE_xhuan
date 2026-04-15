import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging, math
from layers import MLP, FuncToNodeSum
from torch_scatter import scatter

from torch.nn.utils.rnn import pad_sequence

class RulE(torch.nn.Module):
    def __init__(
        self,
        graph,
        p_norm,
        mlp_rule_dim,
        gamma_fact,
        gamma_rule,
        hidden_dim,
        device,
        dataset,
        reasoner_type='dual_pathway',
        g_num_layers=2,
        g_hidden_dim=128,
        g_message_hidden_dim=128,
        g_attn_dim=64,
        g_dropout=0.1,
        g_activation='relu',
        g_layer_norm=False,
        g_readout='multiply',
        reasoner_alpha=5.0,
    ):
        super(RulE, self).__init__()
        self.graph = graph
        self.device = device
        self.num_entities = graph.entity_size
        self.num_relations = graph.relation_size 
        self.total_relations = graph.relation_size * 2
        self.padding_index = graph.relation_size 

        self.hidden_dim = hidden_dim
        # self.entity_dim = hidden_dim * 2 
        # self.relation_dim = hidden_dim

        # self.rule_dim = rule_dim
        # self.rule_dim = self.relation_dim

        self.p = p_norm

        self.mlp_rule_dim = mlp_rule_dim

        
        self.rule_to_entity = FuncToNodeSum(self.mlp_rule_dim)

        if "FB15k-237" in dataset or "wn18rr" in dataset or "YAGO3-10" in dataset:
            self.score_model = MLP(self.mlp_rule_dim, [128, 1]) 
        else:
            self.score_model = MLP(self.mlp_rule_dim, [1]) 

        self.bias = torch.nn.parameter.Parameter(torch.zeros(self.num_entities))
        
        self.epsilon = 2.0

        
        self.gamma_fact = nn.Parameter(
            torch.Tensor([gamma_fact]), 
            requires_grad=False
        )

        self.gamma_rule = nn.Parameter(
            torch.Tensor([gamma_rule]), 
            requires_grad=False
        )
        
        self.embedding_range_fact = nn.Parameter(
            torch.Tensor([(self.gamma_fact.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.embedding_range_rule = nn.Parameter(
            torch.Tensor([(self.gamma_rule.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )

        self.entity_embedding = torch.nn.Embedding(self.num_entities, self.hidden_dim * 2)
        # nn.init.ones_(
        #     tensor=self.entity_embedding.weight
        # )
        nn.init.uniform_(
            tensor=self.entity_embedding.weight, 
            a=-self.embedding_range_fact.item(), 
            b=self.embedding_range_fact.item()
        )
        
        self.relation_embedding = torch.nn.Embedding(self.num_relations + 1, self.hidden_dim, padding_idx=self.padding_index)
        # nn.init.ones_(
        #     tensor=self.relation_embedding.weight
        # )
        nn.init.uniform_(
            tensor=self.relation_embedding.weight, 
            a=-self.embedding_range_fact.item(), 
            b=self.embedding_range_fact.item()
        )

        self.reasoner_type = reasoner_type
        self.g_num_layers = g_num_layers
        self.g_hidden_dim = g_hidden_dim
        self.g_message_hidden_dim = g_message_hidden_dim
        self.g_attn_dim = g_attn_dim
        self.g_activation = getattr(F, g_activation)
        self.g_layer_norm = g_layer_norm
        self.g_readout = g_readout
        self.reasoner_alpha = float(reasoner_alpha)
        self.rule_confidence = None

        self.rule_memory_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.query_seed_proj = nn.Linear(self.hidden_dim, self.g_hidden_dim, bias=False)
        self.head_context_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False)
        self.rule_score_rule_projs = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.g_attn_dim, bias=False) for _ in range(self.g_num_layers)]
        )
        self.rule_score_query_projs = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.g_attn_dim, bias=False) for _ in range(self.g_num_layers)]
        )
        self.rule_score_head_projs = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.g_attn_dim, bias=False) for _ in range(self.g_num_layers)]
        )
        self.rule_score_layers = nn.ModuleList(
            [nn.Linear(self.g_attn_dim, 1) for _ in range(self.g_num_layers)]
        )
        self.rule_confidence_scale = nn.Parameter(torch.tensor(1.0))
        self.message_node_projs = nn.ModuleList(
            [nn.Linear(self.g_hidden_dim, self.g_message_hidden_dim, bias=False) for _ in range(self.g_num_layers)]
        )
        self.message_relation_projs = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.g_message_hidden_dim, bias=False) for _ in range(self.g_num_layers)]
        )
        self.attn_source_projs = nn.ModuleList(
            [nn.Linear(self.g_hidden_dim, self.g_attn_dim, bias=False) for _ in range(self.g_num_layers)]
        )
        self.attn_relation_projs = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.g_attn_dim, bias=False) for _ in range(self.g_num_layers)]
        )
        self.attn_query_projs = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.g_attn_dim) for _ in range(self.g_num_layers)]
        )
        self.attn_score_layers = nn.ModuleList(
            [nn.Linear(self.g_attn_dim, 1) for _ in range(self.g_num_layers)]
        )
        self.message_output_projs = nn.ModuleList(
            [nn.Linear(self.g_message_hidden_dim, self.g_hidden_dim, bias=False) for _ in range(self.g_num_layers)]
        )
        self.reasoner_gru = nn.GRU(self.g_hidden_dim, self.g_hidden_dim)
        self.reasoner_support_scorer = MLP(self.g_hidden_dim, [1], activation=g_activation, dropout=g_dropout)
        self.reasoner_dropout = nn.Dropout(g_dropout) if g_dropout > 0 else None
        self.reasoner_layer_norms = None
        if self.g_layer_norm:
            self.reasoner_layer_norms = nn.ModuleList([nn.LayerNorm(self.g_hidden_dim) for _ in range(self.g_num_layers)])
        self.kge_cache_scores = None
        self.kge_cache_row_index = None
        self.kge_cache_meta = None

        # # Initialize to 1
        # nn.init.zeros_(
        #     tensor=self.relation_embedding.weight[self.padding_index]
        # )
        
        # RNN parameters
        # self.rnn_hidden_dim = rnn_hidden_dim
        # self.num_layers = num_layers
        # self.rnn = torch.nn.LSTM(self.relation_dim + self.rule_dim, self.rnn_hidden_dim, self.num_layers, batch_first=True)
        # self.linear = torch.nn.Linear(self.rnn_hidden_dim, self.relation_dim)
        
        self.pi = 3.14159262358979323846

    # def add_param(self):

    #     # self.mlp_rule_dim = 16
    #     self.mlp_feature = nn.Parameter(torch.zeros(self.num_rules, self.mlp_rule_dim))
    #     # nn.init.kaiming_uniform_(self.mlp_feature, a=math.sqrt(5), mode="fan_in")
        
    #     # self.beta = nn.Parameter(torch.zeros((self.num_relations * 2)))
    #     # torch.nn.init.uniform_(self.beta, a=0, b=1)
        
    #     self.rule_to_entity = FuncToNodeSum(self.mlp_rule_dim)

    #     # self.relation_emb = torch.nn.Embedding(self.num_relations, self.mlp_rule_dim)
    #     self.score_model = MLP(self.mlp_rule_dim, [128, 1]) # 128 for FB15k
        
    #     # if self.device.type == "cuda":
    #     #     self.score_model = self.score_model.cuda(self.device)
    #     #     self.rule_to_entity = self.rule_to_entity.cuda(self.device)

    def set_rules(self, input):
        # input: [rule_id, rule_head, rule_body]

        logging.info('read {} rules from list.'.format(len(input)))
        self.num_rules = len(input)

        # rule_body's length
        self.max_length = max([len(rule[2:]) for rule in input])

        # self.rule_dim = self.hidden_dim * self.max_length
        self.rule_dim = self.hidden_dim 
        
        self.relation2rules = [[] for r in range(self.num_relations*2)]
        for rule in input:
            relation = rule[1]
            self.relation2rules[relation].append([rule[0], (rule[1], rule[2:])])
        

        self.rule_features = []
        rule_masks = list()
        for rule in input:
            rule_ = rule + [self.padding_index for i in range(self.max_length - len(rule[2:]))]
            self.rule_features.append(rule_)
            rule_mask = torch.ones_like(torch.tensor(rule))[2:].bool()

            # self.rule_mask = torch.zeros_like(torch.tensor(rule_))[2:].bool()
            # self.rule_mask[(len(rule[2:]))-1] = True
            rule_masks.append(rule_mask)

        # self.rule_masks = torch.stack(self.rule_masks)
        self.rule_masks = pad_sequence([_ for _ in rule_masks], batch_first=True,padding_value=False)
        self.rule_features = torch.tensor(self.rule_features, dtype=torch.long)


        self.mlp_feature = nn.Parameter(torch.zeros(self.num_rules, self.mlp_rule_dim))
        
        nn.init.kaiming_uniform_(self.mlp_feature, a=math.sqrt(5), mode="fan_in")

        self.rule_emb = torch.nn.Embedding(self.num_rules, self.rule_dim)
        nn.init.kaiming_uniform_(self.rule_emb.weight, a=math.sqrt(5), mode="fan_in")
        # nn.init.uniform_(
        #     tensor=self.rule_emb.weight, 
        #     a=-self.embedding_range_rule.item(), 
        #     b=self.embedding_range_rule.item()
        # )
        
       
    def compute_ruleE(self, sample, mode='single'):

        if mode == 'single':
            rule, mask = sample
           
            score, rule_emb = self.add_ruleE(rule.unsqueeze(1), mask)

        elif mode == 'batch':
            pos_part, mask,  neg_idx, neg_part = sample
            batch_size, negative_sample_size = neg_idx.size(0), neg_idx.size(1)
            
            pos_part = pos_part.unsqueeze(dim=1).repeat(1,negative_sample_size,1)
            
            neg_idx = neg_idx.unsqueeze(dim=2) + 1
            neg_part = neg_part.unsqueeze(dim=2)
            rule_sample = pos_part.scatter(2, neg_idx, neg_part)
           
            score, rule_emb = self.add_ruleE(rule_sample, mask)
            
        return score



    def compute_KGE(self, sample, mode='single'):
        
        if mode == 'single':

            head = self.entity_embedding(sample[:,0]).unsqueeze(1)

            relation = self.relation_embedding(sample[:,1]).unsqueeze(1)
            
            tail = self.entity_embedding(sample[:,2]).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample

            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = self.entity_embedding(head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            relation = self.relation_embedding(tail_part[:,1]).unsqueeze(1)
            tail = self.entity_embedding(tail_part[:,2]).unsqueeze(1)

        elif mode == 'tail-batch':

            head_part, tail_part = sample

            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = self.entity_embedding(head_part[:,0]).unsqueeze(1)
            relation = self.relation_embedding(head_part[:,1]).unsqueeze(1)
            tail = self.entity_embedding(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        return self.RotatE(head,relation,tail,mode), (head, tail)

    

    def compute_g_KGE(self,all_h,all_r):

        all_t = torch.arange(0,self.num_entities,device=all_h.device).unsqueeze(0).repeat(all_h.size(0),1)
       
        relations_flag = torch.pow(-1, all_r // self.num_relations).unsqueeze(-1)
        all_r = all_r % (self.num_relations)

        head = self.entity_embedding(all_h).unsqueeze(1)

        relation = (self.relation_embedding(all_r) * relations_flag).unsqueeze(1)

        tail = self.entity_embedding(all_t.view(-1)).view(all_h.size(0), self.num_entities, -1)
        

        return self.RotatE(head,relation,tail)


    def RotatE(self, head, relation, tail, mode='tail-batch'):
       
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range_fact.item()/self.pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail
      

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)
        
        score = self.gamma_fact.item() - score.sum(dim=2)
        
        return score
    


    def add_ruleE(self, rules, mask):
        inputs = rules[:,:,2:]
        # cal_mask = (~mask).unsqueeze(1).unsqueeze(-1)
        rule_len = mask.sum(-1).unsqueeze(1).unsqueeze(-1)
        cal_mask = mask.unsqueeze(1).unsqueeze(-1)
        relations_flag = torch.pow(-1, torch.div(inputs, self.num_relations, rounding_mode='floor')).unsqueeze(-1)
        inputs_com = inputs % self.num_relations

        inputs_com = torch.where(inputs==self.num_relations * 2, self.padding_index, inputs_com)
        
        embedding = self.relation_embedding(inputs_com) * relations_flag
        
        rule_embedding = self.rule_emb(rules[:,:,0])
        
        
        # rule_head
        embedding_r = self.relation_embedding(rules[:,:,1]%self.num_relations)
        relations_flag = torch.pow(-1, torch.div(rules[:,:,1], self.num_relations, rounding_mode='floor')).unsqueeze(-1)
        embedding_r *= relations_flag

        rule_body = embedding * cal_mask
        
        
        
        outputs = rule_body.sum(-2) + rule_embedding


        # dist = self.gamma_rule.item() - torch.norm((outputs - embedding_r), dim=-1)
        dist = self.gamma_rule.item() - torch.norm((outputs - embedding_r), p=self.p, dim=-1)

        
        return dist, rule_embedding
    


    def add_ruleE_g(self, rules, mask):
        inputs = rules[:,:,2:]
        # cal_mask = (~mask).unsqueeze(1).unsqueeze(-1)
        rule_len = mask.sum(-1).unsqueeze(1).unsqueeze(-1)
        cal_mask = mask.unsqueeze(1).unsqueeze(-1)
        relations_flag = torch.pow(-1, torch.div(inputs, self.num_relations, rounding_mode='floor')).unsqueeze(-1)
        inputs_com = inputs % self.num_relations

        inputs_com = torch.where(inputs==self.num_relations * 2, self.padding_index, inputs_com)
        
        embedding = self.relation_embedding(inputs_com) * relations_flag
        
        rule_embedding = self.rule_emb(rules[:,:,0])
        
        
        # rule_head
        embedding_r = self.relation_embedding(rules[:,:,1]%self.num_relations)
        relations_flag = torch.pow(-1, torch.div(rules[:,:,1], self.num_relations, rounding_mode='floor')).unsqueeze(-1)
        embedding_r *= relations_flag

        rule_body = embedding * cal_mask
        
        
        # outputs = rule_body.sum(-2) + rule_embedding
        outputs = rule_body.sum(-2) + rule_embedding

        dist = self.gamma_rule.item()/self.hidden_dim - torch.pow((outputs - embedding_r), self.p)
        # dist = self.gamma_rule.item() - torch.norm((outputs - embedding_r), p = self.p, dim=-1)

        return dist

    def get_signed_relation_embedding(self, relation_ids):
        relation_ids = relation_ids.long()
        base_ids = relation_ids % self.num_relations
        relation = self.relation_embedding(base_ids)
        sign = 1.0 - 2.0 * (relation_ids >= self.num_relations).float()
        return relation * sign.unsqueeze(-1)

    def normalize_support(self, support):
        scale = support.norm(p=2, dim=-1, keepdim=True).amax(dim=0, keepdim=True).clamp(min=1.0)
        return support / scale

    def apply_support_layer_norm(self, support, layer_norm):
        return layer_norm(support)

    def sparsemax(self, input, dim=-1):
        input = input - input.max(dim=dim, keepdim=True)[0]
        zs = torch.sort(input, dim=dim, descending=True)[0]
        steps = torch.arange(1, zs.size(dim) + 1, device=input.device, dtype=input.dtype)
        view = [1] * zs.dim()
        view[dim] = -1
        steps = steps.view(view)
        cumsum_zs = zs.cumsum(dim)
        support = (1 + steps * zs) > cumsum_zs
        k = support.sum(dim=dim, keepdim=True).clamp(min=1)
        taus = (cumsum_zs.gather(dim, k.long() - 1) - 1) / k
        return torch.clamp(input - taus, min=0.0)

    def precompute_rule_confidence(self, device):
        self.rule_masks = self.rule_masks.to(device)
        self.rule_features = self.rule_features.to(device)
        with torch.no_grad():
            score, _ = self.add_ruleE(self.rule_features.unsqueeze(1), self.rule_masks)
            self.rule_confidence = torch.sigmoid(score.squeeze(1)).detach()

    def get_kge_cache_paths(self, cache_dir):
        return {
            "score": os.path.join(cache_dir, "query_kge_scores.fp16"),
            "row_index": os.path.join(cache_dir, "query_row_index.npy"),
            "meta": os.path.join(cache_dir, "query_kge_meta.json"),
        }

    def collect_kge_cache_queries(self):
        query_keys = set()
        for facts in (self.graph.ground_train_facts, self.graph.valid_facts, self.graph.test_facts):
            for h, r, _ in facts:
                query_keys.add(self.graph.encode_hr(h, r))
        return np.asarray(sorted(query_keys), dtype=np.int64)

    def has_valid_kge_cache(self, paths, checkpoint_path):
        if not all(os.path.exists(path) for path in paths.values()):
            return False

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint_mtime = os.path.getmtime(checkpoint_path)
            cache_mtime = min(os.path.getmtime(path) for path in paths.values())
            if checkpoint_mtime > cache_mtime:
                return False

        with open(paths["meta"], "r") as fi:
            meta = json.load(fi)

        if not meta.get("complete", False):
            return False
        if meta.get("num_entities") != self.num_entities:
            return False
        if meta.get("total_relations") != self.total_relations:
            return False

        return True

    @torch.no_grad()
    def build_kge_cache(self, cache_dir, device, batch_size):
        os.makedirs(cache_dir, exist_ok=True)
        paths = self.get_kge_cache_paths(cache_dir)
        query_keys = self.collect_kge_cache_queries()
        num_queries = len(query_keys)
        batch_size = max(1, int(batch_size))

        logging.info('Building query KGE cache: %d queries x %d entities', num_queries, self.num_entities)

        row_index = np.full(self.total_relations * self.num_entities, -1, dtype=np.int32)
        row_index[query_keys] = np.arange(num_queries, dtype=np.int32)
        np.save(paths["row_index"], row_index)

        score_memmap = np.memmap(
            paths["score"],
            dtype=np.float16,
            mode="w+",
            shape=(num_queries, self.num_entities),
        )

        for start in range(0, num_queries, batch_size):
            end = min(start + batch_size, num_queries)
            hr_batch = torch.from_numpy(query_keys[start:end])
            all_h = torch.remainder(hr_batch, self.num_entities).long().to(device)
            all_r = torch.div(hr_batch, self.num_entities, rounding_mode='floor').long().to(device)

            batch_scores = self.compute_g_KGE(all_h, all_r).detach().cpu().numpy().astype(np.float16)
            score_memmap[start:end] = batch_scores

            if end % max(batch_size * 100, 1) == 0 or end == num_queries:
                logging.info('KGE cache progress: %d / %d queries', end, num_queries)

        score_memmap.flush()

        with open(paths["meta"], "w") as fo:
            json.dump(
                {
                    "num_queries": int(num_queries),
                    "num_entities": int(self.num_entities),
                    "total_relations": int(self.total_relations),
                    "dtype": "float16",
                    "complete": True,
                },
                fo,
            )

    def load_kge_cache(self, cache_dir):
        paths = self.get_kge_cache_paths(cache_dir)
        with open(paths["meta"], "r") as fi:
            meta = json.load(fi)

        self.kge_cache_row_index = np.load(paths["row_index"], mmap_mode='r')
        self.kge_cache_scores = np.memmap(
            paths["score"],
            dtype=np.float16,
            mode='r',
            shape=(meta["num_queries"], meta["num_entities"]),
        )
        self.kge_cache_meta = meta
        logging.info('Loaded query KGE cache from %s', cache_dir)

    def prepare_kge_cache(self, cache_dir, checkpoint_path, device, batch_size):
        paths = self.get_kge_cache_paths(cache_dir)
        if self.has_valid_kge_cache(paths, checkpoint_path):
            self.load_kge_cache(cache_dir)
            return

        self.build_kge_cache(cache_dir, device, batch_size)
        self.load_kge_cache(cache_dir)

    def get_query_kge_score(self, all_h, all_r):
        if self.kge_cache_scores is None or self.kge_cache_row_index is None:
            return self.compute_g_KGE(all_h, all_r)

        query_ids = (all_r.long() * self.num_entities + all_h.long()).detach().cpu().numpy()
        row_ids = self.kge_cache_row_index[query_ids]
        if (row_ids < 0).any():
            return self.compute_g_KGE(all_h, all_r)

        score = np.asarray(self.kge_cache_scores[row_ids], dtype=np.float32)
        return torch.from_numpy(score).to(all_h.device)

    def prepare_reasoner(self, device, cache_dir=None, checkpoint_path=None, kge_batch_size=1):
        self.graph.cache_adjacency(device)
        if self.reasoner_type == 'grounding':
            self.eval_compute_rule_weight(device)
        else:
            self.precompute_rule_confidence(device)
            if cache_dir is not None:
                self.prepare_kge_cache(cache_dir, checkpoint_path, device, kge_batch_size)

    def build_query_rule_states(self, query_r, query_emb, head_context, layer_id, device):
        relation_ids = torch.arange(self.total_relations, device=device)
        base_relation_repr = self.get_signed_relation_embedding(relation_ids)
        batch_size = query_emb.size(0)
        rule_states = base_relation_repr.unsqueeze(1).repeat(1, batch_size, 1)
        query_rules = self.relation2rules[query_r]

        if len(query_rules) == 0:
            return rule_states

        rule_ids = torch.tensor([index for index, _ in query_rules], dtype=torch.long, device=device)
        if self.rule_confidence is None or self.rule_confidence.device != device:
            self.precompute_rule_confidence(device)

        confidence = self.rule_confidence[rule_ids].unsqueeze(0).expand(batch_size, -1)
        rule_emb = self.rule_emb(rule_ids)
        projected_rule_emb = self.rule_memory_proj(self.rule_emb(rule_ids))

        compat = self.rule_score_layers[layer_id](
            self.g_activation(
                self.rule_score_rule_projs[layer_id](rule_emb).unsqueeze(0)
                + self.rule_score_query_projs[layer_id](query_emb).unsqueeze(1)
                + self.rule_score_head_projs[layer_id](head_context).unsqueeze(1)
            )
        ).squeeze(-1)
        rule_weight = self.sparsemax(compat + self.rule_confidence_scale * confidence, dim=-1)

        for rule_offset, (rule_feature, (_, (rule_head, rule_body))) in enumerate(zip(projected_rule_emb, query_rules)):
            del rule_head
            if layer_id >= len(rule_body):
                continue
            relation_id = rule_body[layer_id]
            rule_states[relation_id] = rule_states[relation_id] + rule_weight[:, rule_offset].unsqueeze(-1) * rule_feature.unsqueeze(0)

        return rule_states

    def init_query_hidden(self, all_h, all_r):
        device = all_h.device
        batch_size = all_h.size(0)
        query_emb = self.get_signed_relation_embedding(all_r)
        seed_hidden = self.query_seed_proj(query_emb)
        head_context = self.head_context_proj(self.entity_embedding(all_h))
        hidden = torch.zeros(self.num_entities, batch_size, self.g_hidden_dim, device=device)
        batch_index = torch.arange(batch_size, device=device)
        hidden[all_h, batch_index] = seed_hidden
        return hidden, query_emb, head_context

    def get_relation_edges(self, relation_id, device):
        if self.graph.cached_adjacency is not None and self.graph.cached_adjacency_device == device:
            return self.graph.cached_adjacency[relation_id]

        adjacency = self.graph.relation2adjacency[relation_id][0]
        node_in = adjacency[1]
        node_out = adjacency[0]
        if device.type == "cuda":
            node_in = node_in.cuda(device)
            node_out = node_out.cuda(device)
        return node_in, node_out

    def apply_edge_removal_mask(self, edge_message, relation_id, query_r, edges_to_remove):
        if relation_id != query_r or edges_to_remove is None:
            return edge_message

        if edges_to_remove.dim() == 0:
            edges_to_remove = edges_to_remove.unsqueeze(0)

        masked_message = edge_message
        batch_index = torch.arange(masked_message.size(1), device=masked_message.device)
        valid = (edges_to_remove >= 0) & (edges_to_remove < masked_message.size(0))
        if valid.any():
            masked_message[edges_to_remove[valid], batch_index[valid]] = 0
        return masked_message

    def propagate_single_path_layer(self, hidden, query_emb, relation_states, layer_id, query_r, edges_to_remove):
        device = hidden.device
        batch_size = hidden.size(1)
        aggregated = torch.zeros(self.num_entities, batch_size, self.g_message_hidden_dim, device=device)

        message_node_proj = self.message_node_projs[layer_id]
        message_relation_proj = self.message_relation_projs[layer_id]
        attn_source_proj = self.attn_source_projs[layer_id]
        attn_relation_proj = self.attn_relation_projs[layer_id]
        attn_query = self.attn_query_projs[layer_id](query_emb).unsqueeze(0)
        attn_score_layer = self.attn_score_layers[layer_id]

        for relation_id in range(self.total_relations):
            node_in, node_out = self.get_relation_edges(relation_id, device)
            if node_in.numel() == 0:
                continue

            source_hidden = hidden[node_in]
            relation_state = relation_states[relation_id]

            edge_message = message_node_proj(source_hidden)
            edge_message = edge_message * message_relation_proj(relation_state).unsqueeze(0)

            attn_input = (
                attn_source_proj(source_hidden)
                + attn_relation_proj(relation_state).unsqueeze(0)
                + attn_query
            )
            edge_alpha = torch.sigmoid(attn_score_layer(self.g_activation(attn_input)))
            edge_message = edge_alpha * edge_message
            edge_message = self.apply_edge_removal_mask(edge_message, relation_id, query_r, edges_to_remove)

            aggregated = aggregated + scatter(edge_message, node_out, dim=0, dim_size=self.num_entities, reduce='sum')

        active_mask = (aggregated.abs().sum(dim=-1, keepdim=True) > 0).float()

        updated = self.message_output_projs[layer_id](aggregated)
        updated = self.g_activation(updated)
        if self.reasoner_dropout is not None:
            updated = self.reasoner_dropout(updated)

        updated_flat = updated.reshape(1, -1, self.g_hidden_dim)
        hidden_flat = hidden.reshape(1, -1, self.g_hidden_dim)
        updated_hidden, _ = self.reasoner_gru(updated_flat, hidden_flat)
        updated_hidden = updated_hidden.reshape(self.num_entities, batch_size, self.g_hidden_dim)
        updated_hidden = updated_hidden * active_mask

        if self.reasoner_layer_norms is not None:
            updated_hidden = self.apply_support_layer_norm(updated_hidden, self.reasoner_layer_norms[layer_id])

        return self.normalize_support(updated_hidden)

    def forward_dual_pathway(self, all_h, all_r, edges_to_remove):
        if (all_r != all_r[0]).any():
            raise ValueError('dual_pathway expects one query relation per batch')

        query_r = all_r[0].item()
        device = all_r.device
        batch_size = all_h.size(0)

        hidden, query_emb, head_context = self.init_query_hidden(all_h, all_r)

        for layer_id in range(self.g_num_layers):
            query_rule_states = self.build_query_rule_states(query_r, query_emb, head_context, layer_id, device)
            hidden = self.propagate_single_path_layer(
                hidden,
                query_emb,
                query_rule_states,
                layer_id,
                query_r,
                edges_to_remove,
            )

        hidden_batch = hidden.permute(1, 0, 2)
        if self.g_readout == 'multiply':
            batch_index = torch.arange(batch_size, device=device)
            anchor_hidden = hidden[all_h, batch_index]
            support_score = (hidden_batch * anchor_hidden.unsqueeze(1)).sum(dim=-1)
        else:
            support_score = self.reasoner_support_scorer(hidden_batch).squeeze(-1)
        kge_score = self.get_query_kge_score(all_h, all_r)
        score = support_score + self.reasoner_alpha * kge_score + self.bias.unsqueeze(0)
        mask = torch.ones(batch_size, self.graph.entity_size, device=device).bool()

        return score, mask
    

    def forward(self, all_h, all_r, edges_to_remove):
        if self.reasoner_type == 'grounding':
            return self.forward_grounding(all_h, all_r, edges_to_remove)

        return self.forward_dual_pathway(all_h, all_r, edges_to_remove)

    def forward_grounding(self, all_h, all_r, edges_to_remove):
        query_r = all_r[0].item()
        assert (all_r != query_r).sum() == 0
        device = all_r.device

        if device.type == "cuda":
            self.rule_features = self.rule_features.cuda(device)

        rule_index = list()
        rule_count = list()
        
        
        mask = torch.zeros(all_h.size(0), self.graph.entity_size, device=device)
        for index, (r_head, r_body) in self.relation2rules[query_r]:

            assert r_head == query_r

            count = self.graph.grounding(all_h, r_head, r_body, edges_to_remove).float()
            
            mask += count

            rule_index.append(index)
            rule_count.append(count)


        if mask.sum().item() == 0:
            # return mask + self.bias.unsqueeze(0), (1 - mask).bool(), torch.zeros_like(rule_loss)
            return mask + self.bias.unsqueeze(0), (1 - mask).bool()


        candidate_set = torch.nonzero(mask.view(-1), as_tuple=True)[0]

        rule_index = torch.tensor(rule_index, dtype=torch.long, device=device)
        rule_count = torch.stack(rule_count, dim=0)

        rule_count = rule_count.reshape(rule_index.size(0), -1)[:, candidate_set]
        
        rule_emb = self.rules_weight_emb[rule_index]

        # mlp_feature = self.mlp_feature[rule_index] * rule_emb.unsqueeze(-1)
        mlp_feature = self.mlp_feature[rule_index]

        # output = self.rule_to_entity(rule_count, mlp_feature)
        output = self.rule_to_entity(rule_count, rule_emb, mlp_feature)


        # rel = self.relation_embedding(all_r[0]%self.num_relations)
        # relations_flag = torch.pow(-1,all_r[0] // (self.num_relations)).unsqueeze(-1)
        # rel = (rel * relations_flag).unsqueeze(0).expand(output.size(0), -1)

        # feature = torch.cat([output, rel], dim=-1)
        feature = output

        output = self.score_model(feature).squeeze(-1)

        score = torch.zeros(all_h.size(0) * self.graph.entity_size, device=device)
        score.scatter_(0, candidate_set, output)
        score = score.view(all_h.size(0), self.graph.entity_size)
        score = score + self.bias.unsqueeze(0)
        # kge_score = self.compute_g_KGE(all_h, all_r)
        # kge_score_map = self.map(score, kge_score)
        
        # beta = torch.sigmoid(self.beta[all_r[0]])
        # score = score + self.bias.unsqueeze(0)
        # betax = self.beta[all_r[0]][0]
        # betay = self.beta[all_r[0]][1]
        # beta = self.beta[all_r[0]]
        # score = beta * score + (1 - beta) * kge_score_map
        # score = self.beta[all_r[0]] * score +  kge_score

        mask = torch.ones_like(mask).bool()

        return score, mask




    def eval_compute_rule_weight(self,device):
        '''
        During grounding process, we one time compute the rule score on rule embedding and relation embedding
        '''
        batch = 128
        self.rule_masks = self.rule_masks.to(device)
        self.rule_features = self.rule_features.to(device)
        split_num = self.rule_features.size(0) // batch 
        rule_batches = torch.split(self.rule_features, split_num, 0)
        rule_mask_batches = torch.split(self.rule_masks, split_num, 0)
        rules_weight_emb = list()

        for rules, rules_mask in zip(rule_batches, rule_mask_batches):

            rule_weight_emb = self.add_ruleE_g(rules.unsqueeze(1),rules_mask).squeeze(1)
            rules_weight_emb.append(rule_weight_emb)

        self.rules_weight_emb = torch.cat(rules_weight_emb)
