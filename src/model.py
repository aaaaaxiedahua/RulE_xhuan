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
        reasoner_type='gnn',
        g_num_layers=2,
        g_hidden_dim=128,
        g_dropout=0.1,
        g_activation='relu',
        rule_tf_layers=1,
        rule_num_heads=4,
        rule_dropout=0.1,
        rule_ffn_dim=512,
        conve_num_filters=32,
        conve_kernel_size=3,
        conve_dropout=0.2,
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
        self.g_activation = getattr(F, g_activation)
        self.rule_tf_layers = int(rule_tf_layers)
        self.rule_num_heads = int(rule_num_heads)
        self.rule_dropout = float(rule_dropout)
        self.rule_ffn_dim = int(rule_ffn_dim)
        self.conve_num_filters = int(conve_num_filters)
        self.conve_kernel_size = int(conve_kernel_size)
        self.conve_dropout = float(conve_dropout)
        if self.g_hidden_dim % self.rule_num_heads != 0:
            raise ValueError('g_hidden_dim must be divisible by rule_num_heads')
        self.gnn_cache_device = None
        self.cached_edge_src = None
        self.cached_edge_dst = None
        self.cached_edge_rel = None
        self.cached_cluster_incidence = None
        self.cached_cluster_counts = None
        self.cached_entity_cluster_counts = None

        self.entity_init_proj = nn.Linear(self.hidden_dim * 2, self.g_hidden_dim, bias=False)
        self.relation_init_proj = nn.Linear(self.hidden_dim, self.g_hidden_dim, bias=False)
        self.rule_input_proj = nn.Linear(self.hidden_dim, self.g_hidden_dim, bias=False)
        self.rule_context_proj = nn.Linear(self.g_hidden_dim, self.g_hidden_dim, bias=False)
        self.local_entity_message_projs = nn.ModuleList(
            [nn.Linear(self.g_hidden_dim, self.g_hidden_dim, bias=False) for _ in range(self.g_num_layers)]
        )
        self.local_relation_message_projs = nn.ModuleList(
            [nn.Linear(self.g_hidden_dim, self.g_hidden_dim, bias=False) for _ in range(self.g_num_layers)]
        )
        self.local_triple_message_projs = nn.ModuleList(
            [nn.Linear(self.g_hidden_dim, self.g_hidden_dim, bias=False) for _ in range(self.g_num_layers)]
        )
        self.cluster_proj = nn.Linear(self.g_hidden_dim, self.g_hidden_dim, bias=False)
        self.global_entity_proj = nn.Linear(self.g_hidden_dim, self.g_hidden_dim, bias=False)
        self.reasoner_dropout = nn.Dropout(g_dropout) if g_dropout > 0 else None
        self.kge_cache_scores = None
        self.kge_cache_row_index = None
        self.kge_cache_meta = None
        self.rule_position_embedding = None
        self.rule_transformer = None
        self.rule_cross_attn = None

        self.conve_emb_h, self.conve_emb_w = self.get_embedding_grid_shape(self.g_hidden_dim)
        conv_h = self.conve_emb_h * 2 - self.conve_kernel_size + 1
        conv_w = self.conve_emb_w - self.conve_kernel_size + 1
        if conv_h <= 0 or conv_w <= 0:
            raise ValueError('conve_kernel_size is too large for g_hidden_dim')
        self.conve_conv = nn.Conv2d(
            1,
            self.conve_num_filters,
            kernel_size=(self.conve_kernel_size, self.conve_kernel_size),
            bias=True,
        )
        self.conve_fc = nn.Linear(self.conve_num_filters * conv_h * conv_w, self.g_hidden_dim)
        self.conve_input_dropout = nn.Dropout(self.conve_dropout)
        self.conve_feature_dropout = nn.Dropout2d(self.conve_dropout)
        self.conve_hidden_dropout = nn.Dropout(self.conve_dropout)

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

    def get_embedding_grid_shape(self, dim):
        height = int(math.sqrt(dim))
        while height > 1 and dim % height != 0:
            height -= 1
        width = dim // height
        return height, width

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
        self.rule_position_embedding = nn.Embedding(self.max_length, self.g_hidden_dim)
        transformer_activation = 'gelu' if self.g_activation == F.gelu else 'relu'
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.g_hidden_dim,
            nhead=self.rule_num_heads,
            dim_feedforward=self.rule_ffn_dim,
            dropout=self.rule_dropout,
            activation=transformer_activation,
            batch_first=True,
        )
        self.rule_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.rule_tf_layers)
        self.rule_cross_attn = nn.MultiheadAttention(
            embed_dim=self.g_hidden_dim,
            num_heads=self.rule_num_heads,
            dropout=self.rule_dropout,
            batch_first=True,
        )
        
       
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

    def encode_all_rules(self, device):
        if self.rule_transformer is None:
            raise RuntimeError('set_rules must be called before encoding rules')
        if self.rule_features.device != device:
            self.rule_features = self.rule_features.to(device)
        if self.rule_masks.device != device:
            self.rule_masks = self.rule_masks.to(device)

        rule_bodies = self.rule_features[:, 2:]
        rule_mask = self.rule_masks
        safe_rule_bodies = torch.where(rule_mask, rule_bodies, torch.zeros_like(rule_bodies))
        rule_body_emb = self.get_signed_relation_embedding(safe_rule_bodies) * rule_mask.unsqueeze(-1).float()

        positions = torch.arange(self.max_length, device=device)
        position_emb = self.rule_position_embedding(positions).unsqueeze(0)
        rule_inputs = self.rule_input_proj(rule_body_emb) + position_emb
        encoded_rules = self.rule_transformer(rule_inputs, src_key_padding_mask=~rule_mask)
        return encoded_rules, rule_mask

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
            self.cache_gnn_structure(device)
            if cache_dir is not None:
                self.prepare_kge_cache(cache_dir, checkpoint_path, device, kge_batch_size)

    def scatter_softmax(self, score, index, dim_size):
        max_score = scatter(score, index, dim=0, dim_size=dim_size, reduce='max')
        stabilized = torch.exp(score - max_score[index])
        denom = scatter(stabilized, index, dim=0, dim_size=dim_size, reduce='sum')
        return stabilized / denom[index].clamp(min=1e-12)

    def cache_gnn_structure(self, device):
        if self.gnn_cache_device == device and self.cached_edge_src is not None:
            return

        edge_src = []
        edge_dst = []
        edge_rel = []
        cluster_rows = []
        cluster_cols = []
        for relation_id in range(self.total_relations):
            adjacency = self.graph.relation2adjacency[relation_id][0]
            src = adjacency[1].long()
            dst = adjacency[0].long()
            if src.numel() == 0:
                continue
            edge_src.append(src)
            edge_dst.append(dst)
            edge_rel.append(torch.full((src.numel(),), relation_id, dtype=torch.long))

            unique_src = torch.unique(src)
            unique_dst = torch.unique(dst)
            cluster_rows.append(unique_src)
            cluster_cols.append(torch.full((unique_src.numel(),), relation_id, dtype=torch.long))
            cluster_rows.append(unique_dst)
            cluster_cols.append(torch.full((unique_dst.numel(),), relation_id + self.total_relations, dtype=torch.long))

        if edge_src:
            edge_src = torch.cat(edge_src).to(device)
            edge_dst = torch.cat(edge_dst).to(device)
            edge_rel = torch.cat(edge_rel).to(device)
        else:
            edge_src = torch.empty(0, dtype=torch.long, device=device)
            edge_dst = torch.empty(0, dtype=torch.long, device=device)
            edge_rel = torch.empty(0, dtype=torch.long, device=device)

        if cluster_rows:
            cluster_rows = torch.cat(cluster_rows)
            cluster_cols = torch.cat(cluster_cols)
            cluster_indices = torch.stack([cluster_rows, cluster_cols], dim=0).to(device)
            cluster_values = torch.ones(cluster_indices.size(1), device=device)
        else:
            cluster_indices = torch.empty((2, 0), dtype=torch.long, device=device)
            cluster_values = torch.empty(0, device=device)

        cluster_incidence = torch.sparse_coo_tensor(
            cluster_indices,
            cluster_values,
            size=(self.num_entities, self.total_relations * 2),
            device=device,
        ).coalesce()

        self.cached_edge_src = edge_src
        self.cached_edge_dst = edge_dst
        self.cached_edge_rel = edge_rel
        self.cached_cluster_incidence = cluster_incidence
        self.cached_cluster_counts = torch.sparse.sum(cluster_incidence, dim=0).to_dense().clamp(min=1.0).unsqueeze(-1)
        self.cached_entity_cluster_counts = torch.sparse.sum(cluster_incidence, dim=1).to_dense().clamp(min=1.0).unsqueeze(-1)
        self.gnn_cache_device = device

    def build_rule_enhanced_relations(self, device):
        relation_ids = torch.arange(self.total_relations, device=device)
        base_relation = self.relation_init_proj(self.get_signed_relation_embedding(relation_ids))
        if self.num_rules == 0:
            return base_relation

        encoded_rules, rule_mask = self.encode_all_rules(device)
        enhanced_relation = base_relation.clone()
        scale = math.sqrt(float(self.g_hidden_dim))

        for relation_id, relation_rules in enumerate(self.relation2rules):
            if len(relation_rules) == 0:
                continue
            rule_ids = torch.tensor([index for index, _ in relation_rules], dtype=torch.long, device=device)
            selected_rule_tokens = encoded_rules[rule_ids]
            selected_rule_mask = rule_mask[rule_ids]
            relation_query = base_relation[relation_id].view(1, 1, -1).expand(rule_ids.size(0), 1, self.g_hidden_dim)
            rule_summary, _ = self.rule_cross_attn(
                relation_query,
                selected_rule_tokens,
                selected_rule_tokens,
                key_padding_mask=~selected_rule_mask,
            )
            rule_summary = rule_summary.squeeze(1)
            compat = (rule_summary * base_relation[relation_id].unsqueeze(0)).sum(dim=-1) / scale
            rule_weight = torch.softmax(compat, dim=0)
            relation_context = torch.sum(rule_weight.unsqueeze(-1) * rule_summary, dim=0)
            enhanced_relation[relation_id] = self.g_activation(
                base_relation[relation_id] + self.rule_context_proj(relation_context)
            )

        return enhanced_relation

    def compute_local_context(self, entity_state, relation_state, layer_id):
        src = self.cached_edge_src
        dst = self.cached_edge_dst
        rel = self.cached_edge_rel
        if src.numel() == 0:
            return entity_state

        source_hidden = entity_state[src]
        target_hidden = entity_state[dst]
        relation_hidden = relation_state[rel]

        entity_message = self.local_entity_message_projs[layer_id](source_hidden)
        relation_message = self.local_relation_message_projs[layer_id](relation_hidden)
        triple_message = self.local_triple_message_projs[layer_id](source_hidden * relation_hidden)

        entity_alpha = self.scatter_softmax(
            (entity_message * target_hidden).sum(dim=-1, keepdim=True),
            dst,
            self.num_entities,
        )
        relation_alpha = self.scatter_softmax(
            (relation_message * target_hidden).sum(dim=-1, keepdim=True),
            dst,
            self.num_entities,
        )
        triple_alpha = self.scatter_softmax(
            (triple_message * target_hidden).sum(dim=-1, keepdim=True),
            dst,
            self.num_entities,
        )

        entity_context = scatter(entity_alpha * entity_message, dst, dim=0, dim_size=self.num_entities, reduce='sum')
        relation_context = scatter(relation_alpha * relation_message, dst, dim=0, dim_size=self.num_entities, reduce='sum')
        triple_context = scatter(triple_alpha * triple_message, dst, dim=0, dim_size=self.num_entities, reduce='sum')

        updated = entity_state + entity_context + relation_context + triple_context
        updated = self.g_activation(updated)
        if self.reasoner_dropout is not None:
            updated = self.reasoner_dropout(updated)
        return self.normalize_support(updated)

    def compute_global_cluster_context(self, entity_state):
        cluster_hidden = torch.sparse.mm(self.cached_cluster_incidence.transpose(0, 1), entity_state)
        cluster_hidden = cluster_hidden / self.cached_cluster_counts
        cluster_hidden = self.g_activation(self.cluster_proj(cluster_hidden))

        global_entity = torch.sparse.mm(self.cached_cluster_incidence, cluster_hidden)
        global_entity = global_entity / self.cached_entity_cluster_counts
        global_entity = self.g_activation(self.global_entity_proj(global_entity))
        if self.reasoner_dropout is not None:
            global_entity = self.reasoner_dropout(global_entity)
        return global_entity

    def encode_gnn_graph(self, device):
        self.cache_gnn_structure(device)
        relation_state = self.build_rule_enhanced_relations(device)
        entity_state = self.entity_init_proj(self.entity_embedding.weight)

        for layer_id in range(self.g_num_layers):
            entity_state = self.compute_local_context(entity_state, relation_state, layer_id)

        global_entity = self.compute_global_cluster_context(entity_state)
        entity_state = self.normalize_support(entity_state + global_entity)
        return entity_state, relation_state

    def conve_score(self, head_hidden, relation_hidden, tail_hidden):
        batch_size = head_hidden.size(0)
        head_2d = head_hidden.view(batch_size, 1, self.conve_emb_h, self.conve_emb_w)
        relation_2d = relation_hidden.view(batch_size, 1, self.conve_emb_h, self.conve_emb_w)
        stacked = torch.cat([head_2d, relation_2d], dim=2)
        stacked = self.conve_input_dropout(stacked)
        hidden = self.conve_conv(stacked)
        hidden = F.relu(hidden)
        hidden = self.conve_feature_dropout(hidden)
        hidden = hidden.view(batch_size, -1)
        hidden = self.conve_fc(hidden)
        hidden = self.g_activation(hidden)
        hidden = self.conve_hidden_dropout(hidden)
        return torch.matmul(hidden, tail_hidden.transpose(0, 1)) + self.bias.unsqueeze(0)

    def forward_gnn(self, all_h, all_r, edges_to_remove):
        del edges_to_remove
        device = all_h.device
        entity_state, relation_state = self.encode_gnn_graph(device)
        head_hidden = entity_state[all_h]
        relation_hidden = relation_state[all_r]
        score = self.conve_score(head_hidden, relation_hidden, entity_state)
        mask = torch.ones(all_h.size(0), self.graph.entity_size, device=device).bool()
        return score, mask
    

    def forward(self, all_h, all_r, edges_to_remove):
        if self.reasoner_type == 'grounding':
            return self.forward_grounding(all_h, all_r, edges_to_remove)

        return self.forward_gnn(all_h, all_r, edges_to_remove)

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
