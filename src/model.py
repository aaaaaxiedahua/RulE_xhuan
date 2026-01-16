
import torch
import torch.nn as nn
import logging, math
from layers import MLP, FuncToNodeSum

from torch.nn.utils.rnn import pad_sequence

class RulE(torch.nn.Module):
    def __init__(self, graph, p_norm, mlp_rule_dim, gamma_fact, gamma_rule, hidden_dim, device, dataset):
        super(RulE, self).__init__()
        self.graph = graph
        self.device = device
        self.num_entities = graph.entity_size
        self.num_relations = graph.relation_size 
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

        self.use_adapter = False
        self.use_logic_attention = False
        self.attn_dim = 64
        self.adapter_hidden_dim = 1024
        self.adapter_dropout_p = 0.0
        self.attn_dropout_p = 0.0
        self._build_context_modules()

        # TAPC-RulE相关参数
        self.use_tapc = False
        self.critic = None
        self.type_aware_emb = None
        self.entity_to_type = None

    def _build_context_modules(self):
        entity_dim = self.hidden_dim * 2
        self.adapter_mlp = nn.Sequential(
            nn.Linear(entity_dim, self.adapter_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.adapter_dropout_p),
            nn.Linear(self.adapter_hidden_dim, entity_dim),
            nn.Tanh(),
        )

        self.query_proj = nn.Linear(entity_dim, self.attn_dim, bias=False)
        self.rule_key_proj = nn.Linear(self.hidden_dim, self.attn_dim, bias=False)
        self.attn_v = nn.Linear(self.attn_dim, 1, bias=False)

        self.query_dropout = nn.Dropout(self.attn_dropout_p)
        self.attn_dropout = nn.Dropout(self.attn_dropout_p)
        self.gate_scale = math.sqrt(self.attn_dim)
        self._last_context_log = None

    def configure_context_modules(
        self,
        use_adapter=False,
        use_logic_attention=False,
        attn_dim=64,
        adapter_hidden_dim=1024,
        adapter_dropout=0.0,
        attn_dropout=0.0,
    ):
        attn_dim = int(attn_dim)
        adapter_hidden_dim = int(adapter_hidden_dim)
        adapter_dropout = float(adapter_dropout)
        attn_dropout = float(attn_dropout)

        rebuild = (
            attn_dim != self.attn_dim
            or adapter_hidden_dim != self.adapter_hidden_dim
            or adapter_dropout != self.adapter_dropout_p
            or attn_dropout != self.attn_dropout_p
        )

        self.use_adapter = bool(use_adapter)
        self.use_logic_attention = bool(use_logic_attention)
        self.attn_dim = attn_dim
        self.adapter_hidden_dim = adapter_hidden_dim
        self.adapter_dropout_p = adapter_dropout
        self.attn_dropout_p = attn_dropout

        if rebuild:
            self._build_context_modules()

    def configure_tapc(self, use_tapc=False, critic=None, type_aware_emb=None, entity_to_type=None):
        """
        配置TAPC-RulE模块

        参数:
            use_tapc: 是否使用TAPC
            critic: PathCritic实例
            type_aware_emb: TypeAwareEmbedding实例
            entity_to_type: 实体到类型的映射
        """
        self.use_tapc = bool(use_tapc)
        self.critic = critic
        self.type_aware_emb = type_aware_emb
        self.entity_to_type = entity_to_type

        if self.use_tapc:
            logging.info("TAPC-RulE已启用")

    def _apply_critic_filter(self, paths, r_head, r_body):
        """
        使用Critic对路径进行过滤和打分

        参数:
            paths: List[Tuple] 路径列表
            r_head: 目标关系
            r_body: 规则体

        返回:
            filtered_count: 过滤后的计数矩阵
        """
        if not self.use_tapc or self.critic is None or len(paths) == 0:
            return None

        from tapc_critic import construct_path_sequence

        # 构造路径序列
        batch_sequences = []
        for path in paths:
            sequence = construct_path_sequence(
                path,
                self.entity_embedding,
                self.relation_embedding,
                self.type_aware_emb,
                self.entity_to_type,
                self.device
            )
            batch_sequences.append(sequence)

        # 堆叠成batch
        batch_sequences = torch.stack(batch_sequences, dim=0)

        # Critic打分
        with torch.no_grad():
            scores = self.critic(batch_sequences)  # [num_paths, 1]

        return scores.squeeze(-1)  # [num_paths]

    def _compute_rule_logits(self, all_h, rule_emb):
        entity = self.entity_embedding(all_h)
        if self.use_adapter:
            entity = entity + self.adapter_mlp(entity)

        query = self.query_dropout(self.query_proj(entity))
        key = self.rule_key_proj(rule_emb)

        if self.use_logic_attention:
            logits = self.attn_v(torch.tanh(query.unsqueeze(1) + key.unsqueeze(0))).squeeze(-1)
        else:
            logits = torch.matmul(query, key.t()) / self.gate_scale

        logits = self.attn_dropout(logits)
        return logits

    def _compute_rule_gate(self, all_h, rule_emb):
        logits = self._compute_rule_logits(all_h, rule_emb)
        return 2.0 * torch.sigmoid(logits)

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
        relations_flag = torch.pow(-1,inputs // (self.num_relations)).unsqueeze(-1)
        inputs_com = inputs % self.num_relations

        inputs_com = torch.where(inputs==self.num_relations * 2, self.padding_index, inputs_com)
        
        embedding = self.relation_embedding(inputs_com) * relations_flag
        
        rule_embedding = self.rule_emb(rules[:,:,0])
        
        
        # rule_head
        embedding_r = self.relation_embedding(rules[:,:,1]%self.num_relations)
        relations_flag = torch.pow(-1,rules[:,:,1] // (self.num_relations)).unsqueeze(-1)
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
        relations_flag = torch.pow(-1,inputs // (self.num_relations)).unsqueeze(-1)
        inputs_com = inputs % self.num_relations

        inputs_com = torch.where(inputs==self.num_relations * 2, self.padding_index, inputs_com)
        
        embedding = self.relation_embedding(inputs_com) * relations_flag
        
        rule_embedding = self.rule_emb(rules[:,:,0])
        
        
        # rule_head
        embedding_r = self.relation_embedding(rules[:,:,1]%self.num_relations)
        relations_flag = torch.pow(-1,rules[:,:,1] // (self.num_relations)).unsqueeze(-1)
        embedding_r *= relations_flag

        rule_body = embedding * cal_mask
        
        
        # outputs = rule_body.sum(-2) + rule_embedding
        outputs = rule_body.sum(-2) + rule_embedding

        dist = self.gamma_rule.item()/self.hidden_dim - torch.pow((outputs - embedding_r), self.p)
        # dist = self.gamma_rule.item() - torch.norm((outputs - embedding_r), p = self.p, dim=-1)

        return dist
    

    def forward(self, all_h, all_r, edges_to_remove):
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

            # TAPC-RulE: 使用grounding_with_paths获取路径
            if self.use_tapc and hasattr(self.graph, 'grounding_with_paths'):
                paths, count = self.graph.grounding_with_paths(all_h, r_head, r_body, edges_to_remove)
                count = count.float()

                # 应用Critic过滤
                if len(paths) > 0:
                    path_scores = self._apply_critic_filter(paths, r_head, r_body)
                    if path_scores is not None:
                        # 根据Critic分数调整count
                        # 这里简化处理：将路径分数累加到对应的尾实体上
                        for path, score in zip(paths, path_scores):
                            if len(path) == 5:  # 2-hop: (e0, r1, e1, r2, e2)
                                h_idx = path[0]
                                t_idx = path[4]
                            elif len(path) == 7:  # 3-hop: (e0, r1, e1, r2, e2, r3, e3)
                                h_idx = path[0]
                                t_idx = path[6]
                            else:
                                continue

                            # 找到对应的batch索引
                            batch_idx = (all_h == h_idx).nonzero(as_tuple=True)[0]
                            if len(batch_idx) > 0:
                                count[batch_idx[0], t_idx] *= score.item()
            else:
                # 原始grounding方法
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

        if self.use_logic_attention:
            batch_id = candidate_set // self.graph.entity_size
            logits = self._compute_rule_logits(all_h, rule_emb)  # [B, Nr]
            logits_candidate = logits[batch_id]  # [C, Nr]

            active_mask = (rule_count > 0).transpose(0, 1)  # [C, Nr]
            logits_candidate = logits_candidate.masked_fill(~active_mask, -1e9)

            attn = torch.softmax(logits_candidate, dim=1)  # [C, Nr]
            active_cnt = active_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            weights_candidate = attn * active_cnt
            rule_count = rule_count * weights_candidate.transpose(0, 1)

            with torch.no_grad():
                p = attn.detach()
                stats = {
                    "mode": "masked_softmax_scaled",
                    "batch": int(all_h.size(0)),
                    "nrules": int(rule_index.size(0)),
                    "candidates": int(candidate_set.size(0)),
                    "active_rules_mean": float(active_cnt.mean().item()),
                    "mean": float(weights_candidate.detach().mean().item()),
                    "min": float(weights_candidate.detach().min().item()),
                    "max": float(weights_candidate.detach().max().item()),
                }
                eps = 1e-12
                ent = -(p * (p + eps).log()).sum(dim=1)
                stats["entropy_mean"] = float(ent.mean().item())
                k = min(5, int(weights_candidate.size(1)))
                if k > 0:
                    top_val, top_idx = torch.topk(weights_candidate[0].detach(), k=k, dim=0)
                    stats["query_r"] = int(query_r)
                    stats["top_rule_id"] = rule_index[top_idx].detach().cpu().tolist()
                    stats["top_weight"] = [float(v) for v in top_val.detach().cpu().tolist()]
                self._last_context_log = stats

        elif self.use_adapter:
            batch_id = candidate_set // self.graph.entity_size
            gate = self._compute_rule_gate(all_h, rule_emb)  # [B, Nr]
            weights_candidate = gate[batch_id]  # [C, Nr]
            rule_count = rule_count * weights_candidate.transpose(0, 1)

            with torch.no_grad():
                w = weights_candidate.detach()
                stats = {
                    "mode": "sigmoid2_gate",
                    "batch": int(all_h.size(0)),
                    "nrules": int(rule_index.size(0)),
                    "candidates": int(candidate_set.size(0)),
                    "mean": float(w.mean().item()),
                    "min": float(w.min().item()),
                    "max": float(w.max().item()),
                    "query_r": int(query_r),
                }
                k = min(5, int(w.size(1)))
                if k > 0:
                    top_val, top_idx = torch.topk(w[0], k=k, dim=0)
                    stats["top_rule_id"] = rule_index[top_idx].detach().cpu().tolist()
                    stats["top_weight"] = [float(v) for v in top_val.detach().cpu().tolist()]
                self._last_context_log = stats

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
