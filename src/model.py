
import torch
import torch.nn as nn
import logging, math
from layers import MLP, FuncToNodeSum

from torch.nn.utils.rnn import pad_sequence


class QueryConditionedAttention(nn.Module):
    """
    方案一：Query-Conditioned Rule Attention
    根据查询(h, r)动态计算每条规则的重要性权重
    """
    def __init__(self, hidden_dim, attention_hidden_dim=64, dropout=0.1):
        super(QueryConditionedAttention, self).__init__()

        # 查询编码器：将[h_emb, r_emb]编码为query_vector
        # 输入: hidden_dim*2 (实体) + hidden_dim (关系) = hidden_dim*3
        self.query_encoder = MLP(
            input_dim=hidden_dim * 3,
            hidden_dims=[attention_hidden_dim],
            dropout=dropout
        )

        # 注意力网络：计算query和rule的匹配分数
        # 输入: attention_hidden_dim (query) + hidden_dim (rule)
        self.attention_net = MLP(
            input_dim=attention_hidden_dim + hidden_dim,
            hidden_dims=[attention_hidden_dim, 1],
            dropout=dropout
        )

    def forward(self, h_emb, r_emb, rule_embs):
        """
        计算attention权重

        Args:
            h_emb: 头实体embedding [batch, hidden_dim*2]
            r_emb: 关系embedding [batch, hidden_dim]
            rule_embs: 规则embeddings [num_rules, hidden_dim]

        Returns:
            attention_scores: [batch, num_rules, 1] 范围在(0, 1)之间
        """
        batch_size = h_emb.size(0)
        num_rules = rule_embs.size(0)

        # 1. 编码查询：拼接头实体和关系
        query = torch.cat([h_emb, r_emb], dim=-1)  # [batch, hidden_dim*3]
        query_vec = self.query_encoder(query)  # [batch, attention_hidden_dim]

        # 2. 扩展到所有规则
        query_exp = query_vec.unsqueeze(1).expand(batch_size, num_rules, -1)  # [batch, num_rules, attention_hidden_dim]
        rule_exp = rule_embs.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_rules, hidden_dim]

        # 3. 拼接query和rule特征
        attn_input = torch.cat([query_exp, rule_exp], dim=-1)  # [batch, num_rules, attention_hidden_dim + hidden_dim]

        # 4. 计算attention分数（使用sigmoid，每条规则独立打分）
        attention_scores = torch.sigmoid(self.attention_net(attn_input))  # [batch, num_rules, 1]

        return attention_scores


class QueryConditionedFusionGate(nn.Module):
    """
    方案四：Query-Conditioned Fusion Gate（学习 alpha(h,r) >= 0）
    alpha(h,r) 用于融合：final = grounding_logits + alpha(h,r) * kge_score
    """
    def __init__(self, hidden_dim, gate_hidden_dim=64, dropout=0.0, alpha_init=3.0, alpha_max=10.0):
        super(QueryConditionedFusionGate, self).__init__()

        self.alpha_max = float(alpha_max)

        self.gate_mlp = MLP(
            input_dim=hidden_dim * 3,
            hidden_dims=[gate_hidden_dim, 1],
            dropout=dropout
        )

        # initialize so that alpha(h,r) ~= alpha_init at start
        alpha_init = float(alpha_init)
        inv_softplus = math.log(math.exp(alpha_init) - 1.0) if alpha_init > 1e-6 else -20.0
        self.alpha_bias = nn.Parameter(torch.tensor(inv_softplus, dtype=torch.float))

    def forward(self, h_emb, r_emb):
        """
        Args:
            h_emb: [batch, hidden_dim*2]
            r_emb: [batch, hidden_dim]
        Returns:
            alpha: [batch, 1], in (0, alpha_max]
        """
        q = torch.cat([h_emb, r_emb], dim=-1)  # [batch, hidden_dim*3]
        raw = self.gate_mlp(q)  # [batch, 1]
        alpha = torch.nn.functional.softplus(raw + self.alpha_bias)
        if self.alpha_max > 0:
            alpha = torch.clamp(alpha, max=self.alpha_max)
        return alpha


class RulE(torch.nn.Module):
    def __init__(self, graph, p_norm, mlp_rule_dim, gamma_fact, gamma_rule, hidden_dim, device, dataset,
                 num_samples=5, lambda_0=1.0,
                 use_query_attention=False, attention_hidden_dim=64, attention_dropout=0.1,
                 use_hierarchical_agg=False, quality_thresholds=(0.4, 0.7),
                 hierarchical_hidden_dim=64, hierarchical_dropout=0.1,
                 use_fusion_gate=False, fusion_gate_hidden_dim=64, fusion_gate_dropout=0.0,
                 fusion_alpha_init=3.0, fusion_alpha_max=10.0):
        super(RulE, self).__init__()
        self.graph = graph
        self.device = device
        self.num_entities = graph.entity_size
        self.num_relations = graph.relation_size
        self.padding_index = graph.relation_size

        self.hidden_dim = hidden_dim
        self.num_samples = num_samples
        self.lambda_0 = lambda_0
        self.use_query_attention = use_query_attention
        self.use_hierarchical_agg = use_hierarchical_agg
        self.use_fusion_gate = use_fusion_gate
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

        # 方案一：Query-Conditioned Attention
        if self.use_query_attention:
            self.query_attention = QueryConditionedAttention(
                hidden_dim=hidden_dim,
                attention_hidden_dim=attention_hidden_dim,
                dropout=attention_dropout
            )
            logging.info(f'Query-Conditioned Attention enabled: attention_hidden_dim={attention_hidden_dim}, dropout={attention_dropout}')
        else:
            self.query_attention = None
            logging.info('Query-Conditioned Attention disabled')

        # 方案三：Hierarchical Rule Aggregation（分层规则聚合）
        if self.use_hierarchical_agg:
            thresholds = tuple(sorted(float(x) for x in quality_thresholds))
            self.quality_thresholds = thresholds
            self.num_quality_groups = len(self.quality_thresholds) + 1
            self.hierarchical_gate = MLP(
                input_dim=hidden_dim * 3,
                hidden_dims=[hierarchical_hidden_dim, self.num_quality_groups],
                dropout=hierarchical_dropout
            )
            logging.info(
                'Hierarchical Rule Aggregation enabled: '
                f'groups={self.num_quality_groups}, thresholds={self.quality_thresholds}, '
                f'hidden_dim={hierarchical_hidden_dim}, dropout={hierarchical_dropout}'
            )
        else:
            self.quality_thresholds = None
            self.num_quality_groups = 0
            self.hierarchical_gate = None
            logging.info('Hierarchical Rule Aggregation disabled')

        # 方案四：Query-Conditioned Fusion Gate（学习 alpha(h,r)）
        if self.use_fusion_gate:
            self.fusion_gate = QueryConditionedFusionGate(
                hidden_dim=hidden_dim,
                gate_hidden_dim=fusion_gate_hidden_dim,
                dropout=fusion_gate_dropout,
                alpha_init=fusion_alpha_init,
                alpha_max=fusion_alpha_max
            )
            logging.info(
                'Query-Conditioned Fusion Gate enabled: '
                f'hidden_dim={fusion_gate_hidden_dim}, dropout={fusion_gate_dropout}, '
                f'alpha_init={fusion_alpha_init}, alpha_max={fusion_alpha_max}'
            )
        else:
            self.fusion_gate = None
            logging.info('Query-Conditioned Fusion Gate disabled')

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

    def get_query_embeddings(self, all_h, all_r):
        """
        Build query embeddings for (h, r).

        Args:
            all_h: [batch]
            all_r: [batch] (supports inverse relations via r // num_relations)

        Returns:
            h_emb: [batch, hidden_dim*2]
            r_emb: [batch, hidden_dim]
        """
        h_emb = self.entity_embedding(all_h)
        relations_flag = torch.pow(-1, all_r // self.num_relations).unsqueeze(-1)
        r_id = all_r % self.num_relations
        r_emb = self.relation_embedding(r_id) * relations_flag
        return h_emb, r_emb

    def compute_fusion_alpha(self, all_h, all_r):
        """
        Compute alpha(h,r) for fusion gate, if enabled.
        Returns None when fusion gate is disabled.
        """
        if not self.use_fusion_gate or self.fusion_gate is None:
            return None
        h_emb, r_emb = self.get_query_embeddings(all_h, all_r)
        return self.fusion_gate(h_emb, r_emb)

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

        # 不确定性建模网络（输入为 [R_i, r_body_sum]，输出标量 μ_i / logσ_i²）
        input_dim = self.rule_dim + self.hidden_dim
        self.mu_network = MLP(input_dim, [256, 128, 1])
        self.logvar_network = MLP(input_dim, [256, 128, 1])

        # 加载预计算的support_counts
        import os
        support_count_path = os.path.join(self.graph.data_path, 'support_counts.pt')
        if os.path.exists(support_count_path):
            self.support_counts = torch.load(support_count_path)
            if self.support_counts.shape[0] != self.num_rules:
                logging.warning(
                    f'support_counts length ({self.support_counts.shape[0]}) '
                    f'does not match num_rules ({self.num_rules}); using min(len, num_rules).'
                )
                min_len = min(self.support_counts.shape[0], self.num_rules)
                self.support_counts = self.support_counts[:min_len]
            logging.info(f'Loaded support_counts from {support_count_path}')
        else:
            logging.warning('support_counts.pt not found, using uniform counts')
            self.support_counts = torch.ones(self.num_rules)

        logging.info(
            f'Uncertainty modeling initialized: num_rules={self.num_rules}, '
            f'num_samples={self.num_samples}, lambda_0={self.lambda_0}'
        )

    def reparameterize_sample(self, mu, logvar):
        """
        重参数化技巧: w = μ + σ * ε, 其中 ε ~ N(0,1)

        Args:
            mu: [num_rules, 1]
            logvar: [num_rules, 1]

        Returns:
            w_avg: 平均后的采样权重 [num_rules, 1]
            std: 标准差 [num_rules, 1]
        """
        std = torch.exp(0.5 * logvar)
        samples = []
        for _ in range(self.num_samples):
            eps = torch.randn_like(mu)
            w = mu + std * eps
            samples.append(w)
        w_avg = torch.stack(samples, dim=0).mean(dim=0)
        return w_avg, std

    def get_uncertainty_features(self, device=None):
        """
        构造不确定性建模使用的规则特征:
            features_i = concat([R_i, r_body_sum_i])
        其中:
            R_i 为规则嵌入 rule_emb[i]
            r_body_sum_i 为规则体关系嵌入之和
        """
        if device is None:
            device = self.entity_embedding.weight.device

        # 规则嵌入 R_i
        rule_ids = torch.arange(self.num_rules, device=device)
        R_i = self.rule_emb(rule_ids)  # [num_rules, rule_dim]

        # 规则体关系 id 与 mask
        rule_features = self.rule_features.to(device)  # [num_rules, 2 + max_len]
        rule_masks = self.rule_masks.to(device)        # [num_rules, max_len]

        body = rule_features[:, 2:]                    # [num_rules, max_len]
        mask = rule_masks                              # bool

        relations_flag = torch.pow(-1, body // self.num_relations).unsqueeze(-1)
        inputs_com = body % self.num_relations
        inputs_com = torch.where(body == self.num_relations * 2, self.padding_index, inputs_com)

        embedding = self.relation_embedding(inputs_com) * relations_flag  # [num_rules, max_len, hidden_dim]
        cal_mask = mask.unsqueeze(-1).float()
        body_emb = embedding * cal_mask
        r_body_sum = body_emb.sum(dim=1)                                   # [num_rules, hidden_dim]

        features = torch.cat([R_i, r_body_sum], dim=-1)                    # [num_rules, rule_dim + hidden_dim]

        return features

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

        for idx, (index, (r_head, r_body)) in enumerate(self.relation2rules[query_r]):

            assert r_head == query_r

            count = self.graph.grounding(all_h, r_head, r_body, edges_to_remove).float()

            mask += count

            rule_index.append(index)
            rule_count.append(count)

        if mask.sum().item() == 0:
            # no grounding evidence; if fusion enabled, fall back to KGE for this query
            if self.use_fusion_gate and self.fusion_gate is not None:
                alpha = self.compute_fusion_alpha(all_h, all_r)  # [batch, 1]
                kge_score = self.compute_g_KGE(all_h, all_r)
                score = alpha * kge_score + self.bias.unsqueeze(0)
                mask = torch.ones_like(mask).bool()
                return score, mask
            return mask + self.bias.unsqueeze(0), (1 - mask).bool()


        candidate_set = torch.nonzero(mask.view(-1), as_tuple=True)[0]
        candidate_query_idx = candidate_set // self.graph.entity_size  # [num_candidates]

        rule_index = torch.tensor(rule_index, dtype=torch.long, device=device)
        rule_count = torch.stack(rule_count, dim=0)

        rule_count = rule_count.reshape(rule_index.size(0), -1)[:, candidate_set]

        rule_emb = self.rules_weight_emb[rule_index]

        # grounding 阶段：优先使用预训练后预计算的规则置信度 μ 作为标量权重
        # 如果未预计算（例如仅做前向调试），则退回到当前的 mlp_feature（不加权）
        if hasattr(self, 'rule_mu'):
            w = self.rule_mu[rule_index]                      # [num_selected_rules, 1]
            base_feature = self.mlp_feature[rule_index]       # [num_selected_rules, mlp_rule_dim]
            mlp_feature = base_feature * w                    # broadcast 标量权重到特征维度
        else:
            mlp_feature = self.mlp_feature[rule_index]

        need_query_repr = (
            (self.use_query_attention and self.query_attention is not None)
            or (self.use_hierarchical_agg and self.hierarchical_gate is not None)
            or (self.use_fusion_gate and self.fusion_gate is not None)
        )
        if need_query_repr:
            h_emb, r_emb = self.get_query_embeddings(all_h, all_r)

        # 方案一：Query-Conditioned Attention（真正做到 query-specific）
        if self.use_query_attention and self.query_attention is not None:
            rule_embeddings = self.rule_emb(rule_index)  # [num_selected_rules, hidden_dim]
            attention = self.query_attention(h_emb, r_emb, rule_embeddings).squeeze(-1)  # [batch, num_selected_rules]
            attention = attention / (attention.mean(dim=1, keepdim=True) + 1e-9)  # stabilize
            attention_for_candidates = attention.index_select(0, candidate_query_idx)  # [num_candidates, num_selected_rules]
            rule_count = rule_count * attention_for_candidates.transpose(0, 1)         # [num_selected_rules, num_candidates]

        # 方案三：Hierarchical Rule Aggregation（分层规则聚合）
        if self.use_hierarchical_agg and self.hierarchical_gate is not None:
            # 计算每条规则的质量分数 q∈(0,1)，优先使用预计算的 rule_mu
            if hasattr(self, 'rule_mu'):
                mu_sel = self.rule_mu[rule_index].squeeze(-1)  # [num_selected_rules]
                if hasattr(self, 'rule_mu_min') and hasattr(self, 'rule_mu_max'):
                    mu_min = self.rule_mu_min.to(device=device, dtype=mu_sel.dtype)
                    mu_max = self.rule_mu_max.to(device=device, dtype=mu_sel.dtype)
                else:
                    mu_all = self.rule_mu.squeeze(-1)
                    mu_min = mu_all.min().to(device=device, dtype=mu_sel.dtype)
                    mu_max = mu_all.max().to(device=device, dtype=mu_sel.dtype)
                q = (mu_sel - mu_min) / (mu_max - mu_min + 1e-9)  # min-max normalize to [0,1]
            else:
                q = torch.ones(rule_index.size(0), device=device)

            thresholds = torch.tensor(self.quality_thresholds, device=device, dtype=q.dtype)
            group_ids = torch.bucketize(q, thresholds, right=False)  # [num_selected_rules], 0..G-1

            # 层间 gate：对每个 query 输出每层权重（softmax）
            query_vec = torch.cat([h_emb, r_emb], dim=-1)  # [batch, hidden_dim*3]
            group_logits = self.hierarchical_gate(query_vec)  # [batch, G]
            group_weights = torch.softmax(group_logits, dim=-1)  # [batch, G]
            group_weights_for_candidates = group_weights.index_select(0, candidate_query_idx)  # [num_candidates, G]

            # 分层聚合：先层内 rule_to_entity，再层间加权求和
            final_feature = torch.zeros(candidate_set.size(0), self.mlp_rule_dim, device=device)
            for g in range(self.num_quality_groups):
                g_mask = group_ids == g
                if g_mask.sum().item() == 0:
                    continue

                g_rule_count = rule_count[g_mask]
                g_rule_emb = rule_emb[g_mask]
                g_mlp_feature = mlp_feature[g_mask]

                g_feature = self.rule_to_entity(g_rule_count, g_rule_emb, g_mlp_feature)  # [num_candidates, mlp_rule_dim]
                g_weight = group_weights_for_candidates[:, g].unsqueeze(-1)               # [num_candidates, 1]
                final_feature = final_feature + g_feature * g_weight

            output = final_feature
        else:
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

        # 方案四：融合 KGE（query-conditioned alpha）
        if self.use_fusion_gate and self.fusion_gate is not None:
            alpha = self.fusion_gate(h_emb, r_emb)  # [batch, 1]
            kge_score = self.compute_g_KGE(all_h, all_r)
            score = score + alpha * kge_score
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
