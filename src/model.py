
import torch
import torch.nn as nn
import logging, math
from layers import MLP, FuncToNodeSum

from torch.nn.utils.rnn import pad_sequence

class RulE(torch.nn.Module):
    def __init__(self, graph, p_norm, mlp_rule_dim, gamma_fact, gamma_rule, hidden_dim, device, dataset, rule_compose_mode='add'):
        super(RulE, self).__init__()
        self.graph = graph
        self.device = device
        self.rule_compose_mode = rule_compose_mode
        self.num_entities = graph.entity_size
        self.num_relations = graph.relation_size
        self.padding_index = graph.relation_size

        self.hidden_dim = hidden_dim
        self.p = p_norm

        self.mlp_rule_dim = mlp_rule_dim
        self.dataset = dataset

        self.rule_to_entity = FuncToNodeSum(self.mlp_rule_dim)

        # score_model 输入维度固定为 mlp_rule_dim
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

        # 日志：记录创新模块配置
        logging.info('=' * 50)
        logging.info('RulE Model Configuration:')
        logging.info('  rule_compose_mode: %s', self.rule_compose_mode)
        logging.info('=' * 50)

    def init_rule_structure_feature(self, use_rule_structure=True):
        """
        在 Grounding 阶段初始化规则结构感知模块。
        用规则体的关系 embedding 生成规则特征，替代独立学习的 mlp_feature。

        Args:
            use_rule_structure: 是否使用规则结构感知
        """
        self.use_rule_structure = use_rule_structure
        device = next(self.parameters()).device

        logging.info('=' * 50)
        logging.info('Initializing Rule Structure Feature for Grounding:')
        logging.info('  use_rule_structure: %s', use_rule_structure)

        if use_rule_structure:
            # 规则结构投影层：将 relation_embedding (hidden_dim) 投影到 mlp_rule_dim
            self.structure_proj = nn.Linear(self.hidden_dim, self.mlp_rule_dim).to(device)
            logging.info('  -> structure_proj: Linear(%d -> %d)', self.hidden_dim, self.mlp_rule_dim)
            logging.info('  -> rule_structure_feature will be computed dynamically in forward')

        logging.info('=' * 50)

    def _compute_rule_structure_feature(self, rule_indices, device):
        """
        根据规则体的关系 embedding 动态计算规则结构特征。
        每次 forward 时调用，确保计算图正确。

        Args:
            rule_indices: 需要计算的规则索引 tensor [num_selected_rules]
            device: 计算设备

        Returns:
            rule_structure_feature: [num_selected_rules, mlp_rule_dim]
        """
        # rule_features: [num_rules, max_len+2]，包含 [rule_id, rule_head, body...]
        rule_body = self.rule_features[rule_indices, 2:].to(device)  # [num_selected, max_body_len]

        # 处理逆关系：rule_body 中的值可能 >= num_relations (表示逆关系)
        relations_flag = torch.pow(-1, rule_body // self.num_relations).unsqueeze(-1).float()  # [num_selected, max_body_len, 1]
        rule_body_rel = rule_body % self.num_relations
        rule_body_rel = torch.where(rule_body == self.num_relations * 2, self.padding_index, rule_body_rel)

        # 获取关系 embedding
        body_emb = self.relation_embedding(rule_body_rel)  # [num_selected, max_body_len, hidden_dim]
        body_emb = body_emb * relations_flag  # 应用逆关系标志

        # mask 掉 padding
        body_mask = (rule_body != self.num_relations * 2).unsqueeze(-1).float()  # [num_selected, max_body_len, 1]

        # 聚合：对规则体的关系 embedding 求平均
        body_emb_masked = body_emb * body_mask
        body_emb_sum = body_emb_masked.sum(dim=1)  # [num_selected, hidden_dim]
        body_len = body_mask.sum(dim=1).clamp(min=1)  # [num_selected, 1]
        body_emb_avg = body_emb_sum / body_len  # [num_selected, hidden_dim]

        # 投影到 mlp_rule_dim
        rule_structure_feature = self.structure_proj(body_emb_avg)  # [num_selected, mlp_rule_dim]

        return rule_structure_feature

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
    


    def _rotate_compose_body(self, embedding, cal_mask, rule_embedding, embedding_r):
        """
        RotatE-style complex rotation composition for rule bodies.

        Args:
            embedding: relation embeddings with flag applied, [batch, neg, body_len, hidden_dim]
            cal_mask: body mask, [batch, 1/neg, body_len, 1]
            rule_embedding: rule embeddings, [batch, neg, hidden_dim]
            embedding_r: rule head relation embedding, [batch, neg, hidden_dim]

        Returns:
            (re_combined, im_combined, re_head, im_head) each [batch, neg, hidden_dim]
        """
        phase_factor = self.embedding_range_fact.item() / self.pi

        # Convert body relation embeddings to phases, mask padding positions
        phase_body = (embedding / phase_factor) * cal_mask  # [batch, neg, body_len, hidden_dim]

        # Sum phases across body (equivalent to complex number multiplication)
        phase_sum = phase_body.sum(-2)  # [batch, neg, hidden_dim]

        # Add rule embedding as phase correction
        phase_rule = rule_embedding / phase_factor
        phase_combined = phase_sum + phase_rule  # [batch, neg, hidden_dim]

        # Convert to complex representation
        re_combined = torch.cos(phase_combined)
        im_combined = torch.sin(phase_combined)

        # Convert rule head to complex representation
        phase_head = embedding_r / phase_factor
        re_head = torch.cos(phase_head)
        im_head = torch.sin(phase_head)

        return re_combined, im_combined, re_head, im_head

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

        if self.rule_compose_mode == 'rotate':
            re_combined, im_combined, re_head, im_head = self._rotate_compose_body(
                embedding, cal_mask, rule_embedding, embedding_r
            )
            re_diff = re_combined - re_head
            im_diff = im_combined - im_head
            # Per-dimension distance, then norm across hidden_dim
            # 原始距离（归一化前）
            dist_raw = torch.sqrt(re_diff ** 2 + im_diff ** 2 + 1e-12)
            # 归一化：乘以 embedding_range_fact 使尺度与 add 模式一致
            dist_per_dim = dist_raw * self.embedding_range_fact.item()
            dist_norm = torch.norm(dist_per_dim, p=self.p, dim=-1)
            dist = self.gamma_rule.item() - dist_norm

            # 日志：记录 rotate 模块统计信息
            if not hasattr(self, '_rotate_log_counter'):
                self._rotate_log_counter = 0
            self._rotate_log_counter += 1
            if self._rotate_log_counter % 1000 == 1:
                logging.info('[Rotate] dist_raw: mean=%.4f, max=%.4f | dist_scaled: mean=%.4f, max=%.4f | score: mean=%.4f, min=%.4f, max=%.4f',
                             dist_raw.mean().item(), dist_raw.max().item(),
                             dist_norm.mean().item(), dist_norm.max().item(),
                             dist.mean().item(), dist.min().item(), dist.max().item())
        else:
            outputs = rule_body.sum(-2) + rule_embedding
            diff = outputs - embedding_r
            dist_norm = torch.norm(diff, p=self.p, dim=-1)
            dist = self.gamma_rule.item() - dist_norm

            # 日志：记录 add 模块统计信息
            if not hasattr(self, '_add_log_counter'):
                self._add_log_counter = 0
            self._add_log_counter += 1
            if self._add_log_counter % 1000 == 1:
                logging.info('[Add] diff: mean=%.4f, max=%.4f | dist_norm: mean=%.4f, max=%.4f | score: mean=%.4f, min=%.4f, max=%.4f',
                             diff.abs().mean().item(), diff.abs().max().item(),
                             dist_norm.mean().item(), dist_norm.max().item(),
                             dist.mean().item(), dist.min().item(), dist.max().item())

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

        if self.rule_compose_mode == 'rotate':
            re_combined, im_combined, re_head, im_head = self._rotate_compose_body(
                embedding, cal_mask, rule_embedding, embedding_r
            )
            re_diff = re_combined - re_head
            im_diff = im_combined - im_head
            # Per-dimension distance, consistent with add_ruleE_g output shape [batch, 1, hidden_dim]
            # 归一化：乘以 embedding_range_fact 使尺度与 add 模式一致
            dist = self.gamma_rule.item() / self.hidden_dim - torch.pow(
                torch.sqrt(re_diff ** 2 + im_diff ** 2 + 1e-12) * self.embedding_range_fact.item(), self.p
            )
        else:
            outputs = rule_body.sum(-2) + rule_embedding
            dist = self.gamma_rule.item()/self.hidden_dim - torch.pow((outputs - embedding_r), self.p)

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

        # === 规则结构感知模块 ===
        # 如果启用了规则结构感知，动态计算规则结构特征替代独立学习的 mlp_feature
        if getattr(self, 'use_rule_structure', False) and hasattr(self, 'structure_proj'):
            # 动态计算规则结构特征（每次 forward 都重新计算，确保计算图正确）
            mlp_feature = self._compute_rule_structure_feature(rule_index, device)
        else:
            # 使用原始的独立学习的 mlp_feature
            mlp_feature = self.mlp_feature[rule_index]

        rule_output = self.rule_to_entity(rule_count, rule_emb, mlp_feature)

        output = self.score_model(rule_output).squeeze(-1)

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
