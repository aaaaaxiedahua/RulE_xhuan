
import torch
import torch.nn as nn
import logging, math
from layers import MLP, FuncToNodeSum

from torch.nn.utils.rnn import pad_sequence

class RulE(torch.nn.Module):
    def __init__(self, graph, p_norm, mlp_rule_dim, gamma_fact, gamma_rule, hidden_dim, device, dataset,
                 use_policy_network=False, policy_hidden_dim=256):
        """
        RulE模型初始化

        参数：
            ... (原有参数)
            use_policy_network: 是否使用策略网络（RulE-SSRL模式）
            policy_hidden_dim: 策略网络隐藏层维度
        """
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

        # ========== RulE-SSRL: 策略网络（可选）==========
        self.use_policy_network = use_policy_network
        if use_policy_network:
            from policy_network import RuleGuidedPolicyNetwork
            logging.info('初始化RulE-SSRL策略网络（向量化批处理版本）')
            self.policy_network = RuleGuidedPolicyNetwork(
                entity_dim=hidden_dim * 2,       # RotatE实体维度
                relation_dim=hidden_dim,          # RotatE关系维度
                rule_dim=hidden_dim,              # 规则嵌入维度
                hidden_dim=policy_hidden_dim,     # 策略隐藏维度
                max_num_actions=graph.max_num_actions,  # 使用KG的max_num_actions
                num_layers=1,
                dropout=0.1
            )
        else:
            self.policy_network = None

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
    

    def forward(self, all_h, all_r, edges_to_remove=None):
        """
        推理入口

        根据use_policy_network路由到不同的推理方法：
        - False: 原始RulE的Grounding推理
        - True: 策略网络推理（RulE-SSRL）

        参数：
            all_h: [batch_size] 头实体
            all_r: [batch_size] 查询关系
            edges_to_remove: Grounding期间需要屏蔽的边（可选）

        返回：
            score: [batch_size, num_entities] 实体得分
            mask: [batch_size, num_entities] 有效实体掩码
        """
        if not self.use_policy_network:
            # 原始RulE的Grounding推理
            return self.forward_grounding(all_h, all_r, edges_to_remove)
        else:
            # RulE-SSRL策略网络推理
            return self.forward_policy(all_h, all_r)

    def forward_grounding(self, all_h, all_r, edges_to_remove):
        """
        原始RulE的Grounding推理

        这是原来的forward方法，为清晰起见重命名。
        使用规则grounding来计算实体得分。
        """
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


    # ========== RulE-SSRL: 新增方法 ==========

    def forward_policy(self, all_h, all_r, num_samples=10):
        """
        策略网络推理（RulE-SSRL）

        使用策略网络采样多条路径并聚合结果。

        参数：
            all_h: [batch_size] 头实体
            all_r: [batch_size] 查询关系
            num_samples: 每个查询采样的路径数量

        返回：
            scores: [batch_size, num_entities] 实体得分
            mask: [batch_size, num_entities] 全为True（策略可达任意实体）
        """
        from collections import defaultdict

        batch_size = all_h.size(0)
        device = all_h.device

        # 初始化得分矩阵
        scores = torch.zeros(batch_size, self.num_entities, device=device)

        # 处理每个查询
        for i in range(batch_size):
            h = all_h[i].item()
            r = all_r[i].item()

            # 多路径采样
            entity_visit_count = defaultdict(int)

            for _ in range(num_samples):
                # 用策略网络采样一条路径
                with torch.no_grad():
                    path, final_entity = self.policy_network.rollout(
                        start_entity=h,
                        query_relation=r,
                        graph=self.graph,
                        model=self,
                        max_steps=3
                    )

                entity_visit_count[final_entity] += 1

            # 聚合：简单投票
            for entity, count in entity_visit_count.items():
                scores[i, entity] = count / num_samples

        # mask全为True（策略网络可以潜在到达任意实体）
        mask = torch.ones_like(scores).bool()

        return scores, mask

    def compute_policy_loss(self, query_batch):
        """
        计算规则监督的策略损失

        这是RulE-SSRL的核心创新：用规则作为软标签
        引导策略学习，替代SSRL基于BFS的标签。

        参数：
            query_batch: [batch_size, 3] (h, r, t)三元组张量

        返回：
            loss: 标量张量
        """
        from policy_network import PolicyNetworkTrainingHelper

        return PolicyNetworkTrainingHelper.compute_rule_supervised_loss(
            policy_network=self.policy_network,
            query_batch=query_batch,
            graph=self.graph,
            model=self,
            device=self.device
        )

    # ========== 策略网络辅助方法 ==========

    def get_rules_for_relation(self, relation_id):
        """
        获取与给定关系相关的规则

        参数：
            relation_id: 关系ID (int)

        返回：
            rules_info: (rule_id, rule_emb, rule_body)元组列表
        """
        if relation_id >= len(self.relation2rules):
            return []

        rules_info = []
        for rule_id, (r_head, r_body) in self.relation2rules[relation_id]:
            rule_emb = self.rule_emb(torch.tensor([rule_id], device=self.device))
            rules_info.append((rule_id, rule_emb, r_body))

        return rules_info

    def get_entity_embedding_by_id(self, entity_id):
        """
        根据ID获取实体嵌入（供策略网络使用）

        参数：
            entity_id: 实体ID (int 或 Tensor)
                      支持标量或任意形状的tensor

        返回：
            emb: [entity_dim] 或 [*, entity_dim] 张量
        """
        # 统一处理标量和tensor输入
        if isinstance(entity_id, int):
            entity_id = torch.tensor([entity_id], device=self.device)
        elif not isinstance(entity_id, torch.Tensor):
            entity_id = torch.tensor(entity_id, device=self.device)

        # 如果不在正确设备上，移动到模型设备
        if entity_id.device != self.device:
            entity_id = entity_id.to(self.device)

        return self.entity_embedding(entity_id)

    def get_relation_embedding_by_id(self, relation_id):
        """
        根据ID获取关系嵌入（供策略网络使用，参考SSRL的embedding_lookup）

        参数：
            relation_id: 关系ID (int 或 Tensor)
                        支持标量或任意形状的tensor
                        逆关系ID >= num_relations

        返回：
            emb: [relation_dim] 或 [*, relation_dim] 张量
        """
        # 统一处理标量和tensor输入
        if isinstance(relation_id, int):
            relation_id = torch.tensor([relation_id], device=self.device)
        elif not isinstance(relation_id, torch.Tensor):
            relation_id = torch.tensor(relation_id, device=self.device)

        # 如果不在正确设备上，移动到模型设备
        if relation_id.device != self.device:
            relation_id = relation_id.to(self.device)

        # 向量化处理逆关系（参考SSRL设计）
        # 逆关系：ID >= num_relations，需要对embedding取负
        is_inverse = relation_id >= self.num_relations
        actual_rel_id = torch.where(is_inverse,
                                     relation_id % self.num_relations,
                                     relation_id)

        # 获取embedding
        emb = self.relation_embedding(actual_rel_id)

        # 对逆关系取负
        emb = torch.where(is_inverse.unsqueeze(-1).expand_as(emb), -emb, emb)

        return emb
