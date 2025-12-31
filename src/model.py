
import torch
import torch.nn as nn
import logging, math
from layers import MLP, FuncToNodeSum

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
        use_rule_conf=False,
        count_transform="none",
        rule_conf_init=0.5,
        rule_conf_log_every=200,
    ):
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

        self.use_rule_conf = bool(use_rule_conf)
        self.count_transform = str(count_transform or "none")
        self.rule_conf_init = float(rule_conf_init)
        self.rule_conf_log_every = int(rule_conf_log_every)
        self._forward_calls = 0

        
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

    def _transform_count(self, rule_count: torch.Tensor) -> torch.Tensor:
        if self.count_transform == "none":
            return rule_count
        if self.count_transform == "log1p":
            return torch.log1p(rule_count)
        if self.count_transform == "sqrt":
            return torch.sqrt(rule_count)
        raise ValueError(f"Unknown count_transform={self.count_transform!r}")

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

        if self.use_rule_conf:
            self.rule_conf = nn.Embedding(self.num_rules, 1)
            init = min(max(self.rule_conf_init, 1e-4), 1.0 - 1e-4)
            init_logit = math.log(init / (1.0 - init))
            nn.init.constant_(self.rule_conf.weight, init_logit)

        self.rule_emb = torch.nn.Embedding(self.num_rules, self.rule_dim)
        nn.init.kaiming_uniform_(self.rule_emb.weight, a=math.sqrt(5), mode="fan_in")
        # nn.init.uniform_(
        #     tensor=self.rule_emb.weight, 
        #     a=-self.embedding_range_rule.item(), 
        #     b=self.embedding_range_rule.item()
        # )
        self._build_rule_tries()

    def _build_rule_tries(self):
        self.rule_tries = [None for _ in range(self.num_relations * 2)]
        self.rule_end_node = [-1 for _ in range(self.num_rules)]

        for r_head, rules in enumerate(self.relation2rules):
            if not rules:
                continue

            # Trie nodes: 0 is root.
            children = [dict()]
            end_rule_ids = [[]]

            for rule_id, (_, body) in rules:
                node_id = 0
                for rel in body:
                    next_id = children[node_id].get(rel)
                    if next_id is None:
                        next_id = len(children)
                        children[node_id][rel] = next_id
                        children.append(dict())
                        end_rule_ids.append([])
                    node_id = next_id
                end_rule_ids[node_id].append(rule_id)
                self.rule_end_node[rule_id] = node_id

            self.rule_tries[r_head] = (children, end_rule_ids)

    def _ground_rule_trie(self, all_h, query_r, edges_to_remove):
        children, end_rule_ids = self.rule_tries[query_r]
        device = all_h.device

        # Only keep counts for nodes that correspond to at least one rule.
        node_multiplicity = {i: len(rules) for i, rules in enumerate(end_rule_ids) if rules}
        node_count = {}
        propagate_calls = 0
        visited_nodes = 0

        with torch.no_grad():
            x0 = torch.nn.functional.one_hot(all_h, self.graph.entity_size).transpose(0, 1).unsqueeze(-1)
            if device.type == "cuda":
                x0 = x0.cuda(device)

            stack = [(0, x0)]
            while stack:
                node_id, x = stack.pop()
                visited_nodes += 1

                if node_id in node_multiplicity:
                    node_count[node_id] = x.squeeze(-1).transpose(0, 1).float()

                for rel, child_id in children[node_id].items():
                    next_edges_to_remove = edges_to_remove if (rel == query_r) else None
                    x_child = self.graph.propagate(x, rel, next_edges_to_remove)
                    propagate_calls += 1
                    stack.append((child_id, x_child))

        logging.info(
            "Trie grounding relation=%s: visited_nodes=%s propagate_calls=%s end_nodes_hit=%s edges_to_remove=%s",
            query_r,
            visited_nodes,
            propagate_calls,
            len(node_count),
            edges_to_remove is not None,
        )
        return node_count, node_multiplicity
        
       
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
        self._forward_calls += 1
        query_r = all_r[0].item()
        assert (all_r != query_r).sum() == 0
        device = all_r.device
        
        if device.type == "cuda":
            self.rule_features = self.rule_features.cuda(device)

        rule_index = list()
        rule_count = list()
        
        
        candidate_strength = torch.zeros(all_h.size(0), self.graph.entity_size, device=device)
        if hasattr(self, "rule_tries") and self.rule_tries[query_r] is not None:
            node_count, node_multiplicity = self._ground_rule_trie(all_h, query_r, edges_to_remove)

            for node_id, mult in node_multiplicity.items():
                candidate_strength += node_count[node_id] * float(mult)

            for index, (r_head, _) in self.relation2rules[query_r]:
                assert r_head == query_r
                node_id = self.rule_end_node[index]
                count = node_count[node_id]
                rule_index.append(index)
                rule_count.append(count)
        else:
            for index, (r_head, r_body) in self.relation2rules[query_r]:

                assert r_head == query_r

                count = self.graph.grounding(all_h, r_head, r_body, edges_to_remove).float()
                
                candidate_strength += count

                rule_index.append(index)
                rule_count.append(count)


        candidate_mask = candidate_strength > 0
        try:
            cand_per_query = candidate_mask.sum(dim=1).detach().cpu().tolist()
            logging.info(
                "Candidate entities (count>0) per query in batch: %s",
                cand_per_query,
            )
        except Exception:
            pass
        if candidate_strength.sum().item() == 0:
            score = candidate_strength + self.bias.unsqueeze(0)
            return score, candidate_mask, candidate_strength


        candidate_set = torch.nonzero(candidate_strength.view(-1), as_tuple=True)[0]

        rule_index = torch.tensor(rule_index, dtype=torch.long, device=device)
        rule_count = torch.stack(rule_count, dim=0)

        rule_count = rule_count.reshape(rule_index.size(0), -1)[:, candidate_set]

        if self.count_transform != "none":
            rule_count = self._transform_count(rule_count)

        if self.use_rule_conf:
            if not hasattr(self, "rule_conf"):
                raise RuntimeError("use_rule_conf=True but rule_conf not initialized; call set_rules() first")
            conf = torch.sigmoid(self.rule_conf(rule_index)).view(-1, 1)
            rule_count = rule_count * conf

            if self.rule_conf_log_every > 0 and (self._forward_calls % self.rule_conf_log_every == 0):
                with torch.no_grad():
                    conf_mean = conf.mean().item()
                    conf_std = conf.std(unbiased=False).item() if conf.numel() > 1 else 0.0
                    conf_min = conf.min().item()
                    conf_max = conf.max().item()
                    logging.info(
                        "RuleConf stats: relation=%s rules=%s candidates=%s count_transform=%s conf(mean=%.4g std=%.4g min=%.4g max=%.4g)",
                        query_r,
                        int(rule_index.numel()),
                        int(candidate_set.numel()),
                        self.count_transform,
                        conf_mean,
                        conf_std,
                        conf_min,
                        conf_max,
                    )
        
        rule_weight_emb = self.rules_weight_emb[rule_index]

        mlp_feature = self.mlp_feature[rule_index]

        # output = self.rule_to_entity(rule_count, mlp_feature)
        output = self.rule_to_entity(rule_count, rule_weight_emb, mlp_feature)


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

        return score, candidate_mask, candidate_strength




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
