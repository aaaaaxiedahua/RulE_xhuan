
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
        self.configure_elastic_grounding(enabled=False)

    def configure_elastic_grounding(
        self,
        enabled=False,
        k_soft=16,
        u_soft_cap=32,
        tau=2.0,
        lambda_base=0.5,
        position_lambda=True,
        th_prob=None,
        log_first_n=5,
        log_every=1000,
    ):
        self.elastic_enabled = bool(enabled)
        self.elastic_k_soft = int(k_soft)
        self.elastic_u_soft_cap = int(u_soft_cap)
        self.elastic_tau = float(tau)
        self.elastic_lambda_base = float(lambda_base)
        self.elastic_position_lambda = bool(position_lambda)
        self.elastic_th_prob = None if th_prob is None else float(th_prob)
        self.elastic_log_first_n = int(log_first_n)
        self.elastic_log_every = int(log_every)
        if not hasattr(self, "_elastic_log_calls"):
            self._elastic_log_calls = 0

    def _elastic_lambda_at_depth(self, hop_index):
        lam = self.elastic_lambda_base
        if lam <= 0:
            return 0.0
        if not self.elastic_position_lambda:
            return min(float(lam), 1.0)
        max_len = getattr(self, "max_length", 0)
        exponent = max(int(max_len) - int(hop_index), 0)
        return min(float(lam) ** exponent, 1.0)

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

            if not getattr(self, "elastic_enabled", False):
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

                return node_count, node_multiplicity

            xH0 = x0.float()
            xS0 = torch.zeros_like(xH0)
            stats_dead_h_total = 0
            stats_dead_h_selected = 0
            stats_soft_u_kept = 0
            stats_soft_candidates = 0
            stats_soft_u_filtered = 0
            stats_soft_dead_dropped = 0
            stats_overlap_upgraded = 0
            stack = [(0, 0, xH0, xS0)]
            while stack:
                node_id, depth, xH, xS = stack.pop()
                visited_nodes += 1

                if node_id in node_multiplicity:
                    node_count[node_id] = (xH + xS).squeeze(-1).transpose(0, 1).float()

                for rel, child_id in children[node_id].items():
                    next_edges_to_remove = edges_to_remove if (rel == query_r) else None
                    hop_index = depth + 1

                    xH_hard = self.graph.propagate(xH, rel, next_edges_to_remove)
                    xS_hard = self.graph.propagate(xS, rel, next_edges_to_remove)
                    propagate_calls += 2

                    xS_soft = torch.zeros_like(xS_hard)
                    if self.elastic_k_soft > 0 and self.elastic_u_soft_cap > 0 and self.elastic_tau > 0:
                        head_outdeg = self.graph.relation2head_outdegree[rel]
                        if device.type == "cuda":
                            head_outdeg = head_outdeg.to(device)

                        xH_w = xH.squeeze(-1)
                        dead_mask = (xH_w > 0) & (head_outdeg.unsqueeze(1) == 0)
                        stats_dead_h_total += int(dead_mask.sum().item())

                        xS_w = xS.squeeze(-1)
                        dead_s_mask = (xS_w > 0) & (head_outdeg.unsqueeze(1) == 0)
                        stats_soft_dead_dropped += int(dead_s_mask.sum().item())

                        node_in = None
                        node_out = None
                        removed_head = None
                        removed_tail = None
                        if next_edges_to_remove is not None:
                            node_in = self.graph.relation2adjacency[rel][0][1]
                            node_out = self.graph.relation2adjacency[rel][0][0]
                            if device.type == "cuda":
                                node_in = node_in.to(device)
                                node_out = node_out.to(device)
                            removed_head = node_in[next_edges_to_remove]
                            removed_tail = node_out[next_edges_to_remove]
                            # If the removed edge is the only outgoing edge for that head, treat it as dead for this hop.
                            only_one = head_outdeg[removed_head] == 1
                            if only_one.any():
                                b_idx = torch.arange(xH_w.size(1), device=device)[only_one]
                                dead_mask[removed_head[only_one], b_idx] = True
                                stats_dead_h_total += int(only_one.sum().item())

                        dead_heads = []
                        dead_batch = []
                        dead_weight = []
                        B = xH_w.size(1)
                        for b in range(B):
                            idx = torch.nonzero(dead_mask[:, b], as_tuple=False).squeeze(1)
                            if idx.numel() == 0:
                                continue
                            w = xH_w[idx, b]
                            cap = min(self.elastic_u_soft_cap, idx.numel())
                            if cap < idx.numel():
                                top = torch.topk(w, cap, largest=True).indices
                                idx = idx[top]
                                w = w[top]
                            dead_heads.append(idx)
                            dead_batch.append(torch.full((idx.numel(),), b, dtype=torch.long, device=device))
                            dead_weight.append(w)

                        if dead_heads:
                            dead_heads = torch.cat(dead_heads, dim=0)
                            dead_batch = torch.cat(dead_batch, dim=0)
                            dead_weight = torch.cat(dead_weight, dim=0)
                            stats_dead_h_selected += int(dead_heads.numel())

                            rels = torch.full((dead_heads.size(0),), rel, dtype=torch.long, device=device)
                            scores = self.compute_g_KGE(dead_heads, rels)

                            if removed_head is not None:
                                rh = removed_head[dead_batch]
                                rt = removed_tail[dead_batch]
                                forbid = dead_heads == rh
                                if forbid.any():
                                    rows = torch.nonzero(forbid, as_tuple=False).squeeze(1)
                                    scores[rows, rt[rows]] = -1e9

                            top_scores, top_tails = torch.topk(scores, k=self.elastic_k_soft, dim=1)
                            probs = torch.softmax(top_scores / self.elastic_tau, dim=1)
                            stats_soft_candidates += int(top_tails.numel())

                            if self.elastic_th_prob is not None:
                                keep = probs.max(dim=1).values >= self.elastic_th_prob
                                if not keep.any():
                                    stats_soft_u_filtered += int(probs.size(0))
                                    probs = None
                                else:
                                    stats_soft_u_filtered += int((~keep).sum().item())
                                    dead_batch = dead_batch[keep]
                                    dead_weight = dead_weight[keep]
                                    top_tails = top_tails[keep]
                                    probs = probs[keep]

                            if probs is not None:
                                stats_soft_u_kept += int(probs.size(0))
                                lam_i = self._elastic_lambda_at_depth(hop_index)
                                if lam_i > 0:
                                    contrib = dead_weight.unsqueeze(1) * (lam_i * probs)
                                    flat = dead_batch.unsqueeze(1) * self.graph.entity_size + top_tails
                                    xS_soft_flat = torch.zeros((B * self.graph.entity_size,), device=device)
                                    xS_soft_flat.scatter_add_(0, flat.reshape(-1), contrib.reshape(-1))
                                    xS_soft = xS_soft_flat.view(B, self.graph.entity_size).transpose(0, 1).unsqueeze(-1)

                    xS_total = xS_hard + xS_soft
                    overlap = (xH_hard > 0) & (xS_total > 0)
                    stats_overlap_upgraded += int(overlap.sum().item())
                    xH_child = xH_hard + xS_total * overlap.float()
                    xS_child = xS_total * (~overlap).float()

                    stack.append((child_id, hop_index, xH_child, xS_child))

            self._elastic_log_calls = getattr(self, "_elastic_log_calls", 0) + 1
            log_every = max(getattr(self, "elastic_log_every", 0), 0)
            log_first_n = max(getattr(self, "elastic_log_first_n", 0), 0)
            should_log = (self._elastic_log_calls <= log_first_n) or (log_every and self._elastic_log_calls % log_every == 0)
            if should_log:
                logging.info(
                    "ElasticGrounding call=%d query_r=%d visited=%d propagate=%d deadH=%d selectedH=%d softU=%d filteredU=%d softCand=%d droppedSoftDead=%d upgraded=%d",
                    self._elastic_log_calls,
                    int(query_r),
                    int(visited_nodes),
                    int(propagate_calls),
                    int(stats_dead_h_total),
                    int(stats_dead_h_selected),
                    int(stats_soft_u_kept),
                    int(stats_soft_u_filtered),
                    int(stats_soft_candidates),
                    int(stats_soft_dead_dropped),
                    int(stats_overlap_upgraded),
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
            self.rule_features = self.rule_features.to(device)
            self.rule_masks = self.rule_masks.to(device)

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
        if candidate_strength.sum().item() == 0:
            score = candidate_strength + self.bias.unsqueeze(0)
            return score, candidate_mask, candidate_strength


        candidate_set = torch.nonzero(candidate_strength.view(-1), as_tuple=True)[0]

        rule_index = torch.tensor(rule_index, dtype=torch.long, device=device)
        rule_count = torch.stack(rule_count, dim=0)

        rule_count = rule_count.reshape(rule_index.size(0), -1)[:, candidate_set]

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
