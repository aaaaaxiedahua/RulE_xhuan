
from utils import *
import torch
from torch import nn

from torch.utils.data import DataLoader
from itertools import islice
from data import Iterator, RuleDataset, KGETrainDataset, BidirectionalOneShotIterator
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

class PreTrainer(object):

    def __init__(self, graph, model, valid_set, test_set, ruleset, expectation, device, num_worker=0):
        """
        预训练器初始化

        参数：
            graph: 知识图谱
            model: RulE模型（可能包含策略网络）
            valid_set: 验证集
            test_set: 测试集
            ruleset: 规则数据集
            expectation: 是否使用期望排名
            device: 计算设备
            num_worker: 数据加载worker数量
        """

        self.num_worker = num_worker
        self.device = device

        if self.device.type == "cuda":
            model = model.cuda(self.device)

        self.graph = graph
        self.model = model
        self.valid_set = valid_set
        self.test_set = test_set

        self.RuleSet = ruleset
        self.expectation = expectation
        

    
    def train(self, args):
        
        # Set training configuration
       
        current_learning_rate = float(args.learning_rate)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=float(args.learning_rate), 
            weight_decay=float(args.weight_decay)
            )
            
        
        triplets_dataloader_head = DataLoader(
            KGETrainDataset(
                triples=self.graph.train_facts,
                nentity=self.graph.entity_size, 
                nrelation=self.graph.relation_size,
                negative_sample_size=args.negative_sample_size,
                mode = 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=KGETrainDataset.collate_fn
        )

        triplets_dataloader_tail = DataLoader(
            KGETrainDataset(
                triples=self.graph.train_facts,
                nentity=self.graph.entity_size, 
                nrelation=self.graph.relation_size,
                negative_sample_size=args.negative_sample_size,
                mode = 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=KGETrainDataset.collate_fn
        )


        rules_dataloader = DataLoader(
            self.RuleSet,
            batch_size=args.rule_batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=RuleDataset.collate_fn)

        # RulE-SSRL拆分方案: 预训练阶段只训练KGE+Rule，不训练策略网络
        # 策略网络在独立的Phase 2阶段训练
        logging.info('预训练阶段: 只训练KGE和Rule嵌入（策略网络在Phase 2单独训练）')

        self.triplets_iterator = BidirectionalOneShotIterator(triplets_dataloader_head, triplets_dataloader_tail)
        self.rules_iterator = Iterator(rules_dataloader)

        logging.info('>>>>> ruleE: Pre-training')
        training_logs = []
        best_mrr = 0.0

        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

        for step in range(0, args.max_steps + 1):

            log = self.train_step(optimizer, self.triplets_iterator, self.rules_iterator, args)
            
            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
                if args.disable_adv:
                    args.adversarial_temperature = 0
            
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                mrr = self.evaluate("valid", self.expectation )
                if mrr > best_mrr:
                    save_model(self.model,optimizer, args)
                    best_mrr = mrr
                # save_model(self.model,optimizer, args)


    def train_step(self, optimizer, triplets_iterator, rules_iterator, args):
        '''
        单步训练：应用反向传播并返回损失

        RulE-SSRL拆分方案: 预训练只计算KGE+Rule损失，不计算策略损失
        '''
        
        # self.model.rule_emb.requires_grad = False
        # # self.model.rule_emb.weight.data = self.model.rule_emb.weight / self.model.rule_emb.weight.norm(dim=-1,keepdim=True)
        # self.model.rule_emb.weight.data  = F.normalize( self.model.rule_emb.weight.data , p=2, dim=-1)
        # self.model.rule_emb.requires_grad = True

        model = self.model
        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode= next(triplets_iterator)
        positive_rule, negative_idx, negative_rule, mode_rule, rule_mask = next(rules_iterator)

        if self.device.type == "cuda":
            positive_sample = positive_sample.cuda(self.device)
            negative_sample = negative_sample.cuda(self.device)
            positive_rule = positive_rule.cuda(self.device)
            negative_idx = negative_idx.cuda(self.device)
            negative_rule = negative_rule.cuda(self.device)
            rule_mask = rule_mask.cuda(self.device)
            subsampling_weight = subsampling_weight.cuda(self.device)

        negative_fact_score, _ = model.compute_KGE((positive_sample, negative_sample), mode) 
        negative_rule_score = model.compute_ruleE((positive_rule,  rule_mask, negative_idx, negative_rule), mode=mode_rule) 
        # print(args.adversarial_temperature)
        negative_fact_score = (F.softmax(negative_fact_score * args.adversarial_temperature, dim = 1).detach() 
                            * F.logsigmoid(-negative_fact_score)).sum(dim = 1)
        
        negative_rule_score = (F.softmax(negative_rule_score * args.adversarial_temperature, dim = 1).detach() 
                            * F.logsigmoid(-negative_rule_score)).sum(dim = 1)
        
        
        # negative_rule_score = F.logsigmoid(-negative_rule_score).mean(dim = 1)


        positive_fact_score, ent = model.compute_KGE(positive_sample)
        positive_rule_score = model.compute_ruleE((positive_rule,rule_mask))


        positive_fact_score = F.logsigmoid(positive_fact_score).squeeze(dim = 1)
        positive_rule_score = F.logsigmoid(positive_rule_score)

        negative_rule_score_weight = negative_rule_score 
        positive_rule_score_weight = positive_rule_score 

        if args.uni_weight:
            positive_fact_loss = - positive_fact_score.mean() 
            negative_fact_loss = - negative_fact_score.mean()
        else:
            positive_fact_loss = - (subsampling_weight * positive_fact_score).sum()/subsampling_weight.sum()            
            negative_fact_loss = - (subsampling_weight * negative_fact_score).sum()/subsampling_weight.sum() 

        
        positive_rule_loss = - positive_rule_score_weight.mean() * args.weight_rule
        negative_rule_loss = - negative_rule_score_weight.mean() * args.weight_rule


        loss_fact = (positive_fact_loss + negative_fact_loss)/2
        loss_rule = (positive_rule_loss + negative_rule_loss)/2

        # RulE-SSRL拆分方案: 预训练只有KGE + Rule损失
        # 策略网络在Phase 2独立训练
        loss = loss_fact + loss_rule

        # loss = loss_fact
        # loss = loss_rule

        # wandb.log({'train/loss_fact':loss_fact, 'train/loss_rule':loss_rule})
        # wandb.log({ 'train/loss_rule':loss_rule})
        if args.regularization:
            # Use regularization
            regularization = args.regularization * (
                ent[0].norm(p=2)**2 +
                ent[1].norm(p=2)**2
            ) / ent[0].shape[0]
            # regularization_rule = args.regularization * (
            #     rule.norm(p=2) ** 2
            # ) / rule[0].shape[0]
            loss = loss + regularization 
        else:
            regularization = torch.tensor([0])

        loss.backward()

        optimizer.step()

        log = {

            'positive_fact_loss': positive_fact_loss.item(),
            'negative_fact_loss': negative_fact_loss.item(),
            'positive_rule_loss': positive_rule_loss.item(),
            'negative_rule_loss': negative_rule_loss.item(),
            'regularization': regularization.item(),
            'loss': loss.item()
        }

        return log

    @torch.no_grad()
    def evaluate(self, split, expectation=True):
        
        logging.info('>>>>> RuleE emb: Evaluating on {}'.format(split))
        
        test_set = getattr(self, "%s_set" % split)
        # test_set = self.test_set_data
        dataloader = DataLoader(test_set, batch_size=1, num_workers=self.num_worker)
        model = self.model

        model.eval()
        concat_logits = []
        concat_all_h = []
        concat_all_r = []
        concat_all_t = []
        concat_flag = []
        # concat_mask = []
        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            flag = flag.squeeze(0)
            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                all_t = all_t.cuda(device=self.device)
                flag = flag.cuda(device=self.device)

            KGE_score = model.compute_g_KGE(all_h,all_r)
            
            logits = KGE_score

            concat_logits.append(logits)
            concat_all_h.append(all_h)
            concat_all_r.append(all_r)
            concat_all_t.append(all_t)
            concat_flag.append(flag)

        
        concat_logits = torch.cat(concat_logits, dim=0)
        concat_all_h = torch.cat(concat_all_h, dim=0)
        concat_all_r = torch.cat(concat_all_r, dim=0)
        concat_all_t = torch.cat(concat_all_t, dim=0)
        concat_flag = torch.cat(concat_flag, dim=0)

        
        ranks = []
        for k in range(concat_all_t.size(0)):
            h = concat_all_h[k]
            r = concat_all_r[k]
            t = concat_all_t[k]
            val = concat_logits[k, t]

            L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
            H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
            ranks += [[h, r, t, L, H]]
        ranks = torch.tensor(ranks, dtype=torch.long, device=self.device)
            
        query2LH = dict()
        for h, r, t, L, H in ranks.data.cpu().numpy().tolist():
            query2LH[(h, r, t)] = (L, H)
            
        hit1, hit3, hit10, mr, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        for (L, H) in query2LH.values():
            if expectation:
                for rank in range(L, H):
                    if rank <= 1:
                        hit1 += 1.0 / (H - L)
                    if rank <= 3:
                        hit3 += 1.0 / (H - L)
                    if rank <= 10:
                        hit10 += 1.0 / (H - L)
                    mr += rank / (H - L)
                    mrr += 1.0 / rank / (H - L)
            else:
                rank = H - 1
                if rank <= 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 10:
                    hit10 += 1
                mr += rank
                mrr += 1.0 / rank
        
        hit1 /= len(ranks)
        hit3 /= len(ranks)
        hit10 /= len(ranks)
        mr /= len(ranks)
        mrr /= len(ranks)

        
        logging.info('Data : {}'.format(len(query2LH)))
        logging.info('Hit1 : {:.6f}'.format(hit1))
        logging.info('Hit3 : {:.6f}'.format(hit3))
        logging.info('Hit10: {:.6f}'.format(hit10))
        logging.info('MR   : {:.6f}'.format(mr))
        logging.info('MRR  : {:.6f}'.format(mrr))

        return mrr

    def load(self, checkpoint, load_optimizer=True):
        """
        Load a checkpoint from file.
        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
        """
        
        logging.info("Load checkpoint from %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(state["model"])

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)


    def save(self, checkpoint):
        """
        Save checkpoint to file.
        Parameters:
            checkpoint (file-like): checkpoint file
        """
       
        logging.info("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        if self.rank == 0:
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }
            torch.save(state, checkpoint)




class GroundTrainer(object):
    
    def __init__(self, model, args, train_set, valid_set, test_set,test_kge_set, device, num_worker=0):
        
        self.num_worker = num_worker

        
        self.device = device
        if self.device.type == "cuda":
            model = model.cuda(self.device)

        self.args = args
        self.model = model
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.test_kge_set = test_kge_set
        
    def train(self, args):
        
        # fix the parameters of pre-training

        self.model.entity_embedding.weight.requires_grad = False
        self.model.relation_embedding.weight.requires_grad = False
        self.model.rule_emb.weight.requires_grad = False


        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=float(args.g_lr), 
            weight_decay=float(args.weight_decay))


        self.train_set.make_batches()
        
        train_dataloader = DataLoader(self.train_set, 1, num_workers=self.num_worker)
        
        self.model.eval_compute_rule_weight(self.device)


        
        logging.info('>>>>> RulE: Grounding-Training')
        

        best_valid_mrr = 0.0 
        test_mrr = 0.0

        warm_up_steps = args.num_iters // 2
        current_learning_rate = float(args.g_lr)

        for k in range(args.num_iters):

            
            logging.info('-------------------------')
            logging.info('| Iteration: {}/{}'.format(k + 1, args.num_iters))
            logging.info('-------------------------')
        
            # if k >= warm_up_steps:

            #     current_learning_rate = current_learning_rate / 10
            #     logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, k))
            #     optim = torch.optim.Adam(
            #         filter(lambda p: p.requires_grad, predictor.parameters()), 
            #         lr=current_learning_rate
            #     )
            #     warm_up_steps = warm_up_steps * 3

            self.train_step( optimizer, train_dataloader, args.batch_per_epoch, args.smoothing, args.print_every, args)
            valid_mrr_iter = self.evaluate('valid', args.alpha, expectation=True)
            # test_mrr_iter = self.evaluate('test', args.alpha, expectation=True)
            # test_mrr_iter = self.evaluate_t('test_kge', args.alpha, expectation=True)
            

            if valid_mrr_iter > best_valid_mrr:
                best_valid_mrr = valid_mrr_iter
                # test_mrr = test_mrr_iter
                self.save(args, os.path.join(args.save_path, 'grounding.pt'))
        

        logging.info('-------------------------')
        logging.info('| Final Test MRR: {:.6f}'.format(test_mrr))
        logging.info('-------------------------')

        checkpoint = torch.load(os.path.join(self.args.save_path, 'grounding.pt'))
        self.model.load_state_dict(checkpoint['model'])
        
        test_mrr_iter = self.evaluate('valid', args.alpha, expectation=True)
        test_mrr_iter = self.evaluate('test', args.alpha, expectation=True)
        test_mrr_iter = self.evaluate_t('test_kge', args.alpha, expectation=True)


       
       

    def train_step(self, optimizer, train_dataloader, batch_per_epoch, smoothing, print_every, args):
        
        batch_per_epoch = batch_per_epoch or len(train_dataloader)
        model = self.model
        
        model.train()

        total_loss = 0.0
        total_size = 0.0

        for batch_id, batch in enumerate(islice(train_dataloader, batch_per_epoch)):
            # 归一化

            # self.model.beta.requires_grad = False
            # self.model.beta.data = self.model.beta / self.model.beta.sum(dim=-1,keepdim=True)
            # self.model.beta.requires_grad = True
            
            all_h, all_r, all_t, target, edges_to_remove = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            target = target.squeeze(0)
            edges_to_remove = edges_to_remove.squeeze(0)
            target_t = torch.nn.functional.one_hot(all_t, self.train_set.graph.entity_size)
            
            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                all_t = all_t.cuda(device=self.device)
                target = target.cuda(device=self.device)
                edges_to_remove = edges_to_remove.cuda(device=self.device)
                target_t = target_t.cuda(device=self.device)

            target = target * smoothing + target_t * (1 - smoothing)
            
            grounding_rule_score, mask = model(all_h, all_r, edges_to_remove)
            
            if mask.sum().item() != 0:
                rule_logits = (torch.softmax(grounding_rule_score, dim=1) + 1e-8).log()
                
                loss = -(rule_logits[mask] * target[mask]).sum() / torch.clamp(target[mask].sum(), min=1)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                total_size += mask.sum().item()
            
            if (batch_id + 1) % print_every == 0:
                
                
                logging.info('loss:    {} {} {:.6f} {:.1f}'.format(batch_id + 1, len(train_dataloader), loss, total_size / print_every))
                
                total_loss = 0.0
                total_size = 0.0
                self.save(args, os.path.join(args.save_path, 'grounding.pt'))
        

    @torch.no_grad()
    def evaluate(self, split, alpha=3.0, expectation=True):

        logging.info('>>>>> Predictor: Evaluating on {}'.format(split))
        test_set = getattr(self, "%s_set" % split)
        
        dataloader = DataLoader(test_set, 1, num_workers=self.num_worker)
        model = self.model

        model.eval()
        concat_logits = []
        concat_all_h = []
        concat_all_r = []
        concat_all_t = []
        concat_flag = []
        concat_mask = []
        
        for batch in tqdm(dataloader):

            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            flag = flag.squeeze(0)

            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                all_t = all_t.cuda(device=self.device)
                flag = flag.cuda(device=self.device)

            # logits, mask = model.forward_weight(all_h, all_r, None)
            logits, mask = model(all_h, all_r, None)

            # kge_score = model.compute_g_KGE(all_h,all_r)
            # logits += alpha * kge_score

            concat_logits.append(logits)
            concat_all_h.append(all_h)
            concat_all_r.append(all_r)
            concat_all_t.append(all_t)
            concat_flag.append(flag)
            concat_mask.append(mask)
        
        concat_logits = torch.cat(concat_logits, dim=0)
        concat_all_h = torch.cat(concat_all_h, dim=0)
        concat_all_r = torch.cat(concat_all_r, dim=0)
        concat_all_t = torch.cat(concat_all_t, dim=0)
        concat_flag = torch.cat(concat_flag, dim=0)
        concat_mask = torch.cat(concat_mask, dim=0)
        
        ranks = []
        for k in range(concat_all_t.size(0)):
            h = concat_all_h[k]
            r = concat_all_r[k]
            t = concat_all_t[k]
            if concat_mask[k, t].item() == True:
                val = concat_logits[k, t]
                L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
                H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
            else:
                L = 1
                H = test_set.graph.entity_size + 1
            ranks += [[h, r, t, L, H]]
        ranks = torch.tensor(ranks, dtype=torch.long, device=self.device)
            
        query2LH = dict()
        for h, r, t, L, H in ranks.data.cpu().numpy().tolist():
            query2LH[(h, r, t)] = (L, H)
            
        hit1, hit3, hit10, mr, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        for (L, H) in query2LH.values():
            if expectation:
                for rank in range(L, H):
                    if rank <= 1:
                        hit1 += 1.0 / (H - L)
                    if rank <= 3:
                        hit3 += 1.0 / (H - L)
                    if rank <= 10:
                        hit10 += 1.0 / (H - L)
                    mr += rank / (H - L)
                    mrr += 1.0 / rank / (H - L)
            else:
                rank = H - 1
                if rank <= 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 10:
                    hit10 += 1
                mr += rank
                mrr += 1.0 / rank
            
        hit1 /= len(ranks)
        hit3 /= len(ranks)
        hit10 /= len(ranks)
        mr /= len(ranks)
        mrr /= len(ranks)

        
        logging.info('Data : {}'.format(len(query2LH)))
        logging.info('Hit1 : {:.6f}'.format(hit1))
        logging.info('Hit3 : {:.6f}'.format(hit3))
        logging.info('Hit10: {:.6f}'.format(hit10))
        logging.info('MR   : {:.6f}'.format(mr))
        logging.info('MRR  : {:.6f}'.format(mrr))
    
        
        return mrr


    @torch.no_grad()
    def evaluate_t(self, split, alpha=3.0, expectation=True):
       
        logging.info('>>>>> Predictor: Evaluating on {}'.format(split))
        test_set = getattr(self, "%s_set" % split)
        
        dataloader = DataLoader(test_set, 1, num_workers=self.num_worker)
        model = self.model

        model.eval()
        concat_logits = []
        concat_all_h = []
        concat_all_r = []
        concat_all_t = []
        concat_flag = []
        concat_mask = []
        
        for batch in tqdm(dataloader):

            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            flag = flag.squeeze(0)

            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                all_t = all_t.cuda(device=self.device)
                flag = flag.cuda(device=self.device)

            # logits, mask = model.forward_weight(all_h, all_r, None)
            logits, mask = model(all_h, all_r, None)

            kge_score = model.compute_g_KGE(all_h,all_r)
            
            logits = logits + alpha * kge_score

            concat_logits.append(logits)
            concat_all_h.append(all_h)
            concat_all_r.append(all_r)
            concat_all_t.append(all_t)
            concat_flag.append(flag)
            concat_mask.append(mask)
        
        concat_logits = torch.cat(concat_logits, dim=0)
        concat_all_h = torch.cat(concat_all_h, dim=0)
        concat_all_r = torch.cat(concat_all_r, dim=0)
        concat_all_t = torch.cat(concat_all_t, dim=0)
        concat_flag = torch.cat(concat_flag, dim=0)
        concat_mask = torch.cat(concat_mask, dim=0)
        
        ranks = []
        for k in range(concat_all_t.size(0)):
            h = concat_all_h[k]
            r = concat_all_r[k]
            t = concat_all_t[k]
            if concat_mask[k, t].item() == True:
                val = concat_logits[k, t]
                L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
                H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
            else:
                L = 1
                H = test_set.graph.entity_size + 1
            ranks += [[h, r, t, L, H]]
        ranks = torch.tensor(ranks, dtype=torch.long, device=self.device)
            
        query2LH = dict()
        for h, r, t, L, H in ranks.data.cpu().numpy().tolist():
            query2LH[(h, r, t)] = (L, H)
            
        hit1, hit3, hit10, mr, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        for (L, H) in query2LH.values():
            if expectation:
                for rank in range(L, H):
                    if rank <= 1:
                        hit1 += 1.0 / (H - L)
                    if rank <= 3:
                        hit3 += 1.0 / (H - L)
                    if rank <= 10:
                        hit10 += 1.0 / (H - L)
                    mr += rank / (H - L)
                    mrr += 1.0 / rank / (H - L)
            else:
                rank = H - 1
                if rank <= 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 10:
                    hit10 += 1
                mr += rank
                mrr += 1.0 / rank
            
        hit1 /= len(ranks)
        hit3 /= len(ranks)
        hit10 /= len(ranks)
        mr /= len(ranks)
        mrr /= len(ranks)

        
        logging.info('Data : {}'.format(len(query2LH)))
        logging.info('Hit1 : {:.6f}'.format(hit1))
        logging.info('Hit3 : {:.6f}'.format(hit3))
        logging.info('Hit10: {:.6f}'.format(hit10))
        logging.info('MR   : {:.6f}'.format(mr))
        logging.info('MRR  : {:.6f}'.format(mrr))
    
        
        return mrr


    def save(self, args, checkpoint):
        """
        Save checkpoint to file.
        Parameters:
            checkpoint (file-like): checkpoint file
        """
        # if comm.get_rank() == 0:
        logging.info("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
       
        state = {
            "model": self.model.state_dict(),
        }
       
        torch.save(state, checkpoint)

        g_rule_embedding = self.model.mlp_feature.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'g_rule_embedding'), 
            g_rule_embedding
        )


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


# ========== RulE-SSRL拆分方案: Phase 2 策略网络训练器 ==========

class PolicyTrainer(object):
    """
    策略网络训练器 (Phase 2)

    在预训练完成后，冻结所有嵌入，单独训练策略网络。
    """

    def __init__(self, model, graph, valid_set, test_set, device, num_worker=0):
        """
        初始化策略网络训练器

        参数：
            model: RulE模型（包含预训练好的嵌入和策略网络）
            graph: 知识图谱
            valid_set: 验证集
            test_set: 测试集
            device: 计算设备
            num_worker: 数据加载worker数量
        """
        self.model = model
        self.graph = graph
        self.valid_set = valid_set
        self.test_set = test_set
        self.device = device
        self.num_worker = num_worker

        if self.device.type == "cuda":
            self.model = self.model.cuda(self.device)

    def train(self, args):
        """
        策略网络训练主循环

        参数：
            args: 配置参数，需要包含：
                - policy_num_iters: 训练轮数
                - policy_batch_size: 批大小
                - policy_lr: 学习率
                - policy_log_steps: 日志频率
                - num_policy_samples: 验证时K采样次数
        """
        logging.info('>>>>> Phase 2: 策略网络训练')

        # Step 1: 冻结预训练嵌入
        logging.info('冻结预训练嵌入: entity, relation, rule')
        self.model.entity_embedding.requires_grad_(False)
        self.model.relation_embedding.requires_grad_(False)
        self.model.rule_emb.requires_grad_(False)

        # Step 2: 只优化策略网络参数
        policy_params = list(self.model.policy_network.parameters())
        logging.info('策略网络参数数量: {}'.format(sum(p.numel() for p in policy_params)))

        policy_lr = args.policy_lr if hasattr(args, 'policy_lr') else 0.0001
        optimizer = torch.optim.Adam(policy_params, lr=policy_lr)

        # Step 3: 数据加载
        from data import QueryDataset
        query_dataloader = DataLoader(
            QueryDataset(self.graph.train_facts),
            batch_size=args.policy_batch_size if hasattr(args, 'policy_batch_size') else 32,
            shuffle=True,
            num_workers=max(1, self.num_worker // 2),
            collate_fn=QueryDataset.collate_fn
        )

        logging.info('训练数据: {} 个三元组'.format(len(self.graph.train_facts)))
        logging.info('批大小: {}'.format(args.policy_batch_size if hasattr(args, 'policy_batch_size') else 32))

        # Step 4: 训练参数
        policy_num_iters = args.policy_num_iters if hasattr(args, 'policy_num_iters') else 20
        policy_log_steps = args.policy_log_steps if hasattr(args, 'policy_log_steps') else 100

        logging.info('训练轮数: {}'.format(policy_num_iters))

        # Step 5: 训练循环
        best_mrr = 0.0

        for iter_num in range(1, policy_num_iters + 1):
            logging.info('========== Iteration {}/{} =========='.format(iter_num, policy_num_iters))

            self.model.train()
            total_loss = 0.0
            batch_count = 0

            for batch_id, query_batch in enumerate(query_dataloader):

                if self.device.type == "cuda":
                    query_batch = query_batch.cuda(self.device)

                optimizer.zero_grad()

                # 计算策略损失
                loss = self.model.compute_policy_loss(query_batch)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # 日志
                if (batch_id + 1) % policy_log_steps == 0:
                    avg_loss = total_loss / batch_count
                    logging.info('Iter {}, Batch {}, Avg Loss: {:.6f}'.format(
                        iter_num, batch_id + 1, avg_loss))

            # 每轮结束后验证
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            logging.info('Iteration {} 完成, 平均损失: {:.6f}'.format(iter_num, avg_loss))

            # 验证
            logging.info('验证中...')
            mrr = self.evaluate('valid', args)

            if mrr > best_mrr:
                best_mrr = mrr
                self.save(args, os.path.join(args.save_path, 'policy_checkpoint'))
                logging.info('新的最佳MRR: {:.6f}, 已保存checkpoint'.format(mrr))

        logging.info('>>>>> Phase 2 完成! 最佳验证MRR: {:.6f}'.format(best_mrr))

        # 加载最佳模型
        checkpoint = torch.load(os.path.join(args.save_path, 'policy_checkpoint'))
        self.model.load_state_dict(checkpoint['model'])
        logging.info('已加载最佳策略网络checkpoint')

    @torch.no_grad()
    def evaluate(self, split, args, use_fusion=False):
        """
        使用策略网络推理进行评估

        参数：
            split: 'valid' 或 'test'
            args: 配置参数
            use_fusion: 是否融合KGE评分（测试阶段使用）

        返回：
            mrr: Mean Reciprocal Rank
        """
        logging.info('>>>>> PolicyTrainer: 评估 {} (策略网络推理)'.format(split))

        test_set = getattr(self, "%s_set" % split)
        dataloader = DataLoader(test_set, batch_size=1, num_workers=self.num_worker)

        self.model.eval()

        num_policy_samples = args.num_policy_samples if hasattr(args, 'num_policy_samples') else 10
        alpha = args.alpha if hasattr(args, 'alpha') else 0.5
        beta = args.beta if hasattr(args, 'beta') else 0.5

        concat_logits = []
        concat_all_h = []
        concat_all_r = []
        concat_all_t = []
        concat_flag = []

        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            flag = flag.squeeze(0)

            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                all_t = all_t.cuda(device=self.device)
                flag = flag.cuda(device=self.device)

            # 策略网络推理
            policy_scores, _ = self.model.forward_policy(all_h, all_r, num_samples=num_policy_samples)

            if use_fusion:
                # 融合KGE和策略网络评分
                kge_scores = self.model.compute_g_KGE(all_h, all_r)
                logits = alpha * kge_scores + beta * policy_scores
            else:
                logits = policy_scores

            concat_logits.append(logits)
            concat_all_h.append(all_h)
            concat_all_r.append(all_r)
            concat_all_t.append(all_t)
            concat_flag.append(flag)

        concat_logits = torch.cat(concat_logits, dim=0)
        concat_all_h = torch.cat(concat_all_h, dim=0)
        concat_all_r = torch.cat(concat_all_r, dim=0)
        concat_all_t = torch.cat(concat_all_t, dim=0)
        concat_flag = torch.cat(concat_flag, dim=0)

        # 计算排名
        ranks = []
        for k in range(concat_all_t.size(0)):
            h = concat_all_h[k]
            r = concat_all_r[k]
            t = concat_all_t[k]
            val = concat_logits[k, t]

            L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
            H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
            ranks += [[h, r, t, L, H]]
        ranks = torch.tensor(ranks, dtype=torch.long, device=self.device)

        query2LH = dict()
        for h, r, t, L, H in ranks.data.cpu().numpy().tolist():
            query2LH[(h, r, t)] = (L, H)

        # 计算指标
        hit1, hit3, hit10, mr, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        for (L, H) in query2LH.values():
            for rank in range(L, H):
                if rank <= 1:
                    hit1 += 1.0 / (H - L)
                if rank <= 3:
                    hit3 += 1.0 / (H - L)
                if rank <= 10:
                    hit10 += 1.0 / (H - L)
                mr += rank / (H - L)
                mrr += 1.0 / rank / (H - L)

        hit1 /= len(ranks)
        hit3 /= len(ranks)
        hit10 /= len(ranks)
        mr /= len(ranks)
        mrr /= len(ranks)

        logging.info('Data : {}'.format(len(query2LH)))
        logging.info('Hit1 : {:.6f}'.format(hit1))
        logging.info('Hit3 : {:.6f}'.format(hit3))
        logging.info('Hit10: {:.6f}'.format(hit10))
        logging.info('MR   : {:.6f}'.format(mr))
        logging.info('MRR  : {:.6f}'.format(mrr))

        return mrr

    @torch.no_grad()
    def test_with_fusion(self, args):
        """
        Phase 3: 使用KGE+策略网络融合评分进行测试

        参数：
            args: 配置参数，需要包含：
                - alpha: KGE评分权重
                - beta: 策略评分权重
                - num_policy_samples: K采样次数
        """
        logging.info('>>>>> Phase 3: 测试 (KGE + 策略网络 融合)')

        alpha = args.alpha if hasattr(args, 'alpha') else 0.5
        beta = args.beta if hasattr(args, 'beta') else 0.5
        logging.info('融合权重: alpha(KGE)={}, beta(Policy)={}'.format(alpha, beta))

        # 验证集融合测试
        logging.info('--- 验证集 ---')
        valid_mrr = self.evaluate('valid', args, use_fusion=True)

        # 测试集融合测试
        logging.info('--- 测试集 ---')
        test_mrr = self.evaluate('test', args, use_fusion=True)

        return valid_mrr, test_mrr

    def save(self, args, checkpoint_path):
        """保存checkpoint"""
        logging.info("保存checkpoint到 %s" % checkpoint_path)
        state = {
            "model": self.model.state_dict(),
        }
        torch.save(state, checkpoint_path)