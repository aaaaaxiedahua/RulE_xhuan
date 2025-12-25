
import logging, os, datetime
import argparse
import torch
from data import KnowledgeGraph, TrainDataset, ValidDataset, TestDataset, RuleDataset, KGETrainDataset
from model import RulE
from utils import load_config, save_config, set_logger, set_seed
from trainer import GroundTrainer, PreTrainer
from topk_reasoner import IncrementalNeighborSampler, TopKReasoner

# torch.cuda.set_device(1)

def save_files(rules):
    with open('mined_rules.txt','w') as fw:
        for rule in rules:
            for relation in rule[0:-1]:
                fw.writelines(str(relation) + ' ')

            fw.writelines(str(rule[-1])+'\n')

def formatted_rules(_rules):
    rules = []
    
    for i, _rule in enumerate(_rules):
        rule = [i,len(_rule)]
        rule += _rule
        rules.append(rule)
    return rules

def parse_args(args=None):

    parser = argparse.ArgumentParser(
        description='RNNLogic',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument("--local_rank", type=int, default=0)
    # data path
    parser.add_argument('--data_path', default="../data/wn18rr", type=str, help='dataset path')
    parser.add_argument('--rule_file', default="../data/wn18rr/mined_rules.txt", type=str)
    # device 
    parser.add_argument('--cuda', action='store_true',default=False, help='use GPU')
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)

    parser.add_argument('--seed',default=800, type=int, help='seed')
    
    # pre train process (KGE + rulE)
    parser.add_argument('-b', '--batch_size', default=256, type=int)
    parser.add_argument('-n', '--negative_sample_size', default=256 , type=int)
    parser.add_argument('--rule_batch_size',default=128,type=int, help='rule batch size')
    parser.add_argument('--rule_negative_size',default=64,type=int)

    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g_f', '--gamma_fact', default=6, type=float, help='the triplet margin')
    parser.add_argument('-g_r', '--gamma_rule', default=5, type=float, help='the rule margin')
    parser.add_argument('--disable_adv', action='store_true',default=True, help='disable the adversarial negative sampling')
    # parser.add_argument('-adv', '--negative_adversarial_sampling', default=True, action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=0.5, type=float)
                            
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    parser.add_argument('-lr', '--learning_rate', default=0.00005, type=float)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    parser.add_argument('--g_warm_up_steps', default=None, type=int)
    parser.add_argument('--save_checkpoint_steps', default=10, type=int)
    parser.add_argument('--valid_steps', default=1000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--weight_rule',type=float,default=1)
    parser.add_argument('-reg', '--regularization', default=0, type=float)
    parser.add_argument('--max_steps', default=15000, type=int)
    parser.add_argument('--p_norm', default=2, type=int)

    # save path
    parser.add_argument('-init', '--init_checkpoint_config', default="../config/umls_config.json", type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)

    
    # grounding training process
  
    parser.add_argument('--mlp_rule_dim', default=100, type=int)
    parser.add_argument('--alpha', default=5.0, type=int, help='weight the KGE score')
    parser.add_argument('--smoothing', default=0.5, type=float)
    parser.add_argument('--batch_per_epoch', default=1000000, type=int)
    parser.add_argument('--print_every', default=1000, type=int)
    parser.add_argument('--g_batch_size', default=16, type=int)
    parser.add_argument('--g_lr', default=0.00005, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--num_iters', default=20, type=int)

    # top-k propagation reasoner (AdaProp-style)
    parser.add_argument('--use_topk_reasoner', action='store_true', default=True)
    parser.add_argument('--topk_layers', default=5, type=int)
    parser.add_argument('--topk_hidden_dim', default=64, type=int)
    parser.add_argument('--topk_attn_dim', default=8, type=int)
    parser.add_argument('--topk_topk', default=200, type=int)
    parser.add_argument('--topk_tau', default=0.0, type=float)
    parser.add_argument('--topk_dropout', default=0.1, type=float)
    parser.add_argument('--topk_act', default='relu', type=str)
    parser.add_argument('--topk_use_rule_semantic', action='store_true', default=True)
    parser.add_argument('--topk_use_kge', action='store_true', default=False)
    parser.add_argument('--topk_kge_alpha', default=1.0, type=float)
    parser.add_argument('--topk_epochs', default=0, type=int)
    parser.add_argument('--topk_batch_size', default=32, type=int)
    parser.add_argument('--topk_lr', default=0.001, type=float)
    parser.add_argument('--topk_weight_decay', default=0.0, type=float)
    parser.add_argument('--topk_eval_interval', default=1, type=int)
    return parser.parse_args(args)

def main():
    args = parse_args()

    # read the given config
    if args.init_checkpoint_config:
        cfg = load_config(args.init_checkpoint_config)[0]
        for k, v in cfg.items():
            setattr(args, k, v)

    # wandb.init(project='RulE',group='RotatE', name = args.save_path, config=args)
    if args.save_path is None:
        args.save_path = os.path.join('../outputs', datetime.now().strftime('%Y%m-%d%H-%M%S'))
    # else:
    #     args.save_path = '../outputs/'+ args.save_path
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    save_config(args)

    set_logger(args.save_path)
    set_seed(args.seed)



    # for grounding dataset
    graph = KnowledgeGraph(args.data_path)
    train_set = TrainDataset(graph, args.g_batch_size)
    valid_set = ValidDataset(graph, args.g_batch_size)
    test_set = TestDataset(graph, args.g_batch_size)
    test_kge_set = TestDataset(graph, 16)
    ruleset = RuleDataset(graph.relation_size, args.rule_file, args.rule_negative_size)

    rules = [rule[0] for rule in ruleset.rules]
    
    
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    RulE_model = RulE(graph, args.p_norm, args.mlp_rule_dim, args.gamma_fact, args.gamma_rule, args.hidden_dim, device, args.data_path)
    RulE_model.set_rules(rules)

    
    # For pre-training 

    pre_trainer = PreTrainer(
        graph=graph,
        model=RulE_model,
        valid_set=valid_set,
        test_set=test_set,
        # tripletset=kge_train_set,
        ruleset=ruleset,
        expectation=True,
        device = device,
        num_worker=args.cpu_num
        
    )
    
    # checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
    # RulE_model.load_state_dict(checkpoint['model'])


    # valid_mrr = pre_trainer.evaluate('valid', expectation=True)
    # test_mrr = pre_trainer.evaluate('test', expectation=True)
    
    pre_trainer.train(args)
    
    
    logging.info('Finishing pre-training!')

    print("loading RulE trainer......")

    # load rule embedding and KGE embedding

    checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
    RulE_model.load_state_dict(checkpoint['model'])
    
    
    logging.info('Test the results of pre-training')
    
    valid_mrr = pre_trainer.evaluate('valid', expectation=True)
    test_mrr = pre_trainer.evaluate('test', expectation=True)

    # RulE_model.add_param()

    # checkpoint = torch.load(os.path.join(args.save_path, 'grounding.pt'))
    # RulE_model.load_state_dict(checkpoint['model'])

    ground_trainer = GroundTrainer(
        model=RulE_model,
        args = args,
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        test_kge_set = test_kge_set,
        device=device,
        num_worker=args.cpu_num
    )

    # valid_mrr = ground_trainer.evaluate('valid', expectation=True)
    # test_mrr = ground_trainer.evaluate('test', expectation=True)
    
    # args.g_batch_size = 32
    
    # ---- Original grounding training (disabled) ----
    # ground_trainer.train(args)

    if args.use_topk_reasoner:
        logging.info('Running top-k propagation reasoner (AdaProp-style)')

        sampler = IncrementalNeighborSampler(
            triples=graph.ground_train_facts,
            n_ent=graph.entity_size,
            n_rel=graph.relation_size,
            device=device,
        )

        with torch.no_grad():
            RulE_model.eval_compute_rule_weight(device)

        rules_weight_emb = RulE_model.rules_weight_emb.detach() if args.topk_use_rule_semantic else None

        reasoner = TopKReasoner(
            n_ent=graph.entity_size,
            n_rel=graph.relation_size,
            hidden_dim=args.topk_hidden_dim,
            attn_dim=args.topk_attn_dim,
            n_layer=args.topk_layers,
            n_node_topk=args.topk_topk,
            tau=args.topk_tau,
            dropout=args.topk_dropout,
            act=args.topk_act,
            use_rule_semantic=args.topk_use_rule_semantic,
            rule_vec_dim=rules_weight_emb.size(-1) if args.topk_use_rule_semantic else None,
        ).to(device)

        reasoner.set_kge_fusion(args.topk_use_kge, alpha=args.topk_kge_alpha)

        def kge_score_candidates(q_sub, q_rel, nodes):
            # nodes: [N,2] with (batch_idx, ent_id); return scores aligned to nodes order
            batch_idx = nodes[:, 0]
            tails = nodes[:, 1]
            head = RulE_model.entity_embedding(q_sub)[batch_idx]
            rel = RulE_model.relation_embedding(q_rel % graph.relation_size)[batch_idx]
            flag = torch.pow(-1, (q_rel // graph.relation_size)).unsqueeze(-1)[batch_idx]
            rel = rel * flag
            tail = RulE_model.entity_embedding(tails)
            # RotatE expects [B,*,dim]; implement candidate score directly
            re_head, im_head = torch.chunk(head, 2, dim=-1)
            re_tail, im_tail = torch.chunk(tail, 2, dim=-1)
            phase_relation = rel / (RulE_model.embedding_range_fact.item() / RulE_model.pi)
            re_relation = torch.cos(phase_relation)
            im_relation = torch.sin(phase_relation)
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail
            score = torch.stack([re_score, im_score], dim=0).norm(dim=0)
            score = RulE_model.gamma_fact.item() - score.sum(dim=-1)
            return score

        def evaluate_split(split, dataset, expectation: bool = True):
            reasoner.eval()
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.cpu_num)
            mrr_sum = 0.0
            n_q = 0
            for batch in dataloader:
                all_h, all_r, all_t, flag = batch
                all_h = all_h.squeeze(0).to(device)
                all_r = all_r.squeeze(0).to(device)
                all_t = all_t.squeeze(0).to(device)
                flag = flag.squeeze(0).to(device)

                logits = reasoner(
                    subs=all_h,
                    rels=all_r,
                    sampler=sampler,
                    relation2rules=RulE_model.relation2rules if args.topk_use_rule_semantic else None,
                    rules_weight_emb=rules_weight_emb,
                    kge_score_fn=kge_score_candidates if args.topk_use_kge else None,
                )

                for i in range(all_t.size(0)):
                    t = all_t[i].item()
                    val = logits[i, t]
                    L = (logits[i][flag[i]] > val).sum().item() + 1
                    H = (logits[i][flag[i]] >= val).sum().item() + 2
                    if expectation:
                        denom = max(1, H - L)
                        mrr_sum += sum((1.0 / r) for r in range(L, H)) / denom
                    else:
                        mrr_sum += 1.0 / max(1, H - 1)
                    n_q += 1
            logging.info('TopKReasoner %s MRR: %.6f (%d queries)', split, mrr_sum / max(1, n_q), n_q)
            return mrr_sum / max(1, n_q)

        if args.topk_epochs > 0:
            logging.info('Training top-k reasoner for %d epochs', args.topk_epochs)
            optimizer = torch.optim.Adam(reasoner.parameters(), lr=args.topk_lr, weight_decay=args.topk_weight_decay)
            triples = graph.ground_train_facts
            for epoch in range(1, args.topk_epochs + 1):
                reasoner.train()
                perm = torch.randperm(len(triples))
                total_loss = 0.0
                n_batch = 0
                for start in range(0, len(triples), args.topk_batch_size):
                    idx = perm[start:start + args.topk_batch_size].tolist()
                    batch = [triples[i] for i in idx]
                    subs = [h for h, _, _ in batch]
                    rels = [r for _, r, _ in batch]
                    tails = torch.as_tensor([t for _, _, t in batch], dtype=torch.long, device=device)

                    logits = reasoner(
                        subs=subs,
                        rels=rels,
                        sampler=sampler,
                        relation2rules=RulE_model.relation2rules if args.topk_use_rule_semantic else None,
                        rules_weight_emb=rules_weight_emb,
                        kge_score_fn=kge_score_candidates if args.topk_use_kge else None,
                    )

                    # AdaProp-style negative log-likelihood with max-trick for stability.
                    pos_scores = logits[torch.arange(tails.size(0), device=device), tails]
                    max_n = torch.max(logits, 1, keepdim=True)[0]
                    loss = torch.sum(-pos_scores + max_n.squeeze(1) + torch.log(torch.sum(torch.exp(logits - max_n), 1)))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # AdaProp: avoid NaN/Inf by resetting invalid parameters.
                    for p in reasoner.parameters():
                        X = p.data
                        bad = ~torch.isfinite(X)
                        if bad.any():
                            X[bad] = torch.rand(int(bad.sum().item()), device=X.device, dtype=X.dtype)
                            p.data.copy_(X)
                    total_loss += loss.item()
                    n_batch += 1

                logging.info('TopKReasoner epoch %d loss %.6f', epoch, total_loss / max(1, n_batch))
                if args.topk_eval_interval > 0 and epoch % args.topk_eval_interval == 0:
                    evaluate_split('valid', valid_set)

        # evaluation on valid/test using existing batching datasets
        reasoner.eval()
        evaluate_split('valid', valid_set)
        evaluate_split('test', test_set)

    # return test_mrr


if __name__ == '__main__':
    
    main()
