
import logging, os, datetime
import argparse
import torch
from data import KnowledgeGraph, TrainDataset, ValidDataset, TestDataset, RuleDataset, KGETrainDataset
from model import RulE
from utils import load_config, save_config, set_logger, set_seed
from trainer import GroundTrainer, PreTrainer
from dual_reranker import HeadGNNReranker, DualRerankConfig, DualRerankTrainer

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

    # elastic grounding (bridge-once soft fallback)
    parser.add_argument('--elastic_grounding', action='store_true', default=False)
    parser.add_argument('--elastic_k_soft', default=16, type=int)
    parser.add_argument('--elastic_u_soft_cap', default=32, type=int)
    parser.add_argument('--elastic_tau', default=2.0, type=float)
    parser.add_argument('--elastic_lambda_base', default=0.5, type=float)
    parser.add_argument('--elastic_position_lambda', action='store_true', default=True)
    parser.add_argument('--no_elastic_position_lambda', dest='elastic_position_lambda', action='store_false')
    parser.add_argument('--elastic_th_prob', default=None, type=float)
    parser.add_argument('--elastic_log_every', default=1000, type=int)

    # dual-context GNN reranker (two-stage)
    parser.add_argument('--dual_rerank', action='store_true', default=False)
    parser.add_argument('--dual_rerank_train', action='store_true', default=False)
    parser.add_argument('--dual_rerank_k', default=256, type=int)
    parser.add_argument('--dual_rerank_beta', default=0.5, type=float)
    parser.add_argument('--dual_rerank_dim', default=128, type=int)
    parser.add_argument('--dual_rerank_lr', default=0.001, type=float)
    parser.add_argument('--dual_rerank_steps', default=2000, type=int)
    parser.add_argument('--dual_rerank_batch', default=16, type=int)
    parser.add_argument('--dual_rerank_layers', default=2, type=int)
    # backward-compatible alias
    parser.add_argument('--dual_rerank_hops', dest='dual_rerank_layers', default=2, type=int)
    parser.add_argument('--dual_rerank_node_topk', default=128, type=int)
    parser.add_argument('--dual_rerank_rule_gamma', default=0.0, type=float)
    parser.add_argument('--dual_rerank_rule_eps', default=1e-3, type=float)
    parser.add_argument('--dual_rerank_use_rule_weight', action='store_true', default=True)
    parser.add_argument('--no_dual_rerank_use_rule_weight', dest='dual_rerank_use_rule_weight', action='store_false')
    parser.add_argument('--dual_rerank_tau', default=1.0, type=float)
    parser.add_argument('--dual_rerank_edge_topk', default=-1, type=int)
    parser.add_argument('--dual_rerank_eval_every', default=1000, type=int)
    parser.add_argument('--dual_rerank_log_every', default=200, type=int)

    return parser.parse_args(args)

def main():
    args = parse_args()

    # read the given config
    if args.init_checkpoint_config:
        args = load_config(args.init_checkpoint_config)
        args = args[0]

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

    RulE_model = RulE(
        graph,
        args.p_norm,
        args.mlp_rule_dim,
        args.gamma_fact,
        args.gamma_rule,
        args.hidden_dim,
        device,
        args.data_path,
    )
    RulE_model.set_rules(rules)
    RulE_model.configure_elastic_grounding(
        enabled=getattr(args, "elastic_grounding", False),
        k_soft=getattr(args, "elastic_k_soft", 16),
        u_soft_cap=getattr(args, "elastic_u_soft_cap", 32),
        tau=getattr(args, "elastic_tau", 2.0),
        lambda_base=getattr(args, "elastic_lambda_base", 0.5),
        position_lambda=getattr(args, "elastic_position_lambda", True),
        th_prob=getattr(args, "elastic_th_prob", None),
        log_every=getattr(args, "elastic_log_every", 1000),
    )

    
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
    RulE_model.load_state_dict(checkpoint['model'],strict=False)
    
    
    logging.info('Test the results of pre-training')
    
    valid_mrr = pre_trainer.evaluate('valid', expectation=True)
    test_mrr = pre_trainer.evaluate('test', expectation=True)

    # Dual-context reranker as a grounding replacement (two-stage: KGE retrieval + GNN re-ranking)
    if getattr(args, "dual_rerank", False) or getattr(args, "dual_rerank_train", False):
        logging.info(">>>>> DualRerank: Enabled (replaces grounding stage)")
        RulE_model.dual_reranker = HeadGNNReranker(
            graph=graph,
            entity_embedding=RulE_model.entity_embedding,
            relation_embedding=RulE_model.relation_embedding,
            num_relations=graph.relation_size,
            dim=int(getattr(args, "dual_rerank_dim", 128)),
            layers=int(getattr(args, "dual_rerank_layers", 2)),
            node_topk=int(getattr(args, "dual_rerank_node_topk", 128)),
            rule_gamma=float(getattr(args, "dual_rerank_rule_gamma", 0.0)),
            rule_eps=float(getattr(args, "dual_rerank_rule_eps", 1e-3)),
            use_rule_weight=bool(getattr(args, "dual_rerank_use_rule_weight", True)),
            tau=float(getattr(args, "dual_rerank_tau", 1.0)),
            edge_topk=int(getattr(args, "dual_rerank_edge_topk", -1)),
        ).to(device)

        # Provide explicit rules (no trie) to guide sampling.
        if hasattr(RulE_model, "relation2rules"):
            rule_weight = None
            if bool(getattr(args, "dual_rerank_use_rule_weight", True)) and hasattr(RulE_model, "eval_compute_rule_weight"):
                try:
                    RulE_model.eval_compute_rule_weight(device)
                    rule_weight = getattr(RulE_model, "rules_weight_emb", None)
                    logging.info("DualRerank: computed rule weights for sampling prior.")
                except Exception as e:
                    logging.warning("DualRerank: failed to compute rule weights; fallback to uniform. err=%s", str(e))
            RulE_model.dual_reranker.set_rules(RulE_model.relation2rules, rule_weight)

        cfg = DualRerankConfig(
            k=int(getattr(args, "dual_rerank_k", 256)),
            beta=float(getattr(args, "dual_rerank_beta", 0.5)),
            dim=int(getattr(args, "dual_rerank_dim", 128)),
            lr=float(getattr(args, "dual_rerank_lr", 0.001)),
            steps=int(getattr(args, "dual_rerank_steps", 2000)),
            batch_size=int(getattr(args, "dual_rerank_batch", 16)),
            layers=int(getattr(args, "dual_rerank_layers", 2)),
            node_topk=int(getattr(args, "dual_rerank_node_topk", 128)),
            rule_gamma=float(getattr(args, "dual_rerank_rule_gamma", 0.0)),
            rule_eps=float(getattr(args, "dual_rerank_rule_eps", 1e-3)),
            use_rule_weight=bool(getattr(args, "dual_rerank_use_rule_weight", True)),
            tau=float(getattr(args, "dual_rerank_tau", 1.0)),
            edge_topk=int(getattr(args, "dual_rerank_edge_topk", -1)),
            eval_every=int(getattr(args, "dual_rerank_eval_every", 1000)),
            log_every=int(getattr(args, "dual_rerank_log_every", 200)),
            base="kge",
        )
        dual_trainer = DualRerankTrainer(
            model=RulE_model,
            graph=graph,
            device=device,
            config=cfg,
            save_path=args.save_path,
        )
        if getattr(args, "dual_rerank_train", False):
            dual_trainer.train(alpha=float(getattr(args, "alpha", 3.0)), valid_set=valid_set, num_worker=args.cpu_num)
        else:
            loaded = dual_trainer.load_if_exists()
            if not loaded:
                raise FileNotFoundError(os.path.join(args.save_path, "dual_reranker.pt"))

        dual_trainer.evaluate_dataset(valid_set, split="valid", num_worker=args.cpu_num, expectation=True)
        dual_trainer.evaluate_dataset(test_set, split="test", num_worker=args.cpu_num, expectation=True)
        dual_trainer.evaluate_dataset(test_kge_set, split="test_kge", num_worker=args.cpu_num, expectation=True)
        return

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
    
    ground_trainer.train(args)
    
    # return test_mrr


if __name__ == '__main__':
    
    main()
