
import logging, os, datetime
import argparse
import torch
from data import KnowledgeGraph, TrainDataset, ValidDataset, TestDataset, RuleDataset, KGETrainDataset
from model import RulE
from utils import load_config, save_config, set_logger, set_seed
from trainer import GroundTrainer, PreTrainer
from rl.state_encoder import StateEncoder
from rl.rule_selector import RuleSelectorAgent
from rl.path_finder import PathFinderAgent
from rl.kg_env import KGReasoningEnv
from rl.reward_calculator import RewardCalculator
from rl.trainer_rl import RulERLTrainer

# torch.cuda.set_device(1)
# python main.py --init ../config/umls_pretrain_config.json

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

    # reinforcement learning parameters
    parser.add_argument('--state_dim', default=128, type=int, help='State encoder output dimension for RL')
    parser.add_argument('--history_dim', default=128, type=int, help='History GRU hidden size for RL state encoder')
    parser.add_argument('--policy_hidden_dim', default=256, type=int, help='Hidden size of RL policy network')
    parser.add_argument('--value_hidden_dim', default=256, type=int, help='Hidden size of RL value network')
    parser.add_argument('--top_k_rules', default=5, type=int, help='Number of rules selected by high-level agent')
    parser.add_argument('--rl_max_steps', default=5, type=int, help='Maximum steps per RL episode')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor for RL')
    parser.add_argument('--epsilon_start', default=0.5, type=float, help='Initial epsilon for rule selector exploration')
    parser.add_argument('--epsilon_end', default=0.05, type=float, help='Final epsilon for rule selector exploration')
    parser.add_argument('--ucb_c', default=1.0, type=float, help='UCB exploration coefficient for rule selector')
    parser.add_argument('--rl_reward_alpha', default=0.1, type=float, help='Reward shaping weight for RL')
    parser.add_argument('--lr_policy', default=0.001, type=float, help='Learning rate for RL policy network')
    parser.add_argument('--lr_value', default=0.001, type=float, help='Learning rate for RL value network')
    parser.add_argument('--lr_selector', default=0.0001, type=float, help='Learning rate for rule selector')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='Gradient clipping threshold for RL components')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of RL training epochs')
    parser.add_argument('--log_interval', default=100, type=int, help='Steps between RL logging updates')
    parser.add_argument('--eval_interval', default=5, type=int, help='Epoch interval for RL validation')
    parser.add_argument('--save_interval', default=10, type=int, help='Epoch interval for saving RL checkpoints')
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

    RulE_model = RulE(graph, args.p_norm, args.mlp_rule_dim, args.gamma_fact, args.gamma_rule, args.hidden_dim, device, args.data_path)
    RulE_model.set_rules(rules)
    RulE_model.rules = rules

    
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

    # 以下 grounding 训练阶段暂时停用
    # ground_trainer = GroundTrainer(
    #     model=RulE_model,
    #     args = args,
    #     train_set=train_set,
    #     valid_set=valid_set,
    #     test_set=test_set,
    #     test_kge_set = test_kge_set,
    #     device=device,
    #     num_worker=args.cpu_num
    # )

    # # valid_mrr = ground_trainer.evaluate('valid', expectation=True)
    # # test_mrr = ground_trainer.evaluate('test', expectation=True)
    #
    # # args.g_batch_size = 32
    #
    # ground_trainer.train(args)

    logging.info('开始冻结预训练模型参数，准备进入RulE-RL阶段')
    RulE_model.eval()
    for param in RulE_model.parameters():
        param.requires_grad = False

    frozen_params = sum(p.numel() for p in RulE_model.parameters())
    logging.info('冻结参数总数: {:,}'.format(frozen_params))

    logging.info('初始化RulE-RL组件')
    entity_dim = RulE_model.entity_embedding.embedding_dim
    rel_dim = RulE_model.relation_embedding.embedding_dim
    rule_dim = RulE_model.rule_emb.embedding_dim
    num_relations = graph.relation_size
    num_rules = len(rules)

    logging.info('实体维度: %d, 关系维度: %d, 规则维度: %d', entity_dim, rel_dim, rule_dim)
    logging.info('规则数量: %d, 关系数量: %d', num_rules, num_relations)

    state_encoder = StateEncoder(
        entity_dim=entity_dim,
        rel_dim=rel_dim,
        rule_dim=rule_dim,
        history_dim=args.history_dim,
        state_dim=args.state_dim
    ).to(device)

    rule_selector = RuleSelectorAgent(
        entity_dim=entity_dim,
        rel_dim=rel_dim,
        rule_dim=rule_dim,
        num_rules=num_rules,
        hidden_dim=args.state_dim,
        ucb_c=args.ucb_c
    ).to(device)

    path_finder = PathFinderAgent(
        state_dim=args.state_dim,
        action_dim=num_relations * 2,
        hidden_dim=args.policy_hidden_dim
    ).to(device)

    reward_calculator = RewardCalculator(
        rule_model=RulE_model,
        alpha=args.rl_reward_alpha
    )

    env = KGReasoningEnv(
        graph=graph,
        rule_model=RulE_model,
        state_encoder=state_encoder,
        reward_calculator=reward_calculator,
        max_steps=args.rl_max_steps
    )

    rl_trainer = RulERLTrainer(
        rule_model=RulE_model,
        rule_selector=rule_selector,
        path_finder=path_finder,
        env=env,
        graph=graph,
        args=args
    )

    train_queries = [tuple(fact) for fact in graph.train_facts]
    valid_queries = [tuple(fact) for fact in graph.valid_facts]
    test_queries = [tuple(fact) for fact in graph.test_facts]

    logging.info('RulE-RL训练开始，总训练查询数: %d', len(train_queries))
    rl_metrics = rl_trainer.train(train_queries, valid_queries, test_queries)

    logging.info('RulE-RL训练完成，测试集指标:')
    logging.info('MRR: %.4f | MR: %.2f | Hits@1: %.4f | Hits@3: %.4f | Hits@10: %.4f',
                 rl_metrics["mrr"], rl_metrics["mr"],
                 rl_metrics["hits@1"], rl_metrics["hits@3"], rl_metrics["hits@10"])
    
    # return test_mrr


if __name__ == '__main__':
    
    main()
