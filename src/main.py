
import logging, os, datetime
import argparse
import torch
from data import KnowledgeGraph, TrainDataset, ValidDataset, TestDataset, RuleDataset, KGETrainDataset
from model import RulE
from utils import load_config, save_config, set_logger, set_seed
# RulE-SSRL拆分方案: 导入PreTrainer和PolicyTrainer
from trainer import PreTrainer, PolicyTrainer
try:
    from trainer import GroundTrainer
except ImportError:
    GroundTrainer = None

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


    # ========== grounding训练参数（仅原始RulE模式使用）==========
    parser.add_argument('--mlp_rule_dim', default=100, type=int)
    parser.add_argument('--alpha', default=0.5, type=float, help='KGE评分权重（测试融合时使用）')
    parser.add_argument('--beta', default=0.5, type=float, help='策略评分权重（测试融合时使用）')
    parser.add_argument('--smoothing', default=0.5, type=float)
    parser.add_argument('--batch_per_epoch', default=1000000, type=int)
    parser.add_argument('--print_every', default=1000, type=int)
    parser.add_argument('--g_batch_size', default=16, type=int)
    parser.add_argument('--g_lr', default=0.00005, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--num_iters', default=20, type=int)

    # ========== RulE-SSRL拆分方案: 策略网络参数 ==========
    parser.add_argument('--use_policy_network', action='store_true', default=False,
                       help='使用策略网络模式（RulE-SSRL）')
    parser.add_argument('--max_num_actions', default=200, type=int,
                       help='每个实体的最大动作数')
    parser.add_argument('--policy_hidden_dim', default=128, type=int,
                       help='策略网络隐藏层维度')

    # Phase 2: 策略网络训练参数
    parser.add_argument('--policy_num_iters', default=20, type=int,
                       help='策略网络训练轮数')
    parser.add_argument('--policy_batch_size', default=32, type=int,
                       help='策略网络训练批大小')
    parser.add_argument('--policy_lr', default=0.0001, type=float,
                       help='策略网络学习率')
    parser.add_argument('--policy_log_steps', default=100, type=int,
                       help='策略网络训练日志频率')
    parser.add_argument('--policy_num_rollouts', default=1, type=int,
                       help='策略网络训练时每个查询采样的路径数')
    parser.add_argument('--policy_eval_every', default=1, type=int,
                       help='策略网络训练时每多少轮验证一次')

    # Phase 3: 推理参数
    parser.add_argument('--num_policy_samples', default=10, type=int,
                       help='推理时K次采样数量')
    parser.add_argument('--max_path_length', default=3, type=int,
                       help='策略探索的最大路径长度')
    parser.add_argument('--use_kge_fusion', action='store_true', default=True,
                       help='推理时是否融合KGE评分')
    parser.add_argument('--beam_size', default=50, type=int,
                       help='Beam Search beam width')
    parser.add_argument('--dev_batch_size', default=64, type=int,
                       help='Validation/Test batch size')

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



    # 知识图谱和规则集（两种模式都需要）
    # RulE-SSRL: 传入max_num_actions用于预计算动作空间
    graph = KnowledgeGraph(
        args.data_path,
        max_num_actions=args.max_num_actions if hasattr(args, 'max_num_actions') else 200
    )
    ruleset = RuleDataset(graph.relation_size, args.rule_file, args.rule_negative_size)

    # RulE-SSRL: grounding数据集（仅原始RulE模式需要）
    if not (args.use_policy_network if hasattr(args, 'use_policy_network') else False):
        train_set = TrainDataset(graph, args.g_batch_size)
        valid_set = ValidDataset(graph, args.g_batch_size)
        test_set = TestDataset(graph, args.g_batch_size)
        test_kge_set = TestDataset(graph, 16)
    else:
        # RulE-SSRL模式：使用dev_batch_size作为验证/测试batch大小
        dev_batch_size = args.dev_batch_size if hasattr(args, 'dev_batch_size') else 64
        train_set = None
        valid_set = ValidDataset(graph, dev_batch_size)
        test_set = TestDataset(graph, dev_batch_size)
        test_kge_set = None

    rules = [rule[0] for rule in ruleset.rules]
    
    
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # RulE-SSRL: 构建带可选策略网络的模型
    RulE_model = RulE(
        graph,
        args.p_norm,
        args.mlp_rule_dim,
        args.gamma_fact,
        args.gamma_rule,
        args.hidden_dim,
        device,
        args.data_path,
        use_policy_network=args.use_policy_network if hasattr(args, 'use_policy_network') else False,
        policy_hidden_dim=args.policy_hidden_dim if hasattr(args, 'policy_hidden_dim') else 256,
        rule_bonus_coef=args.rule_bonus_coef if hasattr(args, 'rule_bonus_coef') else 0.1,
        rule_bonus_default=args.rule_bonus_default if hasattr(args, 'rule_bonus_default') else 0.05
    )
    RulE_model.set_rules(rules)

    if args.use_policy_network if hasattr(args, 'use_policy_network') else False:
        logging.info('>>>>> RulE-SSRL拆分方案模式')
    else:
        logging.info('>>>>> 原始RulE模式: 使用grounding')


    # ========== Phase 1: 预训练 ==========
    logging.info('='*60)
    logging.info('Phase 1: 预训练 (KGE + Rule)')
    logging.info('='*60)

    pre_trainer = PreTrainer(
        graph=graph,
        model=RulE_model,
        valid_set=valid_set,
        test_set=test_set,
        ruleset=ruleset,
        expectation=True,
        device=device,
        num_worker=args.cpu_num
    )

    pre_trainer.train(args)

    logging.info('Phase 1 预训练完成!')

    # 加载最佳预训练checkpoint
    print("加载预训练最佳checkpoint...")
    checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
    RulE_model.load_state_dict(checkpoint['model'])

    logging.info('预训练结果评估:')
    valid_mrr = pre_trainer.evaluate('valid', expectation=True)
    test_mrr = pre_trainer.evaluate('test', expectation=True)

    # ========== 根据模式选择后续流程 ==========
    if args.use_policy_network if hasattr(args, 'use_policy_network') else False:
        # ========== RulE-SSRL拆分方案 ==========

        # ========== Phase 2: 策略网络训练 ==========
        logging.info('='*60)
        logging.info('Phase 2: 策略网络训练')
        logging.info('='*60)

        policy_trainer = PolicyTrainer(
            model=RulE_model,
            graph=graph,
            valid_set=valid_set,
            test_set=test_set,
            device=device,
            num_worker=args.cpu_num
        )

        policy_trainer.train(args)

        logging.info('Phase 2 策略网络训练完成!')

        # ========== Phase 3: 测试（KGE + 策略网络 融合）==========
        logging.info('='*60)
        logging.info('Phase 3: 测试 (KGE + Policy 融合)')
        logging.info('='*60)

        valid_mrr, test_mrr = policy_trainer.test_with_fusion(args)

        logging.info('='*60)
        logging.info('RulE-SSRL拆分方案 训练完成!')
        logging.info('最终测试MRR: {:.6f}'.format(test_mrr))
        logging.info('='*60)

    else:
        # ========== 原始RulE模式: Grounding阶段 ==========
        logging.info('='*60)
        logging.info('原始RulE: Grounding阶段')
        logging.info('='*60)

        if GroundTrainer is None:
            logging.error('GroundTrainer未能导入，无法进行grounding训练！')
            raise ImportError('GroundTrainer is required for original RulE mode')

        ground_trainer = GroundTrainer(
            model=RulE_model,
            args=args,
            train_set=train_set,
            valid_set=valid_set,
            test_set=test_set,
            test_kge_set=test_kge_set,
            device=device,
            num_worker=args.cpu_num
        )

        ground_trainer.train(args)

        logging.info('Grounding阶段完成!')
    
    # return test_mrr


if __name__ == '__main__':
    
    main()
