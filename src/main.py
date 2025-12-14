
import logging, os, datetime
import argparse
import torch
from data import KnowledgeGraph, TrainDataset, ValidDataset, TestDataset, RuleDataset, KGETrainDataset
from model import RulE
from utils import load_config, save_config, set_logger, set_seed
# RulE-SSRL: GroundTrainer仅在原始RulE模式中使用
from trainer import PreTrainer
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


    # ========== grounding训练参数（仅原始RulE模式使用，RulE-SSRL不需要）==========
    # 注：当use_policy_network=True时，以下参数不会被使用
    parser.add_argument('--mlp_rule_dim', default=100, type=int)
    parser.add_argument('--alpha', default=5.0, type=int, help='weight the KGE score')
    parser.add_argument('--smoothing', default=0.5, type=float)
    parser.add_argument('--batch_per_epoch', default=1000000, type=int)
    parser.add_argument('--print_every', default=1000, type=int)
    parser.add_argument('--g_batch_size', default=16, type=int)
    parser.add_argument('--g_lr', default=0.00005, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--num_iters', default=20, type=int)

    # ========== RulE-SSRL: 策略网络参数 ==========
    parser.add_argument('--use_policy_network', action='store_true', default=False,
                       help='使用策略网络进行RulE-SSRL模式')
    parser.add_argument('--max_num_actions', default=200, type=int,
                       help='每个实体的最大动作数（策略网络，参考SSRL设计）')
    parser.add_argument('--policy_hidden_dim', default=256, type=int,
                       help='策略网络隐藏层维度')
    parser.add_argument('--policy_batch_size', default=32, type=int,
                       help='策略训练的批大小')
    parser.add_argument('--weight_policy', default=0.5, type=float,
                       help='策略损失的权重')
    parser.add_argument('--num_policy_samples', default=10, type=int,
                       help='推理时采样的路径数量')
    parser.add_argument('--max_path_length', default=3, type=int,
                       help='策略rollout的最大路径长度')

    # ========== RulE-SSRL: 可选RL微调阶段参数 ==========
    parser.add_argument('--enable_rl_finetuning', action='store_true', default=False,
                       help='在预训练后启用RL微调阶段（冻结嵌入，只训练策略网络）')
    parser.add_argument('--rl_finetuning_steps', default=10000, type=int,
                       help='RL微调的训练步数')
    parser.add_argument('--rl_finetuning_lr', default=0.0001, type=float,
                       help='RL微调的学习率')
    parser.add_argument('--rl_log_steps', default=100, type=int,
                       help='RL微调的日志输出频率')

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
        # RulE-SSRL模式：不需要grounding数据集，使用KGE验证集
        train_set = None
        valid_set = ValidDataset(graph, args.g_batch_size)
        test_set = TestDataset(graph, args.g_batch_size)
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
        policy_hidden_dim=args.policy_hidden_dim if hasattr(args, 'policy_hidden_dim') else 256
    )
    RulE_model.set_rules(rules)

    if args.use_policy_network if hasattr(args, 'use_policy_network') else False:
        logging.info('>>>>> RulE-SSRL模式: 使用策略网络')
    else:
        logging.info('>>>>> 原始RulE模式: 使用grounding')


    # For pre-training

    # 创建PreTrainer
    # 注：策略网络训练是模型的核心组成部分，由model.use_policy_network控制
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
    
    # checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
    # RulE_model.load_state_dict(checkpoint['model'])


    # valid_mrr = pre_trainer.evaluate('valid', expectation=True)
    # test_mrr = pre_trainer.evaluate('test', expectation=True)
    
    pre_trainer.train(args)


    logging.info('Finishing pre-training!')

    print("Loading best checkpoint from pre-training...")

    # Load rule embedding and KGE embedding
    checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
    RulE_model.load_state_dict(checkpoint['model'])


    logging.info('Testing results of pre-training')

    valid_mrr = pre_trainer.evaluate('valid', expectation=True)
    test_mrr = pre_trainer.evaluate('test', expectation=True)

    # RulE-SSRL: 条件训练流程
    if args.use_policy_network if hasattr(args, 'use_policy_network') else False:
        # RulE-SSRL模式: 跳过grounding阶段，使用策略网络推理
        logging.info('>>>>> RulE-SSRL: 跳过grounding阶段（使用策略网络）')

        # 可选的RL微调阶段
        if args.enable_rl_finetuning if hasattr(args, 'enable_rl_finetuning') else False:
            logging.info('>>>>> RulE-SSRL: 开始可选RL微调阶段')
            logging.info('>>>>> 冻结KGE和Rule嵌入，只训练策略网络')

            # 冻结实体、关系、规则嵌入
            RulE_model.entity_embedding.weight.requires_grad = False
            RulE_model.relation_embedding.weight.requires_grad = False
            RulE_model.rule_emb.weight.requires_grad = False

            # 创建只优化策略网络的优化器
            rl_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, RulE_model.policy_network.parameters()),
                lr=args.rl_finetuning_lr if hasattr(args, 'rl_finetuning_lr') else 0.0001
            )

            # RL微调训练循环
            from data import QueryDataset, Iterator
            from torch.utils.data import DataLoader
            query_dataloader = DataLoader(
                QueryDataset(graph.train_facts),
                batch_size=args.policy_batch_size if hasattr(args, 'policy_batch_size') else 32,
                shuffle=True,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=QueryDataset.collate_fn
            )
            rl_query_iterator = Iterator(query_dataloader)

            RulE_model.train()
            rl_steps = args.rl_finetuning_steps if hasattr(args, 'rl_finetuning_steps') else 10000
            rl_log_steps = args.rl_log_steps if hasattr(args, 'rl_log_steps') else 100

            logging.info('开始RL微调，共{}步'.format(rl_steps))

            for step in range(1, rl_steps + 1):
                rl_optimizer.zero_grad()

                query_batch = next(rl_query_iterator)
                if device.type == "cuda":
                    query_batch = query_batch.cuda(device)

                # 只计算策略损失
                loss_policy = RulE_model.compute_policy_loss(query_batch)
                loss_policy.backward()
                rl_optimizer.step()

                if step % rl_log_steps == 0:
                    logging.info('RL微调步骤 {}/{}: policy_loss = {:.6f}'.format(
                        step, rl_steps, loss_policy.item()))

            logging.info('>>>>> RL微调阶段完成')

            # 保存RL微调后的模型
            checkpoint = {
                'model': RulE_model.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'checkpoint_rl_finetuned'))
            logging.info('RL微调模型已保存到: {}'.format(
                os.path.join(args.save_path, 'checkpoint_rl_finetuned')))

        # RulE-SSRL: 策略网络推理测试
        logging.info('>>>>> RulE-SSRL: 开始策略网络推理测试')

        if GroundTrainer is not None:
            # 创建GroundTrainer用于评估（它的evaluate会调用model.forward -> forward_policy）
            ground_trainer = GroundTrainer(
                model=RulE_model,
                args=args,
                train_set=None,  # RulE-SSRL不需要train_set
                valid_set=valid_set,
                test_set=test_set,
                test_kge_set=None,
                device=device,
                num_worker=args.cpu_num
            )

            logging.info('>>>>> 策略网络推理 - 验证集')
            ground_trainer.evaluate('valid', alpha=args.alpha, expectation=True)
            logging.info('>>>>> 策略网络推理 - 测试集')
            ground_trainer.evaluate('test', alpha=args.alpha, expectation=True)
        else:
            logging.warning('GroundTrainer未导入，无法进行策略网络推理测试')

        logging.info('>>>>> 训练完成！')
    else:
        # 原始RulE模式: 继续grounding阶段
        logging.info('>>>>> 原始RulE: 开始grounding阶段')

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

        logging.info('>>>>> Grounding阶段完成！')
    
    # return test_mrr


if __name__ == '__main__':
    
    main()
