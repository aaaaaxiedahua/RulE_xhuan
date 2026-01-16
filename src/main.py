
import logging, os, datetime
import argparse
import torch
from data import KnowledgeGraph, TrainDataset, ValidDataset, TestDataset, RuleDataset, KGETrainDataset
from model import RulE
from utils import load_config, save_config, set_logger, set_seed
from trainer import GroundTrainer, PreTrainer

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

    # CaRulE switches (optional)
    parser.add_argument('--use_adapter', action='store_true', default=False)
    parser.add_argument('--use_logic_attention', action='store_true', default=False)
    parser.add_argument('--attn_dim', default=64, type=int)
    parser.add_argument('--adapter_hidden_dim', default=1024, type=int)
    parser.add_argument('--adapter_dropout', default=0.0, type=float)
    parser.add_argument('--attn_dropout', default=0.0, type=float)

    # TAPC-RulE switches (optional)
    parser.add_argument('--use_tapc', action='store_true', default=False, help='use TAPC-RulE')
    parser.add_argument('--num_clusters', default=50, type=int, help='number of type clusters')
    parser.add_argument('--lambda_weight', default=0.3, type=float, help='type fusion weight')
    parser.add_argument('--critic_hidden_dim', default=256, type=int, help='critic hidden dimension')
    parser.add_argument('--critic_num_layers', default=1, type=int, help='critic GRU layers')
    parser.add_argument('--critic_dropout', default=0.1, type=float, help='critic dropout')
    parser.add_argument('--critic_lr', default=0.001, type=float, help='critic learning rate')
    parser.add_argument('--critic_epochs', default=10, type=int, help='critic training epochs')
    parser.add_argument('--critic_batch_size', default=32, type=int, help='critic batch size')
    parser.add_argument('--num_path_samples', default=100000, type=int, help='number of path samples')
    parser.add_argument('--neg_ratio', default=1.0, type=float, help='negative sample ratio')

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
    RulE_model.configure_context_modules(
        use_adapter=getattr(args, 'use_adapter', False),
        use_logic_attention=getattr(args, 'use_logic_attention', False),
        attn_dim=getattr(args, 'attn_dim', 64),
        adapter_hidden_dim=getattr(args, 'adapter_hidden_dim', 1024),
        adapter_dropout=getattr(args, 'adapter_dropout', 0.0),
        attn_dropout=getattr(args, 'attn_dropout', 0.0),
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
    RulE_model.load_state_dict(checkpoint['model'], strict=False)
    
    
    logging.info('Test the results of pre-training')
    
    valid_mrr = pre_trainer.evaluate('valid', expectation=True)
    test_mrr = pre_trainer.evaluate('test', expectation=True)

    # TAPC-RulE训练流程（在pre-training之后，ground-training之前）
    if getattr(args, 'use_tapc', False):
        logging.info("=" * 50)
        logging.info("开始TAPC-RulE训练流程（基于训练好的embedding）")
        logging.info("=" * 50)

        from tapc_type_discovery import TypeDiscovery, extract_entity_embeddings
        from tapc_dataset import PathCriticDataset
        from tapc_critic import PathCritic, TypeAwareEmbedding
        from tapc_trainer import CriticTrainer

        # 步骤1: 类型发现（使用训练好的embedding）
        logging.info("=" * 50)
        logging.info("步骤1: 执行类型发现（K-Means聚类）")
        logging.info("=" * 50)
        num_clusters = getattr(args, 'num_clusters', 50)
        logging.info(f"聚类参数: num_clusters={num_clusters}, random_state={args.seed}")
        type_discovery = TypeDiscovery(
            num_clusters=num_clusters,
            random_state=args.seed
        )

        entity_embeddings = extract_entity_embeddings(RulE_model)
        entity_to_type = type_discovery.fit(entity_embeddings)

        type_save_path = os.path.join(args.save_path, 'type_discovery')
        type_discovery.save(type_save_path)
        logging.info(f"类型发现完成，共{type_discovery.num_clusters}个类型")
        logging.info("=" * 50)

        # 步骤2: 构建Critic训练数据集（基于规则grounding）
        logging.info("=" * 50)
        logging.info("步骤2: 构建PathCritic训练数据集（基于规则grounding）")
        logging.info("=" * 50)
        num_path_samples = getattr(args, 'num_path_samples', 100000)
        neg_ratio = getattr(args, 'neg_ratio', 1.0)
        logging.info(f"数据集参数:")
        logging.info(f"  - 目标正样本数: {num_path_samples}")
        logging.info(f"  - 负样本比例: {neg_ratio}")
        logging.info(f"  - 类型感知采样: True")
        logging.info(f"  - 规则数量: {len(rules)}")

        critic_dataset = PathCriticDataset(
            graph=graph,
            entity_to_type=entity_to_type,
            rules=rules,  # 传入规则（line 143已定义）
            num_samples=num_path_samples,
            neg_ratio=neg_ratio,
            type_aware_sampling=True,
            max_path_length=3
        )
        logging.info(f"数据集构建完成，共{len(critic_dataset)}个样本")
        logging.info("=" * 50)

        # 步骤3: 训练PathCritic
        logging.info("=" * 50)
        logging.info("步骤3: 训练PathCritic模型")
        logging.info("=" * 50)
        critic_trainer = CriticTrainer(
            graph=graph,
            entity_to_type=entity_to_type,
            entity_emb_layer=RulE_model.entity_embedding,
            relation_emb_layer=RulE_model.relation_embedding,
            hidden_dim=getattr(args, 'critic_hidden_dim', 256),
            num_layers=getattr(args, 'critic_num_layers', 1),
            dropout=getattr(args, 'critic_dropout', 0.1),
            lambda_weight=getattr(args, 'lambda_weight', 0.3),
            proj_dim=getattr(args, 'critic_proj_dim', 512),
            lr=getattr(args, 'critic_lr', 0.001),
            device=device
        )

        critic_save_dir = os.path.join(args.save_path, 'critic_checkpoints')
        best_acc = critic_trainer.train(
            train_dataset=critic_dataset,
            val_dataset=None,
            num_epochs=getattr(args, 'critic_epochs', 10),
            batch_size=getattr(args, 'critic_batch_size', 32),
            save_dir=critic_save_dir
        )
        logging.info(f"Critic训练完成，最佳准确率: {best_acc:.4f}")
        logging.info("=" * 50)

        # 步骤4: 配置RulE模型使用TAPC-RulE
        logging.info("=" * 50)
        logging.info("步骤4: 配置RulE模型使用TAPC-RulE")
        logging.info("=" * 50)

        # 冻结Critic参数（Stage 3要求：Critic参数不参与后续训练）
        for param in critic_trainer.critic.parameters():
            param.requires_grad = False
        for param in critic_trainer.type_aware_emb.parameters():
            param.requires_grad = False

        # 统计冻结的参数数量
        critic_params = sum(p.numel() for p in critic_trainer.critic.parameters())
        type_emb_params = sum(p.numel() for p in critic_trainer.type_aware_emb.parameters())
        logging.info(f"Critic参数已冻结:")
        logging.info(f"  - PathCritic参数量: {critic_params:,}")
        logging.info(f"  - TypeAwareEmbedding参数量: {type_emb_params:,}")
        logging.info(f"  - 总冻结参数量: {critic_params + type_emb_params:,}")

        RulE_model.configure_tapc(
            use_tapc=True,
            critic=critic_trainer.critic,
            type_aware_emb=critic_trainer.type_aware_emb,
            entity_to_type=entity_to_type
        )
        logging.info("TAPC-RulE配置完成")
        logging.info("=" * 50)
        logging.info("TAPC-RulE训练流程全部完成！")
        logging.info("=" * 50)

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
