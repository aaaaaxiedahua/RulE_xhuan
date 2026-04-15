import argparse
import copy
import json
import logging
import os
import shutil

import torch
from torch.utils.data import DataLoader

from data import KnowledgeGraph, TrainDataset, ValidDataset, TestDataset, RuleDataset
from main import parse_args as main_parse_args
from model import RulE
from trainer import GroundTrainer
from utils import load_config, save_config, set_logger, set_seed

try:
    import optuna
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError("optuna is required to run optuna_dual_pathway.py") from exc


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna search for stage-2 single-channel rule-conditioned GNN")
    parser.add_argument("--stage1_dir", required=True, type=str,
                        help="Directory containing the fixed stage-1 checkpoint and config.json")
    parser.add_argument("--study_name", default="dual_pathway_search", type=str)
    parser.add_argument("--save_root", default="../outputs/optuna", type=str,
                        help="Root directory for Optuna trial outputs")
    parser.add_argument("--storage", default=None, type=str,
                        help="Optuna storage URL. If omitted, a dataset-specific sqlite DB is created automatically")
    parser.add_argument("--n_trials", default=20, type=int)
    parser.add_argument("--timeout", default=None, type=int)
    parser.add_argument("--sampler_seed", default=800, type=int)
    parser.add_argument("--startup_trials", default=3, type=int,
                        help="Number of random startup trials before TPE begins")
    parser.add_argument("--early_stop_patience", default=3, type=int,
                        help="Stop one trial if valid MRR does not improve for this many evaluations")
    parser.add_argument("--shared_kge_cache_dir", default=None, type=str,
                        help="Optional shared KGE cache directory reused across trials")
    parser.add_argument("--override_config", default=None, type=str,
                        help="Optional config path to override stage1_dir/config.json")
    return parser.parse_args()


def resolve_src_relative(path_value):
    if path_value is None or path_value == "":
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(SRC_DIR, path_value))


def reset_logging():
    root = logging.getLogger("")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def load_stage1_args(cli_args):
    args = main_parse_args([])
    config_path = cli_args.override_config or os.path.join(cli_args.stage1_dir, "config.json")
    loaded_args = load_config(config_path)[0]
    for key, value in loaded_args.items():
        setattr(args, key, value)

    args.data_path = resolve_src_relative(args.data_path)
    args.rule_file = resolve_src_relative(args.rule_file)
    args.init_checkpoint_config = ""
    args.reasoner_type = "single_pathway"
    args.save_path = cli_args.stage1_dir
    return args


def infer_dataset_name(data_path):
    return os.path.basename(os.path.normpath(data_path))


def build_default_storage_url(save_root, dataset_name, study_name):
    dataset_dir = resolve_src_relative(os.path.join(save_root, dataset_name))
    os.makedirs(dataset_dir, exist_ok=True)
    db_path = os.path.abspath(os.path.join(dataset_dir, f"{study_name}.db")).replace(os.sep, "/")
    return "sqlite:///" + db_path


def sample_stage2_params(trial, base_args):
    trial_args = copy.deepcopy(base_args)
    trial_args.reasoner_type = "single_pathway"
    trial_args.g_num_layers = trial.suggest_int("g_num_layers", 2, 8)
    trial_args.g_hidden_dim = trial.suggest_categorical("g_hidden_dim", [64, 128, 256, 512])
    trial_args.g_message_hidden_dim = trial.suggest_categorical("g_message_hidden_dim", [64, 128, 256, 512])
    trial_args.g_attn_dim = trial.suggest_categorical("g_attn_dim", [32, 64, 128, 256])
    trial_args.g_dropout = trial.suggest_float("g_dropout", 0.0, 0.5, step=0.05)
    trial_args.g_activation = trial.suggest_categorical("g_activation", ["relu", "gelu", "tanh"])
    trial_args.g_layer_norm = trial.suggest_categorical("g_layer_norm", [False, True])
    trial_args.g_readout = trial.suggest_categorical("g_readout", ["multiply", "linear"])
    trial_args.rule_tf_layers = trial.suggest_categorical("rule_tf_layers", [1, 2, 3])
    trial_args.rule_num_heads = trial.suggest_categorical("rule_num_heads", [2, 4, 8])
    trial_args.rule_dropout = trial.suggest_float("rule_dropout", 0.0, 0.2, step=0.05)
    trial_args.rule_ffn_dim = trial.suggest_categorical("rule_ffn_dim", [128, 256, 512, 1024])
    trial_args.g_lr = trial.suggest_categorical("g_lr", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
    trial_args.weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
    trial_args.smoothing = trial.suggest_float("smoothing", 0.0, 0.5, step=0.05)
    return trial_args


def build_stage2_components(args, device):
    graph = KnowledgeGraph(args.data_path)
    train_set = TrainDataset(graph, args.g_batch_size)
    valid_set = ValidDataset(graph, args.g_batch_size)
    test_set = TestDataset(graph, args.g_batch_size)
    test_kge_set = TestDataset(graph, 16)
    ruleset = RuleDataset(graph.relation_size, args.rule_file, args.rule_negative_size)
    rules = [rule[0] for rule in ruleset.rules]

    model = RulE(
        graph,
        args.p_norm,
        args.mlp_rule_dim,
        args.gamma_fact,
        args.gamma_rule,
        args.hidden_dim,
        device,
        args.data_path,
        reasoner_type=args.reasoner_type,
        g_num_layers=args.g_num_layers,
        g_hidden_dim=args.g_hidden_dim,
        g_message_hidden_dim=args.g_message_hidden_dim,
        g_attn_dim=args.g_attn_dim,
        g_dropout=args.g_dropout,
        g_activation=args.g_activation,
        g_layer_norm=args.g_layer_norm,
        g_readout=args.g_readout,
        rule_tf_layers=args.rule_tf_layers,
        rule_num_heads=args.rule_num_heads,
        rule_dropout=args.rule_dropout,
        rule_ffn_dim=args.rule_ffn_dim,
    )
    model.set_rules(rules)

    trainer = GroundTrainer(
        model=model,
        args=args,
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        test_kge_set=test_kge_set,
        device=device,
        num_worker=args.cpu_num,
    )
    return model, trainer


def load_stage1_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing:
        logging.info("Stage-1 checkpoint missing keys: %s", missing)
    if unexpected:
        logging.info("Stage-1 checkpoint unexpected keys: %s", unexpected)


def ensure_shared_cache_link(trial_dir, shared_cache_dir):
    if shared_cache_dir is None:
        return

    os.makedirs(shared_cache_dir, exist_ok=True)
    target_cache_dir = os.path.join(trial_dir, "kge_cache")
    if os.path.lexists(target_cache_dir):
        if os.path.islink(target_cache_dir) or os.path.isfile(target_cache_dir):
            os.unlink(target_cache_dir)
        else:
            shutil.rmtree(target_cache_dir)

    try:
        os.symlink(shared_cache_dir, target_cache_dir)
    except OSError:
        logging.info("Symlink for shared KGE cache is unavailable on this platform; trial will build its own cache.")


def save_trial_summary(trial_args, trial_save_path, best_valid_mrr, best_iter, stopped_iter, early_stopped):
    payload = {
        "best_valid_mrr": best_valid_mrr,
        "best_iteration": best_iter,
        "stopped_iteration": stopped_iter,
        "early_stopped": early_stopped,
        "stage2_params": {
            "g_num_layers": trial_args.g_num_layers,
            "g_hidden_dim": trial_args.g_hidden_dim,
            "g_message_hidden_dim": trial_args.g_message_hidden_dim,
            "g_attn_dim": trial_args.g_attn_dim,
            "g_dropout": trial_args.g_dropout,
            "g_activation": trial_args.g_activation,
            "g_layer_norm": trial_args.g_layer_norm,
            "g_readout": trial_args.g_readout,
            "rule_tf_layers": trial_args.rule_tf_layers,
            "rule_num_heads": trial_args.rule_num_heads,
            "rule_dropout": trial_args.rule_dropout,
            "rule_ffn_dim": trial_args.rule_ffn_dim,
            "g_lr": trial_args.g_lr,
            "weight_decay": trial_args.weight_decay,
            "smoothing": trial_args.smoothing,
        },
    }
    with open(os.path.join(trial_save_path, "trial_result.json"), "w") as fo:
        json.dump(payload, fo, indent=2)


def build_objective(cli_args, base_args):
    stage1_checkpoint = os.path.join(cli_args.stage1_dir, "checkpoint")
    shared_cache_dir = resolve_src_relative(cli_args.shared_kge_cache_dir) if cli_args.shared_kge_cache_dir else None
    study_root = resolve_src_relative(os.path.join(cli_args.save_root, cli_args.study_name))
    os.makedirs(study_root, exist_ok=True)

    def objective(trial):
        trial_args = sample_stage2_params(trial, base_args)
        trial_args.seed = int(base_args.seed) + trial.number
        trial_args.save_path = os.path.join(study_root, f"trial_{trial.number:04d}")
        os.makedirs(trial_args.save_path, exist_ok=True)
        ensure_shared_cache_link(trial_args.save_path, shared_cache_dir)

        reset_logging()
        set_logger(trial_args.save_path)
        save_config(trial_args)
        set_seed(trial_args.seed)

        device = torch.device("cuda" if trial_args.cuda and torch.cuda.is_available() else "cpu")
        logging.info("Starting trial %d with params: %s", trial.number, trial.params)

        model, ground_trainer = build_stage2_components(trial_args, device)
        load_stage1_checkpoint(model, stage1_checkpoint, device)

        ground_trainer.model.entity_embedding.weight.requires_grad = False
        ground_trainer.model.relation_embedding.weight.requires_grad = False
        ground_trainer.model.rule_emb.weight.requires_grad = False

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, ground_trainer.model.parameters()),
            lr=float(trial_args.g_lr),
            weight_decay=float(trial_args.weight_decay),
        )

        scheduler = None
        if getattr(trial_args, "scheduler", "none") == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=float(trial_args.scheduler_factor),
                patience=int(trial_args.scheduler_patience),
            )

        ground_trainer.train_set.make_batches()
        train_dataloader = DataLoader(ground_trainer.train_set, 1, num_workers=ground_trainer.num_worker)

        ground_trainer.model.prepare_reasoner(
            device,
            cache_dir=os.path.join(trial_args.save_path, "kge_cache"),
            checkpoint_path=stage1_checkpoint,
            kge_batch_size=trial_args.g_batch_size,
        )

        logging.info(">>>>> RulE: %s-Training (Optuna)", ground_trainer.model.reasoner_type)

        best_valid_mrr = float("-inf")
        best_iter = -1
        stopped_iter = -1
        stale_rounds = 0
        checkpoint_path = os.path.join(trial_args.save_path, "grounding.pt")

        for iteration in range(trial_args.num_iters):
            logging.info("-------------------------")
            logging.info("| Trial: %d | Iteration: %d/%d", trial.number, iteration + 1, trial_args.num_iters)
            logging.info("-------------------------")

            ground_trainer.train_step(
                optimizer,
                train_dataloader,
                trial_args.batch_per_epoch,
                trial_args.smoothing,
                trial_args.print_every,
                trial_args,
            )
            valid_mrr_iter = ground_trainer.evaluate("valid", expectation=True)
            trial.report(valid_mrr_iter, step=iteration)

            if scheduler is not None:
                scheduler.step(valid_mrr_iter)

            if valid_mrr_iter > best_valid_mrr:
                best_valid_mrr = valid_mrr_iter
                best_iter = iteration + 1
                stale_rounds = 0
                ground_trainer.save(trial_args, checkpoint_path)
            else:
                stale_rounds += 1

            if stale_rounds >= cli_args.early_stop_patience:
                stopped_iter = iteration + 1
                logging.info(
                    "Early stop trial %d at iteration %d: valid MRR did not improve for %d consecutive evaluations.",
                    trial.number,
                    stopped_iter,
                    cli_args.early_stop_patience,
                )
                break

        if stopped_iter < 0:
            stopped_iter = trial_args.num_iters

        trial.set_user_attr("best_iteration", best_iter)
        trial.set_user_attr("stopped_iteration", stopped_iter)
        trial.set_user_attr("early_stopped", stopped_iter < trial_args.num_iters)

        save_trial_summary(
            trial_args,
            trial_args.save_path,
            best_valid_mrr,
            best_iter,
            stopped_iter,
            stopped_iter < trial_args.num_iters,
        )
        return best_valid_mrr

    return objective


def save_study_summary(study, study_root):
    summary = {
        "study_name": study.study_name,
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "best_params": study.best_trial.params,
        "num_trials": len(study.trials),
    }
    with open(os.path.join(study_root, "best_params.json"), "w") as fo:
        json.dump(summary, fo, indent=2)


def main():
    cli_args = parse_args()
    base_args = load_stage1_args(cli_args)
    study_root = resolve_src_relative(os.path.join(cli_args.save_root, cli_args.study_name))
    os.makedirs(study_root, exist_ok=True)
    dataset_name = infer_dataset_name(base_args.data_path)
    storage = cli_args.storage or build_default_storage_url(cli_args.save_root, dataset_name, cli_args.study_name)

    sampler = optuna.samplers.TPESampler(
        seed=cli_args.sampler_seed,
        n_startup_trials=cli_args.startup_trials,
    )
    study = optuna.create_study(
        study_name=cli_args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        storage=storage,
        load_if_exists=True,
    )
    objective = build_objective(cli_args, base_args)
    study.optimize(objective, n_trials=cli_args.n_trials, timeout=cli_args.timeout)
    save_study_summary(study, study_root)

    print("Best valid MRR:", study.best_value)
    print("Best params:", study.best_trial.params)


if __name__ == "__main__":
    main()
