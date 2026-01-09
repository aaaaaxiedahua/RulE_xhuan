import argparse
import datetime
import logging
import os
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import KnowledgeGraph, RuleDataset, TrainDataset, ValidDataset, TestDataset
from projerule import ProjeRulE
from utils import save_config, set_logger, set_seed


def parse_args(args=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--init", default=None, type=str, help="optional json config to load defaults from")
    pre_args, _ = pre_parser.parse_known_args(args)

    parser = argparse.ArgumentParser(description="ProjeRulE runner")
    parser.add_argument("--init", default=None, type=str, help="optional json config to load defaults from")

    parser.add_argument("--data_path", default="../data/umls", type=str)
    parser.add_argument("--rule_file", default="../data/umls/mined_rules.txt", type=str)
    parser.add_argument("--save_path", default="projerule_umls", type=str)

    parser.add_argument("--cuda", action="store_true", default=False, help="use GPU")
    parser.add_argument("--cpu_num", default=10, type=int)
    parser.add_argument("--seed", default=800, type=int)

    parser.add_argument("--g_batch_size", default=16, type=int)
    parser.add_argument("--g_lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--num_iters", default=20, type=int)
    parser.add_argument("--batch_per_epoch", default=1000000, type=int)
    parser.add_argument("--smoothing", default=0.2, type=float)
    parser.add_argument("--print_every", default=1000, type=int)

    parser.add_argument("--projerule_dim", default=200, type=int)
    parser.add_argument("--projerule_lambda_rule", default=1.0, type=float)
    parser.add_argument("--projerule_tau_rule", default=1.0, type=float)
    parser.add_argument("--projerule_tau_edge", default=1.0, type=float)
    parser.add_argument("--projerule_beta1", default=10.0, type=float)
    parser.add_argument("--projerule_beta2", default=-5.0, type=float)
    parser.add_argument("--projerule_init_phase_scale", default=3.141592653589793, type=float)
    parser.add_argument("--chunk_size", default=4096, type=int)

    # Load defaults from config (json) if provided.
    init_path = pre_args.init
    if init_path:
        with open(init_path, "r") as f:
            cfg = json.load(f)
        parser.set_defaults(**cfg)

    return parser.parse_args(args)


def _save_checkpoint(model, optimizer, save_path: str, name: str = "checkpoint_projerule.pt"):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, os.path.join(save_path, name))
    np.save(os.path.join(save_path, "entity_embedding"), model.entity_embedding.weight.detach().cpu().numpy())
    np.save(os.path.join(save_path, "relation_phase"), model.relation_phase.weight.detach().cpu().numpy())


@torch.no_grad()
def evaluate(model: ProjeRulE, dataset, device: torch.device, expectation: bool = True, chunk_size: int = 4096):
    dataloader = DataLoader(dataset, 1, num_workers=0)
    model.eval()

    concat_logits = []
    concat_all_h = []
    concat_all_r = []
    concat_all_t = []
    concat_flag = []

    for batch in tqdm(dataloader, desc="eval", leave=False):
        all_h, all_r, all_t, flag = batch
        all_h = all_h.squeeze(0).to(device)
        all_r = all_r.squeeze(0).to(device)
        all_t = all_t.squeeze(0).to(device)
        flag = flag.squeeze(0).to(device)

        logits, _aux = model.score_all_tails(all_h, all_r, chunk_size=chunk_size)
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
        t = concat_all_t[k]
        val = concat_logits[k, t]
        allowed = concat_flag[k]

        L = (concat_logits[k][allowed] > val).sum().item() + 1
        H = (concat_logits[k][allowed] >= val).sum().item() + 2
        ranks.append((L, H))

    hit1 = hit3 = hit10 = mr = mrr = 0.0
    for (L, H) in ranks:
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

    n = max(len(ranks), 1)
    hit1 /= n
    hit3 /= n
    hit10 /= n
    mr /= n
    mrr /= n

    logging.info("Eval count: %d", n)
    logging.info("Hit@1 : %.6f", hit1)
    logging.info("Hit@3 : %.6f", hit3)
    logging.info("Hit@10: %.6f", hit10)
    logging.info("MR    : %.6f", mr)
    logging.info("MRR   : %.6f", mrr)
    return mrr


def main():
    args = parse_args()

    args.save_path = os.path.join("../outputs", str(args.save_path))

    os.makedirs(args.save_path, exist_ok=True)
    save_config(args)
    set_logger(args.save_path)
    set_seed(int(args.seed))

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    logging.info("device=%s", device)

    graph = KnowledgeGraph(args.data_path)
    train_set = TrainDataset(graph, int(args.g_batch_size))
    valid_set = ValidDataset(graph, int(args.g_batch_size))
    test_set = TestDataset(graph, int(args.g_batch_size))

    ruleset = RuleDataset(graph.relation_size, args.rule_file, negative_sample_size=128)
    rules = [rule[0] for rule in ruleset.rules]

    model = ProjeRulE(
        graph=graph,
        dim=int(args.projerule_dim),
        lambda_rule=float(args.projerule_lambda_rule),
        tau_rule=float(args.projerule_tau_rule),
        tau_edge=float(args.projerule_tau_edge),
        beta1=float(args.projerule_beta1),
        beta2=float(args.projerule_beta2),
        init_phase_scale=float(args.projerule_init_phase_scale),
    ).to(device)
    model.set_rules(rules)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.g_lr), weight_decay=float(args.weight_decay))
    bce = torch.nn.BCEWithLogitsLoss(reduction="sum")

    train_set.make_batches()
    train_loader = DataLoader(train_set, 1, num_workers=0)

    num_iters = int(args.num_iters)
    batch_per_epoch = int(args.batch_per_epoch)
    smoothing = float(args.smoothing)
    best_valid_mrr = 0.0

    for it in range(num_iters):
        logging.info("Iteration %d/%d", it + 1, num_iters)
        model.train()
        total_loss = 0.0
        total_count = 0
        alpha_sum = 0.0
        alpha_count = 0

        for step, batch in enumerate(train_loader):
            if step >= batch_per_epoch:
                break

            all_h, all_r, _all_t, target, _edges_to_remove = batch
            all_h = all_h.squeeze(0).to(device)
            all_r = all_r.squeeze(0).to(device)
            target = target.squeeze(0).to(device)

            if smoothing > 0:
                target = target * (1.0 - smoothing) + (smoothing / float(graph.entity_size))

            logits, _aux = model.score_all_tails(all_h, all_r, chunk_size=int(args.chunk_size))
            loss = bce(logits, target) / max(int(all_h.numel()), 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_count += 1
            if "alpha" in _aux:
                alpha_sum += float(_aux["alpha"].detach().mean().item())
                alpha_count += 1

            if args.print_every and (step + 1) % int(args.print_every) == 0:
                alpha_mean = alpha_sum / max(alpha_count, 1)
                logging.info(
                    "step=%d loss=%.6f alpha_mean=%.4f",
                    step + 1,
                    total_loss / max(total_count, 1),
                    alpha_mean,
                )

        alpha_mean = alpha_sum / max(alpha_count, 1)
        logging.info("Train loss: %.6f", total_loss / max(total_count, 1))
        logging.info("Train alpha_mean: %.4f", alpha_mean)

        logging.info("Valid...")
        valid_mrr = evaluate(model, valid_set, device=device, expectation=True, chunk_size=int(args.chunk_size))
        if valid_mrr > best_valid_mrr:
            best_valid_mrr = valid_mrr
            _save_checkpoint(model, optimizer, args.save_path)
        logging.info("Valid MRR: %.6f | Best: %.6f", valid_mrr, best_valid_mrr)

    logging.info("Load best checkpoint and evaluate test...")
    ckpt_path = os.path.join(args.save_path, "checkpoint_projerule.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])

    logging.info("Test...")
    evaluate(model, test_set, device=device, expectation=True, chunk_size=int(args.chunk_size))


if __name__ == "__main__":
    main()
