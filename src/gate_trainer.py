import logging
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from gate import CalibrationGate


def _tensor_stats(x):
    if x is None:
        return None
    x = x.detach()
    return {
        "mean": x.mean().item(),
        "std": x.std().item(),
        "min": x.min().item(),
        "max": x.max().item(),
    }


def _sample_negative_tails(positive_mask, num_negatives):
    """
    Sample negatives for each row where positive_mask[row, tail] == True means tail is positive.
    Returns: [batch, num_negatives] LongTensor
    """
    device = positive_mask.device
    batch_size, nentity = positive_mask.size()
    num_negatives = int(num_negatives)
    if num_negatives <= 0:
        return torch.empty(batch_size, 0, dtype=torch.long, device=device)

    negatives = torch.empty(batch_size, num_negatives, dtype=torch.long, device=device)
    filled = torch.zeros(batch_size, dtype=torch.long, device=device)

    while (filled < num_negatives).any():
        remaining = (num_negatives - filled).clamp(min=0)
        max_need = int(remaining.max().item())
        draw = max(4, max_need * 2)

        candidates = torch.randint(0, nentity, (batch_size, draw), device=device)
        is_neg = ~positive_mask.gather(1, candidates)

        for i in range(batch_size):
            need = int(remaining[i].item())
            if need <= 0:
                continue
            valid = candidates[i][is_neg[i]]
            if valid.numel() == 0:
                continue
            take = min(need, int(valid.numel()))
            start = int(filled[i].item())
            negatives[i, start : start + take] = valid[:take]
            filled[i] = start + take

    return negatives


def _grounding_stats_from_logits(grounding_logits, flag_mask, topk=50):
    """
    Compute (margin_ratio, entropy) from grounding logits, masked by flag_mask (True means allowed).
    """
    if flag_mask.dtype != torch.bool:
        flag_mask = flag_mask.bool()

    masked = grounding_logits.masked_fill(~flag_mask, float("-inf"))
    all_false = (~flag_mask).all(dim=1)
    if all_false.any():
        masked[all_false] = grounding_logits[all_false]

    top2 = masked.topk(2, dim=1).values
    top1 = top2[:, 0]
    top2v = top2[:, 1]
    margin = top1 - top2v
    margin_ratio = margin / (top1.abs() + 1e-9)

    entity_size = masked.size(1)
    k = max(2, min(int(topk), int(entity_size)))
    topk_vals = masked.topk(k, dim=1).values
    probs = torch.softmax(topk_vals, dim=1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
    entropy = entropy / max(1e-9, math.log(k))

    return margin_ratio, entropy


class GateTrainer:
    def __init__(
        self,
        model,
        train_set,
        valid_set,
        test_set,
        test_kge_set,
        device,
        save_path,
        num_worker=0,
        alpha_min=0.0,
        alpha_max=3.0,
        use_stats=True,
        mlp_hidden_dim=256,
        dropout=0.1,
        topk=50,
        neg_size=128,
        lr=1e-3,
        weight_decay=0.0,
        epochs=5,
        log_steps=100,
    ):
        self.model = model
        self.device = device
        self.save_path = save_path
        self.num_worker = int(num_worker)

        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.test_kge_set = test_kge_set

        self.topk = int(topk)
        self.neg_size = int(neg_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.log_steps = int(log_steps)

        self.gate = CalibrationGate(
            hidden_dim=self.model.hidden_dim,
            use_stats=use_stats,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        ).to(self.device)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        gate_params = sum(p.numel() for p in self.gate.parameters())
        logging.info(
            "[Gate] initialized: "
            f"use_stats={self.gate.use_stats}, alpha_range=[{self.gate.alpha_min}, {self.gate.alpha_max}], "
            f"neg_size={self.neg_size}, topk={self.topk}, epochs={self.epochs}, lr={self.lr}, "
            f"weight_decay={self.weight_decay}, params={gate_params}"
        )

    def _iter_train_batches(self):
        self.train_set.make_batches()
        loader = DataLoader(self.train_set, 1, num_workers=self.num_worker)
        for batch in loader:
            all_h, all_r, all_t, target, edges_to_remove = batch
            yield (
                all_h.squeeze(0),
                all_r.squeeze(0),
                all_t.squeeze(0),
                target.squeeze(0),
                edges_to_remove.squeeze(0),
            )

    @torch.no_grad()
    def evaluate(self, split, alpha_fallback=3.0):
        if split == "test_kge":
            dataset = self.test_kge_set
        else:
            dataset = getattr(self, f"{split}_set")

        dataloader = DataLoader(dataset, 1, num_workers=self.num_worker)

        self.model.eval()
        self.gate.eval()

        concat_logits = []
        concat_all_t = []
        concat_flag = []
        concat_alpha = []

        for batch in tqdm(dataloader):
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0).to(self.device)
            all_r = all_r.squeeze(0).to(self.device)
            all_t = all_t.squeeze(0).to(self.device)
            flag = flag.squeeze(0).to(self.device)

            grounding_logits, _ = self.model(all_h, all_r, None)
            kge_score = self.model.compute_g_KGE(all_h, all_r)

            h_emb, r_emb = self.model.get_query_embeddings(all_h, all_r)

            if self.gate.use_stats:
                margin_ratio, entropy = _grounding_stats_from_logits(grounding_logits, flag, topk=self.topk)
                alpha_vec = self.gate(h_emb, r_emb, margin_ratio, entropy)
            else:
                alpha_vec = self.gate(h_emb, r_emb)

            if torch.isnan(alpha_vec).any() or torch.isinf(alpha_vec).any():
                alpha_vec = torch.full_like(all_h.float(), float(alpha_fallback))

            logits = grounding_logits + alpha_vec.unsqueeze(1) * kge_score

            concat_logits.append(logits)
            concat_all_t.append(all_t)
            concat_flag.append(flag)
            concat_alpha.append(alpha_vec.detach().cpu())

        concat_logits = torch.cat(concat_logits, dim=0)
        concat_all_t = torch.cat(concat_all_t, dim=0)
        concat_flag = torch.cat(concat_flag, dim=0)
        alpha_all = torch.cat(concat_alpha, dim=0) if len(concat_alpha) > 0 else None

        ranks = []
        for k in range(concat_all_t.size(0)):
            t = concat_all_t[k]
            val = concat_logits[k, t]
            L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
            H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
            ranks.append((L, H))

        hit1 = hit3 = hit10 = mr = mrr = 0.0
        for (L, H) in ranks:
            for rank in range(L, H):
                if rank <= 1:
                    hit1 += 1.0 / (H - L)
                if rank <= 3:
                    hit3 += 1.0 / (H - L)
                if rank <= 10:
                    hit10 += 1.0 / (H - L)
                mr += rank / (H - L)
                mrr += 1.0 / rank / (H - L)

        n = len(ranks)
        hit1 /= n
        hit3 /= n
        hit10 /= n
        mr /= n
        mrr /= n

        logging.info(f"[Gate] {split} Data : {n}")
        logging.info(f"[Gate] {split} Hit1 : {hit1:.6f}")
        logging.info(f"[Gate] {split} Hit3 : {hit3:.6f}")
        logging.info(f"[Gate] {split} Hit10: {hit10:.6f}")
        logging.info(f"[Gate] {split} MR   : {mr:.6f}")
        logging.info(f"[Gate] {split} MRR  : {mrr:.6f}")

        if alpha_all is not None:
            logging.info(
                "[Gate] alpha stats: "
                f"mean={alpha_all.mean().item():.6f}, std={alpha_all.std().item():.6f}, "
                f"min={alpha_all.min().item():.6f}, max={alpha_all.max().item():.6f}"
            )

        return mrr

    def train(self):
        os.makedirs(self.save_path, exist_ok=True)
        best_gate_path = os.path.join(self.save_path, "gate.pt")

        optimizer = torch.optim.Adam(
            self.gate.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        best_valid_mrr = -1.0
        steps = 0

        for epoch in range(1, self.epochs + 1):
            self.gate.train()
            losses = []
            alpha_samples = []
            margin_samples = []
            entropy_samples = []
            max_stat_samples = 4096

            for all_h, all_r, all_t, target, edges_to_remove in self._iter_train_batches():
                all_h = all_h.to(self.device)
                all_r = all_r.to(self.device)
                all_t = all_t.to(self.device)
                target = target.to(self.device)
                edges_to_remove = edges_to_remove.to(self.device)

                positive_mask = target.bool()

                with torch.no_grad():
                    grounding_logits, _ = self.model(all_h, all_r, edges_to_remove)
                    h_emb, r_emb = self.model.get_query_embeddings(all_h, all_r)

                    flag = ~positive_mask
                    if self.gate.use_stats:
                        margin_ratio, entropy = _grounding_stats_from_logits(grounding_logits, flag, topk=self.topk)
                    else:
                        margin_ratio = entropy = None

                neg_tails = _sample_negative_tails(positive_mask, self.neg_size)
                tail_ids = torch.cat([all_t.unsqueeze(1), neg_tails], dim=1)

                with torch.no_grad():
                    kge_scores = self.model.compute_kge_for_tails(all_h, all_r, tail_ids)
                    grounding_scores = grounding_logits.gather(1, tail_ids)

                alpha_vec = self.gate(h_emb, r_emb, margin_ratio, entropy) if self.gate.use_stats else self.gate(h_emb, r_emb)
                final_scores = grounding_scores + alpha_vec.unsqueeze(1) * kge_scores

                if len(alpha_samples) < max_stat_samples:
                    alpha_samples.append(alpha_vec.detach().cpu())
                    if self.gate.use_stats:
                        margin_samples.append(margin_ratio.detach().cpu())
                        entropy_samples.append(entropy.detach().cpu())

                labels = torch.zeros(final_scores.size(0), dtype=torch.long, device=self.device)
                loss = F.cross_entropy(final_scores, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                steps += 1

                if self.log_steps and steps % self.log_steps == 0:
                    msg = f"[Gate] epoch={epoch} step={steps} loss={sum(losses)/len(losses):.6f}"
                    if len(alpha_samples) > 0:
                        alpha_cat = torch.cat(alpha_samples, dim=0)
                        a = _tensor_stats(alpha_cat)
                        msg += (
                            f" alpha(mean={a['mean']:.4f}, std={a['std']:.4f}, "
                            f"min={a['min']:.4f}, max={a['max']:.4f})"
                        )
                        if self.gate.use_stats and len(margin_samples) > 0:
                            m = _tensor_stats(torch.cat(margin_samples, dim=0))
                            e = _tensor_stats(torch.cat(entropy_samples, dim=0))
                            msg += f" margin_mean={m['mean']:.4f} entropy_mean={e['mean']:.4f}"
                    logging.info(msg)
                    losses = []

            if len(alpha_samples) > 0:
                a = _tensor_stats(torch.cat(alpha_samples, dim=0))
                if self.gate.use_stats and len(margin_samples) > 0:
                    m = _tensor_stats(torch.cat(margin_samples, dim=0))
                    e = _tensor_stats(torch.cat(entropy_samples, dim=0))
                    logging.info(
                        "[Gate] epoch=%d train_summary loss=%.6f alpha(mean=%.4f,std=%.4f,min=%.4f,max=%.4f) "
                        "margin_mean=%.4f entropy_mean=%.4f",
                        epoch,
                        (sum(losses) / len(losses)) if len(losses) > 0 else float("nan"),
                        a["mean"],
                        a["std"],
                        a["min"],
                        a["max"],
                        m["mean"],
                        e["mean"],
                    )
                else:
                    logging.info(
                        "[Gate] epoch=%d train_summary loss=%.6f alpha(mean=%.4f,std=%.4f,min=%.4f,max=%.4f)",
                        epoch,
                        (sum(losses) / len(losses)) if len(losses) > 0 else float("nan"),
                        a["mean"],
                        a["std"],
                        a["min"],
                        a["max"],
                    )

            logging.info(f"[Gate] epoch={epoch} finished, evaluating on valid...")
            valid_mrr = self.evaluate("valid")
            if valid_mrr > best_valid_mrr:
                best_valid_mrr = valid_mrr
                torch.save({"gate": self.gate.state_dict()}, best_gate_path)
                logging.info(f"[Gate] saved best gate to {best_gate_path} (valid MRR={best_valid_mrr:.6f})")

        if os.path.exists(best_gate_path):
            state = torch.load(best_gate_path, map_location=self.device)
            self.gate.load_state_dict(state["gate"])

        logging.info("[Gate] final evaluation with best gate")
        best_valid_mrr = self.evaluate("valid")
        best_test_mrr = self.evaluate("test")
        best_test_kge_mrr = self.evaluate("test_kge")

        logging.info('-------------------------')
        logging.info(f'| [Gate] Best Valid MRR: {best_valid_mrr:.6f}')
        logging.info(f'| [Gate] Best Test MRR : {best_test_mrr:.6f}')
        logging.info(f'| [Gate] Test+KGE MRR  : {best_test_kge_mrr:.6f}')
        logging.info('-------------------------')

        return best_valid_mrr, best_test_mrr, best_test_kge_mrr
