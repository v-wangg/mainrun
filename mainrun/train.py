import utils
import argparse
import math, random, time
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from config import Hyperparameters
from telemetry import wandb, WANDB_AVAILABLE, configure_logging
from model import GPTConfig, GPT
from data import get_titles, get_batch, iter_full_split, train_tokenizer, BPETokenizer
from optim import configure_optimizer, get_lr

logger = None

BEST_CKPT_PATH = Path("checkpoints/best.pt")

def save_checkpoint_atomic(model, hyperparams, step, val_loss, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model_state": model.state_dict(),
        "config": vars(hyperparams),
        "step": step,
        "val_loss": val_loss,
    }, tmp)
    tmp.replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None,
                        help="descriptive label for this run (sets wandb run name)")
    cli_args = parser.parse_args()

    args = Hyperparameters()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    global logger
    logger = configure_logging(args.log_file)

    logger.emit("run_name", name=cli_args.name)

    hyperparams_dict = vars(args)
    logger.emit("hyperparameters_configured", **hyperparams_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.emit("device_info", device=device)

    if WANDB_AVAILABLE:
        wandb_mode = "online" if os.environ.get("WANDB_API_KEY") else "disabled"
        run = wandb.init(
            project="mainrun-sandbox",
            name=cli_args.name,
            config=hyperparams_dict,
            mode=wandb_mode,
        )
        if wandb_mode == "online":
            wandb.save(args.log_file, base_path=".", policy="live")
            logger.emit("wandb_init", mode="online", project="mainrun-sandbox", run_id=run.id, run_url=run.url)
            logger.set_run(run)
        else:
            logger.emit("wandb_disabled", reason="WANDB_API_KEY not set", impact="no run telemetry — Claude cannot see this run")
    else:
        run = None
        logger.emit("wandb_disabled", reason="wandb package not importable", impact="no run telemetry — Claude cannot see this run")

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)

    eos_token = "<eos>"
    tok = BPETokenizer(train_tokenizer(train_titles, args.vocab_size, eos_token=eos_token))
    val_text = eos_token.join(val_titles) + eos_token
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)

    # Initial train_text/train_ids build for tokenizer_info + dataset_info + max_steps.
    # This is overwritten per-epoch in the training loop after a deterministic shuffle.
    train_text = eos_token.join(train_titles) + eos_token
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)

    # train_eval slice: seeded random K headlines, ~equal char count to val_text.
    # Built once before training; pre-loop and stable across epochs so it remains a
    # cross-eval reference. Uses an independent RNG stream from the per-epoch shuffle.
    slice_rng = random.Random(args.seed)
    shuffled_for_slice = list(train_titles)
    slice_rng.shuffle(shuffled_for_slice)
    slice_titles, total = [], 0
    for t in shuffled_for_slice:
        slice_titles.append(t)
        total += len(t) + len(eos_token)
        if total >= len(val_text):
            break
    train_eval_text = eos_token.join(slice_titles) + eos_token
    train_eval_ids = torch.tensor(tok.encode(train_eval_text), dtype=torch.long)

    # Persistent torch.Generator for the data loader's uniform window sampling.
    # Seeded from args.seed so sampling sequence is reproducible across runs.
    data_gen = torch.Generator(device='cpu')
    data_gen.manual_seed(args.seed)

    chars_per_token_val = len(val_text) / len(val_ids)
    chars_per_token_train = len(train_text) / len(train_ids)
    logger.emit("tokenizer_info",
                vocab_size=tok.vocab_size,
                train_text_chars=len(train_text),
                val_text_chars=len(val_text),
                train_tokens=len(train_ids),
                val_tokens=len(val_ids),
                chars_per_token_val=chars_per_token_val,
                chars_per_token_train=chars_per_token_train,
                train_eval_text_chars=len(train_eval_text),
                train_eval_tokens=len(train_eval_ids))
    if run is not None:
        run.summary["chars_per_token_val"] = chars_per_token_val

    batches = len(train_ids) // (args.block_size * args.batch_size)
    max_steps = args.epochs * batches
    eval_interval = batches // args.evals_per_epoch
    logger.emit("dataset_info",
                titles_count=len(train_titles),
                epochs=args.epochs,
                batches_per_epoch=batches,
                tokens_per_epoch=len(train_ids),
                vocab_size=tok.vocab_size)
    logger.emit("loader_info",
                sampling="uniform_independent",
                per_epoch_shuffle=True,
                slice_construction="seeded_random_headlines",
                slice_seed=args.seed,
                slice_chars=len(train_eval_text),
                slice_tokens=len(train_eval_ids))

    cfg = GPTConfig(
        vocab_size = tok.vocab_size,
        block_size = args.block_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        d_model    = args.d_model,
        dropout    = args.dropout,
    )
    model = GPT(cfg).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.emit("model_info",
                parameters_count=model_params,
                init_scheme="gpt2_scaled_residual",
                base_std=0.02,
                residual_std=0.02 * (2 * args.n_layer) ** -0.5,
                n_residual_proj=model.n_residual_proj)

    # Activation RMS taps. Three forward hooks read the residual stream amplitude at
    # entry (post-embed/dropout), middle of the stack, and exit (post-final-LN).
    # Hooks fire every step (compute is trivial); values are sampled into health_step
    # events every args.health_log_interval steps.
    class RMSTaps:
        def __init__(self):
            self.values = {}
        def hook(self, name):
            def fn(module, inputs, output):
                x = output[0] if isinstance(output, tuple) else output
                self.values[name] = x.detach().pow(2).mean().sqrt().item()
            return fn

    taps = RMSTaps()
    mid_idx = max(0, args.n_layer // 2 - 1)
    model.drop.register_forward_hook(taps.hook("post_embed"))
    model.blocks[mid_idx].register_forward_hook(taps.hook("mid_stack"))
    model.ln_f.register_forward_hook(taps.hook("pre_head"))

    opt, n_decay_params, n_no_decay_params, n_decay_tensors, n_no_decay_tensors, use_fused = configure_optimizer(model, args, device)
    logger.emit("optimizer_info",
                optimizer="AdamW",
                fused=use_fused,
                scheduler="warmup_then_cosine (manual get_lr)",
                peak_lr=args.lr,
                min_lr=args.lr * args.min_lr_frac,
                warmup_steps=args.warmup_steps,
                betas=list(args.adam_betas),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
                n_decay_tensors=n_decay_tensors,
                n_no_decay_tensors=n_no_decay_tensors,
                n_decay_params=n_decay_params,
                n_no_decay_params=n_no_decay_params)

    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        return losses / len(val_text)

    def evaluate_train():
        model.eval()
        losses = 0.0
        n_tokens = 0
        with torch.no_grad():
            for xb, yb in iter_full_split(train_eval_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
                n_tokens += B * T
        model.train()
        return losses / len(train_eval_text), losses / n_tokens

    def evaluate_val_per_token():
        # evaluate() is frozen and returns per-character; this parallel pass returns
        # per-token directly using the count of tokens that actually contributed to
        # the loss (iter_full_split drops the tail). Avoids back-deriving via the
        # chars_per_token ratio, which silently divides by len(val_ids) instead.
        model.eval()
        losses = 0.0
        n_tokens = 0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
                n_tokens += B * T
        model.train()
        return losses / n_tokens

    step = 0
    best_val_loss = float("inf")
    best_step = 0
    last_train_loss = float("nan")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # Per-epoch shuffle: deterministic per (seed, epoch). Re-orders headlines
        # before re-tokenization so cross-headline boundaries differ each epoch
        # and the model has no fixed sweep order to memorize.
        epoch_rng = random.Random(args.seed + epoch)
        shuffled_titles = list(train_titles)
        epoch_rng.shuffle(shuffled_titles)
        train_text = eos_token.join(shuffled_titles) + eos_token
        train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)

        for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
            step += 1
            xb, yb = get_batch(train_ids, args.block_size, args.batch_size, device, data_gen)
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
            lr = get_lr(step, args.lr, args.lr * args.min_lr_frac, args.warmup_steps, max_steps)
            for pg in opt.param_groups:
                pg["lr"] = lr

            # Per-group |Δp|/|p| via snapshot diff. Captures the empirical update
            # including the AdamW weight-decay shrink. Memory: ~1× params transient,
            # freed end-of-step.
            pre_groups = [[p.detach().clone() for p in g["params"]] for g in opt.param_groups]
            opt.step()
            group_ratios = []
            for pre_list, g in zip(pre_groups, opt.param_groups):
                upd_sq = sum((p.detach() - pre).pow(2).sum().item() for pre, p in zip(pre_list, g["params"]))
                param_sq = sum(pre.pow(2).sum().item() for pre in pre_list)
                group_ratios.append(math.sqrt(upd_sq / max(param_sq, 1e-12)))
            upd_decay, upd_no_decay = group_ratios  # configure_optimizer order: [decay, no_decay]
            upd_total = math.sqrt(sum(r * r for r in group_ratios) / len(group_ratios))

            last_train_loss = loss.item()
            elapsed = time.time() - t0
            logger.emit("training_step",
                        step=step,
                        max_steps=max_steps,
                        loss=last_train_loss,
                        lr=lr,
                        grad_norm=grad_norm,
                        upd_to_param_decay=upd_decay,
                        upd_to_param_no_decay=upd_no_decay,
                        upd_to_param_total=upd_total,
                        elapsed_time=elapsed,
                        prnt=False)

            if step == 1 or step % args.health_log_interval == 0:
                logger.emit("health_step",
                            step=step,
                            max_steps=max_steps,
                            act_rms_post_embed=taps.values["post_embed"],
                            act_rms_mid_stack=taps.values["mid_stack"],
                            act_rms_pre_head=taps.values["pre_head"],
                            prnt=False)

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate()
                val_loss_per_token = evaluate_val_per_token()
                train_eval_loss, train_eval_loss_per_token = evaluate_train()
                with torch.no_grad():
                    weight_norm_total = math.sqrt(sum(p.detach().pow(2).sum().item() for p in model.parameters() if p.requires_grad))
                # Update best-so-far before the emit so val_loss_best_so_far is current.
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    save_checkpoint_atomic(model, args, step, val_loss, BEST_CKPT_PATH)
                logger.emit("validation_step",
                            step=step,
                            max_steps=max_steps,
                            loss=val_loss,
                            val_loss_per_token=val_loss_per_token,
                            val_perplexity_per_token=math.exp(val_loss_per_token),
                            train_eval_loss=train_eval_loss,
                            train_eval_loss_per_token=train_eval_loss_per_token,
                            train_eval_perplexity_per_token=math.exp(train_eval_loss_per_token),
                            generalization_gap=val_loss - train_eval_loss,
                            generalization_gap_per_token=val_loss_per_token - train_eval_loss_per_token,
                            weight_norm_total=weight_norm_total,
                            val_loss_best_so_far=best_val_loss,
                            epoch=epoch,
                            elapsed_time=elapsed)
    logger.emit("run_summary",
                best_val_loss=best_val_loss,
                best_step=best_step,
                final_train_loss=last_train_loss,
                total_runtime=time.time() - t0,
                wandb_run_url=run.url if run is not None else None)

    if run is not None and BEST_CKPT_PATH.exists():
        artifact = wandb.Artifact(
            name=f"model-{run.id}",
            type="model",
            metadata={"val_loss": best_val_loss, "step": best_step, **hyperparams_dict},
        )
        artifact.add_file(str(BEST_CKPT_PATH))
        run.log_artifact(artifact, aliases=["latest", "best"])

if __name__ == "__main__":
    try:
        main()
    finally:
        if logger is not None:
            logger.close()
        if WANDB_AVAILABLE:
            wandb.finish()
