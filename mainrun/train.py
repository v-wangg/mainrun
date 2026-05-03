import utils
import argparse
import math, random, time, inspect
from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm
import structlog
import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

@dataclass
class Hyperparameters:
    block_size: int = 128
    batch_size: int = 64
    vocab_size: int = 16_000
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 512
    dropout: float = 0.1
    lr: float = 6e-4
    adam_betas: tuple = (0.9, 0.95)
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    warmup_steps: int = 50
    min_lr_frac: float = 0.1
    grad_clip: float = 1.0
    evals_per_epoch: int = 3

    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"

def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = open(log_file, 'w')
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    class DualLogger:
        # structlog file is canonical; wandb metrics are derived from emit() kwargs.
        # Numeric scalars on events that carry `step` are auto-mirrored to run.log
        # under "{event}/{field}". One-time events (no step) stay structlog-only.
        _WANDB_SKIP = {"step", "max_steps", "prnt"}

        def __init__(self, file_handler):
            self.file_handler = file_handler
            self.logger = structlog.get_logger()
            self.run = None

        def set_run(self, run):
            self.run = run

        def emit(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()

            if self.run is not None and "step" in kwargs:
                metrics = {
                    f"{event}/{k}": v
                    for k, v in kwargs.items()
                    if k not in self._WANDB_SKIP and isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                if metrics:
                    self.run.log(metrics, step=kwargs["step"])

            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s")
                else:
                    parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)

    return DualLogger(file_handler)

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

def get_titles(num_titles: int, seed: int, val_frac: float) -> str:
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data").shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]

def get_batch(train_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device, generator: torch.Generator):
    max_start = len(train_ids) - block_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator).tolist()
    x = torch.stack([train_ids[s : s + block_size] for s in starts]).to(device)
    y = torch.stack([train_ids[s + 1 : s + block_size + 1] for s in starts]).to(device)
    return x, y

def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y

def train_tokenizer(titles: list[str], vocab_size: int, unk_token: str = "<unk>", pad_token: str = "<pad>", eos_token: str = "<eos>") -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[pad_token, eos_token, unk_token]
    )
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer

class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer
        self.stoi = {tok: i for tok, i in tokenizer.get_vocab().items()}
        self.itos = {i: tok for tok, i in tokenizer.get_vocab().items()}

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self): return self.tk.get_vocab_size()

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head   = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop= nn.Dropout(cfg.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :] # (B, n_head, T, head_dim) each
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, n_head, T, T)
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, n_head, head_dim) -> (B, T, n_head * head_dim = d_model)
        return self.resid_drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp  = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f      = nn.LayerNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        tok = self.token_emb(idx) # (B, T, d_model)
        pos = self.pos_emb[:, :T, :] # (B, T, d_model)
        x = self.drop(tok + pos)
        for block in self.blocks: x = block(x) # (B, T, d_model)
        x = self.ln_f(x)
        logits = self.head(x) # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        return logits, loss

def configure_optimizer(model: nn.Module, args: "Hyperparameters", device: str):
    # Any param with dim() >= 2 gets weight decay (Linear weights,
    # token_emb.weight, pos_emb), everything else (biases, LayerNorm params) does not.
    # head.weight is tied to token_emb.weight; named_parameters() deduplicates.
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    optim_groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    n_decay_params = sum(p.numel() for p in decay)
    n_no_decay_params = sum(p.numel() for p in no_decay)
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device == "cuda"
    opt = torch.optim.AdamW(
        optim_groups,
        lr=args.lr,
        betas=args.adam_betas,
        eps=args.adam_eps,
        fused=use_fused,
    )
    return opt, n_decay_params, n_no_decay_params, len(decay), len(no_decay), use_fused

def get_lr(step: int, peak_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)

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
    tok = BPETokenizer(train_tokenizer(train_titles+val_titles, args.vocab_size, eos_token=eos_token))
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
    logger.emit("model_info", parameters_count=model_params)

    # Activation RMS taps. Three forward hooks read the residual stream amplitude at
    # entry (post-embed/dropout), middle of the stack, and exit (post-final-LN).
    # Hooks fire every step (compute is trivial); values are sampled into health_step
    # events every 50 steps.
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
        with torch.no_grad():
            for xb, yb in iter_full_split(train_eval_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        return losses / len(train_eval_text)

    def evaluate_position_loss():
        # Per-token (not per-char) val CE averaged over all val batches, broken out by
        # sequence position. One-time diagnostic; healthy autoregressive models show
        # a monotone-decreasing curve (later positions get more context → lower loss).
        # Uses iter_full_split (frozen) but does not mutate evaluate()'s behavior.
        model.eval()
        pos_losses = torch.zeros(args.block_size, device=device)
        n_batches = 0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                per_token = F.cross_entropy(
                    logits.view(-1, V), yb.view(-1), reduction='none'
                ).view(B, T)
                pos_losses += per_token.mean(dim=0)
                n_batches += 1
        model.train()
        return (pos_losses / max(n_batches, 1)).cpu().tolist()

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

            if step == 1 or step % 50 == 0:
                logger.emit("health_step",
                            step=step,
                            max_steps=max_steps,
                            act_rms_post_embed=taps.values["post_embed"],
                            act_rms_mid_stack=taps.values["mid_stack"],
                            act_rms_pre_head=taps.values["pre_head"],
                            prnt=False)

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate()
                train_eval_loss = evaluate_train()
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
                            val_perplexity_per_token=math.exp(val_loss * chars_per_token_val),
                            train_eval_loss=train_eval_loss,
                            train_eval_perplexity_per_token=math.exp(train_eval_loss * (len(train_eval_text) / len(train_eval_ids))),
                            generalization_gap=val_loss - train_eval_loss,
                            weight_norm_total=weight_norm_total,
                            val_loss_best_so_far=best_val_loss,
                            epoch=epoch,
                            elapsed_time=elapsed)
                if step == 1:
                    pos_losses = evaluate_position_loss()
                    # structlog: full per-position curve as a list. The DualLogger's
                    # wandb auto-mirror skips non-scalar values, so this stays
                    # structlog-only — keeps the JSON record complete without
                    # producing 128 single-point line charts in wandb.
                    # NB: don't add scalar context kwargs here — they would auto-mirror
                    # as one-point time-series. block_size is recoverable as len(losses).
                    logger.emit("position_loss",
                                step=step,
                                losses=pos_losses)
                    # wandb: render the curve as a single custom chart with
                    # sequence position on the x-axis. Uses run.log directly
                    # because emit() only handles scalar time-series.
                    if run is not None:
                        pos_table = wandb.Table(
                            data=[[i, l] for i, l in enumerate(pos_losses)],
                            columns=["position", "loss"],
                        )
                        run.log({"position_loss/curve_at_step_1": wandb.plot.line(
                            pos_table, "position", "loss",
                            title="Per-token val loss vs sequence position (step 1)",
                        )}, step=step)

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
        if logger and hasattr(logger, 'file_handler'):
            logger.file_handler.close()
        if WANDB_AVAILABLE:
            wandb.finish()
