from dataclasses import dataclass

@dataclass
class Hyperparameters:
    block_size: int = 128
    batch_size: int = 256
    vocab_size: int = 8000
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 512
    dropout: float = 0.0
    lr: float = 2e-3
    adam_betas: tuple = (0.9, 0.95)
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    warmup_steps: int = 50
    min_lr_frac: float = 0.1
    grad_clip: float = 1.0
    evals_per_epoch: int = 3
    health_log_interval: int = 50
    use_doc_mask: bool = True

    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"
