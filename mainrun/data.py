import torch
from datasets import load_dataset
from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers, decoders

# GPT-4 cl100k_base pretokenization regex (from tiktoken's published source).
# Preprocessing config — no pre-trained weights involved.
CL100K_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

def get_titles(num_titles: int, seed: int, val_frac: float) -> tuple[list[str], list[str]]:
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
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(CL100K_PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
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
        self.stoi = dict(tokenizer.get_vocab())
        self.itos = {i: tok for tok, i in self.stoi.items()}

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self): return self.tk.get_vocab_size()
