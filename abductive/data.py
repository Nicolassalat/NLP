"""Loading, tokenizing and batching the ART (alpha-NLI) dataset."""

import random
import re
from collections import defaultdict
from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"
FIELDS = ("observation_1", "observation_2", "hypothesis_1", "hypothesis_2")

# Lowercased, roughly matching the tokenization GloVe 6B was trained on
# ("don't" -> "do", "n't"; "Chad's" -> "chad", "'s").
_TOKEN_RE = re.compile(r"[a-z]+(?=n't)|n't|'[a-z]+|[a-z0-9]+|[^\sa-z0-9]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def load_splits(val_fraction: float, seed: int) -> dict[str, list[dict]]:
    """Return train/val/test splits.

    The official dev set serves as the test set because the test labels are
    not public. The validation set is carved out of train by story, i.e. by
    the (O1, O2) pair, since every story appears many times in train with
    different hypothesis pairs and a random split would leak.
    """
    raw = load_dataset("allenai/art")

    stories = defaultdict(list)
    for ex in raw["train"]:
        stories[(ex["observation_1"], ex["observation_2"])].append(ex)
    keys = sorted(stories)
    random.Random(seed).shuffle(keys)
    n_val = int(len(keys) * val_fraction)

    return {
        "train": [ex for k in keys[n_val:] for ex in stories[k]],
        "val": [ex for k in keys[:n_val] for ex in stories[k]],
        "test": list(raw["validation"]),
    }


class ARTDataset(Dataset):
    def __init__(self, examples: list[dict], stoi: dict[str, int]):
        unk = stoi[UNK]
        self.items = []
        for ex in examples:
            ids = {f: [stoi.get(t, unk) for t in tokenize(ex[f])] or [unk] for f in FIELDS}
            ids["label"] = ex["label"] - 1  # labels are 1/2 in the raw data
            self.items.append(ids)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _pad(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(s) for s in seqs])
    out = torch.zeros(len(seqs), int(lengths.max()), dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out, lengths


@dataclass
class Collator:
    """Builds batches for either model type.

    "bag": each field is padded separately.
    "rnn": each hypothesis becomes one sequence "O1 <sep> h <sep> O2"; the
    batch holds all h1 sequences followed by all h2 sequences.
    """

    model: str
    sep_id: int
    hypothesis_only: bool = False

    def __call__(self, batch):
        labels = torch.tensor([b["label"] for b in batch])
        if self.model == "bag":
            out = {f: _pad([b[f] for b in batch]) for f in FIELDS}
            out["label"] = labels
            return out

        seqs = []
        for h in ("hypothesis_1", "hypothesis_2"):
            for b in batch:
                if self.hypothesis_only:
                    seqs.append(b[h])
                else:
                    seqs.append(b["observation_1"] + [self.sep_id] + b[h] + [self.sep_id] + b["observation_2"])
        tokens, lengths = _pad(seqs)
        return {"tokens": tokens, "lengths": lengths, "label": labels}
