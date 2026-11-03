"""Building a frozen GloVe embedding matrix for the dataset vocabulary."""

import zipfile
from pathlib import Path

import torch

from abductive.data import DATASET, FIELDS, PAD, SEP, UNK, tokenize

GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"


def build_embeddings(splits: dict[str, list[dict]], dim: int, data_dir: Path) -> tuple[dict[str, int], torch.Tensor]:
    """Return (stoi, embedding matrix) restricted to words that occur in the data.

    Words without a GloVe vector map to <unk>. <pad> and <unk> are zero vectors,
    <sep> gets a fixed random vector. Everything stays frozen during training.
    The result is cached because scanning the GloVe file takes a while.
    """
    cache = data_dir / f"glove.6B.{dim}d.{DATASET.split('/')[-1]}.vocab.pt"
    if cache.exists():
        saved = torch.load(cache)
        return saved["stoi"], saved["vectors"]

    zip_path = data_dir / "glove.6B.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found, download it from {GLOVE_URL}")

    words = {t for examples in splits.values() for ex in examples for f in FIELDS for t in tokenize(ex[f])}

    found = {}
    with zipfile.ZipFile(zip_path) as zf, zf.open(f"glove.6B.{dim}d.txt") as fh:
        for line in fh:
            word, *values = line.decode("utf-8").rstrip().split(" ")
            if word in words:
                found[word] = torch.tensor([float(v) for v in values])

    itos = [PAD, UNK, SEP] + sorted(found)
    vectors = torch.zeros(len(itos), dim)
    vectors[2] = torch.randn(dim, generator=torch.Generator().manual_seed(0)) * 0.4
    for i, word in enumerate(itos[3:], start=3):
        vectors[i] = found[word]

    stoi = {w: i for i, w in enumerate(itos)}
    print(f"GloVe {dim}d: {len(found)}/{len(words)} words covered")
    torch.save({"stoi": stoi, "vectors": vectors}, cache)
    return stoi, vectors
