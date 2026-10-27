# Abductive Commonsense Reasoning on ART

Given two observations O1 and O2, pick the more plausible of two hypotheses
h1 and h2 that explains what happened in between
([ART / alpha-NLI](https://huggingface.co/datasets/allenai/art)).

## Models

Both models score each hypothesis separately and treat the two scores as the
logits of a 2-way classification (cross-entropy loss). The GloVe 6B embeddings
are frozen in both.

1. **Bag of embeddings** (`--model bag`): each of O1, O2 and h is averaged into
   one vector. The features `[o1, o2, h, o1*h, o2*h]` go into a 2-layer MLP with ReLU.
2. **RNN** (`--model rnn`): the sequence `O1 <sep> h <sep> O2` goes through a
   2-layer LSTM or GRU (optionally bidirectional). The last hidden state or a
   max pool over time goes into a 2-layer MLP with ReLU.

`--hypothesis-only` drops the observations as a baseline that measures how much
can be solved from annotation artifacts in the hypotheses alone.

## Data splits

| split | source | size |
|-------|--------|------|
| train | ART train minus validation stories | ~161k |
| val   | 5% of the train stories, split by (O1, O2) | ~8k |
| test  | official ART dev set (test labels are not public) | 1,532 |

The validation split is by story because each story appears many times in
train with different hypothesis pairs.

## Setup

```bash
uv sync
mkdir data
curl -L -o data/glove.6B.zip https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip
cp .env.example .env   # then put your W&B API key in .env
```

On Windows, `uv sync` installs the CUDA 13.0 build of PyTorch (needs an NVIDIA
driver >= 580); on Linux the default PyPI build already includes CUDA. Training
uses the GPU automatically when one is available (override with `--device cpu`).

`.env` is gitignored. The training script loads it automatically; for the
`wandb` CLI use `uv run --env-file .env wandb ...`.

## Running

```bash
uv run python -m abductive.train --model bag
uv run python -m abductive.train --model rnn --cell lstm --pooling max
uv run python -m abductive.train --model bag --hypothesis-only
uv run python -m abductive.train --help
```

Add `--no-wandb` to run without logging, and `--max-train 5000` for a quick check.

Hyperparameter sweeps:

```bash
uv run --env-file .env wandb sweep sweeps/bag.yaml
uv run --env-file .env wandb agent <sweep-id>
```
