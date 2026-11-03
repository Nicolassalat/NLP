# Abductive Commonsense Reasoning on ART

Given two observations O1 and O2, pick the more plausible of two hypotheses
h1 and h2 that explains what happened in between
([ART / alpha-NLI](https://huggingface.co/datasets/allenai/art)), using the course version of the dataset,
[`amarfurt/art-nlp-hs26`](https://huggingface.co/datasets/amarfurt/art-nlp-hs26).

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

The three splits come from the course dataset `amarfurt/art-nlp-hs26` (duplicate observations
removed, splits made by the course) and are used as they are:

| split | size |
|-------|------|
| train | 16,801 |
| val   | 1,000 |
| test  | 1,532 (equal to the official ART dev set, checked example by example) |

Hyperparameters and models are selected on val only; test is evaluated once per run at the end.

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

Hyperparameter sweeps (manual):

```bash
uv run --env-file .env wandb sweep sweeps/bag.yaml
uv run --env-file .env wandb agent <sweep-id>
```

## Reproducing all results

`scripts/run_all.py` runs every experiment from scratch into one W&B project:
the six main runs (bag; RNN with LSTM/max, GRU/max, LSTM/last; hypothesis-only
bag and RNN), both Bayesian sweeps (20 bag runs, 12 RNN runs, selected on
`val/acc`), and 3 seeds of each best sweep config. It is resumable: a finished
step leaves `logs/<project>/<step>.done` and is skipped on the next call.

```bash
uv run python scripts/run_all.py --project art-abductive-v3-course-dataset
```

Then build the results file:

```bash
uv run --env-file .env python scripts/collect_results.py art-abductive-v3-course-dataset   # W&B -> logs/<project>/results.json
uv run python -m scripts.analyze art-abductive-v3-course-dataset                           # val/test gap checks, heuristics, leakage, error analysis
uv run python scripts/make_results.py art-abductive-v3-course-dataset                      # -> RESULTS_art-abductive-v3-course-dataset.md
```

| Script | Purpose |
|---|---|
| `scripts/run_all.py` | main runs, sweeps and seed runs in one W&B project |
| `scripts/collect_results.py` | download all finished runs of a project to `results.json` |
| `scripts/analyze.py` | val/test gap analysis: word-overlap and length heuristics, story/hypothesis leakage, model agreement, error examples |
| `scripts/make_results.py` | render `RESULTS_<project>.md` from the two JSON files |

## Code layout

| File | Purpose |
|---|---|
| `abductive/data.py` | tokenization, loading the course splits, dataset and collator (bag: per-field padding; RNN: `O1 <sep> h <sep> O2`) |
| `abductive/embeddings.py` | builds and caches the frozen GloVe matrix (`<pad>`/`<unk>` zero, `<sep>` fixed random) |
| `abductive/models.py` | `BagOfEmbeddings`, `RNNScorer`, `build_model` |
| `abductive/train.py` | training and evaluation loop, W&B logging, early stopping, best-val checkpointing |
| `sweeps/bag.yaml`, `sweeps/rnn.yaml` | W&B sweep configurations |

## Results and experiment tracking

- Results: [`RESULTS_art-abductive-v3-course-dataset.md`](RESULTS_art-abductive-v3-course-dataset.md)
- W&B project: [`art-abductive-v3-course-dataset`](https://forge.coreweave.com/wandb/nicolassalat-hochschule-luzern/art-abductive-v3-course-dataset)
- Saved workspace view: [`final-results-v3`](https://forge.coreweave.com/wandb/nicolassalat-hochschule-luzern/art-abductive-v3-course-dataset/workspace?nw=an9vd3jm46f)
- Summary (best sweep config, mean ± std over 3 seeds; chance is 50%):

  | Model | Val acc (%) | Test acc (%) |
  |---|---|---|
  | Bag of embeddings | 72.0 ± 0.2 | 52.9 ± 0.4 |
  | Bi-GRU, max pooling | 74.3 ± 1.0 | 53.3 ± 1.2 |
  | Bag, hypothesis-only (1 seed) | 66.6 | 50.8 |
  | Bi-LSTM, hypothesis-only (1 seed) | 70.5 | 53.5 |

- Expectations written down for each model: [`EXPECTATIONS.md`](EXPECTATIONS.md)

### History

- `art-abductive-v1-discarded` (`RESULTS_art-abductive-v1.md`): first run, on the original `allenai/art` data. It used a fused
  two-layer LSTM that crashed the process at exit on Windows and killed the sweep agent. Superseded by v2.
- `art-abductive-v2-full-dataset` (`RESULTS_art-abductive-v2.md`): complete run on the original `allenai/art` data (169k training rows,
  5% story-level validation split, test = official dev set). Superseded by v3, which uses the course splits.
  [W&B project](https://forge.coreweave.com/wandb/nicolassalat-hochschule-luzern/art-abductive-v2-full-dataset),
  [saved view](https://forge.coreweave.com/wandb/nicolassalat-hochschule-luzern/art-abductive-v2-full-dataset/workspace?nw=khl7opu5teb).

## References

- Bhagavatula et al. (2020), *Abductive Commonsense Reasoning* (ART), [arXiv:1908.05739](https://arxiv.org/abs/1908.05739)
- Pennington et al. (2014), GloVe 6B embeddings, <https://nlp.stanford.edu/projects/glove/>
