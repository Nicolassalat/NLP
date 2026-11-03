# Results

Task: ART (alpha-NLI), pick the more plausible hypothesis. Chance is 50%.
All runs are logged in W&B project `art-abductive-v2-full-dataset` (originally named `art-abductive-v2`). Single runs use seed 42, batch size 256 unless noted.
Val = 5% of the train stories (held out by (O1, O2)); test = official ART dev set (1,532 examples).
Val/test accuracy is taken at the epoch with the best val accuracy. Epoch time is the mean wall-clock time per epoch (training + validation) on an RTX 3060.

## Main experiments (full data)

| Configuration | Val acc (%) | Test acc (%) | Trainable params | Epoch time (s) | Epochs |
|---|---|---|---|---|---|
| Bag of embeddings | 71.8 | 55.0 | 384,513 | 7 | 9 |
| RNN: LSTM, max pooling | 76.6 | 56.1 | 901,633 | 24 | 9 |
| RNN: GRU, max pooling | 76.8 | 56.9 | 692,737 | 27 | 10 |
| RNN: LSTM, last-state pooling | 75.2 | 54.6 | 901,633 | 15 | 8 |
| Bag, hypothesis-only | 67.9 | 52.3 | 77,313 | 6 | 10 |
| RNN (LSTM, max), hypothesis-only | 73.1 | 56.2 | 901,633 | 12 | 10 |

## Best configurations over 3 seeds (1, 2, 3)

Mean ± sample standard deviation (n = 3). Each seed also draws a different val split (the seed is passed to `load_splits`), so the val spread includes split variance; the test set is fixed.

| Model | Val acc (%) | Test acc (%) | Trainable params | Epoch time (s) | Per seed (val / test) |
|---|---|---|---|---|---|
| Bag, best sweep config (lr 0.00028, hidden 256, dropout 0.2, batch 64, emb 300) | 73.4 ± 0.5 | 53.9 ± 0.6 | 384,513 | 11 | seed 1: 73.7 / 54.6; seed 2: 73.7 / 53.7; seed 3: 72.9 / 53.4 |
| RNN, best sweep config (LSTM, max pool, rnn_hidden 128, hidden 256, lr 0.0016, dropout 0.1, batch 128, 4 epochs) | 77.9 ± 0.5 | 56.9 ± 0.9 | 901,633 | 31 | seed 1: 78.2 / 55.9; seed 2: 77.3 / 57.4; seed 3: 78.2 / 57.6 |

## Sweep: bag of embeddings (15 runs, sorted by val acc)

| Run | lr | hidden | dropout | batch | emb | Val | Test | Params |
|---|---|---|---|---|---|---|---|---|
| valiant-sweep-15 | 0.00028 | 256 | 0.2 | 64 | 300 | 73.4 | 53.6 | 384,513 |
| ancient-sweep-11 | 0.00011 | 256 | 0 | 64 | 300 | 73.3 | 53.3 | 384,513 |
| spring-sweep-4 | 0.00079 | 512 | 0.2 | 64 | 300 | 73.2 | 54.9 | 769,025 |
| comic-sweep-12 | 0.0002 | 256 | 0 | 64 | 300 | 73.1 | 53.7 | 384,513 |
| bright-sweep-13 | 0.00037 | 256 | 0.2 | 64 | 300 | 73.0 | 54.6 | 384,513 |
| clean-sweep-2 | 0.00077 | 256 | 0.2 | 64 | 300 | 73.0 | 53.1 | 384,513 |
| crisp-sweep-3 | 0.00031 | 512 | 0 | 64 | 300 | 73.0 | 53.7 | 769,025 |
| quiet-sweep-10 | 0.00029 | 512 | 0.2 | 64 | 300 | 73.0 | 54.4 | 769,025 |
| rare-sweep-14 | 0.00011 | 128 | 0 | 64 | 300 | 72.7 | 53.3 | 192,257 |
| grateful-sweep-7 | 0.00018 | 512 | 0.2 | 64 | 300 | 72.7 | 53.3 | 769,025 |
| wobbly-sweep-9 | 0.00012 | 512 | 0.2 | 64 | 300 | 72.3 | 52.7 | 769,025 |
| bumbling-sweep-6 | 0.00078 | 512 | 0 | 64 | 300 | 72.2 | 53.2 | 769,025 |
| avid-sweep-5 | 0.002 | 256 | 0.2 | 64 | 300 | 72.2 | 53.4 | 384,513 |
| silver-sweep-8 | 0.00047 | 512 | 0.4 | 64 | 300 | 71.8 | 53.2 | 769,025 |
| resilient-sweep-1 | 0.00061 | 128 | 0.4 | 128 | 100 | 66.0 | 52.6 | 64,257 |

## Sweep: RNN (8 runs, 4 epochs each, sorted by val acc)

| Run | cell | pooling | rnn_hidden | lr | dropout | Val | Test | Params | Epoch time (s) |
|---|---|---|---|---|---|---|---|---|---|
| whole-sweep-3 | lstm | max | 128 | 0.0016 | 0.1 | 77.6 | 56.6 | 901,633 | 31 |
| faithful-sweep-6 | gru | max | 128 | 0.001 | 0.1 | 77.1 | 56.4 | 692,737 | 33 |
| neat-sweep-8 | gru | max | 256 | 0.002 | 0.1 | 76.9 | 56.5 | 2,171,393 | 44 |
| celestial-sweep-2 | lstm | max | 128 | 0.0017 | 0.3 | 76.1 | 56.3 | 901,633 | 31 |
| sweepy-sweep-1 | lstm | last | 256 | 0.00041 | 0.1 | 76.0 | 54.4 | 2,851,329 | 28 |
| legendary-sweep-5 | lstm | max | 256 | 0.00069 | 0.3 | 75.5 | 54.2 | 2,851,329 | 44 |
| earthy-sweep-4 | lstm | last | 128 | 0.0013 | 0.1 | 75.4 | 57.0 | 901,633 | 23 |
| hearty-sweep-7 | lstm | max | 256 | 0.00043 | 0.3 | 74.8 | 54.5 | 2,851,329 | 44 |

## Val/test gap

Validation accuracy is 67.9-76.8% across the single runs, but test accuracy is 52.3-56.9%.

Checks (seed-42 val split, official dev as test):

| Check | Val | Test |
|---|---|---|
| RNN (LSTM, max), full input | 76.6 | 56.1 |
| RNN (LSTM, max), hypothesis-only | 73.1 | 56.2 |
| Bag, full input | 71.8 | 55.0 |
| Bag, hypothesis-only | 67.9 | 52.3 |
| Heuristic: pick the hypothesis with more word overlap with O1/O2 | 57.6 | 51.3 |
| Heuristic: pick the longer hypothesis | 46.2 | 49.9 |

- Stories and hypotheses do not leak: 0.0% / 0.0% of val / test (O1, O2) pairs appear in train, and 0.4% / 0.2% of val / test hypotheses do.
- Hypothesis-only RNN keeps 95% of the full RNN's val accuracy (73.1 of 76.6); adding the observations gains 3.6 points on val and -0.1 on test.
- On test, the full RNN and the hypothesis-only RNN give the same prediction on 75.1% of examples.

## Observations

- The RNN beats the bag model on the 3-seed best configs: 56.9 ± 0.9 vs 53.9 ± 0.6 test, a gap of 3.0 points against seed spreads of 0.6 and 0.9. The single seed-42 runs differ by only 1.1 points, which is within the seed spread on its own.
- The seed spread of the test accuracy is 0.6-0.9 points, so the differences between LSTM, GRU and max / last-state pooling in the single runs (54.6-56.9 test) should not be interpreted.
- The observations add nothing measurable on the official dev set: the hypothesis-only RNN scores 56.2 vs 56.1 for the full RNN, even though it loses 3.6 points on val. On val the hypotheses alone already give 73.1, so the models rely mostly on properties of the hypotheses themselves. This is a single seed-42 pair; it does not show which artifacts the models use.
- Surface heuristics that work on held-out train stories (more overlap with the observations: 57.6%; picking the longer hypothesis scores 46.2%, below chance) are at chance on dev (51.3% and 49.9%). This is consistent with the dev set having been adversarially filtered to remove instances that simple models solve (the ART authors' stated procedure; not verified here). Val comes from the train distribution, which is not filtered the same way.
- Sweeps select on val accuracy, which is a weak proxy for test here: the 15 bag sweep runs span 52.6-54.9 test and the 14 runs with val above 70% span only 71.8-73.4 val. The best RNN sweep run reaches 77.6 val / 56.6 test, against 76.6 / 56.1 for the untuned LSTM.

## Notes on this run

- This is a complete rerun (project `art-abductive-v2-full-dataset`, originally `art-abductive-v2`) after replacing the fused `num_layers=2` LSTM/GRU by two stacked single-layer modules with dropout in between. The computation is identical, but the fused version crashed the process at exit on Windows (cuDNN dropout state), which killed the sweep agent. Random draws differ, so numbers differ slightly from the first run.
- Single-run results use seed 42; the 3-seed rows use seeds 1-3 on the best sweep config of each model.
