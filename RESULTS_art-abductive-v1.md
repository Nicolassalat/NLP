> **Superseded.** This is the first run (W&B project `art-abductive-v1-discarded`, formerly `art-abductive`). It used the fused `num_layers=2` LSTM/GRU, which crashed the process at exit on Windows (cuDNN dropout state) and killed the sweep agent. The final results are in `RESULTS_art-abductive-v2.md`.

# Results

Task: ART (alpha-NLI), pick the more plausible hypothesis. Chance is 50%.
All runs are logged in W&B project `art-abductive`. Single runs use seed 42, batch size 256 unless noted.
Val = 5% of the train stories (held out by (O1, O2)); test = official ART dev set (1,532 examples).
Val/test accuracy is taken at the epoch with the best val accuracy. Epoch time is the mean wall-clock time per epoch (training + validation) on an RTX 3060.

## Main experiments (full data)

| Configuration | Val acc (%) | Test acc (%) | Trainable params | Epoch time (s) | Epochs |
|---|---|---|---|---|---|
| Bag of embeddings | 71.8 | 55.0 | 384,513 | 18 | 9 |
| RNN: LSTM, max pooling | 77.1 | 57.8 | 901,633 | 31 | 10 |
| RNN: GRU, max pooling | 76.6 | 56.3 | 692,737 | 35 | 10 |
| RNN: LSTM, last-state pooling | 75.9 | 57.4 | 901,633 | 16 | 10 |
| Bag, hypothesis-only | 67.9 | 52.3 | 77,313 | 6 | 10 |
| RNN (LSTM, max), hypothesis-only | 73.2 | 55.6 | 901,633 | 11 | 9 |

## Best configurations over 3 seeds (1, 2, 3)

Mean ± sample standard deviation (n = 3). Each seed also draws a different val split (the seed is passed to `load_splits`), so the val spread includes split variance; the test set is fixed.

| Model | Val acc (%) | Test acc (%) | Trainable params | Epoch time (s) | Per seed (val / test) |
|---|---|---|---|---|---|
| Bag, best sweep config (lr 8.4e-4, hidden 512, dropout 0.2, batch 256) | 73.5 ± 0.9 | 53.4 ± 0.2 | 769,025 | 22 | seed 1: 74.3 / 53.6; seed 2: 73.6 / 53.5; seed 3: 72.6 / 53.2 |
| RNN, best sweep config (GRU, max pool, rnn_hidden 128, hidden 256, lr 1.3e-3, dropout 0.3, batch 128, 4 epochs) | 76.3 ± 0.4 | 55.5 ± 0.3 | 692,737 | 70 | seed 1: 76.0 / 55.2; seed 2: 76.1 / 55.7; seed 3: 76.8 / 55.5 |

## Sweep: bag of embeddings (15 runs, sorted by val acc)

| Run | lr | hidden | dropout | batch | emb | Val | Test | Params |
|---|---|---|---|---|---|---|---|---|
| polished-sweep-13 | 0.00084 | 512 | 0.2 | 256 | 300 | 73.2 | 54.0 | 769,025 |
| denim-sweep-12 | 0.00052 | 512 | 0.2 | 256 | 300 | 73.0 | 54.2 | 769,025 |
| glowing-sweep-10 | 0.0018 | 512 | 0.2 | 256 | 300 | 73.0 | 54.6 | 769,025 |
| resilient-sweep-14 | 0.00028 | 512 | 0.2 | 256 | 300 | 72.5 | 53.6 | 769,025 |
| lunar-sweep-9 | 0.0022 | 512 | 0.2 | 256 | 300 | 72.5 | 53.6 | 769,025 |
| desert-sweep-15 | 0.00084 | 512 | 0 | 256 | 300 | 72.2 | 53.5 | 769,025 |
| feasible-sweep-11 | 0.00058 | 512 | 0.4 | 256 | 300 | 71.5 | 54.0 | 769,025 |
| dry-sweep-8 | 0.0022 | 512 | 0.4 | 256 | 300 | 70.7 | 53.1 | 769,025 |
| wobbly-sweep-7 | 0.0025 | 512 | 0.4 | 256 | 300 | 70.7 | 52.8 | 769,025 |
| dutiful-sweep-6 | 0.0045 | 512 | 0.4 | 256 | 300 | 70.2 | 52.9 | 769,025 |
| proud-sweep-5 | 0.0092 | 256 | 0.4 | 256 | 300 | 69.7 | 52.0 | 384,513 |
| playful-sweep-4 | 0.0073 | 256 | 0.4 | 256 | 300 | 69.7 | 53.6 | 384,513 |
| eager-sweep-2 | 0.0072 | 128 | 0.4 | 256 | 300 | 69.7 | 51.5 | 192,257 |
| proud-sweep-1 | 0.00073 | 128 | 0.4 | 256 | 100 | 65.9 | 52.2 | 64,257 |
| pleasant-sweep-3 | 0.00039 | 128 | 0.4 | 128 | 100 | 65.6 | 52.7 | 64,257 |

## Sweep: RNN (8 runs, 4 epochs each, sorted by val acc)

| Run | cell | pooling | rnn_hidden | lr | dropout | Val | Test | Params | Epoch time (s) |
|---|---|---|---|---|---|---|---|---|---|
| fresh-sweep-8 | gru | max | 128 | 0.0013 | 0.3 | 76.2 | 56.5 | 692,737 | 60 |
| upbeat-sweep-7 | gru | max | 128 | 0.0015 | 0.3 | 75.5 | 55.6 | 692,737 | 68 |
| sleek-sweep-4 | gru | max | 256 | 0.00095 | 0.3 | 75.3 | 55.1 | 2,171,393 | 65 |
| devoted-sweep-5 | gru | max | 256 | 0.00051 | 0.3 | 75.0 | 54.3 | 2,171,393 | 67 |
| ruby-sweep-1 | gru | last | 128 | 0.00059 | 0.3 | 74.7 | 53.3 | 692,737 | 23 |
| ethereal-sweep-6 | gru | max | 128 | 0.00034 | 0.3 | 73.3 | 54.9 | 692,737 | 59 |
| stellar-sweep-3 | lstm | last | 256 | 0.0015 | 0.3 | 61.1 | 51.3 | 2,851,329 | 41 |
| sweepy-sweep-2 | lstm | last | 256 | 0.0022 | 0.1 | 50.5 | 49.3 | 2,851,329 | 49 |

## Val/test gap

Validation accuracy is 72-77% for every full-data model, but test accuracy is 52-58%. The gap persists on full data
and is not a small-sample effect; the seed spread of the test accuracy is only about 0.2-0.3 points.

Checks (seed-42 val split, official dev as test):

| Check | Val | Test |
|---|---|---|
| RNN (LSTM, max), full input | 77.1 | 57.8 |
| RNN (LSTM, max), hypothesis-only | 73.2 | 55.6 |
| Bag, full input | 71.8 | 55.0 |
| Bag, hypothesis-only | 67.9 | 52.3 |
| Heuristic: pick the hypothesis with more word overlap with O1/O2 | 57.6 | 51.3 |
| Heuristic: pick the longer hypothesis | 46.2 | 49.9 |

- Stories and hypotheses do not leak: 0% of val or test (O1, O2) pairs appear in train, and only 0.4% / 0.2% of val / test hypotheses do.
- Without ever seeing the observations, the RNN keeps 95% of its val accuracy (73.2 of 77.1). Adding the observations gains 3.9 points on val but only 2.2 on test. The models therefore rely mostly on properties of the hypotheses themselves.
- Those properties are much weaker on the dev set: simple surface heuristics that work on held-out train stories (more overlap with the observations: 57.6%; picking the longer hypothesis scores 46.2%, i.e. below chance) are at chance on dev (51.3% and 49.9%). Even the hypothesis-only models fall from 68-73% to 52-56%.
- This is consistent with the dev set having been adversarially filtered to remove instances that simple models solve (the ART authors' stated procedure; not verified here). Val comes from the train distribution, which is not filtered the same way.
- Caveat: the hypothesis-only gap shows the val/test difference is largely about the hypotheses, but it does not prove which specific artifacts the models use.

## Observations

- RNN beats the bag model on test by about 2-3 points (57.8 vs 55.0 for default runs; 55.5 vs 53.4 for the 3-seed best configs). The seed spread is about 0.2-0.3, so this is outside noise.
- LSTM vs GRU and max vs last-state pooling differ by 0.4-1.5 test points in single seed-42 runs, which is within what one would expect from run-to-run variation; do not over-interpret them.
- The RNN sweep (4 epochs) did not beat the default LSTM run (10 epochs): best sweep val 76.2 vs 77.1. The 4-epoch cap and the batch size of 128 are part of that.
- Two of the 8 RNN sweep runs (LSTM, last pooling, rnn_hidden 256) learned little in 4 epochs (50.5% and 61.1% val).
- Sweeps select on val accuracy, which is a weak proxy for test here (e.g. bag sweep test accuracy ranges only 51.5-54.6 while val ranges 65.6-73.2).
