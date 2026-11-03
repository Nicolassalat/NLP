# Results

Task: ART (alpha-NLI), pick the more plausible hypothesis. Chance is 50%.
All runs are logged in W&B project `art-abductive-v3-course-dataset`. Single runs use seed 42 and batch size 128 unless noted.
Data: course version of ART (`amarfurt/art-nlp-hs26`), train 16,801 / val 1,000 / test 1,532 examples; the test set is the official ART dev set.
Val/test accuracy is taken at the epoch with the best val accuracy. Epoch time is the mean wall-clock time per epoch (training + validation) on an RTX 3060.

## Main experiments (full data)

| Configuration | Val acc (%) | Test acc (%) | Trainable params | Epoch time (s) | Epochs |
|---|---|---|---|---|---|
| Bag of embeddings | 71.6 | 53.0 | 384,513 | 1 | 27 |
| RNN: LSTM, max pooling | 73.3 | 52.9 | 901,633 | 3 | 18 |
| RNN: GRU, max pooling | 73.8 | 53.0 | 692,737 | 4 | 19 |
| RNN: LSTM, last-state pooling | 74.1 | 53.3 | 901,633 | 3 | 24 |
| Bag, hypothesis-only | 66.6 | 50.8 | 77,313 | 1 | 18 |
| RNN (LSTM, max), hypothesis-only | 70.5 | 53.5 | 901,633 | 2 | 16 |

## Best configurations over 3 seeds (1, 2, 3)

Mean ± sample standard deviation (n = 3). The seed changes the initialization and the batch order; the splits are fixed.

| Model | Val acc (%) | Test acc (%) | Trainable params | Epoch time (s) | Per seed (val / test) |
|---|---|---|---|---|---|
| Bag, best sweep config (lr 0.0013, hidden 512, dropout 0.2, batch 256, emb 300) | 72.0 ± 0.2 | 52.9 ± 0.4 | 769,025 | 2 | seed 1: 72.2 / 52.7; seed 2: 72.1 / 52.7; seed 3: 71.8 / 53.4 |
| RNN, best sweep config (GRU, max pool, rnn_hidden 256, hidden 256, lr 0.00074, dropout 0.1, batch 128, 20 epochs) | 74.3 ± 1.0 | 53.3 ± 1.2 | 2,171,393 | 7 | seed 1: 73.3 / 52.0; seed 2: 75.2 / 54.5; seed 3: 74.5 / 53.3 |

## Sweep: bag of embeddings (20 runs, sorted by val acc)

| Run | lr | hidden | dropout | batch | emb | Val | Test | Params |
|---|---|---|---|---|---|---|---|---|
| honest-sweep-18 | 0.0013 | 512 | 0.2 | 256 | 300 | 72.6 | 53.1 | 769,025 |
| ruby-sweep-19 | 0.0022 | 512 | 0.2 | 256 | 300 | 72.1 | 53.3 | 769,025 |
| fine-sweep-10 | 0.0019 | 256 | 0.2 | 256 | 300 | 71.8 | 53.3 | 384,513 |
| solar-sweep-12 | 0.0024 | 256 | 0.4 | 256 | 300 | 71.4 | 53.7 | 384,513 |
| denim-sweep-6 | 0.00091 | 512 | 0.2 | 128 | 300 | 71.3 | 54.0 | 769,025 |
| eager-sweep-16 | 0.0012 | 512 | 0.4 | 256 | 300 | 70.8 | 53.7 | 769,025 |
| vocal-sweep-15 | 0.0031 | 512 | 0.4 | 256 | 300 | 70.7 | 52.7 | 769,025 |
| driven-sweep-14 | 0.0011 | 512 | 0.4 | 256 | 300 | 70.6 | 52.9 | 769,025 |
| clear-sweep-17 | 0.0023 | 512 | 0.4 | 256 | 300 | 70.6 | 52.6 | 769,025 |
| atomic-sweep-20 | 0.0029 | 512 | 0.4 | 256 | 300 | 70.6 | 52.8 | 769,025 |
| hearty-sweep-9 | 0.00075 | 512 | 0.4 | 256 | 300 | 70.5 | 53.6 | 769,025 |
| fiery-sweep-7 | 0.0015 | 512 | 0.4 | 256 | 300 | 70.4 | 52.5 | 769,025 |
| breezy-sweep-13 | 0.0028 | 512 | 0.4 | 256 | 300 | 70.1 | 52.4 | 769,025 |
| playful-sweep-5 | 0.0074 | 128 | 0 | 64 | 300 | 69.8 | 52.7 | 192,257 |
| colorful-sweep-4 | 0.00035 | 128 | 0 | 128 | 300 | 69.4 | 51.8 | 192,257 |
| logical-sweep-8 | 0.00046 | 512 | 0.4 | 256 | 300 | 68.5 | 52.5 | 769,025 |
| fiery-sweep-11 | 0.0004 | 256 | 0 | 256 | 100 | 68.3 | 53.9 | 128,513 |
| electric-sweep-3 | 0.00017 | 256 | 0 | 256 | 100 | 68.1 | 53.9 | 128,513 |
| swift-sweep-1 | 0.00075 | 128 | 0 | 256 | 100 | 67.8 | 53.9 | 64,257 |
| ethereal-sweep-2 | 0.00076 | 256 | 0 | 128 | 100 | 67.8 | 53.6 | 128,513 |

## Sweep: RNN (12 runs, 20 epochs max each, sorted by val acc)

| Run | cell | pooling | rnn_hidden | lr | dropout | Val | Test | Params | Epoch time (s) |
|---|---|---|---|---|---|---|---|---|---|
| fanciful-sweep-4 | gru | max | 256 | 0.00074 | 0.1 | 74.8 | 54.7 | 2,171,393 | 7 |
| sage-sweep-5 | gru | max | 256 | 0.00051 | 0.1 | 73.8 | 52.7 | 2,171,393 | 6 |
| resilient-sweep-10 | gru | max | 256 | 0.0016 | 0.1 | 73.8 | 55.2 | 2,171,393 | 7 |
| jumping-sweep-2 | gru | max | 256 | 0.00056 | 0.1 | 73.6 | 53.5 | 2,171,393 | 7 |
| smooth-sweep-12 | lstm | last | 256 | 0.00055 | 0.3 | 73.1 | 54.2 | 2,851,329 | 5 |
| sleek-sweep-11 | gru | last | 256 | 0.0027 | 0.1 | 73.0 | 51.8 | 2,171,393 | 6 |
| pretty-sweep-8 | gru | max | 256 | 0.00032 | 0.3 | 72.8 | 53.4 | 2,171,393 | 7 |
| divine-sweep-9 | lstm | last | 256 | 0.0003 | 0.3 | 72.7 | 51.5 | 2,851,329 | 5 |
| iconic-sweep-6 | gru | max | 256 | 0.00039 | 0.3 | 72.4 | 52.1 | 2,171,393 | 7 |
| earthy-sweep-3 | gru | last | 128 | 0.002 | 0.3 | 71.9 | 51.8 | 692,737 | 6 |
| sleek-sweep-7 | gru | max | 256 | 0.00053 | 0.3 | 71.9 | 53.4 | 2,171,393 | 7 |
| firm-sweep-1 | lstm | max | 128 | 0.0013 | 0.3 | 71.5 | 52.2 | 901,633 | 6 |

## Val/test gap

Validation accuracy is 66.6-74.1% across the single runs, but test accuracy is 50.8-53.5%.

Checks (course val split; test = official ART dev set):

| Check | Val | Test |
|---|---|---|
| RNN (LSTM, max), full input | 73.3 | 52.9 |
| RNN (LSTM, max), hypothesis-only | 70.5 | 53.5 |
| Bag, full input | 71.6 | 53.0 |
| Bag, hypothesis-only | 66.6 | 50.8 |
| Heuristic: pick the hypothesis with more word overlap with O1/O2 | 58.4 | 51.3 |
| Heuristic: pick the longer hypothesis | 45.8 | 49.9 |

- Overlap with train: 0.0% / 0.0% of val / test (O1, O2) pairs appear in train, and 0.2% / 0.1% of val / test hypotheses do.
- Hypothesis-only RNN keeps 96% of the full RNN's val accuracy (70.5 of 73.3); adding the observations gains 2.8 points on val and -0.5 on test.
- On test, the full RNN and the hypothesis-only RNN give the same prediction on 75.1% of examples.

## Observations

- **All models are close to chance on test.** The full-input models reach 52.0-54.5% test accuracy over every single and seed run (chance is 50%). The best bag config gives 52.9 ± 0.4% and the best RNN config 53.3 ± 1.2% over 3 seeds. The 0.3 point difference is smaller than the RNN's seed-to-seed spread, so the RNN is not shown to be better than the bag model on test. The test loss of the three single-run RNNs is 0.93-1.15 (bag: 0.76), above the 0.69 of a constant 50/50 guess, so they are confidently wrong on many test examples.
- **The validation set looks much easier than the test set.** Val accuracy is 66-75% for all models and test is 51-55%, a gap of about 20 points (bag best config: 72.0% val, 52.9% test). On val the RNN is ahead of the bag model by 2.3 points (74.3% vs 72.0%), on test it is not.
- **On val, most of the signal is in the hypothesis alone.** Hypothesis-only models reach 66.4% (bag) and 70.5% (RNN) on val, i.e. the hypothesis-only RNN keeps 96% of the full RNN's val accuracy. On test the same models get 50.7% and 53.5%. Adding O1 and O2 gains 2.8 points on val for the RNN and nothing on test.
- **Simple surface heuristics work on val but not on test.** "Pick the hypothesis with more word overlap with O1/O2" gets 58.4% on val and 51.3% on test. "Pick the longer hypothesis" gets 45.8% on val (so picking the shorter one gets 54.2%) and 49.9% on test. Label balance is about 49% on both splits, and no (O1, O2) pair of val or test is in train (hypothesis overlap with train: 0.25% val, 0.07% test), so the gap is not caused by leakage from train.
- **Interpretation.** The models mainly pick up regularities of the hypotheses that exist in the train and val data and that do not carry over to the official dev set. This fits the way ART was built, with pairs filtered that a BERT model could solve, but I did not test that explanation directly, so it is a hypothesis. My expectation in EXPECTATIONS.md that val and test would be about the same was wrong, and so was the expectation that hypothesis-only models would be near chance (true on test only).
- **Selecting on val does not help on test.** Across the 20 bag sweep runs, val accuracy ranges from 67.8% to 72.6% while test accuracy stays between 51.8% and 54.0%, with no correlation (r = -0.09). For the 12 RNN runs val ranges from 71.5% to 74.8% and test from 51.5% to 55.2%. The sweeps picked the best configs by val, but on test the differences are within noise.
- **Test errors overlap strongly.** Bag and RNN are both wrong on 461 of 1,532 test examples and disagree on 520 (260 each way). The full RNN and the hypothesis-only RNN give the same prediction on 75% of test examples.

## Notes on this run

- Project `art-abductive-v3-course-dataset`: one complete run (main runs, both sweeps, 3 seeds of each best sweep config) on the course dataset `amarfurt/art-nlp-hs26`. Single-run results use seed 42; the 3-seed rows use seeds 1-3.
- Earlier runs on the original `allenai/art` data: `RESULTS_art-abductive-v2.md` (final run before switching to the course data) and `RESULTS_art-abductive-v1.md` (superseded, fused-LSTM crash).
