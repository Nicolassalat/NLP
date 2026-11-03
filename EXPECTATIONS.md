# Expectations

Expected best **test** accuracy (official ART dev set, chance = 50%) for each model.

> **Written on:** 2026-11-02, _after_ the first experiments had been run (not before the project started).
> based on general reasoning about the task, not on experience with this exact dataset

| Model | Expected test acc (%) | Why |
|---|---|---|
| Bag of embeddings | 50-55% | Knows no word order so how could it really find any patterns |
| RNN (LSTM/GRU, bidirectional, max pooling) | 52-60% | Has word order and context so should be higher than Bag. |
| Hypothesis-only baseline (bag) | ~50% | not really better than random |
| Hypothesis-only baseline (RNN) | ~50% | not really better than random |

## Expected differences

- RNN vs bag: RNN slightly above Bag because of the word order and actual context patterns
- Hypothesis-only vs full input: hypothesis-only near chance. The full models only a few points above it
- Val vs test accuracy: about the same as both should be unseen data by the models

