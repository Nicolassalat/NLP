"""Build RESULTS_<project>.md for a W&B project from logs/<project>/results.json and analysis.json.

    uv run --env-file .env python scripts/collect_results.py art-abductive-v2
    uv run python -m scripts.analyze art-abductive-v2
    uv run python scripts/make_results.py art-abductive-v2
"""

import json
import statistics as st
import sys
from pathlib import Path

PROJECT = sys.argv[1]
rows = json.load(open(Path("logs") / PROJECT / "results.json"))
stats = json.load(open(Path("logs") / PROJECT / "analysis.json"))
pct = lambda x: f"{100 * x:.1f}"


def find(**kw):
    """The single run whose config matches kw (hypothesis_only defaults to False); sweep runs are excluded."""
    kw.setdefault("hypothesis_only", False)
    kw.setdefault("seed", 42)
    (r,) = [r for r in rows if not r["sweep"] and all(r["config"][k] == v for k, v in kw.items())]
    return r


def line(label, r):
    return f"| {label} | {pct(r['val'])} | {pct(r['test'])} | {r['params']:,} | {r['epoch_time']:.0f} | {r['epochs']} |"


HEAD = "| Configuration | Val acc (%) | Test acc (%) | Trainable params | Epoch time (s) | Epochs |\n|---|---|---|---|---|---|"
main = {
    "bag": ("Bag of embeddings", find(model="bag")),
    "lstm_max": ("RNN: LSTM, max pooling", find(model="rnn", cell="lstm", pooling="max")),
    "gru_max": ("RNN: GRU, max pooling", find(model="rnn", cell="gru", pooling="max")),
    "lstm_last": ("RNN: LSTM, last-state pooling", find(model="rnn", cell="lstm", pooling="last")),
    "bag_hyp": ("Bag, hypothesis-only", find(model="bag", hypothesis_only=True)),
    "rnn_hyp": ("RNN (LSTM, max), hypothesis-only", find(model="rnn", cell="lstm", pooling="max", hypothesis_only=True)),
}


def sweep_runs(model):
    return sorted((r for r in rows if r["sweep"] and r["config"]["model"] == model), key=lambda r: -r["val"])


def sweep_table(model):
    rs = sweep_runs(model)
    if model == "bag":
        h = "| Run | lr | hidden | dropout | batch | emb | Val | Test | Params |\n|---|---|---|---|---|---|---|---|---|"
        body = [f"| {r['name']} | {r['config']['lr']:.2g} | {r['config']['hidden']} | {r['config']['dropout']} | {r['config']['batch_size']} | {r['config']['emb_dim']} | {pct(r['val'])} | {pct(r['test'])} | {r['params']:,} |" for r in rs]
    else:
        h = "| Run | cell | pooling | rnn_hidden | lr | dropout | Val | Test | Params | Epoch time (s) |\n|---|---|---|---|---|---|---|---|---|---|"
        body = [f"| {r['name']} | {r['config']['cell']} | {r['config']['pooling']} | {r['config']['rnn_hidden']} | {r['config']['lr']:.2g} | {r['config']['dropout']} | {pct(r['val'])} | {pct(r['test'])} | {r['params']:,} | {r['epoch_time']:.0f} |" for r in rs]
    return h + "\n" + "\n".join(body)


def seeds(model):
    rs = sorted((r for r in rows if not r["sweep"] and r["config"]["model"] == model and r["config"]["seed"] in (1, 2, 3)), key=lambda r: r["config"]["seed"])
    assert len(rs) == 3, f"expected 3 seed runs for {model}, found {len(rs)}"
    f = lambda k: (st.mean(r[k] for r in rs), st.stdev(r[k] for r in rs))
    (vm, vs), (tm, ts) = f("val"), f("test")
    c = rs[0]["config"]
    if model == "bag":
        desc = f"Bag, best sweep config (lr {c['lr']:.2g}, hidden {c['hidden']}, dropout {c['dropout']}, batch {c['batch_size']}, emb {c['emb_dim']})"
    else:
        desc = (f"RNN, best sweep config ({c['cell'].upper()}, {c['pooling']} pool, rnn_hidden {c['rnn_hidden']}, hidden {c['hidden']}, "
                f"lr {c['lr']:.2g}, dropout {c['dropout']}, batch {c['batch_size']}, {c['epochs']} epochs)")
    per = "; ".join(f"seed {r['config']['seed']}: {pct(r['val'])} / {pct(r['test'])}" for r in rs)
    return (vm, vs, tm, ts), f"| {desc} | {pct(vm)} ± {pct(vs)} | {pct(tm)} ± {pct(ts)} | {rs[0]['params']:,} | {st.mean(r['epoch_time'] for r in rs):.0f} | {per} |"


(bag_vm, _, bag_tm, bag_ts), bag_seed = seeds("bag")
(rnn_vm, _, rnn_tm, rnn_ts), rnn_seed = seeds("rnn")
v, t = stats["val"], stats["test"]
ag = stats["test_agreement"]
m = {k: r for k, (_, r) in main.items()}
all_main = [r for _, r in main.values()]
full = [m["bag"], m["lstm_max"], m["gru_max"], m["lstm_last"]]
bag_sw, rnn_sw = sweep_runs("bag"), sweep_runs("rnn")
rng = lambda rs, k: f"{min(pct(r[k]) for r in rs)}-{max(pct(r[k]) for r in rs)}"

doc = f"""# Results

Task: ART (alpha-NLI), pick the more plausible hypothesis. Chance is 50%.
All runs are logged in W&B project `{PROJECT}`. Single runs use seed 42 and batch size 128 unless noted.
Data: course version of ART (`amarfurt/art-nlp-hs26`), train 16,801 / val 1,000 / test 1,532 examples; the test set is the official ART dev set.
Val/test accuracy is taken at the epoch with the best val accuracy. Epoch time is the mean wall-clock time per epoch (training + validation) on an RTX 3060.

## Main experiments (full data)

{HEAD}
""" + "\n".join(line(label, r) for label, r in main.values()) + f"""

## Best configurations over 3 seeds (1, 2, 3)

Mean ± sample standard deviation (n = 3). The seed changes the initialization and the batch order; the splits are fixed.

| Model | Val acc (%) | Test acc (%) | Trainable params | Epoch time (s) | Per seed (val / test) |
|---|---|---|---|---|---|
{bag_seed}
{rnn_seed}

## Sweep: bag of embeddings ({len(bag_sw)} runs, sorted by val acc)

{sweep_table("bag")}

## Sweep: RNN ({len(rnn_sw)} runs, {rnn_sw[0]['config']['epochs']} epochs max each, sorted by val acc)

{sweep_table("rnn")}

## Val/test gap

Validation accuracy is {rng(full + [m['bag_hyp'], m['rnn_hyp']], 'val')}% across the single runs, but test accuracy is {rng(all_main, 'test')}%.

Checks (course val split; test = official ART dev set):

| Check | Val | Test |
|---|---|---|
| RNN (LSTM, max), full input | {pct(m['lstm_max']['val'])} | {pct(m['lstm_max']['test'])} |
| RNN (LSTM, max), hypothesis-only | {pct(m['rnn_hyp']['val'])} | {pct(m['rnn_hyp']['test'])} |
| Bag, full input | {pct(m['bag']['val'])} | {pct(m['bag']['test'])} |
| Bag, hypothesis-only | {pct(m['bag_hyp']['val'])} | {pct(m['bag_hyp']['test'])} |
| Heuristic: pick the hypothesis with more word overlap with O1/O2 | {pct(v['overlap'])} | {pct(t['overlap'])} |
| Heuristic: pick the longer hypothesis | {pct(v['longer'])} | {pct(t['longer'])} |

- Overlap with train: {pct(v['stories_in_train'])}% / {pct(t['stories_in_train'])}% of val / test (O1, O2) pairs appear in train, and {pct(v['hyps_in_train'])}% / {pct(t['hyps_in_train'])}% of val / test hypotheses do.
- Hypothesis-only RNN keeps {100 * m['rnn_hyp']['val'] / m['lstm_max']['val']:.0f}% of the full RNN's val accuracy ({pct(m['rnn_hyp']['val'])} of {pct(m['lstm_max']['val'])}); adding the observations gains {100 * (m['lstm_max']['val'] - m['rnn_hyp']['val']):.1f} points on val and {100 * (m['lstm_max']['test'] - m['rnn_hyp']['test']):.1f} on test.
- On test, the full RNN and the hypothesis-only RNN give the same prediction on {pct(ag['rnn_vs_rnn_hyp'])}% of examples.
"""
# The observations are interpretation, so they are written by hand after reading the numbers above
# (logs/<project>/observations.md) instead of being generated.
obs = Path("logs") / PROJECT / "observations.md"
doc += "\n## Observations\n\n" + (obs.read_text(encoding="utf-8").strip() if obs.exists() else "_TODO: write after checking the numbers above._") + "\n"
doc += f"""
## Notes on this run

- Project `{PROJECT}`: one complete run (main runs, both sweeps, 3 seeds of each best sweep config) on the course dataset `amarfurt/art-nlp-hs26`. Single-run results use seed 42; the 3-seed rows use seeds 1-3.
- Earlier runs on the original `allenai/art` data: `RESULTS_art-abductive-v2.md` (final run before switching to the course data) and `RESULTS_art-abductive-v1.md` (superseded, fused-LSTM crash).
"""
Path(f"RESULTS_{PROJECT}.md").write_text(doc, encoding="utf-8")
print(doc)
