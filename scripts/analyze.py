"""Val-vs-test gap checks and error analysis for the seed-42 bag and RNN runs of a W&B project.

    uv run python scripts/analyze.py art-abductive-v2   (after collect_results.py)
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from abductive.data import SEP, ARTDataset, Collator, load_splits, tokenize
from abductive.embeddings import build_embeddings
from abductive.models import build_model

PROJECT = sys.argv[1]
RESULTS = json.load(open(Path("logs") / PROJECT / "results.json"))


def ckpt(model, cell=None, pooling=None, hyp=False):
    """Checkpoint of the seed-42 single run (not a sweep run) with this configuration."""
    (r,) = [r for r in RESULTS if not r["sweep"] and r["config"]["seed"] == 42 and r["config"]["model"] == model
            and r["config"]["hypothesis_only"] == hyp and (model == "bag" or (r["config"]["cell"], r["config"]["pooling"]) == (cell, pooling))]
    return f"{r['id']}-{model}"


CKPTS = {
    "bag": ckpt("bag"),
    "rnn": ckpt("rnn", "lstm", "max"),
    "bag_hyp": ckpt("bag", hyp=True),
    "rnn_hyp": ckpt("rnn", "lstm", "max", hyp=True),
}


@torch.no_grad()
def predict(name, examples, stoi, vectors):
    ck = torch.load(Path("checkpoints") / PROJECT / f"{CKPTS[name]}.pt", map_location="cpu", weights_only=False)
    args = argparse.Namespace(**ck["args"])
    model = build_model(args, vectors)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    loader = DataLoader(ARTDataset(examples, stoi), batch_size=512,
                        collate_fn=Collator(args.model, stoi[SEP], args.hypothesis_only))
    return torch.cat([model(b).argmax(-1) for b in loader]).tolist()


def main():
    splits = load_splits()
    stoi, vectors = build_embeddings(splits, 300, Path("data"))
    out, stats = {}, {"ckpts": CKPTS}
    for split in ("val", "test"):
        ex = splits[split]
        gold = [e["label"] - 1 for e in ex]
        preds = {m: predict(m, ex, stoi, vectors) for m in CKPTS}
        out[split] = (ex, gold, preds)
        print(f"\n== {split} (n={len(ex)}) ==")
        print("label balance (h2 correct):", sum(gold) / len(gold))
        stats[split] = {"n": len(ex), "label_balance": sum(gold) / len(gold)}
        for m, p in preds.items():
            print(f"  {m:8s} acc {sum(a == b for a, b in zip(p, gold)) / len(gold):.4f}")
            stats[split][m] = sum(a == b for a, b in zip(p, gold)) / len(gold)
        # simple length heuristic: pick the longer hypothesis
        n = lambda e, k: len(tokenize(e[k]))
        longer = [int(n(e, "hypothesis_2") > n(e, "hypothesis_1")) for e in ex]
        print("  'pick longer hypothesis' acc", sum(a == b for a, b in zip(longer, gold)) / len(gold))
        stats[split]["longer"] = sum(a == b for a, b in zip(longer, gold)) / len(gold)
        # fraction of hypotheses sharing a word with the observations
        def overlap(e, k):
            h = set(tokenize(e[k])); o = set(tokenize(e["observation_1"])) | set(tokenize(e["observation_2"]))
            return len(h & o) / max(len(h), 1)
        ov = [int(overlap(e, "hypothesis_2") > overlap(e, "hypothesis_1")) for e in ex]
        print("  'pick more overlapping' acc", sum(a == b for a, b in zip(ov, gold)) / len(gold))
        stats[split]["overlap"] = sum(a == b for a, b in zip(ov, gold)) / len(gold)

    # overlap of test with train: stories and hypotheses
    train = splits["train"]
    tr_stories = {(e["observation_1"], e["observation_2"]) for e in train}
    tr_hyps = Counter(h for e in train for h in (e["hypothesis_1"], e["hypothesis_2"]))
    val_correct = Counter()
    for e in train:
        val_correct[e["hypothesis_1" if e["label"] == 1 else "hypothesis_2"]] += 1
    for split in ("val", "test"):
        ex = out[split][0]
        stats[split]["stories_in_train"] = sum((e["observation_1"], e["observation_2"]) in tr_stories for e in ex) / len(ex)
        stats[split]["hyps_in_train"] = sum((e[k] in tr_hyps) for e in ex for k in ("hypothesis_1", "hypothesis_2")) / (2 * len(ex))
        print(f"\n{split}: stories seen in train {stats[split]['stories_in_train']:.3f}; hypotheses seen in train {stats[split]['hyps_in_train']:.3f}")

    ex, gold, p = out["test"]
    ok = lambda m, i: p[m][i] == gold[i]
    both_wrong = [i for i in range(len(ex)) if not ok("bag", i) and not ok("rnn", i)]
    rnn_only = [i for i in range(len(ex)) if ok("rnn", i) and not ok("bag", i)]
    bag_only = [i for i in range(len(ex)) if ok("bag", i) and not ok("rnn", i)]
    print(f"\ntest: both wrong {len(both_wrong)}, RNN only right {len(rnn_only)}, bag only right {len(bag_only)}, "
          f"both right {len(ex) - len(both_wrong) - len(rnn_only) - len(bag_only)}")
    hyp_agree = sum(p["rnn"][i] == p["rnn_hyp"][i] for i in range(len(ex))) / len(ex)
    print(f"test: full RNN agrees with hypothesis-only RNN on {hyp_agree:.3f} of examples")
    rnn_only_hyp_wrong = [i for i in rnn_only if not ok("rnn_hyp", i)]
    print(f"test: of {len(rnn_only)} RNN-only-right, hypothesis-only RNN also wrong on {len(rnn_only_hyp_wrong)}")

    def show(i):
        e = ex[i]
        return {"idx": i, "O1": e["observation_1"], "O2": e["observation_2"], "h1": e["hypothesis_1"], "h2": e["hypothesis_2"],
                "gold": gold[i] + 1, "bag": p["bag"][i] + 1, "rnn": p["rnn"][i] + 1, "rnn_hyp": p["rnn_hyp"][i] + 1}

    import random
    r = random.Random(0)
    examples = {
        "both_wrong": [show(i) for i in r.sample(both_wrong, 8)],
        "rnn_only_right": [show(i) for i in r.sample(rnn_only_hyp_wrong or rnn_only, 6)],
    }
    stats["test_agreement"] = {"both_wrong": len(both_wrong), "rnn_only": len(rnn_only), "bag_only": len(bag_only),
                               "rnn_vs_rnn_hyp": hyp_agree, "rnn_only_hyp_also_wrong": len(rnn_only_hyp_wrong)}
    json.dump(stats, open(Path("logs") / PROJECT / "analysis.json", "w"), indent=1)
    json.dump(examples, open(Path("logs") / PROJECT / "examples.json", "w"), indent=1)
    for k, v in examples.items():
        print(f"\n--- {k} ---")
        for s in v:
            print(f"[{s['idx']}] O1: {s['O1']}\n  h1: {s['h1']}\n  h2: {s['h2']}\n  O2: {s['O2']}\n  gold={s['gold']} bag={s['bag']} rnn={s['rnn']} rnn_hyp={s['rnn_hyp']}")


if __name__ == "__main__":
    main()
