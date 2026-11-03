"""Dump all runs of a W&B project to logs/<project>/results.json (val/test acc, params, epoch time, config).

    uv run --env-file .env python scripts/collect_results.py art-abductive-v2
"""

import json
import statistics
import sys
from pathlib import Path

import wandb

PROJECT = sys.argv[1]
api = wandb.Api()
rows = []
for r in api.runs(f"nicolassalat-hochschule-luzern/{PROJECT}"):
    if r.state != "finished" or r.summary.get("best_val_acc") is None:
        continue
    hist = r.history(keys=["epoch_time_s"], pandas=False)
    times = [h["epoch_time_s"] for h in hist if h.get("epoch_time_s") is not None]
    rows.append(
        {
            "id": r.id,
            "name": r.name,
            "group": r.group,
            "sweep": r.sweep.id if r.sweep else None,
            "created": r.created_at,
            "val": r.summary["best_val_acc"],
            "test": r.summary["test/acc"],
            "params": r.summary.get("trainable_params"),
            "epochs": len(times),
            "epoch_time": statistics.mean(times) if times else None,
            "config": {k: r.config.get(k) for k in
                       ["model", "hypothesis_only", "cell", "pooling", "bidirectional", "lr", "hidden",
                        "dropout", "batch_size", "rnn_hidden", "emb_dim", "epochs", "seed", "max_train"]},
        }
    )
rows.sort(key=lambda x: x["created"])
out = Path("logs") / PROJECT / "results.json"
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(out, "w"), indent=1)
for x in rows:
    c = x["config"]
    print(x["name"], x["sweep"] or x["group"] or "-", c["model"], "hyp" if c["hypothesis_only"] else "", c["cell"] if c["model"] == "rnn" else "", c["pooling"] if c["model"] == "rnn" else "",
          f'val={x["val"]:.4f} test={x["test"]:.4f} p={x["params"]} ep={x["epochs"]} t={x["epoch_time"]:.0f}s seed={c["seed"]} max_train={c["max_train"]}')
