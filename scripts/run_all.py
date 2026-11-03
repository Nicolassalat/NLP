"""Run every experiment from scratch into one W&B project: main runs, both sweeps, 3 seeds of the best configs.

    uv run python scripts/run_all.py --project art-abductive-v2

Resumable: a finished step leaves logs/<tag>/<step>.done and is skipped on the next call.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import wandb
import yaml
from dotenv import load_dotenv

load_dotenv()  # WANDB_API_KEY / base URL from .env, needed for the API lookup of the best sweep runs

ENTITY = "nicolassalat-hochschule-luzern"
MAIN_RUNS = {  # step -> train.py arguments (seed 42, batch size 128)
    "01_bag": ["--model", "bag", "--run-name", "bag"],
    "02_rnn_lstm_max": ["--model", "rnn", "--cell", "lstm", "--pooling", "max", "--run-name", "rnn-lstm-max"],
    "03_rnn_gru_max": ["--model", "rnn", "--cell", "gru", "--pooling", "max", "--run-name", "rnn-gru-max"],
    "04_rnn_lstm_last": ["--model", "rnn", "--cell", "lstm", "--pooling", "last", "--run-name", "rnn-lstm-last"],
    "05a_bag_hyp": ["--model", "bag", "--hypothesis-only", "--run-name", "bag-hyp-only"],
    "05b_rnn_hyp": ["--model", "rnn", "--cell", "lstm", "--pooling", "max", "--hypothesis-only", "--run-name", "rnn-hyp-only"],
}
SWEEPS = {"bag": 20, "rnn": 12}  # runs per sweep
BEST_KEYS = ["lr", "hidden", "dropout", "batch_size", "emb_dim", "cell", "pooling", "rnn_hidden", "epochs"]
SEEDS = [1, 2, 3]


def run(cmd, log: Path):
    """Run cmd, tee its output into log, return the exit code."""
    print("$", " ".join(map(str, cmd)), flush=True)
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        return p.wait()


def step(name, logdir, fn):
    done = logdir / f"{name}.done"
    if done.exists():
        print(f"skip {name} (done)", flush=True)
        return
    fn()
    done.write_text("ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    args = ap.parse_args()
    tag = args.project
    logdir, ckpt = Path("logs") / tag, Path("checkpoints") / tag
    logdir.mkdir(parents=True, exist_ok=True)
    train = ["uv", "run", "python", "-m", "abductive.train", "--project", args.project, "--out-dir", str(ckpt)]
    uv_wandb = ["uv", "run", "--env-file", ".env", "wandb"]

    def train_step(name, extra, check=True):
        def go():
            code = run(train + extra, logdir / f"{name}.log")
            if code != 0 and check:
                sys.exit(f"{name} failed with exit code {code}")
        step(name, logdir, go)

    for name, extra in MAIN_RUNS.items():
        train_step(name, extra + ["--batch-size", "128"])

    sweep_ids = {}
    for model, count in SWEEPS.items():
        idfile = logdir / f"sweep_{model}.id"

        def sweep(model=model, count=count, idfile=idfile):
            # copy the sweep config so every run writes its checkpoint under this project's folder
            cfg = yaml.safe_load(open(f"sweeps/{model}.yaml"))
            cfg["command"] += ["--out-dir", str(ckpt)]
            cfg_path = logdir / f"sweep_{model}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            if not idfile.exists():
                log = logdir / f"sweep_{model}_create.log"
                if run(uv_wandb + ["sweep", "--project", args.project, str(cfg_path)], log) != 0:
                    sys.exit(f"creating the {model} sweep failed, see {log}")
                idfile.write_text(re.search(r"wandb agent (\S+)", log.read_text(encoding="utf-8")).group(1))
            code = run(uv_wandb + ["agent", "--count", str(count), idfile.read_text()], logdir / f"sweep_{model}_agent.log")
            if code != 0:
                sys.exit(f"{model} sweep agent exited with {code}")

        step(f"sweep_{model}", logdir, sweep)
        sweep_ids[model] = idfile.read_text().split("/")[-1]

    project_runs = list(wandb.Api().runs(f"{ENTITY}/{args.project}"))
    for model in SWEEPS:
        runs = [r for r in project_runs if r.sweep and r.sweep.id == sweep_ids[model] and r.state == "finished"]
        print(f"{model} sweep {sweep_ids[model]}: {len(runs)} finished runs", flush=True)
        best = max(runs, key=lambda r: r.summary["best_val_acc"])
        extra = ["--model", model]
        for k in BEST_KEYS:
            if k in best.config and (model == "rnn" or k not in ("cell", "pooling", "rnn_hidden")):
                extra += [f"--{k.replace('_', '-')}", str(best.config[k])]
        print(f"best {model} sweep run: {best.name} val={best.summary['best_val_acc']:.4f}", flush=True)
        (logdir / f"best_{model}.txt").write_text(f"{best.name} {' '.join(extra)}\n")
        for seed in SEEDS:
            train_step(f"seed_{model}_{seed}", extra + ["--seed", str(seed), "--run-name", f"{model}-best-seed{seed}"])

    print("all experiments finished", flush=True)


if __name__ == "__main__":
    main()
