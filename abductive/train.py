"""Train and evaluate one model on ART.

Examples:
    uv run python -m abductive.train --model bag
    uv run python -m abductive.train --model rnn --cell gru --bidirectional
    uv run python -m abductive.train --model bag --hypothesis-only --no-wandb
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import DataLoader

from abductive.data import SEP, ARTDataset, Collator, load_splits
from abductive.embeddings import build_embeddings
from abductive.models import build_model

load_dotenv()  # WANDB_API_KEY from .env


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["bag", "rnn"], default="bag")
    p.add_argument("--hypothesis-only", action="store_true", help="ignore the observations (bias baseline)")
    p.add_argument("--emb-dim", type=int, choices=[50, 100, 200, 300], default=300)
    p.add_argument("--hidden", type=int, default=256, help="classifier hidden size")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--cell", choices=["lstm", "gru"], default="lstm")
    p.add_argument("--rnn-hidden", type=int, default=128)
    p.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pooling", choices=["last", "max"], default="max")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5, help="epochs without val improvement before stopping")
    p.add_argument("--max-train", type=int, default=None, help="subsample train for quick runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda or cpu; defaults to cuda when available")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--project", default="art-abductive")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--run-name", default=None, help="W&B run name (default: random)")
    return p.parse_args(argv)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(to_device(v, device) for v in batch)
    return batch.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch)
        total_loss += loss_fn(logits, batch["label"]).item() * len(batch["label"])
        correct += (logits.argmax(-1) == batch["label"]).sum().item()
        n += len(batch["label"])
    return total_loss / n, correct / n


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    run = wandb.init(
        project=args.project,
        name=args.run_name,
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        mode="disabled" if args.no_wandb else None,
    )
    # A sweep agent overrides hyperparameters through wandb.config.
    for key, value in run.config.items():
        if hasattr(args, key) and not isinstance(getattr(args, key), Path):
            setattr(args, key, value)

    splits = load_splits()
    stoi, vectors = build_embeddings(splits, args.emb_dim, args.data_dir)
    if args.max_train:
        splits["train"] = random.Random(args.seed).sample(splits["train"], args.max_train)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    collate = Collator(args.model, stoi[SEP], args.hypothesis_only)
    loaders = {
        name: DataLoader(
            ARTDataset(examples, stoi),
            batch_size=args.batch_size if name == "train" else 512,
            shuffle=name == "train",
            collate_fn=collate,
            pin_memory=device.type == "cuda",
        )
        for name, examples in splits.items()
    }
    print({name: len(loader.dataset) for name, loader in loaders.items()})

    model = build_model(args, vectors).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(model)
    print(f"trainable parameters: {sum(p.numel() for p in trainable):,}")
    run.summary["trainable_params"] = sum(p.numel() for p in trainable)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / f"{run.id or 'local'}-{args.model}.pt"  # id, so renaming a run in W&B does not break lookups
    best_val, bad_epochs, step = -1.0, 0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        for batch in loaders["train"]:
            batch = to_device(batch, device)
            logits = model(batch)
            loss = loss_fn(logits, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            step += 1
            if step % 50 == 0:
                acc = (logits.argmax(-1) == batch["label"]).float().mean().item()
                run.log({"train/loss": loss.item(), "train/acc": acc, "epoch": epoch}, step=step)

        train_loss, train_acc = evaluate(model, loaders["train"], loss_fn, device) if args.max_train else (None, None)
        val_loss, val_acc = evaluate(model, loaders["val"], loss_fn, device)
        metrics = {"val/loss": val_loss, "val/acc": val_acc, "epoch": epoch, "epoch_time_s": time.time() - start}
        if train_acc is not None:
            metrics.update({"train/epoch_loss": train_loss, "train/epoch_acc": train_acc})
        run.log(metrics, step=step)
        print(f"epoch {epoch}: val loss {val_loss:.4f} acc {val_acc:.4f} ({time.time() - start:.0f}s)")

        if val_acc > best_val:
            best_val, bad_epochs = val_acc, 0
            torch.save({"args": vars(args), "state_dict": model.state_dict()}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"])
    test_loss, test_acc = evaluate(model, loaders["test"], loss_fn, device)
    print(f"best val acc {best_val:.4f}, test (official dev) acc {test_acc:.4f}")
    run.summary.update({"best_val_acc": best_val, "test/acc": test_acc, "test/loss": test_loss})
    run.finish()


if __name__ == "__main__":
    main()
