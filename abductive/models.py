"""The two architectures. Both score each hypothesis separately; the two
scores are the logits of a 2-way classification."""

import re

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def classifier(in_dim: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


def mean_pool(emb: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = (torch.arange(emb.size(1), device=emb.device) < lengths[:, None]).unsqueeze(-1)
    return (emb * mask).sum(1) / lengths[:, None]


class BagOfEmbeddings(nn.Module):
    """Architecture 1: averaged frozen embeddings + 2-layer classifier."""

    def __init__(self, vectors: torch.Tensor, hidden: int, dropout: float, hypothesis_only: bool = False):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=True, padding_idx=0)
        self.hypothesis_only = hypothesis_only
        dim = vectors.size(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = classifier(dim if hypothesis_only else 5 * dim, hidden, dropout)

    def encode(self, field):
        tokens, lengths = field
        return mean_pool(self.embedding(tokens), lengths)

    def score(self, o1, o2, h):
        if self.hypothesis_only:
            features = h
        else:
            features = torch.cat([o1, o2, h, o1 * h, o2 * h], dim=-1)
        return self.classifier(self.dropout(features)).squeeze(-1)

    def forward(self, batch):
        o1 = self.encode(batch["observation_1"])
        o2 = self.encode(batch["observation_2"])
        s1 = self.score(o1, o2, self.encode(batch["hypothesis_1"]))
        s2 = self.score(o1, o2, self.encode(batch["hypothesis_2"]))
        return torch.stack([s1, s2], dim=-1)


class RNNScorer(nn.Module):
    """Architecture 2: frozen embeddings -> 2-layer LSTM/GRU -> 2-layer classifier."""

    def __init__(
        self,
        vectors: torch.Tensor,
        cell: str,
        rnn_hidden: int,
        hidden: int,
        dropout: float,
        bidirectional: bool,
        pooling: str,
    ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=True, padding_idx=0)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.bidirectional = bidirectional
        directions = 2 if bidirectional else 1
        # Two stacked single-layer RNNs with dropout in between. This is what
        # nn.LSTM/nn.GRU(num_layers=2, dropout=p) computes, but the built-in
        # inter-layer dropout makes cuDNN keep a dropout state that crashes the
        # process at exit on Windows (exit code 0xC0000409).
        self.layers = nn.ModuleList(
            [
                rnn_cls(in_dim, rnn_hidden, batch_first=True, bidirectional=bidirectional)
                for in_dim in (vectors.size(1), rnn_hidden * directions)
            ]
        )
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = classifier(rnn_hidden * directions, hidden, dropout)
        self._register_load_state_dict_pre_hook(self._rename_fused_keys)

    @staticmethod
    def _rename_fused_keys(state_dict, prefix, *args):
        """Load checkpoints saved with the fused num_layers=2 module (rnn.weight_ih_l1 -> layers.1.weight_ih_l0)."""
        for key in [k for k in state_dict if k.startswith(prefix + "rnn.")]:
            name, layer, reverse = re.fullmatch(r"(.+)_l(\d+)(_reverse)?", key[len(prefix + "rnn.") :]).groups()
            state_dict[f"{prefix}layers.{layer}.{name}_l0{reverse or ''}"] = state_dict.pop(key)

    def forward(self, batch):
        tokens, lengths = batch["tokens"], batch["lengths"]
        emb = self.dropout(self.embedding(tokens))
        out = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        for i, layer in enumerate(self.layers):
            if i:
                out = out._replace(data=self.dropout(out.data))
            out, state = layer(out)

        if self.pooling == "last":
            h_n = state[0] if isinstance(state, tuple) else state  # LSTM returns (h, c)
            rep = torch.cat(list(h_n), dim=-1)  # top layer, all directions
        else:
            out, _ = pad_packed_sequence(out, batch_first=True, padding_value=float("-inf"))
            rep = out.max(dim=1).values

        scores = self.classifier(self.dropout(rep)).squeeze(-1)
        return scores.view(2, -1).t()  # (batch, 2): h1 sequences come first


def build_model(args, vectors: torch.Tensor) -> nn.Module:
    if args.model == "bag":
        return BagOfEmbeddings(vectors, args.hidden, args.dropout, args.hypothesis_only)
    return RNNScorer(vectors, args.cell, args.rnn_hidden, args.hidden, args.dropout, args.bidirectional, args.pooling)
