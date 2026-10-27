"""The two architectures. Both score each hypothesis separately; the two
scores are the logits of a 2-way classification."""

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
        self.rnn = rnn_cls(
            vectors.size(1),
            rnn_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        out_dim = rnn_hidden * (2 if bidirectional else 1)
        self.classifier = classifier(out_dim, hidden, dropout)

    def forward(self, batch):
        tokens, lengths = batch["tokens"], batch["lengths"]
        emb = self.dropout(self.embedding(tokens))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, state = self.rnn(packed)

        if self.pooling == "last":
            h_n = state[0] if isinstance(state, tuple) else state  # LSTM returns (h, c)
            directions = 2 if self.rnn.bidirectional else 1
            rep = torch.cat(list(h_n[-directions:]), dim=-1)  # top layer, all directions
        else:
            out, _ = pad_packed_sequence(out, batch_first=True, padding_value=float("-inf"))
            rep = out.max(dim=1).values

        scores = self.classifier(self.dropout(rep)).squeeze(-1)
        return scores.view(2, -1).t()  # (batch, 2): h1 sequences come first


def build_model(args, vectors: torch.Tensor) -> nn.Module:
    if args.model == "bag":
        return BagOfEmbeddings(vectors, args.hidden, args.dropout, args.hypothesis_only)
    return RNNScorer(vectors, args.cell, args.rnn_hidden, args.hidden, args.dropout, args.bidirectional, args.pooling)
