"""GatedCPAD -- the differentiable instantiation of CPAD.

Each modeling column B is predicted from a gate-weighted mixture of the other columns'
embeddings:  R[B] = sum_A G[B, A] * E[A], with G[B, A] >= 0 and an L1 penalty that
sparsifies the gates so each target depends on FEW sources (the LHS of its FD). Training
is denoising (random per-cell corruption) and fully unsupervised.

Native extraction : the learned gate row G[B, :] ranks candidate sources; the minimal
rule is then read off by confidence (so the same extractor as the discrete miner gives
the final, human-readable FD/DC, including composite ones).
Native scoring    : the surprise 1 - P(t.B | gated sources).

torch is an optional dependency (`pip install cpad[gated]`).
"""
from __future__ import annotations
import numpy as np

from cpad.core.table import Table
from cpad.models.base import CPADModel
from cpad.rules.confidence import fd_confidence
from cpad.rules.extract import minimal_rule


class GatedCPAD(CPADModel):
    name = "CPAD-gated"

    def __init__(self, dim: int = 24, epochs: int = 250, l1: float = 0.3, lr: float = 0.01,
                 corrupt_p: float = 0.15, max_rows: int = 15000, threads: int = 0, seed: int = 0,
                 tau: float = 0.90, lift: float = 0.10, max_lhs: int = 3):
        self.dim, self.epochs, self.l1, self.lr = dim, epochs, l1, lr
        self.corrupt_p, self.max_rows, self.threads, self.seed = corrupt_p, max_rows, threads, seed
        self.tau, self.lift, self.max_lhs = tau, lift, max_lhs

    def fit(self, table: Table) -> "GatedCPAD":
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        if self.threads:
            torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed); np.random.seed(self.seed)

        self.columns_ = table.columns
        self.cols_ = table.modeling_columns()
        nc = len(self.cols_)
        codes, self.cards_ = table.codes(self.cols_)
        Xfull = torch.tensor(codes, dtype=torch.long)
        if len(Xfull) > self.max_rows:                    # subsample only the gradient training
            sel = np.random.default_rng(self.seed).choice(len(Xfull), self.max_rows, replace=False)
            Xtr = Xfull[sel]
        else:
            Xtr = Xfull
        n = len(Xtr)

        embs = nn.ModuleList([nn.Embedding(c, self.dim) for c in self.cards_])
        heads = nn.ModuleList([nn.Linear(self.dim, c) for c in self.cards_])
        gate_logits = nn.Parameter(torch.zeros(nc, nc) - 2.0)
        eye = torch.eye(nc); ce = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(list(embs.parameters()) + list(heads.parameters()) + [gate_logits], lr=self.lr)
        gates = lambda: F.softplus(gate_logits) * (1 - eye)

        def corrupt(xb):
            xc = xb.clone()
            for j in range(nc):
                m = torch.rand(len(xb)) < self.corrupt_p
                xc[m, j] = xb[torch.randperm(len(xb)), j][m]
            return xc

        for _ in range(self.epochs):
            xin = corrupt(Xtr)
            E = torch.stack([embs[j](xin[:, j]) for j in range(nc)], 1)
            G = gates(); R = torch.einsum("ba,nad->nbd", G, E)
            loss = sum(ce(heads[B](R[:, B, :]), Xtr[:, B]) for B in range(nc)) + self.l1 * G.abs().sum()
            opt.zero_grad(); loss.backward(); opt.step()

        self._torch = torch
        self._embs, self._heads = embs, heads
        self._gates = gates().detach()
        self.gate_matrix_ = self._gates.numpy()
        self._extract_rules(table)
        return self

    def _extract_rules(self, table: Table):
        G = self.gate_matrix_
        df = table.df
        rules = []
        for b, rhs in enumerate(self.cols_):
            order = [self.cols_[a] for a in np.argsort(-G[b])]
            fd = minimal_rule(df, rhs, order, tau=self.tau, lift=self.lift, max_lhs=self.max_lhs)
            if fd is not None:
                rules.append(fd)
        self.rules_ = rules
        self.governed_ = {fd.rhs for fd in rules}

    def score(self, table: Table) -> np.ndarray:
        torch = self._torch
        codes, _ = table.codes(self.cols_)
        X = torch.tensor(codes, dtype=torch.long)
        nc = len(self.cols_)
        S = np.zeros((table.n, len(table.columns)))
        idx = {c: j for j, c in enumerate(table.columns)}
        with torch.no_grad():
            for s in range(0, len(X), 20000):
                xb = X[s:s + 20000]
                E = torch.stack([self._embs[j](xb[:, j]) for j in range(nc)], 1)
                R = torch.einsum("ba,nad->nbd", self._gates, E)
                for b, c in enumerate(self.cols_):
                    p = torch.softmax(self._heads[b](R[:, b, :]), 1)
                    S[s:s + len(xb), idx[c]] = 1.0 - p[torch.arange(len(xb)), xb[:, b]].numpy()
        return S
