"""Non-pretrained 1D-CNN baseline over one-hot DNA (DeepBind/DeepSEA-style:
conv -> global pool -> dense head). No HF Trainer, no tokenizer, no
pretrained weights. Supports MC dropout and from-scratch deep ensembles
(independently initialized and trained, no shared pretraining)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
SEQ_LEN = 300


def one_hot_encode(seqs) -> torch.Tensor:
    """[N] sequences -> [N, 4, L] one-hot tensor. Any base outside ACGT
    (e.g. 'N') encodes as all-zero at that position."""
    n = len(seqs)
    out = np.zeros((n, 4, SEQ_LEN), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, b in enumerate(s):
            idx = BASE_TO_IDX.get(b)
            if idx is not None:
                out[i, idx, j] = 1.0
    return torch.from_numpy(out)


class SmallCNN(nn.Module):
    """DeepBind/DeepSEA-lite: one conv+pool block, global max pool, MLP head.
    Two dropout layers (one on pooled conv features, one mid-MLP) so
    MC-dropout has real stochasticity to sample."""

    def __init__(self, n_classes: int = 2, dropout_p: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=9, padding=4)
        self.drop1 = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):  # x: [B, 4, L]
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 4)
        x = F.relu(self.conv2(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # [B, 128]
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)  # [B, n_classes]


def train_cnn(
    X_train: torch.Tensor, y_train: torch.Tensor, seed: int,
    device: str, epochs: int = 15, batch_size: int = 128, lr: float = 1e-3,
    dropout_p: float = 0.3,
) -> SmallCNN:
    torch.manual_seed(seed)
    model = SmallCNN(dropout_p=dropout_p).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    n = X_train.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1000 + epoch))
        model.train()
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def batched_logits(model: SmallCNN, X: torch.Tensor, device: str, batch_size: int = 256) -> torch.Tensor:
    outs = []
    for start in range(0, X.shape[0], batch_size):
        outs.append(model(X[start:start + batch_size].to(device)).cpu())
    return torch.cat(outs, dim=0)
