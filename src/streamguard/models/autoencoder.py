"""Autoencoder anomaly detector — the "unknown unknowns" lane (PLAN.md §9).

PyTorch MLP d→64→32→8→32→64→d with a sigmoid head, trained on NON-FRAUD train
transactions only. Input = robust-scaled features (train-normal median imputation,
1st–99th percentile clip → [0, 1]) concatenated with a missing-mask channel, so
d = 2 × n_features. Anomaly score = reconstruction MSE, converted to a percentile
against the validation-normal error distribution (`calibrated_percentile`).
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass

import numpy as np

HIDDEN = (64, 32, 8, 32, 64)


def import_torch():
    """torch, made safe to share a process with XGBoost. On macOS torch bundles its
    own libomp next to XGBoost's (Homebrew) one; both reproduced failure modes:
    importing torch first → segfault on XGBoost's first OpenMP use; both OpenMP pools
    active → deadlock. So: XGBoost's runtime loads first, torch runs single-threaded
    (the AE is tiny, so one thread costs little)."""
    importlib.import_module("xgboost")  # explicit call: an import sorter can't reorder it
    import torch

    torch.set_num_threads(1)
    return torch


@dataclass
class Preprocessor:
    """Train-normal statistics → [0, 1] scaled features + missing mask."""

    median: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def fit(cls, x_normal: np.ndarray) -> Preprocessor:
        with warnings.catch_warnings():  # all-NaN columns → nan stats, replaced below
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(x_normal, axis=0)
            lo, hi = np.nanquantile(x_normal, [0.01, 0.99], axis=0)
        median, lo, hi = (np.nan_to_num(a, nan=0.0) for a in (median, lo, hi))
        hi = np.where(hi > lo, hi, lo + 1.0)  # constant column → avoid /0
        return cls(median=median, lo=lo, hi=hi)

    def transform(self, x: np.ndarray) -> np.ndarray:
        mask = np.isnan(x)
        filled = np.where(mask, self.median, x)
        scaled = (np.clip(filled, self.lo, self.hi) - self.lo) / (self.hi - self.lo)
        return np.concatenate([scaled, mask], axis=1).astype(np.float32)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {"median": self.median, "lo": self.lo, "hi": self.hi}


def build_network(dim: int):
    nn = import_torch().nn

    sizes = (dim, *HIDDEN, dim)
    layers: list = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:], strict=False)):
        layers.append(nn.Linear(a, b))
        layers.append(nn.Sigmoid() if i == len(sizes) - 2 else nn.ReLU())
    return nn.Sequential(*layers)


def train_autoencoder(z_normal: np.ndarray, epochs: int = 100, batch_size: int = 512,
                      lr: float = 1e-3, seed: int = 42, log_every: int = 10):  # fmt: skip
    """Fit on preprocessed non-fraud rows; returns (network, per-epoch mean losses)."""
    torch = import_torch()

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = build_network(z_normal.shape[1])
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    data = torch.from_numpy(z_normal)
    losses = []
    for epoch in range(epochs):
        order = torch.from_numpy(rng.permutation(len(data)))
        total = 0.0
        for start in range(0, len(data), batch_size):
            batch = data[order[start : start + batch_size]]
            optimizer.zero_grad()
            loss = ((net(batch) - batch) ** 2).mean()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / len(data))
        if log_every and (epoch + 1) % log_every == 0:
            print(f"AE epoch {epoch + 1}/{epochs}: loss {losses[-1]:.6f}", flush=True)
    return net.eval(), losses


def reconstruction_error(net, z: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    torch = import_torch()

    errors = []
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            batch = torch.from_numpy(z[start : start + batch_size])
            errors.append(((net(batch) - batch) ** 2).mean(dim=1).numpy())
    return np.concatenate(errors) if errors else np.empty(0)


def calibrated_percentile(errors: np.ndarray, reference_sorted: np.ndarray) -> np.ndarray:
    """Share of validation-normal errors ≤ each error → anomaly score in [0, 1]."""
    return np.searchsorted(reference_sorted, errors, side="right") / reference_sorted.size
