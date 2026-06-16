from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

# RadioML2016.10a has 11 classes. Your synthetic generator uses 5.
# The active list is set at runtime depending on the data source.
RADIOML_MODULATIONS = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
    "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
]
SYNTHETIC_MODULATIONS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]
SNRS = [-10, -6, -2, 2, 6, 10, 14, 18]


# ─────────────────────────────────────────────
#  SECTION 1 — DATA SOURCES
#  Two paths: real RadioML dataset OR synthetic
#  generator. Both produce the same output
#  format: (X, y, snr_array).
# ─────────────────────────────────────────────

def load_radioml(pkl_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load the RadioML2016.10a pickle file.

    The file is a dict with keys (modulation_name, snr_int).
    Each value has shape (1000, 2, 128):
        1000 samples, 2 channels (I and Q), 128 time steps.

    Returns
    -------
    X     : float32 array  (N, 2, 128)  — normalised I/Q frames
    y     : int array      (N,)          — class index
    snr   : int array      (N,)          — SNR in dB
    mods  : list[str]                    — class names (index = label)
    """
    print(f"Loading RadioML dataset from {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # Discover the modulation names and SNR levels present in the file.
    # sorted() gives a deterministic, alphabetical order → stable label map.
    mods = sorted(set(k[0] for k in data.keys()))
    snrs_available = sorted(set(k[1] for k in data.keys()))
    print(f"  Modulations : {mods}")
    print(f"  SNR levels  : {snrs_available}")

    xs, ys, snr_list = [], [], []
    for label, mod in enumerate(mods):
        for snr in snrs_available:
            samples = data[(mod, snr)]          # shape (1000, 2, 128)
            for s in samples:
                # Normalise each sample to unit RMS power
                rms = np.sqrt(np.mean(s ** 2) + 1e-8)
                xs.append((s / rms).astype(np.float32))
                ys.append(label)
                snr_list.append(snr)

    X   = np.stack(xs)                          # (N, 2, 128)
    y   = np.asarray(ys,       dtype=np.int64)
    snr = np.asarray(snr_list, dtype=np.int32)

    # Shuffle so classes are not grouped
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    print(f"  Total samples: {len(y)}")
    return X[idx], y[idx], snr[idx], mods


def psk_symbols(order: int, count: int, rng: np.random.Generator) -> np.ndarray:
    phases = 2 * np.pi * rng.integers(0, order, size=count) / order
    return np.exp(1j * phases)


def qam_symbols(order: int, count: int, rng: np.random.Generator) -> np.ndarray:
    side = int(np.sqrt(order))
    levels = np.arange(-(side - 1), side, 2)
    symbols = rng.choice(levels, size=count) + 1j * rng.choice(levels, size=count)
    return symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))


def modulation_symbols(name: str, count: int, rng: np.random.Generator) -> np.ndarray:
    if name == "BPSK":   return psk_symbols(2,  count, rng)
    if name == "QPSK":   return psk_symbols(4,  count, rng)
    if name == "8PSK":   return psk_symbols(8,  count, rng)
    if name == "16QAM":  return qam_symbols(16, count, rng)
    if name == "64QAM":  return qam_symbols(64, count, rng)
    raise ValueError(name)


def add_awgn(signal: np.ndarray, snr_db: int, rng: np.random.Generator) -> np.ndarray:
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape)
    )
    return signal + noise


def make_synthetic_dataset(
    samples_per_class_snr: int,
    sequence_length: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Generate synthetic I/Q signals (no real dataset needed).
    Kept intact from the original code — used when --dataset is not provided.
    """
    mods = SYNTHETIC_MODULATIONS
    rng  = np.random.default_rng(seed)
    xs, ys, snrs = [], [], []
    for label, modulation in enumerate(mods):
        for snr in SNRS:
            for _ in range(samples_per_class_snr):
                signal = modulation_symbols(modulation, sequence_length, rng)
                phase_offset     = rng.uniform(-np.pi, np.pi)
                frequency_offset = rng.uniform(-0.015, 0.015)
                t      = np.arange(sequence_length)
                signal = signal * np.exp(1j * (phase_offset + 2 * np.pi * frequency_offset * t))
                signal = add_awgn(signal, snr, rng)
                iq     = np.stack([signal.real, signal.imag]).astype(np.float32)
                iq    /= np.sqrt(np.mean(iq ** 2) + 1e-8)
                xs.append(iq)
                ys.append(label)
                snrs.append(snr)
    indices = rng.permutation(len(ys))
    return (
        np.stack(xs)[indices],
        np.asarray(ys,   dtype=np.int64)[indices],
        np.asarray(snrs, dtype=np.int32)[indices],
        mods,
    )


# ─────────────────────────────────────────────
#  SECTION 2 — FEATURE EXTRACTION
#  Used only by Stages 1–3 (classical ML).
#  Stage 4 (LSTM) skips this entirely.
# ─────────────────────────────────────────────

def extract_weak_features(x: np.ndarray) -> np.ndarray:
    i, q = x[:, 0, :], x[:, 1, :]
    return np.column_stack([i.mean(1), q.mean(1), i.std(1), q.std(1)])


def extract_strong_features(x: np.ndarray) -> np.ndarray:
    i, q = x[:, 0, :], x[:, 1, :]
    z      = i + 1j * q
    amp    = np.abs(z)
    phase  = np.unwrap(np.angle(z), axis=1)
    dphase = np.diff(phase, axis=1)
    fft_mag = np.abs(np.fft.fft(z, axis=1))
    top_fft = np.sort(fft_mag, axis=1)[:, -8:]
    return np.column_stack([
        i.mean(1), q.mean(1), i.std(1), q.std(1),
        amp.mean(1), amp.std(1), amp.max(1), amp.min(1),
        phase.mean(1), phase.std(1),
        dphase.mean(1), dphase.std(1),
        np.mean(i * q, axis=1),
        np.mean(i ** 2 - q ** 2, axis=1),
        *[top_fft[:, idx] for idx in range(top_fft.shape[1])],
    ])


# ─────────────────────────────────────────────
#  SECTION 3 — CLASSICAL ML MODELS (Stages 1–3)
#  Unchanged from original code.
# ─────────────────────────────────────────────

class NearestCentroid:
    def fit(self, x: np.ndarray, y: np.ndarray) -> "NearestCentroid":
        self.mean_ = x.mean(0)
        self.std_  = x.std(0) + 1e-8
        xn = (x - self.mean_) / self.std_
        self.classes_   = np.unique(y)
        self.centroids_ = np.stack([xn[y == lbl].mean(0) for lbl in self.classes_])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xn = (x - self.mean_) / self.std_
        dists = ((xn[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(2)
        return self.classes_[np.argmin(dists, axis=1)]


class SoftmaxRegression:
    def __init__(self, lr: float = 0.08, epochs: int = 250, l2: float = 1e-4) -> None:
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SoftmaxRegression":
        self.mean_ = x.mean(0)
        self.std_  = x.std(0) + 1e-8
        xn = (x - self.mean_) / self.std_
        n, d       = xn.shape
        n_classes  = int(y.max()) + 1
        self.W_    = np.zeros((d, n_classes))
        self.b_    = np.zeros(n_classes)
        y_oh       = np.eye(n_classes)[y]
        for _ in range(self.epochs):
            logits  = xn @ self.W_ + self.b_
            logits -= logits.max(1, keepdims=True)
            probs   = np.exp(logits)
            probs  /= probs.sum(1, keepdims=True)
            err     = (probs - y_oh) / n
            self.W_ -= self.lr * (xn.T @ err + self.l2 * self.W_)
            self.b_ -= self.lr * err.sum(0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xn = (x - self.mean_) / self.std_
        return np.argmax(xn @ self.W_ + self.b_, axis=1)


# ─────────────────────────────────────────────
#  SECTION 4 — LSTM MODEL (Stage 4)
#  PyTorch implementation — uses GPU if
#  available, falls back to CPU silently.
#
#  Architecture:
#    Input  : (batch, 128, 2)   ← raw I/Q sequence
#    LSTM   : hidden_size=128, 2 stacked layers, dropout=0.5
#    Output : hidden state at last time step
#    FC     : (128 → n_classes) ← classification head
#
#  Training: Adam, lr=1e-3, cross-entropy loss.
#  Why PyTorch vs NumPy:
#    - Batched BLAS/cuBLAS kernels → 10–50× faster on CPU,
#      100–200× faster on CUDA GPU.
#    - Autograd handles backprop — no manual BPTT needed.
# ─────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _get_device() -> "torch.device":
    """Pick CUDA > MPS (Apple Silicon) > CPU, in that order."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Using device: {device}")
    return device


class _LSTMNet(nn.Module):
    """
    Two-layer stacked LSTM followed by a linear classifier.

    Input shape  : (batch, seq_len=128, features=2)
    Output shape : (batch, n_classes)
    """
    def __init__(self, n_classes: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = 2,
            hidden_size = hidden_size,
            num_layers  = 2,
            batch_first = True,
            dropout     = 0.5,      # dropout between the two LSTM layers
        )
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x : (B, 128, 2)
        out, _ = self.lstm(x)       # out : (B, 128, hidden_size)
        last   = out[:, -1, :]      # take the final time step : (B, hidden_size)
        return self.fc(last)        # (B, n_classes)


# ─────────────────────────────────────────────
#  SECTION 5 — ResNet + LSTM MODEL (Stage 5)
#
#  Motivation
#  ----------
#  Stage 4's plain LSTM reads raw I/Q samples one
#  time-step at a time.  It can capture temporal
#  patterns but struggles to learn *local* structure
#  (e.g. symbol-level amplitude/phase patterns) before
#  the sequence gets long.
#
#  Adding a stack of 1-D residual convolutional blocks
#  in front of the LSTM lets the network first compress
#  local features into richer per-step representations,
#  then let the LSTM model how those features evolve
#  across the full frame.
#
#  Full data flow
#  --------------
#  Input  (N, 2, 128)   — raw I/Q, channels-first for Conv1d
#    ↓  Stem conv       — projects 2 raw channels → 64 feature maps
#    ↓  ResBlock × 3    — refines local features; skip connections
#                          prevent gradient vanishing in the conv stack
#    ↓  permute(0,2,1)  — reshape to (N, 128, 64) for LSTM's
#                          (batch, seq_len, features) convention
#    ↓  Stacked LSTM    — 2 layers, hidden=128; models temporal
#                          dependencies across the 128-step sequence
#    ↓  last step only  — (N, 128); discards intermediate hidden states
#    ↓  Linear FC       — (128 → n_classes); final classification head
#  Output (N, n_classes) — raw logits fed to CrossEntropyLoss
# ─────────────────────────────────────────────


class ResidualBlock1D(nn.Module):
    """
    One residual block operating on 1-D feature sequences.

    Structure (classic pre-activation layout without pre-activation here):
        x ──► Conv1d ──► BN ──► ReLU ──► Conv1d ──► BN ──► (+) ──► ReLU
        └─────────────────────────────────────────────────────┘
                              identity skip connection

    Why residual / skip connections?
    - Allow gradients to flow directly back to early layers,
      sidestepping the vanishing-gradient problem that hurts
      deep networks trained with SGD/Adam.
    - The block learns a *residual* correction on top of its
      own input rather than a full transformation from scratch,
      which is easier to optimise.

    Input/output shape: (N, channels, seq_len) — unchanged.
    The number of channels stays constant so the identity skip
    requires no projection (no 1×1 conv needed).
    """

    def __init__(self, channels):
        super().__init__()

        
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        # Save input for the skip connection before any transformation.
        identity = x

        # First sub-layer: conv → normalise → activate
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = torch.relu(out)

        return out


class _ResNetLSTMNet(nn.Module):
    """
    Hybrid convolutional + recurrent classifier for I/Q signals.

    Design rationale
    ----------------
    - Conv layers (stem + residual blocks) act as a *local feature extractor*:
      they see a small window of time steps at once and learn patterns like
      symbol transitions, amplitude modulation depth, and phase jumps.
    - The LSTM acts as a *sequence model*: after the convs compress raw I/Q
      into richer 64-dim per-step vectors, the LSTM integrates them across
      the whole 128-step frame to capture global temporal structure.
    - This two-stage design outperforms a plain LSTM (Stage 4) because the
      LSTM no longer has to learn local structure from raw 2-dim I/Q; it
      receives pre-processed 64-dim features instead.

    Input shape  : (N, 2, 128)    — channels-first, as expected by Conv1d
    Output shape : (N, n_classes) — unnormalised logits
    """

    def __init__(
        self,
        n_classes,
        hidden_size=128,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                2,          # in_channels  : I and Q
                64,         # out_channels : learned feature maps
                kernel_size=3,
                padding=1,  # same-padding preserves seq_len=128
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.res1 = ResidualBlock1D(64)
        self.res2 = ResidualBlock1D(64)
        self.res3 = ResidualBlock1D(64)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
        )
        self.fc = nn.Linear(
            hidden_size,
            n_classes,
        )

    def forward(self, x):
      
        x = self.stem(x)
        # Three residual blocks refine the feature maps without changing
        # shape.  Each block's skip connection ensures gradients backprop
        # cleanly all the way to the stem.
        x = self.res1(x)    # (N, 64, 128)
        x = self.res2(x)    # (N, 64, 128)
        x = self.res3(x)    # (N, 64, 128)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]    # (N, hidden_size)
        return self.fc(last)    # (N, n_classes)


class ResNetLSTMClassifier:
    """
    Sklearn-style wrapper around _ResNetLSTMNet.

    Exposes .fit(x, y) and .predict(x) so it plugs into the same
    evaluate() function used by Stages 1–4.

    Training details
    ----------------
    - Loss    : CrossEntropyLoss (softmax + NLL in one numerically
                stable op)
    - Optimiser: Adam with lr=1e-3 (adaptive per-parameter learning
                rates; well suited to the mixed conv+LSTM parameter
                landscape)
    - Epochs  : 30 by default; early stopping not used here — keep it
                simple for benchmarking
    - Batch size: 256 — large enough for stable BN statistics while
                fitting comfortably in GPU memory
    """

    def __init__(
        self,
        hidden_size=128,
        epochs=30,
        batch_size=256,
        lr=1e-3,
        seed=42,
    ):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

    def fit(self, x, y):
        torch.manual_seed(self.seed)

        self.device_ = _get_device()
        x_t = torch.tensor(
            x.astype(np.float32)
        )

        y_t = torch.tensor(
            y,
            dtype=torch.long
        )

        loader = DataLoader(
            TensorDataset(x_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,       # shuffle each epoch to reduce correlation
            num_workers=0,      # 0 = load in the main process (safest on all OS)
        )

        n_classes = int(y.max()) + 1

        self.model_ = _ResNetLSTMNet(
            n_classes=n_classes,
            hidden_size=self.hidden_size,
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
        )

        self.model_.train()

        for epoch in range(self.epochs):

            total_loss = 0.0
            correct = 0
            total = 0

            for xb, yb in loader:

                xb = xb.to(self.device_)
                yb = yb.to(self.device_)

                optimizer.zero_grad()

                # Forward pass: raw I/Q → conv stack → LSTM → logits
                logits = self.model_(xb)

                loss = criterion(
                    logits,
                    yb,
                )

                # Backward pass: autograd computes gradients through
                # both the LSTM (via BPTT) and the conv blocks.
                loss.backward()

                optimizer.step()

                total_loss += loss.item() * len(yb)

                correct += (
                    logits.argmax(1) == yb
                ).sum().item()

                total += len(yb)

            print(
                f"  ResNet+LSTM epoch {epoch+1:3d}/{self.epochs}"
                f"  loss={total_loss/total:.4f}"
                f"  train_acc={correct/total:.4f}",
                flush=True,
            )

        return self

    def predict(self, x):

        self.model_.eval()  # disables dropout and switches BN to eval mode

        x_t = torch.tensor(
            x.astype(np.float32)
        )

        # Use a larger batch size at inference — no gradient storage needed,
        # so memory is less constrained than during training.
        loader = DataLoader(
            TensorDataset(x_t),
            batch_size=self.batch_size * 2,
            shuffle=False,
        )

        preds = []

        with torch.no_grad():   # disables autograd for faster, lighter inference

            for (xb,) in loader:

                xb = xb.to(self.device_)

                pred = self.model_(xb)

                preds.append(
                    pred.argmax(1)
                    .cpu()
                    .numpy()
                )

        return np.concatenate(preds)


class PyTorchLSTMClassifier:
    """
    Sklearn-style wrapper around _LSTMNet.
    Exposes .fit(x, y) and .predict(x) so it plugs into
    the same evaluate() function used by Stages 1–3.
    """
    def __init__(
        self,
        hidden_size: int = 128,
        epochs:      int = 30,
        batch_size:  int = 256,
        lr:         float = 1e-3,
        seed:        int = 42,
    ) -> None:
        self.hidden_size = hidden_size
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.seed        = seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PyTorchLSTMClassifier":
        """
        x : (N, 2, 128) — raw I/Q, already normalised
        y : (N,)        — integer class labels
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Run:\n"
                "  pip install torch\n"
                "then re-run the script."
            )

        torch.manual_seed(self.seed)
        self.device_ = _get_device()

        # Reshape: (N, 2, 128) → (N, 128, 2)  — LSTM wants (batch, seq, features)
        x_seq = torch.tensor(
            x.transpose(0, 2, 1).astype(np.float32)
        )
        y_t = torch.tensor(y, dtype=torch.long)

        dataset    = TensorDataset(x_seq, y_t)
        loader     = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=0, pin_memory=(self.device_.type == "cuda"),
        )

        n_classes  = int(y.max()) + 1
        self.model_ = _LSTMNet(n_classes, self.hidden_size).to(self.device_)
        criterion   = nn.CrossEntropyLoss()
        optimizer   = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        self.model_.train()
        for epoch in range(self.epochs):
            total_loss, correct, total = 0.0, 0, 0

            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)

                optimizer.zero_grad()
                logits = self.model_(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(yb)
                correct    += (logits.argmax(1) == yb).sum().item()
                total      += len(yb)

            avg_loss = total_loss / total
            acc      = correct / total
            print(
                f"  LSTM epoch {epoch + 1:3d}/{self.epochs}"
                f"  loss={avg_loss:.4f}  train_acc={acc:.4f}",
                flush=True,
            )

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model_.eval()
        x_seq  = torch.tensor(x.transpose(0, 2, 1).astype(np.float32))
        loader = DataLoader(
            TensorDataset(x_seq),
            batch_size=self.batch_size * 2,   # larger batch is fine at inference
            shuffle=False, num_workers=0,
        )
        preds = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device_)
                preds.append(self.model_(xb).argmax(1).cpu().numpy())
        return np.concatenate(preds)


# Alias so the rest of main() doesn't need to change
NumpyLSTMClassifier = PyTorchLSTMClassifier


# ─────────────────────────────────────────────
#  SECTION 5 — DATA SPLITTING
# ─────────────────────────────────────────────

def stratified_split(
    y: np.ndarray, snr: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split indices 70 / 15 / 15 stratified by (class, SNR).
    Unchanged from original code.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for label in np.unique(y):
        for snr_value in np.unique(snr):
            group = np.where((y == label) & (snr == snr_value))[0]
            rng.shuffle(group)
            n_train = int(0.70 * len(group))
            n_val   = int(0.15 * len(group))
            train_idx.extend(group[:n_train])
            val_idx.extend(  group[n_train: n_train + n_val])
            test_idx.extend( group[n_train + n_val:])
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


# ─────────────────────────────────────────────
#  SECTION 6 — EVALUATION HELPERS
# ─────────────────────────────────────────────

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> list[list[int]]:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[int(t), int(p)] += 1
    return matrix.tolist()


def evaluate(
    name: str, model,
    features: np.ndarray,    # x or hand-crafted features, same indexing as y/snr
    y: np.ndarray,
    snr: np.ndarray,
    test_idx: np.ndarray,
    num_classes: int,
) -> dict:
    pred = model.predict(features[test_idx])
    result = {
        "model":            name,
        "overall_accuracy": accuracy(y[test_idx], pred),
        "accuracy_by_snr":  {},
        "confusion_matrix": confusion_matrix(y[test_idx], pred, num_classes),
    }
    for snr_val in sorted(np.unique(snr)):
        mask = test_idx[snr[test_idx] == snr_val]
        if len(mask):
            result["accuracy_by_snr"][str(int(snr_val))] = accuracy(
                y[mask], model.predict(features[mask])
            )
    return result


# ─────────────────────────────────────────────
#  SECTION 7 — MAIN ENTRY POINT
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AMC MVP — Stages 1-4")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to RML2016.10a_dict.pkl. "
             "If omitted, the synthetic generator is used instead.",
    )
    parser.add_argument("--samples-per-class-snr", type=int, default=80,
                        help="Synthetic mode only: samples per (class, SNR) pair.")
    parser.add_argument("--sequence-length",       type=int, default=128,
                        help="Synthetic mode only: I/Q frame length.")
    parser.add_argument("--seed",                  type=int, default=7)
    parser.add_argument("--lstm-epochs",           type=int, default=30)
    parser.add_argument("--lstm-hidden",           type=int, default=64)
    parser.add_argument("--skip-lstm",             action="store_true",
                        help="Run Stages 1-3 only (faster for quick checks).")
    parser.add_argument(
    "--skip-stage5",
    action="store_true"
)
    args = parser.parse_args()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    if args.dataset:
        x, y, snr, mods = load_radioml(args.dataset)
    else:
        print("No --dataset provided. Using synthetic generator.")
        x, y, snr, mods = make_synthetic_dataset(
            args.samples_per_class_snr, args.sequence_length, args.seed
        )

    print(f"\nDataset ready — {len(y)} samples, {len(mods)} classes, "
          f"x shape: {x.shape}\n")

    # ── 2. Split ──────────────────────────────────────────────────────────────
    train_idx, val_idx, test_idx = stratified_split(y, snr, args.seed)

    # ── 3. Feature extraction (Stages 1–3 only) ───────────────────────────────
    weak   = extract_weak_features(x)
    strong = extract_strong_features(x)

    # ── 4. Train classical models ─────────────────────────────────────────────
    print("Training Stage 1 — NearestCentroid (weak features) ...")
    s1 = NearestCentroid().fit(weak[train_idx], y[train_idx])

    print("Training Stage 2 — NearestCentroid (strong features) ...")
    s2 = NearestCentroid().fit(strong[train_idx], y[train_idx])

    print("Training Stage 3 — SoftmaxRegression (strong features) ...")
    s3 = SoftmaxRegression().fit(strong[train_idx], y[train_idx])

    n_classes = len(mods)

    results = [
        evaluate("stage_1_weak_baseline",      s1, weak,   y, snr, test_idx, n_classes),
        evaluate("stage_2_better_features",     s2, strong, y, snr, test_idx, n_classes),
        evaluate("stage_3_trainable_softmax",   s3, strong, y, snr, test_idx, n_classes),
    ]

    # ── 5. Train LSTM (Stage 4) ───────────────────────────────────────────────
    if not args.skip_lstm:
        print(f"\nTraining Stage 4 — LSTM "
              f"(hidden={args.lstm_hidden}, epochs={args.lstm_epochs}) ...")
        s4 = NumpyLSTMClassifier(
            hidden_size = args.lstm_hidden,
            epochs      = args.lstm_epochs,
            seed        = args.seed,
        ).fit(x[train_idx], y[train_idx])

        results.append(
            evaluate("stage_4_lstm", s4, x, y, snr, test_idx, n_classes)
        )

    if not args.skip_stage5:

        print(
        f"\nTraining Stage 5 — ResNet + LSTM "
        f"(hidden={args.lstm_hidden}, epochs={args.lstm_epochs}) ..."
        )

        s5 = ResNetLSTMClassifier(
        hidden_size=args.lstm_hidden,
        epochs=args.lstm_epochs,
        seed=args.seed,
        ).fit(
        x[train_idx],
        y[train_idx],
        )

        results.append(
        evaluate(
            "stage_5_resnet_lstm",
            s5,
            x,
            y,
            snr,
            test_idx,
            n_classes,
        )
        )

    # ── 6. Save & print results ───────────────────────────────────────────────

    reports_dir = Path("reports") / "numpy_mvp"
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_path = reports_dir / "metrics.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
        {
            "labels": mods,
            "results": results,
        },
        f,
        indent=2,
        )

# ─────────────────────────────────────────────
#  Print overall test accuracy
#
#  Each stage is evaluated on exactly the
#  same test set for fair comparison.
# ─────────────────────────────────────────────

    print("\n" + "=" * 50)
    print("AMC MVP results")
    print("=" * 50)

    for row in results:

        print(
        f"  {row['model']:32s}"
        f"  overall accuracy = "
        f"{row['overall_accuracy']:.4f}"
        )

# ─────────────────────────────────────────────
#  High-SNR analysis
#
#  Radio signals become much easier to
#  classify when noise is low.
#
#  We therefore report average accuracy
#  for SNR >= 0 dB.
#
#  This is often the metric reported in
#  AMC literature.
# ─────────────────────────────────────────────

    print("\n" + "=" * 50)
    print("High-SNR (>= 0 dB) Performance")
    print("=" * 50)

    for row in results:

        if "accuracy_by_snr" not in row:
            continue

        high_snr_acc = {
        int(k): v
        for k, v in row["accuracy_by_snr"].items()
        if int(k) >= 0
        }

        if len(high_snr_acc) == 0:
            continue

        avg_high = np.mean(
        list(high_snr_acc.values())
        )

        print(
        f"  {row['model']:32s}"
        f"  avg_high_snr = "
        f"{avg_high:.4f}"
        )

# ─────────────────────────────────────────────
# Accuracy specifically at 18 dB
# ─────────────────────────────────────────────

    print("\n" + "=" * 50)
    print("18 dB Accuracy")
    print("=" * 50)

    for row in results:

        if "accuracy_by_snr" not in row:
            continue

        acc_18 = row["accuracy_by_snr"].get("18")

        if acc_18 is None:
            continue

        print(
        f"  {row['model']:32s}"
        f"  acc@18dB = "
        f"{acc_18:.4f}"
        )

    print(f"\nFull metrics saved to {output_path}")

if __name__ == "__main__":
    main()