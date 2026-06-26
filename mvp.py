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
    Two-layer stacked LSTM with mean-pooling + normalised classification head.

    Architecture
    ------------
    Input  : (B, 128, 2)              — raw I/Q sequence, batch-first
    LSTM   : 2 layers, hidden_size, inter-layer dropout=0.3
    Pool   : mean over all 128 time steps → (B, hidden_size)
               Why mean-pool instead of last-step?
               The final hidden state only carries what the LSTM chose to
               remember at t=127. Mean-pooling aggregates evidence from every
               time step, giving the classifier access to the full temporal
               context — critical for modulations whose discriminative features
               (e.g. phase jitter, envelope shape) are spread across the frame.
    Norm   : LayerNorm stabilises the pooled vector before the FC layer,
               preventing large-magnitude hidden states from dominating.
    Head   : Dropout(0.4) → Linear(hidden → n_classes)
               Dropout here regularises the classification layer independently
               of the recurrent dropout, which only acts between LSTM layers.

    Input shape  : (B, 128, 2)
    Output shape : (B, n_classes)
    """
    def __init__(self, n_classes: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = 2,
            hidden_size = hidden_size,
            num_layers  = 2,
            batch_first = True,
            dropout     = 0.3,   # reduced: 0.5 → 0.3 (between LSTM layers only)
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.4)
        self.fc      = nn.Linear(hidden_size, n_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x : (B, 128, 2)
        out, _  = self.lstm(x)         # out : (B, 128, hidden_size)
        pooled  = out.mean(dim=1)      # mean over all time steps → (B, hidden_size)
        pooled  = self.norm(pooled)    # LayerNorm for stable FC input
        return self.fc(self.dropout(pooled))   # (B, n_classes)


class PyTorchLSTMClassifier:
    """
    Sklearn-style wrapper around _LSTMNet.
    Exposes .fit(x, y) and .predict(x) so it plugs into
    the same evaluate() function used by Stages 1–3.

    Key training choices
    --------------------
    - hidden_size=128 : matches the paper [15] capacity.
    - CosineAnnealingLR : decays lr from 1e-3 to 1e-5 so the model
        fine-tunes gently in later epochs instead of oscillating.
    - grad_clip=1.0 : clips the global gradient norm before every
        optimiser step.  LSTMs are notoriously prone to gradient
        explosions; clipping prevents them without needing a lower lr.
    """
    def __init__(
        self,
        hidden_size: int   = 128,   # was 64 in CLI default — corrected to match paper
        epochs:      int   = 30,
        batch_size:  int   = 256,
        lr:          float = 1e-3,
        grad_clip:   float = 1.0,   # max gradient norm; 0 = disabled
        seed:        int   = 42,
    ) -> None:
        self.hidden_size = hidden_size
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.grad_clip   = grad_clip
        self.seed        = seed

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        checkpoint_path: Optional[str] = None,
    ) -> "PyTorchLSTMClassifier":
        """
        x / y         : training set  (N, 2, 128) / (N,)
        x_val / y_val : optional validation set — when provided, a checkpoint
                        is saved every time val accuracy improves.
        checkpoint_path : where to write the .pt file; defaults to
                          checkpoints/stage4_lstm_best.pt
                          Saved dict contains model weights, optimiser state,
                          best epoch, val_acc, n_classes, hidden_size — enough
                          to fully restore or resume training later.
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
        x_seq = torch.tensor(x.transpose(0, 2, 1).astype(np.float32))
        y_t   = torch.tensor(y, dtype=torch.long)

        loader = DataLoader(
            TensorDataset(x_seq, y_t),
            batch_size=self.batch_size,
            shuffle=True, num_workers=0,
            pin_memory=(self.device_.type == "cuda"),
        )

        # Optional validation tensors (kept on CPU; moved per-batch)
        has_val = x_val is not None and y_val is not None
        if has_val:
            x_val_seq  = torch.tensor(x_val.transpose(0, 2, 1).astype(np.float32))
            y_val_t    = torch.tensor(y_val, dtype=torch.long)
            val_loader = DataLoader(
                TensorDataset(x_val_seq, y_val_t),
                batch_size=self.batch_size * 2, shuffle=False, num_workers=0,
                pin_memory=(self.device_.type == "cuda"),
            )

        n_classes   = int(y.max()) + 1
        self.model_ = _LSTMNet(n_classes, self.hidden_size).to(self.device_)
        criterion   = nn.CrossEntropyLoss()
        optimizer   = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        # CosineAnnealingLR: lr decays smoothly from self.lr → 1e-5 over all
        # epochs.  Avoids the "stuck plateau" problem of a fixed learning rate
        # and removes the need to hand-tune a step-decay schedule.
        scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )

        # ── Checkpoint path ───────────────────────────────────────────────────
        # We save whenever val accuracy strictly improves so we always keep
        # the best-generalising weights, not just the final epoch's weights.
        # At end of training the best checkpoint is reloaded automatically.
        best_val_acc = -1.0
        best_epoch   = 0
        ckpt_path    = Path(checkpoint_path) if checkpoint_path else \
                       Path("checkpoints") / "stage4_lstm_best.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

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
                # Gradient clipping: prevents LSTM exploding gradients.
                # Clips the global L2 norm of all parameters to self.grad_clip.
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip)
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct    += (logits.argmax(1) == yb).sum().item()
                total      += len(yb)

            scheduler.step()

            log_msg = (
                f"  LSTM epoch {epoch + 1:3d}/{self.epochs}"
                f"  loss={total_loss / total:.4f}"
                f"  train_acc={correct / total:.4f}"
                f"  lr={scheduler.get_last_lr()[0]:.2e}"
            )

            # ── Validation pass & checkpointing ───────────────────────────────
            if has_val:
                self.model_.eval()
                val_correct, val_total = 0, 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device_)
                        yb = yb.to(self.device_)
                        val_correct += (self.model_(xb).argmax(1) == yb).sum().item()
                        val_total   += len(yb)
                val_acc  = val_correct / val_total
                log_msg += f"  val_acc={val_acc:.4f}"

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch   = epoch + 1
                    torch.save(
                        {
                            "epoch":       best_epoch,
                            "model_state": self.model_.state_dict(),
                            "optim_state": optimizer.state_dict(),
                            "val_acc":     best_val_acc,
                            "n_classes":   n_classes,
                            "hidden_size": self.hidden_size,
                        },
                        ckpt_path,
                    )
                    log_msg += "  ✓ saved"
                self.model_.train()

            print(log_msg, flush=True)

        # ── Reload best weights ───────────────────────────────────────────────
        if has_val and ckpt_path.exists():
            saved = torch.load(ckpt_path, map_location=self.device_, weights_only=True)
            self.model_.load_state_dict(saved["model_state"])
            print(
                f"  LSTM: restored best checkpoint  epoch={best_epoch}"
                f"  val_acc={best_val_acc:.4f}  ← {ckpt_path}"
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
#  SECTION 5 — INCEPTION + RESNET MODEL (Stage 5)
#
#  Reference: [22] — "Modulation classification based on
#  Inception network and ResNet"
#  Dataset  : RadioML2016.10b
#  Target   : ≥ 93.76% accuracy at 14 dB SNR
#
#  Motivation
#  ----------
#  Stage 4's plain LSTM reads the raw I/Q sequence one step
#  at a time and misses local multi-scale structure — e.g.
#  a narrow symbol pattern (kernel=3) and a wider inter-symbol
#  pattern (kernel=7) are both present but a single Conv size
#  can only see one at a time.
#
#  The Inception module solves this with PARALLEL branches of
#  different kernel sizes, then concatenates their outputs so
#  the next layer sees ALL scales at once.
#  Residual skip connections then let the network deepen
#  without vanishing gradients.
#
#  Architecture note
#  -----------------
#  The reference label "Inception Network + ResNet" typically describes an
#  Inception-ResNet style where residual connections are integrated *inside*
#  each Inception module.  Our implementation is a sequential arrangement of
#  separate Inception and ResNet blocks — a simpler but effective variant.
#  The blocks are now ALTERNATED (Inc → Res → Inc → Res) so that each
#  residual block immediately refines the multi-scale features produced by
#  its preceding Inception block, rather than stacking all Inception blocks
#  first and all residual blocks second.
#
#  Full data flow
#  --------------
#  Input  (N, 2, 128)        — raw I/Q, channels-first for Conv1d
#    ↓  Stem (2 convs)       — Conv→BN→ReLU→Conv→BN→ReLU; 2→64 channels
#    ↓  InceptionBlock 1     — 4 parallel branches (k=1,3,5,7) → 256 ch
#    ↓  ResidualBlock 1      — refine with skip connection (256 ch)
#    ↓  InceptionBlock 2     — 4 parallel branches (k=1,3,5,7) → 256 ch
#    ↓  ResidualBlock 2      — refine with skip connection (256 ch)
#    ↓  AdaptiveAvgPool1d(1) — collapse time dimension → (N, 256)
#    ↓  Dropout(0.3)         — regularisation
#    ↓  Linear FC            — 256 → n_classes logits
#  Output (N, n_classes)
# ─────────────────────────────────────────────


class _InceptionBlock1D(nn.Module):
    """
    Inception block for 1-D I/Q signal sequences.

    Four parallel branches with different receptive fields run
    simultaneously on the same input, then their outputs are
    concatenated along the channel dimension.

    Why four branches?
    - kernel=1 : acts like a channel mixer / pointwise feature
                 recombination — zero temporal context but cheap.
    - kernel=3 : captures short-range symbol-level patterns
                 (e.g. phase transitions within one symbol).
    - kernel=5 : intermediate scale — catches patterns that span
                 a few symbols (e.g. amplitude/phase envelopes).
    - kernel=7 : captures wider inter-symbol context
                 (e.g. amplitude envelope, frequency drift).
    The extra intermediate scale (kernel=5) gives the model an
    additional receptive field and usually improves feature diversity.
    Concatenating all four lets downstream layers see multi-scale
    features simultaneously, which is the core Inception insight.

    Input shape  : (N, in_channels, seq_len)
    Output shape : (N, out_channels*4, seq_len)  — same seq_len via padding
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        # Branch 1 — pointwise (kernel size 1, no padding needed)
        self.branch_1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

        # Branch 2 — short-range (kernel size 3, padding=1 → same length)
        self.branch_3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

        # Branch 3 — intermediate-range (kernel size 5, padding=2 → same length)
        self.branch_5 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

        # Branch 4 — wide-range (kernel size 7, padding=3 → same length)
        self.branch_7 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        b1 = self.branch_1(x)   # (N, out_channels, seq_len)
        b3 = self.branch_3(x)   # (N, out_channels, seq_len)
        b5 = self.branch_5(x)   # (N, out_channels, seq_len)
        b7 = self.branch_7(x)   # (N, out_channels, seq_len)
        # Concatenate along channel axis → (N, out_channels*4, seq_len)
        return torch.cat([b1, b3, b5, b7], dim=1)


class _ResBlock1D(nn.Module):
    """
    1-D residual block — same design as Stage 5's ResidualBlock1D.
    Kept as a private class here so Stage 5 and Stage 6 are fully
    self-contained and the channel width can differ.

    Structure:
        x ──► Conv1d ──► BN ──► ReLU ──► Conv1d ──► BN ──► (+x) ──► ReLU

    Input/output shape: (N, channels, seq_len) — unchanged.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)


class _InceptionResNet(nn.Module):
    """
    Hybrid Inception + ResNet classifier for raw I/Q signals.

    Architecture: Stem → Inc → Res → Inc → Res → AvgPool → Dropout → FC

    The blocks are ALTERNATED rather than grouped (all-Inc then all-Res).
    This means each ResBlock immediately refines the multi-scale features
    from the Inception block just before it, giving the network tighter
    feedback between feature extraction and feature refinement.

    Note: this is a sequential Inception+ResNet arrangement, not a true
    Inception-ResNet where skip connections are wired inside each Inception
    module.  The alternating layout is a practical middle ground — simpler
    to implement and still significantly better than pure Inception or pure
    ResNet alone.
    """

    def __init__(self, n_classes: int) -> None:
        super().__init__()

        # Stem: two conv layers for richer low-level feature extraction
        # Conv→BN→ReLU→Conv→BN→ReLU  (both 2→64 then 64→64, same-padding)
        self.stem = nn.Sequential(
            nn.Conv1d(2,  64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Two Inception blocks (4 branches each, out_channels=64 per branch).
        # Each outputs out_channels*4 = 256 channels.
        # Alternated with residual blocks: inc1 → res1 → inc2 → res2
        self.inc1 = _InceptionBlock1D(in_channels=64,  out_channels=64)
        self.res1 = _ResBlock1D(256)   # immediately refines inc1's features
        self.inc2 = _InceptionBlock1D(in_channels=256, out_channels=64)
        self.res2 = _ResBlock1D(256)   # immediately refines inc2's features

        # Global average pooling collapses (N, 256, 128) → (N, 256)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Dropout reduced from 0.5 → 0.3 (less aggressive for RadioML2016)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(256, n_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x : (N, 2, 128)
        x = self.stem(x)    # (N, 64,  128)
        x = self.inc1(x)    # (N, 256, 128)  — multi-scale feature extraction
        x = self.res1(x)    # (N, 256, 128)  — refine inc1's features
        x = self.inc2(x)    # (N, 256, 128)  — second pass multi-scale
        x = self.res2(x)    # (N, 256, 128)  — refine inc2's features
        x = self.pool(x)    # (N, 256, 1)
        x = x.squeeze(-1)   # (N, 256)
        x = self.dropout(x)
        return self.fc(x)   # (N, n_classes)


def _augment_iq(xb: "torch.Tensor") -> "torch.Tensor":
    """
    Minimal, numerically safe augmentation: random phase rotation only.

    Why only phase rotation?
    ------------------------
    Phase rotation (multiply complex signal by e^{jθ}) is the one
    augmentation that is:
      (a) guaranteed not to change the modulation class — all standard
          modulations (PSK, QAM, FSK) are defined up to a global phase
          offset, which the receiver's phase-lock loop corrects anyway.
      (b) lossless — it is a unitary transform; signal energy and
          structure are perfectly preserved.
      (c) trivially correct in floating point — no FFT, no .real
          truncation, no circular shift boundary effects.

    The earlier implementation applied 4 transforms simultaneously
    (FFT-based shift, amplitude scale, frequency ramp, phase rotation),
    which caused two problems:
      1. The FFT shift used `ifft(...).real`, zeroing the imaginary part
         and producing physically wrong (non-complex-baseband) signals.
      2. With p=0.5 each, ~94% of samples received at least one transform
         every batch — the model trained almost entirely on distorted
         signals and never converged (train_acc stalled at ~44%).

    This version applies only phase rotation with p=0.5, which is
    sufficient to prevent the model memorising absolute phase and
    improves generalisation at no cost to signal fidelity.

    Parameters
    ----------
    xb : (B, 2, 128) float32 tensor — I/Q batch, channels-first

    Returns
    -------
    (B, 2, 128) float32 tensor — same shape and dtype
    """
    B, _, L = xb.shape
    device  = xb.device

    # Sample per-element: apply rotation with probability 0.5
    do_rotate = torch.rand(B, device=device) < 0.5          # (B,)
    theta     = torch.rand(B, device=device) * 2 * torch.pi # U(0, 2π)

    cos_t = torch.where(do_rotate, torch.cos(theta), torch.ones(B,  device=device))  # (B,)
    sin_t = torch.where(do_rotate, torch.sin(theta), torch.zeros(B, device=device))  # (B,)

    # Rotate: I' = I·cos(θ) − Q·sin(θ),  Q' = I·sin(θ) + Q·cos(θ)
    I = xb[:, 0, :]                                          # (B, L)
    Q = xb[:, 1, :]                                          # (B, L)
    cos_t = cos_t.unsqueeze(1)                               # (B, 1) for broadcasting
    sin_t = sin_t.unsqueeze(1)

    I_rot = I * cos_t - Q * sin_t
    Q_rot = I * sin_t + Q * cos_t

    return torch.stack([I_rot, Q_rot], dim=1)                # (B, 2, L)


class InceptionResNetClassifier:
    """
    Sklearn-style wrapper around _InceptionResNet.

    Exposes .fit(x, y) and .predict(x) so it plugs into the
    same evaluate() function used by all previous stages.

    Training details
    ----------------
    - Loss      : CrossEntropyLoss with label_smoothing=0.1
                  Prevents overconfident predictions; acts as implicit
                  regularisation and consistently improves generalisation
                  on AMC tasks (same benefit as mixup, simpler to apply).
    - Optimiser : Adam lr=1e-3, weight_decay=1e-4
    - Warmup    : Linear LR warmup for the first `warmup_epochs` epochs
                  (default 5).  Convolutional models with many branches
                  are sensitive to the large initial gradients that hit
                  when all branches compete with random weights.  Warmup
                  ramps lr from lr/10 → lr smoothly so early updates
                  are small and the branches co-adapt before the full
                  learning rate kicks in.
    - Scheduler : CosineAnnealingLR after warmup — decays lr to 1e-5
    - Grad clip : 1.0 (same as LSTM Stage 4) — prevents rare exploding
                  gradients when augmented samples have very large norms.
    - Epochs    : 50 (increased from 30 — CNN models need more epochs
                  than LSTMs to converge; the extra 20 cost little on GPU)
    - Batch     : 512 (faster on T4 GPU without hurting performance)
    """

    def __init__(
        self,
        epochs:          int   = 50,
        batch_size:      int   = 512,
        lr:              float = 1e-3,
        grad_clip:       float = 1.0,
        warmup_epochs:   int   = 5,
        label_smoothing: float = 0.1,
        seed:            int   = 42,
    ) -> None:
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.lr              = lr
        self.grad_clip       = grad_clip
        self.warmup_epochs   = warmup_epochs
        self.label_smoothing = label_smoothing
        self.seed            = seed

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        checkpoint_path: Optional[str] = None,
    ) -> "InceptionResNetClassifier":
        """
        x / y         : training set  (N, 2, 128) / (N,)
        x_val / y_val : optional validation set — when provided, a checkpoint
                        is saved every time val accuracy improves.
        checkpoint_path : where to write the .pt file; defaults to
                          checkpoints/stage5_inception_resnet_best.pt
                          Saved dict: model_state, optim_state, scheduler_state,
                          best_epoch, val_acc, n_classes — fully restorable.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Run:\n"
                "  pip install torch\n"
                "then re-run the script."
            )

        torch.manual_seed(self.seed)
        self.device_ = _get_device()

        # Input is already (N, 2, 128) — correct for Conv1d, no transpose needed
        x_t = torch.tensor(x.astype(np.float32))
        y_t = torch.tensor(y, dtype=torch.long)

        loader = DataLoader(
            TensorDataset(x_t, y_t),
            batch_size=self.batch_size,
            shuffle=True, num_workers=0,
            pin_memory=(self.device_.type == "cuda"),
        )

        # Optional validation tensors
        has_val = x_val is not None and y_val is not None
        if has_val:
            x_val_t    = torch.tensor(x_val.astype(np.float32))
            y_val_t    = torch.tensor(y_val, dtype=torch.long)
            val_loader = DataLoader(
                TensorDataset(x_val_t, y_val_t),
                batch_size=self.batch_size * 2, shuffle=False, num_workers=0,
            )

        n_classes   = int(y.max()) + 1
        self.model_ = _InceptionResNet(n_classes).to(self.device_)
        # Label smoothing: replaces hard 0/1 targets with (ε/K, …, 1-ε+ε/K, …, ε/K).
        # Prevents the model assigning probability ~1 to the correct class at high
        # SNR, which would leave no gradient signal for the low-SNR regime.
        criterion   = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        # weight_decay=1e-4 adds L2 regularisation to all parameters.
        optimizer   = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=1e-4
        )

        # Two-phase LR schedule:
        #   Phase 1 (epochs 0..warmup_epochs-1): linear ramp lr/10 → lr
        #   Phase 2 (epochs warmup_epochs..end): cosine decay lr → 1e-5
        # LinearLR: start_factor=0.1 means initial_lr = lr * 0.1
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = 0.1,
            end_factor   = 1.0,
            total_iters  = self.warmup_epochs,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = max(1, self.epochs - self.warmup_epochs),
            eta_min = 1e-5,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers  = [warmup_sched, cosine_sched],
            milestones  = [self.warmup_epochs],
        )

        # ── Checkpoint path ───────────────────────────────────────────────────
        best_val_acc = -1.0
        best_epoch   = 0
        ckpt_path    = Path(checkpoint_path) if checkpoint_path else \
                       Path("checkpoints") / "stage5_inception_resnet_best.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        self.model_.train()
        for epoch in range(self.epochs):
            total_loss, correct, total = 0.0, 0, 0

            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                # Data augmentation: applied only during training
                xb = _augment_iq(xb)
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                # Gradient clipping: prevents rare exploding gradients from
                # augmented batches with large signal norms.
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip)
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct    += (logits.argmax(1) == yb).sum().item()
                total      += len(yb)

            scheduler.step()

            log_msg = (
                f"  Inception+ResNet epoch {epoch + 1:3d}/{self.epochs}"
                f"  loss={total_loss / total:.4f}"
                f"  train_acc={correct / total:.4f}"
                f"  lr={scheduler.get_last_lr()[0]:.6f}"
            )

            # ── Validation pass & checkpointing ───────────────────────────────
            if has_val:
                self.model_.eval()
                val_correct, val_total = 0, 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device_)
                        yb = yb.to(self.device_)
                        val_correct += (self.model_(xb).argmax(1) == yb).sum().item()
                        val_total   += len(yb)
                val_acc  = val_correct / val_total
                log_msg += f"  val_acc={val_acc:.4f}"

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch   = epoch + 1
                    torch.save(
                        {
                            "epoch":           best_epoch,
                            "model_state":     self.model_.state_dict(),
                            "optim_state":     optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "val_acc":         best_val_acc,
                            "n_classes":       n_classes,
                        },
                        ckpt_path,
                    )
                    log_msg += "  ✓ saved"
                self.model_.train()

            print(log_msg, flush=True)

        # ── Reload best weights ───────────────────────────────────────────────
        if has_val and ckpt_path.exists():
            saved = torch.load(ckpt_path, map_location=self.device_, weights_only=True)
            self.model_.load_state_dict(saved["model_state"])
            print(
                f"  Inception+ResNet: restored best checkpoint  epoch={best_epoch}"
                f"  val_acc={best_val_acc:.4f}  ← {ckpt_path}"
            )

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model_.eval()
        x_t    = torch.tensor(x.astype(np.float32))
        loader = DataLoader(
            TensorDataset(x_t),
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=0,
        )
        preds = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device_)
                preds.append(self.model_(xb).argmax(1).cpu().numpy())
        return np.concatenate(preds)


# ─────────────────────────────────────────────
#  SECTION 6 — DATA SPLITTING
# ─────────────────────────────────────────────

def stratified_split(
    y: np.ndarray, snr: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split indices 70 / 15 / 15 stratified by (class, SNR).
    Returns (train_idx, val_idx, test_idx).
    The val split is used by deep models for checkpointing;
    final reported accuracy always uses the held-out test split.
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
#  SECTION 7 — EVALUATION HELPERS
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
#  SECTION 8 — MAIN ENTRY POINT
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AMC MVP — Stages 1–5")
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
    parser.add_argument("--lstm-hidden",           type=int, default=128,
                        help="LSTM hidden size (default: 128, matching paper [15]).")
    parser.add_argument("--inception-epochs",      type=int, default=50,
                        help="Training epochs for Stage 5 Inception+ResNet (default: 50).")
    parser.add_argument("--skip-lstm",             action="store_true",
                        help="Skip Stage 4 LSTM (run Stages 1–3 and 5 only).")
    parser.add_argument("--skip-stage5",           action="store_true",
                        help="Skip Stage 5 Inception+ResNet.")
    parser.add_argument("--checkpoint-dir",        type=str, default="checkpoints",
                        help="Directory for model checkpoints (default: checkpoints/).")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)

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
        evaluate("stage_1_weak_baseline",    s1, weak,   y, snr, test_idx, n_classes),
        evaluate("stage_2_better_features",  s2, strong, y, snr, test_idx, n_classes),
        evaluate("stage_3_trainable_softmax",s3, strong, y, snr, test_idx, n_classes),
    ]

    # ── 5. Train LSTM (Stage 4) ───────────────────────────────────────────────
    if not args.skip_lstm:
        print(f"\nTraining Stage 4 — LSTM "
              f"(hidden={args.lstm_hidden}, epochs={args.lstm_epochs}) ...")
        s4 = NumpyLSTMClassifier(
            hidden_size = args.lstm_hidden,
            epochs      = args.lstm_epochs,
            seed        = args.seed,
        ).fit(
            x[train_idx], y[train_idx],
            x_val           = x[val_idx],
            y_val           = y[val_idx],
            checkpoint_path = str(ckpt_dir / "stage4_lstm_best.pt"),
        )
        results.append(
            evaluate("stage_4_lstm", s4, x, y, snr, test_idx, n_classes)
        )

    # ── 6. Train Inception + ResNet (Stage 5) ────────────────────────────────
    if not args.skip_stage5:
        print(f"\nTraining Stage 5 — Inception+ResNet "
              f"(epochs={args.inception_epochs}) ...")
        s5 = InceptionResNetClassifier(
            epochs = args.inception_epochs,
            seed   = args.seed,
        ).fit(
            x[train_idx], y[train_idx],
            x_val           = x[val_idx],
            y_val           = y[val_idx],
            checkpoint_path = str(ckpt_dir / "stage5_inception_resnet_best.pt"),
        )
        results.append(
            evaluate("stage_5_inception_resnet", s5, x, y, snr, test_idx, n_classes)
        )

    # ── 7. Save & print results ───────────────────────────────────────────────
    reports_dir = Path("reports") / "numpy_mvp"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"labels": mods, "results": results}, f, indent=2)

    print("\n" + "=" * 60)
    print("AMC MVP results")
    print("=" * 60)
    for row in results:
        print(f"  {row['model']:36s}  overall={row['overall_accuracy']:.4f}")

    # Per-stage high-SNR summary for deep models
    deep_stages = []
    if not args.skip_lstm:
        deep_stages.append(("Stage 4 LSTM",          "stage_4_lstm",             0.90))
    if not args.skip_stage5:
        deep_stages.append(("Stage 5 Inception+ResNet", "stage_5_inception_resnet", 0.9376))

    for label, key, target in deep_stages:
        row = next((r for r in results if r["model"] == key), None)
        if row is None:
            continue
        high_snr = {int(k): v for k, v in row["accuracy_by_snr"].items() if int(k) >= 0}
        if high_snr:
            avg_high = float(np.mean(list(high_snr.values())))
            acc_14   = row["accuracy_by_snr"].get("14", float("nan"))
            print(
                f"\n  {label}: avg acc SNR≥0dB={avg_high:.4f}"
                f"  acc@14dB={acc_14:.4f}  (target≥{target:.4f})"
            )

    print(f"\nCheckpoints : {ckpt_dir}/")
    print(f"Full metrics: {output_path}")


if __name__ == "__main__":
    main()
