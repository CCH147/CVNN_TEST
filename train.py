# train.py
"""
CVNN training script for Hermitian-Symmetric OFDM waveform regression.

This version removes the SNR-aware experiment and fixes two major experimental
issues in the previous training script:

1. No train/test leakage:
   The previous version loaded the first N samples for both training and testing.
   This file uses disjoint per-SNR ranges:
       train: samples [0 : train_samples_per_snr)
       test : samples [train_samples_per_snr : train_samples_per_snr + test_samples_per_snr)

2. Correct structured SER pairing:
   In generate_dataset.py, QPSK symbol k is stored as:
       real part -> a[k]
       imag part -> a[D-k-1]
   Therefore SER must pair:
       a[0] with a[18]
       a[1] with a[17]
       ...
       a[8] with a[10]

3. SNR-aware model is removed:
   The compare experiment now only includes:
       - High-SNR Only
       - All-SNR

Important note:
---------------
If generate_dataset.py currently maps

    ck[k] = sqrt(2)/2 * (a[k] + 1j * a[D-k-1])

while a[D-k-1] already equals j * Im(symbol), then the imaginary component is
multiplied by j twice. This makes ck[k] proportional to Re(symbol) - Im(symbol),
which creates an ambiguity. That issue is in the generator, not only in train.py.
"""

import os
import re
import json
import glob
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm


# =============================================================================
# Basic utilities
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def parse_hidden_dims(text: str) -> Tuple[int, ...]:
    dims = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not dims:
        raise ValueError("--hidden-dims cannot be empty.")
    return dims


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Train CVNN for Hermitian-Symmetric OFDM waveform regression."
    )

    # Paths
    parser.add_argument("--data-dir", type=str, default=str(base_dir / "data1"))
    parser.add_argument("--save-dir", type=str, default=str(base_dir / "checkpoint"))
    parser.add_argument("--results-dir", type=str, default=str(base_dir / "results"))

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="compare",
        choices=["single", "compare", "curriculum"],
    )

    # Dataset split
    parser.add_argument(
        "--train-samples-per-snr",
        type=int,
        default=2500,
        help="Number of training samples per SNR, loaded from the beginning of each file.",
    )
    parser.add_argument(
        "--test-samples-per-snr",
        type=int,
        default=500,
        help="Number of testing samples per SNR, loaded after the training range.",
    )

    # Backward-compatible alias. If set, it overrides train_samples_per_snr.
    parser.add_argument(
        "--samples-per-snr",
        type=int,
        default=None,
        help="Alias for --train-samples-per-snr.",
    )

    # Optimization
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-6)

    # Model
    parser.add_argument("--hidden-dims", type=str, default="1024,512,256,128")
    parser.add_argument("--dropout", type=float, default=0.0)

    # Loss weights
    parser.add_argument("--w-main", type=float, default=1.0)
    parser.add_argument("--w-leak", type=float, default=0.10)
    parser.add_argument("--w-zero", type=float, default=0.20)
    parser.add_argument("--w-l1", type=float, default=0.02)

    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2 if torch.cuda.is_available() else 0)

    args = parser.parse_args()

    if args.samples_per_snr is not None:
        args.train_samples_per_snr = args.samples_per_snr

    return args


# =============================================================================
# MATLAB loading
# =============================================================================

def _unwrap_mat_item(item):
    while isinstance(item, np.ndarray) and item.size == 1:
        item = item.reshape(-1)[0]
    return item


def _to_complex_1d(x) -> np.ndarray:
    x = _unwrap_mat_item(x)
    return np.asarray(x).reshape(-1).astype(np.complex64)


def _to_float_scalar(x, default: float = 20.0) -> float:
    try:
        x = _unwrap_mat_item(x)
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return float(default)


def _load_sample_from_matlab_struct(item) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    item = _unwrap_mat_item(item)

    if hasattr(item, "_fieldnames"):
        rx = _to_complex_1d(getattr(item, "rx"))
        a = _to_complex_1d(getattr(item, "a"))
        snr = _to_float_scalar(getattr(item, "snr", 20.0))
        s = _to_complex_1d(getattr(item, "s", rx))
        return rx, a, snr, s

    if hasattr(item, "dtype") and item.dtype.names is not None:
        rx = _to_complex_1d(item["rx"])
        a = _to_complex_1d(item["a"])
        snr = _to_float_scalar(item["snr"] if "snr" in item.dtype.names else 20.0)
        s = _to_complex_1d(item["s"] if "s" in item.dtype.names else rx)
        return rx, a, snr, s

    if isinstance(item, dict):
        rx = _to_complex_1d(item["rx"])
        a = _to_complex_1d(item["a"])
        snr = _to_float_scalar(item.get("snr", 20.0))
        s = _to_complex_1d(item.get("s", rx))
        return rx, a, snr, s

    raise TypeError(f"Unsupported MATLAB item type: {type(item)}")


def _iter_scipy_mat_samples(mat_path: str):
    mat = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    if "dataset" not in mat:
        raise KeyError(f"{mat_path} does not contain variable 'dataset'.")
    dataset = np.asarray(mat["dataset"]).reshape(-1)
    for item in dataset:
        yield _load_sample_from_matlab_struct(item)


def _decode_hdf5_complex(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return arr.reshape(-1).astype(np.complex64)
    if arr.dtype.fields is not None:
        names = arr.dtype.names
        if "real" in names and "imag" in names:
            return (arr["real"] + 1j * arr["imag"]).reshape(-1).astype(np.complex64)
        if "r" in names and "i" in names:
            return (arr["r"] + 1j * arr["i"]).reshape(-1).astype(np.complex64)
    return arr.reshape(-1).astype(np.complex64)


def _read_hdf5_ref(f: h5py.File, ref):
    return np.asarray(f[ref][()])


def _iter_hdf5_mat_samples(mat_path: str):
    with h5py.File(mat_path, "r") as f:
        ds = f["dataset"]
        if not isinstance(ds, h5py.Group):
            raise RuntimeError("Unsupported HDF5 dataset layout.")

        n = ds["rx"].shape[0]
        for i in range(n):
            rx_raw = _read_hdf5_ref(f, ds["rx"][i, 0]) if ds["rx"].dtype == h5py.ref_dtype else ds["rx"][i]
            a_raw = _read_hdf5_ref(f, ds["a"][i, 0]) if ds["a"].dtype == h5py.ref_dtype else ds["a"][i]
            s_raw = rx_raw
            if "s" in ds:
                s_raw = _read_hdf5_ref(f, ds["s"][i, 0]) if ds["s"].dtype == h5py.ref_dtype else ds["s"][i]

            snr = 20.0
            if "snr" in ds:
                snr_raw = _read_hdf5_ref(f, ds["snr"][i, 0]) if ds["snr"].dtype == h5py.ref_dtype else ds["snr"][i]
                snr = float(np.asarray(snr_raw).reshape(-1)[0])

            yield (
                _decode_hdf5_complex(rx_raw),
                _decode_hdf5_complex(a_raw),
                snr,
                _decode_hdf5_complex(s_raw),
            )


def iter_mat_samples(mat_path: str):
    try:
        yield from _iter_scipy_mat_samples(mat_path)
    except NotImplementedError:
        yield from _iter_hdf5_mat_samples(mat_path)


# =============================================================================
# Dataset
# =============================================================================

def parse_snr_from_filename(path: str) -> Optional[int]:
    name = os.path.basename(path)
    m = re.search(r"dataset_SNR_(-?\d+)dB\.mat", name)
    if m is None:
        return None
    return int(m.group(1))


def snr_passes_filter(snr_val: float, snr_filter: Optional[str]) -> bool:
    if snr_filter is None or snr_filter == "all":
        return True

    text = str(snr_filter).strip().replace(" ", "")

    if "," in text:
        allowed = {float(v) for v in text.split(",") if v != ""}
        return float(snr_val) in allowed
    if text.startswith(">="):
        return snr_val >= float(text[2:])
    if text.startswith("<="):
        return snr_val <= float(text[2:])
    if text.startswith(">"):
        return snr_val > float(text[1:])
    if text.startswith("<"):
        return snr_val < float(text[1:])
    if text.startswith("=="):
        return snr_val == float(text[2:])

    try:
        return snr_val == float(text)
    except ValueError:
        return True


class WaveformDataset(Dataset):
    """
    Dataset with disjoint per-SNR range support.

    start_per_snr:
        Number of samples to skip at the beginning of each SNR file.

    samples_per_snr:
        Number of samples to load after skipping.
    """

    def __init__(
        self,
        data_dir: str,
        samples_per_snr: int,
        start_per_snr: int = 0,
        snr_filter: Optional[str] = "all",
        verbose: bool = True,
    ):
        super().__init__()

        self.data_dir = str(data_dir)
        self.samples_per_snr = int(samples_per_snr)
        self.start_per_snr = int(start_per_snr)
        self.snr_filter = snr_filter

        self.samples: List[Tuple[np.ndarray, np.ndarray, np.float32, np.ndarray]] = []

        mat_files = sorted(glob.glob(os.path.join(self.data_dir, "dataset_SNR_*.mat")))
        if not mat_files:
            raise FileNotFoundError(
                f"No dataset_SNR_*.mat files found in {self.data_dir}. "
                f"Check --data-dir."
            )

        if verbose:
            print(f"Loading data from: {self.data_dir}")
            print(f"SNR filter       : {snr_filter}")
            print(f"start_per_snr    : {self.start_per_snr}")
            print(f"samples_per_snr  : {self.samples_per_snr}")

        for mat_path in mat_files:
            snr_from_file = parse_snr_from_filename(mat_path)
            if snr_from_file is not None and not snr_passes_filter(snr_from_file, snr_filter):
                continue

            loaded = 0
            seen_in_file = 0

            for rx, a, snr, s in iter_mat_samples(mat_path):
                if not snr_passes_filter(snr, snr_filter):
                    continue

                if seen_in_file < self.start_per_snr:
                    seen_in_file += 1
                    continue

                if loaded >= self.samples_per_snr:
                    break

                rx = np.asarray(rx).reshape(-1).astype(np.complex64)
                a = np.asarray(a).reshape(-1).astype(np.complex64)
                s = np.asarray(s).reshape(-1).astype(np.complex64)

                self.samples.append((rx, a, np.float32(snr), s))
                loaded += 1
                seen_in_file += 1

            if verbose:
                print(f"  {os.path.basename(mat_path)}: loaded {loaded}")

        if not self.samples:
            raise ValueError("No samples loaded.")

        self.input_len = int(self.samples[0][0].shape[0])
        self.output_dim = int(self.samples[0][1].shape[0])

        counts = defaultdict(int)
        for _, _, snr, _ in self.samples:
            counts[int(snr)] += 1

        if verbose:
            print(f"Total samples: {len(self.samples)}")
            print(f"Input length : {self.input_len}")
            print(f"Output dim   : {self.output_dim}")
            print(f"SNR counts   : {dict(sorted(counts.items()))}")

            if self.output_dim == 19:
                a0 = self.samples[0][1]
                print("Target structure check:")
                print(f"  abs(a[9])             = {abs(a0[9]):.3e}")
                print(f"  max abs imag(a[:9])   = {np.max(np.abs(a0[:9].imag)):.3e}")
                print(f"  max abs real(a[10:])  = {np.max(np.abs(a0[10:].real)):.3e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rx, a, snr, s = self.samples[idx]
        return (
            torch.from_numpy(rx),
            torch.from_numpy(a),
            torch.tensor(snr, dtype=torch.float32),
            torch.from_numpy(s),
        )


# =============================================================================
# Complex NN layers
# =============================================================================

class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=torch.complex64)
        )
        self.bias = (
            nn.Parameter(torch.zeros(self.out_features, dtype=torch.complex64))
            if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            nn.init.uniform_(self.weight.real, -bound, bound)
            nn.init.uniform_(self.weight.imag, -bound, bound)
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        y = torch.matmul(x_flat, self.weight.conj().t())
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features)


class ComplexLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim, dtype=torch.complex64))
        self.bias = nn.Parameter(torch.zeros(self.dim, dtype=torch.complex64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        var = (centered.real.pow(2) + centered.imag.pow(2)).mean(dim=-1, keepdim=True)
        y = centered / torch.sqrt(var + self.eps)
        return y * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class ModReLU(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(int(dim), dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(z)
        scale = F.relu(mag + self.bias.unsqueeze(0))
        return scale * z / (mag + self.eps)


class ComplexDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0:
            return z
        keep = 1.0 - self.p
        mask = torch.empty_like(z.real).bernoulli_(keep) / keep
        return torch.complex(z.real * mask, z.imag * mask)


class WaveformRegressor(nn.Module):
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (1024, 512, 256, 128),
        dropout: float = 0.0,
    ):
        super().__init__()

        layers: List[nn.Module] = []
        prev = int(input_len)

        for h in hidden_dims:
            layers.append(ComplexLinear(prev, h))
            layers.append(ComplexLayerNorm(h))
            layers.append(ModReLU(h))
            if dropout > 0:
                layers.append(ComplexDropout(dropout))
            prev = h

        layers.append(ComplexLinear(prev, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Loss and metrics
# =============================================================================

def structured_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    w_main: float = 1.0,
    w_leak: float = 0.10,
    w_zero: float = 0.20,
    w_l1: float = 0.02,
) -> torch.Tensor:
    if pred.shape[-1] != 19:
        diff = pred - target
        mse = F.mse_loss(pred.real, target.real) + F.mse_loss(pred.imag, target.imag)
        return mse + w_l1 * torch.mean(torch.abs(diff))

    # Valid target components.
    loss_real = F.mse_loss(pred[:, :9].real, target[:, :9].real)
    loss_imag = F.mse_loss(pred[:, 10:].imag, target[:, 10:].imag)

    # Invalid leakage components.
    leak_front = F.mse_loss(pred[:, :9].imag, torch.zeros_like(pred[:, :9].imag))
    leak_back = F.mse_loss(pred[:, 10:].real, torch.zeros_like(pred[:, 10:].real))

    # Structural zero.
    zero_mid = torch.mean(torch.abs(pred[:, 9]).pow(2))

    # Mild full-vector L1.
    l1 = torch.mean(torch.abs(pred - target))

    return (
        w_main * (loss_real + loss_imag)
        + w_leak * (leak_front + leak_back)
        + w_zero * zero_mid
        + w_l1 * l1
    )


def qpsk_components_for_ser(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return QPSK components in original symbol order.

    generate_dataset.py stores:
        symbol k real -> a[k]
        symbol k imag -> a[D-k-1]

    Therefore:
        real_part[k] = a[k].real
        imag_part[k] = a[18-k].imag
    """
    real_part = x[:, :9].real
    imag_part = torch.flip(x[:, 10:].imag, dims=[1])
    return real_part, imag_part


def structured_ser(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.shape[-1] != 19:
        return float("nan")

    with torch.no_grad():
        pr, pi = qpsk_components_for_ser(pred)
        tr, ti = qpsk_components_for_ser(target)

        real_err = (pr >= 0) != (tr >= 0)
        imag_err = (pi >= 0) != (ti >= 0)

        sym_err = real_err | imag_err
        return sym_err.float().mean().item()


def structured_bit_error_rate(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.shape[-1] != 19:
        return float("nan")

    with torch.no_grad():
        pr, pi = qpsk_components_for_ser(pred)
        tr, ti = qpsk_components_for_ser(target)

        real_err = ((pr >= 0) != (tr >= 0)).float()
        imag_err = ((pi >= 0) != (ti >= 0)).float()

        return torch.cat([real_err, imag_err], dim=1).mean().item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    diff = pred - target
    mse = torch.mean(torch.abs(diff).pow(2)).item()
    mae = torch.mean(torch.abs(diff)).item()
    ref_power = torch.mean(torch.abs(target).pow(2)).item()
    evm_db = 10.0 * math.log10(mse / (ref_power + 1e-12))
    ser = structured_ser(pred, target)
    ber = structured_bit_error_rate(pred, target)
    return {"mse": mse, "mae": mae, "evm_db": evm_db, "ser": ser, "ber": ber}


# =============================================================================
# Training
# =============================================================================

def clip_complex_gradients(model: nn.Module, max_norm: float) -> float:
    if max_norm is None or max_norm <= 0:
        return 0.0

    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0

    total_sq = 0.0
    for p in params:
        g = p.grad
        if torch.is_complex(g):
            total_sq += g.real.detach().norm(2).item() ** 2
            total_sq += g.imag.detach().norm(2).item() ** 2
        else:
            total_sq += g.detach().norm(2).item() ** 2

    total_norm = math.sqrt(total_sq)
    coef = max_norm / (total_norm + 1e-6)

    if coef < 1.0:
        for p in params:
            g = p.grad
            if torch.is_complex(g):
                p.grad = torch.complex(g.real * coef, g.imag * coef).resolve_conj()
            else:
                g.mul_(coef)

    for p in params:
        if p.grad is not None:
            p.grad = p.grad.resolve_conj()

    return total_norm


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def make_model(input_len: int, output_dim: int, args: argparse.Namespace) -> nn.Module:
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    return WaveformRegressor(
        input_len=input_len,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    )


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=False,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.train()
    total = 0.0

    for rx, a, _, _ in tqdm(loader, desc="Training", leave=False):
        rx = rx.to(device, non_blocking=True)
        a = a.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(rx)

        loss = structured_loss(
            pred,
            a,
            w_main=args.w_main,
            w_leak=args.w_leak,
            w_zero=args.w_zero,
            w_l1=args.w_l1,
        )

        loss.backward()
        clip_complex_gradients(model, args.grad_clip)
        optimizer.step()

        total += loss.item()

    return total / max(len(loader), 1)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[float, Dict[int, Dict[str, float]]]:
    model.eval()

    total_loss = 0.0
    by_snr = defaultdict(lambda: {"mse": [], "mae": [], "evm_db": [], "ser": [], "ber": []})

    for rx, a, snr, _ in tqdm(loader, desc="Evaluating", leave=False):
        rx = rx.to(device, non_blocking=True)
        a = a.to(device, non_blocking=True)

        pred = model(rx)

        loss = structured_loss(
            pred,
            a,
            w_main=args.w_main,
            w_leak=args.w_leak,
            w_zero=args.w_zero,
            w_l1=args.w_l1,
        )
        total_loss += loss.item()

        for i in range(rx.shape[0]):
            key = int(round(float(snr[i].item())))
            m = compute_metrics(pred[i:i+1], a[i:i+1])
            for k, v in m.items():
                if not math.isnan(v):
                    by_snr[key][k].append(v)

    summary = {}
    for snr_key, vals in sorted(by_snr.items()):
        summary[snr_key] = {
            "mse": float(np.mean(vals["mse"])),
            "mae": float(np.mean(vals["mae"])),
            "evm_db": float(np.mean(vals["evm_db"])),
            "ser": float(np.mean(vals["ser"])),
            "ber": float(np.mean(vals["ber"])),
            "count": int(len(vals["mse"])),
        }

    return total_loss / max(len(loader), 1), summary


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_checkpoint(
    model: nn.Module,
    path: str,
    input_len: int,
    output_dim: int,
    args: argparse.Namespace,
    extra: Optional[dict] = None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "input_len": int(input_len),
        "output_dim": int(output_dim),
        "hidden_dims": parse_hidden_dims(args.hidden_dims),
        "args": vars(args),
        "extra": extra or {},
    }
    torch.save(payload, path)
    print(f"Saved checkpoint: {path}")


def print_table(results: Dict[str, Dict[int, Dict[str, float]]]) -> None:
    names = list(results.keys())
    snrs = sorted({s for r in results.values() for s in r.keys()})

    print("\n" + "=" * 100)
    print("SNR-wise results")
    print("=" * 100)

    for metric in ["mse", "evm_db", "ser", "ber"]:
        print(f"\nMetric: {metric.upper()}")
        header = f"{'SNR':>6}" + "".join([f"{name:>24}" for name in names])
        print(header)
        print("-" * len(header))

        for snr in snrs:
            row = f"{snr:>6}"
            for name in names:
                val = results[name].get(snr, {}).get(metric, float("nan"))
                row += f"{val:>24.6f}"
            print(row)


def plot_results(results: Dict[str, Dict[int, Dict[str, float]]], path: str) -> None:
    snrs = sorted({s for r in results.values() for s in r.keys()})
    if not snrs:
        return

    metrics = [
        ("mse", "MSE"),
        ("evm_db", "EVM (dB)"),
        ("ser", "Structured SER"),
        ("ber", "Structured BER"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("CVNN Performance vs SNR", fontsize=14)

    for ax, (metric, title) in zip(axes, metrics):
        for name, data in results.items():
            y = [data.get(s, {}).get(metric, float("nan")) for s in snrs]
            ax.plot(snrs, y, marker="o", linewidth=2, label=name)

        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


# =============================================================================
# Experiment runners
# =============================================================================

def make_train_dataset(args: argparse.Namespace, snr_filter: str) -> WaveformDataset:
    return WaveformDataset(
        data_dir=args.data_dir,
        samples_per_snr=args.train_samples_per_snr,
        start_per_snr=0,
        snr_filter=snr_filter,
        verbose=True,
    )


def make_test_dataset(args: argparse.Namespace, snr_filter: str = "all") -> WaveformDataset:
    return WaveformDataset(
        data_dir=args.data_dir,
        samples_per_snr=args.test_samples_per_snr,
        start_per_snr=args.train_samples_per_snr,
        snr_filter=snr_filter,
        verbose=True,
    )


def train_model(
    name: str,
    args: argparse.Namespace,
    train_filter: str,
    test_loader: DataLoader,
    input_len: int,
    output_dim: int,
    device: torch.device,
) -> Dict[int, Dict[str, float]]:
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    train_ds = make_train_dataset(args, train_filter)
    train_loader = make_loader(train_ds, args, shuffle=True)

    model = make_model(input_len, output_dim, args).to(device)
    optimizer = make_optimizer(model, args)

    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, args)
        history.append({"epoch": epoch, "train_loss": loss})
        print(f"Epoch {epoch:03d}/{args.epochs} | train loss = {loss:.6f}")

    _, summary = evaluate(model, test_loader, device, args)

    safe = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()
    save_checkpoint(
        model,
        os.path.join(args.save_dir, f"{safe}.pt"),
        input_len,
        output_dim,
        args,
        extra={"history": history, "train_filter": train_filter},
    )

    return summary


def run_compare(args: argparse.Namespace):
    device = torch.device(args.device)

    print("\n" + "=" * 70)
    print("Compare Mode: High-SNR Only vs All-SNR")
    print("=" * 70)

    # Reference dataset only to get dimensions.
    ref = WaveformDataset(
        data_dir=args.data_dir,
        samples_per_snr=1,
        start_per_snr=0,
        snr_filter="all",
        verbose=True,
    )
    input_len = ref.input_len
    output_dim = ref.output_dim

    test_ds = make_test_dataset(args, "all")
    test_loader = make_loader(test_ds, args, shuffle=False)

    results = {}

    results["High-SNR Only"] = train_model(
        name="CVNN High-SNR Only >=15dB",
        args=args,
        train_filter=">=15",
        test_loader=test_loader,
        input_len=input_len,
        output_dim=output_dim,
        device=device,
    )

    results["All-SNR"] = train_model(
        name="CVNN All-SNR",
        args=args,
        train_filter="all",
        test_loader=test_loader,
        input_len=input_len,
        output_dim=output_dim,
        device=device,
    )

    print_table(results)

    save_json(results, os.path.join(args.results_dir, "compare_metrics_no_snr_aware.json"))
    plot_results(results, os.path.join(args.results_dir, "compare_snr_performance_no_snr_aware.png"))

    return results


def run_single(args: argparse.Namespace):
    device = torch.device(args.device)

    ref = WaveformDataset(
        data_dir=args.data_dir,
        samples_per_snr=1,
        start_per_snr=0,
        snr_filter="all",
        verbose=True,
    )
    input_len = ref.input_len
    output_dim = ref.output_dim

    train_ds = make_train_dataset(args, "all")
    test_ds = make_test_dataset(args, "all")

    train_loader = make_loader(train_ds, args, shuffle=True)
    test_loader = make_loader(test_ds, args, shuffle=False)

    model = make_model(input_len, output_dim, args).to(device)
    optimizer = make_optimizer(model, args)

    best_loss = float("inf")
    bad = 0
    best_path = os.path.join(args.save_dir, "cvnn_single_best.pt")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args)
        test_loss, summary = evaluate(model, test_loader, device, args)
        history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.6f} | test={test_loss:.6f}"
        )

        if test_loss < best_loss - args.min_delta:
            best_loss = test_loss
            bad = 0
            save_checkpoint(
                model,
                best_path,
                input_len,
                output_dim,
                args,
                extra={"best_epoch": epoch, "best_test_loss": best_loss},
            )
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    payload = torch.load(best_path, map_location=device)
    model.load_state_dict(payload["state_dict"])

    _, final_summary = evaluate(model, test_loader, device, args)
    results = {"All-SNR": final_summary}

    print_table(results)
    save_json(
        {"history": history, "summary": final_summary},
        os.path.join(args.results_dir, "single_metrics_no_snr_aware.json"),
    )
    plot_results(results, os.path.join(args.results_dir, "single_snr_performance_no_snr_aware.png"))

    return results


def run_curriculum(args: argparse.Namespace):
    device = torch.device(args.device)

    ref = WaveformDataset(
        data_dir=args.data_dir,
        samples_per_snr=1,
        start_per_snr=0,
        snr_filter="all",
        verbose=True,
    )
    input_len = ref.input_len
    output_dim = ref.output_dim

    model = make_model(input_len, output_dim, args).to(device)

    stages = [
        (">=20", max(1, args.epochs // 3), args.lr),
        (">=10", max(1, args.epochs // 3), args.lr * 0.5),
        ("all", max(1, args.epochs - 2 * max(1, args.epochs // 3)), args.lr * 0.25),
    ]

    history = []

    for stage_idx, (snr_filter, stage_epochs, lr) in enumerate(stages, start=1):
        print("\n" + "=" * 70)
        print(f"Curriculum Stage {stage_idx}: filter={snr_filter}, epochs={stage_epochs}, lr={lr}")
        print("=" * 70)

        ds = WaveformDataset(
            data_dir=args.data_dir,
            samples_per_snr=args.train_samples_per_snr,
            start_per_snr=0,
            snr_filter=snr_filter,
            verbose=True,
        )
        loader = make_loader(ds, args, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, foreach=False)

        for epoch in range(1, stage_epochs + 1):
            loss = train_one_epoch(model, loader, optimizer, device, args)
            history.append(
                {
                    "stage": stage_idx,
                    "filter": snr_filter,
                    "epoch": epoch,
                    "loss": loss,
                }
            )
            print(f"Stage {stage_idx} Epoch {epoch:03d}/{stage_epochs} | loss={loss:.6f}")

    test_ds = make_test_dataset(args, "all")
    test_loader = make_loader(test_ds, args, shuffle=False)

    _, summary = evaluate(model, test_loader, device, args)
    results = {"Curriculum": summary}

    print_table(results)

    save_checkpoint(
        model,
        os.path.join(args.save_dir, "cvnn_curriculum_no_snr_aware.pt"),
        input_len,
        output_dim,
        args,
        extra={"history": history, "stages": stages},
    )
    save_json(
        {"history": history, "summary": summary},
        os.path.join(args.results_dir, "curriculum_metrics_no_snr_aware.json"),
    )
    plot_results(results, os.path.join(args.results_dir, "curriculum_snr_performance_no_snr_aware.png"))

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    configure_torch()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 70)
    print("CVNN Training - No SNR-Aware Version")
    print("=" * 70)
    print(f"Device                : {args.device}")
    print(f"Data dir              : {args.data_dir}")
    print(f"Train samples / SNR   : {args.train_samples_per_snr}")
    print(f"Test samples / SNR    : {args.test_samples_per_snr}")
    print(f"Test start / SNR      : {args.train_samples_per_snr}")
    print(f"Mode                  : {args.mode}")
    print(f"Hidden dims           : {args.hidden_dims}")
    print("=" * 70)

    if args.mode == "compare":
        run_compare(args)
    elif args.mode == "single":
        run_single(args)
    elif args.mode == "curriculum":
        run_curriculum(args)
    else:
        raise ValueError(args.mode)

    print("\nDone.")


if __name__ == "__main__":
    main()
