import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat
from tqdm import tqdm


torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_DATA_DIR = os.path.join(BASE_DIR, "ISAC_Dataset_N34_Q4")
GENERATED_DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_DIR = GENERATED_DATA_DIR if os.path.isdir(GENERATED_DATA_DIR) and glob.glob(os.path.join(GENERATED_DATA_DIR, "dataset_SNR_*.mat")) else LEGACY_DATA_DIR
SAVE_DIR = os.path.join(BASE_DIR, "checkpoint")
os.makedirs(SAVE_DIR, exist_ok=True)
DEFAULT_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a complex-valued waveform regressor")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--test-samples", type=int, default=1500)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default=str(DEFAULT_DEVICE))
    parser.add_argument("--num-workers", type=int, default=2 if torch.cuda.is_available() else 0)
    return parser.parse_args()


def _load_sample_from_matlab_struct(item):
    if isinstance(item, np.ndarray):
        if item.ndim == 0:
            item = item.item()
        elif item.size == 1:
            item = item.reshape(-1)[0]
        else:
            item = item[0]

    if hasattr(item, "dtype") and item.dtype.names is not None:
        rx = item["rx"]
        a = item["a"]
    elif isinstance(item, dict):
        rx = item["rx"]
        a = item["a"]
    elif hasattr(item, "__dict__") and hasattr(item, "rx") and hasattr(item, "a"):
        rx = item.rx
        a = item.a
    else:
        rx = item
        a = item

    if hasattr(rx, "item") and not isinstance(rx, np.ndarray):
        rx = rx.item()
    if hasattr(a, "item") and not isinstance(a, np.ndarray):
        a = a.item()

    if isinstance(rx, np.ndarray):
        rx_arr = rx.reshape(-1)
    elif isinstance(rx, (list, tuple)):
        rx_arr = np.asarray(rx).reshape(-1)
    else:
        rx_arr = np.asarray(rx).reshape(-1)

    if isinstance(a, np.ndarray):
        a_arr = a.reshape(-1)
    elif isinstance(a, (list, tuple)):
        a_arr = np.asarray(a).reshape(-1)
    else:
        a_arr = np.asarray(a).reshape(-1)

    return rx_arr.astype(np.complex64), a_arr.astype(np.complex64)


def _iter_mat_samples(mat_path):
    try:
        mat = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
        dataset = mat["dataset"]
        if dataset.ndim == 2 and dataset.shape[0] == 1:
            dataset = dataset[0]
        for item in dataset:
            rx, a = _load_sample_from_matlab_struct(item)
            yield rx, a
    except NotImplementedError:
        with h5py.File(mat_path, "r") as f:
            ds = f["dataset"]
            for idx in range(ds["rx"].shape[0]):
                rx_ref = ds["rx"][idx, 0]
                a_ref = ds["a"][idx, 0]
                rx = np.array(f[rx_ref][()]).reshape(-1)
                a = np.array(f[a_ref][()]).reshape(-1)
                yield rx.astype(np.complex64), a.astype(np.complex64)


class WaveformDataset(Dataset):
    def __init__(self, data_dir, max_samples=3000, norm_stats=None):
        self.samples = []
        raw_samples = []
        mat_files = sorted(glob.glob(os.path.join(data_dir, "dataset_SNR_*.mat")))
        count = 0
        for mat_path in mat_files:
            for rx, a in _iter_mat_samples(mat_path):
                if count >= max_samples:
                    break
                raw_samples.append((rx, a))
                count += 1
            if count >= max_samples:
                break

        if norm_stats is None:
            rx_real_stack = np.stack([np.asarray(rx.real, dtype=np.float32).reshape(-1) for rx, _ in raw_samples], axis=0)
            rx_imag_stack = np.stack([np.asarray(rx.imag, dtype=np.float32).reshape(-1) for rx, _ in raw_samples], axis=0)
            a_real_stack = np.stack([np.asarray(a.real, dtype=np.float32).reshape(-1) for _, a in raw_samples], axis=0)
            a_imag_stack = np.stack([np.asarray(a.imag, dtype=np.float32).reshape(-1) for _, a in raw_samples], axis=0)

            self.rx_mean = np.stack([
                rx_real_stack.mean(axis=0, keepdims=True),
                rx_imag_stack.mean(axis=0, keepdims=True),
            ], axis=0).astype(np.float32)
            self.rx_std = np.stack([
                rx_real_stack.std(axis=0, keepdims=True),
                rx_imag_stack.std(axis=0, keepdims=True),
            ], axis=0).astype(np.float32)
            self.rx_std = np.where(self.rx_std < 1e-6, 1.0, self.rx_std)

            self.a_mean = np.stack([
                a_real_stack.mean(axis=0, keepdims=True),
                a_imag_stack.mean(axis=0, keepdims=True),
            ], axis=0).astype(np.float32)
            self.a_std = np.stack([
                a_real_stack.std(axis=0, keepdims=True),
                a_imag_stack.std(axis=0, keepdims=True),
            ], axis=0).astype(np.float32)
            self.a_std = np.where(self.a_std < 1e-6, 1.0, self.a_std)
        else:
            self.rx_mean, self.rx_std, self.a_mean, self.a_std = norm_stats

        for rx, a in raw_samples:
            rx_real = np.asarray(rx.real, dtype=np.float32).reshape(-1)
            rx_imag = np.asarray(rx.imag, dtype=np.float32).reshape(-1)
            a_real = np.asarray(a.real, dtype=np.float32).reshape(-1)
            a_imag = np.asarray(a.imag, dtype=np.float32).reshape(-1)

            rx_real = (rx_real - self.rx_mean[0].reshape(-1)) / self.rx_std[0].reshape(-1)
            rx_imag = (rx_imag - self.rx_mean[1].reshape(-1)) / self.rx_std[1].reshape(-1)
            a_real = (a_real - self.a_mean[0].reshape(-1)) / self.a_std[0].reshape(-1)
            a_imag = (a_imag - self.a_mean[1].reshape(-1)) / self.a_std[1].reshape(-1)

            rx_c = rx_real.astype(np.float32) + 1j * rx_imag.astype(np.float32)
            a_c = a_real.astype(np.float32) + 1j * a_imag.astype(np.float32)
            self.samples.append((rx_c.astype(np.complex64), a_c.astype(np.complex64)))

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rx_c, a_c = self.samples[idx]
        return torch.from_numpy(rx_c).to(torch.complex64), torch.from_numpy(a_c).to(torch.complex64)

    def get_norm_stats(self):
        return self.rx_mean, self.rx_std, self.a_mean, self.a_std


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.complex64))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.complex64))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / np.sqrt(self.in_features)
        with torch.no_grad():
            nn.init.uniform_(self.weight.real, -bound, bound)
            nn.init.uniform_(self.weight.imag, -bound, bound)
            if self.bias is not None:
                nn.init.zeros_(self.bias.real)
                nn.init.zeros_(self.bias.imag)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_flat = x.reshape(-1, self.in_features)
        out = torch.matmul(x_flat, self.weight.conj().t())
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*x.shape[:-1], self.out_features)


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features, dtype=torch.complex64))
        self.bias = nn.Parameter(torch.zeros(num_features, dtype=torch.complex64))
        self.register_buffer("running_mean", torch.zeros(num_features, dtype=torch.complex64))
        self.register_buffer("running_var", torch.ones(num_features, dtype=torch.float32))

    def forward(self, x):
        if x.dim() != 2:
            x = x.reshape(-1, self.num_features)
        if self.training:
            mean = x.mean(dim=0)
            centered = x - mean
            var = (centered.real.pow(2) + centered.imag.pow(2)).mean(dim=0)
            var = torch.clamp(var, min=self.eps)
            inv_std = torch.rsqrt(var)
            out = (x - mean) * inv_std.unsqueeze(0)
            self.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)
        else:
            inv_std = torch.rsqrt(self.running_var + self.eps)
            out = (x - self.running_mean) * inv_std.unsqueeze(0)
        return out * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class ComplexReLU(nn.Module):
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))


class WaveformRegressor(nn.Module):
    def __init__(self, input_len, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            ComplexLinear(input_len, 384),
            ComplexBatchNorm1d(384),
            ComplexReLU(),
            ComplexLinear(384, 192),
            ComplexBatchNorm1d(192),
            ComplexReLU(),
            ComplexLinear(192, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def main(args=None):
    args = parse_args() if args is None else args
    device = torch.device(args.device)
    print("=== Training Phase 1 ===")
    train_dataset = WaveformDataset(DATA_DIR, max_samples=args.max_samples)
    dataset_size = len(train_dataset)
    val_size = max(1, int(dataset_size * args.val_split)) if dataset_size > 2 else 0
    train_size = dataset_size - val_size

    if val_size > 0:
        train_subset, val_subset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    else:
        train_subset = train_dataset
        val_loader = None
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    sample_rx, sample_a = train_dataset[0]
    model = WaveformRegressor(sample_rx.shape[0], sample_a.shape[0]).to(device)

    def criterion(pred, target):
        diff = pred - target
        return (
            0.7 * F.mse_loss(pred.real, target.real)
            + 0.7 * F.mse_loss(pred.imag, target.imag)
            + 0.2 * F.mse_loss(pred.abs(), target.abs())
            + 0.1 * torch.mean(torch.abs(diff))
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val_loss = float("inf")
    best_state = None
    patience = 5
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for rx, a in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            rx, a = rx.to(device), a.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(rx)
            loss = criterion(pred, a)
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = p.grad / (p.grad.norm() / 1.0 + 1e-12)
            optimizer.step()
            running += loss.item()

        train_loss = running / max(1, len(train_loader))
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.inference_mode():
                for rx, a in val_loader:
                    rx, a = rx.to(device), a.to(device)
                    pred = model(rx)
                    val_running += criterion(pred, a).item()
            val_loss = val_running / max(1, len(val_loader))
            print(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        else:
            val_loss = train_loss
            print(f"Epoch {epoch} | Loss: {train_loss:.6f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_len": sample_rx.shape[0],
        "output_dim": sample_a.shape[0],
        "rx_mean": train_dataset.rx_mean,
        "rx_std": train_dataset.rx_std,
        "a_mean": train_dataset.a_mean,
        "a_std": train_dataset.a_std,
    }, os.path.join(SAVE_DIR, "phase1_best.pth"))
    print("Model trained and saved!")

    print("\n=== Evaluating ===")
    checkpoint = torch.load(os.path.join(SAVE_DIR, "phase1_best.pth"), map_location=device, weights_only=False)
    model = WaveformRegressor(checkpoint["input_len"], checkpoint["output_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = WaveformDataset(DATA_DIR, max_samples=args.test_samples, norm_stats=train_dataset.get_norm_stats())
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    mse_list, mae_list, rmse_list, evm_list, corr_list = [], [], [], [], []
    with torch.inference_mode():
        for rx, a in tqdm(test_loader, desc="Evaluating"):
            rx = rx.to(device)
            pred = model(rx).cpu()
            diff = pred - a.cpu()
            diff_mag = torch.abs(diff)
            mse = torch.mean(diff_mag ** 2, dim=1)
            mae = torch.mean(diff_mag, dim=1)
            rmse = torch.sqrt(torch.mean(diff_mag ** 2, dim=1))
            ref_power = torch.mean(torch.abs(a.cpu()) ** 2, dim=1)
            err_power = torch.mean(diff_mag ** 2, dim=1)
            evm_db = 20 * torch.log10(torch.sqrt(err_power / (ref_power + 1e-12)))

            pred_flat = pred.detach().numpy().reshape(-1)
            a_flat = a.detach().numpy().reshape(-1)
            corr = np.corrcoef(np.abs(pred_flat), np.abs(a_flat))[0, 1] if pred_flat.size > 1 else 0.0
            if np.isnan(corr):
                corr = 0.0

            mse_list.extend(mse.numpy())
            mae_list.extend(mae.numpy())
            rmse_list.extend(rmse.numpy())
            evm_list.extend(evm_db.numpy())
            corr_list.append(corr)

    print("\n" + "=" * 50)
    print("Phase 1 Evaluation Results")
    print("=" * 50)
    print(f"Test samples     : {len(test_dataset)}")
    print(f"Average MSE      : {np.mean(mse_list):.6f}")
    print(f"Average MAE      : {np.mean(mae_list):.6f}")
    print(f"Average RMSE     : {np.mean(rmse_list):.6f}")
    print(f"Average EVM (dB) : {np.mean(evm_list):.2f} dB")
    print(f"Correlation      : {np.mean(corr_list):.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
