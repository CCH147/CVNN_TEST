import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat
from tqdm import tqdm
from collections import defaultdict

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # 兼容旧版本PyTorch
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        print("Warning: torch.set_float32_matmul_precision not available in this PyTorch version")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_DATA_DIR = os.path.join(BASE_DIR, "ISAC_Dataset_N34_Q4")
GENERATED_DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_DIR = GENERATED_DATA_DIR if os.path.isdir(GENERATED_DATA_DIR) and glob.glob(os.path.join(GENERATED_DATA_DIR, "dataset_SNR_*.mat")) else LEGACY_DATA_DIR
SAVE_DIR = os.path.join(BASE_DIR, "checkpoint")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a complex-valued waveform regressor")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--test-samples", type=int, default=1500)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default=str(DEFAULT_DEVICE))
    parser.add_argument("--num-workers", type=int, default=2 if torch.cuda.is_available() else 0)
    parser.add_argument("--snr-filter", type=str, default=None, help="e.g., '>=15' or '==20' or 'all'")
    parser.add_argument("--mode", type=str, default="compare", choices=["compare", "curriculum", "snr_aware"])
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
        snr = item["snr"] if "snr" in item.dtype.names else 20.0
        s = item["s"] if "s" in item.dtype.names else rx
    elif isinstance(item, dict):
        rx = item["rx"]
        a = item["a"]
        snr = item.get("snr", 20.0)
        s = item.get("s", rx)
    elif hasattr(item, "__dict__") and hasattr(item, "rx") and hasattr(item, "a"):
        rx = item.rx
        a = item.a
        snr = item.snr if hasattr(item, "snr") else 20.0
        s = item.s if hasattr(item, "s") else rx
    else:
        rx = item
        a = item
        snr = 20.0
        s = rx

    def to_array(x):
        if hasattr(x, "item") and not isinstance(x, np.ndarray):
            x = x.item()
        if isinstance(x, np.ndarray):
            return x.reshape(-1)
        elif isinstance(x, (list, tuple)):
            return np.asarray(x).reshape(-1)
        else:
            return np.asarray(x).reshape(-1)

    rx_arr = to_array(rx).astype(np.complex64)
    a_arr = to_array(a).astype(np.complex64)
    s_arr = to_array(s).astype(np.complex64)
    
    if isinstance(snr, np.ndarray):
        snr = float(snr.reshape(-1)[0])
    else:
        snr = float(snr)

    return rx_arr, a_arr, snr, s_arr

def _iter_mat_samples(mat_path):
    try:
        mat = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
        dataset = mat["dataset"]
        if dataset.ndim == 2 and dataset.shape[0] == 1:
            dataset = dataset[0]
        for item in dataset:
            rx, a, snr, s = _load_sample_from_matlab_struct(item)
            yield rx, a, snr, s
    except NotImplementedError:
        with h5py.File(mat_path, "r") as f:
            ds = f["dataset"]
            for idx in range(ds["rx"].shape[0]):
                rx_ref = ds["rx"][idx, 0]
                a_ref = ds["a"][idx, 0]
                rx = np.array(f[rx_ref][()]).reshape(-1).astype(np.complex64)
                a = np.array(f[a_ref][()]).reshape(-1).astype(np.complex64)
                snr = 20.0
                s = rx
                yield rx, a, snr, s

class WaveformDataset(Dataset):
    def __init__(self, data_dir, max_samples=3000, norm_stats=None, snr_filter=None):
        self.samples = []
        raw_samples = []
        mat_files = sorted(glob.glob(os.path.join(data_dir, "dataset_SNR_*.mat")))
        count = 0
        
        print(f"Loading data with SNR filter: {snr_filter}")
        
        for mat_path in mat_files:
            filename = os.path.basename(mat_path)
            try:
                snr_from_file = int(filename.split('_')[2].replace('dB.mat', ''))
            except:
                snr_from_file = None
            
            if snr_filter is not None and snr_from_file is not None:
                if not self._check_snr_filter(snr_from_file, snr_filter):
                    continue
            
            for rx, a, snr, s in _iter_mat_samples(mat_path):
                if count >= max_samples:
                    break
                raw_samples.append((rx, a, snr, s))
                count += 1
            if count >= max_samples:
                break

        if len(raw_samples) == 0:
            raise ValueError(f"No samples loaded! Check data directory: {data_dir}")

        # 统计SNR分布
        snr_counts = defaultdict(int)
        for _, _, snr, _ in raw_samples:
            snr_counts[int(snr)] += 1
        print(f"SNR distribution: {dict(snr_counts)}")

        if norm_stats is None:
            rx_real_stack = np.stack([rx.real for rx, _, _, _ in raw_samples], axis=0)
            rx_imag_stack = np.stack([rx.imag for rx, _, _, _ in raw_samples], axis=0)
            a_real_stack = np.stack([a.real for _, a, _, _ in raw_samples], axis=0)
            a_imag_stack = np.stack([a.imag for _, a, _, _ in raw_samples], axis=0)

            self.rx_mean = np.stack([rx_real_stack.mean(axis=0), rx_imag_stack.mean(axis=0)], axis=0).astype(np.float32)
            self.rx_std = np.stack([rx_real_stack.std(axis=0), rx_imag_stack.std(axis=0)], axis=0).astype(np.float32)
            self.rx_std = np.where(self.rx_std < 1e-6, 1.0, self.rx_std)

            self.a_mean = np.stack([a_real_stack.mean(axis=0), a_imag_stack.mean(axis=0)], axis=0).astype(np.float32)
            self.a_std = np.stack([a_real_stack.std(axis=0), a_imag_stack.std(axis=0)], axis=0).astype(np.float32)
            self.a_std = np.where(self.a_std < 1e-6, 1.0, self.a_std)
        else:
            self.rx_mean, self.rx_std, self.a_mean, self.a_std = norm_stats

        for rx, a, snr, s in raw_samples:
            rx_real = (rx.real - self.rx_mean[0]) / self.rx_std[0]
            rx_imag = (rx.imag - self.rx_mean[1]) / self.rx_std[1]
            a_real = (a.real - self.a_mean[0]) / self.a_std[0]
            a_imag = (a.imag - self.a_mean[1]) / self.a_std[1]

            rx_c = (rx_real + 1j * rx_imag).astype(np.complex64)
            a_c = (a_real + 1j * a_imag).astype(np.complex64)
            s_c = s.astype(np.complex64)
            
            self.samples.append((rx_c, a_c, np.float32(snr), s_c))

        print(f"Loaded {len(self.samples)} samples")

    def _check_snr_filter(self, snr, filter_str):
        if filter_str == "all" or filter_str is None:
            return True
        if ">=" in filter_str:
            threshold = float(filter_str.replace(">=", ""))
            return snr >= threshold
        elif "<=" in filter_str:
            threshold = float(filter_str.replace("<=", ""))
            return snr <= threshold
        elif "==" in filter_str:
            threshold = float(filter_str.replace("==", ""))
            return snr == threshold
        return True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rx_c, a_c, snr, s_c = self.samples[idx]
        return (
            torch.from_numpy(rx_c).to(torch.complex64),
            torch.from_numpy(a_c).to(torch.complex64),
            torch.tensor(snr, dtype=torch.float32),
            torch.from_numpy(s_c).to(torch.complex64)
        )

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
        bound = 1.0 / np.sqrt(2 * self.in_features)
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

class SNRAwareRegressor(nn.Module):
    def __init__(self, input_len, output_dim, snr_embedding_dim=16):
        super().__init__()
        self.snr_embedding = nn.Sequential(
            nn.Linear(1, snr_embedding_dim),
            nn.ReLU(),
            nn.Linear(snr_embedding_dim, snr_embedding_dim)
        )
        
        self.encoder = nn.Sequential(
            ComplexLinear(input_len, 384),
            ComplexBatchNorm1d(384),
            ComplexReLU(),
        )
        
        self.fusion = ComplexLinear(384, 192)
        
        self.snr_modulation = nn.Linear(snr_embedding_dim, 192)
        
        self.decoder = nn.Sequential(
            ComplexBatchNorm1d(192),
            ComplexReLU(),
            ComplexLinear(192, output_dim),
        )
    
    def forward(self, x, snr):
        feat = self.encoder(x)
        
        snr_feat = self.snr_embedding(snr.unsqueeze(1))
        
        fused = self.fusion(feat)
        
        snr_mod = self.snr_modulation(snr_feat)
        snr_mod_complex = torch.complex(snr_mod, torch.zeros_like(snr_mod))
        
        modulated = fused * (1 + 0.1 * snr_mod_complex)
        
        return self.decoder(modulated)

def criterion(pred, target):
    diff = pred - target
    return (
        0.4 * F.mse_loss(pred.real, target.real)
        + 0.4 * F.mse_loss(pred.imag, target.imag)
        + 0.15 * F.mse_loss(pred.abs(), target.abs())
        + 0.05 * torch.mean(torch.abs(diff))
    )

def train_one_epoch(model, loader, optimizer, device, use_snr=False):
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc="Training"):
        if use_snr:
            rx, a, snr, _ = batch
            rx, a, snr = rx.to(device), a.to(device), snr.to(device)
        else:
            rx, a, snr, _ = batch
            rx, a = rx.to(device), a.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_snr:
            pred = model(rx, snr)
        else:
            pred = model(rx)
        
        loss = criterion(pred, a)
        loss.backward()
        
        # 修复：手动实现复数梯度裁剪
        max_norm = 1.0
        total_norm = 0.0
        parameters = [p for p in model.parameters() if p.grad is not None]
        
        # 计算总梯度范数
        for p in parameters:
            if torch.is_complex(p.grad):
                # 复数梯度：使用实部和虚部的范数
                param_norm = torch.sqrt(
                    p.grad.real.data.norm(2) ** 2 + 
                    p.grad.imag.data.norm(2) ** 2
                )
            else:
                param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        
        # 裁剪梯度
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                if torch.is_complex(p.grad):
                    # 分别裁剪实部和虚部
                    p.grad.data = torch.complex(
                        p.grad.real.data * clip_coef,
                        p.grad.imag.data * clip_coef
                    )
                else:
                    p.grad.data.mul_(clip_coef)

        for p in parameters:
            p.grad = p.grad.resolve_conj()
        
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(loader)

def evaluate(model, loader, device, use_snr=False):
    model.eval()
    total_loss = 0.0
    snr_metrics = defaultdict(lambda: {'mse': [], 'mae': [], 'evm': []})
    
    with torch.inference_mode():
        for batch in loader:
            if use_snr:
                rx, a, snr, _ = batch
                rx, a, snr = rx.to(device), a.to(device), snr.to(device)
            else:
                rx, a, snr, _ = batch
                rx, a = rx.to(device), a.to(device)
            
            if use_snr:
                pred = model(rx, snr)
            else:
                pred = model(rx)
            
            loss = criterion(pred, a)
            total_loss += loss.item()
            
            # 按SNR分组计算指标
            for i in range(len(snr)):
                snr_val = int(snr[i].item())
                diff = pred[i] - a[i]
                mse = torch.mean(torch.abs(diff) ** 2).item()
                mae = torch.mean(torch.abs(diff)).item()
                
                ref_power = torch.mean(torch.abs(a[i]) ** 2).item()
                err_power = mse
                evm_db = 10 * np.log10(err_power / (ref_power + 1e-12))
                
                snr_metrics[snr_val]['mse'].append(mse)
                snr_metrics[snr_val]['mae'].append(mae)
                snr_metrics[snr_val]['evm'].append(evm_db)
    
    avg_loss = total_loss / len(loader)
    
    # 汇总SNR指标
    snr_summary = {}
    for snr_val, metrics in snr_metrics.items():
        snr_summary[snr_val] = {
            'mse': np.mean(metrics['mse']),
            'mae': np.mean(metrics['mae']),
            'evm_db': np.mean(metrics['evm']),
            'count': len(metrics['mse'])
        }
    
    return avg_loss, snr_summary

def plot_snr_performance(results_dict, save_path):
    """
    results_dict = {
        'Model_Name': {snr: {'mse': val, 'mae': val, 'evm_db': val}, ...},
        ...
    }
    """
    snr_values = sorted(list(next(iter(results_dict.values())).keys()))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance vs SNR', fontsize=16)
    
    metrics = ['mse', 'mae', 'evm_db']
    titles = ['MSE vs SNR', 'MAE vs SNR', 'EVM (dB) vs SNR', 'Sample Count']
    
    for idx, (metric, title) in enumerate(zip(metrics + ['count'], titles)):
        ax = axes[idx // 2, idx % 2]
        
        for model_name, snr_data in results_dict.items():
            values = [snr_data[snr][metric] for snr in snr_values]
            ax.plot(snr_values, values, marker='o', label=model_name, linewidth=2)
        
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

def compare_noise_impact(args):
    device = torch.device(args.device)
    
    results = {}
    
    # 实验1: 仅高SNR训练
    print("\n" + "="*60)
    print("Experiment 1: Training on High SNR only (>=15 dB)")
    print("="*60)
    
    train_dataset_high = WaveformDataset(DATA_DIR, max_samples=args.max_samples, snr_filter=">=15")
    train_loader_high = DataLoader(train_dataset_high, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    sample_rx, sample_a, _, _ = train_dataset_high[0]
    model_high = WaveformRegressor(sample_rx.shape[0], sample_a.shape[0]).to(device)
    optimizer_high = torch.optim.AdamW(
        model_high.parameters(), lr=args.lr, weight_decay=1e-5, foreach=False
    )
    
    for epoch in range(1, min(15, args.epochs) + 1):
        loss = train_one_epoch(model_high, train_loader_high, optimizer_high, device)
        print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    # 在所有SNR上测试
    test_dataset_all = WaveformDataset(DATA_DIR, max_samples=args.test_samples, norm_stats=train_dataset_high.get_norm_stats())
    test_loader_all = DataLoader(test_dataset_all, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    _, snr_summary_high = evaluate(model_high, test_loader_all, device)
    results['High SNR Only (>=15dB)'] = snr_summary_high
    
    # 实验2: 所有SNR训练
    print("\n" + "="*60)
    print("Experiment 2: Training on All SNR (0-25 dB)")
    print("="*60)
    
    train_dataset_all = WaveformDataset(DATA_DIR, max_samples=args.max_samples, snr_filter="all")
    train_loader_all_train = DataLoader(train_dataset_all, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    model_all = WaveformRegressor(sample_rx.shape[0], sample_a.shape[0]).to(device)
    optimizer_all = torch.optim.AdamW(
        model_all.parameters(), lr=args.lr, weight_decay=1e-5, foreach=False
    )
    
    for epoch in range(1, min(15, args.epochs) + 1):
        loss = train_one_epoch(model_all, train_loader_all_train, optimizer_all, device)
        print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    test_dataset_all2 = WaveformDataset(DATA_DIR, max_samples=args.test_samples, norm_stats=train_dataset_all.get_norm_stats())
    test_loader_all2 = DataLoader(test_dataset_all2, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    _, snr_summary_all = evaluate(model_all, test_loader_all2, device)
    results['All SNR (0-25dB)'] = snr_summary_all
    
    # 实验3: SNR感知模型
    print("\n" + "="*60)
    print("Experiment 3: SNR-Aware Model on All SNR")
    print("="*60)
    
    model_snr_aware = SNRAwareRegressor(sample_rx.shape[0], sample_a.shape[0]).to(device)
    optimizer_snr = torch.optim.AdamW(
        model_snr_aware.parameters(), lr=args.lr, weight_decay=1e-5, foreach=False
    )
    
    for epoch in range(1, min(15, args.epochs) + 1):
        loss = train_one_epoch(model_snr_aware, train_loader_all_train, optimizer_snr, device, use_snr=True)
        print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    _, snr_summary_aware = evaluate(model_snr_aware, test_loader_all2, device, use_snr=True)
    results['SNR-Aware Model'] = snr_summary_aware
    
    # 打印对比表格
    print("\n" + "="*80)
    print("COMPARISON TABLE: Noise Impact Analysis")
    print("="*80)
    print(f"{'SNR (dB)':<10} {'Metric':<10} {'High SNR Only':<18} {'All SNR':<18} {'SNR-Aware':<18}")
    print("-"*80)
    
    all_snrs = sorted(set(snr_summary_all.keys()) | set(snr_summary_high.keys()) | set(snr_summary_aware.keys()))
    
    for snr in all_snrs:
        for metric in ['mse', 'evm_db']:
            val_high = snr_summary_high.get(snr, {}).get(metric, float('nan'))
            val_all = snr_summary_all.get(snr, {}).get(metric, float('nan'))
            val_aware = snr_summary_aware.get(snr, {}).get(metric, float('nan'))
            
            print(f"{snr:<10} {metric.upper():<10} {val_high:<18.6f} {val_all:<18.6f} {val_aware:<18.6f}")
        print("-"*80)
    
    # 绘图
    plot_path = os.path.join(RESULTS_DIR, "noise_impact_comparison.png")
    plot_snr_performance(results, plot_path)
    
    return results

def curriculum_learning(args):
    device = torch.device(args.device)
    
    sample_rx, sample_a, _, _ = WaveformDataset(DATA_DIR, max_samples=10)[0]
    model = WaveformRegressor(sample_rx.shape[0].item() if hasattr(sample_rx.shape[0], 'item') else sample_rx.shape[0], 
                              sample_a.shape[0].item() if hasattr(sample_a.shape[0], 'item') else sample_a.shape[0]).to(device)
    
    stages = [
        (">=20", 8, 2e-4),
        (">=10", 10, 1e-4),
        ("all", 12, 5e-5),
    ]
    
    for stage_name, epochs, lr in stages:
        print(f"\n{'='*60}")
        print(f"Curriculum Stage: SNR {stage_name}, Epochs={epochs}, LR={lr}")
        print(f"{'='*60}")
        
        train_dataset = WaveformDataset(DATA_DIR, max_samples=args.max_samples, snr_filter=stage_name)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-5, foreach=False
        )
        
        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, device)
            print(f"Epoch {epoch}/{epochs}: Loss = {loss:.6f}")
    
    # 最终评估
    test_dataset = WaveformDataset(DATA_DIR, max_samples=args.test_samples, snr_filter="all")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    _, snr_summary = evaluate(model, test_loader, device)
    
    print("\n" + "="*60)
    print("Curriculum Learning Final Results")
    print("="*60)
    for snr, metrics in sorted(snr_summary.items()):
        print(f"SNR {snr:2d} dB: MSE={metrics['mse']:.6f}, EVM={metrics['evm_db']:.2f} dB")
    
    return {'Curriculum Learning': snr_summary}

def main():
    args = parse_args()
    
    if args.mode == "compare":
        results = compare_noise_impact(args)
    elif args.mode == "curriculum":
        results = curriculum_learning(args)
    elif args.mode == "snr_aware":
        print("SNR-aware mode included in compare mode")
        results = compare_noise_impact(args)
    
    print("\n✅ Analysis complete!")
    print(f"Results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()