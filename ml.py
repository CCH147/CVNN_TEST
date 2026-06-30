import numpy as np
import os
from scipy.io import savemat

N = 34
D = N - 1
mSymBits = 2
M = 2 ** mSymBits
SNR_range = [0, 5, 10, 15, 20, 25]
samples_per_snr = 3000
Fs = 3e6
delta_F = 250
T_sym = 1 / delta_F
t = np.arange(0, T_sym, 1/Fs)
L = len(t)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Generating synthetic dataset: N={N}, M={M}, L={L}")

def qammod_gray(M):
    qam_points = np.zeros(M, dtype=np.complex64)
    for i in range(M):
        I = (i % int(np.sqrt(M))) - (np.sqrt(M) - 1) / 2
        Q = (i // int(np.sqrt(M))) - (np.sqrt(M) - 1) / 2
        qam_points[i] = I + 1j * Q
    avg_power = np.mean(np.abs(qam_points)**2)
    return qam_points / np.sqrt(avg_power)

qam_points = qammod_gray(M)

def generate_a(tx_bits, D, mSymBits, qam_points):
    numPairs = D // 2
    tx_syms = []
    for k in range(numPairs):
        idx = 0
        for b in range(mSymBits):
            idx = idx * 2 + tx_bits[k*mSymBits + b]
        tx_syms.append(qam_points[idx])

    a = np.zeros(D, dtype=np.complex64)
    for k in range(numPairs):
        a[k] = np.real(tx_syms[k]) + 0j
        a[D - k - 1] = 1j * np.imag(tx_syms[k])

    L1 = np.sum(np.abs(a)) + 1e-10
    A_boundary = L1 / 1.99
    a = a / A_boundary
    return a.astype(np.complex64)

def a_to_c(a):
    N = len(a) + 1
    D = len(a)
    ck = np.zeros(N-1, dtype=np.complex64)
    for k in range((N-1)//2):
        ck[k] = (np.sqrt(2)/2) * (a[k] + 1j * a[D-k-1])
        ck[N-k-2] = np.conj(ck[k])
    if N % 2 == 0:
        ck[N//2 - 1] = a[D//2]
    c = np.concatenate(([1], ck, [1]))
    return c

for snr in SNR_range:
    print(f"Generating SNR = {snr} dB ...")
    dataset = []
    
    for i in range(samples_per_snr):
        tx_bits = np.random.randint(0, 2, D//2 * mSymBits)
        a = generate_a(tx_bits, D, mSymBits, qam_points)
        
        c = a_to_c(a)
        
        s = np.zeros(L, dtype=np.complex64)
        for k in range(N+1):
            basis = np.exp(1j * 2 * np.pi * (k - N/2) * delta_F * t)
            s += c[k] * basis

        signal_power = np.mean(np.abs(s) ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(L) + 1j * np.random.randn(L))
        rx = s + noise

        rx_norm = rx / (np.max(np.abs(rx)) + 1e-12)

        dataset.append({
            'rx': rx_norm.astype(np.complex64),
            'a': a.astype(np.complex64),
            'snr': np.float32(snr)
        })
    
    save_path = os.path.join(DATA_DIR, f"dataset_SNR_{snr:02d}dB.mat")
    payload = {
        'dataset': np.array(dataset, dtype=object),
        'snr': np.array([snr], dtype=np.float32),
    }
    savemat(save_path, payload, do_compression=True)
    print(f"  Saved {len(dataset)} samples to {save_path}")

print("\n✅ Synthetic dataset generation complete!")
print(f"Location: {DATA_DIR}")
