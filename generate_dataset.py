# generate_dataset.py
"""
Dataset generation script for Hermitian-symmetric OFDM / multicarrier CVNN training.

This version fixes the most important issue in the previous generator:

    OLD a_to_c():
        ck[k] = sqrt(2)/2 * (a[k] + 1j * a[D-k-1])

    But generate_a() already stores the imaginary component as

        a[D-k-1] = 1j * imag(symbol)

    Therefore multiplying by another 1j gives

        1j * (1j * imag) = -imag

    which collapses the QPSK symbol into roughly Re(symbol) - Im(symbol),
    causing ambiguity and making correct SER recovery impossible.

    FIXED a_to_c():
        ck[k] = sqrt(2)/2 * (a[k] + a[D-k-1])

    Since:
        a[k]       = Re(symbol)
        a[D-k-1]   = 1j * Im(symbol)

    then:
        a[k] + a[D-k-1] = Re(symbol) + j Im(symbol)

    which correctly reconstructs the original complex QPSK symbol.

Signal model
------------
N  = 20 subcarrier index span, coefficients c[0..20] -> 21 coefficients
D  = N - 1 = 19 target vector dimension
M  = 4 QPSK
Δf = 250 Hz
T_sym = 1 / Δf = 4 ms
Fs = 3 MHz
L  = 12000 samples per symbol

Target vector a
---------------
a[0..8]    : real parts of 9 QPSK symbols
a[9]       : structural zero
a[10..18] : imaginary parts, stored as pure imaginary values
||a||_1    : normalized to 1.99

Hermitian subcarrier mapping
----------------------------
c[0]   = 1
c[20]  = 1
c[1..9] and c[11..19] are conjugate pairs
c[10]  = a[9] = 0

Output
------
One .mat file per SNR:
    data1/dataset_SNR_XXdB.mat

Each sample contains:
    rx      : noisy received waveform, complex64, shape [L]
    s       : clean transmitted waveform, complex64, shape [L]
    a       : structured target vector, complex64, shape [D]
    c       : subcarrier coefficients, complex64, shape [N+1]
    bits    : original bits, uint8, shape [18]
    symbols : QPSK symbols before target-vector split, complex64, shape [9]
    snr     : SNR label, float32
"""

import os
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import savemat


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SEED = 42
DEFAULT_N = 20
DEFAULT_MSYM_BITS = 2
DEFAULT_SNR_RANGE = [0, 5, 10, 15, 20, 25]
DEFAULT_SAMPLES_PER_SNR = 3000
DEFAULT_FS = 3e6
DEFAULT_DELTA_F = 250.0
DEFAULT_L1_TARGET = 1.99


# =============================================================================
# Argument parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Generate Hermitian-symmetric OFDM dataset for CVNN training."
    )

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--N", type=int, default=DEFAULT_N)
    parser.add_argument("--mSymBits", type=int, default=DEFAULT_MSYM_BITS)
    parser.add_argument("--samples-per-snr", type=int, default=DEFAULT_SAMPLES_PER_SNR)
    parser.add_argument(
        "--snr-range",
        type=str,
        default="0,5,10,15,20,25",
        help="Comma-separated SNR values in dB.",
    )
    parser.add_argument("--Fs", type=float, default=DEFAULT_FS)
    parser.add_argument("--delta-F", type=float, default=DEFAULT_DELTA_F)
    parser.add_argument("--l1-target", type=float, default=DEFAULT_L1_TARGET)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(base_dir / "data1"),
        help="Output directory for .mat dataset files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing dataset files.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run a small consistency check before full generation.",
    )

    return parser.parse_args()


# =============================================================================
# QPSK / QAM constellation
# =============================================================================

def qammod_gray(M: int) -> np.ndarray:
    """
    Return a normalized square-QAM constellation.

    For M=4 this gives QPSK-like constellation:
        (-1-j), (1-j), (-1+j), (1+j), normalized to unit average power.

    Note:
    The previous code called this Gray-coded square QAM. The index ordering is kept
    compatible with the existing bit-to-index rule.
    """
    sqM = int(np.sqrt(M))
    if sqM * sqM != M:
        raise ValueError(f"M={M} is not a square QAM size.")

    qam_points = np.zeros(M, dtype=np.complex64)

    for i in range(M):
        I = (i % sqM) - (sqM - 1) / 2
        Q = (i // sqM) - (sqM - 1) / 2
        qam_points[i] = I + 1j * Q

    avg_power = np.mean(np.abs(qam_points) ** 2)
    return (qam_points / np.sqrt(avg_power)).astype(np.complex64)


def bits_to_symbols(
    tx_bits: np.ndarray,
    mSymBits: int,
    qam_points: np.ndarray,
) -> np.ndarray:
    """
    Map bit stream to QAM/QPSK symbols.

    The bit-to-index rule is kept identical to the previous generator:
        idx = idx * 2 + bit
    """
    num_syms = len(tx_bits) // mSymBits
    symbols = np.zeros(num_syms, dtype=np.complex64)

    for k in range(num_syms):
        idx = 0
        for b in range(mSymBits):
            idx = idx * 2 + int(tx_bits[k * mSymBits + b])
        symbols[k] = qam_points[idx]

    return symbols


# =============================================================================
# Target vector generation
# =============================================================================

def generate_a_from_symbols(
    symbols: np.ndarray,
    D: int,
    l1_target: float = DEFAULT_L1_TARGET,
) -> np.ndarray:
    """
    Generate structured target vector a from QPSK symbols.

    Layout for D=19:
        a[0..8]    = Re(symbol_0..symbol_8)
        a[9]       = 0
        a[10..18]  = j * Im(symbol_8..symbol_0)

    This mirrored layout is kept for compatibility with the existing training
    script and SER calculation.
    """
    num_pairs = D // 2

    if len(symbols) != num_pairs:
        raise ValueError(
            f"Expected {num_pairs} symbols for D={D}, got {len(symbols)}."
        )

    a = np.zeros(D, dtype=np.complex64)

    for k in range(num_pairs):
        a[k] = np.real(symbols[k]) + 0j
        a[D - k - 1] = 1j * np.imag(symbols[k])

    l1 = np.sum(np.abs(a)) + 1e-12
    a = a * (float(l1_target) / l1)

    return a.astype(np.complex64)


def generate_a(
    tx_bits: np.ndarray,
    D: int,
    mSymBits: int,
    qam_points: np.ndarray,
    l1_target: float = DEFAULT_L1_TARGET,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate target vector a and return the original QPSK symbols.

    Returns
    -------
    a : np.ndarray, complex64, shape [D]
    symbols : np.ndarray, complex64, shape [D//2]
    """
    symbols = bits_to_symbols(tx_bits, mSymBits, qam_points)
    a = generate_a_from_symbols(symbols, D, l1_target)
    return a, symbols


# =============================================================================
# Corrected Hermitian mapping
# =============================================================================

def a_to_c(a: np.ndarray) -> np.ndarray:
    """
    Map target vector a (length D=19) to subcarrier coefficients c (length N+1=21).

    Correct mapping:
        symbol_k = a[k] + a[D-k-1]

    because:
        a[k]       = Re(symbol_k)
        a[D-k-1]   = j * Im(symbol_k)

    Therefore:
        a[k] + a[D-k-1] = Re(symbol_k) + j Im(symbol_k)

    Hermitian construction:
        c[0]  = 1
        c[N]  = 1
        c[k] and c[N-k] are conjugate pairs
        c[N/2] = a[D/2] = 0
    """
    D = len(a)
    N = D + 1

    ck = np.zeros(N - 1, dtype=np.complex64)
    num_pairs = D // 2

    for k in range(num_pairs):
        # FIXED: no extra 1j here.
        pos = (np.sqrt(2.0) / 2.0) * (a[k] + a[D - k - 1])

        ck[k] = pos
        ck[N - k - 2] = np.conj(pos)

    if N % 2 == 0:
        ck[N // 2 - 1] = a[D // 2]  # structural zero

    c = np.concatenate(
        (
            np.array([1 + 0j], dtype=np.complex64),
            ck.astype(np.complex64),
            np.array([1 + 0j], dtype=np.complex64),
        )
    )

    return c.astype(np.complex64)


def check_hermitian(c: np.ndarray, atol: float = 1e-5) -> float:
    """
    Return max Hermitian residual:
        max |c[k] - conj(c[N-k])|
    """
    N = len(c) - 1
    residuals = []
    for k in range(N + 1):
        residuals.append(abs(c[k] - np.conj(c[N - k])))
    return float(np.max(residuals))


# =============================================================================
# Signal synthesis
# =============================================================================

def synthesise_signal(
    c: np.ndarray,
    N: int,
    delta_F: float,
    t: np.ndarray,
) -> np.ndarray:
    """
    Synthesize multicarrier signal:

        s(t) = sum_{k=0}^{N} c[k] exp(j 2π (k - N/2) Δf t)
    """
    s = np.zeros(len(t), dtype=np.complex64)

    for k in range(N + 1):
        basis = np.exp(1j * 2 * np.pi * (k - N / 2) * delta_F * t)
        s += c[k] * basis.astype(np.complex64)

    return s.astype(np.complex64)


def add_awgn(
    s: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Add complex AWGN according to measured signal power.

    Returns:
        rx, noise, signal_power, noise_power
    """
    signal_power = float(np.mean(np.abs(s) ** 2))
    noise_power = signal_power / (10 ** (snr_db / 10.0))

    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(len(s)) + 1j * rng.standard_normal(len(s))
    )

    noise = noise.astype(np.complex64)
    rx = (s + noise).astype(np.complex64)

    return rx, noise, signal_power, noise_power


def rms_normalize_pair(
    rx: np.ndarray,
    s: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Joint RMS normalization using rx RMS.

    This keeps rx and s on the same scale.
    """
    rx_rms = float(np.sqrt(np.mean(np.abs(rx) ** 2)))
    scale = rx_rms + 1e-12

    rx_norm = (rx / scale).astype(np.complex64)
    s_norm = (s / scale).astype(np.complex64)

    return rx_norm, s_norm, rx_rms


# =============================================================================
# Consistency checks
# =============================================================================

def run_self_check(
    N: int,
    D: int,
    mSymBits: int,
    qam_points: np.ndarray,
    delta_F: float,
    t: np.ndarray,
    l1_target: float,
) -> None:
    """
    Run quick sanity checks.

    This catches the old double-j mapping bug.
    """
    rng = np.random.default_rng(12345)
    bits = rng.integers(0, 2, D // 2 * mSymBits, dtype=np.uint8)

    a, symbols = generate_a(bits, D, mSymBits, qam_points, l1_target)
    c = a_to_c(a)
    s = synthesise_signal(c, N, delta_F, t)

    l1 = np.sum(np.abs(a))
    herm_res = check_hermitian(c)
    max_imag_s = np.max(np.abs(np.imag(s)))

    # Recover the positive-side symbols from c.
    recovered = []
    for k in range(D // 2):
        recovered.append(c[k + 1] / (np.sqrt(2.0) / 2.0))
    recovered = np.asarray(recovered)

    # Because a is L1-normalized, symbols are scaled by the same factor.
    a_symbols = np.asarray([a[k] + a[D - k - 1] for k in range(D // 2)])
    rec_err = np.max(np.abs(recovered - a_symbols))

    print("=" * 70)
    print("Self-check")
    print("=" * 70)
    print(f"||a||_1 target             : {l1_target}")
    print(f"||a||_1 actual             : {l1:.8f}")
    print(f"Hermitian residual          : {herm_res:.3e}")
    print(f"max |Imag(s)|               : {max_imag_s:.3e}")
    print(f"symbol reconstruction error : {rec_err:.3e}")
    print("=" * 70)

    if abs(l1 - l1_target) > 1e-4:
        raise RuntimeError("Self-check failed: L1 normalization mismatch.")
    if herm_res > 1e-4:
        raise RuntimeError("Self-check failed: Hermitian symmetry mismatch.")
    if rec_err > 1e-5:
        raise RuntimeError("Self-check failed: a_to_c mapping mismatch.")

    print("Self-check passed.\n")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    seed = int(args.seed)
    N = int(args.N)
    D = N - 1
    mSymBits = int(args.mSymBits)
    M = 2 ** mSymBits

    snr_range = [int(x.strip()) for x in args.snr_range.split(",") if x.strip()]
    samples_per_snr = int(args.samples_per_snr)

    Fs = float(args.Fs)
    delta_F = float(args.delta_F)
    T_sym = 1.0 / delta_F
    t = np.arange(0, T_sym, 1.0 / Fs)
    L = len(t)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qam_points = qammod_gray(M)

    print("=" * 70)
    print("Hermitian-Symmetric OFDM Dataset Generator")
    print("=" * 70)
    print(f"Seed                 : {seed}")
    print(f"N                    : {N}")
    print(f"D                    : {D}")
    print(f"M                    : {M}")
    print(f"mSymBits             : {mSymBits}")
    print(f"SNR range            : {snr_range}")
    print(f"Samples per SNR      : {samples_per_snr}")
    print(f"Fs                   : {Fs}")
    print(f"delta_F              : {delta_F}")
    print(f"T_sym                : {T_sym}")
    print(f"L                    : {L}")
    print(f"L1 target            : {args.l1_target}")
    print(f"Output dir           : {output_dir}")
    print("=" * 70)
    print("Important fix:")
    print("  a_to_c now uses a[k] + a[D-k-1], not a[k] + 1j*a[D-k-1].")
    print("=" * 70)

    np.random.seed(seed)

    if args.self_check:
        run_self_check(
            N=N,
            D=D,
            mSymBits=mSymBits,
            qam_points=qam_points,
            delta_F=delta_F,
            t=t,
            l1_target=args.l1_target,
        )

    for snr_idx, snr in enumerate(snr_range):
        snr_seed = seed * 1000 + snr_idx
        rng = np.random.default_rng(snr_seed)

        save_path = output_dir / f"dataset_SNR_{snr:02d}dB.mat"

        if save_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{save_path} already exists. Use --overwrite to replace it."
            )

        print(
            f"[{snr_idx + 1}/{len(snr_range)}] "
            f"SNR={snr:>2d} dB, seed={snr_seed} ...",
            end=" ",
            flush=True,
        )

        dataset = []

        # Statistics for debug / verification.
        max_herm_res = 0.0
        max_imag_s = 0.0
        l1_errors = []

        for _ in range(samples_per_snr):
            # Bits and target vector.
            tx_bits = rng.integers(0, 2, D // 2 * mSymBits, dtype=np.uint8)
            a, symbols = generate_a(
                tx_bits=tx_bits,
                D=D,
                mSymBits=mSymBits,
                qam_points=qam_points,
                l1_target=args.l1_target,
            )

            # Subcarrier coefficients and waveform.
            c = a_to_c(a)
            s = synthesise_signal(c, N, delta_F, t)

            # AWGN.
            rx, noise, signal_power, noise_power = add_awgn(s, snr, rng)

            # RMS normalization.
            rx_norm, s_norm, rx_rms = rms_normalize_pair(rx, s)

            # Debug statistics.
            max_herm_res = max(max_herm_res, check_hermitian(c))
            max_imag_s = max(max_imag_s, float(np.max(np.abs(np.imag(s)))))
            l1_errors.append(abs(np.sum(np.abs(a)) - args.l1_target))

            dataset.append(
                {
                    "rx": rx_norm.astype(np.complex64),
                    "s": s_norm.astype(np.complex64),
                    "a": a.astype(np.complex64),
                    "c": c.astype(np.complex64),
                    "bits": tx_bits.astype(np.uint8),
                    "symbols": symbols.astype(np.complex64),
                    "snr": np.float32(snr),
                    "rx_rms": np.float32(rx_rms),
                    "signal_power": np.float32(signal_power),
                    "noise_power": np.float32(noise_power),
                }
            )

        savemat(
            save_path,
            {
                "dataset": np.array(dataset, dtype=object),
                "snr": np.array([snr], dtype=np.float32),
                "seed": np.array([snr_seed], dtype=np.int64),
                "N": np.array([N], dtype=np.int32),
                "D": np.array([D], dtype=np.int32),
                "M": np.array([M], dtype=np.int32),
                "mSymBits": np.array([mSymBits], dtype=np.int32),
                "L": np.array([L], dtype=np.int32),
                "Fs": np.array([Fs], dtype=np.float64),
                "delta_F": np.array([delta_F], dtype=np.float64),
                "T_sym": np.array([T_sym], dtype=np.float64),
                "l1_target": np.array([args.l1_target], dtype=np.float64),
                "mapping_version": np.array(["fixed_no_double_j"], dtype=object),
            },
            do_compression=True,
        )

        print(
            f"saved {len(dataset)} samples -> {save_path} | "
            f"max Hermitian residual={max_herm_res:.2e}, "
            f"max |Imag(s)|={max_imag_s:.2e}, "
            f"max L1 error={max(l1_errors):.2e}"
        )

    print()
    print("Dataset generation complete.")
    print(f"Total samples: {samples_per_snr * len(snr_range)}")
    print(f"Location     : {output_dir}")


if __name__ == "__main__":
    main()
