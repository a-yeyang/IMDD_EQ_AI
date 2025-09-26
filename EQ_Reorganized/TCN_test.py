#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAM4 IM/DD chain + TCN equalizer (pure Python/PyTorch, single file)

Pipeline:
  Tx: random PAM4 {-3,-1,1,3} -> upsample -> RRC pulse shaping
  Channel: bandwidth-limited FIR + AWGN
  Rx: matched RRC, downsample (1 Sa/s) -> TCN equalizer (causal, dilated)
  Eval: pre/post-equalizer SER/BER (Gray mapping), training curve & hist plots

Usage (optional args):
  python tcn_pam4_equalizer.py --nsym 8000 --epochs 6 --sps 8 --snr_db 14 \
      --ch_bw_ratio 0.22 --tcn_channels 24 --tcn_blocks 5 --win 63

Requires:
  pip install numpy torch matplotlib
"""

import argparse
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ---------------------- Utilities ----------------------
PAM4_LEVELS = np.array([-3.0, -1.0, 1.0, 3.0], dtype=float)

def rrc_filter(beta: float, span: int, sps: int) -> np.ndarray:
    """
    Root Raised Cosine (RRC) filter impulse response (energy normalized).
    beta: 0..1  roll-off
    span: length in symbols
    sps:  samples per symbol
    """
    N = span * sps
    t = np.arange(-N/2, N/2 + 1, dtype=float) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        # singularities
        if abs(1 - (4 * beta * ti) ** 2) < 1e-12:
            h[i] = (beta / math.sqrt(2)) * (
                (1 + 2 / math.pi) * math.sin(math.pi / (4 * beta))
                + (1 - 2 / math.pi) * math.cos(math.pi / (4 * beta))
            )
        elif abs(ti) < 1e-12:
            h[i] = 1 - beta + (4 * beta / math.pi)
        else:
            num = math.sin(math.pi * ti * (1 - beta)) + \
                  4 * beta * ti * math.cos(math.pi * ti * (1 + beta))
            den = math.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    # energy norm (so that conv doesn't amplify power)
    h = h / np.sqrt(np.sum(h**2))
    return h

def bw_limited_fir(cutoff_ratio: float, taps: int) -> np.ndarray:
    """
    Simple low-pass FIR via sinc * Kaiser to emulate bandwidth-limited components.
    cutoff_ratio: normalized to sampling rate (Nyquist=0.5)
    taps: FIR length (odd recommended)
    """
    n = np.arange(taps) - (taps - 1)/2
    h = 2 * cutoff_ratio * np.sinc(2 * cutoff_ratio * n)
    h *= np.kaiser(taps, beta=6.0)
    h /= np.sum(h)
    return h

def gen_dataset_aligned(Nsym:int=8000, sps:int=8, rrc_beta:float=0.3, rrc_span:int=10,
                        snr_db:float=14.0, ch_bw_ratio:float=0.22, ch_taps:int=81,
                        seed:int=7):
    """
    Generate PAM4 sequence -> RRC -> bandwidth-limited channel + AWGN -> matched filter -> downsample (1 Sa/s)
    Return: (truth_symbols, rx_downsampled_normalized, mean, std)
    """
    rng = np.random.default_rng(seed)
    syms_full = rng.choice(PAM4_LEVELS, size=Nsym)

    # Upsample
    up = np.zeros(Nsym * sps)
    up[::sps] = syms_full

    # RRC pulse shaping (Tx) and matched filter (Rx uses same h_rrc)
    h_rrc = rrc_filter(rrc_beta, rrc_span, sps)
    tx = np.convolve(up, h_rrc, mode='same')

    # Channel: LPF + AWGN
    h_ch = bw_limited_fir(ch_bw_ratio, ch_taps)
    ch_out = np.convolve(tx, h_ch, mode='same')

    # AWGN at output
    sig_pow = np.mean(ch_out**2)
    snr_lin = 10**(snr_db/10)
    n0 = sig_pow / snr_lin
    noise = np.sqrt(n0) * rng.standard_normal(size=ch_out.shape)
    rx_raw = ch_out + noise

    # Matched filter & downsample
    rx_mf = np.convolve(rx_raw, h_rrc, mode='same')
    gd = (len(h_rrc) - 1)//2  # group delay samples
    start = gd
    n_avail = (len(rx_mf) - start) // sps
    n_use = min(Nsym, n_avail)

    rx_down = rx_mf[start:start + n_use*sps:sps]
    syms = syms_full[:n_use]

    # Normalize for stable training
    mean = float(np.mean(rx_down))
    std = float(np.std(rx_down) + 1e-8)
    rx_norm = (rx_down - mean)/std
    return syms.astype(np.float32), rx_norm.astype(np.float32), mean, std

def build_windows(x: np.ndarray, y: np.ndarray, W:int=63):
    """
    Build causal windows of length W that end at current symbol n (inclusive).
    x: received 1 Sa/s array
    y: true symbols
    """
    assert len(x) == len(y)
    pad = np.zeros(W-1, dtype=x.dtype)
    xpad = np.concatenate([pad, x])
    X = np.stack([xpad[n:n+W] for n in range(len(x))], axis=0)  # [N, W]
    T = y.reshape(-1, 1)  # [N, 1]
    return X, T

def hard_decision_pam4(x: np.ndarray) -> np.ndarray:
    idx = np.argmin(np.abs(x.reshape(-1,1) - PAM4_LEVELS.reshape(1,-1)), axis=1)
    return PAM4_LEVELS[idx]

def gray_bits_for_level(level: float):
    # Gray mapping: -3->00, -1->01, 1->11, 3->10
    if level == -3: return np.array([0,0])
    if level == -1: return np.array([0,1])
    if level ==  1: return np.array([1,1])
    return np.array([1,0])

def ber_from_symbols(s_true: np.ndarray, s_hat: np.ndarray) -> Tuple[float, float]:
    ser = float(np.mean(s_true != s_hat))
    bt = np.vstack([gray_bits_for_level(float(s)) for s in s_true])
    bh = np.vstack([gray_bits_for_level(float(s)) for s in s_hat])
    ber = float(np.mean(bt != bh))
    return ser, ber


# ---------------------- TCN Model ----------------------
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        padding = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.left = padding
    def forward(self, x):
        out = super().forward(x)
        return out[:, :, :-self.left]  # crop right to enforce causality

class ResidualBlock(nn.Module):
    def __init__(self, ch, kernel_size, dilation, dropout=0.05, use_bn=False):
        super().__init__()
        layers = [
            CausalConv1d(ch, ch, kernel_size, dilation=dilation),
        ]
        if use_bn: layers.append(nn.BatchNorm1d(ch))
        layers += [nn.ReLU(), nn.Dropout(dropout),
                   CausalConv1d(ch, ch, kernel_size, dilation=dilation)]
        if use_bn: layers.append(nn.BatchNorm1d(ch))
        layers += [nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return x + self.net(x)

class TCNEqualizer(nn.Module):
    def __init__(self, in_ch=1, ch=24, kernel_size=3, n_blocks=5, dropout=0.05, use_bn=False):
        super().__init__()
        self.stem = CausalConv1d(in_ch, ch, kernel_size, dilation=1)
        blocks = []
        for b in range(n_blocks):
            blocks.append(ResidualBlock(ch, kernel_size, dilation=2**b, dropout=dropout, use_bn=use_bn))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv1d(ch, 1, kernel_size=1)  # linear readout
    def forward(self, x):
        y = self.stem(x)
        y = self.blocks(y)
        y = self.head(y)
        return y


# ---------------------- Main training/eval ----------------------
def main():
    ap = argparse.ArgumentParser()
    # Data/Channel
    ap.add_argument("--nsym", type=int, default=8000, help="number of PAM4 symbols")
    ap.add_argument("--sps", type=int, default=8, help="samples per symbol for pulse shaping")
    ap.add_argument("--rrc_beta", type=float, default=0.3, help="RRC roll-off")
    ap.add_argument("--rrc_span", type=int, default=10, help="RRC span (in symbols)")
    ap.add_argument("--snr_db", type=float, default=14.0, help="channel SNR in dB")
    ap.add_argument("--ch_bw_ratio", type=float, default=0.22, help="channel cutoff (Nyquist=0.5)")
    ap.add_argument("--ch_taps", type=int, default=81, help="channel FIR taps")
    ap.add_argument("--win", type=int, default=63, help="TCN receptive window (samples)")
    ap.add_argument("--seed", type=int, default=7, help="rng seed")
    # Model/Train
    ap.add_argument("--tcn_channels", type=int, default=24, help="TCN hidden channels")
    ap.add_argument("--tcn_blocks", type=int, default=5, help="number of residual blocks (dilations 1,2,4,...)")
    ap.add_argument("--kernel_size", type=int, default=3, help="TCN kernel size")
    ap.add_argument("--dropout", type=float, default=0.05, help="dropout")
    ap.add_argument("--use_bn", action="store_true", help="use batch norm in residual blocks")
    ap.add_argument("--epochs", type=int, default=6, help="training epochs")
    ap.add_argument("--batch", type=int, default=512, help="mini-batch size")
    ap.add_argument("--lr", type=float, default=2e-3, help="learning rate")
    ap.add_argument("--weight_decay", type=float, default=1e-5, help="L2 weight decay")
    ap.add_argument("--save", type=str, default="tcn_equalizer_pam4.pt", help="checkpoint path")
    args = ap.parse_args()

    # 1) Generate dataset
    syms, rx_down, rx_mean, rx_std = gen_dataset_aligned(
        Nsym=args.nsym, sps=args.sps, rrc_beta=args.rrc_beta, rrc_span=args.rrc_span,
        snr_db=args.snr_db, ch_bw_ratio=args.ch_bw_ratio, ch_taps=args.ch_taps, seed=args.seed
    )

    # 2) Build causal windows (receptive field W)
    X, T = build_windows(rx_down, syms, W=args.win)
    N = len(X)
    i_tr = int(N*0.8)
    i_va = int(N*0.9)
    X_tr, T_tr = X[:i_tr], T[:i_tr]
    X_va, T_va = X[i_tr:i_va], T[i_tr:i_va]
    X_te, T_te = X[i_va:], T[i_va:]

    # 3) Setup device - support for CUDA, MPS (Apple Silicon) and CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Metal Performance Shaders)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    to_tensor = lambda a: torch.from_numpy(a).to(device)

    X_tr_t = to_tensor(X_tr).float().unsqueeze(1)
    T_tr_t = to_tensor(T_tr).float().unsqueeze(-1)
    X_va_t = to_tensor(X_va).float().unsqueeze(1)
    T_va_t = to_tensor(T_va).float().unsqueeze(-1)
    X_te_t = to_tensor(X_te).float().unsqueeze(1)
    T_te_t = to_tensor(T_te).float().unsqueeze(-1)

    # 4) Define model & optimizer
    model = TCNEqualizer(
        in_ch=1, ch=args.tcn_channels, kernel_size=args.kernel_size,
        n_blocks=args.tcn_blocks, dropout=args.dropout, use_bn=args.use_bn
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.MSELoss()

    def iterate_minibatches(X, T, batch=512, shuffle=True):
        N = len(X)
        idx = np.arange(N)
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, N, batch):
            j = idx[i:i+batch]
            yield X[j], T[j]

    # 5) Train
    train_losses, val_losses = [], []
    model.train()
    for ep in range(args.epochs):
        ep_loss = 0.0
        num_batches = 0
        for xb, yb in iterate_minibatches(X_tr_t, T_tr_t, batch=args.batch, shuffle=True):
            opt.zero_grad()
            pred = model(xb)                       # [B,1,W]
            loss = crit(pred[:, :, -1:], yb)       # predict current symbol (causal)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ep_loss += float(loss.item())
            num_batches += 1
        ep_loss /= num_batches  # Average loss per batch
        
        # validation
        model.eval()
        with torch.no_grad():
            pred_va = model(X_va_t)
            val_loss = crit(pred_va[:, :, -1:], T_va_t).item()
        model.train()
        train_losses.append(ep_loss)
        val_losses.append(val_loss)
        print(f"Epoch {ep+1:02d}/{args.epochs}: train_loss={ep_loss:.4e}  val_mse={val_loss:.4e}")

    # 6) Inference & metrics
    model.eval()
    with torch.no_grad():
        pred_te = model(X_te_t)[:, :, -1:].squeeze().cpu().numpy()

    rx_te = rx_down[i_va:]
    pre_dec = hard_decision_pam4(rx_te)
    post_dec = hard_decision_pam4(pred_te)

    pre_ser, pre_ber = ber_from_symbols(T_te.squeeze(), pre_dec)
    post_ser, post_ber = ber_from_symbols(T_te.squeeze(), post_dec)

    print("\n==== Results ====")
    print(f"Pre-equalizer  SER: {pre_ser:.4e}, BER: {pre_ber:.4e}")
    print(f"Post-equalizer SER: {post_ser:.4e}, BER: {post_ber:.4e}")

    # 7) Plots
    plt.figure()
    plt.plot(np.arange(1, len(train_losses)+1), train_losses, label='train')
    plt.plot(np.arange(1, len(val_losses)+1), val_losses, label='val')
    plt.xlabel('Epoch'); plt.ylabel('MSE loss'); plt.title('TCN equalizer training'); plt.legend()
    plt.tight_layout()

    plt.figure()
    K = min(3000, len(rx_te))
    plt.scatter(rx_te[:K], np.zeros(K), s=4, alpha=0.5, label='pre-eq')
    plt.scatter(pred_te[:K], np.ones(K), s=4, alpha=0.5, label='post-eq')
    plt.yticks([0,1], ['pre','post'])
    plt.xlabel('Sample value'); plt.title('Received vs. TCN outputs (slice)'); plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.hist(rx_te, bins=60, alpha=0.6, density=True, label='pre-eq')
    plt.hist(pred_te, bins=60, alpha=0.6, density=True, label='post-eq')
    plt.xlabel('Sample value'); plt.ylabel('PDF'); plt.title('Distributions: pre- vs post-equalization'); plt.legend()
    plt.tight_layout()
    plt.show()

    # 8) Save checkpoint (weights + config + rx normalization)
    torch.save({
        "model_state_dict": model.state_dict(),
        "params": {
            "in_ch": 1, "ch": args.tcn_channels, "kernel_size": args.kernel_size,
            "n_blocks": args.tcn_blocks, "dropout": args.dropout, "W": args.win,
        },
        "rx_norm": {"mean": float(rx_mean), "std": float(rx_std)},
    }, args.save)
    print(f"Checkpoint saved to: {args.save}")


if __name__ == "__main__":
    main()
