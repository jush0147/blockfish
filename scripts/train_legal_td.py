#!/usr/bin/env python3
"""
Pure 100% Legal p=5 Self-Play TD(lambda) / Monte-Carlo Value Trainer (ZERO WG_ds!).
Trains BlockfishHybridResNet (V_hybrid = base_v + delta_ResNet) directly to minimize
true Legal p=5 expected remaining pieces on epsilon-explored self-play trajectories.

Usage:
    python3 scripts/train_legal_td.py \
        --data /tmp/legal_rl_v1.bin \
        --lam 0.85 --epochs 20 --batch-size 1024 \
        --out /tmp/hybrid_td_v1.pt
"""
import argparse
import gzip
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_resnet import GPUUnpacker
from train_hybrid_resnet import BlockfishHybridResNet


REC_BYTES = 60


def load_bfrl0001(path: str, lam: float = 0.85):
    t0 = time.time()
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            buf = f.read()
    else:
        with open(path, "rb") as f:
            buf = f.read()

    if len(buf) < 16 or buf[:8] != b"BFRL0001":
        raise ValueError(f"Invalid BFRL0001 header in {path}")

    n_games, n_states = struct.unpack_from("<II", buf, 8)
    raw = np.frombuffer(buf, dtype=np.uint8, offset=16, count=n_states * REC_BYTES).reshape(n_states, REC_BYTES)
    raw44 = raw[:, :44].copy()
    base_v = raw[:, 44:48].copy().view(np.float32).reshape(n_states)
    true_rem = raw[:, 48:52].copy().view(np.float32).reshape(n_states)
    game_id = raw[:, 52:56].copy().view(np.uint32).reshape(n_states)
    flags = raw[:, 58].copy()
    is_val = (flags & 1) != 0

    # Fit simple offset for base_v on training states so TD bootstrapping has zero constant bias
    tr_mask = ~is_val
    base_offset = float(np.mean(true_rem[tr_mask] - base_v[tr_mask]))
    base_cal = base_v + base_offset

    # Compute TD(lambda) targets backwards within each game
    td_target = true_rem.copy()
    if lam < 0.9999:
        # Find game boundaries
        change = np.nonzero(game_id[1:] != game_id[:-1])[0] + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [n_states]])
        for s, e in zip(starts, ends):
            g_next = true_rem[e - 1]
            td_target[e - 1] = g_next
            for k in range(e - 2, s - 1, -1):
                g_next = 1.0 + (1.0 - lam) * base_cal[k + 1] + lam * g_next
                td_target[k] = g_next

    print(
        f"Loaded {path}: {n_games} games, {n_states:,} states "
        f"(train={tr_mask.sum():,}, val={is_val.sum():,}) in {time.time()-t0:.2f}s | "
        f"base_offset={base_offset:+.2f} pcs | lam={lam:.2f}",
        flush=True,
    )
    return {
        "raw44": raw44,
        "base_v": base_v,
        "true_rem": true_rem,
        "td_target": td_target,
        "is_val": is_val,
        "base_offset": base_offset,
    }


@torch.no_grad()
def evaluate_td(model, unpacker, data, val_idx, batch_size, device, amp_dtype, zero_delta=False):
    model.eval()
    raw44 = data["raw44"]
    base_v = data["base_v"]
    true_rem = data["true_rem"]
    td_target = data["td_target"]

    se_td = 0.0
    ae_td = 0.0
    se_mc = 0.0
    ae_mc = 0.0
    dabs_sum = 0.0
    n = 0
    for s in range(0, len(val_idx), batch_size):
        idx = val_idx[s : s + batch_size]
        bv_t = torch.from_numpy(base_v[idx]).to(device, non_blocking=True)
        td_t = torch.from_numpy(td_target[idx]).to(device, non_blocking=True)
        mc_t = torch.from_numpy(true_rem[idx]).to(device, non_blocking=True)
        if zero_delta:
            dv = torch.zeros_like(bv_t)
            vb = torch.tensor(data["base_offset"], device=device)
        else:
            rt = torch.from_numpy(raw44[idx]).to(device, non_blocking=True)
            boards, gvec = unpacker(rt)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                dv = model.forward_delta(boards, gvec).float()
            vb = model.v_bias
        pred = bv_t + vb + dv
        diff_td = pred - td_t
        diff_mc = pred - mc_t
        b = len(idx)
        se_td += (diff_td * diff_td).sum().item()
        ae_td += diff_td.abs().sum().item()
        se_mc += (diff_mc * diff_mc).sum().item()
        ae_mc += diff_mc.abs().sum().item()
        dabs_sum += dv.abs().sum().item()
        n += b
    return (
        ae_td / max(n, 1),
        (se_td / max(n, 1)) ** 0.5,
        ae_mc / max(n, 1),
        (se_mc / max(n, 1)) ** 0.5,
        dabs_sum / max(n, 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to BFRL0001 .bin or .bin.gz")
    ap.add_argument("--lam", type=float, default=0.85, help="TD(lambda) parameter (1.0=pure Monte Carlo)")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-delta", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--reg-weight", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/tmp/hybrid_td_v1.pt")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Device: {device} | amp_dtype={amp_dtype}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    data = load_bfrl0001(args.data, lam=args.lam)
    all_idx = np.arange(len(data["is_val"]))
    train_idx = all_idx[~data["is_val"]]
    val_idx = all_idx[data["is_val"]]

    unpacker = GPUUnpacker().to(device)
    model = BlockfishHybridResNet(
        channels=args.channels, blocks=args.blocks, max_delta=args.max_delta, temp=0.5
    ).to(device)
    # Initialize v_bias to empirical Legal offset so epoch 0 starts perfectly calibrated!
    with torch.no_grad():
        model.v_bias.fill_(data["base_offset"])

    b_mae, b_rmse, b_mc_mae, b_mc_rmse, _ = evaluate_td(model, unpacker, data, val_idx, 2048, device, amp_dtype, zero_delta=True)
    print(
        f"BASE (spatial_bc4 + offset, delta=0) | val_td_mae={b_mae:.4f} val_td_rmse={b_rmse:.4f} | "
        f"val_mc_mae={b_mc_mae:.4f} val_mc_rmse={b_mc_rmse:.4f}",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_mc_rmse = b_mc_rmse
    rng = np.random.default_rng(args.seed)

    raw44 = data["raw44"]
    base_v = data["base_v"]
    td_target = data["td_target"]

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(train_idx)
        tr_loss = 0.0
        tr_n = 0

        for s in range(0, len(perm), args.batch_size):
            idx = perm[s : s + args.batch_size]
            rt = torch.from_numpy(raw44[idx]).to(device, non_blocking=True)
            bv_t = torch.from_numpy(base_v[idx]).to(device, non_blocking=True)
            td_t = torch.from_numpy(td_target[idx]).to(device, non_blocking=True)

            boards, gvec = unpacker(rt)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                dv = model.forward_delta(boards, gvec).float()
                pred = bv_t + model.v_bias + dv
                loss = F.smooth_l1_loss(pred, td_t, beta=2.0) + args.reg_weight * (dv * dv).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()

            b = len(idx)
            tr_loss += loss.item() * b
            tr_n += b

        sched.step()
        v_mae, v_rmse, v_mc_mae, v_mc_rmse, dabs = evaluate_td(model, unpacker, data, val_idx, 2048, device, amp_dtype)
        dt = time.time() - t0
        marker = ""
        if v_mc_rmse < best_mc_rmse:
            best_mc_rmse = v_mc_rmse
            torch.save(
                {
                    "arch": "hybrid_v1",
                    "state_dict": model.state_dict(),
                    "channels": args.channels,
                    "blocks": args.blocks,
                    "in_ch": 8,
                    "g_dim": 61,
                    "max_delta": args.max_delta,
                    "temp": 0.5,
                    "val_td_mae": v_mae,
                    "val_td_rmse": v_rmse,
                    "val_mc_mae": v_mc_mae,
                    "val_mc_rmse": v_mc_rmse,
                    "base_val_mc_rmse": b_mc_rmse,
                    "val_ce": v_mc_rmse,
                    "epoch": ep,
                },
                args.out,
            )
            marker = " *BEST*"
        print(
            f"ep {ep:2d}/{args.epochs} | lr={sched.get_last_lr()[0]:.5f} | tr_loss={tr_loss/tr_n:.4f} | "
            f"td_mae={v_mae:.3f} td_rmse={v_rmse:.3f} | mc_mae={v_mc_mae:.3f} mc_rmse={v_mc_rmse:.3f} |d|={dabs:.2f} | {dt:.1f}s{marker}",
            flush=True,
        )

    print(f"DONE! Base val_mc_rmse={b_mc_rmse:.4f} -> Best val_mc_rmse={best_mc_rmse:.4f} saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
