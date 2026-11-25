#!/usr/bin/env python3
"""
Train a FiLM-conditioned 2D Convolutional ResNet afterstate evaluator on BFDS0001 datasets.
Designed for fast GPU training on Google Colab (L4 / T4 / A100).

Usage:
    python3 scripts/train_resnet.py \
        --data /tmp/wg_ds_v1_400g.bin.gz \
        --channels 64 --blocks 6 --epochs 25 --batch-size 64 \
        --out models/resnet_v1.pt
"""
import argparse
import gzip
import math
import os
import struct
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MOVE_BYTES = 44
DEC_HDR_BYTES = 12


def load_bfds0001(path: str):
    """Parse BFDS0001 (.bin or .bin.gz) into compact numpy arrays."""
    t0 = time.time()
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            buf = f.read()
    else:
        with open(path, "rb") as f:
            buf = f.read()

    if len(buf) < 16 or buf[:8] != b"BFDS0001":
        raise ValueError(f"Invalid BFDS0001 header in {path}")

    n_games, n_decs = struct.unpack_from("<II", buf, 8)
    print(f"Loading {path}: {n_games} games, {n_decs} decisions ({len(buf)/1e6:.1f} MB raw)...")

    # First pass: index decision offsets & total moves
    offsets = np.empty(n_decs + 1, dtype=np.int64)
    targets = np.empty(n_decs, dtype=np.int64)
    rem_pcs = np.empty(n_decs, dtype=np.float32)
    is_val = np.empty(n_decs, dtype=np.bool_)
    n_moves_arr = np.empty(n_decs, dtype=np.int32)

    pos = 16
    total_moves = 0
    for i in range(n_decs):
        _seed, _step, rp, nm, ti, fl, _res = struct.unpack_from("<IHHBBBB", buf, pos)
        pos += DEC_HDR_BYTES
        offsets[i] = total_moves
        targets[i] = ti
        rem_pcs[i] = float(rp)
        is_val[i] = (fl & 1) != 0
        n_moves_arr[i] = nm
        total_moves += nm
        pos += nm * MOVE_BYTES
    offsets[n_decs] = total_moves

    # Second pass: copy all MoveRec structs into a contiguous (total_moves, 44) uint8 array
    moves_raw = np.empty((total_moves, MOVE_BYTES), dtype=np.uint8)
    pos = 16
    mv_idx = 0
    mv_view = memoryview(buf)
    for i in range(n_decs):
        nm = int(n_moves_arr[i])
        pos += DEC_HDR_BYTES
        nbytes = nm * MOVE_BYTES
        moves_raw[mv_idx : mv_idx + nm] = np.frombuffer(
            mv_view[pos : pos + nbytes], dtype=np.uint8
        ).reshape(nm, MOVE_BYTES)
        mv_idx += nm
        pos += nbytes

    print(
        f"Loaded {total_moves:,} moves in {time.time()-t0:.2f}s "
        f"(train_decs={(~is_val).sum():,}, val_decs={is_val.sum():,})"
    )
    return {
        "moves_raw": moves_raw,
        "offsets": offsets,
        "targets": targets,
        "rem_pcs": rem_pcs,
        "is_val": is_val,
    }


BIT_SHIFTS = np.arange(10, dtype=np.uint16)[None, None, :]  # (1, 1, 10)
ROW_IDX = np.arange(16, dtype=np.uint8)[None, :]            # (1, 16)


def unpack_moves_batch(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert (N, 44) raw MoveRec bytes into:
      boards: (N, 8, 16, 10) float32
      globals_: (N, 61) float32
    """
    N = raw.shape[0]
    rows_u16 = raw[:, :32].view(np.uint16).reshape(N, 16)  # (N, 16)
    occ = ((rows_u16[:, :, None] >> BIT_SHIFTS) & 1).astype(np.float32)  # (N, 16, 10)

    board_rows = raw[:, 32]
    garbage_rows = raw[:, 33]
    rem_g = raw[:, 34]
    held = raw[:, 35]
    next5 = raw[:, 36:41]
    bag_mask = raw[:, 41]
    lines_cleared = raw[:, 42]
    ds_cleared = raw[:, 43]

    is_g_row = (ROW_IDX < garbage_rows[:, None])[:, :, None]  # (N, 16, 1)
    g_mask = occ * is_g_row
    s_mask = occ * (~is_g_row)
    hole_mask = (1.0 - occ) * is_g_row

    # Covered empty cells: empty cell with at least one occupied cell strictly above it
    # Reverse cumsum along height axis (15 down to 0)
    above_occ = np.cumsum(occ[:, ::-1, :], axis=1)[:, ::-1, :] - occ
    covered = ((1.0 - occ) * (above_occ > 0.5)).astype(np.float32)

    # Top garbage hole column (row = garbage_rows - 1) and 2nd garbage hole column (row = garbage_rows - 2)
    h1_col = np.zeros((N, 16, 10), dtype=np.float32)
    h2_col = np.zeros((N, 16, 10), dtype=np.float32)
    has_g1 = garbage_rows >= 1
    if np.any(has_g1):
        idx1 = np.nonzero(has_g1)[0]
        r1 = np.clip(garbage_rows[idx1].astype(np.int64) - 1, 0, 15)
        h1_bits = 1.0 - occ[idx1, r1, :]  # (K, 10)
        h1_col[idx1] = h1_bits[:, None, :]
    has_g2 = garbage_rows >= 2
    if np.any(has_g2):
        idx2 = np.nonzero(has_g2)[0]
        r2 = np.clip(garbage_rows[idx2].astype(np.int64) - 2, 0, 15)
        h2_bits = 1.0 - occ[idx2, r2, :]  # (K, 10)
        h2_col[idx2] = h2_bits[:, None, :]

    # Row fill fraction broadcast across width
    row_fill = (occ.sum(axis=2, keepdims=True) / 10.0) * np.ones((1, 1, 10), dtype=np.float32)

    boards = np.stack(
        [occ, g_mask, s_mask, hole_mask, covered, h1_col, h2_col, row_fill],
        axis=1,
    )  # (N, 8, 16, 10)

    # Global vector (N, 61): held(8) + next5(40) + bag_mask(7) + scalars(6)
    gvec = np.zeros((N, 61), dtype=np.float32)
    ar = np.arange(N)
    gvec[ar, np.clip(held, 0, 7)] = 1.0
    for k in range(5):
        gvec[ar, 8 + 8 * k + np.clip(next5[:, k], 0, 7)] = 1.0
    for p in range(7):
        gvec[:, 48 + p] = ((bag_mask >> p) & 1).astype(np.float32)
    gvec[:, 55] = rem_g.astype(np.float32) / 100.0
    gvec[:, 56] = garbage_rows.astype(np.float32) / 10.0
    gvec[:, 57] = board_rows.astype(np.float32) / 20.0
    gvec[:, 58] = lines_cleared.astype(np.float32) / 4.0
    gvec[:, 59] = ds_cleared.astype(np.float32) / 4.0
    gvec[:, 60] = (rem_g <= 10).astype(np.float32)

    return boards, gvec


class FiLMResBlock(nn.Module):
    def __init__(self, channels: int, se_ratio: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.film = nn.Linear(channels, channels * 2)
        se_mid = max(channels // se_ratio, 16)
        self.se = nn.Sequential(
            nn.Linear(channels, se_mid),
            nn.SiLU(inplace=True),
            nn.Linear(se_mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, g_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        gamma, beta = self.film(g_emb).chunk(2, dim=-1)
        h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        # Squeeze-and-Excitation
        se_w = self.se(h.mean(dim=(2, 3)))[:, :, None, None]
        h = h * se_w
        return F.silu(x + h, inplace=True)


class BlockfishResNet(nn.Module):
    def __init__(self, in_ch: int = 8, g_dim: int = 61, channels: int = 64, blocks: int = 6):
        super().__init__()
        self.in_ch = in_ch
        self.g_dim = g_dim
        self.channels = channels
        self.blocks_n = blocks

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.g_enc = nn.Sequential(
            nn.Linear(g_dim, channels),
            nn.SiLU(inplace=True),
            nn.Linear(channels, channels),
            nn.SiLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList([FiLMResBlock(channels) for _ in range(blocks)])

        head_in = channels * 4
        self.policy_head = nn.Sequential(
            nn.Linear(head_in, channels),
            nn.SiLU(inplace=True),
            nn.Linear(channels, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(head_in, channels),
            nn.SiLU(inplace=True),
            nn.Linear(channels, 1),
        )

    def forward(self, boards: torch.Tensor, gvec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g_emb = self.g_enc(gvec)
        x = self.stem(boards)
        for blk in self.res_blocks:
            x = blk(x, g_emb)
        p_mean = x.mean(dim=(2, 3))
        p_max = x.amax(dim=(2, 3))
        p_bot4 = x[:, :, :4, :].mean(dim=(2, 3))
        feat = torch.cat([p_mean, p_max, p_bot4, g_emb], dim=-1)
        logit = self.policy_head(feat).squeeze(-1)
        val = self.value_head(feat).squeeze(-1)
        return logit, val


def make_batch(data, dec_indices: np.ndarray):
    offsets = data["offsets"]
    moves_raw = data["moves_raw"]
    counts = offsets[dec_indices + 1] - offsets[dec_indices]
    total_m = int(counts.sum())
    chunks = [moves_raw[offsets[i] : offsets[i + 1]] for i in dec_indices]
    raw_cat = np.concatenate(chunks, axis=0)
    boards, gvec = unpack_moves_batch(raw_cat)
    targets = data["targets"][dec_indices]
    rem_pcs = data["rem_pcs"][dec_indices]
    return boards, gvec, counts, targets, rem_pcs


def compute_decision_losses(
    logits: torch.Tensor,
    vals: torch.Tensor,
    counts: np.ndarray,
    targets: torch.Tensor,
    rem_pcs: torch.Tensor,
    val_weight: float,
):
    # Variable-length softmax CE over each decision's legal moves
    splits_l = torch.split(logits, counts.tolist())
    splits_v = torch.split(vals, counts.tolist())
    ce_sum = 0.0
    correct1 = 0
    correct3 = 0
    star_vals = []
    for sl, sv, ti in zip(splits_l, splits_v, targets.tolist()):
        log_p = F.log_softmax(sl, dim=0)
        ce_sum = ce_sum - log_p[ti]
        top3 = torch.topk(sl, k=min(3, sl.numel())).indices
        if top3[0].item() == ti:
            correct1 += 1
        if ti in top3.tolist():
            correct3 += 1
        star_vals.append(sv[ti])
    B = len(counts)
    ce_loss = ce_sum / B
    pred_star_v = torch.stack(star_vals)
    v_loss = F.smooth_l1_loss(pred_star_v, rem_pcs, beta=2.0)
    v_mae = (pred_star_v - rem_pcs).abs().mean().item()
    total_loss = ce_loss + val_weight * v_loss
    return total_loss, ce_loss.item(), v_mae, correct1, correct3


@torch.no_grad()
def evaluate(model, data, val_indices: np.ndarray, batch_size: int, device, amp_dtype):
    model.eval()
    total_ce = 0.0
    total_vmae = 0.0
    total_top1 = 0
    total_top3 = 0
    total_n = 0
    for start in range(0, len(val_indices), batch_size):
        idx = val_indices[start : start + batch_size]
        boards_np, gvec_np, counts, targets_np, rem_np = make_batch(data, idx)
        boards = torch.from_numpy(boards_np).to(device, non_blocking=True)
        gvec = torch.from_numpy(gvec_np).to(device, non_blocking=True)
        targets = torch.from_numpy(targets_np).to(device, non_blocking=True)
        rem_t = torch.from_numpy(rem_np).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits, vals = model(boards, gvec)
            _, ce, vmae, c1, c3 = compute_decision_losses(
                logits.float(), vals.float(), counts, targets, rem_t, 0.02
            )
        b = len(idx)
        total_ce += ce * b
        total_vmae += vmae * b
        total_top1 += c1
        total_top3 += c3
        total_n += b
    return (
        total_ce / max(total_n, 1),
        total_vmae / max(total_n, 1),
        100.0 * total_top1 / max(total_n, 1),
        100.0 * total_top3 / max(total_n, 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to BFDS0001 .bin or .bin.gz")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val-weight", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models/resnet_v1.pt")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"Device: {device} | amp_dtype={amp_dtype}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data = load_bfds0001(args.data)
    all_idx = np.arange(len(data["targets"]))
    train_idx = all_idx[~data["is_val"]]
    val_idx = all_idx[data["is_val"]]

    model = BlockfishResNet(channels=args.channels, blocks=args.blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BlockfishResNet(C={args.channels}, B={args.blocks}) | params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_val_ce = float("inf")
    rng = np.random.default_rng(args.seed)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(train_idx)
        tr_ce = 0.0
        tr_vmae = 0.0
        tr_c1 = 0
        tr_n = 0

        for start in range(0, len(perm), args.batch_size):
            idx = perm[start : start + args.batch_size]
            boards_np, gvec_np, counts, targets_np, rem_np = make_batch(data, idx)
            boards = torch.from_numpy(boards_np).to(device, non_blocking=True)
            gvec = torch.from_numpy(gvec_np).to(device, non_blocking=True)
            targets = torch.from_numpy(targets_np).to(device, non_blocking=True)
            rem_t = torch.from_numpy(rem_np).to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits, vals = model(boards, gvec)
                loss, ce, vmae, c1, _ = compute_decision_losses(
                    logits.float(), vals.float(), counts, targets, rem_t, args.val_weight
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()

            b = len(idx)
            tr_ce += ce * b
            tr_vmae += vmae * b
            tr_c1 += c1
            tr_n += b

        sched.step()
        val_ce, val_vmae, val_top1, val_top3 = evaluate(
            model, data, val_idx, args.batch_size * 2, device, amp_dtype
        )
        dt = time.time() - t0
        marker = ""
        if val_ce < best_val_ce:
            best_val_ce = val_ce
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "channels": args.channels,
                    "blocks": args.blocks,
                    "in_ch": 8,
                    "g_dim": 61,
                    "val_ce": val_ce,
                    "val_vmae": val_vmae,
                    "val_top1": val_top1,
                    "val_top3": val_top3,
                    "epoch": ep,
                },
                args.out,
            )
            marker = " *BEST*"
        print(
            f"ep {ep:2d}/{args.epochs} | lr={sched.get_last_lr()[0]:.5f} | "
            f"tr_ce={tr_ce/tr_n:.4f} tr_top1={100.0*tr_c1/tr_n:5.2f}% tr_vmae={tr_vmae/tr_n:4.2f} | "
            f"val_ce={val_ce:.4f} val_top1={val_top1:5.2f}% val_top3={val_top3:5.2f}% val_vmae={val_vmae:4.2f} | "
            f"{dt:.1f}s{marker}",
            flush=True,
        )

    print(f"DONE! Best val_ce={best_val_ce:.4f} saved to {args.out}")


if __name__ == "__main__":
    main()
