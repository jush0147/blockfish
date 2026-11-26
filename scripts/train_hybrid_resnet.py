#!/usr/bin/env python3
"""
Train a Hybrid Residual 2D Conv ResNet on BFDS0002 Legal p=5 Self-Play datasets:
    V_hybrid(s') = base_v(s') + max_delta * tanh(ResNet(board, queue) / max_delta)
Initialized at delta = 0 so epoch 0 equals the exact spatial_bc4.txt baseline!

Usage:
    python3 scripts/train_hybrid_resnet.py \
        --data /tmp/legal_sp_v1_56g.bin.gz \
        --channels 64 --blocks 4 --epochs 25 --batch-size 64 \
        --out /tmp/hybrid_resnet_v1.pt
"""
import argparse
import gzip
import os
import struct
import sys
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_resnet import FiLMResBlock, GPUUnpacker


MOVE_BYTES_V2 = 48
DEC_HDR_BYTES = 12


def load_bfds0002(path: str):
    """Parse BFDS0002 (.bin or .bin.gz) with 48-byte MoveRecV2 (44-byte MoveRec + f32 base_v)."""
    t0 = time.time()
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            buf = f.read()
    else:
        with open(path, "rb") as f:
            buf = f.read()

    if len(buf) < 16 or buf[:8] != b"BFDS0002":
        raise ValueError(f"Invalid BFDS0002 header in {path}")

    n_games, n_decs = struct.unpack_from("<II", buf, 8)
    print(f"Loading {path}: {n_games} games, {n_decs} decisions ({len(buf)/1e6:.1f} MB raw)...", flush=True)

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
        pos += nm * MOVE_BYTES_V2
    offsets[n_decs] = total_moves

    moves_raw = np.empty((total_moves, 44), dtype=np.uint8)
    base_v = np.empty(total_moves, dtype=np.float32)
    pos = 16
    mv_idx = 0
    mv_view = memoryview(buf)
    for i in range(n_decs):
        nm = int(n_moves_arr[i])
        pos += DEC_HDR_BYTES
        nbytes = nm * MOVE_BYTES_V2
        chunk = np.frombuffer(mv_view[pos : pos + nbytes], dtype=np.uint8).reshape(nm, MOVE_BYTES_V2)
        moves_raw[mv_idx : mv_idx + nm] = chunk[:, :44]
        base_v[mv_idx : mv_idx + nm] = chunk[:, 44:48].copy().view(np.float32).reshape(nm)
        mv_idx += nm
        pos += nbytes

    print(
        f"Loaded {total_moves:,} moves in {time.time()-t0:.2f}s "
        f"(train_decs={(~is_val).sum():,}, val_decs={is_val.sum():,})",
        flush=True,
    )
    return {
        "moves_raw": moves_raw,
        "base_v": base_v,
        "offsets": offsets,
        "targets": targets,
        "rem_pcs": rem_pcs,
        "is_val": is_val,
    }


class BlockfishHybridResNet(nn.Module):
    """Zero-initialized bounded residual delta_v(board, queue) added on top of spatial_bc4 base_v."""

    def __init__(
        self,
        in_ch: int = 8,
        g_dim: int = 61,
        channels: int = 64,
        blocks: int = 4,
        max_delta: float = 12.0,
        temp: float = 0.5,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.g_dim = g_dim
        self.channels = channels
        self.blocks_n = blocks
        self.max_delta = max_delta
        self.temp = temp

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
        self.delta_fc1 = nn.Linear(head_in, channels)
        self.delta_fc2 = nn.Linear(channels, 1)
        # Zero-init final layer so epoch 0 == exact spatial_bc4.txt baseline!
        nn.init.zeros_(self.delta_fc2.weight)
        nn.init.zeros_(self.delta_fc2.bias)

    def forward_delta(self, boards: torch.Tensor, gvec: torch.Tensor) -> torch.Tensor:
        g_emb = self.g_enc(gvec)
        x = self.stem(boards)
        for blk in self.res_blocks:
            x = blk(x, g_emb)
        p_mean = x.mean(dim=(2, 3))
        p_max = x.amax(dim=(2, 3))
        p_bot4 = x[:, :, :4, :].mean(dim=(2, 3))
        feat = torch.cat([p_mean, p_max, p_bot4, g_emb], dim=-1)
        h = F.silu(self.delta_fc1(feat), inplace=True)
        raw = self.delta_fc2(h).squeeze(-1)
        return self.max_delta * torch.tanh(raw / self.max_delta)

    def forward(self, boards: torch.Tensor, gvec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.forward_delta(boards, gvec)
        return -d / self.temp, d


def pack_batch_v2(data, dec_indices: np.ndarray, max_neg: int = 0, rng=None):
    offsets = data["offsets"]
    moves_raw = data["moves_raw"]
    base_v_arr = data["base_v"]
    raw_counts = (offsets[dec_indices + 1] - offsets[dec_indices]).astype(np.int64)
    B = len(dec_indices)
    orig_targets = data["targets"][dec_indices]
    rem_pcs = data["rem_pcs"][dec_indices]

    if max_neg > 0:
        cap = max_neg + 1
        counts = np.minimum(raw_counts, cap)
        M_max = int(counts.max())
        total_m = int(counts.sum())
        raw_cat = np.empty((total_m, 44), dtype=np.uint8)
        bv_cat = np.empty(total_m, dtype=np.float32)
        dec_id = np.empty(total_m, dtype=np.int64)
        pos_in_dec = np.empty(total_m, dtype=np.int64)
        star_flat = np.empty(B, dtype=np.int64)
        targets = np.zeros(B, dtype=np.int64)

        cur = 0
        for b in range(B):
            di = dec_indices[b]
            s, e = int(offsets[di]), int(offsets[di + 1])
            c = e - s
            ti = int(orig_targets[b])
            if c <= cap:
                raw_cat[cur : cur + c] = moves_raw[s:e]
                bv_cat[cur : cur + c] = base_v_arr[s:e]
                dec_id[cur : cur + c] = b
                pos_in_dec[cur : cur + c] = np.arange(c, dtype=np.int64)
                star_flat[b] = cur + ti
                targets[b] = ti
                cur += c
            else:
                raw_cat[cur] = moves_raw[s + ti]
                bv_cat[cur] = base_v_arr[s + ti]
                neg_idx = rng.choice(c - 1, size=max_neg, replace=False)
                neg_idx = neg_idx + (neg_idx >= ti)
                raw_cat[cur + 1 : cur + cap] = moves_raw[s + neg_idx]
                bv_cat[cur + 1 : cur + cap] = base_v_arr[s + neg_idx]
                dec_id[cur : cur + cap] = b
                pos_in_dec[cur : cur + cap] = np.arange(cap, dtype=np.int64)
                star_flat[b] = cur
                targets[b] = 0
                cur += cap
        return raw_cat, bv_cat, dec_id, pos_in_dec, star_flat, targets, rem_pcs, B, M_max

    M_max = int(raw_counts.max())
    total_m = int(raw_counts.sum())
    raw_cat = np.empty((total_m, 44), dtype=np.uint8)
    bv_cat = np.empty(total_m, dtype=np.float32)
    dec_id = np.empty(total_m, dtype=np.int64)
    pos_in_dec = np.empty(total_m, dtype=np.int64)
    star_flat = np.empty(B, dtype=np.int64)
    targets = orig_targets

    cur = 0
    for b in range(B):
        di = dec_indices[b]
        s, e = int(offsets[di]), int(offsets[di + 1])
        c = e - s
        raw_cat[cur : cur + c] = moves_raw[s:e]
        bv_cat[cur : cur + c] = base_v_arr[s:e]
        dec_id[cur : cur + c] = b
        pos_in_dec[cur : cur + c] = np.arange(c, dtype=np.int64)
        star_flat[b] = cur + int(targets[b])
        cur += c
    return raw_cat, bv_cat, dec_id, pos_in_dec, star_flat, targets, rem_pcs, B, M_max


def compute_hybrid_loss(
    base_v: torch.Tensor,
    delta_v: torch.Tensor,
    dec_id: torch.Tensor,
    pos_in_dec: torch.Tensor,
    star_flat: torch.Tensor,
    targets: torch.Tensor,
    rem_pcs: torch.Tensor,
    B: int,
    M_max: int,
    temp: float,
    val_weight: float,
    reg_weight: float,
):
    v_hyb = base_v + delta_v
    logits = -v_hyb / temp
    padded = logits.new_full((B, M_max), -1e9)
    padded[dec_id, pos_in_dec] = logits
    ce_loss = F.cross_entropy(padded, targets)

    pred_star_v = v_hyb[star_flat]
    v_loss = F.smooth_l1_loss(pred_star_v, rem_pcs, beta=2.0)
    v_mae = (pred_star_v - rem_pcs).abs().mean()
    reg_loss = (delta_v * delta_v).mean()

    with torch.no_grad():
        top1_correct = (padded.argmax(dim=1) == targets).sum()
        k3 = min(3, M_max)
        top3_idx = padded.topk(k=k3, dim=1).indices
        top3_correct = (top3_idx == targets.unsqueeze(1)).any(dim=1).sum()
        d_abs = delta_v.abs().mean()

    total_loss = ce_loss + val_weight * v_loss + reg_weight * reg_loss
    return total_loss, ce_loss.detach(), v_mae.detach(), top1_correct, top3_correct, d_abs


@torch.no_grad()
def evaluate_hybrid(model, unpacker, data, val_indices: np.ndarray, batch_size: int, device, amp_dtype, zero_delta: bool = False):
    model.eval()
    total_ce = torch.zeros((), device=device)
    total_vmae = torch.zeros((), device=device)
    total_top1 = torch.zeros((), device=device, dtype=torch.int64)
    total_top3 = torch.zeros((), device=device, dtype=torch.int64)
    total_dabs = torch.zeros((), device=device)
    total_n = 0

    for start in range(0, len(val_indices), batch_size):
        idx = val_indices[start : start + batch_size]
        raw_cat, bv_cat, dec_id, pos_in_dec, star_flat, targets_np, rem_np, B, M_max = pack_batch_v2(data, idx)
        raw_t = torch.from_numpy(raw_cat).to(device, non_blocking=True)
        bv_t = torch.from_numpy(bv_cat).to(device, non_blocking=True)
        dec_id_t = torch.from_numpy(dec_id).to(device, non_blocking=True)
        pos_t = torch.from_numpy(pos_in_dec).to(device, non_blocking=True)
        star_t = torch.from_numpy(star_flat).to(device, non_blocking=True)
        targets_t = torch.from_numpy(targets_np).to(device, non_blocking=True)
        rem_t = torch.from_numpy(rem_np).to(device, non_blocking=True)

        if zero_delta:
            delta_v = torch.zeros_like(bv_t)
        else:
            boards, gvec = unpacker(raw_t)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                delta_v = model.forward_delta(boards, gvec).float()

        _, ce, vmae, c1, c3, dabs = compute_hybrid_loss(
            bv_t, delta_v, dec_id_t, pos_t, star_t, targets_t, rem_t, B, M_max, model.temp, 0.05, 0.0
        )
        total_ce += ce * B
        total_vmae += vmae * B
        total_top1 += c1
        total_top3 += c3
        total_dabs += dabs * B
        total_n += B

    return (
        (total_ce / max(total_n, 1)).item(),
        (total_vmae / max(total_n, 1)).item(),
        100.0 * total_top1.item() / max(total_n, 1),
        100.0 * total_top3.item() / max(total_n, 1),
        (total_dabs / max(total_n, 1)).item(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to BFDS0002 .bin or .bin.gz")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-neg", type=int, default=15)
    ap.add_argument("--max-delta", type=float, default=12.0)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=2e-4)
    ap.add_argument("--val-weight", type=float, default=0.05)
    ap.add_argument("--reg-weight", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/tmp/hybrid_resnet_v1.pt")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"Device: {device} | amp_dtype={amp_dtype}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    data = load_bfds0002(args.data)
    all_idx = np.arange(len(data["targets"]))
    train_idx = all_idx[~data["is_val"]]
    val_idx = all_idx[data["is_val"]]

    unpacker = GPUUnpacker().to(device)
    model = BlockfishHybridResNet(
        channels=args.channels, blocks=args.blocks, max_delta=args.max_delta, temp=args.temp
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"BlockfishHybridResNet(C={args.channels}, B={args.blocks}, max_delta={args.max_delta}, temp={args.temp}) | params={n_params:,}",
        flush=True,
    )

    # Evaluate exact spatial_bc4.txt baseline (delta = 0) before training
    b_ce, b_vmae, b_t1, b_t3, _ = evaluate_hybrid(model, unpacker, data, val_idx, 128, device, amp_dtype, zero_delta=True)
    print(
        f"BASE (spatial_bc4, delta=0) | val_ce={b_ce:.4f} val_top1={b_t1:5.2f}% val_top3={b_t3:5.2f}% val_vmae={b_vmae:5.2f}",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_val_ce = b_ce
    rng = np.random.default_rng(args.seed)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(train_idx)
        tr_ce = torch.zeros((), device=device)
        tr_vmae = torch.zeros((), device=device)
        tr_c1 = torch.zeros((), device=device, dtype=torch.int64)
        tr_n = 0

        for start in range(0, len(perm), args.batch_size):
            idx = perm[start : start + args.batch_size]
            raw_cat, bv_cat, dec_id, pos_in_dec, star_flat, targets_np, rem_np, B, M_max = pack_batch_v2(
                data, idx, max_neg=args.max_neg, rng=rng
            )
            raw_t = torch.from_numpy(raw_cat).to(device, non_blocking=True)
            bv_t = torch.from_numpy(bv_cat).to(device, non_blocking=True)
            dec_id_t = torch.from_numpy(dec_id).to(device, non_blocking=True)
            pos_t = torch.from_numpy(pos_in_dec).to(device, non_blocking=True)
            star_t = torch.from_numpy(star_flat).to(device, non_blocking=True)
            targets_t = torch.from_numpy(targets_np).to(device, non_blocking=True)
            rem_t = torch.from_numpy(rem_np).to(device, non_blocking=True)

            boards, gvec = unpacker(raw_t)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                delta_v = model.forward_delta(boards, gvec)
                loss, ce, vmae, c1, _, _ = compute_hybrid_loss(
                    bv_t, delta_v.float(), dec_id_t, pos_t, star_t, targets_t, rem_t,
                    B, M_max, args.temp, args.val_weight, args.reg_weight
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()

            tr_ce += ce * B
            tr_vmae += vmae * B
            tr_c1 += c1
            tr_n += B

        sched.step()
        val_ce, val_vmae, val_top1, val_top3, dabs = evaluate_hybrid(
            model, unpacker, data, val_idx, 128, device, amp_dtype
        )
        dt = time.time() - t0
        marker = ""
        if val_ce < best_val_ce:
            best_val_ce = val_ce
            torch.save(
                {
                    "arch": "hybrid_v1",
                    "state_dict": model.state_dict(),
                    "channels": args.channels,
                    "blocks": args.blocks,
                    "in_ch": 8,
                    "g_dim": 61,
                    "max_delta": args.max_delta,
                    "temp": args.temp,
                    "val_ce": val_ce,
                    "val_vmae": val_vmae,
                    "val_top1": val_top1,
                    "val_top3": val_top3,
                    "base_val_ce": b_ce,
                    "epoch": ep,
                },
                args.out,
            )
            marker = " *BEST*"
        print(
            f"ep {ep:2d}/{args.epochs} | lr={sched.get_last_lr()[0]:.5f} | "
            f"tr_ce={(tr_ce/tr_n).item():.4f} tr_top1={100.0*tr_c1.item()/tr_n:5.2f}% tr_vmae={(tr_vmae/tr_n).item():5.2f} | "
            f"val_ce={val_ce:.4f} val_top1={val_top1:5.2f}% val_top3={val_top3:5.2f}% val_vmae={val_vmae:5.2f} |d|={dabs:.2f} | "
            f"{dt:.1f}s{marker}",
            flush=True,
        )

    print(f"DONE! Base val_ce={b_ce:.4f} -> Best hybrid val_ce={best_val_ce:.4f} saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
