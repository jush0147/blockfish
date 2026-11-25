#!/usr/bin/env python3
"""
Ultra-fast GPU trainer for FiLM+SE 2D Conv ResNet on BFDS0001 datasets.
All feature unpacking and variable-length decision Cross-Entropy run 100% on GPU
with ZERO per-batch CPU-GPU synchronizations (~5-10s/epoch on Colab L4 GPU).

Usage:
    python3 scripts/train_resnet.py \
        --data /tmp/wg_ds_v1_400g.bin.gz \
        --channels 96 --blocks 6 --epochs 25 --batch-size 64 \
        --out /tmp/resnet_v1_c96b6.pt
"""
import argparse
import gzip
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
    """Parse BFDS0001 (.bin or .bin.gz) into compact numpy arrays (~136 MB RAM)."""
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
        pos += nm * MOVE_BYTES
    offsets[n_decs] = total_moves

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
        f"(train_decs={(~is_val).sum():,}, val_decs={is_val.sum():,})",
        flush=True,
    )
    return {
        "moves_raw": moves_raw,
        "offsets": offsets,
        "targets": targets,
        "rem_pcs": rem_pcs,
        "is_val": is_val,
    }


class GPUUnpacker(nn.Module):
    """Unpacks (N, 44) uint8 MoveRec tensors into (N, 8, 16, 10) boards and (N, 61) globals on GPU."""

    def __init__(self):
        super().__init__()
        self.register_buffer("bit_shifts", torch.arange(10, dtype=torch.int32).view(1, 1, 10))
        self.register_buffer("row_idx", torch.arange(16, dtype=torch.int32).view(1, 16, 1))
        self.register_buffer("bag_shifts", torch.arange(7, dtype=torch.int32).view(1, 7))
        self.register_buffer("piece_ids", torch.arange(8, dtype=torch.int64).view(1, 8))

    @torch.no_grad()
    def forward(self, raw_u8: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = raw_u8.shape[0]
        # Little-endian u16 rows from bytes 0..32
        lo = raw_u8[:, 0:32:2].to(torch.int32)
        hi = raw_u8[:, 1:32:2].to(torch.int32)
        rows_u16 = lo | (hi << 8)  # (N, 16)
        occ = ((rows_u16.unsqueeze(2) >> self.bit_shifts) & 1).to(torch.float32)  # (N, 16, 10)

        board_rows = raw_u8[:, 32].to(torch.float32)
        garbage_rows_i = raw_u8[:, 33].to(torch.int32)
        garbage_rows_f = garbage_rows_i.to(torch.float32)
        rem_g = raw_u8[:, 34].to(torch.float32)
        held = raw_u8[:, 35].to(torch.int64).clamp_(0, 7)
        next5 = raw_u8[:, 36:41].to(torch.int64).clamp_(0, 7)
        bag_mask = raw_u8[:, 41].to(torch.int32)
        lines_cleared = raw_u8[:, 42].to(torch.float32)
        ds_cleared = raw_u8[:, 43].to(torch.float32)

        is_g_row = (self.row_idx < garbage_rows_i.view(N, 1, 1)).to(torch.float32)  # (N, 16, 1)
        g_mask = occ * is_g_row
        s_mask = occ * (1.0 - is_g_row)
        hole_mask = (1.0 - occ) * is_g_row

        # Covered empty cells: empty cell with >=1 occupied cell strictly above it
        occ_rev = torch.flip(occ, dims=[1])
        above_occ = torch.flip(torch.cumsum(occ_rev, dim=1), dims=[1]) - occ
        covered = (1.0 - occ) * (above_occ > 0.5).to(torch.float32)

        # Top garbage hole column (r = g - 1) and 2nd garbage hole column (r = g - 2)
        ar = torch.arange(N, device=raw_u8.device)
        r1 = (garbage_rows_i - 1).clamp(0, 15).to(torch.int64)
        has_g1 = (garbage_rows_i >= 1).to(torch.float32).view(N, 1, 1)
        h1_bits = (1.0 - occ[ar, r1, :]).unsqueeze(1) * has_g1
        h1_col = h1_bits.expand(N, 16, 10)

        r2 = (garbage_rows_i - 2).clamp(0, 15).to(torch.int64)
        has_g2 = (garbage_rows_i >= 2).to(torch.float32).view(N, 1, 1)
        h2_bits = (1.0 - occ[ar, r2, :]).unsqueeze(1) * has_g2
        h2_col = h2_bits.expand(N, 16, 10)

        row_fill = (occ.sum(dim=2, keepdim=True) * 0.1).expand(N, 16, 10)

        boards = torch.stack(
            [occ, g_mask, s_mask, hole_mask, covered, h1_col, h2_col, row_fill],
            dim=1,
        )  # (N, 8, 16, 10)

        # Global vector (N, 61)
        held_oh = (held.unsqueeze(1) == self.piece_ids).to(torch.float32)  # (N, 8)
        next5_oh = (next5.unsqueeze(2) == self.piece_ids.unsqueeze(1)).to(torch.float32).reshape(N, 40)
        bag_bits = ((bag_mask.unsqueeze(1) >> self.bag_shifts) & 1).to(torch.float32)  # (N, 7)
        scalars = torch.stack(
            [
                rem_g * 0.01,
                garbage_rows_f * 0.1,
                board_rows * 0.05,
                lines_cleared * 0.25,
                ds_cleared * 0.25,
                (rem_g <= 10.5).to(torch.float32),
            ],
            dim=1,
        )  # (N, 6)
        gvec = torch.cat([held_oh, next5_oh, bag_bits, scalars], dim=1)  # (N, 61)
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
        se_w = self.se(h.mean(dim=(2, 3)))[:, :, None, None]
        h = h * se_w
        return F.silu(x + h, inplace=True)


class BlockfishResNet(nn.Module):
    def __init__(self, in_ch: int = 8, g_dim: int = 61, channels: int = 96, blocks: int = 6):
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


def pack_batch_indices(data, dec_indices: np.ndarray, max_neg: int = 0, rng=None):
    """Build flat raw_u8 + scatter indices for zero-sync GPU padded Cross-Entropy.
    If max_neg > 0, samples 1 positive + up to max_neg random negatives per decision (~4x faster).
    """
    offsets = data["offsets"]
    moves_raw = data["moves_raw"]
    raw_counts = (offsets[dec_indices + 1] - offsets[dec_indices]).astype(np.int64)
    B = len(dec_indices)
    orig_targets = data["targets"][dec_indices]
    rem_pcs = data["rem_pcs"][dec_indices]

    if max_neg > 0:
        cap = max_neg + 1
        counts = np.minimum(raw_counts, cap)
        M_max = int(counts.max())
        total_m = int(counts.sum())
        raw_cat = np.empty((total_m, MOVE_BYTES), dtype=np.uint8)
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
                dec_id[cur : cur + c] = b
                pos_in_dec[cur : cur + c] = np.arange(c, dtype=np.int64)
                star_flat[b] = cur + ti
                targets[b] = ti
                cur += c
            else:
                # Place positive target at slot 0, sample max_neg distinct negatives
                raw_cat[cur] = moves_raw[s + ti]
                neg_idx = rng.choice(c - 1, size=max_neg, replace=False)
                neg_idx = neg_idx + (neg_idx >= ti)
                raw_cat[cur + 1 : cur + cap] = moves_raw[s + neg_idx]
                dec_id[cur : cur + cap] = b
                pos_in_dec[cur : cur + cap] = np.arange(cap, dtype=np.int64)
                star_flat[b] = cur
                targets[b] = 0
                cur += cap
        return raw_cat, dec_id, pos_in_dec, star_flat, targets, rem_pcs, B, M_max

    M_max = int(raw_counts.max())
    total_m = int(raw_counts.sum())
    raw_cat = np.empty((total_m, MOVE_BYTES), dtype=np.uint8)
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
        dec_id[cur : cur + c] = b
        pos_in_dec[cur : cur + c] = np.arange(c, dtype=np.int64)
        star_flat[b] = cur + int(targets[b])
        cur += c

    return raw_cat, dec_id, pos_in_dec, star_flat, targets, rem_pcs, B, M_max


def compute_batched_loss(
    logits: torch.Tensor,
    vals: torch.Tensor,
    dec_id: torch.Tensor,
    pos_in_dec: torch.Tensor,
    star_flat: torch.Tensor,
    targets: torch.Tensor,
    rem_pcs: torch.Tensor,
    B: int,
    M_max: int,
    val_weight: float,
):
    """100% vectorized GPU loss + top1/top3 accuracy with ZERO CPU-GPU syncs."""
    padded = logits.new_full((B, M_max), -1e9)
    padded[dec_id, pos_in_dec] = logits
    ce_loss = F.cross_entropy(padded, targets)

    pred_star_v = vals[star_flat]
    v_loss = F.smooth_l1_loss(pred_star_v, rem_pcs, beta=2.0)
    v_mae = (pred_star_v - rem_pcs).abs().mean()

    with torch.no_grad():
        top1_correct = (padded.argmax(dim=1) == targets).sum()
        k3 = min(3, M_max)
        top3_idx = padded.topk(k=k3, dim=1).indices
        top3_correct = (top3_idx == targets.unsqueeze(1)).any(dim=1).sum()

    total_loss = ce_loss + val_weight * v_loss
    return total_loss, ce_loss.detach(), v_mae.detach(), top1_correct, top3_correct


@torch.no_grad()
def evaluate(model, unpacker, data, val_indices: np.ndarray, batch_size: int, device, amp_dtype):
    model.eval()
    total_ce = torch.zeros((), device=device)
    total_vmae = torch.zeros((), device=device)
    total_top1 = torch.zeros((), device=device, dtype=torch.int64)
    total_top3 = torch.zeros((), device=device, dtype=torch.int64)
    total_n = 0

    for start in range(0, len(val_indices), batch_size):
        idx = val_indices[start : start + batch_size]
        raw_cat, dec_id, pos_in_dec, star_flat, targets_np, rem_np, B, M_max = pack_batch_indices(data, idx)
        raw_t = torch.from_numpy(raw_cat).to(device, non_blocking=True)
        dec_id_t = torch.from_numpy(dec_id).to(device, non_blocking=True)
        pos_t = torch.from_numpy(pos_in_dec).to(device, non_blocking=True)
        star_t = torch.from_numpy(star_flat).to(device, non_blocking=True)
        targets_t = torch.from_numpy(targets_np).to(device, non_blocking=True)
        rem_t = torch.from_numpy(rem_np).to(device, non_blocking=True)

        boards, gvec = unpacker(raw_t)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits, vals = model(boards, gvec)
            _, ce, vmae, c1, c3 = compute_batched_loss(
                logits.float(), vals.float(), dec_id_t, pos_t, star_t, targets_t, rem_t, B, M_max, 0.02
            )
        total_ce += ce * B
        total_vmae += vmae * B
        total_top1 += c1
        total_top3 += c3
        total_n += B

    return (
        (total_ce / max(total_n, 1)).item(),
        (total_vmae / max(total_n, 1)).item(),
        100.0 * total_top1.item() / max(total_n, 1),
        100.0 * total_top3.item() / max(total_n, 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to BFDS0001 .bin or .bin.gz")
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val-weight", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-neg", type=int, default=11, help="Max negative moves per decision during training (0=all)")
    ap.add_argument("--out", default="/tmp/resnet_v1_c64b4.pt")
    ap.add_argument("--push-status", action="store_true", help="Push live epoch metrics to results/colab_status.txt")
    args = ap.parse_args()

    def push_live_status(msg: str):
        if not args.push_status:
            return
        try:
            import subprocess
            with open("results/colab_status.txt", "w") as f:
                f.write(msg + "\n")
            subprocess.run(["git", "add", "-f", "results/colab_status.txt"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["git", "commit", "-m", f"colab: {msg} [skip ci]", "--quiet"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["git", "push", "origin", "rescue/independent-validation-20261116", "--quiet"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        # MUST be False: total_m (sum of legal moves per batch) varies every step,
        # so cudnn.benchmark=True causes a ~2.4s cuDNN autotune miss on EVERY batch!
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"Device: {device} | amp_dtype={amp_dtype}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    data = load_bfds0001(args.data)
    all_idx = np.arange(len(data["targets"]))
    train_idx = all_idx[~data["is_val"]]
    val_idx = all_idx[data["is_val"]]

    unpacker = GPUUnpacker().to(device)
    model = BlockfishResNet(channels=args.channels, blocks=args.blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BlockfishResNet(C={args.channels}, B={args.blocks}) | params={n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_val_ce = float("inf")
    rng = np.random.default_rng(args.seed)
    total_batches = (len(train_idx) + args.batch_size - 1) // args.batch_size

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(train_idx)
        tr_ce = torch.zeros((), device=device)
        tr_vmae = torch.zeros((), device=device)
        tr_c1 = torch.zeros((), device=device, dtype=torch.int64)
        tr_n = 0

        for bi, start in enumerate(range(0, len(perm), args.batch_size), start=1):
            idx = perm[start : start + args.batch_size]
            raw_cat, dec_id, pos_in_dec, star_flat, targets_np, rem_np, B, M_max = pack_batch_indices(
                data, idx, max_neg=args.max_neg, rng=rng
            )
            raw_t = torch.from_numpy(raw_cat).to(device, non_blocking=True)
            dec_id_t = torch.from_numpy(dec_id).to(device, non_blocking=True)
            pos_t = torch.from_numpy(pos_in_dec).to(device, non_blocking=True)
            star_t = torch.from_numpy(star_flat).to(device, non_blocking=True)
            targets_t = torch.from_numpy(targets_np).to(device, non_blocking=True)
            rem_t = torch.from_numpy(rem_np).to(device, non_blocking=True)

            boards, gvec = unpacker(raw_t)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits, vals = model(boards, gvec)
                loss, ce, vmae, c1, _ = compute_batched_loss(
                    logits.float(), vals.float(), dec_id_t, pos_t, star_t, targets_t, rem_t, B, M_max, args.val_weight
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
            if ep == 1 and bi % 150 == 0:
                print(
                    f"  [ep 1 | batch {bi}/{total_batches} | {time.time()-t0:.1f}s | tr_ce={(tr_ce/tr_n).item():.4f}]",
                    flush=True,
                )

        sched.step()
        val_ce, val_vmae, val_top1, val_top3 = evaluate(
            model, unpacker, data, val_idx, args.batch_size * 2, device, amp_dtype
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
        line = (
            f"ep {ep:2d}/{args.epochs} | lr={sched.get_last_lr()[0]:.5f} | "
            f"tr_ce={(tr_ce/tr_n).item():.4f} tr_top1={100.0*tr_c1.item()/tr_n:5.2f}% tr_vmae={(tr_vmae/tr_n).item():4.2f} | "
            f"val_ce={val_ce:.4f} val_top1={val_top1:5.2f}% val_top3={val_top3:5.2f}% val_vmae={val_vmae:4.2f} | "
            f"{dt:.1f}s{marker}"
        )
        print(line, flush=True)
        if ep == 1 or ep % 3 == 0 or ep == args.epochs:
            push_live_status(f"ep {ep}/{args.epochs} val_ce={val_ce:.4f} top1={val_top1:.1f}% vmae={val_vmae:.2f} ({dt:.1f}s)")

    print(f"DONE! Best val_ce={best_val_ce:.4f} saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
