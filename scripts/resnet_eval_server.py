#!/usr/bin/env python3
"""
High-throughput Unix-socket GPU batch evaluator for BlockfishResNet.
Shares the exact GPUUnpacker and BlockfishResNet from scripts/train_resnet.py
so there is ZERO feature skew between training and inference.

Binary protocol (little-endian):
  Request : <u32 N> followed by N * 44 bytes (BFDS0001 MoveRec)
            N == 0          -> close client connection
            N == 0xFFFFFFFF -> shut down server cleanly
  Response: N * 8 bytes (<f32 logit, f32 val> interleaved)

Usage:
    python3 scripts/resnet_eval_server.py \
        --ckpt /tmp/resnet_v1_c96b6.pt \
        --sock /tmp/blockfish_resnet.sock
"""
import argparse
import os
import socket
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_resnet import BlockfishResNet, GPUUnpacker, MOVE_BYTES


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = conn.recv_into(view[got:], n - got)
        if r == 0:
            raise EOFError("Client disconnected mid-packet")
        got += r
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sock", default="/tmp/blockfish_resnet.sock")
    ap.add_argument("--max-chunk", type=int, default=8192)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model = BlockfishResNet(
        in_ch=ckpt.get("in_ch", 8),
        g_dim=ckpt.get("g_dim", 61),
        channels=ckpt["channels"],
        blocks=ckpt["blocks"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    unpacker = GPUUnpacker().to(device)

    # Warmup CUDA kernels
    if device.type == "cuda":
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=amp_dtype):
            dummy = torch.zeros((256, MOVE_BYTES), dtype=torch.uint8, device=device)
            b, g = unpacker(dummy)
            _ = model(b, g)
        torch.cuda.synchronize()

    if os.path.exists(args.sock):
        os.unlink(args.sock)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.sock)
    srv.listen(4)
    print(
        f"resnet_eval_server READY on {args.sock} | device={device} ({amp_dtype}) | "
        f"C={ckpt['channels']} B={ckpt['blocks']} val_ce={ckpt.get('val_ce', -1):.4f}",
        flush=True,
    )

    total_batches = 0
    total_boards = 0
    t_start = time.time()
    shutdown = False

    try:
        while not shutdown:
            conn, _ = srv.accept()
            with conn:
                while True:
                    hdr = conn.recv(4)
                    if not hdr or len(hdr) < 4:
                        break
                    (n,) = struct.unpack("<I", hdr)
                    if n == 0:
                        break
                    if n == 0xFFFFFFFF:
                        shutdown = True
                        break
                    raw_bytes = recv_exact(conn, n * MOVE_BYTES)
                    raw_np = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n, MOVE_BYTES)

                    out = np.empty((n, 2), dtype=np.float32)
                    with torch.inference_mode():
                        for s in range(0, n, args.max_chunk):
                            e = min(s + args.max_chunk, n)
                            raw_t = torch.from_numpy(raw_np[s:e].copy()).to(device, non_blocking=True)
                            boards, gvec = unpacker(raw_t)
                            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                                logits, vals = model(boards, gvec)
                            out[s:e, 0] = logits.float().cpu().numpy()
                            out[s:e, 1] = vals.float().cpu().numpy()

                    conn.sendall(out.tobytes())
                    total_batches += 1
                    total_boards += n
    finally:
        srv.close()
        if os.path.exists(args.sock):
            os.unlink(args.sock)
        dt = max(time.time() - t_start, 1e-3)
        print(
            f"resnet_eval_server SHUTDOWN | batches={total_batches} boards={total_boards:,} "
            f"({total_boards/dt:,.0f} boards/s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
