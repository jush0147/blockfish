#!/usr/bin/env python3
"""Interleaved serial races; measure process CPU independently of wall scheduling."""
import argparse
import hashlib
import json
import pathlib
import random
import resource
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent

def cpu():
    u = resource.getrusage(resource.RUSAGE_CHILDREN)
    return u.ru_utime + u.ru_stime

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binaries', nargs='+', default=['original', 'cache', 'bit', 'flood'])
    p.add_argument('--config', default='5/5,10,10,10')
    p.add_argument('--count', type=int, default=4)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    tasks = [(s,b) for s in range(a.start, a.start+a.count) for b in a.binaries]
    random.Random(187).shuffle(tasks)
    out = ROOT / a.out
    with out.open('w') as f:
        for s,b in tasks:
            binary = ROOT / 'research/bin' / b
            trace = out.parent / f'{out.stem}-{b}-{s}.json'
            c, t = cpu(), time.monotonic()
            proc = subprocess.run([str(binary),'100','-q','-s',str(s),'-g','9','-G','4',
                                   '-A',a.config,'--trace-file',str(trace)],capture_output=True,text=True,timeout=180)
            elapsed_cpu, elapsed_wall = cpu()-c, time.monotonic()-t
            data = json.loads(trace.read_text())
            r = dict(binary=b, seed=s, config=a.config, cpu_seconds=elapsed_cpu,
                     wall_seconds=elapsed_wall, pieces=len(data['ds']), returncode=proc.returncode,
                     trace=str(trace.relative_to(ROOT)), sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
            f.write(json.dumps(r)+'\n'); f.flush()
            print(json.dumps(r),flush=True)

if __name__ == '__main__':
    main()
