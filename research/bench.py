#!/usr/bin/env python3
"""Paired, resumable full-100L experiments; no third-party dependencies."""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import random
import os
import signal
import subprocess
import time
import math

ROOT = pathlib.Path(__file__).resolve().parent.parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', default='research/bin/original')
    p.add_argument('--binaries', nargs='+', help='interleave variants under the same concurrency')
    p.add_argument('--configs', nargs='+', default=['5/5,10,10,10'])
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--count', type=int, default=12)
    p.add_argument('--seeds', type=int, nargs='+', help='explicit seeds, for separately labelled record hunts')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--move-ms', type=int)
    p.add_argument('--collect-features', action='store_true')
    p.add_argument('--value-weights', help='JSON array of14 frozen weights for tunable-value binaries')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    weights = json.loads(a.value_weights) if a.value_weights else None
    if weights is not None:
        assert len(weights)==14 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in weights)
    weights_json = json.dumps(weights,separators=(',',':')) if weights is not None else None
    weights_sha = hashlib.sha256(weights_json.encode()).hexdigest() if weights_json else None
    binaries = [(ROOT / b).resolve() for b in (a.binaries or [a.binary])]
    digests = {b: hashlib.sha256(b.read_bytes()).hexdigest() for b in binaries}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    old = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
    def collected(r):
        if 'collect_features' in r: return r['collect_features']
        if a.collect_features and r.get('trace'):
            return bool(json.loads((ROOT/r['trace']).read_text()).get('samples'))
        return False
    done = {(r['sha256'], r['config'], r['seed'], r.get('move_ms'), collected(r), r.get('weights_sha256')) for r in old}
    seeds = a.seeds if a.seeds is not None else range(a.start, a.start+a.count)
    tasks = [(b, c, s) for s in seeds for c in a.configs for b in binaries
             if (digests[b], c, s, a.move_ms, a.collect_features, weights_sha) not in done]
    random.Random(20261108).shuffle(tasks)

    def run(task):
        binary, cfg, seed = task
        digest = digests[binary]
        tag = '-'.join([out.stem, binary.name, digest[:8], hashlib.sha256(cfg.encode()).hexdigest()[:8], str(a.move_ms), str(a.collect_features), str(seed)])
        if weights_sha: tag += '-'+weights_sha[:12]
        trace = out.parent / (tag + '.json')
        cmd = [str(binary), '100', '-q', '-s', str(seed), '-g', '9', '-G', '4', '-A', cfg, '--trace-file', str(trace)]
        if a.move_ms is not None:
            cmd += ['--move-ms', str(a.move_ms)]
        if a.collect_features: cmd += ['--collect-features']
        usage_path = trace.with_suffix('.usage.json')
        if (ROOT/'research/bin/measure').exists():
            cmd = [str(ROOT/'research/bin/measure'), str(usage_path)] + cmd
        elif pathlib.Path('/usr/bin/time').exists():
            cmd = ['/usr/bin/time', '-f', '{"user_seconds":%U,"system_seconds":%S,"max_rss_kib":%M}',
                   '-o', str(usage_path), '--'] + cmd
        t = time.monotonic()
        env = os.environ.copy()
        env.pop('BLOCKFISH_VALUE_WEIGHTS',None)
        if weights_json: env['BLOCKFISH_VALUE_WEIGHTS'] = weights_json
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True, env=env)
        try:
            stdout, stderr = proc.communicate(timeout=a.timeout)
            data = json.loads(trace.read_text()) if trace.exists() else {}
            ds = data.get('ds', [])
            result = dict(status='complete' if proc.returncode == 0 and ds and ds[-1] == 100 else 'failed',
                          pieces=len(ds), cleared=ds[-1] if ds else 0, trace=str(trace.relative_to(ROOT)),
                          returncode=proc.returncode, stderr=stderr, race_seconds=data.get('time'),
                          search=data.get('search'))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
            result = dict(status='timeout', pieces=None, cleared=None)
        if usage_path.exists():
            # GNU time can prepend a signal/exit-status diagnostic; its last line
            # is the explicit JSON format even when the program exits nonzero.
            try:
                usage = json.loads(usage_path.read_text().splitlines()[-1])
                result.update(cpu_seconds=usage['user_seconds']+usage['system_seconds'],
                              max_rss_kib=usage['max_rss_kib'])
            except (ValueError, IndexError):
                pass
        return dict(binary=str(binary.relative_to(ROOT)), sha256=digest, config=cfg, seed=seed,
                    value_weights=weights, weights_sha256=weights_sha,
                    move_ms=a.move_ms, collect_features=a.collect_features,
                    wall_seconds=time.monotonic()-t, **result)

    with out.open('a') as f, concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run, task) for task in tasks]):
            result = future.result()
            f.write(json.dumps(result)+'\n')
            f.flush()
            print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
