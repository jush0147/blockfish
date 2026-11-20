#!/usr/bin/env python3
"""Rebuild ablations; run after sourcing the workspace's Rust activation script."""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
UPSTREAM = '739953f9331b7648f269fc52cff5aae330108ccd'
FEATURES = {
    "robust-classic-guide": ["classic-guide"],
    "robust-mirror": ["mirror-fusion"],
    "robust-reveal-boundary": ["reveal-boundary"],
    "robust-dig-geometry": ["dig-geometry"],
    "robust-plan-reuse": ["cheese-robust", "plan-reuse"],
    "robust-plan-observer": ["cheese-robust", "plan-observer"],
    "robust-global-regret": ["cheese-robust", "global-regret"],
    "robust-global-f": ["cheese-robust", "global-absolute"],
    "public-labels": ["public-rollout"],
    "robust-dag": ["cheese-robust", "root-dag"],
    "robust-dag-observer": ["cheese-robust", "dag-observer"],
    'world-diagnostic': ['public-rollout'],
    'world-rollout': ['public-rollout'],
    'world-rollout-placebo': ['rollout-placebo'],
    'robust-focus': ['cheese-robust', 'root-focus'],
    'robust-halving': ['cheese-robust', 'root-halving'],
    'halving-control': ['cheese-robust'],
    'robust-hold-cached': ['cheese-robust', 'hold-context-cache'],
    'robust-hold-context': ['cheese-robust', 'hold-context', 'preserve-hold'],
    'robust-hold-order': ['cheese-robust', 'hold-context'],
    'context-source': ['cheese-quality'],
    'robust-preserve-hold': ['cheese-robust', 'preserve-hold'],
    'robust-bag': ['cheese-robust', 'bag-inference'],
    'robust-bag-placebo': ['cheese-robust', 'bag-placebo'],
    'robust-hierarchy': ['cheese-robust', 'hierarchical-backup'],
    'robust-hierarchy-control': ['cheese-robust'],
    'robust-support-probe': ['cheese-robust', 'support-probe'],
    'robust-branches': ['cheese-robust', 'branch-backup'],
    'robust-tail-hold': ['cheese-robust', 'tail-hold'],
    'robust-tail-hold-placebo': ['cheese-robust', 'tail-hold', 'tail-placebo'],
    'robust-tail': ['cheese-robust', 'tail-expectation'],
    'robust-tail-placebo': ['cheese-robust', 'tail-placebo'],
    'learned-robust-endgame': ['cheese-robust', 'endgame-wide'],
    'cheese-robust': ['cheese-robust'],
    'learned-robust-covered': ['cheese-robust', 'root-complete'],
    'learned-robust-placebo': ['cheese-quality', 'robust-placebo'],
    'learned-robust': ['cheese-quality', 'robust-backup'],
    'learned-direct': ['cheese-quality', 'direct-value'],
    'learned-tunable': ['cheese-quality', 'tunable-value'],
    'learned-endgame-wide': ['cheese-quality', 'endgame-wide'],
    'learned-endgame-wide-placebo': ['cheese-quality', 'endgame-wide', 'endgame-placebo'],
    'learned-endgame-queue': ['cheese-quality', 'endgame-queue-bound'],
    'learned-endgame-strong': ['cheese-quality', 'endgame-color-bound'],
    'learned-endgame': ['cheese-quality', 'endgame-solver'],
    'learned-endgame-placebo': ['cheese-quality', 'endgame-placebo'],
    'learned-rerank': ['cheese-quality', 'root-rerank'],
    'learned-rerank-placebo': ['cheese-quality', 'root-rerank-placebo'],
    'learned-root-complete': ['cheese-quality', 'root-complete'],
    'learned-rank': ['cheese-quality', 'rank-value'],
    'learned-rank-blend': ['cheese-quality', 'rank-blend'],
    'learned-covered': ['cheese-quality', 'root-coverage'],
    'rank-data': ['cheese-quality'],
    'learned-eager': ['cheese-quality', 'eager-leaves'],
    'learned-eager-hold': ['cheese-quality', 'eager-leaves', 'preserve-hold'],
    'learned-beam': ['cheese-quality', 'wide-beam', 'beam-dedup'],
    'learned-race-goal': ['cheese-quality', 'race-goal'],
    'learned-td': ['cheese-quality', 'td-value'],
    'learned-terminal': ['cheese-quality', 'terminal-only'],
    'learned-policy2': ['cheese-quality', 'policy2-value'],
    'cheese-quality': ['cheese-quality'],
    'learned-fast': ['flood-eval', 'eval-cache', 'generated-budget', 'learned-value', 'fast-value'],
    'blend-fast': ['flood-eval', 'eval-cache', 'generated-budget', 'learned-blend', 'fast-value'],
    'global-blend-fast': ['flood-eval', 'eval-cache', 'shared-states', 'learned-blend', 'fast-value'],
    'global-learned': ['flood-eval', 'eval-cache', 'shared-states', 'learned-value'],
    'global-blend': ['flood-eval', 'eval-cache', 'shared-states', 'learned-blend'],
    'global-shared': ['flood-eval', 'eval-cache', 'shared-states'],
    'data-source': ['flood-eval', 'eval-cache'],
    'learned': ['flood-eval', 'eval-cache', 'generated-budget', 'learned-value'],
    'learned-blend': ['flood-eval', 'eval-cache', 'generated-budget', 'learned-blend'],
    'global-dedup': ['flood-eval', 'eval-cache', 'global-dedup'],
    'beam-unique-wide': ['flood-eval', 'eval-cache', 'beam-wider', 'beam-dedup'],
    'beam-unique-held': ['flood-eval', 'eval-cache', 'wide-beam', 'beam-dedup', 'preserve-hold'],
    'beam-wider': ['flood-eval', 'eval-cache', 'beam-wider'],
    'beam-widest': ['flood-eval', 'eval-cache', 'beam-widest'],
    'beam-held': ['flood-eval', 'eval-cache', 'wide-beam', 'beam-leaves', 'preserve-hold'],
    'hold-credit': ['flood-eval', 'eval-cache', 'generated-budget', 'hold-credit'],
    'beam-held-credit': ['flood-eval', 'eval-cache', 'wide-beam', 'beam-leaves', 'hold-credit'],
    'quality-baseline': ['flood-eval', 'eval-cache', 'generated-budget'],
    'hold-horizon': ['flood-eval', 'eval-cache', 'generated-budget', 'preserve-hold'],
    'hold-leaf': ['flood-eval', 'eval-cache', 'generated-budget', 'preserve-hold', 'terminal-only'],
    'beam-all-leaves': ['flood-eval', 'eval-cache', 'wide-beam', 'beam-leaves'],
    'beam-unique': ['flood-eval', 'eval-cache', 'wide-beam', 'beam-dedup'],
    'final-baseline': [], 'final-optimized': ['flood-eval', 'eval-cache'],
    'timed': [], 'baseline': [], 'cache': ['eval-cache'], 'bit': ['bit-eval'],
    'flood': ['flood-eval'], 'fast': ['flood-eval'],
    'fast-cache': ['flood-eval', 'eval-cache'],
    'terminal': ['eval-cache', 'terminal-only'],
    'residue': ['eval-cache', 'goal-residue'],
    'generated': ['generated-budget'], 'dedup': ['eval-cache', 'deduplicate'],
    'shallow': ['bit-eval', 'shallow-discount'], 'surface': ['bit-eval', 'surface-cost'],
    'fast-generated': ['flood-eval', 'generated-budget'],
    'beam': ['flood-eval', 'beam-search'], 'wide-beam': ['flood-eval', 'wide-beam'],
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(binary, name, features, source):
    dst = ROOT / 'research/bin' / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and sha(dst) != sha(binary):
        shutil.copy2(dst, dst.with_name(name + '.' + sha(dst)[:12]))
    shutil.copy2(binary, dst)
    manifest = dict(name=name, features=features, sha256=sha(dst), source=source,
                    rustc=subprocess.check_output(['rustc','--version'],text=True).strip())
    with (ROOT / 'research/bin/builds.jsonl').open('a') as f:
        f.write(json.dumps(manifest)+'\n')

def main():
    for name in sys.argv[1:] or ['fast-cache']:
        if name == 'original':
            with tempfile.TemporaryDirectory(prefix='blockfish-original-') as temp:
                temp = pathlib.Path(temp)
                archive = temp / 'source.tar'
                subprocess.run(['git','archive','-o',str(archive),UPSTREAM],cwd=ROOT,check=True)
                subprocess.run(['tar','xf',str(archive),'-C',str(temp)],check=True)
                subprocess.run(['cargo','build','--release','--locked','-p','blockfish',
                                '--features','race'],cwd=temp,check=True)
                save(temp / 'target/release/blockfish-race',name,['race'],UPSTREAM)
        else:
            features = ['race'] + FEATURES[name]
            target = {'rank-data':'blockfish-rank-data', 'world-diagnostic':'blockfish-world-diagnostic', 'public-labels':'blockfish-public-labels'}.get(name,'blockfish-race')
            source_hash = hashlib.sha256()
            sources = sorted(p for d in ['blockfish-engine','block-stacker','support']
                             for p in (ROOT/d).rglob('*') if p.is_file() and
                             p.suffix in ['.rs','.toml','.json'])
            for p in [ROOT/'Cargo.lock',ROOT/'Cargo.toml'] + sources:
                source_hash.update(str(p.relative_to(ROOT)).encode()); source_hash.update(p.read_bytes())
            subprocess.run(['cargo','build','--release','--locked','-p','blockfish',
                            '--no-default-features','--features',','.join(features),
                            '--bin',target],cwd=ROOT,check=True)
            save(ROOT/'target/release'/target,name,features,source_hash.hexdigest())

if __name__ == '__main__':
    main()
