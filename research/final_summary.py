#!/usr/bin/env python3
"""Audit the locked paired test and produce machine-readable statistics/plots."""
import argparse
import collections
import json
import math
import os
import pathlib
import random
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parent.parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()
    path = ROOT/args.input
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    groups = collections.defaultdict(dict)
    for r in rows:
        name = pathlib.Path(r['binary']).name
        assert r['seed'] not in groups[name], 'duplicate seed/variant'
        groups[name][r['seed']] = r
    assert set(groups)=={'quality-baseline','learned-fast'}
    assert all(set(g)==set(range(20000,20100)) for g in groups.values())
    assert len({(r['config'],r['move_ms']) for r in rows})==1
    summary = dict(seeds=[20000,20099], config=rows[0]['config'],
                   move_ms=rows[0]['move_ms'], variants={})
    for name, group in groups.items():
        completed = [r for r in group.values() if r['status']=='complete']
        pieces = sorted(r['pieces'] for r in completed)
        data = dict(total=len(group), complete=len(completed),
                    sha256=sorted({r['sha256'] for r in group.values()}))
        if pieces:
            search = [s for r in completed for s in (r.get('search') or [])]
            data.update(mean_pieces=st.mean(pieces), median_pieces=st.median(pieces),
                        minimum=min(pieces), p95=pieces[math.ceil(.95*len(pieces))-1],
                        mean_cpu_seconds=st.mean(r['cpu_seconds'] for r in completed),
                        mean_wall_seconds=st.mean(r['wall_seconds'] for r in completed),
                        cpu_seconds_per_piece=sum(r['cpu_seconds'] for r in completed)/sum(pieces),
                        peak_rss_mib=max(r['max_rss_kib'] for r in completed)/1024,
                        mean_generated_per_piece=st.mean(s['generated_nodes'] for s in search),
                        mean_worker_ms=1000*st.mean(s['seconds'] for s in search))
        summary['variants'][name] = data
    baseline, candidate = groups['quality-baseline'], groups['learned-fast']
    pairs = [(baseline[s]['pieces'],candidate[s]['pieces']) for s in sorted(baseline)
             if baseline[s]['status']==candidate[s]['status']=='complete']
    diffs = [c-b for b,c in pairs]
    if diffs:
        rng = random.Random(173)
        means = sorted(st.mean(rng.choices(diffs,k=len(diffs))) for _ in range(10000))
        summary['paired'] = dict(n=len(diffs), mean_delta=st.mean(diffs),
             bootstrap95=[means[250],means[9749]], wins=sum(d<0 for d in diffs),
             ties=sum(d==0 for d in diffs), losses=sum(d>0 for d in diffs))
    path.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if args.plot:
        os.environ.setdefault('MPLCONFIGDIR','/tmp/blockfish-matplotlib')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,2,figsize=(10,4),layout='constrained')
        b,c = zip(*pairs)
        lo,hi=min(*b,*c)-3,max(*b,*c)+3
        ax[0].plot([lo,hi],[lo,hi],color='#999999',linewidth=1)
        ax[0].scatter(b,c,s=22,alpha=.65,color='#237d72')
        ax[0].set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Baseline pieces',ylabel='Learned-value pieces')
        ax[0].set_aspect('equal')
        counts,edges,patches=ax[1].hist(diffs,bins=range(min(diffs)-1,max(diffs)+2),edgecolor='white')
        for edge,patch in zip(edges,patches):patch.set_facecolor('#237d72' if edge<0 else '#be7546')
        ax[1].axvline(0,color='#444444',linewidth=1)
        ax[1].set(xlabel='Learned minus baseline pieces',ylabel='Number of seeds')
        budget='50k generated nodes' if summary['move_ms'] is None else f"{summary['move_ms']} ms per move"
        fig.suptitle(f'100L Cheese: 100 held-out seeds | {budget}')
        for a in ax:a.spines[['top','right']].set_visible(False)
        fig.savefig(path.with_suffix('.png'),dpi=180)
        plt.close(fig)

if __name__=='__main__':main()
