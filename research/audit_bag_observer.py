#!/usr/bin/env python3
"""Check inference using ONLY logged public queues, Hold and chosen inputs.
No seed, RNG recreation, hidden queue, or race outcome is used to derive a type.
"""
import argparse
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser()
p.add_argument('input')
p.add_argument('--control',default='research/results/s16-control-dev.jsonl')
p.add_argument('--out',required=True)
p.add_argument('--no-parity',action='store_true',help='Skip frozen-control parity after the development equivalence audit')
a=p.parse_args()
controls={} if a.no_parity else {r['seed']:r for r in map(json.loads,(ROOT/a.control).read_text().splitlines())}
TYPES='IJLOSTZ';report=[];skipped=[]
for r in map(json.loads,(ROOT/a.input).read_text().splitlines()):
    trace=json.loads((ROOT/r['trace']).read_text())
    if 'bag_events' not in trace:
        assert Path(r['binary']).name in ('learned-fast','learned-robust')
        skipped.append(dict(binary=r['binary'],seed=r['seed']));continue
    assert r['status']=='complete' and trace['bag_valid']
    events=trace['bag_events'];inputs=trace['inputs']
    assert len(events)==len(inputs)==r['pieces']
    mask=127;revealed=0;checks=0;predictions=0;confirmed=0
    for i,event in enumerate(events):
        visible=event['visible_queue'];assert len(visible)==6
        if i==0:
            assert event['hold'] is None
            newly_visible=visible
        else:
            old=events[i-1];oldq=old['visible_queue'];old_inputs=inputs[i-1]
            assert old_inputs.count('Hold')<=1
            holding='Hold' in old_inputs
            consumed=1+int(holding and old['hold'] is None)
            assert visible[1:6-consumed]==oldq[1+consumed:]
            assert visible[0]==oldq[consumed]
            assert event['hold']==(oldq[0] if holding else old['hold'])
            newly_visible=visible[-consumed:]
            if old['inferred'] is not None:
                assert newly_visible[0]==old['inferred'];confirmed+=1
        for color in newly_visible:
            assert color in TYPES
            if mask==0: mask=127
            bit=1<<TYPES.index(color)
            assert mask&bit
            checks+=int(mask.bit_count()==1)
            mask &= ~bit;revealed+=1
        inferred=TYPES[mask.bit_length()-1] if mask.bit_count()==1 else None
        assert event['revealed']==revealed and event['remaining_mask']==mask
        assert event['inferred']==inferred
        placebo=Path(r['binary']).name.endswith('placebo') or Path(r['binary']).name=='world-rollout'
        assert event['applied']==(inferred is not None and not placebo)
        predictions+=int(inferred is not None)
    # The final lock's newly visible preview is not saved as a next decision.
    # Its deterministic number of bag-boundary checks can still be checked;
    # its last prediction is logically audited, but not counted as reveal-confirmed.
    consumed=1+int(events[-1]['hold'] is None and 'Hold' in inputs[-1])
    remaining=mask.bit_count() or 7
    checks+=int(remaining<=consumed)
    assert checks==trace['bag_checks']
    assert all(s['generated_nodes']<=50000 for s in trace['search'])
    parity=None
    if Path(r['binary']).name.endswith('placebo') and not a.no_parity:
        old=json.loads((ROOT/controls[r['seed']]['trace']).read_text())
        parity=old['inputs']==inputs and old['ds']==trace['ds'];assert parity
    report.append(dict(binary=r['binary'],seed=r['seed'],decisions=len(events),
        inferences_from_public_history=predictions,reveal_confirmed=confirmed,
        final_prediction_logically_checked=events[-1]['inferred'] is not None,
        placebo_matches_frozen=parity))
result=dict(games=len(report),passed=len(report),records=report,skipped_non_observer=skipped,
    note='Public-history audit; only final-decision predictions lack a subsequent saved reveal.')
(ROOT/a.out).write_text(json.dumps(result,indent=2)+'\n')
print('Public-history audits passed:',len(report),'inferences:',sum(r['inferences_from_public_history'] for r in report),'reveal-confirmed:',sum(r['reveal_confirmed'] for r in report))
