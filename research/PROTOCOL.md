# Experiment protocol (2026-11-08)

## Search-allocation continuation

New development cohort 62000..62011; validation 62100..62131; reserve final
62200..62263. Do not reuse prior final seeds for selection. Frozen learned-fast
is the primary control, at 50k cumulative generated nodes, no move deadline.

S1: back up every terminal successor immediately upon generation rather than
queueing it by its intermediate f score. Count every successor as before.
After finishing a parent whose children are all terminal, resume best-first
selection at another nonempty level. This tests terminal-rating coverage and
search allocation, not additional information or an increased node budget.
S2: combine S1 with leaving the last held piece unforced, as a separate ablation.
S3: revisit the previous unique-state beam with the frozen learned evaluator;
previous beam comparisons primarily used the original heuristic. Keep warmup
and all generated successors inside the same budget. Screen all complete races,
then select using validation before opening the new final cohort.

S4: with the unchanged frozen learned evaluator and original B* backup, vary
the per-piece path penalty from 10 to 5 or 20 on the same development cohort.
This changes depth allocation and the cost/value balance, not the node budget
or information. Screen these separately from eager backup before combinations.

Advance S1 alone to the 32 validation seeds at 50k: development mean196.4167
versus196.9167, eleven piece-count ties and one six-piece win. This narrow result
is not sufficient for promotion. Reject S2/S3/S4 from further tuning for now.

S5: paired counterfactual action labels. Frozen learned policy generates 80 games
on 73000..73079 at 5k. At placements20,60,100,140, a 50k frozen search proposes
its best3 distinct root moves. Clone the simulator for each and play to100L with
the frozen 5k policy, sharing the same hidden streams for labels only. Features
are the projected child board BEFORE garbage replenishment, computed from the
public board and legal ghost placement. Neither seed nor hidden future is an
inference input. A failed rollout receives a recorded500-piece penalty; no
failed branch is silently discarded. Retain full operation traces for audit.
Train on64 seeds73000..73063, choose ridge on16 seeds73064..73079. Fit action
feature DIFFERENCES to future-piece DIFFERENCES, regularized toward the frozen
model, with nonnegative board coefficients and fixed prior intercept. Evaluate
full games on62000..62011 before independent validation. No test results train
these weights. This specifically tests whether paired action labels reduce the
irrelevant future-return noise present in previous single-state regression.

S5 deployment screens: use the chosen ranking model alone and a conservative
25% ranking /75% frozen-value mixture on development seeds. The blend fraction
is fixed before full-game results; it is not a runtime use of training futures.
S6: data collection reveals some50k searches completed fewer than3 root moves.
Test one completed preview sequence per root action before global refinement:
while the depth0 heap has an unvisited root, select that level; then resume
original B*. Same value/backup, every expansion charged, no extra previews.
Implementation audit before S6 results: selecting depth0 alone guarantees root
EXPANSION, not completion, because shared deeper heaps can switch ancestry.
Keep this initial ablation as learned-covered (S6a). Add learned-root-complete
(S6b), which temporarily selects only descendants of the active root during its
first rollout, restoring all skipped heap nodes; clear that restriction on
terminal backup. Test both independently on the12 development games.

S7: direct deployment of S5 as a leaf value fails (214.25 vs196.92 pieces);
root-only reranking is a separate, better-matched hypothesis. Keep frozen B*
and learned leaf weights, retain its best3 proposed roots, and learn a residual
cost scorer using both original completed-sequence rating and projected child
features. Train64/model-validate16 on the same offline data, never game test
outcomes. Ridge10/100/1000 and override margin0/1 are selected by mean paired
rollout cost on model-validation games (game weighted equally). Apply only
before80 cleared lines and when all candidate ratings>=20 to exclude the special
bottom-clear rating regime. Reserve3 nodes from the50k search cap and charge each
projection, even though the original search already generated those root boards.
Screen development before validation; never replace the leaf evaluator here.

S8: exact known-preview endgame planning, independently of learned ranking.
When1..4 garbage rows remain and all are visible, spend at most1/5 of the move
budget on A* tracking the exact garbage prefix. Lower bound is ceil(empty cells
in remaining target rows /4); learned value only breaks equal-bound ties. Count
every attempted successor, including duplicates. If a winning known-piece plan
is found, execute the cached plan; otherwise give B* the unspent node budget.
No hidden pieces/garbage enter the solver. No extra per-move budget is granted;
disable this solver for deadline-based runs until a combined deadline exists.
Keep a placebo variant that spends the same planning work but ignores the plan.
First test12 development seeds; no changes to the frozen leaf evaluator.

S8b: the weak endgame bound yields12/12 piece-count ties; the10k budget generally
only finds the last3-placement finish. Strengthen the admissible bound with a
relaxed garbage-hole deletion problem: one move may delete any contiguous run
of up to4 equal hole columns, then remaining rows close up. For dense one-hole
garbage rows, any tetromino crossing multiple such rows must use their same hole
column, so this relaxation includes every legal deletion while ignoring access,
known piece types, and blocking material above. Dynamic programming gives a
lower bound, not a learned estimate. Fall back to empty-cell/4 if rows are not
single-hole garbage. Keep all budgets, activation and fallback unchanged.

Advance S8b to32 validation games: development196.75 vs196.9167, one two-piece
win and11 ties. S8c adds a known-queue/Hold relaxation to the bound: O cannot
fill a single-hole dense garbage row. Compute the minimum placements needed to
use at least the hole-deletion bound's number of non-O pieces, allowing all
other non-O types to act optimistically. Unknown pieces are never added. Keep
the same10k endgame share and test S8c independently before choosing a finalist.

S8d: the last4-row planners mostly match already-optimal short finishes. Apply
the queue-aware planner from at most7 fully visible remaining garbage rows,
still using only the initial current+previews+hold and10k of the50k node cap.
Prune an abstractly impossible root immediately (charged one node). Preserve a
matching wide-placebo budget-control variant. Screen12 development games first;
the reserved final cohort remains unopened.

S9: direct policy optimization, not regression MSE. Freeze training seeds
74000..74031 and5k cumulative nodes. Tune only the six previously nonzero
board coefficients [piece estimate, aggregate height, covered rows, row/column
transitions, wells], each within exp(+/-0.7) of its frozen value; keep bias,
other zero coefficients, search, information, and per-piece cost unchanged.
Six deterministic mirrored hill-climb generations, four Gaussian directions
and their negatives per generation, sigma=max(.08,.25*.8^generation), RNG173.
Every candidate runs ALL32 seeds; rank completion count first, then mean pieces.
The incumbent is retained and results cached by exact weight and binary hashes.
No early stopping on partial means. This is tuning only; a frozen selected
model must pass full50k development and independent validation before final.

## Continuation: budget robustness and policy evaluation (2026-11-09)

All previous final seeds remain sealed against tuning. Diagnose budget response
on fresh seeds 50000..50015, using both frozen quality-baseline and learned-fast,
at 50k and 200k cumulative generated nodes per placement, no time deadline.
This cohort is diagnostic/validation, not the next final test. Keep every run,
including failures; per-game timeout 600 seconds. Concurrent fixed-work runs
may share CPU, so report CPU rather than claiming isolated wall speed.

New hypothesis: fitting costs from the old policy leaves a policy-distribution
mismatch. Collect the frozen learned policy at 5k on fresh seeds 6000..6159;
fit on 6000..6127, choose regularization on 6128..6159, keeping the same board
features and target definition. Compare full games on development seeds 0..11
before any model promotion. Also test learned terminal-only backup separately
from refitting. Reserve seeds 51000..51031 for a subsequent locked final 200k
comparison if a finalist is chosen; never train on these games.

R5: a separate 16-step bootstrapped target on the same policy2 training split:
16 + frozen V(next board) minus the cost of garbage revealed during those steps.
Use the training-only garbage price; validate ridge against Monte Carlo returns
on model-validation games. Do not replace the frozen general-use model unless
full-game validation supports it. No training targets use test-set outcomes.

### High-budget final lock

The central budget-robustness question is evaluated independently of the new
training experiments: freeze original `quality-baseline` and prior `learned-fast`
at 200k cumulative nodes, seeds 51000..51031, 32 paired full games. These exact
binaries are unchanged from the preceding round. Start this test without viewing
R1 aggregate outcomes, and do not inspect its aggregate results until all 64 runs
finish. New R2/R5 models cannot be selected using this cohort; any promotion of
those models requires another untouched test cohort. All runs have a 600-second
per-game timeout, no per-move deadline. Concurrent runs only affect runtime,
not the deterministic fixed-node decisions.

Before inspecting R6 aggregate outcomes (52/64 records present, only completion
counts inspected), add the same two frozen binaries at 50k on the same 32 seeds.
This enables a direct paired budget comparison and a difference-in-differences
estimate, rather than comparing different seed cohorts. This addition is not
conditioned on significance or observed piece counts. Keep both batches whole;
do not extend seed count after unblinding.

R7: the 16-step candidate scores 197.833 versus frozen 198.083 on 12 development
games, an inconclusive 0.25-piece improvement. Freeze its fitted weights and
evaluate all 16 diagnostic seeds 50000..50015 at 200k against the existing R1
frozen-model results. This is validation, not a final promotion test. The policy2
Monte Carlo refit is not advanced after 198.500 on development games.

R8: distinguish clearing currently visible garbage from finishing the race.
Retain residue value after a visible-bottom clear while unseen garbage remains;
keep the old shortest-finish rating when all remaining garbage is visible.
Use only public cleared-line count and visible garbage count, no future holes.
No change to legal placements, queue, or search horizon. First test development
seeds 0..11 at 50k with the frozen evaluator. This differs from the previous
unconditional goal-residue ablation, which also penalized final-race residue.

Upstream commit: `739953f9331b7648f269fc52cff5aae330108ccd`.
Rust 1.75.0, release profile, locked dependencies. Environment has a 4-core
cgroup CPU quota. All 64 upstream workspace unit tests passed before changes.

## Fixed rules and information

Use the upstream block-stacker simulator and SRS placement/finesse infrastructure,
10 columns, 5 previews, current piece and hold; no future garbage or hidden queue
is passed to the AI. Target is **100 garbage rows**, not 100 arbitrary clears.
Explicit `-g 9 -G 4` matches the actual upstream defaults (CLI help incorrectly
claimed 2). Seven-bag and nonrepeating adjacent garbage holes use upstream
Pcg64Mcg. This is a local simulation benchmark, not verified Jstris replay parity.
Piece and garbage generators clone the same initial PRNG: keep that unchanged
for paired comparisons, but note it is not an independently sampled Jstris RNG.

## Splits and budgets

- Development seeds: 0..11. Used for selecting hypotheses and weights.
- Validation seeds: 1000..1023. May be used to decide finalists.
- Locked final test seeds: 10000..10099. Do not tune using their results.
- Any later record hunting must be reported separately from held-out averages.
- Stage 1/2 screening: 5,000 frontier nodes per placement, complete 100L races.
- Confirm at 50,000 (the upstream default), and report cost as well as quality.
- Timeout is a failure, never silently omit it. Completion rate precedes mean
  pieces; means and paired intervals refer to completed runs only.
- Upstream `node_count` is frontier occupancy, not cumulative generated nodes:
  `pop` decrements it. Keep this exact budget for weight/rating/cache comparisons.
  It must not be called an equal total-work or equal-time budget.
- Randomize task order with fixed seed; store command configuration, executable
  SHA256, seed, status, trace, wall time. Compare CPU costs in isolated serial
  runs because concurrent experiments share the quota.

`bench.py` runs real executable processes, preserves errors, supports resume and
records one JSON object per race. `summarize.py` reports paired bootstrap intervals
(10,000 resamples, fixed seed); intervals do not correct for tuning-selection bias.
Raw per-piece clear traces are retained beside JSONL results.

## Hypotheses

1. Lower height penalty may let residue-based piece estimate guide efficiency;
   zero height penalty may cause unstable stacking. Sweep one factor at a time.
2. Exact static-score memoization can avoid evaluating transposed matrices without
   changing legal placements, queue information, search ordering or decisions.
3. Parent+terminal score smoothing may discourage good final placements; test
   terminal-only ratings with identical heuristic and budget.
4. Rewarding only depth after clearing the visible bottom row ignores residual
   stack when more garbage remains; test retaining residue score at that boundary.

Experimental Cargo features `eval-cache`, `terminal-only`, `goal-residue` keep
ablations reproducible. Only exact `flood-eval` + `eval-cache` optimizations are
promoted to defaults; use `--no-default-features` for unoptimized ablations.

## Final evaluation lock

Before opening seeds 10000..10099, freeze original B* rating and weights
`5,10,10,10`. Final candidate changes only the implementation of static evaluation
(bitwise I dependencies, bitset flood fill, exact matrix-score memoization).
Compare instrumented original versus candidate:

- 5k frontier limit, no deadline: 100 paired full races; compare every input and
  every clear trace. Original archived binary is also retained for audit.
- 50ms soft search-worker wall deadline plus 50k frontier cap: 100 paired races,
  interleaved randomly across variants with four workers on a four-core quota.
  No other CPU experiments/builds run during this batch. Report actual worker
  milliseconds and generated nodes; deadlines are checked every 64 steps after
  at least one finished sequence. Timing-mode decisions are not deterministic.
- Test results will not be used to retune weights or choose a different search.

The 50ms validation set includes wide-beam to diagnose deadline robustness,
but that algorithm is excluded from the final candidate after observed top-outs.

## External record verification

README links historical Jstris replays 53445238 and 54532835 but gives no numeric
record. Search finds historical claims of 170 pieces; this is not evidence of
the current strongest record. Direct leaderboard/replay reads through web search
failed. Do not claim a world record without a verified current benchmark and a
legally replayable run under matching rules.

## Quality-first continuation

Following the user's clarification, execution speed is a secondary criterion.
The previous independent timing study is complete: 202.79 versus 202.33 pieces,
paired delta -0.46 with 95% bootstrap interval [-1.94, 1.04]. This does **not**
establish lower piece counts. Keep its 100 seeds separate from subsequent tuning.

New quality experiments use no wall deadline and a fixed cumulative generated
node cap, including warmup, failed branches and duplicates. Original and candidates
all use the same visible information, exact evaluator implementation and simulator.
Development remains seeds 0..11 initially; validation remains 1000..1023, with
additional validation seeds 2000..2063 available if selection needs a larger set.
Reserve **20000..20099** as a fresh final quality test set; do not inspect candidate
performance on it until the strategy is frozen. A minimum-cost record hunt is a
separate activity and cannot replace distribution-level evaluation.

Hypotheses in this phase:
- Stop when only held material remains, rather than forcing the final held piece.
- Rate every completed beam leaf with its terminal score; do not discard it by a
  different intermediate priority.
- Preserve distinct board/queue/hold states in the beam rather than many equivalent
  paths, and test width versus depth allocation at fixed total generated work.
- An available held I piece may cover one I dependency; test this queue-aware
  correction separately from horizon and beam changes.

### Quality finalist lock, 2026-11-09

Validation 1000..1023 selects `learned-fast`, the pure learned value function,
without global transposition pruning or hold-horizon changes. All 24 complete;
197.2083 pieces versus baseline 202.2917; paired delta -5.0833, bootstrap 95%
[-9.0417, -1.5417]. The mixed model is weaker at 198.9167. Freeze weights from
`value-fit.json` (training 3000..3127, model-validation 3128..3159), all search
behavior and parameters before opening quality test seeds 20000..20099.

Final tests on these 100 seeds:
1. 50k cumulative generated nodes, no deadline, baseline versus learned-fast.
2. 50ms per-placement soft worker deadline plus the same 50k generated cap,
   baseline versus learned-fast. Four interleaved workers, no other CPU experiments.
Both batches record completion, pieces, generated nodes, CPU, wall time and peak
RSS. `measure.c` obtains per-process resource usage through wait4. Test outcomes
will not be used to adjust the model or select a different variant.

### Separate local record budget scan

After both final batches finish, scan the frozen learned model on development
seed 3 at cumulative budgets 1k, 2k, 3k, 4k, 5k, 6k, 8k, 10k, 15k, 20k, 30k,
50k. This seed was selected because it previously achieved 178 pieces at 5k.
This is explicitly selected-seed record hunting, not a new independent test or
a basis for selecting the general-use budget. Preserve every result, including
worse runs; replay the complete batch. No model or search changes are allowed.

### S9 frozen candidate validation, 2026-11-10

Training finished all 49 models on seeds74000..74031 at5k. Freeze selected
weights hash00f5e0788d2a693811aa7f6a273e9760948d7c69d00c9f21fe7cb9956ca2a360.
Static learned-direct is compared on development62000..62011 at50k, then
validation62100..62131 at both5k and50k against frozen learned-fast.
No retuning on these results. Reserve final62200..62263 unopened; only promote
a supported candidate after a fresh paired final. Completion precedes pieces;
record CPU costs and paired uncertainty, and report budget-specific failures.

### S10 predeclared shrinkage experiment

S9's 50k development mean198.5 does not improve196.9167. Independently of
unfinished validation, test exactly one regularized candidate: arithmetic
midpoint between frozen prior and S9 weights, unchanged B*,50k, development
62000..62011. Hypothesis: reduce training-specific weight displacement while
retaining useful full-game adjustment. This is development selection, not an
independent performance claim. If worse, reject without opening final seeds.

### S11 structural backup experiment (2026-11-10)

Freeze learned-fast weights and all B* expansion priorities. Compare each root
by mean rating of its best4 DISTINCT terminal states (or all available if fewer),
rather than best single sequence. Deduplicate by complete State, including queue
and Hold. Return best sequence for that root. Permit score increases as support
arrives; preserve exact bottom-goal preference. No extra generated nodes or future
information; runtime/memory overhead recorded. This is a heuristic robustness
hypothesis, not an unbiased expectation or chance-node estimator.
Development62000..62011 at50k. If promising, lock and validate new seeds75000..75047
at50k with freshly interleaved frozen control. No use of62100 or final62200 for
tuning. Test fixed4 once; do not search k on this development sample.

S11 validation lock: development12/12 complete,195.3333 vs196.9167,
paired delta-1.5833 CI[-4.8333,1.9167]. Freeze learned-robust before opening
75000..75047. Run both binaries interleaved,50k generated nodes,48 seeds each.
No parameter or implementation changes based on interim results. Formal final
62200..62263 remains sealed. If validation supports improvement, use the full
predefined64-seed final cohort; no early stopping or sample expansion.

S11 implementation placebo: collect the same4 distinct terminal states but
return their minimum instead of mean, with the same quarter-unit scaling.
Run12 development games; compare complete operation traces to frozen control.
Purpose: isolate averaging from collection overhead, trace handling and score
rescaling. No validation selection based on placebo.

S11 validation gate clarified before inspecting any validation piece counts:
advance to the fixed64-seed final only if completion is no worse and mean pieces
is lower than control. Validation CI is reported even if it spans zero; this is
candidate selection, not proof. Require fresh final evidence before promotion;
never expand the final cohort after viewing its outcomes. No interim comparisons.

S11 final lock: validation48/48 both complete,196.5625 vs197.4791667,
delta-0.9166667 CI[-3.5833333,1.6875]. Gate satisfied on mean, not a proof.
Open all final62200..62263 now, frozen learned-robust versus learned-fast,
50k cumulative generated nodes,4 interleaved workers. No other race experiments
in parallel. Promote only if completion is no worse and the fresh final paired
95% bootstrap interval for candidate-minus-control is wholly below zero.
Otherwise retain as unproven research branch, regardless of pooled results.

S11 budget diagnostic locked before viewing final outcomes: after the50k final,
run both frozen variants at5k on the same48 validation seeds75000..75047.
Compare with existing50k validation for a paired budget interaction. This tests
whether averaging support requires enough completed sequences. It cannot replace
failed final evidence or tune k; do not promote from this exploratory diagnostic.

### S12: coverage with robust backup, 2026-11-13

S11 final passed its predeclared gate; preserve original learned-fast and frozen
learned-robust binaries. Low-budget validation5k is worse in point estimate, not
significant. Hypothesis: average backups require more balanced root evaluation,
whereas forcing root coverage previously hurt minimum backups. Test existing
root-complete (one full preview per root before refinement) with robust-backup.
No retuning of k, weights, budgets, or coverage policy. Development62000..62011,
50k; compare against robust195.3333 and original learned196.9167. If development
improves over robust, validate frozen candidate on new75100..75147; reserve
75200..75263 for a subsequent final only. Never tune on opened62200 final.

S12 failed development gate:197.0 vs robust195.3333, all12 complete. Do not
advance or tune coverage.

### S13: exact finish with robust midgame

Combine frozen robust backup with existing endgame-wide (at most7 fully visible
remaining garbage rows, fixed1/5 budget share, known-preview exact A*). All costs
remain inside50k; failures return unused budget to B*. No bound/trigger changes.
Hypothesis: exact remaining-piece objective can complement robust midgame choice.
Development62000..62011; advance only if mean lower and completion no worse than
robust alone. If advancing, interleave both on unused75100..75147, with75200..75263
reserved. Never reuse opened62200 final for selection.

### S14: uniform one-step future at preview boundary, 2026-11-13

Freeze robust baseline. For each non-goal terminal, evaluate best legal placement
for each of7 tetromino types on its board; replace terminal h with one-piece cost
plus mean best successor h. Keep parent-f smoothing, robust backup and legal
placement code. Hypothetical types use a fixed uniform prior, NOT the actual RNG,
seed, hidden queue, or correct conditional7-bag posterior. No new hold choice or
garbage replenishment is modeled in this tail approximation. Unknown placements
are evaluation only, never added to executed traces.
Charge each of7 hypothetical roots and every attempted child as generated work.
One state per step allows strict50k termination; discard incomplete7-type averages.
Control placebo performs identical tail work but keeps original terminal rating.
Both run12 development seeds62000..62011 at50k, versus frozen robust195.3333.
No tail weight/type probability/horizon tuning. Advance only if completion no
worse and mean better than robust, then use reserved75100 validation.

S14 failed: expectation199.9167,workplacebo197.75,robust195.3333; all12 complete.
Do not tune its prior weights/horizon on these results.

### S15: stop before forced Hold; explicit unknown-current/Hold choice

S14 evaluates after forcing all known pieces including held piece. Instead use
existing preserve-hold: when the preview queue is exhausted leave the held piece
available. Each uniform hypothetical next type now competes with known Hold via
the original legal placement generator. Use f(boundary)+E[f(next)] as the smoothed
leaf score, so the adjacent evaluations actually flank the unknown decision.
If no Hold remains, still approximate one unknown placement without empty-Hold
redraw. Uniform prior and frozen evaluator retained; no actual future inspected.
All roots/children charged and incomplete averages discarded as inS14. Include
work placebo with identical boundary/work but original boundary score.12 development
seeds62000..62011,50k,advance only if mean improves over robust and completion
no worse. Unopened75100 validation and75200 final remain reserved.

S15 failed development:202.0 vs robust195.3333;workplacebo208.3333.
All48 S14/S15 runs completed and replayed. Do not advance;75100/75200 remain
unopened. Preserve robust profile; no probability/horizon/weight tuning.

### S16: diversity by second decision, 2026-11-13

Hypothesis: four distinct terminal states can still belong to the same second
action, overstating continuation flexibility. Retain one best terminal per
second placement index within each first move, average the best4 such branches.
Leave k=4, learned weights, goal preference, B* expansion, budget and information
unchanged. Root-conditioned second placement indices refer to the same state,
so they are consistent action identifiers. Distinct branches can converge to
the same board: this tests decision flexibility, not independent random samples.
No weight/probability/count tuning. Preserve all frozen robust and prior binaries.

Avoid further development selection on heavily reused62000: new development
76000..76031,32 seeds,50k,4 interleaved workers with frozen learned-robust.
Advance only if completion no worse and mean lower. If advancing, freeze and
validate75100..75147,48 paired games. Only if both stages improve mean, open
final75200..75263,64 pairs; promotion requires no worse completion and wholly
negative fresh final95% paired bootstrap CI. Do not add final seeds after results.

S16 mechanism probe: on76000,76001,76002 only, run frozen robust scoring with
read-only support statistics before analysis completion. Count fully supported
roots, distinct second actions among their4 leaves, and roots whose4 leaves all
share one second action. Verify complete operation parity with frozen controls.
No candidate modification based on these diagnostic observations.

S16 rejected:32/32 losses,232.78125 vs197.1875,delta+35.59375. The concentration
exists but averaging controllable second choices is harmful. Do not tune k or
diversity pressure. Mechanism probe matched all operations on3/3 games.

### S17: hierarchical min of supported means

Keep the known second action a choice: per second action retain4 best distinct
terminal states, compute their robust mean, then minimize across second actions.
Only branches with4 supports qualify; if none qualify use original global robust
backup. Preserve existing bottom-goal preference. No new nodes or future info,
no changed weights or branch count. This specifically separates controllable
actions from heuristic uncertainty afterS16 disproved forced diversity.
Development remains76000..76031 (now development, not held out),50k. Compare
frozen robust. Verify refactored default reproduces full traces on76000..76002.
Advance only if completion no worse and mean lower; untouched75100 validation
and75200 final, same48/64 sizes and gates asS16. No extra k/fallback tuning.

S17 validation lock: development32/32 complete in both groups,195.40625 vs
197.1875,delta-1.78125 CI[-4.34375,0.875],21 wins1 tie10 losses. Freeze
robust-hierarchy and run both variants interleaved on75100..75147 at50k,
4 workers. No candidate changes or interim comparisons. Formal75200..75263
remains sealed; original gate applies (validation completion no worse and
mean lower before final, final negative95% interval before promotion).

S17 additional frozen control, locked before inspecting validation outcomes:
S11's formal improvement was modest with CI near zero. Add learned-fast to
75100..75147 as an independent replication of robust versus its predecessor.
This third arm runs separately at the same50k generated cap, no deadlines;
CPU/wall scheduling differences must not be treated as strategy gains.
The validation gate is tightened: hierarchy must have no worse completion and
lower mean than BOTH learned-robust and learned-fast. If final opens, interleave
all3 frozen variants on75200..75263,64 seeds each. Promotion requires no worse
completion and wholly negative paired95% intervals against BOTH controls.
No new candidate or parameter is selected from the third arm. No seed counts
expanded; no validation piece counts were inspected before this amendment.

S17 validation failed:hierarchy198.1875 vs robust197.75 and legacy198.875,
all48 complete. Hierarchy-minus-robust+0.4375 CI[-1.7917,2.7083]. No final opened.
Robust replication versus legacy-1.125 CI[-3.5208,1.25], direction consistent
but not independently decisive. Stop changing backup averaging rules.

### S18: exact inference from public seven-bag history

Track only public initial current+5 previews and newly revealed previews after
one draw per lock or two draws when filling an empty Hold. Fresh standard
seven-bag is verified against simulator rules. If six types of the current bag
have appeared, the seventh is logically determined: append that single inferred
piece to the search queue. No seed, RNG state, hidden queue, or future garbage
is consulted. The observer API accepts colors only; invalid streams disable it.
Both arms receive/log the same public information and history; placebo tracks
inference but does not append. Effective horizon differs only through a
provably deduced type, not an extra visible preview. All search work remains
inside50k; log original six-piece public queue and inference separately.

Development76000..76031,32 paired games with robust-bag-placebo. Require
full placebo input/clear parity with frozen robust controls and independent
observer audit from logged public queues/actions. Advance only if completion
no worse and mean lower. If advancing, NEW untouched76100..76147 validation
against observer-placebo and learned-fast;75200..75263 final still sealed,
all three arms if final opens. Same mean/completion gates and final negative
paired95% intervals against both controls. Never reuse75100 for selection.

S18 validation lock:development32/32 both complete,194.625 vs197.1875,
delta-2.5625 CI[-4.84375,-0.375],21 wins2 ties9 losses. Public-history audit
verified1832 inferences,1818 against a subsequent reveal;14 final decisions
only logically verified. All32 observer-placebo input/clear traces equal the
frozen robust controls. Freeze both binaries, run NEW76100..76147 (48 seeds)
with robust-bag,robust-bag-placebo,learned-fast interleaved at50k/4workers.
No candidate changes from interim results. Require no worse completion and
lower mean than both controls before opening75200..75263 final; require
negative final95% paired intervals versus both to promote. Same64 final seeds,
no expansion after viewing outcomes. Bag-observer validation audit can omit
repeat frozen-control parity; it must still check all public histories.

S18 final CI gate also requires all64 pairs complete for each control; otherwise
no promotion from a completer subset. This conservative condition is locked
before validation outcomes are inspected. decide_cohort.py implements both
fixed mean-selection and full-pair CI-promotion gates from saved lock manifests.

S18 validation complete: all three 48/48; candidate195.645833, placebo195.854167, legacy198.125. Candidate-minus-placebo -0.208333 CI[-2.416667,2.020833], candidate-minus-legacy -2.479167 CI[-4.520833,-0.395833]. Fixed mean gate passed, not proof versus robust. Freeze unchanged hashes; open reserved75200..75263, three arms64 each,50k,4workers. Final CI gate fixed before outcomes.

S18 final failed promotion: candidate195.046875 vs robust195 and legacy198.40625, all64/64. Candidate-minus-robust+0.046875 CI[-1.90625,1.953125]. Final75200..75263 retired, do not tune.

S19 diagnostic ablation, before outcomes: isolate preserve-hold alone on cheese-robust, without eager terminal propagation, speculative tail, or bag inference. Earlier S2/S15 combined boundary change with other search changes. Hypothesis: forcing the last held piece at the preview boundary creates artificial forced-use value; retaining Hold may help even without hypothetical unknown types. No new weights, k, or node budget. Development76000..76031,50k; compare existing frozen robust same seeds (s16-control-dev.jsonl), historical CPU not a speed comparison. Require completion no worse and lower mean before NEW76200..76247 validation, three arms incl frozen learned-fast. Only if validation passes, reserve NEW75300..75363 final, negative paired95 intervals versus both controls. No S18 final reuse, no per-seed tuning. If isolated hold preservation fails, stop this boundary-only direction rather than tune smoothing.

S19 complete/rejected:32/32,200.8125 vs197.1875,delta+3.625 CI[0.75,6.5],10 wins5 ties17 losses.65 tests and32 full replays passed. Initial sequence coverage738.06→1149.94 did not improve pieces. Stop boundary-only tuning.76200/75300 remain unopened. See HOLD_REPORT.md.

S20 preregistration: investigate Hold-conditioned residual value, not another boundary threshold. Add only public Hold to BoardSample exports; no inference/search change in context-source (cheese-quality). Collect NEW76400..76559 using5k frozen learned policy and identical100L rules,128 training76400..76527,32 prediction validation76528..76559. Fit one fixed ridge10 model on centered7-type one-hot Hold interactions with [bias,height,piece_estimate,i_dependencies,wells], subtract frozen learned board value from actual remaining cost adjusted for unseen garbage using training-only price. Equal weight per game, exclude unseen<20. No regularization sweep, no final-set access. No-Hold correction zero; mean over7 held types zero by construction. Assess game-weighted validation MSE and paired game errors; predictive improvement alone never promotion. If MSE improves, freeze model and test cheese-robust+preserve-hold+context against both frozen robust and S19 same32 development games. Known distribution mismatch: labels from actual current+5 states, search leaves may contain only Hold. Full-game outcomes decide; do not claim causal Hold value from observational labels.

S20 predictive gate failed: trainingMSE38.31254→38.17633, heldout40.28499→40.32575, game delta+0.04076 CI[-0.15196,0.24548]. Do not integrate the MC residual into search.160 source games complete,3 source input/clear parity games passed.
S21 before new outcomes: test label noise with fixed7-step frozen-value bootstrap target, same128 training games76400..76527, samefeatures/ridge10, NEW32 prediction validation76600..76631. Seven steps motivated by bag length, no horizon sweep. Sourcepolicy5k learned-fast via paritychecked context-source. Unseen adjustment as earlier TD, no current future inputs. Prediction-MSE gate unchanged; if improved freeze model and test full100L at50k on existing32 development seeds against S19 and frozenrobust. Prediction gain is not quality evidence; distribution mismatch remains.

S21 predictive result: training2.770994→2.695545, newvalidation2.735080→2.722706,delta-0.012374 CI[-0.041443,0.017138]. Meets fixed mean gate only. Freeze weights. Apply Hold correction at each evaluated state outside board-only cache, using actual occupied Hold (emptyHold secondpreview is not held). No extra searched nodes, same50k, no hidden inputs. Fullgame development76000..76031 versus frozenrobust and S19preservehold; require lower mean than both before any new policy-validation seeds.

S21 fullgame failed:32/32,198.15625 vs robust197.1875 (+0.96875 CI[-1.875,3.90625]), preservehold200.8125 (-2.65625 CI[-5.34375,0.0625]).32 replays passed, no independent policy validation opened.
S22 fixed-model factorial ablation before outcomes: same S21 weights, original robust horizon, add Hold corrections to evaluated states. Terminal known queues normally exhaust Hold, so effect is primarily internal search ordering and smoothed parent value, not retaining a held leaf. Do not claim purely ordering because parent_f contributes to ratings. Remove automatic preserve-hold dependency from hold-context; S21 build explicitly retains preserve-hold and its frozen binary unchanged. NEW robust-hold-order profile. Same32 development76000..76031,50k, require completion no worse and lower mean vs frozenrobust and S21 before newvalidation. No coefficient/horizon fitting.

S22 development passed fixed gate:32/32,195.96875 vs robust197.1875,delta-1.21875 CI[-4.125,1.90625]; vsS21 -2.1875 CI[-4.78125,0.4375].32 candidate replays plus3 sourceparity replays passed. Not proof of improvement. Freeze candidate/model; open NEW76200..76247,48 each candidate/learned-robust/learned-fast interleaved4workers50k. Require completion no worse and meanlower than both to open reserved75300..75363 three-arm64 final. Final requires negative95 pairedintervals versus both and all64 pairscomplete. No final expansion or parameter changes.

S22 independent validation complete: candidate196.6875, robust197.833333, legacy198.354167, all48/48. Delta vsrobust-1.145833 CI[-3.520833,1.4375]; vslegacy-1.666667 CI[-4.375,1.125].144 replays and strict budget metadata audit passed. Fixed mean gate passed, no improvement claim yet. Open reserved75300..75363,64 per arm, unchanged hashes/weights,50k/4workers. Final gate negative95 paired intervals vsboth, all64pairscomplete, no sample expansion.

S23 exact implementation experiment during frozen S22 final: cache all7 Hold corrections per board for a single analysis, reset with Evaluator. Each component uses identical arithmetic/order to original; no decision, weight, queue, node accounting or model change intended. Test direct/cache across boards/types/repeats then fullinput/clear parity on development76000..76002 against frozenS22. Do not replace S22 final binaries mid-cohort. Only consider cache if exact equality and cost benefit; not new piece-count evidence.

S23 exact cache pilot complete:69 tests,3/3 fullinputs/clears/search-count parity,6 replays. CPU22.62817→20.30587 sec/game; RSS26209→30996KiB. Exploratory cost only, no piece-count improvement and no frozen-final binary replacement.

Conditional follow-up fixed before S22 final inspection: only if the final CI gate passes, run a budget diagnostic at200k on first16 validation seeds76200..76215, frozen robust-hold-order and learned-robust,32 games,4workers,timeout600s both. Reuse exactly these16 already-recorded50k pairs for paired budget interaction; not a new independent quality cohort, not200k promotion evidence. No model or seed selection from per-seed outcomes, no new final set used for tuning. If final fails, do not increase budget to rescue it. Same public information and generated-state accounting.

S22 final FAILED: all3x64 complete, candidate196.546875 vs robust196.21875 and legacy198.671875. Candidate-minus-robust+0.328125 CI[-1.515625,2.15625], candidate-minus-legacy-2.125 CI[-4.25,0].192replays and exact per-decision metadata audit passed. Keep cheese-robust; no sample expansion or200k rescue. Robust-minus-legacy-2.453125 CI[-4.171875,-0.671875] independently supports existing profile only. S20-S23 total601newraces/601replayed,336uniqueIDs including reuseddevelopment,8033.862064CPUsec. Stop Hold residual tuning.76700..76747/75400..75463 remain unused for future different hypotheses.

S24 preregistration before new outcomes: replace shared global-f allocation with independent per-root B* plus sequential halving. Reuse frozen learned evaluator, robust4 backups, terminal preference, exact states/SRS/Hold/traces. Enumerate/count legalfirst placements; scout each root in initial static-score order until one completed continuation or exhaustion; then equal generated-node quotas per surviving root each round, ceil-half retained by latest robust rating, remaining budget divided overceil(log2(n)) rounds. Eliminated roots removed from Analysis, not retained with stale optimistic ratings. No new types/hiddenfuture/RNG, no model/weight change, shared50k total, every step generates at most1 node. Scout may exhaust budget; report its cost/rated roots. Fresh DEVELOPMENT76800..76831 (retire repeated76000 for candidate choice), vs frozenlearned-robust,32 pairs,50k4workers. If development mean improves without completionloss then NEW76700..76747 validation incl learned-fast; only then reserved75400..75463 final, negative95 pairedintervals vsboth, no expansion. Current non-halving code control must match frozen robust on old3development traces; never overwrite frozenbinaries.

S24 failed development:32/32 both, halving199.15625 vs robust196.21875,delta+2.9375 CI[0.0625,5.8125].All67 replays passed. Scout completed every decision, mean21.6625% budget,45.73rated roots,1.512survivors,599.52complete sequences. Coverage increased but no quality gain; this alone does not identify causal failure.
S25 preregistered structural correction: shared B* first25% generated budget (delay cut only until a rated root exists), retain ceil-half of completed roots by current robust rating, prune other frontier nodes and remove their recommendations, continue SAME nodes/current placement iterator/robust leaf supports/cache. No forest restart or repeated root exploration; remaining75% original global-f search within retained roots. All attempted states incl pruned charged cumulatively. Fixed fraction, no sweep, evaluator unchanged. Same development76800..76831, frozenrobust controls reused, no timing equivalence claim; require candidate meanlower and completionno-worse before NEW76700 validation and75400 formal. S24kept as negative structural comparator, not promotion reference.

S25 failed:32/32 complete,198.8125 vs196.21875,delta+2.59375 CI[-0.03125,5.1875],11wins2ties19losses.32 replays passed. Stop hard-elimination/focus-ratio sweeps; keep frozenrobust. Next S26 hypothesis: use original complete simulator for a few sampled conditional public futures, including garbage replenishment, to rank top root candidates. Must build explicit public observation constructor, never clone actualhiddenRNG for online evaluation. No new policy holdout opened yet.

S26 fixed design before policy outcomes: public-only sampled worlds, original simulator transitions incl replenishment. PublicObservation contains coloredcells,current,5previews,held,remainingbag from public reveal history,garbage_generated=cleared+visible,and last publicly revealed garbagehole. Constructor takes NO realgame/RNG, independent piece/garbage planner streams. Fixed FNV hash of canonical public JSON +worldindex gives2 common world seeds across2 root candidates. Base search75% of50k; each candidate×world simulates4 locks incl chosen first action, remaining3 decisions share leftover budget (1040 each for2candidates); charge each virtualroot and each lock state as well as all AI-generated nodes. Endpoint=locks+frozen boardvalue+2.083984375*unseen garbage (original training price); completed goal useslocks only,topout1000+locks; missing rollout suggestion truncates to endpoint, not fake death. No truefuture input. Same-work placebo chooses originalfirst candidate. Develop76800..76831: candidate/placebo/frozenrobust, need beat BOTH controls with no completionloss before NEW76700..76747 three-arm validation; then75400..75463 final64 with negative95 pairedintervals vsboth andallcomplete. No sample/horizon/budget-split sweeps. Synthetic RNG independence intentionally avoids exploiting original simulator stream correlation.

S26 integration checks: 69 engine +7 simulator tests passed; both smoke games on old76000 completed and public-history/world replays passed. Before development: reuse unchanged32 frozenrobust records from s24-control-dev.jsonl; run two new arms interleaved4workers. Historical control CPU is not an equal-wall-time comparison. No parameter changes from smoke outcomes. Lock s26-dev-lock.json.

S26 development FAILED: candidate209.96875, placebo198.1875, frozenrobust196.21875, all32/32. Candidate-minus-placebo+11.78125 CI[8.78125,14.90625]; vsrobust+13.75 CI[11,16.375].64actual/public/world audits passed; no new validation/final opened. No sample/horizon/split sweep. Post-hoc offline diagnostic compares first simulated follow-up with saved37.5k placebo action only when board/current/next/Hold exactly match; normalize input paths by resulting state. This tests policy mismatch, not which choice is optimal, and is not a new quality gate.

S27 preregistration before outcomes: root-local nonterminal transpositions keyed by full State, depth and first-action ID; ignore parent_f only here because succ overwrites it from equal current f. All terminal ratings/robust4 unchanged. Charge duplicate node generation, skip its descendants on expansion and reselect globally. Observer-only profile must match frozen robust inputs/ds/counts on76000..2. Exhaustive small-board root-rating equivalence and legal traces required. Fixed32 development76800..31 vs existing frozen robust s24-control-dev.jsonl,50k,no deadline; lower mean/completion no worse before unopened76700..47 three arms incllegacy, then75400..63 formal64 strict dual negative paired95 intervals/full completion. No key/budget sweep. This differs from earlier failed global dedup by preserving root support; does not presume it will improve. S11/S26 checkpoint hashes verified before experiments.

S27 development FAILED:32/32,197.28125 vs196.21875,delta+1.0625 CI[-1.71875,3.8125],14wins4ties14losses.68tests,3observer exactparity,35actual replays and generation audit passed. No independent validation opened. Stop transposition-key sweeps; duplicate expansion removal alone did not improve quality.

S28 bounded label diagnostic, before collecting new outcomes: NEW training-only76900..76907, frozenrobust50k full source games. At exactly100 locked pieces reconstruct public observation using only observed history; no selecting states by score/disagreement. Base robust50k proposes up to2 roots. Four common public-derived worlds per root, domain-separated piece/garbage generators asS26. Each continuation uses unchanged50k robust policy, at most400 locks inclroot, or until100L/topout; failures recorded1000+locks, never discarded. Maximum64 full counterfactual paths. Save every operation and search count; simulated unknown streams never taken from actual source RNG. Also measure endpoint after4 locks on SAME strong-policy trajectory, and independently S26-style1040-node short-policy4locks on sameworld/root. This separates policy mismatch from truncation/endpoint issues descriptively, without fitting anything. Compare root cost differences across worlds0/1 and2/3, and short/full ranking. Eight source games are independent units, worlds/roots are correlated; no deployment improvement claim or significance from treating paths as independent. No expansion of pilot based on results; no model/policy selected from76700/75400. Offline label-generation cost is separate from any future deployment budget. Fixed source state, samplecount, caps; do not use extra compute to rescueS26.

S28 complete within fixed pilot:8/8 source games replayed,64/64 full counterfactual paths complete,64short paths replayed,8/8 base actions match frozen source. Four-world full means6nonties/2ties; strong-short and weak-short each agree3/6. Split-world preferences2strict same,2opposite,3one-side ties,1both ties. Descriptive only, not8 or64 confirmation games of a new policy.507.88818CPUsec/316406444generated accounting. No expansion, retraining, candidate promotion or validation/final access. SeePUBLIC_LABEL_REPORT.md.

S29/S30 preregistration BEFORE new outcomes: replace B* level-guided expansion by global frontier, fixed frozen evaluator and robust4 terminal f+parent_f scoring. S29 cumulative sum(child.f - minimum sibling.f); S30 absolute child.f. Fixed tie order deeper, lowerf, earlier insertion. Same original legal generator/State/Hold/current+5. No beam cutoff, new model, hypothetical input or uncharged work. Root and every generated child inclpartial sibling batches count toward50k, max1generation/step. Both68tests pass, exhaustive all-root scores matchBstar on small boards and paths legal. NEW development77000..31 three arms regret/absolute/frozenrobust,32each,interleaved4workers,nodeadline,timeout180. Candidate must have completion no worse and strictly lower mean than BOTH others; ties/no winner stop. Unique winner fixed before NEW76700..47 validation winner/robust/legacy; then75400..63 formal64 only if dual mean gate passes; dual negative paired95 intervals and allcomplete required. No priority/tie/width sweeps or sample extension. S11/S28 manifests all verified unchanged.

S29/S30 complete, both REJECTED on NEW77000..31: regret32/32,201.65625 vsrobust32/32,196.15625,delta+5.5 CI[2.96875,7.90625].Absolute28/32,196.07143 amongcompleters; same28pairs vsrobust+0.25 CI[-2.57143,2.85714],not improvement.4no-suggestion failures retained,not replayable due missing operation trace;92completed games fully replayed.64candidate attempted-game generation logs audited inclfailures.1644.949169CPUsec inclfailures. No validation/final opened; no priority/tie/warmup rescue. Summary tool null-search handling corrected for failures; historicalS27 decision unchanged. Full96records,notcomplete-only subset,used for all gates.

S31 preregistration BEFORE game outcomes: cross-decision PlanHint from chosen previous legal best trace, skip executed first action. Carry only predicted pre-board,color,normalized placement,Hold. Retain B*/frozen evaluator/robust4. On first descent, prioritize matching hints only after current legal finder regenerates and charges children; stop on mismatch/first terminal/hint exhaustion. All otherfrontier retained. Observer performs same hint extraction/matching but original pop order. Both preserve first generated root child as legal fallback, charged inside50k, no fake terminal rating.69tests each incl2-node fallback, charged legal hint path and changed-board rejection. S11/S30manifests unchanged. First3observer parity76000..2 againstfrozenrobust required. NEW77100..31 candidate/observer/frozenrobust32each,interleaved4workers,50k,no deadline,timeout180. Candidate must beat both means/completion before unopened76700..47 validation candidate/robust/legacy; then75400..63 formal64 onlyifdual mean gate passes,allcomplete anddualnegative95 pairedintervals. No hint-length/priority/multi-plan sweep, no sample expansion. Historical plan replay is not new search; fresh successors all charged and totalCPU recorded.

S31 完成。每步上限 50,000 個生成節點。

dev（配對盤面）
- robust-plan-reuse 完成 32 局，平均 194.1250 塊，CPU 16.04 秒／完成局。
- robust-plan-observer 完成 32 局，平均 194.7500 塊，CPU 16.23 秒／完成局。
- learned-robust 完成 32 局，平均 194.7500 塊，CPU 15.87 秒／完成局。
- 候選減 robust-plan-observer 平均 -0.6250 塊，95% 配對區間 [-2.09375, 0.8125]。
- 候選減 learned-robust 平均 -0.6250 塊，95% 配對區間 [-2.09375, 0.8125]。

val（配對盤面）
- robust-plan-reuse 完成 48 局，平均 195.2708 塊，CPU 19.62 秒／完成局。
- learned-robust 完成 48 局，平均 195.3125 塊，CPU 19.05 秒／完成局。
- learned-fast 完成 48 局，平均 196.5833 塊，CPU 19.72 秒／完成局。
- 候選減 learned-robust 平均 -0.0417 塊，95% 配對區間 [-1.3333333333333333, 1.1666666666666667]。
- 候選減 learned-fast 平均 -1.3125 塊，95% 配對區間 [-3.8541666666666665, 1.1458333333333333]。

final（配對盤面）
- robust-plan-reuse 完成 64 局，平均 197.4375 塊，CPU 20.05 秒／完成局。
- learned-robust 完成 64 局，平均 197.0625 塊，CPU 19.58 秒／完成局。
- learned-fast 完成 64 局，平均 199.9844 塊，CPU 20.62 秒／完成局。
- 候選減 learned-robust 平均 +0.3750 塊，95% 配對區間 [-1.25, 2.0625]。
- 候選減 learned-fast 平均 -2.5469 塊，95% 配對區間 [-5.03125, -0.015625]。

未達升級門檻，最佳版本繼續採用 cheese-robust。
已開啟的資料包括 dev, val, final。詳見各階段 lock、decision、audit 與 replay JSON。

S32 fixed before fitting. Use only 32 old S26 observer trajectories (seeds 76800..31), no final data. Approximate the top cheese hole from the bottom consecutive single-vacancy rows. Count occupied cells above the target, their vertical span, capped neighboring column depths relative to the target row, wall indicator and neighboring hole travel distance. Fit six coefficients by fixed ridge 4 on within-game seven-lock differences of frozen residual value (old fixed unseen garbage price). Equal total weight per game, uncentered unit-RMS scale for differences, no intercept or parameter sweep. Keep B* and robust4. One frozen candidate, independent 24 paired games 77200..23 at 50k vs frozen robust. Both must complete all and candidate mean lower to open fresh 77300..63 formal64 pairs; require all complete and strictly negative paired bootstrap95 upper endpoint for promotion. Stop on failure.

S32 complete.
## 完整對局結果

- validation learned-robust，24/24 完成，平均 196.5417 塊與 19.81 CPU 秒。
- validation robust-dig-geometry，24/24 完成，平均 195.0000 塊與 19.84 CPU 秒。
- 候選減基準 -1.5417 塊，配對 95% 區間 [-5.166666666666667, 2.1666666666666665]。48 局成功回放。
- final learned-robust，64/64 完成，平均 194.5156 塊與 19.73 CPU 秒。
- final robust-dig-geometry，64/64 完成，平均 196.1406 塊與 20.12 CPU 秒。
- 候選減基準 +1.6250 塊，配對 95% 區間 [-0.296875, 3.5]。128 局成功回放。

開發時鎖定的幾何修正在正式盤面未達升級門檻。cheese-robust 繼續作為最佳版本。兩階段對局只識別整體策略的差異，無法將結果歸因於其中任一個幾何特徵。

S33 fixed before games. Stop a branch at its first mandatory garbage reveal. Visible colored garbage and public remaining supply initialize the search state. Count garbage and normal row clears exactly, using public 9/4 settings. Stop before adding any unknown row. Use existing robust4 terminal scoring, ordinary frozen board value plus depth, and true finish preference only once all supply is visible. Preserve charged fallback via observer-only S31 profile (no plan priority). No parameter search. Fresh 77400..77423 paired validation vs frozen robust at 50k; all complete and lower mean to open fresh77500..77563 formal64, all complete and negative upper paired95 CI for promotion. Replay all complete games. Evaluate total policy effect; no attribution to horizon alone near visible-bottom completion.

S33 complete.
## 完整對局結果

- validation learned-robust，24/24 完成，平均 196.2917 塊與 19.65 CPU 秒。
- validation robust-reveal-boundary，24/24 完成，平均 252.8333 塊與 17.79 CPU 秒。
- 候選減基準 +56.5417 塊，配對 95% 區間 [50.666666666666664, 62.666666666666664]。48 局成功回放。

S33 未達升級門檻。cheese-robust 繼續作為最佳版本。

S34 before probe or race outcomes. Two B* robust searches on original and horizontally reflected public occupancy, with mirrored piece types. Enumerate and charge both legal root sets and their two mapping roots, then split remaining50k equally. Match exact played shape type, Hold usage and four occupied cells in original coordinates. Choose among original-rated moves by summed reciprocal ranks, missing reflected vote zero, ties original rank. Real operations always reconstructed from original finder; if no complete original suggestion use first charged original legal root. Low-cost screen on S26 observer development only, preselected positions after50 and100 locks for all32 games: at least48/64 states with any matched rated root and at least1 fused decision different from native half-budget leader. This is a necessary opportunity screen only. If passed run fresh77600..77623 complete 100L paired vs frozenrobust. Both fully complete and lower mean to open fresh77700..77763, all complete and negative upper paired95 bootstrap bound to promote. No orientation/weight/budget tuning, no formal data reused.

S34 完成。32 局舊開發來源的 64 個固定盤面，64 個有共同合法候選，0 個在排名合併後改選。低成本門檻失敗。未啟用新完整對局或正式資料，最佳仍為凍結 cheese-robust。方法及重現指令見 MIRROR_REPORT.md。
