# W3D — st1/st2 prepare+glue: record hoist + claims-walk shape fix

**Status: GATES PASSED, full retention matrix green, handed to orchestrator
for 2^27 certification (one W3C-amendment item to rule on: st0's overlap
tax, below).** Phase-1 analysis: w3d-analysis.md. Branch `gpu/util-w3d`,
commits `03ebe3f3f` (F1 hoist) + `f52d2529e` (F2 walks); binary sha-256
`2a8248a067e94e01d7e21118bcfb3b32112b78b05a4a8e45705e4b54ab4af154`
(`prover-fixtures,metal` release modular_benchmark).

## What landed

### F1 — trace-record walk hoisted into stage 0 (`03ebe3f3f`)

`TraceRecord::collect` (4.17 s instr @2^27, gpu 5.3%, the single largest
GPU-idle host mass left in st1-st3) is challenge-independent — witness plane
+ `log_t` only. `prove()` now wraps the stage sequence in a thread scope and
spawns the walk at entry on a capped 8-thread pool holding the **process-wide
`BACKGROUND_BUILD_TOKEN`** (extends W2A's token — the record build holds it
across st0-st1, W2A's 6a builds take it post-st4; `st5 −0.1%` @2^25 confirms
zero interaction). Artifacts (record + RamAccessColumns +
SharedInstructionRows + PcRows — the walk's four co-products, unchanged)
cross via an mpsc channel whose receiver parks in the session
(`PrebuiltTraceRecord`); `TraceRecord::shared` joins it or rebuilds inline on
any mismatch/failure — the walk is deterministic, values identical either
way. Knobs: `JOLT_RECORD_HOIST=off` (same-binary ablation),
`JOLT_RECORD_BACKGROUND_THREADS=N`.

`TraceRecord::join` = 0.000 s in every measured run at both scales — the
walk fully hides inside st0's commit window (walk 1.5-1.7 s vs window 4.8 s
@2^25; modeled 7.2 s vs 11.3 s @2^27 from the thread-scaling probe).

### F2 — tensor-split eq + unreduced accumulators in the claims walks (`f52d2529e`)

The four post-rounds opening walks (st1 spartan-outer 35-opening; st2
product-remainder and instruction-claim-reduction, metal + optimized twins)
each materialized a **full T-sized eq table** (4.3 GiB alloc+fill+stream
@2^27) and the ICR walks paid a Montgomery conversion + field multiply per
lane per cycle. Now: `e_hi`/`e_lo` run-factored weights (one `e_hi` scale
per aligned run — `e_hi·Σ e_lo·v = Σ (e_hi·e_lo)·v`, exact regrouping; no
per-row extra multiply) and `fmadd_s256` unreduced accumulation (one Barrett
reduce per run ≡ the same sum mod p — the campaign's standard exact-math
argument). `InstructionOperandRow::field_values` is now test-only.

Byte-identity of both changes is pinned the strongest way available:
**byte-diff 12/12 wire-equal with the legacy prover in BOTH
`prover-fixtures` and `prover-fixtures,metal`**, with the hoist live.

### F3 (not implemented, anatomy mapped)

st3's InstructionInput round-0 host pass (0.455 s @ 0% GPU @2^27,
`native_q_evals`) is the same weighted-reduce shape as the slot's existing
round-1 bind dispatch — a `jk_instr_input_q0` kernel is a clean ~0.4 s
follow-up door. Round 1 (0.79 s @ 30%) is a bandwidth-bound 17 GiB dense
table write — not launch anatomy, no cheap fix. SpartanShift::prepare
γ-decomposition stays parked (≤0.3 s for deep plumbing).

## Gate results

### 2^24 ABBA (T-W-W-T, locked, non-monitor; trunk `aa29569c…` vs W3D `2a8248a0…`)

| stage | trunk mean | W3D mean | Δ |
|---|---:|---:|---:|
| st0 | 2.656 | 2.736 | +3.0% (see watch item) |
| **st1** | **1.045** | **0.537** | **−48.6%** |
| **st2** | **0.607** | **0.399** | **−34.3%** |
| st3 | 0.339 | 0.261 | −23.0% |
| st4 | 1.147 | 1.085 | −5.5% |
| st5 | 1.815 | 1.770 | −2.4% |
| st6a | 0.034 | 0.039 | +14.8% (= +5 ms, noise-class) |
| st6b | 1.267 | 1.237 | −2.4% |
| st7 | 0.177 | 0.167 | −5.2% |
| st8 | 1.139 | 1.149 | +0.9% |
| **st1+st2** | **1.652** | **0.937** | **−43.3%** (gate −25%: **PASS**) |
| SUM stages | 10.226 | 9.381 | −8.3% |

### 2^25 cool ABBA (≥3 min quiet + AC before every run)

| stage | trunk mean | W3D mean | Δ |
|---|---:|---:|---:|
| st0 | 4.547 | 4.811 | **+5.8% / +264 ms** (watch item) |
| **st1** | **1.935** | **1.018** | **−47.4%** |
| **st2** | **1.065** | **0.678** | **−36.4%** |
| st3 | 0.542 | 0.460 | −15.2% |
| st4 | 1.947 | 1.890 | −3.0% |
| st5 | 3.147 | 3.145 | −0.1% |
| st6a | 0.053 | 0.055 | +2.7% (+1.4 ms) |
| st6b | 1.496 | 1.485 | −0.8% |
| st7 | 0.218 | 0.227 | +4.2% (+9 ms, small-stage) |
| st8 | 4.382 | 4.200 | −4.2% |
| **st1+st2** | **2.999** | **1.695** | **−43.5%** |
| **SUM stages** | **19.332** | **17.967** | **−7.1% / −1.365 s** |

st3's consistent bonus (−15…−23%) is SoC-pressure RELIEF: st2's tail no
longer floods the memory system with a 4.3 GiB eq build immediately before
st3's prepares.

## The watch item: st0 +5.8% @2^25 cool — W3C-amendment path

The tax is REAL (tight cool T-pair 4.558/4.536 vs W-pair 4.835/4.788) and
its mechanism is pinned by two discriminating probes, both run same-session:

1. **Pool width 8→6→4** (2^24): st0 did NOT improve (2.74 / 2.76 / 2.80 —
   if anything worse, confounded by session warming); narrower pools only
   lengthen the walk (4-thread walk ~17 s would overflow the 11.3 s window
   @2^27 — width stays 8).
2. **Utility-QoS demotion of the pool** (2^25 cool ×2): st0 4.824/5.031 —
   no improvement over default QoS. Change reverted (unmeasurable
   complexity).

⇒ Not P-core scheduling, not pool width: **memory-bandwidth contention** —
the walk's ~12 GiB of traffic @2^25 competes with commit's GPU streaming.
Intrinsic to overlapping any memory-heavy walk with commit; the only way to
zero it is not to overlap, which forfeits a 4× larger win.

Amendment arithmetic: summed-stage win 1.365 s = **5.2×** the st0 loss
(threshold ≥3×). Per the codified W3C amendment I am NOT self-killing;
deferring the retention call to the orchestrator's 2^27 certification.
@2^27 the walk occupies a larger fraction of the commit window (7.2/11.3 vs
1.6/4.8), so model the tax at +3-6% of st0 (+0.35-0.7 s) against
st1 −3.7-4.0 s + st2 −0.9-1.1 s canonical. Modeled net: **−4.0…−4.7 s,
1.98-2.02 MHz** if the 2^25 ratios hold.

## Retention matrix (all green, tree @f52d2529e)

- jolt-kernels metal: **242/242** (241 + new `background_collect_matches_inline`
  — lane-for-lane background-vs-inline equality + stale-spawn fallback)
- jolt-prover: **20/20** in `prover-fixtures` AND `prover-fixtures,metal`
  (byte-diff **12/12 both arms** — F1+F2 wire-equal with legacy, hoist live)
- jolt-dory: 46/46
- legacy muldiv: 3/3 `host` + 3/3 `host,zk`
- clippy `-D warnings`: host / host,zk / metal; fmt clean
- Known pre-existing flakes seen and ignored per W2A's log: nondeterministic
  nextest "leaky" flags (2 kernels, 1-2 prover)

## Artifacts

- ABBA traces/logs: `/tmp/w3d-ab-{T1,W1,W2,T2,W6,W4}.{json,log}` (2^24),
  `/tmp/w3d-s25-{T1,W1,W2,T2,Q1,Q2}.{json,log}` (2^25 cool; Q* = rejected
  QoS arm)
- Thread-scaling probe: `/tmp/w3d-probe-{A-18t,B-8t,C-4t}.json`
- Analysis scripts: `/tmp/w3d_{ab_stages,pool_probe,qos_probe,span_extract}.py`
- Binaries: `/tmp/w3d-bin` (candidate, sha `2a8248a0…`), `/tmp/trunk-bin`
  (`aa29569c…`)

## Notes for certification

- The 2^27 A/B arms are same-binary env-switched for F1
  (`JOLT_RECORD_HOIST=off`) but F2 has no knob — use the trunk binary for
  the full A arm.
- Watch st0 and the st1 `TraceRecord::join` span (should stay ≈0; if the
  walk overflows the window on a hot box the join residue appears there,
  never exceeding the old inline cost).
- Memory: the record family's +28 GiB moves into st0's window; prove peak
  remains st6b's — no new ceiling interaction expected at the ~90 GiB tier.
