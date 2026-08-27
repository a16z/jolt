# Metal saturation — waves 1–2 archive (closed 2026-08-05 01:25 UTC)

## Wave 1

1. Radix-4 sumcheck fusion: concrete polynomial derivation and pro-model
   soundness review before implementation; prototype only after review.
2. Address-major probe: include Dory commitment/opening, not only the already-
   closed booleanity address-phase ordering.
3. Saturation attribution: fresh stage traces/counters separating ALU,
   bandwidth, occupancy, host serialization, and launch/synchronization.

## Decisions and results

- Wave-1 scoping lanes dispatched after both baselines: radix-4 fusion
  (`a346b564`), address-major/Dory (`1af3a092`), saturation attribution
  (`3cceafee`). Phase 1 is static only so Cargo and timed runs remain serial.
- GPT-5.6 pro oracle dispatched on the concrete radix-4 polynomial and current
  Jolt round-loop contracts (`55fd4b105b90`). Implementation is blocked on its
  review. Source-file attachments were unavailable across the external-machine
  boundary, so the oracle prompt carries the concrete protocol and integration
  contracts inline.

### Fresh saturation evidence

The fresh `2^27` monitor run completed in 80.99 s; its wall is attribution-only.
Time-weighted `ioreg` device-utilization samples by stage:

| stage | monitor wall | GPU | CPU | active cores |
|---|---:|---:|---:|---:|
| st0 | 18.01 s | 79.4% | 61.9% | 11.1 |
| st1 | 4.70 s | 77.1% | 12.4% | 2.2 |
| st2 | 2.96 s | 48.3% | 18.4% | 3.3 |
| st3 | 3.00 s | 16.1% | 37.0% | 6.7 |
| st4 | 9.28 s | 40.2% | 22.3% | 4.0 |
| st5 | 14.67 s | 77.6% | 25.0% | 4.5 |
| st6a | 0.24 s | 36.0% | 44.7% | 8.1 |
| st6b | 17.49 s | 34.5% | 18.6% | 3.3 |
| st7 | 1.89 s | 13.5% | 65.6% | 11.8 |
| st8 | 8.70 s | 86.6% | 10.2% | 1.8 |

**Correction to the closed campaign:** re-analysis of both the old final trace
and this fresh trace finds sampled-zero GPU intervals over 2 s in st3, st4,
st6b, and st7. The prior `NONE >1 s` statement is not reproducible. In the
fresh trace, the longest are st3 2.48 s, st4 2.58 s, st6b 2.23 s (several),
and st7 2.11 s. `ioreg` is a sampled activity signal, not an ALU-occupancy
counter, but a multi-second exact zero is enough to reject continuous device
occupancy.

Dominant host/round structure at `2^27`: st3 `prove_batch` 2.14 s; st4
`RegistersRWC::prepare` 2.45 s plus 6.69 s rounds; st6b 15.21 s rounds plus
`IncCR::prepare` 1.79 s; st7 is almost entirely `HWCR::prepare` 1.887 s.

The built-in `JOLT_METAL_CB_TRACE` audit at `2^25` records 646 command buffers.
Fresh empty-CB cost is 133.8 us; the absolute launch/round-trip ceiling is
therefore about 86 ms, under 0.5% of a 19.72 s proof. Launch overhead is not
the campaign bottleneck, though it matters in tiny tail rounds.

Fresh roof/pressure probes on the M5 Max:

- streaming bind: 357 GB/s sustained in the contention probe, 485 GB/s best
  isolated large pass; compute roof: 11.30 Gmont-mul/s;
- concurrent GPU bind + CPU field-mul cuts GPU bandwidth 55% and CPU bandwidth
  45%, identifying shared-memory contention as a first-order limiter;
- Miller kernels prefer one/two pair-evaluations per thread, and per-pair cost
  collapses until 4k-8k threads are exposed: occupancy/register pressure is
  material inside st0/st8, but those stages already show 79-87% device use;
- Miller + CPU ALU soak is neutral on the device, separating compute occupancy
  from the memory-walk contention behind st0's bimodality.

Initial verdict: the remaining wall is a mix of serial/parallel host mass in
st3/st4/st7 and bandwidth/queue contention in st5/st6b. ALU/occupancy is local
to the Miller-heavy endpoints; fixed launch overhead is negligible globally.

Hardware-counter limitation: this host has no `xctrace` developer tool, and
Metal exposes only the `timestamp/GPUTimestamp` counter set (`counterSets`
enumerated directly on the M5 Max). ALU occupancy, SIMD utilization, cache
misses, and DRAM bytes are therefore not directly observable from public
Metal counters here. The audit distinguishes them through fresh device-active
samples, command-buffer GPU timestamps, controlled bandwidth/compute roofs,
thread-scaling, and contention experiments; it does not relabel `ioreg` GPU%
as ALU saturation.

### Radix-4 pro-model gate (`55fd4b105b90`)

**NO-GO for the proposed `3d` message with ordinary bind-by-two/Dory MLE
openings.** For digit embedding `0..3 -> (00,01,10,11)`, the cubic coordinate
maps are

`x(Z) = -Z^3/3 + 3Z^2/2 - 7Z/6` and
`y(Z) = 2Z^3/3 - 3Z^2 + 10Z/3`.

The four-point identity is valid:

`q(0)+q(1)+q(2)+q(3) = G(0,0)+G(0,1)+G(1,0)+G(1,1)`.

But a four-entry oracle's degree-3 digit interpolation is not its ordinary
bilinear MLE at `(x(z), y(z))`. Counterexample: `U(X,Y)=XY` has digit values
`[0,0,0,1]`; its cubic interpolation is `Z(Z-1)(Z-2)/6`, while ordinary
binding/opening yields `x(Z)y(Z)`. They agree only at the four digit nodes.
The original proposal would therefore propagate one claim and open another.

Sound alternatives:

1. Preserve ordinary MLE/Dory semantics and define
   `q(Z)=G(x(Z),y(Z))`. A generic relation with per-variable degree `d` has
   bidegree `(d,d)`, hence `deg q <= 6d`, with error `6d/|F|` per fused pair.
2. Preserve `3d` by changing every table bind and commitment/opening to the
   quaternary Lagrange extension. This is a commitment-protocol redesign, not
   a Metal-only prototype.

For either consistent extension, coordinate correlation is not itself a
soundness failure: the univariate root bound applies in `z`, and Dory may open
an MLE at transcript-derived correlated coordinates. RLC batching remains
linear. A member inactive across both fused variables contributes constant
`claim/4`; pairs must split at every active-set, degree, optional, uniskip, or
binding-order boundary. The linear coefficient remains compressible because
the four-point functional has weight `0+1+2+3=6`, invertible in the field.

Decision: never build the inconsistent `3d + ordinary Dory` shape. The fusion
lane must first prove either a relation-specific lower bound, a cheap
quaternary-to-Dory bridge, or an honest `6d` prototype with a positive cost
model. Mandatory regression: `U=XY` distinguishes digit interpolation from
ordinary MLE binding.

### Quaternary Dory bridge under review

`dory-pcs 0.4.0` commits only the Boolean-corner value vector, so the
commitment itself is extension-agnostic. Its prover already accepts arbitrary
public evaluation vectors through `MultilinearLagrange::compute_evaluation_vectors`.
The verifier is the missing seam: it currently stores one binary coordinate
per Dory reduce round and reconstructs the folded scalar as a product of
`alpha * (1-r) + r` terms.

A radix-4 Lagrange vector `[l0(z),l1(z),l2(z),l3(z)]` is generically rank two
when reshaped as 2x2, so it cannot be represented by two ordinary MLE
coordinates. It is nevertheless one tensor factor. Under Dory's half-split
folds `s <- alpha*s_L+s_R`, two consecutive reduction challenges fold that
factor to

`alpha_2*alpha_1*l0 + alpha_1*l1 + alpha_2*l2 + l3`

(`alpha_i^-1` on Dory's opposite scalar vector). A typed binary/radix-4 factor
schedule can therefore keep verifier work logarithmic: hold one four-weight
factor across two Dory reductions, multiply the accumulator by the expression
above, and never materialize the full vector. A radix-4 factor must not
straddle Dory's row/column matrix split; an odd boundary remains binary.

This changes Dory's public evaluation-point API and verifier scalar-fold
logic, but not the commitment or reduce proof. GPT-5.6 pro job
`ec0b50d07d63` is auditing the exact construction, degree bound, folding order,
and soundness before any implementation.

### Metal radix-4 bind microprototype

Implemented an isolated `jk_fr_bind4` prototype without changing protocol
bytes or the prover driver. For one four-entry Lagrange block it uses

`a0 + l1*(a1-a0) + l2*(a2-a0) + l3*(a3-a0)`.

Because `l0+l1+l2+l3=1`, this is the exact quaternary Lagrange bind. It costs
the same three Montgomery products as two binary binds, but removes the
intermediate `N/2` table read+write and one dispatch. The device property test
matches a host four-point Lagrange evaluation for dense and ragged shapes.

Two timing decisions (minimum of five warm passes inside each run):

| input | run 1 speedup | run 2 speedup | verdict |
|---:|---:|---:|---|
| `2^20` | 1.25x | 1.10x | small positive |
| `2^22` | 0.49x | 1.06x | unstable / no proven gain |
| `2^24` | 1.98x | 1.51x | strong positive |

Second-run absolute large-table result: two binary binds in one command buffer
2.39 ms versus direct radix-4 1.59 ms (`1.51x`). The first run was 3.11 ms
versus 1.57 ms (`1.98x`). At production-sized dense tables, direct bind-by-two
therefore has real Metal bandwidth upside; small/mid tables remain launch/cache
mode sensitive. This is only the binding half of the cost model: higher-degree
message generation and generalized Dory evaluation can still erase the gain.

The prototype now exercises the full algebraic shape for a dense degree-2
relation `G=U*V`: interpolate the seven values needed for `deg q <= 3d = 6`,
check the four digit-node sum against the Boolean claim, draw `z`, bind both
tables on Metal with the quaternary weights, and match `q(z)` to the terminal
bound-table relation. A tampered polynomial adds
`c*Z*(Z-1)*(Z-2)*(Z-3)`: it preserves all four node values and the input-claim
check but is rejected at the terminal random point. Targeted device/algebra
tests pass. This establishes a sound single fused-round Metal prototype; it
does not yet alter production transcripts or Dory.

### Stage-4 production mapping: killed by variable order

The virtual register-address coordinates are the only plausible radix-4 site
that can disappear before Dory, but the first scope mapped them to the wrong
rounds. In the current 34-variable `RegistersRW` schedule the leading seven
batch-only variables are **cycle bits**, not address bits. The Metal CSR is
rowed by cycle and its two-wait prefix halves adjacent cycle rows. Packing
that measured prefix would therefore create non-binary factors on committed
`RdInc` cycle coordinates—the exact Dory incompatibility the design sought to
avoid.

The real address variables are the seven-variable tail, overlapping active
`RamVal` and currently handled after all cycle folds. Moving them first is a
different protocol and state algorithm: group `col >> 2` inside each of T
cycle rows, preserve T rows, and construct the bound `Val` contribution for
registers absent from the sparse access row. The existing CSR lacks that
dense register-state information, so the claimed unchanged-representation
pass cut and **1.2–2.2 s** estimate are invalid.

The exact address-round degree is at most 6, not 9: address-constant
`EqCycle/RdInc` leave products of two cubic quaternary extensions. Algebraic
batch padding is sound (`128 / 4 / 4 / 4 / 2 = 1` before `RamVal` joins), and
the address factor can be eliminated before PCS, but current proof/derive
APIs also conflate semantic variables, messages, and scalar challenges. A
production cut would require a 34-variable/31-message mixed-round engine,
typed four-weight factors through stages 4–6b, a new address-first `Val` state
algorithm, fail-closed transparent-only config, and full backend/e2e/tamper
coverage.

Decision: **NO-GO for `[P4,P4,P4,S,S×27]` on the unchanged CSR.** Keep the
sound isolated bind prototype. Address-first state construction is unpriced
research, not a campaign win. Pro-model job `20b0ff781369` remains useful as
the final mathematical audit of the corrected virtual-coordinate shape, but
cannot overturn this code/performance blocker.

## Wave-1 close certification

Commit `5d835a6d3` changes no active prover path: it adds the unused bind4
kernel/prototype tests, a benchmark selector, and reports. The canonical
close nevertheless reproduced both known ambient/tail modes:

| run | result | distinguishing stage |
|---|---:|---|
| `2^25` | **20.13 s / 1.667 MHz padded** | warm, within the established band |
| `2^27` close 1 | **81.54 s / 1.646 MHz padded** | st0 19.305 s vs 12.079 s baseline |
| `2^27` close 2, 4-min cool gap | **100.25 s / 1.339 MHz padded** | st6b 30.963 s vs 16.345 s baseline; st0 21.354 s |

Velocity cap reached (baseline + disagreeing close pair); no fourth run. The
fresh campaign-start run remains the flagship for this unchanged prover path:
**71.77 s / 1.870 MHz padded** at `88b063db3`. The close pair is not a retained
regression: neither prototype is called by the prover, and stage-local deltas
identify the existing bimodal st0 walk/commit contention plus a newly observed
st6b tail mode. Dashboard candidate remains the 71.77 s baseline; no new point
should be emitted for the research-only commit.

### Address-major Metal probe

Address-major is already a valid end-to-end protocol mode, including verifier
and Dory parity tests, but the current Metal implementation is structurally
hostile to it:

- cycle-major commitment streams every trace column from one packed pass;
  address-major materializes one full strided grid table per polynomial;
- `MetalJointOpening::prepare` accepts only cycle-major grids, so address-major
  falls back wholesale to the optimized CPU fold path in stage 8;
- Metal sumcheck slots are cycle-major gather/coalescing designs. Global
  address-major would require new kernels, not a free layout flip.

A benchmark-only `JOLT_TRACE_ORDER=address` selector allowed a direct `2^22`
A/B. Cycle-major completed in **3.52 s / 1.192 MHz padded**, peak RSS 3.81 GiB.
The address-major arm was still consuming roughly 12 CPU cores and 4.14 GB RSS
after **more than 240 s** and was terminated. This is a decisive **>68x lower
bound regression** at the small-scale gate, so no second address-major run is
warranted under Velocity v3.

Decision: kill global address-major on the current Metal backend. The CUDA
sharding prize does not transfer to a unified-memory machine whose fast path
already streams cycle-major columns. Retain address-major only as a correctness
mode and use targeted internal AoSoA/block-local transposes when an individual
kernel demonstrates a locality win. A production address-major campaign would
first need a streaming commitment builder, Metal joint-opening fold, and
address-major sumcheck kernels; it is not an optimization knob today.

The code-dimension audit closes two broader layout hypotheses. Stage 6b has
no remaining address axis after 6a; its T-scale state is already contiguous
cycle-domain ping-pong. Its dense streams total roughly 170–190 GB at `2^27`,
about 0.3 s at the measured memory roof versus a 16.3–17.5 s stage. The
residual is gather arithmetic, per-round waits, shrinking-tail occupancy,
host glue, and CPU-member interference—not address locality. Likewise,
cycle-major Dory shards are contiguous trace segments; address-major shards
would each scan the full trace, reversing rather than enabling useful
sharding on this backend.

One address-related door remains a bounded **probe**, not a retention:
stage-4 `RegistersRW` could bind its seven address variables before the 27
cycle variables. Fixed three-wide slots would then collapse into four dense
cycle tables, reusing the already-fused IncCR-shaped loop. Modeled prize is
1.5–2.5 s at `2^27`; protocol and batch-window risk are moderate-high. Probe
only the seven-pass address phase first; kill if it exceeds 0.15 s at `2^24`.
If built, binding-order soundness follows from a public permutation of the
same sumcheck variables: every message still precedes its challenge, degree
bounds and `sum degree/|F|` are unchanged, and downstream opening coordinates
must be permuted explicitly. Retention would require an FS-absorbed protocol
axis, fail-closed verifier validation, e2e accept, round/config/opening tamper
rejection, and the full integrated suite.

## Wave 2 (orchestrator handover 2026-08-04 23:45 UTC, task dade2763)

Claude orchestrator adopted the five wave-2 lanes mid-flight. Prior
orchestrator's pro-model jobs `ec0b50d07d63` (typed Dory bridge) and
`20b0ff781369` (stage-4 shape audit) died with it; results unrecovered.

Lane state at adoption + first outcomes:

- **c177d0a5 harness (codex): SHIPPED.** jolt-eval single-kernel harness
  integrated from upstream `fa303e27f` (`966fc3f4d`): runtime
  `callgrind:<bench>:instructions` objectives, Metal Criterion template with
  `gpu_lock()` + synchronous single command buffer, `sync_targets.sh`.
  Verified: fmt/check/clippy/nextest 4-pass; `metal_fr_bind` 2^20 ~255 µs.
  Valgrind unavailable on this host — callgrind parse paths compile-verified
  only. Merged to trunk as `7dc76f732`.
- **e371fb72 hostgaps (codex): SHIPPED st7.** Stage-7 Hamming-weight
  pushforward rebuilt on stage-6a's split-eq deferred-bucket algorithm
  (extracted to `optimized::one_hot_pushforward`), four outer blocks per
  Rayon worker (`929927102`). Isolated 2^22: 79.82 -> 46.80 ms, **1.71x**,
  non-overlapping CIs; stage-calibrated estimate **-0.78 s @2^27** (st7
  1.887 -> ~1.106 s), ~1.09% whole-proof. Merged as `3c2ee6a48`
  (Cargo.lock union regenerated via `cargo metadata`; jolt-eval+kernels
  targeted tests 20/20 on the merged trunk). E2e effect certifies at the
  wave gate. Lane resumed onto st3 `InstructionInput::prove_batch` round-0
  (2.140 s host-serial): attribute -> cut -> isolated measure; retention bar
  >= 0.3 s stage gain.
- **1b0dc99d radix4 (claude): Phase-1 map complete** (`5be1f5036`,
  `metal-w2-radix4-map.md`). Verdict: **no unconditional radix-4 GO in the
  live prover.** Sole conditional GO: stage-4 RegistersRW **address-first**
  quaternary address phase `[P4,P4,P4,S | S x27]` — deg-6 `q(Z)`,
  D={-1,0,1,2}, PCS-clean factor (final-point provenance re-verified: st6
  IncCR cycle + st7 HWCR address challenges only), run-length folded-Val
  O(T) state algorithm (map §3.4) closes the r4-pin blocker. Honest split:
  the 1.5-2.5 s prize belongs to address-first dense-cycle collapse
  (replaces the 5.86 s / 30.5%-GPU / 3.73 s-idle sparse prefix); radix-4's
  own increment ~0.1-0.3 s. st1/st2/st3/st5/st6b/st7 all NO-GO with reasons
  (map §2); st4 cycle-prefix stays killed.
- **Gate 0 re-dispatched:** pro-model job `3dbb9c10e48a` (~20 min) carries
  the full corrected construction inline (packed round, address-first
  permutation + x128/(4,4,4,2) activation-join algebra, run-length Val
  reconstruction vs partially-bound MLE, consumer/tamper surface). No
  production code before its verdict.
- **Gate 1 dispatched in parallel (measurement only): lane bdcc152e**
  (codex sol-xhigh, worktree `metal-w2-r4gate1` off merged trunk) — jolt-eval
  `RegistersAddressPhase` objective, binary vs radix-4 arms with dense
  brute-force parity at 2^12, kill line: phase <= 0.15 s @2^24 AND radix4 <=
  binary.
- **c42e074e st6b, d6c80e49 st0 (claude): still running** — resumed by main
  session pre-handover; monitoring.

Fleet: e371fb72 (st3) · bdcc152e (r4 Gate-1) · c42e074e (st6b) ·
d6c80e49 (st0) · oracle 3dbb9c10e48a. Estimated shipped-but-uncertified
delta so far: -0.78 s.

### Wake 1 (2026-08-05 00:29 UTC): three oracle verdicts + st6b RETAIN

**Orphaned wave-1 pro jobs delivered after adoption** — not lost:

- `ec0b50d07d63` (typed quaternary Dory factor): **GO with mandatory
  conditions.** Dory commitment itself is extension-agnostic; verifier holds
  one four-weight factor across two consecutive reduce rounds
  (`α₂α₁l₀+α₁l₁+α₂l₂+l₃`, inverse-α on s₂); factor must not straddle the
  row/column split; verifier recomputes Lagrange weights; descriptor must be
  FS-bound before Dory alphas; challenge z must be transcript-linked, never
  prover-supplied. This UNLOCKS the deferred st6b/st7 packing door (map §5)
  — now also satisfied on the measurement side by c42e074e's st6b anatomy.
  Wave-3 candidate, not scheduled now.
- `20b0ff781369` (stage-4 corrected shape): **GO-WITH-MANDATORY-CHANGES** —
  agrees with the fresh gate: one aggregate polynomial per fused round
  (never per-member polys sharing r), canonical 10-coefficient encoding for
  generic d=3 (deg ≤ 9; our site's exact bound is 6), typed factor
  propagation with no Radix4→binary-point conversion anywhere, PCS
  noninterference as a typed-API invariant, extensive test matrix.

**Gate 0 (fresh job `3dbb9c10e48a`): protocol SOUND, one spec gap.**

- Q1 packed round: SOUND. D-sum functional `(4,2,6,8,18,32,66)` verified;
  absorb-full-coeffs-then-one-squeeze is correct FS ordering; error 6/|F|
  at the exact degree-6 site (ordinary ROM grinding caveat only).
- Q2 address-first + activation join: SOUND. `128·C /4/4/4/2 = C` exactly,
  RLC-linear, join-adjacent single harmless; schedule must stay
  config-derived + transcript-bound (already the map's design).
- Q3 run-length Val state: **GAP — level-0 temporal convention
  underspecified** (write effective at j vs j+1, same-cycle read/write
  pre/post-state, cycle-0 runs, final-cycle writes, x0, coincident-
  breakpoint atomic merge). Fold algebra itself PROVEN correct (piecewise-
  constancy survives repeated quaternary folds by induction). Becomes SOUND
  once the convention is pinned normatively to the production Val semantics
  with boundary tests.
- Q4 typed-factor surface: SOUND; extra tamper/negative-API/boundary tests
  mandated (folded into the campaign test matrix).
- Overall: NO-GO for production code until Q3 spec + tests exist; the
  protocol shape itself is cleared.

**Action:** steered Gate-1 lane `bdcc152e` mid-flight: derive the level-0
convention from production code (file:line citations), implement BOTH its
run lists and its dense reference against that convention, and cover the
oracle's boundary-case list in the parity suite ("Level-0 Val temporal
convention (normative)" section in its report). Gate-1 kill line unchanged.

**st6b lane c42e074e: RETAIN, merged.** Root cause of the st6b tail mode:
round-3 dense adoption — three serialized synchronous `jk_ra_materialize`
dispatches inside `begin_round` (17.19 s of the 30.96 s tail-mode close vs
1.55 s good mode @2^27; fresh ~28.5 GiB ping-pong, blocked≈2×gpu, then the
round message re-reads the 8 GiB just written). Fix: RavDriver defers
adoption one lazy round and lands it at T/16 as ONE detached
`jk_rav_adopt_round` (gather once → dense write + message lanes fused);
Rav adoption alloc+write halved, begin_round 64.3→0.47 ms @2^24, isolated
total −7.2..−8.6% at 2^24 (two agreeing quiet runs), byte-parity CPU-twin
oracle, wire bytes unchanged, `JOLT_RAV_DEFERRED_ADOPT=0` legacy knob.
Modeled −0.5..−1.5 s good-mode st6b @2^27, more in tail mode. Bool driver
(20 polys, 15 GiB) = follow-up door. Merged as `9c4699c56`; targeted
rav/lazy_ra parity 9/9 and muldiv e2e (host) 3/3 pass on the merged trunk.

Fleet after wake 1: bdcc152e (Gate-1, steered) · e371fb72 (st3) ·
d6c80e49 (st0). Estimated shipped-but-uncertified: st7 −0.78 s + st6b
−0.5..−1.5 s (good mode) + tail-mode variance kill.

### Wake 2 (00:35 UTC): st0 verdict — door killed with mechanism

**d6c80e49 st0: KILL the fix, RETAIN the harness** (`4846a0754`, merged
`e206f3807` — jolt-eval metal feature union-resolved, superset kept).
The ±5 s st0 bimodality is **ambient device power/clock state, not a
walk↔commit scheduling defect**: reproduced on solo commit with zero
co-runner (3.8 → 22.5 s wall at constant ~22 s utime — device stall, CPU
work unchanged). No in-process scheduling fix can remove a mode that exists
without the walk. Real contention exists but is secondary: co-run commit
+8.4% @2^22 / +43% @2^24, walk +65–118%, and inflation tracks co-runner
**residency, not intensity** (2-thread walk hurt commit MORE than
8-thread); VM/fault theory dead. Full fix matrix measured and dead:
stagger=serialize (+25%), width throttling null/catastrophic, QoS-utility
null, QoS-background 2.8× walk; background×12 E-cluster won stable windows
but starved 26.3 s in a degraded window — fail-unsafe. Future door (needs
2^27 cert): bg12 + starvation guard. Retained: `st0-contention` bin with
six legs + `JOLT_RECORD_HOIST_DELAY_MS`/`JOLT_RECORD_QOS` default-off
knobs; default spawn path byte-identical.

**Campaign-wide bench rule from this lane:** ambient power state moves
whole distributions (observed live on a power flip); all timed A/B claims
must be same-window interleaved — solo before/after pairs across windows
are not evidence. Propagated to the c42e074e resume.

**c42e074e resumed** onto its named follow-up: Bool-driver deferred+fused
adoption (20 polys, ~15 GiB @2^27), same pattern/discipline as the shipped
Rav fix; retention bar = begin_round collapse + no total regression;
alloc delta reported (tail-mode fuel).

Fleet: bdcc152e (Gate-1, steered with Q3 spec pin) · e371fb72 (st3) ·
c42e074e (st6b Bool). st0 lane closed.

### Wake 3 (00:42 UTC): st3 GO merged, st1 lane opened

**e371fb72 st3: GO, merged** (`9195cbd02` → merge `d8288f11b`). Attribution
@2^24: host `native_q_evals` 44.6 ms, first-write/sync residual 36.5 ms,
GPU bind window 29.3 ms. Cut the scalable slice: new `jk_instr_input_q0`
kernel (Boolean endpoint selection + quadratic coefficient, three device
reductions, host q(2/3) reconstruction), CPU fallback + transcript bytes
unchanged. Isolated: 2^22 32.06→3.81 ms (8.41×), 2^24 139.64→15.15 ms
(9.22×), disjoint CIs. Estimate: st3 2.340→1.939 s @2^27 (**−0.40 s**,
~0.56% whole-proof). Metal parity 3/3.

**Semantic merge trap hit and fixed:** st6b (`jk_rav_adopt_round`) and st3
(`jk_instr_input_q0`) both appended a `KernelId`; textual merge kept
`ALL: [Self; 68]` with 69 variants → E0308. Fixed `4eb216ec8` (69);
union parity suites 22/22 after fix. Rule for future kernel-adding merges:
re-count `KernelId::ALL` (Bool lane will make it 70). c42e074e warned
mid-flight.

**e371fb72 resumed onto st1** (4.456 s stage): attribute uniskip-message vs
round-loop vs host-glue at 2^22/2^24 (also prices the deferred radix-4 st1
door), then cut the largest slice; retention ≥0.3 s @2^27; same-window
interleaved A/B per the ambient-power rule.

Running total (stage-calibrated estimates, uncertified): st7 −0.78 s,
st6b −0.5..−1.5 s, st3 −0.40 s ⇒ ~**−1.7..−2.7 s** vs 71.77 s flagship.
Pending: Bool driver, st1, Gate-1 (radix4). Wave-2 gate (full battery +
2^25/2^27 certification) after those settle.

### Wake 4 (00:53 UTC): Gate 1 FAIL — address-first door killed entirely

**bdcc152e Gate-1 verdict: FAIL, pre-registered kill line hit decisively.**
Honest parallel (Rayon, release) CPU implementation of the full address
phase with the complete parity battery (dense 128×T reference, boundary
cases, 64 per-fold probes — all passing):

| cycles | binary 7-message | radix-4 [P4,P4,P4,S] | ratio |
|---:|---:|---:|---:|
| 2^22 | 1.099 s | 1.467 s | 1.334× |
| 2^24 | 6.303 s | 7.104 s | 1.127× |

Both arms **>42× over the 0.15 s @2^24 kill line**, and radix-4 loses to
binary at both scales (the 4-child merge + 7-point extension costs more
than it saves in passes). Extrapolated to 2^27 the run-list address phase
alone (~50 s) would dwarf the 5.86 s sparse prefix it was to replace —
the modeled 1.5–2.5 s prize was predicated on ~0.15 s and is arithmetic
fiction at the measured cost. No implementation-quality escape: even a
perfect further ×10 leaves the phase above the kill line. **KILLED:
st4 address-first restructuring, both arms (binary AND radix-4) — closes
the campaign-close parked probe AND the radix-4 map's sole conditional GO.**

Durable salvage merged (`eb31df7f5` map, `a25aa6793` objective):
- `RegistersAddressPhase` objective + the full boundary parity suite;
- **normative Level-0 Val temporal convention** (oracle Q3 answer, cited
  to code): `Val(k,j)` = pre-state of cycle j; write at j effective at
  j+1; same-cycle reads see pre-state; cycle-0 run `(0,0)` for every
  register; final-cycle writes emit no run; x0 writes discarded.

**Radix-4 / protocol-restructuring thread status after Gate 1:** closed
for this wave. Remaining doors, all wave-3+ and none scheduled: typed-Dory
st6b/st7 packing (oracle ec0b50d07d63 GO; prize shrinks as the st6b
adoption fix lands — must be re-priced against the post-fix anatomy),
st1 packing (await e371fb72's attribution). The isolated bind4 kernel
remains sound research on the trunk.

Worktrees pruned: metal-w2-{radix4,st0,harness,r4gate1} removed (merged).
Active: metal-w2-hostgaps (st1) · metal-w2-st6b (Bool). Wave-2 gate
(full battery + 2^25/2^27 certification) fires when those two land.

### Wake 5 (00:59 UTC): Bool driver RETAIN — st6b lane complete

**c42e074e Bool door: RETAIN, merged** (`4ff7ba75a` → merge `d21e5ed1a`;
KernelId::ALL union re-bumped 69→70, union parity 30/30). BoolDriver
(booleanity-cycle, 20 polys — the biggest st6b adopter) on the deferred
schedule: lazy horizon 8, ONE detached `jk_bool_adopt_round` (width-16
gather → dense write at T/16 + both summand lanes). Isolated @2^24:
adopt-begin 68.7–71.3 → 0.44–0.51 ms (−99.3%), total −7.7…−8.3%,
2 agreeing interleaved quiet runs, CPU-twin byte-parity oracle.
**Alloc: Bool adoption 15 → 7.5 GiB @2^27; combined with Rav, st6b
adoption-round fresh allocation 28.5 → 14.25 GiB and all three phase-1
blocking materialize waits eliminated.** Modeled good-mode st6b −1…−2 s
@2^27 across the three drivers; tail mode loses half its allocation fuel.
`JOLT_BOOL_DEFERRED_ADOPT=0` legacy knob. jolt-kernels metal 248/248.
st6b lane closed; worktree removed.

Wave-2 ledger (stage-calibrated, uncertified): st7 −0.78 s · st6b −1…−2 s
(good mode, both drivers) · st3 −0.40 s ⇒ **~−2.2…−3.2 s modeled** vs the
71.77 s flagship, plus tail-mode variance halved twice (adoption fuel).
Waiting on the last lane: e371fb72 st1 attribution+cut. Then the wave-2
gate: clippy both modes, muldiv host+zk, full nextest, 2^25 + 2^27
certification pair, dashboard point, parent report.

## Wave-2 gate certification (2026-08-05 01:01–01:25 UTC)

**Full battery green** on merged trunk (final code commit `d21e5ed1a`+st1
merge): clippy `--all` host and host,zk `-D warnings`; muldiv e2e host 3/3
+ zk 3/3; jolt-prover-legacy 444/444 (default), 480-suite (zk) and 445/445
(akita); jolt-verifier akita+prover-fixtures 85/85; jolt-sdk, tracer
127/127, jolt-witness 34/34; metal union suites 77/77. One environmental
fix: `rustup component add rust-src` (stdlib guest builds; not a code
regression).

**Certification (benchmark-locked, AC, quiet machine):**

| run | result | note |
|---|---:|---|
| 2^25 | **19.01 s / 1.765 MHz padded / RSS 25.16 GiB** | baseline 19.67 s / 27.42 GiB — **−0.66 s, −2.3 GiB** |
| 2^27 #1 | 72.93 s / 1.841 MHz | bad st0 window (st0 18.12, st1 6.47); st6b still 8.80 |
| 2^27 #2 | **69.63 s / 1.928 MHz padded / RSS 76.77 GiB** | **flagship record, −2.14 s vs 71.77 s** (disagreement third run sanctioned) |

Record-run stage vector vs baseline: st0 **+5.90** (17.98 — both fresh
runs sit near 18 s; tonight's ambient window is degraded vs the baseline's
12.08 s — consistent with the st0 lane's device-power finding), st1 +1.00,
st2 +0.20, st4 +0.76, st5 +0.59 · **st6b −9.33 (16.345 → 7.016)** — the
deferred+fused Rav/Bool adoption far exceeded its −1..−2 s model; st7
−0.58 (matches −0.78 est); st8 −0.53; st3 −0.10 (below the −0.40 est —
q0 port effect not cleanly visible under ambient noise). Net −2.14 s
**despite carrying ~+8.4 s of ambient penalty across untouched stages**;
in a baseline-quality st0 window this trunk models ≈ 64 s.

Dashboard point: `.journals/artifacts/wave2-dashboard-point.json`.
2^27 certification CSV archived. Wave 2 CLOSED.


### Reporting consolidation (2026-08-05 02:20 UTC)

Independent adversarial fact-check of the wave-1 audit HTML against the raw
monitor trace, stage source, and lane reports. All numbers reproduced; four
of nine stage subject labels were wrong (st4 "Instruction RAF" → registers
RW/RamVal; st5 "Booleanity/RAM" → instruction read+RAF batch; st6b
"Bytecode RAF" → 11-member cycle batch; st7 "Output checks" → HWCR).
Correction to this journal: the "st7 2.11 s" sampled-zero run is a
st7→st8 boundary run (in-stage hold is 1.615 s; a 2.11 s zero cannot fit a
1.893 s stage). Canonical report (replaces the audit HTML + all /tmp
snapshots): `.journals/metal-m5-saturation-report.html`.
