# W3D — st1/st2/st3 prepare+glue: dependency & shape analysis (Phase 1)

Lane: gpu/util-w3d off trunk @220529c7c. Targets from the wave-3 attribution
(/tmp/gpu-util-trace-2to27-wave3-20260804.json.gz, instr. walls ~7% inflated):
st1 9.21 s (0.49 of healthy GPU ratio), st2 4.70 s (0.43), st3 2.34 s (0.35).

## 1. Full decomposition (trace re-queried span-by-span, with per-span GPU%/cores)

### st1 = 9.215 s

| span | wall | gpu% | cores | verdict |
|---|---:|---:|---:|---|
| TraceRecord::collect | 4.166 | **5.3** | 16.5 | host walk, GPU idle — THE poison |
| uniskip t1 pass (device `OuterT1`) | 2.068 | 77.3 | 5.2 | already device |
| OuterRemainder::prepare (device `OuterAzbz`) | 1.654 | 84.4 | 4.8 | already device |
| rounds + eq | 0.522 | 93–100 | — | healthy |
| **closing gap**: `claimed_inputs` 35-opening walk + its full-T eq build | 0.796 (+0.158 eq) | **0.0** | 18 | host walk |

The metal slot (`metal/slots/spartan_outer.rs`) runs t1, az/bz materialization,
and all remainder rounds on device. st1's low ratio is entirely the two host
walks bracketing the device work.

### st2 = 4.701 s

| span | wall | gpu% | cores | verdict |
|---|---:|---:|---:|---|
| SpartanProductUniskip::prepare (device `ProductT1` + host eq/wraps) | 0.497 | 31.7 | 13.0 | mixed, small |
| RamReadWriteChecking::prepare | 0.434 | 50.0 | 15.0 | mixed, small |
| ProductRemainder::prepare | 0.290 | 51.0 | 6.0 | small |
| RamRafEvaluation::prepare (incl. 0.160 eq) | 0.344 | 100 | 18 | fine |
| prove_batch rounds | 0.983 | 84 | 7.4 | healthy |
| **tail**: eq 0.149 + gap 0.581 + eq 0.171 + gap 1.247 | **2.148** | **0.0** | 18 | host walks |

**The "~1.8 s unattributed host glue" is IDENTIFIED**: it is the post-rounds
opening-claim computation, not a prepare —
`metal/slots/spartan_product.rs::claimed_inputs` (8 openings) and
`metal/slots/instruction_claim_reduction.rs::operand_claims` (5 openings).
Each builds a **full T-sized eq table** (`EqPolynomial::evals(&reversed, None)`
= 4.3 GiB alloc+fill @2^27, the 0.149/0.171 s eval spans) and then walks all
T cycles. ICR's walk (1.247 s) does 5 **field mults + Montgomery conversions
per cycle** (`weight * Fr::from_u128(...)`); ProductRemainder's walk (0.581 s)
uses the unreduced-accumulator pattern (8 lanes yet 2.1× faster than ICR's 5 —
the field-mult-per-lane is the cost).

### st3 = 2.340 s

| span | wall | gpu% | cores | verdict |
|---|---:|---:|---:|---|
| SpartanShift::prepare | 0.441 | 0.0 | 18 | host Q-table build |
| RegistersClaimReduction::prepare | 0.151 | 0.0 | 18 | small |
| InstructionInput round 0 (host `native_q_evals`, by design) | 0.455 | **0.0** | 18 | host |
| InstructionInput round 1 (device bind-native, 17 GiB table write) | 0.793 | 30.3 | 9.3 | device, bw-bound |
| InstructionInput rounds 2+ | 0.316 | 93–100 | ~2 | healthy |
| SpartanShift + RegistersCR rounds | 0.127 | — | — | small |

InstructionInput's "low util" anatomy: the metal slot deliberately leaves
round 0 on host ("rayon-cheap, not worth a T-sized device promotion" — a
2^22-era judgment; at 2^27 it is 0.455 s of 18-core CPU at 0% GPU) and round 1
is one huge bind dispatch writing 8 dense T/2 tables (~17 GiB) — bandwidth-
bound, wall dominated by the memory write, not launch anatomy.

## 2. Fiat–Shamir dependency map

| work item | needs | available at | hoistable? |
|---|---|---|---|
| TraceRecord::collect (4.17 s) | witness plane + log_t ONLY — zero transcript inputs (signature: `shared(session, witness, log_t)`; tau never enters) | **prove() entry** | **YES → st0 window** |
| uniskip t1 (2.07 s) | tau (drawn after st0 commitments absorbed) | st1 start | no (already device) |
| OuterRemainder az/bz (1.65 s) | uniskip challenge (mid-st1) | mid-st1 | no (already device) |
| st1 claimed_inputs walk (0.96 s) | full st1 bound point | st1 rounds end | no — shape fix only |
| SpartanProductUniskip::prepare (0.50 s) | tau_low = f(st1 remainder point) | st1 end | marginal (overlaps st1 tail at best; both host-saturated — no win) |
| st2 member prepares (~1.6 s) | st2 challenges/gammas | st2 start | no (small, mixed util) |
| st2 claims tail (2.15 s) | full st2 bound point | st2 rounds end | no — shape fix only |
| SpartanShift::prepare (0.44 s) | r_outer (st1 end) + r_product (st2 END) + gamma (st3 draw) | st2 end (gamma-decomposable by linearity) | technically (spawn at st2 rounds-end, γ-split into 5 column walks); plumbing-deep for ≤0.3 s — parked |
| InstructionInput round 0 (0.46 s) | r_product + st3 gamma | st3 start | no — device-port candidate (opportunistic) |

## 3. Parallelism shape (probe, W1D §6 method)

TraceRecord::collect is **already rayon-parallel** (`fill_record_lanes`
recursive join, grain 2^12; the serial `collect_streaming` is only the
no-random-access fallback). Locked 2^24 probe, RAYON_NUM_THREADS ∈ {18, 8, 4}:

| threads | collect wall | core-seconds |
|---:|---:|---:|
| 18 | 0.329 s | 5.9 |
| 8 | 0.608 s | 4.9 |
| 4 | 1.086 s | 4.3 |

Compute-bound decode with GOOD parallel efficiency that *improves* at lower
widths (less DRAM contention). ⇒ "parallelize harder" is not a fix (W3C's
lesson: an 18-core burst is what causes cross-stage SoC pressure); the fix is
**overlap** — move the walk into st0's window where 10 of 18 cores idle
(st0: cpu 44%, gpu 92%).

Window fit @2^27: 69 core-s ⇒ ~7.2 s at 8 background threads (with the
measured efficiency gain) vs st0's 11.3 s window ✓. @2^24 (gate scale):
0.61 s at 8 threads vs st0's 2.68 s window ✓. Both scale ~linearly with T so
the fit is scale-stable.

Memory: hoisting moves the record family's +28 GiB from st1 to st0
(30.4 → ~58 GiB during st0); prove peak stays at st6b's ~77 GiB. Safe at the
2^27 tier, no new peak.

The st1/st2 claims walks are also already 18-core parallel — their fix class
is *make the work cheaper* (kill the 4.3 GiB eq materializations, kill the
per-cycle field mults), not more threads.

## 4. Fix plan (ranked; all protocol-neutral, byte-identical)

### F1 — TraceRecord::collect hoist into st0 (the lane's main prize)

W2A background pattern, adapted for a borrowed witness:

- `prove()` (prover.rs) wraps the stage sequence in `std::thread::scope`
  (precedent: the stage-6b/7 prefetch scope already in that function).
  At scope open, `spawn_trace_record_collect(&mut session, witness, log_t,
  scope)` — the scoped thread takes the **process-wide
  `BACKGROUND_BUILD_TOKEN`** (extend W2A's, never a second pool), builds a
  capped rayon pool (`RECORD_BACKGROUND_THREADS = 8`, env-overridable
  `JOLT_RECORD_BACKGROUND_THREADS`; `JOLT_RECORD_HOIST=off` kill switch for
  A/B), and runs the *extracted* collect body (a pure
  `fn collect_artifacts(witness, log_t) -> Result<RecordArtifacts, _>`
  producing record lanes + RamAccessColumns + SharedInstructionRows +
  PcRows — exactly today's products).
- Result crosses via `mpsc::channel`; the session parks a 'static
  `PrebuiltTraceRecord { log_t, receiver }` (scoped JoinHandles are not
  'static; a Receiver of owned artifacts is).
- `TraceRecord::shared` (unchanged signature): no parked record → take the
  receiver, `recv()` (blocking join), validate log_t, park the four
  artifacts as today. Any failure (disconnected, mismatch, walk error) →
  inline build fallback, identical values (W2A doctrine).
- Token timeline: record build holds the token during st0–st1; W2A's
  booleanity/bytecode builds take it post-st4 — no overlap, one token.

Byte-identity: same deterministic walk, same per-index writes, same parks —
values independent of which thread ran it. Equality oracle: fixture test
comparing background-built vs inline-built lanes (W2A's
background-vs-inline pattern) + `JOLT_RECORD_HOIST=off` same-binary arm.

Modeled @2^27 (instr): st1 −4.17 s minus join residue ~0–0.5 minus st0 tax
0–0.25 ⇒ **−3.4…−4.1 s**. @2^24: st1 −0.33 with st0 flat.

Risk: st0 regression from CPU/DRAM contention (gate clause; pool width is
the lever — 8→6→4). W3C-class aftereffect watch on st4/st5/st8: the hoist
*removes* an 18-core burst from st1 and adds an 8-core one inside st0 —
net SoC pressure should DROP, but measured, not assumed.

### F2 — st2/st1 claims-tail shape fix (tensor eq + unreduced accumulators)

Three sites, one pattern each:

- `metal/slots/instruction_claim_reduction.rs::operand_claims`: replace the
  full-T eq table with the R3-lite split (`e_hi[t>>lo_bits] · e_lo[t&mask]`
  on the fly — exact same field value, mult associativity) AND replace the 5
  per-cycle `weight * Fr::from_*` field mults with
  `SignedProductAccumulator::fmadd_s256` on the raw lanes (ProductRemainder's
  exact pattern; Barrett-reduce per block ≡ Σ mod p — the campaign's
  standard exact-math argument, byte-diff-pinned since wave 0).
- `metal/slots/spartan_product.rs::claimed_inputs`: tensor-split its full-T
  eq table (walk already accumulator-based).
- `optimized/spartan_outer.rs::claimed_inputs` (st1 tail, shared by the
  metal remainder kernel): tensor-split its full-T eq table (walk already
  accumulator-based). Do the optimized ICR/product twins identically so both
  arms carry the same shape (byte-diff 12/12 both arms is the oracle).

Kills 3 × 4.3 GiB alloc+fill+stream @2^27 and ICR's per-cycle field-mult
bill. Modeled @2^27 (instr): st2 tail 2.15 → ~1.0–1.3 (**−0.9…−1.15 s**),
st1 tail **−0.15…−0.25 s**.

### F3 (opportunistic, post-gate) — InstructionInput round-0 device promotion

`native_q_evals` is the same weighted-reduce shape as the slot's existing
round-1 bind dispatch (reads the same native lanes; partials infra exists).
Promote round 0 to a `jk_instr_input_q0` dispatch. ~0.4 s @2^27. Only if the
F1+F2 gates clear early; parity via the slot's lockstep tests.

### Parked (documented, not chartered)

- SpartanShift::prepare γ-decomposition + st2-rounds-end spawn: sound
  (distributivity keeps values exact) but ≤0.3 s for deep driver plumbing.
- st1 claimed_inputs device port (35 mixed-width accumulators): W1A-shaped
  risk for ~0.6 s; the eq fix takes the cheap part.
- st2 member prepares (~1.6 s, mixed 30–100% GPU): no single mass ≥0.5 s.

## 5. Modeled combined win

@2^27 instr: −4.3…−5.4 s ⇒ canonical (÷1.07) **−4.0…−5.0 s** concentrated in
st1 (8.10 → ~4.3) and st2 (4.34 → ~3.4). @2^24 gate arithmetic: st1+st2
combined 1.40 s (this session's probe walls) − hoist ~0.33 − tail ~0.15 ⇒
**−34% ± a few points vs the −25% gate** — clears if the hoist truly hides
(st0 window is 4.4× the 8-thread build at that scale).

No protocol change anywhere in F1/F2 (F3 is also protocol-neutral device
promotion) ⇒ per the charter, proceeding to Phase 2 without GO.
