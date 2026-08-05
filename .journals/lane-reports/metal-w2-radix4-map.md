# Metal-w2-radix4 Phase-1 map — sound production radix-4 live-path sites

Status: DESIGN MAP — no code, no Cargo, no benches run this session. Extends
`metal-sat-fusion-scope.md` (construction + soundness argument) and
`metal-sat-r4-pin.md` (stage-4 invariant ledger); supersedes neither. All new
code citations below were read in this worktree this session.

Hard constraints honored throughout:

- One univariate message `q(Z)`, degree ≤ 3d, ONE challenge per packed round.
  The two-polynomial/shared-challenge shape is forbidden (diagonal kernel
  `Δ = γ·X(X−Y)` accepted w.p. 1 for d ≥ 2 — fusion-scope §3).
- A packed (quaternary Lagrange) factor is a rank-2 weight 4-vector, not two
  scalar coordinates. Ordinary-MLE/Dory opening points cannot carry it;
  `U = XY` distinguishes digit interpolation from ordinary binding. Any
  candidate whose factor reaches `PCS::open/verify/open_batch/verify_batch`
  is illegal unless commitment/opening semantics change (typed Dory factor —
  deferred, §5).
- The stage-4 **cycle-prefix** packing (`[P4,P4,P4,S,S×27]` on the current
  CSR cycle rows) is ruled **invalid** by the campaign (commit `c5adf05f9`,
  orchestrator directive 2026-08-04). Not re-proposed; excluded from the
  candidate set.

## 0. Verdict up front

**No unconditional radix-4 GO exists in the live prover today.**

**Best (and only) sound live-path site: stage-4 `RegistersRW`
address-first — quaternary address phase `[P4,P4,P4,S]` over the seven
virtual register-address variables bound BEFORE the 27 cycle variables,
followed by a dense binary cycle phase.** This is the campaign-close bounded
probe ("modeled prize 1.5–2.5 s @2^27, kill if the address phase exceeds
0.15 s @2^24") with the radix-4 shape riding its new address phase, plus the
missing piece the r4-pin demanded: a concrete performant address-first `Val`
state algorithm (§3.4).

Verdict: **conditional GO**, gated on (1) GPT-5.6 Pro verdict on the exact
polynomial shape in §4 — mandatory before any production code — and (2) the
isolated jolt-eval address-phase objective meeting the 0.15 s @2^24 gate.
Honest attribution: the 1.5–2.5 s prize belongs to the address-first
restructuring (dense cycle collapse replacing the measured 13-pass sparse
prefix); radix-4's own increment is the address-phase cut 7→4 messages and
6→3 host boundaries — bounded, secondary, and killable independently (§3.8).

Everything else surveyed is NO-GO with reasons (§2, §5).

## 1. Legality rule (sharpened)

Radix-4 may bind only coordinates whose factor dies before any committed
opening point. Verified in this tree, this session:

- The final batched PCS point is assembled **only** from the stage-6
  IncClaimReduction cycle challenges, the stage-7 hamming-weight address
  challenges, and precommitted anchors
  (`jolt-claims/src/protocols/jolt/geometry/committed_openings.rs:147-197`).
- `IncClaimReduction` consumes the four upstream cycle points (st2 RamRW,
  st4 RamValCheck, st4 RegistersRW, st5 RegistersValEval) **only** as
  `try_eq_mle(fresh_point, upstream_cycle)` publics; both reduced openings
  anchor at the reversed fresh sumcheck point
  (`jolt-verifier/src/stages/stage6b/inc_claim_reduction.rs:112-146`).
- Akita/lattice: the four inc claims join the lattice ReadRAF address fold
  **by value** at `γ^5..8`; cycle points enter as `StageCycleEq` publics;
  `FusedInc` anchors at the fresh shared stage-6b cycle point
  (`jolt-claims/src/protocols/jolt/lattice/relations/read_raf.rs`).
- st5 RegistersValEvaluation consumes st4's cycle slice only via
  `LtPolynomial::evaluate(fresh_cycle, st4_cycle)` and emits output points
  `[st4_address || fresh_cycle]`
  (`jolt-verifier/src/stages/stage5/registers_val_evaluation.rs:78-119`).
- `LtPolynomial::evaluate` and `eq_mle` are multilinear in every coordinate
  of both arguments (`jolt-poly/src/lt.rs:125-134`); both admit exact O(4)
  per-pair packed arms (weight the 4 cell-substituted transition steps).

Consequence: coordinates that terminate in claim-reduction eq/LT publics are
packable (typed-factor plumbing); coordinates of stage-6b/stage-7 sumchecks
themselves ARE the final opening coordinates and are not (without typed
Dory). The stage-4 register-address factor was already proven PCS-clean with
a complete consumer inventory (r4-pin §3/§4, invariant 4 GO); the fresh
verifications above re-confirm its two terminal routes from this tree's code.

## 2. Site survey

| site | coords | factor reaches PCS? | measured pass prize | verdict |
|---|---|---|---|---|
| st4 addr-first address phase | 7 virtual register-address vars, R-only span | **no** (r4-pin inv.4 GO; §1) | replaces the measured 13-pass/5.86 s/30.5%-GPU-eq/3.73 s-idle sparse prefix via dense collapse; radix-4 slice = 3 passes + 3 host boundaries of the new phase | **conditional GO — the candidate (§3)** |
| st4 cycle prefix (current CSR) | first 7 cycle bits | — | — | **excluded: ruled invalid, commit `c5adf05f9`; do not build** |
| st4 address tail (current order) | 7 addr vars after 27 cycle folds | no | none — tables are 128 elements by then; host phase is µs-scale | NO-GO (no prize) |
| st1 Spartan outer-remaining | post-uniskip cycle vars | no (virtual z; key evals verifier-computed) | stage 4.456 s @2^27 but round-loop vs uniskip-message attribution **unmeasured**; slot already fused | NO-GO now — measurement door: attribute st1 round loop first |
| st2/st3 batches | cycle vars | no | st3 gaps measured 54 ms total; host-dominated (16% GPU) | NO-GO (prize ≤ noise) |
| st5 InstrReadRAF | cycle phase, d=6 | no (Ra claims reduce via st7 HWCR) | 94.2% GPU-eq prefix — no idle headroom; packed 19 pts vs 14 sequential = ALU-negative | NO-GO (saturated) |
| st6b claim reductions | shared st6b cycle point | **yes** — st6b bound point IS the IncCR/FusedInc opening point | idle is gather/host-glue (gaps 0.14 ms); unattributed | NO-GO — blocked on typed Dory factor (§5) |
| st7 HWCR | address point | **yes** — st7 point IS the final address point | stage is prepare-dominated (1.887 s host) | NO-GO (§5) |
| st0/st8 | commitment / Dory opening | n/a | no sumcheck rounds to pack | out of domain |

## 3. The candidate, pinned

### 3.1 Protocol coordinates and schedule

- Config: `registers_rw_phase1_num_rounds = 0`,
  `registers_rw_phase2_num_rounds = 7` — the existing dimension setting for
  address-first; the point builder consumes phase-2 address challenges before
  phase-3 cycle challenges and still emits canonical `address || cycle`
  (`jolt-claims/.../geometry/dimensions.rs:144-179`, verified by r4-pin;
  knobs at `jolt-prover/src/config.rs:145-152`). Only the kernels reject
  address-first today — the geometry does not.
- Batch globals: `[P4(k0,k1), P4(k2,k3), P4(k4,k5), S(k6) | S×27 (cycle)]`.
  34 semantic variables, 31 messages. Globals 0..6 are RegistersRW-only;
  RamValCheck stays tail-aligned at global 7 and now shares exactly
  RegistersRW's 27 cycle variables (cleaner than today's mixed overlap).
  No pair crosses the message-3/4 activation join; the address single sits
  last, abutting the join (fusion-scope §2.6 rule).
- K folds 128 → 32 → 8 → 2 → 1; intermediate levels 64/16/4 never exist.

### 3.2 Exact relation polynomial, degree, domain, wire

Stage-4 RegistersRW summand (verified:
`jolt-claims/.../relations/registers/read_write_checking.rs:100-113`,
`degree() = 3`):

```text
EqCycle(r3,j)·[ Wa(k,j)·(RdInc(j) + Val(k,j)) + γ·Rs1Ra(k,j)·Val(k,j) + γ²·Rs2Ra(k,j)·Val(k,j) ]
```

During address rounds `EqCycle` and `RdInc` are address-constant. Packed
round over pair `(k_b, k_{b+1})`, cells `w = b_lo + 2·b_hi ∈ {0..3}`
(LowToHigh: first-bound bit is the LSB at the current fold level), domain
`D = CenteredIntegerDomain::new(4) = {−1,0,1,2}`, `z_w = w − 1`:

```text
f̂_g(Z,j) := Σ_w L_w(Z)·f(4g+w, j)          deg_Z ≤ 3,  f ∈ {Wa, Rs1Ra, Rs2Ra, Val}

q(Z) := Σ_{g,j} EqCycle(r3,j)·[ Ŵa_g(Z,j)·(RdInc(j) + V̂al_g(Z,j))
                               + γ·R̂s1_g(Z,j)·V̂al_g(Z,j)
                               + γ²·R̂s2_g(Z,j)·V̂al_g(Z,j) ]
```

- **Degree bound: `deg_Z q ≤ 6` exactly** — every Z-dependent product is one
  one-hot cubic × the `Val` cubic (the `Ŵa·RdInc` term is ≤ 3). Seven
  coefficients `c_0..c_6`; prover evaluates at the ascending centered window
  `[−3,−2,−1,0,1,2,3]` (existing uniskip convention — r4-pin inv.1).
- Lagrange basis on `D`: `L_{−1}(Z) = −Z(Z−1)(Z−2)/6`,
  `L_0 = (Z+1)(Z−1)(Z−2)/2`, `L_1 = −(Z+1)Z(Z−2)/2`, `L_2 = (Z+1)Z(Z−1)/6`;
  `Σ_w L_w ≡ 1`.
- **D-sum check** (exists as `CenteredIntegerDomain::new(4)` power sums,
  `jolt-sumcheck/src/domain.rs:84-117`):

```text
4c₀ + 2c₁ + 6c₂ + 8c₃ + 18c₄ + 32c₅ + 66c₆ = running_claim
```

- Then: schedule-typed degree bound (6), absorb FULL seven coefficients
  under a distinct label, squeeze ONE challenge `r`, set `claim' = q(r)`.
  All five member tables bind with the same weight 4-vector
  `(L_{−1}(r), L_0(r), L_1(r), L_2(r))`.
- Soundness per packed round: `Σ_D q* = claim` forced, `q* ≠ q` agree on
  ≤ 6 points ⇒ error `6/|F|` per pair vs `2+2 = 4/|F|` for the two singles
  replaced (address singles are exact degree 2) — +2/2^254 per pair,
  negligible. Completeness: `q(z_w) =` the four cell partial sums; folded
  statement's true sum is `q(r)` by construction (same `L_w(r)` on both
  sides). The Δ-transplant cheat (cells `(0,0,γ,0)`, claim shift `γ`) has
  `Σ_D = γ ≠ 0` — caught deterministically.

### 3.3 Active/inactive batch schedule

```text
members:  RegistersRW 34 vars offset 0 (active all 31 messages)
          RamValCheck 27 vars tail-aligned (inactive messages 0..3)
padding:  RamValCheck claim ×2^(34−27) = ×128 at batch start (variable-based)
inactive: message 0,1,2 (P4): polynomial contribution = coeff·claim/4, claim ×= 1/4
          message 3 (S):      coeff·claim/2, claim ×= 1/2
          ⇒ ×1/128 total — joins at exactly its claim at message 4  ✓ (r4-pin §5)
degrees:  [6,6,6,2 | 3×27]   (uniform 3 on the S-rounds is sound; 2 is exact
          for the address single)
```

Engine prerequisite (unchanged from r4-pin §5): separate semantic variables
from messages; per-message domain/degree/scaling; mixed full-P4 +
compressed-S wire in proof/recorder/verifier/derive.

### 3.4 Address-first `Val` state algorithm (the previously unpriced piece)

The r4-pin blocker: quaternary address binding needs
`V̂al(g,j) = Σ_w L_w(r)·Val(4g+w, j)` including registers **absent** from the
sparse access row — dense `128·T` state is impossible (512 GiB @2^27).

Proposed algorithm — **run-length folded-Val streams**:

- `Val(k,·)` is piecewise-constant in `j` with breakpoints exactly at writes
  to `k`: total runs ≤ `T + 128`. Store per address-node a cycle-sorted run
  list `(start_cycle, value)`; level-0 lists are the per-register write
  histories (the same information `RegistersRWC::prepare` already scans).
- Quaternary fold of node group `G`: merged breakpoint list of the 4
  children with run value `Σ_w L_w(r)·val_w`; total run count across nodes
  is bounded by the sum of children's runs ⇒ O(T) memory at every level,
  O(T + K_ℓ) merge per round.
- `Wa/Rs1Ra/Rs2Ra` lanes stay ≤ 3-sparse per cycle at every level: one-hot
  in `k` per lane per cycle ⇒ at most one nonzero digit per group ⇒ the
  folded entry is a single scalar with a digit index; `Ŵa(Z) =
  coeff·L_{w₀}(Z)` — a scaled Lagrange basis polynomial. The cycle-rowed CSR
  survives with `col >>= 2` per packed round (columns merging by addition on
  collision).
- Message pass: process CSR rows in cycle order; per access term, 4
  cursor-lookups into the children's run lists (O(1) amortized —
  monotone cursors), extend `V̂al` to the 7 points (4-tap LUT fmadd),
  multiply by `coeff·L_{w₀}(z)` and `EqCycle(j)`, accumulate. ≈ 40–50 fmadd
  per access per point-set ⇒ ≈ 150 fmadd/cycle/packed round.
- Cost model vs binary address-first: radix-4 ≈ 1.7× address-phase ALU,
  −3 full-stream passes, −3 host scan/alloc boundaries, skips K-levels
  64/16/4. Same ALU-for-bandwidth trade the bind4 microbenchmark validated
  at 2^24 (1.51–1.98× on dense tables).
- **Cycle-phase collapse (the probe's real prize):** after the address
  phase, K = 1 — the five tables become four dense T-vectors
  (`wa`, `rs1_ra`, `rs2_ra`, `val` folded; `RdInc` already dense). The 27
  cycle rounds run the already-fused dense IncCR-shaped loop: no CSR walk,
  no per-round host prefix-scan/alloc, one pass per round. This removes the
  trunk-measured sparse prefix (13 passes / 13 waits / 6 host boundaries,
  5.862 s wall, 30.5% GPU-eq, 3.73 s sampled-0% @2^27).

Correctness obligation for the oracle (§4): `V̂al` built from run-list
reconstruction must equal the quaternary digit extension of the true `Val`
MLE — quiet-sibling values are genuinely `Val(k,j)` (constant over the quiet
span), so cell-wise linearity gives equality; the unit test "accesses
spanning different 4-register groups / unaccessed-neighbor cross terms"
(r4-pin test matrix) pins it.

### 3.5 Downstream factor consumers

Exactly the r4-pin §3 inventory (compiled for this factor), with the §4
expansion rule: every virtual consumer uses the same 128-weight expansion
(⊗ of three 4-vectors and one 2-vector), or the O(4)-per-pair closed form;
never feed an encoded factor prefix to `EqPolynomial::evals`.

| consumer | obligation |
|---|---|
| st4 point derivation / final claim | `EqCycle` public is **cycle-only** — unaffected. Point assembly emits typed `address-factor ‖ cycle` (dimensions builder already canonicalizes order) |
| st4 output extraction | replace the 7-scalar address-Eq reconstruction with factor expansion (`optimized/registers_read_write.rs:1348-1387`) |
| point storage | typed factor cells replace `Vec<F>` segments (`stages/relations.rs:62-86`, `stage4/outputs.rs:83-121`) |
| st5 RegistersValEval | `RdWa` address fold / K=128 eq table built from per-pair 4-vectors; `LtCycle` public is cycle-only — unaffected |
| st6a BytecodeReadRAF | stage-value folds consume the register-address prefix → 4-expansion (`stage6a/bytecode_read_raf.rs:40-188`) |
| st6b full/committed-program folds | both register-address Eq vectors expand from factors (`stage6b/batch.rs:250-349`, `geometry/claim_reductions/bytecode.rs:381-479`) |
| st6b IncClaimReduction | cycle slices only; the committed `RdInc` claim's address factor is dummy and removed — unaffected (verified §1) |
| BlindFold | fail-closed reject before transcript (packed ∧ zk) |
| stage-8 (Dory + Akita) | fresh canonical points only — factor never arrives (verified §1) |

### 3.6 Transcript shape and fail-closed config

- `RegistersRwSchedule::{BinaryV1, AddressFirstRadix4V1}` in
  `JoltProtocolConfig`; equality-validated in `validate_proof_config`
  **before** transcript construction (seam verified at
  `jolt-verifier/src/verifier.rs:276-311`, `config.rs:97-118`);
  `AddressFirstRadix4V1 ⇒ Transparent ∧ ¬zk ∧ ¬Akita` in v1 (Akita's folds
  are 4-expandable — later door, smaller v1 test surface).
- Schedule absorbed in both consensus preambles beside the four RW phase
  fields (`verifier.rs:555-627, 753-843`; `jolt-prover/src/stages/stage0.rs:150-159`).
- Packed round wire: `LabelWithCount(RADIX4_ROUND_TRANSCRIPT_LABEL, 7)` +
  full `c_0..c_6`, one squeeze. Distinct label ⇒ no stream collision with
  single or uniskip rounds. Wrong variant/count/degree rejected before
  absorbing the stage's first round. The schedule is config-derived, never
  proof-authoritative.
- Pre-deserialization rejection is not available (protocol field lives in
  the deserialized proof, `proof.rs:53-71`); the enforceable seam is
  pre-transcript. Envelope versioning remains a separate decision.

### 3.7 Tamper tests (all must reject)

1. Packed-round coefficient tamper → D-sum or final-claim failure.
2. **Δ-transplant (load-bearing):** claim shift `γ` + packed lift of cells
   `(0,0,γ,0)` → D-sum catches deterministically (asserts the forbidden
   two-polynomial scheme's attack is closed).
3. Degree escalation: 8 coefficients → schedule-typed bound reject.
4. Schedule replay both directions: 7-single stream vs `AddressFirstRadix4V1`;
   4-message packed stream vs `BinaryV1` → fail-closed pre-transcript.
5. Pair crossing the message-3/4 activation join → schedule construction reject.
6. Bit-order tamper (swap cells w=1/w=2 in the prover) → final-claim mismatch.
7. `AddressFirstRadix4V1 ∧ (zk ∨ Akita ∨ non-transparent)` → config reject.
8. e2e witness tamper (flip one register write) → opening reject.
9. Algebra regression `U = XY`: digit interpolation ≠ ordinary MLE binding.
10. Factor-expansion equivalence: 128-weight expansion == ⊗ brute force;
    quiet-sibling `Val` cross-term parity for accesses spanning different
    4-register groups.

### 3.8 Prize decomposition, jolt-eval objective, kill gates

Honest attribution @2^27:

- **address-first restructuring** (dense cycle collapse): modeled
  **1.5–2.5 s** (campaign close), replacing the measured 5.862 s / 30.5%
  GPU-eq / 3.73 s-idle sparse prefix — minus the new T-scale address phase.
- **radix-4 increment on the address phase:** 7 → 4 messages, 6 → 3 host
  boundaries, skip K-levels 64/16/4; ALU ≈ 1.7× on those scans. Bounded
  ≈ 0.1–0.3 s, possibly ≈ 0 — it must never be sold as the prize owner and
  dies independently at Gate 1 without killing the probe.

**Isolated jolt-eval kernel objective** (pattern verified:
`PerformanceObjective` enum + criterion benches,
`jolt-eval/src/objective/mod.rs:137-158`, `jolt-eval/benches/bind_parallel_low_to_high.rs`):
new variant `RegistersAddressPhase` + bench
`jolt-eval/benches/registers_address_first_phase.rs` — synthetic trace-shaped
CSR + run-list Val state at 2^22/2^24; measures (a) binary 7-message
address-first phase, (b) radix-4 `[P4,P4,P4,S]` phase; asserts bound-table
and claim parity between the two; reports seconds.

Gates (Velocity v3):

- **Gate 0 (now):** GPT-5.6 Pro verdict on §4. No production code before it.
- **Gate 1:** objective @2^24 — total address phase ≤ 0.15 s (campaign kill
  line) AND radix-4 ≤ binary arm; max two timed runs. Fail ⇒ drop the
  radix-4 arm (probe may continue binary) or kill the probe.
- **Gate 2 (2^24 e2e prototype):** st4 stage −15% vs same-tree baseline;
  peak footprint ≤ +0.5 GiB.
- **Gate 3 (2^25 pair):** st4 −12%; cross-stage >+2% escalated with
  SoC-pressure framing, not self-killed.
- **Gate 4 (2^27 certification A/B at wave close):** floor −1.2 s stage-4.
- **Soundness:** full §3.7 suite, e2e accept, integrated battery once at close.

## 4. Oracle brief (send verbatim with fusion-scope §2–3)

Statement for GPT-5.6 Pro: stage-4 batched sumcheck over 34 variables binds
the seven virtual register-address variables first as
`[P4,P4,P4,S]` quaternary-Lagrange rounds (§3.2 polynomial, `deg ≤ 6`, seven
coefficients, `D = {−1,0,1,2}`, D-sum functional `(4,2,6,8,18,32,66)`, one
challenge per round, weight-vector binding on all five tables), then 27
binary cycle rounds shared with a second member that is inactive for the
first four messages (padding ×128, scaling /4,/4,/4,/2). The address factor
is consumed only by the §3.5 virtual folds and never reaches a committed
opening point (final PCS point provenance verified: stage-6 IncCR + stage-7
HWCR challenges only). Questions:

1. Is the packed round sound as stated, error `6/|F|` per pair, under
   standard RBR→FS composition with the distinct label?
2. Is binding-order permutation (address before cycle) sound given the
   config-absorbed schedule and explicitly permuted downstream opening
   coordinates — any seam beyond the §3.5 list?
3. Does the run-length quiet-sibling `Val` reconstruction (§3.4) yield
   exactly the quaternary digit extension of the true `Val` MLE required by
   `q(Z)` — any trace shape where run values diverge from MLE cell values?
4. Is the inactive-member `×1/4` recurrence and ×128 padding exactly
   claim-preserving at the activation join; any degree/batch/transcript
   hole we missed?

## 5. Deferred doors (not candidates now)

- **Typed quaternary Dory factor** (unlocks st6b/st7 packing): commitment is
  extension-agnostic; verifier scalar-fold holds one four-weight factor
  across two reduce rounds (`α₂α₁l₀ + α₁l₁ + α₂l₂ + l₃`); must not straddle
  the row/column split. Blocked on oracle `ec0b50d07d63` AND an st6b
  pass-anatomy measurement (its idle is gather/host-glue, measured gaps only
  0.14 ms — the prize is unattributed). Separate lane after both.
- **st1 outer-remaining packing:** legal (virtual coordinates; dense
  Az/Bz/Cz = bind4's best-measured shape) but prize unmeasured — needs a
  round-loop vs uniskip-message attribution trace first; consumer surface
  (shift `EqPlusOne`, inner key evals, product relations) is the widest in
  the codebase.
- **Gruen-packed wire** (send `h(Z)` with `q = êq·h`): valid degree
  reduction once the base shape is retained; no soundness content.
- **st4 dense cycle-phase packing after address-first:** the collapsed
  cycle loop is bind4's dense shape, but stage-4 cycle factors then flow to
  every `StageCycleEq`/LT consumer — re-opens the cycle-factor question the
  campaign closed. Do not revisit without an explicit orchestrator +
  oracle mandate.

## 6. Session verification ledger

Read in this worktree this session (beyond the three journals):
`relations/registers/read_write_checking.rs:60-139` (relation, d=3);
`stage6b/inc_claim_reduction.rs` (full — eq publics, fresh reversed point);
`stage5/registers_val_evaluation.rs` (full — LT public, point splice);
`geometry/committed_openings.rs:23-197` (final point provenance,
embedding scale); `stage4/registers_read_write_checking.rs` (full — EqCycle
cycle-only); `lattice/relations/read_raf.rs` (full — value join, fresh
FusedInc point); `jolt-poly/src/lt.rs:125-150` (LT structure);
`jolt-prover/src/config.rs:130-165` (phase knobs);
`jolt-eval/src/objective/mod.rs:137-158` + `jolt-eval/benches/` (objective
pattern). No Cargo, no builds, no benches, no production edits.
