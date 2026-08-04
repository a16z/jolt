# Metal-sat-fusion scope — radix-4 packed sumcheck rounds (Phase 1, design only)

Status: DESIGN — no code, no cargo, no benches run. Awaiting pro-model oracle
review + orchestrator GO. Evidence base: `.journals/gpu-util.md` (campaign
close ledger), `lane-reports/w15-roundpair-scope.md` (verified file:line
inventory), W2B report (`gpuutil-w2b/.journals/lane-reports/w2b.md`), W3C
report, and direct reads this session of `crates/jolt-sumcheck/src/
{prover,verifier,domain,round_proof}.rs`. Items not yet re-verified in code
are tagged **[P2-pin]** (Phase-2 verification before implementation).

## 0. Verdict up front

**Narrow GO candidate: st4 `RegistersRW` address prefix, vars 0..6, radix-4
pairs (0,1)(2,3)(4,5), var 6 single, non-zk, fail-closed config axis.
Honest prize 1.2–2.2 s @2^27 stage-4. Generic rollout stays dead — now for a
sharper structural reason (§6, the Dory tensor constraint).**

Does radix-4 overturn negative-result #4 ("round-pairing: dead twice over")?
**Partially — the ledger entry conflates three findings, none of which priced
radix-4 on trunk's st4:**

1. w15 measured exposed round-boundary gaps on **already-fused** slots (st5
   5 ms, st6b 0.14 ms, st3 54 ms) and priced the **bivariate two-challenge**
   shape: `(d+1)^2` grid values → 3–3.5× ALU at d5/d6. Radix-4 sends `3d+1`
   values (d3: 10 vs 16; d5: 16 vs 36) — cheaper, but the fused-slot verdict
   stands anyway: there is ~nothing to save there, and §6 blocks their
   coordinates independently.
2. W2B's "pairing confirmed dead for st4" is a **10 ms fused-vs-unfused delta
   measured on W2B's own single-wait rewrite — which was REJECTED** (memory /
   round-speed). Trunk never received that fusion: today's st4 round loop is
   still the legacy two-wait + host-scan shape (W3C changed prepare only,
   "Message/bind kernels and every round representation are unchanged").
   The 1.2–1.8 s sync mass w15 priced was never captured.
3. Neither prior analysis priced pairing as a **bandwidth** play (one table
   generation per pair instead of two, §7) — under the saturation mandate
   that is the point.

So the parked door #2 ("st4 round-loop fusion under a memory-viable
representation — the middle is unexplored") and the fusion lane converge:
**radix-4 on the existing legacy CSR representation is the unexplored
middle** — no representation rewrite (W2B's failure mode not imported), half
the passes/waits/host boundaries, and a sounder one-challenge wire shape.

## 1. Current two-round shape on trunk (reconstruction)

From `jolt-sumcheck/src/prover.rs` (read this session):

```text
prove_batch, per round over max_num_vars variables:
  batched_poly = Σ_k coeff_k · member_poly_k          ← active members
               + Σ_k coeff_k · claim_k / 2            ← inactive members (constant)
  self-check   batched(0) + batched(1) == running_claim
  recorder.absorb_round(batched_poly)  → ONE challenge r      ← FS boundary
  running_claim = batched(r);  member_claims: active ← poly(r), inactive ×= 1/2
  challenge delivered to members fused with the NEXT round's message request
  (ProveRounds::prove_round(bind: Option<F>, round, previous_claim));
  final challenge via finish_rounds(bind).
```

- Wire (clear): `CompressedLabeledRoundPoly` — omits `c_1`, verifier recovers
  it from `s(0)+s(1) = 2c_0 + c_1 + … + c_d = running_sum`
  (`round_proof.rs:102-136`). One `LabelWithCount(SUMCHECK_ROUND_LABEL, n)`
  absorb + one squeeze per round.
- Verifier (`verifier.rs::verify`): per round — degree bound, then
  `domain.check_round_sum(round, running_sum, poly)`, absorb, squeeze, set
  `running_sum = poly(r)`. **The clear verify path is already generic over
  `SumcheckDomain`** — the boolean-hypercube functional is coefficients
  `[2,1,1,…,1]`; uniskip rounds use `CenteredIntegerDomain::new(n)` whose
  functional is the domain power sums (`domain.rs:107-118`). This is the
  exact machinery a radix-4 round needs with `n = 4`.
- Uniskip precedent (`prover.rs::prove_uniskip_clear`): a one-challenge
  multi-variable round already exists as a first-class round type — full
  (uncompressed) coefficients, own transcript label
  (`UNISKIP_ROUND_TRANSCRIPT_LABEL`), centered-integer-domain sum check, one
  squeeze, output claim re-absorbed. Radix-4 = the same round type with
  domain size 4, embedded in the batch loop.
- st4 batch geometry (w15 §3, member table): `RegistersRW` 34 rounds at
  offset 0; `RamVal` 27 rounds tail-aligned (activates at var 7). Vars 0..6
  are RegistersRW-only, Metal-only — 7 address variables, K = 2^7 = 128.
- st4's Metal slot is **the sole unfused slot** (w15 §1): per round r>0 —
  `message()` pass + wait (counts + partials), host prefix-scan + exact
  alloc, `bind()` pass + wait. 13 passes / 13 waits / 6 host boundaries
  across vars 0..6. All other Metal slots fuse bind+eval in one command
  buffer (slot header contract, `metal/slots/mod.rs:14-32`).
- Trunk st4 @2^27 after W3C: stage ≈ 8.1–8.3 s = prepare ≈ 2.0 s + rounds
  ≈ 6.0 s; the device-only prefix was measured at 5.862 s wall, 30.5%
  GPU-eq, 3.73 s sampled-0% (w15 §2; W3C touched prepare only).

## 2. The mandated construction: one radix-4 round

### 2.1 Exact `q(Z)` definition

Let the fused pair be variables `(x, y)` (the two the next two single rounds
would bind), `P` the current partially-bound composed polynomial, and `d` the
member's per-round degree bound (RegistersRW: d = 3).

**Domain.** `D = {z_w} = CenteredIntegerDomain::new(4)` = `{s, s+1, s+2, s+3}`
with `s = centered_domain_start(4)` (expected `s = −1`, i.e. `D =
{−1, 0, 1, 2}`) **[P2-pin: read the helper's exact start]**. Fixed public
cell identification: `w = 2·x_hi + x_lo ∈ {0,1,2,3}`, `z_w = s + w`, where
`x_lo` is the variable the unfused schedule would bind first. The bijection
is pure convention (no soundness content) but MUST match the slot's array
stride at each fold level **[P2-pin against the CSR layout]**.

**Packed input extension.** For each multilinear input `f` of the relation:

```text
f̂(Z, rest) := Σ_{w∈{0,1}²} L_w(Z) · f(w, rest),     deg_Z f̂ ≤ 3,  f̂(z_w) = f(w)
```

with `{L_w}` the Lagrange basis on `D`. For `D = {−1,0,1,2}`:

```text
L_{-1}(Z) = −Z(Z−1)(Z−2)/6        L_0(Z) = (Z+1)(Z−1)(Z−2)/2
L_1(Z)  = −(Z+1)Z(Z−2)/2          L_2(Z) = (Z+1)Z(Z−1)/6      (Σ_w L_w ≡ 1)
```

**Round message.**

```text
q(Z) := Σ_rest REL( f̂_1(Z,rest), …, f̂_m(Z,rest) ),      deg q ≤ 3d
```

`q(z_w) = Σ_rest P(w, rest)` — the four boolean pair-cell partial sums sit ON
the domain; no off-cube diagonal is involved anywhere.

### 2.2 Interpolation / evaluation points

Prover evaluates `q` at `3d+1` points: the 4 domain points plus `3d−3`
small-integer extension points continuing the centered pattern
(d=3: `D ∪ {−2, 3, −3, 4, −4, 5}`, 10 points total — final list mirrors the
existing uniskip extension convention **[P2-pin]**). Extending a stored
4-group to point `t` costs 4 fmadd per input per point with a precomputed
`4 × (3d+1)` Lagrange-coefficient LUT (constant, small integers → cheap
Montgomery forms). Host interpolates the 10 evals to coefficients
(precomputed 10×10 inverse Vandermonde; negligible) and sends **full
coefficients** (uniskip wire precedent; a compressed-packed form that omits
one coefficient recoverable from the D-sum is a later door, not the
prototype).

### 2.3 Verifier identity (per packed round)

For coefficients `(c_0, …, c_{3d})` and running claim `c`:

```text
Σ_{z∈D} q(z) = ⟨coeffs, power_sums(D)⟩ = c
d=3, D={−1,0,1,2}:
4c_0 + 2c_1 + 6c_2 + 8c_3 + 18c_4 + 32c_5 + 66c_6 + 128c_7 + 258c_8 + 512c_9 = c
```

— exactly `CenteredIntegerDomain::new(4).check_round_sum(…)`, which exists.
Then: enforce `deg q ≤ 3d` (schedule-typed bound, NOT the batch-wide
`max_degree`), absorb, squeeze ONE challenge `r`, set `c' = q(r)`, push the
packed challenge. Continuation binding on both sides: every input folds
radix-4, `f ← Σ_w L_w(r)·f(w,·)`; by construction the folded statement's true
sum IS `q(r)`.

### 2.4 Batched context / RLC

Packed rounds are restricted to **single-active-member spans** (st4 vars 0..5
qualify: RegistersRW only). Inactive members contribute the constant
`coeff_k · claim_k / 4` (Σ_D const = 4·const preserves the claim; evaluation
at `r` gives claim/4 = two halvings), and the engine recurrence becomes
`member_claims[inactive] ×= 1/4` on a packed round — the packed analogue of
today's `claim/2` / `×= 1/2` (`prover.rs:247-251, 315`). The batch schedule
becomes variable-indexed: `[P4, P4, P4, S, S×27]`, Σ arity = 34; membership
windows stay in variable space; a packed round must sit strictly inside a
constant-membership span — never across the var-6/7 join.

### 2.5 Transcript order & domain separation

- Batch prelude unchanged (input claims absorbed, RLC coefficients squeezed
  before the loop — `jolt-verifier-derive/src/lib.rs:466-513` per w15).
- Packed round: `LabelWithCount(RADIX4_ROUND_TRANSCRIPT_LABEL, 3d+1)` + full
  coefficients, then ONE squeeze. Distinct label ⇒ no transcript-stream
  collision with single rounds or uniskip.
- The schedule is **not** proof-authoritative: the verifier constructs it
  from its own `JoltProtocolConfig` and rejects shape mismatch fail-closed
  (w15 Option-A pattern; config axis
  `sumcheck_schedule ∈ {Single, St4RegistersPrefixRadix4V1}`, equality-
  validated in `validate_proof_config`, included in the transcript preamble).
- zk: `packed && zk` rejected before parsing. BlindFold constrains one
  univariate + boolean-hypercube sum per round; packed support is the w15
  Option-B 1–2-week item — explicitly out of cut.

### 2.6 Odd-round handling

7-variable prefix → 3 pairs + 1 single, single placed LAST (vars 0-1, 2-3,
4-5 packed; var 6 single). Fold tree: 128 → 32 → 8 → 2 → 1. Rationale:
pairs align with the span start and the leftover single abuts the join,
where the next round is a join anyway (RamVal's first active round). General
rule for any future span: `⌊n/2⌋` pairs + trailing single; never fuse across
a membership change or into a member's first/last active round asymmetrically.

## 3. Soundness argument

**Reduction claim.** One packed round reduces
`c = Σ_{(x,y)∈{0,1}²} Σ_rest P(x,y,rest)` to
`c' = Σ_rest REL(folded inputs)(rest)` with round-by-round error `≤ 3d/|F|`.

**Completeness.** `q(z_w) = Σ_rest P(w,rest)` ⇒ `Σ_D q = S`; the check passes
with `c = S`, and `c' = q(r)` equals the folded statement's true sum by the
definition of `q` (same `L_w(r)` weights on both sides).

**Soundness.** Suppose `c ≠ S`. Any adversarial `q*` with `deg ≤ 3d` either
fails `Σ_D q* = c` (rejected immediately), or satisfies it — then
`Σ_D q* = c ≠ S = Σ_D q` ⇒ `q* ≠ q` as polynomials ⇒ they agree on at most
`3d` points ⇒ `Pr_r[q*(r) = q(r)] ≤ 3d/|F|`. Otherwise `c' = q*(r) ≠ q(r)` =
true sum of the folded statement — the falsehood propagates to the next
round, ultimately to the final claim checked against openings. Per pair:
`3d/|F| = 9/|F|` (d=3) vs `2d/|F| = 6/|F|` for the two single rounds
replaced: +3/2^254 per pair, +9/2^254 for the prefix — negligible at BN254.
Fiat–Shamir: `q*` is absorbed in full, under a distinct label, before `r` is
squeezed; message space is degree-bounded; standard RBR→FS composition is
unchanged.

**Why this is not the unsound naive scheme (mandated distinction).** The
naive scheme sends two degree-d univariates — `u(X) = Σ_y h(X,y)` and, after
`r` is known, `v(Y) = h(r,Y)` — reusing ONE challenge for both bindings, with
checks `u(0)+u(1) = c`, `v(0)+v(1) = u(r)`, continuation `v(r)`. Attack for
`d ≥ 2`: `h ← h + Δ`, `Δ(X,Y) = γ·X(X−Y)`:

```text
Σ_cube Δ = Δ(1,0) = γ            ← claim shifted by γ (false claim accepted)
Σ_y Δ(r,y) = 2γr² − γr           ← v-consistency shifts identically on both sides
Δ(r,r) = γ·r·(r−r) = 0           ← the cheat erases itself at the diagonal
```

⇒ accepted with probability 1. The blind spot: the only cross-check between
the two messages lives on the diagonal `X = Y = r`, where `X−Y` vanishes
identically; the (0,1)/(1,0) cells are never constrained against the
continuation. (For d = 1 the attack needs the `X²` monomial, outside the
degree box, and `(X−Y)·const` sums to 0 over the cube — hence "unsound
exactly for d ≥ 2".)

Radix-4 has no such seam: there is ONE polynomial; "claimed sum" and
"continuation" are two functionals of that one polynomial, and the sum
functional reads all four cells directly. The attack's image — cell values
`(0, 0, γ, 0)` — lifts to a packed correction `q_Δ` with `Σ_D q_Δ = γ ≠ 0`:
**caught deterministically by the D-sum check.** Any correction with
`Σ_D = 0` shifts no claim (no cheat); anything else survives only the
`3d/|F|` Schwartz–Zippel event, which then propagates. Never implement the
two-message shape, including as a "fallback".

## 4. Per-relation degree table (packed economics)

Degrees per w15 §4 (sources: `subprotocols/*` degree constants, verified by
that lane). "seq pts" = evaluation points across the two replaced single
rounds (2·(d+1)); "biv pts" = old bivariate grid ((d+1)²).

| member | d | packed deg 3d | packed pts 3d+1 | seq pts | biv pts | packed legal? |
|---|---:|---:|---:|---:|---:|---|
| **RegistersRW (st4)** | **3** | **9** | **10** | 8 | 16 | **YES — target** |
| RamVal (st4 tail) | 3 | 9 | 10 | 8 | 16 | no (join/CPU member) |
| OuterRem (st1) | 3 | 9 | 10 | 8 | 16 | no (§6) |
| Shift / InstrClaim / RegClaim / RamRAF / Inc / HWCR | 2 | 6 | 7 | 6 | 9 | no (§6 / too small) |
| InstrInput / RamRW / ProductRem / Bytecode / RamH / RamRA | 3 | 9 | 10 | 8 | 16 | no (§6) |
| InstrRA (st6b) | 5 | 15 | 16 | 12 | 36 | no (§6) |
| InstrReadRAF (st5) | 6 | 18 | 19 | 14 | 49 | no (§6; 94.2% GPU-eq) |

Radix-4 vs bivariate at the old no-go degrees: d5 16 vs 36 pts, d6 19 vs 49
— the 3–3.5× ALU objection softens to ~1.4×, **but §6 kills those surfaces
independently of ALU**, so the no-go stands.

## 5. Where the packed challenge goes (blast radius)

After a packed round the two bound variables carry the weight 4-vector
`(L_{00}(r), L_{01}(r), L_{10}(r), L_{11}(r))` — NOT a point `(a,b) ∈ F²`:
the tensor (rank-1) condition `L_{00}·L_{11} = L_{01}·L_{10}` is a nontrivial
quadratic identity in `r`, false for all but O(1) challenge values. Every
downstream consumer of the st4 address point must consume **weights**, not
per-variable scalars:

1. **st4 verifier `expected_final_claim`** — eq factors between the input
   point `r'_addr` (from st3 RegistersClaimReduction, upstream/unaffected)
   and the bound point generalize per pair to
   `Σ_w L_w(r) · eq((r'_i, r'_{i+1}), bits(w))` — O(4) per pair,
   verifier-computable. **[P2-pin: exact factor list in the stage-4 params]**
2. **RegistersValEvaluation (st5 "RegVal" CPU member)** — proves the Val
   claim at st4's point; its prover builds a K=128 `eq_addr` table: tensor of
   per-pair 4-vectors instead of seven 2-vectors — same table size/shape,
   different constructor. **[P2-pin]**
3. **Opening accumulator bookkeeping** — the register-address opening-point
   segment needs a packed representation (4 challenge scalars + schedule tag,
   expandable to the 128-entry weight vector on demand). **[P2-pin: every
   `r_address` slice/len assert on this path]**

## 6. The Dory constraint — why st4 is the ONLY big legal site

Dory (and any tensor-structured PCS) opens committed polynomials at tensor
points `⊗_i (1−r_i, r_i)`. A packed coordinate's rank-2 weight 4-vector
cannot appear in a committed opening point; splitting the claim into the 4
boolean-cell openings quadruples claims per input per pair — dead.

**Legality rule: radix-4 may bind only coordinates that never reach a
committed-polynomial opening point.** Committed witnesses (CLAUDE.md,
`zkvm/witness.rs`): `RdInc`, `RamInc`, `InstructionRa(d)`, `BytecodeRa(d)`,
`RamRa(d)`, advice. Consequences:

- Every **cycle** phase everywhere grounds in `RdInc`/`RamInc`/`Ra` openings
  → blocked.
- Every other **address** phase grounds in a committed one-hot
  (`InstructionRa`, `BytecodeRa`, `RamRa`) → blocked (st5 prefix, st6a/st6b
  address work, RAM).
- **Registers are the unique Twist instance with no committed one-hot**:
  ra/wa/Val are virtual (K=128 small), and the committed neighbor `RdInc` is
  cycle-indexed only. st4's 7 address variables ground exclusively in
  virtual-claim machinery (§5 items 1–3) → **legal**.

This is the sharper structural reason generic pairing stays dead, independent
of the old ALU/sync accounting. It also explains why Jolt's existing uniskip
lives only in Spartan's first rounds: those coordinates land in
verifier-computable R1CS key structures, never in PCS openings — radix-4
extends the same privilege to the one other place that has it.

## 7. Metal kernel / pass / buffer design

Representation: **unchanged legacy CSR** (entries/offsets/inc/operand lanes,
W3C's parallel prepare untouched). Per pair (vs today's 4 passes, 4 waits,
2 host boundaries):

- **Pass A — packed message** (1 dispatch + 1 wait): stream the current CSR
  level once; per output row, gather its 4 child register-rows under the
  current fold level; for each of the 10 evaluation points: extend each
  input 4-group (4 fmadd per input per point, LUT-driven) and accumulate
  `REL` into threadgroup partials; emit next-level union occupancy counts in
  the same sweep.
- **Host boundary** (1): read 10 partials + counts; interpolate → absorb →
  squeeze `r`; compute the 4 fold weights `L_w(r)`; prefix-scan counts;
  allocate the quarter-level table (existing W3C scan machinery, halved
  frequency).
- **Pass B — radix-4 bind** (1 dispatch + 1 wait): sorted 4-way union per
  row; write compacted quarter-level entries folded with `L_w(r)`; bind the
  companion lanes (`rd_inc` etc.) in the same pass.

Prefix totals: 13 passes/waits → 7; host scan/alloc boundaries 6 → 3;
intermediate CSR levels 64/16/4 never materialize (skip odd generations:
128→32→8→2). New buffers: none persistent — partials grow 4→10 slots ×
threadgroups (KiB), plus one constant Lagrange LUT (~40 field elements per
level). CPU optimized twin implements the identical packed math (w15 rule 3:
a mid-batch device failure must fall back to the SAME wire shape — packed
rounds cannot silently revert to singles after the schedule is
transcript-bound). **[P2-pin: exact kernel signatures against
`metal/slots/registers_read_write.rs:117-313` and the shader]**

## 8. Estimates @2^27 (bounded, honest)

| axis | today (trunk) | radix-4 | delta |
|---|---|---|---|
| prefix wall | ≈5.3–5.9 s, 30.5% GPU-eq, 3.7 s sampled-0% | — | **−1.2…−2.2 s stage-4** (gate floor −1.2) |
| passes/waits (prefix) | 13 / 13 | 7 / 7 | −46% |
| host boundaries | 6 scan/alloc | 3 | −50% |
| DRAM traffic (round loop) | R+W per level, 7 generations | 4 generations, one R per pair | ≈−40–45% |
| message ALU | 4 pts × levels | 10 pts × ¼-groups + 4-tap extensions | ≈1.6–2.2× (vs 2× already priced by w15's model; idle headroom 70%) |
| wire bytes (prefix) | 7 compressed d3 = 21 elems ≈ 672 B | 3×10 + 1×3 = 33 elems ≈ 1056 B | +384 B/proof — noise |
| verifier | 7 absorbs/squeezes | 4; D-sum O(10) each; eq factors O(4)/pair | negligible |
| memory | — | no new peak; skipped CSR levels slightly negative | ≈0 (W2B repeat impossible: no representation change) |

Prize band deliberately kept at w15's 1.2–1.8 s plus a bounded bandwidth
increment → **1.2–2.2 s**; the two effects overlap (w15's discount already
charged doubled message ALU), so do not sum them.

## 9. Implementation cut (post-GO)

| area | files |
|---|---|
| jolt-sumcheck | `batch.rs` (variable-indexed schedule, `RoundKind`), `prover.rs` (packed loop arm, inactive ×1/4, `ProveRounds::prove_round_packed` with default `Err`), `recorder.rs` (full-coeff packed absorb + label), `verifier.rs` (packed arm via `CenteredIntegerDomain::new(4)` — clear path already domain-generic), `proof.rs`/`round_proof.rs` (full-coeff round in a compressed stream, `RADIX4_ROUND_TRANSCRIPT_LABEL`) |
| jolt-verifier | `config.rs` (schedule axis, fail-closed equality, zk rejection), `proof.rs`/`verifier.rs` (preamble), `stages/stage4/registers_read_write_checking.rs` (packed eq factors), opening-point packed segment |
| jolt-prover | stage4 driver schedule; `stages/stage5/registers_val_evaluation` eq-table constructor |
| jolt-kernels | `metal/slots/registers_read_write.rs` + shader (packed-message kernel, bind-4 kernel), runtime IDs; `optimized/registers_read_write.rs` CPU twin |
| tests | packed algebra units (q(z_w) grid vs brute-force cell sums; Σ_D q = claim; q(r) vs brute-force folded sum; fold-weight expansion), CPU↔Metal packed parity, tamper suite (§10), e2e accept 2^22/2^24, config/zk rejection matrix |

Not touched: `committed.rs`, BlindFold (beyond fail-closed rejection), Dory,
every other stage.

## 10. Tamper tests (all must reject)

1. Coefficient tamper on a packed round → D-sum or final-claim failure.
2. **Δ-transplant (load-bearing):** shift the claim by γ and add the packed
   lift of `Δ = γX(X−Y)`'s cell values `(0,0,γ,0)` → D-sum catches it
   deterministically (asserts the exact naive-scheme attack is closed).
3. Degree escalation: 3d+2 coefficients → schedule-typed degree bound.
4. Schedule tamper: 7-single stream against packed config, and 3-packed
   stream against `Single` config → fail-closed before transcript work.
5. Join-crossing pair (vars 6-7) → schedule construction/validation reject.
6. Two-challenge bivariate replay → FS divergence.
7. `packed && zk` → `validate_proof_config` reject.
8. e2e witness tamper (flip one register write) → opening check reject.

## 11. Kill gate (velocity v3)

- **Design gate (now):** oracle review + orchestrator GO. No code before GO.
- **2^24 gate (max 2 timed runs):** st4 stage −15% AND registers-prefix span
  −25% vs same-tree baseline; else kill without polish (W1A lesson).
- **2^25 confirm (pair):** st4 −12%; surface any cross-stage >+2% to the
  orchestrator with SoC-pressure framing (W3C amendment) rather than
  self-killing.
- **2^27:** orchestrator-run certification A/B at wave close; floor −1.2 s
  stage-4; peak-footprint delta ≤ +0.5 GiB at every scale.
- **Soundness gates:** full tamper suite green; e2e accept; integrated suite
  once at wave close (velocity rule 1).

## 12. Phase-2 pin list (verification before code, in order)

1. `centered_domain_start(4)` exact value + uniskip extension-point
   convention (`jolt-poly/src/lagrange.rs`).
2. RegistersRW relation & message construction: confirm d=3 wire bound,
   Gruen split-eq usage (W2B mentions "Gruen binding" — prototype packed
   rounds bypass the Gruen linear-factor trick and send full degree-9;
   Gruen-packed is an optimization door, no soundness content).
3. Slot CSR stride/bit-order at each fold level → fix the cell bijection.
4. Enumerate every consumer of st4's address point (§5) — grep the opening
   accumulator path; confirm ra/wa/Val claims never enter the Dory set.
5. Batch prelude `claimed_sum` padding algebra with a 34-var/31-round
   schedule (the `mul_pow_2(max − rounds)` scale is variable-count-based —
   confirm no round-count dependence).
