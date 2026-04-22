# Spec v2: BN254 Fr Native-Field Coprocessor (refactor-crates)

| Field       | Value                                                                |
|-------------|----------------------------------------------------------------------|
| Author(s)   | sdhawan                                                              |
| Created     | 2026-04-13 (v1) / 2026-04-17 (v2)                                    |
| Updated     | 2026-04-21                                                           |
| Status      | Phase 1 DONE; Phase 2a DONE; Phase 2b DONE (R1CS + trace integration + real-guest smoke); Phase 3 OPEN (bridge Module) |
| Scope       | `refactor/crates` worktree at `jolt-refactor-crates/` only           |
| See also    | `native-field-registers-plan.md` for the A/B/C execution history     |

## Summary

Goal: a BN254 Fr coprocessor for the refactor-crates architecture. A single-cycle
`FieldOp` instruction operates on a dedicated 16 × 256-bit field-register file; Fr
values cross the integer/field boundary via per-limb moves
(`FMovIntToFieldLimb` / `FMovFieldToIntLimb`). Target asymptote: ~17 traced
cycles per `Fr::mul`.

Per Markos's framing: **"field inline is just another read/write memory-checking
instance."** The FR Twist is authored as a standalone `Module` in the refactor's
`ModuleBuilder` / `Schedule` / `ClaimFormula` API — a sibling to the existing RAM
and Registers Twists — then glued into the Jolt protocol as an extra Stage-2
batched instance. Soundness closure (binding the Fr values the Twist commits to
against the u64 limbs the guest actually passed) is a single contained bridge
sumcheck, authored as its own Module in Phase 3.

The refactor's architecture eliminates two structural issues by construction: the
`Cycle<T>`-typed per-instruction trace gives a global cycle index natively, and
F-valued `SparseRow` + unified `ClaimFormula` make the prover/verifier Az·Bz
reconstruction symmetric. The remaining work is a design problem — authoring the
bridge sumcheck and the associated R1CS constraints — not a representational
hunt.

## Intent

### Goal

Deliver a BN254 Fr coprocessor on refactor-crates — single-cycle `FieldOp`
over a 16 × 256-bit field-register file, limb-register SDK ABI, sound binding
from integer-register limbs to Fr values — in three phases:

- **Phase 1 — Standalone Twist Module.** Author a `jolt_compiler::Module` that is
  *only* the FieldReg Twist: 16 slots × 256-bit cells, per-cycle read/write events,
  two-phase segmented sumcheck (cycle-binding, then address-binding), claim
  reduction stage, standalone prove/verify harness on `MockCommitmentScheme`.
  Passes honest-accepts and adversarial-rejects tests. No CPU, no RISC-V, no R1CS.
  **DONE** (Plans A + C).
- **Phase 2a — Protocol fold.** Fold the Phase 1 Module into the main Jolt
  protocol Module as an additional Twist instance alongside RAM and Registers.
  Runs through prove/verify on a synthetic empty-events trace alongside the
  baseline muldiv. **DONE** (Plan B).
- **Phase 2b — Real Fr ingestion.** Widen `FieldRegEvent` to `[u64;4]`, port the
  FieldOp instruction + tracer hook + BN254 Fr inline SDK, wire FieldOp
  arithmetic R1CS constraints, land an honest-accepts + adversarial-rejects
  test on a guest program that actually calls `Fr::mul/add/sub/inv`. **OPEN.**
- **Phase 3 — Limb-to-Fr bridge.** Author a single bridge sumcheck that proves
  `FieldOpA / FieldOpB` on each FieldOp cycle equals `Σ_k limbs_k · 2^(64k)`
  where `limbs_k` are the u64s tracked by the Registers Twist via
  `FMovIntToFieldLimb`. The only Fr-in-R1CS point of contact; a single
  contained Module. Closes the soundness gap. **OPEN.**

### Invariants

1. **Natural-form representation.** FieldReg cells store natural-form `[u64; 4]`.
   The R1CS interface uses `F::from_natural_limbs(limbs)` everywhere
   field-register-adjacent. No apparent-Montgomery reinterpretation.
2. **Single access per cycle.** Each Fr SDK source op decomposes into a sequence
   of single-cycle guest instructions. No Fr-coprocessor cycle emits more than
   one FieldReg event.
3. **FieldReg Twist consistency.** Every read of slot `f` at cycle `t` returns
   the last value written to `f` before `t`. Phase 1's Module proves this.
4. **Prover/verifier claim/constraint synchronization.** Enforced mechanically
   by the `ClaimFormula` / `SumcheckDef` pairing — both sides evaluate the same
   `Formula` against the same `Schedule`/`VerifierSchedule`. No hand-mirroring
   of `input_claim` / `input_claim_constraint`.
5. **Acceptance is load-bearing.** "Honest witnesses accept, mutated witnesses
   reject" is the binary pass/fail for every Phase.

### Non-goals

- 32-byte atomic memory loads (RAM Twist stays u64-grain).
- Multi-field coprocessor — BN254 Fr only.
- G1/G2/Gt operations or pairings.
- Rust-compiler-managed field-register allocation.
- Recursive Jolt verifier.

## Architecture — the refactor's Module pattern

### Why the refactor makes this tractable

A new coprocessor Twist on the refactor is a self-contained Module instead of a
cross-cutting edit. The Module pattern gives us:

- **Protocol / Module / Executable pipeline** (`jolt-compiler/`): `Protocol::new`
  declares dims/polys/sumchecks; `compile()` produces a `Module` (a `Schedule` of
  `Op`s + `VerifierSchedule`); `link(module, backend)` produces a backend-specific
  `Executable` whose `prove()` calls are mechanical dispatch.
- **Uniform ClaimFormula**: a single `Formula` carries the sumcheck polynomial
  identity for both prover evaluation and verifier reconstruction — no separate
  "streaming path" vs "structural path" to keep in sync.
- **Per-instruction `Cycle<T>` buffers**: each instruction's tracer emits a
  typed per-cycle record; global cycle indices are tracked by the outer `Trace`
  container — no per-burst collision hazard.
- **BufferProvider** is the single extension point for witness ingestion. A new
  coprocessor Twist implements BufferProvider's three methods (materialize, release,
  lookup_trace) and hands its output to the Module via the same `Polynomials`
  entry-point the main Jolt protocol already uses.
- **Mock commitment scheme** (`jolt-openings::mock`) allows standalone
  prove→verify loops without Dory plumbing. This is what makes Phase 1 feasible.

### The Module authoring pattern (template: RAM Twist)

Primary reference: `crates/jolt-compiler/examples/jolt_core_module.rs` lines
~1681–2230 (RAM Twist Stage 2 module construction) and ~3137–3723 (Register Twist
module construction). The RAM Twist is the closer analog — single access per cycle,
no separate read/write address. Pattern:

```rust
// 1. Declare polynomials and challenges on the ModuleBuilder
let inc = builder.add_poly(PolynomialId::RamInc, "Inc", PolyKind::Committed, log_t);
let val = builder.add_poly(PolynomialId::RamVal, "Val", PolyKind::Virtual, log_k);
let wa  = builder.add_poly(PolynomialId::RamWa,  "Wa",  PolyKind::Virtual, log_t+log_k);
// ... Ra, ReadValue, WriteValue ...
let ch_gamma_rw = builder.add_challenge("rw_gamma");
let ch_tau_rw   = builder.add_challenge("rw_tau");

// 2. Phase-1 cycle-binding kernel
kernels.push(KernelDef {
    spec: KernelSpec {
        formula: Formula::from_terms(vec![
            ProductTerm { coefficient: 1, factors: vec![
                Factor::Input(EQ_TABLE), Factor::Input(WA), Factor::Input(INC),
            ]},
            // γ-batched Ra · (ReadValue) term ...
        ]),
        num_evals: params.rw_checking_degree + 1,
        iteration: Iteration::Dense,
        binding_order: BindingOrder::LowToHigh,
        gruen_hint: Some(GruenHint {
            eq_input: EQ_TABLE,
            eq_challenges: rw_cycle_eq_challenges.clone(),
            q_lincombo: LinComboQ {
                a_input: WA, b_input: INC, c_input: RA,
                gamma_challenge: ch_gamma_rw,
            },
        }),
    },
    inputs: vec![/* EqTable, Provided(Wa), Provided(Inc), Provided(Ra), ... */],
    num_rounds: params.log_t,
    instance_config: None,
});

// 3. ScalarCapture at phase boundary
schedule.push(Op::ScalarCapture { ... });

// 4. Phase-2 address-binding kernel (analogous structure over log_k rounds)
//    ...

// 5. ValEvaluation stage (single sumcheck over Val MLE)
//    ...

// 6. ClaimReduction stage (final reduction of openings)
//    ...

// 7. Hand Module off to link() and prove() via the standard harness
```

### The standalone prove/verify harness (template: e2e.rs)

Primary reference: `crates/jolt-zkvm/tests/e2e.rs` (prove_verify_roundtrip at L81).
The four-step API:

```rust
let module = compile(&protocol, &params, &config, &poly_ids)?;
let backend = CpuBackend;
let executable = link(module, &backend);

let mut polys = Polynomials::<Fr>::new(poly_config);
polys.push(&[CycleInput::PADDING; 4]);
polys.finish();

let mut provider = ProverData::new(&mut polys, r1cs, derived, preprocessed);

let mut transcript = Blake2bTranscript::<Fr>::new(TRANSCRIPT_LABEL);
let proof = prove::<_, _, _, MockPCS>(
    &executable, &mut provider, &backend, &(), &mut transcript, prover_config,
);

let vk = JoltVerifyingKey::<Fr, MockPCS>::new(&executable.module, (), r1cs_key);
verify(&vk, &proof, &[0u8; 32]).expect("proof should verify");
```

Phase 1 will mirror this exact shape with `build_field_register_twist_protocol()`
replacing `build_protocol()` and a dedicated `FrRegStandaloneProvider` replacing the
toy `Polynomials::push([CycleInput::PADDING; 4])` filler.

## Phase 1 — Standalone FieldReg Twist Module (DONE 2026-04-20)

Shipped in Plans A + C of `native-field-registers-plan.md`. Passes 11/11
honest + adversarial acceptance tests (5 adversarial mutations across Inc, Ra,
Val, ReadValue, WriteValue, plus commitment tampering). Witness routed through
the canonical `DerivedSource::with_field_reg` + `FieldRegConfig` path — no
test-bypass provider.

### Module shape

Three stages, mirroring the RAM Twist structure scoped to FieldReg semantics:

1. **`FrRegClaimReduction`** — reduces the Twist's exposed claims (ReadValue,
   WriteValue, Val openings at the challenge point). In Phase 1 the test
   harness asserts the claim equals ground-truth from the witness.
2. **`FrRegReadWriteChecking`** — the heart. Two-phase segmented sumcheck:
   ```
   WriteValue(j) = Wa(j) · (Val(j) + Inc(j))
   ReadValue(j)  = Ra(j) · Val(j)
   ```
   γ-batched across the two. Phase 1 binds cycles first (log_T rounds),
   captures scalar, then binds addresses (log_K = 4 rounds).
3. **`FrRegValEvaluation`** — evaluates the Val MLE at the Stage-2 binding
   point; standard increment-over-LT-of-cycle sum.

### Dimensions

- `K = 16` slots (log_K = 4).
- `T = 2^log_T` cycles. Test exercises `log_T ∈ {4, 10}`.
- Cell width: 256 bits, represented as one `F` (Fr scalar).
- Event shape: at most one write + up to two reads per cycle, flattened to a
  sparse matrix of (cycle, slot, value, is_write) triples.

### PolynomialId aliasing (Phase-1-only)

Phase 1 repurposes the existing `Ram*` `PolynomialId` variants — the FR Module
is structurally identical to a single RAM Twist. Phase 2b renames to dedicated
`FieldRegInc / FieldRegVal / FieldRegWa / FieldRegRa / FieldRegReadValue /
FieldRegWriteValue` so both Twists coexist cleanly (partial rename already
shipped; audit and finish is a 2b item).

### Files

```
crates/jolt-compiler/examples/field_register_module.rs         — Protocol + Module authoring
crates/jolt-equivalence/tests/field_register_twist_standalone.rs — Harness + 11 tests
crates/jolt-witness/src/derived.rs                             — FieldRegConfig, FieldRegEvent
```

### Acceptance gate

`cargo nextest run -p jolt-equivalence --test field_register_twist_standalone`
must be green. This is the regression signal Phase 2+ rely on.

## Phase 2 — Glue into the main Jolt Module

Phase 2 breaks into two halves. **Phase 2a (protocol fold)** is DONE. **Phase 2b
(actually running Fr arithmetic)** is where the remaining refactor-only work lives.

### Phase 2a — Stage-2 protocol fold (DONE 2026-04-21)

Shipped in Plan B of `native-field-registers-plan.md` via
`crates/jolt-compiler/examples/jolt_core_module_with_fieldreg.rs`. FR Twist
is folded as the 6th Stage-2 batched instance with:

- pre-sumcheck γ_fr squeeze + RV/WV bind+evaluate+record+absorb,
- 2-phase kernel (Gruen-hinted segmented cycle phase + Dense address phase with scalar
  captures for eq_bound/inc_bound),
- `first_active_round = stage2_max_rounds - (log_t + log_k_chunk)`, `batch_coeff` = new
  External slot,
- verifier mirror with `fr_output_check` referencing `EqEvalSlice{offset=log_k_chunk}`
  and `SE_FR_{RA,VAL,INC}` (SE_BASE=3 after RV/WV pre-batch evals).

Tests: `modular_self_verify_with_fieldreg` passes; baseline `modular_self_verify`
and all 11 standalone FR tests still pass. Clippy clean.

What this proves: the FR Twist sumcheck protocol is sound when driven from a
synthetic empty-events `FieldRegConfig`. What it does NOT yet prove: that the
Twist correctly ingests real Fr values emitted by a guest program — that's
Phase 2b.

### Phase 2b — Real Fr ingestion + FieldOp arithmetic (DONE 2026-04-21)

Phase 2a landed the witness+sumcheck plumbing with `FieldRegConfig::events
= vec![]`. Phase 2b fills in everything required for a guest
`#[jolt::provable]` function that actually calls `Fr::mul/add/sub/inv` and
produces a cryptographically-enforced trace.

Shipped:
- ✅ `#55` `FieldRegEvent` widened to `[u64; 4]`.
- ✅ `#58` Dedicated `PolynomialId::FieldReg*` variants.
- ✅ `#61` `FieldOp` + `FMov{I2F,F2I}` RISC-V encoding (opcode 0x0B / funct7
  0x40) + tracer hook.
- ✅ `#57` Guest-side `jolt-inlines-bn254-fr` SDK with single-asm-block ABI
  (x10..x17 live across FieldOp).
- ✅ `#59` FADD/FSUB R1CS gates (rv64.rs rows 19-20) binding
  `FieldOpA + FieldOpB = FieldOpResult`.
- ✅ `#49` FMUL/FINV R1CS gates (rv64.rs rows 21-26) via `V_PRODUCT` reuse —
  no `NUM_PRODUCT_CONSTRAINTS` bump, cross-verify parity preserved.
- ✅ `#62` Non-empty-events e2e test via `FieldRegConfig` injection.
- ✅ `#63` Spartan outer-uniskip widened to actually sample rows 19-26
  (baseline modules stay at 19 via `ModuleParams::new_with_constraints`).
  Verified via tampered-FADD-rejection test through the full prove/verify.
- ✅ `#60` Real guest smoke: `bn254-fr-smoke-guest` compiles to RISC-V ELF,
  traces correctly, FieldRegEvents carry the expected FieldOpPayload with
  `a = x[10..=13]` / `b = x[14..=17]` at FieldOp time.

Measured performance (`crates/jolt-equivalence/tests/bn254_fr_smoke.rs`):
- Isolated Fr arithmetic: **~13 cycles per Fr op** via the SDK.
- ark-bn254 software baseline: **~2500 cycles per Fr op**.
- **~190× speedup** on the Fr math alone. Within striking distance of the
  ~250× theoretical ceiling — residual gap is the 4-cycle-per-limb FMov
  load that would vanish with compiler-managed FR-register allocation
  (explicitly out-of-scope for Phase 2b).

Known follow-ups (deferred, not blocking feature completeness):
- End-to-end prove/verify for `bn254-fr-smoke-guest` through the modular
  prover — blocked on generalizing `setup_zkvm_muldiv_with_example` and
  deriving PCS setup without routing through jolt-core. The existing
  FADD-payload e2e test (`modular_self_verify_with_fieldreg_fadd_payload`)
  already validates the full modular pipeline on synthetic events, so the
  real-guest variant is demo-value rather than soundness-value.

## Phase 3 — Limb-to-Fr bridge sumcheck

### Soundness gap (what the bridge closes)

Once Phase 2b has the FieldReg Twist ingesting real Fr events and the R1CS
FADD/FSUB/FMUL/FINV gates firing (task #63 — DONE), the remaining gap is the
binding between the Fr values the Twist commits to and the u64 limbs the guest
actually passed via `FMovIntToFieldLimb`. Without this binding, a malicious
prover can commit arbitrary self-consistent Fr values into `FieldRegInc` and
matching `FieldOpA/B/Result` openings — all Twist sumchecks and R1CS rows
check out, but nothing ties the result to the guest-provided limbs.

### Locked-in SDK ABI (enables the simple bridge)

The `jolt-inlines-bn254-fr` SDK emits every `Fr::{add,sub,mul,inv}` call as a
**single inline-asm block** with fixed register bindings:

```
a.limbs[0..4]  →  a0..a3    (x10..x13)     live across the FieldOp cycle
b.limbs[0..4]  →  a4..a7    (x14..x17)     live across the FieldOp cycle
out.limbs[0..4] ←  a8..a11  (x18..x21)
```

The 8 FMov-I2F loads + the FieldOp + the 4 FMov-F2I stores live in one asm!
block, so the compiler cannot clobber x10..x17 between loading the limbs and
executing the FieldOp. At the FieldOp cycle, `x[10..=13]` hold a's limbs and
`x[14..=17]` hold b's limbs — by construction.

### Bridge formulation

The fixed ABI collapses the bridge to a **single equality per FieldOp cycle**:

```
FieldOpA(r_cycle)  ==  Σ_{k=0..3}  Val_reg(10+k, r_cycle) · 2^{64·k}
FieldOpB(r_cycle)  ==  Σ_{k=0..3}  Val_reg(14+k, r_cycle) · 2^{64·k}  [FMUL/FADD/FSUB only]
```

#### Sumcheck identity

Recast as a single sumcheck over (reg, cycle) ∈ [0, K_reg) × [0, T):

```
0  ≡  Σ_{reg, cycle}  eq(τ_bridge, cycle) · IsFieldOp(cycle) · [
          IsLimbRegA(reg) · 2^{64·(reg-10)} · Val_reg(reg, cycle)
        − IsAnchorReg(reg) · FieldOpA(cycle)
      ]
```

Where:
- `IsFieldOp(c) = IsFieldMul(c) + IsFieldAdd(c) + IsFieldSub(c) + IsFieldInv(c)`
  — at most one FieldOp flag fires per cycle so the sum is 0 or 1.
- `IsLimbRegA(reg)` — indicator `1 for reg ∈ {10,11,12,13}`, else 0. Selects the
  4 int-register slots that hold `a`'s limbs under the SDK ABI.
- `IsAnchorReg(reg)` — indicator `1 for reg = 10`, else 0. Exactly one per-reg
  "anchor" so the `FieldOpA(c)` contribution is counted once per cycle.
- `2^{64·(reg-10)}` weighting function — evaluable at any `reg ∈ {10,11,12,13}`.
- `eq(τ_bridge, ·)` — Fiat-Shamir random weights per cycle that collapse the
  sum to a per-cycle equality (standard sumcheck).

The B-side identity is identical with `IsLimbRegB = 1 for reg ∈ {14..17}`,
`IsAnchorReg_B = 1 for reg = 14`, and gating changed to `IsFieldMul + IsFieldAdd
+ IsFieldSub` (not `IsFieldInv`) — FINV doesn't read `b` so `x[14..17]` is
uninitialised and must not be bound.

Both A and B identities batch into one sumcheck via γ-combination.

#### Claim-reduction coordination

The bridge's output claim opens `Val_reg(r_reg_bridge, r_cycle_bridge)` at a
random (r_reg_bridge, r_cycle_bridge) pair. `Val_reg` is a **Virtual**
polynomial — not committed — so it cannot open directly into Dory. It is
reduced to the committed `RdInc` by Stage 5's `RegistersValEvaluation`
sumcheck.

The bridge's point `(r_reg_bridge, r_cycle_bridge)` differs from Stage 4's
`Val_reg` opening point, so two `Val_reg` openings must be reduced. The
canonical precedent is `IncClaimReduction` (`jolt_core_module.rs:5084-5159`,
claim formula at `:5353-5404`), which γ-batches four openings of
`RamInc`/`RdInc` at distinct points into a single reduction. Stage 5's
`RegistersValEvaluation` is extended identically:

- Input claim becomes γ-batched two-term: `Val_reg@Stage4 + γ_bridge ·
  Val_reg@bridge`, referenced via `ClaimFactor::StagedEval { poly: p.reg_val,
  stage: 3 }` for Stage 4's opening and `ClaimFactor::Eval(p.reg_val)` for
  the bridge's (just-closed) opening.
- Reduction output: two independent openings of the committed `RdInc` at the
  two respective points, batched into Stage 6's existing `IncClaimReduction`.
- No new Val_reg commitment; no new Dory opening proof.

Additional opening claims the bridge emits:
- `FieldOpA(r_cycle_bridge)` — opens the R1CS column (already in the Stage-1
  remaining-sumcheck's r1cs_input_polys evaluation output via `#63`).
- `FieldOpB(r_cycle_bridge)` — same.
- `IsFieldMul/Add/Sub/Inv(r_cycle_bridge)` — opens `OpFlag(14..17)` at the
  bridge cycle. Again already in the r1cs_input_polys output.

All four R1CS-column openings require the bridge's `r_cycle_bridge` to
coincide with the Stage-1 remaining-sumcheck's cycle challenge. The bridge's
Stage-2 placement makes this coincidence automatic — Stage 2 instances
already consume `stage1_cycle_challenges` via their eq tables (FR Twist does
this at `jolt_core_module_with_fieldreg.rs:2458`).

#### Implementation shape — Option A (chosen)

**Decision**: fold bridge into Stage 2 as an additional batched instance; do
NOT author a new `Stage::FieldRegLimbBridge`. Rationale:

1. `jolt_core_module_with_fieldreg.rs:1200-1208` explicitly warns against
   new `BeginStage` ops — the prover emits 8 stages but the verifier
   `num_stages: 4` and the mapping is hand-coded many-to-one. Adding a
   stage forces untangling this mapping.
2. Stages = Fiat-Shamir challenge windows; instances-within-a-stage = sumchecks
   that share a window. The bridge's γ_bridge and τ_bridge depend only on
   `after_stage: 1` challenges already available to Stage 2.
3. The FR Twist chose the same path (`instances[5]` of Stage 2's batch,
   authored at `:2417-2530`) — existing precedent. A hypothetical second
   coprocessor would follow the same pattern; stage proliferation is the
   anti-pattern.

Concretely:
- Module: `crates/jolt-compiler/examples/jolt_core_module_with_fieldreg.rs`
  (the FR-extended Module only — baseline `jolt_core_module.rs` stays
  bridgeless and cross-verifies with jolt-core).
- Add `γ_bridge = ch.add("gamma_bridge", ChallengeSource::FiatShamir {
  after_stage: 1 })` in the Stage-2 pre-sumcheck block alongside `γ_fr`.
- New batched instance pushed to `stage2_batched_sumchecks` at
  `InstanceIdx(6)`, authored adjacent to `fr_phase2_kernel`. Bump
  `STAGE2_NUM_INSTANCES_FR` 6 → 7.
- Extend Stage 5 `RegistersValEvaluation` input-claim formula to two-term
  γ-batched form per the `IncClaimReduction` template.
- Verifier mirror: add `SumcheckInstance` entry to Stage-2 verifier op with
  `at_stage: VerifierStageIndex(1)` — `num_stages: 4` stays put.

Kernel parameters:
- `num_rounds = log_K_reg + log_t` (= 7 + log_t).
- `degree = 3` (eq × IsFieldOp × Val_reg; the `FieldOpA` term is degree 1 in
  cycle, 0 in reg).
- `input_claim = 0` (trivial — the identity is of the form Σ … = 0).
- `output_check` evaluates the full per-cycle, per-reg expression at the
  final (r_reg, r_cycle) point using `LagrangeKernelDomain { domain_start:
  10, domain_size: 4 }` for the A-side indicators (mirror at `14` for
  B-side) and explicit `2^{64k}` powers-of-2 for weights.

#### Honest/adversarial acceptance criteria

- Honest guest (SDK-emitted trace): bridge sumcheck accepts.
- Tampered prover that commits `FieldOpA` ≠ actual limb sum: bridge rejects.
- Tampered prover that commits matching FieldOpA but mutates Val_reg at one
  of reg ∈ {10..13}: rejected by the Registers Twist (Val is already bound).

These two adversarial tests — one per row of the identity — are the
acceptance gate for Phase 3.

### Why this actually closes the gap

- Honest guest: loads a's limbs from `a.limbs[k]` into x[10+k] via Rust asm
  input bindings, FMov-I2F copies each into `field_regs[1][k]`, FieldOp reads
  `field_regs[1]`. Bridge equality holds.
- Malicious prover: to lie about `FieldOpA`, must either (1) commit
  `V_FIELD_OP_A` ≠ actual FR Twist opening of `field_regs[1]` at that cycle
  — rejected by the R1CS gate + FR Twist binding — or (2) commit bogus
  `RegVal(10+k)` — rejected by the Registers Twist. Both Twists are
  already cryptographically bound.

### Deferred items

- FINV-is-zero handling: the R1CS FINV gate doesn't cover `FINV(0) = 0`.
  Guest must not invert zero; the bridge sumcheck inherits this restriction.
- Fr-across-mul-chains optimization (avoid reloading limbs between chained
  multiplies) — depends on compiler-managed field-register allocation.

## Design decisions

### Core

- **Natural-form `[u64;4]` representation** for FieldReg cells and the R1CS
  interface. `F::from_natural_limbs(limbs)` everywhere field-register-adjacent.
  No apparent-Montgomery reinterpretation.
- **Limb-register SDK ABI.** Fr values cross the SDK boundary as `[u64;4]`
  through integer registers — no RAM round-trip. Enables the ~17 cycles/mul
  target.
- **16 × 256-bit FieldReg file, single-cycle FieldOp.** One write + up to two
  reads per FieldOp cycle. No Fr-coprocessor cycle emits more than one
  `FieldRegEvent`.
- **MSB-first limb load order** for `FMovIntToFieldLimb` — matches the Horner
  recurrence that Phase 3's bridge sumcheck will prove.
- **Standalone Module first, then glue.** Phase 1 proves the Twist in isolation
  with honest+adversarial tests; Phase 2a folds it into the main protocol;
  Phase 2b wires real Fr arithmetic. Each layer has an independent pass/fail
  signal.
- **Soundness closure is one contained Module** — the Phase 3 limb-to-Fr
  bridge. Not a distributed fix.
- **Adversarial-rejects tests are mandatory acceptance criteria** at every
  phase. Honest-only tests do not count.

### Out of scope

- **FLOAD / FSTORE / FLoadHorner ISA encodings.** Memory-backed Fr is not
  needed for the limb-register SDK ABI. Not reintroducing unless a concrete
  consumer demands it.
- **Fr-through-registers across mul chains.** Getting below ~17 cycles/mul
  toward the ~250× theoretical ceiling depends on compiler-managed
  field-register allocation. Revisit after Phase 3 lands.

## Naming (Phase 2b target)

Phase 2b introduces dedicated `PolynomialId` variants so RAM and FieldReg Twists
coexist without aliasing:

- `PolynomialId::FieldRegInc` — Twist Inc poly (committed, dense, cycle-major)
- `PolynomialId::FieldRegVal` — Twist Val poly (virtual, sparse)
- `PolynomialId::FieldRegWa` / `FieldRegRa` — write and read address one-hot polys
- `PolynomialId::FieldRegReadValue` / `FieldRegWriteValue` — Fr values read/written
  on FieldOp and FMov cycles

Per-Module stages within the FR Twist Module:
- `FrRegReadWriteChecking` — the 2-phase segmented sumcheck (Phase 1)
- `FrRegValEvaluation` — Val MLE eval at the Stage-2 binding point
- `FrRegClaimReduction` — claim reduction stage

## References

### Primary refactor-side files

- `crates/jolt-compiler/examples/jolt_core_module.rs` — RAM Twist template
  (L1681–2230), Register Twist template (L3137–3723).
- `crates/jolt-compiler/src/builder.rs` — `ModuleBuilder` API.
- `crates/jolt-zkvm/tests/e2e.rs` — standalone prove/verify harness template.
- `crates/jolt-openings/src/mock.rs` — `MockCommitmentScheme` used for Phase 1.
- `crates/jolt-witness/src/lib.rs` — `Polynomials`, `BufferProvider`,
  `CycleInput`.
- `crates/jolt-verifier/src/lib.rs` — `JoltVerifyingKey`, `verify`,
  `ProverConfig`, `TRANSCRIPT_LABEL`.
- `crates/jolt-zkvm/src/prove.rs` — `prove()` entry point.
- `crates/jolt-cpu/src/lib.rs` — `CpuBackend` used by Phase 1.

### Open tasks

**Phase 1 — DONE.**
- ✅ Standalone FR Twist Module + 11-test honest/adversarial harness
  (`crates/jolt-equivalence/tests/field_register_twist_standalone.rs`).

**Phase 2a — DONE.**
- ✅ FR folded as 6th Stage-2 batched instance in
  `crates/jolt-compiler/examples/jolt_core_module_with_fieldreg.rs`;
  `modular_self_verify_with_fieldreg` passes alongside baseline.

**Phase 2b — PARTIALLY DONE (R1CS + bridge shipped, trace integration remains).**

Done:
- ✅ `#55` Widen `FieldRegEvent { old, new }` from `u64` to `[u64;4]`.
- ✅ `#58` Dedicated `PolynomialId::FieldReg*` variants (RAM aliases removed).
- ✅ `#61` FieldOp encoding (opcode 0x0B, funct7 0x40, funct3 selector) +
  tracer hook + FMov{I2F,F2I} instructions. Event stream plumbed to witness.
- ✅ `#57` `jolt-inlines-bn254-fr` SDK crate with `Fr::{add,sub,mul,inv}`.
- ✅ `#59` FieldOp arithmetic R1CS: FADD/FSUB gates (rows 19/20, direct bind
  against `V_FIELD_OP_{A,B,RESULT}`).
- ✅ `#49` FMUL/FINV gates (rows 21-26, routed through existing `V_PRODUCT`
  to avoid bumping `NUM_PRODUCT_CONSTRAINTS`). Unit-tested accept+reject for
  all four gates; cross-verify with jolt-core still passes.
- ✅ `#62` Non-empty-events test using synthetic `FieldRegEvent`s (honest +
  two adversarial mutations). Exercises FR Twist sumcheck, NOT the R1CS
  gates — those require real FieldOp cycles (see below).
- ✅ `FieldOpPayload` + `apply_field_op_events_to_r1cs` bridge
  (`jolt-host`): writes V_FLAG_IS_FIELD_* + V_FIELD_OP_* + V_LEFT/V_RIGHT/
  V_PRODUCT + V_LOOKUP_OPERAND_* columns from a `FieldOpPayload` stream.

Recently completed (this session):
- ✅ CycleRow for `FieldOp` / `FMov{I2F,F2I}` (special-cased in
  `tracer_cycle.rs`; funct3-driven flag dispatch for FieldOp).
- ✅ `r1cs_cycle_witness` populates `V_FLAG_IS_FIELD_*` from CycleRow flags.
- ✅ Bench harness applies `apply_field_op_events_to_r1cs` after witness
  build (`jolt-bench/src/stacks/modular.rs`).
- ✅ Integration tests in `jolt-host` (`real_field_op_cycle_through_
  witness_builder`) + `jolt-equivalence` (`modular_self_verify_with_
  fieldreg_fadd_payload`) exercising real `Cycle::FieldOp` through the
  witness builder and the event overlay.

### CRITICAL SOUNDNESS GAP — task #63

The FADD/FSUB/FMUL/FINV R1CS rows (matrix indices 19-26) are authored in
`jolt-r1cs/src/constraints/rv64.rs` and pass `check_witness` unit tests, but
they are **not sampled by Spartan's outer uniskip sumcheck**.
`jolt-compiler/src/params.rs::NUM_R1CS_CONSTRAINTS = 19` caps the uniskip
domain at the original eq-constraint count, so rows 19-26 are silently
dropped during prove/verify.

A tampered-result rejection test
(`modular_self_verify_with_fieldreg_fadd_tampered_result_rejects`) exposed
this: FieldOpResult=580 when a+b=579 still verified successfully end-to-end.

Remediation (see task #63): bump `NUM_R1CS_CONSTRAINTS` to 27, recompute
`UNISKIP_DOMAIN_SIZE` / `NUM_GROUP2_CONSTRAINTS` / `outer_uniskip_*`,
regenerate both `jolt_core_module*.rs` to emit the new dimensions, and
manage the cross-verify regression (jolt-core also hardcodes 19). Substantial
multi-session protocol change.

Until #63 lands, Phase 2b is **functionally complete at the matrix level**
(honest paths prove + verify; R1CS unit tests accept/reject correctly) but
**not soundness-complete end-to-end** — a malicious prover can supply
incorrect FieldOpResult values and produce a verifying proof.

Other remaining #60 subitems:
  5. Minimal `#[jolt::provable]` guest calling `Fr::mul/add/sub/inv` in a
     loop; exercise the inline SDK's assembly emission path.
  6. Adversarial FieldOp test — blocked on #63 until Spartan enforces the
     rows.

**Phase 3 — OPEN.** Soundness closure:

- `#52` Limb-to-Fr bridge sumcheck (new `SumcheckId::FieldRegLimbBridge`,
  Stage 3). Required for any security-sensitive guest using Fr.

**Orthogonal track — OPEN.**

- `#56` ZK mode for FR Twist on refactor-crates. `crates/jolt-blindfold/` is
  currently an empty stub; multi-week substrate work independent of Phase
  2b/3. Deferred until the non-ZK productization path is proven end-to-end.

### Commands (Phase 1)

```bash
cd /Users/sdhawan/Work/jolt-refactor-crates

# Core correctness (full refactor suite):
cargo nextest run --cargo-quiet

# Phase 1 smoke:
cargo nextest run -p jolt-equivalence field_register_twist --cargo-quiet

# Lint:
cargo clippy -p jolt-compiler -p jolt-compute -p jolt-openings -p jolt-zkvm \
    -p jolt-equivalence --all-targets -- -D warnings
```
