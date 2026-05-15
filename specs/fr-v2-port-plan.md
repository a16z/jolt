# FR Coprocessor v2 — Port Plan onto Modular-SDK Base

**Status:** Port plan synthesized from 12 parallel investigation agents.
**Source branch:** `feat/fr-coprocessor-v2` (33 FR-specific commits).
**Target base:** `sagar/modular-sdk` (currently HEAD of #1535, atop `origin/jolt-v2/equivalence`).
**Scope:** ~4,300 LOC hand-written + ~1,500 LOC regenerated, across 5 phases.

## Why this is a port, not a rebase

The FR branch was authored against an earlier `refactor/crates` state — predating:

- `jolt-instructions` → `jolt-riscv` rename and the `for_each_instruction_kind!` macro list.
- `jolt-trace` → `jolt-program` rename and the bolt-emit-driven `OracleGeneration` model.
- The `jolt-compiler` → bolt-emit MLIR pipeline (jolt-compiler is gone; oracle identity is now string symbols in `Stage*OpeningInputPlan` records, not `PolynomialId` enum variants).
- `jolt-witness` rescoping to primitive-oracle kernels (no more `derived.rs`/`field_reg.rs`).
- `gen_ram_memory_states` exposure + `tracer::trace_row_from_cycle` exposure (modular-sdk work).

Most FR commits cannot apply as cherry-picks; they must be replayed against new APIs.

## Dependency graph

```
          Agent 01 (CircuitFlags 14→23)
                    │
                    ▼
Agent 02 (CPU state)   Agent 06 (jolt-riscv structs)
       │                            │
       ├─→ Agent 03 (FieldOp) ──────┤
       │           │                │
       │           ├─→ Agent 04 (event stream)
       │           └─→ Agent 05 (5 more instructions)
       │                            │
       │                            ▼
       │                     Agent 07 (13 R1CS rows + slots)
       │                            │
       │                            ▼
       │                     Agent 08 (FR Twist scaffolding)
       │                            │
       │                            ▼
       │                     Agent 09 (3-sumcheck chain) ← HARDEST
       │                            │
       └─→ Agent 10A (SDK drop-in)  │
                  │                 ▼
                  ▼          Agent 12 (audit C1–C11)
           Agent 10B (FieldRegConfig)
                  │
                  ▼
           Agent 11 (Poseidon2 examples)
```

## Phase 1 — ISA & tracer foundations (~600 LOC, mostly cherry-pickable)

| Step | Agent | Scope | New-base touchpoints |
|---|---|---|---|
| 1a | 01 | 9 new CircuitFlags | `crates/jolt-riscv/src/flags.rs:20-49` (enum extension auto-bumps `NUM_CIRCUIT_FLAGS` via `strum::EnumCount`). Then propagate `[bool; 14]` → `[bool; NUM_CIRCUIT_FLAGS]` across 8 downstream consumers (jolt-witness `lib.rs:545`, jolt-verifier `stage6.rs:38,55` + `common.rs:1198`, jolt-kernels `stage6.rs:371` + `stage1/rv64_typed.rs:17`, bolt-emit `stage6.rs:1246,1263`, `verifier_common.rs.template:1198,1578`). |
| 1b | 02 | FieldReg CPU state | `tracer/src/emulator/cpu.rs` — 4 insertion points (lines 157, 185, 350, 1141), ~27 LOC. Adds `FIELD_REG_COUNT=16`, `FieldRegEvent` struct, `Cpu.field_regs: [[u64;4]; 16]` and `Cpu.field_reg_events: Vec<FieldRegEvent>`. |
| 1c | 03 | FieldOp instruction | New `tracer/src/instruction/field_op.rs` (MASK=0xfe00007f, MATCH funct7=0x40 + opcode 0x0B). Split decode arm at `tracer/src/instruction/mod.rs:1078` on funct7. Append `FieldOp` to `for_each_instruction_kind!` at `crates/jolt-riscv/src/lib.rs:22`. Add `ark-bn254` + `ark-ff` deps to `tracer/Cargo.toml`. |
| 1d | 04 | FieldRegEvent stream | Extend `tracer::trace` return tuple from 5-tuple to 6-tuple (additive `Vec<FieldRegEvent>`). Additive `TraceOutput::with_field_reg_events` constructor. Thread through `jolt-core::host::Program::trace`, `guest::program::trace`, `jolt-core::zkvm::prover` callsites. Port `Emulator::take_field_reg_events` + `CheckpointingTracer::take_field_reg_events`. |
| 1e | 05 | 5 more field instructions | New files `tracer/src/instruction/field_{mov,sll64,sll128,sll192,assert_eq}.rs`. Append 5 kinds to `for_each_instruction_kind!`. Expand funct7 decode split for 0x41 (SLL variants). |
| 1f | 06 | jolt-riscv per-instruction structs | New `crates/jolt-riscv/src/instructions/field/{mod,field_mul,field_add,field_sub,field_inv,field_assert_eq,field_mov,field_sll64,field_sll128,field_sll192}.rs` via `jolt_instruction!` macro. **DO NOT add to `LookupInstruction`** — exclusion is intentional (FR ops don't feed RV lookup argument). Matches existing Csrrs/Mret/Amo* pattern. |
| 1g | 10A | bn254-fr SDK drop-in | New `jolt-inlines/bn254-fr/{Cargo.toml, src/lib.rs, src/encode.rs, src/sdk.rs}`. `encode.rs` is pure `const fn`; `sdk.rs` has three modes (host → ark_bn254, RISC-V compute_advice → `VirtualHostIO`, RISC-V pass-2 → inline asm). Register workspace member + path dep. |

**Validation gate after Phase 1:** `cargo nextest run -p bolt --test commitment_ir` still 53/53. muldiv/fibonacci e2e still produce `valid: true` (FR additions are inert until R1CS slots wire them in Phase 2).

## Phase 2 — R1CS shape change (~400 LOC + goldens regen #1)

| Step | Agent | Scope |
|---|---|---|
| 2a | 07 | 13 FR R1CS rows + 12 witness slots (9 flag slots 36..=44 + 3 virtual operands 45..=47). Shift `V_BRANCH`/`V_NEXT_IS_NOOP` to 48/49. `NUM_VARS_PER_CYCLE 38→50`. `NUM_EQ_CONSTRAINTS 19→32`. `NUM_R1CS_INPUTS 35→47`. `num_vars_padded=64` survives. Helper `row_bigcoeff<F>` for `2^128`/`2^192` coefficients. |

**Files touched:** `jolt-r1cs/src/constraints/rv64.rs`, `bolt/src/protocols/jolt/params.rs`, `bolt/src/protocols/jolt/emit/rust/stage6.rs`, `bolt/src/protocols/jolt/verifier_common.rs.template`, `bolt/src/protocols/jolt/phases/stage{1,2,3,6}.rs`, `jolt-kernels/src/stage1/rv64_typed.rs`.

**Goldens regen #1:** A/B/C matrix coefficient hashes refresh + OpFlag enumeration sweep + ~20–40 `commitment_ir` numeric assertion deltas.

**Validation gate after Phase 2:** commitment_ir 53/53 at new R1CS shape; muldiv proves & verifies; FR flag bits inert (no FR instructions executed yet).

## Phase 3 — FR Twist scaffolding (~600 LOC)

| Step | Agent | Scope |
|---|---|---|
| 3a | 08 | New `crates/jolt-witness/src/field_reg.rs` (291 LOC verbatim — `FrLimbs`, `FrCycleData`, `FrCycleBytecode`, `FieldRegEvent`, `replay_field_regs`, `sub_limbs`, 5 unit tests). `LOG_K_FR = 4`. Widen `CycleInput` to `dense: [i128; 3]` + `one_hot: [Option<u128>; 4]`. `num_committed = 3 + instruction_d + bytecode_d + ram_d + field_reg_d`. Add `field_reg_d` to `JoltProtocolParams`. New oracle entries via `crates/bolt/src/protocols/jolt/oracles.rs::append_committed_oracles` — `FieldRegInc` (`DenseTrace`) + `FieldRegRa_{d}` (`OneHotChunk`). Insert replay call in `jolt-host/src/lib.rs` between `extract_trace_rows` and `commitment_trace_sources`. |

**Critical finding:** `OracleGeneration` enum needs **no new variant**. All FR polys fit existing `DenseTrace` / `OneHotChunk` / `Reference`. Virtual oracles (`FieldRegVal`, `Wa`, `RaRs1`, `RaRs2`, `FrdGatherIndex`) declared as `Reference` in Phase 4 sumcheck phases.

**Architectural shift:** OLD `PolynomialId::FieldRegX` enum variants → MLIR-string oracle symbols (no central enum on new base).

**Goldens regen #2:** new oracle plans + commitment-batch shape, smaller blast than #1.

## Phase 4 — 3-sumcheck FR Twist chain (~2,000 LOC + goldens regen #3, the largest)

| Step | Agent | Scope | LOC |
|---|---|---|---|
| 4a | 09 | Stage 3 `FieldRegClaimReduction` — mirror of `RegistersClaimReduction`. γ-batched `FieldRdValue + γ·FieldRs1Value + γ²·FieldRs2Value`. Smallest blast radius; validates FR oracle plumbing + BatchEq(200) slot. | ~500 |
| 4b | 09 | Stage 5 `FieldRegValEvaluation` — mirror of `RegistersValEvaluation`. Degree-3 `inc × eq_gather × LT` kernel. `fr_val_eval_rounds = log_t`. BatchEq(202)/BatchEq(203) reserved. | ~650 |
| 4c | 09 | Stage 4 `FieldRegRW` — sparse phase-segmented sumcheck with `LOG_K_FR=4` (vs `LOG_K_REG=7`). `inner_only=[true,false,false,false,false,true]`, `first_active_round = log_k_reg - LOG_K_FR = 3`. Verifier-fragile ScalarCapture transition. | ~1020 |

**Critical synchronization invariant** (per CLAUDE.md): each new sumcheck instance keeps four expressions in lockstep —
- `input_claim()`: γ-batched sum the prover proves
- `output_claim()`: kernel evaluation at final round point
- `input_claim_constraint()`: verifier-side recomputation from prior-stage openings
- `output_claim_constraint()`: verifier-side recompute from this stage's evals

The muldiv e2e test catches synchronization gaps. Failure mode: `<stage>N relation input/output claim mismatch`.

**Files touched:** `jolt-kernels/src/stage{3,4,5}.rs` (~1,050 LOC), `jolt-witness/src/lib.rs` (derived materializers ~45 LOC, plus Stage6WitnessInputs extensions), `bolt/src/protocols/jolt/emit/rust/stage{3,4,5}.rs` (~900 LOC).

**Recommended sub-order:** 4a → 4b → 4c (Stage 4 is hardest, save for last; Stage 3 validates the FR oracle wiring with smallest blast radius).

**Goldens regen #3 (largest):** ~1,116 LOC churn across `crates/jolt-prover/src/stages/stage{3,4,5}.rs`. Full descriptor table regen.

## Phase 5 — Examples + audit fixes (~700 LOC)

| Step | Agent | Scope |
|---|---|---|
| 5a | 11 | `examples/bn254-fr-poseidon2-external` — portable today (stock arkworks Fr on RV; no FR coprocessor needed). 2.88× baseline reference. Could land earlier; placed here next to SDK comparison. |
| 5b | 10B | FieldRegConfig replay materializers — substantial re-architecture against new `Stage6WitnessParams` shape. The old commit's `derived.rs`/`field_reg.rs` files have no direct target on new base. Add `FieldRegRaRs1`/`RaRs2`/`Wa`/`Val` materializers (K_FR×T zero-shape default) + `FrdGatherIndex` (T-length `u64::MAX` sentinels) + `limbs_to_field<F>([u64;4])` next to `stage6_witness_polynomials`. |
| 5c | 11 | `examples/bn254-fr-poseidon2-sdk` — guest lib.rs is ~431 LOC verbatim port; host main.rs follows muldiv's `ProofBundle` pattern. Macro's `make_modular_compile_func` already emits `build_with_features(target_dir, &["compute_advice"])` — works as-is. |
| 5d | 12 | Audit fixes C1–C11 + late patches: verifier wiring (C1-C6) onto new `jolt-verifier/src/stages/`; bytecode anchoring (C7) onto extended `FrCycleBytecode`; replay validation asserts (C9,C10) onto Phase 3 replay; `field_reg_inc_polynomial` helper (C11); SDK `Fr::inverse() → Option<Fr>` + tracer FINV(0) panic (C8); drop `FieldRegRa(d)` commitment (`4b3769bd7`); mask `frs1/frs2/frd & 0xF` at producer (`5f8b71f90`). Drop `specs/fr-v2-audit.md` under `specs/`. |

**C4 (Stage 5 verifier stub):** likely obsolete — new base has a real `stage5.rs` file. Verify and skip if so.
**C12 (`num_constraints_padded` 64):** check whether the upstream refactor already aligned to `next_power_of_two()=64` in `jolt-r1cs/src/key.rs` / `jolt-kernels/src/stage1.rs`. Likely already correct.

**Validation gate after Phase 5:** Poseidon2 SDK example proves with `valid: true`; performance ≥ 2× vs external arkworks baseline.

## Aggregate scope

| Phase | Hand-written LOC | Regen output | Risk |
|---|---|---|---|
| 1 | ~600 | — | Low (additive, mostly cherry-pickable) |
| 2 | ~400 | ~30 assertions + matrix hashes | Medium (8-file cascade, transcript-divergence risk) |
| 3 | ~600 | new oracle plans | Medium (oracle-ID model shift) |
| 4 | ~2,000 | ~1,116 LOC stages 3/4/5 | **High** (sumcheck synchronization + ScalarCapture) |
| 5 | ~700 | new example crates | Low–Medium |
| **Total** | **~4,300 LOC** | **~1,500 LOC** | |

## Realistic time estimate

12–16 hours of focused work. Roughly 5× the SDK port (which was ~810 LOC).

## Commit plan

Land as **multiple commits on a single PR** stacked on top of #1535:

1. **Commit 1:** Phase 1a — CircuitFlags 14→23 + `[bool; 14]` cascade.
2. **Commit 2:** Phase 1b — FieldReg CPU state in tracer.
3. **Commit 3:** Phase 1c — FieldOp instruction + jolt-riscv kind registration.
4. **Commit 4:** Phase 1d — FieldRegEvent stream threading.
5. **Commit 5:** Phase 1e — 5 more field instructions.
6. **Commit 6:** Phase 1f — jolt-riscv per-instruction structs.
7. **Commit 7:** Phase 1g — bn254-fr SDK drop-in.
8. **Commit 8 + regen:** Phase 2 — 13 FR R1CS rows + slots.
9. **Commit 9 + regen:** Phase 3 — FR Twist scaffolding + replay.
10. **Commit 10:** Phase 4a — Stage 3 FieldRegClaimReduction sumcheck.
11. **Commit 11:** Phase 4b — Stage 5 FieldRegValEvaluation sumcheck.
12. **Commit 12 + regen:** Phase 4c — Stage 4 FieldRegRW sumcheck.
13. **Commit 13:** Phase 5a — bn254-fr-poseidon2-external example.
14. **Commit 14:** Phase 5b — FieldRegConfig materializers.
15. **Commit 15:** Phase 5c — bn254-fr-poseidon2-sdk example.
16. **Commit 16:** Phase 5d — Audit fixes C1–C11.

Each commit should leave the tree compiling and the test suite at its appropriate gate.

## Key risks

1. **Transcript-divergence in Phase 2.** The `[bool; 14]` → `[bool; 23]` cascade hits 8 files; any consumer reading/writing the wrong-shape array silently corrupts Fiat-Shamir. The `cargo nextest run -p jolt-equivalence transcript_divergence` test must pass after Phase 2.
2. **Sumcheck claim/constraint synchronization in Phase 4.** Each of the 3 new sumchecks has 4 expressions that must stay in lockstep. The muldiv e2e test catches gaps but only at runtime.
3. **Goldens churn fatigue.** Three regen sweeps; each can surface assertions that need numeric updates in `bolt/tests/commitment_ir.rs`. Allocate buffer time.
4. **Stage 4 ScalarCapture in Phase 4c.** The verifier-fragile transition between FR address-phase rounds (4 rounds at `LOG_K_FR=4`) and cycle-phase rounds (`log_t`). Test extensively against the modular_self_verify gate.
5. **Equivalence stack churn.** Markos's `jolt-v2/equivalence` may rebase mid-port. Re-anchor `sagar/modular-sdk` before starting Phase 2 and again before Phase 4 to minimize blast radius.

## Validation gates

| After | Gate |
|---|---|
| Phase 1 | `cargo nextest run -p bolt --test commitment_ir` 53/53; muldiv `valid: true` |
| Phase 2 | Same as Phase 1, at new R1CS shape; goldens regen clean |
| Phase 3 | Same as Phase 2; new FR oracles render in `commitment_ir`; `replay_field_regs` unit tests pass |
| Phase 4 | Same + new FR e2e smoke test (`#[ignore]` until 5c) compiles; muldiv unaffected |
| Phase 5 | Poseidon2 SDK e2e proves `valid: true`; performance ≥ 2× vs external baseline |
