# Phase 5c (SDK example) + 5d (audit) — working note

This is a working note for the next session, not a long-lived spec.
Delete after Phase 5c.

## State at HEAD

- Phase 5b is end-to-end:
  - `5f9cbe882 feat(phase-5b): FieldRegReplay materializers for FR Twist witness polys`
  - `10a9306c4 feat(phase-5b): wire FieldRegReplay into Stage45SparseTraceWitness`
  - `9eb43ae25 feat(phase-5b): plumb tracer FR events into Stage45SparseTraceWitness`

jolt-host's `prove_program` already constructs a `FieldRegReplay` from the
tracer's `FieldRegEvent` stream and per-cycle decoded `FrCycleBytecode`,
then attaches it to `Stage45SparseTraceWitness` before driving Stage 4/5.
For FR-inactive programs (muldiv et al) the materializers short-circuit to
zero buffers — same trivially-satisfied path as before. Gates green:
muldiv host, jolt-witness 30/30, commitment_ir 53/53, clippy clean.

What's blocking Phase 5c: the source-branch `examples/bn254-fr-poseidon2-sdk`
example uses a high-level `Fr` newtype (`jolt_inlines_bn254_fr::Fr`) with
`add` / `mul` / `sub` / `inv` methods backed by a 2-pass `compute_advice`
mechanism. The current modular-sdk's `jolt-inlines-bn254-fr` only ships
the low-level primitives (`fmul` / `fadd` / etc taking raw frd/frs1/frs2
indices). Porting the 2-pass machinery is ~170 LOC of substantial work
(`compute_advice` feature, `VirtualHostIO`, ADVICE_LD reads, FieldAssertEq
result-binding).

## Phase 5c steps

### 1. Port the high-level `Fr` newtype to `jolt-inlines/bn254-fr/src/sdk.rs`

Cherry-pick the `Fr` struct + `add`/`sub`/`mul`/`inv` methods from
source commit `11fd62596:jolt-inlines/bn254-fr/src/sdk.rs` (485 LOC vs
current 315). Key adds:

- `pub struct Fr { pub limbs: [u64; 4] }` with `from_limbs` / `to_limbs`
  / `zero` / `one`.
- `compute_advice` cargo feature on `jolt-inlines/bn254-fr/Cargo.toml`
  (already 2-pass-aware in the macro, gates `dep:ark-bn254` + `dep:ark-ff`).
- 3 backend paths per op (host / compute_advice / RISC-V), gated on
  `cfg(feature = "...")` + `cfg(target_arch = "...")`. The Pass-1 path
  computes via `ark_bn254::Fr` and writes 4 result limbs via
  `VirtualHostIO`; the Pass-2 path emits the 7-cycle Horner load
  sequence (`FieldMov` + `FieldSLL64/128/192` + `FieldAdd`), one
  `FieldOp` cycle, then 4 × `ADVICE_LD` + 7-cycle reconstruction +
  `FieldAssertEq` to bind the FieldOp output to the advice limbs.

### 2. Add `examples/bn254-fr-poseidon2-sdk/` (1387 LOC guest + ~40 LOC host)

```bash
# Get guest lib (1387 LOC) — verbatim port from source
git show 11fd62596:examples/bn254-fr-poseidon2-sdk/guest/src/lib.rs \
    > examples/bn254-fr-poseidon2-sdk/guest/src/lib.rs
```

Then edit `#[jolt::provable(...)]` to add `backend = "modular"` and
clamp `max_trace_length` to `262_144` (= 2^18, current goldens ceiling).

Host main mirrors `examples/muldiv/src/main.rs`'s `compile_*` /
`prove_*` / `verify_*` pattern (already drafted in this commit's history
under `examples/bn254-fr-poseidon2-arkworks/src/main.rs`).

Cargo.toml workspace member additions:
```toml
"examples/bn254-fr-poseidon2-sdk",
"examples/bn254-fr-poseidon2-sdk/guest",
```

### 3. Validate end-to-end

```bash
cargo install --path . --locked
cargo run --release -p bn254-fr-poseidon2-sdk
```

Expected: `valid: true`, prove time meaningfully lower than the arkworks
baseline (Phase 5a). Trace cycle count should be ~35K vs arkworks ~253K
(the source's measured 7× advantage).

If the verifier rejects the proof, the most likely culprit is a
materializer correctness issue — `frs1_ra` / `frs2_ra` are gated on the
conservative `(reads_frs1, reads_frs2) = (true, true)` for all FieldOp
cycles in `crates/jolt-host/src/lib.rs::fr_bytecode_from_trace`. FINV
cycles overshoot (rs2 unused but flag says read). Poseidon2 doesn't use
FINV so this should be safe in practice. If it bites: distinguish
FieldOp variants via `CircuitFlags::IsField{Mul,Add,Sub,Inv,AssertEq}`
which are already populated on each cycle's flag set (see
`crates/jolt-kernels/src/trace.rs:561-565`).

## Phase 5d — Audit fixes C1–C11

After 5c lands, the FR coprocessor is end-to-end working. The remaining
audit items from `specs/fr-v2-port-plan.md` lines 117-122 (carried over
from source branch's `specs/fr-v2-audit.md`):

- **C1–C6**: verifier wiring onto `jolt-verifier/src/stages/`
- **C7**: bytecode anchoring onto extended `FrCycleBytecode`
- **C8**: SDK `Fr::inverse() → Option<Fr>` + tracer FINV(0) panic
- **C9, C10**: replay validation asserts onto Phase 3 replay
- **C11**: `field_reg_inc_polynomial` helper
- Drop `FieldRegRa(d)` commitment (per source commit `4b3769bd7`)
- Mask `frs1/frs2/frd & 0xF` at producer (per source commit `5f8b71f90`)

C4 (Stage 5 verifier stub) and C12 (`num_constraints_padded`=64) are
likely already obsolete on modular-sdk's newer base — verify and skip.

After 5d lands, delete `specs/fr-v2-port-plan.md` and the audit task
graph — the port is complete.
