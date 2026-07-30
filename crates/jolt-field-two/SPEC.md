# Spec: jolt-field-two — minimal-LOC rebuild of jolt-field

| Field   | Value                                              |
|---------|----------------------------------------------------|
| Status  | approved — building (checkpoint 1)                 |
| Baseline| `jolt-field` @ PR #1684 head (`fe1d5d41f`)         |
| Goal    | functional parity at ≤ 6,300 counted LOC (baseline: 11,410) |

## Goal

Rebuild `crates/jolt-field` from first principles minimizing source LOC while
preserving functionality: both backends (BN254 arkworks + full Solinas stack),
wire/transcript **byte** compatibility, and static dispatch. Trait names and
boundaries are redesigned from scratch — old-name compatibility is explicitly
NOT a goal (approved); consumers rebind at replacement time. The crate lives
at `crates/jolt-field-two` until ready to replace `jolt-field`.

## Counting rules

**Counted:** non-blank, non-comment lines under `src/`, excluding `#[cfg(test)]`
regions. Doc comments and regular comments are **free** — the budget must never
create pressure to strip documentation. Inline test modules must be the final
item of their file (the counter cuts at the first `#[cfg(test)]`).

Measurement (run from the crate root):

```bash
for f in $(find src -name '*.rs'); do
  awk '/^[[:space:]]*#\[cfg\(test\)\]/{exit} !/^[[:space:]]*(\/\/|$)/{c++} END{print c+0}' "$f"
done | paste -sd+ - | bc
```

Tests, benches, and fuzz targets are uncounted and unlimited.

## Trait system (first principles)

**15 public traits** (baseline: 22; pre-consolidation: 46). Two backends, one
spine; every merge below is justified by "same implementor set or a strict
capability subset with defaulted members".

### Spine (unconditional, 7)

| Trait | Replaces | Contents |
|---|---|---|
| `AdditiveGroup` | same | `Zero` + add/sub/neg (owned + by-ref) + `Copy/Send/Sync` |
| `Ring` | `RingCore` + `FromPrimitiveInt` | ring ops, `square`/`pow2`, **integer embedding** (`from_u64/i64/u128/i128` required; small widths + `mul_*`/`mul_pow_2` defaulted). Rationale: every unital ring embeds ℤ; keeping embedding separate bought nothing and doubled bounds everywhere |
| `Field` | `FieldCore` + `HalvingField` | `inverse`, `inv_or_zero`, `random`, and defaulted `half`/`two_inv` (fast impls override). Rationale: char ≠ 2 always, so halving is field-generic with a default |
| `CanonicalEncoding` | `CanonicalRepr` + `CanonicalField` | one canonicity surface: `NUM_BYTES`, `MODULUS_BITS` (bit length of \|F\|), `to_bytes_le`, `from_bytes_le_reduced`, `from_bytes_le_checked`, `to_u128_checked`, `from_u128_checked/_reduced`, `num_bits`, defaulted challenge derivation. Transcript bytes are specified here; wire serde reuses `from_bytes_le_checked` so canonical-rejection is uniform |
| `Accumulator` | same | `add`/`merge`/`reduce`/`fmadd` + defaulted small-scalar fmadds |
| `WithAccumulator` | same (bound: `Ring`) | associated `Accumulator` |
| `JoltField` | `Field` umbrella | **blanket-implemented** marker: `Field + CanonicalEncoding + WithAccumulator + Serialize + DeserializeOwned`. Blanket impl means it can never be forgotten; serde in the umbrella because every proof-system field needs a wire format |

### Solinas (feature-gated, 8)

| Trait | Replaces | Contents |
|---|---|---|
| `PseudoMersenne` (defined unconditionally in `algebra.rs`, per the file table) | `PseudoMersenneField` + `ExtMulBackend` | `const OFFSET: u128` (bits live on `CanonicalEncoding`) + the degree-4/8 ext-mul/square kernel hooks (`ext4_mul`, `ext4_square`, `ext8_mul`, `ext8_square`) with generic coefficient-formula defaults (`schedules.rs`). No base field overrides them: the baseline's fused-accumulation `Fp32` override lost the checkpoint-6 bench gate (see dropped-specialization evidence) |
| `ExtField<F>` | same | degree, `lift_base`, `mul_base`, coeff access, Frobenius |
| `Ext2Config<F>` | `FpExt2Config` | quadratic non-residue config (ZST pattern), `IS_NEG_ONE` fast path |
| `MulBaseUnreduced<F>` | same | tiny overridable ext×base deferred multiply — **deferred to checkpoint 7**: its contract is stated in terms of `Unreduced::Product`, which does not exist until the unreduced checkpoint |
| `Unreduced` | `HasUnreducedOps` + `HasWide` + `ReduceTo` | **one deferred-reduction companion surface**: `type Product`, `type SmallProduct`, `type Wide` (i32-lane), `SUM_IS_EXACT`, widening muls + `reduce_*` for each, `scale_wide`. Rationale: these were three fragments of one concept — "the unreduced value algebra around a field"; routing reduction through the field type kills `ReduceTo`'s ambiguity workarounds |
| `Fold` | `HasOptimizedFold` | `precompute(r) -> Ctx`, `fold_one(ctx, even, odd)` — documented honestly as the multilinear bind `even + r·(odd − even)`, a protocol-support hook that lives here because implementations exploit field representation |
| `Packed` | `PackedField` | lanes: `Scalar`, `WIDTH`, `from_fn`/`extract`/`broadcast` + defaulted slice helpers + packed ext2 kernel hook |
| `WithPacking` | `HasPacking` | associated `Packing` |

**Deleted outright:** `MontgomeryConstants` (approved), the `akita` bootstrap
(approved), `CanonicalField`, `HalvingField`, `ExtMulBackend`, `HasWide`,
`HasUnreducedOps`, `ReduceTo`, `FromPrimitiveInt` (all merged as above).

**Exported stamping macros** (`ops.rs`): `impl_ring_ops!` (full operator
matrix + `Zero`/`One`/`Sum`/`Product` from raw add/sub/mul/neg),
`impl_group_ops!` (the additive-only subset, for accumulator types),
`impl_serde_bytes!` (canonical-checked serde over `CanonicalEncoding`, byte-format
identical to baseline). Exported so third-party field implementors pay the
same near-zero boilerplate we do — the `mersenne61`-style compat test consumes
them as a third party would.

## Scope

**Parity (functionality, not names):** everything jolt-field @ baseline does —
BN254 `Fr`/`Fq`/`WideAccumulator`; Solinas `Fp32`/`Fp64`/`Fp128` + the 9
registered prime offsets (count tracks the baseline registry); `FpExt2/4/8` + Frobenius/Moore machinery; packed
NEON/AVX2/AVX-512 × {32,64,128} + packed ext + `NoPacking`; lane accumulators
and fold matrices; `S64`–`S256` + hi32 variants; `Limbs<N>`; rayon helpers;
`allocative` derives. Features: `default = ["bn254"]`, `solinas`, `parallel`,
`allocative`.

**Byte compatibility (hard invariants):**
- BN254 serde bytes and Fiat-Shamir transcript bytes identical to jolt-field.
- Solinas serde+bincode wire bytes and `CanonicalEncoding` transcript bytes identical
  to jolt-field (replacement must not change proof bytes).

**Dropped (approved):** `akita` bootstrap feature/module, `MontgomeryConstants`.

**Dropped-specialization evidence (checkpoint 5, fp128):**

- **Kept:** the baseline's AArch64 inline-asm `mul`/`sqr` kernels — the only
  per-arch specializations with recorded evidence (1.29x throughput on Apple
  M4, per the baseline's own doc comment) and the prover hot path. An
  AArch64-only unit test in `fp128.rs` cross-checks them against the portable
  fold on random + boundary inputs for all four registered offsets.
- **Dropped:** AArch64 and x86-64 inline-asm add/sub kernels
  (`add_raw_{aarch64,x86_64}_{imm,reg}` + dispatchers, ~470 source lines).
  Same carry-chain algorithm as the portable path with hand-scheduled flag
  flow (`ccmp`/`sbb`-mask selects); no benchmark recorded in-tree, only
  qualitative comments. The portable path is branchless and compiles to a
  near-identical adds/adcs/csel (resp. add/adc/cmov) sequence. The x86-64
  imm-vs-reg dispatch subtlety (sign-extended imm32 unusable for C ≥ 2^31,
  i.e. `Prime128OffsetA7F7`) dies with it. Baseline x86-64 `mul` was already
  portable — nothing dropped there.
- **Dropped:** the AArch64 `mul_add` asm kernel together with the
  `mul_add`/`add_128_into_256` fused multiply-add surface: no consumer
  anywhere in the parity scope (only baseline fp128's own tests call it).
- **Dropped:** `mul_wide_limbs<M, OUT>` (generic loop + M/OUT-unrolled
  hot-path specializations, ~270 lines): its only workspace consumer is
  `jolt-prover-legacy`'s akita field glue, and the akita bootstrap is
  dropped (approved above). `mul_wide`/`mul_wide_u64`/`mul_wide_u128` and
  the ≤10-limb `solinas_reduce` — the surfaces the in-scope unreduced
  accumulators use — are ported.
- **Dropped:** `from_i64_const` (const-evaluable embedding): akita-only
  (its `MONTGOMERY_R` constants).

**Dropped-specialization evidence (checkpoint 6, extensions):**

- **Dropped:** the baseline's fused `Fp32` deg-4 ext-mul/square override
  (u128 column accumulation with `P²` biases, one reduction per output
  coefficient). Bench gate (`benches/ext4_kernels.rs`, aarch64 / Apple M4,
  release codegen, 4096 batched ops × 100 reps, best of 7, over
  `Prime32Offset99`): generic-schedule mul **12.3 ns/op** vs fused port
  **31.1 ns/op** (fused 2.5x slower; the baseline crate's own fused
  override measured 31.0 ns — the port reproduces it exactly); square
  **15.3 ns** vs **28.3 ns** (1.85x slower; baseline 28.6 ns).
  Keep-threshold was a >10% fused win, so all
  four `PseudoMersenne` kernel hooks retain their generic defaults and no
  base field overrides them. Caveat: measured on aarch64 only (immediate
  word-sized reductions pipeline better than u128 accumulation chains
  there); the fused port stays in the bench harness — rerun it on x86-64
  before reintroducing an override.
- **Deferred:** the `MulBaseUnreduced` contract to checkpoint 7 — its
  baseline definition (`mul_base_to_product_accum`) returns
  `Unreduced::Product`, which does not exist until the unreduced surface
  lands; inventing a placeholder shape now would just be churn.
- **Added (not in baseline):** an `ext8_square` hook on `PseudoMersenne`
  defaulting to the deg-8 squaring schedule. The baseline computed
  `FpExt8::square` as a full multiply and used its square schedule only in
  the packed kernels; routing scalar squaring through the same schedule is
  value-identical (pure ring ops), saves base ops, and gives the schedule
  its scalar consumer.

## Design pillars

1. **Const-generic scalar core**: `Fp64<const P: u64>` etc., fold constants
   derived at monomorphization, `C(C+1) < P` const-asserted — in exactly one
   place per layer.
2. **`macro_rules!` only — no proc macros.** We control every type; nothing
   needs to parse Rust. No `jolt-field-derive` (decision recorded; revisit
   only if the operator matrices become unmanageable).
3. **One fold-algebra source of truth per axis of variation:** widths — one
   `define_solinas_prime!` body stamps `Fp32`/`Fp64` (fp128 is genuinely
   two-limb, hand-written); ISAs — one packed engine macro stamped per
   (width × ISA) over per-ISA primitive vocabularies (`simd.rs`); kernels that
   differ *algorithmically* (AVX2's missing 64-bit widening mul) live in the
   primitive layer.
4. **Differential testing against jolt-field as the oracle** (dev-dependency):
   random-op equivalence per type, byte-equality for serde and transcript
   encodings, packed-vs-scalar, num-bigint as independent third oracle.
5. **Workspace member from day one** (approved): inherits lints, dep versions,
   and the arkworks `[patch.crates-io]` fork. Every checkpoint leaves the
   workspace green.

## Backend architecture

The crate is two layers with a one-way dependency:

1. **Contract layer** (crate root, unconditional): every trait the crate
   defines, the stamping macros, and the backend-neutral value types
   (`Limbs`, `signed`). Contract files contain **trait definitions only — no
   backend impls, no arithmetic**. The full capability surface of the crate
   is readable from the root regardless of which features are enabled.
2. **Backend layer** (feature-gated modules): implementations of the
   contracts. `bn254/` adapts an external implementation (arkworks) plus
   first-party Barrett/Montgomery kernels; `solinas/` is a fully first-party
   implementation. A backend module may be deleted (or a new one added)
   without touching the contract layer.

Rules: contract files never reference backend modules; backends never
reference each other; a backend's public surface is exactly its `impl
<Contract> for <Type>` items plus its concrete types — enumerable by
grepping the module. Adding a backend = implement the spine (JoltField,
serde, and the conformance-law tests come free via the blanket impl and
exported macros — the Mersenne-61 spine test demonstrates the full recipe);
optionally implement the capability contracts (`PseudoMersenne`, extensions,
`Unreduced`, `Fold`, `Packed`) to light up the generic machinery that bounds
on them.

## File structure and budgets

| File | Budget | Contents |
|---|---|---|
| **Contract layer (root, unconditional)** | | |
| `src/lib.rs` | 70 | crate docs, feature gates, re-exports, `FieldError` |
| `src/algebra.rs` | 260 | spine: 7 traits + `NaiveAccumulator` + `PseudoMersenne` |
| `src/extension.rs` | 60 | contracts: `ExtField`, `Ext2Config` + NR config ZSTs (`MulBaseUnreduced` lands with checkpoint 7) |
| `src/unreduced.rs` | 70 | contracts: `Unreduced`, `Fold` |
| `src/packed.rs` | 90 | contracts: `Packed`, `WithPacking` + generic `NoPacking` |
| `src/ops.rs` | 180 | `impl_ring_ops!`, `impl_serde_bytes!` (backend-neutral) |
| `src/schedules.rs` | 140 | lane-generic deg-4/8 ext coefficient schedules — unconditional because the `PseudoMersenne` hook defaults (algebra.rs) and the packed lanes (checkpoint 8) share them; carved out of the old `ext.rs` budget (890 → 750, component total unchanged) |
| `src/limbs.rs` | 220 | `Limbs<N>` |
| `src/signed.rs` | 420 | signed bigint families (consumer-audited surface) |
| **bn254 backend** | | |
| `src/bn254/mod.rs` | 400 | `Fr`, `Fq` via one wrapping macro; serde; transcript bytes |
| `src/bn254/mont.rs` | 300 | Barrett/Montgomery kernel + `WideAccumulator` |
| **solinas backend** | | |
| `src/solinas/mod.rs` | 90 | offset registry, aliases, shared helpers |
| `src/solinas/word.rs` | 380 | `define_solinas_prime!` → `Fp32`, `Fp64` |
| `src/solinas/fp128.rs` | 700 | two-limb add/sub/mul/reduce/wide |
| `src/solinas/ext.rs` | 750 | FpExt2/4/8 impls, ExtField impls, Frobenius + Moore |
| `src/solinas/unreduced.rs` | 530 | lane accumulators, fold matrices, contract impls |
| `src/solinas/parallel.rs` | 80 | rayon helpers |
| `src/solinas/packed/mod.rs` | 30 | backend selection |
| `src/solinas/packed/simd.rs` | 350 | per-ISA primitive vocabulary (neon/avx2/avx512) |
| `src/solinas/packed/engine.rs` | 550 | shared packed algebra, stamped per width × ISA |
| `src/solinas/packed/fp128.rs` | 350 | 128-bit-lane engine |
| `src/solinas/packed/ext.rs` | 230 | packed FpExt2/4/8 |
| **Total** | **6,240** | vs 11,410 baseline (−45%) |

Component budgets are unchanged — the contract/impl split carves each
component's contracts out of its old single-file budget (extensions 950 =
60 + 140 + 750, unreduced 600 = 70 + 530, packed selection 120 = 90 + 30).

Baseline per component (same metric): packed 3,720 → 1,600 · prime 2,140 →
1,170 · ext 1,661 → 950 · arkworks 1,170 → 700 · unreduced 1,073 → 600 ·
signed 1,004 → 420 · limbs 246 → 220 · spine+glue ~400 → 430 · parallel 78 → 80.

**Budget discipline:** if a component lands ≤10% over after honest
compression, we discuss trade rather than golf. Riskiest: `packed/` and `ext.rs`.

## Build order (one review checkpoint each)

1. **Scaffold + spine** — crate, workspace membership, `algebra.rs`, `ops.rs`,
   `lib.rs`. Accept: compiles no-default-features; a Mersenne-61 toy field in
   `tests/` implements the full spine through the exported macros (third-party
   implementability, no arkworks); law suite green.
2. **BN254** — early end-to-end validation of the spine with the field Jolt
   actually runs. Accept: differential vs jolt-field `Fr`/`Fq` (ops, serde
   bytes, transcript bytes), `WideAccumulator` exactness.
3. **Limbs + signed** — accept: differential + boundary tests.
4. **Solinas words** — `word.rs` + registry. Accept: differential vs
   jolt-field `Fp32`/`Fp64` across all registered ≤64-bit offsets; num-bigint
   oracle; serde byte-equality.
5. **Fp128** — same acceptance for 128-bit offsets.
6. **Extensions** — accept: differential + schoolbook oracle + Frobenius/Moore
   parity; bench the generic-default deg-4 kernel vs jolt-field's `Fp32`
   override before deciding to keep the override.
7. **Unreduced** — accept: accumulator-exactness vs direct mul, fold parity.
8. **Packed** — accept: packed-vs-scalar on native ISA; `cargo check` with
   `-Ctarget-feature` for the other ISAs; NEON run on this machine (aarch64).
9. **Parallel + assembly** — final LOC audit, feature-matrix build, crate docs.

## Non-goals

- Replacing `jolt-field` in consumers (separate PR once accepted).
- Porting the Criterion bench suite (thin comparison bench later; uncounted).
- CI wiring (follows at replacement; will include a target-feature lane so
  SIMD is not CI-dark — fixing a baseline gap).

## Resolved questions

1. Counting rules — **approved**.
2. Drop `akita` bootstrap + `MontgomeryConstants` — **approved** (bn254 +
   solinas are the two supported stacks).
3. Trait redesign — **approved and widened**: full first-principles redesign,
   breaking old names is fine (see Trait system).
4. Workspace membership from day one — **approved**.
5. Budgets/order — **approved**.
