# Spec: jolt-field-two — minimal-LOC rebuild of jolt-field

| Field   | Value                                              |
|---------|----------------------------------------------------|
| Status  | replaced-in — all nine checkpoints complete (5,103 counted LOC, budget 6,240); the crate now lives at `crates/jolt-field` and the baseline implementation is deleted |
| Baseline| `jolt-field` @ PR #1684 head (`fe1d5d41f`)         |
| Goal    | functional parity at ≤ 6,300 counted LOC (baseline: 11,410) |

## Goal

Rebuild `crates/jolt-field` from first principles minimizing source LOC while
preserving functionality: both backends (BN254 arkworks + full Solinas stack),
wire/transcript **byte** compatibility, and static dispatch. Trait names and
boundaries are redesigned from scratch — old-name compatibility is explicitly
NOT a goal (approved); consumers rebind at replacement time. The crate lived
at `crates/jolt-field-two` during the rebuild; at replacement it took over
`crates/jolt-field` and the package name `jolt-field`.

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

**16 public traits** (baseline: 22; pre-consolidation: 46). Two backends, one
spine; every merge below is justified by "same implementor set or a strict
capability subset with defaulted members".

### Spine (unconditional, 7)

| Trait | Replaces | Contents |
|---|---|---|
| `AdditiveGroup` | same | `Zero` + add/sub/neg (owned + by-ref) + `Copy/Send/Sync` |
| `Ring` | `RingCore` + `FromPrimitiveInt` | ring ops, `square`/`pow2`, **integer embedding** (`from_u64/i64/u128/i128` required; small widths + `mul_*`/`mul_pow_2` defaulted). Rationale: every unital ring embeds ℤ; keeping embedding separate bought nothing and doubled bounds everywhere |
| `Field` | `FieldCore` + `HalvingField` | `inverse`, `inv_or_zero`, `random`, defaulted `mul_add`, and defaulted `half`/`two_inv` (fast impls override). Rationale: char ≠ 2 always, so halving is field-generic with a default |
| `CanonicalEncoding` | `CanonicalRepr` + `CanonicalField` | one canonicity surface: `NUM_BYTES`, `MODULUS_BITS` (bit length of \|F\|), `to_bytes_le`, `from_bytes_le_reduced`, `from_bytes_le_checked`, `to_u128_checked`, `from_u128_checked/_reduced`, optional zero-copy canonical `u32` slices, `num_bits`, defaulted challenge derivation. Transcript bytes are specified here; wire serde reuses `from_bytes_le_checked` so canonical-rejection is uniform |
| `Accumulator` | same | `add`/`merge`/`reduce`/`fmadd` + defaulted small-scalar fmadds |
| `WithAccumulator` | same (bound: `Ring`) | associated `Accumulator` |
| `JoltField` | `Field` umbrella | **blanket-implemented** marker: `Field + CanonicalEncoding + WithAccumulator + Serialize + DeserializeOwned`. Blanket impl means it can never be forgotten; serde in the umbrella because every proof-system field needs a wire format |

### Solinas (feature-gated, 9)

| Trait | Replaces | Contents |
|---|---|---|
| `PseudoMersenne` (defined unconditionally in `algebra.rs`, per the file table) | `PseudoMersenneField` + `ExtMulBackend` | `const OFFSET: u128` (bits live on `CanonicalEncoding`) + the degree-4/8 ext-mul/square kernel hooks (`ext4_mul`, `ext4_square`, `ext8_mul`, `ext8_square`) with generic coefficient-formula defaults (`schedules.rs`). No base field overrides them: the baseline's fused-accumulation `Fp32` override lost the checkpoint-6 bench gate (see dropped-specialization evidence) |
| `ExtField<F>` | same | degree, `lift_base`, `mul_base`, coeff access, Frobenius |
| `Ext2Config<F>` | `FpExt2Config` | quadratic non-residue config (ZST pattern), `IS_NEG_ONE` fast path |
| `MulBaseUnreduced<F>` | same | tiny overridable ext×base deferred multiply, stated in terms of `Unreduced::Product` (deferred from checkpoint 6, landed with checkpoint 7; lives in `extension.rs` with a degree-1 blanket impl) |
| `Unreduced` | `HasUnreducedOps` + `HasWide` + `ReduceTo` | **one deferred-reduction companion surface**: `type Product`, `type SmallProduct`, `type Wide` (i32-lane), `SUM_IS_EXACT`, widening muls + `reduce_*` for each, `scale_wide`. Rationale: these were three fragments of one concept — "the unreduced value algebra around a field"; routing reduction through the field type kills `ReduceTo`'s ambiguity workarounds |
| `WithCommitAccumulator` | `HasCommitAccum` | marks base fields whose existing `Unreduced::Wide` accumulator supports unit-scale commitment streams and states the exact safe addition limit without duplicating the accumulator type |
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
- **Initially dropped, later restored:** AArch64 and x86-64 inline-asm add/sub kernels
  (`add_raw_{aarch64,x86_64}_{imm,reg}` + dispatchers, ~470 source lines).
  Same carry-chain algorithm as the portable path with hand-scheduled flag
  flow (`ccmp`/`sbb`-mask selects); no benchmark recorded in-tree, only
  qualitative comments. The portable path is branchless and compiles to a
  near-identical adds/adcs/csel (resp. add/adc/cmov) sequence. The x86-64
  imm-vs-reg dispatch subtlety matters because a sign-extended imm32 is
  unusable for C ≥ 2^31, including `Prime128OffsetA7F7`. The proof continuity
  follow-up restores both architecture families. It also binds the A7F7
  AArch64 words and x86-64 bytes to HOL Light theorems and an optimized public witness. See
  `specs/jolt-field-proof-continuity.md`.
- **Initially dropped:** the AArch64 `mul_add` asm kernel together with the
  `mul_add`/`add_128_into_256` fused multiply-add surface had no consumer in
  the original parity scope. Current Akita now uses this hook, so the refresh
  restores it under the canonical `Field` trait. See replacement-time
  deviation 5.
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
  **Un-deferred with checkpoint 7:** landed in `extension.rs` as
  `mul_base_unreduced` with a lift-then-`mul_unreduced` default body, a
  degree-1 blanket impl, the coordinate-scaling `FpExt4<Fp32>` override,
  and default-body impls for `FpExt2<Fp64>` and the identity-shape
  extension variants.
- **Added (not in baseline):** an `ext8_square` hook on `PseudoMersenne`
  defaulting to the deg-8 squaring schedule. The baseline computed
  `FpExt8::square` as a full multiply and used its square schedule only in
  the packed kernels; routing scalar squaring through the same schedule is
  value-identical (pure ring ops), saves base ops, and gives the schedule
  its scalar consumer.

**Dropped-specialization evidence (checkpoint 7, unreduced):**

- **Dropped:** the baseline's aarch64 NEON intrinsic `Add`/`Sub`/`Neg`
  paths on the `i32`-lane wide accumulators (`Fp64x4i32`, `Fp128x8i32`;
  ~120 source lines of `unsafe` intrinsics). Evidence: `rustc -O` compiles
  the portable element-wise `[i32; N]` code to the identical instructions
  the intrinsics hand-write — `ldr/ldp q` + `add.4s`/`sub.4s`/`neg.4s`
  (and `mul.4s` for lane scaling, which the baseline never vectorized) —
  verified by inspecting `--emit asm` output for 4- and 8-lane add, sub,
  neg, and scale on this machine (aarch64, Apple M4). The portable path
  additionally panics on lane overflow in debug builds, turning headroom
  violations into test failures instead of silent wrapping.
- **Corrected (baseline doc bugs, no code change):** the baseline's lane
  headroom comment says `i32::MAX / u16::MAX ≈ 32,769` additions; the safe
  count is 32768 (`32769 · 0xFFFF > 2^31 − 1`). Its `FpExt4<Fp32>` accum
  comment claims per-term slot contributions of `7·P² ≈ 2^65` and `2^63`
  accumulations; the correct figures are `7·P² < 2^67` and `2^61` terms.
  Both re-derived and documented in `solinas/unreduced.rs`, with the
  32768-boundary case tested exactly (one past asserted to panic in debug).
- **Restricted (baseline latent footgun):** the fused `FpExt2<Fp64>`
  product accumulation is only correct for non-residues −1 and 2, but the
  baseline compiled its two-case body for arbitrary `FpExt2Config`s; the
  port debug-asserts `NR ∈ {−1, 2}`.

**Dropped-specialization evidence (checkpoint 8, packed):**

- **Mechanism note:** the "engine macro stamped per width × ISA" pillar is
  realized as generic types (`PackedFp32/64/128<P, I: SimdWord>`) over a
  per-ISA vocabulary trait in `simd.rs`, with `macro_rules!` used only for
  operator stamping (`impl_packed_arith!`) and vocabulary forwarding
  (`fwd!`). Same one-source-of-truth outcome, stronger type checking, and
  it moved weight from `engine.rs` (284/550) into `simd.rs` (352/350).
- **Added (not in baseline's contract shape):** `ext4_mul`/`ext4_square`/
  `ext8_mul`/`ext8_square` kernel hooks on `Packed` with schedule defaults
  (`schedules.rs`), so the fp32 engines can override the deg-4 kernels with
  fused deferred-reduction dot products and every backend shares one
  formula source. This puts `packed.rs` at 105 vs its 90 budget; the hooks
  cannot live elsewhere (overridable defaults need the trait).
- **Kept:** the NEON 31-bit pseudo-Mersenne multiply kernel (`mul_pm31`,
  all lanes stay 32-bit via `vqdmulhq_s32`), generalized to cover `C = 1`:
  it serves the registered `Prime31Offset19` on the benched native ISA.
- **Dropped:** the dedicated Mersenne31 (`C == 1`) multiply kernels on all
  three ISAs — no registered prime has `C = 1`; NEON's kept `mul_pm31`
  subsumes the case, x86 falls back to the value-identical generic fold.
- **Dropped:** the `BITS == 31` immediate-shift fold variants
  (`solinas_reduce_bits31` and friends, per-ISA) — value-identical
  micro-opts duplicating the whole fold; 31-bit packed multiplies now go
  through `mul_pm31` on NEON anyway, so only the ext dot products take the
  variable-shift 64-bit fold.
- **Dropped:** the NEON per-C shift-add chains for `C ∈ {19, 35, 99}`;
  replaced by an ISA-generic `C = 2^a ± 1` shift-add fast path in the
  shared engine (`mul_by_offset`, covers `C = 3`) — no in-tree benchmark
  existed for the chains, and the generic `mul_small` handles the rest.
- **Dropped:** the NEON `BITS == 32` dot-product carry-tracking machinery
  (`add_u64_with_carry`/`carry_correction`/`SHIFT64_MOD_P`); all ISAs now
  use the x86 per-product prefold strategy (value-identical, comparable op
  count, one shared bound argument).
- **Dropped:** the vectorized packed ext2/ext4 inverse formulas; all packed
  inversion is lane-wise scalar (the `Packed::inverse` default). Every
  formulation performs one lane-serial base-field Fermat inversion per
  lane, which dominates; only ~20 non-inversion multiplies per lane change
  from packed to scalar, on a cold path.
- **Changed:** packed `Fp128` multiplication calls the scalar kernel per
  lane on every ISA — on AArch64 that is the inline-asm multiply, strictly
  better than the baseline NEON backend's duplicated portable fold
  (avx2/avx512 baselines already went lane-by-lane).
- **No unreduced coupling:** the packed layer consumes only `schedules.rs`
  and the scalar field types; nothing awaits the checkpoint-7 `Unreduced`
  surface.

**Dropped-specialization evidence (checkpoint 9, parallel):**

- **Audit finding:** none of the baseline's seven `cfg_*!` macros
  (`cfg_iter`, `cfg_iter_mut`, `cfg_into_iter`, `cfg_chunks`,
  `cfg_chunks_mut`, `cfg_join`, `cfg_fold_reduce`) has a single consumer —
  not in this workspace branch, not in the main checkout, and not in the
  rebuild's own code. No workspace crate even enables `jolt-field`'s
  `parallel` feature; its only in-tree activation is the baseline's own
  `solinas_field_arith` bench, which uses rayon directly rather than
  through the macros (workspace crates that parallelize — `jolt-poly`,
  `jolt-kernels` — carry their own rayon deps).
- **Kept whole anyway:** the parity scope explicitly names the rayon
  helpers, and with all seven macros equally unconsumed there is no
  evidence basis for a partial subset. Ported unchanged (78 counted LOC vs
  80 budget) as `solinas/parallel.rs`, wired exactly as the baseline:
  optional `rayon` dep behind `parallel = ["dep:rayon"]`, module gated on
  `solinas`, macros `#[macro_export]`ed with expansion-site `cfg` so they
  dispatch on the consuming crate's own `parallel` feature. **Flag for the
  replacement PR:** if no consumer materializes when consumers rebind, the
  whole component (and possibly the feature) is a deletion candidate.
- **Coverage:** `tests/parallel_macros.rs` exercises every macro against an
  explicitly serial computation; the suite runs in both configurations
  (without `parallel` → sequential expansions, `--all-features` → rayon),
  so a green run in both is the serial-vs-parallel equivalence proof.

## Final LOC audit (checkpoint 9)

Final per-file actuals are recorded in the file-structure table below
(counted with the awk counter above): **5,103 total vs the 6,240 budget
(−1,137, 18% under; 55% below the 11,410 baseline).**

**Budget trades for review** (every over-budget file, consolidated):

- `src/solinas/mod.rs` 113/90 (+26%): the registry grew four `Fp128`
  aliases plus the `reduce_le_bytes_mod_order` shared helper; worst
  percentage overrun, but it is the crate's declarative registry — golfing
  it means deleting doc-typed aliases.
- `src/packed.rs` 105/90 (+17%): the four `ext4/ext8` kernel hooks with
  schedule defaults must live on the `Packed` trait (overridable defaults
  need the trait); recorded at checkpoint 8.
- `src/solinas/packed/mod.rs` 36/30 (+20%): per-ISA backend selection for
  three ISAs plus `NoPacking` fallback; component (packed selection
  120 = 90 + 30) is at 141 (+17.5%).
- `src/limbs.rs` 237/220 (+8%): within the ≤10% discussion band.
- `src/bn254/mont.rs` 310/300 (+3%): within the band; offset by
  `bn254/mod.rs` at −126 (component 700 budget → 584 actual).
- `src/solinas/packed/simd.rs` 352/350 (+1%): weight moved here from
  `engine.rs` (284/550) by the generic-types-over-vocabulary design;
  recorded at checkpoint 8.

## Replacement-time deviations

1. **`JoltField` drops the serde bounds** (`Serialize + DeserializeOwned`)
   while the temporary `akita` bootstrap edge exists: the pre-cutover
   `akita-field` type is foreign, so the orphan rule forbids giving it serde
   impls here, and it must satisfy `JoltField` for the akita lanes to build.
   This matches the old umbrella (`Field` never carried serde bounds); every
   first-party type keeps its `impl_serde_bytes!` impls. Restore the bounds
   at the akita cutover when the bootstrap edge is deleted.
2. **`CanonicalEncoding` re-split**: byte surface extracted as a bare
   `CanonicalBytes` supertrait (transcript absorption and `NoCommitment`
   bind to bytes only; same decision as the baseline's ff5bf9c split).
3. **bn254 `From<primitive>` impls added** (`from_primitives!`): the old
   crate's `Fr`/`Fq` exposed the plain arkworks `From` conversions; 94+
   consumer call sites rely on them.
4. **Post-baseline signed accumulator optimizations retained through the
   consolidated contract**: current main added real users of BN254's small
   signed-scalar and signed 256-bit product accumulators after this rebuild
   had deleted their then-unused standalone trait families. `WithAccumulator`
   now exposes general, small-scalar, and signed-product associated types, all
   implementing the single canonical `Accumulator` trait. BN254 keeps the
   specialized representations; other backends use `NaiveAccumulator`. The
   duplicate `SignedScalarAccumulator`, `SignedProductAccumulator`, and
   wrapper traits remain deleted.
5. **Post-baseline Akita field capabilities moved into the canonical Jolt
   contracts**: current Akita added a zero-copy canonical `u32` slice, a
   bounded commitment accumulator role, `Field::mul_add`, and the
   `canonical_extension_basis` name after the original cutover base. The
   refreshed Jolt crate owns these capabilities directly. `Fp128` restores
   the combined-reduction multiply-add kernel, and the commitment role reuses
   `Unreduced::Wide` instead of introducing a second accumulator type.
6. **The Fp32 quartic deferred-reduction kernel is restored on x86-64 only.**
   The original Apple M4 result still justifies the generic scalar schedule on
   AArch64. Akita's shared-field cutover then supplied the missing x86-64
   system-level evidence: same-runner interleaved profile samples on an AMD
   EPYC 7763 put both Fp32 prove cases about 9% behind the pre-cutover field,
   while Fp64/Fp128 results were mixed and near noise. The x86-64 hook now
   reuses the canonical `FpExt4Fp32ProductAccum` formula and reduces four
   slots once; sub-32-bit Fp32 moduli retain the generic schedule.
7. **Fp128 architecture add and subtract kernels are restored.** The A7F7
   AArch64 and x86-64 instruction bodies are shared by production inline assembly and
   standalone proof objects. HOL Light proves the object bytes. An optimized
   public witness checks that Rust includes the same words.
   The same proof path now covers baseline x86-64 multiplication. That kernel
   uses `mulq`, `add`, and `adc`, so it does not require BMI2 or ADX. The proved
   inline path is about 20% faster than the former portable path in the focused
   x86-64 Criterion benchmark under Rosetta.
8. **Scalar challenge decoding is field specific.** BN254 and Dory preserve
   the historical byte reversal. Fp128 uses Akita's direct little endian
   convention. Old packed Akita proofs are not supported.
9. **The parallel helper set gains `cfg_try_fold_reduce!`.** `cfg_join!` now
   resolves Rayon through a hidden crate reexport, so downstream expansion
   does not require a direct Rayon dependency.

## Replacement validation evidence

The replacement (swap `cf8a66ae3`, consumer rebind `079356e30`) is
validated by the following battery, all green:

- **Proof bytes unchanged (the hard invariant, verified at the strongest
  applicable strength):** a standard-mode `muldiv` proof built from the
  pre-replacement commit (`2b800e1ca`, old crate) and one from the
  replacement branch, with the Dory URS cache pinned to a shared directory,
  are byte-identical in all 63,372 bytes (`cmp` finds no difference). ZK
  proofs are randomized (BlindFold), so the applicable check is size
  equality: 65,947 bytes on both builds. (Earlier recorded baselines
  63,371/65,946 predate the PR branch's merge of main; the one-byte drift
  exists identically on both sides of the replacement.)
- **e2e:** `muldiv` passes in `--features host` and `--features host,zk`
  (3/3 each, including the committed-program variants).
- **Test lanes:** workspace default sweep 2,338/2,338; jolt-prover-legacy
  host 444/444, zk 480/480, akita 445/445; jolt-verifier
  akita,prover-fixtures 84/84; jolt-field solinas-only 110/110. The crate's
  own oracle-free suite: 125/125 all-features.
- **Clippy lanes (`--all-targets -- -D warnings`):** host; host,zk;
  jolt-verifier akita and akita,prover-fixtures; jolt-prover-legacy akita;
  field-inline; plus `+avx2` and `+avx512f,+avx512dq` `cargo check`
  cross-compiles to `x86_64-apple-darwin`.

## Remaining after replacement

1. **x86-64 runtime validation:** AVX2/AVX-512 packed backends and the
   fp128 portable mul path are `cargo check`-validated with
   `-C target-feature` only (checkpoint 8/9 acceptance); run the packed
   differential suite and the fp128 differentials on real x86-64 hardware.
2. **Bench re-evaluation resolved for the Akita production profile:** Akita's
   interleaved x86-64 end-to-end profile exposed the Fp32 loss and motivated
   the target-specific hook recorded in replacement-time deviation 6. Keep
   `benches/ext4_kernels.rs` as the focused architecture comparison harness.
3. **CI wiring:** a target-feature lane so SIMD is not CI-dark.
4. **Parallel helpers: deletion withdrawn.** The checkpoint 9 audit's
   "zero consumers" verdict was workspace-blind: Akita's cutover branch
   consumes `jolt_field::parallel` at 32 sites. The helpers stay; the
   Akita rebind binds them at their new path.
5. **Akita cutover follow-ups:** delete the `akita` bootstrap edge and
   restore `JoltField`'s serde bounds (deviation 1 above).

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

Actuals are the checkpoint-9 final audit (awk counter above).

| File | Budget | Actual | Contents |
|---|---|---|---|
| **Contract layer (root, unconditional)** | | | |
| `src/lib.rs` | 70 | 43 | crate docs, feature gates, re-exports, `FieldError` |
| `src/algebra.rs` | 260 | 252 | spine: 7 traits + `NaiveAccumulator` + `PseudoMersenne` |
| `src/extension.rs` | 60 | 60 | contracts: `ExtField`, `Ext2Config` + NR config ZSTs, `MulBaseUnreduced` |
| `src/unreduced.rs` | 70 | 18 | contracts: `Unreduced`, `Fold` |
| `src/packed.rs` | 90 | **105** | contracts: `Packed`, `WithPacking` + generic `NoPacking` |
| `src/ops.rs` | 180 | 160 | `impl_ring_ops!`, `impl_serde_bytes!` (backend-neutral) |
| `src/schedules.rs` | 140 | 133 | lane-generic deg-4/8 ext coefficient schedules — unconditional because the `PseudoMersenne` hook defaults (algebra.rs) and the packed lanes (checkpoint 8) share them; carved out of the old `ext.rs` budget (890 → 750, component total unchanged) |
| `src/limbs.rs` | 220 | **237** | `Limbs<N>` |
| `src/signed.rs` | 420 | 357 | signed bigint families (consumer-audited surface) |
| **bn254 backend** | | | |
| `src/bn254/mod.rs` | 400 | 274 | `Fr`, `Fq` via one wrapping macro; serde; transcript bytes |
| `src/bn254/mont.rs` | 300 | **310** | Barrett/Montgomery kernel + `WideAccumulator` |
| **solinas backend** | | | |
| `src/solinas/mod.rs` | 90 | **113** | offset registry, aliases, shared helpers |
| `src/solinas/word.rs` | 380 | 355 | `define_solinas_prime!` → `Fp32`, `Fp64` |
| `src/solinas/fp128.rs` | 700 | 528 | two-limb add/sub/mul/reduce/wide |
| `src/solinas/ext.rs` | 750 | 617 | FpExt2/4/8 impls, ExtField impls, Frobenius + Moore |
| `src/solinas/unreduced.rs` | 530 | 501 | lane accumulators, fold matrices, contract impls |
| `src/solinas/parallel.rs` | 80 | 78 | rayon helpers |
| `src/solinas/packed/mod.rs` | 30 | **36** | backend selection |
| `src/solinas/packed/simd.rs` | 350 | **352** | per-ISA primitive vocabulary (neon/avx2/avx512) |
| `src/solinas/packed/engine.rs` | 550 | 284 | shared packed algebra, stamped per width × ISA |
| `src/solinas/packed/fp128.rs` | 350 | 99 | 128-bit-lane engine |
| `src/solinas/packed/ext.rs` | 230 | 191 | packed FpExt2/4/8 |
| **Total** | **6,240** | **5,103** | vs 11,410 baseline (budget −45%, actual −55%) |

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
