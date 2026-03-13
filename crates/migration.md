# Arkworks Fork Migration Plan

**Status:** Draft
**Date:** 2026-03-04
**Scope:** Remove `a16z/arkworks-algebra` fork (`dev/twist-shout`), switch to upstream `ark-*` v0.5.0 from crates.io, relocate all fork-specific functionality into Jolt crates.

---

## 1. Background

Jolt depends on a fork of `arkworks-rs/algebra` (branch `dev/twist-shout`) that is **178 commits ahead / 95 commits behind** upstream. The fork patches `ark-ff`, `ark-ec`, `ark-bn254`, and `ark-serialize`, and adds an entirely new `jolt-optimizations` crate. This fork is identified in the RFC (finding #6) as unmaintainable, not upstream-able, and a CI liability.

The refactoring spec (section 2.3) mandates: *"Arkworks types implement Jolt traits. Arkworks never appears in trait bounds or public API signatures. Contained in `arkworks` submodules behind the crate's own traits."*

This document defines how to get there.

---

## 2. Inventory of Fork-Specific Features

### 2.1 Actively Used

| # | Feature | Where in fork | Usage in Jolt | Call sites |
|---|---------|---------------|---------------|------------|
| 1 | `PrimeField::from_u64::<WINDOW>` — precomputed Montgomery lookup table for small integer → field conversion | `ark-ff` `montgomery_backend.rs` | `jolt-field` `bn254.rs`, `jolt-core` `field/ark.rs` | 8 sites, all `from_u{8,16,32,64}` paths |
| 2 | `Fp::mul_u64::<W>`, `mul_i64::<W>`, `mul_u128::<W,L>`, `mul_i128::<W,L>` — optimized field×scalar multiplication | `ark-ff` `Fp` impl | `jolt-field` `bn254.rs`, `jolt-core` `field/ark.rs` | 8 sites |
| 3 | `BigInt::mul_trunc::<M, L>` — truncated bigint multiplication (L output limbs from N×M input) | `ark-ff` `BigInt` impl | `UnreducedOps` impl for `Fr`, R1CS evaluation, accumulation | ~20 sites across 6 files |
| 4 | `Fr::from_montgomery_reduce::<L, W>`, `Fr::from_barrett_reduce::<L, W>` — const-generic reduction | `ark-ff` `Fp` impl | `ReductionOps` impl for `Fr` | 4 sites |
| 5 | `Fr::from_bigint_unchecked` — skip modular reduction | `ark-ff` `Fp` impl | Challenge → field conversion | 6 sites |
| 6 | `Fr::mul_by_hi_2limbs` — multiply by value with only 2 high limbs populated | `ark-ff` `Fp` impl | `Challenge * Field` hot path (every sumcheck round) | 3 sites |
| 7 | `SignedBigInt<N>` / `SignedBigIntHi32<N>` (`S64`, `S128`, `S160`, `S192`, `S256`) — sign-magnitude big integers | `ark-ff` `biginteger/signed.rs`, `signed_hi_32.rs` | R1CS evaluation, accumulation, increment witness gen, SmallScalar impls | ~100+ usages across 10+ files |
| 8 | Typed MSM functions (`msm_binary`, `msm_u8`, `msm_u16`, `msm_u32`, `msm_u64`, `msm_i64`, `msm_u128`, `msm_i128`, `msm_s64`, `msm_s128`) | `ark-ec` `variable_base/mod.rs` | `jolt-core/msm/mod.rs`, Dory wrappers, SmallScalar trait impls | ~40 sites across 4 files |
| 9 | `jolt-optimizations` crate: `batch_g1_additions_multi`, `vector_scalar_mul_add_gamma_g1_online`, `fixed_base_vector_msm_g1`, `vector_add_scalar_mul_g{1,2}_online`, `glv_four_scalar_mul_online`, etc. | Separate crate in fork | Dory commitment scheme, one-hot polynomial commitments | 9 sites in 3 files |
| 10 | Extended Jacobian `Bucket<P>` type (XYZZ coordinates) | `ark-ec` `bucket.rs` | Internal to typed MSMs (not called directly from Jolt) | 0 direct, used internally by #8 |
| 11 | Faster Montgomery multiplication (Ingonyama `ingo_skyscraper`) | `ark-ff` `montgomery_backend.rs` | Implicit — all field multiplication goes through this | Pervasive |

### 2.2 Not Used (Can Be Dropped)

| Feature | Notes |
|---------|-------|
| `CompressibleFq12` / `CompressedFq12` / compressed pairing | Not referenced anywhere in Jolt |
| `find_relaxed_naf` | Not referenced |
| `mul_hi_bigint_u128` | Not referenced |
| `FromPsi6Pow` trait | Not referenced |
| `Fq12` multilinear evaluation (`fq12_to_multilinear_evals`, etc.) | Not referenced |
| SageMath build scripts for BN254 tables | Dependency of `jolt-optimizations` internal tables |

---

## 3. Migration Strategy

### 3.0 Principle

Every fork modification falls into exactly one of three buckets:

1. **Drop** — unused, or superseded by an upstream improvement in v0.5.0
2. **Internalize** — move into the appropriate Jolt crate behind Jolt's own traits
3. **Upstream** — submit a PR to `arkworks-rs/algebra` (only if the change is generally useful and small enough to be accepted)

We do **not** attempt to upstream anything that is BN254-specific, Jolt-specific, or would require const-generic API changes to arkworks. Practically, almost everything is internalized.

### 3.1 Destination Mapping

| Fork feature | Bucket | Destination crate | Rationale |
|---|---|---|---|
| `from_u64::<W>` precomputed table | Internalize | `jolt-field` | Implement as a static lookup table in `jolt-field/src/arkworks/precomp.rs`. The `Field::from_u{8,16,32,64}` trait methods already abstract this; only the impl changes. |
| `Fp::mul_{u,i}{64,128}::<W>` | Internalize | `jolt-field` | Implement as standalone functions in `jolt-field/src/arkworks/scalar_mul.rs`. Call from `Field::mul_{u,i}{64,128}` impls. |
| `BigInt::mul_trunc::<M,L>` | Internalize | `jolt-field` | Implement as a free function `mul_trunc(a: &[u64; N], b: &[u64; M]) -> [u64; L]` in `jolt-field/src/arkworks/bigint.rs`. Used only through `UnreducedOps`. |
| `from_montgomery_reduce::<L,W>` / `from_barrett_reduce::<L,W>` | Internalize | `jolt-field` | Implement in `jolt-field/src/arkworks/reduction.rs`. Used only through `ReductionOps`. |
| `from_bigint_unchecked` | Internalize | `jolt-field` | Trivial `unsafe { transmute(bigint) }` in the BN254 impl. |
| `mul_by_hi_2limbs` | Internalize | `jolt-field` | Implement in `jolt-field/src/arkworks/scalar_mul.rs`. Used by Challenge multiplication. |
| `SignedBigInt` / `S64` / `S128` / etc. | Internalize | `jolt-field` | New module `jolt-field/src/signed.rs` (or `jolt-field/src/signed/`). Pure arithmetic, no arkworks dependency. These are Jolt-specific types; upstream arkworks has no concept of signed bigints. |
| Typed MSM (`msm_u8`, `msm_u16`, ...) | Internalize | `jolt-dory` (Dory-specific MSMs) + `jolt-field` (via a `TypedMsm` trait) | The MSM dispatch is only used by Dory commitment and SmallScalar. Implement Pippenger-style bucket MSM for each scalar width. This is the largest migration item. |
| `Bucket<P>` (XYZZ coordinates) | Internalize | Co-located with typed MSM | Internal accumulator type for the MSM implementations. |
| `jolt-optimizations` crate | Internalize | `jolt-dory` | Move all Dory-specific routines (GLV, batch addition, vector ops) into `jolt-dory/src/optimizations/`. These are exclusively used by the Dory commitment scheme. |
| Faster Montgomery multiplication | Drop / Accept regression | N/A | Upstream ark-ff v0.5.0 includes its own Montgomery improvements. Benchmark to measure actual delta. If regression is unacceptable, implement the Ingonyama path as an inline asm module in `jolt-field`. |

---

## 4. Migration Phases

### Phase 1: Signed BigInt Extraction

**Goal:** Remove dependency on `ark_ff::biginteger::{S64, S128, S160, S192, S256, SignedBigInt, SignedBigIntHi32}`.

**Steps:**

1. Create `jolt-field/src/signed.rs` (or `jolt-field/src/signed/mod.rs` + sub-files if large).
2. Implement `SignedBigInt<const N: usize>` as a standalone Jolt type:
   - `[u64; N]` magnitude + `bool` sign
   - Arithmetic: `add`, `sub`, `mul`, `neg`, `abs`
   - `mul_trunc::<M, P>` — truncated multiplication (critical hot path)
   - Conversions: `from_u64`, `from_i64`, `from_u64_with_sign`, `from_i128`, `magnitude_as_u128`, `to_i128`
   - `SmallScalar` compatibility: `.is_positive`, `.magnitude`
3. Type aliases: `S64 = SignedBigInt<1>`, `S128 = SignedBigInt<2>`, `S160 = SignedBigIntHi32<3>` (or flatten into explicit types if the hi32 variant is simpler standalone).
4. Re-export from `jolt-field` so all downstream crates import from `jolt_field::signed::S64` instead of `ark_ff::biginteger::S64`.
5. Update all import sites (~10 files).
6. Run `muldiv` e2e test.

**Risk:** Medium. `mul_trunc` is on the hot path. Must be bit-exact with the fork's implementation.

**Validation:** Property tests comparing output against the fork's `S64/S128` for random inputs.

---

### Phase 2: Field Arithmetic Internalization

**Goal:** Remove dependency on fork-specific `ark-ff` methods (`from_u64::<W>`, `mul_u64::<W>`, `mul_trunc`, `from_montgomery_reduce`, `from_barrett_reduce`, `from_bigint_unchecked`, `mul_by_hi_2limbs`).

**Steps:**

1. **Precomputed Montgomery table** — `jolt-field/src/arkworks/precomp.rs`:
   - Static array of `[Fr; 16384]` mapping `i → Montgomery(i)` for BN254.
   - Generated at compile time via `const fn` or `lazy_static`.
   - `from_u64` checks `n < 16384` → table lookup, else falls back to upstream `Fr::from(BigUint)`.
   - This replaces `PrimeField::from_u64::<5>`.

2. **Scalar multiplication** — `jolt-field/src/arkworks/scalar_mul.rs`:
   - `fn mul_u64_fast(f: Fr, n: u64) -> Fr` — double-and-add with windowed approach.
   - `fn mul_i64_fast(f: Fr, n: i64) -> Fr` — sign-split, delegate to `mul_u64_fast`.
   - `fn mul_u128_fast(f: Fr, n: u128) -> Fr` — split into two 64-bit limbs, two multiplies.
   - `fn mul_i128_fast(f: Fr, n: i128) -> Fr`.
   - `fn mul_by_hi_2limbs(f: Fr, lo: u64, hi: u64) -> Fr` — specialized 2-limb multiply for Challenge path.

3. **Truncated BigInt multiplication** — `jolt-field/src/arkworks/bigint.rs`:
   - `fn mul_trunc<const N: usize, const M: usize, const L: usize>(a: &[u64; N], b: &[u64; M]) -> [u64; L]`
   - Schoolbook multiplication, truncated to L output limbs.
   - Replaces `BigInt::mul_trunc` method.

4. **Reduction** — `jolt-field/src/arkworks/reduction.rs`:
   - `fn montgomery_reduce<const L: usize>(limbs: [u64; L]) -> Fr`
   - `fn barrett_reduce<const L: usize>(limbs: [u64; L]) -> Fr`
   - `fn from_bigint_unchecked(bigint: BigInt<4>) -> Fr` — direct transmute.

5. Update `bn254.rs` to call these new functions instead of the fork methods.
6. Update `challenge/mont_u128.rs` and `challenge/macros.rs` similarly.
7. Run `muldiv` e2e test.

**Risk:** High. These are the hottest paths in the prover. Every field operation flows through here.

**Validation:**
- Unit tests comparing output against known BN254 values.
- Benchmark suite comparing performance against the fork. Must be within 5% or better.
- Full e2e test suite pass.

---

### Phase 3: Typed MSM Internalization

**Goal:** Remove dependency on fork-specific `ark-ec` MSM functions (`msm_binary`, `msm_u8`, ..., `msm_s128`).

**Steps:**

1. Create `jolt-core/src/msm/pippenger.rs` (or `jolt-dory/src/msm/` depending on final crate structure):
   - Generic Pippenger bucket MSM parametric on scalar width.
   - Bucket accumulator type using extended Jacobian (XYZZ) coordinates — port `Bucket<P>` from the fork.
   - Per-width specializations: `msm_u8`, `msm_u16`, `msm_u32`, `msm_u64` with window sizes tuned for the scalar bit-width.
   - Signed variants (`msm_i64`, `msm_i128`, `msm_s64`, `msm_s128`): partition into positive/negative, two unsigned MSMs, subtract.

2. The `SmallScalar` trait in `jolt-core/src/utils/small_scalar.rs` already dispatches to these — update the dispatch to call the new implementations.

3. Update `jolt-core/src/msm/mod.rs` to remove the `ark_ec::scalar_mul::variable_base` imports.

4. Update Dory wrappers (`jolt-core/src/poly/commitment/dory/wrappers.rs`) to call the new MSMs.

5. Run `muldiv` e2e test.

**Risk:** High. MSM is the second-hottest path after field arithmetic.

**Validation:**
- Correctness: for each MSM variant, compare `msm_typed(bases, scalars)` against `arkworks::VariableBaseMSM::msm(bases, scalars_as_fr)` for random inputs.
- Performance: benchmark against the fork's MSM. Must match or beat.

**Note:** If the new crate structure places MSM in `jolt-dory`, then `jolt-core/src/msm/` becomes a thin trait + dispatch layer.

---

### Phase 4: jolt-optimizations Extraction

**Goal:** Remove dependency on the `jolt-optimizations` crate from the arkworks fork.

**Steps:**

1. Create `jolt-dory/src/optimizations/` with the following modules:
   - `glv.rs` — GLV-2 (G1) and GLV-4 (G2) scalar multiplication, Strauss-Shamir interleaving.
   - `batch_add.rs` — batched affine addition using Montgomery's trick (`batch_g1_additions_multi`).
   - `vector_ops.rs` — `vector_scalar_mul_add_gamma_g1_online`, `fixed_base_vector_msm_g1`, `vector_add_scalar_mul_g{1,2}_online`, `vector_scalar_mul_add_gamma_g2_online`.

2. Port each function from the fork's `jolt-optimizations` crate. These are all BN254-specific and use `ark_bn254::G1Affine`, `G2Affine`, `G1Projective`, `G2Projective` directly — that's fine, this crate is not backend-generic.

3. The GLV endomorphism constants (lambda, beta for G1; Frobenius coefficients for G2) need to be ported as `const` values. The SageMath-generated tables from the fork should be converted to Rust `const` arrays.

4. Update call sites:
   - `jolt-core/src/poly/commitment/dory/commitment_scheme.rs` (2 sites)
   - `jolt-core/src/poly/commitment/dory/jolt_dory_routines.rs` (6 sites)
   - `jolt-core/src/poly/one_hot_polynomial.rs` (2 sites)

5. Run `muldiv` e2e test.

**Risk:** Medium. These are performance-critical but well-encapsulated. The porting is mostly mechanical.

**Validation:**
- Unit tests for GLV: `glv_mul(base, scalar) == scalar * base` for random inputs.
- Batch addition: `batch_add(points) == points.iter().sum()`.
- Benchmark Dory commitment against the fork version.

---

### Phase 5: Switch to Upstream Arkworks

**Goal:** Replace all `[patch.crates-io]` entries and git dependencies with crates.io `v0.5.0`.

**Steps:**

1. Update `Cargo.toml`:
   ```toml
   # Remove entirely:
   [patch.crates-io]
   ark-bn254 = { git = "..." }
   ark-ff = { git = "..." }
   ark-ec = { git = "..." }
   ark-serialize = { git = "..." }

   # Change workspace dependencies to:
   ark-bn254 = { version = "0.5", default-features = false }
   ark-grumpkin = { version = "0.5", default-features = false }
   ark-ec = { version = "0.5", default-features = false }
   ark-ff = { version = "0.5", default-features = false }
   ark-serialize = { version = "0.5", default-features = false, features = ["derive"] }
   ark-secp256k1 = { version = "0.5", default-features = false }

   # Remove:
   jolt-optimizations = { git = "..." }
   ```

2. Fix all compilation errors. Expected breakages:
   - `PrimeField::from_u64::<5>` → call the new `precomp` module instead.
   - `Fp::mul_u64::<5>` → call `scalar_mul::mul_u64_fast`.
   - `BigInt::mul_trunc::<M, L>` → call `bigint::mul_trunc`.
   - `Fr::from_montgomery_reduce::<L, 5>` → call `reduction::montgomery_reduce`.
   - `Fr::from_barrett_reduce::<L, 5>` → call `reduction::barrett_reduce`.
   - `Fr::from_bigint_unchecked` → call `reduction::from_bigint_unchecked`.
   - `S64`, `S128`, etc. → import from `jolt_field::signed`.
   - `msm_u8`, etc. → import from the new MSM module.
   - `jolt_optimizations::*` → import from `jolt_dory::optimizations`.
   - Upstream v0.5.0 API changes (if any) in `ark_serialize`, `ark_ec` trait methods.

3. `cargo clippy --all --message-format=short -q --all-targets --features allocative,host -- -D warnings`

4. `cargo nextest run -p jolt-core muldiv --cargo-quiet`

5. Full test suite pass.

**Risk:** Medium. Most of the work was done in phases 1–4. This phase is primarily `Cargo.toml` changes and fixing remaining compilation errors from upstream API differences.

---

### Phase 6: Montgomery Multiplication Performance Validation

**Goal:** Quantify and address any regression from losing the Ingonyama `ingo_skyscraper` Montgomery multiplication.

**Steps:**

1. Benchmark baseline (fork) prover time on `sha3` and `muldiv`.
2. Benchmark post-migration prover time on the same programs.
3. If regression > 5%:
   - Profile to confirm Montgomery multiplication is the bottleneck.
   - Option A: Implement optimized Montgomery as inline assembly in `jolt-field/src/arkworks/mont_asm.rs` (ARM NEON / x86 ADX paths).
   - Option B: Wrap upstream's multiplication and add an optimization layer for the specific BN254 modulus.
   - Option C: Submit the Montgomery improvement as an upstream PR to `arkworks-rs/algebra`.
4. If regression ≤ 5%: accept and move on.

**Risk:** Unknown until benchmarked. Upstream v0.5.0 may have already incorporated similar improvements.

---

## 5. File Changes Summary

### New files

| File | Contents |
|------|----------|
| `jolt-field/src/signed.rs` (or `signed/mod.rs`) | `SignedBigInt<N>`, `S64`, `S128`, `S160`, `S192`, `S256` |
| `jolt-field/src/arkworks/precomp.rs` | Precomputed Montgomery table for small → field conversion |
| `jolt-field/src/arkworks/scalar_mul.rs` | `mul_u64_fast`, `mul_i64_fast`, `mul_u128_fast`, `mul_i128_fast`, `mul_by_hi_2limbs` |
| `jolt-field/src/arkworks/bigint.rs` | `mul_trunc` free function |
| `jolt-field/src/arkworks/reduction.rs` | `montgomery_reduce`, `barrett_reduce`, `from_bigint_unchecked` |
| `jolt-dory/src/optimizations/mod.rs` | Module root |
| `jolt-dory/src/optimizations/glv.rs` | GLV-2 and GLV-4 scalar multiplication |
| `jolt-dory/src/optimizations/batch_add.rs` | Batched affine addition (Montgomery's trick) |
| `jolt-dory/src/optimizations/vector_ops.rs` | Dory vector operations |
| `jolt-dory/src/msm/` or `jolt-core/src/msm/pippenger.rs` | Typed Pippenger MSM + XYZZ Bucket type |

### Modified files

| File | Change |
|------|--------|
| `Cargo.toml` (root) | Remove `[patch.crates-io]`, switch git deps to crates.io v0.5.0, remove `jolt-optimizations` |
| `jolt-core/Cargo.toml` | Remove `jolt-optimizations` dependency |
| `jolt-field/src/arkworks/bn254.rs` | Replace fork method calls with internal implementations |
| `jolt-field/src/challenge/macros.rs` | Replace `mul_by_hi_2limbs` call |
| `jolt-field/src/challenge/mont_u128.rs` | Replace `from_bigint_unchecked` call |
| `jolt-core/src/msm/mod.rs` | Replace fork MSM imports with internal MSM |
| `jolt-core/src/utils/small_scalar.rs` | Update `S64`/`S128` imports, MSM dispatch |
| `jolt-core/src/utils/math.rs` | Update `S64` import |
| `jolt-core/src/utils/accumulation.rs` | Update signed bigint imports |
| `jolt-core/src/zkvm/r1cs/inputs.rs` | Update signed bigint imports |
| `jolt-core/src/zkvm/r1cs/evaluation.rs` | Update signed bigint imports |
| `jolt-core/src/zkvm/claim_reductions/increments.rs` | Update signed bigint imports |
| `jolt-core/src/poly/multilinear_polynomial.rs` | Update `S128` import |
| `jolt-core/src/poly/commitment/dory/commitment_scheme.rs` | Replace `jolt_optimizations::` calls |
| `jolt-core/src/poly/commitment/dory/jolt_dory_routines.rs` | Replace `jolt_optimizations::` calls |
| `jolt-core/src/poly/commitment/dory/wrappers.rs` | Replace typed MSM calls |
| `jolt-core/src/poly/one_hot_polynomial.rs` | Replace `batch_g1_additions_multi` calls |

### Deleted dependencies

| Dependency | How removed |
|------------|-------------|
| `a16z/arkworks-algebra` git (all 4 crates) | Replaced by crates.io `v0.5.0` |
| `jolt-optimizations` | Code moved into `jolt-dory` |

---

## 6. Testing Strategy

### Per-Phase Gates

Each phase must pass before proceeding to the next:

1. `cargo clippy --all --message-format=short -q --all-targets --features allocative,host -- -D warnings`
2. `cargo nextest run -p jolt-core muldiv --cargo-quiet`
3. No performance regression > 5% on `sha3` profile benchmark

### Correctness Tests (New)

| Test | Location | What it validates |
|------|----------|-------------------|
| `signed_bigint_arithmetic` | `jolt-field/tests/` | `S64`/`S128` add, sub, mul, mul_trunc match expected values |
| `precomp_table_correctness` | `jolt-field/tests/` | Every entry `i` in `[0, 16384)` satisfies `table[i] == Fr::from(i as u64)` (upstream method) |
| `scalar_mul_consistency` | `jolt-field/tests/` | `mul_u64_fast(f, n) == f * Fr::from(n)` for random f, n |
| `reduction_roundtrip` | `jolt-field/tests/` | `montgomery_reduce(a.mul_unreduced(b)) == a * b` |
| `typed_msm_correctness` | `jolt-core/tests/` or `jolt-dory/tests/` | `msm_u8(bases, scalars) == VariableBaseMSM::msm(bases, scalars.map(Fr::from))` |
| `glv_correctness` | `jolt-dory/tests/` | `glv_mul(base, scalar) == base * scalar` for random inputs |
| `batch_add_correctness` | `jolt-dory/tests/` | `batch_add(points) == naive_sum(points)` |

### Performance Benchmarks

| Benchmark | Target |
|-----------|--------|
| `field_from_u64` | < 5% regression vs fork |
| `field_mul_u64` | < 5% regression vs fork |
| `mul_trunc` | < 5% regression vs fork |
| `msm_u8` (1M points) | < 5% regression vs fork |
| `dory_commit` (2^20 poly) | < 5% regression vs fork |
| `sha3` e2e prove | < 5% regression vs fork |

---

## 7. Ordering & Dependencies

```
Phase 1 (Signed BigInt)
    │
    ▼
Phase 2 (Field Arithmetic)  ──── can start in parallel with Phase 3
    │                                       │
    ▼                                       ▼
Phase 3 (Typed MSM)          Phase 4 (jolt-optimizations)
    │                                       │
    └──────────────┬────────────────────────┘
                   ▼
            Phase 5 (Switch to upstream)
                   │
                   ▼
            Phase 6 (Performance validation)
```

Phases 1 and 2 are sequential (Phase 2 depends on `SignedBigInt` types being available). Phases 3 and 4 can proceed in parallel once Phase 2 is complete. Phase 5 is the final integration step. Phase 6 is validation.

---

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Montgomery multiplication regression | Medium | High | Benchmark early (Phase 6). Inline ASM fallback if needed. |
| `mul_trunc` bit-exactness | Low | High | Property tests comparing against the fork for 10M random inputs. |
| Upstream v0.5.0 API breaks beyond `[patch]` removal | Low | Medium | Review upstream changelog before starting. Compile early. |
| Typed MSM performance regression | Medium | High | Port the exact Pippenger window sizes from the fork. Benchmark per-variant. |
| GLV constant correctness | Low | Critical | Verify against the SageMath scripts in the fork. Cross-check with test vectors. |
| `ark-grumpkin` / `ark-secp256k1` upstream availability | Low | Low | These are in `arkworks-rs/curves` v0.5.0 on crates.io. |
