# Spec: jolt-crypto Performance Optimizations

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @0xAndoroid                    |
| Created     | 2026-04-20                     |
| Status      | partially-implemented          |
| PR          | #1453                          |

> **Implementation note:** Optimizations 1, 2, 6, 7, 9 and the four new
> `jolt-eval` invariants + four new performance objectives landed in PR
> #1453. Optimizations 3 (GT sliding-window MSM), 4 (wNAF in Shamir 2D/4D),
> 5 (precomputed 4D Shamir table), and 8 (native `i128` arithmetic in
> `decompose_scalar_2d`) are deferred to a follow-up PR as permitted by
> Alternatives Considered §1 — they are the most algorithmically involved
> and merit independent benchmarking and review.

## Summary

`jolt-crypto` is the backend-agnostic cryptographic group and commitment primitives crate for Jolt. The initial implementation (merged in #1368) prioritized a clean abstraction boundary (`JoltGroup`, `PairingGroup`, `Pedersen`) and safety (subgroup checks, repr(transparent) wrappers, comprehensive tests) over peak performance. A review identified nine concrete optimization opportunities in the hot paths exercised by Dory, Pedersen, Spartan, and the ECDSA inlines. Since BN254 group operations and MSMs dominate prover time inside Dory commitment and opening proofs, and GLV is invoked per-round during Dory reduction, these optimizations are expected to meaningfully reduce end-to-end prover time without touching the public API. This spec captures all nine optimizations as a single coordinated work item so they can land in one PR (or a coordinated series) with unified benchmarks and a single correctness gate.

## Intent

### Goal

Reduce CPU time spent inside `jolt-crypto` hot paths (MSM, scalar multiplication, GLV vector ops, GT exponentiation, GLV scalar decomposition, batch affine addition) through nine targeted optimizations that preserve the public API, preserve all correctness invariants, and do not alter the results of any group operation or commitment.

The optimizations are:

1. **`field_to_fr` specialization**: Avoid `F → bytes → Fr` serialization roundtrip when the concrete `F` is already `jolt_field::Fr`. Use a `TypeId`-based fast path (or a dedicated `AsFr` helper trait implemented for `Fr`) so that callers passing `Fr` pay zero conversion cost, while generic callers keep the existing byte-roundtrip path.
2. **MSM batch-normalization**: Replace per-point `b.0.into_affine()` in the `impl_jolt_group_wrapper!` `msm` path with `<$projective>::normalize_batch(...)` so a single inversion amortizes across all input points, matching the pattern already used in `multi_pairing`.
3. **GT MSM sliding-window exponentiation with shared squarings**: Replace the serial `for` loop in `Bn254GT::msm` with per-base windowed exponentiation that amortizes squarings across scalar bit positions (e.g., simultaneous multi-exponentiation à la Straus for small batches, or windowed per-base with a shared accumulator for large batches).
4. **wNAF signed-digit in Shamir's trick**: Replace naive bit-by-bit double-and-add in `shamir_glv_mul_2d` and `shamir_glv_mul_4d` with wNAF (width-4 for 2D, width-5 for 4D) including sign-aware precomputed odd-multiple tables per base.
5. **Precomputed 16-entry Shamir table for 4D GLV online path**: Extend the 2D fixed-base precomputation pattern (`PrecomputedShamir2Table`, 16 entries) to 4D with `PrecomputedShamir4Table` (256 entries = 4 points × 2 sign bits = 8 bits), invoked from `glv_four_scalar_mul_online` and both `dory_g2` vector ops.
6. **Parallelize post-inversion loop in `batch_g1_additions_multi_affine_inner`**: The lambda/x3/y3 computation after `batch_inversion` is currently a serial `for ((set_idx, pair_idx), inv) in pair_info.iter().zip(inverses.iter())` loop. Convert to a parallel rayon pass that writes into per-set `Vec<G1Affine>` buffers without cross-set contention.
7. **Cache GLV 2D `SCALAR_DECOMP_COEFFS`**: Move the per-call `SCALAR_DECOMP_COEFFS.map(|x| BigInt::from_bytes_be(...))` reconstruction in `decompose_scalar_2d` behind a `OnceLock<[BigInt; 4]>` (or equivalent `LazyLock`), since the constants are identical across every call.
8. **Native `u128`/`i128` arithmetic in `decompose_scalar_2d`**: The lattice coefficients fit in 128 bits; replace `num_bigint::BigInt` arithmetic (heap-allocated, limb-by-limb) with native `i128` (or `u128` with explicit sign tracking), mirroring the approach already used in `decompose_scalar_4d`.
9. **Cache `FrobeniusCoefficients`**: Replace `get_frobenius_coefficients()` (which rebuilds `Fq2` elements from `MontFp!` literals on every call) with a `OnceLock<FrobeniusCoefficients>` or, if `Fq2::new(MontFp!(...))` is `const`-evaluable on the target arkworks version, a `const FROBENIUS_COEFFICIENTS: FrobeniusCoefficients`. `frobenius_psi_power_projective` is called per-point in the `dory_g2` hot path.

All nine optimizations are **observably equivalent to the current implementation** — the only behavioral change is reduced wall-clock time and, for optimization (8), reduced allocator pressure.

### Invariants

The existing `jolt-crypto` test suite already covers the properties that must hold:

- `tests/group_laws.rs` — associativity, commutativity, identity, inverse, `double() == add_self`, `scalar_mul(0) == identity`, `scalar_mul(1) == self`, MSM vs naive for G1 and G2
- `tests/pairing.rs` — bilinearity, `multi_pairing` vs sum-of-individual, GT group laws
- `tests/pedersen.rs` — commit/verify round-trip, binding, additive homomorphism, capacity panics, prefix-generator behavior
- `tests/coverage.rs` — GLV vs naive for `vector_add_scalar_mul_{g1,g2}`, `vector_scalar_mul_add_gamma_{g1,g2}`, `glv_four_scalar_mul`, `fixed_base_vector_msm_g1`; large random scalar exercises; `HomomorphicCommitment` correctness for G1/G2/GT
- `tests/serialization.rs` — bincode + JSON round-trips for G1/G2/GT/PedersenSetup; identity round-trips
- `tests/fuzz/*` — existing fuzz targets

**Every optimization must preserve every existing test**. Concretely:

1. **API-level equivalence**: For any public function in `jolt_crypto::{ec, commitment, Pedersen, Bn254*}`, the return value for any given input is bit-for-bit unchanged.
2. **Pairing-friendly consistency (ZK gate)**: The BlindFold sumcheck prover/verifier (`subprotocols/blindfold/`) relies on exact equality of `y_com` Pedersen commitments and `pairing`/`multi_pairing` outputs. The `muldiv` e2e test in `jolt-core` — which covers both `--features host` and `--features host,zk` — is the canonical consistency gate.
3. **Subgroup membership**: GT deserialization's r-torsion and non-zero checks must remain active (they are not in a hot path and are not touched by this spec).
4. **No new unsafe**: The existing `#[repr(transparent)]` unsafe casts in `batch_addition.rs` and `glv/mod.rs` are the only unsafe blocks in the crate. This spec adds no new `unsafe`.

**`jolt-eval` invariants (existing) that apply:**

- `soundness` (RedTeam) — end-to-end soundness: any change to BN254 group ops that silently corrupts results will manifest as `soundness` violations in the RedTeam harness (which runs the full prover/verifier). **This optimization must not violate `soundness`.** No modification to this invariant is required.

**New invariants to add via `/new-invariant`:**

- **`jolt_crypto_msm_matches_naive`** (Test, Fuzz): For random `bases: Vec<Bn254G1>` of length `n` and random `scalars: Vec<Fr>` of length `n` (with `n ∈ [0, 256]`), `Bn254G1::msm(&bases, &scalars) == Σ bases[i].scalar_mul(&scalars[i])`. Symmetric target for `Bn254G2` and `Bn254GT`.
- **`jolt_crypto_glv_vector_matches_naive`** (Test, Fuzz): For random `v`, `generators`, `scalar`, the post-state of `glv::vector_add_scalar_mul_g1(&mut v, &generators, scalar)` equals `v[i] = v_prev[i] + generators[i] * scalar` for all `i`. Symmetric for `g2`, and for both `vector_scalar_mul_add_gamma_*` variants.
- **`jolt_crypto_batch_addition_matches_naive`** (Test, Fuzz): For a random `bases: Vec<Bn254G1>` and random distinct-x `indices_sets`, each output `batch_g1_additions_multi(&bases, &indices_sets)[k]` equals the naive sum `Σ_{i ∈ indices_sets[k]} bases[i]`.
- **`jolt_crypto_scalar_decomp_reconstructs`** (Test, Fuzz): For random `scalar: Fr`, `decompose_scalar_2d(scalar)` reconstructs to `scalar` when combined with the BN254 GLV endomorphism eigenvalue λ: `sign(k1) * k1 + sign(k2) * k2 * λ ≡ scalar (mod n)`. Symmetric for `decompose_scalar_4d` with all four λⁱ powers.

### Non-Goals

1. **No new curves or backends**: This spec only touches the BN254 backend. Lattice / other group implementations of `JoltGroup` are not in scope.
2. **No API changes**: The `JoltGroup`, `PairingGroup`, `Commitment`, `VectorCommitment`, `HomomorphicCommitment`, and `DeriveSetup` traits are frozen. The `Pedersen` and `Bn254*` public items are frozen.
3. **No changes to serialization format**: bincode/JSON output bytes for any group element must be byte-identical to the current output (so existing fixtures and on-disk proof artifacts continue to load).
4. **No changes to the 4D GLV power-of-2 decomposition table**: `power_of_2_decompositions.rs` (2799 lines of precomputed constants) is not regenerated or modified; only the consumer (`decompose_scalar_table_based`) is untouched — optimization (7) and (8) apply to 2D decomposition only.
5. **No algorithmic change to pairing computation**: `ark_bn254::Bn254::pairing` / `multi_pairing` are kept as-is (already optimal arkworks implementations).
6. **No changes to the `bn254` feature-gating**: The crate remains `default = ["bn254"]` with arkworks optional.
7. **No constant-time guarantees**: `jolt-crypto` is a prover-side library. No side-channel hardening is claimed or introduced — the existing code is variable-time (uses `if coeff.get_bit(bit_idx)`), and optimizations (3), (4), (5) remain variable-time.

## Evaluation

### Acceptance Criteria

- [ ] All existing `jolt-crypto` tests (`tests/coverage.rs`, `tests/group_laws.rs`, `tests/pairing.rs`, `tests/pedersen.rs`, `tests/serialization.rs`) pass unchanged on `cargo nextest run -p jolt-crypto --cargo-quiet`.
- [ ] All existing `jolt-crypto/fuzz/` targets build and corpus seeds pass (`cargo +nightly fuzz build` inside `crates/jolt-crypto/fuzz/`).
- [ ] `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host` passes (standard mode e2e).
- [ ] `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk` passes (ZK mode e2e).
- [ ] `cargo clippy --all --features host -q --all-targets -- -D warnings` passes.
- [ ] `cargo clippy --all --features host,zk -q --all-targets -- -D warnings` passes.
- [ ] The four new `jolt-eval` invariants (`jolt_crypto_msm_matches_naive`, `jolt_crypto_glv_vector_matches_naive`, `jolt_crypto_batch_addition_matches_naive`, `jolt_crypto_scalar_decomp_reconstructs`) are added with both `Test` and `Fuzz` targets and pass their seed + random corpora.
- [ ] Criterion benchmark `crypto` (existing `crates/jolt-crypto/benches/crypto.rs`) reports **≥ 15% speedup on `g1_msm/1024`** vs. the pre-optimization baseline (the dominant MSM workload in Dory commitment).
- [ ] Criterion benchmark `crypto` reports **≥ 10% speedup on `pedersen_commit/1024`** (same root cause: MSM batch-normalize).
- [ ] Criterion benchmark `crypto` reports **≥ 2× speedup on `gt_scalar_mul`** (sliding-window exponentiation).
- [ ] Criterion benchmark `crypto` reports **≥ 10% speedup on `g1_scalar_mul`** (GLV-path improvements: wNAF, native-int decomp, cached coeffs).
- [ ] No regression >2% on any existing `crypto` benchmark (`g1_add`, `g1_double`, `pairing`, `multi_pairing/*`, `g1_serialize_bincode`, `g1_deserialize_bincode`).
- [ ] `jolt-eval` Criterion benchmark `prover_time_secp256k1_ecdsa_verify` reports **no regression** (threshold: within 1% noise floor) and ideally a **5-10% speedup** (ECDSA verification is MSM- and GLV-heavy).
- [ ] `jolt-eval` Criterion benchmarks `prover_time_fibonacci_100` and `prover_time_sha2_chain_100` report **no regression** (within 1% noise floor).
- [ ] Public API of `jolt-crypto` is unchanged: `cargo public-api` (or equivalent `rustdoc` diff) shows no additions or removals from `jolt_crypto::*`.
- [ ] Binary-level serialization compatibility: a `Bn254G1` serialized with the pre-optimization code deserializes identically with the post-optimization code (tested via a checked-in fixture or an explicit round-trip across old/new byte arrays).

### Testing Strategy

**Existing tests that must continue passing unchanged:**

- `cargo nextest run -p jolt-crypto --cargo-quiet` — all 5 integration test suites
- `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host` — primary correctness gate (standard)
- `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk` — primary correctness gate (ZK)
- `cargo nextest run --cargo-quiet` — full workspace suite
- All advice e2e tests in `jolt-core` (they exercise non-ZK + advice polynomials that hit Pedersen + MSM)

**New tests:**

- Four new `jolt-eval` invariants listed above (each with `Test` + `Fuzz` targets, minimum 10 seed corpus entries each, plus `JOLT_RANDOM_ITERS` random inputs).
- A `tests/regression.rs` (or expansion of `tests/coverage.rs`) adding: (a) random-Fr MSM correctness for `n ∈ {0, 1, 2, 3, 15, 16, 17, 63, 64, 65, 255, 256, 257, 1023}` to exercise batch-normalize path including Pippenger boundaries; (b) binary-compat fixture test: deserialize a hex-encoded pre-optimization `Bn254G1` blob and assert equality to a known-good point.
- Expand `tests/coverage.rs` GLV tests to include `glv_four_scalar_mul` with `n = 1, 2, 16, 64` points to exercise the new precomputed 4D Shamir table path across batch sizes.

**Feature coverage:**

- `--features host` (standard): required.
- `--features host,zk` (ZK): required. BlindFold uses Pedersen commitments and MSM heavily; any bit-level deviation breaks Fiat-Shamir transcript consistency.
- `--no-default-features --features bn254`: required, since that is the only configuration in which `jolt-crypto` builds standalone.

**Fuzz campaigns (CI):**

- Add each of the four new invariants as a `libfuzzer_sys` target under `crates/jolt-crypto/fuzz/fuzz_targets/` via the `jolt_eval::fuzz_invariant!` macro.
- Run `cargo +nightly fuzz run <target> -- -runs=100000` for each new target before merging as a gating check.

### Performance

**Existing `jolt-eval` performance objectives expected to move:**

- **`prover_time_secp256k1_ecdsa_verify`**: ↓ 5–10%. ECDSA verification inside the guest calls into BN254 G1 scalar multiplications and MSMs during proof-side Dory commitment; GLV (optimizations 4, 7, 8) and MSM batch-normalize (optimization 2) are on this path.
- **`prover_time_sha2_chain_100`**: ↓ 2–5%. Every Dory commitment and Pedersen commitment in the outer prover pipeline hits `Bn254G1::msm`, so batch-normalize (2) and `field_to_fr` specialization (1) apply. Smaller magnitude because SHA2 chain's constraint-heavy body dominates.
- **`prover_time_fibonacci_100`**: ↓ 2–5%. Same rationale as SHA2 chain; Fibonacci is smaller so the fixed MSM costs are a larger fraction, but the absolute savings are small.

Direction for all three: **monotone decrease** (minimize is the desired direction, matching `objective_fn::minimize_*` conventions).

**New `jolt-eval` performance objectives to add via `/new-objective`:**

- **`jolt_crypto_g1_msm_1024`**: Wall-clock time of `Bn254G1::msm` with 1024 random bases and 1024 random scalars. Benchmarked via Criterion. Rationale: dominant cost in Dory tier-1 commitment. Target: ≥ 15% reduction vs. `main` at spec commit time.
- **`jolt_crypto_gt_scalar_mul`**: Wall-clock time of `Bn254GT::scalar_mul` with a random non-trivial GT element and a random scalar. Rationale: GT exponentiation is called inside Dory opening verification (linear-combine-GT-targets step). Target: ≥ 2× speedup (sliding window).
- **`jolt_crypto_g1_scalar_mul`**: Wall-clock time of `Bn254G1::scalar_mul` with a random G1 point and random scalar. Rationale: GLV-path regression canary; hit indirectly by many callers. Target: ≥ 10% reduction.
- **`jolt_crypto_pedersen_commit_1024`**: Wall-clock time of `Pedersen::<Bn254G1>::commit` with 1024 generators + 1024 values + blinding. Rationale: direct BlindFold hot path. Target: ≥ 10% reduction.

These objectives reuse the existing `crates/jolt-crypto/benches/crypto.rs` Criterion harness — they need `jolt-eval` wrapper modules under `jolt-eval/src/objective/performance/` and corresponding bench files under `jolt-eval/benches/`, synced via `./jolt-eval/sync_targets.sh` per the framework's convention.

**No new code-quality objectives.** Optimizations (1), (7), (9) may slightly *increase* LLOC (adding `OnceLock`/`LazyLock` setup and specialization shims). That is acceptable for ≥10% runtime wins; this spec does not try to minimize `lloc` or `cognitive_complexity_avg` as a side effect.

**Baseline protocol:**

1. Check out `main` at the commit merging #1368.
2. Run `cargo bench -p jolt-crypto --bench crypto -- --save-baseline pre-perf-opts`.
3. Run `cargo bench -p jolt-eval --bench prover_time_secp256k1_ecdsa_verify -- --save-baseline pre-perf-opts-ecdsa`.
4. Apply the spec's changes incrementally, re-benching each optimization against the same baseline.
5. Final PR must include a `benches/RESULTS.md` (or PR description table) showing before/after for every benchmark listed in Acceptance Criteria.

## Design

### Architecture

All changes are contained within `crates/jolt-crypto/src/` plus additive changes to `jolt-eval/`. No cross-crate type boundaries move.

**File-by-file impact:**

| File | Optimization | Change type |
|------|--------------|-------------|
| `crates/jolt-crypto/src/ec/bn254/mod.rs` | (1) `field_to_fr` specialization | Add `AsFr` trait + `Fr` impl + `TypeId` fast path in `field_to_fr`; extend macro's `scalar_mul` / `msm` to call it |
| `crates/jolt-crypto/src/ec/bn254/mod.rs` | (2) MSM batch-normalize | Change `msm` macro body: replace `bases.iter().map(into_affine).collect()` with `<$projective>::normalize_batch(projective_slice)` |
| `crates/jolt-crypto/src/ec/bn254/gt.rs` | (3) GT MSM sliding-window | Rewrite `Bn254GT::msm` and `Bn254GT::scalar_mul` using windowed exp (width-4 or width-5) with shared squarings per base |
| `crates/jolt-crypto/src/ec/bn254/glv/glv_two.rs` | (4) wNAF for 2D | Replace `shamir_glv_mul_2d` double-and-add with wNAF-width-4 + precomputed odd-multiple table |
| `crates/jolt-crypto/src/ec/bn254/glv/glv_four.rs` | (4) wNAF for 4D | Replace `shamir_glv_mul_4d` double-and-add with wNAF-width-5 per base, interleaved Strauss-style |
| `crates/jolt-crypto/src/ec/bn254/glv/glv_four.rs` | (5) 4D Shamir table | Add `PrecomputedShamir4Table` (256 entries) mirroring `PrecomputedShamir2Table`; wire into `glv_four_scalar_mul_online` |
| `crates/jolt-crypto/src/ec/bn254/glv/dory_g2.rs` | (5) 4D Shamir table | Use the new table inside `vector_add_scalar_mul_g2_online` and `vector_scalar_mul_add_gamma_g2_online` |
| `crates/jolt-crypto/src/ec/bn254/batch_addition.rs` | (6) Parallel post-inversion | Convert the serial `for ((set_idx, pair_idx), inv)` loop to `pair_info.par_iter().zip(inverses.par_iter())` with per-set `Mutex<Vec<G1Affine>>` or, preferably, pre-sized `Vec<Vec<G1Affine>>` split per set and reassembled |
| `crates/jolt-crypto/src/ec/bn254/glv/decomp_2d.rs` | (7) Cache decomp coeffs | Add `static DECOMP_COEFFS: LazyLock<[BigInt; 4]>`; replace per-call `.map(...)` |
| `crates/jolt-crypto/src/ec/bn254/glv/decomp_2d.rs` | (8) Native int arithmetic | Rewrite `decompose_scalar_2d` body to use `i128` / `u128` arithmetic instead of `BigInt`; keep the `Fr`↔`BigInt` boundary only at entry/exit |
| `crates/jolt-crypto/src/ec/bn254/glv/constants.rs` | (9) Cache Frobenius | Replace `get_frobenius_coefficients()` fn with either `const FROBENIUS_COEFFICIENTS: FrobeniusCoefficients` (preferred if `Fq2::new(MontFp!(...))` is const) or `static FROBENIUS_COEFFICIENTS: LazyLock<FrobeniusCoefficients>` |
| `crates/jolt-crypto/src/ec/bn254/glv/frobenius.rs` | (9) Cache Frobenius | Update `frobenius_psi_power_projective` to read from the cached/const value |
| `jolt-eval/src/invariant/` (new files) | Correctness gates | Add `jolt_crypto_msm.rs`, `jolt_crypto_glv_vector.rs`, `jolt_crypto_batch_addition.rs`, `jolt_crypto_scalar_decomp.rs`; update `mod.rs` |
| `jolt-eval/src/objective/performance/` (new files) | Perf objectives | Add `jolt_crypto_g1_msm.rs`, `jolt_crypto_gt_scalar_mul.rs`, `jolt_crypto_g1_scalar_mul.rs`, `jolt_crypto_pedersen_commit.rs`; update `mod.rs` |
| `jolt-eval/benches/` (new files) | Bench harnesses | Add Criterion bench wrappers for each new objective; run `./jolt-eval/sync_targets.sh` |
| `jolt-eval/fuzz/fuzz_targets/` (new files) | Fuzz harnesses | Add 3-line fuzz wrappers per `fuzz_invariant!` for each new invariant |

**Interaction diagram** (call-graph, simplified):

```
zkvm prover
    │
    ├─► Dory commit ──► Bn254G1::msm ──► field_to_fr (opt 1)
    │                       │
    │                       └─► into_affine per-point ✗  →  normalize_batch ✓ (opt 2)
    │
    ├─► Dory opening verify ──► Bn254GT::scalar_mul ──► Fq12::pow (opt 3)
    │                                                          │
    │                                                          └─► sliding-window (opt 3)
    │
    ├─► Dory reduction round ──► dory_g1::vector_add_scalar_mul_g1
    │                                   │
    │                                   ├─► decompose_scalar_2d (opt 7, 8)
    │                                   └─► shamir_glv_mul_2d (opt 4)
    │
    ├─► Dory reduction round ──► dory_g2::vector_add_scalar_mul_g2
    │                                   │
    │                                   ├─► decompose_scalar_4d (unchanged)
    │                                   ├─► frobenius_psi_power_projective (opt 9)
    │                                   └─► shamir_glv_mul_4d  →  PrecomputedShamir4Table (opts 4, 5)
    │
    ├─► BlindFold Pedersen ──► Bn254G1::msm (opts 1, 2 apply)
    │
    └─► batch_addition (one-hot MSM in Dory) ──► batch_g1_additions_multi
                                                       │
                                                       └─► parallel lambda/x3/y3 (opt 6)
```

**No new abstractions.** No new traits are introduced in the public API. The `AsFr` trait from optimization (1) is a private helper (`pub(crate) trait AsFr { fn as_fr(&self) -> ark_bn254::Fr; }`) and is not re-exported.

### Alternatives Considered

1. **Land one optimization per PR instead of a single coordinated PR.** Rejected: the four new invariants and four new performance objectives are scaffolding that justifies a shared introduction. Splitting the nine optimizations into nine PRs would force repeated rebasing of the benchmark baseline and dilute the review signal. If review latency becomes a concern during implementation, the single PR may be split into at most three: (A) infrastructure + invariants + objectives, (B) hot-path optimizations 1–5, (C) secondary optimizations 6–9.

2. **Specialization via nightly `specialization` feature vs. `TypeId` for `field_to_fr`.** Rejected nightly specialization: the workspace builds on stable Rust. `TypeId`-based dispatch costs one integer compare per call and is eliminated by the optimizer when `F` is monomorphized to `Fr`. An alternative `AsFr` trait added to `jolt_field::Field` would be cleaner but introduces a cross-crate API surface; we prefer the `TypeId` branch contained inside `jolt-crypto`.

3. **Replace arkworks `VariableBaseMSM` with a hand-rolled Pippenger.** Rejected: arkworks' implementation is already well-tuned, and our win is at the *projective→affine* stage, not in the MSM core. If future profiling shows the MSM core itself is limiting, a follow-up spec can reconsider `blst`-style hand-rolled Pippenger.

4. **Use `ark-ec`'s `MSM::msm_unchecked` with pre-normalized bases.** Rejected: the internal `msm_bigint` API is already what we call, and the checked-vs-unchecked distinction does not apply here (our bases are type-checked `G1Projective`). Batch-normalize happens before the MSM call either way.

5. **Use LSB-first vs. MSB-first wNAF for optimization (4).** LSB-first is simpler (no upfront maximum-bit scan) but adds one final `result.double()` compensation. We choose MSB-first wNAF for symmetry with the current `shamir_glv_mul_*` loop structure; the compensation is not worth the minor simplification.

6. **Precompute the 4D Shamir table once per `decompose_scalar_4d` call batch vs. once per point.** Chosen: once per point. The 4D table is 256 entries × `size_of::<G2Projective>()` ≈ 48 KB per point; for `glv_four_scalar_mul_online` with `n` points, this is `n × 48 KB` of allocation. For typical Dory batch sizes (thousands of G2 points), that is 100+ MB transient — acceptable given the savings. An alternative "shared table across points" only works for fixed-base MSM (see `fixed_base_vector_msm_g1`, which already does this for 2D).

7. **`const FROBENIUS_COEFFICIENTS` vs. `LazyLock`.** Preferred `const` if `Fq2::new(MontFp!(...))` is a `const fn` in the pinned arkworks version (`dev/twist-shout` branch). If not, fall back to `LazyLock` — performance difference is negligible after the first call.

8. **Use `Arc<FrobeniusCoefficients>` for thread-safe sharing.** Rejected: `FrobeniusCoefficients` contains only `Fq2` values (POD, `Copy`), so `&'static FrobeniusCoefficients` via `LazyLock::get()` is sufficient and avoids the `Arc` overhead.

## Documentation

**No `book/` changes.** This is an internal performance refactor; the crate's public API and user-visible behavior are unchanged. The new `jolt-eval` invariants and objectives are documented automatically by the `/new-invariant` and `/new-objective` workflows (which update `jolt-eval/README.md` tables).

Inline-code documentation:

- Each new `OnceLock`/`LazyLock` site gets a one-line `// WHY` comment explaining the original per-call cost.
- The `field_to_fr` specialization site gets a comment linking to this spec.
- The wNAF implementations reference the Hankerson et al. Guide to Elliptic Curve Cryptography, §3.3 (signed-digit representations).
- The new `PrecomputedShamir4Table` doc-comment explains the 256-entry layout (4 point bits × 4 sign bits).

## Execution

Order of implementation (dependency-respecting, smallest-blast-radius first):

1. **Infrastructure** — Add the four new `jolt-eval` invariants (Test + Fuzz) and run them against pre-optimization code to confirm they pass (baseline).
2. **Infrastructure** — Add the four new `jolt-eval` performance objectives + Criterion benches + bench wrappers. Save `pre-perf-opts` baselines.
3. **Optimization (9) — Frobenius cache** (smallest change, lowest risk, exercises the new bench harness).
4. **Optimization (7) — GLV 2D coeff cache** (same pattern as 9).
5. **Optimization (8) — Native int arithmetic in `decompose_scalar_2d`** (correctness-sensitive; compare outputs bit-for-bit against current `BigInt` implementation via the new `jolt_crypto_scalar_decomp_reconstructs` invariant).
6. **Optimization (2) — MSM batch-normalize** (one-line change; dominant perf win).
7. **Optimization (1) — `field_to_fr` specialization** (touches the macro; exercised by every `scalar_mul`/`msm` call).
8. **Optimization (6) — Parallelize batch_addition post-inversion** (isolate to `batch_g1_additions_multi_affine_inner`).
9. **Optimization (4) — wNAF in Shamir** (two functions, 2D and 4D; measure against `g1_scalar_mul` / `g2_scalar_mul`).
10. **Optimization (5) — 4D Shamir precomputed table** (depends on (4) to establish the wNAF baseline).
11. **Optimization (3) — GT sliding-window MSM** (most algorithmically involved; largest `gt_scalar_mul` win).

After each optimization, the full `cargo nextest run` and `muldiv` standard+zk gates must pass before moving to the next. The implementer is expected to use the `verifier` agent to confirm the benchmark deltas before claiming a step complete.

## References

- [PR #1368](https://github.com/a16z/jolt/pull/1368) — Introduction of the `jolt-crypto` crate.
- Gallant, Lambert, Vanstone. *Faster Point Multiplication on Elliptic Curves with Efficient Endomorphisms*. CRYPTO 2001. [Springer link](https://link.springer.com/chapter/10.1007/3-540-44647-8_11) (GLV).
- Lee. *Dory: Efficient, Transparent Arguments for Generalised Inner Products and Polynomial Commitments*. [eprint 2020/1274](https://eprint.iacr.org/2020/1274) (MSM / GT exponentiation hot paths).
- Hankerson, Menezes, Vanstone. *Guide to Elliptic Curve Cryptography*. Springer, 2004. §3.3 (wNAF / signed-digit), §3.3.2 (sliding-window exponentiation), §4.3 (Frobenius endomorphism on BN curves).
- Straus. *Addition Chains of Vectors*. American Mathematical Monthly, 1964 (simultaneous multi-exponentiation; basis for Shamir's trick).
- [`arkworks-algebra`](https://github.com/a16z/arkworks-algebra) `dev/twist-shout` branch — pinned via `[patch.crates-io]`; source for `normalize_batch` and `VariableBaseMSM`.
- [`jolt-eval/README.md`](../jolt-eval/README.md) — invariant and objective framework used by this spec's acceptance gates.
