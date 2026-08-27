//! Device lane for the Dory reduce-round vector folds
//! (`jk_g1_scalar_mul_add`, `jk_g2_scalar_mul_add`, `jk_g2_fixed_base_mul`).
//!
//! dory-pcs's reduce-and-fold rounds spend their EC time in two
//! uniform-scalar vector ops per group — `vs[i] += s·bases[i]` (apply the β
//! challenge) and `vs[i] = s·vs[i] + addends[i]` (fold under α) — both
//! instances of one kernel shape: `out[i] = s·P[i] + Q[i]` with a single
//! scalar shared by the whole vector. Thread-per-element double-and-add
//! keeps every bit branch warp-uniform (the W3b combine-rows insight); the
//! host batch-normalizes both point vectors to affine so the device runs
//! only the parity-tested mixed formulas, with identities lowered to the
//! `(0, 0)` sentinel. The G2 side adds the VMV preamble's fixed-base sweep
//! `out[i] = base·scalars[i]` (v₂ = v_vec · Γ2,fin): per-thread scalars,
//! shared base, uniform doublings. Results return as Jacobian points —
//! group-equal to the CPU fold; every consumer (pairings, MSMs, later
//! folds, serialized final message) normalizes before use, so proof bytes
//! are unchanged.

use ark_bn254::{
    Fq as ArkFq, Fq2 as ArkFq2, Fr as ArkFr, G1Affine, G1Projective, G2Affine, G2Projective,
};
use ark_ec::CurveGroup;
use ark_ff::{AdditiveGroup, PrimeField, Zero};

use super::field::FR_U32_LIMBS;
use super::g1::{bases_as_u32s, jac_from_device_limbs, JAC_U32S};
use super::g2::{g2_bases_as_u32s, g2_jac_from_device_limbs, G2_JAC_U32S};
use super::runtime::{KernelId, MetalContext};
use super::{metal_gate, testing, MetalError};

/// Canonical (integer) little-endian u32 limbs of a scalar plus its highest
/// set bit — the kernel ladder's start. A zero scalar reports bit 0; the
/// ladder still terminates with an identity accumulator.
pub(super) fn scalar_limbs_and_start_bit(scalar: &ArkFr) -> ([u32; FR_U32_LIMBS], u32) {
    let big = scalar.into_bigint();
    let mut limbs = [0u32; FR_U32_LIMBS];
    let mut start_bit = 0u32;
    for (word_index, word) in big.0.iter().enumerate() {
        let lo = *word as u32;
        let hi = (*word >> 32) as u32;
        limbs[2 * word_index] = lo;
        limbs[2 * word_index + 1] = hi;
        for (half, limb) in [(0u32, lo), (1u32, hi)] {
            if limb != 0 {
                let bit = (word_index as u32) * 64 + half * 32 + limb.ilog2();
                start_bit = start_bit.max(bit);
            }
        }
    }
    (limbs, start_bit)
}

/// Batch-normalize to affine with identities lowered to the `(0, 0)`
/// sentinel (not on `y² = x³ + 3`, nor on the G2 curve) the kernels skip.
/// Only nonzero points reach `normalize_batch`, so its shared inversion
/// never sees a zero Z.
fn g1_affine_sentinels(points: &[G1Projective]) -> Vec<G1Affine> {
    let nonzero: Vec<G1Projective> = points.iter().copied().filter(|p| !p.is_zero()).collect();
    let normalized = G1Projective::normalize_batch(&nonzero);
    let mut normalized_iter = normalized.into_iter();
    points
        .iter()
        .map(|p| {
            if p.is_zero() {
                G1Affine::new_unchecked(ArkFq::zero(), ArkFq::zero())
            } else {
                #[expect(
                    clippy::expect_used,
                    reason = "normalize_batch returns one point per nonzero input"
                )]
                normalized_iter
                    .next()
                    .expect("normalized point per nonzero input")
            }
        })
        .collect()
}

/// One `jk_g1_scalar_mul_add` dispatch: `out[i] = scalar·ps[i] + qs[i]`.
/// Synchronous; both vectors must have equal length.
pub fn g1_scalar_mul_add_device(
    ctx: &MetalContext,
    ps: &[G1Projective],
    qs: &[G1Projective],
    scalar: &ArkFr,
) -> Result<Vec<G1Projective>, MetalError> {
    assert_eq!(ps.len(), qs.len(), "vector lengths must match");
    let n = ps.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let (scalar_limbs, start_bit) = scalar_limbs_and_start_bit(scalar);

    let ps_affine = g1_affine_sentinels(ps);
    let qs_affine = g1_affine_sentinels(qs);
    let ps_buffer = ctx.wrap_slice(bases_as_u32s(&ps_affine))?;
    let qs_buffer = ctx.wrap_slice(bases_as_u32s(&qs_affine))?;
    let out_buffer = ctx.alloc_u32s(n * JAC_U32S)?;
    testing::note_copied_buffers(
        u64::from(ps_buffer.was_copied()) + u64::from(qs_buffer.was_copied()),
    );

    let mut params = [0u32; 2 + FR_U32_LIMBS];
    params[0] = u32::try_from(n)
        .map_err(|_| MetalError::Execution("fold vector length overflows u32".to_owned()))?;
    params[1] = start_bit;
    params[2..].copy_from_slice(&scalar_limbs);
    ctx.run_once(
        KernelId::G1ScalarMulAdd,
        &params,
        &[&ps_buffer, &qs_buffer, &out_buffer],
        n,
    )?;
    testing::note_device_round();

    let mut jac = vec![0u32; n * JAC_U32S];
    out_buffer.copy_to_u32s(&mut jac);
    Ok(jac
        .chunks_exact(JAC_U32S)
        .map(jac_from_device_limbs)
        .collect())
}

/// The G2 twin of [`g1_affine_sentinels`]: `(0, 0)` is off the twist too
/// (its `b` is nonzero).
fn g2_affine_sentinels(points: &[G2Projective]) -> Vec<G2Affine> {
    let nonzero: Vec<G2Projective> = points.iter().copied().filter(|p| !p.is_zero()).collect();
    let normalized = G2Projective::normalize_batch(&nonzero);
    let mut normalized_iter = normalized.into_iter();
    points
        .iter()
        .map(|p| {
            if p.is_zero() {
                G2Affine::new_unchecked(ArkFq2::zero(), ArkFq2::zero())
            } else {
                #[expect(
                    clippy::expect_used,
                    reason = "normalize_batch returns one point per nonzero input"
                )]
                normalized_iter
                    .next()
                    .expect("normalized point per nonzero input")
            }
        })
        .collect()
}

/// Canonical LE u32 limbs of one 4-GLV subscalar.
fn bigint_limbs(big: &ark_ff::BigInt<4>) -> [u32; FR_U32_LIMBS] {
    let mut limbs = [0u32; FR_U32_LIMBS];
    for (word_index, word) in big.0.iter().enumerate() {
        limbs[2 * word_index] = *word as u32;
        limbs[2 * word_index + 1] = (*word >> 32) as u32;
    }
    limbs
}

/// The four ψ^k-transformed, sign-folded affine base blocks for the GLV
/// ladder, block-major (`[k·n + i]`). ψ on affine points is conjugation
/// plus one Fq2 coefficient multiplication per coordinate — no inversions
/// — and maps the `(0, 0)` identity sentinel to itself.
fn g2_psi_blocks(ps_affine: &[G2Affine], signs: &[bool; 4]) -> Vec<G2Affine> {
    let coeffs = jolt_optimizations::constants::get_frobenius_coefficients();
    let psi_xy: [(ArkFq2, ArkFq2); 3] = [
        (coeffs.psi1_coef2, coeffs.psi1_coef3),
        (coeffs.psi2_coef2, coeffs.psi2_coef3),
        (coeffs.psi3_coef2, coeffs.psi3_coef3),
    ];
    let n = ps_affine.len();
    let mut blocks = Vec::with_capacity(4 * n);
    for (k, block_sign) in signs.iter().enumerate() {
        for a in ps_affine {
            let (mut x, mut y) = (a.x, a.y);
            if k > 0 {
                if k & 1 == 1 {
                    let _ = x.conjugate_in_place();
                    let _ = y.conjugate_in_place();
                }
                let (cx, cy) = psi_xy[k - 1];
                x *= cx;
                y *= cy;
            }
            if *block_sign {
                y = -y;
            }
            blocks.push(G2Affine::new_unchecked(x, y));
        }
    }
    blocks
}

/// One `jk_g2_scalar_mul_add` dispatch: `out[i] = scalar·ps[i] + qs[i]`,
/// via the 4-GLV ψ-decomposition (`s = Σ ±s_k·λ^k`, ~64-bit subscalars —
/// a 4× shorter ladder than the scalar's 254 bits). Synchronous; both
/// vectors must have equal length.
pub fn g2_scalar_mul_add_device(
    ctx: &MetalContext,
    ps: &[G2Projective],
    qs: &[G2Projective],
    scalar: &ArkFr,
) -> Result<Vec<G2Projective>, MetalError> {
    assert_eq!(ps.len(), qs.len(), "vector lengths must match");
    let n = ps.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let (coeffs, signs) = jolt_optimizations::decomp_4d::decompose_scalar_4d(*scalar);
    let start_bit = coeffs
        .iter()
        .map(ark_ff::BigInteger::num_bits)
        .max()
        .unwrap_or(0)
        .saturating_sub(1);

    let ps_blocks = g2_psi_blocks(&g2_affine_sentinels(ps), &signs);
    let qs_affine = g2_affine_sentinels(qs);
    let ps_buffer = ctx.wrap_slice(g2_bases_as_u32s(&ps_blocks))?;
    let qs_buffer = ctx.wrap_slice(g2_bases_as_u32s(&qs_affine))?;
    let out_buffer = ctx.alloc_u32s(n * G2_JAC_U32S)?;
    testing::note_copied_buffers(
        u64::from(ps_buffer.was_copied()) + u64::from(qs_buffer.was_copied()),
    );

    let mut params = [0u32; 2 + 4 * FR_U32_LIMBS];
    params[0] = u32::try_from(n)
        .map_err(|_| MetalError::Execution("fold vector length overflows u32".to_owned()))?;
    params[1] = start_bit;
    for (k, coeff) in coeffs.iter().enumerate() {
        params[2 + k * FR_U32_LIMBS..2 + (k + 1) * FR_U32_LIMBS]
            .copy_from_slice(&bigint_limbs(coeff));
    }
    ctx.run_once(
        KernelId::G2ScalarMulAdd,
        &params,
        &[&ps_buffer, &qs_buffer, &out_buffer],
        n,
    )?;
    testing::note_device_round();

    let mut jac = vec![0u32; n * G2_JAC_U32S];
    out_buffer.copy_to_u32s(&mut jac);
    Ok(jac
        .chunks_exact(G2_JAC_U32S)
        .map(g2_jac_from_device_limbs)
        .collect())
}

/// Window-table toggle for the fixed-base sweep
/// (`JOLT_DORY_FIXED_BASE_TABLE=0` restores the plain per-thread ladder).
/// The base is shared by every thread, so the host prices one 16-ary table
/// (`d·16^win·base`, ≤64×15 affine entries, ~2 ms) and each thread pays only
/// its ~60 nonzero-nibble mixed adds — the plain ladder's ~254 doublings,
/// its dominant term, disappear. Group-equal results (different op order,
/// same sum); every consumer normalizes before serializing.
fn fixed_base_table_enabled() -> bool {
    !std::env::var("JOLT_DORY_FIXED_BASE_TABLE").is_ok_and(|value| value.trim() == "0")
}

/// One `jk_g2_fixed_base_table` dispatch: `out[i] = base·scalars[i]` off a
/// host-built window table. `windows` covers the top nonzero nibble across
/// the scalar batch.
fn g2_fixed_base_table_device(
    ctx: &MetalContext,
    base_affine: &G2Affine,
    scalar_limbs: &[u32],
    start_bit: u32,
    n: usize,
) -> Result<Vec<G2Projective>, MetalError> {
    let windows = (start_bit / 4 + 1) as usize;
    let mut entries: Vec<G2Projective> = Vec::with_capacity(windows * 15);
    let mut cur: G2Projective = (*base_affine).into();
    for _ in 0..windows {
        let row_start = entries.len();
        entries.push(cur);
        for _ in 1..15 {
            let last = entries[entries.len() - 1];
            entries.push(last + cur);
        }
        cur = entries[row_start + 7].double(); // 16·cur = 2·(8·cur)
    }
    let table_affine = G2Projective::normalize_batch(&entries);

    let scalars_buffer = ctx.wrap_slice(scalar_limbs)?;
    let table_buffer = ctx.wrap_slice(g2_bases_as_u32s(&table_affine))?;
    let out_buffer = ctx.alloc_u32s(n * G2_JAC_U32S)?;
    testing::note_copied_buffers(
        u64::from(scalars_buffer.was_copied()) + u64::from(table_buffer.was_copied()),
    );

    let params = [
        u32::try_from(n)
            .map_err(|_| MetalError::Execution("scalar vector length overflows u32".to_owned()))?,
        u32::try_from(windows).map_err(|_| MetalError::Execution("window count".to_owned()))?,
    ];
    ctx.run_once(
        KernelId::G2FixedBaseTable,
        &params,
        &[&scalars_buffer, &table_buffer, &out_buffer],
        n,
    )?;
    testing::note_device_round();

    let mut jac = vec![0u32; n * G2_JAC_U32S];
    out_buffer.copy_to_u32s(&mut jac);
    Ok(jac
        .chunks_exact(G2_JAC_U32S)
        .map(g2_jac_from_device_limbs)
        .collect())
}

/// One `jk_g2_fixed_base_table` (default) or `jk_g2_fixed_base_mul`
/// dispatch: `out[i] = base·scalars[i]`. An identity base short-circuits to
/// identities without dispatching (neither kernel's base slot has a
/// sentinel check).
pub fn g2_fixed_base_mul_device(
    ctx: &MetalContext,
    base: &G2Projective,
    scalars: &[ArkFr],
) -> Result<Vec<G2Projective>, MetalError> {
    let n = scalars.len();
    if n == 0 {
        return Ok(vec![]);
    }
    if base.is_zero() {
        return Ok(vec![G2Projective::zero(); n]);
    }
    let base_affine = base.into_affine();

    // Canonical limbs per thread plus the shared ladder start (max top bit).
    let mut scalar_limbs: Vec<u32> = Vec::with_capacity(n * FR_U32_LIMBS);
    let mut start_bit = 0u32;
    for scalar in scalars {
        let (limbs, bit) = scalar_limbs_and_start_bit(scalar);
        scalar_limbs.extend_from_slice(&limbs);
        start_bit = start_bit.max(bit);
    }

    if fixed_base_table_enabled() {
        return g2_fixed_base_table_device(ctx, &base_affine, &scalar_limbs, start_bit, n);
    }

    let scalars_buffer = ctx.wrap_slice(&scalar_limbs)?;
    let out_buffer = ctx.alloc_u32s(n * G2_JAC_U32S)?;
    testing::note_copied_buffers(u64::from(scalars_buffer.was_copied()));

    let mut params = [0u32; 2 + 4 * FR_U32_LIMBS];
    params[0] = u32::try_from(n)
        .map_err(|_| MetalError::Execution("scalar vector length overflows u32".to_owned()))?;
    params[1] = start_bit;
    params[2..]
        .copy_from_slice(&g2_bases_as_u32s(std::slice::from_ref(&base_affine))[..4 * FR_U32_LIMBS]);
    ctx.run_once(
        KernelId::G2FixedBaseMul,
        &params,
        &[&scalars_buffer, &out_buffer],
        n,
    )?;
    testing::note_device_round();

    let mut jac = vec![0u32; n * G2_JAC_U32S];
    out_buffer.copy_to_u32s(&mut jac);
    Ok(jac
        .chunks_exact(G2_JAC_U32S)
        .map(g2_jac_from_device_limbs)
        .collect())
}

/// Gate scaling, calibrated to the MEASURED crossovers (@2^22 in-proof
/// traces, M4), not to a per-point work model: the per-lane ladder is
/// latency-floored below a few thousand threads, so each kernel's
/// profitable region starts where its floor undercuts the CPU. Under the
/// 2^18 default these shifts land the crossovers at:
/// - G1 fold (254-bit ladder): len ≥ 1024 (3.2× at 8192, 1.7× at 1024).
/// - G2 fold (4-GLV ~64-iteration ladder): len ≥ 512 (1.5× at 8192 over
///   the plain-ladder device, 1.6× at 2048, floor ~7 ms).
/// - G2 fixed-base (plain 254-bit ladder, ~21 ms floor): len ≥ 2048 —
///   the device lost to the CPU at 512 pre-GLV, and this kernel still
///   walks the full scalar width.
const G1_WORK_PER_POINT_LOG2: usize = 8;
const G2_FOLD_WORK_LOG2: usize = 9;
const G2_FIXED_BASE_WORK_LOG2: usize = 7;

/// The `RoutineHooks::g1_scalar_mul_add` candidate: `Some(out)` when the
/// device served the call, `None` (undersized, dead device, or failed) for
/// the CPU path.
pub(super) fn g1_scalar_mul_add_hook(
    ps: &[G1Projective],
    qs: &[G1Projective],
    scalar: &ArkFr,
) -> Option<Vec<G1Projective>> {
    if !metal_gate("dory_fold_g1", ps.len() << G1_WORK_PER_POINT_LOG2) {
        return None;
    }
    let context = MetalContext::global().ok()?;
    match g1_scalar_mul_add_device(context, ps, qs, scalar) {
        Ok(out) => Some(out),
        Err(error) => {
            tracing::warn!(slot = "dory_fold_g1", %error, "device G1 fold failed; CPU fallback");
            None
        }
    }
}

/// The `RoutineHooks::g2_scalar_mul_add` candidate.
pub(super) fn g2_scalar_mul_add_hook(
    ps: &[G2Projective],
    qs: &[G2Projective],
    scalar: &ArkFr,
) -> Option<Vec<G2Projective>> {
    if !metal_gate("dory_fold_g2", ps.len() << G2_FOLD_WORK_LOG2) {
        return None;
    }
    let context = MetalContext::global().ok()?;
    match g2_scalar_mul_add_device(context, ps, qs, scalar) {
        Ok(out) => Some(out),
        Err(error) => {
            tracing::warn!(slot = "dory_fold_g2", %error, "device G2 fold failed; CPU fallback");
            None
        }
    }
}

/// The `RoutineHooks::g2_fixed_base_mul` candidate.
pub(super) fn g2_fixed_base_mul_hook(
    base: &G2Projective,
    scalars: &[ArkFr],
) -> Option<Vec<G2Projective>> {
    if !metal_gate("dory_fixed_base", scalars.len() << G2_FIXED_BASE_WORK_LOG2) {
        return None;
    }
    let context = MetalContext::global().ok()?;
    match g2_fixed_base_mul_device(context, base, scalars) {
        Ok(out) => Some(out),
        Err(error) => {
            tracing::warn!(
                slot = "dory_fixed_base",
                %error,
                "device G2 fixed-base sweep failed; CPU fallback"
            );
            None
        }
    }
}

/// The `RoutineHooks::g1_msm` candidate (the VMV preamble's host MSMs).
pub(super) fn g1_msm_hook(bases: &[G1Projective], scalars: &[ArkFr]) -> Option<G1Projective> {
    // SAFETY: dory's ArkFr is repr(transparent) over ark_bn254::Fr (the
    // layout fact every adapter in this module rests on).
    let scalars: &[dory::backends::arkworks::ArkFr] = unsafe {
        std::slice::from_raw_parts(
            scalars.as_ptr().cast::<dory::backends::arkworks::ArkFr>(),
            scalars.len(),
        )
    };
    super::dory_reduce::host_msm_g1(bases, scalars).map(|out| out.0)
}

/// The [`jolt_dory::RoutineHooks`] bundle the metal joint-opening slot
/// installs for the proof's Dory reduce rounds.
pub(super) fn routine_hooks() -> jolt_dory::RoutineHooks {
    jolt_dory::RoutineHooks {
        g1_scalar_mul_add: g1_scalar_mul_add_hook,
        g2_scalar_mul_add: g2_scalar_mul_add_hook,
        g2_fixed_base_mul: g2_fixed_base_mul_hook,
        g1_msm: g1_msm_hook,
    }
}

/// Group-level parity against arkworks on random, identity-planted, and
/// adversarial (equal / inverse at the final add) inputs.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use ark_ff::UniformRand;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::super::testing::{device_probe_count, gpu_lock};
    use super::*;

    fn ctx() -> &'static MetalContext {
        MetalContext::global().unwrap()
    }

    /// Random points with Z ≠ 1 (sum of two randoms) so normalization is
    /// exercised, identities planted every fifth slot.
    fn random_points(n: usize, seed: u64) -> Vec<G1Projective> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..n)
            .map(|i| {
                if i % 5 == 2 {
                    G1Projective::zero()
                } else {
                    G1Projective::rand(&mut rng) + G1Projective::rand(&mut rng)
                }
            })
            .collect()
    }

    fn host_mul_add(ps: &[G1Projective], qs: &[G1Projective], s: &ArkFr) -> Vec<G1Projective> {
        ps.iter().zip(qs).map(|(p, q)| *p * s + q).collect()
    }

    #[test]
    fn g1_scalar_mul_add_matches_arkworks() {
        let _lock = gpu_lock();
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let ps = random_points(67, 2);
        let qs = random_points(67, 3);

        for scalar in [
            ArkFr::rand(&mut rng),
            ArkFr::from(0u64),
            ArkFr::from(1u64),
            ArkFr::from(3u64),
            -ArkFr::from(1u64), // top-heavy canonical limbs: full 254-bit sweep
        ] {
            let probes_before = device_probe_count();
            let device = g1_scalar_mul_add_device(ctx(), &ps, &qs, &scalar).unwrap();
            assert_eq!(
                device_probe_count() - probes_before,
                1,
                "one dispatch per call"
            );
            assert_eq!(device, host_mul_add(&ps, &qs, &scalar), "scalar {scalar}");
        }
    }

    /// The final mixed add's special cases, unreachable through random data:
    /// Q = s·P (equal → doubling), Q = −s·P (inverse → identity), plus both
    /// vectors identity at a slot.
    #[test]
    fn g1_scalar_mul_add_edge_cases() {
        let _lock = gpu_lock();
        let mut rng = ChaCha20Rng::seed_from_u64(4);
        let scalar = ArkFr::rand(&mut rng);
        let p = G1Projective::rand(&mut rng);
        let ps = vec![p, p, G1Projective::zero(), p];
        let qs = vec![p * scalar, -(p * scalar), G1Projective::zero(), p];

        let device = g1_scalar_mul_add_device(ctx(), &ps, &qs, &scalar).unwrap();
        assert_eq!(device, host_mul_add(&ps, &qs, &scalar));
        assert!(device[1].is_zero(), "inverse add must yield identity");
        assert!(
            device[2].is_zero(),
            "identity ⊕ identity must stay identity"
        );
    }

    /// Random G2 points with Z ≠ 1, identities planted every fifth slot.
    fn random_g2_points(n: usize, seed: u64) -> Vec<G2Projective> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..n)
            .map(|i| {
                if i % 5 == 2 {
                    G2Projective::zero()
                } else {
                    G2Projective::rand(&mut rng) + G2Projective::rand(&mut rng)
                }
            })
            .collect()
    }

    fn host_g2_mul_add(ps: &[G2Projective], qs: &[G2Projective], s: &ArkFr) -> Vec<G2Projective> {
        ps.iter().zip(qs).map(|(p, q)| *p * s + q).collect()
    }

    #[test]
    fn g2_scalar_mul_add_matches_arkworks() {
        let _lock = gpu_lock();
        let mut rng = ChaCha20Rng::seed_from_u64(5);
        let ps = random_g2_points(53, 6);
        let qs = random_g2_points(53, 7);

        for scalar in [
            ArkFr::rand(&mut rng),
            ArkFr::from(0u64),
            ArkFr::from(1u64),
            ArkFr::from(3u64),
            -ArkFr::from(1u64), // top-heavy canonical limbs: full 254-bit sweep
        ] {
            let probes_before = device_probe_count();
            let device = g2_scalar_mul_add_device(ctx(), &ps, &qs, &scalar).unwrap();
            assert_eq!(
                device_probe_count() - probes_before,
                1,
                "one dispatch per call"
            );
            assert_eq!(
                device,
                host_g2_mul_add(&ps, &qs, &scalar),
                "scalar {scalar}"
            );
        }
    }

    /// G2 doubling / inverse / identity paths at the final mixed add.
    #[test]
    fn g2_scalar_mul_add_edge_cases() {
        let _lock = gpu_lock();
        let mut rng = ChaCha20Rng::seed_from_u64(8);
        let scalar = ArkFr::rand(&mut rng);
        let p = G2Projective::rand(&mut rng);
        let ps = vec![p, p, G2Projective::zero(), p];
        let qs = vec![p * scalar, -(p * scalar), G2Projective::zero(), p];

        let device = g2_scalar_mul_add_device(ctx(), &ps, &qs, &scalar).unwrap();
        assert_eq!(device, host_g2_mul_add(&ps, &qs, &scalar));
        assert!(device[1].is_zero(), "inverse add must yield identity");
        assert!(
            device[2].is_zero(),
            "identity ⊕ identity must stay identity"
        );
    }

    /// The full seam through jolt-dory's routines: installed hooks serve the
    /// fold calls on the device (probe-verified — a silent CPU fallback
    /// fails), results match the CPU routines, and the dropped guard
    /// uninstalls. nextest's process-per-test isolates the env + global.
    #[test]
    fn routine_hooks_serve_dory_folds_on_device() {
        use dory::backends::arkworks::{ArkFr as DoryFr, ArkG1, ArkG2};
        use dory::primitives::arithmetic::DoryRoutines;
        use jolt_dory::install_routine_hooks;

        let _lock = gpu_lock();
        // nextest runs one process per test, so env mutation is safe.
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS", "0");

        let mut rng = ChaCha20Rng::seed_from_u64(10);
        let scalar = DoryFr(ArkFr::rand(&mut rng));
        let bases_g1: Vec<ArkG1> = (0..33)
            .map(|_| ArkG1(G1Projective::rand(&mut rng)))
            .collect();
        let vs_g1: Vec<ArkG1> = (0..33)
            .map(|_| ArkG1(G1Projective::rand(&mut rng)))
            .collect();
        let bases_g2: Vec<ArkG2> = (0..33)
            .map(|_| ArkG2(G2Projective::rand(&mut rng)))
            .collect();
        let vs_g2: Vec<ArkG2> = (0..33)
            .map(|_| ArkG2(G2Projective::rand(&mut rng)))
            .collect();
        let scalars: Vec<DoryFr> = (0..33).map(|_| DoryFr(ArkFr::rand(&mut rng))).collect();

        // CPU references, computed unhooked.
        let mut cpu_g1 = vs_g1.clone();
        jolt_dory::JoltG1Routines::fixed_scalar_mul_bases_then_add(&bases_g1, &mut cpu_g1, &scalar);
        let mut cpu_g2 = vs_g2.clone();
        jolt_dory::JoltG2Routines::fixed_scalar_mul_bases_then_add(&bases_g2, &mut cpu_g2, &scalar);
        let cpu_fixed =
            jolt_dory::JoltG2Routines::fixed_base_vector_scalar_mul(&bases_g2[0], &scalars);

        let guard = install_routine_hooks(super::routine_hooks());
        let probes_before = device_probe_count();
        let mut hooked_g1 = vs_g1.clone();
        jolt_dory::JoltG1Routines::fixed_scalar_mul_bases_then_add(
            &bases_g1,
            &mut hooked_g1,
            &scalar,
        );
        let mut hooked_g2 = vs_g2.clone();
        jolt_dory::JoltG2Routines::fixed_scalar_mul_bases_then_add(
            &bases_g2,
            &mut hooked_g2,
            &scalar,
        );
        let hooked_fixed =
            jolt_dory::JoltG2Routines::fixed_base_vector_scalar_mul(&bases_g2[0], &scalars);
        assert_eq!(
            device_probe_count() - probes_before,
            3,
            "each hooked op must run as one device dispatch"
        );
        assert_eq!(hooked_g1, cpu_g1);
        assert_eq!(hooked_g2, cpu_g2);
        assert_eq!(hooked_fixed, cpu_fixed);
        drop(guard);

        let probes_after = device_probe_count();
        let mut unhooked_g1 = vs_g1.clone();
        jolt_dory::JoltG1Routines::fixed_scalar_mul_bases_then_add(
            &bases_g1,
            &mut unhooked_g1,
            &scalar,
        );
        assert_eq!(
            device_probe_count(),
            probes_after,
            "a dropped guard must uninstall the hooks"
        );
        assert_eq!(unhooked_g1, cpu_g1);
    }

    /// Fixed-base sweep against arkworks, both kernel arms (window table and
    /// plain ladder): random scalars plus the degenerate set (0, 1, −1), an
    /// identity base, and a Z ≠ 1 base.
    #[test]
    fn g2_fixed_base_mul_matches_arkworks() {
        let _lock = gpu_lock();
        let mut rng = ChaCha20Rng::seed_from_u64(9);
        let base = G2Projective::rand(&mut rng) + G2Projective::rand(&mut rng);
        let mut scalars: Vec<ArkFr> = (0..29).map(|_| ArkFr::rand(&mut rng)).collect();
        scalars[3] = ArkFr::from(0u64);
        scalars[11] = ArkFr::from(1u64);
        scalars[17] = -ArkFr::from(1u64);
        let expected: Vec<G2Projective> = scalars.iter().map(|s| base * s).collect();

        for (env, label) in [("1", "table"), ("0", "ladder")] {
            std::env::set_var("JOLT_DORY_FIXED_BASE_TABLE", env);
            let probes_before = device_probe_count();
            let device = g2_fixed_base_mul_device(ctx(), &base, &scalars).unwrap();
            assert_eq!(
                device_probe_count() - probes_before,
                1,
                "one dispatch per call ({label})"
            );
            assert_eq!(device, expected, "{label} arm diverged");
            assert!(device[3].is_zero(), "zero scalar must yield identity");

            let identity_out =
                g2_fixed_base_mul_device(ctx(), &G2Projective::zero(), &scalars).unwrap();
            assert!(identity_out.iter().all(|p| p.is_zero()));
        }
        std::env::remove_var("JOLT_DORY_FIXED_BASE_TABLE");
    }
}
