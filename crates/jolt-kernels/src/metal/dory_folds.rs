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
use ark_ff::{PrimeField, Zero};

use super::field::FR_U32_LIMBS;
use super::g1::{bases_as_u32s, jac_from_device_limbs, JAC_U32S};
use super::g2::{g2_bases_as_u32s, g2_jac_from_device_limbs, G2_JAC_U32S};
use super::runtime::{KernelId, MetalContext};
use super::{testing, MetalError};

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

/// One `jk_g2_scalar_mul_add` dispatch: `out[i] = scalar·ps[i] + qs[i]`.
/// Synchronous; both vectors must have equal length.
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
    let (scalar_limbs, start_bit) = scalar_limbs_and_start_bit(scalar);

    let ps_affine = g2_affine_sentinels(ps);
    let qs_affine = g2_affine_sentinels(qs);
    let ps_buffer = ctx.wrap_slice(g2_bases_as_u32s(&ps_affine))?;
    let qs_buffer = ctx.wrap_slice(g2_bases_as_u32s(&qs_affine))?;
    let out_buffer = ctx.alloc_u32s(n * G2_JAC_U32S)?;
    testing::note_copied_buffers(
        u64::from(ps_buffer.was_copied()) + u64::from(qs_buffer.was_copied()),
    );

    let mut params = [0u32; 2 + FR_U32_LIMBS];
    params[0] = u32::try_from(n)
        .map_err(|_| MetalError::Execution("fold vector length overflows u32".to_owned()))?;
    params[1] = start_bit;
    params[2..].copy_from_slice(&scalar_limbs);
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

/// One `jk_g2_fixed_base_mul` dispatch: `out[i] = base·scalars[i]`.
/// An identity base short-circuits to identities without dispatching (the
/// kernel's base slot has no sentinel check).
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

    /// Fixed-base sweep against arkworks: random scalars plus the degenerate
    /// set (0, 1, −1), an identity base, and a Z ≠ 1 base.
    #[test]
    fn g2_fixed_base_mul_matches_arkworks() {
        let _lock = gpu_lock();
        let mut rng = ChaCha20Rng::seed_from_u64(9);
        let base = G2Projective::rand(&mut rng) + G2Projective::rand(&mut rng);
        let mut scalars: Vec<ArkFr> = (0..29).map(|_| ArkFr::rand(&mut rng)).collect();
        scalars[3] = ArkFr::from(0u64);
        scalars[11] = ArkFr::from(1u64);
        scalars[17] = -ArkFr::from(1u64);

        let probes_before = device_probe_count();
        let device = g2_fixed_base_mul_device(ctx(), &base, &scalars).unwrap();
        assert_eq!(
            device_probe_count() - probes_before,
            1,
            "one dispatch per call"
        );
        let expected: Vec<G2Projective> = scalars.iter().map(|s| base * s).collect();
        assert_eq!(device, expected);
        assert!(device[3].is_zero(), "zero scalar must yield identity");

        let identity_out =
            g2_fixed_base_mul_device(ctx(), &G2Projective::zero(), &scalars).unwrap();
        assert!(identity_out.iter().all(|p| p.is_zero()));
    }
}
