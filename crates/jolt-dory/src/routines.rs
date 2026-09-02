//! Jolt-optimized [`DoryRoutines`] plugged into `dory::prove`/`dory::verify`.
//!
//! As of dory-pcs 0.4.1 the stock routines batch-normalize MSM bases and
//! parallelize the vector ops behind the `parallel` feature (upstreamed from
//! here in a16z/dory#27), so `msm` and `fold_field_vectors` just delegate —
//! those wrappers only contribute the `JoltG1Routines::msm`/
//! `JoltG2Routines::msm` span labels the profiling telemetry grammar
//! addresses. The remaining overrides are the GLV kernels from
//! `jolt-optimizations` (2D decomposition for G1, 4D Frobenius for G2),
//! which replace the stock full-width scalar multiplications per element and
//! have no crates.io home yet. These mirror the legacy prover's
//! `JoltG1Routines`/`JoltG2Routines`. Group results are exact, so proofs are
//! byte-identical to the stock routines'.

use ark_bn254::{Fr as ArkworksFr, G1Projective, G2Projective};
use dory::backends::arkworks::{ArkFr, ArkG1, ArkG2, G1Routines, G2Routines};
use dory::primitives::arithmetic::DoryRoutines;
use rayon::prelude::*;

// The transmutes below rely on ArkFr/ArkG1/ArkG2 being repr(transparent)
// over ark_bn254::{Fr, G1Projective, G2Projective} (the same layout facts
// `crate::scheme`'s conversions rest on).

#[inline]
fn ark_fr_slice(scalars: &[ArkFr]) -> &[ArkworksFr] {
    // SAFETY: ArkFr is repr(transparent) over ark_bn254::Fr.
    unsafe { std::slice::from_raw_parts(scalars.as_ptr().cast::<ArkworksFr>(), scalars.len()) }
}

#[inline]
fn g1_slice(points: &[ArkG1]) -> &[G1Projective] {
    // SAFETY: ArkG1 is repr(transparent) over G1Projective.
    unsafe { std::slice::from_raw_parts(points.as_ptr().cast::<G1Projective>(), points.len()) }
}

#[inline]
fn g1_slice_mut(points: &mut [ArkG1]) -> &mut [G1Projective] {
    // SAFETY: ArkG1 is repr(transparent) over G1Projective.
    unsafe {
        std::slice::from_raw_parts_mut(points.as_mut_ptr().cast::<G1Projective>(), points.len())
    }
}

#[inline]
fn g2_slice(points: &[ArkG2]) -> &[G2Projective] {
    // SAFETY: ArkG2 is repr(transparent) over G2Projective.
    unsafe { std::slice::from_raw_parts(points.as_ptr().cast::<G2Projective>(), points.len()) }
}

#[inline]
fn g2_slice_mut(points: &mut [ArkG2]) -> &mut [G2Projective] {
    // SAFETY: ArkG2 is repr(transparent) over G2Projective.
    unsafe {
        std::slice::from_raw_parts_mut(points.as_mut_ptr().cast::<G2Projective>(), points.len())
    }
}

pub struct JoltG1Routines;

impl JoltG1Routines {
    pub(crate) fn host_fixed_scalar_mul_bases_then_add(
        bases: &[ArkG1],
        vs: &mut [ArkG1],
        scalar: &ArkFr,
    ) {
        jolt_optimizations::vector_add_scalar_mul_g1_online(
            g1_slice_mut(vs),
            g1_slice(bases),
            scalar.0,
        );
    }

    pub(crate) fn host_fixed_scalar_mul_vs_then_add(
        vs: &mut [ArkG1],
        addends: &[ArkG1],
        scalar: &ArkFr,
    ) {
        jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(
            g1_slice_mut(vs),
            scalar.0,
            g1_slice(addends),
        );
    }
}

impl DoryRoutines<ArkG1> for JoltG1Routines {
    #[tracing::instrument(skip_all, name = "JoltG1Routines::msm", fields(len = bases.len()))]
    fn msm(bases: &[ArkG1], scalars: &[ArkFr]) -> ArkG1 {
        if let Some(hooks) = crate::routines_hook::routine_hooks() {
            if let Some(out) = (hooks.g1_msm)(g1_slice(bases), ark_fr_slice(scalars)) {
                return ArkG1(out);
            }
        }
        G1Routines::msm(bases, scalars)
    }

    #[tracing::instrument(skip_all, name = "JoltG1Routines::fixed_base_vector_scalar_mul", fields(len = scalars.len()))]
    fn fixed_base_vector_scalar_mul(base: &ArkG1, scalars: &[ArkFr]) -> Vec<ArkG1> {
        if scalars.is_empty() {
            return vec![];
        }
        let results = jolt_optimizations::fixed_base_vector_msm_g1(&base.0, ark_fr_slice(scalars));
        results.into_iter().map(ArkG1).collect()
    }

    #[tracing::instrument(skip_all, name = "JoltG1Routines::fixed_scalar_mul_bases_then_add", fields(len = vs.len()))]
    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG1], vs: &mut [ArkG1], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "lengths must match");
        // v[i] = v[i] + scalar * bases[i]
        if let Some(hooks) = crate::routines_hook::routine_hooks() {
            if let Some(out) = (hooks.g1_scalar_mul_add)(g1_slice(bases), g1_slice(vs), &scalar.0) {
                g1_slice_mut(vs).copy_from_slice(&out);
                return;
            }
        }
        Self::host_fixed_scalar_mul_bases_then_add(bases, vs, scalar);
    }

    #[tracing::instrument(skip_all, name = "JoltG1Routines::fixed_scalar_mul_vs_then_add", fields(len = vs.len()))]
    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG1], addends: &[ArkG1], scalar: &ArkFr) {
        assert_eq!(vs.len(), addends.len(), "lengths must match");
        // v[i] = scalar * v[i] + addends[i]
        if let Some(hooks) = crate::routines_hook::routine_hooks() {
            if let Some(out) = (hooks.g1_scalar_mul_add)(g1_slice(vs), g1_slice(addends), &scalar.0)
            {
                g1_slice_mut(vs).copy_from_slice(&out);
                return;
            }
        }
        Self::host_fixed_scalar_mul_vs_then_add(vs, addends, scalar);
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        <G1Routines as DoryRoutines<ArkG1>>::fold_field_vectors(left, right, scalar);
    }
}

pub struct JoltG2Routines;

impl JoltG2Routines {
    pub(crate) fn host_fixed_scalar_mul_bases_then_add(
        bases: &[ArkG2],
        vs: &mut [ArkG2],
        scalar: &ArkFr,
    ) {
        jolt_optimizations::vector_add_scalar_mul_g2_online(
            g2_slice_mut(vs),
            g2_slice(bases),
            scalar.0,
        );
    }

    pub(crate) fn host_fixed_scalar_mul_vs_then_add(
        vs: &mut [ArkG2],
        addends: &[ArkG2],
        scalar: &ArkFr,
    ) {
        jolt_optimizations::vector_scalar_mul_add_gamma_g2_online(
            g2_slice_mut(vs),
            scalar.0,
            g2_slice(addends),
        );
    }
}

impl DoryRoutines<ArkG2> for JoltG2Routines {
    #[tracing::instrument(skip_all, name = "JoltG2Routines::msm", fields(len = bases.len()))]
    fn msm(bases: &[ArkG2], scalars: &[ArkFr]) -> ArkG2 {
        G2Routines::msm(bases, scalars)
    }

    #[tracing::instrument(skip_all, name = "JoltG2Routines::fixed_base_vector_scalar_mul", fields(len = scalars.len()))]
    fn fixed_base_vector_scalar_mul(base: &ArkG2, scalars: &[ArkFr]) -> Vec<ArkG2> {
        if scalars.is_empty() {
            return vec![];
        }
        let base_proj = base.0;
        if let Some(hooks) = crate::routines_hook::routine_hooks() {
            if let Some(out) = (hooks.g2_fixed_base_mul)(&base_proj, ark_fr_slice(scalars)) {
                return out.into_iter().map(ArkG2).collect();
            }
        }
        scalars
            .par_iter()
            .map(|scalar| {
                #[expect(
                    clippy::indexing_slicing,
                    reason = "glv_four_scalar_mul_online returns one point per base and is passed exactly one base"
                )]
                ArkG2(jolt_optimizations::glv_four_scalar_mul_online(scalar.0, &[base_proj])[0])
            })
            .collect()
    }

    #[tracing::instrument(skip_all, name = "JoltG2Routines::fixed_scalar_mul_bases_then_add", fields(len = vs.len()))]
    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG2], vs: &mut [ArkG2], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "lengths must match");
        // v[i] = v[i] + scalar * bases[i]
        if let Some(hooks) = crate::routines_hook::routine_hooks() {
            if let Some(out) = (hooks.g2_scalar_mul_add)(g2_slice(bases), g2_slice(vs), &scalar.0) {
                g2_slice_mut(vs).copy_from_slice(&out);
                return;
            }
        }
        Self::host_fixed_scalar_mul_bases_then_add(bases, vs, scalar);
    }

    #[tracing::instrument(skip_all, name = "JoltG2Routines::fixed_scalar_mul_vs_then_add", fields(len = vs.len()))]
    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG2], addends: &[ArkG2], scalar: &ArkFr) {
        assert_eq!(vs.len(), addends.len(), "lengths must match");
        // v[i] = scalar * v[i] + addends[i]
        if let Some(hooks) = crate::routines_hook::routine_hooks() {
            if let Some(out) = (hooks.g2_scalar_mul_add)(g2_slice(vs), g2_slice(addends), &scalar.0)
            {
                g2_slice_mut(vs).copy_from_slice(&out);
                return;
            }
        }
        Self::host_fixed_scalar_mul_vs_then_add(vs, addends, scalar);
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        <G2Routines as DoryRoutines<ArkG2>>::fold_field_vectors(left, right, scalar);
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::indexing_slicing, reason = "tests index fixture data")]

    use super::*;
    use dory::primitives::arithmetic::{Field as DoryField, Group};

    fn random_fr() -> ArkFr {
        <ArkFr as DoryField>::random()
    }

    fn random_g1() -> ArkG1 {
        <ArkG1 as Group>::random()
    }

    fn random_g2() -> ArkG2 {
        <ArkG2 as Group>::random()
    }

    #[test]
    fn g1_routines_match_stock() {
        let mut bases: Vec<ArkG1> = (0..33).map(|_| random_g1()).collect();
        let mut scalars: Vec<ArkFr> = (0..33).map(|_| random_fr()).collect();
        let scalar = random_fr();
        // Identity points and zero scalars exercise the GLV decomposition
        // and batch-normalization edge cases the random fixtures miss.
        bases[5] = ArkG1::identity();
        scalars[9] = <ArkFr as DoryField>::zero();

        assert_eq!(
            JoltG1Routines::msm(&bases, &scalars),
            G1Routines::msm(&bases, &scalars)
        );
        assert_eq!(
            JoltG1Routines::fixed_base_vector_scalar_mul(&bases[0], &scalars),
            G1Routines::fixed_base_vector_scalar_mul(&bases[0], &scalars)
        );
        assert_eq!(
            JoltG1Routines::fixed_base_vector_scalar_mul(&ArkG1::identity(), &scalars),
            G1Routines::fixed_base_vector_scalar_mul(&ArkG1::identity(), &scalars)
        );

        let mut vs_jolt: Vec<ArkG1> = (0..33).map(|_| random_g1()).collect();
        vs_jolt[3] = ArkG1::identity();
        let mut vs_stock = vs_jolt.clone();
        JoltG1Routines::fixed_scalar_mul_bases_then_add(&bases, &mut vs_jolt, &scalar);
        G1Routines::fixed_scalar_mul_bases_then_add(&bases, &mut vs_stock, &scalar);
        assert_eq!(vs_jolt, vs_stock);

        JoltG1Routines::fixed_scalar_mul_vs_then_add(&mut vs_jolt, &bases, &scalar);
        G1Routines::fixed_scalar_mul_vs_then_add(&mut vs_stock, &bases, &scalar);
        assert_eq!(vs_jolt, vs_stock);

        let right: Vec<ArkFr> = (0..33).map(|_| random_fr()).collect();
        let mut left_jolt: Vec<ArkFr> = (0..33).map(|_| random_fr()).collect();
        let mut left_stock = left_jolt.clone();
        <JoltG1Routines as DoryRoutines<ArkG1>>::fold_field_vectors(
            &mut left_jolt,
            &right,
            &scalar,
        );
        <G1Routines as DoryRoutines<ArkG1>>::fold_field_vectors(&mut left_stock, &right, &scalar);
        assert_eq!(left_jolt, left_stock);
    }

    #[test]
    fn g2_routines_match_stock() {
        let mut bases: Vec<ArkG2> = (0..17).map(|_| random_g2()).collect();
        let mut scalars: Vec<ArkFr> = (0..17).map(|_| random_fr()).collect();
        let scalar = random_fr();
        // Identity points and zero scalars exercise the GLV decomposition
        // and batch-normalization edge cases the random fixtures miss.
        bases[5] = ArkG2::identity();
        scalars[9] = <ArkFr as DoryField>::zero();

        assert_eq!(
            JoltG2Routines::msm(&bases, &scalars),
            G2Routines::msm(&bases, &scalars)
        );
        assert_eq!(
            JoltG2Routines::fixed_base_vector_scalar_mul(&bases[0], &scalars),
            G2Routines::fixed_base_vector_scalar_mul(&bases[0], &scalars)
        );
        assert_eq!(
            JoltG2Routines::fixed_base_vector_scalar_mul(&ArkG2::identity(), &scalars),
            G2Routines::fixed_base_vector_scalar_mul(&ArkG2::identity(), &scalars)
        );

        let mut vs_jolt: Vec<ArkG2> = (0..17).map(|_| random_g2()).collect();
        vs_jolt[3] = ArkG2::identity();
        let mut vs_stock = vs_jolt.clone();
        JoltG2Routines::fixed_scalar_mul_bases_then_add(&bases, &mut vs_jolt, &scalar);
        G2Routines::fixed_scalar_mul_bases_then_add(&bases, &mut vs_stock, &scalar);
        assert_eq!(vs_jolt, vs_stock);

        JoltG2Routines::fixed_scalar_mul_vs_then_add(&mut vs_jolt, &bases, &scalar);
        G2Routines::fixed_scalar_mul_vs_then_add(&mut vs_stock, &bases, &scalar);
        assert_eq!(vs_jolt, vs_stock);
    }

    /// The routine-hooks seam: a served hook's results are used verbatim, a
    /// declining hook falls through to the CPU arithmetic, and a dropped
    /// guard uninstalls the seam. Hook fns compute the same group values as
    /// the CPU path (the seam's semantics contract) and bump a counter to
    /// prove which path ran. nextest's process-per-test isolates the global.
    #[test]
    fn routine_hooks_scope_and_fall_through() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use ark_bn254::Fr as ArkworksFr;

        use crate::routines_hook::{install_routine_hooks, RoutineHooks};

        static HOOK_CALLS: AtomicUsize = AtomicUsize::new(0);

        fn g1_serve(
            ps: &[G1Projective],
            qs: &[G1Projective],
            s: &ArkworksFr,
        ) -> Option<Vec<G1Projective>> {
            let _ = HOOK_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(ps.iter().zip(qs).map(|(p, q)| *p * s + q).collect())
        }
        fn g2_serve(
            ps: &[G2Projective],
            qs: &[G2Projective],
            s: &ArkworksFr,
        ) -> Option<Vec<G2Projective>> {
            let _ = HOOK_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(ps.iter().zip(qs).map(|(p, q)| *p * s + q).collect())
        }
        fn g2_fixed_serve(
            base: &G2Projective,
            scalars: &[ArkworksFr],
        ) -> Option<Vec<G2Projective>> {
            let _ = HOOK_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(scalars.iter().map(|s| *base * s).collect())
        }
        fn g1_decline(
            _: &[G1Projective],
            _: &[G1Projective],
            _: &ArkworksFr,
        ) -> Option<Vec<G1Projective>> {
            let _ = HOOK_CALLS.fetch_add(1, Ordering::SeqCst);
            None
        }
        fn g2_decline(
            _: &[G2Projective],
            _: &[G2Projective],
            _: &ArkworksFr,
        ) -> Option<Vec<G2Projective>> {
            let _ = HOOK_CALLS.fetch_add(1, Ordering::SeqCst);
            None
        }
        fn g2_fixed_decline(_: &G2Projective, _: &[ArkworksFr]) -> Option<Vec<G2Projective>> {
            let _ = HOOK_CALLS.fetch_add(1, Ordering::SeqCst);
            None
        }
        fn g1_msm_serve(bases: &[G1Projective], scalars: &[ArkworksFr]) -> Option<G1Projective> {
            let _ = HOOK_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(
                bases
                    .iter()
                    .zip(scalars)
                    .map(|(base, s)| *base * s)
                    .sum::<G1Projective>(),
            )
        }
        fn g1_msm_decline(_: &[G1Projective], _: &[ArkworksFr]) -> Option<G1Projective> {
            let _ = HOOK_CALLS.fetch_add(1, Ordering::SeqCst);
            None
        }

        let bases_g1: Vec<ArkG1> = (0..9).map(|_| random_g1()).collect();
        let bases_g2: Vec<ArkG2> = (0..9).map(|_| random_g2()).collect();
        let scalars: Vec<ArkFr> = (0..9).map(|_| random_fr()).collect();
        let scalar = random_fr();
        let vs_reference_g1: Vec<ArkG1> = (0..9).map(|_| random_g1()).collect();
        let vs_reference_g2: Vec<ArkG2> = (0..9).map(|_| random_g2()).collect();

        // Unhooked reference results.
        let mut expected_g1 = vs_reference_g1.clone();
        JoltG1Routines::fixed_scalar_mul_bases_then_add(&bases_g1, &mut expected_g1, &scalar);
        let mut expected_g2 = vs_reference_g2.clone();
        JoltG2Routines::fixed_scalar_mul_bases_then_add(&bases_g2, &mut expected_g2, &scalar);
        let expected_fixed = JoltG2Routines::fixed_base_vector_scalar_mul(&bases_g2[0], &scalars);
        let expected_msm = JoltG1Routines::msm(&bases_g1, &scalars);
        assert_eq!(HOOK_CALLS.load(Ordering::SeqCst), 0, "no hook installed");

        // Serving hooks: consulted once per call, results used.
        let guard = install_routine_hooks(RoutineHooks {
            g1_scalar_mul_add: g1_serve,
            g2_scalar_mul_add: g2_serve,
            g2_fixed_base_mul: g2_fixed_serve,
            g1_msm: g1_msm_serve,
        });
        let mut hooked_g1 = vs_reference_g1.clone();
        JoltG1Routines::fixed_scalar_mul_bases_then_add(&bases_g1, &mut hooked_g1, &scalar);
        let mut hooked_g2 = vs_reference_g2.clone();
        JoltG2Routines::fixed_scalar_mul_bases_then_add(&bases_g2, &mut hooked_g2, &scalar);
        let hooked_fixed = JoltG2Routines::fixed_base_vector_scalar_mul(&bases_g2[0], &scalars);
        let hooked_msm = JoltG1Routines::msm(&bases_g1, &scalars);
        assert_eq!(
            HOOK_CALLS.load(Ordering::SeqCst),
            4,
            "each op consults once"
        );
        assert_eq!(hooked_g1, expected_g1);
        assert_eq!(hooked_g2, expected_g2);
        assert_eq!(hooked_fixed, expected_fixed);
        assert_eq!(hooked_msm, expected_msm);
        drop(guard);

        // Declining hooks: consulted, then the CPU path answers.
        let guard = install_routine_hooks(RoutineHooks {
            g1_scalar_mul_add: g1_decline,
            g2_scalar_mul_add: g2_decline,
            g2_fixed_base_mul: g2_fixed_decline,
            g1_msm: g1_msm_decline,
        });
        let mut declined_g1 = vs_reference_g1.clone();
        JoltG1Routines::fixed_scalar_mul_vs_then_add(&mut declined_g1, &bases_g1, &scalar);
        let mut stock_g1 = vs_reference_g1.clone();
        G1Routines::fixed_scalar_mul_vs_then_add(&mut stock_g1, &bases_g1, &scalar);
        let declined_msm = JoltG1Routines::msm(&bases_g1, &scalars);
        assert_eq!(HOOK_CALLS.load(Ordering::SeqCst), 6);
        assert_eq!(declined_g1, stock_g1);
        assert_eq!(declined_msm, expected_msm);
        drop(guard);

        // Dropped guard: no consultation.
        let mut unhooked_g1 = vs_reference_g1.clone();
        JoltG1Routines::fixed_scalar_mul_bases_then_add(&bases_g1, &mut unhooked_g1, &scalar);
        assert_eq!(
            HOOK_CALLS.load(Ordering::SeqCst),
            6,
            "guard drop uninstalls"
        );
        assert_eq!(unhooked_g1, expected_g1);
    }
}
