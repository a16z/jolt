//! Jolt-optimized dory-pcs group routines.
//!
//! These mirror the historical in-core Dory routines so extracting Dory into
//! this crate does not silently fall back to slower default dory-pcs group
//! operations in Stage 8.

#![expect(
    clippy::expect_used,
    reason = "DoryRoutines cannot return Result; MSM length mismatches are caller invariant violations"
)]

use std::mem::transmute_copy;
use std::slice::{from_raw_parts, from_raw_parts_mut};

use ark_bn254::{Fr as ArkworksFr, G1Projective, G2Projective};
use ark_ec::scalar_mul::variable_base::VariableBaseMSM;
use ark_ec::CurveGroup;
use dory::backends::arkworks::{ArkFr, ArkG1, ArkG2};
use dory::primitives::arithmetic::DoryRoutines;
use jolt_optimizations::{
    fixed_base_vector_msm_g1, glv_four_scalar_mul_online, vector_add_scalar_mul_g1_online,
    vector_add_scalar_mul_g2_online, vector_scalar_mul_add_gamma_g1_online,
    vector_scalar_mul_add_gamma_g2_online,
};
use rayon::prelude::*;

fn raw_scalars(scalars: &[ArkFr]) -> &[ArkworksFr] {
    // SAFETY: ArkFr is repr(transparent) over ark_bn254::Fr.
    unsafe { from_raw_parts(scalars.as_ptr().cast::<ArkworksFr>(), scalars.len()) }
}

fn raw_scalar(scalar: &ArkFr) -> ArkworksFr {
    // SAFETY: ArkFr is repr(transparent) over ark_bn254::Fr.
    unsafe { transmute_copy::<ArkFr, ArkworksFr>(scalar) }
}

fn g1_points(points: &[ArkG1]) -> &[G1Projective] {
    // SAFETY: ArkG1 is repr(transparent) over ark_bn254::G1Projective.
    unsafe { from_raw_parts(points.as_ptr().cast::<G1Projective>(), points.len()) }
}

fn g1_points_mut(points: &mut [ArkG1]) -> &mut [G1Projective] {
    // SAFETY: ArkG1 is repr(transparent) over ark_bn254::G1Projective.
    unsafe { from_raw_parts_mut(points.as_mut_ptr().cast::<G1Projective>(), points.len()) }
}

fn g2_points(points: &[ArkG2]) -> &[G2Projective] {
    // SAFETY: ArkG2 is repr(transparent) over ark_bn254::G2Projective.
    unsafe { from_raw_parts(points.as_ptr().cast::<G2Projective>(), points.len()) }
}

fn g2_points_mut(points: &mut [ArkG2]) -> &mut [G2Projective] {
    // SAFETY: ArkG2 is repr(transparent) over ark_bn254::G2Projective.
    unsafe { from_raw_parts_mut(points.as_mut_ptr().cast::<G2Projective>(), points.len()) }
}

fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
    assert_eq!(left.len(), right.len(), "Dory vector lengths must match");
    left.par_iter_mut()
        .zip(right.par_iter())
        .for_each(|(left, right)| {
            *left = *left * *scalar + *right;
        });
}

pub struct JoltG1Routines;

impl DoryRoutines<ArkG1> for JoltG1Routines {
    fn msm(bases: &[ArkG1], scalars: &[ArkFr]) -> ArkG1 {
        let affines = G1Projective::normalize_batch(g1_points(bases));
        let result = <G1Projective as VariableBaseMSM>::msm_serial(&affines, raw_scalars(scalars))
            .expect("Dory G1 MSM input lengths should match");
        ArkG1(result)
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG1, scalars: &[ArkFr]) -> Vec<ArkG1> {
        if scalars.is_empty() {
            return vec![];
        }
        fixed_base_vector_msm_g1(&base.0, raw_scalars(scalars))
            .into_iter()
            .map(ArkG1)
            .collect()
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG1], vs: &mut [ArkG1], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "Dory vector lengths must match");
        vector_add_scalar_mul_g1_online(g1_points_mut(vs), g1_points(bases), raw_scalar(scalar));
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG1], addends: &[ArkG1], scalar: &ArkFr) {
        assert_eq!(vs.len(), addends.len(), "Dory vector lengths must match");
        vector_scalar_mul_add_gamma_g1_online(
            g1_points_mut(vs),
            raw_scalar(scalar),
            g1_points(addends),
        );
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        fold_field_vectors(left, right, scalar);
    }
}

pub struct JoltG2Routines;

impl DoryRoutines<ArkG2> for JoltG2Routines {
    fn msm(bases: &[ArkG2], scalars: &[ArkFr]) -> ArkG2 {
        let affines = G2Projective::normalize_batch(g2_points(bases));
        let result = <G2Projective as VariableBaseMSM>::msm_serial(
            &affines[..scalars.len()],
            raw_scalars(scalars),
        )
        .expect("Dory G2 MSM input lengths should match");
        ArkG2(result)
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG2, scalars: &[ArkFr]) -> Vec<ArkG2> {
        if scalars.is_empty() {
            return vec![];
        }
        raw_scalars(scalars)
            .par_iter()
            .map(|&scalar| ArkG2(glv_four_scalar_mul_online(scalar, &[base.0])[0]))
            .collect()
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG2], vs: &mut [ArkG2], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "Dory vector lengths must match");
        vector_add_scalar_mul_g2_online(g2_points_mut(vs), g2_points(bases), raw_scalar(scalar));
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG2], addends: &[ArkG2], scalar: &ArkFr) {
        assert_eq!(vs.len(), addends.len(), "Dory vector lengths must match");
        vector_scalar_mul_add_gamma_g2_online(
            g2_points_mut(vs),
            raw_scalar(scalar),
            g2_points(addends),
        );
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        fold_field_vectors(left, right, scalar);
    }
}
