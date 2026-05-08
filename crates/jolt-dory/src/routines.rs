use ark_bn254::{G1Projective, G2Projective};
use ark_ec::{CurveGroup, VariableBaseMSM};
use dory::primitives::arithmetic::DoryRoutines;
use jolt_crypto::{Bn254G1, Bn254G2};
use jolt_field::Fr;
use rayon::prelude::*;

use crate::scheme::{ark_to_jolt_fr, ArkFr, ArkG1, ArkG2};

pub struct JoltG1Routines;

impl DoryRoutines<ArkG1> for JoltG1Routines {
    fn msm(bases: &[ArkG1], scalars: &[ArkFr]) -> ArkG1 {
        let _span = tracing::info_span!(
            "DoryRoutines::G1::msm",
            len = bases.len(),
            scalars = scalars.len()
        )
        .entered();
        assert_eq!(
            bases.len(),
            scalars.len(),
            "MSM requires equal length vectors"
        );
        if bases.is_empty() {
            return ArkG1::default();
        }

        let projective_points = ark_g1_slice_as_projective(bases);
        let affines = G1Projective::normalize_batch(projective_points);
        let raw_scalars = ark_fr_slice_as_ark(scalars);
        match G1Projective::msm_serial(&affines, raw_scalars) {
            Ok(result) => dory::backends::arkworks::ArkG1(result),
            Err(_) => std::process::abort(),
        }
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG1, scalars: &[ArkFr]) -> Vec<ArkG1> {
        let _span = tracing::info_span!(
            "DoryRoutines::G1::fixed_base_vector_scalar_mul",
            len = scalars.len()
        )
        .entered();
        let scalars = ark_fr_slice_as_jolt(scalars);
        let results =
            jolt_crypto::ec::bn254::glv::fixed_base_vector_msm_g1(ark_g1_as_jolt(base), scalars);
        jolt_g1_vec_into_ark(results)
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG1], vs: &mut [ArkG1], scalar: &ArkFr) {
        let _span = tracing::info_span!(
            "DoryRoutines::G1::fixed_scalar_mul_bases_then_add",
            len = bases.len(),
            vs = vs.len()
        )
        .entered();
        let bases = ark_g1_slice_as_jolt(bases);
        let vs = ark_g1_slice_as_jolt_mut(vs);
        jolt_crypto::ec::bn254::glv::vector_add_scalar_mul_g1(vs, bases, ark_to_jolt_fr(scalar));
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG1], addends: &[ArkG1], scalar: &ArkFr) {
        let _span = tracing::info_span!(
            "DoryRoutines::G1::fixed_scalar_mul_vs_then_add",
            len = addends.len(),
            vs = vs.len()
        )
        .entered();
        let vs = ark_g1_slice_as_jolt_mut(vs);
        let addends = ark_g1_slice_as_jolt(addends);
        jolt_crypto::ec::bn254::glv::vector_scalar_mul_add_gamma_g1(
            vs,
            ark_to_jolt_fr(scalar),
            addends,
        );
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        let _span = tracing::info_span!(
            "DoryRoutines::G1::fold_field_vectors",
            len = left.len(),
            right = right.len()
        )
        .entered();
        fold_field_vectors(left, right, scalar);
    }
}

pub struct JoltG2Routines;

impl DoryRoutines<ArkG2> for JoltG2Routines {
    fn msm(bases: &[ArkG2], scalars: &[ArkFr]) -> ArkG2 {
        let _span = tracing::info_span!(
            "DoryRoutines::G2::msm",
            len = bases.len(),
            scalars = scalars.len()
        )
        .entered();
        assert_eq!(
            bases.len(),
            scalars.len(),
            "MSM requires equal length vectors"
        );
        if bases.is_empty() {
            return ArkG2::default();
        }

        let projective_points = ark_g2_slice_as_projective(bases);
        let affines = G2Projective::normalize_batch(projective_points);
        let raw_scalars = ark_fr_slice_as_ark(scalars);
        match G2Projective::msm_serial(&affines, raw_scalars) {
            Ok(result) => dory::backends::arkworks::ArkG2(result),
            Err(_) => std::process::abort(),
        }
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG2, scalars: &[ArkFr]) -> Vec<ArkG2> {
        let _span = tracing::info_span!(
            "DoryRoutines::G2::fixed_base_vector_scalar_mul",
            len = scalars.len()
        )
        .entered();
        let base = *ark_g2_as_projective(base);
        let scalars = ark_fr_slice_as_ark(scalars);
        let results: Vec<G2Projective> = scalars
            .par_iter()
            .map(|&scalar| {
                jolt_crypto::ec::bn254::glv::glv_four::glv_four_scalar_mul_online(scalar, &[base])
                    [0]
            })
            .collect();
        ark_g2_vec_into_ark(results)
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG2], vs: &mut [ArkG2], scalar: &ArkFr) {
        let _span = tracing::info_span!(
            "DoryRoutines::G2::fixed_scalar_mul_bases_then_add",
            len = bases.len(),
            vs = vs.len()
        )
        .entered();
        let bases = ark_g2_slice_as_jolt(bases);
        let vs = ark_g2_slice_as_jolt_mut(vs);
        jolt_crypto::ec::bn254::glv::vector_add_scalar_mul_g2(vs, bases, ark_to_jolt_fr(scalar));
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG2], addends: &[ArkG2], scalar: &ArkFr) {
        let _span = tracing::info_span!(
            "DoryRoutines::G2::fixed_scalar_mul_vs_then_add",
            len = addends.len(),
            vs = vs.len()
        )
        .entered();
        let vs = ark_g2_slice_as_jolt_mut(vs);
        let addends = ark_g2_slice_as_jolt(addends);
        jolt_crypto::ec::bn254::glv::vector_scalar_mul_add_gamma_g2(
            vs,
            ark_to_jolt_fr(scalar),
            addends,
        );
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        let _span = tracing::info_span!(
            "DoryRoutines::G2::fold_field_vectors",
            len = left.len(),
            right = right.len()
        )
        .entered();
        fold_field_vectors(left, right, scalar);
    }
}

fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
    assert_eq!(left.len(), right.len(), "Lengths must match");
    let scalar = ark_to_jolt_fr(scalar);
    let left = ark_fr_slice_as_jolt_mut(left);
    let right = ark_fr_slice_as_jolt(right);
    left.par_iter_mut()
        .zip(right.par_iter())
        .for_each(|(left, right)| {
            *left = *left * scalar + *right;
        });
}

fn ark_fr_slice_as_jolt(slice: &[ArkFr]) -> &[Fr] {
    // SAFETY: ArkFr and Fr are transparent wrappers over ark_bn254::Fr.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<Fr>(), slice.len()) }
}

fn ark_fr_slice_as_ark(slice: &[ArkFr]) -> &[ark_bn254::Fr] {
    // SAFETY: ArkFr is a transparent wrapper over ark_bn254::Fr.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<ark_bn254::Fr>(), slice.len()) }
}

fn ark_fr_slice_as_jolt_mut(slice: &mut [ArkFr]) -> &mut [Fr] {
    // SAFETY: ArkFr and Fr are transparent wrappers over ark_bn254::Fr.
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<Fr>(), slice.len()) }
}

fn ark_g1_as_jolt(value: &ArkG1) -> &Bn254G1 {
    // SAFETY: ArkG1 and Bn254G1 are both transparent wrappers over ark_bn254::G1Projective.
    unsafe { &*(std::ptr::from_ref(value).cast::<Bn254G1>()) }
}

fn ark_g1_slice_as_jolt(slice: &[ArkG1]) -> &[Bn254G1] {
    // SAFETY: ArkG1 and Bn254G1 are both transparent wrappers over ark_bn254::G1Projective.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<Bn254G1>(), slice.len()) }
}

fn ark_g1_slice_as_jolt_mut(slice: &mut [ArkG1]) -> &mut [Bn254G1] {
    // SAFETY: ArkG1 and Bn254G1 are both transparent wrappers over ark_bn254::G1Projective.
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<Bn254G1>(), slice.len()) }
}

fn ark_g1_slice_as_projective(slice: &[ArkG1]) -> &[G1Projective] {
    // SAFETY: ArkG1 is a transparent wrapper over G1Projective.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<G1Projective>(), slice.len()) }
}

fn ark_g2_slice_as_jolt(slice: &[ArkG2]) -> &[Bn254G2] {
    // SAFETY: ArkG2 and Bn254G2 are both transparent wrappers over ark_bn254::G2Projective.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<Bn254G2>(), slice.len()) }
}

fn ark_g2_slice_as_projective(slice: &[ArkG2]) -> &[G2Projective] {
    // SAFETY: ArkG2 is a transparent wrapper over G2Projective.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<G2Projective>(), slice.len()) }
}

fn ark_g2_as_projective(value: &ArkG2) -> &G2Projective {
    // SAFETY: ArkG2 is a transparent wrapper over ark_bn254::G2Projective.
    unsafe { &*(std::ptr::from_ref(value).cast::<G2Projective>()) }
}

fn ark_g2_slice_as_jolt_mut(slice: &mut [ArkG2]) -> &mut [Bn254G2] {
    // SAFETY: ArkG2 and Bn254G2 are both transparent wrappers over ark_bn254::G2Projective.
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<Bn254G2>(), slice.len()) }
}

fn jolt_g1_vec_into_ark(values: Vec<Bn254G1>) -> Vec<ArkG1> {
    // SAFETY: ArkG1 and Bn254G1 have identical size/alignment and transparent layout.
    unsafe { std::mem::transmute(values) }
}

fn ark_g2_vec_into_ark(values: Vec<G2Projective>) -> Vec<ArkG2> {
    // SAFETY: ArkG2 is a transparent wrapper over ark_bn254::G2Projective.
    unsafe { std::mem::transmute(values) }
}
