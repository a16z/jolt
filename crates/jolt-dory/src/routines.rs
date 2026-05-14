use dory::backends::arkworks::{ArkFr, ArkG1, ArkG2, G1Routines, G2Routines};
use dory::primitives::arithmetic::DoryRoutines;
use jolt_crypto::ec::bn254::glv::{dory_g1, dory_g2, glv_four, glv_two};
use rayon::prelude::*;

pub struct JoltG1Routines;

impl DoryRoutines<ArkG1> for JoltG1Routines {
    fn msm(bases: &[ArkG1], scalars: &[ArkFr]) -> ArkG1 {
        G1Routines::msm(bases, scalars)
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG1, scalars: &[ArkFr]) -> Vec<ArkG1> {
        if scalars.is_empty() {
            return Vec::new();
        }
        let raw_scalars = scalars.iter().map(|scalar| scalar.0).collect::<Vec<_>>();
        let results = glv_two::fixed_base_vector_msm_g1(&base.0, &raw_scalars);
        results.into_iter().map(ArkG1).collect()
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG1], vs: &mut [ArkG1], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "bases and vs must have same length");
        let bases = ark_g1_slice(bases);
        let vs = ark_g1_slice_mut(vs);
        dory_g1::vector_add_scalar_mul_g1_online(vs, bases, scalar.0);
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG1], addends: &[ArkG1], scalar: &ArkFr) {
        assert_eq!(
            vs.len(),
            addends.len(),
            "vs and addends must have same length"
        );
        let addends = ark_g1_slice(addends);
        let vs = ark_g1_slice_mut(vs);
        dory_g1::vector_scalar_mul_add_gamma_g1_online(vs, scalar.0, addends);
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        fold_field_vectors(left, right, scalar);
    }
}

pub struct JoltG2Routines;

impl DoryRoutines<ArkG2> for JoltG2Routines {
    fn msm(bases: &[ArkG2], scalars: &[ArkFr]) -> ArkG2 {
        G2Routines::msm(bases, scalars)
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG2, scalars: &[ArkFr]) -> Vec<ArkG2> {
        scalars
            .par_iter()
            .map(|scalar| ArkG2(glv_four::glv_four_scalar_mul_online(scalar.0, &[base.0])[0]))
            .collect()
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG2], vs: &mut [ArkG2], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "bases and vs must have same length");
        let bases = ark_g2_slice(bases);
        let vs = ark_g2_slice_mut(vs);
        dory_g2::vector_add_scalar_mul_g2_online(vs, bases, scalar.0);
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG2], addends: &[ArkG2], scalar: &ArkFr) {
        assert_eq!(
            vs.len(),
            addends.len(),
            "vs and addends must have same length"
        );
        let addends = ark_g2_slice(addends);
        let vs = ark_g2_slice_mut(vs);
        dory_g2::vector_scalar_mul_add_gamma_g2_online(vs, scalar.0, addends);
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        fold_field_vectors(left, right, scalar);
    }
}

fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
    assert_eq!(left.len(), right.len(), "field vector lengths must match");
    left.par_iter_mut()
        .zip(right.par_iter())
        .for_each(|(left, right)| {
            *left = *left * *scalar + *right;
        });
}

fn ark_g1_slice(values: &[ArkG1]) -> &[ark_bn254::G1Projective] {
    // SAFETY: ArkG1 is repr(transparent) over ark_bn254::G1Projective.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
}

fn ark_g1_slice_mut(values: &mut [ArkG1]) -> &mut [ark_bn254::G1Projective] {
    // SAFETY: ArkG1 is repr(transparent) over ark_bn254::G1Projective.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len()) }
}

fn ark_g2_slice(values: &[ArkG2]) -> &[ark_bn254::G2Projective] {
    // SAFETY: ArkG2 is repr(transparent) over ark_bn254::G2Projective.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
}

fn ark_g2_slice_mut(values: &mut [ArkG2]) -> &mut [ark_bn254::G2Projective] {
    // SAFETY: ArkG2 is repr(transparent) over ark_bn254::G2Projective.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len()) }
}
