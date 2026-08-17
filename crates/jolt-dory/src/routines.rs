use ark_bn254::{G1Projective, G2Projective};
use dory::backends::arkworks::{ArkFr, ArkG1, ArkG2, G1Routines, G2Routines};
use dory::primitives::arithmetic::DoryRoutines;
use jolt_crypto::ec::bn254::glv::{dory_g1, dory_g2};
use rayon::prelude::*;

fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
    assert_eq!(left.len(), right.len(), "fold: lengths must match");
    left.par_iter_mut()
        .zip(right.par_iter())
        .for_each(|(l, r)| *l = *l * *scalar + *r);
}

pub struct JoltG1Routines;

impl DoryRoutines<ArkG1> for JoltG1Routines {
    fn msm(bases: &[ArkG1], scalars: &[ArkFr]) -> ArkG1 {
        <G1Routines as DoryRoutines<ArkG1>>::msm(bases, scalars)
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG1, scalars: &[ArkFr]) -> Vec<ArkG1> {
        <G1Routines as DoryRoutines<ArkG1>>::fixed_base_vector_scalar_mul(base, scalars)
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG1], vs: &mut [ArkG1], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "bases and vs must have equal length");
        // SAFETY: `ArkG1` is `repr(transparent)` over `G1Projective`, so the
        // two slices have identical layout, length and validity. The mutable
        // borrow of `vs` is not aliased by the shared borrow of `bases` —
        // dory passes the SRS as `bases` and the round state as `vs`, and the
        // length equality above is the only precondition the callee has.
        let (vs, bases) = unsafe {
            (
                std::slice::from_raw_parts_mut(vs.as_mut_ptr().cast::<G1Projective>(), vs.len()),
                std::slice::from_raw_parts(bases.as_ptr().cast::<G1Projective>(), bases.len()),
            )
        };
        dory_g1::vector_add_scalar_mul_g1_online(vs, bases, scalar.0);
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG1], addends: &[ArkG1], scalar: &ArkFr) {
        assert_eq!(
            vs.len(),
            addends.len(),
            "vs and addends must have equal length"
        );
        // SAFETY: as above — `repr(transparent)` reinterpretation of two
        // non-overlapping slices of equal length.
        let (vs, addends) = unsafe {
            (
                std::slice::from_raw_parts_mut(vs.as_mut_ptr().cast::<G1Projective>(), vs.len()),
                std::slice::from_raw_parts(addends.as_ptr().cast::<G1Projective>(), addends.len()),
            )
        };
        dory_g1::vector_scalar_mul_add_gamma_g1_online(vs, scalar.0, addends);
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        fold_field_vectors(left, right, scalar);
    }
}

pub struct JoltG2Routines;

impl DoryRoutines<ArkG2> for JoltG2Routines {
    fn msm(bases: &[ArkG2], scalars: &[ArkFr]) -> ArkG2 {
        <G2Routines as DoryRoutines<ArkG2>>::msm(bases, scalars)
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG2, scalars: &[ArkFr]) -> Vec<ArkG2> {
        <G2Routines as DoryRoutines<ArkG2>>::fixed_base_vector_scalar_mul(base, scalars)
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG2], vs: &mut [ArkG2], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "bases and vs must have equal length");
        // SAFETY: `ArkG2` is `repr(transparent)` over `G2Projective`; same
        // argument as the G1 impl above.
        let (vs, bases) = unsafe {
            (
                std::slice::from_raw_parts_mut(vs.as_mut_ptr().cast::<G2Projective>(), vs.len()),
                std::slice::from_raw_parts(bases.as_ptr().cast::<G2Projective>(), bases.len()),
            )
        };
        dory_g2::vector_add_scalar_mul_g2_online(vs, bases, scalar.0);
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG2], addends: &[ArkG2], scalar: &ArkFr) {
        assert_eq!(
            vs.len(),
            addends.len(),
            "vs and addends must have equal length"
        );
        // SAFETY: as above.
        let (vs, addends) = unsafe {
            (
                std::slice::from_raw_parts_mut(vs.as_mut_ptr().cast::<G2Projective>(), vs.len()),
                std::slice::from_raw_parts(addends.as_ptr().cast::<G2Projective>(), addends.len()),
            )
        };
        dory_g2::vector_scalar_mul_add_gamma_g2_online(vs, scalar.0, addends);
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        fold_field_vectors(left, right, scalar);
    }
}

#[cfg(test)]
mod tests {
    use ark_ec::PrimeGroup;

    use super::*;

    fn g1_points(count: usize, offset: u64) -> Vec<ArkG1> {
        (0..count)
            .map(|i| ArkG1(G1Projective::generator() * ark_bn254::Fr::from(offset + i as u64 + 1)))
            .collect()
    }

    fn g2_points(count: usize, offset: u64) -> Vec<ArkG2> {
        (0..count)
            .map(|i| ArkG2(G2Projective::generator() * ark_bn254::Fr::from(offset + i as u64 + 1)))
            .collect()
    }

    fn scalar(seed: u64) -> ArkFr {
        ArkFr(ark_bn254::Fr::from(
            seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
        ))
    }

    #[test]
    fn g1_scaled_base_add_matches_the_dory_crate_routine() {
        let bases = g1_points(64, 7);
        let scalar = scalar(11);
        let mut ours = g1_points(64, 101);
        let mut theirs = ours.clone();
        JoltG1Routines::fixed_scalar_mul_bases_then_add(&bases, &mut ours, &scalar);
        <G1Routines as DoryRoutines<ArkG1>>::fixed_scalar_mul_bases_then_add(
            &bases,
            &mut theirs,
            &scalar,
        );
        assert_eq!(
            ours.iter().map(|p| p.0).collect::<Vec<_>>(),
            theirs.iter().map(|p| p.0).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn g1_scaled_vs_add_matches_the_dory_crate_routine() {
        let addends = g1_points(64, 7);
        let scalar = scalar(13);
        let mut ours = g1_points(64, 101);
        let mut theirs = ours.clone();
        JoltG1Routines::fixed_scalar_mul_vs_then_add(&mut ours, &addends, &scalar);
        <G1Routines as DoryRoutines<ArkG1>>::fixed_scalar_mul_vs_then_add(
            &mut theirs,
            &addends,
            &scalar,
        );
        assert_eq!(
            ours.iter().map(|p| p.0).collect::<Vec<_>>(),
            theirs.iter().map(|p| p.0).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn g2_scaled_base_add_matches_the_dory_crate_routine() {
        let bases = g2_points(64, 7);
        let scalar = scalar(17);
        let mut ours = g2_points(64, 101);
        let mut theirs = ours.clone();
        JoltG2Routines::fixed_scalar_mul_bases_then_add(&bases, &mut ours, &scalar);
        <G2Routines as DoryRoutines<ArkG2>>::fixed_scalar_mul_bases_then_add(
            &bases,
            &mut theirs,
            &scalar,
        );
        assert_eq!(
            ours.iter().map(|p| p.0).collect::<Vec<_>>(),
            theirs.iter().map(|p| p.0).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn g2_scaled_vs_add_matches_the_dory_crate_routine() {
        let addends = g2_points(64, 7);
        let scalar = scalar(19);
        let mut ours = g2_points(64, 101);
        let mut theirs = ours.clone();
        JoltG2Routines::fixed_scalar_mul_vs_then_add(&mut ours, &addends, &scalar);
        <G2Routines as DoryRoutines<ArkG2>>::fixed_scalar_mul_vs_then_add(
            &mut theirs,
            &addends,
            &scalar,
        );
        assert_eq!(
            ours.iter().map(|p| p.0).collect::<Vec<_>>(),
            theirs.iter().map(|p| p.0).collect::<Vec<_>>(),
        );
    }
}
