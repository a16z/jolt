use dory::backends::arkworks::{ArkFr, ArkG1, ArkG2};
use jolt_field::Fr;

use super::arena::{self, Family};
use super::handle::{
    load_all, load_all_g2, span, span_g2, store_all, store_all_g2, DeviceG1, DeviceG2,
};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::msm::ResidentAxpy;
use dory::primitives::arithmetic::DoryRoutines;

use jolt_dory::{JoltG1Routines, JoltG2Routines};

pub struct CudaG1Routines;

fn resident_lengths(a: usize, b: usize, out: usize) -> Result<(), CudaError> {
    if a == out && b == out {
        return Ok(());
    }
    Err(CudaError::LengthMismatch {
        expected: out,
        got: a.min(b),
    })
}

fn resident_axpy(
    a: &[DeviceG1],
    b: &[DeviceG1],
    out: &[DeviceG1],
    scalar: &ArkFr,
) -> Result<(), CudaError> {
    resident_lengths(a.len(), b.len(), out.len())?;
    if out.is_empty() {
        return Ok(());
    }
    let (Some(a_offset), Some(b_offset), Some(out_offset)) = (span(a), span(b), span(out)) else {
        return Err(CudaError::InvariantViolation {
            reason: "a resident G1 axpy needs three contiguous handle spans",
        });
    };
    arena::axpy(
        Family::G1,
        ResidentAxpy {
            a_offset,
            b_offset,
            out_offset,
            count: out.len(),
        },
        Fr::from(scalar.0),
    )
}

fn resident_axpy_g2(
    a: &[DeviceG2],
    b: &[DeviceG2],
    out: &[DeviceG2],
    scalar: &ArkFr,
) -> Result<(), CudaError> {
    resident_lengths(a.len(), b.len(), out.len())?;
    if out.is_empty() {
        return Ok(());
    }
    let (Some(a_offset), Some(b_offset), Some(out_offset)) = (span_g2(a), span_g2(b), span_g2(out))
    else {
        return Err(CudaError::InvariantViolation {
            reason: "a resident G2 axpy needs three contiguous handle spans",
        });
    };
    arena::axpy(
        Family::G2,
        ResidentAxpy {
            a_offset,
            b_offset,
            out_offset,
            count: out.len(),
        },
        Fr::from(scalar.0),
    )
}

fn resident_fixed_base_g2(base: &DeviceG2, scalars: &[ArkFr]) -> Result<Vec<DeviceG2>, CudaError> {
    if scalars.is_empty() {
        return Ok(Vec::new());
    }
    let weights: Vec<Fr> = scalars.iter().map(|scalar| Fr::from(scalar.0)).collect();
    let out = store_all_g2(&vec![ark_bn254::G2Projective::default(); scalars.len()]);
    let (Some(base_offset), Some(out_offset)) = (span_g2(&[*base]), span_g2(&out)) else {
        return Err(CudaError::InvariantViolation {
            reason: "a resident G2 fixed-base scaling needs a contiguous output span",
        });
    };
    arena::g2_fixed_base(base_offset, out_offset, &weights)?;
    Ok(out)
}

impl DoryRoutines<DeviceG1> for CudaG1Routines {
    #[tracing::instrument(skip_all, name = "cuda_g1r_msm", fields(len = bases.len()))]
    fn msm(bases: &[DeviceG1], scalars: &[ArkFr]) -> DeviceG1 {
        let hosted: Vec<ArkG1> = load_all(bases).into_iter().map(ArkG1).collect();
        DeviceG1::store(&JoltG1Routines::msm(&hosted, scalars).0)
    }

    #[tracing::instrument(skip_all, name = "cuda_g1r_fixed_base_vector", fields(len = scalars.len()))]
    fn fixed_base_vector_scalar_mul(base: &DeviceG1, scalars: &[ArkFr]) -> Vec<DeviceG1> {
        let hosted = JoltG1Routines::fixed_base_vector_scalar_mul(&ArkG1(base.load()), scalars);
        store_all(&hosted.into_iter().map(|point| point.0).collect::<Vec<_>>())
    }

    #[tracing::instrument(skip_all, name = "cuda_g1r_bases_then_add", fields(len = bases.len()))]
    fn fixed_scalar_mul_bases_then_add(bases: &[DeviceG1], vs: &mut [DeviceG1], scalar: &ArkFr) {
        match resident_axpy(bases, vs, vs, scalar) {
            Ok(()) => (),
            Err(error) => {
                tracing::warn!(
                    ?error,
                    "the resident G1 axpy declined; falling back to the host"
                );
                let hosted_bases: Vec<ArkG1> = load_all(bases).into_iter().map(ArkG1).collect();
                let mut hosted_vs: Vec<ArkG1> = load_all(vs).into_iter().map(ArkG1).collect();
                JoltG1Routines::fixed_scalar_mul_bases_then_add(
                    &hosted_bases,
                    &mut hosted_vs,
                    scalar,
                );
                write_back(vs, &hosted_vs);
            }
        }
    }

    #[tracing::instrument(skip_all, name = "cuda_g1r_vs_then_add", fields(len = vs.len()))]
    fn fixed_scalar_mul_vs_then_add(vs: &mut [DeviceG1], addends: &[DeviceG1], scalar: &ArkFr) {
        match resident_axpy(vs, addends, vs, scalar) {
            Ok(()) => (),
            Err(error) => {
                tracing::warn!(
                    ?error,
                    "the resident G1 aypx declined; falling back to the host"
                );
                let mut hosted_vs: Vec<ArkG1> = load_all(vs).into_iter().map(ArkG1).collect();
                let hosted_addends: Vec<ArkG1> = load_all(addends).into_iter().map(ArkG1).collect();
                JoltG1Routines::fixed_scalar_mul_vs_then_add(
                    &mut hosted_vs,
                    &hosted_addends,
                    scalar,
                );
                write_back(vs, &hosted_vs);
            }
        }
    }

    #[tracing::instrument(skip_all, name = "cuda_g1r_fold_field", fields(len = left.len()))]
    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        <JoltG1Routines as DoryRoutines<ArkG1>>::fold_field_vectors(left, right, scalar);
    }
}

fn write_back(handles: &mut [DeviceG1], points: &[ArkG1]) {
    for (handle, point) in handles.iter_mut().zip(points) {
        handle.overwrite(&point.0);
    }
}

pub struct CudaG2Routines;

fn write_back_g2(handles: &mut [DeviceG2], points: &[ArkG2]) {
    for (handle, point) in handles.iter_mut().zip(points) {
        handle.overwrite(&point.0);
    }
}

impl DoryRoutines<DeviceG2> for CudaG2Routines {
    #[tracing::instrument(skip_all, name = "cuda_g2r_msm", fields(len = bases.len()))]
    fn msm(bases: &[DeviceG2], scalars: &[ArkFr]) -> DeviceG2 {
        let hosted: Vec<ArkG2> = load_all_g2(bases).into_iter().map(ArkG2).collect();
        DeviceG2::store(&JoltG2Routines::msm(&hosted, scalars).0)
    }

    #[tracing::instrument(skip_all, name = "cuda_g2r_fixed_base_vector", fields(len = scalars.len()))]
    fn fixed_base_vector_scalar_mul(base: &DeviceG2, scalars: &[ArkFr]) -> Vec<DeviceG2> {
        match resident_fixed_base_g2(base, scalars) {
            Ok(out) => out,
            Err(error) => {
                tracing::warn!(
                    ?error,
                    "the resident G2 fixed-base scaling declined; falling back to the host"
                );
                let hosted =
                    JoltG2Routines::fixed_base_vector_scalar_mul(&ArkG2(base.load()), scalars);
                store_all_g2(&hosted.into_iter().map(|point| point.0).collect::<Vec<_>>())
            }
        }
    }

    #[tracing::instrument(skip_all, name = "cuda_g2r_bases_then_add", fields(len = bases.len()))]
    fn fixed_scalar_mul_bases_then_add(bases: &[DeviceG2], vs: &mut [DeviceG2], scalar: &ArkFr) {
        match resident_axpy_g2(bases, vs, vs, scalar) {
            Ok(()) => (),
            Err(error) => {
                tracing::warn!(
                    ?error,
                    "the resident G2 axpy declined; falling back to the host"
                );
                let hosted_bases: Vec<ArkG2> = load_all_g2(bases).into_iter().map(ArkG2).collect();
                let mut hosted_vs: Vec<ArkG2> = load_all_g2(vs).into_iter().map(ArkG2).collect();
                JoltG2Routines::fixed_scalar_mul_bases_then_add(
                    &hosted_bases,
                    &mut hosted_vs,
                    scalar,
                );
                write_back_g2(vs, &hosted_vs);
            }
        }
    }

    #[tracing::instrument(skip_all, name = "cuda_g2r_vs_then_add", fields(len = vs.len()))]
    fn fixed_scalar_mul_vs_then_add(vs: &mut [DeviceG2], addends: &[DeviceG2], scalar: &ArkFr) {
        match resident_axpy_g2(vs, addends, vs, scalar) {
            Ok(()) => (),
            Err(error) => {
                tracing::warn!(
                    ?error,
                    "the resident G2 aypx declined; falling back to the host"
                );
                let mut hosted_vs: Vec<ArkG2> = load_all_g2(vs).into_iter().map(ArkG2).collect();
                let hosted_addends: Vec<ArkG2> =
                    load_all_g2(addends).into_iter().map(ArkG2).collect();
                JoltG2Routines::fixed_scalar_mul_vs_then_add(
                    &mut hosted_vs,
                    &hosted_addends,
                    scalar,
                );
                write_back_g2(vs, &hosted_vs);
            }
        }
    }

    #[tracing::instrument(skip_all, name = "cuda_g2r_fold_field", fields(len = left.len()))]
    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        <JoltG2Routines as DoryRoutines<ArkG2>>::fold_field_vectors(left, right, scalar);
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::UniformRand;
    use dory::backends::arkworks::{ArkFr, ArkG1, ArkG2};
    use dory::primitives::arithmetic::DoryRoutines;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use jolt_dory::{JoltG1Routines, JoltG2Routines};

    use super::{
        arena, load_all, load_all_g2, store_all, store_all_g2, CudaG1Routines, CudaG2Routines,
        DeviceG1, DeviceG2,
    };
    use crate::cuda::common::context::shared_context;

    fn resident(points: &[ArkG1]) -> Vec<DeviceG1> {
        store_all(&points.iter().map(|point| point.0).collect::<Vec<_>>())
    }

    fn hosted(handles: &[DeviceG1]) -> Vec<ArkG1> {
        load_all(handles).into_iter().map(ArkG1).collect()
    }

    fn points(count: usize, seed: u64) -> Vec<ArkG1> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..count)
            .map(|index| {
                if index % 41 == 0 {
                    ArkG1(ark_bn254::G1Projective::default())
                } else {
                    ArkG1(ark_bn254::G1Projective::rand(&mut rng))
                }
            })
            .collect()
    }

    fn weight(seed: u64) -> ArkFr {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        ArkFr(ark_bn254::Fr::rand(&mut rng))
    }

    const SHAPES: [usize; 5] = [1, 2, 129, 1024, 4096];

    fn affine(points: &[ArkG1]) -> Vec<ark_bn254::G1Affine> {
        points.iter().map(|point| point.0.into()).collect()
    }

    #[test]
    fn g1_bases_then_add_matches_reference_dory() {
        if shared_context().is_none() {
            return;
        }
        for (index, len) in SHAPES.into_iter().enumerate() {
            let bases = points(len, 300 + index as u64);
            let start = points(len, 700 + index as u64);
            let scalar = weight(900 + index as u64);

            let mut expected = start.clone();
            JoltG1Routines::fixed_scalar_mul_bases_then_add(&bases, &mut expected, &scalar);

            let Ok(guard) = arena::open(2 * len + 64, 2 * len + 64) else {
                return;
            };
            let resident_bases = resident(&bases);
            let mut resident_vs = resident(&start);
            CudaG1Routines::fixed_scalar_mul_bases_then_add(
                &resident_bases,
                &mut resident_vs,
                &scalar,
            );
            let got = hosted(&resident_vs);
            assert!(!arena::poisoned(), "the arena poisoned at len {len}");
            drop(guard);

            assert_eq!(
                affine(&got),
                affine(&expected),
                "bases_then_add diverged at len {len}"
            );
        }
    }

    #[test]
    fn g1_vs_then_add_matches_reference_dory() {
        if shared_context().is_none() {
            return;
        }
        for (index, len) in SHAPES.into_iter().enumerate() {
            let addends = points(len, 1_300 + index as u64);
            let start = points(len, 1_700 + index as u64);
            let scalar = weight(1_900 + index as u64);

            let mut expected = start.clone();
            JoltG1Routines::fixed_scalar_mul_vs_then_add(&mut expected, &addends, &scalar);

            let Ok(guard) = arena::open(2 * len + 64, 2 * len + 64) else {
                return;
            };
            let mut resident_vs = resident(&start);
            let resident_addends = resident(&addends);
            CudaG1Routines::fixed_scalar_mul_vs_then_add(
                &mut resident_vs,
                &resident_addends,
                &scalar,
            );
            let got = hosted(&resident_vs);
            assert!(!arena::poisoned(), "the arena poisoned at len {len}");
            drop(guard);

            assert_eq!(
                affine(&got),
                affine(&expected),
                "vs_then_add diverged at len {len}"
            );
        }
    }

    fn resident_g2(points: &[ArkG2]) -> Vec<DeviceG2> {
        store_all_g2(&points.iter().map(|point| point.0).collect::<Vec<_>>())
    }

    fn hosted_g2(handles: &[DeviceG2]) -> Vec<ArkG2> {
        load_all_g2(handles).into_iter().map(ArkG2).collect()
    }

    fn points_g2(count: usize, seed: u64) -> Vec<ArkG2> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..count)
            .map(|index| {
                if index % 41 == 0 {
                    ArkG2(ark_bn254::G2Projective::default())
                } else {
                    ArkG2(ark_bn254::G2Projective::rand(&mut rng))
                }
            })
            .collect()
    }

    fn affine_g2(points: &[ArkG2]) -> Vec<ark_bn254::G2Affine> {
        points.iter().map(|point| point.0.into()).collect()
    }

    fn scalars(count: usize, seed: u64) -> Vec<ArkFr> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..count)
            .map(|index| {
                if index % 17 == 0 {
                    ArkFr(ark_bn254::Fr::from(0u64))
                } else {
                    ArkFr(ark_bn254::Fr::rand(&mut rng))
                }
            })
            .collect()
    }

    #[test]
    fn g2_bases_then_add_matches_reference_dory() {
        if shared_context().is_none() {
            return;
        }
        for (index, len) in SHAPES.into_iter().enumerate() {
            let bases = points_g2(len, 2_300 + index as u64);
            let start = points_g2(len, 2_700 + index as u64);
            let scalar = weight(2_900 + index as u64);

            let mut expected = start.clone();
            JoltG2Routines::fixed_scalar_mul_bases_then_add(&bases, &mut expected, &scalar);

            let Ok(guard) = arena::open(64, 2 * len + 64) else {
                return;
            };
            let resident_bases = resident_g2(&bases);
            let mut resident_vs = resident_g2(&start);
            CudaG2Routines::fixed_scalar_mul_bases_then_add(
                &resident_bases,
                &mut resident_vs,
                &scalar,
            );
            let got = hosted_g2(&resident_vs);
            assert!(!arena::poisoned(), "the arena poisoned at len {len}");
            drop(guard);

            assert_eq!(
                affine_g2(&got),
                affine_g2(&expected),
                "bases_then_add diverged at len {len}"
            );
        }
    }

    #[test]
    fn g2_vs_then_add_matches_reference_dory() {
        if shared_context().is_none() {
            return;
        }
        for (index, len) in SHAPES.into_iter().enumerate() {
            let addends = points_g2(len, 3_300 + index as u64);
            let start = points_g2(len, 3_700 + index as u64);
            let scalar = weight(3_900 + index as u64);

            let mut expected = start.clone();
            JoltG2Routines::fixed_scalar_mul_vs_then_add(&mut expected, &addends, &scalar);

            let Ok(guard) = arena::open(64, 2 * len + 64) else {
                return;
            };
            let mut resident_vs = resident_g2(&start);
            let resident_addends = resident_g2(&addends);
            CudaG2Routines::fixed_scalar_mul_vs_then_add(
                &mut resident_vs,
                &resident_addends,
                &scalar,
            );
            let got = hosted_g2(&resident_vs);
            assert!(!arena::poisoned(), "the arena poisoned at len {len}");
            drop(guard);

            assert_eq!(
                affine_g2(&got),
                affine_g2(&expected),
                "vs_then_add diverged at len {len}"
            );
        }
    }

    #[test]
    fn g2_fixed_base_vector_matches_reference_dory() {
        if shared_context().is_none() {
            return;
        }
        for (index, len) in SHAPES.into_iter().enumerate() {
            let base = points_g2(1, 4_100 + index as u64)[0];
            let weights = scalars(len, 4_300 + index as u64);

            let expected = JoltG2Routines::fixed_base_vector_scalar_mul(&base, &weights);

            let Ok(guard) = arena::open(64, len + 64) else {
                return;
            };
            let resident_base = resident_g2(&[base]);
            let got = hosted_g2(&CudaG2Routines::fixed_base_vector_scalar_mul(
                &resident_base[0],
                &weights,
            ));
            assert!(!arena::poisoned(), "the arena poisoned at len {len}");
            drop(guard);

            assert_eq!(
                affine_g2(&got),
                affine_g2(&expected),
                "fixed_base_vector diverged at len {len}"
            );
        }
    }
}
