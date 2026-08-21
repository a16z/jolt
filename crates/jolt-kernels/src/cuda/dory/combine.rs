use ark_bn254::{Fq, G1Projective};
use ark_ff::BigInt;
use jolt_crypto::Bn254G1;
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_openings::AdditivelyHomomorphic;

use jolt_dory::{DoryHint, DoryScheme};

use crate::cuda::common::context::shared_context;
use crate::cuda::common::error::CudaError;
use crate::cuda::common::msm::{JacobianLimbs, FQ_LIMBS};

fn jacobian_limbs(point: Bn254G1) -> JacobianLimbs {
    let point = point.into_inner();
    JacobianLimbs {
        x: point.x.0 .0,
        y: point.y.0 .0,
        z: point.z.0 .0,
    }
}

fn fq(limbs: [u64; FQ_LIMBS]) -> Fq {
    Fq::new_unchecked(BigInt(limbs))
}

fn jolt_g1(point: JacobianLimbs) -> Bn254G1 {
    Bn254G1::from(G1Projective::new_unchecked(
        fq(point.x),
        fq(point.y),
        fq(point.z),
    ))
}

fn fill_row(dst: &mut [JacobianLimbs], hint: &DoryHint) {
    for (row, target) in dst.iter_mut().enumerate() {
        if let Some(&commitment) = hint.row_commitments.get(row) {
            *target = jacobian_limbs(commitment);
        }
    }
}

#[cfg(feature = "parallel")]
fn fill_bases(bases: &mut [JacobianLimbs], hints: &[DoryHint], rows: usize) {
    use rayon::prelude::*;

    bases
        .par_chunks_mut(rows)
        .zip(hints.par_iter())
        .for_each(|(dst, hint)| fill_row(dst, hint));
}

#[cfg(not(feature = "parallel"))]
fn fill_bases(bases: &mut [JacobianLimbs], hints: &[DoryHint], rows: usize) {
    for (dst, hint) in bases.chunks_mut(rows).zip(hints) {
        fill_row(dst, hint);
    }
}

pub(super) fn combine_hints(hints: Vec<DoryHint>, scalars: &[Fr]) -> DoryHint {
    match combine_on_device(&hints, scalars) {
        Ok(hint) => hint,
        Err(error) => {
            tracing::warn!(
                ?error,
                "the device hint combination declined; falling back to DoryScheme"
            );
            DoryScheme::combine_hints(hints, scalars)
        }
    }
}

#[tracing::instrument(skip_all, name = "cuda_combine_hints", fields(hints = hints.len(), rows))]
pub(super) fn combine_on_device(hints: &[DoryHint], scalars: &[Fr]) -> Result<DoryHint, CudaError> {
    if hints.is_empty() || hints.len() != scalars.len() {
        return Err(CudaError::LengthMismatch {
            expected: hints.len(),
            got: scalars.len(),
        });
    }
    let context = shared_context().ok_or(CudaError::NotImplemented {
        kernel: "no CUDA device is present for the hint combination",
    })?;

    let rows = hints
        .iter()
        .map(|hint| hint.row_commitments.len())
        .max()
        .unwrap_or(0);
    if rows == 0 {
        return Err(CudaError::InvariantViolation {
            reason: "every opening hint carries an empty row-commitment vector",
        });
    }
    let _ = tracing::Span::current().record("rows", rows);

    let mut bases = vec![JacobianLimbs::IDENTITY; rows * hints.len()];
    fill_bases(&mut bases, hints, rows);

    let combined = context.msm_rows_shared_scalars(&bases, scalars, rows)?;
    if combined.len() != rows {
        return Err(CudaError::LengthMismatch {
            expected: rows,
            got: combined.len(),
        });
    }

    let blind = hints
        .iter()
        .zip(scalars)
        .fold(Fr::from_u64(0), |acc, (hint, &scalar)| {
            acc + scalar * hint.commit_blind
        });

    Ok(DoryHint::new(
        combined.into_iter().map(jolt_g1).collect(),
        blind,
    ))
}
