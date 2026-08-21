use ark_ec::pairing::{MillerLoopOutput, Pairing};
use jolt_crypto::Bn254GT;
use jolt_dory::{DoryCommitment, DoryProverSetup};

use super::{arena, curve};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::error::CudaError;
use crate::cuda::common::msm::{JacobianLimbs, FQ_LIMBS};
use crate::cuda::common::pairing::FQ12_LIMBS;

fn exponentiate(words: &[u64]) -> Option<Bn254GT> {
    <ark_bn254::Bn254 as Pairing>::final_exponentiation(MillerLoopOutput(curve::fq12(words)))
        .map(|output| Bn254GT::from(output.0))
}

fn lane_words(limbs: &[u64], lane: usize) -> Option<&[u64]> {
    limbs.get(lane * FQ12_LIMBS..(lane + 1) * FQ12_LIMBS)
}

#[cfg(feature = "parallel")]
fn final_exponentiations(limbs: &[u64], lanes: usize) -> Option<Vec<Bn254GT>> {
    use rayon::prelude::*;

    (0..lanes)
        .into_par_iter()
        .map(|lane| lane_words(limbs, lane).and_then(exponentiate))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn final_exponentiations(limbs: &[u64], lanes: usize) -> Option<Vec<Bn254GT>> {
    (0..lanes)
        .map(|lane| lane_words(limbs, lane).and_then(exponentiate))
        .collect()
}

fn write_row(dst: &mut [u64], row: &JacobianLimbs) {
    dst[..FQ_LIMBS].copy_from_slice(&row.x);
    dst[FQ_LIMBS..2 * FQ_LIMBS].copy_from_slice(&row.y);
    dst[2 * FQ_LIMBS..3 * FQ_LIMBS].copy_from_slice(&row.z);
}

fn locate<'a>(
    columns: &'a [Vec<JacobianLimbs>],
    members: &[usize],
    count: usize,
    index: usize,
) -> Option<&'a JacobianLimbs> {
    let member = *members.get(index / count)?;
    columns.get(member)?.get(index % count)
}

#[cfg(feature = "parallel")]
fn flatten_rows(
    columns: &[Vec<JacobianLimbs>],
    members: &[usize],
    count: usize,
    out: &mut [u64],
) -> bool {
    use rayon::prelude::*;

    out.par_chunks_mut(3 * FQ_LIMBS)
        .enumerate()
        .map(
            |(index, dst)| match locate(columns, members, count, index) {
                Some(row) => {
                    write_row(dst, row);
                    true
                }
                None => false,
            },
        )
        .reduce(|| true, |a, b| a && b)
}

#[cfg(not(feature = "parallel"))]
fn flatten_rows(
    columns: &[Vec<JacobianLimbs>],
    members: &[usize],
    count: usize,
    out: &mut [u64],
) -> bool {
    for (index, dst) in out.chunks_mut(3 * FQ_LIMBS).enumerate() {
        match locate(columns, members, count, index) {
            Some(row) => write_row(dst, row),
            None => return false,
        }
    }
    true
}

pub(crate) fn tier2_batched(
    context: &CudaKernelContext,
    setup: &DoryProverSetup,
    columns: &[Vec<JacobianLimbs>],
) -> Result<Vec<DoryCommitment>, CudaError> {
    if columns.is_empty() {
        return Ok(Vec::new());
    }
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for (index, rows) in columns.iter().enumerate() {
        match groups.iter_mut().find(|(count, _)| *count == rows.len()) {
            Some((_, members)) => members.push(index),
            None => groups.push((rows.len(), vec![index])),
        }
    }

    let mut placed: Vec<(usize, DoryCommitment)> = Vec::with_capacity(columns.len());
    for (count, members) in groups {
        if count == 0 || count > setup.0.g2_vec.len() {
            return Err(CudaError::LengthMismatch {
                expected: setup.0.g2_vec.len(),
                got: count,
            });
        }
        let g2: Vec<u64> = setup.0.g2_vec[..count]
            .iter()
            .flat_map(|base| arena::g2_limbs(&base.0))
            .collect();
        let device_g2 = context.upload_raw_u64(&g2)?;

        let mut g1 = vec![0u64; members.len() * count * 3 * FQ_LIMBS];
        if !flatten_rows(columns, &members, count, &mut g1) {
            return Err(CudaError::InvariantViolation {
                reason: "a tier-2 group named a column outside the batch",
            });
        }
        let device_g1 = context.upload_raw_u64(&g1)?;

        let segments: Vec<(usize, usize)> =
            (0..members.len()).map(|lane| (lane * count, 0)).collect();
        let limbs = context.multi_miller_batch(&device_g1, &device_g2, &segments, count)?;

        let outputs =
            final_exponentiations(&limbs, members.len()).ok_or(CudaError::InvariantViolation {
                reason: "a batched tier-2 Miller output was degenerate",
            })?;
        for (value, &member) in outputs.into_iter().zip(members.iter()) {
            placed.push((member, DoryCommitment(value)));
        }
    }

    placed.sort_by_key(|&(member, _)| member);
    Ok(placed.into_iter().map(|(_, value)| value).collect())
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: PCS and device operations fail loudly"
)]
mod tests {
    use ark_ec::PrimeGroup;
    use jolt_dory::DoryScheme;
    use jolt_openings::StreamingCommitment;

    use super::tier2_batched;
    use crate::cuda::commitment::DeviceTier1Commitment;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::msm::{JacobianLimbs, FQ_LIMBS};

    const NUM_VARS: usize = 8;

    fn limbs(point: &ark_bn254::G1Projective) -> JacobianLimbs {
        JacobianLimbs {
            x: point.x.0 .0,
            y: point.y.0 .0,
            z: point.z.0 .0,
        }
    }

    fn columns(shape: &[usize]) -> Vec<Vec<JacobianLimbs>> {
        let step = ark_bn254::G1Projective::generator();
        let mut walk = step;
        shape
            .iter()
            .enumerate()
            .map(|(column, &rows)| {
                (0..rows)
                    .map(|row| {
                        walk += step;
                        if (column + row) % 37 == 0 {
                            JacobianLimbs::IDENTITY
                        } else {
                            limbs(&walk)
                        }
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn tier2_batched_matches_reference_dory() {
        let Some(context) = shared_context() else {
            return;
        };
        let setup = DoryScheme::setup_prover(NUM_VARS);
        for shape in [
            vec![4usize],
            vec![4, 4, 4],
            vec![16, 16],
            vec![8, 8, 8, 8, 8],
        ] {
            let source = columns(&shape);
            let expected: Vec<_> = source
                .iter()
                .map(|rows| {
                    let partial = DoryScheme::partial_from_rows(&setup, rows).expect("partial");
                    DoryScheme::finish_with_hint(partial, &setup).0
                })
                .collect();

            let got = tier2_batched(context, &setup, &source).expect("batched tier-2");

            assert_eq!(
                got.len(),
                expected.len(),
                "column count diverged for shape {shape:?}"
            );
            let divergence = got
                .iter()
                .zip(&expected)
                .position(|(got, expected)| got != expected);
            assert_eq!(
                divergence, None,
                "batched tier-2 diverged for shape {shape:?}"
            );
        }
        assert_eq!(FQ_LIMBS, 4, "the row limb layout assumes four-limb Fq");
    }

    #[test]
    fn tier2_batched_matches_reference_dory_over_many_columns() {
        let Some(context) = shared_context() else {
            return;
        };
        let mut shape = vec![1024usize; 40];
        shape.extend([512, 512]);
        let setup = DoryScheme::setup_prover(20);
        let source = columns(&shape);

        let expected: Vec<_> = source
            .iter()
            .map(|rows| {
                let partial = DoryScheme::partial_from_rows(&setup, rows).expect("partial");
                DoryScheme::finish_with_hint(partial, &setup).0
            })
            .collect();

        let got = tier2_batched(context, &setup, &source).expect("batched tier-2");

        let divergence = got
            .iter()
            .zip(&expected)
            .position(|(got, expected)| got != expected);
        assert_eq!(
            divergence,
            None,
            "batched tier-2 diverged over {} columns",
            shape.len()
        );
    }
}
