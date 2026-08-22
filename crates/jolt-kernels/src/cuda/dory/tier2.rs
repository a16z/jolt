use ark_ec::pairing::{MillerLoopOutput, Pairing};
use jolt_crypto::Bn254GT;
use jolt_dory::{DoryCommitment, DoryProverSetup};

use super::{arena, curve};
use crate::cuda::common::context::{context_for, device_count, shared_context, CudaKernelContext};
use crate::cuda::common::devices::{fan_out, DeviceTask};
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

fn tier2_selected(
    context: &CudaKernelContext,
    setup: &DoryProverSetup,
    columns: &[Vec<JacobianLimbs>],
    selection: &[usize],
) -> Result<Vec<(usize, DoryCommitment)>, CudaError> {
    if selection.is_empty() {
        return Ok(Vec::new());
    }
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for &index in selection {
        let rows = columns.get(index).ok_or(CudaError::InvariantViolation {
            reason: "a tier-2 selection named a column outside the batch",
        })?;
        match groups.iter_mut().find(|(count, _)| *count == rows.len()) {
            Some((_, members)) => members.push(index),
            None => groups.push((rows.len(), vec![index])),
        }
    }

    let mut placed: Vec<(usize, DoryCommitment)> = Vec::with_capacity(selection.len());
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

    Ok(placed)
}

pub(crate) fn tier2_batched(
    context: &CudaKernelContext,
    setup: &DoryProverSetup,
    columns: &[Vec<JacobianLimbs>],
) -> Result<Vec<DoryCommitment>, CudaError> {
    let selection: Vec<usize> = (0..columns.len()).collect();
    gather(
        tier2_selected(context, setup, columns, &selection)?,
        columns,
    )
}

fn gather(
    mut placed: Vec<(usize, DoryCommitment)>,
    columns: &[Vec<JacobianLimbs>],
) -> Result<Vec<DoryCommitment>, CudaError> {
    if placed.len() != columns.len() {
        return Err(CudaError::LengthMismatch {
            expected: columns.len(),
            got: placed.len(),
        });
    }
    placed.sort_by_key(|&(member, _)| member);
    Ok(placed.into_iter().map(|(_, value)| value).collect())
}

fn device_selections(columns: &[Vec<JacobianLimbs>], devices: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..columns.len()).collect();
    order.sort_by_key(|&index| {
        (
            core::cmp::Reverse(columns.get(index).map_or(0, Vec::len)),
            index,
        )
    });
    let mut load = vec![0usize; devices];
    let mut selections: Vec<Vec<usize>> = (0..devices).map(|_| Vec::new()).collect();
    for index in order {
        let rows = columns.get(index).map_or(0, Vec::len);
        let lightest = load
            .iter()
            .enumerate()
            .min_by_key(|&(device, &pending)| (pending, device))
            .map_or(0, |(device, _)| device);
        if let Some(pending) = load.get_mut(lightest) {
            *pending += rows;
        }
        if let Some(selection) = selections.get_mut(lightest) {
            selection.push(index);
        }
    }
    for selection in &mut selections {
        selection.sort_unstable();
    }
    selections
}

pub(crate) fn tier2_columns(
    setup: &DoryProverSetup,
    columns: &[Vec<JacobianLimbs>],
) -> Result<Vec<DoryCommitment>, CudaError> {
    if columns.is_empty() {
        return Ok(Vec::new());
    }
    let absent = || CudaError::NotImplemented {
        kernel: "no CUDA device is present for the batched tier-2",
    };
    let devices = device_count().min(columns.len());
    if devices <= 1 {
        return tier2_batched(shared_context().ok_or_else(absent)?, setup, columns);
    }
    let selections = device_selections(columns, devices);
    let tasks: Vec<DeviceTask<'_, Vec<(usize, DoryCommitment)>, CudaError>> = selections
        .iter()
        .enumerate()
        .map(|(ordinal, selection)| {
            let task: DeviceTask<'_, Vec<(usize, DoryCommitment)>, CudaError> =
                Box::new(move || {
                    let context = context_for(ordinal).ok_or_else(absent)?;
                    tracing::info_span!(
                        "cuda_commit_tier2_window",
                        device = ordinal,
                        columns = selection.len()
                    )
                    .in_scope(|| tier2_selected(context, setup, columns, selection))
                });
            task
        })
        .collect();
    gather(fan_out(tasks)?.concat(), columns)
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

    use super::{device_selections, gather, tier2_batched, tier2_selected};
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
    fn tier2_device_selections_match_the_whole_batch() {
        let Some(context) = shared_context() else {
            return;
        };
        let setup = DoryScheme::setup_prover(NUM_VARS);
        let mut shape = vec![16usize; 7];
        shape.extend([8, 8, 4]);
        let source = columns(&shape);
        let expected = tier2_batched(context, &setup, &source).expect("whole batch");

        for devices in [2usize, 3, 4] {
            let selections = device_selections(&source, devices);
            let placed: Vec<_> = selections
                .iter()
                .flat_map(|selection| {
                    tier2_selected(context, &setup, &source, selection).expect("selected tier-2")
                })
                .collect();
            let got = gather(placed, &source).expect("gather");
            let divergence = got
                .iter()
                .zip(&expected)
                .position(|(got, expected)| got != expected);
            assert_eq!(
                divergence, None,
                "splitting tier-2 across {devices} devices changed a column commitment; each \
                 column's Miller batch is independent, so the partition must be exact",
            );
        }
    }

    #[test]
    fn device_selections_cover_every_column_once_and_balance_rows() {
        let shape = [1024usize, 1024, 1024, 512, 512, 16];
        let source = columns(&shape);
        for devices in [1usize, 2, 3, 5, 8] {
            let selections = device_selections(&source, devices);
            let mut seen: Vec<usize> = selections.concat();
            seen.sort_unstable();
            assert_eq!(
                seen,
                (0..shape.len()).collect::<Vec<_>>(),
                "every column must land on exactly one device",
            );
            let loads: Vec<usize> = selections
                .iter()
                .map(|selection| selection.iter().map(|&index| shape[index]).sum())
                .collect();
            let spread =
                loads.iter().max().copied().unwrap_or(0) - loads.iter().min().copied().unwrap_or(0);
            let widest = shape.iter().max().copied().unwrap_or(0);
            assert!(
                spread <= widest,
                "row load spread {spread} exceeds the widest column {widest} across {devices} \
                 devices: {loads:?}",
            );
        }
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
