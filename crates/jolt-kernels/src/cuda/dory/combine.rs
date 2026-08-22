use ark_bn254::{Fq, G1Projective};
use ark_ff::BigInt;
use jolt_crypto::Bn254G1;
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_openings::AdditivelyHomomorphic;

use jolt_dory::{DoryHint, DoryScheme};

use crate::cuda::common::context::{context_for, shared_context, CudaKernelContext};
use crate::cuda::common::devices::{device_windows, fan_out, CycleWindow, DeviceTask};
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

fn fill_row(dst: &mut [JacobianLimbs], hint: &DoryHint, start: usize) {
    for (offset, target) in dst.iter_mut().enumerate() {
        if let Some(&commitment) = hint.row_commitments.get(start + offset) {
            *target = jacobian_limbs(commitment);
        }
    }
}

#[cfg(feature = "parallel")]
fn fill_bases(bases: &mut [JacobianLimbs], hints: &[DoryHint], window: &CycleWindow) {
    use rayon::prelude::*;

    bases
        .par_chunks_mut(window.len)
        .zip(hints.par_iter())
        .for_each(|(dst, hint)| fill_row(dst, hint, window.start));
}

#[cfg(not(feature = "parallel"))]
fn fill_bases(bases: &mut [JacobianLimbs], hints: &[DoryHint], window: &CycleWindow) {
    for (dst, hint) in bases.chunks_mut(window.len).zip(hints) {
        fill_row(dst, hint, window.start);
    }
}

fn combine_window(
    context: &CudaKernelContext,
    hints: &[DoryHint],
    scalars: &[Fr],
    window: &CycleWindow,
) -> Result<Vec<JacobianLimbs>, CudaError> {
    let mut bases = vec![JacobianLimbs::IDENTITY; window.len * hints.len()];
    fill_bases(&mut bases, hints, window);
    let combined = context.msm_rows_shared_scalars(&bases, scalars, window.len)?;
    if combined.len() != window.len {
        return Err(CudaError::LengthMismatch {
            expected: window.len,
            got: combined.len(),
        });
    }
    Ok(combined)
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
    let absent = || CudaError::NotImplemented {
        kernel: "no CUDA device is present for the hint combination",
    };

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

    let windows = device_windows(rows, 1);
    let combined = if windows.len() > 1 {
        let tasks: Vec<DeviceTask<'_, Vec<JacobianLimbs>, CudaError>> = windows
            .iter()
            .enumerate()
            .map(|(ordinal, window)| {
                let task: DeviceTask<'_, Vec<JacobianLimbs>, CudaError> = Box::new(move || {
                    let context = context_for(ordinal).ok_or_else(absent)?;
                    tracing::info_span!(
                        "cuda_combine_hints_window",
                        device = ordinal,
                        rows = window.len
                    )
                    .in_scope(|| combine_window(context, hints, scalars, window))
                });
                task
            })
            .collect();
        fan_out(tasks)?.concat()
    } else {
        let window = windows.first().ok_or(CudaError::InvariantViolation {
            reason: "the hint-combination row partition produced no windows",
        })?;
        combine_window(shared_context().ok_or_else(absent)?, hints, scalars, window)?
    };
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

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use ark_ec::PrimeGroup as _;
    use jolt_field::FromPrimitiveInt as _;

    use super::{combine_window, CycleWindow, DoryHint, Fr};
    use crate::cuda::common::context::shared_context;

    const ROWS: usize = 37;
    const HINTS: usize = 6;

    fn hints() -> Vec<DoryHint> {
        let generator = ark_bn254::G1Projective::generator();
        (0..HINTS)
            .map(|hint| {
                let commitments = (0..ROWS)
                    .map(|row| {
                        let scalar = ark_bn254::Fr::from((hint * ROWS + row + 1) as u64);
                        jolt_crypto::Bn254G1::from(generator * scalar)
                    })
                    .collect();
                DoryHint::new(commitments, Fr::from_u64(hint as u64))
            })
            .collect()
    }

    #[test]
    fn combine_row_windows_match_the_whole_row_range() {
        let Some(context) = shared_context() else {
            return;
        };
        let hints = hints();
        let scalars: Vec<Fr> = (0..HINTS)
            .map(|term| Fr::from_u64(7 * term as u64 + 3))
            .collect();
        let whole = combine_window(
            context,
            &hints,
            &scalars,
            &CycleWindow {
                start: 0,
                len: ROWS,
            },
        )
        .expect("whole row range");

        for devices in [2usize, 3, 8] {
            let windows = super::device_windows(ROWS, 1);
            let windows = if windows.len() == 1 {
                (0..devices)
                    .map(|device| {
                        let start = device * ROWS / devices;
                        CycleWindow {
                            start,
                            len: (device + 1) * ROWS / devices - start,
                        }
                    })
                    .collect()
            } else {
                windows
            };
            let got: Vec<_> = windows
                .iter()
                .flat_map(|window| {
                    combine_window(context, &hints, &scalars, window).expect("row window")
                })
                .collect();
            let divergence = got
                .iter()
                .zip(&whole)
                .position(|(got, expected)| jacobian_limbs_differ(*got, *expected));
            assert_eq!(got.len(), whole.len(), "row count diverged");
            assert_eq!(
                divergence, None,
                "splitting the hint combination across {devices} row windows changed a row \
                 commitment; each output row is an independent MSM over the same scalars, so the \
                 partition must be exact",
            );
        }
    }

    fn jacobian_limbs_differ(
        got: crate::cuda::common::msm::JacobianLimbs,
        expected: crate::cuda::common::msm::JacobianLimbs,
    ) -> bool {
        super::jolt_g1(got) != super::jolt_g1(expected)
    }
}
