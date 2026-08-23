use cudarc::driver::PushKernelArg;
use jolt_field::{Field, Fr};
use jolt_poly::EqPolynomial;

use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::{require_fr, require_fr_slice, DeviceFrVec};
use crate::cuda::common::error::CudaError;

pub struct DeviceCoefficient {
    values: DeviceFrVec,
}

impl DeviceCoefficient {
    pub fn new<F: Field>(
        context: &CudaKernelContext,
        stage_points: &[Vec<F>],
        weights: &[F],
        entry: F,
        log_t: usize,
    ) -> Result<Self, CudaError> {
        let stages = stage_points.len();
        if stages == 0 || weights.len() != stages {
            return Err(CudaError::InvariantViolation {
                reason: "the bytecode coefficient needs one weight per stage cycle point",
            });
        }
        if log_t == 0 || stage_points.iter().any(|point| point.len() != log_t) {
            return Err(CudaError::InvariantViolation {
                reason: "every bytecode stage cycle point spans the cycle variables",
            });
        }

        let split = log_t / 2;
        let in_bits = log_t - split;
        let e_out_len = 1usize << split;
        let e_in_len = 1usize << in_bits;

        let mut e_in = Vec::with_capacity(stages * e_in_len);
        let mut e_out = Vec::with_capacity(stages * e_out_len);
        for point in stage_points {
            let (outer, inner) = point.split_at(split);
            e_out.extend(EqPolynomial::<F>::evals(outer, None));
            e_in.extend(EqPolynomial::<F>::evals(inner, None));
        }

        let len = 1usize << log_t;
        let device_in = context.upload(require_fr_slice(&e_in)?)?;
        let device_out = context.upload(require_fr_slice(&e_out)?)?;
        let device_weights = context.upload(require_fr_slice(weights)?)?;
        let device_entry = context.upload(&[require_fr(entry)?])?;
        let mut values = context.alloc(len)?;

        let stage_count = CudaKernelContext::count_of(stages)?;
        let inner_len = CudaKernelContext::count_of(e_in_len)?;
        let outer_len = CudaKernelContext::count_of(e_out_len)?;
        let bits = CudaKernelContext::count_of(in_bits)?;
        let count = CudaKernelContext::count_of(len)?;
        let mut builder = context.stream().launch_builder(context.brr_coefficient());
        let _ = builder.arg(device_in.limbs());
        let _ = builder.arg(device_out.limbs());
        let _ = builder.arg(device_weights.limbs());
        let _ = builder.arg(device_entry.limbs());
        let _ = builder.arg(values.limbs_mut());
        let _ = builder.arg(&stage_count);
        let _ = builder.arg(&inner_len);
        let _ = builder.arg(&outer_len);
        let _ = builder.arg(&bits);
        let _ = builder.arg(&count);
        // SAFETY: thread `j < len` reads `e_in[s * e_in_len + (j & (2^in_bits - 1))]`
        // and `e_out[s * e_out_len + (j >> in_bits)]` for every `s < stages` — both
        // inside their `stages * e_in_len` and `stages * e_out_len` elements because
        // `len == e_in_len * e_out_len` — plus `weights[s]` of `stages` and, at
        // `j == 0` only, the single-element `entry`. It writes only `out[j]`, one slot
        // per thread, inside `out`'s `len`; `out` is a fresh allocation.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;

        Ok(Self { values })
    }

    pub fn new_window<F: Field>(
        context: &CudaKernelContext,
        stage_points: &[Vec<F>],
        weights: &[F],
        entry: F,
        log_t: usize,
        shard: usize,
        shards: usize,
    ) -> Result<Self, CudaError> {
        if shards == 1 {
            return Self::new(context, stage_points, weights, entry, log_t);
        }
        let stages = stage_points.len();
        if stages == 0 || weights.len() != stages {
            return Err(CudaError::InvariantViolation {
                reason: "the bytecode coefficient needs one weight per stage cycle point",
            });
        }
        if log_t == 0 || stage_points.iter().any(|point| point.len() != log_t) {
            return Err(CudaError::InvariantViolation {
                reason: "every bytecode stage cycle point spans the cycle variables",
            });
        }
        if shards == 0 || !shards.is_power_of_two() || shard >= shards {
            return Err(CudaError::InvariantViolation {
                reason: "a bytecode coefficient window needs a power-of-two shard count",
            });
        }

        let split = log_t / 2;
        let in_bits = log_t - split;
        let e_out_len = 1usize << split;
        let e_in_len = 1usize << in_bits;
        if e_out_len < shards {
            return Err(CudaError::InvariantViolation {
                reason: "a bytecode coefficient window needs one outer eq entry per shard",
            });
        }
        let outer_window = e_out_len / shards;

        let mut e_in = Vec::with_capacity(stages * e_in_len);
        let mut e_out = Vec::with_capacity(stages * outer_window);
        for point in stage_points {
            let (outer, inner) = point.split_at(split);
            let evals = EqPolynomial::<F>::evals(outer, None);
            let start = shard * outer_window;
            e_out.extend_from_slice(evals.get(start..start + outer_window).ok_or(
                CudaError::InvariantViolation {
                    reason: "a bytecode coefficient window falls outside the outer eq table",
                },
            )?);
            e_in.extend(EqPolynomial::<F>::evals(inner, None));
        }

        let len = e_in_len * outer_window;
        let boundary = if shard == 0 { entry } else { F::zero() };
        let device_in = context.upload(require_fr_slice(&e_in)?)?;
        let device_out = context.upload(require_fr_slice(&e_out)?)?;
        let device_weights = context.upload(require_fr_slice(weights)?)?;
        let device_entry = context.upload(&[require_fr(boundary)?])?;
        let mut values = context.alloc(len)?;

        let stage_count = CudaKernelContext::count_of(stages)?;
        let inner_len = CudaKernelContext::count_of(e_in_len)?;
        let outer_len = CudaKernelContext::count_of(outer_window)?;
        let bits = CudaKernelContext::count_of(in_bits)?;
        let count = CudaKernelContext::count_of(len)?;
        let mut builder = context.stream().launch_builder(context.brr_coefficient());
        let _ = builder.arg(device_in.limbs());
        let _ = builder.arg(device_out.limbs());
        let _ = builder.arg(device_weights.limbs());
        let _ = builder.arg(device_entry.limbs());
        let _ = builder.arg(values.limbs_mut());
        let _ = builder.arg(&stage_count);
        let _ = builder.arg(&inner_len);
        let _ = builder.arg(&outer_len);
        let _ = builder.arg(&bits);
        let _ = builder.arg(&count);
        // SAFETY: thread `j < len` reads `e_in[s * e_in_len + (j & (2^in_bits - 1))]`
        // and `e_out[s * outer_window + (j >> in_bits)]` for every `s < stages` — both
        // inside their `stages * e_in_len` and `stages * outer_window` elements because
        // `len == e_in_len * outer_window` — plus `weights[s]` of `stages` and, at
        // `j == 0` only, the single-element `entry`. It writes only `out[j]`, one slot
        // per thread, inside `out`'s `len`; `out` is a fresh allocation.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;

        Ok(Self { values })
    }

    pub fn from_host(context: &CudaKernelContext, values: &[Fr]) -> Result<Self, CudaError> {
        Ok(Self {
            values: context.upload(values)?,
        })
    }

    pub const fn values(&self) -> &DeviceFrVec {
        &self.values
    }

    pub fn window_scalar(&self) -> Result<Fr, CudaError> {
        let host = self.values.to_host()?;
        match host.as_slice() {
            [value] => Ok(*value),
            other => Err(CudaError::LengthMismatch {
                expected: 1,
                got: other.len(),
            }),
        }
    }

    pub fn bind<F: Field>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        let challenge = require_fr(challenge)?;
        self.values = context.bind_rows(&self.values, self.values.len(), challenge)?;
        Ok(())
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{BindingOrder, EqPolynomial, Polynomial};
    use proptest::prelude::*;

    use super::DeviceCoefficient;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{arb_point, fr};

    fn expected(
        stage_points: &[Vec<Fr>],
        weights: &[Fr],
        entry: Fr,
        log_t: usize,
    ) -> Polynomial<Fr> {
        let mut table = vec![Fr::from_u64(0); 1usize << log_t];
        for (point, &weight) in stage_points.iter().zip(weights) {
            for (slot, eq) in table.iter_mut().zip(EqPolynomial::<Fr>::evals(point, None)) {
                *slot += weight * eq;
            }
        }
        table[0] += entry;
        Polynomial::new(table)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]
        #[test]
        fn device_coefficient_matches_cpu_round_for_round(
            log_t in 1usize..9,
            stages in 1usize..6,
            seed in any::<u64>(),
            entry in any::<u64>().prop_map(fr),
            challenges in arb_point(9),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let stage_points: Vec<Vec<Fr>> = (0..stages)
                .map(|s| {
                    (0..log_t)
                        .map(|i| fr(seed ^ ((s * 131 + i) as u64 * 17 + 5)))
                        .collect()
                })
                .collect();
            let weights: Vec<Fr> = (0..stages)
                .map(|s| fr(seed ^ (s as u64 * 9_973 + 41)))
                .collect();

            let mut want = expected(&stage_points, &weights, entry, log_t);
            let mut got =
                DeviceCoefficient::new(context, &stage_points, &weights, entry, log_t)
                    .expect("device coefficient");

            for (round, &challenge) in challenges.iter().take(log_t).enumerate() {
                prop_assert_eq!(
                    got.values().to_host().expect("download coefficient"),
                    want.evals().to_vec(),
                    "the coefficient table diverged at round {}", round
                );
                want.bind_with_order(challenge, BindingOrder::LowToHigh);
                got.bind(context, challenge).expect("device bind");
            }
            prop_assert_eq!(
                got.values().to_host().expect("download coefficient"),
                want.evals().to_vec(),
                "the fully bound coefficient diverged"
            );
        }
    }
}
