use cudarc::driver::CudaSlice;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_field::{Fr, JoltField};

use crate::cuda::common::context::{context_for, CudaKernelContext, BLOCK};
use crate::cuda::common::device::{fr_into, require_fr, DeviceFrVec, LIMBS};
use crate::cuda::common::devices::{fan_out, DeviceTask};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::primitives::reduce_lanes;
use crate::cuda::common::split_eq::DeviceSplitEq;

const LANES: usize = 3;
const CONSOLIDATE_AT: usize = 1 << 12;

pub struct DeviceHammingWeight {
    weights: DeviceFrVec,
    one: DeviceFrVec,
}

impl DeviceHammingWeight {
    #[cfg(test)]
    pub fn new(context: &CudaKernelContext, weights: &[u64]) -> Result<Self, CudaError> {
        Self::require_power_of_two(weights.len())?;
        Ok(Self {
            weights: context.u64_to_montgomery(weights)?,
            one: context.upload(&[Fr::from(1u64)])?,
        })
    }

    pub fn from_device(
        context: &CudaKernelContext,
        weights: &CudaSlice<u64>,
        cycles: usize,
    ) -> Result<Self, CudaError> {
        Self::require_power_of_two(cycles)?;
        Ok(Self {
            weights: context.u64_to_montgomery_device(weights, cycles)?,
            one: context.upload(&[Fr::from(1u64)])?,
        })
    }

    fn from_table(context: &CudaKernelContext, values: &[Fr]) -> Result<Self, CudaError> {
        Self::require_power_of_two(values.len())?;
        Ok(Self {
            weights: context.upload(values)?,
            one: context.upload(&[Fr::from(1u64)])?,
        })
    }

    fn require_power_of_two(len: usize) -> Result<(), CudaError> {
        if len.is_power_of_two() {
            return Ok(());
        }
        Err(CudaError::LengthMismatch {
            expected: len.next_power_of_two(),
            got: len,
        })
    }

    pub fn bind<F: JoltField>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        let len = self.weights.len();
        if len < 2 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: len,
            });
        }
        self.weights = context.bind_rows(&self.weights, len, require_fr(challenge)?)?;
        Ok(())
    }

    pub fn final_claim<F: JoltField>(&self) -> Result<F, CudaError> {
        if self.weights.len() != 1 {
            return Err(CudaError::LengthMismatch {
                expected: 1,
                got: self.weights.len(),
            });
        }
        fr_into(self.weights.first()?).ok_or(CudaError::NotImplemented {
            kernel: "CUDA kernels support only the BN254 scalar field",
        })
    }

    pub fn round_coefficients<F: JoltField>(
        &self,
        context: &CudaKernelContext,
        eq: &DeviceSplitEq<F>,
    ) -> Result<(F, F, F), CudaError> {
        let half = self.weights.len() / 2;
        if half == 0 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: self.weights.len(),
            });
        }
        let e_in_len = eq.e_in_len();
        if eq.e_out_current().len() * e_in_len != half {
            return Err(CudaError::LengthMismatch {
                expected: half,
                got: eq.e_out_current().len() * e_in_len,
            });
        }

        let half_count = CudaKernelContext::count_of(half)?;
        let blocks = half_count.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(LANES * blocks as usize)?;
        let polys = CudaKernelContext::count_of(1)?;
        let e_in_arg = CudaKernelContext::count_of(e_in_len)?;
        let num_x_in_bits = e_in_len.max(1).ilog2();
        let e_in = eq.e_in_current();
        let e_out = eq.e_out_current();

        let mut builder = context.stream().launch_builder(context.brc_message_dense());
        let _ = builder.arg(self.weights.limbs());
        let _ = builder.arg(self.one.limbs());
        let _ = builder.arg(&polys);
        let _ = builder.arg(&half_count);
        let _ = builder.arg(e_in.limbs());
        let _ = builder.arg(&e_in_arg);
        let _ = builder.arg(e_out.limbs());
        let _ = builder.arg(&num_x_in_bits);
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: thread `g < half` reads `weights[2g]` and `weights[2g + 1]` — the
        // single-polynomial case of the kernel's `polys` loop, inside `weights`'s
        // `2 * half` elements — plus `one[0]` of its one element, and
        // `e_in[g & mask]` and `e_out[g >> num_x_in_bits]`, both bounded because
        // `e_out.len() * e_in.len() == half` is checked above. It writes only
        // `partials[lane * gridDim.x + blockIdx.x]` for `lane < 2`, of `2 * blocks`.
        // Shared memory is `BLOCK * LIMBS` u64s, matching `shared_mem_bytes`, and
        // the block reduction sits outside the `g < half` guard so every thread
        // reaches each `__syncthreads()`.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;

        let totals = reduce_lanes(
            context,
            partials,
            CudaKernelContext::count_of(LANES)?,
            blocks,
        )?;
        let host = totals.to_host()?;
        let unsupported = || CudaError::NotImplemented {
            kernel: "CUDA kernels support only the BN254 scalar field",
        };
        let constant = fr_into(host[0]).ok_or_else(unsupported)?;
        let quadratic = fr_into(host[1]).ok_or_else(unsupported)?;
        Ok((
            constant,
            quadratic,
            fr_into(host[2]).ok_or_else(unsupported)?,
        ))
    }
}

pub(crate) struct HammingShard<F: JoltField> {
    pub(crate) ordinal: usize,
    pub(crate) weights: DeviceHammingWeight,
    pub(crate) eq: DeviceSplitEq<F>,
}

pub(crate) struct ShardedHammingWeight<F: JoltField> {
    shards: Vec<HammingShard<F>>,
    collapsed: Option<DeviceHammingWeight>,
}

impl<F: JoltField> ShardedHammingWeight<F> {
    pub(crate) fn new(shards: Vec<HammingShard<F>>, log_t: usize) -> Result<Self, CudaError> {
        let count = shards.len();
        if count == 0 || !count.is_power_of_two() {
            return Err(CudaError::InvariantViolation {
                reason: "a sharded hamming weight needs a power-of-two shard count",
            });
        }
        let tail_rounds = count.trailing_zeros() as usize;
        if tail_rounds > log_t {
            return Err(CudaError::InvariantViolation {
                reason: "a sharded hamming weight cannot split more windows than cycle rounds",
            });
        }
        let local_len = 1usize << (log_t - tail_rounds);
        if shards.iter().enumerate().any(|(ordinal, shard)| {
            shard.ordinal != ordinal || shard.weights.weights.len() != local_len
        }) {
            return Err(CudaError::InvariantViolation {
                reason: "hamming weight shards must cover equal cycle windows in device order",
            });
        }
        if count == 1 {
            let shard = shards
                .into_iter()
                .next()
                .ok_or(CudaError::InvariantViolation {
                    reason: "a single-shard hamming weight lost its state",
                })?;
            return Ok(Self {
                shards: Vec::new(),
                collapsed: Some(shard.weights),
            });
        }
        let mut weights = Self {
            shards,
            collapsed: None,
        };
        if local_len <= CONSOLIDATE_AT {
            weights.collapse()?;
        }
        Ok(weights)
    }

    pub(crate) fn round_coefficients(
        &self,
        whole_eq: &DeviceSplitEq<F>,
    ) -> Result<(F, F, F), CudaError> {
        if let Some(collapsed) = &self.collapsed {
            let context = context_for(0).ok_or(absent())?;
            return collapsed.round_coefficients(context, whole_eq);
        }
        let tasks: Vec<DeviceTask<'_, (F, F, F), CudaError>> = self
            .shards
            .iter()
            .map(|shard| {
                let task: DeviceTask<'_, (F, F, F), CudaError> = Box::new(move || {
                    let context = context_for(shard.ordinal).ok_or(absent())?;
                    shard.weights.round_coefficients(context, &shard.eq)
                });
                task
            })
            .collect();
        let mut total = (F::zero(), F::zero(), F::zero());
        for part in fan_out(tasks)? {
            total.0 += part.0;
            total.1 += part.1;
            total.2 += part.2;
        }
        Ok(total)
    }

    pub(crate) fn bind(&mut self, challenge: F) -> Result<(), CudaError> {
        if let Some(collapsed) = &mut self.collapsed {
            let context = context_for(0).ok_or(absent())?;
            return collapsed.bind(context, challenge);
        }
        let tasks: Vec<DeviceTask<'_, (), CudaError>> = self
            .shards
            .iter_mut()
            .map(|shard| {
                let task: DeviceTask<'_, (), CudaError> = Box::new(move || {
                    let context = context_for(shard.ordinal).ok_or(absent())?;
                    shard.weights.bind(context, challenge)?;
                    shard.eq.bind(challenge);
                    Ok(())
                });
                task
            })
            .collect();
        let _ = fan_out(tasks)?;
        if self
            .shards
            .first()
            .is_some_and(|shard| shard.weights.weights.len() <= CONSOLIDATE_AT)
        {
            self.collapse()?;
        }
        Ok(())
    }

    fn collapse(&mut self) -> Result<(), CudaError> {
        let context = context_for(0).ok_or(absent())?;
        let shards = std::mem::take(&mut self.shards);
        let len = shards.iter().map(|shard| shard.weights.weights.len()).sum();
        let mut values = Vec::with_capacity(len);
        for shard in &shards {
            values.extend(shard.weights.weights.to_host()?);
        }
        self.collapsed = Some(DeviceHammingWeight::from_table(context, &values)?);
        Ok(())
    }

    pub(crate) fn final_claim(&self) -> Result<F, CudaError> {
        self.collapsed
            .as_ref()
            .ok_or(CudaError::InvariantViolation {
                reason: "a sharded hamming weight was asked for its claim before the tail rounds",
            })?
            .final_claim()
    }
}

const fn absent() -> CudaError {
    CudaError::InvariantViolation {
        reason: "a sharded hamming weight window names an absent device",
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, Ring};
    use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial};
    use proptest::prelude::*;

    use super::{DeviceHammingWeight, HammingShard, ShardedHammingWeight};
    use crate::cuda::common::context::{context_for, shared_context};
    use crate::cuda::common::split_eq::DeviceSplitEq;
    use crate::cuda::common::testing::arb_point;

    fn indicator(seed: u64, cycles: usize, stride: usize) -> Vec<u64> {
        (0..cycles)
            .map(|cycle| {
                let mix = seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(cycle as u64);
                u64::from(((mix ^ (mix >> 29)) as usize).is_multiple_of(stride))
            })
            .collect()
    }

    fn cpu_coefficients(eq: &GruenSplitEqPolynomial<Fr>, weights: &Polynomial<Fr>) -> (Fr, Fr, Fr) {
        let e_in = eq.e_in_current();
        let e_out = eq.e_out_current();
        let bits = e_in.len().max(1).ilog2();
        let mut constant = Fr::from_u64(0);
        let mut quadratic = Fr::from_u64(0);
        let mut at_one = Fr::from_u64(0);
        for g in 0..weights.len() / 2 {
            let weight = e_out[g >> bits]
                * if e_in.len() <= 1 {
                    Fr::from_u64(1)
                } else {
                    e_in[g & ((1usize << bits) - 1)]
                };
            let h0 = weights.evals()[2 * g];
            let h1 = weights.evals()[2 * g + 1];
            let delta = h1 - h0;
            constant += weight * (h0 * h0 - h0);
            at_one += weight * (h1 * h1 - h1);
            quadratic += weight * delta * delta;
        }
        (constant, quadratic, at_one)
    }

    #[test]
    fn sharded_hamming_weight_preserves_rounds_and_claim_through_consolidation() {
        let Some(second) = context_for(1) else {
            return;
        };
        let contexts = [context_for(0).expect("first device"), second];
        for log_t in [8, 15, 18] {
            let cycles = 1usize << log_t;
            let column = indicator(17, cycles, 3);
            let point: Vec<_> = (0..log_t)
                .map(|i| Fr::from_u64((i * 13 + 7) as u64))
                .collect();
            let shards = contexts
                .iter()
                .enumerate()
                .map(|(ordinal, context)| {
                    let start = ordinal * cycles / contexts.len();
                    let end = start + cycles / contexts.len();
                    HammingShard {
                        ordinal,
                        weights: DeviceHammingWeight::new(context, &column[start..end])
                            .expect("shard weights"),
                        eq: DeviceSplitEq::new_window(
                            context,
                            &point,
                            BindingOrder::LowToHigh,
                            ordinal,
                            contexts.len(),
                        )
                        .expect("shard equality"),
                    }
                })
                .collect();
            let mut got = ShardedHammingWeight::new(shards, log_t).expect("sharded weights");
            let mut got_eq = DeviceSplitEq::new(contexts[0], &point, BindingOrder::LowToHigh)
                .expect("whole equality");
            let mut expected_weights =
                Polynomial::new(column.into_iter().map(Fr::from_u64).collect::<Vec<_>>());
            let mut expected_eq = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
            for round in 0..log_t {
                assert_eq!(
                    got.round_coefficients(&got_eq).expect("round coefficients"),
                    cpu_coefficients(&expected_eq, &expected_weights),
                    "2^{log_t} round {round}",
                );
                let challenge = Fr::from_u64(match round % 4 {
                    0 => 0,
                    1 => 1,
                    _ => (round * 19 + 11) as u64,
                });
                got.bind(challenge).expect("sharded bind");
                got_eq.bind(challenge);
                expected_weights.bind_with_order(challenge, BindingOrder::LowToHigh);
                expected_eq.bind(challenge);
            }
            assert_eq!(
                got.final_claim().expect("sharded final claim"),
                expected_weights.evals()[0],
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(6))]
        #[test]
        fn device_hamming_weight_matches_cpu_round_for_round(
            log_t in 2usize..9,
            stride in 2usize..7,
            seed in any::<u64>(),
            point in arb_point(9),
            challenges in arb_point(9),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let cycles = 1usize << log_t;
            let column = indicator(seed, cycles, stride);
            prop_assume!(column.contains(&0) && column.contains(&1));

            let mut expected_weights = Polynomial::new(
                column.iter().map(|value| Fr::from_u64(*value)).collect::<Vec<Fr>>(),
            );
            let mut expected_eq =
                GruenSplitEqPolynomial::<Fr>::new(&point[..log_t], BindingOrder::LowToHigh);
            let mut got_weights =
                DeviceHammingWeight::new(context, &column).expect("device hamming weight");
            let mut got_eq =
                DeviceSplitEq::<Fr>::new(context, &point[..log_t], BindingOrder::LowToHigh)
                    .expect("device split-eq");

            for (round, &challenge) in challenges.iter().take(log_t).enumerate() {
                let expected = cpu_coefficients(&expected_eq, &expected_weights);
                let got: (Fr, Fr, Fr) = got_weights
                    .round_coefficients(context, &got_eq)
                    .expect("device round coefficients");
                prop_assert_eq!(
                    got,
                    expected,
                    "round coefficients diverged at round {}", round
                );

                expected_weights.bind_with_order(challenge, BindingOrder::LowToHigh);
                expected_eq.bind(challenge);
                got_weights.bind(context, challenge).expect("device bind");
                got_eq.bind(challenge);
            }

            prop_assert_eq!(
                got_weights.final_claim::<Fr>().expect("device final claim"),
                expected_weights.evals()[0],
                "final claim diverged"
            );
        }
    }
}
