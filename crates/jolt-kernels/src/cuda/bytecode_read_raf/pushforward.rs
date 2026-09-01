use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_field::{Field, Fr};

use crate::cuda::common::context::{CudaKernelContext, BLOCK};
use crate::cuda::common::device::{fr_into, require_fr, require_fr_slice, DeviceFrVec, LIMBS};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::one_hot_fold::{affine_table, FoldTuning, OneHotShards};
use crate::cuda::common::primitives::reduce_lanes;

const LANES: usize = 2;

pub const TERMS: usize = 6;

pub const STAGES: usize = 5;

pub struct DeviceBytecodePushforward {
    left: DeviceFrVec,
    right: DeviceFrVec,
    int: DeviceFrVec,
    raf: [Fr; STAGES],
    len: usize,
}

pub struct PushforwardInputs<'a, F: Field> {
    pub stage_cycle_points: &'a [Vec<F>; STAGES],
    pub stage_values: &'a [[F; STAGES]],
    pub entry_trace_index: usize,
    pub entry_expected_index: usize,
    pub gamma: F,
}

impl DeviceBytecodePushforward {
    #[tracing::instrument(skip_all, name = "brap_pushforward")]
    pub fn new<F: Field>(
        context: &CudaKernelContext,
        shards: &OneHotShards,
        inputs: PushforwardInputs<'_, F>,
    ) -> Result<Self, CudaError> {
        let columns = shards.whole()?;
        let addresses = columns.addresses();
        if columns.polys() != 1 {
            return Err(CudaError::InvariantViolation {
                reason: "the bytecode read-RAF address phase pushes forward exactly one PC column",
            });
        }
        if addresses < 2 || inputs.stage_values.len() != addresses {
            return Err(CudaError::LengthMismatch {
                expected: addresses,
                got: inputs.stage_values.len(),
            });
        }
        if inputs.entry_trace_index >= addresses || inputs.entry_expected_index >= addresses {
            return Err(CudaError::InvariantViolation {
                reason: "a bytecode read-RAF entry index escapes the padded bytecode domain",
            });
        }

        let gamma = require_fr(inputs.gamma)?;
        let mut powers = [Fr::from(1u64); 8];
        for index in 1..8 {
            powers[index] = powers[index - 1] * gamma;
        }
        let raf = [
            powers[5],
            Fr::from(0u64),
            powers[4],
            Fr::from(0u64),
            Fr::from(0u64),
        ];

        let int = affine_table(context, 0, 1, addresses)?;
        let mut left = context.alloc(TERMS * addresses)?;
        let mut right = context.alloc(TERMS * addresses)?;

        tracing::info_span!("brap_left_terms").in_scope(|| -> Result<(), CudaError> {
            for (stage, point) in inputs.stage_cycle_points.iter().enumerate() {
                let folded = shards.fold(point, FoldTuning::default())?;
                Self::term(
                    context,
                    &folded,
                    &int,
                    [powers[stage], Fr::from(0u64)],
                    &mut left,
                    stage * addresses,
                    addresses,
                )?;
            }
            Ok(())
        })?;
        let entry_trace = Self::one_hot(context, inputs.entry_trace_index, addresses)?;
        Self::term(
            context,
            &entry_trace,
            &int,
            [Fr::from(1u64), Fr::from(0u64)],
            &mut left,
            STAGES * addresses,
            addresses,
        )?;

        tracing::info_span!("brap_right_terms").in_scope(|| -> Result<(), CudaError> {
            for stage in 0..STAGES {
                let column: Vec<F> = inputs
                    .stage_values
                    .iter()
                    .map(|values| values[stage])
                    .collect();
                let uploaded = context.upload(require_fr_slice(&column)?)?;
                Self::term(
                    context,
                    &uploaded,
                    &int,
                    [Fr::from(1u64), raf[stage]],
                    &mut right,
                    stage * addresses,
                    addresses,
                )?;
            }
            Ok(())
        })?;
        let entry_expected = Self::one_hot(context, inputs.entry_expected_index, addresses)?;
        Self::term(
            context,
            &entry_expected,
            &int,
            [powers[7], Fr::from(0u64)],
            &mut right,
            STAGES * addresses,
            addresses,
        )?;

        Ok(Self {
            left,
            right,
            int,
            raf,
            len: addresses,
        })
    }

    fn one_hot(
        context: &CudaKernelContext,
        index: usize,
        addresses: usize,
    ) -> Result<DeviceFrVec, CudaError> {
        let mut table = context.alloc(addresses)?;
        let count = CudaKernelContext::count_of(addresses)?;
        let hot = CudaKernelContext::count_of(index)?;
        let mut builder = context.stream().launch_builder(context.brap_one_hot());
        let _ = builder.arg(table.limbs_mut());
        let _ = builder.arg(&hot);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` writes only `out[i]` of `count` field elements, a
        // fresh allocation, and reads nothing but the two by-value scalars and the
        // `FR_ONE` constant. Threads with `i >= count` return before any access.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
        Ok(table)
    }

    fn term(
        context: &CudaKernelContext,
        table: &DeviceFrVec,
        addend: &DeviceFrVec,
        scales: [Fr; 2],
        out: &mut DeviceFrVec,
        offset: usize,
        addresses: usize,
    ) -> Result<(), CudaError> {
        if table.len() != addresses || addend.len() != addresses {
            return Err(CudaError::LengthMismatch {
                expected: addresses,
                got: table.len(),
            });
        }
        let scales = context.upload(&scales)?;
        let count = CudaKernelContext::count_of(addresses)?;
        let offset_arg = CudaKernelContext::count_of(offset)?;
        let mut builder = context.stream().launch_builder(context.brap_term());
        let _ = builder.arg(table.limbs());
        let _ = builder.arg(addend.limbs());
        let _ = builder.arg(scales.limbs());
        let _ = builder.arg(out.limbs_mut());
        let _ = builder.arg(&offset_arg);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads `table[i]` and `addend[i]` — both hold
        // `addresses == count` field elements, checked above — plus the two elements of
        // `scales`, and writes only `out[offset + i]`. `offset + addresses` is within
        // `out`'s `TERMS * addresses` because every caller passes a term slot below
        // `TERMS`. Index sets are disjoint across threads and `out` is distinct from
        // both inputs.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
        Ok(())
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub fn round_lanes<F: Field>(&self, context: &CudaKernelContext) -> Result<(F, F), CudaError> {
        let half = self.len / 2;
        if half == 0 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: self.len,
            });
        }
        let half_count = CudaKernelContext::count_of(half)?;
        let blocks = half_count.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(LANES * blocks as usize)?;
        let terms = CudaKernelContext::count_of(TERMS)?;

        let mut builder = context.stream().launch_builder(context.brap_message());
        let _ = builder.arg(self.left.limbs());
        let _ = builder.arg(self.right.limbs());
        let _ = builder.arg(&terms);
        let _ = builder.arg(&half_count);
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: thread `y < half` reads `left[t * 2 * half + 2y]`,
        // `left[t * 2 * half + 2y + 1]` and the same two slots of `right` for every
        // `t < TERMS` — both buffers hold `TERMS * 2 * half` elements, since
        // `len == 2 * half` is their current row length — and writes only
        // `partials[lane * gridDim.x + blockIdx.x]` for `lane < 2`, of `2 * blocks`.
        // Shared memory is `BLOCK * LIMBS` u64s, matching `shared_mem_bytes`, and the
        // block reduction sits outside the `y < half` guard so every thread reaches
        // each `__syncthreads()`.
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
        let at_zero = fr_into(host[0]).ok_or_else(unsupported)?;
        let at_two = fr_into(host[1]).ok_or_else(unsupported)?;
        Ok((at_zero, at_two))
    }

    pub fn bind<F: Field>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        if self.len < 2 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: self.len,
            });
        }
        let challenge = require_fr(challenge)?;
        self.left = context.bind_rows(&self.left, self.len, challenge)?;
        self.right = context.bind_rows(&self.right, self.len, challenge)?;
        self.int = context.bind_rows(&self.int, self.len, challenge)?;
        self.len /= 2;
        Ok(())
    }

    pub fn intermediate_claim<F: Field>(&self) -> Result<F, CudaError> {
        if self.len != 1 {
            return Err(CudaError::LengthMismatch {
                expected: 1,
                got: self.len,
            });
        }
        let left = self.left.to_host()?;
        let right = self.right.to_host()?;
        if left.len() != TERMS || right.len() != TERMS {
            return Err(CudaError::LengthMismatch {
                expected: TERMS,
                got: left.len().min(right.len()),
            });
        }
        let mut claim = Fr::from(0u64);
        for term in 0..TERMS {
            claim += left[term] * right[term];
        }
        fr_into(claim).ok_or(CudaError::NotImplemented {
            kernel: "CUDA kernels support only the BN254 scalar field",
        })
    }

    pub fn val_claims<F: Field>(&self) -> Result<Vec<F>, CudaError> {
        if self.len != 1 {
            return Err(CudaError::LengthMismatch {
                expected: 1,
                got: self.len,
            });
        }
        let right = self.right.to_host()?;
        if right.len() != TERMS {
            return Err(CudaError::LengthMismatch {
                expected: TERMS,
                got: right.len(),
            });
        }
        let int = self.int.first()?;
        let unsupported = || CudaError::NotImplemented {
            kernel: "CUDA kernels support only the BN254 scalar field",
        };
        (0..STAGES)
            .map(|stage| fr_into(right[stage] - self.raf[stage] * int).ok_or_else(unsupported))
            .collect()
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

    use super::{DeviceBytecodePushforward, PushforwardInputs, STAGES, TERMS};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::one_hot_fold::{DeviceOneHotColumns, OneHotShards};
    use crate::cuda::common::testing::fr;

    fn pc_column(seed: u64, cycles: usize, addresses: usize) -> Vec<u32> {
        (0..cycles)
            .map(|cycle| {
                let mixed = seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(cycle as u64);
                ((mixed ^ (mixed >> 29)) % (addresses as u64 - 1)) as u32
            })
            .collect()
    }

    struct Oracle {
        pushforwards: [Polynomial<Fr>; STAGES],
        values: [Polynomial<Fr>; STAGES],
        int: Polynomial<Fr>,
        entry_trace: Polynomial<Fr>,
        entry_expected: Polynomial<Fr>,
        stage_weights: [Fr; STAGES],
        raf_weights: [Fr; STAGES],
        entry_weight: Fr,
    }

    impl Oracle {
        fn new(
            pcs: &[u32],
            stage_cycle_points: &[Vec<Fr>; STAGES],
            stage_values: &[[Fr; STAGES]],
            entry_trace_index: usize,
            entry_expected_index: usize,
            gamma: Fr,
        ) -> Self {
            let addresses = stage_values.len();
            let mut powers = [Fr::from_u64(1); 8];
            for index in 1..8 {
                powers[index] = powers[index - 1] * gamma;
            }
            let pushforwards = core::array::from_fn(|stage| {
                let eq = EqPolynomial::new(stage_cycle_points[stage].clone()).evaluations();
                let mut table = vec![Fr::from_u64(0); addresses];
                for (cycle, &pc) in pcs.iter().enumerate() {
                    table[pc as usize] += eq[cycle];
                }
                Polynomial::new(table)
            });
            let values = core::array::from_fn(|stage| {
                Polynomial::new(stage_values.iter().map(|row| row[stage]).collect())
            });
            let one_hot = |index: usize| {
                let mut table = vec![Fr::from_u64(0); addresses];
                table[index] = Fr::from_u64(1);
                Polynomial::new(table)
            };
            Self {
                pushforwards,
                values,
                int: Polynomial::new((0..addresses).map(|k| Fr::from_u64(k as u64)).collect()),
                entry_trace: one_hot(entry_trace_index),
                entry_expected: one_hot(entry_expected_index),
                stage_weights: core::array::from_fn(|stage| powers[stage]),
                raf_weights: [
                    powers[5],
                    Fr::from_u64(0),
                    powers[4],
                    Fr::from_u64(0),
                    Fr::from_u64(0),
                ],
                entry_weight: powers[7],
            }
        }

        fn lanes(&self) -> (Fr, Fr) {
            let half = self.int.evals().len() / 2;
            let mut out = [Fr::from_u64(0); 2];
            for (lane, point) in [Fr::from_u64(0), Fr::from_u64(2)].into_iter().enumerate() {
                let extend = |table: &Polynomial<Fr>, y: usize| {
                    table.sumcheck_round_eval_with_order(y, point, BindingOrder::LowToHigh)
                };
                let mut sum = Fr::from_u64(0);
                for y in 0..half {
                    let int = extend(&self.int, y);
                    for stage in 0..STAGES {
                        sum += self.stage_weights[stage]
                            * extend(&self.pushforwards[stage], y)
                            * (extend(&self.values[stage], y) + self.raf_weights[stage] * int);
                    }
                    sum += self.entry_weight
                        * extend(&self.entry_trace, y)
                        * extend(&self.entry_expected, y);
                }
                out[lane] = sum;
            }
            (out[0], out[1])
        }

        fn bind(&mut self, challenge: Fr) {
            for table in self
                .pushforwards
                .iter_mut()
                .chain(self.values.iter_mut())
                .chain([
                    &mut self.int,
                    &mut self.entry_trace,
                    &mut self.entry_expected,
                ])
            {
                table.bind_with_order(challenge, BindingOrder::LowToHigh);
            }
        }

        fn val_claims(&self) -> Vec<Fr> {
            self.values
                .iter()
                .map(|table| table.evals()[0])
                .collect::<Vec<_>>()
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(6))]
        #[test]
        fn bytecode_pushforward_matches_the_separate_table_form_round_for_round(
            log_t in 4usize..9,
            log_k in 3usize..6,
            seed in any::<u64>(),
            gamma in any::<u64>().prop_map(fr),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let cycles = 1usize << log_t;
            let addresses = 1usize << log_k;
            let pcs = pc_column(seed, cycles, addresses);
            let stage_cycle_points: [Vec<Fr>; STAGES] = core::array::from_fn(|stage| {
                (0..log_t)
                    .map(|i| fr(seed ^ (0x9E37 * stage as u64 + i as u64 + 1)))
                    .collect()
            });
            let stage_values: Vec<[Fr; STAGES]> = (0..addresses)
                .map(|k| {
                    core::array::from_fn(|stage| fr(seed ^ (0x51ED * stage as u64 + k as u64 + 7)))
                })
                .collect();
            let entry_trace_index = pcs[0] as usize;
            let entry_expected_index = (seed as usize) % addresses;
            let challenges: Vec<Fr> =
                (0..log_k).map(|i| fr(seed ^ (i as u64 * 97 + 41))).collect();

            let uploaded = DeviceOneHotColumns::new(
                context, &[], &[], &pcs, [0, 0, 1], log_k, cycles,
            )
            .expect("upload the PC column");
            let mut got = DeviceBytecodePushforward::new(
                context,
                &OneHotShards::single(uploaded),
                PushforwardInputs {
                    stage_cycle_points: &stage_cycle_points,
                    stage_values: &stage_values,
                    entry_trace_index,
                    entry_expected_index,
                    gamma,
                },
            )
            .expect("device bytecode pushforward");
            let mut expected = Oracle::new(
                &pcs,
                &stage_cycle_points,
                &stage_values,
                entry_trace_index,
                entry_expected_index,
                gamma,
            );

            for (round, &challenge) in challenges.iter().enumerate() {
                prop_assert_eq!(
                    got.round_lanes::<Fr>(context).expect("device round lanes"),
                    expected.lanes(),
                    "round {} lanes diverged", round
                );
                got.bind(context, challenge).expect("device bind");
                expected.bind(challenge);
            }

            prop_assert_eq!(got.len(), 1, "the tables never bound down to one address");
            prop_assert_eq!(
                got.val_claims::<Fr>().expect("device val claims"),
                expected.val_claims(),
                "the raw bound Val claims diverged"
            );
        }
    }

    #[test]
    fn bytecode_pushforward_rejects_a_short_stage_value_table() {
        let Some(context) = shared_context() else {
            return;
        };
        let log_t = 5;
        let log_k = 4;
        let cycles = 1usize << log_t;
        let addresses = 1usize << log_k;
        let pcs = pc_column(11, cycles, addresses);
        let uploaded = DeviceOneHotColumns::new(context, &[], &[], &pcs, [0, 0, 1], log_k, cycles)
            .expect("upload the PC column");
        let stage_cycle_points: [Vec<Fr>; STAGES] =
            core::array::from_fn(|_| (0..log_t).map(|i| fr(i as u64 + 3)).collect());
        let stage_values: Vec<[Fr; STAGES]> = (0..addresses - 1)
            .map(|k| core::array::from_fn(|_| fr(k as u64)))
            .collect();
        assert!(
            DeviceBytecodePushforward::new(
                context,
                &OneHotShards::single(uploaded),
                PushforwardInputs {
                    stage_cycle_points: &stage_cycle_points,
                    stage_values: &stage_values,
                    entry_trace_index: 0,
                    entry_expected_index: 0,
                    gamma: fr(5),
                },
            )
            .is_err(),
            "a stage-value table shorter than the padded bytecode must not be padded silently",
        );
    }

    #[test]
    fn bytecode_pushforward_carries_one_term_per_stage_plus_the_entry() {
        assert_eq!(TERMS, STAGES + 1);
    }
}
