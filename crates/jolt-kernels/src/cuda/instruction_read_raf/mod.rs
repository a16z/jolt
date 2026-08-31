pub(super) mod address_driver;
pub(super) mod address_phase;
mod combine;
mod cycle_handoff;
mod cycle_rounds;
mod prefixes;
mod suffixes;

use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::instruction::CANONICAL_INSTRUCTION_ADDRESS;
use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionReadRafInputClaims, InstructionReadRafOutputClaims,
};
use jolt_field::{Field, Fr};
use jolt_lookup_tables::lookup_bits::LookupBits;
use jolt_lookup_tables::tables::prefixes::{PrefixEval, ALL_PREFIXES};
use jolt_lookup_tables::tables::suffixes::SuffixEval;
use jolt_lookup_tables::tables::LookupTableKind;
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::stage5::instruction_read_raf::InstructionReadRaf;
use jolt_witness::backend::cuda::FLAG_BIT_RAF;
use jolt_witness::JoltWitnessPlane;

use crate::cuda::witness::session_window_residency;

use self::address_driver::{AddressShard, DeviceAddressPhase};
use self::address_phase::{flag_claims, DeviceRows, NO_TABLE};
use self::cycle_handoff::{build_cycle_tables, HandoffInputs};
use self::cycle_rounds::{CycleShard, DeviceCycleRounds};
use super::{require_context, CudaBackend};
use crate::cuda::common::context::{context_for, CudaKernelContext};
use crate::cuda::common::device::{fr_into, require_fr, require_fr_slice, DeviceFrVec};
use crate::cuda::common::device_columns::device_trace_columns;
use crate::cuda::common::devices::{fan_out, witness_windows, DeviceTask};
use crate::cuda::common::error::backend;
use crate::cuda::common::error::CudaError;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const CHUNK_LEN: usize = 8;
const ADDRESS_BITS: usize = 128;
const RAF_CHECKPOINTS: usize = 4;
const HINT_POINTS: usize = 2;

fn raf_initial_checkpoints<F: Field>() -> [F; RAF_CHECKPOINTS] {
    let mut checkpoints = [F::zero(); RAF_CHECKPOINTS];
    if CANONICAL_INSTRUCTION_ADDRESS {
        checkpoints[3] = F::one();
    }
    checkpoints
}

pub(crate) struct RowShard {
    pub(crate) ordinal: usize,
    pub(crate) rows: Arc<DeviceRows>,
    pub(crate) cycles: usize,
}

type PreparedWindow = (
    RowShard,
    AddressShard,
    [bool; LookupTableKind::<RISCV_XLEN>::COUNT],
);

pub struct DeviceInstructionReadRaf<F: Field> {
    device: Option<DeviceAddressPhase>,
    cycle: Option<DeviceCycleRounds>,
    row_shards: Vec<RowShard>,
    r_reduction: Vec<Fr>,
    cycle_challenges: Vec<Fr>,
    prefix_checkpoints: Vec<PrefixEval<F>>,
    raf_checkpoints: [F; RAF_CHECKPOINTS],
    ra_count: usize,
    rounds: usize,
    gamma: F,
    context: &'static CudaKernelContext,
    rounds_bound: usize,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for DeviceInstructionReadRaf<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("r_reduction"),
            self.r_reduction.len() * size_of::<Fr>(),
        );
        visitor.visit_simple(
            allocative::Key::new("cycle_challenges"),
            self.cycle_challenges.len() * size_of::<Fr>(),
        );
        visitor.visit_simple(
            allocative::Key::new("prefix_checkpoints"),
            self.prefix_checkpoints.len() * size_of::<PrefixEval<F>>(),
        );
        visitor.exit();
    }
}

impl<F: Field> DeviceInstructionReadRaf<F> {
    fn field(value: jolt_field::Fr) -> Result<F, SumcheckError<F>> {
        fr_into(value).ok_or(SumcheckError::MissingEvaluationSource {
            kind: "cuda instruction read-RAF field",
        })
    }

    fn window_flag_claims(&self, r_cycle: &[Fr]) -> Result<(Vec<Fr>, Fr), CudaError> {
        let count = self.row_shards.len();
        let tasks: Vec<DeviceTask<'_, (Vec<Fr>, Fr), CudaError>> = self
            .row_shards
            .iter()
            .enumerate()
            .map(|(index, shard)| {
                let task: DeviceTask<'_, (Vec<Fr>, Fr), CudaError> = Box::new(move || {
                    let device =
                        context_for(shard.ordinal).ok_or(CudaError::InvariantViolation {
                            reason: "an instruction read-RAF flag window names an absent device",
                        })?;
                    let eq = device.eq_evals_shard(r_cycle, index, count)?;
                    flag_claims(
                        device,
                        &shard.rows,
                        shard.cycles,
                        &eq,
                        LookupTableKind::<RISCV_XLEN>::COUNT,
                    )
                });
                task
            })
            .collect();
        let mut flags: Vec<Fr> = Vec::new();
        let mut raf = Fr::from(0u64);
        for (part, part_raf) in fan_out(tasks)? {
            if flags.is_empty() {
                flags = part;
            } else {
                if part.len() != flags.len() {
                    return Err(CudaError::LengthMismatch {
                        expected: flags.len(),
                        got: part.len(),
                    });
                }
                for (slot, value) in flags.iter_mut().zip(&part) {
                    *slot += *value;
                }
            }
            raf += part_raf;
        }
        Ok((flags, raf))
    }

    fn enter_cycle_rounds(&mut self) -> Result<(), SumcheckError<F>> {
        let device = self
            .device
            .take()
            .ok_or(SumcheckError::MissingEvaluationSource {
                kind: "cuda address phase",
            })?;
        let failed = || SumcheckError::MissingEvaluationSource {
            kind: "cuda address-phase handoff",
        };

        let prefix_checkpoints = device
            .checkpoints(self.context)
            .map_err(backend("cuda address-phase handoff"))?;
        let prefix_checkpoints: Vec<F> = prefix_checkpoints
            .into_iter()
            .map(Self::field)
            .collect::<Result<_, _>>()?;

        let raf = device
            .raf_checkpoints(self.context)
            .map_err(backend("cuda address-phase handoff"))?;
        if raf.len() != RAF_CHECKPOINTS {
            return Err(failed());
        }
        let mut raf_checkpoints = [F::zero(); RAF_CHECKPOINTS];
        for (slot, value) in raf_checkpoints.iter_mut().zip(raf) {
            *slot = Self::field(value)?;
        }

        if prefix_checkpoints.len() != self.prefix_checkpoints.len()
            || device.v_tables().len() != ADDRESS_BITS / CHUNK_LEN
        {
            return Err(failed());
        }

        self.prefix_checkpoints = prefix_checkpoints
            .into_iter()
            .map(PrefixEval::from)
            .collect();
        self.raf_checkpoints = raf_checkpoints;

        let gamma_sqr = self.gamma * self.gamma;
        let empty = LookupBits::new(0, 0);
        let table_values: Vec<Fr> = LookupTableKind::<RISCV_XLEN>::iter()
            .map(|table| {
                let suffixes: Vec<SuffixEval<F>> = table
                    .suffixes()
                    .iter()
                    .map(|suffix| SuffixEval::from(F::from_u64(suffix.suffix_mle(empty))))
                    .collect();
                require_fr(table.combine(&self.prefix_checkpoints, &suffixes))
            })
            .collect::<Result<_, _>>()
            .map_err(backend("cuda address-phase handoff"))?;
        let raf_interleaved =
            self.gamma * self.raf_checkpoints[0] + gamma_sqr * self.raf_checkpoints[1];
        let mut raf_identity = gamma_sqr * self.raf_checkpoints[2];
        if CANONICAL_INSTRUCTION_ADDRESS {
            raf_identity += gamma_sqr * self.gamma * self.raf_checkpoints[3];
        }

        let raf_interleaved =
            require_fr(raf_interleaved).map_err(backend("cuda address-phase handoff"))?;
        let raf_identity =
            require_fr(raf_identity).map_err(backend("cuda address-phase handoff"))?;
        let host_v: Vec<Vec<Fr>> = device
            .v_tables()
            .iter()
            .map(DeviceFrVec::to_host)
            .collect::<Result<_, _>>()
            .map_err(backend("cuda address-phase handoff"))?;

        let host_v = &host_v;
        let table_values = &table_values;
        let ra_count = self.ra_count;
        let tasks: Vec<DeviceTask<'_, CycleShard, CudaError>> = self
            .row_shards
            .iter()
            .map(|shard| {
                let task: DeviceTask<'_, CycleShard, CudaError> = Box::new(move || {
                    let device =
                        context_for(shard.ordinal).ok_or(CudaError::InvariantViolation {
                            reason: "an instruction read-RAF cycle window names an absent device",
                        })?;
                    let v_tables: Vec<DeviceFrVec> = host_v
                        .iter()
                        .map(|table| device.upload(table))
                        .collect::<Result<_, _>>()?;
                    let tables = build_cycle_tables(
                        device,
                        &HandoffInputs {
                            rows: &shard.rows,
                            cycles: shard.cycles,
                            v_tables: &v_tables,
                            table_values,
                            raf_interleaved,
                            raf_identity,
                            ra_count,
                            address_bits: ADDRESS_BITS,
                        },
                    )?;
                    let mut factors = Vec::with_capacity(tables.ra.len() + 1);
                    factors.push(tables.combined_val);
                    factors.extend(tables.ra);
                    Ok(CycleShard {
                        ordinal: shard.ordinal,
                        factors,
                    })
                });
                task
            })
            .collect();
        let shards = fan_out(tasks).map_err(backend("cuda address-phase handoff"))?;

        self.cycle = Some(
            DeviceCycleRounds::from_windows(&self.r_reduction, shards, self.rounds - ADDRESS_BITS)
                .map_err(backend("cuda address-phase handoff"))?,
        );
        Ok(())
    }
}

impl<F: Field> PrepareKernel<F, InstructionReadRaf<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionReadRaf<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionReadRaf<F>>>, KernelError<F>> {
        let context = require_context()?;
        let dimensions = inputs.relation.dimensions();
        if dimensions.instruction_address_bits() != ADDRESS_BITS
            || !ADDRESS_BITS.is_multiple_of(CHUNK_LEN)
        {
            return Err(KernelError::Unsupported {
                reason: "the CUDA instruction read-RAF address phase supports only the \
                         2·XLEN interleaved-operand address width in 8-variable phases",
            });
        }
        let cycles = 1usize << dimensions.log_t();
        let unsupported = || KernelError::Unsupported {
            reason: "the CUDA instruction read-RAF kernel supports only the BN254 scalar field",
        };
        let r_reduction = require_fr_slice(&inputs.points.lookup_output)
            .map_err(|_| unsupported())?
            .to_vec();

        let windows = witness_windows(cycles);
        let shards = windows.len();
        let mut resident = Vec::with_capacity(shards);
        for (ordinal, window) in windows.iter().enumerate() {
            let shard = context_for(ordinal).ok_or(KernelError::InvariantViolation {
                reason: "an instruction read-RAF cycle window names an absent device",
            })?;
            let (trace, atoms) = session_window_residency(shard, session, witness, cycles, window)?;
            let limbs =
                device_trace_columns::<F>(shard, session, witness, cycles, window, [1, 0, 0], 0)?
                    .lookup;
            resident.push((ordinal, shard, trace, atoms, limbs));
        }

        let reduction: &[Fr] = &r_reduction;
        let tasks: Vec<DeviceTask<'_, PreparedWindow, KernelError<F>>> = resident
            .into_iter()
            .zip(windows.iter())
            .map(|((ordinal, shard, trace, atoms, limbs), window)| {
                let task: DeviceTask<'_, PreparedWindow, KernelError<F>> =
                    Box::new(move || {
                        let flags = trace
                            .flag_bit_bytes(&atoms.flags, FLAG_BIT_RAF)
                            .map_err(|_| unsupported())?;
                        let table_index = shard.download_u32(&atoms.table_index)?;
                        let mut used = [false; LookupTableKind::<RISCV_XLEN>::COUNT];
                        for &index in table_index
                            .get(..window.len)
                            .ok_or(KernelError::InvariantViolation {
                            reason:
                                "an instruction read-RAF window has fewer table slots than cycles",
                        })? {
                            if index != NO_TABLE {
                                *used.get_mut(index as usize).ok_or(
                                    KernelError::InvariantViolation {
                                        reason: "a stage-5 row selects an unknown lookup table",
                                    },
                                )? = true;
                            }
                        }
                        let rows = Arc::new(
                            DeviceRows::from_device_columns(
                                limbs,
                                shard.clone_u32(&atoms.table_index)?,
                                flags,
                                window.len,
                            )
                            .map_err(|_| unsupported())?,
                        );
                        Ok((
                            RowShard {
                                ordinal,
                                rows: Arc::clone(&rows),
                                cycles: window.len,
                            },
                            AddressShard {
                                ordinal,
                                rows,
                                u_evals: shard
                                    .eq_evals_shard(reduction, ordinal, shards)
                                    .map_err(|_| unsupported())?,
                            },
                            used,
                        ))
                    });
                task
            })
            .collect();

        let mut row_shards = Vec::with_capacity(shards);
        let mut address_shards = Vec::with_capacity(shards);
        let mut used = [false; LookupTableKind::<RISCV_XLEN>::COUNT];
        for (row, address, window_used) in fan_out(tasks)? {
            row_shards.push(row);
            address_shards.push(address);
            for (slot, hit) in used.iter_mut().zip(window_used) {
                *slot |= hit;
            }
        }

        let device = DeviceAddressPhase::with_shards(context, address_shards, &used, ADDRESS_BITS)
            .map_err(|_| unsupported())?;

        Ok(Box::new(DeviceInstructionReadRaf {
            device: Some(device),
            cycle: None,
            row_shards,
            r_reduction,
            cycle_challenges: Vec::with_capacity(dimensions.log_t()),
            prefix_checkpoints: ALL_PREFIXES
                .iter()
                .map(|prefix| prefix.default_checkpoint::<F>())
                .collect(),
            raf_checkpoints: raf_initial_checkpoints(),
            ra_count: dimensions.num_virtual_ra_polys(),
            rounds: dimensions.sumcheck_rounds(),
            gamma: inputs.challenges.gamma,
            context,
            rounds_bound: 0,
        }))
    }
}

impl<F: Field> ProveRounds<F> for DeviceInstructionReadRaf<F> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge)?;
        }
        if self.rounds_bound < ADDRESS_BITS {
            let device = self
                .device
                .as_ref()
                .ok_or(SumcheckError::MissingEvaluationSource {
                    kind: "cuda address phase",
                })?;
            let evals = device
                .round_message_hinted(
                    self.context,
                    require_fr(self.gamma).map_err(|_| SumcheckError::MissingEvaluationSource {
                        kind: "cuda address gamma",
                    })?,
                    require_fr(previous_claim).map_err(|_| {
                        SumcheckError::MissingEvaluationSource {
                            kind: "cuda address claim hint",
                        }
                    })?,
                )
                .map_err(|_| SumcheckError::MissingEvaluationSource {
                    kind: "cuda address round message",
                })?;
            let mut host = [F::zero(); HINT_POINTS];
            for (slot, value) in host.iter_mut().zip(evals) {
                *slot = Self::field(value)?;
            }
            return Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &host));
        }

        let cycle = self
            .cycle
            .as_ref()
            .ok_or(SumcheckError::MissingEvaluationSource {
                kind: "cuda cycle rounds",
            })?;
        cycle
            .round_message(self.context, previous_claim)
            .map_err(|_| SumcheckError::MissingEvaluationSource {
                kind: "cuda cycle round message",
            })
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: Field> DeviceInstructionReadRaf<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        if self.rounds_bound < ADDRESS_BITS {
            let scalar =
                require_fr(challenge).map_err(|_| SumcheckError::MissingEvaluationSource {
                    kind: "cuda address-phase challenge",
                })?;
            self.device
                .as_mut()
                .ok_or(SumcheckError::MissingEvaluationSource {
                    kind: "cuda address phase",
                })?
                .bind(self.context, scalar)
                .map_err(|_| SumcheckError::MissingEvaluationSource {
                    kind: "cuda address-phase bind",
                })?;
            self.rounds_bound += 1;
            if self.rounds_bound == ADDRESS_BITS {
                self.enter_cycle_rounds()?;
            }
            Ok(())
        } else {
            let scalar =
                require_fr(challenge).map_err(|_| SumcheckError::MissingEvaluationSource {
                    kind: "cuda cycle-round challenge",
                })?;
            self.cycle
                .as_mut()
                .ok_or(SumcheckError::MissingEvaluationSource {
                    kind: "cuda cycle rounds",
                })?
                .bind(self.context, scalar)
                .map_err(|_| SumcheckError::MissingEvaluationSource {
                    kind: "cuda cycle-round bind",
                })?;
            self.rounds_bound += 1;
            self.cycle_challenges
                .push(require_fr(challenge).map_err(|_| {
                    SumcheckError::MissingEvaluationSource {
                        kind: "cuda cycle-round challenge",
                    }
                })?);
            Ok(())
        }
    }
}

impl<F: Field> SumcheckKernel<F> for DeviceInstructionReadRaf<F> {
    type Relation = InstructionReadRaf<F>;

    fn output_claims(
        &mut self,
        _inputs: &InstructionReadRafInputClaims<F>,
    ) -> Result<InstructionReadRafOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.rounds - self.rounds_bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let cycle = self
            .cycle
            .as_ref()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "cycle rounds absent after full binding",
            })?;
        let instruction_ra: Vec<F> =
            cycle
                .ra_finals(self.context)
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "CUDA instruction RA claim readback failed",
                })?;
        let r_cycle: Vec<Fr> = self.cycle_challenges.iter().rev().copied().collect();
        let (flags, raf_flag) = self.window_flag_claims(&r_cycle).map_err(|_| {
            SumcheckKernelError::InvariantViolation {
                reason: "CUDA flag claim readback failed",
            }
        })?;
        let lookup_table_flags: Vec<F> = flags
            .into_iter()
            .map(|value| {
                fr_into(value).ok_or(SumcheckKernelError::InvariantViolation {
                    reason: "CUDA kernels support only the BN254 scalar field",
                })
            })
            .collect::<Result<_, _>>()?;
        let instruction_raf_flag =
            fr_into(raf_flag).ok_or(SumcheckKernelError::InvariantViolation {
                reason: "CUDA kernels support only the BN254 scalar field",
            })?;
        Ok(InstructionReadRafOutputClaims {
            lookup_table_flags,
            instruction_ra,
            instruction_raf_flag,
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use std::num::NonZeroUsize;

    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_claims::protocols::jolt::relations::instruction::{
        InstructionReadRafChallenges, InstructionReadRafInputClaims,
    };
    use jolt_claims::protocols::jolt::JoltOneHotConfig;
    use jolt_claims::OutputClaims;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_riscv::JoltInstructionKind;
    use jolt_verifier::stages::stage5::InstructionReadRaf;
    use proptest::prelude::*;

    use super::{CudaBackend, ADDRESS_BITS};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{
        arb_point, drive, fr, reference_input_claim, with_instruction_kind_witness,
    };
    use crate::optimized::instruction_read_raf::OptimizedInstructionReadRaf;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    const LOG_T: usize = 8;
    const RA_POLYS: usize = 8;

    const SWEPT_KINDS: [JoltInstructionKind; 3] = [
        JoltInstructionKind::XOR,
        JoltInstructionKind::VirtualSRAI,
        JoltInstructionKind::SLT,
    ];

    fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 16,
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2))]
        #[test]
        fn instruction_read_raf_matches_optimized_round_for_round(
            seed in any::<u64>(),
            lookup_output in arb_point(LOG_T),
            gamma in any::<u64>().prop_map(fr),
            challenges in arb_point(ADDRESS_BITS + LOG_T),
            kind in prop::sample::select(SWEPT_KINDS.to_vec()),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            let dimensions = InstructionReadRafDimensions::new(
                LOG_T,
                ADDRESS_BITS,
                NonZeroUsize::new(RA_POLYS).expect("a nonzero virtual ra count"),
            );
            let relation = InstructionReadRaf::<Fr>::new(dimensions);
            let claims = InstructionReadRafInputClaims {
                lookup_output: Fr::from_u64(0),
                left_lookup_operand: Fr::from_u64(0),
                right_lookup_operand: Fr::from_u64(0),
            };
            let points = InstructionReadRafInputClaims {
                lookup_output: lookup_output.clone(),
                left_lookup_operand: lookup_output.clone(),
                right_lookup_operand: lookup_output.clone(),
            };
            let challenge_set = InstructionReadRafChallenges { gamma };

            with_instruction_kind_witness(kind, LOG_T, one_hot(), seed, |witness| {
                let make_inputs = || ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenge_set,
                };

                let input_claim = reference_input_claim(witness, make_inputs);
                let mut expected_kernel = OptimizedInstructionReadRaf
                    .prepare(&mut ProofSession::default(), witness, make_inputs())
                    .expect("optimized prepare");
                let mut got_kernel = CudaBackend
                    .prepare(&mut ProofSession::default(), witness, make_inputs())
                    .expect("cuda prepare");

                let expected = drive(&mut *expected_kernel, input_claim, &challenges);
                let got = drive(&mut *got_kernel, input_claim, &challenges);
                prop_assert_eq!(got, expected, "round polynomials diverged for {:?}", kind);

                let expected_claims = expected_kernel
                    .output_claims(&claims)
                    .expect("optimized claims");
                let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
                prop_assert_eq!(
                    got_claims.opening_values(),
                    expected_claims.opening_values(),
                    "output claims diverged"
                );
                Ok(())
            })?;
        }
    }
}
