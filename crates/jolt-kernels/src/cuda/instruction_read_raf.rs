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
use jolt_witness::{collect_bundles, JoltWitnessPlane};

use super::address_driver::DeviceAddressPhase;
use super::address_phase::{flag_claims, DeviceRows};
use super::context::CudaKernelContext;
use super::cycle_handoff::{build_cycle_tables, HandoffInputs};
use super::cycle_rounds::DeviceCycleRounds;
use super::device::{fr_into, require_fr, require_fr_slice};
use super::{require_context, CudaBackend};
use crate::reference::instruction_read_raf::InstructionReadRafWitness;
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

pub struct DeviceInstructionReadRaf<F: Field> {
    device: Option<DeviceAddressPhase>,
    cycle: Option<DeviceCycleRounds>,
    rows: DeviceRows,
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

impl<F: Field> DeviceInstructionReadRaf<F> {
    fn field(value: jolt_field::Fr) -> Result<F, SumcheckError<F>> {
        fr_into(value).ok_or(SumcheckError::MissingEvaluationSource {
            kind: "cuda instruction read-RAF field",
        })
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

        let prefix_checkpoints = device.checkpoints(self.context).map_err(|_| failed())?;
        let prefix_checkpoints: Vec<F> = prefix_checkpoints
            .into_iter()
            .map(Self::field)
            .collect::<Result<_, _>>()?;

        let raf = device.raf_checkpoints(self.context).map_err(|_| failed())?;
        if raf.len() != RAF_CHECKPOINTS {
            return Err(failed());
        }
        let mut raf_checkpoints = [F::zero(); RAF_CHECKPOINTS];
        for (slot, value) in raf_checkpoints.iter_mut().zip(raf) {
            *slot = Self::field(value)?;
        }

        let mut v_tables = Vec::with_capacity(device.v_tables().len());
        for table in device.v_tables() {
            let host: Vec<F> = table
                .to_host()
                .map_err(|_| failed())?
                .into_iter()
                .map(Self::field)
                .collect::<Result<_, _>>()?;
            v_tables.push(host);
        }
        if prefix_checkpoints.len() != self.prefix_checkpoints.len()
            || v_tables.len() != ADDRESS_BITS / CHUNK_LEN
        {
            return Err(failed());
        }

        self.prefix_checkpoints = prefix_checkpoints
            .into_iter()
            .map(PrefixEval::from)
            .collect();
        self.raf_checkpoints = raf_checkpoints;
        let _ = v_tables;

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
            .map_err(|_| failed())?;
        let raf_interleaved =
            self.gamma * self.raf_checkpoints[0] + gamma_sqr * self.raf_checkpoints[1];
        let mut raf_identity = gamma_sqr * self.raf_checkpoints[2];
        if CANONICAL_INSTRUCTION_ADDRESS {
            raf_identity += gamma_sqr * self.gamma * self.raf_checkpoints[3];
        }

        let tables = build_cycle_tables(
            self.context,
            &HandoffInputs {
                rows: &self.rows,
                v_tables: device.v_tables(),
                table_values: &table_values,
                raf_interleaved: require_fr(raf_interleaved).map_err(|_| failed())?,
                raf_identity: require_fr(raf_identity).map_err(|_| failed())?,
                ra_count: self.ra_count,
                address_bits: ADDRESS_BITS,
            },
        )
        .map_err(|_| failed())?;

        let eq_reduction = self
            .context
            .eq_evals(&self.r_reduction)
            .map_err(|_| failed())?;
        self.cycle = Some(
            DeviceCycleRounds::from_device(
                eq_reduction,
                tables.combined_val,
                tables.ra,
                self.rounds - ADDRESS_BITS,
            )
            .map_err(|_| failed())?,
        );
        Ok(())
    }
}

impl<F: Field> PrepareKernel<F, InstructionReadRaf<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
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
        let rows: Vec<InstructionReadRafWitness> =
            collect_bundles(witness, 1 << dimensions.log_t())?;

        let lookup_index: Vec<u128> = rows.iter().map(|row| row.lookup_index.0).collect();
        let table_index: Vec<Option<usize>> = rows.iter().map(|row| row.table_index.0).collect();
        let raf_flag: Vec<bool> = rows.iter().map(|row| row.raf_flag.0).collect();

        let unsupported = || KernelError::Unsupported {
            reason: "the CUDA instruction read-RAF kernel supports only the BN254 scalar field",
        };
        let device = DeviceAddressPhase::new(
            context,
            &lookup_index,
            &table_index,
            &raf_flag,
            &inputs.points.lookup_output,
            ADDRESS_BITS,
        )
        .map_err(|_| unsupported())?;

        let r_reduction = require_fr_slice(&inputs.points.lookup_output)
            .map_err(|_| unsupported())?
            .to_vec();
        let device_rows = DeviceRows::new(context, &lookup_index, &table_index, &raf_flag)
            .map_err(|_| unsupported())?;
        Ok(Box::new(DeviceInstructionReadRaf {
            device: Some(device),
            cycle: None,
            rows: device_rows,
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
        round: usize,
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
        let evals = cycle.round_message(self.context).map_err(|_| {
            SumcheckError::MissingEvaluationSource {
                kind: "cuda cycle round message",
            }
        })?;
        let round_sum = evals[0] + evals[1];
        if round_sum != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: round_sum,
            });
        }
        Ok(UnivariatePoly::from_evals(&evals))
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
        let eq_cycle = self.context.eq_evals(&r_cycle).map_err(|_| {
            SumcheckKernelError::InvariantViolation {
                reason: "CUDA cycle eq table construction failed",
            }
        })?;
        let (flags, raf_flag) = flag_claims(
            self.context,
            &self.rows,
            &eq_cycle,
            LookupTableKind::<RISCV_XLEN>::COUNT,
        )
        .map_err(|_| SumcheckKernelError::InvariantViolation {
            reason: "CUDA flag claim readback failed",
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
    clippy::unwrap_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafChallenges;
    use jolt_claims::SumcheckChallenges;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_program::execution::TraceRow;
    use jolt_program::execution::{RegisterRead, RegisterState, RegisterWrite};
    use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands};
    use proptest::prelude::*;
    use std::num::NonZeroUsize;

    use super::super::context::shared_context;
    use super::super::testing::{arb_point, drive, fr, reference_input_claim, RowPlane};
    use super::super::CudaBackend;
    use super::{InstructionReadRaf, InstructionReadRafInputClaims, ADDRESS_BITS};
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    const KINDS: [JoltInstructionKind; 5] = [
        JoltInstructionKind::OR,
        JoltInstructionKind::SLTU,
        JoltInstructionKind::ADD,
        JoltInstructionKind::SUB,
        JoltInstructionKind::MUL,
    ];

    fn trace(log_t: usize, seed: u64) -> Vec<TraceRow> {
        (0..1usize << log_t)
            .map(|j| {
                let mixed = (j as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(seed);
                TraceRow {
                    instruction: JoltInstructionRow {
                        instruction_kind: KINDS[(mixed % KINDS.len() as u64) as usize],
                        address: 0,
                        operands: NormalizedOperands {
                            rd: Some(1),
                            rs1: Some(2),
                            rs2: Some(3),
                            imm: 0,
                        },
                        virtual_sequence_remaining: None,
                        is_first_in_sequence: false,
                        is_compressed: false,
                    },
                    registers: RegisterState {
                        rs1: Some(RegisterRead {
                            register: 2,
                            value: mixed,
                        }),
                        rs2: Some(RegisterRead {
                            register: 3,
                            value: mixed.rotate_left(17),
                        }),
                        rd: Some(RegisterWrite {
                            register: 1,
                            pre_value: 0,
                            post_value: 0,
                        }),
                    },
                    ..TraceRow::default()
                }
            })
            .collect()
    }

    proptest! {
        #[test]
        fn instruction_read_raf_matches_reference(
            log_t in 4usize..=7,
            seed in any::<u64>(),
            gamma in any::<u64>().prop_map(fr),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };
            let witness = RowPlane::new(trace(log_t, seed), log_t);
            let dimensions = InstructionReadRafDimensions::new(
                log_t,
                ADDRESS_BITS,
                NonZeroUsize::new(8).unwrap(),
            );
            let relation = InstructionReadRaf::<Fr>::new(dimensions);
            let challenge_set =
                InstructionReadRafChallenges::from_transcript_values([gamma].into_iter())
                    .expect("challenges");
            let claims = InstructionReadRafInputClaims {
                lookup_output: Fr::from_u64(0),
                left_lookup_operand: Fr::from_u64(0),
                right_lookup_operand: Fr::from_u64(0),
            };
            let r_cycle: Vec<Fr> = (0..log_t).map(|i| fr(seed + i as u64 + 3)).collect();
            let points = InstructionReadRafInputClaims {
                lookup_output: r_cycle.clone(),
                left_lookup_operand: r_cycle.clone(),
                right_lookup_operand: r_cycle.clone(),
            };
            let make_inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenge_set,
            };

            let input_claim = reference_input_claim(&witness, make_inputs);
            let rounds = ADDRESS_BITS + log_t;
            let challenges: Vec<Fr> =
                (0..rounds).map(|i| fr(seed + i as u64 + 71)).collect();

            let mut host = ReferenceBackend
                .prepare(&mut ProofSession::default(), &witness, make_inputs())
                .expect("reference prepare");
            let mut device = CudaBackend
                .prepare(&mut ProofSession::default(), &witness, make_inputs())
                .expect("cuda prepare");

            let expected = drive(host.as_mut(), input_claim, &challenges);
            let got = drive(device.as_mut(), input_claim, &challenges);

            prop_assert_eq!(got.len(), expected.len());
            for (round, (got, want)) in got.iter().zip(&expected).enumerate() {
                prop_assert_eq!(
                    got.coefficients(),
                    want.coefficients(),
                    "round {} polynomial diverged",
                    round
                );
            }

            let want_claims = host.output_claims(&claims).expect("reference output claims");
            let got_claims = device.output_claims(&claims).expect("cuda output claims");
            prop_assert_eq!(
                got_claims.lookup_table_flags,
                want_claims.lookup_table_flags,
                "lookup table flag claims diverged"
            );
            prop_assert_eq!(
                got_claims.instruction_ra,
                want_claims.instruction_ra,
                "instruction ra claims diverged"
            );
            prop_assert_eq!(
                got_claims.instruction_raf_flag,
                want_claims.instruction_raf_flag,
                "instruction raf flag claim diverged"
            );
        }
    }
}
