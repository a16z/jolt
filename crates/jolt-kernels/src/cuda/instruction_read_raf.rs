use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionReadRafInputClaims, InstructionReadRafOutputClaims,
};
use jolt_claims::protocols::jolt::geometry::instruction::CANONICAL_INSTRUCTION_ADDRESS;
use jolt_field::{Field, Fr};
use jolt_lookup_tables::lookup_bits::LookupBits;
use jolt_lookup_tables::tables::prefixes::PrefixEval;
use jolt_lookup_tables::tables::suffixes::SuffixEval;
use jolt_lookup_tables::tables::LookupTableKind;
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::stage5::instruction_read_raf::InstructionReadRaf;
use jolt_witness::{collect_bundles, JoltWitnessPlane};

use super::address_driver::DeviceAddressPhase;
use super::context::CudaKernelContext;
use super::cycle_handoff::{build_cycle_tables, HandoffInputs};
use super::cycle_rounds::DeviceCycleRounds;
use super::device::{fr_into, require_fr, require_fr_slice};
use super::{require_context, CudaBackend};
use crate::reference::instruction_read_raf::{InstructionReadRafKernel, InstructionReadRafWitness};
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const CHUNK_LEN: usize = 8;
const ADDRESS_BITS: usize = 128;
const RAF_CHECKPOINTS: usize = 4;

pub struct DeviceInstructionReadRaf<F: Field> {
    host: InstructionReadRafKernel<F>,
    device: Option<DeviceAddressPhase>,
    cycle: Option<DeviceCycleRounds>,
    r_reduction: Vec<Fr>,
    ra_count: usize,
    gamma: Fr,
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
        if prefix_checkpoints.len() != self.host.prefix_checkpoints.len()
            || v_tables.len() != self.host.phases()
        {
            return Err(failed());
        }

        self.host.prefix_checkpoints = prefix_checkpoints
            .into_iter()
            .map(PrefixEval::from)
            .collect();
        self.host.raf_left.checkpoint = raf_checkpoints[0];
        self.host.raf_right.checkpoint = raf_checkpoints[1];
        self.host.raf_identity.checkpoint = raf_checkpoints[2];
        self.host.raf_upper_all_ones.checkpoint = raf_checkpoints[3];
        self.host.v_tables = v_tables;
        self.host.rounds_bound = ADDRESS_BITS;

        let gamma_sqr = self.host.gamma * self.host.gamma;
        let empty = LookupBits::new(0, 0);
        let table_values: Vec<Fr> = LookupTableKind::<RISCV_XLEN>::iter()
            .map(|table| {
                let suffixes: Vec<SuffixEval<F>> = table
                    .suffixes()
                    .iter()
                    .map(|suffix| SuffixEval::from(F::from_u64(suffix.suffix_mle(empty))))
                    .collect();
                require_fr(table.combine(self.host.prefix_checkpoints(), &suffixes))
            })
            .collect::<Result<_, _>>()
            .map_err(|_| failed())?;
        let raf_interleaved = self.host.gamma * self.host.raf_left.checkpoint
            + gamma_sqr * self.host.raf_right.checkpoint;
        let mut raf_identity = gamma_sqr * self.host.raf_identity.checkpoint;
        if CANONICAL_INSTRUCTION_ADDRESS {
            raf_identity +=
                gamma_sqr * self.host.gamma * self.host.raf_upper_all_ones.checkpoint;
        }

        let tables = build_cycle_tables(
            self.context,
            &HandoffInputs {
                rows: device.rows(),
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
                self.host.num_rounds() - ADDRESS_BITS,
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
        let gamma = require_fr(inputs.challenges.gamma).map_err(|_| unsupported())?;
        let device = DeviceAddressPhase::new(
            context,
            &lookup_index,
            &table_index,
            &raf_flag,
            &inputs.points.lookup_output,
            ADDRESS_BITS,
        )
        .map_err(|_| unsupported())?;

        let host = InstructionReadRafKernel::new(
            dimensions,
            &inputs.points.lookup_output,
            rows,
            inputs.challenges.gamma,
        )?;
        let r_reduction = require_fr_slice(&inputs.points.lookup_output)
            .map_err(|_| unsupported())?
            .to_vec();
        Ok(Box::new(DeviceInstructionReadRaf {
            host,
            device: Some(device),
            cycle: None,
            r_reduction,
            ra_count: dimensions.num_virtual_ra_polys(),
            gamma,
            context,
            rounds_bound: 0,
        }))
    }
}

impl<F: Field> ProveRounds<F> for DeviceInstructionReadRaf<F> {
    fn num_rounds(&self) -> usize {
        self.host.num_rounds()
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
        let evals = if self.rounds_bound < ADDRESS_BITS {
            let device = self
                .device
                .as_ref()
                .ok_or(SumcheckError::MissingEvaluationSource {
                    kind: "cuda address phase",
                })?;
            let evals = device
                .round_message(self.context, self.gamma)
                .map_err(|_| SumcheckError::MissingEvaluationSource {
                    kind: "cuda address round message",
                })?;
            let mut host = [F::zero(); 3];
            for (slot, value) in host.iter_mut().zip(evals) {
                *slot = Self::field(value)?;
            }
            host.to_vec()
        } else {
            let cycle = self
                .cycle
                .as_ref()
                .ok_or(SumcheckError::MissingEvaluationSource {
                    kind: "cuda cycle rounds",
                })?;
            cycle.round_message(self.context).map_err(|_| {
                SumcheckError::MissingEvaluationSource {
                    kind: "cuda cycle round message",
                }
            })?
        };
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
            self.host.cycle_challenges.push(challenge);
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
        let remaining = self.host.num_rounds() - self.rounds_bound;
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
        let r_cycle: Vec<F> = self.host.cycle_challenges.iter().rev().copied().collect();
        let eq_cycle = eq_table(&r_cycle);
        let mut lookup_table_flags = vec![F::zero(); LookupTableKind::<RISCV_XLEN>::COUNT];
        let mut instruction_raf_flag = F::zero();
        for (row, &eq) in self.host.rows().iter().zip(&eq_cycle) {
            if let Some(index) = row.table_index.0 {
                lookup_table_flags[index] += eq;
            }
            if row.raf_flag.0 {
                instruction_raf_flag += eq;
            }
        }
        Ok(InstructionReadRafOutputClaims {
            lookup_table_flags,
            instruction_ra,
            instruction_raf_flag,
        })
    }
}
