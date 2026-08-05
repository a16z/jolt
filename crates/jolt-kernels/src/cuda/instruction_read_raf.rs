use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionReadRafInputClaims, InstructionReadRafOutputClaims,
};
use jolt_field::Field;
use jolt_lookup_tables::tables::prefixes::PrefixEval;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::stage5::instruction_read_raf::InstructionReadRaf;
use jolt_witness::{collect_bundles, JoltWitnessPlane};

use super::address_driver::DeviceAddressPhase;
use super::context::CudaKernelContext;
use super::device::{fr_into, require_fr};
use super::{require_context, CudaBackend};
use crate::reference::instruction_read_raf::{InstructionReadRafKernel, InstructionReadRafWitness};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const CHUNK_LEN: usize = 8;
const ADDRESS_BITS: usize = 128;
const RAF_CHECKPOINTS: usize = 4;

pub struct DeviceInstructionReadRaf<F: Field> {
    host: InstructionReadRafKernel<F>,
    device: Option<DeviceAddressPhase>,
    gamma: jolt_field::Fr,
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
        self.host.init_cycle_rounds();
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
        Ok(Box::new(DeviceInstructionReadRaf {
            host,
            device: Some(device),
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
            self.host.cycle_message()?
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
            self.rounds_bound += 1;
            self.host.bind(challenge)
        }
    }
}

impl<F: Field> SumcheckKernel<F> for DeviceInstructionReadRaf<F> {
    type Relation = InstructionReadRaf<F>;

    fn output_claims(
        &mut self,
        inputs: &InstructionReadRafInputClaims<F>,
    ) -> Result<InstructionReadRafOutputClaims<F>, SumcheckKernelError<F>> {
        self.host.output_claims(inputs)
    }
}
