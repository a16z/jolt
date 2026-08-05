use jolt_field::AkitaField;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6b::booleanity::{Booleanity, BooleanityCyclePhaseChallenges};
use jolt_witness::JoltWitnessPlane;

use super::instruction_read_raf::MetalBackend;
use super::solinas::{BooleanityRows, BooleanitySequence, BooleanitySequenceConfig};
use crate::optimized::booleanity::{
    prepare_optimized_booleanity_cycle, OptimizedBooleanityCycleKernel,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BooleanityMetalConfig {
    pub trace_cutoff_elements: usize,
    pub cutoff_elements: usize,
    pub dispatch: BooleanitySequenceConfig,
}

impl Default for BooleanityMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 18,
            cutoff_elements: 1 << 10,
            dispatch: BooleanitySequenceConfig {
                threads_per_threadgroup: Some(256),
                dense_threads_per_threadgroup: Some(128),
                materialize_width: 8,
            },
        }
    }
}

impl PrepareKernel<AkitaField, Booleanity<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, Booleanity<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = Booleanity<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let trace_elements = 1usize << inputs.relation.dimensions().log_t;
        let mut cpu = prepare_optimized_booleanity_cycle(session, witness, inputs)?;
        if trace_elements < self.config.booleanity_cycle.trace_cutoff_elements {
            let _ = session.take::<BooleanityRows>();
            return Ok(Box::new(cpu));
        }
        let resident_rows = match session.take::<BooleanityRows>() {
            Some(rows) if rows.len() == trace_elements => rows,
            _ => cpu.metal_prepare_rows(&self.context)?,
        };
        let sequence = cpu.metal_offload(
            &self.context,
            resident_rows,
            self.config.booleanity_cycle.dispatch,
        )?;
        Ok(Box::new(MetalBooleanityKernel::new(
            cpu,
            sequence,
            self.config.booleanity_cycle.cutoff_elements,
        )?))
    }
}

pub(crate) struct MetalBooleanityKernel {
    cpu: OptimizedBooleanityCycleKernel<AkitaField>,
    sequence: Option<BooleanitySequence>,
    host_tail: Vec<AkitaField>,
    cutoff_elements: usize,
    metal_rounds: usize,
}

impl MetalBooleanityKernel {
    fn new(
        cpu: OptimizedBooleanityCycleKernel<AkitaField>,
        sequence: BooleanitySequence,
        cutoff_elements: usize,
    ) -> Result<Self, SumcheckError<AkitaField>> {
        let host_elements = cpu
            .metal_polys()
            .checked_mul(cutoff_elements)
            .ok_or_else(|| metal_error("Booleanity host-tail capacity overflow"))?;
        Ok(Self {
            cpu,
            sequence: Some(sequence),
            host_tail: vec![AkitaField::zero(); host_elements],
            cutoff_elements,
            metal_rounds: 0,
        })
    }

    fn restore_cpu_tail(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let sequence = self
            .sequence
            .take()
            .ok_or_else(|| metal_error("Booleanity sequence disappeared before readback"))?;
        if !sequence.is_dense() {
            return Err(metal_error(
                "Booleanity CPU handoff requires resident dense tables",
            ));
        }
        let elements = sequence.current_elements();
        let output_len = self
            .cpu
            .metal_polys()
            .checked_mul(elements)
            .ok_or_else(|| metal_error("Booleanity readback length overflow"))?;
        if output_len > self.host_tail.len() {
            return Err(metal_error(
                "Booleanity readback exceeds the preallocated CPU tail",
            ));
        }
        sequence
            .read_current_tables(&mut self.host_tail[..output_len])
            .map_err(|error| metal_error(error.to_string()))?;
        self.cpu
            .metal_restore_dense(&self.host_tail[..output_len], elements)
    }
}

impl ProveRounds<AkitaField> for MetalBooleanityKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<jolt_poly::UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if self.sequence.as_ref().is_some_and(|sequence| {
            sequence.is_dense() && sequence.current_elements() <= self.cutoff_elements
        }) {
            self.restore_cpu_tail()?;
            return self.cpu.prove_round(bind, round, previous_claim);
        }

        if self.sequence.is_some() {
            let message = if let Some(challenge) = bind {
                self.cpu.metal_bind_offloaded(challenge)?;
                let (e_in, e_out) = self.cpu.metal_weights()?;
                self.sequence
                    .as_mut()
                    .ok_or_else(|| metal_error("Booleanity sequence disappeared before bind"))?
                    .bind_and_message(challenge, e_in, e_out)
                    .map_err(|error| metal_error(error.to_string()))?
            } else {
                let (e_in, e_out) = self.cpu.metal_weights()?;
                self.sequence
                    .as_mut()
                    .ok_or_else(|| metal_error("Booleanity sequence disappeared before message"))?
                    .message(e_in, e_out)
                    .map_err(|error| metal_error(error.to_string()))?
            };
            self.metal_rounds += 1;
            return self.cpu.metal_message(message, previous_claim);
        }

        self.cpu.prove_round(bind, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.sequence.is_some() {
            self.restore_cpu_tail()?;
        }
        self.cpu.finish_rounds(bind)
    }
}

impl SumcheckKernel<AkitaField> for MetalBooleanityKernel {
    type Relation = Booleanity<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<AkitaField, Self::Relation>, SumcheckKernelError<AkitaField>>
    {
        self.cpu.output_claims(inputs)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &BooleanityCyclePhaseChallenges<AkitaField>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        self.cpu
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}

fn metal_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Metal parity test setup")]
mod tests {
    use jolt_claims::protocols::jolt::lattice::relations::booleanity::LatticeBooleanityDimensions;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage6b::booleanity::BooleanityInputClaims;

    use super::*;
    use crate::optimized::booleanity::{
        testing::with_booleanity_backend, OptimizedBooleanityCycle,
    };
    use crate::optimized::instruction_read_raf::{
        collect_instruction_cycle_rows, InstructionCycleRow,
    };

    fn point(seed: u64, len: usize) -> Vec<AkitaField> {
        (0..len as u64)
            .map(|index| AkitaField::from_u64(seed + 37 * index + 5))
            .collect()
    }

    #[test]
    fn prepare_kernel_matches_optimized_cpu_with_resident_rows() {
        let log_t = 10;
        with_booleanity_backend(log_t, 8, |witness, dimensions| {
            let r_address = point(110, dimensions.log_k_chunk);
            let reference_address = point(700, dimensions.log_k_chunk);
            let reference_cycle = point(400, log_t);
            let relation = Booleanity::new(
                LatticeBooleanityDimensions::new(dimensions).unwrap(),
                r_address.clone(),
                reference_address,
                reference_cycle,
            );
            let claims = BooleanityInputClaims {
                address_phase: AkitaField::from_u64(17),
            };
            let points = BooleanityInputClaims {
                address_phase: r_address,
            };
            let challenges = BooleanityCyclePhaseChallenges {
                gamma: AkitaField::from_u64(31),
            };
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let mut expected = OptimizedBooleanityCycle
                .prepare(&mut ProofSession::default(), witness, inputs())
                .unwrap();
            let metal = MetalBackend::new(super::super::MetalConfig {
                booleanity_cycle: BooleanityMetalConfig {
                    trace_cutoff_elements: 2,
                    cutoff_elements: 4,
                    dispatch: BooleanitySequenceConfig {
                        threads_per_threadgroup: Some(256),
                        dense_threads_per_threadgroup: Some(128),
                        materialize_width: 2,
                    },
                },
                ..Default::default()
            })
            .unwrap();
            let packed = collect_instruction_cycle_rows::<AkitaField>(witness, 1 << log_t).unwrap();
            let resident = metal
                .context
                .prepare_booleanity_rows(InstructionCycleRow::metal_booleanity_rows(&packed))
                .unwrap();
            let retained = resident.clone();
            let mut session = ProofSession::default();
            session.park(resident);
            let bytecode_rows = session.state::<BooleanityRows>().cloned().unwrap();
            assert!(retained.shares_allocation(&bytecode_rows));
            let mut actual = metal.prepare(&mut session, witness, inputs()).unwrap();
            assert!(session.state::<BooleanityRows>().is_none());
            assert_eq!(bytecode_rows.len(), 1 << log_t);
            assert_eq!(
                bytecode_rows.device_registry_id(),
                metal.context.device_registry_id()
            );
            assert!(retained.shares_allocation(&bytecode_rows));

            let mut claim = claims.address_phase;
            let mut bind = None;
            let mut round_challenges = Vec::new();
            for round in 0..expected.num_rounds() {
                let expected_poly = expected.prove_round(bind, round, claim).unwrap();
                let actual_poly = actual.prove_round(bind, round, claim).unwrap();
                assert_eq!(actual_poly, expected_poly, "round {round}");
                let challenge = AkitaField::from_u64(0x1234_5678 + 1000 * round as u64 + 7);
                claim = expected_poly.evaluate(challenge);
                round_challenges.push(challenge);
                bind = Some(challenge);
            }
            let final_bind = *round_challenges.last().unwrap();
            expected.finish_rounds(final_bind).unwrap();
            actual.finish_rounds(final_bind).unwrap();
            assert_eq!(
                actual.output_claims(&claims).unwrap(),
                expected.output_claims(&claims).unwrap()
            );

            let output_points = relation
                .derive_opening_points(&round_challenges, &points)
                .unwrap();
            actual
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
        });
    }
}
