//! Metal offload seams for the stage-6b bytecode read-RAF cycle kernel: the
//! device-shell construction (weights and eq tables the GPU sequence
//! consumes) and the device-resident round state machine.

use jolt_claims::protocols::jolt::geometry::bytecode;
use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_field::AkitaField;
use jolt_poly::{
    eq_index_msb, IdentityPolynomial, MultilinearEvaluation, Polynomial, UnivariatePoly,
};
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle;
use jolt_witness::witnesses::RaChunkSelector;

use crate::optimized::bytecode_read_raf::{
    bytecode_cycle_state_error, BytecodeCycleAlgebra, CycleCpuTables, CycleKernel, CycleTableState,
    LazyFusedInc,
};
use crate::optimized::lazy_ra::LazyFoldedRa;
use crate::optimized::support::{eq_table, round_poly_from_skipped_evals};
use crate::{KernelError, ProverInputs};

pub(crate) struct MetalBytecodeCycleInputs {
    pub stage_points: Vec<Vec<AkitaField>>,
    pub stage_weights: Vec<AkitaField>,
    pub entry_weight: AkitaField,
    pub ra0: Vec<AkitaField>,
    pub ra1: Vec<AkitaField>,
}

pub(crate) fn prepare_metal_bytecode_cycle_shell(
    inputs: ProverInputs<'_, AkitaField, BytecodeReadRafCycle<AkitaField>>,
    cpu_tail_algebra: BytecodeCycleAlgebra,
) -> Result<(CycleKernel<AkitaField>, MetalBytecodeCycleInputs), KernelError<AkitaField>> {
    let relation = inputs.relation;
    let dimensions = relation.dimensions();
    let cycles = 1usize << dimensions.log_t();
    let num_ra = dimensions.num_committed_ra_polys();
    if relation.degree() != 4 || num_ra != 2 || relation.committed_chunk_bits() != 8 {
        return Err(KernelError::InvariantViolation {
            reason: "Metal bytecode cycle shell requires degree four and two 8-bit RA chunks",
        });
    }

    let stage_points = relation.stage_cycle_points();
    let base_stages = bytecode::BYTECODE_STAGE_GAMMA_COUNTS.len();
    let num_stages = stage_points.len();
    if base_stages != 5 || num_stages != 9 {
        return Err(KernelError::InvariantViolation {
            reason: "Metal bytecode cycle shell requires five base and four fused stages",
        });
    }
    for point in stage_points {
        if point.len() != dimensions.log_t() {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode stage cycle point has the wrong variable count",
            });
        }
    }

    let stage_values = relation.stage_values_at_r_address()?;
    let gamma = inputs.challenges.gamma;
    let mut gamma_powers = vec![AkitaField::one(); num_stages + 3];
    for index in 1..gamma_powers.len() {
        gamma_powers[index] = gamma_powers[index - 1] * gamma;
    }
    let r_address = relation.r_address();
    let int_at_r_address = IdentityPolynomial::new(r_address.len()).evaluate(r_address);
    let mut stage_weights = (0..base_stages)
        .map(|stage| gamma_powers[stage] * stage_values[stage])
        .collect::<Vec<_>>();
    stage_weights[0] += gamma_powers[num_stages] * int_at_r_address;
    stage_weights[2] += gamma_powers[num_stages + 1] * int_at_r_address;
    let store = stage_values[base_stages];
    stage_weights.extend((base_stages..num_stages).map(|stage| {
        let value = if stage < base_stages + 2 {
            store
        } else {
            AkitaField::one() - store
        };
        gamma_powers[stage] * value
    }));

    let entry_index = u128::try_from(relation.entry_bytecode_index()).map_err(|_| {
        KernelError::InvariantViolation {
            reason: "bytecode entry index exceeds u128",
        }
    })?;
    let entry_weight =
        gamma_powers[num_stages + 2] * eq_index_msb::<AkitaField>(r_address, entry_index);
    let chunks = committed_address_chunks(r_address, relation.committed_chunk_bits());
    if chunks.len() != 2 || chunks.iter().any(|chunk| chunk.len() != 8) {
        return Err(KernelError::InvariantViolation {
            reason: "Metal bytecode cycle RA chunk geometry is not two by eight",
        });
    }
    let selectors = (0..2)
        .map(|index| RaChunkSelector::new(index, 2, 8).map_err(KernelError::from))
        .collect::<Result<Vec<_>, _>>()?;
    if selectors[0].shift() != 8 || selectors[1].shift() != 0 {
        return Err(KernelError::InvariantViolation {
            reason: "Metal bytecode cycle RA chunks are not most-significant first",
        });
    }
    let ra0 = eq_table(&chunks[0]);
    let ra1 = eq_table(&chunks[1]);
    let output_openings = bytecode::read_raf_output_openings(dimensions).bytecode_ra;
    if output_openings.len() != 2 {
        return Err(KernelError::InvariantViolation {
            reason: "Metal bytecode cycle output opening count is not two",
        });
    }

    Ok((
        CycleKernel {
            rounds: relation.rounds(),
            degree: relation.degree(),
            algebra: cpu_tail_algebra,
            tables: CycleTableState::Offloaded { elements: cycles },
            output_openings,
            rounds_bound: 0,
        },
        MetalBytecodeCycleInputs {
            stage_points: stage_points.to_vec(),
            stage_weights,
            entry_weight,
            ra0,
            ra1,
        },
    ))
}

pub(crate) struct BytecodeCycleDenseState {
    pub combined: Vec<jolt_field::AkitaField>,
    pub fused_combined: Vec<jolt_field::AkitaField>,
    pub fused_inc: Vec<jolt_field::AkitaField>,
    pub ra0: Vec<jolt_field::AkitaField>,
    pub ra1: Vec<jolt_field::AkitaField>,
}

impl CycleKernel<AkitaField> {
    pub(crate) fn metal_message(
        &self,
        evals: [AkitaField; 4],
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if self.degree != 4 {
            return Err(bytecode_cycle_state_error(
                "Metal bytecode cycle message requires degree four",
            ));
        }
        let _ = self.metal_elements()?;
        Ok(round_poly_from_skipped_evals(&evals, previous_claim))
    }

    pub(crate) fn metal_commit_bind(
        &mut self,
        device_elements: usize,
    ) -> Result<(), SumcheckError<AkitaField>> {
        let CycleTableState::Offloaded { elements } = &mut self.tables else {
            return Err(bytecode_cycle_state_error(
                "Metal bytecode cycle bind commit requires offloaded tables",
            ));
        };
        if *elements < 2 || device_elements != *elements / 2 || self.rounds_bound >= self.rounds {
            return Err(bytecode_cycle_state_error(
                "Metal bytecode cycle bind commit disagrees with the resident length",
            ));
        }
        *elements = device_elements;
        self.rounds_bound += 1;
        Ok(())
    }

    pub(crate) fn metal_restore_dense(
        &mut self,
        state: BytecodeCycleDenseState,
    ) -> Result<(), SumcheckError<AkitaField>> {
        let expected = self.metal_elements()?;
        for (name, got) in [
            ("combined", state.combined.len()),
            ("fused_combined", state.fused_combined.len()),
            ("fused_inc", state.fused_inc.len()),
            ("ra0", state.ra0.len()),
            ("ra1", state.ra1.len()),
        ] {
            if got != expected {
                return Err(bytecode_cycle_state_error(format!(
                    "Metal bytecode cycle {name} readback has length {got}, expected {expected}"
                )));
            }
        }
        self.tables = CycleTableState::Cpu(CycleCpuTables {
            ra: LazyFoldedRa::Dense(vec![Polynomial::new(state.ra0), Polynomial::new(state.ra1)]),
            combined: Polynomial::new(state.combined),
            fused_inc: LazyFusedInc::Dense(Polynomial::new(state.fused_inc)),
            fused_combined: Polynomial::new(state.fused_combined),
        });
        Ok(())
    }

    pub(crate) fn metal_elements(&self) -> Result<usize, SumcheckError<AkitaField>> {
        match &self.tables {
            CycleTableState::Offloaded { elements } => Ok(*elements),
            CycleTableState::Cpu(_) => Err(bytecode_cycle_state_error(
                "Metal bytecode cycle operation reached restored CPU tables",
            )),
        }
    }

    #[cfg(test)]
    pub(crate) const fn metal_rounds_bound(&self) -> usize {
        self.rounds_bound
    }
}
