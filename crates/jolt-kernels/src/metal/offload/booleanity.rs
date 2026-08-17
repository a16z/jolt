//! Metal offload seams for the stage-6a/6b Booleanity kernels: the address
//! pushforward plan, the resident cycle-phase construction, and the
//! device-resident cycle round state machine.

use std::sync::Arc;

use jolt_field::AkitaField;
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6a::booleanity::{
    BooleanityAddressPhase, BooleanityAddressPhaseChallenges,
};
use jolt_verifier::stages::stage6b::booleanity::Booleanity;
use jolt_witness::witnesses::UnsignedIncLane;
use jolt_witness::JoltWitnessPlane;

use crate::metal::solinas::{
    BooleanityRows, BooleanitySelector, BooleanitySequence, BooleanitySequenceConfig, SolinasMetal,
};
use crate::optimized::booleanity::{
    booleanity_address_points, build_booleanity_cycle_kernel, column_selectors, ColumnSelector,
    OptimizedBooleanityAddressKernel, OptimizedBooleanityCycleKernel,
};
use crate::optimized::instruction_read_raf::InstructionCycleRow;
use crate::optimized::lazy_ra::LazyFoldedRa;
use crate::{KernelError, ProverInputs, SumcheckKernel};

impl ColumnSelector {
    fn metal_selector(&self) -> BooleanitySelector {
        match self {
            Self::Instruction(selector) => BooleanitySelector::Lookup {
                shift: selector.shift() as u32,
            },
            Self::Bytecode(selector) => BooleanitySelector::Bytecode {
                shift: selector.shift() as u32,
            },
            Self::Ram(selector) => BooleanitySelector::Ram {
                shift: selector.shift() as u32,
            },
            Self::UnsignedInc(UnsignedIncLane::Chunk { width, index }) => {
                BooleanitySelector::FusedInc {
                    shift: (width * index) as u32,
                }
            }
            Self::UnsignedInc(UnsignedIncLane::Msb { .. }) => BooleanitySelector::FusedIncMsb,
        }
    }
}

pub(crate) struct BooleanityAddressMetalPlan {
    selectors: Vec<BooleanitySelector>,
    reference_cycle: Vec<AkitaField>,
    reference_address: Vec<AkitaField>,
    gamma: AkitaField,
    rounds: usize,
    k: usize,
}

impl BooleanityAddressMetalPlan {
    pub(crate) fn new(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        relation: &BooleanityAddressPhase<AkitaField>,
        challenges: &BooleanityAddressPhaseChallenges<AkitaField>,
    ) -> Result<Self, KernelError<AkitaField>> {
        let (dimensions, reference_cycle) = booleanity_address_points(relation, challenges)?;
        let columns = column_selectors(witness, dimensions)?;
        Ok(Self {
            selectors: columns
                .selectors
                .iter()
                .map(ColumnSelector::metal_selector)
                .collect(),
            reference_cycle,
            reference_address: challenges.reference_address.clone(),
            gamma: challenges.gamma,
            rounds: relation.rounds(),
            k: 1usize << dimensions.log_k_chunk,
        })
    }

    pub(crate) fn selectors(&self) -> &[BooleanitySelector] {
        &self.selectors
    }

    pub(crate) fn reference_cycle(&self) -> &[AkitaField] {
        &self.reference_cycle
    }

    pub(crate) fn finish(
        self,
        flat_masses: Vec<AkitaField>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = BooleanityAddressPhase<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let expected = self.selectors.len().checked_mul(self.k).ok_or_else(|| {
            KernelError::InvalidGeometry {
                reason: "Booleanity address mass count overflows usize".to_owned(),
            }
        })?;
        if flat_masses.len() != expected {
            return Err(KernelError::TableSizeMismatch {
                table: "Metal Booleanity address masses".to_owned(),
                expected,
                got: flat_masses.len(),
            });
        }
        let masses = flat_masses
            .chunks_exact(self.k)
            .map(<[AkitaField]>::to_vec)
            .collect();
        Ok(Box::new(OptimizedBooleanityAddressKernel::new(
            self.rounds,
            self.gamma,
            &self.reference_address,
            masses,
        )))
    }
}

pub(crate) fn prepare_metal_booleanity_cycle(
    witness: &dyn JoltWitnessPlane<AkitaField>,
    inputs: ProverInputs<'_, AkitaField, Booleanity<AkitaField>>,
) -> Result<OptimizedBooleanityCycleKernel<AkitaField>, KernelError<AkitaField>> {
    let dimensions = inputs.relation.dimensions();
    let columns = column_selectors(witness, dimensions)?;
    let cycles = 1usize << dimensions.log_t;
    build_booleanity_cycle_kernel(
        inputs.relation,
        inputs.challenges.gamma,
        columns,
        Arc::new(Vec::new()),
        cycles,
    )
}

impl OptimizedBooleanityCycleKernel<AkitaField> {
    pub(crate) fn metal_row_source(
        &self,
    ) -> Result<&[InstructionCycleRow], SumcheckError<AkitaField>> {
        let LazyFoldedRa::Lazy { width, source, .. } = &self.tables else {
            return Err(booleanity_metal_state_error(
                "resident row preparation requires lazy Booleanity tables",
            ));
        };
        if *width != 1 || self.rounds_bound != 0 || self.metal_offloaded {
            return Err(booleanity_metal_state_error(
                "resident row preparation requires the initial unbound state",
            ));
        }
        if source.rows.len() != source.cycles {
            return Err(booleanity_metal_state_error(
                "resident-only Booleanity state has no CPU row source",
            ));
        }
        Ok(&source.rows)
    }

    pub(crate) fn metal_offload(
        &mut self,
        context: &SolinasMetal,
        resident_rows: BooleanityRows,
        config: BooleanitySequenceConfig,
    ) -> Result<BooleanitySequence, SumcheckError<AkitaField>> {
        let LazyFoldedRa::Lazy {
            tables,
            width,
            source,
        } = &self.tables
        else {
            return Err(booleanity_metal_state_error(
                "Booleanity offload requires lazy source tables",
            ));
        };
        if *width != 1 || self.rounds_bound != 0 || self.metal_offloaded {
            return Err(booleanity_metal_state_error(
                "Booleanity offload requires the initial unbound state",
            ));
        }
        if resident_rows.len() != source.cycles {
            return Err(booleanity_metal_state_error(
                "resident Booleanity row count disagrees with the CPU source",
            ));
        }
        let k = tables.first().map_or(0, Vec::len);
        if k == 0 || tables.iter().any(|table| table.len() != k) {
            return Err(booleanity_metal_state_error(
                "Booleanity base tables have inconsistent lengths",
            ));
        }
        let selectors = source
            .selectors
            .iter()
            .map(ColumnSelector::metal_selector)
            .collect::<Vec<_>>();
        let base_tables = tables.iter().flatten().copied().collect::<Vec<_>>();
        let sequence = context
            .prepare_booleanity_sequence_with_rows(
                resident_rows,
                &selectors,
                &base_tables,
                &self.gamma_powers,
                k,
                self.eq.e_in_current_len(),
                self.eq.e_out_current_len(),
                config,
            )
            .map_err(booleanity_metal_error)?;
        self.tables = LazyFoldedRa::Dense(Vec::new());
        self.metal_offloaded = true;
        Ok(sequence)
    }

    pub(crate) fn metal_bind_offloaded(
        &mut self,
        challenge: AkitaField,
    ) -> Result<(), SumcheckError<AkitaField>> {
        if !self.metal_offloaded {
            return Err(booleanity_metal_state_error(
                "device bind requires offloaded Booleanity tables",
            ));
        }
        self.eq.bind(challenge);
        self.rounds_bound += 1;
        Ok(())
    }

    pub(crate) fn metal_weights(
        &self,
    ) -> Result<(&[AkitaField], &[AkitaField]), SumcheckError<AkitaField>> {
        if !self.metal_offloaded {
            return Err(booleanity_metal_state_error(
                "device weights require offloaded Booleanity tables",
            ));
        }
        Ok((self.eq.e_in_current(), self.eq.e_out_current()))
    }

    pub(crate) fn metal_message(
        &self,
        q_coefficients: [AkitaField; 2],
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if !self.metal_offloaded {
            return Err(booleanity_metal_state_error(
                "device message requires offloaded Booleanity tables",
            ));
        }
        Ok(self
            .eq
            .gruen_poly_deg_3(q_coefficients[0], q_coefficients[1], previous_claim))
    }

    pub(crate) fn metal_restore_dense(
        &mut self,
        flat: &[AkitaField],
        elements: usize,
    ) -> Result<(), SumcheckError<AkitaField>> {
        let expected = self.gamma_powers.len() * elements;
        if !self.metal_offloaded || elements == 0 || flat.len() != expected {
            return Err(booleanity_metal_state_error(
                "Booleanity readback shape disagrees with the offloaded state",
            ));
        }
        self.tables = LazyFoldedRa::Dense(
            flat.chunks_exact(elements)
                .map(|evals| Polynomial::new(evals.to_vec()))
                .collect(),
        );
        self.metal_offloaded = false;
        Ok(())
    }

    pub(crate) fn metal_polys(&self) -> usize {
        self.gamma_powers.len()
    }
}

fn booleanity_metal_error(error: crate::metal::solinas::MetalError) -> SumcheckError<AkitaField> {
    booleanity_metal_state_error(error.to_string())
}

fn booleanity_metal_state_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}
