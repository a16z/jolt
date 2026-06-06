use jolt_backends::{
    BackendValueSlot, SumcheckEvaluationRequest, SumcheckMaterializationRequest,
    SumcheckViewEvaluationRequest, SumcheckViewMaterializationRequest,
};
use jolt_claims::protocols::jolt::{
    formulas::{dimensions::REGISTER_ADDRESS_BITS, instruction, ram, registers},
    JoltOpeningId,
};
use jolt_field::Field;
use jolt_witness::{
    protocols::jolt_vm::{jolt_opening_oracle_ref, JoltVmNamespace},
    OracleRef, ViewRequirement, WitnessNamespace, WitnessProvider,
};

use super::output::{Stage5RegularBatchInputClaims, Stage5RegularBatchPrefixOutput};
use crate::ProverError;

pub const STAGE5_LOOKUP_TABLE_FLAG_OPENING_SLOT_START: u32 = 0;
pub const STAGE5_INSTRUCTION_RA_OPENING_SLOT_START: u32 = 1024;
pub const STAGE5_INSTRUCTION_RAF_FLAG_OPENING_SLOT: BackendValueSlot = BackendValueSlot(2048);
pub const STAGE5_RAM_RA_CLAIM_REDUCTION_OPENING_SLOT: BackendValueSlot = BackendValueSlot(2050);
pub const STAGE5_REGISTERS_VAL_EVALUATION_OPENING_SLOT_START: u32 = 2064;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5RegularBatchRequest<F: Field> {
    pub input_claims: Stage5RegularBatchInputClaims<F>,
    pub instruction_gamma: F,
    pub ram_gamma: F,
}

impl<F: Field> Stage5RegularBatchRequest<F> {
    pub fn from_prefix(prefix: &Stage5RegularBatchPrefixOutput<F>) -> Self {
        Self {
            input_claims: prefix.input_claims.clone(),
            instruction_gamma: prefix.instruction_gamma,
            ram_gamma: prefix.ram_gamma,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5OutputOpeningEvaluationRequest<F: Field> {
    pub evaluations: SumcheckEvaluationRequest<F, JoltVmNamespace>,
    pub instruction_lookup_table_flags: Vec<Stage5OutputOpeningRequest>,
    pub instruction_ra: Vec<Stage5OutputOpeningRequest>,
    pub instruction_raf_flag: Stage5OutputOpeningRequest,
    pub ram_ra_claim_reduction: Stage5OutputOpeningRequest,
    pub registers_val_evaluation: Vec<Stage5OutputOpeningRequest>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5OutputOpeningMaterializationRequest {
    pub materializations: SumcheckMaterializationRequest<JoltVmNamespace>,
    pub ram_ra_claim_reduction: Stage5OutputOpeningRequest,
    pub registers_val_evaluation: Vec<Stage5OutputOpeningRequest>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5OutputOpeningRequest {
    pub slot: BackendValueSlot,
    pub view: ViewRequirement<JoltVmNamespace>,
}

pub fn build_stage5_output_opening_evaluation_request<F, W>(
    config: super::input::Stage5ProverConfig,
    witness: &W,
    instruction_lookup_table_flag_opening_point: Vec<F>,
    instruction_ra_opening_points: Vec<Vec<F>>,
    instruction_raf_flag_opening_point: Vec<F>,
    ram_ra_claim_reduction_opening_point: Vec<F>,
    registers_val_evaluation_opening_point: Vec<F>,
) -> Result<Stage5OutputOpeningEvaluationRequest<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    if instruction_lookup_table_flag_opening_point.len() != config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 5 lookup table flag opening point has {} variables, expected {}",
                instruction_lookup_table_flag_opening_point.len(),
                config.log_t
            ),
        });
    }
    if instruction_raf_flag_opening_point.len() != config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 5 instruction RAF flag opening point has {} variables, expected {}",
                instruction_raf_flag_opening_point.len(),
                config.log_t
            ),
        });
    }
    let instruction_ra_count = config
        .instruction_read_raf_dimensions
        .num_virtual_ra_polys();
    let instruction_ra_chunk_size = config
        .instruction_read_raf_dimensions
        .instruction_address_bits()
        .checked_div(instruction_ra_count)
        .ok_or_else(|| ProverError::InvalidStageRequest {
            reason: "Stage 5 instruction read-RAF config has no virtual RA polynomials".to_owned(),
        })?;
    if instruction_ra_chunk_size * instruction_ra_count
        != config
            .instruction_read_raf_dimensions
            .instruction_address_bits()
    {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 5 instruction address bit count {} is not divisible by virtual RA count {instruction_ra_count}",
                config.instruction_read_raf_dimensions.instruction_address_bits()
            ),
        });
    }
    let expected_instruction_ra_vars = instruction_ra_chunk_size + config.log_t;
    for point in &instruction_ra_opening_points {
        if point.len() != expected_instruction_ra_vars {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "Stage 5 instruction RA opening point has {} variables, expected {expected_instruction_ra_vars}",
                    point.len()
                ),
            });
        }
    }
    if instruction_ra_opening_points.len() != instruction_ra_count {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 5 instruction RA opening count is {}, expected {}",
                instruction_ra_opening_points.len(),
                instruction_ra_count
            ),
        });
    }
    if ram_ra_claim_reduction_opening_point.len() != config.log_k + config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 5 RAM RA claim-reduction opening point has {} variables, expected {}",
                ram_ra_claim_reduction_opening_point.len(),
                config.log_k + config.log_t
            ),
        });
    }
    if registers_val_evaluation_opening_point.len() != REGISTER_ADDRESS_BITS + config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 5 register value-evaluation opening point has {} variables, expected {}",
                registers_val_evaluation_opening_point.len(),
                REGISTER_ADDRESS_BITS + config.log_t
            ),
        });
    }

    let instruction_output_openings =
        instruction::read_raf_output_openings(config.instruction_read_raf_dimensions);
    let instruction_lookup_table_flags = instruction_output_openings
        .lookup_table_flags
        .iter()
        .copied()
        .enumerate()
        .map(|(index, opening)| {
            build_opening_request(
                witness,
                stage5_oracle_from_opening(opening)?,
                instruction_lookup_table_flag_opening_slot(index),
            )
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let instruction_ra = instruction_output_openings
        .instruction_ra
        .iter()
        .copied()
        .enumerate()
        .map(|(index, opening)| {
            build_opening_request(
                witness,
                stage5_oracle_from_opening(opening)?,
                instruction_ra_opening_slot(index),
            )
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let instruction_raf_flag = build_opening_request(
        witness,
        stage5_oracle_from_opening(instruction_output_openings.instruction_raf_flag)?,
        STAGE5_INSTRUCTION_RAF_FLAG_OPENING_SLOT,
    )?;
    let [ram_ra_claim_reduction_opening] = ram::ra_claim_reduction_output_openings();
    let ram_ra_claim_reduction = build_opening_request(
        witness,
        stage5_oracle_from_opening(ram_ra_claim_reduction_opening)?,
        STAGE5_RAM_RA_CLAIM_REDUCTION_OPENING_SLOT,
    )?;
    let registers_val_evaluation = registers::val_evaluation_output_openings()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, opening)| {
            build_opening_request(
                witness,
                stage5_oracle_from_opening(opening)?,
                registers_val_evaluation_opening_slot(index),
            )
        })
        .collect::<Result<Vec<_>, ProverError>>()?;

    let (_, registers_cycle_point) =
        registers_val_evaluation_opening_point.split_at(REGISTER_ADDRESS_BITS);
    let mut views =
        Vec::with_capacity(instruction_lookup_table_flags.len() + instruction_ra.len() + 1 + 1 + 2);
    views.extend(instruction_lookup_table_flags.iter().map(|opening| {
        SumcheckViewEvaluationRequest::new(
            opening.slot,
            opening.view,
            instruction_lookup_table_flag_opening_point.clone(),
        )
    }));
    views.extend(
        instruction_ra
            .iter()
            .zip(instruction_ra_opening_points)
            .map(|(opening, point)| {
                SumcheckViewEvaluationRequest::new(opening.slot, opening.view, point)
            }),
    );
    views.push(SumcheckViewEvaluationRequest::new(
        instruction_raf_flag.slot,
        instruction_raf_flag.view,
        instruction_raf_flag_opening_point,
    ));
    views.push(SumcheckViewEvaluationRequest::new(
        ram_ra_claim_reduction.slot,
        ram_ra_claim_reduction.view,
        ram_ra_claim_reduction_opening_point,
    ));
    let [register_rd_inc, register_rd_wa] = registers_val_evaluation.as_slice() else {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 5 register value-evaluation request has {} openings, expected 2",
                registers_val_evaluation.len()
            ),
        });
    };
    views.push(SumcheckViewEvaluationRequest::new(
        register_rd_inc.slot,
        register_rd_inc.view,
        registers_cycle_point.to_vec(),
    ));
    views.push(SumcheckViewEvaluationRequest::new(
        register_rd_wa.slot,
        register_rd_wa.view,
        registers_val_evaluation_opening_point,
    ));

    Ok(Stage5OutputOpeningEvaluationRequest {
        evaluations: SumcheckEvaluationRequest::new("stage5.batch.output_openings", views),
        instruction_lookup_table_flags,
        instruction_ra,
        instruction_raf_flag,
        ram_ra_claim_reduction,
        registers_val_evaluation,
    })
}

pub fn build_stage5_regular_batch_materialization_request<F, W>(
    witness: &W,
) -> Result<Stage5OutputOpeningMaterializationRequest, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let [ram_ra_claim_reduction_opening] = ram::ra_claim_reduction_output_openings();
    let ram_ra_claim_reduction = build_opening_request(
        witness,
        stage5_oracle_from_opening(ram_ra_claim_reduction_opening)?,
        STAGE5_RAM_RA_CLAIM_REDUCTION_OPENING_SLOT,
    )?;
    let registers_val_evaluation = registers::val_evaluation_output_openings()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, opening)| {
            build_opening_request(
                witness,
                stage5_oracle_from_opening(opening)?,
                registers_val_evaluation_opening_slot(index),
            )
        })
        .collect::<Result<Vec<_>, ProverError>>()?;

    let mut views = Vec::with_capacity(1 + registers_val_evaluation.len());
    views.push(SumcheckViewMaterializationRequest::new(
        ram_ra_claim_reduction.slot,
        ram_ra_claim_reduction.view,
    ));
    views.extend(
        registers_val_evaluation
            .iter()
            .map(|opening| SumcheckViewMaterializationRequest::new(opening.slot, opening.view)),
    );

    Ok(Stage5OutputOpeningMaterializationRequest {
        materializations: SumcheckMaterializationRequest::new(
            "stage5.batch.output_materializations",
            views,
        ),
        ram_ra_claim_reduction,
        registers_val_evaluation,
    })
}

pub const fn instruction_lookup_table_flag_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE5_LOOKUP_TABLE_FLAG_OPENING_SLOT_START + index as u32)
}

pub const fn instruction_ra_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE5_INSTRUCTION_RA_OPENING_SLOT_START + index as u32)
}

pub const fn registers_val_evaluation_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE5_REGISTERS_VAL_EVALUATION_OPENING_SLOT_START + index as u32)
}

fn build_opening_request<F, W>(
    witness: &W,
    oracle: OracleRef<JoltVmNamespace>,
    slot: BackendValueSlot,
) -> Result<Stage5OutputOpeningRequest, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let view = primary_view_requirement::<F, W, JoltVmNamespace>(witness, oracle)?;
    Ok(Stage5OutputOpeningRequest { slot, view })
}

fn stage5_oracle_from_opening(
    opening: JoltOpeningId,
) -> Result<OracleRef<JoltVmNamespace>, ProverError> {
    Ok(jolt_opening_oracle_ref(opening)?)
}

fn primary_view_requirement<F, W, N>(
    witness: &W,
    oracle: OracleRef<N>,
) -> Result<ViewRequirement<N>, ProverError>
where
    F: Field,
    N: WitnessNamespace,
    W: WitnessProvider<F, N>,
{
    let Some(requirement) = witness.view_requirements(oracle)?.into_iter().next() else {
        return Err(ProverError::InvalidStageRequest {
            reason: format!("witness returned no view requirement for {:?}", oracle.kind),
        });
    };
    Ok(requirement)
}
