use jolt_backends::{BackendValueSlot, SumcheckEvaluationRequest, SumcheckViewEvaluationRequest};
use jolt_claims::protocols::jolt::{
    formulas::{booleanity, bytecode, claim_reductions::increments},
    formulas::{instruction, ram},
    JoltOpeningId,
};
use jolt_field::Field;
use jolt_witness::{
    protocols::jolt_vm::{jolt_opening_oracle_ref, JoltVmNamespace},
    OracleRef, ViewRequirement, WitnessNamespace, WitnessProvider,
};

use super::input::Stage6ProverConfig;
use super::output::{Stage6RegularBatchInputClaims, Stage6RegularBatchPrefixOutput};
use crate::ProverError;

pub const STAGE6_BYTECODE_RA_OPENING_SLOT_START: u32 = 0;
pub const STAGE6_BOOLEANITY_INSTRUCTION_RA_OPENING_SLOT_START: u32 = 1024;
pub const STAGE6_BOOLEANITY_BYTECODE_RA_OPENING_SLOT_START: u32 = 2048;
pub const STAGE6_BOOLEANITY_RAM_RA_OPENING_SLOT_START: u32 = 3072;
pub const STAGE6_RAM_HAMMING_BOOLEANITY_OPENING_SLOT: BackendValueSlot = BackendValueSlot(4096);
pub const STAGE6_RAM_RA_VIRTUALIZATION_OPENING_SLOT_START: u32 = 4100;
pub const STAGE6_INSTRUCTION_RA_VIRTUALIZATION_OPENING_SLOT_START: u32 = 5120;
pub const STAGE6_INC_RAM_OPENING_SLOT: BackendValueSlot = BackendValueSlot(6144);
pub const STAGE6_INC_RD_OPENING_SLOT: BackendValueSlot = BackendValueSlot(6145);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6RegularBatchRequest<F: Field> {
    pub input_claims: Stage6RegularBatchInputClaims<F>,
    pub bytecode_gamma_powers: Vec<F>,
    pub stage1_gammas: Vec<F>,
    pub stage2_gammas: Vec<F>,
    pub stage3_gammas: Vec<F>,
    pub stage4_gammas: Vec<F>,
    pub stage5_gammas: Vec<F>,
    pub booleanity_reference_address: Vec<F>,
    pub booleanity_reference_cycle: Vec<F>,
    pub booleanity_gamma: F,
    pub instruction_ra_gamma_powers: Vec<F>,
    pub inc_gamma: F,
    #[cfg(feature = "field-inline")]
    pub field_inc_gamma: F,
}

impl<F: Field> Stage6RegularBatchRequest<F> {
    pub fn from_prefix(prefix: &Stage6RegularBatchPrefixOutput<F>) -> Self {
        Self {
            input_claims: prefix.input_claims.clone(),
            bytecode_gamma_powers: prefix.bytecode_gamma_powers.clone(),
            stage1_gammas: prefix.stage1_gammas.clone(),
            stage2_gammas: prefix.stage2_gammas.clone(),
            stage3_gammas: prefix.stage3_gammas.clone(),
            stage4_gammas: prefix.stage4_gammas.clone(),
            stage5_gammas: prefix.stage5_gammas.clone(),
            booleanity_reference_address: prefix.booleanity_reference_address.clone(),
            booleanity_reference_cycle: prefix.booleanity_reference_cycle.clone(),
            booleanity_gamma: prefix.booleanity_gamma,
            instruction_ra_gamma_powers: prefix.instruction_ra_gamma_powers.clone(),
            inc_gamma: prefix.inc_gamma,
            #[cfg(feature = "field-inline")]
            field_inc_gamma: prefix.field_inc_gamma,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6OutputOpeningEvaluationRequest<F: Field> {
    pub evaluations: SumcheckEvaluationRequest<F, JoltVmNamespace>,
    pub bytecode_read_raf_bytecode_ra: Vec<Stage6OutputOpeningRequest>,
    pub booleanity_instruction_ra: Vec<Stage6OutputOpeningRequest>,
    pub booleanity_bytecode_ra: Vec<Stage6OutputOpeningRequest>,
    pub booleanity_ram_ra: Vec<Stage6OutputOpeningRequest>,
    pub ram_hamming_booleanity: Stage6OutputOpeningRequest,
    pub ram_ra_virtualization: Vec<Stage6OutputOpeningRequest>,
    pub instruction_ra_virtualization: Vec<Stage6OutputOpeningRequest>,
    pub inc_claim_reduction: [Stage6OutputOpeningRequest; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6OutputOpeningRequest {
    pub slot: BackendValueSlot,
    pub view: ViewRequirement<JoltVmNamespace>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 output openings have distinct verifier-derived points."
)]
pub fn build_stage6_output_opening_evaluation_request<F, W>(
    config: Stage6ProverConfig,
    witness: &W,
    bytecode_ra_opening_points: Vec<Vec<F>>,
    booleanity_opening_point: Vec<F>,
    ram_hamming_opening_point: Vec<F>,
    ram_ra_opening_points: Vec<Vec<F>>,
    instruction_ra_opening_points: Vec<Vec<F>>,
    inc_opening_point: Vec<F>,
) -> Result<Stage6OutputOpeningEvaluationRequest<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let committed_ra_point_len = config.committed_chunk_bits + config.log_t;
    validate_points(
        "Stage 6 bytecode RA opening",
        &bytecode_ra_opening_points,
        config.bytecode_read_raf_dimensions.num_committed_ra_polys(),
        committed_ra_point_len,
    )?;
    validate_point(
        "Stage 6 booleanity opening",
        &booleanity_opening_point,
        committed_ra_point_len,
    )?;
    validate_point(
        "Stage 6 RAM Hamming booleanity opening",
        &ram_hamming_opening_point,
        config.log_t,
    )?;
    validate_points(
        "Stage 6 RAM RA virtualization opening",
        &ram_ra_opening_points,
        config
            .ram_ra_virtualization_dimensions
            .num_committed_ra_polys(),
        committed_ra_point_len,
    )?;
    validate_points(
        "Stage 6 instruction RA virtualization opening",
        &instruction_ra_opening_points,
        config
            .instruction_ra_virtualization_dimensions
            .num_committed_ra_polys(),
        committed_ra_point_len,
    )?;
    validate_point(
        "Stage 6 increment claim-reduction opening",
        &inc_opening_point,
        config.log_t,
    )?;

    let bytecode_output_openings =
        bytecode::read_raf_output_openings(config.bytecode_read_raf_dimensions);
    let bytecode_read_raf_bytecode_ra = bytecode_output_openings
        .bytecode_ra
        .iter()
        .copied()
        .enumerate()
        .map(|(index, opening)| {
            build_opening_request(
                witness,
                stage6_oracle_from_opening(opening)?,
                bytecode_ra_slot(index),
            )
        })
        .collect::<Result<Vec<_>, ProverError>>()?;

    let booleanity_output_openings =
        booleanity::booleanity_output_openings(config.booleanity_dimensions.layout);
    let instruction_count = config.booleanity_dimensions.layout.instruction();
    let bytecode_count = config.booleanity_dimensions.layout.bytecode();
    let ram_count = config.booleanity_dimensions.layout.ram();
    let expected_booleanity = instruction_count + bytecode_count + ram_count;
    if booleanity_output_openings.len() != expected_booleanity {
        return Err(invalid_stage_request(format!(
            "Stage 6 booleanity output opening count is {}, expected {expected_booleanity}",
            booleanity_output_openings.len()
        )));
    }
    let booleanity_instruction_ra = booleanity_output_openings[..instruction_count]
        .iter()
        .copied()
        .enumerate()
        .map(|(index, opening)| {
            build_opening_request(
                witness,
                stage6_oracle_from_opening(opening)?,
                booleanity_instruction_ra_slot(index),
            )
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let booleanity_bytecode_ra = booleanity_output_openings
        [instruction_count..instruction_count + bytecode_count]
        .iter()
        .copied()
        .enumerate()
        .map(|(index, opening)| {
            build_opening_request(
                witness,
                stage6_oracle_from_opening(opening)?,
                booleanity_bytecode_ra_slot(index),
            )
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let booleanity_ram_ra = booleanity_output_openings[instruction_count + bytecode_count..]
        .iter()
        .copied()
        .enumerate()
        .map(|(index, opening)| {
            build_opening_request(
                witness,
                stage6_oracle_from_opening(opening)?,
                booleanity_ram_ra_slot(index),
            )
        })
        .collect::<Result<Vec<_>, ProverError>>()?;

    let [ram_hamming_opening] = ram::hamming_booleanity_output_openings();
    let ram_hamming_booleanity = build_opening_request(
        witness,
        stage6_oracle_from_opening(ram_hamming_opening)?,
        STAGE6_RAM_HAMMING_BOOLEANITY_OPENING_SLOT,
    )?;

    let ram_ra_virtualization =
        ram::ra_virtualization_output_openings(config.ram_ra_virtualization_dimensions)
            .iter()
            .copied()
            .enumerate()
            .map(|(index, opening)| {
                build_opening_request(
                    witness,
                    stage6_oracle_from_opening(opening)?,
                    ram_ra_virtualization_slot(index),
                )
            })
            .collect::<Result<Vec<_>, ProverError>>()?;

    let instruction_ra_virtualization = instruction::ra_virtualization_output_openings(
        config.instruction_ra_virtualization_dimensions,
    )
    .all()
    .iter()
    .copied()
    .enumerate()
    .map(|(index, opening)| {
        build_opening_request(
            witness,
            stage6_oracle_from_opening(opening)?,
            instruction_ra_virtualization_slot(index),
        )
    })
    .collect::<Result<Vec<_>, ProverError>>()?;

    let [ram_inc, rd_inc] = increments::claim_reduction_output_openings();
    let inc_claim_reduction = [
        build_opening_request(
            witness,
            stage6_oracle_from_opening(ram_inc)?,
            STAGE6_INC_RAM_OPENING_SLOT,
        )?,
        build_opening_request(
            witness,
            stage6_oracle_from_opening(rd_inc)?,
            STAGE6_INC_RD_OPENING_SLOT,
        )?,
    ];

    let mut views = Vec::with_capacity(
        bytecode_read_raf_bytecode_ra.len()
            + booleanity_instruction_ra.len()
            + booleanity_bytecode_ra.len()
            + booleanity_ram_ra.len()
            + 1
            + ram_ra_virtualization.len()
            + instruction_ra_virtualization.len()
            + 2,
    );
    views.extend(
        bytecode_read_raf_bytecode_ra
            .iter()
            .zip(bytecode_ra_opening_points)
            .map(|(opening, point)| {
                SumcheckViewEvaluationRequest::new(opening.slot, opening.view, point)
            }),
    );
    views.extend(
        booleanity_instruction_ra
            .iter()
            .chain(&booleanity_bytecode_ra)
            .chain(&booleanity_ram_ra)
            .map(|opening| {
                SumcheckViewEvaluationRequest::new(
                    opening.slot,
                    opening.view,
                    booleanity_opening_point.clone(),
                )
            }),
    );
    views.push(SumcheckViewEvaluationRequest::new(
        ram_hamming_booleanity.slot,
        ram_hamming_booleanity.view,
        ram_hamming_opening_point,
    ));
    views.extend(ram_ra_virtualization.iter().zip(ram_ra_opening_points).map(
        |(opening, point)| SumcheckViewEvaluationRequest::new(opening.slot, opening.view, point),
    ));
    views.extend(
        instruction_ra_virtualization
            .iter()
            .zip(instruction_ra_opening_points)
            .map(|(opening, point)| {
                SumcheckViewEvaluationRequest::new(opening.slot, opening.view, point)
            }),
    );
    views.extend(inc_claim_reduction.iter().map(|opening| {
        SumcheckViewEvaluationRequest::new(opening.slot, opening.view, inc_opening_point.clone())
    }));
    Ok(Stage6OutputOpeningEvaluationRequest {
        evaluations: SumcheckEvaluationRequest::new("stage6.batch.output_openings", views),
        bytecode_read_raf_bytecode_ra,
        booleanity_instruction_ra,
        booleanity_bytecode_ra,
        booleanity_ram_ra,
        ram_hamming_booleanity,
        ram_ra_virtualization,
        instruction_ra_virtualization,
        inc_claim_reduction,
    })
}

pub const fn bytecode_ra_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE6_BYTECODE_RA_OPENING_SLOT_START + index as u32)
}

pub const fn booleanity_instruction_ra_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE6_BOOLEANITY_INSTRUCTION_RA_OPENING_SLOT_START + index as u32)
}

pub const fn booleanity_bytecode_ra_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE6_BOOLEANITY_BYTECODE_RA_OPENING_SLOT_START + index as u32)
}

pub const fn booleanity_ram_ra_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE6_BOOLEANITY_RAM_RA_OPENING_SLOT_START + index as u32)
}

pub const fn ram_ra_virtualization_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE6_RAM_RA_VIRTUALIZATION_OPENING_SLOT_START + index as u32)
}

pub const fn instruction_ra_virtualization_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE6_INSTRUCTION_RA_VIRTUALIZATION_OPENING_SLOT_START + index as u32)
}

fn build_opening_request<F, W>(
    witness: &W,
    oracle: OracleRef<JoltVmNamespace>,
    slot: BackendValueSlot,
) -> Result<Stage6OutputOpeningRequest, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let view = primary_view_requirement::<F, W, JoltVmNamespace>(witness, oracle)?;
    Ok(Stage6OutputOpeningRequest { slot, view })
}

fn stage6_oracle_from_opening(
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
    let requirements = witness.view_requirements(oracle).map_err(|error| {
        invalid_stage_request(format!("witness view for {:?}: {error}", oracle.kind))
    })?;
    let Some(requirement) = requirements.into_iter().next() else {
        return Err(invalid_stage_request(format!(
            "witness returned no view requirement for {:?}",
            oracle.kind
        )));
    };
    Ok(requirement)
}

fn validate_points<F: Field>(
    label: &'static str,
    points: &[Vec<F>],
    expected_count: usize,
    expected_len: usize,
) -> Result<(), ProverError> {
    if points.len() != expected_count {
        return Err(invalid_stage_request(format!(
            "{label} count is {}, expected {expected_count}",
            points.len()
        )));
    }
    for point in points {
        validate_point(label, point, expected_len)?;
    }
    Ok(())
}

fn validate_point<F: Field>(
    label: &'static str,
    point: &[F],
    expected_len: usize,
) -> Result<(), ProverError> {
    if point.len() != expected_len {
        return Err(invalid_stage_request(format!(
            "{label} point has {} variables, expected {expected_len}",
            point.len()
        )));
    }
    Ok(())
}

fn invalid_stage_request(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidStageRequest {
        reason: error.to_string(),
    }
}
