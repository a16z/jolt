use jolt_backends::{
    BackendValueSlot, SumcheckEvaluationRequest, SumcheckMaterializationRequest,
    SumcheckViewEvaluationRequest, SumcheckViewMaterializationRequest,
};
use jolt_claims::protocols::jolt::JoltVirtualPolynomial;
use jolt_field::Field;
use jolt_riscv::{CircuitFlags, InstructionFlags};
use jolt_witness::{
    protocols::jolt_vm::JoltVmNamespace, OracleRef, ViewRequirement, WitnessNamespace,
    WitnessProvider,
};

use crate::ProverError;

use super::input::Stage3ProverConfig;

pub const STAGE3_SHIFT_OPENING_SLOT_START: u32 = 0;
pub const STAGE3_INSTRUCTION_INPUT_OPENING_SLOT_START: u32 = 16;
pub const STAGE3_REGISTERS_CLAIM_REDUCTION_OPENING_SLOT_START: u32 = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3OutputOpeningEvaluationRequest<F: Field> {
    pub evaluations: SumcheckEvaluationRequest<F, JoltVmNamespace>,
    pub shift_openings: Vec<Stage3OutputOpeningRequest>,
    pub instruction_input_openings: Vec<Stage3OutputOpeningRequest>,
    pub registers_claim_reduction_openings: Vec<Stage3OutputOpeningRequest>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3OutputOpeningMaterializationRequest {
    pub materializations: SumcheckMaterializationRequest<JoltVmNamespace>,
    pub shift_openings: Vec<Stage3OutputOpeningRequest>,
    pub instruction_input_openings: Vec<Stage3OutputOpeningRequest>,
    pub registers_claim_reduction_openings: Vec<Stage3OutputOpeningRequest>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3OutputOpeningRequest {
    pub variable: JoltVirtualPolynomial,
    pub slot: BackendValueSlot,
    pub view: ViewRequirement<JoltVmNamespace>,
}

pub fn build_stage3_output_opening_evaluation_request<F, W>(
    config: Stage3ProverConfig,
    witness: &W,
    shift_opening_point: Vec<F>,
    instruction_input_opening_point: Vec<F>,
    registers_claim_reduction_opening_point: Vec<F>,
) -> Result<Stage3OutputOpeningEvaluationRequest<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    validate_stage3_opening_point("shift", config, shift_opening_point.len())?;
    validate_stage3_opening_point(
        "instruction-input",
        config,
        instruction_input_opening_point.len(),
    )?;
    validate_stage3_opening_point(
        "registers claim-reduction",
        config,
        registers_claim_reduction_opening_point.len(),
    )?;

    let shift_openings = build_opening_requests(
        witness,
        &stage3_shift_opening_variables(),
        shift_opening_slot,
    )?;
    let instruction_input_openings = build_opening_requests(
        witness,
        &stage3_instruction_input_opening_variables(),
        instruction_input_opening_slot,
    )?;
    let registers_claim_reduction_openings = build_opening_requests(
        witness,
        &stage3_registers_claim_reduction_opening_variables(),
        registers_claim_reduction_opening_slot,
    )?;

    let mut views = Vec::with_capacity(
        shift_openings.len()
            + instruction_input_openings.len()
            + registers_claim_reduction_openings.len(),
    );
    views.extend(shift_openings.iter().map(|opening| {
        SumcheckViewEvaluationRequest::new(opening.slot, opening.view, shift_opening_point.clone())
    }));
    views.extend(instruction_input_openings.iter().map(|opening| {
        SumcheckViewEvaluationRequest::new(
            opening.slot,
            opening.view,
            instruction_input_opening_point.clone(),
        )
    }));
    views.extend(registers_claim_reduction_openings.iter().map(|opening| {
        SumcheckViewEvaluationRequest::new(
            opening.slot,
            opening.view,
            registers_claim_reduction_opening_point.clone(),
        )
    }));

    Ok(Stage3OutputOpeningEvaluationRequest {
        evaluations: SumcheckEvaluationRequest::new("stage3.batch.output_openings", views),
        shift_openings,
        instruction_input_openings,
        registers_claim_reduction_openings,
    })
}

pub fn build_stage3_output_opening_materialization_request<F, W>(
    witness: &W,
) -> Result<Stage3OutputOpeningMaterializationRequest, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let shift_openings = build_opening_requests(
        witness,
        &stage3_shift_opening_variables(),
        shift_opening_slot,
    )?;
    let instruction_input_openings = build_opening_requests(
        witness,
        &stage3_instruction_input_opening_variables(),
        instruction_input_opening_slot,
    )?;
    let registers_claim_reduction_openings = build_opening_requests(
        witness,
        &stage3_registers_claim_reduction_opening_variables(),
        registers_claim_reduction_opening_slot,
    )?;

    let mut views = Vec::with_capacity(
        shift_openings.len()
            + instruction_input_openings.len()
            + registers_claim_reduction_openings.len(),
    );
    views.extend(
        shift_openings
            .iter()
            .map(|opening| SumcheckViewMaterializationRequest::new(opening.slot, opening.view)),
    );
    views.extend(
        instruction_input_openings
            .iter()
            .map(|opening| SumcheckViewMaterializationRequest::new(opening.slot, opening.view)),
    );
    views.extend(
        registers_claim_reduction_openings
            .iter()
            .map(|opening| SumcheckViewMaterializationRequest::new(opening.slot, opening.view)),
    );

    Ok(Stage3OutputOpeningMaterializationRequest {
        materializations: SumcheckMaterializationRequest::new(
            "stage3.batch.output_materializations",
            views,
        ),
        shift_openings,
        instruction_input_openings,
        registers_claim_reduction_openings,
    })
}

pub fn build_stage3_instruction_registers_opening_materialization_request<F, W>(
    witness: &W,
) -> Result<Stage3OutputOpeningMaterializationRequest, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let instruction_input_openings = build_opening_requests(
        witness,
        &stage3_instruction_input_opening_variables(),
        instruction_input_opening_slot,
    )?;
    let registers_claim_reduction_openings = build_opening_requests(
        witness,
        &stage3_registers_claim_reduction_opening_variables(),
        registers_claim_reduction_opening_slot,
    )?;

    let mut views = Vec::with_capacity(
        instruction_input_openings.len() + registers_claim_reduction_openings.len(),
    );
    views.extend(
        instruction_input_openings
            .iter()
            .map(|opening| SumcheckViewMaterializationRequest::new(opening.slot, opening.view)),
    );
    views.extend(
        registers_claim_reduction_openings
            .iter()
            .map(|opening| SumcheckViewMaterializationRequest::new(opening.slot, opening.view)),
    );

    Ok(Stage3OutputOpeningMaterializationRequest {
        materializations: SumcheckMaterializationRequest::new(
            "stage3.batch.instruction_registers_output_materializations",
            views,
        ),
        shift_openings: Vec::new(),
        instruction_input_openings,
        registers_claim_reduction_openings,
    })
}

pub fn stage3_shift_opening_variables() -> [JoltVirtualPolynomial; 5] {
    [
        JoltVirtualPolynomial::UnexpandedPC,
        JoltVirtualPolynomial::PC,
        JoltVirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
        JoltVirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
    ]
}

pub const fn shift_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE3_SHIFT_OPENING_SLOT_START + index as u32)
}

pub fn stage3_instruction_input_opening_variables() -> [JoltVirtualPolynomial; 8] {
    [
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsRs2Value),
        JoltVirtualPolynomial::Rs2Value,
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsImm),
        JoltVirtualPolynomial::Imm,
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsRs1Value),
        JoltVirtualPolynomial::Rs1Value,
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsPC),
        JoltVirtualPolynomial::UnexpandedPC,
    ]
}

pub const fn instruction_input_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE3_INSTRUCTION_INPUT_OPENING_SLOT_START + index as u32)
}

pub fn stage3_registers_claim_reduction_opening_variables() -> [JoltVirtualPolynomial; 3] {
    [
        JoltVirtualPolynomial::RdWriteValue,
        JoltVirtualPolynomial::Rs1Value,
        JoltVirtualPolynomial::Rs2Value,
    ]
}

pub const fn registers_claim_reduction_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE3_REGISTERS_CLAIM_REDUCTION_OPENING_SLOT_START + index as u32)
}

fn validate_stage3_opening_point(
    label: &'static str,
    config: Stage3ProverConfig,
    actual: usize,
) -> Result<(), ProverError> {
    if actual == config.log_t {
        return Ok(());
    }
    Err(ProverError::InvalidStageRequest {
        reason: format!(
            "Stage 3 {label} opening point has {actual} variables, expected {}",
            config.log_t
        ),
    })
}

fn build_opening_requests<F, W>(
    witness: &W,
    variables: &[JoltVirtualPolynomial],
    slot_for_index: impl Fn(usize) -> BackendValueSlot,
) -> Result<Vec<Stage3OutputOpeningRequest>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    variables
        .iter()
        .copied()
        .enumerate()
        .map(|(index, variable)| {
            let oracle = OracleRef::virtual_polynomial(variable);
            let view = primary_view_requirement::<F, W, JoltVmNamespace>(witness, oracle)?;
            Ok(Stage3OutputOpeningRequest {
                variable,
                slot: slot_for_index(index),
                view,
            })
        })
        .collect()
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
    if requirement.oracle.kind != oracle.kind {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "witness returned requirement for {:?}, expected {oracle:?}",
                requirement.oracle.kind,
                oracle = oracle.kind
            ),
        });
    }
    Ok(requirement)
}
