use jolt_backends::{
    BackendRelationId, BackendValueSlot, SumcheckEvaluationRequest, SumcheckSlot,
    SumcheckViewEvaluationRequest,
};
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltVirtualPolynomial};
use jolt_field::Field;
use jolt_riscv::{CircuitFlags, InstructionFlags};
use jolt_witness::{
    protocols::jolt_vm::{JoltVmNamespace, JOLT_VM_NAMESPACE},
    OracleRef, ViewRequirement, WitnessNamespace, WitnessProvider,
};

use crate::ProverError;

use super::input::{Stage2BatchProverConfig, Stage2ProverConfig};

pub const SPARTAN_PRODUCT_UNISKIP_RELATION: BackendRelationId = BackendRelationId::new(
    JOLT_VM_NAMESPACE.name,
    "spartan_product.uniskip_first_round",
);
pub const STAGE2_PRODUCT_UNISKIP_OPTIMIZATION_IDS: &[&str] = &["OPT-SC-007", "OPT-EQ-004"];

pub const STAGE2_PRODUCT_UNISKIP_SLOT: SumcheckSlot = SumcheckSlot(0);
pub const STAGE2_PRODUCT_UNISKIP_INPUT_SLOT: BackendValueSlot = BackendValueSlot(0);
pub const STAGE2_PRODUCT_UNISKIP_OUTPUT_SLOT: BackendValueSlot = BackendValueSlot(1);

pub const STAGE2_PRODUCT_REMAINDER_OPENING_SLOT_START: u32 = 16;
pub const STAGE2_INSTRUCTION_CLAIM_OPENING_SLOT_START: u32 = 32;
pub const STAGE2_RAM_READ_WRITE_OPENING_SLOT_START: u32 = 48;
pub const STAGE2_RAM_TERMINAL_OPENING_SLOT_START: u32 = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ProductRemainderOpeningEvaluationRequest<F: Field> {
    pub evaluations: SumcheckEvaluationRequest<F, JoltVmNamespace>,
    pub openings: Vec<Stage2ProductRemainderOpeningRequest>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2ProductRemainderOpeningRequest {
    pub variable: JoltVirtualPolynomial,
    pub slot: BackendValueSlot,
    pub view: ViewRequirement<JoltVmNamespace>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2InstructionClaimOpeningEvaluationRequest<F: Field> {
    pub evaluations: SumcheckEvaluationRequest<F, JoltVmNamespace>,
    pub openings: Vec<Stage2InstructionClaimOpeningRequest>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2InstructionClaimOpeningRequest {
    pub variable: JoltVirtualPolynomial,
    pub slot: BackendValueSlot,
    pub view: ViewRequirement<JoltVmNamespace>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2RamReadWriteOpeningEvaluationRequest<F: Field> {
    pub evaluations: SumcheckEvaluationRequest<F, JoltVmNamespace>,
    pub openings: Vec<Stage2RamReadWriteOpeningRequest>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2RamReadWriteOpeningRequest {
    pub oracle: Stage2RamReadWriteOpeningOracle,
    pub slot: BackendValueSlot,
    pub view: ViewRequirement<JoltVmNamespace>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage2RamReadWriteOpeningOracle {
    RamVal,
    RamRa,
    RamInc,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2RamTerminalOpeningEvaluationRequest<F: Field> {
    pub evaluations: SumcheckEvaluationRequest<F, JoltVmNamespace>,
    pub openings: Vec<Stage2RamTerminalOpeningRequest>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2RamTerminalOpeningRequest {
    pub oracle: Stage2RamTerminalOpeningOracle,
    pub slot: BackendValueSlot,
    pub view: ViewRequirement<JoltVmNamespace>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage2RamTerminalOpeningOracle {
    RamRafEvaluationRa,
    RamOutputCheckValFinal,
}

pub fn build_stage2_product_remainder_opening_evaluation_request<F, W>(
    config: Stage2ProverConfig,
    witness: &W,
    opening_point: Vec<F>,
) -> Result<Stage2ProductRemainderOpeningEvaluationRequest<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    if opening_point.len() != config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 product-remainder opening point has {} variables, expected {}",
                opening_point.len(),
                config.log_t
            ),
        });
    }

    let openings = product_remainder_opening_variables()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, variable)| {
            let oracle = OracleRef::virtual_polynomial(variable);
            let view = primary_view_requirement::<F, W, JoltVmNamespace>(witness, oracle)?;
            Ok(Stage2ProductRemainderOpeningRequest {
                variable,
                slot: product_remainder_opening_slot(index),
                view,
            })
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let views = openings
        .iter()
        .map(|opening| {
            SumcheckViewEvaluationRequest::new(opening.slot, opening.view, opening_point.clone())
        })
        .collect();

    Ok(Stage2ProductRemainderOpeningEvaluationRequest {
        evaluations: SumcheckEvaluationRequest::new(
            "stage2.product_remainder.output_openings",
            views,
        ),
        openings,
    })
}

pub fn build_stage2_instruction_claim_opening_evaluation_request<F, W>(
    config: Stage2ProverConfig,
    witness: &W,
    opening_point: Vec<F>,
) -> Result<Stage2InstructionClaimOpeningEvaluationRequest<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    if opening_point.len() != config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 instruction-claim opening point has {} variables, expected {}",
                opening_point.len(),
                config.log_t
            ),
        });
    }

    let openings = instruction_claim_opening_variables()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, variable)| {
            let oracle = OracleRef::virtual_polynomial(variable);
            let view = primary_view_requirement::<F, W, JoltVmNamespace>(witness, oracle)?;
            Ok(Stage2InstructionClaimOpeningRequest {
                variable,
                slot: instruction_claim_opening_slot(index),
                view,
            })
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let views = openings
        .iter()
        .map(|opening| {
            SumcheckViewEvaluationRequest::new(opening.slot, opening.view, opening_point.clone())
        })
        .collect();

    Ok(Stage2InstructionClaimOpeningEvaluationRequest {
        evaluations: SumcheckEvaluationRequest::new(
            "stage2.instruction_claim_reduction.output_openings",
            views,
        ),
        openings,
    })
}

pub fn build_stage2_ram_read_write_opening_evaluation_request<F, W>(
    config: Stage2BatchProverConfig,
    witness: &W,
    opening_point: Vec<F>,
) -> Result<Stage2RamReadWriteOpeningEvaluationRequest<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let expected = config.log_k + config.log_t;
    if opening_point.len() != expected {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 RAM read-write opening point has {} variables, expected {expected}",
                opening_point.len()
            ),
        });
    }

    let (_, r_cycle) = opening_point.split_at(config.log_k);
    let r_cycle = r_cycle.to_vec();
    let openings = ram_read_write_opening_oracles()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, oracle)| {
            let oracle_ref = oracle.oracle_ref();
            let view = primary_view_requirement::<F, W, JoltVmNamespace>(witness, oracle_ref)?;
            Ok(Stage2RamReadWriteOpeningRequest {
                oracle,
                slot: ram_read_write_opening_slot(index),
                view,
            })
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let views = openings
        .iter()
        .map(|opening| {
            let point = match opening.oracle {
                Stage2RamReadWriteOpeningOracle::RamInc => r_cycle.clone(),
                Stage2RamReadWriteOpeningOracle::RamVal
                | Stage2RamReadWriteOpeningOracle::RamRa => opening_point.clone(),
            };
            SumcheckViewEvaluationRequest::new(opening.slot, opening.view, point)
        })
        .collect();

    Ok(Stage2RamReadWriteOpeningEvaluationRequest {
        evaluations: SumcheckEvaluationRequest::new("stage2.ram_read_write.output_openings", views),
        openings,
    })
}

pub fn build_stage2_ram_terminal_opening_evaluation_request<F, W>(
    config: Stage2BatchProverConfig,
    witness: &W,
    ram_raf_opening_point: Vec<F>,
    ram_output_check_opening_point: Vec<F>,
) -> Result<Stage2RamTerminalOpeningEvaluationRequest<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let ram_raf_expected = config.log_k + config.log_t;
    if ram_raf_opening_point.len() != ram_raf_expected {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 RAM RAF evaluation opening point has {} variables, expected {ram_raf_expected}",
                ram_raf_opening_point.len()
            ),
        });
    }
    if ram_output_check_opening_point.len() != config.log_k {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 RAM output-check opening point has {} variables, expected {}",
                ram_output_check_opening_point.len(),
                config.log_k
            ),
        });
    }

    let openings = ram_terminal_opening_oracles()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, oracle)| {
            let oracle_ref = oracle.oracle_ref();
            let view = primary_view_requirement::<F, W, JoltVmNamespace>(witness, oracle_ref)?;
            Ok(Stage2RamTerminalOpeningRequest {
                oracle,
                slot: ram_terminal_opening_slot(index),
                view,
            })
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let views = openings
        .iter()
        .map(|opening| {
            let point = match opening.oracle {
                Stage2RamTerminalOpeningOracle::RamRafEvaluationRa => ram_raf_opening_point.clone(),
                Stage2RamTerminalOpeningOracle::RamOutputCheckValFinal => {
                    ram_output_check_opening_point.clone()
                }
            };
            SumcheckViewEvaluationRequest::new(opening.slot, opening.view, point)
        })
        .collect();

    Ok(Stage2RamTerminalOpeningEvaluationRequest {
        evaluations: SumcheckEvaluationRequest::new("stage2.ram_terminal.output_openings", views),
        openings,
    })
}

pub fn product_remainder_opening_variables() -> [JoltVirtualPolynomial; 8] {
    [
        JoltVirtualPolynomial::LeftInstructionInput,
        JoltVirtualPolynomial::RightInstructionInput,
        JoltVirtualPolynomial::OpFlags(CircuitFlags::Jump),
        JoltVirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
        JoltVirtualPolynomial::LookupOutput,
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
        JoltVirtualPolynomial::NextIsNoop,
        JoltVirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
    ]
}

pub const fn product_remainder_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE2_PRODUCT_REMAINDER_OPENING_SLOT_START + index as u32)
}

pub fn instruction_claim_opening_variables() -> [JoltVirtualPolynomial; 2] {
    [
        JoltVirtualPolynomial::LeftLookupOperand,
        JoltVirtualPolynomial::RightLookupOperand,
    ]
}

pub const fn instruction_claim_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE2_INSTRUCTION_CLAIM_OPENING_SLOT_START + index as u32)
}

pub const fn ram_read_write_opening_oracles() -> [Stage2RamReadWriteOpeningOracle; 3] {
    [
        Stage2RamReadWriteOpeningOracle::RamVal,
        Stage2RamReadWriteOpeningOracle::RamRa,
        Stage2RamReadWriteOpeningOracle::RamInc,
    ]
}

pub const fn ram_read_write_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE2_RAM_READ_WRITE_OPENING_SLOT_START + index as u32)
}

pub const fn ram_terminal_opening_oracles() -> [Stage2RamTerminalOpeningOracle; 2] {
    [
        Stage2RamTerminalOpeningOracle::RamRafEvaluationRa,
        Stage2RamTerminalOpeningOracle::RamOutputCheckValFinal,
    ]
}

pub const fn ram_terminal_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE2_RAM_TERMINAL_OPENING_SLOT_START + index as u32)
}

impl Stage2RamReadWriteOpeningOracle {
    pub const fn oracle_ref(self) -> OracleRef<JoltVmNamespace> {
        match self {
            Self::RamVal => OracleRef::virtual_polynomial(JoltVirtualPolynomial::RamVal),
            Self::RamRa => OracleRef::virtual_polynomial(JoltVirtualPolynomial::RamRa),
            Self::RamInc => OracleRef::committed(JoltCommittedPolynomial::RamInc),
        }
    }
}

impl Stage2RamTerminalOpeningOracle {
    pub const fn oracle_ref(self) -> OracleRef<JoltVmNamespace> {
        match self {
            Self::RamRafEvaluationRa => OracleRef::virtual_polynomial(JoltVirtualPolynomial::RamRa),
            Self::RamOutputCheckValFinal => {
                OracleRef::virtual_polynomial(JoltVirtualPolynomial::RamValFinal)
            }
        }
    }
}

pub(super) fn primary_view_requirement<F, W, N>(
    witness: &W,
    oracle: OracleRef<N>,
) -> Result<ViewRequirement<N>, ProverError>
where
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
