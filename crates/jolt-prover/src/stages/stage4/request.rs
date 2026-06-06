use jolt_backends::{
    BackendValueSlot, SumcheckEvaluationRequest, SumcheckMaterializationRequest,
    SumcheckViewEvaluationRequest, SumcheckViewMaterializationRequest,
};
use jolt_claims::protocols::jolt::{
    formulas::dimensions::REGISTER_ADDRESS_BITS, JoltCommittedPolynomial, JoltVirtualPolynomial,
};
use jolt_field::Field;
use jolt_witness::{
    protocols::jolt_vm::JoltVmNamespace, OracleRef, ViewRequirement, WitnessNamespace,
    WitnessProvider,
};

use crate::ProverError;

use super::output::{Stage4RegularBatchInputClaims, Stage4RegularBatchPrefixOutput};

pub const STAGE4_REGISTERS_READ_WRITE_OPENING_SLOT_START: u32 = 0;
pub const STAGE4_RAM_VAL_CHECK_OPENING_SLOT_START: u32 = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4RegularBatchRequest<F: Field> {
    pub input_claims: Stage4RegularBatchInputClaims<F>,
    pub registers_gamma: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_gamma: F,
    pub ram_val_check_gamma: F,
}

impl<F: Field> Stage4RegularBatchRequest<F> {
    pub fn from_prefix(prefix: &Stage4RegularBatchPrefixOutput<F>) -> Self {
        Self {
            input_claims: prefix.input_claims.clone(),
            registers_gamma: prefix.registers_gamma,
            #[cfg(feature = "field-inline")]
            field_registers_gamma: prefix.field_registers_gamma,
            ram_val_check_gamma: prefix.ram_val_check_gamma,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4OutputOpeningEvaluationRequest<F: Field> {
    pub evaluations: SumcheckEvaluationRequest<F, JoltVmNamespace>,
    pub registers_read_write_openings: Vec<Stage4OutputOpeningRequest>,
    pub ram_val_check_openings: Vec<Stage4OutputOpeningRequest>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4OutputOpeningMaterializationRequest {
    pub materializations: SumcheckMaterializationRequest<JoltVmNamespace>,
    pub registers_read_write_openings: Vec<Stage4OutputOpeningRequest>,
    pub ram_val_check_openings: Vec<Stage4OutputOpeningRequest>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4OutputOpeningRequest {
    pub oracle: Stage4OutputOpeningOracle,
    pub slot: BackendValueSlot,
    pub view: ViewRequirement<JoltVmNamespace>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage4OutputOpeningOracle {
    RegistersVal,
    Rs1Ra,
    Rs2Ra,
    RdWa,
    RdInc,
    RamRa,
    RamInc,
}

impl Stage4OutputOpeningOracle {
    pub const fn oracle_ref(self) -> OracleRef<JoltVmNamespace> {
        match self {
            Self::RegistersVal => {
                OracleRef::virtual_polynomial(JoltVirtualPolynomial::RegistersVal)
            }
            Self::Rs1Ra => OracleRef::virtual_polynomial(JoltVirtualPolynomial::Rs1Ra),
            Self::Rs2Ra => OracleRef::virtual_polynomial(JoltVirtualPolynomial::Rs2Ra),
            Self::RdWa => OracleRef::virtual_polynomial(JoltVirtualPolynomial::RdWa),
            Self::RdInc => OracleRef::committed(JoltCommittedPolynomial::RdInc),
            Self::RamRa => OracleRef::virtual_polynomial(JoltVirtualPolynomial::RamRa),
            Self::RamInc => OracleRef::committed(JoltCommittedPolynomial::RamInc),
        }
    }
}

pub fn build_stage4_output_opening_evaluation_request<F, W>(
    config: super::input::Stage4ProverConfig,
    witness: &W,
    registers_read_write_opening_point: Vec<F>,
    ram_val_check_opening_point: Vec<F>,
) -> Result<Stage4OutputOpeningEvaluationRequest<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    if registers_read_write_opening_point.len() != REGISTER_ADDRESS_BITS + config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 register read-write opening point has {} variables, expected {}",
                registers_read_write_opening_point.len(),
                REGISTER_ADDRESS_BITS + config.log_t
            ),
        });
    }
    if ram_val_check_opening_point.len() != config.log_k + config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 RAM value-check opening point has {} variables, expected {}",
                ram_val_check_opening_point.len(),
                config.log_k + config.log_t
            ),
        });
    }

    let (_, register_cycle_point) =
        registers_read_write_opening_point.split_at(REGISTER_ADDRESS_BITS);
    let (_, ram_cycle_point) = ram_val_check_opening_point.split_at(config.log_k);

    let registers_read_write_openings = build_opening_requests(
        witness,
        &registers_read_write_opening_oracles(),
        registers_read_write_opening_slot,
    )?;
    let ram_val_check_openings = build_opening_requests(
        witness,
        &ram_val_check_opening_oracles(),
        ram_val_check_opening_slot,
    )?;

    let mut views =
        Vec::with_capacity(registers_read_write_openings.len() + ram_val_check_openings.len());
    views.extend(registers_read_write_openings.iter().map(|opening| {
        let point = match opening.oracle {
            Stage4OutputOpeningOracle::RdInc => register_cycle_point.to_vec(),
            _ => registers_read_write_opening_point.clone(),
        };
        SumcheckViewEvaluationRequest::new(opening.slot, opening.view, point)
    }));
    views.extend(ram_val_check_openings.iter().map(|opening| {
        let point = match opening.oracle {
            Stage4OutputOpeningOracle::RamInc => ram_cycle_point.to_vec(),
            _ => ram_val_check_opening_point.clone(),
        };
        SumcheckViewEvaluationRequest::new(opening.slot, opening.view, point)
    }));

    Ok(Stage4OutputOpeningEvaluationRequest {
        evaluations: SumcheckEvaluationRequest::new("stage4.batch.output_openings", views),
        registers_read_write_openings,
        ram_val_check_openings,
    })
}

pub fn build_stage4_output_opening_materialization_request<F, W>(
    witness: &W,
) -> Result<Stage4OutputOpeningMaterializationRequest, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let registers_read_write_openings = build_opening_requests(
        witness,
        &registers_read_write_opening_oracles(),
        registers_read_write_opening_slot,
    )?;
    let ram_val_check_openings = build_opening_requests(
        witness,
        &ram_val_check_opening_oracles(),
        ram_val_check_opening_slot,
    )?;

    let mut views =
        Vec::with_capacity(registers_read_write_openings.len() + ram_val_check_openings.len());
    views.extend(
        registers_read_write_openings
            .iter()
            .map(|opening| SumcheckViewMaterializationRequest::new(opening.slot, opening.view)),
    );
    views.extend(
        ram_val_check_openings
            .iter()
            .map(|opening| SumcheckViewMaterializationRequest::new(opening.slot, opening.view)),
    );

    Ok(Stage4OutputOpeningMaterializationRequest {
        materializations: SumcheckMaterializationRequest::new(
            "stage4.batch.output_materializations",
            views,
        ),
        registers_read_write_openings,
        ram_val_check_openings,
    })
}

pub const fn registers_read_write_opening_oracles() -> [Stage4OutputOpeningOracle; 5] {
    [
        Stage4OutputOpeningOracle::RegistersVal,
        Stage4OutputOpeningOracle::Rs1Ra,
        Stage4OutputOpeningOracle::Rs2Ra,
        Stage4OutputOpeningOracle::RdWa,
        Stage4OutputOpeningOracle::RdInc,
    ]
}

pub const fn ram_val_check_opening_oracles() -> [Stage4OutputOpeningOracle; 2] {
    [
        Stage4OutputOpeningOracle::RamRa,
        Stage4OutputOpeningOracle::RamInc,
    ]
}

pub const fn registers_read_write_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE4_REGISTERS_READ_WRITE_OPENING_SLOT_START + index as u32)
}

pub const fn ram_val_check_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE4_RAM_VAL_CHECK_OPENING_SLOT_START + index as u32)
}

fn build_opening_requests<F, W>(
    witness: &W,
    oracles: &[Stage4OutputOpeningOracle],
    slot_for_index: impl Fn(usize) -> BackendValueSlot,
) -> Result<Vec<Stage4OutputOpeningRequest>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    oracles
        .iter()
        .copied()
        .enumerate()
        .map(|(index, oracle)| {
            let oracle_ref = oracle.oracle_ref();
            let view = primary_view_requirement::<F, W, JoltVmNamespace>(witness, oracle_ref)?;
            Ok(Stage4OutputOpeningRequest {
                oracle,
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
