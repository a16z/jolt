use jolt_backends::{
    BackendRelationId, BackendValueSlot, SumcheckEvaluationRequest, SumcheckInstanceRequest,
    SumcheckMaterializationRequest, SumcheckRequest, SumcheckSlot, SumcheckViewEvaluationRequest,
    SumcheckViewMaterializationRequest,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    formulas::spartan::FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS, FieldInlineVirtualPolynomial,
};
use jolt_claims::protocols::jolt::JoltVirtualPolynomial;
use jolt_field::Field;
#[cfg(feature = "field-inline")]
use jolt_witness::protocols::jolt_vm::field_inline::FieldInlineNamespace;
use jolt_witness::{
    protocols::jolt_vm::{JoltVmNamespace, JOLT_VM_NAMESPACE},
    OracleRef, ViewRequirement, WitnessNamespace, WitnessProvider,
};

use crate::ProverError;

use super::input::Stage1ProverConfig;

pub const SPARTAN_OUTER_UNISKIP_RELATION: BackendRelationId =
    BackendRelationId::new(JOLT_VM_NAMESPACE.name, "spartan_outer.uniskip_first_round");
pub const SPARTAN_OUTER_REMAINDER_RELATION: BackendRelationId =
    BackendRelationId::new(JOLT_VM_NAMESPACE.name, "spartan_outer.remainder");
pub const STAGE1_SPARTAN_OUTER_OPTIMIZATION_IDS: &[&str] =
    &["OPT-SC-007", "OPT-SC-008", "OPT-EQ-004"];

pub const STAGE1_UNISKIP_SLOT: SumcheckSlot = SumcheckSlot(0);
pub const STAGE1_REMAINDER_SLOT: SumcheckSlot = SumcheckSlot(1);

pub const STAGE1_UNISKIP_INPUT_SLOT: BackendValueSlot = BackendValueSlot(0);
pub const STAGE1_UNISKIP_OUTPUT_SLOT: BackendValueSlot = BackendValueSlot(1);
pub const STAGE1_REMAINDER_OUTPUT_SLOT: BackendValueSlot = BackendValueSlot(2);
pub const STAGE1_R1CS_INPUT_SLOT_START: u32 = 16;
#[cfg(feature = "field-inline")]
pub const STAGE1_FIELD_INLINE_R1CS_INPUT_SLOT_START: u32 = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1Request {
    pub sumchecks: SumcheckRequest<JoltVmNamespace>,
    pub r1cs_inputs: Vec<Stage1R1csInputRequest>,
}

impl Stage1Request {
    pub fn expected_value_slots(&self) -> Vec<BackendValueSlot> {
        let mut slots = Vec::with_capacity(2 + self.r1cs_inputs.len());
        slots.push(STAGE1_UNISKIP_OUTPUT_SLOT);
        slots.push(STAGE1_REMAINDER_OUTPUT_SLOT);
        slots.extend(self.r1cs_inputs.iter().map(|input| input.slot));
        slots
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1R1csEvaluationRequest<F: Field> {
    pub evaluations: SumcheckEvaluationRequest<F, JoltVmNamespace>,
    pub r1cs_inputs: Vec<Stage1R1csInputRequest>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1R1csMaterializationRequest {
    pub materializations: SumcheckMaterializationRequest<JoltVmNamespace>,
    pub r1cs_inputs: Vec<Stage1R1csInputRequest>,
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1FieldInlineR1csEvaluationRequest<F: Field> {
    pub evaluations: SumcheckEvaluationRequest<F, FieldInlineNamespace>,
    pub r1cs_inputs: Vec<Stage1FieldInlineR1csInputRequest>,
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1FieldInlineR1csMaterializationRequest {
    pub materializations: SumcheckMaterializationRequest<FieldInlineNamespace>,
    pub r1cs_inputs: Vec<Stage1FieldInlineR1csInputRequest>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1R1csInputRequest {
    pub variable: JoltVirtualPolynomial,
    pub slot: BackendValueSlot,
    pub view: ViewRequirement<JoltVmNamespace>,
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1FieldInlineR1csInputRequest {
    pub variable: FieldInlineVirtualPolynomial,
    pub slot: BackendValueSlot,
    pub view: ViewRequirement<FieldInlineNamespace>,
}

pub fn build_stage1_request<F, W>(
    config: Stage1ProverConfig,
    witness: &W,
) -> Result<Stage1Request, ProverError>
where
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let dimensions = config.dimensions();
    let r1cs_inputs = dimensions
        .variables()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, variable)| {
            let oracle = OracleRef::virtual_polynomial(variable);
            let view = primary_view_requirement::<F, W, JoltVmNamespace>(witness, oracle)?;
            Ok(Stage1R1csInputRequest {
                variable,
                slot: r1cs_input_slot(index),
                view,
            })
        })
        .collect::<Result<Vec<_>, ProverError>>()?;

    let witness_views = r1cs_inputs
        .iter()
        .map(|input| input.view)
        .collect::<Vec<_>>();
    let uniskip_spec = dimensions.uniskip_sumcheck();
    let remainder_spec = dimensions.remainder_sumcheck();

    let uniskip = with_zk_round_mode(
        SumcheckInstanceRequest::new(
            STAGE1_UNISKIP_SLOT,
            SPARTAN_OUTER_UNISKIP_RELATION,
            witness_views.clone(),
            uniskip_spec.rounds,
            uniskip_spec.degree,
            STAGE1_UNISKIP_INPUT_SLOT,
            STAGE1_UNISKIP_OUTPUT_SLOT,
        )
        .with_optimization_ids(STAGE1_SPARTAN_OUTER_OPTIMIZATION_IDS),
        config,
    );
    let remainder = with_zk_round_mode(
        SumcheckInstanceRequest::new(
            STAGE1_REMAINDER_SLOT,
            SPARTAN_OUTER_REMAINDER_RELATION,
            witness_views,
            remainder_spec.rounds,
            remainder_spec.degree,
            STAGE1_UNISKIP_OUTPUT_SLOT,
            STAGE1_REMAINDER_OUTPUT_SLOT,
        )
        .with_optimization_ids(STAGE1_SPARTAN_OUTER_OPTIMIZATION_IDS),
        config,
    );

    Ok(Stage1Request {
        sumchecks: SumcheckRequest::new("stage1.spartan_outer", vec![uniskip, remainder]),
        r1cs_inputs,
    })
}

pub fn build_stage1_r1cs_evaluation_request<F, W>(
    config: Stage1ProverConfig,
    witness: &W,
    point: Vec<F>,
) -> Result<Stage1R1csEvaluationRequest<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    if point.len() != config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 1 R1CS evaluation point has {} variables, expected {}",
                point.len(),
                config.log_t
            ),
        });
    }
    let r1cs_inputs = config
        .dimensions()
        .variables()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, variable)| {
            let oracle = OracleRef::virtual_polynomial(variable);
            let view = primary_view_requirement::<F, W, JoltVmNamespace>(witness, oracle)?;
            Ok(Stage1R1csInputRequest {
                variable,
                slot: r1cs_input_slot(index),
                view,
            })
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let views = r1cs_inputs
        .iter()
        .map(|input| SumcheckViewEvaluationRequest::new(input.slot, input.view, point.clone()))
        .collect();

    Ok(Stage1R1csEvaluationRequest {
        evaluations: SumcheckEvaluationRequest::new("stage1.spartan_outer.r1cs_inputs", views),
        r1cs_inputs,
    })
}

pub fn build_stage1_r1cs_materialization_request<F, W>(
    config: Stage1ProverConfig,
    witness: &W,
) -> Result<Stage1R1csMaterializationRequest, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let r1cs_inputs = config
        .dimensions()
        .variables()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, variable)| {
            let oracle = OracleRef::virtual_polynomial(variable);
            let view = primary_view_requirement::<F, W, JoltVmNamespace>(witness, oracle)?;
            Ok(Stage1R1csInputRequest {
                variable,
                slot: r1cs_input_slot(index),
                view,
            })
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let views = r1cs_inputs
        .iter()
        .map(|input| SumcheckViewMaterializationRequest::new(input.slot, input.view))
        .collect();

    Ok(Stage1R1csMaterializationRequest {
        materializations: SumcheckMaterializationRequest::new(
            "stage1.spartan_outer.r1cs_inputs.materialize",
            views,
        ),
        r1cs_inputs,
    })
}

#[cfg(feature = "field-inline")]
pub fn build_stage1_field_inline_r1cs_materialization_request<F, W>(
    config: Stage1ProverConfig,
    witness: &W,
) -> Result<Stage1FieldInlineR1csMaterializationRequest, ProverError>
where
    F: Field,
    W: WitnessProvider<F, FieldInlineNamespace>,
{
    let r1cs_inputs = FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS
        .iter()
        .copied()
        .enumerate()
        .map(|(index, variable)| {
            let oracle = OracleRef::virtual_polynomial(variable);
            let view = primary_view_requirement::<F, W, FieldInlineNamespace>(witness, oracle)?;
            Ok(Stage1FieldInlineR1csInputRequest {
                variable,
                slot: field_inline_r1cs_input_slot(index),
                view,
            })
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let views = r1cs_inputs
        .iter()
        .map(|input| SumcheckViewMaterializationRequest::new(input.slot, input.view))
        .collect();

    let _ = config;
    Ok(Stage1FieldInlineR1csMaterializationRequest {
        materializations: SumcheckMaterializationRequest::new(
            "stage1.field_inline.spartan_outer.r1cs_inputs.materialize",
            views,
        ),
        r1cs_inputs,
    })
}

#[cfg(feature = "field-inline")]
pub fn build_stage1_field_inline_r1cs_evaluation_request<F, W>(
    config: Stage1ProverConfig,
    witness: &W,
    point: Vec<F>,
) -> Result<Stage1FieldInlineR1csEvaluationRequest<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, FieldInlineNamespace>,
{
    if point.len() != config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 1 field-inline R1CS evaluation point has {} variables, expected {}",
                point.len(),
                config.log_t
            ),
        });
    }
    let r1cs_inputs = FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS
        .iter()
        .copied()
        .enumerate()
        .map(|(index, variable)| {
            let oracle = OracleRef::virtual_polynomial(variable);
            let view = primary_view_requirement::<F, W, FieldInlineNamespace>(witness, oracle)?;
            Ok(Stage1FieldInlineR1csInputRequest {
                variable,
                slot: field_inline_r1cs_input_slot(index),
                view,
            })
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let views = r1cs_inputs
        .iter()
        .map(|input| SumcheckViewEvaluationRequest::new(input.slot, input.view, point.clone()))
        .collect();

    Ok(Stage1FieldInlineR1csEvaluationRequest {
        evaluations: SumcheckEvaluationRequest::new(
            "stage1.field_inline.spartan_outer.r1cs_inputs",
            views,
        ),
        r1cs_inputs,
    })
}

pub const fn r1cs_input_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE1_R1CS_INPUT_SLOT_START + index as u32)
}

#[cfg(feature = "field-inline")]
pub const fn field_inline_r1cs_input_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE1_FIELD_INLINE_R1CS_INPUT_SLOT_START + index as u32)
}

fn primary_view_requirement<F, W, N>(
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

fn with_zk_round_mode(
    instance: SumcheckInstanceRequest<JoltVmNamespace>,
    config: Stage1ProverConfig,
) -> SumcheckInstanceRequest<JoltVmNamespace> {
    #[cfg(feature = "zk")]
    {
        let mut instance = instance;
        instance.committed_rounds = config.committed_rounds;
        instance
    }
    #[cfg(not(feature = "zk"))]
    {
        let _ = config;
        instance
    }
}
