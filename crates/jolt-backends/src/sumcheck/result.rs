use jolt_field::Field;
use jolt_poly::{Polynomial, TensorEqTable, UnivariatePoly};
use jolt_witness::{OracleDescriptor, ViewRequirement, WitnessNamespace};

use crate::{BackendValueSlot, SumcheckSlot};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedSumcheckView<N: WitnessNamespace> {
    pub slot: SumcheckSlot,
    pub view_index: usize,
    pub requirement: ViewRequirement<N>,
    pub descriptor: OracleDescriptor<N>,
}

impl<N: WitnessNamespace> ResolvedSumcheckView<N> {
    pub const fn new(
        slot: SumcheckSlot,
        view_index: usize,
        requirement: ViewRequirement<N>,
        descriptor: OracleDescriptor<N>,
    ) -> Self {
        Self {
            slot,
            view_index,
            requirement,
            descriptor,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckViewResolution<N: WitnessNamespace> {
    pub resolved_witness: Vec<ResolvedSumcheckView<N>>,
}

impl<N: WitnessNamespace> SumcheckViewResolution<N> {
    pub const fn new(resolved_witness: Vec<ResolvedSumcheckView<N>>) -> Self {
        Self { resolved_witness }
    }

    pub fn is_empty(&self) -> bool {
        self.resolved_witness.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckProofOutput<Proof> {
    pub slot: SumcheckSlot,
    pub proof: Proof,
}

impl<Proof> SumcheckProofOutput<Proof> {
    pub const fn new(slot: SumcheckSlot, proof: Proof) -> Self {
        Self { slot, proof }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckMaterializationOutput<F: Field> {
    pub slot: BackendValueSlot,
    pub values: Vec<F>,
}

impl<F: Field> SumcheckMaterializationOutput<F> {
    pub const fn new(slot: BackendValueSlot, values: Vec<F>) -> Self {
        Self { slot, values }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckLinearProductOutput<F: Field> {
    pub slot: BackendValueSlot,
    pub value: F,
}

impl<F: Field> SumcheckLinearProductOutput<F> {
    pub const fn new(slot: BackendValueSlot, value: F) -> Self {
        Self { slot, value }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckFieldRegistersValEvaluationOutput<F: Field> {
    pub field_rd_inc: F,
    pub field_rd_wa: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckSpartanOuterRemainderState<F: Field> {
    pub label: &'static str,
    pub eq_point: Vec<F>,
    pub eq_tables: Vec<TensorEqTable<F>>,
    pub left: Vec<F>,
    pub right: Vec<F>,
    pub active_len: usize,
    pub scale: F,
}

impl<F: Field> SumcheckSpartanOuterRemainderState<F> {
    pub fn new(
        label: &'static str,
        eq_point: Vec<F>,
        left: Vec<F>,
        right: Vec<F>,
        active_len: usize,
        scale: F,
    ) -> Self {
        let active_log = active_len.checked_ilog2().map_or(0, |bits| bits as usize);
        let eq_tables = (0..active_log)
            .map(|vars| TensorEqTable::<F>::new(&eq_point[..vars.min(eq_point.len())]))
            .collect();
        Self {
            label,
            eq_point,
            eq_tables,
            left,
            right,
            active_len,
            scale,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckSpartanOuterRemainderRound<F: Field> {
    pub q_at_zero: F,
    pub q_at_infinity: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRegularBatchRound<F: Field> {
    pub instance_index: usize,
    pub polynomial: UnivariatePoly<F>,
}

#[derive(Clone, Debug)]
pub struct SumcheckStage7HammingState<F: Field> {
    pub label: &'static str,
    pub g: Vec<Polynomial<F>>,
    pub eq_bool: Polynomial<F>,
    pub eq_virt: Vec<Polynomial<F>>,
    pub gamma_powers: Vec<F>,
    pub scratch_g: Vec<Vec<F>>,
    pub scratch_eq_bool: Vec<F>,
    pub scratch_eq_virt: Vec<Vec<F>>,
}

impl<F: Field> SumcheckStage7HammingState<F> {
    pub const fn new(
        label: &'static str,
        g: Vec<Polynomial<F>>,
        eq_bool: Polynomial<F>,
        eq_virt: Vec<Polynomial<F>>,
        gamma_powers: Vec<F>,
    ) -> Self {
        Self {
            label,
            g,
            eq_bool,
            eq_virt,
            gamma_powers,
            scratch_g: Vec::new(),
            scratch_eq_bool: Vec::new(),
            scratch_eq_virt: Vec::new(),
        }
    }

    pub fn num_polys(&self) -> usize {
        self.g.len()
    }

    pub fn num_rounds(&self) -> usize {
        self.g.first().map_or(0, Polynomial::num_vars)
    }
}

impl<F: Field> PartialEq for SumcheckStage7HammingState<F> {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label
            && self.g == other.g
            && self.eq_bool == other.eq_bool
            && self.eq_virt == other.eq_virt
            && self.gamma_powers == other.gamma_powers
    }
}

impl<F: Field> Eq for SumcheckStage7HammingState<F> {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckStage7AdviceAddressState<F: Field> {
    pub label: &'static str,
    pub rounds: usize,
    pub advice_words: Vec<u64>,
    pub bound_advice: Option<Polynomial<F>>,
    pub eq: Polynomial<F>,
    pub scale: F,
}

impl<F: Field> SumcheckStage7AdviceAddressState<F> {
    pub const fn new(
        label: &'static str,
        rounds: usize,
        advice_words: Vec<u64>,
        eq: Polynomial<F>,
        scale: F,
    ) -> Self {
        Self {
            label,
            rounds,
            advice_words,
            bound_advice: None,
            eq,
            scale,
        }
    }

    pub const fn num_rounds(&self) -> usize {
        self.rounds
    }

    pub fn final_advice_opening(&self) -> Option<F> {
        if let Some(advice) = &self.bound_advice {
            return (advice.len() == 1).then(|| advice.evaluations()[0]);
        }
        (self.advice_words.len() == 1).then(|| F::from_u64(self.advice_words[0]))
    }
}

impl<F: Field> SumcheckRegularBatchRound<F> {
    pub const fn new(instance_index: usize, polynomial: UnivariatePoly<F>) -> Self {
        Self {
            instance_index,
            polynomial,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckRegistersReadWriteOutput<F: Field> {
    pub registers_val: F,
    pub rs1_ra: F,
    pub rs2_ra: F,
    pub rd_wa: F,
    pub rd_inc: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckRamValCheckOutput<F: Field> {
    pub ram_ra: F,
    pub ram_inc: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckRamRaClaimReductionOutput<F: Field> {
    pub ram_ra: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckInstructionReadRafOutput<F: Field> {
    pub lookup_table_flags: Vec<F>,
    pub instruction_ra: Vec<F>,
    pub instruction_raf_flag: F,
    pub handoff_claim: F,
    pub final_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckBytecodeReadRafOutput<F: Field> {
    pub bytecode_ra: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckBooleanityOutput<F: Field> {
    pub instruction_ra: Vec<F>,
    pub bytecode_ra: Vec<F>,
    pub ram_ra: Vec<F>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckRamHammingBooleanityOutput<F: Field> {
    pub ram_hamming_weight: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRamRaVirtualizationOutput<F: Field> {
    pub ram_ra: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckInstructionRaVirtualizationOutput<F: Field> {
    pub instruction_ra: Vec<F>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckIncClaimReductionOutput<F: Field> {
    pub ram_inc: F,
    pub rd_inc: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckFieldRegistersIncClaimReductionOutput<F: Field> {
    pub field_rd_inc: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckRegistersValEvaluationOutput<F: Field> {
    pub rd_inc: F,
    pub rd_wa: F,
}

impl<F: Field> SumcheckSpartanOuterRemainderRound<F> {
    pub const fn new(q_at_zero: F, q_at_infinity: F) -> Self {
        Self {
            q_at_zero,
            q_at_infinity,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckEvaluationOutput<F: Field> {
    pub slot: BackendValueSlot,
    pub value: F,
}

impl<F: Field> SumcheckEvaluationOutput<F> {
    pub const fn new(slot: BackendValueSlot, value: F) -> Self {
        Self { slot, value }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckResult<F: Field, Proof> {
    pub proofs: Vec<SumcheckProofOutput<Proof>>,
    pub evaluations: Vec<SumcheckEvaluationOutput<F>>,
}

impl<F: Field, Proof> SumcheckResult<F, Proof> {
    pub const fn new(
        proofs: Vec<SumcheckProofOutput<Proof>>,
        evaluations: Vec<SumcheckEvaluationOutput<F>>,
    ) -> Self {
        Self {
            proofs,
            evaluations,
        }
    }
}
