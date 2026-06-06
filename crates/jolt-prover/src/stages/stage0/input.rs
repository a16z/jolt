use jolt_claims::protocols::jolt::formulas::{
    dimensions::TracePolynomialOrder, ra::JoltRaPolynomialLayout,
};
use jolt_openings::CommitmentScheme;
use jolt_verifier::config::JoltProtocolConfig;
#[cfg(feature = "field-inline")]
use jolt_witness::{
    protocols::jolt_vm::field_inline::FieldInlineNamespace, CommittedWitnessProvider,
};

#[cfg(feature = "field-inline")]
pub type FieldInlineCommitmentWitness<'a, F> =
    dyn CommittedWitnessProvider<F, FieldInlineNamespace> + Sync + 'a;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentStageConfig {
    pub ra_layout: JoltRaPolynomialLayout,
    pub include_trusted_advice: bool,
    pub include_untrusted_advice: bool,
    pub trace_polynomial_order: TracePolynomialOrder,
    pub final_opening_trace_rows: Option<usize>,
    pub final_opening_address_columns: Option<usize>,
}

impl CommitmentStageConfig {
    pub const fn new(
        ra_layout: JoltRaPolynomialLayout,
        include_trusted_advice: bool,
        include_untrusted_advice: bool,
    ) -> Self {
        Self {
            ra_layout,
            include_trusted_advice,
            include_untrusted_advice,
            trace_polynomial_order: TracePolynomialOrder::CycleMajor,
            final_opening_trace_rows: None,
            final_opening_address_columns: None,
        }
    }

    pub const fn with_trace_polynomial_order(
        mut self,
        trace_polynomial_order: TracePolynomialOrder,
    ) -> Self {
        self.trace_polynomial_order = trace_polynomial_order;
        self
    }

    pub fn with_final_opening_trace_embedding(
        mut self,
        log_t: usize,
        committed_chunk_bits: usize,
        trace_polynomial_order: TracePolynomialOrder,
    ) -> Self {
        self.trace_polynomial_order = trace_polynomial_order;
        self.final_opening_trace_rows = Some(1usize << log_t);
        self.final_opening_address_columns = Some(1usize << committed_chunk_bits);
        self
    }
}

#[derive(Clone, Copy)]
pub struct CommitmentStageInput<'a, W, PCS: CommitmentScheme> {
    pub witness: &'a W,
    pub setup: &'a PCS::ProverSetup,
    pub config: CommitmentStageConfig,
    pub protocol: JoltProtocolConfig,
    #[cfg(feature = "field-inline")]
    pub field_inline_witness: &'a FieldInlineCommitmentWitness<'a, PCS::Field>,
}

impl<'a, W, PCS: CommitmentScheme> CommitmentStageInput<'a, W, PCS> {
    pub fn new(
        witness: &'a W,
        setup: &'a PCS::ProverSetup,
        config: CommitmentStageConfig,
        protocol: JoltProtocolConfig,
        #[cfg(feature = "field-inline")] field_inline_witness: &'a FieldInlineCommitmentWitness<
            'a,
            PCS::Field,
        >,
    ) -> Self {
        Self {
            witness,
            setup,
            config,
            protocol,
            #[cfg(feature = "field-inline")]
            field_inline_witness,
        }
    }
}
