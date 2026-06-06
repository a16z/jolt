use jolt_claims::protocols::jolt::formulas::dimensions::TracePolynomialOrder;
use jolt_witness::{ViewRequirement, WitnessNamespace};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CommitmentSlot(pub u32);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CommitmentMode {
    #[default]
    Transparent,
    Zk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TracePolynomialEmbedding {
    pub trace_rows: usize,
    pub address_columns: usize,
    pub trace_polynomial_order: TracePolynomialOrder,
}

impl TracePolynomialEmbedding {
    pub const fn new(
        trace_rows: usize,
        address_columns: usize,
        trace_polynomial_order: TracePolynomialOrder,
    ) -> Self {
        Self {
            trace_rows,
            address_columns,
            trace_polynomial_order,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentRequestItem<N: WitnessNamespace> {
    pub slot: CommitmentSlot,
    pub requirement: ViewRequirement<N>,
    pub mode: CommitmentMode,
    pub trace_polynomial_order: TracePolynomialOrder,
    pub trace_embedding: Option<TracePolynomialEmbedding>,
}

impl<N: WitnessNamespace> CommitmentRequestItem<N> {
    pub const fn new(slot: CommitmentSlot, requirement: ViewRequirement<N>) -> Self {
        Self {
            slot,
            requirement,
            mode: CommitmentMode::Transparent,
            trace_polynomial_order: TracePolynomialOrder::CycleMajor,
            trace_embedding: None,
        }
    }

    pub const fn with_mode(
        slot: CommitmentSlot,
        requirement: ViewRequirement<N>,
        mode: CommitmentMode,
    ) -> Self {
        Self {
            slot,
            requirement,
            mode,
            trace_polynomial_order: TracePolynomialOrder::CycleMajor,
            trace_embedding: None,
        }
    }

    pub const fn with_trace_polynomial_order(
        mut self,
        trace_polynomial_order: TracePolynomialOrder,
    ) -> Self {
        self.trace_polynomial_order = trace_polynomial_order;
        self
    }

    pub const fn with_trace_embedding(
        mut self,
        trace_embedding: Option<TracePolynomialEmbedding>,
    ) -> Self {
        self.trace_embedding = trace_embedding;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentRequest<N: WitnessNamespace> {
    pub items: Vec<CommitmentRequestItem<N>>,
}

impl<N: WitnessNamespace> CommitmentRequest<N> {
    pub const fn new(items: Vec<CommitmentRequestItem<N>>) -> Self {
        Self { items }
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
