use jolt_field::Field;

/// Front-loaded batch alignment: instance `i` is inactive for rounds `r < offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundOffset {
    pub offset: usize,
}

impl RoundOffset {
    pub const fn new(offset: usize) -> Self {
        Self { offset }
    }

    pub const ZERO: Self = Self { offset: 0 };
}

/// Witness material consumed by the reference backend, keyed by instance `label`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WitnessBinding<F: Field> {
    None,
    DenseMultilinear(Vec<F>),
}

/// One batched sumcheck instance before Fiat–Shamir batching coefficients are drawn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckInstance<F: Field> {
    pub label: &'static str,
    pub input_claim: F,
    pub num_vars: usize,
    pub degree: usize,
    pub alignment: RoundOffset,
    pub bindings: WitnessBinding<F>,
}

impl<F: Field> SumcheckInstance<F> {
    pub fn new(
        label: &'static str,
        input_claim: F,
        num_vars: usize,
        degree: usize,
        alignment: RoundOffset,
    ) -> Self {
        Self {
            label,
            input_claim,
            num_vars,
            degree,
            alignment,
            bindings: WitnessBinding::None,
        }
    }

    pub fn with_dense_bindings(mut self, evals: Vec<F>) -> Self {
        self.bindings = WitnessBinding::DenseMultilinear(evals);
        self
    }

    pub fn is_active_at_round(&self, round: usize) -> bool {
        round >= self.alignment.offset
    }
}

/// Canonical batched-sumcheck input consumed by [`crate::prove_sumcheck`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchedSumcheckSpec<F: Field> {
    pub label: &'static str,
    pub instances: Vec<SumcheckInstance<F>>,
}

impl<F: Field> BatchedSumcheckSpec<F> {
    pub fn new(label: &'static str, instances: Vec<SumcheckInstance<F>>) -> Self {
        Self { label, instances }
    }

    pub fn num_rounds(&self) -> usize {
        self.instances
            .iter()
            .map(|instance| instance.alignment.offset + instance.num_vars)
            .max()
            .unwrap_or(0)
    }

    pub fn max_num_vars(&self) -> usize {
        self.num_rounds()
    }

    pub fn max_degree(&self) -> usize {
        self.instances
            .iter()
            .map(|instance| instance.degree)
            .max()
            .unwrap_or(0)
    }
}
