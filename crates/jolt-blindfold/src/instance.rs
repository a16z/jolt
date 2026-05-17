use jolt_claims::Expr;
use jolt_crypto::Commitment;
use jolt_sumcheck::{CommittedSumcheckCheck, SumcheckShape};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instance<F, O, P = (), Ch = usize> {
    pub stages: Vec<Stage<F, O, P, Ch>>,
}

impl<F, O, P, Ch> Instance<F, O, P, Ch> {
    pub fn new(stages: Vec<Stage<F, O, P, Ch>>) -> Self {
        Self { stages }
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage<F, O, P = (), Ch = usize> {
    pub name: String,
    pub shape: SumcheckShape,
    pub input_claim: Expr<F, O, P, Ch>,
    pub output_claim: Expr<F, O, P, Ch>,
}

impl<F, O, P, Ch> Stage<F, O, P, Ch> {
    pub fn new(
        name: impl Into<String>,
        shape: SumcheckShape,
        input_claim: Expr<F, O, P, Ch>,
        output_claim: Expr<F, O, P, Ch>,
    ) -> Self {
        Self {
            name: name.into(),
            shape,
            input_claim,
            output_claim,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Inputs<F, C> {
    pub stages: Vec<StageInput<F, C>>,
}

impl<F, C> Inputs<F, C> {
    pub fn new(stages: Vec<StageInput<F, C>>) -> Self {
        Self { stages }
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageInput<F, C> {
    pub check: CommittedSumcheckCheck<F, C>,
}

impl<F, C> StageInput<F, C> {
    pub fn new(check: CommittedSumcheckCheck<F, C>) -> Self {
        Self { check }
    }
}

pub type VectorStageInput<F, VC> = StageInput<F, <VC as Commitment>::Output>;
pub type VectorInputs<F, VC> = Inputs<F, <VC as Commitment>::Output>;

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_claims::{opening, Expr};
    use jolt_field::Fr;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Opening {
        A,
    }

    #[test]
    fn instance_groups_stages() {
        let claim: Expr<Fr, Opening> = opening(Opening::A);
        let stage = Stage::new("stage", SumcheckShape::new(2, 2), claim.clone(), claim);
        let instance = Instance::new(vec![stage]);

        assert_eq!(instance.stages.len(), 1);
        assert_eq!(instance.stage_count(), 1);
        assert_eq!(instance.stages[0].name, "stage");
        assert_eq!(instance.stages[0].shape, SumcheckShape::new(2, 2));
    }
}
