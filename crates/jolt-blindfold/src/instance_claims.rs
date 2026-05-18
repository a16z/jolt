use jolt_claims::Expr;
use jolt_sumcheck::SumcheckShape;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstanceClaims<F, O, P = (), Ch = usize> {
    pub stages: Vec<StageClaims<F, O, P, Ch>>,
}

impl<F, O, P, Ch> InstanceClaims<F, O, P, Ch> {
    pub fn new(stages: Vec<StageClaims<F, O, P, Ch>>) -> Self {
        Self { stages }
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageClaims<F, O, P = (), Ch = usize> {
    pub name: String,
    pub shape: SumcheckShape,
    pub input_claim: Expr<F, O, P, Ch>,
    pub output_claim: Expr<F, O, P, Ch>,
}

impl<F, O, P, Ch> StageClaims<F, O, P, Ch> {
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
    fn instance_claims_group_stages() {
        let claim: Expr<Fr, Opening> = opening(Opening::A);
        let stage = StageClaims::new("stage", SumcheckShape::new(2, 2), claim.clone(), claim);
        let claims = InstanceClaims::new(vec![stage]);

        assert_eq!(claims.stages.len(), 1);
        assert_eq!(claims.stage_count(), 1);
        assert_eq!(claims.stages[0].name, "stage");
        assert_eq!(claims.stages[0].shape, SumcheckShape::new(2, 2));
    }
}
