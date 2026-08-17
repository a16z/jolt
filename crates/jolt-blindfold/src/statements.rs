use jolt_claims::Expr;
use jolt_sumcheck::{
    CommittedOutputClaims, CommittedSumcheckConsistency, SumcheckDomainSpec, SumcheckStatement,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldStatement<F, O, Com, P = (), Ch = usize> {
    pub stages: Vec<BlindFoldStage<F, O, Com, P, Ch>>,
    pub final_openings: Vec<FinalOpeningBinding<F, O, Com>>,
}

impl<F, O, Com, P, Ch> BlindFoldStatement<F, O, Com, P, Ch> {
    pub fn new(
        stages: Vec<BlindFoldStage<F, O, Com, P, Ch>>,
        final_openings: Vec<FinalOpeningBinding<F, O, Com>>,
    ) -> Self {
        Self {
            stages,
            final_openings,
        }
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

impl<F, O, Com: Clone, P, Ch> BlindFoldStatement<F, O, Com, P, Ch> {
    pub fn final_opening_commitments(&self) -> Vec<Com> {
        self.final_openings
            .iter()
            .map(|binding| binding.evaluation_commitment.clone())
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldStage<F, O, Com, P = (), Ch = usize> {
    pub name: String,
    pub statement: SumcheckStatement,
    pub domain: SumcheckDomainSpec,
    pub consistency: CommittedSumcheckConsistency<F, Com>,
    pub output_claim_rows: CommittedClaimRows<O, Com>,
    pub input_claim: Expr<F, O, P, Ch>,
    pub output_claim: Expr<F, O, P, Ch>,
}

impl<F, O, Com, P, Ch> BlindFoldStage<F, O, Com, P, Ch> {
    pub fn new(
        name: impl Into<String>,
        statement: SumcheckStatement,
        domain: SumcheckDomainSpec,
        consistency: CommittedSumcheckConsistency<F, Com>,
        output_claim_rows: CommittedClaimRows<O, Com>,
        input_claim: Expr<F, O, P, Ch>,
        output_claim: Expr<F, O, P, Ch>,
    ) -> Self {
        Self {
            name: name.into(),
            statement,
            domain,
            consistency,
            output_claim_rows,
            input_claim,
            output_claim,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedClaimRows<O, Com> {
    pub opening_ids: Vec<O>,
    pub opening_aliases: Vec<OpeningAlias<O>>,
    pub opening_equalities: Vec<OpeningEquality<O>>,
    pub row_len: usize,
    pub commitments: CommittedOutputClaims<Com>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningAlias<O> {
    pub alias: O,
    pub source: O,
}

impl<O> OpeningAlias<O> {
    pub fn new(alias: O, source: O) -> Self {
        Self { alias, source }
    }
}

/// An equality constraint between two openings that BOTH have their own
/// committed rows (unlike [`OpeningAlias`], where the alias id has no row and
/// resolves to the source's variable). Lowered as an R1CS equality between the
/// two hidden row values, so a statement can bind duplicate semantic openings
/// that are separately committed — e.g. a claim-reduction output row and the
/// appendage row it must equal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningEquality<O> {
    pub left: O,
    pub right: O,
}

impl<O> OpeningEquality<O> {
    pub fn new(left: O, right: O) -> Self {
        Self { left, right }
    }
}

impl<O, Com> CommittedClaimRows<O, Com> {
    pub fn new(
        opening_ids: Vec<O>,
        row_len: usize,
        commitments: CommittedOutputClaims<Com>,
    ) -> Self {
        Self {
            opening_ids,
            opening_aliases: Vec::new(),
            opening_equalities: Vec::new(),
            row_len,
            commitments,
        }
    }

    pub fn with_aliases(mut self, aliases: impl IntoIterator<Item = OpeningAlias<O>>) -> Self {
        self.opening_aliases.extend(aliases);
        self
    }

    pub fn with_equalities(
        mut self,
        equalities: impl IntoIterator<Item = OpeningEquality<O>>,
    ) -> Self {
        self.opening_equalities.extend(equalities);
        self
    }

    pub fn empty() -> Self {
        Self {
            opening_ids: Vec::new(),
            opening_aliases: Vec::new(),
            opening_equalities: Vec::new(),
            row_len: 0,
            commitments: CommittedOutputClaims {
                commitments: Vec::new(),
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinalOpeningBinding<F, O, Com> {
    pub opening_ids: Vec<O>,
    pub coefficients: Vec<F>,
    pub evaluation_commitment: Com,
}

impl<F, O, Com> FinalOpeningBinding<F, O, Com> {
    pub fn new(opening_ids: Vec<O>, coefficients: Vec<F>, evaluation_commitment: Com) -> Self {
        Self {
            opening_ids,
            coefficients,
            evaluation_commitment,
        }
    }
}

#[cfg(test)]
#[expect(clippy::indexing_slicing, reason = "tests index fixture data")]
mod tests {
    use super::*;
    use jolt_claims::{opening, Expr};
    use jolt_field::Fr;
    use jolt_sumcheck::{CommittedSumcheckConsistency, SumcheckDomainSpec};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Opening {
        A,
    }

    #[test]
    fn blindfold_statement_groups_stages() {
        let claim: Expr<Fr, Opening> = opening(Opening::A);
        let stage = BlindFoldStage::new(
            "stage",
            SumcheckStatement::new(2, 2),
            SumcheckDomainSpec::BooleanHypercube,
            CommittedSumcheckConsistency::<Fr, ()> { rounds: Vec::new() },
            CommittedClaimRows::empty(),
            claim.clone(),
            claim,
        );
        let statement = BlindFoldStatement::new(vec![stage], Vec::new());

        assert_eq!(statement.stages.len(), 1);
        assert_eq!(statement.stage_count(), 1);
        assert_eq!(statement.stages[0].name, "stage");
        assert_eq!(statement.stages[0].statement, SumcheckStatement::new(2, 2));
    }
}
