use jolt_claims::Expr;
use jolt_field::Field;
use jolt_sumcheck::{
    CommittedOutputClaims, CommittedSumcheckConsistency, SumcheckDomainSpec, SumcheckStatement,
};

use crate::{
    BlindFoldConstruction, BlindFoldProtocol, BlindFoldStage, BlindFoldStatement,
    CommittedClaimRows, Error, FinalOpeningBinding, OpeningAlias, VerificationError,
};

#[derive(Clone, Debug)]
pub struct BlindFoldProtocolBuilder<F, O, Com, P = (), Ch = usize> {
    stages: Vec<BlindFoldStage<F, O, Com, P, Ch>>,
    final_openings: Vec<FinalOpeningBinding<F, O, Com>>,
    publics: Vec<(P, F)>,
    challenges: Vec<(Ch, F)>,
}

impl<F, O, Com, P, Ch> BlindFoldProtocolBuilder<F, O, Com, P, Ch> {
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            final_openings: Vec::new(),
            publics: Vec::new(),
            challenges: Vec::new(),
        }
    }

    pub fn public(mut self, id: P, value: F) -> Self {
        self.publics.push((id, value));
        self
    }

    pub fn challenge(mut self, id: Ch, value: F) -> Self {
        self.challenges.push((id, value));
        self
    }

    pub fn stage(self, name: impl Into<String>) -> BlindFoldStageBuilder<F, O, Com, P, Ch> {
        BlindFoldStageBuilder::new(self, name.into())
    }

    pub fn final_opening(
        mut self,
        opening_ids: Vec<O>,
        coefficients: Vec<F>,
        evaluation_commitment: Com,
    ) -> Self {
        self.final_openings.push(FinalOpeningBinding::new(
            opening_ids,
            coefficients,
            evaluation_commitment,
        ));
        self
    }
}

impl<F, O, Com, P, Ch> Default for BlindFoldProtocolBuilder<F, O, Com, P, Ch> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F, O, Com, P, Ch> BlindFoldProtocolBuilder<F, O, Com, P, Ch>
where
    F: Field + Clone,
    O: Clone + PartialEq,
    Com: Clone,
    P: Clone + PartialEq,
    Ch: Clone + PartialEq,
{
    pub fn build(self) -> Result<BlindFoldProtocol<F, Com>, VerificationError<F>> {
        self.build_construction()
            .map(|construction| construction.protocol)
    }

    /// Build the protocol while keeping the statement and baked sources — the
    /// prover-side entry point: witness assembly needs the statement to re-run
    /// the layout/constraint pass in assignment mode.
    pub fn build_construction(
        self,
    ) -> Result<BlindFoldConstruction<F, O, Com, P, Ch>, VerificationError<F>> {
        let statement = BlindFoldStatement::new(self.stages, self.final_openings);
        let protocol = BlindFoldProtocol::from_parts(&statement, &self.publics, &self.challenges)?;
        Ok(BlindFoldConstruction {
            protocol,
            statement,
            publics: self.publics,
            challenges: self.challenges,
        })
    }
}

#[derive(Clone, Debug)]
pub struct BlindFoldStageBuilder<F, O, Com, P = (), Ch = usize> {
    parent: BlindFoldProtocolBuilder<F, O, Com, P, Ch>,
    name: String,
    statement: Option<SumcheckStatement>,
    domain: Option<SumcheckDomainSpec>,
    consistency: Option<CommittedSumcheckConsistency<F, Com>>,
    output_claim_rows: Option<CommittedClaimRows<O, Com>>,
    input_claim: Option<Expr<F, O, P, Ch>>,
    output_claim: Option<Expr<F, O, P, Ch>>,
}

impl<F, O, Com, P, Ch> BlindFoldStageBuilder<F, O, Com, P, Ch> {
    fn new(parent: BlindFoldProtocolBuilder<F, O, Com, P, Ch>, name: String) -> Self {
        Self {
            parent,
            name,
            statement: None,
            domain: None,
            consistency: None,
            output_claim_rows: None,
            input_claim: None,
            output_claim: None,
        }
    }

    pub fn sumcheck(mut self, statement: SumcheckStatement) -> Self {
        self.statement = Some(statement);
        self
    }

    pub fn domain(mut self, domain: SumcheckDomainSpec) -> Self {
        self.domain = Some(domain);
        self
    }

    pub fn consistency(mut self, consistency: CommittedSumcheckConsistency<F, Com>) -> Self {
        self.consistency = Some(consistency);
        self
    }

    pub fn output_claim_rows(
        mut self,
        opening_ids: Vec<O>,
        row_len: usize,
        commitments: CommittedOutputClaims<Com>,
    ) -> Self {
        self.output_claim_rows = Some(CommittedClaimRows::new(opening_ids, row_len, commitments));
        self
    }

    pub fn output_claim_aliases(
        mut self,
        aliases: impl IntoIterator<Item = OpeningAlias<O>>,
    ) -> Self {
        let rows = self
            .output_claim_rows
            .take()
            .unwrap_or_else(CommittedClaimRows::empty)
            .with_aliases(aliases);
        self.output_claim_rows = Some(rows);
        self
    }

    pub fn input_claim(mut self, input_claim: Expr<F, O, P, Ch>) -> Self {
        self.input_claim = Some(input_claim);
        self
    }

    pub fn output_claim(mut self, output_claim: Expr<F, O, P, Ch>) -> Self {
        self.output_claim = Some(output_claim);
        self
    }

    pub fn finish_stage(mut self) -> Result<BlindFoldProtocolBuilder<F, O, Com, P, Ch>, Error> {
        let stage = BlindFoldStage::new(
            self.name.clone(),
            self.statement
                .take()
                .ok_or_else(|| self.missing("sumcheck"))?,
            self.domain
                .take()
                .unwrap_or(SumcheckDomainSpec::BooleanHypercube),
            self.consistency
                .take()
                .ok_or_else(|| self.missing("consistency"))?,
            self.output_claim_rows
                .take()
                .unwrap_or_else(CommittedClaimRows::empty),
            self.input_claim
                .take()
                .ok_or_else(|| self.missing("input claim"))?,
            self.output_claim
                .take()
                .ok_or_else(|| self.missing("output claim"))?,
        );
        self.parent.stages.push(stage);
        Ok(self.parent)
    }

    fn missing(&self, component: &'static str) -> Error {
        Error::MissingStageComponent {
            stage: self.name.clone(),
            component,
        }
    }
}
