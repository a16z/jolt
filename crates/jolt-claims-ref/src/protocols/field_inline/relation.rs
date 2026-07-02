use serde::{Deserialize, Serialize};

use crate::util::extend_unique;
use crate::{ClaimExpression, ConsistencyClaim, Expr, InputClaimExpression, OutputClaimExpression};

use super::{
    FieldInlineChallengeId, FieldInlineOpeningId, FieldInlinePublicId, FieldInlineRelationId,
    FieldInlineSumcheckSpec,
};

pub type FieldInlineExpr<F> =
    Expr<F, FieldInlineOpeningId, FieldInlinePublicId, FieldInlineChallengeId>;
pub type FieldInlineInputClaimExpression<F> =
    InputClaimExpression<F, FieldInlineOpeningId, FieldInlinePublicId, FieldInlineChallengeId>;
pub type FieldInlineOutputClaimExpression<F> =
    OutputClaimExpression<F, FieldInlineOpeningId, FieldInlinePublicId, FieldInlineChallengeId>;
pub type FieldInlineConsistencyClaim<F> =
    ConsistencyClaim<F, FieldInlineOpeningId, FieldInlinePublicId, FieldInlineChallengeId>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldInlineRelationClaims<F> {
    pub id: FieldInlineRelationId,
    pub sumcheck: FieldInlineSumcheckSpec,
    pub input: FieldInlineInputClaimExpression<F>,
    pub output: FieldInlineOutputClaimExpression<F>,
    pub consistency: Vec<FieldInlineConsistencyClaim<F>>,
}

impl<F> FieldInlineRelationClaims<F> {
    pub fn new(
        id: FieldInlineRelationId,
        sumcheck: FieldInlineSumcheckSpec,
        input: FieldInlineExpr<F>,
        output: FieldInlineExpr<F>,
    ) -> Self {
        Self {
            id,
            sumcheck,
            input: ClaimExpression::from(input),
            output: ClaimExpression::from(output),
            consistency: Vec::new(),
        }
    }

    pub fn with_consistency<I, C>(mut self, consistency: I) -> Self
    where
        I: IntoIterator<Item = C>,
        C: Into<FieldInlineConsistencyClaim<F>>,
    {
        self.consistency
            .extend(consistency.into_iter().map(Into::into));
        self
    }

    pub fn push_consistency<C>(&mut self, consistency: C)
    where
        C: Into<FieldInlineConsistencyClaim<F>>,
    {
        self.consistency.push(consistency.into());
    }

    pub fn with_input_challenges<I>(mut self, challenges: I) -> Self
    where
        I: IntoIterator<Item = FieldInlineChallengeId>,
    {
        self.input.pull_challenges_for_transcript_sync(challenges);
        self
    }

    pub fn required_openings(&self) -> Vec<FieldInlineOpeningId> {
        let mut openings = self.input.required_openings.clone();
        extend_unique(&mut openings, &self.output.required_openings);
        for consistency in &self.consistency {
            extend_unique(&mut openings, &consistency.required_openings());
        }
        openings
    }

    pub fn required_publics(&self) -> Vec<FieldInlinePublicId> {
        let mut publics = self.input.required_publics.clone();
        extend_unique(&mut publics, &self.output.required_publics);
        for consistency in &self.consistency {
            extend_unique(&mut publics, &consistency.required_publics());
        }
        publics
    }

    pub fn required_challenges(&self) -> Vec<FieldInlineChallengeId> {
        let mut challenges = self.input.required_challenges.clone();
        extend_unique(&mut challenges, &self.output.required_challenges);
        for consistency in &self.consistency {
            extend_unique(&mut challenges, &consistency.required_challenges());
        }
        challenges
    }

    pub fn challenge_index(&self, id: FieldInlineChallengeId) -> Option<usize> {
        self.required_challenges()
            .iter()
            .position(|challenge| *challenge == id)
    }

    pub fn num_challenges(&self) -> usize {
        self.required_challenges().len()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldInlineProtocolClaims<F> {
    pub relations: Vec<FieldInlineRelationClaims<F>>,
}

impl<F> FieldInlineProtocolClaims<F> {
    pub fn new(relations: Vec<FieldInlineRelationClaims<F>>) -> Self {
        debug_assert_unique_relation_ids(&relations);
        Self { relations }
    }

    pub fn push(&mut self, relation: FieldInlineRelationClaims<F>) {
        self.relations.push(relation);
    }

    pub fn iter(&self) -> std::slice::Iter<'_, FieldInlineRelationClaims<F>> {
        self.relations.iter()
    }

    pub fn relation(&self, id: FieldInlineRelationId) -> Option<&FieldInlineRelationClaims<F>> {
        self.relations.iter().find(|relation| relation.id == id)
    }

    pub fn len(&self) -> usize {
        self.relations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.relations.is_empty()
    }

    pub fn required_openings(&self) -> Vec<FieldInlineOpeningId> {
        let mut openings = Vec::new();
        for relation in &self.relations {
            extend_unique(&mut openings, &relation.required_openings());
        }
        openings
    }

    pub fn required_publics(&self) -> Vec<FieldInlinePublicId> {
        let mut publics = Vec::new();
        for relation in &self.relations {
            extend_unique(&mut publics, &relation.required_publics());
        }
        publics
    }

    pub fn required_challenges(&self) -> Vec<FieldInlineChallengeId> {
        let mut challenges = Vec::new();
        for relation in &self.relations {
            extend_unique(&mut challenges, &relation.required_challenges());
        }
        challenges
    }
}

impl<'a, F> IntoIterator for &'a FieldInlineProtocolClaims<F> {
    type Item = &'a FieldInlineRelationClaims<F>;
    type IntoIter = std::slice::Iter<'a, FieldInlineRelationClaims<F>>;

    fn into_iter(self) -> Self::IntoIter {
        self.relations.iter()
    }
}

fn debug_assert_unique_relation_ids<F>(relations: &[FieldInlineRelationClaims<F>]) {
    debug_assert!(
        relations
            .iter()
            .enumerate()
            .all(|(index, relation)| !relations[..index].iter().any(|prev| prev.id == relation.id)),
        "field-inline protocol claims contain duplicate relation IDs"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opening;
    use jolt_field::Fr;

    use super::super::{FieldInlineCommittedPolynomial, FieldInlineVirtualPolynomial};

    #[test]
    fn protocol_claims_preserve_relation_order_and_deduplicate_dependencies() {
        let rd_inc = FieldInlineOpeningId::committed(
            FieldInlineCommittedPolynomial::FieldRdInc,
            FieldInlineRelationId::FieldRegistersReadWriteChecking,
        );
        let registers_val = FieldInlineOpeningId::virtual_polynomial(
            FieldInlineVirtualPolynomial::FieldRegistersVal,
            FieldInlineRelationId::FieldRegistersValEvaluation,
        );

        let read_write: FieldInlineRelationClaims<Fr> = FieldInlineRelationClaims::new(
            FieldInlineRelationId::FieldRegistersReadWriteChecking,
            FieldInlineSumcheckSpec::boolean(12, 3),
            opening(rd_inc),
            opening(registers_val),
        );
        let val_evaluation: FieldInlineRelationClaims<Fr> = FieldInlineRelationClaims::new(
            FieldInlineRelationId::FieldRegistersValEvaluation,
            FieldInlineSumcheckSpec::boolean(8, 3),
            opening(registers_val),
            opening(rd_inc),
        );
        let protocol = FieldInlineProtocolClaims::new(vec![read_write, val_evaluation]);

        let relation_ids = protocol
            .iter()
            .map(|relation| relation.id)
            .collect::<Vec<_>>();

        assert_eq!(
            relation_ids,
            vec![
                FieldInlineRelationId::FieldRegistersReadWriteChecking,
                FieldInlineRelationId::FieldRegistersValEvaluation,
            ]
        );
        assert_eq!(
            protocol
                .relation(FieldInlineRelationId::FieldRegistersValEvaluation)
                .map(|relation| relation.id),
            Some(FieldInlineRelationId::FieldRegistersValEvaluation)
        );
        assert_eq!(protocol.required_openings(), vec![rd_inc, registers_val]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "field-inline protocol claims contain duplicate relation IDs")]
    fn protocol_claims_reject_duplicate_relation_ids_in_debug() {
        let rd_inc = FieldInlineOpeningId::committed(
            FieldInlineCommittedPolynomial::FieldRdInc,
            FieldInlineRelationId::FieldRegistersReadWriteChecking,
        );
        let relation: FieldInlineRelationClaims<Fr> = FieldInlineRelationClaims::new(
            FieldInlineRelationId::FieldRegistersReadWriteChecking,
            FieldInlineSumcheckSpec::boolean(12, 3),
            opening(rd_inc),
            opening(rd_inc),
        );

        let _ = FieldInlineProtocolClaims::new(vec![relation.clone(), relation]);
    }
}
