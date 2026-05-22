use serde::{Deserialize, Serialize};

use crate::util::extend_unique;
use crate::{ClaimExpression, ConsistencyClaim, Expr, InputClaimExpression, OutputClaimExpression};

use super::{JoltChallengeId, JoltOpeningId, JoltPublicId, JoltRelationId, JoltSumcheckSpec};

pub type JoltExpr<F> = Expr<F, JoltOpeningId, JoltPublicId, JoltChallengeId>;
pub type JoltInputClaimExpression<F> =
    InputClaimExpression<F, JoltOpeningId, JoltPublicId, JoltChallengeId>;
pub type JoltOutputClaimExpression<F> =
    OutputClaimExpression<F, JoltOpeningId, JoltPublicId, JoltChallengeId>;
pub type JoltConsistencyClaim<F> =
    ConsistencyClaim<F, JoltOpeningId, JoltPublicId, JoltChallengeId>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltRelationClaims<F> {
    pub id: JoltRelationId,
    pub sumcheck: JoltSumcheckSpec,
    pub input: JoltInputClaimExpression<F>,
    pub output: JoltOutputClaimExpression<F>,
    pub consistency: Vec<JoltConsistencyClaim<F>>,
}

impl<F> JoltRelationClaims<F> {
    pub fn new(
        id: JoltRelationId,
        sumcheck: JoltSumcheckSpec,
        input: JoltExpr<F>,
        output: JoltExpr<F>,
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
        C: Into<JoltConsistencyClaim<F>>,
    {
        self.consistency
            .extend(consistency.into_iter().map(Into::into));
        self
    }

    pub fn push_consistency<C>(&mut self, consistency: C)
    where
        C: Into<JoltConsistencyClaim<F>>,
    {
        self.consistency.push(consistency.into());
    }

    pub fn with_input_challenges<I>(mut self, challenges: I) -> Self
    where
        I: IntoIterator<Item = JoltChallengeId>,
    {
        self.input.pull_challenges_for_transcript_sync(challenges);
        self
    }

    pub fn required_openings(&self) -> Vec<JoltOpeningId> {
        let mut openings = self.input.required_openings.clone();
        extend_unique(&mut openings, &self.output.required_openings);
        for consistency in &self.consistency {
            extend_unique(&mut openings, &consistency.required_openings());
        }
        openings
    }

    pub fn required_publics(&self) -> Vec<JoltPublicId> {
        let mut publics = self.input.required_publics.clone();
        extend_unique(&mut publics, &self.output.required_publics);
        for consistency in &self.consistency {
            extend_unique(&mut publics, &consistency.required_publics());
        }
        publics
    }

    pub fn required_challenges(&self) -> Vec<JoltChallengeId> {
        let mut challenges = self.input.required_challenges.clone();
        extend_unique(&mut challenges, &self.output.required_challenges);
        for consistency in &self.consistency {
            extend_unique(&mut challenges, &consistency.required_challenges());
        }
        challenges
    }

    pub fn challenge_index(&self, id: JoltChallengeId) -> Option<usize> {
        self.required_challenges()
            .iter()
            .position(|challenge| *challenge == id)
    }

    pub fn num_challenges(&self) -> usize {
        self.required_challenges().len()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltProtocolClaims<F> {
    pub relations: Vec<JoltRelationClaims<F>>,
}

impl<F> JoltProtocolClaims<F> {
    pub fn new(relations: Vec<JoltRelationClaims<F>>) -> Self {
        debug_assert_unique_relation_ids(&relations);
        Self { relations }
    }

    pub fn push(&mut self, relation: JoltRelationClaims<F>) {
        self.relations.push(relation);
    }

    pub fn iter(&self) -> std::slice::Iter<'_, JoltRelationClaims<F>> {
        self.relations.iter()
    }

    pub fn relation(&self, id: JoltRelationId) -> Option<&JoltRelationClaims<F>> {
        self.relations.iter().find(|relation| relation.id == id)
    }

    pub fn len(&self) -> usize {
        self.relations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.relations.is_empty()
    }

    pub fn required_openings(&self) -> Vec<JoltOpeningId> {
        let mut openings = Vec::new();
        for relation in &self.relations {
            extend_unique(&mut openings, &relation.required_openings());
        }
        openings
    }

    pub fn required_publics(&self) -> Vec<JoltPublicId> {
        let mut publics = Vec::new();
        for relation in &self.relations {
            extend_unique(&mut publics, &relation.required_publics());
        }
        publics
    }

    pub fn required_challenges(&self) -> Vec<JoltChallengeId> {
        let mut challenges = Vec::new();
        for relation in &self.relations {
            extend_unique(&mut challenges, &relation.required_challenges());
        }
        challenges
    }
}

impl<'a, F> IntoIterator for &'a JoltProtocolClaims<F> {
    type Item = &'a JoltRelationClaims<F>;
    type IntoIter = std::slice::Iter<'a, JoltRelationClaims<F>>;

    fn into_iter(self) -> Self::IntoIter {
        self.relations.iter()
    }
}

fn debug_assert_unique_relation_ids<F>(relations: &[JoltRelationClaims<F>]) {
    debug_assert!(
        relations
            .iter()
            .enumerate()
            .all(|(index, relation)| !relations[..index].iter().any(|prev| prev.id == relation.id)),
        "Jolt protocol claims contain duplicate relation IDs"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{challenge, constant, opening, public, SameEvaluationAs};
    use jolt_field::{Fr, FromPrimitiveInt};

    use super::super::{JoltCommittedPolynomial, JoltVirtualPolynomial, RamReadWriteChallenge};

    #[test]
    fn relation_claims_capture_expression_metadata() {
        let ram_inc = JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltRelationId::RamReadWriteChecking,
        );
        let ram_val = JoltOpeningId::virtual_polynomial(
            JoltVirtualPolynomial::RamVal,
            JoltRelationId::RamReadWriteChecking,
        );
        let rd_inc = JoltOpeningId::committed(
            JoltCommittedPolynomial::RdInc,
            JoltRelationId::RegistersReadWriteChecking,
        );

        let gamma = JoltChallengeId::from(RamReadWriteChallenge::Gamma);
        let sumcheck = JoltSumcheckSpec::boolean(8, 3);

        let input = opening(ram_inc) + challenge(gamma) * public(JoltPublicId::TraceLength);
        let output = opening(ram_val) + constant(Fr::from_u64(7));
        let relation = JoltRelationClaims::new(
            JoltRelationId::RamReadWriteChecking,
            sumcheck,
            input,
            output,
        )
        .with_consistency([rd_inc.same_evaluation_as(ram_val)]);

        assert_eq!(relation.id, JoltRelationId::RamReadWriteChecking);
        assert_eq!(relation.sumcheck, sumcheck);
        assert_eq!(relation.required_openings(), vec![ram_inc, ram_val, rd_inc]);
        assert_eq!(relation.required_publics(), vec![JoltPublicId::TraceLength]);
        assert_eq!(relation.required_challenges(), vec![gamma]);
        assert_eq!(relation.challenge_index(gamma), Some(0));
        assert_eq!(relation.num_challenges(), 1);
        assert_eq!(
            relation.consistency,
            vec![JoltConsistencyClaim::same_evaluation(rd_inc, ram_val)]
        );
    }

    #[test]
    fn relation_claims_carry_sumcheck_metadata() {
        let ram_inc = JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltRelationId::RamReadWriteChecking,
        );
        let sumcheck = JoltSumcheckSpec::boolean(8, 3);

        let relation: JoltRelationClaims<Fr> = JoltRelationClaims::new(
            JoltRelationId::RamReadWriteChecking,
            sumcheck,
            opening(ram_inc),
            opening(ram_inc),
        );

        assert_eq!(relation.sumcheck, sumcheck);
    }

    #[test]
    fn protocol_claims_preserve_relation_order_and_deduplicate_dependencies() {
        let rd_inc = JoltOpeningId::committed(
            JoltCommittedPolynomial::RdInc,
            JoltRelationId::RegistersReadWriteChecking,
        );
        let ram_inc = JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltRelationId::RamReadWriteChecking,
        );

        let registers: JoltRelationClaims<Fr> = JoltRelationClaims::new(
            JoltRelationId::RegistersReadWriteChecking,
            JoltSumcheckSpec::boolean(12, 3),
            opening(rd_inc) + public(JoltPublicId::PaddedTraceLength),
            opening(rd_inc),
        );
        let ram: JoltRelationClaims<Fr> = JoltRelationClaims::new(
            JoltRelationId::RamReadWriteChecking,
            JoltSumcheckSpec::boolean(20, 3),
            opening(ram_inc) + public(JoltPublicId::PaddedTraceLength),
            opening(ram_inc),
        );
        let protocol = JoltProtocolClaims::new(vec![registers, ram]);

        let relation_ids = protocol
            .iter()
            .map(|relation| relation.id)
            .collect::<Vec<_>>();

        assert_eq!(
            relation_ids,
            vec![
                JoltRelationId::RegistersReadWriteChecking,
                JoltRelationId::RamReadWriteChecking,
            ]
        );
        assert_eq!(
            protocol
                .relation(JoltRelationId::RamReadWriteChecking)
                .map(|relation| relation.id),
            Some(JoltRelationId::RamReadWriteChecking)
        );
        assert_eq!(protocol.required_openings(), vec![rd_inc, ram_inc]);
        assert_eq!(
            protocol.required_publics(),
            vec![JoltPublicId::PaddedTraceLength]
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "Jolt protocol claims contain duplicate relation IDs")]
    fn protocol_claims_reject_duplicate_relation_ids_in_debug() {
        let rd_inc = JoltOpeningId::committed(
            JoltCommittedPolynomial::RdInc,
            JoltRelationId::RegistersReadWriteChecking,
        );
        let relation: JoltRelationClaims<Fr> = JoltRelationClaims::new(
            JoltRelationId::RegistersReadWriteChecking,
            JoltSumcheckSpec::boolean(12, 3),
            opening(rd_inc),
            opening(rd_inc),
        );

        let _ = JoltProtocolClaims::new(vec![relation.clone(), relation]);
    }
}
