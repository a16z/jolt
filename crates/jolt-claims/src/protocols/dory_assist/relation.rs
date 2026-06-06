use serde::{Deserialize, Serialize};

use crate::util::extend_unique;
use crate::{ClaimExpression, ConsistencyClaim, Expr, InputClaimExpression, OutputClaimExpression};

use super::{
    DoryAssistChallengeId, DoryAssistOpeningId, DoryAssistPublicId, DoryAssistRelationId,
    DoryAssistSumcheckSpec,
};

pub type DoryAssistExpr<F> =
    Expr<F, DoryAssistOpeningId, DoryAssistPublicId, DoryAssistChallengeId>;
pub type DoryAssistInputClaimExpression<F> =
    InputClaimExpression<F, DoryAssistOpeningId, DoryAssistPublicId, DoryAssistChallengeId>;
pub type DoryAssistOutputClaimExpression<F> =
    OutputClaimExpression<F, DoryAssistOpeningId, DoryAssistPublicId, DoryAssistChallengeId>;
pub type DoryAssistConsistencyClaim<F> =
    ConsistencyClaim<F, DoryAssistOpeningId, DoryAssistPublicId, DoryAssistChallengeId>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistRelationClaims<F> {
    pub id: DoryAssistRelationId,
    pub sumcheck: DoryAssistSumcheckSpec,
    pub input: DoryAssistInputClaimExpression<F>,
    pub output: DoryAssistOutputClaimExpression<F>,
    pub consistency: Vec<DoryAssistConsistencyClaim<F>>,
}

impl<F> DoryAssistRelationClaims<F> {
    pub fn new(
        id: DoryAssistRelationId,
        sumcheck: DoryAssistSumcheckSpec,
        input: DoryAssistExpr<F>,
        output: DoryAssistExpr<F>,
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
        C: Into<DoryAssistConsistencyClaim<F>>,
    {
        self.consistency
            .extend(consistency.into_iter().map(Into::into));
        self
    }

    pub fn push_consistency<C>(&mut self, consistency: C)
    where
        C: Into<DoryAssistConsistencyClaim<F>>,
    {
        self.consistency.push(consistency.into());
    }

    pub fn with_input_challenges<I>(mut self, challenges: I) -> Self
    where
        I: IntoIterator<Item = DoryAssistChallengeId>,
    {
        self.input.pull_challenges_for_transcript_sync(challenges);
        self
    }

    pub fn with_auxiliary_openings<I>(mut self, openings: I) -> Self
    where
        I: IntoIterator<Item = DoryAssistOpeningId>,
    {
        for opening in openings {
            if !self.output.required_openings.contains(&opening) {
                self.output.required_openings.push(opening);
            }
        }
        self
    }

    pub fn required_openings(&self) -> Vec<DoryAssistOpeningId> {
        let mut openings = self.input.required_openings.clone();
        extend_unique(&mut openings, &self.output.required_openings);
        for consistency in &self.consistency {
            extend_unique(&mut openings, &consistency.required_openings());
        }
        openings
    }

    pub fn required_publics(&self) -> Vec<DoryAssistPublicId> {
        let mut publics = self.input.required_publics.clone();
        extend_unique(&mut publics, &self.output.required_publics);
        for consistency in &self.consistency {
            extend_unique(&mut publics, &consistency.required_publics());
        }
        publics
    }

    pub fn required_challenges(&self) -> Vec<DoryAssistChallengeId> {
        let mut challenges = self.input.required_challenges.clone();
        extend_unique(&mut challenges, &self.output.required_challenges);
        for consistency in &self.consistency {
            extend_unique(&mut challenges, &consistency.required_challenges());
        }
        challenges
    }

    pub fn challenge_index(&self, id: DoryAssistChallengeId) -> Option<usize> {
        self.required_challenges()
            .iter()
            .position(|challenge| *challenge == id)
    }

    pub fn num_challenges(&self) -> usize {
        self.required_challenges().len()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryAssistProtocolClaims<F> {
    pub relations: Vec<DoryAssistRelationClaims<F>>,
}

impl<F> DoryAssistProtocolClaims<F> {
    pub fn new(relations: Vec<DoryAssistRelationClaims<F>>) -> Self {
        debug_assert_unique_relation_ids(&relations);
        Self { relations }
    }

    pub fn push(&mut self, relation: DoryAssistRelationClaims<F>) {
        self.relations.push(relation);
    }

    pub fn iter(&self) -> std::slice::Iter<'_, DoryAssistRelationClaims<F>> {
        self.relations.iter()
    }

    pub fn relation(&self, id: DoryAssistRelationId) -> Option<&DoryAssistRelationClaims<F>> {
        self.relations.iter().find(|relation| relation.id == id)
    }

    pub fn len(&self) -> usize {
        self.relations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.relations.is_empty()
    }

    pub fn required_openings(&self) -> Vec<DoryAssistOpeningId> {
        let mut openings = Vec::new();
        for relation in &self.relations {
            extend_unique(&mut openings, &relation.required_openings());
        }
        openings
    }

    pub fn required_publics(&self) -> Vec<DoryAssistPublicId> {
        let mut publics = Vec::new();
        for relation in &self.relations {
            extend_unique(&mut publics, &relation.required_publics());
        }
        publics
    }

    pub fn required_challenges(&self) -> Vec<DoryAssistChallengeId> {
        let mut challenges = Vec::new();
        for relation in &self.relations {
            extend_unique(&mut challenges, &relation.required_challenges());
        }
        challenges
    }
}

impl<'a, F> IntoIterator for &'a DoryAssistProtocolClaims<F> {
    type Item = &'a DoryAssistRelationClaims<F>;
    type IntoIter = std::slice::Iter<'a, DoryAssistRelationClaims<F>>;

    fn into_iter(self) -> Self::IntoIter {
        self.relations.iter()
    }
}

fn debug_assert_unique_relation_ids<F>(relations: &[DoryAssistRelationClaims<F>]) {
    debug_assert!(
        relations
            .iter()
            .enumerate()
            .all(|(index, relation)| !relations[..index].iter().any(|prev| prev.id == relation.id)),
        "Dory-assist protocol claims contain duplicate relation IDs"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{challenge, opening, public, SameEvaluationAs};
    use jolt_field::{Fr, FromPrimitiveInt};

    use super::super::{
        DoryAssistCommittedPolynomial, DoryAssistVirtualPolynomial, GtChallenge, GtPolynomial,
        PackingChallenge, PackingPolynomial, WiringChallenge, WiringPolynomial,
    };

    #[test]
    fn relation_claims_capture_expression_metadata() {
        let source = DoryAssistOpeningId::virtual_polynomial(
            DoryAssistVirtualPolynomial::Wiring(WiringPolynomial::Source),
            DoryAssistRelationId::WiringGt,
        );
        let destination = DoryAssistOpeningId::virtual_polynomial(
            DoryAssistVirtualPolynomial::Wiring(WiringPolynomial::Destination),
            DoryAssistRelationId::WiringGt,
        );
        let dense = DoryAssistOpeningId::dense_witness(DoryAssistRelationId::PrefixPacking);
        let packed = DoryAssistOpeningId::virtual_polynomial(
            DoryAssistVirtualPolynomial::Packing(PackingPolynomial::DenseWitness),
            DoryAssistRelationId::PrefixPacking,
        );

        let relation: DoryAssistRelationClaims<Fr> = DoryAssistRelationClaims::new(
            DoryAssistRelationId::WiringGt,
            DoryAssistSumcheckSpec::boolean(8, 2),
            opening(source) - opening(destination)
                + challenge(DoryAssistChallengeId::from(WiringChallenge::EdgeBatch))
                    * public(DoryAssistPublicId::JoltEvaluationClaim(0)),
            opening(packed),
        )
        .with_consistency([dense.same_evaluation_as(packed)])
        .with_input_challenges([DoryAssistChallengeId::from(
            WiringChallenge::TupleCompression,
        )]);

        assert_eq!(relation.id, DoryAssistRelationId::WiringGt);
        assert_eq!(
            relation.required_openings(),
            vec![source, destination, packed, dense]
        );
        assert_eq!(
            relation.required_publics(),
            vec![DoryAssistPublicId::JoltEvaluationClaim(0)]
        );
        assert_eq!(
            relation.required_challenges(),
            vec![
                DoryAssistChallengeId::from(WiringChallenge::EdgeBatch),
                DoryAssistChallengeId::from(WiringChallenge::TupleCompression),
            ]
        );
        assert_eq!(
            relation.challenge_index(DoryAssistChallengeId::from(WiringChallenge::EdgeBatch)),
            Some(0)
        );
        assert_eq!(relation.num_challenges(), 2);
    }

    #[test]
    fn relation_claims_can_record_auxiliary_openings_for_later_wiring() {
        let dense = DoryAssistOpeningId::dense_witness(DoryAssistRelationId::PrefixPacking);
        let gt_accumulator = DoryAssistOpeningId::virtual_polynomial(
            DoryAssistVirtualPolynomial::Gt(GtPolynomial::ExpAccumulator),
            DoryAssistRelationId::GtExponentiation,
        );
        let relation: DoryAssistRelationClaims<Fr> = DoryAssistRelationClaims::new(
            DoryAssistRelationId::GtExponentiation,
            DoryAssistSumcheckSpec::boolean(4, 2),
            opening(gt_accumulator),
            opening(gt_accumulator),
        )
        .with_auxiliary_openings([dense, gt_accumulator]);

        assert_eq!(relation.required_openings(), vec![gt_accumulator, dense]);
    }

    #[test]
    fn protocol_claims_preserve_relation_order_and_deduplicate_dependencies() {
        let gt_accumulator = DoryAssistOpeningId::virtual_polynomial(
            DoryAssistVirtualPolynomial::Gt(GtPolynomial::ExpAccumulator),
            DoryAssistRelationId::GtExponentiation,
        );
        let gt_output = DoryAssistOpeningId::virtual_polynomial(
            DoryAssistVirtualPolynomial::Gt(GtPolynomial::MulOutput),
            DoryAssistRelationId::GtMultiplication,
        );

        let exp: DoryAssistRelationClaims<Fr> = DoryAssistRelationClaims::new(
            DoryAssistRelationId::GtExponentiation,
            DoryAssistSumcheckSpec::boolean(11, 8),
            opening(gt_accumulator),
            challenge(DoryAssistChallengeId::from(GtChallenge::InstanceBatch)) * opening(gt_output),
        );
        let mul: DoryAssistRelationClaims<Fr> = DoryAssistRelationClaims::new(
            DoryAssistRelationId::GtMultiplication,
            DoryAssistSumcheckSpec::boolean(4, 2),
            opening(gt_output),
            opening(gt_accumulator),
        );
        let protocol = DoryAssistProtocolClaims::new(vec![exp, mul]);

        let relation_ids = protocol
            .iter()
            .map(|relation| relation.id)
            .collect::<Vec<_>>();

        assert_eq!(
            relation_ids,
            vec![
                DoryAssistRelationId::GtExponentiation,
                DoryAssistRelationId::GtMultiplication,
            ]
        );
        assert_eq!(
            protocol
                .relation(DoryAssistRelationId::GtMultiplication)
                .map(|relation| relation.id),
            Some(DoryAssistRelationId::GtMultiplication)
        );
        assert_eq!(
            protocol.required_openings(),
            vec![gt_accumulator, gt_output]
        );
        assert_eq!(
            protocol.required_challenges(),
            vec![DoryAssistChallengeId::from(GtChallenge::InstanceBatch)]
        );
    }

    #[test]
    fn dense_witness_constructor_uses_committed_polynomial() {
        assert_eq!(
            DoryAssistOpeningId::dense_witness(DoryAssistRelationId::PrefixPacking),
            DoryAssistOpeningId::committed(
                DoryAssistCommittedPolynomial::DenseWitness,
                DoryAssistRelationId::PrefixPacking,
            )
        );
    }

    #[test]
    fn expression_evaluation_uses_protocol_ids() {
        let dense = DoryAssistOpeningId::dense_witness(DoryAssistRelationId::PrefixPacking);
        let expr: DoryAssistExpr<Fr> =
            opening(dense) + challenge(DoryAssistChallengeId::from(PackingChallenge::PrefixPoint));

        let value = expr.evaluate(
            |_| Fr::from_u64(3),
            |_| Fr::from_u64(5),
            |_| Fr::from_u64(0),
        );

        assert_eq!(value, Fr::from_u64(8));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "Dory-assist protocol claims contain duplicate relation IDs")]
    fn protocol_claims_reject_duplicate_relation_ids_in_debug() {
        let dense = DoryAssistOpeningId::dense_witness(DoryAssistRelationId::PrefixPacking);
        let relation: DoryAssistRelationClaims<Fr> = DoryAssistRelationClaims::new(
            DoryAssistRelationId::PrefixPacking,
            DoryAssistSumcheckSpec::boolean(6, 1),
            opening(dense),
            opening(dense),
        );

        let _ = DoryAssistProtocolClaims::new(vec![relation.clone(), relation]);
    }
}
