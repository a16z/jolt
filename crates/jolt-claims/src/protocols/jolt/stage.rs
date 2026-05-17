use serde::{Deserialize, Serialize};

use crate::{ClaimExpression, Expr, InputClaimExpression, OutputClaimExpression};

use super::{JoltChallengeId, JoltOpeningId, JoltPublicId, JoltStageId};

pub type JoltExpr<F> = Expr<F, JoltOpeningId, JoltPublicId, JoltChallengeId>;
pub type JoltInputClaimExpression<F> =
    InputClaimExpression<F, JoltOpeningId, JoltPublicId, JoltChallengeId>;
pub type JoltOutputClaimExpression<F> =
    OutputClaimExpression<F, JoltOpeningId, JoltPublicId, JoltChallengeId>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltStageClaims<F> {
    pub id: JoltStageId,
    pub input: JoltInputClaimExpression<F>,
    pub output: JoltOutputClaimExpression<F>,
}

impl<F> JoltStageClaims<F> {
    pub fn new(id: JoltStageId, input: JoltExpr<F>, output: JoltExpr<F>) -> Self {
        Self {
            id,
            input: ClaimExpression::from(input),
            output: ClaimExpression::from(output),
        }
    }

    pub fn required_openings(&self) -> Vec<JoltOpeningId> {
        let mut openings = self.input.required_openings.clone();
        extend_unique(&mut openings, &self.output.required_openings);
        openings
    }

    pub fn required_publics(&self) -> Vec<JoltPublicId> {
        let mut publics = self.input.required_publics.clone();
        extend_unique(&mut publics, &self.output.required_publics);
        publics
    }

    pub fn required_challenges(&self) -> Vec<JoltChallengeId> {
        let mut challenges = self.input.required_challenges.clone();
        extend_unique(&mut challenges, &self.output.required_challenges);
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
    pub stages: Vec<JoltStageClaims<F>>,
}

impl<F> JoltProtocolClaims<F> {
    pub fn new(stages: Vec<JoltStageClaims<F>>) -> Self {
        Self { stages }
    }

    pub fn push(&mut self, stage: JoltStageClaims<F>) {
        self.stages.push(stage);
    }

    pub fn iter(&self) -> std::slice::Iter<'_, JoltStageClaims<F>> {
        self.stages.iter()
    }

    pub fn stage(&self, id: JoltStageId) -> Option<&JoltStageClaims<F>> {
        self.stages.iter().find(|stage| stage.id == id)
    }

    pub fn len(&self) -> usize {
        self.stages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    pub fn required_openings(&self) -> Vec<JoltOpeningId> {
        let mut openings = Vec::new();
        for stage in &self.stages {
            extend_unique(&mut openings, &stage.required_openings());
        }
        openings
    }

    pub fn required_publics(&self) -> Vec<JoltPublicId> {
        let mut publics = Vec::new();
        for stage in &self.stages {
            extend_unique(&mut publics, &stage.required_publics());
        }
        publics
    }

    pub fn required_challenges(&self) -> Vec<JoltChallengeId> {
        let mut challenges = Vec::new();
        for stage in &self.stages {
            extend_unique(&mut challenges, &stage.required_challenges());
        }
        challenges
    }
}

impl<'a, F> IntoIterator for &'a JoltProtocolClaims<F> {
    type Item = &'a JoltStageClaims<F>;
    type IntoIter = std::slice::Iter<'a, JoltStageClaims<F>>;

    fn into_iter(self) -> Self::IntoIter {
        self.stages.iter()
    }
}

fn extend_unique<T: Copy + Eq>(target: &mut Vec<T>, values: &[T]) {
    for &value in values {
        if !target.contains(&value) {
            target.push(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{challenge, constant, opening, public};
    use jolt_field::{Fr, FromPrimitiveInt};

    use super::super::{JoltCommittedPolynomial, JoltVirtualPolynomial, RamReadWriteChallenge};

    #[test]
    fn stage_claims_capture_expression_metadata() {
        let ram_inc = JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltStageId::RamReadWriteChecking,
        );
        let ram_val = JoltOpeningId::virtual_polynomial(
            JoltVirtualPolynomial::RamVal,
            JoltStageId::RamReadWriteChecking,
        );

        let gamma = JoltChallengeId::from(RamReadWriteChallenge::Gamma);

        let input = opening(ram_inc) + challenge(gamma) * public(JoltPublicId::TraceLength);
        let output = opening(ram_val) + constant(Fr::from_u64(7));
        let stage = JoltStageClaims::new(JoltStageId::RamReadWriteChecking, input, output);

        assert_eq!(stage.id, JoltStageId::RamReadWriteChecking);
        assert_eq!(stage.required_openings(), vec![ram_inc, ram_val]);
        assert_eq!(stage.required_publics(), vec![JoltPublicId::TraceLength]);
        assert_eq!(stage.required_challenges(), vec![gamma]);
        assert_eq!(stage.challenge_index(gamma), Some(0));
        assert_eq!(stage.num_challenges(), 1);
    }

    #[test]
    fn protocol_claims_preserve_stage_order_and_deduplicate_dependencies() {
        let rd_inc = JoltOpeningId::committed(
            JoltCommittedPolynomial::RdInc,
            JoltStageId::RegistersReadWriteChecking,
        );
        let ram_inc = JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltStageId::RamReadWriteChecking,
        );

        let registers: JoltStageClaims<Fr> = JoltStageClaims::new(
            JoltStageId::RegistersReadWriteChecking,
            opening(rd_inc) + public(JoltPublicId::PaddedTraceLength),
            opening(rd_inc),
        );
        let ram: JoltStageClaims<Fr> = JoltStageClaims::new(
            JoltStageId::RamReadWriteChecking,
            opening(ram_inc) + public(JoltPublicId::PaddedTraceLength),
            opening(ram_inc),
        );
        let protocol = JoltProtocolClaims::new(vec![registers, ram]);

        let stage_ids = protocol.iter().map(|stage| stage.id).collect::<Vec<_>>();

        assert_eq!(
            stage_ids,
            vec![
                JoltStageId::RegistersReadWriteChecking,
                JoltStageId::RamReadWriteChecking,
            ]
        );
        assert_eq!(
            protocol
                .stage(JoltStageId::RamReadWriteChecking)
                .map(|stage| stage.id),
            Some(JoltStageId::RamReadWriteChecking)
        );
        assert_eq!(protocol.required_openings(), vec![rd_inc, ram_inc]);
        assert_eq!(
            protocol.required_publics(),
            vec![JoltPublicId::PaddedTraceLength]
        );
    }
}
