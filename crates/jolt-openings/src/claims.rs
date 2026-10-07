use jolt_field::{CanonicalBytes, JoltField};
use jolt_poly::EvaluationClaim;
use jolt_transcript::Channel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZkEvaluationClaim<'a, F, C> {
    pub point: &'a [F],
    pub hiding_commitment: &'a C,
}

impl<'a, F, C> ZkEvaluationClaim<'a, F, C> {
    pub fn new(point: &'a [F], hiding_commitment: &'a C) -> Self {
        Self {
            point,
            hiding_commitment,
        }
    }
}

impl<F: JoltField, C: CanonicalBytes> ZkEvaluationClaim<'_, F, C> {
    /// Absorbs the opening point, then the hiding commitment to the evaluation.
    pub fn absorb<Ch: Channel>(&self, channel: &mut Ch) {
        channel.public_all(self.point);
        channel.public(self.hiding_commitment);
    }
}

#[derive(Clone, Debug)]
pub struct VerifierOpeningClaim<F: JoltField, C> {
    pub commitment: C,
    pub evaluation: EvaluationClaim<F>,
}

/// Absorbs an opening claim's point, then its value.
pub(crate) fn absorb_evaluation<F: JoltField, Ch: Channel>(
    channel: &mut Ch,
    point: &[F],
    value: &F,
) {
    channel.public_all(point);
    channel.public(value);
}
