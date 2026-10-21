use jolt_field::JoltField;
use jolt_poly::EvaluationClaim;
use jolt_transcript::{Encoding, Label, LabelWithCount};

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

impl<F, C> Encoding for ZkEvaluationClaim<'_, F, C>
where
    F: JoltField,
    C: Encoding,
{
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut out = LabelWithCount(b"zk_opening_point", self.point.len() as u64)
            .encode()
            .as_ref()
            .to_vec();
        for coordinate in self.point {
            out.extend_from_slice(coordinate.encode().as_ref());
        }
        out.extend_from_slice(Label(b"zk_eval_commitment").encode().as_ref());
        out.extend_from_slice(self.hiding_commitment.encode().as_ref());
        out
    }
}

#[derive(Clone, Debug)]
pub struct VerifierOpeningClaim<F: JoltField, C> {
    pub commitment: C,
    pub evaluation: EvaluationClaim<F>,
}

pub(crate) struct VerifierRlcClaims<'a, F: JoltField, C>(pub &'a [VerifierOpeningClaim<F, C>]);

impl<F, C> Encoding for VerifierRlcClaims<'_, F, C>
where
    F: JoltField,
{
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut out = LabelWithCount(b"rlc_claims", self.0.len() as u64)
            .encode()
            .as_ref()
            .to_vec();
        for claim in self.0 {
            out.extend_from_slice(claim.evaluation.value.encode().as_ref());
        }
        out
    }
}
