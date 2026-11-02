use jolt_field::JoltField;
use jolt_poly::EvaluationClaim;
use jolt_transcript::Encoding;

/// A hiding evaluation claim: a point and a commitment to the value there.
///
/// # Security
/// The [`Encoding`] absorbs the point's coordinates and then the hiding
/// commitment, with no count. Claims whose points differ in length can then
/// absorb as the same bytes, so the encoding binds a claim only when the
/// point's length is already fixed. A caller that absorbs a claim built from
/// proof data must first check `point.len()` against the length its protocol
/// fixes from bound data, or absorb that length as a `u64` word itself.
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
        let mut out = Vec::new();
        for coordinate in self.point {
            out.extend_from_slice(coordinate.encode().as_ref());
        }
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
        let mut out = Vec::new();
        for claim in self.0 {
            out.extend_from_slice(claim.evaluation.value.encode().as_ref());
        }
        out
    }
}
