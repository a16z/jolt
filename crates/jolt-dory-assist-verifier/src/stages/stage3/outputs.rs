//! Typed outputs produced by stage 3 verification.

use crate::proof::DoryAssistOpeningClaim;
use jolt_field::Fq;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3Output {
    pub packed_eval: Fq,
    pub reduced_claims: Vec<DoryAssistOpeningClaim>,
    pub expected_packed_eval: Fq,
    pub challenge: Fq,
}
