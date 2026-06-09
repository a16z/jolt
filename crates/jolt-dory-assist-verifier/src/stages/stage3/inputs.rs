//! Typed inputs consumed by stage 3.

use jolt_claims::protocols::dory_assist::DoryAssistOpeningId;
use jolt_crypto::GrumpkinPoint;
use jolt_field::Fq;
use jolt_hyrax::{HyraxCommitment, HyraxOpeningProof};
use serde::{Deserialize, Serialize};

use crate::{
    proof::{DoryAssistProofClaims, DoryAssistPublicOutputs},
    stages::{stage1::Stage1Output, stage2::Stage2Output},
    verifier::CheckedInputs,
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stage3Proof {
    pub packed_eval: Fq,
    pub reduced_openings: Vec<DoryAssistOpeningId>,
}

#[derive(Clone, Copy)]
pub struct Stage3Inputs<'a, 'p> {
    pub checked: &'a CheckedInputs<'p>,
    pub proof: &'a Stage3Proof,
    pub opening_proof: &'a HyraxOpeningProof<Fq>,
    pub claims: &'a DoryAssistProofClaims,
    pub dense_commitment: &'a HyraxCommitment<GrumpkinPoint>,
    pub public_outputs: &'a DoryAssistPublicOutputs,
    pub stage1: &'a Stage1Output,
    pub stage2: &'a Stage2Output,
}
