#![expect(
    clippy::panic,
    reason = "test fixtures should fail loudly when their assumed proof shape changes"
)]

use crate::{support, support::dory_pedersen};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_poly::CompressedPoly;
use jolt_sumcheck::{ClearProof, SumcheckProof};

#[test]
fn tampered_stage1_sumcheck_payload_reject() {
    let mut case = dory_pedersen::standard_case();
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) =
        &mut case.proof.stages.stage1_sumcheck_proof
    else {
        panic!("standard fixture must use a clear compressed Stage 1 proof");
    };
    proof.round_polynomials[0] = CompressedPoly::new(vec![Fr::from_u64(1)]);

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}
