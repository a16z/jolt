#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "attack constructors require a fixture with the exact audited proof shape"
)]

#[cfg(not(feature = "akita"))]
use jolt_crypto::HomomorphicCommitment;
use jolt_crypto::VectorCommitment;
use jolt_field::{Field, Ring};
use jolt_openings::CommitmentScheme;
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_r1cs::constraints::jolt::SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE;
use jolt_sumcheck::{CenteredIntegerDomain, ClearProof, SumcheckProof};
use jolt_verifier::{fs_audit::FsScope, JoltProof, JoltProofClaims};
use num_traits::One;

use super::fs_transcript::{ChallengeKind, ChallengeTape};

/// Changes two Dory commitments in the kernel of the frozen final-opening RLC.
#[cfg(not(feature = "akita"))]
pub fn cancel_dory_final_opening_commitments<PCS, VC, ZkProof>(
    proof: &mut JoltProof<PCS, VC, ZkProof>,
    tape: &ChallengeTape<PCS::Field>,
) where
    PCS: CommitmentScheme,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let gamma_base = tape
        .records
        .iter()
        .find(|record| {
            record.id.scope == FsScope::Stage8
                && matches!(record.id.kind, ChallengeKind::PowersBase { len } if len >= 2)
        })
        .expect("stage-8 final-opening batching challenge is missing");
    let gamma_0 = PCS::Field::one();
    let gamma_1 = gamma_base.values[0];
    let direction = proof
        .commitments
        .instruction_ra
        .first()
        .expect("proof has no instruction Ra commitment")
        .clone();

    let original_ram_inc = proof.commitments.ram_inc.clone();
    let original_rd_inc = proof.commitments.rd_inc.clone();
    proof.commitments.ram_inc =
        PCS::Output::linear_combine(&original_ram_inc, &direction, &gamma_1);
    proof.commitments.rd_inc = PCS::Output::linear_combine(&original_rd_inc, &direction, &-gamma_0);
    assert_ne!(proof.commitments.ram_inc, original_ram_inc);
    assert_ne!(proof.commitments.rd_inc, original_rd_inc);
}

/// Rewrites the clear stage-1 proof while preserving every algebraic check at
/// the recorded challenges.
pub fn equivocate_stage1_clear<PCS, VC, ZkProof>(
    proof: &mut JoltProof<PCS, VC, ZkProof>,
    tape: &ChallengeTape<PCS::Field>,
    delta: PCS::Field,
) where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let mut stage1 = tape
        .records
        .iter()
        .filter(|record| record.id.scope == FsScope::Stage1);

    let tau = stage1.next().expect("stage 1 tau challenge is missing");
    assert!(matches!(
        tau.id.kind,
        ChallengeKind::VectorElement { index: 0, .. }
    ));

    let uniskip_challenge = stage1
        .find(|record| record.id.kind == ChallengeKind::Challenge)
        .expect("stage 1 uni-skip challenge is missing")
        .values[0];
    let batching_coefficient = stage1
        .find(|record| record.id.kind == ChallengeKind::Scalar)
        .expect("stage 1 batching coefficient is missing")
        .values[0];
    let remainder_challenge = stage1
        .find(|record| record.id.kind == ChallengeKind::Challenge)
        .expect("stage 1 remainder challenge is missing")
        .values[0];

    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
        panic!("stage-1 clear equivocation requires a clear proof");
    };
    claims.stage1.uniskip_output_claim += delta;

    let SumcheckProof::Clear(ClearProof::Full(uniskip)) =
        &mut proof.stages.stage1_uni_skip_first_round_proof
    else {
        panic!("stage-1 uni-skip proof is not full cleartext");
    };
    let round = uniskip
        .round_polynomials
        .first_mut()
        .expect("stage-1 uni-skip round is missing");
    let mut coefficients = round.coefficients().to_vec();
    assert!(coefficients.len() >= 2);

    let domain = CenteredIntegerDomain::new(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE);
    let power_sums = domain
        .power_sums(2)
        .expect("stage-1 uni-skip domain is invalid");
    let domain_size = PCS::Field::from_i128(power_sums[0]);
    let mean = PCS::Field::from_i128(power_sums[1])
        * domain_size.inverse().expect("domain size is zero in field");
    let slope = delta
        * (uniskip_challenge - mean)
            .inverse()
            .expect("uni-skip challenge equals the domain mean");
    coefficients[0] -= slope * mean;
    coefficients[1] += slope;
    *round = UnivariatePoly::new(coefficients);

    let SumcheckProof::Clear(ClearProof::Compressed(remainder)) =
        &mut proof.stages.stage1_sumcheck_proof
    else {
        panic!("stage-1 remainder proof is not compressed cleartext");
    };
    let round = remainder
        .round_polynomials
        .first_mut()
        .expect("stage-1 remainder round is missing");
    let mut coefficients = round.coeffs_except_linear_term().to_vec();
    let hint_delta = batching_coefficient * delta;
    if coefficients.len() >= 2 {
        let denominator = remainder_challenge * remainder_challenge - remainder_challenge;
        coefficients[1] -= remainder_challenge
            * hint_delta
            * denominator
                .inverse()
                .expect("remainder challenge is Boolean");
    } else {
        let denominator = PCS::Field::one() - PCS::Field::from_u64(2) * remainder_challenge;
        coefficients[0] -= remainder_challenge
            * hint_delta
            * denominator
                .inverse()
                .expect("remainder challenge is one half");
    }
    *round = CompressedPoly::new(coefficients);
}
