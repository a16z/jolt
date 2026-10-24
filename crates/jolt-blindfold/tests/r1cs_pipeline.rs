#![expect(clippy::expect_used, reason = "integration tests should fail loudly")]

mod support;

use jolt_sumcheck::{grind_round, RoundMessage};
use jolt_transcript::{prover_transcript, Blake2b512, FieldTranscript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use support::*;

#[test]
fn r1cs_pipeline_rejects_late_stage_mismatch() {
    let mut prover = SumcheckTestProver::new(ChaCha20Rng::from_seed([6; 32]));
    let (stage1, stage2, stage3, values) = generated_deep_triple(&mut prover);
    let mut tampered_stage3 = stage3.clone();
    tampered_stage3.proof = stage2.proof.clone();

    // Regrind the swapped rounds' proof of work at stage 3's transcript
    // position, so the mismatch reaches the R1CS check under test.
    let mut transcript =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"blindfold-r1cs-e2e"), b"");
    for stage in [&stage1, &stage2] {
        let _consistency = stage
            .proof
            .verify_committed_consistency::<F, _>(stage.statement, &mut transcript)
            .expect("honest stage verifies");
    }
    let mut pow_nonces = Vec::new();
    for (round, committed) in tampered_stage3.proof.rounds.iter().enumerate() {
        committed.append_to_transcript(&mut transcript);
        grind_round::<F, _>(
            tampered_stage3.statement.degree,
            round,
            &mut pow_nonces,
            &mut transcript,
        )
        .expect("proof of work");
        let _challenge: F = transcript.challenge().expect("challenge draw");
    }
    tampered_stage3.proof.pow_nonces = pow_nonces;

    assert!(build_deep_relation(&stage1, &stage2, &tampered_stage3, &values).is_err());
}
