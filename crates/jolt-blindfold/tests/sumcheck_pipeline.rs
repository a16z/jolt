#![expect(clippy::expect_used, reason = "integration tests should fail loudly")]

mod support;

use jolt_crypto::VectorCommitment;
use jolt_sumcheck::RoundMessage;
use jolt_transcript::{prover_transcript, Blake2b512, FsChallenge};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use support::*;

#[test]
fn committed_sumcheck_pipeline_satisfies_deep_r1cs_and_randomness_checks() {
    const SAMPLES: usize = 256;
    let mut rng = ChaCha20Rng::from_seed([21; 32]);
    let mut projections = [
        StatisticalProjection::new("stage1_round_commitment", SAMPLES),
        StatisticalProjection::new("stage2_round_commitment", SAMPLES),
        StatisticalProjection::new("stage3_round_commitment", SAMPLES),
        StatisticalProjection::new("stage1_round_blinding", SAMPLES),
        StatisticalProjection::new("stage2_round_blinding", SAMPLES),
        StatisticalProjection::new("stage3_round_blinding", SAMPLES),
        StatisticalProjection::new("stage1_output_claim", SAMPLES),
        StatisticalProjection::new("stage2_output_claim", SAMPLES),
        StatisticalProjection::new("stage3_output_claim", SAMPLES),
    ];

    for _ in 0..SAMPLES {
        let mut prover = SumcheckTestProver::new(&mut rng);
        let (stage1, stage2, stage3, values) = generated_deep_triple(&mut prover);

        assert!(build_deep_relation(&stage1, &stage2, &stage3, &values).is_ok());
        let sample = [
            transcript_projection(&stage1.proof.rounds[0].commitment),
            transcript_projection(&stage2.proof.rounds[0].commitment),
            transcript_projection(&stage3.proof.rounds[0].commitment),
            field_low_u64(stage1.blindings[0]),
            field_low_u64(stage2.blindings[0]),
            field_low_u64(stage3.blindings[0]),
            field_low_u64(*stage1.claim_outs.last().expect("stage 1 output exists")),
            field_low_u64(*stage2.claim_outs.last().expect("stage 2 output exists")),
            field_low_u64(*stage3.claim_outs.last().expect("stage 3 output exists")),
        ];

        for (projection, value) in projections.iter_mut().zip(sample) {
            projection.push(value);
        }
    }

    for projection in &projections {
        assert_empirical_distribution(projection);
    }
    assert_empirical_pairwise_independence(&projections[0], &projections[3]);
    assert_empirical_pairwise_independence(&projections[1], &projections[4]);
    assert_empirical_pairwise_independence(&projections[2], &projections[5]);
    assert_empirical_pairwise_independence(&projections[3], &projections[6]);
    assert_empirical_pairwise_independence(&projections[4], &projections[7]);
    assert_empirical_pairwise_independence(&projections[5], &projections[8]);
}

#[test]
fn committed_round_blinding_is_empirically_independent_from_commitments_and_challenges() {
    const SAMPLES: usize = 512;
    let setup = pedersen_setup(4);
    let coefficients = vec![f(34), f(55), f(89), f(144)];
    let mut rng = ChaCha20Rng::from_seed([41; 32]);
    let mut blindings = StatisticalProjection::new("round_blinding", SAMPLES);
    let mut commitments = StatisticalProjection::new("round_commitment", SAMPLES);
    let mut challenges = StatisticalProjection::new("round_transcript_challenge", SAMPLES);

    for _ in 0..SAMPLES {
        let blinding = rng_field(&mut rng);
        let round = commit_round_with_blinding(&setup, coefficients.clone(), blinding);
        let mut transcript = prover_transcript(
            b"blindfold-r1cs-independence",
            [0u8; 32],
            Blake2b512::default(),
        );
        RoundMessage::<F>::append_to_transcript(&round, &mut transcript);
        let challenge: F = FsChallenge::<F>::challenge(&mut transcript);

        assert!(VC::verify(
            &setup,
            &round.commitment,
            &coefficients,
            &blinding
        ));
        assert!(!VC::verify(
            &setup,
            &round.commitment,
            &coefficients,
            &(blinding + f(1))
        ));
        blindings.push(field_low_u64(blinding));
        commitments.push(transcript_projection(&round.commitment));
        challenges.push(field_low_u64(challenge));
    }

    assert_empirical_distribution(&blindings);
    assert_empirical_distribution(&commitments);
    assert_empirical_distribution(&challenges);
    assert_empirical_pairwise_independence(&blindings, &commitments);
    assert_empirical_pairwise_independence(&commitments, &challenges);
}
