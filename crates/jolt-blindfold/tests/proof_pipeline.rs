#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "integration tests should fail loudly"
)]

mod support;

use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use support::*;

#[test]
fn blindfold_protocol_pipeline_verifies_committed_sumcheck_outputs_and_eval_commitments() {
    let mut rng = ChaCha20Rng::from_seed([81; 32]);
    let full = prove_blindfold_protocol_pipeline(&mut rng);
    let protocol = &full.instance.fixture.protocol;

    assert!(protocol.dimensions.coefficient_rows > 0);
    assert!(protocol.dimensions.output_claim_rows > 0);
    assert!(protocol.dimensions.auxiliary_rows > 0);
    assert!(!protocol.eval_commitments.is_empty());
    full.instance
        .fixture
        .verify(&full.narg)
        .expect("reference prover's BlindFold proof verifies");
}

#[test]
fn blindfold_proof_randomness_is_empirically_independent() {
    const SAMPLES: usize = 128;
    let mut rng = ChaCha20Rng::from_seed([61; 32]);
    let mut projections = [
        StatisticalProjection::new("random_u", SAMPLES),
        StatisticalProjection::new("auxiliary_commitment", SAMPLES),
        StatisticalProjection::new("random_round_commitment", SAMPLES),
        StatisticalProjection::new("random_error_commitment", SAMPLES),
        StatisticalProjection::new("cross_term_commitment", SAMPLES),
        StatisticalProjection::new("outer_sumcheck", SAMPLES),
        StatisticalProjection::new("inner_sumcheck", SAMPLES),
        StatisticalProjection::new("witness_opening", SAMPLES),
        StatisticalProjection::new("error_opening", SAMPLES),
    ];

    for _ in 0..SAMPLES {
        let instance = build_protocol_backed_instance(&mut rng);
        let narg = instance.prove_real(&mut rng).expect("real prover succeeds");
        instance
            .fixture
            .verify(&narg)
            .expect("sample proof verifies");
        let messages = instance.fixture.messages(&narg);

        let values = [
            field_low_u64(messages.random_u),
            projection(b"auxiliary_commitment", &messages.auxiliary_rows[..1]),
            projection(b"random_round_commitment", &messages.random_rounds[..1]),
            projection(b"random_error_commitment", &messages.random_error_rows[..1]),
            projection(
                b"cross_term_commitment",
                &messages.cross_term_error_rows[..1],
            ),
            projection(b"outer_sumcheck", &messages.outer_rounds),
            projection(b"inner_sumcheck", &messages.inner_rounds),
            opening_projection(b"witness_opening", &messages.witness_opening),
            opening_projection(b"error_opening", &messages.error_opening),
        ];

        for (projection, value) in projections.iter_mut().zip(values) {
            projection.push(value);
        }
    }

    for projection in &projections {
        assert_empirical_distribution(projection);
    }
    assert_empirical_pairwise_independence(&projections[0], &projections[1]);
    assert_empirical_pairwise_independence(&projections[0], &projections[7]);
    assert_empirical_pairwise_independence(&projections[1], &projections[2]);
    assert_empirical_pairwise_independence(&projections[3], &projections[4]);
    assert_empirical_pairwise_independence(&projections[5], &projections[6]);
    assert_empirical_pairwise_independence(&projections[7], &projections[8]);
}

#[test]
fn final_eval_openings_use_dedicated_rows() {
    let mut rng = ChaCha20Rng::from_seed([110; 32]);
    let fixture = two_stage_fixture(&mut rng, Stage2Input::Constant);
    let coordinates = fixture
        .protocol
        .final_opening_witness_coordinates()
        .expect("final opening coordinates are valid");
    let eval = coordinates[0]
        .evaluation
        .expect("final opening has an evaluation coordinate");
    let blinding = coordinates[0]
        .blinding
        .expect("final opening has a blinding coordinate");

    assert_eq!(eval.column, 0);
    assert_eq!(blinding.column, 0);
    assert_ne!(eval.row, blinding.row);
}
