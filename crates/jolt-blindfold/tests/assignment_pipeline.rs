#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "integration tests should fail loudly"
)]

mod support;

use jolt_blindfold::{AssignedBlindFoldWitness, ProverError};
use jolt_crypto::JoltGroup;
use jolt_sumcheck::{CommittedSumcheckWitness, SumcheckDomainSpec};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use support::*;

const STAGE_DOMAINS: [SumcheckDomainSpec; 2] = [
    SumcheckDomainSpec::BooleanHypercube,
    SumcheckDomainSpec::BooleanHypercube,
];

fn assign(
    fixture: &TwoStageFixture,
    stage_witnesses: &[&CommittedSumcheckWitness<F>],
    rng: &mut ChaCha20Rng,
) -> Result<AssignedBlindFoldWitness<F>, ProverError<F>> {
    fixture.protocol.assign_witness(
        &STAGE_DOMAINS,
        stage_witnesses,
        &fixture.eval_outputs,
        &fixture.eval_blindings,
        rng,
    )
}

#[test]
fn assigned_witness_proves_and_verifies_through_the_real_prover() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x00C0_57AB);
    let fixture = two_stage_fixture(&mut rng, Stage2Input::Constant);

    let assigned = assign(&fixture, &fixture.stage_witnesses(), &mut rng).expect("witness assigns");
    let dimensions = &fixture.protocol.dimensions;
    assert_eq!(assigned.rows.len(), dimensions.witness.row_count);
    assert_eq!(assigned.blindings.len(), dimensions.witness.row_count);
    assert!(assigned
        .rows
        .iter()
        .all(|row| row.len() == dimensions.witness.row_len));

    let narg = fixture
        .prove(&assigned.rows, &assigned.blindings, &mut rng)
        .expect("assigned witness proves");
    fixture
        .verify(&narg)
        .expect("assigned-witness proof verifies");
}

#[test]
fn product_auxiliaries_solve_and_prove_through_the_real_prover() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x00C0_57AE);
    let fixture = two_stage_fixture(&mut rng, Stage2Input::ProductOfStage1Openings);

    // The point of the fixture: the claim lowering allocated at least one
    // product auxiliary, so `assign_witness` runs the solver.
    assert!(fixture.protocol.dimensions.auxiliary_values > 0);

    let assigned =
        assign(&fixture, &fixture.stage_witnesses(), &mut rng).expect("product witness assigns");
    let narg = fixture
        .prove(&assigned.rows, &assigned.blindings, &mut rng)
        .expect("product witness proves");
    fixture
        .verify(&narg)
        .expect("product-auxiliary proof verifies");
}

#[test]
fn assign_witness_rejects_stage_count_mismatch() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x00C0_57AC);
    let fixture = two_stage_fixture(&mut rng, Stage2Input::Constant);

    let result = assign(&fixture, &fixture.stage_witnesses()[..1], &mut rng);
    assert!(matches!(
        result,
        Err(ProverError::LengthMismatch {
            name: "stage witnesses",
            ..
        })
    ));
}

#[test]
fn assign_witness_rejects_round_shape_mismatch() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x00C0_57AD);
    let fixture = two_stage_fixture(&mut rng, Stage2Input::Constant);
    let mut truncated = fixture.stage_witnesses()[0].clone();
    let _ = truncated.round_coefficients.pop();
    let _ = truncated.round_blindings.pop();

    let result = assign(
        &fixture,
        &[&truncated, fixture.stage_witnesses()[1]],
        &mut rng,
    );
    assert!(matches!(
        result,
        Err(ProverError::StageWitnessShape {
            stage_index: 0,
            name: "round count",
            ..
        })
    ));
}

#[test]
fn assign_witness_rejects_round_blinding_count_mismatch() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x00C0_57AE);
    let fixture = two_stage_fixture(&mut rng, Stage2Input::Constant);
    let mut truncated = fixture.stage_witnesses()[0].clone();
    let _ = truncated.round_blindings.pop();

    let result = assign(
        &fixture,
        &[&truncated, fixture.stage_witnesses()[1]],
        &mut rng,
    );
    assert!(matches!(
        result,
        Err(ProverError::StageWitnessShape {
            stage_index: 0,
            name: "round blinding count",
            ..
        })
    ));
}

#[test]
fn assign_witness_rejects_output_claim_blinding_count_mismatch() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x00C0_57AF);
    let fixture = two_stage_fixture(&mut rng, Stage2Input::Constant);
    let mut extended = fixture.stage_witnesses()[0].clone();
    // A surplus blind is the silent-truncation direction of the old bug.
    let surplus = extended
        .output_claim_blindings
        .first()
        .copied()
        .expect("fixture stages carry output-claim blinds");
    extended.output_claim_blindings.push(surplus);

    let result = assign(
        &fixture,
        &[&extended, fixture.stage_witnesses()[1]],
        &mut rng,
    );
    assert!(matches!(
        result,
        Err(ProverError::StageWitnessShape {
            stage_index: 0,
            name: "output claim blinding count",
            ..
        })
    ));
}

/// The final-opening rows are opened at fixed coordinates, so the proof
/// publishes their real-instance commitments together with the folded
/// opening blindings. Unblinding a real row with the published folded
/// blinding must not recover a deterministic commitment to the hidden
/// evaluation or to its Dory blinding.
#[test]
fn final_opening_rows_stay_hidden_from_public_proof_data() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x00C0_57AC);
    let fixture = two_stage_fixture(&mut rng, Stage2Input::Constant);
    let assigned = assign(&fixture, &fixture.stage_witnesses(), &mut rng).expect("witness assigns");
    let narg = fixture
        .prove(&assigned.rows, &assigned.blindings, &mut rng)
        .expect("assigned witness proves");
    let messages = fixture.messages(&narg);

    let g0 = fixture.setup.message_generators[0];
    let h = fixture.setup.blinding_generator;
    let aux_start = fixture.protocol.dimensions.witness_rows.auxiliary.start;
    let coordinates = fixture
        .protocol
        .final_opening_witness_coordinates()
        .expect("final opening coordinates")[0];

    let eval_row = coordinates.evaluation.expect("evaluation row").row;
    let real_eval_row = messages.auxiliary_rows[eval_row - aux_start];
    let folded_eval_blinding = messages.eval_output_openings[0].combined_blinding;
    assert_ne!(
        real_eval_row - h.scalar_mul(&folded_eval_blinding),
        g0.scalar_mul(&fixture.eval_outputs[0]),
        "the hidden batched evaluation is recoverable as g0^y from public proof data"
    );

    let blinding_row = coordinates.blinding.expect("blinding row").row;
    let real_blinding_row = messages.auxiliary_rows[blinding_row - aux_start];
    let folded_blinding_blinding = messages.eval_blinding_openings[0].combined_blinding;
    assert_ne!(
        real_blinding_row - h.scalar_mul(&folded_blinding_blinding),
        g0.scalar_mul(&fixture.eval_blindings[0]),
        "the hidden Dory evaluation blinding is recoverable as g0^r from public proof data"
    );
}
