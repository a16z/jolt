//! End-to-end pipeline over the prover-side assignment API:
//! `BlindFoldProtocol::assign_witness` → `jolt_blindfold::prove` →
//! `BlindFoldProtocol::verify`, cross-validating the crate's own prover
//! against the verifier on witnesses assembled from committed sumcheck data
//! and the protocol's public parts alone.

#![expect(clippy::expect_used, reason = "integration tests should fail loudly")]

mod support;

use jolt_blindfold::{BlindFoldProtocol, BlindFoldWitness, ProverError};
use jolt_claims::constant;
use jolt_crypto::{Bn254G1, PedersenSetup, VectorCommitment};
use jolt_sumcheck::{CommittedSumcheckWitness, SumcheckDomainSpec, SumcheckStatement};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use support::*;

const TRANSCRIPT_LABEL: &[u8] = b"assignment-backed-blindfold";

fn committed_witness(stage: &GeneratedStage) -> CommittedSumcheckWitness<F> {
    CommittedSumcheckWitness {
        round_coefficients: stage.coefficients.clone(),
        round_blindings: stage.blindings.clone(),
        output_claim_rows: stage.output_claim_rows.clone(),
        output_claim_blindings: stage.output_claim_blindings.clone(),
    }
}

struct AssignmentFixture {
    setup: PedersenSetup<Bn254G1>,
    protocol: BlindFoldProtocol<F, Bn254G1>,
    stage_witnesses: Vec<CommittedSumcheckWitness<F>>,
    eval_outputs: Vec<F>,
    eval_blindings: Vec<F>,
}

const STAGE_DOMAINS: [SumcheckDomainSpec; 2] = [
    SumcheckDomainSpec::BooleanHypercube,
    SumcheckDomainSpec::BooleanHypercube,
];

/// The two-stage committed pipeline of the proof tests, rebuilt through the
/// public builder: constant claim expressions, one final opening bound to
/// the first output claim.
fn assignment_fixture(rng: &mut impl RngCore) -> AssignmentFixture {
    let setup = pedersen_setup(4);
    let statement1 = SumcheckStatement::new(3, 3);
    let statement2 = SumcheckStatement::new(2, 3);
    let input1 = f(37);
    let input2 = f(89);

    let (stage1, stage2) = {
        let mut prover = SumcheckTestProver::new(&mut *rng);
        let mut transcript = Blake2bTranscript::<F>::new(TRANSCRIPT_LABEL);
        let stage1 =
            prover.prove_stage_with_output_claims(&setup, &mut transcript, statement1, input1, 2);
        let stage2 =
            prover.prove_stage_with_output_claims(&setup, &mut transcript, statement2, input2, 1);
        (stage1, stage2)
    };
    let stage1_output = *stage1.claim_outs.last().expect("stage has rounds");
    let stage2_output = *stage2.claim_outs.last().expect("stage has rounds");
    let eval_outputs = vec![stage1.output_claim_rows[0][0]];
    let eval_blindings = vec![rng_field(rng)];
    let eval_commitment = VC::commit(&setup, &[eval_outputs[0]], &eval_blindings[0]);

    let mut transcript = Blake2bTranscript::<F>::new(TRANSCRIPT_LABEL);
    let stage1_consistency = stage1
        .proof
        .verify_committed_consistency(statement1, &mut transcript)
        .expect("stage 1 committed proof transcript verifies");
    let stage2_consistency = stage2
        .proof
        .verify_committed_consistency(statement2, &mut transcript)
        .expect("stage 2 committed proof transcript verifies");

    let protocol = BlindFoldProtocol::<F, Bn254G1>::builder::<usize, (), usize>()
        .stage("assignment-stage-1")
        .sumcheck(statement1)
        .domain(SumcheckDomainSpec::BooleanHypercube)
        .consistency(stage1_consistency)
        .output_claim_rows(
            (0..stage1.proof.output_claims.commitments.len() * (statement1.degree + 1)).collect(),
            statement1.degree + 1,
            stage1.proof.output_claims.clone(),
        )
        .input_claim(constant(input1))
        .output_claim(constant(stage1_output))
        .finish_stage()
        .expect("stage 1 is complete")
        .stage("assignment-stage-2")
        .sumcheck(statement2)
        .domain(SumcheckDomainSpec::BooleanHypercube)
        .consistency(stage2_consistency)
        .output_claim_rows(
            (100..100 + stage2.proof.output_claims.commitments.len() * (statement2.degree + 1))
                .collect(),
            statement2.degree + 1,
            stage2.proof.output_claims.clone(),
        )
        .input_claim(constant(input2))
        .output_claim(constant(stage2_output))
        .finish_stage()
        .expect("stage 2 is complete")
        .final_opening(vec![0usize], vec![f(1)], eval_commitment)
        .build()
        .expect("protocol builds");

    AssignmentFixture {
        setup,
        protocol,
        stage_witnesses: vec![committed_witness(&stage1), committed_witness(&stage2)],
        eval_outputs,
        eval_blindings,
    }
}

#[test]
fn assigned_witness_proves_and_verifies_through_the_real_prover() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x00C0_57AB);
    let fixture = assignment_fixture(&mut rng);
    let stage_refs: Vec<&CommittedSumcheckWitness<F>> = fixture.stage_witnesses.iter().collect();

    let assigned = fixture
        .protocol
        .assign_witness(
            &STAGE_DOMAINS,
            &stage_refs,
            &fixture.eval_outputs,
            &fixture.eval_blindings,
            &mut rng,
        )
        .expect("witness assigns");
    let dimensions = &fixture.protocol.dimensions;
    assert_eq!(assigned.rows.len(), dimensions.witness.row_count);
    assert_eq!(assigned.blindings.len(), dimensions.witness.row_count);
    assert!(assigned
        .rows
        .iter()
        .all(|row| row.len() == dimensions.witness.row_len));

    let mut prover_transcript = Blake2bTranscript::<F>::new(TRANSCRIPT_LABEL);
    append_protocol_transcript_prefix(&fixture.protocol, &mut prover_transcript);
    let proof = jolt_blindfold::prove::<F, VC, _, _>(
        &fixture.setup,
        &fixture.protocol,
        &mut prover_transcript,
        BlindFoldWitness {
            rows: &assigned.rows,
            blindings: &assigned.blindings,
            eval_outputs: &fixture.eval_outputs,
            eval_blindings: &fixture.eval_blindings,
        },
        &mut rng,
    )
    .expect("assigned witness proves");

    let mut verifier_transcript = Blake2bTranscript::<F>::new(TRANSCRIPT_LABEL);
    append_protocol_transcript_prefix(&fixture.protocol, &mut verifier_transcript);
    fixture
        .protocol
        .verify::<VC, _>(&proof, &fixture.setup, &mut verifier_transcript)
        .expect("assigned-witness proof verifies");
}

#[test]
fn assign_witness_rejects_stage_count_mismatch() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x00C0_57AC);
    let fixture = assignment_fixture(&mut rng);
    let stage_refs: Vec<&CommittedSumcheckWitness<F>> =
        fixture.stage_witnesses.iter().take(1).collect();

    let result = fixture.protocol.assign_witness(
        &STAGE_DOMAINS,
        &stage_refs,
        &fixture.eval_outputs,
        &fixture.eval_blindings,
        &mut rng,
    );
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
    let fixture = assignment_fixture(&mut rng);
    let mut truncated = fixture.stage_witnesses.clone();
    let _ = truncated[0].round_coefficients.pop();
    let _ = truncated[0].round_blindings.pop();
    let stage_refs: Vec<&CommittedSumcheckWitness<F>> = truncated.iter().collect();

    let result = fixture.protocol.assign_witness(
        &STAGE_DOMAINS,
        &stage_refs,
        &fixture.eval_outputs,
        &fixture.eval_blindings,
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
