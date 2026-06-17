#![cfg_attr(
    feature = "core-fixtures",
    expect(
        clippy::expect_used,
        clippy::panic,
        reason = "fixture statement-shape tests should fail loudly when Stage 8 changes"
    )
)]

#[cfg(feature = "core-fixtures")]
use common::constants::XLEN as RISCV_XLEN;
#[cfg(feature = "core-fixtures")]
use jolt_claims::protocols::jolt::formulas::{
    committed_openings::final_opening_ids, dimensions::JoltFormulaDimensions,
};
#[cfg(feature = "core-fixtures")]
use jolt_crypto::{Bn254G1, Pedersen};
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_dory::DoryCommitment;
#[cfg(feature = "core-fixtures")]
use jolt_dory::DoryScheme;
#[cfg(feature = "core-fixtures")]
use jolt_field::Fr;
#[cfg(feature = "core-fixtures")]
use jolt_openings::PhysicalView;
#[cfg(feature = "core-fixtures")]
use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
#[cfg(feature = "core-fixtures")]
use jolt_verifier::stages::stage8::{Stage8BatchStatement, Stage8OpeningId};

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use crate::support::core_fixtures::CoreVerifierCase;

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage8_clear_batch_statement_matches_dory_direct_manifest() {
    let case = crate::support::core_fixtures::standard_muldiv_case();
    let Stage8BatchStatement::Clear(batch) = jolt_verifier::stage8_batch_statement::<
        Fr,
        DoryScheme,
        Pedersen<Bn254G1>,
        Blake2bTranscript,
        _,
    >(
        &case.preprocessing,
        &case.public_io,
        &case.proof,
        case.trusted_advice_commitment.as_ref(),
        false,
    )
    .expect("build Stage 8 batch statement") else {
        panic!("standard proof should build clear Stage 8 statement");
    };

    assert_clear_direct_statement(&case, &batch);
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn stage8_clear_batch_statement_includes_committed_program_layout() {
    let case = crate::support::core_fixtures::standard_committed_muldiv_case();
    let Stage8BatchStatement::Clear(batch) = jolt_verifier::stage8_batch_statement::<
        Fr,
        DoryScheme,
        Pedersen<Bn254G1>,
        Blake2bTranscript,
        _,
    >(
        &case.preprocessing,
        &case.public_io,
        &case.proof,
        case.trusted_advice_commitment.as_ref(),
        false,
    )
    .expect("build committed-program Stage 8 batch statement") else {
        panic!("standard proof should build clear Stage 8 statement");
    };

    assert_clear_direct_statement(&case, &batch);
    assert!(
        case.preprocessing.program.committed().is_some(),
        "fixture should use committed-program preprocessing"
    );
}

#[test]
#[cfg(all(feature = "core-fixtures", feature = "zk"))]
fn stage8_zk_batch_statement_matches_hidden_claim_manifest() {
    let case = crate::support::core_fixtures::zk_muldiv_case();
    let Stage8BatchStatement::Zk(batch) = jolt_verifier::stage8_batch_statement::<
        Fr,
        DoryScheme,
        Pedersen<Bn254G1>,
        Blake2bTranscript,
        _,
    >(
        &case.preprocessing,
        &case.public_io,
        &case.proof,
        None,
        true,
    )
    .expect("build ZK Stage 8 batch statement") else {
        panic!("ZK proof should build hidden-claim Stage 8 statement");
    };

    let expected_ids = expected_opening_ids(
        &case.preprocessing,
        case.proof.trace_length,
        case.proof.ram_K,
        &case.proof.one_hot_config,
    );
    assert_eq!(batch.opening_ids, expected_ids);
    assert_eq!(batch.statement.claims.len(), batch.opening_ids.len());
    assert_eq!(
        batch.statement.layout_digest,
        case.preprocessing.preprocessing_digest
    );
    assert_eq!(batch.statement.logical_point, batch.statement.pcs_point);
    assert_eq!(
        batch.statement.pcs_point,
        batch.pcs_opening_point.as_slice().to_vec()
    );

    for (claim, id) in batch.statement.claims.iter().zip(&batch.opening_ids) {
        assert_eq!(claim.id, *id);
        assert_eq!(claim.relation, *id);
        assert_eq!(claim.view, PhysicalView::Direct);
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn assert_clear_direct_statement(
    case: &CoreVerifierCase,
    batch: &jolt_verifier::stages::stage8::Stage8ClearBatchStatement<Fr, DoryCommitment>,
) {
    let expected_ids = expected_opening_ids(
        &case.preprocessing,
        case.proof.trace_length,
        case.proof.ram_K,
        &case.proof.one_hot_config,
    );
    assert_eq!(batch.opening_ids, expected_ids);
    assert_eq!(batch.statement.claims.len(), batch.opening_ids.len());
    assert_eq!(batch.opening_claims.len(), batch.opening_ids.len());
    assert_eq!(
        batch.statement.layout_digest,
        case.preprocessing.preprocessing_digest
    );
    assert_eq!(batch.statement.logical_point, batch.statement.pcs_point);
    assert_eq!(
        batch.statement.pcs_point,
        batch.pcs_opening_point.as_slice().to_vec()
    );

    for ((claim, opening_claim), id) in batch
        .statement
        .claims
        .iter()
        .zip(&batch.opening_claims)
        .zip(&batch.opening_ids)
    {
        assert_eq!(claim.id, *id);
        assert_eq!(claim.relation, *id);
        assert_eq!(claim.view, PhysicalView::Direct);
        assert_eq!(opening_claim.commitment, claim.commitment);
        assert_eq!(opening_claim.evaluation.point, batch.pcs_opening_point);
        assert_eq!(opening_claim.evaluation.value, claim.claim * claim.scale);
    }
}

#[cfg(feature = "core-fixtures")]
fn expected_opening_ids<PCS, VC>(
    preprocessing: &jolt_verifier::JoltVerifierPreprocessing<PCS, VC>,
    trace_length: usize,
    ram_k: usize,
    one_hot_config: &jolt_claims::protocols::jolt::JoltOneHotConfig,
) -> Vec<Stage8OpeningId>
where
    PCS: jolt_openings::CommitmentScheme<Field = Fr>,
    VC: jolt_crypto::VectorCommitment<Field = Fr>,
{
    let log_t = trace_length.ilog2() as usize;
    let dimensions = JoltFormulaDimensions::try_from(one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        preprocessing.program.bytecode_len(),
        ram_k,
    ))
    .expect("fixture dimensions should be valid");
    let committed_chunks = preprocessing
        .program
        .committed()
        .map(|committed| committed.bytecode_chunk_count());

    final_opening_ids(dimensions.ra_layout, false, false, committed_chunks)
        .into_iter()
        .map(Stage8OpeningId::from)
        .collect()
}

#[test]
#[cfg(not(feature = "core-fixtures"))]
#[ignore = "enable --features core-fixtures to inspect live Stage 8 batch statements"]
fn stage8_clear_batch_statement_matches_dory_direct_manifest() {}
