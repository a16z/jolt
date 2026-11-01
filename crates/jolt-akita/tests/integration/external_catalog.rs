#![expect(
    clippy::expect_used,
    reason = "integration fixture failures must be explicit"
)]

use std::{fs, process::Command, sync::Arc};

use akita_config::{policy_of, CommitmentConfig};
use akita_params::{
    CommittedGroupBatchProfile, GroupCommitPhaseParams, PolynomialGroupLayout, ScheduleLookupKey,
};
use akita_schedules::ValidatedScheduleCatalog;
use jolt_akita::configs::{JoltOneHotK16, JoltOneHotK16Direct};
use jolt_akita::{
    AkitaBatchProof, AkitaChunkProfile, AkitaCommitment, AkitaField, AkitaNativeBatching,
    AkitaScheduleArtifacts, AkitaScheme, AkitaSetupParams, AkitaVerifierSetup,
};
use jolt_openings::{BatchOpeningScheme, CommitmentScheme};
use jolt_poly::{MultilinearPoly, OneHotPolynomial};
use jolt_transcript::{prover_transcript, Blake2b512};

use crate::support;

const TRANSPORT: &str = "JOLT_EXTERNAL_CATALOG_TEST_TRANSPORT";
const NUM_VARS: usize = 17;
const DOMAIN: &[u8] = b"external-multichunk-catalog";

/// The child process only receives the public setup, claims, and proof. Its
/// catalog directory has already been removed, and it never creates a prover setup.
#[test]
fn external_multichunk_catalog_survives_transport() {
    if let Some(path) = std::env::var_os(TRANSPORT) {
        let (setup, commitment, point, evaluation, proof): (
            AkitaVerifierSetup,
            AkitaCommitment,
            Vec<AkitaField>,
            AkitaField,
            AkitaBatchProof,
        ) = serde_json::from_slice(&fs::read(path).expect("read public verification inputs"))
            .expect("restore verifier without source catalogs");
        setup
            .prepare()
            .expect("key load prepares the restored setup");
        let mut statement = support::single_statement(commitment, &point, evaluation);
        assert_eq!(setup.akita_chunk_profile(), AkitaChunkProfile::new(2, 4));
        let mut transcript =
            prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(DOMAIN), b"");
        AkitaNativeBatching::verify_batch(&setup, &statement, &proof, &mut transcript)
            .expect("upstream verifier accepts the transported multi-chunk proof");
        let mut wrong = serde_json::to_value(&setup).expect("encode trusted setup");
        wrong["schedule_artifacts"]["one_hot"]["profile"]["levels"] = serde_json::json!(2);
        let wrong: AkitaVerifierSetup =
            serde_json::from_value(wrong).expect("decode altered setup");
        assert!(AkitaNativeBatching::verify_batch(
            &wrong,
            &statement,
            &proof,
            &mut prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(DOMAIN), b"")
        )
        .is_err());
        statement[0].evaluation.value += support::f(1);
        let mut transcript =
            prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(DOMAIN), b"");
        assert!(
            AkitaNativeBatching::verify_batch(&setup, &statement, &proof, &mut transcript).is_err()
        );
        return;
    }

    let directory =
        std::env::temp_dir().join(format!("jolt-external-catalog-{}", std::process::id()));
    fs::create_dir_all(&directory).expect("create application catalog directory");
    let catalogs = AkitaScheduleArtifacts::packaged_directory();
    for family in ["dense-bounded", "dense-full", "onehot-k16", "onehot-k256"] {
        let name = format!("jolt-fp128-{family}.aks");
        let _ = fs::copy(catalogs.join(&name), directory.join(name))
            .expect("install application-supplied catalogs");
    }
    let key = ScheduleLookupKey::single(PolynomialGroupLayout::new(NUM_VARS, 1));
    let packaged = AkitaScheduleArtifacts::from_directory(
        AkitaScheduleArtifacts::packaged_directory(),
        AkitaChunkProfile::new(8, 2),
    )
    .expect("load packaged fixture catalogs");
    assert!(
        packaged
            .one_hot_catalog_for_profile(16, AkitaChunkProfile::new(2, 4))
            .is_err(),
        "the external geometry must be absent from the packaged fixtures"
    );
    let schedule = akita_planner::find_schedule(
        &key,
        JoltOneHotK16Direct::committed_source_contract().expect("direct source contract"),
        &[],
        &policy_of::<JoltOneHotK16Direct>(AkitaChunkProfile::new(2, 4).witness_cfg()),
        JoltOneHotK16Direct::ring_challenge_config,
    )
    .expect("plan external-only shape")
    .schedule;
    let profile = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &schedule.root.params,
        )
        .expect("derive external committed profile"),
        precommitteds: Vec::new(),
    };
    let external = ValidatedScheduleCatalog::try_new(
        JoltOneHotK16::schedule_family_name(),
        vec![(profile, schedule)],
        &policy_of::<JoltOneHotK16>(AkitaChunkProfile::new(2, 4).witness_cfg()),
        JoltOneHotK16::ring_challenge_config,
    )
    .expect("admit external-only catalog");
    fs::write(
        directory.join("jolt-fp128-onehot-k16-w2r4.aks"),
        external
            .to_artifact_bytes()
            .expect("encode external catalog"),
    )
    .expect("install external-only catalog");
    let artifacts = Arc::new(
        AkitaScheduleArtifacts::from_directory(&directory, AkitaChunkProfile::new(8, 2))
            .expect("load external catalogs"),
    );
    let (prover, verifier) = AkitaScheme::setup(
        AkitaSetupParams::one_hot_only(NUM_VARS, 1, support::layout(3), 16, artifacts)
            .with_akita_chunk_profile(AkitaChunkProfile::new(2, 4)),
    )
    .expect("prepare multi-chunk setup");
    let polynomials = vec![OneHotPolynomial::new(
        16,
        (0..(1 << (NUM_VARS - 4)))
            .map(|i| Some((i % 16) as u8))
            .collect(),
    )];
    let point = (0..NUM_VARS)
        .map(|i| support::f((i + 2) as u64))
        .collect::<Vec<_>>();
    let evaluation = polynomials[0].evaluate(&point);
    let (commitment, hint) =
        AkitaScheme::commit_one_hot_group(&prover, support::layout(3), &polynomials)
            .expect("commit one-hot source");
    let statement = support::single_statement(commitment.clone(), &point, evaluation);
    let mut transcript =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(DOMAIN), b"");
    let proof = AkitaNativeBatching::prove_batch(
        &prover,
        statement.clone(),
        polynomials
            .iter()
            .map(|p| p as &dyn MultilinearPoly<AkitaField>)
            .collect(),
        hint,
        &mut transcript,
    )
    .expect("produce multi-chunk proof");
    let inputs = serde_json::to_vec(&(&verifier, &commitment, &point, evaluation, &proof))
        .expect("export public verification inputs");
    drop(prover);
    drop(verifier);
    fs::remove_dir_all(&directory).expect("remove all source catalogs");
    fs::create_dir(&directory).expect("create verification input directory");
    let transport = directory.join("public.json");
    fs::write(&transport, inputs).expect("publish public verification inputs");
    // libtest names a test by its module path without the crate name.
    let test_name = format!(
        "{}::external_multichunk_catalog_survives_transport",
        module_path!()
            .split_once("::")
            .expect("integration test module")
            .1
    );
    let result = Command::new(std::env::current_exe().expect("integration binary"))
        .args(["--exact", &test_name, "--nocapture"])
        .env(TRANSPORT, &transport)
        .env("JOLT_AKITA_SCHEDULE_DIR", directory.join("absent-catalogs"))
        .output()
        .expect("run independent upstream verifier process");
    fs::remove_dir_all(directory).expect("remove verification inputs");
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        result.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    // An `--exact` filter that matches nothing also exits successfully.
    assert!(
        stdout.contains("1 passed"),
        "the verifier process must run the transport test:\n{stdout}"
    );
}
