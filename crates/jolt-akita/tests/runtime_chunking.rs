#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "fixtures assert successful proof operations"
)]

use akita_config::{policy_of, CommitmentConfig};
use akita_params::{
    ChunkedWitnessCfg, CommittedGroupBatchProfile, GroupCommitPhaseParams, PolynomialGroupLayout,
    ScheduleLookupKey,
};
use akita_schedules::ValidatedScheduleCatalog;
use jolt_akita::configs::{JoltOneHotK16, JoltOneHotK16Direct};
use jolt_akita::{
    AkitaChunkProfile, AkitaScheduleArtifacts, AkitaScheme, AkitaSetupParams, AkitaVerifierSetup,
};
use jolt_openings::CommitmentScheme;
use jolt_transcript::{Blake2bTranscript, Transcript};
use std::{fs, sync::Arc};

#[expect(
    dead_code,
    reason = "shared fixture helpers are compiled per integration target"
)]
mod support;

const NUM_VARS: usize = 17;
const DOMAIN: &[u8] = b"configured-chunk-folds";

fn plan_catalog(chunks: usize, levels: usize) -> Vec<u8> {
    let chunking = ChunkedWitnessCfg {
        num_chunks: chunks,
        num_activated_levels: levels,
    };
    let key = ScheduleLookupKey::single(PolynomialGroupLayout::new(NUM_VARS, 1));
    let schedule = akita_planner::find_schedule(
        &key,
        JoltOneHotK16Direct::committed_source_contract().unwrap(),
        &[],
        &policy_of::<JoltOneHotK16Direct>(chunking),
        JoltOneHotK16Direct::ring_challenge_config,
    )
    .unwrap()
    .schedule;
    assert!(schedule.num_fold_levels() > levels);
    for (level, params) in std::iter::once(&schedule.root.params)
        .chain(schedule.recursive_folds.iter().map(|fold| &fold.params))
        .enumerate()
    {
        assert_eq!(
            params.witness_chunk.num_chunks,
            if level < levels { chunks } else { 1 }
        );
    }
    let profile = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &schedule.root.params,
        )
        .unwrap(),
        precommitteds: Vec::new(),
    };
    ValidatedScheduleCatalog::try_new(
        JoltOneHotK16::schedule_family_name(),
        vec![(profile, schedule)],
        &policy_of::<JoltOneHotK16>(chunking),
        JoltOneHotK16::ring_challenge_config,
    )
    .unwrap()
    .to_artifact_bytes()
    .unwrap()
}

#[test]
fn catalog_bundle_discovers_and_preserves_multiple_geometries() {
    let directory = std::env::temp_dir().join(format!(
        "jolt-runtime-catalog-bundle-{}",
        std::process::id()
    ));
    fs::create_dir_all(&directory).unwrap();
    for family in ["dense-bounded", "dense-full"] {
        let filename = format!("jolt-fp128-{family}.aks");
        let _ = fs::copy(
            AkitaScheduleArtifacts::packaged_directory().join(&filename),
            directory.join(filename),
        )
        .unwrap();
    }
    for levels in [3, 4] {
        fs::write(
            directory.join(format!("jolt-fp128-onehot-k16-w2r{levels}.aks")),
            plan_catalog(2, levels),
        )
        .unwrap();
    }
    let bundle =
        AkitaScheduleArtifacts::from_directory(&directory, AkitaChunkProfile::new(8, 2)).unwrap();
    let transported: AkitaScheduleArtifacts =
        serde_json::from_slice(&serde_json::to_vec(&bundle).unwrap()).unwrap();
    for levels in [3, 4] {
        let catalog = transported
            .one_hot_catalog_for_profile(16, AkitaChunkProfile::new(2, levels))
            .unwrap();
        let row = catalog
            .resolve_key(&ScheduleLookupKey::single(PolynomialGroupLayout::new(
                NUM_VARS, 1,
            )))
            .unwrap();
        let chunked_folds = std::iter::once(&row.schedule().root.params)
            .chain(
                row.schedule()
                    .recursive_folds
                    .iter()
                    .map(|fold| &fold.params),
            )
            .filter(|params| params.witness_chunk.num_chunks == 2)
            .count();
        assert_eq!(chunked_folds, levels);
    }
    assert!(transported
        .one_hot_catalog_for_profile(16, AkitaChunkProfile::new(2, 2))
        .is_err());
    let _ = fs::copy(
        directory.join("jolt-fp128-onehot-k16-w2r3.aks"),
        directory.join("jolt-fp128-onehot-k16-w02r3.aks"),
    )
    .unwrap();
    assert!(
        AkitaScheduleArtifacts::from_directory(&directory, AkitaChunkProfile::new(8, 2)).is_err()
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn dense_geometry_survives_verifier_transport() {
    use jolt_akita::configs::JoltDenseBounded;

    let profile = AkitaChunkProfile::new(1, 0);
    let policy = policy_of::<JoltDenseBounded>(profile.witness_cfg());
    let key = ScheduleLookupKey::single(PolynomialGroupLayout::new(14, 1));
    let schedule = akita_planner::find_schedule(
        &key,
        JoltDenseBounded::committed_source_contract().unwrap(),
        &[],
        &policy,
        JoltDenseBounded::ring_challenge_config,
    )
    .unwrap()
    .schedule;
    let catalog = ValidatedScheduleCatalog::try_new(
        JoltDenseBounded::schedule_family_name(),
        [(
            CommittedGroupBatchProfile {
                final_group: GroupCommitPhaseParams::try_from_params(
                    key.final_group,
                    &schedule.root.params,
                )
                .unwrap(),
                precommitteds: Vec::new(),
            },
            schedule,
        )],
        &policy,
        JoltDenseBounded::ring_challenge_config,
    )
    .unwrap()
    .to_artifact_bytes()
    .unwrap();
    let artifacts = AkitaScheduleArtifacts::new(profile, catalog, Vec::new(), Vec::new()).unwrap();
    let (prover, verifier) = AkitaScheme::setup(AkitaSetupParams::dense_only(
        14,
        1,
        support::layout(7),
        Arc::new(artifacts),
    ))
    .unwrap();
    let polynomial = support::polynomial(14, 1);
    let point: Vec<_> = (0..14).map(|i| support::f(i + 2)).collect();
    let evaluation = polynomial.evaluate(&point);
    let (commitment, hint) = AkitaScheme::commit(&polynomial, &prover).unwrap();
    let proof = AkitaScheme::open(
        &polynomial,
        &point,
        evaluation,
        &prover,
        Some(hint),
        &mut Blake2bTranscript::new(DOMAIN),
    )
    .unwrap();
    let mut encoded = serde_json::to_value(&verifier).unwrap();
    drop(verifier);
    drop(prover);
    let transported: AkitaVerifierSetup = serde_json::from_value(encoded.clone()).unwrap();
    AkitaScheme::verify(
        &commitment,
        &point,
        evaluation,
        &proof,
        &transported,
        &mut Blake2bTranscript::new(DOMAIN),
    )
    .unwrap();
    encoded["schedule_artifacts"]["dense"]["dense_profile"]["chunks"] = serde_json::json!(8);
    encoded["schedule_artifacts"]["dense"]["dense_profile"]["levels"] = serde_json::json!(2);
    let wrong: AkitaVerifierSetup = serde_json::from_value(encoded).unwrap();
    assert!(AkitaScheme::verify(
        &commitment,
        &point,
        evaluation,
        &proof,
        &wrong,
        &mut Blake2bTranscript::new(DOMAIN)
    )
    .is_err());
}
