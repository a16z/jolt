#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "fixtures assert successful proof operations"
)]

use akita_config::{policy_of, CommitmentConfig};
use akita_params::{
    ChunkedWitnessCfg, CommittedGroupBatchProfile, GroupCommitPhaseParams, PolynomialGroupLayout,
    ScheduleLookupKey, SisTableKey,
};
use akita_schedules::ValidatedScheduleCatalog;
use jolt_akita::configs::{JoltOneHotK16, JoltOneHotK16Direct};
use jolt_akita::{
    AkitaChunkProfile, AkitaOneHotScheduleArtifact, AkitaScheduleArtifacts, AkitaScheme,
    AkitaSetupParams, AkitaVerifierSetup,
};
use jolt_openings::CommitmentScheme;
use jolt_transcript::{prover_transcript, Blake2b512};
use serde_json::Value;
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

fn one_hot_bundle(bytes: Vec<u8>, profile: AkitaChunkProfile) -> AkitaScheduleArtifacts {
    AkitaScheduleArtifacts::new(
        AkitaChunkProfile::default(),
        Vec::new(),
        Vec::new(),
        vec![AkitaOneHotScheduleArtifact {
            k: 16,
            profile,
            bytes,
        }],
    )
    .unwrap()
}

#[test]
fn catalog_admission_rejects_malformed_and_noncanonical_rows() {
    let profile = AkitaChunkProfile::new(2, 3);
    let bytes = plan_catalog(profile.chunks, profile.levels);
    let _catalog = one_hot_bundle(bytes.clone(), profile)
        .one_hot_catalog_for_profile(16, profile)
        .unwrap();

    let mut malformed: Value = serde_json::from_slice(&bytes).unwrap();
    *malformed
        .pointer_mut(
            "/rows/0/schedule/root/params/groups/entries/0/profile/inner/digits/num_digits",
        )
        .unwrap() = serde_json::json!("invalid digit count");
    let malformed = serde_json::to_vec(&malformed).unwrap();
    let mut noncanonical = bytes;
    noncanonical.push(b' ');
    for (bytes, expected) in [
        (malformed, "invalid schedule artifact row 0"),
        (noncanonical, "not in canonical JSON form"),
    ] {
        let error = one_hot_bundle(bytes, profile)
            .one_hot_catalog_for_profile(16, profile)
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn catalog_admission_rejects_wrong_chunk_count_and_depth() {
    let profile = AkitaChunkProfile::new(2, 3);
    let bytes = plan_catalog(profile.chunks, profile.levels);
    let _catalog = one_hot_bundle(bytes.clone(), profile)
        .one_hot_catalog_for_profile(16, profile)
        .unwrap();
    for wrong in [AkitaChunkProfile::new(4, 3), AkitaChunkProfile::new(2, 4)] {
        let error = one_hot_bundle(bytes.clone(), wrong)
            .one_hot_catalog_for_profile(16, wrong)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("policy does not match the runtime config"),
            "{error}"
        );
    }
}

#[test]
fn catalog_admission_audits_declared_a_bound() {
    let profile = AkitaChunkProfile::new(2, 3);
    let bytes = plan_catalog(profile.chunks, profile.levels);
    let _catalog = one_hot_bundle(bytes.clone(), profile)
        .one_hot_catalog_for_profile(16, profile)
        .unwrap();
    let mut artifact: Value = serde_json::from_slice(&bytes).unwrap();
    let key = artifact
        .pointer_mut("/rows/0/schedule/root/params/groups/entries/0/profile/inner/matrix/security_route/Linf")
        .unwrap();
    let original: SisTableKey = serde_json::from_value(key.clone()).unwrap();
    // Keep a canonical table key so rejection must reach the layout's norm audit.
    let smaller = akita_params::sis::sis_table_key_for_linf_bound(
        original.policy,
        original.table_digest,
        original.modulus_profile,
        original.role,
        original.ring_dimension,
        1,
    )
    .unwrap();
    assert!(smaller.coeff_linf_bound < original.coeff_linf_bound);
    *key = serde_json::to_value(smaller).unwrap();
    let error = one_hot_bundle(serde_json::to_vec(&artifact).unwrap(), profile)
        .one_hot_catalog_for_profile(16, profile)
        .unwrap_err();
    assert!(
        error.to_string().contains(&format!(
            "declared coefficient bound {} is below required bound",
            smaller.coeff_linf_bound
        )),
        "{error}"
    );
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
        &mut prover_transcript::<Blake2b512>(DOMAIN, b""),
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
        &mut prover_transcript::<Blake2b512>(DOMAIN, b""),
    )
    .unwrap();
    let mut tampered = encoded.clone();
    let catalog = tampered
        .pointer_mut("/schedule_artifacts/dense/dense")
        .unwrap()
        .as_array_mut()
        .unwrap();
    // Keep transport valid while making the embedded catalog noncanonical.
    catalog.push(serde_json::json!(b' '));
    let tampered: AkitaVerifierSetup = serde_json::from_value(tampered).unwrap();
    let error = AkitaScheme::verify(
        &commitment,
        &point,
        evaluation,
        &proof,
        &tampered,
        &mut prover_transcript::<Blake2b512>(DOMAIN, b""),
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("not in canonical JSON form"),
        "{error}"
    );
    encoded["schedule_artifacts"]["dense"]["dense_profile"]["chunks"] = serde_json::json!(8);
    encoded["schedule_artifacts"]["dense"]["dense_profile"]["levels"] = serde_json::json!(2);
    let wrong: AkitaVerifierSetup = serde_json::from_value(encoded).unwrap();
    assert!(AkitaScheme::verify(
        &commitment,
        &point,
        evaluation,
        &proof,
        &wrong,
        &mut prover_transcript::<Blake2b512>(DOMAIN, b"")
    )
    .is_err());
}
