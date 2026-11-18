//! A prepared verifier setup survives transport the way a guest receives it:
//! payloads detached from the record and attached back in place.
#![expect(clippy::unwrap_used, reason = "tests unwrap successful PCS operations")]

use std::sync::Arc;

use akita_params::FlatMatrix;
use akita_pcs::{AkitaDeserialize, AkitaSerialize, Compress, Validate};
use akita_types::{
    sample_akita_setup_seed, AkitaExpandedSetup, AkitaSetupDescriptor, AkitaSetupSeed,
    AkitaVerifierSetup as BackendSetup, SetupPrefixVerifierRegistry,
};
use jolt_akita::{
    AkitaBatchProof, AkitaChunkProfile, AkitaCommitment, AkitaField, AkitaNativeBatching,
    AkitaScheduleArtifacts, AkitaScheme, AkitaSetupParams, AkitaVerifierResources,
    AkitaVerifierSetup, TrustedBytes,
};
use jolt_field::Ring;
use jolt_openings::{BatchOpeningScheme, CommitmentScheme, EvaluationClaim, VerifierOpeningClaim};
use jolt_poly::{MultilinearPoly, OneHotPolynomial, Polynomial, TensorEqTable};
use jolt_transcript::Blake2b512;

#[test]
fn detached_payloads_verify_once_attached_in_place() {
    let artifacts =
        AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
    let (prover_setup, verifier_setup) = AkitaScheme::setup(AkitaSetupParams::dense_only(
        14,
        1,
        [7; 32],
        Arc::clone(&artifacts),
    ))
    .unwrap();
    let polynomial = Polynomial::new(
        (0..(1u64 << 14))
            .map(|i| AkitaField::from_u64(3 + 7 * i))
            .collect(),
    );
    let (commitment, hint) = AkitaScheme::commit(&polynomial, &prover_setup).unwrap();
    let point = (5..19).map(AkitaField::from_u64).collect::<Vec<_>>();
    let value = polynomial.evaluate(&point);
    let statement = vec![VerifierOpeningClaim {
        commitment,
        evaluation: EvaluationClaim::new(point, value),
    }];
    let mut prover_transcript = jolt_transcript::prover_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"prepared"),
        b"",
    );
    let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        vec![&polynomial],
        hint,
        &mut prover_transcript,
    )
    .unwrap();

    let mut prepared = verifier_setup;
    prepared
        .prepare_verifier(proof.schedule_row_digest())
        .unwrap();
    let config = bincode::config::standard();
    let encoded = bincode::serde::encode_to_vec(&prepared, config).unwrap();
    let (mut transported, _): (AkitaVerifierSetup, _) =
        bincode::serde::decode_from_slice(&encoded, config).unwrap();
    let bodies = transported.detach_prepared_payloads().unwrap();
    let bodies: Vec<TrustedBytes> = bodies
        .into_iter()
        // SAFETY: the bodies are this test's own detached payloads.
        .map(|body| unsafe { TrustedBytes::new(Box::leak(body.into_boxed_slice())) })
        .collect();

    assert!(transported
        .clone()
        .attach_prepared_payloads(&bodies[1..])
        .is_err());
    transported.attach_prepared_payloads(&bodies).unwrap();
    let mut transcript = jolt_transcript::prover_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"prepared"),
        b"",
    );
    <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
        &transported,
        &statement,
        &proof,
        &mut transcript,
    )
    .unwrap();
}

struct ScalarFixture {
    setup: AkitaVerifierSetup,
    commitment: AkitaCommitment,
    point: Vec<AkitaField>,
    value: AkitaField,
    proof: AkitaBatchProof,
}

impl ScalarFixture {
    fn new(variables: usize, one_hot: bool) -> Self {
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let params = if one_hot {
            AkitaSetupParams::one_hot_only(variables, 1, [7; 32], 16, artifacts)
        } else {
            AkitaSetupParams::dense_only(variables, 1, [7; 32], artifacts)
        };
        let (prover, setup) = AkitaScheme::setup(params).unwrap();
        let polynomial: Box<dyn MultilinearPoly<AkitaField>> = if one_hot {
            Box::new(OneHotPolynomial::new(
                16,
                (0..1 << (variables - 4))
                    .map(|i| Some((i % 16) as u8))
                    .collect(),
            ))
        } else {
            Box::new(Polynomial::new(
                (0..1 << variables)
                    .map(|i| AkitaField::from_u64(i % 16))
                    .collect(),
            ))
        };
        let point = vec![AkitaField::from_u64(3); variables];
        let value = if let Some(indices) = polynomial.one_hot_indices() {
            let equality = TensorEqTable::<AkitaField>::new(&point);
            indices
                .iter()
                .enumerate()
                .filter_map(|(row, col)| {
                    col.map(|col| equality.evaluate_index(row * 16 + usize::from(col)))
                })
                .fold(AkitaField::from_u64(0), |sum, value| sum + value)
        } else {
            polynomial.evaluate(&point)
        };
        let (commitment, hint) = AkitaScheme::commit(&*polynomial, &prover).unwrap();
        let proof = AkitaScheme::open(
            &*polynomial,
            &point,
            value,
            &prover,
            Some(hint),
            &mut jolt_transcript::prover_transcript::<Blake2b512>(
                &jolt_transcript::protocol_tag(b"general-trusted-setup"),
                b"",
            ),
        )
        .unwrap();
        Self {
            setup,
            commitment,
            point,
            value,
            proof,
        }
    }

    fn verify(&self, value: AkitaField) -> bool {
        let config = bincode::config::standard();
        let encoded = bincode::serde::encode_to_vec(&self.setup, config).unwrap();
        let (transported, _): (AkitaVerifierSetup, _) =
            bincode::serde::decode_from_slice(&encoded, config).unwrap();
        AkitaScheme::verify(
            &self.commitment,
            &self.point,
            value,
            &self.proof,
            &transported,
            &mut jolt_transcript::prover_transcript::<Blake2b512>(
                &jolt_transcript::protocol_tag(b"general-trusted-setup"),
                b"",
            ),
        )
        .is_ok()
    }
}

fn trust_setup(bytes: Vec<u8>) -> TrustedBytes {
    // SAFETY: test bodies contain our producer's canonical matrix words and
    // immutable storage is retained for the entire test process. Negative cases
    // alter framing/metadata or substitute known canonical matrix words.
    unsafe { TrustedBytes::new(Box::leak(bytes.into_boxed_slice())) }
}

#[test]
fn full_setup_import_verifies_distinct_rows_without_preselection() {
    let fixtures = [ScalarFixture::new(14, false), ScalarFixture::new(15, false)];
    assert_ne!(
        fixtures[0].proof.schedule_row_digest(),
        fixtures[1].proof.schedule_row_digest()
    );
    let capacity = AkitaScheme::verifier_setup_from_params(&AkitaSetupParams::dense_only(
        16,
        2,
        [7; 32],
        AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2)),
    ))
    .unwrap();
    let body = AkitaVerifierResources::prepare([Ok(capacity.clone())])
        .unwrap()
        .prepared_setup_bytes()
        .unwrap();
    let imported = AkitaVerifierResources::prepare_with_trusted_setup(
        [Ok(capacity.clone())],
        trust_setup(body.clone()),
    )
    .unwrap();
    for fixture in &fixtures {
        assert!(fixture.verify(fixture.value));
        assert!(!fixture.verify(fixture.value + AkitaField::from_u64(1)));
    }
    drop(imported);

    // Canonical substituted words must affect verification; silently deriving
    // the original matrix instead would make the honest proofs pass.
    let skew = usize::from(*body.first().unwrap());
    let key =
        BackendSetup::<AkitaField>::deserialize_uncompressed_exact(body.get(skew..).unwrap(), &())
            .unwrap();
    let descriptor = key.expanded().descriptor().clone();
    let substituted = BackendSetup::from_parts(
        Arc::new(
            AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                descriptor.clone(),
                FlatMatrix::from_flat_data(vec![
                    AkitaField::from_u64(0);
                    descriptor.num_field_elements
                ]),
            ),
        ),
        key.prefix_slots().clone(),
    )
    .unwrap();
    let mut changed = body.get(..skew).unwrap().to_vec();
    substituted
        .serialize_with_mode(&mut changed, Compress::No)
        .unwrap();
    let _imported =
        AkitaVerifierResources::prepare_with_trusted_setup([Ok(capacity)], trust_setup(changed))
            .unwrap();
    for fixture in &fixtures {
        assert!(!fixture.verify(fixture.value));
    }
}

#[test]
fn full_setup_import_verifies_offloaded_fold_and_requires_its_prefixes() {
    let fixture = ScalarFixture::new(27, true);
    let catalog = fixture.setup.one_hot_catalog().unwrap();
    let row = catalog
        .rows()
        .find(|row| row.selection().row_digest.as_bytes() == &fixture.proof.schedule_row_digest())
        .unwrap();
    assert!(row
        .schedule()
        .recursive_folds
        .iter()
        .any(|fold| fold.params.setup_prefix().is_some()));
    let body = AkitaVerifierResources::prepare([Ok(fixture.setup.clone())])
        .unwrap()
        .prepared_setup_bytes()
        .unwrap();
    let _imported = AkitaVerifierResources::prepare_with_trusted_setup(
        [Ok(fixture.setup.clone())],
        trust_setup(body.clone()),
    )
    .unwrap();
    assert!(fixture.verify(fixture.value));
    assert!(!fixture.verify(fixture.value + AkitaField::from_u64(1)));

    let skew = usize::from(*body.first().unwrap());
    let key =
        BackendSetup::<AkitaField>::deserialize_uncompressed_exact(body.get(skew..).unwrap(), &())
            .unwrap();
    let without_prefixes = BackendSetup::from_parts(
        Arc::clone(key.expanded()),
        SetupPrefixVerifierRegistry::new(sample_akita_setup_seed()),
    )
    .unwrap();
    let mut missing = body.get(..skew).unwrap().to_vec();
    without_prefixes
        .serialize_with_mode(&mut missing, Compress::No)
        .unwrap();
    assert!(AkitaVerifierResources::prepare_with_trusted_setup(
        [Ok(fixture.setup)],
        trust_setup(missing)
    )
    .is_err());
}

#[test]
fn full_setup_import_rejects_wrong_capacity_seed_and_truncated_bodies() {
    let artifacts =
        AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
    let setup = AkitaScheme::verifier_setup_from_params(&AkitaSetupParams::dense_only(
        14,
        1,
        [7; 32],
        Arc::clone(&artifacts),
    ))
    .unwrap();
    let body = AkitaVerifierResources::prepare([Ok(setup.clone())])
        .unwrap()
        .prepared_setup_bytes()
        .unwrap();
    let larger = AkitaScheme::verifier_setup_from_params(&AkitaSetupParams::dense_only(
        15, 1, [7; 32], artifacts,
    ))
    .unwrap();
    assert!(AkitaVerifierResources::prepare_with_trusted_setup(
        [Ok(larger)],
        trust_setup(body.clone())
    )
    .is_err());
    assert!(AkitaVerifierResources::prepare_with_trusted_setup(
        [Ok(setup.clone())],
        trust_setup(body.get(..body.len() - 1).unwrap().to_vec())
    )
    .is_err());
    assert!(AkitaVerifierResources::prepare_with_trusted_setup(
        [Ok(setup.clone())],
        trust_setup(Vec::new())
    )
    .is_err());

    let skew = usize::from(*body.first().unwrap());
    let mut reader = body.get(skew..).unwrap();
    let descriptor =
        AkitaSetupDescriptor::deserialize_with_mode(&mut reader, Compress::No, Validate::Yes, &())
            .unwrap();
    let mut wrong_seed = descriptor.clone();
    wrong_seed.setup_seed = AkitaSetupSeed::shake256_paged_v1([42; 32]);
    let mut wrong_count = descriptor;
    wrong_count.num_field_elements = usize::MAX;
    for descriptor in [wrong_seed, wrong_count] {
        let mut invalid = body.clone();
        let mut header = Vec::new();
        descriptor
            .serialize_with_mode(&mut header, Compress::No)
            .unwrap();
        invalid
            .get_mut(skew..skew + header.len())
            .unwrap()
            .copy_from_slice(&header);
        assert!(AkitaVerifierResources::prepare_with_trusted_setup(
            [Ok(setup.clone())],
            trust_setup(invalid)
        )
        .is_err());
    }
}
