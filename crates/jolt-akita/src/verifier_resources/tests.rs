#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "fixtures assert provisioning outcomes and fail loudly"
)]
use super::*;
use crate::{
    AkitaBatchProof, AkitaChunkProfile, AkitaCommitment, AkitaScheduleArtifacts, AkitaScheme,
    AkitaSetupParams,
};
use jolt_field::Ring;
use jolt_openings::CommitmentScheme;
use jolt_poly::{MultilinearPoly, OneHotPolynomial, Polynomial};
use jolt_transcript::{Blake2bTranscript, Transcript};

fn params(vars: usize) -> AkitaSetupParams {
    AkitaSetupParams::dense_only(
        vars,
        1,
        [3; 32],
        AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2)),
    )
}

fn metadata(params: &AkitaSetupParams) -> AkitaVerifierSetup {
    AkitaScheme::verifier_setup_from_params(params).unwrap()
}

struct DenseFixture {
    commitment: AkitaCommitment,
    point: Vec<AkitaField>,
    evaluation: AkitaField,
    proof: AkitaBatchProof,
}

impl DenseFixture {
    fn new(vars: usize) -> Self {
        let (prover, _) = AkitaScheme::setup(params(vars)).unwrap();
        let polynomial = Polynomial::new(
            (0..1 << vars)
                .map(|i| AkitaField::from_u64(i % 16))
                .collect(),
        );
        let point = vec![AkitaField::from_u64(3); vars];
        let evaluation = polynomial.evaluate(&point);
        let (commitment, hint) = AkitaScheme::commit(&polynomial, &prover).unwrap();
        let proof = AkitaScheme::open(
            &polynomial,
            &point,
            evaluation,
            &prover,
            Some(hint),
            &mut Blake2bTranscript::new(b"verifier-resources"),
        )
        .unwrap();
        Self {
            commitment,
            point,
            evaluation,
            proof,
        }
    }

    fn verify(&self, setup: &AkitaVerifierSetup, evaluation: AkitaField) -> bool {
        AkitaScheme::verify(
            &self.commitment,
            &self.point,
            evaluation,
            &self.proof,
            setup,
            &mut Blake2bTranscript::new(b"verifier-resources"),
        )
        .is_ok()
    }
}

fn prepared(registry: &PreparedRegistry, setups: Vec<AkitaVerifierSetup>) -> Arc<Prepared> {
    let prepared = Arc::new(Prepared::new(setups.into_iter().map(Ok)).unwrap());
    registry.publish(&prepared);
    prepared
}

#[test]
fn unprepared_setups_build_unpublished_state_shared_by_clones() {
    let fixture = DenseFixture::new(14);
    let registry = PreparedRegistry::new();
    let setup = metadata(&params(14));
    registry.attach(&setup).unwrap();
    let clone = setup.clone();
    let other = metadata(&params(14));
    registry.attach(&other).unwrap();
    assert!(registry.live().is_empty());
    assert!(std::ptr::eq(
        setup.dense_verifier().unwrap(),
        clone.dense_verifier().unwrap()
    ));
    assert!(!std::ptr::eq(
        setup.dense_verifier().unwrap(),
        other.dense_verifier().unwrap()
    ));
    assert!(fixture.verify(&setup, fixture.evaluation));
    assert!(fixture.verify(&other, fixture.evaluation));
}

#[test]
fn setups_attach_to_the_smallest_covering_prepared_state() {
    let registry = PreparedRegistry::new();
    let large = prepared(&registry, vec![metadata(&params(30))]);
    let small = prepared(&registry, vec![metadata(&params(15))]);
    let setup = metadata(&params(14));
    registry.attach(&setup).unwrap();
    assert!(Arc::ptr_eq(
        setup.dense_verifier().unwrap().setup().expanded(),
        small.material.expanded()
    ));
    let transported: AkitaVerifierSetup =
        serde_json::from_str(&serde_json::to_string(&setup).unwrap()).unwrap();
    registry.attach(&transported).unwrap();
    assert!(std::ptr::eq(
        setup.dense_verifier().unwrap(),
        transported.dense_verifier().unwrap()
    ));
    let wide = metadata(&params(30));
    registry.attach(&wide).unwrap();
    assert!(Arc::ptr_eq(
        wide.dense_verifier().unwrap().setup().expanded(),
        large.material.expanded()
    ));
    assert_eq!(registry.live().len(), 2);
}

#[test]
fn a_group_wider_than_the_batch_is_rejected() {
    let mut setup = metadata(&params(14));
    setup.max_num_polys_per_commitment_group = setup.max_total_batch_polys + 1;
    assert!(PreparedRegistry::new().attach(&setup).is_err());
    assert!(AkitaVerifierResources::prepare([Ok(setup)]).is_err());
}

#[test]
fn prepared_setup_verifies_after_the_resources_are_dropped() {
    let fixture = DenseFixture::new(14);
    let setup = metadata(&params(14));
    let resources = AkitaVerifierResources::prepare([Ok(setup.clone())]).unwrap();
    assert!(fixture.verify(&setup, fixture.evaluation));
    drop(resources);
    assert!(fixture.verify(&setup, fixture.evaluation));
    assert!(!fixture.verify(&setup, fixture.evaluation + AkitaField::from_u64(1)));
}

#[test]
fn one_hot_proofs_use_the_same_warm_backend_at_different_arities() {
    let artifacts = params(14).schedule_artifacts;
    let request =
        |vars| AkitaSetupParams::one_hot_only(vars, 1, [5; 32], 16, Arc::clone(&artifacts));
    let registry = PreparedRegistry::new();
    let _state = prepared(
        &registry,
        vec![metadata(&request(15)), metadata(&params(15))],
    );
    let bound = [14, 15].map(|vars| {
        let setup = metadata(&request(vars));
        registry.attach(&setup).unwrap();
        (vars, setup)
    });
    assert!(std::ptr::eq(
        bound[0].1.one_hot_verifier().unwrap(),
        bound[1].1.one_hot_verifier().unwrap()
    ));
    for (vars, verifier) in bound {
        let (prover, _) = AkitaScheme::setup(request(vars)).unwrap();
        let polynomial = OneHotPolynomial::new(
            16,
            (0..1 << (vars - 4)).map(|i| Some((i % 16) as u8)).collect(),
        );
        let point = vec![AkitaField::from_u64(3); vars];
        let evaluation = polynomial.evaluate(&point);
        let (commitment, hint) = AkitaScheme::commit(&polynomial, &prover).unwrap();
        let proof = AkitaScheme::open(
            &polynomial,
            &point,
            evaluation,
            &prover,
            Some(hint),
            &mut Blake2bTranscript::new(b"warm-onehot"),
        )
        .unwrap();
        AkitaScheme::verify(
            &commitment,
            &point,
            evaluation,
            &proof,
            &verifier,
            &mut Blake2bTranscript::new(b"warm-onehot"),
        )
        .unwrap();
    }
}

#[test]
fn prepared_resources_hold_every_required_prefix_and_only_the_verifier_matrix() {
    let setup = metadata(&AkitaSetupParams::one_hot_only(
        25,
        51,
        [0; 32],
        16,
        params(14).schedule_artifacts,
    ));
    let requirements = AkitaVerifierRequirements::for_setups(std::slice::from_ref(&setup)).unwrap();
    assert!(!requirements.prefix_slots.is_empty());
    assert!(requirements.generation_field_elements > requirements.retained_field_elements);
    let state = Prepared::new([Ok(setup)]).unwrap();
    let material = &state.material;
    assert_eq!(
        material.expanded().descriptor().num_field_elements,
        requirements.retained_field_elements
    );
    assert!(requirements
        .prefix_slots
        .iter()
        .all(|id| material.prefix_slots().get(id).is_some()));
}

#[test]
fn every_catalog_holds_the_prefixes_of_the_rows_its_state_admits() {
    let artifacts = params(14).schedule_artifacts;
    let one_hot = metadata(&AkitaSetupParams::one_hot_only(
        20,
        1,
        [0; 32],
        16,
        Arc::clone(&artifacts),
    ));
    let dense = metadata(&AkitaSetupParams::dense_only(30, 1, [0; 32], artifacts));
    let state = Prepared::new([Ok(one_hot.clone()), Ok(dense)]).unwrap();
    let descriptor = state.material.expanded().descriptor();
    with_one_hot_family!(scheme one_hot.one_hot_scheme().unwrap(), |scheme| {
        let admitted = SetupRequirements::from_catalog(
            scheme.schedules(),
            descriptor.max_num_vars,
            descriptor.max_num_batched_polys,
        )
        .unwrap();
        assert!(!admitted.prefix_slot_ids().is_empty());
        assert!(admitted
            .prefix_slot_ids()
            .iter()
            .all(|id| state.material.prefix_slots().get(id).is_some()));
    });
}

#[test]
fn prepare_stops_at_the_first_error() {
    let setups = [
        Ok(metadata(&params(14))),
        Err(invalid_setup("unbuildable setup")),
    ]
    .into_iter()
    .chain(std::iter::from_fn(|| -> Option<Result<_, _>> {
        panic!("prepare read past the first error")
    }));
    assert!(AkitaVerifierResources::prepare(setups).is_err());
}
