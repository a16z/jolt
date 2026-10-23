#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "fixtures assert provisioning outcomes and fail loudly"
)]
use super::*;
use crate::{AkitaChunkProfile, AkitaScheduleArtifacts, AkitaScheme, AkitaSetupParams};
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
    commitment: crate::AkitaCommitment,
    point: Vec<AkitaField>,
    evaluation: AkitaField,
    proof: crate::AkitaBatchProof,
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

fn current(lazy: &LazyVerifierResources) -> Arc<Prepared> {
    lazy.live().last().and_then(Weak::upgrade).unwrap()
}

#[test]
fn lazy_setups_share_one_state_and_grow_in_either_order() {
    let fixtures = [14, 15].map(DenseFixture::new);
    for reverse in [false, true] {
        let lazy = LazyVerifierResources::new();
        let setups = [metadata(&params(14)), metadata(&params(15))];
        let order: [usize; 2] = if reverse { [1, 0] } else { [0, 1] };
        for index in order {
            lazy.attach(&setups[index]).unwrap();
        }
        for (fixture, setup) in fixtures.iter().zip(&setups) {
            assert!(fixture.verify(setup, fixture.evaluation));
        }
        let shared = current(&lazy);
        assert_eq!(shared.catalogs.len(), 1);
        let fresh = metadata(&params(14));
        lazy.attach(&fresh).unwrap();
        assert!(Arc::ptr_eq(&shared, &current(&lazy)));
        assert!(Arc::ptr_eq(
            fresh.dense_verifier().unwrap().setup().expanded(),
            shared.material.expanded()
        ));
        if reverse {
            assert!(std::ptr::eq(
                setups[0].dense_verifier().unwrap(),
                setups[1].dense_verifier().unwrap()
            ));
        }
    }
}

#[test]
fn growth_keeps_computed_prefixes_and_extends_the_same_stream() {
    let lazy = LazyVerifierResources::new();
    let small = metadata(&AkitaSetupParams::one_hot_only(
        20,
        30,
        [0; 32],
        16,
        params(14).schedule_artifacts,
    ));
    lazy.attach(&small).unwrap();
    let before = current(&lazy);
    let large = metadata(&AkitaSetupParams::one_hot_only(
        25,
        51,
        [0; 32],
        16,
        params(14).schedule_artifacts,
    ));
    lazy.attach(&large).unwrap();
    let after = current(&lazy);
    assert!(!Arc::ptr_eq(&before, &after));
    for id in AkitaVerifierRequirements::for_setups(std::slice::from_ref(&small))
        .unwrap()
        .prefix_slots
    {
        assert_eq!(
            before.material.prefix_slots().get(&id),
            after.material.prefix_slots().get(&id)
        );
    }
    let old = before.material.expanded().shared_matrix().as_field_slice();
    let new = after.material.expanded().shared_matrix().as_field_slice();
    assert_eq!(Some(old), new.get(..old.len()));
    drop(before);
    assert!(small.one_hot_verifier().is_ok());
}

fn prepared(lazy: &LazyVerifierResources, setups: Vec<AkitaVerifierSetup>) -> Arc<Prepared> {
    let prepared = Arc::new(Prepared::grow(setups.into_iter().map(Ok), None).unwrap());
    lazy.publish(&prepared);
    prepared
}

#[test]
fn setups_attach_to_prepared_state_without_growing_it() {
    let lazy = LazyVerifierResources::new();
    let state = prepared(&lazy, vec![metadata(&params(15))]);
    let small = metadata(&params(14));
    lazy.attach(&small).unwrap();
    let transported: AkitaVerifierSetup =
        serde_json::from_str(&serde_json::to_string(&small).unwrap()).unwrap();
    lazy.attach(&transported).unwrap();
    assert!(Arc::ptr_eq(&state, &current(&lazy)));
    assert!(std::ptr::eq(
        small.dense_verifier().unwrap(),
        transported.dense_verifier().unwrap()
    ));
    assert!(Arc::ptr_eq(
        small.dense_verifier().unwrap().setup().expanded(),
        state.material.expanded()
    ));
    let larger = metadata(&params(16));
    lazy.attach(&larger).unwrap();
    assert!(!Arc::ptr_eq(&state, &current(&lazy)));
    drop(larger);
    let later = metadata(&params(14));
    lazy.attach(&later).unwrap();
    assert!(Arc::ptr_eq(&state, &current(&lazy)));
    assert!(std::ptr::eq(
        small.dense_verifier().unwrap(),
        later.dense_verifier().unwrap()
    ));
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
    let lazy = LazyVerifierResources::new();
    let _state = prepared(&lazy, vec![metadata(&request(15)), metadata(&params(15))]);
    let bound = [14, 15].map(|vars| {
        let setup = metadata(&request(vars));
        lazy.attach(&setup).unwrap();
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
    let state = Prepared::grow([Ok(setup)], None).unwrap();
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
fn each_catalog_is_sized_by_its_own_largest_setup() {
    let one_hot = metadata(&AkitaSetupParams::one_hot_only(
        20,
        30,
        [0; 32],
        16,
        params(14).schedule_artifacts,
    ));
    let dense = metadata(&params(30));
    let alone = |setup: &AkitaVerifierSetup| {
        AkitaVerifierRequirements::for_setups(std::slice::from_ref(setup)).unwrap()
    };
    let together =
        AkitaVerifierRequirements::for_setups(&[one_hot.clone(), dense.clone()]).unwrap();
    let (one_hot, dense) = (alone(&one_hot), alone(&dense));
    assert_eq!(
        together.prefix_slots,
        &one_hot.prefix_slots | &dense.prefix_slots
    );
    assert_eq!(
        together.retained_field_elements,
        one_hot
            .retained_field_elements
            .max(dense.retained_field_elements)
    );
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

#[test]
fn growing_one_catalog_rebuilds_its_verifier_with_the_new_prefixes() {
    let artifacts = params(14).schedule_artifacts;
    let one_hot = |vars, polys| {
        metadata(&AkitaSetupParams::one_hot_only(
            vars,
            polys,
            [0; 32],
            16,
            Arc::clone(&artifacts),
        ))
    };
    // The wide dense demand sizes the matrix, so growing the one-hot catalog
    // adds prefix commitments while keeping the matrix.
    let dense = metadata(&AkitaSetupParams::dense_only(
        30,
        64,
        [0; 32],
        Arc::clone(&artifacts),
    ));
    let large = one_hot(25, 51);
    let before = Prepared::grow([Ok(dense), Ok(one_hot(20, 30))], None).unwrap();
    let after = Prepared::grow([Ok(large.clone())], Some(&before)).unwrap();
    assert!(Arc::ptr_eq(
        before.material.expanded(),
        after.material.expanded()
    ));
    let required = AkitaVerifierRequirements::for_setups(std::slice::from_ref(&large))
        .unwrap()
        .prefix_slots;
    let verifier = after.catalog(&large).unwrap().one_hot_verifier().unwrap();
    with_one_hot_family!(verifier verifier, |verifier| {
        let slots = verifier.setup().prefix_slots();
        assert!(required.iter().all(|id| slots.get(id).is_some()));
    });
}
