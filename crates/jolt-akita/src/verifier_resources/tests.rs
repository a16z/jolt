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
use jolt_transcript::{protocol_tag, prover_transcript, Blake2b512};

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
            &mut prover_transcript::<Blake2b512>(&protocol_tag(b"verifier-resources"), b""),
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
            &mut prover_transcript::<Blake2b512>(&protocol_tag(b"verifier-resources"), b""),
        )
        .is_ok()
    }
}

/// Preprocessing: store the joint prefix commitments in each setup.
fn provisioned(mut setups: Vec<AkitaVerifierSetup>) -> Vec<AkitaVerifierSetup> {
    let known = stored_prefixes(&setups).unwrap();
    provision_prefix_commitments(&mut setups, &known, None).unwrap();
    setups
}

fn prepared(registry: &PreparedRegistry, setups: Vec<AkitaVerifierSetup>) -> Arc<Prepared> {
    let prepared = Arc::new(Prepared::new(provisioned(setups).into_iter().map(Ok)).unwrap());
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
    assert!(registry.states().is_empty());
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
    assert_eq!(registry.states().len(), 2);
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
            &mut prover_transcript::<Blake2b512>(&protocol_tag(b"warm-onehot"), b""),
        )
        .unwrap();
        AkitaScheme::verify(
            &commitment,
            &point,
            evaluation,
            &proof,
            &verifier,
            &mut prover_transcript::<Blake2b512>(&protocol_tag(b"warm-onehot"), b""),
        )
        .unwrap();
    }
}

/// A merged setup verifies each admitted shape through its own view, refuses
/// shapes it does not admit, and rejects a proof whose selection names another
/// admitted shape's row.
#[test]
fn merged_setup_verifies_each_admitted_shape_and_only_its_own_row() {
    let artifacts = params(14).schedule_artifacts;
    let request =
        |vars| AkitaSetupParams::one_hot_only(vars, 1, [5; 32], 16, Arc::clone(&artifacts));
    let setups = [14, 15].map(|vars| AkitaScheme::setup(request(vars)).unwrap());
    let verifiers = setups.iter().map(|(_, v)| v.clone()).collect::<Vec<_>>();
    let merged = crate::verifier_setup::merge_verifier_setups(&verifiers, None).unwrap();
    assert_eq!(merged.admitted_shapes.len(), 2);
    assert!(merged.for_shape(16, 1, [5; 32]).is_none());
    assert!(merged.for_shape(14, 1, [6; 32]).is_none());
    assert!(merged.for_shape(14, 2, [5; 32]).is_none());
    let (mut outside, _) = AkitaScheme::setup(request(16)).unwrap();
    assert!(outside.adopt_verifier_setup(&merged).is_err());
    merged.prepare().unwrap();

    let transcript = || prover_transcript::<Blake2b512>(&protocol_tag(b"merged-onehot"), b"");
    let proved = setups.map(|(mut prover, _)| {
        prover.adopt_verifier_setup(&merged).unwrap();
        let vars = prover.verifier.max_num_vars;
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
            &mut transcript(),
        )
        .unwrap();
        let view = merged.for_shape(vars, 1, [5; 32]).unwrap();
        assert_eq!(view.max_num_vars, vars);
        (commitment, point, evaluation, proof, view)
    });
    let verifies = |(commitment, point, evaluation, _, _): &(_, Vec<_>, _, _, _),
                    proof: &AkitaBatchProof,
                    view: &AkitaVerifierSetup| {
        AkitaScheme::verify(
            commitment,
            point,
            *evaluation,
            proof,
            view,
            &mut transcript(),
        )
        .is_ok()
    };
    let [first, second] = &proved;
    for (shape, other) in [(first, second), (second, first)] {
        assert!(verifies(shape, &shape.3, &shape.4));
        assert!(!verifies(shape, &shape.3, &other.4));
        let mut foreign_row = shape.3.clone();
        foreign_row.schedule_selection = other.3.schedule_selection;
        assert_ne!(foreign_row.schedule_selection, shape.3.schedule_selection);
        assert!(!verifies(shape, &foreign_row, &shape.4));
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
    assert!(requirements.prefix_field_elements > requirements.retained_field_elements);
    // Key load without stored commitments fails rather than computing them.
    assert!(Prepared::new([Ok(setup.clone())]).is_err());
    let state = Prepared::new(provisioned(vec![setup]).into_iter().map(Ok)).unwrap();
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
    let state = Prepared::new(
        provisioned(vec![one_hot.clone(), dense])
            .into_iter()
            .map(Ok),
    )
    .unwrap();
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

#[test]
fn dropping_every_handle_stops_new_attachments() {
    let registry = PreparedRegistry::new();
    let state = prepared(&registry, vec![metadata(&params(15))]);
    let attached = metadata(&params(14));
    registry.attach(&attached).unwrap();
    drop(state);
    assert!(registry.states().is_empty());
    let later = metadata(&params(14));
    registry.attach(&later).unwrap();
    assert!(!std::ptr::eq(
        attached.dense_verifier().unwrap(),
        later.dense_verifier().unwrap()
    ));
}

#[test]
fn the_newest_of_equal_states_is_attached() {
    let registry = PreparedRegistry::new();
    let _older = prepared(&registry, vec![metadata(&params(15))]);
    let newer = prepared(&registry, vec![metadata(&params(15))]);
    let setup = metadata(&params(14));
    registry.attach(&setup).unwrap();
    assert!(Arc::ptr_eq(
        setup.dense_verifier().unwrap().setup().expanded(),
        newer.material.expanded()
    ));
}

#[test]
fn setups_attach_up_to_the_state_capacity() {
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
    let registry = PreparedRegistry::new();
    let dense = metadata(&AkitaSetupParams::dense_only(
        30,
        1,
        [0; 32],
        Arc::clone(&artifacts),
    ));
    let state = prepared(&registry, vec![one_hot(20, 1), dense]);
    let wider = one_hot(25, 1);
    registry.attach(&wider).unwrap();
    assert!(std::ptr::eq(
        wider.one_hot_verifier().unwrap(),
        state.catalog(&wider).unwrap().one_hot_verifier().unwrap()
    ));
    let descriptor = state.material.expanded().descriptor();
    assert!(state
        .catalog(&one_hot(descriptor.max_num_vars + 1, 1))
        .is_none());
    assert!(state
        .catalog(&one_hot(20, descriptor.max_num_batched_polys + 1))
        .is_none());
}

#[test]
fn transported_setups_verify_a_proof_through_a_setup_prefix_fold() {
    // One polynomial at 27 variables selects a shipped one-hot row that offloads a
    // setup contribution; the assertion below checks it.
    let artifacts = params(14).schedule_artifacts;
    let request = AkitaSetupParams::one_hot_only(27, 1, [5; 32], 16, Arc::clone(&artifacts));
    let (prover, verifier) = AkitaScheme::setup(request).unwrap();
    let polynomial =
        OneHotPolynomial::new(16, (0..1 << 23).map(|i| Some((i % 16) as u8)).collect());
    let point = vec![AkitaField::from_u64(3); 27];
    let evaluation = polynomial.evaluate(&point);
    let (commitment, hint) = AkitaScheme::commit(&polynomial, &prover).unwrap();
    let proof = AkitaScheme::open(
        &polynomial,
        &point,
        evaluation,
        &prover,
        Some(hint),
        &mut prover_transcript::<Blake2b512>(&protocol_tag(b"prefix-fold"), b""),
    )
    .unwrap();
    drop((prover, polynomial));
    assert!(
        with_one_hot_family!(scheme verifier.one_hot_scheme().unwrap(), |scheme| {
            scheme
                .schedules()
                .resolve_selection(proof.selection())
                .unwrap()
                .schedule()
                .recursive_folds
                .iter()
                .any(|fold| fold.params.setup_prefix().is_some())
        })
    );
    let transported = || -> AkitaVerifierSetup {
        serde_json::from_str(&serde_json::to_string(&verifier).unwrap()).unwrap()
    };
    let verifies = |setup: &AkitaVerifierSetup, evaluation| {
        AkitaScheme::verify(
            &commitment,
            &point,
            evaluation,
            &proof,
            setup,
            &mut prover_transcript::<Blake2b512>(&protocol_tag(b"prefix-fold"), b""),
        )
        .is_ok()
    };
    // Verification never builds state: an unprepared transported setup is
    // rejected, and so is key load once its stored commitments are removed.
    let unprepared = transported();
    assert!(unprepared.prefix_commitments.is_some());
    assert!(!verifies(&unprepared, evaluation));
    let mut stripped = transported();
    stripped.prefix_commitments = None;
    assert!(PreparedRegistry::new().attach(&stripped).is_err());

    let registry = PreparedRegistry::new();
    let alone = transported();
    registry.attach(&alone).unwrap();
    assert!(verifies(&alone, evaluation));
    assert!(!verifies(&alone, evaluation + AkitaField::from_u64(1)));
    drop(alone);

    // A wider state: the attached verifier has more capacity than the proof's setup.
    let dense = metadata(&AkitaSetupParams::dense_only(30, 1, [0; 32], artifacts));
    let state = prepared(&registry, vec![transported(), dense]);
    assert!(state.material.expanded().descriptor().max_num_vars > 27);
    let attached = transported();
    registry.attach(&attached).unwrap();
    let expanded = with_one_hot_family!(verifier attached.one_hot_verifier().unwrap(), |verifier| {
        Arc::clone(verifier.setup().expanded())
    });
    assert!(Arc::ptr_eq(&expanded, state.material.expanded()));
    assert!(verifies(&attached, evaluation));
    assert!(!verifies(&attached, evaluation + AkitaField::from_u64(1)));
}
