#![cfg(all(feature = "fs-audit", feature = "prover-fixtures"))]
#![expect(
    dead_code,
    reason = "the shared support module is compiled into every integration-test target but only partially used per feature configuration"
)]

mod support;

#[cfg(not(feature = "akita"))]
use support::fs_mutations::cancel_dory_final_opening_commitments;
#[cfg(not(feature = "zk"))]
use support::fs_mutations::equivocate_stage1_clear;
use support::fs_transcript::{record_challenges, replay_challenges, AuditTranscript};

#[cfg(not(feature = "akita"))]
use jolt_crypto::{Bn254G1, Pedersen};
#[cfg(not(feature = "akita"))]
use jolt_dory::DoryScheme;
#[cfg(not(feature = "akita"))]
use jolt_field::Fr;
#[cfg(all(not(feature = "akita"), not(feature = "zk")))]
use jolt_field::Ring;
#[cfg(not(feature = "akita"))]
use jolt_transcript::LegacyBlake2bTranscript;

#[cfg(all(not(feature = "akita"), not(feature = "zk")))]
#[test]
fn dory_clear_preprocessing_digest_requires_fiat_shamir_binding() {
    use support::verifier_fixtures::standard_muldiv_case;

    let case = standard_muldiv_case();
    let (honest, tape) = record_challenges::<Fr, _>(|| {
        jolt_verifier::verify::<
            Fr,
            DoryScheme,
            Pedersen<Bn254G1>,
            AuditTranscript<LegacyBlake2bTranscript<Fr>>,
        >(
            &case.preprocessing,
            &case.public_io,
            &case.proof,
            case.trusted_advice_commitment.as_ref(),
        )
    });
    assert!(
        honest.is_ok(),
        "honest Dory clear fixture rejected: {honest:?}"
    );

    let mut preprocessing = case.preprocessing.clone();
    preprocessing.preprocessing_digest[0] ^= 1;
    let frozen = replay_challenges(&tape, || {
        jolt_verifier::verify::<
            Fr,
            DoryScheme,
            Pedersen<Bn254G1>,
            AuditTranscript<LegacyBlake2bTranscript<Fr>>,
        >(
            &preprocessing,
            &case.public_io,
            &case.proof,
            case.trusted_advice_commitment.as_ref(),
        )
    });
    assert!(
        frozen.output.is_ok(),
        "frozen Dory clear verifier rejected the replay attack: {:?}",
        frozen.output
    );
    assert_eq!(frozen.consumed, frozen.expected);

    let (production, mutated_tape) = record_challenges::<Fr, _>(|| {
        jolt_verifier::verify::<
            Fr,
            DoryScheme,
            Pedersen<Bn254G1>,
            AuditTranscript<LegacyBlake2bTranscript<Fr>>,
        >(
            &preprocessing,
            &case.public_io,
            &case.proof,
            case.trusted_advice_commitment.as_ref(),
        )
    });
    assert!(
        production.is_err(),
        "production Dory clear verifier accepted a proof under different preprocessing"
    );
    assert!(
        tape.first_value_divergence(&mutated_tape).is_some(),
        "preprocessing mutation did not alter a production challenge"
    );
}

#[cfg(all(not(feature = "akita"), not(feature = "zk")))]
#[test]
fn dory_clear_stage1_sumcheck_requires_fiat_shamir_challenges() {
    use support::verifier_fixtures::standard_muldiv_case;

    let case = standard_muldiv_case();
    let (honest, tape) = record_challenges::<Fr, _>(|| {
        jolt_verifier::verify::<
            Fr,
            DoryScheme,
            Pedersen<Bn254G1>,
            AuditTranscript<LegacyBlake2bTranscript<Fr>>,
        >(
            &case.preprocessing,
            &case.public_io,
            &case.proof,
            case.trusted_advice_commitment.as_ref(),
        )
    });
    assert!(
        honest.is_ok(),
        "honest Dory clear fixture rejected: {honest:?}"
    );

    let mut proof = case.proof.clone();
    equivocate_stage1_clear(&mut proof, &tape, Fr::from_u64(1));
    let frozen = replay_challenges(&tape, || {
        jolt_verifier::verify::<
            Fr,
            DoryScheme,
            Pedersen<Bn254G1>,
            AuditTranscript<LegacyBlake2bTranscript<Fr>>,
        >(
            &case.preprocessing,
            &case.public_io,
            &proof,
            case.trusted_advice_commitment.as_ref(),
        )
    });
    assert!(
        frozen.output.is_ok(),
        "frozen Dory clear verifier rejected stage-1 equivocation: {:?}",
        frozen.output
    );
    assert_eq!(frozen.consumed, frozen.expected);

    let (production, mutated_tape) = record_challenges::<Fr, _>(|| {
        jolt_verifier::verify::<
            Fr,
            DoryScheme,
            Pedersen<Bn254G1>,
            AuditTranscript<LegacyBlake2bTranscript<Fr>>,
        >(
            &case.preprocessing,
            &case.public_io,
            &proof,
            case.trusted_advice_commitment.as_ref(),
        )
    });
    assert!(
        production.is_err(),
        "production Dory clear verifier accepted stage-1 equivocation"
    );
    assert!(
        tape.first_value_divergence(&mutated_tape).is_some(),
        "stage-1 equivocation did not alter a production challenge"
    );
}

#[cfg(all(not(feature = "akita"), not(feature = "zk")))]
#[test]
fn dory_clear_final_opening_batch_requires_commitment_binding() {
    use support::verifier_fixtures::standard_muldiv_case;

    let case = standard_muldiv_case();
    let (honest, tape) = record_challenges::<Fr, _>(|| {
        jolt_verifier::verify::<
            Fr,
            DoryScheme,
            Pedersen<Bn254G1>,
            AuditTranscript<LegacyBlake2bTranscript<Fr>>,
        >(
            &case.preprocessing,
            &case.public_io,
            &case.proof,
            case.trusted_advice_commitment.as_ref(),
        )
    });
    assert!(
        honest.is_ok(),
        "honest Dory clear fixture rejected: {honest:?}"
    );

    let mut proof = case.proof.clone();
    cancel_dory_final_opening_commitments(&mut proof, &tape);
    let frozen = replay_challenges(&tape, || {
        jolt_verifier::verify::<
            Fr,
            DoryScheme,
            Pedersen<Bn254G1>,
            AuditTranscript<LegacyBlake2bTranscript<Fr>>,
        >(
            &case.preprocessing,
            &case.public_io,
            &proof,
            case.trusted_advice_commitment.as_ref(),
        )
    });
    assert!(
        frozen.output.is_ok(),
        "frozen Dory clear verifier rejected opening-batch cancellation: {:?}",
        frozen.output
    );
    assert_eq!(frozen.consumed, frozen.expected);

    let (production, mutated_tape) = record_challenges::<Fr, _>(|| {
        jolt_verifier::verify::<
            Fr,
            DoryScheme,
            Pedersen<Bn254G1>,
            AuditTranscript<LegacyBlake2bTranscript<Fr>>,
        >(
            &case.preprocessing,
            &case.public_io,
            &proof,
            case.trusted_advice_commitment.as_ref(),
        )
    });
    assert!(
        production.is_err(),
        "production Dory clear verifier accepted false individual openings"
    );
    assert!(
        tape.first_value_divergence(&mutated_tape).is_some(),
        "commitment cancellation did not alter a production challenge"
    );
}

#[cfg(all(not(feature = "akita"), feature = "zk"))]
#[test]
fn dory_zk_preprocessing_digest_requires_fiat_shamir_binding() {
    use support::verifier_fixtures::zk_muldiv_case;

    let case = zk_muldiv_case();
    let (honest, tape) = record_challenges::<Fr, _>(|| {
        jolt_verifier::verify::<
            Fr,
            DoryScheme,
            Pedersen<Bn254G1>,
            AuditTranscript<LegacyBlake2bTranscript<Fr>>,
        >(&case.preprocessing, &case.public_io, &case.proof, None)
    });
    assert!(
        honest.is_ok(),
        "honest Dory ZK fixture rejected: {honest:?}"
    );

    let mut preprocessing = case.preprocessing.clone();
    preprocessing.preprocessing_digest[0] ^= 1;
    let frozen = replay_challenges(&tape, || {
        jolt_verifier::verify::<
            Fr,
            DoryScheme,
            Pedersen<Bn254G1>,
            AuditTranscript<LegacyBlake2bTranscript<Fr>>,
        >(&preprocessing, &case.public_io, &case.proof, None)
    });
    assert!(
        frozen.output.is_ok(),
        "frozen Dory ZK verifier rejected the replay attack: {:?}",
        frozen.output
    );
    assert_eq!(frozen.consumed, frozen.expected);

    let (production, mutated_tape) = record_challenges::<Fr, _>(|| {
        jolt_verifier::verify::<
            Fr,
            DoryScheme,
            Pedersen<Bn254G1>,
            AuditTranscript<LegacyBlake2bTranscript<Fr>>,
        >(&preprocessing, &case.public_io, &case.proof, None)
    });
    assert!(
        production.is_err(),
        "production Dory ZK verifier accepted a proof under different preprocessing"
    );
    assert!(
        tape.first_value_divergence(&mutated_tape).is_some(),
        "preprocessing mutation did not alter a production challenge"
    );
}

#[cfg(all(not(feature = "akita"), feature = "zk"))]
#[test]
fn dory_zk_final_opening_batch_requires_commitment_binding() {
    use support::verifier_fixtures::zk_muldiv_case;

    let case = zk_muldiv_case();
    let (honest, tape) = record_challenges::<Fr, _>(|| {
        jolt_verifier::verify::<
            Fr,
            DoryScheme,
            Pedersen<Bn254G1>,
            AuditTranscript<LegacyBlake2bTranscript<Fr>>,
        >(&case.preprocessing, &case.public_io, &case.proof, None)
    });
    assert!(
        honest.is_ok(),
        "honest Dory ZK fixture rejected: {honest:?}"
    );

    let mut proof = case.proof.clone();
    cancel_dory_final_opening_commitments(&mut proof, &tape);
    let frozen = replay_challenges(&tape, || {
        jolt_verifier::verify::<
            Fr,
            DoryScheme,
            Pedersen<Bn254G1>,
            AuditTranscript<LegacyBlake2bTranscript<Fr>>,
        >(&case.preprocessing, &case.public_io, &proof, None)
    });
    assert!(
        frozen.output.is_ok(),
        "frozen Dory ZK verifier rejected opening-batch cancellation: {:?}",
        frozen.output
    );
    assert_eq!(frozen.consumed, frozen.expected);

    let (production, mutated_tape) = record_challenges::<Fr, _>(|| {
        jolt_verifier::verify::<
            Fr,
            DoryScheme,
            Pedersen<Bn254G1>,
            AuditTranscript<LegacyBlake2bTranscript<Fr>>,
        >(&case.preprocessing, &case.public_io, &proof, None)
    });
    assert!(
        production.is_err(),
        "production Dory ZK verifier accepted false individual openings"
    );
    assert!(
        tape.first_value_divergence(&mutated_tape).is_some(),
        "commitment cancellation did not alter a production challenge"
    );
}

#[cfg(feature = "akita")]
#[test]
fn akita_clear_preprocessing_digest_requires_fiat_shamir_binding() {
    use jolt_prover_legacy::zkvm::packed::{AkitaField, AkitaScheme, AkitaTranscript, AkitaVc};
    use support::akita_fixtures::akita_muldiv_case;

    let case = akita_muldiv_case();
    let (honest, tape) = record_challenges::<AkitaField, _>(|| {
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AuditTranscript<AkitaTranscript>>(
            &case.preprocessing,
            &case.public_io,
            &case.proof,
            case.trusted_advice_commitment.as_ref(),
        )
    });
    assert!(honest.is_ok(), "honest Akita fixture rejected: {honest:?}");

    let mut preprocessing = case.preprocessing.clone();
    preprocessing.preprocessing_digest[0] ^= 1;
    let frozen = replay_challenges(&tape, || {
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AuditTranscript<AkitaTranscript>>(
            &preprocessing,
            &case.public_io,
            &case.proof,
            case.trusted_advice_commitment.as_ref(),
        )
    });
    assert!(
        frozen.output.is_ok(),
        "frozen Akita verifier rejected the replay attack: {:?}",
        frozen.output
    );
    assert_eq!(frozen.consumed, frozen.expected);

    let (production, mutated_tape) = record_challenges::<AkitaField, _>(|| {
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AuditTranscript<AkitaTranscript>>(
            &preprocessing,
            &case.public_io,
            &case.proof,
            case.trusted_advice_commitment.as_ref(),
        )
    });
    assert!(
        production.is_err(),
        "production Akita verifier accepted a proof under different preprocessing"
    );
    assert!(
        tape.first_value_divergence(&mutated_tape).is_some(),
        "preprocessing mutation did not alter a production challenge"
    );
}

#[cfg(feature = "akita")]
#[test]
fn akita_clear_stage1_sumcheck_requires_fiat_shamir_challenges() {
    use jolt_field::Ring;
    use jolt_prover_legacy::zkvm::packed::{AkitaField, AkitaScheme, AkitaTranscript, AkitaVc};
    use support::akita_fixtures::akita_muldiv_case;

    let case = akita_muldiv_case();
    let (honest, tape) = record_challenges::<AkitaField, _>(|| {
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AuditTranscript<AkitaTranscript>>(
            &case.preprocessing,
            &case.public_io,
            &case.proof,
            case.trusted_advice_commitment.as_ref(),
        )
    });
    assert!(honest.is_ok(), "honest Akita fixture rejected: {honest:?}");

    let mut proof = case.proof.clone();
    equivocate_stage1_clear(&mut proof, &tape, AkitaField::from_u64(1));
    let frozen = replay_challenges(&tape, || {
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AuditTranscript<AkitaTranscript>>(
            &case.preprocessing,
            &case.public_io,
            &proof,
            case.trusted_advice_commitment.as_ref(),
        )
    });
    assert!(
        frozen.output.is_ok(),
        "frozen Akita verifier rejected stage-1 equivocation: {:?}",
        frozen.output
    );
    assert_eq!(frozen.consumed, frozen.expected);

    let (production, mutated_tape) = record_challenges::<AkitaField, _>(|| {
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AuditTranscript<AkitaTranscript>>(
            &case.preprocessing,
            &case.public_io,
            &proof,
            case.trusted_advice_commitment.as_ref(),
        )
    });
    assert!(
        production.is_err(),
        "production Akita verifier accepted stage-1 equivocation"
    );
    assert!(
        tape.first_value_divergence(&mutated_tape).is_some(),
        "stage-1 equivocation did not alter a production challenge"
    );
}
