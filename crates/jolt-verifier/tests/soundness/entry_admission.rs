//! Admission requires the key's entry to name a real instruction row. Bytecode
//! read-RAF pins cycle 0 to the entry row, so with a padding entry an
//! all-padding trace would verify under either panic claim. Each case re-keys
//! an accepted fixture with its entry moved to the padding row.

#![expect(clippy::unwrap_used, reason = "soundness fixtures fail loudly")]

use std::sync::Arc;

use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_verifier::{JoltVerifierPreprocessing, ProgramPreprocessing, VerifierError};

fn with_padding_entry<PCS, VC>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
) -> JoltVerifierPreprocessing<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    JoltVerifierPreprocessing<PCS, VC>: Clone,
{
    let JoltVerifierPreprocessing {
        mut program,
        pcs_setup,
        vc_setup,
        ..
    } = preprocessing.clone();
    match &mut program {
        ProgramPreprocessing::Full(full) => Arc::make_mut(full).bytecode.entry_address = 0,
        ProgramPreprocessing::Committed(committed) => committed.meta.entry_bytecode_index = 0,
    }
    JoltVerifierPreprocessing::new(program, pcs_setup, vc_setup).unwrap()
}

fn assert_rejects_the_entry(result: Result<(), VerifierError>) {
    assert!(
        matches!(result, Err(VerifierError::EntryNotInstruction { .. })),
        "{result:?}"
    );
}

#[cfg(not(any(feature = "zk", feature = "akita")))]
#[test]
fn padding_entries_are_rejected() {
    #[cfg(feature = "field-inline")]
    use crate::support::verifier_fixtures::standard_field_inline_eqpoly_case;
    use crate::support::verifier_fixtures::VerifierFixtureCase;
    #[cfg(not(feature = "field-inline"))]
    use crate::support::verifier_fixtures::{standard_committed_muldiv_case, standard_muldiv_case};

    let check = |mut case: VerifierFixtureCase| {
        case.preprocessing = with_padding_entry(&case.preprocessing);
        assert_rejects_the_entry(case.verify());
    };
    #[cfg(not(feature = "field-inline"))]
    {
        check(standard_muldiv_case());
        check(standard_committed_muldiv_case());
    }
    #[cfg(feature = "field-inline")]
    check(standard_field_inline_eqpoly_case());
}

#[cfg(all(feature = "zk", not(feature = "field-inline")))]
#[test]
fn zk_padding_entries_are_rejected() {
    use crate::support::verifier_fixtures::{
        zk_committed_muldiv_case, zk_muldiv_case, ZkVerifierFixtureCase,
    };

    let check = |mut case: ZkVerifierFixtureCase| {
        case.preprocessing = with_padding_entry(&case.preprocessing);
        assert_rejects_the_entry(case.verify());
    };
    check(zk_muldiv_case());
    check(zk_committed_muldiv_case());
}

#[cfg(feature = "akita")]
#[test]
fn akita_padding_entries_are_rejected() {
    use crate::support::akita_fixtures::AkitaFixtureCase;
    #[cfg(feature = "field-inline")]
    use crate::support::akita_fixtures::{
        akita_committed_field_inline_eqpoly_case, akita_field_inline_eqpoly_case,
    };
    #[cfg(not(feature = "field-inline"))]
    use crate::support::akita_fixtures::{akita_committed_muldiv_case, akita_muldiv_case};

    let check = |case: &AkitaFixtureCase| {
        let rekeyed = AkitaFixtureCase {
            preprocessing: with_padding_entry(&case.preprocessing),
            public_io: case.public_io.clone(),
            proof: case.proof.clone(),
            trusted_advice_commitment: case.trusted_advice_commitment.clone(),
        };
        assert_rejects_the_entry(rekeyed.verify());
    };
    #[cfg(not(feature = "field-inline"))]
    {
        check(akita_muldiv_case());
        check(akita_committed_muldiv_case());
    }
    #[cfg(feature = "field-inline")]
    {
        check(akita_field_inline_eqpoly_case());
        check(akita_committed_field_inline_eqpoly_case());
    }
}
