//! Deep fold-schedule coverage. The rest of the suite stays at the
//! 13/14-variable planner floor where schedules carry one to three recursive
//! folds; this exercises a deeper recursion (17 variables, four recursive
//! folds) end to end, plus the `valid_proof || garbage` rejection Akita's
//! argument parser must enforce.

#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

#[expect(
    dead_code,
    reason = "shared integration-test support is compiled independently per test file"
)]
mod support;

use akita_types::{PolynomialGroupLayout, ScheduleLookupKey};
use jolt_akita::{AkitaCommitment, AkitaField, AkitaScheduleArtifacts, AkitaScheme};
use jolt_openings::{CommitmentScheme, OpeningsError};
use jolt_transcript::TranscriptError;
use support::{
    assert_transcripts_agree, f, layout, new_prover_transcript, new_verifier_transcript,
    polynomial, setup_for,
};

struct ProofFixture {
    verifier_setup: <AkitaScheme as CommitmentScheme>::VerifierSetup,
    commitment: AkitaCommitment,
    point: Vec<AkitaField>,
    eval: AkitaField,
    proof: Vec<u8>,
    label: &'static [u8],
}

impl ProofFixture {
    /// Verifies `proof` as the whole argument string of a standalone opening.
    fn verify(&self, proof: &[u8]) -> Result<(), OpeningsError> {
        let mut transcript = new_verifier_transcript(self.label, proof);
        AkitaScheme::verify(
            &self.commitment,
            &self.point,
            self.eval,
            &self.verifier_setup,
            &mut transcript,
        )?;
        Ok(transcript.finish()?)
    }
}

fn fold_roundtrip(num_vars: usize, label: &'static [u8]) -> ProofFixture {
    let (prover_setup, verifier_setup) = setup_for(num_vars, 1, layout(7));
    let poly = polynomial(num_vars, 5);
    let point: Vec<_> = (0..num_vars).map(|index| f(index as u64 + 2)).collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) =
        AkitaScheme::commit(&poly, &prover_setup).expect("dense commit should succeed");

    let mut prover_transcript = new_prover_transcript(label);
    AkitaScheme::open(
        &poly,
        &point,
        eval,
        &prover_setup,
        Some(hint),
        &mut prover_transcript,
    )
    .expect("fold-schedule proof should be produced");
    let proof = prover_transcript.narg().to_vec();

    let mut verifier_transcript = new_verifier_transcript(label, &proof);
    AkitaScheme::verify(
        &commitment,
        &point,
        eval,
        &verifier_setup,
        &mut verifier_transcript,
    )
    .expect("fold-schedule proof should verify");
    assert_transcripts_agree(prover_transcript, verifier_transcript);

    ProofFixture {
        verifier_setup,
        commitment,
        point,
        eval,
        proof,
        label,
    }
}

/// 17 variables resolve to four recursive fold levels — deeper than any
/// other single-polynomial suite fixture — and a tampered evaluation must
/// still reject. The depth is asserted so a catalog regeneration cannot
/// quietly shrink the fixture.
#[test]
fn deep_recursive_fold_schedule_roundtrips() {
    const NUM_VARS: usize = 17;
    let depth = AkitaScheduleArtifacts::shared_from_default_directory()
        .dense_catalog()
        .expect("dense catalog")
        .resolve_key(&ScheduleLookupKey::single(PolynomialGroupLayout::new(
            NUM_VARS, 1,
        )))
        .expect("deep fixture row must resolve")
        .schedule()
        .recursive_folds
        .len();
    assert_eq!(depth, 4, "the deep fixture must keep four recursive folds");
    let fixture = fold_roundtrip(NUM_VARS, b"akita-fold-deep");

    let mut tampered_eval = fixture.eval;
    tampered_eval += f(1);
    let mut transcript = new_verifier_transcript(fixture.label, &fixture.proof);
    assert!(
        AkitaScheme::verify(
            &fixture.commitment,
            &fixture.point,
            tampered_eval,
            &fixture.verifier_setup,
            &mut transcript,
        )
        .is_err(),
        "tampered evaluation must reject on the deep fold path too"
    );
}

/// The opening must consume exactly one complete argument string.
#[test]
fn proof_payloads_with_trailing_or_missing_bytes_reject() {
    let fixture = fold_roundtrip(14, b"akita-fold-trailing");

    let mut extended = fixture.proof.clone();
    extended.push(0);
    assert_eq!(
        fixture.verify(&extended),
        Err(OpeningsError::Transcript(TranscriptError::TrailingBytes)),
        "trailing proof bytes must be rejected"
    );

    let proof_len = fixture.proof.len();
    for length in [0, 1, proof_len / 2, proof_len - 1] {
        assert!(
            fixture.verify(&fixture.proof[..length]).is_err(),
            "proof truncated to {length} bytes must reject"
        );
    }
}
