
#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

use super::*;
use jolt_openings::BatchOpeningClaim;

#[derive(Default)]
struct RecordingTranscript {
    bytes: Vec<u8>,
}

impl Transcript for RecordingTranscript {
    type Challenge = AkitaField;

    fn new(label: &'static [u8]) -> Self {
        let mut transcript = Self::default();
        transcript.append_bytes(label);
        transcript
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    fn challenge(&mut self) -> Self::Challenge {
        AkitaField::zero()
    }

    fn state(&self) -> [u8; 32] {
        [0; 32]
    }
}

#[test]
fn setup_key_transcript_binds_native_shape_without_upstream_revision() {
    let setup = AkitaVerifierSetup {
        max_num_vars: 4,
        max_num_polys_per_commitment_group: 1,
        default_layout_digest: [7; 32],
        packed_layout: None,
        native: vec![1, 2, 3],
    };
    let mut transcript = RecordingTranscript::new(b"akita-setup-key-test");

    bind_verifier_setup_key(&setup, &mut transcript);

    assert!(contains_subslice(&transcript.bytes, b"akita/fp128/d64full"));
    assert!(contains_subslice(
        &transcript.bytes,
        b"akita_verifier_setup"
    ));
}

#[test]
fn direct_opening_requires_statement_commitment_layout_digest() {
    let setup_params = AkitaSetupParams::new(1, 1, [7; 32]);
    let (prover_setup, verifier_setup) = AkitaScheme::setup(setup_params);
    let polynomial = Polynomial::new(vec![AkitaField::from_u64(2), AkitaField::from_u64(5)]);
    let commitment_digest = [9; 32];
    let (commitment, hint) = AkitaScheme::commit_group(
        &prover_setup,
        commitment_digest,
        std::slice::from_ref(&polynomial),
    )
    .expect("direct commitment may use its own layout digest");
    assert_eq!(commitment.layout_digest, commitment_digest);

    let point = vec![AkitaField::from_u64(3)];
    let claim = polynomial.evaluate(&point);
    let statement = BatchOpeningStatement {
        logical_point: point.clone(),
        pcs_point: point,
        layout_digest: commitment_digest,
        claims: vec![BatchOpeningClaim {
            id: (),
            relation: (),
            commitment: commitment.clone(),
            claim,
            view: PhysicalView::Direct,
            scale: AkitaField::one(),
        }],
    };
    assert_eq!(statement.layout_digest, commitment.layout_digest);

    let mut prover_transcript = RecordingTranscript::new(b"akita-direct-layout");
    let proof = AkitaScheme::prove_batch(
        &prover_setup,
        &mut prover_transcript,
        &statement,
        &[polynomial],
        vec![hint],
    )
    .expect("direct proof should prove");
    assert!(contains_subslice(
        &prover_transcript.bytes,
        &commitment_digest
    ));

    let mut changed_wrapper_statement = statement.clone();
    changed_wrapper_statement.layout_digest = [13; 32];
    let mut verifier_transcript = RecordingTranscript::new(b"akita-direct-layout");
    let _error = AkitaScheme::verify_batch(
        &verifier_setup,
        &mut verifier_transcript,
        &changed_wrapper_statement,
        &proof,
    )
    .expect_err("changed direct statement digest should reject");

    let mut changed_commitment_statement = statement;
    changed_commitment_statement.claims[0]
        .commitment
        .layout_digest = [15; 32];
    let mut verifier_transcript = RecordingTranscript::new(b"akita-direct-layout");
    let _error = AkitaScheme::verify_batch(
        &verifier_setup,
        &mut verifier_transcript,
        &changed_commitment_statement,
        &proof,
    )
    .expect_err("changed direct commitment digest should reject");
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}
