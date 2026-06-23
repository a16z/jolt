use akita_pcs::{CommitmentProver, ComputeBackendSetup, CpuBackend};
use akita_prover::AkitaPolyOps;
use jolt_crypto::Commitment;
use jolt_openings::{
    BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme, OpeningsError, PhysicalView,
};
use jolt_poly::{MultilinearPoly, Polynomial};
use jolt_transcript::{AppendToTranscript, Label, Transcript};
use serde::{Deserialize, Serialize};

use crate::batch::prove_batch_with_native_polynomials;
use crate::native::{
    akita_error, dense_polynomials, invalid_batch, polynomial_evaluations, serialize_akita,
};
use crate::types::{
    append_field_slice, AkitaBatchProof, AkitaCommitment, AkitaField, AkitaProverHint,
    AkitaProverSetup, AkitaSetupParams, AkitaSparsePolynomial, AkitaVerifierSetup, NativeScheme,
    AKITA_D,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaScheme;

impl AkitaScheme {
    pub fn commit_group(
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
        polynomials: &[Polynomial<AkitaField>],
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let num_vars = validate_commit_polynomials(setup, polynomials)?;
        let dense = dense_polynomials(polynomials)?;
        let dense_refs = dense.iter().collect::<Vec<_>>();
        commit_native_group(
            setup,
            layout_digest,
            num_vars,
            polynomials.len(),
            dense_refs.as_slice(),
        )
    }

    pub fn commit_sparse_polynomial(
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
        polynomial: &AkitaSparsePolynomial,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        commit_native_group(
            setup,
            layout_digest,
            polynomial.num_vars(),
            1,
            &[&polynomial.native],
        )
    }

    pub fn prove_sparse_batch<T, OpeningId, RelationId>(
        setup: &AkitaProverSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
        polynomial: &AkitaSparsePolynomial,
        hint: AkitaProverHint,
    ) -> Result<AkitaBatchProof, OpeningsError>
    where
        T: Transcript<Challenge = AkitaField>,
    {
        prove_batch_with_native_polynomials(
            setup,
            transcript,
            statement,
            &[&polynomial.native],
            vec![hint],
        )
    }
}

fn commit_native_group<P>(
    setup: &AkitaProverSetup,
    layout_digest: [u8; 32],
    num_vars: usize,
    poly_count: usize,
    polynomials: &[P],
) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError>
where
    P: AkitaPolyOps<AkitaField, AKITA_D>,
{
    validate_native_commit_shape(setup, num_vars, poly_count)?;
    if polynomials.len() != poly_count {
        return Err(invalid_batch(format!(
            "Akita native commit received {} polynomials for {poly_count} commitment slots",
            polynomials.len()
        )));
    }
    for polynomial in polynomials {
        if polynomial.num_vars() != num_vars {
            return Err(invalid_batch(format!(
                "Akita native commit mixes {}-variable and {num_vars}-variable polynomials",
                polynomial.num_vars()
            )));
        }
    }

    let (native_commitment, native_hint) =
        NativeScheme::commit(&setup.native, &CpuBackend, &setup.prepared, polynomials)
            .map_err(akita_error)?;
    let commitment = AkitaCommitment {
        layout_digest,
        num_vars,
        poly_count,
        native: serialize_akita(&native_commitment)?,
    };
    Ok((
        commitment.clone(),
        AkitaProverHint {
            commitment,
            native: Some(native_hint),
        },
    ))
}

fn validate_native_commit_shape(
    setup: &AkitaProverSetup,
    num_vars: usize,
    poly_count: usize,
) -> Result<(), OpeningsError> {
    if num_vars > setup.max_num_vars {
        return Err(OpeningsError::PolynomialTooLarge {
            poly_size: num_vars,
            setup_max: setup.max_num_vars,
        });
    }
    if poly_count > setup.max_num_polys_per_commitment_group {
        return Err(invalid_batch(format!(
            "Akita commitment group has {poly_count} polynomials but setup supports {}",
            setup.max_num_polys_per_commitment_group
        )));
    }
    Ok(())
}

fn validate_commit_polynomials(
    setup: &AkitaProverSetup,
    polynomials: &[Polynomial<AkitaField>],
) -> Result<usize, OpeningsError> {
    let first = polynomials
        .first()
        .ok_or_else(|| invalid_batch("Akita commitment group must contain a polynomial"))?;
    let num_vars = first.num_vars();
    validate_native_commit_shape(setup, num_vars, polynomials.len())?;
    for polynomial in polynomials {
        if polynomial.num_vars() != num_vars {
            return Err(invalid_batch(format!(
                "Akita commitment group mixes {}-variable and {num_vars}-variable polynomials",
                polynomial.num_vars()
            )));
        }
    }
    Ok(num_vars)
}

impl Commitment for AkitaScheme {
    type Output = AkitaCommitment;
}

impl CommitmentScheme for AkitaScheme {
    type Field = AkitaField;
    type Proof = AkitaBatchProof;
    type ProverSetup = AkitaProverSetup;
    type VerifierSetup = AkitaVerifierSetup;
    type Polynomial = Polynomial<AkitaField>;
    type OpeningHint = AkitaProverHint;
    type SetupParams = AkitaSetupParams;

    #[expect(
        clippy::panic,
        reason = "CommitmentScheme::setup cannot return native Akita setup errors"
    )]
    fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
        let native = NativeScheme::setup_prover(
            params.max_num_vars,
            params.max_num_polys_per_commitment_group,
        )
        .unwrap_or_else(|err| panic!("Akita setup failed: {err}"));
        let prepared = CpuBackend
            .prepare_setup(&native)
            .unwrap_or_else(|err| panic!("Akita setup preparation failed: {err}"));
        let native_verifier = NativeScheme::setup_verifier(&native);
        let verifier = AkitaVerifierSetup {
            max_num_vars: params.max_num_vars,
            max_num_polys_per_commitment_group: params.max_num_polys_per_commitment_group,
            default_layout_digest: params.default_layout_digest,
            native: serialize_akita(&native_verifier)
                .unwrap_or_else(|err| panic!("Akita verifier setup serialization failed: {err}")),
        };
        let prover = AkitaProverSetup {
            max_num_vars: params.max_num_vars,
            max_num_polys_per_commitment_group: params.max_num_polys_per_commitment_group,
            default_layout_digest: params.default_layout_digest,
            native,
            prepared,
            verifier: verifier.clone(),
        };
        (prover, verifier)
    }

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        prover_setup.verifier.clone()
    }

    #[expect(
        clippy::panic,
        reason = "CommitmentScheme::commit cannot return native Akita commit errors"
    )]
    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        let polynomial = Polynomial::from(polynomial_evaluations(poly));
        Self::commit_group(setup, setup.default_layout_digest, &[polynomial])
            .unwrap_or_else(|err| panic!("Akita commit failed: {err}"))
    }

    #[expect(
        clippy::panic,
        reason = "CommitmentScheme::open cannot return native Akita prove errors"
    )]
    fn open(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        let hint = hint.unwrap_or_else(|| Self::commit(poly, setup).1);
        let statement = singleton_statement(hint.commitment.clone(), point, eval);
        <Self as BatchOpeningScheme>::prove_batch(
            setup,
            transcript,
            &statement,
            std::slice::from_ref(poly),
            vec![hint],
        )
        .unwrap_or_else(|err| panic!("Akita open failed: {err}"))
    }

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        let statement = singleton_statement(commitment.clone(), point, eval);
        <Self as BatchOpeningScheme>::verify_batch(setup, transcript, &statement, proof).map(|_| ())
    }

    fn bind_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        eval: &Self::Field,
    ) {
        transcript.append(&Label(b"akita_opening_inputs"));
        append_field_slice(transcript, b"akita_opening_point", point);
        eval.append_to_transcript(transcript);
    }
}

fn singleton_statement(
    commitment: AkitaCommitment,
    point: &[AkitaField],
    eval: AkitaField,
) -> BatchOpeningStatement<AkitaField, AkitaCommitment> {
    BatchOpeningStatement {
        logical_point: point.to_vec(),
        pcs_point: point.to_vec(),
        layout_digest: commitment.layout_digest,
        claims: vec![jolt_openings::BatchOpeningClaim {
            id: (),
            relation: (),
            commitment,
            claim: eval,
            view: PhysicalView::Direct,
            scale: AkitaField::one(),
        }],
    }
}

#[cfg(test)]
mod tests {

    #![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

    use super::*;
    use crate::transcript::bind_verifier_setup_key;
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
    fn setup_key_transcript_binds_native_shape() {
        let setup = AkitaVerifierSetup {
            max_num_vars: 4,
            max_num_polys_per_commitment_group: 1,
            default_layout_digest: [7; 32],
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
}
