
#![expect(
    clippy::expect_used,
    reason = "test setup should fail loudly when helper contracts change"
)]

use super::*;
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_openings::{BatchOpeningResult, OpeningsError};
use jolt_poly::{MultilinearPoly, Polynomial};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProofCheckingPcs;

impl jolt_crypto::Commitment for ProofCheckingPcs {
    type Output = u64;
}

impl CommitmentScheme for ProofCheckingPcs {
    type Field = Fr;
    type Proof = Fr;
    type ProverSetup = ();
    type VerifierSetup = ();
    type Polynomial = Polynomial<Fr>;
    type OpeningHint = ();
    type SetupParams = ();

    fn setup(_params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
        ((), ())
    }

    fn verifier_setup(_prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {}

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        _poly: &P,
        _setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        (0, ())
    }

    fn open(
        _poly: &Self::Polynomial,
        _point: &[Self::Field],
        eval: Self::Field,
        _setup: &Self::ProverSetup,
        _hint: Option<Self::OpeningHint>,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        eval
    }

    fn verify(
        _commitment: &Self::Output,
        _point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        if eval == *proof {
            Ok(())
        } else {
            Err(OpeningsError::VerificationFailed)
        }
    }

    fn bind_opening_inputs(
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
        _point: &[Self::Field],
        _eval: &Self::Field,
    ) {
    }
}

impl BatchOpeningScheme for ProofCheckingPcs {
    fn prove_batch<T, OpeningId, RelationId>(
        _setup: &Self::ProverSetup,
        _transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
        _polynomials: &[Self::Polynomial],
        _hints: Vec<Self::OpeningHint>,
    ) -> Result<Self::Proof, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        statement
            .claims
            .first()
            .map(|claim| claim.claim)
            .ok_or(OpeningsError::VerificationFailed)
    }

    fn verify_batch<T, OpeningId, RelationId>(
        _setup: &Self::VerifierSetup,
        _transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
        proof: &Self::Proof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        let claim = statement
            .claims
            .first()
            .ok_or(OpeningsError::VerificationFailed)?;
        if claim.claim != *proof {
            return Err(OpeningsError::VerificationFailed);
        }
        Ok(BatchOpeningResult {
            coefficients: vec![claim.scale],
            joint_commitment: claim.commitment,
            reduced_opening: claim.claim * claim.scale,
        })
    }
}

fn proof_checking_precommitted_statement(
    id: Stage8OpeningId,
    claim: Fr,
) -> Stage8OpeningStatement<Fr, u64, Fr> {
    BatchOpeningStatement {
        logical_point: vec![Fr::from_u64(0)],
        pcs_point: vec![Fr::from_u64(0)],
        layout_digest: [17; 32],
        claims: vec![BatchOpeningClaim {
            id,
            relation: id,
            commitment: 3,
            claim,
            view: PhysicalView::Direct,
            scale: Fr::from_u64(1),
        }],
    }
}

#[test]
fn precommitted_opening_batches_require_exact_ordered_proofs() {
    let statements = vec![
        proof_checking_precommitted_statement(
            final_opening_id(JoltCommittedPolynomial::BytecodeChunk(0)).into(),
            Fr::from_u64(3),
        ),
        proof_checking_precommitted_statement(
            final_opening_id(JoltCommittedPolynomial::ProgramImageInit).into(),
            Fr::from_u64(5),
        ),
    ];
    let proofs = vec![Fr::from_u64(3), Fr::from_u64(5)];
    let mut transcript = jolt_transcript::Blake2bTranscript::new(b"st8-precom-order");
    let coefficients = verify_precommitted_opening_batches::<ProofCheckingPcs, _>(
        &(),
        &mut transcript,
        &statements,
        &proofs,
    )
    .expect("ordered precommitted proofs should verify");
    assert_eq!(coefficients, vec![Fr::from_u64(1), Fr::from_u64(1)]);

    let mut transcript = jolt_transcript::Blake2bTranscript::new(b"st8-precom-extra");
    let mut extra_proofs = proofs.clone();
    extra_proofs.push(Fr::from_u64(8));
    assert!(matches!(
        verify_precommitted_opening_batches::<ProofCheckingPcs, _>(
            &(),
            &mut transcript,
            &statements,
            &extra_proofs,
        ),
        Err(VerifierError::FinalOpeningVerificationFailed { reason })
            if reason.contains("expected 2 precommitted opening proofs, got 3")
    ));

    let mut transcript = jolt_transcript::Blake2bTranscript::new(b"st8-precom-reorder");
    let reordered = vec![Fr::from_u64(5), Fr::from_u64(3)];
    assert!(matches!(
        verify_precommitted_opening_batches::<ProofCheckingPcs, _>(
            &(),
            &mut transcript,
            &statements,
            &reordered,
        ),
        Err(VerifierError::FinalOpeningVerificationFailed { .. })
    ));
}

#[cfg(feature = "akita")]
#[test]
fn precommitted_statements_use_lattice_commitment_layout_digest() {
    let digest = [23; 32];
    let default_digest = [17; 32];
    let commitment = jolt_akita::AkitaCommitment {
        layout_digest: digest,
        num_vars: 1,
        poly_count: 1,
        native: vec![1],
    };
    let id = Stage8OpeningId::from(JoltOpeningId::committed(
        JoltCommittedPolynomial::TrustedAdvice,
        JoltRelationId::AdviceClaimReduction,
    ));
    let entry = Stage8BatchEntry {
        id,
        commitment: &commitment,
        opening_claim: Some(Fr::from_u64(7)),
        own_point: vec![Fr::from_u64(0)],
        scale: Fr::from_u64(1),
    };
    let point = vec![Fr::from_u64(0)];
    let pcs_opening_point = Point::high_to_low(point.clone());
    let (_, statements) =
        precommitted_clear_statements(&[entry], default_digest, &point, &pcs_opening_point)
            .expect("precommitted statement should build");

    assert_eq!(statements.len(), 1);
    assert_eq!(statements[0].layout_digest, digest);
    assert_ne!(statements[0].layout_digest, default_digest);
}

#[test]
fn committed_program_batch_entries_require_final_openings() {
    let layout = JoltRaPolynomialLayout::new(1, 0, 0).expect("test RA layout should be valid");
    let opening_point = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
    let hamming_opening_point = vec![Fr::from_u64(1)];
    let inc_opening_point = vec![Fr::from_u64(1)];
    let commitment = 9_u64;

    let program_image_only = vec![PrecommittedFinalOpening {
        polynomial: JoltCommittedPolynomial::ProgramImageInit,
        point: vec![Fr::from_u64(3)],
        opening_claim: Some(Fr::from_u64(30)),
    }];
    let error = batch_entries(
        layout,
        Some(1),
        false,
        false,
        &opening_point,
        &hamming_opening_point,
        &inc_opening_point,
        &program_image_only,
        None,
        true,
        |_| Ok(&commitment),
        #[cfg(feature = "field-inline")]
        &commitment,
    )
    .err()
    .expect("missing bytecode chunk opening should fail");
    assert!(matches!(
        error,
        VerifierError::MissingOpeningClaim { id }
            if id == final_opening_id(JoltCommittedPolynomial::BytecodeChunk(0))
    ));

    let bytecode_only = vec![PrecommittedFinalOpening {
        polynomial: JoltCommittedPolynomial::BytecodeChunk(0),
        point: vec![Fr::from_u64(2)],
        opening_claim: Some(Fr::from_u64(20)),
    }];
    let error = batch_entries(
        layout,
        Some(1),
        false,
        false,
        &opening_point,
        &hamming_opening_point,
        &inc_opening_point,
        &bytecode_only,
        None,
        true,
        |_| Ok(&commitment),
        #[cfg(feature = "field-inline")]
        &commitment,
    )
    .err()
    .expect("missing program image opening should fail");
    assert!(matches!(
        error,
        VerifierError::MissingOpeningClaim { id }
            if id == final_opening_id(JoltCommittedPolynomial::ProgramImageInit)
    ));
}

#[test]
fn lattice_unsigned_increment_entries_require_sources_and_chunk_count() {
    let commitment = 9_u64;
    let opening_point = vec![Fr::from_u64(1), Fr::from_u64(2)];
    let error = lattice_unsigned_inc_batch_entries::<Fr, _>(8, &opening_point, None, &commitment)
        .err()
        .expect("lattice unsigned increment openings require stage sources");
    assert!(matches!(
        error,
        VerifierError::MissingOpeningClaim { id }
            if id == lattice_formulas::unsigned_inc_chunk_opening(0)
    ));

    let short_chunks = vec![Fr::from_u64(0); 7];
    let sources = LatticeUnsignedIncFinalOpenings {
        chunk_point: &opening_point,
        chunk_claims: Some(&short_chunks),
        msb_point: &opening_point,
        msb_claim: Some(Fr::from_u64(0)),
    };
    let error =
        lattice_unsigned_inc_batch_entries::<Fr, _>(8, &opening_point, Some(&sources), &commitment)
            .err()
            .expect("lattice unsigned increment openings require every lower chunk");
    assert!(matches!(
        error,
        VerifierError::FinalOpeningBatchFailed { reason }
            if reason.contains("unsigned increment final chunk opening count mismatch")
    ));
}

#[test]
fn lattice_unsigned_increment_entries_use_chunk_and_msb_points() {
    let commitment = 9_u64;
    let opening_point = vec![
        Fr::from_u64(1),
        Fr::from_u64(2),
        Fr::from_u64(3),
        Fr::from_u64(4),
        Fr::from_u64(5),
    ];
    let chunk_point = vec![Fr::from_u64(3), Fr::from_u64(4)];
    let msb_point = vec![Fr::from_u64(5)];
    let chunks = vec![Fr::from_u64(0); 8];
    let sources = LatticeUnsignedIncFinalOpenings {
        chunk_point: &chunk_point,
        chunk_claims: Some(&chunks),
        msb_point: &msb_point,
        msb_claim: Some(Fr::from_u64(1)),
    };

    let entries =
        lattice_unsigned_inc_batch_entries::<Fr, _>(8, &opening_point, Some(&sources), &commitment)
            .expect("complete lattice unsigned increment sources should produce entries");

    assert_eq!(entries.len(), 9);
    for (index, entry) in entries.iter().take(8).enumerate() {
        assert_eq!(
            entry.id,
            Stage8OpeningId::from(lattice_formulas::unsigned_inc_chunk_opening(index))
        );
        assert_eq!(entry.own_point, chunk_point);
    }
    let msb_entry = entries.last().expect("MSB entry should be last");
    assert_eq!(
        msb_entry.id,
        Stage8OpeningId::from(lattice_formulas::unsigned_inc_msb_opening())
    );
    assert_eq!(msb_entry.own_point, msb_point);
}
