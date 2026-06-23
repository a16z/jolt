use super::{outputs::Stage8OpeningStatement, verify::Stage8BatchEntry};
use crate::VerifierError;
use jolt_field::Field;
use jolt_openings::{
    BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, CommitmentLayoutDigest,
    EvaluationClaim, PhysicalView, VerifierOpeningClaim,
};
use jolt_poly::Point;
use jolt_transcript::Transcript;

type Stage8PrecommittedStatementBuild<F, C> = (
    Vec<VerifierOpeningClaim<F, C>>,
    Vec<Stage8OpeningStatement<F, C, F>>,
);

pub(super) fn verify_precommitted_opening_batches<PCS, T>(
    setup: &PCS::VerifierSetup,
    transcript: &mut T,
    statements: &[Stage8OpeningStatement<PCS::Field, PCS::Output, PCS::Field>],
    proofs: &[PCS::Proof],
) -> Result<Vec<PCS::Field>, VerifierError>
where
    PCS: BatchOpeningScheme,
    T: Transcript<Challenge = PCS::Field>,
{
    if statements.len() != proofs.len() {
        return Err(VerifierError::FinalOpeningVerificationFailed {
            reason: format!(
                "expected {} precommitted opening proofs, got {}",
                statements.len(),
                proofs.len()
            ),
        });
    }

    let mut coefficients = Vec::new();
    for (statement, proof) in statements.iter().zip(proofs) {
        let result = PCS::verify_batch(setup, transcript, statement, proof).map_err(|error| {
            VerifierError::FinalOpeningVerificationFailed {
                reason: error.to_string(),
            }
        })?;
        coefficients.extend(result.coefficients);
    }
    Ok(coefficients)
}

pub(super) fn precommitted_clear_statements<F, C>(
    entries: &[Stage8BatchEntry<'_, F, C>],
    default_layout_digest: [u8; 32],
) -> Result<Stage8PrecommittedStatementBuild<F, C>, VerifierError>
where
    F: Field,
    C: Clone + CommitmentLayoutDigest,
{
    let mut opening_claims = Vec::with_capacity(entries.len());
    let mut statements = Vec::with_capacity(entries.len());
    for entry in entries {
        let opening_claim =
            entry
                .opening_claim
                .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
                    reason: "missing clear opening claim in final batch".to_string(),
                })?;
        let own_point = Point::high_to_low(entry.own_point.clone());
        opening_claims.push(VerifierOpeningClaim {
            commitment: entry.commitment.clone(),
            evaluation: EvaluationClaim::new(own_point, opening_claim * entry.scale),
        });
        statements.push(BatchOpeningStatement {
            logical_point: entry.own_point.clone(),
            pcs_point: entry.own_point.clone(),
            layout_digest: direct_statement_layout_digest(entry.commitment, default_layout_digest),
            claims: vec![BatchOpeningClaim {
                id: entry.id,
                relation: entry.id,
                commitment: entry.commitment.clone(),
                claim: opening_claim,
                view: PhysicalView::Direct,
                scale: entry.scale,
            }],
        });
    }

    Ok((opening_claims, statements))
}

fn direct_statement_layout_digest<C: CommitmentLayoutDigest>(
    commitment: &C,
    default_layout_digest: [u8; 32],
) -> [u8; 32] {
    commitment.layout_digest().unwrap_or(default_layout_digest)
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "test setup should fail loudly when helper contracts change"
    )]

    use super::*;
    use crate::stages::stage8::outputs::{Stage8OpeningId, Stage8OpeningStatement};
    use jolt_claims::protocols::jolt::{
        formulas::committed_openings::final_opening_id, JoltCommittedPolynomial,
    };
    #[cfg(feature = "akita")]
    use jolt_claims::protocols::jolt::{JoltOpeningId, JoltRelationId};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_openings::{BatchOpeningClaim, BatchOpeningResult, CommitmentScheme, OpeningsError};
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
        let (_, statements) = precommitted_clear_statements(&[entry], default_digest)
            .expect("precommitted statement should build");

        assert_eq!(statements.len(), 1);
        assert_eq!(statements[0].logical_point, vec![Fr::from_u64(0)]);
        assert_eq!(statements[0].pcs_point, vec![Fr::from_u64(0)]);
        assert_eq!(statements[0].layout_digest, digest);
        assert_ne!(statements[0].layout_digest, default_digest);
    }

    #[cfg(feature = "akita")]
    #[test]
    fn precommitted_statements_use_entry_point_not_unified_point() {
        let commitment = jolt_akita::AkitaCommitment {
            layout_digest: [23; 32],
            num_vars: 1,
            poly_count: 1,
            native: vec![1],
        };
        let id = Stage8OpeningId::from(JoltOpeningId::committed(
            JoltCommittedPolynomial::ProgramImageInit,
            JoltRelationId::ProgramImageClaimReduction,
        ));
        let entry = Stage8BatchEntry {
            id,
            commitment: &commitment,
            opening_claim: Some(Fr::from_u64(7)),
            own_point: vec![Fr::from_u64(5)],
            scale: Fr::from_u64(3),
        };

        let (claims, statements) = precommitted_clear_statements(&[entry], [17; 32])
            .expect("precommitted statement should build");

        assert_eq!(claims[0].evaluation.point.as_slice(), &[Fr::from_u64(5)]);
        assert_eq!(claims[0].evaluation.value, Fr::from_u64(21));
        assert_eq!(statements[0].logical_point, vec![Fr::from_u64(5)]);
        assert_eq!(statements[0].pcs_point, vec![Fr::from_u64(5)]);
        assert_eq!(statements[0].claims[0].scale, Fr::from_u64(3));
    }
}
