use akita_pcs::{
    AkitaTranscript, CommitmentProver, CpuBackend, ProverCommitmentGroup, ProverOpeningBatch,
    RootPolyShape,
};
use akita_types::{
    BasisMode, CommitmentGroup, CommitmentVerifier, PointVariableSelection, SetupContributionMode,
    VerifierOpeningBatch,
};
use jolt_openings::{BatchOpeningScheme, OpeningsError, VerifierOpeningClaim};
use jolt_poly::MultilinearPoly;
use jolt_transcript::Transcript;

use crate::adapters::{
    akita_error, bridge_jolt_statement_challenge, deserialize_akita, invalid_batch,
    jolt_to_akita_evals, polynomial_evaluations, serialize_akita, AkitaBackendCommitment,
    AkitaBackendDensePoly, AkitaBackendProof, AkitaBackendProofShape, AkitaBackendScheme,
    AkitaBackendVerifier, AkitaBatchProof, AkitaBlackBoxBatchStatementTranscript, AkitaCommitment,
    AkitaField, AkitaProverHint, AkitaProverSetup, AkitaSourceKind, AkitaSparsePolynomial,
    AkitaVerifierSetup,
};

/// Adapts Akita's backend same-point batched opening API to Jolt's batch trait.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AkitaBlackBoxBatching;

pub type AkitaBlackBoxBatchStatement = Vec<VerifierOpeningClaim<AkitaField, AkitaCommitment>>;

pub type AkitaBlackBoxBatchWitness<'a> = (
    Vec<&'a (dyn MultilinearPoly<AkitaField> + 'a)>,
    AkitaProverHint,
);

impl BatchOpeningScheme for AkitaBlackBoxBatching {
    type Field = AkitaField;
    type ProverSetup = AkitaProverSetup;
    type VerifierSetup = AkitaVerifierSetup;
    type Statement = AkitaBlackBoxBatchStatement;
    type BatchingWitness<'a>
        = AkitaBlackBoxBatchWitness<'a>
    where
        Self: 'a;
    type Proof = AkitaBatchProof;

    fn prove_batch<'a, T>(
        setup: &Self::ProverSetup,
        statement: Self::Statement,
        witness: Self::BatchingWitness<'a>,
        transcript: &mut T,
    ) -> Result<Self::Proof, OpeningsError>
    where
        Self: 'a,
        T: Transcript<Challenge = Self::Field>,
    {
        let (polynomials, hint) = witness;
        let first = statement
            .first()
            .ok_or_else(|| invalid_batch("Akita black-box batching requires at least one claim"))?;
        let commitment = first.commitment.clone();
        let point = first.evaluation.point.as_slice().to_vec();

        // Validate statement, setup, and witness shape before using Akita.
        if point.len() != commitment.num_vars {
            return Err(invalid_batch(format!(
                "Akita opening point has {} variables but commitment has {}",
                point.len(),
                commitment.num_vars
            )));
        }
        for claim in &statement {
            if claim.commitment != commitment {
                return Err(invalid_batch(
                    "Akita batch statement must use exactly one commitment group",
                ));
            }
            if claim.evaluation.point.as_slice() != point.as_slice() {
                return Err(invalid_batch(
                    "Akita black-box batching claims must use one common point",
                ));
            }
        }
        if commitment.poly_count != statement.len() {
            return Err(invalid_batch(format!(
                "Akita commitment covers {} polynomials but statement has {} claims",
                commitment.poly_count,
                statement.len()
            )));
        }
        if commitment.num_vars > setup.max_num_vars {
            return Err(OpeningsError::PolynomialTooLarge {
                poly_size: commitment.num_vars,
                setup_max: setup.max_num_vars,
            });
        }
        if commitment.num_vars != setup.max_num_vars {
            return Err(invalid_batch(format!(
                "Akita commitment dimension {} does not match exact setup dimension {}",
                commitment.num_vars, setup.max_num_vars
            )));
        }
        if commitment.poly_count > setup.max_num_polys_per_commitment_group {
            return Err(invalid_batch(format!(
                "Akita commitment covers {} polynomials but setup supports {}",
                commitment.poly_count, setup.max_num_polys_per_commitment_group
            )));
        }
        if hint.commitment != commitment {
            return Err(invalid_batch(
                "Akita prover hint does not match the statement commitment",
            ));
        }

        match hint.source_kind {
            AkitaSourceKind::Dense => {
                let dense = polynomials
                    .iter()
                    .map(|polynomial| {
                        let evals = polynomial_evaluations(*polynomial);
                        let akita_evals = jolt_to_akita_evals(polynomial.num_vars(), &evals)?;
                        AkitaBackendDensePoly::from_field_evals(polynomial.num_vars(), &akita_evals)
                            .map_err(akita_error)
                    })
                    .collect::<Result<Vec<_>, OpeningsError>>()?;
                if dense.len() != commitment.poly_count {
                    return Err(invalid_batch(format!(
                        "Akita prover received {} polynomials for {} commitment slots",
                        dense.len(),
                        commitment.poly_count
                    )));
                }
                for polynomial in &dense {
                    if polynomial.num_vars() != commitment.num_vars {
                        return Err(invalid_batch(format!(
                            "Akita polynomial has {} variables but commitment has {}",
                            polynomial.num_vars(),
                            commitment.num_vars
                        )));
                    }
                }

                transcript.append(&setup.verifier);
                transcript.append(&AkitaBlackBoxBatchStatementTranscript::new(
                    &statement,
                    &commitment,
                    point.as_slice(),
                ));
                let mut akita_transcript = AkitaTranscript::<AkitaField>::new(b"jolt-akita/batch");
                let statement_bridge =
                    bridge_jolt_statement_challenge(transcript, &mut akita_transcript);

                let backend_commitment = deserialize_akita::<AkitaBackendCommitment>(
                    &commitment.serialized_backend_bytes,
                    &(),
                )?;
                let backend_hint = hint.backend_hint.ok_or_else(|| {
                    invalid_batch("Akita prover hint is missing backend opening data")
                })?;
                let poly_refs = dense.iter().collect::<Vec<_>>();
                let claims = ProverOpeningBatch {
                    point: point.as_slice().into(),
                    groups: vec![ProverCommitmentGroup {
                        point_vars: PointVariableSelection::prefix(point.len(), point.len())
                            .map_err(akita_error)?,
                        polynomials: poly_refs.as_slice(),
                        commitment: (backend_commitment, backend_hint),
                    }],
                };
                let stack = akita_prover::UniformProverStack::uniform(
                    &CpuBackend,
                    &setup.prepared_backend_setup,
                    setup.backend_prover_setup.expanded.as_ref(),
                )
                .map_err(akita_error)?;

                let backend_proof = AkitaBackendScheme::batched_prove(
                    &setup.backend_prover_setup,
                    claims,
                    &stack,
                    &mut akita_transcript,
                    BasisMode::Lagrange,
                    SetupContributionMode::Direct,
                )
                .map_err(akita_error)?;
                let proof_shape = backend_proof.shape();
                let proof = AkitaBatchProof {
                    commitment,
                    statement_bridge,
                    serialized_akita_proof_shape: serialize_akita(&proof_shape)?,
                    serialized_akita_proof: serialize_akita(&backend_proof)?,
                };
                transcript.append(&proof);
                Ok(proof)
            }
            AkitaSourceKind::SparseUnit => {
                let mut sparse = Vec::with_capacity(polynomials.len());
                for polynomial in polynomials {
                    if !polynomial.is_one_hot() {
                        return Err(invalid_batch(
                            "Akita sparse prover hint requires one-hot witness polynomials",
                        ));
                    }
                    let mut indices = Vec::new();
                    polynomial.for_each_one(&mut |index| indices.push(index));
                    sparse.push(AkitaSparsePolynomial::from_jolt_unit_indices(
                        polynomial.num_vars(),
                        indices,
                    )?);
                }
                let backend_polynomials = sparse
                    .into_iter()
                    .map(|polynomial| polynomial.backend_polynomial)
                    .collect::<Vec<_>>();
                if backend_polynomials.len() != commitment.poly_count {
                    return Err(invalid_batch(format!(
                        "Akita prover received {} polynomials for {} commitment slots",
                        backend_polynomials.len(),
                        commitment.poly_count
                    )));
                }
                for polynomial in &backend_polynomials {
                    if polynomial.num_vars() != commitment.num_vars {
                        return Err(invalid_batch(format!(
                            "Akita polynomial has {} variables but commitment has {}",
                            polynomial.num_vars(),
                            commitment.num_vars
                        )));
                    }
                }

                transcript.append(&setup.verifier);
                transcript.append(&AkitaBlackBoxBatchStatementTranscript::new(
                    &statement,
                    &commitment,
                    point.as_slice(),
                ));
                let mut akita_transcript = AkitaTranscript::<AkitaField>::new(b"jolt-akita/batch");
                let statement_bridge =
                    bridge_jolt_statement_challenge(transcript, &mut akita_transcript);

                let backend_commitment = deserialize_akita::<AkitaBackendCommitment>(
                    &commitment.serialized_backend_bytes,
                    &(),
                )?;
                let backend_hint = hint.backend_hint.ok_or_else(|| {
                    invalid_batch("Akita prover hint is missing backend opening data")
                })?;
                let poly_refs = backend_polynomials.iter().collect::<Vec<_>>();
                let claims = ProverOpeningBatch {
                    point: point.as_slice().into(),
                    groups: vec![ProverCommitmentGroup {
                        point_vars: PointVariableSelection::prefix(point.len(), point.len())
                            .map_err(akita_error)?,
                        polynomials: poly_refs.as_slice(),
                        commitment: (backend_commitment, backend_hint),
                    }],
                };
                let stack = akita_prover::UniformProverStack::uniform(
                    &CpuBackend,
                    &setup.prepared_backend_setup,
                    setup.backend_prover_setup.expanded.as_ref(),
                )
                .map_err(akita_error)?;

                let backend_proof = AkitaBackendScheme::batched_prove(
                    &setup.backend_prover_setup,
                    claims,
                    &stack,
                    &mut akita_transcript,
                    BasisMode::Lagrange,
                    SetupContributionMode::Direct,
                )
                .map_err(akita_error)?;
                let proof_shape = backend_proof.shape();
                let proof = AkitaBatchProof {
                    commitment,
                    statement_bridge,
                    serialized_akita_proof_shape: serialize_akita(&proof_shape)?,
                    serialized_akita_proof: serialize_akita(&backend_proof)?,
                };
                transcript.append(&proof);
                Ok(proof)
            }
        }
    }

    fn verify_batch<T>(
        setup: &Self::VerifierSetup,
        statement: Self::Statement,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<(), OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        let first = statement
            .first()
            .ok_or_else(|| invalid_batch("Akita black-box batching requires at least one claim"))?;
        let commitment = first.commitment.clone();
        let point = first.evaluation.point.as_slice().to_vec();

        // Validate statement and setup shape before using Akita.
        if point.len() != commitment.num_vars {
            return Err(invalid_batch(format!(
                "Akita opening point has {} variables but commitment has {}",
                point.len(),
                commitment.num_vars
            )));
        }
        let mut openings = Vec::with_capacity(statement.len());
        for claim in &statement {
            if claim.commitment != commitment {
                return Err(invalid_batch(
                    "Akita batch statement must use exactly one commitment group",
                ));
            }
            if claim.evaluation.point.as_slice() != point.as_slice() {
                return Err(invalid_batch(
                    "Akita black-box batching claims must use one common point",
                ));
            }
            openings.push(claim.evaluation.value);
        }
        if commitment.poly_count != statement.len() {
            return Err(invalid_batch(format!(
                "Akita commitment covers {} polynomials but statement has {} claims",
                commitment.poly_count,
                statement.len()
            )));
        }
        if proof.commitment != commitment {
            return Err(OpeningsError::VerificationFailed);
        }
        if commitment.num_vars > setup.max_num_vars {
            return Err(OpeningsError::PolynomialTooLarge {
                poly_size: commitment.num_vars,
                setup_max: setup.max_num_vars,
            });
        }
        if commitment.num_vars != setup.max_num_vars {
            return Err(invalid_batch(format!(
                "Akita commitment dimension {} does not match exact setup dimension {}",
                commitment.num_vars, setup.max_num_vars
            )));
        }
        if commitment.poly_count > setup.max_num_polys_per_commitment_group {
            return Err(invalid_batch(format!(
                "Akita commitment covers {} polynomials but setup supports {}",
                commitment.poly_count, setup.max_num_polys_per_commitment_group
            )));
        }

        transcript.append(setup);
        transcript.append(&AkitaBlackBoxBatchStatementTranscript::new(
            &statement,
            &commitment,
            point.as_slice(),
        ));
        let mut akita_transcript = AkitaTranscript::<AkitaField>::new(b"jolt-akita/batch");
        let statement_bridge = bridge_jolt_statement_challenge(transcript, &mut akita_transcript);
        if proof.statement_bridge != statement_bridge {
            return Err(OpeningsError::VerificationFailed);
        }
        transcript.append(proof);

        let backend_verifier =
            deserialize_akita::<AkitaBackendVerifier>(&setup.serialized_backend_bytes, &())?;
        let backend_commitment = deserialize_akita::<AkitaBackendCommitment>(
            &proof.commitment.serialized_backend_bytes,
            &(),
        )?;
        let proof_shape =
            deserialize_akita::<AkitaBackendProofShape>(&proof.serialized_akita_proof_shape, &())?;
        let backend_proof =
            deserialize_akita::<AkitaBackendProof>(&proof.serialized_akita_proof, &proof_shape)?;
        let claims = VerifierOpeningBatch::from_groups(
            point,
            vec![CommitmentGroup {
                claims: openings,
                commitment: &backend_commitment,
            }],
        )
        .map_err(akita_error)?;
        AkitaBackendScheme::batched_verify(
            &backend_proof,
            &backend_verifier,
            &mut akita_transcript,
            claims,
            BasisMode::Lagrange,
            SetupContributionMode::Direct,
        )
        .map_err(|_| OpeningsError::VerificationFailed)
    }
}
