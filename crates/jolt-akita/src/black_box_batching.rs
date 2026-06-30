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
use tracing::info_span;

use crate::adapters::{
    akita_error, bridge_jolt_statement_challenge, deserialize_akita, invalid_batch,
    jolt_to_akita_evals, one_hot_polynomial, polynomial_evaluations, reverse_point,
    serialize_akita, AkitaBackendCommitment, AkitaBackendDensePoly, AkitaBackendFlavor,
    AkitaBackendProof, AkitaBackendProofShape, AkitaBackendScheme, AkitaBackendVerifier,
    AkitaBatchProof, AkitaBlackBoxBatchStatementTranscript, AkitaCommitment, AkitaField,
    AkitaHintPolynomials, AkitaOneHotBackendScheme, AkitaProverHint, AkitaProverSetup,
    AkitaSourceKind, AkitaSparsePolynomial, AkitaVerifierSetup, AkitaVerifierSetupTranscript,
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
        let _span = info_span!(
            "AkitaBlackBoxBatching::prove_batch",
            source_kind = ?hint.source_kind,
            num_vars = point.len(),
            num_claims = statement.len(),
            poly_count = commitment.poly_count,
        )
        .entered();

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
        if hint.source_kind.backend_flavor() != commitment.backend_flavor {
            return Err(invalid_batch(
                "Akita prover hint backend flavor does not match commitment",
            ));
        }
        if polynomials.len() != commitment.poly_count {
            return Err(invalid_batch(format!(
                "Akita prover received {} polynomials for {} commitment slots",
                polynomials.len(),
                commitment.poly_count
            )));
        }
        for polynomial in &polynomials {
            if polynomial.num_vars() != commitment.num_vars {
                return Err(invalid_batch(format!(
                    "Akita witness polynomial has {} variables but commitment has {}",
                    polynomial.num_vars(),
                    commitment.num_vars
                )));
            }
        }

        match hint.source_kind {
            AkitaSourceKind::Dense => {
                let dense = match hint.backend_polynomials {
                    Some(AkitaHintPolynomials::Dense(dense)) => dense,
                    Some(AkitaHintPolynomials::OneHot(_)) => {
                        return Err(invalid_batch(
                            "Akita dense prover hint contains one-hot backend polynomials",
                        ));
                    }
                    Some(AkitaHintPolynomials::SparseUnit(_)) => {
                        return Err(invalid_batch(
                            "Akita dense prover hint contains sparse backend polynomials",
                        ));
                    }
                    None => {
                        let _span = info_span!("AkitaBlackBoxBatching::materialize_dense_witness")
                            .entered();
                        polynomials
                            .iter()
                            .map(|polynomial| {
                                let evals = polynomial_evaluations(*polynomial);
                                let akita_evals =
                                    jolt_to_akita_evals(polynomial.num_vars(), &evals)?;
                                AkitaBackendDensePoly::from_field_evals(
                                    polynomial.num_vars(),
                                    &akita_evals,
                                )
                                .map_err(akita_error)
                            })
                            .collect::<Result<Vec<_>, OpeningsError>>()?
                            .into()
                    }
                };
                if dense.len() != commitment.poly_count {
                    return Err(invalid_batch(format!(
                        "Akita prover received {} polynomials for {} commitment slots",
                        dense.len(),
                        commitment.poly_count
                    )));
                }
                for polynomial in dense.iter() {
                    if polynomial.num_vars() != commitment.num_vars {
                        return Err(invalid_batch(format!(
                            "Akita polynomial has {} variables but commitment has {}",
                            polynomial.num_vars(),
                            commitment.num_vars
                        )));
                    }
                }

                {
                    let _span =
                        info_span!("AkitaBlackBoxBatching::append_setup_and_statement").entered();
                    transcript.append(&AkitaVerifierSetupTranscript::new(
                        &setup.verifier,
                        commitment.backend_flavor,
                    ));
                    transcript.append(&AkitaBlackBoxBatchStatementTranscript::new(
                        &statement,
                        &commitment,
                        point.as_slice(),
                    ));
                }
                let mut akita_transcript = AkitaTranscript::<AkitaField>::new(b"jolt-akita/batch");
                let statement_bridge = {
                    let _span = info_span!("AkitaBlackBoxBatching::bridge_transcripts").entered();
                    bridge_jolt_statement_challenge(transcript, &mut akita_transcript)
                };

                let backend_commitment = if let Some(backend_commitment) = hint.backend_commitment {
                    backend_commitment
                } else {
                    let _span = info_span!("AkitaBlackBoxBatching::deserialize_backend_commitment")
                        .entered();
                    deserialize_akita::<AkitaBackendCommitment>(
                        &commitment.serialized_backend_bytes,
                        &(),
                    )?
                };
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
                let stack = {
                    let _span = info_span!("AkitaBlackBoxBatching::make_backend_stack").entered();
                    akita_prover::UniformProverStack::uniform(
                        &CpuBackend,
                        &setup.prepared_backend_setup,
                        setup.backend_prover_setup.expanded.as_ref(),
                    )
                    .map_err(akita_error)?
                };

                let backend_proof = {
                    let _span =
                        info_span!("AkitaBlackBoxBatching::backend_batched_prove").entered();
                    AkitaBackendScheme::batched_prove(
                        &setup.backend_prover_setup,
                        claims,
                        &stack,
                        &mut akita_transcript,
                        BasisMode::Lagrange,
                        SetupContributionMode::Direct,
                    )
                    .map_err(akita_error)?
                };
                let proof = {
                    let _span =
                        info_span!("AkitaBlackBoxBatching::serialize_backend_proof").entered();
                    let proof_shape = backend_proof.shape();
                    AkitaBatchProof {
                        commitment,
                        statement_bridge,
                        serialized_akita_proof_shape: serialize_akita(&proof_shape)?,
                        serialized_akita_proof: serialize_akita(&backend_proof)?,
                    }
                };
                {
                    let _span = info_span!("AkitaBlackBoxBatching::append_proof").entered();
                    transcript.append(&proof);
                }
                Ok(proof)
            }
            AkitaSourceKind::OneHot => {
                for polynomial in &polynomials {
                    if !polynomial.is_one_hot() {
                        return Err(invalid_batch(
                            "Akita one-hot prover hint requires one-hot witness polynomials",
                        ));
                    }
                }
                let one_hot = match hint.backend_polynomials {
                    Some(AkitaHintPolynomials::OneHot(one_hot)) => one_hot,
                    Some(AkitaHintPolynomials::Dense(_)) => {
                        return Err(invalid_batch(
                            "Akita one-hot prover hint contains dense backend polynomials",
                        ));
                    }
                    Some(AkitaHintPolynomials::SparseUnit(_)) => {
                        return Err(invalid_batch(
                            "Akita one-hot prover hint contains sparse backend polynomials",
                        ));
                    }
                    None => {
                        let _span =
                            info_span!("AkitaBlackBoxBatching::materialize_one_hot_witness")
                                .entered();
                        polynomials
                            .iter()
                            .map(|polynomial| {
                                one_hot_polynomial(*polynomial)?.ok_or_else(|| {
                                    invalid_batch(
                                        "Akita one-hot backend requires row-major one-hot polynomials with k=256",
                                    )
                                })
                            })
                            .collect::<Result<Vec<_>, OpeningsError>>()?
                            .into()
                    }
                };
                if one_hot.len() != commitment.poly_count {
                    return Err(invalid_batch(format!(
                        "Akita prover received {} polynomials for {} commitment slots",
                        one_hot.len(),
                        commitment.poly_count
                    )));
                }
                for polynomial in one_hot.iter() {
                    if polynomial.num_vars() != commitment.num_vars {
                        return Err(invalid_batch(format!(
                            "Akita polynomial has {} variables but commitment has {}",
                            polynomial.num_vars(),
                            commitment.num_vars
                        )));
                    }
                }

                {
                    let _span =
                        info_span!("AkitaBlackBoxBatching::append_setup_and_statement").entered();
                    transcript.append(&AkitaVerifierSetupTranscript::new(
                        &setup.verifier,
                        commitment.backend_flavor,
                    ));
                    transcript.append(&AkitaBlackBoxBatchStatementTranscript::new(
                        &statement,
                        &commitment,
                        point.as_slice(),
                    ));
                }
                let mut akita_transcript = AkitaTranscript::<AkitaField>::new(b"jolt-akita/batch");
                let statement_bridge = {
                    let _span = info_span!("AkitaBlackBoxBatching::bridge_transcripts").entered();
                    bridge_jolt_statement_challenge(transcript, &mut akita_transcript)
                };

                let backend_commitment = if let Some(backend_commitment) = hint.backend_commitment {
                    backend_commitment
                } else {
                    let _span = info_span!("AkitaBlackBoxBatching::deserialize_backend_commitment")
                        .entered();
                    deserialize_akita::<AkitaBackendCommitment>(
                        &commitment.serialized_backend_bytes,
                        &(),
                    )?
                };
                let backend_hint = hint.backend_hint.ok_or_else(|| {
                    invalid_batch("Akita prover hint is missing backend opening data")
                })?;
                let backend_point = reverse_point(&point);
                let poly_refs = one_hot.iter().collect::<Vec<_>>();
                let claims = ProverOpeningBatch {
                    point: backend_point.as_slice().into(),
                    groups: vec![ProverCommitmentGroup {
                        point_vars: PointVariableSelection::prefix(
                            backend_point.len(),
                            backend_point.len(),
                        )
                        .map_err(akita_error)?,
                        polynomials: poly_refs.as_slice(),
                        commitment: (backend_commitment, backend_hint),
                    }],
                };
                let backend_prover_setup =
                    setup.one_hot_backend_prover_setup.as_ref().ok_or_else(|| {
                        invalid_batch("Akita verifier/prover setup has no one-hot backend")
                    })?;
                let prepared_backend_setup = setup
                    .prepared_one_hot_backend_setup
                    .as_ref()
                    .ok_or_else(|| {
                        invalid_batch("Akita prover setup has no prepared one-hot backend")
                    })?;
                let stack = {
                    let _span = info_span!("AkitaBlackBoxBatching::make_backend_stack").entered();
                    akita_prover::UniformProverStack::uniform(
                        &CpuBackend,
                        prepared_backend_setup,
                        backend_prover_setup.expanded.as_ref(),
                    )
                    .map_err(akita_error)?
                };

                let backend_proof = {
                    let _span =
                        info_span!("AkitaBlackBoxBatching::backend_batched_prove").entered();
                    AkitaOneHotBackendScheme::batched_prove(
                        backend_prover_setup,
                        claims,
                        &stack,
                        &mut akita_transcript,
                        BasisMode::Lagrange,
                        SetupContributionMode::Direct,
                    )
                    .map_err(akita_error)?
                };
                let proof = {
                    let _span =
                        info_span!("AkitaBlackBoxBatching::serialize_backend_proof").entered();
                    let proof_shape = backend_proof.shape();
                    AkitaBatchProof {
                        commitment,
                        statement_bridge,
                        serialized_akita_proof_shape: serialize_akita(&proof_shape)?,
                        serialized_akita_proof: serialize_akita(&backend_proof)?,
                    }
                };
                {
                    let _span = info_span!("AkitaBlackBoxBatching::append_proof").entered();
                    transcript.append(&proof);
                }
                Ok(proof)
            }
            AkitaSourceKind::SparseUnit => {
                for polynomial in &polynomials {
                    if !polynomial.is_one_hot() {
                        return Err(invalid_batch(
                            "Akita sparse prover hint requires one-hot witness polynomials",
                        ));
                    }
                }
                let backend_polynomials = match hint.backend_polynomials {
                    Some(AkitaHintPolynomials::SparseUnit(sparse)) => sparse,
                    Some(AkitaHintPolynomials::OneHot(_)) => {
                        return Err(invalid_batch(
                            "Akita sparse prover hint contains one-hot backend polynomials",
                        ));
                    }
                    Some(AkitaHintPolynomials::Dense(_)) => {
                        return Err(invalid_batch(
                            "Akita sparse prover hint contains dense backend polynomials",
                        ));
                    }
                    None => {
                        let _span = info_span!("AkitaBlackBoxBatching::materialize_sparse_witness")
                            .entered();
                        let mut sparse = Vec::with_capacity(polynomials.len());
                        for polynomial in &polynomials {
                            let mut indices = Vec::new();
                            polynomial.for_each_one(&mut |index| indices.push(index));
                            sparse.push(AkitaSparsePolynomial::from_jolt_unit_indices(
                                polynomial.num_vars(),
                                indices,
                            )?);
                        }
                        sparse
                            .into_iter()
                            .map(|polynomial| polynomial.backend_polynomial)
                            .collect::<Vec<_>>()
                            .into()
                    }
                };
                if backend_polynomials.len() != commitment.poly_count {
                    return Err(invalid_batch(format!(
                        "Akita prover received {} polynomials for {} commitment slots",
                        backend_polynomials.len(),
                        commitment.poly_count
                    )));
                }
                for polynomial in backend_polynomials.iter() {
                    if polynomial.num_vars() != commitment.num_vars {
                        return Err(invalid_batch(format!(
                            "Akita polynomial has {} variables but commitment has {}",
                            polynomial.num_vars(),
                            commitment.num_vars
                        )));
                    }
                }

                {
                    let _span =
                        info_span!("AkitaBlackBoxBatching::append_setup_and_statement").entered();
                    transcript.append(&AkitaVerifierSetupTranscript::new(
                        &setup.verifier,
                        commitment.backend_flavor,
                    ));
                    transcript.append(&AkitaBlackBoxBatchStatementTranscript::new(
                        &statement,
                        &commitment,
                        point.as_slice(),
                    ));
                }
                let mut akita_transcript = AkitaTranscript::<AkitaField>::new(b"jolt-akita/batch");
                let statement_bridge = {
                    let _span = info_span!("AkitaBlackBoxBatching::bridge_transcripts").entered();
                    bridge_jolt_statement_challenge(transcript, &mut akita_transcript)
                };

                let backend_commitment = if let Some(backend_commitment) = hint.backend_commitment {
                    backend_commitment
                } else {
                    let _span = info_span!("AkitaBlackBoxBatching::deserialize_backend_commitment")
                        .entered();
                    deserialize_akita::<AkitaBackendCommitment>(
                        &commitment.serialized_backend_bytes,
                        &(),
                    )?
                };
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
                let stack = {
                    let _span = info_span!("AkitaBlackBoxBatching::make_backend_stack").entered();
                    akita_prover::UniformProverStack::uniform(
                        &CpuBackend,
                        &setup.prepared_backend_setup,
                        setup.backend_prover_setup.expanded.as_ref(),
                    )
                    .map_err(akita_error)?
                };

                let backend_proof = {
                    let _span =
                        info_span!("AkitaBlackBoxBatching::backend_batched_prove").entered();
                    AkitaBackendScheme::batched_prove(
                        &setup.backend_prover_setup,
                        claims,
                        &stack,
                        &mut akita_transcript,
                        BasisMode::Lagrange,
                        SetupContributionMode::Direct,
                    )
                    .map_err(akita_error)?
                };
                let proof = {
                    let _span =
                        info_span!("AkitaBlackBoxBatching::serialize_backend_proof").entered();
                    let proof_shape = backend_proof.shape();
                    AkitaBatchProof {
                        commitment,
                        statement_bridge,
                        serialized_akita_proof_shape: serialize_akita(&proof_shape)?,
                        serialized_akita_proof: serialize_akita(&backend_proof)?,
                    }
                };
                {
                    let _span = info_span!("AkitaBlackBoxBatching::append_proof").entered();
                    transcript.append(&proof);
                }
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

        transcript.append(&AkitaVerifierSetupTranscript::new(
            setup,
            commitment.backend_flavor,
        ));
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

        let backend_verifier_bytes = match commitment.backend_flavor {
            AkitaBackendFlavor::Full => setup.serialized_backend_bytes.as_slice(),
            AkitaBackendFlavor::OneHot => setup
                .serialized_one_hot_backend_bytes
                .as_deref()
                .ok_or_else(|| invalid_batch("Akita verifier setup has no one-hot backend"))?,
        };
        let backend_verifier =
            deserialize_akita::<AkitaBackendVerifier>(backend_verifier_bytes, &())?;
        let backend_commitment = deserialize_akita::<AkitaBackendCommitment>(
            &proof.commitment.serialized_backend_bytes,
            &(),
        )?;
        let proof_shape =
            deserialize_akita::<AkitaBackendProofShape>(&proof.serialized_akita_proof_shape, &())?;
        let backend_proof =
            deserialize_akita::<AkitaBackendProof>(&proof.serialized_akita_proof, &proof_shape)?;
        let backend_point = match commitment.backend_flavor {
            AkitaBackendFlavor::Full => point,
            AkitaBackendFlavor::OneHot => reverse_point(&point),
        };
        let claims = VerifierOpeningBatch::from_groups(
            backend_point,
            vec![CommitmentGroup {
                claims: openings,
                commitment: &backend_commitment,
            }],
        )
        .map_err(akita_error)?;
        match commitment.backend_flavor {
            AkitaBackendFlavor::Full => AkitaBackendScheme::batched_verify(
                &backend_proof,
                &backend_verifier,
                &mut akita_transcript,
                claims,
                BasisMode::Lagrange,
                SetupContributionMode::Direct,
            ),
            AkitaBackendFlavor::OneHot => AkitaOneHotBackendScheme::batched_verify(
                &backend_proof,
                &backend_verifier,
                &mut akita_transcript,
                claims,
                BasisMode::Lagrange,
                SetupContributionMode::Direct,
            ),
        }
        .map_err(|_| OpeningsError::VerificationFailed)
    }
}
