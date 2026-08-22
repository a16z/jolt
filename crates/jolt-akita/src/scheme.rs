use akita_pcs::{ComputeBackendSetup, CpuBackend};
use akita_types::PrecommittedGroupProfiles;
use jolt_crypto::Commitment;
use jolt_field::CanonicalBytes;
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, GroupOpeningClaim, OpeningsError,
    PrecommittedClaim, PrecommittedOpening, PrecommittedRole, TransparentObjectSetup,
    VerifierOpeningClaim, ZkBatchOpeningScheme, ZkOpeningScheme,
};
use jolt_poly::{MultilinearPoly, OneHotPolynomial, Polynomial};
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::adapters::{
    akita_error, akita_ordered_evaluations, backend_stack, commit_failed, dense_polynomials,
    domain_size, invalid_batch, one_hot_polynomial, owned_one_hot_polynomial, serialize_akita,
    sparse_unit_polynomial, transparent_zk_error, validate_one_hot_k, with_backend_pool,
    AkitaBackendCommitment, AkitaBackendDensePoly, AkitaBackendHint, AkitaBackendOneHotPoly,
    AkitaBackendScheme, AkitaBatchProof, AkitaCommitment, AkitaField, AkitaHidingCommitment,
    AkitaHintPolynomials, AkitaLayoutDigest, AkitaOneHotK16BackendScheme,
    AkitaOneHotK256BackendScheme, AkitaProverHint, AkitaProverSetup, AkitaSetupParams,
    AkitaVerifierSetup, AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256, AKITA_SOURCE_RING_DIMENSION,
};
use crate::native_batching::{AkitaNativeBatchPolynomials, AkitaNativeBatching};
use crate::trace_onehot::{TraceOneHotRows, TracePackedOneHot};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaScheme;

fn split_commit_output(
    output: akita_prover::CommitOutput<AkitaField>,
) -> (AkitaBackendCommitment, AkitaBackendHint) {
    (output.committed_group, output.hint)
}

/// Prover seam for committing the packed trace directly from selected one-hot rows.
pub trait TraceOneHotCommitment: CommitmentScheme {
    fn commit_trace_one_hot(
        setup: &Self::ProverSetup,
        layout_digest: [u8; 32],
        column_capacity: usize,
        rows: Arc<dyn TraceOneHotRows>,
        precommitted_hints: &[&Self::OpeningHint],
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError>;

    /// Releases backend state that can be rebuilt before the opening proof.
    fn release_post_commit_residency(setup: &Self::ProverSetup) -> Result<(), OpeningsError>;
}

/// Strictly ascending roles make the ordered precommitted group list
/// unambiguous and forbid duplicate or permuted groups.
pub(crate) fn validate_precommitted_order(
    roles: impl IntoIterator<Item = PrecommittedRole>,
) -> Result<(), OpeningsError> {
    let mut previous: Option<PrecommittedRole> = None;
    for role in roles {
        if let Some(previous) = previous {
            if role <= previous {
                return Err(invalid_batch(format!(
                    "Akita precommitted groups must be in canonical ascending order, found {} after {}",
                    role.diagnostic_name(),
                    previous.diagnostic_name()
                )));
            }
        }
        previous = Some(role);
    }
    Ok(())
}

impl AkitaScheme {
    /// Returns true when the Akita backend sparse-ring path can represent a
    /// unit-valued sparse polynomial with this multilinear dimension.
    pub fn supports_unit_sparse_dimension(num_vars: usize) -> bool {
        domain_size(num_vars).is_some_and(|size| size >= AKITA_SOURCE_RING_DIMENSION)
    }

    pub fn commit_group(
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
        polynomials: &[Polynomial<AkitaField>],
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let first = polynomials
            .first()
            .ok_or_else(|| invalid_batch("Akita commitment group must contain a polynomial"))?;
        let num_vars = first.num_vars();

        Self::validate_commit_shape(setup, num_vars, polynomials.len())?;
        for polynomial in polynomials {
            if polynomial.num_vars() != num_vars {
                return Err(invalid_batch(format!(
                    "Akita commitment group mixes {}-variable and {num_vars}-variable polynomials",
                    polynomial.num_vars()
                )));
            }
        }

        let dense = dense_polynomials(polynomials)?;
        Self::commit_dense_backend(setup, layout_digest, num_vars, dense)
    }

    /// Commits a group of row-major one-hot polynomials through the
    /// backend's one-hot flavor as one commitment object whose members are
    /// opened together at a shared point.
    pub fn commit_one_hot_group(
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
        polynomials: &[OneHotPolynomial],
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let first = polynomials
            .first()
            .ok_or_else(|| invalid_batch("Akita commitment group must contain a polynomial"))?;
        let num_vars = first.num_vars();
        Self::validate_commit_shape(setup, num_vars, polynomials.len())?;
        let backend_polynomials = polynomials
            .iter()
            .map(|polynomial| {
                if polynomial.num_vars() != num_vars {
                    return Err(invalid_batch(format!(
                        "Akita commitment group mixes {}-variable and {num_vars}-variable polynomials",
                        polynomial.num_vars()
                    )));
                }
                one_hot_polynomial(polynomial, setup.one_hot_k())?
                .ok_or_else(|| {
                    invalid_batch(format!(
                        "Akita one-hot commitment group requires row-major K={} one-hot polynomials",
                        setup.one_hot_k()
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (backend_commitment, backend_hint) =
            Self::commit_one_hot_backend(setup, &backend_polynomials)?;
        Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintPolynomials::OneHot(backend_polynomials.into()),
        )
    }

    /// Commits owned one-hot columns without cloning their hot-index buffers
    /// at the Jolt/Akita boundary. The opening hint retains the backend
    /// representations needed by the prover.
    pub fn commit_one_hot_group_owned(
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
        polynomials: Vec<OneHotPolynomial>,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let first = polynomials
            .first()
            .ok_or_else(|| invalid_batch("Akita commitment group must contain a polynomial"))?;
        let num_vars = first.num_vars();
        Self::validate_commit_shape(setup, num_vars, polynomials.len())?;
        let backend_polynomials = polynomials
            .into_iter()
            .map(|polynomial| {
                if polynomial.num_vars() != num_vars {
                    return Err(invalid_batch(format!(
                        "Akita commitment group mixes {}-variable and {num_vars}-variable polynomials",
                        polynomial.num_vars()
                    )));
                }
                owned_one_hot_polynomial(polynomial, setup.one_hot_k())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (backend_commitment, backend_hint) =
            Self::commit_one_hot_backend(setup, &backend_polynomials)?;
        Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintPolynomials::OneHot(backend_polynomials.into()),
        )
    }

    /// Contextual owned one-hot final commit used by the legacy packed path.
    /// The witness buffers move into the opening hint without cloning.
    pub fn commit_one_hot_group_owned_with_precommitted(
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
        polynomials: Vec<OneHotPolynomial>,
        precommitted_hints: &[&AkitaProverHint],
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let first = polynomials
            .first()
            .ok_or_else(|| invalid_batch("Akita commitment group must contain a polynomial"))?;
        let num_vars = first.num_vars();
        Self::validate_commit_shape(setup, num_vars, polynomials.len())?;
        let backend_polynomials = polynomials
            .into_iter()
            .map(|polynomial| {
                if polynomial.num_vars() != num_vars {
                    return Err(invalid_batch(format!(
                        "Akita commitment group mixes {}-variable and {num_vars}-variable polynomials",
                        polynomial.num_vars()
                    )));
                }
                owned_one_hot_polynomial(polynomial, setup.one_hot_k())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let profiles = Self::precommitted_profiles(setup, precommitted_hints)?;
        let (backend_commitment, backend_hint) =
            Self::commit_one_hot_backend_with_precommitted(setup, &backend_polynomials, &profiles)?;
        Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintPolynomials::OneHot(backend_polynomials.into()),
        )
    }

    /// Commits the prefix-packed trace without constructing padded per-column
    /// index vectors or Akita's generic one-hot block representation.
    pub fn commit_trace_one_hot(
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
        column_capacity: usize,
        rows: Arc<dyn TraceOneHotRows>,
        precommitted_hints: &[&AkitaProverHint],
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let profiles = if precommitted_hints.is_empty() {
            None
        } else {
            Some(Self::precommitted_profiles(setup, precommitted_hints)?)
        };
        let source = TracePackedOneHot::new(
            setup.one_hot_k(),
            AKITA_SOURCE_RING_DIMENSION,
            column_capacity,
            rows,
        )
        .map_err(commit_failed)?;
        let num_vars = akita_prover::RootPolyMeta::num_vars(&source);
        Self::validate_commit_shape(setup, num_vars, 1)?;
        let (backend_prover_setup, prepared_backend_setup) = setup.one_hot_backend()?;
        let stack = backend_stack(backend_prover_setup, prepared_backend_setup)?;
        let (backend_commitment, backend_hint) =
            with_backend_pool(|| match (setup.one_hot_k(), profiles.as_ref()) {
                (AKITA_ONE_HOT_K16, None) => AkitaOneHotK16BackendScheme::commit(
                    backend_prover_setup,
                    std::slice::from_ref(&source),
                    &stack,
                    akita_prover::GroupContext::scheduler_without_precommitted_groups(),
                ),
                (AKITA_ONE_HOT_K16, Some(profiles)) => AkitaOneHotK16BackendScheme::commit(
                    backend_prover_setup,
                    std::slice::from_ref(&source),
                    &stack,
                    akita_prover::GroupContext::scheduler_with_precommitted_groups(profiles),
                ),
                (AKITA_ONE_HOT_K256, None) => AkitaOneHotK256BackendScheme::commit(
                    backend_prover_setup,
                    std::slice::from_ref(&source),
                    &stack,
                    akita_prover::GroupContext::scheduler_without_precommitted_groups(),
                ),
                (AKITA_ONE_HOT_K256, Some(profiles)) => AkitaOneHotK256BackendScheme::commit(
                    backend_prover_setup,
                    std::slice::from_ref(&source),
                    &stack,
                    akita_prover::GroupContext::scheduler_with_precommitted_groups(profiles),
                ),
                _ => unreachable!("the one-hot setup geometry was validated during setup"),
            })
            .map(split_commit_output)
            .map_err(commit_failed)?;
        Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintPolynomials::TraceOneHot(source),
        )
    }

    fn commit_one_hot_backend(
        setup: &AkitaProverSetup,
        polynomials: &[AkitaBackendOneHotPoly],
    ) -> Result<(AkitaBackendCommitment, AkitaBackendHint), OpeningsError> {
        let (backend_prover_setup, prepared_backend_setup) = setup.one_hot_backend()?;
        let stack = backend_stack(backend_prover_setup, prepared_backend_setup)?;
        with_backend_pool(|| match setup.one_hot_k() {
            AKITA_ONE_HOT_K16 => AkitaOneHotK16BackendScheme::commit(
                backend_prover_setup,
                polynomials,
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            ),
            AKITA_ONE_HOT_K256 => AkitaOneHotK256BackendScheme::commit(
                backend_prover_setup,
                polynomials,
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            ),
            _ => unreachable!("the one-hot setup geometry was validated during setup"),
        })
        .map(split_commit_output)
        .map_err(commit_failed)
    }

    fn commit_one_hot_backend_with_precommitted(
        setup: &AkitaProverSetup,
        polynomials: &[AkitaBackendOneHotPoly],
        profiles: &PrecommittedGroupProfiles,
    ) -> Result<(AkitaBackendCommitment, AkitaBackendHint), OpeningsError> {
        let (backend_prover_setup, prepared_backend_setup) = setup.one_hot_backend()?;
        let stack = backend_stack(backend_prover_setup, prepared_backend_setup)?;
        with_backend_pool(|| match setup.one_hot_k() {
            AKITA_ONE_HOT_K16 => AkitaOneHotK16BackendScheme::commit(
                backend_prover_setup,
                polynomials,
                &stack,
                akita_prover::GroupContext::scheduler_with_precommitted_groups(profiles),
            ),
            AKITA_ONE_HOT_K256 => AkitaOneHotK256BackendScheme::commit(
                backend_prover_setup,
                polynomials,
                &stack,
                akita_prover::GroupContext::scheduler_with_precommitted_groups(profiles),
            ),
            _ => unreachable!("the one-hot setup geometry was validated during setup"),
        })
        .map(split_commit_output)
        .map_err(commit_failed)
    }

    /// Freezes the ordered precommitted profiles the final trace group commits
    /// against. Order is the caller's canonical role order; the backend keys the
    /// grouped row on this exact sequence.
    fn precommitted_profiles(
        setup: &AkitaProverSetup,
        precommitted_hints: &[&AkitaProverHint],
    ) -> Result<PrecommittedGroupProfiles, OpeningsError> {
        if precommitted_hints.is_empty() {
            return Err(invalid_batch(
                "Akita grouped trace opening requires at least one precommitted group",
            ));
        }
        // Every precommitted group plus the final trace group must fit the
        // setup's total batch capacity.
        let required = precommitted_hints
            .len()
            .checked_add(1)
            .ok_or_else(|| invalid_batch("Akita precommitted group count overflows"))?;
        if setup.max_total_batch_polys() < required {
            return Err(invalid_batch(format!(
                "Akita grouped trace opening requires total polynomial capacity {required}, setup has {}",
                setup.max_total_batch_polys()
            )));
        }
        let mut profiles = Vec::with_capacity(precommitted_hints.len());
        for hint in precommitted_hints {
            if hint.commitment.backend_flavor != crate::adapters::AkitaBackendFlavor::Dense
                || hint.commitment.poly_count != 1
                || !matches!(hint.polynomials, AkitaHintPolynomials::Dense(_))
            {
                return Err(invalid_batch(
                    "Akita trace precommit must be one advice polynomial",
                ));
            }
            let (precommitted_group, _) = hint.backend.as_ref().ok_or_else(|| {
                invalid_batch("Akita advice precommit is missing backend opening data")
            })?;
            profiles.push(precommitted_group.profile);
        }
        PrecommittedGroupProfiles::from_profiles(profiles).map_err(akita_error)
    }

    /// Validates the commitment shape before handing values to Akita.
    fn validate_commit_shape(
        setup: &AkitaProverSetup,
        num_vars: usize,
        poly_count: usize,
    ) -> Result<(), OpeningsError> {
        if num_vars != setup.max_num_vars() {
            return Err(invalid_batch(format!(
                "Akita commitment dimension {num_vars} does not match exact setup dimension {}",
                setup.max_num_vars()
            )));
        }
        if poly_count > setup.max_num_polys_per_commitment_group() {
            return Err(invalid_batch(format!(
                "Akita commitment group has {poly_count} polynomials but setup supports {}",
                setup.max_num_polys_per_commitment_group()
            )));
        }
        Ok(())
    }

    /// Wraps a backend commitment and its opening data into the adapter's
    /// commitment/hint pair; the flavor and polynomial count come from the
    /// hint polynomials themselves.
    fn package_commitment(
        layout_digest: AkitaLayoutDigest,
        num_vars: usize,
        backend_commitment: AkitaBackendCommitment,
        backend_hint: AkitaBackendHint,
        polynomials: AkitaHintPolynomials,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let backend_flavor = polynomials.backend_flavor();
        let one_hot_k = match backend_flavor {
            crate::adapters::AkitaBackendFlavor::Dense => 0,
            crate::adapters::AkitaBackendFlavor::OneHot => polynomials
                .one_hot_k()
                .ok_or_else(|| invalid_batch("Akita one-hot commitment group must not be empty"))?,
        };
        let commitment = AkitaCommitment {
            backend_flavor,
            layout_digest,
            num_vars,
            poly_count: polynomials.len(),
            one_hot_k,
            backend_coeff_len: backend_commitment.rows().coeff_len(),
            serialized_backend_bytes: serialize_akita(backend_commitment.commitment())?,
        };
        Ok((
            commitment.clone(),
            AkitaProverHint {
                commitment,
                backend: Some((backend_commitment, backend_hint)),
                polynomials,
            },
        ))
    }

    fn commit_dense_backend(
        setup: &AkitaProverSetup,
        layout_digest: AkitaLayoutDigest,
        num_vars: usize,
        dense: Vec<AkitaBackendDensePoly>,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let (backend_prover_setup, prepared_backend_setup) = setup.dense_backend()?;
        let stack = backend_stack(backend_prover_setup, prepared_backend_setup)?;
        let (backend_commitment, backend_hint) = with_backend_pool(|| {
            AkitaBackendScheme::commit(
                backend_prover_setup,
                dense.as_slice(),
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
        })
        .map(split_commit_output)
        .map_err(commit_failed)?;
        Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintPolynomials::Dense(dense.into()),
        )
    }
}

impl TraceOneHotCommitment for AkitaScheme {
    fn commit_trace_one_hot(
        setup: &Self::ProverSetup,
        layout_digest: [u8; 32],
        column_capacity: usize,
        rows: Arc<dyn TraceOneHotRows>,
        precommitted_hints: &[&Self::OpeningHint],
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
        Self::commit_trace_one_hot(
            setup,
            layout_digest,
            column_capacity,
            rows,
            precommitted_hints,
        )
    }

    fn release_post_commit_residency(setup: &Self::ProverSetup) -> Result<(), OpeningsError> {
        setup.release_post_commit_ntt_residency()
    }
}

impl Commitment for AkitaScheme {
    type Output = AkitaCommitment;
}

impl CommitmentScheme for AkitaScheme {
    type Field = AkitaField;
    type Proof = AkitaBatchProof;
    type ProverSetup = AkitaProverSetup;
    type VerifierSetup = AkitaVerifierSetup;
    type OpeningHint = AkitaProverHint;
    type SetupParams = AkitaSetupParams;

    fn setup(
        params: Self::SetupParams,
    ) -> Result<(Self::ProverSetup, Self::VerifierSetup), OpeningsError> {
        let invalid_setup =
            |err: &dyn std::fmt::Display| OpeningsError::InvalidSetup(err.to_string());
        debug_assert!(
            !(params.one_hot_only && params.dense_only),
            "a setup cannot skip both backend flavors"
        );
        let one_hot_log_k = validate_one_hot_k(params.one_hot_k)
            .map_err(|err| OpeningsError::InvalidSetup(err.to_string()))?;
        let (backend_prover_setup, prepared_backend_setup, backend_verifier_setup) = if params
            .one_hot_only
        {
            (None, None, None)
        } else {
            let backend_prover_setup = with_backend_pool(|| {
                AkitaBackendScheme::setup_prover(params.max_num_vars, params.max_total_batch_polys)
            })
            .map_err(|err| invalid_setup(&err))?;
            let prepared_backend_setup =
                with_backend_pool(|| CpuBackend::DEFAULT.prepare_setup(&backend_prover_setup))
                    .map_err(|err| invalid_setup(&err))?;
            let backend_verifier_setup =
                with_backend_pool(|| AkitaBackendScheme::setup_verifier(&backend_prover_setup))
                    .map_err(|err| invalid_setup(&err))?;
            (
                Some(std::sync::Arc::new(backend_prover_setup)),
                Some(std::sync::Arc::new(prepared_backend_setup)),
                Some(backend_verifier_setup),
            )
        };
        let (
            one_hot_backend_prover_setup,
            prepared_one_hot_backend_setup,
            one_hot_backend_verifier_setup,
        ) = if params.max_num_vars >= one_hot_log_k && !params.dense_only {
            let backend_prover_setup = crate::adapters::one_hot_setup_prover(
                params.one_hot_k,
                params.max_num_vars,
                params.max_total_batch_polys,
            )
            .map_err(|err| invalid_setup(&err))?;
            let prepared_backend_setup =
                with_backend_pool(|| CpuBackend::DEFAULT.prepare_setup(&backend_prover_setup))
                    .map_err(|err| invalid_setup(&err))?;
            let backend_verifier_setup =
                crate::adapters::one_hot_setup_verifier(params.one_hot_k, &backend_prover_setup)?;
            (
                Some(std::sync::Arc::new(backend_prover_setup)),
                Some(std::sync::Arc::new(prepared_backend_setup)),
                Some(backend_verifier_setup),
            )
        } else {
            (None, None, None)
        };
        let verifier = AkitaVerifierSetup {
            max_num_vars: params.max_num_vars,
            max_num_polys_per_commitment_group: params.max_num_polys_per_commitment_group,
            max_total_batch_polys: params.max_total_batch_polys,
            default_layout_digest: params.default_layout_digest,
            one_hot_k: params.one_hot_k,
            backend_cache: Default::default(),
        };
        verifier.prime_backend_cache(backend_verifier_setup, one_hot_backend_verifier_setup);
        let prover = AkitaProverSetup {
            backend_prover_setup,
            prepared_backend_setup,
            one_hot_backend_prover_setup,
            prepared_one_hot_backend_setup,
            verifier: verifier.clone(),
        };
        Ok((prover, verifier))
    }

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        prover_setup.verifier.clone()
    }

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
        if let Some(one_hot) = one_hot_polynomial(poly, setup.one_hot_k())? {
            let num_vars = akita_prover::RootPolyMeta::num_vars(&one_hot);
            Self::validate_commit_shape(setup, num_vars, 1)?;
            let (backend_commitment, backend_hint) =
                Self::commit_one_hot_backend(setup, std::slice::from_ref(&one_hot))?;
            return Self::package_commitment(
                setup.default_layout_digest(),
                num_vars,
                backend_commitment,
                backend_hint,
                AkitaHintPolynomials::OneHot(vec![one_hot].into()),
            );
        }

        if poly.is_one_hot() && Self::supports_unit_sparse_dimension(poly.num_vars()) {
            let mut indices = Vec::new();
            poly.for_each_one(&mut |index| indices.push(index));
            let sparse = sparse_unit_polynomial(poly.num_vars(), indices)?;
            let num_vars = sparse.num_vars();
            Self::validate_commit_shape(setup, num_vars, 1)?;
            let (backend_prover_setup, prepared_backend_setup) = setup.dense_backend()?;
            let stack = backend_stack(backend_prover_setup, prepared_backend_setup)?;
            let (backend_commitment, backend_hint) = with_backend_pool(|| {
                AkitaBackendScheme::commit(
                    backend_prover_setup,
                    std::slice::from_ref(&sparse),
                    &stack,
                    akita_prover::GroupContext::scheduler_without_precommitted_groups(),
                )
            })
            .map(split_commit_output)
            .map_err(commit_failed)?;
            return Self::package_commitment(
                setup.default_layout_digest(),
                num_vars,
                backend_commitment,
                backend_hint,
                AkitaHintPolynomials::SparseUnit(vec![sparse].into()),
            );
        }

        let num_vars = poly.num_vars();
        Self::validate_commit_shape(setup, num_vars, 1)?;
        let evals = akita_ordered_evaluations(poly)?;
        let dense = vec![AkitaBackendDensePoly::from_field_evals(
            num_vars,
            AKITA_SOURCE_RING_DIMENSION,
            &evals,
        )
        .map_err(akita_error)?];
        Self::commit_dense_backend(setup, setup.default_layout_digest(), num_vars, dense)
    }

    fn open<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::Proof, OpeningsError> {
        let hint = match hint {
            Some(hint) => hint,
            None => Self::commit(poly, setup)?.1,
        };
        let statement = vec![VerifierOpeningClaim {
            commitment: hint.commitment.clone(),
            evaluation: EvaluationClaim::new(point.to_vec(), eval),
        }];
        let polynomials: AkitaNativeBatchPolynomials<'_> =
            vec![&poly as &(dyn MultilinearPoly<AkitaField> + '_)];
        <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            setup,
            statement,
            polynomials,
            hint,
            transcript,
        )
    }

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        let statement = vec![VerifierOpeningClaim {
            commitment: commitment.clone(),
            evaluation: EvaluationClaim::new(point.to_vec(), eval),
        }];
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            setup, &statement, proof, transcript,
        )
    }

    /// Commits a group of row-major one-hot polynomials through the backend's
    /// one-hot flavor — [`AkitaScheme::commit_one_hot_group`] behind the
    /// scheme-generic seam (the trait object's `one_hot_indices` accessor
    /// feeds the same backend representation, so the commitment is
    /// byte-identical to the inherent paths).
    fn commit_batch(
        polynomials: &[&dyn MultilinearPoly<Self::Field>],
        layout_digest: [u8; 32],
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
        let first = polynomials
            .first()
            .ok_or_else(|| invalid_batch("Akita commitment group must contain a polynomial"))?;
        let num_vars = first.num_vars();
        Self::validate_commit_shape(setup, num_vars, polynomials.len())?;
        let backend_polynomials = polynomials
            .iter()
            .map(|polynomial| {
                if polynomial.num_vars() != num_vars {
                    return Err(invalid_batch(format!(
                        "Akita commitment group mixes {}-variable and {num_vars}-variable polynomials",
                        polynomial.num_vars()
                    )));
                }
                one_hot_polynomial(*polynomial, setup.one_hot_k())?
                .ok_or_else(|| {
                    invalid_batch(format!(
                        "Akita one-hot commitment group requires row-major K={} one-hot polynomials",
                        setup.one_hot_k()
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (backend_commitment, backend_hint) =
            Self::commit_one_hot_backend(setup, &backend_polynomials)?;
        Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintPolynomials::OneHot(backend_polynomials.into()),
        )
    }

    fn prove_batch(
        setup: &Self::ProverSetup,
        precommitted: Vec<PrecommittedOpening<Self::Field, Self::Output, Self::OpeningHint>>,
        final_group: GroupOpeningClaim<Self::Field, Self::Output>,
        final_hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::Proof, OpeningsError> {
        AkitaNativeBatching::prove_trace_batch(
            setup,
            precommitted,
            final_group,
            final_hint,
            transcript,
        )
    }

    fn verify_batch(
        setup: &Self::VerifierSetup,
        precommitted: &[PrecommittedClaim<Self::Field, Self::Output>],
        final_group: &GroupOpeningClaim<Self::Field, Self::Output>,
        proof: &Self::Proof,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        AkitaNativeBatching::verify_trace_batch(setup, precommitted, final_group, proof, transcript)
    }
}

impl TransparentObjectSetup for AkitaScheme {
    /// The singleton commitment-object setup convention (advice byte columns,
    /// `ProgramOneHot`): one polynomial at `num_vars`, seeded by the object
    /// plan's layout digest. Auxiliary packed objects use the sparse-unit/dense
    /// flavor, so their setup omits the costly one-hot backend.
    fn transparent_object_setup(
        num_vars: usize,
        layout_digest: [u8; 32],
    ) -> Result<(AkitaProverSetup, AkitaVerifierSetup), OpeningsError> {
        Self::setup(AkitaSetupParams::dense_only(num_vars, 1, layout_digest))
    }
}

impl ZkOpeningScheme for AkitaScheme {
    type HidingCommitment = AkitaHidingCommitment;
    type Blind = ();

    fn commit_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
        Self::commit(poly, setup)
    }

    fn open_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(Self::Proof, Self::HidingCommitment, Self::Blind), OpeningsError> {
        let proof = Self::open(poly, point, eval, setup, Some(hint), transcript)?;
        Ok((
            proof,
            AkitaHidingCommitment::new(eval.to_bytes_le_vec()),
            (),
        ))
    }

    fn verify_zk(
        _commitment: &Self::Output,
        _point: &[Self::Field],
        _proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::HidingCommitment, OpeningsError> {
        Err(transparent_zk_error())
    }
}

impl ZkBatchOpeningScheme for AkitaNativeBatching {
    type Commitment = AkitaCommitment;
    type HidingCommitment = AkitaHidingCommitment;
    type Blind = ();

    fn prove_batch_zk<'a, T>(
        _setup: &Self::ProverSetup,
        _point: jolt_poly::Point<{ jolt_poly::HIGH_TO_LOW }, Self::Field>,
        _commitments: Vec<Self::Commitment>,
        _polynomials: Self::Polynomials<'a>,
        _hints: Self::Hints,
        _evaluations: Vec<Self::Field>,
        _transcript: &mut T,
    ) -> Result<jolt_openings::ZkBatchOpening<Self>, OpeningsError>
    where
        Self: 'a,
        T: Transcript<Challenge = Self::Field>,
    {
        Err(transparent_zk_error())
    }

    fn verify_batch_zk<T>(
        _setup: &Self::VerifierSetup,
        _point: jolt_poly::Point<{ jolt_poly::HIGH_TO_LOW }, Self::Field>,
        _commitments: Vec<Self::Commitment>,
        _proof: &Self::Proof,
        _transcript: &mut T,
    ) -> Result<Self::HidingCommitment, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        Err(transparent_zk_error())
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests unwrap successful PCS operations")]
    #![expect(clippy::expect_used, reason = "tests assert successful proof setup")]
    #![expect(clippy::indexing_slicing, reason = "tests index fixture data")]

    use super::*;
    use crate::adapters::{append_verifier_setup, AkitaBackendFlavor};
    use jolt_transcript::Blake2bTranscript;

    #[test]
    fn setup_key_transcript_binds_backend_shape() {
        let setup = AkitaVerifierSetup {
            max_num_vars: 4,
            max_num_polys_per_commitment_group: 1,
            max_total_batch_polys: 1,
            default_layout_digest: [7; 32],
            one_hot_k: AKITA_ONE_HOT_K256,
            backend_cache: Default::default(),
        };
        let mut baseline = Blake2bTranscript::<AkitaField>::new(b"akita-setup-key-test");
        let initial_state = baseline.state();

        append_verifier_setup(&mut baseline, &setup, AkitaBackendFlavor::Dense);
        assert_ne!(baseline.state(), initial_state);

        let mut same = Blake2bTranscript::<AkitaField>::new(b"akita-setup-key-test");
        append_verifier_setup(&mut same, &setup, AkitaBackendFlavor::Dense);
        assert_eq!(baseline.state(), same.state());

        let mut flavor_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-setup-key-test");
        append_verifier_setup(&mut flavor_transcript, &setup, AkitaBackendFlavor::OneHot);
        assert_ne!(baseline.state(), flavor_transcript.state());

        let mut changed_shape = setup.clone();
        changed_shape.max_num_vars = 5;
        let mut shape_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-setup-key-test");
        append_verifier_setup(
            &mut shape_transcript,
            &changed_shape,
            AkitaBackendFlavor::Dense,
        );
        assert_ne!(baseline.state(), shape_transcript.state());

        let mut changed_digest = setup;
        changed_digest.default_layout_digest = [8; 32];
        let mut digest_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-setup-key-test");
        append_verifier_setup(
            &mut digest_transcript,
            &changed_digest,
            AkitaBackendFlavor::Dense,
        );
        assert_ne!(baseline.state(), digest_transcript.state());

        let mut changed_k = changed_digest;
        changed_k.one_hot_k = AKITA_ONE_HOT_K16;
        let mut k_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-setup-key-test");
        append_verifier_setup(&mut k_transcript, &changed_k, AkitaBackendFlavor::Dense);
        assert_ne!(digest_transcript.state(), k_transcript.state());
    }

    fn one_hot_roundtrip(one_hot_k: usize) {
        let num_vars = one_hot_k.ilog2() as usize + 8;
        let setup_params = AkitaSetupParams::one_hot_only(num_vars, 1, [4; 32], one_hot_k);
        let (prover_setup, verifier_setup) = AkitaScheme::setup(setup_params).unwrap();
        let indices = (0..256usize)
            .map(|row| {
                if row == 2 {
                    None
                } else {
                    Some((row % one_hot_k) as u8)
                }
            })
            .collect::<Vec<_>>();
        let polynomial = OneHotPolynomial::new(one_hot_k, indices);
        let (commitment, hint) = AkitaScheme::commit_one_hot_group(
            &prover_setup,
            [4; 32],
            std::slice::from_ref(&polynomial),
        )
        .unwrap();
        assert_eq!(commitment.one_hot_k(), one_hot_k);

        let point = vec![AkitaField::from_u64(3); num_vars];
        let value = polynomial.evaluate(&point);
        let statement = vec![VerifierOpeningClaim {
            commitment: commitment.clone(),
            evaluation: EvaluationClaim::new(point, value),
        }];
        let mut prover_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-one-hot-k");
        let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement.clone(),
            vec![&polynomial],
            hint,
            &mut prover_transcript,
        )
        .unwrap();
        let mut verifier_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-one-hot-k");
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &statement,
            &proof,
            &mut verifier_transcript,
        )
        .unwrap();
        assert_eq!(prover_transcript.state(), verifier_transcript.state());

        let mut wrong_k_statement = statement;
        wrong_k_statement[0].commitment.one_hot_k = if one_hot_k == AKITA_ONE_HOT_K16 {
            AKITA_ONE_HOT_K256
        } else {
            AKITA_ONE_HOT_K16
        };
        let mut verifier_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-one-hot-k");
        let _ = <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &wrong_k_statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect_err("commitment K must match verifier setup K");
    }

    #[test]
    fn one_hot_k16_roundtrip() {
        one_hot_roundtrip(AKITA_ONE_HOT_K16);
    }

    #[test]
    fn one_hot_k256_roundtrip() {
        one_hot_roundtrip(AKITA_ONE_HOT_K256);
    }

    /// A serde roundtrip drops the primed key cache; the transported setup
    /// must re-derive the same backend key from its shape.
    #[test]
    fn serde_transported_setup_rederives_the_backend_key() {
        let (_, verifier_setup) =
            AkitaScheme::setup(AkitaSetupParams::new(14, 1, [3; 32])).unwrap();
        let json = serde_json::to_string(&verifier_setup).unwrap();
        let transported: AkitaVerifierSetup = serde_json::from_str(&json).unwrap();
        assert_eq!(transported, verifier_setup);
        let rederived = transported
            .backend_verifier(AkitaBackendFlavor::Dense)
            .expect("shape-only setup re-derives its backend key");
        let primed = verifier_setup
            .backend_verifier(AkitaBackendFlavor::Dense)
            .expect("primed cache returns the built key");
        assert_eq!(
            serialize_akita(rederived).unwrap(),
            serialize_akita(primed).unwrap(),
            "re-derived backend key must match the primed one"
        );
    }

    #[test]
    fn direct_opening_requires_statement_commitment_layout_digest() {
        let setup_params = AkitaSetupParams::new(14, 1, [7; 32]);
        let (prover_setup, verifier_setup) = AkitaScheme::setup(setup_params).unwrap();
        let polynomial = Polynomial::new(
            (0..(1u64 << 14))
                .map(|i| AkitaField::from_u64(2 + 5 * i))
                .collect(),
        );
        let commitment_digest = [9; 32];
        let (commitment, hint) = AkitaScheme::commit_group(
            &prover_setup,
            commitment_digest,
            std::slice::from_ref(&polynomial),
        )
        .expect("direct commitment may use its own layout digest");
        assert_eq!(commitment.layout_digest, commitment_digest);

        let point = (3..17).map(AkitaField::from_u64).collect::<Vec<_>>();
        let claim = polynomial.evaluate(&point);
        let statement = vec![VerifierOpeningClaim {
            commitment: commitment.clone(),
            evaluation: EvaluationClaim::new(point.clone(), claim),
        }];

        let mut prover_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-direct-layout");
        let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement.clone(),
            vec![&polynomial],
            hint,
            &mut prover_transcript,
        )
        .expect("direct proof should prove");

        let mut verifier_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-direct-layout");
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect("direct proof should verify");
        assert_eq!(prover_transcript.state(), verifier_transcript.state());

        let mut changed_commitment_statement = statement.clone();
        changed_commitment_statement[0].commitment.layout_digest = [15; 32];
        let mut verifier_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-direct-layout");
        let _error = <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &changed_commitment_statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect_err("changed direct commitment digest should reject");

        let mut changed_setup = verifier_setup;
        changed_setup.default_layout_digest = commitment_digest;
        let mut verifier_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-direct-layout");
        let _error = <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &changed_setup,
            &statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect_err("direct commitment layout must not be accepted through setup default");
    }
}

/// Timed comparison of the two OneHotTrace commitment formats at production shape:
/// one sparse-unit union polynomial (`slots` slots × `2^(8+log_t)` cells)
/// versus one batched one-hot group (`slots` polynomials of `8+log_t`
/// variables each) — commit + batched open + verify for both.
#[cfg(test)]
mod flavor_bench {
    #![expect(
        clippy::unwrap_used,
        reason = "bench unwraps successful PCS operations"
    )]
    #![expect(clippy::print_stderr, reason = "bench reports timings to stderr")]
    #![expect(clippy::indexing_slicing, reason = "bench indexes fixture data")]
    #![expect(
        clippy::unimplemented,
        reason = "the bench stand-in exposes only the one-hot polynomial interface"
    )]

    use super::*;
    use jolt_transcript::Blake2bTranscript;
    use std::time::Instant;

    const LOG_K: usize = 8;
    const K: usize = 1 << LOG_K;

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Big-endian eq table: index bit of weight `2^(len-1-j)` pairs with
    /// `point[j]`.
    fn eq_table(point: &[AkitaField]) -> Vec<AkitaField> {
        let one = AkitaField::from_u64(1);
        let mut table = vec![one];
        for &p in point {
            let one_minus = one - p;
            let mut next = Vec::with_capacity(table.len() * 2);
            for &w in &table {
                next.push(w * one_minus);
                next.push(w * p);
            }
            table = next;
        }
        table
    }

    struct EqSplit {
        hi: Vec<AkitaField>,
        lo: Vec<AkitaField>,
        low_bits: usize,
        mask: usize,
    }

    impl EqSplit {
        fn new(point: &[AkitaField]) -> Self {
            let n = point.len();
            let low_bits = n / 2;
            Self {
                hi: eq_table(&point[..n - low_bits]),
                lo: eq_table(&point[n - low_bits..]),
                low_bits,
                mask: (1 << low_bits) - 1,
            }
        }

        fn weight(&self, index: usize) -> AkitaField {
            self.hi[index >> self.low_bits] * self.lo[index & self.mask]
        }
    }

    fn sparse_eval(poly: &dyn MultilinearPoly<AkitaField>, tables: &EqSplit) -> AkitaField {
        let mut acc = AkitaField::from_u64(0);
        poly.for_each_one(&mut |index| acc += tables.weight(index));
        acc
    }

    /// Bench stand-in for the packed union polynomial: unit-sparse over the
    /// slot-prefixed cell domain, exposing only the one-hot interface the
    /// sparse-unit commit path consumes.
    struct UnionSparse {
        num_vars: usize,
        ones: Vec<usize>,
    }

    impl MultilinearPoly<AkitaField> for UnionSparse {
        fn num_vars(&self) -> usize {
            self.num_vars
        }

        fn evaluate(&self, point: &[AkitaField]) -> AkitaField {
            let tables = EqSplit::new(point);
            let mut acc = AkitaField::from_u64(0);
            for &one in &self.ones {
                acc += tables.weight(one);
            }
            acc
        }

        fn for_each_row(&self, _sigma: usize, _f: &mut dyn FnMut(usize, &[AkitaField])) {
            unimplemented!("bench union polynomial exposes only the one-hot interface")
        }

        fn is_one_hot(&self) -> bool {
            true
        }

        fn for_each_one(&self, f: &mut dyn FnMut(usize)) {
            for &one in &self.ones {
                f(one);
            }
        }
    }

    #[test]
    fn sparse_eval_matches_the_trait_evaluation_convention() {
        let mut state = 7;
        let indices: Vec<Option<u8>> = (0..16)
            .map(|_| Some((splitmix(&mut state) % 4) as u8))
            .collect();
        let poly = OneHotPolynomial::new(4, indices);
        let point: Vec<AkitaField> = (0..poly.num_vars())
            .map(|_| AkitaField::from_u64(splitmix(&mut state)))
            .collect();
        let expected = MultilinearPoly::<AkitaField>::evaluate(&poly, &point);
        assert_eq!(sparse_eval(&poly, &EqSplit::new(&point)), expected);
    }

    #[test]
    #[ignore = "release-only setup-split probe, run explicitly"]
    fn setup_cost_split_by_flavor() {
        use crate::adapters::{AkitaBackendScheme, AkitaOneHotK256BackendScheme};
        let num_vars: usize = std::env::var("BENCH_VARS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(28);
        let polys: usize = std::env::var("BENCH_SLOTS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(30);
        let start = Instant::now();
        let dense = AkitaBackendScheme::setup_prover(num_vars, polys).unwrap();
        eprintln!("dense setup ({num_vars},{polys}): {:.2?}", start.elapsed());
        drop(dense);
        let start = Instant::now();
        let one_hot = AkitaOneHotK256BackendScheme::setup_prover(num_vars, polys).unwrap();
        eprintln!(
            "one-hot setup ({num_vars},{polys}): {:.2?}",
            start.elapsed()
        );
        drop(one_hot);
    }

    #[test]
    #[ignore = "release-only flavor bench, run explicitly"]
    fn flavor_bench_sparse_union_vs_batched_one_hot() {
        let log_t: usize = std::env::var("BENCH_LOG_T")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(20);
        let slots: usize = std::env::var("BENCH_SLOTS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(16);
        let t = 1usize << log_t;
        let cell_vars = LOG_K + log_t;
        let union_vars = cell_vars + slots.next_power_of_two().ilog2() as usize;
        let mut state = 0x1234_5678;

        // Selected rows for each slot; the last slot uses only two rows.
        let slot_indices: Vec<Vec<Option<u8>>> = (0..slots)
            .map(|slot| {
                (0..t)
                    .map(|_| {
                        let r = splitmix(&mut state);
                        if slot == slots - 1 {
                            Some((r & 1) as u8)
                        } else {
                            Some((r & 0xFF) as u8)
                        }
                    })
                    .collect()
            })
            .collect();

        // Batched one-hot group.
        let skip_one_hot = std::env::var("BENCH_SKIP_ONEHOT").is_ok();
        if !skip_one_hot {
            let start = Instant::now();
            let (prover_setup, verifier_setup) =
                AkitaScheme::setup(AkitaSetupParams::one_hot_only(cell_vars, slots, [1; 32], K))
                    .unwrap();
            eprintln!(
                "one-hot setup ({cell_vars} vars, {slots} polys): {:.2?}",
                start.elapsed()
            );
            let polys: Vec<OneHotPolynomial> = slot_indices
                .iter()
                .map(|indices| OneHotPolynomial::new(K, indices.clone()))
                .collect();
            let start = Instant::now();
            let (commitment, hint) =
                AkitaScheme::commit_one_hot_group(&prover_setup, [2; 32], &polys).unwrap();
            eprintln!("one-hot commit: {:.2?}", start.elapsed());

            let point: Vec<AkitaField> = (0..cell_vars)
                .map(|_| AkitaField::from_u64(splitmix(&mut state)))
                .collect();
            let tables = EqSplit::new(&point);
            let statement: Vec<VerifierOpeningClaim<AkitaField, AkitaCommitment>> = polys
                .iter()
                .map(|poly| VerifierOpeningClaim {
                    commitment: commitment.clone(),
                    evaluation: EvaluationClaim::new(point.clone(), sparse_eval(poly, &tables)),
                })
                .collect();
            let poly_refs: AkitaNativeBatchPolynomials<'_> = polys
                .iter()
                .map(|poly| poly as &dyn MultilinearPoly<AkitaField>)
                .collect();
            let mut prover_transcript = Blake2bTranscript::<AkitaField>::new(b"flavor-bench");
            let start = Instant::now();
            let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
                &prover_setup,
                statement.clone(),
                poly_refs,
                hint,
                &mut prover_transcript,
            )
            .unwrap();
            eprintln!("one-hot batched open: {:.2?}", start.elapsed());
            let mut verifier_transcript = Blake2bTranscript::<AkitaField>::new(b"flavor-bench");
            let start = Instant::now();
            <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
                &verifier_setup,
                &statement,
                &proof,
                &mut verifier_transcript,
            )
            .unwrap();
            eprintln!("one-hot verify: {:.2?}", start.elapsed());
            assert_eq!(prover_transcript.state(), verifier_transcript.state());
        }

        if std::env::var("BENCH_SKIP_UNION").is_ok() {
            return;
        }

        // Sparse-unit union of the same content.
        let start = Instant::now();
        let (prover_setup, verifier_setup) =
            AkitaScheme::setup(AkitaSetupParams::new(union_vars, 1, [1; 32])).unwrap();
        eprintln!("union setup ({union_vars} vars): {:.2?}", start.elapsed());
        let mut ones = Vec::with_capacity(slots * t);
        for (slot, indices) in slot_indices.iter().enumerate() {
            for (cycle, &row) in indices.iter().enumerate() {
                let row = row.unwrap() as usize;
                ones.push((slot << cell_vars) | (row << log_t) | cycle);
            }
        }
        ones.sort_unstable();
        let union = UnionSparse {
            num_vars: union_vars,
            ones,
        };
        let start = Instant::now();
        let (commitment, hint) =
            <AkitaScheme as CommitmentScheme>::commit(&union, &prover_setup).unwrap();
        eprintln!("union commit: {:.2?}", start.elapsed());

        let point: Vec<AkitaField> = (0..union_vars)
            .map(|_| AkitaField::from_u64(splitmix(&mut state)))
            .collect();
        let value = union.evaluate(&point);
        let statement = vec![VerifierOpeningClaim {
            commitment: commitment.clone(),
            evaluation: EvaluationClaim::new(point.clone(), value),
        }];
        let mut prover_transcript = Blake2bTranscript::<AkitaField>::new(b"flavor-bench");
        let start = Instant::now();
        let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement.clone(),
            vec![&union as &dyn MultilinearPoly<AkitaField>],
            hint,
            &mut prover_transcript,
        )
        .unwrap();
        eprintln!("union open: {:.2?}", start.elapsed());
        let mut verifier_transcript = Blake2bTranscript::<AkitaField>::new(b"flavor-bench");
        let start = Instant::now();
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &statement,
            &proof,
            &mut verifier_transcript,
        )
        .unwrap();
        eprintln!("union verify: {:.2?}", start.elapsed());
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }
}
