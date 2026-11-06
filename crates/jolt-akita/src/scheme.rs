use crate::AkitaScheduleArtifacts;
use akita_params::PrecommittedGroupProfiles;
use akita_pcs::custom_source::RootPolyMeta;
#[cfg(feature = "zk-prover")]
use akita_pcs::HidingRandomness;
use akita_pcs::{AkitaError, CommitOutput, CpuBackend, GroupContext, SourceHandle};
use akita_transcript::TranscriptSponge;
use akita_types::PendingEof;
use jolt_crypto::Commitment;
use jolt_openings::{
    BatchOpeningScheme, CommitmentGroupRole, CommitmentScheme, EvaluationClaim, GroupBatchOpening,
    GroupOpeningClaim, GroupOpeningWithHint, OpeningsError, TaggedGroupOpeningClaim,
    TransparentObjectSetup, VerifierOpeningClaim,
};
use jolt_poly::{MultilinearPoly, OneHotPolynomial, Polynomial};
use jolt_transcript::{FieldTranscript, ProverState, VerifierState};
#[cfg(feature = "zk-prover")]
use jolt_zk::masking::{AffineRow, CommittedPads, PadLayout, ResidualRelation, ZkError};
#[cfg(feature = "zk-prover")]
use jolt_zk::whiteout::WhiteoutPadOpening;
#[cfg(feature = "zk-prover")]
use jolt_zk::whiteout::{DenseSource, MaskedOpeningShape, PrecommittedShape};
#[cfg(feature = "zk-prover")]
use rand_core::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use std::{
    collections::{btree_map::Entry, BTreeMap},
    sync::Arc,
};

use crate::adapters::{
    akita_error, akita_ordered_evaluations, commit_failed, dense_polynomials, invalid_batch,
    invalid_setup, one_hot_polynomial, owned_one_hot_polynomial, serialize_akita,
    validate_one_hot_k, with_backend_pool, AkitaBackend, AkitaBackendCommitment,
    AkitaBackendDensePoly, AkitaBackendExtField, AkitaBackendFlavor, AkitaBackendHint,
    AkitaBackendOneHotPoly, AkitaBatchProof, AkitaCommitment, AkitaField, AkitaHintSource,
    AkitaLayoutDigest, AkitaProverHint, AkitaProverSetup, AkitaSetupFlavor, AkitaSetupParams,
    AkitaVerifierSetup, FullWidthBackendSetup, AKITA_SOURCE_RING_DIMENSION,
};
use crate::configs::AkitaChunkProfile;
use crate::native_batching::{AkitaNativeBatchPolynomials, AkitaNativeBatching};
use crate::one_hot_family::with_one_hot_family;
use crate::trace_onehot::{TraceOneHotColumn, TraceOneHotRows};
#[cfg(feature = "zk-prover")]
use crate::whiteout::{ProverGroup, WhiteoutResidual};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaScheme;

/// Jolt's prover transcript over the sponge Akita's opening runs in.
pub type AkitaProverTranscript = ProverState<TranscriptSponge>;
/// Jolt's verifier transcript over the sponge Akita's opening runs in.
pub type AkitaVerifierTranscript<'a> = VerifierState<'a, TranscriptSponge>;

fn split_commit_output(
    output: CommitOutput<AkitaField, AkitaBackendExtField>,
) -> (AkitaBackendCommitment, AkitaBackendHint) {
    (output.committed_group, output.private_handle)
}

/// Commits one imported one-hot source group under the setup's K-specific
/// scheme; `context` names the precommitted groups the row is keyed on.
fn commit_one_hot_source(
    setup: &AkitaProverSetup,
    backend: &AkitaBackend,
    source: &SourceHandle<AkitaField, AkitaBackendExtField>,
    context: GroupContext<'_>,
) -> Result<(AkitaBackendCommitment, AkitaBackendHint), AkitaError> {
    let scheme = setup
        .verifier
        .one_hot_scheme()
        .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
    with_one_hot_family!(scheme scheme, |scheme| backend.commit(
        scheme.schedules(),
        source,
        context
    ))
    .map(split_commit_output)
}

/// Prover seam for committing the native trace columns directly from selected one-hot rows.
pub trait TraceOneHotCommitment: CommitmentScheme {
    /// Witness chunk profile fixed by preprocessing.
    fn akita_chunk_profile(setup: &Self::VerifierSetup) -> AkitaChunkProfile;

    fn commit_trace_one_hot(
        setup: &Self::ProverSetup,
        layout_digest: [u8; 32],
        rows: Arc<dyn TraceOneHotRows>,
        group_hints: &[&Self::OpeningHint],
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError>;

    /// Releases backend state that can be rebuilt before the opening proof.
    fn release_post_commit_residency(setup: &Self::ProverSetup) -> Result<(), OpeningsError>;

    /// Whiteout: commits `polynomial`, one dense polynomial, hiding as a
    /// precommitted group of `shape` on the setup's Whiteout backend, the one
    /// that opens it (Akita `specs/whiteout-akita.md` §3.5). Per-proof groups and statement-fixed
    /// ones (trusted advice, a committed program) alike: a hiding commitment
    /// cannot move to another backend.
    #[cfg(feature = "zk-prover")]
    fn whiteout_commit_precommitted<P: MultilinearPoly<Self::Field> + ?Sized>(
        setup: &Self::ProverSetup,
        shape: PrecommittedShape,
        polynomial: &P,
        layout_digest: [u8; 32],
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError>;

    /// Whiteout: rebuilds the hint of the hiding precommitted group
    /// `commitment` of `shape` from `polynomial` and the randomness it was
    /// made with, so a process other than the committing one can open it
    /// (`specs/whiteout-jolt-integration.md` §5). Fails unless they rebuild exactly `commitment`, which makes
    /// this the check of `commitment` against (`polynomial`, `randomness`)
    /// too.
    #[cfg(feature = "zk-prover")]
    fn whiteout_recommit_precommitted<P: MultilinearPoly<Self::Field> + ?Sized>(
        setup: &Self::ProverSetup,
        shape: PrecommittedShape,
        polynomial: &P,
        layout_digest: [u8; 32],
        commitment: &Self::Output,
        randomness: &HidingRandomness,
    ) -> Result<Self::OpeningHint, OpeningsError>;

    /// Whiteout: the randomness a hiding precommitted group's hint was
    /// committed with, for
    /// [`whiteout_recommit_precommitted`](Self::whiteout_recommit_precommitted).
    #[cfg(feature = "zk-prover")]
    fn whiteout_hiding_randomness(
        hint: &Self::OpeningHint,
    ) -> Result<HidingRandomness, OpeningsError>;

    /// Whiteout: samples, commits and sends the pads of `layout` for the
    /// masked opening of `shape` (spec §2, row 2).
    #[cfg(feature = "zk-prover")]
    fn whiteout_commit_pads<R: RngCore + CryptoRng>(
        setup: &Self::ProverSetup,
        shape: &MaskedOpeningShape,
        layout: PadLayout,
        transcript: &mut AkitaProverTranscript,
        rng: &mut R,
    ) -> Result<WhiteoutPads<Self::Field>, ZkError>;

    /// Whiteout: proves the masked levels of the opening of `shape` through
    /// the cut: the `precommitted` groups in canonical order, then `trace`,
    /// their claims stated by `forms` in the same order (spec §2, row 5).
    /// Returns the residual opening and the masked levels' auxiliary values,
    /// which lead the auxiliary block.
    #[cfg(feature = "zk-prover")]
    #[expect(
        clippy::too_many_arguments,
        reason = "the opening's groups, their forms, and the pads"
    )]
    fn whiteout_prove_masked(
        setup: &Self::ProverSetup,
        shape: &MaskedOpeningShape,
        precommitted: Vec<GroupOpeningWithHint<Self::Field, Self::Output, Self::OpeningHint>>,
        trace: GroupOpeningClaim<Self::Field, Self::Output>,
        hint: Self::OpeningHint,
        forms: &[AffineRow<Self::Field>],
        pads: &mut WhiteoutPads<Self::Field>,
        transcript: &mut AkitaProverTranscript,
    ) -> Result<(WhiteoutResidual, Vec<Self::Field>), ZkError>;

    /// Whiteout: proves `relation` by the Σ-protocol and the response proof
    /// (spec §2, rows 7 and 8).
    #[cfg(feature = "zk-prover")]
    fn whiteout_prove_tail<R: RngCore + CryptoRng + Send>(
        setup: &Self::ProverSetup,
        shape: &MaskedOpeningShape,
        relation: &ResidualRelation<Self::Field>,
        residual: &WhiteoutResidual,
        pads: &WhiteoutPads<Self::Field>,
        transcript: &mut AkitaProverTranscript,
        rng: &mut R,
    ) -> Result<(), ZkError>;
}

/// Whiteout's committed pads.
#[cfg(feature = "zk-prover")]
pub type WhiteoutPads<F> = CommittedPads<F, WhiteoutPadOpening<F>>;

/// Strictly ascending roles make the ordered commitment group list
/// unambiguous and forbid duplicate or permuted groups.
pub(crate) fn validate_group_order(
    roles: impl IntoIterator<Item = CommitmentGroupRole>,
) -> Result<(), OpeningsError> {
    let mut previous: Option<CommitmentGroupRole> = None;
    for role in roles {
        if let Some(previous) = previous {
            if role.order() <= previous.order() {
                return Err(invalid_batch(format!(
                    "Akita commitment groups must be in canonical ascending order, found {} after {}",
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
        let poly_count = backend_polynomials.len();
        let (backend_commitment, backend_hint) =
            Self::commit_one_hot_backend(setup, backend_polynomials)?;
        Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintSource::OneHot {
                poly_count,
                one_hot_k: setup.one_hot_k(),
            },
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
        let poly_count = backend_polynomials.len();
        let (backend_commitment, backend_hint) =
            Self::commit_one_hot_backend(setup, backend_polynomials)?;
        Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintSource::OneHot {
                poly_count,
                one_hot_k: setup.one_hot_k(),
            },
        )
    }

    /// Commits the native trace batch without constructing per-column
    /// index vectors or Akita's generic one-hot block representation.
    pub fn commit_trace_one_hot(
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
        rows: Arc<dyn TraceOneHotRows>,
        group_hints: &[&AkitaProverHint],
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let source = TraceOneHotColumn::new(setup.one_hot_k(), AKITA_SOURCE_RING_DIMENSION, rows)
            .map_err(commit_failed)?;
        let num_vars = RootPolyMeta::num_vars(
            source
                .first()
                .ok_or_else(|| commit_failed("empty trace batch"))?,
        );
        let poly_count = source.len();
        Self::validate_commit_shape(setup, num_vars, poly_count)?;
        let one_hot_k = setup.one_hot_k();
        #[cfg(all(feature = "zk", not(feature = "zk-prover")))]
        if setup.verifier.whiteout().is_some() {
            return Err(commit_failed(
                "a Whiteout setup commits hiding only with jolt-akita's `zk-prover`",
            ));
        }
        #[cfg(feature = "zk-prover")]
        if setup.verifier.whiteout().is_some() {
            // Whiteout commits the trace group hiding, under the
            // zero-knowledge row keyed on its precommitted groups.
            let precommitted = group_hints
                .iter()
                .map(|hint| match hint.source {
                    AkitaHintSource::WhiteoutPrecommitted { shape } => Ok(shape),
                    AkitaHintSource::Dense { .. }
                    | AkitaHintSource::OneHot { .. }
                    | AkitaHintSource::TraceOneHot { .. } => Err(invalid_batch(
                        "a Whiteout trace group folds only hiding precommitted groups",
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let shape = crate::whiteout::opening_shape(
                &setup.verifier,
                &MaskedOpeningShape {
                    precommitted,
                    num_vars,
                    num_polys: poly_count,
                },
            )
            .map_err(invalid_setup)?;
            let row = shape.row();
            let (backend_commitment, backend_hint) = with_backend_pool(|| {
                let source = row.catalog().backend()?.import_source(source)?;
                row.commit_hiding(&source)
            })
            .map_err(commit_failed)?;
            return Self::package_commitment(
                layout_digest,
                num_vars,
                backend_commitment,
                backend_hint,
                AkitaHintSource::TraceOneHot {
                    poly_count,
                    one_hot_k,
                },
            );
        }
        let profiles = if group_hints.is_empty() {
            None
        } else {
            Some(Self::group_profiles(setup, group_hints)?)
        };
        let (_, backend) = setup.one_hot_backend()?;
        let (backend_commitment, backend_hint) = with_backend_pool(|| {
            let source = backend.import_source(source)?;
            let context = profiles.as_ref().map_or_else(
                GroupContext::scheduler_without_precommitted_groups,
                GroupContext::scheduler_with_precommitted_groups,
            );
            commit_one_hot_source(setup, backend, &source, context)
        })
        .map_err(commit_failed)?;
        Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintSource::TraceOneHot {
                poly_count,
                one_hot_k,
            },
        )
    }

    /// Whiteout: commits one dense polynomial hiding as a precommitted group
    /// of `shape`, on the setup's Whiteout backend: with fresh randomness,
    /// or, given `reopen`, rebuilding exactly that commitment from the
    /// randomness it was made with.
    #[cfg(feature = "zk-prover")]
    fn commit_whiteout_precommitted<P: MultilinearPoly<AkitaField> + ?Sized>(
        setup: &AkitaProverSetup,
        shape: PrecommittedShape,
        polynomial: &P,
        layout_digest: [u8; 32],
        reopen: Option<(&AkitaCommitment, &HidingRandomness)>,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let num_vars = polynomial.num_vars();
        if num_vars != shape.num_vars {
            return Err(invalid_batch(format!(
                "a Whiteout precommitted group of {shape:?} cannot hold a {num_vars}-variable polynomial"
            )));
        }
        let whiteout = crate::whiteout::whiteout(&setup.verifier).map_err(invalid_setup)?;
        let catalog = whiteout
            .catalog(
                setup.max_num_vars(),
                setup.max_num_polys_per_commitment_group(),
            )
            .map_err(invalid_setup)?;
        let dense = vec![AkitaBackendDensePoly::from_field_evals(
            num_vars,
            akita_ordered_evaluations(polynomial)?,
        )
        .map_err(akita_error)?];
        let expected = reopen
            .map(|(commitment, randomness)| {
                crate::adapters::deserialize_akita(
                    &commitment.serialized_backend_bytes,
                    &commitment.backend_coeff_len,
                )
                .map(|payload| (payload, randomness))
            })
            .transpose()?;
        let (backend_commitment, backend_hint) = with_backend_pool(|| {
            let source = catalog.backend()?.import_source(dense)?;
            match &expected {
                None => catalog.commit_precommitted(shape, &source),
                Some((payload, randomness)) => {
                    catalog.recommit_precommitted(shape, &source, randomness, payload)
                }
            }
        })
        .map_err(commit_failed)?;
        let (commitment, hint) = Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintSource::WhiteoutPrecommitted { shape },
        )?;
        // The backend matched the payload; the header must match too.
        if reopen.is_some_and(|(expected, _)| *expected != commitment) {
            return Err(invalid_batch(
                "the rebuilt commitment's header differs from the expected commitment",
            ));
        }
        Ok((commitment, hint))
    }

    fn commit_one_hot_backend(
        setup: &AkitaProverSetup,
        polynomials: Vec<AkitaBackendOneHotPoly>,
    ) -> Result<(AkitaBackendCommitment, AkitaBackendHint), OpeningsError> {
        let (_, backend) = setup.one_hot_backend()?;
        with_backend_pool(|| {
            let source = backend.import_source(polynomials)?;
            commit_one_hot_source(
                setup,
                backend,
                &source,
                GroupContext::scheduler_without_precommitted_groups(),
            )
        })
        .map_err(commit_failed)
    }

    /// Freezes the ordered group profiles the final trace group commits
    /// against. Order is the caller's canonical role order; the backend keys the
    /// grouped row on this exact sequence.
    fn group_profiles(
        setup: &AkitaProverSetup,
        group_hints: &[&AkitaProverHint],
    ) -> Result<PrecommittedGroupProfiles, OpeningsError> {
        if group_hints.is_empty() {
            return Err(invalid_batch(
                "Akita grouped trace opening requires at least one auxiliary group",
            ));
        }
        let required = group_hints
            .len()
            .checked_add(1)
            .ok_or_else(|| invalid_batch("Akita auxiliary group count overflows"))?;
        if setup.max_total_batch_polys() < required {
            return Err(invalid_batch(format!(
                "Akita grouped trace opening requires total polynomial capacity {required}, setup has {}",
                setup.max_total_batch_polys()
            )));
        }
        let mut profiles = Vec::with_capacity(group_hints.len());
        for hint in group_hints {
            if hint.commitment.backend_flavor != AkitaBackendFlavor::Dense
                || hint.commitment.poly_count != 1
                || !matches!(hint.source, AkitaHintSource::Dense { .. })
            {
                return Err(invalid_batch(
                    "Akita auxiliary group must contain one dense polynomial",
                ));
            }
            let (auxiliary_group, _) = hint.backend.as_ref().ok_or_else(|| {
                invalid_batch("Akita auxiliary group is missing backend opening data")
            })?;
            profiles.push(*auxiliary_group.profile());
        }
        PrecommittedGroupProfiles::from_profiles(profiles).map_err(akita_error)
    }

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

    fn package_commitment(
        layout_digest: AkitaLayoutDigest,
        num_vars: usize,
        backend_commitment: AkitaBackendCommitment,
        backend_hint: AkitaBackendHint,
        source: AkitaHintSource,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let backend_flavor = source.backend_flavor();
        let one_hot_k = match source {
            AkitaHintSource::Dense { .. } => 0,
            #[cfg(feature = "zk")]
            AkitaHintSource::WhiteoutPrecommitted { .. } => 0,
            AkitaHintSource::OneHot { one_hot_k, .. }
            | AkitaHintSource::TraceOneHot { one_hot_k, .. } => one_hot_k,
        };
        let commitment = AkitaCommitment {
            backend_flavor,
            layout_digest,
            num_vars,
            poly_count: source.len(),
            one_hot_k,
            backend_coeff_len: backend_commitment.rows().coeff_len(),
            serialized_backend_bytes: serialize_akita(backend_commitment.commitment())?,
        };
        Ok((
            commitment.clone(),
            AkitaProverHint {
                commitment,
                backend: Some((backend_commitment, backend_hint)),
                source,
            },
        ))
    }

    fn commit_dense_backend(
        setup: &AkitaProverSetup,
        layout_digest: AkitaLayoutDigest,
        num_vars: usize,
        dense: Vec<AkitaBackendDensePoly>,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let poly_count = dense.len();
        let (_, backend) = setup.dense_backend()?;
        let (backend_commitment, backend_hint) = with_backend_pool(|| {
            let source = backend.import_source(dense)?;
            backend.commit(
                setup
                    .verifier
                    .dense_scheme()
                    .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?
                    .schedules(),
                &source,
                GroupContext::scheduler_without_precommitted_groups(),
            )
        })
        .map(split_commit_output)
        .map_err(commit_failed)?;
        Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintSource::Dense { poly_count },
        )
    }
}

impl TraceOneHotCommitment for AkitaScheme {
    fn akita_chunk_profile(setup: &Self::VerifierSetup) -> AkitaChunkProfile {
        setup.akita_chunk_profile()
    }

    fn commit_trace_one_hot(
        setup: &Self::ProverSetup,
        layout_digest: [u8; 32],
        rows: Arc<dyn TraceOneHotRows>,
        group_hints: &[&Self::OpeningHint],
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
        Self::commit_trace_one_hot(setup, layout_digest, rows, group_hints)
    }

    fn release_post_commit_residency(setup: &Self::ProverSetup) -> Result<(), OpeningsError> {
        setup.release_post_commit_ntt_residency()
    }

    #[cfg(feature = "zk-prover")]
    fn whiteout_commit_precommitted<P: MultilinearPoly<Self::Field> + ?Sized>(
        setup: &Self::ProverSetup,
        shape: PrecommittedShape,
        polynomial: &P,
        layout_digest: [u8; 32],
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
        Self::commit_whiteout_precommitted(setup, shape, polynomial, layout_digest, None)
    }

    #[cfg(feature = "zk-prover")]
    fn whiteout_recommit_precommitted<P: MultilinearPoly<Self::Field> + ?Sized>(
        setup: &Self::ProverSetup,
        shape: PrecommittedShape,
        polynomial: &P,
        layout_digest: [u8; 32],
        commitment: &Self::Output,
        randomness: &HidingRandomness,
    ) -> Result<Self::OpeningHint, OpeningsError> {
        Self::commit_whiteout_precommitted(
            setup,
            shape,
            polynomial,
            layout_digest,
            Some((commitment, randomness)),
        )
        .map(|(_, hint)| hint)
    }

    #[cfg(feature = "zk-prover")]
    fn whiteout_hiding_randomness(
        hint: &Self::OpeningHint,
    ) -> Result<HidingRandomness, OpeningsError> {
        if !matches!(hint.source, AkitaHintSource::WhiteoutPrecommitted { .. }) {
            return Err(invalid_batch(
                "only a hiding precommitted group's hint exports its randomness",
            ));
        }
        let (_, handle) = hint
            .backend
            .as_ref()
            .ok_or_else(|| invalid_batch("a precommitted hint has no backend payload"))?;
        handle.hiding_randomness().map_err(akita_error)
    }

    #[cfg(feature = "zk-prover")]
    fn whiteout_commit_pads<R: RngCore + CryptoRng>(
        setup: &Self::ProverSetup,
        shape: &MaskedOpeningShape,
        layout: PadLayout,
        transcript: &mut AkitaProverTranscript,
        rng: &mut R,
    ) -> Result<WhiteoutPads<AkitaField>, ZkError> {
        let shape = crate::whiteout::opening_shape(&setup.verifier, shape)?;
        shape.tail().commit_pads(layout, transcript, rng)
    }

    #[cfg(feature = "zk-prover")]
    fn whiteout_prove_masked(
        setup: &Self::ProverSetup,
        shape: &MaskedOpeningShape,
        precommitted: Vec<GroupOpeningWithHint<AkitaField, AkitaCommitment, AkitaProverHint>>,
        trace: GroupOpeningClaim<AkitaField, AkitaCommitment>,
        hint: AkitaProverHint,
        forms: &[AffineRow<AkitaField>],
        pads: &mut WhiteoutPads<AkitaField>,
        transcript: &mut AkitaProverTranscript,
    ) -> Result<(WhiteoutResidual, Vec<AkitaField>), ZkError> {
        if precommitted.len() != shape.precommitted.len() {
            return Err(AkitaError::InvalidInput(
                "the precommitted groups disagree with the opening's shape".into(),
            )
            .into());
        }
        let opening = crate::whiteout::opening_shape(&setup.verifier, shape)?;
        let mut groups = Vec::with_capacity(precommitted.len() + 1);
        for ((entry, hint), expected) in precommitted.into_iter().zip(&shape.precommitted) {
            if !matches!(
                hint.source,
                AkitaHintSource::WhiteoutPrecommitted { shape } if shape == *expected
            ) {
                return Err(AkitaError::InvalidInput(format!(
                    "the {} hint is not a hiding precommitted group of {expected:?}",
                    entry.role.diagnostic_name()
                ))
                .into());
            }
            let (commitment, handle) = hint.backend.ok_or_else(|| {
                AkitaError::InvalidInput("a precommitted hint has no backend payload".into())
            })?;
            groups.push(ProverGroup {
                commitment,
                handle,
                point: crate::adapters::reverse_point(&entry.claim.point),
                evaluations: entry.claim.evaluations,
            });
        }
        let (commitment, handle) = hint.backend.ok_or_else(|| {
            AkitaError::InvalidInput("the trace hint has no backend payload".into())
        })?;
        groups.push(ProverGroup {
            commitment,
            handle,
            point: crate::adapters::reverse_point(&trace.point),
            evaluations: trace.evaluations,
        });
        let compression = opening.tail().compression(pads.layout())?;
        let (opening, auxiliary) =
            opening
                .row()
                .prove_masked(groups, forms, pads, transcript, compression.as_ref())?;
        Ok((WhiteoutResidual { opening }, auxiliary))
    }

    #[cfg(feature = "zk-prover")]
    fn whiteout_prove_tail<R: RngCore + CryptoRng + Send>(
        setup: &Self::ProverSetup,
        shape: &MaskedOpeningShape,
        relation: &ResidualRelation<AkitaField>,
        residual: &WhiteoutResidual,
        pads: &WhiteoutPads<AkitaField>,
        transcript: &mut AkitaProverTranscript,
        rng: &mut R,
    ) -> Result<(), ZkError> {
        let shape = crate::whiteout::opening_shape(&setup.verifier, shape)?;
        shape
            .tail()
            .prove(relation, &residual.opening, pads, transcript, rng)
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
        let verifier = Self::verifier_setup_from_params(&params)?;
        let artifacts = &params.schedule_artifacts;
        let one_hot_log_k = validate_one_hot_k(params.one_hot_k).map_err(invalid_setup)?;
        let (backend_prover_setup, cpu_backend, backend_verifier_setup) =
            if params.flavor == AkitaSetupFlavor::OneHot {
                (None, None, None)
            } else {
                let scheme = verifier.dense_scheme()?;
                let backend_prover_setup = with_backend_pool(|| {
                    scheme.setup_prover(params.max_num_vars, params.max_total_batch_polys)
                })
                .map_err(invalid_setup)?;
                let cpu_backend = CpuBackend::new(Arc::clone(&backend_prover_setup.expanded))
                    .map_err(invalid_setup)?;
                let backend_verifier_setup =
                    with_backend_pool(|| scheme.setup_verifier(&backend_prover_setup))
                        .map_err(invalid_setup)?;
                (
                    Some(Arc::new(backend_prover_setup)),
                    Some(Arc::new(cpu_backend)),
                    Some(backend_verifier_setup),
                )
            };
        let (one_hot_backend_prover_setup, one_hot_cpu_backend, one_hot_backend_verifier_setup) =
            if params.max_num_vars >= one_hot_log_k && params.flavor != AkitaSetupFlavor::Dense {
                let backend_prover_setup = crate::adapters::one_hot_setup_prover(&verifier)?;
                let cpu_backend = CpuBackend::new(Arc::clone(&backend_prover_setup.expanded))
                    .map_err(invalid_setup)?;
                let backend_verifier_setup =
                    crate::adapters::one_hot_setup_verifier(&verifier, &backend_prover_setup)?;
                (
                    Some(Arc::new(backend_prover_setup)),
                    Some(Arc::new(cpu_backend)),
                    Some(backend_verifier_setup),
                )
            } else {
                (None, None, None)
            };
        let mut full_width = BTreeMap::new();
        if let Some(request) = &params.grouped_schedule {
            let arities = request.full_width_arities().collect::<Vec<_>>();
            if !arities.is_empty() {
                let scheme = artifacts.full_dense_scheme().map_err(invalid_setup)?;
                for num_vars in arities {
                    if let Entry::Vacant(entry) = full_width.entry(num_vars) {
                        let setup = with_backend_pool(|| scheme.setup_prover(num_vars, 1))
                            .map_err(invalid_setup)?;
                        let backend =
                            CpuBackend::new(Arc::clone(&setup.expanded)).map_err(invalid_setup)?;
                        let _ = entry.insert(Arc::new(FullWidthBackendSetup {
                            scheme: scheme.clone(),
                            backend,
                        }));
                    }
                }
            }
        }
        verifier.prime_backend_cache(backend_verifier_setup, one_hot_backend_verifier_setup)?;
        let prover = AkitaProverSetup {
            full_width,
            backend_prover_setup,
            cpu_backend,
            one_hot_backend_prover_setup,
            one_hot_cpu_backend,
            schedule_artifacts: params.schedule_artifacts,
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
        let num_vars = poly.num_vars();
        Self::validate_commit_shape(setup, num_vars, 1)?;
        if let Some(one_hot) = one_hot_polynomial(poly, setup.one_hot_k())? {
            let (backend_commitment, backend_hint) =
                Self::commit_one_hot_backend(setup, vec![one_hot])?;
            return Self::package_commitment(
                setup.default_layout_digest(),
                num_vars,
                backend_commitment,
                backend_hint,
                AkitaHintSource::OneHot {
                    poly_count: 1,
                    one_hot_k: setup.one_hot_k(),
                },
            );
        }

        if poly.is_one_hot() {
            return Err(invalid_batch(format!(
                "Akita one-hot commitments require row-major K={}",
                setup.one_hot_k()
            )));
        }

        let evals = akita_ordered_evaluations(poly)?;
        let dense =
            vec![AkitaBackendDensePoly::from_field_evals(num_vars, evals).map_err(akita_error)?];
        Self::commit_dense_backend(setup, setup.default_layout_digest(), num_vars, dense)
    }

    fn open<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl FieldTranscript<Self::Field>,
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
        transcript: &mut impl FieldTranscript<Self::Field>,
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
        let poly_count = backend_polynomials.len();
        let (backend_commitment, backend_hint) =
            Self::commit_one_hot_backend(setup, backend_polynomials)?;
        Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintSource::OneHot {
                poly_count,
                one_hot_k: setup.one_hot_k(),
            },
        )
    }
}

/// Jolt's trace batch, run inside Jolt's own sponge.
impl GroupBatchOpening for AkitaScheme {
    type Sponge = TranscriptSponge;
    type PendingEof = PendingEof;

    fn prove_batch(
        setup: &Self::ProverSetup,
        auxiliary_groups: Vec<GroupOpeningWithHint<Self::Field, Self::Output, Self::OpeningHint>>,
        final_group: GroupOpeningClaim<Self::Field, Self::Output>,
        final_hint: Self::OpeningHint,
        transcript: &mut ProverState<Self::Sponge>,
    ) -> Result<Self::Proof, OpeningsError> {
        AkitaNativeBatching::prove_trace_batch(
            setup,
            auxiliary_groups,
            final_group,
            final_hint,
            transcript,
        )
    }

    fn verify_batch(
        setup: &Self::VerifierSetup,
        auxiliary_groups: &[TaggedGroupOpeningClaim<Self::Field, Self::Output>],
        final_group: &GroupOpeningClaim<Self::Field, Self::Output>,
        proof: &Self::Proof,
        transcript: &mut VerifierState<'_, Self::Sponge>,
    ) -> Result<PendingEof, OpeningsError> {
        AkitaNativeBatching::verify_trace_batch(
            setup,
            auxiliary_groups,
            final_group,
            proof,
            transcript,
        )
    }

    fn check_eof(
        pending: PendingEof,
        transcript: VerifierState<'_, Self::Sponge>,
    ) -> Result<(), OpeningsError> {
        pending
            .check_eof(transcript)
            .map(|_| ())
            .map_err(|_| OpeningsError::VerificationFailed)
    }
}

impl TransparentObjectSetup for AkitaScheme {
    type SetupContext = Arc<AkitaScheduleArtifacts>;

    /// The singleton commitment-object setup convention for advice and direct
    /// committed-program objects: one polynomial at `num_vars`, seeded by the
    /// object plan's layout digest. Every bounded-dense object commits through
    /// the dense flavor, so the costly one-hot backend setup is never built.
    fn transparent_object_setup(
        context: &Self::SetupContext,
        num_vars: usize,
        layout_digest: [u8; 32],
    ) -> Result<(AkitaProverSetup, AkitaVerifierSetup), OpeningsError> {
        Self::setup(AkitaSetupParams::dense_only(
            num_vars,
            1,
            layout_digest,
            context.clone(),
        ))
    }

    fn commit_full_width_object<P: MultilinearPoly<Self::Field> + ?Sized>(
        setup: &Self::ProverSetup,
        poly: &P,
        layout_digest: [u8; 32],
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let num_vars = poly.num_vars();
        #[cfg(all(feature = "zk", not(feature = "zk-prover")))]
        if setup.verifier.whiteout().is_some() {
            return Err(commit_failed(
                "a Whiteout setup commits hiding only with jolt-akita's `zk-prover`",
            ));
        }
        #[cfg(feature = "zk-prover")]
        if setup.verifier.whiteout().is_some() {
            return Self::commit_whiteout_precommitted(
                setup,
                PrecommittedShape {
                    num_vars,
                    source: DenseSource::FullField,
                },
                poly,
                layout_digest,
                None,
            );
        }
        let cached = setup.full_width.get(&num_vars).ok_or_else(|| {
            invalid_setup("full-width object arity was not admitted during preprocessing")
        })?;
        let dense = vec![AkitaBackendDensePoly::from_field_evals(
            num_vars,
            akita_ordered_evaluations(poly)?,
        )
        .map_err(akita_error)?];
        let (backend_commitment, backend_hint) = with_backend_pool(|| {
            let source = cached.backend.import_source(dense)?;
            cached.backend.commit(
                cached.scheme.schedules(),
                &source,
                GroupContext::scheduler_without_precommitted_groups(),
            )
        })
        .map(split_commit_output)
        .map_err(commit_failed)?;
        Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintSource::Dense { poly_count: 1 },
        )
    }

    fn transparent_setup_context(setup: &Self::ProverSetup) -> &Self::SetupContext {
        &setup.schedule_artifacts
    }

    fn retag_transparent_object_setup(
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
    ) -> Result<(AkitaProverSetup, AkitaVerifierSetup), OpeningsError> {
        let _ = setup.dense_backend()?;
        if setup.max_num_polys_per_commitment_group() != 1 || setup.max_total_batch_polys() != 1 {
            return Err(invalid_batch(
                "a transparent object setup must admit one polynomial",
            ));
        }
        let mut retagged = setup.clone();
        retagged.verifier.default_layout_digest = layout_digest;
        let verifier = retagged.verifier.clone();
        Ok((retagged, verifier))
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests unwrap successful PCS operations")]
    #![expect(clippy::expect_used, reason = "tests assert successful proof setup")]
    #![expect(clippy::indexing_slicing, reason = "tests index fixture data")]
    #![expect(clippy::panic, reason = "tests reject impossible fixture variants")]

    use super::*;
    use crate::adapters::{AkitaVerifierScheduleArtifacts, AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256};
    use crate::configs::JoltDenseBounded;
    use crate::one_hot_family::OneHotFamily;
    use crate::AkitaOneHotScheduleArtifact;
    use akita_config::CommitmentConfig;
    use akita_params::PolynomialGroupLayout;
    use akita_schedules::ValidatedScheduleCatalog;
    use jolt_algebra::Ring;
    use jolt_transcript::{prover_transcript, Blake2b512};

    #[test]
    fn full_width_objects_do_not_widen_bounded_dense_commitments() {
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        use crate::schedule_registry::{DenseGroupLayout, GroupedScheduleParams};
        let mut params = AkitaSetupParams::dense_only(14, 1, [7; 32], artifacts);
        params.grouped_schedule = Some(GroupedScheduleParams::new(
            None,
            None,
            vec![DenseGroupLayout::FullWidth { num_vars: 14 }],
            PolynomialGroupLayout::new(14, 1),
        ));
        let (bounded, _) = AkitaScheme::setup(params).unwrap();
        let polynomial = Polynomial::new(vec![AkitaField::pow2(80); 1 << 14]);
        assert!(AkitaScheme::commit(&polynomial, &bounded).is_err());
        let (commitment, hint) =
            AkitaScheme::commit_full_width_object(&bounded, &polynomial, [7; 32]).unwrap();
        assert_eq!(commitment, hint.commitment);
        assert_eq!(commitment.num_vars, polynomial.num_vars());
    }

    #[test]
    fn transparent_object_setup_reuses_backend_for_a_new_layout() {
        let context =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let (base, _) = <AkitaScheme as TransparentObjectSetup>::transparent_object_setup(
            &context, 14, [3; 32],
        )
        .unwrap();
        let (retagged, verifier) =
            <AkitaScheme as TransparentObjectSetup>::retag_transparent_object_setup(&base, [4; 32])
                .unwrap();

        assert!(Arc::ptr_eq(
            base.backend_prover_setup.as_ref().unwrap(),
            retagged.backend_prover_setup.as_ref().unwrap()
        ));
        assert!(Arc::ptr_eq(
            base.cpu_backend.as_ref().unwrap(),
            retagged.cpu_backend.as_ref().unwrap()
        ));
        assert_eq!(retagged.default_layout_digest(), [4; 32]);
        assert_eq!(verifier.default_layout_digest(), [4; 32]);
    }

    fn one_hot_roundtrip(
        one_hot_k: usize,
        profile: AkitaChunkProfile,
        num_vars: usize,
        num_polys: usize,
    ) {
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let setup_params =
            AkitaSetupParams::one_hot_only(num_vars, num_polys, [4; 32], one_hot_k, artifacts)
                .with_akita_chunk_profile(profile);
        let (prover_setup, verifier_setup) = AkitaScheme::setup(setup_params).unwrap();
        let row_count = 1usize << (num_vars - one_hot_k.ilog2() as usize);
        let polynomials = (0..num_polys)
            .map(|poly| {
                let indices = (0..row_count)
                    .map(|row| {
                        if row == 2 {
                            None
                        } else {
                            Some(((row + poly) % one_hot_k) as u8)
                        }
                    })
                    .collect::<Vec<_>>();
                OneHotPolynomial::new(one_hot_k, indices)
            })
            .collect::<Vec<_>>();
        let (commitment, hint) =
            AkitaScheme::commit_one_hot_group(&prover_setup, [4; 32], &polynomials).unwrap();
        assert_eq!(commitment.one_hot_k(), one_hot_k);

        let point = vec![AkitaField::from_u64(3); num_vars];
        let statement = polynomials
            .iter()
            .map(|polynomial| VerifierOpeningClaim {
                commitment: commitment.clone(),
                evaluation: EvaluationClaim::new(point.clone(), polynomial.evaluate(&point)),
            })
            .collect::<Vec<_>>();
        let mut prover_state = prover_transcript::<Blake2b512>(
            &jolt_transcript::protocol_tag(b"akita-one-hot-k"),
            b"",
        );
        let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement.clone(),
            polynomials
                .iter()
                .map(|poly| poly as &dyn MultilinearPoly<AkitaField>)
                .collect(),
            hint,
            &mut prover_state,
        )
        .unwrap();
        let mut verifier_transcript = prover_transcript::<Blake2b512>(
            &jolt_transcript::protocol_tag(b"akita-one-hot-k"),
            b"",
        );
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &statement,
            &proof,
            &mut verifier_transcript,
        )
        .unwrap();
        assert_eq!(
            prover_state.verifier_message::<[u8; 32]>(),
            verifier_transcript.verifier_message::<[u8; 32]>()
        );

        if num_vars == 16 && num_polys == 1 {
            let one_hot = match verifier_setup.schedule_artifacts.as_ref() {
                AkitaVerifierScheduleArtifacts::OneHot { one_hot, .. } => one_hot.clone(),
                AkitaVerifierScheduleArtifacts::Dense { .. }
                | AkitaVerifierScheduleArtifacts::Both { .. } => {
                    panic!("one-hot fixture must carry a one-hot-only catalog")
                }
                #[cfg(feature = "zk")]
                AkitaVerifierScheduleArtifacts::OneHotWhiteout { .. } => {
                    panic!("one-hot fixture must carry a one-hot-only catalog")
                }
            };
            let wrong_profile = if profile == AkitaChunkProfile::new(4, 2) {
                AkitaChunkProfile::new(8, 2)
            } else {
                AkitaChunkProfile::new(4, 2)
            };
            for schedule_artifacts in [
                AkitaVerifierScheduleArtifacts::OneHot {
                    profile: wrong_profile,
                    one_hot: one_hot.clone(),
                },
                AkitaVerifierScheduleArtifacts::OneHot {
                    profile: AkitaChunkProfile::default(),
                    one_hot,
                },
            ] {
                if &schedule_artifacts == verifier_setup.schedule_artifacts.as_ref() {
                    continue;
                }
                let mut tampered = verifier_setup.clone();
                tampered.schedule_artifacts = schedule_artifacts.into();
                // Transport drops the honest setup's primed backend caches.
                let bytes = serde_json::to_vec(&tampered).unwrap();
                let transported: AkitaVerifierSetup = serde_json::from_slice(&bytes).unwrap();
                let mut transcript = prover_transcript::<Blake2b512>(
                    &jolt_transcript::protocol_tag(b"akita-one-hot-k"),
                    b"",
                );
                let error = <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
                    &transported,
                    &statement,
                    &proof,
                    &mut transcript,
                )
                .expect_err("a verifier profile must match its embedded catalog");
                assert!(matches!(error, OpeningsError::InvalidSetup(_)));
            }
        }

        let mut wrong_k_statement = statement;
        wrong_k_statement[0].commitment.one_hot_k = if one_hot_k == AKITA_ONE_HOT_K16 {
            AKITA_ONE_HOT_K256
        } else {
            AKITA_ONE_HOT_K16
        };
        let mut verifier_transcript = prover_transcript::<Blake2b512>(
            &jolt_transcript::protocol_tag(b"akita-one-hot-k"),
            b"",
        );
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
        for profile in [AkitaChunkProfile::new(1, 0), AkitaChunkProfile::new(8, 2)] {
            let family = OneHotFamily::from_parts(AKITA_ONE_HOT_K16, profile).unwrap();
            for num_polys in [1, 2] {
                for num_vars in family.num_vars_range(num_polys).0..=16 {
                    one_hot_roundtrip(AKITA_ONE_HOT_K16, profile, num_vars, num_polys);
                }
            }
        }
    }

    #[test]
    fn one_hot_k256_roundtrip() {
        for profile in [AkitaChunkProfile::new(1, 0), AkitaChunkProfile::new(8, 2)] {
            let family = OneHotFamily::from_parts(AKITA_ONE_HOT_K256, profile).unwrap();
            for num_polys in [1, 2] {
                for num_vars in family.num_vars_range(num_polys).0..=16 {
                    one_hot_roundtrip(AKITA_ONE_HOT_K256, profile, num_vars, num_polys);
                }
            }
        }
    }

    /// A serde roundtrip drops the primed key cache; the transported setup
    /// re-derives a verifier-sized backend key from its shape. Proofs verified
    /// through such keys are covered in `verifier_resources::tests`.
    #[test]
    fn serde_transported_setup_rederives_the_backend_key() {
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let (_, verifier_setup) =
            AkitaScheme::setup(AkitaSetupParams::new(14, 1, [3; 32], artifacts)).unwrap();
        let json = serde_json::to_string(&verifier_setup).unwrap();
        let transported: AkitaVerifierSetup = serde_json::from_str(&json).unwrap();
        assert_eq!(transported, verifier_setup);
        let binary = bincode::serde::encode_to_vec(&verifier_setup, bincode::config::standard())
            .expect("verifier setup must encode with Jolt's binary transport");
        let (binary_transport, consumed): (AkitaVerifierSetup, usize) =
            bincode::serde::decode_from_slice(&binary, bincode::config::standard())
                .expect("verifier setup must decode with Jolt's binary transport");
        assert_eq!(consumed, binary.len());
        assert_eq!(binary_transport, verifier_setup);
        assert!(
            transported.dense_verifier().is_ok(),
            "shape-only setup re-derives its backend key"
        );
    }

    #[test]
    fn serde_transported_recursive_grouped_setup_restores_its_schedule_rows() {
        use crate::schedule_registry::{GroupedScheduleParams, FIXTURE_TRUSTED_ADVICE_GROUP};
        use crate::schedules::emit::{K16_COLUMN_VARIABLES, RECURSIVE_TRACE_LOG_T_CUTOVER};

        // This admitted width exercises setup offloading; narrower groups at
        // the same arity may remain direct under the planner's cost objective.
        let final_group =
            PolynomialGroupLayout::new(RECURSIVE_TRACE_LOG_T_CUTOVER + K16_COLUMN_VARIABLES, 51);
        let grouped_schedule = GroupedScheduleParams::new(
            None,
            Some(FIXTURE_TRUSTED_ADVICE_GROUP.num_vars()),
            Vec::new(),
            final_group,
        );
        let (_, verifier_setup) = AkitaScheme::setup(AkitaSetupParams::one_hot_only_grouped(
            final_group.num_vars(),
            final_group.num_polynomials(),
            final_group.num_polynomials() + 1,
            [3; 32],
            AKITA_ONE_HOT_K16,
            Some(grouped_schedule),
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2)),
        ))
        .unwrap();
        let selection = with_one_hot_family!(scheme verifier_setup.one_hot_scheme().unwrap(), |scheme| {
            scheme
                .schedules()
                .rows()
                .find(|row| !row.profiles().precommitteds.is_empty())
                .unwrap()
                .selection()
        });
        let json = serde_json::to_string(&verifier_setup).unwrap();

        let transported: AkitaVerifierSetup = serde_json::from_str(&json).unwrap();
        let resolved = with_one_hot_family!(scheme transported.one_hot_scheme().unwrap(), |scheme| {
            scheme
                .schedules()
                .resolve_selection(selection)
                .expect("transported grouped setup must carry its schedule row")
        });
        assert!(!resolved.profiles().precommitteds.is_empty());
        assert!(resolved
            .schedule()
            .recursive_folds
            .iter()
            .any(|fold| fold.params.setup_prefix().is_some()));
    }

    #[test]
    fn grouped_setup_rejects_a_final_arity_different_from_the_main_setup() {
        use crate::schedule_registry::GroupedScheduleParams;

        let request = GroupedScheduleParams::new(
            None,
            Some(14),
            Vec::new(),
            PolynomialGroupLayout::new(15, 1),
        );
        let error = AkitaScheme::setup(AkitaSetupParams::one_hot_only_grouped(
            14,
            1,
            2,
            [3; 32],
            AKITA_ONE_HOT_K16,
            Some(request),
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2)),
        ))
        .expect_err("a grouped request for another final arity must fail during setup");
        assert!(error
            .to_string()
            .contains("final shape must equal setup shape"));
    }

    #[test]
    fn catalog_digest_prevents_cross_verification_with_the_same_selected_row() {
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let (prover_setup, verifier_setup) = AkitaScheme::setup(AkitaSetupParams::dense_only(
            14,
            1,
            [7; 32],
            Arc::clone(&artifacts),
        ))
        .unwrap();
        let polynomial = Polynomial::new(
            (0..(1u64 << 14))
                .map(|i| AkitaField::from_u64(2 + 5 * i))
                .collect(),
        );
        let (commitment, hint) = AkitaScheme::commit(&polynomial, &prover_setup).unwrap();
        let point = (3..17).map(AkitaField::from_u64).collect::<Vec<_>>();
        let value = polynomial.evaluate(&point);
        let statement = vec![VerifierOpeningClaim {
            commitment,
            evaluation: EvaluationClaim::new(point, value),
        }];
        let mut prover_state =
            prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"catalog-replay"), b"");
        let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement.clone(),
            vec![&polynomial],
            hint,
            &mut prover_state,
        )
        .unwrap();

        let selected = proof.selection();
        let full_catalog = artifacts.dense_catalog().unwrap();
        let omitted = full_catalog
            .rows()
            .find(|row| row.selection() != selected)
            .expect("the base catalog must contain an unused row")
            .selection();
        let reduced_catalog = ValidatedScheduleCatalog::try_new(
            JoltDenseBounded::schedule_family_name(),
            full_catalog
                .rows()
                .filter(|row| row.selection() != omitted)
                .map(|row| (row.profiles().clone(), row.schedule().clone())),
            full_catalog.policy(),
            JoltDenseBounded::ring_challenge_config,
        )
        .unwrap();
        assert!(reduced_catalog.resolve_selection(selected).is_ok());
        assert_ne!(
            full_catalog.catalog_digest(),
            reduced_catalog.catalog_digest()
        );
        let alternate_artifacts = Arc::new(
            AkitaScheduleArtifacts::new(
                artifacts.dense_profile(),
                reduced_catalog.to_artifact_bytes().unwrap(),
                artifacts
                    .full_dense_catalog()
                    .unwrap()
                    .to_artifact_bytes()
                    .unwrap(),
                [AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256]
                    .into_iter()
                    .map(|k| AkitaOneHotScheduleArtifact {
                        k,
                        profile: AkitaChunkProfile::default(),
                        bytes: artifacts
                            .one_hot_catalog(k)
                            .unwrap()
                            .to_artifact_bytes()
                            .unwrap(),
                    })
                    .collect(),
            )
            .unwrap(),
        );
        let (_, alternate_verifier_setup) = AkitaScheme::setup(AkitaSetupParams::dense_only(
            14,
            1,
            [7; 32],
            alternate_artifacts,
        ))
        .unwrap();
        let mut verifier_transcript =
            prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"catalog-replay"), b"");
        let _ = <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &alternate_verifier_setup,
            &statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect_err("a proof must not replay across catalogs with different digests");

        let mut original_transcript =
            prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"catalog-replay"), b"");
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &statement,
            &proof,
            &mut original_transcript,
        )
        .unwrap();
    }

    #[test]
    fn prepared_verifier_verifies_the_selected_row_after_transport() {
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let (prover_setup, verifier_setup) = AkitaScheme::setup(AkitaSetupParams::dense_only(
            14,
            1,
            [7; 32],
            Arc::clone(&artifacts),
        ))
        .unwrap();
        let polynomial = Polynomial::new(
            (0..(1u64 << 14))
                .map(|i| AkitaField::from_u64(3 + 7 * i))
                .collect(),
        );
        let (commitment, hint) = AkitaScheme::commit(&polynomial, &prover_setup).unwrap();
        let point = (5..19).map(AkitaField::from_u64).collect::<Vec<_>>();
        let value = polynomial.evaluate(&point);
        let statement = vec![VerifierOpeningClaim {
            commitment,
            evaluation: EvaluationClaim::new(point, value),
        }];
        let mut prover_transcript = jolt_transcript::prover_transcript::<Blake2b512>(
            &jolt_transcript::protocol_tag(b"prepared"),
            b"",
        );
        let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement.clone(),
            vec![&polynomial],
            hint,
            &mut prover_transcript,
        )
        .unwrap();

        let mut prepared = verifier_setup.clone();
        assert!(prepared.prepare_verifier([0; 32]).is_err());
        prepared
            .prepare_verifier(proof.schedule_row_digest())
            .unwrap();
        let full_digest = verifier_setup
            .dense_scheme()
            .unwrap()
            .schedules()
            .catalog_digest();
        let encoded =
            bincode::serde::encode_to_vec(&prepared, bincode::config::standard()).unwrap();
        let (mut transported, _): (AkitaVerifierSetup, _) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
        let verify = |setup: &AkitaVerifierSetup| {
            let mut transcript = jolt_transcript::prover_transcript::<Blake2b512>(
                &jolt_transcript::protocol_tag(b"prepared"),
                b"",
            );
            <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
                setup,
                &statement,
                &proof,
                &mut transcript,
            )
        };
        // The view keeps the complete catalog's transcript identity.
        for setup in [&prepared, &transported] {
            assert_eq!(
                setup.dense_scheme().unwrap().schedules().catalog_digest(),
                full_digest
            );
        }
        verify(&prepared).unwrap();
        // A decoded record's inline key is never used unchecked.
        assert!(verify(&transported).is_err());

        // Detached bodies serve again only once attached in place
        // (`tests/prepared_transport.rs`).
        let _ = transported.detach_prepared_payloads().unwrap();
        assert!(verify(&transported).is_err());
    }

    #[test]
    fn direct_opening_requires_statement_commitment_layout_digest() {
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let setup_params = AkitaSetupParams::new(14, 1, [7; 32], artifacts);
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

        let mut prover_state = prover_transcript::<Blake2b512>(
            &jolt_transcript::protocol_tag(b"akita-direct-layout"),
            b"",
        );
        let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement.clone(),
            vec![&polynomial],
            hint,
            &mut prover_state,
        )
        .expect("direct proof should prove");

        let mut verifier_transcript = prover_transcript::<Blake2b512>(
            &jolt_transcript::protocol_tag(b"akita-direct-layout"),
            b"",
        );
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect("direct proof should verify");
        assert_eq!(
            prover_state.verifier_message::<[u8; 32]>(),
            verifier_transcript.verifier_message::<[u8; 32]>()
        );

        let mut changed_commitment_statement = statement.clone();
        changed_commitment_statement[0].commitment.layout_digest = [15; 32];
        let mut verifier_transcript = prover_transcript::<Blake2b512>(
            &jolt_transcript::protocol_tag(b"akita-direct-layout"),
            b"",
        );
        let _error = <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &changed_commitment_statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect_err("changed direct commitment digest should reject");

        let mut changed_setup = verifier_setup;
        changed_setup.default_layout_digest = commitment_digest;
        let mut verifier_transcript = prover_transcript::<Blake2b512>(
            &jolt_transcript::protocol_tag(b"akita-direct-layout"),
            b"",
        );
        let _error = <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &changed_setup,
            &statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect_err("direct commitment layout must not be accepted through setup default");
    }
}
