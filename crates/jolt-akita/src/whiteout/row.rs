use super::*;
use akita_types::AkitaExpandedSetup;

/// One admitted masked opening row and its eagerly prepared residual matrices.
#[derive(Clone)]
pub struct WhiteoutRow {
    pub(super) catalog: Arc<WhiteoutCatalog>,
    pub(super) key: ScheduleLookupKey,
    pub(super) shape: ResidualShape,
    pub(super) matrices: RowMatrices,
    pub(super) verifier: Arc<AkitaVerifier<WhiteoutOneHot>>,
}

impl WhiteoutRow {
    /// The residual commitment's ring dimension, which the pad commitment
    /// and the Σ-protocol share.
    pub fn ring_dim(&self) -> usize {
        self.shape.ring_dim()
    }

    /// The schedule row both roles bind.
    pub fn selection(&self) -> Result<OpeningScheduleSelection, AkitaError> {
        Ok(self
            .catalog
            .scheme
            .schedules()
            .resolve_key(&self.key)?
            .selection())
    }

    /// The setup's catalog.
    pub fn catalog(&self) -> &WhiteoutCatalog {
        &self.catalog
    }

    /// Commits the trace group `source` hiding, under this row's profile,
    /// keyed on its precommitted groups: the backend draws fresh randomness
    /// for every masked-level slice.
    ///
    /// # Errors
    ///
    /// The backend's errors.
    #[cfg(feature = "zk-prover")]
    pub fn commit_hiding(
        &self,
        source: &SourceHandle<F, F>,
    ) -> Result<(CommittedGroup<F>, AkitaBackendHint), AkitaError> {
        let backend = self.catalog.backend()?;
        let schedules = self.catalog.scheme.schedules();
        let output = if self.key.precommitteds.is_empty() {
            backend.commit(
                schedules,
                source,
                GroupContext::scheduler_without_precommitted_groups(),
            )?
        } else {
            let profiles =
                PrecommittedGroupProfiles::from_profiles(self.key.precommitteds.clone())?;
            backend.commit(
                schedules,
                source,
                GroupContext::scheduler_with_precommitted_groups(&profiles),
            )?
        };
        Ok((output.committed_group, output.private_handle))
    }

    /// Proves the masked levels through the cut for `groups` (precommitted,
    /// then the trace group; each its commitment, handle, point in Akita's
    /// order and true evaluations), whose claims `forms` state in the same
    /// order, sending u_a compressed under `compression`
    /// ([`WhiteoutTail::compression`]). Returns the residual opening and
    /// Akita's auxiliary values.
    ///
    /// # Errors
    ///
    /// [`AkitaError::FoldOverflow`] on a one-shot fold overflow (the caller
    /// reruns), and the masked prover's errors.
    #[cfg(feature = "zk-prover")]
    pub fn prove_masked(
        &self,
        groups: Vec<ProverGroup>,
        forms: &[AffineRow<F>],
        pads: &mut CommittedPads<F, WhiteoutPadOpening<F>>,
        transcript: &mut AkitaProverTranscript,
        compression: &dyn ImageCompression<F>,
    ) -> Result<(ResidualOpening<F, F>, Vec<F>), AkitaError> {
        let CatalogProver { setup, backend } = self.catalog.prover()?;
        let mut claims = Vec::with_capacity(groups.len());
        let mut handles = Vec::with_capacity(groups.len());
        for group in groups {
            claims.push(PolynomialGroupClaims::new(
                group.point,
                group.evaluations,
                group.commitment,
            )?);
            handles.push(group.handle);
        }
        let opening = SelectedProverOpeningData::from_committed_claims::<WhiteoutOneHot>(
            OpeningClaims::from_groups(claims)?,
            handles,
            self.catalog.scheme.schedules(),
        )?;
        if opening.selection() != self.selection()? {
            return Err(AkitaError::InvalidInput(
                "the opening's groups select another Whiteout row".into(),
            ));
        }
        let schedules = self.catalog.scheme.schedules();
        transcript.append(&schedules.catalog_digest());
        transcript.append(opening.selection().row_digest.as_bytes());
        let resolved = schedules.resolve_selection(opening.selection())?;
        let prefix_ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
            resolved.schedule(),
            opening.opening_layout(),
        )?;
        let prefixes = backend.import_setup_prefixes(&setup.prefix_slots, &prefix_ids)?;
        let prefix = prove_masked_prefix::<WhiteoutOneHot, _, _>(
            setup.expanded.descriptor(),
            schedules,
            backend,
            &prefixes,
            opening,
            &forms.to_vec(),
            pads,
            transcript,
            compression,
            BasisMode::Lagrange,
        )?;
        // The record closes a message where the next begins; Jolt writes Σ
        // next, so close the cut's last message here.
        #[cfg(feature = "logging")]
        akita_transcript::finish_proof_ranges(transcript);
        Ok(prefix)
    }

    /// Replays the masked levels through the cut on `pads`; `groups` are the
    /// decoded commitments at their points (Akita's order), precommitted
    /// first.
    ///
    /// # Errors
    ///
    /// The masked verifier's errors, or a row other than this shape's.
    pub fn verify_masked<C>(
        &self,
        groups: &[(CommittedGroup<F>, Vec<F>)],
        forms: Vec<AffineRow<F>>,
        pads: &mut CommittedPadsView<C>,
        transcript: &mut AkitaVerifierTranscript<'_>,
    ) -> Result<MaskedPrefix<F, F>, AkitaError> {
        let selection = self.absorb_selection(transcript)?;
        self.verifier.verify_masked_prefix(
            transcript,
            pads,
            masked_statement(selection, groups)?,
            forms,
            &CompressionShape::<WhiteoutResponse>::default(),
            BasisMode::Lagrange,
        )
    }

    /// The pad-layout replay of [`Self::verify_masked`]: counts the pads of
    /// the masked messages.
    ///
    /// # Errors
    ///
    /// The counting verifier's errors.
    pub fn count_masked(
        &self,
        groups: &[(CommittedGroup<F>, Vec<F>)],
        forms: Vec<AffineRow<F>>,
        view: &mut CommittedPadsView<()>,
        transcript: &mut AkitaVerifierTranscript<'_>,
    ) -> Result<MaskedPrefix<F, F>, AkitaError> {
        let selection = self.absorb_selection(transcript)?;
        self.verifier.count_masked_prefix(
            transcript,
            view,
            masked_statement(selection, groups)?,
            forms,
            &CompressionShape::<WhiteoutResponse>::default(),
            BasisMode::Lagrange,
        )
    }

    /// Absorbs the catalog and the selected row, both derived from the
    /// statement: the prover sends no selection.
    fn absorb_selection(
        &self,
        transcript: &mut AkitaVerifierTranscript<'_>,
    ) -> Result<OpeningScheduleSelection, AkitaError> {
        let selection = self.selection()?;
        transcript.append(&self.catalog.scheme.schedules().catalog_digest());
        transcript.append(selection.row_digest.as_bytes());
        Ok(selection)
    }

    /// Decodes the precommitted and trace commitments under this row's
    /// profiles, in the row's group order.
    pub(crate) fn backend_commitments(
        &self,
        precommitted: &[&AkitaCommitment],
        trace: &AkitaCommitment,
    ) -> Result<(Vec<CommittedGroup<F>>, CommittedGroup<F>), AkitaError> {
        if precommitted.len() != self.key.precommitteds.len() {
            return Err(AkitaError::InvalidInput(format!(
                "the Whiteout row opens {} precommitted groups, not {}",
                self.key.precommitteds.len(),
                precommitted.len()
            )));
        }
        crate::shape_guard::deserialize_checked_grouped_backend_payload(
            self.catalog.scheme.schedules(),
            precommitted,
            trace,
            self.selection()?,
        )
        .map_err(|error| AkitaError::InvalidSetup(error.to_string()))
    }
}

#[derive(Clone)]
pub(super) enum RowMatrices {
    D64(Arc<ResidualMatrices<64>>),
    D128(Arc<ResidualMatrices<128>>),
    D256(Arc<ResidualMatrices<256>>),
}

struct ResidualCoefficients(Arc<AkitaExpandedSetup<F>>);

impl AsRef<[F]> for ResidualCoefficients {
    fn as_ref(&self) -> &[F] {
        self.0.shared_matrix().as_field_slice()
    }
}

impl RowMatrices {
    pub(super) fn new(
        shape: &ResidualShape,
        stream: &Arc<AkitaExpandedSetup<F>>,
    ) -> Result<Self, ZkError> {
        let _span = tracing::info_span!("whiteout_residual_matrices").entered();
        let coefficients = Arc::new(ResidualCoefficients(Arc::clone(stream)));
        macro_rules! prepare {
            ($dimension:literal, $variant:ident) => {{
                const D: usize = $dimension;
                let (a, b) = shape.matrix_shapes();
                let a = PreparedMatrix::<F, D>::from_shared(a, Arc::clone(&coefficients))
                    .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
                let b = a
                    .with_shared(b, coefficients)
                    .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
                let matrices = Arc::new(ResidualMatrices {
                    a: Arc::new(a),
                    b: Arc::new(b),
                });
                Ok(Self::$variant(matrices))
            }};
        }
        match shape.ring_dim() {
            64 => prepare!(64, D64),
            128 => prepare!(128, D128),
            256 => prepare!(256, D256),
            other => Err(AkitaError::UnsupportedSchedule(format!(
                "residual ring dimension {other}"
            ))
            .into()),
        }
    }
}

/// The verifier's statement for `groups` at their points (Akita's
/// order): its evaluations are never read, the masked forms replace
/// them.
fn masked_statement(
    selection: OpeningScheduleSelection,
    groups: &[(CommittedGroup<F>, Vec<F>)],
) -> Result<GroupBatchStatement<'_, F, F>, AkitaError> {
    let claims = groups
        .iter()
        .map(|(commitment, point)| {
            PolynomialGroupClaims::new(
                point.clone(),
                vec![F::default(); commitment.profile().group.num_polynomials()],
                commitment,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    GroupBatchStatement::new(selection, OpeningClaims::from_groups(claims)?)
}

/// One group the prover opens: its commitment, the handle retaining its
/// source, its point in Akita's order and its true evaluations.
#[cfg(feature = "zk-prover")]
pub struct ProverGroup {
    pub commitment: CommittedGroup<F>,
    pub handle: AkitaBackendHint,
    pub point: Vec<F>,
    pub evaluations: Vec<F>,
}
