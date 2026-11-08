use super::*;
use crate::schedules::emit::RESPONSE_NUM_VARS;

/// The response rows a setup holds before any proof, audited when its
/// catalog is: its persisted response set, which fixes every row, or else
/// the packaged response catalog, if the setup carries one.
pub(super) enum ResponseRows {
    Persisted(BTreeMap<usize, ResponseRow>),
    Packaged(Option<TrustedScheduleCatalog<WhiteoutResponse>>),
}

impl ResponseRows {
    /// Audits `persisted`, else `packaged`. A row or catalog that fails
    /// admission is refused, never replaced by planning.
    pub(super) fn new(
        persisted: Option<&BTreeMap<usize, Vec<u8>>>,
        packaged: Option<&[u8]>,
    ) -> Result<Self, ZkError> {
        if let Some(persisted) = persisted {
            if let Some(num_vars) = persisted
                .keys()
                .find(|num_vars| !(RESPONSE_NUM_VARS.0..=RESPONSE_NUM_VARS.1).contains(*num_vars))
            {
                return Err(
                    WhiteoutPlanError::Statement(AkitaError::UnsupportedSchedule(format!(
                        "no response proof has {num_vars} variables"
                    )))
                    .into(),
                );
            }
            return Ok(Self::Persisted(
                persisted
                    .iter()
                    .map(|(num_vars, artifact)| {
                        Ok((*num_vars, ResponseRow::load(*num_vars, artifact)?))
                    })
                    .collect::<Result<_, ZkError>>()?,
            ));
        }
        let Some(bytes) = packaged else {
            return Ok(Self::Packaged(None));
        };
        let _span = tracing::info_span!("whiteout_load_response_catalog").entered();
        let rows = Self::Packaged(Some(
            TrustedScheduleCatalog::from_artifact_bytes(
                bytes,
                ChunkedWitnessCfg::default_non_chunked(),
            )
            .map_err(WhiteoutPlanError::Inadmissible)?,
        ));
        for num_vars in RESPONSE_NUM_VARS.0..=RESPONSE_NUM_VARS.1 {
            if rows.row(num_vars)?.is_none() {
                return Err(WhiteoutPlanError::MissingResponse { num_vars }.into());
            }
        }
        Ok(rows)
    }

    /// Covers every row's response verifier in `requirements`.
    pub(super) fn include(
        &self,
        requirements: &mut AkitaVerifierRequirements,
    ) -> Result<(), ZkError> {
        match self {
            Self::Persisted(rows) => {
                for (num_vars, row) in rows {
                    include_response(requirements, row.scheme.schedules(), *num_vars)?;
                }
            }
            Self::Packaged(Some(catalog)) => {
                let longest = catalog
                    .rows()
                    .map(|row| row.profiles().final_group.group.num_vars())
                    .max();
                if let Some(num_vars) = longest {
                    include_response(requirements, catalog, num_vars)?;
                }
            }
            Self::Packaged(None) => {}
        }
        Ok(())
    }

    /// The row of `e` with `num_vars` variables.
    pub(super) fn row(&self, num_vars: usize) -> Result<Option<ResponseRow>, ZkError> {
        match self {
            Self::Persisted(rows) => rows.get(&num_vars).cloned().map(Some).ok_or_else(|| {
                WhiteoutPlanError::Statement(AkitaError::UnsupportedSchedule(format!(
                    "no persisted response plan for e of {num_vars} variables"
                )))
                .into()
            }),
            Self::Packaged(Some(catalog)) => {
                let group = PolynomialGroupLayout::singleton(num_vars);
                catalog
                    .rows()
                    .find(|row| {
                        row.profiles().final_group.group == group
                            && row.profiles().precommitteds.is_empty()
                    })
                    .map(|row| {
                        ResponseRow::from_row(
                            num_vars,
                            row.profiles().clone(),
                            row.schedule().clone(),
                        )
                    })
                    .transpose()
            }
            Self::Packaged(None) => Ok(None),
        }
    }
}

/// Covers in `requirements` what a response verifier over `catalog` of
/// every `e` of up to `num_vars` variables reads.
fn include_response(
    requirements: &mut AkitaVerifierRequirements,
    catalog: &TrustedScheduleCatalog<WhiteoutResponse>,
    num_vars: usize,
) -> Result<(), ZkError> {
    requirements
        .include(catalog, num_vars, 1)
        .map_err(material_error)?;
    requirements.retain(AkitaResponseVerifier::setup_field_elements(
        catalog, num_vars,
    )?);
    Ok(())
}

/// Eager response verifiers indexed by the response variable count.
pub(super) struct ResponsePlans {
    pub(super) prepared: BTreeMap<usize, Arc<PreparedResponse>>,
}

impl ResponsePlans {
    pub(super) fn new(
        rows: ResponseRows,
        material: AkitaBackendVerifierSetup,
        payloads: Option<&WhiteoutPayloads>,
    ) -> Result<Self, ZkError> {
        let lengths: Vec<_> = match &rows {
            ResponseRows::Persisted(rows) => rows.keys().copied().collect(),
            ResponseRows::Packaged(_) => (RESPONSE_NUM_VARS.0..=RESPONSE_NUM_VARS.1).collect(),
        };
        let mut prepared = BTreeMap::new();
        for num_vars in lengths {
            let row = rows.row(num_vars)?.ok_or_else(|| {
                WhiteoutPlanError::Statement(AkitaError::UnsupportedSchedule(format!(
                    "no response row for {num_vars} variables"
                )))
            })?;
            let _ = prepared.insert(
                num_vars,
                Arc::new(PreparedResponse::new(
                    row,
                    &material,
                    payloads
                        .map(|payloads| payloads.response_cache(num_vars))
                        .transpose()?,
                )?),
            );
        }
        Ok(Self { prepared })
    }

    pub(super) fn get(&self, num_vars: usize) -> Result<Arc<PreparedResponse>, ZkError> {
        self.prepared
            .get(&num_vars)
            .cloned()
            .ok_or_else(|| WhiteoutPlanError::MissingResponse { num_vars }.into())
    }

    pub(super) fn plans(&self) -> BTreeMap<usize, Vec<u8>> {
        self.prepared
            .iter()
            .map(|(num_vars, response)| (*num_vars, response.row.plan.clone()))
            .collect()
    }
}

/// One response row: its one-row scheme, its plan as persisted and the
/// plan's catalog digest, which the pads absorb.
#[derive(Clone)]
pub(super) struct ResponseRow {
    pub(super) num_vars: usize,
    pub(super) scheme: AkitaCommitmentScheme<WhiteoutResponse>,
    pub(super) plan: Vec<u8>,
    pub(super) digest: [u8; 32],
}

impl ResponseRow {
    /// Plans the response row of `e` with `num_vars` variables.
    #[cfg(all(test, feature = "zk-prover"))]
    pub(super) fn plan(num_vars: usize) -> Result<Self, ZkError> {
        let _span = tracing::info_span!("whiteout_plan_response", num_vars).entered();
        let chunking = ChunkedWitnessCfg::default_non_chunked();
        let group = PolynomialGroupLayout::singleton(num_vars);
        let key = ScheduleLookupKey::single(group);
        let planned = tracing::info_span!("find_schedule").in_scope(|| {
            akita_planner::find_schedule(
                &key,
                WhiteoutResponse::committed_source_contract()?,
                &[],
                &policy_of::<WhiteoutResponse>(chunking),
                WhiteoutResponse::ring_challenge_config,
            )
        })?;
        let profile = CommittedGroupBatchProfile {
            final_group: GroupCommitPhaseParams::try_from_params(
                group,
                &planned.schedule.root.params,
            )?,
            precommitteds: Vec::new(),
        };
        Self::from_row(num_vars, profile, planned.schedule)
    }

    /// The response row of `e` with `num_vars` variables as its one-row
    /// catalog, audited by Akita.
    pub(super) fn from_row(
        num_vars: usize,
        profile: CommittedGroupBatchProfile,
        schedule: FoldSchedule,
    ) -> Result<Self, ZkError> {
        let catalog = TrustedScheduleCatalog::try_from_rows(
            [(profile, schedule)],
            ChunkedWitnessCfg::default_non_chunked(),
        )?;
        let plan = catalog.to_artifact_bytes()?;
        Ok(Self::new(num_vars, catalog, plan))
    }

    /// The response row of `e` with `num_vars` variables from its persisted
    /// one-row catalog, audited by Akita and required to be that length's
    /// row ([`load_single_row`]).
    pub(super) fn load(num_vars: usize, artifact: &[u8]) -> Result<Self, ZkError> {
        let _span = tracing::info_span!("whiteout_load_response", num_vars).entered();
        let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(num_vars));
        let (_, catalog) = load_single_row::<WhiteoutResponse>(artifact, &key)?;
        Ok(Self::new(num_vars, catalog, artifact.to_vec()))
    }

    pub(super) fn new(
        num_vars: usize,
        catalog: TrustedScheduleCatalog<WhiteoutResponse>,
        plan: Vec<u8>,
    ) -> Self {
        Self {
            num_vars,
            digest: catalog.catalog_digest(),
            scheme: AkitaCommitmentScheme::new(catalog),
            plan,
        }
    }
}

/// An eager response verifier and a lazily built response prover.
pub(super) struct PreparedResponse {
    pub(super) row: ResponseRow,
    pub(super) verifier: AkitaResponseVerifier<WhiteoutResponse>,
    #[cfg(feature = "zk-prover")]
    pub(super) prover: OnceLock<Arc<AkitaResponseProver<WhiteoutResponse>>>,
}

impl PreparedResponse {
    /// `row`'s verifier over a view of `material` at `row`'s length.
    pub(super) fn new(
        row: ResponseRow,
        material: &AkitaBackendVerifierSetup,
        cache: Option<TrustedTerminalCache<'_>>,
    ) -> Result<Self, AkitaError> {
        let selection = row
            .scheme
            .schedules()
            .resolve_key(&ScheduleLookupKey::single(
                PolynomialGroupLayout::singleton(row.num_vars),
            ))?
            .selection();
        let verifier = AkitaVerifier::for_selection(
            material.with_capacity(row.num_vars, 1),
            row.scheme.schedules().clone(),
            selection,
            cache,
        )?;
        Ok(Self {
            verifier: AkitaResponseVerifier::from_verifier(verifier)?,
            row,
            #[cfg(feature = "zk-prover")]
            prover: OnceLock::new(),
        })
    }

    /// `row`'s response prover, over its own prover setup.
    #[cfg(feature = "zk-prover")]
    pub(super) fn prover(&self) -> Result<Arc<AkitaResponseProver<WhiteoutResponse>>, AkitaError> {
        if let Some(prover) = self.prover.get() {
            return Ok(Arc::clone(prover));
        }
        // Rayon work stays outside the cell to avoid worker reentrancy deadlocks.
        let setup = AkitaResponseProver::setup(&self.row.scheme, self.row.num_vars)?;
        let prover = Arc::new(AkitaResponseProver::new(self.row.scheme.clone(), setup)?);
        Ok(Arc::clone(self.prover.get_or_init(|| prover)))
    }
}
