use super::*;

/// Finite row state prepared at construction or trusted-payload attachment.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(into = "WhiteoutSetupWire", try_from = "WhiteoutSetupWire")]
pub struct WhiteoutSetup {
    pub(super) grouped: Option<GroupedScheduleParams>,
    /// The persisted plans the setup was constructed with.
    pub(super) persisted: Option<WhiteoutPlans>,
    /// The packaged response catalog ([`ResponsePlans`]). `None` when the
    /// persisted plans carry a response set, which then fixes every row
    /// ([`Self::from_parts`]).
    pub(super) response_catalog: Option<Arc<Vec<u8>>>,
    pub(crate) prepared: Option<Arc<PreparedWhiteout>>,
    pub(super) capacity: WhiteoutCapacity,
    pub(super) payloads: Option<Box<WhiteoutPayloads>>,
}

/// [`WhiteoutSetup`]'s wire form.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WhiteoutSetupWire {
    pub(super) version: u32,
    pub(super) capacity: WhiteoutCapacity,
    pub(super) grouped: Option<GroupedScheduleParams>,
    pub(super) plans: Option<WhiteoutPlans>,
    pub(super) response_catalog: Option<Vec<u8>>,
    pub(super) payloads: Option<Box<WhiteoutPayloads>>,
}

impl From<WhiteoutSetup> for WhiteoutSetupWire {
    fn from(setup: WhiteoutSetup) -> Self {
        let plans = if setup.payloads.is_some() {
            None
        } else {
            setup.plans()
        };
        Self {
            version: 1,
            capacity: setup.capacity,
            grouped: setup.grouped,
            plans,
            response_catalog: if setup.payloads.is_some() {
                None
            } else {
                setup.response_catalog.map(Arc::unwrap_or_clone)
            },
            payloads: setup.payloads,
        }
    }
}

impl TryFrom<WhiteoutSetupWire> for WhiteoutSetup {
    type Error = ZkError;
    fn try_from(wire: WhiteoutSetupWire) -> Result<Self, ZkError> {
        if wire.version != 1 {
            return Err(
                AkitaError::InvalidSetup("unsupported Whiteout setup version".into()).into(),
            );
        }
        if let Some(plans) = &wire.plans {
            plans.validate_header()?;
        }
        wire.capacity.validate()?;
        if wire.payloads.is_some() {
            return Ok(Self {
                grouped: wire.grouped,
                persisted: wire.plans,
                response_catalog: wire.response_catalog.map(Arc::new),
                prepared: None,
                capacity: wire.capacity,
                payloads: wire.payloads,
            });
        }
        let trace = wire
            .plans
            .as_ref()
            .map(|plans| (plans.catalog.num_vars, plans.catalog.num_polys));
        Self::from_parts(
            wire.grouped,
            wire.plans,
            wire.response_catalog,
            trace,
            wire.capacity,
        )
    }
}

/// [`WhiteoutSetup`]'s derived state.
#[derive(Clone)]
pub(crate) struct PreparedWhiteout {
    pub(super) catalog: Arc<WhiteoutCatalog>,
    pub(super) shapes: BTreeMap<Vec<PrecommittedShape>, Arc<WhiteoutShape>>,
    /// The catalog as persisted.
    pub(super) plan: CatalogPlan,
    pub(super) responses: Arc<ResponsePlans>,
}

impl PreparedWhiteout {
    pub(crate) fn shape(&self, groups: &MaskedOpeningShape) -> Result<Arc<WhiteoutShape>, ZkError> {
        if (groups.num_vars, groups.num_polys) != (self.catalog.num_vars, self.catalog.num_polys) {
            return Err(
                AkitaError::InvalidSetup("Whiteout prepared trace shape mismatch".into()).into(),
            );
        }
        self.shapes
            .get(&groups.precommitted)
            .cloned()
            .ok_or_else(|| {
                AkitaError::UnsupportedSchedule(
                    "Whiteout precommitted shape is not admitted".into(),
                )
                .into()
            })
    }

    pub(super) fn new(
        catalog: WhiteoutCatalog,
        residuals: PlannedRows,
        plan: CatalogPlan,
        responses: ResponseRows,
        streams: Arc<CapacityStreams>,
        payloads: Option<&WhiteoutPayloads>,
    ) -> Result<Self, ZkError> {
        let responses = Arc::new(ResponsePlans::new(
            responses,
            catalog.material.clone(),
            payloads,
        )?);
        let catalog = Arc::new(catalog);
        let shapes = residuals
            .into_iter()
            .map(|(precommitted, (key, shape))| {
                let selection = catalog.scheme.schedules().resolve_key(&key)?.selection();
                let cache = payloads
                    .map(|payloads| payloads.masked_cache(selection.row_digest.as_bytes()))
                    .transpose()?;
                let verifier = Arc::new(AkitaVerifier::for_selection(
                    catalog.material.clone(),
                    catalog.scheme.schedules().clone(),
                    selection,
                    cache,
                )?);
                let row = WhiteoutRow {
                    verifier,
                    catalog: Arc::clone(&catalog),
                    key,
                    matrices: RowMatrices::new(&shape, catalog.material.expanded())?,
                    shape,
                };
                Ok((
                    precommitted,
                    Arc::new(WhiteoutShape {
                        tail: WhiteoutTail::new(row, Arc::clone(&responses), Arc::clone(&streams)),
                    }),
                ))
            })
            .collect::<Result<_, ZkError>>()?;
        Ok(Self {
            catalog,
            shapes,
            plan,
            responses,
        })
    }
}

impl Debug for WhiteoutSetup {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("WhiteoutSetup")
            .field("grouped", &self.grouped)
            .finish_non_exhaustive()
    }
}

/// Setup identity includes plans, coefficient capacity and payload bytes.
impl PartialEq for WhiteoutSetup {
    fn eq(&self, other: &Self) -> bool {
        self.grouped == other.grouped
            && self.capacity == other.capacity
            && self.payloads == other.payloads
            && (self.payloads.is_some()
                || (self.plans() == other.plans()
                    && self.response_catalog == other.response_catalog))
    }
}

impl Eq for WhiteoutSetup {}

impl WhiteoutSetup {
    pub(crate) fn new(
        grouped: Option<GroupedScheduleParams>,
        source: WhiteoutSource,
        trace: Option<(usize, usize)>,
        response_catalog: Option<Vec<u8>>,
    ) -> Result<Self, ZkError> {
        let persisted = match source {
            WhiteoutSource::Plan => None,
            WhiteoutSource::Load(plans) => Some(plans),
        };
        Self::from_parts(
            grouped,
            persisted,
            response_catalog,
            trace,
            WhiteoutCapacity::default(),
        )
    }

    pub(super) fn from_parts(
        grouped: Option<GroupedScheduleParams>,
        persisted: Option<WhiteoutPlans>,
        response_catalog: Option<Vec<u8>>,
        trace: Option<(usize, usize)>,
        capacity: WhiteoutCapacity,
    ) -> Result<Self, ZkError> {
        let carries_responses = persisted
            .as_ref()
            .is_some_and(|plans| plans.responses.is_some());
        let mut setup = Self {
            grouped,
            persisted,
            response_catalog: response_catalog
                .filter(|_| !carries_responses)
                .map(Arc::new),
            prepared: None,
            capacity,
            payloads: None,
        };
        if let Some((num_vars, num_polys)) = trace {
            setup.prepared = Some(Arc::new(setup.prepare(num_vars, num_polys)?));
        }
        Ok(setup)
    }

    pub(super) fn set_capacity(&mut self, capacity: WhiteoutCapacity) -> Result<(), ZkError> {
        if self.prepared.is_none() {
            return Err(AkitaError::InvalidSetup(
                "Whiteout setup must be prepared before resizing".into(),
            )
            .into());
        }
        let streams = Arc::new(CapacityStreams::new(capacity)?);
        if let Some(prepared) = &mut self.prepared {
            let prepared = Arc::make_mut(prepared);
            prepared.shapes = prepared
                .shapes
                .iter()
                .map(|(key, shape)| {
                    (
                        key.clone(),
                        Arc::new(WhiteoutShape {
                            tail: WhiteoutTail::new(
                                shape.row().clone(),
                                Arc::clone(&prepared.responses),
                                Arc::clone(&streams),
                            ),
                        }),
                    )
                })
                .collect();
        }
        self.capacity = capacity;
        self.payloads = None;
        Ok(())
    }

    pub(super) fn admitted(&self) -> BTreeSet<Vec<PrecommittedShape>> {
        self.grouped.as_ref().map_or_else(
            || std::iter::once(Vec::new()).collect(),
            GroupedScheduleParams::whiteout_admitted,
        )
    }

    pub(super) fn plans(&self) -> Option<WhiteoutPlans> {
        if let Some(plans) = &self.persisted {
            return Some(plans.clone());
        }
        self.prepared.as_deref().map(|planned| WhiteoutPlans {
            magic: WHITEOUT_PLANS_MAGIC,
            version: WHITEOUT_PLANS_VERSION,
            catalog: planned.plan.clone(),
            responses: None,
        })
    }

    /// Every plan the setup holds or has derived, the response schedules
    /// included, as a setup constructed from them would persist them: its
    /// response set is then exhaustive.
    pub(crate) fn export(&self) -> Option<WhiteoutPlans> {
        let mut plans = self.plans()?;
        if let Some(planned) = self.prepared.as_deref() {
            plans.responses = Some(planned.responses.plans());
        }
        Some(plans)
    }

    pub(super) fn prepared(
        &self,
        num_vars: usize,
        num_polys: usize,
    ) -> Result<&PreparedWhiteout, ZkError> {
        let prepared = self
            .prepared
            .as_deref()
            .ok_or_else(|| AkitaError::InvalidSetup("Whiteout setup is not prepared".into()))?;
        if (prepared.catalog.num_vars, prepared.catalog.num_polys) != (num_vars, num_polys) {
            return Err(WhiteoutPlanError::Statement(AkitaError::InvalidSetup(
                "Whiteout setup has another trace shape".into(),
            ))
            .into());
        }
        Ok(prepared)
    }

    pub(super) fn prepare(
        &self,
        num_vars: usize,
        num_polys: usize,
    ) -> Result<PreparedWhiteout, ZkError> {
        if let Some(plans) = &self.persisted {
            plans.validate_header()?;
        }
        self.capacity.validate()?;
        let streams = Arc::new(match &self.payloads {
            Some(payloads) => payloads.streams(self.capacity)?,
            None => CapacityStreams::new(self.capacity)?,
        });
        let admitted = self.admitted();
        let planned = if let Some(plans) = &self.persisted {
            let _span = tracing::info_span!("whiteout_load_catalog").entered();
            let responses = ResponseRows::new(
                plans.responses.as_ref(),
                self.response_catalog.as_deref().map(Vec::as_slice),
            )?;
            let (catalog, residuals) = WhiteoutCatalog::load(
                &plans.catalog,
                num_vars,
                num_polys,
                &admitted,
                &responses,
                self.payloads
                    .as_deref()
                    .map(WhiteoutPayloads::material)
                    .transpose()?,
            )?;
            PreparedWhiteout::new(
                catalog,
                residuals,
                plans.catalog.clone(),
                responses,
                streams,
                self.payloads.as_deref(),
            )
        } else {
            let _span = tracing::info_span!("whiteout_plan_catalog").entered();
            let responses =
                ResponseRows::new(None, self.response_catalog.as_deref().map(Vec::as_slice))?;
            let (catalog, residuals) =
                WhiteoutCatalog::plan(num_vars, num_polys, &admitted, &responses)?;
            let plan = catalog.plan_of()?;
            PreparedWhiteout::new(
                catalog,
                residuals,
                plan,
                responses,
                streams,
                self.payloads.as_deref(),
            )
        };
        planned
    }

    /// The setup's catalog, for a trace group of `num_polys` polynomials in
    /// `num_vars` variables.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if an admitted shape has no zero-knowledge row, or a
    /// persisted plan is refused ([`WhiteoutPlanError`]).
    #[cfg(feature = "zk-prover")]
    pub(crate) fn catalog(
        &self,
        num_vars: usize,
        num_polys: usize,
    ) -> Result<&WhiteoutCatalog, ZkError> {
        Ok(&self.prepared(num_vars, num_polys)?.catalog)
    }

    /// The masked opening of `groups`.
    ///
    /// # Errors
    ///
    /// [`AkitaError::UnsupportedSchedule`] if the setup does not admit the
    /// shape, and [`ZkError`] if an admitted shape has no zero-knowledge row
    /// or a persisted plan is refused ([`WhiteoutPlanError`]).
    pub(crate) fn shape(&self, groups: &MaskedOpeningShape) -> Result<Arc<WhiteoutShape>, ZkError> {
        self.prepared(groups.num_vars, groups.num_polys)?
            .shape(groups)
    }
}
