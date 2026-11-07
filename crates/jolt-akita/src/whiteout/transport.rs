use super::capacity::StreamCoefficients;
use super::*;
use crate::adapters::{invalid_setup, AkitaVerifierScheduleArtifacts};
use crate::prepared::PreparedBytes;
use crate::AkitaVerifierSetup;
use akita_pcs::{
    build_riscv64_terminal_ntt_cache, AkitaDeserialize, AkitaSerialize, Compress, Validate,
};
use akita_types::AkitaExpandedSetup;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(super) struct WhiteoutPayloads {
    plans: TransportPlans,
    key: PreparedBytes,
    pad: PreparedBytes,
    v: PreparedBytes,
    masked: BTreeMap<[u8; 32], PreparedBytes>,
    responses: BTreeMap<usize, PreparedBytes>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct TransportPlans {
    magic: [u8; 4],
    version: u32,
    num_vars: usize,
    num_polys: usize,
    precommitted: Vec<PreparedBytes>,
    trace: PreparedBytes,
    responses: BTreeMap<usize, PreparedBytes>,
}

impl TransportPlans {
    fn new(plans: WhiteoutPlans) -> Result<Self, OpeningsError> {
        Ok(Self {
            magic: plans.magic,
            version: plans.version,
            num_vars: plans.catalog.num_vars,
            num_polys: plans.catalog.num_polys,
            precommitted: plans
                .catalog
                .precommitted
                .into_iter()
                .map(PreparedBytes::from)
                .collect(),
            trace: plans.catalog.trace.into(),
            responses: plans
                .responses
                .ok_or_else(|| invalid_setup("prepared response plans missing"))?
                .into_iter()
                .map(|(length, bytes)| (length, bytes.into()))
                .collect(),
        })
    }

    fn load(&self) -> Result<WhiteoutPlans, OpeningsError> {
        Ok(WhiteoutPlans {
            magic: self.magic,
            version: self.version,
            catalog: CatalogPlan {
                num_vars: self.num_vars,
                num_polys: self.num_polys,
                precommitted: self
                    .precommitted
                    .iter()
                    .map(|bytes| Ok(bytes.bytes()?.to_vec()))
                    .collect::<Result<_, OpeningsError>>()?,
                trace: self.trace.bytes()?.to_vec(),
            },
            responses: Some(
                self.responses
                    .iter()
                    .map(|(length, bytes)| Ok((*length, bytes.bytes()?.to_vec())))
                    .collect::<Result<_, OpeningsError>>()?,
            ),
        })
    }

    fn slots(&mut self) -> Vec<(&mut PreparedBytes, bool)> {
        self.precommitted
            .iter_mut()
            .chain(std::iter::once(&mut self.trace))
            .chain(self.responses.values_mut())
            .map(|bytes| (bytes, false))
            .collect()
    }
}

fn encode(value: &impl AkitaSerialize) -> Result<PreparedBytes, OpeningsError> {
    let mut bytes = Vec::new();
    value
        .serialize_with_mode(&mut bytes, Compress::No)
        .map_err(invalid_setup)?;
    Ok(bytes.into())
}

impl WhiteoutPayloads {
    fn new(setup: &WhiteoutSetup) -> Result<Self, OpeningsError> {
        let prepared = setup
            .prepared
            .as_ref()
            .ok_or_else(|| invalid_setup("Whiteout setup is not prepared"))?;
        let shape = prepared
            .shapes
            .values()
            .next()
            .ok_or_else(|| invalid_setup("Whiteout setup has no rows"))?;
        let streams = &shape.tail.streams;
        let material = &prepared.catalog.material;
        let mut masked = BTreeMap::new();
        for shape in prepared.shapes.values() {
            let selection = shape.row().selection().map_err(invalid_setup)?;
            let row = prepared
                .catalog
                .scheme
                .schedules()
                .resolve_selection(selection)
                .map_err(invalid_setup)?;
            let bytes =
                build_riscv64_terminal_ntt_cache(material, row.schedule(), selection.row_digest)
                    .map_err(invalid_setup)?;
            let _ = masked.insert(*selection.row_digest.as_bytes(), bytes.into());
        }
        let mut responses = BTreeMap::new();
        for (&num_vars, response) in &prepared.responses.prepared {
            let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(num_vars));
            let row = response
                .row
                .scheme
                .schedules()
                .resolve_key(&key)
                .map_err(invalid_setup)?;
            let bytes = build_riscv64_terminal_ntt_cache(
                &material.with_capacity(num_vars, 1),
                row.schedule(),
                row.selection().row_digest,
            )
            .map_err(invalid_setup)?;
            let _ = responses.insert(num_vars, bytes.into());
        }
        Ok(Self {
            plans: TransportPlans::new(
                setup
                    .export()
                    .ok_or_else(|| invalid_setup("Whiteout plans missing"))?,
            )?,
            key: encode(material)?,
            pad: encode(&streams.pad_data.0)?,
            v: encode(&streams.v_data.0)?,
            masked,
            responses,
        })
    }

    pub fn slots(&mut self) -> Vec<(&mut PreparedBytes, bool)> {
        let mut slots = vec![
            (&mut self.key, true),
            (&mut self.pad, true),
            (&mut self.v, true),
        ];
        slots.extend(self.plans.slots());
        slots.extend(self.masked.values_mut().map(|bytes| (bytes, false)));
        slots.extend(self.responses.values_mut().map(|bytes| (bytes, false)));
        slots
    }

    pub fn material(&self) -> Result<AkitaBackendVerifierSetup, ZkError> {
        Ok(self.key.backend_key().map_err(material_error)?)
    }

    pub fn streams(&self, capacity: WhiteoutCapacity) -> Result<CapacityStreams, ZkError> {
        let decode = |bytes: &PreparedBytes| -> Result<Arc<StreamCoefficients>, ZkError> {
            let expanded = match bytes {
                PreparedBytes::Attached(bytes) => {
                    let (expanded, rest) = AkitaExpandedSetup::borrow_trusted(*bytes)
                        .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
                    if !rest.bytes().is_empty() {
                        return Err(AkitaError::InvalidSetup("trailing stream bytes".into()).into());
                    }
                    expanded
                }
                PreparedBytes::Owned(_) | PreparedBytes::Decoded(_) | PreparedBytes::Detached => {
                    AkitaExpandedSetup::deserialize_with_mode(
                        bytes.trusted_bytes().map_err(material_error)?,
                        Compress::No,
                        Validate::No,
                        &(),
                    )
                    .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?
                }
            };
            Ok(Arc::new(StreamCoefficients(expanded)))
        };
        CapacityStreams::from_streams(capacity, decode(&self.pad)?, decode(&self.v)?)
    }

    pub fn masked_cache(&self, row: &[u8; 32]) -> Result<TrustedTerminalCache<'_>, ZkError> {
        Ok(self
            .masked
            .get(row)
            .ok_or_else(|| AkitaError::InvalidSetup("missing masked terminal payload".into()))?
            .terminal_cache()
            .map_err(material_error)?)
    }

    pub fn response_cache(&self, num_vars: usize) -> Result<TrustedTerminalCache<'_>, ZkError> {
        Ok(self
            .responses
            .get(&num_vars)
            .ok_or_else(|| AkitaError::InvalidSetup("missing response terminal payload".into()))?
            .terminal_cache()
            .map_err(material_error)?)
    }
}

impl WhiteoutSetup {
    pub(crate) fn payload_slots(&mut self) -> Vec<(&mut PreparedBytes, bool)> {
        self.payloads
            .as_deref_mut()
            .map_or_else(Vec::new, WhiteoutPayloads::slots)
    }

    fn attach(&mut self) -> Result<(), OpeningsError> {
        let Some(payloads) = &self.payloads else {
            return Ok(());
        };
        let plans = payloads.plans.load()?;
        let trace = (plans.catalog.num_vars, plans.catalog.num_polys);
        self.persisted = Some(plans);
        self.response_catalog = None;
        self.prepared = Some(Arc::new(
            self.prepare(trace.0, trace.1).map_err(invalid_setup)?,
        ));
        Ok(())
    }
}

impl AkitaVerifierSetup {
    pub(crate) fn prepare_whiteout_payloads(&mut self) -> Result<(), OpeningsError> {
        if let AkitaVerifierScheduleArtifacts::OneHotWhiteout { whiteout, .. } =
            Arc::make_mut(&mut self.schedule_artifacts)
        {
            if whiteout.payloads.is_none() {
                whiteout.payloads = Some(Box::new(WhiteoutPayloads::new(whiteout)?));
            }
        }
        Ok(())
    }

    pub(crate) fn attach_whiteout_payloads(&mut self) -> Result<(), OpeningsError> {
        if let AkitaVerifierScheduleArtifacts::OneHotWhiteout { whiteout, .. } =
            Arc::make_mut(&mut self.schedule_artifacts)
        {
            whiteout.attach()?;
        }
        Ok(())
    }
}
