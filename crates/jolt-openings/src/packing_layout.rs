use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Display, Formatter};

use crate::{OpeningsError, PackingAddress, PackingFamily, PackingFamilyRef, PackingLayout};
use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use jolt_field::Field;
use jolt_poly::Polynomial;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackingWitnessLayout {
    pub families: Vec<PackingLayoutFamily>,
    pub cells: usize,
    pub dimension: usize,
    pub digest: [u8; 32],
}

impl PackingWitnessLayout {
    pub fn new(
        specs: impl IntoIterator<Item = PackingFamilySpec>,
    ) -> Result<Self, PackingLayoutError> {
        let mut specs = specs.into_iter().collect::<Vec<_>>();
        if specs.is_empty() {
            return Err(PackingLayoutError::EmptyLayout);
        }
        specs.sort_by(|left, right| left.id.cmp(&right.id));

        let mut families = Vec::with_capacity(specs.len());
        let mut offset = 0usize;
        let mut previous_id = None;
        for spec in specs {
            if previous_id.as_ref() == Some(&spec.id) {
                return Err(PackingLayoutError::DuplicateFamily { id: spec.id });
            }
            previous_id = Some(spec.id.clone());

            if spec.limbs == 0 {
                return Err(PackingLayoutError::ZeroLimbs { id: spec.id });
            }
            let rows = spec.domain.rows()?;
            let alphabet_size = spec.alphabet.size();
            if alphabet_size == 0 {
                return Err(PackingLayoutError::ZeroAlphabet { id: spec.id });
            }
            let row_cells = rows
                .checked_mul(spec.limbs)
                .and_then(|value| value.checked_mul(alphabet_size))
                .ok_or_else(|| PackingLayoutError::CellCountOverflow {
                    id: spec.id.clone(),
                })?;
            let cell_count = row_cells;
            let next_offset = offset.checked_add(cell_count).ok_or_else(|| {
                PackingLayoutError::CellCountOverflow {
                    id: spec.id.clone(),
                }
            })?;

            families.push(PackingLayoutFamily {
                id: spec.id,
                domain: spec.domain,
                limbs: spec.limbs,
                alphabet: spec.alphabet,
                offset,
                cell_count,
                view_kind: spec.view_kind,
            });
            offset = next_offset;
        }

        let cells = offset;
        let dimension = ceil_log2(cells);
        if dimension >= usize::BITS as usize {
            return Err(PackingLayoutError::HypercubeTooLarge { dimension });
        }
        let digest = layout_digest(&families, cells, dimension);

        Ok(Self {
            families,
            cells,
            dimension,
            digest,
        })
    }

    pub fn family(&self, id: &PackingFamilyId) -> Option<&PackingLayoutFamily> {
        self.families.iter().find(|family| family.id == *id)
    }

    pub fn rank(&self, address: &PackingCellAddress) -> Result<usize, PackingLayoutError> {
        let family =
            self.family(&address.family)
                .ok_or_else(|| PackingLayoutError::MissingFamily {
                    id: address.family.clone(),
                })?;
        let rows = family.domain.rows()?;
        let alphabet_size = family.alphabet.size();
        if address.row >= rows || address.limb >= family.limbs || address.symbol >= alphabet_size {
            return Err(PackingLayoutError::AddressOutOfRange {
                family: address.family.clone(),
                row: address.row,
                limb: address.limb,
                symbol: address.symbol,
            });
        }

        let local = address
            .row
            .checked_mul(family.limbs)
            .and_then(|value| value.checked_add(address.limb))
            .and_then(|value| value.checked_mul(alphabet_size))
            .and_then(|value| value.checked_add(address.symbol))
            .ok_or_else(|| PackingLayoutError::CellCountOverflow {
                id: address.family.clone(),
            })?;
        family
            .offset
            .checked_add(local)
            .ok_or_else(|| PackingLayoutError::CellCountOverflow {
                id: address.family.clone(),
            })
    }

    pub fn unrank(&self, rank: usize) -> Option<PackingCellAddress> {
        if rank >= self.cells {
            return None;
        }

        self.families.iter().find_map(|family| {
            let end = family.offset.checked_add(family.cell_count)?;
            if rank < family.offset || rank >= end {
                return None;
            }
            let alphabet_size = family.alphabet.size();
            if alphabet_size == 0 {
                return None;
            }
            let local = rank - family.offset;
            let symbol = local % alphabet_size;
            let row_limb = local / alphabet_size;
            let limb = row_limb % family.limbs;
            let row = row_limb / family.limbs;
            Some(PackingCellAddress {
                family: family.id.clone(),
                row,
                limb,
                symbol,
            })
        })
    }

    pub fn dummy_cell_count(&self) -> usize {
        (1usize << self.dimension) - self.cells
    }

    pub fn audit(&self) -> PackingLayoutAudit {
        let mut alphabet_counts = PackingAlphabetCounts::default();
        let mut cells_by_domain = PackingDomainCellCounts::default();
        let mut trace_cells_by_log_t = BTreeMap::<usize, usize>::new();
        let mut rectangular_lane_equivalent = 0usize;

        for family in &self.families {
            match family.alphabet {
                PackingAlphabet::Bit => alphabet_counts.bit += 1,
                PackingAlphabet::Byte => alphabet_counts.byte += 1,
                PackingAlphabet::Fixed { .. } => alphabet_counts.fixed += 1,
            }

            match family.domain {
                PackingFactDomain::TraceRows { log_t } => {
                    cells_by_domain.trace_rows =
                        cells_by_domain.trace_rows.saturating_add(family.cell_count);
                    let entry = trace_cells_by_log_t.entry(log_t).or_default();
                    *entry = entry.saturating_add(family.cell_count);
                }
                PackingFactDomain::BytecodeRows { .. } => {
                    cells_by_domain.bytecode_rows = cells_by_domain
                        .bytecode_rows
                        .saturating_add(family.cell_count);
                }
                PackingFactDomain::ProgramImageWords { .. } => {
                    cells_by_domain.program_image_words = cells_by_domain
                        .program_image_words
                        .saturating_add(family.cell_count);
                }
                PackingFactDomain::AdviceBytes { .. } => {
                    cells_by_domain.advice_bytes = cells_by_domain
                        .advice_bytes
                        .saturating_add(family.cell_count);
                }
            }

            if let Ok(rows) = family.domain.rows() {
                rectangular_lane_equivalent = rectangular_lane_equivalent.saturating_add(
                    rows.saturating_mul(family.limbs).saturating_mul(
                        family
                            .alphabet
                            .rectangular_cell_size()
                            .unwrap_or(usize::MAX),
                    ),
                );
            }
        }

        let trace_cells_per_row = if trace_cells_by_log_t.len() == 1 {
            trace_cells_by_log_t
                .first_key_value()
                .and_then(|(&log_t, &trace_cells)| {
                    if log_t < usize::BITS as usize {
                        Some(trace_cells >> log_t)
                    } else {
                        None
                    }
                })
        } else {
            None
        };

        PackingLayoutAudit {
            fact_count_by_alphabet: alphabet_counts,
            cells_by_domain,
            trace_cells_per_row,
            rectangular_lane_equivalent,
            d_pack: self.dimension,
        }
    }

    pub fn validate_view_families(
        &self,
        families: &[PackingFamilyId],
    ) -> Result<(), PackingLayoutError> {
        for id in families {
            if self.family(id).is_none() {
                return Err(PackingLayoutError::MissingViewFamily { id: id.clone() });
            }
        }
        Ok(())
    }
}

impl PackingLayout for PackingWitnessLayout {
    fn digest(&self) -> [u8; 32] {
        self.digest
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn cells(&self) -> usize {
        self.cells
    }

    fn family(&self, family: PackingFamilyRef) -> Result<Option<PackingFamily>, OpeningsError> {
        self.families
            .iter()
            .find(|candidate| candidate.id.physical_ref() == family)
            .map(|candidate| {
                let rows = candidate.domain.rows().map_err(layout_error)?;
                Ok(PackingFamily {
                    id: family,
                    offset: candidate.offset,
                    rows,
                    limbs: candidate.limbs,
                    alphabet_size: candidate.alphabet.size(),
                })
            })
            .transpose()
    }

    fn rank(&self, address: PackingAddress) -> Result<usize, OpeningsError> {
        let family = self
            .families
            .iter()
            .find(|candidate| candidate.id.physical_ref() == address.family)
            .ok_or_else(|| {
                OpeningsError::InvalidBatch("packing term references an unknown family".to_string())
            })?;
        self.rank(&PackingCellAddress {
            family: family.id.clone(),
            row: address.row,
            limb: address.limb,
            symbol: address.symbol,
        })
        .map_err(layout_error)
    }
}

fn layout_error(error: impl ToString) -> OpeningsError {
    OpeningsError::InvalidBatch(error.to_string())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackingLayoutFamily {
    pub id: PackingFamilyId,
    pub domain: PackingFactDomain,
    pub limbs: usize,
    pub alphabet: PackingAlphabet,
    pub offset: usize,
    pub cell_count: usize,
    pub view_kind: PackingViewKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackingFamilySpec {
    pub id: PackingFamilyId,
    pub domain: PackingFactDomain,
    pub limbs: usize,
    pub alphabet: PackingAlphabet,
    pub view_kind: PackingViewKind,
}

impl PackingFamilySpec {
    pub fn new(
        id: PackingFamilyId,
        domain: PackingFactDomain,
        limbs: usize,
        alphabet: PackingAlphabet,
        view_kind: PackingViewKind,
    ) -> Self {
        Self {
            id,
            domain,
            limbs,
            alphabet,
            view_kind,
        }
    }

    pub fn direct(
        id: PackingFamilyId,
        domain: PackingFactDomain,
        limbs: usize,
        alphabet: PackingAlphabet,
    ) -> Self {
        Self::new(id, domain, limbs, alphabet, PackingViewKind::Direct)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PackingFamilyId {
    InstructionRa {
        index: usize,
    },
    BytecodeRa {
        index: usize,
    },
    RamRa {
        index: usize,
    },
    UnsignedIncChunk {
        index: usize,
    },
    UnsignedIncMsb,
    FieldRdIncByte {
        index: usize,
    },
    FieldRdIncSign,
    AdviceBytes {
        kind: PackingAdviceKind,
        index: usize,
    },
    BytecodeChunk {
        index: usize,
    },
    BytecodeRegisterSelector {
        chunk: usize,
        selector: usize,
    },
    BytecodeCircuitFlag {
        chunk: usize,
        flag: usize,
    },
    BytecodeInstructionFlag {
        chunk: usize,
        flag: usize,
    },
    BytecodeLookupSelector {
        chunk: usize,
    },
    BytecodeRafFlag {
        chunk: usize,
    },
    BytecodeUnexpandedPcBytes {
        chunk: usize,
    },
    BytecodeImmBytes {
        chunk: usize,
    },
    ProgramImageInit,
    Custom {
        namespace: u32,
        index: usize,
    },
}

const JOLT_PACKING_FAMILY_NAMESPACE: u64 = 0x6a6f_6c74_7063_7301;

impl PackingFamilyId {
    pub fn physical_ref(&self) -> PackingFamilyRef {
        let (id, index): (u64, u64) = match self {
            Self::InstructionRa { index } => (0, *index as u64),
            Self::BytecodeRa { index } => (1, *index as u64),
            Self::RamRa { index } => (2, *index as u64),
            Self::UnsignedIncChunk { index } => (3, *index as u64),
            Self::UnsignedIncMsb => (4, 0),
            Self::FieldRdIncByte { index } => (9, *index as u64),
            Self::FieldRdIncSign => (10, 0),
            Self::AdviceBytes { kind, index } => match kind {
                PackingAdviceKind::Trusted => (11, *index as u64),
                PackingAdviceKind::Untrusted => (12, *index as u64),
            },
            Self::BytecodeChunk { index } => (13, *index as u64),
            Self::ProgramImageInit => (14, 0),
            Self::BytecodeRegisterSelector { chunk, selector } => {
                (15, combine_two_indices(*chunk, *selector))
            }
            Self::BytecodeCircuitFlag { chunk, flag } => (16, combine_two_indices(*chunk, *flag)),
            Self::BytecodeInstructionFlag { chunk, flag } => {
                (17, combine_two_indices(*chunk, *flag))
            }
            Self::BytecodeLookupSelector { chunk } => (18, *chunk as u64),
            Self::BytecodeRafFlag { chunk } => (19, *chunk as u64),
            Self::BytecodeUnexpandedPcBytes { chunk } => (20, *chunk as u64),
            Self::BytecodeImmBytes { chunk } => (21, *chunk as u64),
            Self::Custom { namespace, index } => {
                return PackingFamilyRef::new(u64::from(*namespace), 0, *index as u64);
            }
        };
        PackingFamilyRef::new(JOLT_PACKING_FAMILY_NAMESPACE, id, index)
    }
}

fn combine_two_indices(left: usize, right: usize) -> u64 {
    ((left as u64) << 32) | right as u64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PackingAdviceKind {
    Trusted,
    Untrusted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackingFactDomain {
    TraceRows {
        log_t: usize,
    },
    BytecodeRows {
        log_bytecode: usize,
    },
    ProgramImageWords {
        log_words: usize,
    },
    AdviceBytes {
        kind: PackingAdviceKind,
        log_bytes: usize,
    },
}

impl PackingFactDomain {
    pub fn rows(&self) -> Result<usize, PackingLayoutError> {
        let log_rows = match *self {
            Self::TraceRows { log_t } => log_t,
            Self::BytecodeRows { log_bytecode } => log_bytecode,
            Self::ProgramImageWords { log_words } => log_words,
            Self::AdviceBytes { log_bytes, .. } => log_bytes,
        };
        if log_rows >= usize::BITS as usize {
            return Err(PackingLayoutError::DomainTooLarge { log_rows });
        }
        Ok(1usize << log_rows)
    }

    fn digest_tag(self) -> u8 {
        match self {
            Self::TraceRows { .. } => 0,
            Self::BytecodeRows { .. } => 1,
            Self::ProgramImageWords { .. } => 2,
            Self::AdviceBytes { .. } => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackingAlphabet {
    Bit,
    Byte,
    Fixed { size: usize },
}

impl PackingAlphabet {
    pub fn size(self) -> usize {
        match self {
            Self::Bit => 2,
            Self::Byte => 256,
            Self::Fixed { size } => size,
        }
    }

    fn rectangular_cell_size(self) -> Option<usize> {
        let size = self.size();
        if size == 0 {
            return None;
        }
        Some(size.div_ceil(256).saturating_mul(256))
    }

    fn digest_tag(self) -> u8 {
        match self {
            Self::Bit => 0,
            Self::Byte => 1,
            Self::Fixed { .. } => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackingViewKind {
    Direct,
    LinearDecode,
    MaskedReduction,
}

impl PackingViewKind {
    fn digest_tag(self) -> u8 {
        match self {
            Self::Direct => 0,
            Self::LinearDecode => 1,
            Self::MaskedReduction => 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackingCellAddress {
    pub family: PackingFamilyId,
    pub row: usize,
    pub limb: usize,
    pub symbol: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackingAlphabetCounts {
    pub bit: usize,
    pub byte: usize,
    pub fixed: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackingDomainCellCounts {
    pub trace_rows: usize,
    pub bytecode_rows: usize,
    pub program_image_words: usize,
    pub advice_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackingLayoutAudit {
    pub fact_count_by_alphabet: PackingAlphabetCounts,
    pub cells_by_domain: PackingDomainCellCounts,
    pub trace_cells_per_row: Option<usize>,
    pub rectangular_lane_equivalent: usize,
    pub d_pack: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackingValidityKind {
    ExactOneHot,
    OptionalOneHot,
    BooleanIndicator { symbol: usize },
    BytecodeStoreRdDisjoint,
    FieldElementCanonicalBytes { byte_width: usize, modulus: u128 },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackingValidityRequirement {
    pub family: PackingFamilyId,
    pub limbs: usize,
    pub alphabet_size: usize,
    pub kind: PackingValidityKind,
}

impl PackingValidityRequirement {
    pub fn exact_one_hot(family: PackingFamilyId, limbs: usize, alphabet_size: usize) -> Self {
        Self {
            family,
            limbs,
            alphabet_size,
            kind: PackingValidityKind::ExactOneHot,
        }
    }

    pub fn optional_one_hot(family: PackingFamilyId, limbs: usize, alphabet_size: usize) -> Self {
        Self {
            family,
            limbs,
            alphabet_size,
            kind: PackingValidityKind::OptionalOneHot,
        }
    }

    pub fn boolean_indicator(
        family: PackingFamilyId,
        limbs: usize,
        alphabet_size: usize,
        symbol: usize,
    ) -> Self {
        Self {
            family,
            limbs,
            alphabet_size,
            kind: PackingValidityKind::BooleanIndicator { symbol },
        }
    }

    pub fn bytecode_store_rd_disjoint(chunk: usize, store_flag: usize) -> Self {
        Self {
            family: PackingFamilyId::BytecodeCircuitFlag {
                chunk,
                flag: store_flag,
            },
            limbs: 1,
            alphabet_size: 2,
            kind: PackingValidityKind::BytecodeStoreRdDisjoint,
        }
    }

    pub fn field_element_canonical_bytes(
        family: PackingFamilyId,
        byte_width: usize,
        modulus: u128,
    ) -> Self {
        Self {
            family,
            limbs: 1,
            alphabet_size: 256,
            kind: PackingValidityKind::FieldElementCanonicalBytes {
                byte_width,
                modulus,
            },
        }
    }
}

pub type PackingValidityDigest = [u8; 32];

const PACKING_VALIDITY_DIGEST_DOMAIN: &[u8] = b"jolt-claims/lattice-packed-validity/v1";

pub fn packing_validity_digest(
    requirements: &[PackingValidityRequirement],
) -> PackingValidityDigest {
    let mut encoded_requirements = requirements
        .iter()
        .map(encode_validity_requirement)
        .collect::<Vec<_>>();
    encoded_requirements.sort();

    let mut bytes = Vec::new();
    // Preserve the pre-refactor domain separator so existing configs keep matching.
    bytes.extend_from_slice(PACKING_VALIDITY_DIGEST_DOMAIN);
    write_usize(&mut bytes, encoded_requirements.len());
    for requirement in encoded_requirements {
        bytes.extend_from_slice(&requirement);
    }

    let mut hasher = Blake2b::<U32>::new();
    hasher.update(&bytes);
    let result = hasher.finalize();
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&result);
    digest
}

fn encode_validity_requirement(requirement: &PackingValidityRequirement) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_family_id(&mut bytes, &requirement.family);
    write_usize(&mut bytes, requirement.limbs);
    write_usize(&mut bytes, requirement.alphabet_size);
    write_validity_kind(&mut bytes, &requirement.kind);
    bytes
}

fn write_validity_kind(bytes: &mut Vec<u8>, kind: &PackingValidityKind) {
    match kind {
        PackingValidityKind::ExactOneHot => bytes.push(0),
        PackingValidityKind::OptionalOneHot => bytes.push(1),
        PackingValidityKind::BooleanIndicator { symbol } => {
            bytes.push(2);
            write_usize(bytes, *symbol);
        }
        PackingValidityKind::BytecodeStoreRdDisjoint => bytes.push(3),
        PackingValidityKind::FieldElementCanonicalBytes {
            byte_width,
            modulus,
        } => {
            bytes.push(4);
            write_usize(bytes, *byte_width);
            bytes.extend_from_slice(&modulus.to_le_bytes());
        }
    }
}

pub trait PackingWitnessSource<F: Field> {
    fn layout(&self) -> &PackingWitnessLayout;
    fn for_each_nonzero(&self, f: impl FnMut(usize, F));
    fn eval_direct_fact(&self, address: &PackingCellAddress) -> Result<F, PackingLayoutError>;
}

pub fn validate_packing_source_layout<F, S>(
    layout: &PackingWitnessLayout,
    source: &S,
) -> Result<(), OpeningsError>
where
    F: Field,
    S: PackingWitnessSource<F>,
{
    let source_layout = source.layout();
    if source_layout.digest != layout.digest || source_layout.dimension != layout.dimension {
        return Err(invalid_packing_source(
            "packed witness source layout does not match packed statement",
        ));
    }
    Ok(())
}

pub fn validate_packing_source_dimension(
    max_num_vars: usize,
    layout: &PackingWitnessLayout,
) -> Result<(), OpeningsError> {
    if layout.dimension > max_num_vars {
        return Err(OpeningsError::PolynomialTooLarge {
            poly_size: layout.dimension,
            setup_max: max_num_vars,
        });
    }
    Ok(())
}

pub fn packing_witness_source_polynomial<F, S>(source: &S) -> Result<Polynomial<F>, OpeningsError>
where
    F: Field,
    S: PackingWitnessSource<F>,
{
    let layout = source.layout();
    if layout.cells == 0 {
        return Err(invalid_packing_source(
            "packed witness layout must contain at least one cell",
        ));
    }
    if layout.dimension >= usize::BITS as usize {
        return Err(invalid_packing_source(format!(
            "packed witness dimension {} exceeds usize bit width",
            layout.dimension
        )));
    }
    let domain_size = 1usize << layout.dimension;
    if layout.cells > domain_size {
        return Err(invalid_packing_source(format!(
            "packed witness has {} cells but dimension {} supports {domain_size}",
            layout.cells, layout.dimension
        )));
    }

    let mut evals = vec![F::zero(); domain_size];
    let mut seen = vec![false; layout.cells];
    let mut result = Ok(());
    source.for_each_nonzero(|rank, value| {
        if result.is_err() {
            return;
        }
        if rank >= layout.cells {
            result = Err(invalid_packing_source(format!(
                "packed witness source emitted rank {rank} outside {} real cells",
                layout.cells
            )));
            return;
        }
        if seen[rank] {
            result = Err(invalid_packing_source(format!(
                "packed witness source emitted rank {rank} more than once"
            )));
            return;
        }
        if value.is_zero() {
            result = Err(invalid_packing_source(format!(
                "packed witness source emitted zero at rank {rank}"
            )));
            return;
        }

        seen[rank] = true;
        evals[rank] = value;
    });
    result?;

    Ok(Polynomial::new(evals))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparsePackingWitness<F> {
    layout: PackingWitnessLayout,
    entries: Vec<(usize, F)>,
}

fn invalid_packing_source(reason: impl Into<String>) -> OpeningsError {
    OpeningsError::InvalidBatch(reason.into())
}

impl<F: Field> SparsePackingWitness<F> {
    pub fn try_new(
        layout: PackingWitnessLayout,
        mut entries: Vec<(usize, F)>,
    ) -> Result<Self, PackingLayoutError> {
        entries.sort_by_key(|(rank, _)| *rank);
        let mut previous_rank = None;
        for (rank, value) in &entries {
            if *rank >= layout.cells {
                return Err(PackingLayoutError::RankOutOfRange {
                    rank: *rank,
                    cells: layout.cells,
                });
            }
            if previous_rank == Some(*rank) {
                return Err(PackingLayoutError::DuplicateSourceRank { rank: *rank });
            }
            if value.is_zero() {
                return Err(PackingLayoutError::ZeroSourceEntry { rank: *rank });
            }
            previous_rank = Some(*rank);
        }
        Ok(Self { layout, entries })
    }

    pub fn try_from_cells(
        layout: PackingWitnessLayout,
        cells: impl IntoIterator<Item = (PackingCellAddress, F)>,
    ) -> Result<Self, PackingLayoutError> {
        let entries = cells
            .into_iter()
            .map(|(address, value)| layout.rank(&address).map(|rank| (rank, value)))
            .collect::<Result<Vec<_>, _>>()?;
        Self::try_new(layout, entries)
    }

    pub fn entries(&self) -> &[(usize, F)] {
        &self.entries
    }
}

impl<F: Field> PackingWitnessSource<F> for SparsePackingWitness<F> {
    fn layout(&self) -> &PackingWitnessLayout {
        &self.layout
    }

    fn for_each_nonzero(&self, mut f: impl FnMut(usize, F)) {
        for &(rank, value) in &self.entries {
            f(rank, value);
        }
    }

    fn eval_direct_fact(&self, address: &PackingCellAddress) -> Result<F, PackingLayoutError> {
        let rank = self.layout.rank(address)?;
        match self.entries.binary_search_by_key(&rank, |(rank, _)| *rank) {
            Ok(index) => Ok(self.entries[index].1),
            Err(_) => Ok(F::zero()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackingLayoutError {
    EmptyLayout,
    DuplicateFamily {
        id: PackingFamilyId,
    },
    ZeroLimbs {
        id: PackingFamilyId,
    },
    ZeroAlphabet {
        id: PackingFamilyId,
    },
    DomainTooLarge {
        log_rows: usize,
    },
    CellCountOverflow {
        id: PackingFamilyId,
    },
    HypercubeTooLarge {
        dimension: usize,
    },
    MissingFamily {
        id: PackingFamilyId,
    },
    AddressOutOfRange {
        family: PackingFamilyId,
        row: usize,
        limb: usize,
        symbol: usize,
    },
    RankOutOfRange {
        rank: usize,
        cells: usize,
    },
    DuplicateSourceRank {
        rank: usize,
    },
    ZeroSourceEntry {
        rank: usize,
    },
    MissingViewFamily {
        id: PackingFamilyId,
    },
}

impl Display for PackingLayoutError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLayout => f.write_str("packed witness layout must contain a family"),
            Self::DuplicateFamily { id } => {
                write!(f, "duplicate packed witness family {id:?}")
            }
            Self::ZeroLimbs { id } => {
                write!(f, "packed witness family {id:?} has zero limbs")
            }
            Self::ZeroAlphabet { id } => {
                write!(f, "packed witness family {id:?} has zero alphabet size")
            }
            Self::DomainTooLarge { log_rows } => {
                write!(f, "packed witness domain 2^{log_rows} does not fit in usize")
            }
            Self::CellCountOverflow { id } => {
                write!(f, "packed witness family {id:?} cell count overflows usize")
            }
            Self::HypercubeTooLarge { dimension } => {
                write!(f, "packed witness hypercube dimension {dimension} is too large")
            }
            Self::MissingFamily { id } => {
                write!(f, "packed witness family {id:?} is not in the layout")
            }
            Self::AddressOutOfRange {
                family,
                row,
                limb,
                symbol,
            } => write!(
                f,
                "packed witness address ({family:?}, row {row}, limb {limb}, symbol {symbol}) is out of range"
            ),
            Self::RankOutOfRange { rank, cells } => {
                write!(f, "packed witness source rank {rank} is outside {cells} cells")
            }
            Self::DuplicateSourceRank { rank } => {
                write!(f, "packed witness source rank {rank} appears twice")
            }
            Self::ZeroSourceEntry { rank } => {
                write!(f, "packed witness source rank {rank} is zero")
            }
            Self::MissingViewFamily { id } => {
                write!(f, "packed witness view references missing family {id:?}")
            }
        }
    }
}

impl Error for PackingLayoutError {}

fn ceil_log2(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

fn layout_digest(families: &[PackingLayoutFamily], cells: usize, dimension: usize) -> [u8; 32] {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"jolt-openings/packing-layout/v1");
    write_usize(&mut bytes, families.len());
    write_usize(&mut bytes, cells);
    write_usize(&mut bytes, dimension);
    for family in families {
        write_family_id(&mut bytes, &family.id);
        write_domain(&mut bytes, family.domain);
        write_usize(&mut bytes, family.limbs);
        write_alphabet(&mut bytes, family.alphabet);
        write_usize(&mut bytes, family.offset);
        write_usize(&mut bytes, family.cell_count);
        bytes.push(family.view_kind.digest_tag());
    }

    let mut hasher = Blake2b::<U32>::new();
    hasher.update(&bytes);
    let result = hasher.finalize();
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&result);
    digest
}

fn write_family_id(bytes: &mut Vec<u8>, id: &PackingFamilyId) {
    match id {
        PackingFamilyId::InstructionRa { index } => {
            bytes.push(0);
            write_usize(bytes, *index);
        }
        PackingFamilyId::BytecodeRa { index } => {
            bytes.push(1);
            write_usize(bytes, *index);
        }
        PackingFamilyId::RamRa { index } => {
            bytes.push(2);
            write_usize(bytes, *index);
        }
        PackingFamilyId::UnsignedIncChunk { index } => {
            bytes.push(3);
            write_usize(bytes, *index);
        }
        PackingFamilyId::UnsignedIncMsb => bytes.push(4),
        PackingFamilyId::FieldRdIncByte { index } => {
            bytes.push(9);
            write_usize(bytes, *index);
        }
        PackingFamilyId::FieldRdIncSign => bytes.push(10),
        PackingFamilyId::AdviceBytes { kind, index } => {
            bytes.push(11);
            bytes.push(advice_kind_tag(*kind));
            write_usize(bytes, *index);
        }
        PackingFamilyId::BytecodeChunk { index } => {
            bytes.push(12);
            write_usize(bytes, *index);
        }
        PackingFamilyId::ProgramImageInit => bytes.push(13),
        PackingFamilyId::BytecodeRegisterSelector { chunk, selector } => {
            bytes.push(15);
            write_usize(bytes, *chunk);
            write_usize(bytes, *selector);
        }
        PackingFamilyId::BytecodeCircuitFlag { chunk, flag } => {
            bytes.push(16);
            write_usize(bytes, *chunk);
            write_usize(bytes, *flag);
        }
        PackingFamilyId::BytecodeInstructionFlag { chunk, flag } => {
            bytes.push(17);
            write_usize(bytes, *chunk);
            write_usize(bytes, *flag);
        }
        PackingFamilyId::BytecodeLookupSelector { chunk } => {
            bytes.push(18);
            write_usize(bytes, *chunk);
        }
        PackingFamilyId::BytecodeRafFlag { chunk } => {
            bytes.push(19);
            write_usize(bytes, *chunk);
        }
        PackingFamilyId::BytecodeUnexpandedPcBytes { chunk } => {
            bytes.push(20);
            write_usize(bytes, *chunk);
        }
        PackingFamilyId::BytecodeImmBytes { chunk } => {
            bytes.push(21);
            write_usize(bytes, *chunk);
        }
        PackingFamilyId::Custom { namespace, index } => {
            bytes.push(14);
            bytes.extend_from_slice(&namespace.to_le_bytes());
            write_usize(bytes, *index);
        }
    }
}

fn write_domain(bytes: &mut Vec<u8>, domain: PackingFactDomain) {
    bytes.push(domain.digest_tag());
    match domain {
        PackingFactDomain::TraceRows { log_t } => write_usize(bytes, log_t),
        PackingFactDomain::BytecodeRows { log_bytecode } => write_usize(bytes, log_bytecode),
        PackingFactDomain::ProgramImageWords { log_words } => write_usize(bytes, log_words),
        PackingFactDomain::AdviceBytes { kind, log_bytes } => {
            bytes.push(advice_kind_tag(kind));
            write_usize(bytes, log_bytes);
        }
    }
}

fn write_alphabet(bytes: &mut Vec<u8>, alphabet: PackingAlphabet) {
    bytes.push(alphabet.digest_tag());
    match alphabet {
        PackingAlphabet::Bit | PackingAlphabet::Byte => {}
        PackingAlphabet::Fixed { size } => write_usize(bytes, size),
    }
}

fn write_usize(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

fn advice_kind_tag(kind: PackingAdviceKind) -> u8 {
    match kind {
        PackingAdviceKind::Trusted => 0,
        PackingAdviceKind::Untrusted => 1,
    }
}

#[cfg(test)]
mod tests {

    #![expect(
        clippy::expect_used,
        reason = "tests assert successful layout construction"
    )]

    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn trace(log_t: usize) -> PackingFactDomain {
        PackingFactDomain::TraceRows { log_t }
    }

    fn byte_family(id: PackingFamilyId, log_t: usize) -> PackingFamilySpec {
        PackingFamilySpec::direct(id, trace(log_t), 1, PackingAlphabet::Byte)
    }

    fn bit_family(id: PackingFamilyId, log_t: usize) -> PackingFamilySpec {
        PackingFamilySpec::direct(id, trace(log_t), 1, PackingAlphabet::Bit)
    }

    fn base_ra_specs(log_t: usize) -> Vec<PackingFamilySpec> {
        let mut specs = Vec::new();
        specs.extend(
            (0..16).map(|index| byte_family(PackingFamilyId::InstructionRa { index }, log_t)),
        );
        specs.extend((0..3).map(|index| byte_family(PackingFamilyId::BytecodeRa { index }, log_t)));
        specs.extend((0..4).map(|index| byte_family(PackingFamilyId::RamRa { index }, log_t)));
        specs
    }

    fn unsigned_increment_specs(log_t: usize) -> Vec<PackingFamilySpec> {
        let mut specs = (0..8)
            .map(|index| byte_family(PackingFamilyId::UnsignedIncChunk { index }, log_t))
            .collect::<Vec<_>>();
        specs.push(bit_family(PackingFamilyId::UnsignedIncMsb, log_t));
        specs
    }

    #[test]
    fn packed_witness_layout_digest_stable() {
        let mut specs = vec![
            byte_family(PackingFamilyId::RamRa { index: 0 }, 4),
            bit_family(PackingFamilyId::UnsignedIncMsb, 4),
            byte_family(PackingFamilyId::InstructionRa { index: 0 }, 4),
        ];
        let layout_a = PackingWitnessLayout::new(specs.clone()).expect("layout should build");
        specs.reverse();
        let layout_b = PackingWitnessLayout::new(specs).expect("layout should build");

        assert_eq!(layout_a.digest, layout_b.digest);
        assert_eq!(layout_a.families, layout_b.families);
    }

    #[test]
    fn packed_witness_layout_rejects_duplicate_ranges() {
        let specs = vec![
            byte_family(PackingFamilyId::RamRa { index: 0 }, 3),
            byte_family(PackingFamilyId::RamRa { index: 0 }, 3),
        ];
        assert!(matches!(
            PackingWitnessLayout::new(specs),
            Err(PackingLayoutError::DuplicateFamily { .. })
        ));
    }

    #[test]
    fn large_trace_base_cells_are_5888_per_row() {
        let layout = PackingWitnessLayout::new(base_ra_specs(20)).expect("layout should build");
        let audit = layout.audit();

        assert_eq!(audit.trace_cells_per_row, Some(5_888));
        assert_eq!(layout.dimension, 33);
    }

    #[test]
    fn unsigned_increment_budget_is_n_plus_13() {
        let log_t = 20;
        let mut specs = base_ra_specs(log_t);
        specs.extend(unsigned_increment_specs(log_t));

        let layout = PackingWitnessLayout::new(specs).expect("layout should build");
        let audit = layout.audit();

        assert_eq!(audit.trace_cells_per_row, Some(7_938));
        assert_eq!(layout.dimension, log_t + 13);
    }

    #[test]
    fn bit_fact_costs_two_cells_per_row() {
        let layout = PackingWitnessLayout::new([bit_family(PackingFamilyId::UnsignedIncMsb, 5)])
            .expect("layout should build");
        let audit = layout.audit();

        assert_eq!(layout.cells, 64);
        assert_eq!(audit.trace_cells_per_row, Some(2));
        assert_eq!(audit.fact_count_by_alphabet.bit, 1);
    }

    #[test]
    fn rank_unrank_roundtrip() {
        let layout = PackingWitnessLayout::new([
            byte_family(PackingFamilyId::RamRa { index: 0 }, 1),
            bit_family(PackingFamilyId::UnsignedIncMsb, 1),
            PackingFamilySpec::direct(
                PackingFamilyId::BytecodeChunk { index: 0 },
                PackingFactDomain::BytecodeRows { log_bytecode: 1 },
                2,
                PackingAlphabet::Fixed { size: 3 },
            ),
        ])
        .expect("layout should build");

        for rank in 0..layout.cells {
            let address = layout.unrank(rank).expect("non-dummy rank should unrank");
            assert_eq!(layout.rank(&address).expect("address should rank"), rank);
        }
    }

    #[test]
    fn committed_bytecode_lane_families_are_distinct() {
        let bytecode = PackingFactDomain::BytecodeRows { log_bytecode: 1 };
        let families = [
            PackingFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 0,
            },
            PackingFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 1,
            },
            PackingFamilyId::BytecodeCircuitFlag { chunk: 0, flag: 0 },
            PackingFamilyId::BytecodeInstructionFlag { chunk: 0, flag: 0 },
            PackingFamilyId::BytecodeLookupSelector { chunk: 0 },
            PackingFamilyId::BytecodeRafFlag { chunk: 0 },
            PackingFamilyId::BytecodeUnexpandedPcBytes { chunk: 0 },
            PackingFamilyId::BytecodeImmBytes { chunk: 0 },
        ];
        let layout = PackingWitnessLayout::new([
            PackingFamilySpec::direct(
                families[0].clone(),
                bytecode,
                1,
                PackingAlphabet::Fixed { size: 32 },
            ),
            PackingFamilySpec::direct(
                families[1].clone(),
                bytecode,
                1,
                PackingAlphabet::Fixed { size: 32 },
            ),
            PackingFamilySpec::direct(families[2].clone(), bytecode, 1, PackingAlphabet::Bit),
            PackingFamilySpec::direct(families[3].clone(), bytecode, 1, PackingAlphabet::Bit),
            PackingFamilySpec::direct(
                families[4].clone(),
                bytecode,
                1,
                PackingAlphabet::Fixed { size: 4 },
            ),
            PackingFamilySpec::direct(families[5].clone(), bytecode, 1, PackingAlphabet::Bit),
            PackingFamilySpec::direct(families[6].clone(), bytecode, 8, PackingAlphabet::Byte),
            PackingFamilySpec::direct(families[7].clone(), bytecode, 16, PackingAlphabet::Byte),
        ])
        .expect("layout should build");

        for family in &families {
            assert!(layout.family(family).is_some());
        }
        for left in 0..families.len() {
            for right in left + 1..families.len() {
                assert_ne!(
                    families[left].physical_ref(),
                    families[right].physical_ref()
                );
            }
        }

        let address = PackingCellAddress {
            family: PackingFamilyId::BytecodeImmBytes { chunk: 0 },
            row: 1,
            limb: 15,
            symbol: 7,
        };
        let rank = layout.rank(&address).expect("bytecode byte address ranks");
        assert_eq!(layout.unrank(rank), Some(address));
    }

    #[test]
    fn dummy_cells_are_zero_and_unreferenced() {
        let layout = PackingWitnessLayout::new([PackingFamilySpec::direct(
            PackingFamilyId::Custom {
                namespace: 1,
                index: 0,
            },
            trace(0),
            1,
            PackingAlphabet::Fixed { size: 3 },
        )])
        .expect("layout should build");

        assert_eq!(layout.cells, 3);
        assert_eq!(layout.dimension, 2);
        assert_eq!(layout.dummy_cell_count(), 1);
        assert!(layout.unrank(layout.cells).is_none());

        let source = SparsePackingWitness::<Fr>::try_new(layout.clone(), Vec::new())
            .expect("empty source should build");
        let zero_address = PackingCellAddress {
            family: PackingFamilyId::Custom {
                namespace: 1,
                index: 0,
            },
            row: 0,
            limb: 0,
            symbol: 2,
        };
        assert_eq!(
            source
                .eval_direct_fact(&zero_address)
                .expect("address is in range"),
            Fr::from_u64(0)
        );
    }

    #[test]
    fn layout_sort_order_is_stable() {
        let layout = PackingWitnessLayout::new([
            byte_family(PackingFamilyId::RamRa { index: 3 }, 2),
            byte_family(PackingFamilyId::InstructionRa { index: 1 }, 2),
            byte_family(PackingFamilyId::BytecodeRa { index: 2 }, 2),
        ])
        .expect("layout should build");

        assert_eq!(
            layout
                .families
                .iter()
                .map(|family| &family.id)
                .collect::<Vec<_>>(),
            vec![
                &PackingFamilyId::InstructionRa { index: 1 },
                &PackingFamilyId::BytecodeRa { index: 2 },
                &PackingFamilyId::RamRa { index: 3 },
            ]
        );
    }

    #[test]
    fn committed_program_families_use_non_trace_domains() {
        let layout = PackingWitnessLayout::new([
            PackingFamilySpec::direct(
                PackingFamilyId::BytecodeChunk { index: 0 },
                PackingFactDomain::BytecodeRows { log_bytecode: 4 },
                1,
                PackingAlphabet::Byte,
            ),
            PackingFamilySpec::direct(
                PackingFamilyId::ProgramImageInit,
                PackingFactDomain::ProgramImageWords { log_words: 3 },
                8,
                PackingAlphabet::Byte,
            ),
        ])
        .expect("layout should build");
        let audit = layout.audit();

        assert_eq!(audit.cells_by_domain.trace_rows, 0);
        assert_eq!(audit.cells_by_domain.bytecode_rows, 16 * 256);
        assert_eq!(audit.cells_by_domain.program_image_words, 8 * 8 * 256);
        assert_eq!(audit.trace_cells_per_row, None);
    }

    #[test]
    fn planner_audit_fields_are_reported() {
        let layout = PackingWitnessLayout::new([
            byte_family(PackingFamilyId::InstructionRa { index: 0 }, 2),
            bit_family(PackingFamilyId::UnsignedIncMsb, 2),
            PackingFamilySpec::direct(
                PackingFamilyId::AdviceBytes {
                    kind: PackingAdviceKind::Trusted,
                    index: 0,
                },
                PackingFactDomain::AdviceBytes {
                    kind: PackingAdviceKind::Trusted,
                    log_bytes: 1,
                },
                1,
                PackingAlphabet::Fixed { size: 4 },
            ),
        ])
        .expect("layout should build");
        let audit = layout.audit();

        assert_eq!(
            audit.fact_count_by_alphabet,
            PackingAlphabetCounts {
                bit: 1,
                byte: 1,
                fixed: 1,
            }
        );
        assert_eq!(audit.cells_by_domain.trace_rows, 4 * 256 + 4 * 2);
        assert_eq!(audit.cells_by_domain.advice_bytes, 2 * 4);
        assert_eq!(audit.d_pack, layout.dimension);
        assert!(audit.rectangular_lane_equivalent >= layout.cells);
    }

    #[test]
    fn packed_witness_source_respects_layout() {
        let layout = PackingWitnessLayout::new([
            byte_family(PackingFamilyId::RamRa { index: 0 }, 1),
            bit_family(PackingFamilyId::UnsignedIncMsb, 1),
        ])
        .expect("layout should build");
        let one_address = PackingCellAddress {
            family: PackingFamilyId::RamRa { index: 0 },
            row: 1,
            limb: 0,
            symbol: 17,
        };
        let sign_address = PackingCellAddress {
            family: PackingFamilyId::UnsignedIncMsb,
            row: 0,
            limb: 0,
            symbol: 1,
        };
        let source = SparsePackingWitness::try_from_cells(
            layout.clone(),
            [
                (one_address.clone(), Fr::from_u64(11)),
                (sign_address.clone(), Fr::from_u64(1)),
            ],
        )
        .expect("source should build");

        let mut streamed = Vec::new();
        source.for_each_nonzero(|rank, value| streamed.push((rank, value)));

        assert_eq!(source.layout().digest, layout.digest);
        assert_eq!(streamed.len(), 2);
        assert!(streamed.iter().all(|(rank, _)| *rank < layout.cells));
        assert_eq!(
            source
                .eval_direct_fact(&one_address)
                .expect("address is in range"),
            Fr::from_u64(11)
        );
        assert_eq!(
            source
                .eval_direct_fact(&sign_address)
                .expect("address is in range"),
            Fr::from_u64(1)
        );

        let polynomial =
            packing_witness_source_polynomial(&source).expect("source should materialize densely");
        assert_eq!(polynomial.num_vars(), layout.dimension);
        assert_eq!(
            polynomial.evaluations()[layout.rank(&one_address).expect("address should rank")],
            Fr::from_u64(11)
        );
        assert_eq!(
            polynomial.evaluations()[layout.rank(&sign_address).expect("address should rank")],
            Fr::from_u64(1)
        );
        assert!(
            polynomial.evaluations()[layout.cells..]
                .iter()
                .all(|value| value == &Fr::from_u64(0)),
            "dummy cells should materialize as zero"
        );
    }

    #[test]
    fn view_catalog_references_existing_families() {
        let layout =
            PackingWitnessLayout::new([byte_family(PackingFamilyId::RamRa { index: 0 }, 2)])
                .expect("layout should build");

        layout
            .validate_view_families(&[PackingFamilyId::RamRa { index: 0 }])
            .expect("existing family should validate");
        assert!(matches!(
            layout.validate_view_families(&[PackingFamilyId::UnsignedIncMsb]),
            Err(PackingLayoutError::MissingViewFamily { .. })
        ));
    }

    #[test]
    fn sparse_source_rejects_out_of_layout_ranks() {
        let layout = PackingWitnessLayout::new([bit_family(PackingFamilyId::UnsignedIncMsb, 0)])
            .expect("layout should build");

        assert!(matches!(
            SparsePackingWitness::try_new(layout.clone(), vec![(layout.cells, Fr::from_u64(1))]),
            Err(PackingLayoutError::RankOutOfRange { .. })
        ));
    }
}
