use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Display, Formatter};

use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::types::{AkitaLayoutDigest, AkitaSetupParams};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedWitnessLayout {
    pub families: Vec<PackedFamily>,
    pub cells: usize,
    pub dimension: usize,
    pub digest: AkitaLayoutDigest,
}

impl PackedWitnessLayout {
    pub fn new(
        specs: impl IntoIterator<Item = PackedFamilySpec>,
    ) -> Result<Self, PackedLayoutError> {
        let mut specs = specs.into_iter().collect::<Vec<_>>();
        if specs.is_empty() {
            return Err(PackedLayoutError::EmptyLayout);
        }
        specs.sort_by(|left, right| left.id.cmp(&right.id));

        let mut families = Vec::with_capacity(specs.len());
        let mut offset = 0usize;
        let mut previous_id = None;
        for spec in specs {
            if previous_id.as_ref() == Some(&spec.id) {
                return Err(PackedLayoutError::DuplicateFamily { id: spec.id });
            }
            previous_id = Some(spec.id.clone());

            if spec.limbs == 0 {
                return Err(PackedLayoutError::ZeroLimbs { id: spec.id });
            }
            let rows = spec.domain.rows()?;
            let alphabet_size = spec.alphabet.size();
            if alphabet_size == 0 {
                return Err(PackedLayoutError::ZeroAlphabet { id: spec.id });
            }
            let row_cells = rows
                .checked_mul(spec.limbs)
                .and_then(|value| value.checked_mul(alphabet_size))
                .ok_or_else(|| PackedLayoutError::CellCountOverflow {
                    id: spec.id.clone(),
                })?;
            let cell_count = row_cells;
            let next_offset = offset.checked_add(cell_count).ok_or_else(|| {
                PackedLayoutError::CellCountOverflow {
                    id: spec.id.clone(),
                }
            })?;

            families.push(PackedFamily {
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
            return Err(PackedLayoutError::HypercubeTooLarge { dimension });
        }
        let digest = layout_digest(&families, cells, dimension);

        Ok(Self {
            families,
            cells,
            dimension,
            digest,
        })
    }

    pub fn family(&self, id: &PackedFamilyId) -> Option<&PackedFamily> {
        self.families.iter().find(|family| family.id == *id)
    }

    pub fn rank(&self, address: &PackedCellAddress) -> Result<usize, PackedLayoutError> {
        let family =
            self.family(&address.family)
                .ok_or_else(|| PackedLayoutError::MissingFamily {
                    id: address.family.clone(),
                })?;
        let rows = family.domain.rows()?;
        let alphabet_size = family.alphabet.size();
        if address.row >= rows || address.limb >= family.limbs || address.symbol >= alphabet_size {
            return Err(PackedLayoutError::AddressOutOfRange {
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
            .ok_or_else(|| PackedLayoutError::CellCountOverflow {
                id: address.family.clone(),
            })?;
        family
            .offset
            .checked_add(local)
            .ok_or_else(|| PackedLayoutError::CellCountOverflow {
                id: address.family.clone(),
            })
    }

    pub fn unrank(&self, rank: usize) -> Option<PackedCellAddress> {
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
            Some(PackedCellAddress {
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

    pub fn audit(&self) -> PackedLayoutAudit {
        let mut alphabet_counts = PackedAlphabetCounts::default();
        let mut cells_by_domain = PackedDomainCellCounts::default();
        let mut trace_cells_by_log_t = BTreeMap::<usize, usize>::new();
        let mut rectangular_lane_equivalent = 0usize;

        for family in &self.families {
            match family.alphabet {
                PackedAlphabet::Bit => alphabet_counts.bit += 1,
                PackedAlphabet::Byte => alphabet_counts.byte += 1,
                PackedAlphabet::Fixed { .. } => alphabet_counts.fixed += 1,
            }

            match family.domain {
                PackedFactDomain::TraceRows { log_t } => {
                    cells_by_domain.trace_rows =
                        cells_by_domain.trace_rows.saturating_add(family.cell_count);
                    let entry = trace_cells_by_log_t.entry(log_t).or_default();
                    *entry = entry.saturating_add(family.cell_count);
                }
                PackedFactDomain::BytecodeRows { .. } => {
                    cells_by_domain.bytecode_rows = cells_by_domain
                        .bytecode_rows
                        .saturating_add(family.cell_count);
                }
                PackedFactDomain::ProgramImageWords { .. } => {
                    cells_by_domain.program_image_words = cells_by_domain
                        .program_image_words
                        .saturating_add(family.cell_count);
                }
                PackedFactDomain::AdviceBytes { .. } => {
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

        PackedLayoutAudit {
            fact_count_by_alphabet: alphabet_counts,
            cells_by_domain,
            trace_cells_per_row,
            rectangular_lane_equivalent,
            d_pack: self.dimension,
        }
    }

    pub fn validate_view_families(
        &self,
        families: &[PackedFamilyId],
    ) -> Result<(), PackedLayoutError> {
        for id in families {
            if self.family(id).is_none() {
                return Err(PackedLayoutError::MissingViewFamily { id: id.clone() });
            }
        }
        Ok(())
    }
}

impl AkitaSetupParams {
    pub fn from_packed_layout(
        layout: &PackedWitnessLayout,
        max_num_polys_per_commitment_group: usize,
    ) -> Self {
        Self::new(
            layout.dimension,
            max_num_polys_per_commitment_group,
            layout.digest,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedFamily {
    pub id: PackedFamilyId,
    pub domain: PackedFactDomain,
    pub limbs: usize,
    pub alphabet: PackedAlphabet,
    pub offset: usize,
    pub cell_count: usize,
    pub view_kind: PackedViewKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedFamilySpec {
    pub id: PackedFamilyId,
    pub domain: PackedFactDomain,
    pub limbs: usize,
    pub alphabet: PackedAlphabet,
    pub view_kind: PackedViewKind,
}

impl PackedFamilySpec {
    pub fn new(
        id: PackedFamilyId,
        domain: PackedFactDomain,
        limbs: usize,
        alphabet: PackedAlphabet,
        view_kind: PackedViewKind,
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
        id: PackedFamilyId,
        domain: PackedFactDomain,
        limbs: usize,
        alphabet: PackedAlphabet,
    ) -> Self {
        Self::new(id, domain, limbs, alphabet, PackedViewKind::Direct)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PackedFamilyId {
    InstructionRa {
        index: usize,
    },
    BytecodeRa {
        index: usize,
    },
    RamRa {
        index: usize,
    },
    IncByte {
        index: usize,
    },
    IncSign,
    RamIncByte {
        index: usize,
    },
    RamIncSign,
    RdIncByte {
        index: usize,
    },
    RdIncSign,
    FieldRdIncByte {
        index: usize,
    },
    FieldRdIncSign,
    AdviceBytes {
        kind: PackedAdviceKind,
        index: usize,
    },
    BytecodeChunk {
        index: usize,
    },
    ProgramImageInit,
    Custom {
        namespace: u32,
        index: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PackedAdviceKind {
    Trusted,
    Untrusted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackedFactDomain {
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
        kind: PackedAdviceKind,
        log_bytes: usize,
    },
}

impl PackedFactDomain {
    fn rows(&self) -> Result<usize, PackedLayoutError> {
        let log_rows = match *self {
            Self::TraceRows { log_t } => log_t,
            Self::BytecodeRows { log_bytecode } => log_bytecode,
            Self::ProgramImageWords { log_words } => log_words,
            Self::AdviceBytes { log_bytes, .. } => log_bytes,
        };
        if log_rows >= usize::BITS as usize {
            return Err(PackedLayoutError::DomainTooLarge { log_rows });
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
pub enum PackedAlphabet {
    Bit,
    Byte,
    Fixed { size: usize },
}

impl PackedAlphabet {
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
pub enum PackedViewKind {
    Direct,
    LinearDecode,
    MaskedReduction,
}

impl PackedViewKind {
    fn digest_tag(self) -> u8 {
        match self {
            Self::Direct => 0,
            Self::LinearDecode => 1,
            Self::MaskedReduction => 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedCellAddress {
    pub family: PackedFamilyId,
    pub row: usize,
    pub limb: usize,
    pub symbol: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedAlphabetCounts {
    pub bit: usize,
    pub byte: usize,
    pub fixed: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedDomainCellCounts {
    pub trace_rows: usize,
    pub bytecode_rows: usize,
    pub program_image_words: usize,
    pub advice_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedLayoutAudit {
    pub fact_count_by_alphabet: PackedAlphabetCounts,
    pub cells_by_domain: PackedDomainCellCounts,
    pub trace_cells_per_row: Option<usize>,
    pub rectangular_lane_equivalent: usize,
    pub d_pack: usize,
}

pub trait PackedWitnessSource<F: Field> {
    fn layout(&self) -> &PackedWitnessLayout;
    fn for_each_nonzero(&self, f: impl FnMut(usize, F));
    fn eval_direct_fact(&self, address: &PackedCellAddress) -> Result<F, PackedLayoutError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparsePackedWitness<F> {
    layout: PackedWitnessLayout,
    entries: Vec<(usize, F)>,
}

impl<F: Field> SparsePackedWitness<F> {
    pub fn try_new(
        layout: PackedWitnessLayout,
        mut entries: Vec<(usize, F)>,
    ) -> Result<Self, PackedLayoutError> {
        entries.sort_by_key(|(rank, _)| *rank);
        let mut previous_rank = None;
        for (rank, value) in &entries {
            if *rank >= layout.cells {
                return Err(PackedLayoutError::RankOutOfRange {
                    rank: *rank,
                    cells: layout.cells,
                });
            }
            if previous_rank == Some(*rank) {
                return Err(PackedLayoutError::DuplicateSourceRank { rank: *rank });
            }
            if value.is_zero() {
                return Err(PackedLayoutError::ZeroSourceEntry { rank: *rank });
            }
            previous_rank = Some(*rank);
        }
        Ok(Self { layout, entries })
    }

    pub fn try_from_cells(
        layout: PackedWitnessLayout,
        cells: impl IntoIterator<Item = (PackedCellAddress, F)>,
    ) -> Result<Self, PackedLayoutError> {
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

impl<F: Field> PackedWitnessSource<F> for SparsePackedWitness<F> {
    fn layout(&self) -> &PackedWitnessLayout {
        &self.layout
    }

    fn for_each_nonzero(&self, mut f: impl FnMut(usize, F)) {
        for &(rank, value) in &self.entries {
            f(rank, value);
        }
    }

    fn eval_direct_fact(&self, address: &PackedCellAddress) -> Result<F, PackedLayoutError> {
        let rank = self.layout.rank(address)?;
        match self.entries.binary_search_by_key(&rank, |(rank, _)| *rank) {
            Ok(index) => Ok(self.entries[index].1),
            Err(_) => Ok(F::zero()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackedLayoutError {
    EmptyLayout,
    DuplicateFamily {
        id: PackedFamilyId,
    },
    ZeroLimbs {
        id: PackedFamilyId,
    },
    ZeroAlphabet {
        id: PackedFamilyId,
    },
    DomainTooLarge {
        log_rows: usize,
    },
    CellCountOverflow {
        id: PackedFamilyId,
    },
    HypercubeTooLarge {
        dimension: usize,
    },
    MissingFamily {
        id: PackedFamilyId,
    },
    AddressOutOfRange {
        family: PackedFamilyId,
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
        id: PackedFamilyId,
    },
}

impl Display for PackedLayoutError {
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

impl Error for PackedLayoutError {}

fn ceil_log2(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

fn layout_digest(families: &[PackedFamily], cells: usize, dimension: usize) -> AkitaLayoutDigest {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"jolt-akita/packed-witness-layout/v1");
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

fn write_family_id(bytes: &mut Vec<u8>, id: &PackedFamilyId) {
    match id {
        PackedFamilyId::InstructionRa { index } => {
            bytes.push(0);
            write_usize(bytes, *index);
        }
        PackedFamilyId::BytecodeRa { index } => {
            bytes.push(1);
            write_usize(bytes, *index);
        }
        PackedFamilyId::RamRa { index } => {
            bytes.push(2);
            write_usize(bytes, *index);
        }
        PackedFamilyId::IncByte { index } => {
            bytes.push(3);
            write_usize(bytes, *index);
        }
        PackedFamilyId::IncSign => bytes.push(4),
        PackedFamilyId::RamIncByte { index } => {
            bytes.push(5);
            write_usize(bytes, *index);
        }
        PackedFamilyId::RamIncSign => bytes.push(6),
        PackedFamilyId::RdIncByte { index } => {
            bytes.push(7);
            write_usize(bytes, *index);
        }
        PackedFamilyId::RdIncSign => bytes.push(8),
        PackedFamilyId::FieldRdIncByte { index } => {
            bytes.push(9);
            write_usize(bytes, *index);
        }
        PackedFamilyId::FieldRdIncSign => bytes.push(10),
        PackedFamilyId::AdviceBytes { kind, index } => {
            bytes.push(11);
            bytes.push(advice_kind_tag(*kind));
            write_usize(bytes, *index);
        }
        PackedFamilyId::BytecodeChunk { index } => {
            bytes.push(12);
            write_usize(bytes, *index);
        }
        PackedFamilyId::ProgramImageInit => bytes.push(13),
        PackedFamilyId::Custom { namespace, index } => {
            bytes.push(14);
            bytes.extend_from_slice(&namespace.to_le_bytes());
            write_usize(bytes, *index);
        }
    }
}

fn write_domain(bytes: &mut Vec<u8>, domain: PackedFactDomain) {
    bytes.push(domain.digest_tag());
    match domain {
        PackedFactDomain::TraceRows { log_t } => write_usize(bytes, log_t),
        PackedFactDomain::BytecodeRows { log_bytecode } => write_usize(bytes, log_bytecode),
        PackedFactDomain::ProgramImageWords { log_words } => write_usize(bytes, log_words),
        PackedFactDomain::AdviceBytes { kind, log_bytes } => {
            bytes.push(advice_kind_tag(kind));
            write_usize(bytes, log_bytes);
        }
    }
}

fn write_alphabet(bytes: &mut Vec<u8>, alphabet: PackedAlphabet) {
    bytes.push(alphabet.digest_tag());
    match alphabet {
        PackedAlphabet::Bit | PackedAlphabet::Byte => {}
        PackedAlphabet::Fixed { size } => write_usize(bytes, size),
    }
}

fn write_usize(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

fn advice_kind_tag(kind: PackedAdviceKind) -> u8 {
    match kind {
        PackedAdviceKind::Trusted => 0,
        PackedAdviceKind::Untrusted => 1,
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "tests assert successful layout construction"
    )]

    use super::*;
    use crate::AkitaField;

    fn trace(log_t: usize) -> PackedFactDomain {
        PackedFactDomain::TraceRows { log_t }
    }

    fn byte_family(id: PackedFamilyId, log_t: usize) -> PackedFamilySpec {
        PackedFamilySpec::direct(id, trace(log_t), 1, PackedAlphabet::Byte)
    }

    fn bit_family(id: PackedFamilyId, log_t: usize) -> PackedFamilySpec {
        PackedFamilySpec::direct(id, trace(log_t), 1, PackedAlphabet::Bit)
    }

    fn base_ra_specs(log_t: usize) -> Vec<PackedFamilySpec> {
        let mut specs = Vec::new();
        specs.extend(
            (0..16).map(|index| byte_family(PackedFamilyId::InstructionRa { index }, log_t)),
        );
        specs.extend((0..3).map(|index| byte_family(PackedFamilyId::BytecodeRa { index }, log_t)));
        specs.extend((0..4).map(|index| byte_family(PackedFamilyId::RamRa { index }, log_t)));
        specs
    }

    fn fused_increment_specs(log_t: usize) -> Vec<PackedFamilySpec> {
        let mut specs = (0..8)
            .map(|index| byte_family(PackedFamilyId::IncByte { index }, log_t))
            .collect::<Vec<_>>();
        specs.push(bit_family(PackedFamilyId::IncSign, log_t));
        specs
    }

    fn split_increment_specs(log_t: usize) -> Vec<PackedFamilySpec> {
        let mut specs = Vec::new();
        specs.extend((0..8).map(|index| byte_family(PackedFamilyId::RamIncByte { index }, log_t)));
        specs.push(bit_family(PackedFamilyId::RamIncSign, log_t));
        specs.extend((0..8).map(|index| byte_family(PackedFamilyId::RdIncByte { index }, log_t)));
        specs.push(bit_family(PackedFamilyId::RdIncSign, log_t));
        specs
    }

    #[test]
    fn packed_witness_layout_digest_stable() {
        let mut specs = vec![
            byte_family(PackedFamilyId::RamRa { index: 0 }, 4),
            bit_family(PackedFamilyId::IncSign, 4),
            byte_family(PackedFamilyId::InstructionRa { index: 0 }, 4),
        ];
        let layout_a = PackedWitnessLayout::new(specs.clone()).expect("layout should build");
        specs.reverse();
        let layout_b = PackedWitnessLayout::new(specs).expect("layout should build");

        assert_eq!(layout_a.digest, layout_b.digest);
        assert_eq!(layout_a.families, layout_b.families);
    }

    #[test]
    fn packed_witness_layout_rejects_duplicate_ranges() {
        let specs = vec![
            byte_family(PackedFamilyId::RamRa { index: 0 }, 3),
            byte_family(PackedFamilyId::RamRa { index: 0 }, 3),
        ];
        assert!(matches!(
            PackedWitnessLayout::new(specs),
            Err(PackedLayoutError::DuplicateFamily { .. })
        ));
    }

    #[test]
    fn large_trace_base_cells_are_5888_per_row() {
        let layout = PackedWitnessLayout::new(base_ra_specs(20)).expect("layout should build");
        let audit = layout.audit();

        assert_eq!(audit.trace_cells_per_row, Some(5_888));
        assert_eq!(layout.dimension, 33);
    }

    #[test]
    fn separate_increment_budget_is_n_plus_14() {
        let log_t = 20;
        let mut specs = base_ra_specs(log_t);
        specs.extend(split_increment_specs(log_t));

        let layout = PackedWitnessLayout::new(specs).expect("layout should build");
        let audit = layout.audit();

        assert_eq!(audit.trace_cells_per_row, Some(9_988));
        assert_eq!(layout.dimension, log_t + 14);
    }

    #[test]
    fn fused_increment_budget_is_n_plus_13() {
        let log_t = 20;
        let mut specs = base_ra_specs(log_t);
        specs.extend(fused_increment_specs(log_t));

        let layout = PackedWitnessLayout::new(specs).expect("layout should build");
        let audit = layout.audit();

        assert_eq!(audit.trace_cells_per_row, Some(7_938));
        assert_eq!(layout.dimension, log_t + 13);
    }

    #[test]
    fn bit_fact_costs_two_cells_per_row() {
        let layout = PackedWitnessLayout::new([bit_family(PackedFamilyId::IncSign, 5)])
            .expect("layout should build");
        let audit = layout.audit();

        assert_eq!(layout.cells, 64);
        assert_eq!(audit.trace_cells_per_row, Some(2));
        assert_eq!(audit.fact_count_by_alphabet.bit, 1);
    }

    #[test]
    fn rank_unrank_roundtrip() {
        let layout = PackedWitnessLayout::new([
            byte_family(PackedFamilyId::RamRa { index: 0 }, 1),
            bit_family(PackedFamilyId::IncSign, 1),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeChunk { index: 0 },
                PackedFactDomain::BytecodeRows { log_bytecode: 1 },
                2,
                PackedAlphabet::Fixed { size: 3 },
            ),
        ])
        .expect("layout should build");

        for rank in 0..layout.cells {
            let address = layout.unrank(rank).expect("non-dummy rank should unrank");
            assert_eq!(layout.rank(&address).expect("address should rank"), rank);
        }
    }

    #[test]
    fn dummy_cells_are_zero_and_unreferenced() {
        let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
            PackedFamilyId::Custom {
                namespace: 1,
                index: 0,
            },
            trace(0),
            1,
            PackedAlphabet::Fixed { size: 3 },
        )])
        .expect("layout should build");

        assert_eq!(layout.cells, 3);
        assert_eq!(layout.dimension, 2);
        assert_eq!(layout.dummy_cell_count(), 1);
        assert!(layout.unrank(layout.cells).is_none());

        let source = SparsePackedWitness::<AkitaField>::try_new(layout.clone(), Vec::new())
            .expect("empty source should build");
        let zero_address = PackedCellAddress {
            family: PackedFamilyId::Custom {
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
            AkitaField::zero()
        );
    }

    #[test]
    fn layout_sort_order_is_stable() {
        let layout = PackedWitnessLayout::new([
            byte_family(PackedFamilyId::RamRa { index: 3 }, 2),
            byte_family(PackedFamilyId::InstructionRa { index: 1 }, 2),
            byte_family(PackedFamilyId::BytecodeRa { index: 2 }, 2),
        ])
        .expect("layout should build");

        assert_eq!(
            layout
                .families
                .iter()
                .map(|family| &family.id)
                .collect::<Vec<_>>(),
            vec![
                &PackedFamilyId::InstructionRa { index: 1 },
                &PackedFamilyId::BytecodeRa { index: 2 },
                &PackedFamilyId::RamRa { index: 3 },
            ]
        );
    }

    #[test]
    fn committed_program_families_use_non_trace_domains() {
        let layout = PackedWitnessLayout::new([
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeChunk { index: 0 },
                PackedFactDomain::BytecodeRows { log_bytecode: 4 },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::ProgramImageInit,
                PackedFactDomain::ProgramImageWords { log_words: 3 },
                8,
                PackedAlphabet::Byte,
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
        let layout = PackedWitnessLayout::new([
            byte_family(PackedFamilyId::InstructionRa { index: 0 }, 2),
            bit_family(PackedFamilyId::IncSign, 2),
            PackedFamilySpec::direct(
                PackedFamilyId::AdviceBytes {
                    kind: PackedAdviceKind::Trusted,
                    index: 0,
                },
                PackedFactDomain::AdviceBytes {
                    kind: PackedAdviceKind::Trusted,
                    log_bytes: 1,
                },
                1,
                PackedAlphabet::Fixed { size: 4 },
            ),
        ])
        .expect("layout should build");
        let audit = layout.audit();

        assert_eq!(
            audit.fact_count_by_alphabet,
            PackedAlphabetCounts {
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
        let layout = PackedWitnessLayout::new([
            byte_family(PackedFamilyId::RamRa { index: 0 }, 1),
            bit_family(PackedFamilyId::IncSign, 1),
        ])
        .expect("layout should build");
        let one_address = PackedCellAddress {
            family: PackedFamilyId::RamRa { index: 0 },
            row: 1,
            limb: 0,
            symbol: 17,
        };
        let sign_address = PackedCellAddress {
            family: PackedFamilyId::IncSign,
            row: 0,
            limb: 0,
            symbol: 1,
        };
        let source = SparsePackedWitness::try_from_cells(
            layout.clone(),
            [
                (one_address.clone(), AkitaField::from_u64(11)),
                (sign_address.clone(), AkitaField::one()),
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
            AkitaField::from_u64(11)
        );
        assert_eq!(
            source
                .eval_direct_fact(&sign_address)
                .expect("address is in range"),
            AkitaField::one()
        );
    }

    #[test]
    fn view_catalog_references_existing_families() {
        let layout = PackedWitnessLayout::new([byte_family(PackedFamilyId::RamRa { index: 0 }, 2)])
            .expect("layout should build");

        layout
            .validate_view_families(&[PackedFamilyId::RamRa { index: 0 }])
            .expect("existing family should validate");
        assert!(matches!(
            layout.validate_view_families(&[PackedFamilyId::IncSign]),
            Err(PackedLayoutError::MissingViewFamily { .. })
        ));
    }

    #[test]
    fn setup_params_report_packed_dimension_and_digest() {
        let layout = PackedWitnessLayout::new([byte_family(PackedFamilyId::RamRa { index: 0 }, 2)])
            .expect("layout should build");
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);

        assert_eq!(params.max_num_vars, layout.dimension);
        assert_eq!(params.default_layout_digest, layout.digest);
    }

    #[test]
    fn sparse_source_rejects_out_of_layout_ranks() {
        let layout = PackedWitnessLayout::new([bit_family(PackedFamilyId::IncSign, 0)])
            .expect("layout should build");

        assert!(matches!(
            SparsePackedWitness::try_new(layout.clone(), vec![(layout.cells, AkitaField::one())]),
            Err(PackedLayoutError::RankOutOfRange { .. })
        ));
    }
}
