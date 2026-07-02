use std::error::Error;
use std::fmt::{self, Display, Formatter};

use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
        specs.sort_by_key(|spec| spec.id);

        let mut families = Vec::with_capacity(specs.len());
        let mut offset = 0usize;
        let mut previous_id = None;
        for spec in specs {
            if previous_id.as_ref() == Some(&spec.id) {
                return Err(PackingLayoutError::DuplicateFamily { id: spec.id });
            }
            previous_id = Some(spec.id);

            if spec.limbs == 0 {
                return Err(PackingLayoutError::ZeroLimbs { id: spec.id });
            }
            let rows = spec.domain.rows()?;
            let alphabet_size = spec.alphabet.size();
            if alphabet_size == 0 {
                return Err(PackingLayoutError::ZeroAlphabet { id: spec.id });
            }
            let cell_count = rows
                .checked_mul(spec.limbs)
                .and_then(|value| value.checked_mul(alphabet_size))
                .ok_or(PackingLayoutError::CellCountOverflow { id: spec.id })?;
            let next_offset = offset
                .checked_add(cell_count)
                .ok_or(PackingLayoutError::CellCountOverflow { id: spec.id })?;

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

    pub fn dummy_cell_count(&self) -> usize {
        (1usize << self.dimension) - self.cells
    }

    pub fn validate_view_families(
        &self,
        families: &[PackingFamilyId],
    ) -> Result<(), PackingLayoutError> {
        for id in families {
            if self.family(id).is_none() {
                return Err(PackingLayoutError::MissingViewFamily { id: *id });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackingFamilyId {
    pub namespace: u64,
    pub id: u64,
    pub index: u64,
}

impl PackingFamilyId {
    pub const fn new(namespace: u64, id: u64, index: u64) -> Self {
        Self {
            namespace,
            id,
            index,
        }
    }
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
pub enum PackingValidityKind {
    ExactOneHot,
    OptionalOneHot,
    BooleanIndicator { symbol: usize },
    BytecodeStoreRdDisjoint,
    FieldElementCanonicalBytes { byte_width: usize, modulus: u128 },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

    pub fn bytecode_store_rd_disjoint(family: PackingFamilyId) -> Self {
        Self {
            family,
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
    bytes.extend_from_slice(PACKING_VALIDITY_DIGEST_DOMAIN);
    write_usize(&mut bytes, encoded_requirements.len());
    for requirement in encoded_requirements {
        bytes.extend_from_slice(&requirement);
    }

    digest_bytes(&bytes)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackingViewFormula<F> {
    Direct {
        family: PackingFamilyId,
        limb: usize,
        symbol: usize,
    },
    LinearDecoded {
        terms: Vec<PackingViewTerm<F>>,
        validity: PackingViewValidity,
    },
    ReducedMasked {
        terms: Vec<PackingViewTerm<F>>,
    },
    MaskedDecoded,
}

impl<F> PackingViewFormula<F> {
    pub fn direct(family: PackingFamilyId, limb: usize, symbol: usize) -> Self {
        Self::Direct {
            family,
            limb,
            symbol,
        }
    }

    pub fn linear_decoded(terms: Vec<PackingViewTerm<F>>) -> Self {
        Self::LinearDecoded {
            terms,
            validity: PackingViewValidity::Proven,
        }
    }

    pub fn unchecked_linear_decoded(terms: Vec<PackingViewTerm<F>>) -> Self {
        Self::LinearDecoded {
            terms,
            validity: PackingViewValidity::Unchecked,
        }
    }

    pub fn reduced_masked(terms: Vec<PackingViewTerm<F>>) -> Self {
        Self::ReducedMasked { terms }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackingViewValidity {
    Proven,
    Unchecked,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackingViewTerm<F> {
    pub coefficient: F,
    pub family: PackingFamilyId,
    pub limb: usize,
    pub symbol: usize,
}

impl<F> PackingViewTerm<F> {
    pub fn new(coefficient: F, family: PackingFamilyId, limb: usize, symbol: usize) -> Self {
        Self {
            coefficient,
            family,
            limb,
            symbol,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackingLayoutError {
    EmptyLayout,
    DuplicateFamily { id: PackingFamilyId },
    ZeroLimbs { id: PackingFamilyId },
    ZeroAlphabet { id: PackingFamilyId },
    DomainTooLarge { log_rows: usize },
    CellCountOverflow { id: PackingFamilyId },
    HypercubeTooLarge { dimension: usize },
    MissingViewFamily { id: PackingFamilyId },
}

impl Display for PackingLayoutError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLayout => f.write_str("packed witness layout must contain a family"),
            Self::DuplicateFamily { id } => write!(f, "duplicate packed witness family {id:?}"),
            Self::ZeroLimbs { id } => {
                write!(f, "packed witness family {id:?} has zero limbs")
            }
            Self::ZeroAlphabet { id } => {
                write!(f, "packed witness family {id:?} has zero alphabet size")
            }
            Self::DomainTooLarge { log_rows } => {
                write!(
                    f,
                    "packed witness domain 2^{log_rows} does not fit in usize"
                )
            }
            Self::CellCountOverflow { id } => {
                write!(f, "packed witness family {id:?} cell count overflows usize")
            }
            Self::HypercubeTooLarge { dimension } => {
                write!(
                    f,
                    "packed witness hypercube dimension {dimension} is too large"
                )
            }
            Self::MissingViewFamily { id } => {
                write!(f, "packed witness view references missing family {id:?}")
            }
        }
    }
}

impl Error for PackingLayoutError {}

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

fn ceil_log2(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

fn layout_digest(families: &[PackingLayoutFamily], cells: usize, dimension: usize) -> [u8; 32] {
    let mut bytes = Vec::new();
    // This is protocol data; keep the domain stable across crate ownership moves.
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

    digest_bytes(&bytes)
}

fn write_family_id(bytes: &mut Vec<u8>, id: &PackingFamilyId) {
    bytes.extend_from_slice(&id.namespace.to_le_bytes());
    bytes.extend_from_slice(&id.id.to_le_bytes());
    bytes.extend_from_slice(&id.index.to_le_bytes());
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

fn digest_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Blake2b::<U32>::new();
    hasher.update(bytes);
    let result = hasher.finalize();
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&result);
    digest
}
