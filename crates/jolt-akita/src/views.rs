use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Display, Formatter};

use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use jolt_field::Field;
use jolt_openings::{PackedLinearTerm, PhysicalView};
use jolt_poly::EqPolynomial;

use crate::layout::{
    PackedCellAddress, PackedFactDomain, PackedFamilyId, PackedLayoutError, PackedWitnessLayout,
    PackedWitnessSource,
};
use crate::types::AkitaLayoutDigest;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedViewCatalog<OpeningId, RelationId, F> {
    pub entries: Vec<PackedViewEntry<OpeningId, RelationId, F>>,
    pub digest: AkitaLayoutDigest,
}

impl<OpeningId, RelationId, F> PackedViewCatalog<OpeningId, RelationId, F>
where
    OpeningId: Clone + Ord,
    RelationId: Clone + Ord,
    F: Field,
{
    pub fn new(
        layout: &PackedWitnessLayout,
        entries: impl IntoIterator<Item = PackedViewEntry<OpeningId, RelationId, F>>,
    ) -> Result<Self, PackedViewError> {
        let mut entries = entries.into_iter().collect::<Vec<_>>();
        entries.sort_by(|left, right| {
            left.id
                .cmp(&right.id)
                .then_with(|| left.relation.cmp(&right.relation))
        });
        if entries.is_empty() {
            return Err(PackedViewError::EmptyCatalog);
        }

        let mut previous_key = None;
        for entry in &entries {
            let key = (&entry.id, &entry.relation);
            if previous_key == Some(key) {
                return Err(PackedViewError::DuplicateView);
            }
            entry.formula.validate(layout)?;
            previous_key = Some(key);
        }

        let digest = catalog_digest(layout.digest, &entries);
        Ok(Self { entries, digest })
    }

    pub fn lookup(
        &self,
        id: &OpeningId,
        relation: &RelationId,
    ) -> Result<&PackedViewFormula<F>, PackedViewError> {
        self.entries
            .iter()
            .find(|entry| entry.id == *id && entry.relation == *relation)
            .map(|entry| &entry.formula)
            .ok_or(PackedViewError::MissingView)
    }

    pub fn verify_digest(&self, expected: &AkitaLayoutDigest) -> Result<(), PackedViewError> {
        if &self.digest == expected {
            Ok(())
        } else {
            Err(PackedViewError::CatalogDigestMismatch {
                expected: *expected,
                actual: self.digest,
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedViewEntry<OpeningId, RelationId, F> {
    pub id: OpeningId,
    pub relation: RelationId,
    pub formula: PackedViewFormula<F>,
}

impl<OpeningId, RelationId, F> PackedViewEntry<OpeningId, RelationId, F> {
    pub fn new(id: OpeningId, relation: RelationId, formula: PackedViewFormula<F>) -> Self {
        Self {
            id,
            relation,
            formula,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackedViewFormula<F> {
    Direct {
        family: PackedFamilyId,
        limb: usize,
        symbol: usize,
    },
    LinearDecoded {
        terms: Vec<PackedViewTerm<F>>,
        validity: PackedViewValidity,
    },
    ReducedMasked {
        terms: Vec<PackedViewTerm<F>>,
    },
    MaskedDecoded,
}

impl<F: Field> PackedViewFormula<F> {
    pub fn direct(family: PackedFamilyId, limb: usize, symbol: usize) -> Self {
        Self::Direct {
            family,
            limb,
            symbol,
        }
    }

    pub fn linear_decoded(terms: Vec<PackedViewTerm<F>>) -> Self {
        Self::LinearDecoded {
            terms,
            validity: PackedViewValidity::Proven,
        }
    }

    pub fn unchecked_linear_decoded(terms: Vec<PackedViewTerm<F>>) -> Self {
        Self::LinearDecoded {
            terms,
            validity: PackedViewValidity::Unchecked,
        }
    }

    pub fn reduced_masked(terms: Vec<PackedViewTerm<F>>) -> Self {
        Self::ReducedMasked { terms }
    }

    pub fn validate(&self, layout: &PackedWitnessLayout) -> Result<(), PackedViewError> {
        match self {
            Self::Direct {
                family,
                limb,
                symbol,
            } => validate_term_shape(layout, family, *limb, *symbol).map(|_| ()),
            Self::LinearDecoded { terms, validity } => {
                if *validity != PackedViewValidity::Proven {
                    return Err(PackedViewError::DecodedViewNeedsValidity);
                }
                validate_linear_terms(layout, terms)
            }
            Self::ReducedMasked { terms } => validate_linear_terms(layout, terms),
            Self::MaskedDecoded => Err(PackedViewError::MaskedViewRequiresTranslation),
        }
    }

    pub fn physical_view(
        &self,
        layout: &PackedWitnessLayout,
    ) -> Result<PhysicalView<F>, PackedViewError> {
        self.physical_view_at(layout, &[])
    }

    pub fn physical_view_at(
        &self,
        layout: &PackedWitnessLayout,
        row_point: &[F],
    ) -> Result<PhysicalView<F>, PackedViewError> {
        self.validate(layout)?;
        let terms = match self {
            Self::Direct {
                family,
                limb,
                symbol,
            } => vec![
                PackedLinearTerm::new(F::one(), family.physical_ref(), *limb, *symbol)
                    .with_row_point(row_point.to_vec()),
            ],
            Self::LinearDecoded { terms, .. } | Self::ReducedMasked { terms } => terms
                .iter()
                .map(|term| {
                    PackedLinearTerm::new(
                        term.coefficient,
                        term.family.physical_ref(),
                        term.limb,
                        term.symbol,
                    )
                    .with_row_point(row_point.to_vec())
                })
                .collect::<Vec<_>>(),
            Self::MaskedDecoded => return Err(PackedViewError::MaskedViewRequiresTranslation),
        };
        Ok(PhysicalView::PackedLinear {
            layout_digest: layout.digest,
            terms,
        })
    }

    pub fn eval_row<S>(&self, source: &S, row: usize) -> Result<F, PackedViewError>
    where
        S: PackedWitnessSource<F>,
    {
        let layout = source.layout();
        self.validate(layout)?;
        let mut result = F::zero();
        match self {
            Self::Direct {
                family,
                limb,
                symbol,
            } => {
                let address = PackedCellAddress {
                    family: family.clone(),
                    row,
                    limb: *limb,
                    symbol: *symbol,
                };
                result = source.eval_direct_fact(&address)?;
            }
            Self::LinearDecoded { terms, .. } | Self::ReducedMasked { terms } => {
                for term in terms {
                    let address = PackedCellAddress {
                        family: term.family.clone(),
                        row,
                        limb: term.limb,
                        symbol: term.symbol,
                    };
                    result += term.coefficient * source.eval_direct_fact(&address)?;
                }
            }
            Self::MaskedDecoded => return Err(PackedViewError::MaskedViewRequiresTranslation),
        }
        Ok(result)
    }

    pub fn eval_row_point<S>(&self, source: &S, row_point: &[F]) -> Result<F, PackedViewError>
    where
        S: PackedWitnessSource<F>,
    {
        let layout = source.layout();
        self.validate(layout)?;
        let domain = self.row_domain(layout)?;
        let expected = log_rows(domain)?;
        if row_point.len() != expected {
            return Err(PackedViewError::InvalidRowPointDimension {
                expected,
                actual: row_point.len(),
            });
        }

        let row_weights = EqPolynomial::<F>::evals(row_point, None);
        let coefficients = self.term_coefficients();
        let mut result = F::zero();
        let mut error = None;
        source.for_each_nonzero(|rank, value| {
            if error.is_some() {
                return;
            }
            let Some(address) = layout.unrank(rank) else {
                error = Some(PackedViewError::Layout(PackedLayoutError::RankOutOfRange {
                    rank,
                    cells: layout.cells,
                }));
                return;
            };
            let row = address.row;
            let key = (address.family, address.limb, address.symbol);
            if let Some(coefficient) = coefficients.get(&key) {
                result += *coefficient * row_weights[row] * value;
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
        Ok(result)
    }

    fn row_domain(
        &self,
        layout: &PackedWitnessLayout,
    ) -> Result<PackedFactDomain, PackedViewError> {
        match self {
            Self::Direct { family, .. } => layout
                .family(family)
                .map(|family| family.domain)
                .ok_or_else(|| {
                    PackedViewError::Layout(PackedLayoutError::MissingFamily { id: family.clone() })
                }),
            Self::LinearDecoded { terms, .. } | Self::ReducedMasked { terms } => {
                let Some(term) = terms.first() else {
                    return Err(PackedViewError::EmptyLinearView);
                };
                layout
                    .family(&term.family)
                    .map(|family| family.domain)
                    .ok_or_else(|| {
                        PackedViewError::Layout(PackedLayoutError::MissingFamily {
                            id: term.family.clone(),
                        })
                    })
            }
            Self::MaskedDecoded => Err(PackedViewError::MaskedViewRequiresTranslation),
        }
    }

    fn term_coefficients(&self) -> BTreeMap<(PackedFamilyId, usize, usize), F> {
        let mut coefficients = BTreeMap::new();
        match self {
            Self::Direct {
                family,
                limb,
                symbol,
            } => {
                let _ = coefficients.insert((family.clone(), *limb, *symbol), F::one());
            }
            Self::LinearDecoded { terms, .. } | Self::ReducedMasked { terms } => {
                for term in terms {
                    *coefficients
                        .entry((term.family.clone(), term.limb, term.symbol))
                        .or_insert_with(F::zero) += term.coefficient;
                }
            }
            Self::MaskedDecoded => {}
        }
        coefficients
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedViewValidity {
    Proven,
    Unchecked,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedViewTerm<F> {
    pub coefficient: F,
    pub family: PackedFamilyId,
    pub limb: usize,
    pub symbol: usize,
}

impl<F> PackedViewTerm<F> {
    pub fn new(coefficient: F, family: PackedFamilyId, limb: usize, symbol: usize) -> Self {
        Self {
            coefficient,
            family,
            limb,
            symbol,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackedViewError {
    EmptyCatalog,
    DuplicateView,
    MissingView,
    EmptyLinearView,
    MixedDomains,
    DecodedViewNeedsValidity,
    MaskedViewRequiresTranslation,
    InvalidRowPointDimension {
        expected: usize,
        actual: usize,
    },
    CatalogDigestMismatch {
        expected: AkitaLayoutDigest,
        actual: AkitaLayoutDigest,
    },
    Layout(PackedLayoutError),
}

impl Display for PackedViewError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCatalog => f.write_str("packed view catalog must contain a view"),
            Self::DuplicateView => {
                f.write_str("packed view catalog contains a duplicate opening/relation key")
            }
            Self::MissingView => f.write_str("packed view catalog is missing the requested view"),
            Self::EmptyLinearView => {
                f.write_str("packed linear view must contain at least one term")
            }
            Self::MixedDomains => {
                f.write_str("packed linear view terms must live in the same row domain")
            }
            Self::DecodedViewNeedsValidity => {
                f.write_str("decoded packed view requires an explicit validity relation")
            }
            Self::MaskedViewRequiresTranslation => {
                f.write_str("masked packed view requires a prior translation sumcheck")
            }
            Self::InvalidRowPointDimension { expected, actual } => write!(
                f,
                "packed view row point has dimension {actual}, expected {expected}"
            ),
            Self::CatalogDigestMismatch { .. } => {
                f.write_str("packed view catalog digest does not match expected digest")
            }
            Self::Layout(error) => Display::fmt(error, f),
        }
    }
}

impl Error for PackedViewError {}

impl From<PackedLayoutError> for PackedViewError {
    fn from(error: PackedLayoutError) -> Self {
        Self::Layout(error)
    }
}

fn validate_linear_terms<F: Field>(
    layout: &PackedWitnessLayout,
    terms: &[PackedViewTerm<F>],
) -> Result<(), PackedViewError> {
    if terms.is_empty() {
        return Err(PackedViewError::EmptyLinearView);
    }
    let mut domain = None;
    for term in terms {
        let term_domain = validate_term_shape(layout, &term.family, term.limb, term.symbol)?;
        if let Some(domain) = domain {
            if domain != term_domain {
                return Err(PackedViewError::MixedDomains);
            }
        } else {
            domain = Some(term_domain);
        }
    }
    Ok(())
}

fn validate_term_shape(
    layout: &PackedWitnessLayout,
    family_id: &PackedFamilyId,
    limb: usize,
    symbol: usize,
) -> Result<crate::PackedFactDomain, PackedViewError> {
    let family = layout
        .family(family_id)
        .ok_or_else(|| PackedLayoutError::MissingFamily {
            id: family_id.clone(),
        })?;
    if limb >= family.limbs || symbol >= family.alphabet.size() {
        return Err(PackedLayoutError::AddressOutOfRange {
            family: family_id.clone(),
            row: 0,
            limb,
            symbol,
        }
        .into());
    }
    Ok(family.domain)
}

fn log_rows(domain: PackedFactDomain) -> Result<usize, PackedViewError> {
    let rows = domain.rows()?;
    Ok(rows.trailing_zeros() as usize)
}

fn catalog_digest<OpeningId, RelationId, F>(
    layout_digest: AkitaLayoutDigest,
    entries: &[PackedViewEntry<OpeningId, RelationId, F>],
) -> AkitaLayoutDigest
where
    F: Field,
{
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"jolt-akita/packed-view-catalog/v1");
    bytes.extend_from_slice(&layout_digest);
    write_usize(&mut bytes, entries.len());
    for entry in entries {
        write_formula(&mut bytes, &entry.formula);
    }

    let mut hasher = Blake2b::<U32>::new();
    hasher.update(&bytes);
    let result = hasher.finalize();
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&result);
    digest
}

fn write_formula<F: Field>(bytes: &mut Vec<u8>, formula: &PackedViewFormula<F>) {
    match formula {
        PackedViewFormula::Direct {
            family,
            limb,
            symbol,
        } => {
            bytes.push(0);
            write_family_id(bytes, family);
            write_usize(bytes, *limb);
            write_usize(bytes, *symbol);
        }
        PackedViewFormula::LinearDecoded { terms, validity } => {
            bytes.push(1);
            write_validity(bytes, *validity);
            write_terms(bytes, terms);
        }
        PackedViewFormula::ReducedMasked { terms } => {
            bytes.push(2);
            write_terms(bytes, terms);
        }
        PackedViewFormula::MaskedDecoded => bytes.push(3),
    }
}

fn write_terms<F: Field>(bytes: &mut Vec<u8>, terms: &[PackedViewTerm<F>]) {
    write_usize(bytes, terms.len());
    for term in terms {
        write_field(bytes, term.coefficient);
        write_family_id(bytes, &term.family);
        write_usize(bytes, term.limb);
        write_usize(bytes, term.symbol);
    }
}

fn write_validity(bytes: &mut Vec<u8>, validity: PackedViewValidity) {
    bytes.push(match validity {
        PackedViewValidity::Proven => 0,
        PackedViewValidity::Unchecked => 1,
    });
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
        PackedFamilyId::UnsignedIncChunk { index } => {
            bytes.push(3);
            write_usize(bytes, *index);
        }
        PackedFamilyId::UnsignedIncMsb => bytes.push(4),
        PackedFamilyId::FieldRdIncByte { index } => {
            bytes.push(9);
            write_usize(bytes, *index);
        }
        PackedFamilyId::FieldRdIncSign => bytes.push(10),
        PackedFamilyId::AdviceBytes { kind, index } => {
            bytes.push(11);
            bytes.push(match kind {
                crate::PackedAdviceKind::Trusted => 0,
                crate::PackedAdviceKind::Untrusted => 1,
            });
            write_usize(bytes, *index);
        }
        PackedFamilyId::BytecodeChunk { index } => {
            bytes.push(12);
            write_usize(bytes, *index);
        }
        PackedFamilyId::ProgramImageInit => bytes.push(13),
        PackedFamilyId::BytecodeRegisterSelector { chunk, selector } => {
            bytes.push(15);
            write_usize(bytes, *chunk);
            write_usize(bytes, *selector);
        }
        PackedFamilyId::BytecodeCircuitFlag { chunk, flag } => {
            bytes.push(16);
            write_usize(bytes, *chunk);
            write_usize(bytes, *flag);
        }
        PackedFamilyId::BytecodeInstructionFlag { chunk, flag } => {
            bytes.push(17);
            write_usize(bytes, *chunk);
            write_usize(bytes, *flag);
        }
        PackedFamilyId::BytecodeLookupSelector { chunk } => {
            bytes.push(18);
            write_usize(bytes, *chunk);
        }
        PackedFamilyId::BytecodeRafFlag { chunk } => {
            bytes.push(19);
            write_usize(bytes, *chunk);
        }
        PackedFamilyId::BytecodeUnexpandedPcBytes { chunk } => {
            bytes.push(20);
            write_usize(bytes, *chunk);
        }
        PackedFamilyId::BytecodeImmBytes { chunk } => {
            bytes.push(21);
            write_usize(bytes, *chunk);
        }
        PackedFamilyId::Custom { namespace, index } => {
            bytes.push(14);
            bytes.extend_from_slice(&namespace.to_le_bytes());
            write_usize(bytes, *index);
        }
    }
}

fn write_field<F: Field>(bytes: &mut Vec<u8>, value: F) {
    let mut field_bytes = vec![0u8; F::NUM_BYTES];
    value.to_bytes_le(&mut field_bytes);
    bytes.extend_from_slice(&field_bytes);
}

fn write_usize(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "tests assert successful packed-view setup"
    )]

    use super::*;
    use crate::{
        AkitaField, PackedAlphabet, PackedFactDomain, PackedFamilySpec, PackedWitnessLayout,
        SparsePackedWitness,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum OpeningId {
        A,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum RelationId {
        First,
        Second,
    }

    fn f(value: u64) -> AkitaField {
        AkitaField::from_u64(value)
    }

    fn byte_layout() -> PackedWitnessLayout {
        PackedWitnessLayout::new([
            PackedFamilySpec::direct(
                PackedFamilyId::RamRa { index: 0 },
                PackedFactDomain::TraceRows { log_t: 1 },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::UnsignedIncMsb,
                PackedFactDomain::TraceRows { log_t: 1 },
                1,
                PackedAlphabet::Bit,
            ),
        ])
        .expect("layout should build")
    }

    fn byte_decode_terms(family: PackedFamilyId) -> Vec<PackedViewTerm<AkitaField>> {
        (0..256)
            .map(|symbol| PackedViewTerm::new(f(symbol as u64), family.clone(), 0, symbol))
            .collect()
    }

    #[test]
    fn direct_view_translation_matches_packed_eval() {
        let layout = byte_layout();
        let address = PackedCellAddress {
            family: PackedFamilyId::UnsignedIncMsb,
            row: 1,
            limb: 0,
            symbol: 1,
        };
        let source =
            SparsePackedWitness::try_from_cells(layout.clone(), [(address, AkitaField::one())])
                .expect("source should build");
        let formula = PackedViewFormula::<AkitaField>::direct(PackedFamilyId::UnsignedIncMsb, 0, 1);

        assert_eq!(
            formula.eval_row(&source, 1).expect("view should evaluate"),
            AkitaField::one()
        );
        assert!(matches!(
            formula
                .physical_view(&layout)
                .expect("view should lower to the opening API"),
            PhysicalView::PackedLinear { terms, .. }
                if terms.len() == 1
                    && terms[0].coefficient == AkitaField::one()
                    && terms[0].family == PackedFamilyId::UnsignedIncMsb.physical_ref()
                    && terms[0].symbol == 1
        ));
    }

    #[test]
    fn linear_decode_translation_matches_direct_sum() {
        let layout = byte_layout();
        let address = PackedCellAddress {
            family: PackedFamilyId::RamRa { index: 0 },
            row: 0,
            limb: 0,
            symbol: 7,
        };
        let source =
            SparsePackedWitness::try_from_cells(layout.clone(), [(address, AkitaField::one())])
                .expect("source should build");
        let formula = PackedViewFormula::linear_decoded(byte_decode_terms(PackedFamilyId::RamRa {
            index: 0,
        }));

        assert_eq!(
            formula.eval_row(&source, 0).expect("view should evaluate"),
            f(7)
        );
        assert!(matches!(
            formula
                .physical_view(&layout)
                .expect("view should lower to the opening API"),
            PhysicalView::PackedLinear { layout_digest, terms }
                if layout_digest == layout.digest
                    && terms.len() == 256
                    && terms[7].coefficient == f(7)
                    && terms[7].family == (PackedFamilyId::RamRa { index: 0 }).physical_ref()
                    && terms[7].symbol == 7
        ));
    }

    #[test]
    fn direct_view_point_eval_interpolates_rows() {
        let layout = byte_layout();
        let address = PackedCellAddress {
            family: PackedFamilyId::UnsignedIncMsb,
            row: 1,
            limb: 0,
            symbol: 1,
        };
        let source = SparsePackedWitness::try_from_cells(layout, [(address, AkitaField::one())])
            .expect("source should build");
        let formula = PackedViewFormula::<AkitaField>::direct(PackedFamilyId::UnsignedIncMsb, 0, 1);
        let point = [f(3)];

        assert_eq!(
            formula
                .eval_row_point(&source, &point)
                .expect("view should evaluate at point"),
            point[0]
        );
    }

    #[test]
    fn linear_decode_point_eval_interpolates_rows() {
        let layout = byte_layout();
        let source = SparsePackedWitness::try_from_cells(
            layout,
            [
                (
                    PackedCellAddress {
                        family: PackedFamilyId::RamRa { index: 0 },
                        row: 0,
                        limb: 0,
                        symbol: 7,
                    },
                    AkitaField::one(),
                ),
                (
                    PackedCellAddress {
                        family: PackedFamilyId::RamRa { index: 0 },
                        row: 1,
                        limb: 0,
                        symbol: 11,
                    },
                    AkitaField::one(),
                ),
            ],
        )
        .expect("source should build");
        let formula = PackedViewFormula::linear_decoded(byte_decode_terms(PackedFamilyId::RamRa {
            index: 0,
        }));
        let point = [f(5)];
        let expected = (AkitaField::one() - point[0]) * f(7) + point[0] * f(11);

        assert_eq!(
            formula
                .eval_row_point(&source, &point)
                .expect("view should evaluate at point"),
            expected
        );
    }

    #[test]
    fn row_point_dimension_mismatch_rejects() {
        let layout = byte_layout();
        let source = SparsePackedWitness::try_from_cells(
            layout,
            [(
                PackedCellAddress {
                    family: PackedFamilyId::UnsignedIncMsb,
                    row: 1,
                    limb: 0,
                    symbol: 1,
                },
                AkitaField::one(),
            )],
        )
        .expect("source should build");
        let formula = PackedViewFormula::<AkitaField>::direct(PackedFamilyId::UnsignedIncMsb, 0, 1);

        assert!(matches!(
            formula.eval_row_point(&source, &[]),
            Err(PackedViewError::InvalidRowPointDimension {
                expected: 1,
                actual: 0
            })
        ));
    }

    #[test]
    fn masked_view_requires_translation_sumcheck() {
        let layout = byte_layout();
        let formula = PackedViewFormula::<AkitaField>::MaskedDecoded;

        assert!(matches!(
            formula.physical_view(&layout),
            Err(PackedViewError::MaskedViewRequiresTranslation)
        ));
    }

    #[test]
    fn translation_layout_digest_mismatch_rejects() {
        let layout = byte_layout();
        let catalog_a = PackedViewCatalog::new(
            &layout,
            [PackedViewEntry::new(
                OpeningId::A,
                RelationId::First,
                PackedViewFormula::<AkitaField>::direct(PackedFamilyId::UnsignedIncMsb, 0, 1),
            )],
        )
        .expect("catalog should build");
        let catalog_b = PackedViewCatalog::new(
            &layout,
            [PackedViewEntry::new(
                OpeningId::A,
                RelationId::First,
                PackedViewFormula::<AkitaField>::direct(PackedFamilyId::UnsignedIncMsb, 0, 0),
            )],
        )
        .expect("catalog should build");

        assert_ne!(catalog_a.digest, catalog_b.digest);
        assert!(matches!(
            catalog_a.verify_digest(&catalog_b.digest),
            Err(PackedViewError::CatalogDigestMismatch { .. })
        ));
    }

    #[test]
    fn decoded_view_without_validity_rejects_or_is_not_enabled() {
        let layout = byte_layout();
        let formula =
            PackedViewFormula::unchecked_linear_decoded(byte_decode_terms(PackedFamilyId::RamRa {
                index: 0,
            }));

        assert!(matches!(
            formula.physical_view(&layout),
            Err(PackedViewError::DecodedViewNeedsValidity)
        ));
    }

    #[test]
    fn same_polynomial_different_relation_ids_distinct() {
        let layout = byte_layout();
        let catalog = PackedViewCatalog::new(
            &layout,
            [
                PackedViewEntry::new(
                    OpeningId::A,
                    RelationId::Second,
                    PackedViewFormula::<AkitaField>::direct(PackedFamilyId::UnsignedIncMsb, 0, 0),
                ),
                PackedViewEntry::new(
                    OpeningId::A,
                    RelationId::First,
                    PackedViewFormula::<AkitaField>::direct(PackedFamilyId::UnsignedIncMsb, 0, 1),
                ),
            ],
        )
        .expect("catalog should build");

        assert_eq!(
            catalog
                .lookup(&OpeningId::A, &RelationId::First)
                .expect("first relation should exist"),
            &PackedViewFormula::<AkitaField>::direct(PackedFamilyId::UnsignedIncMsb, 0, 1)
        );
        assert_eq!(
            catalog
                .lookup(&OpeningId::A, &RelationId::Second)
                .expect("second relation should exist"),
            &PackedViewFormula::<AkitaField>::direct(PackedFamilyId::UnsignedIncMsb, 0, 0)
        );
    }

    #[test]
    fn bound_precommitted_program_view_formula_validates_against_supplied_layout() {
        let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
            PackedFamilyId::ProgramImageInit,
            PackedFactDomain::ProgramImageWords { log_words: 2 },
            8,
            PackedAlphabet::Byte,
        )])
        .expect("layout should build");
        let formula =
            PackedViewFormula::linear_decoded(byte_decode_terms(PackedFamilyId::ProgramImageInit));

        formula.validate(&layout).expect("formula should validate");
        assert_eq!(
            layout
                .family(&PackedFamilyId::ProgramImageInit)
                .expect("program family should exist")
                .domain,
            PackedFactDomain::ProgramImageWords { log_words: 2 }
        );
    }
}
