//! Prefix packing for multiple logical multilinear polynomials in one PCS
//! commitment.
//!
//! A packing assigns each logical polynomial id `i` a prefix-free Boolean
//! address `prefix(i)` so that `P_i(x) = W(prefix(i) || x)` on the Boolean
//! hypercube. The packed polynomial has
//! `ceil_log2(sum_i 2^num_vars_i)` variables; unassigned cells in that rounded
//! domain are zero-filled.
//!
//! Slot assignment is deterministic: logical polynomial declarations are sorted
//! by descending arity, then by `Id: Ord`, and placed by advancing a
//! power-of-two aligned cursor through the packed Boolean domain. If a slot has
//! `num_vars = n`, its evaluations are copied to indices
//! `(prefix_value << n) | local_index`, where prefix bits are interpreted
//! high-to-low.

use std::{
    collections::{btree_map::Iter, BTreeMap, BTreeSet},
    fmt::Debug,
    ops::Index,
};

use jolt_field::Field;
use jolt_poly::{boolean_bits_msb, eq_index_msb, MultilinearPoly};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use serde::{Deserialize, Serialize};

use crate::{CommitmentScheme, EvaluationClaim, OpeningsError};

/// Logical polynomial declaration used to build a deterministic packing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedPolynomial<Id> {
    pub id: Id,
    pub num_vars: usize,
}

impl<Id> PackedPolynomial<Id> {
    pub fn new(id: Id, num_vars: usize) -> Self {
        Self { id, num_vars }
    }
}

impl<Id> From<(Id, usize)> for PackedPolynomial<Id> {
    fn from((id, num_vars): (Id, usize)) -> Self {
        Self { id, num_vars }
    }
}

/// Physical slot assigned to one logical polynomial.
///
/// The prefix is stored as high-to-low Boolean bits. For a slot with `n`
/// logical variables, local Boolean index `j` is placed at packed index
/// `(prefix_value << n) | j`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixSlot {
    pub prefix: Vec<bool>,
    pub num_vars: usize,
}

impl PrefixSlot {
    fn new(prefix_value: usize, prefix_len: usize, num_vars: usize) -> Self {
        Self {
            prefix: boolean_bits_msb(prefix_len, prefix_value),
            num_vars,
        }
    }

    /// The slot's prefix bits as an integer (msb-first).
    pub(crate) fn prefix_index(&self) -> usize {
        self.prefix
            .iter()
            .fold(0usize, |acc, bit| (acc << 1) | usize::from(*bit))
    }

    /// Packed evaluation index of this slot's cell `local` — where witness
    /// assembly places the cell in the packed table.
    pub fn packed_index(&self, local: usize) -> usize {
        debug_assert!(local < 1usize << self.num_vars);
        (self.prefix_index() << self.num_vars) | local
    }
}

/// Deterministic prefix-free assignment from logical ids to packed slots.
///
/// A physical opening point is `r_pack = r_prefix || x`. For a slot with prefix
/// length `k`, `x` is the suffix `r_pack[k..]`. Selector evaluations use the
/// equality polynomial `eq(r_pack[..k], prefix)`, so one opening of `W` at a
/// full packed point checks a linear combination of logical claims at their
/// suffix-compatible logical points.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "Id: Serialize",
    deserialize = "Id: Ord + Deserialize<'de>"
))]
#[serde(deny_unknown_fields)]
pub struct PrefixPacking<Id> {
    pub packed_num_vars: usize,
    pub(crate) slots: BTreeMap<Id, PrefixSlot>,
}

impl<Id> PrefixPacking<Id> {
    pub fn iter(&self) -> Iter<'_, Id, PrefixSlot> {
        self.slots.iter()
    }
}

impl<Id> PrefixPacking<Id>
where
    Id: Clone + Ord,
{
    /// Builds the canonical prefix assignment for the given logical polynomials.
    pub fn new<P>(polynomials: impl IntoIterator<Item = P>) -> Result<Self, OpeningsError>
    where
        P: Into<PackedPolynomial<Id>>,
    {
        let mut polynomials = polynomials.into_iter().map(Into::into).collect::<Vec<_>>();
        if polynomials.is_empty() {
            return Err(OpeningsError::InvalidSetup(
                "prefix packing requires at least one polynomial".to_owned(),
            ));
        }

        let mut total_cells = 0usize;
        for polynomial in &polynomials {
            let cells = DomainSize::try_from(polynomial.num_vars)?.cells();
            total_cells = total_cells.checked_add(cells).ok_or_else(|| {
                OpeningsError::InvalidSetup("prefix packing domain size overflow".to_owned())
            })?;
        }
        let packed_num_vars = usize::BITS as usize - (total_cells - 1).leading_zeros() as usize;
        let packed_domain_size = DomainSize::try_from(packed_num_vars)?.cells();

        polynomials.sort_by(|left, right| {
            right
                .num_vars
                .cmp(&left.num_vars)
                .then_with(|| left.id.cmp(&right.id))
        });

        let mut cursor = 0usize;
        let mut slots = BTreeMap::new();
        for polynomial in polynomials {
            let cells = DomainSize::try_from(polynomial.num_vars)?.cells();
            if !cursor.is_multiple_of(cells) {
                return Err(OpeningsError::InvalidSetup(
                    "prefix packing cursor is not aligned to polynomial domain".to_owned(),
                ));
            }
            let prefix_len = packed_num_vars
                .checked_sub(polynomial.num_vars)
                .ok_or_else(|| {
                    OpeningsError::InvalidSetup(
                        "logical polynomial has too many variables".to_owned(),
                    )
                })?;
            let prefix_value = cursor >> polynomial.num_vars;
            let slot = PrefixSlot::new(prefix_value, prefix_len, polynomial.num_vars);
            if slots.insert(polynomial.id, slot).is_some() {
                return Err(OpeningsError::InvalidSetup(
                    "duplicate packed polynomial id".to_owned(),
                ));
            }
            cursor += cells;
        }

        debug_assert!(cursor <= packed_domain_size);
        Ok(Self {
            packed_num_vars,
            slots,
        })
    }

    /// Returns the number of prefix challenges needed before a logical point
    /// with `logical_num_vars` variables.
    pub fn prefix_challenge_len(&self, logical_num_vars: usize) -> Result<usize, OpeningsError> {
        self.packed_num_vars
            .checked_sub(logical_num_vars)
            .ok_or_else(|| {
                OpeningsError::InvalidBatch(format!(
                    "logical point has {logical_num_vars} variables but packed witness has {}",
                    self.packed_num_vars
                ))
            })
    }

    /// Forms the physical point `prefix_point || logical_point`.
    pub fn pack_point<F: Field>(
        &self,
        prefix_point: &[F],
        logical_point: &[F],
    ) -> Result<Vec<F>, OpeningsError> {
        let prefix_len = self.prefix_challenge_len(logical_point.len())?;
        if prefix_point.len() != prefix_len {
            return Err(OpeningsError::InvalidBatch(format!(
                "prefix point has {} variables but logical arity requires {prefix_len}",
                prefix_point.len()
            )));
        }
        let mut point = Vec::with_capacity(self.packed_num_vars);
        point.extend_from_slice(prefix_point);
        point.extend_from_slice(logical_point);
        Ok(point)
    }

    /// Extracts the logical suffix point for `id` from a full packed point.
    pub fn logical_point<F: Field>(
        &self,
        id: &Id,
        packed_point: &[F],
    ) -> Result<Vec<F>, OpeningsError> {
        let slot = self.slots.get(id).ok_or_else(|| {
            OpeningsError::InvalidBatch("unknown packed polynomial id".to_owned())
        })?;
        if packed_point.len() != self.packed_num_vars {
            return Err(OpeningsError::InvalidBatch(format!(
                "packed point has {} variables but packing has {}",
                packed_point.len(),
                self.packed_num_vars
            )));
        }
        Ok(packed_point[slot.prefix.len()..].to_vec())
    }

    pub(crate) fn prepare_statement<'a, F, C>(
        &'a self,
        statement: &'a PrefixPackedStatement<F, Id, C>,
    ) -> Result<PreparedPrefixPackedStatement<'a, F, Id, C>, OpeningsError>
    where
        F: Field,
        Id: Debug,
    {
        let claims = statement.claims.as_slice();
        if claims.is_empty() {
            return Err(OpeningsError::InvalidBatch(
                "packing opening requires at least one claim".to_owned(),
            ));
        }

        let mut seen = BTreeSet::new();
        let mut constrained_point = vec![None; self.packed_num_vars];
        let mut ordered_claims = Vec::with_capacity(claims.len());

        for claim in claims {
            let slot = self.slots.get(&claim.id).ok_or_else(|| {
                OpeningsError::InvalidBatch(format!("unknown packed polynomial id: {:?}", claim.id))
            })?;
            if claim.evaluation.point.len() != slot.num_vars {
                return Err(OpeningsError::InvalidBatch(format!(
                    "claim for packed polynomial id {:?} has point arity {} but slot has {} variables",
                    claim.id,
                    claim.evaluation.point.len(),
                    slot.num_vars
                )));
            }
            if !seen.insert(claim.id.clone()) {
                return Err(OpeningsError::InvalidBatch(format!(
                    "duplicate claim for packed polynomial id {:?}",
                    claim.id
                )));
            }

            let suffix_start = slot.prefix.len();
            for (offset, point_coord) in claim.evaluation.point.iter().copied().enumerate() {
                let packed_index = suffix_start + offset;
                if let Some(existing) = constrained_point[packed_index] {
                    if existing != point_coord {
                        return Err(OpeningsError::InvalidBatch(format!(
                            "claim for packed polynomial id {:?} is not suffix-compatible at packed coordinate {packed_index}",
                            claim.id
                        )));
                    }
                } else {
                    constrained_point[packed_index] = Some(point_coord);
                }
            }

            ordered_claims.push((claim, slot));
        }

        if seen.len() != self.slots.len() {
            let missing = self
                .slots
                .keys()
                .find(|id| !seen.contains(*id))
                .ok_or_else(|| {
                    OpeningsError::InvalidBatch("packed batch has duplicate claim ids".to_owned())
                })?;
            return Err(OpeningsError::InvalidBatch(format!(
                "missing claim for packed polynomial id {missing:?}"
            )));
        }

        ordered_claims.sort_by(|(left, _), (right, _)| left.id.cmp(&right.id));

        Ok(PreparedPrefixPackedStatement {
            packed_num_vars: self.packed_num_vars,
            commitment: &statement.commitment,
            constrained_point,
            ordered_claims,
        })
    }
}

impl<'a, Id> IntoIterator for &'a PrefixPacking<Id> {
    type Item = (&'a Id, &'a PrefixSlot);
    type IntoIter = Iter<'a, Id, PrefixSlot>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<Id> Index<&Id> for PrefixPacking<Id>
where
    Id: Ord,
{
    type Output = PrefixSlot;

    fn index(&self, id: &Id) -> &Self::Output {
        &self.slots[id]
    }
}

/// Borrowed prover-side source for opening a prefix-packed witness.
///
/// The PCS sees any [`MultilinearPoly`] source for `W`, plus the opening hint
/// produced while committing to that same source. Dense reference polynomials,
/// sparse representations, and lazy packed sources can all use the same
/// opening path without materializing the full packed evaluation table.
pub struct PackedWitness<'a, F: Field, H> {
    pub polynomial: &'a (dyn MultilinearPoly<F> + 'a),
    pub hint: H,
}

impl<'a, F, H> PackedWitness<'a, F, H>
where
    F: Field,
{
    pub fn new(polynomial: &'a (dyn MultilinearPoly<F> + 'a), hint: H) -> Self {
        Self { polynomial, hint }
    }
}

/// One logical opening claim inside a prefix-packed commitment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixPackedClaim<F, Id> {
    pub id: Id,
    pub evaluation: EvaluationClaim<F>,
}

impl<F, Id> PrefixPackedClaim<F, Id> {
    pub fn new(id: Id, evaluation: EvaluationClaim<F>) -> Self {
        Self { id, evaluation }
    }
}

/// A batch statement for opening one packed witness commitment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixPackedStatement<F, Id, C> {
    pub commitment: C,
    pub claims: Vec<PrefixPackedClaim<F, Id>>,
}

impl<F, Id, C> PrefixPackedStatement<F, Id, C> {
    pub fn new(commitment: C, claims: impl Into<Vec<PrefixPackedClaim<F, Id>>>) -> Self {
        Self {
            commitment,
            claims: claims.into(),
        }
    }
}

pub(crate) struct PreparedPrefixPackedStatement<'a, F: Field, Id, C> {
    packed_num_vars: usize,
    pub(crate) commitment: &'a C,
    constrained_point: Vec<Option<F>>,
    ordered_claims: Vec<(&'a PrefixPackedClaim<F, Id>, &'a PrefixSlot)>,
}

impl<F, Id, C> PreparedPrefixPackedStatement<'_, F, Id, C>
where
    F: Field,
{
    pub(crate) fn opening_point<T>(&self, transcript: &mut T) -> Result<Vec<F>, OpeningsError>
    where
        T: Transcript<Challenge = F>,
    {
        let missing_count = self
            .constrained_point
            .iter()
            .filter(|coord| coord.is_none())
            .count();
        transcript.append(&LabelWithCount(
            b"prefix_pack_missing",
            missing_count as u64,
        ));
        let challenges = transcript.challenge_vector(missing_count);
        let mut challenges = challenges.into_iter();
        let mut point = Vec::with_capacity(self.constrained_point.len());
        for coord in &self.constrained_point {
            if let Some(value) = coord {
                point.push(*value);
            } else {
                let challenge = challenges.next().ok_or_else(|| {
                    OpeningsError::InvalidBatch(
                        "packed point completion ran out of challenges".to_owned(),
                    )
                })?;
                point.push(challenge);
            }
        }
        Ok(point)
    }

    /// Computes the compact prefix-packing reduction:
    /// `W(r_pack) = sum_i eq(prefix_i, r_pack[..|prefix_i|]) * P_i(r_pack[|prefix_i|..])`.
    ///
    /// The statement is accepted only after suffix compatibility has forced all
    /// logical claims to agree on overlapping packed coordinates.
    pub(crate) fn reduced_eval(&self, packed_point: &[F]) -> F {
        self.ordered_claims
            .iter()
            .fold(F::zero(), |acc, (claim, slot)| {
                acc + eq_index_msb(
                    &packed_point[..slot.prefix.len()],
                    slot.prefix_index() as u128,
                ) * claim.evaluation.value
            })
    }
}

impl<F, Id, C> AppendToTranscript for PreparedPrefixPackedStatement<'_, F, Id, C>
where
    F: Field,
    C: AppendToTranscript,
{
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&Label(b"prefix_pack_batch"));
        transcript.append(&U64Word(self.packed_num_vars as u64));
        self.commitment.append_to_transcript(transcript);
        transcript.append(&LabelWithCount(
            b"prefix_pack_claims",
            self.ordered_claims.len() as u64,
        ));
        for (claim, slot) in &self.ordered_claims {
            transcript.append(&U64Word(slot.num_vars as u64));
            transcript.append(&LabelWithCount(
                b"prefix_pack_prefix",
                slot.prefix.len() as u64,
            ));
            let prefix_bytes = slot
                .prefix
                .iter()
                .map(|bit| u8::from(*bit))
                .collect::<Vec<_>>();
            transcript.append_bytes(&prefix_bytes);
            transcript.append(&LabelWithCount(
                b"prefix_pack_point",
                claim.evaluation.point.len() as u64,
            ));
            for value in claim.evaluation.point.as_slice() {
                value.append_to_transcript(transcript);
            }
            claim.evaluation.value.append_to_transcript(transcript);
        }
    }
}

/// Prover setup for opening a prefix-packed witness with a concrete PCS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixPackedProverSetup<PCS: CommitmentScheme, Id = u64> {
    pub pcs: PCS::ProverSetup,
    pub packing: PrefixPacking<Id>,
}

/// Verifier setup for opening a prefix-packed witness with a concrete PCS.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "PCS::VerifierSetup: Serialize, Id: Serialize",
    deserialize = "PCS::VerifierSetup: Deserialize<'de>, Id: Ord + Deserialize<'de>"
))]
#[serde(deny_unknown_fields)]
pub struct PrefixPackedVerifierSetup<PCS: CommitmentScheme, Id = u64> {
    pub pcs: PCS::VerifierSetup,
    pub packing: PrefixPacking<Id>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DomainSize(usize);

impl DomainSize {
    fn cells(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for DomainSize {
    type Error = OpeningsError;

    fn try_from(num_vars: usize) -> Result<Self, Self::Error> {
        if num_vars >= usize::BITS as usize {
            return Err(OpeningsError::InvalidSetup(format!(
                "polynomial with {num_vars} variables exceeds addressable domain"
            )));
        }
        1usize
            .checked_shl(num_vars as u32)
            .map(Self)
            .ok_or_else(|| {
                OpeningsError::InvalidSetup("polynomial domain size overflow".to_owned())
            })
    }
}
