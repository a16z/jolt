//! Fixed-width prefix packing for logical multilinear polynomials.
//!
//! A layout places `m` logical polynomials with `n` variables into the first
//! `m` slots of a power-of-two capacity `c`. The physical polynomial has
//! `n + log2(c)` variables. At a point `(s, x)` its value is
//!
//! `sum_i eq(s, i) * P_i(x)`.
//!
//! Every logical claim must use the same suffix point `x`. The claims and
//! their semantic layout digest are absorbed before `s` is sampled. Unused
//! slots are outside the logical statement; this API does not constrain their
//! coefficients directly. The reduced opening requires their aggregate
//! contribution at `s` to vanish. Arbitrary-point claim reduction is not part
//! of this API.

use std::collections::BTreeMap;

use jolt_field::JoltField;
use jolt_poly::{eq_index_msb, Point, HIGH_TO_LOW};
use jolt_transcript::{Label, Transcript, U64Word};

use crate::{EvaluationClaim, OpeningsError};

/// Ordered fixed-capacity placement of equal-arity logical polynomials.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixPackedLayout<Id> {
    logical_num_vars: usize,
    selector_num_vars: usize,
    slot_capacity: usize,
    ids: Vec<Id>,
    slot_indices: BTreeMap<Id, usize>,
}

impl<Id> PrefixPackedLayout<Id>
where
    Id: Clone + Ord,
{
    /// Creates a layout using the supplied identifier order as protocol data.
    pub fn new(
        logical_num_vars: usize,
        slot_capacity: usize,
        ids: impl IntoIterator<Item = Id>,
    ) -> Result<Self, OpeningsError> {
        if slot_capacity == 0 || !slot_capacity.is_power_of_two() {
            return Err(OpeningsError::InvalidSetup(
                "prefix-packed slot capacity must be a nonzero power of two".to_owned(),
            ));
        }
        let selector_num_vars = slot_capacity.ilog2() as usize;
        let packed_num_vars = logical_num_vars
            .checked_add(selector_num_vars)
            .ok_or_else(|| {
                OpeningsError::InvalidSetup("prefix-packed variable count exceeds usize".to_owned())
            })?;
        let _packed_coefficients = coefficient_count(packed_num_vars)?;

        let ids = ids.into_iter().collect::<Vec<_>>();
        if ids.is_empty() {
            return Err(OpeningsError::InvalidSetup(
                "prefix-packed layout requires at least one logical polynomial".to_owned(),
            ));
        }
        if ids.len() > slot_capacity {
            return Err(OpeningsError::InvalidSetup(format!(
                "prefix-packed layout has {} polynomials but capacity is {slot_capacity}",
                ids.len()
            )));
        }

        let mut slot_indices = BTreeMap::new();
        for (slot, id) in ids.iter().cloned().enumerate() {
            if slot_indices.insert(id, slot).is_some() {
                return Err(OpeningsError::InvalidSetup(
                    "prefix-packed layout contains a duplicate polynomial id".to_owned(),
                ));
            }
        }

        Ok(Self {
            logical_num_vars,
            selector_num_vars,
            slot_capacity,
            ids,
            slot_indices,
        })
    }

    /// Number of variables in each logical polynomial.
    pub const fn logical_num_vars(&self) -> usize {
        self.logical_num_vars
    }

    /// Number of variables selecting a prefix slot.
    pub const fn selector_num_vars(&self) -> usize {
        self.selector_num_vars
    }

    /// Number of variables in the physical packed polynomial.
    pub const fn packed_num_vars(&self) -> usize {
        self.logical_num_vars + self.selector_num_vars
    }

    /// Number of physical prefix slots, including unused slots.
    pub const fn slot_capacity(&self) -> usize {
        self.slot_capacity
    }

    /// Logical identifiers in their physical slot order.
    pub fn ids(&self) -> &[Id] {
        &self.ids
    }

    /// Physical slot assigned to `id`.
    pub fn slot_index(&self, id: &Id) -> Option<usize> {
        self.slot_indices.get(id).copied()
    }

    /// Physical Boolean index of one logical coefficient.
    pub fn packed_index(&self, id: &Id, logical_index: usize) -> Result<usize, OpeningsError> {
        let slot = self.slot_index(id).ok_or_else(|| {
            OpeningsError::InvalidBatch("unknown prefix-packed polynomial id".to_owned())
        })?;
        let logical_len = coefficient_count(self.logical_num_vars)?;
        if logical_index >= logical_len {
            return Err(OpeningsError::InvalidBatch(format!(
                "logical coefficient index {logical_index} exceeds polynomial length {logical_len}"
            )));
        }
        Ok((slot << self.logical_num_vars) | logical_index)
    }

    /// Forms the physical point `(selector || logical)`.
    pub fn pack_point<F: JoltField>(
        &self,
        selector_point: &[F],
        logical_point: &[F],
    ) -> Result<Vec<F>, OpeningsError> {
        if selector_point.len() != self.selector_num_vars {
            return Err(OpeningsError::InvalidBatch(format!(
                "prefix selector has {} variables, expected {}",
                selector_point.len(),
                self.selector_num_vars
            )));
        }
        if logical_point.len() != self.logical_num_vars {
            return Err(OpeningsError::InvalidBatch(format!(
                "logical point has {} variables, expected {}",
                logical_point.len(),
                self.logical_num_vars
            )));
        }
        let mut point = Vec::with_capacity(self.packed_num_vars());
        point.extend_from_slice(selector_point);
        point.extend_from_slice(logical_point);
        Ok(point)
    }

    /// Reduces ordered logical evaluations at one common point.
    pub fn reduce_evaluations<F: JoltField>(
        &self,
        selector_point: &[F],
        evaluations: &[F],
    ) -> Result<F, OpeningsError> {
        if selector_point.len() != self.selector_num_vars {
            return Err(OpeningsError::InvalidBatch(format!(
                "prefix selector has {} variables, expected {}",
                selector_point.len(),
                self.selector_num_vars
            )));
        }
        if evaluations.len() != self.ids.len() {
            return Err(OpeningsError::InvalidBatch(format!(
                "prefix-packed statement has {} evaluations, expected {}",
                evaluations.len(),
                self.ids.len()
            )));
        }
        Ok(evaluations
            .iter()
            .enumerate()
            .fold(F::zero(), |sum, (slot, evaluation)| {
                sum + *evaluation * eq_index_msb(selector_point, slot as u128)
            }))
    }

    /// Binds the semantic statement, samples the selector, and returns the
    /// corresponding claim on the physical polynomial.
    pub fn reduce_claims<F, T>(
        &self,
        claims: &PrefixPackedClaims<F>,
        transcript: &mut T,
    ) -> Result<EvaluationClaim<F>, OpeningsError>
    where
        F: JoltField,
        T: Transcript<Challenge = F>,
    {
        if claims.point.len() != self.logical_num_vars {
            return Err(OpeningsError::InvalidBatch(format!(
                "logical point has {} variables, expected {}",
                claims.point.len(),
                self.logical_num_vars
            )));
        }
        if claims.evaluations.len() != self.ids.len() {
            return Err(OpeningsError::InvalidBatch(format!(
                "prefix-packed statement has {} evaluations, expected {}",
                claims.evaluations.len(),
                self.ids.len()
            )));
        }

        transcript.append(&Label(b"prefix_packed_claim"));
        transcript.append(&U64Word(self.logical_num_vars as u64));
        transcript.append(&U64Word(self.slot_capacity as u64));
        transcript.append(&Label(b"prefix_pack_layout"));
        transcript.append_bytes(&claims.layout_digest);
        transcript.append_values(b"prefix_pack_point", claims.point.as_slice());
        transcript.append_values(b"prefix_pack_evals", &claims.evaluations);

        let selector = transcript.challenge_vector(self.selector_num_vars);
        let point = self.pack_point(&selector, claims.point.as_slice())?;
        let evaluation = self.reduce_evaluations(&selector, &claims.evaluations)?;
        Ok(EvaluationClaim::new(point, evaluation))
    }
}

/// Semantic claims reduced to one opening of a prefix-packed polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixPackedClaims<F> {
    layout_digest: [u8; 32],
    point: Point<HIGH_TO_LOW, F>,
    evaluations: Vec<F>,
}

impl<F> PrefixPackedClaims<F> {
    /// Creates the statement in canonical logical-slot order.
    pub fn new(
        layout_digest: [u8; 32],
        point: impl Into<Point<HIGH_TO_LOW, F>>,
        evaluations: Vec<F>,
    ) -> Self {
        Self {
            layout_digest,
            point: point.into(),
            evaluations,
        }
    }

    /// Protocol digest binding the logical identifiers and layout version.
    pub const fn layout_digest(&self) -> &[u8; 32] {
        &self.layout_digest
    }

    /// Common logical opening point.
    pub fn point(&self) -> &[F] {
        self.point.as_slice()
    }

    /// Evaluations in physical slot order, excluding unused slots.
    pub fn evaluations(&self) -> &[F] {
        &self.evaluations
    }
}

fn coefficient_count(num_vars: usize) -> Result<usize, OpeningsError> {
    1usize.checked_shl(num_vars as u32).ok_or_else(|| {
        OpeningsError::InvalidSetup(format!(
            "polynomial with {num_vars} variables exceeds the addressable domain"
        ))
    })
}
