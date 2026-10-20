//! Where each pad lives in the private vector `s` (`specs/whiteout-sigma.md`
//! §1.6). The pad commitment builds its digits by this placement and the
//! lift reads them by it, so both use this one owner.
//!
//! - Regular pads, in pad-index order without the operand pads, are grouped
//!   64 to a challenge-ring element; each group is written as 32 balanced
//!   radix-16 digit planes, plane `j` at ring `32·group + j`, so pad `p`
//!   decodes as `Σ_j 16^j · plane_j[p mod 64]`. The mask g takes the group
//!   after the last regular one.
//! - Operand pad `o` (in `operand_pads` order) is 16 constant ring elements
//!   at rings `16·o + j`, its balanced radix-256 digits:
//!   `P_x = Σ_j 256^j p_{x,j}` and the pad is `ct(P_x)`.

use std::collections::BTreeMap;

use jolt_crypto::lattice::CHALLENGE_RING_DIM;

use super::WhiteoutError;
use crate::masking::{PadIndex, ProductRow, ResidualRelation};

/// Radix-16 digit planes of a regular pad group or of g.
pub const PAD_LEVELS: usize = 32;
pub const PAD_LOG_BASIS: u32 = 4;
/// Radix-256 digit polynomials of an operand pad.
pub const OPERAND_LEVELS: usize = 16;
pub const OPERAND_LOG_BASIS: u32 = 8;

/// A pad's place in `s`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadPlace {
    /// Group and coefficient position of a regular pad.
    Regular { group: usize, position: usize },
    /// Index in `operand_pads`.
    Operand(usize),
}

/// The placement of one pad layout's pads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PadPlacement {
    places: Vec<PadPlace>,
    operand_pads: Vec<PadIndex>,
    regular_groups: usize,
}

impl PadPlacement {
    /// The placement of `num_pads` pads, of which `operand_pads` (distinct,
    /// each below `num_pads`) are committed only as operands.
    pub fn new(num_pads: usize, operand_pads: Vec<PadIndex>) -> Result<Self, WhiteoutError> {
        let mut operand_of = BTreeMap::new();
        for (index, pad) in operand_pads.iter().enumerate() {
            if pad.0 >= num_pads || operand_of.insert(pad.0, index).is_some() {
                return Err(WhiteoutError::InvalidRelation(
                    "operand pads must be distinct pads of the layout",
                ));
            }
        }
        let mut regular = 0;
        let places = (0..num_pads)
            .map(|pad| {
                if let Some(&operand) = operand_of.get(&pad) {
                    return PadPlace::Operand(operand);
                }
                let place = PadPlace::Regular {
                    group: regular / CHALLENGE_RING_DIM,
                    position: regular % CHALLENGE_RING_DIM,
                };
                regular += 1;
                place
            })
            .collect();
        Ok(Self {
            places,
            operand_pads,
            regular_groups: regular.div_ceil(CHALLENGE_RING_DIM),
        })
    }

    /// The operand pads of `relation`, in first-use order: every product's
    /// masked operand. The masking layer derives the same list from its
    /// shape routine before the pads are committed (`OperandShapeDomain`).
    pub fn operands_of<F>(relation: &ResidualRelation<F>) -> Vec<PadIndex> {
        let mut seen = Vec::new();
        for ProductRow { x, .. } in &relation.products {
            if !seen.contains(&x.pad) {
                seen.push(x.pad);
            }
        }
        seen
    }

    /// Release check that every product's operand is a committed operand pad.
    pub fn check_products<F>(&self, relation: &ResidualRelation<F>) -> Result<(), WhiteoutError> {
        if relation.layout.num_pads != self.places.len() {
            return Err(WhiteoutError::InvalidRelation(
                "relation and placement have different pad counts",
            ));
        }
        if relation
            .products
            .iter()
            .all(|product| self.operand_pads.contains(&product.x.pad))
        {
            Ok(())
        } else {
            Err(WhiteoutError::InvalidRelation(
                "a product's operand is not an operand pad",
            ))
        }
    }

    pub fn place(&self, pad: PadIndex) -> Result<PadPlace, WhiteoutError> {
        self.places
            .get(pad.0)
            .copied()
            .ok_or(WhiteoutError::InvalidRelation("pad outside the layout"))
    }

    pub fn operand_pads(&self) -> &[PadIndex] {
        &self.operand_pads
    }

    pub fn num_pads(&self) -> usize {
        self.places.len()
    }

    /// Regular pad groups; g is group `regular_groups()`.
    pub fn regular_groups(&self) -> usize {
        self.regular_groups
    }

    /// Challenge-ring elements the regular pads and g take.
    pub fn pad_rings(&self) -> usize {
        (self.regular_groups + 1) * PAD_LEVELS
    }

    /// Challenge-ring elements the operand digit polynomials take.
    pub fn operand_rings(&self) -> usize {
        self.operand_pads.len() * OPERAND_LEVELS
    }
}
