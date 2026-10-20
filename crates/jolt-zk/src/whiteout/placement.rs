//! Where each pad lives in the private vector `s` (`specs/whiteout-sigma.md`
//! §1.6). The pad commitment builds its digits by this placement and the lift
//! reads them by it; every ring index into the pad segments comes from here.
//!
//! - Regular pads, in pad-index order without the operand pads, are grouped
//!   64 to a challenge-ring element; each group is written as 32 balanced
//!   radix-16 digit planes, plane `j` at pad ring `32·group + j`, so pad `p`
//!   decodes as `Σ_j 16^j · plane_j[p mod 64]`. The mask g is the group after
//!   the last regular one.
//! - Operand pad `o` (in `operand_pads` order) is 16 constant ring elements
//!   at operand rings `16·o + j`, its balanced radix-256 digits:
//!   `P_x = Σ_j 256^j p_{x,j}`, and the pad is `ct(P_x)`.

use std::collections::BTreeMap;

use jolt_crypto::lattice::CHALLENGE_RING_DIM;

use super::WhiteoutError;
use crate::masking::{PadIndex, PadLayout, ResidualRelation};

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
    layout: PadLayout,
    places: Vec<PadPlace>,
    regular_groups: usize,
}

impl PadPlacement {
    /// The placement of `layout`'s pads; its operand pads (distinct pads of
    /// the layout) are committed only as operands.
    pub fn new(layout: &PadLayout) -> Result<Self, WhiteoutError> {
        let mut operand_of = BTreeMap::new();
        for (index, pad) in layout.operand_pads.iter().enumerate() {
            if pad.0 >= layout.num_pads || operand_of.insert(pad.0, index).is_some() {
                return Err(WhiteoutError::InvalidRelation(
                    "operand pads must be distinct pads of the layout",
                ));
            }
        }
        let mut regular = 0;
        let places = (0..layout.num_pads)
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
            layout: layout.clone(),
            places,
            regular_groups: regular.div_ceil(CHALLENGE_RING_DIM),
        })
    }

    /// Release check that `relation` has this placement's layout and that
    /// every product's operand is an operand pad.
    pub fn check_relation<F>(&self, relation: &ResidualRelation<F>) -> Result<(), WhiteoutError> {
        if relation.layout != self.layout {
            return Err(WhiteoutError::InvalidRelation(
                "relation and placement have different pad layouts",
            ));
        }
        if relation
            .products
            .iter()
            .all(|product| self.layout.operand_pads.contains(&product.x.pad))
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

    #[cfg(feature = "whiteout-prover")]
    pub(crate) fn num_pads(&self) -> usize {
        self.places.len()
    }

    pub(crate) fn num_operands(&self) -> usize {
        self.layout.operand_pads.len()
    }

    /// Regular pad groups; g is the next group.
    #[cfg(feature = "whiteout-prover")]
    pub(crate) fn regular_groups(&self) -> usize {
        self.regular_groups
    }

    /// Pad-segment ring of digit plane `level` of regular group `group`.
    pub(crate) fn regular_ring(group: usize, level: usize) -> usize {
        group * PAD_LEVELS + level
    }

    /// Pad-segment ring of digit plane `level` of the mask g.
    pub(crate) fn g_ring(&self, level: usize) -> usize {
        Self::regular_ring(self.regular_groups, level)
    }

    /// Operand-segment ring of digit polynomial `level` of operand `operand`.
    pub(crate) fn operand_ring(operand: usize, level: usize) -> usize {
        operand * OPERAND_LEVELS + level
    }

    /// Challenge-ring elements the regular pads and g take.
    pub(crate) fn pad_rings(&self) -> usize {
        (self.regular_groups + 1) * PAD_LEVELS
    }

    /// Challenge-ring elements the operand digit polynomials take.
    pub(crate) fn operand_rings(&self) -> usize {
        self.num_operands() * OPERAND_LEVELS
    }
}
