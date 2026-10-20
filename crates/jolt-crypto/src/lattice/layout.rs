//! Akita's two-level commitment, described as one part of Whiteout's
//! ring-linear map N.
//!
//! This module computes nothing: Akita computes the residual commitment.
//! The layout fixes the column semantics N uses on it:
//!
//! - inner `t_i = A · G⁻¹(w_i)` per block;
//! - each `t_i` decomposed into outer digits, a d_A ring split into d_A/d_B
//!   subcolumns of consecutive coefficients, ordered
//!   `[A row][subcolumn][digit][coefficient]`;
//! - slice inputs polynomial-major, each polynomial's segment zero-padded to
//!   `max_blocks_per_slice · per_block`;
//! - slice `s`'s hiding randomness in the last `randomness_columns` columns of
//!   its input (the layout W1 implements; byte-exact with Akita today only
//!   for zero randomness columns);
//! - `u` the concatenation of the slice outputs.

use super::ajtai::AjtaiLevel;
use crate::CryptoError;

/// Ring dimension of Whiteout's challenge ring; N must be a module over it.
pub const CHALLENGE_RING_DIM: usize = 64;

/// How the committed source is cut into blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockGeometry {
    /// Live ring elements per polynomial.
    pub live_rings: usize,
    /// Ring elements per block (`M`).
    pub rings_per_block: usize,
    /// Live blocks per polynomial.
    pub live_blocks: usize,
}

/// Number of outer slices; blocks split dyadically,
/// slice `s` covering blocks `⌊B s/S⌋..⌊B (s+1)/S⌋`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceCount {
    One,
    Two,
    Four,
    Eight,
}

impl SliceCount {
    pub fn get(self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}

/// Akita's two-level commitment layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoLevelLayout {
    pub blocks: BlockGeometry,
    pub num_polys: usize,
    /// `A`, applied to every block; it carries no randomness.
    pub inner: AjtaiLevel,
    /// `[B | B′]`, applied once per slice.
    pub outer: AjtaiLevel,
    pub slices: SliceCount,
}

impl TwoLevelLayout {
    /// The largest number of blocks any slice holds.
    pub fn max_blocks_per_slice(&self) -> usize {
        self.blocks.live_blocks.div_ceil(self.slices.get())
    }

    /// Checks the internal consistency of the layout: inner input is one
    /// block, the inner level is binding only, and the outer message is
    /// exactly the decomposed inner images of one slice.
    pub fn validate(&self) -> Result<(), CryptoError> {
        let inner_dim = self.inner.matrix.ring_dim;
        let outer_dim = self.outer.matrix.ring_dim;
        if self.inner.randomness_columns != 0 {
            return Err(CryptoError::InvalidSetup(
                "the inner level carries no randomness".into(),
            ));
        }
        if self.inner.message_rings != self.blocks.rings_per_block {
            return Err(CryptoError::InvalidSetup(format!(
                "inner level takes {} rings, blocks hold {}",
                self.inner.message_rings, self.blocks.rings_per_block
            )));
        }
        if outer_dim == 0 || !inner_dim.is_multiple_of(outer_dim) {
            return Err(CryptoError::InvalidSetup(format!(
                "inner ring dimension {inner_dim} is not a multiple of outer {outer_dim}"
            )));
        }
        let outer_message = self
            .inner
            .matrix
            .rows
            .checked_mul(inner_dim / outer_dim)
            .and_then(|rings| rings.checked_mul(self.max_blocks_per_slice()))
            .and_then(|rings| rings.checked_mul(self.num_polys))
            .ok_or_else(|| CryptoError::InvalidSetup("outer message size overflows".into()))?;
        if outer_message != self.outer.message_rings {
            return Err(CryptoError::InvalidSetup(format!(
                "outer level takes {} rings, one slice of inner images is {outer_message}",
                self.outer.message_rings
            )));
        }
        Ok(())
    }

    /// Whether N on this commitment is linear over Whiteout's challenge ring:
    /// restriction of scalars needs every multiplication to act on
    /// stride-embedded inputs, and Akita's consecutive-coefficient split from
    /// d_A to d_B is not a module map, so both dimensions must be equal
    /// multiples of 64.
    pub fn is_module_over_challenge_ring(&self) -> bool {
        let inner_dim = self.inner.matrix.ring_dim;
        inner_dim == self.outer.matrix.ring_dim && inner_dim.is_multiple_of(CHALLENGE_RING_DIM)
    }
}
