//! The ring-linear rows `N·s = t` of the residual relation: the openings of
//! the residual and pad commitments (Whiteout Definition `wo:def:residual`,
//! item 1).
//!
//! `N` is a list of blocks, each `M·x[input] − G(x[gadget]) = target`: an
//! Ajtai matrix on a range of `s`, minus an optional gadget reconstruction
//! (the inner rows `A w_i − G t̂_i = 0`). Every block works over one native
//! ring dimension `D`, a multiple of 64 shared by both commitments, so `N` is
//! linear over the challenge ring under the stride view.

use std::ops::Range;
use std::sync::Arc;

use jolt_algebra::{CanonicalEncoding, CyclotomicRing, Field};
use jolt_crypto::lattice::{PreparedMatrix, CHALLENGE_RING_DIM};

use super::segments::lift_ct;
use super::WhiteoutError;

/// `G(x) = Σ_j b^j · plane_j` per output row, over `rows × levels` native
/// elements ordered `[row][digit]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GadgetTerm {
    /// Native coefficient range of `s`.
    pub input: Range<usize>,
    pub log_basis: u32,
    pub levels: usize,
}

/// One block `M·x[input] − G(x[gadget]) = target`.
pub struct LinearBlock<F: Field, const D: usize> {
    pub matrix: Arc<PreparedMatrix<F, D>>,
    /// Native coefficient range of `s`, `matrix.width · D` long.
    pub input: Range<usize>,
    pub gadget: Option<GadgetTerm>,
    pub target: Vec<CyclotomicRing<F, D>>,
}

/// `N` and its targets `t`.
pub struct RingLinearMap<F: Field, const D: usize> {
    blocks: Vec<LinearBlock<F, D>>,
}

impl<F: Field + CanonicalEncoding, const D: usize> RingLinearMap<F, D> {
    pub fn new(blocks: Vec<LinearBlock<F, D>>) -> Result<Self, WhiteoutError> {
        for block in &blocks {
            let shape = block.matrix.shape();
            let columns = block.input.len();
            let rows_ok = block.target.len() == shape.rows;
            let width_ok = columns == shape.width * D;
            let gadget_ok = block.gadget.as_ref().is_none_or(|gadget| {
                gadget.input.len() == shape.rows * gadget.levels * D
                    && (1..=8).contains(&gadget.log_basis)
            });
            if !(rows_ok && width_ok && gadget_ok) {
                return Err(WhiteoutError::InvalidRelation(
                    "linear block shape differs from its matrix",
                ));
            }
        }
        Ok(Self { blocks })
    }

    /// Number of output ring elements of `N`.
    pub fn num_outputs(&self) -> usize {
        self.blocks.iter().map(|block| block.target.len()).sum()
    }

    /// `N·x` for centered coefficients `x` laid out like `s`, each at most
    /// `bound` in magnitude: `bound` must be a hard bound on every input
    /// (the Gaussian tail cut for a mask), never a statistical one.
    ///
    /// Constant time in `x`: the Ajtai map is, and the gadget lift is
    /// branch-free.
    pub fn apply(&self, x: &[i64], bound: u32) -> Result<Vec<CyclotomicRing<F, D>>, WhiteoutError> {
        let mut out = Vec::with_capacity(self.num_outputs());
        for block in &self.blocks {
            let input = x
                .get(block.input.clone())
                .ok_or(WhiteoutError::InvalidRelation(
                    "linear block reads outside s",
                ))?;
            let mut rows = block.matrix.apply_centered(input, bound)?;
            if let Some(gadget) = &block.gadget {
                let digits = x
                    .get(gadget.input.clone())
                    .ok_or(WhiteoutError::InvalidRelation("gadget reads outside s"))?;
                let radix = F::from_u64(1 << gadget.log_basis);
                for (row, planes) in rows.iter_mut().zip(digits.chunks_exact(gadget.levels * D)) {
                    let mut power = F::one();
                    for plane in planes.chunks_exact(D) {
                        let lifted: Vec<F> =
                            plane.iter().map(|&value| lift_ct::<F>(value)).collect();
                        *row -= CyclotomicRing::from_slice(&lifted).scale(&power);
                        power *= radix;
                    }
                }
            }
            out.extend(rows);
        }
        Ok(out)
    }

    /// The targets `t`, block by block.
    pub fn targets(&self) -> impl Iterator<Item = &CyclotomicRing<F, D>> {
        self.blocks.iter().flat_map(|block| block.target.iter())
    }
}

/// The extra row `L` of the pad commitment (Whiteout §4.2):
/// `L·r := π_0(Σ_j a_j r_j)`, row `rank` of the pad view on its randomness
/// columns, reduced to stride component 0. It is never part of the
/// commitment; it hides the cross term `E_1`.
pub struct LRow<F: Field, const D: usize> {
    pub matrix: Arc<PreparedMatrix<F, D>>,
    /// Native coefficient range of `s`: the pad commitment's randomness.
    pub input: Range<usize>,
}

impl<F: Field + CanonicalEncoding, const D: usize> LRow<F, D> {
    /// `L·x` for centered coefficients `x` laid out like `s`, each at most
    /// the hard bound `bound`; constant time in `x`.
    pub fn apply(
        &self,
        x: &[i64],
        bound: u32,
    ) -> Result<CyclotomicRing<F, CHALLENGE_RING_DIM>, WhiteoutError> {
        let input = x
            .get(self.input.clone())
            .ok_or(WhiteoutError::InvalidRelation("L reads outside s"))?;
        let [row] =
            <[CyclotomicRing<F, D>; 1]>::try_from(self.matrix.apply_centered(input, bound)?)
                .map_err(|_| WhiteoutError::InvalidRelation("L is not one row"))?;
        let stride = D / CHALLENGE_RING_DIM;
        let component: Vec<F> = row
            .coefficients()
            .iter()
            .step_by(stride.max(1))
            .copied()
            .collect();
        Ok(CyclotomicRing::from_slice(&component))
    }
}
