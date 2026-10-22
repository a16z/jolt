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
use crate::secret::SecretVec;

/// One block of the residual commitment's opening rows,
/// `M·(input segments) − G(gadget segment) = target`, addressed by residual
/// segment index: `SigmaSetup` turns the indices into positions in `s`, so a
/// caller never handles raw offsets.
pub struct ResidualBlock<F: Field, const D: usize> {
    pub matrix: Arc<PreparedMatrix<F, D>>,
    /// Consecutive residual segments the matrix reads.
    pub input: Range<usize>,
    pub gadget: Option<ResidualGadget>,
    pub target: BlockTarget<F, D>,
}

/// What a block of `N` must reach, as the verifier holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockTarget<F: Field, const D: usize> {
    /// A public image, such as the inner rows' zero.
    Public(Vec<CyclotomicRing<F, D>>),
    /// A commitment image the verifier holds only as its compressed payload
    /// (Whiteout §2); the prover also holds the image.
    Compressed {
        payload: Vec<F>,
        image: Option<Vec<CyclotomicRing<F, D>>>,
    },
}

/// The gadget term of a [`ResidualBlock`]: the residual segment holding the
/// digits of the matrix's image, `rows × levels` native elements ordered
/// `[row][digit]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidualGadget {
    pub segment: usize,
    pub log_basis: u32,
    pub levels: usize,
}

/// `G(x) = Σ_j b^j · plane_j` per output row, over `rows × levels` native
/// elements ordered `[row][digit]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GadgetTerm {
    /// Native coefficient range of `s`.
    pub input: Range<usize>,
    pub log_basis: u32,
    pub levels: usize,
}

/// One block `M·x[input] − G(x[gadget]) = target`.
pub(crate) struct LinearBlock<F: Field, const D: usize> {
    pub matrix: Arc<PreparedMatrix<F, D>>,
    /// Native coefficient range of `s`, `matrix.width · D` long.
    pub input: Range<usize>,
    pub gadget: Option<GadgetTerm>,
    pub target: BlockTarget<F, D>,
}

/// `N` and its targets `t`.
pub(crate) struct RingLinearMap<F: Field, const D: usize> {
    blocks: Vec<LinearBlock<F, D>>,
}

impl<F: Field + CanonicalEncoding, const D: usize> RingLinearMap<F, D> {
    pub fn new(blocks: Vec<LinearBlock<F, D>>) -> Result<Self, WhiteoutError> {
        for block in &blocks {
            let shape = block.matrix.shape();
            let columns = block.input.len();
            let rows_ok = match &block.target {
                BlockTarget::Public(image)
                | BlockTarget::Compressed {
                    image: Some(image), ..
                } => image.len() == shape.rows,
                BlockTarget::Compressed { image: None, .. } => true,
            };
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
        self.blocks
            .iter()
            .map(|block| block.matrix.shape().rows)
            .sum()
    }

    /// `N·x` for centered coefficients `x` laid out like `s`, each at most
    /// `bound` in magnitude: `bound` must be a hard bound on every input
    /// (the Gaussian tail cut for a mask), never a statistical one.
    ///
    /// Constant time in `x`: the Ajtai map is, and the gadget lift is
    /// branch-free. `N·x` is as secret as `x` (a mask), so it is zeroed
    /// when dropped.
    pub fn apply(
        &self,
        x: &[i64],
        bound: u64,
    ) -> Result<SecretVec<CyclotomicRing<F, D>>, WhiteoutError> {
        let mut out = SecretVec::from(vec![CyclotomicRing::zero(); self.num_outputs()]);
        let mut written = 0;
        for block in &self.blocks {
            let input = x
                .get(block.input.clone())
                .ok_or(WhiteoutError::InvalidRelation(
                    "linear block reads outside s",
                ))?;
            let mut rows = SecretVec::from(block.matrix.apply_centered(input, bound)?);
            if let Some(gadget) = &block.gadget {
                let digits = x
                    .get(gadget.input.clone())
                    .ok_or(WhiteoutError::InvalidRelation("gadget reads outside s"))?;
                let radix = F::from_u64(1 << gadget.log_basis);
                for (row, planes) in rows.iter_mut().zip(digits.chunks_exact(gadget.levels * D)) {
                    let mut power = F::one();
                    for plane in planes.chunks_exact(D) {
                        for (dst, &value) in row.coefficients_mut().iter_mut().zip(plane) {
                            *dst -= lift_ct::<F>(value) * power;
                        }
                        power *= radix;
                    }
                }
            }
            out.get_mut(written..written + rows.len())
                .ok_or(WhiteoutError::InvalidRelation(
                    "a block's image differs from its target",
                ))?
                .copy_from_slice(&rows);
            written += rows.len();
        }
        Ok(out)
    }

    /// Adds to `out` (laid out like `s`) the weights of `⟨ω, N·x⟩` for output
    /// weights `weights`, `D` per output ring in [`Self::apply`]'s order.
    pub fn adjoint(&self, weights: &[F], out: &mut [F]) -> Result<(), WhiteoutError> {
        if weights.len() != self.num_outputs() * D {
            return Err(WhiteoutError::InvalidRelation(
                "linear weights differ from N's outputs",
            ));
        }
        let mut rest = weights;
        for block in &self.blocks {
            let (block_weights, tail) = rest.split_at(block.matrix.shape().rows * D);
            rest = tail;
            let rows: Vec<CyclotomicRing<F, D>> = block_weights
                .chunks_exact(D)
                .map(CyclotomicRing::from_slice)
                .collect();
            matrix_adjoint(&block.matrix, &rows, block.input.clone(), out)?;
            if let Some(gadget) = &block.gadget {
                let targets = out
                    .get_mut(gadget.input.clone())
                    .ok_or(WhiteoutError::InvalidRelation("gadget reads outside s"))?;
                let radix = F::from_u64(1 << gadget.log_basis);
                for (row, planes) in rows.iter().zip(targets.chunks_exact_mut(gadget.levels * D)) {
                    let mut power = F::one();
                    for plane in planes.chunks_exact_mut(D) {
                        for (dst, weight) in plane.iter_mut().zip(row.coefficients()) {
                            *dst -= *weight * power;
                        }
                        power *= radix;
                    }
                }
            }
        }
        Ok(())
    }

    /// `N·x` for `x` laid out like `s` whose native element `E` is
    /// `scales[E]·lo`: a column combination and one ring product per row,
    /// with no product per matrix entry.
    pub fn apply_factored(
        &self,
        scales: &[F],
        lo: &CyclotomicRing<F, D>,
    ) -> Result<Vec<CyclotomicRing<F, D>>, WhiteoutError> {
        let mut out = Vec::with_capacity(self.num_outputs());
        for block in &self.blocks {
            let columns = scales
                .get(block.input.start / D..block.input.end / D)
                .ok_or(WhiteoutError::InvalidRelation(
                    "linear block reads outside s",
                ))?;
            let mut rows: Vec<CyclotomicRing<F, D>> = block
                .matrix
                .combine_columns(columns)?
                .into_iter()
                .map(|row| row * *lo)
                .collect();
            if let Some(gadget) = &block.gadget {
                let planes = scales
                    .get(gadget.input.start / D..gadget.input.end / D)
                    .ok_or(WhiteoutError::InvalidRelation("gadget reads outside s"))?;
                let radix = F::from_u64(1 << gadget.log_basis);
                for (row, planes) in rows.iter_mut().zip(planes.chunks_exact(gadget.levels)) {
                    let mut power = F::one();
                    let mut sum = F::zero();
                    for scale in planes {
                        sum += *scale * power;
                        power *= radix;
                    }
                    *row -= lo.scale(&sum);
                }
            }
            out.extend(rows);
        }
        Ok(out)
    }

    /// Each block's output count and target, in block order.
    pub fn targets(&self) -> impl Iterator<Item = (usize, &BlockTarget<F, D>)> {
        self.blocks
            .iter()
            .map(|block| (block.matrix.shape().rows, &block.target))
    }
}

/// The extra row `L` of the pad commitment (Whiteout §4.2):
/// `L·r := π_0(Σ_j a_j r_j)`, row `rank` of the pad view on its randomness
/// columns, reduced to stride component 0. It is never part of the
/// commitment; it hides the cross term `E_1`.
pub(crate) struct LRow<F: Field, const D: usize> {
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
        bound: u64,
    ) -> Result<CyclotomicRing<F, CHALLENGE_RING_DIM>, WhiteoutError> {
        let input = x
            .get(self.input.clone())
            .ok_or(WhiteoutError::InvalidRelation("L reads outside s"))?;
        let rows = SecretVec::from(self.matrix.apply_centered(input, bound)?);
        let [row] = &*rows else {
            return Err(WhiteoutError::InvalidRelation("L is not one row"));
        };
        Ok(stride_component(row))
    }

    /// `L·x` for `x` factored as in [`RingLinearMap::apply_factored`].
    pub fn apply_factored(
        &self,
        scales: &[F],
        lo: &CyclotomicRing<F, D>,
    ) -> Result<CyclotomicRing<F, CHALLENGE_RING_DIM>, WhiteoutError> {
        let columns = scales
            .get(self.input.start / D..self.input.end / D)
            .ok_or(WhiteoutError::InvalidRelation("L reads outside s"))?;
        let [row] = <[CyclotomicRing<F, D>; 1]>::try_from(self.matrix.combine_columns(columns)?)
            .map_err(|_| WhiteoutError::InvalidRelation("L is not one row"))?;
        Ok(stride_component(&(row * *lo)))
    }

    /// Adds to `out` (laid out like `s`) the weights of `⟨ω, L·x⟩`.
    pub fn adjoint(
        &self,
        weight: &CyclotomicRing<F, CHALLENGE_RING_DIM>,
        out: &mut [F],
    ) -> Result<(), WhiteoutError> {
        let stride = (D / CHALLENGE_RING_DIM).max(1);
        let mut native = CyclotomicRing::<F, D>::zero();
        for (dst, w) in native
            .coefficients_mut()
            .iter_mut()
            .step_by(stride)
            .zip(weight.coefficients())
        {
            *dst = *w;
        }
        matrix_adjoint(&self.matrix, &[native], self.input.clone(), out)
    }
}

/// Stride component 0 of a native element: `π_0`.
fn stride_component<F: Field, const D: usize>(
    row: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, CHALLENGE_RING_DIM> {
    let mut out = CyclotomicRing::zero();
    for (dst, value) in out.coefficients_mut().iter_mut().zip(
        row.coefficients()
            .iter()
            .step_by((D / CHALLENGE_RING_DIM).max(1)),
    ) {
        *dst = *value;
    }
    out
}

/// Adds to `out[input]` the weights of `⟨ω, M·x[input]⟩`, `ω` one ring per
/// row: column `j` takes `σ(Σ_r σ(ω_r)·M[r][j])`, since
/// `⟨ω, m·x⟩ = ct(σ(ω)·m·x)`.
fn matrix_adjoint<F: Field + CanonicalEncoding, const D: usize>(
    matrix: &PreparedMatrix<F, D>,
    weights: &[CyclotomicRing<F, D>],
    input: Range<usize>,
    out: &mut [F],
) -> Result<(), WhiteoutError> {
    let sigma: Vec<CyclotomicRing<F, D>> = weights.iter().map(CyclotomicRing::sigma_m1).collect();
    let columns = matrix.apply_transposed(&sigma)?;
    let targets = out
        .get_mut(input)
        .ok_or(WhiteoutError::InvalidRelation("matrix reads outside s"))?;
    for (column, dst) in columns.iter().zip(targets.chunks_exact_mut(D)) {
        for (dst, w) in dst.iter_mut().zip(column.sigma_m1().coefficients()) {
            *dst += *w;
        }
    }
    Ok(())
}
