//! The lattice pad commitment (`specs/whiteout-sigma.md` §1.6).
//!
//! It commits, before the first challenge, the regular pads and the mask g as
//! radix-16 digit planes, the operand pads as radix-256 constant digit
//! polynomials, and the hiding randomness `r_p`, laid out by
//! [`PadPlacement`], with one Ajtai level of rank `rank` over `R_D`. The
//! commitment is sent raw until W5 provides compression. Row `rank` of the
//! same view, on the randomness columns, is the Σ-protocol's row `L`.

use std::sync::Arc;

#[cfg(feature = "whiteout-prover")]
use jolt_algebra::balanced_decompose_coefficients_pow2_i8_into;
#[cfg(feature = "whiteout-prover")]
use jolt_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use jolt_algebra::{CyclotomicRing, JoltField};
use jolt_crypto::lattice::{MatrixShape, MatrixView, PreparedMatrix, CHALLENGE_RING_DIM};

use super::linear::{LRow, LinearBlock};
#[cfg(feature = "whiteout-prover")]
use super::placement::{PadPlace, OPERAND_LEVELS, PAD_LEVELS};
use super::placement::{PadPlacement, OPERAND_LOG_BASIS, PAD_LOG_BASIS};
use super::segments::{Segment, SegmentMask, SegmentRole, SegmentShape};
use super::WhiteoutError;
#[cfg(feature = "whiteout-prover")]
use crate::masking::PadIndex;

/// Radix of the hiding randomness digits.
pub const RANDOMNESS_LOG_BASIS: u32 = 4;

/// The prover's opening of a pad commitment: the mask g and the committed
/// digits, which are the pad segments of `s` in native order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatticePadOpening<F: JoltField> {
    pub g: CyclotomicRing<F, CHALLENGE_RING_DIM>,
    pub digits: Vec<i64>,
}

/// One pad layout's lattice pad commitment over `R_D`.
pub struct LatticePadCommitment<F: JoltField, const D: usize> {
    placement: PadPlacement,
    matrix: Arc<PreparedMatrix<F, D>>,
    l_matrix: Arc<PreparedMatrix<F, D>>,
    pad_rings: usize,
    operand_rings: usize,
    randomness_columns: usize,
}

impl<F: JoltField, const D: usize> LatticePadCommitment<F, D> {
    /// The commitment of rank `rank` with `randomness_columns` hiding
    /// columns, its matrix and row `L` cut from the public `stream`.
    pub fn new(
        placement: PadPlacement,
        stream: &[F],
        rank: usize,
        randomness_columns: usize,
    ) -> Result<Self, WhiteoutError> {
        let stride = D / CHALLENGE_RING_DIM;
        if stride == 0 || !D.is_multiple_of(CHALLENGE_RING_DIM) {
            return Err(WhiteoutError::InvalidRelation(
                "pad commitment ring dimension must be a multiple of 64",
            ));
        }
        let pad_rings = placement.pad_rings().next_multiple_of(stride);
        let operand_rings = placement.operand_rings().next_multiple_of(stride);
        let width = (pad_rings + operand_rings) / stride + randomness_columns;
        let full = MatrixView::new(
            MatrixShape {
                rows: rank + 1,
                width,
                ring_dim: D,
            },
            stream,
        )?;
        let commitment = MatrixView::new(
            MatrixShape {
                rows: rank,
                width,
                ring_dim: D,
            },
            stream,
        )?;
        let mut l_entries = Vec::with_capacity(randomness_columns * D);
        for column in width - randomness_columns..width {
            l_entries.extend_from_slice(full.entry(rank, column)?);
        }
        let l_view = MatrixView::new(
            MatrixShape {
                rows: 1,
                width: randomness_columns,
                ring_dim: D,
            },
            &l_entries,
        )?;
        Ok(Self {
            placement,
            matrix: Arc::new(PreparedMatrix::new(&commitment)?),
            l_matrix: Arc::new(PreparedMatrix::new(&l_view)?),
            pad_rings,
            operand_rings,
            randomness_columns,
        })
    }

    /// The placement both roles lift the relation by.
    pub fn placement(&self) -> &PadPlacement {
        &self.placement
    }

    /// The pad segments of `s`, in order: pad digits, operand digits (if any
    /// operand pads), randomness. All Gaussian-masked.
    pub fn segments(&self) -> Vec<Segment> {
        let segment = |role, len, log_basis, shape| Segment {
            role,
            len,
            native_dim: D,
            log_basis,
            shape,
            mask: SegmentMask::Gaussian,
        };
        let mut segments = vec![segment(
            SegmentRole::PadDigits,
            self.pad_rings,
            PAD_LOG_BASIS,
            SegmentShape::Dense,
        )];
        if self.operand_rings > 0 {
            segments.push(segment(
                SegmentRole::OperandDigits,
                self.operand_rings,
                OPERAND_LOG_BASIS,
                SegmentShape::ConstantPoly,
            ));
        }
        segments.push(segment(
            SegmentRole::PadRandomness,
            self.randomness_columns * D / CHALLENGE_RING_DIM,
            RANDOMNESS_LOG_BASIS,
            SegmentShape::Dense,
        ));
        segments
    }

    /// Native coefficients of the pad segments.
    pub fn num_coefficients(&self) -> usize {
        (self.pad_rings + self.operand_rings) * CHALLENGE_RING_DIM + self.randomness_columns * D
    }

    /// The opening rows `A_p·(pad segments) = image`, for pad segments
    /// starting at native coefficient `offset` of `s`.
    pub fn linear_block(
        &self,
        offset: usize,
        image: Vec<CyclotomicRing<F, D>>,
    ) -> LinearBlock<F, D> {
        LinearBlock {
            matrix: Arc::clone(&self.matrix),
            input: offset..offset + self.num_coefficients(),
            gadget: None,
            target: image,
        }
    }

    /// Row `L`, for pad segments starting at native coefficient `offset`.
    pub fn l_row(&self, offset: usize) -> LRow<F, D> {
        let end = offset + self.num_coefficients();
        LRow {
            matrix: Arc::clone(&self.l_matrix),
            input: end - self.randomness_columns * D..end,
        }
    }

    /// Commits `pads` (the whole schedule), the mask `g` and the box digits
    /// `randomness` (`randomness_columns · D` of them). Constant time in
    /// the pads, g and the randomness: their digits come from the branch-free
    /// decomposition and the Ajtai map.
    #[cfg(feature = "whiteout-prover")]
    pub fn commit(
        &self,
        pads: &[F],
        g: CyclotomicRing<F, CHALLENGE_RING_DIM>,
        randomness: &[i8],
    ) -> Result<(Vec<CyclotomicRing<F, D>>, LatticePadOpening<F>), WhiteoutError> {
        if pads.len() != self.placement.num_pads()
            || randomness.len() != self.randomness_columns * D
        {
            return Err(WhiteoutError::InvalidRelation(
                "pads or randomness differ from the commitment's layout",
            ));
        }
        let mut regular = vec![F::zero(); self.placement.regular_groups() * CHALLENGE_RING_DIM];
        let mut operands = vec![F::zero(); self.placement.operand_pads().len()];
        for (pad, &value) in pads.iter().enumerate() {
            let slot = match self.placement.place(PadIndex(pad))? {
                PadPlace::Regular { group, position } => {
                    regular.get_mut(group * CHALLENGE_RING_DIM + position)
                }
                PadPlace::Operand(operand) => operands.get_mut(operand),
            };
            *slot.ok_or(WhiteoutError::InvalidRelation(
                "pad placed outside its group",
            ))? = value;
        }
        regular.extend_from_slice(g.coefficients());

        let mut rings = vec![[0i64; CHALLENGE_RING_DIM]; self.pad_rings + self.operand_rings];
        let pad_params = BalancedDecomposePow2Params::<F>::new(PAD_LEVELS, PAD_LOG_BASIS);
        let mut planes = vec![0i8; PAD_LEVELS * CHALLENGE_RING_DIM];
        for (group, values) in regular.chunks_exact(CHALLENGE_RING_DIM).enumerate() {
            balanced_decompose_coefficients_pow2_i8_into(values, &mut planes, &pad_params);
            for (level, plane) in planes.chunks_exact(CHALLENGE_RING_DIM).enumerate() {
                let ring = rings
                    .get_mut(group * PAD_LEVELS + level)
                    .ok_or(WhiteoutError::InvalidRelation("pad digits overflow"))?;
                for (dst, &digit) in ring.iter_mut().zip(plane) {
                    *dst = i64::from(digit);
                }
            }
        }
        let operand_params =
            BalancedDecomposePow2Params::<F>::new(OPERAND_LEVELS, OPERAND_LOG_BASIS);
        let mut digits = vec![0i8; OPERAND_LEVELS];
        for (operand, value) in operands.iter().enumerate() {
            balanced_decompose_coefficients_pow2_i8_into(
                std::slice::from_ref(value),
                &mut digits,
                &operand_params,
            );
            for (level, &digit) in digits.iter().enumerate() {
                let ring = rings
                    .get_mut(self.pad_rings + operand * OPERAND_LEVELS + level)
                    .ok_or(WhiteoutError::InvalidRelation("operand digits overflow"))?;
                if let Some(constant) = ring.first_mut() {
                    *constant = i64::from(digit);
                }
            }
        }

        let mut native = pack_native::<D>(&rings);
        native.extend(randomness.iter().map(|&digit| i64::from(digit)));
        let image = self.matrix.apply_centered(&native, 1 << 7)?;
        Ok((image, LatticePadOpening { g, digits: native }))
    }
}

/// Packs challenge-ring elements into native `R_D` elements by stride:
/// native coefficient `m·i + k` of element `e` is coefficient `i` of ring
/// `m·e + k`.
#[cfg(feature = "whiteout-prover")]
fn pack_native<const D: usize>(rings: &[[i64; CHALLENGE_RING_DIM]]) -> Vec<i64> {
    let stride = D / CHALLENGE_RING_DIM;
    let mut native = vec![0i64; rings.len() * CHALLENGE_RING_DIM];
    for (element, components) in native.chunks_exact_mut(D).zip(rings.chunks_exact(stride)) {
        for (k, ring) in components.iter().enumerate() {
            for (dst, &value) in element.iter_mut().skip(k).step_by(stride).zip(ring) {
                *dst = value;
            }
        }
    }
    native
}
