//! The lattice pad commitment (`specs/whiteout-sigma.md` §1.6).
//!
//! It commits, before the first challenge, the regular pads and the mask g as
//! radix-16 digit planes, the operand pads as radix-256 constant digit
//! polynomials, and the hiding randomness `r_p`, laid out by
//! [`PadPlacement`], with one Ajtai level of rank `rank` over `R_D`. The
//! image t_p is sent compressed, as cm(t_p) (`committed_pads.rs`). Row `rank`
//! of the same view, on the randomness columns, is the Σ-protocol's row `L`.

use std::sync::Arc;

#[cfg(feature = "whiteout-prover")]
use jolt_algebra::balanced_decompose_coefficients_pow2_i8_into;
#[cfg(feature = "whiteout-prover")]
use jolt_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
#[cfg(feature = "whiteout-prover")]
use jolt_algebra::CyclotomicRing;
use jolt_algebra::JoltField;
use jolt_crypto::lattice::{MatrixShape, PreparedMatrix, CHALLENGE_RING_DIM};
#[cfg(feature = "whiteout-prover")]
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

#[cfg(feature = "whiteout-prover")]
use super::dit::with_dit;
#[cfg(feature = "whiteout-prover")]
use super::placement::{PadPlace, OPERAND_LEVELS, PAD_LEVELS};
use super::placement::{PadPlacement, OPERAND_LOG_BASIS, PAD_LOG_BASIS};
use super::segments::{Segment, SegmentMask, SegmentRole, SegmentShape};
use super::WhiteoutError;
#[cfg(feature = "whiteout-prover")]
use crate::masking::PadIndex;
#[cfg(feature = "whiteout-prover")]
use crate::secret::SecretVec;

/// Radix of the hiding randomness digits `r_p`.
pub const PAD_RANDOMNESS_LOG_BASIS: u32 = 4;
pub(crate) const RANDOMNESS_LOG_BASIS: u32 = PAD_RANDOMNESS_LOG_BASIS;

#[cfg(feature = "whiteout-prover")]
/// The prover's opening of a pad commitment: the committed digits, which are
/// the pad segments of `s` in native order. Opaque and without `Debug`: it
/// holds the pads, g and the hiding randomness, and is zeroed when dropped.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct LatticePadOpening {
    pub(crate) digits: Vec<i64>,
}

/// Checked matrix and segment geometry, independent of public coefficients.
#[derive(Clone)]
pub struct PadGeometry<const D: usize> {
    placement: PadPlacement,
    rank: usize,
    pad_rings: usize,
    operand_rings: usize,
    randomness_columns: usize,
    width: usize,
    stream_len: usize,
}

impl<const D: usize> PadGeometry<D> {
    pub fn new(
        placement: PadPlacement,
        rank: usize,
        randomness_columns: usize,
    ) -> Result<Self, WhiteoutError> {
        let (pad_rings, operand_rings, width) = Self::dimensions(&placement, randomness_columns)?;
        let stream_len = Self::stream_len(&placement, rank, randomness_columns)?;
        Ok(Self {
            placement,
            rank,
            pad_rings,
            operand_rings,
            randomness_columns,
            width,
            stream_len,
        })
    }

    pub fn matrix_shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.rank,
            width: self.width,
            ring_dim: D,
        }
    }

    pub fn required_coefficients(&self) -> usize {
        self.stream_len
    }

    /// The pad and operand rings, padded to whole `R_D` elements, and the
    /// matrix width in `R_D` columns.
    fn dimensions(
        placement: &PadPlacement,
        randomness_columns: usize,
    ) -> Result<(usize, usize, usize), WhiteoutError> {
        let stride = D / CHALLENGE_RING_DIM;
        if stride == 0 || !D.is_multiple_of(CHALLENGE_RING_DIM) {
            return Err(WhiteoutError::InvalidRelation(
                "pad commitment ring dimension must be a multiple of 64",
            ));
        }
        let pad_rings = placement
            .pad_rings()
            .checked_next_multiple_of(stride)
            .ok_or(WhiteoutError::Overflow)?;
        let operand_rings = placement
            .operand_rings()
            .checked_next_multiple_of(stride)
            .ok_or(WhiteoutError::Overflow)?;
        Ok((
            pad_rings,
            operand_rings,
            pad_rings
                .checked_add(operand_rings)
                .and_then(|rings| (rings / stride).checked_add(randomness_columns))
                .ok_or(WhiteoutError::Overflow)?,
        ))
    }

    /// The matrix width in `R_D` columns of the commitment
    /// [`Self::new`] builds for `placement`.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn width(
        placement: &PadPlacement,
        randomness_columns: usize,
    ) -> Result<usize, WhiteoutError> {
        Ok(Self::dimensions(placement, randomness_columns)?.2)
    }

    /// The public stream length [`Self::new`] reads: rows `0..=rank` of the
    /// width [`Self::width`] over `R_D`.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn stream_len(
        placement: &PadPlacement,
        rank: usize,
        randomness_columns: usize,
    ) -> Result<usize, WhiteoutError> {
        let width = Self::width(placement, randomness_columns)?;
        rank.checked_add(1)
            .and_then(|rows| rows.checked_mul(width))
            .and_then(|len| len.checked_mul(D))
            .ok_or(WhiteoutError::Overflow)
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
}

/// One pad layout's lattice pad commitment over `R_D`.
pub struct LatticePadCommitment<F: JoltField, const D: usize> {
    geometry: PadGeometry<D>,
    matrix: Arc<PreparedMatrix<F, D>>,
    l_matrix: Arc<PreparedMatrix<F, D>>,
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
        let geometry = PadGeometry::new(placement, rank, randomness_columns)?;
        let source = PreparedMatrix::from_shared(
            MatrixShape {
                rows: 1,
                width: geometry.required_coefficients() / D,
                ring_dim: D,
            },
            Arc::new(stream.to_vec()),
        )?;
        Self::from_geometry(geometry, &source)
    }

    pub fn from_geometry(
        geometry: PadGeometry<D>,
        source: &PreparedMatrix<F, D>,
    ) -> Result<Self, WhiteoutError> {
        let shape = geometry.matrix_shape();
        let l_offset = geometry
            .stream_len
            .checked_sub(
                geometry
                    .randomness_columns
                    .checked_mul(D)
                    .ok_or(WhiteoutError::Overflow)?,
            )
            .ok_or(WhiteoutError::Overflow)?;
        Ok(Self {
            matrix: Arc::new(source.view(shape, 0)?),
            l_matrix: Arc::new(source.view(
                MatrixShape {
                    rows: 1,
                    width: geometry.randomness_columns,
                    ring_dim: D,
                },
                l_offset,
            )?),
            geometry,
        })
    }

    pub fn width(
        placement: &PadPlacement,
        randomness_columns: usize,
    ) -> Result<usize, WhiteoutError> {
        PadGeometry::<D>::width(placement, randomness_columns)
    }

    pub(crate) fn geometry(&self) -> &PadGeometry<D> {
        &self.geometry
    }

    /// The placement both roles lift the relation by.
    pub fn placement(&self) -> &PadPlacement {
        &self.geometry.placement
    }

    /// Coefficients of the image `A_p · s_p`: `rank` elements of `R_D`.
    pub fn image_len(&self) -> usize {
        self.geometry.rank * D
    }

    /// The hiding randomness columns `r_p` takes.
    pub fn randomness_columns(&self) -> usize {
        self.geometry.randomness_columns
    }

    /// The commitment matrix `A_p` (rows `0..rank`).
    pub(crate) fn matrix(&self) -> Arc<PreparedMatrix<F, D>> {
        Arc::clone(&self.matrix)
    }

    /// Row `L` on the randomness columns.
    pub(crate) fn l_matrix(&self) -> Arc<PreparedMatrix<F, D>> {
        Arc::clone(&self.l_matrix)
    }

    #[cfg(feature = "whiteout-prover")]
    /// The largest digit magnitude of the pad segments, a hard bound on
    /// every committed coefficient.
    fn digit_bound(&self) -> u64 {
        self.geometry
            .segments()
            .iter()
            .map(|segment| segment.digit_bound())
            .max()
            .unwrap_or(0)
    }

    /// Commits `pads` (the whole schedule), the mask `g` and the box digits
    /// `randomness` (`randomness_columns · D` of them). Constant time in
    /// the pads, g and the randomness: their digits come from the branch-free
    /// decomposition and the Ajtai map, run with DIT set on every thread
    /// (A11).
    #[cfg(feature = "whiteout-prover")]
    pub fn commit(
        &self,
        pads: &[F],
        g: CyclotomicRing<F, CHALLENGE_RING_DIM>,
        randomness: &[i8],
    ) -> Result<(Vec<CyclotomicRing<F, D>>, LatticePadOpening), WhiteoutError> {
        with_dit(|| self.commit_under_dit(pads, g, randomness))?
    }

    #[cfg(feature = "whiteout-prover")]
    fn commit_under_dit(
        &self,
        pads: &[F],
        g: CyclotomicRing<F, CHALLENGE_RING_DIM>,
        randomness: &[i8],
    ) -> Result<(Vec<CyclotomicRing<F, D>>, LatticePadOpening), WhiteoutError> {
        if pads.len() != self.geometry.placement.num_pads()
            || randomness.len() != self.geometry.randomness_columns * D
        {
            return Err(WhiteoutError::InvalidRelation(
                "pads or randomness differ from the commitment's layout",
            ));
        }
        // g is the group after the regular ones (`PadPlacement::g_ring`).
        let g_start = self.geometry.placement.regular_groups() * CHALLENGE_RING_DIM;
        let mut regular = SecretVec::from(vec![F::zero(); g_start + CHALLENGE_RING_DIM]);
        let mut operands = SecretVec::from(vec![F::zero(); self.geometry.placement.num_operands()]);
        for (pad, &value) in pads.iter().enumerate() {
            let slot = match self.geometry.placement.place(PadIndex(pad))? {
                PadPlace::Regular { group, position } => {
                    regular.get_mut(group * CHALLENGE_RING_DIM + position)
                }
                PadPlace::Operand(operand) => operands.get_mut(operand),
            };
            *slot.ok_or(WhiteoutError::InvalidRelation(
                "pad placed outside its group",
            ))? = value;
        }
        regular
            .get_mut(g_start..)
            .ok_or(WhiteoutError::InvalidRelation("g outside its group"))?
            .copy_from_slice(g.coefficients());

        let mut rings = Zeroizing::new(vec![
            [0i64; CHALLENGE_RING_DIM];
            self.geometry.pad_rings + self.geometry.operand_rings
        ]);
        let pad_params = BalancedDecomposePow2Params::<F>::new(PAD_LEVELS, PAD_LOG_BASIS);
        let mut planes = Zeroizing::new(vec![0i8; PAD_LEVELS * CHALLENGE_RING_DIM]);
        for (group, values) in regular
            .as_chunks::<CHALLENGE_RING_DIM>()
            .0
            .iter()
            .enumerate()
        {
            balanced_decompose_coefficients_pow2_i8_into(values, &mut planes, &pad_params);
            for (level, plane) in planes
                .as_chunks::<CHALLENGE_RING_DIM>()
                .0
                .iter()
                .enumerate()
            {
                let ring = rings
                    .get_mut(PadPlacement::regular_ring(group, level))
                    .ok_or(WhiteoutError::InvalidRelation("pad digits overflow"))?;
                for (dst, &digit) in ring.iter_mut().zip(plane) {
                    *dst = i64::from(digit);
                }
            }
        }
        let operand_params =
            BalancedDecomposePow2Params::<F>::new(OPERAND_LEVELS, OPERAND_LOG_BASIS);
        let mut digits = Zeroizing::new(vec![0i8; OPERAND_LEVELS]);
        for (operand, value) in operands.iter().enumerate() {
            balanced_decompose_coefficients_pow2_i8_into(
                std::slice::from_ref(value),
                &mut digits,
                &operand_params,
            );
            for (level, &digit) in digits.iter().enumerate() {
                let ring = rings
                    .get_mut(self.geometry.pad_rings + PadPlacement::operand_ring(operand, level))
                    .ok_or(WhiteoutError::InvalidRelation("operand digits overflow"))?;
                if let Some(constant) = ring.first_mut() {
                    *constant = i64::from(digit);
                }
            }
        }

        let packed_len = rings.len() * CHALLENGE_RING_DIM;
        let mut opening = LatticePadOpening {
            digits: vec![0i64; packed_len + randomness.len()],
        };
        let (packed, tail) = opening.digits.split_at_mut(packed_len);
        pack_native::<D>(&rings, packed);
        for (dst, &digit) in tail.iter_mut().zip(randomness) {
            *dst = i64::from(digit);
        }
        let image = self
            .matrix
            .apply_centered(&opening.digits, self.digit_bound())?;
        Ok((image, opening))
    }
}

/// Packs challenge-ring elements into native `R_D` elements by stride:
/// native coefficient `m·i + k` of element `e` is coefficient `i` of ring
/// `m·e + k`. `native` has `rings.len() · 64` coefficients.
#[cfg(feature = "whiteout-prover")]
fn pack_native<const D: usize>(rings: &[[i64; CHALLENGE_RING_DIM]], native: &mut [i64]) {
    let stride = D / CHALLENGE_RING_DIM;
    for (element, components) in native
        .as_chunks_mut::<D>()
        .0
        .iter_mut()
        .zip(rings.chunks_exact(stride))
    {
        for (k, ring) in components.iter().enumerate() {
            for (dst, &value) in element.iter_mut().skip(k).step_by(stride).zip(ring) {
                *dst = value;
            }
        }
    }
}
