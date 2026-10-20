//! A synthetic Whiteout instance over `R_D`: a residual witness (public and
//! Gaussian parts) under a two-level commitment, five masked messages and one
//! masked product. Shared by the in-crate tests and the timing diagnostics.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "a fixed synthetic fixture on valid shapes"
)]

use std::sync::Arc;

use jolt_algebra::{
    balanced_decompose_coefficients_pow2_i8_into, ring::cyclotomic::BalancedDecomposePow2Params,
    CyclotomicRing, Field, One, Prime128OffsetA7F7, Ring,
};
use jolt_crypto::lattice::{
    derive_public_matrix_prefix, sample_box_digits, MatrixShape, MatrixView, PreparedMatrix,
    PublicMatrixSeed, CHALLENGE_RING_DIM,
};
use jolt_poly::EqPolynomial;
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use super::placement::PadPlacement;
use super::segments::{Segment, SegmentMask, SegmentRole, SegmentShape};
use super::{
    sample_zero_constant_term, EvaluationClaims, LatticePadCommitment, ResidualBlock,
    ResidualGadget, SigmaSetup, FIELD_LEVELS,
};
use crate::masking::{
    AffineRow, MaskedOperand, PadIndex, PadLayout, ProductRow, ResidualRelation, ResidualVar,
};

pub(crate) type F = Prime128OffsetA7F7;
pub(crate) const BASE_PADS: usize = 5;
pub(crate) const PAD_RANK: usize = 2;
pub(crate) const PAD_RANDOMNESS_COLUMNS: usize = 2;

pub(crate) fn stream(len: usize, seed: u8) -> Vec<F> {
    derive_public_matrix_prefix(len, &PublicMatrixSeed::shake256_paged_v1([seed; 32])).unwrap()
}

pub(crate) fn prepared<const D: usize>(
    rows: usize,
    width: usize,
    seed: u8,
) -> Arc<PreparedMatrix<F, D>> {
    let shape = MatrixShape {
        rows,
        width,
        ring_dim: D,
    };
    let data = stream(shape.num_field_elements().unwrap(), seed);
    Arc::new(PreparedMatrix::new(&MatrixView::new(shape, &data).unwrap()).unwrap())
}

pub(crate) fn digits(rng: &mut impl RngCore, len: usize) -> Vec<i64> {
    (0..len)
        .map(|_| i64::from(rng.next_u32() % 16) - 8)
        .collect()
}

pub(crate) fn ring_digits<const D: usize>(ring: &CyclotomicRing<F, D>) -> Vec<i64> {
    let params = BalancedDecomposePow2Params::<F>::new(FIELD_LEVELS, 4);
    let mut out = vec![0i8; FIELD_LEVELS * D];
    balanced_decompose_coefficients_pow2_i8_into(ring.coefficients(), &mut out, &params);
    out.into_iter().map(i64::from).collect()
}

/// Segment lengths, in challenge-ring elements; `scale` multiplies the
/// witness `w`.
pub(crate) fn lengths<const D: usize>(scale: usize) -> [usize; 4] {
    let m = D / CHALLENGE_RING_DIM;
    [2 * m * scale, 4 * m * scale, FIELD_LEVELS * m, m]
}

pub(crate) fn residual_segments<const D: usize>(scale: usize) -> Vec<Segment> {
    let [public, witness, inner, randomness] = lengths::<D>(scale);
    let segment = |role, len, mask| Segment {
        role,
        len,
        native_dim: D,
        log_basis: 4,
        shape: SegmentShape::Dense,
        mask,
    };
    vec![
        segment(SegmentRole::Witness, public, SegmentMask::Public),
        segment(SegmentRole::Witness, witness, SegmentMask::Gaussian),
        segment(SegmentRole::InnerDigits, inner, SegmentMask::Gaussian),
        segment(
            SegmentRole::ResidualRandomness,
            randomness,
            SegmentMask::Gaussian,
        ),
    ]
}

/// Inner `A` (rank 1 over the witness) and outer `B` (rank 2 over the inner
/// digits and the randomness).
pub(crate) fn residual_matrices<const D: usize>(
    scale: usize,
) -> (Arc<PreparedMatrix<F, D>>, Arc<PreparedMatrix<F, D>>) {
    let [public, witness, inner, randomness] = lengths::<D>(scale);
    let m = D / CHALLENGE_RING_DIM;
    (
        prepared::<D>(1, (public + witness) / m, 11),
        prepared::<D>(2, (inner + randomness) / m, 12),
    )
}

pub(crate) fn layout() -> PadLayout {
    PadLayout::new(BASE_PADS, 1)
        .unwrap()
        .with_operand_pads(vec![PadIndex(0)])
}

pub(crate) fn pad_commitment<const D: usize>() -> LatticePadCommitment<F, D> {
    let placement = PadPlacement::new(&layout()).unwrap();
    LatticePadCommitment::new(
        placement,
        &stream(1 << 18, 21),
        PAD_RANK,
        PAD_RANDOMNESS_COLUMNS,
    )
    .unwrap()
}

/// One synthetic instance: a residual witness `w` (a public and a Gaussian
/// part) under a two-level commitment, five masked messages and one masked
/// product, over `R_D`.
pub(crate) struct Instance<const D: usize> {
    pub(crate) relation: ResidualRelation<F>,
    pub(crate) claims: EvaluationClaims<F>,
    /// `w ‖ t̂ ‖ r_a`, then the pad digits.
    pub(crate) s: Vec<i64>,
    pub(crate) u: Vec<CyclotomicRing<F, D>>,
    pub(crate) t_p: Vec<CyclotomicRing<F, D>>,
    /// The multiple of the base witness length.
    pub(crate) scale: usize,
}

impl<const D: usize> Instance<D> {
    /// Rows `m2 + m3 = sum`, `w̃(r) = m4`, and `m0 · m1 = aux`.
    #[cfg(test)]
    pub(crate) fn new(seed: u64) -> Self {
        Self::scaled(seed, 1)
    }

    /// [`Self::new`] with a witness `scale` times longer.
    pub(crate) fn scaled(seed: u64, scale: usize) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let c = CHALLENGE_RING_DIM;
        let [public, witness, _, randomness] = lengths::<D>(scale);
        let layout = layout();
        let pads: Vec<F> = (0..layout.num_pads).map(|_| F::random(&mut rng)).collect();
        let mut messages: Vec<F> = (0..BASE_PADS).map(|_| F::random(&mut rng)).collect();

        let w = digits(&mut rng, (public + witness) * c);
        let num_vars = w.len().next_power_of_two().trailing_zeros() as usize;
        let point: Vec<F> = (0..num_vars).map(|_| F::random(&mut rng)).collect();
        let eq = EqPolynomial::new(point.clone()).evaluations();
        messages[4] = w.iter().zip(&eq).map(|(&w, e)| F::from_i64(w) * *e).sum();
        let auxiliary = messages[0] * messages[1];
        let masked: Vec<F> = messages
            .iter()
            .chain(std::iter::once(&auxiliary))
            .zip(&pads)
            .map(|(m, p)| *m + *p)
            .collect();
        let slot = |index: usize| AffineRow::masked(masked[index], PadIndex(index));
        let mut relation = ResidualRelation::new(layout, 1);
        relation.affine.push(
            slot(2)
                .add_scaled(&slot(3), F::one())
                .add_scaled(&AffineRow::constant(-(messages[2] + messages[3])), F::one()),
        );
        relation
            .affine
            .push(AffineRow::variable(ResidualVar::Witness(0)).add_scaled(&slot(4), -F::one()));
        relation.products.push(ProductRow {
            x: MaskedOperand {
                masked: masked[0],
                pad: PadIndex(0),
            },
            y: slot(1),
            z: slot(5),
        });

        let pads_commitment = pad_commitment::<D>();
        let g = sample_zero_constant_term::<F, CHALLENGE_RING_DIM, _>(&mut rng);
        let r_p = sample_box_digits(&mut rng, PAD_RANDOMNESS_COLUMNS * D, 4).unwrap();
        let (t_p, opening) = pads_commitment.commit(&pads, g, &r_p).unwrap();

        let (a, b) = residual_matrices::<D>(scale);
        let t = a.apply_centered(&w, 8).unwrap();
        let t_hat: Vec<i64> = t.iter().flat_map(ring_digits::<D>).collect();
        let r_a = digits(&mut rng, randomness * c);
        let outer: Vec<i64> = t_hat.iter().chain(&r_a).copied().collect();
        let u = b.apply_centered(&outer, 8).unwrap();
        let s = w
            .iter()
            .chain(&t_hat)
            .chain(&r_a)
            .chain(&opening.digits)
            .copied()
            .collect();
        Self {
            relation,
            claims: EvaluationClaims {
                points: vec![point],
            },
            s,
            u,
            t_p,
            scale,
        }
    }

    pub(crate) fn setup(&self) -> SigmaSetup<F, D> {
        let (a, b) = residual_matrices::<D>(self.scale);
        let blocks = vec![
            ResidualBlock {
                matrix: a,
                input: 0..2,
                gadget: Some(ResidualGadget {
                    segment: 2,
                    log_basis: 4,
                    levels: FIELD_LEVELS,
                }),
                target: vec![CyclotomicRing::zero()],
            },
            ResidualBlock {
                matrix: b,
                input: 2..4,
                gadget: None,
                target: self.u.clone(),
            },
        ];
        SigmaSetup::new(
            residual_segments::<D>(self.scale),
            blocks,
            &pad_commitment::<D>(),
            self.t_p.clone(),
            1,
            &stream(1 << 18, 31),
        )
        .unwrap()
    }

    /// Recomputes both images from `s`, so `N·s = t` holds after an edit.
    #[cfg(test)]
    pub(crate) fn reopen(&mut self) {
        let setup = self.setup();
        let images = setup.linear().apply(&self.s, 1 << 7).unwrap();
        let (u, t_p) = images.split_at(1 + self.u.len());
        self.u = u[1..].to_vec();
        self.t_p = t_p.to_vec();
    }

    /// The first coefficient of `role`'s segment in `s`.
    #[cfg(test)]
    pub(crate) fn start(&self, role: SegmentRole) -> usize {
        self.setup().table().unique(role).unwrap().coefficient
    }
}
