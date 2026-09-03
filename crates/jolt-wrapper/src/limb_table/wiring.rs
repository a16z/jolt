//! Wiring: the program's slots are a public sparse matrix
//! `X_s(row) = κ_s(row)·Z(src_x(row, s))`, `Y_s(row) = Z(src_y(row, s))`.
//! The row sumcheck consumes the operand limbs as virtual polynomials; the
//! linking sumcheck proves their claimed evaluations at the row point `r`
//! against the committed `z` chunks:
//! `Σ_{s,side,a} γ_{s,side,a}·V_{s,side,a}(r) = Σ_v Σ_a Wc_a(v)·Z_a(v)`,
//! with `Wc_a(v) = Σ_row eq(r,row)·Σ_{s,side} γ_{s,side,a}·κ·[v = src]`.
//! At the end the verifier needs `Σ_a Z_a(r')·Wc_a(r')`, which regroups as
//! `Σ_pieces K_piece(r, r')·G_{s,side}` with `G_{s,side} = Σ_a γ_{s,side,a}·Z_a(r')`:
//! the fixed pieces are kernel MLEs ([`super::layout::Kernel::mle`]), the
//! digit-selected pieces are proven by the operand-selection sumcheck.

use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use rayon::prelude::*;

use super::columns::{recompose, Columns, Constants, LIMBS, Z_CHUNKS};
use super::layout::{Piece, Side};
use super::program::{signed, Program};
use super::relation::RowClaims;

/// The wiring of a program: slot `s` of `row` reads `κ·Z(x)` and `Z(y)`.
pub struct Wiring<'p> {
    pub program: &'p Program,
    pub num_slots: usize,
}

impl Wiring<'_> {
    /// Fills `out` (`6·num_slots` entries, slot-major `x_0 x_1 x_2 y_0 y_1 y_2`)
    /// with the operand limbs of `row`; padding rows, unused slots and
    /// zero-coefficient placeholders are zero.
    pub fn operand_limbs(&self, columns: &Columns, row: usize, out: &mut [Fr]) {
        out.fill(Fr::zero());
        let Some(spec) = self.program.rows.get(row) else {
            return;
        };
        for (s, slot) in spec.slots.iter().enumerate() {
            if slot.kappa == 0 {
                continue;
            }
            let kappa = fr_signed(slot.kappa);
            let x = &columns.limbs[slot.x as usize];
            let y = &columns.limbs[slot.y as usize];
            for a in 0..LIMBS {
                out[2 * LIMBS * s + a] = kappa * x[a];
                out[2 * LIMBS * s + LIMBS + a] = y[a];
            }
        }
    }
}

fn fr_signed(k: i32) -> Fr {
    Fr::from_i64(i64::from(k))
}

/// Public weights of the linking claim: `gamma[6s + a]` for `x_{s,a}`,
/// `gamma[6s + 3 + a]` for `y_{s,a}`.
pub struct LinkParams {
    pub log_rows: usize,
    pub gamma: Vec<Fr>,
}

/// `G_{s,side} = Σ_a γ_{s,side,a}·Z_a(r')`: the per-slot-side weight of a
/// wiring piece once the source point's limb claims are known.
pub struct SlotWeights {
    weights: Vec<[Fr; 2]>,
}

impl SlotWeights {
    pub fn get(&self, slot: u8, side: Side) -> Fr {
        let pair = self.weights[usize::from(slot)];
        match side {
            Side::X => pair[0],
            Side::Y => pair[1],
        }
    }
}

impl LinkParams {
    pub fn new(log_rows: usize, num_slots: usize, gamma_root: Fr) -> Self {
        let mut gamma = Vec::with_capacity(2 * LIMBS * num_slots);
        let mut power = Fr::one();
        for _ in 0..2 * LIMBS * num_slots {
            gamma.push(power);
            power *= gamma_root;
        }
        Self { log_rows, gamma }
    }

    pub fn num_slots(&self) -> usize {
        self.gamma.len() / (2 * LIMBS)
    }

    /// The linking sumcheck's input claim from the row sumcheck's operand claims.
    pub fn claim(&self, claims: &RowClaims) -> Fr {
        assert_eq!(claims.operand_limbs.len(), self.gamma.len());
        self.gamma
            .iter()
            .zip(&claims.operand_limbs)
            .fold(Fr::zero(), |acc, (g, v)| acc + *g * *v)
    }

    pub fn slot_weights(&self, z_limbs: &[Fr; LIMBS]) -> SlotWeights {
        let weights = (0..self.num_slots())
            .map(|s| {
                let g = &self.gamma[2 * LIMBS * s..2 * LIMBS * (s + 1)];
                let side = |offset: usize| {
                    (0..LIMBS).fold(Fr::zero(), |acc, a| acc + g[offset + a] * z_limbs[a])
                };
                [side(0), side(LIMBS)]
            })
            .collect();
        SlotWeights { weights }
    }

    /// The fixed pieces' contribution `Σ_pieces K_piece(r, r')·G_{s,side}`
    /// (little-endian points); the selected pieces' contribution comes from
    /// the operand-selection sumcheck.
    pub fn fixed_value(
        &self,
        pieces: &[Piece],
        r_le: &[Fr],
        r_src_le: &[Fr],
        weights: &SlotWeights,
    ) -> Fr {
        pieces
            .par_iter()
            .map(|piece| piece.kernel.mle(r_le, r_src_le) * weights.get(piece.slot, piece.side))
            .sum()
    }
}

/// `Z_a = Σ_j 2^{16j}·chunk_{6a+j}` from the sixteen `z` chunks.
pub fn z_limbs(chunks: &[Fr], constants: &Constants) -> [Fr; LIMBS] {
    assert_eq!(chunks.len(), Z_CHUNKS);
    [
        recompose(&chunks[..6], constants),
        recompose(&chunks[6..12], constants),
        recompose(&chunks[12..], constants),
    ]
}

/// The linking sumcheck prover: dense `Wc_a` and `Z_a` columns over the row
/// index (every slot, fixed and selected, from the explicit rows), degree 2,
/// folded low bit first.
pub struct WiringSumcheck {
    wc: [Vec<Fr>; LIMBS],
    z: [Vec<Fr>; LIMBS],
    rows: usize,
    round: usize,
    log_rows: usize,
    pub cheat: bool,
}

impl WiringSumcheck {
    pub fn new(params: &LinkParams, program: &Program, columns: &Columns, r_be: &[Fr]) -> Self {
        let rows = columns.rows();
        assert_eq!(rows, 1 << params.log_rows);
        let eq_r = EqPolynomial::<Fr>::evals(r_be, None);
        let mut wc: [Vec<Fr>; LIMBS] = std::array::from_fn(|_| vec![Fr::zero(); rows]);
        for (spec, &weight) in program.rows.iter().zip(&eq_r) {
            for (s, slot) in spec.slots.iter().enumerate() {
                if slot.kappa == 0 {
                    continue;
                }
                let x = weight * fr_signed(slot.kappa);
                let gamma = &params.gamma[2 * LIMBS * s..2 * LIMBS * (s + 1)];
                for (a, column) in wc.iter_mut().enumerate() {
                    column[slot.x as usize] += gamma[a] * x;
                    column[slot.y as usize] += gamma[LIMBS + a] * weight;
                }
            }
        }
        let z: [Vec<Fr>; LIMBS] =
            std::array::from_fn(|a| (0..rows).map(|row| columns.limbs[row][a]).collect());
        Self {
            wc,
            z,
            rows,
            round: 0,
            log_rows: params.log_rows,
            cheat: false,
        }
    }

    /// `Σ_v Σ_a Wc_a(v)·Z_a(v)`, the honest input claim.
    pub fn input_claim(&self) -> Fr {
        (0..self.rows)
            .into_par_iter()
            .map(|i| (0..LIMBS).fold(Fr::zero(), |acc, a| acc + self.wc[a][i] * self.z[a][i]))
            .sum()
    }

    /// The bound `Z_a(r')` after every round (what the chunk openings at `r'` must recompose to).
    pub fn z_claims(&self) -> [Fr; LIMBS] {
        assert_eq!(self.rows, 1);
        std::array::from_fn(|a| self.z[a][0])
    }

    fn round_poly(&self, claim: Fr) -> Vec<Fr> {
        let half = self.rows / 2;
        let evals: [Fr; 3] = (0..half)
            .into_par_iter()
            .fold(
                || [Fr::zero(); 3],
                |mut acc, i| {
                    for a in 0..LIMBS {
                        let (w0, w1) = (self.wc[a][2 * i], self.wc[a][2 * i + 1]);
                        let (z0, z1) = (self.z[a][2 * i], self.z[a][2 * i + 1]);
                        acc[0] += w0 * z0;
                        acc[1] += w1 * z1;
                        let w2 = w1 + w1 - w0;
                        let z2 = z1 + z1 - z0;
                        acc[2] += w2 * z2;
                    }
                    acc
                },
            )
            .reduce(
                || [Fr::zero(); 3],
                |a, b| std::array::from_fn(|i| a[i] + b[i]),
            );
        let mut coefficients = UnivariatePoly::from_evals(&evals).into_coefficients();
        if self.cheat {
            let tail: Fr = coefficients[1..].iter().fold(Fr::zero(), |acc, c| acc + *c);
            coefficients[0] = (claim - tail)
                * Fr::from_u64(2)
                    .inverse()
                    .unwrap_or_else(|| unreachable!("2 is invertible"));
        }
        coefficients
    }

    fn bind(&mut self, r: Fr) {
        let half = self.rows / 2;
        for column in self.wc.iter_mut().chain(self.z.iter_mut()) {
            for i in 0..half {
                column[i] = column[2 * i] + r * (column[2 * i + 1] - column[2 * i]);
            }
            column.truncate(half);
        }
        self.rows = half;
        self.round += 1;
    }
}

impl ProveRounds<Fr> for WiringSumcheck {
    fn num_rounds(&self) -> usize {
        self.log_rows
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        debug_assert_eq!(round, self.round);
        Ok(UnivariatePoly::new(self.round_poly(previous_claim)))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind(bind);
        Ok(())
    }
}

/// `signed` re-export for callers that scale native values by a slot's `κ`.
pub fn kappa_fq(k: i32) -> ark_bn254::Fq {
    signed(k)
}
