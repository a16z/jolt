//! The digit-link member (stage A, no extra rounds): `Σ_x ω(x)·D(x)` with the
//! public weight `ω(x) = ρ^{kd}/mult·16^{63−w}` on every op's first slotted
//! row and the committed digit value column `D` (bound to the digit bits by
//! the row member), so the sum equals `Σ_kd ρ^{kd}·s_kd` — the R lane's
//! `Σ_s ρ^s·scalar_s` plus `ρ^K` for the constant-one base.

use jolt_field::{Fr, Zero};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use rayon::prelude::*;

use super::layout::LOG_ROWS;
use super::relation::Col;
use super::terms::{AffineForm, ColumnId, Term};

pub struct LinkMember {
    omega: Vec<Fr>,
    digit: Vec<Fr>,
    size: usize,
    round: usize,
}

impl LinkMember {
    pub fn new(omega: Vec<Fr>, digit_values: &[Fr]) -> Self {
        let size = 1usize << LOG_ROWS;
        assert_eq!(omega.len(), size);
        assert_eq!(digit_values.len(), size);
        Self {
            omega,
            digit: digit_values.to_vec(),
            size,
            round: 0,
        }
    }

    /// `Σ_x ω(x)·d(x)`.
    pub fn input_claim(&self) -> Fr {
        (0..self.size)
            .into_par_iter()
            .map(|i| self.omega[i] * self.digit[i])
            .sum()
    }

    pub const fn degree() -> usize {
        2
    }

    fn round_poly(&self) -> Vec<Fr> {
        let half = self.size / 2;
        let evals: [Fr; 3] = (0..half)
            .into_par_iter()
            .fold(
                || [Fr::zero(); 3],
                |mut acc, i| {
                    let (o0, o1) = (self.omega[i], self.omega[i + half]);
                    let (d0, d1) = (self.digit[i], self.digit[i + half]);
                    acc[0] += o0 * d0;
                    acc[1] += o1 * d1;
                    acc[2] += (o1 + o1 - o0) * (d1 + d1 - d0);
                    acc
                },
            )
            .reduce(
                || [Fr::zero(); 3],
                |a, b| std::array::from_fn(|i| a[i] + b[i]),
            );
        UnivariatePoly::from_evals(&evals).into_coefficients()
    }

    /// Most significant remaining row bit first: row `i` pairs with `i + half`.
    fn bind(&mut self, r: Fr) {
        let half = self.size / 2;
        let fold = |column: &mut Vec<Fr>| {
            let (lo, hi) = column.split_at_mut(half);
            for (l, &h) in lo.iter_mut().zip(hi.iter()) {
                *l += r * (h - *l);
            }
            column.truncate(half);
        };
        fold(&mut self.omega);
        fold(&mut self.digit);
        self.size = half;
        self.round += 1;
    }

    /// The bound digit value (the claimed `D` column evaluation) and `ω̃`.
    pub fn final_values(&self) -> (Fr, Fr) {
        assert_eq!(self.size, 1);
        (self.digit[0], self.omega[0])
    }
}

impl ProveRounds<Fr> for LinkMember {
    fn num_rounds(&self) -> usize {
        LOG_ROWS
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        _previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        debug_assert_eq!(round, self.round);
        Ok(UnivariatePoly::new(self.round_poly()))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind(bind);
        Ok(())
    }
}

/// The member's final relation as one linear term: `ω̃(r)·D(r)`.
pub fn link_term(omega: Fr) -> Term {
    Term::new(omega, vec![AffineForm::column(ColumnId(Col::D as u32))])
}
