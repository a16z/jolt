//! The digit-link member (stage A, no extra rounds): `Σ_x ω(x)·D(x)` with the
//! public weight `ω(x) = ρ^{kd}/mult·16^{63−w}` on every op's first slotted
//! row and the committed digit value column `D` (bound to the digit bits by
//! the row member), so the sum equals `Σ_kd ρ^{kd}·s_kd` — the R lane's
//! `Σ_s ρ^s·scalar_s` plus `ρ^K` for the constant-one base.

use jolt_field::{Field, Fr, Ring, Zero};
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
    pub cheat: bool,
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
            cheat: false,
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

    fn round_poly(&self, claim: Fr) -> Vec<Fr> {
        let half = self.size / 2;
        let evals: [Fr; 3] = (0..half)
            .into_par_iter()
            .fold(
                || [Fr::zero(); 3],
                |mut acc, i| {
                    let (o0, o1) = (self.omega[2 * i], self.omega[2 * i + 1]);
                    let (d0, d1) = (self.digit[2 * i], self.digit[2 * i + 1]);
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
        let mut coefficients = UnivariatePoly::from_evals(&evals).into_coefficients();
        if self.cheat {
            let tail: Fr = coefficients[1..].iter().fold(Fr::zero(), |acc, c| acc + *c);
            coefficients[0] = (claim - tail) * two_inverse();
        }
        coefficients
    }

    fn bind(&mut self, r: Fr) {
        let half = self.size / 2;
        let fold = |column: &mut Vec<Fr>| {
            for i in 0..half {
                column[i] = column[2 * i] + r * (column[2 * i + 1] - column[2 * i]);
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

fn two_inverse() -> Fr {
    Field::inverse(&Fr::from_u64(2)).unwrap_or_else(|| unreachable!("2 is invertible"))
}

impl ProveRounds<Fr> for LinkMember {
    fn num_rounds(&self) -> usize {
        LOG_ROWS
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

/// The member's final relation as one linear term: `ω̃(r)·D(r)`.
pub fn link_term(omega: Fr) -> Term {
    Term::new(omega, vec![AffineForm::column(ColumnId(Col::D as u32))])
}
