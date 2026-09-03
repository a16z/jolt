//! The digit-link member (stage A, `LOG_ROWS` rounds, degree 2):
//! `Σ_x ω(x)·D(x) + κ(x)·V(x) + κ'(x)·V'(x)` over the rows, public weights
//! times committed columns ([`link_columns`]).
//!
//! On every op's first slotted row `ω(x) = ρ^o·16^{63−w}` (chain-base
//! occurrence `o`, window `w`), so `Σ ω·D = Σ_o ρ^o·t_o` with `t_o` the
//! occurrence's recoded integer — one equation per occurrence against R's
//! `Σ_k W_k(ρ)·s_k` (`W_k = Σ_{o of k} ρ^o`, [`super::lookup::link_weights`])
//! plus the constant-one and offset bases ([`super::stream::link_input_claim`]).
//!
//! The top 16 windows also carry `ρ^{M+o}·16^{15−w}` (`M` the occurrence
//! count), and the window row of `o` (`Cells::WINDOW`) carries `κ =
//! ρ^o·(ρ^{M+256} − ρ^M)` on `V = Σ_{j<4} 2^{16j}·chunk_j` and `κ' =
//! ρ^{M+256+o}` on `V' = Σ_{j<4} 2^{16j}·chunk_{4+j}`, so the sum also
//! contains `Σ_o ρ^{M+o}·(V_hi(o) − V(o)) + Σ_{o<256} ρ^{M+256+o}·(V(o) +
//! V'(o) − WINDOW_BOUND)` (the constant is part of the input claim). With
//! the chunks range-checked, this pins every occurrence's top window `V_hi`
//! to `0..=WINDOW_BOUND`, which admits exactly one recoding per scalar
//! ([`super::digits::WINDOW_BOUND`]). All coefficients are distinct powers
//! of `ρ`, so a false batched identity has probability below `(M + 512)/r`.

use jolt_field::{Fr, Ring, Zero};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use rayon::prelude::*;

use super::columns::Constants;
use super::layout::LOG_ROWS;
use super::lookup::{link_columns, LinkColumns, LinkEvals};
use super::relation::Col;
use super::schedule::Layout;
use super::terms::{AffineForm, ColumnId, Term};

/// Chunk columns of the window value `V` (`V'` follows).
const WINDOW_CHUNKS: usize = 4;

pub struct LinkMember {
    weights: LinkColumns,
    digit: Vec<Fr>,
    v: Vec<Fr>,
    v_prime: Vec<Fr>,
    size: usize,
    round: usize,
}

/// The member's final values: the bound committed columns and public weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinkFinals {
    pub digit: Fr,
    pub v: Fr,
    pub v_prime: Fr,
    pub evals: LinkEvals,
}

impl LinkMember {
    /// `digit_values` is the committed `D` column, `chunks[j]` chunk column
    /// `j` (the first `2·WINDOW_CHUNKS` are read).
    pub fn new(layout: &Layout, rho: Fr, digit_values: &[Fr], chunks: &[Vec<Fr>]) -> Self {
        let size = 1usize << LOG_ROWS;
        assert_eq!(digit_values.len(), size);
        let pow_chunk = Constants::get().pow_chunk;
        let pack = |offset: usize| -> Vec<Fr> {
            (0..size)
                .into_par_iter()
                .map(|x| {
                    (0..WINDOW_CHUNKS).fold(Fr::zero(), |acc, j| {
                        acc + chunks[offset + j][x] * pow_chunk[j]
                    })
                })
                .collect()
        };
        Self {
            weights: link_columns(layout, rho),
            digit: digit_values.to_vec(),
            v: pack(0),
            v_prime: pack(WINDOW_CHUNKS),
            size,
            round: 0,
        }
    }

    /// `Σ_x ω·D + κ·V + κ'·V'`.
    pub fn input_claim(&self) -> Fr {
        (0..self.size)
            .into_par_iter()
            .map(|i| self.summand(i))
            .sum()
    }

    fn summand(&self, i: usize) -> Fr {
        self.weights.omega[i] * self.digit[i]
            + self.weights.kappa[i] * self.v[i]
            + self.weights.kappa_prime[i] * self.v_prime[i]
    }

    pub const fn degree() -> usize {
        2
    }

    /// Most significant remaining row bit first: row `i` pairs with `i + half`.
    fn round_poly(&self) -> Vec<Fr> {
        let half = self.size / 2;
        let at = |i: usize, x: Fr| -> Fr {
            // The summand at `X = x` on the pair `(i, i + half)`: each column
            // extrapolated as `lo + x·(hi − lo)`.
            let ext = |column: &[Fr]| column[i] + x * (column[i + half] - column[i]);
            ext(&self.weights.omega) * ext(&self.digit)
                + ext(&self.weights.kappa) * ext(&self.v)
                + ext(&self.weights.kappa_prime) * ext(&self.v_prime)
        };
        let evals: [Fr; 3] = (0..half)
            .into_par_iter()
            .fold(
                || [Fr::zero(); 3],
                |mut acc, i| {
                    acc[0] += self.summand(i);
                    acc[1] += self.summand(i + half);
                    acc[2] += at(i, Fr::from_u64(2));
                    acc
                },
            )
            .reduce(
                || [Fr::zero(); 3],
                |a, b| std::array::from_fn(|i| a[i] + b[i]),
            );
        UnivariatePoly::from_evals(&evals).into_coefficients()
    }

    fn bind(&mut self, r: Fr) {
        let half = self.size / 2;
        let fold = |column: &mut Vec<Fr>| {
            let (lo, hi) = column.split_at_mut(half);
            for (l, &h) in lo.iter_mut().zip(hi.iter()) {
                *l += r * (h - *l);
            }
            column.truncate(half);
        };
        fold(&mut self.weights.omega);
        fold(&mut self.weights.kappa);
        fold(&mut self.weights.kappa_prime);
        fold(&mut self.digit);
        fold(&mut self.v);
        fold(&mut self.v_prime);
        self.size = half;
        self.round += 1;
    }

    pub fn final_values(&self) -> LinkFinals {
        assert_eq!(self.size, 1);
        LinkFinals {
            digit: self.digit[0],
            v: self.v[0],
            v_prime: self.v_prime[0],
            evals: LinkEvals {
                omega: self.weights.omega[0],
                kappa: self.weights.kappa[0],
                kappa_prime: self.weights.kappa_prime[0],
            },
        }
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

/// `Σ_{j<4} 2^{16j}·chunk_{offset+j}` as a form over the claimed columns.
fn window_form(offset: usize) -> AffineForm {
    let pow_chunk = Constants::get().pow_chunk;
    let mut form = AffineForm::default();
    for (j, weight) in pow_chunk.iter().take(WINDOW_CHUNKS).enumerate() {
        form.add_column(ColumnId((Col::CHUNKS + offset + j) as u32), *weight);
    }
    form
}

/// The member's final relation as terms: `ω̃·D + κ̃·V + κ̃'·V'`.
pub fn link_terms(evals: &LinkEvals) -> Vec<Term> {
    vec![
        Term::new(
            evals.omega,
            vec![AffineForm::column(ColumnId(Col::D as u32))],
        ),
        Term::new(evals.kappa, vec![window_form(0)]),
        Term::new(evals.kappa_prime, vec![window_form(WINDOW_CHUNKS)]),
    ]
}
