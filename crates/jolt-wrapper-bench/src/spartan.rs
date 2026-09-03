//! Synthetic Spartan members: dense columns and a per-row function, with
//! an optional `eq(τ,·)` factor (outer: `eq·(A·B − C)`, inner: `M·Z`).

use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::prelude::*;

use crate::sumcheck::{horner, Instance};

pub struct DenseInstance {
    cols: Vec<Vec<Fr>>,
    tau: Option<Vec<Fr>>,
    f: fn(&[Fr]) -> Fr,
    f_degree: usize,
    rounds: usize,
    round: usize,
    eq_prefix: Fr,
    claim: Fr,
    running: Fr,
    last: Vec<Fr>,
}

pub fn random_columns(count: usize, log_rows: usize, seed: u64) -> Vec<Vec<Fr>> {
    (0..count)
        .into_par_iter()
        .map(|c| {
            let mut rng = ChaCha20Rng::seed_from_u64(seed ^ (c as u64));
            (0..1usize << log_rows)
                .map(|_| Fr::random(&mut rng))
                .collect()
        })
        .collect()
}

pub fn outer(v: &[Fr]) -> Fr {
    v[0] * v[1] - v[2]
}

pub fn inner(v: &[Fr]) -> Fr {
    v[0] * v[1]
}

impl DenseInstance {
    pub fn new(
        cols: Vec<Vec<Fr>>,
        tau: Option<Vec<Fr>>,
        f: fn(&[Fr]) -> Fr,
        f_degree: usize,
    ) -> Self {
        let rows = cols[0].len();
        let eq = tau.as_ref().map(|tau| EqPolynomial::<Fr>::evals(tau, None));
        let claim = (0..rows)
            .into_par_iter()
            .map(|row| {
                let values: Vec<Fr> = cols.iter().map(|c| c[row]).collect();
                eq.as_ref().map_or(Fr::one(), |e| e[row]) * f(&values)
            })
            .sum();
        Self {
            rounds: rows.trailing_zeros() as usize,
            cols,
            tau,
            f,
            f_degree,
            round: 0,
            eq_prefix: Fr::one(),
            claim,
            running: claim,
            last: Vec::new(),
        }
    }

    pub fn finals(&self) -> Vec<Fr> {
        self.cols.iter().map(|c| c[0]).collect()
    }

    /// Expected final value from the column claims at `point_be`.
    pub fn expected(tau: Option<&[Fr]>, f: fn(&[Fr]) -> Fr, claims: &[Fr], point_be: &[Fr]) -> Fr {
        tau.map_or(Fr::one(), |tau| EqPolynomial::<Fr>::mle(tau, point_be)) * f(claims)
    }
}

impl Instance for DenseInstance {
    fn rounds(&self) -> usize {
        self.rounds
    }

    fn input_claim(&self) -> Fr {
        self.claim
    }

    fn round_poly(&mut self) -> Vec<Fr> {
        let n = self.rounds;
        let half = self.cols[0].len() / 2;
        let e = self
            .tau
            .as_ref()
            .map(|tau| EqPolynomial::<Fr>::evals(&tau[..n - 1 - self.round], None));
        let points = self.f_degree + 1;
        let evals: Vec<Fr> = (0..points)
            .map(|x| {
                let x = Fr::from_u64(x as u64);
                (0..half)
                    .into_par_iter()
                    .map(|i| {
                        let values: Vec<Fr> = self
                            .cols
                            .iter()
                            .map(|c| c[2 * i] + x * (c[2 * i + 1] - c[2 * i]))
                            .collect();
                        e.as_ref().map_or(Fr::one(), |e| e[i]) * (self.f)(&values)
                    })
                    .sum()
            })
            .collect();
        let t = UnivariatePoly::from_evals(&evals).into_coefficients();
        let coefficients = match &self.tau {
            None => t,
            Some(tau) => {
                let tau_v = tau[n - 1 - self.round];
                let a = self.eq_prefix * (Fr::one() - tau_v);
                let b = self.eq_prefix * (tau_v + tau_v - Fr::one());
                let mut out = vec![Fr::zero(); t.len() + 1];
                for (i, &ti) in t.iter().enumerate() {
                    out[i] += a * ti;
                    out[i + 1] += b * ti;
                }
                out
            }
        };
        self.last.clone_from(&coefficients);
        coefficients
    }

    fn bind(&mut self, r: Fr) {
        let n = self.rounds;
        for c in &mut self.cols {
            let half = c.len() / 2;
            for i in 0..half {
                c[i] = c[2 * i] + r * (c[2 * i + 1] - c[2 * i]);
            }
            c.truncate(half);
        }
        if let Some(tau) = &self.tau {
            let tau_v = tau[n - 1 - self.round];
            self.eq_prefix *= (Fr::one() - tau_v) * (Fr::one() - r) + tau_v * r;
        }
        self.running = horner(&self.last, r);
        self.round += 1;
    }
}
