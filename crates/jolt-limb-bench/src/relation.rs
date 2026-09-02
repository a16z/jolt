//! The row relation and its sumcheck: for every row,
//! `eq(τ,row)·Φ(row) + λ·(Σ_j h_j(row) − m(row)·inv(row))` sums to zero, where
//! Φ batches the native mod-r identity, the three limb-carry identities and
//! the LogUp range terms `h_j·(α − chunk_j) − 1` with powers of γ.

use jolt_field::{Fr, One, Ring, Zero};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_transcript::Transcript;
use rayon::prelude::*;

use crate::table::{
    recompose, Constants, Table, CARRIES, CARRY_CHUNKS, CHUNK_BITS, CHUNK_COLUMNS, K_CHUNKS, LIMBS,
    OPERAND_CHUNKS, Z_CHUNKS,
};

pub const TABLE_LOG: usize = 16;
const DEGREE: usize = 3;

pub struct Public {
    pub log_rows: usize,
    pub t: usize,
    pub num_chunk_columns: usize,
    pub commit_operands: bool,
    pub alpha: Fr,
    pub tau: Vec<Fr>,
    pub gamma: Vec<Fr>,
    pub lambda: Fr,
    pub constants: Constants,
}

impl Public {
    pub fn draw<T: Transcript<Challenge = Fr>>(
        transcript: &mut T,
        log_rows: usize,
        t: usize,
        num_chunk_columns: usize,
        alpha: Fr,
        commit_operands: bool,
    ) -> Self {
        let tau = (0..log_rows).map(|_| transcript.challenge()).collect();
        let gamma_root: Fr = transcript.challenge();
        let lambda = transcript.challenge();
        let num_terms =
            1 + CARRIES + num_chunk_columns + if commit_operands { 2 * LIMBS * t } else { 0 };
        let mut gamma = Vec::with_capacity(num_terms);
        let mut acc = Fr::one();
        for _ in 0..num_terms {
            gamma.push(acc);
            acc *= gamma_root;
        }
        Self {
            log_rows,
            t,
            num_chunk_columns,
            commit_operands,
            alpha,
            tau,
            gamma,
            lambda,
            constants: Constants::new(),
        }
    }

    /// Public range-table vector `1/(α − v)` for `v < 2^16`.
    fn inverse_table(&self) -> Vec<Fr> {
        let mut values: Vec<ark_bn254::Fr> = (0..1u64 << CHUNK_BITS)
            .map(|v| ark_bn254::Fr::from(self.alpha - Fr::from_u64(v)))
            .collect();
        ark_ff::batch_inversion(&mut values);
        values.into_iter().map(Fr::from).collect()
    }

    /// Φ at one row (or one bound point): `chunks` are the chunk column
    /// values, `inverses` the LogUp inverses, `operands` the 6t operand limbs.
    fn phi(&self, chunks: &[Fr], inverses: &[Fr], operands: &[Fr]) -> Fr {
        let c = &self.constants;
        let limb_of = |start: usize, count: usize| recompose(&chunks[start..start + count], c);
        let z = [limb_of(0, 6), limb_of(6, 6), limb_of(12, Z_CHUNKS - 12)];
        let k = [
            limb_of(Z_CHUNKS, 6),
            limb_of(Z_CHUNKS + 6, 6),
            limb_of(Z_CHUNKS + 12, K_CHUNKS - 12),
        ];
        let carry: [Fr; CARRIES] = std::array::from_fn(|i| {
            limb_of(Z_CHUNKS + K_CHUNKS + CARRY_CHUNKS * i, CARRY_CHUNKS) - c.carry_offset
        });
        let full =
            |limbs: &[Fr; LIMBS]| limbs[0] + limbs[1] * c.pow_limb[1] + limbs[2] * c.pow_limb[2];

        let mut native = -(full(&k) * c.q) - full(&z);
        let mut positions = [Fr::zero(); CARRIES];
        for i in 0..self.t {
            let x = &operands[2 * LIMBS * i..2 * LIMBS * i + LIMBS];
            let y = &operands[2 * LIMBS * i + LIMBS..2 * LIMBS * (i + 1)];
            let x_full = x[0] + x[1] * c.pow_limb[1] + x[2] * c.pow_limb[2];
            let y_full = y[0] + y[1] * c.pow_limb[1] + y[2] * c.pow_limb[2];
            native += x_full * y_full;
            positions[0] += x[0] * y[0];
            positions[1] += x[0] * y[1] + x[1] * y[0];
            positions[2] += x[0] * y[2] + x[1] * y[1] + x[2] * y[0];
        }
        positions[0] -= k[0] * c.q_limbs[0] + z[0];
        positions[1] -= k[0] * c.q_limbs[1] + k[1] * c.q_limbs[0] + z[1];
        positions[2] -= k[0] * c.q_limbs[2] + k[1] * c.q_limbs[1] + k[2] * c.q_limbs[0] + z[2];

        let mut phi = self.gamma[0] * native;
        let mut carry_in = Fr::zero();
        for i in 0..CARRIES {
            phi += self.gamma[1 + i] * (positions[i] + carry_in - carry[i] * c.pow_limb[1]);
            carry_in = carry[i];
        }
        let mut gammas = self.gamma[1 + CARRIES..].iter();
        for (&h, &chunk) in inverses.iter().zip(chunks) {
            phi += *gammas.next().unwrap() * (h * (self.alpha - chunk) - Fr::one());
        }
        if self.commit_operands {
            for (i, limb) in operands.iter().enumerate() {
                let operand = i / LIMBS;
                let a = i % LIMBS;
                let start = CHUNK_COLUMNS + operand * OPERAND_CHUNKS + 6 * a;
                let count = if a == LIMBS - 1 {
                    OPERAND_CHUNKS - 12
                } else {
                    6
                };
                phi += *gammas.next().unwrap() * (limb_of(start, count) - *limb);
            }
        }
        phi
    }

    /// Full summand at one row given every polynomial's value there.
    fn summand(&self, values: &Layout<'_>) -> Fr {
        let phi = self.phi(values.chunks, values.inverses, values.operands);
        let logup: Fr = values.inverses.iter().fold(Fr::zero(), |acc, h| acc + *h)
            - values.multiplicity * values.inverse_table;
        values.eq * phi + self.lambda * logup
    }
}

/// Borrowed view of one row in the row-major matrix, entry order:
/// chunks (C), inverses (C), multiplicity (1), operands (6t), eq (1), inverse table (1).
struct Layout<'a> {
    chunks: &'a [Fr],
    inverses: &'a [Fr],
    multiplicity: Fr,
    operands: &'a [Fr],
    eq: Fr,
    inverse_table: Fr,
}

fn layout(values: &[Fr], c: usize, t: usize) -> Layout<'_> {
    let operands_end = 2 * c + 1 + 2 * LIMBS * t;
    Layout {
        chunks: &values[..c],
        inverses: &values[c..2 * c],
        multiplicity: values[2 * c],
        operands: &values[2 * c + 1..operands_end],
        eq: values[operands_end],
        inverse_table: values[operands_end + 1],
    }
}

pub struct Claims {
    /// Evaluations of the committed columns: chunks, inverses, multiplicity.
    pub committed: Vec<Fr>,
    /// Evaluations of the operand limb polynomials (supplied by the wiring lane in the real design).
    pub operand_limbs: Vec<Fr>,
}

pub struct Prover {
    table: Table,
    inverses: Vec<Vec<Fr>>,
    multiplicities: Vec<u32>,
    /// Row-major working values (`rows × width`), bound in place round by round.
    matrix: Vec<Fr>,
    scratch: Vec<Fr>,
    width: usize,
    num_chunk_columns: usize,
    t: usize,
}

impl Prover {
    pub fn new(
        mut table: Table,
        inverses: Vec<Vec<Fr>>,
        multiplicities: Vec<u32>,
        public: &Public,
    ) -> Self {
        let rows = table.rows;
        let (c, t) = (table.chunks.len(), table.t);
        let operands = 2 * LIMBS * t;
        let width = 2 * c + 1 + operands + 2;
        let eq = EqPolynomial::<Fr>::evals(&public.tau, None);
        let inverse_table = public.inverse_table();
        let mut matrix = vec![Fr::zero(); rows * width];
        matrix
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, slot)| {
                for (j, value) in slot[..c].iter_mut().enumerate() {
                    *value = table.chunk(j, row);
                }
                for (value, column) in slot[c..2 * c].iter_mut().zip(&inverses) {
                    *value = column[row];
                }
                slot[2 * c] = multiplicities
                    .get(row)
                    .map_or(Fr::zero(), |&m| Fr::from_u64(u64::from(m)));
                slot[2 * c + 1..2 * c + 1 + operands]
                    .copy_from_slice(&table.operand_rows[row * operands..(row + 1) * operands]);
                slot[2 * c + 1 + operands] = eq[row];
                slot[2 * c + 2 + operands] = inverse_table.get(row).copied().unwrap_or(Fr::zero());
            });
        table.operand_rows = Vec::new();
        Self {
            table,
            inverses,
            multiplicities,
            matrix,
            scratch: Vec::new(),
            width,
            num_chunk_columns: c,
            t,
        }
    }

    pub fn num_committed(&self) -> usize {
        2 * self.num_chunk_columns + 1
    }

    /// RLC of the committed columns (chunks, inverses, multiplicity) with `weights`.
    pub fn rlc(&self, weights: &[Fr]) -> Vec<Fr> {
        let c = self.num_chunk_columns;
        (0..self.table.rows)
            .into_par_iter()
            .map(|row| {
                let mut acc = Fr::zero();
                for (j, weight) in weights[..c].iter().enumerate() {
                    acc += *weight * self.table.chunk(j, row);
                }
                for (weight, column) in weights[c..2 * c].iter().zip(&self.inverses) {
                    acc += *weight * column[row];
                }
                if let Some(&m) = self.multiplicities.get(row) {
                    acc += weights[2 * c] * Fr::from_u64(u64::from(m));
                }
                acc
            })
            .collect()
    }

    /// Runs the sumcheck; returns the round polynomials (coefficients), the
    /// challenge point (round order) and the final claims.
    pub fn prove<T: Transcript<Challenge = Fr>>(
        &mut self,
        public: &Public,
        transcript: &mut T,
        self_check: bool,
    ) -> (Vec<Vec<Fr>>, Vec<Fr>, Claims) {
        let width = self.width;
        let (c, t) = (self.num_chunk_columns, self.t);
        let mut rows = self.table.rows;
        if self_check {
            let total: Fr = self
                .matrix
                .par_chunks(width)
                .map(|row| public.summand(&layout(row, c, t)))
                .sum();
            assert_eq!(
                total,
                Fr::zero(),
                "honest witness must satisfy the relation"
            );
        }
        let two = Fr::from_u64(2);
        let three = Fr::from_u64(3);
        let mut claim = Fr::zero();
        let mut round_polys = Vec::with_capacity(public.log_rows);
        let mut point = Vec::with_capacity(public.log_rows);
        while rows > 1 {
            let half = rows / 2;
            // Round polynomial at t = 0, 2, 3 (t = 1 follows from the running claim).
            let sums = self.matrix[..rows * width]
                .par_chunks(2 * width)
                .fold(
                    || ([Fr::zero(); DEGREE], vec![Fr::zero(); width]),
                    |(mut acc, mut scratch), pair| {
                        let (even, odd) = pair.split_at(width);
                        acc[0] += public.summand(&layout(even, c, t));
                        for (k, tt) in [two, three].into_iter().enumerate() {
                            for ((slot, &e), &o) in scratch.iter_mut().zip(even).zip(odd) {
                                *slot = e + tt * (o - e);
                            }
                            acc[1 + k] += public.summand(&layout(&scratch, c, t));
                        }
                        (acc, scratch)
                    },
                )
                .map(|(acc, _)| acc)
                .reduce(
                    || [Fr::zero(); DEGREE],
                    |a, b| std::array::from_fn(|i| a[i] + b[i]),
                );
            let evals = [sums[0], claim - sums[0], sums[1], sums[2]];
            let poly = UnivariatePoly::from_evals(&evals);
            for coefficient in poly.coefficients() {
                transcript.append(coefficient);
            }
            let r: Fr = transcript.challenge();
            claim = poly.evaluate(r);
            round_polys.push(poly.coefficients().to_vec());
            point.push(r);

            // Bind into the ping-pong buffer, then swap.
            if self.scratch.len() < half * width {
                self.scratch.resize(half * width, Fr::zero());
            }
            self.scratch[..half * width]
                .par_chunks_mut(width)
                .zip(self.matrix[..rows * width].par_chunks(2 * width))
                .for_each(|(out, pair)| {
                    let (even, odd) = pair.split_at(width);
                    for ((slot, &e), &o) in out.iter_mut().zip(even).zip(odd) {
                        *slot = e + r * (o - e);
                    }
                });
            std::mem::swap(&mut self.matrix, &mut self.scratch);
            rows = half;
        }
        let finals = &self.matrix[..width];
        let claims = Claims {
            committed: finals[..=2 * c].to_vec(),
            operand_limbs: finals[2 * c + 1..2 * c + 1 + 2 * LIMBS * t].to_vec(),
        };
        (round_polys, point, claims)
    }
}

impl Public {
    /// Replays the sumcheck and checks the final claim against Φ at the point;
    /// returns the point (round order) for the opening.
    pub fn verify<T: Transcript<Challenge = Fr>>(
        &self,
        round_polys: &[Vec<Fr>],
        claims: &Claims,
        transcript: &mut T,
    ) -> Result<Vec<Fr>, &'static str> {
        let public = self;
        if round_polys.len() != public.log_rows {
            return Err("round count");
        }
        let mut claim = Fr::zero();
        let mut point = Vec::with_capacity(public.log_rows);
        for coefficients in round_polys {
            if coefficients.len() > DEGREE + 1 {
                return Err("degree");
            }
            let poly = UnivariatePoly::new(coefficients.clone());
            if poly.evaluate(Fr::zero()) + poly.evaluate(Fr::one()) != claim {
                return Err("round sum");
            }
            for coefficient in coefficients {
                transcript.append(coefficient);
            }
            let r: Fr = transcript.challenge();
            claim = poly.evaluate(r);
            point.push(r);
        }
        let c = public.num_chunk_columns;
        if claims.committed.len() != 2 * c + 1 || claims.operand_limbs.len() != 2 * LIMBS * public.t
        {
            return Err("claim count");
        }
        // Variables were bound low-to-high; eq tables are big-endian in tau/point.
        let point_be: Vec<Fr> = point.iter().rev().copied().collect();
        let eq = EqPolynomial::<Fr>::mle(&public.tau, &point_be);
        let eq_point = EqPolynomial::<Fr>::evals(&point_be, None);
        let inverse_table = public
            .inverse_table()
            .iter()
            .zip(&eq_point)
            .fold(Fr::zero(), |acc, (inv, weight)| acc + *inv * *weight);
        let phi = public.phi(
            &claims.committed[..c],
            &claims.committed[c..2 * c],
            &claims.operand_limbs,
        );
        let logup = claims.committed[c..2 * c]
            .iter()
            .fold(Fr::zero(), |acc, h| acc + *h)
            - claims.committed[2 * c] * inverse_table;
        if eq * phi + public.lambda * logup != claim {
            return Err("final claim");
        }
        Ok(point)
    }
}
