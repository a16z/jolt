//! The row relation and its sumcheck: for every row,
//! `eq(τ,row)·Φ(row) + λ·(Σ_g h_g(row)·e_{s−1,g}(row) − m(row)·inv(row))` sums to
//! zero, where Φ batches the native mod-r identity, the three limb-carry
//! identities and the grouped LogUp range terms `h_g·Π_{i∈g}(α − chunk_i) − 1`
//! (one helper per `s` chunk columns) with powers of γ. Prover uses split-eq
//! (the eq factor is applied to the round polynomial, not evaluated per row)
//! and unreduced product accumulation for the limb sums.

use jolt_field::{Accumulator, Field, Fr, One, Ring, WithAccumulator, Zero};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_transcript::Transcript;
use rayon::prelude::*;

use crate::table::{
    recompose, Constants, Table, CARRIES, CARRY_CHUNKS, CHUNK_BITS, CHUNK_COLUMNS, K_CHUNKS, LIMBS,
    OPERAND_CHUNKS, Z_CHUNKS,
};

pub const TABLE_LOG: usize = 16;

type Acc = <Fr as WithAccumulator>::Accumulator;

pub struct Public {
    pub log_rows: usize,
    pub t: usize,
    pub num_chunk_columns: usize,
    pub group_size: usize,
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
        group_size: usize,
        alpha: Fr,
        commit_operands: bool,
    ) -> Self {
        let tau = (0..log_rows).map(|_| transcript.challenge()).collect();
        let gamma_root: Fr = transcript.challenge();
        let lambda = transcript.challenge();
        let num_groups = num_chunk_columns.div_ceil(group_size);
        let num_terms = 1 + CARRIES + num_groups + if commit_operands { 2 * LIMBS * t } else { 0 };
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
            group_size,
            commit_operands,
            alpha,
            tau,
            gamma,
            lambda,
            constants: Constants::new(),
        }
    }

    pub fn num_groups(&self) -> usize {
        self.num_chunk_columns.div_ceil(self.group_size)
    }

    /// Round-polynomial degree: `eq · h · Π_{s}(α − c)`.
    pub fn degree(&self) -> usize {
        (self.group_size + 2).max(3)
    }

    /// Public range-table vector `1/(α − v)` for `v < 2^16`.
    fn inverse_table(&self) -> Vec<Fr> {
        let mut values: Vec<ark_bn254::Fr> = (0..1u64 << CHUNK_BITS)
            .map(|v| ark_bn254::Fr::from(self.alpha - Fr::from_u64(v)))
            .collect();
        ark_ff::batch_inversion(&mut values);
        values.into_iter().map(Fr::from).collect()
    }

    /// Degree-2 part of Φ: native identity, carry identities, operand consistency.
    fn phi_low(&self, chunks: &[Fr], operands: &[Fr]) -> Fr {
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

        let mut native = Acc::default();
        let mut positions = [Acc::default(); CARRIES];
        for i in 0..self.t {
            let x = &operands[2 * LIMBS * i..2 * LIMBS * i + LIMBS];
            let y = &operands[2 * LIMBS * i + LIMBS..2 * LIMBS * (i + 1)];
            let x_full = x[0] + x[1] * c.pow_limb[1] + x[2] * c.pow_limb[2];
            let y_full = y[0] + y[1] * c.pow_limb[1] + y[2] * c.pow_limb[2];
            native.fmadd(x_full, y_full);
            positions[0].fmadd(x[0], y[0]);
            positions[1].fmadd(x[0], y[1]);
            positions[1].fmadd(x[1], y[0]);
            positions[2].fmadd(x[0], y[2]);
            positions[2].fmadd(x[1], y[1]);
            positions[2].fmadd(x[2], y[0]);
        }
        let native = native.reduce() - full(&k) * c.q - full(&z);
        let mut positions = positions.map(Acc::reduce);
        positions[0] -= k[0] * c.q_limbs[0] + z[0];
        positions[1] -= k[0] * c.q_limbs[1] + k[1] * c.q_limbs[0] + z[1];
        positions[2] -= k[0] * c.q_limbs[2] + k[1] * c.q_limbs[1] + k[2] * c.q_limbs[0] + z[2];

        let mut phi = self.gamma[0] * native;
        let mut carry_in = Fr::zero();
        for i in 0..CARRIES {
            phi += self.gamma[1 + i] * (positions[i] + carry_in - carry[i] * c.pow_limb[1]);
            carry_in = carry[i];
        }
        if self.commit_operands {
            let gammas = &self.gamma[1 + CARRIES + self.num_groups()..];
            for ((i, limb), gamma) in operands.iter().enumerate().zip(gammas) {
                let operand = i / LIMBS;
                let a = i % LIMBS;
                let start = CHUNK_COLUMNS + operand * OPERAND_CHUNKS + 6 * a;
                let count = if a == LIMBS - 1 {
                    OPERAND_CHUNKS - 12
                } else {
                    6
                };
                phi += *gamma * (limb_of(start, count) - *limb);
            }
        }
        phi
    }

    /// Grouped LogUp terms: returns `(Σ_g γ_g·(h_g·Π_{i∈g}(α − c_i) − 1), Σ_g h_g·e_{s−1,g})`,
    /// where `e_{s−1,g} = Σ_i Π_{j≠i}(α − c_j)` so that `h_g·e_{s−1,g} = Σ_{i∈g} 1/(α − c_i)`.
    fn range_and_logup(&self, chunks: &[Fr], helpers: &[Fr]) -> (Fr, Fr) {
        let s = self.group_size;
        let gammas = &self.gamma[1 + CARRIES..];
        let mut range = Fr::zero();
        let mut logup = Fr::zero();
        let mut prefix = [Fr::one(); 16];
        let mut suffix = [Fr::one(); 16];
        let mut factors = [Fr::zero(); 16];
        for ((group, &h), &gamma) in chunks.chunks(s).zip(helpers).zip(gammas) {
            let n = group.len();
            for (factor, &c) in factors.iter_mut().zip(group) {
                *factor = self.alpha - c;
            }
            // Prefix/suffix products give Π and e_{n−1} = Σ_i Π_{j≠i} in 3n mults.
            for i in 0..n {
                prefix[i + 1] = prefix[i] * factors[i];
            }
            suffix[n] = Fr::one();
            for i in (1..n).rev() {
                suffix[i] = suffix[i + 1] * factors[i];
            }
            let mut elementary = Fr::zero();
            for i in 0..n {
                elementary += prefix[i] * suffix[i + 1];
            }
            range += gamma * (h * prefix[n] - Fr::one());
            logup += h * elementary;
        }
        (range, logup)
    }

    /// Full summand at one row: `eq·(Φ_low + Φ_range) + λ·(Σ h·e − m·inv)`.
    fn summand(&self, eq: Fr, values: &Layout<'_>) -> Fr {
        let (range, logup) = self.range_and_logup(values.chunks, values.helpers);
        eq * (self.phi_low(values.chunks, values.operands) + range)
            + self.lambda * (logup - values.multiplicity * values.inverse_table)
    }

    /// The relation's value at the sumcheck's end: `eq(τ, point)·Φ(claims) + λ·L(claims)`
    /// with the public inverse table evaluated at `point_be` (big-endian).
    pub fn final_value(&self, claims: &Claims, point_be: &[Fr]) -> Result<Fr, &'static str> {
        let (c, g) = (self.num_chunk_columns, self.num_groups());
        if claims.committed.len() != c + g + 1 || claims.operand_limbs.len() != 2 * LIMBS * self.t {
            return Err("claim count");
        }
        let eq = EqPolynomial::<Fr>::mle(&self.tau, point_be);
        let eq_point = EqPolynomial::<Fr>::evals(point_be, None);
        let inverse_table = self
            .inverse_table()
            .iter()
            .zip(&eq_point)
            .fold(Fr::zero(), |acc, (inv, weight)| acc + *inv * *weight);
        let values = Layout {
            chunks: &claims.committed[..c],
            helpers: &claims.committed[c..c + g],
            multiplicity: claims.committed[c + g],
            operands: &claims.operand_limbs,
            inverse_table,
        };
        Ok(self.summand(eq, &values))
    }

    /// Replays the sumcheck and checks the final claim against the relation at
    /// the point; returns the point (round order) for the opening.
    pub fn verify<T: Transcript<Challenge = Fr>>(
        &self,
        round_polys: &[Vec<Fr>],
        claims: &Claims,
        transcript: &mut T,
    ) -> Result<Vec<Fr>, &'static str> {
        if round_polys.len() != self.log_rows {
            return Err("round count");
        }
        let mut claim = Fr::zero();
        let mut point = Vec::with_capacity(self.log_rows);
        for coefficients in round_polys {
            if coefficients.len() > self.degree() + 1 {
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
        // Variables were bound low-to-high; eq tables are big-endian in tau/point.
        let point_be: Vec<Fr> = point.iter().rev().copied().collect();
        if self.final_value(claims, &point_be)? != claim {
            return Err("final claim");
        }
        Ok(point)
    }
}

/// Borrowed view of one row in the row-major matrix, entry order:
/// chunks (C), helpers (G), multiplicity (1), operands (6t), inverse table (1).
struct Layout<'a> {
    chunks: &'a [Fr],
    helpers: &'a [Fr],
    multiplicity: Fr,
    operands: &'a [Fr],
    inverse_table: Fr,
}

fn layout(values: &[Fr], c: usize, g: usize, t: usize) -> Layout<'_> {
    let operands_end = c + g + 1 + 2 * LIMBS * t;
    Layout {
        chunks: &values[..c],
        helpers: &values[c..c + g],
        multiplicity: values[c + g],
        operands: &values[c + g + 1..operands_end],
        inverse_table: values[operands_end],
    }
}

pub struct Claims {
    /// Evaluations of the committed columns: chunks, helpers, multiplicity.
    pub committed: Vec<Fr>,
    /// Evaluations of the operand limb polynomials (supplied by the wiring lane in the real design).
    pub operand_limbs: Vec<Fr>,
}

pub struct Prover {
    table: Table,
    helpers: Vec<Vec<Fr>>,
    multiplicities: Vec<u32>,
    /// Row-major working values (`rows × width`), bound round by round.
    matrix: Vec<Fr>,
    scratch: Vec<Fr>,
    width: usize,
    num_chunk_columns: usize,
    num_groups: usize,
    t: usize,
    rows: usize,
    round: usize,
    eq_prefix: Fr,
    claim: Fr,
    /// Evaluation points `X = 0..s+2` for the degree-(s+1) range part.
    points: Vec<Fr>,
    /// Honest prover self-checks; a tampered run instead forces every round
    /// check to pass so that rejection happens at the final relation check.
    pub honest: bool,
}

/// Linear extrapolation of a row pair to `X = x`: `even + x·(odd − even)`.
fn extrapolate(out: &mut [Fr], even: &[Fr], odd: &[Fr], x: Fr) {
    for ((slot, &e), &o) in out.iter_mut().zip(even).zip(odd) {
        *slot = e + x * (o - e);
    }
}

fn add_coefficients(into: &mut Vec<Fr>, poly: &[Fr], scale: Fr) {
    if into.len() < poly.len() {
        into.resize(poly.len(), Fr::zero());
    }
    for (slot, &coefficient) in into.iter_mut().zip(poly) {
        *slot += scale * coefficient;
    }
}

impl Prover {
    pub fn new(
        mut table: Table,
        helpers: Vec<Vec<Fr>>,
        multiplicities: Vec<u32>,
        public: &Public,
    ) -> Self {
        let rows = table.rows;
        let (c, t) = (table.chunks.len(), table.t);
        let g = helpers.len();
        let operands = 2 * LIMBS * t;
        let width = c + g + 1 + operands + 1;
        let inverse_table = public.inverse_table();
        let mut matrix = vec![Fr::zero(); rows * width];
        matrix
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, slot)| {
                for (j, value) in slot[..c].iter_mut().enumerate() {
                    *value = table.chunk(j, row);
                }
                for (value, column) in slot[c..c + g].iter_mut().zip(&helpers) {
                    *value = column[row];
                }
                slot[c + g] = multiplicities
                    .get(row)
                    .map_or(Fr::zero(), |&m| Fr::from_u64(u64::from(m)));
                slot[c + g + 1..c + g + 1 + operands]
                    .copy_from_slice(&table.operand_rows[row * operands..(row + 1) * operands]);
                slot[c + g + 1 + operands] = inverse_table.get(row).copied().unwrap_or(Fr::zero());
            });
        table.operand_rows = Vec::new();
        let high_points = public.group_size + 2;
        Self {
            table,
            helpers,
            multiplicities,
            matrix,
            scratch: Vec::new(),
            width,
            num_chunk_columns: c,
            num_groups: g,
            t,
            rows,
            round: 0,
            eq_prefix: Fr::one(),
            claim: Fr::zero(),
            points: (0..high_points as u64).map(Fr::from_u64).collect(),
            honest: true,
        }
    }

    pub fn num_committed(&self) -> usize {
        self.num_chunk_columns + self.num_groups + 1
    }

    pub fn rows(&self) -> usize {
        self.table.rows
    }

    /// Committed column `(column, row)`: chunks, then helpers, then the multiplicity column.
    pub fn committed(&self, column: usize, row: usize) -> Fr {
        let (c, g) = (self.num_chunk_columns, self.num_groups);
        if column < c {
            self.table.chunk(column, row)
        } else if column < c + g {
            self.helpers[column - c][row]
        } else {
            self.multiplicities
                .get(row)
                .map_or(Fr::zero(), |&m| Fr::from_u64(u64::from(m)))
        }
    }

    /// RLC of the committed columns (chunks, helpers, multiplicity) with `weights`.
    pub fn rlc(&self, weights: &[Fr]) -> Vec<Fr> {
        (0..self.table.rows)
            .into_par_iter()
            .map(|row| {
                weights
                    .iter()
                    .enumerate()
                    .fold(Fr::zero(), |acc, (j, w)| acc + *w * self.committed(j, row))
            })
            .collect()
    }

    /// `Σ_row eq(τ,row)·summand(row)` — zero for an honest witness.
    pub fn input_claim(&self, public: &Public) -> Fr {
        let (c, g, t) = (self.num_chunk_columns, self.num_groups, self.t);
        let eq = EqPolynomial::<Fr>::evals(&public.tau, None);
        self.matrix
            .par_chunks(self.width)
            .zip(&eq)
            .map(|(row, &eq)| public.summand(eq, &layout(row, c, g, t)))
            .sum()
    }

    /// Coefficients of this round's polynomial
    /// `s(X) = eq_prefix·eq1(τ_round, X)·(q_low + q_high)(X) + λ·L(X)`.
    #[expect(clippy::unwrap_used, reason = "2 is invertible")]
    pub fn round_poly(&mut self, public: &Public) -> Vec<Fr> {
        let width = self.width;
        let (c, g, t) = (self.num_chunk_columns, self.num_groups, self.t);
        let n = public.log_rows;
        let s = public.group_size;
        let high_points = s + 2;
        let logup_points = s.max(2) + 1;
        let points = &self.points;
        let rows = self.rows;
        let round = self.round;
        let tau_round = public.tau[n - 1 - round];
        let eq_rest = if round + 1 == n {
            vec![Fr::one()]
        } else {
            EqPolynomial::<Fr>::evals(&public.tau[..n - 1 - round], None)
        };
        let sums = self.matrix[..rows * width]
            .par_chunks(2 * width)
            .zip(&eq_rest)
            .fold(
                || {
                    (
                        vec![Fr::zero(); 3 + 2 * high_points],
                        vec![Fr::zero(); width],
                        vec![Fr::zero(); c + g],
                    )
                },
                |(mut acc, mut full, mut partial), (pair, &eq)| {
                    let (even, odd) = pair.split_at(width);
                    let (q_low, rest) = acc.split_at_mut(3);
                    let (q_high, l) = rest.split_at_mut(high_points);
                    // Low part at X = 0, 1 (direct rows) and 2 (extrapolated).
                    let even_view = layout(even, c, g, t);
                    let odd_view = layout(odd, c, g, t);
                    q_low[0] += eq * public.phi_low(even_view.chunks, even_view.operands);
                    q_low[1] += eq * public.phi_low(odd_view.chunks, odd_view.operands);
                    extrapolate(&mut full, even, odd, points[2]);
                    let two_view = layout(&full, c, g, t);
                    q_low[2] += eq * public.phi_low(two_view.chunks, two_view.operands);
                    // Range/LogUp part at X = 0..high_points (chunks + helpers only).
                    let mut high = |x: usize, view: &Layout<'_>| {
                        let (range, logup) = public.range_and_logup(view.chunks, view.helpers);
                        q_high[x] += eq * range;
                        let m = even_view.multiplicity
                            + points[x] * (odd_view.multiplicity - even_view.multiplicity);
                        let inv = even_view.inverse_table
                            + points[x] * (odd_view.inverse_table - even_view.inverse_table);
                        l[x] += logup - m * inv;
                    };
                    high(0, &even_view);
                    high(1, &odd_view);
                    high(2, &two_view);
                    for (x, &point) in points.iter().enumerate().skip(3) {
                        extrapolate(&mut partial, &even[..c + g], &odd[..c + g], point);
                        let view = Layout {
                            chunks: &partial[..c],
                            helpers: &partial[c..c + g],
                            multiplicity: Fr::zero(),
                            operands: &[],
                            inverse_table: Fr::zero(),
                        };
                        high(x, &view);
                    }
                    (acc, full, partial)
                },
            )
            .map(|(acc, _, _)| acc)
            .reduce(
                || vec![Fr::zero(); 3 + 2 * high_points],
                |a, b| a.iter().zip(&b).map(|(x, y)| *x + *y).collect(),
            );
        let (q_low, rest) = sums.split_at(3);
        let (q_high, l) = rest.split_at(high_points);
        let mut q = UnivariatePoly::from_evals(q_low).into_coefficients();
        add_coefficients(
            &mut q,
            &UnivariatePoly::from_evals(q_high).into_coefficients(),
            Fr::one(),
        );
        let a = self.eq_prefix * (Fr::one() - tau_round);
        let b = self.eq_prefix * (tau_round + tau_round - Fr::one());
        let mut coefficients = vec![Fr::zero(); q.len() + 1];
        for (i, &qi) in q.iter().enumerate() {
            coefficients[i] += a * qi;
            coefficients[i + 1] += b * qi;
        }
        add_coefficients(
            &mut coefficients,
            &UnivariatePoly::from_evals(&l[..logup_points]).into_coefficients(),
            public.lambda,
        );
        while coefficients.len() > 2 && coefficients.last() == Some(&Fr::zero()) {
            let _ = coefficients.pop();
        }
        if !self.honest {
            let tail: Fr = coefficients[1..].iter().fold(Fr::zero(), |acc, c| acc + *c);
            coefficients[0] = (self.claim - tail) * Fr::from_u64(2).inverse().unwrap();
        }
        coefficients
    }

    /// Binds the current round's variable to `r` (the round polynomial's
    /// evaluation there becomes the running claim).
    pub fn bind(&mut self, public: &Public, coefficients: &[Fr], r: Fr) {
        let width = self.width;
        let rows = self.rows;
        let half = rows / 2;
        let tau_round = public.tau[public.log_rows - 1 - self.round];
        if self.scratch.len() < half * width {
            self.scratch.resize(half * width, Fr::zero());
        }
        self.scratch[..half * width]
            .par_chunks_mut(width)
            .zip(self.matrix[..rows * width].par_chunks(2 * width))
            .for_each(|(out, pair)| {
                let (even, odd) = pair.split_at(width);
                extrapolate(out, even, odd, r);
            });
        std::mem::swap(&mut self.matrix, &mut self.scratch);
        self.eq_prefix *= (Fr::one() - tau_round) * (Fr::one() - r) + tau_round * r;
        self.claim = UnivariatePoly::new(coefficients.to_vec()).evaluate(r);
        self.rows = half;
        self.round += 1;
    }

    /// Final claims after every round: committed columns and operand limbs.
    pub fn claims(&self) -> Claims {
        let (c, g, t) = (self.num_chunk_columns, self.num_groups, self.t);
        let finals = &self.matrix[..self.width];
        Claims {
            committed: finals[..=c + g].to_vec(),
            operand_limbs: finals[c + g + 1..c + g + 1 + 2 * LIMBS * t].to_vec(),
        }
    }

    /// Runs the sumcheck alone; returns the round polynomials (coefficients),
    /// the challenge point (round order) and the final claims.
    pub fn prove<T: Transcript<Challenge = Fr>>(
        &mut self,
        public: &Public,
        transcript: &mut T,
        self_check: bool,
    ) -> (Vec<Vec<Fr>>, Vec<Fr>, Claims) {
        self.honest = self_check;
        if self_check {
            assert_eq!(
                self.input_claim(public),
                Fr::zero(),
                "honest witness must satisfy the relation"
            );
        }
        let n = public.log_rows;
        let mut round_polys = Vec::with_capacity(n);
        let mut point = Vec::with_capacity(n);
        for _ in 0..n {
            let coefficients = self.round_poly(public);
            for coefficient in &coefficients {
                transcript.append(coefficient);
            }
            let r: Fr = transcript.challenge();
            self.bind(public, &coefficients, r);
            round_polys.push(coefficients);
            point.push(r);
        }
        (round_polys, point, self.claims())
    }
}
