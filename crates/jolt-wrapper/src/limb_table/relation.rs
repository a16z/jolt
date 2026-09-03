//! The row relation and its sumcheck (lane M2's relation with a signed
//! quotient and public pins): for every row,
//! `eq(τ,row)·Φ(row) + λ·(Σ_g h_g·e_{2,g} − m·inv)` sums to zero, where `Φ`
//! batches the native mod-`r` identity, the three limb-carry identities, the
//! public-pin identity and the grouped LogUp range terms with powers of `γ`.
//! The prover uses split-eq (the `eq` factor multiplies the round polynomial)
//! and unreduced product accumulation for the limb sums.

use jolt_field::{Accumulator, Field, Fr, One, Ring, WithAccumulator, Zero};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use rayon::prelude::*;

use super::columns::{
    fq_limbs, recompose, Columns, Constants, CARRIES, CARRY_CHUNKS, CHUNK_BITS, CHUNK_COLUMNS,
    GROUP_SIZE, HELPER_COLUMNS, K_CHUNKS, LIMBS, Z_CHUNKS,
};
use super::program::Program;
use super::wiring::Wiring;

type Acc = <Fr as WithAccumulator>::Accumulator;

/// Batching terms: native, three carries, three pinned limbs, then one per
/// LogUp group. Pins compare 96-bit limbs, which are exact integers in `Fr`
/// (a full value below `2^256` is only determined modulo `r`).
pub const GAMMA_TERMS: usize = 1 + CARRIES + LIMBS + HELPER_COLUMNS;
const GAMMA_PIN: usize = 1 + CARRIES;
const GAMMA_RANGE: usize = GAMMA_PIN + LIMBS;

/// Public parameters of one row sumcheck; `tau` is big-endian over the row
/// index (`tau[0]` is the most significant bit), bound last.
pub struct RowRelation {
    pub log_rows: usize,
    pub num_slots: usize,
    pub alpha: Fr,
    pub tau: Vec<Fr>,
    pub gamma: Vec<Fr>,
    pub lambda: Fr,
    pub constants: Constants,
}

/// Prover-sent evaluations at the row sumcheck's point: the committed
/// columns (chunks, helpers, multiplicity) and the virtual operand limbs
/// (`x_{s,0..3}, y_{s,0..3}` per slot).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowClaims {
    pub committed: Vec<Fr>,
    pub operand_limbs: Vec<Fr>,
}

impl RowRelation {
    pub fn new(
        log_rows: usize,
        num_slots: usize,
        alpha: Fr,
        tau: Vec<Fr>,
        gamma_root: Fr,
        lambda: Fr,
    ) -> Self {
        assert_eq!(tau.len(), log_rows);
        let mut gamma = Vec::with_capacity(GAMMA_TERMS);
        let mut power = Fr::one();
        for _ in 0..GAMMA_TERMS {
            gamma.push(power);
            power *= gamma_root;
        }
        Self {
            log_rows,
            num_slots,
            alpha,
            tau,
            gamma,
            lambda,
            constants: Constants::new(),
        }
    }

    /// Round-polynomial degree: `eq · h · Π_{3}(α − c)`.
    pub const fn degree() -> usize {
        GROUP_SIZE + 2
    }

    pub fn operand_columns(&self) -> usize {
        2 * LIMBS * self.num_slots
    }

    /// Public range-table vector `1/(α − v)` for `v < 2^16`.
    fn inverse_table(&self) -> Vec<Fr> {
        let mut values: Vec<ark_bn254::Fr> = (0..1u64 << CHUNK_BITS)
            .map(|v| ark_bn254::Fr::from(self.alpha - Fr::from_u64(v)))
            .collect();
        ark_ff::batch_inversion(&mut values);
        values.into_iter().map(Fr::from).collect()
    }

    /// Degree-2 part of `Φ`: native identity, carry identities, pins.
    fn phi_low(&self, chunks: &[Fr], operands: &[Fr], pin: Fr, pin_value: &[Fr; LIMBS]) -> Fr {
        let c = &self.constants;
        let limb_of = |start: usize, count: usize| recompose(&chunks[start..start + count], c);
        let z = [limb_of(0, 6), limb_of(6, 6), limb_of(12, Z_CHUNKS - 12)];
        let k = [
            limb_of(Z_CHUNKS, 6),
            limb_of(Z_CHUNKS + 6, 6),
            limb_of(Z_CHUNKS + 12, K_CHUNKS - 12) - c.k_offset_top_limb,
        ];
        let carry: [Fr; CARRIES] = std::array::from_fn(|i| {
            limb_of(Z_CHUNKS + K_CHUNKS + CARRY_CHUNKS * i, CARRY_CHUNKS) - c.carry_offset
        });
        let full =
            |limbs: &[Fr; LIMBS]| limbs[0] + limbs[1] * c.pow_limb[1] + limbs[2] * c.pow_limb[2];

        let mut native = Acc::default();
        let mut positions = [Acc::default(); CARRIES];
        for s in 0..self.num_slots {
            let x = &operands[2 * LIMBS * s..2 * LIMBS * s + LIMBS];
            let y = &operands[2 * LIMBS * s + LIMBS..2 * LIMBS * (s + 1)];
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
        for a in 0..LIMBS {
            phi += self.gamma[GAMMA_PIN + a] * (pin * z[a] - pin_value[a]);
        }
        phi
    }

    /// Grouped LogUp terms: `(Σ_g γ_g·(h_g·Π_{i∈g}(α − c_i) − 1), Σ_g h_g·e_{2,g})`
    /// with `e_{2,g} = Σ_i Π_{j≠i}(α − c_j)`, so `h_g·e_{2,g} = Σ_{i∈g} 1/(α − c_i)`.
    fn range_and_logup(&self, chunks: &[Fr], helpers: &[Fr]) -> (Fr, Fr) {
        let gammas = &self.gamma[GAMMA_RANGE..];
        let mut range = Fr::zero();
        let mut logup = Fr::zero();
        for ((group, &h), &gamma) in chunks.chunks(GROUP_SIZE).zip(helpers).zip(gammas) {
            let f: [Fr; GROUP_SIZE] = std::array::from_fn(|i| self.alpha - group[i]);
            let product = f[0] * f[1] * f[2];
            let elementary = f[1] * f[2] + f[0] * f[2] + f[0] * f[1];
            range += gamma * (h * product - Fr::one());
            logup += h * elementary;
        }
        (range, logup)
    }

    fn summand(&self, eq: Fr, row: &RowView<'_>) -> Fr {
        let (range, logup) = self.range_and_logup(row.chunks, row.helpers);
        eq * (self.phi_low(row.chunks, row.operands, row.pin, &row.pin_value) + range)
            + self.lambda * (logup - row.multiplicity * row.inverse_table)
    }

    /// The relation's value at the sumcheck's end: `eq(τ, point)·Φ(claims) +
    /// λ·L(claims)`, with the public pin and range-table polynomials evaluated
    /// natively at `point_be` (big-endian) from the program.
    pub fn final_value(
        &self,
        program: &Program,
        claims: &RowClaims,
        point_be: &[Fr],
    ) -> Result<Fr, &'static str> {
        let c = CHUNK_COLUMNS;
        let g = HELPER_COLUMNS;
        if claims.committed.len() != c + g + 1
            || claims.operand_limbs.len() != self.operand_columns()
            || point_be.len() != self.log_rows
        {
            return Err("claim count");
        }
        let eq = EqPolynomial::<Fr>::mle(&self.tau, point_be);
        let eq_point = EqPolynomial::<Fr>::evals(point_be, None);
        let inverse_table = self
            .inverse_table()
            .iter()
            .zip(&eq_point)
            .fold(Fr::zero(), |acc, (inv, weight)| acc + *inv * *weight);
        let mut pin = Fr::zero();
        let mut pin_value = [Fr::zero(); LIMBS];
        for (row, value) in program.pinned_rows() {
            pin += eq_point[row];
            for (acc, limb) in pin_value.iter_mut().zip(fq_limbs(&value)) {
                *acc += eq_point[row] * limb;
            }
        }
        let view = RowView {
            chunks: &claims.committed[..c],
            helpers: &claims.committed[c..c + g],
            multiplicity: claims.committed[c + g],
            operands: &claims.operand_limbs,
            pin,
            pin_value,
            inverse_table,
        };
        Ok(self.summand(eq, &view))
    }
}

/// Borrowed view of one row of the prover matrix, entry order: chunks (54),
/// helpers (18), multiplicity (1), operand limbs (6·slots), pin, pin value,
/// inverse table.
struct RowView<'a> {
    chunks: &'a [Fr],
    helpers: &'a [Fr],
    multiplicity: Fr,
    operands: &'a [Fr],
    pin: Fr,
    pin_value: [Fr; LIMBS],
    inverse_table: Fr,
}

/// Pin flag, three pinned limbs, range-table entry.
const PUBLIC_COLUMNS: usize = 2 + LIMBS;

fn view(values: &[Fr], operands: usize) -> RowView<'_> {
    let (c, g) = (CHUNK_COLUMNS, HELPER_COLUMNS);
    let end = c + g + 1 + operands;
    RowView {
        chunks: &values[..c],
        helpers: &values[c..c + g],
        multiplicity: values[c + g],
        operands: &values[c + g + 1..end],
        pin: values[end],
        pin_value: [values[end + 1], values[end + 2], values[end + 3]],
        inverse_table: values[end + 4],
    }
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

/// The row sumcheck prover as a batch member: a row-major matrix of every
/// committed, virtual and public column, folded round by round (low bit first).
pub struct RowSumcheck<'a> {
    relation: &'a RowRelation,
    matrix: Vec<Fr>,
    scratch: Vec<Fr>,
    width: usize,
    operands: usize,
    rows: usize,
    round: usize,
    eq_prefix: Fr,
    points: Vec<Fr>,
    /// A cheating prover forces every round check to pass so that rejection
    /// happens at the verifier's final relation check.
    pub cheat: bool,
}

impl<'a> RowSumcheck<'a> {
    /// Assembles the matrix from the committed columns, the LogUp witness and
    /// the wiring's virtual operand limbs.
    pub fn new(
        relation: &'a RowRelation,
        program: &Program,
        columns: &Columns,
        helpers: &[Vec<Fr>],
        multiplicities: &[u32],
        wiring: &Wiring,
    ) -> Self {
        let rows = columns.rows();
        assert_eq!(rows, 1 << relation.log_rows);
        let operands = relation.operand_columns();
        let width = CHUNK_COLUMNS + HELPER_COLUMNS + 1 + operands + PUBLIC_COLUMNS;
        let inverse_table = relation.inverse_table();
        let mut pins = vec![(Fr::zero(), [Fr::zero(); LIMBS]); rows];
        for (row, value) in program.pinned_rows() {
            pins[row] = (Fr::one(), fq_limbs(&value));
        }
        let mut matrix = vec![Fr::zero(); rows * width];
        matrix
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, slot)| {
                for (j, value) in slot[..CHUNK_COLUMNS].iter_mut().enumerate() {
                    *value = columns.chunk(row, j);
                }
                for (value, column) in slot[CHUNK_COLUMNS..CHUNK_COLUMNS + HELPER_COLUMNS]
                    .iter_mut()
                    .zip(helpers)
                {
                    *value = column[row];
                }
                let base = CHUNK_COLUMNS + HELPER_COLUMNS;
                slot[base] = Fr::from_u64(u64::from(multiplicities[row]));
                wiring.operand_limbs(columns, row, &mut slot[base + 1..base + 1 + operands]);
                let end = base + 1 + operands;
                slot[end] = pins[row].0;
                slot[end + 1..end + 1 + LIMBS].copy_from_slice(&pins[row].1);
                slot[end + 1 + LIMBS] = inverse_table.get(row).copied().unwrap_or(Fr::zero());
            });
        Self {
            relation,
            matrix,
            scratch: Vec::new(),
            width,
            operands,
            rows,
            round: 0,
            eq_prefix: Fr::one(),
            points: (0..(GROUP_SIZE + 2) as u64).map(Fr::from_u64).collect(),
            cheat: false,
        }
    }

    /// `Σ_row eq(τ,row)·summand(row)` — zero for an honest witness.
    pub fn input_claim(&self) -> Fr {
        let eq = EqPolynomial::<Fr>::evals(&self.relation.tau, None);
        self.matrix
            .par_chunks(self.width)
            .zip(&eq)
            .map(|(row, &eq)| self.relation.summand(eq, &view(row, self.operands)))
            .sum()
    }

    /// Coefficients of this round's polynomial
    /// `s(X) = eq_prefix·eq1(τ_round, X)·(q_low + q_range)(X) + λ·L(X)`.
    fn round_poly(&self, claim: Fr) -> Vec<Fr> {
        let relation = self.relation;
        let (width, operands) = (self.width, self.operands);
        let n = relation.log_rows;
        let high_points = GROUP_SIZE + 2;
        let logup_points = GROUP_SIZE + 1;
        let points = &self.points;
        let tau_round = relation.tau[n - 1 - self.round];
        let eq_rest = if self.round + 1 == n {
            vec![Fr::one()]
        } else {
            EqPolynomial::<Fr>::evals(&relation.tau[..n - 1 - self.round], None)
        };
        let sums = self.matrix[..self.rows * width]
            .par_chunks(2 * width)
            .zip(&eq_rest)
            .fold(
                || {
                    (
                        vec![Fr::zero(); 3 + 2 * high_points],
                        vec![Fr::zero(); width],
                        vec![Fr::zero(); CHUNK_COLUMNS + HELPER_COLUMNS],
                    )
                },
                |(mut acc, mut full, mut partial), (pair, &eq)| {
                    let (even, odd) = pair.split_at(width);
                    let (q_low, rest) = acc.split_at_mut(3);
                    let (q_high, l) = rest.split_at_mut(high_points);
                    let even_view = view(even, operands);
                    let odd_view = view(odd, operands);
                    let low = |v: &RowView<'_>| {
                        relation.phi_low(v.chunks, v.operands, v.pin, &v.pin_value)
                    };
                    q_low[0] += eq * low(&even_view);
                    q_low[1] += eq * low(&odd_view);
                    extrapolate(&mut full, even, odd, points[2]);
                    let two_view = view(&full, operands);
                    q_low[2] += eq * low(&two_view);
                    let mut high = |x: usize, v: &RowView<'_>| {
                        let (range, logup) = relation.range_and_logup(v.chunks, v.helpers);
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
                    let c = CHUNK_COLUMNS + HELPER_COLUMNS;
                    for (x, &point) in points.iter().enumerate().skip(3) {
                        extrapolate(&mut partial, &even[..c], &odd[..c], point);
                        let v = RowView {
                            chunks: &partial[..CHUNK_COLUMNS],
                            helpers: &partial[CHUNK_COLUMNS..c],
                            multiplicity: Fr::zero(),
                            operands: &[],
                            pin: Fr::zero(),
                            pin_value: [Fr::zero(); LIMBS],
                            inverse_table: Fr::zero(),
                        };
                        high(x, &v);
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
            relation.lambda,
        );
        while coefficients.len() > 2 && coefficients.last() == Some(&Fr::zero()) {
            let _ = coefficients.pop();
        }
        if self.cheat {
            let tail: Fr = coefficients[1..].iter().fold(Fr::zero(), |acc, c| acc + *c);
            coefficients[0] = (claim - tail) * two_inverse();
        }
        coefficients
    }

    fn bind(&mut self, r: Fr) {
        let width = self.width;
        let half = self.rows / 2;
        let tau_round = self.relation.tau[self.relation.log_rows - 1 - self.round];
        if self.scratch.len() < half * width {
            self.scratch.resize(half * width, Fr::zero());
        }
        self.scratch[..half * width]
            .par_chunks_mut(width)
            .zip(self.matrix[..self.rows * width].par_chunks(2 * width))
            .for_each(|(out, pair)| {
                let (even, odd) = pair.split_at(width);
                extrapolate(out, even, odd, r);
            });
        std::mem::swap(&mut self.matrix, &mut self.scratch);
        self.eq_prefix *= (Fr::one() - tau_round) * (Fr::one() - r) + tau_round * r;
        self.rows = half;
        self.round += 1;
    }

    /// Final evaluations after every round: committed columns and operand limbs.
    pub fn claims(&self) -> RowClaims {
        assert_eq!(self.rows, 1, "sumcheck not finished");
        let finals = &self.matrix[..self.width];
        let c = CHUNK_COLUMNS + HELPER_COLUMNS + 1;
        RowClaims {
            committed: finals[..c].to_vec(),
            operand_limbs: finals[c..c + self.operands].to_vec(),
        }
    }
}

fn two_inverse() -> Fr {
    Fr::from_u64(2)
        .inverse()
        .unwrap_or_else(|| unreachable!("2 is invertible"))
}

impl ProveRounds<Fr> for RowSumcheck<'_> {
    fn num_rounds(&self) -> usize {
        self.relation.log_rows
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
