use jolt_field::{AdditiveAccumulator, Field, RingAccumulator, WithAccumulator};
use jolt_poly::{BindingOrder, EqPlusOnePrefixSuffix, EqPolynomial, Polynomial, UnivariatePoly};
use rayon::prelude::*;

use crate::{BackendError, SumcheckStage3ShiftRow, SumcheckStage3ShiftStateRequest};

const STAGE3_SHIFT_DEGREE_EVALS: usize = 2;

pub struct SumcheckStage3ShiftState<F: Field> {
    label: &'static str,
    log_t: usize,
    rows: Vec<SumcheckStage3ShiftRow>,
    outer_point: Vec<F>,
    product_point: Vec<F>,
    gamma: F,
    gamma2: F,
    gamma3: F,
    gamma4: F,
    round: usize,
    phase: ShiftPhase<F>,
}

enum ShiftPhase<F: Field> {
    Prefix(ShiftPrefixState<F>),
    Suffix(ShiftSuffixState<F>),
    Taken,
}

struct ShiftPrefixState<F: Field> {
    p_outer: [Polynomial<F>; 2],
    p_product: [Polynomial<F>; 2],
    q_outer: [Polynomial<F>; 2],
    q_product: [Polynomial<F>; 2],
    challenges: Vec<F>,
}

struct ShiftSuffixState<F: Field> {
    unexpanded_pc: Polynomial<F>,
    pc: Polynomial<F>,
    is_virtual: Polynomial<F>,
    is_first_in_sequence: Polynomial<F>,
    is_noop: Polynomial<F>,
    eq_plus_one_outer: Polynomial<F>,
    eq_plus_one_product: Polynomial<F>,
}

impl<F> SumcheckStage3ShiftState<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckStage3ShiftStateRequest<F>,
    ) -> Result<Self, BackendError> {
        validate_request(backend, task, request)?;

        let gamma2 = request.gamma * request.gamma;
        let gamma3 = gamma2 * request.gamma;
        let gamma4 = gamma3 * request.gamma;
        let phase = ShiftPhase::Prefix(ShiftPrefixState::new(
            &request.rows,
            &request.outer_point,
            &request.product_point,
            request.gamma,
            gamma2,
            gamma3,
            gamma4,
        ));

        Ok(Self {
            label: request.label,
            log_t: request.log_t,
            rows: request.rows.clone(),
            outer_point: request.outer_point.clone(),
            product_point: request.product_point.clone(),
            gamma: request.gamma,
            gamma2,
            gamma3,
            gamma4,
            round: 0,
            phase,
        })
    }

    pub fn evaluate_round(
        &self,
        backend: &'static str,
        task: &'static str,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError> {
        if self.round >= self.log_t {
            return invalid(
                backend,
                task,
                format!(
                    "{} round {} is outside {} rounds",
                    self.label, self.round, self.log_t
                ),
            );
        }
        Ok(match &self.phase {
            ShiftPhase::Prefix(state) => state.evaluate_round(previous_claim),
            ShiftPhase::Suffix(state) => state.evaluate_round(
                previous_claim,
                self.gamma,
                self.gamma2,
                self.gamma3,
                self.gamma4,
            ),
            ShiftPhase::Taken => {
                return invalid(backend, task, format!("{} state was moved", self.label));
            }
        })
    }

    pub fn bind(
        &mut self,
        backend: &'static str,
        task: &'static str,
        challenge: F,
    ) -> Result<(), BackendError> {
        if self.round >= self.log_t {
            return invalid(
                backend,
                task,
                format!(
                    "{} bind round {} is outside {} rounds",
                    self.label, self.round, self.log_t
                ),
            );
        }

        let phase = std::mem::replace(&mut self.phase, ShiftPhase::Taken);
        self.phase = match phase {
            ShiftPhase::Prefix(mut state) => {
                if state.should_transition_to_suffix() {
                    state.challenges.push(challenge);
                    ShiftPhase::Suffix(ShiftSuffixState::new(
                        &self.rows,
                        &self.outer_point,
                        &self.product_point,
                        &state.challenges,
                    ))
                } else {
                    state.bind(challenge);
                    ShiftPhase::Prefix(state)
                }
            }
            ShiftPhase::Suffix(mut state) => {
                state.bind(challenge);
                ShiftPhase::Suffix(state)
            }
            ShiftPhase::Taken => {
                return invalid(backend, task, format!("{} state was moved", self.label));
            }
        };
        self.round += 1;
        Ok(())
    }

    pub fn output_openings(
        &self,
        backend: &'static str,
        task: &'static str,
    ) -> Result<[F; 5], BackendError> {
        if self.round != self.log_t {
            return invalid(
                backend,
                task,
                format!(
                    "{} output openings requested after {} of {} rounds",
                    self.label, self.round, self.log_t
                ),
            );
        }
        let ShiftPhase::Suffix(state) = &self.phase else {
            return invalid(
                backend,
                task,
                format!(
                    "{} output openings requested before suffix phase",
                    self.label
                ),
            );
        };
        state.output_openings(backend, task, self.label)
    }
}

impl<F> ShiftPrefixState<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    fn new(
        rows: &[SumcheckStage3ShiftRow],
        outer_point: &[F],
        product_point: &[F],
        gamma: F,
        gamma2: F,
        gamma3: F,
        gamma4: F,
    ) -> Self {
        let outer = EqPlusOnePrefixSuffix::new(outer_point);
        let product = EqPlusOnePrefixSuffix::new(product_point);
        let prefix_vars = outer.prefix_0.len().ilog2() as usize;
        let suffix_vars = outer.suffix_0.len().ilog2() as usize;
        let prefix_len = 1usize << prefix_vars;
        let suffix_len = 1usize << suffix_vars;

        let mut q_outer_0 = vec![F::zero(); prefix_len];
        let mut q_outer_1 = vec![F::zero(); prefix_len];
        let mut q_product_0 = vec![F::zero(); prefix_len];
        let mut q_product_1 = vec![F::zero(); prefix_len];

        const BLOCK_SIZE: usize = 32;
        (
            q_outer_0.par_chunks_mut(BLOCK_SIZE),
            q_outer_1.par_chunks_mut(BLOCK_SIZE),
            q_product_0.par_chunks_mut(BLOCK_SIZE),
            q_product_1.par_chunks_mut(BLOCK_SIZE),
        )
            .into_par_iter()
            .enumerate()
            .for_each(
                |(chunk_index, (q_outer_0, q_outer_1, q_product_0, q_product_1))| {
                    let chunk_len = q_outer_0.len();
                    let mut outer_0 = [<F as WithAccumulator>::Accumulator::default(); BLOCK_SIZE];
                    let mut outer_1 = [<F as WithAccumulator>::Accumulator::default(); BLOCK_SIZE];
                    let mut product_0 =
                        [<F as WithAccumulator>::Accumulator::default(); BLOCK_SIZE];
                    let mut product_1 =
                        [<F as WithAccumulator>::Accumulator::default(); BLOCK_SIZE];

                    for high in 0..suffix_len {
                        let row_base = high << prefix_vars;
                        for low_offset in 0..chunk_len {
                            let low = chunk_index * BLOCK_SIZE + low_offset;
                            let row = rows[row_base + low];
                            let mut value = F::from_u64(row.unexpanded_pc);
                            value += gamma * F::from_u64(row.pc);
                            if row.is_virtual {
                                value += gamma2;
                            }
                            if row.is_first_in_sequence {
                                value += gamma3;
                            }

                            outer_0[low_offset].fmadd(value, outer.suffix_0[high]);
                            outer_1[low_offset].fmadd(value, outer.suffix_1[high]);
                            if !row.is_noop {
                                product_0[low_offset].fmadd(product.suffix_0[high], gamma4);
                                product_1[low_offset].fmadd(product.suffix_1[high], gamma4);
                            }
                        }
                    }

                    for i in 0..chunk_len {
                        q_outer_0[i] = outer_0[i].reduce();
                        q_outer_1[i] = outer_1[i].reduce();
                        q_product_0[i] = product_0[i].reduce();
                        q_product_1[i] = product_1[i].reduce();
                    }
                },
            );

        Self {
            p_outer: [
                Polynomial::new(outer.prefix_0),
                Polynomial::new(outer.prefix_1),
            ],
            p_product: [
                Polynomial::new(product.prefix_0),
                Polynomial::new(product.prefix_1),
            ],
            q_outer: [Polynomial::new(q_outer_0), Polynomial::new(q_outer_1)],
            q_product: [Polynomial::new(q_product_0), Polynomial::new(q_product_1)],
            challenges: Vec::with_capacity(prefix_vars),
        }
    }

    fn evaluate_round(&self, previous_claim: F) -> UnivariatePoly<F> {
        let evals = [&self.p_outer, &self.p_product]
            .into_par_iter()
            .zip([&self.q_outer, &self.q_product])
            .map(|(p, q)| prefix_suffix_pair_evals(p, q))
            .reduce(
                || [F::zero(); STAGE3_SHIFT_DEGREE_EVALS],
                sum_arrays::<F, STAGE3_SHIFT_DEGREE_EVALS>,
            );
        UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, challenge: F) {
        self.challenges.push(challenge);
        for poly in self
            .p_outer
            .iter_mut()
            .chain(self.p_product.iter_mut())
            .chain(self.q_outer.iter_mut())
            .chain(self.q_product.iter_mut())
        {
            poly.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
    }

    fn should_transition_to_suffix(&self) -> bool {
        self.p_outer[0].num_vars() == 1
    }
}

impl<F> ShiftSuffixState<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    fn new(
        rows: &[SumcheckStage3ShiftRow],
        outer_point: &[F],
        product_point: &[F],
        prefix_challenges: &[F],
    ) -> Self {
        let prefix_vars = prefix_challenges.len();
        let prefix_len = 1usize << prefix_vars;
        let r_prefix = prefix_challenges.iter().rev().copied().collect::<Vec<_>>();
        let eq_prefix = EqPolynomial::new(r_prefix.clone()).evaluations();

        let outer = bound_eq_plus_one_suffix(outer_point, &r_prefix);
        let product = bound_eq_plus_one_suffix(product_point, &r_prefix);

        let aggregates = rows
            .par_chunks(prefix_len)
            .map(|chunk| aggregate_shift_suffix_chunk(chunk, &eq_prefix))
            .collect::<Vec<_>>();

        Self {
            unexpanded_pc: Polynomial::new(
                aggregates.iter().map(|row| row.unexpanded_pc).collect(),
            ),
            pc: Polynomial::new(aggregates.iter().map(|row| row.pc).collect()),
            is_virtual: Polynomial::new(aggregates.iter().map(|row| row.is_virtual).collect()),
            is_first_in_sequence: Polynomial::new(
                aggregates
                    .iter()
                    .map(|row| row.is_first_in_sequence)
                    .collect(),
            ),
            is_noop: Polynomial::new(aggregates.iter().map(|row| row.is_noop).collect()),
            eq_plus_one_outer: Polynomial::new(outer),
            eq_plus_one_product: Polynomial::new(product),
        }
    }

    fn evaluate_round(
        &self,
        previous_claim: F,
        gamma: F,
        gamma2: F,
        gamma3: F,
        gamma4: F,
    ) -> UnivariatePoly<F> {
        let evals = (0..self.unexpanded_pc.len() / 2)
            .into_par_iter()
            .map(|index| {
                let unexpanded_pc = evals_over_integer_domain(&self.unexpanded_pc, index);
                let pc = evals_over_integer_domain(&self.pc, index);
                let is_virtual = evals_over_integer_domain(&self.is_virtual, index);
                let is_first_in_sequence =
                    evals_over_integer_domain(&self.is_first_in_sequence, index);
                let is_noop = evals_over_integer_domain(&self.is_noop, index);
                let outer = evals_over_integer_domain(&self.eq_plus_one_outer, index);
                let product = evals_over_integer_domain(&self.eq_plus_one_product, index);
                std::array::from_fn(|point| {
                    outer[point]
                        * (unexpanded_pc[point]
                            + gamma * pc[point]
                            + gamma2 * is_virtual[point]
                            + gamma3 * is_first_in_sequence[point])
                        + product[point] * gamma4 * (F::one() - is_noop[point])
                })
            })
            .reduce(
                || [F::zero(); STAGE3_SHIFT_DEGREE_EVALS],
                sum_arrays::<F, STAGE3_SHIFT_DEGREE_EVALS>,
            );
        UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, challenge: F) {
        for poly in [
            &mut self.unexpanded_pc,
            &mut self.pc,
            &mut self.is_virtual,
            &mut self.is_first_in_sequence,
            &mut self.is_noop,
            &mut self.eq_plus_one_outer,
            &mut self.eq_plus_one_product,
        ] {
            poly.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
    }

    fn output_openings(
        &self,
        backend: &'static str,
        task: &'static str,
        label: &'static str,
    ) -> Result<[F; 5], BackendError> {
        Ok([
            final_bound_value(backend, task, label, "unexpanded_pc", &self.unexpanded_pc)?,
            final_bound_value(backend, task, label, "pc", &self.pc)?,
            final_bound_value(backend, task, label, "is_virtual", &self.is_virtual)?,
            final_bound_value(
                backend,
                task,
                label,
                "is_first_in_sequence",
                &self.is_first_in_sequence,
            )?,
            final_bound_value(backend, task, label, "is_noop", &self.is_noop)?,
        ])
    }
}

#[derive(Clone, Copy)]
struct ShiftSuffixAggregate<F: Field> {
    unexpanded_pc: F,
    pc: F,
    is_virtual: F,
    is_first_in_sequence: F,
    is_noop: F,
}

fn aggregate_shift_suffix_chunk<F: Field>(
    rows: &[SumcheckStage3ShiftRow],
    eq_prefix: &[F],
) -> ShiftSuffixAggregate<F>
where
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    let mut unexpanded_pc = <F as WithAccumulator>::Accumulator::default();
    let mut pc = <F as WithAccumulator>::Accumulator::default();
    let mut is_virtual = <F as WithAccumulator>::Accumulator::default();
    let mut is_first_in_sequence = <F as WithAccumulator>::Accumulator::default();
    let mut is_noop = <F as WithAccumulator>::Accumulator::default();

    for (row, eq) in rows.iter().zip(eq_prefix.iter().copied()) {
        unexpanded_pc.fmadd_u64(eq, row.unexpanded_pc);
        pc.fmadd_u64(eq, row.pc);
        is_virtual.fmadd_bool(eq, row.is_virtual);
        is_first_in_sequence.fmadd_bool(eq, row.is_first_in_sequence);
        is_noop.fmadd_bool(eq, row.is_noop);
    }

    ShiftSuffixAggregate {
        unexpanded_pc: unexpanded_pc.reduce(),
        pc: pc.reduce(),
        is_virtual: is_virtual.reduce(),
        is_first_in_sequence: is_first_in_sequence.reduce(),
        is_noop: is_noop.reduce(),
    }
}

fn final_bound_value<F: Field>(
    backend: &'static str,
    task: &'static str,
    label: &'static str,
    polynomial: &'static str,
    values: &Polynomial<F>,
) -> Result<F, BackendError> {
    let [value] = values.evaluations() else {
        return invalid(
            backend,
            task,
            format!(
                "{label} output polynomial {polynomial} has {} evaluations, expected 1",
                values.len()
            ),
        );
    };
    Ok(*value)
}

fn bound_eq_plus_one_suffix<F: Field>(point: &[F], r_prefix: &[F]) -> Vec<F> {
    let decomposition = EqPlusOnePrefixSuffix::new(point);
    let prefix_0 = Polynomial::new(decomposition.prefix_0).evaluate(r_prefix);
    let prefix_1 = Polynomial::new(decomposition.prefix_1).evaluate(r_prefix);
    decomposition
        .suffix_0
        .iter()
        .zip(decomposition.suffix_1)
        .map(|(&suffix_0, suffix_1)| prefix_0 * suffix_0 + prefix_1 * suffix_1)
        .collect()
}

fn prefix_suffix_pair_evals<F: Field>(
    p: &[Polynomial<F>; 2],
    q: &[Polynomial<F>; 2],
) -> [F; STAGE3_SHIFT_DEGREE_EVALS] {
    p.par_iter()
        .zip(q.par_iter())
        .map(|(p, q)| {
            (0..p.len() / 2)
                .into_par_iter()
                .map(|index| {
                    let p = evals_over_integer_domain(p, index);
                    let q = evals_over_integer_domain(q, index);
                    std::array::from_fn(|point| p[point] * q[point])
                })
                .reduce(
                    || [F::zero(); STAGE3_SHIFT_DEGREE_EVALS],
                    sum_arrays::<F, STAGE3_SHIFT_DEGREE_EVALS>,
                )
        })
        .reduce(
            || [F::zero(); STAGE3_SHIFT_DEGREE_EVALS],
            sum_arrays::<F, STAGE3_SHIFT_DEGREE_EVALS>,
        )
}

fn evals_over_integer_domain<F: Field>(
    polynomial: &Polynomial<F>,
    index: usize,
) -> [F; STAGE3_SHIFT_DEGREE_EVALS] {
    let (lo, hi) = polynomial.sumcheck_eval_pair(index, BindingOrder::LowToHigh);
    let step = hi - lo;
    [lo, hi + step]
}

fn sum_arrays<F: Field, const N: usize>(mut left: [F; N], right: [F; N]) -> [F; N] {
    for (left, right) in left.iter_mut().zip(right) {
        *left += right;
    }
    left
}

fn validate_request<F: Field>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckStage3ShiftStateRequest<F>,
) -> Result<(), BackendError> {
    if request.log_t == 0 {
        return invalid(backend, task, "Stage 3 shift requires at least one round");
    }
    if request.outer_point.len() != request.log_t {
        return invalid(
            backend,
            task,
            format!(
                "Stage 3 shift outer point has {} variables, expected {}",
                request.outer_point.len(),
                request.log_t
            ),
        );
    }
    if request.product_point.len() != request.log_t {
        return invalid(
            backend,
            task,
            format!(
                "Stage 3 shift product point has {} variables, expected {}",
                request.product_point.len(),
                request.log_t
            ),
        );
    }
    let expected_rows =
        1usize
            .checked_shl(request.log_t as u32)
            .ok_or_else(|| BackendError::InvalidRequest {
                backend,
                task,
                reason: format!(
                    "Stage 3 shift row count overflows for log_t={}",
                    request.log_t
                ),
            })?;
    if request.rows.len() != expected_rows {
        return invalid(
            backend,
            task,
            format!(
                "Stage 3 shift request has {} rows, expected {expected_rows}",
                request.rows.len()
            ),
        );
    }
    Ok(())
}

fn invalid<T>(
    backend: &'static str,
    task: &'static str,
    reason: impl std::fmt::Display,
) -> Result<T, BackendError> {
    Err(BackendError::InvalidRequest {
        backend,
        task,
        reason: reason.to_string(),
    })
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "unit tests should fail with context")]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::EqPlusOnePolynomial;

    #[test]
    fn stage3_shift_prefix_suffix_matches_dense_reference() {
        let log_t = 5;
        let rows = (0..1usize << log_t)
            .map(|index| {
                SumcheckStage3ShiftRow::new(
                    index as u64,
                    (index as u64) * 3 + 1,
                    index % 3 == 0,
                    index % 5 == 0,
                    index % 7 == 0,
                )
            })
            .collect::<Vec<_>>();
        let outer_point = (0..log_t)
            .map(|i| Fr::from_u64((i as u64) + 2))
            .collect::<Vec<_>>();
        let product_point = (0..log_t)
            .map(|i| Fr::from_u64((i as u64) + 11))
            .collect::<Vec<_>>();
        let gamma = Fr::from_u64(19);
        let request = SumcheckStage3ShiftStateRequest::new(
            "test.stage3_shift",
            log_t,
            outer_point.clone(),
            product_point.clone(),
            gamma,
            rows.clone(),
        );
        let mut state = SumcheckStage3ShiftState::new("cpu", "test.stage3_shift", &request)
            .expect("state should materialize");

        let mut reference =
            DenseShiftProductReference::new(&rows, &outer_point, &product_point, gamma);
        let mut previous_claim = reference.claim();

        for round in 0..log_t {
            let actual = state
                .evaluate_round("cpu", "test.stage3_shift", previous_claim)
                .expect("round should evaluate");
            let expected = reference.round_polynomial(previous_claim);
            assert_eq!(
                actual.coefficients(),
                expected.coefficients(),
                "round {round} mismatch"
            );

            let challenge = Fr::from_u64(101 + round as u64);
            previous_claim = actual.evaluate(challenge);
            state
                .bind("cpu", "test.stage3_shift", challenge)
                .expect("state should bind");
            reference.bind(challenge);
        }

        assert_eq!(reference.claim(), previous_claim);
    }

    struct DenseShiftProductReference {
        unexpanded_pc: Polynomial<Fr>,
        pc: Polynomial<Fr>,
        is_virtual: Polynomial<Fr>,
        is_first_in_sequence: Polynomial<Fr>,
        is_noop: Polynomial<Fr>,
        outer: Polynomial<Fr>,
        product: Polynomial<Fr>,
        gamma: Fr,
        gamma2: Fr,
        gamma3: Fr,
        gamma4: Fr,
    }

    impl DenseShiftProductReference {
        fn new(
            rows: &[SumcheckStage3ShiftRow],
            outer_point: &[Fr],
            product_point: &[Fr],
            gamma: Fr,
        ) -> Self {
            let gamma2 = gamma * gamma;
            let gamma3 = gamma2 * gamma;
            let gamma4 = gamma3 * gamma;
            Self {
                unexpanded_pc: Polynomial::new(
                    rows.iter()
                        .map(|row| Fr::from_u64(row.unexpanded_pc))
                        .collect(),
                ),
                pc: Polynomial::new(rows.iter().map(|row| Fr::from_u64(row.pc)).collect()),
                is_virtual: Polynomial::new(
                    rows.iter()
                        .map(|row| Fr::from_bool(row.is_virtual))
                        .collect(),
                ),
                is_first_in_sequence: Polynomial::new(
                    rows.iter()
                        .map(|row| Fr::from_bool(row.is_first_in_sequence))
                        .collect(),
                ),
                is_noop: Polynomial::new(
                    rows.iter().map(|row| Fr::from_bool(row.is_noop)).collect(),
                ),
                outer: Polynomial::new(EqPlusOnePolynomial::evals(outer_point, None).1),
                product: Polynomial::new(EqPlusOnePolynomial::evals(product_point, None).1),
                gamma,
                gamma2,
                gamma3,
                gamma4,
            }
        }

        fn claim(&self) -> Fr {
            (0..self.unexpanded_pc.len())
                .map(|index| {
                    self.outer.evals()[index]
                        * (self.unexpanded_pc.evals()[index]
                            + self.gamma * self.pc.evals()[index]
                            + self.gamma2 * self.is_virtual.evals()[index]
                            + self.gamma3 * self.is_first_in_sequence.evals()[index])
                        + self.product.evals()[index]
                            * self.gamma4
                            * (Fr::from_u64(1) - self.is_noop.evals()[index])
                })
                .sum()
        }

        fn round_polynomial(&self, previous_claim: Fr) -> UnivariatePoly<Fr> {
            let evals = (0..self.unexpanded_pc.len() / 2)
                .map(|index| {
                    let unexpanded_pc = evals_over_integer_domain(&self.unexpanded_pc, index);
                    let pc = evals_over_integer_domain(&self.pc, index);
                    let is_virtual = evals_over_integer_domain(&self.is_virtual, index);
                    let is_first_in_sequence =
                        evals_over_integer_domain(&self.is_first_in_sequence, index);
                    let is_noop = evals_over_integer_domain(&self.is_noop, index);
                    let outer = evals_over_integer_domain(&self.outer, index);
                    let product = evals_over_integer_domain(&self.product, index);
                    std::array::from_fn(|point| {
                        outer[point]
                            * (unexpanded_pc[point]
                                + self.gamma * pc[point]
                                + self.gamma2 * is_virtual[point]
                                + self.gamma3 * is_first_in_sequence[point])
                            + product[point] * self.gamma4 * (Fr::from_u64(1) - is_noop[point])
                    })
                })
                .fold([Fr::from_u64(0); STAGE3_SHIFT_DEGREE_EVALS], sum_arrays);
            UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
        }

        fn bind(&mut self, challenge: Fr) {
            for polynomial in [
                &mut self.unexpanded_pc,
                &mut self.pc,
                &mut self.is_virtual,
                &mut self.is_first_in_sequence,
                &mut self.is_noop,
                &mut self.outer,
                &mut self.product,
            ] {
                polynomial.bind_with_order(challenge, BindingOrder::LowToHigh);
            }
        }
    }
}
