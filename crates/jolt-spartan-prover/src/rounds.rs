use jolt_field::JoltField;
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_spartan_verifier::{SpartanError, SpartanKey, INNER_DEGREE, OUTER_DEGREE};
use jolt_sumcheck::{ProveRounds, SumcheckError};

pub(crate) struct OuterRounds<F: JoltField> {
    polynomials: [Polynomial<F>; 4],
    rounds: usize,
}

impl<F: JoltField> OuterRounds<F> {
    pub(crate) fn new(
        key: &SpartanKey<F>,
        assignment: &[F],
        tau: &[F],
    ) -> Result<Self, SpartanError<F>> {
        let matrices = key.matrices();
        let products: [Result<Polynomial<F>, SpartanError<F>>; 3] =
            [&matrices.a, &matrices.b, &matrices.c].map(|matrix| {
                let mut values = Vec::with_capacity(key.padded_rows());
                for row in matrix {
                    let mut sum = F::zero();
                    for (column, coefficient) in row {
                        sum += *assignment.get(*column).ok_or(SpartanError::InternalShape)?
                            * coefficient;
                    }
                    values.push(sum);
                }
                values.resize(key.padded_rows(), F::zero());
                Ok(Polynomial::new(values))
            });
        let [a, b, c] = products;
        let (a, b, c) = (a?, b?, c?);
        for (row, ((a, b), c)) in a.evals().iter().zip(b.evals()).zip(c.evals()).enumerate() {
            if *a * b != *c {
                return Err(SpartanError::Unsatisfied(row));
            }
        }
        Ok(Self {
            polynomials: [
                Polynomial::new(EqPolynomial::new(tau.to_vec()).evaluations()),
                a,
                b,
                c,
            ],
            rounds: key.row_vars(),
        })
    }

    pub(crate) fn evaluations(&self) -> Result<[F; 3], SpartanError<F>> {
        let [_, a, b, c] = &self.polynomials;
        Ok([
            *a.evals().first().ok_or(SpartanError::InternalShape)?,
            *b.evals().first().ok_or(SpartanError::InternalShape)?,
            *c.evals().first().ok_or(SpartanError::InternalShape)?,
        ])
    }

    fn bind(&mut self, value: F) {
        for polynomial in &mut self.polynomials {
            polynomial.bind_with_order(value, BindingOrder::HighToLow);
        }
    }
}

impl<F: JoltField> ProveRounds<F> for OuterRounds<F> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        _claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(bind) = bind {
            self.bind(bind);
        }
        let [eq, a, b, c] = &self.polynomials;
        let mut evaluations = [F::zero(); OUTER_DEGREE + 1];
        for index in 0..a.len() / 2 {
            for (x, result) in evaluations.iter_mut().enumerate() {
                let x = F::from_u64(x as u64);
                *result += eq.sumcheck_round_eval(index, x)
                    * (a.sumcheck_round_eval(index, x) * b.sumcheck_round_eval(index, x)
                        - c.sumcheck_round_eval(index, x));
            }
        }
        Ok(UnivariatePoly::from_evals(&evaluations))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

pub(crate) struct InnerRounds<F: JoltField> {
    polynomials: [Polynomial<F>; 2],
    rounds: usize,
}

impl<F: JoltField> InnerRounds<F> {
    pub(crate) fn new(linear: Vec<F>, witness: Vec<F>, rounds: usize) -> Self {
        Self {
            polynomials: [Polynomial::new(linear), Polynomial::new(witness)],
            rounds,
        }
    }

    pub(crate) fn evaluations(&self) -> Result<[F; 2], SpartanError<F>> {
        let [linear, witness] = &self.polynomials;
        Ok([
            *linear.evals().first().ok_or(SpartanError::InternalShape)?,
            *witness.evals().first().ok_or(SpartanError::InternalShape)?,
        ])
    }

    fn bind(&mut self, value: F) {
        for polynomial in &mut self.polynomials {
            polynomial.bind_with_order(value, BindingOrder::HighToLow);
        }
    }
}

impl<F: JoltField> ProveRounds<F> for InnerRounds<F> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        _claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(bind) = bind {
            self.bind(bind);
        }
        let [linear, witness] = &self.polynomials;
        let mut evaluations = [F::zero(); INNER_DEGREE + 1];
        for index in 0..linear.len() / 2 {
            for (x, result) in evaluations.iter_mut().enumerate() {
                let x = F::from_u64(x as u64);
                *result +=
                    linear.sumcheck_round_eval(index, x) * witness.sumcheck_round_eval(index, x);
            }
        }
        Ok(UnivariatePoly::from_evals(&evaluations))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}
