//! The batched final relation exported as terms over column evaluations at
//! the stage's common point: `G(v) = Σ_t coefficient_t · Π_j L_{t,j}(v)` with
//! affine `L`. Prover and verifier build the same list from the transcript
//! challenges; the stream's term stage compresses it to one claim.

use jolt_field::{Fr, Ring, Zero};

/// Index into the table's exported column list ([`super::export`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ColumnId(pub u32);

/// `constant + Σ w_c · v_c`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct AffineForm {
    pub constant: Fr,
    pub weights: Vec<(ColumnId, Fr)>,
}

impl AffineForm {
    pub fn constant(constant: Fr) -> Self {
        Self {
            constant,
            weights: Vec::new(),
        }
    }

    pub fn column(id: ColumnId) -> Self {
        Self::scaled(id, Fr::from_u64(1))
    }

    pub fn scaled(id: ColumnId, weight: Fr) -> Self {
        Self {
            constant: Fr::zero(),
            weights: vec![(id, weight)],
        }
    }

    pub fn plus(mut self, other: &AffineForm) -> Self {
        self.accumulate(other);
        self
    }

    pub fn accumulate(&mut self, other: &AffineForm) {
        self.constant += other.constant;
        for (id, weight) in &other.weights {
            self.add_column(*id, *weight);
        }
    }

    pub fn add_column(&mut self, id: ColumnId, weight: Fr) {
        match self.weights.iter_mut().find(|(c, _)| *c == id) {
            Some((_, w)) => *w += weight,
            None => self.weights.push((id, weight)),
        }
    }

    pub fn scale(mut self, factor: Fr) -> Self {
        self.constant *= factor;
        for (_, weight) in &mut self.weights {
            *weight *= factor;
        }
        self
    }

    pub fn evaluate(&self, values: &[Fr]) -> Fr {
        self.weights
            .iter()
            .fold(self.constant, |acc, (id, weight)| {
                acc + *weight * values[id.0 as usize]
            })
    }
}

/// `coefficient · Π_j factors_j(v)`; an empty factor list is the constant term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term {
    pub coefficient: Fr,
    pub factors: Vec<AffineForm>,
}

impl Term {
    pub fn new(coefficient: Fr, factors: Vec<AffineForm>) -> Self {
        Self {
            coefficient,
            factors,
        }
    }

    pub fn linear(form: AffineForm) -> Self {
        Self::new(Fr::from_u64(1), vec![form])
    }

    pub fn degree(&self) -> usize {
        self.factors.len()
    }

    pub fn evaluate(&self, values: &[Fr]) -> Fr {
        self.factors
            .iter()
            .fold(self.coefficient, |acc, form| acc * form.evaluate(values))
    }
}

/// `Σ_t term_t(v)`.
pub fn evaluate_terms(terms: &[Term], values: &[Fr]) -> Fr {
    terms
        .iter()
        .fold(Fr::zero(), |acc, term| acc + term.evaluate(values))
}

/// Folds every degree-1 term into one linear term so the list carries one
/// `d = 1` term plus the genuine products.
pub fn fold_linear(terms: Vec<Term>) -> Vec<Term> {
    let mut linear = AffineForm::default();
    let mut products = Vec::with_capacity(terms.len());
    for term in terms {
        match term.factors.len() {
            0 => linear.constant += term.coefficient,
            1 => {
                let [form] = <[AffineForm; 1]>::try_from(term.factors)
                    .unwrap_or_else(|_| unreachable!("one factor"));
                linear.accumulate(&form.scale(term.coefficient));
            }
            _ => products.push(term),
        }
    }
    let mut out = vec![Term::linear(linear)];
    out.extend(products);
    out
}
