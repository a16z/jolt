//! The batched final relation exported as terms over column evaluations at
//! the stage's common point: `G(v) = Σ_t coefficient_t · Π_j L_{t,j}(v)` with
//! affine `L`. Prover and verifier build the same list from the transcript
//! challenges; the stream's term stage compresses it to one claim.

use jolt_field::{Fr, Ring, Zero};

/// A (possibly observed) field multiplication the verifier-side derivations
/// route their constant products through.
pub type Mul<'a> = &'a mut dyn FnMut(Fr, Fr) -> Fr;

/// The uncounted multiplication (prover side, tests).
pub fn plain(left: Fr, right: Fr) -> Fr {
    left * right
}

/// `1, root, root², …` (`count` powers) with the products observed.
pub fn powers_with(root: Fr, count: usize, mul: Mul<'_>) -> Vec<Fr> {
    let mut out = Vec::with_capacity(count);
    let mut power = Fr::from_u64(1);
    for i in 0..count {
        out.push(power);
        if i + 1 < count {
            power = if i == 0 { root } else { mul(power, root) };
        }
    }
    out
}

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

    pub fn scale_with(mut self, factor: Fr, mul: Mul<'_>) -> Self {
        self.constant = mul(self.constant, factor);
        for (_, weight) in &mut self.weights {
            *weight = mul(*weight, factor);
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

/// `−form`, no field multiplication.
impl std::ops::Neg for AffineForm {
    type Output = Self;

    fn neg(mut self) -> Self {
        self.constant = -self.constant;
        for (_, weight) in &mut self.weights {
            *weight = -*weight;
        }
        self
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
