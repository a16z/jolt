//! BlindFold Zero-Knowledge Protocol
//!
//! This module implements the BlindFold protocol for making Jolt's sumcheck proofs
//! zero-knowledge. The core idea is to encode sumcheck verification into a small R1CS
//! circuit and apply a folding scheme.
//!
//! The verifier R1CS only checks O(log n) algebraic relations - the sumcheck round
//! consistency checks - rather than the full O(n) computation.
//!
//! Key components:
//! - [`VerifierR1CS`]: Sparse R1CS matrices for sumcheck verification
//! - [`SumcheckRoundGadget`]: Constraint generation for a single sumcheck round
//! - [`BlindFoldWitness`]: Witness assignment for the verifier circuit

mod r1cs;
mod witness;

pub use r1cs::{SparseR1CSMatrix, VerifierR1CS, VerifierR1CSBuilder};
pub use witness::{BlindFoldWitness, RoundWitness, StageWitness};

use crate::field::JoltField;

/// Configuration for a single sumcheck stage
#[derive(Clone, Debug)]
pub struct StageConfig {
    /// Number of sumcheck rounds in this stage
    pub num_rounds: usize,
    /// Degree of the round polynomials (typically 3 for cubic)
    pub poly_degree: usize,
}

impl StageConfig {
    pub fn new(num_rounds: usize, poly_degree: usize) -> Self {
        Self {
            num_rounds,
            poly_degree,
        }
    }
}

/// Jolt's 6 sumcheck stages configuration
pub fn jolt_stage_configs(num_rounds_per_stage: usize) -> [StageConfig; 6] {
    [
        StageConfig::new(num_rounds_per_stage, 3), // Stage 1: Spartan Outer
        StageConfig::new(num_rounds_per_stage, 3), // Stage 2: Product Virtualization
        StageConfig::new(num_rounds_per_stage, 3), // Stage 3: Instruction Constraints
        StageConfig::new(num_rounds_per_stage, 3), // Stage 4: Register + RAM
        StageConfig::new(num_rounds_per_stage, 3), // Stage 5: Value + Lookup
        StageConfig::new(num_rounds_per_stage, 3), // Stage 6: One-Hot + Hamming
    ]
}

/// Variable index in the witness vector Z
///
/// Z is laid out as: [1, public_inputs..., witness...]
/// - Index 0 is always the constant 1
/// - Public inputs follow (challenges, initial claim)
/// - Witness variables come last (coefficients, intermediates)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Variable(pub usize);

impl Variable {
    /// The constant 1 variable (always at index 0)
    pub const ONE: Variable = Variable(0);

    /// Create a new variable with the given index
    pub const fn new(idx: usize) -> Self {
        Variable(idx)
    }

    /// Get the raw index
    pub const fn index(&self) -> usize {
        self.0
    }
}

/// A term in a linear combination: coefficient * variable
#[derive(Clone, Copy, Debug)]
pub struct Term<F> {
    pub var: Variable,
    pub coeff: F,
}

impl<F: JoltField> Term<F> {
    pub fn new(var: Variable, coeff: F) -> Self {
        Self { var, coeff }
    }

    pub fn one(var: Variable) -> Self {
        Self::new(var, F::one())
    }

    pub fn neg_one(var: Variable) -> Self {
        Self::new(var, -F::one())
    }
}

/// A linear combination of variables: Σ coeff_i * var_i
#[derive(Clone, Debug, Default)]
pub struct LinearCombination<F> {
    pub terms: Vec<Term<F>>,
}

impl<F: JoltField> LinearCombination<F> {
    pub fn new() -> Self {
        Self { terms: Vec::new() }
    }

    pub fn constant(value: F) -> Self {
        Self {
            terms: vec![Term::new(Variable::ONE, value)],
        }
    }

    pub fn variable(var: Variable) -> Self {
        Self {
            terms: vec![Term::one(var)],
        }
    }

    pub fn add_term(mut self, var: Variable, coeff: F) -> Self {
        self.terms.push(Term::new(var, coeff));
        self
    }

    pub fn add_var(mut self, var: Variable) -> Self {
        self.terms.push(Term::one(var));
        self
    }

    pub fn sub_var(mut self, var: Variable) -> Self {
        self.terms.push(Term::neg_one(var));
        self
    }

    /// Evaluate the linear combination given the witness vector Z
    pub fn evaluate(&self, z: &[F]) -> F {
        self.terms
            .iter()
            .map(|term| term.coeff * z[term.var.index()])
            .sum()
    }
}

/// An R1CS constraint: (A · Z) * (B · Z) = (C · Z)
#[derive(Clone, Debug)]
pub struct Constraint<F> {
    pub a: LinearCombination<F>,
    pub b: LinearCombination<F>,
    pub c: LinearCombination<F>,
}

impl<F: JoltField> Constraint<F> {
    pub fn new(a: LinearCombination<F>, b: LinearCombination<F>, c: LinearCombination<F>) -> Self {
        Self { a, b, c }
    }

    /// Check if the constraint is satisfied by the witness vector Z
    pub fn is_satisfied(&self, z: &[F]) -> bool {
        let a_val = self.a.evaluate(z);
        let b_val = self.b.evaluate(z);
        let c_val = self.c.evaluate(z);
        a_val * b_val == c_val
    }
}
