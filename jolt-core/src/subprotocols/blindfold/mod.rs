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

mod folding;
mod output_constraint;
mod protocol;
mod r1cs;
mod relaxed_r1cs;
mod witness;

pub use folding::{compute_cross_term, sample_random_satisfying_pair};
pub use output_constraint::{
    InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource,
};
pub use protocol::{
    BlindFoldProof, BlindFoldProver, BlindFoldVerifier, BlindFoldVerifyError, FinalOutputInfo,
};
pub use r1cs::{SparseR1CSMatrix, VerifierR1CS, VerifierR1CSBuilder};
pub use relaxed_r1cs::{RelaxedR1CSInstance, RelaxedR1CSWitness};
pub use witness::{
    BlindFoldWitness, ExtraConstraintWitness, FinalOutputWitness, RoundWitness, StageWitness,
};

use crate::field::JoltField;

/// Configuration for final output binding at end of a chain.
///
/// Supports two modes:
/// 1. Simple linear: final_claim = Σⱼ αⱼ · yⱼ (legacy, num_evaluations only)
/// 2. General sum-of-products: output = Σᵢ coeffᵢ * ∏ⱼ factorᵢⱼ (uses constraint)
#[derive(Clone, Debug, Default)]
pub struct FinalOutputConfig {
    /// Number of batched polynomial evaluations in this final constraint.
    /// Used for simple linear constraints. Each evaluation yⱼ is a witness variable.
    pub num_evaluations: usize,

    /// General constraint description for sum-of-products form.
    /// When present, overrides the simple linear constraint.
    pub constraint: Option<OutputClaimConstraint>,

    /// Exact number of witness variables to allocate (verifier-only).
    /// When set, bypasses normal variable calculation.
    /// Used by verifier to match prover's R1CS structure without knowing constraint details.
    pub exact_num_witness_vars: Option<usize>,
}

impl FinalOutputConfig {
    pub fn new(num_evaluations: usize) -> Self {
        Self {
            num_evaluations,
            constraint: None,
            exact_num_witness_vars: None,
        }
    }

    /// Create a final output config with a general sum-of-products constraint.
    pub fn with_constraint(constraint: OutputClaimConstraint) -> Self {
        let num_evaluations = constraint.required_openings.len();
        Self {
            num_evaluations,
            constraint: Some(constraint),
            exact_num_witness_vars: None,
        }
    }

    /// Create a verifier-only config that allocates exactly the specified number of witness variables.
    /// Used by verifier to match prover's R1CS structure.
    pub fn verifier_placeholder(num_witness_vars: usize) -> Self {
        Self {
            num_evaluations: 0,
            constraint: None,
            exact_num_witness_vars: Some(num_witness_vars),
        }
    }
}

/// Configuration for a single sumcheck stage
#[derive(Clone, Debug)]
pub struct StageConfig {
    /// Number of sumcheck rounds in this stage
    pub num_rounds: usize,
    /// Degree of the round polynomials (typically 3 for cubic)
    pub poly_degree: usize,
    /// Whether this stage starts a new independent chain with its own initial claim.
    /// When true, the first round of this stage uses a separate initial_claim
    /// rather than chaining from the previous stage's last round.
    pub starts_new_chain: bool,
    /// Power sums for uni-skip sum constraint (if this is a uni-skip round).
    /// For standard sumcheck, this is None and uses `2*c0 + c1 + c2 + ... = claimed_sum`.
    /// For uni-skip, contains `[PowerSum[0], PowerSum[1], ...]` where
    /// `PowerSum[k] = Σ_{t in symmetric domain} t^k`.
    pub uniskip_power_sums: Option<Vec<i128>>,
    /// Final output binding configuration.
    /// If set, adds constraint at end of this stage's chain: final_claim = Σⱼ αⱼ · yⱼ
    pub final_output: Option<FinalOutputConfig>,
    /// Initial input binding configuration.
    /// If set, adds constraint at start of this stage: initial_claim = f(openings, challenges)
    /// Verifies that the input claim is correctly derived from previous sumcheck openings.
    pub initial_input: Option<FinalOutputConfig>,
}

impl StageConfig {
    pub fn new(num_rounds: usize, poly_degree: usize) -> Self {
        Self {
            num_rounds,
            poly_degree,
            starts_new_chain: false,
            uniskip_power_sums: None,
            final_output: None,
            initial_input: None,
        }
    }

    /// Create a stage config that starts a new independent chain.
    pub fn new_chain(num_rounds: usize, poly_degree: usize) -> Self {
        Self {
            num_rounds,
            poly_degree,
            starts_new_chain: true,
            uniskip_power_sums: None,
            final_output: None,
            initial_input: None,
        }
    }

    /// Create a uni-skip stage config.
    /// `power_sums` are precomputed: `PowerSum[k] = Σ_{t in symmetric domain} t^k`
    pub fn new_uniskip(poly_degree: usize, power_sums: Vec<i128>) -> Self {
        Self {
            num_rounds: 1,
            poly_degree,
            starts_new_chain: false,
            uniskip_power_sums: Some(power_sums),
            final_output: None,
            initial_input: None,
        }
    }

    /// Create a uni-skip stage config that starts a new chain.
    pub fn new_uniskip_chain(poly_degree: usize, power_sums: Vec<i128>) -> Self {
        Self {
            num_rounds: 1,
            poly_degree,
            starts_new_chain: true,
            uniskip_power_sums: Some(power_sums),
            final_output: None,
            initial_input: None,
        }
    }

    /// Set final output binding for this stage.
    /// Adds constraint: final_claim = Σⱼ αⱼ · yⱼ at end of this stage's chain.
    pub fn with_final_output(mut self, num_evaluations: usize) -> Self {
        self.final_output = Some(FinalOutputConfig::new(num_evaluations));
        self
    }

    /// Set final output binding with a general sum-of-products constraint.
    /// The constraint describes: output = Σᵢ coeffᵢ * ∏ⱼ factorᵢⱼ
    pub fn with_constraint(mut self, constraint: OutputClaimConstraint) -> Self {
        self.final_output = Some(FinalOutputConfig::with_constraint(constraint));
        self
    }

    /// Set initial input binding with a general sum-of-products constraint.
    /// The constraint describes: input_claim = Σᵢ coeffᵢ * ∏ⱼ factorᵢⱼ
    /// Verifies input claim is correctly derived from previous sumcheck openings.
    pub fn with_input_constraint(mut self, constraint: InputClaimConstraint) -> Self {
        self.initial_input = Some(FinalOutputConfig::with_constraint(constraint));
        self
    }

    /// Returns true if this is a uni-skip round.
    pub fn is_uniskip(&self) -> bool {
        self.uniskip_power_sums.is_some()
    }
}

/// Variable index in the witness vector Z
///
/// For relaxed R1CS, Z is laid out as: [u, public_inputs..., witness...]
/// - Index 0 is the scalar u (u=1 for non-relaxed, u=u1+r*u2 for folded)
/// - Public inputs follow (challenges, initial claim)
/// - Witness variables come last (coefficients, intermediates)
///
/// This layout allows proper folding: when Z' = Z1 + r*Z2, the u position
/// naturally becomes u1 + r*u2 = u', which is the folded scalar.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Variable(pub usize);

impl Variable {
    /// The u scalar variable (at index 0)
    /// For non-relaxed instances, u = 1. For folded instances, u = u1 + r*u2.
    /// In constraints, use this instead of constants.
    pub const U: Variable = Variable(0);

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

    /// Create a linear combination representing `value * u` where u is the relaxation scalar.
    /// For non-relaxed instances (u=1), this evaluates to `value`.
    /// For relaxed instances, this evaluates to `value * u`.
    pub fn constant(value: F) -> Self {
        Self {
            terms: vec![Term::new(Variable::U, value)],
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
