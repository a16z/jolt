#![allow(non_snake_case)]

use arbitrary::Arbitrary;
use enumset::EnumSet;

use ark_bn254::Fr;
use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::split_eq_poly::GruenSplitEqPolynomial;

use super::{Invariant, InvariantViolation, SynthesisTarget};

type Challenge = <Fr as JoltField>::Challenge;

/// Input for the split-eq bind invariants: a number of variables and a
/// seed from which we derive all challenge values deterministically.
#[derive(Debug, Clone, Arbitrary, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SplitEqBindInput {
    /// Number of variables (clamped to 2..=20 in check).
    pub num_vars: u8,
    /// Seed bytes used to derive challenge values via simple hashing.
    pub seed: [u8; 32],
}

fn challenges_from_seed(seed: &[u8; 32], count: usize) -> Vec<Challenge> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::from_seed(*seed);
    (0..count).map(|_| Challenge::random(&mut rng)).collect()
}

// ── LowToHigh ────────────────────────────────────────────────────────

#[jolt_eval_macros::invariant]
#[derive(Default)]
pub struct SplitEqBindLowHighInvariant;

impl Invariant for SplitEqBindLowHighInvariant {
    type Setup = ();
    type Input = SplitEqBindInput;

    fn name(&self) -> &str {
        "split_eq_bind_low_high"
    }

    fn description(&self) -> String {
        "GruenSplitEqPolynomial::bind (LowToHigh) must match \
         DensePolynomial::bound_poly_var_bot at every round."
            .to_string()
    }

    fn targets(&self) -> EnumSet<SynthesisTarget> {
        SynthesisTarget::Test | SynthesisTarget::Fuzz
    }

    fn setup(&self) {}

    fn check(&self, _setup: &(), input: SplitEqBindInput) -> Result<(), InvariantViolation> {
        let num_vars = (input.num_vars as usize).clamp(2, 20);
        let challenges = challenges_from_seed(&input.seed, 2 * num_vars);
        let (w, rs) = challenges.split_at(num_vars);

        let mut regular_eq = DensePolynomial::<Fr>::new(EqPolynomial::evals(w));
        let mut split_eq = GruenSplitEqPolynomial::<Fr>::new(w, BindingOrder::LowToHigh);

        let merged = split_eq.merge();
        if regular_eq.Z[..regular_eq.len()] != merged.Z[..merged.len()] {
            return Err(InvariantViolation::with_details(
                "Initial merge mismatch (LowToHigh)",
                format!("num_vars={num_vars}"),
            ));
        }

        for (round, r) in rs.iter().enumerate() {
            regular_eq.bound_poly_var_bot(r);
            split_eq.bind(*r);

            let merged = split_eq.merge();
            if regular_eq.Z[..regular_eq.len()] != merged.Z[..merged.len()] {
                return Err(InvariantViolation::with_details(
                    "Bind mismatch (LowToHigh)",
                    format!("num_vars={num_vars}, round={round}"),
                ));
            }
        }

        Ok(())
    }

    fn seed_corpus(&self) -> Vec<SplitEqBindInput> {
        vec![
            SplitEqBindInput {
                num_vars: 2,
                seed: [0u8; 32],
            },
            SplitEqBindInput {
                num_vars: 10,
                seed: [1u8; 32],
            },
            SplitEqBindInput {
                num_vars: 17,
                seed: [42u8; 32],
            },
        ]
    }
}

// ── HighToLow ────────────────────────────────────────────────────────

#[jolt_eval_macros::invariant]
#[derive(Default)]
pub struct SplitEqBindHighLowInvariant;

impl Invariant for SplitEqBindHighLowInvariant {
    type Setup = ();
    type Input = SplitEqBindInput;

    fn name(&self) -> &str {
        "split_eq_bind_high_low"
    }

    fn description(&self) -> String {
        "GruenSplitEqPolynomial::bind (HighToLow) must match \
         DensePolynomial::bound_poly_var_top at every round."
            .to_string()
    }

    fn targets(&self) -> EnumSet<SynthesisTarget> {
        SynthesisTarget::Test | SynthesisTarget::Fuzz
    }

    fn setup(&self) {}

    fn check(&self, _setup: &(), input: SplitEqBindInput) -> Result<(), InvariantViolation> {
        let num_vars = (input.num_vars as usize).clamp(2, 20);
        let challenges = challenges_from_seed(&input.seed, 2 * num_vars);
        let (w, rs) = challenges.split_at(num_vars);

        let mut regular_eq = DensePolynomial::<Fr>::new(EqPolynomial::evals(w));
        let mut split_eq = GruenSplitEqPolynomial::<Fr>::new(w, BindingOrder::HighToLow);

        let merged = split_eq.merge();
        if regular_eq.Z[..regular_eq.len()] != merged.Z[..merged.len()] {
            return Err(InvariantViolation::with_details(
                "Initial merge mismatch (HighToLow)",
                format!("num_vars={num_vars}"),
            ));
        }

        for (round, r) in rs.iter().enumerate() {
            regular_eq.bound_poly_var_top(r);
            split_eq.bind(*r);

            let merged = split_eq.merge();
            if regular_eq.Z[..regular_eq.len()] != merged.Z[..merged.len()] {
                return Err(InvariantViolation::with_details(
                    "Bind mismatch (HighToLow)",
                    format!("num_vars={num_vars}, round={round}"),
                ));
            }
        }

        Ok(())
    }

    fn seed_corpus(&self) -> Vec<SplitEqBindInput> {
        vec![
            SplitEqBindInput {
                num_vars: 2,
                seed: [0u8; 32],
            },
            SplitEqBindInput {
                num_vars: 10,
                seed: [1u8; 32],
            },
            SplitEqBindInput {
                num_vars: 17,
                seed: [42u8; 32],
            },
        ]
    }
}
