#![allow(non_snake_case)]

use arbitrary::{Arbitrary, Unstructured};

use jolt_field::Fr;
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial};

use super::{CheckError, Invariant, InvariantViolation};

type Challenge = Fr;

/// Input for the split-eq bind invariants.
///
/// `w` are the initial eq-polynomial challenges, `rs` are the binding
/// round challenges. Stored as `u128` for serde/Arbitrary compatibility;
/// converted to `Challenge` via `From<u128>` in the check methods.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SplitEqBindInput {
    pub w: Vec<u128>,
    pub rs: Vec<u128>,
}

impl<'a> Arbitrary<'a> for SplitEqBindInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let num_vars = u.int_in_range(2u8..=16)? as usize;
        let w: Vec<u128> = (0..num_vars)
            .map(|_| u.arbitrary())
            .collect::<arbitrary::Result<_>>()?;
        let rs: Vec<u128> = (0..num_vars)
            .map(|_| u.arbitrary())
            .collect::<arbitrary::Result<_>>()?;
        Ok(Self { w, rs })
    }
}

fn to_challenges(vals: &[u128]) -> Vec<Challenge> {
    vals.iter().copied().map(Challenge::from).collect()
}

// ── LowToHigh ────────────────────────────────────────────────────────

#[jolt_eval_macros::invariant(Test, Fuzz, RedTeam)]
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
         Polynomial::bind_with_order at every round."
            .to_string()
    }

    fn setup(&self) {}

    fn check(&self, _setup: &(), input: SplitEqBindInput) -> Result<(), CheckError> {
        if input.w.len() < 2 {
            return Err(CheckError::InvalidInput("w.len() < 2".into()));
        }
        let w = to_challenges(&input.w);
        let rs = to_challenges(&input.rs);
        let num_vars = w.len();

        let mut regular_eq = Polynomial::<Fr>::new(EqPolynomial::evals(&w, None));
        let mut split_eq = GruenSplitEqPolynomial::<Fr>::new(&w, BindingOrder::LowToHigh);

        let merged = split_eq.merge();
        if regular_eq.evals() != merged.evals() {
            return Err(CheckError::Violation(InvariantViolation::with_details(
                "Initial merge mismatch (LowToHigh)",
                format!("num_vars={num_vars}"),
            )));
        }

        for (round, r) in rs.iter().enumerate() {
            regular_eq.bind_with_order(*r, BindingOrder::LowToHigh);
            split_eq.bind(*r);

            let merged = split_eq.merge();
            if regular_eq.evals() != merged.evals() {
                return Err(CheckError::Violation(InvariantViolation::with_details(
                    "Bind mismatch (LowToHigh)",
                    format!("num_vars={num_vars}, round={round}"),
                )));
            }
        }

        Ok(())
    }

    fn seed_corpus(&self) -> Vec<SplitEqBindInput> {
        vec![
            SplitEqBindInput {
                w: vec![0, 1],
                rs: vec![2, 3],
            },
            SplitEqBindInput {
                w: (0..10).collect(),
                rs: (10..20).collect(),
            },
            SplitEqBindInput {
                w: (0..17).map(|i| u128::MAX - i).collect(),
                rs: (0..17).map(|i| i * 1000).collect(),
            },
        ]
    }
}

// ── HighToLow ────────────────────────────────────────────────────────

#[jolt_eval_macros::invariant(Test, Fuzz)]
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
         Polynomial::bind_with_order at every round."
            .to_string()
    }

    fn setup(&self) {}

    fn check(&self, _setup: &(), input: SplitEqBindInput) -> Result<(), CheckError> {
        if input.w.len() < 2 {
            return Err(CheckError::InvalidInput("w.len() < 2".into()));
        }
        let w = to_challenges(&input.w);
        let rs = to_challenges(&input.rs);
        let num_vars = w.len();

        let mut regular_eq = Polynomial::<Fr>::new(EqPolynomial::evals(&w, None));
        let mut split_eq = GruenSplitEqPolynomial::<Fr>::new(&w, BindingOrder::HighToLow);

        let merged = split_eq.merge();
        if regular_eq.evals() != merged.evals() {
            return Err(CheckError::Violation(InvariantViolation::with_details(
                "Initial merge mismatch (HighToLow)",
                format!("num_vars={num_vars}"),
            )));
        }

        for (round, r) in rs.iter().enumerate() {
            regular_eq.bind_with_order(*r, BindingOrder::HighToLow);
            split_eq.bind(*r);

            let merged = split_eq.merge();
            if regular_eq.evals() != merged.evals() {
                return Err(CheckError::Violation(InvariantViolation::with_details(
                    "Bind mismatch (HighToLow)",
                    format!("num_vars={num_vars}, round={round}"),
                )));
            }
        }

        Ok(())
    }

    fn seed_corpus(&self) -> Vec<SplitEqBindInput> {
        vec![
            SplitEqBindInput {
                w: vec![0, 1],
                rs: vec![2, 3],
            },
            SplitEqBindInput {
                w: (0..10).collect(),
                rs: (10..20).collect(),
            },
            SplitEqBindInput {
                w: (0..16).map(|i| u128::MAX - i).collect(),
                rs: (0..16).map(|i| i * 1000).collect(),
            },
        ]
    }
}
