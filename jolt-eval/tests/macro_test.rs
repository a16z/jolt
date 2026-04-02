use jolt_eval::invariant::{Invariant, InvariantViolation};

// ---------------------------------------------------------------------------
// AlwaysPass: trivial invariant to test macro synthesis
// ---------------------------------------------------------------------------

#[jolt_eval_macros::invariant(Test, Fuzz, RedTeam)]
#[derive(Default)]
pub struct AlwaysPassInvariant;

impl Invariant for AlwaysPassInvariant {
    type Setup = ();
    type Input = u8;

    fn name(&self) -> &str {
        "always_pass"
    }
    fn description(&self) -> String {
        "Trivial invariant that always passes — used to test macro synthesis.".to_string()
    }
    fn setup(&self) -> Self::Setup {}
    fn check(&self, _: &(), _input: u8) -> Result<(), InvariantViolation> {
        Ok(())
    }
    fn seed_corpus(&self) -> Vec<u8> {
        vec![0, 1, 42, 128, 255]
    }
}

// ---------------------------------------------------------------------------
// BoundsCheck: uses a struct Input type
// ---------------------------------------------------------------------------

#[derive(
    Debug,
    Clone,
    jolt_eval::arbitrary::Arbitrary,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
pub struct RangeInput {
    pub lo: u32,
    pub hi: u32,
}

#[jolt_eval_macros::invariant(Test, Fuzz)]
#[derive(Default)]
pub struct BoundsCheckInvariant;

impl Invariant for BoundsCheckInvariant {
    type Setup = ();
    type Input = RangeInput;

    fn name(&self) -> &str {
        "bounds_check"
    }
    fn description(&self) -> String {
        "Checks that max(lo,hi) >= min(lo,hi).".to_string()
    }
    fn setup(&self) -> Self::Setup {}
    fn check(&self, _: &(), input: RangeInput) -> Result<(), InvariantViolation> {
        let lo = input.lo.min(input.hi);
        let hi = input.lo.max(input.hi);
        if hi >= lo {
            Ok(())
        } else {
            Err(InvariantViolation::new("max < min — impossible"))
        }
    }
    fn seed_corpus(&self) -> Vec<RangeInput> {
        vec![
            RangeInput { lo: 0, hi: 0 },
            RangeInput {
                lo: 0,
                hi: u32::MAX,
            },
            RangeInput {
                lo: u32::MAX,
                hi: 0,
            },
            RangeInput { lo: 100, hi: 50 },
        ]
    }
}

// ===========================================================================
// The #[test] functions `seed_corpus` and `random_inputs` inside the
// generated `*_synthesized` modules are auto-discovered by nextest.
// ===========================================================================
