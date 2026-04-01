use enumset::EnumSet;
use jolt_eval::invariant::{Invariant, InvariantViolation, SynthesisTarget};

// ---------------------------------------------------------------------------
// AlwaysPass: exercises Test + RedTeam synthesis targets
// ---------------------------------------------------------------------------

#[jolt_eval_macros::invariant(targets = [Test, RedTeam])]
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
    fn targets(&self) -> EnumSet<SynthesisTarget> {
        SynthesisTarget::Test | SynthesisTarget::Fuzz | SynthesisTarget::RedTeam
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
// BoundsCheck: Test only, uses a struct Input type
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

#[jolt_eval_macros::invariant(targets = [Test])]
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
    fn targets(&self) -> EnumSet<SynthesisTarget> {
        SynthesisTarget::Test | SynthesisTarget::Fuzz
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

// ---------------------------------------------------------------------------
// RedTeamOnly: only the RedTeam target
// ---------------------------------------------------------------------------

#[jolt_eval_macros::invariant(targets = [RedTeam])]
#[derive(Default)]
pub struct RedTeamOnlyInvariant;

impl Invariant for RedTeamOnlyInvariant {
    type Setup = String;
    type Input = u16;

    fn name(&self) -> &str {
        "redteam_only"
    }
    fn description(&self) -> String {
        "An invariant that only generates a red-team description.".to_string()
    }
    fn targets(&self) -> EnumSet<SynthesisTarget> {
        SynthesisTarget::RedTeam.into()
    }
    fn setup(&self) -> String {
        "setup_value".to_string()
    }
    fn check(&self, setup: &String, _input: u16) -> Result<(), InvariantViolation> {
        if setup.is_empty() {
            Err(InvariantViolation::new("empty setup"))
        } else {
            Ok(())
        }
    }
    fn seed_corpus(&self) -> Vec<u16> {
        vec![0, 1000, u16::MAX]
    }
}

// ===========================================================================
// Tests that verify the macro-generated functions exist and work correctly
// ===========================================================================

// --- Red-team description functions ---

#[test]
fn redteam_always_pass_description() {
    let desc = always_pass_invariant_redteam_description();
    assert!(
        desc.contains("always passes"),
        "Expected description to mention 'always passes', got: {desc}"
    );
}

#[test]
fn redteam_only_description() {
    let desc = red_team_only_invariant_redteam_description();
    assert!(
        desc.contains("red-team description"),
        "Expected description to mention 'red-team description', got: {desc}"
    );
}

// --- Synthesized test modules are auto-discovered by nextest ---
// The #[test] functions `seed_corpus` and `random_inputs` inside the
// generated `*_synthesized` modules are run automatically. We verify
// their presence indirectly: if `cargo nextest run` reports them, the
// macro is working.
