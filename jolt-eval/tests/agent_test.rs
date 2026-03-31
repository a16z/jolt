use std::path::Path;

use enumset::EnumSet;
use jolt_eval::agent::{AgentError, AgentHarness, AgentResponse, MockAgent};
use jolt_eval::invariant::synthesis::redteam::{auto_redteam, RedTeamConfig, RedTeamResult};
use jolt_eval::invariant::{Invariant, InvariantViolation, SynthesisTarget};

// =========================================================================
// Test invariants
// =========================================================================

/// Always passes -- the red-team loop should never find a violation.
struct AlwaysPassInvariant;
impl Invariant for AlwaysPassInvariant {
    type Setup = ();
    type Input = u8;
    fn name(&self) -> &str {
        "always_pass"
    }
    fn description(&self) -> String {
        "This invariant always passes.".into()
    }
    fn targets(&self) -> EnumSet<SynthesisTarget> {
        SynthesisTarget::Test | SynthesisTarget::RedTeam
    }
    fn setup(&self) {}
    fn check(&self, _: &(), _: u8) -> Result<(), InvariantViolation> {
        Ok(())
    }
    fn seed_corpus(&self) -> Vec<u8> {
        vec![0, 1, 255]
    }
}

/// Always fails -- the red-team loop should find a violation immediately.
struct AlwaysFailInvariant;
impl Invariant for AlwaysFailInvariant {
    type Setup = ();
    type Input = u8;
    fn name(&self) -> &str {
        "always_fail"
    }
    fn description(&self) -> String {
        "This invariant always fails.".into()
    }
    fn targets(&self) -> EnumSet<SynthesisTarget> {
        SynthesisTarget::Test | SynthesisTarget::RedTeam
    }
    fn setup(&self) {}
    fn check(&self, _: &(), input: u8) -> Result<(), InvariantViolation> {
        Err(InvariantViolation::new(format!("always fails ({input})")))
    }
    fn seed_corpus(&self) -> Vec<u8> {
        vec![42]
    }
}

/// Fails only when the input is 0 -- tests that fuzz inputs can trigger it.
struct FailsOnZeroInvariant;
impl Invariant for FailsOnZeroInvariant {
    type Setup = ();
    type Input = u8;
    fn name(&self) -> &str {
        "fails_on_zero"
    }
    fn description(&self) -> String {
        "Fails when input is 0.".into()
    }
    fn targets(&self) -> EnumSet<SynthesisTarget> {
        SynthesisTarget::Test | SynthesisTarget::RedTeam
    }
    fn setup(&self) {}
    fn check(&self, _: &(), input: u8) -> Result<(), InvariantViolation> {
        if input == 0 {
            Err(InvariantViolation::new("input was zero"))
        } else {
            Ok(())
        }
    }
    fn seed_corpus(&self) -> Vec<u8> {
        vec![1, 2, 3] // seed corpus avoids 0
    }
}

// =========================================================================
// MockAgent tests
// =========================================================================

#[test]
fn mock_always_ok_returns_text() {
    let agent = MockAgent::always_ok("hello world");
    let resp = agent.invoke(Path::new("/tmp"), "test prompt").unwrap();
    assert_eq!(resp.text, "hello world");
    assert!(resp.diff.is_none());
}

#[test]
fn mock_always_err_returns_error() {
    let agent = MockAgent::always_err("boom");
    let err = agent.invoke(Path::new("/tmp"), "test").unwrap_err();
    assert_eq!(err.message, "boom");
}

#[test]
fn mock_records_prompts() {
    let agent = MockAgent::always_ok("ok");
    agent.invoke(Path::new("/tmp"), "prompt 1").unwrap();
    agent.invoke(Path::new("/tmp"), "prompt 2").unwrap();
    agent.invoke(Path::new("/tmp"), "prompt 3").unwrap();

    let prompts = agent.recorded_prompts();
    assert_eq!(prompts.len(), 3);
    assert_eq!(prompts[0], "prompt 1");
    assert_eq!(prompts[1], "prompt 2");
    assert_eq!(prompts[2], "prompt 3");
}

#[test]
fn mock_always_ok_repeats_indefinitely() {
    let agent = MockAgent::always_ok("same");
    for _ in 0..100 {
        let resp = agent.invoke(Path::new("/tmp"), "x").unwrap();
        assert_eq!(resp.text, "same");
    }
}

#[test]
fn mock_always_err_repeats_indefinitely() {
    let agent = MockAgent::always_err("fail");
    for _ in 0..100 {
        let err = agent.invoke(Path::new("/tmp"), "x").unwrap_err();
        assert_eq!(err.message, "fail");
    }
}

#[test]
fn mock_from_responses_returns_in_order() {
    let agent = MockAgent::from_responses(vec![
        Ok(AgentResponse {
            text: "first".into(),
            diff: None,
        }),
        Ok(AgentResponse {
            text: "second".into(),
            diff: Some("diff".into()),
        }),
        Err(AgentError::new("third fails")),
    ]);

    let r1 = agent.invoke(Path::new("/tmp"), "a").unwrap();
    assert_eq!(r1.text, "first");
    assert!(r1.diff.is_none());

    let r2 = agent.invoke(Path::new("/tmp"), "b").unwrap();
    assert_eq!(r2.text, "second");
    assert_eq!(r2.diff.as_deref(), Some("diff"));

    let r3 = agent.invoke(Path::new("/tmp"), "c").unwrap_err();
    assert_eq!(r3.message, "third fails");
}

#[test]
fn mock_from_responses_last_entry_repeats() {
    let agent = MockAgent::from_responses(vec![
        Ok(AgentResponse {
            text: "first".into(),
            diff: None,
        }),
        Ok(AgentResponse {
            text: "last".into(),
            diff: None,
        }),
    ]);

    agent.invoke(Path::new("/tmp"), "a").unwrap();
    let r2 = agent.invoke(Path::new("/tmp"), "b").unwrap();
    assert_eq!(r2.text, "last");
    // After exhausting queue, last response repeats
    let r3 = agent.invoke(Path::new("/tmp"), "c").unwrap();
    assert_eq!(r3.text, "last");
}

#[test]
fn mock_with_diff() {
    let agent = MockAgent::from_responses(vec![Ok(AgentResponse {
        text: "I optimized the code".into(),
        diff: Some("--- a/foo\n+++ b/foo\n@@ ...\n-old\n+new".into()),
    })]);

    let resp = agent.invoke(Path::new("/tmp"), "optimize").unwrap();
    assert!(resp.diff.is_some());
    assert!(resp.diff.unwrap().contains("+new"));
}

// =========================================================================
// auto_redteam tests with MockAgent
// =========================================================================

#[test]
fn redteam_no_violation_when_invariant_always_passes() {
    let invariant = AlwaysPassInvariant;
    let agent = MockAgent::always_ok("I analyzed the code and found nothing.");
    let config = RedTeamConfig {
        num_iterations: 3,
        num_fuzz_per_iteration: 5,
    };

    let result = auto_redteam(&invariant, &config, &agent, Path::new("/tmp"));

    match result {
        RedTeamResult::NoViolation { attempts } => {
            assert_eq!(attempts.len(), 3);
            for a in &attempts {
                assert_eq!(a.failure_reason, "Invariant check passed for all inputs");
            }
        }
        RedTeamResult::Violation { .. } => {
            panic!("Expected no violation for AlwaysPassInvariant");
        }
    }

    // Agent should have been invoked exactly 3 times
    assert_eq!(agent.recorded_prompts().len(), 3);
}

#[test]
fn redteam_finds_violation_immediately_when_invariant_always_fails() {
    let invariant = AlwaysFailInvariant;
    let agent = MockAgent::always_ok("Trying something.");
    let config = RedTeamConfig {
        num_iterations: 10,
        num_fuzz_per_iteration: 0, // seed corpus alone triggers failure
    };

    let result = auto_redteam(&invariant, &config, &agent, Path::new("/tmp"));

    match result {
        RedTeamResult::Violation { error, .. } => {
            assert!(error.contains("always fails"));
        }
        RedTeamResult::NoViolation { .. } => {
            panic!("Expected violation for AlwaysFailInvariant");
        }
    }

    // Should stop after first iteration (found violation)
    assert_eq!(agent.recorded_prompts().len(), 1);
}

#[test]
fn redteam_finds_violation_via_fuzz_inputs() {
    let invariant = FailsOnZeroInvariant;
    let agent = MockAgent::always_ok("Analyzing...");
    let config = RedTeamConfig {
        num_iterations: 3,
        // High fuzz count makes it very likely a 0 byte appears
        num_fuzz_per_iteration: 1000,
    };

    let result = auto_redteam(&invariant, &config, &agent, Path::new("/tmp"));

    // With 1000 random u8 inputs per iteration, the chance of never hitting 0
    // across 3 iterations is (255/256)^3000 ≈ 0.  So we expect a violation.
    match result {
        RedTeamResult::Violation { error, .. } => {
            assert!(error.contains("zero"));
        }
        RedTeamResult::NoViolation { .. } => {
            panic!("Expected violation for FailsOnZeroInvariant with high fuzz count");
        }
    }
}

#[test]
fn redteam_handles_agent_errors_gracefully() {
    let invariant = AlwaysPassInvariant;
    let agent = MockAgent::always_err("network timeout");
    let config = RedTeamConfig {
        num_iterations: 3,
        num_fuzz_per_iteration: 0,
    };

    let result = auto_redteam(&invariant, &config, &agent, Path::new("/tmp"));

    match result {
        RedTeamResult::NoViolation { attempts } => {
            assert_eq!(attempts.len(), 3);
            for a in &attempts {
                assert_eq!(a.approach, "Agent invocation failed");
                assert!(a.failure_reason.contains("network timeout"));
            }
        }
        RedTeamResult::Violation { .. } => {
            panic!("Expected no violation when agent always errors");
        }
    }
}

#[test]
fn redteam_prompt_includes_invariant_description() {
    let invariant = AlwaysPassInvariant;
    let agent = MockAgent::always_ok("ok");
    let config = RedTeamConfig {
        num_iterations: 1,
        num_fuzz_per_iteration: 0,
    };

    auto_redteam(&invariant, &config, &agent, Path::new("/tmp"));

    let prompts = agent.recorded_prompts();
    assert_eq!(prompts.len(), 1);
    assert!(prompts[0].contains("This invariant always passes."));
    assert!(prompts[0].contains("VIOLATES"));
}

#[test]
fn redteam_prompt_includes_failed_attempts_after_first_iteration() {
    let invariant = AlwaysPassInvariant;
    let agent = MockAgent::always_ok("I tried X but it didn't work.");
    let config = RedTeamConfig {
        num_iterations: 3,
        num_fuzz_per_iteration: 0,
    };

    auto_redteam(&invariant, &config, &agent, Path::new("/tmp"));

    let prompts = agent.recorded_prompts();
    assert_eq!(prompts.len(), 3);

    // First prompt should NOT contain "Previous Failed Attempts"
    assert!(!prompts[0].contains("Previous Failed Attempts"));

    // Second prompt should contain the first attempt's approach
    assert!(prompts[1].contains("Previous Failed Attempts"));
    assert!(prompts[1].contains("I tried X but it didn't work."));

    // Third prompt should contain both prior attempts
    assert!(prompts[2].contains("Iteration 1"));
    assert!(prompts[2].contains("Iteration 2"));
}

#[test]
fn redteam_zero_iterations_returns_immediately() {
    let invariant = AlwaysPassInvariant;
    let agent = MockAgent::always_ok("should not be called");
    let config = RedTeamConfig {
        num_iterations: 0,
        num_fuzz_per_iteration: 0,
    };

    let result = auto_redteam(&invariant, &config, &agent, Path::new("/tmp"));

    match result {
        RedTeamResult::NoViolation { attempts } => {
            assert!(attempts.is_empty());
        }
        _ => panic!("Expected NoViolation with empty attempts"),
    }

    assert!(agent.recorded_prompts().is_empty());
}

#[test]
fn redteam_mixed_agent_responses() {
    let invariant = AlwaysPassInvariant;
    let agent = MockAgent::from_responses(vec![
        Ok(AgentResponse {
            text: "first try".into(),
            diff: None,
        }),
        Err(AgentError::new("transient error")),
        Ok(AgentResponse {
            text: "third try".into(),
            diff: None,
        }),
    ]);
    let config = RedTeamConfig {
        num_iterations: 3,
        num_fuzz_per_iteration: 0,
    };

    let result = auto_redteam(&invariant, &config, &agent, Path::new("/tmp"));

    match result {
        RedTeamResult::NoViolation { attempts } => {
            assert_eq!(attempts.len(), 3);
            assert!(attempts[0].approach.contains("first try"));
            assert_eq!(attempts[1].approach, "Agent invocation failed");
            assert!(attempts[2].approach.contains("third try"));
        }
        _ => panic!("Expected NoViolation"),
    }
}

// =========================================================================
// AgentHarness trait object tests
// =========================================================================

#[test]
fn agent_harness_is_object_safe() {
    // Verify we can use AgentHarness as a trait object
    let agent: Box<dyn AgentHarness> = Box::new(MockAgent::always_ok("hi"));
    let resp = agent.invoke(Path::new("/tmp"), "hello").unwrap();
    assert_eq!(resp.text, "hi");
}

#[test]
fn agent_harness_works_with_arc() {
    use std::sync::Arc;
    let agent: Arc<dyn AgentHarness> = Arc::new(MockAgent::always_ok("shared"));
    let resp = agent.invoke(Path::new("/tmp"), "test").unwrap();
    assert_eq!(resp.text, "shared");
}

/// A custom multi-agent harness that fans out to N agents and returns the
/// first successful response. Demonstrates the trait's extensibility.
struct FirstSuccessHarness {
    agents: Vec<Box<dyn AgentHarness>>,
}

impl AgentHarness for FirstSuccessHarness {
    fn invoke(&self, repo_dir: &Path, prompt: &str) -> Result<AgentResponse, AgentError> {
        for agent in &self.agents {
            if let Ok(resp) = agent.invoke(repo_dir, prompt) {
                return Ok(resp);
            }
        }
        Err(AgentError::new("All agents failed"))
    }
}

#[test]
fn custom_multi_agent_harness() {
    let harness = FirstSuccessHarness {
        agents: vec![
            Box::new(MockAgent::always_err("agent 1 down")),
            Box::new(MockAgent::always_err("agent 2 down")),
            Box::new(MockAgent::always_ok("agent 3 succeeded")),
        ],
    };

    let resp = harness.invoke(Path::new("/tmp"), "test").unwrap();
    assert_eq!(resp.text, "agent 3 succeeded");
}

#[test]
fn custom_multi_agent_all_fail() {
    let harness = FirstSuccessHarness {
        agents: vec![
            Box::new(MockAgent::always_err("nope")),
            Box::new(MockAgent::always_err("nope")),
        ],
    };

    let err = harness.invoke(Path::new("/tmp"), "test").unwrap_err();
    assert_eq!(err.message, "All agents failed");
}

#[test]
fn custom_harness_plugs_into_auto_redteam() {
    let harness = FirstSuccessHarness {
        agents: vec![
            Box::new(MockAgent::always_err("agent 1 down")),
            Box::new(MockAgent::always_ok("agent 2 found nothing")),
        ],
    };

    let invariant = AlwaysPassInvariant;
    let config = RedTeamConfig {
        num_iterations: 2,
        num_fuzz_per_iteration: 0,
    };

    let result = auto_redteam(&invariant, &config, &harness, Path::new("/tmp"));

    match result {
        RedTeamResult::NoViolation { attempts } => {
            assert_eq!(attempts.len(), 2);
            // The harness should have used agent 2's response
            assert!(attempts[0].approach.contains("agent 2 found nothing"));
        }
        _ => panic!("Expected NoViolation"),
    }
}
