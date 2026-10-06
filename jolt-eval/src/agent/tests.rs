use std::collections::HashMap;
use std::path::Path;

use enumset::EnumSet;

use crate::agent::{AgentError, AgentResponse, MockAgent};
use crate::invariant::synthesis::redteam::{auto_redteam, RedTeamConfig, RedTeamResult};
use crate::invariant::{
    CheckError, Invariant, InvariantTargets, InvariantViolation, SynthesisTarget,
};
use crate::objective::objective_fn::ObjectiveFunction;
use crate::objective::optimize::{auto_optimize, OptimizeConfig, OptimizeEnv};
use crate::objective::{OptimizationObjective, HALSTEAD_BUGS, LLOC};

struct AlwaysPassInvariant;
impl InvariantTargets for AlwaysPassInvariant {
    fn targets(&self) -> EnumSet<SynthesisTarget> {
        SynthesisTarget::Test | SynthesisTarget::RedTeam
    }
}
impl Invariant for AlwaysPassInvariant {
    type Setup = ();
    type Input = u8;
    fn name(&self) -> &str {
        "always_pass"
    }
    fn description(&self) -> String {
        "This invariant always passes.".into()
    }
    fn setup(&self) {}
    fn check(&self, _: &(), _: u8) -> Result<(), CheckError> {
        Ok(())
    }
    fn seed_corpus(&self) -> Vec<u8> {
        vec![0, 1, 255]
    }
}

struct AlwaysFailInvariant;
impl InvariantTargets for AlwaysFailInvariant {
    fn targets(&self) -> EnumSet<SynthesisTarget> {
        SynthesisTarget::Test | SynthesisTarget::RedTeam
    }
}
impl Invariant for AlwaysFailInvariant {
    type Setup = ();
    type Input = u8;
    fn name(&self) -> &str {
        "always_fail"
    }
    fn description(&self) -> String {
        "This invariant always fails.".into()
    }
    fn setup(&self) {}
    fn check(&self, _: &(), input: u8) -> Result<(), CheckError> {
        Err(CheckError::Violation(InvariantViolation::new(format!(
            "always fails ({input})"
        ))))
    }
    fn seed_corpus(&self) -> Vec<u8> {
        vec![42]
    }
}

fn envelope(analysis: &str, counterexample: impl serde::Serialize) -> String {
    serde_json::json!({
        "analysis": analysis,
        "approach_summary": analysis,
        "counterexample": counterexample,
    })
    .to_string()
}

#[test]
fn redteam_no_violation_when_invariant_always_passes() {
    let invariant = AlwaysPassInvariant;
    let agent = MockAgent::always_ok(&envelope("I analyzed the code.", 42));
    let config = RedTeamConfig {
        num_iterations: 3,
        ..Default::default()
    };

    let result = auto_redteam(&invariant, &config, &agent, Path::new("/tmp"));

    match result {
        RedTeamResult::NoViolation { attempts } => {
            assert_eq!(attempts.len(), 3);
            for a in &attempts {
                assert!(a.failure_reason.contains("did not violate"));
            }
        }
        RedTeamResult::Violation { .. } => {
            panic!("Expected no violation for AlwaysPassInvariant");
        }
    }

    assert_eq!(agent.recorded_prompts().len(), 3);
}

#[test]
fn redteam_finds_violation_with_structured_response() {
    let invariant = AlwaysFailInvariant;
    let agent = MockAgent::always_ok(&envelope("I found a bug!", 99));
    let config = RedTeamConfig {
        num_iterations: 10,
        ..Default::default()
    };

    let result = auto_redteam(&invariant, &config, &agent, Path::new("/tmp"));

    match result {
        RedTeamResult::Violation {
            input_json, error, ..
        } => {
            assert_eq!(input_json, "99");
            assert!(error.contains("always fails"));
        }
        RedTeamResult::NoViolation { .. } => {
            panic!("Expected violation for AlwaysFailInvariant");
        }
    }

    assert_eq!(agent.recorded_prompts().len(), 1);
}

#[test]
fn redteam_handles_agent_errors_gracefully() {
    let invariant = AlwaysPassInvariant;
    let agent = MockAgent::always_err("network timeout");
    let config = RedTeamConfig {
        num_iterations: 3,
        ..Default::default()
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
fn redteam_handles_no_json_in_response() {
    let invariant = AlwaysPassInvariant;
    let agent = MockAgent::always_ok("I looked around but have no candidate to offer.");
    let config = RedTeamConfig {
        num_iterations: 1,
        ..Default::default()
    };

    let result = auto_redteam(&invariant, &config, &agent, Path::new("/tmp"));

    match result {
        RedTeamResult::NoViolation { attempts } => {
            assert_eq!(attempts.len(), 1);
            assert!(attempts[0]
                .failure_reason
                .contains("did not contain valid JSON"));
        }
        _ => panic!("Expected NoViolation"),
    }
}

#[test]
fn redteam_handles_invalid_counterexample_type() {
    let invariant = AlwaysPassInvariant;
    let agent = MockAgent::always_ok(&envelope("Here", "not_a_number"));
    let config = RedTeamConfig {
        num_iterations: 1,
        ..Default::default()
    };

    let result = auto_redteam(&invariant, &config, &agent, Path::new("/tmp"));

    match result {
        RedTeamResult::NoViolation { attempts } => {
            assert_eq!(attempts.len(), 1);
            assert!(attempts[0].failure_reason.contains("Could not deserialize"));
        }
        RedTeamResult::Violation { .. } => {
            panic!("Parse error should not be treated as a violation");
        }
    }
}

#[test]
fn redteam_fallback_extracts_json_from_freeform_text() {
    let invariant = AlwaysFailInvariant;
    let agent = MockAgent::always_ok("Found it!\n```json\n77\n```");
    let config = RedTeamConfig {
        num_iterations: 1,
        ..Default::default()
    };

    let result = auto_redteam(&invariant, &config, &agent, Path::new("/tmp"));

    match result {
        RedTeamResult::Violation { input_json, .. } => {
            assert_eq!(input_json, "77");
        }
        _ => panic!("Expected violation via extract_json fallback"),
    }
}

#[test]
fn redteam_zero_iterations_returns_immediately() {
    let invariant = AlwaysPassInvariant;
    let agent = MockAgent::always_ok("should not be called");
    let config = RedTeamConfig {
        num_iterations: 0,
        ..Default::default()
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

fn lloc() -> OptimizationObjective {
    LLOC
}

fn halstead() -> OptimizationObjective {
    HALSTEAD_BUGS
}

struct MockOptimizeEnv {
    measurements: Vec<HashMap<OptimizationObjective, f64>>,
    measure_index: usize,
    invariants_pass: Vec<bool>,
    invariant_index: usize,
    applied_diffs: Vec<String>,
    accepted: Vec<usize>,
    rejected: usize,
}

impl MockOptimizeEnv {
    fn new() -> Self {
        Self {
            measurements: vec![],
            measure_index: 0,
            invariants_pass: vec![true],
            invariant_index: 0,
            applied_diffs: vec![],
            accepted: vec![],
            rejected: 0,
        }
    }

    fn with_measurements(mut self, measurements: Vec<HashMap<OptimizationObjective, f64>>) -> Self {
        self.measurements = measurements;
        self
    }

    fn with_invariants(mut self, pass: Vec<bool>) -> Self {
        self.invariants_pass = pass;
        self
    }
}

impl OptimizeEnv for MockOptimizeEnv {
    fn work_dir(&self) -> &Path {
        Path::new("/tmp")
    }

    fn measure(
        &mut self,
        _objectives: &[OptimizationObjective],
    ) -> HashMap<OptimizationObjective, f64> {
        if self.measurements.is_empty() {
            return HashMap::new();
        }
        let idx = self.measure_index.min(self.measurements.len() - 1);
        self.measure_index += 1;
        self.measurements[idx].clone()
    }

    fn check_invariants(&mut self) -> bool {
        if self.invariants_pass.is_empty() {
            return true;
        }
        let idx = self.invariant_index.min(self.invariants_pass.len() - 1);
        self.invariant_index += 1;
        self.invariants_pass[idx]
    }

    fn apply_diff(&mut self, diff: &str) {
        self.applied_diffs.push(diff.to_string());
    }

    fn accept(&mut self, iteration: usize, _commit_msg: &str) {
        self.accepted.push(iteration);
    }

    fn reject(&mut self) {
        self.rejected += 1;
    }
}

fn m(pairs: &[(OptimizationObjective, f64)]) -> HashMap<OptimizationObjective, f64> {
    pairs.iter().cloned().collect()
}

fn lloc_obj() -> ObjectiveFunction {
    const INPUTS: &[OptimizationObjective] = &[LLOC];
    ObjectiveFunction {
        name: "test_lloc",
        inputs: INPUTS,
        evaluate: |m, _| m.get(&LLOC).copied().unwrap_or(f64::INFINITY),
    }
}

fn opt_config(iterations: usize) -> OptimizeConfig {
    OptimizeConfig {
        num_iterations: iterations,
        ..Default::default()
    }
}

#[test]
fn optimize_custom_objective_function() {
    const INPUTS: &[OptimizationObjective] = &[LLOC, HALSTEAD_BUGS];
    let weighted = ObjectiveFunction {
        name: "weighted",
        inputs: INPUTS,
        evaluate: |m, _| 2.0 * m.get(&LLOC).unwrap_or(&0.0) + m.get(&HALSTEAD_BUGS).unwrap_or(&0.0),
    };

    let agent = MockAgent::from_responses(vec![Ok(AgentResponse {
        text: "optimized".into(),
        diff: Some("diff".into()),
    })]);

    let mut env = MockOptimizeEnv::new().with_measurements(vec![
        m(&[(lloc(), 10.0), (halstead(), 100.0)]),
        m(&[(lloc(), 8.0), (halstead(), 110.0)]),
    ]);

    let config = opt_config(1);
    let result = auto_optimize(&agent, &mut env, &weighted, &config, Path::new("/tmp"));

    assert_eq!(result.best_score, 120.0);
    assert!(env.accepted.is_empty());
    assert_eq!(env.rejected, 1);
}

#[test]
fn optimize_multi_iteration_progressive_improvement() {
    let agent = MockAgent::from_responses(vec![
        Ok(AgentResponse {
            text: "iter 1".into(),
            diff: Some("diff1".into()),
        }),
        Ok(AgentResponse {
            text: "iter 2".into(),
            diff: Some("diff2".into()),
        }),
        Ok(AgentResponse {
            text: "iter 3".into(),
            diff: Some("diff3".into()),
        }),
    ]);

    let mut env = MockOptimizeEnv::new().with_measurements(vec![
        m(&[(lloc(), 10.0)]),
        m(&[(lloc(), 8.0)]),
        m(&[(lloc(), 9.0)]),
        m(&[(lloc(), 6.0)]),
    ]);

    let config = opt_config(3);
    let obj = lloc_obj();
    let result = auto_optimize(&agent, &mut env, &obj, &config, Path::new("/tmp"));

    assert_eq!(result.attempts.len(), 3);
    assert_eq!(result.best_score, 6.0);
    assert_eq!(env.accepted, vec![1, 3]);
    assert_eq!(env.rejected, 1);

    assert!(result.attempts[0].accepted);
    assert_eq!(result.attempts[0].score_delta_vs_best, -2.0);
    assert_eq!(result.attempts[0].score_delta_vs_baseline, -2.0);

    assert!(!result.attempts[1].accepted);
    assert_eq!(result.attempts[1].score_delta_vs_best, 1.0);
    assert_eq!(result.attempts[1].score_delta_vs_baseline, -1.0);

    assert!(result.attempts[2].accepted);
    assert_eq!(result.attempts[2].score_delta_vs_best, -2.0);
    assert_eq!(result.attempts[2].score_delta_vs_baseline, -4.0);
}

#[test]
fn optimize_stops_when_agent_produces_no_diff() {
    let agent = MockAgent::from_responses(vec![
        Ok(AgentResponse {
            text: "changed".into(),
            diff: Some("diff1".into()),
        }),
        Ok(AgentResponse {
            text: "nothing else".into(),
            diff: None,
        }),
    ]);

    let mut env =
        MockOptimizeEnv::new().with_measurements(vec![m(&[(lloc(), 10.0)]), m(&[(lloc(), 9.0)])]);

    let config = opt_config(5);
    let obj = lloc_obj();
    let result = auto_optimize(&agent, &mut env, &obj, &config, Path::new("/tmp"));

    assert_eq!(result.attempts.len(), 1);
}

#[test]
fn optimize_stops_when_agent_errors() {
    let agent = MockAgent::from_responses(vec![
        Ok(AgentResponse {
            text: "change".into(),
            diff: Some("diff".into()),
        }),
        Err(AgentError::new("agent crashed")),
    ]);

    let mut env =
        MockOptimizeEnv::new().with_measurements(vec![m(&[(lloc(), 10.0)]), m(&[(lloc(), 10.0)])]);

    let config = opt_config(5);
    let obj = lloc_obj();
    let result = auto_optimize(&agent, &mut env, &obj, &config, Path::new("/tmp"));

    assert_eq!(result.attempts.len(), 1);
}

#[test]
fn optimize_zero_iterations() {
    let agent = MockAgent::always_ok("should not be called");
    let mut env = MockOptimizeEnv::new().with_measurements(vec![m(&[(lloc(), 10.0)])]);

    let config = opt_config(0);
    let obj = lloc_obj();
    let result = auto_optimize(&agent, &mut env, &obj, &config, Path::new("/tmp"));

    assert!(result.attempts.is_empty());
    assert_eq!(result.baseline_score, 10.0);
    assert_eq!(result.best_score, 10.0);
    assert!(agent.recorded_prompts().is_empty());
}

#[test]
fn optimize_prompt_includes_measurements_and_hint() {
    let agent = MockAgent::from_responses(vec![Ok(AgentResponse {
        text: "done".into(),
        diff: Some("diff".into()),
    })]);

    let mut env =
        MockOptimizeEnv::new().with_measurements(vec![m(&[(lloc(), 42.0)]), m(&[(lloc(), 42.0)])]);

    let config = OptimizeConfig {
        num_iterations: 1,
        hint: Some("Focus on the inner loop".into()),
        ..Default::default()
    };
    let obj = lloc_obj();
    auto_optimize(&agent, &mut env, &obj, &config, Path::new("/tmp"));

    let prompts = agent.recorded_prompts();
    assert_eq!(prompts.len(), 1);
    assert!(prompts[0].contains("42.0"));
    assert!(prompts[0].contains("Focus on the inner loop"));
}

#[test]
fn optimize_diff_is_applied() {
    let agent = MockAgent::from_responses(vec![Ok(AgentResponse {
        text: "changed something".into(),
        diff: Some("--- a/x\n+++ b/x\n".into()),
    })]);

    let mut env =
        MockOptimizeEnv::new().with_measurements(vec![m(&[(lloc(), 10.0)]), m(&[(lloc(), 10.0)])]);

    let config = opt_config(1);
    let obj = lloc_obj();
    auto_optimize(&agent, &mut env, &obj, &config, Path::new("/tmp"));

    assert_eq!(env.applied_diffs.len(), 1);
    assert!(env.applied_diffs[0].contains("--- a/x"));
}

#[test]
fn optimize_invariant_failure_mid_sequence() {
    let agent = MockAgent::from_responses(vec![
        Ok(AgentResponse {
            text: "i1".into(),
            diff: Some("d1".into()),
        }),
        Ok(AgentResponse {
            text: "i2".into(),
            diff: Some("d2".into()),
        }),
        Ok(AgentResponse {
            text: "i3".into(),
            diff: Some("d3".into()),
        }),
    ]);

    let mut env = MockOptimizeEnv::new()
        .with_measurements(vec![
            m(&[(lloc(), 10.0)]),
            m(&[(lloc(), 8.0)]),
            m(&[(lloc(), 5.0)]),
            m(&[(lloc(), 7.0)]),
        ])
        .with_invariants(vec![true, false, true]);

    let config = opt_config(3);
    let obj = lloc_obj();
    let result = auto_optimize(&agent, &mut env, &obj, &config, Path::new("/tmp"));

    assert_eq!(result.attempts.len(), 3);
    assert!(result.attempts[0].invariants_passed);
    assert!(!result.attempts[1].invariants_passed);
    assert!(result.attempts[2].invariants_passed);
    assert_eq!(env.accepted, vec![1, 3]);
    assert_eq!(env.rejected, 1);
    assert_eq!(result.best_score, 7.0);
}
