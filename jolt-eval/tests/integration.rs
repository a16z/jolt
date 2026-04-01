use jolt_eval::invariant::synthesis::SynthesisRegistry;
use jolt_eval::invariant::{DynInvariant, InvariantReport, InvariantViolation, SynthesisTarget};
use jolt_eval::objective::{AbstractObjective, Direction, MeasurementError};

/// A trivial invariant for testing the framework itself.
struct TrivialInvariant;

impl jolt_eval::Invariant for TrivialInvariant {
    type Setup = ();
    type Input = u8;

    fn name(&self) -> &str {
        "trivial"
    }

    fn description(&self) -> String {
        "Always passes".to_string()
    }

    fn targets(&self) -> enumset::EnumSet<SynthesisTarget> {
        SynthesisTarget::Test.into()
    }

    fn setup(&self) -> Self::Setup {}

    fn check(&self, _setup: &Self::Setup, _input: u8) -> Result<(), InvariantViolation> {
        Ok(())
    }

    fn seed_corpus(&self) -> Vec<u8> {
        vec![0, 1, 255]
    }
}

/// An invariant that always fails, for testing violation reporting.
struct FailingInvariant;

impl jolt_eval::Invariant for FailingInvariant {
    type Setup = ();
    type Input = u8;

    fn name(&self) -> &str {
        "failing"
    }

    fn description(&self) -> String {
        "Always fails".to_string()
    }

    fn targets(&self) -> enumset::EnumSet<SynthesisTarget> {
        SynthesisTarget::Test.into()
    }

    fn setup(&self) -> Self::Setup {}

    fn check(&self, _setup: &Self::Setup, input: u8) -> Result<(), InvariantViolation> {
        Err(InvariantViolation::new(format!("failed for input {input}")))
    }

    fn seed_corpus(&self) -> Vec<u8> {
        vec![42]
    }
}

/// A trivial objective for testing the framework.
struct ConstantObjective {
    label: &'static str,
    value: f64,
    direction: Direction,
}

impl AbstractObjective for ConstantObjective {
    fn name(&self) -> &str {
        self.label
    }

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        Ok(self.value)
    }

    fn direction(&self) -> Direction {
        self.direction
    }
}

#[test]
fn test_trivial_invariant_passes() {
    let inv = TrivialInvariant;
    let results = inv.run_checks(5);
    // 3 seed corpus + 5 random
    assert!(results.len() >= 3);
    assert!(results.iter().all(|r| r.is_ok()));
}

#[test]
fn test_failing_invariant_reports_violations() {
    let inv = FailingInvariant;
    let results = inv.run_checks(0);
    assert_eq!(results.len(), 1); // 1 seed corpus item
    assert!(results[0].is_err());
}

#[test]
fn test_invariant_report() {
    let results: Vec<Result<(), InvariantViolation>> =
        vec![Ok(()), Ok(()), Err(InvariantViolation::new("bad"))];
    let report = InvariantReport::from_results("test", &results);
    assert_eq!(report.total, 3);
    assert_eq!(report.passed, 2);
    assert_eq!(report.failed, 1);
    assert_eq!(report.violations.len(), 1);
}

#[test]
fn test_synthesis_registry() {
    let mut registry = SynthesisRegistry::new();
    registry.register(Box::new(TrivialInvariant));
    registry.register(Box::new(FailingInvariant));

    assert_eq!(registry.invariants().len(), 2);
    assert_eq!(registry.for_target(SynthesisTarget::Test).len(), 2);
    assert_eq!(registry.for_target(SynthesisTarget::Fuzz).len(), 0);
}

#[test]
fn test_constant_objective() {
    let obj = ConstantObjective {
        label: "latency",
        value: 42.0,
        direction: Direction::Minimize,
    };
    assert_eq!(obj.name(), "latency");
    assert_eq!(obj.collect_measurement().unwrap(), 42.0);
    assert_eq!(obj.direction(), Direction::Minimize);
}

#[test]
fn test_measure_objectives() {
    use jolt_eval::objective::measure_dyn;

    let objectives: Vec<Box<dyn AbstractObjective>> = vec![
        Box::new(ConstantObjective {
            label: "prover_time",
            value: 3.14,
            direction: Direction::Minimize,
        }),
        Box::new(ConstantObjective {
            label: "inline_count",
            value: 256.0,
            direction: Direction::Maximize,
        }),
    ];

    let results = measure_dyn(&objectives);
    assert_eq!(results.len(), 2);
    assert_eq!(results["prover_time"], 3.14);
    assert_eq!(results["inline_count"], 256.0);
}
