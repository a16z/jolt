use jolt_eval::invariant::{CheckError, Invariant, InvariantTargets, InvariantViolation};
use jolt_eval::objective::{AbstractObjective, Direction, MeasurementError};

/// A trivial invariant for testing the framework itself.
struct TrivialInvariant;
impl InvariantTargets for TrivialInvariant {}

impl Invariant for TrivialInvariant {
    type Setup = ();
    type Input = u8;

    fn name(&self) -> &str {
        "trivial"
    }

    fn description(&self) -> String {
        "Always passes".to_string()
    }

    fn setup(&self) -> Self::Setup {}

    fn check(&self, _setup: &Self::Setup, _input: u8) -> Result<(), CheckError> {
        Ok(())
    }

    fn seed_corpus(&self) -> Vec<u8> {
        vec![0, 1, 255]
    }
}

/// An invariant that always fails, for testing violation reporting.
struct FailingInvariant;
impl InvariantTargets for FailingInvariant {}

impl Invariant for FailingInvariant {
    type Setup = ();
    type Input = u8;

    fn name(&self) -> &str {
        "failing"
    }

    fn description(&self) -> String {
        "Always fails".to_string()
    }

    fn setup(&self) -> Self::Setup {}

    fn check(&self, _setup: &Self::Setup, input: u8) -> Result<(), CheckError> {
        Err(CheckError::Violation(InvariantViolation::new(format!(
            "failed for input {input}"
        ))))
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
    for input in inv.seed_corpus() {
        inv.check(&(), input).unwrap();
    }
}

#[test]
fn test_failing_invariant_reports_violations() {
    let inv = FailingInvariant;
    for input in inv.seed_corpus() {
        assert!(inv.check(&(), input).is_err());
    }
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
            value: 3.125,
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
    assert_eq!(results["prover_time"], 3.125);
    assert_eq!(results["inline_count"], 256.0);
}
