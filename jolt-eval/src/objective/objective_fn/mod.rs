use std::collections::HashMap;

use crate::agent::DiffScope;

use super::{
    OptimizationObjective, BIND_HIGH_TO_LOW, BIND_LOW_TO_HIGH, COGNITIVE_COMPLEXITY, HALSTEAD_BUGS,
    LLOC,
};

/// A concrete objective function that the optimizer minimizes.
///
/// Contains the list of measurements it depends on and a pure
/// function that combines them into a scalar.
#[derive(Clone, Copy)]
pub struct ObjectiveFunction {
    /// CLI-visible name (e.g. `"minimize_lloc"`).
    pub name: &'static str,
    /// The [`OptimizationObjective`]s this function reads.
    pub inputs: &'static [OptimizationObjective],
    /// Combine measurements into a scalar to minimize.
    /// The HashMap is guaranteed to contain all keys from [`inputs`].
    pub evaluate: fn(&HashMap<OptimizationObjective, f64>) -> f64,
}

impl ObjectiveFunction {
    /// All registered objective functions.
    pub fn all() -> &'static [ObjectiveFunction] {
        &[
            MINIMIZE_LLOC,
            MINIMIZE_COGNITIVE_COMPLEXITY,
            MINIMIZE_HALSTEAD_BUGS,
            MINIMIZE_BIND_LOW_TO_HIGH,
            MINIMIZE_BIND_HIGH_TO_LOW,
        ]
    }

    /// Look up an objective function by CLI name.
    pub fn by_name(name: &str) -> Option<&'static ObjectiveFunction> {
        Self::all().iter().find(|f| f.name == name)
    }

    /// Derive a [`DiffScope`] from the union of all input objectives' diff paths.
    pub fn diff_scope(&self) -> DiffScope {
        let mut paths = Vec::new();
        for input in self.inputs {
            for &p in input.diff_paths() {
                let s = p.to_string();
                if !paths.contains(&s) {
                    paths.push(s);
                }
            }
        }
        DiffScope::Include(paths)
    }
}

pub const MINIMIZE_LLOC: ObjectiveFunction = ObjectiveFunction {
    name: "minimize_lloc",
    inputs: &[LLOC],
    evaluate: |m| m.get(&LLOC).copied().unwrap_or(f64::INFINITY),
};

pub const MINIMIZE_COGNITIVE_COMPLEXITY: ObjectiveFunction = ObjectiveFunction {
    name: "minimize_cognitive_complexity",
    inputs: &[COGNITIVE_COMPLEXITY],
    evaluate: |m| {
        m.get(&COGNITIVE_COMPLEXITY)
            .copied()
            .unwrap_or(f64::INFINITY)
    },
};

pub const MINIMIZE_HALSTEAD_BUGS: ObjectiveFunction = ObjectiveFunction {
    name: "minimize_halstead_bugs",
    inputs: &[HALSTEAD_BUGS],
    evaluate: |m| m.get(&HALSTEAD_BUGS).copied().unwrap_or(f64::INFINITY),
};

pub const MINIMIZE_BIND_LOW_TO_HIGH: ObjectiveFunction = ObjectiveFunction {
    name: "minimize_bind_low_to_high",
    inputs: &[BIND_LOW_TO_HIGH],
    evaluate: |m| m.get(&BIND_LOW_TO_HIGH).copied().unwrap_or(f64::INFINITY),
};

pub const MINIMIZE_BIND_HIGH_TO_LOW: ObjectiveFunction = ObjectiveFunction {
    name: "minimize_bind_high_to_low",
    inputs: &[BIND_HIGH_TO_LOW],
    evaluate: |m| m.get(&BIND_HIGH_TO_LOW).copied().unwrap_or(f64::INFINITY),
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimize_lloc_evaluates() {
        let mut m = HashMap::new();
        m.insert(LLOC, 5000.0);
        assert_eq!((MINIMIZE_LLOC.evaluate)(&m), 5000.0);
    }

    #[test]
    fn missing_input_returns_infinity() {
        let m = HashMap::new();
        assert_eq!((MINIMIZE_LLOC.evaluate)(&m), f64::INFINITY);
    }

    #[test]
    fn by_name_finds_registered() {
        let f = ObjectiveFunction::by_name("minimize_lloc").unwrap();
        assert_eq!(f.name, "minimize_lloc");
    }

    #[test]
    fn by_name_returns_none_for_unknown() {
        assert!(ObjectiveFunction::by_name("nonexistent").is_none());
    }

    #[test]
    fn all_returns_expected_count() {
        assert_eq!(ObjectiveFunction::all().len(), 5);
    }

    #[test]
    fn custom_composite_objective() {
        const INPUTS: &[OptimizationObjective] = &[LLOC, HALSTEAD_BUGS];
        let weighted = ObjectiveFunction {
            name: "weighted",
            inputs: INPUTS,
            evaluate: |m| {
                2.0 * m.get(&LLOC).unwrap_or(&0.0) + m.get(&HALSTEAD_BUGS).unwrap_or(&0.0)
            },
        };

        let mut m = HashMap::new();
        m.insert(LLOC, 10.0);
        m.insert(HALSTEAD_BUGS, 100.0);
        assert_eq!((weighted.evaluate)(&m), 120.0);
    }

    #[test]
    fn normalized_composite_objective() {
        use crate::objective::normalized;

        // LLOC baseline is 5500, Halstead baseline is 80.
        // Without normalization, LLOC dominates due to magnitude.
        // With normalization, both contribute on a comparable scale.
        const INPUTS: &[OptimizationObjective] = &[LLOC, HALSTEAD_BUGS];
        let balanced = ObjectiveFunction {
            name: "balanced_quality",
            inputs: INPUTS,
            evaluate: |m| 0.5 * normalized(&LLOC, m) + 0.5 * normalized(&HALSTEAD_BUGS, m),
        };

        let mut m = HashMap::new();
        m.insert(LLOC, 5500.0); // exactly at baseline → normalized = 1.0
        m.insert(HALSTEAD_BUGS, 80.0); // exactly at baseline → normalized = 1.0
        let score = (balanced.evaluate)(&m);
        assert!((score - 1.0).abs() < 1e-9, "expected 1.0, got {score}");

        // 10% improvement in LLOC
        m.insert(LLOC, 4950.0);
        let score2 = (balanced.evaluate)(&m);
        assert!(score2 < score, "10% LLOC improvement should reduce score");
        // 0.5 * (4950/5500) + 0.5 * (80/80) = 0.5 * 0.9 + 0.5 = 0.95
        assert!((score2 - 0.95).abs() < 1e-9, "expected 0.95, got {score2}");
    }
}
