use std::collections::HashMap;

use crate::agent::DiffScope;

use super::{
    OptimizationObjective, BIND_HIGH_TO_LOW, BIND_LOW_TO_HIGH, COGNITIVE_COMPLEXITY, HALSTEAD_BUGS,
    LLOC, NAIVE_SORT_TIME,
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
    /// The first HashMap contains the current measurements; the second
    /// contains the baseline measurements (captured at the start of the
    /// optimization run) for use with [`normalized()`](super::normalized).
    pub evaluate:
        fn(&HashMap<OptimizationObjective, f64>, &HashMap<OptimizationObjective, f64>) -> f64,
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
            MINIMIZE_NAIVE_SORT_TIME,
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
    evaluate: |m, _| m.get(&LLOC).copied().unwrap_or(f64::INFINITY),
};

pub const MINIMIZE_COGNITIVE_COMPLEXITY: ObjectiveFunction = ObjectiveFunction {
    name: "minimize_cognitive_complexity",
    inputs: &[COGNITIVE_COMPLEXITY],
    evaluate: |m, _| {
        m.get(&COGNITIVE_COMPLEXITY)
            .copied()
            .unwrap_or(f64::INFINITY)
    },
};

pub const MINIMIZE_HALSTEAD_BUGS: ObjectiveFunction = ObjectiveFunction {
    name: "minimize_halstead_bugs",
    inputs: &[HALSTEAD_BUGS],
    evaluate: |m, _| m.get(&HALSTEAD_BUGS).copied().unwrap_or(f64::INFINITY),
};

pub const MINIMIZE_BIND_LOW_TO_HIGH: ObjectiveFunction = ObjectiveFunction {
    name: "minimize_bind_low_to_high",
    inputs: &[BIND_LOW_TO_HIGH],
    evaluate: |m, _| m.get(&BIND_LOW_TO_HIGH).copied().unwrap_or(f64::INFINITY),
};

pub const MINIMIZE_BIND_HIGH_TO_LOW: ObjectiveFunction = ObjectiveFunction {
    name: "minimize_bind_high_to_low",
    inputs: &[BIND_HIGH_TO_LOW],
    evaluate: |m, _| m.get(&BIND_HIGH_TO_LOW).copied().unwrap_or(f64::INFINITY),
};

pub const MINIMIZE_NAIVE_SORT_TIME: ObjectiveFunction = ObjectiveFunction {
    name: "minimize_naive_sort_time",
    inputs: &[NAIVE_SORT_TIME],
    evaluate: |m, _| m.get(&NAIVE_SORT_TIME).copied().unwrap_or(f64::INFINITY),
};

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_baselines() -> HashMap<OptimizationObjective, f64> {
        HashMap::new()
    }

    #[test]
    fn minimize_lloc_evaluates() {
        let mut m = HashMap::new();
        m.insert(LLOC, 5000.0);
        assert_eq!((MINIMIZE_LLOC.evaluate)(&m, &empty_baselines()), 5000.0);
    }

    #[test]
    fn missing_input_returns_infinity() {
        let m = HashMap::new();
        assert_eq!(
            (MINIMIZE_LLOC.evaluate)(&m, &empty_baselines()),
            f64::INFINITY
        );
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
    fn custom_composite_objective() {
        const INPUTS: &[OptimizationObjective] = &[LLOC, HALSTEAD_BUGS];
        let weighted = ObjectiveFunction {
            name: "weighted",
            inputs: INPUTS,
            evaluate: |m, _| {
                2.0 * m.get(&LLOC).unwrap_or(&0.0) + m.get(&HALSTEAD_BUGS).unwrap_or(&0.0)
            },
        };

        let mut m = HashMap::new();
        m.insert(LLOC, 10.0);
        m.insert(HALSTEAD_BUGS, 100.0);
        assert_eq!((weighted.evaluate)(&m, &empty_baselines()), 120.0);
    }

    #[test]
    fn normalized_composite_objective() {
        use crate::objective::normalized;

        // Baselines are the initial measurements. Normalization divides
        // each value by its baseline, yielding a dimensionless ratio.
        const INPUTS: &[OptimizationObjective] = &[LLOC, HALSTEAD_BUGS];
        let balanced = ObjectiveFunction {
            name: "balanced_quality",
            inputs: INPUTS,
            evaluate: |m, b| 0.5 * normalized(&LLOC, m, b) + 0.5 * normalized(&HALSTEAD_BUGS, m, b),
        };

        let mut baselines = HashMap::new();
        baselines.insert(LLOC, 5500.0);
        baselines.insert(HALSTEAD_BUGS, 80.0);

        // At baseline values → normalized = 1.0 for each → score = 1.0
        let score = (balanced.evaluate)(&baselines, &baselines);
        assert!((score - 1.0).abs() < 1e-9, "expected 1.0, got {score}");

        // 10% improvement in LLOC
        let mut m = baselines.clone();
        m.insert(LLOC, 4950.0);
        let score2 = (balanced.evaluate)(&m, &baselines);
        assert!(score2 < score, "10% LLOC improvement should reduce score");
        // 0.5 * (4950/5500) + 0.5 * (80/80) = 0.5 * 0.9 + 0.5 = 0.95
        assert!((score2 - 0.95).abs() < 1e-9, "expected 0.95, got {score2}");
    }
}
