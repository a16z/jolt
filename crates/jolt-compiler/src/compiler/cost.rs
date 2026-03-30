//! Cost model types for staging optimization.
//!
//! Provides compile-time parameters, solver configuration, and cost comparison
//! for the L0 → L1 lowering pass. All cost estimation is heuristic — no field
//! arithmetic or expression evaluation required.

/// Compile-time parameters for cost estimation.
///
/// Dimension sizes determine sumcheck round counts and buffer allocations.
/// Field size and PCS proof cost are heuristic inputs.
#[derive(Clone, Debug)]
pub struct CompileParams {
    /// `dim_sizes[i]` = number of sumcheck variables for dimension `i`.
    /// For example, `log_T = 20` means 20 sumcheck rounds over a `2^20` table.
    pub dim_sizes: Vec<u64>,
    /// Size of one field element in bytes (e.g., 32 for BN254 Fr).
    pub field_size_bytes: u64,
    /// PCS proof cost per distinct opening point, in field elements.
    pub pcs_proof_size: u64,
}

impl CompileParams {
    /// Total sumcheck variables for a binding order: `Σ dim_sizes[d]`.
    pub fn total_variables(&self, binding_order: &[usize]) -> u64 {
        binding_order.iter().map(|&d| self.dim_sizes[d]).sum()
    }
}

/// How the solver treats an optimization objective.
#[derive(Clone, Debug)]
pub enum Objective {
    /// Free variable — solver minimizes this.
    Minimize,
    /// Hard constraint — must be ≤ bound. Solver returns error if infeasible.
    Bounded(u64),
    /// Not considered in optimization or feasibility.
    Ignore,
}

/// Solver configuration: which objectives to optimize vs constrain.
///
/// Multiple `Minimize` objectives are compared lexicographically in
/// declaration order (proof_size first, then peak_memory, then prover_time).
#[derive(Clone, Debug)]
pub struct SolverConfig {
    pub proof_size: Objective,
    pub peak_memory: Objective,
    pub prover_time: Objective,
}

/// Evaluated cost of a candidate staging solution.
#[derive(Clone, Debug, Default)]
pub struct Cost {
    /// Total proof payload in field elements (sumcheck round polys + PCS proofs).
    pub proof_size: u64,
    /// Number of distinct PCS opening points.
    pub eval_points: usize,
    /// Peak prover memory in bytes across all stages.
    pub peak_memory: u64,
    /// Estimated total prover work units.
    pub prover_time: u64,
}

/// Lexicographic comparison: true if `a` is strictly better than `b`.
///
/// Compares `Minimize` objectives in declaration order. First differing
/// objective determines the result. Equal on all minimized objectives → false.
pub fn is_better(a: &Cost, b: &Cost, config: &SolverConfig) -> bool {
    if matches!(config.proof_size, Objective::Minimize) && a.proof_size != b.proof_size {
        return a.proof_size < b.proof_size;
    }
    if matches!(config.peak_memory, Objective::Minimize) && a.peak_memory != b.peak_memory {
        return a.peak_memory < b.peak_memory;
    }
    if matches!(config.prover_time, Objective::Minimize) && a.prover_time != b.prover_time {
        return a.prover_time < b.prover_time;
    }
    false
}

/// True if all `Bounded` objectives are within their limits.
pub fn satisfies(cost: &Cost, config: &SolverConfig) -> bool {
    bounded_ok(&config.proof_size, cost.proof_size)
        && bounded_ok(&config.peak_memory, cost.peak_memory)
        && bounded_ok(&config.prover_time, cost.prover_time)
}

fn bounded_ok(obj: &Objective, val: u64) -> bool {
    match obj {
        Objective::Bounded(limit) => val <= *limit,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_better_lexicographic() {
        let config = SolverConfig {
            proof_size: Objective::Minimize,
            peak_memory: Objective::Minimize,
            prover_time: Objective::Ignore,
        };
        let a = Cost {
            proof_size: 10,
            peak_memory: 100,
            ..Default::default()
        };
        let b = Cost {
            proof_size: 20,
            peak_memory: 50,
            ..Default::default()
        };
        // proof_size wins first
        assert!(is_better(&a, &b, &config));
        assert!(!is_better(&b, &a, &config));
    }

    #[test]
    fn is_better_tiebreak() {
        let config = SolverConfig {
            proof_size: Objective::Minimize,
            peak_memory: Objective::Minimize,
            prover_time: Objective::Ignore,
        };
        let a = Cost {
            proof_size: 10,
            peak_memory: 50,
            ..Default::default()
        };
        let b = Cost {
            proof_size: 10,
            peak_memory: 100,
            ..Default::default()
        };
        assert!(is_better(&a, &b, &config));
    }

    #[test]
    fn is_better_equal_is_false() {
        let config = SolverConfig {
            proof_size: Objective::Minimize,
            peak_memory: Objective::Ignore,
            prover_time: Objective::Ignore,
        };
        let a = Cost {
            proof_size: 10,
            ..Default::default()
        };
        assert!(!is_better(&a, &a, &config));
    }

    #[test]
    fn satisfies_bounded() {
        let config = SolverConfig {
            proof_size: Objective::Bounded(100),
            peak_memory: Objective::Ignore,
            prover_time: Objective::Ignore,
        };
        assert!(satisfies(
            &Cost {
                proof_size: 50,
                ..Default::default()
            },
            &config
        ));
        assert!(satisfies(
            &Cost {
                proof_size: 100,
                ..Default::default()
            },
            &config
        ));
        assert!(!satisfies(
            &Cost {
                proof_size: 101,
                ..Default::default()
            },
            &config
        ));
    }

    #[test]
    fn satisfies_minimize_always_ok() {
        let config = SolverConfig {
            proof_size: Objective::Minimize,
            peak_memory: Objective::Minimize,
            prover_time: Objective::Minimize,
        };
        let c = Cost {
            proof_size: u64::MAX,
            peak_memory: u64::MAX,
            prover_time: u64::MAX,
            ..Default::default()
        };
        assert!(satisfies(&c, &config));
    }

    #[test]
    fn total_variables() {
        let p = CompileParams {
            dim_sizes: vec![20, 16],
            field_size_bytes: 32,
            pcs_proof_size: 100,
        };
        assert_eq!(p.total_variables(&[0]), 20);
        assert_eq!(p.total_variables(&[1]), 16);
        assert_eq!(p.total_variables(&[0, 1]), 36);
        assert_eq!(p.total_variables(&[]), 0);
    }
}
