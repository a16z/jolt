pub mod fuzz;
pub mod redteam;
pub mod test;

use super::{DynInvariant, SynthesisTarget};

/// Registry of invariants available for synthesis.
pub struct SynthesisRegistry {
    invariants: Vec<Box<dyn DynInvariant>>,
}

impl SynthesisRegistry {
    pub fn new() -> Self {
        Self {
            invariants: Vec::new(),
        }
    }

    pub fn register(&mut self, invariant: Box<dyn DynInvariant>) {
        self.invariants.push(invariant);
    }

    pub fn invariants(&self) -> &[Box<dyn DynInvariant>] {
        &self.invariants
    }

    /// Return invariants that include the given synthesis target.
    pub fn for_target(&self, target: SynthesisTarget) -> Vec<&dyn DynInvariant> {
        self.invariants
            .iter()
            .filter(|inv| inv.targets().contains(target))
            .map(|inv| inv.as_ref())
            .collect()
    }
}

impl Default for SynthesisRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Return all registered invariant names.
impl SynthesisRegistry {
    pub fn names(&self) -> Vec<&str> {
        self.invariants.iter().map(|inv| inv.name()).collect()
    }

    pub fn names_for_target(&self, target: SynthesisTarget) -> Vec<&str> {
        self.for_target(target)
            .iter()
            .map(|inv| inv.name())
            .collect()
    }
}

/// Canonical list of built-in Jolt invariant names.
///
/// This is the single source of truth used by all CLI binaries for
/// `--list` output and error messages.  It does not require constructing
/// a `TestCase` or `SynthesisRegistry`.
pub const BUILTIN_INVARIANT_NAMES: &[&str] = &[
    "soundness",
    "verifier_completeness",
    "prover_completeness",
    "determinism",
    "serialization_roundtrip",
    "zk_consistency",
];
