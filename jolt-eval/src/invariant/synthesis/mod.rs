pub mod fuzz;
pub mod redteam;
pub mod test;

use std::sync::Arc;

use super::{registered_invariants, DynInvariant, SynthesisTarget};
use crate::TestCase;

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

    /// Build a registry from all `inventory`-registered invariants.
    pub fn from_inventory(test_case: Arc<TestCase>, default_inputs: Vec<u8>) -> Self {
        let mut registry = Self::new();
        for entry in registered_invariants() {
            registry.register((entry.build)(Arc::clone(&test_case), default_inputs.clone()));
        }
        registry
    }

    pub fn register(&mut self, invariant: Box<dyn DynInvariant>) {
        self.invariants.push(invariant);
    }

    pub fn invariants(&self) -> &[Box<dyn DynInvariant>] {
        &self.invariants
    }

    /// Consume the registry and return the invariant list.
    pub fn into_invariants(self) -> Vec<Box<dyn DynInvariant>> {
        self.invariants
    }

    /// Return invariants that include the given synthesis target.
    pub fn for_target(&self, target: SynthesisTarget) -> Vec<&dyn DynInvariant> {
        self.invariants
            .iter()
            .filter(|inv| inv.targets().contains(target))
            .map(|inv| inv.as_ref())
            .collect()
    }

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

impl Default for SynthesisRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Return the names of all `inventory`-registered invariants.
/// Does not require a `TestCase`.
pub fn invariant_names() -> Vec<&'static str> {
    registered_invariants().map(|e| e.name).collect()
}
