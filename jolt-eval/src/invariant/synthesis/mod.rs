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
