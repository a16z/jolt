use std::sync::Arc;

use super::super::{CheckJsonResult, DynInvariant, SynthesisTarget};
use super::SynthesisRegistry;
use crate::TestCase;

/// Fuzz a named invariant with raw byte data from libfuzzer.
///
/// `data` is fed through `arbitrary::Unstructured` to produce the
/// invariant's `Input` type, which is then checked against the
/// invariant.  Setup is performed once and cached for the process
/// lifetime.
///
/// Panics on invariant violation (which is what libfuzzer needs to
/// detect a finding).
///
/// # Usage in a fuzz target
///
/// ```ignore
/// #![no_main]
/// use libfuzzer_sys::fuzz_target;
/// fuzz_target!(|data: &[u8]| {
///     jolt_eval::invariant::synthesis::fuzz::fuzz_invariant("split_eq_bind_low_high", data);
/// });
/// ```
///
/// For invariants that require a guest ELF, set `JOLT_FUZZ_ELF` to the
/// path of a pre-compiled guest ELF before running `cargo fuzz`.
/// Invariants that don't need a guest work without it.
pub fn fuzz_invariant(invariant_name: &str, data: &[u8]) {
    use std::any::Any;
    use std::sync::LazyLock;

    struct CachedInvariant {
        inv: Box<dyn DynInvariant>,
        setup: Box<dyn Any + Send + Sync>,
    }

    static CACHE: LazyLock<Vec<CachedInvariant>> = LazyLock::new(|| {
        let test_case: Option<Arc<TestCase>> = std::env::var("JOLT_FUZZ_ELF").ok().map(|elf_path| {
            let elf_bytes = std::fs::read(&elf_path)
                .unwrap_or_else(|e| panic!("Failed to read {elf_path}: {e}"));
            let memory_config = common::jolt_device::MemoryConfig {
                max_input_size: 4096,
                max_output_size: 4096,
                max_untrusted_advice_size: common::constants::DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
                max_trusted_advice_size: common::constants::DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
                stack_size: 65536,
                heap_size: 32768,
                program_size: None,
            };
            Arc::new(TestCase {
                elf_contents: elf_bytes,
                memory_config,
                max_trace_length: 65536,
            })
        });
        let registry = SynthesisRegistry::from_inventory(test_case, vec![]);
        registry
            .into_invariants()
            .into_iter()
            .map(|inv| {
                let setup = inv.dyn_setup();
                CachedInvariant { inv, setup }
            })
            .collect()
    });

    let cached = CACHE
        .iter()
        .find(|c| c.inv.name() == invariant_name)
        .unwrap_or_else(|| panic!("Invariant '{invariant_name}' not found"));

    if let Ok(json_str) = std::str::from_utf8(data) {
        match cached.inv.check_json_input(&*cached.setup, json_str) {
            CheckJsonResult::Violation(e) => {
                panic!(
                    "Invariant '{}' violated: {e}\nInput JSON: {json_str}",
                    cached.inv.name()
                );
            }
            CheckJsonResult::Pass | CheckJsonResult::BadInput(_) => {}
        }
    }
}

/// List all invariants suitable for fuzz target generation.
pub fn fuzzable_invariants(registry: &SynthesisRegistry) -> Vec<&dyn DynInvariant> {
    registry.for_target(SynthesisTarget::Fuzz)
}

/// Return the names of all registered invariants that
/// include [`SynthesisTarget::Fuzz`].
pub fn fuzzable_invariant_names() -> Vec<&'static str> {
    super::super::registered_invariants()
        .filter(|e| (e.targets)().contains(SynthesisTarget::Fuzz))
        .map(|e| e.name)
        .collect()
}
