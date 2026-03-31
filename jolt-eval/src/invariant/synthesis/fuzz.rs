use super::super::{DynInvariant, SynthesisTarget};
use super::SynthesisRegistry;

/// Generate `libfuzzer_sys` fuzz target source code for a named invariant.
///
/// The generated code should be placed in a `fuzz/fuzz_targets/` directory
/// and compiled as a separate binary with `cargo fuzz`.
pub fn generate_fuzz_target(_invariant_name: &str, struct_path: &str) -> String {
    format!(
        r#"#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::{{Arbitrary, Unstructured}};
use jolt_eval::Invariant;

// Lazily initialize the invariant and setup (expensive one-time cost)
use std::sync::LazyLock;
static SETUP: LazyLock<({struct_path}, <{struct_path} as Invariant>::Setup)> = LazyLock::new(|| {{
    let invariant = {struct_path}::default();
    let setup = invariant.setup();
    (invariant, setup)
}});

fuzz_target!(|data: &[u8]| {{
    let mut u = Unstructured::new(data);
    if let Ok(input) = <<{struct_path} as Invariant>::Input as Arbitrary>::arbitrary(&mut u) {{
        let (invariant, setup) = &*SETUP;
        // We don't panic on invariant violations during fuzzing --
        // instead we log them. The fuzzer's job is to find inputs
        // that trigger violations.
        if let Err(e) = invariant.check(setup, input) {{
            eprintln!("INVARIANT VIOLATION: {{}}", e);
            panic!("Invariant '{{}}' violated: {{}}", invariant.name(), e);
        }}
    }}
}});
"#
    )
}

/// List all invariants suitable for fuzz target generation.
pub fn fuzzable_invariants(registry: &SynthesisRegistry) -> Vec<&dyn DynInvariant> {
    registry.for_target(SynthesisTarget::Fuzz)
}
