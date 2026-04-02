use super::super::{InvariantReport, JoltInvariants, SynthesisTarget};

/// Run all invariants that include the `Test` synthesis target.
pub fn run_test_suite(invariants: &[JoltInvariants], num_random: usize) -> Vec<InvariantReport> {
    invariants
        .iter()
        .filter(|inv| inv.targets().contains(SynthesisTarget::Test))
        .map(|inv| {
            let results = inv.run_checks(num_random);
            InvariantReport::from_results(inv.name(), &results)
        })
        .collect()
}

/// Generate `#[test]` function source code for a named invariant.
pub fn generate_test_source(invariant_name: &str, struct_path: &str) -> String {
    format!(
        r#"#[cfg(test)]
mod {invariant_name}_tests {{
    use super::*;
    use jolt_eval::Invariant;

    #[test]
    fn test_{invariant_name}_seed_corpus() {{
        let invariant = {struct_path}::default();
        let setup = invariant.setup();
        for (i, input) in invariant.seed_corpus().into_iter().enumerate() {{
            invariant.check(&setup, input).unwrap_or_else(|e| {{
                panic!("Invariant '{{}}' violated on seed {{}}: {{}}", invariant.name(), i, e);
            }});
        }}
    }}

    #[test]
    fn test_{invariant_name}_random() {{
        use rand::RngCore;
        let invariant = {struct_path}::default();
        let setup = invariant.setup();
        let mut rng = rand::thread_rng();
        for _ in 0..10 {{
            let mut raw = vec![0u8; 4096];
            rng.fill_bytes(&mut raw);
            let mut u = arbitrary::Unstructured::new(&raw);
            if let Ok(input) = <_ as arbitrary::Arbitrary>::arbitrary(&mut u) {{
                invariant.check(&setup, input).unwrap_or_else(|e| {{
                    panic!("Invariant '{{}}' violated: {{}}", invariant.name(), e);
                }});
            }}
        }}
    }}
}}
"#
    )
}
