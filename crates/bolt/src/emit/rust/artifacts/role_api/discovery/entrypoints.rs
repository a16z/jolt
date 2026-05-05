use super::super::source_scan::{find_public_fn, find_public_fn_containing};

pub(super) struct DiscoveredEntrypoints {
    pub(super) verifier_fn: Option<String>,
    pub(super) with_program_verifier_fn: Option<String>,
    pub(super) prover_fn: Option<String>,
    pub(super) with_program_prover_fn: Option<String>,
}

pub(super) fn discover_entrypoints(
    source: &str,
    prover_prefixes: &[&str],
) -> DiscoveredEntrypoints {
    DiscoveredEntrypoints {
        verifier_fn: find_public_fn(source, &["verify_"]),
        with_program_verifier_fn: find_public_fn_containing(source, &["verify_"], "_with_program"),
        prover_fn: find_public_fn(source, prover_prefixes),
        with_program_prover_fn: find_public_fn_containing(source, prover_prefixes, "_with_program"),
    }
}
