//! Claim-graph analysis, Dory PIOP. See `specs/claim-graph-analysis.md`.
#![expect(
    clippy::expect_used,
    reason = "the analysis harness fails loudly with full context"
)]

mod claim_graph;

use claim_graph::graph::ClaimGraph;
use claim_graph::registry::{all_aliases, all_vertices};
use claim_graph::{assert_graph_snapshot, Piop, ProtocolConfig};

fn assert_well_formed(context: &str, config: &ProtocolConfig) {
    let records = all_vertices(config);
    let graph = ClaimGraph::build(Piop::Dory, &records, config, all_aliases());
    let violations = graph.check();
    assert!(
        violations.is_empty(),
        "[{context}] claim-graph violations:\n  {}\n\ngraph:\n{graph}",
        violations.join("\n  ")
    );
}

#[test]
fn dory_clear_graph_is_well_formed() {
    assert_well_formed("dory, clear, small", &ProtocolConfig::small());
}

#[test]
fn dory_clear_graph_is_well_formed_with_advice() {
    assert_well_formed(
        "dory, clear, small + trusted/untrusted advice",
        &ProtocolConfig::small_with_advice(),
    );
}

#[test]
fn dory_clear_graph_is_well_formed_with_committed_program() {
    assert_well_formed(
        "dory, clear, small + committed program",
        &ProtocolConfig::small_committed_program(),
    );
}

// The akita cargo feature changes cfg'd constants (NUM_BYTECODE_VAL_STAGES)
// that reshape even the Dory relations, so the Dory goldens are pinned under
// the PIOP's canonical build only; the akita lane still runs the
// well-formedness cells above.
#[cfg(not(feature = "akita"))]
#[test]
fn dory_graph_snapshots() {
    for (name, config) in [
        ("dory_small", ProtocolConfig::small()),
        ("dory_small_advice", ProtocolConfig::small_with_advice()),
        (
            "dory_small_committed",
            ProtocolConfig::small_committed_program(),
        ),
    ] {
        let records = all_vertices(&config);
        let graph = ClaimGraph::build(Piop::Dory, &records, &config, all_aliases());
        assert_graph_snapshot(name, &graph);
    }
}

/// On-demand terminal rendering:
/// `cargo nextest run -p jolt-claims dump_claim_graph --run-ignored all --no-capture`
#[test]
#[ignore = "on-demand graph dump"]
#[expect(clippy::print_stdout, reason = "the dump test exists to print")]
fn dump_claim_graph() {
    let config = ProtocolConfig::small();
    let records = all_vertices(&config);
    println!(
        "{}",
        ClaimGraph::build(Piop::Dory, &records, &config, all_aliases())
    );
}
