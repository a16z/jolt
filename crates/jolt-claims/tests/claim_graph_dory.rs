//! Claim-graph analysis, Dory PIOP. See `specs/claim-graph-analysis.md`.
#![expect(
    clippy::expect_used,
    reason = "the analysis harness fails loudly with full context"
)]

mod claim_graph;

use claim_graph::graph::ClaimGraph;
use claim_graph::registry::{
    all_aliases, all_relation_ids, all_vertices, registration, Registration,
};
use claim_graph::{Piop, ProtocolConfig};

/// Every relation id is carried by a registered vertex, and every registered
/// vertex's relation id is classified `Registered`.
#[test]
fn vertex_set_is_exhaustive() {
    let config = ProtocolConfig::small();
    let records = all_vertices(&config);
    let covered: std::collections::BTreeSet<_> =
        records.iter().map(|record| record.relation).collect();
    let mut missing = Vec::new();
    for id in all_relation_ids() {
        match registration(id) {
            Registration::Registered => {
                assert!(
                    covered.contains(&id),
                    "{id:?} is classified Registered but no manifest vertex carries it"
                );
            }
            Registration::Pending => missing.push(id),
        }
    }
    assert!(
        missing.is_empty(),
        "unregistered relations (add ProtocolVertex impls and manifest entries): {missing:?}"
    );
}

#[test]
fn dory_clear_graph_is_well_formed() {
    let config = ProtocolConfig::small();
    let records = all_vertices(&config);
    let graph = ClaimGraph::build(Piop::Dory, &records, &config, all_aliases());
    let violations = graph.check();
    assert!(
        violations.is_empty(),
        "claim-graph violations:\n  {}\n\ngraph:\n{graph}",
        violations.join("\n  ")
    );
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
