//! The claim graph and its checks.
#![expect(
    clippy::zero_sized_map_values,
    reason = "SinkKind is single-variant until the zk and akita sink kinds land"
)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::{Piop, ProtocolConfig, VertexRecord};
use jolt_claims::protocols::jolt::geometry::committed_openings::{
    final_opening_id, final_opening_polynomial_order,
};
use jolt_claims::protocols::jolt::{
    JoltCommittedPolynomial, JoltOpeningId, JoltPolynomialId, JoltVirtualPolynomial,
};

/// How a terminal claim is discharged. Extended as first-run triage discovers
/// further discharge mechanisms (see the spec's dangling-claims section).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SinkKind {
    /// Opened against its commitment in the stage-8 joint PCS batch.
    PcsOpened,
}

pub struct ClaimGraph {
    pub piop: Piop,
    pub vertices: Vec<VertexRecord>,
    pub sinks: BTreeMap<JoltOpeningId, SinkKind>,
    /// `(aliased, source)` wire-copy equality pairs: the generated stage
    /// drivers enforce that both openings carry one value at one point, so a
    /// pair's members discharge together. Declared by relations via
    /// `aliased_output_openings` in jolt-verifier, assembled here from the
    /// same jolt-claims geometry functions.
    pub aliases: Vec<(JoltOpeningId, JoltOpeningId)>,
}

impl ClaimGraph {
    pub fn build(
        piop: Piop,
        records: &[VertexRecord],
        config: &ProtocolConfig,
        aliases: Vec<(JoltOpeningId, JoltOpeningId)>,
    ) -> Self {
        let vertices = records
            .iter()
            .filter(|record| record.graphs.contains(&piop))
            .cloned()
            .collect();
        let sinks = match piop {
            Piop::Dory => dory_sinks(config),
            Piop::Akita => BTreeMap::new(), // populated in the akita slice
        };
        Self {
            piop,
            vertices,
            sinks,
            aliases,
        }
    }

    /// Runs every check, returning all violations (empty means the graph is
    /// well-formed).
    pub fn check(&self) -> Vec<String> {
        let mut violations = Vec::new();
        let produced = self.check_unique_production(&mut violations);
        self.check_resolution(&produced, &mut violations);
        self.check_aliases(&produced, &mut violations);
        self.check_dangling(&produced, &mut violations);
        self.check_sink_typing(&mut violations);
        self.check_acyclic(&produced, &mut violations);
        self.check_adjacency(&mut violations);
        violations
    }

    /// Each opening id is produced by exactly one vertex. Returns id -> vertex
    /// index for the edge checks.
    fn check_unique_production(
        &self,
        violations: &mut Vec<String>,
    ) -> BTreeMap<JoltOpeningId, usize> {
        let mut produced = BTreeMap::new();
        for (index, vertex) in self.vertices.iter().enumerate() {
            for id in &vertex.out_edges {
                if let Some(&first) = produced.get(id) {
                    let first: &VertexRecord = &self.vertices[first];
                    violations.push(format!(
                        "double production: {id:?} produced by both {} and {}",
                        first.name, vertex.name
                    ));
                } else {
                    let _ = produced.insert(*id, index);
                }
            }
        }
        produced
    }

    /// Every consumed id is produced by the relation embedded in the id.
    fn check_resolution(
        &self,
        produced: &BTreeMap<JoltOpeningId, usize>,
        violations: &mut Vec<String>,
    ) {
        for vertex in &self.vertices {
            for id in &vertex.in_edges {
                match produced.get(id) {
                    None => violations.push(format!(
                        "unresolved consumption: {} consumes {id:?}, which no vertex produces",
                        vertex.name
                    )),
                    Some(&producer) => {
                        let producer = &self.vertices[producer];
                        if producer.relation != id_relation(id) {
                            violations.push(format!(
                                "producer mismatch: {id:?} names relation {:?} but is produced by {} ({:?})",
                                id_relation(id),
                                producer.name,
                                producer.relation
                            ));
                        }
                    }
                }
            }
        }
    }

    /// `produced - (consumed + sinks)` must be empty: an unchecked claim is a
    /// prover-controlled free variable.
    fn check_dangling(
        &self,
        produced: &BTreeMap<JoltOpeningId, usize>,
        violations: &mut Vec<String>,
    ) {
        let consumed: BTreeSet<JoltOpeningId> = self
            .vertices
            .iter()
            .flat_map(|vertex| vertex.in_edges.iter().copied())
            .collect();
        // Alias pairs carry one value at one point, so a pair's members
        // discharge together: extend the discharged set to alias closure.
        let discharged = |id: &JoltOpeningId| consumed.contains(id) || self.sinks.contains_key(id);
        let mut discharged_via_alias: BTreeSet<JoltOpeningId> = BTreeSet::new();
        // Alias chains are short (pairs today); one relaxation pass per pair
        // plus a fixpoint loop covers transitive chains if they ever appear.
        loop {
            let mut changed = false;
            for (left, right) in &self.aliases {
                let left_ok = discharged(left) || discharged_via_alias.contains(left);
                let right_ok = discharged(right) || discharged_via_alias.contains(right);
                if left_ok && !right_ok {
                    changed |= discharged_via_alias.insert(*right);
                }
                if right_ok && !left_ok {
                    changed |= discharged_via_alias.insert(*left);
                }
            }
            if !changed {
                break;
            }
        }
        for (id, &producer) in produced {
            if !discharged(id) && !discharged_via_alias.contains(id) {
                violations.push(format!(
                    "dangling claim: {id:?} produced by {} is neither consumed, a sink, nor aliased to one",
                    self.vertices[producer].name
                ));
            }
        }
    }

    /// Every alias endpoint must be a produced opening: a pair citing an id no
    /// vertex produces is stale.
    fn check_aliases(
        &self,
        produced: &BTreeMap<JoltOpeningId, usize>,
        violations: &mut Vec<String>,
    ) {
        for (aliased, source) in &self.aliases {
            for id in [aliased, source] {
                if !produced.contains_key(id) {
                    violations.push(format!(
                        "stale alias: {id:?} appears in an alias pair but no vertex produces it"
                    ));
                }
            }
        }
    }

    /// The PCS sink accepts only committed-polynomial and advice ids: virtual
    /// claims have no commitments and must reach committed claims through the
    /// reduction chain.
    fn check_sink_typing(&self, violations: &mut Vec<String>) {
        for (id, kind) in &self.sinks {
            if *kind == SinkKind::PcsOpened {
                if let JoltOpeningId::Polynomial {
                    polynomial: JoltPolynomialId::Virtual(virtual_polynomial),
                    ..
                } = id
                {
                    violations.push(format!(
                        "sink typing: virtual polynomial {virtual_polynomial:?} cannot be PCS-opened"
                    ));
                }
            }
        }
    }

    /// Kahn's toposort: the produce/consume relation must be a DAG. Returns
    /// generations for the `Display` rendering.
    fn generations(
        &self,
        produced: &BTreeMap<JoltOpeningId, usize>,
    ) -> Result<Vec<Vec<usize>>, Vec<usize>> {
        let n = self.vertices.len();
        let mut successors: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        let mut in_degree = vec![0usize; n];
        for (consumer, vertex) in self.vertices.iter().enumerate() {
            for id in &vertex.in_edges {
                if let Some(&producer) = produced.get(id) {
                    if producer != consumer && successors[producer].insert(consumer) {
                        in_degree[consumer] += 1;
                    }
                }
            }
        }
        let mut generations = Vec::new();
        let mut frontier: Vec<usize> = (0..n).filter(|&v| in_degree[v] == 0).collect();
        let mut seen = frontier.len();
        while !frontier.is_empty() {
            let mut next = Vec::new();
            for &vertex in &frontier {
                for &successor in &successors[vertex] {
                    in_degree[successor] -= 1;
                    if in_degree[successor] == 0 {
                        next.push(successor);
                        seen += 1;
                    }
                }
            }
            generations.push(frontier);
            frontier = next;
        }
        if seen == n {
            Ok(generations)
        } else {
            Err((0..n).filter(|&v| in_degree[v] > 0).collect())
        }
    }

    fn check_acyclic(
        &self,
        produced: &BTreeMap<JoltOpeningId, usize>,
        violations: &mut Vec<String>,
    ) {
        if let Err(cyclic) = self.generations(produced) {
            let names: Vec<&str> = cyclic
                .into_iter()
                .map(|vertex| self.vertices[vertex].name)
                .collect();
            violations.push(format!("cycle: vertices {names:?} form a dependency cycle"));
        }
    }

    /// The derive-emitted struct adjacency and the expression edges must agree
    /// (expression ids collapsed to family representatives). A struct edge
    /// absent from the expression must be conditional (Optional, or a Family
    /// that expands to zero members under this configuration).
    fn check_adjacency(&self, violations: &mut Vec<String>) {
        for vertex in &self.vertices {
            for (direction, expression, structural) in [
                ("in", &vertex.in_edges, vertex.struct_in_edges),
                ("out", &vertex.out_edges, vertex.struct_out_edges),
            ] {
                let expression: BTreeSet<JoltOpeningId> = expression
                    .iter()
                    .map(|id| family_representative(*id))
                    .collect();
                let struct_ids: BTreeSet<JoltOpeningId> =
                    structural.iter().map(|edge| edge.id).collect();
                for id in expression.difference(&struct_ids) {
                    violations.push(format!(
                        "adjacency drift: {} {direction}-edge {id:?} appears in the expression but not the claim struct",
                        vertex.name
                    ));
                }
                // A struct edge absent from the expression is a *forwarded*
                // opening (constrained downstream); it enters the vertex's
                // produced/consumed set, so the dangling check binds it. Only
                // the expression-without-struct direction above is drift.
            }
        }
    }
}

/// Collapses an indexed-family id to its index-0 representative, matching the
/// derive-emitted adjacency convention.
pub fn family_representative(id: JoltOpeningId) -> JoltOpeningId {
    let JoltOpeningId::Polynomial {
        polynomial,
        relation,
    } = id
    else {
        return id;
    };
    let polynomial = match polynomial {
        JoltPolynomialId::Virtual(virtual_polynomial) => {
            JoltPolynomialId::Virtual(match virtual_polynomial {
                JoltVirtualPolynomial::InstructionRa(_) => JoltVirtualPolynomial::InstructionRa(0),
                JoltVirtualPolynomial::LookupTableFlag(_) => {
                    JoltVirtualPolynomial::LookupTableFlag(0)
                }
                JoltVirtualPolynomial::BytecodeValClaim(_) => {
                    JoltVirtualPolynomial::BytecodeValClaim(0)
                }
                other => other,
            })
        }
        JoltPolynomialId::Committed(committed) => JoltPolynomialId::Committed(match committed {
            JoltCommittedPolynomial::InstructionRa(_) => JoltCommittedPolynomial::InstructionRa(0),
            JoltCommittedPolynomial::BytecodeRa(_) => JoltCommittedPolynomial::BytecodeRa(0),
            JoltCommittedPolynomial::RamRa(_) => JoltCommittedPolynomial::RamRa(0),
            JoltCommittedPolynomial::BytecodeChunk(_) => JoltCommittedPolynomial::BytecodeChunk(0),
            JoltCommittedPolynomial::UnsignedIncChunk(_) => {
                JoltCommittedPolynomial::UnsignedIncChunk(0)
            }
            other => other,
        }),
    };
    JoltOpeningId::Polynomial {
        polynomial,
        relation,
    }
}

fn id_relation(id: &JoltOpeningId) -> jolt_claims::protocols::jolt::JoltRelationId {
    match id {
        JoltOpeningId::Polynomial { relation, .. }
        | JoltOpeningId::UntrustedAdvice { relation }
        | JoltOpeningId::TrustedAdvice { relation } => *relation,
    }
}

fn dory_sinks(config: &ProtocolConfig) -> BTreeMap<JoltOpeningId, SinkKind> {
    final_opening_polynomial_order(
        config.formula.ra_layout,
        config.trusted_advice,
        config.untrusted_advice,
        config.committed_program_chunks,
    )
    .into_iter()
    .map(|polynomial| (final_opening_id(polynomial), SinkKind::PcsOpened))
    .collect()
}

impl fmt::Display for ClaimGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let edge_count: usize = self
            .vertices
            .iter()
            .map(|vertex| vertex.in_edges.len())
            .sum();
        writeln!(
            f,
            "ClaimGraph({:?}): {} vertices, {} consumption edges, {} sinks",
            self.piop,
            self.vertices.len(),
            edge_count,
            self.sinks.len()
        )?;
        let mut produced = BTreeMap::new();
        for (index, vertex) in self.vertices.iter().enumerate() {
            for id in &vertex.out_edges {
                let _ = produced.entry(*id).or_insert(index);
            }
        }
        let generations = match self.generations(&produced) {
            Ok(generations) => generations,
            Err(cyclic) => {
                writeln!(f, "  CYCLIC — unordered vertices:")?;
                for vertex in cyclic {
                    writeln!(f, "    {}", self.vertices[vertex].name)?;
                }
                return Ok(());
            }
        };
        for (level, generation) in generations.iter().enumerate() {
            writeln!(f, "generation {level}:")?;
            for &index in generation {
                let vertex = &self.vertices[index];
                writeln!(
                    f,
                    "  {} [{:?}] rounds={} degree={}",
                    vertex.name, vertex.relation, vertex.rounds, vertex.degree
                )?;
                for id in &vertex.in_edges {
                    writeln!(f, "    <- {}", render_id(id))?;
                }
                for id in &vertex.out_edges {
                    let sink = self
                        .sinks
                        .get(&family_representative(*id))
                        .map_or("", |kind| match kind {
                            SinkKind::PcsOpened => "  [PcsOpened]",
                        });
                    writeln!(f, "    -> {}{sink}", render_id(id))?;
                }
            }
        }
        writeln!(f, "sinks:")?;
        for (id, kind) in &self.sinks {
            writeln!(f, "  {} [{kind:?}]", render_id(id))?;
        }
        Ok(())
    }
}

fn render_id(id: &JoltOpeningId) -> String {
    match id {
        JoltOpeningId::Polynomial {
            polynomial: JoltPolynomialId::Virtual(polynomial),
            relation,
        } => format!("virt {polynomial:?} @ {relation:?}"),
        JoltOpeningId::Polynomial {
            polynomial: JoltPolynomialId::Committed(polynomial),
            relation,
        } => format!("comm {polynomial:?} @ {relation:?}"),
        JoltOpeningId::TrustedAdvice { relation } => format!("trusted-advice @ {relation:?}"),
        JoltOpeningId::UntrustedAdvice { relation } => format!("untrusted-advice @ {relation:?}"),
    }
}

#[cfg(test)]
mod planted {
    use super::*;
    use jolt_claims::protocols::jolt::JoltRelationId;

    fn vertex(
        name: &'static str,
        relation: JoltRelationId,
        in_edges: &[JoltOpeningId],
        out_edges: &[JoltOpeningId],
    ) -> VertexRecord {
        VertexRecord {
            name,
            relation,
            graphs: &[Piop::Dory],
            rounds: 1,
            degree: 1,
            in_edges: in_edges.iter().copied().collect(),
            out_edges: out_edges.iter().copied().collect(),
            struct_in_edges: &[],
            struct_out_edges: &[],
        }
    }

    fn opening(relation: JoltRelationId) -> JoltOpeningId {
        JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::PC, relation)
    }

    fn graph(vertices: Vec<VertexRecord>) -> ClaimGraph {
        ClaimGraph {
            piop: Piop::Dory,
            vertices,
            sinks: BTreeMap::new(),
            aliases: Vec::new(),
        }
    }

    #[test]
    fn detects_dangling_claim() {
        let graph = graph(vec![vertex(
            "a",
            JoltRelationId::SpartanOuter,
            &[],
            &[opening(JoltRelationId::SpartanOuter)],
        )]);
        let violations = graph.check();
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("dangling claim")),
            "{violations:?}"
        );
    }

    #[test]
    fn detects_double_production() {
        let id = opening(JoltRelationId::SpartanOuter);
        let graph = graph(vec![
            vertex("a", JoltRelationId::SpartanOuter, &[], &[id]),
            vertex("b", JoltRelationId::SpartanOuter, &[id], &[]),
            vertex("c", JoltRelationId::SpartanOuter, &[], &[id]),
        ]);
        let violations = graph.check();
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("double production")),
            "{violations:?}"
        );
    }

    #[test]
    fn detects_producer_mismatch() {
        // The id names SpartanShift as producer, but SpartanOuter produces it.
        let id = opening(JoltRelationId::SpartanShift);
        let graph = graph(vec![
            vertex("a", JoltRelationId::SpartanOuter, &[], &[id]),
            vertex("b", JoltRelationId::SpartanShift, &[id], &[]),
        ]);
        let violations = graph.check();
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("producer mismatch")),
            "{violations:?}"
        );
    }

    #[test]
    fn detects_cycle() {
        let ab = opening(JoltRelationId::SpartanOuter);
        let ba = opening(JoltRelationId::SpartanShift);
        let graph = graph(vec![
            vertex("a", JoltRelationId::SpartanOuter, &[ba], &[ab]),
            vertex("b", JoltRelationId::SpartanShift, &[ab], &[ba]),
        ]);
        let violations = graph.check();
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("cycle")),
            "{violations:?}"
        );
    }

    #[test]
    fn alias_to_consumed_opening_discharges_its_pair() {
        let consumed = opening(JoltRelationId::SpartanOuter);
        let aliased = opening(JoltRelationId::SpartanShift);
        let mut graph = graph(vec![
            vertex("a", JoltRelationId::SpartanOuter, &[], &[consumed]),
            vertex("b", JoltRelationId::SpartanShift, &[], &[aliased]),
            vertex("c", JoltRelationId::SpartanOuter, &[consumed], &[]),
        ]);
        graph.aliases.push((aliased, consumed));
        let violations = graph.check();
        assert!(
            !violations
                .iter()
                .any(|violation| violation.contains("dangling")),
            "aliased claim should discharge via its consumed pair: {violations:?}"
        );
    }

    #[test]
    fn detects_stale_alias() {
        let produced = opening(JoltRelationId::SpartanOuter);
        let phantom = opening(JoltRelationId::SpartanShift);
        let mut graph = graph(vec![vertex(
            "a",
            JoltRelationId::SpartanOuter,
            &[],
            &[produced],
        )]);
        graph.aliases.push((produced, phantom));
        let violations = graph.check();
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("stale alias")),
            "{violations:?}"
        );
    }

    #[test]
    fn detects_virtual_id_at_pcs_sink() {
        let id = opening(JoltRelationId::SpartanOuter);
        let mut graph = graph(vec![vertex("a", JoltRelationId::SpartanOuter, &[], &[id])]);
        let _ = graph.sinks.insert(id, SinkKind::PcsOpened);
        let violations = graph.check();
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("sink typing")),
            "{violations:?}"
        );
    }
}
