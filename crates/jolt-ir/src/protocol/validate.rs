//! Validation passes for the protocol graph.
//!
//! Analyses partition into two levels:
//! - **Claim graph** (invariant): claim completeness, dimension consistency, acyclicity
//! - **Staging** (choice): valid layering, point convergence, commitment consistency

use std::collections::{HashMap, HashSet};
use std::fmt;

use super::types::*;
use crate::PolynomialId;

#[derive(Debug)]
pub enum GraphError {
    /// A produced claim is never referenced by any downstream vertex.
    DanglingClaim(ClaimId),
    /// A claim is referenced but never produced.
    MissingClaim(ClaimId),
    /// A claim has multiple producers.
    DuplicateProducer {
        claim: ClaimId,
        producers: Vec<VertexId>,
    },
    /// A vertex references itself (self-loop).
    SelfLoop(VertexId),
    /// The vertex graph contains a cycle.
    Cycle,
    /// A committed polynomial has no claim reaching an Opening vertex.
    UnreachableCommitted(PolynomialId),
}

#[derive(Debug)]
pub enum StagingError {
    /// A vertex appears in more than one stage.
    DuplicateVertex(VertexId),
    /// A vertex in the claim graph is not assigned to any stage.
    UnassignedVertex(VertexId),
    /// Two vertices in the same stage have a dependency edge (antichain violated).
    IntraStageEdge {
        stage: StageId,
        from: VertexId,
        to: VertexId,
    },
    /// A dependency flows from a later stage to an earlier stage.
    BackwardEdge {
        from_stage: StageId,
        to_stage: StageId,
        claim: ClaimId,
    },
}

#[derive(Debug)]
pub enum CommitmentError {
    /// A committed polynomial is not in any commitment group.
    Ungrouped(PolynomialId),
    /// A committed polynomial appears in multiple commitment groups.
    MultipleGroups {
        polynomial: PolynomialId,
        groups: Vec<CommitmentGroupId>,
    },
    /// A virtual polynomial appears in a commitment group.
    VirtualInGroup {
        polynomial: PolynomialId,
        group: CommitmentGroupId,
    },
    /// An opening vertex's polynomial has no commitment group.
    OpeningWithoutGroup(VertexId),
    /// A commitment group is missing from the transcript order.
    MissingFromTranscript(CommitmentGroupId),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DanglingClaim(id) => write!(f, "claim {:?} produced but never referenced", id),
            Self::MissingClaim(id) => write!(f, "claim {:?} referenced but never produced", id),
            Self::DuplicateProducer { claim, producers } => {
                write!(
                    f,
                    "claim {:?} has multiple producers: {:?}",
                    claim, producers
                )
            }
            Self::SelfLoop(id) => write!(f, "vertex {:?} references itself", id),
            Self::Cycle => write!(f, "vertex graph contains a cycle"),
            Self::UnreachableCommitted(id) => {
                write!(
                    f,
                    "committed polynomial {:?} has no claim reaching Opening",
                    id
                )
            }
        }
    }
}

impl fmt::Display for StagingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateVertex(id) => write!(f, "vertex {:?} in multiple stages", id),
            Self::UnassignedVertex(id) => write!(f, "vertex {:?} not in any stage", id),
            Self::IntraStageEdge { stage, from, to } => {
                write!(f, "intra-stage edge in {:?}: {:?} -> {:?}", stage, from, to)
            }
            Self::BackwardEdge {
                from_stage,
                to_stage,
                claim,
            } => {
                write!(
                    f,
                    "backward edge: claim {:?} from {:?} to {:?}",
                    claim, from_stage, to_stage
                )
            }
        }
    }
}

impl fmt::Display for CommitmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ungrouped(id) => write!(f, "committed poly {:?} not in any group", id),
            Self::MultipleGroups { polynomial, groups } => {
                write!(f, "poly {:?} in multiple groups: {:?}", polynomial, groups)
            }
            Self::VirtualInGroup { polynomial, group } => {
                write!(f, "virtual poly {:?} in group {:?}", polynomial, group)
            }
            Self::OpeningWithoutGroup(id) => {
                write!(f, "opening vertex {:?} poly has no group", id)
            }
            Self::MissingFromTranscript(id) => {
                write!(f, "group {:?} missing from transcript order", id)
            }
        }
    }
}

#[derive(Debug)]
pub enum ChallengeSpecError {
    /// Two challenge specs in the same stage share a label.
    DuplicateLabel {
        stage: StageId,
        label: &'static str,
    },
}

#[derive(Debug)]
pub enum ClaimFlowError {
    NoOpening(PolynomialId),
    DuplicateOpening { polynomial: PolynomialId, count: usize },
}

/// All Opening vertices must consume claims at the same symbolic point.
#[derive(Debug)]
pub struct PointConvergenceError {
    pub expected: SymbolicPoint,
    pub got: SymbolicPoint,
    pub polynomial: PolynomialId,
}

impl fmt::Display for ChallengeSpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateLabel { stage, label } => {
                write!(f, "duplicate label {:?} in {:?}", label, stage)
            }
        }
    }
}

impl fmt::Display for ClaimFlowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoOpening(id) => write!(f, "committed poly {:?} has no Opening vertex", id),
            Self::DuplicateOpening { polynomial, count } => {
                write!(
                    f,
                    "committed poly {:?} has {} Opening vertices",
                    polynomial, count
                )
            }
        }
    }
}

#[derive(Debug)]
pub enum EvalOrderingError {
    /// ClaimIds are not dense: expected N claims but max ClaimId is higher.
    NonDenseClaimIds { num_claims: usize, max_id: u32 },
    /// A sumcheck stage produces zero claims.
    EmptyStage(StageId),
    /// A claim is produced by more than one vertex within the same stage.
    DuplicateProducedClaim { stage: StageId, claim: ClaimId },
}

impl fmt::Display for EvalOrderingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonDenseClaimIds { num_claims, max_id } => {
                write!(
                    f,
                    "claim IDs not dense: {} claims but max ID is {}",
                    num_claims, max_id
                )
            }
            Self::EmptyStage(id) => write!(f, "sumcheck stage {:?} produces zero claims", id),
            Self::DuplicateProducedClaim { stage, claim } => {
                write!(
                    f,
                    "claim {:?} produced by multiple vertices in {:?}",
                    claim, stage
                )
            }
        }
    }
}

impl std::error::Error for GraphError {}
impl std::error::Error for StagingError {}
impl std::error::Error for CommitmentError {}
impl std::error::Error for ChallengeSpecError {}
impl std::error::Error for ClaimFlowError {}
impl std::error::Error for EvalOrderingError {}

impl ClaimGraph {
    /// Validate claim completeness and structural integrity.
    ///
    /// Checks:
    /// - Every produced claim is referenced by at least one downstream vertex.
    /// - No claim is referenced without being produced (except root claims).
    /// - Each claim has exactly one producer.
    /// - No self-loops.
    /// - The vertex graph is acyclic.
    pub fn validate(&self, root_claims: &HashSet<ClaimId>) -> Vec<GraphError> {
        let mut errors = Vec::new();

        // Build producer map: ClaimId → producing VertexId
        let mut producers: HashMap<ClaimId, Vec<VertexId>> = HashMap::new();
        for vertex in &self.vertices {
            for claim_id in vertex.all_produced_claims() {
                producers.entry(claim_id).or_default().push(vertex.id());
            }
        }

        // Check: each claim has exactly one producer
        for (claim_id, vertex_ids) in &producers {
            if vertex_ids.len() > 1 {
                errors.push(GraphError::DuplicateProducer {
                    claim: *claim_id,
                    producers: vertex_ids.clone(),
                });
            }
        }

        // Build referenced set: all claims that appear in any vertex's deps/consumes
        let mut referenced: HashSet<ClaimId> = HashSet::new();
        for vertex in &self.vertices {
            for &claim_id in vertex.dep_claims() {
                let _ = referenced.insert(claim_id);
            }
        }

        // Check: every produced claim is referenced (no dangling obligations)
        let produced: HashSet<ClaimId> = producers.keys().copied().collect();
        for &claim_id in &produced {
            if !referenced.contains(&claim_id) {
                errors.push(GraphError::DanglingClaim(claim_id));
            }
        }

        // Check: every referenced claim is produced (except root claims)
        for &claim_id in &referenced {
            if !produced.contains(&claim_id) && !root_claims.contains(&claim_id) {
                errors.push(GraphError::MissingClaim(claim_id));
            }
        }

        // Check: no self-loops
        for vertex in &self.vertices {
            let produced_set: HashSet<ClaimId> = vertex.all_produced_claims().into_iter().collect();
            for &dep in vertex.dep_claims() {
                if produced_set.contains(&dep) {
                    errors.push(GraphError::SelfLoop(vertex.id()));
                    break;
                }
            }
        }

        // Check: acyclicity via topological sort (Kahn's algorithm)
        if !self.is_acyclic(&producers) {
            errors.push(GraphError::Cycle);
        }

        errors
    }

    /// Check that the vertex DAG is acyclic using Kahn's algorithm.
    fn is_acyclic(&self, producers: &HashMap<ClaimId, Vec<VertexId>>) -> bool {
        let n = self.vertices.len();

        // Build adjacency: vertex → set of successor vertices
        let mut in_degree: HashMap<VertexId, usize> = HashMap::new();
        let mut successors: HashMap<VertexId, HashSet<VertexId>> = HashMap::new();

        for vertex in &self.vertices {
            let vid = vertex.id();
            let _ = in_degree.entry(vid).or_insert(0);
            let _ = successors.entry(vid).or_default();
        }

        for vertex in &self.vertices {
            let vid = vertex.id();
            for &dep_claim in vertex.dep_claims() {
                if let Some(dep_vertices) = producers.get(&dep_claim) {
                    for &dep_vid in dep_vertices {
                        if dep_vid != vid && successors.entry(dep_vid).or_default().insert(vid) {
                            *in_degree.entry(vid).or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        // Kahn's: process vertices with in_degree 0
        let mut queue: Vec<VertexId> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&v, _)| v)
            .collect();
        let mut visited = 0usize;

        while let Some(vid) = queue.pop() {
            visited += 1;
            if let Some(succs) = successors.get(&vid) {
                for &s in succs {
                    let d = in_degree.get_mut(&s).unwrap();
                    *d -= 1;
                    if *d == 0 {
                        queue.push(s);
                    }
                }
            }
        }

        visited == n
    }

    /// Compute a topological ordering of the vertex DAG.
    ///
    /// Returns `None` if the graph contains a cycle.
    pub fn topological_order(&self) -> Option<Vec<VertexId>> {
        let mut producers: HashMap<ClaimId, Vec<VertexId>> = HashMap::new();
        for vertex in &self.vertices {
            for claim_id in vertex.all_produced_claims() {
                producers.entry(claim_id).or_default().push(vertex.id());
            }
        }

        let mut in_degree: HashMap<VertexId, usize> = HashMap::new();
        let mut successors: HashMap<VertexId, HashSet<VertexId>> = HashMap::new();

        for vertex in &self.vertices {
            let vid = vertex.id();
            let _ = in_degree.entry(vid).or_insert(0);
            let _ = successors.entry(vid).or_default();
        }

        for vertex in &self.vertices {
            let vid = vertex.id();
            for &dep_claim in vertex.dep_claims() {
                if let Some(dep_vertices) = producers.get(&dep_claim) {
                    for &dep_vid in dep_vertices {
                        if dep_vid != vid && successors.entry(dep_vid).or_default().insert(vid) {
                            *in_degree.entry(vid).or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        let mut queue: Vec<VertexId> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&v, _)| v)
            .collect();
        queue.sort(); // deterministic ordering

        let mut order = Vec::with_capacity(self.vertices.len());
        while let Some(vid) = queue.pop() {
            order.push(vid);
            if let Some(succs) = successors.get(&vid) {
                let mut next: Vec<VertexId> = Vec::new();
                for &s in succs {
                    let d = in_degree.get_mut(&s).unwrap();
                    *d -= 1;
                    if *d == 0 {
                        next.push(s);
                    }
                }
                next.sort();
                queue.extend(next);
            }
        }

        if order.len() == self.vertices.len() {
            Some(order)
        } else {
            None
        }
    }
}

impl ProtocolGraph {
    /// Validate that the staging is a valid topological layering.
    ///
    /// Checks:
    /// - Every vertex appears in exactly one stage (or the opening stage).
    /// - No vertex in stage k depends on another vertex in stage k (antichain).
    /// - All dependencies flow from earlier stages to later stages.
    pub fn validate_staging(&self) -> Vec<StagingError> {
        let mut errors = Vec::new();

        // Build vertex → stage map
        let mut vertex_stage: HashMap<VertexId, StageId> = HashMap::new();
        let mut seen: HashSet<VertexId> = HashSet::new();

        for stage in &self.staging.stages {
            for &vid in &stage.vertices {
                if !seen.insert(vid) {
                    errors.push(StagingError::DuplicateVertex(vid));
                }
                let _ = vertex_stage.insert(vid, stage.id);
            }
        }

        // Opening stage vertices
        for &vid in &self.staging.opening.vertices {
            if !seen.insert(vid) {
                errors.push(StagingError::DuplicateVertex(vid));
            }
        }

        // Check all vertices are assigned
        for vertex in &self.claim_graph.vertices {
            if !seen.contains(&vertex.id()) {
                errors.push(StagingError::UnassignedVertex(vertex.id()));
            }
        }

        // Build claim → producer vertex map
        let mut claim_producer: HashMap<ClaimId, VertexId> = HashMap::new();
        for vertex in &self.claim_graph.vertices {
            for claim_id in vertex.all_produced_claims() {
                let _ = claim_producer.insert(claim_id, vertex.id());
            }
        }

        // Build stage ordering: StageId → position index
        let stage_order: HashMap<StageId, usize> = self
            .staging
            .stages
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id, i))
            .collect();

        // Check antichain and forward-flow properties
        for vertex in &self.claim_graph.vertices {
            let vid = vertex.id();
            let Some(&my_stage) = vertex_stage.get(&vid) else {
                continue; // opening vertex, skip
            };

            for &dep_claim in vertex.dep_claims() {
                let Some(&dep_vid) = claim_producer.get(&dep_claim) else {
                    continue; // root claim
                };
                let Some(&dep_stage) = vertex_stage.get(&dep_vid) else {
                    continue; // opening stage producer
                };

                if dep_stage == my_stage {
                    errors.push(StagingError::IntraStageEdge {
                        stage: my_stage,
                        from: dep_vid,
                        to: vid,
                    });
                }

                let dep_pos = stage_order.get(&dep_stage).copied().unwrap_or(usize::MAX);
                let my_pos = stage_order.get(&my_stage).copied().unwrap_or(usize::MAX);
                if dep_pos > my_pos {
                    errors.push(StagingError::BackwardEdge {
                        from_stage: dep_stage,
                        to_stage: my_stage,
                        claim: dep_claim,
                    });
                }
            }
        }

        errors
    }

    /// Validate the commitment strategy against the claim graph.
    ///
    /// Checks:
    /// - Every committed polynomial belongs to exactly one group.
    /// - No virtual polynomial appears in a commitment group.
    /// - Every opening vertex's polynomial has a commitment group.
    /// - The transcript ordering covers all groups.
    pub fn validate_commitment(&self) -> Vec<CommitmentError> {
        let mut errors = Vec::new();

        // Build polynomial → groups map
        let mut poly_groups: HashMap<PolynomialId, Vec<CommitmentGroupId>> = HashMap::new();
        for group in &self.commitment.groups {
            for &poly_id in &group.polynomials {
                poly_groups.entry(poly_id).or_default().push(group.id);
            }
        }

        // Check: every committed polynomial in exactly one group
        for poly in &self.claim_graph.polynomials {
            match &poly.kind {
                PolynomialKind::Committed { group: _ } => match poly_groups.get(&poly.id) {
                    None => errors.push(CommitmentError::Ungrouped(poly.id)),
                    Some(groups) if groups.len() > 1 => {
                        errors.push(CommitmentError::MultipleGroups {
                            polynomial: poly.id,
                            groups: groups.clone(),
                        });
                    }
                    _ => {}
                },
                PolynomialKind::Virtual => {
                    if let Some(groups) = poly_groups.get(&poly.id) {
                        for &gid in groups {
                            errors.push(CommitmentError::VirtualInGroup {
                                polynomial: poly.id,
                                group: gid,
                            });
                        }
                    }
                }
            }
        }

        // Check: every opening vertex's polynomial has a group
        for &vid in &self.staging.opening.vertices {
            let vertex = self.claim_graph.vertex(vid);
            if let Vertex::Opening(o) = vertex {
                let claim = self.claim_graph.claim(o.consumes);
                if let Some(poly) = self.claim_graph.polynomial(claim.polynomial) {
                    if matches!(poly.kind, PolynomialKind::Committed { .. })
                        && !poly_groups.contains_key(&poly.id)
                    {
                        errors.push(CommitmentError::OpeningWithoutGroup(vid));
                    }
                }
            }
        }

        // Check: transcript order covers all groups
        let ordered: HashSet<CommitmentGroupId> =
            self.commitment.transcript_order.iter().copied().collect();
        for group in &self.commitment.groups {
            if !ordered.contains(&group.id) {
                errors.push(CommitmentError::MissingFromTranscript(group.id));
            }
        }

        errors
    }

    /// Validate challenge specs across all stages.
    ///
    /// Checks:
    /// - Labels are unique within each stage (no two specs share a label).
    pub fn validate_challenge_specs(&self) -> Vec<ChallengeSpecError> {
        let mut errors = Vec::new();

        for stage in &self.staging.stages {
            let mut seen_labels: HashSet<&'static str> = HashSet::new();
            for spec in &stage.pre_squeeze {
                let label = match spec {
                    ChallengeSpec::Scalar { label } => label,
                    ChallengeSpec::Vector { label, .. } => label,
                    ChallengeSpec::GammaPowers { label, .. } => label,
                };
                if !seen_labels.insert(label) {
                    errors.push(ChallengeSpecError::DuplicateLabel {
                        stage: stage.id,
                        label,
                    });
                }
            }
        }

        errors
    }

    /// Validate eval ordering invariants required by the graph-driven verifier.
    ///
    /// Checks:
    /// - ClaimIds are dense: values form `[0, N)` so the verifier's `EvalCache`
    ///   (a flat `Vec<Option<F>>`) has no wasted slots.
    /// - Every sumcheck stage produces at least one claim.
    /// - No claim is produced by two vertices within the same stage.
    pub fn validate_eval_ordering(&self) -> Vec<EvalOrderingError> {
        let mut errors = Vec::new();

        // Dense claim IDs
        let num_claims = self.claim_graph.claims.len();
        if num_claims > 0 {
            let max_id = self
                .claim_graph
                .claims
                .iter()
                .map(|c| c.id.0)
                .max()
                .unwrap_or(0);
            if max_id as usize != num_claims - 1 {
                errors.push(EvalOrderingError::NonDenseClaimIds { num_claims, max_id });
            }
        }

        // Per-stage checks
        for stage in &self.staging.stages {
            let mut stage_produced: HashSet<ClaimId> = HashSet::new();
            let mut has_sumcheck = false;

            for &vid in &stage.vertices {
                let vertex = self.claim_graph.vertex(vid);
                if matches!(vertex, Vertex::Sumcheck(_)) {
                    has_sumcheck = true;
                }
                for &cid in vertex.produced_claims() {
                    if !stage_produced.insert(cid) {
                        errors.push(EvalOrderingError::DuplicateProducedClaim {
                            stage: stage.id,
                            claim: cid,
                        });
                    }
                }
            }

            if has_sumcheck && stage_produced.is_empty() {
                errors.push(EvalOrderingError::EmptyStage(stage.id));
            }
        }

        errors
    }

    /// Validate claim flow: every committed polynomial is opened exactly once.
    ///
    /// Checks:
    /// - Every committed polynomial has exactly one Opening vertex.
    /// - No committed polynomial is missing from the opening stage.
    pub fn validate_claim_flow(&self) -> Vec<ClaimFlowError> {
        let mut errors = Vec::new();

        // Count Opening vertices per committed polynomial
        let mut opening_count: HashMap<PolynomialId, usize> = HashMap::new();
        for vertex in &self.claim_graph.vertices {
            if let Vertex::Opening(o) = vertex {
                let claim = self.claim_graph.claim(o.consumes);
                *opening_count.entry(claim.polynomial).or_insert(0) += 1;
            }
        }

        // Every committed polynomial must have exactly one Opening
        // (except SpartanWitness which is opened by Spartan's inner sumcheck)
        for poly in &self.claim_graph.polynomials {
            if matches!(poly.kind, PolynomialKind::Committed { .. })
                && poly.id != PolynomialId::SpartanWitness
            {
                match opening_count.get(&poly.id).copied().unwrap_or(0) {
                    0 => errors.push(ClaimFlowError::NoOpening(poly.id)),
                    1 => {} // correct
                    n => errors.push(ClaimFlowError::DuplicateOpening {
                        polynomial: poly.id,
                        count: n,
                    }),
                }
            }
        }

        errors
    }

    /// Validates that all Opening vertices consume claims at the same symbolic point.
    ///
    /// This is the terminal invariant: all committed polynomial claims converge
    /// to a single evaluation point via claim reduction + normalization. If any
    /// Opening vertex has a different point, it means the RLC reduction would
    /// produce multiple PCS proofs instead of one.
    pub fn validate_point_convergence(&self) -> Option<PointConvergenceError> {
        let mut opening_points: Vec<(PolynomialId, &SymbolicPoint)> = Vec::new();

        for vertex in &self.claim_graph.vertices {
            if let Vertex::Opening(o) = vertex {
                let claim = self.claim_graph.claim(o.consumes);
                opening_points.push((claim.polynomial, &claim.point));
            }
        }

        if opening_points.len() <= 1 {
            return None;
        }

        // All points should be structurally equal
        let first_point = opening_points[0].1;
        for &(poly_id, point) in &opening_points[1..] {
            if point != first_point {
                return Some(PointConvergenceError {
                    expected: first_point.clone(),
                    got: point.clone(),
                    polynomial: poly_id,
                });
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::symbolic::{Symbol, SymbolicExpr};
    use crate::ClaimDefinition;

    fn simple_claim(id: u32, poly: PolynomialId) -> Claim {
        Claim {
            id: ClaimId(id),
            polynomial: poly,
            point: SymbolicPoint::Challenges(StageId(0)),
        }
    }

    #[test]
    fn empty_graph_is_valid() {
        let graph = ClaimGraph {
            polynomials: vec![],
            claims: vec![],
            vertices: vec![],
        };
        let errors = graph.validate(&HashSet::new());
        assert!(errors.is_empty());
    }

    #[test]
    fn single_opening_valid() {
        // One committed poly, one claim, one opening vertex
        let poly = Polynomial {
            id: PolynomialId::RamInc,
            kind: PolynomialKind::Committed {
                group: CommitmentGroupId(0),
            },
            num_vars: SymbolicExpr::symbol(Symbol::LOG_T),
        };
        let claim = simple_claim(0, PolynomialId::RamInc);
        let vertex = Vertex::Opening(OpeningVertex {
            id: VertexId(0),
            consumes: ClaimId(0),
        });

        let graph = ClaimGraph {
            polynomials: vec![poly],
            claims: vec![claim],
            vertices: vec![vertex],
        };

        // ClaimId(0) is a root claim (not produced by any vertex)
        let mut roots = HashSet::new();
        let _ = roots.insert(ClaimId(0));
        let errors = graph.validate(&roots);
        assert!(errors.is_empty(), "errors: {:?}", errors);
    }

    #[test]
    fn detects_missing_claim() {
        // Vertex references a claim that nobody produces and isn't a root
        let vertex = Vertex::Opening(OpeningVertex {
            id: VertexId(0),
            consumes: ClaimId(99),
        });
        let graph = ClaimGraph {
            polynomials: vec![],
            claims: vec![],
            vertices: vec![vertex],
        };
        let errors = graph.validate(&HashSet::new());
        assert!(errors
            .iter()
            .any(|e| matches!(e, GraphError::MissingClaim(ClaimId(99)))));
    }

    #[test]
    fn detects_duplicate_producer() {
        use crate::builder::ExprBuilder;

        let b = ExprBuilder::new();
        let dummy_expr = b.build(b.zero());
        let dummy_formula = ClaimFormula {
            definition: ClaimDefinition {
                expr: dummy_expr.clone(),
                opening_bindings: vec![],
                num_challenges: 0,
            },
            opening_claims: HashMap::new(),
        };

        let v0 = Vertex::Sumcheck(Box::new(SumcheckVertex {
            id: VertexId(0),
            deps: vec![],
            input: InputClaim::Constant(0),
            produces: vec![ClaimId(0)],
            side_effect_claims: vec![],
            formula: dummy_formula.clone(),
            degree: 2,
            num_vars: SymbolicExpr::Concrete(10),
            weighting: PublicPolynomial::Eq,
            phases: vec![Phase {
                num_vars: SymbolicExpr::Concrete(10),
                variable_group: VariableGroup::Cycle,
            }],
            output_challenge_spec: OutputChallengeSpec::None,
        }));
        let v1 = Vertex::Sumcheck(Box::new(SumcheckVertex {
            id: VertexId(1),
            deps: vec![],
            input: InputClaim::Constant(0),
            produces: vec![ClaimId(0)], // same claim produced by v0
            side_effect_claims: vec![],
            formula: dummy_formula,
            degree: 2,
            num_vars: SymbolicExpr::Concrete(10),
            weighting: PublicPolynomial::Eq,
            phases: vec![Phase {
                num_vars: SymbolicExpr::Concrete(10),
                variable_group: VariableGroup::Cycle,
            }],
            output_challenge_spec: OutputChallengeSpec::None,
        }));
        // v2 references ClaimId(0) so it's not dangling
        let v2 = Vertex::Opening(OpeningVertex {
            id: VertexId(2),
            consumes: ClaimId(0),
        });

        let graph = ClaimGraph {
            polynomials: vec![],
            claims: vec![],
            vertices: vec![v0, v1, v2],
        };
        let errors = graph.validate(&HashSet::new());
        assert!(errors
            .iter()
            .any(|e| matches!(e, GraphError::DuplicateProducer { .. })));
    }

    #[test]
    fn topological_order_linear_chain() {
        use crate::builder::ExprBuilder;

        let b = ExprBuilder::new();
        let dummy_expr = b.build(b.zero());
        let formula = || ClaimFormula {
            definition: ClaimDefinition {
                expr: dummy_expr.clone(),
                opening_bindings: vec![],
                num_challenges: 0,
            },
            opening_claims: HashMap::new(),
        };
        let phase = || Phase {
            num_vars: SymbolicExpr::Concrete(10),
            variable_group: VariableGroup::Cycle,
        };

        // v0 produces c0, v1 deps c0 produces c1, v2 deps c1
        let v0 = Vertex::Sumcheck(Box::new(SumcheckVertex {
            id: VertexId(0),
            deps: vec![],
            input: InputClaim::Constant(0),
            produces: vec![ClaimId(0)],
            side_effect_claims: vec![],
            formula: formula(),
            degree: 2,
            num_vars: SymbolicExpr::Concrete(10),
            weighting: PublicPolynomial::Eq,
            phases: vec![phase()],
            output_challenge_spec: OutputChallengeSpec::None,
        }));
        let v1 = Vertex::Sumcheck(Box::new(SumcheckVertex {
            id: VertexId(1),
            deps: vec![ClaimId(0)],
            input: InputClaim::Constant(0),
            produces: vec![ClaimId(1)],
            side_effect_claims: vec![],
            formula: formula(),
            degree: 2,
            num_vars: SymbolicExpr::Concrete(10),
            weighting: PublicPolynomial::Eq,
            phases: vec![phase()],
            output_challenge_spec: OutputChallengeSpec::None,
        }));
        let v2 = Vertex::Opening(OpeningVertex {
            id: VertexId(2),
            consumes: ClaimId(1),
        });

        let graph = ClaimGraph {
            polynomials: vec![],
            claims: vec![],
            vertices: vec![v0, v1, v2],
        };

        let order = graph.topological_order().expect("should be acyclic");
        // v0 must come before v1, v1 before v2
        let pos = |vid: VertexId| order.iter().position(|&v| v == vid).unwrap();
        assert!(pos(VertexId(0)) < pos(VertexId(1)));
        assert!(pos(VertexId(1)) < pos(VertexId(2)));
    }

    #[test]
    fn challenge_specs_no_duplicates_valid() {
        let graph = ProtocolGraph {
            claim_graph: ClaimGraph {
                polynomials: vec![],
                claims: vec![],
                vertices: vec![],
            },
            staging: Staging {
                stages: vec![Stage {
                    id: StageId(0),
                    vertices: vec![],
                    challenge_point: ChallengePoint {
                        num_vars: SymbolicExpr::Concrete(10),
                    },
                    batching: vec![],
                    pre_squeeze: vec![
                        ChallengeSpec::Scalar { label: "alpha" },
                        ChallengeSpec::Scalar { label: "beta" },
                        ChallengeSpec::GammaPowers {
                            label: "gamma",
                            count: SymbolicExpr::Concrete(5),
                        },
                    ],
                }],
                opening: OpeningStage {
                    vertices: vec![],
                    point: SymbolicPoint::Challenges(StageId(0)),
                    reduction: ReductionStrategy::Rlc,
                    opening_groups: vec![],
                },
            },
            commitment: CommitmentStrategy {
                groups: vec![],
                transcript_order: vec![],
            },
        };
        let errors = graph.validate_challenge_specs();
        assert!(errors.is_empty(), "errors: {:?}", errors);
    }

    #[test]
    fn challenge_specs_detects_duplicate_label() {
        let graph = ProtocolGraph {
            claim_graph: ClaimGraph {
                polynomials: vec![],
                claims: vec![],
                vertices: vec![],
            },
            staging: Staging {
                stages: vec![Stage {
                    id: StageId(0),
                    vertices: vec![],
                    challenge_point: ChallengePoint {
                        num_vars: SymbolicExpr::Concrete(10),
                    },
                    batching: vec![],
                    pre_squeeze: vec![
                        ChallengeSpec::Scalar { label: "gamma" },
                        ChallengeSpec::GammaPowers {
                            label: "gamma",
                            count: SymbolicExpr::Concrete(5),
                        },
                    ],
                }],
                opening: OpeningStage {
                    vertices: vec![],
                    point: SymbolicPoint::Challenges(StageId(0)),
                    reduction: ReductionStrategy::Rlc,
                    opening_groups: vec![],
                },
            },
            commitment: CommitmentStrategy {
                groups: vec![],
                transcript_order: vec![],
            },
        };
        let errors = graph.validate_challenge_specs();
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            ChallengeSpecError::DuplicateLabel {
                label: "gamma",
                ..
            }
        ));
    }

    #[test]
    fn claim_flow_valid_single_opening() {
        let poly = Polynomial {
            id: PolynomialId::RamInc,
            kind: PolynomialKind::Committed {
                group: CommitmentGroupId(0),
            },
            num_vars: SymbolicExpr::symbol(Symbol::LOG_T),
        };
        let claim = simple_claim(0, PolynomialId::RamInc);
        let vertex = Vertex::Opening(OpeningVertex {
            id: VertexId(0),
            consumes: ClaimId(0),
        });

        let graph = ProtocolGraph {
            claim_graph: ClaimGraph {
                polynomials: vec![poly],
                claims: vec![claim],
                vertices: vec![vertex],
            },
            staging: Staging {
                stages: vec![],
                opening: OpeningStage {
                    vertices: vec![VertexId(0)],
                    point: SymbolicPoint::Challenges(StageId(0)),
                    reduction: ReductionStrategy::Rlc,
                    opening_groups: vec![],
                },
            },
            commitment: CommitmentStrategy {
                groups: vec![CommitmentGroup {
                    id: CommitmentGroupId(0),
                    polynomials: vec![PolynomialId::RamInc],
                }],
                transcript_order: vec![CommitmentGroupId(0)],
            },
        };
        let errors = graph.validate_claim_flow();
        assert!(errors.is_empty(), "errors: {:?}", errors);
    }

    #[test]
    fn claim_flow_detects_missing_opening() {
        let poly = Polynomial {
            id: PolynomialId::RamInc,
            kind: PolynomialKind::Committed {
                group: CommitmentGroupId(0),
            },
            num_vars: SymbolicExpr::symbol(Symbol::LOG_T),
        };

        let graph = ProtocolGraph {
            claim_graph: ClaimGraph {
                polynomials: vec![poly],
                claims: vec![],
                vertices: vec![], // no opening vertex
            },
            staging: Staging {
                stages: vec![],
                opening: OpeningStage {
                    vertices: vec![],
                    point: SymbolicPoint::Challenges(StageId(0)),
                    reduction: ReductionStrategy::Rlc,
                    opening_groups: vec![],
                },
            },
            commitment: CommitmentStrategy {
                groups: vec![CommitmentGroup {
                    id: CommitmentGroupId(0),
                    polynomials: vec![PolynomialId::RamInc],
                }],
                transcript_order: vec![CommitmentGroupId(0)],
            },
        };
        let errors = graph.validate_claim_flow();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0], ClaimFlowError::NoOpening(PolynomialId::RamInc)));
    }

    #[test]
    fn fan_out_is_valid() {
        use crate::builder::ExprBuilder;

        let b = ExprBuilder::new();
        let dummy_expr = b.build(b.zero());
        let formula = || ClaimFormula {
            definition: ClaimDefinition {
                expr: dummy_expr.clone(),
                opening_bindings: vec![],
                num_challenges: 0,
            },
            opening_claims: HashMap::new(),
        };
        let phase = || Phase {
            num_vars: SymbolicExpr::Concrete(10),
            variable_group: VariableGroup::Cycle,
        };

        // v0 produces c0; v1 and v2 both dep on c0 (fan-out)
        let v0 = Vertex::Sumcheck(Box::new(SumcheckVertex {
            id: VertexId(0),
            deps: vec![],
            input: InputClaim::Constant(0),
            produces: vec![ClaimId(0)],
            side_effect_claims: vec![],
            formula: formula(),
            degree: 2,
            num_vars: SymbolicExpr::Concrete(10),
            weighting: PublicPolynomial::Eq,
            phases: vec![phase()],
            output_challenge_spec: OutputChallengeSpec::None,
        }));
        let v1 = Vertex::Sumcheck(Box::new(SumcheckVertex {
            id: VertexId(1),
            deps: vec![ClaimId(0)],
            input: InputClaim::Constant(0),
            produces: vec![ClaimId(1)],
            side_effect_claims: vec![],
            formula: formula(),
            degree: 2,
            num_vars: SymbolicExpr::Concrete(10),
            weighting: PublicPolynomial::Eq,
            phases: vec![phase()],
            output_challenge_spec: OutputChallengeSpec::None,
        }));
        let v2 = Vertex::Sumcheck(Box::new(SumcheckVertex {
            id: VertexId(2),
            deps: vec![ClaimId(0)], // fan-out: same claim as v1
            input: InputClaim::Constant(0),
            produces: vec![ClaimId(2)],
            side_effect_claims: vec![],
            formula: formula(),
            degree: 2,
            num_vars: SymbolicExpr::Concrete(10),
            weighting: PublicPolynomial::Eq,
            phases: vec![phase()],
            output_challenge_spec: OutputChallengeSpec::None,
        }));
        let v3 = Vertex::Opening(OpeningVertex {
            id: VertexId(3),
            consumes: ClaimId(1),
        });
        let v4 = Vertex::Opening(OpeningVertex {
            id: VertexId(4),
            consumes: ClaimId(2),
        });

        let graph = ClaimGraph {
            polynomials: vec![],
            claims: vec![],
            vertices: vec![v0, v1, v2, v3, v4],
        };

        let errors = graph.validate(&HashSet::new());
        assert!(
            errors.is_empty(),
            "fan-out should be valid, got: {:?}",
            errors
        );
    }

    #[test]
    fn full_graph_challenge_specs_valid() {
        use crate::protocol::build::{build_jolt_protocol, ProtocolConfig};

        let graph = build_jolt_protocol(ProtocolConfig {
            d_instr: 8,
            d_bc: 4,
            d_ram: 3,
            d_instr_chunks_per_virtual: 2,
            n_lookup_tables: 41,
            n_circuit_flags: 14,
            n_advice: 0,
        });
        let errors = graph.validate_challenge_specs();
        assert!(errors.is_empty(), "challenge spec errors: {:?}", errors);
    }

    #[test]
    fn full_graph_claim_flow_valid() {
        use crate::protocol::build::{build_jolt_protocol, ProtocolConfig};

        let graph = build_jolt_protocol(ProtocolConfig {
            d_instr: 8,
            d_bc: 4,
            d_ram: 3,
            d_instr_chunks_per_virtual: 2,
            n_lookup_tables: 41,
            n_circuit_flags: 14,
            n_advice: 0,
        });
        let errors = graph.validate_claim_flow();
        assert!(errors.is_empty(), "claim flow errors: {:?}", errors);
    }

    #[test]
    fn full_graph_single_opening_point() {
        use crate::protocol::build::{build_jolt_protocol, ProtocolConfig};

        let graph = build_jolt_protocol(ProtocolConfig {
            d_instr: 8,
            d_bc: 4,
            d_ram: 3,
            d_instr_chunks_per_virtual: 2,
            n_lookup_tables: 41,
            n_circuit_flags: 14,
            n_advice: 0,
        });
        let error = graph.validate_point_convergence();
        assert!(
            error.is_none(),
            "all opening claims should converge to single point: {error:?}"
        );
    }

    #[test]
    fn eval_ordering_detects_non_dense_ids() {
        // Two claims but with IDs 0 and 5 — gap means non-dense
        let graph = ProtocolGraph {
            claim_graph: ClaimGraph {
                polynomials: vec![],
                claims: vec![
                    simple_claim(0, PolynomialId::RamInc),
                    simple_claim(5, PolynomialId::RdInc),
                ],
                vertices: vec![],
            },
            staging: Staging {
                stages: vec![],
                opening: OpeningStage {
                    vertices: vec![],
                    point: SymbolicPoint::Challenges(StageId(0)),
                    reduction: ReductionStrategy::Rlc,
                    opening_groups: vec![],
                },
            },
            commitment: CommitmentStrategy {
                groups: vec![],
                transcript_order: vec![],
            },
        };
        let errors = graph.validate_eval_ordering();
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            EvalOrderingError::NonDenseClaimIds {
                num_claims: 2,
                max_id: 5,
            }
        ));
    }

    #[test]
    fn eval_ordering_detects_duplicate_produced_claim_in_stage() {
        use crate::builder::ExprBuilder;

        let b = ExprBuilder::new();
        let dummy_expr = b.build(b.zero());
        let formula = || ClaimFormula {
            definition: ClaimDefinition {
                expr: dummy_expr.clone(),
                opening_bindings: vec![],
                num_challenges: 0,
            },
            opening_claims: HashMap::new(),
        };
        let phase = || Phase {
            num_vars: SymbolicExpr::Concrete(10),
            variable_group: VariableGroup::Cycle,
        };

        // Two vertices in the same stage both produce ClaimId(0)
        let v0 = Vertex::Sumcheck(Box::new(SumcheckVertex {
            id: VertexId(0),
            deps: vec![],
            input: InputClaim::Constant(0),
            produces: vec![ClaimId(0)],
            side_effect_claims: vec![],
            formula: formula(),
            degree: 2,
            num_vars: SymbolicExpr::Concrete(10),
            weighting: PublicPolynomial::Eq,
            phases: vec![phase()],
            output_challenge_spec: OutputChallengeSpec::None,
        }));
        let v1 = Vertex::Sumcheck(Box::new(SumcheckVertex {
            id: VertexId(1),
            deps: vec![],
            input: InputClaim::Constant(0),
            produces: vec![ClaimId(0)],
            side_effect_claims: vec![],
            formula: formula(),
            degree: 2,
            num_vars: SymbolicExpr::Concrete(10),
            weighting: PublicPolynomial::Eq,
            phases: vec![phase()],
            output_challenge_spec: OutputChallengeSpec::None,
        }));

        let graph = ProtocolGraph {
            claim_graph: ClaimGraph {
                polynomials: vec![],
                claims: vec![simple_claim(0, PolynomialId::RamInc)],
                vertices: vec![v0, v1],
            },
            staging: Staging {
                stages: vec![Stage {
                    id: StageId(0),
                    vertices: vec![VertexId(0), VertexId(1)],
                    challenge_point: ChallengePoint {
                        num_vars: SymbolicExpr::Concrete(10),
                    },
                    batching: vec![],
                    pre_squeeze: vec![],
                }],
                opening: OpeningStage {
                    vertices: vec![],
                    point: SymbolicPoint::Challenges(StageId(0)),
                    reduction: ReductionStrategy::Rlc,
                    opening_groups: vec![],
                },
            },
            commitment: CommitmentStrategy {
                groups: vec![],
                transcript_order: vec![],
            },
        };
        let errors = graph.validate_eval_ordering();
        assert!(errors.iter().any(|e| matches!(
            e,
            EvalOrderingError::DuplicateProducedClaim {
                stage: StageId(0),
                claim: ClaimId(0),
            }
        )));
    }

    #[test]
    fn full_graph_eval_ordering_valid() {
        use crate::protocol::build::{build_jolt_protocol, ProtocolConfig};

        let graph = build_jolt_protocol(ProtocolConfig {
            d_instr: 8,
            d_bc: 4,
            d_ram: 3,
            d_instr_chunks_per_virtual: 2,
            n_lookup_tables: 41,
            n_circuit_flags: 14,
            n_advice: 0,
        });
        let errors = graph.validate_eval_ordering();
        assert!(errors.is_empty(), "eval ordering errors: {:?}", errors);
    }
}
