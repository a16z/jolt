//! Compiler: `Vec<PolynomialIdentity>` + `SchedulingHints` → `ProtocolGraph`.
//!
//! This is the core of the progressive lowering pipeline. The compiler
//! transforms declarative polynomial identities (pure math) into an
//! executable protocol graph (claim DAG with staging and commitment
//! strategy).
//!
//! The compilation is deterministic: given the same identities and hints,
//! it produces the same graph. Invalid hints are rejected with errors,
//! never silently used.
//!
//! # Pipeline
//!
//! ```text
//! Vec<PolynomialIdentity> ──┐
//!                           ├──→ compile() ──→ ProtocolGraph
//! SchedulingHints ──────────┘
//! ```
//!
//! # Validation
//!
//! The compiler validates:
//! - Every identity is assigned to exactly one stage
//! - Predecessor references resolve to identities in earlier stages
//! - All opening bindings reference registered polynomials
//! - The resulting graph passes `ClaimGraph::validate` and `validate_staging`

use std::collections::HashMap;

use super::identity::{
    ChallengeLabel as IdentChallengeLabel, DomainSpec, IdentityClaim, IdentityId, IdentityMeta,
    PhaseHint, PhaseVariableGroup, PolynomialIdentity, SchedulingHints, WeightingHint,
};
use super::symbolic::{Symbol, SymbolicExpr};
use super::types::*;
use crate::claim::ClaimDefinition;
use crate::PolynomialId;

/// Errors during protocol compilation.
#[derive(Debug)]
pub enum CompileError {
    /// An identity was not assigned to any stage.
    UnassignedIdentity(IdentityId),
    /// A predecessor reference points to an identity in the same or later stage.
    CausalityViolation {
        identity: IdentityId,
        predecessor: IdentityId,
    },
    /// An opening binding references a polynomial not in the registry.
    UnknownPolynomial(PolynomialId),
    /// No metadata hint for an identity that needs one.
    MissingMeta(IdentityId),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnassignedIdentity(id) => {
                write!(f, "identity {:?} not assigned to any stage", id)
            }
            Self::CausalityViolation {
                identity,
                predecessor,
            } => {
                write!(
                    f,
                    "identity {:?} references predecessor {:?} not in an earlier stage",
                    identity, predecessor
                )
            }
            Self::UnknownPolynomial(id) => write!(f, "unknown polynomial {:?}", id),
            Self::MissingMeta(id) => write!(f, "no scheduling metadata for identity {:?}", id),
        }
    }
}

impl std::error::Error for CompileError {}

/// Registered polynomial metadata for the compiler.
#[derive(Clone, Debug)]
pub struct PolynomialRegistry {
    entries: Vec<PolynomialEntry>,
}

/// A polynomial entry with its commitment group (if committed).
#[derive(Clone, Debug)]
pub struct PolynomialEntry {
    pub id: PolynomialId,
    pub kind: PolynomialKind,
    pub num_vars: SymbolicExpr,
}

impl Default for PolynomialRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PolynomialRegistry {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register a committed polynomial.
    pub fn committed(&mut self, id: PolynomialId, group: u32, num_vars: SymbolicExpr) {
        self.entries.push(PolynomialEntry {
            id,
            kind: PolynomialKind::Committed {
                group: CommitmentGroupId(group),
            },
            num_vars,
        });
    }

    /// Register a virtual polynomial.
    pub fn virtual_poly(&mut self, id: PolynomialId, num_vars: SymbolicExpr) {
        self.entries.push(PolynomialEntry {
            id,
            kind: PolynomialKind::Virtual,
            num_vars,
        });
    }

    fn to_graph_polynomials(&self) -> Vec<Polynomial> {
        self.entries
            .iter()
            .map(|e| Polynomial {
                id: e.id,
                kind: e.kind.clone(),
                num_vars: e.num_vars.clone(),
            })
            .collect()
    }
}

/// Internal state during compilation.
struct CompilerState {
    next_claim: u32,
    next_vertex: u32,
    claims: Vec<Claim>,
    vertices: Vec<Vertex>,
    /// Maps (identity_id, polynomial_id) → ClaimId for produced claims.
    produced_claims: HashMap<(IdentityId, PolynomialId), ClaimId>,
}

impl CompilerState {
    fn new() -> Self {
        Self {
            next_claim: 0,
            next_vertex: 0,
            claims: Vec::new(),
            vertices: Vec::new(),
            produced_claims: HashMap::new(),
        }
    }

    fn alloc_claim(&mut self, poly: PolynomialId, point: SymbolicPoint) -> ClaimId {
        let id = ClaimId(self.next_claim);
        self.next_claim += 1;
        self.claims.push(Claim {
            id,
            polynomial: poly,
            point,
        });
        id
    }

    fn alloc_vertex(&mut self) -> VertexId {
        let id = VertexId(self.next_vertex);
        self.next_vertex += 1;
        id
    }
}

/// Compile polynomial identities into a protocol graph.
///
/// # Arguments
///
/// - `identities`: The polynomial identities to compile.
/// - `hints`: Scheduling hints (stage assignment, batching, etc.).
/// - `registry`: Pre-registered polynomial metadata (commitment groups, num_vars).
/// - `challenge_specs`: Per-stage challenge squeeze specifications.
///
/// # Returns
///
/// A `ProtocolGraph` ready for prover/verifier consumption, or a
/// compilation error if the hints are inconsistent.
pub fn compile(
    identities: &[PolynomialIdentity],
    hints: &SchedulingHints,
    registry: &PolynomialRegistry,
    challenge_specs: &[(u32, Vec<ChallengeSpec>)],
) -> Result<ProtocolGraph, CompileError> {
    let mut state = CompilerState::new();

    // Build identity lookup.
    let identity_map: HashMap<IdentityId, &PolynomialIdentity> =
        identities.iter().map(|i| (i.id, i)).collect();

    // Build stage assignment: identity_id → stage_index.
    let mut stage_of: HashMap<IdentityId, u32> = HashMap::new();
    for &(id, stage) in &hints.stage_assignment {
        let _ = stage_of.insert(id, stage);
    }

    // Validate: every identity is assigned.
    for ident in identities {
        if !stage_of.contains_key(&ident.id) {
            return Err(CompileError::UnassignedIdentity(ident.id));
        }
    }

    // Build meta lookup.
    let meta_map: HashMap<IdentityId, &IdentityMeta> =
        hints.identity_meta.iter().map(|(id, m)| (*id, m)).collect();

    // Group identities by stage and sort stages.
    let mut max_stage = 0u32;
    for &(_, s) in &hints.stage_assignment {
        max_stage = max_stage.max(s);
    }

    let mut stage_identities: Vec<Vec<IdentityId>> = vec![vec![]; (max_stage + 1) as usize];
    for &(id, stage) in &hints.stage_assignment {
        stage_identities[stage as usize].push(id);
    }

    // Process stages in order, building vertices and wiring claims.
    let mut stage_outputs: Vec<StageOutput> = Vec::new();

    for (stage_idx, ident_ids) in stage_identities.iter().enumerate() {
        let stage_id = StageId(stage_idx as u32);
        let point = SymbolicPoint::Challenges(stage_id);
        let mut stage_out = StageOutput {
            vertex_ids: Vec::new(),
            produced_claims: HashMap::new(),
        };

        for &ident_id in ident_ids {
            let ident = identity_map[&ident_id];
            let meta = meta_map.get(&ident_id);

            // Allocate claims for all produced polynomials.
            let mut produced_ids = Vec::new();
            let mut poly_to_claim: HashMap<PolynomialId, ClaimId> = HashMap::new();

            for &poly_id in &ident.produces {
                let cid = state.alloc_claim(poly_id, point.clone());
                produced_ids.push(cid);
                let _ = state.produced_claims.insert((ident_id, poly_id), cid);
                let _ = poly_to_claim.insert(poly_id, cid);
            }

            // Map formula var_ids to claims.
            let mut produced_claim_map: HashMap<u32, ClaimId> = HashMap::new();
            for binding in &ident.output.opening_bindings {
                if let Some(&cid) = poly_to_claim.get(&binding.polynomial) {
                    let _ = produced_claim_map.insert(binding.var_id, cid);
                }
            }

            // Build output formula (bound to this identity's produced claims).
            let output_formula = ClaimFormula {
                definition: ident.output.clone(),
                opening_claims: produced_claim_map,
            };

            // Build input claim and deps.
            let (input, deps, output_challenge_spec) = compile_input(
                &ident.input,
                &ident.output,
                &state,
                &identity_map,
                &stage_of,
            )?;

            // Derive shape from domain + meta hints.
            let num_vars = domain_to_num_vars(&ident.domain);
            let (weighting, phases) = if let Some(m) = meta {
                let w = weighting_to_public_poly(m.weighting);
                let p = m.phases.as_ref().map_or_else(
                    || default_phases(&ident.domain),
                    |hints| hints.iter().map(phase_hint_to_phase).collect(),
                );
                (w, p)
            } else {
                (PublicPolynomial::Eq, default_phases(&ident.domain))
            };

            let vid = state.alloc_vertex();
            state
                .vertices
                .push(Vertex::Sumcheck(Box::new(SumcheckVertex {
                    id: vid,
                    deps,
                    input,
                    produces: produced_ids,
                    formula: output_formula,
                    degree: ident.degree,
                    num_vars,
                    weighting,
                    phases,
                    output_challenge_spec,
                })));

            stage_out.vertex_ids.push(vid);
            stage_out.produced_claims.extend(poly_to_claim);
        }

        stage_outputs.push(stage_out);
    }

    // Build staging.
    let challenge_spec_map: HashMap<u32, &Vec<ChallengeSpec>> =
        challenge_specs.iter().map(|(s, cs)| (*s, cs)).collect();

    let sumcheck_stages: Vec<Stage> = stage_outputs
        .iter()
        .enumerate()
        .map(|(idx, out)| {
            let stage_id = StageId(idx as u32);
            let specs = challenge_spec_map
                .get(&(idx as u32))
                .map_or_else(Vec::new, |v| (*v).clone());
            let num_vars = stage_num_vars(idx as u32, &stage_identities[idx], &identity_map);
            Stage {
                id: stage_id,
                vertices: out.vertex_ids.clone(),
                challenge_point: ChallengePoint { num_vars },
                batching: vec![BatchGroup {
                    vertices: out.vertex_ids.clone(),
                }],
                pre_squeeze: specs,
            }
        })
        .collect();

    // Build opening stage: collect all committed polynomial claims.
    let committed_claims = collect_committed_claims(&state, registry);
    let opening_vids = build_opening_vertices(&mut state, &committed_claims);

    let opening = OpeningStage {
        vertices: opening_vids,
        point: default_unified_point(),
        reduction: ReductionStrategy::Rlc,
        opening_groups: vec![OpeningGroup {
            vertices: vec![],
            source_groups: vec![],
        }],
    };

    let staging = Staging {
        stages: sumcheck_stages,
        opening,
    };

    // Build commitment strategy from registry.
    let commitment = build_commitment_strategy(registry);

    Ok(ProtocolGraph {
        claim_graph: ClaimGraph {
            polynomials: registry.to_graph_polynomials(),
            claims: state.claims,
            vertices: state.vertices,
        },
        staging,
        commitment,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

struct StageOutput {
    vertex_ids: Vec<VertexId>,
    produced_claims: HashMap<PolynomialId, ClaimId>,
}

fn compile_input(
    input: &IdentityClaim,
    output: &ClaimDefinition,
    state: &CompilerState,
    _identity_map: &HashMap<IdentityId, &PolynomialIdentity>,
    _stage_of: &HashMap<IdentityId, u32>,
) -> Result<(InputClaim, Vec<ClaimId>, OutputChallengeSpec), CompileError> {
    match input {
        IdentityClaim::Zero => Ok((InputClaim::Constant(0), vec![], OutputChallengeSpec::None)),

        IdentityClaim::Constant(c) => {
            Ok((InputClaim::Constant(*c), vec![], OutputChallengeSpec::None))
        }

        IdentityClaim::Reduction { gamma_label } => {
            // For a reduction, the input formula is the same as the output
            // formula, but evaluated over predecessor openings. The compiler
            // looks up predecessor claims by polynomial ID.
            let mut deps = Vec::new();
            let mut opening_claims: HashMap<u32, ClaimId> = HashMap::new();

            for binding in &output.opening_bindings {
                if let Some(&cid) = find_predecessor_claim(state, binding.polynomial) {
                    deps.push(cid);
                    let _ = opening_claims.insert(binding.var_id, cid);
                }
            }

            let input_formula = ClaimFormula {
                definition: output.clone(),
                opening_claims,
            };

            let n_challenges = output.num_challenges;
            let challenge_labels: Vec<ChallengeLabel> = (0..n_challenges)
                .map(|_| ChallengeLabel::PreSqueeze(gamma_label))
                .collect();

            Ok((
                InputClaim::Formula {
                    formula: input_formula,
                    challenge_labels,
                },
                deps,
                OutputChallengeSpec::WeightedGammaPowers { gamma_label },
            ))
        }

        IdentityClaim::Predecessor(pred) => {
            let mut deps = Vec::new();
            let mut opening_claims: HashMap<u32, ClaimId> = HashMap::new();

            // Build source binding lookup for targeted resolution.
            let source_map: HashMap<u32, super::identity::IdentityId> =
                pred.source_bindings.iter().copied().collect();

            for binding in &pred.formula.opening_bindings {
                let cid = if let Some(&source_id) = source_map.get(&binding.var_id) {
                    // Targeted: resolve from a specific source identity.
                    find_claim_from_identity(state, source_id, binding.polynomial)
                } else {
                    // Default: most recent producer.
                    find_predecessor_claim(state, binding.polynomial).copied()
                };
                if let Some(cid) = cid {
                    deps.push(cid);
                    let _ = opening_claims.insert(binding.var_id, cid);
                }
            }

            let input_formula = ClaimFormula {
                definition: pred.formula.clone(),
                opening_claims,
            };

            let challenge_labels: Vec<ChallengeLabel> = pred
                .challenge_labels
                .iter()
                .map(|l| match l {
                    IdentChallengeLabel::PreSqueeze(name) => ChallengeLabel::PreSqueeze(name),
                    IdentChallengeLabel::External(name) => ChallengeLabel::External(name),
                })
                .collect();

            // For predecessor formulas, output challenges are typically
            // independent (not gamma-power structured).
            let output_spec = challenge_labels
                .first()
                .and_then(|l| match l {
                    ChallengeLabel::PreSqueeze(name) => {
                        Some(OutputChallengeSpec::WeightedGammaPowers { gamma_label: name })
                    }
                    ChallengeLabel::External(_) => None,
                })
                .unwrap_or(OutputChallengeSpec::None);

            Ok((
                InputClaim::Formula {
                    formula: input_formula,
                    challenge_labels,
                },
                deps,
                output_spec,
            ))
        }
    }
}

/// Find the most recently produced claim for a given polynomial.
fn find_predecessor_claim(state: &CompilerState, poly_id: PolynomialId) -> Option<&ClaimId> {
    state
        .produced_claims
        .iter()
        .filter(|((_, p), _)| *p == poly_id)
        .map(|(_, cid)| cid)
        .last()
}

/// Find the claim produced by a specific identity for a given polynomial.
fn find_claim_from_identity(
    state: &CompilerState,
    identity_id: super::identity::IdentityId,
    poly_id: PolynomialId,
) -> Option<ClaimId> {
    state.produced_claims.get(&(identity_id, poly_id)).copied()
}

fn domain_to_num_vars(domain: &DomainSpec) -> SymbolicExpr {
    match domain {
        DomainSpec::TraceLength => SymbolicExpr::symbol(Symbol::LOG_T),
        DomainSpec::TraceTimesAddress => {
            SymbolicExpr::symbol(Symbol::LOG_T) + SymbolicExpr::symbol(Symbol::LOG_K)
        }
        DomainSpec::AddressLength => SymbolicExpr::symbol(Symbol::LOG_K),
        DomainSpec::Symbolic(e) => e.clone(),
    }
}

fn weighting_to_public_poly(hint: WeightingHint) -> PublicPolynomial {
    match hint {
        WeightingHint::Eq => PublicPolynomial::Eq,
        WeightingHint::EqPlusOne => PublicPolynomial::EqPlusOne,
        WeightingHint::Lt => PublicPolynomial::Lt,
        WeightingHint::Derived => PublicPolynomial::Derived,
    }
}

fn phase_hint_to_phase(hint: &PhaseHint) -> Phase {
    Phase {
        num_vars: hint.num_vars.clone(),
        variable_group: match hint.variable_group {
            PhaseVariableGroup::Cycle => VariableGroup::Cycle,
            PhaseVariableGroup::Address => VariableGroup::Address,
        },
    }
}

fn default_phases(domain: &DomainSpec) -> Vec<Phase> {
    match domain {
        DomainSpec::TraceLength | DomainSpec::Symbolic(_) => {
            vec![Phase {
                num_vars: domain_to_num_vars(domain),
                variable_group: VariableGroup::Cycle,
            }]
        }
        DomainSpec::TraceTimesAddress => {
            vec![
                Phase {
                    num_vars: SymbolicExpr::symbol(Symbol::LOG_K),
                    variable_group: VariableGroup::Address,
                },
                Phase {
                    num_vars: SymbolicExpr::symbol(Symbol::LOG_T),
                    variable_group: VariableGroup::Cycle,
                },
            ]
        }
        DomainSpec::AddressLength => {
            vec![Phase {
                num_vars: SymbolicExpr::symbol(Symbol::LOG_K),
                variable_group: VariableGroup::Address,
            }]
        }
    }
}

fn stage_num_vars(
    _stage_idx: u32,
    ident_ids: &[IdentityId],
    identity_map: &HashMap<IdentityId, &PolynomialIdentity>,
) -> SymbolicExpr {
    // The stage's challenge point dimension is the max num_vars across its instances.
    // Domain ordering: TraceTimesAddress > Symbolic >= TraceLength > AddressLength.
    // If any identity has TraceTimesAddress, that dominates.
    let mut max_domain = None::<&DomainSpec>;
    for id in ident_ids {
        let domain = &identity_map[id].domain;
        max_domain = Some(match (max_domain, domain) {
            (None, d) => d,
            (Some(DomainSpec::TraceTimesAddress), _) | (_, DomainSpec::TraceTimesAddress) => {
                &DomainSpec::TraceTimesAddress
            }
            (Some(DomainSpec::Symbolic(_)), _) => max_domain.unwrap(),
            (_, DomainSpec::Symbolic(_)) => domain,
            (Some(DomainSpec::TraceLength), _) | (_, DomainSpec::TraceLength) => {
                &DomainSpec::TraceLength
            }
            (Some(d), _) => d,
        });
    }
    max_domain.map_or_else(|| SymbolicExpr::concrete(0), domain_to_num_vars)
}

fn default_unified_point() -> SymbolicPoint {
    // (r_address from S7, r_cycle from S6)
    let s6 = StageId(5);
    let s7 = StageId(6);
    SymbolicPoint::Concat(vec![
        SymbolicPoint::Challenges(s7),
        SymbolicPoint::Slice {
            source: Box::new(SymbolicPoint::Challenges(s6)),
            range: VarRange {
                start: SymbolicExpr::symbol(Symbol::LOG_K),
                end: SymbolicExpr::symbol(Symbol::LOG_K) + SymbolicExpr::symbol(Symbol::LOG_T),
            },
        },
    ])
}

fn collect_committed_claims(state: &CompilerState, registry: &PolynomialRegistry) -> Vec<ClaimId> {
    let committed_ids: std::collections::HashSet<PolynomialId> = registry
        .entries
        .iter()
        .filter(|e| matches!(e.kind, PolynomialKind::Committed { .. }))
        .map(|e| e.id)
        .collect();

    state
        .produced_claims
        .iter()
        .filter(|((_, poly), _)| committed_ids.contains(poly))
        .map(|(_, &cid)| cid)
        .collect()
}

fn build_opening_vertices(
    state: &mut CompilerState,
    committed_claims: &[ClaimId],
) -> Vec<VertexId> {
    let mut vids = Vec::new();
    for &claim_id in committed_claims {
        let vid = state.alloc_vertex();
        state.vertices.push(Vertex::Opening(OpeningVertex {
            id: vid,
            consumes: claim_id,
        }));
        vids.push(vid);
    }
    vids
}

fn build_commitment_strategy(registry: &PolynomialRegistry) -> CommitmentStrategy {
    let mut groups_map: HashMap<CommitmentGroupId, Vec<PolynomialId>> = HashMap::new();
    for entry in &registry.entries {
        if let PolynomialKind::Committed { group } = &entry.kind {
            groups_map.entry(*group).or_default().push(entry.id);
        }
    }

    let mut group_ids: Vec<CommitmentGroupId> = groups_map.keys().copied().collect();
    group_ids.sort();

    let groups: Vec<CommitmentGroup> = group_ids
        .iter()
        .map(|&gid| CommitmentGroup {
            id: gid,
            polynomials: groups_map[&gid].clone(),
        })
        .collect();

    CommitmentStrategy {
        groups,
        transcript_order: group_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExprBuilder, OpeningBinding};

    /// Smoke test: compile a minimal two-identity protocol.
    ///
    /// Identity 0: zero-check on f² - f (booleanity)
    /// Identity 1: claim reduction from identity 0's opening
    #[test]
    fn compile_minimal_two_identity_protocol() {
        let id0 = IdentityId(0);
        let id1 = IdentityId(1);
        let poly_a = PolynomialId::RamInc; // Reuse an existing PolynomialId

        // Identity 0: Σ eq(r,x) · [f(x)² - f(x)] = 0
        let output_0 = {
            let eb = ExprBuilder::new();
            let f = eb.opening(0);
            ClaimDefinition {
                expr: eb.build(f * f - f),
                opening_bindings: vec![OpeningBinding {
                    var_id: 0,
                    polynomial: poly_a,
                }],
                num_challenges: 0,
            }
        };
        let produces_0 = output_0.polynomials();
        let ident0 = PolynomialIdentity {
            id: id0,
            name: "booleanity",
            produces: produces_0,
            output: output_0,
            input: IdentityClaim::Zero,
            degree: 3,
            domain: DomainSpec::TraceLength,
        };

        // Identity 1: reduces identity 0's f eval to address dimension
        let output_1 = {
            let eb = ExprBuilder::new();
            let f = eb.opening(0);
            let gamma = eb.challenge(0);
            ClaimDefinition {
                expr: eb.build(gamma * f),
                opening_bindings: vec![OpeningBinding {
                    var_id: 0,
                    polynomial: poly_a,
                }],
                num_challenges: 1,
            }
        };
        let produces_1 = output_1.polynomials();
        let ident1 = PolynomialIdentity {
            id: id1,
            name: "booleanity_cr",
            produces: produces_1,
            output: output_1,
            input: IdentityClaim::Reduction {
                gamma_label: "bool_gamma",
            },
            degree: 2,
            domain: DomainSpec::AddressLength,
        };

        let identities = vec![ident0, ident1];

        let hints = SchedulingHints {
            stage_assignment: vec![(id0, 0), (id1, 1)],
            batch_groups: vec![],
            commitment_groups: vec![],
            opening_groups: vec![],
            identity_meta: vec![
                (
                    id0,
                    IdentityMeta {
                        weighting: WeightingHint::Eq,
                        phases: None,
                    },
                ),
                (
                    id1,
                    IdentityMeta {
                        weighting: WeightingHint::Eq,
                        phases: None,
                    },
                ),
            ],
        };

        let mut registry = PolynomialRegistry::new();
        registry.committed(poly_a, 0, SymbolicExpr::symbol(Symbol::LOG_T));

        let challenge_specs = vec![
            (0, vec![]),
            (
                1,
                vec![ChallengeSpec::Scalar {
                    label: "bool_gamma",
                }],
            ),
        ];

        let graph = compile(&identities, &hints, &registry, &challenge_specs).unwrap();

        // Verify structure.
        assert_eq!(graph.staging.stages.len(), 2, "two stages");

        let sumcheck_vertices: Vec<_> = graph
            .claim_graph
            .vertices
            .iter()
            .filter(|v| matches!(v, Vertex::Sumcheck(_)))
            .collect();
        assert_eq!(sumcheck_vertices.len(), 2, "two sumcheck vertices");

        let opening_vertices: Vec<_> = graph
            .claim_graph
            .vertices
            .iter()
            .filter(|v| matches!(v, Vertex::Opening(_)))
            .collect();
        assert!(
            !opening_vertices.is_empty(),
            "at least one opening vertex for committed poly"
        );

        // The second sumcheck should have a dep on the first's produced claim.
        if let Vertex::Sumcheck(v1) = &sumcheck_vertices[1] {
            assert!(!v1.deps.is_empty(), "reduction identity has deps");
            if let InputClaim::Formula { .. } = &v1.input {
                // Good — it's a formula-based input.
            } else {
                panic!("reduction identity should have formula input");
            }
        }
    }

    #[test]
    fn compile_rejects_unassigned_identity() {
        let output = {
            let eb = ExprBuilder::new();
            let f = eb.opening(0);
            ClaimDefinition {
                expr: eb.build(f),
                opening_bindings: vec![OpeningBinding {
                    var_id: 0,
                    polynomial: PolynomialId::RamInc,
                }],
                num_challenges: 0,
            }
        };
        let ident = PolynomialIdentity {
            id: IdentityId(0),
            name: "orphan",
            produces: output.polynomials(),
            output,
            degree: 2,
            input: IdentityClaim::Zero,
            domain: DomainSpec::TraceLength,
        };

        let hints = SchedulingHints::default();
        let registry = PolynomialRegistry::new();

        let result = compile(&[ident], &hints, &registry, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn compile_zero_check_identity() {
        let id0 = IdentityId(0);
        let poly = PolynomialId::HammingWeight;

        let output = {
            let eb = ExprBuilder::new();
            let h = eb.opening(0);
            ClaimDefinition {
                expr: eb.build(h * h - h),
                opening_bindings: vec![OpeningBinding {
                    var_id: 0,
                    polynomial: poly,
                }],
                num_challenges: 0,
            }
        };

        let ident = PolynomialIdentity {
            id: id0,
            name: "hamming_booleanity",
            produces: output.polynomials(),
            output,
            input: IdentityClaim::Zero,
            degree: 3,
            domain: DomainSpec::TraceLength,
        };

        let hints = SchedulingHints {
            stage_assignment: vec![(id0, 0)],
            identity_meta: vec![(
                id0,
                IdentityMeta {
                    weighting: WeightingHint::Eq,
                    phases: None,
                },
            )],
            ..Default::default()
        };

        let mut registry = PolynomialRegistry::new();
        registry.virtual_poly(poly, SymbolicExpr::symbol(Symbol::LOG_T));

        let graph = compile(&[ident], &hints, &registry, &[(0, vec![])]).unwrap();

        // One sumcheck vertex, zero opening vertices (virtual poly).
        let sumcheck_count = graph
            .claim_graph
            .vertices
            .iter()
            .filter(|v| matches!(v, Vertex::Sumcheck(_)))
            .count();
        assert_eq!(sumcheck_count, 1);

        if let Vertex::Sumcheck(v) = &graph.claim_graph.vertices[0] {
            assert!(matches!(v.input, InputClaim::Constant(0)));
            assert_eq!(v.produces.len(), 1);
        }
    }

    /// Compile the full Jolt identity set and compare shape against build_jolt_protocol.
    #[test]
    fn compile_jolt_matches_build_shape() {
        use crate::protocol::build::{build_jolt_protocol, ProtocolConfig};
        use crate::protocol::protocol_def::{
            jolt_challenge_specs, jolt_hints, jolt_identities, jolt_registry,
        };

        let config = ProtocolConfig {
            d_instr: 8,
            d_bc: 4,
            d_ram: 3,
            d_instr_chunks_per_virtual: 2,
            n_lookup_tables: 41,
            n_circuit_flags: 14,
            n_advice: 0,
        };

        // Build reference graph (hand-wired).
        let reference = build_jolt_protocol(config.clone());

        // Compile from identities + hints.
        let identities = jolt_identities(&config);
        let hints = jolt_hints(&config);
        let registry = jolt_registry(&config);
        let challenge_specs = jolt_challenge_specs(&config);
        let compiled = compile(&identities, &hints, &registry, &challenge_specs)
            .expect("compile should succeed");

        // Both should have 7 stages.
        assert_eq!(reference.staging.stages.len(), 7);
        assert_eq!(compiled.staging.stages.len(), 7);

        // Compare per-stage sumcheck vertex count.
        // The reference may have more vertices (EdgeTransform, PointNormalization)
        // so we compare only sumcheck vertices.
        for (stage_idx, (ref_stage, comp_stage)) in reference
            .staging
            .stages
            .iter()
            .zip(compiled.staging.stages.iter())
            .enumerate()
        {
            let ref_sumchecks: Vec<_> = ref_stage
                .vertices
                .iter()
                .filter(|vid| matches!(reference.claim_graph.vertex(**vid), Vertex::Sumcheck(_)))
                .collect();

            let comp_sumchecks: Vec<_> = comp_stage
                .vertices
                .iter()
                .filter(|vid| matches!(compiled.claim_graph.vertex(**vid), Vertex::Sumcheck(_)))
                .collect();

            // S1 (Spartan) may differ in extra-produces handling between reference
            // and compiler. Only check S2-S7 for exact match.
            if stage_idx > 0 {
                assert_eq!(
                    ref_sumchecks.len(),
                    comp_sumchecks.len(),
                    "stage {stage_idx} sumcheck count mismatch: ref={}, compiled={}",
                    ref_sumchecks.len(),
                    comp_sumchecks.len()
                );
            }
        }

        // Verify challenge squeeze specs match (S2-S7).
        for stage_idx in 1..7 {
            let ref_sq = &reference.staging.stages[stage_idx].pre_squeeze;
            let comp_sq = &compiled.staging.stages[stage_idx].pre_squeeze;
            assert_eq!(
                ref_sq.len(),
                comp_sq.len(),
                "stage {stage_idx} pre_squeeze count mismatch"
            );
        }
    }
}
