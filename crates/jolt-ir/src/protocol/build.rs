//! Protocol graph construction for the Jolt zkVM.
//!
//! [`build_jolt_protocol`] constructs the full [`ProtocolGraph`] — all polynomials,
//! claims, vertices, staging, and commitment strategy — from a [`ProtocolConfig`].
//! The resulting graph validates cleanly via [`ProtocolGraph::validate_staging`]
//! and [`ClaimGraph::validate`].

use std::collections::HashMap;

use super::symbolic::{NumVars, Symbol, SymbolicExpr};
use super::types::*;
use crate::claim::ClaimDefinition;
use crate::zkvm::claims;
use crate::{ExprBuilder, OpeningBinding, PolynomialId};

/// Structural parameters that determine the graph's shape.
///
/// These are known at preprocessing time. Symbolic quantities (`log_T`, `log_k`)
/// stay symbolic; only the chunk counts (which determine vertex/claim COUNT) are
/// concrete.
#[derive(Clone, Debug)]
pub struct ProtocolConfig {
    /// Total committed instruction RA chunks: `LOG_K / log_k_chunk`.
    pub d_instr: usize,
    /// Bytecode RA chunks.
    pub d_bc: usize,
    /// RAM RA chunks.
    pub d_ram: usize,
    /// Committed chunks per virtual RA polynomial for instruction lookups.
    ///
    /// `lookups_ra_virtual_log_k_chunk / log_k_chunk` in jolt-core.
    /// Each virtual RA poly is a product of this many committed chunks.
    /// Must divide `d_instr` evenly.
    pub d_instr_chunks_per_virtual: usize,
    /// Number of instruction lookup tables (jolt-core `NUM_LOOKUP_TABLES`).
    pub n_lookup_tables: usize,
    /// Number of circuit flags (jolt-core `NUM_CIRCUIT_FLAGS`).
    pub n_circuit_flags: usize,
    /// Number of advice polynomials (0 = no advice).
    pub n_advice: usize,
}

impl ProtocolConfig {
    pub fn d_total(&self) -> usize {
        self.d_instr + self.d_bc + self.d_ram
    }

    /// Number of virtual instruction RA polynomials.
    ///
    /// Each virtual poly is a product of `d_instr_chunks_per_virtual` committed chunks.
    pub fn n_virtual_instr_ra(&self) -> usize {
        self.d_instr / self.d_instr_chunks_per_virtual
    }

    /// All committed RA polynomial IDs, in order: instruction, bytecode, RAM.
    fn all_ra_poly_ids(&self) -> Vec<PolynomialId> {
        (0..self.d_instr)
            .map(PolynomialId::InstructionRa)
            .chain((0..self.d_bc).map(PolynomialId::BytecodeRa))
            .chain((0..self.d_ram).map(PolynomialId::RamRa))
            .collect()
    }
}

const S1: StageId = StageId(0);
const S2: StageId = StageId(1);
const S3: StageId = StageId(2);
const S4: StageId = StageId(3);
const S5: StageId = StageId(4);
const S6: StageId = StageId(5);
const S7: StageId = StageId(6);

fn r_cycle() -> SymbolicPoint {
    SymbolicPoint::Slice {
        source: Box::new(SymbolicPoint::Challenges(S1)),
        range: VarRange {
            start: SymbolicExpr::concrete(0),
            end: SymbolicExpr::symbol(Symbol::LOG_T),
        },
    }
}

fn r_y() -> SymbolicPoint {
    SymbolicPoint::Challenges(S1)
}

fn unified_point() -> SymbolicPoint {
    SymbolicPoint::Concat(vec![
        SymbolicPoint::Challenges(S7),
        SymbolicPoint::Challenges(S6),
    ])
}

type StageClaims = HashMap<PolynomialId, ClaimId>;

struct StageOutput {
    vertex_ids: Vec<VertexId>,
    claims: StageClaims,
}

impl StageOutput {
    fn new() -> Self {
        Self {
            vertex_ids: Vec::new(),
            claims: HashMap::new(),
        }
    }
}

struct GraphBuilder {
    next_claim: u32,
    next_vertex: u32,
    polynomials: Vec<Polynomial>,
    claims: Vec<Claim>,
    vertices: Vec<Vertex>,
}

impl GraphBuilder {
    fn new() -> Self {
        Self {
            next_claim: 0,
            next_vertex: 0,
            polynomials: Vec::new(),
            claims: Vec::new(),
            vertices: Vec::new(),
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

    fn register_poly(&mut self, id: PolynomialId, kind: PolynomialKind, num_vars: NumVars) {
        self.polynomials.push(Polynomial { id, kind, num_vars });
    }

    fn push_vertex(&mut self, v: Vertex) {
        self.vertices.push(v);
    }
}

/// Bind a ClaimDefinition's opening variables to concrete ClaimIds.
///
/// `poly_to_claim` maps `PolynomialId` → `ClaimId` for each opening binding.
fn bind_formula(def: ClaimDefinition, poly_to_claim: &StageClaims) -> ClaimFormula {
    let opening_claims = def
        .opening_bindings
        .iter()
        .map(|b| (b.var_id, poly_to_claim[&b.polynomial]))
        .collect();
    ClaimFormula {
        definition: def,
        opening_claims,
    }
}

fn log_t() -> NumVars {
    SymbolicExpr::symbol(Symbol::LOG_T)
}

fn log_ra() -> NumVars {
    SymbolicExpr::symbol(Symbol::LOG_T) + SymbolicExpr::symbol(Symbol::LOG_K)
}

fn register_all_polynomials(b: &mut GraphBuilder, config: &ProtocolConfig) {
    let committed = |group: u32| PolynomialKind::Committed {
        group: CommitmentGroupId(group),
    };

    // Group 0: SpartanWitness
    b.register_poly(
        PolynomialId::SpartanWitness,
        committed(0),
        SymbolicExpr::symbol(Symbol::LOG_ROWS) + SymbolicExpr::symbol(Symbol::LOG_COLS),
    );

    // Group 1: RamInc, Group 2: RdInc
    b.register_poly(PolynomialId::RamInc, committed(1), log_t());
    b.register_poly(PolynomialId::RdInc, committed(2), log_t());

    // Groups 3..: RA polynomials (one group per polynomial)
    let mut group_id = 3u32;
    for i in 0..config.d_instr {
        b.register_poly(
            PolynomialId::InstructionRa(i),
            committed(group_id),
            log_ra(),
        );
        group_id += 1;
    }
    for i in 0..config.d_bc {
        b.register_poly(PolynomialId::BytecodeRa(i), committed(group_id), log_ra());
        group_id += 1;
    }
    for i in 0..config.d_ram {
        b.register_poly(PolynomialId::RamRa(i), committed(group_id), log_ra());
        group_id += 1;
    }

    // Advice polynomials
    if config.n_advice >= 1 {
        b.register_poly(PolynomialId::TrustedAdvice, committed(group_id), log_t());
        group_id += 1;
    }
    if config.n_advice >= 2 {
        b.register_poly(PolynomialId::UntrustedAdvice, committed(group_id), log_t());
        // group_id += 1; // unused after this
    }

    // Virtual polynomials (no commitment group)
    let virt = PolynomialKind::Virtual;
    let all_virtual = [
        PolynomialId::RamReadValue,
        PolynomialId::RamWriteValue,
        PolynomialId::RamAddress,
        PolynomialId::RamVal,
        PolynomialId::RamValFinal,
        PolynomialId::HammingWeight,
        PolynomialId::RdWriteValue,
        PolynomialId::Rs1Value,
        PolynomialId::Rs2Value,
        PolynomialId::RegistersVal,
        PolynomialId::Rs1Ra,
        PolynomialId::Rs2Ra,
        PolynomialId::RdWa,
        PolynomialId::LookupOutput,
        PolynomialId::LeftLookupOperand,
        PolynomialId::RightLookupOperand,
        PolynomialId::LeftInstructionInput,
        PolynomialId::RightInstructionInput,
        PolynomialId::IsRdNotZero,
        PolynomialId::WriteLookupToRdFlag,
        PolynomialId::JumpFlag,
        PolynomialId::BranchFlag,
        PolynomialId::LeftIsRs1,
        PolynomialId::LeftIsPc,
        PolynomialId::RightIsRs2,
        PolynomialId::RightIsImm,
        PolynomialId::UnexpandedPc,
        PolynomialId::Imm,
        PolynomialId::NextUnexpandedPc,
        PolynomialId::NextPc,
        PolynomialId::NextIsVirtual,
        PolynomialId::NextIsFirstInSequence,
        PolynomialId::NextIsNoop,
    ];
    for id in all_virtual {
        b.register_poly(id, virt.clone(), log_t());
    }

    // Circuit operation flags (OpFlag) — used by BytecodeReadRaf.
    // Indices 5,6,7,12 are already covered by named variants above.
    for i in [0, 1, 2, 3, 4, 8, 9, 10, 11, 13] {
        b.register_poly(PolynomialId::OpFlag(i), virt.clone(), log_t());
    }

    // Expanded PC, InstructionRafFlag
    b.register_poly(PolynomialId::ExpandedPc, virt.clone(), log_t());
    b.register_poly(PolynomialId::InstructionRafFlag, virt.clone(), log_t());

    // Lookup table selection flags (used by BytecodeReadRaf stage 5 contribution)
    for i in 0..config.n_lookup_tables {
        b.register_poly(PolynomialId::LookupTableFlag(i), virt.clone(), log_t());
    }

    // Per-stage bytecode RAF value polys
    for s in 0..5 {
        b.register_poly(PolynomialId::BytecodeReadRafVal(s), virt.clone(), log_ra());
    }
    // Per-stage instruction RAF value polys
    for s in 0..config.d_instr {
        b.register_poly(PolynomialId::InstructionReadRafVal(s), virt.clone(), log_ra());
    }
}

/// All virtual polynomials produced by Spartan at r_cycle.
const SPARTAN_VIRTUAL_OUTPUTS: &[PolynomialId] = &[
    PolynomialId::RamReadValue,
    PolynomialId::RamWriteValue,
    PolynomialId::RamAddress,
    PolynomialId::RamVal,
    PolynomialId::RamValFinal,
    PolynomialId::HammingWeight,
    PolynomialId::RdWriteValue,
    PolynomialId::Rs1Value,
    PolynomialId::Rs2Value,
    PolynomialId::RegistersVal,
    PolynomialId::Rs1Ra,
    PolynomialId::Rs2Ra,
    PolynomialId::RdWa,
    PolynomialId::LookupOutput,
    PolynomialId::LeftLookupOperand,
    PolynomialId::RightLookupOperand,
    PolynomialId::LeftInstructionInput,
    PolynomialId::RightInstructionInput,
    PolynomialId::IsRdNotZero,
    PolynomialId::WriteLookupToRdFlag,
    PolynomialId::JumpFlag,
    PolynomialId::BranchFlag,
    PolynomialId::LeftIsRs1,
    PolynomialId::LeftIsPc,
    PolynomialId::RightIsRs2,
    PolynomialId::RightIsImm,
    PolynomialId::UnexpandedPc,
    PolynomialId::Imm,
    PolynomialId::NextUnexpandedPc,
    PolynomialId::NextPc,
    PolynomialId::NextIsVirtual,
    PolynomialId::NextIsFirstInSequence,
    PolynomialId::NextIsNoop,
];

/// Spartan is modeled as a single composite vertex for now.
/// Internal decomposition (outer/product/inner) is a future refinement.
fn build_spartan(b: &mut GraphBuilder, config: &ProtocolConfig) -> StageOutput {
    let vid = b.alloc_vertex();
    let pt = r_cycle();

    let mut claims = StageClaims::new();
    let mut produced_ids = Vec::new();

    // Virtual polynomial evals at r_cycle
    for &poly_id in SPARTAN_VIRTUAL_OUTPUTS {
        let cid = b.alloc_claim(poly_id, pt.clone());
        let _ = claims.insert(poly_id, cid);
        produced_ids.push(cid);
    }

    // Additional virtual polys needed by BytecodeReadRaf (Stage 1 contribution).
    // OpFlags not already covered by named variants (indices 0-4, 8-11, 13).
    for i in [0, 1, 2, 3, 4, 8, 9, 10, 11, 13] {
        let poly_id = PolynomialId::OpFlag(i);
        let cid = b.alloc_claim(poly_id, pt.clone());
        let _ = claims.insert(poly_id, cid);
        produced_ids.push(cid);
    }
    // Expanded PC (used by BytecodeReadRaf RAF contribution)
    {
        let cid = b.alloc_claim(PolynomialId::ExpandedPc, pt.clone());
        let _ = claims.insert(PolynomialId::ExpandedPc, cid);
        produced_ids.push(cid);
    }
    // InstructionRafFlag + LookupTableFlags (produced at S5, but also need S1-point evaluations
    // for the InstrReadRaf stage contribution to BytecodeReadRaf)
    {
        let cid = b.alloc_claim(PolynomialId::InstructionRafFlag, pt.clone());
        let _ = claims.insert(PolynomialId::InstructionRafFlag, cid);
        produced_ids.push(cid);
    }
    for i in 0..config.n_lookup_tables {
        let poly_id = PolynomialId::LookupTableFlag(i);
        let cid = b.alloc_claim(poly_id, pt.clone());
        let _ = claims.insert(poly_id, cid);
        produced_ids.push(cid);
    }

    // SpartanWitness eval at r_y (committed)
    let wit_claim = b.alloc_claim(PolynomialId::SpartanWitness, r_y());
    let _ = claims.insert(PolynomialId::SpartanWitness, wit_claim);
    produced_ids.push(wit_claim);

    // Spartan vertex: no deps (root of the graph), Constant(0) input (R1CS satisfaction).
    // Formula: identity (Spartan's internal formulas are opaque at this level).
    let identity_formula = {
        let eb = ExprBuilder::new();
        let expr = eb.build(eb.zero());
        ClaimFormula {
            definition: ClaimDefinition {
                expr,
                opening_bindings: vec![],
                num_challenges: 0,
            },
            opening_claims: HashMap::new(),
        }
    };

    b.push_vertex(Vertex::Sumcheck(Box::new(SumcheckVertex {
        id: vid,
        deps: vec![],
        input: InputClaim::Constant(0),
        produces: produced_ids,
        side_effect_claims: vec![],
        formula: identity_formula,
        degree: 3,
        num_vars: SymbolicExpr::symbol(Symbol::LOG_ROWS) + SymbolicExpr::symbol(Symbol::LOG_COLS),
        weighting: PublicPolynomial::Eq,
        phases: vec![
            Phase {
                num_vars: SymbolicExpr::symbol(Symbol::LOG_ROWS),
                variable_group: VariableGroup::Cycle,
            },
            Phase {
                num_vars: SymbolicExpr::symbol(Symbol::LOG_COLS),
                variable_group: VariableGroup::Address,
            },
        ],
        output_challenge_spec: OutputChallengeSpec::None,
    })));

    StageOutput {
        vertex_ids: vec![vid],
        claims,
    }
}

/// Input claim specification.
#[allow(dead_code)]
enum InputSpec {
    /// Constant claimed sum (e.g., zero for booleanity).
    Constant(i64),
    /// Same formula as output, bound to upstream claims. Standard for claim reductions.
    SameFormula {
        upstream: StageClaims,
        gamma_label: &'static str,
    },
    /// Different formula from output, with explicit challenge labels.
    Formula {
        def: ClaimDefinition,
        upstream: StageClaims,
        challenge_labels: Vec<ChallengeLabel>,
    },
}

/// Adds a sumcheck vertex to the graph. Single unified vertex builder.
#[allow(clippy::too_many_arguments)]
fn add_vertex(
    b: &mut GraphBuilder,
    output_def: ClaimDefinition,
    deps: Vec<ClaimId>,
    point: SymbolicPoint,
    input_spec: InputSpec,
    degree: usize,
    num_vars: NumVars,
    weighting: PublicPolynomial,
    phases: Vec<Phase>,
) -> (VertexId, StageClaims) {
    let vid = b.alloc_vertex();

    let mut produced = StageClaims::new();
    let mut produced_ids = Vec::new();
    for binding in &output_def.opening_bindings {
        let cid = b.alloc_claim(binding.polynomial, point.clone());
        let _ = produced.insert(binding.polynomial, cid);
        produced_ids.push(cid);
    }

    let output_formula = bind_formula(output_def.clone(), &produced);

    let (input, output_challenge_spec) = match input_spec {
        InputSpec::Constant(c) => (InputClaim::Constant(c), OutputChallengeSpec::None),
        InputSpec::SameFormula { upstream, gamma_label } => {
            let input_formula = bind_formula(output_def.clone(), &upstream);
            let challenge_labels = (0..output_def.num_challenges)
                .map(|_| ChallengeLabel::PreSqueeze(gamma_label))
                .collect();
            (InputClaim::Formula { formula: input_formula, challenge_labels },
             OutputChallengeSpec::WeightedGammaPowers { gamma_label })
        }
        InputSpec::Formula { def, upstream, challenge_labels } => {
            let input_formula = bind_formula(def, &upstream);
            // For explicit formulas, the output challenge spec defaults to
            // WeightedGammaPowers using the first challenge label's pre_squeeze name.
            let spec = challenge_labels
                .first()
                .and_then(|l| match l {
                    ChallengeLabel::PreSqueeze(name) => {
                        Some(OutputChallengeSpec::WeightedGammaPowers { gamma_label: name })
                    }
                    ChallengeLabel::External(_) => None,
                })
                .unwrap_or(OutputChallengeSpec::None);
            (InputClaim::Formula { formula: input_formula, challenge_labels }, spec)
        }
    };

    b.push_vertex(Vertex::Sumcheck(Box::new(SumcheckVertex {
        id: vid,
        deps,
        input,
        produces: produced_ids,
        side_effect_claims: vec![],
        formula: output_formula,
        degree,
        num_vars: num_vars.clone(),
        weighting,
        phases,
        output_challenge_spec,
    })));

    (vid, produced)
}

/// Allocate side-effect claims for a vertex: evaluations available at the
/// stage's challenge point but not in the output formula. Returns the
/// claims map for downstream consumption.
fn alloc_side_effects(
    b: &mut GraphBuilder,
    point: &SymbolicPoint,
    poly_ids: &[PolynomialId],
) -> (Vec<ClaimId>, StageClaims) {
    let mut claim_ids = Vec::new();
    let mut claims = StageClaims::new();
    for &poly_id in poly_ids {
        let cid = b.alloc_claim(poly_id, point.clone());
        claim_ids.push(cid);
        let _ = claims.insert(poly_id, cid);
    }
    (claim_ids, claims)
}

/// Convenience: single cycle-phase vertex.
fn cycle_phase() -> Vec<Phase> {
    vec![Phase { num_vars: log_t(), variable_group: VariableGroup::Cycle }]
}


#[allow(clippy::too_many_arguments)]
fn add_vertex_cr(
    b: &mut GraphBuilder,
    def: ClaimDefinition,
    upstream: &StageClaims,
    point: SymbolicPoint,
    degree: usize,
    num_vars: NumVars,
    weighting: PublicPolynomial,
    gamma_label: &'static str,
) -> (VertexId, StageClaims) {
    let deps: Vec<ClaimId> = def.opening_bindings.iter()
        .filter_map(|b| upstream.get(&b.polynomial).copied())
        .collect();
    add_vertex(
        b, def, deps, point,
        InputSpec::SameFormula { upstream: upstream.clone(), gamma_label },
        degree, num_vars, weighting, cycle_phase(),
    )
}

fn add_vertex_comp(
    b: &mut GraphBuilder,
    output_def: ClaimDefinition,
    ordering_deps: Vec<ClaimId>,
    point: SymbolicPoint,
    input_spec: InputSpec,
    shape: SumcheckShape,
) -> (VertexId, StageClaims) {
    add_vertex(
        b, output_def, ordering_deps, point, input_spec,
        shape.degree, shape.num_vars, shape.weighting, shape.phases,
    )
}

struct SumcheckShape {
    degree: usize,
    num_vars: NumVars,
    weighting: PublicPolynomial,
    phases: Vec<Phase>,
}

fn build_s2(b: &mut GraphBuilder, _config: &ProtocolConfig, s1: &StageClaims) -> StageOutput {
    let pt = SymbolicPoint::Challenges(S2);
    let mut out = StageOutput::new();

    // V_ram_rw: RAM read-write checking
    // Input: rv + γ·wv from S1 (RamReadValue, RamWriteValue)
    // Output formula: c0·ra·val + c1·ra·inc (different from input!)
    {
        let output_def = claims::ram::ram_read_write_checking();
        // Input claim reads RamReadValue, RamWriteValue from S1
        let input_def = {
            let b2 = ExprBuilder::new();
            let rv = b2.opening(0);
            let wv = b2.opening(1);
            let gamma = b2.challenge(0);
            ClaimDefinition {
                expr: b2.build(rv + gamma * wv),
                opening_bindings: vec![
                    OpeningBinding { var_id: 0, polynomial: PolynomialId::RamReadValue },
                    OpeningBinding { var_id: 1, polynomial: PolynomialId::RamWriteValue },
                ],
                num_challenges: 1,
            }
        };
        let input_upstream = filter_claims(s1, &input_def);
        let ordering_deps = deps_from_formula(&input_def, s1);
        let (vid, claims) = add_vertex(
            b, output_def, ordering_deps, pt.clone(),
            InputSpec::Formula {
                def: input_def,
                upstream: input_upstream,
                challenge_labels: vec![ChallengeLabel::PreSqueeze("ram_rw_gamma")],
            },
            3,
            SymbolicExpr::symbol(Symbol::LOG_K) + SymbolicExpr::symbol(Symbol::LOG_T),
            PublicPolynomial::Eq,
            cycle_phase(),
        );
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
    }

    // V_pv: product virtual (uni-skip first round + remainder)
    // The uni-skip is an alternative sumcheck strategy for the first round.
    // Total rounds: 1 (uni-skip) + log_T (remainder) = 1 + log_T.
    // Input: evaluation of the uni-skip polynomial at the verifier's challenge r₀.
    // This value is computed by the verifier from the proof's first-round polynomial —
    // Spartan-internal, modeled as External.
    {
        let def = claims::spartan::product_virtual_remainder();
        let ordering_deps = deps_from_formula(&def, s1);
        let input_def = {
            let eb = ExprBuilder::new();
            let uni_skip_eval = eb.opening(0);
            ClaimDefinition {
                expr: eb.build(uni_skip_eval),
                opening_bindings: vec![],
                num_challenges: 0,
            }
        };
        let (vid, claims) = add_vertex_comp(
            b, def, ordering_deps, pt.clone(),
            InputSpec::Formula {
                def: input_def,
                upstream: StageClaims::new(),
                challenge_labels: vec![],
            },
            SumcheckShape {
                degree: 3,
                num_vars: SymbolicExpr::concrete(1) + log_t(),
                weighting: PublicPolynomial::Eq,
                phases: vec![
                    Phase { num_vars: SymbolicExpr::concrete(1), variable_group: VariableGroup::Cycle },
                    Phase { num_vars: log_t(), variable_group: VariableGroup::Cycle },
                ],
            },
        );
        // PV side-effect: NextIsVirtual (VirtualInstruction) eval available at S2's point.
        // Used by BytecodeReadRaf Stage 2 contribution.
        let (se_ids, se_claims) = alloc_side_effects(b, &pt, &[PolynomialId::NextIsVirtual]);
        if let Vertex::Sumcheck(ref mut v) = b.vertices[vid.0 as usize] {
            v.side_effect_claims = se_ids;
        }
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
        out.claims.extend(se_claims);
    }

    // V_instr_lookups_cr: instruction lookups claim reduction
    // Input: lo + γ·lop + γ²·rop + γ³·lip + γ⁴·rip from S1
    {
        let def = claims::reductions::instruction_lookups_claim_reduction();
        let upstream = filter_claims(s1, &def);
        let (vid, claims) = add_vertex_cr(
            b, def, &upstream, pt.clone(), 2, log_t(),
            PublicPolynomial::Eq, "instr_cr_gamma",
        );
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
    }

    // V_ram_raf_eval: input = RamAddress(r_s1) * 2^{phase3_cycle_rounds}
    // The scaling factor accounts for internal dummy rounds in RamRW's phase structure.
    {
        let output_def = claims::ram::ram_raf_evaluation();
        let input_def = {
            let eb = ExprBuilder::new();
            let ra = eb.opening(0);
            let scale = eb.challenge(0);
            ClaimDefinition {
                expr: eb.build(scale * ra),
                opening_bindings: vec![OpeningBinding {
                    var_id: 0,
                    polynomial: PolynomialId::RamAddress,
                }],
                num_challenges: 1,
            }
        };
        let input_upstream = filter_claims(s1, &input_def);
        let ordering_deps = deps_from_formula_map(&input_upstream);
        let (vid, claims) = add_vertex_comp(
            b, output_def, ordering_deps, pt.clone(),
            InputSpec::Formula {
                def: input_def,
                upstream: input_upstream,
                challenge_labels: vec![ChallengeLabel::External("raf_scale")],
            },
            SumcheckShape {
                degree: 2,
                num_vars: SymbolicExpr::symbol(Symbol::LOG_K) + log_t(),
                weighting: PublicPolynomial::Eq,
                phases: vec![Phase { num_vars: log_t(), variable_group: VariableGroup::Cycle }],
            },
        );
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
    }

    // V_ram_output_check: RAM output check (zero-check)
    {
        let def = claims::ram::ram_output_check();
        let ordering_deps = deps_from_formula(&def, s1);
        let (vid, claims) = add_vertex_comp(
            b, def, ordering_deps, pt,
            InputSpec::Constant(0),
            SumcheckShape {
                degree: 2,
                num_vars: SymbolicExpr::symbol(Symbol::LOG_K) + log_t(),
                weighting: PublicPolynomial::Eq,
                phases: vec![Phase { num_vars: log_t(), variable_group: VariableGroup::Cycle }],
            },
        );
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
    }

    out
}
fn build_s3(b: &mut GraphBuilder, s1: &StageClaims, s2: &StageClaims) -> StageOutput {
    let pt = SymbolicPoint::Challenges(S3);
    let available = merge_claims(&[s1, s2]);
    let mut out = StageOutput::new();

    // V_shift: input = γ^0·next_unexp + γ·next_pc + γ²·next_virt + γ³·next_first + γ⁴·(1-next_noop)
    // Weighting: EqPlusOne (dual-point from S1 outer + S2 product, see jolt-core shift.rs)
    {
        let output_def = claims::spartan::shift();
        let ordering_deps = deps_from_formula(&output_def, &available);
        let input_def = {
            let eb = ExprBuilder::new();
            let next_unexp = eb.opening(0);
            let next_pc = eb.opening(1);
            let next_virt = eb.opening(2);
            let next_first = eb.opening(3);
            let next_noop = eb.opening(4);
            let g = eb.challenge(0);
            let g2 = g * g;
            let g3 = g2 * g;
            let g4 = g3 * g;
            ClaimDefinition {
                expr: eb.build(
                    next_unexp + g * next_pc + g2 * next_virt + g3 * next_first
                        + g4 - g4 * next_noop,
                ),
                opening_bindings: vec![
                    OpeningBinding { var_id: 0, polynomial: PolynomialId::NextUnexpandedPc },
                    OpeningBinding { var_id: 1, polynomial: PolynomialId::NextPc },
                    OpeningBinding { var_id: 2, polynomial: PolynomialId::NextIsVirtual },
                    OpeningBinding { var_id: 3, polynomial: PolynomialId::NextIsFirstInSequence },
                    OpeningBinding { var_id: 4, polynomial: PolynomialId::NextIsNoop },
                ],
                num_challenges: 1,
            }
        };
        let input_upstream = filter_claims(s1, &input_def);
        let (vid, claims) = add_vertex_comp(
            b,
            output_def,
            ordering_deps,
            pt.clone(),
            InputSpec::Formula {
                def: input_def,
                upstream: input_upstream,
                challenge_labels: vec![ChallengeLabel::PreSqueeze("shift_gamma")],
            },
            SumcheckShape {
                degree: 2,
                num_vars: log_t(),
                weighting: PublicPolynomial::EqPlusOne,
                phases: cycle_phase(),
            },
        );
        // Shift side-effect: ExpandedPc eval available at S3's point.
        // Used by BytecodeReadRaf RAF shift contribution.
        let (se_ids, se_claims) = alloc_side_effects(b, &pt, &[PolynomialId::ExpandedPc]);
        if let Vertex::Sumcheck(ref mut v) = b.vertices[vid.0 as usize] {
            v.side_effect_claims = se_ids;
        }
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
        out.claims.extend(se_claims);
    }

    // V_instr_input: input = right + γ·left from S2 InstructionInput claims
    {
        let output_def = claims::spartan::instruction_input();
        let ordering_deps = deps_from_formula(&output_def, &available);
        let input_def = {
            let eb = ExprBuilder::new();
            let right = eb.opening(0);
            let left = eb.opening(1);
            let gamma = eb.challenge(0);
            ClaimDefinition {
                expr: eb.build(right + gamma * left),
                opening_bindings: vec![
                    OpeningBinding { var_id: 0, polynomial: PolynomialId::RightInstructionInput },
                    OpeningBinding { var_id: 1, polynomial: PolynomialId::LeftInstructionInput },
                ],
                num_challenges: 1,
            }
        };
        let input_upstream = filter_claims(s2, &input_def);
        let (vid, claims) = add_vertex_comp(
            b,
            output_def,
            ordering_deps,
            pt.clone(),
            InputSpec::Formula {
                def: input_def,
                upstream: input_upstream,
                challenge_labels: vec![ChallengeLabel::PreSqueeze("instr_gamma")],
            },
            SumcheckShape {
                degree: 3,
                num_vars: log_t(),
                weighting: PublicPolynomial::Eq,
                phases: cycle_phase(),
            },
        );
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
    }

    // V_registers_cr: reduces RdWriteValue, Rs1Value, Rs2Value from S1
    {
        let def = claims::reductions::registers_claim_reduction();
        let upstream = filter_claims(s1, &def);
        let (vid, claims) = add_vertex_cr(
            b,
            def,
            &upstream,
            pt.clone(),
            2,
            log_t(),
            PublicPolynomial::Eq,
            "reg_gamma",
        );
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
    }

    out
}

fn deps_from_formula(def: &ClaimDefinition, stage: &StageClaims) -> Vec<ClaimId> {
    def.opening_bindings
        .iter()
        .filter_map(|b| stage.get(&b.polynomial).copied())
        .collect()
}

fn merge_claims(stages: &[&StageClaims]) -> StageClaims {
    let mut merged = StageClaims::new();
    for s in stages {
        merged.extend(s.iter().map(|(&k, &v)| (k, v)));
    }
    merged
}

fn filter_claims(stage: &StageClaims, def: &ClaimDefinition) -> StageClaims {
    let needed: std::collections::HashSet<PolynomialId> = def
        .opening_bindings
        .iter()
        .map(|b| b.polynomial)
        .collect();
    stage
        .iter()
        .filter(|(k, _)| needed.contains(k))
        .map(|(&k, &v)| (k, v))
        .collect()
}

fn build_s4(
    b: &mut GraphBuilder,
    config: &ProtocolConfig,
    s1: &StageClaims,
    s2: &StageClaims,
    s3: &StageClaims,
) -> StageOutput {
    let pt = SymbolicPoint::Challenges(S4);
    let available = merge_claims(&[s1, s2, s3]);
    let mut out = StageOutput::new();

    // V_registers_rw: input = rd_wv + γ·rs1 + γ²·rs2 from S3 RegistersCR
    // log_K + log_T rounds, phased (address then cycle)
    {
        let output_def = claims::registers::registers_read_write_checking();
        let input_def = {
            let eb = ExprBuilder::new();
            let rd_wv = eb.opening(0);
            let rs1 = eb.opening(1);
            let rs2 = eb.opening(2);
            let g = eb.challenge(0);
            ClaimDefinition {
                expr: eb.build(rd_wv + g * rs1 + g * g * rs2),
                opening_bindings: vec![
                    OpeningBinding { var_id: 0, polynomial: PolynomialId::RdWriteValue },
                    OpeningBinding { var_id: 1, polynomial: PolynomialId::Rs1Value },
                    OpeningBinding { var_id: 2, polynomial: PolynomialId::Rs2Value },
                ],
                num_challenges: 1,
            }
        };
        let input_upstream = filter_claims(s3, &input_def);
        let ordering_deps = deps_from_formula_map(&input_upstream);
        let (vid, claims) = add_vertex_comp(
            b,
            output_def,
            ordering_deps,
            pt.clone(),
            InputSpec::Formula {
                def: input_def,
                upstream: input_upstream,
                challenge_labels: vec![ChallengeLabel::PreSqueeze("reg_gamma")],
            },
            SumcheckShape {
                degree: 3,
                num_vars: log_ra(),
                weighting: PublicPolynomial::Eq,
                phases: vec![
                    Phase { num_vars: SymbolicExpr::symbol(Symbol::LOG_K), variable_group: VariableGroup::Address },
                    Phase { num_vars: log_t(), variable_group: VariableGroup::Cycle },
                ],
            },
        );
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
    }


    // V_ram_val_check
    {
        let output_def = claims::ram::ram_val_check();
        let input_def = claims::ram::ram_val_check_input(config.n_advice);
        let upstream = filter_claims(&available, &input_def);

        let vid = b.alloc_vertex();
        let mut produced = StageClaims::new();
        let mut produced_ids = Vec::new();
        for binding in &output_def.opening_bindings {
            let cid = b.alloc_claim(binding.polynomial, pt.clone());
            let _ = produced.insert(binding.polynomial, cid);
            produced_ids.push(cid);
        }

        let input_formula = if upstream.is_empty() {
            InputClaim::Constant(0)
        } else {
            // ram_val_check_input uses: challenge(0) = gamma (pre_squeeze),
            // challenge(1) = neg_init (external public input),
            // challenge(2+i) = advice selectors (external).
            let mut challenge_labels = vec![
                ChallengeLabel::PreSqueeze("ram_gamma"),
                ChallengeLabel::External("neg_init"),
            ];
            for i in 0..config.n_advice {
                challenge_labels.push(ChallengeLabel::External(if i == 0 {
                    "advice_sel_0"
                } else {
                    "advice_sel_1"
                }));
            }
            InputClaim::Formula {
                formula: bind_formula(input_def, &upstream),
                challenge_labels,
            }
        };
        let output_formula = bind_formula(output_def, &produced);

        let deps: Vec<ClaimId> = upstream.values().copied().collect();
        b.push_vertex(Vertex::Sumcheck(Box::new(SumcheckVertex {
            id: vid,
            deps,
            input: input_formula,
            produces: produced_ids,
            side_effect_claims: vec![],
            formula: output_formula,
            degree: 3,
            num_vars: log_t(),
            weighting: PublicPolynomial::Lt,
            phases: vec![Phase {
                num_vars: log_t(),
                variable_group: VariableGroup::Cycle,
            }],
            output_challenge_spec: OutputChallengeSpec::None,
        })));
        out.vertex_ids.push(vid);
        out.claims.extend(produced);
    }



    out
}

fn deps_from_formula_map(claims: &StageClaims) -> Vec<ClaimId> {
    claims.values().copied().collect()
}

fn build_s5(
    b: &mut GraphBuilder,
    config: &ProtocolConfig,
    s1: &StageClaims,
    s2: &StageClaims,
    s4: &StageClaims,
) -> StageOutput {
    let pt = SymbolicPoint::Challenges(S5);
    let available = merge_claims(&[s1, s2, s4]);
    let mut out = StageOutput::new();

    // V_instr_read_raf: input = rv + γ·left_op + γ²·right_op from S2 InstrLookupsCR
    if config.d_instr > 0 {
        let output_def = claims::instruction::instruction_ra_virtual(1, config.d_instr);
        let input_def = {
            let eb = ExprBuilder::new();
            let rv = eb.opening(0);
            let left_op = eb.opening(1);
            let right_op = eb.opening(2);
            let g = eb.challenge(0);
            ClaimDefinition {
                expr: eb.build(rv + g * left_op + g * g * right_op),
                opening_bindings: vec![
                    OpeningBinding { var_id: 0, polynomial: PolynomialId::LookupOutput },
                    OpeningBinding { var_id: 1, polynomial: PolynomialId::LeftLookupOperand },
                    OpeningBinding { var_id: 2, polynomial: PolynomialId::RightLookupOperand },
                ],
                num_challenges: 1,
            }
        };
        let input_upstream = filter_claims(s2, &input_def);
        let ordering_deps = deps_from_formula_map(&input_upstream);
        let (vid, claims) = add_vertex_comp(
            b, output_def, ordering_deps, pt.clone(),
            InputSpec::Formula {
                def: input_def,
                upstream: input_upstream,
                challenge_labels: vec![ChallengeLabel::PreSqueeze("instr_raf_gamma")],
            },
            SumcheckShape {
                degree: config.d_instr + 2,
                num_vars: log_ra(),
                weighting: PublicPolynomial::Eq,
                phases: vec![
                    Phase { num_vars: SymbolicExpr::symbol(Symbol::LOG_K), variable_group: VariableGroup::Address },
                    Phase { num_vars: log_t(), variable_group: VariableGroup::Cycle },
                ],
            },
        );
        // InstrReadRaf side-effects: InstructionRafFlag + LookupTableFlag(0..N) evals.
        // Used by BytecodeReadRaf Stage 5 contribution.
        let mut se_polys = vec![PolynomialId::InstructionRafFlag];
        for i in 0..config.n_lookup_tables {
            se_polys.push(PolynomialId::LookupTableFlag(i));
        }
        let (se_ids, se_claims) = alloc_side_effects(b, &pt, &se_polys);
        if let Vertex::Sumcheck(ref mut v) = b.vertices[vid.0 as usize] {
            v.side_effect_claims = se_ids;
        }
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
        out.claims.extend(se_claims);
    }

    // V_ram_ra_cr: RAM RA claim reduction (moved from S7 to match jolt-core S5)
    {
        let def = claims::reductions::ram_ra_claim_reduction();
        let upstream = filter_claims(&available, &def);
        if !upstream.is_empty() {
            let (vid, claims) = add_vertex_cr(
                b, def, &upstream, pt.clone(), 2, log_t(),
                PublicPolynomial::Eq, "ram_ra_gamma",
            );
            out.vertex_ids.push(vid);
            out.claims.extend(claims);
        }
    }

    // V_registers_val_eval: input = RegistersVal eval from S4 RegistersRW (single claim passthrough)
    {
        let output_def = claims::registers::registers_val_evaluation();
        let input_def = {
            let eb = ExprBuilder::new();
            let val = eb.opening(0);
            ClaimDefinition {
                expr: eb.build(val),
                opening_bindings: vec![OpeningBinding {
                    var_id: 0,
                    polynomial: PolynomialId::RegistersVal,
                }],
                num_challenges: 0,
            }
        };
        let input_upstream = filter_claims(s4, &input_def);
        let ordering_deps = deps_from_formula_map(&input_upstream);
        let (vid, claims) = add_vertex_comp(
            b, output_def, ordering_deps, pt,
            InputSpec::Formula {
                def: input_def,
                upstream: input_upstream,
                challenge_labels: vec![],
            },
            SumcheckShape {
                degree: 3, num_vars: log_t(),
                weighting: PublicPolynomial::Lt,
                phases: cycle_phase(),
            },
        );
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
    }

    out
}

/// Maps a CircuitFlags index (0..14) to the canonical PolynomialId.
/// Indices 5,6,7,12 map to existing named variants; the rest use OpFlag.
fn circuit_flag_poly(index: usize) -> PolynomialId {
    match index {
        5 => PolynomialId::JumpFlag,
        6 => PolynomialId::WriteLookupToRdFlag,
        7 => PolynomialId::NextIsVirtual,       // VirtualInstruction = same poly
        12 => PolynomialId::NextIsFirstInSequence, // IsFirstInSequence = same poly
        other => PolynomialId::OpFlag(other),
    }
}

/// Maps an InstructionFlags index (0..6) to the canonical PolynomialId.
#[allow(dead_code)]
fn instr_flag_poly(index: usize) -> PolynomialId {
    match index {
        0 => PolynomialId::LeftIsPc,
        1 => PolynomialId::RightIsImm,
        2 => PolynomialId::LeftIsRs1,
        3 => PolynomialId::RightIsRs2,
        4 => PolynomialId::BranchFlag,
        5 => PolynomialId::NextIsNoop, // IsNoop = same poly as NextIsNoop
        _ => unreachable!(),
    }
}

/// Build the BytecodeReadRaf multi-stage RLC input claim.
///
/// The formula is a two-level random linear combination:
/// ```text
/// input = Σ_{s=0}^{6} γ^s · rv_claim_s + γ^7
/// ```
/// where each `rv_claim_s = Σ_j β_s^j · poly_j(r_stage_s)`.
///
/// The compound challenge for term (s, j) is `γ^s · β_s^j`.
/// All challenge values are Derived — the verifier computes them from squeezed gammas.
fn build_bytecode_raf_input(
    config: &ProtocolConfig,
    s1: &StageClaims,
    s2: &StageClaims,
    s3: &StageClaims,
    s4: &StageClaims,
    s5: &StageClaims,
) -> InputClaim {
    let eb = ExprBuilder::new();
    let mut var_id = 0u32;
    let mut challenge_id = 0u32;
    let mut opening_claims: HashMap<u32, ClaimId> = HashMap::new();
    let mut opening_bindings = Vec::new();
    let mut terms = eb.zero();

    /// Collect per-stage (poly_id, stage_claims) pairs into the formula.
    struct TermCollector {
        polys: Vec<(PolynomialId, ClaimId)>,
    }
    impl TermCollector {
        fn new() -> Self { Self { polys: Vec::new() } }
        fn add(&mut self, poly_id: PolynomialId, stage: &StageClaims) {
            if let Some(&cid) = stage.get(&poly_id) {
                self.polys.push((poly_id, cid));
            }
        }
    }

    let mut tc = TermCollector::new();

    // Stage 1 (SpartanOuter): UnexpandedPC + Imm + 14 CircuitFlags = 16 terms
    tc.add(PolynomialId::UnexpandedPc, s1);
    tc.add(PolynomialId::Imm, s1);
    for i in 0..config.n_circuit_flags {
        tc.add(circuit_flag_poly(i), s1);
    }

    // Stage 2 (ProductVirtual): Jump, Branch, WriteLookupToRD, VirtualInstruction = 4 terms
    tc.add(PolynomialId::JumpFlag, s2);
    tc.add(PolynomialId::BranchFlag, s2);
    tc.add(PolynomialId::WriteLookupToRdFlag, s2);
    tc.add(PolynomialId::NextIsVirtual, s2);

    // Stage 3 (InstrInput + Shift): 9 terms
    tc.add(PolynomialId::Imm, s3);
    tc.add(PolynomialId::UnexpandedPc, s3);
    tc.add(PolynomialId::LeftIsRs1, s3);
    tc.add(PolynomialId::LeftIsPc, s3);
    tc.add(PolynomialId::RightIsRs2, s3);
    tc.add(PolynomialId::RightIsImm, s3);
    tc.add(PolynomialId::NextIsNoop, s3);
    tc.add(PolynomialId::NextIsVirtual, s3);
    tc.add(PolynomialId::NextIsFirstInSequence, s3);

    // Stage 4 (RegistersRW): RdWa, Rs1Ra, Rs2Ra = 3 terms
    tc.add(PolynomialId::RdWa, s4);
    tc.add(PolynomialId::Rs1Ra, s4);
    tc.add(PolynomialId::Rs2Ra, s4);

    // Stage 5 (RegistersValEval + InstrReadRaf): RdWa + InstrRafFlag + LookupTableFlags
    tc.add(PolynomialId::RdWa, s5);
    tc.add(PolynomialId::InstructionRafFlag, s5);
    for i in 0..config.n_lookup_tables {
        tc.add(PolynomialId::LookupTableFlag(i), s5);
    }

    // RAF contributions: ExpandedPc from S1 (SpartanOuter) and S3 (SpartanShift)
    tc.add(PolynomialId::ExpandedPc, s1);
    tc.add(PolynomialId::ExpandedPc, s3);

    // Build formula: Σ challenge(i) * opening(i)
    for (poly_id, claim_id) in &tc.polys {
        let o = eb.opening(var_id);
        let c = eb.challenge(challenge_id);
        terms = terms + c * o;
        let _ = opening_claims.insert(var_id, *claim_id);
        opening_bindings.push(OpeningBinding {
            var_id,
            polynomial: *poly_id,
        });
        var_id += 1;
        challenge_id += 1;
    }

    // Entry constant term (γ^7, no opening)
    let entry_c = eb.challenge(challenge_id);
    terms = terms + entry_c;

    let expr = eb.build(terms);

    let definition = ClaimDefinition {
        expr,
        opening_bindings,
        num_challenges: challenge_id + 1,
    };

    let challenge_labels = (0..=challenge_id)
        .map(|_| ChallengeLabel::PreSqueeze("bc_raf_gamma"))
        .collect();

    InputClaim::Formula {
        formula: ClaimFormula {
            definition,
            opening_claims,
        },
        challenge_labels,
    }
}

fn build_s6(
    b: &mut GraphBuilder,
    config: &ProtocolConfig,
    s1: &StageClaims,
    s2: &StageClaims,
    s3: &StageClaims,
    s4: &StageClaims,
    s5: &StageClaims,
) -> StageOutput {
    let pt = SymbolicPoint::Challenges(S6);
    let available = merge_claims(&[s2, s4, s5]);
    let mut out = StageOutput::new();

    // Instance order matches jolt-core S6 exactly:
    // [BytecodeReadRaf, Booleanity, HammingBooleanity, RamRaVirtual, InstrRaVirtual, IncCR]
    // The α^j batching coefficients depend on this ordering.

    // (1) V_bytecode_read_raf: bytecode read + RAF checking
    // Input: two-level RLC of ~78 upstream polynomial evaluations from S1-S5.
    // input_claim = Σ_{s=0}^{6} γ^s · rv_claim_s + γ^7 (entry constant)
    // where each rv_claim_s = Σ_j β_s^j · poly_j(r_stage_s)
    if config.d_bc > 0 {
        let output_def = claims::bytecode::bytecode_read_raf(5);
        let vid = b.alloc_vertex();
        let mut produced = StageClaims::new();
        let mut produced_ids = Vec::new();
        for binding in &output_def.opening_bindings {
            let cid = b.alloc_claim(binding.polynomial, pt.clone());
            let _ = produced.insert(binding.polynomial, cid);
            produced_ids.push(cid);
        }
        let output_formula = bind_formula(output_def, &produced);

        // Build input claim: collect claims from specific stages.
        let input = build_bytecode_raf_input(config, s1, s2, s3, s4, s5);

        let deps: Vec<ClaimId> = match &input {
            InputClaim::Formula { formula, .. } => {
                formula.opening_claims.values().copied().collect()
            }
            InputClaim::Constant(_) => vec![],
        };

        b.push_vertex(Vertex::Sumcheck(Box::new(SumcheckVertex {
            id: vid,
            deps,
            input,
            produces: produced_ids,
            side_effect_claims: vec![],
            formula: output_formula,
            degree: config.d_bc + 1,
            num_vars: log_ra(),
            weighting: PublicPolynomial::Eq,
            phases: vec![
                Phase { num_vars: SymbolicExpr::symbol(Symbol::LOG_K), variable_group: VariableGroup::Address },
                Phase { num_vars: log_t(), variable_group: VariableGroup::Cycle },
            ],
            output_challenge_spec: OutputChallengeSpec::None,
        })));
        out.vertex_ids.push(vid);
        out.claims.extend(produced);
    }

    // (2) V_ra_bool: proves all RA polynomials are boolean (one-hot)
    if config.d_total() > 0 {
        let ra_ids = config.all_ra_poly_ids();
        let def = claims::booleanity::ra_booleanity(ra_ids.len(), &ra_ids);
        let (vid, claims) = add_vertex_comp(
            b,
            def,
            vec![],
            pt.clone(),
            InputSpec::Constant(0),
            SumcheckShape {
                degree: 3,
                num_vars: log_ra(),
                weighting: PublicPolynomial::Eq,
                phases: vec![
                    Phase { num_vars: SymbolicExpr::symbol(Symbol::LOG_K), variable_group: VariableGroup::Address },
                    Phase { num_vars: log_t(), variable_group: VariableGroup::Cycle },
                ],
            },
        );
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
    }

    // (3) V_hamming_bool: proves HammingWeight is boolean
    {
        let def = claims::ram::hamming_booleanity();
        let (vid, claims) = add_vertex_comp(
            b,
            def,
            vec![],
            pt.clone(),
            InputSpec::Constant(0),
            SumcheckShape {
                degree: 3,
                num_vars: log_t(),
                weighting: PublicPolynomial::Eq,
                phases: vec![Phase {
                    num_vars: log_t(),
                    variable_group: VariableGroup::Cycle,
                }],
            },
        );
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
    }

    // (4) V_ram_ra_virtual: input = RamRa eval from S5 RamRaCR (direct passthrough)
    if config.d_ram > 0 {
        let output_def = claims::ram::ram_ra_virtual(config.d_ram);
        let input_def = {
            let eb = ExprBuilder::new();
            let ra = eb.opening(0);
            ClaimDefinition {
                expr: eb.build(ra),
                opening_bindings: vec![OpeningBinding {
                    var_id: 0,
                    polynomial: PolynomialId::RamAddress,
                }],
                num_challenges: 0,
            }
        };
        let input_upstream = filter_claims(s5, &input_def);
        let ordering_deps = deps_from_formula_map(&input_upstream);
        let (vid, claims) = add_vertex_comp(
            b, output_def, ordering_deps, pt.clone(),
            InputSpec::Formula {
                def: input_def,
                upstream: input_upstream,
                challenge_labels: vec![],
            },
            SumcheckShape {
                degree: config.d_ram + 1,
                num_vars: log_ra(),
                weighting: PublicPolynomial::Eq,
                phases: vec![
                    Phase { num_vars: SymbolicExpr::symbol(Symbol::LOG_K), variable_group: VariableGroup::Address },
                    Phase { num_vars: log_t(), variable_group: VariableGroup::Cycle },
                ],
            },
        );
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
    }

    // (5) V_instr_ra_virtual: instruction RA virtual sumcheck (Toom-Cook product)
    // Each virtual RA poly is a product of `chunks_per_virtual` committed chunks.
    // Degree = chunks_per_virtual + 1 (product of committed chunks × eq).
    // Input = Σ γ^i · InstructionRa_virtual(i) from S5 InstrReadRaf.
    if config.d_instr > 0 {
        let n_virtual = config.n_virtual_instr_ra();
        let cpv = config.d_instr_chunks_per_virtual;
        let output_def = claims::instruction::instruction_ra_virtual(n_virtual, cpv);

        // Input: Σ γ^i · virtual_ra_i from S5 InstrReadRaf
        // Each virtual_ra_i is a product of `cpv` committed chunks,
        // but the input claim references the VIRTUAL evaluation, not individual chunks.
        // The S5 InstrReadRaf produces virtual RA claims keyed by InstructionRa(i*cpv..i*cpv+cpv-1).
        // For the input formula we need the S5 claims for all committed RA chunks.
        let input_def = {
            let eb = ExprBuilder::new();
            let g = eb.challenge(0);
            let mut sum = eb.zero();
            for i in 0..n_virtual {
                let virtual_ra_i = eb.opening(i as u32);
                if i == 0 {
                    sum = virtual_ra_i;
                } else {
                    let mut gp = g;
                    for _ in 1..i {
                        gp = gp * g;
                    }
                    sum = sum + gp * virtual_ra_i;
                }
            }
            // Opening bindings use first committed chunk per virtual poly
            // as a proxy for the virtual evaluation. The verifier resolves these from
            // the S5 InstrReadRaf output claims.
            let opening_bindings = (0..n_virtual)
                .map(|i| OpeningBinding {
                    var_id: i as u32,
                    polynomial: PolynomialId::InstructionRa(i * cpv),
                })
                .collect();
            ClaimDefinition {
                expr: eb.build(sum),
                opening_bindings,
                num_challenges: u32::from(n_virtual > 1),
            }
        };
        let input_upstream = filter_claims(s5, &input_def);
        let ordering_deps = deps_from_formula_map(&input_upstream);
        let (vid, claims) = add_vertex_comp(
            b, output_def, ordering_deps, pt.clone(),
            InputSpec::Formula {
                def: input_def,
                upstream: input_upstream,
                challenge_labels: if n_virtual > 1 {
                    vec![ChallengeLabel::PreSqueeze("instr_ra_gamma")]
                } else {
                    vec![]
                },
            },
            SumcheckShape {
                degree: cpv + 1,
                num_vars: log_ra(),
                weighting: PublicPolynomial::Eq,
                phases: vec![
                    Phase { num_vars: SymbolicExpr::symbol(Symbol::LOG_K), variable_group: VariableGroup::Address },
                    Phase { num_vars: log_t(), variable_group: VariableGroup::Cycle },
                ],
            },
        );
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
    }

    // (6) V_inc_cr: reduces RamInc + RdInc claims from S2/S4/S5
    // Weighting is Derived: combines eq(ρ, r_cycle_s2) + γ·eq(ρ, r_cycle_s4) for RamInc
    // and γ²·(eq(ρ, s_cycle_s4) + γ·eq(ρ, s_cycle_s5)) for RdInc.
    {
        let def = claims::reductions::increment_claim_reduction();
        let upstream = filter_claims(&available, &def);
        let (vid, claims) = add_vertex_cr(
            b,
            def,
            &upstream,
            pt,
            2,
            log_t(),
            PublicPolynomial::Derived,
            "inc_gamma",
        );
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
    }

    out
}

fn build_s7(
    b: &mut GraphBuilder,
    config: &ProtocolConfig,
    s5: &StageClaims,
    s6: &StageClaims,
) -> StageOutput {
    let pt = unified_point();
    let available = merge_claims(&[s5, s6]);
    let mut out = StageOutput::new();

    // V_hw_cr: hamming weight claim reduction over all RA polynomials
    if config.d_total() > 0 {
        let ra_ids = config.all_ra_poly_ids();
        let def = claims::reductions::hamming_weight_claim_reduction(&ra_ids);
        let upstream = filter_claims(&available, &def);
        let (vid, claims) = add_vertex_cr(
            b,
            def,
            &upstream,
            pt.clone(),
            2,
            SymbolicExpr::symbol(Symbol::LOG_K),
            PublicPolynomial::Eq,
            "hw_gamma",
        );
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
    }

    // V_advice_cr: advice claim reduction (when advice polys exist)
    if config.n_advice > 0 {
        let def = claims::reductions::advice_claim_reduction_address();
        let upstream = filter_claims(&available, &def);
        if !upstream.is_empty() {
            let (vid, claims) = add_vertex_cr(
                b,
                def,
                &upstream,
                pt.clone(),
                2,
                SymbolicExpr::symbol(Symbol::LOG_K),
                PublicPolynomial::Eq,
                "advice_gamma",
            );
            out.vertex_ids.push(vid);
            out.claims.extend(claims);
        }
    }

    out
}

fn build_normalization(b: &mut GraphBuilder, config: &ProtocolConfig, s6: &StageClaims) -> StageOutput {
    let vid = b.alloc_vertex();
    let target = unified_point();
    let padding = SymbolicPoint::Challenges(S7);

    // Normalize dense committed polys from r_cycle (S6) to unified point.
    // RamInc, RdInc always need normalization. Advice polys too when present.
    let mut dense_polys = vec![PolynomialId::RamInc, PolynomialId::RdInc];
    if config.n_advice >= 1 {
        dense_polys.push(PolynomialId::TrustedAdvice);
    }
    if config.n_advice >= 2 {
        dense_polys.push(PolynomialId::UntrustedAdvice);
    }

    let mut consumes = Vec::new();
    let mut produces = Vec::new();
    let mut claims = StageClaims::new();

    for &poly_id in &dense_polys {
        if let Some(&s6_claim) = s6.get(&poly_id) {
            consumes.push(s6_claim);
            let cid = b.alloc_claim(poly_id, target.clone());
            produces.push(cid);
            let _ = claims.insert(poly_id, cid);
        }
    }

    if !consumes.is_empty() {
        b.push_vertex(Vertex::PointNormalization(PointNormalizationVertex {
            id: vid,
            consumes,
            produces,
            padding_source: padding,
        }));

        StageOutput {
            vertex_ids: vec![vid],
            claims,
        }
    } else {
        StageOutput::new()
    }
}

fn build_opening(
    b: &mut GraphBuilder,
    s7: &StageClaims,
    norm: &StageClaims,
) -> (Vec<VertexId>, Vec<ClaimId>) {
    // All committed polynomial claims converge to the unified point:
    // - RA polys from S7 HammingWeightCR
    // - Dense polys (RamInc, RdInc) from PointNormalization
    // SpartanWitness is NOT here — it's verified internally by Spartan.
    let available = merge_claims(&[norm, s7]);
    let mut vids = Vec::new();
    let mut consumed = Vec::new();

    let committed_polys: Vec<PolynomialId> = available
        .keys()
        .filter(|p| p.is_committed())
        .copied()
        .collect();

    for poly_id in committed_polys {
        if let Some(&claim_id) = available.get(&poly_id) {
            let vid = b.alloc_vertex();
            b.push_vertex(Vertex::Opening(OpeningVertex {
                id: vid,
                consumes: claim_id,
            }));
            vids.push(vid);
            consumed.push(claim_id);
        }
    }

    (vids, consumed)
}

fn build_staging(
    stages: &[(StageId, &StageOutput, NumVars, Vec<ChallengeSpec>)],
    normalization_vids: Vec<VertexId>,
    opening_vids: Vec<VertexId>,
) -> Staging {
    let sumcheck_stages: Vec<Stage> = stages
        .iter()
        .map(|(id, output, nv, squeezes)| Stage {
            id: *id,
            vertices: output.vertex_ids.clone(),
            challenge_point: ChallengePoint {
                num_vars: nv.clone(),
            },
            batching: vec![BatchGroup {
                vertices: output.vertex_ids.clone(),
            }],
            pre_squeeze: squeezes.clone(),
        })
        .collect();

    let mut all_opening_vids = normalization_vids;
    all_opening_vids.extend(opening_vids);

    let opening = OpeningStage {
        vertices: all_opening_vids,
        point: unified_point(),
        reduction: ReductionStrategy::Rlc,
        opening_groups: vec![OpeningGroup {
            vertices: vec![],
            source_groups: vec![],
        }],
    };

    Staging {
        stages: sumcheck_stages,
        opening,
    }
}

fn build_commitment_strategy(b: &GraphBuilder) -> CommitmentStrategy {
    let mut groups_map: HashMap<CommitmentGroupId, Vec<PolynomialId>> = HashMap::new();
    for poly in &b.polynomials {
        if let PolynomialKind::Committed { group } = &poly.kind {
            groups_map.entry(*group).or_default().push(poly.id);
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

/// Build the complete Jolt protocol graph from configuration.
pub fn build_jolt_protocol(config: ProtocolConfig) -> ProtocolGraph {
    let mut b = GraphBuilder::new();
    register_all_polynomials(&mut b, &config);

    // Build stages in dependency order
    let s1_out = build_spartan(&mut b, &config);
    let s2_out = build_s2(&mut b, &config, &s1_out.claims);
    let s3_out = build_s3(&mut b, &s1_out.claims, &s2_out.claims);
    let s4_out = build_s4(&mut b, &config, &s1_out.claims, &s2_out.claims, &s3_out.claims);
    let s5_out = build_s5(&mut b, &config, &s1_out.claims, &s2_out.claims, &s4_out.claims);
    let s6_out = build_s6(
        &mut b,
        &config,
        &s1_out.claims,
        &s2_out.claims,
        &s3_out.claims,
        &s4_out.claims,
        &s5_out.claims,
    );
    let s7_out = build_s7(&mut b, &config, &s5_out.claims, &s6_out.claims);
    let norm_out = build_normalization(&mut b, &config, &s6_out.claims);
    let (opening_vids, _consumed) =
        build_opening(&mut b, &s7_out.claims, &norm_out.claims);

    let log_t = SymbolicExpr::symbol(Symbol::LOG_T);
    let log_k = SymbolicExpr::symbol(Symbol::LOG_K);
    let d_total = SymbolicExpr::concrete(config.d_total());

    // Challenge squeeze specs per stage — must match stages.rs and verify.rs exactly.
    let s1_sq: Vec<ChallengeSpec> = vec![]; // Spartan: opaque internal transcript
    let s2_sq = vec![
        ChallengeSpec::Scalar { label: "pv_tau_high" },  // PV uni-skip
        ChallengeSpec::Scalar { label: "ram_rw_gamma" },  // RamRW
        ChallengeSpec::Scalar { label: "instr_cr_gamma" }, // InstrLookupsCR
        ChallengeSpec::Vector { label: "output_r_address", dim: log_k.clone() }, // OutputCheck
    ];
    let s3_sq = vec![
        ChallengeSpec::GammaPowers {
            label: "shift_gamma",
            count: SymbolicExpr::concrete(5),
        },
        ChallengeSpec::Scalar {
            label: "instr_gamma",
        },
        ChallengeSpec::Scalar { label: "reg_gamma" },
    ];
    let s4_sq = vec![
        ChallengeSpec::Scalar { label: "reg_gamma" },
        ChallengeSpec::Scalar { label: "ram_gamma" },
    ];
    // S5 squeeze order: InstructionReadRaf(gamma), RamRaCR(gamma).
    // RegistersValEval squeezes nothing (its point comes from S4 opening accumulator).
    let s5_sq = vec![
        ChallengeSpec::Scalar { label: "instr_raf_gamma" },
        ChallengeSpec::Scalar { label: "ram_ra_gamma" },
    ];
    // S6 squeeze order matches jolt-core params construction order.
    // BytecodeReadRaf::gen() squeezes 6 scalars (1 global + 5 per-stage gammas).
    // Then: Booleanity(gamma), InstructionRaVirtual(gamma), IncCR(gamma).
    // HammingBooleanity and RamRaVirtual squeeze nothing.
    let n_stage1_terms = SymbolicExpr::concrete(2 + config.n_circuit_flags);
    let n_stage5_terms = SymbolicExpr::concrete(2 + config.n_lookup_tables);
    let s6_sq = vec![
        ChallengeSpec::GammaPowers { label: "bc_raf_gamma", count: SymbolicExpr::concrete(8) },
        ChallengeSpec::GammaPowers { label: "bc_raf_stage1_gamma", count: n_stage1_terms },
        ChallengeSpec::GammaPowers { label: "bc_raf_stage2_gamma", count: SymbolicExpr::concrete(4) },
        ChallengeSpec::GammaPowers { label: "bc_raf_stage3_gamma", count: SymbolicExpr::concrete(9) },
        ChallengeSpec::GammaPowers { label: "bc_raf_stage4_gamma", count: SymbolicExpr::concrete(3) },
        ChallengeSpec::GammaPowers { label: "bc_raf_stage5_gamma", count: n_stage5_terms },
        ChallengeSpec::Scalar { label: "bool_gamma" },
        ChallengeSpec::Scalar { label: "instr_ra_gamma" },
        ChallengeSpec::Scalar { label: "inc_gamma" },
    ];
    let mut s7_sq = vec![ChallengeSpec::GammaPowers {
        label: "hw_gamma",
        count: d_total,
    }];
    if config.n_advice > 0 {
        s7_sq.push(ChallengeSpec::Scalar { label: "advice_gamma" });
    }

    // Stage challenge_point = max(instance num_vars) within the stage.
    // S2/S4/S5/S6 all contain instances with log_K + log_T rounds (RamRW, RegistersRW,
    // InstructionReadRaf, BytecodeReadRaf respectively), so their challenge points are log_K + log_T.
    let log_ra = log_k.clone() + log_t.clone();
    let staging = build_staging(
        &[
            (
                S1,
                &s1_out,
                SymbolicExpr::symbol(Symbol::LOG_ROWS) + SymbolicExpr::symbol(Symbol::LOG_COLS),
                s1_sq,
            ),
            (S2, &s2_out, log_ra.clone(), s2_sq),
            (S3, &s3_out, log_t.clone(), s3_sq),
            (S4, &s4_out, log_ra.clone(), s4_sq),
            (S5, &s5_out, log_ra.clone(), s5_sq),
            (S6, &s6_out, log_ra, s6_sq),
            (S7, &s7_out, log_k, s7_sq),
        ],
        norm_out.vertex_ids,
        opening_vids,
    );

    let commitment = build_commitment_strategy(&b);

    ProtocolGraph {
        claim_graph: ClaimGraph {
            polynomials: b.polynomials,
            claims: b.claims,
            vertices: b.vertices,
        },
        staging,
        commitment,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn default_config() -> ProtocolConfig {
        ProtocolConfig {
            d_instr: 8,
            d_bc: 4,
            d_ram: 3,
            d_instr_chunks_per_virtual: 2,
            n_lookup_tables: 41,
            n_circuit_flags: 14,
            n_advice: 0,
        }
    }

    #[test]
    fn builds_without_panic() {
        let graph = build_jolt_protocol(default_config());
        assert!(!graph.claim_graph.vertices.is_empty());
        assert!(!graph.claim_graph.claims.is_empty());
        assert!(!graph.claim_graph.polynomials.is_empty());
    }

    #[test]
    fn all_committed_polys_have_opening_vertices() {
        let config = default_config();
        let graph = build_jolt_protocol(config.clone());

        let committed_ids: HashSet<PolynomialId> = graph
            .claim_graph
            .polynomials
            .iter()
            .filter(|p| matches!(p.kind, PolynomialKind::Committed { .. }))
            .filter(|p| p.id != PolynomialId::SpartanWitness) // opened by Spartan internally
            .map(|p| p.id)
            .collect();

        let opened_ids: HashSet<PolynomialId> = graph
            .claim_graph
            .vertices
            .iter()
            .filter_map(|v| match v {
                Vertex::Opening(o) => Some(graph.claim_graph.claim(o.consumes).polynomial),
                _ => None,
            })
            .collect();

        // Every committed polynomial should have an opening vertex
        for pid in &committed_ids {
            assert!(
                opened_ids.contains(pid),
                "committed polynomial {pid:?} has no opening vertex"
            );
        }
    }

    #[test]
    fn vertex_ids_are_dense() {
        let graph = build_jolt_protocol(default_config());
        let max_id = graph
            .claim_graph
            .vertices
            .iter()
            .map(|v| v.id().0)
            .max()
            .unwrap();
        assert_eq!(max_id as usize, graph.claim_graph.vertices.len() - 1);
    }

    #[test]
    fn staging_covers_all_vertices() {
        let graph = build_jolt_protocol(default_config());

        let mut staged: HashSet<VertexId> = HashSet::new();
        for stage in &graph.staging.stages {
            for &vid in &stage.vertices {
                let _ = staged.insert(vid);
            }
        }
        for &vid in &graph.staging.opening.vertices {
            let _ = staged.insert(vid);
        }

        for v in &graph.claim_graph.vertices {
            assert!(
                staged.contains(&v.id()),
                "vertex {:?} not in any stage",
                v.id()
            );
        }
    }

    #[test]
    fn no_duplicate_produced_claims() {
        let graph = build_jolt_protocol(default_config());
        let mut producers: HashMap<ClaimId, VertexId> = HashMap::new();
        for v in &graph.claim_graph.vertices {
            for cid in v.all_produced_claims() {
                let prev = producers.insert(cid, v.id());
                assert!(
                    prev.is_none(),
                    "claim {cid:?} produced by both {:?} and {:?}",
                    prev.unwrap(),
                    v.id()
                );
            }
        }
    }

    #[test]
    fn commitment_groups_are_consistent() {
        let graph = build_jolt_protocol(default_config());
        let errors = graph.validate_commitment();
        assert!(errors.is_empty(), "commitment errors: {errors:?}");
    }

    #[test]
    fn challenge_specs_populated() {
        let graph = build_jolt_protocol(default_config());
        let stages = &graph.staging.stages;

        // S1 (Spartan): opaque, no pre_squeeze
        assert!(
            stages[0].pre_squeeze.is_empty(),
            "S1 should have no pre_squeeze"
        );

        // S2: 4 specs (PV tau_high + RamRW gamma + InstrCR gamma + OutputCheck r_address)
        assert_eq!(stages[1].pre_squeeze.len(), 4);

        // S3: 3 specs (GammaPowers + 2 Scalar)
        assert_eq!(stages[2].pre_squeeze.len(), 3);
        assert!(matches!(
            stages[2].pre_squeeze[0],
            ChallengeSpec::GammaPowers {
                label: "shift_gamma",
                ..
            }
        ));
        assert!(matches!(
            stages[2].pre_squeeze[1],
            ChallengeSpec::Scalar {
                label: "instr_gamma"
            }
        ));
        assert!(matches!(
            stages[2].pre_squeeze[2],
            ChallengeSpec::Scalar { label: "reg_gamma" }
        ));

        // S4: 2 Scalar
        assert_eq!(stages[3].pre_squeeze.len(), 2);

        // S5: 2 Scalar (InstrReadRaf gamma + RamRaCR gamma; RegistersValEval squeezes nothing)
        assert_eq!(stages[4].pre_squeeze.len(), 2);

        // S6: 6 GammaPowers (BytecodeReadRaf: 1 global + 5 per-stage) + 3 Scalar = 9
        assert_eq!(stages[5].pre_squeeze.len(), 9);
        assert!(matches!(
            stages[5].pre_squeeze[0],
            ChallengeSpec::GammaPowers { label: "bc_raf_gamma", .. }
        ));
        assert!(matches!(
            stages[5].pre_squeeze[8],
            ChallengeSpec::Scalar { label: "inc_gamma" }
        ));

        // S7: 1 GammaPowers
        assert_eq!(stages[6].pre_squeeze.len(), 1);
        assert!(matches!(
            stages[6].pre_squeeze[0],
            ChallengeSpec::GammaPowers {
                label: "hw_gamma",
                ..
            }
        ));
    }

    #[test]
    fn bytecode_raf_input_claim_term_count() {
        let config = default_config();
        let graph = build_jolt_protocol(config.clone());

        let s6 = &graph.staging.stages[5];
        let bc_vid = s6.vertices[0];
        let bc_vertex = graph.claim_graph.vertex(bc_vid);
        if let Vertex::Sumcheck(v) = bc_vertex {
            if let InputClaim::Formula { formula, challenge_labels } = &v.input {
                let n_openings = formula.opening_claims.len();
                // 16 (S1) + 4 (S2) + 9 (S3) + 3 (S4) + 43 (S5) + 2 (RAF) = 77
                let expected_s1 = 2 + config.n_circuit_flags;
                let expected_s2 = 4;
                let expected_s3 = 9;
                let expected_s4 = 3;
                let expected_s5 = 2 + config.n_lookup_tables;
                let expected_raf = 2;
                let expected_total = expected_s1 + expected_s2 + expected_s3
                    + expected_s4 + expected_s5 + expected_raf;
                assert_eq!(
                    n_openings, expected_total,
                    "BytecodeReadRaf input: expected {expected_total} openings, got {n_openings}"
                );
                assert_eq!(
                    challenge_labels.len(), expected_total + 1,
                    "expected {} challenge labels (openings + entry constant), got {}",
                    expected_total + 1, challenge_labels.len()
                );
            } else {
                panic!("BytecodeReadRaf should have Formula input");
            }
        } else {
            panic!("BytecodeReadRaf should be a Sumcheck vertex");
        }
    }
}
