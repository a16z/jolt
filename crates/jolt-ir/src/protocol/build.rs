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
use crate::zkvm::tags::poly as ptag;
use crate::{ExprBuilder, PolynomialId};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Structural parameters that determine the graph's shape.
///
/// These are known at preprocessing time. Symbolic quantities (`log_T`, `log_k`)
/// stay symbolic; only the chunk counts (which determine vertex/claim COUNT) are
/// concrete.
#[derive(Clone, Debug)]
pub struct ProtocolConfig {
    /// Instruction RA chunks (e.g., 8).
    pub d_instr: usize,
    /// Bytecode RA chunks (e.g., 4).
    pub d_bc: usize,
    /// RAM RA chunks (e.g., 3).
    pub d_ram: usize,
    /// Number of advice polynomials (0 = no advice).
    pub n_advice: usize,
}

impl ProtocolConfig {
    pub fn d_total(&self) -> usize {
        self.d_instr + self.d_bc + self.d_ram
    }

    /// All committed RA polynomial IDs, in order: instruction, bytecode, RAM.
    #[allow(dead_code)]
    fn all_ra_poly_ids(&self) -> Vec<PolynomialId> {
        (0..self.d_instr)
            .map(PolynomialId::InstructionRa)
            .chain((0..self.d_bc).map(PolynomialId::BytecodeRa))
            .chain((0..self.d_ram).map(PolynomialId::RamRa))
            .collect()
    }

    /// All committed RA polynomial tags (old u64 system), matching `all_ra_poly_ids` order.
    fn all_ra_tags(&self) -> Vec<u64> {
        (0..self.d_instr)
            .map(ptag::instruction_ra)
            .chain((0..self.d_bc).map(ptag::bytecode_ra))
            .chain((0..self.d_ram).map(ptag::ram_ra_committed))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Stage IDs (fixed layout)
// ---------------------------------------------------------------------------

const S1: StageId = StageId(0);
const S2: StageId = StageId(1);
const S3: StageId = StageId(2);
const S4: StageId = StageId(3);
const S5: StageId = StageId(4);
const S6: StageId = StageId(5);
const S7: StageId = StageId(6);

// ---------------------------------------------------------------------------
// Symbolic points
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Per-stage claim tables
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Builder state
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Tag → PolynomialId bridge
// ---------------------------------------------------------------------------

fn poly_id_from_tag(tag: u64) -> PolynomialId {
    match tag {
        ptag::RAM_INC => PolynomialId::RamInc,
        ptag::RD_INC => PolynomialId::RdInc,
        ptag::TRUSTED_ADVICE => PolynomialId::TrustedAdvice,
        ptag::UNTRUSTED_ADVICE => PolynomialId::UntrustedAdvice,
        ptag::RAM_READ_VALUE => PolynomialId::RamReadValue,
        ptag::RAM_WRITE_VALUE => PolynomialId::RamWriteValue,
        ptag::RAM_RA => PolynomialId::RamAddress, // virtual RA decomposition
        ptag::RAM_VAL => PolynomialId::RamVal,
        ptag::RAM_VAL_FINAL => PolynomialId::RamValFinal,
        ptag::RAM_ADDRESS => PolynomialId::RamAddress,
        ptag::RAM_HAMMING_WEIGHT => PolynomialId::HammingWeight,
        ptag::RD_WRITE_VALUE => PolynomialId::RdWriteValue,
        ptag::RS1_VALUE => PolynomialId::Rs1Value,
        ptag::RS2_VALUE => PolynomialId::Rs2Value,
        ptag::REGISTERS_VAL => PolynomialId::RegistersVal,
        ptag::RD_WA => PolynomialId::RdWa,
        ptag::RS1_RA => PolynomialId::Rs1Ra,
        ptag::RS2_RA => PolynomialId::Rs2Ra,
        ptag::LOOKUP_OUTPUT => PolynomialId::LookupOutput,
        ptag::LEFT_LOOKUP_OPERAND => PolynomialId::LeftLookupOperand,
        ptag::RIGHT_LOOKUP_OPERAND => PolynomialId::RightLookupOperand,
        ptag::LEFT_INSTRUCTION_INPUT => PolynomialId::LeftInstructionInput,
        ptag::RIGHT_INSTRUCTION_INPUT => PolynomialId::RightInstructionInput,
        ptag::IS_RD_NOT_ZERO => PolynomialId::IsRdNotZero,
        ptag::WRITE_LOOKUP_OUTPUT_TO_RD_FLAG => PolynomialId::WriteLookupToRdFlag,
        ptag::JUMP_FLAG => PolynomialId::JumpFlag,
        ptag::BRANCH_FLAG => PolynomialId::BranchFlag,
        ptag::LEFT_IS_RS1 => PolynomialId::LeftIsRs1,
        ptag::LEFT_IS_PC => PolynomialId::LeftIsPc,
        ptag::RIGHT_IS_RS2 => PolynomialId::RightIsRs2,
        ptag::RIGHT_IS_IMM => PolynomialId::RightIsImm,
        ptag::UNEXPANDED_PC => PolynomialId::UnexpandedPc,
        ptag::IMM => PolynomialId::Imm,
        ptag::NEXT_PC => PolynomialId::NextPc,
        ptag::NEXT_UNEXPANDED_PC => PolynomialId::NextUnexpandedPc,
        ptag::NEXT_IS_VIRTUAL => PolynomialId::NextIsVirtual,
        ptag::NEXT_IS_FIRST_IN_SEQUENCE => PolynomialId::NextIsFirstInSequence,
        ptag::NEXT_IS_NOOP => PolynomialId::NextIsNoop,
        t if (ptag::INSTRUCTION_RA..ptag::BYTECODE_RA).contains(&t) => {
            PolynomialId::InstructionRa((t - ptag::INSTRUCTION_RA) as usize)
        }
        t if (ptag::BYTECODE_RA..ptag::RAM_RA_COMMITTED).contains(&t) => {
            PolynomialId::BytecodeRa((t - ptag::BYTECODE_RA) as usize)
        }
        t if (ptag::RAM_RA_COMMITTED..ptag::TRUSTED_ADVICE).contains(&t) => {
            PolynomialId::RamRa((t - ptag::RAM_RA_COMMITTED) as usize)
        }
        _ => panic!("unknown polynomial tag: {tag}"),
    }
}

// ---------------------------------------------------------------------------
// Formula binding helpers
// ---------------------------------------------------------------------------

/// Bind a ClaimDefinition's opening variables to concrete ClaimIds.
///
/// `tag_to_claim` maps `polynomial_tag` → `ClaimId` for each opening binding.
fn bind_formula(def: ClaimDefinition, tag_to_claim: &HashMap<u64, ClaimId>) -> ClaimFormula {
    let opening_claims = def
        .opening_bindings
        .iter()
        .map(|b| (b.var_id, tag_to_claim[&b.polynomial_tag]))
        .collect();
    ClaimFormula {
        definition: def,
        opening_claims,
    }
}

/// Build a tag → ClaimId map from a StageClaims table, for binding formulas.
fn tag_map_from_claims(claims: &StageClaims) -> HashMap<u64, ClaimId> {
    claims
        .iter()
        .flat_map(|(&poly_id, &claim_id)| {
            poly_id_to_tags(poly_id)
                .into_iter()
                .map(move |tag| (tag, claim_id))
        })
        .collect()
}

/// PolynomialId → possible u64 tags (inverse of poly_id_from_tag).
fn poly_id_to_tags(id: PolynomialId) -> Vec<u64> {
    match id {
        PolynomialId::RamInc => vec![ptag::RAM_INC],
        PolynomialId::RdInc => vec![ptag::RD_INC],
        PolynomialId::TrustedAdvice => vec![ptag::TRUSTED_ADVICE],
        PolynomialId::UntrustedAdvice => vec![ptag::UNTRUSTED_ADVICE],
        PolynomialId::RamReadValue => vec![ptag::RAM_READ_VALUE],
        PolynomialId::RamWriteValue => vec![ptag::RAM_WRITE_VALUE],
        PolynomialId::RamAddress => vec![ptag::RAM_ADDRESS, ptag::RAM_RA],
        PolynomialId::RamVal => vec![ptag::RAM_VAL],
        PolynomialId::RamValFinal => vec![ptag::RAM_VAL_FINAL],
        PolynomialId::HammingWeight => vec![ptag::RAM_HAMMING_WEIGHT],
        PolynomialId::RdWriteValue => vec![ptag::RD_WRITE_VALUE],
        PolynomialId::Rs1Value => vec![ptag::RS1_VALUE],
        PolynomialId::Rs2Value => vec![ptag::RS2_VALUE],
        PolynomialId::RegistersVal => vec![ptag::REGISTERS_VAL],
        PolynomialId::RdWa => vec![ptag::RD_WA],
        PolynomialId::Rs1Ra => vec![ptag::RS1_RA],
        PolynomialId::Rs2Ra => vec![ptag::RS2_RA],
        PolynomialId::LookupOutput => vec![ptag::LOOKUP_OUTPUT],
        PolynomialId::LeftLookupOperand => vec![ptag::LEFT_LOOKUP_OPERAND],
        PolynomialId::RightLookupOperand => vec![ptag::RIGHT_LOOKUP_OPERAND],
        PolynomialId::LeftInstructionInput => vec![ptag::LEFT_INSTRUCTION_INPUT],
        PolynomialId::RightInstructionInput => vec![ptag::RIGHT_INSTRUCTION_INPUT],
        PolynomialId::IsRdNotZero => vec![ptag::IS_RD_NOT_ZERO],
        PolynomialId::WriteLookupToRdFlag => vec![ptag::WRITE_LOOKUP_OUTPUT_TO_RD_FLAG],
        PolynomialId::JumpFlag => vec![ptag::JUMP_FLAG],
        PolynomialId::BranchFlag => vec![ptag::BRANCH_FLAG],
        PolynomialId::LeftIsRs1 => vec![ptag::LEFT_IS_RS1],
        PolynomialId::LeftIsPc => vec![ptag::LEFT_IS_PC],
        PolynomialId::RightIsRs2 => vec![ptag::RIGHT_IS_RS2],
        PolynomialId::RightIsImm => vec![ptag::RIGHT_IS_IMM],
        PolynomialId::UnexpandedPc => vec![ptag::UNEXPANDED_PC],
        PolynomialId::Imm => vec![ptag::IMM],
        PolynomialId::NextPc => vec![ptag::NEXT_PC],
        PolynomialId::NextUnexpandedPc => vec![ptag::NEXT_UNEXPANDED_PC],
        PolynomialId::NextIsVirtual => vec![ptag::NEXT_IS_VIRTUAL],
        PolynomialId::NextIsFirstInSequence => vec![ptag::NEXT_IS_FIRST_IN_SEQUENCE],
        PolynomialId::NextIsNoop => vec![ptag::NEXT_IS_NOOP],
        PolynomialId::InstructionRa(i) => vec![ptag::instruction_ra(i)],
        PolynomialId::BytecodeRa(i) => vec![ptag::bytecode_ra(i)],
        PolynomialId::RamRa(i) => vec![ptag::ram_ra_committed(i)],
        PolynomialId::SpartanWitness => vec![],
    }
}

// ---------------------------------------------------------------------------
// Vertex construction helpers
// ---------------------------------------------------------------------------

/// Create a claim-reduction vertex: same formula for input and output, deps ↔ upstream.
///
/// `gamma_label` identifies the pre_squeeze label for the formula's challenge
/// variable (all claim reductions use a single gamma challenge).
#[allow(clippy::too_many_arguments)]
fn add_claim_reduction(
    b: &mut GraphBuilder,
    def: ClaimDefinition,
    upstream: &StageClaims,
    point: SymbolicPoint,
    degree: usize,
    num_vars: NumVars,
    weighting: PublicPolynomial,
    gamma_label: &'static str,
) -> (VertexId, StageClaims) {
    let vid = b.alloc_vertex();
    let upstream_tag_map = tag_map_from_claims(upstream);

    // Deps: upstream claims for each polynomial in the formula
    let deps: Vec<ClaimId> = def
        .opening_bindings
        .iter()
        .map(|binding| upstream_tag_map[&binding.polynomial_tag])
        .collect();

    // Produced claims: same polynomials, new point
    let mut produced = StageClaims::new();
    let mut produced_ids = Vec::new();
    let mut produced_tag_map = HashMap::new();
    for binding in &def.opening_bindings {
        let poly_id = poly_id_from_tag(binding.polynomial_tag);
        let cid = b.alloc_claim(poly_id, point.clone());
        let _ = produced.insert(poly_id, cid);
        produced_ids.push(cid);
        let _ = produced_tag_map.insert(binding.polynomial_tag, cid);
    }

    let challenge_labels: Vec<ChallengeLabel> = def
        .challenge_bindings
        .iter()
        .map(|_| ChallengeLabel::PreSqueeze(gamma_label))
        .collect();

    let input_formula = bind_formula(def.clone(), &upstream_tag_map);
    let output_formula = bind_formula(def, &produced_tag_map);

    b.push_vertex(Vertex::Sumcheck(Box::new(SumcheckVertex {
        id: vid,
        deps,
        input: InputClaim::Formula {
            formula: input_formula,
            challenge_labels,
        },
        produces: produced_ids,
        formula: output_formula,
        degree,
        num_vars,
        weighting,
        phases: vec![Phase {
            num_vars: SymbolicExpr::symbol(Symbol::LOG_T),
            variable_group: VariableGroup::Cycle,
        }],
    })));

    (vid, produced)
}

/// Sumcheck shape parameters bundled to reduce argument count.
struct SumcheckShape {
    degree: usize,
    num_vars: NumVars,
    weighting: PublicPolynomial,
    phases: Vec<Phase>,
}

/// Input claim specification for vertex construction.
enum InputSpec {
    /// Input claim is a constant (e.g., zero for booleanity).
    Constant(i64),
    /// Input claim is a formula over upstream evals. Provide:
    /// - The formula definition (same structure as output formula)
    /// - Upstream claims to bind the formula's openings to
    /// - Label for the challenge variable
    Formula {
        def: ClaimDefinition,
        upstream: StageClaims,
        gamma_label: &'static str,
    },
}

/// Create a composition vertex with a proper input claim.
fn add_composition(
    b: &mut GraphBuilder,
    output_def: ClaimDefinition,
    ordering_deps: Vec<ClaimId>,
    point: SymbolicPoint,
    input_spec: InputSpec,
    shape: SumcheckShape,
) -> (VertexId, StageClaims) {
    let vid = b.alloc_vertex();

    let mut produced = StageClaims::new();
    let mut produced_ids = Vec::new();
    let mut produced_tag_map = HashMap::new();
    for binding in &output_def.opening_bindings {
        let poly_id = poly_id_from_tag(binding.polynomial_tag);
        let cid = b.alloc_claim(poly_id, point.clone());
        let _ = produced.insert(poly_id, cid);
        produced_ids.push(cid);
        let _ = produced_tag_map.insert(binding.polynomial_tag, cid);
    }

    let formula = bind_formula(output_def, &produced_tag_map);

    let input = match input_spec {
        InputSpec::Constant(c) => InputClaim::Constant(c),
        InputSpec::Formula {
            def,
            upstream,
            gamma_label,
        } => {
            let upstream_tag_map = tag_map_from_claims(&upstream);
            let input_formula = bind_formula(def.clone(), &upstream_tag_map);
            let challenge_labels = def
                .challenge_bindings
                .iter()
                .map(|_| ChallengeLabel::PreSqueeze(gamma_label))
                .collect();
            InputClaim::Formula {
                formula: input_formula,
                challenge_labels,
            }
        }
    };

    b.push_vertex(Vertex::Sumcheck(Box::new(SumcheckVertex {
        id: vid,
        deps: ordering_deps,
        input,
        produces: produced_ids,
        formula,
        degree: shape.degree,
        num_vars: shape.num_vars,
        weighting: shape.weighting,
        phases: shape.phases,
    })));

    (vid, produced)
}

// ---------------------------------------------------------------------------
// Polynomial registration
// ---------------------------------------------------------------------------

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
}

// ---------------------------------------------------------------------------
// S1: Spartan (composite — outer + product + inner)
// ---------------------------------------------------------------------------

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
fn build_spartan(b: &mut GraphBuilder) -> StageOutput {
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
                challenge_bindings: vec![],
            },
            opening_claims: HashMap::new(),
        }
    };

    b.push_vertex(Vertex::Sumcheck(Box::new(SumcheckVertex {
        id: vid,
        deps: vec![],
        input: InputClaim::Constant(0),
        produces: produced_ids,
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
    })));

    StageOutput {
        vertex_ids: vec![vid],
        claims,
    }
}

// ---------------------------------------------------------------------------
// S2: Product Virtual + RA Virtual sumchecks
// ---------------------------------------------------------------------------

fn build_s2(b: &mut GraphBuilder, config: &ProtocolConfig, s1: &StageClaims) -> StageOutput {
    let pt = SymbolicPoint::Challenges(S2);
    let mut out = StageOutput::new();

    // V_pv: product virtual remainder
    {
        let def = claims::spartan::product_virtual_remainder();
        let ordering_deps = deps_from_formula(&def, s1);
        let (vid, claims) = add_composition(
            b,
            def,
            ordering_deps,
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

    // V_instr_ra_virtual: instruction RA one-hot
    if config.d_instr > 0 {
        let def = claims::instruction::instruction_ra_virtual(1, config.d_instr);
        let ordering_deps = deps_from_formula(&def, s1);
        let (vid, claims) = add_composition(
            b,
            def,
            ordering_deps,
            pt.clone(),
            InputSpec::Constant(0),
            SumcheckShape {
                degree: config.d_instr + 1,
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

    // V_bc_ra_virtual: bytecode RA one-hot
    if config.d_bc > 0 {
        let def = claims::bytecode::bytecode_ra_virtual(config.d_bc);
        let ordering_deps = deps_from_formula(&def, s1);
        let (vid, claims) = add_composition(
            b,
            def,
            ordering_deps,
            pt.clone(),
            InputSpec::Constant(0),
            SumcheckShape {
                degree: config.d_bc + 1,
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

    // V_ram_ra_virtual: RAM RA one-hot
    if config.d_ram > 0 {
        let def = claims::ram::ram_ra_virtual(config.d_ram);
        let ordering_deps = deps_from_formula(&def, s1);
        let (vid, claims) = add_composition(
            b,
            def,
            ordering_deps,
            pt,
            InputSpec::Constant(0),
            SumcheckShape {
                degree: config.d_ram + 1,
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

    out
}

/// Extract dep ClaimIds for a formula's opening bindings from a stage's claims.
fn deps_from_formula(def: &ClaimDefinition, stage: &StageClaims) -> Vec<ClaimId> {
    let tag_map = tag_map_from_claims(stage);
    def.opening_bindings
        .iter()
        .filter_map(|b| tag_map.get(&b.polynomial_tag).copied())
        .collect()
}

/// Merge multiple StageClaims tables, choosing claims from later stages when
/// the same polynomial appears in multiple stages.
fn merge_claims(stages: &[&StageClaims]) -> StageClaims {
    let mut merged = StageClaims::new();
    for s in stages {
        merged.extend(s.iter().map(|(&k, &v)| (k, v)));
    }
    merged
}

// ---------------------------------------------------------------------------
// S3: Shift + InstructionInput + RegistersCR + InstrLookupsCR
// ---------------------------------------------------------------------------

fn build_s3(b: &mut GraphBuilder, s1: &StageClaims, s2: &StageClaims) -> StageOutput {
    let pt = SymbolicPoint::Challenges(S3);
    let available = merge_claims(&[s1, s2]);
    let mut out = StageOutput::new();

    // V_shift
    {
        let def = claims::spartan::shift();
        let ordering_deps = deps_from_formula(&def, &available);
        let (vid, claims) = add_composition(
            b,
            def,
            ordering_deps,
            pt.clone(),
            InputSpec::Constant(0),
            SumcheckShape {
                degree: 2,
                num_vars: log_t(),
                weighting: PublicPolynomial::EqPlusOne,
                phases: vec![Phase {
                    num_vars: log_t(),
                    variable_group: VariableGroup::Cycle,
                }],
            },
        );
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
    }

    // V_instr_input
    {
        let def = claims::spartan::instruction_input();
        let ordering_deps = deps_from_formula(&def, &available);
        let (vid, claims) = add_composition(
            b,
            def,
            ordering_deps,
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

    // V_registers_cr: reduces RdWriteValue, Rs1Value, Rs2Value from S1
    {
        let def = claims::reductions::registers_claim_reduction();
        let upstream = filter_claims(s1, &def);
        let (vid, claims) = add_claim_reduction(
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

    // V_instr_lookups_cr: reduces LookupOutput, operands, instruction inputs
    {
        let def = claims::reductions::instruction_lookups_claim_reduction();
        // These claims come from S1 (original Spartan evals) or S2 (if V_pv updated them).
        let upstream = filter_claims(&available, &def);
        let (vid, claims) = add_claim_reduction(
            b,
            def,
            &upstream,
            pt,
            2,
            log_t(),
            PublicPolynomial::Eq,
            "instr_gamma",
        );
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
    }

    out
}

/// Filter a StageClaims table to only include polynomials referenced by a formula.
fn filter_claims(stage: &StageClaims, def: &ClaimDefinition) -> StageClaims {
    let needed: std::collections::HashSet<PolynomialId> = def
        .opening_bindings
        .iter()
        .map(|b| poly_id_from_tag(b.polynomial_tag))
        .collect();
    stage
        .iter()
        .filter(|(k, _)| needed.contains(k))
        .map(|(&k, &v)| (k, v))
        .collect()
}

// ---------------------------------------------------------------------------
// S4: RegistersRW + RamValCheck + RamRW + RamOutputCheck + RamRafEval
// ---------------------------------------------------------------------------

fn build_s4(
    b: &mut GraphBuilder,
    config: &ProtocolConfig,
    s1: &StageClaims,
    s3: &StageClaims,
) -> StageOutput {
    let pt = SymbolicPoint::Challenges(S4);
    let available = merge_claims(&[s1, s3]);
    let mut out = StageOutput::new();

    // V_registers_rw
    {
        let def = claims::registers::registers_read_write_checking();
        let upstream = filter_claims(&available, &def);
        let (vid, claims) = add_composition(
            b,
            def,
            deps_from_formula_map(&upstream),
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

    // V_ram_rw
    {
        let def = claims::ram::ram_read_write_checking();
        let upstream = filter_claims(&available, &def);
        let (vid, claims) = add_composition(
            b,
            def,
            deps_from_formula_map(&upstream),
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

    // V_ram_val_check
    {
        let output_def = claims::ram::ram_val_check();
        let input_def = claims::ram::ram_val_check_input(config.n_advice);
        let upstream = filter_claims(&available, &input_def);
        let upstream_tag_map = tag_map_from_claims(&upstream);

        let vid = b.alloc_vertex();
        let mut produced = StageClaims::new();
        let mut produced_ids = Vec::new();
        let mut produced_tag_map = HashMap::new();
        for binding in &output_def.opening_bindings {
            let poly_id = poly_id_from_tag(binding.polynomial_tag);
            let cid = b.alloc_claim(poly_id, pt.clone());
            let _ = produced.insert(poly_id, cid);
            produced_ids.push(cid);
            let _ = produced_tag_map.insert(binding.polynomial_tag, cid);
        }

        let input_formula = if upstream_tag_map.is_empty() {
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
                formula: bind_formula(input_def, &upstream_tag_map),
                challenge_labels,
            }
        };
        let output_formula = bind_formula(output_def, &produced_tag_map);

        let deps: Vec<ClaimId> = upstream.values().copied().collect();
        b.push_vertex(Vertex::Sumcheck(Box::new(SumcheckVertex {
            id: vid,
            deps,
            input: input_formula,
            produces: produced_ids,
            formula: output_formula,
            degree: 3,
            num_vars: log_t(),
            weighting: PublicPolynomial::Lt,
            phases: vec![Phase {
                num_vars: log_t(),
                variable_group: VariableGroup::Cycle,
            }],
        })));
        out.vertex_ids.push(vid);
        out.claims.extend(produced);
    }

    // V_ram_output_check
    {
        let def = claims::ram::ram_output_check();
        let upstream = filter_claims(&available, &def);
        let (vid, claims) = add_composition(
            b,
            def,
            deps_from_formula_map(&upstream),
            pt.clone(),
            InputSpec::Constant(0),
            SumcheckShape {
                degree: 2,
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

    // V_ram_raf_eval
    {
        let def = claims::ram::ram_raf_evaluation();
        let upstream = filter_claims(&available, &def);
        let (vid, claims) = add_composition(
            b,
            def,
            deps_from_formula_map(&upstream),
            pt,
            InputSpec::Constant(0),
            SumcheckShape {
                degree: 2,
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

    out
}

fn deps_from_formula_map(claims: &StageClaims) -> Vec<ClaimId> {
    claims.values().copied().collect()
}

// ---------------------------------------------------------------------------
// S5: RegistersValEval
// ---------------------------------------------------------------------------

fn build_s5(b: &mut GraphBuilder, s1: &StageClaims, s4: &StageClaims) -> StageOutput {
    let pt = SymbolicPoint::Challenges(S5);
    let available = merge_claims(&[s1, s4]);
    let mut out = StageOutput::new();

    let def = claims::registers::registers_val_evaluation();
    let ordering_deps = deps_from_formula(&def, &available);
    let (vid, claims) = add_composition(
        b,
        def,
        ordering_deps,
        pt,
        InputSpec::Constant(0),
        SumcheckShape {
            degree: 3,
            num_vars: log_t(),
            weighting: PublicPolynomial::Lt,
            phases: vec![Phase {
                num_vars: log_t(),
                variable_group: VariableGroup::Cycle,
            }],
        },
    );
    out.vertex_ids.push(vid);
    out.claims.extend(claims);
    out
}

// ---------------------------------------------------------------------------
// S6: IncrementCR + HammingBooleanity + RA Booleanity
// ---------------------------------------------------------------------------

fn build_s6(
    b: &mut GraphBuilder,
    config: &ProtocolConfig,
    s2: &StageClaims,
    s4: &StageClaims,
    s5: &StageClaims,
) -> StageOutput {
    let pt = SymbolicPoint::Challenges(S6);
    let available = merge_claims(&[s2, s4, s5]);
    let mut out = StageOutput::new();

    // V_inc_cr: reduces RamInc + RdInc claims from S2/S4/S5
    {
        let def = claims::reductions::increment_claim_reduction();
        let upstream = filter_claims(&available, &def);
        let (vid, claims) = add_claim_reduction(
            b,
            def,
            &upstream,
            pt.clone(),
            2,
            log_t(),
            PublicPolynomial::Derived,
            "inc_gamma",
        );
        out.vertex_ids.push(vid);
        out.claims.extend(claims);
    }

    // V_hamming_bool: proves HammingWeight is boolean
    {
        let def = claims::ram::hamming_booleanity();
        let (vid, claims) = add_composition(
            b,
            def,
            vec![], // no upstream deps — independent zero-check
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

    // V_ra_bool: proves all RA polynomials are boolean (one-hot)
    if config.d_total() > 0 {
        let tags = config.all_ra_tags();
        let def = claims::booleanity::ra_booleanity(tags.len(), &tags);
        let (vid, claims) = add_composition(
            b,
            def,
            vec![], // independent zero-check
            pt,
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

    out
}

// ---------------------------------------------------------------------------
// S7: HammingWeightCR (+ RamRaCR, AdviceCR)
// ---------------------------------------------------------------------------

fn build_s7(b: &mut GraphBuilder, config: &ProtocolConfig, s6: &StageClaims) -> StageOutput {
    let pt = unified_point();
    let mut out = StageOutput::new();

    // V_hw_cr: hamming weight claim reduction over all RA polynomials
    if config.d_total() > 0 {
        let tags = config.all_ra_tags();
        let def = claims::reductions::hamming_weight_claim_reduction(&tags);
        let upstream = filter_claims(s6, &def);
        let (vid, claims) = add_claim_reduction(
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

    // V_ram_ra_cr
    {
        let def = claims::reductions::ram_ra_claim_reduction();
        let upstream = filter_claims(s6, &def);
        if !upstream.is_empty() {
            let (vid, claims) = add_claim_reduction(
                b,
                def,
                &upstream,
                pt,
                2,
                log_t(),
                PublicPolynomial::Eq,
                "hw_gamma",
            );
            out.vertex_ids.push(vid);
            out.claims.extend(claims);
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Point normalization: dense poly claims (r_cycle dim) → unified point dim
// ---------------------------------------------------------------------------

fn build_normalization(b: &mut GraphBuilder, s6: &StageClaims) -> StageOutput {
    let vid = b.alloc_vertex();
    let target = unified_point();
    let padding = SymbolicPoint::Challenges(S7);

    // Normalize RamInc and RdInc from r_cycle (S6) to unified
    let mut consumes = Vec::new();
    let mut produces = Vec::new();
    let mut claims = StageClaims::new();

    for &poly_id in &[PolynomialId::RamInc, PolynomialId::RdInc] {
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

// ---------------------------------------------------------------------------
// Opening stage
// ---------------------------------------------------------------------------

fn build_opening(
    b: &mut GraphBuilder,
    s1: &StageClaims,
    s7: &StageClaims,
    norm: &StageClaims,
) -> (Vec<VertexId>, Vec<ClaimId>) {
    let available = merge_claims(&[norm, s7, s1]);
    let mut vids = Vec::new();
    let mut consumed = Vec::new();

    // Open all committed polynomial claims
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

// ---------------------------------------------------------------------------
// Staging construction
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Commitment strategy
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Top-level construction
// ---------------------------------------------------------------------------

/// Build the complete Jolt protocol graph from configuration.
pub fn build_jolt_protocol(config: ProtocolConfig) -> ProtocolGraph {
    let mut b = GraphBuilder::new();
    register_all_polynomials(&mut b, &config);

    // Build stages in dependency order
    let s1_out = build_spartan(&mut b);
    let s2_out = build_s2(&mut b, &config, &s1_out.claims);
    let s3_out = build_s3(&mut b, &s1_out.claims, &s2_out.claims);
    let s4_out = build_s4(&mut b, &config, &s1_out.claims, &s3_out.claims);
    let s5_out = build_s5(&mut b, &s1_out.claims, &s4_out.claims);
    let s6_out = build_s6(
        &mut b,
        &config,
        &s2_out.claims,
        &s4_out.claims,
        &s5_out.claims,
    );
    let s7_out = build_s7(&mut b, &config, &s6_out.claims);
    let norm_out = build_normalization(&mut b, &s6_out.claims);
    let (opening_vids, _consumed) =
        build_opening(&mut b, &s1_out.claims, &s7_out.claims, &norm_out.claims);

    let log_t = SymbolicExpr::symbol(Symbol::LOG_T);
    let log_k = SymbolicExpr::symbol(Symbol::LOG_K);
    let d_total = SymbolicExpr::concrete(config.d_total());

    // Challenge squeeze specs per stage — must match stages.rs and verify.rs exactly.
    let s1_sq: Vec<ChallengeSpec> = vec![]; // Spartan: opaque internal transcript
    let s2_sq = vec![ChallengeSpec::GammaPowers {
        label: "pv_gamma",
        count: SymbolicExpr::concrete(5),
    }];
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
    let s5_sq = vec![
        ChallengeSpec::Scalar {
            label: "instr_raf_gamma",
        },
        ChallengeSpec::Scalar {
            label: "ram_ra_gamma",
        },
        ChallengeSpec::Vector {
            label: "reg_val_r",
            dim: log_t.clone(),
        },
    ];
    let s6_sq = vec![
        ChallengeSpec::Scalar {
            label: "bytecode_gamma",
        },
        ChallengeSpec::Vector {
            label: "bool_eq",
            dim: log_t.clone(),
        },
        ChallengeSpec::Scalar {
            label: "bool_gamma",
        },
        ChallengeSpec::Vector {
            label: "h_eq_point",
            dim: log_t.clone(),
        },
        ChallengeSpec::Scalar {
            label: "ra_virt_gamma",
        },
        ChallengeSpec::Scalar {
            label: "instr_ra_gamma",
        },
        ChallengeSpec::Scalar { label: "inc_gamma" },
    ];
    let s7_sq = vec![ChallengeSpec::GammaPowers {
        label: "hw_gamma",
        count: d_total,
    }];

    let staging = build_staging(
        &[
            (
                S1,
                &s1_out,
                SymbolicExpr::symbol(Symbol::LOG_ROWS) + SymbolicExpr::symbol(Symbol::LOG_COLS),
                s1_sq,
            ),
            (S2, &s2_out, log_t.clone(), s2_sq),
            (S3, &s3_out, log_t.clone(), s3_sq),
            (S4, &s4_out, log_t.clone(), s4_sq),
            (S5, &s5_out, log_t.clone(), s5_sq),
            (S6, &s6_out, log_t, s6_sq),
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn default_config() -> ProtocolConfig {
        ProtocolConfig {
            d_instr: 8,
            d_bc: 4,
            d_ram: 3,
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
            for &cid in v.produced_claims() {
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

        // S2: 1 GammaPowers
        assert_eq!(stages[1].pre_squeeze.len(), 1);
        assert!(matches!(
            stages[1].pre_squeeze[0],
            ChallengeSpec::GammaPowers {
                label: "pv_gamma",
                ..
            }
        ));

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

        // S5: 2 Scalar + 1 Vector
        assert_eq!(stages[4].pre_squeeze.len(), 3);
        assert!(matches!(
            stages[4].pre_squeeze[2],
            ChallengeSpec::Vector {
                label: "reg_val_r",
                ..
            }
        ));

        // S6: 5 Scalar + 2 Vector
        assert_eq!(stages[5].pre_squeeze.len(), 7);
        assert!(matches!(
            stages[5].pre_squeeze[6],
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
    fn poly_id_tag_roundtrip() {
        let test_cases = [
            (ptag::RAM_INC, PolynomialId::RamInc),
            (ptag::RD_INC, PolynomialId::RdInc),
            (ptag::instruction_ra(3), PolynomialId::InstructionRa(3)),
            (ptag::bytecode_ra(2), PolynomialId::BytecodeRa(2)),
            (ptag::ram_ra_committed(1), PolynomialId::RamRa(1)),
            (ptag::LOOKUP_OUTPUT, PolynomialId::LookupOutput),
            (ptag::NEXT_IS_NOOP, PolynomialId::NextIsNoop),
        ];
        for (tag, expected) in test_cases {
            assert_eq!(poly_id_from_tag(tag), expected, "tag={tag}");
        }
    }
}
