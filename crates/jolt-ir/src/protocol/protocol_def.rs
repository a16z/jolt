//! Jolt protocol definition: the pure math layer.
//!
//! [`jolt_identities`] enumerates all polynomial identities that the Jolt
//! SNARK proves. [`jolt_hints`] returns the scheduling decisions currently
//! baked into `build_jolt_protocol()`, extracted into a data structure.
//!
//! Together, `compile(jolt_identities(..), jolt_hints(..), ..)` produces
//! the same `ProtocolGraph` as the hand-wired `build_jolt_protocol()`.
//!
//! # Identity naming
//!
//! Each identity has a short `name` matching the corresponding struct/stage
//! in jolt-core. The `IdentityId` is deterministic and stable.

use super::identity::*;
use super::symbolic::{Symbol, SymbolicExpr};
use crate::claim::ClaimDefinition;
use crate::zkvm::claims;
use crate::{ExprBuilder, OpeningBinding, PolynomialId};

use super::build::ProtocolConfig;

// ---------------------------------------------------------------------------
// Identity ID constants — deterministic, stable across recompilations.
// ---------------------------------------------------------------------------

pub const SPARTAN_OUTER: IdentityId = IdentityId(0);
pub const SPARTAN_INNER: IdentityId = IdentityId(1);
pub const RAM_READ_WRITE: IdentityId = IdentityId(2);
pub const PRODUCT_VIRTUAL: IdentityId = IdentityId(3);
pub const INSTR_LOOKUPS_CR: IdentityId = IdentityId(4);
pub const RAM_RAF_EVAL: IdentityId = IdentityId(5);
pub const RAM_OUTPUT_CHECK: IdentityId = IdentityId(6);
pub const SHIFT: IdentityId = IdentityId(7);
pub const INSTRUCTION_INPUT: IdentityId = IdentityId(8);
pub const REGISTERS_CR: IdentityId = IdentityId(9);
pub const REGISTERS_RW: IdentityId = IdentityId(10);
pub const RAM_VAL_CHECK: IdentityId = IdentityId(11);
pub const INSTR_READ_RAF: IdentityId = IdentityId(12);
pub const RAM_RA_CR: IdentityId = IdentityId(13);
pub const REGISTERS_VAL_EVAL: IdentityId = IdentityId(14);
pub const BYTECODE_READ_RAF: IdentityId = IdentityId(15);
pub const BOOLEANITY: IdentityId = IdentityId(16);
pub const HAMMING_BOOLEANITY: IdentityId = IdentityId(17);
pub const RAM_RA_VIRTUAL: IdentityId = IdentityId(18);
pub const INSTR_RA_VIRTUAL: IdentityId = IdentityId(19);
pub const INCREMENT_CR: IdentityId = IdentityId(20);
pub const HAMMING_WEIGHT_CR: IdentityId = IdentityId(21);
pub const ADVICE_CR: IdentityId = IdentityId(22);

/// All polynomial identities that the Jolt SNARK proves.
///
/// Returns a `Vec` of ~23 identities (some config-conditional). Each
/// identity is a pure mathematical statement — no scheduling decisions.
pub fn jolt_identities(config: &ProtocolConfig) -> Vec<PolynomialIdentity> {
    let mut ids = Vec::with_capacity(23);

    // -----------------------------------------------------------------------
    // S1: Spartan R1CS — outer (row) + inner (column) sumchecks
    // -----------------------------------------------------------------------
    ids.push(spartan_outer_identity(config));
    ids.push(spartan_inner_identity());

    // -----------------------------------------------------------------------
    // S2: RAM read-write checking + claim reductions
    // -----------------------------------------------------------------------
    ids.push(ram_read_write_identity());
    ids.push(product_virtual_identity());
    ids.push(instr_lookups_cr_identity());
    ids.push(ram_raf_eval_identity());
    ids.push(ram_output_check_identity());

    // -----------------------------------------------------------------------
    // S3: Shift + instruction input + registers CR
    // -----------------------------------------------------------------------
    ids.push(shift_identity());
    ids.push(instruction_input_identity());
    ids.push(registers_cr_identity());

    // -----------------------------------------------------------------------
    // S4: Registers RW + RAM val check
    // -----------------------------------------------------------------------
    ids.push(registers_rw_identity());
    ids.push(ram_val_check_identity(config.n_advice));

    // -----------------------------------------------------------------------
    // S5: Instruction read RAF + RAM RA CR + registers val eval
    // -----------------------------------------------------------------------
    if config.d_instr > 0 {
        ids.push(instr_read_raf_identity(config));
    }
    ids.push(ram_ra_cr_identity());
    ids.push(registers_val_eval_identity());

    // -----------------------------------------------------------------------
    // S6: Bytecode + booleanity + virtual sumchecks + increment
    // -----------------------------------------------------------------------
    if config.d_bc > 0 {
        ids.push(bytecode_read_raf_identity(config));
    }
    if config.d_total() > 0 {
        ids.push(booleanity_identity(config));
    }
    ids.push(hamming_booleanity_identity());
    if config.d_ram > 0 {
        ids.push(ram_ra_virtual_identity(config));
    }
    if config.d_instr > 0 {
        ids.push(instr_ra_virtual_identity(config));
    }
    ids.push(increment_cr_identity());

    // -----------------------------------------------------------------------
    // S7: Hamming weight CR + advice CR
    // -----------------------------------------------------------------------
    if config.d_total() > 0 {
        ids.push(hamming_weight_cr_identity(config));
    }
    if config.n_advice > 0 {
        ids.push(advice_cr_identity());
    }

    ids
}

/// Scheduling hints matching the current hand-tuned `build_jolt_protocol()`.
pub fn jolt_hints(config: &ProtocolConfig) -> SchedulingHints {
    let log_t = || SymbolicExpr::symbol(Symbol::LOG_T);
    let log_k = || SymbolicExpr::symbol(Symbol::LOG_K);

    let mut assignment = vec![
        (SPARTAN_OUTER, 0),
        (SPARTAN_INNER, 0),
        (RAM_READ_WRITE, 1),
        (PRODUCT_VIRTUAL, 1),
        (INSTR_LOOKUPS_CR, 1),
        (RAM_RAF_EVAL, 1),
        (RAM_OUTPUT_CHECK, 1),
        (SHIFT, 2),
        (INSTRUCTION_INPUT, 2),
        (REGISTERS_CR, 2),
        (REGISTERS_RW, 3),
        (RAM_VAL_CHECK, 3),
    ];

    // S5 identities
    if config.d_instr > 0 {
        assignment.push((INSTR_READ_RAF, 4));
    }
    assignment.push((RAM_RA_CR, 4));
    assignment.push((REGISTERS_VAL_EVAL, 4));

    // S6 identities
    if config.d_bc > 0 {
        assignment.push((BYTECODE_READ_RAF, 5));
    }
    if config.d_total() > 0 {
        assignment.push((BOOLEANITY, 5));
    }
    assignment.push((HAMMING_BOOLEANITY, 5));
    if config.d_ram > 0 {
        assignment.push((RAM_RA_VIRTUAL, 5));
    }
    if config.d_instr > 0 {
        assignment.push((INSTR_RA_VIRTUAL, 5));
    }
    assignment.push((INCREMENT_CR, 5));

    // S7 identities
    if config.d_total() > 0 {
        assignment.push((HAMMING_WEIGHT_CR, 6));
    }
    if config.n_advice > 0 {
        assignment.push((ADVICE_CR, 6));
    }

    let addr_cycle_phases = |log_k_expr, log_t_expr| -> Option<Vec<PhaseHint>> {
        Some(vec![
            PhaseHint {
                num_vars: log_k_expr,
                variable_group: PhaseVariableGroup::Address,
            },
            PhaseHint {
                num_vars: log_t_expr,
                variable_group: PhaseVariableGroup::Cycle,
            },
        ])
    };

    let identity_meta = vec![
        (
            SPARTAN_OUTER,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: None,
            },
        ),
        (
            SPARTAN_INNER,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: None,
            },
        ),
        (
            RAM_READ_WRITE,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: None, // cycle-only
            },
        ),
        (
            PRODUCT_VIRTUAL,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: Some(vec![
                    PhaseHint {
                        num_vars: SymbolicExpr::concrete(1),
                        variable_group: PhaseVariableGroup::Cycle,
                    },
                    PhaseHint {
                        num_vars: log_t(),
                        variable_group: PhaseVariableGroup::Cycle,
                    },
                ]),
            },
        ),
        (
            INSTR_LOOKUPS_CR,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: None,
            },
        ),
        (
            RAM_RAF_EVAL,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: None, // cycle-only despite log_K+log_T domain
            },
        ),
        (
            RAM_OUTPUT_CHECK,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: None, // cycle-only
            },
        ),
        (
            SHIFT,
            IdentityMeta {
                weighting: WeightingHint::EqPlusOne,
                phases: None,
            },
        ),
        (
            INSTRUCTION_INPUT,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: None,
            },
        ),
        (
            REGISTERS_CR,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: None,
            },
        ),
        (
            REGISTERS_RW,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: addr_cycle_phases(log_k(), log_t()),
            },
        ),
        (
            RAM_VAL_CHECK,
            IdentityMeta {
                weighting: WeightingHint::Lt,
                phases: None,
            },
        ),
        (
            RAM_RA_CR,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: None,
            },
        ),
        (
            REGISTERS_VAL_EVAL,
            IdentityMeta {
                weighting: WeightingHint::Lt,
                phases: None,
            },
        ),
        (
            HAMMING_BOOLEANITY,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: None,
            },
        ),
        (
            INCREMENT_CR,
            IdentityMeta {
                weighting: WeightingHint::Derived,
                phases: None,
            },
        ),
        (
            HAMMING_WEIGHT_CR,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: None,
            },
        ),
    ];

    // Config-conditional meta entries
    let mut meta = identity_meta;
    if config.d_instr > 0 {
        meta.push((
            INSTR_READ_RAF,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: addr_cycle_phases(log_k(), log_t()),
            },
        ));
        meta.push((
            INSTR_RA_VIRTUAL,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: addr_cycle_phases(log_k(), log_t()),
            },
        ));
    }
    if config.d_bc > 0 {
        meta.push((
            BYTECODE_READ_RAF,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: addr_cycle_phases(log_k(), log_t()),
            },
        ));
    }
    if config.d_total() > 0 {
        meta.push((
            BOOLEANITY,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: addr_cycle_phases(log_k(), log_t()),
            },
        ));
    }
    if config.d_ram > 0 {
        meta.push((
            RAM_RA_VIRTUAL,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: addr_cycle_phases(log_k(), log_t()),
            },
        ));
    }
    if config.n_advice > 0 {
        meta.push((
            ADVICE_CR,
            IdentityMeta {
                weighting: WeightingHint::Eq,
                phases: None,
            },
        ));
    }

    SchedulingHints {
        stage_assignment: assignment,
        batch_groups: vec![],
        commitment_groups: vec![],
        opening_groups: vec![],
        identity_meta: meta,
    }
}

// ---------------------------------------------------------------------------
// Per-identity constructors
// ---------------------------------------------------------------------------

/// R1CS outer sumcheck: Σ eq(τ,x) · [Az(x)·Bz(x) − Cz(x)] = 0
///
/// Runs over the row dimension ({0,1}^log_rows). Produces Az, Bz, Cz at the
/// row challenge point r_x, plus all virtual polynomial evaluations at r_cycle
/// (which equals r_x restricted to LOG_T variables).
fn spartan_outer_identity(config: &ProtocolConfig) -> PolynomialIdentity {
    let output = claims::spartan::r1cs_outer();
    let mut produces = output.polynomials(); // [Az, Bz, Cz]

    // All virtual polynomial evaluations at r_cycle (the outer's challenge point).
    produces.extend_from_slice(&[
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
        PolynomialId::OpFlag(0),
        PolynomialId::OpFlag(1),
        PolynomialId::OpFlag(2),
        PolynomialId::OpFlag(3),
        PolynomialId::OpFlag(4),
        PolynomialId::OpFlag(8),
        PolynomialId::OpFlag(9),
        PolynomialId::OpFlag(10),
        PolynomialId::OpFlag(11),
        PolynomialId::OpFlag(13),
        PolynomialId::ExpandedPc,
        PolynomialId::InstructionRafFlag,
    ]);
    for i in 0..config.n_lookup_tables {
        produces.push(PolynomialId::LookupTableFlag(i));
    }

    PolynomialIdentity {
        id: SPARTAN_OUTER,
        name: "spartan_outer",
        produces,
        output,
        input: IdentityClaim::Zero,
        degree: 3,
        domain: DomainSpec::Symbolic(SymbolicExpr::symbol(Symbol::LOG_ROWS)),
    }
}

/// R1CS inner sumcheck: Σ eq(s,y) · combined_row(y) · W(y) = c_a·Az + c_b·Bz + c_c·Cz
///
/// Runs over the column dimension ({0,1}^log_cols). The `combined_row` polynomial
/// is materialized by the edge transform between outer and inner:
/// `combined_row(y) = c_a·A(r_x,y) + c_b·B(r_x,y) + c_c·C(r_x,y)`.
///
/// Produces CombinedRow and SpartanWitness evaluations at r_y.
fn spartan_inner_identity() -> PolynomialIdentity {
    let input_formula = {
        let eb = ExprBuilder::new();
        let az = eb.opening(0);
        let bz = eb.opening(1);
        let cz = eb.opening(2);
        let gamma = eb.challenge(0);
        ClaimDefinition {
            expr: eb.build(az + gamma * bz + gamma * gamma * cz),
            opening_bindings: vec![
                OpeningBinding {
                    var_id: 0,
                    polynomial: PolynomialId::Az,
                },
                OpeningBinding {
                    var_id: 1,
                    polynomial: PolynomialId::Bz,
                },
                OpeningBinding {
                    var_id: 2,
                    polynomial: PolynomialId::Cz,
                },
            ],
            num_challenges: 1,
        }
    };

    let output = claims::spartan::r1cs_inner();
    PolynomialIdentity {
        id: SPARTAN_INNER,
        name: "spartan_inner",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Predecessor(PredecessorClaim {
            formula: input_formula,
            challenge_labels: vec![ChallengeLabel::PreSqueeze("spartan_rlc")],
            source_bindings: vec![],
        }),
        degree: 3,
        domain: DomainSpec::Symbolic(SymbolicExpr::symbol(Symbol::LOG_COLS)),
    }
}

fn ram_read_write_identity() -> PolynomialIdentity {
    let input_formula = {
        let eb = ExprBuilder::new();
        let rv = eb.opening(0);
        let wv = eb.opening(1);
        let gamma = eb.challenge(0);
        ClaimDefinition {
            expr: eb.build(rv + gamma * wv),
            opening_bindings: vec![
                OpeningBinding {
                    var_id: 0,
                    polynomial: PolynomialId::RamReadValue,
                },
                OpeningBinding {
                    var_id: 1,
                    polynomial: PolynomialId::RamWriteValue,
                },
            ],
            num_challenges: 1,
        }
    };

    let output = claims::ram::ram_read_write_checking();
    PolynomialIdentity {
        id: RAM_READ_WRITE,
        name: "ram_read_write",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Predecessor(PredecessorClaim {
            formula: input_formula,
            challenge_labels: vec![ChallengeLabel::PreSqueeze("ram_rw_gamma")],
            source_bindings: vec![],
        }),
        degree: 3,
        domain: DomainSpec::TraceTimesAddress,
    }
}

fn product_virtual_identity() -> PolynomialIdentity {
    let output = claims::spartan::product_virtual_remainder();
    let mut produces = output.polynomials();
    produces.push(PolynomialId::NextIsVirtual);
    PolynomialIdentity {
        id: PRODUCT_VIRTUAL,
        name: "product_virtual",
        produces,
        output,
        input: IdentityClaim::Constant(0),
        degree: 3,
        domain: DomainSpec::Symbolic(
            SymbolicExpr::concrete(1) + SymbolicExpr::symbol(Symbol::LOG_T),
        ),
    }
}

fn instr_lookups_cr_identity() -> PolynomialIdentity {
    let output = claims::reductions::instruction_lookups_claim_reduction();
    PolynomialIdentity {
        id: INSTR_LOOKUPS_CR,
        name: "instr_lookups_cr",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Reduction {
            gamma_label: "instr_cr_gamma",
        },
        degree: 2,
        domain: DomainSpec::TraceLength,
    }
}

fn ram_raf_eval_identity() -> PolynomialIdentity {
    let input_formula = {
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

    let output = claims::ram::ram_raf_evaluation();
    PolynomialIdentity {
        id: RAM_RAF_EVAL,
        name: "ram_raf_eval",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Predecessor(PredecessorClaim {
            formula: input_formula,
            challenge_labels: vec![ChallengeLabel::External("raf_scale")],
            source_bindings: vec![],
        }),
        degree: 2,
        domain: DomainSpec::TraceTimesAddress,
    }
}

fn ram_output_check_identity() -> PolynomialIdentity {
    let output = claims::ram::ram_output_check();
    PolynomialIdentity {
        id: RAM_OUTPUT_CHECK,
        name: "ram_output_check",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Zero,
        degree: 2,
        domain: DomainSpec::TraceTimesAddress,
    }
}

fn shift_identity() -> PolynomialIdentity {
    let input_formula = {
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
                next_unexp + g * next_pc + g2 * next_virt + g3 * next_first + g4 - g4 * next_noop,
            ),
            opening_bindings: vec![
                OpeningBinding {
                    var_id: 0,
                    polynomial: PolynomialId::NextUnexpandedPc,
                },
                OpeningBinding {
                    var_id: 1,
                    polynomial: PolynomialId::NextPc,
                },
                OpeningBinding {
                    var_id: 2,
                    polynomial: PolynomialId::NextIsVirtual,
                },
                OpeningBinding {
                    var_id: 3,
                    polynomial: PolynomialId::NextIsFirstInSequence,
                },
                OpeningBinding {
                    var_id: 4,
                    polynomial: PolynomialId::NextIsNoop,
                },
            ],
            num_challenges: 1,
        }
    };

    let output = claims::spartan::shift();
    let mut produces = output.polynomials();
    produces.push(PolynomialId::ExpandedPc);
    PolynomialIdentity {
        id: SHIFT,
        name: "shift",
        produces,
        output,
        input: IdentityClaim::Predecessor(PredecessorClaim {
            formula: input_formula,
            challenge_labels: vec![ChallengeLabel::PreSqueeze("shift_gamma")],
            source_bindings: vec![],
        }),
        degree: 2,
        domain: DomainSpec::TraceLength,
    }
}

fn instruction_input_identity() -> PolynomialIdentity {
    let input_formula = {
        let eb = ExprBuilder::new();
        let right = eb.opening(0);
        let left = eb.opening(1);
        let gamma = eb.challenge(0);
        ClaimDefinition {
            expr: eb.build(right + gamma * left),
            opening_bindings: vec![
                OpeningBinding {
                    var_id: 0,
                    polynomial: PolynomialId::RightInstructionInput,
                },
                OpeningBinding {
                    var_id: 1,
                    polynomial: PolynomialId::LeftInstructionInput,
                },
            ],
            num_challenges: 1,
        }
    };

    let output = claims::spartan::instruction_input();
    PolynomialIdentity {
        id: INSTRUCTION_INPUT,
        name: "instruction_input",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Predecessor(PredecessorClaim {
            formula: input_formula,
            challenge_labels: vec![ChallengeLabel::PreSqueeze("instr_gamma")],
            source_bindings: vec![],
        }),
        degree: 3,
        domain: DomainSpec::TraceLength,
    }
}

fn registers_cr_identity() -> PolynomialIdentity {
    let output = claims::reductions::registers_claim_reduction();
    PolynomialIdentity {
        id: REGISTERS_CR,
        name: "registers_cr",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Reduction {
            gamma_label: "reg_gamma",
        },
        degree: 2,
        domain: DomainSpec::TraceLength,
    }
}

fn registers_rw_identity() -> PolynomialIdentity {
    let input_formula = {
        let eb = ExprBuilder::new();
        let rd_wv = eb.opening(0);
        let rs1 = eb.opening(1);
        let rs2 = eb.opening(2);
        let g = eb.challenge(0);
        ClaimDefinition {
            expr: eb.build(rd_wv + g * rs1 + g * g * rs2),
            opening_bindings: vec![
                OpeningBinding {
                    var_id: 0,
                    polynomial: PolynomialId::RdWriteValue,
                },
                OpeningBinding {
                    var_id: 1,
                    polynomial: PolynomialId::Rs1Value,
                },
                OpeningBinding {
                    var_id: 2,
                    polynomial: PolynomialId::Rs2Value,
                },
            ],
            num_challenges: 1,
        }
    };

    let output = claims::registers::registers_read_write_checking();
    PolynomialIdentity {
        id: REGISTERS_RW,
        name: "registers_rw",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Predecessor(PredecessorClaim {
            formula: input_formula,
            challenge_labels: vec![ChallengeLabel::PreSqueeze("reg_gamma")],
            source_bindings: vec![],
        }),
        degree: 3,
        domain: DomainSpec::TraceTimesAddress,
    }
}

fn ram_val_check_identity(n_advice: usize) -> PolynomialIdentity {
    let input = if n_advice > 0 {
        let input_def = claims::ram::ram_val_check_input(n_advice);
        let mut labels = vec![
            ChallengeLabel::PreSqueeze("ram_gamma"),
            ChallengeLabel::External("neg_init"),
        ];
        for i in 0..n_advice {
            labels.push(ChallengeLabel::External(if i == 0 {
                "advice_sel_0"
            } else {
                "advice_sel_1"
            }));
        }
        IdentityClaim::Predecessor(PredecessorClaim {
            formula: input_def,
            challenge_labels: labels,
            source_bindings: vec![],
        })
    } else {
        IdentityClaim::Constant(0)
    };

    let output = claims::ram::ram_val_check();
    PolynomialIdentity {
        id: RAM_VAL_CHECK,
        name: "ram_val_check",
        produces: output.polynomials(),
        output,
        input,
        degree: 3,
        domain: DomainSpec::TraceLength,
    }
}

fn instr_read_raf_identity(config: &ProtocolConfig) -> PolynomialIdentity {
    let input_formula = {
        let eb = ExprBuilder::new();
        let rv = eb.opening(0);
        let left_op = eb.opening(1);
        let right_op = eb.opening(2);
        let g = eb.challenge(0);
        ClaimDefinition {
            expr: eb.build(rv + g * left_op + g * g * right_op),
            opening_bindings: vec![
                OpeningBinding {
                    var_id: 0,
                    polynomial: PolynomialId::LookupOutput,
                },
                OpeningBinding {
                    var_id: 1,
                    polynomial: PolynomialId::LeftLookupOperand,
                },
                OpeningBinding {
                    var_id: 2,
                    polynomial: PolynomialId::RightLookupOperand,
                },
            ],
            num_challenges: 1,
        }
    };

    let output = claims::instruction::instruction_ra_virtual(1, config.d_instr);
    let mut produces = output.polynomials();
    produces.push(PolynomialId::InstructionRafFlag);
    for i in 0..config.n_lookup_tables {
        produces.push(PolynomialId::LookupTableFlag(i));
    }

    PolynomialIdentity {
        id: INSTR_READ_RAF,
        name: "instr_read_raf",
        produces,
        output,
        input: IdentityClaim::Predecessor(PredecessorClaim {
            formula: input_formula,
            challenge_labels: vec![ChallengeLabel::PreSqueeze("instr_raf_gamma")],
            source_bindings: vec![],
        }),
        degree: config.d_instr + 2,
        domain: DomainSpec::TraceTimesAddress,
    }
}

fn ram_ra_cr_identity() -> PolynomialIdentity {
    let output = claims::reductions::ram_ra_claim_reduction();
    PolynomialIdentity {
        id: RAM_RA_CR,
        name: "ram_ra_cr",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Reduction {
            gamma_label: "ram_ra_gamma",
        },
        degree: 2,
        domain: DomainSpec::TraceLength,
    }
}

fn registers_val_eval_identity() -> PolynomialIdentity {
    let input_formula = {
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

    let output = claims::registers::registers_val_evaluation();
    PolynomialIdentity {
        id: REGISTERS_VAL_EVAL,
        name: "registers_val_eval",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Predecessor(PredecessorClaim {
            formula: input_formula,
            challenge_labels: vec![],
            source_bindings: vec![],
        }),
        degree: 3,
        domain: DomainSpec::TraceLength,
    }
}

fn bytecode_read_raf_identity(config: &ProtocolConfig) -> PolynomialIdentity {
    let (input_formula, source_bindings) = build_bytecode_raf_input_formula(config);
    let n_challenges = input_formula.num_challenges;
    let challenge_labels = (0..n_challenges)
        .map(|_| ChallengeLabel::PreSqueeze("bc_raf_gamma"))
        .collect();

    let output = claims::bytecode::bytecode_read_raf(5);
    PolynomialIdentity {
        id: BYTECODE_READ_RAF,
        name: "bytecode_read_raf",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Predecessor(PredecessorClaim {
            formula: input_formula,
            challenge_labels,
            source_bindings,
        }),
        degree: config.d_bc + 1,
        domain: DomainSpec::TraceTimesAddress,
    }
}

/// Build the BytecodeReadRaf two-level RLC input formula.
///
/// Formula: `Σ_{s=0}^{6} γ^s · (Σ_j β_s^j · poly_j) + γ^7`
///
/// The compound challenge for term (s, j) is `γ^s · β_s^j`.
fn build_bytecode_raf_input_formula(
    config: &ProtocolConfig,
) -> (ClaimDefinition, Vec<(u32, IdentityId)>) {
    // Collect (polynomial, source_identity) pairs in term order.
    let mut poly_terms: Vec<(PolynomialId, IdentityId)> = Vec::new();

    // Stage 1 (Spartan outer): UnexpandedPc + Imm + circuit flags
    poly_terms.push((PolynomialId::UnexpandedPc, SPARTAN_OUTER));
    poly_terms.push((PolynomialId::Imm, SPARTAN_OUTER));
    for i in 0..config.n_circuit_flags {
        poly_terms.push((circuit_flag_poly(i), SPARTAN_OUTER));
    }

    // Stage 2 (ProductVirtual): 4 terms
    for poly in [
        PolynomialId::JumpFlag,
        PolynomialId::BranchFlag,
        PolynomialId::WriteLookupToRdFlag,
        PolynomialId::NextIsVirtual,
    ] {
        poly_terms.push((poly, PRODUCT_VIRTUAL));
    }

    // Stage 3 (InstructionInput + Shift): 9 terms
    for poly in [
        PolynomialId::Imm,
        PolynomialId::UnexpandedPc,
        PolynomialId::LeftIsRs1,
        PolynomialId::LeftIsPc,
        PolynomialId::RightIsRs2,
        PolynomialId::RightIsImm,
        PolynomialId::NextIsNoop,
        PolynomialId::NextIsVirtual,
        PolynomialId::NextIsFirstInSequence,
    ] {
        poly_terms.push((poly, SHIFT));
    }

    // Stage 4 (RegistersRW): 3 terms
    for poly in [PolynomialId::RdWa, PolynomialId::Rs1Ra, PolynomialId::Rs2Ra] {
        poly_terms.push((poly, REGISTERS_RW));
    }

    // Stage 5: RdWa from RegistersValEval, InstrRafFlag + LookupTableFlags from InstrReadRaf
    poly_terms.push((PolynomialId::RdWa, REGISTERS_VAL_EVAL));
    poly_terms.push((PolynomialId::InstructionRafFlag, INSTR_READ_RAF));
    for i in 0..config.n_lookup_tables {
        poly_terms.push((PolynomialId::LookupTableFlag(i), INSTR_READ_RAF));
    }

    // RAF contributions: ExpandedPc from S1 (Spartan outer) and S3 (Shift)
    poly_terms.push((PolynomialId::ExpandedPc, SPARTAN_OUTER));
    poly_terms.push((PolynomialId::ExpandedPc, SHIFT));

    // Build formula: Σ challenge(i) * opening(i) + entry_constant
    let eb = ExprBuilder::new();
    let mut opening_bindings = Vec::with_capacity(poly_terms.len());
    let mut source_bindings = Vec::with_capacity(poly_terms.len());
    let mut expr = eb.zero();

    for (var_id, (poly_id, source)) in poly_terms.iter().enumerate() {
        let var_id = var_id as u32;
        let o = eb.opening(var_id);
        let c = eb.challenge(var_id);
        expr = expr + c * o;
        opening_bindings.push(OpeningBinding {
            var_id,
            polynomial: *poly_id,
        });
        source_bindings.push((var_id, *source));
    }

    // Entry constant term (γ^7, no opening)
    let entry_challenge = poly_terms.len() as u32;
    let entry_c = eb.challenge(entry_challenge);
    expr = expr + entry_c;

    let definition = ClaimDefinition {
        expr: eb.build(expr),
        opening_bindings,
        num_challenges: entry_challenge + 1,
    };

    (definition, source_bindings)
}

fn circuit_flag_poly(index: usize) -> PolynomialId {
    match index {
        5 => PolynomialId::JumpFlag,
        6 => PolynomialId::WriteLookupToRdFlag,
        7 => PolynomialId::NextIsVirtual,
        12 => PolynomialId::NextIsFirstInSequence,
        other => PolynomialId::OpFlag(other),
    }
}

fn booleanity_identity(config: &ProtocolConfig) -> PolynomialIdentity {
    let ra_ids = all_ra_poly_ids(config);
    let output = claims::booleanity::ra_booleanity(ra_ids.len(), &ra_ids);
    PolynomialIdentity {
        id: BOOLEANITY,
        name: "booleanity",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Zero,
        degree: 3,
        domain: DomainSpec::TraceTimesAddress,
    }
}

fn hamming_booleanity_identity() -> PolynomialIdentity {
    let output = claims::ram::hamming_booleanity();
    PolynomialIdentity {
        id: HAMMING_BOOLEANITY,
        name: "hamming_booleanity",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Zero,
        degree: 3,
        domain: DomainSpec::TraceLength,
    }
}

fn ram_ra_virtual_identity(config: &ProtocolConfig) -> PolynomialIdentity {
    let input_formula = {
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

    let output = claims::ram::ram_ra_virtual(config.d_ram);
    PolynomialIdentity {
        id: RAM_RA_VIRTUAL,
        name: "ram_ra_virtual",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Predecessor(PredecessorClaim {
            formula: input_formula,
            challenge_labels: vec![],
            source_bindings: vec![],
        }),
        degree: config.d_ram + 1,
        domain: DomainSpec::TraceTimesAddress,
    }
}

fn instr_ra_virtual_identity(config: &ProtocolConfig) -> PolynomialIdentity {
    let n_virtual = config.n_virtual_instr_ra();
    let cpv = config.d_instr_chunks_per_virtual;

    let input_formula = {
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

    let output = claims::instruction::instruction_ra_virtual(n_virtual, cpv);
    PolynomialIdentity {
        id: INSTR_RA_VIRTUAL,
        name: "instr_ra_virtual",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Predecessor(PredecessorClaim {
            formula: input_formula,
            challenge_labels: if n_virtual > 1 {
                vec![ChallengeLabel::PreSqueeze("instr_ra_gamma")]
            } else {
                vec![]
            },
            source_bindings: vec![],
        }),
        degree: cpv + 1,
        domain: DomainSpec::TraceTimesAddress,
    }
}

fn increment_cr_identity() -> PolynomialIdentity {
    let output = claims::reductions::increment_claim_reduction();
    PolynomialIdentity {
        id: INCREMENT_CR,
        name: "increment_cr",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Reduction {
            gamma_label: "inc_gamma",
        },
        degree: 2,
        domain: DomainSpec::TraceLength,
    }
}

fn hamming_weight_cr_identity(config: &ProtocolConfig) -> PolynomialIdentity {
    let ra_ids = all_ra_poly_ids(config);
    let output = claims::reductions::hamming_weight_claim_reduction(&ra_ids);
    PolynomialIdentity {
        id: HAMMING_WEIGHT_CR,
        name: "hamming_weight_cr",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Reduction {
            gamma_label: "hw_gamma",
        },
        degree: 2,
        domain: DomainSpec::AddressLength,
    }
}

fn advice_cr_identity() -> PolynomialIdentity {
    let output = claims::reductions::advice_claim_reduction_address();
    PolynomialIdentity {
        id: ADVICE_CR,
        name: "advice_cr",
        produces: output.polynomials(),
        output,
        input: IdentityClaim::Reduction {
            gamma_label: "advice_gamma",
        },
        degree: 2,
        domain: DomainSpec::AddressLength,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn all_ra_poly_ids(config: &ProtocolConfig) -> Vec<PolynomialId> {
    (0..config.d_instr)
        .map(PolynomialId::InstructionRa)
        .chain((0..config.d_bc).map(PolynomialId::BytecodeRa))
        .chain((0..config.d_ram).map(PolynomialId::RamRa))
        .collect()
}

/// Build a [`PolynomialRegistry`] matching `register_all_polynomials` in build.rs.
pub fn jolt_registry(config: &ProtocolConfig) -> super::compiler::PolynomialRegistry {
    use super::compiler::PolynomialRegistry;

    let log_t = || SymbolicExpr::symbol(Symbol::LOG_T);
    let log_ra = || SymbolicExpr::symbol(Symbol::LOG_T) + SymbolicExpr::symbol(Symbol::LOG_K);

    let mut reg = PolynomialRegistry::new();

    // Group 0: SpartanWitness
    reg.committed(
        PolynomialId::SpartanWitness,
        0,
        SymbolicExpr::symbol(Symbol::LOG_ROWS) + SymbolicExpr::symbol(Symbol::LOG_COLS),
    );

    // Group 1: RamInc, Group 2: RdInc
    reg.committed(PolynomialId::RamInc, 1, log_t());
    reg.committed(PolynomialId::RdInc, 2, log_t());

    // Groups 3..: RA polynomials
    let mut group_id = 3u32;
    for i in 0..config.d_instr {
        reg.committed(PolynomialId::InstructionRa(i), group_id, log_ra());
        group_id += 1;
    }
    for i in 0..config.d_bc {
        reg.committed(PolynomialId::BytecodeRa(i), group_id, log_ra());
        group_id += 1;
    }
    for i in 0..config.d_ram {
        reg.committed(PolynomialId::RamRa(i), group_id, log_ra());
        group_id += 1;
    }

    if config.n_advice >= 1 {
        reg.committed(PolynomialId::TrustedAdvice, group_id, log_t());
        group_id += 1;
    }
    if config.n_advice >= 2 {
        reg.committed(PolynomialId::UntrustedAdvice, group_id, log_t());
        let _ = group_id; // suppress unused warning
    }

    // R1CS virtual polynomials (Spartan outer/inner)
    let log_rows = || SymbolicExpr::symbol(Symbol::LOG_ROWS);
    let log_cols = || SymbolicExpr::symbol(Symbol::LOG_COLS);
    reg.virtual_poly(PolynomialId::Az, log_rows());
    reg.virtual_poly(PolynomialId::Bz, log_rows());
    reg.virtual_poly(PolynomialId::Cz, log_rows());
    reg.virtual_poly(PolynomialId::CombinedRow, log_cols());

    // Virtual polynomials
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
        reg.virtual_poly(id, log_t());
    }

    for i in [0, 1, 2, 3, 4, 8, 9, 10, 11, 13] {
        reg.virtual_poly(PolynomialId::OpFlag(i), log_t());
    }

    reg.virtual_poly(PolynomialId::ExpandedPc, log_t());
    reg.virtual_poly(PolynomialId::InstructionRafFlag, log_t());

    for i in 0..config.n_lookup_tables {
        reg.virtual_poly(PolynomialId::LookupTableFlag(i), log_t());
    }

    for s in 0..5 {
        reg.virtual_poly(PolynomialId::BytecodeReadRafVal(s), log_ra());
    }
    for s in 0..config.d_instr {
        reg.virtual_poly(PolynomialId::InstructionReadRafVal(s), log_ra());
    }

    reg
}

/// Challenge squeeze specifications per stage, matching `build_jolt_protocol`.
pub fn jolt_challenge_specs(
    config: &ProtocolConfig,
) -> Vec<(u32, Vec<super::types::ChallengeSpec>)> {
    use super::types::ChallengeSpec;

    let log_k = || SymbolicExpr::symbol(Symbol::LOG_K);

    let s1_sq: Vec<ChallengeSpec> = vec![];
    let s2_sq = vec![
        ChallengeSpec::Scalar {
            label: "pv_tau_high",
        },
        ChallengeSpec::Scalar {
            label: "ram_rw_gamma",
        },
        ChallengeSpec::Scalar {
            label: "instr_cr_gamma",
        },
        ChallengeSpec::Vector {
            label: "output_r_address",
            dim: log_k(),
        },
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
    let s5_sq = vec![
        ChallengeSpec::Scalar {
            label: "instr_raf_gamma",
        },
        ChallengeSpec::Scalar {
            label: "ram_ra_gamma",
        },
    ];
    let n_stage1_terms = SymbolicExpr::concrete(2 + config.n_circuit_flags);
    let n_stage5_terms = SymbolicExpr::concrete(2 + config.n_lookup_tables);
    let s6_sq = vec![
        ChallengeSpec::GammaPowers {
            label: "bc_raf_gamma",
            count: SymbolicExpr::concrete(8),
        },
        ChallengeSpec::GammaPowers {
            label: "bc_raf_stage1_gamma",
            count: n_stage1_terms,
        },
        ChallengeSpec::GammaPowers {
            label: "bc_raf_stage2_gamma",
            count: SymbolicExpr::concrete(4),
        },
        ChallengeSpec::GammaPowers {
            label: "bc_raf_stage3_gamma",
            count: SymbolicExpr::concrete(9),
        },
        ChallengeSpec::GammaPowers {
            label: "bc_raf_stage4_gamma",
            count: SymbolicExpr::concrete(3),
        },
        ChallengeSpec::GammaPowers {
            label: "bc_raf_stage5_gamma",
            count: n_stage5_terms,
        },
        ChallengeSpec::Scalar {
            label: "bool_gamma",
        },
        ChallengeSpec::Scalar {
            label: "instr_ra_gamma",
        },
        ChallengeSpec::Scalar { label: "inc_gamma" },
    ];
    let d_total = SymbolicExpr::concrete(config.d_total());
    let mut s7_sq = vec![ChallengeSpec::GammaPowers {
        label: "hw_gamma",
        count: d_total,
    }];
    if config.n_advice > 0 {
        s7_sq.push(ChallengeSpec::Scalar {
            label: "advice_gamma",
        });
    }

    vec![
        (0, s1_sq),
        (1, s2_sq),
        (2, s3_sq),
        (3, s4_sq),
        (4, s5_sq),
        (5, s6_sq),
        (6, s7_sq),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn jolt_identities_count() {
        let config = default_config();
        let ids = jolt_identities(&config);
        // Expected: 22 identities for default config (no advice, all chunks > 0)
        assert_eq!(
            ids.len(),
            22,
            "identities: {}",
            ids.iter().map(|i| i.name).collect::<Vec<_>>().join(", ")
        );
    }

    #[test]
    fn jolt_identities_unique_ids() {
        let config = default_config();
        let ids = jolt_identities(&config);
        let mut seen = std::collections::HashSet::new();
        for id in &ids {
            assert!(seen.insert(id.id), "duplicate identity id {:?}", id.id);
        }
    }

    #[test]
    fn jolt_hints_covers_all_identities() {
        let config = default_config();
        let ids = jolt_identities(&config);
        let hints = jolt_hints(&config);

        let assigned: std::collections::HashSet<IdentityId> =
            hints.stage_assignment.iter().map(|&(id, _)| id).collect();
        for ident in &ids {
            assert!(
                assigned.contains(&ident.id),
                "identity {:?} ({}) not in stage_assignment",
                ident.id,
                ident.name
            );
        }
    }

    #[test]
    fn jolt_hints_stages_monotonic() {
        let config = default_config();
        let hints = jolt_hints(&config);
        for &(_, stage) in &hints.stage_assignment {
            assert!(stage <= 6, "stage index out of range: {stage}");
        }
    }

    #[test]
    fn advice_config_adds_identity() {
        let mut config = default_config();
        let n_without = jolt_identities(&config).len();
        config.n_advice = 2;
        let n_with = jolt_identities(&config).len();
        assert_eq!(n_with, n_without + 1, "advice adds one identity");
    }

    #[test]
    fn bytecode_raf_input_term_count() {
        let config = default_config();
        let ids = jolt_identities(&config);
        let bc_raf = ids.iter().find(|i| i.name == "bytecode_read_raf").unwrap();

        if let IdentityClaim::Predecessor(pred) = &bc_raf.input {
            let n_openings = pred.formula.opening_bindings.len();
            // 2 + n_circuit_flags + 4 + 9 + 3 + (2 + n_lookup_tables) + 2 = 77
            let expected = 2 + config.n_circuit_flags + 4 + 9 + 3 + 2 + config.n_lookup_tables + 2;
            assert_eq!(n_openings, expected, "opening count mismatch");
            // +1 for entry constant term
            assert_eq!(
                pred.formula.num_challenges as usize,
                expected + 1,
                "challenge count mismatch"
            );
            assert_eq!(
                pred.source_bindings.len(),
                n_openings,
                "source binding count"
            );
        } else {
            panic!(
                "bytecode_read_raf input should be Predecessor, not {:?}",
                bc_raf.input
            );
        }
    }

    #[test]
    fn spartan_outer_produces_count() {
        let config = default_config();
        let ids = jolt_identities(&config);
        let outer = ids.iter().find(|i| i.name == "spartan_outer").unwrap();
        // Az + Bz + Cz + 33 named + 10 OpFlags + ExpandedPc + InstructionRafFlag + n_lookup_tables
        let expected = 3 + 33 + 10 + 1 + 1 + config.n_lookup_tables;
        assert_eq!(
            outer.produces.len(),
            expected,
            "spartan_outer produces count"
        );
    }

    #[test]
    fn spartan_inner_produces_count() {
        let config = default_config();
        let ids = jolt_identities(&config);
        let inner = ids.iter().find(|i| i.name == "spartan_inner").unwrap();
        // CombinedRow + SpartanWitness
        assert_eq!(inner.produces.len(), 2, "spartan_inner produces count");
    }

    #[test]
    fn zero_chunks_removes_identities() {
        let config = ProtocolConfig {
            d_instr: 0,
            d_bc: 0,
            d_ram: 0,
            d_instr_chunks_per_virtual: 1,
            n_lookup_tables: 0,
            n_circuit_flags: 14,
            n_advice: 0,
        };
        let ids = jolt_identities(&config);
        // Without instruction/bytecode/ram chunks, we lose:
        // INSTR_READ_RAF, BYTECODE_READ_RAF, BOOLEANITY, RAM_RA_VIRTUAL,
        // INSTR_RA_VIRTUAL, HAMMING_WEIGHT_CR = 6 fewer
        assert!(
            ids.len() < 23,
            "fewer identities without chunks: {}",
            ids.len()
        );
        // Verify no identity references missing polynomials
        for id in &ids {
            assert_ne!(id.name, "instr_read_raf");
            assert_ne!(id.name, "bytecode_read_raf");
            assert_ne!(id.name, "booleanity");
        }
    }
}
