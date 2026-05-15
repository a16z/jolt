#![expect(
    clippy::too_many_arguments,
    reason = "Stage 6/7 verifier relation helpers thread typed plan symbols and store/eval slices through their argument lists"
)]

//! Tier B: Audited Jolt verifier core.
//!
//! This module is **not** generic Bolt scaffolding. It is the hand-written
//! Jolt-specific verifier math that the Bolt compiler does not yet lower
//! from MLIR. It includes:
//!
//! - point normalizations for the Jolt bytecode and instruction RA
//!   read-RAF lookup arguments
//! - the `Stage67BytecodeEntry` contract a Jolt bytecode row must implement
//! - typed bytecode read-RAF plan data and its small Jolt-specific evaluator
//! - the small Jolt-specific field-math helpers
//!   (`operand_polynomial_eval`, `identity_polynomial_eval`,
//!   `bytecode_gamma_powers`) used only by Jolt verification
//!
//! Treat changes here as Jolt protocol changes, not as compiler-output
//! cleanups. Generic Bolt verifier scaffolding (typed plan structs,
//! `ValueStore`, generic sumcheck verification, generic field-expr
//! dispatch) lives in `bolt_verifier_runtime` instead.
//!
//! See `crates/bolt/GOAL.md` "Audit Tiers" for the full tier definition.

use jolt_field::{Field, Fr, MulPow2};
use jolt_lookup_tables::LookupTableKind;
use jolt_poly::EqPolynomial;

use bolt_verifier_runtime::{
    field_powers, prefix_point, store_point, store_scalar, suffix_point, NamedEvalFamilyPlan,
    NamedScalar, RuntimePlanError, SumcheckInstanceResultPlan, ValueStore,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[rustfmt::skip]
pub enum JoltRelationKind { Stage1OuterUniskip, Stage1OuterRemaining, Stage2ProductVirtualUniskip, Stage2RamReadWrite, Stage2ProductVirtualRemainder, Stage2InstructionLookupClaimReduction, Stage2RamRafEvaluation, Stage2RamOutputCheck, Stage2Batched, Stage3SpartanShift, Stage3InstructionInput, Stage3RegistersClaimReduction, Stage3Batched, Stage4RegistersReadWrite, Stage4RamValCheck, Stage4Batched, Stage5InstructionReadRaf, Stage5RamRaClaimReduction, Stage5RegistersValEvaluation, Stage5Batched, Stage6BytecodeReadRaf, Stage6Booleanity, Stage6HammingBooleanity, Stage6RamRaVirtual, Stage6InstructionRaVirtual, Stage6IncClaimReduction, Stage6Batched, Stage7HammingWeightClaimReduction, Stage7Batched }

pub fn bytecode_gamma_powers(gamma: Fr) -> [Fr; 8] {
    let mut powers = [Fr::from_u64(1); 8];
    for index in 1..powers.len() {
        powers[index] = powers[index - 1] * gamma;
    }
    powers
}

pub fn normalize_bytecode_read_raf_point<F: Field>(
    point: &[F],
    log_t: usize,
    input: &'static str,
) -> Result<Vec<F>, RuntimePlanError> {
    let log_k = point
        .len()
        .checked_sub(log_t)
        .ok_or(RuntimePlanError::InvalidInputLength {
            input,
            expected: log_t,
            actual: point.len(),
        })?;
    let mut normalized = point.to_vec();
    normalized[..log_k].reverse();
    normalized[log_k..].reverse();
    Ok(normalized)
}

pub fn normalize_instruction_read_raf_point<F: Field>(
    point: &[F],
    input: &'static str,
) -> Result<Vec<F>, RuntimePlanError> {
    const LOG_K: usize = 128;
    if point.len() < LOG_K {
        return Err(RuntimePlanError::InvalidInputLength {
            input,
            expected: LOG_K,
            actual: point.len(),
        });
    }
    let mut normalized = point.to_vec();
    normalized[LOG_K..].reverse();
    Ok(normalized)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage67RelationSymbols {
    pub hamming_booleanity_instance: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5InstructionReadRafPlan {
    pub point: &'static str,
    pub lookup_output_point: &'static str,
    pub table_flag_evals: &'static NamedEvalFamilyPlan,
    pub instruction_ra_evals: &'static NamedEvalFamilyPlan,
    pub raf_flag_eval: &'static str,
    pub gamma: &'static str,
    pub point_values: &'static [Stage5InstructionReadRafPointValuePlan],
    pub log_k: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5InstructionReadRafPointValuePlan {
    pub symbol: &'static str,
    pub kind: Stage5InstructionReadRafPointValueKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage5InstructionReadRafPointValueKind {
    LookupTable { index: usize },
    LeftOperand,
    RightOperand,
    Identity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage67BytecodeReadRafPlan {
    pub point: &'static str,
    pub gamma: &'static str,
    pub bytecode_ra_evals: &'static NamedEvalFamilyPlan,
    pub entries: &'static str,
    pub entry_bytecode_index: &'static str,
    pub stages: &'static [Stage67BytecodeStagePlan],
    pub output_terms: &'static [Stage67BytecodeOutputTermPlan],
    pub output_contribution: &'static str,
    pub registers: Stage67BytecodeRegisterSymbols,
    pub entry_lookup_table: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage67BytecodeStagePlan {
    pub gamma: &'static str,
    pub cycle_point: &'static str,
    pub register_point: Option<&'static str>,
    pub terms: &'static [Stage67BytecodeTermPlan],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage67BytecodeOutputTermPlan {
    StageValue {
        stage_index: usize,
        gamma_power: usize,
        identity_gamma_power: Option<usize>,
    },
    Entry {
        gamma_power: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage67BytecodeRegisterSymbols {
    pub rd: &'static str,
    pub rs1: &'static str,
    pub rs2: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage67BytecodeTermPlan {
    Address {
        gamma_power: usize,
    },
    Imm {
        gamma_power: usize,
    },
    CircuitFlag {
        index: usize,
        gamma_power: usize,
    },
    EntryFlag {
        flag: Stage67BytecodeFlag,
        expected: bool,
        gamma_power: usize,
    },
    RegisterEq {
        register: Stage67BytecodeRegister,
        gamma_power: usize,
    },
    LookupTable {
        gamma_base: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage67BytecodeFlag {
    IsInterleaved,
    IsBranch,
    LeftIsRs1,
    LeftIsPc,
    RightIsRs2,
    RightIsImm,
    IsNoop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage67BytecodeRegister {
    Rd,
    Rs1,
    Rs2,
}

pub trait Stage67BytecodeEntry {
    fn address(&self) -> Fr;
    fn imm(&self) -> Fr;
    fn circuit_flags(&self) -> &[bool; 14];
    fn rd(&self) -> Option<usize>;
    fn rs1(&self) -> Option<usize>;
    fn rs2(&self) -> Option<usize>;
    fn lookup_table(&self) -> Option<usize>;
    fn is_interleaved(&self) -> bool;
    fn is_branch(&self) -> bool;
    fn left_is_rs1(&self) -> bool;
    fn left_is_pc(&self) -> bool;
    fn right_is_rs2(&self) -> bool;
    fn right_is_imm(&self) -> bool;
    fn is_noop(&self) -> bool;
}

pub fn evaluate_stage5_instruction_read_raf_point_scalars(
    plan: &Stage5InstructionReadRafPlan,
    local_point: &[Fr],
) -> Result<Vec<NamedScalar<Fr>>, RuntimePlanError> {
    let (r_address_prime, _) = instruction_read_raf_point_parts(plan, local_point)?;
    let mut scalars = Vec::with_capacity(plan.point_values.len());
    for value in plan.point_values {
        let scalar = evaluate_stage5_instruction_read_raf_point_value(value.kind, r_address_prime)?;
        scalars.push(NamedScalar {
            symbol: value.symbol,
            value: scalar,
        });
    }
    Ok(scalars)
}

fn evaluate_stage5_instruction_read_raf_point_value(
    kind: Stage5InstructionReadRafPointValueKind,
    r_address_prime: &[Fr],
) -> Result<Fr, RuntimePlanError> {
    const XLEN: usize = 64;
    Ok(match kind {
        Stage5InstructionReadRafPointValueKind::LookupTable { index } => {
            let tables = LookupTableKind::<XLEN>::all();
            let table = tables
                .get(index)
                .ok_or(RuntimePlanError::InvalidInputLength {
                    input: "stage5.instruction_read_raf.lookup_table",
                    expected: tables.len(),
                    actual: index + 1,
                })?;
            table.evaluate_mle::<Fr, Fr>(r_address_prime)
        }
        Stage5InstructionReadRafPointValueKind::LeftOperand => {
            operand_polynomial_eval(r_address_prime, true)
        }
        Stage5InstructionReadRafPointValueKind::RightOperand => {
            operand_polynomial_eval(r_address_prime, false)
        }
        Stage5InstructionReadRafPointValueKind::Identity => {
            identity_polynomial_eval(r_address_prime)
        }
    })
}

fn instruction_read_raf_point_parts<'a>(
    plan: &Stage5InstructionReadRafPlan,
    local_point: &'a [Fr],
) -> Result<(&'a [Fr], &'a [Fr]), RuntimePlanError> {
    if local_point.len() < plan.log_k {
        return Err(RuntimePlanError::InvalidInputLength {
            input: plan.point,
            expected: plan.log_k,
            actual: local_point.len(),
        });
    }
    Ok(local_point.split_at(plan.log_k))
}

pub fn stage67_trace_rounds(
    instance_results: &[SumcheckInstanceResultPlan<JoltRelationKind>],
    symbols: &Stage67RelationSymbols,
) -> Result<usize, RuntimePlanError> {
    instance_results
        .iter()
        .find(|instance| instance.relation == JoltRelationKind::Stage6HammingBooleanity)
        .map(|instance| instance.num_rounds)
        .ok_or(RuntimePlanError::MissingValue {
            symbol: symbols.hamming_booleanity_instance,
        })
}

pub fn evaluate_stage67_bytecode_read_raf_output_scalars<E: Stage67BytecodeEntry>(
    plan: &Stage67BytecodeReadRafPlan,
    entries: &[E],
    entry_bytecode_index: usize,
    num_lookup_tables: usize,
    store: &ValueStore<Fr>,
    local_point: &[Fr],
    log_t: usize,
) -> Result<Vec<NamedScalar<Fr>>, RuntimePlanError> {
    let output = stage67_bytecode_read_raf_output_contribution(
        plan,
        entries,
        entry_bytecode_index,
        num_lookup_tables,
        store,
        local_point,
        log_t,
    )?;
    Ok(vec![NamedScalar {
        symbol: plan.output_contribution,
        value: output,
    }])
}

fn stage67_bytecode_read_raf_output_contribution<E: Stage67BytecodeEntry>(
    plan: &Stage67BytecodeReadRafPlan,
    entries: &[E],
    entry_bytecode_index: usize,
    num_lookup_tables: usize,
    store: &ValueStore<Fr>,
    local_point: &[Fr],
    log_t: usize,
) -> Result<Fr, RuntimePlanError> {
    let opening_point = normalize_bytecode_read_raf_point(local_point, log_t, plan.point)?;
    let log_k = opening_point.len() - log_t;
    let (r_address_prime, r_cycle_prime) = opening_point.split_at(log_k);

    let gamma = store_scalar(store, plan.gamma)?;
    let gamma_powers = bytecode_gamma_powers(gamma);
    let stage_value_evals = stage67_bytecode_stage_value_evals(
        plan,
        entries,
        entry_bytecode_index,
        num_lookup_tables,
        store,
        r_address_prime,
        r_cycle_prime.len(),
    )?;
    let output_contrib = stage67_bytecode_output_contribution(
        plan,
        store,
        &stage_value_evals,
        &gamma_powers,
        entry_bytecode_index,
        r_address_prime,
        r_cycle_prime,
        log_k,
    )?;
    Ok(output_contrib)
}

fn stage67_bytecode_output_contribution(
    plan: &Stage67BytecodeReadRafPlan,
    store: &ValueStore<Fr>,
    stage_value_evals: &[Fr],
    gamma_powers: &[Fr],
    entry_bytecode_index: usize,
    r_address_prime: &[Fr],
    r_cycle_prime: &[Fr],
    log_k: usize,
) -> Result<Fr, RuntimePlanError> {
    let int_eval = identity_polynomial_eval(r_address_prime);
    let zero_cycle = vec![Fr::from_u64(0); r_cycle_prime.len()];
    let entry_address_eq =
        EqPolynomial::<Fr>::try_mle_at_boolean_index(entry_bytecode_index, r_address_prime).ok_or(
            RuntimePlanError::InvalidInputLength {
                input: plan.entry_bytecode_index,
                expected: 1usize.checked_shl(log_k as u32).unwrap_or(usize::MAX),
                actual: entry_bytecode_index.saturating_add(1),
            },
        )?;

    let mut output = Fr::from_u64(0);
    for term in plan.output_terms {
        output += match *term {
            Stage67BytecodeOutputTermPlan::StageValue {
                stage_index,
                gamma_power,
                identity_gamma_power,
            } => {
                let stage =
                    plan.stages
                        .get(stage_index)
                        .ok_or(RuntimePlanError::InvalidInputLength {
                            input: plan.entries,
                            expected: stage_index + 1,
                            actual: plan.stages.len(),
                        })?;
                let value = stage_value_evals.get(stage_index).copied().ok_or(
                    RuntimePlanError::InvalidInputLength {
                        input: plan.entries,
                        expected: stage_index + 1,
                        actual: stage_value_evals.len(),
                    },
                )?;
                let cycle_point =
                    stage67_bytecode_stage_cycle_point(store, stage, r_cycle_prime.len())?;
                let identity_contrib = identity_gamma_power
                    .map_or(Fr::from_u64(0), |power| gamma_powers[power] * int_eval);
                (value + identity_contrib)
                    * EqPolynomial::<Fr>::mle(&cycle_point, r_cycle_prime)
                    * gamma_powers[gamma_power]
            }
            Stage67BytecodeOutputTermPlan::Entry { gamma_power } => {
                gamma_powers[gamma_power]
                    * entry_address_eq
                    * EqPolynomial::<Fr>::mle(&zero_cycle, r_cycle_prime)
            }
        };
    }
    Ok(output)
}

fn stage67_bytecode_stage_cycle_point(
    store: &ValueStore<Fr>,
    stage: &Stage67BytecodeStagePlan,
    log_t: usize,
) -> Result<Vec<Fr>, RuntimePlanError> {
    suffix_point(
        store_point(store, stage.cycle_point)?,
        log_t,
        stage.cycle_point,
    )
    .map(|point| point.to_vec())
}

fn stage67_bytecode_stage_value_evals<E: Stage67BytecodeEntry>(
    plan: &Stage67BytecodeReadRafPlan,
    entries: &[E],
    entry_bytecode_index: usize,
    num_lookup_tables: usize,
    store: &ValueStore<Fr>,
    r_address: &[Fr],
    log_t: usize,
) -> Result<Vec<Fr>, RuntimePlanError> {
    let expected_len =
        1usize
            .checked_shl(r_address.len() as u32)
            .ok_or(RuntimePlanError::InvalidInputLength {
                input: plan.entries,
                expected: usize::BITS as usize,
                actual: r_address.len(),
            })?;
    if entries.len() != expected_len {
        return Err(RuntimePlanError::InvalidInputLength {
            input: plan.entries,
            expected: expected_len,
            actual: entries.len(),
        });
    }
    if entry_bytecode_index >= expected_len {
        return Err(RuntimePlanError::InvalidInputLength {
            input: plan.entry_bytecode_index,
            expected: expected_len,
            actual: entry_bytecode_index + 1,
        });
    }

    let stage_contexts = plan
        .stages
        .iter()
        .map(|stage| {
            Ok(Stage67BytecodeStageContext {
                plan: stage,
                gamma_powers: field_powers(
                    store_scalar(store, stage.gamma)?,
                    stage67_bytecode_stage_gamma_power_count(stage, num_lookup_tables),
                ),
                register_point: match stage.register_point {
                    Some(symbol) => Some(stage67_register_prefix_point(store, symbol, log_t)?),
                    None => None,
                },
            })
        })
        .collect::<Result<Vec<_>, RuntimePlanError>>()?;

    let mut evals = vec![Fr::from_u64(0); plan.stages.len()];
    for (index, entry) in entries.iter().enumerate() {
        let eq = EqPolynomial::<Fr>::try_mle_at_boolean_index(index, r_address).ok_or(
            RuntimePlanError::InvalidInputLength {
                input: plan.entries,
                expected: expected_len,
                actual: index + 1,
            },
        )?;
        let values = stage67_bytecode_entry_stage_values(plan, entry, &stage_contexts)?;
        for stage in 0..evals.len() {
            evals[stage] += eq * values[stage];
        }
    }
    Ok(evals)
}

struct Stage67BytecodeStageContext<'a> {
    plan: &'a Stage67BytecodeStagePlan,
    gamma_powers: Vec<Fr>,
    register_point: Option<&'a [Fr]>,
}

fn stage67_bytecode_entry_stage_values<E: Stage67BytecodeEntry>(
    plan: &Stage67BytecodeReadRafPlan,
    entry: &E,
    stage_contexts: &[Stage67BytecodeStageContext<'_>],
) -> Result<Vec<Fr>, RuntimePlanError> {
    stage_contexts
        .iter()
        .map(|context| stage67_bytecode_entry_stage_value(plan, entry, context))
        .collect()
}

fn stage67_bytecode_entry_stage_value<E: Stage67BytecodeEntry>(
    plan: &Stage67BytecodeReadRafPlan,
    entry: &E,
    context: &Stage67BytecodeStageContext<'_>,
) -> Result<Fr, RuntimePlanError> {
    let mut value = Fr::from_u64(0);
    for term in context.plan.terms {
        value += match *term {
            Stage67BytecodeTermPlan::Address { gamma_power } => {
                entry.address() * context.gamma_powers[gamma_power]
            }
            Stage67BytecodeTermPlan::Imm { gamma_power } => {
                entry.imm() * context.gamma_powers[gamma_power]
            }
            Stage67BytecodeTermPlan::CircuitFlag { index, gamma_power } => {
                if entry.circuit_flags()[index] {
                    context.gamma_powers[gamma_power]
                } else {
                    Fr::from_u64(0)
                }
            }
            Stage67BytecodeTermPlan::EntryFlag {
                flag,
                expected,
                gamma_power,
            } => {
                if stage67_bytecode_entry_flag(entry, flag) == expected {
                    context.gamma_powers[gamma_power]
                } else {
                    Fr::from_u64(0)
                }
            }
            Stage67BytecodeTermPlan::RegisterEq {
                register,
                gamma_power,
            } => {
                let register_point =
                    context
                        .register_point
                        .ok_or(RuntimePlanError::MissingValue {
                            symbol: context.plan.cycle_point,
                        })?;
                stage67_register_eq(
                    stage67_bytecode_entry_register(entry, register),
                    register_point,
                    stage67_bytecode_register_symbol(plan, register),
                )? * context.gamma_powers[gamma_power]
            }
            Stage67BytecodeTermPlan::LookupTable { gamma_base } => {
                let Some(table) = entry.lookup_table() else {
                    continue;
                };
                if table >= plan_lookup_table_count(context, gamma_base) {
                    return Err(RuntimePlanError::InvalidInputLength {
                        input: plan.entry_lookup_table,
                        expected: plan_lookup_table_count(context, gamma_base),
                        actual: table + 1,
                    });
                }
                context.gamma_powers[gamma_base + table]
            }
        };
    }
    Ok(value)
}

fn stage67_bytecode_stage_gamma_power_count(
    stage: &Stage67BytecodeStagePlan,
    num_lookup_tables: usize,
) -> usize {
    stage
        .terms
        .iter()
        .map(|term| match *term {
            Stage67BytecodeTermPlan::Address { gamma_power }
            | Stage67BytecodeTermPlan::Imm { gamma_power }
            | Stage67BytecodeTermPlan::CircuitFlag { gamma_power, .. }
            | Stage67BytecodeTermPlan::EntryFlag { gamma_power, .. }
            | Stage67BytecodeTermPlan::RegisterEq { gamma_power, .. } => gamma_power + 1,
            Stage67BytecodeTermPlan::LookupTable { gamma_base } => gamma_base + num_lookup_tables,
        })
        .max()
        .unwrap_or(0)
}

fn plan_lookup_table_count(context: &Stage67BytecodeStageContext<'_>, gamma_base: usize) -> usize {
    context.gamma_powers.len().saturating_sub(gamma_base)
}

fn stage67_bytecode_entry_flag<E: Stage67BytecodeEntry>(
    entry: &E,
    flag: Stage67BytecodeFlag,
) -> bool {
    match flag {
        Stage67BytecodeFlag::IsInterleaved => entry.is_interleaved(),
        Stage67BytecodeFlag::IsBranch => entry.is_branch(),
        Stage67BytecodeFlag::LeftIsRs1 => entry.left_is_rs1(),
        Stage67BytecodeFlag::LeftIsPc => entry.left_is_pc(),
        Stage67BytecodeFlag::RightIsRs2 => entry.right_is_rs2(),
        Stage67BytecodeFlag::RightIsImm => entry.right_is_imm(),
        Stage67BytecodeFlag::IsNoop => entry.is_noop(),
    }
}

fn stage67_bytecode_entry_register<E: Stage67BytecodeEntry>(
    entry: &E,
    register: Stage67BytecodeRegister,
) -> Option<usize> {
    match register {
        Stage67BytecodeRegister::Rd => entry.rd(),
        Stage67BytecodeRegister::Rs1 => entry.rs1(),
        Stage67BytecodeRegister::Rs2 => entry.rs2(),
    }
}

fn stage67_bytecode_register_symbol(
    plan: &Stage67BytecodeReadRafPlan,
    register: Stage67BytecodeRegister,
) -> &'static str {
    match register {
        Stage67BytecodeRegister::Rd => plan.registers.rd,
        Stage67BytecodeRegister::Rs1 => plan.registers.rs1,
        Stage67BytecodeRegister::Rs2 => plan.registers.rs2,
    }
}

fn stage67_register_eq(
    index: Option<usize>,
    point: &[Fr],
    input: &'static str,
) -> Result<Fr, RuntimePlanError> {
    let Some(index) = index else {
        return Ok(Fr::from_u64(0));
    };
    let register_count =
        1usize
            .checked_shl(point.len() as u32)
            .ok_or(RuntimePlanError::InvalidInputLength {
                input,
                expected: usize::BITS as usize,
                actual: point.len(),
            })?;
    if index >= register_count {
        return Err(RuntimePlanError::InvalidInputLength {
            input,
            expected: register_count,
            actual: index + 1,
        });
    }
    EqPolynomial::<Fr>::try_mle_at_boolean_index(index, point).ok_or(
        RuntimePlanError::InvalidInputLength {
            input,
            expected: register_count,
            actual: index + 1,
        },
    )
}

fn stage67_register_prefix_point<'a>(
    store: &'a ValueStore<Fr>,
    symbol: &'static str,
    log_t: usize,
) -> Result<&'a [Fr], RuntimePlanError> {
    let point = store_point(store, symbol)?;
    let register_len =
        point
            .len()
            .checked_sub(log_t)
            .ok_or(RuntimePlanError::InvalidInputLength {
                input: symbol,
                expected: log_t,
                actual: point.len(),
            })?;
    prefix_point(point, register_len, symbol)
}

pub fn operand_polynomial_eval(point: &[Fr], left: bool) -> Fr {
    let stride_offset = usize::from(!left);
    let operand_bits = point.len() / 2;
    (0..operand_bits)
        .map(|index| point[2 * index + stride_offset].mul_pow_2(operand_bits - 1 - index))
        .sum()
}

pub fn identity_polynomial_eval(point: &[Fr]) -> Fr {
    point
        .iter()
        .enumerate()
        .map(|(index, value)| value.mul_pow_2(point.len() - 1 - index))
        .sum()
}
