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
//! - the remaining `expected_stage67_*` relation evaluators for Stage 6
//!   booleanity and bytecode-read-RAF relations
//! - the small Jolt-specific field-math helpers
//!   (`operand_polynomial_eval`, `identity_polynomial_eval`,
//!   `lt_polynomial_eval`, `bytecode_gamma_powers`) used only by Jolt
//!   verification
//!
//! Treat changes here as Jolt protocol changes, not as compiler-output
//! cleanups. Generic Bolt verifier scaffolding (typed plan structs,
//! `ValueStore`, generic sumcheck verification, generic field-expr
//! dispatch) lives in `bolt_verifier_runtime` instead.
//!
//! See `crates/bolt/GOAL.md` "Audit Tiers" for the full tier definition.

use jolt_field::{Field, Fr, MulPow2, RingCore};
use jolt_poly::EqPolynomial;

use bolt_verifier_runtime::{
    field_powers, indexed_boolean_eq, indexed_evals_by_prefix_any, prefix_point, store_point,
    store_scalar, suffix_point, RuntimePlanError, StageNamedEval, SumcheckInstanceResultPlan,
    ValueStore,
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
    pub booleanity_point: &'static str,
    pub stage5_instruction_ra0: &'static str,
    pub booleanity_combined_point: &'static str,
    pub booleanity_gamma: &'static str,
    pub booleanity_instruction_ra_prefix: &'static str,
    pub booleanity_bytecode_ra_prefix: &'static str,
    pub booleanity_ram_ra_prefix: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage67BytecodeSymbols {
    pub point: &'static str,
    pub gamma: &'static str,
    pub bytecode_ra_eval_prefix: &'static str,
    pub entries: &'static str,
    pub entry_bytecode_index: &'static str,
    pub stage_gammas: [&'static str; 5],
    pub stage_cycle_points: [&'static str; 5],
    pub stage4_register_point: &'static str,
    pub stage5_register_point: &'static str,
    pub entry_rd: &'static str,
    pub entry_rs1: &'static str,
    pub entry_rs2: &'static str,
    pub entry_lookup_table: &'static str,
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

pub fn expected_stage67_bytecode_read_raf<E: Stage67BytecodeEntry>(
    entries: &[E],
    entry_bytecode_index: usize,
    num_lookup_tables: usize,
    store: &ValueStore<Fr>,
    evals: &[StageNamedEval<Fr>],
    local_point: &[Fr],
    log_t: usize,
    symbols: &Stage67BytecodeSymbols,
) -> Result<Fr, RuntimePlanError> {
    let opening_point = normalize_bytecode_read_raf_point(local_point, log_t, symbols.point)?;
    let log_k = opening_point.len() - log_t;
    let (r_address_prime, r_cycle_prime) = opening_point.split_at(log_k);

    let gamma = store_scalar(store, symbols.gamma)?;
    let gamma_powers = bytecode_gamma_powers(gamma);
    let int_eval = identity_polynomial_eval(r_address_prime);
    let stage_value_evals = stage67_bytecode_stage_value_evals(
        entries,
        entry_bytecode_index,
        num_lookup_tables,
        store,
        r_address_prime,
        r_cycle_prime.len(),
        symbols,
    )?;
    let stage_cycle_points =
        stage67_bytecode_stage_cycle_points(store, r_cycle_prime.len(), symbols)?;
    let int_contrib = [
        gamma_powers[5] * int_eval,
        Fr::from_u64(0),
        gamma_powers[4] * int_eval,
        Fr::from_u64(0),
        Fr::from_u64(0),
    ];

    let mut val = Fr::from_u64(0);
    for index in 0..stage_value_evals.len() {
        val += (stage_value_evals[index] + int_contrib[index])
            * EqPolynomial::<Fr>::mle(&stage_cycle_points[index], r_cycle_prime)
            * gamma_powers[index];
    }

    let entry_bits = (0..log_k)
        .map(|index| Fr::from_u64(((entry_bytecode_index >> (log_k - 1 - index)) & 1) as u64))
        .collect::<Vec<_>>();
    let zero_cycle = vec![Fr::from_u64(0); r_cycle_prime.len()];
    let entry_contrib = gamma_powers[7]
        * EqPolynomial::<Fr>::mle(&entry_bits, r_address_prime)
        * EqPolynomial::<Fr>::mle(&zero_cycle, r_cycle_prime);
    let bytecode_ra = indexed_evals_by_prefix_any(evals, symbols.bytecode_ra_eval_prefix)?
        .into_iter()
        .product::<Fr>();
    Ok((val + entry_contrib) * bytecode_ra)
}

pub fn expected_stage67_booleanity(
    store: &ValueStore<Fr>,
    evals: &[StageNamedEval<Fr>],
    local_point: &[Fr],
    log_t: usize,
    symbols: &Stage67RelationSymbols,
) -> Result<Fr, RuntimePlanError> {
    let log_k_chunk =
        local_point
            .len()
            .checked_sub(log_t)
            .ok_or(RuntimePlanError::InvalidInputLength {
                input: symbols.booleanity_point,
                expected: log_t,
                actual: local_point.len(),
            })?;
    let stage5_point = store_point(store, symbols.stage5_instruction_ra0)?;
    let stage5_address_len =
        stage5_point
            .len()
            .checked_sub(log_t)
            .ok_or(RuntimePlanError::InvalidInputLength {
                input: symbols.stage5_instruction_ra0,
                expected: log_t,
                actual: stage5_point.len(),
            })?;
    if stage5_address_len < log_k_chunk {
        return Err(RuntimePlanError::InvalidInputLength {
            input: symbols.stage5_instruction_ra0,
            expected: log_k_chunk + log_t,
            actual: stage5_point.len(),
        });
    }

    let mut stage5_addr = stage5_point[..stage5_address_len].to_vec();
    stage5_addr.reverse();
    let mut combined_r = stage5_addr[stage5_address_len - log_k_chunk..].to_vec();
    combined_r.extend(stage5_point[stage5_address_len..].iter().rev().copied());
    if combined_r.len() != local_point.len() {
        return Err(RuntimePlanError::InvalidInputLength {
            input: symbols.booleanity_combined_point,
            expected: local_point.len(),
            actual: combined_r.len(),
        });
    }
    let mut verifier_point = combined_r[..log_k_chunk].to_vec();
    verifier_point.reverse();
    verifier_point.extend(combined_r[log_k_chunk..].iter().rev().copied());
    let eq_eval = EqPolynomial::<Fr>::mle(local_point, &verifier_point);

    let gamma = store_scalar(store, symbols.booleanity_gamma)?;
    let gamma_sq = gamma.square();
    let mut gamma_power = Fr::from_u64(1);
    let mut booleanity = Fr::from_u64(0);
    for ra in stage67_booleanity_evals(evals, symbols)? {
        booleanity += gamma_power * (ra.square() - ra);
        gamma_power *= gamma_sq;
    }
    Ok(eq_eval * booleanity)
}

fn stage67_booleanity_evals(
    evals: &[StageNamedEval<Fr>],
    symbols: &Stage67RelationSymbols,
) -> Result<Vec<Fr>, RuntimePlanError> {
    let mut values = indexed_evals_by_prefix_any(evals, symbols.booleanity_instruction_ra_prefix)?;
    values.extend(indexed_evals_by_prefix_any(
        evals,
        symbols.booleanity_bytecode_ra_prefix,
    )?);
    values.extend(indexed_evals_by_prefix_any(
        evals,
        symbols.booleanity_ram_ra_prefix,
    )?);
    Ok(values)
}

fn stage67_bytecode_stage_cycle_points(
    store: &ValueStore<Fr>,
    log_t: usize,
    symbols: &Stage67BytecodeSymbols,
) -> Result<[Vec<Fr>; 5], RuntimePlanError> {
    let point = |index| {
        let symbol = symbols.stage_cycle_points[index];
        suffix_point(store_point(store, symbol)?, log_t, symbol).map(|point| point.to_vec())
    };
    Ok([point(0)?, point(1)?, point(2)?, point(3)?, point(4)?])
}

fn stage67_bytecode_stage_value_evals<E: Stage67BytecodeEntry>(
    entries: &[E],
    entry_bytecode_index: usize,
    num_lookup_tables: usize,
    store: &ValueStore<Fr>,
    r_address: &[Fr],
    log_t: usize,
    symbols: &Stage67BytecodeSymbols,
) -> Result<[Fr; 5], RuntimePlanError> {
    let expected_len =
        1usize
            .checked_shl(r_address.len() as u32)
            .ok_or(RuntimePlanError::InvalidInputLength {
                input: symbols.entries,
                expected: usize::BITS as usize,
                actual: r_address.len(),
            })?;
    if entries.len() != expected_len {
        return Err(RuntimePlanError::InvalidInputLength {
            input: symbols.entries,
            expected: expected_len,
            actual: entries.len(),
        });
    }
    if entry_bytecode_index >= expected_len {
        return Err(RuntimePlanError::InvalidInputLength {
            input: symbols.entry_bytecode_index,
            expected: expected_len,
            actual: entry_bytecode_index + 1,
        });
    }

    let stage1_gamma_powers = field_powers(store_scalar(store, symbols.stage_gammas[0])?, 16);
    let stage2_gamma_powers = field_powers(store_scalar(store, symbols.stage_gammas[1])?, 4);
    let stage3_gamma_powers = field_powers(store_scalar(store, symbols.stage_gammas[2])?, 9);
    let stage4_gamma_powers = field_powers(store_scalar(store, symbols.stage_gammas[3])?, 3);
    let stage5_gamma_powers = field_powers(
        store_scalar(store, symbols.stage_gammas[4])?,
        num_lookup_tables + 2,
    );

    let stage4_register_point =
        stage67_register_prefix_point(store, symbols.stage4_register_point, log_t)?;
    let stage5_register_point =
        stage67_register_prefix_point(store, symbols.stage5_register_point, log_t)?;

    let mut evals = [Fr::from_u64(0); 5];
    for (index, entry) in entries.iter().enumerate() {
        let eq = indexed_boolean_eq(index, r_address);
        let values = stage67_bytecode_entry_stage_values(
            entry,
            num_lookup_tables,
            stage4_register_point,
            stage5_register_point,
            &stage1_gamma_powers,
            &stage2_gamma_powers,
            &stage3_gamma_powers,
            &stage4_gamma_powers,
            &stage5_gamma_powers,
            symbols,
        )?;
        for stage in 0..evals.len() {
            evals[stage] += eq * values[stage];
        }
    }
    Ok(evals)
}

fn stage67_bytecode_entry_stage_values<E: Stage67BytecodeEntry>(
    entry: &E,
    num_lookup_tables: usize,
    stage4_register_point: &[Fr],
    stage5_register_point: &[Fr],
    stage1_gamma_powers: &[Fr],
    stage2_gamma_powers: &[Fr],
    stage3_gamma_powers: &[Fr],
    stage4_gamma_powers: &[Fr],
    stage5_gamma_powers: &[Fr],
    symbols: &Stage67BytecodeSymbols,
) -> Result<[Fr; 5], RuntimePlanError> {
    let flags = entry.circuit_flags();
    let mut stage1 = entry.address() + entry.imm() * stage1_gamma_powers[1];
    for (flag, gamma) in flags.iter().zip(stage1_gamma_powers.iter().skip(2)) {
        if *flag {
            stage1 += *gamma;
        }
    }

    let mut stage2 = Fr::from_u64(0);
    if flags[5] {
        stage2 += stage2_gamma_powers[0];
    }
    if entry.is_branch() {
        stage2 += stage2_gamma_powers[1];
    }
    if flags[6] {
        stage2 += stage2_gamma_powers[2];
    }
    if flags[7] {
        stage2 += stage2_gamma_powers[3];
    }

    let mut stage3 = entry.imm() + entry.address() * stage3_gamma_powers[1];
    if entry.left_is_rs1() {
        stage3 += stage3_gamma_powers[2];
    }
    if entry.left_is_pc() {
        stage3 += stage3_gamma_powers[3];
    }
    if entry.right_is_rs2() {
        stage3 += stage3_gamma_powers[4];
    }
    if entry.right_is_imm() {
        stage3 += stage3_gamma_powers[5];
    }
    if entry.is_noop() {
        stage3 += stage3_gamma_powers[6];
    }
    if flags[7] {
        stage3 += stage3_gamma_powers[7];
    }
    if flags[12] {
        stage3 += stage3_gamma_powers[8];
    }

    let stage4 = stage67_register_eq(entry.rd(), stage4_register_point, symbols.entry_rd)?
        * stage4_gamma_powers[0]
        + stage67_register_eq(entry.rs1(), stage4_register_point, symbols.entry_rs1)?
            * stage4_gamma_powers[1]
        + stage67_register_eq(entry.rs2(), stage4_register_point, symbols.entry_rs2)?
            * stage4_gamma_powers[2];

    let mut stage5 = stage67_register_eq(entry.rd(), stage5_register_point, symbols.entry_rd)?
        * stage5_gamma_powers[0];
    if !entry.is_interleaved() {
        stage5 += stage5_gamma_powers[1];
    }
    if let Some(table) = entry.lookup_table() {
        if table >= num_lookup_tables {
            return Err(RuntimePlanError::InvalidInputLength {
                input: symbols.entry_lookup_table,
                expected: num_lookup_tables,
                actual: table + 1,
            });
        }
        stage5 += stage5_gamma_powers[2 + table];
    }

    Ok([stage1, stage2, stage3, stage4, stage5])
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
    Ok(indexed_boolean_eq(index, point))
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

pub fn lt_polynomial_eval(x: &[Fr], y: &[Fr]) -> Fr {
    let mut lt_eval = Fr::from_u64(0);
    let mut eq_term = Fr::from_u64(1);
    for (x_i, y_i) in x.iter().zip(y.iter()) {
        lt_eval += (Fr::from_u64(1) - *x_i) * *y_i * eq_term;
        eq_term *= Fr::from_u64(1) - *x_i - *y_i + *x_i * *y_i + *x_i * *y_i;
    }
    lt_eval
}
