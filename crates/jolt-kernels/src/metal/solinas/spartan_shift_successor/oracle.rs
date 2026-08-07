//! Dense algebra oracle independent of the successor factorization.

use jolt_field::Field;
use jolt_poly::{EqPlusOnePrefixSuffix, EqPolynomial};

use super::abi::SpartanShiftSuccessorGeometry;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpartanShiftSuccessorRow {
    pub unexpanded_pc: u64,
    pub pc: u64,
    pub is_virtual: bool,
    pub is_first_in_sequence: bool,
    pub is_noop: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OuterComponentTables<F> {
    pub unexpanded_pc_current: Vec<F>,
    pub unexpanded_pc_successor: Vec<F>,
    pub pc_current: Vec<F>,
    pub pc_successor: Vec<F>,
    pub is_virtual_current: Vec<F>,
    pub is_virtual_successor: Vec<F>,
    pub is_first_current: Vec<F>,
    pub is_first_successor: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductComponentTables<F> {
    pub nonnoop_current: Vec<F>,
    pub nonnoop_successor: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixQTables<F> {
    pub outer_current: Vec<F>,
    pub outer_successor: Vec<F>,
    pub product_current: Vec<F>,
    pub product_successor: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DenseOutputs<T> {
    pub unexpanded_pc: T,
    pub pc: T,
    pub is_virtual: T,
    pub is_first_in_sequence: T,
    pub is_noop: T,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectTrace<F> {
    pub initial_claim: F,
    pub round_endpoints: Vec<[F; 2]>,
    pub outputs: DenseOutputs<F>,
    pub final_relation: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DirectDenseState<F> {
    eq_plus_one_outer: Vec<F>,
    eq_plus_one_product: Vec<F>,
    unexpanded_pc: Vec<F>,
    pc: Vec<F>,
    is_virtual: Vec<F>,
    is_first_in_sequence: Vec<F>,
    is_noop: Vec<F>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpartanShiftSuccessorOracleError {
    InvalidRows,
    WrongPointLength,
    WrongChallengeCount,
    WrongTableLength,
    EmptyDenseState,
}

/// Evaluates every dense table and binds it exactly as the reference member.
/// No prefix-suffix identity is used in this path.
pub fn direct_trace<F: Field>(
    rows: &[SpartanShiftSuccessorRow],
    r_outer: &[F],
    r_product: &[F],
    gamma: F,
    challenges: &[F],
) -> Result<DirectTrace<F>, SpartanShiftSuccessorOracleError> {
    let geometry = checked_geometry(rows, r_outer, r_product)?;
    if challenges.len() != geometry.log_t {
        return Err(SpartanShiftSuccessorOracleError::WrongChallengeCount);
    }
    let mut state = direct_dense_state(rows, r_outer, r_product)?;
    let initial_claim = direct_sum(&state, gamma)?;
    let mut round_endpoints = Vec::with_capacity(challenges.len());
    for &challenge in challenges {
        round_endpoints.push(direct_round_endpoints(&state, gamma)?);
        bind_direct_state(&mut state, challenge)?;
    }
    let outputs = direct_outputs(&state)?;
    let final_relation = direct_sum(&state, gamma)?;
    Ok(DirectTrace {
        initial_claim,
        round_endpoints,
        outputs,
        final_relation,
    })
}

pub fn outer_component_tables<F: Field>(
    rows: &[SpartanShiftSuccessorRow],
    r_outer: &[F],
) -> Result<OuterComponentTables<F>, SpartanShiftSuccessorOracleError> {
    let geometry = checked_geometry(rows, r_outer, r_outer)?;
    let high_weights = EqPolynomial::<F>::evals(&r_outer[..geometry.suffix_vars], None);
    let mut tables = OuterComponentTables {
        unexpanded_pc_current: zero_table(geometry.prefix_elements),
        unexpanded_pc_successor: zero_table(geometry.prefix_elements),
        pc_current: zero_table(geometry.prefix_elements),
        pc_successor: zero_table(geometry.prefix_elements),
        is_virtual_current: zero_table(geometry.prefix_elements),
        is_virtual_successor: zero_table(geometry.prefix_elements),
        is_first_current: zero_table(geometry.prefix_elements),
        is_first_successor: zero_table(geometry.prefix_elements),
    };

    for high in 0..geometry.suffix_elements {
        let current_weight = high_weights[high];
        let successor_weight = high.checked_sub(1).map(|index| high_weights[index]);
        for low in 0..geometry.prefix_elements {
            let row = rows[high * geometry.prefix_elements + low];
            tables.unexpanded_pc_current[low] += current_weight * F::from_u64(row.unexpanded_pc);
            tables.pc_current[low] += current_weight * F::from_u64(row.pc);
            if row.is_virtual {
                tables.is_virtual_current[low] += current_weight;
            }
            if row.is_first_in_sequence {
                tables.is_first_current[low] += current_weight;
            }
            if let Some(weight) = successor_weight {
                tables.unexpanded_pc_successor[low] += weight * F::from_u64(row.unexpanded_pc);
                tables.pc_successor[low] += weight * F::from_u64(row.pc);
                if row.is_virtual {
                    tables.is_virtual_successor[low] += weight;
                }
                if row.is_first_in_sequence {
                    tables.is_first_successor[low] += weight;
                }
            }
        }
    }
    Ok(tables)
}

pub fn product_component_tables<F: Field>(
    rows: &[SpartanShiftSuccessorRow],
    r_product: &[F],
) -> Result<ProductComponentTables<F>, SpartanShiftSuccessorOracleError> {
    let geometry = checked_geometry(rows, r_product, r_product)?;
    let high_weights = EqPolynomial::<F>::evals(&r_product[..geometry.suffix_vars], None);
    let mut tables = ProductComponentTables {
        nonnoop_current: zero_table(geometry.prefix_elements),
        nonnoop_successor: zero_table(geometry.prefix_elements),
    };
    for high in 0..geometry.suffix_elements {
        let current_weight = high_weights[high];
        let successor_weight = high.checked_sub(1).map(|index| high_weights[index]);
        for low in 0..geometry.prefix_elements {
            let row = rows[high * geometry.prefix_elements + low];
            if !row.is_noop {
                tables.nonnoop_current[low] += current_weight;
                if let Some(weight) = successor_weight {
                    tables.nonnoop_successor[low] += weight;
                }
            }
        }
    }
    Ok(tables)
}

pub fn combine_q<F: Field>(
    outer: &OuterComponentTables<F>,
    product: &ProductComponentTables<F>,
    gamma: F,
) -> Result<PrefixQTables<F>, SpartanShiftSuccessorOracleError> {
    let length = outer.unexpanded_pc_current.len();
    validate_component_lengths(outer, product, length)?;
    let powers = gamma_powers(gamma);
    let combine_outer = |upc: &[F], pc: &[F], virtual_flag: &[F], first: &[F]| {
        (0..length)
            .map(|index| {
                upc[index]
                    + powers[1] * pc[index]
                    + powers[2] * virtual_flag[index]
                    + powers[3] * first[index]
            })
            .collect::<Vec<F>>()
    };
    Ok(PrefixQTables {
        outer_current: combine_outer(
            &outer.unexpanded_pc_current,
            &outer.pc_current,
            &outer.is_virtual_current,
            &outer.is_first_current,
        ),
        outer_successor: combine_outer(
            &outer.unexpanded_pc_successor,
            &outer.pc_successor,
            &outer.is_virtual_successor,
            &outer.is_first_successor,
        ),
        product_current: product
            .nonnoop_current
            .iter()
            .map(|&value| powers[4] * value)
            .collect(),
        product_successor: product
            .nonnoop_successor
            .iter()
            .map(|&value| powers[4] * value)
            .collect(),
    })
}

pub fn factorized_initial_claim<F: Field>(
    q: &PrefixQTables<F>,
    r_outer: &[F],
    r_product: &[F],
) -> Result<F, SpartanShiftSuccessorOracleError> {
    if r_outer.len() != r_product.len() || r_outer.is_empty() {
        return Err(SpartanShiftSuccessorOracleError::WrongPointLength);
    }
    let outer = EqPlusOnePrefixSuffix::new(r_outer);
    let product = EqPlusOnePrefixSuffix::new(r_product);
    let length = outer.prefix_0.len();
    for table in [
        &outer.prefix_1,
        &product.prefix_0,
        &product.prefix_1,
        &q.outer_current,
        &q.outer_successor,
        &q.product_current,
        &q.product_successor,
    ] {
        if table.len() != length {
            return Err(SpartanShiftSuccessorOracleError::WrongTableLength);
        }
    }
    let mut claim = F::zero();
    for index in 0..length {
        claim += outer.prefix_0[index] * q.outer_current[index]
            + outer.prefix_1[index] * q.outer_successor[index]
            + product.prefix_0[index] * q.product_current[index]
            + product.prefix_1[index] * q.product_successor[index];
    }
    Ok(claim)
}

pub fn fold_all_columns<F: Field>(
    rows: &[SpartanShiftSuccessorRow],
    prefix_challenges: &[F],
) -> Result<DenseOutputs<Vec<F>>, SpartanShiftSuccessorOracleError> {
    let geometry = SpartanShiftSuccessorGeometry::new(rows.len())
        .map_err(|_| SpartanShiftSuccessorOracleError::InvalidRows)?;
    if prefix_challenges.len() != geometry.prefix_vars {
        return Err(SpartanShiftSuccessorOracleError::WrongChallengeCount);
    }
    let point = prefix_challenges.iter().rev().copied().collect::<Vec<_>>();
    let weights = EqPolynomial::<F>::evals(&point, None);
    let mut output = DenseOutputs {
        unexpanded_pc: zero_table(geometry.suffix_elements),
        pc: zero_table(geometry.suffix_elements),
        is_virtual: zero_table(geometry.suffix_elements),
        is_first_in_sequence: zero_table(geometry.suffix_elements),
        is_noop: zero_table(geometry.suffix_elements),
    };
    for high in 0..geometry.suffix_elements {
        for (low, &weight) in weights.iter().enumerate() {
            let row = rows[high * geometry.prefix_elements + low];
            output.unexpanded_pc[high] += weight * F::from_u64(row.unexpanded_pc);
            output.pc[high] += weight * F::from_u64(row.pc);
            if row.is_virtual {
                output.is_virtual[high] += weight;
            }
            if row.is_first_in_sequence {
                output.is_first_in_sequence[high] += weight;
            }
            if row.is_noop {
                output.is_noop[high] += weight;
            }
        }
    }
    Ok(output)
}

pub fn fold_residual_columns<F: Field>(
    rows: &[SpartanShiftSuccessorRow],
    prefix_challenges: &[F],
) -> Result<DenseOutputs<Vec<F>>, SpartanShiftSuccessorOracleError> {
    let mut output = fold_all_columns(rows, prefix_challenges)?;
    output.unexpanded_pc.clear();
    Ok(output)
}

pub fn attach_midpoint_upc<F: Field>(
    mut residual: DenseOutputs<Vec<F>>,
    unexpanded_pc: Vec<F>,
) -> Result<DenseOutputs<Vec<F>>, SpartanShiftSuccessorOracleError> {
    let length = residual.pc.len();
    if unexpanded_pc.len() != length
        || !residual.unexpanded_pc.is_empty()
        || residual.is_virtual.len() != length
        || residual.is_first_in_sequence.len() != length
        || residual.is_noop.len() != length
    {
        return Err(SpartanShiftSuccessorOracleError::WrongTableLength);
    }
    residual.unexpanded_pc = unexpanded_pc;
    Ok(residual)
}

fn direct_dense_state<F: Field>(
    rows: &[SpartanShiftSuccessorRow],
    r_outer: &[F],
    r_product: &[F],
) -> Result<DirectDenseState<F>, SpartanShiftSuccessorOracleError> {
    let _ = checked_geometry(rows, r_outer, r_product)?;
    Ok(DirectDenseState {
        eq_plus_one_outer: direct_eq_plus_one(r_outer),
        eq_plus_one_product: direct_eq_plus_one(r_product),
        unexpanded_pc: rows
            .iter()
            .map(|row| F::from_u64(row.unexpanded_pc))
            .collect(),
        pc: rows.iter().map(|row| F::from_u64(row.pc)).collect(),
        is_virtual: rows
            .iter()
            .map(|row| F::from_bool(row.is_virtual))
            .collect(),
        is_first_in_sequence: rows
            .iter()
            .map(|row| F::from_bool(row.is_first_in_sequence))
            .collect(),
        is_noop: rows.iter().map(|row| F::from_bool(row.is_noop)).collect(),
    })
}

fn direct_sum<F: Field>(
    state: &DirectDenseState<F>,
    gamma: F,
) -> Result<F, SpartanShiftSuccessorOracleError> {
    let length = state.eq_plus_one_outer.len();
    validate_direct_lengths(state, length)?;
    let powers = gamma_powers(gamma);
    let mut claim = F::zero();
    for index in 0..length {
        claim += state.eq_plus_one_outer[index]
            * (state.unexpanded_pc[index]
                + powers[1] * state.pc[index]
                + powers[2] * state.is_virtual[index]
                + powers[3] * state.is_first_in_sequence[index])
            + state.eq_plus_one_product[index] * powers[4] * (F::one() - state.is_noop[index]);
    }
    Ok(claim)
}

fn direct_round_endpoints<F: Field>(
    state: &DirectDenseState<F>,
    gamma: F,
) -> Result<[F; 2], SpartanShiftSuccessorOracleError> {
    let length = state.eq_plus_one_outer.len();
    if length < 2 || !length.is_power_of_two() {
        return Err(SpartanShiftSuccessorOracleError::EmptyDenseState);
    }
    validate_direct_lengths(state, length)?;
    let powers = gamma_powers(gamma);
    let mut endpoints = [F::zero(); 2];
    for pair in 0..length / 2 {
        for (node, sample) in [F::zero(), F::from_u64(2)].into_iter().enumerate() {
            let eq_outer = extend(&state.eq_plus_one_outer, pair, sample);
            let eq_product = extend(&state.eq_plus_one_product, pair, sample);
            let upc = extend(&state.unexpanded_pc, pair, sample);
            let pc = extend(&state.pc, pair, sample);
            let virtual_flag = extend(&state.is_virtual, pair, sample);
            let first = extend(&state.is_first_in_sequence, pair, sample);
            let noop = extend(&state.is_noop, pair, sample);
            endpoints[node] += eq_outer
                * (upc + powers[1] * pc + powers[2] * virtual_flag + powers[3] * first)
                + eq_product * powers[4] * (F::one() - noop);
        }
    }
    Ok(endpoints)
}

fn bind_direct_state<F: Field>(
    state: &mut DirectDenseState<F>,
    challenge: F,
) -> Result<(), SpartanShiftSuccessorOracleError> {
    let length = state.eq_plus_one_outer.len();
    if length < 2 || !length.is_power_of_two() {
        return Err(SpartanShiftSuccessorOracleError::EmptyDenseState);
    }
    validate_direct_lengths(state, length)?;
    for table in [
        &mut state.eq_plus_one_outer,
        &mut state.eq_plus_one_product,
        &mut state.unexpanded_pc,
        &mut state.pc,
        &mut state.is_virtual,
        &mut state.is_first_in_sequence,
        &mut state.is_noop,
    ] {
        bind_table(table, challenge);
    }
    Ok(())
}

fn direct_outputs<F: Field>(
    state: &DirectDenseState<F>,
) -> Result<DenseOutputs<F>, SpartanShiftSuccessorOracleError> {
    validate_direct_lengths(state, 1)?;
    Ok(DenseOutputs {
        unexpanded_pc: state.unexpanded_pc[0],
        pc: state.pc[0],
        is_virtual: state.is_virtual[0],
        is_first_in_sequence: state.is_first_in_sequence[0],
        is_noop: state.is_noop[0],
    })
}

fn checked_geometry<F: Field>(
    rows: &[SpartanShiftSuccessorRow],
    r_outer: &[F],
    r_product: &[F],
) -> Result<SpartanShiftSuccessorGeometry, SpartanShiftSuccessorOracleError> {
    let geometry = SpartanShiftSuccessorGeometry::new(rows.len())
        .map_err(|_| SpartanShiftSuccessorOracleError::InvalidRows)?;
    if r_outer.len() != geometry.log_t || r_product.len() != geometry.log_t {
        return Err(SpartanShiftSuccessorOracleError::WrongPointLength);
    }
    Ok(geometry)
}

fn validate_component_lengths<F: Field>(
    outer: &OuterComponentTables<F>,
    product: &ProductComponentTables<F>,
    expected: usize,
) -> Result<(), SpartanShiftSuccessorOracleError> {
    for table in [
        &outer.unexpanded_pc_current,
        &outer.unexpanded_pc_successor,
        &outer.pc_current,
        &outer.pc_successor,
        &outer.is_virtual_current,
        &outer.is_virtual_successor,
        &outer.is_first_current,
        &outer.is_first_successor,
        &product.nonnoop_current,
        &product.nonnoop_successor,
    ] {
        if table.len() != expected {
            return Err(SpartanShiftSuccessorOracleError::WrongTableLength);
        }
    }
    Ok(())
}

fn validate_direct_lengths<F: Field>(
    state: &DirectDenseState<F>,
    expected: usize,
) -> Result<(), SpartanShiftSuccessorOracleError> {
    for table in [
        &state.eq_plus_one_outer,
        &state.eq_plus_one_product,
        &state.unexpanded_pc,
        &state.pc,
        &state.is_virtual,
        &state.is_first_in_sequence,
        &state.is_noop,
    ] {
        if table.len() != expected {
            return Err(SpartanShiftSuccessorOracleError::WrongTableLength);
        }
    }
    Ok(())
}

fn gamma_powers<F: Field>(gamma: F) -> [F; 5] {
    let mut powers = [F::one(); 5];
    for index in 1..powers.len() {
        powers[index] = powers[index - 1] * gamma;
    }
    powers
}

fn direct_eq_plus_one<F: Field>(point: &[F]) -> Vec<F> {
    let equality = EqPolynomial::<F>::evals(point, None);
    core::iter::once(F::zero())
        .chain(equality.iter().take(equality.len() - 1).copied())
        .collect()
}

fn bind_table<F: Field>(table: &mut Vec<F>, challenge: F) {
    let half = table.len() / 2;
    for index in 0..half {
        table[index] = extend(table, index, challenge);
    }
    table.truncate(half);
}

fn extend<F: Field>(table: &[F], pair: usize, sample: F) -> F {
    let low = table[2 * pair];
    low + sample * (table[2 * pair + 1] - low)
}

fn zero_table<F: Field>(length: usize) -> Vec<F> {
    vec![F::zero(); length]
}
