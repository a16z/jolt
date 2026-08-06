//! Scalar relation oracle kept independent of the factorized shader helpers.

use jolt_field::Field;

use super::abi::{IncrementAccessRow, RamValSuccessorRowError, MESSAGE_COLUMNS};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OracleError {
    InvalidCycleDomain(usize),
    InvalidAddressDomain(usize),
    InvalidDenseLtLength {
        expected: usize,
        got: usize,
    },
    InvalidFactorization {
        rows: usize,
        low: usize,
        high: usize,
        eq_high: usize,
    },
    WrongChallengeCount {
        expected: usize,
        got: usize,
    },
    InvalidRow {
        index: usize,
        source: RamValSuccessorRowError,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinalClaim<F> {
    pub ram_inc: F,
    pub ram_ra: F,
    pub lt_cycle_plus_gamma: F,
    pub product: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InitContribution<F> {
    pub selector: F,
    pub opening: F,
}

/// The canonical verifier input expression. `selector` values are the derived
/// `InitSelector` values supplied by the concrete relation.
pub fn input_claim<F: Field>(
    ram_val: F,
    ram_val_final: F,
    init_eval: F,
    contributions: &[InitContribution<F>],
    gamma: F,
) -> F {
    let init = contributions.iter().fold(init_eval, |init, contribution| {
        init - contribution.selector * contribution.opening
    });
    ram_val + gamma * ram_val_final - (F::one() + gamma) * init
}

/// Directly evaluates the degree-three round polynomial against a dense
/// `LT + gamma` table. This is the parity authority for the split shader. A
/// Store row with no remapped address retains `RamInc` and evaluates `RamRa`
/// to zero, matching the production address-zero convention.
pub fn direct_first_message<F: Field>(
    rows: &[IncrementAccessRow],
    eq_address: &[F],
    dense_lt_plus_gamma: &[F],
) -> Result<[F; MESSAGE_COLUMNS], OracleError> {
    validate_rows(rows, eq_address)?;
    if dense_lt_plus_gamma.len() != rows.len() {
        return Err(OracleError::InvalidDenseLtLength {
            expected: rows.len(),
            got: dense_lt_plus_gamma.len(),
        });
    }
    let mut output = [F::zero(); MESSAGE_COLUMNS];
    for (pair_index, pair) in rows.chunks_exact(2).enumerate() {
        let base = 2 * pair_index;
        let inc = [pair[0].ram_increment(), pair[1].ram_increment()];
        let ra = [
            pair[0]
                .ram_ra(eq_address)
                .map_err(|source| OracleError::InvalidRow {
                    index: base,
                    source,
                })?,
            pair[1]
                .ram_ra(eq_address)
                .map_err(|source| OracleError::InvalidRow {
                    index: base + 1,
                    source,
                })?,
        ];
        let lt = [dense_lt_plus_gamma[base], dense_lt_plus_gamma[base + 1]];
        for (column, sample) in [0_u64, 2, 3].into_iter().enumerate() {
            let sample = F::from_u64(sample);
            output[column] += bind(inc[0], inc[1], sample)
                * bind(ra[0], ra[1], sample)
                * bind(lt[0], lt[1], sample);
        }
    }
    Ok(output)
}

/// Mirrors only the mathematical split identity, not the shader's control
/// flow. Pairs whose two RAM increments are zero are omitted exactly. A
/// nonzero address-zero Store is not omitted because a neighboring load can
/// make the interpolated `RamRa` nonzero away from the Boolean endpoints.
pub fn factorized_sparse_first_message<F: Field>(
    rows: &[IncrementAccessRow],
    eq_address: &[F],
    lt_low: &[F],
    lt_high: &[F],
    eq_high: &[F],
) -> Result<[F; MESSAGE_COLUMNS], OracleError> {
    validate_rows(rows, eq_address)?;
    validate_factorization(rows.len(), lt_low, lt_high, eq_high)?;
    let mut output = [F::zero(); MESSAGE_COLUMNS];
    for (pair_index, pair) in rows.chunks_exact(2).enumerate() {
        if !pair[0].has_nonzero_ram_increment() && !pair[1].has_nonzero_ram_increment() {
            continue;
        }
        let base = 2 * pair_index;
        let high = base / lt_low.len();
        let low = base % lt_low.len();
        let inc = [pair[0].ram_increment(), pair[1].ram_increment()];
        let ra = [
            pair[0]
                .ram_ra(eq_address)
                .map_err(|source| OracleError::InvalidRow {
                    index: base,
                    source,
                })?,
            pair[1]
                .ram_ra(eq_address)
                .map_err(|source| OracleError::InvalidRow {
                    index: base + 1,
                    source,
                })?,
        ];
        for (column, sample) in [0_u64, 2, 3].into_iter().enumerate() {
            let sample = F::from_u64(sample);
            let lt_low_at = bind(lt_low[low], lt_low[low + 1], sample);
            let lt = lt_high[high] + eq_high[high] * lt_low_at;
            output[column] += bind(inc[0], inc[1], sample) * bind(ra[0], ra[1], sample) * lt;
        }
    }
    Ok(output)
}

pub fn dense_lt_from_split<F: Field>(
    rows: usize,
    lt_low: &[F],
    lt_high: &[F],
    eq_high: &[F],
) -> Result<Vec<F>, OracleError> {
    validate_factorization(rows, lt_low, lt_high, eq_high)?;
    Ok((0..rows)
        .map(|index| {
            let high = index / lt_low.len();
            let low = index % lt_low.len();
            lt_high[high] + eq_high[high] * lt_low[low]
        })
        .collect())
}

/// Fully binds the three output factors in low-to-high challenge order and
/// returns the exact verifier-side output relation.
pub fn final_claim<F: Field>(
    rows: &[IncrementAccessRow],
    eq_address: &[F],
    dense_lt_plus_gamma: &[F],
    challenges: &[F],
) -> Result<FinalClaim<F>, OracleError> {
    validate_rows(rows, eq_address)?;
    if dense_lt_plus_gamma.len() != rows.len() {
        return Err(OracleError::InvalidDenseLtLength {
            expected: rows.len(),
            got: dense_lt_plus_gamma.len(),
        });
    }
    let expected_challenges = rows.len().ilog2() as usize;
    if challenges.len() != expected_challenges {
        return Err(OracleError::WrongChallengeCount {
            expected: expected_challenges,
            got: challenges.len(),
        });
    }
    let mut inc = rows
        .iter()
        .copied()
        .map(|row| row.ram_increment())
        .collect::<Vec<F>>();
    let mut ra = rows
        .iter()
        .copied()
        .enumerate()
        .map(|(index, row)| {
            row.ram_ra(eq_address)
                .map_err(|source| OracleError::InvalidRow { index, source })
        })
        .collect::<Result<Vec<F>, _>>()?;
    let mut lt = dense_lt_plus_gamma.to_vec();
    for challenge in challenges.iter().copied() {
        bind_table(&mut inc, challenge);
        bind_table(&mut ra, challenge);
        bind_table(&mut lt, challenge);
    }
    let product = inc[0] * ra[0] * lt[0];
    Ok(FinalClaim {
        ram_inc: inc[0],
        ram_ra: ra[0],
        lt_cycle_plus_gamma: lt[0],
        product,
    })
}

fn validate_rows<F: Field>(
    rows: &[IncrementAccessRow],
    eq_address: &[F],
) -> Result<(), OracleError> {
    if rows.len() < 2 || !rows.len().is_power_of_two() {
        return Err(OracleError::InvalidCycleDomain(rows.len()));
    }
    if eq_address.is_empty() || !eq_address.len().is_power_of_two() {
        return Err(OracleError::InvalidAddressDomain(eq_address.len()));
    }
    let address_domain = u32::try_from(eq_address.len())
        .map_err(|_| OracleError::InvalidAddressDomain(eq_address.len()))?;
    for (index, row) in rows.iter().copied().enumerate() {
        row.validate_address_domain(address_domain)
            .map_err(|source| OracleError::InvalidRow { index, source })?;
    }
    Ok(())
}

fn validate_factorization<F: Field>(
    rows: usize,
    lt_low: &[F],
    lt_high: &[F],
    eq_high: &[F],
) -> Result<(), OracleError> {
    if lt_low.len() < 2
        || !lt_low.len().is_power_of_two()
        || lt_high.is_empty()
        || !lt_high.len().is_power_of_two()
        || eq_high.len() != lt_high.len()
        || lt_low.len().checked_mul(lt_high.len()) != Some(rows)
    {
        return Err(OracleError::InvalidFactorization {
            rows,
            low: lt_low.len(),
            high: lt_high.len(),
            eq_high: eq_high.len(),
        });
    }
    Ok(())
}

fn bind_table<F: Field>(table: &mut Vec<F>, challenge: F) {
    let half = table.len() / 2;
    for index in 0..half {
        table[index] = bind(table[2 * index], table[2 * index + 1], challenge);
    }
    table.truncate(half);
}

fn bind<F: Field>(low: F, high: F, challenge: F) -> F {
    low + challenge * (high - low)
}
