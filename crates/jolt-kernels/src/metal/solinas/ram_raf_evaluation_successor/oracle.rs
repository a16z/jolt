//! Scalar oracles independent of the split-equality and tiled shader paths.

use jolt_field::Field;

use super::abi::{
    validate_access_records, validate_bucket_projection, RamRafAccessRecord,
    RamRafBucketProjection, RamRafCompactError, RAM_RAF_SUCCESSOR_INNER_LENGTH,
    RAM_RAF_SUCCESSOR_TILE_ADDRESSES,
};

pub const RAM_RAF_ORACLE_NO_ACCESS: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OracleError {
    InvalidRows(usize),
    InvalidAddressDomain(usize),
    PointTooWide(usize),
    PointLength { expected: usize, got: usize },
    IndexOutsideDomain { index: usize, rows: usize },
    AddressOutsideDomain { row: usize, address: u32 },
    Compact(RamRafCompactError),
    InvalidMassTable(usize),
    ChallengeLength { expected: usize, got: usize },
    LowestAddressOverflow,
    RoundClaimMismatch { round: usize },
}

impl From<RamRafCompactError> for OracleError {
    fn from(value: RamRafCompactError) -> Self {
        Self::Compact(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuadraticEvaluations<F> {
    pub at_zero: F,
    pub at_one: F,
    pub at_two: F,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AffineProofOutput<F> {
    pub messages: Vec<QuadraticEvaluations<F>>,
    pub ram_ra: F,
    pub unmap_address: F,
    pub final_claim: F,
}

/// Evaluates `eq(point, index)` directly in big-endian Boolean-index order.
pub fn direct_equality<F: Field>(point: &[F], index: usize) -> Result<F, OracleError> {
    if point.len() >= usize::BITS as usize {
        return Err(OracleError::PointTooWide(point.len()));
    }
    let rows = 1usize << point.len();
    if index >= rows {
        return Err(OracleError::IndexOutsideDomain { index, rows });
    }
    Ok(point
        .iter()
        .copied()
        .enumerate()
        .fold(F::one(), |equality, (position, challenge)| {
            let shift = point.len() - 1 - position;
            if (index >> shift) & 1 == 1 {
                equality * challenge
            } else {
                equality * (F::one() - challenge)
            }
        }))
}

/// Direct dense-plane definition of the cycle pushforward.
pub fn dense_pushforward_oracle<F: Field>(
    addresses: &[u32],
    cycle_point: &[F],
    address_domain: usize,
) -> Result<Vec<F>, OracleError> {
    validate_point(addresses.len(), cycle_point, address_domain)?;
    let mut output = vec![F::zero(); address_domain];
    for (cycle, address) in addresses.iter().copied().enumerate() {
        if address == RAM_RAF_ORACLE_NO_ACCESS {
            continue;
        }
        let address_index = address as usize;
        if address_index >= address_domain {
            return Err(OracleError::AddressOutsideDomain {
                row: cycle,
                address,
            });
        }
        output[address_index] += direct_equality(cycle_point, cycle)?;
    }
    Ok(output)
}

/// Direct compact-record definition. It shares no split tables or bucketing.
pub fn compact_pushforward_oracle<F: Field>(
    records: &[RamRafAccessRecord],
    rows: usize,
    cycle_point: &[F],
    address_domain: usize,
) -> Result<Vec<F>, OracleError> {
    validate_point(rows, cycle_point, address_domain)?;
    validate_access_records(records, rows, address_domain)?;
    let mut output = vec![F::zero(); address_domain];
    for record in records.iter().copied() {
        output[record.address() as usize] += direct_equality(cycle_point, record.cycle() as usize)?;
    }
    Ok(output)
}

/// Reconstructs global cycle and address indices from the bucket ABI, then
/// evaluates the unsplit equality formula for each record.
pub fn bucket_pushforward_oracle<F: Field>(
    projection: &RamRafBucketProjection,
    rows: usize,
    cycle_point: &[F],
    address_domain: usize,
) -> Result<Vec<F>, OracleError> {
    validate_point(rows, cycle_point, address_domain)?;
    validate_bucket_projection(projection, rows, address_domain)?;
    let mut output = vec![F::zero(); address_domain];
    for descriptor in projection.descriptors.iter().copied() {
        let first = descriptor.first_record() as usize;
        let end = first + descriptor.record_count() as usize;
        for record in projection.records[first..end].iter().copied() {
            let cycle = descriptor.outer() as usize * RAM_RAF_SUCCESSOR_INNER_LENGTH
                + record.inner() as usize;
            let address = descriptor.tile() as usize * RAM_RAF_SUCCESSOR_TILE_ADDRESSES
                + record.local_address() as usize;
            output[address] += direct_equality(cycle_point, cycle)?;
        }
    }
    Ok(output)
}

/// Materializes the affine `UnmapAddress` table and runs all address rounds.
/// This is independent of the production affine-tail shortcut.
pub fn prove_affine_address_rounds<F: Field>(
    masses: &[F],
    lowest_address: u64,
    challenges: &[F],
) -> Result<AffineProofOutput<F>, OracleError> {
    if masses.is_empty() || !masses.len().is_power_of_two() {
        return Err(OracleError::InvalidMassTable(masses.len()));
    }
    let rounds = masses.len().ilog2() as usize;
    if challenges.len() != rounds {
        return Err(OracleError::ChallengeLength {
            expected: rounds,
            got: challenges.len(),
        });
    }
    let last_offset = 8u64
        .checked_mul((masses.len() - 1) as u64)
        .ok_or(OracleError::LowestAddressOverflow)?;
    let _ = lowest_address
        .checked_add(last_offset)
        .ok_or(OracleError::LowestAddressOverflow)?;

    let mut ra = masses.to_vec();
    let mut unmap = (0..masses.len())
        .map(|address| F::from_u64(lowest_address + 8 * address as u64))
        .collect::<Vec<_>>();
    let mut claim = unmap
        .iter()
        .copied()
        .zip(ra.iter().copied())
        .map(|(unmap, ra)| unmap * ra)
        .sum();
    let mut messages = Vec::with_capacity(rounds);

    for (round, challenge) in challenges.iter().copied().enumerate() {
        let mut evaluations = [F::zero(); 3];
        for (unmap_pair, ra_pair) in unmap.chunks_exact(2).zip(ra.chunks_exact(2)) {
            for (sample_index, sample) in [0u64, 1, 2].into_iter().enumerate() {
                let sample = F::from_u64(sample);
                let unmap_at = bind(unmap_pair[0], unmap_pair[1], sample);
                let ra_at = bind(ra_pair[0], ra_pair[1], sample);
                evaluations[sample_index] += unmap_at * ra_at;
            }
        }
        if evaluations[0] + evaluations[1] != claim {
            return Err(OracleError::RoundClaimMismatch { round });
        }
        messages.push(QuadraticEvaluations {
            at_zero: evaluations[0],
            at_one: evaluations[1],
            at_two: evaluations[2],
        });
        bind_table(&mut unmap, challenge);
        bind_table(&mut ra, challenge);
        claim = unmap
            .iter()
            .copied()
            .zip(ra.iter().copied())
            .map(|(unmap, ra)| unmap * ra)
            .sum();
    }

    Ok(AffineProofOutput {
        messages,
        ram_ra: ra[0],
        unmap_address: unmap[0],
        final_claim: claim,
    })
}

fn validate_point<F: Field>(
    rows: usize,
    point: &[F],
    address_domain: usize,
) -> Result<(), OracleError> {
    if rows == 0 || !rows.is_power_of_two() {
        return Err(OracleError::InvalidRows(rows));
    }
    if address_domain == 0 || !address_domain.is_power_of_two() {
        return Err(OracleError::InvalidAddressDomain(address_domain));
    }
    let expected = rows.ilog2() as usize;
    if point.len() != expected {
        return Err(OracleError::PointLength {
            expected,
            got: point.len(),
        });
    }
    Ok(())
}

fn bind<F: Field>(zero: F, one: F, challenge: F) -> F {
    zero + challenge * (one - zero)
}

fn bind_table<F: Field>(table: &mut Vec<F>, challenge: F) {
    let half = table.len() / 2;
    for index in 0..half {
        table[index] = bind(table[2 * index], table[2 * index + 1], challenge);
    }
    table.truncate(half);
}
