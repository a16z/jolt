//! Small-prime oracle independent of the proposed Metal representation.

use core::ops::{Add, AddAssign, Mul, Sub};

pub const TILE_WIDTH: usize = 16;
const MODULUS: u64 = 97;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fp(u64);

impl Fp {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);

    pub const fn new(value: u64) -> Self {
        Self(value % MODULUS)
    }
}

impl Add for Fp {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.0 + rhs.0)
    }
}

impl AddAssign for Fp {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Fp {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(MODULUS + self.0 - rhs.0)
    }
}

impl Mul for Fp {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.0 * rhs.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MicrotileView {
    pub masks: Vec<u16>,
    pub offsets: Vec<u32>,
    pub addresses: Vec<u16>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectTrace {
    pub initial_claim: Fp,
    pub messages: Vec<[Fp; 4]>,
    pub factor_outputs: [Fp; 2],
    pub cycle_eq_output: Fp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OracleError {
    InvalidRows,
    AddressOutsideDomain,
    InvalidPoint,
    InvalidChallengeCount,
    InvalidMicrotile,
}

pub fn encode_microtiles(addresses: &[Option<u16>]) -> Result<MicrotileView, OracleError> {
    if addresses.len() < TILE_WIDTH || !addresses.len().is_power_of_two() {
        return Err(OracleError::InvalidRows);
    }
    let mut masks = Vec::with_capacity(addresses.len() / TILE_WIDTH);
    let mut offsets = Vec::with_capacity(masks.capacity() + 1);
    let mut compact = Vec::new();
    offsets.push(0);
    for tile in addresses.chunks_exact(TILE_WIDTH) {
        let mut mask = 0_u16;
        for (lane, address) in tile.iter().copied().enumerate() {
            if let Some(address) = address {
                if address >= 1 << 13 {
                    return Err(OracleError::AddressOutsideDomain);
                }
                mask |= 1 << lane;
                compact.push(address);
            }
        }
        masks.push(mask);
        offsets.push(u32::try_from(compact.len()).map_err(|_| OracleError::InvalidMicrotile)?);
    }
    Ok(MicrotileView {
        masks,
        offsets,
        addresses: compact,
    })
}

pub fn decode_microtiles(view: &MicrotileView) -> Result<Vec<Option<u16>>, OracleError> {
    if view.masks.is_empty() || view.offsets.len() != view.masks.len() + 1 {
        return Err(OracleError::InvalidMicrotile);
    }
    let mut dense = vec![None; view.masks.len() * TILE_WIDTH];
    for (tile, &mask) in view.masks.iter().enumerate() {
        let begin = view.offsets[tile] as usize;
        let end = view.offsets[tile + 1] as usize;
        if end < begin || end > view.addresses.len() || end - begin != mask.count_ones() as usize {
            return Err(OracleError::InvalidMicrotile);
        }
        let mut compact = begin;
        for lane in 0..TILE_WIDTH {
            if mask & (1 << lane) != 0 {
                let address = view.addresses[compact];
                if address >= 1 << 13 {
                    return Err(OracleError::AddressOutsideDomain);
                }
                dense[tile * TILE_WIDTH + lane] = Some(address);
                compact += 1;
            }
        }
    }
    if view.offsets.last().copied() != Some(view.addresses.len() as u32) {
        return Err(OracleError::InvalidMicrotile);
    }
    Ok(dense)
}

/// Direct dense relation; it does not use the microtile grouping or Gruen split.
pub fn direct_trace(
    addresses: &[Option<u16>],
    address_point: &[Fp],
    cycle_point: &[Fp],
    challenges: &[Fp],
) -> Result<DirectTrace, OracleError> {
    let log_t = checked_inputs(addresses, address_point, cycle_point, challenges)?;
    let tables = chunk_tables(address_point)?;
    let mut factors = dense_factors(addresses, &tables)?;
    let mut cycle_eq = eq_table(cycle_point);
    let initial_claim = relation_sum(&cycle_eq, &factors)?;
    let mut messages = Vec::with_capacity(log_t);
    for &challenge in challenges {
        messages.push(round_message(&cycle_eq, &factors)?);
        bind(&mut cycle_eq, challenge)?;
        bind(&mut factors[0], challenge)?;
        bind(&mut factors[1], challenge)?;
    }
    Ok(DirectTrace {
        initial_claim,
        messages,
        factor_outputs: [factors[0][0], factors[1][0]],
        cycle_eq_output: cycle_eq[0],
    })
}

pub fn microtile_trace(
    view: &MicrotileView,
    address_point: &[Fp],
    cycle_point: &[Fp],
    challenges: &[Fp],
) -> Result<DirectTrace, OracleError> {
    direct_trace(
        &decode_microtiles(view)?,
        address_point,
        cycle_point,
        challenges,
    )
}

pub fn dense_h_gather(
    addresses: &[Option<u16>],
    address_point: &[Fp],
    prefix_point: &[Fp],
) -> Result<Vec<Fp>, OracleError> {
    if addresses.len() < TILE_WIDTH || !addresses.len().is_power_of_two() {
        return Err(OracleError::InvalidRows);
    }
    let prefix = 1_usize << prefix_point.len();
    if !addresses.len().is_multiple_of(prefix) || address_point.len() != 13 {
        return Err(OracleError::InvalidPoint);
    }
    let eq_address = eq_table(address_point);
    let eq_prefix = eq_table(prefix_point);
    let mut output = vec![Fp::ZERO; addresses.len() / prefix];
    for (row, address) in addresses.iter().copied().enumerate() {
        if let Some(address) = address {
            output[row / prefix] += eq_address[address as usize] * eq_prefix[row % prefix];
        }
    }
    Ok(output)
}

pub fn microtile_h_gather(
    view: &MicrotileView,
    address_point: &[Fp],
    prefix_point: &[Fp],
) -> Result<Vec<Fp>, OracleError> {
    dense_h_gather(&decode_microtiles(view)?, address_point, prefix_point)
}

pub fn chunk_product_identity(
    address: u16,
    address_point: &[Fp],
) -> Result<(Fp, Fp, Fp), OracleError> {
    let tables = chunk_tables(address_point)?;
    if address >= 1 << 13 {
        return Err(OracleError::AddressOutsideDomain);
    }
    let high = tables[0][usize::from(address >> 8)];
    let low = tables[1][usize::from(address & 0xff)];
    Ok((high, low, eq_at(address_point, usize::from(address))))
}

fn checked_inputs(
    addresses: &[Option<u16>],
    address_point: &[Fp],
    cycle_point: &[Fp],
    challenges: &[Fp],
) -> Result<usize, OracleError> {
    if addresses.len() < TILE_WIDTH || !addresses.len().is_power_of_two() {
        return Err(OracleError::InvalidRows);
    }
    let log_t = addresses.len().ilog2() as usize;
    if address_point.len() != 13 || cycle_point.len() != log_t {
        return Err(OracleError::InvalidPoint);
    }
    if challenges.len() != log_t {
        return Err(OracleError::InvalidChallengeCount);
    }
    if addresses
        .iter()
        .flatten()
        .any(|address| *address >= 1 << 13)
    {
        return Err(OracleError::AddressOutsideDomain);
    }
    Ok(log_t)
}

fn chunk_tables(address_point: &[Fp]) -> Result<[Vec<Fp>; 2], OracleError> {
    if address_point.len() != 13 {
        return Err(OracleError::InvalidPoint);
    }
    let mut padded = vec![Fp::ZERO; 3];
    padded.extend_from_slice(address_point);
    Ok([eq_table(&padded[..8]), eq_table(&padded[8..])])
}

fn dense_factors(
    addresses: &[Option<u16>],
    tables: &[Vec<Fp>; 2],
) -> Result<[Vec<Fp>; 2], OracleError> {
    let mut factors = [
        Vec::with_capacity(addresses.len()),
        Vec::with_capacity(addresses.len()),
    ];
    for address in addresses {
        match address {
            Some(address) if *address < 1 << 13 => {
                factors[0].push(tables[0][usize::from(*address >> 8)]);
                factors[1].push(tables[1][usize::from(*address & 0xff)]);
            }
            Some(_) => return Err(OracleError::AddressOutsideDomain),
            None => {
                factors[0].push(Fp::ZERO);
                factors[1].push(Fp::ZERO);
            }
        }
    }
    Ok(factors)
}

fn round_message(cycle_eq: &[Fp], factors: &[Vec<Fp>; 2]) -> Result<[Fp; 4], OracleError> {
    if cycle_eq.len() < 2
        || !cycle_eq.len().is_power_of_two()
        || factors.iter().any(|factor| factor.len() != cycle_eq.len())
    {
        return Err(OracleError::InvalidRows);
    }
    let mut message = [Fp::ZERO; 4];
    for pair in 0..cycle_eq.len() / 2 {
        for (sample, output) in message.iter_mut().enumerate() {
            let t = Fp::new(sample as u64);
            let eq = interpolate(cycle_eq[2 * pair], cycle_eq[2 * pair + 1], t);
            let f0 = interpolate(factors[0][2 * pair], factors[0][2 * pair + 1], t);
            let f1 = interpolate(factors[1][2 * pair], factors[1][2 * pair + 1], t);
            *output += eq * f0 * f1;
        }
    }
    Ok(message)
}

fn relation_sum(cycle_eq: &[Fp], factors: &[Vec<Fp>; 2]) -> Result<Fp, OracleError> {
    if factors.iter().any(|factor| factor.len() != cycle_eq.len()) {
        return Err(OracleError::InvalidRows);
    }
    Ok((0..cycle_eq.len()).fold(Fp::ZERO, |sum, row| {
        sum + cycle_eq[row] * factors[0][row] * factors[1][row]
    }))
}

fn bind(values: &mut Vec<Fp>, challenge: Fp) -> Result<(), OracleError> {
    if values.len() < 2 || !values.len().is_power_of_two() {
        return Err(OracleError::InvalidRows);
    }
    let half = values.len() / 2;
    for pair in 0..half {
        values[pair] = interpolate(values[2 * pair], values[2 * pair + 1], challenge);
    }
    values.truncate(half);
    Ok(())
}

fn interpolate(low: Fp, high: Fp, point: Fp) -> Fp {
    low + point * (high - low)
}

fn eq_table(point: &[Fp]) -> Vec<Fp> {
    (0..1_usize << point.len())
        .map(|index| eq_at(point, index))
        .collect()
}

fn eq_at(point: &[Fp], index: usize) -> Fp {
    point
        .iter()
        .enumerate()
        .fold(Fp::ONE, |weight, (coordinate, value)| {
            let bit = (index >> (point.len() - coordinate - 1)) & 1;
            weight * if bit == 0 { Fp::ONE - *value } else { *value }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(length: usize, seed: u64) -> Vec<Fp> {
        (0..length)
            .map(|index| Fp::new(seed + 7 * index as u64))
            .collect()
    }

    fn fixture() -> Vec<Option<u16>> {
        (0..64)
            .map(|row| (row % 3 == 0 || row == 63).then_some((17 * row % 8192) as u16))
            .collect()
    }

    #[test]
    fn microtile_round_trip_and_relation_match_dense() {
        let addresses = fixture();
        let view = encode_microtiles(&addresses).unwrap();
        assert_eq!(decode_microtiles(&view).unwrap(), addresses);
        let address_point = point(13, 5);
        let cycle_point = point(6, 11);
        let challenges = point(6, 19);
        assert_eq!(
            direct_trace(&addresses, &address_point, &cycle_point, &challenges).unwrap(),
            microtile_trace(&view, &address_point, &cycle_point, &challenges).unwrap()
        );
    }

    #[test]
    fn microtile_replaces_the_claim_reduction_high_view() {
        let addresses = fixture();
        let view = encode_microtiles(&addresses).unwrap();
        let address_point = point(13, 23);
        let prefix_point = point(3, 31);
        assert_eq!(
            dense_h_gather(&addresses, &address_point, &prefix_point).unwrap(),
            microtile_h_gather(&view, &address_point, &prefix_point).unwrap()
        );
    }

    #[test]
    fn committed_chunks_multiply_to_the_full_address_equality() {
        let point = point(13, 41);
        for address in [0, 1, 255, 256, 4095, 8191] {
            let (high, low, full) = chunk_product_identity(address, &point).unwrap();
            assert_eq!(high * low, full);
        }
    }

    #[test]
    fn malformed_offsets_fail_closed() {
        let mut view = encode_microtiles(&fixture()).unwrap();
        view.offsets[1] += 1;
        assert_eq!(decode_microtiles(&view), Err(OracleError::InvalidMicrotile));
    }
}
