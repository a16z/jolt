#[cfg(test)]
use crate::cuda::common::context::CudaKernelContext;
#[cfg(test)]
use crate::cuda::common::device::DeviceFrVec;
#[cfg(test)]
use crate::cuda::common::error::CudaError;

#[cfg(all(test, feature = "parallel"))]
use rayon::prelude::*;

pub const COLD: u32 = u32::MAX;

#[cfg(test)]
pub const PACK_CHUNK: usize = 1 << 14;

#[inline]
#[cfg(test)]
pub fn encode_address(address: Option<u64>, addresses: usize) -> Result<u32, u64> {
    match address {
        None => Ok(COLD),
        Some(address) => match u32::try_from(address) {
            Ok(packed) if packed != COLD && (packed as usize) < addresses => Ok(packed),
            _ => Err(address),
        },
    }
}

#[derive(Debug)]
#[cfg(test)]
pub struct IncAndAddress {
    pub inc: Vec<i128>,
    pub address: Vec<u32>,
}

#[cfg(test)]
pub fn pack_inc_and_address<R: Sync>(
    rows: &[R],
    addresses: usize,
    extract: impl Fn(&R) -> (i128, Option<u64>) + Sync,
) -> Result<IncAndAddress, u64> {
    let mut inc = vec![0i128; rows.len()];
    let mut address = vec![COLD; rows.len()];

    #[cfg(feature = "parallel")]
    let rejected = inc
        .par_chunks_mut(PACK_CHUNK)
        .zip(address.par_chunks_mut(PACK_CHUNK))
        .zip(rows.par_chunks(PACK_CHUNK))
        .filter_map(|((inc, address), rows)| fill(inc, address, rows, addresses, &extract))
        .min();
    #[cfg(not(feature = "parallel"))]
    let rejected = inc
        .chunks_mut(PACK_CHUNK)
        .zip(address.chunks_mut(PACK_CHUNK))
        .zip(rows.chunks(PACK_CHUNK))
        .filter_map(|((inc, address), rows)| fill(inc, address, rows, addresses, &extract))
        .min();

    match rejected {
        Some(address) => Err(address),
        None => Ok(IncAndAddress { inc, address }),
    }
}

#[cfg(test)]
fn fill<R>(
    inc: &mut [i128],
    address: &mut [u32],
    rows: &[R],
    addresses: usize,
    extract: &(impl Fn(&R) -> (i128, Option<u64>) + Sync),
) -> Option<u64> {
    let mut rejected: Option<u64> = None;
    for (index, row) in rows.iter().enumerate() {
        let (increment, hot) = extract(row);
        inc[index] = increment;
        match encode_address(hot, addresses) {
            Ok(packed) => address[index] = packed,
            Err(value) => {
                rejected = Some(rejected.map_or(value, |seen: u64| seen.min(value)));
            }
        }
    }
    rejected
}

#[cfg(test)]
pub fn device_inc_and_address<R: Sync>(
    context: &CudaKernelContext,
    rows: &[R],
    addresses: usize,
    extract: impl Fn(&R) -> (i128, Option<u64>) + Sync,
) -> Result<(DeviceFrVec, Vec<u32>), CudaError> {
    let packed = pack_inc_and_address(rows, addresses, extract).map_err(|address| {
        CudaError::LengthMismatch {
            expected: addresses,
            got: address as usize,
        }
    })?;
    Ok((context.i128_to_montgomery(&packed.inc)?, packed.address))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::{encode_address, pack_inc_and_address, COLD, PACK_CHUNK};

    #[test]
    fn encode_address_maps_absent_to_cold_and_rejects_out_of_range() {
        assert_eq!(encode_address(None, 8).unwrap(), COLD);
        assert_eq!(encode_address(Some(0), 8).unwrap(), 0);
        assert_eq!(encode_address(Some(7), 8).unwrap(), 7);
        assert_eq!(encode_address(Some(8), 8).unwrap_err(), 8);
        assert_eq!(
            encode_address(Some(u64::from(COLD)), 1usize << 32).unwrap_err(),
            u64::from(COLD)
        );
        assert_eq!(encode_address(Some(1 << 40), 1 << 20).unwrap_err(), 1 << 40);
    }

    #[test]
    fn pack_spans_more_than_one_chunk_and_keeps_row_order() {
        let rows: Vec<(i128, Option<u64>)> = (0..(2 * PACK_CHUNK + 7))
            .map(|index| {
                let inc = if index % 3 == 0 {
                    -(index as i128)
                } else {
                    index as i128
                };
                let hot = if index % 5 == 2 {
                    None
                } else {
                    Some((index % 64) as u64)
                };
                (inc, hot)
            })
            .collect();
        let packed = pack_inc_and_address(&rows, 64, |row| *row).unwrap();
        for (index, (inc, hot)) in rows.iter().enumerate() {
            assert_eq!(packed.inc[index], *inc, "increment at {index}");
            let expected = hot.map_or(COLD, |address| address as u32);
            assert_eq!(packed.address[index], expected, "address at {index}");
        }
    }

    #[test]
    fn pack_reports_the_smallest_rejected_address() {
        let rows = vec![
            (0i128, Some(3u64)),
            (0, Some(1 << 33)),
            (0, Some(1 << 32)),
            (0, None),
        ];
        assert_eq!(
            pack_inc_and_address(&rows, 8, |row| *row).unwrap_err(),
            1 << 32
        );
    }
}
