//! Distinct-position sampling for sparse ring fold challenges.
//!
//! Fisher-Yates partial shuffle helpers shared by the signed-sparse sampler.

use crate::sampler::xof::XofCursor;
use akita_error::AkitaError;

/// Max ring dimension supported by the stack-buffer sampling paths.
///
/// Public API functions reject `D` above this with an error before reaching
/// the sampling internals.
pub(crate) const MAX_STACK_RING_DIM: usize = 2048;

/// Largest stack array used by one concrete sampler tier.
const MAX_STACK_TIER_RING_DIM: usize = 2048;

/// Reused sparse representation of the swaps performed by partial
/// Fisher-Yates. Missing entries retain their identity-permutation value.
pub(crate) struct DistinctPositionScratch {
    keys: Vec<usize>,
    values: Vec<usize>,
    generations: Vec<u32>,
    generation: u32,
}

impl DistinctPositionScratch {
    pub(crate) const fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            generations: Vec::new(),
            generation: 0,
        }
    }

    fn prepare(&mut self, weight: usize) -> Result<(), AkitaError> {
        let capacity = weight
            .checked_mul(4)
            .and_then(usize::checked_next_power_of_two)
            .ok_or_else(|| AkitaError::InvalidInput("sparse permutation size overflow".into()))?
            .max(4);
        if self.keys.len() >= capacity {
            return Ok(());
        }
        let additional = capacity
            .checked_sub(self.keys.len())
            .ok_or(AkitaError::InvalidProof)?;
        self.keys
            .try_reserve_exact(additional)
            .map_err(|_| AkitaError::InvalidInput("sparse permutation allocation failed".into()))?;
        self.values
            .try_reserve_exact(additional)
            .map_err(|_| AkitaError::InvalidInput("sparse permutation allocation failed".into()))?;
        self.generations
            .try_reserve_exact(additional)
            .map_err(|_| AkitaError::InvalidInput("sparse permutation allocation failed".into()))?;
        self.keys.resize(capacity, 0);
        self.values.resize(capacity, 0);
        self.generations.resize(capacity, 0);
        Ok(())
    }

    fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.generations.fill(0);
            self.generation = 1;
        }
    }

    fn find_slot(&self, key: usize) -> Result<(usize, bool), AkitaError> {
        let mask = self.keys.len().checked_sub(1).ok_or_else(|| {
            AkitaError::InvalidSetup("sparse permutation scratch is empty".into())
        })?;
        let mut slot = key & mask;
        for _ in 0..self.keys.len() {
            let occupied = self.generations.get(slot).copied().ok_or_else(|| {
                AkitaError::Internal("sparse permutation slot exceeds the generation map".into())
            })? == self.generation;
            if !occupied {
                return Ok((slot, false));
            }
            if self.keys.get(slot).copied() == Some(key) {
                return Ok((slot, true));
            }
            slot = slot.wrapping_add(1) & mask;
        }
        Err(AkitaError::InvalidSetup(
            "sparse permutation scratch capacity exhausted".into(),
        ))
    }

    fn get(&self, key: usize) -> Result<usize, AkitaError> {
        let (slot, occupied) = self.find_slot(key)?;
        if occupied {
            self.values.get(slot).copied().ok_or_else(|| {
                AkitaError::Internal("occupied sparse permutation slot has no value".into())
            })
        } else {
            Ok(key)
        }
    }

    fn insert(&mut self, key: usize, value: usize) -> Result<(), AkitaError> {
        let (slot, _) = self.find_slot(key)?;
        *self.keys.get_mut(slot).ok_or_else(|| {
            AkitaError::Internal("sparse permutation insertion key slot is absent".into())
        })? = key;
        *self.values.get_mut(slot).ok_or_else(|| {
            AkitaError::Internal("sparse permutation insertion value slot is absent".into())
        })? = value;
        *self.generations.get_mut(slot).ok_or_else(|| {
            AkitaError::Internal("sparse permutation insertion generation slot is absent".into())
        })? = self.generation;
        Ok(())
    }
}

/// Fisher-Yates partial shuffle: sample `out.len()` distinct values from
/// `0..universe` into `out`.
#[inline]
pub(crate) fn sample_distinct_positions_into(
    cursor: &mut XofCursor,
    universe: usize,
    out: &mut [u32],
    scratch: &mut DistinctPositionScratch,
) -> Result<(), AkitaError> {
    debug_assert!(out.len() <= universe);
    debug_assert!(universe <= MAX_STACK_RING_DIM);
    // A dense challenge already has output size Theta(universe), and the flat
    // permutation has the best concrete locality. Sparse production families
    // instead retain only the O(weight) swaps touched by partial Fisher-Yates.
    if universe > out.len().saturating_mul(64) {
        return sample_distinct_positions_into_sparse(cursor, universe, out, scratch);
    }
    match universe {
        0..=8 => sample_distinct_positions_into_stack::<u8, 8>(cursor, universe, out),
        9..=16 => sample_distinct_positions_into_stack::<u8, 16>(cursor, universe, out),
        17..=32 => sample_distinct_positions_into_stack::<u8, 32>(cursor, universe, out),
        33..=64 => sample_distinct_positions_into_stack::<u8, 64>(cursor, universe, out),
        65..=128 => sample_distinct_positions_into_stack::<u8, 128>(cursor, universe, out),
        129..=256 => sample_distinct_positions_into_stack::<u8, 256>(cursor, universe, out),
        257..=512 => sample_distinct_positions_into_stack::<u16, 512>(cursor, universe, out),
        513..=1024 => sample_distinct_positions_into_stack::<u16, 1024>(cursor, universe, out),
        1025..=MAX_STACK_TIER_RING_DIM => sample_distinct_positions_into_stack::<
            u16,
            MAX_STACK_TIER_RING_DIM,
        >(cursor, universe, out),
        _ => {
            return Err(AkitaError::InvalidInput(format!(
                "ring dimension must be <= {MAX_STACK_RING_DIM}"
            )))
        }
    }?;
    Ok(())
}

#[inline]
fn sample_distinct_positions_into_sparse(
    cursor: &mut XofCursor,
    universe: usize,
    out: &mut [u32],
    scratch: &mut DistinctPositionScratch,
) -> Result<(), AkitaError> {
    scratch.prepare(out.len())?;
    scratch.clear();
    for (i, dst) in out.iter_mut().enumerate() {
        let j = i + cursor.next_usize_mod(universe - i);
        let left = scratch.get(i)?;
        let right = scratch.get(j)?;
        scratch.insert(i, right)?;
        scratch.insert(j, left)?;
        *dst = u32::try_from(right).map_err(|_| {
            AkitaError::Internal("sampled sparse permutation position does not fit u32".into())
        })?;
    }
    Ok(())
}

#[inline]
fn sample_distinct_positions_into_stack<T, const N: usize>(
    cursor: &mut XofCursor,
    universe: usize,
    out: &mut [u32],
) -> Result<(), AkitaError>
where
    T: Copy + Default + TryFrom<usize>,
    u32: From<T>,
{
    if out.len() > universe || universe > N || N > MAX_STACK_TIER_RING_DIM {
        return Err(AkitaError::InvalidInput(
            "permutation tier has invalid dimensions".into(),
        ));
    }
    let mut perm = [T::default(); N];
    for (i, slot) in perm[..universe].iter_mut().enumerate() {
        *slot = T::try_from(i).map_err(|_| {
            AkitaError::Internal("stack permutation position does not fit its tier type".into())
        })?;
    }
    for (i, dst) in out.iter_mut().enumerate() {
        let j = i + cursor.next_usize_mod(universe - i);
        perm.swap(i, j);
        *dst = u32::from(perm[i]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_virtual_permutation_matches_dense_partial_fisher_yates() {
        for (universe, weight) in [(128usize, 7usize), (256, 23), (1024, 16), (2048, 14)] {
            let seed = [universe.trailing_zeros() as u8; 32];
            let prefix = crate::sampler::IndexedXofPrefix::new(&seed).unwrap();
            let mut dense_cursor = XofCursor::from_indexed_prefix(&prefix, 0);
            let mut sparse_cursor = XofCursor::from_indexed_prefix(&prefix, 0);
            let mut dense = vec![0u32; weight];
            let mut sparse = vec![0u32; weight];
            let mut scratch = DistinctPositionScratch::new();
            for round in 0..16 {
                match universe {
                    128 => sample_distinct_positions_into_stack::<u8, 128>(
                        &mut dense_cursor,
                        universe,
                        &mut dense,
                    ),
                    256 => sample_distinct_positions_into_stack::<u8, 256>(
                        &mut dense_cursor,
                        universe,
                        &mut dense,
                    ),
                    1024 => sample_distinct_positions_into_stack::<u16, 1024>(
                        &mut dense_cursor,
                        universe,
                        &mut dense,
                    ),
                    2048 => sample_distinct_positions_into_stack::<u16, 2048>(
                        &mut dense_cursor,
                        universe,
                        &mut dense,
                    ),
                    _ => unreachable!(),
                }
                .unwrap();
                sample_distinct_positions_into_sparse(
                    &mut sparse_cursor,
                    universe,
                    &mut sparse,
                    &mut scratch,
                )
                .unwrap();
                assert_eq!(
                    sparse, dense,
                    "universe={universe}, weight={weight}, round={round}"
                );
            }
        }
    }
}
