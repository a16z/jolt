//! Prover-side lattice (packed) witness assembly.
//!
//! Scatters every committed column into the single packed one-hot witness
//! `W` through `PrefixSlot::packed_index` — nothing is densified; `W` is
//! carried as its one-positions. Cell order within a one-hot column is
//! `(symbol ‖ cycle)` msb-first, matching the jolt-claims packing tests
//! (`lattice_semantics`), so `P_i(x) = W(prefix_i ‖ x)` holds per slot.

use jolt_claims::protocols::jolt::lattice::{
    LatticeColumn, UnsignedIncChunking, UNSIGNED_INC_BITS,
};
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_openings::PrefixPacking;

/// The per-cycle fused increment stream: the RAM delta on store cycles, the
/// rd delta otherwise (per-cycle disjointness is a public bytecode fact).
/// Padding cycles carry `delta = 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusedIncCycle {
    pub delta: i128,
}

impl FusedIncCycle {
    /// The shifted unsigned encoding `2^64 + delta`: the msb bit and the
    /// low-64-bit chunk symbols. Padding (`delta = 0`) encodes as msb hot
    /// with every chunk at symbol 0 (the jolt-claims padding invariant).
    fn shifted(self) -> u128 {
        debug_assert!(self.delta.unsigned_abs() < 1u128 << UNSIGNED_INC_BITS);
        (self.delta + (1i128 << UNSIGNED_INC_BITS)) as u128
    }

    pub fn msb(self) -> bool {
        self.shifted() >> UNSIGNED_INC_BITS == 1
    }

    pub fn chunk_symbol(self, chunking: UnsignedIncChunking, index: usize) -> usize {
        let low = self.shifted() & ((1u128 << UNSIGNED_INC_BITS) - 1);
        let width = chunking.chunk_width();
        ((low >> (width * index)) & ((1u128 << width) - 1)) as usize
    }
}

/// The packed witness as its one-positions (ascending packed index), plus
/// the packed arity — the sparse form both the Akita commit and the packed
/// reduction prover consume.
#[derive(Clone, Debug)]
pub struct PackedWitness {
    pub packed_num_vars: usize,
    pub one_positions: Vec<usize>,
}

/// Scatters the committed columns into `W`.
///
/// `ra_indices` carries, per committed `Ra` polynomial (in the packing's
/// column identity), the per-cycle hot address — `None` on padding cycles
/// (no access), matching the one-hot witness generation.
pub fn assemble_packed_witness(
    packing: &PrefixPacking<LatticeColumn>,
    chunking: UnsignedIncChunking,
    log_t: usize,
    ra_indices: &dyn Fn(JoltCommittedPolynomial) -> Option<Vec<Option<usize>>>,
    fused_inc: &[FusedIncCycle],
) -> Result<PackedWitness, String> {
    debug_assert!(fused_inc.len() <= 1 << log_t);
    let mut one_positions = Vec::new();
    for (column, slot) in packing {
        let LatticeColumn::Committed(polynomial) = column else {
            return Err(format!("non-committed column {column:?} in proof packing"));
        };
        match polynomial {
            JoltCommittedPolynomial::InstructionRa(_)
            | JoltCommittedPolynomial::BytecodeRa(_)
            | JoltCommittedPolynomial::RamRa(_) => {
                let indices = ra_indices(*polynomial)
                    .ok_or_else(|| format!("missing ra indices for {polynomial:?}"))?;
                for (cycle, address) in indices.iter().enumerate() {
                    let Some(address) = address else { continue };
                    one_positions.push(slot.packed_index((address << log_t) | cycle));
                }
            }
            JoltCommittedPolynomial::UnsignedIncChunk(index) => {
                for (cycle, inc) in fused_inc.iter().enumerate() {
                    let symbol = inc.chunk_symbol(chunking, *index);
                    one_positions.push(slot.packed_index((symbol << log_t) | cycle));
                }
            }
            JoltCommittedPolynomial::UnsignedIncMsb => {
                for (cycle, inc) in fused_inc.iter().enumerate() {
                    if inc.msb() {
                        one_positions.push(slot.packed_index(cycle));
                    }
                }
            }
            other => {
                return Err(format!(
                    "polynomial {other:?} is not part of the per-proof packed witness"
                ));
            }
        }
    }
    one_positions.sort_unstable();
    Ok(PackedWitness {
        packed_num_vars: packing.packed_num_vars,
        one_positions,
    })
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use jolt_claims::protocols::jolt::lattice::{proof_packing, ProofPackingShape};
    use jolt_field::{Fr, FromPrimitiveInt};

    const LOG_T: usize = 3;
    const LOG_K_CHUNK: usize = 8;

    fn packing() -> PrefixPacking<LatticeColumn> {
        proof_packing(&ProofPackingShape {
            ra_layout: JoltRaPolynomialLayout::new(1, 1, 1).unwrap(),
            log_t: LOG_T,
            log_k_chunk: LOG_K_CHUNK,
            untrusted_advice_word_vars: None,
        })
        .unwrap()
    }

    fn chunking() -> UnsignedIncChunking {
        UnsignedIncChunking::new(LOG_K_CHUNK).unwrap()
    }

    fn fused_trace() -> Vec<FusedIncCycle> {
        [7i128, -3, 0, (1 << 40) + 5, -(1 << 63), 1, -1, 0]
            .into_iter()
            .map(|delta| FusedIncCycle { delta })
            .collect()
    }

    fn witness() -> PackedWitness {
        let ra = |polynomial: JoltCommittedPolynomial| {
            let base = match polynomial {
                JoltCommittedPolynomial::InstructionRa(_) => 3usize,
                JoltCommittedPolynomial::BytecodeRa(_) => 5,
                JoltCommittedPolynomial::RamRa(_) => 7,
                _ => return None,
            };
            Some(
                (0..1 << LOG_T)
                    .map(|cycle| (cycle != 2).then_some((base * (cycle + 1)) % 256))
                    .collect(),
            )
        };
        assemble_packed_witness(&packing(), chunking(), LOG_T, &ra, &fused_trace()).unwrap()
    }

    fn dense(witness: &PackedWitness) -> Vec<Fr> {
        let mut evals = vec![Fr::from_u64(0); 1 << witness.packed_num_vars];
        for &position in &witness.one_positions {
            evals[position] = Fr::from_u64(1);
        }
        evals
    }

    #[test]
    fn positions_are_strictly_increasing_and_in_range() {
        let witness = witness();
        assert!(witness
            .one_positions
            .windows(2)
            .all(|pair| pair[0] < pair[1]));
        assert!(witness
            .one_positions
            .iter()
            .all(|&position| position >> witness.packed_num_vars == 0));
    }

    #[test]
    fn chunk_columns_reconstruct_the_shifted_fused_increment() {
        let witness = dense(&witness());
        let packing = packing();
        let chunking = chunking();
        let trace = fused_trace();
        for (cycle, inc) in trace.iter().enumerate() {
            // Hamming: exactly one hot symbol per chunk column and cycle.
            let mut reconstructed = 0u128;
            for index in 0..chunking.chunk_count() {
                let slot = &packing
                    [&LatticeColumn::Committed(JoltCommittedPolynomial::UnsignedIncChunk(index))];
                let hot: Vec<usize> = (0..1usize << LOG_K_CHUNK)
                    .filter(|&symbol| {
                        witness[slot.packed_index((symbol << LOG_T) | cycle)] == Fr::from_u64(1)
                    })
                    .collect();
                assert_eq!(hot.len(), 1, "chunk {index} cycle {cycle}");
                reconstructed |= (hot[0] as u128) << (chunking.chunk_width() * index);
            }
            let msb_slot =
                &packing[&LatticeColumn::Committed(JoltCommittedPolynomial::UnsignedIncMsb)];
            let msb = witness[msb_slot.packed_index(cycle)] == Fr::from_u64(1);
            reconstructed |= (msb as u128) << UNSIGNED_INC_BITS;
            assert_eq!(
                reconstructed as i128 - (1i128 << UNSIGNED_INC_BITS),
                inc.delta,
                "cycle {cycle}"
            );
        }
    }

    #[test]
    fn padding_cycles_encode_msb_hot_chunks_at_zero() {
        let padding = FusedIncCycle { delta: 0 };
        assert!(padding.msb());
        for index in 0..chunking().chunk_count() {
            assert_eq!(padding.chunk_symbol(chunking(), index), 0);
        }
    }

    #[test]
    fn ra_cells_land_in_their_slot_subcube() {
        let witness = dense(&witness());
        let packing = packing();
        let slot = &packing[&LatticeColumn::Committed(JoltCommittedPolynomial::InstructionRa(0))];
        for cycle in 0..1usize << LOG_T {
            let hot: Vec<usize> = (0..1usize << LOG_K_CHUNK)
                .filter(|&address| {
                    witness[slot.packed_index((address << LOG_T) | cycle)] == Fr::from_u64(1)
                })
                .collect();
            let expected = (cycle != 2).then_some((3 * (cycle + 1)) % 256);
            assert_eq!(
                hot,
                expected.into_iter().collect::<Vec<_>>(),
                "cycle {cycle}"
            );
        }
    }
}
