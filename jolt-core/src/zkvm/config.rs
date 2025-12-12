use crate::field::JoltField;
use crate::utils::math::Math;
use crate::zkvm::instruction_lookups::LOG_K;

/// Helper to get log_k_chunk based on log_T
#[inline]
pub const fn get_log_k_chunk(log_T: usize) -> usize {
    // TODO: Determine best point to switch based on empirical data.
    if log_T < 23 {
        4
    } else {
        8
    }
}

pub const fn get_lookups_ra_virtual_log_k_chunk(log_T: usize) -> usize {
    // TODO: Determine best point to switch based on empirical data.
    if log_T < 23 {
        LOG_K / 8
    } else {
        LOG_K / 4
    }
}

/// Compute the number of phases for instruction lookups based on trace length.
/// For traces below 2^23 cycles we want to use 16 phases, otherwise 8.
/// TODO: explore using other number of phases
/// NOTE: currently only divisors of 128 are supported
#[inline]
pub const fn instruction_sumcheck_phases(log_T: usize) -> usize {
    if log_T < 23 {
        16
    } else {
        8
    }
}

/// Helper to compute d (number of chunks) from log_k and log_k_chunk.
#[inline]
fn compute_d(log_k: usize, log_chunk: usize) -> usize {
    log_k.div_ceil(log_chunk)
}

#[derive(Clone, Debug)]
pub struct OneHotParams {
    pub log_k_chunk: usize,
    pub lookups_ra_virtual_log_k_chunk: usize,
    pub k_chunk: usize,

    pub bytecode_k: usize,
    pub ram_k: usize,

    pub instruction_d: usize,
    pub bytecode_d: usize,
    pub ram_d: usize,

    instruction_shifts: Vec<usize>,
    ram_shifts: Vec<usize>,
    bytecode_shifts: Vec<usize>,
}

impl OneHotParams {
    // TODO: Should check the params are valid. Return a Result.
    pub fn new(log_T: usize, bytecode_k: usize, ram_k: usize) -> Self {
        let log_k_chunk = get_log_k_chunk(log_T);
        let lookups_ra_virtual_log_k_chunk = get_lookups_ra_virtual_log_k_chunk(log_T);
        Self::new_with_log_k_chunk(
            log_k_chunk,
            lookups_ra_virtual_log_k_chunk,
            bytecode_k,
            ram_k,
        )
    }

    // TODO: Should check the params are valid. Return a Result.
    pub fn new_with_log_k_chunk(
        log_k_chunk: usize,
        lookups_ra_virtual_log_k_chunk: usize,
        bytecode_k: usize,
        ram_k: usize,
    ) -> Self {
        let instruction_d = compute_d(LOG_K, log_k_chunk);
        let bytecode_d = compute_d(bytecode_k.log_2(), log_k_chunk);
        let ram_d = compute_d(ram_k.log_2(), log_k_chunk);

        let instruction_shifts = (0..instruction_d)
            .map(|i| log_k_chunk * (instruction_d - 1 - i))
            .collect();
        let ram_shifts = (0..ram_d).map(|i| log_k_chunk * (ram_d - 1 - i)).collect();
        let bytecode_shifts = (0..bytecode_d)
            .map(|i| log_k_chunk * (bytecode_d - 1 - i))
            .collect();

        Self {
            log_k_chunk,
            lookups_ra_virtual_log_k_chunk,
            k_chunk: 1 << log_k_chunk,
            bytecode_k,
            ram_k,
            instruction_d,
            bytecode_d,
            ram_d,
            instruction_shifts,
            ram_shifts,
            bytecode_shifts,
        }
    }

    pub fn ram_address_chunk(&self, address: u64, idx: usize) -> u16 {
        ((address >> self.ram_shifts[idx]) % self.k_chunk as u64) as u16
    }

    pub fn bytecode_pc_chunk(&self, pc: usize, idx: usize) -> u16 {
        ((pc >> self.bytecode_shifts[idx]) % self.k_chunk) as u16
    }

    pub fn lookup_index_chunk(&self, index: u128, idx: usize) -> u16 {
        ((index >> self.instruction_shifts[idx]) % self.k_chunk as u128) as u16
    }

    pub fn compute_r_address_chunks<F: JoltField>(
        &self,
        r_address: &[F::Challenge],
    ) -> Vec<Vec<F::Challenge>> {
        let r_address = if r_address.len().is_multiple_of(self.log_k_chunk) {
            r_address.to_vec()
        } else {
            [
                &vec![
                    F::Challenge::from(0_u128);
                    self.log_k_chunk - (r_address.len() % self.log_k_chunk)
                ],
                r_address,
            ]
            .concat()
        };

        let r_address_chunks: Vec<Vec<F::Challenge>> = r_address
            .chunks(self.log_k_chunk)
            .map(|chunk| chunk.to_vec())
            .collect();

        r_address_chunks
    }
}
