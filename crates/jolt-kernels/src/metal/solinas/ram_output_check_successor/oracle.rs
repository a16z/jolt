//! Representation-independent relation and dispatch oracles.

use jolt_field::Field;

use super::abi::{
    TARGET_ADDRESSES, TARGET_BLOCKS, TARGET_BLOCK_ELEMENTS, TARGET_CHALLENGES,
    TARGET_CHUNKS_PER_BLOCK, TARGET_PARTIALS, TARGET_THREADS,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OracleError {
    InvalidLength {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    InvalidDomain,
    InvalidMask,
    UncertifiedPublicIo,
    NonZeroDeferredMessage {
        round: usize,
    },
    ClaimMismatch {
        round: usize,
    },
    FullyBound,
}

pub fn low_binding_weights<F: Field>(challenges: &[F]) -> Vec<F> {
    let mut weights = vec![F::one()];
    for &challenge in challenges {
        let old_len = weights.len();
        let mut next = vec![F::zero(); 2 * old_len];
        for (index, &weight) in weights.iter().enumerate() {
            let high = weight * challenge;
            next[index] = weight - high;
            next[index + old_len] = high;
        }
        weights = next;
    }
    weights
}

/// Computes one low-bit equality coefficient without using the table builder.
pub fn direct_weight<F: Field>(low_index: usize, challenges: &[F]) -> F {
    challenges
        .iter()
        .enumerate()
        .fold(F::one(), |weight, (bit, &challenge)| {
            let factor = if (low_index >> bit) & 1 == 0 {
                F::one() - challenge
            } else {
                challenge
            };
            weight * factor
        })
}

pub fn chunk_partials_host_weights<F: Field>(
    values: &[u64],
    weights: &[F],
) -> Result<Vec<F>, OracleError> {
    validate_target_values(values)?;
    require_length("low weights", weights.len(), TARGET_BLOCK_ELEMENTS as usize)?;
    let mut partials = vec![F::zero(); TARGET_PARTIALS as usize];
    for block in 0..TARGET_BLOCKS as usize {
        for chunk in 0..TARGET_CHUNKS_PER_BLOCK as usize {
            let low_start = chunk * TARGET_THREADS as usize;
            let source_start = block * TARGET_BLOCK_ELEMENTS as usize + low_start;
            let mut sum = F::zero();
            for offset in 0..TARGET_THREADS as usize {
                sum += F::from_u64(values[source_start + offset]) * weights[low_start + offset];
            }
            partials[block * TARGET_CHUNKS_PER_BLOCK as usize + chunk] = sum;
        }
    }
    Ok(partials)
}

pub fn chunk_partials_device_weights<F: Field>(
    values: &[u64],
    challenges: &[F],
) -> Result<Vec<F>, OracleError> {
    validate_target_values(values)?;
    require_length(
        "prefix challenges",
        challenges.len(),
        TARGET_CHALLENGES as usize,
    )?;
    let mut partials = vec![F::zero(); TARGET_PARTIALS as usize];
    for block in 0..TARGET_BLOCKS as usize {
        for chunk in 0..TARGET_CHUNKS_PER_BLOCK as usize {
            let low_start = chunk * TARGET_THREADS as usize;
            let source_start = block * TARGET_BLOCK_ELEMENTS as usize + low_start;
            let mut sum = F::zero();
            for offset in 0..TARGET_THREADS as usize {
                let low = low_start + offset;
                sum += F::from_u64(values[source_start + offset]) * direct_weight(low, challenges);
            }
            partials[block * TARGET_CHUNKS_PER_BLOCK as usize + chunk] = sum;
        }
    }
    Ok(partials)
}

pub fn reduce_partials<F: Field>(partials: &[F]) -> Result<[F; 8], OracleError> {
    require_length("chunk partials", partials.len(), TARGET_PARTIALS as usize)?;
    Ok(std::array::from_fn(|block| {
        let start = block * TARGET_CHUNKS_PER_BLOCK as usize;
        partials[start..start + TARGET_CHUNKS_PER_BLOCK as usize]
            .iter()
            .fold(F::zero(), |sum, &value| sum + value)
    }))
}

pub fn fold_native_blocks<F: Field>(
    values: &[u64],
    challenges: &[F],
) -> Result<[F; 8], OracleError> {
    let weights = low_binding_weights(challenges);
    reduce_partials(&chunk_partials_host_weights(values, &weights)?)
}

/// A direct four-table implementation of the symbolic degree-three relation.
#[derive(Clone, Debug)]
pub struct DenseOracle<F: Field> {
    eq_address: Vec<F>,
    io_mask: Vec<F>,
    val_io: Vec<F>,
    val_final: Vec<F>,
    round: usize,
}

impl<F: Field> DenseOracle<F> {
    pub fn new(
        output_address: &[F],
        mask_start: usize,
        mask_end: usize,
        val_io: &[u64],
        val_final: &[u64],
    ) -> Result<Self, OracleError> {
        let addresses = val_final.len();
        if addresses == 0
            || !addresses.is_power_of_two()
            || output_address.len() != addresses.ilog2() as usize
        {
            return Err(OracleError::InvalidDomain);
        }
        require_length("public IO", val_io.len(), addresses)?;
        if mask_start >= mask_end || mask_end > addresses {
            return Err(OracleError::InvalidMask);
        }
        Ok(Self {
            eq_address: equality_table(output_address),
            io_mask: (0..addresses)
                .map(|index| {
                    if index >= mask_start && index < mask_end {
                        F::one()
                    } else {
                        F::zero()
                    }
                })
                .collect(),
            val_io: val_io.iter().map(|&value| F::from_u64(value)).collect(),
            val_final: val_final.iter().map(|&value| F::from_u64(value)).collect(),
            round: 0,
        })
    }

    pub fn message_evals(&self) -> Result<[F; 4], OracleError> {
        if self.eq_address.len() < 2 {
            return Err(OracleError::FullyBound);
        }
        Ok(std::array::from_fn(|node| {
            let t = F::from_u64(node as u64);
            (0..self.eq_address.len() / 2).fold(F::zero(), |sum, pair| {
                let eq = extend_pair(&self.eq_address, pair, t);
                let mask = extend_pair(&self.io_mask, pair, t);
                let val_io = extend_pair(&self.val_io, pair, t);
                let val_final = extend_pair(&self.val_final, pair, t);
                sum + eq * mask * (val_final - val_io)
            })
        }))
    }

    pub fn checked_message(&self, previous_claim: F) -> Result<[F; 4], OracleError> {
        let evals = self.message_evals()?;
        if evals[0] + evals[1] != previous_claim {
            return Err(OracleError::ClaimMismatch { round: self.round });
        }
        Ok(evals)
    }

    pub fn bind(&mut self, challenge: F) -> Result<(), OracleError> {
        if self.eq_address.len() < 2 {
            return Err(OracleError::FullyBound);
        }
        for table in [
            &mut self.eq_address,
            &mut self.io_mask,
            &mut self.val_io,
            &mut self.val_final,
        ] {
            bind_low(table, challenge);
        }
        self.round += 1;
        Ok(())
    }

    pub fn defer_zero_prefix(&mut self, challenges: &[F]) -> Result<(), OracleError> {
        for &challenge in challenges {
            if self
                .message_evals()?
                .iter()
                .any(|&value| value != F::zero())
            {
                return Err(OracleError::NonZeroDeferredMessage { round: self.round });
            }
            self.bind(challenge)?;
        }
        Ok(())
    }

    pub fn bound_values(&self) -> Option<(F, F, F, F)> {
        (self.eq_address.len() == 1).then(|| {
            (
                self.eq_address[0],
                self.io_mask[0],
                self.val_io[0],
                self.val_final[0],
            )
        })
    }

    pub fn val_final_table(&self) -> &[F] {
        &self.val_final
    }
}

/// The eight-element host tail reconstructed from the certified native fold.
#[derive(Clone, Debug)]
pub struct SuccessorTail<F: Field> {
    eq_address: Vec<F>,
    io_mask: Vec<F>,
    val_io: Vec<F>,
    val_final: Vec<F>,
    round: usize,
}

impl<F: Field> SuccessorTail<F> {
    pub fn new(
        output_address: &[F],
        prefix_challenges: &[F],
        folded_final: [F; 8],
        public_io_certified: bool,
    ) -> Result<Self, OracleError> {
        require_length("output address", output_address.len(), 13)?;
        require_length(
            "prefix challenges",
            prefix_challenges.len(),
            TARGET_CHALLENGES as usize,
        )?;
        if !public_io_certified {
            return Err(OracleError::UncertifiedPublicIo);
        }
        let low_eq = equality_evaluation(
            &output_address[..TARGET_CHALLENGES as usize],
            prefix_challenges,
        )?;
        let mut eq_address = equality_table(&output_address[TARGET_CHALLENGES as usize..]);
        for value in &mut eq_address {
            *value *= low_eq;
        }
        let val_final = folded_final.to_vec();
        let io_mask = (0..TARGET_BLOCKS as usize)
            .map(|block| {
                if (1..4).contains(&block) {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect::<Vec<_>>();
        let val_io = folded_final
            .into_iter()
            .enumerate()
            .map(|(block, value)| {
                if (1..4).contains(&block) {
                    value
                } else {
                    F::zero()
                }
            })
            .collect();
        Ok(Self {
            eq_address,
            io_mask,
            val_io,
            val_final,
            round: TARGET_CHALLENGES as usize,
        })
    }

    pub fn message_evals(&self) -> Result<[F; 4], OracleError> {
        DenseOracle {
            eq_address: self.eq_address.clone(),
            io_mask: self.io_mask.clone(),
            val_io: self.val_io.clone(),
            val_final: self.val_final.clone(),
            round: self.round,
        }
        .message_evals()
    }

    pub fn checked_message(&self, previous_claim: F) -> Result<[F; 4], OracleError> {
        let evals = self.message_evals()?;
        if evals[0] + evals[1] != previous_claim {
            return Err(OracleError::ClaimMismatch { round: self.round });
        }
        Ok(evals)
    }

    pub fn bind(&mut self, challenge: F) -> Result<(), OracleError> {
        if self.eq_address.len() < 2 {
            return Err(OracleError::FullyBound);
        }
        for table in [
            &mut self.eq_address,
            &mut self.io_mask,
            &mut self.val_io,
            &mut self.val_final,
        ] {
            bind_low(table, challenge);
        }
        self.round += 1;
        Ok(())
    }

    pub fn bound_values(&self) -> Option<(F, F, F, F)> {
        (self.eq_address.len() == 1).then(|| {
            (
                self.eq_address[0],
                self.io_mask[0],
                self.val_io[0],
                self.val_final[0],
            )
        })
    }
}

pub fn equality_table<F: Field>(point: &[F]) -> Vec<F> {
    let mut table = vec![F::one()];
    for &challenge in point {
        let old_len = table.len();
        let mut next = vec![F::zero(); 2 * old_len];
        for (index, &value) in table.iter().enumerate() {
            let high = value * challenge;
            next[index] = value - high;
            next[index + old_len] = high;
        }
        table = next;
    }
    table
}

pub fn equality_evaluation<F: Field>(left: &[F], right: &[F]) -> Result<F, OracleError> {
    require_length("equality right point", right.len(), left.len())?;
    Ok(left
        .iter()
        .zip(right)
        .fold(F::one(), |value, (&left, &right)| {
            value * (left * right + (F::one() - left) * (F::one() - right))
        }))
}

fn validate_target_values(values: &[u64]) -> Result<(), OracleError> {
    require_length(
        "native RamValFinal",
        values.len(),
        TARGET_ADDRESSES as usize,
    )
}

fn require_length(name: &'static str, got: usize, expected: usize) -> Result<(), OracleError> {
    if got != expected {
        return Err(OracleError::InvalidLength {
            name,
            expected,
            got,
        });
    }
    Ok(())
}

fn extend_pair<F: Field>(table: &[F], pair: usize, t: F) -> F {
    let low = table[2 * pair];
    low + t * (table[2 * pair + 1] - low)
}

fn bind_low<F: Field>(table: &mut Vec<F>, challenge: F) {
    let bound_len = table.len() / 2;
    for pair in 0..bound_len {
        table[pair] = extend_pair(table, pair, challenge);
    }
    table.truncate(bound_len);
}
