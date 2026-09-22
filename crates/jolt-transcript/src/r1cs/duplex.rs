//! The 64-bit spongefish `Hash<Blake2b512>` byte relation, pinned at d2d190b.
//!
//! Source: `spongefish/src/instantiations/hash.rs`. All operation lengths and
//! transitions are public circuit shape. ONE must be fixed externally to one;
//! byte handles must belong to the same builder. Returned bytes must be bound
//! by a consumer. This module does not decode field challenges.

use jolt_field::Fr;
use jolt_r1cs::{bn254_bits::ByteVar, R1csBuilder};

use super::{blake2b512, Blake2bR1csError};

#[derive(Clone, Debug)]
enum Mode {
    Start,
    Absorb(Vec<ByteVar>),
    Squeeze {
        consumed: u64,
        leftovers: Vec<ByteVar>,
    },
}

/// Constrained spongefish BLAKE2b-512 duplex state for a 64-bit native verifier.
///
/// Absorbs concatenate until squeezing or ratcheting. Squeezes stream across
/// calls. Empty operations retain the upstream transition semantics. Counts
/// use eight big-endian bytes, matching upstream's `usize` on 64-bit targets.
/// Schedules that would overflow upstream's rounded byte count are rejected.
#[derive(Clone, Debug)]
pub struct Blake2bDuplexVar {
    cv: [ByteVar; 64],
    mode: Mode,
}

impl Default for Blake2bDuplexVar {
    fn default() -> Self {
        Self {
            cv: std::array::from_fn(|_| ByteVar::constant(0)),
            mode: Mode::Start,
        }
    }
}

impl Blake2bDuplexVar {
    fn framed(&self, tag: u8) -> Vec<ByteVar> {
        let mut message = vec![ByteVar::constant(0); 127];
        message.push(ByteVar::constant(tag));
        message.extend(self.cv.iter().cloned());
        message
    }

    fn squeeze_end(&mut self, builder: &mut R1csBuilder<Fr>) -> Result<(), Blake2bR1csError> {
        if let Mode::Squeeze { consumed, .. } = &self.mode {
            let mut message = self.framed(2);
            message.extend(consumed.to_be_bytes().map(ByteVar::constant));
            self.cv = blake2b512(builder, &message)?;
            self.mode = Mode::Start;
        }
        Ok(())
    }

    /// Absorb bytes, ending any squeeze run with its exact consumed-byte count.
    pub fn absorb(
        &mut self,
        builder: &mut R1csBuilder<Fr>,
        input: &[ByteVar],
    ) -> Result<(), Blake2bR1csError> {
        for byte in input {
            byte.validate_indices(builder)?;
        }
        self.squeeze_end(builder)?;
        if matches!(self.mode, Mode::Start) {
            self.mode = Mode::Absorb(self.framed(0));
        }
        if let Mode::Absorb(message) = &mut self.mode {
            let _ = message
                .len()
                .checked_add(input.len())
                .ok_or(Blake2bR1csError::MessageTooLong)?;
            message.extend_from_slice(input);
        }
        Ok(())
    }

    /// Match upstream's double-hash ratchet, including its empty-hasher case.
    ///
    /// Outside an absorb run, upstream finalizes an empty hasher after ending
    /// squeezing, so this intentionally resets cv to H(H(empty)).
    pub fn ratchet(&mut self, builder: &mut R1csBuilder<Fr>) -> Result<(), Blake2bR1csError> {
        self.squeeze_end(builder)?;
        let message = match &self.mode {
            Mode::Absorb(message) => message.as_slice(),
            Mode::Start | Mode::Squeeze { .. } => &[],
        };
        let inner = blake2b512(builder, message)?;
        self.cv = blake2b512(builder, &inner)?;
        self.mode = Mode::Start;
        Ok(())
    }

    /// Squeeze exactly `len` bytes; zero length still ends an absorb run.
    pub fn squeeze(
        &mut self,
        builder: &mut R1csBuilder<Fr>,
        len: usize,
    ) -> Result<Vec<ByteVar>, Blake2bR1csError> {
        let consumed = match &self.mode {
            Mode::Squeeze { consumed, .. } => *consumed,
            Mode::Start | Mode::Absorb(_) => 0,
        };
        let len64 = u64::try_from(len).map_err(|_| Blake2bR1csError::MessageTooLong)?;
        let _ = consumed
            .checked_add(len64)
            .filter(|&end| end <= u64::MAX - 63)
            .ok_or(Blake2bR1csError::MessageTooLong)?;
        if matches!(self.mode, Mode::Absorb(_)) {
            self.ratchet(builder)?;
        }
        if matches!(self.mode, Mode::Start) {
            self.mode = Mode::Squeeze {
                consumed: 0,
                leftovers: Vec::new(),
            };
        }
        let mut output = Vec::with_capacity(len);
        while output.len() < len {
            let Mode::Squeeze {
                consumed,
                leftovers,
            } = &self.mode
            else {
                return Err(Blake2bR1csError::InvalidLength);
            };
            if leftovers.is_empty() {
                let mut message = self.framed(1);
                message.extend((consumed / 64).to_be_bytes().map(ByteVar::constant));
                let block = blake2b512(builder, &message)?;
                if let Mode::Squeeze { leftovers, .. } = &mut self.mode {
                    leftovers.extend(block);
                }
            }
            if let Mode::Squeeze {
                consumed,
                leftovers,
            } = &mut self.mode
            {
                let take = (len - output.len()).min(leftovers.len());
                output.extend(leftovers.drain(..take));
                *consumed += take as u64;
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test vectors and completed-witness mutations"
)]
mod tests {
    use super::*;
    use jolt_field::Ring;
    use jolt_r1cs::{LinearCombination, Variable};
    use spongefish::{instantiations::Blake2b512, DuplexSpongeInterface};

    enum Op {
        Absorb(Vec<u8>),
        Squeeze(usize),
        Ratchet,
    }

    fn check(ops: &[Op]) {
        let mut native = Blake2b512::default();
        let mut circuit = Blake2bDuplexVar::default();
        let mut builder = R1csBuilder::new();
        let mut mutation_targets = Vec::new();
        for op in ops {
            match op {
                Op::Absorb(input) => {
                    let _ = native.absorb(input);
                    let bytes: Vec<_> = input
                        .iter()
                        .map(|&b| ByteVar::allocate(&mut builder, Some(b)))
                        .collect();
                    if let Some(byte) = bytes.first() {
                        mutation_targets.push(byte.clone());
                    }
                    circuit.absorb(&mut builder, &bytes).unwrap();
                }
                Op::Squeeze(len) => {
                    let mut expected = vec![0; *len];
                    let _ = native.squeeze(&mut expected);
                    let bytes = circuit.squeeze(&mut builder, *len).unwrap();
                    for (byte, value) in bytes.iter().zip(expected) {
                        builder.assert_equal(
                            byte.expression(),
                            LinearCombination::constant(Fr::from_u64(u64::from(value))),
                        );
                    }
                    if let Some(byte) = bytes.first() {
                        mutation_targets.push(byte.clone());
                    }
                }
                Op::Ratchet => {
                    let _ = native.ratchet();
                    circuit.ratchet(&mut builder).unwrap();
                }
            }
        }
        let witness = builder.witness().unwrap();
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());
        for byte in mutation_targets {
            let variable = byte
                .expression()
                .terms
                .iter()
                .find(|(v, _)| *v != Variable::ONE)
                .unwrap()
                .0;
            let mut changed = witness.clone();
            changed[variable.index()] = Fr::from_u64(1) - changed[variable.index()];
            assert!(matrices.check_witness(&changed).is_err());
        }
    }

    #[test]
    fn native_streaming_boundaries_and_squeeze_end_count() {
        check(&[
            Op::Absorb(vec![7; 63]),
            Op::Absorb(vec![9; 66]),
            Op::Squeeze(0),
            Op::Squeeze(1),
            Op::Squeeze(63),
            Op::Squeeze(65),
            Op::Absorb(vec![]),
            Op::Absorb(vec![4]),
            Op::Squeeze(33),
        ]);
    }

    #[test]
    fn native_empty_operations_and_explicit_ratchets() {
        check(&[
            Op::Squeeze(0),
            Op::Absorb(vec![]),
            Op::Squeeze(1),
            Op::Ratchet,
            Op::Squeeze(1),
            Op::Ratchet,
            Op::Ratchet,
            Op::Absorb(vec![]),
            Op::Ratchet,
            Op::Squeeze(1),
        ]);
    }

    #[test]
    fn unknown_witness_preserves_duplex_shape() {
        let emit = |value| {
            let mut builder = R1csBuilder::new();
            let mut circuit = Blake2bDuplexVar::default();
            let input = ByteVar::allocate(&mut builder, value);
            circuit.absorb(&mut builder, &[input]).unwrap();
            let _ = circuit.squeeze(&mut builder, 1).unwrap();
            builder.into_matrices()
        };
        let known = emit(Some(123));
        let unknown = emit(None);
        assert_eq!(known.num_vars, unknown.num_vars);
        assert_eq!(known.a, unknown.a);
        assert_eq!(known.b, unknown.b);
        assert_eq!(known.c, unknown.c);
    }

    #[test]
    fn overflow_rejected_before_emission() {
        let mut builder = R1csBuilder::new();
        let mut circuit = Blake2bDuplexVar {
            mode: Mode::Squeeze {
                consumed: u64::MAX - 63,
                leftovers: Vec::new(),
            },
            ..Blake2bDuplexVar::default()
        };
        assert!(matches!(
            circuit.squeeze(&mut builder, 1),
            Err(Blake2bR1csError::MessageTooLong)
        ));
        assert_eq!(builder.num_vars(), 1);
    }
}
