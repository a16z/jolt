//! Boolean bytes and 64-bit modular words over BN254 R1CS.
//!
//! XOR is x+y-2xy for Boolean operands. Word addition constrains 64 output
//! bits, bounded carry bits and the whole integer sum; all residuals are
//! below 2^67, far below the BN254 scalar modulus. The proof statement must
//! fix the constant-one column. Handles must stay in their allocating builder.
use jolt_field::{Fr, Ring};
use thiserror::Error;

use crate::{LinearCombination, R1csBuilder, Variable};

/// Invalid use of a bit expression with a smaller constraint builder.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum BitsError {
    /// At least one referenced coordinate is outside the supplied builder.
    #[error("bit expression references out-of-range variable {variable:?}")]
    UnknownVariable { variable: Variable },
}

#[derive(Clone, Debug)]
struct Bit {
    expression: LinearCombination<Fr>,
    witness: Option<bool>,
}

impl Bit {
    fn allocate(builder: &mut R1csBuilder<Fr>, witness: Option<bool>) -> Self {
        let variable = builder.alloc_witness(witness.map(|b| Fr::from_u64(u64::from(b))));
        let expression = LinearCombination::variable(variable);
        builder.assert_product(
            expression.clone(),
            expression.clone() - LinearCombination::one(),
            LinearCombination::zero(),
        );
        Self {
            expression,
            witness,
        }
    }

    fn constant(value: bool) -> Self {
        Self {
            expression: LinearCombination::constant(Fr::from_u64(u64::from(value))),
            witness: Some(value),
        }
    }

    fn xor(&self, builder: &mut R1csBuilder<Fr>, rhs: &Self) -> Self {
        let witness = self.witness.zip(rhs.witness).map(|(a, b)| a ^ b);
        let variable = builder.alloc_witness(witness.map(|b| Fr::from_u64(u64::from(b))));
        let expression = LinearCombination::variable(variable);
        builder.assert_product(
            self.expression.clone().scale(Fr::from_u64(2)),
            rhs.expression.clone(),
            self.expression.clone() + rhs.expression.clone() - expression.clone(),
        );
        Self {
            expression,
            witness,
        }
    }

    fn validate_indices(&self, builder: &R1csBuilder<Fr>) -> Result<(), BitsError> {
        for &(variable, _) in &self.expression.terms {
            if variable.index() >= builder.num_vars() {
                return Err(BitsError::UnknownVariable { variable });
            }
        }
        Ok(())
    }
}

/// A byte whose little-endian bits are Boolean by construction or constraints.
#[derive(Clone, Debug)]
pub struct ByteVar {
    bits: [Bit; 8],
}

impl ByteVar {
    /// Allocate eight Boolean bits; `None` emits the same layout without assignment.
    pub fn allocate(builder: &mut R1csBuilder<Fr>, witness: Option<u8>) -> Self {
        Self {
            bits: std::array::from_fn(|i| {
                Bit::allocate(builder, witness.map(|x| (x >> i) & 1 != 0))
            }),
        }
    }

    /// A fixed byte, requiring no witness coordinates.
    pub fn constant(value: u8) -> Self {
        Self {
            bits: std::array::from_fn(|i| Bit::constant((value >> i) & 1 != 0)),
        }
    }

    /// Packed byte value; consumers can bind this to a public or private output.
    pub fn expression(&self) -> LinearCombination<Fr> {
        self.bits
            .iter()
            .enumerate()
            .fold(LinearCombination::zero(), |sum, (i, bit)| {
                sum + bit.expression.clone().scale(Fr::from_u64(1u64 << i))
            })
    }

    /// Check index bounds, not builder provenance.
    pub fn validate_indices(&self, builder: &R1csBuilder<Fr>) -> Result<(), BitsError> {
        for bit in &self.bits {
            bit.validate_indices(builder)?;
        }
        Ok(())
    }
}

/// A 64-bit word with constrained modular addition, XOR and bit permutations.
#[derive(Clone, Debug)]
pub struct Word64Var {
    bits: [Bit; 64],
}

impl Word64Var {
    /// A fixed word, requiring no witness coordinates.
    pub fn constant(value: u64) -> Self {
        Self {
            bits: std::array::from_fn(|i| Bit::constant((value >> i) & 1 != 0)),
        }
    }

    /// Reinterpret eight constrained bytes as a little-endian word.
    #[expect(
        clippy::indexing_slicing,
        reason = "i<64 maps into eight bytes of eight bits"
    )]
    pub fn from_le_bytes(bytes: &[ByteVar; 8]) -> Self {
        Self {
            bits: std::array::from_fn(|i| bytes[i / 8].bits[i % 8].clone()),
        }
    }

    /// Reinterpret the word as eight little-endian constrained bytes.
    #[expect(
        clippy::indexing_slicing,
        reason = "eight groups of eight bits cover the fixed 64-bit array"
    )]
    pub fn to_le_bytes(&self) -> [ByteVar; 8] {
        std::array::from_fn(|byte| ByteVar {
            bits: std::array::from_fn(|bit| self.bits[8 * byte + bit].clone()),
        })
    }

    /// Permute bits using a public rotation count reduced modulo 64.
    #[expect(
        clippy::indexing_slicing,
        reason = "rotation indices are reduced modulo the fixed 64-bit width"
    )]
    pub fn rotate_right(&self, count: usize) -> Self {
        let count = count % 64;
        Self {
            bits: std::array::from_fn(|i| self.bits[(i + count) % 64].clone()),
        }
    }

    /// Constrain XOR without adding redundant Boolean rows for derived bits.
    #[expect(
        clippy::indexing_slicing,
        reason = "both operands are fixed 64-bit arrays and i<64"
    )]
    pub fn xor(&self, builder: &mut R1csBuilder<Fr>, rhs: &Self) -> Result<Self, BitsError> {
        self.validate_indices(builder)?;
        rhs.validate_indices(builder)?;
        Ok(Self {
            bits: std::array::from_fn(|i| self.bits[i].xor(builder, &rhs.bits[i])),
        })
    }

    /// Add two words modulo 2^64 with a constrained one-bit carry.
    pub fn add(&self, builder: &mut R1csBuilder<Fr>, rhs: &Self) -> Result<Self, BitsError> {
        Self::add_words(builder, &[self, rhs], 1)
    }

    /// Add three words modulo 2^64 with a constrained two-bit carry.
    pub fn add_three(
        &self,
        builder: &mut R1csBuilder<Fr>,
        b: &Self,
        c: &Self,
    ) -> Result<Self, BitsError> {
        Self::add_words(builder, &[self, b, c], 2)
    }

    fn add_words(
        builder: &mut R1csBuilder<Fr>,
        words: &[&Self],
        carry_bits: usize,
    ) -> Result<Self, BitsError> {
        for word in words {
            word.validate_indices(builder)?;
        }
        let sum = words
            .iter()
            .try_fold(0u128, |sum, word| word.value().map(|x| sum + u128::from(x)));
        let output = Self {
            bits: std::array::from_fn(|i| Bit::allocate(builder, sum.map(|x| (x >> i) & 1 != 0))),
        };
        let mut carry = LinearCombination::zero();
        for i in 0..carry_bits {
            let bit = Bit::allocate(builder, sum.map(|x| (x >> (64 + i)) & 1 != 0));
            carry = carry + bit.expression.scale(Fr::from_u128(1u128 << (64 + i)));
        }
        let input = words.iter().fold(LinearCombination::zero(), |sum, word| {
            sum + word.expression()
        });
        builder.assert_equal(input, output.expression() + carry);
        Ok(output)
    }

    fn value(&self) -> Option<u64> {
        self.bits.iter().enumerate().try_fold(0, |sum, (i, bit)| {
            bit.witness.map(|b| sum | (u64::from(b) << i))
        })
    }

    fn expression(&self) -> LinearCombination<Fr> {
        self.bits
            .iter()
            .enumerate()
            .fold(LinearCombination::zero(), |sum, (i, bit)| {
                sum + bit.expression.clone().scale(Fr::from_u64(1u64 << i))
            })
    }

    fn validate_indices(&self, builder: &R1csBuilder<Fr>) -> Result<(), BitsError> {
        for bit in &self.bits {
            bit.validate_indices(builder)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests inspect word assignments and tamper carry bits"
)]
mod tests {
    use super::*;

    #[test]
    fn additions_constrain_overflow_and_output_bits() {
        for third in [false, true] {
            let mut builder = R1csBuilder::new();
            let bytes = std::array::from_fn(|_| ByteVar::allocate(&mut builder, Some(255)));
            let word = Word64Var::from_le_bytes(&bytes);
            let output = if third {
                word.add_three(&mut builder, &word, &word).unwrap()
            } else {
                word.add(&mut builder, &word).unwrap()
            };
            let expected = if third { u64::MAX - 2 } else { u64::MAX - 1 };
            for (byte, value) in output.to_le_bytes().iter().zip(expected.to_le_bytes()) {
                builder.assert_equal(
                    byte.expression(),
                    LinearCombination::constant(Fr::from_u64(u64::from(value))),
                );
            }
            let witness = builder.witness().unwrap();
            let matrices = builder.into_matrices();
            assert!(matrices.check_witness(&witness).is_ok());
            let mut bad_carry = witness.clone();
            let index = bad_carry.len() - 1;
            bad_carry[index] = Fr::from_u64(1) - bad_carry[index];
            assert!(matrices.check_witness(&bad_carry).is_err());
            let mut nonboolean = witness;
            nonboolean[1] = Fr::from_u64(2);
            assert!(matrices.check_witness(&nonboolean).is_err());
        }
    }
}
