//! Shared lookup materialization semantics.
//!
//! A table runs the same semantic function with a concrete backend during
//! preprocessing and with a symbolic backend during extraction.

/// Operations available to a lookup materializer.
///
/// The interface contains the operations required by the currently certified
/// tables. New operations should be added only when another table needs them.
pub trait MaterializerBackend {
    type Bit: Clone;
    type Nat: Clone;
    type Output;

    fn input_bit(&mut self, index: usize) -> Self::Bit;
    fn and(&mut self, left: Self::Bit, right: Self::Bit) -> Self::Bit;
    fn not(&mut self, value: Self::Bit) -> Self::Bit;
    fn bit_to_nat(&mut self, value: Self::Bit) -> Self::Nat;
    fn nat_constant(&mut self, value: u128) -> Self::Nat;
    fn nat_add(&mut self, left: Self::Nat, right: Self::Nat) -> Self::Nat;
    fn nat_mul(&mut self, left: Self::Nat, right: Self::Nat) -> Self::Nat;
    fn bits_be<const N: usize>(&mut self, bits: [Self::Bit; N]) -> Self::Nat;
    fn output(&mut self, value: Self::Nat) -> Self::Output;
}

/// A lookup table whose materializer can run over any supported backend.
pub trait LookupMaterializer<const XLEN: usize> {
    fn materialize<B: MaterializerBackend>(&self, backend: &mut B) -> B::Output;
}

/// Concrete materializer backend for a `2 * XLEN` bit lookup index.
pub struct U128Materializer<const XLEN: usize> {
    index: u128,
}

impl<const XLEN: usize> U128Materializer<XLEN> {
    pub fn new(index: u128) -> Self {
        assert!(XLEN <= 64, "lookup words must fit in a u64");
        Self { index }
    }
}

impl<const XLEN: usize> MaterializerBackend for U128Materializer<XLEN> {
    type Bit = bool;
    type Nat = u128;
    type Output = u64;

    fn input_bit(&mut self, index: usize) -> Self::Bit {
        assert!(index < 2 * XLEN, "lookup input bit is out of range");
        let shift = 2 * XLEN - 1 - index;
        (self.index >> shift) & 1 == 1
    }

    fn and(&mut self, left: Self::Bit, right: Self::Bit) -> Self::Bit {
        left && right
    }

    fn not(&mut self, value: Self::Bit) -> Self::Bit {
        !value
    }

    fn bit_to_nat(&mut self, value: Self::Bit) -> Self::Nat {
        u128::from(value)
    }

    fn nat_constant(&mut self, value: u128) -> Self::Nat {
        value
    }

    fn nat_add(&mut self, left: Self::Nat, right: Self::Nat) -> Self::Nat {
        left + right
    }

    fn nat_mul(&mut self, left: Self::Nat, right: Self::Nat) -> Self::Nat {
        left * right
    }

    fn bits_be<const N: usize>(&mut self, bits: [Self::Bit; N]) -> Self::Nat {
        assert!(N <= 64, "lookup output must fit in a u64");
        bits.into_iter()
            .fold(0, |value, bit| (value << 1) | u128::from(bit))
    }

    #[expect(
        clippy::expect_used,
        reason = "lookup materializers are required to produce one XLEN-bit word"
    )]
    fn output(&mut self, value: Self::Nat) -> Self::Output {
        u64::try_from(value).expect("lookup output must fit in a u64")
    }
}
