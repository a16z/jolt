//! Shared lookup materialization semantics.
//!
//! A table runs the same semantic function with a concrete backend during
//! preprocessing and with a symbolic backend during extraction.

/// Operations available to a lookup materializer.
///
/// The interface starts with the operations required by the AND table. New
/// operations should be added only when another table needs them.
pub trait MaterializerBackend {
    type Bit;
    type Output;

    fn input_bit(&mut self, index: usize) -> Self::Bit;
    fn and(&mut self, left: Self::Bit, right: Self::Bit) -> Self::Bit;
    fn bits_be<const N: usize>(&mut self, bits: [Self::Bit; N]) -> Self::Output;
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
    type Output = u64;

    fn input_bit(&mut self, index: usize) -> Self::Bit {
        assert!(index < 2 * XLEN, "lookup input bit is out of range");
        let shift = 2 * XLEN - 1 - index;
        (self.index >> shift) & 1 == 1
    }

    fn and(&mut self, left: Self::Bit, right: Self::Bit) -> Self::Bit {
        left && right
    }

    fn bits_be<const N: usize>(&mut self, bits: [Self::Bit; N]) -> Self::Output {
        assert!(N <= 64, "lookup output must fit in a u64");
        bits.into_iter()
            .fold(0, |value, bit| (value << 1) | u64::from(bit))
    }
}
