//! A space-efficient implementation of Pippenger's algorithm.
use ark_ff::{PrimeField, Zero};

use ark_std::{borrow::Borrow, vec::Vec};
use hashbrown::HashMap;

use super::VariableBaseMSM;

/// Struct for the chunked Pippenger algorithm.
pub struct ChunkedPippenger<G: VariableBaseMSM> {
    scalars_buffer: Vec<<G::ScalarField as PrimeField>::BigInt>,
    bases_buffer: Vec<G::MulBase>,
    result: G,
    buf_size: usize,
}

impl<G: VariableBaseMSM> ChunkedPippenger<G> {
    /// Initialize a chunked Pippenger instance with default parameters.
    pub fn new(max_msm_buffer: usize) -> Self {
        Self {
            scalars_buffer: Vec::with_capacity(max_msm_buffer),
            bases_buffer: Vec::with_capacity(max_msm_buffer),
            result: G::zero(),
            buf_size: max_msm_buffer,
        }
    }

    /// Initialize a chunked Pippenger instance with the given buffer size.
    pub fn with_size(buf_size: usize) -> Self {
        Self {
            scalars_buffer: Vec::with_capacity(buf_size),
            bases_buffer: Vec::with_capacity(buf_size),
            result: G::zero(),
            buf_size,
        }
    }

    /// Add a new (base, scalar) pair into the instance.
    #[inline(always)]
    pub fn add<B, S>(&mut self, base: B, scalar: S)
    where
        B: Borrow<G::MulBase>,
        S: Borrow<<G::ScalarField as PrimeField>::BigInt>,
    {
        self.scalars_buffer.push(*scalar.borrow());
        self.bases_buffer.push(*base.borrow());
        if self.scalars_buffer.len() == self.buf_size {
            self.result.add_assign(G::msm_bigint(
                self.bases_buffer.as_slice(),
                self.scalars_buffer.as_slice(),
            ));
            self.scalars_buffer.clear();
            self.bases_buffer.clear();
        }
    }

    /// Output the final Pippenger algorithm result.
    #[inline(always)]
    pub fn finalize(mut self) -> G {
        if !self.scalars_buffer.is_empty() {
            self.result +=
                G::msm_bigint(self.bases_buffer.as_slice(), self.scalars_buffer.as_slice());
        }
        self.result
    }
}

/// Hash map struct for Pippenger algorithm.
pub struct HashMapPippenger<G: VariableBaseMSM> {
    buffer: HashMap<G::MulBase, G::ScalarField>,
    result: G,
    buf_size: usize,
}

impl<G: VariableBaseMSM> HashMapPippenger<G> {
    /// Produce a new hash map with the maximum msm buffer size.
    pub fn new(max_msm_buffer: usize) -> Self {
        Self {
            buffer: HashMap::with_capacity(max_msm_buffer),
            result: G::zero(),
            buf_size: max_msm_buffer,
        }
    }

    /// Add a new (base, scalar) pair into the hash map.
    #[inline(always)]
    pub fn add<B, S>(&mut self, base: B, scalar: S)
    where
        B: Borrow<G::MulBase>,
        S: Borrow<G::ScalarField>,
    {
        // update the entry, guarding the possibility that it has been already set.
        let entry = self
            .buffer
            .entry(*base.borrow())
            .or_insert(G::ScalarField::zero());
        *entry += *scalar.borrow();
        if self.buffer.len() == self.buf_size {
            let bases = self.buffer.keys().cloned().collect::<Vec<_>>();
            let scalars = self
                .buffer
                .values()
                .map(|s| s.into_bigint())
                .collect::<Vec<_>>();
            self.result += G::msm_bigint(&bases, &scalars);
            self.buffer.clear();
        }
    }

    /// Update the final result with (base, scalar) pairs in the hash map.
    #[inline(always)]
    pub fn finalize(mut self) -> G {
        if !self.buffer.is_empty() {
            let bases = self.buffer.keys().cloned().collect::<Vec<_>>();
            let scalars = self
                .buffer
                .values()
                .map(|s| s.into_bigint())
                .collect::<Vec<_>>();

            self.result += G::msm_bigint(&bases, &scalars);
        }
        self.result
    }
}
