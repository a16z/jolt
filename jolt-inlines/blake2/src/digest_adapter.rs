//! [`digest`] trait adapter over the inline Blake2b.
//!
//! `Digest`-generic consumers (Fiat-Shamir transcripts) get the same bytes as
//! the `blake2` crate's `Blake2b<OutSize>`, computed with one inline compress
//! per 128-byte block on a guest and in software on a host.

use core::marker::PhantomData;

use digest::array::ArraySize;
use digest::common::BlockSizeUser;
use digest::{FixedOutput, FixedOutputReset, HashMarker, Output, OutputSizeUser, Reset, Update};

pub use digest::consts::{U128, U32, U64};

use crate::sdk::Blake2b as Inline;

/// Blake2b with an `OutSize`-byte digest over the inline compress.
pub struct Blake2b<OutSize: ArraySize> {
    state: Inline,
    _out: PhantomData<OutSize>,
}

impl<OutSize: ArraySize> Blake2b<OutSize> {
    fn fresh() -> Inline {
        Inline::new_with_output_len(OutSize::USIZE)
    }
}

impl<OutSize: ArraySize> Default for Blake2b<OutSize> {
    fn default() -> Self {
        Self {
            state: Self::fresh(),
            _out: PhantomData,
        }
    }
}

impl<OutSize: ArraySize> Clone for Blake2b<OutSize> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            _out: PhantomData,
        }
    }
}

impl<OutSize: ArraySize> HashMarker for Blake2b<OutSize> {}

impl<OutSize: ArraySize> OutputSizeUser for Blake2b<OutSize> {
    type OutputSize = OutSize;
}

impl<OutSize: ArraySize> BlockSizeUser for Blake2b<OutSize> {
    type BlockSize = U128;
}

impl<OutSize: ArraySize> Update for Blake2b<OutSize> {
    fn update(&mut self, data: &[u8]) {
        self.state.update(data);
    }
}

impl<OutSize: ArraySize> FixedOutput for Blake2b<OutSize> {
    #[inline(always)]
    fn finalize_into(self, out: &mut Output<Self>) {
        self.state.finalize_into(&mut out[..]);
    }
}

impl<OutSize: ArraySize> Reset for Blake2b<OutSize> {
    fn reset(&mut self) {
        self.state = Self::fresh();
    }
}

impl<OutSize: ArraySize> FixedOutputReset for Blake2b<OutSize> {
    fn finalize_into_reset(&mut self, out: &mut Output<Self>) {
        let state = core::mem::replace(&mut self.state, Self::fresh());
        state.finalize_into(&mut out[..]);
    }
}
