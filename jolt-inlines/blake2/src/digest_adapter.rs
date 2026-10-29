//! [`digest`] trait adapter over the inline Blake2b.
//!
//! `Digest`-generic consumers (Fiat-Shamir transcripts) get the same bytes as
//! the `blake2` crate's `Blake2b<OutSize>`, computed with one inline compress
//! per 128-byte block on a guest and in software on a host.

use core::marker::PhantomData;

use digest::array::ArraySize;
use digest::common::BlockSizeUser;
use digest::{FixedOutput, FixedOutputReset, HashMarker, Output, OutputSizeUser, Reset, Update};

use digest::consts::U128;
pub use digest::consts::{U32, U64};

use crate::sdk::Blake2b as Inline;

/// Blake2b with an `OutSize`-byte digest over the inline compress.
#[derive(Clone)]
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

#[cfg(all(test, feature = "host"))]
mod tests {
    use super::Blake2b;
    use blake2::{Blake2b as ReferenceBlake2b, Blake2b512};
    use digest::consts::{U32, U64};
    use digest::Digest;

    fn input(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 131 + 7) as u8).collect()
    }

    /// The adapter matches the `blake2` crate for both digest widths, across
    /// block boundaries and split updates.
    #[test]
    fn adapter_matches_blake2_crate() {
        for len in [0, 1, 63, 64, 127, 128, 129, 255, 256, 257, 300] {
            let data = input(len);
            for split in [0, len / 3, len] {
                let (head, tail) = data.split_at(split);
                let mut ours = Blake2b::<U32>::new();
                ours.update(head);
                ours.update(tail);
                assert_eq!(
                    ours.finalize().as_slice(),
                    ReferenceBlake2b::<U32>::digest(&data).as_slice(),
                    "U32 len {len} split {split}"
                );
                let mut ours = Blake2b::<U64>::new();
                ours.update(head);
                ours.update(tail);
                assert_eq!(
                    ours.finalize().as_slice(),
                    Blake2b512::digest(&data).as_slice(),
                    "U64 len {len} split {split}"
                );
            }
        }
    }
}
