//! Fixed-output digest bridge adapted from spongefish 0.8.0 (MIT OR Apache-2.0).
//!
//! The masks and state transitions follow spongefish's `instantiations::Hash`.
//! Squeeze block indices and committed byte lengths use checked `u64` arithmetic.

use core::sync::atomic::{compiler_fence, Ordering};

use blake2::digest::{
    block_api::{Block, BlockSizeUser},
    typenum::Unsigned,
    Digest, FixedOutputReset, Output,
};
use spongefish::{DuplexSpongeInit, DuplexSpongeInterface};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// A duplex sponge over a fixed-output digest.
///
/// Squeeze block indices and run lengths are encoded as big-endian `u64`.
/// The bridge panics if either count exceeds `u64::MAX`.
#[derive(Clone)]
pub struct HashBridge<D: BlockSizeUser + Digest + Clone + FixedOutputReset> {
    hasher: D,
    cv: Output<D>,
    mode: Mode,
    leftover: Output<D>,
    unread: u64,
}

#[derive(Clone, PartialEq, Eq)]
enum Mode {
    Start,
    Absorb,
    Squeeze(u64),
}

impl<D: BlockSizeUser + Digest + Clone + FixedOutputReset> Default for HashBridge<D> {
    fn default() -> Self {
        Self {
            hasher: D::new(),
            cv: Output::<D>::default(),
            mode: Mode::Start,
            leftover: Output::<D>::default(),
            unread: 0,
        }
    }
}

impl<D: BlockSizeUser + Digest + Clone + FixedOutputReset> DuplexSpongeInit for HashBridge<D> {
    fn init(session_id: &[u8; 32]) -> Self {
        let mut sponge = Self::default();
        let _ = sponge.absorb(session_id);
        sponge
    }
}

impl<D: BlockSizeUser + Digest + Clone + FixedOutputReset> DuplexSpongeInterface for HashBridge<D> {
    type U = u8;

    fn absorb(&mut self, input: &[u8]) -> &mut Self {
        self.squeeze_end();
        if self.mode == Mode::Start {
            self.mode = Mode::Absorb;
            Digest::update(&mut self.hasher, Self::mask(0x00));
            Digest::update(&mut self.hasher, &self.cv);
        }
        Digest::update(&mut self.hasher, input);
        self
    }

    #[expect(
        clippy::expect_used,
        reason = "the squeeze block index must fit in u64"
    )]
    fn squeeze(&mut self, mut output: &mut [u8]) -> &mut Self {
        if self.mode == Mode::Absorb {
            let _ = self.ratchet();
        }
        if self.mode == Mode::Start {
            self.mode = Mode::Squeeze(0);
            Digest::update(&mut self.hasher, Self::mask(0x01));
            Digest::update(&mut self.hasher, &self.cv);
        }
        let Mode::Squeeze(blocks) = &mut self.mode else {
            unreachable!("squeeze mode is installed above");
        };

        if self.unread != 0 {
            let unread = self.unread as usize;
            let (_, buffered) = self.leftover.split_at(D::OutputSize::USIZE - unread);
            let take = output.len().min(unread);
            let (head, tail) = output.split_at_mut(take);
            head.copy_from_slice(buffered.split_at(take).0);
            self.unread -= take as u64;
            output = tail;
        }

        while !output.is_empty() {
            let next_block = blocks
                .checked_add(1)
                .expect("squeeze block count exceeds u64");
            let mut hasher = self.hasher.clone();
            Digest::update(&mut hasher, blocks.to_be_bytes());
            let digest = hasher.finalize();

            let take = output.len().min(digest.len());
            let (head, tail) = output.split_at_mut(take);
            let (prefix, remainder) = digest.split_at(take);
            head.copy_from_slice(prefix);
            if !remainder.is_empty() {
                self.unread = remainder.len() as u64;
                self.leftover = digest;
            }
            *blocks = next_block;
            output = tail;
        }
        self
    }
}

impl<D: BlockSizeUser + Digest + Clone + FixedOutputReset> HashBridge<D> {
    /// Commits the current absorb or squeeze run and returns to the initial mode.
    pub fn ratchet(&mut self) -> &mut Self {
        self.squeeze_end();
        if self.mode == Mode::Absorb {
            self.cv = D::digest(self.hasher.finalize_reset());
        }
        self.leftover.as_mut_slice().zeroize();
        self.mode = Mode::Start;
        self
    }

    #[expect(
        clippy::expect_used,
        reason = "both Blake2b backends have a nonempty block"
    )]
    fn mask(tag: u8) -> Block<D> {
        let mut mask = Block::<D>::default();
        *mask.last_mut().expect("digest block cannot be empty") = tag;
        mask
    }

    #[expect(
        clippy::expect_used,
        reason = "the committed squeeze length must fit in u64"
    )]
    fn squeeze_end(&mut self) {
        if let Mode::Squeeze(blocks) = self.mode {
            let byte_count = match blocks.checked_sub(1) {
                None => 0,
                // The last block may contain unread bytes even at u64::MAX.
                Some(full_blocks) => full_blocks
                    .checked_mul(D::OutputSize::U64)
                    .and_then(|bytes| bytes.checked_add(D::OutputSize::U64 - self.unread))
                    .expect("squeeze byte count exceeds u64"),
            };

            Digest::reset(&mut self.hasher);
            let mut hasher = D::new();
            Digest::update(&mut hasher, Self::mask(0x02));
            Digest::update(&mut hasher, &self.cv);
            Digest::update(&mut hasher, byte_count.to_be_bytes());
            self.cv = hasher.finalize();
            self.mode = Mode::Start;
            self.unread = 0;
        }
    }

    #[cfg(test)]
    fn with_squeeze_count(cv: Output<D>, blocks: u64, unread: u64) -> Self {
        let mut sponge = Self::default();
        sponge.cv = cv;
        sponge.mode = Mode::Squeeze(blocks);
        sponge.unread = unread;
        sponge
    }
}

impl<D: BlockSizeUser + Digest + Clone + FixedOutputReset> Zeroize for HashBridge<D> {
    fn zeroize(&mut self) {
        self.cv.as_mut_slice().zeroize();
        self.leftover.as_mut_slice().zeroize();
        self.unread = 0;
        compiler_fence(Ordering::SeqCst);
        Digest::reset(&mut self.hasher);
    }
}

impl<D: BlockSizeUser + Digest + Clone + FixedOutputReset> Drop for HashBridge<D> {
    fn drop(&mut self) {
        self.zeroize();
    }
}

/// The bridge wipes its chaining value and buffered output on drop. The
/// `D: ZeroizeOnDrop` bound ensures the digest also wipes its own state.
impl<D: BlockSizeUser + Digest + Clone + FixedOutputReset + ZeroizeOnDrop> ZeroizeOnDrop
    for HashBridge<D>
{
}

#[cfg(test)]
mod tests {
    use spongefish::instantiations::Hash;
    use zeroize::Zeroize;

    use super::{Digest, DuplexSpongeInit, DuplexSpongeInterface};
    use crate::{Blake2b512, Blake2b512Digest};

    #[expect(
        clippy::indexing_slicing,
        reason = "absorb lengths in the schedule are bounded by the fixture"
    )]
    #[test]
    fn bridge_matches_spongefish() {
        let session = [0x37; 32];
        let message = [0x59; 129];
        let mut actual = Blake2b512::init(&session);
        let mut reference = Hash::<Blake2b512Digest>::init(&session);
        for (absorb_len, squeeze_lengths) in [
            (0, [0, 0, 0]),
            (1, [1, 0, 3]),
            (127, [63, 1, 0]),
            (128, [64, 65, 7]),
            (129, [65, 63, 193]),
        ] {
            let _ = actual.absorb(&message[..absorb_len]);
            let _ = reference.absorb(&message[..absorb_len]);
            for len in squeeze_lengths {
                assert_eq!(actual.squeeze_boxed(len), reference.squeeze_boxed(len));
            }
        }

        let mut actual_clone = actual.clone();
        let mut reference_clone = reference.clone();
        let _ = actual_clone.ratchet().ratchet();
        let _ = reference_clone.ratchet().ratchet();
        assert_eq!(
            actual_clone.squeeze_array::<65>(),
            reference_clone.squeeze_array::<65>()
        );
        assert_eq!(
            actual.squeeze_array::<65>(),
            reference.squeeze_array::<65>()
        );
    }

    #[test]
    fn squeeze_end_commits_big_endian_u64_length() {
        let cv = Blake2b512Digest::digest(b"chaining value");
        for (blocks, unread, length) in [
            (1 << 26, 0, [0, 0, 0, 1, 0, 0, 0, 0]),
            (1 << 58, 1, [0xff; 8]),
        ] {
            let mut sponge = Blake2b512::with_squeeze_count(cv, blocks, unread);
            let _ = sponge.ratchet();

            let mut preimage = [0; 128];
            preimage[127] = 0x02;
            let mut expected = Blake2b512Digest::new();
            expected.update(preimage);
            expected.update(cv);
            expected.update(length);
            assert_eq!(sponge.cv, expected.finalize());
        }
    }

    #[test]
    fn zeroize_wipes_partial_squeeze_and_preserves_squeeze_position() {
        let session = [0x37; 32];
        let mut sponge = Blake2b512::init(&session);
        let _ = sponge.squeeze_array::<1>();
        let zero = &[0; 64];
        assert_ne!(sponge.cv.as_slice(), zero);
        assert_ne!(sponge.leftover.as_slice(), zero);

        let mut ratcheted = sponge.clone();
        let _ = ratcheted.ratchet();
        assert_eq!(ratcheted.leftover.as_slice(), zero);

        sponge.zeroize();
        assert_eq!(sponge.cv.as_slice(), zero);
        assert_eq!(sponge.leftover.as_slice(), zero);
        let expected = Blake2b512Digest::digest(1u64.to_be_bytes());
        assert_eq!(sponge.squeeze_array::<64>().as_slice(), expected.as_slice());

        ratcheted.zeroize();
        let _ = ratcheted.absorb(&session);
        assert_eq!(
            ratcheted.squeeze_array::<64>(),
            Blake2b512::init(&session).squeeze_array::<64>()
        );
    }
}
