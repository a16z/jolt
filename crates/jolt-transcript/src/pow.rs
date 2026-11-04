//! Proof of work on the transcript sponge, as a standalone step.
//!
//! The prover sends a nonce below `2^(bits + 7)`, a `u64` for every supported
//! difficulty. The NARG string carries it as canonical LEB128; the sponge
//! absorbs it as one `u64` little-endian word, in one prover message. Both
//! parties then squeeze eight bytes and accept iff the low `bits` bits of
//! their little-endian `u64` are zero. The absorb and the check are each one
//! word, so the step keeps the transcript word aligned. The step protects the
//! verifier messages that follow it, up to the next prover message.
//!
//! The prover scans candidate nonces in increasing order, on the calling
//! thread, against an owned copy of the sponge state: the oracle of
//! [`ProverTranscript::verifier_message_oracle`], shaped like spongefish's
//! (arkworks-rs/spongefish#306). Spongefish 0.8.0 exposes that state only
//! under its `yolocrypto` feature, so the search is behind this crate's
//! `prover-grinding` feature. Cargo unifies features across a build, so a
//! binary that links a grinding prover compiles its verifier with
//! `yolocrypto` too; in 0.8.0 the feature changes visibility only, not bytes.
//! The layout of the step follows arkworks-rs/spongefish#301 and its
//! difficulty rule #307.
//!
//! A proof that carries its nonces outside the NARG string runs the same step
//! with the nonce absorbed as a public message
//! ([`Transcript::check_public_grind`] and [`ProverTranscript::public_grind`]).
//!
//! [`Transcript::check_public_grind`]: crate::Transcript::check_public_grind
//! [`ProverTranscript::public_grind`]: crate::ProverTranscript::public_grind
//! [`ProverTranscript::verifier_message_oracle`]: crate::ProverTranscript::verifier_message_oracle

use crate::MAX_POW_BITS;

/// Bytes squeezed for a proof-of-work check: one word.
pub(crate) const CHECK_BYTES: usize = 8;

/// The absorbed form of a proof-of-work nonce: one `u64` little-endian word.
pub(crate) fn nonce_word(nonce: u64) -> [u8; 8] {
    nonce.to_le_bytes()
}

/// The layout of one proof-of-work step at a difficulty of `1..=MAX_POW_BITS`.
#[derive(Clone, Copy)]
pub(crate) struct SpongePow {
    bits: u32,
}

impl SpongePow {
    /// `None` unless `1 <= bits <= MAX_POW_BITS`.
    pub(crate) const fn new(bits: u32) -> Option<Self> {
        if bits == 0 || bits > MAX_POW_BITS {
            None
        } else {
            Some(Self { bits })
        }
    }

    /// The largest nonce, `2^(bits + 7) - 1`: `u64::MAX` at `MAX_POW_BITS`.
    pub(crate) const fn last_nonce(self) -> u64 {
        u64::MAX >> (MAX_POW_BITS - self.bits)
    }

    /// `ceil((bits + 7) / 7)` bytes encode every nonce, at most 10.
    pub(crate) const fn max_nonce_len(self) -> usize {
        (self.bits as usize + 13) / 7
    }

    /// The low `bits` bits of the little-endian `u64` of the
    /// [`CHECK_BYTES`] check bytes are zero. Rejects check bytes of the wrong
    /// length.
    pub(crate) fn accepts(self, check: &[u8]) -> bool {
        let Ok(word) = <[u8; CHECK_BYTES]>::try_from(check) else {
            return false;
        };
        // `bits <= MAX_POW_BITS < 64`, so the shift cannot overflow.
        u64::from_le_bytes(word) & ((1u64 << self.bits) - 1) == 0
    }

    /// Reads a nonce byte by byte through `next`. Returns `None` unless its
    /// encoding is canonical LEB128 and it is at most `last_nonce`; read
    /// errors pass through.
    pub(crate) fn read_nonce<E>(
        self,
        mut next: impl FnMut() -> Result<u8, E>,
    ) -> Result<Option<u64>, E> {
        // At most 10 bytes of 7 bits: below `2^70`, so no bit is lost.
        let mut value = 0u128;
        for i in 0..self.max_nonce_len() {
            let byte = next()?;
            value |= u128::from(byte & 0x7f) << (7 * i);
            if byte & 0x80 == 0 {
                // A canonical encoding has no trailing zero byte.
                let canonical = i == 0 || byte != 0;
                return Ok(u64::try_from(value)
                    .ok()
                    .filter(|&nonce| canonical && nonce <= self.last_nonce()));
            }
        }
        Ok(None)
    }

    /// The least nonce whose absorbed word makes `oracle`'s next
    /// [`CHECK_BYTES`] bytes pass the check, scanning in increasing order on
    /// the calling thread. `None` if no nonce up to `last_nonce` passes.
    #[cfg(feature = "prover-grinding")]
    pub(crate) fn search(self, oracle: impl Fn(&[u8], &mut [u8])) -> Option<u64> {
        let mut check = [0u8; CHECK_BYTES];
        (0..=self.last_nonce()).find(|&nonce| {
            oracle(&nonce_word(nonce), &mut check);
            self.accepts(&check)
        })
    }
}

/// The canonical LEB128 encoding of a `u64`: seven bits per byte, least
/// significant first, the high bit set on every byte but the last. This is
/// the nonce's NARG form; [`SpongePow::read_nonce`] parses it.
#[cfg(feature = "prover-grinding")]
#[derive(Clone, Copy)]
pub(crate) struct Leb128 {
    bytes: [u8; 10],
    len: usize,
}

#[cfg(feature = "prover-grinding")]
impl Leb128 {
    pub(crate) fn new(mut value: u64) -> Self {
        let mut bytes = [0u8; 10];
        let mut len = 0;
        for byte in &mut bytes {
            len += 1;
            if value < 0x80 {
                *byte = value as u8;
                break;
            }
            *byte = (value as u8) | 0x80;
            value >>= 7;
        }
        Self { bytes, len }
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        self.bytes.get(..self.len).unwrap_or(&self.bytes)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::{SpongePow, MAX_POW_BITS};

    #[test]
    fn difficulty_bounds() {
        assert!(SpongePow::new(0).is_none());
        assert!(SpongePow::new(MAX_POW_BITS).is_some());
        assert!(SpongePow::new(MAX_POW_BITS + 1).is_none());
    }

    #[test]
    fn layout() {
        let layout = |bits| {
            let pow = SpongePow::new(bits).unwrap();
            (pow.last_nonce(), pow.max_nonce_len())
        };
        assert_eq!(layout(1), (255, 2));
        assert_eq!(layout(8), ((1 << 15) - 1, 3));
        assert_eq!(layout(10), ((1 << 17) - 1, 3));
        // The cap was 25 bits with a `u32` nonce; the layout up to it is
        // unchanged.
        assert_eq!(layout(25), (u64::from(u32::MAX), 5));
        assert_eq!(layout(26), ((1 << 33) - 1, 5));
        assert_eq!(layout(28), ((1 << 35) - 1, 5));
        assert_eq!(layout(29), ((1 << 36) - 1, 6));
        assert_eq!(layout(56), ((1 << 63) - 1, 9));
        assert_eq!(layout(MAX_POW_BITS), (u64::MAX, 10));
        assert_eq!(MAX_POW_BITS, 57);
    }

    #[test]
    fn check_reads_the_low_bits_of_one_le_word() {
        let pow = SpongePow::new(9).unwrap();
        // Bits 0..9 zero, every higher bit set: accepted.
        let pass = (u64::MAX << 9).to_le_bytes();
        assert!(pow.accepts(&pass));
        // Bit 8, in the second byte, set: rejected.
        let fail = (1u64 << 8).to_le_bytes();
        assert!(!pow.accepts(&fail));
        // Seven or nine bytes: rejected whatever their value.
        assert!(!pow.accepts(&[0u8; 7]));
        assert!(!pow.accepts(&[0u8; 9]));
        let full = SpongePow::new(MAX_POW_BITS).unwrap();
        assert!(full.accepts(&(u64::MAX << MAX_POW_BITS).to_le_bytes()));
        assert!(!full.accepts(&(1u64 << (MAX_POW_BITS - 1)).to_le_bytes()));
        assert_eq!(super::nonce_word(0x0102_0304), [4, 3, 2, 1, 0, 0, 0, 0]);
    }

    #[test]
    fn read_nonce_accepts_only_canonical_nonces_in_range() {
        let read = |bits, bytes: &[u8]| {
            let mut bytes = bytes.iter().copied();
            SpongePow::new(bits)
                .unwrap()
                .read_nonce(|| bytes.next().ok_or("short"))
        };
        assert_eq!(read(1, &[0x00]), Ok(Some(0)));
        assert_eq!(read(1, &[0x7f]), Ok(Some(127)));
        assert_eq!(read(1, &[0x80, 0x01]), Ok(Some(128)));
        assert_eq!(read(1, &[0xff, 0x01]), Ok(Some(255)));
        assert_eq!(
            read(25, &[0xff, 0xff, 0xff, 0xff, 0x0f]),
            Ok(Some(u32::MAX.into()))
        );
        assert_eq!(read(26, &[0x80, 0x80, 0x80, 0x80, 0x10]), Ok(Some(1 << 32)));
        let max = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
        assert_eq!(read(57, &max), Ok(Some(u64::MAX)));
        assert_eq!(read(57, &[0x00]), Ok(Some(0)));
        // A trailing zero byte, a nonce of 2^8 at 1 bit, a continuation past
        // the longest encoding, a nonce of 2^32 at 25 bits, and a short read.
        assert_eq!(read(1, &[0x80, 0x00]), Ok(None));
        assert_eq!(read(8, &[0x81, 0x80, 0x00]), Ok(None));
        assert_eq!(read(1, &[0x80, 0x02]), Ok(None));
        assert_eq!(read(1, &[0x80, 0x80, 0x01]), Ok(None));
        assert_eq!(read(25, &[0x80, 0x80, 0x80, 0x80, 0x10]), Ok(None));
        assert_eq!(read(8, &[0x80]), Err("short"));
        // At 57 bits: a tenth byte carrying a bit past `2^64`, a trailing
        // zero tenth byte, a continuation past ten bytes, a nonce of `2^63`
        // at 56 bits, and a short read.
        let high = |last: u8| {
            let mut bytes = [0xff; 10];
            bytes[9] = last;
            bytes
        };
        assert_eq!(read(57, &high(0x02)), Ok(None));
        assert_eq!(read(57, &high(0x7f)), Ok(None));
        assert_eq!(read(57, &high(0x00)), Ok(None));
        assert_eq!(read(57, &high(0x81)), Ok(None));
        let two_63 = [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01];
        assert_eq!(read(57, &two_63), Ok(Some(1 << 63)));
        assert_eq!(read(56, &two_63), Ok(None));
        assert_eq!(read(57, &[0xff; 9]), Err("short"));
    }

    #[cfg(feature = "prover-grinding")]
    mod search {
        use super::super::{nonce_word, Leb128, SpongePow, MAX_POW_BITS};

        #[test]
        fn leb128_is_canonical() {
            let encode = |value| Leb128::new(value).as_bytes().to_vec();
            assert_eq!(encode(0), [0]);
            assert_eq!(encode(127), [0x7f]);
            assert_eq!(encode(128), [0x80, 0x01]);
            assert_eq!(encode(300), [0xac, 0x02]);
            assert_eq!(encode(u32::MAX.into()), [0xff, 0xff, 0xff, 0xff, 0x0f]);
            assert_eq!(encode(1 << 32), [0x80, 0x80, 0x80, 0x80, 0x10]);
            assert_eq!(
                encode(u64::MAX),
                [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]
            );
            for bits in 1..=MAX_POW_BITS {
                let pow = SpongePow::new(bits).unwrap();
                let longest = Leb128::new(pow.last_nonce());
                assert_eq!(longest.as_bytes().len(), pow.max_nonce_len());
                let mut bytes = longest.as_bytes().iter().copied();
                assert_eq!(
                    pow.read_nonce(|| bytes.next().ok_or(())),
                    Ok(Some(pow.last_nonce()))
                );
            }
        }

        /// An oracle that accepts exactly the nonces in `targets`.
        fn accept(targets: Vec<u64>) -> impl Fn(&[u8], &mut [u8]) {
            move |candidate, out| {
                let hit = targets
                    .iter()
                    .any(|&target| candidate == nonce_word(target));
                out.fill(if hit { 0 } else { 0xff });
            }
        }

        #[test]
        fn search_finds_a_lone_nonce_anywhere() {
            for bits in [1, 9] {
                let pow = SpongePow::new(bits).unwrap();
                let last = pow.last_nonce();
                assert_eq!(pow.search(accept(Vec::new())), None);
                for target in [0, 1, last / 2, last] {
                    assert_eq!(pow.search(accept(vec![target])), Some(target));
                }
            }
        }

        /// At `MAX_POW_BITS` the nonce space is all of `u64`; a full search
        /// is out of reach, so this checks searches whose answer comes early.
        #[test]
        fn search_runs_at_the_cap() {
            let pow = SpongePow::new(MAX_POW_BITS).unwrap();
            assert_eq!(pow.last_nonce(), u64::MAX);
            for target in [0, 1, 1500] {
                assert_eq!(pow.search(accept(vec![target])), Some(target));
            }
        }

        #[test]
        fn search_returns_the_least_of_several_accepted_nonces() {
            for bits in [1, 9] {
                let pow = SpongePow::new(bits).unwrap();
                let last = pow.last_nonce();
                for targets in [
                    vec![last, last / 2],
                    vec![last / 2 + 1, last / 3, last],
                    vec![last - 1, 1, last],
                    vec![last - 1, last],
                ] {
                    let least = targets.iter().min().copied();
                    assert_eq!(pow.search(accept(targets)), least);
                }
            }
        }
    }
}
