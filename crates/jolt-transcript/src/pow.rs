//! Proof of work on the transcript sponge, as a standalone step.
//!
//! The prover
//! sends a nonce below `2^(bits + 7)` as canonical LEB128, one byte per prover
//! message, and both parties squeeze a check and accept iff its low `bits`
//! bits are zero. On a byte sponge the check is `ceil(bits / 8)`
//! little-endian bytes; on `PoseidonSponge` it is one element, read as its
//! canonical integer ([`SpongeUnit`]). The step protects the verifier messages
//! that follow it, up to the next prover message. The check is squeezed in one
//! call; on every sponge here that gives the same units as squeezing it one
//! unit at a time, because each buffers its squeeze output.
//!
//! [`SpongeUnit`]: crate::SpongeUnit
//!
//! The same layout, search primitive and probe are proposed for spongefish in
//! arkworks-rs/spongefish#301 (`PowTranscriptExt::sponge_pow`,
//! `MessageSearch`, `Probe`); none of them is in the published 0.8.0.
//!
//! The prover searches by evaluating candidate nonces on a copy of the sponge
//! state ([`MessageSearch`]). Spongefish 0.8.0 exposes that state only under
//! its `yolocrypto` feature, so the search is behind this crate's
//! `prover-grinding` feature. Cargo unifies features across a build, so a
//! binary that links a grinding prover compiles its verifier with
//! `yolocrypto` too; in 0.8.0 the feature changes visibility only, not bytes.
//!
//! A proof that carries its nonces outside the NARG string runs the same step
//! with the nonce absorbed as a public message
//! ([`Transcript::check_public_grind`] and [`ProverTranscript::public_grind`]).
//!
//! [`Transcript::check_public_grind`]: crate::Transcript::check_public_grind
//! [`ProverTranscript::public_grind`]: crate::ProverTranscript::public_grind

#[cfg(feature = "prover-grinding")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "prover-grinding")]
use spongefish::DuplexSpongeInterface;

use crate::{SpongeUnit, MAX_POW_BITS};

#[cfg(feature = "prover-grinding")]
/// Evaluates the transcript on a candidate prover message without sending it.
///
/// A probe is [`Sync`], so a search may run on
/// several threads at once.
pub trait Probe<U = u8>: Sync {
    /// Writes into `out` the units the transcript would squeeze right after
    /// absorbing `candidate`. The transcript itself is unchanged.
    fn probe(&self, candidate: &[u8], out: &mut [U]);
}

/// A copy of the prover's public sponge, for searching over a prover message:
/// a proof-of-work nonce, or a retry counter whose following challenges must
/// satisfy a prover-side condition.
///
/// A probe absorbs the candidate one byte at a time, each byte as a one-byte
/// message, exactly as sending it as single-byte prover messages does. For the
/// Keccak and Blake2b sponges this equals one absorb of the whole candidate;
/// [`PoseidonSponge`] packs a length with every message, so for it only the
/// byte-by-byte form matches.
///
/// It holds no private randomness, writes no proof bytes and cannot restore
/// or advance the live prover state.
///
/// [`PoseidonSponge`]: crate::PoseidonSponge
#[cfg(feature = "prover-grinding")]
#[derive(Clone)]
pub struct MessageSearch<H> {
    sponge: H,
}

#[cfg(feature = "prover-grinding")]
impl<H> MessageSearch<H> {
    pub(crate) const fn new(sponge: H) -> Self {
        Self { sponge }
    }
}

#[cfg(feature = "prover-grinding")]
impl<H> Probe<H::U> for MessageSearch<H>
where
    H: DuplexSpongeInterface<U: SpongeUnit> + Sync,
{
    fn probe(&self, candidate: &[u8], out: &mut [H::U]) {
        let mut sponge = self.sponge.clone();
        for byte in candidate {
            let _ = sponge.absorb(H::U::encode_message(byte).as_ref());
        }
        let _ = sponge.squeeze(out);
    }
}

/// Nonces the calling thread tries before the search wakes the thread pool.
/// Waking the pool's workers costs about as much as this many probes
/// (200-450 us against about 0.3 us per probe, measured on 8 threads), so
/// scanning this far first keeps a search within about twice the cost of the
/// better of the two strategies. Grinds of up to 8 bits almost always end
/// here.
#[cfg(feature = "parallel")]
const CALLER_NONCES: u32 = 1024;

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

    /// The largest nonce, `2^(bits + 7) - 1`.
    pub(crate) const fn last_nonce(self) -> u32 {
        u32::MAX >> (MAX_POW_BITS - self.bits)
    }

    /// `ceil((bits + 7) / 7)` bytes encode every nonce, at most 5.
    pub(crate) const fn max_nonce_len(self) -> usize {
        (self.bits as usize + 13) / 7
    }

    /// The low `bits` bits of the check are zero ([`SpongeUnit::low_word`]).
    /// Rejects a check of the wrong length.
    pub(crate) fn accepts<U: SpongeUnit>(self, check: &[U]) -> bool {
        check.len() == U::check_len(self.bits)
            && U::low_word(check).is_some_and(|word| word & ((1u32 << self.bits) - 1) == 0)
    }

    /// Reads a nonce byte by byte through `next`. Returns `None` unless its
    /// encoding is canonical LEB128 and it is at most `last_nonce`; read
    /// errors pass through.
    pub(crate) fn read_nonce<E>(
        self,
        mut next: impl FnMut() -> Result<u8, E>,
    ) -> Result<Option<u32>, E> {
        let mut value = 0u64;
        for i in 0..self.max_nonce_len() {
            let byte = next()?;
            value |= u64::from(byte & 0x7f) << (7 * i);
            if byte & 0x80 == 0 {
                // A canonical encoding has no trailing zero byte.
                let canonical = i == 0 || byte != 0;
                return Ok(u32::try_from(value)
                    .ok()
                    .filter(|&nonce| canonical && nonce <= self.last_nonce()));
            }
        }
        Ok(None)
    }

    /// Whether sending `nonce` makes the next squeezed check pass.
    #[cfg(feature = "prover-grinding")]
    fn accepts_nonce<U: SpongeUnit>(self, probe: &impl Probe<U>, nonce: u32) -> bool {
        let mut check = [U::ZERO; 4];
        let Some(out) = check.get_mut(..U::check_len(self.bits)) else {
            return false;
        };
        probe.probe(Leb128::new(nonce).as_bytes(), out);
        self.accepts(out)
    }

    /// The least accepted nonce. The calling thread scans the first
    /// [`CALLER_NONCES`] nonces; if none passes, every thread of the current
    /// pool scans the rest. Builds without `parallel` scan everything on the
    /// calling thread. Either way the result is the least accepted nonce, so
    /// the proof does not depend on the thread count.
    #[cfg(feature = "prover-grinding")]
    pub(crate) fn search<U: SpongeUnit>(self, probe: &impl Probe<U>) -> Option<u32> {
        let last = self.last_nonce();
        // `u64::MAX` is no nonce, so it marks "not found".
        let least = AtomicU64::new(u64::MAX);
        #[cfg(feature = "parallel")]
        {
            let caller_last = last.min(CALLER_NONCES - 1);
            self.scan(probe, 0, 1, caller_last, &least);
            if least.load(Ordering::Relaxed) == u64::MAX && caller_last < last {
                // Each thread scans its own residue class of the rest.
                drop(rayon::broadcast(|ctx| {
                    let first = u64::from(caller_last) + 1 + ctx.index() as u64;
                    self.scan(probe, first, ctx.num_threads(), last, &least);
                }));
            }
        }
        #[cfg(not(feature = "parallel"))]
        self.scan(probe, 0, 1, last, &least);
        u32::try_from(least.into_inner()).ok()
    }

    /// Scans the nonces `first, first + stride, ...` up to `last` in
    /// increasing order and lowers `least` to the first one accepted. Stops
    /// once it cannot beat `least`, which other scans may lower meanwhile.
    #[cfg(feature = "prover-grinding")]
    fn scan<U: SpongeUnit>(
        self,
        probe: &impl Probe<U>,
        first: u64,
        stride: usize,
        last: u32,
        least: &AtomicU64,
    ) {
        let nonces = (first..=u64::from(last))
            .step_by(stride)
            .filter_map(|nonce| u32::try_from(nonce).ok());
        for nonce in nonces {
            if u64::from(nonce) >= least.load(Ordering::Relaxed) {
                break;
            }
            if self.accepts_nonce(probe, nonce) {
                let _ = least.fetch_min(u64::from(nonce), Ordering::SeqCst);
                break;
            }
        }
    }
}

/// The canonical LEB128 encoding of a `u32`: seven bits per byte, least
/// significant first, the high bit set on every byte but the last.
#[derive(Clone, Copy)]
pub(crate) struct Leb128 {
    bytes: [u8; 5],
    len: usize,
}

impl Leb128 {
    pub(crate) fn new(mut value: u32) -> Self {
        let mut bytes = [0u8; 5];
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
    use super::{SpongePow, SpongeUnit, MAX_POW_BITS};

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
            (pow.last_nonce(), pow.max_nonce_len(), u8::check_len(bits))
        };
        assert_eq!(layout(1), (255, 2, 1));
        assert_eq!(layout(8), ((1 << 15) - 1, 3, 1));
        assert_eq!(layout(10), ((1 << 17) - 1, 3, 2));
        assert_eq!(layout(25), (u32::MAX, 5, 4));
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
            Ok(Some(u32::MAX))
        );
        // A trailing zero byte, a nonce of 2^8 at 1 bit, a continuation past
        // the longest encoding, a nonce of 2^32 at 25 bits, and a short read.
        assert_eq!(read(1, &[0x80, 0x00]), Ok(None));
        assert_eq!(read(8, &[0x81, 0x80, 0x00]), Ok(None));
        assert_eq!(read(1, &[0x80, 0x02]), Ok(None));
        assert_eq!(read(1, &[0x80, 0x80, 0x01]), Ok(None));
        assert_eq!(read(25, &[0x80, 0x80, 0x80, 0x80, 0x10]), Ok(None));
        assert_eq!(read(8, &[0x80]), Err("short"));
    }

    #[cfg(feature = "prover-grinding")]
    mod search {
        use super::super::{Leb128, Probe, SpongePow, MAX_POW_BITS};

        #[test]
        fn leb128_is_canonical() {
            let encode = |value| Leb128::new(value).as_bytes().to_vec();
            assert_eq!(encode(0), [0]);
            assert_eq!(encode(127), [0x7f]);
            assert_eq!(encode(128), [0x80, 0x01]);
            assert_eq!(encode(300), [0xac, 0x02]);
            assert_eq!(encode(u32::MAX), [0xff, 0xff, 0xff, 0xff, 0x0f]);
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

        /// The least accepted nonce, by a plain scan.
        fn least_accepted(pow: SpongePow, probe: &impl Probe) -> Option<u32> {
            (0..=pow.last_nonce()).find(|&nonce| pow.accepts_nonce(probe, nonce))
        }

        /// A probe that accepts exactly the nonce `target`, or none.
        struct AcceptOnly(Option<u32>);

        impl Probe for AcceptOnly {
            fn probe(&self, candidate: &[u8], out: &mut [u8]) {
                let hit = self
                    .0
                    .is_some_and(|target| candidate == Leb128::new(target).as_bytes());
                out.fill(if hit { 0 } else { 0xff });
            }
        }

        #[test]
        fn search_finds_a_lone_nonce_anywhere_and_matches_the_sequential_scan() {
            for bits in [1, 9] {
                let pow = SpongePow::new(bits).unwrap();
                let last = pow.last_nonce();
                for target in [None, Some(0), Some(1), Some(last / 2), Some(last)] {
                    let probe = AcceptOnly(target);
                    assert_eq!(least_accepted(pow, &probe), target);
                    assert_eq!(pow.search(&probe), target);
                }
            }
        }

        /// A probe that accepts every nonce in a set.
        struct AcceptAny(Vec<u32>);

        impl Probe for AcceptAny {
            fn probe(&self, candidate: &[u8], out: &mut [u8]) {
                let hit = self
                    .0
                    .iter()
                    .any(|&target| candidate == Leb128::new(target).as_bytes());
                out.fill(if hit { 0 } else { 0xff });
            }
        }

        /// With accepted nonces spread over the space, and so over several
        /// threads' ranges, the search still returns the least one.
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
                    vec![last / 4, last / 4 + 1, last / 2, 3 * (last / 4)],
                ] {
                    let least = targets.iter().min().copied();
                    let probe = AcceptAny(targets);
                    assert_eq!(least_accepted(pow, &probe), least);
                    assert_eq!(pow.search(&probe), least);
                }
            }
        }

        /// Wherever the least accepted nonce falls (inside the calling
        /// thread's stretch, at either side of its edge, or past it) and on
        /// any number of threads, the search returns it, so proofs do not
        /// depend on the thread count.
        #[cfg(feature = "parallel")]
        #[test]
        fn search_is_independent_of_the_thread_count() {
            use super::super::CALLER_NONCES;
            use rayon::ThreadPoolBuilder;

            for bits in [2, 9] {
                let pow = SpongePow::new(bits).unwrap();
                let last = pow.last_nonce();
                let candidates = [
                    0,
                    5,
                    CALLER_NONCES - 1,
                    CALLER_NONCES,
                    CALLER_NONCES + 17,
                    last / 2,
                    last,
                ];
                for least in candidates.into_iter().filter(|&least| least <= last) {
                    let probe = AcceptAny(vec![last, least, (least + 1).min(last)]);
                    for threads in [1, 2, 3, 8] {
                        let pool = ThreadPoolBuilder::new()
                            .num_threads(threads)
                            .build()
                            .unwrap();
                        assert_eq!(pool.install(|| pow.search(&probe)), Some(least));
                    }
                }
            }
        }
    }
}
