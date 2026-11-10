//! Private prover randomness, independent of the public transcript sponge.

#[cfg(feature = "prover-streams")]
use chacha20::rand_core::{Rng, SeedableRng};
#[cfg(feature = "prover-streams")]
use chacha20::ChaCha20Rng;
use rand_core::{CryptoRng, Error, RngCore};
use spongefish::{DuplexSpongeInit, PrivateRng};
#[cfg(feature = "prover-streams")]
use zeroize::Zeroizing;

use crate::Shake128;

trait CoinSource {
    fn fill(&mut self, output: &mut [u8]);
}

impl<R: DuplexSpongeInit<U = u8> + Send> CoinSource for PrivateRng<R> {
    fn fill(&mut self, output: &mut [u8]) {
        self.fill_bytes(output);
    }
}

/// A borrowed private sponge adapted to rand_core 0.6 samplers.
/// Drawing coins never absorbs or squeezes the public transcript.
pub struct ProverCoins<'a>(&'a mut (dyn CoinSource + Send));

impl<'a> ProverCoins<'a> {
    pub(crate) fn new<R: DuplexSpongeInit<U = u8> + Send>(rng: &'a mut PrivateRng<R>) -> Self {
        Self(rng)
    }

    /// Draw a fresh 32-byte key for an owned ChaCha20 stream, independent of
    /// subsequent parent draws. The stream does not borrow the transcript.
    /// Security additionally relies on ChaCha20 as a pseudorandom generator.
    /// ChaCha20's `zeroize` feature wipes its key and buffered output on drop.
    #[cfg(feature = "prover-streams")]
    pub fn child(&mut self) -> ChildCoins {
        let mut seed = Zeroizing::new([0u8; 32]);
        self.fill_bytes(seed.as_mut());
        ChildCoins(ChaCha20Rng::from_seed(*seed))
    }
}

impl RngCore for ProverCoins<'_> {
    fn next_u32(&mut self) -> u32 {
        rand_core::impls::next_u32_via_fill(self)
    }

    fn next_u64(&mut self) -> u64 {
        rand_core::impls::next_u64_via_fill(self)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for ProverCoins<'_> {}

/// Private randomness for preprocessing that has no prover transcript.
/// The SHAKE128 state and reader wipe on drop through `shake/zeroize`.
pub struct OwnedCoins(PrivateRng<Shake128>);

impl OwnedCoins {
    /// Seed a fresh private sponge from the operating system.
    ///
    /// # Panics
    /// Panics if the operating system's entropy source fails.
    #[cfg(feature = "prover-entropy")]
    pub fn from_os_entropy() -> Self {
        Self(PrivateRng::from_os_entropy())
    }

    /// Borrow the private sponge for a sampler.
    pub fn coins(&mut self) -> ProverCoins<'_> {
        ProverCoins::new(&mut self.0)
    }

    /// Mix a 32-byte hedge into the private sponge before drawing coins.
    pub fn mix_entropy(&mut self, data: &[u8; 32]) {
        self.0.mix_entropy(data);
    }
}

/// An owned ChaCha20 stream keyed by a fresh draw from private prover coins.
/// Available only with `prover-streams`; wipes its state on drop.
#[cfg(feature = "prover-streams")]
pub struct ChildCoins(ChaCha20Rng);

#[cfg(feature = "prover-streams")]
impl RngCore for ChildCoins {
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

#[cfg(feature = "prover-streams")]
impl CryptoRng for ChildCoins {}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;
    use crate::transcript::sealed::Sealed;
    use crate::{
        protocol_tag, verifier_transcript, BytesMsg, ProverInit, ProverState, ProverTranscript,
        VerifierTranscript,
    };

    const SESSION: [u8; 32] = protocol_tag(b"prover-coins-test");
    const SEED: [u8; 32] = [7; 32];

    const _: fn() = || {
        fn assert_send<T: Send>() {}
        fn assert_rng<T: RngCore + CryptoRng>() {}
        assert_send::<PrivateRng<Shake128>>();
        assert_send::<ProverCoins<'_>>();
        assert_send::<OwnedCoins>();
        assert_rng::<ProverCoins<'_>>();
        #[cfg(feature = "prover-streams")]
        {
            use zeroize::ZeroizeOnDrop;
            fn assert_wipes<T: ZeroizeOnDrop>() {}
            assert_send::<ChildCoins>();
            assert_rng::<ChildCoins>();
            assert_wipes::<ChaCha20Rng>();
        }
    };

    #[test]
    fn adapter_and_hedges_match_private_rng() {
        let mut transcript = ProverState::<Shake128>::init_seeded(&SESSION, &0u64, SEED);
        let mut owned = OwnedCoins(PrivateRng::from_seed(SEED));
        let mut expected = PrivateRng::<Shake128>::from_seed(SEED);

        let mut word = [0; 4];
        expected.fill_bytes(&mut word);
        assert_eq!(transcript.coins().next_u32(), u32::from_le_bytes(word));
        assert_eq!(owned.coins().next_u32(), u32::from_le_bytes(word));
        let mut word = [0; 8];
        expected.fill_bytes(&mut word);
        assert_eq!(transcript.coins().next_u64(), u64::from_le_bytes(word));
        assert_eq!(owned.coins().next_u64(), u64::from_le_bytes(word));

        for hedge in [[3; 32], [5; 32]] {
            ProverTranscript::mix_entropy(&mut transcript, &hedge);
            owned.mix_entropy(&hedge);
            expected.mix_entropy(&hedge);
            let mut reference = [0; 337];
            let mut actual = [0; 337];
            expected.fill_bytes(&mut reference);
            transcript.coins().fill_bytes(&mut actual);
            assert_eq!(actual, reference);
            assert!(owned.coins().try_fill_bytes(&mut actual).is_ok());
            assert_eq!(actual, reference);
        }
    }

    #[test]
    fn private_operations_preserve_public_transcript_and_narg() {
        let statement = 42u64;
        let mut prover = ProverState::<Shake128>::init_seeded(&SESSION, &statement, SEED);
        ProverTranscript::mix_entropy(&mut prover, &[3; 32]);
        prover.coins().fill_bytes(&mut [0; 337]);
        #[cfg(feature = "prover-streams")]
        let mut child = prover.coins().child();
        prover.send(&BytesMsg(b"commitment"));
        #[cfg(feature = "prover-streams")]
        child.fill_bytes(&mut [0; 337]);
        let challenge = prover.squeeze(32, <[u8]>::to_vec);
        let narg = prover.narg_string();
        let mut verifier = verifier_transcript::<Shake128>(&SESSION, &statement, narg);
        let message: BytesMsg = VerifierTranscript::prover_message(&mut verifier).unwrap();
        assert_eq!(message.0, b"commitment");
        assert_eq!(verifier.squeeze(32, <[u8]>::to_vec), challenge);
        verifier.check_eof().unwrap();
    }

    #[cfg(feature = "prover-streams")]
    #[test]
    fn child_uses_the_next_32_bytes_as_a_chacha20_key() {
        let mut transcript = ProverState::<Shake128>::init_seeded(&SESSION, &0u64, SEED);
        let mut expected = PrivateRng::<Shake128>::from_seed(SEED);
        transcript.coins().fill_bytes(&mut [0; 19]);
        expected.fill_bytes(&mut [0; 19]);
        let mut key = [0; 32];
        expected.fill_bytes(&mut key);
        let mut reference = ChaCha20Rng::from_seed(key);
        let mut child = transcript.coins().child();
        assert_eq!(child.next_u32(), reference.next_u32());
        assert_eq!(child.next_u64(), reference.next_u64());
        for len in [0, 1, 63, 64, 257, 1024] {
            let mut actual = vec![0; len];
            let mut reference_bytes = vec![0; len];
            assert!(child.try_fill_bytes(&mut actual).is_ok());
            reference.fill_bytes(&mut reference_bytes);
            assert_eq!(actual, reference_bytes);
        }
        let mut parent = [0; 337];
        let mut reference_parent = [0; 337];
        transcript.coins().fill_bytes(&mut parent);
        expected.fill_bytes(&mut reference_parent);
        assert_eq!(parent, reference_parent);
    }

    #[cfg(feature = "test-utils")]
    #[test]
    fn scripted_transcript_delegates_private_operations() {
        use crate::ScriptedTranscript;
        use std::collections::VecDeque;

        type Scripted = ScriptedTranscript<ProverState<Shake128>, u64, VecDeque<u64>>;
        let mut transcript = Scripted::init_seeded(&SESSION, &42u64, SEED);
        let mut expected = PrivateRng::<Shake128>::from_seed(SEED);
        transcript.mix_entropy(&[3; 32]);
        expected.mix_entropy(&[3; 32]);
        let mut actual = [0; 337];
        let mut reference = [0; 337];
        transcript.coins().fill_bytes(&mut actual);
        expected.fill_bytes(&mut reference);
        assert_eq!(actual, reference);
        assert_eq!(transcript.events(), []);
    }
}
