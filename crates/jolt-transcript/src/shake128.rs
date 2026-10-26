//! SHAKE128 (FIPS 202) as a spongefish sponge.

use shake::digest::{ExtendableOutput, Update};
use spongefish::instantiations::{XofRate, XOF};

/// The SHAKE128 suite of draft-irtf-cfrg-fiat-shamir, over the `shake` crate.
///
/// spongefish 0.8 defines the same suite but builds it only together with a
/// non-FIPS XOF, so this crate supplies its own. It derives the session
/// identifier ([`crate::session_id`]) and the prover's private coins
/// ([`ProverState`]).
pub type Shake128 = XOF<Shake128Hasher>;

/// [`shake::Shake128`] with the rate spongefish's [`XOF`] bridge reads.
#[derive(Clone, Default)]
pub struct Shake128Hasher(shake::Shake128);

impl Update for Shake128Hasher {
    fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }
}

impl ExtendableOutput for Shake128Hasher {
    type Reader = shake::Shake128Reader;

    fn finalize_xof(self) -> Self::Reader {
        self.0.finalize_xof()
    }
}

impl XofRate for Shake128Hasher {
    /// SHAKE128's rate in bytes.
    const RATE: usize = 168;
}

/// Spongefish's prover state over sponge `H`, with [`Shake128`] for the
/// prover's private coins.
pub type ProverState<H> = spongefish::ProverState<H, Shake128>;
