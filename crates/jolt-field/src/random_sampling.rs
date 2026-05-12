use rand_core::RngCore;

/// RNG-backed sampling for tests and witnesses.
pub trait RandomSampling {
    /// Samples a random element.
    fn random<R: RngCore>(rng: &mut R) -> Self;
}
