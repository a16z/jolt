use std::ops::{Deref, DerefMut};

use dory::{setup, ProverSetup, VerifierSetup};
use rand::RngCore;

use crate::poly::commitment::dory::JoltBn254;

#[derive(Clone, Debug)]
pub struct DoryProverSetup(pub ProverSetup<JoltBn254>);

#[derive(Clone, Debug)]
pub struct DoryVerifierSetup(pub VerifierSetup<JoltBn254>);

impl DoryProverSetup {
    /// Generate new prover setup with transparent randomness
    ///
    /// For square matrices, generates n = 2^((max_log_n+1)/2) generators for both G1 and G2,
    /// supporting polynomials up to 2^max_log_n coefficients arranged as n×n matrices.
    ///
    /// # Parameters
    /// - `rng`: Random number generator
    /// - `max_log_n`: Maximum log₂ of polynomial size (for n×n matrix with n² = 2^max_log_n)
    pub fn new<R: RngCore>(rng: &mut R, max_log_n: usize) -> Self {
        Self(ProverSetup::new(rng, max_log_n))
    }

    /// Load prover setup from disk cache, or generate and cache if not available
    pub fn new_from_urs<R: RngCore>(rng: &mut R, max_log_n: usize) -> Self {
        let (prover_setup, _) = setup::<JoltBn254, _>(rng, max_log_n);
        Self(prover_setup)
    }

    /// Derive verifier setup from this prover setup
    pub fn to_verifier_setup(&self) -> DoryVerifierSetup {
        DoryVerifierSetup(self.0.to_verifier_setup())
    }

    /// Unwrap into inner `ProverSetup<BN254>`
    pub fn into_inner(self) -> ProverSetup<JoltBn254> {
        self.0
    }
}

impl DoryVerifierSetup {
    /// Unwrap into inner `VerifierSetup<BN254>`
    pub fn into_inner(self) -> VerifierSetup<JoltBn254> {
        self.0
    }
}

impl From<ProverSetup<JoltBn254>> for DoryProverSetup {
    fn from(setup: ProverSetup<JoltBn254>) -> Self {
        Self(setup)
    }
}

impl From<DoryProverSetup> for ProverSetup<JoltBn254> {
    fn from(setup: DoryProverSetup) -> Self {
        setup.0
    }
}

impl From<VerifierSetup<JoltBn254>> for DoryVerifierSetup {
    fn from(setup: VerifierSetup<JoltBn254>) -> Self {
        Self(setup)
    }
}

impl From<DoryVerifierSetup> for VerifierSetup<JoltBn254> {
    fn from(setup: DoryVerifierSetup) -> Self {
        setup.0
    }
}

impl Deref for DoryProverSetup {
    type Target = ProverSetup<JoltBn254>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for DoryProverSetup {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Deref for DoryVerifierSetup {
    type Target = VerifierSetup<JoltBn254>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for DoryVerifierSetup {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
