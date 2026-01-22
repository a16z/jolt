use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::{DoryContext, DoryGlobals};
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::zkvm::ram::RAMPreprocessing;

/// Trusted commitment to the initial RAM program-image words polynomial.
///
/// This commits to the *packed* `u64` words emitted by `RAMPreprocessing::preprocess(memory_init)`,
/// padded to a power-of-two length with trailing zeros.
///
/// The verifier treats this as a preprocessing-time trust anchor in `BytecodeMode::Committed`.
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct TrustedProgramImageCommitment<PCS: CommitmentScheme> {
    pub commitment: PCS::Commitment,
    /// Unpadded number of program-image words (may be 0).
    pub unpadded_len_words: usize,
    /// Power-of-two padded length used for the committed polynomial (minimum 1).
    pub padded_len_words: usize,
}

impl<PCS: CommitmentScheme> TrustedProgramImageCommitment<PCS> {
    /// Derive the trusted commitment from the program-image words in RAM preprocessing.
    ///
    /// Returns the trusted commitment and a PCS opening-proof hint for Stage 8 batching.
    pub fn derive(
        ram_preprocessing: &RAMPreprocessing,
        generators: &PCS::ProverSetup,
    ) -> (Self, PCS::OpeningProofHint) {
        let unpadded_len_words = ram_preprocessing.bytecode_words.len();
        let padded_len_words = unpadded_len_words.next_power_of_two().max(1);

        let mut coeffs = ram_preprocessing.bytecode_words.clone();
        coeffs.resize(padded_len_words, 0u64);
        let poly: MultilinearPolynomial<PCS::Field> = MultilinearPolynomial::from(coeffs);

        // Program-image commitment lives in its own Dory context.
        DoryGlobals::initialize_context(1, padded_len_words, DoryContext::ProgramImage, None);
        let _ctx = DoryGlobals::with_context(DoryContext::ProgramImage);

        let (commitment, hint) = PCS::commit(&poly, generators);
        (
            Self {
                commitment,
                unpadded_len_words,
                padded_len_words,
            },
            hint,
        )
    }

    /// Build the (padded) program-image polynomial to be included in the Stage 8 streaming RLC.
    pub fn build_polynomial<F: crate::field::JoltField>(
        ram_preprocessing: &RAMPreprocessing,
        padded_len_words: usize,
    ) -> MultilinearPolynomial<F> {
        debug_assert!(padded_len_words.is_power_of_two());
        debug_assert!(padded_len_words > 0);

        let mut coeffs = ram_preprocessing.bytecode_words.clone();
        coeffs.resize(padded_len_words, 0u64);
        MultilinearPolynomial::from(coeffs)
    }
}

