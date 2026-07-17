//! Program-image byte reconstruction sumcheck (lattice/packed mode).
//!
//! The trusted-advice decode shape over the program image byte column:
//! settles the completed `ProgramImageInit` claim against the precommitted
//! `ProgramImageBytes` one-hot column of `W_prog`. Only the `(byte ‖ place)`
//! variables bind (the word point stays fixed by the incoming claim); the
//! single decode leg `byte · 256^place` is degree 2, and the produced byte
//! opening lands at `(bound ‖ r_word)` — the column's own packed-slot point.

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
#[cfg(feature = "prover")]
use rayon::prelude::*;

use crate::field::JoltField;
#[cfg(feature = "prover")]
use crate::poly::eq_poly::EqPolynomial;
#[cfg(feature = "prover")]
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
#[cfg(feature = "prover")]
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, SumcheckId, BIG_ENDIAN, LITTLE_ENDIAN,
};
#[cfg(feature = "prover")]
use crate::poly::unipoly::UniPoly;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint};
#[cfg(feature = "prover")]
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::witness::CommittedPolynomial;
use jolt_claims::protocols::jolt::lattice::geometry::{word_byte_num_vars, WORD_BYTES};

#[derive(Allocative, Clone)]
pub struct ProgramImageReconstructionSumcheckParams<F: JoltField> {
    /// The completed program-image claim's point (the fixed word point).
    pub r_word: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> ProgramImageReconstructionSumcheckParams<F> {
    pub fn new(accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let (r_word, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::ProgramImageInit,
            SumcheckId::ProgramImageClaimReduction,
        );
        Self { r_word }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ProgramImageReconstructionSumcheckParams<F> {
    fn degree(&self) -> usize {
        2
    }

    /// Only the `(byte ‖ place)` variables bind; the word point stays fixed
    /// by the incoming claim (the byte column is pre-bound at `r_word`).
    fn num_rounds(&self) -> usize {
        word_byte_num_vars(0)
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::ProgramImageInit,
                SumcheckId::ProgramImageClaimReduction,
            )
            .1
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        unimplemented!(
            "zk x lattice is rejected fail-closed; ProgramImageReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; ProgramImageReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; ProgramImageReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, _sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; ProgramImageReconstruction carries no BlindFold plumbing"
        )
    }
}

#[cfg(feature = "prover")]
#[derive(Allocative)]
pub struct ProgramImageReconstructionSumcheckProver<F: JoltField> {
    /// The byte one-hot column pre-bound at `r_word` over the word
    /// variables: a scatter of `eq(r_word, ·)` into the `(byte ‖ place)`
    /// cells.
    bytes: MultilinearPolynomial<F>,
    /// The decode kernel `byte · 256^place`.
    kernel: MultilinearPolynomial<F>,
    pub params: ProgramImageReconstructionSumcheckParams<F>,
}

#[cfg(feature = "prover")]
impl<F: JoltField> ProgramImageReconstructionSumcheckProver<F> {
    #[tracing::instrument(
        skip_all,
        name = "ProgramImageReconstructionSumcheckProver::initialize"
    )]
    pub fn initialize(
        params: ProgramImageReconstructionSumcheckParams<F>,
        image_words: &[u64],
    ) -> Self {
        let word_vars = params.r_word.r.len();
        let limb_bits = WORD_BYTES.log_2();
        let cell_vars = word_byte_num_vars(0);
        debug_assert!(image_words.len() <= 1 << word_vars);

        // Zero-padded words scatter byte 0 (the witness encodes padding as
        // symbol-0 hot), so the column matches the committed one exactly.
        let eq_word = EqPolynomial::<F>::evals(&params.r_word.r);
        let mut bytes = vec![F::zero(); 1 << cell_vars];
        for limb in 0..WORD_BYTES {
            for word_index in 0..(1usize << word_vars) {
                let byte = image_words
                    .get(word_index)
                    .map_or(0, |word| (word >> (8 * limb)) as u8)
                    as usize;
                bytes[(byte << limb_bits) | limb] += eq_word[word_index];
            }
        }

        let kernel = (0..1usize << cell_vars)
            .into_par_iter()
            .map(|cell| {
                let limb = cell & (WORD_BYTES - 1);
                let symbol = cell >> limb_bits;
                F::from_u64(symbol as u64) * F::from_u64(1u64 << (8 * limb))
            })
            .collect::<Vec<F>>();

        Self {
            bytes: MultilinearPolynomial::from(bytes),
            kernel: MultilinearPolynomial::from(kernel),
            params,
        }
    }
}

#[cfg(feature = "prover")]
impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for ProgramImageReconstructionSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(
        skip_all,
        name = "ProgramImageReconstructionSumcheckProver::compute_message"
    )]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let half = self.bytes.len() / 2;
        let [eval_0, eval_2] = (0..half)
            .into_par_iter()
            .map(|g| {
                let b0 = self.bytes.get_bound_coeff(2 * g);
                let b1 = self.bytes.get_bound_coeff(2 * g + 1);
                let k0 = self.kernel.get_bound_coeff(2 * g);
                let k1 = self.kernel.get_bound_coeff(2 * g + 1);
                [k0 * b0, (k1 + (k1 - k0)) * (b1 + (b1 - b0))]
            })
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
        UniPoly::from_evals(&[eval_0, previous_claim - eval_0, eval_2])
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.bytes.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.kernel.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // The byte-column opening lands at the column's own packed-slot
        // point: the bound (byte ‖ place) prefix suffixed with the fixed
        // word point.
        let mut point: Vec<F::Challenge> = sumcheck_challenges.to_vec();
        point.reverse();
        point.extend_from_slice(&self.params.r_word.r);
        accumulator.append_dense(
            CommittedPolynomial::ProgramImageBytes,
            SumcheckId::ProgramImageReconstruction,
            point,
            self.bytes.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
