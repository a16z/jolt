//! The 7x advice-bytes reconstruction sumcheck (lattice/packed mode only).
//!
//! One sumcheck over the untrusted-advice byte one-hot column's
//! `(symbol ‖ limb ‖ word)` cell domain, γ-batching three legs (see
//! `UntrustedAdviceReconstruction` in jolt-claims):
//!
//! - booleanity (`γ⁰`): `eq(cell, r_ref) · (B² − B)` sums to zero,
//! - hamming (`γ¹`): `eq((limb,word), r_ref_lw) · B` sums to one,
//! - word reconstruction (`γ²`): `id(symbol) · 256^limb · eq(word, r_word) ·
//!   B` sums to the completed untrusted-advice word claim.
//!
//! The reference point `r_ref` is drawn fresh over the cell domain before
//! the instance gamma; the word kernel binds the completed advice claim's
//! own point. The column opening produced here is the leaf claim the packed
//! stage-8 `UntrustedAdviceOneHot` statement consumes.

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use jolt_claims::protocols::jolt::lattice::geometry::{word_byte_num_vars, BYTE_BITS, WORD_BYTES};
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId, BIG_ENDIAN,
    LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::claim_reductions::advice::AdviceKind;

const DEGREE_BOUND: usize = 3;

#[derive(Allocative, Clone)]
pub struct UntrustedAdviceReconstructionSumcheckParams<F: JoltField> {
    pub word_vars: usize,
    /// The fresh reference point over the cell domain (msb-first), drawn
    /// before `gamma`.
    pub r_reference: Vec<F::Challenge>,
    pub gamma: F,
    /// The completed untrusted-advice word claim's point (the word kernel).
    pub r_word: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> UntrustedAdviceReconstructionSumcheckParams<F> {
    pub fn new(
        word_vars: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let r_reference = transcript.challenge_vector_optimized::<F>(word_byte_num_vars(word_vars));
        let gamma: F = transcript.challenge_scalar();
        let (r_word, _) = accumulator
            .get_advice_opening(AdviceKind::Untrusted, SumcheckId::AdviceClaimReduction)
            .expect("completed untrusted advice claim must exist before the 7x phase");
        Self {
            word_vars,
            r_reference,
            gamma,
            r_word,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for UntrustedAdviceReconstructionSumcheckParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        word_byte_num_vars(self.word_vars)
    }

    /// Booleanity sums to zero, hamming to one, reconstruction to the
    /// completed word claim: `γ + γ²·A(r_word)`.
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, word_claim) = accumulator
            .get_advice_opening(AdviceKind::Untrusted, SumcheckId::AdviceClaimReduction)
            .expect("completed untrusted advice claim must exist before the 7x phase");
        self.gamma + self.gamma * self.gamma * word_claim
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
            "zk x lattice is rejected fail-closed; UntrustedAdviceReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; UntrustedAdviceReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; UntrustedAdviceReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, _sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; UntrustedAdviceReconstruction carries no BlindFold plumbing"
        )
    }
}

#[derive(Allocative)]
pub struct UntrustedAdviceReconstructionSumcheckProver<F: JoltField> {
    /// The 0/1 byte one-hot column over the cell domain.
    bytes: MultilinearPolynomial<F>,
    /// The booleanity kernel `eq(cell, r_ref)`.
    k_bool: MultilinearPolynomial<F>,
    /// The linear kernel `γ·eq_lw + γ²·id·place·eq_word`, multiplying the
    /// column directly (the hamming and reconstruction legs).
    k_lin: MultilinearPolynomial<F>,
    pub params: UntrustedAdviceReconstructionSumcheckParams<F>,
}

impl<F: JoltField> UntrustedAdviceReconstructionSumcheckProver<F> {
    #[tracing::instrument(
        skip_all,
        name = "UntrustedAdviceReconstructionSumcheckProver::initialize"
    )]
    pub fn initialize(
        params: UntrustedAdviceReconstructionSumcheckParams<F>,
        words: &[u64],
    ) -> Self {
        let word_vars = params.word_vars;
        let limb_bits = WORD_BYTES.log_2();
        let cell_vars = word_byte_num_vars(word_vars);
        debug_assert!(words.len() <= 1 << word_vars);
        debug_assert_eq!(cell_vars, BYTE_BITS + limb_bits + word_vars);

        let mut bytes = vec![0u8; 1 << cell_vars];
        for limb in 0..WORD_BYTES {
            for word_index in 0..(1usize << word_vars) {
                let byte = words
                    .get(word_index)
                    .map_or(0, |word| (word >> (8 * limb)) as u8)
                    as usize;
                bytes[(((byte << limb_bits) | limb) << word_vars) | word_index] = 1;
            }
        }

        let k_bool = EqPolynomial::<F>::evals(&params.r_reference);
        let eq_lw = EqPolynomial::<F>::evals(&params.r_reference[BYTE_BITS..]);
        let eq_word = EqPolynomial::<F>::evals(&params.r_word.r);
        let gamma = params.gamma;
        let gamma_squared = gamma * gamma;
        let k_lin = (0..1usize << cell_vars)
            .into_par_iter()
            .map(|cell| {
                let word_index = cell & ((1 << word_vars) - 1);
                let limb = (cell >> word_vars) & (WORD_BYTES - 1);
                let symbol = cell >> (word_vars + limb_bits);
                let place = F::from_u64(1u64 << (8 * limb));
                gamma * eq_lw[cell & ((1 << (limb_bits + word_vars)) - 1)]
                    + gamma_squared * F::from_u64(symbol as u64) * place * eq_word[word_index]
            })
            .collect::<Vec<F>>();

        Self {
            bytes: MultilinearPolynomial::from(bytes),
            k_bool: MultilinearPolynomial::from(k_bool),
            k_lin: MultilinearPolynomial::from(k_lin),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for UntrustedAdviceReconstructionSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(
        skip_all,
        name = "UntrustedAdviceReconstructionSumcheckProver::compute_message"
    )]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let half = self.bytes.len() / 2;
        let [eval_0, eval_2, eval_3] = (0..half)
            .into_par_iter()
            .map(|g| {
                let b0 = self.bytes.get_bound_coeff(2 * g);
                let b1 = self.bytes.get_bound_coeff(2 * g + 1);
                let kb0 = self.k_bool.get_bound_coeff(2 * g);
                let kb1 = self.k_bool.get_bound_coeff(2 * g + 1);
                let kl0 = self.k_lin.get_bound_coeff(2 * g);
                let kl1 = self.k_lin.get_bound_coeff(2 * g + 1);
                let b_delta = b1 - b0;
                let kb_delta = kb1 - kb0;
                let kl_delta = kl1 - kl0;
                let term = |b: F, kb: F, kl: F| kb * (b.square() - b) + kl * b;
                let b2 = b1 + b_delta;
                let b3 = b2 + b_delta;
                [
                    term(b0, kb0, kl0),
                    term(b2, kb1 + kb_delta, kl1 + kl_delta),
                    term(b3, kb1 + kb_delta + kb_delta, kl1 + kl_delta + kl_delta),
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
            );
        UniPoly::from_evals(&[eval_0, previous_claim - eval_0, eval_2, eval_3])
    }

    #[tracing::instrument(
        skip_all,
        name = "UntrustedAdviceReconstructionSumcheckProver::ingest_challenge"
    )]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.bytes.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.k_bool.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.k_lin.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        accumulator.append_untrusted_advice(
            SumcheckId::UntrustedAdviceReconstruction,
            self.params.normalize_opening_point(sumcheck_challenges),
            self.bytes.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// The trusted-advice byte reconstruction: the standalone degree-2
/// reconstruction leg (`id(symbol) · 256^limb · eq(word, r_word) · B`) over
/// the precommitted trusted byte column — no booleanity or hamming legs (the
/// trusted committer attests the encoding), no drawn challenges.
#[derive(Allocative, Clone)]
pub struct TrustedAdviceReconstructionSumcheckParams<F: JoltField> {
    pub word_vars: usize,
    /// The completed trusted-advice word claim's point (the word kernel).
    pub r_word: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> TrustedAdviceReconstructionSumcheckParams<F> {
    pub fn new(word_vars: usize, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let (r_word, _) = accumulator
            .get_advice_opening(AdviceKind::Trusted, SumcheckId::AdviceClaimReduction)
            .expect("completed trusted advice claim must exist before the 7x phase");
        Self { word_vars, r_word }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for TrustedAdviceReconstructionSumcheckParams<F> {
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
            .get_advice_opening(AdviceKind::Trusted, SumcheckId::AdviceClaimReduction)
            .expect("completed trusted advice claim must exist before the 7x phase")
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
            "zk x lattice is rejected fail-closed; TrustedAdviceReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; TrustedAdviceReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; TrustedAdviceReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, _sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; TrustedAdviceReconstruction carries no BlindFold plumbing"
        )
    }
}

#[derive(Allocative)]
pub struct TrustedAdviceReconstructionSumcheckProver<F: JoltField> {
    /// The byte one-hot column pre-bound at `r_word` over the word
    /// variables: a scatter of `eq(r_word, ·)` into the `(byte ‖ place)`
    /// cells.
    bytes: MultilinearPolynomial<F>,
    /// The decode kernel `byte · 256^place`.
    kernel: MultilinearPolynomial<F>,
    pub params: TrustedAdviceReconstructionSumcheckParams<F>,
}

impl<F: JoltField> TrustedAdviceReconstructionSumcheckProver<F> {
    #[tracing::instrument(
        skip_all,
        name = "TrustedAdviceReconstructionSumcheckProver::initialize"
    )]
    pub fn initialize(params: TrustedAdviceReconstructionSumcheckParams<F>, words: &[u64]) -> Self {
        let word_vars = params.word_vars;
        let limb_bits = WORD_BYTES.log_2();
        let cell_vars = word_byte_num_vars(0);
        debug_assert!(words.len() <= 1 << word_vars);
        debug_assert_eq!(params.r_word.r.len(), word_vars);

        // Zero-padded words scatter byte 0 (the witness encodes padding as
        // symbol-0 hot), so the column matches the committed one exactly.
        let eq_word = EqPolynomial::<F>::evals(&params.r_word.r);
        let mut bytes = vec![F::zero(); 1 << cell_vars];
        for limb in 0..WORD_BYTES {
            for word_index in 0..(1usize << word_vars) {
                let byte = words
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

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for TrustedAdviceReconstructionSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(
        skip_all,
        name = "TrustedAdviceReconstructionSumcheckProver::compute_message"
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

    #[tracing::instrument(
        skip_all,
        name = "TrustedAdviceReconstructionSumcheckProver::ingest_challenge"
    )]
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
        accumulator.append_trusted_advice(
            SumcheckId::TrustedAdviceReconstruction,
            OpeningPoint::new(point),
            self.bytes.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::poly::multilinear_polynomial::PolynomialEvaluation;
    use crate::transcripts::Blake2bTranscript;
    use ark_bn254::Fr;

    type Challenge = <Fr as JoltField>::Challenge;

    const WORD_VARS: usize = 2;

    /// Drives the full round loop over a synthetic advice column and checks
    /// every message folds the previous claim, the final claim equals the
    /// legacy verifier half's closed form, and the cached byte opening
    /// decodes directly against the column — pinning the msb-first cell
    /// conventions non-circularly.
    #[test]
    fn round_loop_reduces_to_the_byte_column_opening() {
        let words: Vec<u64> = vec![0x0102030405060708, 0, u64::MAX, 0xdeadbeef];
        let r_word: Vec<Challenge> = (0..WORD_VARS)
            .map(|i| Challenge::from((31 + 7 * i as u64) as u128))
            .collect();
        let word_claim = MultilinearPolynomial::<Fr>::from(words.clone()).evaluate(&r_word);

        let mut accumulator = ProverOpeningAccumulator::<Fr>::new(WORD_VARS);
        accumulator.append_untrusted_advice(
            SumcheckId::AdviceClaimReduction,
            OpeningPoint::new(r_word.clone()),
            word_claim,
        );

        let mut transcript = Blake2bTranscript::new(b"advice-bytes-test");
        let params = UntrustedAdviceReconstructionSumcheckParams::<Fr>::new(
            WORD_VARS,
            &accumulator,
            &mut transcript,
        );
        let mut prover =
            UntrustedAdviceReconstructionSumcheckProver::initialize(params.clone(), &words);

        let mut claim = params.input_claim(&accumulator);
        let mut challenges = Vec::new();
        for round in 0..params.num_rounds() {
            let message = SumcheckInstanceProver::<Fr, Blake2bTranscript>::compute_message(
                &mut prover,
                round,
                claim,
            );
            assert_eq!(
                message.eval_at_zero() + message.eval_at_one(),
                claim,
                "round {round} message must fold the previous claim"
            );
            let r_j = Challenge::from((211 + 13 * round) as u128);
            claim = message.evaluate(&r_j);
            challenges.push(r_j);
            SumcheckInstanceProver::<Fr, Blake2bTranscript>::ingest_challenge(
                &mut prover,
                r_j,
                round,
            );
        }
        SumcheckInstanceProver::<Fr, Blake2bTranscript>::cache_openings(
            &prover,
            &mut accumulator,
            &challenges,
        );

        // The verifier's closed form over the cached opening: the same five
        // publics the jolt-verifier ConcreteSumcheck derives (eq splits,
        // msb-first identity and place kernels).
        let opening_point = params.normalize_opening_point(&challenges);
        let (_, bytes_claim) = accumulator
            .get_advice_opening(
                AdviceKind::Untrusted,
                SumcheckId::UntrustedAdviceReconstruction,
            )
            .unwrap();
        let limb_bits = WORD_BYTES.log_2();
        let point: Vec<Fr> = opening_point.r.iter().map(|&c| c.into()).collect();
        let reference: Vec<Fr> = params.r_reference.iter().map(|&c| c.into()).collect();
        let (r_symbol, r_limb_word) = point.split_at(BYTE_BITS);
        let (r_limb, r_word_bound) = r_limb_word.split_at(limb_bits);
        let r_word_field: Vec<Fr> = r_word.iter().map(|&c| c.into()).collect();
        let eq_cell_mle: Fr = EqPolynomial::mle(&point, &reference);
        let eq_limb_word: Fr = EqPolynomial::mle(r_limb_word, &reference[BYTE_BITS..]);
        let eq_word: Fr = EqPolynomial::mle(r_word_bound, &r_word_field);
        let identity_at_symbol = r_symbol
            .iter()
            .fold(Fr::from_u64(0), |acc, coordinate| acc + acc + *coordinate);
        let place_at_limb =
            r_limb
                .iter()
                .enumerate()
                .fold(Fr::from_u64(1), |acc, (position, coordinate)| {
                    let weight = 1usize << (limb_bits - 1 - position);
                    let place = Fr::from_u64(1u64 << (8 * weight));
                    acc * ((place - Fr::from_u64(1)) * *coordinate + Fr::from_u64(1))
                });
        let gamma = params.gamma;
        let expected = eq_cell_mle * (bytes_claim.square() - bytes_claim)
            + gamma * eq_limb_word * bytes_claim
            + gamma * gamma * identity_at_symbol * place_at_limb * eq_word * bytes_claim;
        assert_eq!(claim, expected, "final claim must equal the closed form");

        // The cached opening must equal the byte column evaluated directly at
        // the produced msb-first cell point.
        let cell_vars = word_byte_num_vars(WORD_VARS);
        let eq_cell = EqPolynomial::<Fr>::evals(&opening_point.r);
        let limb_bits = WORD_BYTES.log_2();
        let mut direct = Fr::from_u64(0);
        for limb in 0..WORD_BYTES {
            for (word_index, word) in words.iter().enumerate() {
                let byte = ((word >> (8 * limb)) & 0xff) as usize;
                direct += eq_cell[(((byte << limb_bits) | limb) << WORD_VARS) | word_index];
            }
        }
        assert_eq!(cell_vars, opening_point.len());
        assert_eq!(bytes_claim, direct, "byte opening must decode the column");
    }
}
