//! Program-image (initial RAM) claim reduction.
//!
//! In committed bytecode mode, Stage 4 consumes prover-supplied scalar claims for the
//! program-image contribution to `Val_init(r_address)` without materializing the initial RAM.
//! This sumcheck binds those scalars to a trusted commitment to the program-image words polynomial.

use allocative::Allocative;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::config::ReadWriteConfig;
use crate::zkvm::ram::remap_address;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use tracer::JoltDevice;

const DEGREE_BOUND: usize = 2;

#[derive(Clone, Allocative)]
pub struct ProgramImageClaimReductionParams<F: JoltField> {
    pub gamma: F,
    pub single_opening: bool,
    pub ram_num_vars: usize,
    pub start_index: usize,
    pub padded_len_words: usize,
    pub m: usize,
    pub r_addr_rw: Vec<F::Challenge>,
    pub r_addr_raf: Option<Vec<F::Challenge>>,
}

impl<F: JoltField> ProgramImageClaimReductionParams<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        program_io: &JoltDevice,
        ram_min_bytecode_address: u64,
        padded_len_words: usize,
        ram_K: usize,
        trace_len: usize,
        rw_config: &ReadWriteConfig,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let ram_num_vars = ram_K.log_2();
        let start_index =
            remap_address(ram_min_bytecode_address, &program_io.memory_layout).unwrap() as usize;
        let m = padded_len_words.log_2();
        debug_assert!(padded_len_words.is_power_of_two());
        debug_assert!(padded_len_words > 0);

        // r_address_rw comes from RamVal/RamReadWriteChecking (Stage 2).
        let (r_rw, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_addr_rw, _) = r_rw.split_at(ram_num_vars);

        // r_address_raf comes from RamValFinal/RamOutputCheck (Stage 2), but may equal r_address_rw.
        let log_t = trace_len.log_2();
        let single_opening = rw_config.needs_single_advice_opening(log_t);
        let r_addr_raf = if single_opening {
            None
        } else {
            let (r_raf, _) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            );
            let (r_addr_raf, _) = r_raf.split_at(ram_num_vars);
            Some(r_addr_raf.r)
        };

        // Sample gamma for combining rw + raf.
        let gamma: F = transcript.challenge_scalar();

        Self {
            gamma,
            single_opening,
            ram_num_vars,
            start_index,
            padded_len_words,
            m,
            r_addr_rw: r_addr_rw.r,
            r_addr_raf,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ProgramImageClaimReductionParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        // Scalar claims were staged in Stage 4 as virtual openings.
        let (_, c_rw) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::ProgramImageInitContributionRw,
            SumcheckId::RamValEvaluation,
        );
        if self.single_opening {
            c_rw
        } else {
            let (_, c_raf) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::ProgramImageInitContributionRaf,
                SumcheckId::RamValFinalEvaluation,
            );
            c_rw + self.gamma * c_raf
        }
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.m
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        // Challenges are in little-endian round order (LSB first) when binding LowToHigh.
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

#[derive(Allocative)]
pub struct ProgramImageClaimReductionProver<F: JoltField> {
    pub params: ProgramImageClaimReductionParams<F>,
    program_word: MultilinearPolynomial<F>,
    eq_slice: MultilinearPolynomial<F>,
    /// Number of trailing dummy rounds in a batched Stage 6b sumcheck.
    batch_dummy_rounds: AtomicUsize,
}

fn build_eq_slice_table<F: JoltField>(
    r_addr: &[F::Challenge],
    start_index: usize,
    len: usize,
) -> Vec<F> {
    debug_assert!(len.is_power_of_two());
    let mut out = Vec::with_capacity(len);
    let mut idx = start_index;
    let mut off = 0usize;
    while off < len {
        let remaining = len - off;
        let (block_size, block_evals) =
            EqPolynomial::<F>::evals_for_max_aligned_block(r_addr, idx, remaining);
        out.extend_from_slice(&block_evals);
        idx += block_size;
        off += block_size;
    }
    debug_assert_eq!(out.len(), len);
    out
}

impl<F: JoltField> ProgramImageClaimReductionProver<F> {
    #[tracing::instrument(skip_all, name = "ProgramImageClaimReductionProver::initialize")]
    pub fn initialize(
        params: ProgramImageClaimReductionParams<F>,
        program_image_words_padded: Vec<u64>,
    ) -> Self {
        debug_assert_eq!(program_image_words_padded.len(), params.padded_len_words);
        debug_assert_eq!(params.padded_len_words, 1usize << params.m);

        let program_word: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(program_image_words_padded);

        let eq_rw = build_eq_slice_table::<F>(
            &params.r_addr_rw,
            params.start_index,
            params.padded_len_words,
        );
        let mut eq_comb = eq_rw;
        if !params.single_opening {
            let r_raf = params.r_addr_raf.as_ref().expect("missing raf address");
            let eq_raf =
                build_eq_slice_table::<F>(r_raf, params.start_index, params.padded_len_words);
            for (c, e) in eq_comb.iter_mut().zip(eq_raf.iter()) {
                *c += params.gamma * *e;
            }
        }
        let eq_slice: MultilinearPolynomial<F> = MultilinearPolynomial::from(eq_comb);

        Self {
            params,
            program_word,
            eq_slice,
            batch_dummy_rounds: AtomicUsize::new(0),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for ProgramImageClaimReductionProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn round_offset(&self, max_num_rounds: usize) -> usize {
        // Align to the *start* of the Stage 6b challenge vector so that the resulting
        // big-endian opening point is the suffix (LSB side) of the full log_T cycle point.
        // This is required for Stage 8 embedding when log_T > m.
        let dummy_rounds = max_num_rounds.saturating_sub(self.params.num_rounds());
        self.batch_dummy_rounds
            .store(dummy_rounds, Ordering::Relaxed);
        0
    }

    #[tracing::instrument(skip_all, name = "ProgramImageClaimReductionProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let half = self.program_word.len() / 2;
        let program_word = &self.program_word;
        let eq_slice = &self.eq_slice;
        let mut evals: [F; DEGREE_BOUND] = (0..half)
            .into_par_iter()
            .map(|j| {
                let pw =
                    program_word.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let eq = eq_slice.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let mut out = [F::zero(); DEGREE_BOUND];
                for i in 0..DEGREE_BOUND {
                    out[i] = pw[i] * eq[i];
                }
                out
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |mut acc, arr| {
                    acc.iter_mut().zip(arr.iter()).for_each(|(a, b)| *a += *b);
                    acc
                },
            );
        // If this instance has trailing dummy rounds, `previous_claim` is scaled by 2^{dummy_rounds}
        // in the batched sumcheck. Scale the per-round univariate evaluations accordingly so the
        // sumcheck consistency checks pass (mirrors BytecodeClaimReduction).
        let dummy_rounds = self.batch_dummy_rounds.load(Ordering::Relaxed);
        if dummy_rounds != 0 {
            let scale = F::one().mul_pow_2(dummy_rounds);
            for e in evals.iter_mut() {
                *e *= scale;
            }
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    #[tracing::instrument(skip_all, name = "ProgramImageClaimReductionProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.program_word
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_slice.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let claim = self.program_word.final_sumcheck_claim();
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::ProgramImageInit,
            SumcheckId::ProgramImageClaimReduction,
            opening_point.r,
            claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct ProgramImageClaimReductionVerifier<F: JoltField> {
    pub params: ProgramImageClaimReductionParams<F>,
}

fn eval_eq_slice_at_r_star_lsb_dp<F: JoltField>(
    r_addr_be: &[F::Challenge],
    start_index: usize,
    m: usize,
    r_star_lsb: &[F::Challenge],
) -> F {
    let ell = r_addr_be.len();
    debug_assert_eq!(r_star_lsb.len(), m);
    debug_assert!(m <= ell);

    // DP over carry bit, iterating LSB -> MSB across the RAM address bits.
    let mut dp0 = F::one(); // carry=0
    let mut dp1 = F::zero(); // carry=1

    for i in 0..ell {
        let start_bit = ((start_index >> i) & 1) as u8;
        let y_var = i < m;
        let r_y: F = if y_var {
            r_star_lsb[i].into()
        } else {
            F::zero()
        };

        let r_addr_bit: F = r_addr_be[ell - 1 - i].into(); // LSB-first mapping
        let k0 = F::one() - r_addr_bit;
        let k1 = r_addr_bit;

        let mut ndp0 = F::zero();
        let mut ndp1 = F::zero();

        // Transition from carry=0
        if !dp0.is_zero() {
            if y_var {
                // y=0
                let sum0 = start_bit;
                let k_bit0 = sum0 & 1;
                let carry0 = (sum0 >> 1) & 1;
                let addr_factor0 = if k_bit0 == 1 { k1 } else { k0 };
                let y_factor0 = F::one() - r_y;
                if carry0 == 0 {
                    ndp0 += dp0 * addr_factor0 * y_factor0;
                } else {
                    ndp1 += dp0 * addr_factor0 * y_factor0;
                }
                // y=1
                let sum1 = start_bit + 1;
                let k_bit1 = sum1 & 1;
                let carry1 = (sum1 >> 1) & 1;
                let addr_factor1 = if k_bit1 == 1 { k1 } else { k0 };
                let y_factor1 = r_y;
                if carry1 == 0 {
                    ndp0 += dp0 * addr_factor1 * y_factor1;
                } else {
                    ndp1 += dp0 * addr_factor1 * y_factor1;
                }
            } else {
                // y is fixed 0
                let sum0 = start_bit;
                let k_bit0 = sum0 & 1;
                let carry0 = (sum0 >> 1) & 1;
                let addr_factor0 = if k_bit0 == 1 { k1 } else { k0 };
                if carry0 == 0 {
                    ndp0 += dp0 * addr_factor0;
                } else {
                    ndp1 += dp0 * addr_factor0;
                }
            }
        }

        // Transition from carry=1
        if !dp1.is_zero() {
            if y_var {
                // y=0
                let sum0 = start_bit + 1;
                let k_bit0 = sum0 & 1;
                let carry0 = (sum0 >> 1) & 1;
                let addr_factor0 = if k_bit0 == 1 { k1 } else { k0 };
                let y_factor0 = F::one() - r_y;
                if carry0 == 0 {
                    ndp0 += dp1 * addr_factor0 * y_factor0;
                } else {
                    ndp1 += dp1 * addr_factor0 * y_factor0;
                }
                // y=1
                let sum1 = start_bit + 1 + 1;
                let k_bit1 = sum1 & 1;
                let carry1 = (sum1 >> 1) & 1;
                let addr_factor1 = if k_bit1 == 1 { k1 } else { k0 };
                let y_factor1 = r_y;
                if carry1 == 0 {
                    ndp0 += dp1 * addr_factor1 * y_factor1;
                } else {
                    ndp1 += dp1 * addr_factor1 * y_factor1;
                }
            } else {
                // y is fixed 0
                let sum0 = start_bit + 1;
                let k_bit0 = sum0 & 1;
                let carry0 = (sum0 >> 1) & 1;
                let addr_factor0 = if k_bit0 == 1 { k1 } else { k0 };
                if carry0 == 0 {
                    ndp0 += dp1 * addr_factor0;
                } else {
                    ndp1 += dp1 * addr_factor0;
                }
            }
        }

        dp0 = ndp0;
        dp1 = ndp1;
    }

    // Discard carry-out paths: indices >= 2^ell are out-of-range and contribute 0.
    dp0
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for ProgramImageClaimReductionVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn round_offset(&self, _max_num_rounds: usize) -> usize {
        // Must mirror prover: align to the start of Stage 6b challenge vector.
        0
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let (_, pw_eval) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::ProgramImageInit,
            SumcheckId::ProgramImageClaimReduction,
        );

        // sumcheck_challenges are LSB-first (binding LowToHigh), which is exactly what the DP uses.
        let eq_rw = eval_eq_slice_at_r_star_lsb_dp::<F>(
            &self.params.r_addr_rw,
            self.params.start_index,
            self.params.m,
            sumcheck_challenges,
        );
        let eq_comb = if self.params.single_opening {
            eq_rw
        } else {
            let r_raf = self
                .params
                .r_addr_raf
                .as_ref()
                .expect("missing raf address");
            let eq_raf = eval_eq_slice_at_r_star_lsb_dp::<F>(
                r_raf,
                self.params.start_index,
                self.params.m,
                sumcheck_challenges,
            );
            eq_rw + self.params.gamma * eq_raf
        };

        pw_eval * eq_comb
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::ProgramImageInit,
            SumcheckId::ProgramImageClaimReduction,
            opening_point.r,
        );
    }
}
