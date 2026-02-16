use common::jolt_device::MemoryLayout;
use num_traits::Zero;

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;
use tracer::instruction::Cycle;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::UnmapRamAddressPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_claim::{Claim, InputOutputClaims, SumcheckFrontend, VerifierEvaluablePolynomial},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::{
        config::{OneHotParams, ReadWriteConfig},
        ram::remap_address,
        witness::VirtualPolynomial,
    },
};

// RAM RAF evaluation sumcheck
//
// Proves the relation:
//   Σ_{k=0}^{K-1} ra(k) ⋅ unmap(k) = raf_claim,
// where:
// - ra(k) = Σ_j eq(r_cycle, j) ⋅ 1[address(j) = k] aggregates access counts per address k.
// - unmap(k) converts the remapped address k back to its original address.
// - raf_claim is the claimed sum of unmapped addresses over the trace from the Spartan outer sumcheck.

/// Degree bound of the sumcheck round polynomials in [`RafEvaluationSumcheckVerifier`].
const DEGREE_BOUND: usize = 2;

#[derive(Allocative, Clone)]
pub struct RafEvaluationSumcheckParams<F: JoltField> {
    /// log K (number of rounds)
    pub log_K: usize,
    /// log T (cycle variables in Stage 2)
    pub log_T: usize,
    /// RAM RW-check phase 1 rounds (cycle variables)
    pub phase1_num_rounds: usize,
    /// RAM RW-check phase 2 rounds (address variables)
    pub phase2_num_rounds: usize,
    /// Start address for unmap polynomial
    pub start_address: u64,
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> RafEvaluationSumcheckParams<F> {
    pub fn new(
        memory_layout: &MemoryLayout,
        one_hot_params: &OneHotParams,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        trace_len: usize,
        rw_config: &ReadWriteConfig,
    ) -> Self {
        let start_address = memory_layout.get_lowest_address();
        let log_K = one_hot_params.ram_k.log_2();
        let log_T = trace_len.log_2();
        let (r_cycle, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamAddress,
            SumcheckId::SpartanOuter,
        );
        debug_assert_eq!(r_cycle.len(), log_T);
        Self {
            log_K,
            log_T,
            phase1_num_rounds: rw_config.ram_rw_phase1_num_rounds as usize,
            phase2_num_rounds: rw_config.ram_rw_phase2_num_rounds as usize,
            start_address,
            r_cycle,
        }
    }

    #[inline]
    fn phase3_cycle_rounds(&self) -> usize {
        self.log_T - self.phase1_num_rounds
    }

    #[inline]
    fn is_internal_cycle_gap_round(&self, round: usize) -> bool {
        // Local round indexing (after round_offset):
        // [0..phase2_num_rounds) are Phase 2 address rounds
        // [phase2_num_rounds..phase2_num_rounds + phase3_cycle_rounds) are RW Phase 3 cycle rounds
        // remaining rounds are RW Phase 3 address rounds
        let start = self.phase2_num_rounds;
        let end = start + self.phase3_cycle_rounds();
        round >= start && round < end
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RafEvaluationSumcheckParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        // Align to RW-check's address binding schedule: active from RW Phase 2 start through
        // the end of Stage 2, with internal dummy rounds for RW Phase 3's cycle rounds.
        self.log_T + self.log_K - self.phase1_num_rounds
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, raf_input_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamAddress,
            SumcheckId::SpartanOuter,
        );
        // Renormalize to account for internal dummy rounds (cycle gap) that are now part of this
        // instance's active window.
        raf_input_claim.mul_pow_2(self.phase3_cycle_rounds())
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        // Extract the address challenges, skipping the internal dummy cycle-gap rounds.
        let phase2_addr = self.phase2_num_rounds;
        let gap = self.phase3_cycle_rounds();
        let phase3_addr_start = phase2_addr + gap;
        let addr_challenges = [
            challenges[..phase2_addr].to_vec(),
            challenges[phase3_addr_start..].to_vec(),
        ]
        .concat();
        debug_assert_eq!(addr_challenges.len(), self.log_K);
        OpeningPoint::<LITTLE_ENDIAN, F>::new(addr_challenges).match_endianness()
    }
}

#[derive(Allocative)]
pub struct RafEvaluationSumcheckProver<F: JoltField> {
    /// The ra polynomial
    ra: MultilinearPolynomial<F>,
    /// The unmap polynomial
    unmap: UnmapRamAddressPolynomial<F>,
    pub params: RafEvaluationSumcheckParams<F>,
}

impl<F: JoltField> RafEvaluationSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "RamRafEvaluationSumcheckProver::initialize")]
    pub fn initialize(
        params: RafEvaluationSumcheckParams<F>,
        trace: &[Cycle],
        memory_layout: &MemoryLayout,
    ) -> Self {
        let T = trace.len();
        let K = 1 << params.log_K;

        // Two-table split-eq:
        // EqPolynomial::evals uses big-endian bit order: r_cycle[0] is MSB, r_cycle[last] is LSB.
        // To get contiguous blocks in the cycle index, we split off the LSB half (suffix) as E_lo.
        let r_cycle = &params.r_cycle.r;
        let log_T = r_cycle.len();
        let lo_bits = log_T / 2;
        let hi_bits = log_T - lo_bits;
        let (r_hi, r_lo) = r_cycle.split_at(hi_bits);

        let (E_hi, E_lo) = rayon::join(
            || EqPolynomial::<F>::evals(r_hi),
            || EqPolynomial::<F>::evals(r_lo),
        );

        let in_len = E_lo.len(); // 2^lo_bits

        // Split E_hi into chunks for parallel processing
        let num_threads = rayon::current_num_threads();
        let chunk_size = E_hi.len().div_ceil(num_threads);

        // Each thread computes partial ra_evals using split-eq optimization
        let ra_evals: Vec<F> = E_hi
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let mut partial: Vec<F> = unsafe_allocate_zero_vec(K);

                let chunk_start = chunk_idx * chunk_size;
                for (local_idx, &e_hi) in chunk.iter().enumerate() {
                    let c_hi = chunk_start + local_idx;
                    let c_hi_base = c_hi * in_len;

                    for c_lo in 0..in_len {
                        let j = c_hi_base + c_lo;
                        if j >= T {
                            break;
                        }

                        if let Some(k) =
                            remap_address(trace[j].ram_access().address() as u64, memory_layout)
                        {
                            partial[k as usize] += e_hi * E_lo[c_lo];
                        }
                    }
                }

                partial
            })
            .reduce(
                || unsafe_allocate_zero_vec(K),
                |mut running, new| {
                    running
                        .par_iter_mut()
                        .zip(new.par_iter())
                        .for_each(|(x, y)| *x += *y);
                    running
                },
            );

        let ra = MultilinearPolynomial::from(ra_evals);
        let lowest_memory_address = memory_layout.get_lowest_address();
        let unmap = UnmapRamAddressPolynomial::new(K.log_2(), lowest_memory_address);

        Self { ra, unmap, params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RafEvaluationSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "RamRafEvaluationSumcheckProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if self.params.is_internal_cycle_gap_round(round) {
            let two_inv = F::from_u64(2).inverse().unwrap();
            return UniPoly::from_coeff(vec![previous_claim * two_inv]);
        }

        let evals = (0..self.ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = self
                    .ra
                    .sumcheck_evals_array::<DEGREE_BOUND>(i, BindingOrder::LowToHigh);
                let unmap_evals =
                    self.unmap
                        .sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);

                // Compute the product evaluations
                [
                    ra_evals[0].mul_unreduced::<9>(unmap_evals[0]),
                    ra_evals[1].mul_unreduced::<9>(unmap_evals[1]),
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); DEGREE_BOUND],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            )
            .map(F::from_montgomery_reduce);

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    #[tracing::instrument(skip_all, name = "RamRafEvaluationSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if self.params.is_internal_cycle_gap_round(round) {
            return;
        }
        self.ra.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.unmap.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn round_offset(&self, max_num_rounds: usize) -> usize {
        // Align to RW-check's Phase 2 start (global round = phase1_num_rounds), even if the batch
        // has additional leading rounds from other instances.
        let stage2_rounds = self.params.log_T + self.params.log_K;
        let stage2_offset = max_num_rounds - stage2_rounds;
        stage2_offset + self.params.phase1_num_rounds
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_address = self.params.normalize_opening_point(sumcheck_challenges);
        let r_cycle = &self.params.r_cycle;
        let ra_opening_point = OpeningPoint::new([&*r_address.r, &*r_cycle.r].concat());
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamRafEvaluation,
            ra_opening_point,
            self.ra.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct RafEvaluationSumcheckVerifier<F: JoltField> {
    params: RafEvaluationSumcheckParams<F>,
}

impl<F: JoltField> RafEvaluationSumcheckVerifier<F> {
    pub fn new(
        memory_layout: &MemoryLayout,
        one_hot_params: &OneHotParams,
        trace_len: usize,
        rw_config: &ReadWriteConfig,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = RafEvaluationSumcheckParams::new(
            memory_layout,
            one_hot_params,
            opening_accumulator,
            trace_len,
            rw_config,
        );
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for RafEvaluationSumcheckVerifier<F>
{
    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        let result = self.params.input_claim(accumulator);

        #[cfg(test)]
        {
            let reference_result = Self::input_output_claims().input_claim(&[F::one()], accumulator);
            assert_eq!(
                result,
                reference_result.mul_pow_2(self.params.phase3_cycle_rounds()),
                "RafEvaluation input claim should include dummy-round scaling"
            );
        }

        result
    }

    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = self.params.normalize_opening_point(sumcheck_challenges);

        // Compute unmap evaluation at r
        let unmap_eval =
            UnmapRamAddressPolynomial::<F>::new(self.params.log_K, self.params.start_address)
                .evaluate(&r.r);

        let (_, ra_input_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation);

        // Return unmap(r) * ra(r)
        let result = unmap_eval * ra_input_claim;

        #[cfg(test)]
        {
            use crate::subprotocols::sumcheck_claim::VerifierEvaluationParams;

            let eval_params =
                VerifierEvaluationParams::new(self.params.log_K, self.params.start_address);
            let reference_result = Self::input_output_claims()
                .expected_output_claim_with_batching_parameters(
                    &eval_params,
                    &r,
                    &[F::one()],
                    accumulator,
                );

            assert_eq!(result, reference_result);
        }

        result
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_address = self.params.normalize_opening_point(sumcheck_challenges);
        let r_cycle = &self.params.r_cycle;
        let ra_opening_point = OpeningPoint::new([&*r_address.r, &*r_cycle.r].concat());
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamRafEvaluation,
            ra_opening_point,
        );
    }

    fn round_offset(&self, max_num_rounds: usize) -> usize {
        let stage2_rounds = self.params.log_T + self.params.log_K;
        let stage2_offset = max_num_rounds - stage2_rounds;
        stage2_offset + self.params.phase1_num_rounds
    }
}

impl<F: JoltField> SumcheckFrontend<F> for RafEvaluationSumcheckVerifier<F> {
    fn input_output_claims() -> InputOutputClaims<F> {
        let ram_address = VirtualPolynomial::RamAddress.into();
        let ram_ra = VirtualPolynomial::RamRa.into();

        InputOutputClaims {
            claims: vec![Claim {
                input_sumcheck_id: SumcheckId::SpartanOuter,
                input_claim_expr: ram_address,
                batching_poly: VerifierEvaluablePolynomial::UnmapRamAddress,
                expected_output_claim_expr: ram_ra,
            }],
            output_sumcheck_id: SumcheckId::RamRafEvaluation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RafEvaluationSumcheckParams;
    use crate::{
        field::JoltField,
        poly::opening_proof::{OpeningPoint, ProverOpeningAccumulator, SumcheckId, BIG_ENDIAN},
        subprotocols::sumcheck_verifier::SumcheckInstanceParams,
        transcripts::{KeccakTranscript, Transcript},
        zkvm::ram::{
            output_check::OutputSumcheckParams, read_write_checking::RamReadWriteCheckingParams,
        },
        zkvm::witness::VirtualPolynomial,
    };
    use ark_bn254::Fr;
    use common::jolt_device::JoltDevice;

    fn make_challenges(n: usize) -> Vec<<Fr as JoltField>::Challenge> {
        (0..n)
            .map(|i| <Fr as JoltField>::Challenge::from((i as u128) + 1))
            .collect()
    }

    #[test]
    fn stage2_address_round_alignment_matches_rw_schedule() {
        type F = Fr;

        for log_t in [3_usize, 4, 6] {
            for log_k in [2_usize, 3, 5] {
                let t = 1 << log_t;
                let k = 1 << log_k;

                for p1 in 0..=log_t {
                    for p2 in 0..=log_k {
                        let stage2_rounds = log_t + log_k;
                        let global_challenges = make_challenges(stage2_rounds);

                        let rw_params = RamReadWriteCheckingParams::<F> {
                            K: k,
                            T: t,
                            gamma: F::from_u64(1),
                            r_cycle: OpeningPoint::<BIG_ENDIAN, F>::new(vec![
                                <F as JoltField>::Challenge::from(0u128);
                                log_t
                            ]),
                            phase1_num_rounds: p1,
                            phase2_num_rounds: p2,
                        };
                        let rw_opening = rw_params.normalize_opening_point(&global_challenges);
                        let (rw_addr, _) = rw_opening.split_at_r(log_k);

                        // Raf/Output are active starting at global round p1.
                        let local_slice = &global_challenges[p1..];

                        let raf_params = RafEvaluationSumcheckParams::<F> {
                            log_K: log_k,
                            log_T: log_t,
                            phase1_num_rounds: p1,
                            phase2_num_rounds: p2,
                            start_address: 0,
                            r_cycle: OpeningPoint::<BIG_ENDIAN, F>::new(vec![
                                <F as JoltField>::Challenge::from(0u128);
                                log_t
                            ]),
                        };
                        let raf_addr = raf_params.normalize_opening_point(local_slice);
                        assert_eq!(raf_addr.r, rw_addr.to_vec());

                        let out_params = OutputSumcheckParams::<F> {
                            K: k,
                            log_T: log_t,
                            phase1_num_rounds: p1,
                            phase2_num_rounds: p2,
                            r_address: vec![<F as JoltField>::Challenge::from(0u128); log_k],
                            program_io: JoltDevice::default(),
                        };
                        let out_addr = out_params.normalize_opening_point(local_slice);
                        assert_eq!(out_addr.r, rw_addr.to_vec());
                    }
                }
            }
        }
    }

    #[test]
    fn raf_input_claim_is_scaled_by_internal_cycle_gap_rounds() {
        type F = Fr;

        let log_t = 6_usize;
        let log_k = 4_usize;
        let p1 = 2_usize;
        let p2 = 3_usize;

        let mut transcript = KeccakTranscript::new(b"raf_scaling_test");
        let mut acc = ProverOpeningAccumulator::<F>::new(log_t);

        let raf_claim = F::from_u64(7);
        acc.append_virtual(
            &mut transcript,
            VirtualPolynomial::RamAddress,
            SumcheckId::SpartanOuter,
            OpeningPoint::<BIG_ENDIAN, F>::new(vec![]),
            raf_claim,
        );

        let params = RafEvaluationSumcheckParams::<F> {
            log_K: log_k,
            log_T: log_t,
            phase1_num_rounds: p1,
            phase2_num_rounds: p2,
            start_address: 0,
            r_cycle: OpeningPoint::<BIG_ENDIAN, F>::new(vec![
                <F as JoltField>::Challenge::from(0u128);
                log_t
            ]),
        };

        let expected = raf_claim.mul_pow_2(log_t - p1);
        assert_eq!(params.input_claim(&acc), expected);
    }
}
