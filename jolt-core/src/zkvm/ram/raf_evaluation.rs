use common::jolt_device::MemoryLayout;
use fixedbitset::FixedBitSet;
use num_traits::Zero;

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;
use tracer::instruction::Cycle;

use crate::{
    field::JoltField,
    poly::{
        identity_poly::UnmapRamAddressPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::{config::OneHotParams, ram::remap_address, witness::VirtualPolynomial},
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

pub struct RafEvaluationSumcheckParams<F: JoltField> {
    /// log K (number of rounds)
    pub log_K: usize,
    /// Start address for unmap polynomial
    pub start_address: u64,
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> RafEvaluationSumcheckParams<F> {
    pub fn new(
        memory_layout: &MemoryLayout,
        one_hot_params: &OneHotParams,
        opening_accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let start_address = memory_layout.get_lowest_address();
        let log_K = one_hot_params.ram_k.log_2();
        let (r_cycle, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamAddress,
            SumcheckId::SpartanOuter,
        );
        Self {
            log_K,
            start_address,
            r_cycle,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RafEvaluationSumcheckParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.log_K
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, raf_input_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamAddress,
            SumcheckId::SpartanOuter,
        );
        raf_input_claim
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

/// Sumcheck prover for [`RafEvaluationSumcheckVerifier`].
#[derive(Allocative)]
pub struct RafEvaluationSumcheckProver<F: JoltField> {
    /// The ra polynomial
    ra: MultilinearPolynomial<F>,
    /// The unmap polynomial
    unmap: UnmapRamAddressPolynomial<F>,
    #[allocative(skip)]
    params: RafEvaluationSumcheckParams<F>,
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

        // Build split-eq polynomial over r_cycle
        // This gives us E_out, E_in tables and the current_w (last challenge)
        let split_eq =
            GruenSplitEqPolynomial::<F>::new(&params.r_cycle.r, BindingOrder::LowToHigh);

        let E_in = split_eq.E_in_current();
        let E_out = split_eq.E_out_current();
        let w_current = split_eq.get_current_w();
        let factor_0 = F::one() - w_current;
        let factor_1: F = w_current.into();

        let in_len = E_in.len();
        let x_in_bits = in_len.log_2();

        // Precompute merged inner weights: [E_in[x_in] * (1-w), E_in[x_in] * w]
        let merged_in: Vec<F> = {
            let mut merged: Vec<F> = unsafe_allocate_zero_vec(2 * in_len);
            merged
                .par_chunks_exact_mut(2)
                .zip(E_in.par_iter())
                .for_each(|(chunk, &e)| {
                    chunk[0] = e * factor_0;
                    chunk[1] = e * factor_1;
                });
            merged
        };

        // Divide work evenly among threads (by x_out index)
        let num_threads = rayon::current_num_threads();
        let out_len = E_out.len();
        let chunk_size = out_len.div_ceil(num_threads);

        let chunk_ranges: Vec<(usize, usize)> = (0..num_threads)
            .map(|t| {
                let start = t * chunk_size;
                let end = std::cmp::min(start + chunk_size, out_len);
                (start, end)
            })
            .filter(|(start, end)| start < end)
            .collect();

        // Each thread computes partial ra_evals using split-eq optimization
        let ra_evals: Vec<F> = chunk_ranges
            .into_par_iter()
            .map(|(chunk_start, chunk_end)| {
                let mut partial: Vec<F> = unsafe_allocate_zero_vec(K);

                // Reusable local unreduced accumulator (5-limb) and touched flags
                let mut local_unreduced: Vec<F::Unreduced<5>> = unsafe_allocate_zero_vec(K);
                let mut touched: FixedBitSet = FixedBitSet::with_capacity(K);

                for x_out in chunk_start..chunk_end {
                    let e_out = E_out[x_out];
                    let x_out_base = x_out << (x_in_bits + 1);

                    // Clear touched flags and local accumulators for this x_out
                    for k in touched.ones() {
                        local_unreduced[k] = Default::default();
                    }
                    touched.clear();

                    // Process all x_in for this x_out
                    for x_in in 0..in_len {
                        let j0 = x_out_base + (x_in << 1);
                        let j1 = j0 + 1;
                        let off = 2 * x_in;

                        // Get 4-limb unreduced representations
                        let add0 = *merged_in[off].as_unreduced_ref();
                        let add1 = *merged_in[off + 1].as_unreduced_ref();

                        // Process cycle j0 (last_bit = 0)
                        if j0 < T {
                            if let Some(k) =
                                remap_address(trace[j0].ram_access().address() as u64, memory_layout)
                            {
                                let k = k as usize;
                                if !touched.contains(k) {
                                    touched.insert(k);
                                }
                                local_unreduced[k] += add0;
                            }
                        }

                        // Process cycle j1 (last_bit = 1)
                        if j1 < T {
                            if let Some(k) =
                                remap_address(trace[j1].ram_access().address() as u64, memory_layout)
                            {
                                let k = k as usize;
                                if !touched.contains(k) {
                                    touched.insert(k);
                                }
                                local_unreduced[k] += add1;
                            }
                        }
                    }

                    // Barrett reduce and scale by E_out[x_out], only for touched indices
                    for k in touched.ones() {
                        let reduced = F::from_barrett_reduce::<5>(local_unreduced[k]);
                        partial[k] += e_out * reduced;
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
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
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
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.ra.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.unmap.bind_parallel(r_j, BindingOrder::LowToHigh);
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
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params =
            RafEvaluationSumcheckParams::new(memory_layout, one_hot_params, opening_accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for RafEvaluationSumcheckVerifier<F>
{
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
        unmap_eval * ra_input_claim
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
}
