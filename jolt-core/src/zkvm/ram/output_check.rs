//! # OutputCheck (Stage 2)
//!
//! Source: `jolt-core/src/zkvm/ram/output_check.rs`
//!
//!
//! ## Schwartz–Zippel randomness
//!
//! - `r_address ∈ F^{log₂ K_RAM}`: fresh address challenge vector
//!
//!
//! ## Sumcheck
//!
//! `eq(r_address, X_k)` is the [multilinear Lagrange basis polynomial][ml].
//!
//! [ml]: https://en.wikipedia.org/wiki/Multilinear_polynomial
//!
//! ```text
//! LHS := Σ_{X_k}  eq(r_address, X_k) · io_mask(X_k)
//!                · (RamValFinal(X_k) - ValIO(X_k))
//!
//! RHS := 0   (zero-check)
//!
//! where  X_k ∈ {0,1}^{log₂ K_RAM}
//! ```
//!
//! Dimensions: `log₂ K_RAM` rounds (address only).
//!
//! - `io_mask(X_k)`: 1 if address `X_k` is in the I/O region,
//!   0 otherwise. Verifier-computable.
//! - `ValIO(X_k)`: publicly claimed output value at address `X_k`.
//!   Verifier-computable (public input).
//! - `RamValFinal(X_k)`: final RAM state at address `X_k` after
//!   all cycles (virtual).
//!
//!
//! ## Opening point
//!
//! After sumcheck: `r^(2)_{K_RAM} ∈ F^{log₂ K_RAM}`.
//!
//!
//! ## Verifier opening claim
//!
//! The verifier checks that the final sumcheck message equals:
//!
//! ```text
//! eq(r_address, r^(2)_{K_RAM}) · io_mask(r^(2)_{K_RAM})
//!   · (RamValFinal(r^(2)_{K_RAM}) - ValIO(r^(2)_{K_RAM}))
//! ```
//!
//! `eq`, `io_mask`, and `ValIO` are computable by the verifier.
//! The prover supplies the opening for `RamValFinal`.
//!
//!
//! ## VirtualPolynomials opened at `r^(2)_{K_RAM}`
//!
//! ```text
//! RamValFinal
//! ```
//!
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        range_mask_polynomial::RangeMaskPolynomial,
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::math::Math,
    zkvm::{config::ReadWriteConfig, ram::remap_address, witness::VirtualPolynomial},
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::{constants::RAM_START_ADDRESS, jolt_device::MemoryLayout};
use rayon::prelude::*;
use tracer::JoltDevice;

/// Degree bonud of the sumcheck round polynomials in [`OutputSumcheckVerifier`].
const OUTPUT_SUMCHECK_DEGREE_BOUND: usize = 3;

#[derive(Allocative, Clone)]
pub struct OutputSumcheckParams<F: JoltField> {
    pub K: usize,
    pub log_T: usize,
    pub phase1_num_rounds: usize,
    pub phase2_num_rounds: usize,
    pub r_address: Vec<F::Challenge>,
    pub program_io: JoltDevice,
}

impl<F: JoltField> OutputSumcheckParams<F> {
    pub fn new(
        ram_K: usize,
        program_io: &JoltDevice,
        transcript: &mut impl Transcript,
        trace_len: usize,
        rw_config: &ReadWriteConfig,
    ) -> Self {
        let r_address = transcript.challenge_vector_optimized::<F>(ram_K.log_2());
        Self {
            K: ram_K,
            log_T: trace_len.log_2(),
            phase1_num_rounds: rw_config.ram_rw_phase1_num_rounds as usize,
            phase2_num_rounds: rw_config.ram_rw_phase2_num_rounds as usize,
            r_address,
            program_io: program_io.clone(),
        }
    }

    #[inline]
    fn log_K(&self) -> usize {
        self.K.log_2()
    }

    #[inline]
    fn phase3_cycle_rounds(&self) -> usize {
        self.log_T - self.phase1_num_rounds
    }

    #[inline]
    fn is_internal_cycle_gap_round(&self, round: usize) -> bool {
        let start = self.phase2_num_rounds;
        let end = start + self.phase3_cycle_rounds();
        round >= start && round < end
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for OutputSumcheckParams<F> {
    fn degree(&self) -> usize {
        OUTPUT_SUMCHECK_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.log_T + self.log_K() - self.phase1_num_rounds
    }

    fn input_claim(&self, _: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        // Extract the address challenges, skipping RW Phase 3's internal cycle gap.
        let phase2_addr = self.phase2_num_rounds;
        let gap = self.phase3_cycle_rounds();
        let phase3_addr_start = phase2_addr + gap;
        let addr_challenges = [
            challenges[..phase2_addr].to_vec(),
            challenges[phase3_addr_start..].to_vec(),
        ]
        .concat();
        debug_assert_eq!(addr_challenges.len(), self.log_K());
        OpeningPoint::<LITTLE_ENDIAN, F>::new(addr_challenges).match_endianness()
    }
}

#[derive(Allocative)]
pub struct OutputSumcheckProver<F: JoltField> {
    /// The MLE of the final RAM state
    val_final: MultilinearPolynomial<F>,
    /// Val_io(k) = Val_final(k) if k is in the "IO" region of memory,
    /// and 0 otherwise.
    /// Equivalently, Val_io(k) = Val(k, T) * io_mask(k) for
    /// k \in {0, 1}^log(K)
    val_io: MultilinearPolynomial<F>,
    /// Split-EQ structure over the address variables (Gruen + Dao-Thaler)
    eq_r_address: GruenSplitEqPolynomial<F>,
    /// io_mask(k) serves as a "mask" for the IO region of memory,
    /// i.e. io_mask(k) = 1 if k is in the "IO" region of memory,
    /// and 0 otherwise.
    io_mask: MultilinearPolynomial<F>,
    #[allocative(skip)]
    pub params: OutputSumcheckParams<F>,
}

impl<F: JoltField> OutputSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "OutputSumcheckProver::initialize")]
    pub fn initialize(
        params: OutputSumcheckParams<F>,
        initial_ram_state: &[u64],
        final_ram_state: &[u64],
        memory_layout: &MemoryLayout,
    ) -> Self {
        let K = final_ram_state.len();
        debug_assert_eq!(initial_ram_state.len(), final_ram_state.len());
        debug_assert!(K.is_power_of_two());

        // Compute the witness indices corresponding to the start and end of the IO
        // region of memory
        let io_start = remap_address(memory_layout.input_start, memory_layout).unwrap() as usize;
        let io_end = remap_address(RAM_START_ADDRESS, memory_layout).unwrap() as usize;

        // Compute Val_io by copying the relevant slice of Val_final
        let mut val_io = vec![0; K];
        val_io[io_start..io_end]
            .par_iter_mut()
            .zip(final_ram_state[io_start..io_end].par_iter())
            .for_each(|(dest, src)| *dest = *src);

        // Compute io_mask by setting the relevant coefficients to true
        let mut io_mask = vec![false; K];
        io_mask[io_start..io_end]
            .par_iter_mut()
            .for_each(|k| *k = true);

        let eq_r_address = GruenSplitEqPolynomial::new(&params.r_address, BindingOrder::LowToHigh);

        Self {
            val_final: final_ram_state.to_vec().into(),
            val_io: val_io.into(),
            eq_r_address,
            io_mask: io_mask.into(),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for OutputSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "OutputSumcheckProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if self.params.is_internal_cycle_gap_round(round) {
            let two_inv = F::from_u64(2).inverse().unwrap();
            return UniPoly::from_coeff(vec![previous_claim * two_inv]);
        }

        let Self {
            eq_r_address,
            io_mask,
            val_final,
            val_io,
            ..
        } = self;

        // For s(X) = eq_lin(X) * q(X), where q(X) = io_mask(X) * (val_final(X) - val_io(X))
        // q is quadratic in the current variable. Compute:
        //   c0 = q(0) = io0 * (vf0 - vio0)
        //   e  = coeff of X^2 in q(X) = (io1 - io0) * ((vf1 - vio1) - (vf0 - vio0))
        let [q_constant, q_quadratic] = eq_r_address.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let io0 = io_mask.get_bound_coeff(2 * g);
            let io1 = io_mask.get_bound_coeff(2 * g + 1);
            let vf0 = val_final.get_bound_coeff(2 * g);
            let vf1 = val_final.get_bound_coeff(2 * g + 1);
            let vio0 = val_io.get_bound_coeff(2 * g);
            let vio1 = val_io.get_bound_coeff(2 * g + 1);

            let v0 = vf0 - vio0;
            let v1 = vf1 - vio1;
            let c0 = io0 * v0;
            let e = (io1 - io0) * (v1 - v0);
            [c0, e]
        });

        eq_r_address.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }

    #[tracing::instrument(skip_all, name = "OutputSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if self.params.is_internal_cycle_gap_round(round) {
            return;
        }
        // Bind address variable
        let Self {
            val_final,
            val_io,
            eq_r_address,
            io_mask,
            ..
        } = self;

        val_final.bind_parallel(r_j, BindingOrder::LowToHigh);
        val_io.bind_parallel(r_j, BindingOrder::LowToHigh);
        eq_r_address.bind(r_j);
        io_mask.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn round_offset(&self, max_num_rounds: usize) -> usize {
        let stage2_rounds = self.params.log_T + self.params.log_K();
        let stage2_offset = max_num_rounds - stage2_rounds;
        stage2_offset + self.params.phase1_num_rounds
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let Self { val_final, .. } = self;
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
            opening_point.clone(),
            val_final.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct OutputSumcheckVerifier<F: JoltField> {
    params: OutputSumcheckParams<F>,
}

impl<F: JoltField> OutputSumcheckVerifier<F> {
    pub fn new(
        ram_K: usize,
        program_io: &JoltDevice,
        transcript: &mut impl Transcript,
        trace_len: usize,
        rw_config: &ReadWriteConfig,
    ) -> Self {
        let params = OutputSumcheckParams::new(ram_K, program_io, transcript, trace_len, rw_config);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for OutputSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let val_final_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .1;

        let r_address = &self.params.r_address;
        // Derive r' using the same endianness conversion as used when caching openings
        let r_address_prime = self.params.normalize_opening_point(sumcheck_challenges).r;
        let program_io = &self.params.program_io;

        let io_mask = RangeMaskPolynomial::<F>::new(
            remap_address(
                program_io.memory_layout.input_start,
                &program_io.memory_layout,
            )
            .unwrap() as u128,
            remap_address(RAM_START_ADDRESS, &program_io.memory_layout).unwrap() as u128,
        );
        let eq_eval: F = EqPolynomial::<F>::mle(r_address, &r_address_prime);
        let io_mask_eval = io_mask.evaluate_mle(&r_address_prime);
        let val_io_eval: F = super::eval_io_mle::<F>(program_io, &r_address_prime);

        // Recall that the sumcheck expression is:
        //   0 = \sum_k eq(r_address, k) * io_range(k) * (Val_final(k) - Val_io(k))
        eq_eval * io_mask_eval * (val_final_claim - val_io_eval)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
            opening_point.clone(),
        );
    }

    fn round_offset(&self, max_num_rounds: usize) -> usize {
        let stage2_rounds = self.params.log_T + self.params.log_K();
        let stage2_offset = max_num_rounds - stage2_rounds;
        stage2_offset + self.params.phase1_num_rounds
    }
}
