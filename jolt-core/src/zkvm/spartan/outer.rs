use crate::field::JoltField;
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::spartan_interleaved_poly::SpartanInterleavedPolynomial;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::transcripts::{AppendToTranscript, Transcript};
use crate::utils::errors::ProofVerifyError;
// #[cfg(not(target_arch = "wasm32"))]
// use crate::utils::profiling::print_current_memory_usage;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::utils::thread::drop_in_background_thread;
use crate::zkvm::JoltSharedPreprocessing;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use tracer::instruction::Cycle;
use crate::subprotocols::sumcheck::SumcheckInstanceProof;

impl<F: JoltField, ProofTranscript: Transcript> SumcheckInstanceProof<F, ProofTranscript> {
  #[tracing::instrument(skip_all, name = "Spartan::prove_spartan_outer")]
  pub fn prove_spartan_outer(
      preprocessing: &JoltSharedPreprocessing,
      trace: &[Cycle],
      num_rounds: usize,
      tau: &[F],
      transcript: &mut ProofTranscript,
  ) -> (Self, Vec<F>, [F; 3]) {
      let mut r = Vec::new();
      let mut polys = Vec::new();
      let mut claim = F::zero();

      let (extended_evals, mut az_bz_cz_poly) = SpartanInterleavedPolynomial::<F>::svo_sumcheck_round(preprocessing, trace, tau);
      #[cfg(feature = "allocative")]
      print_data_structure_heap_usage("SpartanInterleavedPolynomial", &az_bz_cz_poly);

      // Process the first sum-check round with univariate skip (of given degree)
      // We have: s_1(Z) = eq(w_z, Z) * t_1(Z), where
      // t_1(Z) = \sum_{x} eq(w_rest, x) * \sum_{y} eq(w_y, y) * (Az(x, y, Z) * Bz(x, y, Z) - Cz(x, y, Z))
      // t_1(Z) has degree 13 + 13 = 26 currently, with 14 evals equal to zero. We have the list
      // of the 13 other evals. So we just need to interpolate the polynomial (i.e. in coefficient format)
      // based on the evals, then derive s_1(Z) via process_eq_sumcheck_round.

      let mut eq_poly = GruenSplitEqPolynomial::new(tau, BindingOrder::LowToHigh);

      // We stream over the trace again for this round
      az_bz_cz_poly.streaming_sumcheck_round(
          preprocessing,
          trace,
          &mut eq_poly,
          transcript,
          &mut r,
          &mut polys,
          &mut claim,
      );

      for _ in 2..num_rounds {
          az_bz_cz_poly.remaining_sumcheck_round(
              &mut eq_poly,
              transcript,
              &mut r,
              &mut polys,
              &mut claim,
          );
      }

      (
          SumcheckInstanceProof::new(polys),
          r,
          az_bz_cz_poly.final_sumcheck_evals(),
      )
  }
}