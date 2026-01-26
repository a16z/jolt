use std::{array, iter::once, sync::Arc};

use num_traits::Zero;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        ra_poly::RaPolynomial,
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        mles_product_sum::eval_linear_prod_assign,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{
        errors::ProofVerifyError, math::Math, small_scalar::SmallScalar,
        thread::unsafe_allocate_zero_vec,
    },
    zkvm::{
        config::{OneHotParams, ProgramMode},
        instruction::{
            CircuitFlags, Flags, InstructionFlags, InstructionLookup, InterleavedBitsMarker,
            NUM_CIRCUIT_FLAGS,
        },
        lookup_table::{LookupTables, NUM_LOOKUP_TABLES},
        program::ProgramPreprocessing,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::constants::{REGISTER_COUNT, XLEN};
use itertools::{zip_eq, Itertools};
use rayon::prelude::*;
use strum::{EnumCount, IntoEnumIterator};
use tracer::instruction::{Cycle, Instruction};

/// Number of batched read-checking sumchecks bespokely
const N_STAGES: usize = 5;

/// Bytecode instruction: multi-stage Read + RAF sumcheck (N_STAGES = 5).
///
/// Stages virtualize different claim families (Stage1: Spartan outer; Stage2: product-virtualized
/// flags; Stage3: Shift; Stage4: Registers RW; Stage5: Registers val-eval + Instruction lookups).
///
/// The input claim is a γ-weighted RLC of stage rv_claims plus RAF contributions folded into
/// stages 1 and 3 via the identity polynomial. Address vars are bound in `d` chunks; cycle vars
/// are bound with per-stage `GruenSplitEqPolynomial` (low-to-high binding), producing univariates
/// of degree `d + 1` (cubic only when `d = 2`).
///
/// Challenge notation:
/// - γ: the stage-folding scalar with powers `params.gamma_powers = transcript.challenge_scalar_powers(7)`.
/// - β_s: per-stage scalars used *within* Val_s encodings (`stage{s}_gammas = transcript.challenge_scalar_powers(...)`),
///   sampled separately for each stage.
///
/// Mathematical claim:
/// - Let K = 2^{log_K} and T = 2^{log_T}.
/// - For stage s ∈ {1,2,3,4,5}, let r_s ∈ F^{log_T} and define eq_s(j) = EqPolynomial(j; r_s).
/// - Let r_addr ∈ F^{log_K}. Let ra(k, j) ∈ {0,1} be the indicator that cycle j maps to bytecode
///   row index k (i.e. `k = get_pc(cycle_j)`; this is *not* the ELF/instruction address).
///   Implemented as ∏_{i=0}^{d-1} ra_i(k_i, j) via one-hot chunking of the bytecode index k.
/// - Int(k) = 1 for all k (evaluation of the IdentityPolynomial over address variables).
/// - Define per-stage Val_s(k) (address-only) as implemented by `compute_val_*`:
///   * Stage1: Val_1(k) = unexpanded_pc(k) + β_1·imm(k) + Σ_t β_1^{2+t}·circuit_flag_t(k).
///   * Stage2: Val_2(k) = 1_{jump}(k) + β_2·1_{branch}(k) + β_2^2·rd_addr(k) + β_2^3·1_{write_lookup_to_rd}(k).
///   * Stage3: Val_3(k) = imm(k) + β_3·unexpanded_pc(k) + β_3^2·1_{L_is_rs1}(k) + β_3^3·1_{L_is_pc}(k)
///   + β_3^4·1_{R_is_rs2}(k) + β_3^5·1_{R_is_imm}(k) + β_3^6·1_{IsNoop}(k)
///   + β_3^7·1_{VirtualInstruction}(k) + β_3^8·1_{IsFirstInSequence}(k).
///   * Stage4: Val_4(k) = 1_{rd=r}(k) + β_4·1_{rs1=r}(k) + β_4^2·1_{rs2=r}(k), where r is fixed by opening.
///   * Stage5: Val_5(k) = 1_{rd=r}(k) + β_5·1_{¬interleaved}(k) + Σ_i β_5^{2+i}·1_{table=i}(k).
///
///   Here, unexpanded_pc(k) is the instruction's ELF/address field (`instr.address`) stored in the bytecode row k.
///
/// Accumulator-provided LHS (RLC of stage claims with RAF):
///   rv_1(r_1) + γ·rv_2(r_2) + γ^2·rv_3(r_3) + γ^3·rv_4(r_4) + γ^4·rv_5(r_5)
///   + γ^5·raf_1(r_1) + γ^6·raf_3(r_3).
///
/// Sumcheck RHS proved (double sum over cycles and addresses):
///   Σ_{j=0}^{T-1} Σ_{k=0}^{K-1} ra(k, j) · [
///       γ^0·eq_1(j)·Val_1(k) + γ^1·eq_2(j)·Val_2(k) + γ^2·eq_3(j)·Val_3(k)
///     + γ^3·eq_4(j)·Val_4(k) + γ^4·eq_5(j)·Val_5(k)
///     + γ^5·eq_1(j)·Int(k)   + γ^6·eq_3(j)·Int(k)
///   ].
///
/// Thus the identity established by this sumcheck is:
///   rv_1(r_1) + γ·rv_2(r_2) + γ^2·rv_3(r_3) + γ^3·rv_4(r_4) + γ^4·rv_5(r_5)
///   + γ^5·raf_1(r_1) + γ^6·raf_3(r_3)
///     = Σ_{j,k} ra(k, j) · [ Σ_{s=1}^{5} γ^{s-1}·eq_s(j)·Val_s(k) + γ^5·eq_1(j)·Int(k) + γ^6·eq_3(j)·Int(k) ].
///
/// Binding/implementation notes:
/// - Address variables are bound first (low→high in the sumcheck binding order) in `d` chunks,
///   accumulating `F_i` and `v` tables;
///   this materializes the address-only Val_s(k) evaluations and sets up `ra_i` polynomials.
/// - Cycle variables are then bound (low→high) per stage with `GruenSplitEqPolynomial`, using
///   previous-round claims to recover the degree-(d+1) univariate each round.
/// - RAF injection uses `VirtualPolynomial::PC` (not `UnexpandedPC`): `raf_claim` comes from
///   `SumcheckId::SpartanOuter` and `raf_shift_claim` from `SumcheckId::SpartanShift`.
/// - The Stage3 RAF weight is “offset inside the stage”: the prover uses `γ^4 * raf_shift_claim`
///   in the Stage3 per-stage claim, then the stage itself is folded with an outer factor `γ^2`,
///   yielding the advertised `γ^6` overall.
#[derive(Allocative)]
pub struct BytecodeReadRafSumcheckProver<F: JoltField> {
    /// Per-stage address MLEs F_i(k) built from eq(r_cycle_stage_i, (chunk_index, j)),
    /// bound low-to-high during the address-binding phase.
    F: [MultilinearPolynomial<F>; N_STAGES],
    /// Chunked RA polynomials over address variables (one per dimension `d`), used to form
    /// the product ∏_i ra_i during the cycle-binding phase.
    ra: Vec<RaPolynomial<u8, F>>,
    /// Binding challenges for the first log_K variables of the sumcheck
    r_address_prime: Vec<F::Challenge>,
    /// Per-stage Gruen-split eq polynomials over cycle vars (low-to-high binding order).
    gruen_eq_polys: [GruenSplitEqPolynomial<F>; N_STAGES],
    /// Previous-round claims s_i(0)+s_i(1) per stage, needed for degree-(d+1) univariate recovery.
    prev_round_claims: [F; N_STAGES],
    /// Round polynomials per stage for advancing to the next claim at r_j.
    prev_round_polys: Option<[UniPoly<F>; N_STAGES]>,
    /// Final sumcheck claims of stage Val polynomials (with RAF Int folded where applicable).
    bound_val_evals: Option<[F; N_STAGES]>,
    /// Trace for computing PCs on the fly in init_log_t_rounds.
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    /// Bytecode preprocessing for computing PCs.
    #[allocative(skip)]
    program: Arc<ProgramPreprocessing>,
    pub params: BytecodeReadRafSumcheckParams<F>,
}

impl<F: JoltField> BytecodeReadRafSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheckProver::initialize")]
    pub fn initialize(
        params: BytecodeReadRafSumcheckParams<F>,
        trace: Arc<Vec<Cycle>>,
        program: Arc<ProgramPreprocessing>,
    ) -> Self {
        let claim_per_stage = [
            params.rv_claims[0] + params.gamma_powers[5] * params.raf_claim,
            params.rv_claims[1],
            params.rv_claims[2] + params.gamma_powers[4] * params.raf_shift_claim,
            params.rv_claims[3],
            params.rv_claims[4],
        ];

        // Two-table split-eq optimization for computing F[stage][k] = Σ_{c: PC(c)=k} eq(r_cycle, c).
        //
        // Double summation pattern:
        //   F[stage][k] = Σ_{c_hi} E_hi[c_hi] × ( Σ_{c_lo : PC(c)=k} E_lo[c_lo] )
        //
        // Inner sum (over c_lo): ADDITIONS ONLY - accumulate E_lo contributions by PC
        // Outer sum (over c_hi): ONE multiplication per touched PC, not per cycle
        //
        // This reduces multiplications from O(T × N_STAGES) to O(touched_PCs × out_len × N_STAGES)
        let T = trace.len();
        let K = params.K;
        let log_T = params.log_T;

        // Optimal split: sqrt(T) for balanced tables
        let lo_bits = log_T / 2;
        let hi_bits = log_T - lo_bits;
        let in_len: usize = 1 << lo_bits; // E_lo size (inner loop)
        let out_len: usize = 1 << hi_bits; // E_hi size (outer loop)

        // Pre-compute E_hi[stage][c_hi] and E_lo[stage][c_lo] for all stages in parallel
        let (E_hi, E_lo): ([Vec<F>; N_STAGES], [Vec<F>; N_STAGES]) = rayon::join(
            || {
                params
                    .r_cycles
                    .each_ref()
                    .map(|r_cycle| EqPolynomial::evals(&r_cycle[..hi_bits]))
            },
            || {
                params
                    .r_cycles
                    .each_ref()
                    .map(|r_cycle| EqPolynomial::evals(&r_cycle[hi_bits..]))
            },
        );

        // Process by c_hi blocks, distributing work evenly among threads
        let num_threads = rayon::current_num_threads();
        let chunk_size = out_len.div_ceil(num_threads);

        // Double summation: outer sum over c_hi, inner sum over c_lo
        let F: [Vec<F>; N_STAGES] = E_hi[0]
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                // Per-thread accumulators for final F
                let mut partial: [Vec<F>; N_STAGES] =
                    array::from_fn(|_| unsafe_allocate_zero_vec(K));

                // Per-c_hi inner accumulators (reused across c_hi iterations)
                let mut inner: [Vec<F>; N_STAGES] = array::from_fn(|_| unsafe_allocate_zero_vec(K));

                // Track which PCs were touched in this c_hi block
                let mut touched = Vec::with_capacity(in_len);

                let chunk_start = chunk_idx * chunk_size;
                for (local_idx, _) in chunk.iter().enumerate() {
                    let c_hi = chunk_start + local_idx;
                    let c_hi_base = c_hi * in_len;

                    // Clear inner accumulators for touched PCs only
                    for &k in &touched {
                        for stage in 0..N_STAGES {
                            inner[stage][k] = F::zero();
                        }
                    }
                    touched.clear();

                    // INNER SUM: accumulate E_lo by PC (ADDITIONS ONLY, no multiplications)
                    for c_lo in 0..in_len {
                        let c = c_hi_base + c_lo;
                        if c >= T {
                            break;
                        }

                        let pc = program.get_pc(&trace[c]);

                        // Track touched PCs (avoid duplicates with a simple check)
                        if inner[0][pc].is_zero() {
                            touched.push(pc);
                        }

                        // Accumulate E_lo contributions (addition only!)
                        for stage in 0..N_STAGES {
                            inner[stage][pc] += E_lo[stage][c_lo];
                        }
                    }

                    // OUTER SUM: multiply by E_hi and add to partial (sparse)
                    for &k in &touched {
                        for stage in 0..N_STAGES {
                            partial[stage][k] += E_hi[stage][c_hi] * inner[stage][k];
                        }
                    }
                }

                partial
            })
            .reduce(
                || array::from_fn(|_| unsafe_allocate_zero_vec(K)),
                |mut a, b| {
                    for stage in 0..N_STAGES {
                        a[stage]
                            .par_iter_mut()
                            .zip(b[stage].par_iter())
                            .for_each(|(a, b)| *a += *b);
                    }
                    a
                },
            );

        #[cfg(test)]
        {
            // Verify that for each stage i: sum(val_i[k] * F_i[k] * eq_i[k]) = rv_claim_i
            for i in 0..N_STAGES {
                let computed_claim: F = (0..params.K)
                    .into_par_iter()
                    .map(|k| {
                        let val_k = params.val_polys[i].get_bound_coeff(k);
                        let F_k = F[i][k];
                        val_k * F_k
                    })
                    .sum();
                assert_eq!(
                    computed_claim,
                    params.rv_claims[i],
                    "Stage {} mismatch: computed {} != expected {}",
                    i + 1,
                    computed_claim,
                    params.rv_claims[i]
                );
            }
        }

        let F = F.map(MultilinearPolynomial::from);

        let gruen_eq_polys = params
            .r_cycles
            .each_ref()
            .map(|r_cycle| GruenSplitEqPolynomial::new(r_cycle, BindingOrder::LowToHigh));

        Self {
            F,
            ra: Vec::with_capacity(params.d),
            r_address_prime: Vec::with_capacity(params.log_K),
            gruen_eq_polys,
            prev_round_claims: claim_per_stage,
            prev_round_polys: None,
            bound_val_evals: None,
            trace,
            program,
            params,
        }
    }

    fn init_log_t_rounds(&mut self) {
        let int_poly = self.params.int_poly.final_sumcheck_claim();

        // We have a separate Val polynomial for each stage
        // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
        // So we would have:
        // Stage 1: gamma^0 * (Val_1 + gamma^5 * Int)
        // Stage 2: gamma^1 * (Val_2)
        // Stage 3: gamma^2 * (Val_3 + gamma^4 * Int)
        // Stage 4: gamma^3 * (Val_4)
        // Stage 5: gamma^4 * (Val_5)
        // Which matches with the input claim:
        // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * rv_4 + gamma^4 * rv_5 + gamma^5 * raf_1 + gamma^6 * raf_3
        self.bound_val_evals = Some(
            self.params
                .val_polys
                .iter()
                .zip([
                    int_poly * self.params.gamma_powers[5],
                    F::zero(),
                    int_poly * self.params.gamma_powers[4],
                    F::zero(),
                    F::zero(),
                ])
                .map(|(poly, int_poly)| poly.final_sumcheck_claim() + int_poly)
                .collect::<Vec<F>>()
                .try_into()
                .unwrap(),
        );

        // Reverse r_address_prime to get the correct order (it was built low-to-high)
        let mut r_address = std::mem::take(&mut self.r_address_prime);
        r_address.reverse();

        // Drop log_K phase data that's no longer needed (val_polys reduced to bound_val_evals)
        // F polynomials are fully bound and can be dropped
        self.F = array::from_fn(|_| MultilinearPolynomial::default());
        // val_polys are reduced to scalars in bound_val_evals
        self.params.val_polys = array::from_fn(|_| MultilinearPolynomial::default());
        // int_poly is reduced to a scalar
        self.params.int_poly = IdentityPolynomial::new(0);

        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(&r_address);

        // Build RA polynomials by iterating over trace and computing PCs on the fly
        self.ra = r_address_chunks
            .iter()
            .enumerate()
            .map(|(i, r_address_chunk)| {
                let ra_i: Vec<Option<u8>> = self
                    .trace
                    .par_iter()
                    .map(|cycle| {
                        let pc = self.program.get_pc(cycle);
                        Some(self.params.one_hot_params.bytecode_pc_chunk(pc, i))
                    })
                    .collect();
                RaPolynomial::new(Arc::new(ra_i), EqPolynomial::evals(r_address_chunk))
            })
            .collect();

        // Drop trace and preprocessing - no longer needed after this
        self.trace = Arc::new(Vec::new());
    }

    fn compute_message_internal(&mut self, round: usize, _previous_claim: F) -> UniPoly<F> {
        if round < self.params.log_K {
            const DEGREE: usize = 2;

            // Evaluation at [0, 2] for each stage.
            let eval_per_stage: [[F; DEGREE]; N_STAGES] = (0..self.params.val_polys[0].len() / 2)
                .into_par_iter()
                .map(|i| {
                    let ra_evals = self.F.each_ref().map(|poly| {
                        poly.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh)
                    });

                    let int_evals =
                        self.params
                            .int_poly
                            .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                    // We have a separate Val polynomial for each stage
                    // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
                    // So we would have:
                    // Stage 1: Val_1 + gamma^5 * Int
                    // Stage 2: Val_2
                    // Stage 3: Val_3 + gamma^4 * Int
                    // Stage 4: Val_4
                    // Stage 5: Val_5
                    // Which matches with the input claim:
                    // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * rv_4 + gamma^4 * rv_5 + gamma^5 * raf_1 + gamma^6 * raf_3
                    let mut val_evals = self
                        .params
                        .val_polys
                        .iter()
                        // Val polynomials
                        .map(|val| val.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh))
                        // Here are the RAF polynomials and their powers
                        .zip([Some(&int_evals), None, Some(&int_evals), None, None])
                        .zip([
                            Some(self.params.gamma_powers[5]),
                            None,
                            Some(self.params.gamma_powers[4]),
                            None,
                            None,
                        ])
                        .map(|((val_evals, int_evals), gamma)| {
                            std::array::from_fn::<F, DEGREE, _>(|j| {
                                val_evals[j]
                                    + int_evals.map_or(F::zero(), |int_evals| {
                                        int_evals[j] * gamma.unwrap()
                                    })
                            })
                        });

                    array::from_fn(|stage| {
                        let [ra_at_0, ra_at_2] = ra_evals[stage];
                        let [val_at_0, val_at_2] = val_evals.next().unwrap();
                        [ra_at_0 * val_at_0, ra_at_2 * val_at_2]
                    })
                })
                .reduce(
                    || [[F::zero(); DEGREE]; N_STAGES],
                    |a, b| array::from_fn(|i| array::from_fn(|j| a[i][j] + b[i][j])),
                );

            let mut round_polys: [_; N_STAGES] = array::from_fn(|_| UniPoly::zero());
            let mut agg_round_poly = UniPoly::zero();

            for (stage, evals) in eval_per_stage.into_iter().enumerate() {
                let [eval_at_0, eval_at_2] = evals;
                let eval_at_1 = self.prev_round_claims[stage] - eval_at_0;
                let round_poly = UniPoly::from_evals(&[eval_at_0, eval_at_1, eval_at_2]);
                agg_round_poly += &(&round_poly * self.params.gamma_powers[stage]);
                round_polys[stage] = round_poly;
            }

            self.prev_round_polys = Some(round_polys);

            agg_round_poly
        } else {
            let degree = self.params.degree();

            let out_len = self.gruen_eq_polys[0].E_out_current().len();
            let in_len = self.gruen_eq_polys[0].E_in_current().len();
            let in_n_vars = in_len.log_2();

            // Evaluations on [1, ..., degree - 2, inf] (for each stage).
            let mut evals_per_stage: [Vec<F>; N_STAGES] = (0..out_len)
                .into_par_iter()
                .map(|j_hi| {
                    let mut ra_eval_pairs = vec![(F::zero(), F::zero()); self.ra.len()];
                    let mut ra_prod_evals = vec![F::zero(); degree - 1];
                    let mut evals_per_stage: [_; N_STAGES] =
                        array::from_fn(|_| vec![F::Unreduced::zero(); degree - 1]);

                    for j_lo in 0..in_len {
                        let j = j_lo + (j_hi << in_n_vars);

                        for (i, ra_i) in self.ra.iter().enumerate() {
                            let ra_i_eval_at_j_0 = ra_i.get_bound_coeff(j * 2);
                            let ra_i_eval_at_j_1 = ra_i.get_bound_coeff(j * 2 + 1);
                            ra_eval_pairs[i] = (ra_i_eval_at_j_0, ra_i_eval_at_j_1);
                        }
                        // Eval prod_i ra_i(x).
                        eval_linear_prod_assign(&ra_eval_pairs, &mut ra_prod_evals);

                        for stage in 0..N_STAGES {
                            let eq_in_eval = self.gruen_eq_polys[stage].E_in_current()[j_lo];
                            for i in 0..degree - 1 {
                                evals_per_stage[stage][i] +=
                                    eq_in_eval.mul_unreduced::<9>(ra_prod_evals[i]);
                            }
                        }
                    }

                    array::from_fn(|stage| {
                        let eq_out_eval = self.gruen_eq_polys[stage].E_out_current()[j_hi];
                        evals_per_stage[stage]
                            .iter()
                            .map(|v| eq_out_eval * F::from_montgomery_reduce(*v))
                            .collect()
                    })
                })
                .reduce(
                    || array::from_fn(|_| vec![F::zero(); degree - 1]),
                    |a, b| array::from_fn(|i| zip_eq(&a[i], &b[i]).map(|(a, b)| *a + *b).collect()),
                );
            // Multiply by bound values.
            let bound_val_evals = self.bound_val_evals.as_ref().unwrap();
            for (stage, evals) in evals_per_stage.iter_mut().enumerate() {
                evals.iter_mut().for_each(|v| *v *= bound_val_evals[stage]);
            }

            let mut round_polys: [_; N_STAGES] = array::from_fn(|_| UniPoly::zero());
            let mut agg_round_poly = UniPoly::zero();

            // Obtain round poly for each stage and perform RLC.
            for (stage, evals) in evals_per_stage.iter().enumerate() {
                let claim = self.prev_round_claims[stage];
                let round_poly = self.gruen_eq_polys[stage].gruen_poly_from_evals(evals, claim);
                agg_round_poly += &(&round_poly * self.params.gamma_powers[stage]);
                round_polys[stage] = round_poly;
            }

            self.prev_round_polys = Some(round_polys);

            agg_round_poly
        }
    }

    fn ingest_challenge_internal(&mut self, r_j: F::Challenge, round: usize) {
        if let Some(prev_round_polys) = self.prev_round_polys.take() {
            self.prev_round_claims = prev_round_polys.map(|poly| poly.evaluate(&r_j));
        }

        if round < self.params.log_K {
            self.params
                .val_polys
                .iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
            self.params
                .int_poly
                .bind_parallel(r_j, BindingOrder::LowToHigh);
            self.F
                .iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
            self.r_address_prime.push(r_j);
            if round == self.params.log_K - 1 {
                self.init_log_t_rounds();
            }
        } else {
            self.ra
                .iter_mut()
                .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh));
            self.gruen_eq_polys
                .iter_mut()
                .for_each(|poly| poly.bind(r_j));
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for BytecodeReadRafSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheckProver::compute_message")]
    fn compute_message(&mut self, round: usize, _previous_claim: F) -> UniPoly<F> {
        self.compute_message_internal(round, _previous_claim)
    }

    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.ingest_challenge_internal(r_j, round)
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let (r_address, r_cycle) = opening_point.split_at(self.params.log_K);

        // Compute r_address_chunks with proper padding
        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(&r_address.r);

        for i in 0..self.params.d {
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::BytecodeRa(i)],
                SumcheckId::BytecodeReadRaf,
                r_address_chunks[i].clone(),
                r_cycle.clone().into(),
                vec![self.ra[i].final_sumcheck_claim()],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Bytecode Read+RAF Address-Phase Sumcheck Prover.
///
/// This prover handles only the first `log_K` rounds (address variables).
/// The cycle-phase prover is constructed separately from witness + accumulator (Option B).
#[derive(Allocative)]
pub struct BytecodeReadRafAddressSumcheckProver<F: JoltField> {
    /// Per-stage address MLEs F_i(k) built from eq(r_cycle_stage_i, (chunk_index, j)).
    F: [MultilinearPolynomial<F>; N_STAGES],
    /// Binding challenges for the first log_K variables.
    r_address_prime: Vec<F::Challenge>,
    /// Previous-round claims s_i(0)+s_i(1) per stage.
    prev_round_claims: [F; N_STAGES],
    /// Round polynomials per stage for advancing to the next claim.
    prev_round_polys: Option<[UniPoly<F>; N_STAGES]>,
    /// Parameters (shared with cycle prover).
    pub params: BytecodeReadRafSumcheckParams<F>,
}

impl<F: JoltField> BytecodeReadRafAddressSumcheckProver<F> {
    /// Initialize a BytecodeReadRafAddressSumcheckProver.
    #[tracing::instrument(skip_all, name = "BytecodeReadRafAddressSumcheckProver::initialize")]
    pub fn initialize(
        params: BytecodeReadRafSumcheckParams<F>,
        trace: Arc<Vec<Cycle>>,
        program: Arc<ProgramPreprocessing>,
    ) -> Self {
        let claim_per_stage = [
            params.rv_claims[0] + params.gamma_powers[5] * params.raf_claim,
            params.rv_claims[1],
            params.rv_claims[2] + params.gamma_powers[4] * params.raf_shift_claim,
            params.rv_claims[3],
            params.rv_claims[4],
        ];

        // Two-table split-eq optimization for computing F[stage][k] = Σ_{c: PC(c)=k} eq(r_cycle, c).
        let T = trace.len();
        let K = params.K;
        let log_T = params.log_T;

        let lo_bits = log_T / 2;
        let hi_bits = log_T - lo_bits;
        let in_len: usize = 1 << lo_bits;
        let out_len: usize = 1 << hi_bits;

        let (E_hi, E_lo): ([Vec<F>; N_STAGES], [Vec<F>; N_STAGES]) = rayon::join(
            || {
                params
                    .r_cycles
                    .each_ref()
                    .map(|r_cycle| EqPolynomial::evals(&r_cycle[..hi_bits]))
            },
            || {
                params
                    .r_cycles
                    .each_ref()
                    .map(|r_cycle| EqPolynomial::evals(&r_cycle[hi_bits..]))
            },
        );

        let num_threads = rayon::current_num_threads();
        let chunk_size = out_len.div_ceil(num_threads);

        let F_polys: [Vec<F>; N_STAGES] = E_hi[0]
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let mut partial: [Vec<F>; N_STAGES] =
                    array::from_fn(|_| unsafe_allocate_zero_vec(K));
                let mut inner: [Vec<F>; N_STAGES] = array::from_fn(|_| unsafe_allocate_zero_vec(K));
                let mut touched = Vec::with_capacity(in_len);

                let chunk_start = chunk_idx * chunk_size;
                for (local_idx, _) in chunk.iter().enumerate() {
                    let c_hi = chunk_start + local_idx;
                    let c_hi_base = c_hi * in_len;

                    for &k in &touched {
                        for stage in 0..N_STAGES {
                            inner[stage][k] = F::zero();
                        }
                    }
                    touched.clear();

                    for c_lo in 0..in_len {
                        let c = c_hi_base + c_lo;
                        if c >= T {
                            break;
                        }

                        let pc = program.get_pc(&trace[c]);
                        if inner[0][pc].is_zero() {
                            touched.push(pc);
                        }
                        for stage in 0..N_STAGES {
                            inner[stage][pc] += E_lo[stage][c_lo];
                        }
                    }

                    for &k in &touched {
                        for stage in 0..N_STAGES {
                            partial[stage][k] += E_hi[stage][c_hi] * inner[stage][k];
                        }
                    }
                }
                partial
            })
            .reduce(
                || array::from_fn(|_| unsafe_allocate_zero_vec(K)),
                |mut a, b| {
                    for stage in 0..N_STAGES {
                        a[stage]
                            .par_iter_mut()
                            .zip(b[stage].par_iter())
                            .for_each(|(a, b)| *a += *b);
                    }
                    a
                },
            );

        let F = F_polys.map(MultilinearPolynomial::from);

        Self {
            F,
            r_address_prime: Vec::with_capacity(params.log_K),
            prev_round_claims: claim_per_stage,
            prev_round_polys: None,
            params,
        }
    }

    fn compute_message_impl(&mut self, _previous_claim: F) -> UniPoly<F> {
        const DEGREE: usize = 2;

        let eval_per_stage: [[F; DEGREE]; N_STAGES] = (0..self.params.val_polys[0].len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = self
                    .F
                    .each_ref()
                    .map(|poly| poly.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh));

                let int_evals =
                    self.params
                        .int_poly
                        .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                let mut val_evals = self
                    .params
                    .val_polys
                    .iter()
                    .map(|val| val.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh))
                    .zip([Some(&int_evals), None, Some(&int_evals), None, None])
                    .zip([
                        Some(self.params.gamma_powers[5]),
                        None,
                        Some(self.params.gamma_powers[4]),
                        None,
                        None,
                    ])
                    .map(|((val_evals, int_evals), gamma)| {
                        std::array::from_fn::<F, DEGREE, _>(|j| {
                            val_evals[j]
                                + int_evals
                                    .map_or(F::zero(), |int_evals| int_evals[j] * gamma.unwrap())
                        })
                    });

                array::from_fn(|stage| {
                    let [ra_at_0, ra_at_2] = ra_evals[stage];
                    let [val_at_0, val_at_2] = val_evals.next().unwrap();
                    [ra_at_0 * val_at_0, ra_at_2 * val_at_2]
                })
            })
            .reduce(
                || [[F::zero(); DEGREE]; N_STAGES],
                |a, b| array::from_fn(|i| array::from_fn(|j| a[i][j] + b[i][j])),
            );

        let mut round_polys: [_; N_STAGES] = array::from_fn(|_| UniPoly::zero());
        let mut agg_round_poly = UniPoly::zero();

        for (stage, evals) in eval_per_stage.into_iter().enumerate() {
            let [eval_at_0, eval_at_2] = evals;
            let eval_at_1 = self.prev_round_claims[stage] - eval_at_0;
            let round_poly = UniPoly::from_evals(&[eval_at_0, eval_at_1, eval_at_2]);
            agg_round_poly += &(&round_poly * self.params.gamma_powers[stage]);
            round_polys[stage] = round_poly;
        }

        self.prev_round_polys = Some(round_polys);
        agg_round_poly
    }

    fn ingest_challenge_impl(&mut self, r_j: F::Challenge) {
        if let Some(prev_round_polys) = self.prev_round_polys.take() {
            self.prev_round_claims = prev_round_polys.map(|poly| poly.evaluate(&r_j));
        }

        self.params
            .val_polys
            .iter_mut()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        self.params
            .int_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.F
            .iter_mut()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        self.r_address_prime.push(r_j);
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for BytecodeReadRafAddressSumcheckProver<F>
{
    fn degree(&self) -> usize {
        self.params.degree()
    }

    fn num_rounds(&self) -> usize {
        self.params.log_K
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim(_accumulator)
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        self.compute_message_impl(previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.ingest_challenge_impl(r_j)
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let mut r_address = sumcheck_challenges.to_vec();
        r_address.reverse();
        let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(r_address);
        let address_claim: F = self
            .prev_round_claims
            .iter()
            .zip(self.params.gamma_powers.iter())
            .take(N_STAGES)
            .map(|(claim, gamma)| *claim * *gamma)
            .sum();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::BytecodeReadRafAddrClaim,
            SumcheckId::BytecodeReadRafAddressPhase,
            opening_point.clone(),
            address_claim,
        );

        // Emit Val-only claims at the Stage 6a boundary only when the staged-Val/claim-reduction
        // path is enabled.
        if self.params.use_staged_val_claims {
            for stage in 0..N_STAGES {
                let claim = self.params.val_polys[stage].final_sumcheck_claim();
                accumulator.append_virtual(
                    transcript,
                    VirtualPolynomial::BytecodeValStage(stage),
                    SumcheckId::BytecodeReadRafAddressPhase,
                    opening_point.clone(),
                    claim,
                );
            }
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Bytecode Read+RAF Cycle-Phase Sumcheck Prover.
///
/// This prover handles the remaining `log_T` rounds (cycle variables).
/// It is constructed from scratch via [`BytecodeReadRafCycleSumcheckProver::initialize`].
#[derive(Allocative)]
pub struct BytecodeReadRafCycleSumcheckProver<F: JoltField> {
    /// Chunked RA polynomials over address variables.
    ra: Vec<RaPolynomial<u8, F>>,
    /// Per-stage Gruen-split eq polynomials over cycle vars.
    gruen_eq_polys: [GruenSplitEqPolynomial<F>; N_STAGES],
    /// Final sumcheck claims of stage Val polynomials (with RAF Int folded).
    bound_val_evals: [F; N_STAGES],
    /// Parameters.
    pub params: BytecodeReadRafSumcheckParams<F>,
}

impl<F: JoltField> BytecodeReadRafCycleSumcheckProver<F> {
    /// Initialize the cycle-phase prover from Stage 6a openings (no replay).
    #[tracing::instrument(skip_all, name = "BytecodeReadRafCycleSumcheckProver::initialize")]
    pub fn initialize(
        params: BytecodeReadRafSumcheckParams<F>,
        trace: Arc<Vec<Cycle>>,
        program: Arc<ProgramPreprocessing>,
        accumulator: &ProverOpeningAccumulator<F>,
    ) -> Self {
        // Recover Stage 6a address challenges from the accumulator.
        // Address-phase cache_openings stored them as BIG_ENDIAN (MSB-first).
        let (r_address_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::BytecodeReadRafAddrClaim,
            SumcheckId::BytecodeReadRafAddressPhase,
        );

        // Compute bound_val_evals at r_address (Val + RAF Int folds).
        let int_eval = params.int_poly.evaluate(&r_address_point.r);
        let int_terms = [
            int_eval * params.gamma_powers[5], // RAF for Stage1
            F::zero(),                         // No RAF for Stage2
            int_eval * params.gamma_powers[4], // RAF for Stage3
            F::zero(),                         // No RAF for Stage4
            F::zero(),                         // No RAF for Stage5
        ];
        let bound_val_evals: [F; N_STAGES] = if params.use_staged_val_claims {
            (0..N_STAGES)
                .map(|stage| {
                    let val_claim = accumulator
                        .get_virtual_polynomial_opening(
                            VirtualPolynomial::BytecodeValStage(stage),
                            SumcheckId::BytecodeReadRafAddressPhase,
                        )
                        .1;
                    val_claim + int_terms[stage]
                })
                .collect::<Vec<F>>()
                .try_into()
                .unwrap()
        } else {
            // Full mode: evaluate Val polynomials directly at r_address.
            params
                .val_polys
                .iter()
                .enumerate()
                .map(|(stage, poly)| poly.evaluate(&r_address_point.r) + int_terms[stage])
                .collect::<Vec<F>>()
                .try_into()
                .unwrap()
        };

        // Build RA polynomials from witness using MSB-first address challenges.
        let r_address_chunks = params
            .one_hot_params
            .compute_r_address_chunks::<F>(&r_address_point.r);
        let ra: Vec<RaPolynomial<u8, F>> = r_address_chunks
            .iter()
            .enumerate()
            .map(|(i, r_address_chunk)| {
                let ra_i: Vec<Option<u8>> = trace
                    .par_iter()
                    .map(|cycle| {
                        let pc = program.get_pc(cycle);
                        Some(params.one_hot_params.bytecode_pc_chunk(pc, i))
                    })
                    .collect();
                RaPolynomial::new(Arc::new(ra_i), EqPolynomial::evals(r_address_chunk))
            })
            .collect();

        let gruen_eq_polys = params
            .r_cycles
            .each_ref()
            .map(|r_cycle| GruenSplitEqPolynomial::new(r_cycle, BindingOrder::LowToHigh));

        Self {
            ra,
            gruen_eq_polys,
            bound_val_evals,
            params,
        }
    }

    fn compute_message_impl(&mut self, _previous_claim: F) -> UniPoly<F> {
        let degree = self.params.degree();

        let out_len = self.gruen_eq_polys[0].E_out_current().len();
        let in_len = self.gruen_eq_polys[0].E_in_current().len();
        let in_n_vars = in_len.log_2();

        let (mut q0_per_stage, mut q_evals_per_stage): ([F; N_STAGES], [Vec<F>; N_STAGES]) = (0
            ..out_len)
            .into_par_iter()
            .map(|j_hi| {
                let mut ra_eval_pairs = vec![(F::zero(), F::zero()); self.ra.len()];
                let mut ra_prod_evals = vec![F::zero(); degree - 1];
                let mut q0_unreduced: [_; N_STAGES] = array::from_fn(|_| F::Unreduced::zero());
                let mut q_unreduced: [_; N_STAGES] =
                    array::from_fn(|_| vec![F::Unreduced::zero(); degree - 1]);

                for j_lo in 0..in_len {
                    let j = j_lo + (j_hi << in_n_vars);

                    for (i, ra_i) in self.ra.iter().enumerate() {
                        let ra_i_eval_at_j_0 = ra_i.get_bound_coeff(j * 2);
                        let ra_i_eval_at_j_1 = ra_i.get_bound_coeff(j * 2 + 1);
                        ra_eval_pairs[i] = (ra_i_eval_at_j_0, ra_i_eval_at_j_1);
                    }

                    // Product polynomial evaluations on U_d = [1, 2, ..., d-1, ∞].
                    eval_linear_prod_assign(&ra_eval_pairs, &mut ra_prod_evals);
                    // Also compute P(0) = ∏_i ra_i(0) (needed to build q(0) directly).
                    let prod_at_0 = ra_eval_pairs
                        .iter()
                        .fold(F::one(), |acc, (p0, _p1)| acc * *p0);

                    for stage in 0..N_STAGES {
                        let eq_in_eval = self.gruen_eq_polys[stage].E_in_current()[j_lo];
                        q0_unreduced[stage] += eq_in_eval.mul_unreduced::<9>(prod_at_0);
                        for i in 0..degree - 1 {
                            q_unreduced[stage][i] +=
                                eq_in_eval.mul_unreduced::<9>(ra_prod_evals[i]);
                        }
                    }
                }

                let q0: [F; N_STAGES] = array::from_fn(|stage| {
                    let eq_out_eval = self.gruen_eq_polys[stage].E_out_current()[j_hi];
                    eq_out_eval * F::from_montgomery_reduce(q0_unreduced[stage])
                });
                let q_evals: [Vec<F>; N_STAGES] = array::from_fn(|stage| {
                    let eq_out_eval = self.gruen_eq_polys[stage].E_out_current()[j_hi];
                    q_unreduced[stage]
                        .iter()
                        .map(|v| eq_out_eval * F::from_montgomery_reduce(*v))
                        .collect()
                });
                (q0, q_evals)
            })
            .reduce(
                || {
                    (
                        array::from_fn(|_| F::zero()),
                        array::from_fn(|_| vec![F::zero(); degree - 1]),
                    )
                },
                |mut a, b| {
                    for stage in 0..N_STAGES {
                        a.0[stage] += b.0[stage];
                        a.1[stage]
                            .iter_mut()
                            .zip(b.1[stage].iter())
                            .for_each(|(x, y)| *x += *y);
                    }
                    a
                },
            );

        // Multiply by bound values (push into q).
        for stage in 0..N_STAGES {
            q0_per_stage[stage] *= self.bound_val_evals[stage];
            q_evals_per_stage[stage]
                .iter_mut()
                .for_each(|v| *v *= self.bound_val_evals[stage]);
        }

        let mut agg_round_poly = UniPoly::zero();
        for stage in 0..N_STAGES {
            let round_poly = self.gruen_eq_polys[stage]
                .gruen_poly_from_evals_with_q0(&q_evals_per_stage[stage], q0_per_stage[stage]);
            agg_round_poly += &(&round_poly * self.params.gamma_powers[stage]);
        }
        agg_round_poly
    }

    fn ingest_challenge_impl(&mut self, r_j: F::Challenge) {
        self.ra
            .iter_mut()
            .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh));
        self.gruen_eq_polys
            .iter_mut()
            .for_each(|poly| poly.bind(r_j));
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for BytecodeReadRafCycleSumcheckProver<F>
{
    fn degree(&self) -> usize {
        self.params.degree()
    }

    fn num_rounds(&self) -> usize {
        self.params.log_T
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::BytecodeReadRafAddrClaim,
                SumcheckId::BytecodeReadRafAddressPhase,
            )
            .1
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        self.compute_message_impl(previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.ingest_challenge_impl(r_j)
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let (r_address_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::BytecodeReadRafAddrClaim,
            SumcheckId::BytecodeReadRafAddressPhase,
        );
        let mut r_address_le = r_address_point.r;
        r_address_le.reverse();
        let mut full_challenges = r_address_le;
        full_challenges.extend_from_slice(sumcheck_challenges);
        let opening_point = self.params.normalize_opening_point(&full_challenges);
        let (r_address, r_cycle) = opening_point.split_at(self.params.log_K);

        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(&r_address.r);

        for i in 0..self.params.d {
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::BytecodeRa(i)],
                SumcheckId::BytecodeReadRaf,
                r_address_chunks[i].clone(),
                r_cycle.clone().into(),
                vec![self.ra[i].final_sumcheck_claim()],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct BytecodeReadRafSumcheckVerifier<F: JoltField> {
    params: BytecodeReadRafSumcheckParams<F>,
}

impl<F: JoltField> BytecodeReadRafSumcheckVerifier<F> {
    pub fn gen(
        program: &ProgramPreprocessing,
        n_cycle_vars: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        Self {
            params: BytecodeReadRafSumcheckParams::gen(
                program,
                n_cycle_vars,
                one_hot_params,
                opening_accumulator,
                transcript,
            ),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for BytecodeReadRafSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let (r_address_prime, r_cycle_prime) = opening_point.split_at(self.params.log_K);
        // r_cycle is bound LowToHigh, so reverse

        let int_poly = self.params.int_poly.evaluate(&r_address_prime.r);

        let ra_claims = (0..self.params.d).map(|i| {
            accumulator
                .get_committed_polynomial_opening(
                    CommittedPolynomial::BytecodeRa(i),
                    SumcheckId::BytecodeReadRaf,
                )
                .1
        });

        // We have a separate Val polynomial for each stage
        // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
        // So we would have:
        // Stage 1: gamma^0 * (Val_1 + gamma^5 * Int)
        // Stage 2: gamma^1 * (Val_2)
        // Stage 3: gamma^2 * (Val_3 + gamma^4 * Int)
        // Stage 4: gamma^3 * (Val_4)
        // Stage 5: gamma^4 * (Val_5)
        // Which matches with the input claim:
        // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * rv_4 + gamma^4 * rv_5 + gamma^5 * raf_1 + gamma^6 * raf_3
        let val = self
            .params
            .val_polys
            .iter()
            .zip(&self.params.r_cycles)
            .zip(&self.params.gamma_powers)
            .zip([
                int_poly * self.params.gamma_powers[5], // RAF for Stage1
                F::zero(),                              // There's no raf for Stage2
                int_poly * self.params.gamma_powers[4], // RAF for Stage3
                F::zero(),                              // There's no raf for Stage4
                F::zero(),                              // There's no raf for Stage5
            ])
            .map(|(((val, r_cycle), gamma), int_poly)| {
                (val.evaluate(&r_address_prime.r) + int_poly)
                    * EqPolynomial::<F>::mle(r_cycle, &r_cycle_prime.r)
                    * gamma
            })
            .sum::<F>();

        ra_claims.fold(val, |running, ra_claim| running * ra_claim)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let (r_address, r_cycle) = opening_point.split_at(self.params.log_K);

        // Compute r_address_chunks with proper padding
        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(&r_address.r);

        (0..self.params.d).for_each(|i| {
            let opening_point = [&r_address_chunks[i][..], &r_cycle.r].concat();
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::BytecodeRa(i)],
                SumcheckId::BytecodeReadRaf,
                opening_point,
            );
        });
    }
}

pub struct BytecodeReadRafAddressSumcheckVerifier<F: JoltField> {
    params: BytecodeReadRafSumcheckParams<F>,
}

impl<F: JoltField> BytecodeReadRafAddressSumcheckVerifier<F> {
    pub fn new(
        program: Option<&ProgramPreprocessing>,
        n_cycle_vars: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        program_mode: ProgramMode,
    ) -> Result<Self, ProofVerifyError> {
        let mut params = match program_mode {
            // Commitment mode: verifier MUST avoid O(K_bytecode) work here, and later stages will
            // relate staged Val claims to committed bytecode.
            ProgramMode::Committed => BytecodeReadRafSumcheckParams::gen_verifier(
                n_cycle_vars,
                one_hot_params,
                opening_accumulator,
                transcript,
            ),
            // Full mode: verifier materializes/evaluates bytecode-dependent polynomials (O(K_bytecode)).
            ProgramMode::Full => BytecodeReadRafSumcheckParams::gen(
                program.ok_or_else(|| {
                    ProofVerifyError::BytecodeTypeMismatch(
                        "expected Full bytecode preprocessing, got Committed".to_string(),
                    )
                })?,
                n_cycle_vars,
                one_hot_params,
                opening_accumulator,
                transcript,
            ),
        };
        params.use_staged_val_claims = program_mode == ProgramMode::Committed;
        Ok(Self { params })
    }

    /// Consume this verifier and return the underlying parameters (for Option B orchestration).
    pub fn into_params(self) -> BytecodeReadRafSumcheckParams<F> {
        self.params
    }

    pub fn into_cycle_verifier(self) -> BytecodeReadRafCycleSumcheckVerifier<F> {
        BytecodeReadRafCycleSumcheckVerifier {
            params: self.params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for BytecodeReadRafAddressSumcheckVerifier<F>
{
    fn degree(&self) -> usize {
        self.params.degree()
    }

    fn num_rounds(&self) -> usize {
        self.params.log_K
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::BytecodeReadRafAddrClaim,
                SumcheckId::BytecodeReadRafAddressPhase,
            )
            .1
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let mut r_address = sumcheck_challenges.to_vec();
        r_address.reverse();
        let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(r_address);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::BytecodeReadRafAddrClaim,
            SumcheckId::BytecodeReadRafAddressPhase,
            opening_point.clone(),
        );

        // Populate opening points for the Val-only bytecode stage claims emitted in Stage 6a,
        // but only when the staged-Val/claim-reduction path is enabled.
        if self.params.use_staged_val_claims {
            for stage in 0..N_STAGES {
                accumulator.append_virtual(
                    transcript,
                    VirtualPolynomial::BytecodeValStage(stage),
                    SumcheckId::BytecodeReadRafAddressPhase,
                    opening_point.clone(),
                );
            }
        }
    }
}

pub struct BytecodeReadRafCycleSumcheckVerifier<F: JoltField> {
    params: BytecodeReadRafSumcheckParams<F>,
}

impl<F: JoltField> BytecodeReadRafCycleSumcheckVerifier<F> {
    pub fn new(params: BytecodeReadRafSumcheckParams<F>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for BytecodeReadRafCycleSumcheckVerifier<F>
{
    fn degree(&self) -> usize {
        self.params.degree()
    }

    fn num_rounds(&self) -> usize {
        self.params.log_T
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::BytecodeReadRafAddrClaim,
                SumcheckId::BytecodeReadRafAddressPhase,
            )
            .1
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let (r_address_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::BytecodeReadRafAddrClaim,
            SumcheckId::BytecodeReadRafAddressPhase,
        );
        let mut r_address_le = r_address_point.r;
        r_address_le.reverse();
        let mut full_challenges = r_address_le;
        full_challenges.extend_from_slice(sumcheck_challenges);
        let opening_point = self.params.normalize_opening_point(&full_challenges);
        let (r_address_prime, r_cycle_prime) = opening_point.split_at(self.params.log_K);

        let int_poly = self.params.int_poly.evaluate(&r_address_prime.r);

        let ra_claims = (0..self.params.d).map(|i| {
            accumulator
                .get_committed_polynomial_opening(
                    CommittedPolynomial::BytecodeRa(i),
                    SumcheckId::BytecodeReadRaf,
                )
                .1
        });

        let int_terms = [
            int_poly * self.params.gamma_powers[5], // RAF for Stage1
            F::zero(),                              // There's no raf for Stage2
            int_poly * self.params.gamma_powers[4], // RAF for Stage3
            F::zero(),                              // There's no raf for Stage4
            F::zero(),                              // There's no raf for Stage5
        ];
        let val = if self.params.use_staged_val_claims {
            // Fast verifier path: consume Val_s(r_bc) claims emitted at the Stage 6a boundary,
            // rather than re-evaluating `val_polys` (O(K_bytecode)).
            (0..N_STAGES)
                .zip(self.params.r_cycles.iter())
                .zip(self.params.gamma_powers.iter())
                .zip(int_terms)
                .map(|(((stage, r_cycle), gamma), int_term)| {
                    let val_claim = accumulator
                        .get_virtual_polynomial_opening(
                            VirtualPolynomial::BytecodeValStage(stage),
                            SumcheckId::BytecodeReadRafAddressPhase,
                        )
                        .1;
                    (val_claim + int_term)
                        * EqPolynomial::<F>::mle(r_cycle, &r_cycle_prime.r)
                        * *gamma
                })
                .sum::<F>()
        } else {
            // Legacy verifier path: directly evaluate Val polynomials at r_bc (O(K_bytecode)).
            self.params
                .val_polys
                .iter()
                .zip(&self.params.r_cycles)
                .zip(&self.params.gamma_powers)
                .zip(int_terms)
                .map(|(((val, r_cycle), gamma), int_term)| {
                    (val.evaluate(&r_address_prime.r) + int_term)
                        * EqPolynomial::<F>::mle(r_cycle, &r_cycle_prime.r)
                        * *gamma
                })
                .sum::<F>()
        };

        ra_claims.fold(val, |running, ra_claim| running * ra_claim)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let (r_address_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::BytecodeReadRafAddrClaim,
            SumcheckId::BytecodeReadRafAddressPhase,
        );
        let mut r_address_le = r_address_point.r;
        r_address_le.reverse();
        let mut full_challenges = r_address_le;
        full_challenges.extend_from_slice(sumcheck_challenges);
        let opening_point = self.params.normalize_opening_point(&full_challenges);
        let (r_address, r_cycle) = opening_point.split_at(self.params.log_K);

        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(&r_address.r);

        (0..self.params.d).for_each(|i| {
            let opening_point = [&r_address_chunks[i][..], &r_cycle.r].concat();
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::BytecodeRa(i)],
                SumcheckId::BytecodeReadRaf,
                opening_point,
            );
        });
    }
}

#[derive(Allocative, Clone)]
pub struct BytecodeReadRafSumcheckParams<F: JoltField> {
    /// Index `i` stores `gamma^i`.
    pub gamma_powers: Vec<F>,
    /// RLC of stage rv_claims and RAF claims (per Stage1/Stage3) used as the sumcheck LHS.
    pub input_claim: F,
    /// RaParams
    pub one_hot_params: OneHotParams,
    /// Bytecode length.
    pub K: usize,
    /// log2(K) and log2(T) used to determine round counts.
    pub log_K: usize,
    pub log_T: usize,
    /// If true, Stage 6a emits `Val_s(r_bc)` as virtual openings and Stage 6b consumes them
    /// (instead of verifier re-materializing/evaluating `val_polys`).
    pub use_staged_val_claims: bool,
    /// Number of address chunks (and RA polynomials in the product).
    pub d: usize,
    /// Stage Val polynomials evaluated over address vars.
    pub val_polys: [MultilinearPolynomial<F>; N_STAGES],
    /// Stage rv claims.
    pub rv_claims: [F; N_STAGES],
    pub raf_claim: F,
    pub raf_shift_claim: F,
    /// Identity polynomial over address vars used to inject RAF contributions.
    pub int_poly: IdentityPolynomial<F>,
    pub r_cycles: [Vec<F::Challenge>; N_STAGES],
    /// Stage-specific batching gammas used to define Val(k) polynomials.
    /// Stored so later claim reductions can reconstruct lane weights without resampling the transcript.
    pub stage1_gammas: Vec<F>,
    pub stage2_gammas: Vec<F>,
    pub stage3_gammas: Vec<F>,
    pub stage4_gammas: Vec<F>,
    pub stage5_gammas: Vec<F>,
}

impl<F: JoltField> BytecodeReadRafSumcheckParams<F> {
    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheckParams::gen")]
    pub fn gen(
        program: &ProgramPreprocessing,
        n_cycle_vars: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        Self::gen_impl(
            Some(program),
            n_cycle_vars,
            one_hot_params,
            opening_accumulator,
            transcript,
            true,
        )
    }

    /// Verifier-side generator: avoids materializing Val(k) polynomials (O(K_bytecode)).
    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheckParams::gen_verifier")]
    pub fn gen_verifier(
        n_cycle_vars: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        Self::gen_impl(
            None,
            n_cycle_vars,
            one_hot_params,
            opening_accumulator,
            transcript,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gen_impl(
        program: Option<&ProgramPreprocessing>,
        n_cycle_vars: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        compute_val_polys: bool,
    ) -> Self {
        let gamma_powers = transcript.challenge_scalar_powers(7);

        // Generate all stage-specific gamma powers upfront (order must match verifier)
        let stage1_gammas: Vec<F> = transcript.challenge_scalar_powers(2 + NUM_CIRCUIT_FLAGS);
        let stage2_gammas: Vec<F> = transcript.challenge_scalar_powers(4);
        let stage3_gammas: Vec<F> = transcript.challenge_scalar_powers(9);
        let stage4_gammas: Vec<F> = transcript.challenge_scalar_powers(3);
        let stage5_gammas: Vec<F> = transcript.challenge_scalar_powers(2 + NUM_LOOKUP_TABLES);

        // Compute rv_claims (these don't iterate bytecode, just query opening accumulator)
        let rv_claim_1 = Self::compute_rv_claim_1(opening_accumulator, &stage1_gammas);
        let rv_claim_2 = Self::compute_rv_claim_2(opening_accumulator, &stage2_gammas);
        let rv_claim_3 = Self::compute_rv_claim_3(opening_accumulator, &stage3_gammas);
        let rv_claim_4 = Self::compute_rv_claim_4(opening_accumulator, &stage4_gammas);
        let rv_claim_5 = Self::compute_rv_claim_5(opening_accumulator, &stage5_gammas);
        let rv_claims = [rv_claim_1, rv_claim_2, rv_claim_3, rv_claim_4, rv_claim_5];

        let val_polys = if compute_val_polys {
            let instructions = &program
                .expect("compute_val_polys requires program preprocessing")
                .instructions;
            // Pre-compute eq_r_register for stages 4 and 5 (they use different r_register points)
            let r_register_4 = opening_accumulator
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::RdWa,
                    SumcheckId::RegistersReadWriteChecking,
                )
                .0
                .r;
            let eq_r_register_4 =
                EqPolynomial::<F>::evals(&r_register_4[..(REGISTER_COUNT as usize).log_2()]);

            let r_register_5 = opening_accumulator
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::RdWa,
                    SumcheckId::RegistersValEvaluation,
                )
                .0
                .r;
            let eq_r_register_5 =
                EqPolynomial::<F>::evals(&r_register_5[..(REGISTER_COUNT as usize).log_2()]);

            // Fused pass: compute all val polynomials in a single parallel iteration
            Self::compute_val_polys(
                instructions,
                &eq_r_register_4,
                &eq_r_register_5,
                &stage1_gammas,
                &stage2_gammas,
                &stage3_gammas,
                &stage4_gammas,
                &stage5_gammas,
            )
        } else {
            // Verifier doesn't need these (and must not iterate over bytecode).
            array::from_fn(|_| MultilinearPolynomial::default())
        };

        let int_poly = IdentityPolynomial::new(one_hot_params.bytecode_k.log_2());

        let (_, raf_claim) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        let (_, raf_shift_claim) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        let input_claim = [
            rv_claim_1,
            rv_claim_2,
            rv_claim_3,
            rv_claim_4,
            rv_claim_5,
            raf_claim,
            raf_shift_claim,
        ]
        .iter()
        .zip(&gamma_powers)
        .map(|(claim, g)| *claim * g)
        .sum();

        let (r_cycle_1, _) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);
        let (r_cycle_2, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::Jump),
            SumcheckId::SpartanProductVirtualization,
        );
        let (r_cycle_3, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );
        let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, r_cycle_4) = r.split_at((REGISTER_COUNT as usize).log_2());
        let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, r_cycle_5) = r.split_at((REGISTER_COUNT as usize).log_2());
        let r_cycles = [
            r_cycle_1.r,
            r_cycle_2.r,
            r_cycle_3.r,
            r_cycle_4.r,
            r_cycle_5.r,
        ];

        // Note: We don't have r_address at this point (it comes from sumcheck_challenges),
        // so we initialize r_address_chunks as empty and will compute it later
        Self {
            gamma_powers,
            input_claim,
            one_hot_params: one_hot_params.clone(),
            K: one_hot_params.bytecode_k,
            log_K: one_hot_params.bytecode_k.log_2(),
            d: one_hot_params.bytecode_d,
            log_T: n_cycle_vars,
            use_staged_val_claims: false,
            val_polys,
            rv_claims,
            raf_claim,
            raf_shift_claim,
            int_poly,
            r_cycles,
            stage1_gammas,
            stage2_gammas,
            stage3_gammas,
            stage4_gammas,
            stage5_gammas,
        }
    }

    /// Fused computation of all Val polynomials in a single parallel pass over bytecode.
    ///
    /// This computes all 5 stage-specific Val(k) polynomials simultaneously, avoiding
    /// 5 separate passes through the bytecode. Each stage has its own gamma powers
    /// and formula for Val(k).
    #[allow(clippy::too_many_arguments)]
    fn compute_val_polys(
        bytecode: &[Instruction],
        eq_r_register_4: &[F],
        eq_r_register_5: &[F],
        stage1_gammas: &[F],
        stage2_gammas: &[F],
        stage3_gammas: &[F],
        stage4_gammas: &[F],
        stage5_gammas: &[F],
    ) -> [MultilinearPolynomial<F>; N_STAGES] {
        let K = bytecode.len();

        // Pre-allocate output vectors for each stage
        let mut vals: [Vec<F>; N_STAGES] = array::from_fn(|_| unsafe_allocate_zero_vec(K));
        let [v0, v1, v2, v3, v4] = &mut vals;

        // Fused parallel iteration: compute all 5 val entries for each instruction
        bytecode
            .par_iter()
            .zip(v0.par_iter_mut())
            .zip(v1.par_iter_mut())
            .zip(v2.par_iter_mut())
            .zip(v3.par_iter_mut())
            .zip(v4.par_iter_mut())
            .for_each(|(((((instruction, o0), o1), o2), o3), o4)| {
                let instr = instruction.normalize();
                let circuit_flags = instruction.circuit_flags();
                let instr_flags = instruction.instruction_flags();

                // Stage 1 (Spartan outer sumcheck)
                // Val(k) = unexpanded_pc(k) + γ·imm(k)
                //          + γ²·circuit_flags[0](k) + γ³·circuit_flags[1](k) + ...
                // This virtualizes claims output by Spartan's "outer" sumcheck.
                {
                    let mut lc = F::from_u64(instr.address as u64);
                    lc += instr.operands.imm.field_mul(stage1_gammas[1]);
                    // sanity check
                    debug_assert!(
                        !circuit_flags[CircuitFlags::IsCompressed]
                            || !circuit_flags[CircuitFlags::DoNotUpdateUnexpandedPC]
                    );
                    for (flag, gamma_power) in circuit_flags.iter().zip(stage1_gammas[2..].iter()) {
                        if *flag {
                            lc += *gamma_power;
                        }
                    }
                    *o0 = lc;
                }

                // Stage 2 (product virtualization, de-duplicated factors)
                // Val(k) = jump_flag(k) + γ·branch_flag(k)
                //          + γ²·is_rd_not_zero_flag(k) + γ³·write_lookup_output_to_rd_flag(k)
                // where jump_flag(k) = 1 if instruction k is a jump, 0 otherwise;
                //       branch_flag(k) = 1 if instruction k is a branch, 0 otherwise;
                //       is_rd_not_zero_flag(k) = 1 if instruction k has rd != 0;
                //       write_lookup_output_to_rd_flag(k) = 1 if instruction k writes lookup output to rd.
                // This Val matches the fused product sumcheck.
                {
                    let mut lc = F::zero();
                    if circuit_flags[CircuitFlags::Jump] {
                        lc += stage2_gammas[0];
                    }
                    if instr_flags[InstructionFlags::Branch] {
                        lc += stage2_gammas[1];
                    }
                    if instr_flags[InstructionFlags::IsRdNotZero] {
                        lc += stage2_gammas[2];
                    }
                    if circuit_flags[CircuitFlags::WriteLookupOutputToRD] {
                        lc += stage2_gammas[3];
                    }
                    *o1 = lc;
                }

                // Stage 3 (Shift sumcheck)
                // Val(k) = imm(k) + γ·unexpanded_pc(k)
                //          + γ²·left_operand_is_rs1_value(k) + γ³·left_operand_is_pc(k)
                //          + γ⁴·right_operand_is_rs2_value(k) + γ⁵·right_operand_is_imm(k)
                //          + γ⁶·is_noop(k) + γ⁷·virtual_instruction(k) + γ⁸·is_first_in_sequence(k)
                // This virtualizes claims output by the ShiftSumcheck.
                {
                    let mut lc = F::from_i128(instr.operands.imm);
                    lc += stage3_gammas[1].mul_u64(instr.address as u64);
                    if instr_flags[InstructionFlags::LeftOperandIsRs1Value] {
                        lc += stage3_gammas[2];
                    }
                    if instr_flags[InstructionFlags::LeftOperandIsPC] {
                        lc += stage3_gammas[3];
                    }
                    if instr_flags[InstructionFlags::RightOperandIsRs2Value] {
                        lc += stage3_gammas[4];
                    }
                    if instr_flags[InstructionFlags::RightOperandIsImm] {
                        lc += stage3_gammas[5];
                    }
                    if instr_flags[InstructionFlags::IsNoop] {
                        lc += stage3_gammas[6];
                    }
                    if circuit_flags[CircuitFlags::VirtualInstruction] {
                        lc += stage3_gammas[7];
                    }
                    if circuit_flags[CircuitFlags::IsFirstInSequence] {
                        lc += stage3_gammas[8];
                    }
                    *o2 = lc;
                }

                // Stage 4 (registers read/write checking sumcheck)
                // Val(k) = eq(rd(k), r_register) + γ·eq(rs1(k), r_register) + γ²·eq(rs2(k), r_register)
                // where rd(k, r) = 1 if the k'th instruction in the bytecode has rd = r,
                // and analogously for rs1(k, r) and rs2(k, r).
                // This virtualizes claims output by the registers read/write checking sumcheck.
                {
                    let rd_eq = instr
                        .operands
                        .rd
                        .map_or(F::zero(), |r| eq_r_register_4[r as usize]);
                    let rs1_eq = instr
                        .operands
                        .rs1
                        .map_or(F::zero(), |r| eq_r_register_4[r as usize]);
                    let rs2_eq = instr
                        .operands
                        .rs2
                        .map_or(F::zero(), |r| eq_r_register_4[r as usize]);
                    *o3 = rd_eq * stage4_gammas[0]
                        + rs1_eq * stage4_gammas[1]
                        + rs2_eq * stage4_gammas[2];
                }

                // Stage 5 (registers val-evaluation + instruction lookups sumcheck)
                // Val(k) = eq(rd(k), r_register) + γ·raf_flag(k)
                //          + γ²·lookup_table_flag[0](k) + γ³·lookup_table_flag[1](k) + ...
                // where rd(k, r) = 1 if the k'th instruction in the bytecode has rd = r,
                // and raf_flag(k) = 1 if instruction k is NOT interleaved operands.
                // This virtualizes the claim output by the registers val-evaluation sumcheck
                // and the instruction lookups sumcheck.
                {
                    let mut lc = instr
                        .operands
                        .rd
                        .map_or(F::zero(), |r| eq_r_register_5[r as usize]);
                    if !circuit_flags.is_interleaved_operands() {
                        lc += stage5_gammas[1];
                    }
                    if let Some(table) = instruction.lookup_table() {
                        let table_index = LookupTables::enum_index(&table);
                        lc += stage5_gammas[2 + table_index];
                    }
                    *o4 = lc;
                }
            });

        vals.map(MultilinearPolynomial::from)
    }

    fn compute_rv_claim_1(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        gamma_powers: &[F],
    ) -> F {
        let (_, unexpanded_pc_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, imm_claim) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);

        let circuit_flag_claims: Vec<F> = CircuitFlags::iter()
            .map(|flag| {
                opening_accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::OpFlags(flag),
                        SumcheckId::SpartanOuter,
                    )
                    .1
            })
            .collect();

        std::iter::once(unexpanded_pc_claim)
            .chain(std::iter::once(imm_claim))
            .chain(circuit_flag_claims)
            .zip_eq(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }

    fn compute_rv_claim_2(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        gamma_powers: &[F],
    ) -> F {
        let (_, jump_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::Jump),
            SumcheckId::SpartanProductVirtualization,
        );
        let (_, branch_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
            SumcheckId::SpartanProductVirtualization,
        );
        let (_, rd_wa_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsRdNotZero),
            SumcheckId::SpartanProductVirtualization,
        );
        let (_, write_lookup_output_to_rd_flag_claim) = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
                SumcheckId::SpartanProductVirtualization,
            );

        [
            jump_claim,
            branch_claim,
            rd_wa_claim,
            write_lookup_output_to_rd_flag_claim,
        ]
        .into_iter()
        .zip_eq(gamma_powers)
        .map(|(claim, gamma)| claim * gamma)
        .sum()
    }

    fn compute_rv_claim_3(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        gamma_powers: &[F],
    ) -> F {
        let (_, imm_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Imm,
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, spartan_shift_unexpanded_pc_claim) = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::UnexpandedPC,
                SumcheckId::SpartanShift,
            );
        let (_, instruction_input_unexpanded_pc_claim) = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::UnexpandedPC,
                SumcheckId::InstructionInputVirtualization,
            );

        assert_eq!(
            spartan_shift_unexpanded_pc_claim,
            instruction_input_unexpanded_pc_claim
        );

        let unexpanded_pc_claim = spartan_shift_unexpanded_pc_claim;
        let (_, left_is_rs1_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsRs1Value),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, left_is_pc_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsPC),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, right_is_rs2_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsRs2Value),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, right_is_imm_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsImm),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, is_noop_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
        );
        let (_, is_virtual_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
        );
        let (_, is_first_in_sequence_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
        );

        [
            imm_claim,
            unexpanded_pc_claim,
            left_is_rs1_claim,
            left_is_pc_claim,
            right_is_rs2_claim,
            right_is_imm_claim,
            is_noop_claim,
            is_virtual_claim,
            is_first_in_sequence_claim,
        ]
        .into_iter()
        .zip_eq(gamma_powers)
        .map(|(claim, gamma)| claim * gamma)
        .sum()
    }

    fn compute_rv_claim_4(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        gamma_powers: &[F],
    ) -> F {
        std::iter::empty()
            .chain(once(VirtualPolynomial::RdWa))
            .chain(once(VirtualPolynomial::Rs1Ra))
            .chain(once(VirtualPolynomial::Rs2Ra))
            .map(|vp| {
                opening_accumulator
                    .get_virtual_polynomial_opening(vp, SumcheckId::RegistersReadWriteChecking)
                    .1
            })
            .zip(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum::<F>()
    }

    fn compute_rv_claim_5(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        gamma_powers: &[F],
    ) -> F {
        let (_, rd_wa_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );

        let (_, raf_flag_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
        );

        let mut sum = rd_wa_claim * gamma_powers[0];
        sum += raf_flag_claim * gamma_powers[1];

        // Add lookup table flag claims from InstructionReadRaf
        for i in 0..LookupTables::<XLEN>::COUNT {
            let (_, claim) = opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
            );
            sum += claim * gamma_powers[2 + i];
        }

        sum
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for BytecodeReadRafSumcheckParams<F> {
    fn degree(&self) -> usize {
        self.d + 1
    }

    fn num_rounds(&self) -> usize {
        self.log_K + self.log_T
    }

    fn input_claim(&self, _: &dyn OpeningAccumulator<F>) -> F {
        self.input_claim
    }

    fn normalize_opening_point(
        &self,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut r = sumcheck_challenges.to_vec();
        r[0..self.log_K].reverse();
        r[self.log_K..].reverse();
        OpeningPoint::new(r)
    }
}
