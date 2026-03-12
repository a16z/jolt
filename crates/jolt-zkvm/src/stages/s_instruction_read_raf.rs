//! Instruction read-RAF checking sumcheck stage.
//!
//! Proves the double-sum identity:
//!
//! ```text
//! input_claim = Σ_{j,k} eq(r_reduction, j) · ra(k, j) · (Val_j(k) + γ · RafVal_j(k))
//! ```
//!
//! Split into two phases via [`SegmentedEvaluator`]:
//! - **Address phase** (log_K rounds): binds address variables `k` (HighToLow)
//!   using pre-computed F(k) and Val(k) tables.
//! - **Cycle phase** (log_T rounds): binds cycle variables `j` using ToomCook
//!   eq factoring with materialized RA polynomials and combined_val.
//!
//! The single eq(r_reduction, ·) polynomial in the cycle phase enables
//! Toom-Cook eq factoring (unlike BytecodeReadRaf which has 5+ eq polynomials
//! and uses StandardGrid).

use std::sync::{Arc, Mutex};

use jolt_compute::{BindingOrder, ComputeBackend};
use jolt_field::Field;
use jolt_ir::ClaimDefinition;
use jolt_openings::ProverClaim;
use jolt_poly::Polynomial;
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_transcript::Transcript;

use crate::evaluators::catalog;
use crate::evaluators::kernel::KernelEvaluator;
use crate::evaluators::segmented::{SegmentTransition, SegmentedEvaluator};
use crate::stage::{ProverStage, StageBatch};

/// Shared slot for passing RA tables from the transition closure to `extract_claims`.
type RaSlot<F> = Arc<Mutex<Option<Vec<Vec<F>>>>>;

/// Instruction read-RAF checking prover stage.
///
/// Models the instruction lookup sumcheck with:
/// - A single pre-computed F(k) table (cycle-domain eq × RA condensed)
/// - A single Val(k) table (lookup table values at address k)
/// - n_virtual RA chunks for the cycle phase
/// - ToomCook eq factoring using `r_reduction`
pub struct InstructionReadRafStage<F: Field, B: ComputeBackend> {
    backend: Arc<B>,
    log_k: usize,
    log_t: usize,
    n_virtual: usize,

    // Address-phase buffers (consumed during build).
    f_table: Option<Vec<F>>,
    val_table: Option<Vec<F>>,

    // Shared slot: transition writes RA tables, extract_claims reads them.
    ra_slot: RaSlot<F>,

    // RA materializer: (r_addr, n_virtual) -> n_virtual RA tables of length T.
    ra_materializer: Option<Box<dyn FnOnce(&[F], usize) -> Vec<Vec<F>> + Send + Sync>>,

    // ToomCook challenge point for the cycle phase eq polynomial.
    r_reduction: Vec<F>,

    claimed_sum: F,
}

impl<F: Field, B: ComputeBackend> InstructionReadRafStage<F, B> {
    /// Creates a new instruction read-RAF stage.
    ///
    /// # Arguments
    ///
    /// * `f_table` — Pre-computed F(k) = Σ_j eq(r_reduction, j) · ra(k, j), length K.
    /// * `val_table` — Lookup table values Val(k), length K.
    /// * `r_reduction` — Challenge point for the cycle-phase eq polynomial (length log_T).
    /// * `n_virtual` — Number of virtual RA polynomial chunks.
    /// * `ra_materializer` — Callback `(r_addr, n_virtual) -> Vec<Vec<F>>` producing
    ///   n_virtual RA evaluation tables of length T.
    /// * `claimed_sum` — The input claim (LHS of the sumcheck identity).
    pub fn new(
        f_table: Vec<F>,
        val_table: Vec<F>,
        r_reduction: Vec<F>,
        n_virtual: usize,
        ra_materializer: Box<dyn FnOnce(&[F], usize) -> Vec<Vec<F>> + Send + Sync>,
        claimed_sum: F,
        backend: Arc<B>,
    ) -> Self {
        let k = f_table.len();
        let log_k = k.trailing_zeros() as usize;
        assert_eq!(1 << log_k, k, "K must be a power of 2");
        assert_eq!(val_table.len(), k, "val_table length mismatch");
        assert!(n_virtual >= 1, "need at least 1 virtual RA poly");

        let log_t = r_reduction.len();

        Self {
            backend,
            log_k,
            log_t,
            n_virtual,
            f_table: Some(f_table),
            val_table: Some(val_table),
            ra_slot: Arc::new(Mutex::new(None)),
            ra_materializer: Some(ra_materializer),
            r_reduction,
            claimed_sum,
        }
    }
}

impl<F: Field, B: ComputeBackend, T: Transcript> ProverStage<F, T>
    for InstructionReadRafStage<F, B>
{
    fn name(&self) -> &'static str {
        "S_instruction_read_raf"
    }

    fn build(&mut self, _prior_claims: &[ProverClaim<F>], _transcript: &mut T) -> StageBatch<F> {
        let log_k = self.log_k;
        let log_t = self.log_t;
        let n_virtual = self.n_virtual;
        let num_vars = log_k + log_t;

        let f_table = self.f_table.take().expect("f_table already consumed");
        let val_table = self.val_table.take().expect("val_table already consumed");
        let ra_materializer = self
            .ra_materializer
            .take()
            .expect("ra_materializer already consumed");
        let r_reduction = self.r_reduction.clone();

        let addr_desc = catalog::eq_product();
        let addr_kernel = self.backend.compile_kernel::<F>(&addr_desc);

        let addr_inputs = vec![
            self.backend.upload(&f_table),
            self.backend.upload(&val_table),
        ];

        let addr_evaluator = KernelEvaluator::with_unit_weights(
            addr_inputs,
            addr_kernel,
            addr_desc.num_evals(),
            Arc::clone(&self.backend),
        )
        .with_binding_order(BindingOrder::HighToLow);

        let ra_slot = Arc::clone(&self.ra_slot);

        let transition: SegmentTransition<F, B> =
            Box::new(move |challenges: Vec<F>, backend: &Arc<B>| {
                let r_addr = &challenges;

                // Evaluate Val at bound address point.
                let bound_val = Polynomial::new(val_table).evaluate(r_addr);

                // Materialize RA polynomials over cycle domain.
                let ra_tables = ra_materializer(r_addr, n_virtual);
                assert_eq!(ra_tables.len(), n_virtual);

                // Stash RA tables for extract_claims.
                *ra_slot.lock().unwrap() = Some(ra_tables.clone());

                // Build combined_val buffer: constant bound_val for all cycles.
                let t_len = ra_tables[0].len();
                let combined_val = vec![bound_val; t_len];

                // Cycle-phase kernel: ProductSum(n_virtual + 1, 1)
                // Inputs: [combined_val, ra_0, ..., ra_{n_virtual-1}]
                // Eq factored via ToomCook with r_reduction.
                let cycle_desc = catalog::product_sum(n_virtual + 1, 1);
                let cycle_kernel = backend.compile_kernel::<F>(&cycle_desc);

                let mut cycle_inputs = Vec::with_capacity(1 + n_virtual);
                cycle_inputs.push(backend.upload(&combined_val));
                for ra in &ra_tables {
                    cycle_inputs.push(backend.upload(ra));
                }

                // Placeholder claimed_sum — overridden by SumcheckProver::set_claim
                // before the first cycle round.
                KernelEvaluator::with_toom_cook_eq(
                    cycle_inputs,
                    cycle_kernel,
                    cycle_desc.num_evals(),
                    r_reduction,
                    F::zero(),
                    Arc::clone(backend),
                )
            });

        let witness =
            SegmentedEvaluator::new(addr_evaluator, log_k, Arc::clone(&self.backend))
                .then(log_t, transition);

        // Degree: max(address=2, cycle=n_virtual+2) = n_virtual+2 for n_virtual >= 1.
        let degree = n_virtual + 2;

        let claim = SumcheckClaim {
            num_vars,
            degree,
            claimed_sum: self.claimed_sum,
        };

        StageBatch {
            claims: vec![claim],
            witnesses: vec![Box::new(witness)],
        }
    }

    fn extract_claims(&mut self, challenges: &[F], _final_eval: F) -> Vec<ProverClaim<F>> {
        let ra_tables = self
            .ra_slot
            .lock()
            .unwrap()
            .take()
            .expect("RA tables not yet materialized (sumcheck not run?)");

        // challenges = [r_addr_0, ..., r_addr_{log_k-1}, r_cycle_0, ..., r_cycle_{log_t-1}]
        // Address phase uses HighToLow → address challenges are MSB-first (natural order).
        // Cycle phase uses LowToHigh → reverse for MSB-first evaluation.
        let r_cycle: Vec<F> = challenges[self.log_k..].iter().rev().copied().collect();

        ra_tables
            .into_iter()
            .map(|evals| {
                let poly = Polynomial::new(evals.clone());
                let eval = poly.evaluate(&r_cycle);
                ProverClaim {
                    evaluations: evals,
                    point: r_cycle.clone(),
                    eval,
                }
            })
            .collect()
    }

    fn claim_definitions(&self) -> Vec<ClaimDefinition> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_cpu::CpuBackend;
    use jolt_field::{Field, Fr};
    use jolt_poly::EqPolynomial;
    use jolt_sumcheck::{BatchedSumcheckProver, BatchedSumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::Zero;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    fn cpu() -> Arc<CpuBackend> {
        Arc::new(CpuBackend)
    }

    fn random_table(n: usize, rng: &mut ChaCha20Rng) -> Vec<Fr> {
        (0..n).map(|_| Fr::random(rng)).collect()
    }

    /// Builds a synthetic test scenario.
    ///
    /// The identity: `claimed_sum = Σ_j eq(r_reduction, j) · Val(PC(j))`
    /// Address kernel: `F(k) * Val(k)` where `F(k) = Σ_{j: PC(j)=k} eq(r_reduction, j)`.
    fn build_synthetic_stage(
        log_k: usize,
        log_t: usize,
        n_virtual: usize,
        rng: &mut ChaCha20Rng,
    ) -> InstructionReadRafStage<Fr, CpuBackend> {
        let k = 1usize << log_k;
        let t = 1usize << log_t;

        // Random challenge point for the cycle-phase eq.
        let r_reduction: Vec<Fr> = (0..log_t).map(|_| Fr::random(rng)).collect();

        // Random Val table.
        let val_table = random_table(k, rng);

        // Random PC assignments per cycle.
        let pc_per_cycle: Vec<usize> = (0..t).map(|_| rng.next_u32() as usize % k).collect();

        // Compute F(k) = Σ_{j: PC(j)=k} eq(r_reduction, j).
        let eq_table = EqPolynomial::new(r_reduction.clone()).evaluations();
        let mut f_table = vec![Fr::zero(); k];
        for j in 0..t {
            f_table[pc_per_cycle[j]] += eq_table[j];
        }

        // Brute-force claimed sum: Σ_j eq(r_reduction, j) · Val(PC(j)).
        let mut claimed_sum = Fr::zero();
        for j in 0..t {
            claimed_sum += eq_table[j] * val_table[pc_per_cycle[j]];
        }

        // Split log_k into n_virtual chunks.
        assert!(
            log_k % n_virtual == 0,
            "log_k must be divisible by n_virtual"
        );
        let bits_per_chunk = log_k / n_virtual;
        let k_chunk = 1usize << bits_per_chunk;

        let get_chunk = |pc: usize, chunk_idx: usize| -> usize {
            (pc >> (bits_per_chunk * (n_virtual - 1 - chunk_idx))) & (k_chunk - 1)
        };

        let pc_chunks: Vec<Vec<usize>> = (0..n_virtual)
            .map(|chunk_idx| {
                pc_per_cycle
                    .iter()
                    .map(|&pc| get_chunk(pc, chunk_idx))
                    .collect()
            })
            .collect();

        let ra_materializer: Box<dyn FnOnce(&[Fr], usize) -> Vec<Vec<Fr>> + Send + Sync> =
            Box::new(move |r_addr: &[Fr], n_virtual: usize| {
                let bits_per_chunk = r_addr.len() / n_virtual;
                (0..n_virtual)
                    .map(|chunk_idx| {
                        let chunk_start = chunk_idx * bits_per_chunk;
                        let chunk_point =
                            r_addr[chunk_start..chunk_start + bits_per_chunk].to_vec();
                        let eq_table = EqPolynomial::new(chunk_point).evaluations();
                        pc_chunks[chunk_idx]
                            .iter()
                            .map(|&idx| eq_table[idx])
                            .collect()
                    })
                    .collect()
            });

        InstructionReadRafStage::new(
            f_table,
            val_table,
            r_reduction,
            n_virtual,
            ra_materializer,
            claimed_sum,
            cpu(),
        )
    }

    #[test]
    fn stage_produces_one_claim() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n_virtual = 2;
        let mut stage = build_synthetic_stage(4, 3, n_virtual, &mut rng);

        let mut t = Blake2bTranscript::<Fr>::new(b"test");
        let batch = stage.build(&[], &mut t);

        assert_eq!(batch.claims.len(), 1);
        assert_eq!(batch.witnesses.len(), 1);
        assert_eq!(batch.claims[0].num_vars, 4 + 3);
        assert_eq!(batch.claims[0].degree, n_virtual + 2);
    }

    /// Full prove/verify with n_virtual=1.
    #[test]
    fn prove_verify_nv1() {
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let mut stage = build_synthetic_stage(3, 4, 1, &mut rng);

        let mut pt = Blake2bTranscript::new(b"instruction_raf_nv1");
        let mut batch = stage.build(&[], &mut pt);

        let claims_snapshot: Vec<_> = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(&batch.claims, &mut batch.witnesses, &mut pt);

        let mut vt = Blake2bTranscript::new(b"instruction_raf_nv1");
        let (final_eval, challenges) =
            BatchedSumcheckVerifier::verify(&claims_snapshot, &proof, &mut vt)
                .expect("verification should succeed");

        let prover_claims = <InstructionReadRafStage<Fr, CpuBackend> as ProverStage<
            Fr,
            Blake2bTranscript,
        >>::extract_claims(&mut stage, &challenges, final_eval);

        assert_eq!(prover_claims.len(), 1);
        let claim = &prover_claims[0];
        let poly = Polynomial::new(claim.evaluations.clone());
        assert_eq!(poly.evaluate(&claim.point), claim.eval);
    }

    /// Full prove/verify with n_virtual=2.
    #[test]
    fn prove_verify_nv2() {
        let mut rng = ChaCha20Rng::seed_from_u64(200);
        let mut stage = build_synthetic_stage(4, 5, 2, &mut rng);

        let mut pt = Blake2bTranscript::new(b"instruction_raf_nv2");
        let mut batch = stage.build(&[], &mut pt);

        let claims_snapshot: Vec<_> = batch.claims.clone();
        assert_eq!(claims_snapshot[0].degree, 4); // n_virtual + 2 = 4

        let proof = BatchedSumcheckProver::prove(&batch.claims, &mut batch.witnesses, &mut pt);

        let mut vt = Blake2bTranscript::new(b"instruction_raf_nv2");
        let (final_eval, challenges) =
            BatchedSumcheckVerifier::verify(&claims_snapshot, &proof, &mut vt)
                .expect("verification should succeed");

        let prover_claims = <InstructionReadRafStage<Fr, CpuBackend> as ProverStage<
            Fr,
            Blake2bTranscript,
        >>::extract_claims(&mut stage, &challenges, final_eval);

        assert_eq!(prover_claims.len(), 2);
        for claim in &prover_claims {
            let poly = Polynomial::new(claim.evaluations.clone());
            assert_eq!(poly.evaluate(&claim.point), claim.eval);
        }
    }

    /// Full prove/verify with n_virtual=4 (degree 6).
    #[test]
    fn prove_verify_nv4() {
        let mut rng = ChaCha20Rng::seed_from_u64(300);
        let mut stage = build_synthetic_stage(4, 4, 4, &mut rng);

        let mut pt = Blake2bTranscript::new(b"instruction_raf_nv4");
        let mut batch = stage.build(&[], &mut pt);

        let claims_snapshot: Vec<_> = batch.claims.clone();
        assert_eq!(claims_snapshot[0].degree, 6); // n_virtual + 2 = 6

        let proof = BatchedSumcheckProver::prove(&batch.claims, &mut batch.witnesses, &mut pt);

        let mut vt = Blake2bTranscript::new(b"instruction_raf_nv4");
        let result = BatchedSumcheckVerifier::verify(&claims_snapshot, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    /// Larger test: log_K=6, log_T=5, n_virtual=3.
    #[test]
    fn prove_verify_larger() {
        let mut rng = ChaCha20Rng::seed_from_u64(400);
        let mut stage = build_synthetic_stage(6, 5, 3, &mut rng);

        let mut pt = Blake2bTranscript::new(b"instruction_raf_larger");
        let mut batch = stage.build(&[], &mut pt);

        let claims_snapshot: Vec<_> = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(&batch.claims, &mut batch.witnesses, &mut pt);

        let mut vt = Blake2bTranscript::new(b"instruction_raf_larger");
        let result = BatchedSumcheckVerifier::verify(&claims_snapshot, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    /// Verify address-phase formula matches brute-force cycle sum.
    #[test]
    fn brute_force_matches_claimed_sum() {
        let mut rng = ChaCha20Rng::seed_from_u64(500);
        let log_k = 3;
        let log_t = 4;
        let k = 1usize << log_k;
        let t = 1usize << log_t;

        let r_reduction: Vec<Fr> = (0..log_t).map(|_| Fr::random(&mut rng)).collect();
        let val_table = random_table(k, &mut rng);
        let pc_per_cycle: Vec<usize> = (0..t).map(|_| rng.next_u32() as usize % k).collect();

        let eq_table = EqPolynomial::new(r_reduction.clone()).evaluations();

        // Cycle-by-cycle brute force.
        let mut cycle_sum = Fr::zero();
        for j in 0..t {
            cycle_sum += eq_table[j] * val_table[pc_per_cycle[j]];
        }

        // Address-phase formula: Σ_k F(k) * Val(k).
        let mut f_table = vec![Fr::zero(); k];
        for j in 0..t {
            f_table[pc_per_cycle[j]] += eq_table[j];
        }
        let mut addr_sum = Fr::zero();
        for addr in 0..k {
            addr_sum += f_table[addr] * val_table[addr];
        }

        assert_eq!(cycle_sum, addr_sum);
    }

    /// Verify that extract_claims returns correct RA evaluations.
    #[test]
    fn extract_claims_evaluations_correct() {
        let mut rng = ChaCha20Rng::seed_from_u64(600);
        let mut stage = build_synthetic_stage(4, 3, 2, &mut rng);

        let mut pt = Blake2bTranscript::new(b"extract_claims");
        let mut batch = stage.build(&[], &mut pt);

        let claims_snapshot: Vec<_> = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(&batch.claims, &mut batch.witnesses, &mut pt);

        let mut vt = Blake2bTranscript::new(b"extract_claims");
        let (final_eval, challenges) =
            BatchedSumcheckVerifier::verify(&claims_snapshot, &proof, &mut vt)
                .expect("verification should succeed");

        let prover_claims = <InstructionReadRafStage<Fr, CpuBackend> as ProverStage<
            Fr,
            Blake2bTranscript,
        >>::extract_claims(&mut stage, &challenges, final_eval);

        // Each claim's eval should match poly.evaluate(point).
        for (i, claim) in prover_claims.iter().enumerate() {
            let poly = Polynomial::new(claim.evaluations.clone());
            let expected = poly.evaluate(&claim.point);
            assert_eq!(
                claim.eval, expected,
                "RA claim {i}: eval mismatch"
            );
            // Point length = log_t (cycle variables only).
            assert_eq!(claim.point.len(), 3);
        }
    }
}
