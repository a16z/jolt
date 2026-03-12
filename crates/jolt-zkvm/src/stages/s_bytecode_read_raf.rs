//! Bytecode read-RAF checking sumcheck stage.
//!
//! Proves the double-sum identity:
//!
//! ```text
//! input_claim = Σ_{j,k} ra(k,j) · [Σ_s γ^s · eq_s(j) · Val'_s(k)
//!                                    + γ_entry · eq_entry(j) · f_trace(k) · f_expected(k)]
//! ```
//!
//! Split into two phases via [`SegmentedEvaluator`]:
//! - **Address phase** (log_K rounds): binds address variables `k` using
//!   pre-computed F_s(k) tables absorbing the cycle-domain eq and RA.
//! - **Cycle phase** (log_T rounds): binds cycle variables `j` using
//!   materialized RA polynomials with combined_eq as weights.

use std::sync::{Arc, Mutex};

use jolt_compute::ComputeBackend;
use jolt_field::Field;
use jolt_ir::{ClaimDefinition, ExprBuilder, KernelDescriptor, KernelShape};
use jolt_openings::ProverClaim;
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_transcript::Transcript;

use crate::evaluators::kernel::KernelEvaluator;
use crate::evaluators::segmented::{SegmentTransition, SegmentedEvaluator};
use crate::stage::{ProverStage, StageBatch};

/// Number of prior stages whose claims are folded into bytecode read-RAF.
const N_STAGES: usize = 5;

/// Shared slot for passing RA tables from the transition closure to `extract_claims`.
type RaSlot<F> = Arc<Mutex<Option<Vec<Vec<F>>>>>;

/// Bytecode read-RAF checking prover stage.
pub struct BytecodeReadRafStage<F: Field, B: ComputeBackend> {
    backend: Arc<B>,
    log_k: usize,
    log_t: usize,
    d: usize,

    // Address-phase buffers (consumed during build).
    f_tables: Option<Vec<Vec<F>>>,
    val_tables: Option<Vec<Vec<F>>>,
    f_entry_trace: Option<Vec<F>>,
    f_entry_expected: Option<Vec<F>>,

    // Shared slot: transition writes RA tables, extract_claims reads them.
    ra_slot: RaSlot<F>,

    // Challenge data.
    gamma_powers: Vec<F>,
    r_cycles: Vec<Vec<F>>,
    ra_materializer: Option<Box<dyn FnOnce(&[F], usize) -> Vec<Vec<F>> + Send + Sync>>,
    claimed_sum: F,
}

impl<F: Field, B: ComputeBackend> BytecodeReadRafStage<F, B> {
    /// Creates a new stage from pre-computed tables and challenge data.
    ///
    /// # Arguments
    ///
    /// * `f_tables` — Per-stage F_s(k) tables (length K each), pre-computed
    ///   via split-eq: `F_s(k) = Σ_j eq_s(j) · ra(k, j)`.
    /// * `val_tables` — Per-stage Val'_s(k) tables (length K each), with
    ///   RAF identity corrections folded in.
    /// * `f_entry_trace` — One-hot at PC(cycle_0), length K.
    /// * `f_entry_expected` — One-hot at entry_bytecode_index, length K.
    /// * `gamma_powers` — `[γ^0, ..., γ^{N_STAGES-1}, γ^5, γ^6, γ_entry]`.
    /// * `r_cycles` — Per-stage cycle challenge points (N_STAGES vectors of length log_T).
    /// * `d` — Number of RA address chunks.
    /// * `ra_materializer` — Callback `(r_addr, d) -> Vec<Vec<F>>` producing d
    ///   RA evaluation tables of length T.
    /// * `claimed_sum` — The input claim (LHS of the sumcheck identity).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        f_tables: Vec<Vec<F>>,
        val_tables: Vec<Vec<F>>,
        f_entry_trace: Vec<F>,
        f_entry_expected: Vec<F>,
        gamma_powers: Vec<F>,
        r_cycles: Vec<Vec<F>>,
        d: usize,
        ra_materializer: Box<dyn FnOnce(&[F], usize) -> Vec<Vec<F>> + Send + Sync>,
        claimed_sum: F,
        backend: Arc<B>,
    ) -> Self {
        assert_eq!(f_tables.len(), N_STAGES, "need {N_STAGES} F tables");
        assert_eq!(val_tables.len(), N_STAGES, "need {N_STAGES} Val tables");
        assert_eq!(r_cycles.len(), N_STAGES, "need {N_STAGES} r_cycle points");

        let k = f_tables[0].len();
        let log_k = k.trailing_zeros() as usize;
        assert_eq!(1 << log_k, k, "K must be a power of 2");
        let log_t = r_cycles[0].len();

        for (i, ft) in f_tables.iter().enumerate() {
            assert_eq!(ft.len(), k, "F_table[{i}] length mismatch");
        }
        for (i, vt) in val_tables.iter().enumerate() {
            assert_eq!(vt.len(), k, "Val_table[{i}] length mismatch");
        }
        assert_eq!(f_entry_trace.len(), k);
        assert_eq!(f_entry_expected.len(), k);

        Self {
            backend,
            log_k,
            log_t,
            d,
            f_tables: Some(f_tables),
            val_tables: Some(val_tables),
            f_entry_trace: Some(f_entry_trace),
            f_entry_expected: Some(f_entry_expected),
            ra_slot: Arc::new(Mutex::new(None)),
            gamma_powers,
            r_cycles,
            ra_materializer: Some(ra_materializer),
            claimed_sum,
        }
    }
}

impl<F: Field, B: ComputeBackend, T: Transcript> ProverStage<F, T>
    for BytecodeReadRafStage<F, B>
{
    fn name(&self) -> &'static str {
        "S_bytecode_read_raf"
    }

    fn build(&mut self, _prior_claims: &[ProverClaim<F>], _transcript: &mut T) -> StageBatch<F> {
        let log_k = self.log_k;
        let log_t = self.log_t;
        let d = self.d;
        let num_vars = log_k + log_t;

        let f_tables = self.f_tables.take().expect("f_tables already consumed");
        let val_tables = self.val_tables.take().expect("val_tables already consumed");
        let f_entry_trace = self.f_entry_trace.take().expect("f_entry_trace already consumed");
        let f_entry_expected = self
            .f_entry_expected
            .take()
            .expect("f_entry_expected already consumed");
        let ra_materializer = self
            .ra_materializer
            .take()
            .expect("ra_materializer already consumed");

        let (addr_desc, addr_challenges) = build_address_descriptor::<F>(&self.gamma_powers);
        let addr_kernel = self
            .backend
            .compile_kernel_with_challenges::<F>(&addr_desc, &addr_challenges);

        let mut addr_inputs = Vec::with_capacity(2 * N_STAGES + 2);
        for ft in &f_tables {
            addr_inputs.push(self.backend.upload(ft));
        }
        for vt in &val_tables {
            addr_inputs.push(self.backend.upload(vt));
        }
        addr_inputs.push(self.backend.upload(&f_entry_trace));
        addr_inputs.push(self.backend.upload(&f_entry_expected));

        let addr_evaluator = KernelEvaluator::with_unit_weights(
            addr_inputs,
            addr_kernel,
            addr_desc.num_evals(),
            Arc::clone(&self.backend),
        );

        let gamma_powers = self.gamma_powers.clone();
        let r_cycles = self.r_cycles.clone();
        let ra_slot = Arc::clone(&self.ra_slot);

        let transition: SegmentTransition<F, B> =
            Box::new(move |challenges: Vec<F>, backend: &Arc<B>| {
                let r_addr = &challenges;

                // Evaluate Val'_s and f_entry_expected at the bound address point.
                let bound_vals: Vec<F> = val_tables
                    .iter()
                    .map(|vt| Polynomial::new(vt.clone()).evaluate(r_addr))
                    .collect();
                let bound_f_entry =
                    Polynomial::new(f_entry_expected).evaluate(r_addr);

                // Materialize RA polynomials over cycle domain.
                let ra_tables = ra_materializer(r_addr, d);
                assert_eq!(ra_tables.len(), d);
                let t_len = ra_tables[0].len();

                // Stash RA tables for extract_claims.
                *ra_slot.lock().unwrap() = Some(ra_tables.clone());

                // Build combined_eq[j] = Σ_s γ^s · bound_val_s · eq_s(j)
                //                       + γ_entry · bound_f_entry · eq_entry(j)
                let gamma_entry = *gamma_powers.last().unwrap();
                let eq_tables: Vec<Vec<F>> = r_cycles
                    .iter()
                    .map(|rc| EqPolynomial::new(rc.clone()).evaluations())
                    .collect();
                let eq_entry =
                    EqPolynomial::new(vec![F::zero(); r_cycles[0].len()]).evaluations();

                let mut combined_eq = vec![F::zero(); t_len];
                for j in 0..t_len {
                    let mut v = F::zero();
                    for s in 0..N_STAGES {
                        v += gamma_powers[s] * bound_vals[s] * eq_tables[s][j];
                    }
                    v += gamma_entry * bound_f_entry * eq_entry[j];
                    combined_eq[j] = v;
                }

                // Cycle-phase kernel: combined_eq(j) · Π_{i=0}^{d-1} ra_i(j)
                // combined_eq is a regular input (not weight buffer), degree = d+1.
                let cycle_desc = build_cycle_descriptor(d);
                let cycle_kernel = backend.compile_kernel::<F>(&cycle_desc);

                // Inputs: [combined_eq, ra_0, ..., ra_{d-1}]
                let mut cycle_inputs = Vec::with_capacity(1 + d);
                cycle_inputs.push(backend.upload(&combined_eq));
                for ra in &ra_tables {
                    cycle_inputs.push(backend.upload(ra));
                }

                KernelEvaluator::with_unit_weights(
                    cycle_inputs,
                    cycle_kernel,
                    cycle_desc.num_evals(),
                    Arc::clone(backend),
                )
            });

        let witness =
            SegmentedEvaluator::new(addr_evaluator, log_k, Arc::clone(&self.backend))
                .then(log_t, transition);

        // The claim degree is max(address_degree, cycle_degree) = max(2, d+1) = d+1 for d >= 1.
        let degree = d + 1;

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
        // LowToHigh binding → reverse for MSB-first evaluation.
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

/// Builds the address-phase kernel descriptor.
///
/// Formula: `Σ_s γ^s · F_s(k) · Val_s(k) + γ_entry · f_trace(k) · f_expected(k)`
///
/// Input layout: `[F_0, ..., F_4, Val_0, ..., Val_4, f_trace, f_expected]`
fn build_address_descriptor<F: Field>(gamma_powers: &[F]) -> (KernelDescriptor, Vec<F>) {
    let num_inputs = 2 * N_STAGES + 2;
    let b = ExprBuilder::new();

    let mut sum = b.challenge(0) * b.opening(0) * b.opening(N_STAGES as u32);
    for s in 1..N_STAGES {
        sum = sum
            + b.challenge(s as u32)
                * b.opening(s as u32)
                * b.opening((N_STAGES + s) as u32);
    }

    let trace_idx = (2 * N_STAGES) as u32;
    let expected_idx = (2 * N_STAGES + 1) as u32;
    sum = sum + b.challenge(N_STAGES as u32) * b.opening(trace_idx) * b.opening(expected_idx);

    let mut challenges = Vec::with_capacity(N_STAGES + 1);
    for s in 0..N_STAGES {
        challenges.push(gamma_powers[s]);
    }
    challenges.push(*gamma_powers.last().expect("gamma_powers non-empty"));

    let desc = KernelDescriptor {
        shape: KernelShape::Custom {
            expr: b.build(sum),
            num_inputs,
        },
        degree: 2,
        tensor_split: None,
    };

    (desc, challenges)
}

/// Builds the cycle-phase kernel descriptor:
/// `opening(0) · Π_{i=1}^{d} opening(i)` = `combined_eq · Π ra_i`.
///
/// Input layout: `[combined_eq, ra_0, ..., ra_{d-1}]`, total d+1 inputs.
/// Degree = d+1 (combined_eq adds one degree to the d-way RA product).
fn build_cycle_descriptor(d: usize) -> KernelDescriptor {
    assert!(d >= 1, "d must be at least 1");
    let b = ExprBuilder::new();
    // opening(0) = combined_eq, opening(1..d) = ra polys
    let mut product = b.opening(0);
    for i in 1..=d {
        product = product * b.opening(i as u32);
    }
    KernelDescriptor {
        shape: KernelShape::Custom {
            expr: b.build(product),
            num_inputs: d + 1,
        },
        degree: d + 1,
        tensor_split: None,
    }
}

/// Brute-force computation of the bytecode read-RAF sumcheck claim.
///
/// Used in tests to verify the sumcheck identity.
pub fn brute_force_bytecode_read_raf<F: Field>(
    f_tables: &[Vec<F>],
    val_tables: &[Vec<F>],
    f_entry_trace: &[F],
    f_entry_expected: &[F],
    gamma_powers: &[F],
    r_cycles: &[Vec<F>],
    ra_materializer: &dyn Fn(usize, usize) -> F,
    d: usize,
) -> F {
    let k = f_tables[0].len();
    let t = 1usize << r_cycles[0].len();
    let gamma_entry = *gamma_powers.last().unwrap();

    let eq_tables: Vec<Vec<F>> = r_cycles
        .iter()
        .map(|rc| EqPolynomial::new(rc.clone()).evaluations())
        .collect();
    let eq_entry = EqPolynomial::new(vec![F::zero(); r_cycles[0].len()]).evaluations();

    let mut sum = F::zero();
    for j in 0..t {
        for addr in 0..k {
            // ra(addr, j) = product of per-chunk RA evals
            let mut ra_prod = F::one();
            for chunk in 0..d {
                ra_prod *= ra_materializer(addr * d + chunk, j);
            }

            // Σ_s γ^s · eq_s(j) · Val_s(addr) + γ_entry · eq_entry(j) · f_trace · f_expected
            let mut inner = F::zero();
            for s in 0..N_STAGES {
                inner += gamma_powers[s] * eq_tables[s][j] * val_tables[s][addr];
            }
            inner +=
                gamma_entry * eq_entry[j] * f_entry_trace[addr] * f_entry_expected[addr];

            sum += ra_prod * inner;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_cpu::CpuBackend;
    use jolt_field::{Field, Fr};
    use jolt_sumcheck::{BatchedSumcheckProver, BatchedSumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn cpu() -> Arc<CpuBackend> {
        Arc::new(CpuBackend)
    }

    fn random_table(n: usize, rng: &mut ChaCha20Rng) -> Vec<Fr> {
        (0..n).map(|_| Fr::random(rng)).collect()
    }

    /// Builds a synthetic test scenario with known RA structure.
    ///
    /// Returns `(stage, claimed_sum)` ready for prove/verify.
    fn build_synthetic_stage(
        log_k: usize,
        log_t: usize,
        d: usize,
        rng: &mut ChaCha20Rng,
    ) -> BytecodeReadRafStage<Fr, CpuBackend> {
        let k = 1usize << log_k;
        let t = 1usize << log_t;

        // Random per-stage challenge points.
        let r_cycles: Vec<Vec<Fr>> = (0..N_STAGES)
            .map(|_| (0..log_t).map(|_| Fr::random(rng)).collect())
            .collect();

        // Random gamma powers.
        let gamma = Fr::random(rng);
        let mut gamma_powers = Vec::with_capacity(N_STAGES + 3);
        let mut g = Fr::one();
        for _ in 0..(N_STAGES + 3) {
            gamma_powers.push(g);
            g *= gamma;
        }

        // Random Val tables (length K each).
        let val_tables: Vec<Vec<Fr>> = (0..N_STAGES).map(|_| random_table(k, rng)).collect();

        // Random RA structure: for each chunk i and address k, store which
        // cycle indices map to this address. For simplicity, use random
        // per-position values in [0, k_chunk).
        //
        // We'll model ra_i(chunk_val, j) = eq(chunk_val, pc_chunk_i(j))
        // where pc_chunk_i(j) is the i-th chunk of cycle j's bytecode address.
        //
        // For synthetic testing: generate random PC assignments per cycle,
        // then compute F_s, RA tables, and brute-force the claimed sum.
        let pc_per_cycle: Vec<usize> = (0..t).map(|_| rng.next_u32() as usize % k).collect();

        // Compute F_s(k) = Σ_j eq_s(j) · ra(k, j)
        // For this test, ra(k, j) = 1 if PC(j) == k, else 0.
        // So F_s(k) = Σ_{j: PC(j)=k} eq_s(j).
        let eq_tables: Vec<Vec<Fr>> = r_cycles
            .iter()
            .map(|rc| EqPolynomial::new(rc.clone()).evaluations())
            .collect();

        let mut f_tables: Vec<Vec<Fr>> = (0..N_STAGES).map(|_| vec![Fr::zero(); k]).collect();
        for j in 0..t {
            let pc = pc_per_cycle[j];
            for s in 0..N_STAGES {
                f_tables[s][pc] += eq_tables[s][j];
            }
        }

        // Entry constraint: f_trace is one-hot at PC(0), f_expected at some random address.
        let entry_pc = pc_per_cycle[0];
        let entry_expected = rng.next_u32() as usize % k;
        let mut f_entry_trace = vec![Fr::zero(); k];
        f_entry_trace[entry_pc] = Fr::one();
        let mut f_entry_expected = vec![Fr::zero(); k];
        f_entry_expected[entry_expected] = Fr::one();

        // Split log_K into d chunks of bits_per_chunk bits each.
        assert!(log_k % d == 0, "log_k must be divisible by d");
        let bits_per_chunk = log_k / d;
        let k_chunk = 1usize << bits_per_chunk;

        // Extract chunk i from a PC value.
        let get_chunk = |pc: usize, chunk_idx: usize| -> usize {
            (pc >> (bits_per_chunk * (d - 1 - chunk_idx))) & (k_chunk - 1)
        };

        // Per-cycle, per-chunk indices for RA materialization.
        let pc_chunks: Vec<Vec<usize>> = (0..d)
            .map(|chunk_idx| {
                pc_per_cycle
                    .iter()
                    .map(|&pc| get_chunk(pc, chunk_idx))
                    .collect()
            })
            .collect();

        let pc_chunks_clone = pc_chunks.clone();
        let ra_materializer: Box<dyn FnOnce(&[Fr], usize) -> Vec<Vec<Fr>> + Send + Sync> =
            Box::new(move |r_addr: &[Fr], d: usize| {
                // Split r_addr into d chunks of bits_per_chunk each.
                let bits_per_chunk = r_addr.len() / d;
                (0..d)
                    .map(|chunk_idx| {
                        let chunk_start = chunk_idx * bits_per_chunk;
                        let chunk_point =
                            r_addr[chunk_start..chunk_start + bits_per_chunk].to_vec();
                        let eq_table = EqPolynomial::new(chunk_point).evaluations();
                        pc_chunks_clone[chunk_idx]
                            .iter()
                            .map(|&idx| eq_table[idx])
                            .collect()
                    })
                    .collect()
            });

        // Compute brute-force claimed sum.
        let gamma_entry = gamma_powers[N_STAGES + 2];
        let eq_entry = EqPolynomial::new(vec![Fr::zero(); log_t]).evaluations();

        let mut claimed_sum = Fr::zero();
        for j in 0..t {
            let pc = pc_per_cycle[j];
            // ra(k, j) = Π_i 1_{chunk_i(k) == chunk_i(PC(j))}
            // Only addr = PC(j) gives ra_prod = 1 (all chunks match).
            let addr = pc;

            let mut inner = Fr::zero();
            for s in 0..N_STAGES {
                inner += gamma_powers[s] * eq_tables[s][j] * val_tables[s][addr];
            }
            inner +=
                gamma_entry * eq_entry[j] * f_entry_trace[addr] * f_entry_expected[addr];
            claimed_sum += inner;
        }

        BytecodeReadRafStage::new(
            f_tables,
            val_tables,
            f_entry_trace,
            f_entry_expected,
            gamma_powers,
            r_cycles,
            d,
            ra_materializer,
            claimed_sum,
            cpu(),
        )
    }

    use rand_core::RngCore;

    /// Basic test: stage produces exactly 1 sumcheck claim.
    #[test]
    fn stage_produces_one_claim() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut stage = build_synthetic_stage(3, 4, 1, &mut rng);

        let mut t = Blake2bTranscript::<Fr>::new(b"test");
        let batch = stage.build(&[], &mut t);

        assert_eq!(batch.claims.len(), 1);
        assert_eq!(batch.witnesses.len(), 1);
        assert_eq!(batch.claims[0].num_vars, 3 + 4);
        assert_eq!(batch.claims[0].degree, 2); // d=1, degree = d+1 = 2
    }

    /// Full prove/verify round trip with d=1.
    #[test]
    fn prove_verify_d1() {
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let mut stage = build_synthetic_stage(2, 3, 1, &mut rng);

        let mut pt = Blake2bTranscript::new(b"bytecode_raf");
        let mut batch = stage.build(&[], &mut pt);

        let claims_snapshot: Vec<_> = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(&batch.claims, &mut batch.witnesses, &mut pt);

        let mut vt = Blake2bTranscript::new(b"bytecode_raf");
        let (final_eval, challenges) =
            BatchedSumcheckVerifier::verify(&claims_snapshot, &proof, &mut vt)
                .expect("verification should succeed");

        let prover_claims = <BytecodeReadRafStage<Fr, CpuBackend> as ProverStage<
            Fr,
            Blake2bTranscript,
        >>::extract_claims(&mut stage, &challenges, final_eval);

        // d=1 → 1 RA claim
        assert_eq!(prover_claims.len(), 1);

        // Verify that the claimed eval matches polynomial evaluation.
        let claim = &prover_claims[0];
        let poly = Polynomial::new(claim.evaluations.clone());
        assert_eq!(poly.evaluate(&claim.point), claim.eval);
    }

    /// Larger test: log_K=4, log_T=5, d=1.
    #[test]
    fn prove_verify_larger() {
        let mut rng = ChaCha20Rng::seed_from_u64(200);
        let mut stage = build_synthetic_stage(4, 5, 1, &mut rng);

        let mut pt = Blake2bTranscript::new(b"bytecode_raf_larger");
        let mut batch = stage.build(&[], &mut pt);

        let claims_snapshot: Vec<_> = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(&batch.claims, &mut batch.witnesses, &mut pt);

        let mut vt = Blake2bTranscript::new(b"bytecode_raf_larger");
        let result = BatchedSumcheckVerifier::verify(&claims_snapshot, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    /// Multi-chunk test: log_K=4, log_T=5, d=2 (2 chunks of 2 bits each).
    #[test]
    fn prove_verify_d2() {
        let mut rng = ChaCha20Rng::seed_from_u64(400);
        let mut stage = build_synthetic_stage(4, 5, 2, &mut rng);

        let mut pt = Blake2bTranscript::new(b"bytecode_raf_d2");
        let mut batch = stage.build(&[], &mut pt);

        let claims_snapshot: Vec<_> = batch.claims.clone();
        assert_eq!(claims_snapshot[0].degree, 3); // d=2, degree = d+1 = 3

        let proof = BatchedSumcheckProver::prove(&batch.claims, &mut batch.witnesses, &mut pt);

        let mut vt = Blake2bTranscript::new(b"bytecode_raf_d2");
        let (final_eval, challenges) =
            BatchedSumcheckVerifier::verify(&claims_snapshot, &proof, &mut vt)
                .expect("verification should succeed");

        let prover_claims = <BytecodeReadRafStage<Fr, CpuBackend> as ProverStage<
            Fr,
            Blake2bTranscript,
        >>::extract_claims(&mut stage, &challenges, final_eval);

        // d=2 → 2 RA claims
        assert_eq!(prover_claims.len(), 2);
        for claim in &prover_claims {
            let poly = Polynomial::new(claim.evaluations.clone());
            assert_eq!(poly.evaluate(&claim.point), claim.eval);
        }
    }

    /// Large multi-chunk: log_K=6, log_T=4, d=3 (3 chunks of 2 bits each).
    #[test]
    fn prove_verify_d3() {
        let mut rng = ChaCha20Rng::seed_from_u64(500);
        let mut stage = build_synthetic_stage(6, 4, 3, &mut rng);

        let mut pt = Blake2bTranscript::new(b"bytecode_raf_d3");
        let mut batch = stage.build(&[], &mut pt);

        let claims_snapshot: Vec<_> = batch.claims.clone();
        assert_eq!(claims_snapshot[0].degree, 4); // d=3, degree = d+1 = 4

        let proof = BatchedSumcheckProver::prove(&batch.claims, &mut batch.witnesses, &mut pt);

        let mut vt = Blake2bTranscript::new(b"bytecode_raf_d3");
        let result = BatchedSumcheckVerifier::verify(&claims_snapshot, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    /// Verify address-phase formula matches cycle-per-cycle brute force.
    ///
    /// For d=1 and entry constraint matching (entry_pc == entry_expected),
    /// the double sum collapses: only addr = PC(j) contributes.
    /// The address-phase formula Σ_k F_s(k)·Val_s(k) should equal
    /// the cycle-by-cycle sum.
    #[test]
    fn brute_force_matches_claimed_sum() {
        let mut rng = ChaCha20Rng::seed_from_u64(300);
        let log_k = 2;
        let log_t = 3;
        let k = 1usize << log_k;
        let t = 1usize << log_t;

        let r_cycles: Vec<Vec<Fr>> = (0..N_STAGES)
            .map(|_| (0..log_t).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let gamma = Fr::random(&mut rng);
        let mut gamma_powers = Vec::new();
        let mut g = Fr::one();
        for _ in 0..(N_STAGES + 3) {
            gamma_powers.push(g);
            g *= gamma;
        }

        let val_tables: Vec<Vec<Fr>> = (0..N_STAGES).map(|_| random_table(k, &mut rng)).collect();
        let pc_per_cycle: Vec<usize> = (0..t).map(|_| rng.next_u32() as usize % k).collect();

        let eq_tables: Vec<Vec<Fr>> = r_cycles
            .iter()
            .map(|rc| EqPolynomial::new(rc.clone()).evaluations())
            .collect();

        let mut f_tables: Vec<Vec<Fr>> = (0..N_STAGES).map(|_| vec![Fr::zero(); k]).collect();
        for j in 0..t {
            let pc = pc_per_cycle[j];
            for s in 0..N_STAGES {
                f_tables[s][pc] += eq_tables[s][j];
            }
        }

        let entry_pc = pc_per_cycle[0];
        let mut f_entry_trace = vec![Fr::zero(); k];
        f_entry_trace[entry_pc] = Fr::one();
        let mut f_entry_expected = vec![Fr::zero(); k];
        f_entry_expected[entry_pc] = Fr::one();

        let gamma_entry = gamma_powers[N_STAGES + 2];
        let eq_entry = EqPolynomial::new(vec![Fr::zero(); log_t]).evaluations();

        // Cycle-by-cycle brute force (d=1: only PC(j) contributes).
        let mut cycle_sum = Fr::zero();
        for j in 0..t {
            let addr = pc_per_cycle[j];
            let mut inner = Fr::zero();
            for s in 0..N_STAGES {
                inner += gamma_powers[s] * eq_tables[s][j] * val_tables[s][addr];
            }
            inner += gamma_entry * eq_entry[j] * f_entry_trace[addr] * f_entry_expected[addr];
            cycle_sum += inner;
        }

        // Address-phase formula: Σ_k [Σ_s γ^s · F_s(k) · Val_s(k) + γ_entry · f_trace · f_expected]
        let mut addr_sum = Fr::zero();
        for addr in 0..k {
            let mut val = Fr::zero();
            for s in 0..N_STAGES {
                val += gamma_powers[s] * f_tables[s][addr] * val_tables[s][addr];
            }
            val += gamma_entry * f_entry_trace[addr] * f_entry_expected[addr];
            addr_sum += val;
        }

        assert_eq!(cycle_sum, addr_sum);
    }
}
