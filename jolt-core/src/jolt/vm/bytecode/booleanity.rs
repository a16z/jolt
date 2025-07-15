use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use crate::jolt::vm::bytecode::BytecodePreprocessing;
use crate::poly::opening_proof::{OpeningPoint, VerifierOpeningAccumulator, BIG_ENDIAN};
use crate::subprotocols::sumcheck::CacheSumcheckOpenings;
use crate::utils::errors::ProofVerifyError;
use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::ProverOpeningAccumulator,
    },
    subprotocols::sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    utils::{math::Math, thread::unsafe_allocate_zero_vec, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;
use tracer::instruction::RV32IMCycle;

struct BooleanityProverState<F: JoltField> {
    B: MultilinearPolynomial<F>,
    D: MultilinearPolynomial<F>,
    H: MultilinearPolynomial<F>,
    G: Vec<F>,
    F: Vec<F>,
    eq_r_r: F,
    // Precomputed arrays for phase 1
    eq_km_c: [[F; 3]; 2],
    eq_km_c_squared: [[F; 3]; 2],
}

struct BooleanityVerifierState<F: JoltField> {
    r_address: Option<Vec<F>>,
    r_cycle: Option<Vec<F>>,
}

pub struct BooleanitySumcheck<F: JoltField> {
    /// Input claim: always F::zero() for booleanity
    input_claim: F,
    /// K value shared by prover and verifier
    K: usize,
    /// T value shared by prover and verifier
    T: usize,
    /// Prover state
    prover_state: Option<BooleanityProverState<F>>,
    /// Verifier state
    verifier_state: Option<BooleanityVerifierState<F>>,
    /// Cached ra claim after sumcheck completes
    pub ra_claim_prime: Option<F>,
    /// Current round
    current_round: usize,
    /// Store preprocessing and trace for phase transition
    preprocessing: Option<Arc<BytecodePreprocessing>>,
    trace: Option<Arc<[RV32IMCycle]>>,
}

impl<F: JoltField> BooleanitySumcheck<F> {
    pub fn new(
        preprocessing: &BytecodePreprocessing,
        trace: &[RV32IMCycle],
        r: &[F],
        D: Vec<F>,
        G: Vec<F>,
        K: usize,
        T: usize,
    ) -> Self {
        let B = MultilinearPolynomial::from(EqPolynomial::evals(r));

        // Initialize F for the first phase
        let mut F_vec: Vec<F> = unsafe_allocate_zero_vec(K);
        F_vec[0] = F::one();

        // Compute H (will be used in phase 2)
        let H: Vec<F> = trace
            .par_iter()
            .map(|cycle| preprocessing.get_pc(cycle))
            .map(|_pc| F::zero()) // Will be computed during phase 1
            .collect();
        let H = MultilinearPolynomial::from(H);
        let D = MultilinearPolynomial::from(D);

        // Precompute EQ(k_m, c) for k_m \in {0, 1} and c \in {0, 2, 3}
        let eq_km_c: [[F; 3]; 2] = [
            [
                F::one(),        // eq(0, 0) = 0 * 0 + (1 - 0) * (1 - 0)
                F::from_i64(-1), // eq(0, 2) = 0 * 2 + (1 - 0) * (1 - 2)
                F::from_i64(-2), // eq(0, 3) = 0 * 3 + (1 - 0) * (1 - 3)
            ],
            [
                F::zero(),     // eq(1, 0) = 1 * 0 + (1 - 1) * (1 - 0)
                F::from_u8(2), // eq(1, 2) = 1 * 2 + (1 - 1) * (1 - 2)
                F::from_u8(3), // eq(1, 3) = 1 * 3 + (1 - 1) * (1 - 3)
            ],
        ];

        // Precompute EQ(k_m, c)^2 for k_m \in {0, 1} and c \in {0, 2, 3}
        let eq_km_c_squared: [[F; 3]; 2] = [
            [F::one(), F::one(), F::from_u8(4)],
            [F::zero(), F::from_u8(4), F::from_u8(9)],
        ];

        Self {
            input_claim: F::zero(),
            K,
            T,
            prover_state: Some(BooleanityProverState {
                B,
                D,
                H,
                G,
                F: F_vec,
                eq_r_r: F::zero(), // Will be set after phase 1
                eq_km_c,
                eq_km_c_squared,
            }),
            verifier_state: None,
            ra_claim_prime: None,
            current_round: 0,
            preprocessing: Some(Arc::new(preprocessing.clone())),
            trace: Some(Arc::from(trace)),
        }
    }

    pub fn new_verifier(
        K: usize,
        T: usize,
        r_address: Vec<F>,
        r_cycle: Vec<F>,
        ra_claim_prime: F,
    ) -> Self {
        Self {
            input_claim: F::zero(),
            K,
            T,
            prover_state: None,
            verifier_state: Some(BooleanityVerifierState::<F> {
                r_address: Some(r_address),
                r_cycle: Some(r_cycle),
            }),
            ra_claim_prime: Some(ra_claim_prime),
            current_round: 0,
            preprocessing: None,
            trace: None,
        }
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2() + self.T.log_2()
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message(&mut self, round: usize) -> Vec<F> {
        let K_log = self.K.log_2();

        if round < K_log {
            // Phase 1: First log(K) rounds
            self.compute_phase1_message(round)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message(round - K_log)
        }
    }

    fn bind(&mut self, r_j: F, round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        let K_log = self.K.log_2();

        if round < K_log {
            // Phase 1: Bind B and update F
            prover_state.B.bind_parallel(r_j, BindingOrder::LowToHigh);

            // Update F for this round (see Equation 55)
            let (F_left, F_right) = prover_state.F.split_at_mut(1 << round);
            F_left
                .par_iter_mut()
                .zip(F_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r_j;
                    *x -= *y;
                });

            // If transitioning to phase 2, prepare H
            if round == K_log - 1 {
                prover_state.eq_r_r = prover_state.B.final_sumcheck_claim();
                // Compute H using the final F values
                let preprocessing = self.preprocessing.as_ref().unwrap();
                let trace = self.trace.as_ref().unwrap();
                let H_vec: Vec<F> = trace
                    .par_iter()
                    .map(|cycle| preprocessing.get_pc(cycle))
                    .map(|pc| prover_state.F[pc as usize])
                    .collect();
                prover_state.H = MultilinearPolynomial::from(H_vec);
            }
        } else {
            // Phase 2: Bind D and H
            rayon::join(
                || prover_state.D.bind_parallel(r_j, BindingOrder::LowToHigh),
                || prover_state.H.bind_parallel(r_j, BindingOrder::LowToHigh),
            );
        }

        self.current_round += 1;
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let ra_claim_prime = self.ra_claim_prime.expect("ra_claim_prime not set");
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");

        // Split r into r_address_prime and r_cycle_prime
        let (r_address_prime, r_cycle_prime) = r.split_at(self.K.log_2());

        let r_address = verifier_state
            .r_address
            .as_ref()
            .expect("r_address not set");
        let r_cycle = verifier_state.r_cycle.as_ref().expect("r_cycle not set");

        let eq_eval_address = EqPolynomial::mle(r_address, r_address_prime);
        let eq_eval_cycle = EqPolynomial::mle(r_cycle, r_cycle_prime);

        eq_eval_address * eq_eval_cycle * (ra_claim_prime.square() - ra_claim_prime)
    }
}

impl<F, PCS> CacheSumcheckOpenings<F, PCS> for BooleanitySumcheck<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn cache_openings_prover(
        &mut self,
        _accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        debug_assert!(self.ra_claim_prime.is_none());
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        self.ra_claim_prime = Some(prover_state.H.final_sumcheck_claim());
    }

    fn cache_openings_verifier(
        &mut self,
        _accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F, PCS>>>>,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        todo!()
    }
}

impl<F: JoltField> BooleanitySumcheck<F> {
    fn compute_phase1_message(&self, round: usize) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();
        let m = round + 1;
        const DEGREE: usize = 3;

        let univariate_poly_evals: [F; 3] = (0..prover_state.B.len() / 2)
            .into_par_iter()
            .map(|k_prime| {
                // Get B evaluations at points 0, 2, 3
                let B_evals = prover_state
                    .B
                    .sumcheck_evals_array::<DEGREE>(k_prime, BindingOrder::LowToHigh);

                let inner_sum = prover_state.G[k_prime << m..(k_prime + 1) << m]
                    .par_iter()
                    .enumerate()
                    .map(|(k, &G_k)| {
                        // Since we're binding variables from low to high, k_m is the high bit
                        let k_m = k >> (m - 1);
                        // We then index into F using (k_{m-1}, ..., k_1)
                        let F_k = prover_state.F[k % (1 << (m - 1))];
                        // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                        let G_times_F = G_k * F_k;

                        // For c \in {0, 2, 3} compute:
                        //    G[k] * (F[k_1, ...., k_{m-1}, c]^2 - F[k_1, ...., k_{m-1}, c])
                        //    = G_times_F * (eq(k_m, c)^2 * F[k_1, ...., k_{m-1}] - eq(k_m, c))
                        [
                            G_times_F
                                * (prover_state.eq_km_c_squared[k_m][0] * F_k
                                    - prover_state.eq_km_c[k_m][0]),
                            G_times_F
                                * (prover_state.eq_km_c_squared[k_m][1] * F_k
                                    - prover_state.eq_km_c[k_m][1]),
                            G_times_F
                                * (prover_state.eq_km_c_squared[k_m][2] * F_k
                                    - prover_state.eq_km_c[k_m][2]),
                        ]
                    })
                    .reduce(
                        || [F::zero(); 3],
                        |running, new| {
                            [
                                running[0] + new[0],
                                running[1] + new[1],
                                running[2] + new[2],
                            ]
                        },
                    );

                [
                    B_evals[0] * inner_sum[0],
                    B_evals[1] * inner_sum[1],
                    B_evals[2] * inner_sum[2],
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |mut running, new| {
                    for i in 0..3 {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals.to_vec()
    }

    fn compute_phase2_message(&self, _round: usize) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();
        const DEGREE: usize = 3;

        let mut univariate_poly_evals: [F; 3] = (0..prover_state.D.len() / 2)
            .into_par_iter()
            .map(|i| {
                // Get D and H evaluations at points 0, 2, 3
                let D_evals = prover_state
                    .D
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let H_evals = prover_state
                    .H
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                [
                    D_evals[0] * (H_evals[0].square() - H_evals[0]),
                    D_evals[1] * (H_evals[1].square() - H_evals[1]),
                    D_evals[2] * (H_evals[2].square() - H_evals[2]),
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |mut running, new| {
                    for i in 0..3 {
                        running[i] += new[i];
                    }
                    running
                },
            );

        // Multiply by eq_r_r
        for eval in &mut univariate_poly_evals {
            *eval *= prover_state.eq_r_r;
        }

        univariate_poly_evals.to_vec()
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct BooleanityProof<F: JoltField, ProofTranscript: Transcript> {
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub ra_claim_prime: F,
}

impl<F: JoltField, ProofTranscript: Transcript> BooleanityProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "BooleanityProof::prove")]
    pub fn prove(
        preprocessing: &BytecodePreprocessing,
        trace: &[RV32IMCycle],
        r: &[F],
        D: Vec<F>,
        G: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>) {
        let K = r.len().pow2();
        let T = trace.len();

        let mut booleanity_sumcheck = BooleanitySumcheck::new(preprocessing, trace, r, D, G, K, T);

        let (sumcheck_proof, r_combined) = booleanity_sumcheck.prove_single(transcript);

        let (r_address_prime, r_cycle_prime) = r_combined.split_at(K.log_2());

        let ra_claim_prime = booleanity_sumcheck
            .ra_claim_prime
            .expect("ra_claim_prime should be set after prove_single");

        let proof = Self {
            sumcheck_proof,
            ra_claim_prime,
        };

        (proof, r_address_prime.to_vec(), r_cycle_prime.to_vec())
    }

    pub fn verify(
        &self,
        r_address: &[F],
        r_cycle: &[F],
        K: usize,
        T: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(Vec<F>, F), ProofVerifyError> {
        let booleanity_sumcheck = BooleanitySumcheck::new_verifier(
            K,
            T,
            r_address.to_vec(),
            r_cycle.to_vec(),
            self.ra_claim_prime,
        );

        let r_combined = booleanity_sumcheck.verify_single(&self.sumcheck_proof, transcript)?;

        Ok((r_combined, self.ra_claim_prime))
    }
}
