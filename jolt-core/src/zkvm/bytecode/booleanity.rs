use crate::poly::opening_proof::{OpeningAccumulator, SumcheckId};
use crate::subprotocols::booleanity::{
    BooleanityConfig, BooleanityProverState, BooleanitySumcheck,
};
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme, multilinear_polynomial::BindingOrder,
        split_eq_poly::GruenSplitEqPolynomial,
    },
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
};
use allocative::Allocative;
use rayon::prelude::*;

// Bytecode booleanity sumcheck
//
// Proves a zero-check of the form
//   0 = Σ_k Σ_j eq(r_address, k) · eq(r_cycle, j) · (Σ_{i=0}^{d-1} γ_i · (H_i(k, j)^2 − H_i(k, j)))
// where:
// - r_address are the address-chunk variables bound in phase 1
// - r_cycle are the time/cycle variables bound in phase 2
// - H_i is the routing/selection indicator for the i-th address chunk (boolean per point)
#[derive(Allocative)]
pub struct BytecodeBooleanitySumcheck<F: JoltField> {
    /// gamma: optimized batching challenges γ_i (length d).
    gamma: Vec<F::Challenge>,
    /// d: number of address chunks in the decomposition.
    d: usize,
    /// log_T: number of time/cycle variables.
    log_T: usize,
    /// log_K_chunk: number of address-chunk variables per chunk.
    log_K_chunk: usize,
    /// prover_state: prover-side working state for both phases.
    prover_state: Option<BooleanityProverState<F>>,
    /// r_address: address binding point (for endianness and output claim).
    r_address: Vec<F::Challenge>,
}

impl<F: JoltField> BytecodeBooleanitySumcheck<F> {
    #[tracing::instrument(skip_all, name = "BytecodeBooleanitySumcheck::new_prover")]
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        r_cycle: Vec<F::Challenge>,
        G: Vec<Vec<F>>,
    ) -> Self {
        let (preprocessing, trace, _, _) = sm.get_prover_data();
        let d = preprocessing.shared.bytecode.d;
        let log_K = preprocessing.shared.bytecode.bytecode.len().log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let gamma = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(d);

        let r_address: Vec<F::Challenge> = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(log_K_chunk);

        // Build the H_indices for each chunk
        let pc_by_cycle: Vec<Vec<Option<u8>>> = (0..d)
            .into_par_iter()
            .map(|i| {
                trace
                    .par_iter()
                    .map(|cycle| {
                        let k = preprocessing.shared.bytecode.get_pc(cycle);
                        Some(((k >> (log_K_chunk * (d - i - 1))) % (1 << log_K_chunk)) as u8)
                    })
                    .collect()
            })
            .collect();

        // Create prover state
        let B = GruenSplitEqPolynomial::new(&r_address, BindingOrder::LowToHigh);
        let D = GruenSplitEqPolynomial::new(&r_cycle, BindingOrder::LowToHigh);
        let mut F: Vec<F> = unsafe_allocate_zero_vec(log_K.pow2());
        F[0] = F::one();

        let prover_state = BooleanityProverState {
            B,
            D,
            H: vec![], // Will be initialized during bind
            G,
            F,
            eq_r_r: F::zero(),
            H_indices: pc_by_cycle,
        };

        Self {
            gamma,
            prover_state: Some(prover_state),
            d,
            log_T: trace.len().log_2(),
            log_K_chunk,
            r_address,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let (_, _, T) = sm.get_verifier_data();
        let d = sm.get_verifier_data().0.shared.bytecode.d;
        let log_K = sm.get_bytecode().len().log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let gamma = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(d);
        let r_address: Vec<F::Challenge> = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(log_K_chunk);
        Self {
            gamma,
            prover_state: None,
            log_T: T.log_2(),
            r_address,
            log_K_chunk,
            d,
        }
    }
}

impl<F: JoltField> BooleanityConfig for BytecodeBooleanitySumcheck<F> {
    fn d(&self) -> usize {
        self.d
    }

    fn log_k_chunk(&self) -> usize {
        self.log_K_chunk
    }

    fn log_t(&self) -> usize {
        self.log_T
    }

    fn polynomial_type(i: usize) -> CommittedPolynomial {
        CommittedPolynomial::BytecodeRa(i)
    }

    fn sumcheck_id() -> SumcheckId {
        SumcheckId::BytecodeBooleanity
    }
}

impl<F: JoltField, T: Transcript> BooleanitySumcheck<F, T> for BytecodeBooleanitySumcheck<F> {
    fn gamma(&self) -> &[F::Challenge] {
        &self.gamma
    }

    fn r_address(&self) -> &[F::Challenge] {
        &self.r_address
    }

    fn prover_state(&self) -> Option<&BooleanityProverState<F>> {
        self.prover_state.as_ref()
    }

    fn prover_state_mut(&mut self) -> Option<&mut BooleanityProverState<F>> {
        self.prover_state.as_mut()
    }

    fn get_r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> Vec<F::Challenge> {
        accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .clone()
    }
}
