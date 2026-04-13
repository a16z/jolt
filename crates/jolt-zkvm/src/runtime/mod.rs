//! Prover runtime: execute a linked schedule to produce a proof.
//!
//! [`execute`] walks the complete [`Op`] sequence — compute, PCS, and
//! orchestration ops — and returns a [`JoltProof`]. It dispatches to:
//! - [`ComputeBackend`] for polynomial arithmetic (sumcheck, bind, evaluate)
//! - [`CommitmentScheme`] for cryptographic ops (commit, reduce, open)
//! - Direct calls for orchestration (transcript absorb/squeeze, lifecycle)
//!
//! ```text
//! Protocol → compile() → Module → link(backend) → Executable<B,F>
//!                                                       │
//!                                          execute(exe, provider, backend, pcs, transcript)
//!                                                       │
//!                                                       ▼
//!                                                 JoltProof<F, PCS>
//! ```

mod handlers;
mod helpers;

use std::collections::{HashMap, HashSet};

use jolt_compiler::PolynomialId;
use jolt_compute::{Buf, BufferProvider, ComputeBackend, Executable, LookupTraceData};
use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;
use jolt_openings::{AdditivelyHomomorphic, ProverClaim};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::proof::SumcheckProof;
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::proof::{JoltProof, StageProof};
use jolt_verifier::ProverConfig;

use helpers::PendingClaim;

/// Per-stage proof being incrementally built.
pub(super) struct StageBuilder<F: Field> {
    pub(super) round_polys: Vec<UnivariatePoly<F>>,
    pub(super) evals: Vec<F>,
}

impl<F: Field> StageBuilder<F> {
    pub(super) fn new() -> Self {
        Self {
            round_polys: Vec::new(),
            evals: Vec::new(),
        }
    }

    pub(super) fn finalize(self) -> StageProof<F> {
        StageProof {
            round_polys: SumcheckProof {
                round_polynomials: self.round_polys,
            },
            evals: self.evals,
        }
    }
}

/// Mutable state accumulated during schedule execution.
pub(super) struct RuntimeState<B: ComputeBackend, F: Field, PCS: AdditivelyHomomorphic<Field = F>>
where
    PCS::Output: HomomorphicCommitment<F>,
{
    pub(super) config: ProverConfig,
    pub(super) challenges: Vec<F>,
    pub(super) evaluations: HashMap<PolynomialId, F>,
    pub(super) last_round_coeffs: Vec<F>,
    pub(super) last_round_poly: Option<UnivariatePoly<F>>,
    pub(super) last_squeezed: F,

    pub(super) batch_instance_claims: Vec<Vec<F>>,
    pub(super) last_round_instance_evals: Vec<Vec<F>>,
    pub(super) batch_combined: Vec<F>,
    pub(super) bound_this_round: HashSet<PolynomialId>,
    pub(super) current_batch_round: usize,
    pub(super) segmented_outer_eqs: HashMap<(usize, usize), Vec<F>>,
    pub(super) instance_states: HashMap<(usize, usize), B::InstanceState<F>>,

    pub(super) current_stage: Option<StageBuilder<F>>,
    pub(super) stage_proofs: Vec<StageProof<F>>,

    pub(super) commitments: Vec<PCS::Output>,
    pub(super) hints: HashMap<PolynomialId, PCS::OpeningHint>,
    pub(super) pending_claims: Vec<PendingClaim<F>>,
    pub(super) pending_hints: Vec<PCS::OpeningHint>,
    pub(super) reduced_claims: Vec<ProverClaim<F>>,
    pub(super) reduced_hints: Vec<PCS::OpeningHint>,
    pub(super) opening_proofs: Vec<PCS::Proof>,
    pub(super) padded_poly_data: HashMap<PolynomialId, Vec<F>>,
}

/// Execute the full prover schedule and return a complete proof.
///
/// Walks every op in the schedule, dispatching compute ops to `backend`,
/// PCS ops to the commitment scheme, and orchestration ops directly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute<B, F, T, PCS>(
    executable: &Executable<B, F>,
    provider: &mut impl BufferProvider<F>,
    backend: &B,
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut T,
    config: ProverConfig,
    lookup_trace: Option<LookupTraceData>,
    bytecode_data: Option<jolt_witness::bytecode_raf::BytecodeData<F>>,
) -> JoltProof<F, PCS>
where
    B: ComputeBackend,
    F: Field,
    T: Transcript<Challenge = F>,
    PCS: AdditivelyHomomorphic<Field = F>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<F>,
{
    let module = &executable.module;

    let batch_instance_claims: Vec<Vec<F>> = module
        .prover
        .batched_sumchecks
        .iter()
        .map(|b| vec![F::zero(); b.instances.len()])
        .collect();

    let mut state = RuntimeState::<B, F, PCS> {
        config,
        challenges: vec![F::zero(); module.challenges.len()],
        evaluations: HashMap::new(),
        last_round_coeffs: Vec::new(),
        last_round_poly: None,
        last_squeezed: F::zero(),
        batch_instance_claims,
        last_round_instance_evals: Vec::new(),
        batch_combined: Vec::new(),
        bound_this_round: HashSet::new(),
        current_batch_round: 0,
        segmented_outer_eqs: HashMap::new(),
        instance_states: HashMap::new(),
        current_stage: None,
        stage_proofs: Vec::new(),
        commitments: Vec::new(),
        hints: HashMap::new(),
        pending_claims: Vec::new(),
        pending_hints: Vec::new(),
        reduced_claims: Vec::new(),
        reduced_hints: Vec::new(),
        opening_proofs: Vec::new(),
        padded_poly_data: HashMap::new(),
    };

    let stage_point_indices: Vec<Vec<usize>> = helpers::precompute_stage_points(module);

    let mut device_buffers: HashMap<PolynomialId, Buf<B, F>> = HashMap::new();

    for op in &executable.ops {
        handlers::dispatch_op(
            op,
            &mut state,
            &mut device_buffers,
            executable,
            provider,
            backend,
            pcs_setup,
            transcript,
            &stage_point_indices,
            bytecode_data.as_ref(),
            lookup_trace.as_ref(),
        );
    }

    if let Some(builder) = state.current_stage.take() {
        state.stage_proofs.push(builder.finalize());
    }

    JoltProof {
        config: state.config,
        stage_proofs: state.stage_proofs,
        opening_proofs: state.opening_proofs,
        commitments: state.commitments,
    }
}
