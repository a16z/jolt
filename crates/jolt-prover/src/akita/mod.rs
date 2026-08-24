//! The Akita (lattice) prove path — `crate::dory` is the elliptic-curve
//! sibling.
//!
//! The pipeline mirrors `jolt-prover-legacy`'s `zkvm::packed` with the
//! lattice stage swaps: one native Akita commitment group `OneHotTrace`
//! replaces the per-polynomial streaming Dory commits at stage 0, the
//! nine-stage bytecode read-raf discharges the reduced inc claims through its
//! fused-inc val stages, the lattice booleanity carries the fused-inc
//! columns, stage 7 folds the increment one-hot claims into the
//! hamming-weight claim reduction, the reconstruction phase settles the
//! auxiliary advice/bytecode/image columns at the head of the stage-8 region,
//! and stage 8 uses one native same-point Akita opening for `OneHotTrace`
//! plus packed openings for auxiliaries.
//!
//! Everything here stays generic over the scheme through the `jolt-openings`
//! seams ([`commit_batch`](CommitmentScheme::commit_batch)/
//! [`open_batch`](CommitmentScheme::open_batch) for the native group,
//! [`TransparentObjectSetup`] for the auxiliary objects); the concrete Akita
//! types bind at the call site.

use common::jolt_device::JoltDevice;
use jolt_crypto::VectorCommitment;
use jolt_field::{CanonicalBytes, Field};
use jolt_kernels::{JoltBackend, ProofSession, ReferenceBackend};
use jolt_openings::{CommitmentScheme, GroupSetupMetadata, TransparentObjectSetup};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::proof::JoltProof;
use jolt_witness::JoltWitnessPlane;

use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

mod prover;
mod reconstruction;
mod stage0;
mod stage8;
pub mod witness;

/// The packed slot registry: the akita analog of a bare [`JoltBackend`]. A
/// parallel struct rather than cfg-gated [`JoltBackend`] fields. The
/// packed-only commitment and reconstruction pieces live on this crate's
/// Akita side of the fence; shared sumcheck slots remain in `jolt-kernels`.
///
/// The packed PIOP shares its stage 1–7 members with the base protocol, so
/// they resolve through the embedded [`JoltBackend`] registry (whose commit
/// slot is an unreachable stub: the packed path commits one native
/// `OneHotTrace` group in its own stage 0, never through the streaming
/// commit seam). The reconstruction-phase members' kernels are implemented
/// directly on this type (`reconstruction.rs`).
pub struct JoltAkitaBackend<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    /// The shared stage 1–7 slot registry (naive-served).
    pub base: JoltBackend<F, PCS>,
    trace_commitment: jolt_akita::TraceCommitmentBackend,
}

/// The packed path's stand-in for the streaming witness-commit slot: stage 0
/// commits the native `OneHotTrace` group directly, so this slot is never
/// reached.
struct PackedCommitStub;

impl<F, PCS> jolt_kernels::CommitWitness<F, PCS> for PackedCommitStub
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    fn commit_witness(
        &self,
        _session: &mut ProofSession,
        _source: &dyn jolt_witness::JoltWitnessPlane<F>,
        _ids: &[jolt_claims::protocols::jolt::JoltCommittedPolynomial],
        _grid: jolt_kernels::CommitmentGrid,
        _setup: &PCS::ProverSetup,
    ) -> Result<Vec<jolt_kernels::WitnessCommitment<PCS>>, jolt_kernels::KernelError<F>> {
        Err(jolt_kernels::KernelError::Unsupported {
            reason: "the packed (Akita) path commits one native OneHotTrace group in stage 0; \
                     the streaming witness-commit slot is unreachable",
        })
    }

    fn commit_advice(
        &self,
        _session: &mut ProofSession,
        _witness: &dyn jolt_witness::JoltWitnessOracle<F>,
        _id: jolt_claims::protocols::jolt::JoltCommittedPolynomial,
        _grid: jolt_kernels::CommitmentGrid,
        _setup: &PCS::ProverSetup,
    ) -> Result<jolt_kernels::WitnessCommitment<PCS>, jolt_kernels::KernelError<F>> {
        Err(jolt_kernels::KernelError::Unsupported {
            reason: "the packed (Akita) path commits advice byte one-hot objects in stage 0; \
                     the streaming advice-commit slot is unreachable",
        })
    }
}

impl<F, PCS> JoltAkitaBackend<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    /// The always-present packed reference registry: every shared stage 1–7
    /// slot naive-served (the reference kernels adapt to the packed
    /// jolt-claims shape at runtime), the commit slot stubbed out (the packed
    /// commit lives in stage 0), and the reconstruction members implemented
    /// on this type directly.
    pub fn reference() -> Self {
        Self {
            base: JoltBackend {
                commit: Box::new(PackedCommitStub),
                round_scheduler: Box::new(ReferenceBackend),
                spartan_outer_uniskip: Box::new(ReferenceBackend),
                spartan_outer_remainder: Box::new(jolt_kernels::reference::spartan_outer::ReferenceOuterRemainder),
                spartan_product_uniskip: Box::new(ReferenceBackend),
                spartan_product_remainder: Box::new(jolt_kernels::reference::spartan_product::ReferenceProductRemainder),
                ram_read_write: Box::new(ReferenceBackend),
                instruction_claim_reduction: Box::new(ReferenceBackend),
                ram_raf_evaluation: Box::new(ReferenceBackend),
                ram_output_check: Box::new(ReferenceBackend),
                spartan_shift: Box::new(ReferenceBackend),
                instruction_input: Box::new(ReferenceBackend),
                registers_claim_reduction: Box::new(ReferenceBackend),
                registers_read_write: Box::new(ReferenceBackend),
                ram_val_check: Box::new(ReferenceBackend),
                advice_opening: Box::new(ReferenceBackend),
                instruction_read_raf: Box::new(ReferenceBackend),
                ram_ra_claim_reduction: Box::new(ReferenceBackend),
                registers_val_evaluation: Box::new(ReferenceBackend),
                bytecode_read_raf_address: Box::new(ReferenceBackend),
                booleanity_address: Box::new(ReferenceBackend),
                bytecode_read_raf_cycle: Box::new(ReferenceBackend),
                booleanity_cycle: Box::new(ReferenceBackend),
                ram_hamming_booleanity: Box::new(ReferenceBackend),
                ram_ra_virtualization: Box::new(ReferenceBackend),
                instruction_ra_virtualization: Box::new(ReferenceBackend),
                inc_claim_reduction: Box::new(ReferenceBackend),
                trusted_advice_cycle: Box::new(ReferenceBackend),
                untrusted_advice_cycle: Box::new(ReferenceBackend),
                bytecode_reduction_cycle: Box::new(ReferenceBackend),
                program_image_reduction_cycle: Box::new(ReferenceBackend),
                hamming_weight_claim_reduction: Box::new(ReferenceBackend),
                trusted_advice_address: Box::new(
                    jolt_kernels::reference::precommitted_reduction::ReferencePrecommittedAddress::new(
                        "stage 6b parked no trusted-advice reduction state for the scheduled address phase",
                    ),
                ),
                untrusted_advice_address: Box::new(
                    jolt_kernels::reference::precommitted_reduction::ReferencePrecommittedAddress::new(
                        "stage 6b parked no untrusted-advice reduction state for the scheduled address phase",
                    ),
                ),
                bytecode_reduction_address: Box::new(
                    jolt_kernels::reference::precommitted_reduction::ReferencePrecommittedAddress::new(
                        "stage 6b parked no bytecode reduction state for the scheduled address phase",
                    ),
                ),
                program_image_reduction_address: Box::new(
                    jolt_kernels::reference::precommitted_reduction::ReferencePrecommittedAddress::new(
                        "stage 6b parked no program-image reduction state for the scheduled address phase",
                    ),
                ),
                joint_opening: Box::new(ReferenceBackend),
            },
            trace_commitment: jolt_akita::TraceCommitmentBackend::cpu(),
        }
    }

    /// The packed backend with optimized stage 1–7 arithmetic and native
    /// Akita commitment/opening boundaries.
    pub fn optimized() -> Self {
        let mut backend = Self::reference();
        backend.base = backend.base.with_optimized_compute();
        backend
    }

    /// Open the proof-scoped session that slot state lives in — the same
    /// contract as [`JoltBackend::begin_proof`].
    pub fn begin_proof(&self) -> ProofSession {
        ProofSession::default()
    }

    pub(crate) fn trace_commitment_backend(&self) -> &jolt_akita::TraceCommitmentBackend {
        &self.trace_commitment
    }

    pub fn prepare_trace_commitment(
        &self,
        setup: &PCS::ProverSetup,
        column_capacity: usize,
        num_columns: usize,
        num_rows: usize,
    ) -> Result<(), jolt_openings::OpeningsError>
    where
        PCS: jolt_akita::TraceOneHotCommitment,
    {
        PCS::prepare_trace_one_hot_backend(
            &self.trace_commitment,
            setup,
            column_capacity,
            num_columns,
            num_rows,
        )
    }

    pub fn last_trace_commitment_metrics(
        &self,
    ) -> Result<Option<jolt_akita::TraceMetalCommitMetrics>, jolt_openings::OpeningsError> {
        self.trace_commitment.last_metal_commit_metrics()
    }

    pub fn last_trace_opening_metrics(
        &self,
    ) -> Result<Option<jolt_akita::TraceMetalOpeningMetrics>, jolt_openings::OpeningsError> {
        self.trace_commitment.last_metal_opening_metrics()
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[derive(Debug, thiserror::Error)]
pub enum JoltAkitaMetalError {
    #[error("Jolt PIOP Metal backend initialization failed: {0}")]
    Piop(#[from] jolt_kernels::metal::solinas::MetalError),
    #[error("Akita commitment Metal backend initialization failed: {0}")]
    Commitment(#[from] jolt_akita::TraceCommitmentMetalError),
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl<PCS> JoltAkitaBackend<jolt_field::AkitaField, PCS>
where
    PCS: CommitmentScheme<Field = jolt_field::AkitaField>,
{
    /// Installs a caller-configured PIOP Metal backend together with the
    /// required Metal trace-commitment route.
    pub fn with_metal_compute(
        mut self,
        metal: &jolt_kernels::metal::MetalBackend,
    ) -> Result<Self, jolt_akita::TraceCommitmentMetalError> {
        self.base = self.base.with_metal_compute(metal);
        self.trace_commitment = jolt_akita::TraceCommitmentBackend::metal_required()?;
        Ok(self)
    }

    /// Builds the optimized Akita backend and replaces available slots with
    /// their hybrid Metal implementations.
    pub fn metal() -> Result<Self, JoltAkitaMetalError> {
        let metal = jolt_kernels::metal::MetalBackend::production()?;
        Ok(Self::optimized().with_metal_compute(&metal)?)
    }
}

/// Prove one execution over the packed (Akita) protocol: the analog of
/// `dory::prove`, emitting the packed-envelope [`JoltProof`] (single
/// `OneHotTrace` commitment, reconstruction claims, native same-point joint
/// opening).
///
/// `trusted_advice` and `program_one_hot` are the precommitted auxiliary
/// objects' commitments, passed exactly when the guest consumes trusted
/// advice / the preprocessing is committed-program. The objects' opening
/// material is transparently re-derived at prove time (the byte columns from
/// the public advice bytes / the retained full program, the setups from the
/// public shapes with the fixed seed) and cross-checked against the passed
/// commitments. Untrusted advice needs no input — its one-hot column is
/// committed at prove time from the public advice bytes.
pub fn prove<F, PCS, VC, T, W>(
    backend: &JoltAkitaBackend<F, PCS>,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&PCS::Output>,
    program_one_hot: Option<&[PCS::Output]>,
    witness: &W,
    public_io: &JoltDevice,
) -> Result<JoltProof<PCS, VC>, ProverError<F>>
where
    F: Field + CanonicalBytes + AppendToTranscript,
    PCS: CommitmentScheme<Field = F>
        + TransparentObjectSetup
        + jolt_akita::PostCommitmentCleanup
        + jolt_akita::TraceOneHotCommitment,
    PCS::ProverSetup: GroupSetupMetadata,
    PCS::Output: Clone + PartialEq + AppendToTranscript,
    VC: VectorCommitment<Field = F>,
    VC::Output: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
    W: JoltWitnessPlane<F> + Sync,
{
    prover::prove::<F, PCS, VC, T, W>(
        backend,
        preprocessing,
        config,
        trusted_advice,
        program_one_hot,
        witness,
        public_io,
    )
}
