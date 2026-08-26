//! The Akita (lattice) prove path — `crate::dory` is the elliptic-curve
//! sibling.
//!
//! The pipeline mirrors `jolt-prover-legacy`'s `zkvm::packed` with the
//! lattice stage swaps: one prefix-packed `OneHotTrace` polynomial (with the
//! virtualized families' digit-zero rows omitted) replaces the
//! per-polynomial streaming Dory commits at stage 0, the nine-stage bytecode
//! read-raf discharges the reduced inc claims through its fused-inc val
//! stages, the lattice booleanity carries the balanced-inc columns, stage 7
//! runs the digit-zero claim reduction (recentered legs plus the balanced
//! decode), the reconstruction phase settles the auxiliary
//! advice/bytecode/image columns at the head of the stage-8 region, and
//! stage 8 reduces the `OneHotTrace` columns to one native Akita opening
//! plus one packed opening per auxiliary object.
//!
//! Everything here stays generic over the scheme through the `jolt-openings`
//! seams ([`commit_batch`](CommitmentScheme::commit_batch)/
//! [`open_batch_from_hint`](CommitmentScheme::open_batch_from_hint) for the packed polynomial,
//! [`TransparentObjectSetup`] for the auxiliary objects); the concrete Akita
//! types bind at the call site.

use common::jolt_device::JoltDevice;
use jolt_akita::TraceOneHotCommitment;
use jolt_crypto::VectorCommitment;
use jolt_field::{CanonicalBytes, JoltField};
use jolt_kernels::{JoltBackend, KernelSlots, PrepareKernel, ProofSession, ReferenceBackend};
use jolt_openings::{CommitmentScheme, GroupSetupMetadata, TransparentObjectSetup};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::proof::JoltProof;
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage8::reconstruction::FieldIncLimbReconstructionInstance;
use jolt_verifier::stages::stage8::reconstruction::{
    BytecodeChunkReconstructionInstance, ProgramImageReconstructionInstance,
    TrustedAdviceReconstructionInstance, UntrustedAdviceReconstructionInstance,
};
use jolt_witness::{JoltWitnessPlane, RowSource};

use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

/// The packed FR seam: limb extraction, the stage-0 limb commit, the FR
/// reconstruction kernel, and the stage-8 limb opening.
#[cfg(feature = "field-inline")]
mod field_inline_packed;
mod prover;
mod setup;
pub use setup::one_hot_trace_setup_shape;
mod reconstruction;
mod stage0;
mod stage8;
pub mod witness;

/// The packed slot registry: the akita analog of a bare [`JoltBackend`]. A
/// parallel struct rather than cfg-gated [`JoltBackend`] fields —
/// `jolt-kernels` deliberately has no `akita` feature (a local `cfg!` there
/// would silently read `false` and desynchronize the prover from the
/// verifier; see `jolt_claims`'s `CANONICAL_INSTRUCTION_ADDRESS`), so the
/// packed-only pieces live on this crate's akita-only side of the fence.
///
/// The packed PIOP shares its stage 1–7 members with the base protocol, so
/// they resolve through the embedded [`JoltBackend`] registry (whose commit
/// slot is an unreachable stub: the packed path commits one native
/// `OneHotTrace` group in its own stage 0, never through the streaming
/// commit seam). The reconstruction-phase members resolve through their own
/// replaceable slots, exactly like the shared registry's — an optimized
/// packed backend swaps the boxes, never the type.
#[derive(KernelSlots)]
pub struct JoltAkitaBackend<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    /// The shared stage 1–7 slot registry (naive-served).
    pub base: JoltBackend<F, PCS>,
    pub untrusted_advice_reconstruction:
        Box<dyn PrepareKernel<F, UntrustedAdviceReconstructionInstance<F>>>,
    pub trusted_advice_reconstruction:
        Box<dyn PrepareKernel<F, TrustedAdviceReconstructionInstance<F>>>,
    pub bytecode_reconstruction: Box<dyn PrepareKernel<F, BytecodeChunkReconstructionInstance<F>>>,
    pub program_image_reconstruction:
        Box<dyn PrepareKernel<F, ProgramImageReconstructionInstance<F>>>,
    #[cfg(feature = "field-inline")]
    pub field_inc_limb_reconstruction:
        Box<dyn PrepareKernel<F, FieldIncLimbReconstructionInstance<F>>>,
}

/// The packed path's stand-in for the streaming witness-commit slot: stage 0
/// commits the native `OneHotTrace` group directly, so this slot is never
/// reached.
struct PackedCommitStub;

impl<F, PCS> jolt_kernels::CommitWitness<F, PCS> for PackedCommitStub
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn commit_witness(
        &self,
        _session: &mut ProofSession,
        _source: &dyn RowSource,
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

    #[cfg(feature = "field-inline")]
    fn commit_field_inline_witness(
        &self,
        _session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        _ids: &[jolt_claims::protocols::field_inline::FieldInlineCommittedPolynomial],
        _grid: jolt_kernels::CommitmentGrid,
        _setup: &PCS::ProverSetup,
    ) -> Result<Vec<jolt_kernels::FieldInlineWitnessCommitment<PCS>>, jolt_kernels::KernelError<F>>
    {
        Err(jolt_kernels::KernelError::Unsupported {
            reason: "the packed (Akita) path commits the FR limb object in stage 0; the \
                     streaming field-inline commit slot is unreachable",
        })
    }
}

impl<F, PCS> JoltAkitaBackend<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    /// The always-present packed reference registry: every shared stage 1–7
    /// slot naive-served (the reference kernels adapt to the packed
    /// jolt-claims shape at runtime), the commit slot stubbed out (the packed
    /// commit lives in stage 0), and every reconstruction slot served by the
    /// reference reconstruction kernels.
    pub fn reference() -> Self {
        Self {
            untrusted_advice_reconstruction: Box::new(
                reconstruction::ReferenceReconstruction,
            ),
            trusted_advice_reconstruction: Box::new(reconstruction::ReferenceReconstruction),
            bytecode_reconstruction: Box::new(reconstruction::ReferenceReconstruction),
            program_image_reconstruction: Box::new(reconstruction::ReferenceReconstruction),
            #[cfg(feature = "field-inline")]
            field_inc_limb_reconstruction: Box::new(reconstruction::ReferenceReconstruction),
            base: JoltBackend {
                commit: Box::new(PackedCommitStub),
                round_scheduler: Box::new(ReferenceBackend),
                spartan_outer_uniskip: Box::new(ReferenceBackend),
                spartan_outer_remainder: Box::new(jolt_kernels::reference::spartan_outer::ReferenceOuterRemainder),
                spartan_product_uniskip: Box::new(ReferenceBackend),
                spartan_product_remainder: Box::new(jolt_kernels::reference::spartan_product::ReferenceProductRemainder),
                ram_read_write: Box::new(ReferenceBackend),
                instruction_claim_reduction: Box::new(ReferenceBackend),
                #[cfg(feature = "field-inline")]
                field_registers_claim_reduction: Box::new(ReferenceBackend),
                ram_raf_evaluation: Box::new(ReferenceBackend),
                ram_output_check: Box::new(ReferenceBackend),
                spartan_shift: Box::new(ReferenceBackend),
                instruction_input: Box::new(ReferenceBackend),
                registers_claim_reduction: Box::new(ReferenceBackend),
                registers_read_write: Box::new(ReferenceBackend),
                #[cfg(feature = "field-inline")]
                field_registers_read_write: Box::new(ReferenceBackend),
                ram_val_check: Box::new(ReferenceBackend),
                advice_opening: Box::new(ReferenceBackend),
                instruction_read_raf: Box::new(ReferenceBackend),
                ram_ra_claim_reduction: Box::new(ReferenceBackend),
                registers_val_evaluation: Box::new(ReferenceBackend),
                #[cfg(feature = "field-inline")]
                field_registers_val_evaluation: Box::new(ReferenceBackend),
                bytecode_read_raf_address: Box::new(ReferenceBackend),
                booleanity_address: Box::new(ReferenceBackend),
                bytecode_read_raf_cycle: Box::new(ReferenceBackend),
                booleanity_cycle: Box::new(ReferenceBackend),
                ram_hamming_booleanity: Box::new(ReferenceBackend),
                ram_ra_virtualization: Box::new(ReferenceBackend),
                instruction_ra_virtualization: Box::new(ReferenceBackend),
                inc_claim_reduction: Box::new(ReferenceBackend),
                #[cfg(feature = "field-inline")]
                field_registers_inc_claim_reduction: Box::new(ReferenceBackend),
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
}

/// Prove one execution over the packed (Akita) protocol: the analog of
/// `dory::prove`, emitting the packed-envelope [`JoltProof`] (single
/// `OneHotTrace` commitment, reconstruction claims, native same-point joint
/// opening).
///
/// `trusted_advice` is the precommitted trusted-advice object
/// ([`witness::commit_advice_one_hot`], built out of band like legacy's
/// `commit_trusted_advice_one_hot`), passed exactly when the guest consumes
/// trusted advice. The precommitted `ProgramOneHot` objects ride the
/// preprocessing ([`crate::CommittedProgramProverData::program_one_hot`]);
/// stage 0 cross-checks their commitments against the verifier
/// preprocessing fail-closed. Untrusted advice needs no input — its one-hot
/// column is committed at prove time from the public advice bytes.
pub fn prove<F, PCS, VC, T, W>(
    backend: &JoltAkitaBackend<F, PCS>,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&witness::AdviceOneHot<PCS>>,
    witness: &W,
    public_io: &JoltDevice,
) -> Result<JoltProof<PCS, VC>, ProverError<F>>
where
    F: JoltField + CanonicalBytes + AppendToTranscript,
    PCS: CommitmentScheme<Field = F> + TransparentObjectSetup + TraceOneHotCommitment,
    PCS::ProverSetup: GroupSetupMetadata,
    PCS::Output: Clone + PartialEq + AppendToTranscript,
    VC: VectorCommitment<Field = F>,
    VC::Output: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
    W: JoltWitnessPlane<F>,
{
    prover::prove::<F, PCS, VC, T, W>(
        backend,
        preprocessing,
        config,
        trusted_advice,
        witness,
        public_io,
    )
}
