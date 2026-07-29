//! The Akita (lattice) prove path — port scaffolding; `crate::dory` is the
//! elliptic-curve sibling.
//!
//! The modular pipeline this module will grow into mirrors
//! `jolt-prover-legacy`'s `zkvm::packed` with the lattice stage swaps: one
//! native Akita commitment group `OneHotTrace` replaces the per-polynomial
//! streaming Dory commits at stage 0, the nine-stage bytecode read-raf
//! discharges the reduced inc claims through its fused-inc val stages, the
//! lattice booleanity carries the fused-inc columns, stage 7 folds the
//! increment one-hot claims into the hamming-weight claim reduction, the
//! reconstruction phase settles the auxiliary advice/bytecode/image columns
//! at the head of the stage-8 region, and stage 8 uses one native same-point
//! Akita opening for `OneHotTrace` plus packed openings for auxiliaries.
//!
//! Everything here stays generic over the scheme; the concrete Akita types
//! (`jolt_akita::{AkitaField, AkitaScheme}`, the legacy-hosted
//! `AkitaVc`/`AkitaTranscript` aliases) bind at the call site. Beyond these
//! seams the port still owes:
//! - stage-0 `OneHotTrace` assembly and commit off the witness plane, with
//!   setup params derived from [`ProverConfig`] + the program shape (today's
//!   test scaffolding sizes the setup through the legacy prover's
//!   `one_hot_trace_setup_params`);
//! - the advice / program one-hot commitment-object constructors (legacy:
//!   `commit_trusted_advice_one_hot`, `commit_program_one_hot`) and a packed
//!   committed-program prover-data shape carrying their opening material;
//! - [`JoltAkitaBackend`]'s slot registry and its kernel implementations.

use std::marker::PhantomData;

use common::jolt_device::JoltDevice;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_kernels::ProofSession;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;
use jolt_verifier::proof::JoltProof;
use jolt_witness::JoltWitnessPlane;

use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

/// The packed slot registry: the akita analog of [`JoltBackend`]. A parallel
/// struct rather than cfg-gated [`JoltBackend`] fields — `jolt-kernels`
/// deliberately has no `akita` feature (a local `cfg!` there would silently
/// read `false` and desynchronize the prover from the verifier; see
/// `jolt_claims`'s `CANONICAL_INSTRUCTION_ADDRESS`), so the packed-only
/// registry lives on this crate's akita-only side of the fence.
///
/// The packed PIOP shares most sumcheck members with the base protocol, so
/// most slots will mirror [`JoltBackend`]'s — one `Box<dyn PrepareKernel<F,
/// R>>` per member with `#[derive(KernelSlots)]` emitting the delegating
/// impls the stage drivers resolve through. The swaps are the commit seam
/// (one native `OneHotTrace` group commit instead of streaming
/// per-polynomial commits — the reason [`JoltBackend::reference`]'s
/// `StreamingCommitment` bound can never hold for the packed scheme), the
/// lattice members (fused-inc bytecode read-raf, lattice booleanity), the
/// reconstruction-phase members, and the native same-point opening. The slot
/// set is finalized by the port; until then the registry is empty and
/// [`prove`] only pins the seam.
///
/// [`JoltBackend`]: crate::JoltBackend
/// [`JoltBackend::reference`]: crate::JoltBackend::reference
pub struct JoltAkitaBackend<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    _marker: PhantomData<fn() -> (F, PCS)>,
}

impl<F, PCS> JoltAkitaBackend<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    /// The always-present packed reference registry: every slot naive-served
    /// once slots land with the port.
    pub fn reference() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Open the proof-scoped session that slot state lives in — the same
    /// contract as [`JoltBackend::begin_proof`](crate::JoltBackend::begin_proof).
    pub fn begin_proof(&self) -> ProofSession {
        ProofSession::default()
    }
}

/// Prove one execution over the packed (Akita) protocol: the analog of
/// `dory::prove`, emitting the packed-envelope [`JoltProof`] (single
/// `OneHotTrace` commitment, reconstruction claims, native same-point joint
/// opening).
///
/// `trusted_advice` and `program_one_hot` are the precommitted auxiliary
/// objects' commitments, passed exactly when the guest consumes trusted
/// advice / the preprocessing is committed-program (the prover-retained
/// opening material for both will ride a packed prover-data shape once the
/// port defines it). Untrusted advice needs no input — its one-hot column is
/// committed at prove time from the witness when `public_io` carries advice
/// bytes.
pub fn prove<F, PCS, VC, T, W>(
    _backend: &JoltAkitaBackend<F, PCS>,
    _preprocessing: &JoltProverPreprocessing<PCS, VC>,
    _config: &ProverConfig,
    _trusted_advice: Option<&PCS::Output>,
    _program_one_hot: Option<&PCS::Output>,
    _witness: &W,
    _public_io: &JoltDevice,
) -> Result<JoltProof<PCS, VC>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
    W: JoltWitnessPlane<F>,
{
    Err(ProverError::Unsupported {
        reason: "the packed (Akita) prove path is not yet ported to the modular prover",
    })
}
