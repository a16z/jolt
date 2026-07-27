//! The witness-commitment slot: committed witness polynomials as PCS
//! commitments over the proof's shared embedding grid.
//!
//! Every witness polynomial is committed as a matrix in one common grid shape
//! (`2^⌈total_vars/2⌉` columns), not per-polynomial squares: the stage-8 joint
//! opening combines the commitments homomorphically, which is only meaningful
//! when they share row geometry.

use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, TracePolynomialOrder};
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_witness::witnesses::{LookupIndex, MappedPc, RamInc, RdInc, RemappedRamAddress};
use jolt_witness::{JoltWitnessOracle, RowSource, WitnessBundle};

use crate::{KernelError, ProofSession};

/// The per-cycle facts every committed column derives from — the commitment
/// consumer's bundle. The runtime-arity chunk selection (which `InstructionRa`
/// chunk of the lookup index, etc.) lives in the consumer, which owns the
/// proof config; the bundle carries only the trace-derived values.
#[derive(Clone, Copy, Debug, WitnessBundle)]
pub struct CommittedColumnsWitness {
    pub rd_inc: RdInc,
    pub ram_inc: RamInc,
    pub lookup_index: LookupIndex,
    pub bytecode_pc: MappedPc,
    pub ram_address: RemappedRamAddress,
}

/// The shared embedding grid every witness polynomial is committed in:
/// `2^⌈total_vars/2⌉` columns, where `total_vars` is the maximum over the
/// one-hot main matrix (`log_k_chunk + log_t`) and any precommitted-candidate
/// shapes (advice, committed program). `order` is the proof's
/// coefficient-placement mode; dedicated advice grids are always
/// [`TracePolynomialOrder::CycleMajor`] (their placement is contiguous in
/// both proof layouts — legacy's strides collapse outside the main context)
/// with `log_k_chunk` 0 (no one-hot polynomials).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentGrid {
    pub total_vars: usize,
    pub log_t: usize,
    /// The committed one-hot address width — the main matrix contributes
    /// `log_k_chunk + log_t` of `total_vars`; the rest is the
    /// precommitted-candidate embedding extra.
    pub log_k_chunk: usize,
    pub order: TracePolynomialOrder,
}

impl CommitmentGrid {
    pub const fn num_columns(&self) -> usize {
        1 << self.total_vars.div_ceil(2)
    }

    /// The address-major per-cycle block width: cycle `t`'s coefficients
    /// occupy grid indices `[t · cycle_stride, (t+1) · cycle_stride)` —
    /// legacy's `dense_stride = 2^(e + log_k_chunk)`.
    pub const fn cycle_stride(&self) -> usize {
        debug_assert!(self.total_vars >= self.log_t);
        1 << (self.total_vars - self.log_t)
    }

    /// The address-major within-block address stride — legacy's
    /// `one_hot_stride = 2^e`, where `e` is the embedding extra a
    /// precommitted candidate wider than the main matrix leaves between
    /// addresses (`1` on unwidened grids).
    pub const fn one_hot_stride(&self) -> usize {
        debug_assert!(self.total_vars >= self.log_t + self.log_k_chunk);
        1 << (self.total_vars - self.log_t - self.log_k_chunk)
    }
}

/// One committed witness polynomial: its id, commitment, and the opening hint
/// the stage-8 joint opening consumes.
pub struct WitnessCommitment<PCS: CommitmentScheme> {
    pub id: JoltCommittedPolynomial,
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
}

/// The witness-commitment slot: commit every trace-derived polynomial in
/// `ids` as a consumer of the witness stream over the shared embedding grid.
/// Results are returned in `ids` order; execution order, batching, and
/// streaming strategy are the implementation's business (the trait
/// deliberately does not require
/// [`StreamingCommitment`](jolt_openings::StreamingCommitment) — that is
/// the reference implementation's
/// strategy). Advice polynomials are not trace-derived and commit through
/// [`commit_advice`](Self::commit_advice) instead. Transcript-free: the
/// caller absorbs the returned commitments.
pub trait CommitWitness<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    fn commit_witness(
        &self,
        session: &mut ProofSession,
        source: &dyn RowSource,
        ids: &[JoltCommittedPolynomial],
        grid: CommitmentGrid,
        setup: &PCS::ProverSetup,
    ) -> Result<Vec<WitnessCommitment<PCS>>, KernelError<F>>;

    /// Commit one advice polynomial in its dedicated cycle-major grid.
    fn commit_advice(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessOracle<F>,
        id: JoltCommittedPolynomial,
        grid: CommitmentGrid,
        setup: &PCS::ProverSetup,
    ) -> Result<WitnessCommitment<PCS>, KernelError<F>>;
}
