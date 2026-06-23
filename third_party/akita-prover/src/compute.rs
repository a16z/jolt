//! Prover compute backend boundary.
//!
//! The first backend is the existing CPU/Rayon implementation. The boundary is
//! intentionally operation-shaped: migrated prover code asks the backend to run
//! named commit/protocol kernels, and does not reach through prepared setup for
//! raw CPU matrices or NTT slots.

use crate::backend::onehot::{column_sweep_ajtai_onehot, MultiChunkEntry, SingleChunkEntry};
use crate::backend::sparse_ring::{column_sweep_sparse, SparseRingBlockEntry};
use crate::kernels::crt_ntt::{build_ntt_slot, NttSlotCache};
#[cfg(test)]
use crate::kernels::linear::fused_split_eq_quotients;
use crate::kernels::linear::{
    fused_split_eq_quotients_prover_bounds, mat_vec_mul_ntt_dense_digits_i8_trusted,
    mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_i8_dense_single_row, mat_vec_mul_ntt_i8_strided,
    mat_vec_mul_ntt_raw_i8_strided, mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic,
    selected_crt_i8_capacity_profile, CrtI8CapacityProfile,
};
#[cfg(any(test, feature = "zk"))]
use crate::validation::MAX_I8_LOG_BASIS;
use crate::AkitaProverSetup;
use akita_algebra::CyclotomicRing;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore, HalvingField};
use akita_types::AkitaExpandedSetup;
use std::array::from_fn;
use std::sync::Arc;
#[cfg(feature = "zk")]
use std::sync::OnceLock;

/// Flat block table handed to a compute backend.
///
/// `entries[offsets[i]..offsets[i + 1]]` is the entry slice for block `i`.
/// This is the canonical compact representation for sparse per-block work:
/// CPU code may recover per-block slices, while accelerator backends can upload
/// one contiguous entry table plus one offsets table.
#[derive(Debug, Clone, Copy)]
pub struct FlatBlockTable<'a, E> {
    entries: &'a [E],
    offsets: &'a [u32],
}

impl<'a, E> FlatBlockTable<'a, E> {
    /// Build a flat block table from validated storage.
    #[inline]
    pub(crate) fn new(entries: &'a [E], offsets: &'a [u32]) -> Self {
        Self { entries, offsets }
    }

    /// Contiguous sparse entries.
    #[inline]
    pub fn entries(&self) -> &'a [E] {
        self.entries
    }

    /// Block offsets into [`Self::entries`].
    #[inline]
    pub fn offsets(&self) -> &'a [u32] {
        self.offsets
    }

    /// Number of logical blocks.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Entry slice for one block.
    pub fn block(&self, idx: usize) -> Result<&'a [E], AkitaError> {
        let lo = self.offsets.get(idx).copied().ok_or_else(|| {
            AkitaError::InvalidSetup(format!("flat block table missing offset {idx}"))
        })? as usize;
        let hi = self.offsets.get(idx + 1).copied().ok_or_else(|| {
            AkitaError::InvalidSetup(format!("flat block table missing offset {}", idx + 1))
        })? as usize;
        if lo > hi || hi > self.entries.len() {
            return Err(AkitaError::InvalidSetup(format!(
                "flat block table has malformed offsets for block {idx}: {lo}..{hi} over {} entries",
                self.entries.len()
            )));
        }
        Ok(&self.entries[lo..hi])
    }

    fn block_slices(&self) -> Result<Vec<&'a [E]>, AkitaError> {
        (0..self.num_blocks()).map(|idx| self.block(idx)).collect()
    }
}

/// Dense polynomial commit representation handed to the compute backend.
pub enum DenseCommitInput<'a, F: FieldCore, const D: usize> {
    /// Balanced digit planes are already cached by the polynomial.
    CachedDigits {
        /// Per-block digit slices.
        digit_block_slices: Vec<&'a [[i8; D]]>,
        /// Logarithm of the gadget basis used to produce the cached digits.
        log_basis: u32,
    },
    /// Ring coefficients need backend-side digit decomposition.
    CoeffBlocks {
        /// Per-block coefficient slices.
        block_slices: Vec<&'a [CyclotomicRing<F, D>]>,
        /// Number of balanced digits used for the A-side commit.
        num_digits_commit: usize,
        /// Logarithm of the gadget basis.
        log_basis: u32,
    },
}

/// Dense commit operation plan.
pub struct DenseCommitRowsPlan<'a, F: FieldCore, const D: usize> {
    /// Number of A rows to produce.
    pub n_a: usize,
    /// Dense polynomial input representation.
    pub input: DenseCommitInput<'a, F, D>,
}

/// One-hot commit input representation.
///
/// The contained entry slices are read-only plan views. They are public so
/// accelerator crates can implement [`CommitmentComputeBackend`] without
/// depending on CPU-prepared storage, while construction remains owned by the
/// polynomial representations.
pub enum OneHotCommitBlocks<'a> {
    /// One ring has at most one hot coefficient.
    SingleChunk(FlatBlockTable<'a, SingleChunkEntry>),
    /// One ring may contain several hot coefficients.
    MultiChunk(FlatBlockTable<'a, MultiChunkEntry>),
}

/// One-hot commit operation plan.
pub struct OneHotCommitRowsPlan<'a> {
    /// Number of A rows to produce.
    pub n_a: usize,
    /// Root block length in ring elements.
    pub block_len: usize,
    /// Number of balanced digits used for the A-side commit.
    pub num_digits_commit: usize,
    /// Per-block one-hot entries.
    pub(crate) blocks: OneHotCommitBlocks<'a>,
}

impl<'a> OneHotCommitRowsPlan<'a> {
    /// Per-block one-hot entries.
    #[inline]
    pub fn blocks(&self) -> &OneHotCommitBlocks<'a> {
        &self.blocks
    }
}

/// Sparse signed-ring commit operation plan.
pub struct SparseRingCommitRowsPlan<'a> {
    /// Number of A rows to produce.
    pub n_a: usize,
    /// Root block length in ring elements.
    pub block_len: usize,
    /// Number of balanced digits used for the A-side commit.
    pub num_digits_commit: usize,
    /// Per-block sparse signed coefficients.
    pub(crate) blocks: FlatBlockTable<'a, SparseRingBlockEntry>,
}

impl<'a> SparseRingCommitRowsPlan<'a> {
    /// Per-block sparse signed coefficients.
    #[inline]
    pub fn blocks(&self) -> FlatBlockTable<'a, SparseRingBlockEntry> {
        self.blocks
    }
}

/// Recursive witness commit operation plan.
pub struct RecursiveWitnessCommitRowsPlan<'a, const D: usize> {
    /// Recursive witness digit rows, chunked at `D`.
    pub coeffs: &'a [[i8; D]],
    /// Number of rows to produce.
    pub n_rows: usize,
    /// Recursive block length.
    pub block_len: usize,
    /// Number of logical blocks.
    pub num_blocks: usize,
    /// Number of balanced digits used for the A-side commit.
    pub num_digits_commit: usize,
    /// Logarithm of the gadget basis.
    pub log_basis: u32,
}

/// Shared prepared-setup contract for prover compute backends.
pub trait ComputeBackendSetup<F>: Send + Sync
where
    F: FieldCore + CanonicalField,
{
    /// Backend-prepared setup for a concrete ring dimension.
    type PreparedSetup<const D: usize>: Send + Sync;

    /// Prepare backend state from a prover setup wrapper.
    fn prepare_setup<const D: usize>(
        &self,
        setup: &AkitaProverSetup<F, D>,
    ) -> Result<Self::PreparedSetup<D>, AkitaError> {
        self.prepare_expanded::<D>(setup.expanded.clone())
    }

    /// Prepare backend state from already-expanded setup data.
    fn prepare_expanded<const D: usize>(
        &self,
        expanded: Arc<AkitaExpandedSetup<F>>,
    ) -> Result<Self::PreparedSetup<D>, AkitaError>;

    /// Expanded setup used to prepare this backend context.
    fn prepared_expanded_setup<'a, const D: usize>(
        &self,
        prepared: &'a Self::PreparedSetup<D>,
    ) -> &'a AkitaExpandedSetup<F>;

    /// Ensure explicit setup metadata and backend-prepared state match.
    fn validate_prepared_setup<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<(), AkitaError> {
        let prepared_expanded = self.prepared_expanded_setup::<D>(prepared);
        // Valid setup matrices are deterministic from the seed; compare the
        // compact setup identity so independently materialized equivalent
        // setups validate without re-hashing the matrix on every prover call.
        if prepared_expanded.seed() != expanded.seed() {
            return Err(AkitaError::InvalidSetup(
                "prepared compute context was built for a different setup".to_string(),
            ));
        }
        Ok(())
    }
}

/// Negacyclic digit mat-vec operations shared by commitment and protocol code.
pub trait DigitRowsComputeBackend<F>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
{
    /// Negacyclic single-input digit mat-vec rows.
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;

    /// Negacyclic ZK B-blinding digit mat-vec rows.
    #[cfg(feature = "zk")]
    fn zk_b_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;

    /// Negacyclic ZK D-blinding digit mat-vec rows.
    #[cfg(feature = "zk")]
    fn zk_d_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;
}

/// Cyclic digit mat-vec operations needed by ring-switch relation code.
pub trait CyclicRowsComputeBackend<F>: DigitRowsComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
    /// Cyclic single-input digit mat-vec rows.
    fn cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;

    /// Cyclic ZK B-blinding digit mat-vec rows.
    #[cfg(feature = "zk")]
    fn zk_b_cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;

    /// Cyclic ZK D-blinding digit mat-vec rows.
    #[cfg(feature = "zk")]
    fn zk_d_cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;
}

/// Commitment row operations for migrated root/ring commitment work.
pub trait CommitmentComputeBackend<F>: DigitRowsComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
    /// Dense A-side commit rows.
    fn dense_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: DenseCommitRowsPlan<'_, F, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>;

    /// One-hot A-side commit rows.
    fn onehot_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: OneHotCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>;

    /// Sparse signed-ring A-side commit rows.
    fn sparse_ring_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: SparseRingCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>;

    /// Recursive witness A-side commit rows.
    fn recursive_witness_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: RecursiveWitnessCommitRowsPlan<'_, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>;
}

/// Full ring-switch relation operation input.
pub struct RingSwitchRelationRowsPlan<'a, const D: usize> {
    /// Number of D-side cyclic rows to produce.
    pub n_d: usize,
    /// Number of B-side cyclic rows to produce.
    pub n_b: usize,
    /// Number of A-side quotient rows to produce.
    pub n_a: usize,
    /// Flat decomposed `e_hat` digits for the D-side relation rows.
    pub e_hat: &'a [[i8; D]],
    /// Flat decomposed inner-commitment digits for the B-side relation rows.
    pub t_hat: &'a [[i8; D]],
    /// One centered `z` segment contributing to A-side quotient rows.
    pub z_segment: &'a [[i32; D]],
    /// Infinity norm of the full centered `z_folded_rings` witness.
    pub z_folded_centered_inf_norm: u32,
    /// Logarithm of the gadget basis used to produce `e_hat` and `t_hat`.
    pub log_basis: u32,
}

/// Additional public-row quotient operation input.
pub struct RingSwitchQuotientRowsPlan<'a, const D: usize> {
    /// Number of A-side quotient rows to produce.
    pub n_a: usize,
    /// One centered `z` segment contributing to A-side quotient rows.
    pub z_segment: &'a [[i32; D]],
    /// Infinity norm of the full centered `z_folded_rings` witness.
    pub z_folded_centered_inf_norm: u32,
}

/// Named ring-switch relation rows returned by a backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingSwitchRelationRows<F: FieldCore, const D: usize> {
    /// D-side cyclic rows.
    pub d_cyclic: Vec<CyclotomicRing<F, D>>,
    /// B-side cyclic rows.
    pub b_cyclic: Vec<CyclotomicRing<F, D>>,
    /// A-side quotient rows.
    pub a_quotients: Vec<CyclotomicRing<F, D>>,
}

/// Ring-switch relation operations for migrated proving work.
pub trait RingSwitchComputeBackend<F>: CyclicRowsComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
    /// Fused cyclic/quotient rows used by ring-switch finalization.
    fn ring_switch_relation_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: RingSwitchRelationRowsPlan<'_, D>,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
    where
        F: HalvingField;

    /// A-side quotient rows for an additional public-row segment.
    fn ring_switch_quotient_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: RingSwitchQuotientRowsPlan<'_, D>,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
    where
        F: HalvingField;
}

/// Full first-PR prover compute surface.
pub trait ProverComputeBackend<F>:
    CommitmentComputeBackend<F> + RingSwitchComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
}

impl<F, B> ProverComputeBackend<F> for B
where
    F: FieldCore + CanonicalField,
    B: CommitmentComputeBackend<F> + RingSwitchComputeBackend<F>,
{
}

/// CPU backend using the existing Rust/Rayon kernels.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

/// CPU-prepared setup for one field/ring-dimension pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuPreparedSetup<F: FieldCore, const D: usize> {
    expanded: Arc<AkitaExpandedSetup<F>>,
    ntt_shared: NttSlotCache<D>,
    ntt_i8_capacity: CrtI8CapacityProfile,
    #[cfg(feature = "zk")]
    ntt_zk_b: OnceLock<NttSlotCache<D>>,
    #[cfg(feature = "zk")]
    ntt_zk_d: OnceLock<NttSlotCache<D>>,
}

/// CRT/NTT profile and universal i8 capacity metadata for a prepared setup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedCrtNttProfile {
    /// Stable profile identifier used by benchmark/report tooling.
    pub profile_id: &'static str,
    /// Number of CRT primes in the selected profile.
    pub num_primes: usize,
    /// Signed limb width used by the CRT NTT representation.
    pub limb_bits: u32,
    /// Largest balanced i8 log basis accepted by prover i8 kernels.
    pub max_i8_log_basis: u32,
    /// Safe accumulation width for balanced i8 digits at `max_i8_log_basis`.
    pub balanced_digit_safe_width: usize,
    /// Safe accumulation width for raw signed i8 recursive-witness inputs.
    pub raw_i8_safe_width: usize,
}

impl From<CrtI8CapacityProfile> for PreparedCrtNttProfile {
    fn from(profile: CrtI8CapacityProfile) -> Self {
        Self {
            profile_id: profile.profile_id,
            num_primes: profile.num_primes,
            limb_bits: profile.limb_bits,
            max_i8_log_basis: profile.max_i8_log_basis,
            balanced_digit_safe_width: profile.balanced_digit_safe_width,
            raw_i8_safe_width: profile.raw_i8_safe_width,
        }
    }
}

impl<F: FieldCore, const D: usize> CpuPreparedSetup<F, D> {
    /// In-memory byte footprint of the shared setup NTT cache (negacyclic plus
    /// cyclic slots). Diagnostic surface for the profiler / bench report.
    pub fn shared_ntt_cache_bytes(&self) -> usize {
        self.ntt_shared.cache_bytes()
    }

    /// CRT/NTT profile and universal i8 capacity metadata for the shared setup
    /// cache. The capacity widths are the boundary checked during backend
    /// preparation before hot i8 kernels can rely on their internal invariant.
    pub fn shared_ntt_profile(&self) -> PreparedCrtNttProfile {
        self.ntt_i8_capacity.into()
    }
}

fn validate_digit_row_request(
    row_len: usize,
    row_width: usize,
    total_ring_elements: usize,
) -> Result<(), AkitaError> {
    if row_width == 0 {
        return Err(AkitaError::InvalidSetup(
            "prepared setup row width must be nonzero".to_string(),
        ));
    }
    let required = row_len.checked_mul(row_width).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "digit row request overflows: row_len={row_len} row_width={row_width}"
        ))
    })?;
    if required > total_ring_elements {
        return Err(AkitaError::InvalidSetup(format!(
            "digit row request needs {required} setup ring elements but prepared setup has {total_ring_elements}"
        )));
    }
    Ok(())
}

#[cfg(feature = "zk")]
fn zk_digit_rows_from_slot<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    row_len: usize,
    row_width: usize,
    total_ring_elements: usize,
    digits: &[[i8; D]],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    if digits.is_empty() {
        return Ok(vec![CyclotomicRing::zero(); row_len]);
    }
    if digits.len() > row_width {
        return Err(AkitaError::InvalidSetup(
            "ZK matrix digit columns exceed row width".to_string(),
        ));
    }
    validate_digit_row_request(row_len, row_width, total_ring_elements)?;
    mat_vec_mul_ntt_single_i8(slot, row_len, row_width, digits, MAX_I8_LOG_BASIS)
}

#[cfg(feature = "zk")]
fn zk_cyclic_digit_rows_from_slot<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    row_len: usize,
    row_width: usize,
    total_ring_elements: usize,
    digits: &[[i8; D]],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    if digits.is_empty() {
        return Ok(vec![CyclotomicRing::zero(); row_len]);
    }
    if digits.len() > row_width {
        return Err(AkitaError::InvalidSetup(
            "ZK matrix digit columns exceed row width".to_string(),
        ));
    }
    validate_digit_row_request(row_len, row_width, total_ring_elements)?;
    mat_vec_mul_ntt_single_i8_cyclic(slot, row_len, row_width, digits, MAX_I8_LOG_BASIS)
}

#[cfg(feature = "zk")]
fn zk_b_slot<F: FieldCore + CanonicalField, const D: usize>(
    prepared: &CpuPreparedSetup<F, D>,
) -> Result<&NttSlotCache<D>, AkitaError> {
    if let Some(slot) = prepared.ntt_zk_b.get() {
        return Ok(slot);
    }
    let total = prepared
        .expanded
        .zk_b_matrix
        .total_ring_elements_at::<D>()?;
    let slot = build_ntt_slot(prepared.expanded.zk_b_matrix.ring_view::<D>(1, total)?)?;
    let _ = prepared.ntt_zk_b.set(slot);
    prepared.ntt_zk_b.get().ok_or_else(|| {
        AkitaError::InvalidSetup("failed to initialize ZK B prepared slot".to_string())
    })
}

#[cfg(feature = "zk")]
fn zk_d_slot<F: FieldCore + CanonicalField, const D: usize>(
    prepared: &CpuPreparedSetup<F, D>,
) -> Result<&NttSlotCache<D>, AkitaError> {
    if let Some(slot) = prepared.ntt_zk_d.get() {
        return Ok(slot);
    }
    let total = prepared
        .expanded
        .zk_d_matrix
        .total_ring_elements_at::<D>()?;
    let slot = build_ntt_slot(prepared.expanded.zk_d_matrix.ring_view::<D>(1, total)?)?;
    let _ = prepared.ntt_zk_d.set(slot);
    prepared.ntt_zk_d.get().ok_or_else(|| {
        AkitaError::InvalidSetup("failed to initialize ZK D prepared slot".to_string())
    })
}

impl<F> ComputeBackendSetup<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    type PreparedSetup<const D: usize> = CpuPreparedSetup<F, D>;

    fn prepare_expanded<const D: usize>(
        &self,
        expanded: Arc<AkitaExpandedSetup<F>>,
    ) -> Result<Self::PreparedSetup<D>, AkitaError> {
        let ntt_i8_capacity = selected_crt_i8_capacity_profile::<F, D>()?;
        let total = expanded.shared_matrix.total_ring_elements_at::<D>()?;
        let ntt_shared = build_ntt_slot(expanded.shared_matrix.ring_view::<D>(1, total)?)?;
        Ok(CpuPreparedSetup {
            expanded,
            ntt_shared,
            ntt_i8_capacity,
            #[cfg(feature = "zk")]
            ntt_zk_b: OnceLock::new(),
            #[cfg(feature = "zk")]
            ntt_zk_d: OnceLock::new(),
        })
    }

    fn prepared_expanded_setup<'a, const D: usize>(
        &self,
        prepared: &'a Self::PreparedSetup<D>,
    ) -> &'a AkitaExpandedSetup<F> {
        prepared.expanded.as_ref()
    }
}

impl<F> CommitmentComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn dense_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: DenseCommitRowsPlan<'_, F, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        match plan.input {
            DenseCommitInput::CachedDigits {
                digit_block_slices,
                log_basis,
            } => {
                let row_width = digit_block_slices.first().map_or(0, |digits| digits.len());
                mat_vec_mul_ntt_dense_digits_i8_trusted(
                    &prepared.ntt_shared,
                    plan.n_a,
                    row_width,
                    &digit_block_slices,
                    log_basis,
                )
            }
            DenseCommitInput::CoeffBlocks {
                block_slices,
                num_digits_commit,
                log_basis,
            } => {
                let row_width = block_slices.first().map_or(Ok(0usize), |block| {
                    block.len().checked_mul(num_digits_commit).ok_or_else(|| {
                        AkitaError::InvalidSetup("dense coefficient row width overflow".to_string())
                    })
                })?;
                if plan.n_a == 1 {
                    Ok(mat_vec_mul_ntt_i8_dense_single_row(
                        &prepared.ntt_shared,
                        row_width,
                        &block_slices,
                        num_digits_commit,
                        log_basis,
                    )?
                    .into_iter()
                    .map(|ring| vec![ring])
                    .collect())
                } else {
                    mat_vec_mul_ntt_i8_dense(
                        &prepared.ntt_shared,
                        plan.n_a,
                        row_width,
                        &block_slices,
                        num_digits_commit,
                        log_basis,
                    )
                }
            }
        }
    }

    fn onehot_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: OneHotCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    {
        let active_a_cols = plan
            .block_len
            .checked_mul(plan.num_digits_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("active A width overflow".to_string()))?;
        let a_view = prepared
            .expanded
            .shared_matrix
            .ring_view::<D>(plan.n_a, active_a_cols)?;
        Ok(match plan.blocks {
            OneHotCommitBlocks::SingleChunk(blocks) => {
                column_sweep_ajtai_onehot::<SingleChunkEntry, F, D>(
                    &a_view,
                    &blocks.block_slices()?,
                    plan.n_a,
                    active_a_cols,
                    plan.num_digits_commit,
                )
            }
            OneHotCommitBlocks::MultiChunk(blocks) => {
                column_sweep_ajtai_onehot::<MultiChunkEntry, F, D>(
                    &a_view,
                    &blocks.block_slices()?,
                    plan.n_a,
                    active_a_cols,
                    plan.num_digits_commit,
                )
            }
        })
    }

    fn sparse_ring_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: SparseRingCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    {
        let active_a_cols = plan
            .block_len
            .checked_mul(plan.num_digits_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("active A width overflow".to_string()))?;
        let a_view = prepared
            .expanded
            .shared_matrix
            .ring_view::<D>(plan.n_a, active_a_cols)?;
        let a_rows = (0..plan.n_a)
            .map(|idx| a_view.row(idx))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(column_sweep_sparse(
            &a_rows,
            &plan.blocks.block_slices()?,
            plan.n_a,
            plan.block_len,
            plan.num_digits_commit,
        ))
    }

    fn recursive_witness_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: RecursiveWitnessCommitRowsPlan<'_, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        let row_width = plan
            .block_len
            .checked_mul(plan.num_digits_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("recursive A width overflow".to_string()))?;
        if plan.num_digits_commit == 1 {
            mat_vec_mul_ntt_raw_i8_strided(
                &prepared.ntt_shared,
                plan.n_rows,
                row_width,
                plan.coeffs,
                plan.num_blocks,
                plan.block_len,
            )
        } else {
            let ring_elems: Vec<CyclotomicRing<F, D>> = plan
                .coeffs
                .iter()
                .map(|digit| {
                    let coeffs = from_fn(|k| F::from_i8(digit[k]));
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect();
            mat_vec_mul_ntt_i8_strided(
                &prepared.ntt_shared,
                plan.n_rows,
                row_width,
                &ring_elems,
                plan.num_blocks,
                plan.block_len,
                plan.num_digits_commit,
                plan.log_basis,
            )
        }
    }
}

impl<F> DigitRowsComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        validate_digit_row_request(
            row_len,
            digits.len(),
            prepared
                .expanded
                .shared_matrix
                .total_ring_elements_at::<D>()?,
        )?;
        mat_vec_mul_ntt_single_i8(
            &prepared.ntt_shared,
            row_len,
            digits.len(),
            digits,
            log_basis,
        )
    }

    #[cfg(feature = "zk")]
    fn zk_b_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        zk_digit_rows_from_slot(
            zk_b_slot(prepared)?,
            row_len,
            row_width,
            prepared
                .expanded
                .zk_b_matrix
                .total_ring_elements_at::<D>()?,
            digits,
        )
    }

    #[cfg(feature = "zk")]
    fn zk_d_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        zk_digit_rows_from_slot(
            zk_d_slot(prepared)?,
            row_len,
            row_width,
            prepared
                .expanded
                .zk_d_matrix
                .total_ring_elements_at::<D>()?,
            digits,
        )
    }
}

impl<F> CyclicRowsComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        validate_digit_row_request(
            row_len,
            digits.len(),
            prepared
                .expanded
                .shared_matrix
                .total_ring_elements_at::<D>()?,
        )?;
        mat_vec_mul_ntt_single_i8_cyclic(
            &prepared.ntt_shared,
            row_len,
            digits.len(),
            digits,
            log_basis,
        )
    }

    #[cfg(feature = "zk")]
    fn zk_b_cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        zk_cyclic_digit_rows_from_slot(
            zk_b_slot(prepared)?,
            row_len,
            row_width,
            prepared
                .expanded
                .zk_b_matrix
                .total_ring_elements_at::<D>()?,
            digits,
        )
    }

    #[cfg(feature = "zk")]
    fn zk_d_cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        zk_cyclic_digit_rows_from_slot(
            zk_d_slot(prepared)?,
            row_len,
            row_width,
            prepared
                .expanded
                .zk_d_matrix
                .total_ring_elements_at::<D>()?,
            digits,
        )
    }
}

impl<F> RingSwitchComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn ring_switch_relation_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: RingSwitchRelationRowsPlan<'_, D>,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
    where
        F: HalvingField,
    {
        let (d_cyclic, b_cyclic, a_quotients) = fused_split_eq_quotients_prover_bounds(
            &prepared.ntt_shared,
            plan.n_d,
            plan.n_b,
            plan.n_a,
            plan.e_hat,
            plan.t_hat,
            plan.z_segment,
            plan.z_folded_centered_inf_norm,
            plan.log_basis,
        )?;
        Ok(RingSwitchRelationRows {
            d_cyclic,
            b_cyclic,
            a_quotients,
        })
    }

    fn ring_switch_quotient_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: RingSwitchQuotientRowsPlan<'_, D>,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
    where
        F: HalvingField,
    {
        let (_d_cyclic, _b_cyclic, a_quotients) = fused_split_eq_quotients_prover_bounds(
            &prepared.ntt_shared,
            0,
            0,
            plan.n_a,
            &[][..],
            &[][..],
            plan.z_segment,
            plan.z_folded_centered_inf_norm,
            1,
        )?;
        Ok(a_quotients)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Fp64;
    #[cfg(feature = "zk")]
    use akita_types::FlatMatrix;
    use akita_types::SetupMatrixEnvelope;

    type F = Fp64<4294967197>;
    const D: usize = 32;

    fn setup_envelope(max_setup_len: usize) -> SetupMatrixEnvelope {
        SetupMatrixEnvelope {
            max_setup_len,
            #[cfg(feature = "zk")]
            max_zk_b_len: 1,
            #[cfg(feature = "zk")]
            max_zk_d_len: 1,
        }
    }

    #[cfg(feature = "zk")]
    fn setup_envelope_with_zk(
        max_setup_len: usize,
        max_zk_b_len: usize,
        max_zk_d_len: usize,
    ) -> SetupMatrixEnvelope {
        SetupMatrixEnvelope {
            max_setup_len,
            max_zk_b_len,
            max_zk_d_len,
        }
    }

    fn prepared() -> CpuPreparedSetup<F, D> {
        let setup =
            AkitaProverSetup::<F, D>::generate_with_capacity(8, 1, setup_envelope(32)).unwrap();
        CpuBackend.prepare_setup(&setup).unwrap()
    }

    #[cfg(feature = "zk")]
    fn direct_negacyclic_rows(
        matrix: &FlatMatrix<F>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Vec<CyclotomicRing<F, D>> {
        let view = matrix.ring_view::<D>(row_len, row_width).unwrap();
        let digit_rings = digits
            .iter()
            .map(|digit| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
                    F::from_i64(digit[idx] as i64)
                }))
            })
            .collect::<Vec<_>>();
        (0..row_len)
            .map(|row_idx| {
                let row_start = row_idx * row_width;
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (entry, digit) in view.as_slice()[row_start..row_start + digit_rings.len()]
                    .iter()
                    .zip(digit_rings.iter())
                {
                    entry.mul_accumulate_into(digit, &mut acc);
                }
                acc
            })
            .collect()
    }

    #[test]
    fn cpu_prepared_setup_identity_rejects_mismatched_setup() {
        let setup_a =
            AkitaProverSetup::<F, D>::generate_with_capacity(8, 1, setup_envelope(32)).unwrap();
        let setup_b =
            AkitaProverSetup::<F, D>::generate_with_capacity(9, 1, setup_envelope(32)).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup_a).unwrap();

        CpuBackend
            .validate_prepared_setup::<D>(&prepared, setup_a.expanded.as_ref())
            .expect("matching setup");
        assert!(
            CpuBackend
                .validate_prepared_setup::<D>(&prepared, setup_b.expanded.as_ref())
                .is_err(),
            "prepared context must stay bound to the setup used to create it"
        );
    }

    #[test]
    fn cpu_prepared_setup_identity_accepts_equivalent_setup() {
        let setup_a =
            AkitaProverSetup::<F, D>::generate_with_capacity(8, 1, setup_envelope(32)).unwrap();
        let setup_b =
            AkitaProverSetup::<F, D>::generate_with_capacity(8, 1, setup_envelope(32)).unwrap();
        assert!(!Arc::ptr_eq(&setup_a.expanded, &setup_b.expanded));

        let prepared = CpuBackend.prepare_setup(&setup_a).unwrap();

        CpuBackend
            .validate_prepared_setup::<D>(&prepared, setup_b.expanded.as_ref())
            .expect("equivalent deterministic setup should validate");
    }

    #[test]
    fn cpu_prepared_setup_reports_checked_crt_capacity_profile() {
        let prepared = prepared();
        let profile = prepared.shared_ntt_profile();

        assert_eq!(profile.profile_id, "Q32/2xi32");
        assert_eq!(profile.num_primes, 2);
        assert_eq!(profile.limb_bits, 32);
        assert_eq!(profile.max_i8_log_basis, MAX_I8_LOG_BASIS);
        assert!(profile.balanced_digit_safe_width > 0);
        assert!(profile.raw_i8_safe_width > 0);
    }

    #[test]
    fn cpu_digit_rows_match_direct_kernel() {
        let prepared = prepared();
        let digits = vec![[1i8; D], [-1i8; D], [2i8; D]];
        let log_basis = 3;
        let via_backend = CpuBackend
            .digit_rows::<D>(&prepared, 2, &digits, log_basis)
            .expect("backend digit rows");
        let direct =
            mat_vec_mul_ntt_single_i8(&prepared.ntt_shared, 2, digits.len(), &digits, log_basis)
                .expect("direct digit rows");
        assert_eq!(via_backend, direct);
    }

    #[test]
    fn cpu_digit_rows_accept_logical_input_longer_than_stride() {
        let prepared = prepared();
        let digits = vec![[1i8; D]; 12];
        let log_basis = 3;
        let via_backend = CpuBackend
            .digit_rows::<D>(&prepared, 2, &digits, log_basis)
            .expect("backend digit rows");
        let direct =
            mat_vec_mul_ntt_single_i8(&prepared.ntt_shared, 2, digits.len(), &digits, log_basis)
                .expect("direct digit rows");
        assert_eq!(via_backend, direct);
    }

    #[test]
    fn cpu_cyclic_digit_rows_match_direct_kernel() {
        let prepared = prepared();
        let digits = vec![[1i8; D], [0i8; D], [-2i8; D], [3i8; D]];
        let log_basis = 3;
        let via_backend = CpuBackend
            .cyclic_digit_rows::<D>(&prepared, 2, &digits, log_basis)
            .expect("backend cyclic digit rows");
        let direct = mat_vec_mul_ntt_single_i8_cyclic(
            &prepared.ntt_shared,
            2,
            digits.len(),
            &digits,
            log_basis,
        )
        .expect("direct cyclic digit rows");
        assert_eq!(via_backend, direct);
    }

    #[cfg(feature = "zk")]
    #[test]
    fn cpu_zk_digit_rows_match_direct_negacyclic_product() {
        let row_len = 2;
        let row_width = 3;
        let setup = AkitaProverSetup::<F, D>::generate_with_capacity(
            8,
            1,
            setup_envelope_with_zk(32, row_len * row_width, row_len * row_width),
        )
        .unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let digits = vec![[1i8; D], [-2i8; D]];

        let b_rows = CpuBackend
            .zk_b_digit_rows::<D>(&prepared, row_len, row_width, &digits)
            .expect("backend zkB digit rows");
        let b_direct =
            direct_negacyclic_rows(setup.expanded.zk_b_matrix(), row_len, row_width, &digits);
        assert_eq!(b_rows, b_direct);

        let d_rows = CpuBackend
            .zk_d_digit_rows::<D>(&prepared, row_len, row_width, &digits)
            .expect("backend zkD digit rows");
        let d_direct =
            direct_negacyclic_rows(setup.expanded.zk_d_matrix(), row_len, row_width, &digits);
        assert_eq!(d_rows, d_direct);
    }

    #[test]
    fn cpu_ring_switch_relation_rows_match_direct_kernel() {
        let prepared = prepared();
        let e_hat = vec![[1i8; D], [2i8; D]];
        let t_hat = vec![[-1i8; D], [3i8; D]];
        let z_segment = vec![[1i32; D], [-2i32; D], [3i32; D]];
        let via_backend = CpuBackend
            .ring_switch_relation_rows::<D>(
                &prepared,
                RingSwitchRelationRowsPlan {
                    n_d: 1,
                    n_b: 1,
                    n_a: 1,
                    e_hat: &e_hat,
                    t_hat: &t_hat,
                    z_segment: &z_segment,
                    z_folded_centered_inf_norm: 3,
                    log_basis: 3,
                },
            )
            .expect("backend ring-switch relation rows");
        let direct =
            fused_split_eq_quotients(&prepared.ntt_shared, 1, 1, 1, &e_hat, &t_hat, &z_segment, 3)
                .expect("direct fused split-eq rows");
        assert_eq!(
            (
                via_backend.d_cyclic,
                via_backend.b_cyclic,
                via_backend.a_quotients
            ),
            direct
        );
    }
}
