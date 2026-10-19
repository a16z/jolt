//! Shared setup data shapes for Akita prover and verifier APIs.

use super::setup_prefix::SetupPrefixVerifierRegistry;
use akita_error::AkitaError;
use akita_params::FlatMatrix;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
#[allow(unused_imports)]
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};
use rand_core::{CryptoRng, RngCore};
use shake::digest::{ExtendableOutput, Update, XofReader};
use shake::Shake256;
use std::io::{Read, Write};
use std::sync::Arc;

/// Versioned derivation algorithm for the public field stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicMatrixDerivation {
    /// Fixed 4096-field-element SHAKE256 pages with exact field sampling.
    Shake256PagedV1,
}

impl PublicMatrixDerivation {
    /// Number of field elements in one independently derived page.
    #[must_use]
    pub const fn page_field_elements(self) -> usize {
        match self {
            Self::Shake256PagedV1 => 4096,
        }
    }
}

/// Semantic identity of the infinite public field stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaSetupSeed {
    /// Coefficient derivation algorithm.
    pub derivation: PublicMatrixDerivation,
    /// Public entropy absorbed by the derivation algorithm.
    pub seed: [u8; 32],
}

impl AkitaSetupSeed {
    /// Construct a v1 paged SHAKE256 public-matrix identity.
    #[must_use]
    pub const fn shake256_paged_v1(seed: [u8; 32]) -> Self {
        Self {
            derivation: PublicMatrixDerivation::Shake256PagedV1,
            seed,
        }
    }
}

impl From<[u8; 32]> for AkitaSetupSeed {
    fn from(seed: [u8; 32]) -> Self {
        Self::shake256_paged_v1(seed)
    }
}

/// Maximum setup matrix field elements accepted by self-describing setup
/// deserialization.
///
/// This cap protects generic verifier-facing setup decoding from allocating
/// directly from attacker-controlled seed metadata. It is not a protocol limit
/// on the deterministic public stream. Context-backed decoders should instead
/// enforce an expected shape and caller-supplied resource budget.
pub const MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS: usize = 1 << 26;

const PUBLIC_MATRIX_DOMAIN: &[u8] = b"akita/commitment/public-field-stream";
const PUBLIC_MATRIX_DERIVATION_TAG: &[u8] = b"shake256-paged-v1";

/// Seed-only stage for deterministic setup expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaSetupDescriptor {
    /// Provisioning variable bound inherited from setup construction.
    pub max_num_vars: usize,
    /// Provisioning polynomial-count bound inherited from setup construction.
    pub max_num_batched_polys: usize,
    /// Number of materialized public-matrix field elements.
    pub num_field_elements: usize,
    /// Semantic identity of the infinite public field stream.
    pub setup_seed: AkitaSetupSeed,
}

/// Expanded setup stage containing materialized public matrices.
///
/// Base role matrices (A, B, D) are packed row/column prefix views of
/// `shared_matrix`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaExpandedSetup<F: Field> {
    /// Setup seed and runtime layout metadata.
    pub descriptor: AkitaSetupDescriptor,
    /// Shared 1D flat backing vector.
    pub shared_matrix: FlatMatrix<F>,
}

/// Verifier setup artifact derived from prover setup.
///
/// Semantic state is immutable so clones can safely share prepared matrix
/// caches. To replace a setup, construct a new value with [`Self::from_parts`].
///
/// ```compile_fail
/// use akita_types::AkitaVerifierSetup;
/// use jolt_field::Prime32Offset99;
/// fn replace(setup: &mut AkitaVerifierSetup<Prime32Offset99>,
///            other: AkitaVerifierSetup<Prime32Offset99>) {
///     setup.expanded = other.expanded;
/// }
/// ```
///
/// ```compile_fail
/// use akita_types::AkitaVerifierSetup;
/// use jolt_field::Prime32Offset99;
/// fn replace(setup: &mut AkitaVerifierSetup<Prime32Offset99>,
///            other: AkitaVerifierSetup<Prime32Offset99>) {
///     setup.prefix_slots = other.prefix_slots;
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaVerifierSetup<F: Field> {
    /// Expanded matrix stage used for verification.
    expanded: Arc<AkitaExpandedSetup<F>>,
    /// Public setup-prefix commitment metadata for setup-claim offloading.
    prefix_slots: SetupPrefixVerifierRegistry<F>,
}

impl<F: Field> AkitaVerifierSetup<F> {
    /// Borrow the immutable expanded matrix stage.
    pub fn expanded(&self) -> &Arc<AkitaExpandedSetup<F>> {
        &self.expanded
    }

    /// Borrow the public setup-prefix commitment metadata.
    pub fn prefix_slots(&self) -> &SetupPrefixVerifierRegistry<F> {
        &self.prefix_slots
    }

    /// Construct verifier setup state from expanded setup and structurally checked prefix metadata.
    ///
    /// This constructor binds the registry to the public matrix identity. It
    /// does not prove that each stored prefix commitment was derived from that
    /// matrix. Callers loading external registries must establish that
    /// provenance at their setup-installation boundary.
    pub fn from_parts(
        expanded: Arc<AkitaExpandedSetup<F>>,
        prefix_slots: SetupPrefixVerifierRegistry<F>,
    ) -> Result<Self, AkitaError> {
        if prefix_slots.setup_seed() != &expanded.descriptor.setup_seed {
            return Err(AkitaError::InvalidSetup(
                "setup-prefix registry belongs to a different public matrix".to_string(),
            ));
        }
        Ok(Self {
            expanded,
            prefix_slots,
        })
    }
}

impl<F: Field> AkitaExpandedSetup<F> {
    /// Build an expanded setup from a trusted matrix the caller has already
    /// derived from `descriptor.setup_seed`.
    ///
    /// This constructor deliberately does not rederive or validate the matrix. Use
    /// [`Self::from_verified_parts`] for untrusted serialized setup bytes.
    #[must_use]
    pub fn from_trusted_seed_derived_parts_unchecked(
        descriptor: AkitaSetupDescriptor,
        shared_matrix: FlatMatrix<F>,
    ) -> Self {
        Self {
            descriptor,
            shared_matrix,
        }
    }

    /// Setup seed and runtime layout metadata.
    #[must_use]
    pub fn descriptor(&self) -> &AkitaSetupDescriptor {
        &self.descriptor
    }

    /// Shared coefficient-form matrix backing all setup roles.
    #[must_use]
    pub fn shared_matrix(&self) -> &FlatMatrix<F> {
        &self.shared_matrix
    }
}

impl<F> AkitaExpandedSetup<F>
where
    F: Field + CanonicalEncoding + Valid,
{
    /// Build an expanded setup from untrusted parts and verify the materialized
    /// matrix against the public seed.
    ///
    /// # Errors
    ///
    /// Returns a serialization error if the seed/matrix shape is malformed or
    /// the matrix was not deterministically derived from the seed.
    pub fn from_verified_parts(
        descriptor: AkitaSetupDescriptor,
        shared_matrix: FlatMatrix<F>,
    ) -> Result<Self, SerializationError> {
        let out = Self {
            descriptor,
            shared_matrix,
        };
        out.check()?;
        Ok(out)
    }
}

/// Fixed public seed for deterministic, reproducible setup.
#[must_use]
pub fn sample_akita_setup_seed() -> AkitaSetupSeed {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&0xDEAD_BEEF_CAFE_BABEu64.to_le_bytes());
    AkitaSetupSeed::shake256_paged_v1(seed)
}

/// Derive an exact flat prefix of public field elements from a seed.
///
/// The coefficient stream uses the page size owned by the selected derivation
/// policy.
/// Ring dimensions are not absorbed into the XOF and do not affect this
/// derivation. Consumers reshape the returned field prefix into role-specific
/// ring-matrix views. Equal field-length requests therefore derive identical
/// coefficient prefixes for every schedule.
///
/// Each page owns one SHAKE256 stream and repeated [`Field::random`]
/// calls consume that stream sequentially. Pages may be derived in parallel,
/// while concatenation in page-index order preserves deterministic prefix
/// semantics.
#[tracing::instrument(skip_all, name = "derive_public_matrix_prefix")]
#[must_use]
pub fn derive_public_matrix_prefix<F: Field + CanonicalEncoding>(
    num_field_elements: usize,
    id: &AkitaSetupSeed,
) -> FlatMatrix<F> {
    let mut data = vec![F::zero(); num_field_elements];
    cfg_chunks_mut!(data, id.derivation.page_field_elements())
        .enumerate()
        .for_each(|(page_index, coeffs)| {
            let mut page_rng = SetupSeedPageXof::new::<F>(id, page_index);
            for coeff in coeffs.iter_mut() {
                *coeff = F::random(&mut page_rng);
            }
        });

    FlatMatrix::from_flat_data(data)
}

/// Check that a materialized public matrix has exactly the shape declared by
/// `descriptor`.
///
/// # Errors
///
/// Returns an error if either side is structurally malformed or if the matrix
/// field-element count differs from the descriptor.
pub fn validate_public_matrix_shape_matches_seed<F: Field + Valid>(
    shared_matrix: &FlatMatrix<F>,
    descriptor: &AkitaSetupDescriptor,
) -> Result<(), SerializationError> {
    descriptor.check()?;
    shared_matrix.check()?;
    if shared_matrix.num_field_elements() != descriptor.num_field_elements {
        return Err(SerializationError::InvalidData(
            "setup shared_matrix field count does not match setup descriptor".to_string(),
        ));
    }
    Ok(())
}

/// Check that a materialized public matrix is exactly the deterministic matrix
/// derived from `seed`.
///
/// # Errors
///
/// Returns an error if the matrix shape is malformed or if any coefficient
/// differs from the seed-derived public matrix.
pub fn validate_public_matrix_matches_seed<F: Field + CanonicalEncoding + Valid>(
    shared_matrix: &FlatMatrix<F>,
    descriptor: &AkitaSetupDescriptor,
) -> Result<(), SerializationError> {
    validate_public_matrix_shape_matches_seed(shared_matrix, descriptor)?;
    let mut expected = vec![F::zero(); descriptor.setup_seed.derivation.page_field_elements()];
    for (page_index, coeffs) in shared_matrix
        .as_field_slice()
        .chunks(descriptor.setup_seed.derivation.page_field_elements())
        .enumerate()
    {
        let mut page_rng = SetupSeedPageXof::new::<F>(&descriptor.setup_seed, page_index);
        for value in &mut expected[..coeffs.len()] {
            *value = F::random(&mut page_rng);
        }
        if coeffs != &expected[..coeffs.len()] {
            return Err(SerializationError::InvalidData(
                "setup shared_matrix does not match public matrix seed".to_string(),
            ));
        }
    }
    Ok(())
}

/// Concrete SHAKE256 XOF reader for one public-matrix page.
type SetupSeedXofReader = <Shake256 as ExtendableOutput>::Reader;

struct SetupSeedPageXof {
    reader: SetupSeedXofReader,
}

impl SetupSeedPageXof {
    fn new<F: Field + CanonicalEncoding>(id: &AkitaSetupSeed, page_index: usize) -> Self {
        let mut xof = Shake256::default();
        absorb_len_prefixed(&mut xof, b"domain", PUBLIC_MATRIX_DOMAIN);
        let derivation_tag = match id.derivation {
            PublicMatrixDerivation::Shake256PagedV1 => PUBLIC_MATRIX_DERIVATION_TAG,
        };
        absorb_len_prefixed(&mut xof, b"derivation", derivation_tag);
        absorb_len_prefixed(
            &mut xof,
            b"page_field_elements",
            &(id.derivation.page_field_elements() as u64).to_le_bytes(),
        );
        absorb_len_prefixed(&mut xof, b"seed", &id.seed);
        absorb_len_prefixed(&mut xof, b"field", &field_modulus_bytes::<F>());
        absorb_len_prefixed(&mut xof, b"page", &(page_index as u64).to_le_bytes());
        Self {
            reader: xof.finalize_xof(),
        }
    }
}

fn field_modulus_bytes<F: Field + CanonicalEncoding>() -> [u8; 32] {
    akita_params::field_modulus_be_bytes::<F>()
        .expect("setup fields must have a modulus of at most 256 bits")
}

impl RngCore for SetupSeedPageXof {
    fn next_u32(&mut self) -> u32 {
        let mut buf = [0u8; 4];
        self.fill_bytes(&mut buf);
        u32::from_le_bytes(buf)
    }

    fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.fill_bytes(&mut buf);
        u64::from_le_bytes(buf)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        XofReader::read(&mut self.reader, dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for SetupSeedPageXof {}

fn absorb_len_prefixed(xof: &mut Shake256, label: &[u8], data: &[u8]) {
    xof.update(&(label.len() as u64).to_le_bytes());
    xof.update(label);
    xof.update(&(data.len() as u64).to_le_bytes());
    xof.update(data);
}

impl Valid for PublicMatrixDerivation {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl AkitaSerialize for PublicMatrixDerivation {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let tag = match self {
            Self::Shake256PagedV1 => 1u8,
        };
        tag.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1u8.serialized_size(compress)
    }
}

impl AkitaDeserialize for PublicMatrixDerivation {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        match u8::deserialize_with_mode(&mut reader, compress, validate, &())? {
            1 => Ok(Self::Shake256PagedV1),
            tag => Err(SerializationError::InvalidData(format!(
                "unsupported public matrix derivation tag {tag}"
            ))),
        }
    }
}

impl Valid for AkitaSetupSeed {
    fn check(&self) -> Result<(), SerializationError> {
        self.derivation.check()
    }
}

impl AkitaSerialize for AkitaSetupSeed {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.derivation.serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.seed)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.derivation.serialized_size(compress) + self.seed.len()
    }
}

impl AkitaDeserialize for AkitaSetupSeed {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let derivation =
            PublicMatrixDerivation::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let mut seed = [0u8; 32];
        reader.read_exact(&mut seed)?;
        let out = Self { derivation, seed };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for AkitaSetupDescriptor {
    fn check(&self) -> Result<(), SerializationError> {
        if self.max_num_batched_polys == 0 {
            return Err(SerializationError::InvalidData(
                "setup descriptor max_num_batched_polys must be at least 1".to_string(),
            ));
        }
        if self.num_field_elements == 0 {
            return Err(SerializationError::InvalidData(
                "setup descriptor num_field_elements must be non-zero".to_string(),
            ));
        }
        self.setup_seed.check()?;
        Ok(())
    }
}

impl AkitaSerialize for AkitaSetupDescriptor {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.max_num_vars
            .serialize_with_mode(&mut writer, compress)?;
        self.max_num_batched_polys
            .serialize_with_mode(&mut writer, compress)?;
        self.num_field_elements
            .serialize_with_mode(&mut writer, compress)?;
        self.setup_seed.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.max_num_vars.serialized_size(compress)
            + self.max_num_batched_polys.serialized_size(compress)
            + self.num_field_elements.serialized_size(compress)
            + self.setup_seed.serialized_size(compress)
    }
}

impl AkitaDeserialize for AkitaSetupDescriptor {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let max_num_vars = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let max_num_batched_polys =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let num_field_elements =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let setup_seed =
            AkitaSetupSeed::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            max_num_vars,
            max_num_batched_polys,
            num_field_elements,
            setup_seed,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: Field + CanonicalEncoding + Valid> Valid for AkitaExpandedSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.descriptor.check()?;
        self.shared_matrix.check()?;
        validate_public_matrix_matches_seed(&self.shared_matrix, &self.descriptor)?;
        Ok(())
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for AkitaExpandedSetup<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.descriptor.serialize_with_mode(&mut writer, compress)?;
        self.shared_matrix
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.descriptor.serialized_size(compress) + self.shared_matrix.serialized_size(compress)
    }
}

impl<F: Field + CanonicalEncoding + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for AkitaExpandedSetup<F>
{
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let descriptor =
            AkitaSetupDescriptor::deserialize_with_mode(&mut reader, compress, validate, &())?;
        descriptor.check()?;
        let shared_matrix = FlatMatrix::deserialize_with_expected_shape(
            &mut reader,
            compress,
            validate,
            descriptor.num_field_elements,
            MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
        )?;
        if matches!(validate, Validate::Yes) {
            Self::from_verified_parts(descriptor, shared_matrix)
        } else {
            Ok(Self::from_trusted_seed_derived_parts_unchecked(
                descriptor,
                shared_matrix,
            ))
        }
    }
}

impl<F: Field + CanonicalEncoding + Valid> Valid for AkitaVerifierSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()?;
        if self.prefix_slots.setup_seed() != &self.expanded.descriptor.setup_seed {
            return Err(SerializationError::InvalidData(
                "setup-prefix registry belongs to a different public matrix".to_string(),
            ));
        }
        self.prefix_slots.check()
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for AkitaVerifierSetup<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let mut writer = writer;
        self.expanded.serialize_with_mode(&mut writer, compress)?;
        self.prefix_slots.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.expanded.serialized_size(compress) + self.prefix_slots.serialized_size(compress)
    }
}

impl<F: Field + CanonicalEncoding + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for AkitaVerifierSetup<F>
{
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let mut reader = reader;
        let expanded = Arc::new(AkitaExpandedSetup::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &(),
        )?);
        let prefix_slots =
            SetupPrefixVerifierRegistry::deserialize_with_mode(reader, compress, validate, &())?;
        Self::from_parts(expanded, prefix_slots)
            .map_err(|err| SerializationError::InvalidData(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Zero;
    use jolt_field::{
        Fp32, Fp64, Prime128Offset275, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59,
    };

    type F = Prime128OffsetA7F7;
    const D: usize = 4;
    type SmallF = Fp64<4294967197>;
    const SMALL_D: usize = 64;

    fn prefix_commitment_params(
        n_prefix: usize,
        d_setup: usize,
    ) -> akita_params::GroupOpenPhaseParams {
        let a_bound = *akita_params::sis::inner_coeff_linf_bounds(
            akita_params::sis::SisModulusProfileId::Q128OffsetA7F7,
            u32::try_from(d_setup).expect("test ring dimension"),
        )
        .first()
        .expect("exact prefix A bounds");
        let inner_commit_matrix = akita_params::InnerCommitMatrixParams::try_new_with_min_rank(
            akita_params::SisTableKey {
                policy: akita_params::sis::DEFAULT_SIS_SECURITY_POLICY,
                table_digest: akita_params::sis::SisTableDigest::CURRENT,
                modulus_profile: akita_params::sis::SisModulusProfileId::Q128OffsetA7F7,
                role: akita_params::sis::SisMatrixRole::Inner,
                ring_dimension: u32::try_from(d_setup).expect("test ring dimension"),
                coeff_linf_bound: a_bound,
            },
            1,
        )
        .expect("audited prefix A matrix");
        let outer_commit_matrix = akita_params::OuterCommitMatrixParams::try_new_with_min_rank(
            akita_params::SisTableKey {
                policy: akita_params::sis::DEFAULT_SIS_SECURITY_POLICY,
                table_digest: akita_params::sis::SisTableDigest::CURRENT,
                modulus_profile: akita_params::sis::SisModulusProfileId::Q128OffsetA7F7,
                role: akita_params::sis::SisMatrixRole::Outer,
                ring_dimension: u32::try_from(d_setup).expect("test ring dimension"),
                coeff_linf_bound: 3,
            },
            inner_commit_matrix.output_rank() * (n_prefix / d_setup),
        )
        .expect("audited prefix B matrix");
        akita_params::GroupOpenPhaseParams {
            setup_natural_len: None,
            profile: akita_params::GroupCommitPhaseParams {
                version: akita_params::GroupCommitPhaseParams::VERSION,
                group: akita_params::PolynomialGroupLayout::singleton(
                    n_prefix.trailing_zeros() as usize
                ),
                blocks: akita_params::BlockGeometry::new(n_prefix / d_setup, 1, n_prefix / d_setup),
                outer_slice_count: akita_params::CommitmentSliceCount::ONE,
                inner: akita_params::RoleParams::new(
                    akita_params::GadgetDigits::new(1, 1),
                    inner_commit_matrix,
                ),
                outer: akita_params::RoleParams::new(
                    akita_params::GadgetDigits::new(1, 1),
                    outer_commit_matrix,
                ),
            },
            opening: akita_params::GroupOpeningPlan::evaluation_trace(
                akita_challenges::SparseChallengeConfig::pm1_only(0),
                1,
                1,
                1,
            ),
        }
    }

    fn seed(public_matrix_seed: [u8; 32]) -> AkitaSetupDescriptor {
        AkitaSetupDescriptor {
            max_num_vars: 8,
            max_num_batched_polys: 1,
            num_field_elements: 2 * D,
            setup_seed: public_matrix_seed.into(),
        }
    }

    #[test]
    fn verifier_setup_prefix_slots_roundtrip() {
        use crate::proof::{RingVec, SetupPrefixPublicCommitment, SetupPrefixVerifierSlot};

        let setup_seed = seed([7u8; 32]);
        let shared_matrix = derive_public_matrix_prefix::<F>(2 * D, &setup_seed.setup_seed);
        let mut prefix_slots = SetupPrefixVerifierRegistry::new(setup_seed.setup_seed.clone());
        let d_setup = 64;
        let commitment_params = prefix_commitment_params(d_setup, d_setup);
        let matrix = &commitment_params.profile.outer.matrix;
        let payload_coefficients = akita_params::CompressionChainPlan::for_complete_source(
            matrix.sis_modulus_profile(),
            matrix.output_rank() * matrix.ring_dimension(),
        )
        .expect("setup-prefix compression plan")
        .terminal_coefficients();
        let slot = SetupPrefixVerifierSlot {
            id: akita_params::scheduled_setup_prefix(d_setup - 1, commitment_params)
                .slot_id()
                .expect("setup prefix group"),
            commitment: SetupPrefixPublicCommitment {
                rows: vec![RingVec::from_coeffs(vec![F::zero(); payload_coefficients])],
            },
        };
        prefix_slots.insert(slot).expect("insert prefix slot");
        let setup = AkitaVerifierSetup {
            expanded: Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    setup_seed,
                    shared_matrix,
                ),
            ),
            prefix_slots,
        };

        let mut bytes = Vec::new();
        setup.serialize_compressed(&mut bytes).expect("serialize");
        let decoded = AkitaVerifierSetup::<F>::deserialize_compressed_exact(&bytes, &())
            .expect("deserialize");

        assert_eq!(decoded.prefix_slots.len(), 1);
        assert_eq!(decoded, setup);

        for suffix in [0, 0xa5] {
            let mut suffixed = bytes.clone();
            suffixed.push(suffix);
            assert!(AkitaVerifierSetup::<F>::deserialize_compressed_exact(&suffixed, &()).is_err());
        }

        let mut expanded_bytes = Vec::new();
        setup
            .expanded
            .serialize_compressed(&mut expanded_bytes)
            .expect("serialize expanded setup");
        assert!(
            AkitaExpandedSetup::<F>::deserialize_compressed_exact(&expanded_bytes, &()).is_ok()
        );
        for suffix in [0, 0xa5] {
            let mut suffixed = expanded_bytes.clone();
            suffixed.push(suffix);
            assert!(AkitaExpandedSetup::<F>::deserialize_compressed_exact(&suffixed, &()).is_err());
        }
    }

    #[test]
    fn verifier_setup_rejects_prefix_registry_from_another_public_matrix() {
        let setup_seed = seed([7u8; 32]);
        let shared_matrix = derive_public_matrix_prefix::<F>(2 * D, &setup_seed.setup_seed);
        let expanded = Arc::new(
            AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                setup_seed,
                shared_matrix,
            ),
        );
        let foreign_registry =
            SetupPrefixVerifierRegistry::new(AkitaSetupSeed::shake256_paged_v1([9u8; 32]));

        let err = AkitaVerifierSetup::from_parts(expanded, foreign_registry)
            .expect_err("cross-seed prefix registry must be rejected");
        assert!(err.to_string().contains("different public matrix"));
    }

    #[test]
    fn strict_verifier_setup_decode_rejects_matrix_not_derived_from_seed() {
        let descriptor = seed([7u8; 32]);
        let setup_seed = descriptor.setup_seed.clone();
        let wrong_seed = AkitaSetupSeed::shake256_paged_v1([9u8; 32]);
        let wrong_matrix = derive_public_matrix_prefix::<F>(2 * D, &wrong_seed);
        let setup = AkitaVerifierSetup {
            expanded: Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    descriptor,
                    wrong_matrix,
                ),
            ),
            prefix_slots: SetupPrefixVerifierRegistry::new(setup_seed),
        };

        let mut bytes = Vec::new();
        setup.serialize_compressed(&mut bytes).unwrap();
        let err = AkitaVerifierSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(err
            .to_string()
            .contains("setup shared_matrix does not match public matrix seed"));
    }

    #[test]
    fn strict_verifier_setup_decode_rejects_truncated_seed_prefix_matrix() {
        let descriptor = seed([7u8; 32]);
        let setup_seed = descriptor.setup_seed.clone();
        let short_matrix = derive_public_matrix_prefix::<F>(D, &setup_seed);
        let setup = AkitaVerifierSetup {
            expanded: Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    descriptor,
                    short_matrix,
                ),
            ),
            prefix_slots: SetupPrefixVerifierRegistry::new(setup_seed),
        };

        let mut bytes = Vec::new();
        setup.serialize_compressed(&mut bytes).unwrap();
        let err = AkitaVerifierSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(err
            .to_string()
            .contains("flat matrix field count does not match expected setup shape"));
    }

    #[test]
    fn strict_setup_decode_rejects_matrix_shape_before_payload() {
        let setup_seed = seed([7u8; 32]);
        let mut bytes = Vec::new();
        setup_seed.serialize_compressed(&mut bytes).unwrap();
        usize::MAX.serialize_compressed(&mut bytes).unwrap();
        let err = AkitaExpandedSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(err
            .to_string()
            .contains("flat matrix field count does not match expected setup shape"));
    }

    #[test]
    fn setup_seed_validity_is_not_the_generic_decode_allocation_cap() {
        let setup_seed = AkitaSetupDescriptor {
            max_num_vars: 32,
            max_num_batched_polys: 1,
            num_field_elements: MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS + 1,
            setup_seed: [7u8; 32].into(),
        };

        setup_seed.check().unwrap();
        assert!(setup_seed.num_field_elements > MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS);
    }

    #[test]
    fn generic_setup_decode_still_rejects_shapes_above_allocation_cap() {
        let setup_seed = AkitaSetupDescriptor {
            max_num_vars: 32,
            max_num_batched_polys: 1,
            num_field_elements: MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS + 1,
            setup_seed: [7u8; 32].into(),
        };
        let mut bytes = Vec::new();
        setup_seed.serialize_compressed(&mut bytes).unwrap();

        let err = AkitaExpandedSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(matches!(
            err,
            SerializationError::LengthLimitExceeded { max, .. }
                if max == MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS
        ));
    }

    #[test]
    fn flat_derivation_is_deterministic_for_same_seed() {
        let seed = AkitaSetupSeed::shake256_paged_v1([42u8; 32]);
        let m1 = derive_public_matrix_prefix::<SmallF>(15 * SMALL_D, &seed);
        let m2 = derive_public_matrix_prefix::<SmallF>(15 * SMALL_D, &seed);
        assert_eq!(m1, m2);
    }

    #[test]
    fn flat_derivation_is_prefix_stable() {
        let seed = AkitaSetupSeed::shake256_paged_v1([7u8; 32]);
        let small = derive_public_matrix_prefix::<SmallF>(6 * SMALL_D, &seed);
        let large = derive_public_matrix_prefix::<SmallF>(24 * SMALL_D, &seed);
        let small_view = small.ring_view::<SMALL_D>(1, 6).unwrap();
        let large_view = large.ring_view::<SMALL_D>(1, 6).unwrap();
        for c in 0..6 {
            assert_eq!(small_view.row(0).unwrap()[c], large_view.row(0).unwrap()[c]);
        }
    }

    #[test]
    fn flat_derivation_matches_sequential_page_stream() {
        let seed = AkitaSetupSeed::shake256_paged_v1([5u8; 32]);
        let got = derive_public_matrix_prefix::<SmallF>(6 * SMALL_D, &seed);
        let mut page = SetupSeedPageXof::new::<SmallF>(&seed, 0);
        let expected = (0..6 * SMALL_D)
            .map(|_| SmallF::random(&mut page))
            .collect::<Vec<_>>();

        assert_eq!(got.as_field_slice(), expected.as_slice());
    }

    #[test]
    fn flat_derivation_is_independent_of_ring_dimension() {
        let seed = AkitaSetupSeed::shake256_paged_v1([17u8; 32]);
        let d64 = derive_public_matrix_prefix::<SmallF>(8 * 64, &seed);
        let d128 = derive_public_matrix_prefix::<SmallF>(4 * 128, &seed);

        assert_eq!(d64.as_field_slice(), d128.as_field_slice());
    }

    #[test]
    fn flat_derivation_is_prefix_stable_across_page_boundary() {
        let seed = AkitaSetupSeed::shake256_paged_v1([23u8; 32]);
        let page_field_elements = seed.derivation.page_field_elements();
        let through_page_zero = derive_public_matrix_prefix::<SmallF>(page_field_elements, &seed);
        let into_page_one = derive_public_matrix_prefix::<SmallF>(page_field_elements + 64, &seed);

        assert_eq!(
            through_page_zero.as_field_slice(),
            &into_page_one.as_field_slice()[..page_field_elements]
        );
    }

    #[test]
    fn paged_derivation_golden_vector() {
        let seed = AkitaSetupSeed::shake256_paged_v1([31u8; 32]);
        fn samples<F: Field + CanonicalEncoding>(seed: &AkitaSetupSeed) -> [u128; 3] {
            let page_field_elements = seed.derivation.page_field_elements();
            let derived = derive_public_matrix_prefix::<F>(page_field_elements + 64, seed);
            let canonical = derived
                .as_field_slice()
                .iter()
                .map(|value| {
                    value
                        .to_u128_checked()
                        .expect("Akita field element must fit in u128")
                })
                .collect::<Vec<_>>();
            [
                canonical[0],
                canonical[page_field_elements - 1],
                canonical[page_field_elements],
            ]
        }

        assert_eq!(
            samples::<Prime32Offset99>(&seed),
            [985_701_565, 215_851_758, 196_317_274]
        );
        assert_eq!(
            samples::<Prime64Offset59>(&seed),
            [
                15_459_661_060_209_904_737,
                1_106_841_764_157_043_686,
                11_841_567_322_073_738_392,
            ]
        );
        assert_eq!(
            samples::<Prime128Offset275>(&seed),
            [
                9_840_922_769_526_400_152_209_492_837_491_680_711,
                87_058_165_705_274_552_119_584_186_413_843_782_366,
                214_441_952_004_995_181_775_787_633_634_410_275_750,
            ]
        );
    }

    #[test]
    fn page_xof_binds_the_field_modulus() {
        type OtherF = Fp32<4294967291>;

        let seed = AkitaSetupSeed::shake256_paged_v1([29u8; 32]);
        let mut small = SetupSeedPageXof::new::<SmallF>(&seed, 0);
        let mut other = SetupSeedPageXof::new::<OtherF>(&seed, 0);
        let mut small_bytes = [0u8; 32];
        let mut other_bytes = [0u8; 32];
        small.fill_bytes(&mut small_bytes);
        other.fill_bytes(&mut other_bytes);

        assert_ne!(small_bytes, other_bytes);
    }

    #[test]
    fn different_shapes_from_same_flat() {
        let seed = AkitaSetupSeed::shake256_paged_v1([13u8; 32]);
        let flat = derive_public_matrix_prefix::<SmallF>(12 * SMALL_D, &seed);
        let view_3x4 = flat.ring_view::<SMALL_D>(3, 4).unwrap();
        let view_2x6 = flat.ring_view::<SMALL_D>(2, 6).unwrap();

        assert_eq!(view_3x4.row(0).unwrap()[0], view_2x6.row(0).unwrap()[0]);
        assert_eq!(view_3x4.row(0).unwrap()[3], view_2x6.row(0).unwrap()[3]);
        assert_ne!(view_3x4.row(1).unwrap()[0], view_2x6.row(1).unwrap()[0]);
    }
}
