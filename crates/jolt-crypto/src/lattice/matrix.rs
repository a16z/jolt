//! Deterministic derivation of Ajtai public matrices from a public seed.
//!
//! The matrices form one flat stream of field elements. Consumers reshape
//! prefixes of it into role-specific ring matrices; any prefix of a uniformly
//! random vector is uniformly random, so every such view is binding.

use crate::CryptoError;
use jolt_algebra::parallel::*;
use jolt_algebra::{field_modulus_be_bytes, CanonicalEncoding, Field};
use rand_core::{CryptoRng, Error, RngCore};
use shake::digest::{ExtendableOutput, Update, XofReader};
use shake::Shake256;

const PUBLIC_MATRIX_DOMAIN: &[u8] = b"akita/commitment/public-field-stream";
const PUBLIC_MATRIX_DERIVATION_TAG: &[u8] = b"shake256-paged-v1";

/// Versioned derivation algorithm for the public field stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicMatrixDerivation {
    /// Fixed 4096-field-element SHAKE256 pages with exact field sampling.
    Shake256PagedV1,
}

impl PublicMatrixDerivation {
    /// Canonical one-byte tag of [`Self::Shake256PagedV1`].
    pub const SHAKE256_PAGED_V1_TAG: u8 = 1;

    /// Canonical one-byte tag naming this derivation.
    #[must_use]
    pub const fn tag(self) -> u8 {
        match self {
            Self::Shake256PagedV1 => Self::SHAKE256_PAGED_V1_TAG,
        }
    }

    /// Derivation named by a canonical tag, or `None` for an unknown tag.
    #[must_use]
    pub const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            Self::SHAKE256_PAGED_V1_TAG => Some(Self::Shake256PagedV1),
            _ => None,
        }
    }

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
pub struct PublicMatrixSeed {
    /// Coefficient derivation algorithm.
    pub derivation: PublicMatrixDerivation,
    /// Public entropy absorbed by the derivation algorithm.
    pub seed: [u8; 32],
}

impl PublicMatrixSeed {
    /// Construct a v1 paged SHAKE256 public-matrix identity.
    #[must_use]
    pub const fn shake256_paged_v1(seed: [u8; 32]) -> Self {
        Self {
            derivation: PublicMatrixDerivation::Shake256PagedV1,
            seed,
        }
    }

    /// Length of [`Self::to_canonical_bytes`].
    pub const CANONICAL_LEN: usize = 33;

    /// Canonical encoding: the derivation tag, then the 32 seed bytes. Setup
    /// digests hash exactly these bytes.
    #[must_use]
    pub fn to_canonical_bytes(&self) -> [u8; Self::CANONICAL_LEN] {
        let mut bytes = [self.derivation.tag(); Self::CANONICAL_LEN];
        for (byte, &seed_byte) in bytes.iter_mut().skip(1).zip(&self.seed) {
            *byte = seed_byte;
        }
        bytes
    }

    /// Decode [`Self::to_canonical_bytes`].
    ///
    /// Returns an error for an unknown derivation tag.
    pub fn from_canonical_bytes(bytes: &[u8; Self::CANONICAL_LEN]) -> Result<Self, CryptoError> {
        let [tag, seed @ ..] = *bytes;
        let derivation = PublicMatrixDerivation::from_tag(tag).ok_or_else(|| {
            CryptoError::InvalidInput(format!("unsupported public matrix derivation tag {tag}"))
        })?;
        Ok(Self { derivation, seed })
    }
}

impl From<[u8; 32]> for PublicMatrixSeed {
    fn from(seed: [u8; 32]) -> Self {
        Self::shake256_paged_v1(seed)
    }
}

/// Derive an exact flat prefix of public field elements from a seed.
///
/// The coefficient stream uses the page size owned by the selected derivation
/// policy. Ring dimensions are not absorbed into the XOF and do not affect
/// this derivation, so equal field-length requests derive identical
/// coefficient prefixes for every consumer shape.
///
/// Each page owns one SHAKE256 stream and repeated [`Field::random`] calls
/// consume that stream sequentially. Pages may be derived in parallel, while
/// concatenation in page-index order preserves deterministic prefix semantics.
///
/// Returns an error if the field modulus exceeds 256 bits.
pub fn derive_public_matrix_prefix<F: Field + CanonicalEncoding>(
    num_field_elements: usize,
    id: &PublicMatrixSeed,
) -> Result<Vec<F>, CryptoError> {
    let modulus = field_modulus_be_bytes::<F>()?;
    let mut data = vec![F::zero(); num_field_elements];
    cfg_chunks_mut!(data, id.derivation.page_field_elements())
        .enumerate()
        .for_each(|(page_index, coeffs)| {
            let mut page_rng = SetupSeedPageXof::new(id, &modulus, page_index);
            for coeff in coeffs.iter_mut() {
                *coeff = F::random(&mut page_rng);
            }
        });
    Ok(data)
}

/// Check that `matrix` is exactly the prefix of the public stream derived
/// from `id`.
///
/// WARNING: this checks only the coefficients present, so an empty or
/// truncated matrix passes. Callers must check the matrix length against the
/// shape they expect.
///
/// Returns an error if the field modulus exceeds 256 bits or if any
/// coefficient differs from the derived stream.
pub fn check_public_matrix_prefix<F: Field + CanonicalEncoding>(
    matrix: &[F],
    id: &PublicMatrixSeed,
) -> Result<(), CryptoError> {
    let modulus = field_modulus_be_bytes::<F>()?;
    let page_field_elements = id.derivation.page_field_elements();
    let mut expected = vec![F::zero(); page_field_elements];
    for (page_index, coeffs) in matrix.chunks(page_field_elements).enumerate() {
        let mut page_rng = SetupSeedPageXof::new(id, &modulus, page_index);
        let expected = expected.get_mut(..coeffs.len()).ok_or_else(|| {
            CryptoError::Internal("public matrix page exceeds the page size".into())
        })?;
        for value in expected.iter_mut() {
            *value = F::random(&mut page_rng);
        }
        if coeffs != expected {
            return Err(CryptoError::InvalidInput(
                "public matrix does not match its seed".into(),
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
    fn new(id: &PublicMatrixSeed, field_modulus: &[u8; 32], page_index: usize) -> Self {
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
        absorb_len_prefixed(&mut xof, b"field", field_modulus);
        absorb_len_prefixed(&mut xof, b"page", &(page_index as u64).to_le_bytes());
        Self {
            reader: xof.finalize_xof(),
        }
    }
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

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
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

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "test code"
)]
mod tests {
    use super::*;
    use jolt_algebra::{Fp32, Fp64, One, Prime128Offset275, Prime32Offset99, Prime64Offset59};

    #[test]
    fn seed_canonical_bytes_are_tag_then_seed() {
        let seed = PublicMatrixSeed::shake256_paged_v1([9u8; 32]);
        let bytes = seed.to_canonical_bytes();
        assert_eq!(bytes[0], 1);
        assert_eq!(bytes[1..], [9u8; 32]);
        assert_eq!(PublicMatrixSeed::from_canonical_bytes(&bytes), Ok(seed));

        let mut unknown = bytes;
        unknown[0] = 2;
        assert!(matches!(
            PublicMatrixSeed::from_canonical_bytes(&unknown),
            Err(CryptoError::InvalidInput(_))
        ));
    }

    type SmallF = Fp64<4_294_967_197>;
    const SMALL_D: usize = 64;

    fn page_xof<F: Field + CanonicalEncoding>(
        id: &PublicMatrixSeed,
        page_index: usize,
    ) -> SetupSeedPageXof {
        SetupSeedPageXof::new(id, &field_modulus_be_bytes::<F>().unwrap(), page_index)
    }

    #[test]
    fn flat_derivation_is_deterministic_for_same_seed() {
        let seed = PublicMatrixSeed::shake256_paged_v1([42u8; 32]);
        let m1 = derive_public_matrix_prefix::<SmallF>(15 * SMALL_D, &seed).unwrap();
        let m2 = derive_public_matrix_prefix::<SmallF>(15 * SMALL_D, &seed).unwrap();
        assert_eq!(m1, m2);
    }

    #[test]
    fn flat_derivation_matches_sequential_page_stream() {
        let seed = PublicMatrixSeed::shake256_paged_v1([5u8; 32]);
        let got = derive_public_matrix_prefix::<SmallF>(6 * SMALL_D, &seed).unwrap();
        let mut page = page_xof::<SmallF>(&seed, 0);
        let expected = (0..6 * SMALL_D)
            .map(|_| SmallF::random(&mut page))
            .collect::<Vec<_>>();

        assert_eq!(got, expected);
    }

    #[test]
    fn flat_derivation_is_prefix_stable_across_page_boundary() {
        let seed = PublicMatrixSeed::shake256_paged_v1([23u8; 32]);
        let page_field_elements = seed.derivation.page_field_elements();
        let through_page_zero =
            derive_public_matrix_prefix::<SmallF>(page_field_elements, &seed).unwrap();
        let into_page_one =
            derive_public_matrix_prefix::<SmallF>(page_field_elements + 64, &seed).unwrap();

        assert_eq!(through_page_zero, into_page_one[..page_field_elements]);
    }

    #[test]
    fn paged_derivation_golden_vector() {
        let seed = PublicMatrixSeed::shake256_paged_v1([31u8; 32]);
        fn samples<F: Field + CanonicalEncoding>(seed: &PublicMatrixSeed) -> [u128; 3] {
            let page_field_elements = seed.derivation.page_field_elements();
            let derived = derive_public_matrix_prefix::<F>(page_field_elements + 64, seed).unwrap();
            let canonical = derived
                .iter()
                .map(|value| {
                    value
                        .to_u128_checked()
                        .expect("field element must fit in u128")
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
        type OtherF = Fp32<4_294_967_291>;

        let seed = PublicMatrixSeed::shake256_paged_v1([29u8; 32]);
        let mut small = page_xof::<SmallF>(&seed, 0);
        let mut other = page_xof::<OtherF>(&seed, 0);
        let mut small_bytes = [0u8; 32];
        let mut other_bytes = [0u8; 32];
        small.fill_bytes(&mut small_bytes);
        other.fill_bytes(&mut other_bytes);

        assert_ne!(small_bytes, other_bytes);
    }

    #[test]
    fn prefix_check_rejects_any_changed_coefficient() {
        let seed = PublicMatrixSeed::shake256_paged_v1([3u8; 32]);
        let page_field_elements = seed.derivation.page_field_elements();
        let mut matrix =
            derive_public_matrix_prefix::<SmallF>(page_field_elements + 7, &seed).unwrap();
        assert_eq!(check_public_matrix_prefix(&matrix, &seed), Ok(()));

        matrix[page_field_elements + 3] += SmallF::one();
        assert!(matches!(
            check_public_matrix_prefix(&matrix, &seed),
            Err(CryptoError::InvalidInput(_))
        ));
    }
}
