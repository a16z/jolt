//! Matrix PRG backends shared by commitment/JL derivation.
//!
//! The PRG is keyed per matrix entry using domain-separated context bytes.

use aes::Aes128;
use ctr::cipher::{KeyIvInit, StreamCipher};
use rand_core::{CryptoRng, RngCore};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;

const MATRIX_PRG_DOMAIN: &[u8] = b"akita/matrix-prg";
const MATRIX_PRG_SHAKE_DOMAIN: &[u8] = b"akita/matrix-prg/shake256";
const MATRIX_PRG_AES_DOMAIN: &[u8] = b"akita/matrix-prg/aes128-ctr";

/// Stable backend identifiers for transcript/context binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MatrixPrgBackendId {
    /// SHAKE256 XOF backend.
    Shake256 = 0,
    /// AES-128-CTR backend.
    Aes128Ctr = 1,
}

impl TryFrom<u8> for MatrixPrgBackendId {
    type Error = akita_field::AkitaError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Shake256),
            1 => Ok(Self::Aes128Ctr),
            _ => Err(akita_field::AkitaError::InvalidInput(format!(
                "unknown matrix PRG backend id: {value}"
            ))),
        }
    }
}

impl From<MatrixPrgBackendId> for u8 {
    fn from(value: MatrixPrgBackendId) -> Self {
        value as u8
    }
}

/// Input context used for deterministic matrix-entry sampling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixPrgContext<'a> {
    /// Public seed.
    pub seed: &'a [u8; 32],
    /// Matrix label (`A`, `B`, `D`, etc.).
    pub matrix_label: &'a [u8],
    /// Matrix row count.
    pub rows: usize,
    /// Matrix column count.
    pub cols: usize,
    /// Matrix-entry row index.
    pub row: usize,
    /// Matrix-entry column index.
    pub col: usize,
}

/// Backend trait for matrix-entry PRG streams.
pub trait MatrixPrgBackend: Clone + Send + Sync + 'static {
    /// Stable backend identifier.
    fn backend_id(&self) -> MatrixPrgBackendId;
    /// Construct a stream RNG for one matrix entry.
    fn entry_rng(&self, context: &MatrixPrgContext<'_>) -> MatrixPrgRng;
}

/// Runtime backend selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatrixPrgBackendChoice {
    /// SHAKE256 XOF stream.
    #[default]
    Shake256,
    /// AES-128-CTR stream.
    Aes128Ctr,
}

impl MatrixPrgBackendChoice {
    /// Return the stable backend id.
    pub fn backend_id(self) -> MatrixPrgBackendId {
        match self {
            Self::Shake256 => MatrixPrgBackendId::Shake256,
            Self::Aes128Ctr => MatrixPrgBackendId::Aes128Ctr,
        }
    }

    /// Construct a stream RNG for one matrix entry.
    pub fn entry_rng(self, context: &MatrixPrgContext<'_>) -> MatrixPrgRng {
        match self {
            Self::Shake256 => Shake256Backend.entry_rng(context),
            Self::Aes128Ctr => Aes128CtrBackend.entry_rng(context),
        }
    }
}

/// SHAKE256 backend implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Shake256Backend;

impl MatrixPrgBackend for Shake256Backend {
    fn backend_id(&self) -> MatrixPrgBackendId {
        MatrixPrgBackendId::Shake256
    }

    fn entry_rng(&self, context: &MatrixPrgContext<'_>) -> MatrixPrgRng {
        MatrixPrgRng::Shake(ShakeEntryRng::new(context))
    }
}

/// AES-128-CTR backend implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Aes128CtrBackend;

impl MatrixPrgBackend for Aes128CtrBackend {
    fn backend_id(&self) -> MatrixPrgBackendId {
        MatrixPrgBackendId::Aes128Ctr
    }

    fn entry_rng(&self, context: &MatrixPrgContext<'_>) -> MatrixPrgRng {
        let (key, iv) = derive_aes_key_iv(context);
        // On aarch64, the `aes` crate uses target-feature intrinsics when
        // available; we still gate this branch for explicit architecture intent.
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("aes") {
                return MatrixPrgRng::AesCtr(Aes128CtrEntryRng::new(&key, &iv));
            }
        }
        // TODO(x86_64): add explicit AES-NI runtime path selection once CI has
        // dedicated hardware coverage. Today we use the `aes` crate default.
        #[cfg(target_arch = "x86_64")]
        {
            let _ = std::arch::is_x86_feature_detected!("aes");
        }
        MatrixPrgRng::AesCtr(Aes128CtrEntryRng::new(&key, &iv))
    }
}

/// Matrix-entry RNG wrapper over supported PRG backends.
#[allow(clippy::large_enum_variant)]
pub enum MatrixPrgRng {
    /// SHAKE256 XOF-backed RNG.
    Shake(ShakeEntryRng),
    /// AES-128-CTR-backed RNG.
    AesCtr(Aes128CtrEntryRng),
}

impl RngCore for MatrixPrgRng {
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
        match self {
            Self::Shake(rng) => rng.fill_bytes(dest),
            Self::AesCtr(rng) => rng.fill_bytes(dest),
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for MatrixPrgRng {}

/// SHAKE256-backed matrix-entry RNG.
pub struct ShakeEntryRng {
    reader: Box<dyn XofReader>,
}

impl ShakeEntryRng {
    fn new(context: &MatrixPrgContext<'_>) -> Self {
        let mut xof = Shake256::default();
        absorb_matrix_context(&mut xof, MATRIX_PRG_SHAKE_DOMAIN, context);
        Self {
            reader: Box::new(xof.finalize_xof()),
        }
    }
}

impl RngCore for ShakeEntryRng {
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
        self.reader.read(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for ShakeEntryRng {}

type AesCtrCipher = ctr::Ctr128BE<Aes128>;

/// AES-128-CTR-backed matrix-entry RNG.
pub struct Aes128CtrEntryRng {
    cipher: AesCtrCipher,
}

impl Aes128CtrEntryRng {
    fn new(key: &[u8; 16], iv: &[u8; 16]) -> Self {
        Self {
            cipher: AesCtrCipher::new(key.into(), iv.into()),
        }
    }
}

impl RngCore for Aes128CtrEntryRng {
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
        dest.fill(0u8);
        self.cipher.apply_keystream(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for Aes128CtrEntryRng {}

fn derive_aes_key_iv(context: &MatrixPrgContext<'_>) -> ([u8; 16], [u8; 16]) {
    let mut xof = Shake256::default();
    absorb_matrix_context(&mut xof, MATRIX_PRG_AES_DOMAIN, context);
    let mut out = [0u8; 32];
    xof.finalize_xof().read(&mut out);
    let key: [u8; 16] = out[..16].try_into().expect("XOF produced 32 bytes");
    let iv: [u8; 16] = out[16..].try_into().expect("XOF produced 32 bytes");
    (key, iv)
}

fn absorb_matrix_context(
    xof: &mut Shake256,
    backend_domain: &[u8],
    context: &MatrixPrgContext<'_>,
) {
    absorb_len_prefixed(xof, b"domain", MATRIX_PRG_DOMAIN);
    absorb_len_prefixed(xof, b"backend", backend_domain);
    absorb_len_prefixed(xof, b"seed", context.seed);
    absorb_len_prefixed(xof, b"matrix", context.matrix_label);
    absorb_len_prefixed(xof, b"rows", &(context.rows as u64).to_le_bytes());
    absorb_len_prefixed(xof, b"cols", &(context.cols as u64).to_le_bytes());
    absorb_len_prefixed(xof, b"row", &(context.row as u64).to_le_bytes());
    absorb_len_prefixed(xof, b"col", &(context.col as u64).to_le_bytes());
}

pub(crate) fn absorb_len_prefixed(xof: &mut Shake256, label: &[u8], data: &[u8]) {
    xof.update(&(label.len() as u64).to_le_bytes());
    xof.update(label);
    xof.update(&(data.len() as u64).to_le_bytes());
    xof.update(data);
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;

    fn context<'a>(seed: &'a [u8; 32], row: usize, col: usize) -> MatrixPrgContext<'a> {
        MatrixPrgContext {
            seed,
            matrix_label: b"A",
            rows: 4,
            cols: 5,
            row,
            col,
        }
    }

    #[test]
    fn shake_backend_is_deterministic() {
        let seed = [42u8; 32];
        let ctx = context(&seed, 1, 3);
        let mut rng1 = Shake256Backend.entry_rng(&ctx);
        let mut rng2 = Shake256Backend.entry_rng(&ctx);
        let mut a = [0u8; 96];
        let mut b = [0u8; 96];
        rng1.fill_bytes(&mut a);
        rng2.fill_bytes(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn aes_backend_is_deterministic() {
        let seed = [7u8; 32];
        let ctx = context(&seed, 0, 2);
        let mut rng1 = Aes128CtrBackend.entry_rng(&ctx);
        let mut rng2 = Aes128CtrBackend.entry_rng(&ctx);
        let mut a = [0u8; 96];
        let mut b = [0u8; 96];
        rng1.fill_bytes(&mut a);
        rng2.fill_bytes(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn row_col_changes_separate_streams() {
        let seed = [9u8; 32];
        let mut rng_a = Shake256Backend.entry_rng(&context(&seed, 0, 0));
        let mut rng_b = Shake256Backend.entry_rng(&context(&seed, 0, 1));
        let mut a = [0u8; 64];
        let mut b = [0u8; 64];
        rng_a.fill_bytes(&mut a);
        rng_b.fill_bytes(&mut b);
        assert_ne!(a, b);
    }

    #[test]
    fn backend_choice_changes_stream() {
        let seed = [5u8; 32];
        let ctx = context(&seed, 2, 4);
        let mut shake = MatrixPrgBackendChoice::Shake256.entry_rng(&ctx);
        let mut aes = MatrixPrgBackendChoice::Aes128Ctr.entry_rng(&ctx);
        let mut a = [0u8; 64];
        let mut b = [0u8; 64];
        shake.fill_bytes(&mut a);
        aes.fill_bytes(&mut b);
        assert_ne!(a, b);
    }
}
