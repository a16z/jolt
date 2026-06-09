//! Setup and SRS file handling for HyperKZG.

use std::path::{Path, PathBuf};

use jolt_crypto::{DeriveSetup, JoltGroup, PairingGroup, PedersenSetup};
use jolt_field::RandomSampling;
use serde::{de::DeserializeOwned, Serialize};

use crate::error::HyperKZGError;
use crate::scheme::HyperKZGScheme;
use crate::types::{
    HyperKZGProverSetup, HyperKZGSrsFile, HyperKZGSrsKind, HYPERKZG_SRS_NAME, HYPERKZG_SRS_VERSION,
};

impl<P: PairingGroup> HyperKZGScheme<P> {
    /// Returns the canonical filename for a ceremony-generated HyperKZG SRS.
    ///
    /// The `k` in `hyperkzg_{k}.srs` is the exponent for the supported
    /// multilinear evaluation table length: `hyperkzg_20.srs` supports
    /// polynomials with up to `2^20` evaluations. The serialized prover setup
    /// stores one additional G1 power internally, following the KZG SRS
    /// convention used by this crate.
    #[must_use]
    pub fn srs_file_name(k: usize) -> String {
        format!("hyperkzg_{k}.srs")
    }

    /// Returns the canonical filename for a ceremony-generated ZK HyperKZG SRS.
    #[cfg(feature = "zk")]
    #[must_use]
    pub fn zk_srs_file_name(k: usize) -> String {
        format!("hyperkzg_zk_{k}.srs")
    }

    /// Returns `dir / hyperkzg_{k}.srs`.
    #[must_use]
    pub fn srs_path(dir: impl AsRef<Path>, k: usize) -> PathBuf {
        dir.as_ref().join(Self::srs_file_name(k))
    }

    /// Returns `dir / hyperkzg_zk_{k}.srs`.
    #[cfg(feature = "zk")]
    #[must_use]
    pub fn zk_srs_path(dir: impl AsRef<Path>, k: usize) -> PathBuf {
        dir.as_ref().join(Self::zk_srs_file_name(k))
    }

    pub(crate) fn supported_evaluations(setup: &HyperKZGProverSetup<P>) -> usize {
        setup.g1_powers.len().saturating_sub(1)
    }

    fn setup_srs_kind(setup: &HyperKZGProverSetup<P>) -> HyperKZGSrsKind {
        if setup.hiding_g1_sequence.is_some() {
            HyperKZGSrsKind::Zk
        } else {
            HyperKZGSrsKind::Plain
        }
    }

    fn validate_setup_kind(
        setup: &HyperKZGProverSetup<P>,
        expected: HyperKZGSrsKind,
    ) -> Result<(), HyperKZGError> {
        let got = Self::setup_srs_kind(setup);
        if got != expected {
            return Err(HyperKZGError::WrongSrsSetupKind { expected, got });
        }
        Ok(())
    }

    fn capacity_exponent(capacity: usize) -> Result<usize, HyperKZGError> {
        if !capacity.is_power_of_two() {
            return Err(HyperKZGError::SrsFileCapacityNotPowerOfTwo { capacity });
        }
        Ok(capacity.ilog2() as usize)
    }

    fn required_evaluations(k: usize) -> Result<usize, HyperKZGError> {
        if k >= usize::BITS as usize {
            return Err(HyperKZGError::SrsExponentTooLarge { k });
        }
        Ok(1usize << k)
    }

    fn validate_srs_capacity(
        setup: &HyperKZGProverSetup<P>,
        k: usize,
    ) -> Result<(), HyperKZGError> {
        let required = Self::required_evaluations(k)?;
        let supported = Self::supported_evaluations(setup);
        if supported < required {
            return Err(HyperKZGError::SrsFileCapacityMismatch {
                k,
                supported,
                required,
            });
        }
        Ok(())
    }

    fn validate_srs_file(
        path: &Path,
        file: &HyperKZGSrsFile<P>,
        expected_kind: HyperKZGSrsKind,
    ) -> Result<(), HyperKZGError> {
        if file.name != HYPERKZG_SRS_NAME {
            return Err(HyperKZGError::SrsFileNameMismatch {
                path: path.to_path_buf(),
                name: file.name.clone(),
            });
        }
        if file.version != HYPERKZG_SRS_VERSION {
            return Err(HyperKZGError::SrsFileVersionUnsupported {
                path: path.to_path_buf(),
                version: file.version,
            });
        }
        if file.kind != expected_kind {
            return Err(HyperKZGError::SrsFileKindMismatch {
                path: path.to_path_buf(),
                expected: expected_kind,
                got: file.kind,
            });
        }

        let required = Self::required_evaluations(file.k)?;
        if file.capacity != required {
            return Err(HyperKZGError::SrsFileCapacityMismatch {
                k: file.k,
                supported: file.capacity,
                required,
            });
        }

        let supported = Self::supported_evaluations(&file.setup);
        if supported < file.capacity {
            return Err(HyperKZGError::SrsFileCapacityMismatch {
                k: file.k,
                supported,
                required: file.capacity,
            });
        }

        Ok(())
    }

    fn read_srs_file_with_kind(
        path: impl AsRef<Path>,
        expected_kind: HyperKZGSrsKind,
    ) -> Result<HyperKZGProverSetup<P>, HyperKZGError>
    where
        P::G1: DeserializeOwned,
        P::G2: DeserializeOwned,
    {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| HyperKZGError::SrsFileRead {
            path: path.to_path_buf(),
            source,
        })?;
        let (file, _): (HyperKZGSrsFile<P>, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).map_err(
                |source| HyperKZGError::SrsFileDecode {
                    path: path.to_path_buf(),
                    source,
                },
            )?;
        Self::validate_srs_file(path, &file, expected_kind)?;
        Ok(file.setup)
    }

    /// Reads a bincode-encoded prover SRS from an explicit path.
    ///
    /// Production proving code should use this with a ceremony-generated file,
    /// not generate `beta` in the proving process.
    pub fn read_srs_file(path: impl AsRef<Path>) -> Result<HyperKZGProverSetup<P>, HyperKZGError>
    where
        P::G1: DeserializeOwned,
        P::G2: DeserializeOwned,
    {
        Self::read_srs_file_with_kind(path, HyperKZGSrsKind::Plain)
    }

    /// Reads a bincode-encoded ZK prover SRS from an explicit path.
    #[cfg(feature = "zk")]
    pub fn read_zk_srs_file(path: impl AsRef<Path>) -> Result<HyperKZGProverSetup<P>, HyperKZGError>
    where
        P::G1: DeserializeOwned,
        P::G2: DeserializeOwned,
    {
        Self::read_srs_file_with_kind(path, HyperKZGSrsKind::Zk)
    }

    /// Reads `dir / hyperkzg_{k}.srs` and verifies that it supports `2^k`
    /// evaluations.
    pub fn read_srs_from_dir(
        dir: impl AsRef<Path>,
        k: usize,
    ) -> Result<HyperKZGProverSetup<P>, HyperKZGError>
    where
        P::G1: DeserializeOwned,
        P::G2: DeserializeOwned,
    {
        let setup = Self::read_srs_file(Self::srs_path(dir, k))?;
        Self::validate_srs_capacity(&setup, k)?;
        Ok(setup)
    }

    /// Reads `dir / hyperkzg_zk_{k}.srs` and verifies that it supports `2^k`
    /// evaluations.
    #[cfg(feature = "zk")]
    pub fn read_zk_srs_from_dir(
        dir: impl AsRef<Path>,
        k: usize,
    ) -> Result<HyperKZGProverSetup<P>, HyperKZGError>
    where
        P::G1: DeserializeOwned,
        P::G2: DeserializeOwned,
    {
        let setup = Self::read_zk_srs_file(Self::zk_srs_path(dir, k))?;
        Self::validate_srs_capacity(&setup, k)?;
        Ok(setup)
    }

    /// Writes a bincode-encoded prover SRS to an explicit path.
    ///
    /// This helper is intended for trusted setup tooling and tests. A
    /// production prover should load the ceremony output with
    /// [`read_srs_file`](Self::read_srs_file) or
    /// [`read_srs_from_dir`](Self::read_srs_from_dir).
    pub fn write_srs_file(
        setup: &HyperKZGProverSetup<P>,
        path: impl AsRef<Path>,
    ) -> Result<(), HyperKZGError>
    where
        P::G1: Serialize,
        P::G2: Serialize,
    {
        Self::write_srs_file_with_kind(setup, path, HyperKZGSrsKind::Plain)
    }

    /// Writes a bincode-encoded ZK prover SRS to an explicit path.
    #[cfg(feature = "zk")]
    pub fn write_zk_srs_file(
        setup: &HyperKZGProverSetup<P>,
        path: impl AsRef<Path>,
    ) -> Result<(), HyperKZGError>
    where
        P::G1: Serialize,
        P::G2: Serialize,
    {
        Self::write_srs_file_with_kind(setup, path, HyperKZGSrsKind::Zk)
    }

    fn write_srs_file_with_kind(
        setup: &HyperKZGProverSetup<P>,
        path: impl AsRef<Path>,
        expected_kind: HyperKZGSrsKind,
    ) -> Result<(), HyperKZGError>
    where
        P::G1: Serialize,
        P::G2: Serialize,
    {
        Self::validate_setup_kind(setup, expected_kind)?;
        let path = path.as_ref();
        let capacity = Self::supported_evaluations(setup);
        let k = Self::capacity_exponent(capacity)?;
        let file = HyperKZGSrsFile {
            name: HYPERKZG_SRS_NAME.to_string(),
            version: HYPERKZG_SRS_VERSION,
            kind: expected_kind,
            k,
            capacity,
            setup: setup.clone(),
        };
        let bytes = bincode::serde::encode_to_vec(&file, bincode::config::standard()).map_err(
            |source| HyperKZGError::SrsFileEncode {
                path: path.to_path_buf(),
                source,
            },
        )?;
        std::fs::write(path, bytes).map_err(|source| HyperKZGError::SrsFileWrite {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Writes `dir / hyperkzg_{k}.srs` after verifying that the setup supports
    /// `2^k` evaluations.
    pub fn write_srs_to_dir(
        setup: &HyperKZGProverSetup<P>,
        dir: impl AsRef<Path>,
        k: usize,
    ) -> Result<(), HyperKZGError>
    where
        P::G1: Serialize,
        P::G2: Serialize,
    {
        Self::validate_srs_capacity(setup, k)?;
        Self::write_srs_file(setup, Self::srs_path(dir, k))
    }

    /// Writes `dir / hyperkzg_zk_{k}.srs` after verifying that the setup
    /// supports `2^k` evaluations.
    #[cfg(feature = "zk")]
    pub fn write_zk_srs_to_dir(
        setup: &HyperKZGProverSetup<P>,
        dir: impl AsRef<Path>,
        k: usize,
    ) -> Result<(), HyperKZGError>
    where
        P::G1: Serialize,
        P::G2: Serialize,
    {
        Self::validate_srs_capacity(setup, k)?;
        Self::write_zk_srs_file(setup, Self::zk_srs_path(dir, k))
    }

    /// Generates an SRS from a random generator and secret scalar.
    ///
    /// WARNING: this is suitable for tests or trusted setup tooling only. A
    /// production prover/verifier should load a ceremony-generated
    /// `hyperkzg_{k}.srs` file via [`read_srs_file`](Self::read_srs_file) or
    /// [`read_srs_from_dir`](Self::read_srs_from_dir), so the live runtime never
    /// observes the KZG trapdoor `beta`.
    ///
    /// `max_degree` is the maximum polynomial length (number of evaluations).
    /// The SRS will contain `max_degree + 1` G1 powers and 2 G2 powers.
    pub fn setup<R: rand_core::RngCore>(
        rng: &mut R,
        max_degree: usize,
        g1: P::G1,
        g2: P::G2,
    ) -> HyperKZGProverSetup<P> {
        let beta = P::ScalarField::random(rng);
        Self::setup_from_secret(beta, max_degree, g1, g2)
    }

    /// Generates SRS from a known secret.
    ///
    /// WARNING: this is not a production runtime API. It is only appropriate
    /// for deterministic tests or trusted setup tooling that destroys `beta`.
    /// Anyone who knows `beta` can break KZG binding. Production proving and
    /// verifying should import a ceremony-generated `hyperkzg_{k}.srs` file
    /// instead.
    pub fn setup_from_secret(
        beta: P::ScalarField,
        max_degree: usize,
        g1: P::G1,
        g2: P::G2,
    ) -> HyperKZGProverSetup<P> {
        let mut g1_powers = Vec::with_capacity(max_degree + 1);
        let mut cur = g1;
        for _ in 0..=max_degree {
            g1_powers.push(cur);
            cur = cur.scalar_mul(&beta);
        }

        let g2_powers = vec![g2, g2.scalar_mul(&beta)];

        HyperKZGProverSetup {
            g1_powers,
            g2_powers,
            hiding_g1_sequence: None,
        }
    }

    /// Generates a ZK-capable SRS from a random secret scalar.
    ///
    /// WARNING: this is suitable for tests or trusted setup tooling only. A
    /// production prover/verifier should load a ceremony-generated
    /// `hyperkzg_zk_{k}.srs` file so the live runtime never observes `beta`.
    #[cfg(feature = "zk")]
    pub fn setup_zk<R: rand_core::RngCore>(
        rng: &mut R,
        max_degree: usize,
        g1: P::G1,
        hiding_g1: P::G1,
        g2: P::G2,
    ) -> HyperKZGProverSetup<P> {
        let beta = P::ScalarField::random(rng);
        Self::setup_zk_from_secret(beta, max_degree, g1, hiding_g1, g2)
    }

    /// Generates a ZK-capable SRS from a known secret.
    ///
    /// WARNING: this is not a production runtime API. It is only appropriate
    /// for deterministic tests or trusted setup tooling that destroys `beta`.
    #[cfg(feature = "zk")]
    pub fn setup_zk_from_secret(
        beta: P::ScalarField,
        max_degree: usize,
        g1: P::G1,
        hiding_g1: P::G1,
        g2: P::G2,
    ) -> HyperKZGProverSetup<P> {
        let mut g1_powers = Vec::with_capacity(max_degree + 1);
        let mut hiding_g1_sequence = Vec::with_capacity(max_degree + 1);
        let mut cur_g1 = g1;
        let mut cur_hiding_g1 = hiding_g1;
        for _ in 0..=max_degree {
            g1_powers.push(cur_g1);
            hiding_g1_sequence.push(cur_hiding_g1);
            cur_g1 = cur_g1.scalar_mul(&beta);
            cur_hiding_g1 = cur_hiding_g1.scalar_mul(&beta);
        }

        let g2_powers = vec![g2, g2.scalar_mul(&beta)];

        HyperKZGProverSetup {
            g1_powers,
            g2_powers,
            hiding_g1_sequence: Some(hiding_g1_sequence),
        }
    }
}

/// # Security note
///
/// Uses KZG SRS powers as Pedersen generators. Pedersen binding shares the
/// KZG trapdoor `beta`; both are sound once `beta` is destroyed, but the two
/// schemes do not have independent security assumptions.
impl<P: PairingGroup> DeriveSetup<HyperKZGProverSetup<P>> for PedersenSetup<P::G1> {
    fn derive(source: &HyperKZGProverSetup<P>, capacity: usize) -> Self {
        assert!(
            source.g1_powers.len() > capacity,
            "SRS has {} G1 powers, need at least {} (capacity + 1 for blinding)",
            source.g1_powers.len(),
            capacity + 1,
        );
        let message_generators = source.g1_powers[..capacity].to_vec();
        let blinding_generator = source.g1_powers[capacity];
        PedersenSetup::new(message_generators, blinding_generator)
    }
}
