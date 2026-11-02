//! Catalog byte storage with unique, sorted (K, geometry) keys.
//! Filesystem and deserialization inputs share the same collection validation.

use crate::configs::{AkitaChunkProfile, JoltDenseBounded, JoltDenseFull};
use crate::one_hot_family::{with_one_hot_family, OneHotFamily};
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_pcs::{AkitaCommitmentScheme, AkitaError};
use akita_schedules::ValidatedScheduleCatalog;
use jolt_openings::OpeningsError;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

/// Runtime bytes for Jolt's base schedule families.
///
/// These bytes are ordinary input data. They are intentionally neither
/// generated Rust nor embedded with `include_bytes!`.
///
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "ScheduleArtifactInputs")]
pub struct AkitaScheduleArtifacts {
    dense_profile: AkitaChunkProfile,
    dense: Vec<u8>,
    full_dense: Vec<u8>,
    one_hot: Vec<AkitaOneHotScheduleArtifact>,
}

/// One application-supplied catalog and the trusted geometry expected at admission.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaOneHotScheduleArtifact {
    pub k: usize,
    pub profile: AkitaChunkProfile,
    pub bytes: Vec<u8>,
}

impl AkitaScheduleArtifacts {
    const DIRECTORY_ENV: &'static str = "JOLT_AKITA_SCHEDULE_DIR";

    /// Validate geometry and normalize unique catalog keys. Bytes are audited on admission.
    pub fn new(
        dense_profile: AkitaChunkProfile,
        dense: Vec<u8>,
        full_dense: Vec<u8>,
        one_hot: Vec<AkitaOneHotScheduleArtifact>,
    ) -> Result<Self, AkitaError> {
        dense_profile.witness_cfg().validate()?;
        let mut one_hot = one_hot;
        for entry in &one_hot {
            let _family = OneHotFamily::from_parts(entry.k, entry.profile)?;
        }
        one_hot.sort_by_key(|entry| (entry.k, entry.profile.chunks, entry.profile.levels));
        for pair in one_hot.windows(2) {
            if let [a, b] = pair {
                if (a.k, a.profile) == (b.k, b.profile) {
                    return Err(AkitaError::InvalidSetup(format!(
                        "duplicate one-hot catalog for K={} profile {}",
                        a.k, a.profile
                    )));
                }
            }
        }
        Ok(Self {
            dense_profile,
            dense,
            full_dense,
            one_hot,
        })
    }

    /// Load dense catalogs and discover one-hot catalogs named
    /// `jolt-fp128-onehot-k<K>-w<chunks>r<levels>.aks`.
    /// The unsuffixed one-hot filename denotes one chunk and zero chunked folds.
    /// `dense_profile` is the caller-trusted geometry for both dense catalogs.
    pub fn from_directory(
        directory: impl AsRef<Path>,
        dense_profile: AkitaChunkProfile,
    ) -> Result<Self, OpeningsError> {
        let directory = directory.as_ref();
        let read = |path: &Path| {
            std::fs::read(path).map_err(|error| {
                OpeningsError::InvalidSetup(format!(
                    "read Akita schedule artifact {}: {error}",
                    path.display()
                ))
            })
        };
        let mut one_hot = Vec::new();
        let entries = std::fs::read_dir(directory).map_err(|error| {
            OpeningsError::InvalidSetup(format!("read {}: {error}", directory.display()))
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| OpeningsError::InvalidSetup(error.to_string()))?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(stem) = name
                .strip_prefix("jolt-fp128-onehot-k")
                .and_then(|name| name.strip_suffix(".aks"))
            else {
                continue;
            };
            let (k, geometry) = stem
                .split_once('-')
                .map_or((stem, None), |(k, geometry)| (k, Some(geometry)));
            let invalid =
                || OpeningsError::InvalidSetup(format!("invalid one-hot catalog filename {name}"));
            let k = k.parse::<usize>().map_err(|_| invalid())?;
            let profile = match geometry {
                None => AkitaChunkProfile::default(),
                Some(geometry) => {
                    let (chunks, levels) = geometry
                        .strip_prefix('w')
                        .and_then(|g| g.split_once('r'))
                        .ok_or_else(invalid)?;
                    AkitaChunkProfile::new(
                        chunks.parse().map_err(|_| invalid())?,
                        levels.parse().map_err(|_| invalid())?,
                    )
                }
            };
            one_hot.push(AkitaOneHotScheduleArtifact {
                k,
                profile,
                bytes: read(&entry.path())?,
            });
        }
        Self::new(
            dense_profile,
            read(&directory.join(format!("{}.aks", JoltDenseBounded::schedule_family_name())))?,
            read(&directory.join(format!("{}.aks", JoltDenseFull::schedule_family_name())))?,
            one_hot,
        )
        .map_err(|error| OpeningsError::InvalidSetup(error.to_string()))
    }

    /// The `schedules/` directory packaged with this crate: the fallback the
    /// default loader uses when `JOLT_AKITA_SCHEDULE_DIR` is unset, and the
    /// fixed location the checked-in catalog guards read directly, because
    /// they validate the committed artifacts rather than whatever an override
    /// points at.
    pub fn packaged_directory() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("schedules")
    }

    /// Host/dev loader in the shared-handle form every setup call takes: the
    /// one way hosts, benches, and tests load a bundle they intend to hand to
    /// [`crate::AkitaSetupParams`]. Reads `JOLT_AKITA_SCHEDULE_DIR`, falling back to
    /// [`Self::packaged_directory`]. The caller supplies the trusted dense geometry.
    ///
    /// The handle is what is shared, not the bytes: every call re-reads the
    /// dense `.aks` files and all discovered one-hot catalogs, so hosts
    /// still load once at preprocessing and pass
    /// the bundle to each setup. Callers that must compare setup provenance
    /// keep their own handle rather than calling this twice — the Akita
    /// prover's advice guards test bundle identity with `Arc::ptr_eq`.
    ///
    /// Panics rather than returning: the packaged `schedules/` directory is
    /// part of the installation, so a missing or unreadable catalog is a
    /// broken deployment, not a condition any caller can act on. This is host
    /// and deployment-side only; nothing on the verify path reaches it.
    /// Deployments that must handle a missing catalog load a versioned,
    /// deployment-owned path with [`Self::from_directory`] instead.
    #[expect(
        clippy::expect_used,
        reason = "an unreadable packaged schedule directory is a broken installation"
    )]
    pub fn shared_from_default_directory(dense_profile: AkitaChunkProfile) -> Arc<Self> {
        let directory = std::env::var_os(Self::DIRECTORY_ENV)
            .map_or_else(Self::packaged_directory, PathBuf::from);
        Arc::new(
            Self::from_directory(directory, dense_profile)
                .expect("the external Akita schedule artifacts must load"),
        )
    }

    pub fn dense_profile(&self) -> AkitaChunkProfile {
        self.dense_profile
    }

    pub fn dense_catalog(&self) -> Result<ValidatedScheduleCatalog, AkitaError> {
        TrustedScheduleCatalog::<JoltDenseBounded>::from_artifact_bytes(
            &self.dense,
            self.dense_profile.witness_cfg(),
        )
        .map(|catalog| catalog.catalog().clone())
    }

    pub fn full_dense_catalog(&self) -> Result<ValidatedScheduleCatalog, AkitaError> {
        TrustedScheduleCatalog::<JoltDenseFull>::from_artifact_bytes(
            &self.full_dense,
            self.dense_profile.witness_cfg(),
        )
        .map(|catalog| catalog.catalog().clone())
    }

    pub(crate) fn full_dense_scheme(
        &self,
    ) -> Result<AkitaCommitmentScheme<JoltDenseFull>, AkitaError> {
        AkitaCommitmentScheme::<JoltDenseFull>::from_schedule_artifact(
            &self.full_dense,
            self.dense_profile.witness_cfg(),
        )
    }

    pub fn one_hot_catalog(
        &self,
        one_hot_k: usize,
    ) -> Result<ValidatedScheduleCatalog, AkitaError> {
        self.one_hot_catalog_for_profile(one_hot_k, AkitaChunkProfile::new(1, 0))
    }

    /// Catalog inputs available for selection; each entry is audited when admitted.
    pub fn one_hot_artifacts(&self) -> &[AkitaOneHotScheduleArtifact] {
        &self.one_hot
    }

    pub fn one_hot_catalog_for_profile(
        &self,
        one_hot_k: usize,
        profile: AkitaChunkProfile,
    ) -> Result<ValidatedScheduleCatalog, AkitaError> {
        let family = OneHotFamily::from_parts(one_hot_k, profile)?;
        let index = self
            .one_hot
            .binary_search_by_key(&(one_hot_k, profile.chunks, profile.levels), |entry| {
                (entry.k, entry.profile.chunks, entry.profile.levels)
            })
            .map_err(|_| {
                AkitaError::InvalidSetup(format!(
                    "missing one-hot catalog for K={one_hot_k} profile {profile}"
                ))
            })?;
        let entry = self.one_hot.get(index).ok_or_else(|| {
            AkitaError::InvalidSetup(
                "catalog index is outside the normalized collection".to_owned(),
            )
        })?;
        with_one_hot_family!(family, |Cfg| {
            TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(&entry.bytes, profile.witness_cfg())
                .map(|catalog| catalog.catalog().clone())
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleArtifactInputs {
    dense_profile: AkitaChunkProfile,
    dense: Vec<u8>,
    full_dense: Vec<u8>,
    one_hot: Vec<AkitaOneHotScheduleArtifact>,
}

impl TryFrom<ScheduleArtifactInputs> for AkitaScheduleArtifacts {
    type Error = AkitaError;

    fn try_from(input: ScheduleArtifactInputs) -> Result<Self, Self::Error> {
        Self::new(
            input.dense_profile,
            input.dense,
            input.full_dense,
            input.one_hot,
        )
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "catalog transport fixtures must construct successfully"
)]
mod tests;
