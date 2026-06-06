use serde::{Deserialize, Serialize};

use crate::{FeatureMode, HarnessResult};

#[cfg(any(
    not(feature = "core-fixtures"),
    all(feature = "core-fixtures", feature = "field-inline")
))]
use crate::HarnessError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FixtureKind {
    MuldivSmall,
    FibonacciSmall,
    MemoryOps,
    Sha2Chain2Pow16,
    Sha2Chain2Pow20,
    AdviceConsumer,
    ZkMuldivSmall,
    ZkAdviceConsumer,
    FieldInlineSmall,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixtureRequest {
    pub kind: FixtureKind,
    pub feature_mode: FeatureMode,
}

impl FixtureRequest {
    pub const fn new(kind: FixtureKind, feature_mode: FeatureMode) -> Self {
        Self { kind, feature_mode }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixtureSource {
    CoreCompatibility,
    ModularSynthetic,
    External(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixtureArtifacts {
    pub kind: FixtureKind,
    pub feature_mode: FeatureMode,
    pub source: FixtureSource,
    pub notes: Vec<String>,
}

impl FixtureArtifacts {
    pub fn new(kind: FixtureKind, feature_mode: FeatureMode, source: FixtureSource) -> Self {
        Self {
            kind,
            feature_mode,
            source,
            notes: Vec::new(),
        }
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
}

pub trait FixtureProvider {
    fn load(&self, request: &FixtureRequest) -> HarnessResult<FixtureArtifacts>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StaticFixtureProvider;

impl FixtureProvider for StaticFixtureProvider {
    fn load(&self, request: &FixtureRequest) -> HarnessResult<FixtureArtifacts> {
        Ok(FixtureArtifacts::new(
            request.kind,
            request.feature_mode,
            FixtureSource::ModularSynthetic,
        )
        .with_note("static harness typed artifact; not valid for frontier acceptance"))
    }
}

#[cfg(feature = "core-fixtures")]
#[derive(Clone, Copy, Debug, Default)]
pub struct CoreFixtureProvider;

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
impl FixtureProvider for CoreFixtureProvider {
    fn load(&self, request: &FixtureRequest) -> HarnessResult<FixtureArtifacts> {
        let _fixture = crate::core_fixture::load_core_verifier_fixture(request)?;
        Ok(FixtureArtifacts::new(
            request.kind,
            request.feature_mode,
            FixtureSource::CoreCompatibility,
        )
        .with_note("converted core proof verified by the modular verifier"))
    }
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
impl FixtureProvider for CoreFixtureProvider {
    fn load(&self, request: &FixtureRequest) -> HarnessResult<FixtureArtifacts> {
        let _fixture = crate::core_zk_fixture::load_zk_core_verifier_fixture(request)?;
        Ok(FixtureArtifacts::new(
            request.kind,
            request.feature_mode,
            FixtureSource::CoreCompatibility,
        )
        .with_note("converted ZK core proof verified by the modular verifier"))
    }
}

#[cfg(all(feature = "core-fixtures", feature = "field-inline"))]
impl FixtureProvider for CoreFixtureProvider {
    fn load(&self, request: &FixtureRequest) -> HarnessResult<FixtureArtifacts> {
        Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "core verifier fixtures are not available for this feature combination yet",
        })
    }
}

#[cfg(not(feature = "core-fixtures"))]
#[derive(Clone, Copy, Debug, Default)]
pub struct CoreFixtureProvider;

#[cfg(not(feature = "core-fixtures"))]
impl FixtureProvider for CoreFixtureProvider {
    fn load(&self, request: &FixtureRequest) -> HarnessResult<FixtureArtifacts> {
        Err(HarnessError::FixtureUnavailable {
            fixture: request.kind,
            context: "core fixtures require the core-fixtures feature",
        })
    }
}
