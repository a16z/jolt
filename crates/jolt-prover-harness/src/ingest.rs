use serde::{Deserialize, Serialize};

use crate::{FeatureMode, FixtureKind, HarnessError, HarnessResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IngestionSurface {
    Sdk,
    Cli,
    HarnessCoreFixture,
    HarnessSynthetic,
}

impl IngestionSurface {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sdk => "sdk",
            Self::Cli => "cli",
            Self::HarnessCoreFixture => "harness-core-fixture",
            Self::HarnessSynthetic => "harness-synthetic",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProgramArtifactKind {
    JoltProgramExecution,
    CoreCompatibilityExecution,
    SyntheticExecution,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProverInputDescriptor {
    pub surface: IngestionSurface,
    pub artifact: ProgramArtifactKind,
    pub feature_mode: FeatureMode,
    pub fixture: Option<FixtureKind>,
    pub normalized_jolt_program_artifact: bool,
    pub uses_tracer_internals: bool,
}

impl ProverInputDescriptor {
    pub const fn sdk(feature_mode: FeatureMode) -> Self {
        Self {
            surface: IngestionSurface::Sdk,
            artifact: ProgramArtifactKind::JoltProgramExecution,
            feature_mode,
            fixture: None,
            normalized_jolt_program_artifact: true,
            uses_tracer_internals: false,
        }
    }

    pub const fn cli(feature_mode: FeatureMode) -> Self {
        Self {
            surface: IngestionSurface::Cli,
            artifact: ProgramArtifactKind::JoltProgramExecution,
            feature_mode,
            fixture: None,
            normalized_jolt_program_artifact: true,
            uses_tracer_internals: false,
        }
    }

    pub const fn harness_core_fixture(kind: FixtureKind, feature_mode: FeatureMode) -> Self {
        Self {
            surface: IngestionSurface::HarnessCoreFixture,
            artifact: ProgramArtifactKind::CoreCompatibilityExecution,
            feature_mode,
            fixture: Some(kind),
            normalized_jolt_program_artifact: true,
            uses_tracer_internals: false,
        }
    }

    pub const fn harness_synthetic_fixture(kind: FixtureKind, feature_mode: FeatureMode) -> Self {
        Self {
            surface: IngestionSurface::HarnessSynthetic,
            artifact: ProgramArtifactKind::SyntheticExecution,
            feature_mode,
            fixture: Some(kind),
            normalized_jolt_program_artifact: true,
            uses_tracer_internals: false,
        }
    }

    pub fn validate(&self) -> HarnessResult<()> {
        if !self.normalized_jolt_program_artifact {
            return Err(HarnessError::InvalidIngestion {
                surface: self.surface.as_str().to_owned(),
                reason:
                    "prover inputs must pass through the normalized jolt-program execution artifact"
                        .to_owned(),
            });
        }
        if self.uses_tracer_internals {
            return Err(HarnessError::InvalidIngestion {
                surface: self.surface.as_str().to_owned(),
                reason: "jolt-prover and jolt-witness must not consume tracer internals directly"
                    .to_owned(),
            });
        }
        if matches!(
            self.surface,
            IngestionSurface::HarnessCoreFixture | IngestionSurface::HarnessSynthetic
        ) && self.fixture.is_none()
        {
            return Err(HarnessError::InvalidIngestion {
                surface: self.surface.as_str().to_owned(),
                reason: "fixture-backed prover inputs must name a fixture".to_owned(),
            });
        }
        Ok(())
    }
}
