#[cfg(feature = "test-utils")]
use std::sync::Arc;

#[cfg(feature = "test-utils")]
use sha2::{Digest, Sha256};

#[cfg(feature = "test-utils")]
use super::super::MetalError;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OuterBindingPlan {
    #[default]
    BOnlyV1,
    SplitAbV1,
}

impl OuterBindingPlan {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BOnlyV1 => "b_only_v1",
            Self::SplitAbV1 => "split_ab_v1",
        }
    }

    pub fn from_id(value: &str) -> Option<Self> {
        match value {
            "b_only_v1" => Some(Self::BOnlyV1),
            "split_ab_v1" => Some(Self::SplitAbV1),
            _ => None,
        }
    }
}

#[cfg(feature = "test-utils")]
#[derive(Clone)]
pub struct OuterKernelArtifact {
    source: Arc<str>,
    source_sha256: [u8; 32],
    binding_plan: OuterBindingPlan,
}

#[cfg(feature = "test-utils")]
impl OuterKernelArtifact {
    const MAX_SOURCE_BYTES: usize = 2 * 1024 * 1024;

    #[doc(hidden)]
    pub fn new(source: String, binding_plan: OuterBindingPlan) -> Result<Self, MetalError> {
        if source.is_empty()
            || source.len() > Self::MAX_SOURCE_BYTES
            || source.as_bytes().contains(&0)
        {
            return Err(MetalError::InvalidOuterArtifactSource);
        }
        let source_sha256 = Sha256::digest(source.as_bytes()).into();
        Ok(Self {
            source: Arc::from(source),
            source_sha256,
            binding_plan,
        })
    }

    #[doc(hidden)]
    pub fn embedded(binding_plan: OuterBindingPlan) -> Result<Self, MetalError> {
        Self::new(super::shader::SOURCE.to_owned(), binding_plan)
    }

    pub(crate) fn source(&self) -> &str {
        &self.source
    }

    #[doc(hidden)]
    pub const fn source_sha256(&self) -> [u8; 32] {
        self.source_sha256
    }

    #[doc(hidden)]
    pub const fn binding_plan(&self) -> OuterBindingPlan {
        self.binding_plan
    }
}

#[cfg(all(test, feature = "test-utils"))]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn artifact_closes_source_and_binding_plan() {
        let artifact = OuterKernelArtifact::new(
            "kernel void candidate() {}".to_owned(),
            OuterBindingPlan::SplitAbV1,
        )
        .unwrap();

        assert_eq!(artifact.binding_plan(), OuterBindingPlan::SplitAbV1);
        assert_ne!(artifact.source_sha256(), [0; 32]);
        assert_eq!(artifact.source(), "kernel void candidate() {}");
    }

    #[test]
    fn artifact_rejects_empty_nul_and_oversized_sources() {
        for source in [
            String::new(),
            "kernel\0void candidate() {}".to_owned(),
            "x".repeat(OuterKernelArtifact::MAX_SOURCE_BYTES + 1),
        ] {
            assert!(matches!(
                OuterKernelArtifact::new(source, OuterBindingPlan::BOnlyV1),
                Err(MetalError::InvalidOuterArtifactSource)
            ));
        }
    }
}
