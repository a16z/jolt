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
    BOnlyPadded56V1,
}

impl OuterBindingPlan {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BOnlyV1 => "b_only_v1",
            Self::BOnlyPadded56V1 => "b_only_padded_56_v1",
        }
    }

    pub fn from_id(value: &str) -> Option<Self> {
        match value {
            "b_only_v1" => Some(Self::BOnlyV1),
            "b_only_padded_56_v1" => Some(Self::BOnlyPadded56V1),
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
            || Self::has_external_source_reference(source.as_bytes())
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

    fn has_external_source_reference(source: &[u8]) -> bool {
        if source.starts_with(b"\xef\xbb\xbf")
            || source.contains(&b'\r')
            || source.contains(&b'"')
            || source.contains(&b'\'')
            || source.windows(3).any(|window| {
                window.starts_with(b"??")
                    && matches!(
                        window[2],
                        b'=' | b'/' | b'\'' | b'(' | b')' | b'!' | b'<' | b'>' | b'-'
                    )
            })
        {
            return true;
        }
        let mut spliced = Vec::with_capacity(source.len());
        let mut index = 0;
        while index < source.len() {
            if source[index] == b'\\' && source.get(index + 1) == Some(&b'\n') {
                index += 2;
            } else {
                spliced.push(source[index]);
                index += 1;
            }
        }

        let mut normalized = Vec::with_capacity(spliced.len());
        index = 0;
        while index < spliced.len() {
            if spliced[index..].starts_with(b"/*") {
                index += 2;
                while index < spliced.len() && !spliced[index..].starts_with(b"*/") {
                    index += 1;
                }
                index = (index + 2).min(spliced.len());
            } else if spliced[index..].starts_with(b"//") {
                index += 2;
                while index < spliced.len() && spliced[index] != b'\n' {
                    index += 1;
                }
            } else if spliced[index..].starts_with(b"%:") {
                normalized.push(b'#');
                index += 2;
            } else {
                normalized.push(spliced[index]);
                index += 1;
            }
        }
        if [b"__has_include".as_slice(), b"__has_embed".as_slice()]
            .iter()
            .any(|needle| {
                normalized
                    .windows(needle.len())
                    .any(|window| window == *needle)
            })
        {
            return true;
        }
        normalized.split(|byte| *byte == b'\n').any(|line| {
            let line = line
                .iter()
                .copied()
                .skip_while(u8::is_ascii_whitespace)
                .collect::<Vec<_>>();
            let Some(directive) = line.strip_prefix(b"#") else {
                return false;
            };
            let directive = directive
                .iter()
                .copied()
                .skip_while(u8::is_ascii_whitespace)
                .collect::<Vec<_>>();
            directive.starts_with(b"include")
                || directive.starts_with(b"import")
                || directive.starts_with(b"embed")
        })
    }

    #[doc(hidden)]
    pub fn embedded(binding_plan: OuterBindingPlan) -> Result<Self, MetalError> {
        Self::new(super::shader::PADDED_56_SOURCE.to_owned(), binding_plan)
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
            OuterBindingPlan::BOnlyV1,
        )
        .unwrap();

        assert_eq!(artifact.binding_plan(), OuterBindingPlan::BOnlyV1);
        assert_ne!(artifact.source_sha256(), [0; 32]);
        assert_eq!(artifact.source(), "kernel void candidate() {}");
    }

    #[test]
    fn binding_plan_ids_are_stable() {
        for (id, plan) in [
            ("b_only_v1", OuterBindingPlan::BOnlyV1),
            ("b_only_padded_56_v1", OuterBindingPlan::BOnlyPadded56V1),
        ] {
            assert_eq!(plan.as_str(), id);
            assert_eq!(OuterBindingPlan::from_id(id), Some(plan));
        }
        assert_eq!(OuterBindingPlan::from_id("split_ab_v1"), None);
    }

    #[test]
    fn artifact_rejects_unclosed_sources() {
        for source in [
            String::new(),
            "kernel\0void candidate() {}".to_owned(),
            "x".repeat(OuterKernelArtifact::MAX_SOURCE_BYTES + 1),
            "# include \"/tmp/unsealed.metal\"".to_owned(),
            "#inc\\\nlude \"/tmp/unsealed.metal\"".to_owned(),
            "#inc/**/lude \"/tmp/unsealed.metal\"".to_owned(),
            "#if __has_include(\"/tmp/unsealed.metal\")".to_owned(),
            "%:include \"/tmp/unsealed.metal\"".to_owned(),
            "??=include \"/tmp/unsealed.metal\"".to_owned(),
            "\u{feff}#include \"/tmp/unsealed.metal\"".to_owned(),
            "kernel void candidate() {}\r#include \"/tmp/unsealed.metal\"".to_owned(),
            "constant char *a = \"/*\";\n#include <unsealed.metal>\nconstant char *b = \"*/\";"
                .to_owned(),
            "// /*\n#include <unsealed.metal>\n// */".to_owned(),
            "#inc/\\\n*comment*/lude <unsealed.metal>".to_owned(),
        ] {
            assert!(matches!(
                OuterKernelArtifact::new(source, OuterBindingPlan::BOnlyV1),
                Err(MetalError::InvalidOuterArtifactSource)
            ));
        }
    }
}
