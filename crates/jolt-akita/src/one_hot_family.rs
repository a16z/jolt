use akita_config::CommitmentConfig;
use akita_pcs::{AkitaCommitmentScheme, AkitaError, AkitaVerifier};
use akita_types::AkitaVerifierSetup;

use crate::adapters::AkitaScheduleArtifacts;
use crate::configs::{AkitaOneHotChunkProfile, JoltOneHotK16};

pub const AKITA_ONE_HOT_K16: usize = 16;
pub const AKITA_ONE_HOT_K256: usize = 256;

macro_rules! one_hot_families {
    ($callback:ident { $($args:tt)* }) => {
        $callback! {
            $($args)*;
            (K16Single, $crate::one_hot_family::AKITA_ONE_HOT_K16, Single, $crate::configs::JoltOneHotK16, $crate::configs::JoltOneHotK16Direct, one_hot_k16),
            (K256Single, $crate::one_hot_family::AKITA_ONE_HOT_K256, Single, $crate::configs::JoltOneHotK256, $crate::configs::JoltOneHotK256Direct, one_hot_k256),
            (K16W2R2, $crate::one_hot_family::AKITA_ONE_HOT_K16, Two, $crate::configs::JoltOneHotK16W2R2, $crate::configs::JoltOneHotK16W2R2Direct, one_hot_k16_w2r2),
            (K256W2R2, $crate::one_hot_family::AKITA_ONE_HOT_K256, Two, $crate::configs::JoltOneHotK256W2R2, $crate::configs::JoltOneHotK256W2R2Direct, one_hot_k256_w2r2),
            (K16W4R2, $crate::one_hot_family::AKITA_ONE_HOT_K16, Four, $crate::configs::JoltOneHotK16W4R2, $crate::configs::JoltOneHotK16W4R2Direct, one_hot_k16_w4r2),
            (K256W4R2, $crate::one_hot_family::AKITA_ONE_HOT_K256, Four, $crate::configs::JoltOneHotK256W4R2, $crate::configs::JoltOneHotK256W4R2Direct, one_hot_k256_w4r2),
            (K16W8R2, $crate::one_hot_family::AKITA_ONE_HOT_K16, Eight, $crate::configs::JoltOneHotK16W8R2, $crate::configs::JoltOneHotK16W8R2Direct, one_hot_k16_w8r2),
            (K256W8R2, $crate::one_hot_family::AKITA_ONE_HOT_K256, Eight, $crate::configs::JoltOneHotK256W8R2, $crate::configs::JoltOneHotK256W8R2Direct, one_hot_k256_w8r2),
        }
    };
}
pub(crate) use one_hot_families;

macro_rules! define_family_types {
    (; $(($variant:ident, $k:path, $profile:ident, $cfg:ty, $direct:ty, $artifact:ident)),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum OneHotFamily {
            $($variant),+
        }

        pub(crate) enum AkitaOneHotBackendScheme {
            $($variant(AkitaCommitmentScheme<$cfg>)),+
        }

        pub(crate) enum AkitaOneHotBackendVerifier {
            $($variant(AkitaVerifier<$cfg>)),+
        }

        impl OneHotFamily {
            pub(crate) const ALL: &'static [Self] = &[$(Self::$variant),+];

            pub(crate) fn from_parts(
                k: usize,
                profile: AkitaOneHotChunkProfile,
            ) -> Result<Self, AkitaError> {
                match (k, profile) {
                    $(($k, AkitaOneHotChunkProfile::$profile) => Ok(Self::$variant)),+,
                    _ => Err(AkitaError::InvalidSetup(format!(
                        "unsupported Akita one-hot K={k} with profile {profile:?}"
                    ))),
                }
            }

            pub(crate) const fn k(self) -> usize {
                match self {
                    $(Self::$variant => $k),+
                }
            }

            pub(crate) const fn profile(self) -> AkitaOneHotChunkProfile {
                match self {
                    $(Self::$variant => AkitaOneHotChunkProfile::$profile),+
                }
            }

            pub(crate) fn family_name(self) -> &'static str {
                match self {
                    $(Self::$variant => <$cfg>::schedule_family_name()),+
                }
            }

            pub(crate) fn artifact(self, artifacts: &AkitaScheduleArtifacts) -> &[u8] {
                match self {
                    $(Self::$variant => &artifacts.$artifact),+
                }
            }
        }

        impl AkitaOneHotBackendScheme {
            pub(crate) fn from_artifact(
                family: OneHotFamily,
                bytes: &[u8],
            ) -> Result<Self, AkitaError> {
                match family {
                    $(OneHotFamily::$variant =>
                        AkitaCommitmentScheme::<$cfg>::from_schedule_artifact(bytes)
                            .map(Self::$variant)),+
                }
            }

            pub(crate) fn verifier(
                &self,
                setup: AkitaVerifierSetup<<JoltOneHotK16 as CommitmentConfig>::Field>,
            ) -> Result<AkitaOneHotBackendVerifier, AkitaError> {
                match self {
                    $(Self::$variant(scheme) => scheme.verifier(setup)
                        .map(AkitaOneHotBackendVerifier::$variant)),+
                }
            }
        }
    };
}
one_hot_families!(define_family_types {});

macro_rules! dispatch_family {
    ($family:expr, |$cfg:ident| $body:expr; $(($variant:ident, $k:path, $profile:ident, $family_cfg:ty, $direct:ty, $artifact:ident)),+ $(,)?) => {{
        match $family {
            $(
                $crate::one_hot_family::OneHotFamily::$variant => {
                    type $cfg = $family_cfg;
                    $body
                }
            ),+
        }
    }};
    ($family:expr, |$cfg:ident, $direct_cfg:ident| $body:expr; $(($variant:ident, $k:path, $profile:ident, $family_cfg:ty, $direct:ty, $artifact:ident)),+ $(,)?) => {{
        match $family {
            $(
                $crate::one_hot_family::OneHotFamily::$variant => {
                    type $cfg = $family_cfg;
                    type $direct_cfg = $direct;
                    $body
                }
            ),+
        }
    }};
}
pub(crate) use dispatch_family;

macro_rules! dispatch_scheme {
    ($scheme:expr, |$typed:ident| $body:expr; $(($variant:ident, $k:path, $profile:ident, $family_cfg:ty, $direct:ty, $artifact:ident)),+ $(,)?) => {{
        match $scheme {
            $(
                $crate::one_hot_family::AkitaOneHotBackendScheme::$variant($typed) => $body
            ),+
        }
    }};
    ($scheme:expr, |$typed:ident, $cfg:ident| $body:expr; $(($variant:ident, $k:path, $profile:ident, $family_cfg:ty, $direct:ty, $artifact:ident)),+ $(,)?) => {{
        match $scheme {
            $(
                $crate::one_hot_family::AkitaOneHotBackendScheme::$variant($typed) => {
                    type $cfg = $family_cfg;
                    $body
                }
            ),+
        }
    }};
}
pub(crate) use dispatch_scheme;

macro_rules! dispatch_verifier {
    ($verifier:expr, |$typed:ident| $body:expr; $(($variant:ident, $k:path, $profile:ident, $family_cfg:ty, $direct:ty, $artifact:ident)),+ $(,)?) => {{
        match $verifier {
            $(
                $crate::one_hot_family::AkitaOneHotBackendVerifier::$variant($typed) => $body
            ),+
        }
    }};
}
pub(crate) use dispatch_verifier;

macro_rules! with_one_hot_family {
    ($family:expr, |$cfg:ident| $body:expr) => {{
        use $crate::one_hot_family::dispatch_family;
        $crate::one_hot_family::one_hot_families!(dispatch_family { $family, |$cfg| $body })
    }};
    ($family:expr, |$cfg:ident, $direct_cfg:ident| $body:expr) => {{
        use $crate::one_hot_family::dispatch_family;
        $crate::one_hot_family::one_hot_families!(dispatch_family { $family, |$cfg, $direct_cfg| $body })
    }};
    (scheme $scheme:expr, |$typed:ident| $body:expr) => {{
        use $crate::one_hot_family::dispatch_scheme;
        $crate::one_hot_family::one_hot_families!(dispatch_scheme { $scheme, |$typed| $body })
    }};
    (scheme $scheme:expr, |$typed:ident, $cfg:ident| $body:expr) => {{
        use $crate::one_hot_family::dispatch_scheme;
        $crate::one_hot_family::one_hot_families!(dispatch_scheme { $scheme, |$typed, $cfg| $body })
    }};
    (verifier $verifier:expr, |$typed:ident| $body:expr) => {{
        use $crate::one_hot_family::dispatch_verifier;
        $crate::one_hot_family::one_hot_families!(dispatch_verifier { $verifier, |$typed| $body })
    }};
}
pub(crate) use with_one_hot_family;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_parts_round_trip() {
        for family in OneHotFamily::ALL.iter().copied() {
            assert_eq!(
                OneHotFamily::from_parts(family.k(), family.profile()),
                Ok(family)
            );
            assert!(!family.family_name().is_empty());
        }
    }
}
