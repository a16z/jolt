use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_pcs::{AkitaCommitmentScheme, AkitaError, AkitaVerifier};
use akita_types::AkitaVerifierSetup;

use crate::configs::{AkitaChunkProfile, JoltOneHotK16};
use crate::family_contract::OneHotFamilyContract;
pub use crate::family_contract::{AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256};

macro_rules! one_hot_families {
    ($callback:ident { $($args:tt)* }) => {
        $callback! {
            $($args)*;
            (K16, $crate::one_hot_family::AKITA_ONE_HOT_K16, $crate::configs::JoltOneHotK16),
            (K256, $crate::one_hot_family::AKITA_ONE_HOT_K256, $crate::configs::JoltOneHotK256),
        }
    };
}
pub(crate) use one_hot_families;

macro_rules! define_family_types {
    (; $(($variant:ident, $k:path, $cfg:ty)),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum OneHotFamilyKind {
            $($variant),+
        }

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) struct OneHotFamily {
            pub(crate) kind: OneHotFamilyKind,
            profile: AkitaChunkProfile,
        }

        pub(crate) enum AkitaOneHotBackendScheme {
            $($variant(AkitaCommitmentScheme<$cfg>)),+
        }

        pub(crate) enum AkitaOneHotBackendVerifier {
            $($variant(AkitaVerifier<$cfg>)),+
        }

        impl OneHotFamily {
            pub(crate) fn from_parts(
                k: usize,
                profile: AkitaChunkProfile,
            ) -> Result<Self, AkitaError> {
                profile.witness_cfg().validate()?;
                match k {
                    $($k => Ok(Self { kind: OneHotFamilyKind::$variant, profile })),+,
                    _ => Err(AkitaError::InvalidSetup(format!(
                        "unsupported Akita one-hot K={k} with profile {profile:?}"
                    ))),
                }
            }

            pub(crate) const fn k(self) -> usize {
                match self.kind {
                    $(OneHotFamilyKind::$variant => $k),+
                }
            }

            pub(crate) const fn profile(self) -> AkitaChunkProfile {
                self.profile
            }

            pub(crate) fn num_vars_range(self, num_polys: usize) -> (usize, usize) {
                let contract = if self.k() == AKITA_ONE_HOT_K16 {
                    OneHotFamilyContract::K16(self.profile())
                } else {
                    OneHotFamilyContract::K256(self.profile())
                };
                contract.num_vars_range(num_polys)
            }

            pub(crate) fn validate_num_vars(
                self,
                num_vars: usize,
                num_polys: usize,
            ) -> Result<(), AkitaError> {
                let (min, max) = self.num_vars_range(num_polys);
                if !(min..=max).contains(&num_vars) {
                    return Err(AkitaError::InvalidSetup(format!(
                        "one-hot K={} profile {:?} final arity {num_vars} with {num_polys} polynomials is outside the supported range {min}..={max}",
                        self.k(), self.profile(),
                    )));
                }
                Ok(())
            }


        }

        impl AkitaOneHotBackendScheme {
            pub(crate) fn from_artifact(
                family: OneHotFamily,
                bytes: &[u8],
            ) -> Result<Self, AkitaError> {
                match family.kind {
                    $(OneHotFamilyKind::$variant =>
                        TrustedScheduleCatalog::<$cfg>::from_artifact_bytes(
                            bytes, family.profile().witness_cfg(),
                        ).map(AkitaCommitmentScheme::new).map(Self::$variant)),+
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
    ($family:expr, |$cfg:ident| $body:expr; $(($variant:ident, $k:path, $family_cfg:ty)),+ $(,)?) => {{
        match $family.kind {
            $(
                $crate::one_hot_family::OneHotFamilyKind::$variant => {
                    type $cfg = $family_cfg;
                    $body
                }
            ),+
        }
    }};

}
pub(crate) use dispatch_family;

macro_rules! dispatch_scheme {
    ($scheme:expr, |$typed:ident| $body:expr; $(($variant:ident, $k:path, $family_cfg:ty)),+ $(,)?) => {{
        match $scheme {
            $(
                $crate::one_hot_family::AkitaOneHotBackendScheme::$variant($typed) => $body
            ),+
        }
    }};
    ($scheme:expr, |$typed:ident, $cfg:ident| $body:expr; $(($variant:ident, $k:path, $family_cfg:ty)),+ $(,)?) => {{
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
    ($verifier:expr, |$typed:ident| $body:expr; $(($variant:ident, $k:path, $family_cfg:ty)),+ $(,)?) => {{
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
