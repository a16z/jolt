//! Prevalidated one-hot catalog views (feature `prevalidated-catalog`, default
//! off).
//!
//! Loading the prepared one-hot catalog view repeats, on every verification,
//! the catalog's semantic audit, its digest over every row identity and a
//! canonical re-encode. All of it is a function of the view bytes and the
//! family's compile-time config. The recursion host loads the view it embeds
//! into the guest image through Akita's canonical loader before the guest
//! build, records what loading derived, and names a generated file in
//! `JOLT_PREVALIDATED_CATALOG_RS`; the guest includes it. At run time the
//! record is used only for view bytes equal to the recorded ones (length, then
//! every byte, then the runtime policy digest); any other view is loaded by
//! the canonical path, so the record never changes what is accepted.

use akita_config::TrustedScheduleCatalog;
use akita_pcs::{AkitaCommitmentScheme, AkitaError};
use akita_schedules::PrevalidatedVerifierView;

use crate::one_hot_family::{one_hot_families, AkitaOneHotBackendScheme, OneHotFamily};

/// Environment variable naming the generated file for a guest build.
pub const PREVALIDATED_CATALOG_VARIABLE: &str = "JOLT_PREVALIDATED_CATALOG_RS";

#[cfg(target_arch = "riscv64")]
mod compiled {
    use akita_schedules::PrevalidatedVerifierView;
    include!(env!("JOLT_PREVALIDATED_CATALOG_RS"));
}
#[cfg(not(target_arch = "riscv64"))]
mod compiled {
    use akita_schedules::PrevalidatedVerifierView;
    pub(super) static VIEWS: &[PrevalidatedVerifierView<'static>] = &[];
}

macro_rules! define_prevalidated_dispatch {
    (; $(($variant:ident, $k:path, $profile:ident, $cfg:ty, $direct:ty, $artifact:ident)),+ $(,)?) => {
        fn rebuild(
            family: OneHotFamily,
            view: &[u8],
            prevalidated: &PrevalidatedVerifierView<'_>,
        ) -> Option<AkitaOneHotBackendScheme> {
            match family {
                $(OneHotFamily::$variant => {
                    TrustedScheduleCatalog::<$cfg>::from_prevalidated_verifier_view(view, prevalidated)
                        .map(|catalog| AkitaOneHotBackendScheme::$variant(AkitaCommitmentScheme::new(catalog)))
                }),+
            }
        }

        #[cfg(not(target_arch = "riscv64"))]
        fn prevalidate(
            family: OneHotFamily,
            view: &[u8],
        ) -> Result<akita_schedules::PrevalidatedVerifierRecord, AkitaError> {
            match family {
                $(OneHotFamily::$variant => {
                    TrustedScheduleCatalog::<$cfg>::prevalidate_verifier_view(view)
                }),+
            }
        }
    };
}
one_hot_families!(define_prevalidated_dispatch {});

/// The one-hot scheme over `view` from the compiled prevalidated views, or
/// `None` when no compiled view matches `view` (the caller then loads `view`
/// canonically).
pub(crate) fn one_hot_scheme(
    family: OneHotFamily,
    view: &[u8],
) -> Option<Result<AkitaOneHotBackendScheme, AkitaError>> {
    compiled::VIEWS
        .iter()
        .find_map(|prevalidated| rebuild(family, view, prevalidated))
        .map(Ok)
}

/// Host: the generated file's source for `view` (one prevalidated view), or
/// an error when canonical loading or the rebuild check fails.
#[cfg(not(target_arch = "riscv64"))]
pub(crate) fn render(family: OneHotFamily, view: &[u8]) -> Result<String, AkitaError> {
    let record = prevalidate(family, view)?;
    let prevalidated = PrevalidatedVerifierView {
        view,
        policy_digest: record.policy_digest,
        catalog_digest: record.catalog_digest,
        carried_row_digests: &record.carried_row_digests,
    };
    if rebuild(family, view, &prevalidated).is_none() {
        return Err(AkitaError::InvalidSetup(
            "prevalidated one-hot catalog view does not rebuild".into(),
        ));
    }
    let bytes = |bytes: &[u8]| {
        bytes
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(",")
    };
    let rows = record
        .carried_row_digests
        .iter()
        .map(|digest| format!("[{}]", bytes(digest)))
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!(
        "pub(super) static VIEWS: &[PrevalidatedVerifierView<'static>] = &[PrevalidatedVerifierView {{\nview: &[{}],\npolicy_digest: [{}],\ncatalog_digest: [{}],\ncarried_row_digests: &[{}],\n}}];\n",
        bytes(view),
        bytes(&record.policy_digest),
        bytes(&record.catalog_digest),
        rows,
    ))
}

/// Host: the generated file's source with no prevalidated view (every
/// lookup misses and the canonical path runs).
pub const EMPTY_PREVALIDATED_CATALOG_SOURCE: &str =
    "pub(super) static VIEWS: &[PrevalidatedVerifierView<'static>] = &[];\n";
