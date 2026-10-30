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
//! record is used only for the recorded family and for view bytes equal to the
//! recorded ones (length, then every byte, then the runtime policy digest); any
//! other view is loaded by the canonical path, so the record never changes
//! what is accepted. The generated file instantiates the rebuild for its own
//! family only, so a guest carries one copy.

use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_pcs::AkitaError;
use akita_schedules::PrevalidatedVerifierView;

use crate::one_hot_family::{AkitaOneHotBackendScheme, OneHotFamily};

/// Environment variable naming the generated file for a guest build.
pub const PREVALIDATED_CATALOG_VARIABLE: &str = "JOLT_PREVALIDATED_CATALOG_RS";

/// The commitment config of each one-hot family variant (the
/// `one_hot_families!` list); the host dispatch below checks every arm against
/// the variant's constructor at compile time.
#[cfg_attr(target_arch = "riscv64", allow(unused_macros))]
macro_rules! cfg_of {
    (K16Single) => {
        crate::configs::JoltOneHotK16
    };
    (K256Single) => {
        crate::configs::JoltOneHotK256
    };
    (K16W2R2) => {
        crate::configs::JoltOneHotK16W2R2
    };
    (K256W2R2) => {
        crate::configs::JoltOneHotK256W2R2
    };
    (K16W4R2) => {
        crate::configs::JoltOneHotK16W4R2
    };
    (K256W4R2) => {
        crate::configs::JoltOneHotK256W4R2
    };
    (K16W8R2) => {
        crate::configs::JoltOneHotK16W8R2
    };
    (K256W8R2) => {
        crate::configs::JoltOneHotK256W8R2
    };
}

/// The scheme over `view` for `expected`'s config, or `None` when `family` is
/// another family or the view does not match the record.
#[cfg_attr(target_arch = "riscv64", allow(dead_code))]
#[inline(never)]
fn rebuild_one<Cfg: CommitmentConfig>(
    family: OneHotFamily,
    expected: OneHotFamily,
    variant: fn(TrustedScheduleCatalog<Cfg>) -> AkitaOneHotBackendScheme,
    view: &[u8],
    prevalidated: &PrevalidatedVerifierView<'_>,
) -> Option<AkitaOneHotBackendScheme> {
    if family != expected {
        return None;
    }
    TrustedScheduleCatalog::<Cfg>::from_prevalidated_verifier_view(view, prevalidated).map(variant)
}

/// The generated file's view bytes, aligned to 8 bytes. The setup section
/// embeds the same bytes 8-aligned, and the run-time comparison of the two
/// copies takes its word loop only when both addresses agree modulo 8, so the
/// alignment keeps that comparison independent of the guest's data layout.
#[cfg_attr(
    all(not(target_arch = "riscv64"), not(test)),
    expect(
        dead_code,
        reason = "only the generated guest file and the tests name it"
    )
)]
#[repr(C, align(8))]
struct AlignedViewBytes<const N: usize>([u8; N]);

#[cfg(target_arch = "riscv64")]
mod compiled {
    include!(env!("JOLT_PREVALIDATED_CATALOG_RS"));
}
#[cfg(not(target_arch = "riscv64"))]
mod compiled {
    use super::{AkitaOneHotBackendScheme, OneHotFamily};

    pub(super) fn rebuild(_family: OneHotFamily, _view: &[u8]) -> Option<AkitaOneHotBackendScheme> {
        None
    }
}

/// The one-hot scheme over `view` from the compiled prevalidated view, or
/// `None` when it does not match (the caller then loads `view` canonically).
#[inline(never)]
pub(crate) fn one_hot_scheme(
    family: OneHotFamily,
    view: &[u8],
) -> Option<Result<AkitaOneHotBackendScheme, AkitaError>> {
    compiled::rebuild(family, view).map(Ok)
}

#[cfg(not(target_arch = "riscv64"))]
macro_rules! define_prevalidated_host {
    (; $(($variant:ident, $k:path, $profile:ident, $family_cfg:ty, $direct:ty, $artifact:ident)),+ $(,)?) => {
        /// Host: canonical load of `view` for `family`, the variant's name, and a
        /// rebuild of the result through `rebuild_one` with the same config the
        /// generated file names (a wrong `cfg_of!` arm does not compile).
        fn prevalidate(
            family: OneHotFamily,
            view: &[u8],
        ) -> Result<(akita_schedules::PrevalidatedVerifierRecord, &'static str, bool), AkitaError> {
            match family {
                $(OneHotFamily::$variant => {
                    let record = TrustedScheduleCatalog::<cfg_of!($variant)>::prevalidate_verifier_view(view)?;
                    let prevalidated = PrevalidatedVerifierView {
                        view,
                        policy_digest: record.policy_digest,
                        catalog_digest: record.catalog_digest,
                        carried_row_digests: &record.carried_row_digests,
                    };
                    let rebuilt = rebuild_one::<cfg_of!($variant)>(
                        family,
                        OneHotFamily::$variant,
                        |catalog| AkitaOneHotBackendScheme::$variant(akita_pcs::AkitaCommitmentScheme::new(catalog)),
                        view,
                        &prevalidated,
                    )
                    .is_some();
                    Ok((record, stringify!($variant), rebuilt))
                }),+
            }
        }
    };
}
#[cfg(not(target_arch = "riscv64"))]
crate::one_hot_family::one_hot_families!(define_prevalidated_host {});

/// Host: the generated file's source for `view` (one prevalidated view of
/// `family`), or an error when canonical loading or the rebuild check fails.
#[cfg(not(target_arch = "riscv64"))]
pub(crate) fn render(family: OneHotFamily, view: &[u8]) -> Result<String, AkitaError> {
    let (record, variant, rebuilt) = prevalidate(family, view)?;
    if !rebuilt {
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
    Ok(source(
        variant,
        view.len(),
        &bytes(view),
        &bytes(&record.policy_digest),
        &bytes(&record.catalog_digest),
        &rows,
    ))
}

/// Host: the generated file's text from its rendered parts. The view bytes
/// go into an `AlignedViewBytes` static that the view borrows.
#[cfg(not(target_arch = "riscv64"))]
fn source(
    variant: &str,
    view_len: usize,
    view: &str,
    policy_digest: &str,
    catalog_digest: &str,
    rows: &str,
) -> String {
    format!(
        "static VIEW_BYTES: super::AlignedViewBytes<{view_len}> = super::AlignedViewBytes([{view}]);\nstatic VIEW: akita_schedules::PrevalidatedVerifierView<'static> = akita_schedules::PrevalidatedVerifierView {{\nview: &VIEW_BYTES.0,\npolicy_digest: [{policy_digest}],\ncatalog_digest: [{catalog_digest}],\ncarried_row_digests: &[{rows}],\n}};\npub(super) fn rebuild(family: super::OneHotFamily, view: &[u8]) -> Option<super::AkitaOneHotBackendScheme> {{\nsuper::rebuild_one::<cfg_of!({variant})>(family, super::OneHotFamily::{variant}, |catalog| super::AkitaOneHotBackendScheme::{variant}(akita_pcs::AkitaCommitmentScheme::new(catalog)), view, &VIEW)\n}}\n",
    )
}

/// Host: the generated file's source with no prevalidated view (every
/// lookup misses and the canonical path runs).
pub const EMPTY_PREVALIDATED_CATALOG_SOURCE: &str = "pub(super) fn rebuild(_family: super::OneHotFamily, _view: &[u8]) -> Option<super::AkitaOneHotBackendScheme> {\nNone\n}\n";

#[cfg(all(test, not(target_arch = "riscv64")))]
mod tests {
    use super::{source, AlignedViewBytes};

    #[test]
    fn view_bytes_static_is_8_aligned() {
        static BYTES: AlignedViewBytes<11> = AlignedViewBytes([7; 11]);
        assert_eq!(core::mem::align_of::<AlignedViewBytes<11>>(), 8);
        assert_eq!(BYTES.0.as_ptr().addr() % 8, 0);
        let view: &'static [u8] = &BYTES.0;
        assert_eq!(view, &[7; 11]);
    }

    #[test]
    fn generated_source_borrows_the_aligned_view_bytes() {
        let text = source("K16Single", 3, "1,2,3", "4", "5", "[6]");
        assert!(text.starts_with(
            "static VIEW_BYTES: super::AlignedViewBytes<3> = super::AlignedViewBytes([1,2,3]);\n"
        ));
        assert!(text.contains("\nview: &VIEW_BYTES.0,\npolicy_digest: [4],\ncatalog_digest: [5],\ncarried_row_digests: &[[6]],\n};\n"));
        assert!(text.contains(
            "super::rebuild_one::<cfg_of!(K16Single)>(family, super::OneHotFamily::K16Single,"
        ));
        assert_eq!(text.matches("1,2,3").count(), 1);
        assert_eq!(text.matches("static ").count(), 2);
    }
}
