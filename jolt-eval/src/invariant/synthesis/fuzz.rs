use std::any::Any;
use std::sync::LazyLock;

use super::super::{CheckJsonResult, JoltInvariants};

struct CachedInvariant {
    inv: JoltInvariants,
    setup: Box<dyn Any + Send + Sync>,
}

static CACHE: LazyLock<Vec<CachedInvariant>> = LazyLock::new(|| {
    JoltInvariants::all()
        .into_iter()
        .map(|inv| {
            let setup = inv.dyn_setup();
            CachedInvariant { inv, setup }
        })
        .collect()
});

/// Fuzz a named invariant with raw byte data from libfuzzer.
///
/// Panics on invariant violation (which is what libfuzzer needs to
/// detect a finding).
///
/// # Usage in a fuzz target
///
/// ```ignore
/// #![no_main]
/// use libfuzzer_sys::fuzz_target;
/// fuzz_target!(|data: &[u8]| {
///     jolt_eval::invariant::synthesis::fuzz::fuzz_invariant("split_eq_bind_low_high", data);
/// });
/// ```
pub fn fuzz_invariant(invariant_name: &str, data: &[u8]) {
    let cached = CACHE
        .iter()
        .find(|c| c.inv.name() == invariant_name)
        .unwrap_or_else(|| panic!("Invariant '{invariant_name}' not found"));

    if let Ok(json_str) = std::str::from_utf8(data) {
        match cached.inv.check_json_input(&*cached.setup, json_str) {
            CheckJsonResult::Violation(e) => {
                panic!(
                    "Invariant '{}' violated: {e}\nInput JSON: {json_str}",
                    cached.inv.name()
                );
            }
            CheckJsonResult::Pass | CheckJsonResult::BadInput(_) => {}
        }
    }
}
