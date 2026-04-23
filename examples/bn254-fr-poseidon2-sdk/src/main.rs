//! Placeholder host binary for the Poseidon2 SDK benchmark guest.
//!
//! The real cycle-count comparison lives in
//! `crates/jolt-equivalence/tests/bn254_fr_smoke.rs` under
//! `poseidon2_cycle_count_vs_arkworks`.
fn main() {
    eprintln!(
        "This binary is a placeholder. See the \
         `poseidon2_cycle_count_vs_arkworks` test in \
         `crates/jolt-equivalence/tests/bn254_fr_smoke.rs`."
    );
    std::process::exit(0);
}
