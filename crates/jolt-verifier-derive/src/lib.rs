//! Derive macros for jolt-verifier's per-stage sumcheck-batch aggregates.
//!
//! [`macro@SumcheckBatch`] generates a stage's aggregate claim types
//! (`StageNInputClaims` / `StageNOutputClaims` / `StageNChallenges`) from a
//! source-of-truth struct whose fields are the stage's `ConcreteSumcheck`
//! instances. See `specs/sumcheck-batch-derive.md`.
//!
//! Scaffold only: code generation lands in a follow-up. The derive currently
//! accepts the annotated struct and emits nothing.

use proc_macro::TokenStream;

/// Generate a stage's aggregate claim types from a struct of `ConcreteSumcheck`
/// instances. Currently a no-op scaffold (see the module docs).
#[proc_macro_derive(SumcheckBatch)]
pub fn derive_sumcheck_batch(_input: TokenStream) -> TokenStream {
    TokenStream::new()
}
