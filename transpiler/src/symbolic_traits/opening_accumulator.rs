//! AST Opening Accumulator for symbolic transpilation.
//!
//! # Overview
//!
//! This module provides an `OpeningAccumulator` implementation that works with `MleAst`
//! symbolic values instead of concrete field elements. It allows the verifier to be
//! transpiled while maintaining the exact same code structure as the real verifier.
//!
//! # Background: What is an Opening Accumulator?
//!
//! In the Jolt verifier, multiple polynomial evaluations need to be verified:
//! - Virtual polynomials: Computed from combinations of committed polynomials
//! - Dense polynomials: Directly committed and opened
//!
//! Instead of verifying each opening individually (expensive), openings are
//! "accumulated" and verified in a single batched check at the end (PCS stage).
//!
//! # How the Accumulator Works
//!
//! **Real verifier flow:**
//! 1. Proof contains pre-computed opening claims (evaluations at specific points)
//! 2. Verifier loads claims into accumulator at start
//! 3. During stages 1-7, verifier computes expected claims and checks against stored
//! 4. Stage 8 (PCS) batch-verifies all accumulated openings
//!
//! **Symbolic execution flow:**
//! 1. Claims become `MleAst::Var` inputs to the circuit
//! 2. Points are computed symbolically from transcript challenges
//! 3. Equality checks become `api.AssertIsEqual` constraints
//! 4. PCS verification is skipped (handled natively in Gnark, not transpiled)
//!
//! # Why This Design?
//!
//! The `OpeningAccumulator` trait is deeply integrated into Jolt's verifier. By
//! implementing it for `MleAst`, we can run the exact same verifier code for
//! transpilation. No separate "circuit verifier" implementation needed.

// Allow non_snake_case to match VerifierOpeningAccumulator naming (log_T)
#![allow(non_snake_case)]

use jolt_core::poly::opening_proof::{
    OpeningAccumulator, OpeningId, OpeningPoint, SumcheckId, BIG_ENDIAN,
};
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::claim_reductions::AdviceKind;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use std::collections::BTreeMap;
use zklean_extractor::mle_ast::MleAst;

// =============================================================================
// Type Definition
// =============================================================================

/// Opening accumulator for MleAst symbolic execution.
///
/// Stores polynomial opening claims as MleAst symbolic values,
/// allowing the verifier to be transpiled to a Gnark circuit.
#[derive(Clone, Debug)]
pub struct AstOpeningAccumulator {
    /// Map from opening ID to (point, claim)
    /// - point: Vec<MleAst> representing the evaluation point (challenges)
    /// - claim: MleAst representing the claimed evaluation
    pub openings: BTreeMap<OpeningId, (Vec<MleAst>, MleAst)>,
    /// Log of trace length (matches VerifierOpeningAccumulator for parity).
    /// Currently unused but stored for potential Stage 8 batch opening logic.
    pub log_T: usize,
}

// =============================================================================
// Inherent Methods
// =============================================================================

impl AstOpeningAccumulator {
    /// Create a new accumulator with no claims.
    pub fn new(log_T: usize) -> Self {
        Self {
            openings: BTreeMap::new(),
            log_T,
        }
    }

    /// Create an accumulator pre-populated with claims from the proof.
    ///
    /// The claims are MleAst variables that will become circuit inputs.
    /// Points are initialized as empty and will be populated during verification.
    ///
    /// # Arguments
    /// * `claims` - Iterator of (OpeningId, MleAst) pairs representing the claims
    /// * `log_T` - Log of trace length
    pub fn new_with_claims<I>(claims: I, log_T: usize) -> Self
    where
        I: IntoIterator<Item = (OpeningId, MleAst)>,
    {
        let mut openings = BTreeMap::new();
        for (key, claim) in claims {
            // Point is initially empty, will be set via append_* methods
            openings.insert(key, (vec![], claim));
        }
        Self { openings, log_T }
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /// Get an opening by key, returning (point, claim).
    fn get_opening(&self, key: &OpeningId) -> (OpeningPoint<BIG_ENDIAN, MleAst>, MleAst) {
        let (point, claim) = self
            .openings
            .get(key)
            .unwrap_or_else(|| panic!("No opening found for {key:?}"));
        (OpeningPoint::new(point.clone()), *claim)
    }

    /// Append an opening: add claim to transcript and store the point.
    fn append_opening<T: Transcript>(
        &mut self,
        transcript: &mut T,
        key: &OpeningId,
        point: Vec<MleAst>,
    ) {
        if let Some((stored_point, claim)) = self.openings.get_mut(key) {
            // CRITICAL: Must append the claim to transcript, matching VerifierOpeningAccumulator.
            // Without this, the transcript state diverges and challenges are incorrect.
            transcript.append_scalar(b"opening_claim", claim);
            *stored_point = point;
        } else {
            panic!("No opening found for {key:?}");
        }
    }

    // -------------------------------------------------------------------------
    // Test-only helpers
    // -------------------------------------------------------------------------

    /// Add a claim for a virtual polynomial.
    #[cfg(test)]
    pub fn add_virtual_claim(
        &mut self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        claim: MleAst,
    ) {
        let key = OpeningId::virtual_poly(polynomial, sumcheck);
        self.openings.insert(key, (vec![], claim));
    }

    /// Update the opening point for a virtual polynomial.
    #[cfg(test)]
    fn set_virtual_point(
        &mut self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        point: Vec<MleAst>,
    ) {
        let key = OpeningId::virtual_poly(polynomial, sumcheck);
        if let Some((stored_point, _)) = self.openings.get_mut(&key) {
            *stored_point = point;
        } else {
            panic!(
                "AstOpeningAccumulator::set_virtual_point: no claim found for {polynomial:?} {sumcheck:?}"
            );
        }
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl Default for AstOpeningAccumulator {
    fn default() -> Self {
        Self::new(0)
    }
}

impl OpeningAccumulator<MleAst> for AstOpeningAccumulator {
    // Methods ordered to match trait definition in opening_proof.rs

    fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, MleAst>, MleAst) {
        self.get_opening(&OpeningId::virtual_poly(polynomial, sumcheck))
    }

    fn get_committed_polynomial_opening(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, MleAst>, MleAst) {
        self.get_opening(&OpeningId::committed(polynomial, sumcheck))
    }

    fn get_advice_opening(
        &self,
        kind: AdviceKind,
        sumcheck_id: SumcheckId,
    ) -> Option<(OpeningPoint<BIG_ENDIAN, MleAst>, MleAst)> {
        let key = match kind {
            AdviceKind::Trusted => OpeningId::TrustedAdvice(sumcheck_id),
            AdviceKind::Untrusted => OpeningId::UntrustedAdvice(sumcheck_id),
        };
        let (point, claim) = self.openings.get(&key)?;
        Some((OpeningPoint::new(point.clone()), *claim))
    }

    fn append_virtual<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, MleAst>,
    ) {
        self.append_opening(
            transcript,
            &OpeningId::virtual_poly(polynomial, sumcheck),
            opening_point.r,
        );
    }

    fn append_untrusted_advice<T: Transcript>(
        &mut self,
        transcript: &mut T,
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, MleAst>,
    ) {
        self.append_opening(
            transcript,
            &OpeningId::UntrustedAdvice(sumcheck_id),
            opening_point.r,
        );
    }

    fn append_trusted_advice<T: Transcript>(
        &mut self,
        transcript: &mut T,
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, MleAst>,
    ) {
        self.append_opening(
            transcript,
            &OpeningId::TrustedAdvice(sumcheck_id),
            opening_point.r,
        );
    }

    fn append_dense<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
        opening_point: Vec<MleAst>,
    ) {
        self.append_opening(
            transcript,
            &OpeningId::committed(polynomial, sumcheck),
            opening_point,
        );
    }

    fn append_sparse<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        opening_point: Vec<MleAst>,
    ) {
        for polynomial in polynomials {
            self.append_opening(
                transcript,
                &OpeningId::committed(polynomial, sumcheck),
                opening_point.clone(),
            );
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_basic() {
        let mut acc = AstOpeningAccumulator::new(10); // log_T = 10

        // Add a claim
        let claim = MleAst::from_var(0);
        acc.add_virtual_claim(VirtualPolynomial::Product, SumcheckId::SpartanOuter, claim);

        // Update point
        let point = vec![MleAst::from_var(1), MleAst::from_var(2)];
        acc.set_virtual_point(VirtualPolynomial::Product, SumcheckId::SpartanOuter, point.clone());

        // Retrieve
        let (retrieved_point, _retrieved_claim) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::Product,
            SumcheckId::SpartanOuter,
        );

        assert_eq!(retrieved_point.r.len(), 2);
        // Claims should match (MleAst comparison)
    }
}
