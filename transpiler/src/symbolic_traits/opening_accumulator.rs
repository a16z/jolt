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
#[derive(Clone, Debug, Default)]
pub struct AstOpeningAccumulator {
    /// Map from opening ID to (point, claim)
    /// - point: Vec<MleAst> representing the evaluation point (challenges)
    /// - claim: MleAst representing the claimed evaluation
    pub openings: BTreeMap<OpeningId, (Vec<MleAst>, MleAst)>,
    /// Claims pending transcript flush, matching VerifierOpeningAccumulator behavior.
    pub pending_claims: Vec<MleAst>,
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
            pending_claims: Vec::new(),
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
        Self {
            openings,
            pending_claims: Vec::new(),
            log_T,
        }
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

    /// Store the opening point and push claim to pending_claims.
    /// Claims are flushed to transcript via `flush_to_transcript`.
    fn store_opening(&mut self, key: &OpeningId, point: Vec<MleAst>) {
        if let Some((stored_point, claim)) = self.openings.get_mut(key) {
            self.pending_claims.push(*claim);
            *stored_point = point;
        } else {
            panic!("No opening found for {key:?}");
        }
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl OpeningAccumulator<MleAst> for AstOpeningAccumulator {
    // Methods ordered to match trait definition in opening_proof.rs

    fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, MleAst>, MleAst) {
        self.get_opening(&OpeningId::virt(polynomial, sumcheck))
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

    fn append_virtual(
        &mut self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, MleAst>,
    ) {
        self.store_opening(&OpeningId::virt(polynomial, sumcheck), opening_point.r);
    }

    fn append_untrusted_advice(
        &mut self,
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, MleAst>,
    ) {
        self.store_opening(&OpeningId::UntrustedAdvice(sumcheck_id), opening_point.r);
    }

    fn append_trusted_advice(
        &mut self,
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, MleAst>,
    ) {
        self.store_opening(&OpeningId::TrustedAdvice(sumcheck_id), opening_point.r);
    }

    fn append_dense(
        &mut self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
        opening_point: Vec<MleAst>,
    ) {
        self.store_opening(&OpeningId::committed(polynomial, sumcheck), opening_point);
    }

    fn append_sparse(
        &mut self,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        opening_point: Vec<MleAst>,
    ) {
        for polynomial in polynomials {
            self.store_opening(
                &OpeningId::committed(polynomial, sumcheck),
                opening_point.clone(),
            );
        }
    }

    fn flush_to_transcript<T: Transcript>(&mut self, transcript: &mut T) {
        for claim in self.pending_claims.drain(..) {
            transcript.append_scalar(b"opening_claim", &claim);
        }
    }

    fn take_pending_claims(&mut self) -> Vec<MleAst> {
        std::mem::take(&mut self.pending_claims)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_core::field::JoltField;
    use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

    /// Verifies new_with_claims stores all three opening ID variants (virtual,
    /// committed, advice) with correct claims and empty points. Uses distinct
    /// sumcheck IDs to ensure keys don't collide.
    #[test]
    fn test_new_with_claims_populates_openings() {
        let claims = vec![
            (
                OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanOuter),
                MleAst::from_u64(100),
            ),
            (
                OpeningId::committed(CommittedPolynomial::RdInc, SumcheckId::SpartanOuter),
                MleAst::from_u64(200),
            ),
            (
                OpeningId::TrustedAdvice(SumcheckId::SpartanProductVirtualization),
                MleAst::from_u64(300),
            ),
        ];

        let accumulator = AstOpeningAccumulator::new_with_claims(claims.clone(), 10);

        // Verify all claims were stored
        assert_eq!(accumulator.openings.len(), 3);
        assert_eq!(accumulator.log_T, 10);

        // Verify each claim is stored with empty point
        for (key, expected_claim) in claims {
            let (point, claim) = accumulator.openings.get(&key).unwrap();
            assert_eq!(point.len(), 0, "Point should be empty initially");
            assert_eq!(claim.root(), expected_claim.root());
        }
    }

    /// Verifies append_virtual replaces the initially-empty point with the
    /// provided OpeningPoint while preserving the original claim.
    #[test]
    fn test_append_virtual_updates_point() {
        let claims = vec![(
            OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanOuter),
            MleAst::from_u64(100),
        )];

        let mut accumulator = AstOpeningAccumulator::new_with_claims(claims, 10);

        // Append opening with a point
        let point_values = vec![MleAst::from_u64(1), MleAst::from_u64(2)];
        let opening_point = OpeningPoint::new(point_values.clone());

        accumulator.append_virtual(
            VirtualPolynomial::PC,
            SumcheckId::SpartanOuter,
            opening_point,
        );

        // Verify point was stored
        let key = OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        let (stored_point, _) = accumulator.openings.get(&key).unwrap();
        assert_eq!(stored_point.len(), 2);
        assert_eq!(stored_point[0].root(), point_values[0].root());
        assert_eq!(stored_point[1].root(), point_values[1].root());
    }

    /// Verifies get_committed_polynomial_opening retrieves the correct claim
    /// for a committed polynomial before any point has been appended.
    #[test]
    fn test_get_committed_polynomial_opening() {
        let claim = MleAst::from_u64(12345);
        let claims = vec![(
            OpeningId::committed(CommittedPolynomial::RdInc, SumcheckId::SpartanOuter),
            claim,
        )];

        let accumulator = AstOpeningAccumulator::new_with_claims(claims, 10);

        let (point, retrieved_claim) = accumulator
            .get_committed_polynomial_opening(CommittedPolynomial::RdInc, SumcheckId::SpartanOuter);

        // Verify correct claim was retrieved
        assert_eq!(retrieved_claim.root(), claim.root());
        // Point should be empty (not yet set)
        assert_eq!(point.r.len(), 0);
    }

    /// Verifies the full round-trip for virtual polynomials:
    /// create claims → append point → get returns both correct point and claim.
    /// This exercises the append→get contract that the verifier relies on.
    #[test]
    fn test_virtual_polynomial_round_trip() {
        let claim = MleAst::from_u64(42);
        let claims = vec![(
            OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanOuter),
            claim,
        )];

        let mut accumulator = AstOpeningAccumulator::new_with_claims(claims, 10);

        // Before append: point is empty, claim is set
        let (point, retrieved_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        assert_eq!(retrieved_claim.root(), claim.root());
        assert_eq!(point.r.len(), 0, "Point should be empty before append");

        // After append: point is populated, claim is unchanged
        let point_values = vec![MleAst::from_u64(10), MleAst::from_u64(20)];
        accumulator.append_virtual(
            VirtualPolynomial::PC,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(point_values.clone()),
        );

        let (point, retrieved_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        assert_eq!(
            retrieved_claim.root(),
            claim.root(),
            "Claim must survive append"
        );
        assert_eq!(point.r.len(), 2);
        assert_eq!(point.r[0].root(), point_values[0].root());
        assert_eq!(point.r[1].root(), point_values[1].root());
    }

    /// Verifies get_advice_opening dispatches correctly on AdviceKind, returning
    /// the right claim for Trusted vs Untrusted, and None for missing keys.
    #[test]
    fn test_get_advice_opening_dispatch() {
        let trusted_claim = MleAst::from_u64(111);
        let untrusted_claim = MleAst::from_u64(222);
        let claims = vec![
            (
                OpeningId::TrustedAdvice(SumcheckId::SpartanOuter),
                trusted_claim,
            ),
            (
                OpeningId::UntrustedAdvice(SumcheckId::SpartanOuter),
                untrusted_claim,
            ),
        ];

        let accumulator = AstOpeningAccumulator::new_with_claims(claims, 10);

        // Trusted returns trusted claim (not untrusted)
        let (_, claim) = accumulator
            .get_advice_opening(AdviceKind::Trusted, SumcheckId::SpartanOuter)
            .expect("trusted advice should exist");
        assert_eq!(claim.root(), trusted_claim.root());
        assert_ne!(
            claim.root(),
            untrusted_claim.root(),
            "Must not confuse trusted and untrusted"
        );

        // Untrusted returns untrusted claim
        let (_, claim) = accumulator
            .get_advice_opening(AdviceKind::Untrusted, SumcheckId::SpartanOuter)
            .expect("untrusted advice should exist");
        assert_eq!(claim.root(), untrusted_claim.root());

        // Missing sumcheck ID returns None
        let result = accumulator.get_advice_opening(
            AdviceKind::Trusted,
            SumcheckId::SpartanProductVirtualization,
        );
        assert!(result.is_none(), "Missing advice should return None");
    }

    /// Verifies the full round-trip for dense (committed) polynomials:
    /// append_dense stores the point, then get_committed retrieves it correctly.
    /// Tests a different code path than append_virtual (takes Vec directly, not OpeningPoint).
    #[test]
    fn test_dense_polynomial_round_trip() {
        let claim = MleAst::from_u64(100);
        let claims = vec![(
            OpeningId::committed(CommittedPolynomial::RdInc, SumcheckId::SpartanOuter),
            claim,
        )];

        let mut accumulator = AstOpeningAccumulator::new_with_claims(claims, 10);

        let point_values = vec![MleAst::from_u64(5), MleAst::from_u64(6)];
        accumulator.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::SpartanOuter,
            point_values.clone(),
        );

        // Retrieve via the get_ accessor (not raw map access)
        let (point, retrieved_claim) = accumulator
            .get_committed_polynomial_opening(CommittedPolynomial::RdInc, SumcheckId::SpartanOuter);
        assert_eq!(retrieved_claim.root(), claim.root());
        assert_eq!(point.r.len(), 2);
        assert_eq!(point.r[0].root(), point_values[0].root());
        assert_eq!(point.r[1].root(), point_values[1].root());
    }

    /// Verifies the round-trip for both advice variants: append stores points,
    /// get_advice_opening retrieves them correctly by AdviceKind dispatch.
    #[test]
    fn test_advice_round_trip() {
        let trusted_claim = MleAst::from_u64(100);
        let untrusted_claim = MleAst::from_u64(200);
        let claims = vec![
            (
                OpeningId::TrustedAdvice(SumcheckId::SpartanOuter),
                trusted_claim,
            ),
            (
                OpeningId::UntrustedAdvice(SumcheckId::SpartanOuter),
                untrusted_claim,
            ),
        ];

        let mut accumulator = AstOpeningAccumulator::new_with_claims(claims, 10);

        let trusted_point = vec![MleAst::from_u64(7), MleAst::from_u64(8)];
        accumulator.append_trusted_advice(
            SumcheckId::SpartanOuter,
            OpeningPoint::new(trusted_point.clone()),
        );

        let untrusted_point = vec![MleAst::from_u64(9)];
        accumulator.append_untrusted_advice(
            SumcheckId::SpartanOuter,
            OpeningPoint::new(untrusted_point.clone()),
        );

        // Trusted: correct point length and values
        let (point, claim) = accumulator
            .get_advice_opening(AdviceKind::Trusted, SumcheckId::SpartanOuter)
            .unwrap();
        assert_eq!(claim.root(), trusted_claim.root());
        assert_eq!(point.r.len(), 2);
        assert_eq!(point.r[0].root(), trusted_point[0].root());

        // Untrusted: different point length (verifies no cross-contamination)
        let (point, claim) = accumulator
            .get_advice_opening(AdviceKind::Untrusted, SumcheckId::SpartanOuter)
            .unwrap();
        assert_eq!(claim.root(), untrusted_claim.root());
        assert_eq!(point.r.len(), 1);
        assert_eq!(point.r[0].root(), untrusted_point[0].root());
    }

    /// Verifies that each append pushes the stored claim (not the point) to pending_claims.
    /// This is the mechanism the verifier uses to batch claims for transcript flushing.
    /// Uses different append methods to ensure they all share the same store_opening path.
    #[test]
    fn test_append_pushes_claims_to_pending() {
        let claim_a = MleAst::from_u64(100);
        let claim_b = MleAst::from_u64(200);
        let claim_c = MleAst::from_u64(300);
        let claims = vec![
            (
                OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanOuter),
                claim_a,
            ),
            (
                OpeningId::committed(CommittedPolynomial::RdInc, SumcheckId::SpartanOuter),
                claim_b,
            ),
            (OpeningId::TrustedAdvice(SumcheckId::SpartanOuter), claim_c),
        ];

        let mut accumulator = AstOpeningAccumulator::new_with_claims(claims, 10);
        assert!(accumulator.pending_claims.is_empty(), "Should start empty");

        // Virtual append
        accumulator.append_virtual(
            VirtualPolynomial::PC,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(vec![MleAst::from_u64(1)]),
        );
        assert_eq!(accumulator.pending_claims.len(), 1);
        assert_eq!(accumulator.pending_claims[0].root(), claim_a.root());

        // Dense append
        accumulator.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::SpartanOuter,
            vec![MleAst::from_u64(2)],
        );
        assert_eq!(accumulator.pending_claims.len(), 2);
        assert_eq!(accumulator.pending_claims[1].root(), claim_b.root());

        // Advice append
        accumulator.append_trusted_advice(
            SumcheckId::SpartanOuter,
            OpeningPoint::new(vec![MleAst::from_u64(3)]),
        );
        assert_eq!(accumulator.pending_claims.len(), 3);
        assert_eq!(accumulator.pending_claims[2].root(), claim_c.root());
    }

    /// Verifies take_pending_claims returns accumulated claims and drains the buffer.
    /// Also verifies that claims accumulated after a take are independent of previous ones.
    #[test]
    fn test_take_pending_claims_drains_and_resets() {
        let claim_a = MleAst::from_u64(100);
        let claim_b = MleAst::from_u64(200);
        let claims = vec![
            (
                OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanOuter),
                claim_a,
            ),
            (
                OpeningId::committed(CommittedPolynomial::RdInc, SumcheckId::SpartanOuter),
                claim_b,
            ),
        ];

        let mut accumulator = AstOpeningAccumulator::new_with_claims(claims, 10);

        // Append first claim
        accumulator.append_virtual(
            VirtualPolynomial::PC,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(vec![MleAst::from_u64(1)]),
        );

        // Take drains the first claim
        let taken = accumulator.take_pending_claims();
        assert_eq!(taken.len(), 1);
        assert_eq!(taken[0].root(), claim_a.root());
        assert!(
            accumulator.pending_claims.is_empty(),
            "Must be empty after take"
        );

        // Append second claim after the take
        accumulator.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::SpartanOuter,
            vec![MleAst::from_u64(2)],
        );

        // Second take only returns the new claim, not the old one
        let taken2 = accumulator.take_pending_claims();
        assert_eq!(taken2.len(), 1);
        assert_eq!(taken2[0].root(), claim_b.root());

        // Third take is empty
        assert!(accumulator.take_pending_claims().is_empty());
    }

    /// Verifies append_sparse broadcasts the same opening point to multiple
    /// committed polynomials in a single call. This is the batch path used for
    /// sparse polynomial openings where all share the same evaluation point.
    #[test]
    fn test_append_sparse_multiple_polynomials() {
        let claims = vec![
            (
                OpeningId::committed(CommittedPolynomial::RdInc, SumcheckId::SpartanOuter),
                MleAst::from_u64(100),
            ),
            (
                OpeningId::committed(CommittedPolynomial::RamInc, SumcheckId::SpartanOuter),
                MleAst::from_u64(200),
            ),
            (
                OpeningId::committed(CommittedPolynomial::TrustedAdvice, SumcheckId::SpartanOuter),
                MleAst::from_u64(300),
            ),
        ];

        let mut accumulator = AstOpeningAccumulator::new_with_claims(claims, 10);

        let point_values = vec![MleAst::from_u64(1), MleAst::from_u64(2)];
        let polynomials = vec![
            CommittedPolynomial::RdInc,
            CommittedPolynomial::RamInc,
            CommittedPolynomial::TrustedAdvice,
        ];

        accumulator.append_sparse(
            polynomials.clone(),
            SumcheckId::SpartanOuter,
            point_values.clone(),
        );

        // Verify all polynomials have the same point stored
        for poly in polynomials {
            let key = OpeningId::committed(poly, SumcheckId::SpartanOuter);
            let (stored_point, _) = accumulator.openings.get(&key).unwrap();
            assert_eq!(stored_point.len(), 2);
            assert_eq!(stored_point[0].root(), point_values[0].root());
            assert_eq!(stored_point[1].root(), point_values[1].root());
        }
    }
}
