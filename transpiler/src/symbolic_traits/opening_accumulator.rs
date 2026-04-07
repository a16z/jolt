//! AST Opening Accumulator for symbolic transpilation.
//!
//! This module provides an `OpeningAccumulator` implementation that works with `MleAst`
//! symbolic values instead of concrete field elements. It allows the verifier to be
//! transpiled while maintaining the exact same code structure as the real verifier.
//!
//! Post-rebase, the real `VerifierOpeningAccumulator` deduplicates openings: when the
//! same polynomial is opened at the same point by two different sumcheck IDs, the second
//! becomes an alias of the first. Aliased openings are NOT pushed to `pending_claims`
//! and `get_*` follows the alias chain. This accumulator replicates that behavior.

#![allow(non_snake_case)]

use ark_ff::Zero;
use jolt_core::poly::opening_proof::{
    AbstractVerifierOpeningAccumulator, OpeningAccumulator, OpeningId, OpeningPoint, PolynomialId,
    SumcheckId, BIG_ENDIAN,
};
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::claim_reductions::AdviceKind;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use std::collections::BTreeMap;
use zklean_extractor::mle_ast::MleAst;

/// Replicate jolt-core's private `underlying_polynomial_id`.
fn underlying_polynomial_id(opening_id: &OpeningId) -> PolynomialId {
    match opening_id {
        OpeningId::Polynomial(poly_id, _) => *poly_id,
        OpeningId::TrustedAdvice(_) => PolynomialId::Committed(CommittedPolynomial::TrustedAdvice),
        OpeningId::UntrustedAdvice(_) => {
            PolynomialId::Committed(CommittedPolynomial::UntrustedAdvice)
        }
    }
}

/// Opening accumulator for MleAst symbolic execution.
///
/// Mirrors `VerifierOpeningAccumulator` including its aliasing/deduplication logic.
#[derive(Clone, Debug, Default)]
pub struct AstOpeningAccumulator {
    pub openings: BTreeMap<OpeningId, (Vec<MleAst>, MleAst)>,
    pub pending_claims: Vec<MleAst>,
    pub log_T: usize,
    /// Alias map: key -> canonical opening ID.
    aliases: BTreeMap<OpeningId, OpeningId>,
    /// Index of populated opening IDs by underlying polynomial.
    opening_ids_by_poly: BTreeMap<PolynomialId, Vec<OpeningId>>,
}

impl AstOpeningAccumulator {
    pub fn new(log_T: usize) -> Self {
        Self {
            openings: BTreeMap::new(),
            pending_claims: Vec::new(),
            log_T,
            aliases: BTreeMap::new(),
            opening_ids_by_poly: BTreeMap::new(),
        }
    }

    pub fn new_with_claims<I>(claims: I, log_T: usize) -> Self
    where
        I: IntoIterator<Item = (OpeningId, MleAst)>,
    {
        let mut openings = BTreeMap::new();
        for (key, claim) in claims {
            openings.insert(key, (vec![], claim));
        }
        Self {
            openings,
            pending_claims: Vec::new(),
            log_T,
            aliases: BTreeMap::new(),
            opening_ids_by_poly: BTreeMap::new(),
        }
    }

    /// Follow the alias chain to the canonical opening ID.
    fn resolve_alias(&self, mut key: OpeningId) -> OpeningId {
        while let Some(next) = self.aliases.get(&key) {
            key = *next;
        }
        key
    }

    /// Find an existing opening of the same polynomial at the same point.
    /// Compares points element-wise by AST NodeId (root).
    fn find_existing_opening_at_point(
        &self,
        poly_id: PolynomialId,
        point: &[MleAst],
    ) -> Option<OpeningId> {
        self.opening_ids_by_poly.get(&poly_id).and_then(|ids| {
            ids.iter().find_map(|existing_id| {
                let (existing_point, _) = self
                    .openings
                    .get(existing_id)
                    .expect("indexed opening missing");
                if existing_point.is_empty() {
                    return None;
                }
                if existing_point.len() == point.len()
                    && existing_point
                        .iter()
                        .zip(point.iter())
                        .all(|(a, b)| a.root() == b.root())
                {
                    Some(*existing_id)
                } else {
                    None
                }
            })
        })
    }

    /// Index an opening after its point is populated.
    fn index_opening_id(&mut self, key: OpeningId) {
        let Some((point, _)) = self.openings.get(&key) else {
            return;
        };
        if point.is_empty() {
            return;
        }
        let entry = self
            .opening_ids_by_poly
            .entry(underlying_polynomial_id(&key))
            .or_default();
        if !entry.contains(&key) {
            entry.push(key);
        }
    }

    fn get_opening(&self, key: &OpeningId) -> (OpeningPoint<BIG_ENDIAN, MleAst>, MleAst) {
        let resolved = self.resolve_alias(*key);
        let (point, claim) = self
            .openings
            .get(&resolved)
            .unwrap_or_else(|| panic!("No opening found for {key:?} (resolved: {resolved:?})"));
        (OpeningPoint::new(point.clone()), *claim)
    }

    /// Store an opening point, replicating VerifierOpeningAccumulator::populate_or_alias_opening.
    ///
    /// Mirrors jolt-core/src/poly/opening_proof.rs:834-868 exactly:
    /// - Case A (key in openings): alias if same poly already opened at same point,
    ///   otherwise store normally.
    /// - Case B (key NOT in openings): alias if same poly at same point exists,
    ///   otherwise create with zero claim.
    fn store_opening(&mut self, key: &OpeningId, point: Vec<MleAst>) {
        if let Some((_, claim)) = self.openings.get(key) {
            // Case A: key in openings (pre-populated from proof)
            let claim = *claim;
            if let Some(existing_id) =
                self.find_existing_opening_at_point(underlying_polynomial_id(key), &point)
            {
                if existing_id != *key {
                    self.aliases.insert(*key, existing_id);
                    return;
                }
            }
            self.pending_claims.push(claim);
            self.openings.insert(*key, (point, claim));
            self.index_opening_id(*key);
        } else {
            // Case B: key NOT in openings (prover aliased it away)
            if let Some(existing_id) =
                self.find_existing_opening_at_point(underlying_polynomial_id(key), &point)
            {
                self.aliases.insert(*key, existing_id);
                return;
            }
            let claim = MleAst::zero();
            self.pending_claims.push(claim);
            self.openings.insert(*key, (point, claim));
            self.index_opening_id(*key);
        }
    }
}

impl OpeningAccumulator<MleAst> for AstOpeningAccumulator {
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
        let resolved = self.resolve_alias(key);
        let (point, claim) = self.openings.get(&resolved)?;
        Some((OpeningPoint::new(point.clone()), *claim))
    }
}

impl AbstractVerifierOpeningAccumulator<MleAst> for AstOpeningAccumulator {
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

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_core::field::JoltField;
    use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

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

        assert_eq!(accumulator.openings.len(), 3);
        assert_eq!(accumulator.log_T, 10);

        for (key, expected_claim) in claims {
            let (point, claim) = accumulator.openings.get(&key).unwrap();
            assert_eq!(point.len(), 0, "Point should be empty initially");
            assert_eq!(claim.root(), expected_claim.root());
        }
    }

    #[test]
    fn test_append_virtual_updates_point() {
        let claims = vec![(
            OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanOuter),
            MleAst::from_u64(100),
        )];

        let mut accumulator = AstOpeningAccumulator::new_with_claims(claims, 10);

        let point_values = vec![MleAst::from_u64(1), MleAst::from_u64(2)];
        let opening_point = OpeningPoint::new(point_values.clone());

        accumulator.append_virtual(
            VirtualPolynomial::PC,
            SumcheckId::SpartanOuter,
            opening_point,
        );

        let key = OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        let (stored_point, _) = accumulator.openings.get(&key).unwrap();
        assert_eq!(stored_point.len(), 2);
        assert_eq!(stored_point[0].root(), point_values[0].root());
        assert_eq!(stored_point[1].root(), point_values[1].root());
    }

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

        assert_eq!(retrieved_claim.root(), claim.root());
        assert_eq!(point.r.len(), 0);
    }

    #[test]
    fn test_virtual_polynomial_round_trip() {
        let claim = MleAst::from_u64(42);
        let claims = vec![(
            OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanOuter),
            claim,
        )];

        let mut accumulator = AstOpeningAccumulator::new_with_claims(claims, 10);

        let (point, retrieved_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        assert_eq!(retrieved_claim.root(), claim.root());
        assert_eq!(point.r.len(), 0, "Point should be empty before append");

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

        let (_, claim) = accumulator
            .get_advice_opening(AdviceKind::Trusted, SumcheckId::SpartanOuter)
            .expect("trusted advice should exist");
        assert_eq!(claim.root(), trusted_claim.root());
        assert_ne!(
            claim.root(),
            untrusted_claim.root(),
            "Must not confuse trusted and untrusted"
        );

        let (_, claim) = accumulator
            .get_advice_opening(AdviceKind::Untrusted, SumcheckId::SpartanOuter)
            .expect("untrusted advice should exist");
        assert_eq!(claim.root(), untrusted_claim.root());

        let result = accumulator.get_advice_opening(
            AdviceKind::Trusted,
            SumcheckId::SpartanProductVirtualization,
        );
        assert!(result.is_none(), "Missing advice should return None");
    }

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

        let (point, retrieved_claim) = accumulator
            .get_committed_polynomial_opening(CommittedPolynomial::RdInc, SumcheckId::SpartanOuter);
        assert_eq!(retrieved_claim.root(), claim.root());
        assert_eq!(point.r.len(), 2);
        assert_eq!(point.r[0].root(), point_values[0].root());
        assert_eq!(point.r[1].root(), point_values[1].root());
    }

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

        let (point, claim) = accumulator
            .get_advice_opening(AdviceKind::Trusted, SumcheckId::SpartanOuter)
            .unwrap();
        assert_eq!(claim.root(), trusted_claim.root());
        assert_eq!(point.r.len(), 2);
        assert_eq!(point.r[0].root(), trusted_point[0].root());

        let (point, claim) = accumulator
            .get_advice_opening(AdviceKind::Untrusted, SumcheckId::SpartanOuter)
            .unwrap();
        assert_eq!(claim.root(), untrusted_claim.root());
        assert_eq!(point.r.len(), 1);
        assert_eq!(point.r[0].root(), untrusted_point[0].root());
    }

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

        accumulator.append_virtual(
            VirtualPolynomial::PC,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(vec![MleAst::from_u64(1)]),
        );
        assert_eq!(accumulator.pending_claims.len(), 1);
        assert_eq!(accumulator.pending_claims[0].root(), claim_a.root());

        accumulator.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::SpartanOuter,
            vec![MleAst::from_u64(2)],
        );
        assert_eq!(accumulator.pending_claims.len(), 2);
        assert_eq!(accumulator.pending_claims[1].root(), claim_b.root());

        accumulator.append_trusted_advice(
            SumcheckId::SpartanOuter,
            OpeningPoint::new(vec![MleAst::from_u64(3)]),
        );
        assert_eq!(accumulator.pending_claims.len(), 3);
        assert_eq!(accumulator.pending_claims[2].root(), claim_c.root());
    }

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

        accumulator.append_virtual(
            VirtualPolynomial::PC,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(vec![MleAst::from_u64(1)]),
        );

        let taken = accumulator.take_pending_claims();
        assert_eq!(taken.len(), 1);
        assert_eq!(taken[0].root(), claim_a.root());
        assert!(
            accumulator.pending_claims.is_empty(),
            "Must be empty after take"
        );

        accumulator.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::SpartanOuter,
            vec![MleAst::from_u64(2)],
        );

        let taken2 = accumulator.take_pending_claims();
        assert_eq!(taken2.len(), 1);
        assert_eq!(taken2[0].root(), claim_b.root());

        assert!(accumulator.take_pending_claims().is_empty());
    }

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

        for poly in polynomials {
            let key = OpeningId::committed(poly, SumcheckId::SpartanOuter);
            let (stored_point, _) = accumulator.openings.get(&key).unwrap();
            assert_eq!(stored_point.len(), 2);
            assert_eq!(stored_point[0].root(), point_values[0].root());
            assert_eq!(stored_point[1].root(), point_values[1].root());
        }
    }

    /// Test aliasing: when a key is not pre-populated (prover aliased it) and
    /// the same polynomial was already opened at the same point, it should alias.
    #[test]
    fn test_aliasing_skips_pending_claims() {
        let canonical_claim = MleAst::from_u64(42);
        let claims = vec![(
            OpeningId::virt(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanProductVirtualization,
            ),
            canonical_claim,
        )];
        let mut accumulator = AstOpeningAccumulator::new_with_claims(claims, 10);

        // Append canonical opening (populates point, pushes claim)
        let p0 = MleAst::from_u64(1);
        let p1 = MleAst::from_u64(2);
        accumulator.append_virtual(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanProductVirtualization,
            OpeningPoint::new(vec![p0, p1]),
        );
        assert_eq!(accumulator.pending_claims.len(), 1);

        // Aliased opening (NOT pre-populated) with SAME point nodes
        accumulator.append_virtual(
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
            OpeningPoint::new(vec![p0, p1]),
        );
        assert_eq!(
            accumulator.pending_claims.len(),
            1,
            "Aliased opening must not push to pending_claims"
        );

        let (_, claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
        );
        assert_eq!(
            claim.root(),
            canonical_claim.root(),
            "Aliased get should return canonical claim"
        );
    }

    /// Test Case B with different point: NOT pre-populated, same poly but different
    /// point -> should NOT alias, should create with zero claim.
    #[test]
    fn test_case_b_different_point_no_alias() {
        let canonical_claim = MleAst::from_u64(42);
        let claims = vec![(
            OpeningId::virt(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanProductVirtualization,
            ),
            canonical_claim,
        )];
        let mut accumulator = AstOpeningAccumulator::new_with_claims(claims, 10);

        accumulator.append_virtual(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanProductVirtualization,
            OpeningPoint::new(vec![MleAst::from_u64(1)]),
        );
        assert_eq!(accumulator.pending_claims.len(), 1);

        // Different point -> no alias, creates new with zero
        accumulator.append_virtual(
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
            OpeningPoint::new(vec![MleAst::from_u64(99)]),
        );
        assert_eq!(
            accumulator.pending_claims.len(),
            2,
            "Different point -> no alias -> push zero claim"
        );
    }

    /// Test Case A with distinct points: both pre-populated, different points -> both store.
    #[test]
    fn test_case_a_distinct_points_both_store() {
        let claim_a = MleAst::from_u64(100);
        let claim_b = MleAst::from_u64(200);
        let claims = vec![
            (
                OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanOuter),
                claim_a,
            ),
            (
                OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanShift),
                claim_b,
            ),
        ];
        let mut accumulator = AstOpeningAccumulator::new_with_claims(claims, 10);

        accumulator.append_virtual(
            VirtualPolynomial::PC,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(vec![MleAst::from_u64(1)]),
        );
        assert_eq!(accumulator.pending_claims.len(), 1);

        // Different point -> stores normally (no alias)
        accumulator.append_virtual(
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            OpeningPoint::new(vec![MleAst::from_u64(2)]),
        );
        assert_eq!(
            accumulator.pending_claims.len(),
            2,
            "Different points -> both push"
        );

        let (_, claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        assert_eq!(claim.root(), claim_b.root());
    }

    /// Test Case A with same point: both pre-populated, same point -> second aliases.
    #[test]
    fn test_case_a_same_point_aliases() {
        let shared_point_node = MleAst::from_u64(99);
        let claim_a = MleAst::from_u64(100);
        let claim_b = MleAst::from_u64(100); // same claim (jolt-core asserts equality)
        let claims = vec![
            (
                OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanOuter),
                claim_a,
            ),
            (
                OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanShift),
                claim_b,
            ),
        ];
        let mut accumulator = AstOpeningAccumulator::new_with_claims(claims, 10);

        accumulator.append_virtual(
            VirtualPolynomial::PC,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(vec![shared_point_node]),
        );
        assert_eq!(accumulator.pending_claims.len(), 1);

        // Same point, same poly -> Case A aliases (no push)
        accumulator.append_virtual(
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            OpeningPoint::new(vec![shared_point_node]),
        );
        assert_eq!(
            accumulator.pending_claims.len(),
            1,
            "Same point -> alias, no extra push"
        );

        // get resolves alias to canonical claim
        let (_, claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        assert_eq!(claim.root(), claim_a.root());
    }
}
