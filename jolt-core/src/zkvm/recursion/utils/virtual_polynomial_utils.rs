//! Generic utilities for managing virtual polynomial claims
//!
//! This module provides generic helpers to reduce duplication across
//! different constraint types in stage1 implementations.

use crate::{
    field::JoltField,
    poly::opening_proof::{
        OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
    },
    transcripts::Transcript,
    zkvm::witness::VirtualPolynomial,
};

/// Generic helper to append multiple virtual polynomial claims
///
/// This replaces the duplicated append_*_virtual_claims functions across stage1 implementations
pub fn append_virtual_claims<F: JoltField, T: Transcript>(
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, F>,
    claims: &[(VirtualPolynomial, F)],
) {
    for (polynomial, claim) in claims {
        accumulator.append_virtual(
            transcript,
            *polynomial,
            sumcheck_id,
            opening_point.clone(),
            *claim,
        );
    }
}

/// Generic helper to append virtual polynomial openings for verifier
pub fn append_virtual_openings<F: JoltField, T: Transcript>(
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, F>,
    polynomials: &[VirtualPolynomial],
) {
    for polynomial in polynomials {
        accumulator.append_virtual(transcript, *polynomial, sumcheck_id, opening_point.clone());
    }
}

/// Generic helper to retrieve multiple virtual polynomial claims
pub fn get_virtual_claims<F: JoltField>(
    accumulator: &VerifierOpeningAccumulator<F>,
    sumcheck_id: SumcheckId,
    polynomials: &[VirtualPolynomial],
) -> Vec<F> {
    use crate::poly::opening_proof::OpeningAccumulator;
    polynomials
        .iter()
        .map(|poly| {
            let (_, claim) = accumulator.get_virtual_polynomial_opening(*poly, sumcheck_id);
            claim
        })
        .collect()
}

/// Macro to simplify creating polynomial-claim pairs for append_virtual_claims
#[macro_export]
macro_rules! virtual_claims {
    ($($poly:expr => $claim:expr),* $(,)?) => {
        vec![$(($poly, $claim)),*]
    };
}
