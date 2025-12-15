//! Common utilities for managing virtual polynomial claims
//!
//! This module provides generic helpers for appending and retrieving virtual
//! polynomial openings across different constraint types in the recursion SNARK.

use crate::{
    poly::opening_proof::{
        OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
        BIG_ENDIAN,
    },
    transcripts::Transcript,
    zkvm::witness::VirtualPolynomial,
};
use ark_bn254::Fq;

/// Appends a virtual polynomial opening to the prover accumulator
pub fn append_virtual_claim<T: Transcript>(
    accumulator: &mut ProverOpeningAccumulator<Fq>,
    transcript: &mut T,
    polynomial: VirtualPolynomial,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, Fq>,
    claim: Fq,
) {
    accumulator.append_virtual(transcript, polynomial, sumcheck_id, opening_point.clone(), claim);
}

/// Appends a virtual polynomial opening point to the verifier accumulator
pub fn append_virtual_opening<T: Transcript>(
    accumulator: &mut VerifierOpeningAccumulator<Fq>,
    transcript: &mut T,
    polynomial: VirtualPolynomial,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, Fq>,
) {
    accumulator.append_virtual(transcript, polynomial, sumcheck_id, opening_point.clone());
}

/// Retrieves a virtual polynomial claim from the verifier accumulator
pub fn get_virtual_claim(
    accumulator: &VerifierOpeningAccumulator<Fq>,
    polynomial: VirtualPolynomial,
    sumcheck_id: SumcheckId,
) -> Fq {
    let opening_key = OpeningId::Virtual(polynomial, sumcheck_id);
    accumulator
        .openings
        .get(&opening_key)
        .map(|(_, claim)| *claim)
        .unwrap_or_else(|| panic!("Virtual polynomial {:?} not found for sumcheck {:?}", polynomial, sumcheck_id))
}

