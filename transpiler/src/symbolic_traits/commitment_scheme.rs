//! Stub `CommitmentScheme` for symbolic transpilation (stages 1-7).
//!
//! # Purpose
//!
//! Jolt's verifier is generic over `PCS: CommitmentScheme`. To run the verifier with
//! symbolic types (`MleAst` instead of `Fr`), we need a `CommitmentScheme` that works
//! with `MleAst`. This module provides that stub.
//!
//! # Current Usage (Stages 1-7)
//!
//! The transpiler runs the Jolt verifier symbolically to record all field operations:
//!
//! ```ignore
//! // In main.rs - the verifier is instantiated with symbolic types
//! let verifier = TranspilableVerifier::<
//!     MleAst,                    // Symbolic field (records operations)
//!     AstCommitmentScheme,       // This stub (satisfies trait bounds)
//!     PoseidonAstTranscript,     // Symbolic transcript
//!     MleOpeningAccumulator,     // Collects opening claims
//! >::new(...);
//!
//! verifier.verify(&proof, ...);  // Runs stages 1-7, records AST
//! ```
//!
//! Stages 1-7 are **sumcheck-based** and don't call PCS methods. This stub satisfies
//! the `CommitmentScheme` trait bound without doing any work. Methods that would be
//! called during proving (`commit`, `prove`) panic since we only run verification.
//!
//! # Future: Stage 8 (PCS Verification)
//!
//! Stage 8 verifies polynomial commitment openings. Currently NOT transpiled because:
//! - **Dory**: Uses pairings (very expensive in-circuit, not practical)
//! - **Hyrax**: Technically transpilable (MSM becomes sumchecks + scalar muls), but
//!   the current `TranspilableVerifier` uses Dory, not Hyrax with recursion support
//!
//! With proper recursion setup (Hyrax over Grumpkin, sumcheck-based MSM verification),
//! stage 8 could be transpiled.
//! The `todo!()` methods (`combine_commitments`, `verify`) mark where this would plug in.
//!
//! # Why Vec<MleAst> Instead of Unit Types?
//!
//! Types like `AstProof(Vec<MleAst>)` use vectors instead of `()` for future
//! extensibility. If Hyrax-over-Grumpkin is ever transpiled, these types could
//! hold symbolic curve points.

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Read, SerializationError, Valid, Write,
};
use jolt_core::field::JoltField;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::transcripts::Transcript;
use jolt_core::utils::errors::ProofVerifyError;
use std::borrow::Borrow;
use zklean_extractor::mle_ast::MleAst;
use zklean_extractor::AstCommitment;

// =============================================================================
// Type Definitions
// =============================================================================

/// Symbolic commitment scheme for MleAst transpilation.
///
/// This is used to instantiate `TranspilableVerifier<MleAst, AstCommitmentScheme, ...>`
/// for symbolic execution that generates Gnark circuits.
#[derive(Clone, Debug)]
pub struct AstCommitmentScheme;

/// Verifier setup - empty for symbolic execution
#[derive(Clone, Debug, Default)]
pub struct AstVerifierSetup;

/// Prover setup - empty, never used in verification
#[derive(Clone, Debug, Default)]
pub struct AstProverSetup;

/// Opening proof - vector of MleAst for future PCS extensibility
#[derive(Clone, Debug, Default)]
pub struct AstProof(pub Vec<MleAst>);

/// Batched opening proof - vector of MleAst for future PCS extensibility
#[derive(Clone, Debug, Default)]
pub struct AstBatchedProof(pub Vec<MleAst>);

/// Opening proof hint - not used in verification
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AstOpeningHint;

// =============================================================================
// CommitmentScheme Implementation
// =============================================================================

impl CommitmentScheme for AstCommitmentScheme {
    type Field = MleAst;
    type ProverSetup = AstProverSetup;
    type VerifierSetup = AstVerifierSetup;
    type Commitment = AstCommitment;
    type Proof = AstProof;
    type BatchedProof = AstBatchedProof;
    type OpeningProofHint = AstOpeningHint;

    fn setup_prover(_max_num_vars: usize) -> Self::ProverSetup {
        panic!("AstCommitmentScheme::setup_prover should never be called during verification")
    }

    fn setup_verifier(_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        AstVerifierSetup
    }

    fn commit(
        _poly: &MultilinearPolynomial<Self::Field>,
        _setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        panic!("AstCommitmentScheme::commit should never be called during verification")
    }

    fn batch_commit<U>(
        _polys: &[U],
        _gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        panic!("AstCommitmentScheme::batch_commit should never be called during verification")
    }

    fn combine_commitments<C: Borrow<Self::Commitment>>(
        _commitments: &[C],
        _coeffs: &[Self::Field],
    ) -> Self::Commitment {
        // TODO: Implement for stage 8 PCS verification
        // This should combine AstCommitments symbolically
        todo!("AstCommitmentScheme::combine_commitments - implement for stage 8")
    }

    fn prove<ProofTranscript: Transcript>(
        _setup: &Self::ProverSetup,
        _poly: &MultilinearPolynomial<Self::Field>,
        _opening_point: &[<Self::Field as JoltField>::Challenge],
        _hint: Option<Self::OpeningProofHint>,
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        panic!("AstCommitmentScheme::prove should never be called during verification")
    }

    fn verify<ProofTranscript: Transcript>(
        _proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut ProofTranscript,
        _opening_point: &[<Self::Field as JoltField>::Challenge],
        _opening: &Self::Field,
        _commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        // TODO: Implement for stage 8 PCS verification
        // This should generate symbolic constraints for the opening check
        todo!("AstCommitmentScheme::verify - implement for stage 8")
    }

    fn protocol_name() -> &'static [u8] {
        b"AstCommitmentScheme"
    }
}

// =============================================================================
// Serialization Implementations (required by trait bounds)
// =============================================================================
//
// These are boilerplate implementations required by CommitmentScheme trait bounds.
// They perform no actual serialization since these types are only used during
// symbolic execution, never serialized to disk.

impl Valid for AstVerifierSetup {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for AstVerifierSetup {
    fn serialize_with_mode<W: Write>(
        &self,
        _writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        0
    }
}

impl CanonicalDeserialize for AstVerifierSetup {
    fn deserialize_with_mode<R: Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self)
    }
}

impl Valid for AstProverSetup {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for AstProverSetup {
    fn serialize_with_mode<W: Write>(
        &self,
        _writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        0
    }
}

impl CanonicalDeserialize for AstProverSetup {
    fn deserialize_with_mode<R: Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self)
    }
}

impl Valid for AstProof {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for AstProof {
    fn serialize_with_mode<W: Write>(
        &self,
        _writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        0
    }
}

impl CanonicalDeserialize for AstProof {
    fn deserialize_with_mode<R: Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self::default())
    }
}

impl Valid for AstBatchedProof {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for AstBatchedProof {
    fn serialize_with_mode<W: Write>(
        &self,
        _writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        0
    }
}

impl CanonicalDeserialize for AstBatchedProof {
    fn deserialize_with_mode<R: Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self::default())
    }
}

impl Valid for AstOpeningHint {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for AstOpeningHint {
    fn serialize_with_mode<W: Write>(
        &self,
        _writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        0
    }
}

impl CanonicalDeserialize for AstOpeningHint {
    fn deserialize_with_mode<R: Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self)
    }
}
