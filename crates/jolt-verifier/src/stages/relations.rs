//! Shared per-relation opening-claim plumbing.
//!
//! These traits are implemented by `#[derive(OutputClaims)]` /
//! `#[derive(InputClaims)]` (crate `jolt-verifier-derive`) on each relation's
//! claim struct. They make the canonical opening **order** and **count** a
//! single-sourced consequence of a struct's field declaration order, instead of
//! the three hand-written copies (`*_output_claim_values`, `append_*`, and the
//! `+ 1 + 1 + 2`-style count literals) that historically drift apart.

use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_field::Field;
use jolt_transcript::Transcript;

/// Canonical encoders and the output-formula resolver for a relation's
/// *produced* opening-claim struct.
///
/// The implementor's field declaration order is the single definition of
/// canonical opening order: [`opening_values`](Self::opening_values),
/// [`opening_count`](Self::opening_count), and
/// [`append_openings`](Self::append_openings) all derive from it, so they cannot
/// disagree.
pub trait OutputClaims<F: Field> {
    /// Produced opening scalars in canonical (field-declaration) order.
    fn opening_values(&self) -> Vec<F>;

    /// Number of produced openings; equals `opening_values().len()` but is
    /// computed without allocating.
    fn opening_count(&self) -> usize;

    /// Append every produced opening to the transcript in canonical order, each
    /// under the `b"opening_claim"` label. This is the Fiat-Shamir order and
    /// MUST match the order in which the prover commits the openings.
    fn append_openings<T: Transcript<Challenge = F>>(&self, transcript: &mut T);

    /// Resolve a produced opening's value by id, for evaluating the relation's
    /// output `Expr`. Returns `None` for ids this struct does not carry (callers
    /// turn that into [`VerifierError::MissingOpeningClaim`](crate::VerifierError)).
    fn resolve_output(&self, id: &JoltOpeningId) -> Option<F>;
}

/// The input-formula resolver for a relation's *consumed* opening-claim struct
/// (populated by explicit cross-stage wiring).
pub trait InputClaims<F: Field> {
    /// Resolve a consumed opening's value by id, for evaluating the relation's
    /// input `Expr`. Returns `None` for ids this struct does not carry.
    fn resolve_input(&self, id: &JoltOpeningId) -> Option<F>;
}

#[cfg(test)]
mod tests {
    use super::*;

    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier_derive::{InputClaims, OutputClaims};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn virt(polynomial: JoltVirtualPolynomial, relation: JoltRelationId) -> JoltOpeningId {
        JoltOpeningId::virtual_polynomial(polynomial, relation)
    }

    fn committed(polynomial: JoltCommittedPolynomial, relation: JoltRelationId) -> JoltOpeningId {
        JoltOpeningId::committed(polynomial, relation)
    }

    /// A minimal `Transcript` double that records each appended byte chunk, so
    /// that append order can be compared without depending on the digest.
    #[derive(Clone, Default)]
    struct RecordingTranscript {
        chunks: Vec<Vec<u8>>,
    }

    impl Transcript for RecordingTranscript {
        type Challenge = Fr;

        fn new(_label: &'static [u8]) -> Self {
            Self::default()
        }

        fn append_bytes(&mut self, bytes: &[u8]) {
            self.chunks.push(bytes.to_vec());
        }

        fn challenge(&mut self) -> Self::Challenge {
            Fr::from_u64(0)
        }

        fn state(&self) -> [u8; 32] {
            [0u8; 32]
        }
    }

    /// The chunk stream produced by appending `opening_values()` one-by-one is
    /// the reference Fiat-Shamir order; `append_openings` must reproduce it.
    fn assert_append_matches_values<C: OutputClaims<Fr>>(claims: &C) {
        let mut via_append = RecordingTranscript::default();
        claims.append_openings(&mut via_append);

        let mut via_values = RecordingTranscript::default();
        for value in claims.opening_values() {
            via_values.append_labeled(b"opening_claim", &value);
        }

        assert_eq!(via_append.chunks, via_values.chunks);
    }

    #[derive(OutputClaims)]
    #[relation(InstructionReadRaf)]
    struct InstructionLeaf<F: Field> {
        #[opening(LookupTableFlag)]
        lookup_table_flags: Vec<F>,
        #[opening(InstructionRa)]
        instruction_ra: Vec<F>,
        #[opening(InstructionRafFlag)]
        instruction_raf_flag: F,
    }

    #[test]
    fn output_leaf_encoders_follow_declaration_order() {
        let claims = InstructionLeaf {
            lookup_table_flags: vec![fr(1), fr(2)],
            instruction_ra: vec![fr(3), fr(4), fr(5)],
            instruction_raf_flag: fr(6),
        };

        assert_eq!(claims.opening_count(), 6);
        assert_eq!(
            claims.opening_values(),
            vec![fr(1), fr(2), fr(3), fr(4), fr(5), fr(6)],
        );
        assert_eq!(claims.opening_values().len(), claims.opening_count());
        assert_append_matches_values(&claims);
    }

    #[test]
    fn output_leaf_resolves_indexed_and_scalar_ids() {
        let claims = InstructionLeaf {
            lookup_table_flags: vec![fr(10), fr(11)],
            instruction_ra: vec![fr(20), fr(21)],
            instruction_raf_flag: fr(30),
        };
        let relation = JoltRelationId::InstructionReadRaf;

        assert_eq!(
            claims.resolve_output(&virt(JoltVirtualPolynomial::LookupTableFlag(1), relation)),
            Some(fr(11)),
        );
        assert_eq!(
            claims.resolve_output(&virt(JoltVirtualPolynomial::InstructionRa(0), relation)),
            Some(fr(20)),
        );
        assert_eq!(
            claims.resolve_output(&virt(JoltVirtualPolynomial::InstructionRafFlag, relation)),
            Some(fr(30)),
        );
        // Out-of-range index and wrong relation both miss.
        assert_eq!(
            claims.resolve_output(&virt(JoltVirtualPolynomial::LookupTableFlag(2), relation)),
            None,
        );
        assert_eq!(
            claims.resolve_output(&virt(
                JoltVirtualPolynomial::InstructionRafFlag,
                JoltRelationId::RamRaClaimReduction,
            )),
            None,
        );
    }

    #[derive(OutputClaims)]
    #[relation(RamReadWriteChecking)]
    struct CommittedLeaf<F: Field> {
        #[opening(committed = RamInc)]
        ram_inc: F,
        #[opening(committed = BytecodeChunk)]
        bytecode_chunks: Vec<F>,
    }

    #[test]
    fn output_leaf_resolves_committed_ids() {
        let claims = CommittedLeaf {
            ram_inc: fr(7),
            bytecode_chunks: vec![fr(8), fr(9)],
        };
        let relation = JoltRelationId::RamReadWriteChecking;

        assert_eq!(claims.opening_count(), 3);
        assert_eq!(claims.opening_values(), vec![fr(7), fr(8), fr(9)]);
        assert_eq!(
            claims.resolve_output(&committed(JoltCommittedPolynomial::RamInc, relation)),
            Some(fr(7)),
        );
        assert_eq!(
            claims.resolve_output(&committed(JoltCommittedPolynomial::BytecodeChunk(1), relation)),
            Some(fr(9)),
        );
        assert_append_matches_values(&claims);
    }

    #[derive(OutputClaims)]
    struct OutputAggregate<F: Field> {
        instruction: InstructionLeaf<F>,
        committed: CommittedLeaf<F>,
    }

    #[test]
    fn output_aggregate_recurses_into_sub_structs() {
        let claims = OutputAggregate {
            instruction: InstructionLeaf {
                lookup_table_flags: vec![fr(1)],
                instruction_ra: vec![fr(2)],
                instruction_raf_flag: fr(3),
            },
            committed: CommittedLeaf {
                ram_inc: fr(4),
                bytecode_chunks: vec![fr(5)],
            },
        };

        assert_eq!(claims.opening_count(), 5);
        assert_eq!(claims.opening_values(), vec![fr(1), fr(2), fr(3), fr(4), fr(5)]);
        assert_eq!(
            claims.resolve_output(&virt(
                JoltVirtualPolynomial::InstructionRafFlag,
                JoltRelationId::InstructionReadRaf,
            )),
            Some(fr(3)),
        );
        assert_eq!(
            claims.resolve_output(&committed(
                JoltCommittedPolynomial::RamInc,
                JoltRelationId::RamReadWriteChecking,
            )),
            Some(fr(4)),
        );
        assert_append_matches_values(&claims);
    }

    #[derive(InputClaims)]
    struct ReductionInputs<F: Field> {
        #[opening(RamRa, from = RamRafEvaluation)]
        raf: F,
        #[opening(RamRa, from = RamReadWriteChecking)]
        read_write: F,
        #[opening(RamRa, from = RamValCheck)]
        val_check: F,
    }

    #[test]
    fn input_leaf_resolves_same_polynomial_across_relations() {
        let inputs = ReductionInputs {
            raf: fr(1),
            read_write: fr(2),
            val_check: fr(3),
        };

        assert_eq!(
            inputs.resolve_input(&virt(JoltVirtualPolynomial::RamRa, JoltRelationId::RamRafEvaluation)),
            Some(fr(1)),
        );
        assert_eq!(
            inputs.resolve_input(&virt(
                JoltVirtualPolynomial::RamRa,
                JoltRelationId::RamReadWriteChecking,
            )),
            Some(fr(2)),
        );
        assert_eq!(
            inputs.resolve_input(&virt(JoltVirtualPolynomial::RamRa, JoltRelationId::RamValCheck)),
            Some(fr(3)),
        );
        assert_eq!(
            inputs.resolve_input(&virt(
                JoltVirtualPolynomial::RamRa,
                JoltRelationId::RamRaClaimReduction,
            )),
            None,
        );
    }

    #[derive(InputClaims)]
    struct OptionalInputs<F: Field> {
        #[opening(LookupOutput, from = InstructionClaimReduction)]
        lookup_output: Option<F>,
        #[opening(LeftLookupOperand, from = InstructionClaimReduction)]
        left_lookup_operand: F,
    }

    #[test]
    fn input_leaf_surfaces_option_fields_directly() {
        let relation = JoltRelationId::InstructionClaimReduction;
        let present = OptionalInputs {
            lookup_output: Some(fr(9)),
            left_lookup_operand: fr(8),
        };
        assert_eq!(
            present.resolve_input(&virt(JoltVirtualPolynomial::LookupOutput, relation)),
            Some(fr(9)),
        );
        assert_eq!(
            present.resolve_input(&virt(JoltVirtualPolynomial::LeftLookupOperand, relation)),
            Some(fr(8)),
        );

        let absent = OptionalInputs {
            lookup_output: None,
            left_lookup_operand: fr(8),
        };
        assert_eq!(
            absent.resolve_input(&virt(JoltVirtualPolynomial::LookupOutput, relation)),
            None,
        );
    }
}
