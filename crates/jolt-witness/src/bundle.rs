//! Consumer bundles: a consumer's witness data flow, stated as a type.

use jolt_claims::protocols::jolt::JoltPolynomialId;
use jolt_field::Field;
use jolt_program::execution::TraceRow;

use crate::witnesses::WitnessEnv;
use crate::WitnessError;

pub use jolt_witness_derive::WitnessBundle;

/// A struct of atomic witnesses — exactly the values one consumer needs per
/// cycle. Derive with `#[derive(WitnessBundle)]`: fields extract through
/// their single-sourced `Extract` impls, and `#[opening(..)]` annotations tie
/// fields to jolt-claims ids (unannotated fields are consumer facts with no
/// protocol id).
pub trait WitnessBundle: Sized {
    /// The trace-backend constructor, composing the field extractors over
    /// one cycle window.
    fn from_row(
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError>;

    /// The jolt-claims ids of the annotated fields, in declaration order.
    fn annotated_ids() -> Vec<JoltPolynomialId>;

    /// The annotated fields' columns as field elements, keyed and ordered
    /// like [`annotated_ids`](Self::annotated_ids) — the consistency surface
    /// pinning the typed path to the id path.
    fn annotated_columns<F: Field>(rows: &[Self]) -> Vec<(JoltPolynomialId, Vec<F>)>;
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_riscv::CircuitFlags;

    use super::WitnessBundle;
    use crate::testing::with_sample_backend;
    use crate::witnesses::{LookupIndex, LookupOutput, OpFlag, Product, ToField, UnexpandedPc};
    use crate::BundleSource;

    /// Exercises every `#[opening(..)]` form: fact fields (no annotation),
    /// a bare virtual variant, and the payload-carrying indexed-family form.
    #[derive(Clone, Copy, Debug, WitnessBundle)]
    struct DeriveCoverageBundle {
        lookup_index: LookupIndex,
        product: Product,
        #[opening(LookupOutput)]
        lookup_output: LookupOutput,
        #[opening(UnexpandedPC)]
        unexpanded_pc: UnexpandedPc,
        #[opening(OpFlags(CircuitFlags::AddOperands))]
        add_operands: OpFlag,
    }

    #[test]
    fn annotated_ids_follow_declaration_order() {
        assert_eq!(
            DeriveCoverageBundle::annotated_ids(),
            vec![
                JoltPolynomialId::Virtual(JoltVirtualPolynomial::LookupOutput),
                JoltPolynomialId::Virtual(JoltVirtualPolynomial::UnexpandedPC),
                JoltPolynomialId::Virtual(JoltVirtualPolynomial::OpFlags(
                    CircuitFlags::AddOperands
                )),
            ]
        );
    }

    #[test]
    fn from_row_composes_the_field_extractors() {
        with_sample_backend(|backend| {
            let rows: Vec<DeriveCoverageBundle> = backend.bundles().unwrap();
            assert_eq!(rows.len(), 4);
            // The fixture's first cycle: ADDI rs1=5, imm=3 -> output 8.
            assert_eq!(rows[0].lookup_output, LookupOutput(8));
            assert_eq!(rows[0].unexpanded_pc.0, 0x8000_0000);
            assert_eq!(rows[0].add_operands, OpFlag(true));
            // Fact fields extract too, with no protocol ids attached: the
            // first cycle's product is rs1 * imm = 5 * 3.
            assert_eq!(rows[0].product.to_field::<Fr>(), Fr::from_u64(15));
            assert_eq!(rows[1].lookup_index, LookupIndex(0));
            assert_eq!(
                DeriveCoverageBundle::annotated_columns::<Fr>(&rows).len(),
                3
            );
        });
    }
}
