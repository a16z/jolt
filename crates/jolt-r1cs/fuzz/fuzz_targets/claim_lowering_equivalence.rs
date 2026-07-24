#![no_main]

//! Differential oracle for symbolic claim expressions and their R1CS lowering.
//!
//! The fuzzer controls a bounded sum-of-products expression and whether each
//! opening, challenge, and derived value is represented as a constant or a
//! witness variable. The directly evaluated result must satisfy the lowered
//! constraints, while changing only the claimed result must violate them.

use jolt_claims::{Expr, Source, Term};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_r1cs::{assert_claim_expr_eq, ClaimSourceTable, R1csBuilder, SourceValue};
use libfuzzer_sys::fuzz_target;

const SOURCE_COUNT: usize = 4;
const MAX_TERMS: usize = 8;
const MAX_FACTORS: usize = 4;

fn word(data: &[u8], cursor: &mut usize) -> u64 {
    let mut bytes = [0u8; 8];
    for byte in &mut bytes {
        *byte = data[*cursor % data.len()];
        *cursor += 1;
    }
    u64::from_le_bytes(bytes)
}

fn source_value(
    builder: &mut R1csBuilder<Fr>,
    value: Fr,
    use_variable: bool,
) -> SourceValue<Fr> {
    if use_variable {
        SourceValue::variable(builder.alloc(value))
    } else {
        SourceValue::Constant(value)
    }
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mut cursor = 1;
    let opening_values: [Fr; SOURCE_COUNT] =
        std::array::from_fn(|_| Fr::from_u64(word(data, &mut cursor)));
    let challenge_values: [Fr; SOURCE_COUNT] =
        std::array::from_fn(|_| Fr::from_u64(word(data, &mut cursor)));
    let derived_values: [Fr; SOURCE_COUNT] =
        std::array::from_fn(|_| Fr::from_u64(word(data, &mut cursor)));

    let term_count = data[0] as usize % (MAX_TERMS + 1);
    let mut terms = Vec::with_capacity(term_count);
    for _ in 0..term_count {
        let coefficient = Fr::from_u64(word(data, &mut cursor));
        let factor_count = data[cursor % data.len()] as usize % (MAX_FACTORS + 1);
        cursor += 1;
        let mut factors = Vec::with_capacity(factor_count);
        for _ in 0..factor_count {
            let selector = data[cursor % data.len()];
            cursor += 1;
            let id = selector % SOURCE_COUNT as u8;
            factors.push(match (selector / SOURCE_COUNT as u8) % 3 {
                0 => Source::Opening(id),
                1 => Source::Challenge(id),
                _ => Source::Derived(id),
            });
        }
        terms.push(Term {
            coefficient,
            factors,
        });
    }
    let expression: Expr<Fr, u8, u8, u8> = Expr { terms };
    let expected = expression.evaluate(
        |id| opening_values[*id as usize],
        |id| challenge_values[*id as usize],
        |id| derived_values[*id as usize],
    );

    let mut builder = R1csBuilder::<Fr>::new();
    let mut sources = ClaimSourceTable::<Fr, u8, u8, u8>::new();
    for id in 0..SOURCE_COUNT {
        let flags = data[cursor % data.len()];
        cursor += 1;
        sources.insert_opening_source(
            id as u8,
            source_value(&mut builder, opening_values[id], flags & 1 != 0),
        );
        sources.insert_challenge_source(
            id as u8,
            source_value(&mut builder, challenge_values[id], flags & 2 != 0),
        );
        sources.insert_public_source(
            id as u8,
            source_value(&mut builder, derived_values[id], flags & 4 != 0),
        );
    }

    let output = builder.alloc(expected);
    assert_claim_expr_eq(&mut builder, &expression, output, &mut sources)
        .expect("all generated sources are registered");
    let witness = builder.witness().expect("all generated variables are assigned");
    let matrices = builder.into_matrices();
    assert!(
        matrices.check_witness(&witness).is_ok(),
        "direct evaluation disagrees with lowered R1CS"
    );

    let mut corrupted = witness;
    corrupted[output.index()] += Fr::from_u64(1);
    assert!(
        matrices.check_witness(&corrupted).is_err(),
        "lowered claim constraint accepted a corrupted output"
    );
});
