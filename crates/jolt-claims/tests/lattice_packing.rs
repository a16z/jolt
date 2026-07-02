//! Round-trips the lattice packed-column descriptions through
//! `jolt-openings::PrefixPacking` — the consumer contract: `jolt-claims`
//! names ids and arities, `jolt-openings` owns slot assignment.

use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
use jolt_claims::protocols::jolt::lattice::{
    packed_column_leaf, precommitted_packed_columns, proof_packed_columns, LatticeColumn,
    PrecommittedPackingShape, ProofPackingShape,
};
use jolt_openings::PrefixPacking;

#[expect(clippy::unwrap_used)]
fn proof_shape() -> ProofPackingShape {
    ProofPackingShape {
        ra_layout: JoltRaPolynomialLayout::new(2, 1, 2).unwrap(),
        log_t: 10,
        log_k_chunk: 8,
        untrusted_advice_word_vars: Some(4),
    }
}

#[test]
#[expect(clippy::unwrap_used)]
fn proof_columns_build_a_prefix_packing() {
    let columns = proof_packed_columns(&proof_shape()).unwrap();
    let packing = PrefixPacking::new(columns.clone()).unwrap();

    // Every declared column got a slot with its declared arity, and every
    // relation-produced column has a claim source for the packed statement.
    for (column, num_vars) in &columns {
        assert_eq!(packing[column].num_vars, *num_vars);
        if matches!(column, LatticeColumn::Committed(_)) {
            assert!(packed_column_leaf(*column).is_some());
        }
    }

    // 5 Ra columns + 8 chunk columns at 18 vars, msb at 10, advice at 15:
    // 13 * 2^18 + 2^10 + 2^15 cells round up to a 2^22 packed hypercube.
    assert_eq!(packing.packed_num_vars, 22);
}

#[test]
#[expect(clippy::unwrap_used)]
fn precommitted_columns_build_a_prefix_packing() {
    let shape = PrecommittedPackingShape {
        bytecode_chunks: 4,
        log_bytecode_rows: 6,
        imm_byte_width: 16,
        program_image_log_words: Some(12),
        trusted_advice_word_vars: Some(4),
    };
    let columns = precommitted_packed_columns(&shape).unwrap();
    let packing = PrefixPacking::new(columns.clone()).unwrap();

    for (column, num_vars) in &columns {
        assert_eq!(packing[column].num_vars, *num_vars);
    }
}

#[test]
#[expect(clippy::unwrap_used)]
fn packing_is_deterministic_across_rebuilds() {
    let columns = proof_packed_columns(&proof_shape()).unwrap();
    assert_eq!(
        PrefixPacking::new(columns.clone()).unwrap(),
        PrefixPacking::new(columns).unwrap()
    );
}
