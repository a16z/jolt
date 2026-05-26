use std::collections::BTreeSet;

use jolt_prover_harness::matrix::REQUIRED_FEATURE_MATRIX;

#[test]
fn required_feature_matrix_has_no_duplicate_rows() {
    let mut names = BTreeSet::new();
    for row in REQUIRED_FEATURE_MATRIX {
        assert!(names.insert(row.name), "duplicate feature row {}", row.name);
    }
}

#[test]
fn required_feature_matrix_covers_core_modes() {
    let names = REQUIRED_FEATURE_MATRIX
        .iter()
        .map(|row| row.name)
        .collect::<BTreeSet<_>>();

    for required in [
        "default",
        "field-inline",
        "zk",
        "zk,field-inline",
        "core-fixtures",
        "core-fixtures,zk",
        "core-fixtures,field-inline",
        "core-fixtures,zk,field-inline",
    ] {
        assert!(names.contains(required), "missing feature row {required}");
    }
}
