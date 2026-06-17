use crate::support::{
    assert_rejects, clear_base_case, tamper_dense_commitment, tamper_hyrax_opening_row,
    tamper_hyrax_opening_scalar, tamper_opening_claim_eval, tamper_opening_claim_point,
};

#[test]
fn tampered_opening_claim_point_rejects() {
    let mut case = clear_base_case();
    tamper_opening_claim_point(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_opening_claim_eval_rejects() {
    let mut case = clear_base_case();
    tamper_opening_claim_eval(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_hyrax_opening_row_rejects() {
    let mut case = clear_base_case();
    tamper_hyrax_opening_row(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_hyrax_opening_scalar_rejects() {
    let mut case = clear_base_case();
    tamper_hyrax_opening_scalar(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_dense_commitment_rejects() {
    let mut case = clear_base_case();
    tamper_dense_commitment(&mut case);
    assert_rejects(case.verify_clear());
}
