use crate::support::{
    assert_accepts, clear_base_case, clear_multiround_case, zk_base_case, zk_multiround_case,
};

#[test]
fn clear_valid_assist_proof_accepts() {
    assert_accepts(clear_base_case().verify_clear());
}

#[test]
fn clear_multiround_valid_assist_proof_accepts() {
    assert_accepts(clear_multiround_case().verify_clear());
}

#[test]
fn zk_valid_assist_proof_accepts() {
    assert_accepts(zk_base_case().verify_zk());
}

#[test]
fn zk_multiround_valid_assist_proof_accepts() {
    assert_accepts(zk_multiround_case().verify_zk());
}
