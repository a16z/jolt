use crate::support::{
    assert_rejects, clear_base_case, tamper_native_final_input, tamper_public_output, zk_base_case,
};

#[test]
fn tampered_public_output_rejects() {
    let mut case = clear_base_case();
    tamper_public_output(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_zk_public_output_rejects() {
    let mut case = zk_base_case();
    tamper_public_output(&mut case);
    assert_rejects(case.verify_zk());
}

#[test]
fn tampered_native_final_input_rejects() {
    let mut case = clear_base_case();
    tamper_native_final_input(&mut case);
    assert_rejects(case.verify_clear());
}

#[test]
fn tampered_zk_native_final_input_rejects() {
    let mut case = zk_base_case();
    tamper_native_final_input(&mut case);
    assert_rejects(case.verify_zk());
}
