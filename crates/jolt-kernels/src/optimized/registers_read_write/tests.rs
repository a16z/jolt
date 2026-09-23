use jolt_claims::protocols::jolt::geometry::dimensions::{
    ReadWriteDimensions, REGISTER_ADDRESS_BITS,
};
use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
use jolt_field::{Fr, Ring};
use jolt_poly::Polynomial;
use jolt_verifier::stages::stage4::registers_read_write_checking::{
    RegistersReadWriteChallenges, RegistersReadWriteChecking, RegistersReadWriteInputClaims,
};
use jolt_witness::JoltWitnessOracle;

use super::test_support::{
    assert_kernel_parity, assert_nontrivial, challenge_sequence, structured_fixture, TraceFixture,
};
use super::OptimizedRegistersReadWrite;

fn run_parity(fixture: TraceFixture, log_t: usize, seed: u64) {
    fixture.with_plane(log_t, |backend| {
        let relation = RegistersReadWriteChecking::<Fr>::new(ReadWriteDimensions::new(
            log_t,
            REGISTER_ADDRESS_BITS,
            log_t,
            0,
        ));
        let r_cycle = challenge_sequence(log_t, seed ^ 0xA5A5);
        let evaluate = |polynomial: JoltVirtualPolynomial| {
            let table = JoltWitnessOracle::<Fr>::oracle_table(
                backend,
                JoltPolynomialId::Virtual(polynomial),
            )
            .unwrap();
            Polynomial::new(table).evaluate(&r_cycle)
        };
        let gamma = Fr::from_u64(0x5EED_1234_5678_9ABC);
        let claims = RegistersReadWriteInputClaims {
            rd_write_value: evaluate(JoltVirtualPolynomial::RdWriteValue),
            rs1_value: evaluate(JoltVirtualPolynomial::Rs1Value),
            rs2_value: evaluate(JoltVirtualPolynomial::Rs2Value),
        };
        let points = RegistersReadWriteInputClaims {
            rd_write_value: r_cycle.clone(),
            rs1_value: r_cycle.clone(),
            rs2_value: r_cycle,
        };
        let input_claim =
            claims.rd_write_value + gamma * claims.rs1_value + gamma * gamma * claims.rs2_value;
        assert_nontrivial(input_claim);
        let round_challenges = challenge_sequence(log_t + REGISTER_ADDRESS_BITS, seed);
        assert_kernel_parity(
            &OptimizedRegistersReadWrite,
            backend,
            &relation,
            &claims,
            &points,
            &RegistersReadWriteChallenges { gamma },
            input_claim,
            &round_challenges,
        );
    });
}

#[test]
fn parity_structured_odd_log_t() {
    run_parity(structured_fixture(8), 3, 17);
}

#[test]
fn parity_structured_even_log_t() {
    run_parity(structured_fixture(16), 4, 23);
}

#[test]
fn parity_past_lut_saturation() {
    // log_t = 6 runs three LUT-mode binds, the deref at the fourth, and
    // two more cycle binds on direct field coefficients.
    run_parity(structured_fixture(60), 6, 29);
}

#[test]
fn parity_minimal_padded_trace() {
    // Three real cycles padded to four: exercises the padding rows and
    // registers that are never touched.
    let mut fixture = TraceFixture::new();
    fixture.op(Some(3), Some(1), Some(2));
    fixture.op(Some(3), Some(3), None);
    fixture.op(None, Some(3), Some(3));
    run_parity(fixture, 2, 31);
}

#[test]
fn parity_single_cycle_round() {
    let mut fixture = TraceFixture::new();
    fixture.op(Some(9), Some(9), Some(9));
    fixture.op(Some(9), None, Some(9));
    run_parity(fixture, 1, 41);
}
