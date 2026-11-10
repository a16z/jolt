use super::super::test_support::{assert_kernel_parity, structured_fixture};
use super::*;
use crate::optimized::parity::ExceptionalEq;
use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
use jolt_field::Ring;
use jolt_poly::Polynomial;
use jolt_verifier::stages::stage4::registers_read_write_checking::{
    RegistersReadWriteChallenges, RegistersReadWriteInputClaims,
};
use jolt_witness::JoltWitnessOracle;

#[test]
fn sparse_metal_rounds_preserve_boundaries_and_lut_transition() {
    let mut slot = MetalRegistersReadWrite::new().unwrap();
    slot.min_rows = 0;
    // Exercise resident direct folds before transferring the surviving prefix.
    slot.tail_entries = 16;
    let log_t = 7;
    let sequence = |len: usize, seed: u64| {
        (0..len)
            .map(|i| F::from_u64(seed + 2 * i as u64))
            .collect::<Vec<_>>()
    };
    let round_challenges = sequence(log_t + REGISTER_ADDRESS_BITS, 19);
    for exceptional in [
        None,
        Some(ExceptionalEq::Zero),
        Some(ExceptionalEq::One),
        Some(ExceptionalEq::ZeroPrefix),
    ] {
        // Includes duplicate operands, absent sides, writes, high addresses,
        // interleaved no-ops, and 65 live rows: the first sparse fold has an
        // odd 9-row prefix, so the fused message must fill a missing high row.
        structured_fixture(65).with_plane(log_t, |witness| {
            let point = exceptional.map_or_else(
                || sequence(log_t, 47),
                |case| case.point(log_t, round_challenges[0]),
            );
            let evaluate = |polynomial| {
                let table = JoltWitnessOracle::<F>::oracle_table(
                    witness,
                    JoltPolynomialId::Virtual(polynomial),
                )
                .unwrap();
                Polynomial::new(table).evaluate(&point)
            };
            let claims = RegistersReadWriteInputClaims {
                rd_write_value: evaluate(JoltVirtualPolynomial::RdWriteValue),
                rs1_value: evaluate(JoltVirtualPolynomial::Rs1Value),
                rs2_value: evaluate(JoltVirtualPolynomial::Rs2Value),
            };
            let points = RegistersReadWriteInputClaims {
                rd_write_value: point.clone(),
                rs1_value: point.clone(),
                rs2_value: point,
            };
            let gamma = F::from_u64(37);
            let challenges = RegistersReadWriteChallenges { gamma };
            let input_claim =
                claims.rd_write_value + gamma * claims.rs1_value + gamma * gamma * claims.rs2_value;
            for (phase1, phase2) in [(log_t, 0), (0, REGISTER_ADDRESS_BITS)] {
                let relation = RegistersReadWriteChecking::<F>::new(ReadWriteDimensions::new(
                    log_t,
                    REGISTER_ADDRESS_BITS,
                    phase1,
                    phase2,
                ));
                assert_kernel_parity(
                    &slot,
                    witness,
                    &relation,
                    &claims,
                    &points,
                    &challenges,
                    input_claim,
                    &round_challenges,
                );
            }
        });
    }
}
