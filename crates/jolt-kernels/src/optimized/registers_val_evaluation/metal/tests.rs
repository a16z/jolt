use super::*;
use crate::optimized::registers_read_write::test_support::{
    assert_kernel_parity_with_session, structured_fixture,
};
use crate::optimized::registers_read_write::RegisterCycleRow;
use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
use jolt_claims::NoChallenges;
use jolt_field::Ring;
use jolt_poly::Polynomial;
use jolt_verifier::stages::stage5::registers_val_evaluation::RegistersValEvaluationInputClaims;
use jolt_witness::{collect_bundles, JoltWitnessOracle, RowSource};
use std::sync::Mutex;

#[test]
fn retained_seed_value_rounds_match_reference_through_handoff() {
    let device = Arc::new(Device::system_default().unwrap());
    let mut slot = MetalRegistersVal::new(Arc::clone(&device)).unwrap();
    slot.min_prefix = 1;
    slot.cpu_tail = 1;
    let log_t = 10;
    let sequence = |len: usize, seed: u64| {
        (0..len)
            .map(|i| F::from_u64(seed + 2 * i as u64))
            .collect::<Vec<_>>()
    };
    structured_fixture(257).with_plane(log_t, |witness| {
        let rows: Vec<RegisterCycleRow> = collect_bundles(witness, 1 << log_t).unwrap();
        let rd = rows
            .iter()
            .map(|row| row.rd.map(|(index, ..)| index))
            .collect();
        // Use the same canonical source collector conversion, not a second oracle.
        let mut increments: Vec<i128> =
            rows.iter().map(RegisterCycleRow::write_increment).collect();
        let prefix = increments
            .iter()
            .rposition(|&inc| inc != 0)
            .map_or(0, |i| i + 1);
        increments.truncate(prefix);
        let owner = Arc::new(Mutex::new(
            DeviceBuffer::from_slice(&device, &increments).unwrap(),
        ));
        let weak = Arc::downgrade(&owner);
        let mut session = ProofSession::default();
        session.park(SharedRdIndices::new(rd, 1 << log_t));
        session.park(SharedRdIncrements {
            values: owner,
            device: Arc::clone(&device),
            source: RowSource::random_access(witness).unwrap(),
            cycles: 1 << log_t,
        });
        let relation = RegistersValEvaluation::<F>::new(TraceDimensions::new(log_t));
        let point = sequence(REGISTER_ADDRESS_BITS + log_t, 709);
        let grid = JoltWitnessOracle::<F>::oracle_table(
            witness,
            JoltPolynomialId::Virtual(JoltVirtualPolynomial::RegistersVal),
        )
        .unwrap();
        let claim = Polynomial::new(grid).evaluate(&point);
        let inputs = RegistersValEvaluationInputClaims {
            registers_val: claim,
        };
        let points = RegistersValEvaluationInputClaims {
            registers_val: point,
        };
        assert_kernel_parity_with_session(
            &mut session,
            &slot,
            witness,
            &relation,
            &inputs,
            &points,
            &NoChallenges::default(),
            claim,
            &sequence(log_t, 823),
        );
        assert!(session.state::<SharedRdIncrements>().is_none());
        assert!(weak.upgrade().is_none());
    });
    assert!(slot.gpu.rounds.load(std::sync::atomic::Ordering::Relaxed) >= 3);
}
