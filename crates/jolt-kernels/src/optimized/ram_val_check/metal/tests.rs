use super::*;
use crate::optimized::registers_read_write::test_support::assert_kernel_parity_with_session;
use crate::optimized::testing::{with_ram_fixture_field, FixtureShape, RamOp};
use crate::reference::views::address_fold;
use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
use jolt_claims::protocols::jolt::geometry::ram::{
    ram_inc_val_check, ram_ra_val_check, RamValCheckInit,
};
use jolt_claims::protocols::jolt::relations::ram::{RamValCheckChallenges, RamValCheckInputClaims};
use jolt_field::Ring;
use jolt_poly::LtPolynomial;

#[test]
fn captured_ram_sparse_rounds_match_reference_through_handoff() {
    let mut slot = MetalRamVal::new().unwrap();
    slot.min_prefix = 1;
    let shape = FixtureShape {
        log_t: 16,
        ram_k: 8,
    };
    let mut ops = vec![RamOp::None; (1 << 15) + 256];
    for (index, op) in ops.iter_mut().enumerate() {
        *op = match index % 8 {
            0 => RamOp::Write {
                word: 2,
                post: u64::MAX,
            },
            1 => RamOp::Read { word: 3 },
            2 => RamOp::Write { word: 2, post: 1 },
            3 | 4 => RamOp::Read { word: 2 },
            5 => RamOp::Write { word: 3, post: 17 },
            _ => RamOp::None,
        };
    }
    let sequence = |len: usize, seed: u64| {
        (0..len)
            .map(|i| F::from_u64(seed + i as u64 * 2))
            .collect::<Vec<_>>()
    };
    with_ram_fixture_field::<F, _>(shape, Vec::new(), ops, |witness| {
        let mut session = ProofSession::default();
        let columns = RamAccessColumns::collect_full(&mut session, witness, shape.log_t).unwrap();
        let rows = witness.random_access().unwrap();
        let (inc, carry) =
            SparseSource::capture(&columns, rows.clone(), 1 << shape.log_t, 512 << 20);
        let carry = carry.unwrap();
        assert_eq!(carry.prefix % 2, 1);
        let (fallback, declined) =
            SparseSource::capture(&columns, rows, 1 << shape.log_t, 128 << 10);
        assert!(declined.is_none());
        let expected = witness
            .oracle_table(ram_inc_val_check().polynomial_id())
            .unwrap();
        assert_eq!(inc, expected[..inc.len()]);
        assert_eq!(fallback, expected[..fallback.len()]);
        session.park(carry);
        drop(columns);
        let r_address = sequence(shape.log_k(), 7);
        let r_cycle = sequence(shape.log_t, 23);
        let gamma = F::from_u64(47);
        let relation = RamValCheck::<F>::new(
            TraceDimensions::new(shape.log_t),
            shape.log_k(),
            RamValCheckInit::full(F::zero()),
        );
        let claims = RamValCheckInputClaims {
            ram_val: F::zero(),
            ram_val_final: F::zero(),
            untrusted_advice: None,
            trusted_advice: None,
            program_image: None,
        };
        let points = RamValCheckInputClaims {
            ram_val: [r_address.clone(), r_cycle.clone()].concat(),
            ram_val_final: r_address.clone(),
            untrusted_advice: None,
            trusted_advice: None,
            program_image: None,
        };
        let challenges = RamValCheckChallenges { gamma };
        let ra = address_fold::<F>(witness, ram_ra_val_check(), shape.log_t, &r_address).unwrap();
        let lt = LtPolynomial::evaluations(&r_cycle);
        let claim = expected
            .iter()
            .zip(&ra)
            .zip(&lt)
            .map(|((&inc, &ra), &lt)| inc * ra * (lt + gamma))
            .sum();
        assert_kernel_parity_with_session(
            &mut session,
            &slot,
            witness,
            &relation,
            &claims,
            &points,
            &challenges,
            claim,
            &sequence(shape.log_t, 101),
        );
        assert!(session.state::<SparseSource>().is_none());
    });
}
