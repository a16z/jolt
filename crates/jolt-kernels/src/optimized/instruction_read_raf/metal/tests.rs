#![expect(clippy::unwrap_used, reason = "exact reference comparison")]

use std::num::NonZeroUsize;

use jolt_field::Ring;

use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafInputClaims;
use jolt_witness::witnesses::{InstructionRafFlag, LookupIndex, TableIndex};

use super::super::tests::{challenge, fixture_rows, input_claim, pack};
use super::*;
use crate::reference::instruction_read_raf::InstructionReadRafKernel;

#[test]
fn compact_read_raf_matches_reference_through_device_handoff() {
    let gpu = MetalInstructionReadRaf::new().unwrap();
    address::check_suffixes(&gpu.0).unwrap();
    // Nine factors covers the cached constant products; seventeen factors
    // reaches width four on a small domain with the production lazy policy.
    for (log_t, virtuals, prefix, zero_tail) in [(10, 8, 1001, true), (16, 16, 40001, false)] {
        let dimensions = InstructionReadRafDimensions::new(
            log_t,
            2 * RISCV_XLEN,
            NonZeroUsize::new(virtuals).unwrap(),
        );
        let mut witness = fixture_rows(log_t, 947);
        for (index, row) in witness[..prefix].iter_mut().enumerate() {
            if index.is_multiple_of(3) {
                row.lookup_index = LookupIndex((1u128 << 32) | (index % 5) as u128);
                row.table_index = TableIndex(Some(RangePrefixSums::<AkitaField>::table().index()));
                row.raf_flag = InstructionRafFlag(true);
            }
        }
        let mut tail = witness[0];
        if zero_tail {
            tail.lookup_index = LookupIndex(0);
            tail.table_index = TableIndex(None);
            tail.raf_flag = InstructionRafFlag(false);
        }
        witness[prefix..].fill(tail);
        let gamma = AkitaField::from_u64(if zero_tail { 0 } else { 37 });
        let mut point: Vec<_> = (0..log_t).map(challenge::<AkitaField>).collect();
        point[log_t - if virtuals == 8 { 2 } else { 3 }] = AkitaField::from_u64(1);
        let mut reference =
            InstructionReadRafKernel::new(dimensions, &point, witness.clone(), gamma).unwrap();
        let source = Arc::new(ConstantSuffix::from(pack(&witness)));
        let mut session = ProofSession::default();
        let owner = ResidentInstructionRows::shared(&mut session, &source).unwrap();
        let weak = Arc::downgrade(&owner);
        let cpu = OptimizedInstructionReadRafKernel::new_with_scanner(
            dimensions,
            &point,
            Arc::clone(&source),
            gamma,
            |host| {
                let device =
                    address::DeviceAddressScan::try_new(Arc::clone(&gpu.0), &mut session, &host, 0)
                        .map_err(compute_error)?
                        .unwrap();
                Ok(Box::new(device))
            },
        )
        .unwrap();
        let mut kernel = Box::new(MetalReadRafKernel {
            cpu,
            rows: Some(owner),
            device: None,
            gpu: Arc::clone(&gpu.0),
        });
        let mut claim = input_claim(&witness, &point, gamma);
        let rounds = reference.num_rounds();
        let mut saw_device = false;
        let mut saw_dense_bind = false;
        let mut lazy_widths = Vec::new();
        let draw = |round| {
            if zero_tail && round == 0 {
                AkitaField::from_u64(1)
            } else {
                challenge::<AkitaField>(round)
            }
        };
        for round in 0..rounds {
            let bind = round.checked_sub(1).map(draw);
            let expected = reference.prove_round(bind, round, claim).unwrap();
            let was_resident = kernel
                .device
                .as_ref()
                .is_some_and(|state| state.prefixes[0].max(state.prefixes[1]) > CPU_TAIL_ROWS);
            let actual = kernel.round_with_cutoff(bind, round, claim, 0).unwrap();
            assert_eq!(
                actual.coefficients(),
                expected.coefficients(),
                "virtuals={virtuals}, round={round}"
            );
            saw_device |= kernel.device.is_some();
            saw_dense_bind |= was_resident;
            if let Some(CycleState {
                ra: LazyFoldedRa::Lazy { width, .. },
                ..
            }) = &kernel.cpu.cycle
            {
                lazy_widths.push(*width);
            }
            claim = expected.evaluate(draw(round));
        }
        assert!(saw_device);
        if virtuals == 16 {
            assert!(saw_dense_bind);
            assert!(lazy_widths.contains(&2) && lazy_widths.contains(&4));
        }
        let final_bind = draw(rounds - 1);
        reference.finish_rounds(final_bind).unwrap();
        kernel.finish_rounds(final_bind).unwrap();
        let inputs = InstructionReadRafInputClaims {
            lookup_output: AkitaField::zero(),
            left_lookup_operand: AkitaField::zero(),
            right_lookup_operand: AkitaField::zero(),
        };
        let expected = reference.output_claims(&inputs).unwrap();
        let actual = kernel.output_claims(&inputs).unwrap();
        assert_eq!(actual.instruction_ra, expected.instruction_ra);
        assert_eq!(actual.lookup_table_flags, expected.lookup_table_flags);
        assert_eq!(actual.instruction_raf_flag, expected.instruction_raf_flag);
        kernel.park_residue(&mut session);
        assert!(
            weak.upgrade().is_some(),
            "next stage retains the uploaded selectors"
        );
        let consumer = ResidentInstructionRows::shared(&mut session, &source).unwrap();
        let second = ResidentInstructionRows::shared(&mut session, &source).unwrap();
        assert!(Arc::ptr_eq(&consumer, &second));
        drop(consumer);
        drop(second);
        assert!(
            weak.upgrade().is_none(),
            "the session returns to a weak carry"
        );
    }
}
