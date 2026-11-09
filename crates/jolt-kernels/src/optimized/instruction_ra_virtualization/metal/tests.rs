#![expect(clippy::unwrap_used, reason = "tests assert exact field arithmetic")]

use jolt_claims::protocols::jolt::relations::instruction::InstructionRaVirtualizationInputClaims;
use jolt_field::Ring;
use jolt_poly::Polynomial;
use jolt_witness::witnesses::FusedInc;

use super::*;

#[test]
fn compact_ra_rounds_match_dense_multilinears_through_handoff() {
    let gpu = MetalInstructionRa::new().unwrap();
    let log_t = 12;
    let cycles = 1usize << log_t;
    for (bits, prefix, gamma, exceptional) in [
        (4, 33, 0, false),
        (4, 4093, 13, true),
        (4, 0, 13, false),
        (8, 253, 13, true),
    ] {
        let columns = 128 / bits;
        let virtuals = columns / 4;
        let gamma = AkitaField::from_u64(gamma);
        let address: Vec<_> = (0..128)
            .map(|i| AkitaField::from_u64(300 + 13 * i))
            .collect();
        let mut cycle_point: Vec<_> = (0..log_t)
            .map(|i| AkitaField::from_u64(7000 + 29 * i as u64))
            .collect();
        if exceptional {
            cycle_point[log_t - 2] = AkitaField::from_u64(1);
        }
        let tail = 0x9876_5432_10fe_dcba_7654_3210_9abc_def0_u128;
        let mut state = 901u128;
        let indices: Vec<_> = (0..cycles)
            .map(|row| {
                state = state.wrapping_mul(0xda94_2042_e4dd_58b5).wrapping_add(1);
                if row < prefix {
                    state
                } else {
                    tail
                }
            })
            .collect();
        let rows = indices
            .iter()
            .take(prefix + 1)
            .map(|&index| {
                InstructionCycleRow::new(index, None, false, 0, None, FusedInc::default())
            })
            .collect();
        let cpu = OptimizedInstructionRaVirtualizationKernel::new(
            log_t,
            virtuals,
            4,
            &address,
            &cycle_point,
            bits,
            Arc::new(ConstantSuffix::new(rows, cycles)),
            gamma,
        )
        .unwrap();
        let resident_rows = match cpu.folded_ra.as_ref() {
            Some(LazyFoldedRa::Lazy { source, .. }) => Some(
                ResidentInstructionRows::shared(&mut ProofSession::default(), source.rows())
                    .unwrap(),
            ),
            _ => None,
        };
        let mut kernel = MetalInstructionRaKernel {
            cpu,
            indices: resident_rows,
            device: None,
            tail_rows: 32,
            gpu: Arc::clone(&gpu.0),
        };
        // Dense point-mass address folds, independent of lazy branch tables.
        let chunks = committed_address_chunks(&address, bits);
        let mut dense: Vec<_> = chunks
            .iter()
            .enumerate()
            .map(|(column, point)| {
                let scale = eq_table(point);
                let shift = (columns - 1 - column) * bits;
                Polynomial::new(
                    indices
                        .iter()
                        .map(|index| scale[((index >> shift) & ((1 << bits) - 1)) as usize])
                        .collect(),
                )
            })
            .collect();
        let mut equality = Polynomial::new(eq_table(&cycle_point));
        let mut claim = None;
        let mut saw_device = false;
        let mut saw_dense_bind = false;
        let mut pending_bind = None;
        for round in 0..log_t {
            let half = equality.evals().len() / 2;
            let samples: Vec<_> = (0..6)
                .map(|t| {
                    let t = AkitaField::from_u64(t);
                    let value = |poly: &Polynomial<AkitaField>, pair: usize| {
                        let lo = poly.evals()[2 * pair];
                        lo + t * (poly.evals()[2 * pair + 1] - lo)
                    };
                    (0..half)
                        .map(|pair| {
                            let mut sum = AkitaField::zero();
                            let mut weight = AkitaField::from_u64(1);
                            for group in dense.chunks_exact(4) {
                                sum += weight
                                    * group
                                        .iter()
                                        .map(|poly| value(poly, pair))
                                        .product::<AkitaField>();
                                weight *= gamma;
                            }
                            value(&equality, pair) * sum
                        })
                        .sum::<AkitaField>()
                })
                .collect();
            let previous = claim.unwrap_or(samples[0] + samples[1]);
            assert_eq!(previous, samples[0] + samples[1]);
            let expected = UnivariatePoly::from_evals(&samples);
            saw_dense_bind |= kernel
                .device
                .as_ref()
                .is_some_and(|state| state.prefix > kernel.tail_rows);
            let actual = kernel
                .round_with_cutoff(pending_bind, round, previous, 0)
                .unwrap();
            saw_device |= kernel.device.is_some();
            assert_eq!(
                actual.coefficients(),
                expected.coefficients(),
                "bits={bits}, prefix={prefix}, round={round}"
            );
            let challenge = AkitaField::from_u64([0, 1, 43][round % 3]);
            claim = Some(actual.evaluate(challenge));
            for polynomial in &mut dense {
                polynomial.bind_with_order(challenge, BindingOrder::LowToHigh);
            }
            equality.bind_with_order(challenge, BindingOrder::LowToHigh);
            pending_bind = Some(challenge);
        }
        kernel.finish_rounds(pending_bind.unwrap()).unwrap();
        assert!(saw_device);
        if prefix > 4000 {
            assert!(saw_dense_bind);
        }
        assert!(
            kernel.indices.is_none(),
            "materialization releases the device source"
        );
        let output = kernel
            .output_claims(&InstructionRaVirtualizationInputClaims {
                instruction_ra: Vec::new(),
            })
            .unwrap();
        assert_eq!(
            output.committed_instruction_ra,
            dense.iter().map(|poly| poly.evals()[0]).collect::<Vec<_>>()
        );
    }
}
