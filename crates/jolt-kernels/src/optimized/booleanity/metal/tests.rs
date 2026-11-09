#![expect(clippy::unwrap_used, reason = "tests assert exact field arithmetic")]

use jolt_field::{Field, Ring};
use jolt_verifier::stages::stage6b::booleanity::BooleanityInputClaims;
use jolt_witness::witnesses::FusedInc;

use super::super::super::constant_suffix::ConstantSuffix;
use super::*;

#[test]
fn compact_booleanity_matches_dense_rounds_and_releases_shared_rows() {
    let gpu = MetalBooleanity::new().unwrap();
    for (bits, log_t, prefix, gamma, cold_tail, exceptional) in [
        (4, 12, 33, 0, false, false),
        (4, 12, 4093, 13, true, true),
        (4, 12, 0, 13, false, false),
        (8, 14, 253, 13, true, true),
    ] {
        let cycles = 1 << log_t;
        let layout = JoltRaPolynomialLayout::new(128 / bits, 3, 4).unwrap();
        let selectors: Vec<_> = layout
            .polynomials()
            .map(|polynomial| match polynomial {
                JoltRaPolynomial::Instruction(i) => ColumnSelector::Instruction(
                    RaChunkSelector::new(i, layout.instruction(), bits).unwrap(),
                ),
                JoltRaPolynomial::Bytecode(i) => ColumnSelector::Bytecode(
                    RaChunkSelector::new(i, layout.bytecode(), bits).unwrap(),
                ),
                JoltRaPolynomial::Ram(i) => {
                    ColumnSelector::Ram(RaChunkSelector::new(i, layout.ram(), bits).unwrap())
                }
            })
            .collect();
        let mut state = 901u128;
        let mut rows: Vec<_> = (0..prefix)
            .map(|row| {
                state = state.wrapping_mul(0xda94_2042_e4dd_58b5).wrapping_add(1);
                InstructionCycleRow::new(
                    state,
                    None,
                    false,
                    (row * 31) & ((1 << (3 * bits)) - 1),
                    (row % 3 != 0).then_some((state as u64) & ((1 << (4 * bits)) - 1)),
                    FusedInc::default(),
                )
            })
            .collect();
        rows.push(InstructionCycleRow::new(
            0x9876_5432_10fe_dcba_7654_3210_9abc_def0,
            None,
            false,
            0x321,
            (!cold_tail).then_some(0),
            FusedInc::default(),
        ));
        let rows = Arc::new(ConstantSuffix::new(rows, cycles));
        let address: Vec<_> = (0..bits)
            .map(|i| AkitaField::from_u64(300 + 13 * i as u64))
            .collect();
        let scales = eq_table(&address);
        let mut dense: Vec<_> = selectors
            .iter()
            .map(|selector| {
                Polynomial::new(
                    (0..cycles)
                        .map(|row| {
                            selector
                                .index(&rows[row])
                                .map_or(AkitaField::zero(), |hot| scales[hot])
                        })
                        .collect(),
                )
            })
            .collect();
        let mut cycle_point: Vec<_> = (0..log_t)
            .map(|i| AkitaField::from_u64(700 + 29 * i as u64))
            .collect();
        if exceptional {
            cycle_point[log_t - 3] = AkitaField::zero();
        }
        let scalar = AkitaField::from_u64(7);
        let mut equality = Polynomial::new(
            eq_table(&cycle_point)
                .into_iter()
                .map(|value| scalar * value)
                .collect(),
        );
        let gamma = AkitaField::from_u64(gamma);
        let weights = gamma_powers(gamma, selectors.len());
        let tables = weights
            .iter()
            .map(|rho| {
                scales
                    .iter()
                    .map(|v| if rho.is_zero() { *v } else { *rho * *v })
                    .collect()
            })
            .collect();
        let unscale = gamma.inverse().map_or_else(
            || vec![AkitaField::from_u64(1); selectors.len()],
            |inverse| gamma_powers(inverse, selectors.len()),
        );
        let cpu = OptimizedBooleanityCycleKernel {
            progress: RoundProgress::new(log_t),
            eq: GruenSplitEqPolynomial::new_with_scaling(
                &cycle_point,
                BindingOrder::LowToHigh,
                Some(scalar),
            ),
            tables: LazyFoldedRa::new(
                tables,
                BooleanityChunks {
                    rows: Arc::clone(&rows),
                    prefix_len: rows.prefix_len(),
                    selectors,
                    instruction_chunks: layout.instruction(),
                    chunk_bits: bits,
                },
            ),
            gamma_powers: weights.clone(),
            opening_unscale: unscale,
            openings: layout.openings(JoltRelationId::Booleanity).collect(),
        };
        let mut session = ProofSession::default();
        let owner = ResidentInstructionRows::shared(&mut session, &rows).unwrap();
        assert!(Arc::ptr_eq(
            &owner,
            &ResidentInstructionRows::shared(&mut session, &rows).unwrap()
        ));
        let other = Arc::new(rows.as_ref().clone());
        assert!(
            ResidentInstructionRows::shared(&mut session, &other).is_err(),
            "a different allocation cannot reuse the proof's selectors"
        );
        let weak = Arc::downgrade(&owner);
        let mut kernel = MetalBooleanityKernel {
            cpu,
            rows: Some(owner),
            gpu: Arc::clone(&gpu.0),
        };
        let mut claim = None;
        let mut gpu_rounds = 0;
        for round in 0..log_t {
            let half = equality.evals().len() / 2;
            let samples: Vec<_> = (0..4)
                .map(|t| {
                    let t = AkitaField::from_u64(t);
                    let value = |poly: &Polynomial<AkitaField>, pair: usize| {
                        let lo = poly.evals()[2 * pair];
                        lo + t * (poly.evals()[2 * pair + 1] - lo)
                    };
                    (0..half)
                        .map(|pair| {
                            value(&equality, pair)
                                * dense
                                    .iter()
                                    .zip(&weights)
                                    .map(|(poly, rho)| {
                                        let v = value(poly, pair);
                                        rho.square() * v * (v - AkitaField::from_u64(1))
                                    })
                                    .sum::<AkitaField>()
                        })
                        .sum::<AkitaField>()
                })
                .collect();
            let previous = claim.unwrap_or(samples[0] + samples[1]);
            assert_eq!(previous, samples[0] + samples[1]);
            let expected = UnivariatePoly::from_evals(&samples);
            let actual = if let Some(message) = kernel.gpu_message(round, previous).unwrap() {
                gpu_rounds += 1;
                message
            } else {
                kernel.cpu.message(round, previous).unwrap()
            };
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
            kernel.cpu.bind(challenge);
        }
        assert_eq!(gpu_rounds, if bits == 4 { 2 } else { 1 });
        assert!(kernel.rows.is_none());
        assert!(
            weak.upgrade().is_none(),
            "the weak session carry retains no device storage"
        );
        let output = kernel
            .output_claims(&BooleanityInputClaims {
                address_phase: AkitaField::zero(),
            })
            .unwrap();
        assert_eq!(
            output.opening_values(),
            dense.iter().map(|p| p.evals()[0]).collect::<Vec<_>>()
        );
    }
}
