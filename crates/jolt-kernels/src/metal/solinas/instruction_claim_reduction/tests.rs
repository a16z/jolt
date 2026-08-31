use core::mem::{align_of, size_of};

use crate::optimized::instruction_claim_reduction::InstructionOperandRow;
use jolt_field::{AkitaField, FromPrimitiveInt};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, UnivariatePoly};

use super::*;

fn operand_planes(
    core: &[InstructionClaimCoreRow],
    right_input: &[InstructionClaimRightInput],
) -> InstructionClaimOperandPlanes {
    InstructionClaimOperandPlanes::new(
        core.iter().map(|row| row.lookup_output()).collect(),
        core.iter().map(|row| row.left_lookup_operand()).collect(),
        core.iter()
            .map(|row| InstructionClaimRightLookup::new(row.right_lookup_operand()))
            .collect(),
        core.iter()
            .map(|row| row.left_instruction_input())
            .collect(),
        right_input.to_vec(),
    )
    .expect("the operand planes are valid")
}

#[test]
fn metal_entry_points_compile() {
    let Ok(context) = super::super::SolinasMetal::for_akita() else {
        return;
    };
    for name in [
        MATERIALIZE_PIPELINE,
        "solinas_instruction_claim_materialize_stage1_rows",
        TRANSITION_PIPELINE,
        CORE_OPENING_PIPELINE,
        ALIASED_OPENING_PIPELINE,
        "solinas_instruction_claim_open_stage1_lookup_operands",
        ALL_OPENING_PIPELINE,
        REDUCTION_PIPELINE,
    ] {
        let pipeline = context
            .compile_named_pipeline(name)
            .expect("instruction claim pipeline should compile");
        let limits = super::super::SolinasMetal::limits(&pipeline);
        assert_eq!(limits.thread_execution_width, INSTRUCTION_CLAIM_SIMD_WIDTH);
        assert!(limits.max_total_threads_per_threadgroup >= 128);
    }
}

#[test]
fn native_plane_abis_and_word_order_are_fixed() {
    assert_eq!(INSTRUCTION_CLAIM_AKITA_OFFSET, 0xffff_a7f7);
    assert_eq!(size_of::<InstructionClaimCoreRow>(), 40);
    assert_eq!(align_of::<InstructionClaimCoreRow>(), 8);
    assert_eq!(size_of::<InstructionClaimRightInput>(), 16);
    assert_eq!(align_of::<InstructionClaimRightInput>(), 8);
    assert_eq!(size_of::<InstructionClaimRightLookup>(), 16);
    assert_eq!(align_of::<InstructionClaimRightLookup>(), 8);
    assert_eq!(size_of::<InstructionClaimPhaseParams>(), 16);
    assert_eq!(size_of::<InstructionClaimOpeningParams>(), 16);
    assert_eq!(size_of::<InstructionClaimReductionParams>(), 16);
    assert_eq!(align_of::<InstructionClaimPhaseParams>(), 4);
    assert_eq!(align_of::<InstructionClaimOpeningParams>(), 4);
    assert_eq!(align_of::<InstructionClaimReductionParams>(), 4);

    let core = InstructionClaimCoreRow::new(1, 2, u128::MAX - 3, 4);
    assert_eq!(core.words(), [1, 2, u64::MAX - 3, u64::MAX, 4]);
    assert_eq!(core.lookup_output(), 1);
    assert_eq!(core.left_lookup_operand(), 2);
    assert_eq!(core.right_lookup_operand(), u128::MAX - 3);
    assert_eq!(core.left_instruction_input(), 4);
    let right_lookup = InstructionClaimRightLookup::new(u128::MAX - 3);
    assert_eq!(right_lookup.words(), [u64::MAX - 3, u64::MAX]);
    assert_eq!(
        InstructionClaimRightLookup::from_words(right_lookup.words()),
        right_lookup
    );
    assert_eq!(right_lookup.value(), u128::MAX - 3);
}

#[test]
fn optimized_rows_split_into_native_planes() {
    use jolt_witness::witnesses::{
        LeftInstructionInput, LeftLookupOperand, LookupOutput, RightInstructionInput,
        RightLookupOperand,
    };

    let rows = [
        InstructionOperandRow {
            lookup_output: LookupOutput(1),
            left_lookup_operand: LeftLookupOperand(2),
            right_lookup_operand: RightLookupOperand(u128::MAX - 3),
            left_instruction_input: LeftInstructionInput(4),
            right_instruction_input: RightInstructionInput(-5),
        },
        InstructionOperandRow {
            lookup_output: LookupOutput(6),
            left_lookup_operand: LeftLookupOperand(7),
            right_lookup_operand: RightLookupOperand(8),
            left_instruction_input: LeftInstructionInput(9),
            right_instruction_input: RightInstructionInput(10),
        },
    ];
    let planes = split_operand_rows(&rows).expect("the row count is valid");

    assert_eq!(planes.lookup_output(), [1, 6]);
    assert_eq!(planes.left_lookup_operand(), [2, 7]);
    assert_eq!(
        planes.right_lookup_operand(),
        [
            InstructionClaimRightLookup::new(u128::MAX - 3),
            InstructionClaimRightLookup::new(8),
        ]
    );
    assert_eq!(planes.left_instruction_input(), [4, 9]);
    assert_eq!(
        planes.right_instruction_input(),
        [
            InstructionClaimRightInput::new(-5),
            InstructionClaimRightInput::new(10),
        ]
    );
}

#[test]
fn operand_planes_reject_mismatched_or_invalid_lengths() {
    assert_eq!(
        InstructionClaimOperandPlanes::new(
            vec![0; 2],
            vec![0; 4],
            vec![InstructionClaimRightLookup::default(); 2],
            vec![0; 2],
            vec![InstructionClaimRightInput::default(); 2],
        ),
        Err(InstructionClaimShapeError::OperandPlaneLength {
            name: "left lookup operand",
            expected: 2,
            got: 4,
        })
    );
    assert_eq!(
        InstructionClaimOperandPlanes::new(
            vec![0; 3],
            vec![0; 3],
            vec![InstructionClaimRightLookup::default(); 3],
            vec![0; 3],
            vec![InstructionClaimRightInput::default(); 3],
        ),
        Err(InstructionClaimShapeError::InvalidRows(3))
    );
}

#[test]
fn signed_right_input_round_trips_twos_complement() {
    for value in [i128::MIN, -1, 0, 1, i128::MAX] {
        let encoded = InstructionClaimRightInput::new(value);
        assert_eq!(
            InstructionClaimRightInput::from_words(encoded.words()),
            encoded
        );
        assert_eq!(encoded.value(), value);
    }
}

#[test]
fn exact_gruen_geometry_covers_even_and_odd_log_t() {
    let even = InstructionClaimGeometry::new(16).expect("the geometry is valid");
    assert_eq!(even.log_t(), 4);
    assert_eq!(
        (0..4)
            .map(|round| even.message(round).expect("the round is valid").weights())
            .collect::<Vec<_>>(),
        vec![
            InstructionClaimWeightGeometry {
                e_in_length: 2,
                e_out_length: 4,
            },
            InstructionClaimWeightGeometry {
                e_in_length: 1,
                e_out_length: 4,
            },
            InstructionClaimWeightGeometry {
                e_in_length: 1,
                e_out_length: 2,
            },
            InstructionClaimWeightGeometry {
                e_in_length: 1,
                e_out_length: 1,
            },
        ]
    );
    assert_eq!(
        even.opening(),
        InstructionClaimWeightGeometry {
            e_in_length: 4,
            e_out_length: 4,
        }
    );

    let odd = InstructionClaimGeometry::new(8).expect("the geometry is valid");
    assert_eq!(
        (0..3)
            .map(|round| odd.message(round).expect("the round is valid").weights())
            .collect::<Vec<_>>(),
        vec![
            InstructionClaimWeightGeometry {
                e_in_length: 2,
                e_out_length: 2,
            },
            InstructionClaimWeightGeometry {
                e_in_length: 1,
                e_out_length: 2,
            },
            InstructionClaimWeightGeometry {
                e_in_length: 1,
                e_out_length: 1,
            },
        ]
    );
    assert_eq!(
        odd.opening(),
        InstructionClaimWeightGeometry {
            e_in_length: 4,
            e_out_length: 2,
        }
    );
    assert!(matches!(
        InstructionClaimOpeningParams::new(odd, 2, 4, InstructionClaimOpeningMode::AllColumns,),
        Err(InstructionClaimShapeError::WeightLayout { .. })
    ));
    assert!(matches!(
        InstructionClaimStorageLayout::new(8, 2, 4),
        Err(InstructionClaimShapeError::WeightCapacity { .. })
    ));
}

#[test]
fn oracle_materializes_hinted_message_endpoints() {
    let core: Vec<_> = [1, 2, 3, 4]
        .into_iter()
        .map(|lookup| InstructionClaimCoreRow::new(lookup, 0, 0, 0))
        .collect();
    let right = vec![InstructionClaimRightInput::default(); 4];
    let planes = operand_planes(&core, &right);
    let result = oracle::materialize_message(
        &planes,
        AkitaField::from_u64(9),
        &[AkitaField::from_u64(3)],
        &[AkitaField::from_u64(5), AkitaField::from_u64(7)],
    )
    .expect("the shape is valid");

    assert_eq!(
        result.state,
        [1, 2, 3, 4].map(AkitaField::from_u64).to_vec()
    );
    assert_eq!(result.partials, [15, 63, 45, 105].map(AkitaField::from_u64));
    assert_eq!(
        result.q_endpoints,
        [AkitaField::from_u64(78), AkitaField::from_u64(150)]
    );
}

#[test]
fn oracle_fuses_bind_with_the_next_message() {
    let state = [1, 2, 3, 4].map(AkitaField::from_u64);
    let geometry = InstructionClaimGeometry::new(4).expect("the geometry is valid");
    let result = oracle::bind_and_message(
        &state,
        geometry,
        1,
        AkitaField::from_u64(2),
        &[AkitaField::one()],
        &[AkitaField::from_u64(3)],
    )
    .expect("the shape is valid");

    assert_eq!(
        result.state,
        [AkitaField::from_u64(3), AkitaField::from_u64(5)]
    );
    assert_eq!(
        result.partials,
        [AkitaField::from_u64(9), AkitaField::from_u64(21)]
    );
    assert_eq!(
        result.q_endpoints,
        [AkitaField::from_u64(9), AkitaField::from_u64(21)]
    );
}

#[test]
fn host_endpoint_scaling_and_final_bind_match_the_protocol_formula() {
    let scaled = scale_q_endpoints(
        [AkitaField::from_u64(3), AkitaField::from_u64(5)],
        [AkitaField::from_u64(7), AkitaField::from_u64(11)],
    );
    assert_eq!(scaled, [AkitaField::from_u64(21), AkitaField::from_u64(75)]);
    assert_eq!(
        finish_bind(
            [AkitaField::from_u64(3), AkitaField::from_u64(5)],
            AkitaField::from_u64(7),
        ),
        AkitaField::from_u64(17)
    );

    let previous_claim = AkitaField::from_u64(65);
    let actual = round_polynomial_from_q_endpoints(
        previous_claim,
        [AkitaField::from_u64(3), AkitaField::from_u64(5)],
        [AkitaField::from_u64(7), AkitaField::from_u64(11)],
    );
    let expected = UnivariatePoly::from_evals(&[
        AkitaField::from_u64(21),
        AkitaField::from_u64(44),
        AkitaField::from_u64(75),
    ]);
    assert_eq!(actual.coefficients(), expected.coefficients());
}

#[test]
fn oracle_openings_cover_core_and_signed_planes() {
    let core = [
        InstructionClaimCoreRow::new(1, 2, 3, 4),
        InstructionClaimCoreRow::new(6, 7, 8, 9),
        InstructionClaimCoreRow::new(11, 12, 13, 14),
        InstructionClaimCoreRow::new(16, 17, 18, 19),
    ];
    let right = [-5, 10, -15, 20].map(InstructionClaimRightInput::new);
    let planes = operand_planes(&core, &right);
    let e_in = [AkitaField::from_u64(2), AkitaField::from_u64(3)];
    let e_out = [AkitaField::from_u64(5), AkitaField::from_u64(7)];

    assert_eq!(
        oracle::core_openings(&planes, &e_in, &e_out).expect("the opening shape is valid"),
        [590, 650, 710, 770].map(AkitaField::from_u64)
    );
    assert_eq!(
        oracle::core_opening_partials(&planes, &e_in, &e_out).expect("the opening shape is valid"),
        [100, 490, 125, 525, 150, 560, 175, 595].map(AkitaField::from_u64)
    );
    assert_eq!(
        oracle::all_openings(&planes, &e_in, &e_out)
            .expect("the opening shape is valid")
            .into_array(),
        [590, 650, 710, 770, 310].map(AkitaField::from_u64)
    );
    assert_eq!(
        oracle::all_opening_partials(&planes, &e_in, &e_out).expect("the opening shape is valid"),
        [100, 490, 125, 525, 150, 560, 175, 595, 100, 210,].map(AkitaField::from_u64)
    );
    assert_eq!(
        oracle::aliased_openings(&planes, &e_in, &e_out)
            .expect("the aliased opening shape is valid"),
        [650, 710].map(AkitaField::from_u64)
    );
    assert_eq!(
        oracle::aliased_opening_partials(&planes, &e_in, &e_out)
            .expect("the aliased opening shape is valid"),
        [125, 525, 150, 560].map(AkitaField::from_u64)
    );
}

#[test]
fn nonzero_gamma_recovers_the_fifth_opening() {
    let gamma = AkitaField::from_u64(2);
    let core = [3, 5, 7, 11].map(AkitaField::from_u64);
    let right = AkitaField::from_u64(13);
    let powers = nontrivial_gamma_powers(gamma);
    let combined = core[0]
        + powers[0] * core[1]
        + powers[1] * core[2]
        + powers[2] * core[3]
        + powers[3] * right;

    assert_eq!(
        InstructionClaimOpeningMode::for_gamma(gamma),
        InstructionClaimOpeningMode::CoreAndRecover
    );
    assert_eq!(recover_right_input(gamma, combined, core), Ok(right));
    assert_eq!(
        finalize_openings(
            InstructionClaimOpeningMode::CoreAndRecover,
            gamma,
            combined,
            core,
            None,
        )
        .expect("nonzero gamma is invertible")
        .into_array(),
        [core[0], core[1], core[2], core[3], right]
    );
}

#[test]
fn opening_geometry_matches_low_to_high_resident_binding() {
    let gamma = AkitaField::from_u64(2);
    let first_challenge = AkitaField::from_u64(3);
    let final_challenge = AkitaField::from_u64(5);
    let core = [
        InstructionClaimCoreRow::new(1, 2, 3, 4),
        InstructionClaimCoreRow::new(6, 7, 8, 9),
        InstructionClaimCoreRow::new(11, 12, 13, 14),
        InstructionClaimCoreRow::new(16, 17, 18, 19),
    ];
    let right = [-5, 10, -15, 20].map(InstructionClaimRightInput::new);
    let planes = operand_planes(&core, &right);
    let combined: [AkitaField; 4] =
        std::array::from_fn(|index| core[index].combined(right[index], gamma));
    let bound_once = [
        finish_bind([combined[0], combined[1]], first_challenge),
        finish_bind([combined[2], combined[3]], first_challenge),
    ];
    let final_combined = finish_bind(bound_once, final_challenge);

    let e_in = [AkitaField::one() - first_challenge, first_challenge];
    let e_out = [AkitaField::one() - final_challenge, final_challenge];
    let openings =
        oracle::all_openings(&planes, &e_in, &e_out).expect("the opening shape is valid");
    assert_eq!(openings.combined(gamma), final_combined);

    let core_openings =
        oracle::core_openings(&planes, &e_in, &e_out).expect("the opening shape is valid");
    assert_eq!(
        finalize_openings(
            InstructionClaimOpeningMode::CoreAndRecover,
            gamma,
            final_combined,
            core_openings,
            None,
        )
        .expect("nonzero gamma is invertible"),
        openings
    );
}

#[test]
fn gamma_zero_selects_and_requires_the_all_column_path() {
    let zero = AkitaField::zero();
    let core = [3, 5, 7, 11].map(AkitaField::from_u64);
    let right = AkitaField::from_u64(13);
    assert_eq!(
        InstructionClaimOpeningMode::for_gamma(zero),
        InstructionClaimOpeningMode::AllColumns
    );
    assert_eq!(
        recover_right_input(zero, core[0], core),
        Err(InstructionClaimOpeningError::ZeroGammaRecovery)
    );
    assert_eq!(
        finalize_openings(
            InstructionClaimOpeningMode::AllColumns,
            zero,
            core[0],
            core,
            None,
        ),
        Err(InstructionClaimOpeningError::MissingRightInputOpening)
    );
    assert_eq!(
        finalize_openings(
            InstructionClaimOpeningMode::AllColumns,
            zero,
            core[0],
            core,
            Some(right),
        )
        .expect("the fifth column was scanned")
        .right_instruction_input,
        right
    );
}

#[test]
fn aliased_openings_cover_zero_and_nonzero_gamma() {
    let aliases = InstructionClaimAliasedOpenings {
        lookup_output: AkitaField::from_u64(3),
        left_instruction_input: AkitaField::from_u64(11),
        right_instruction_input: AkitaField::from_u64(13),
    };
    let lookup_operands = [AkitaField::from_u64(5), AkitaField::from_u64(7)];

    for gamma in [AkitaField::zero(), AkitaField::from_u64(2)] {
        let expected = InstructionClaimOpenings {
            lookup_output: aliases.lookup_output,
            left_lookup_operand: lookup_operands[0],
            right_lookup_operand: lookup_operands[1],
            left_instruction_input: aliases.left_instruction_input,
            right_instruction_input: aliases.right_instruction_input,
        };
        let combined = expected.combined(gamma);
        assert_eq!(
            finalize_aliased_openings(gamma, combined, lookup_operands, aliases),
            Ok(expected)
        );
        assert_eq!(
            verifier_output_term(AkitaField::from_u64(17), expected, gamma),
            AkitaField::from_u64(17) * combined
        );
    }
}

#[test]
fn finalization_rejects_openings_from_the_wrong_point() {
    let gamma = AkitaField::from_u64(2);
    let core = [3, 5, 7, 11].map(AkitaField::from_u64);
    assert_eq!(
        finalize_openings(
            InstructionClaimOpeningMode::AllColumns,
            gamma,
            AkitaField::from_u64(123),
            core,
            Some(AkitaField::from_u64(13)),
        ),
        Err(InstructionClaimOpeningError::CombinedClaimMismatch)
    );
}

#[test]
fn reduction_plan_and_oracle_cover_the_recursive_entry_point() {
    let plan =
        InstructionClaimReductionPlan::new(8192, 2).expect("the reduction geometry is valid");
    assert_eq!(
        plan.passes(),
        [
            InstructionClaimReductionPass {
                input_count: 8192,
                output_count: 256,
                dispatched_threads: 8192,
            },
            InstructionClaimReductionPass {
                input_count: 256,
                output_count: 8,
                dispatched_threads: 256,
            },
            InstructionClaimReductionPass {
                input_count: 8,
                output_count: 1,
                dispatched_threads: 32,
            },
        ]
    );

    let first_column: Vec<_> = (1..=33).map(AkitaField::from_u64).collect();
    let second_column: Vec<_> = (0..33)
        .map(|index| AkitaField::from_u64(100 + index))
        .collect();
    let input = [first_column, second_column].concat();
    let reduced = oracle::reduce_once(&input, 33, 2).expect("the shape is valid");
    assert_eq!(
        reduced,
        [
            AkitaField::from_u64((1..=32).sum()),
            AkitaField::from_u64(33),
            AkitaField::from_u64((100..132).sum()),
            AkitaField::from_u64(132),
        ]
    );
}

#[test]
fn native_conversion_edges_match_field_constructors() {
    let core = InstructionClaimCoreRow::new(u64::MAX, u64::MAX - 1, u128::MAX, 1_u64 << 63);
    let right = InstructionClaimRightInput::new(i128::MIN);
    let gamma = AkitaField::from_u64(17);
    let powers = nontrivial_gamma_powers(gamma);
    let expected = AkitaField::from_u64(u64::MAX)
        + powers[0] * AkitaField::from_u64(u64::MAX - 1)
        + powers[1] * AkitaField::from_u128(u128::MAX)
        + powers[2] * AkitaField::from_u64(1_u64 << 63)
        + powers[3] * AkitaField::from_i128(i128::MIN);
    assert_eq!(core.combined(right, gamma), expected);

    let modulus = u128::MAX - u128::from(INSTRUCTION_CLAIM_AKITA_OFFSET) + 1;
    for (value, expected) in [
        (modulus - 1, -AkitaField::one()),
        (modulus, AkitaField::zero()),
        (modulus + 1, AkitaField::one()),
    ] {
        let row = InstructionClaimCoreRow::new(0, 0, value, 0);
        assert_eq!(
            row.combined(InstructionClaimRightInput::default(), AkitaField::one()),
            expected
        );
    }
}

#[test]
fn config_and_column_shapes_reject_invalid_dispatches() {
    assert_eq!(
        InstructionClaimOpeningMode::aliased_pipeline(),
        "solinas_instruction_claim_open_lookup_operands"
    );
    assert_eq!(
        InstructionClaimKernelConfig::default().validate(),
        Ok(InstructionClaimKernelConfig::default())
    );
    let config = InstructionClaimKernelConfig::default();
    assert_eq!(config.materialize_threadgroup_bytes(), Ok(128));
    assert_eq!(config.transition_threadgroup_bytes(), Ok(64));
    assert_eq!(
        config.opening_threadgroup_bytes(INSTRUCTION_CLAIM_ALIASED_OPENINGS),
        Ok(128)
    );
    assert_eq!(
        config.opening_threadgroup_bytes(INSTRUCTION_CLAIM_CORE_OPENINGS),
        Ok(256)
    );
    assert_eq!(
        config.opening_threadgroup_bytes(INSTRUCTION_CLAIM_ALL_OPENINGS),
        Ok(320)
    );
    let invalid = InstructionClaimKernelConfig {
        transition_threads_per_threadgroup: 48,
        ..InstructionClaimKernelConfig::default()
    };
    assert_eq!(
        invalid.validate(),
        Err(InstructionClaimShapeError::InvalidThreadgroupWidth {
            phase: "transition",
            width: 48,
        })
    );

    for columns in [
        INSTRUCTION_CLAIM_MESSAGE_COLUMNS,
        INSTRUCTION_CLAIM_CORE_OPENINGS,
        INSTRUCTION_CLAIM_ALL_OPENINGS,
    ] {
        assert!(InstructionClaimReductionParams::new(33, columns).is_ok());
    }
    assert_eq!(
        InstructionClaimReductionParams::new(33, 3),
        Err(InstructionClaimShapeError::InvalidReductionColumns(3))
    );
    let geometry = InstructionClaimGeometry::new(8).expect("the geometry is valid");
    assert!(InstructionClaimOpeningParams::new(
        geometry,
        4,
        2,
        InstructionClaimOpeningMode::AllColumns,
    )
    .is_ok());
    assert!(matches!(
        InstructionClaimOpeningParams::new(
            geometry,
            2,
            4,
            InstructionClaimOpeningMode::CoreAndRecover,
        ),
        Err(InstructionClaimShapeError::WeightLayout {
            phase: "openings",
            ..
        })
    ));
}

#[test]
fn target_scale_storage_layout_is_pinned() {
    let rows = 1usize << 26;
    let e_in_capacity = 1usize << 13;
    let e_out_capacity = 1usize << 13;
    let layout = InstructionClaimStorageLayout::new(rows, e_in_capacity, e_out_capacity)
        .expect("the target-scale layout is valid");
    let expected_workspace_fields = rows
        + rows / 2
        + e_in_capacity
        + e_out_capacity
        + 2 * INSTRUCTION_CLAIM_ALL_OPENINGS * e_out_capacity
        + INSTRUCTION_CLAIM_NONTRIVIAL_GAMMA_POWERS;
    let expected_workspace_bytes = expected_workspace_fields * 16;

    assert_eq!(layout.rows(), rows);
    assert_eq!(layout.lookup_output_bytes(), rows * 8);
    assert_eq!(layout.left_lookup_operand_bytes(), rows * 8);
    assert_eq!(layout.right_lookup_operand_bytes(), rows * 16);
    assert_eq!(layout.left_instruction_input_bytes(), rows * 8);
    assert_eq!(layout.right_input_bytes(), rows * 16);
    assert_eq!(layout.maximum_operand_plane_bytes(), 1usize << 30);
    assert_eq!(layout.maximum_buffer_bytes(), 1usize << 30);
    assert_eq!(layout.validate_max_buffer_length(1usize << 31), Ok(layout));
    assert_eq!(
        layout.validate_max_buffer_length((1usize << 30) - 1),
        Err(InstructionClaimShapeError::BufferLengthLimit {
            required: 1usize << 30,
            maximum: (1usize << 30) - 1,
        })
    );
    assert_eq!(
        layout.gamma_power_fields(),
        INSTRUCTION_CLAIM_NONTRIVIAL_GAMMA_POWERS
    );
    assert_eq!(layout.state_a_fields(), rows);
    assert_eq!(layout.state_b_fields(), rows / 2);
    assert_eq!(layout.e_in_fields(), e_in_capacity);
    assert_eq!(layout.e_out_fields(), e_out_capacity);
    assert_eq!(
        layout.partial_fields(),
        INSTRUCTION_CLAIM_ALL_OPENINGS * e_out_capacity
    );
    assert_eq!(layout.workspace_bytes(), expected_workspace_bytes);
    assert_eq!(
        layout.resident_bytes(),
        expected_workspace_bytes + rows * 56
    );
}

#[test]
fn resident_sequence_matches_every_oracle_intermediate() {
    assert_resident_sequence(AkitaField::from_u64(17), 1 << 9);
}

#[test]
fn resident_sequence_gamma_zero_scans_the_signed_column() {
    assert_resident_sequence(AkitaField::zero(), 1 << 7);
}

#[test]
fn stage1_sequence_matches_standalone_sequence() {
    let Ok(context) = super::super::SolinasMetal::for_akita() else {
        return;
    };
    let rows: usize = 1 << 8;
    let gamma = AkitaField::from_u64(17);
    let core = (0..rows)
        .map(|index| {
            InstructionClaimCoreRow::new(
                13 * index as u64 + 1,
                17 * index as u64 + 2,
                (u128::from(index as u64) << 73) | u128::from(19 * index as u64 + 3),
                23 * index as u64 + 4,
            )
        })
        .collect::<Vec<_>>();
    let right = (0..rows)
        .map(|index| {
            InstructionClaimRightInput::new(if index.is_multiple_of(2) {
                -(29 * index as i128 + 5)
            } else {
                31 * index as i128 + 6
            })
        })
        .collect::<Vec<_>>();
    let planes = operand_planes(&core, &right);
    let stage1 = core
        .iter()
        .zip(&right)
        .map(|(core, right)| {
            let mut words = [0u64; 20];
            let right_input = right.value().unsigned_abs();
            let right_lookup = core.right_lookup_operand();
            words[0] = core.left_instruction_input();
            words[1] = right_input as u64;
            words[2] = (right_input >> 64) as u64;
            words[13] = core.left_lookup_operand();
            words[14] = right_lookup as u64;
            words[15] = (right_lookup >> 64) as u64;
            words[18] = core.lookup_output();
            words[19] = u64::from(right.value() >= 0) << 17;
            super::super::SpartanOuterUniskipRow::from_words(words)
        })
        .collect::<Vec<_>>();
    let stage1 = context
        .prepare_spartan_outer_uniskip_rows(&stage1)
        .expect("Stage-1 rows should prepare")
        .share_product_remainder_rows()
        .expect("Stage-1 rows should expose an instruction view");
    let direct = context
        .prepare_instruction_claim_sequence_with_stage1_rows(
            stage1,
            gamma,
            InstructionClaimKernelConfig::default(),
        )
        .expect("Stage-1 instruction sequence should prepare");
    let mut standalone = context
        .prepare_instruction_claim_sequence(&planes, gamma, InstructionClaimKernelConfig::default())
        .expect("standalone instruction sequence should prepare");
    let log_t = rows.trailing_zeros() as usize;
    let point = (0..log_t)
        .map(|index| AkitaField::from_u64(101 + 2 * index as u64))
        .collect::<Vec<_>>();
    let challenges = (0..log_t)
        .map(|index| AkitaField::from_u64(401 + 4 * index as u64))
        .collect::<Vec<_>>();
    let mut gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
    let direct_pending = direct
        .submit_initial_message(gruen.e_in_current(), gruen.e_out_current())
        .unwrap();
    let (mut direct, direct_message, timing) = direct_pending.join().unwrap();
    assert!(timing.wall >= timing.gpu_active);
    assert_eq!(
        direct_message,
        standalone
            .message(gruen.e_in_current(), gruen.e_out_current())
            .unwrap()
    );
    for round in 1..log_t {
        let challenge = challenges[round - 1];
        gruen.bind(challenge);
        let direct_message = direct
            .bind_and_message(challenge, gruen.e_in_current(), gruen.e_out_current())
            .unwrap();
        let standalone_message = standalone
            .bind_and_message(challenge, gruen.e_in_current(), gruen.e_out_current())
            .unwrap();
        assert_eq!(direct_message, standalone_message, "round {round}");
    }
    let direct_final = direct.finish(challenges[log_t - 1]).unwrap();
    let standalone_final = standalone.finish(challenges[log_t - 1]).unwrap();
    assert_eq!(direct_final, standalone_final);
    let reversed = challenges.iter().rev().copied().collect::<Vec<_>>();
    let (r_hi, r_lo) = reversed.split_at(log_t / 2);
    let e_out = EqPolynomial::evals(r_hi, None);
    let e_in = EqPolynomial::evals(r_lo, None);
    let layout = direct.storage_layout();
    let expected_retired_bytes =
        (layout.state_a_fields() + layout.state_b_fields()) * size_of::<[u64; 2]>();
    assert_eq!(
        direct.retire_transition_state().unwrap(),
        expected_retired_bytes
    );
    assert!(direct.read_current_state().is_err());
    let direct_openings = direct.aliased_openings(&e_in, &e_out).unwrap();
    let standalone_openings = standalone.aliased_openings(&e_in, &e_out).unwrap();
    assert_eq!(direct_openings, standalone_openings);
}

#[test]
fn stale_cpu_tail_is_rejected_after_reset() {
    let Ok(context) = super::super::SolinasMetal::for_akita() else {
        return;
    };
    let rows = 1 << 4;
    let core = (0..rows)
        .map(|index| InstructionClaimCoreRow::new(index as u64, 2, 3, 4))
        .collect::<Vec<_>>();
    let right = (0..rows)
        .map(|index| InstructionClaimRightInput::new(index as i128 - 8))
        .collect::<Vec<_>>();
    let planes = operand_planes(&core, &right);
    let point = (0..4)
        .map(|index| AkitaField::from_u64(101 + 2 * index))
        .collect::<Vec<_>>();
    let challenges = (0..4)
        .map(|index| AkitaField::from_u64(401 + 4 * index))
        .collect::<Vec<_>>();
    let mut sequence = context
        .prepare_instruction_claim_sequence(
            &planes,
            AkitaField::from_u64(17),
            InstructionClaimKernelConfig::default(),
        )
        .expect("the resident sequence should prepare");

    let mut gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
    let _ = sequence
        .message(gruen.e_in_current(), gruen.e_out_current())
        .unwrap();
    let mut stale = sequence.handoff_to_cpu().unwrap();
    for round in 1..4 {
        gruen.bind(challenges[round - 1]);
        let _ = stale
            .bind_and_message(
                challenges[round - 1],
                gruen.e_in_current(),
                gruen.e_out_current(),
            )
            .unwrap();
    }
    assert_eq!(stale.current_elements(), 2);

    sequence.reset();
    let gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
    let _ = sequence
        .message(gruen.e_in_current(), gruen.e_out_current())
        .unwrap();
    let _current = sequence.handoff_to_cpu().unwrap();
    let error = sequence.finish_cpu_tail(stale, challenges[3]).unwrap_err();
    assert!(error.to_string().contains("generation"));
}

fn assert_resident_sequence(gamma: AkitaField, rows: usize) {
    let Ok(context) = super::super::SolinasMetal::for_akita() else {
        return;
    };
    let core = (0..rows)
        .map(|index| {
            InstructionClaimCoreRow::new(
                (index as u64).wrapping_mul(0x9e37_79b9),
                (index as u64).rotate_left(17) ^ u64::MAX,
                ((index as u128) << 79) | ((index as u128).wrapping_mul(0x1_0000_01b3)),
                (index as u64).wrapping_mul(0xd6e8_feb8_6659_fd93),
            )
        })
        .collect::<Vec<_>>();
    let right = (0..rows)
        .map(|index| {
            let magnitude = ((index as i128) << 61) | (index as i128 + 1);
            InstructionClaimRightInput::new(if index.is_multiple_of(3) {
                -magnitude
            } else {
                magnitude
            })
        })
        .collect::<Vec<_>>();
    let planes = operand_planes(&core, &right);
    let log_t = rows.trailing_zeros() as usize;
    let point = (0..log_t)
        .map(|index| AkitaField::from_u64(101 + 2 * index as u64))
        .collect::<Vec<_>>();
    let challenges = (0..log_t)
        .map(|index| AkitaField::from_u64(401 + 4 * index as u64))
        .collect::<Vec<_>>();
    let mut gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
    let mut sequence = context
        .prepare_instruction_claim_sequence(&planes, gamma, InstructionClaimKernelConfig::default())
        .expect("the resident sequence should prepare");
    let allocations = sequence.allocation_identities();

    let expected =
        oracle::materialize_message(&planes, gamma, gruen.e_in_current(), gruen.e_out_current())
            .expect("the first oracle message should evaluate");
    let actual = sequence
        .message(gruen.e_in_current(), gruen.e_out_current())
        .expect("the first Metal message should execute");
    assert_eq!(actual, expected.q_endpoints);
    assert_eq!(sequence.read_current_state().unwrap(), expected.state);
    let mut state = expected.state;

    for round in 1..log_t {
        let challenge = challenges[round - 1];
        gruen.bind(challenge);
        let expected = oracle::bind_and_message(
            &state,
            InstructionClaimGeometry::new(rows).unwrap(),
            round,
            challenge,
            gruen.e_in_current(),
            gruen.e_out_current(),
        )
        .expect("the transition oracle should evaluate");
        let actual = sequence
            .bind_and_message(challenge, gruen.e_in_current(), gruen.e_out_current())
            .expect("the transition Metal message should execute");
        assert_eq!(actual, expected.q_endpoints, "round {round}");
        assert_eq!(
            sequence.read_current_state().unwrap(),
            expected.state,
            "round {round} resident state"
        );
        state = expected.state;
    }

    let final_claim = sequence
        .finish(challenges[log_t - 1])
        .expect("the final pair should bind");
    assert_eq!(
        final_claim,
        finish_bind([state[0], state[1]], challenges[log_t - 1])
    );

    let reversed = challenges.iter().rev().copied().collect::<Vec<_>>();
    let split = reversed.len() / 2;
    let (r_hi, r_lo) = reversed.split_at(split);
    let e_out = EqPolynomial::<AkitaField>::evals(r_hi, None);
    let e_in = EqPolynomial::<AkitaField>::evals(r_lo, None);
    let expected_openings =
        oracle::all_openings(&planes, &e_in, &e_out).expect("the opening oracle should evaluate");
    let actual_openings = sequence
        .openings(&e_in, &e_out)
        .expect("the opening Metal scan should execute");
    assert_eq!(actual_openings, expected_openings);
    assert_eq!(actual_openings.combined(gamma), final_claim);

    let expected_aliases = oracle::aliased_openings(&planes, &e_in, &e_out)
        .expect("the aliased opening oracle should evaluate");
    let actual_aliases = sequence
        .aliased_openings(&e_in, &e_out)
        .expect("the aliased opening Metal scan should execute");
    assert_eq!(actual_aliases, expected_aliases);
    assert_eq!(sequence.allocation_identities(), allocations);
    assert_eq!(sequence.round_device_buffer_allocations(), 0);

    sequence.reset();
    assert_eq!(sequence.current_elements(), rows);
    assert_eq!(sequence.allocation_identities(), allocations);

    let mut gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
    let expected =
        oracle::materialize_message(&planes, gamma, gruen.e_in_current(), gruen.e_out_current())
            .unwrap();
    assert_eq!(
        sequence
            .message(gruen.e_in_current(), gruen.e_out_current())
            .unwrap(),
        expected.q_endpoints
    );
    let mut state = expected.state;
    let gpu_rounds = 3.min(log_t);
    for round in 1..gpu_rounds {
        gruen.bind(challenges[round - 1]);
        let expected = oracle::bind_and_message(
            &state,
            InstructionClaimGeometry::new(rows).unwrap(),
            round,
            challenges[round - 1],
            gruen.e_in_current(),
            gruen.e_out_current(),
        )
        .unwrap();
        assert_eq!(
            sequence
                .bind_and_message(
                    challenges[round - 1],
                    gruen.e_in_current(),
                    gruen.e_out_current(),
                )
                .unwrap(),
            expected.q_endpoints
        );
        state = expected.state;
    }
    let mut tail = sequence.handoff_to_cpu().unwrap();
    assert_eq!(tail.current_elements(), state.len());
    for round in gpu_rounds..log_t {
        gruen.bind(challenges[round - 1]);
        let expected = oracle::bind_and_message(
            &state,
            InstructionClaimGeometry::new(rows).unwrap(),
            round,
            challenges[round - 1],
            gruen.e_in_current(),
            gruen.e_out_current(),
        )
        .unwrap();
        assert_eq!(
            tail.bind_and_message(
                challenges[round - 1],
                gruen.e_in_current(),
                gruen.e_out_current(),
            )
            .unwrap(),
            expected.q_endpoints,
            "CPU tail round {round}"
        );
        state = expected.state;
    }
    assert_eq!(tail.current_elements(), 2);
    assert_eq!(tail.round_device_buffer_allocations(), 0);
    assert_eq!(
        sequence
            .finish_cpu_tail(tail, challenges[log_t - 1])
            .unwrap(),
        finish_bind([state[0], state[1]], challenges[log_t - 1])
    );
    assert_eq!(sequence.allocation_identities(), allocations);
}
