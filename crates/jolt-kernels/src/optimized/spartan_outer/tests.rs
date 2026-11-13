use crate::optimized::parity::ExceptionalEq;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    geometry::spartan::FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS, FieldInlinePolynomialId,
};
use jolt_claims::protocols::jolt::geometry::spartan::SPARTAN_OUTER_R1CS_INPUTS;
use jolt_claims::protocols::jolt::JoltPolynomialId;
use jolt_claims::NoChallenges;
use jolt_field::signed::S128;
use jolt_field::{Fr, Ring};
use jolt_riscv::CIRCUIT_FLAGS;
use jolt_verifier::stages::stage1::outer_remainder::outer_remainder_input_values_from_uniskip_output;
use jolt_witness::testing::with_sample_backend;
use jolt_witness::witnesses::ToField;
#[cfg(feature = "field-inline")]
use jolt_witness::FixedFieldInline;
use jolt_witness::{BundleSource, JoltWitnessOracle};
use jolt_witness::{FixedBackend, PolynomialEncoding, Shape, TraceBackend};

use super::*;
use crate::reference::spartan_outer::{ReferenceOuterRemainder, SpartanOuterKernel};
use crate::ReferenceBackend;

fn variable_field_value<F: JoltField>(row: &SpartanOuterRow, index: usize) -> F {
    match index {
        0 => row.left_instruction_input.to_field(),
        1 => row.right_instruction_input.to_field(),
        2 => row.product.to_field(),
        3 => row.should_branch.to_field(),
        4 => row.pc.to_field(),
        5 => row.unexpanded_pc.to_field(),
        6 => row.imm.to_field(),
        7 => row.ram_address.to_field(),
        8 => row.rs1_value.to_field(),
        9 => row.rs2_value.to_field(),
        10 => row.rd_write_value.to_field(),
        11 => row.ram_read_value.to_field(),
        12 => row.ram_write_value.to_field(),
        13 => row.left_lookup_operand.to_field(),
        14 => row.right_lookup_operand.to_field(),
        15 => row.next_unexpanded_pc.to_field(),
        16 => row.next_pc.to_field(),
        17 => row.next_is_virtual.to_field(),
        18 => row.next_is_first_in_sequence.to_field(),
        19 => row.lookup_output.to_field(),
        20 => row.should_jump.to_field(),
        21 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::AddOperands])),
        22 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::SubtractOperands])),
        23 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::MultiplyOperands])),
        24 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::Load])),
        25 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::Store])),
        26 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::Jump])),
        27 => F::from_u64(u64::from(
            row.circuit_flags[CircuitFlags::WriteLookupOutputToRD],
        )),
        28 => F::from_u64(u64::from(
            row.circuit_flags[CircuitFlags::VirtualInstruction],
        )),
        29 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::Assert])),
        30 => F::from_u64(u64::from(
            row.circuit_flags[CircuitFlags::DoNotUpdateUnexpandedPC],
        )),
        31 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::Advice])),
        32 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::IsCompressed])),
        33 => F::from_u64(u64::from(
            row.circuit_flags[CircuitFlags::IsFirstInSequence],
        )),
        34 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::IsLastInSequence])),
        #[cfg(feature = "field-inline")]
        35 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::FieldAdd])),
        #[cfg(feature = "field-inline")]
        36 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::FieldSub])),
        #[cfg(feature = "field-inline")]
        37 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::FieldMul])),
        #[cfg(feature = "field-inline")]
        38 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::FieldInv])),
        #[cfg(feature = "field-inline")]
        39 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::FieldAssertEq])),
        #[cfg(feature = "field-inline")]
        40 => F::from_u64(u64::from(
            row.circuit_flags[CircuitFlags::FieldLoadAccumulateFromRegister],
        )),
        #[cfg(feature = "field-inline")]
        41 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::FieldAssertZero])),
        #[cfg(feature = "field-inline")]
        42 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::FieldLoadImm])),
        #[cfg(feature = "field-inline")]
        43 => F::from_u64(u64::from(
            row.circuit_flags[CircuitFlags::FieldLoadAccumulateFromMemory],
        )),
        #[cfg(feature = "field-inline")]
        44 => F::from_u64(u64::from(row.circuit_flags[CircuitFlags::FieldAdviceLimb])),
        _ => unreachable!("canonical R1CS inputs"),
    }
}

/// Structured pseudo-random rows: full-range `u64`s, mixed-sign `i128`s,
/// two-limb `u128`/`S128` values (both wide B-row paths), diverse flags.
/// No satisfying-witness structure — parity must hold pointwise on any
/// witness.
fn synthetic_rows(log_t: usize, seed: u64) -> Vec<SpartanOuterRow> {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..1usize << log_t)
        .map(|_| {
            let mut bit = {
                let value = next();
                let mut position = 0;
                move || {
                    position += 1;
                    (value >> position) & 1 == 1
                }
            };
            let wide = |low: u64, high: u64| (u128::from(high) << 64) | u128::from(low);
            let signed = |value: u64| match value & 3 {
                0 => i128::MIN,
                1 => i128::MAX,
                2 => -(i128::from(value) << 63),
                _ => i128::from(value) << 63,
            };
            SpartanOuterRow {
                left_instruction_input: LeftInstructionInput(next()),
                right_instruction_input: RightInstructionInput(signed(next())),
                product: Product(S128::new([next() | 1, next()], next() & 1 == 1)),
                should_branch: ShouldBranch(bit()),
                pc: Pc(next() >> 20),
                unexpanded_pc: UnexpandedPc(next()),
                imm: Imm(signed(next())),
                ram_address: RamAddress(next()),
                rs1_value: Rs1Value(next()),
                rs2_value: Rs2Value(next()),
                rd_write_value: RdWriteValue(next()),
                ram_read_value: RamReadValue(next()),
                ram_write_value: RamWriteValue(next()),
                left_lookup_operand: LeftLookupOperand(next()),
                right_lookup_operand: RightLookupOperand(wide(next(), next())),
                next_unexpanded_pc: NextUnexpandedPc(next()),
                next_pc: NextPc(next() >> 20),
                next_is_virtual: NextIsVirtual(bit()),
                next_is_first_in_sequence: NextIsFirstInSequence(bit()),
                lookup_output: LookupOutput(next()),
                should_jump: ShouldJump(bit()),
                circuit_flags: CIRCUIT_FLAGS.into_iter().fold(
                    CircuitFlagSet::default(),
                    |flags, flag| if bit() { flags.set(flag) } else { flags },
                ),
            }
        })
        .collect()
}

#[cfg(feature = "field-inline")]
fn synthetic_field_rows(log_t: usize, seed: u64) -> Vec<(usize, FieldInlineSpartanRow<Fr>)> {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let base = Fr::from_u64(state);
        base * base + base
    };
    (0..1usize << log_t)
        .filter(|cycle| cycle % 3 == 1 || log_t == 1)
        .map(|cycle| {
            (
                cycle,
                FieldInlineSpartanRow {
                    rs1_value: next(),
                    rs2_value: next(),
                    rd_value: next(),
                    product: next(),
                    inv_product: next(),
                },
            )
        })
        .collect()
}

/// The dense image of the sparse field-inline rows for one field-inline column
/// index (the `FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS` position) — what the fixed
/// backend's field-inline view serves the reference kernel.
#[cfg(feature = "field-inline")]
fn field_column_table(
    field_rows: &[(usize, FieldInlineSpartanRow<Fr>)],
    cycles: usize,
    index: usize,
) -> Vec<Fr> {
    let mut table = vec![Fr::from_u64(0); cycles];
    for (cycle, row) in field_rows {
        table[*cycle] = row.columns()[index];
    }
    table
}

fn fixed_backend_from_rows(
    log_t: usize,
    rows: &[SpartanOuterRow],
    #[cfg(feature = "field-inline")] field_rows: &[(usize, FieldInlineSpartanRow<Fr>)],
) -> FixedBackend<Fr> {
    let mut backend = FixedBackend::new();
    for (index, variable) in SPARTAN_OUTER_R1CS_INPUTS.iter().enumerate() {
        let values: Vec<Fr> = rows
            .iter()
            .map(|row| variable_field_value(row, index))
            .collect();
        backend
            .insert(
                JoltPolynomialId::Virtual(*variable),
                Shape::new(log_t, PolynomialEncoding::Dense),
                values,
            )
            .unwrap();
    }
    #[cfg(feature = "field-inline")]
    {
        let mut field_inline = FixedFieldInline::default();
        for (index, id) in FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS.iter().enumerate() {
            field_inline
                .insert(
                    FieldInlinePolynomialId::Virtual(*id),
                    Shape::new(log_t, PolynomialEncoding::Dense),
                    field_column_table(field_rows, 1 << log_t, index),
                )
                .unwrap();
        }
        backend.set_field_inline(field_inline);
    }
    backend
}

/// The remainder's true input claim
/// `Σ_{t,s} kernel · eq(τ_low, (t,s)) · Az(t,s) · Bz(t,s)`, computed
/// through the public `jolt-claims` column-weight path over the COMPOSED
/// opening selection (independent of both kernels' row-value pipelines).
fn true_input_claim(
    rows: &[SpartanOuterRow],
    #[cfg(feature = "field-inline")] field_rows: &[(usize, FieldInlineSpartanRow<Fr>)],
    tau: &[Fr],
    r0: Fr,
    log_t: usize,
) -> Fr {
    let tau_low = &tau[..=log_t];
    let tau_high = tau[log_t + 1];
    let eq = EqPolynomial::new(tau_low.to_vec()).evaluations();
    let kernel = centered_lagrange_kernel::<Fr>(DOMAIN, tau_high, r0).unwrap();
    let matrices = spartan_outer_constraints::<Fr>();
    let columns: Vec<usize> = spartan_outer_opening_columns();
    let value = |t: usize, position: usize| -> Fr {
        if position < VARIABLE_COUNT {
            return variable_field_value(&rows[t], position);
        }
        #[cfg(feature = "field-inline")]
        {
            field_rows
                .iter()
                .find(|(cycle, _)| *cycle == t)
                .map_or_else(
                    || Fr::from_u64(0),
                    |(_, row)| row.columns()[position - VARIABLE_COUNT],
                )
        }
        #[cfg(not(feature = "field-inline"))]
        unreachable!("the rv64 selection has exactly 35 columns")
    };
    let mut total = Fr::from_u64(0);
    for (s, stream) in [Fr::from_u64(0), Fr::from_u64(1)].into_iter().enumerate() {
        let weights = spartan_outer_row_weights(r0, stream).unwrap();
        let weighted = matrices.weighted_columns(&weights, &columns).unwrap();
        let constants = matrices
            .public_column_contributions(&weights, 0, Fr::from_u64(1))
            .unwrap();
        for t in 0..rows.len() {
            let mut az = constants.a;
            let mut bz = constants.b;
            for (index, (&a, &b)) in weighted.a.iter().zip(&weighted.b).enumerate() {
                let value = value(t, index);
                az += a * value;
                bz += b * value;
            }
            total += eq[(t << 1) | s] * az * bz;
        }
    }
    kernel * total
}

fn parity_case(dummy_plane: &dyn JoltWitnessPlane<Fr>, log_t: usize, seed: u64) {
    parity_case_with_eq(dummy_plane, log_t, seed, None, false, None, None);
}

fn parity_case_with_eq(
    dummy_plane: &dyn JoltWitnessPlane<Fr>,
    log_t: usize,
    seed: u64,
    exceptional: Option<ExceptionalEq>,
    zero_scale: bool,
    prefix: Option<usize>,
    period: Option<usize>,
) {
    let mut rows = synthetic_rows(log_t, seed);
    if let Some(period) = period {
        for t in period..rows.len() {
            rows[t] = rows[t % period];
        }
    }
    if let Some(prefix) = prefix {
        let tail = rows[rows.len() - 1];
        rows[prefix..].fill(tail);
    }
    #[cfg(feature = "field-inline")]
    let mut field_rows = synthetic_field_rows(log_t, seed ^ 0xF1E1D);
    #[cfg(feature = "field-inline")]
    if let Some(prefix) = prefix {
        // Odd seeds keep field rows past the integer suffix boundary.
        crate::optimized::testing::trim_field_rows(&mut field_rows, log_t, prefix, seed % 2 == 1);
    }
    let mut tau: Vec<Fr> = exceptional.map_or_else(
        || {
            (0..log_t + 2)
                .map(|i| Fr::from_u64(3 + seed + 7 * i as u64))
                .collect()
        },
        |case| {
            let mut point = case.point(log_t + 1, Fr::from_u64(7919 + seed));
            point.push(Fr::from_u64(3 + seed + 7 * (log_t + 1) as u64));
            point
        },
    );
    if zero_scale {
        tau[log_t + 1] = Fr::from_u64(0);
    }
    let backend = fixed_backend_from_rows(
        log_t,
        &rows,
        #[cfg(feature = "field-inline")]
        &field_rows,
    );

    let mut reference_session = ProofSession::default();
    reference_session.park(SpartanOuterKernel::<Fr>::prepare(log_t, &tau, &backend).unwrap());
    let reference_uniskip =
        <ReferenceBackend as UniskipKernel<Fr, OuterRemainder<Fr>>>::first_round_poly(
            &ReferenceBackend,
            &mut reference_session,
            &[],
            &(),
        )
        .unwrap();

    let mut optimized_session = ProofSession::default();
    OptimizedOuterUniskip::prepare_from_rows(
        &mut optimized_session,
        log_t,
        &tau,
        rows.clone(),
        #[cfg(feature = "field-inline")]
        field_rows.clone(),
    )
    .unwrap();
    let optimized_uniskip =
        <OptimizedOuterUniskip as UniskipKernel<Fr, OuterRemainder<Fr>>>::first_round_poly(
            &OptimizedOuterUniskip,
            &mut optimized_session,
            &[],
            &(),
        )
        .unwrap();
    assert_eq!(
        optimized_uniskip, reference_uniskip,
        "uni-skip first-round polynomial, log_t = {log_t}"
    );

    let r0 = Fr::from_u64(if zero_scale { 1 } else { 40961 + seed });
    let input_claim = true_input_claim(
        &rows,
        #[cfg(feature = "field-inline")]
        &field_rows,
        &tau,
        r0,
        log_t,
    );
    let relation = OuterRemainder::new(SpartanOuterDimensions::rv64(log_t), tau.clone(), r0);
    let claims = outer_remainder_input_values_from_uniskip_output(input_claim);
    let points = SumcheckInputPoints::<Fr, OuterRemainder<Fr>>::default();
    let no_challenges = NoChallenges::<Fr>::default();

    let mut reference_kernel = ReferenceOuterRemainder
        .prepare(
            &mut reference_session,
            dummy_plane,
            ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &no_challenges,
            },
        )
        .unwrap();
    let mut optimized_kernel = OptimizedOuterRemainder
        .prepare(
            &mut optimized_session,
            dummy_plane,
            ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &no_challenges,
            },
        )
        .unwrap();

    let rounds = log_t + 1;
    let challenges: Vec<Fr> = (0..rounds)
        .map(|i| {
            if prefix.is_some() && seed.is_multiple_of(2) && i < 2 {
                Fr::from_u64(i as u64)
            } else {
                Fr::from_u64(7919 + seed + 31 * i as u64)
            }
        })
        .collect();
    let mut bind = None;
    let mut previous = input_claim;
    for (round, &challenge) in challenges.iter().enumerate() {
        let reference_round = reference_kernel.prove_round(bind, round, previous).unwrap();
        let optimized_round = optimized_kernel.prove_round(bind, round, previous).unwrap();
        assert_eq!(
            optimized_round, reference_round,
            "round {round} polynomial, log_t = {log_t}"
        );
        previous = reference_round.evaluate(challenge);
        bind = Some(challenge);
    }
    let last = bind.unwrap();
    reference_kernel.finish_rounds(last).unwrap();
    optimized_kernel.finish_rounds(last).unwrap();

    let reference_outputs = reference_kernel.output_claims(&claims).unwrap();
    let optimized_outputs = optimized_kernel.output_claims(&claims).unwrap();
    assert_eq!(
        optimized_outputs, reference_outputs,
        "typed output claims, log_t = {log_t}"
    );

    let output_points = relation
        .derive_opening_points(&challenges, &points)
        .unwrap();
    reference_kernel
        .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
        .unwrap();
    optimized_kernel
        .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
        .unwrap();
}

#[test]
fn remainder_matches_reference_at_exceptional_eq_and_zero_scaling() {
    with_sample_backend(|dummy| {
        for case in ExceptionalEq::ALL {
            parity_case_with_eq(dummy, 4, 389, Some(case), false, None, None);
        }
        parity_case_with_eq(dummy, 4, 389, None, true, None, None);
    });
}

#[test]
fn remainder_matches_reference_with_repeated_guard_rows() {
    with_sample_backend(|dummy| {
        // log_t = 12 gives 64-row blocks; period 48 repeats 16 rows per
        // block and collides keys in the 128-slot guard-endpoint cache.
        parity_case_with_eq(dummy, 12, 389, None, false, None, Some(48));
        parity_case_with_eq(dummy, 12, 390, None, false, Some(3000), Some(48));
    });
}

#[test]
fn constant_stream_tails_match_reference_rounds_and_openings() {
    with_sample_backend(|dummy| {
        for prefix in [0, 1, 3, 7, 8, 9, 15, 16] {
            for seed in [389, 390] {
                parity_case_with_eq(dummy, 4, seed, None, false, Some(prefix), None);
                for case in ExceptionalEq::ALL {
                    parity_case_with_eq(dummy, 4, seed, Some(case), false, Some(prefix), None);
                }
            }
            parity_case_with_eq(dummy, 4, 389, None, true, Some(prefix), None);
        }
    });
}

#[test]
fn uniskip_matches_reference_with_many_guard_patterns() {
    let log_t = 12;
    let rows = synthetic_rows(log_t, 0x6a_726f_7773);
    #[cfg(feature = "field-inline")]
    let field_rows = synthetic_field_rows(log_t, 0x6a_6669_656c);
    let tau: Vec<Fr> = (0..log_t + 2)
        .map(|i| Fr::from_u64(991 + 17 * i as u64))
        .collect();
    let backend = fixed_backend_from_rows(
        log_t,
        &rows,
        #[cfg(feature = "field-inline")]
        &field_rows,
    );
    let mut reference = ProofSession::default();
    reference.park(SpartanOuterKernel::<Fr>::prepare(log_t, &tau, &backend).unwrap());
    let expected = <ReferenceBackend as UniskipKernel<Fr, OuterRemainder<Fr>>>::first_round_poly(
        &ReferenceBackend,
        &mut reference,
        &[],
        &(),
    )
    .unwrap();
    let mut optimized = ProofSession::default();
    OptimizedOuterUniskip::prepare_from_rows(
        &mut optimized,
        log_t,
        &tau,
        rows,
        #[cfg(feature = "field-inline")]
        field_rows,
    )
    .unwrap();
    let actual =
        <OptimizedOuterUniskip as UniskipKernel<Fr, OuterRemainder<Fr>>>::first_round_poly(
            &OptimizedOuterUniskip,
            &mut optimized,
            &[],
            &(),
        )
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn synthetic_parity_with_reference_kernels() {
    with_sample_backend(|dummy| {
        for (log_t, seed) in [(1usize, 111u64), (2, 222), (3, 333), (4, 444)] {
            parity_case(dummy, log_t, seed);
        }
    });
}

/// The trait-path parity body over a real trace backend: the optimized
/// bundle walk against the reference's oracle tables, with the remainder
/// driven by the true joint-domain sum. The trace fixtures are
/// witness-extraction fixtures, not constraint-satisfying traces, so the
/// uni-skip reduction at r0 need not equal the joint-domain sum here; the
/// remainder runs on the true sum, which is what the naive reference
/// self-checks against.
fn sample_case(backend: &TraceBackend, log_t: usize) {
    let tau: Vec<Fr> = (0..log_t + 2)
        .map(|i| Fr::from_u64(29 + 13 * i as u64))
        .collect();

    let mut reference_session = ProofSession::default();
    <ReferenceBackend as UniskipKernel<Fr, OuterRemainder<Fr>>>::prepare(
        &ReferenceBackend,
        &mut reference_session,
        log_t,
        &tau,
        backend,
    )
    .unwrap();
    let reference_uniskip =
        <ReferenceBackend as UniskipKernel<Fr, OuterRemainder<Fr>>>::first_round_poly(
            &ReferenceBackend,
            &mut reference_session,
            &[],
            &(),
        )
        .unwrap();

    let mut optimized_session = ProofSession::default();
    <OptimizedOuterUniskip as UniskipKernel<Fr, OuterRemainder<Fr>>>::prepare(
        &OptimizedOuterUniskip,
        &mut optimized_session,
        log_t,
        &tau,
        backend,
    )
    .unwrap();
    let optimized_uniskip =
        <OptimizedOuterUniskip as UniskipKernel<Fr, OuterRemainder<Fr>>>::first_round_poly(
            &OptimizedOuterUniskip,
            &mut optimized_session,
            &[],
            &(),
        )
        .unwrap();
    assert_eq!(optimized_uniskip, reference_uniskip);

    let r0 = Fr::from_u64(9173);
    let rows: Vec<SpartanOuterRow> = backend.bundles().unwrap();
    #[cfg(feature = "field-inline")]
    let field_rows = JoltWitnessOracle::<Fr>::field_inline(backend)
        .unwrap()
        .field_inline_spartan_rows()
        .unwrap();
    let input_claim = true_input_claim(
        &rows,
        #[cfg(feature = "field-inline")]
        &field_rows,
        &tau,
        r0,
        log_t,
    );

    let relation = OuterRemainder::new(SpartanOuterDimensions::rv64(log_t), tau.clone(), r0);
    let claims = outer_remainder_input_values_from_uniskip_output(input_claim);
    let points = SumcheckInputPoints::<Fr, OuterRemainder<Fr>>::default();
    let no_challenges = NoChallenges::<Fr>::default();
    let mut reference_kernel = ReferenceOuterRemainder
        .prepare(
            &mut reference_session,
            backend,
            ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &no_challenges,
            },
        )
        .unwrap();
    let mut optimized_kernel = OptimizedOuterRemainder
        .prepare(
            &mut optimized_session,
            backend,
            ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &no_challenges,
            },
        )
        .unwrap();

    let challenges: Vec<Fr> = (0..=log_t)
        .map(|i| Fr::from_u64(523 + 17 * i as u64))
        .collect();
    let mut bind = None;
    let mut previous = input_claim;
    for (round, &challenge) in challenges.iter().enumerate() {
        let reference_round = reference_kernel.prove_round(bind, round, previous).unwrap();
        let optimized_round = optimized_kernel.prove_round(bind, round, previous).unwrap();
        assert_eq!(optimized_round, reference_round, "round {round}");
        previous = reference_round.evaluate(challenge);
        bind = Some(challenge);
    }
    let last = bind.unwrap();
    reference_kernel.finish_rounds(last).unwrap();
    optimized_kernel.finish_rounds(last).unwrap();
    assert_eq!(
        optimized_kernel.output_claims(&claims).unwrap(),
        reference_kernel.output_claims(&claims).unwrap()
    );
}

/// Full trait-path parity: without field-inline on the canned sample trace; with
/// field-inline enabled over a field-inline fixture trace (the sample backend
/// carries no field-inline view), exercising the trace-backed sparse field-inline
/// row seam.
#[test]
fn sample_trace_parity_through_the_trait_path() {
    #[cfg(not(feature = "field-inline"))]
    with_sample_backend(|backend| sample_case(backend, 2));
    #[cfg(feature = "field-inline")]
    crate::optimized::field_registers_testing::structured_field_register_fixture(12)
        .with_plane(4, |backend| sample_case(backend, 4));
}

#[test]
#[cfg(not(feature = "field-inline"))]
fn prepared_b_folds_match_canonical_matrix() {
    #[cfg(feature = "akita")]
    use jolt_field::Prime128Offset6A23801F;
    fn check<F: JoltField>() {
        let mut rows = synthetic_rows(8, 0xB0_1D);
        for negative in [false, true] {
            let mut row = rows[0];
            row.right_lookup_operand.0 = u128::MAX;
            row.right_instruction_input.0 = if negative { i128::MIN } else { i128::MAX };
            row.imm.0 = row.right_instruction_input.0;
            row.product.0 = S128::new([u64::MAX; 2], !negative);
            row.ram_address.0 = u64::MAX;
            row.ram_read_value.0 = u64::MAX;
            row.ram_write_value.0 = u64::MAX;
            rows.push(row);
        }
        // Wide columns at the zero, `u64` and two-limb boundaries.
        for (right_lookup, right_input, product) in [
            (0u128, -1i128, S128::new([1, 0], false)),
            (
                u128::from(u64::MAX),
                i128::from(u64::MAX),
                S128::new([u64::MAX, 0], true),
            ),
            (1u128 << 64, -(1i128 << 64), S128::new([0, 1], false)),
        ] {
            let mut row = rows[0];
            row.right_lookup_operand.0 = right_lookup;
            row.right_instruction_input.0 = right_input;
            row.imm.0 = right_input;
            row.product.0 = product;
            rows.push(row);
        }
        let large = F::from_u128(u128::MAX) * F::from_u128(u128::MAX - 0x1234_5678);
        for challenge in [F::zero(), F::one(), F::from_u64(19), -F::one(), large] {
            let derived = OuterRemainderKernel::<F>::derived_weights(challenge).unwrap();
            let folds = PreparedBFolds::new(&derived).unwrap();
            for row in &rows {
                let expected = std::array::from_fn(|stream| {
                    derived.bz_constant[stream]
                        + derived.bz_weights[stream]
                            .iter()
                            .enumerate()
                            .map(|(index, &weight)| weight * variable_field_value::<F>(row, index))
                            .sum::<F>()
                });
                assert_eq!(folds.fold(row), expected);
            }
            // A matrix that reads a column the fold does not support fails
            // loudly instead of silently dropping it.
            let mut changed = derived;
            changed.bz_weights[0][1] = F::one();
            assert!(PreparedBFolds::new(&changed).is_err());
        }
    }
    check::<Fr>();
    #[cfg(feature = "akita")]
    check::<Prime128Offset6A23801F>();
}

/// The integer extension of each base-window basis vector is exactly the
/// field Lagrange basis evaluated at every extended node — by linearity,
/// the fact that ties the integer pipeline to the reference's field
/// pipeline.
#[test]
fn extension_matches_field_lagrange() {
    for i in 0..DOMAIN {
        let mut basis = [0i128; DOMAIN];
        basis[i] = 1;
        for (position, value) in EXTENDED_POSITIONS.into_iter().zip(extend(&basis)) {
            let node = EXTENDED_START + position as i64;
            let expected = centered_lagrange_evals::<Fr>(DOMAIN, Fr::from_i64(node)).unwrap();
            assert_eq!(Fr::from_i128(value), expected[i], "node {node}, basis {i}");
        }
    }
}

/// The [`RowGroupValues`] ranges the [`EXTENSION_GAIN`] assertions
/// assume: guards of magnitude ≤ 1 except one ≤ 2 and one ≤ 3 per group,
/// B values (or halves) below 2^65.
#[test]
fn group_values_within_extension_ranges() {
    for row in synthetic_rows(8, 0xB0_0D) {
        let values = row.group_values();
        for guards in [values.a_first.as_slice(), values.a_second.as_slice()] {
            let excess: u64 = guards
                .iter()
                .map(|guard| guard.unsigned_abs().saturating_sub(1))
                .sum();
            assert!(excess <= 3);
        }
        assert!(values.b_first.iter().all(|b| b.unsigned_abs() <= 1 << 64));
        assert!(values
            .b_second_lo
            .iter()
            .chain(&values.b_second_hi)
            .all(|b| b.unsigned_abs() < 1 << 65));
    }
}

/// The typed bundle's columns equal the oracle tables the reference
/// kernel materializes — the two witness paths meeting at the shared
/// `Extract` impls, for all ordinary R1CS inputs.
#[test]
fn bundle_columns_match_oracle_tables() {
    with_sample_backend(|backend| {
        let rows: Vec<SpartanOuterRow> = backend.bundles().unwrap();
        for (index, variable) in SPARTAN_OUTER_R1CS_INPUTS.iter().enumerate() {
            let table: Vec<Fr> = JoltWitnessOracle::<Fr>::oracle_table(
                backend,
                JoltPolynomialId::Virtual(*variable),
            )
            .unwrap();
            let column: Vec<Fr> = rows
                .iter()
                .map(|row| variable_field_value(row, index))
                .collect();
            assert_eq!(column, table, "{variable:?}");
        }
    });
}
