#![cfg(all(feature = "metal", target_os = "macos"))]
#![expect(clippy::expect_used, reason = "integration test")]

use jolt_kernels::metal::solinas::{InstructionRaMaterializeWidth, MetalError, SolinasMetal};

#[path = "../examples/support/instruction_ra.rs"]
#[expect(
    dead_code,
    reason = "shared evaluator support includes benchmark-only entry points"
)]
mod instruction_ra;

use instruction_ra::{
    derived_eq_cycle_is_exact, expected_cpu_states, expected_hybrid_states,
    final_relation_is_exact, first_factor_only_gamma_unscale, run_cpu, run_hybrid, Capture,
    SequenceDispatch, Workload, FACTORS,
};

#[test]
fn instruction_ra_sequence_matches_lazy_cpu_across_materialization_schedules() {
    const LOG_N: usize = 12;
    const CUTOFF: usize = 8;

    let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
    let workload = Workload::new(LOG_N, 1).expect("small production geometry should build");

    for (materialize_width, reuse_inverse_for_dense) in [
        (InstructionRaMaterializeWidth::W16, false),
        (InstructionRaMaterializeWidth::W256, true),
    ] {
        let setup_plane = workload
            .prepare_plane(&context)
            .expect("resident Instruction RA plane should prepare");
        assert_eq!(setup_plane.len(), 1 << LOG_N);
        assert_eq!(setup_plane.logical_bytes(), 20 * (1 << LOG_N));
        let persistent_plane = (!reuse_inverse_for_dense).then(|| setup_plane.clone());
        let run_plane = if reuse_inverse_for_dense {
            workload
                .prepare_plane(&context)
                .expect("one-shot Instruction RA plane should prepare")
        } else {
            persistent_plane
                .as_ref()
                .expect("persistent Instruction RA plane should exist")
                .clone()
        };
        let dispatch = SequenceDispatch {
            message_threads: 128,
            materialize_threads: 64,
            materialize_width,
            reuse_inverse_for_dense,
        };
        let mut sequence = workload
            .prepare_sequence(&context, setup_plane, dispatch)
            .expect("Instruction RA sequence should prepare");
        let capture = Capture::validation(workload.rows(), materialize_width.elements());
        let cpu = run_cpu(&workload, CUTOFF, capture)
            .expect("optimized lazy CPU control should complete");
        let hybrid = run_hybrid(&mut sequence, run_plane, &workload, CUTOFF, capture)
            .expect("Metal hybrid should complete");

        assert_eq!(hybrid.trace.q_evals, cpu.trace.q_evals, "four q samples");
        assert_eq!(hybrid.trace.round_polys, cpu.trace.round_polys);
        assert_eq!(hybrid.trace.challenges, cpu.trace.challenges);
        assert_eq!(cpu.trace.states, expected_cpu_states(LOG_N));
        assert_eq!(
            hybrid.trace.states,
            expected_hybrid_states(LOG_N, materialize_width.elements(), CUTOFF)
        );
        if materialize_width != InstructionRaMaterializeWidth::W16 {
            assert_ne!(hybrid.trace.states, cpu.trace.states);
        }
        assert_eq!(hybrid.trace.scheduled_tables, cpu.trace.scheduled_tables);
        assert!(hybrid.trace.scheduled_tables.is_some());
        assert_eq!(hybrid.trace.cutoff_tables, cpu.trace.cutoff_tables);
        assert!(hybrid.trace.cutoff_tables.is_some());
        assert_eq!(hybrid.trace.raw_final_claims, cpu.trace.raw_final_claims);
        assert_eq!(hybrid.trace.final_claims, cpu.trace.final_claims);
        assert_eq!(hybrid.trace.final_claims.len(), FACTORS);
        assert_eq!(
            hybrid.trace.final_sumcheck_claim,
            cpu.trace.final_sumcheck_claim
        );
        assert_eq!(hybrid.trace.derived_eq_cycle, cpu.trace.derived_eq_cycle);
        assert_eq!(hybrid.trace.transcript_state, cpu.trace.transcript_state);
        assert!(first_factor_only_gamma_unscale(
            &hybrid.trace,
            workload.gamma
        ));
        assert!(derived_eq_cycle_is_exact(&workload, &hybrid.trace));
        assert!(final_relation_is_exact(&hybrid.trace));
        assert!(hybrid.resident_plane_zero_copy);
        assert!(hybrid.static_device_buffers_stable);
        assert!(hybrid.inverse_dense_b_handoff_exact);
        assert_eq!(
            hybrid.preallocated_readback_bytes,
            FACTORS * (CUTOFF + workload.rows() / materialize_width.elements()) * 16
        );

        let replay_plane = if reuse_inverse_for_dense {
            workload
                .prepare_plane(&context)
                .expect("fresh replay plane should prepare")
        } else {
            persistent_plane
                .as_ref()
                .expect("persistent Instruction RA plane should exist")
                .clone()
        };
        let replay = run_hybrid(&mut sequence, replay_plane, &workload, CUTOFF, capture)
            .expect("reset Metal sequence should complete");
        assert_eq!(replay.trace, hybrid.trace);
    }
}

#[test]
fn instruction_ra_lookup_plane_rejects_out_of_range_inverse() {
    let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
    let lookups = vec![0u128; 32];
    let mut inverse = (0..32u32).collect::<Vec<_>>();
    inverse[7] = 32;
    assert!(matches!(
        context.prepare_instruction_ra_lookup_plane(&lookups, &inverse),
        Err(MetalError::InputTooLong(32))
    ));
}

#[test]
fn instruction_ra_inverse_reuse_rejects_a_stale_plane_clone() {
    let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
    let workload = Workload::new(10, 1).expect("small production geometry should build");
    let plane = workload
        .prepare_plane(&context)
        .expect("resident Instruction RA plane should prepare");
    let stale = plane.clone();
    let dispatch = SequenceDispatch {
        message_threads: 128,
        materialize_threads: 64,
        materialize_width: InstructionRaMaterializeWidth::W32,
        reuse_inverse_for_dense: true,
    };
    let mut sequence = workload
        .prepare_sequence(&context, plane, dispatch)
        .expect("the first inverse owner should prepare");
    assert!(matches!(
        sequence.reset(stale, &workload.chunk_tables),
        Err(MetalError::InvalidInstructionRaState(_))
    ));
}
