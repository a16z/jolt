//! Replay-backed witness rows against the eager trace of the same guest: the
//! row handle's visits and the trace-derived oracle tables must not depend on
//! whether rows are retained or re-executed from tracer checkpoints.

#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
mod support;

#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, reason = "fixtures fail loudly")]
mod replay_rows {
    use std::ops::Range;
    use std::sync::Arc;

    #[cfg(feature = "akita")]
    use jolt_akita::AkitaField as Field;
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    #[cfg(not(feature = "akita"))]
    use jolt_field::Fr as Field;
    use jolt_program::execution::TraceInputs;
    use jolt_prover::ProverConfig;
    use jolt_riscv::{JoltTraceRow, CIRCUIT_FLAGS};
    use jolt_witness::{
        JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessOracle, RowSource, TraceBackend,
        TraceRows,
    };
    use tracer::execution_backend::TracerBackend;

    use crate::support::{self, GuestCase};

    type Callbacks = Vec<(Vec<JoltTraceRow>, Option<JoltTraceRow>)>;

    fn visits(rows: &TraceRows, range: Range<usize>, chunk: usize) -> Callbacks {
        let mut callbacks = Vec::new();
        rows.visit(range, chunk, &mut |buffer, next, _| {
            callbacks.push((buffer.to_vec(), next.copied()));
            Ok(())
        })
        .expect("visit");
        callbacks
    }

    /// The committed columns plus each other row-reading path (cycle column,
    /// lookahead column, indexed flag family, register grids). `check_guest`
    /// compares every row and lookahead, so further cycle columns would repeat
    /// the same walk; the `ram_K x T` RAM grids are compared on a small
    /// fixture in jolt-witness.
    fn trace_tables(witness: &TraceBackend) -> Vec<JoltPolynomialId> {
        use JoltVirtualPolynomial as V;
        let committed = witness.committed_polynomial_order().expect("order");
        let mut ids: Vec<_> = committed.into_iter().map(JoltPolynomialId::from).collect();
        ids.extend(CIRCUIT_FLAGS.map(|flag| JoltPolynomialId::from(V::OpFlags(flag))));
        ids.extend(
            [V::RamReadValue, V::NextPC, V::RegistersVal, V::RdWa].map(JoltPolynomialId::from),
        );
        ids
    }

    fn check_guest(case: GuestCase) {
        let prepared = support::prepare(&case);
        let config = ProverConfig::derive_from_dimensions::<Field>(
            prepared.trace.dimensions,
            &prepared.preprocessing.memory_layout,
            prepared.preprocessing.ram.min_bytecode_address,
            prepared.preprocessing.ram.bytecode_words.len(),
            prepared.preprocessing.max_padded_trace_length,
        )
        .expect("derive config");
        let witness_config = JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        );
        let inputs = TraceInputs::new(
            case.inputs.clone(),
            case.untrusted_advice.clone(),
            case.trusted_advice.clone(),
            support::memory_config(&prepared.preprocessing.memory_layout),
        );
        let eager_rows = prepared.trace.trace.proof_rows();
        // The first row that continues the previous row's instruction, so a
        // visit starting there seeks into a multi-row tick.
        let tick_interior = eager_rows
            .windows(2)
            .position(|pair| {
                pair[0]
                    .virtual_sequence_remaining()
                    .is_some_and(|left| left > 0)
            })
            .map(|index| index + 1)
            .expect("the guest executes a virtual sequence");
        let proof_len = eager_rows.len();
        let preprocessing = Arc::new(prepared.preprocessing);
        let resident = TraceBackend::new(
            witness_config.clone(),
            JoltVmWitnessInputs::new(&prepared.program, &preprocessing, prepared.trace.clone()),
        );
        let resident_rows = resident.rows().expect("resident rows");
        let cycles = resident_rows.cycles();
        assert!(proof_len + 1 < cycles, "{}: no padding to cross", case.name);
        let ranges = [
            0..cycles,
            tick_interior..cycles - 1,
            tick_interior..(tick_interior + 5).min(cycles),
            proof_len - 2..(proof_len + 3).min(cycles),
            proof_len + 1..cycles,
        ];

        for segment_rows in [4, 64] {
            let replayed = TracerBackend::new()
                .execute_replay(&prepared.program, inputs.clone(), segment_rows)
                .expect("replay");
            assert_eq!(replayed.dimensions, prepared.trace.dimensions);
            let replay = TraceBackend::new(
                witness_config.clone(),
                JoltVmWitnessInputs::replayed(&prepared.program, &preprocessing, replayed),
            );
            let replay_rows = replay.rows().expect("replay rows");
            assert_eq!(replay_rows.segment_len(), segment_rows);
            for range in &ranges {
                for chunk in [1, 3, 64, cycles] {
                    assert!(
                        visits(&replay_rows, range.clone(), chunk)
                            == visits(&resident_rows, range.clone(), chunk),
                        "{}: segment {segment_rows}, range {range:?}, chunk {chunk}",
                        case.name,
                    );
                }
            }
            for id in trace_tables(&resident) {
                let table = |witness| {
                    JoltWitnessOracle::<Field>::oracle_table(witness, id).expect("oracle table")
                };
                assert!(
                    table(&replay) == table(&resident),
                    "{}: segment {segment_rows}, {id:?}",
                    case.name,
                );
            }
        }
    }

    #[test]
    fn muldiv_replay_rows_match_eager_rows() {
        check_guest(GuestCase {
            inputs: postcard::to_stdvec(&[9u32, 5, 3]).expect("serialize inputs"),
            expected_output: Some(postcard::to_stdvec(&15u32).expect("serialize output")),
            ..GuestCase::new("muldiv-guest")
        });
    }

    #[test]
    fn fibonacci_replay_rows_match_eager_rows() {
        check_guest(GuestCase {
            inputs: postcard::to_stdvec(&100u32).expect("serialize inputs"),
            expected_output: Some(
                postcard::to_stdvec(&354_224_848_179_261_915_075u128).expect("serialize output"),
            ),
            ..GuestCase::new("fibonacci-guest")
        });
    }
}
