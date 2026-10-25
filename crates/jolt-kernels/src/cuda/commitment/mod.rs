//! Signed witness increments for CUDA row commitments.
//!
//! CUDA subtracts packed pre/post values on device and keeps each difference
//! as a `u64` magnitude plus a sign byte. Optimized constructs signed `i128`
//! increments from host witness bundles for its streaming commitment calls.

use super::common::device_columns::whole_domain;
use super::common::msm::IncrementKind;
use crate::reference::commitment::{column_kinds, ColumnKind};

pub(super) const fn increment_kind(kind: ColumnKind) -> Option<IncrementKind> {
    match kind {
        ColumnKind::RdInc => Some(IncrementKind::Rd),
        ColumnKind::RamInc => Some(IncrementKind::Ram),
        ColumnKind::InstructionRa(_) | ColumnKind::BytecodeRa(_) | ColumnKind::RamRa(_) => None,
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltOneHotConfig, TracePolynomialOrder,
    };
    use jolt_field::Fr;
    use jolt_witness::backend::cuda::{
        EXTRA_RAM_READ, EXTRA_RAM_WRITE, EXTRA_RD_POST, EXTRA_RD_PRE, EXTRA_REGISTERS, EXTRA_WORDS,
        REGISTER_ABSENT,
    };
    use jolt_witness::{JoltWitnessOracle, JoltWitnessPlane};

    use crate::commitment::{CommitmentGrid, CommittedColumnsWitness};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::with_r1cs_witness;
    use crate::ProofSession;

    const RAM_K: usize = 1 << 10;
    const LOG_K_CHUNK: usize = 8;
    const TOTAL_VARS: usize = 16;

    const fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: LOG_K_CHUNK as u8,
            lookups_ra_virtual_log_k_chunk: 32,
        }
    }

    const fn grid_at(order: TracePolynomialOrder, log_t: usize) -> CommitmentGrid {
        CommitmentGrid {
            total_vars: TOTAL_VARS,
            log_t,
            log_k_chunk: LOG_K_CHUNK,
            order,
        }
    }

    fn trace_ids(witness: &dyn JoltWitnessOracle<Fr>) -> Vec<JoltCommittedPolynomial> {
        witness
            .committed_order()
            .expect("committed order")
            .into_iter()
            .filter(|id| {
                !matches!(
                    id,
                    JoltCommittedPolynomial::TrustedAdvice
                        | JoltCommittedPolynomial::UntrustedAdvice
                )
            })
            .collect()
    }

    #[test]
    fn the_increment_kernel_source_agrees_on_the_packed_layout() {
        let source = include_str!("../kernels/commit_increments.cu");
        for (name, value) in [
            ("CI_EXTRA_WORDS", EXTRA_WORDS),
            ("CI_EXTRA_RD_POST", EXTRA_RD_POST),
            ("CI_EXTRA_RAM_READ", EXTRA_RAM_READ),
            ("CI_EXTRA_RAM_WRITE", EXTRA_RAM_WRITE),
            ("CI_EXTRA_REGISTERS", EXTRA_REGISTERS),
            ("CI_EXTRA_RD_PRE", EXTRA_RD_PRE),
            ("CI_REGISTER_ABSENT", REGISTER_ABSENT as usize),
        ] {
            let expected = format!("#define {name} {value}");
            assert!(
                source.contains(&expected),
                "the CUDA source must declare `{expected}`",
            );
        }
    }

    #[test]
    fn device_increment_columns_match_the_host_encoder() {
        let Some(context) = shared_context() else {
            return;
        };
        let log_t = 8;
        let cycles = 1usize << log_t;
        let grid = grid_at(TracePolynomialOrder::CycleMajor, log_t);
        with_r1cs_witness(log_t, RAM_K, one_hot(), 23, |witness| {
            let ids = trace_ids(witness);
            let kinds = super::column_kinds::<Fr>(&ids, grid).expect("column kinds");
            let plane = witness as &dyn JoltWitnessPlane<Fr>;
            let rows =
                crate::optimized::support::collect_rows::<CommittedColumnsWitness>(plane, cycles)
                    .expect("reference rows");

            let mut session = ProofSession::default();
            let trace = crate::cuda::witness::session_device_trace_window(
                context,
                &mut session,
                plane,
                cycles,
                &super::whole_domain(cycles),
            )
            .expect("device residency");

            let mut dense_columns = 0usize;
            for (&id, &kind) in ids.iter().zip(&kinds) {
                let Some(increment) = super::increment_kind(kind) else {
                    continue;
                };
                dense_columns += 1;
                let expected: Vec<i128> = rows.iter().map(|row| kind.increment(row)).collect();
                assert!(
                    expected.iter().any(|&value| value > 0),
                    "{id:?}: no positive increment, so a kernel that dropped the value would pass",
                );
                assert!(
                    expected.contains(&0),
                    "{id:?}: no zero increment, so the absent-operand case is untested",
                );
                let expected = context.signed_column(&expected).expect("host column");

                let got = context
                    .increment_column(trace.extras(), increment, cycles)
                    .expect("device increment column");

                assert_eq!(
                    context.download_u64(got.magnitudes()).expect("magnitudes"),
                    context
                        .download_u64(expected.magnitudes())
                        .expect("magnitudes"),
                    "{id:?}: the device increment magnitudes diverge from the host encoder",
                );
                assert_eq!(
                    context.download_u8(got.signs()).expect("signs"),
                    context.download_u8(expected.signs()).expect("signs"),
                    "{id:?}: the device increment signs diverge from the host encoder",
                );
            }
            assert_eq!(
                dense_columns, 2,
                "the fixture must commit both increment columns",
            );
        });
    }
}
