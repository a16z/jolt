//! The explicit per-proof memory budget: an operational request beside the
//! proof shape ([`ProverConfig`]). It changes no dimension, schedule, kernel,
//! transcript or PCS choice; [`MemoryBudget::check_known_eager`] only rejects
//! a request below the optimized prover's known eager reservation, after
//! tracing and before proving starts.

use jolt_field::JoltField;
use jolt_kernels::optimized::instruction_input::OptimizedInstructionInput;
use jolt_kernels::optimized::ram_access_columns_bytes;
use jolt_kernels::optimized::registers_claim_reduction::OptimizedRegistersClaimReduction;
use jolt_kernels::optimized::spartan_outer::OptimizedOuterRemainder;
use jolt_kernels::optimized::spartan_product::OptimizedProductRemainder;
use jolt_kernels::optimized::spartan_shift::OptimizedSpartanShift;
use jolt_program::execution::TraceData;

use crate::{ProverConfig, ProverError};

/// One batch member's peak bytes under the kernel
/// [`JoltBackend::optimized`](crate::JoltBackend::optimized) installs in its
/// slot. Every member of the priced batches has an arm, so a member added to
/// one of them fails to compile here until it is classified.
macro_rules! member_peak_bytes {
    ($F:ty, $log_t:ident, outer_remainder) => {
        OptimizedOuterRemainder::peak_bytes::<$F>($log_t)
    };
    ($F:ty, $log_t:ident, product_remainder) => {
        OptimizedProductRemainder::peak_bytes::<$F>($log_t)
    };
    ($F:ty, $log_t:ident, shift) => {
        OptimizedSpartanShift::peak_bytes::<$F>($log_t)
    };
    ($F:ty, $log_t:ident, instruction_input) => {
        OptimizedInstructionInput::peak_bytes::<$F>($log_t)
    };
    ($F:ty, $log_t:ident, registers_claim_reduction) => {
        OptimizedRegistersClaimReduction::peak_bytes::<$F>($log_t)
    };
    // Not priced yet: the reservation omits these members' allocations.
    ($F:ty, $log_t:ident, ram_read_write) => {
        0
    };
    ($F:ty, $log_t:ident, instruction_claim_reduction) => {
        0
    };
    ($F:ty, $log_t:ident, field_registers_claim_reduction) => {
        0
    };
    ($F:ty, $log_t:ident, ram_raf_evaluation) => {
        0
    };
    ($F:ty, $log_t:ident, ram_output_check) => {
        0
    };
}

/// One stage batch's priced bytes, folded over the verifier's member list: the
/// sum of the members' peaks, which bounds what they hold at once.
macro_rules! batch_peak_bytes {
    (
        $F:ty, $log_t:ident;
        batch = $batch:ident,
        label = $label:literal,
        aggregates = $aggregates:tt,
        shape = $shape:ident,
        members = [$({ name: $name:ident, $($member:tt)* },)+]
    ) => {
        0usize $(.saturating_add(member_peak_bytes!($F, $log_t, $name)))+
    };
}

/// An explicit memory request, in bytes, for one optimized proof. Only
/// [`MemoryBudget::check_known_eager`] reads it; it does not cap process
/// memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    bytes: usize,
}

impl MemoryBudget {
    pub const fn from_bytes(bytes: usize) -> Self {
        Self { bytes }
    }

    pub const fn bytes(self) -> usize {
        self.bytes
    }

    /// Rejects this budget with [`ProverError::MemoryBudgetExceeded`] when it
    /// is below the bytes reserved for a named subset of the eager
    /// allocations of an optimized proof of `trace` under `config`, which
    /// must be the canonical configuration derived from `trace`; the check
    /// does not validate it. The SDK and profile callers run it after tracing
    /// and before building the witness.
    ///
    /// The reservation is the trace buffers at their reserved capacity
    /// ([`TraceData::heap_bytes`]) plus the largest stage-1–3 envelope: the
    /// sum of the batch members' priced peaks, each the member kernel's own
    /// `peak_bytes`, and in stages 2 and 3 one generation of RAM access
    /// columns ([`ram_access_columns_bytes`]). Stage 2 collects them, and the
    /// address column stays parked in the session. The read-write kernel
    /// releases the value columns when its prepare returns (address-first) or
    /// queues them for a background drop at its first bind (cycle-first); no
    /// stage boundary waits for that drop. Stage 3 therefore still reserves
    /// them as an outstanding generation, which can reject a request even
    /// when the drop has already run. Summing peaks exceeds the batch's true
    /// simultaneous maximum by at most the members' transients that do not
    /// coincide, `O(√T)` in stage 3.
    ///
    /// Passing does not mean the proof fits, and the reservation is neither a
    /// lower bound on the memory a proof needs nor an allocator or RSS
    /// figure. Not reserved: the unpriced batch members (the RAM read-write
    /// kernel's increment, value and matrix tables among them) and every
    /// later stage (stage 4 collects a second pair of value columns, possibly
    /// beside the queued first pair), uni-skip and field-inline carries,
    /// split-eq tables, stage outputs and the recorder, preprocessing and PCS
    /// setup, commitment and opening state, BlindFold, other queued
    /// background drops, allocator retention, the trace output's final memory
    /// image, I/O device and advice tape, and the tracer, which has already
    /// run.
    pub fn check_known_eager<F: JoltField>(
        self,
        config: &ProverConfig,
        trace: &TraceData,
    ) -> Result<(), ProverError<F>> {
        let log_t = config.trace_length.ilog2() as usize;
        let ram_columns = ram_access_columns_bytes(log_t);
        let stage1 = jolt_verifier::stage1_batch_sumchecks_members!(batch_peak_bytes F, log_t;);
        let stage2 = jolt_verifier::stage2_batch_sumchecks_members!(batch_peak_bytes F, log_t;)
            .saturating_add(ram_columns);
        let stage3 = jolt_verifier::stage3_sumchecks_members!(batch_peak_bytes F, log_t;)
            .saturating_add(ram_columns);
        let required = trace
            .heap_bytes()
            .saturating_add(stage1.max(stage2).max(stage3));
        if self.bytes < required {
            return Err(ProverError::MemoryBudgetExceeded {
                budget: self.bytes,
                required,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::panic, reason = "test module")]
mod tests {
    use std::sync::Arc;

    use common::jolt_device::JoltDevice;
    #[cfg(feature = "akita")]
    use jolt_akita::AkitaField;
    use jolt_field::{Fr, JoltField};
    #[cfg(feature = "field-inline")]
    use jolt_program::execution::{FieldEvent, FieldInlineTraceData};
    use jolt_program::execution::{JoltProgram, TraceData, TraceOutput};
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
    };
    use jolt_riscv::JoltTraceRow;
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

    use super::MemoryBudget;
    use crate::{ProverConfig, ProverError};

    const LOG_T: usize = 13;
    const CYCLES: usize = 1 << LOG_T;
    /// Rows the fixture's trace buffer reserves beyond its `CYCLES` rows, so
    /// a check that counted rows by length instead of capacity would pass
    /// requests below the trace's actual allocation.
    const RESERVED_ROWS: usize = 1 << 24;
    const TRACE_BYTES: usize = RESERVED_ROWS * size_of::<JoltTraceRow>();

    fn config<F: JoltField>() -> ProverConfig {
        ProverConfig::from_padded_dimensions::<F>(CYCLES, 64, CYCLES).unwrap()
    }

    /// `CYCLES` default rows in a buffer reserved at `RESERVED_ROWS`.
    fn rows() -> Vec<JoltTraceRow> {
        let mut rows = Vec::with_capacity(RESERVED_ROWS);
        rows.resize(CYCLES, JoltTraceRow::default());
        rows
    }

    /// A real witness over [`rows`], and its proof shape.
    fn witness<F: JoltField>() -> (ProverConfig, TraceBackend) {
        let config = config::<F>();
        let preprocessing = JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing {
                code_size: 32,
                ..Default::default()
            },
            ram: RAMPreprocessing::default(),
            memory_layout: Default::default(),
            max_padded_trace_length: CYCLES,
        };
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(LOG_T, config.ram_K, config.one_hot_config),
            JoltVmWitnessInputs::new(
                &Arc::new(JoltProgram::default()),
                &Arc::new(preprocessing),
                TraceOutput::new(TraceData::new(rows()), JoltDevice::default(), None, None),
            ),
        );
        (config, witness)
    }

    fn check<F: JoltField>(
        budget: usize,
        config: &ProverConfig,
        trace: &TraceData,
    ) -> Result<(), ProverError<F>> {
        MemoryBudget::from_bytes(budget).check_known_eager::<F>(config, trace)
    }

    /// S1-01 regression through the production request boundary: the
    /// unchanged stage-1 Spartan outer remainder allocates `az` and `bz`, two
    /// tables of `2T` field elements, beside the trace buffer, so a request
    /// below that independently counted floor must fail typed.
    fn rejects_below_the_stage_one_floor<F: JoltField>() {
        let (config, witness) = witness::<F>();
        let floor = TRACE_BYTES + 2 * (2 * CYCLES) * size_of::<F>();
        let budget = floor - 1;
        let result = check::<F>(budget, &config, &witness.trace.trace);
        assert!(
            matches!(result, Err(ProverError::MemoryBudgetExceeded { .. })),
            "budget {budget} below the stage-1 floor {floor} was admitted"
        );
    }

    #[test]
    fn s1_01_stage_one_floor_fr() {
        rejects_below_the_stage_one_floor::<Fr>();
    }

    #[cfg(feature = "akita")]
    #[test]
    fn s1_01_stage_one_floor_akita_field() {
        rejects_below_the_stage_one_floor::<AkitaField>();
    }

    /// The stage-3 state a request must cover beside the trace: after stage
    /// 3's first bind the instruction input holds eight `T/2`-entry tables
    /// while the shift keeps its eight prefix–suffix tables and the register
    /// claim reduction its two, `2^ceil(log_t/2)` entries each. The session
    /// keeps the `u32` RAM address column, and the two `u64` value columns
    /// stage 2 queued for a background drop stay reserved as an outstanding
    /// generation.
    fn stage_three_floor<F: JoltField>() -> usize {
        let prefix = 1usize << LOG_T.div_ceil(2);
        let tables = 8 * (CYCLES / 2) + 8 * prefix + 2 * prefix;
        let ram_columns = CYCLES * (size_of::<u32>() + 2 * size_of::<u64>());
        tables * size_of::<F>() + ram_columns
    }

    /// The stage-3 state exceeds the stage-1 peak, so a request below it
    /// must fail.
    fn rejects_below_the_stage_three_floor<F: JoltField>() {
        let (config, witness) = witness::<F>();
        let floor = TRACE_BYTES + stage_three_floor::<F>();
        let stage_one_peak = TRACE_BYTES + (4 * CYCLES + CYCLES / 4) * size_of::<F>();
        assert!(floor > stage_one_peak);
        assert!(matches!(
            check::<F>(floor - 1, &config, &witness.trace.trace),
            Err(ProverError::MemoryBudgetExceeded { .. })
        ));
    }

    #[test]
    fn stage_three_simultaneous_state_fr() {
        rejects_below_the_stage_three_floor::<Fr>();
    }

    #[cfg(feature = "akita")]
    #[test]
    fn stage_three_simultaneous_state_akita_field() {
        rejects_below_the_stage_three_floor::<AkitaField>();
    }

    /// Field events count at their reserved capacity beside the rows: a
    /// request below the rows, the events and the stage-3 floor fails.
    #[cfg(feature = "field-inline")]
    #[test]
    fn field_event_capacity_is_reserved() {
        const FIELD_EVENTS: usize = CYCLES;
        let mut events = Vec::with_capacity(FIELD_EVENTS);
        events.push(FieldEvent {
            cycle: 0,
            data: FieldInlineTraceData {
                rs1: None,
                rs2: None,
                rd: None,
            },
        });
        let trace = TraceData::from_parts(rows(), events).unwrap();
        let floor =
            TRACE_BYTES + FIELD_EVENTS * size_of::<FieldEvent>() + stage_three_floor::<Fr>();
        assert!(matches!(
            check::<Fr>(floor - 1, &config::<Fr>(), &trace),
            Err(ProverError::MemoryBudgetExceeded { .. })
        ));
    }

    /// A budget at the reported requirement passes and one byte less fails
    /// with the same requirement; an unbounded budget passes.
    #[test]
    fn budget_at_the_reported_requirement_passes() {
        let (config, witness) = witness::<Fr>();
        let trace = &witness.trace.trace;
        assert!(check::<Fr>(usize::MAX, &config, trace).is_ok());
        let Err(ProverError::MemoryBudgetExceeded {
            budget: 0,
            required,
        }) = check::<Fr>(0, &config, trace)
        else {
            panic!("an empty budget passed");
        };
        assert!(check::<Fr>(required, &config, trace).is_ok());
        assert!(matches!(
            check::<Fr>(required - 1, &config, trace),
            Err(ProverError::MemoryBudgetExceeded { required: same, .. }) if same == required
        ));
    }
}
