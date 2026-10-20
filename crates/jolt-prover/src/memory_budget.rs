//! The explicit per-proof memory budget: an operational request beside the
//! proof shape ([`ProverConfig`]). It changes no dimension, schedule, kernel,
//! transcript or PCS choice; [`MemoryBudget::check_known_eager`] only rejects
//! a request below the optimized prover's known eager reservation, after
//! tracing and before proving starts.

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_field::JoltField;
use jolt_kernels::optimized::instruction_claim_reduction::OptimizedInstructionClaimReduction;
use jolt_kernels::optimized::instruction_input::OptimizedInstructionInput;
use jolt_kernels::optimized::ram_access_columns_bytes;
use jolt_kernels::optimized::ram_read_write;
use jolt_kernels::optimized::ram_val_check;
use jolt_kernels::optimized::registers_claim_reduction::OptimizedRegistersClaimReduction;
use jolt_kernels::optimized::registers_read_write::OptimizedRegistersReadWrite;
use jolt_kernels::optimized::registers_val_evaluation::OptimizedRegistersValEvaluation;
use jolt_kernels::optimized::spartan_outer::OptimizedOuterRemainder;
use jolt_kernels::optimized::spartan_product::OptimizedProductRemainder;
use jolt_kernels::optimized::spartan_shift::OptimizedSpartanShift;
use jolt_program::execution::TraceData;

use crate::{ProverConfig, ProverError};

/// One batch member's priced bytes under the kernel
/// [`JoltBackend::optimized`](crate::JoltBackend::optimized) installs in its
/// slot: the kernel's own peak unless the arm's comment says otherwise. Every
/// member of the priced batches has an arm, so a member added to one of them
/// fails to compile here until it is classified.
macro_rules! member_peak_bytes {
    ($F:ty, $log_t:ident, $registers:ident, outer_remainder) => {
        OptimizedOuterRemainder::peak_bytes::<$F>($log_t)
    };
    ($F:ty, $log_t:ident, $registers:ident, product_remainder) => {
        OptimizedProductRemainder::peak_bytes::<$F>($log_t)
    };
    ($F:ty, $log_t:ident, $registers:ident, shift) => {
        OptimizedSpartanShift::peak_bytes::<$F>($log_t)
    };
    ($F:ty, $log_t:ident, $registers:ident, instruction_input) => {
        OptimizedInstructionInput::peak_bytes::<$F>($log_t)
    };
    ($F:ty, $log_t:ident, $registers:ident, registers_claim_reduction) => {
        OptimizedRegistersClaimReduction::peak_bytes::<$F>($log_t)
    };
    ($F:ty, $log_t:ident, $registers:ident, registers_read_write) => {
        OptimizedRegistersReadWrite::peak_bytes::<$F>($registers)
    };
    ($F:ty, $log_t:ident, $registers:ident, registers_val_evaluation) => {
        OptimizedRegistersValEvaluation::peak_bytes::<$F>($log_t)
    };
    // The post-preparation window alone. Every member prepares before any
    // binds, and stage 2's RAM term (`ram_read_write::first_bind_bytes`)
    // prices the RAM matrix at its largest entry count, which covers the
    // reduction's construction collect. A census that lowers that term must
    // restore `other members + max(collect + RAM prepared, this window + RAM
    // first bind)`; shrinking stage 4's terms also reopens this proof.
    ($F:ty, $log_t:ident, $registers:ident, instruction_claim_reduction) => {
        OptimizedInstructionClaimReduction::post_prepare_peak_bytes::<$F>($log_t)
    };
    // Priced beside the stage-4 fold with stage 4's RAM access columns, from
    // preparation through the first bind (`ram_val_check::first_bind_bytes`).
    ($F:ty, $log_t:ident, $registers:ident, ram_val_check) => {
        0
    };
    // Priced beside the stage-2 fold from the read-write kernel's preparation
    // through its first bind, RAF's folded and unmap tables included
    // (`ram_read_write::first_bind_bytes`); their later states are not priced.
    ($F:ty, $log_t:ident, $registers:ident, ram_read_write) => {
        0
    };
    ($F:ty, $log_t:ident, $registers:ident, ram_raf_evaluation) => {
        0
    };
    // Not priced yet: the reservation omits these members' own allocations
    // (the RAM access columns some of them hold are added per stage).
    ($F:ty, $log_t:ident, $registers:ident, field_registers_claim_reduction) => {
        0
    };
    ($F:ty, $log_t:ident, $registers:ident, ram_output_check) => {
        0
    };
    ($F:ty, $log_t:ident, $registers:ident, field_registers_read_write) => {
        0
    };
    ($F:ty, $log_t:ident, $registers:ident, instruction_read_raf) => {
        0
    };
    ($F:ty, $log_t:ident, $registers:ident, ram_ra_claim_reduction) => {
        0
    };
    ($F:ty, $log_t:ident, $registers:ident, field_registers_val_evaluation) => {
        0
    };
}

/// One stage batch's priced bytes, folded over the verifier's member list: the
/// sum of the members' priced bytes. Where each is a kernel's own peak, the
/// sum bounds what the members hold at once; stage 2's instruction claim
/// reduction relies on the pairing [`MemoryBudget::check_known_eager`]
/// describes.
macro_rules! batch_peak_bytes {
    (
        $F:ty, $log_t:ident, $registers:ident;
        batch = $batch:ident,
        label = $label:literal,
        aggregates = $aggregates:tt,
        shape = $shape:ident,
        members = [$({ name: $name:ident, $($member:tt)* },)+]
    ) => {
        0usize $(.saturating_add(member_peak_bytes!($F, $log_t, $registers, $name)))+
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
    /// ([`TraceData::heap_bytes`]) plus the largest stage-1–5 envelope: the
    /// sum of the batch members' priced bytes, each the member kernel's own
    /// `peak_bytes` except stage 2's instruction claim reduction (below), and
    /// one RAM term in each of stages 2–5: the RAM read-write kernel through
    /// its first bind in stage 2, the RAM access columns that may still be
    /// held in stages 3 and 5 ([`ram_access_columns_bytes`]), and the value
    /// check through its first bind in stage 4. Stage 4 prices the register
    /// read/write kernel of the configured phase order with the `rd` indices
    /// it parks, stage 5 the value evaluation that takes them over. Summing
    /// peaks exceeds the batch's true simultaneous maximum by at most the
    /// members' transients that do not coincide: `O(√T)` in stage 3, while in
    /// stage 4 the register peak and the value check's first-bind window need
    /// not be live at once.
    ///
    /// Stage 2 instead pairs the instruction claim reduction with RAM by
    /// phase. The driver prepares every member before any member binds; a
    /// preparation frees its collector's partial vectors before it returns,
    /// while no queued background drop is assumed to have run. Stage 2
    /// reserves the RAM read-write kernel from its preparation through its
    /// first bind, with RAF's folded and unmap tables
    /// ([`ram_read_write::first_bind_bytes`]), and the reduction only after
    /// its preparation
    /// ([`OptimizedInstructionClaimReduction::post_prepare_peak_bytes`]). The
    /// RAM term prices the matrix at its largest entry count, enough to cover
    /// the reduction's construction collect beside RAM's preparation; that
    /// collect never coexists with a bind. The other stages keep their plain
    /// sums.
    ///
    /// The check assumes [`JoltBackend::optimized`](crate::JoltBackend::optimized)
    /// with the `SequentialRounds` member scheduler it inherits, which runs
    /// each member's round to completion in declaration order; address-first
    /// RAM's allowance for one half-length copy of a `K`-entry table at a time
    /// depends on it. `round_scheduler` is a public field, so a concurrent
    /// replacement falls outside this bound, and no type enforces the single
    /// copy.
    ///
    /// Stage 2 collects one generation of RAM access columns, and its address
    /// column stays parked in the session, one backing that every later
    /// handle shares. The read-write kernel releases the value columns when its
    /// prepare returns (address-first) or queues them for a background drop
    /// at its first bind (cycle-first); no stage boundary waits for that drop.
    /// Stages 3 and 5 therefore still reserve that generation. Stage 4
    /// reserves the value check from its preparation through its first bind
    /// ([`ram_val_check::first_bind_bytes`]): that generation, the second pair
    /// of value columns the check collects and drops at its second bind,
    /// before stage 5, and the larger of its collector's chunk views and its
    /// address and split-LT tables. These allowances cover storage a queued
    /// drop may still hold, not observed co-liveness, so they can reject a
    /// request even when the drop has already run.
    ///
    /// Passing does not mean the proof fits: this checks a partial set of
    /// named requests, not whole-memory admission, and the reservation is
    /// neither a lower bound on the memory a proof needs nor an allocator or
    /// RSS figure. Not reserved: the unpriced batch members' own tables (among
    /// them the output-check and RA claim-reduction kernels', the instruction
    /// read-RAF, and the field-register kernels, stage 2's field-inline claim
    /// reduction included); the RAM read-write and RAF kernels' later binds
    /// and the matrices, conversions and copies they create; the value check's
    /// allocations after its first bind and its `Vec` and `Arc` headers, and
    /// stages 6–8; storage queued for a background drop other than the RAM
    /// value columns above, such as each kernel released at its stage's end,
    /// the eq and split-eq tables the RAM read-write kernel retires and the
    /// register seed entries after the second cycle bind; the other kernels'
    /// split-eq tables (the product remainder's among them) and round
    /// challenges, round messages, streamed-collect extraction buffers and the
    /// row collects that kernels fall back to without random-access rows or
    /// parked `rd` indices; kernel control state such as boxes, maps and
    /// shared handles; uni-skip and field-inline carries, the driver's output
    /// points, stage outputs and the recorder; the caller's program,
    /// preprocessing and PCS setup, commitment and opening state, and
    /// BlindFold; allocator state, retention and the macOS purge's cache-pump
    /// requests; the trace output's final memory image, I/O device and advice
    /// tape; and the tracer, which has already run.
    pub fn check_known_eager<F: JoltField>(
        self,
        config: &ProverConfig,
        trace: &TraceData,
    ) -> Result<(), ProverError<F>> {
        let log_t = config.trace_length.ilog2() as usize;
        let registers = config
            .rw_config
            .register_dimensions(log_t, REGISTER_ADDRESS_BITS);
        let ram = config
            .rw_config
            .ram_dimensions(log_t, config.ram_K.ilog2() as usize);
        let ram_columns = ram_access_columns_bytes(log_t, 1);
        let stage1 =
            jolt_verifier::stage1_batch_sumchecks_members!(batch_peak_bytes F, log_t, registers;);
        let stage2 =
            jolt_verifier::stage2_batch_sumchecks_members!(batch_peak_bytes F, log_t, registers;)
                .saturating_add(ram_read_write::first_bind_bytes::<F>(ram));
        let stage3 =
            jolt_verifier::stage3_sumchecks_members!(batch_peak_bytes F, log_t, registers;)
                .saturating_add(ram_columns);
        let stage4 =
            jolt_verifier::stage4_sumchecks_members!(batch_peak_bytes F, log_t, registers;)
                .saturating_add(ram_val_check::first_bind_bytes::<F>(log_t, config.ram_K));
        let stage5 =
            jolt_verifier::stage5_sumchecks_members!(batch_peak_bytes F, log_t, registers;)
                .saturating_add(ram_columns);
        let required = trace
            .heap_bytes()
            .saturating_add(stage1.max(stage2).max(stage3).max(stage4).max(stage5));
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
    use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
    use jolt_field::{Fr, JoltField};
    use jolt_poly::{BindingOrder, Polynomial};
    #[cfg(feature = "field-inline")]
    use jolt_program::execution::{FieldEvent, FieldInlineTraceData};
    use jolt_program::execution::{JoltProgram, TraceData, TraceOutput};
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
    };
    use jolt_riscv::JoltTraceRow;
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
    #[cfg(feature = "parallel")]
    use rayon::ThreadPoolBuilder;

    use super::MemoryBudget;
    use crate::{ProverConfig, ProverError};

    const LOG_T: usize = 13;
    const CYCLES: usize = 1 << LOG_T;
    /// Rows the fixture's trace buffer reserves beyond its `2^log_t` rows, so
    /// a check that counted rows by length instead of capacity would pass
    /// requests below the trace's actual allocation.
    const RESERVED_ROWS: usize = 1 << 24;
    const TRACE_BYTES: usize = RESERVED_ROWS * size_of::<JoltTraceRow>();

    fn config<F: JoltField>(log_t: usize) -> ProverConfig {
        let cycles = 1 << log_t;
        ProverConfig::from_padded_dimensions::<F>(cycles, 64, cycles).unwrap()
    }

    /// `2^log_t` default rows in a buffer reserved at `RESERVED_ROWS`.
    fn rows(log_t: usize) -> Vec<JoltTraceRow> {
        let mut rows = Vec::with_capacity(RESERVED_ROWS);
        rows.resize(1 << log_t, JoltTraceRow::default());
        rows
    }

    /// A real witness over [`rows`], and its proof shape.
    fn witness<F: JoltField>(log_t: usize) -> (ProverConfig, TraceBackend) {
        let config = config::<F>(log_t);
        let preprocessing = JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing {
                code_size: 32,
                ..Default::default()
            },
            ram: RAMPreprocessing::default(),
            memory_layout: Default::default(),
            max_padded_trace_length: 1 << log_t,
        };
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
            JoltVmWitnessInputs::new(
                &Arc::new(JoltProgram::default()),
                &Arc::new(preprocessing),
                TraceOutput::new(
                    TraceData::new(rows(log_t)),
                    JoltDevice::default(),
                    None,
                    None,
                ),
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
        let (config, witness) = witness::<F>(LOG_T);
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
        let (config, witness) = witness::<F>(LOG_T);
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

    /// Entries the stage-4 increment table's storage holds while the fourth
    /// bind builds the direct generation, as `Polynomial` stores it in this
    /// build: the second bind's `T/4`-entry table after the third bind binds
    /// it once.
    fn retained_increments<F: JoltField>(cycles: usize) -> usize {
        let mut increments = Polynomial::new(vec![F::zero(); cycles / 4]);
        increments.bind_with_order(F::one(), BindingOrder::LowToHigh);
        increments.into_evals().capacity()
    }

    /// The RAM access columns stage 4 may hold over `cycles` cycles: the
    /// session's `u32` address column and two pairs of `u64` value columns,
    /// the value check's own and stage 2's awaiting its background drop.
    fn stage_four_ram_columns(cycles: usize) -> usize {
        cycles * (size_of::<u32>() + 2 * 2 * size_of::<u64>())
    }

    /// The cycle-first stage-4 state over `2^log_t` cycles while the fourth
    /// bind dereferences the indexed generation into direct entries, with
    /// every entry generation at its source bound of three entries per cycle
    /// (`rs1`, `rs2`, `rd`; binding only merges rows): the three 2-byte
    /// register index columns (`rs1`, `rs2`, and `rd` parked for stage 5),
    /// the indexed generation's values and 25-byte packed metadata, the
    /// direct entries, the read LUT at 2^16 values and the write LUT at 2^8,
    /// and the increments' storage; plus the RAM access columns and the trace.
    fn cycle_first_transition_floor<F: JoltField>(
        log_t: usize,
        direct_entry_bytes: usize,
    ) -> usize {
        let (f, cycles) = (size_of::<F>(), 1usize << log_t);
        TRACE_BYTES
            + stage_four_ram_columns(cycles)
            + 3 * 2 * cycles
            + 3 * cycles * (f + 25 + direct_entry_bytes)
            + ((1 << 16) + (1 << 8)) * f
            + retained_increments::<F>(cycles) * f
    }

    /// The address-first stage-4 handoff to the cycle rounds: 32-byte access
    /// rows, a 2048-byte increment checkpoint per 4096 cycles, the 2-byte `rd`
    /// indices parked for stage 5, 128 bound address weights and five
    /// `T`-entry cycle tables; plus the RAM access columns and the trace.
    fn address_first_handoff_floor<F: JoltField>() -> usize {
        let f = size_of::<F>();
        TRACE_BYTES
            + stage_four_ram_columns(CYCLES)
            + 32 * CYCLES
            + CYCLES.div_ceil(4096) * 2048
            + 2 * CYCLES
            + 128 * f
            + 5 * CYCLES * f
    }

    /// Stage 4 reserves the register read/write state of the configured
    /// order. Cycle-first, the fourth bind holds the indexed and direct
    /// generations at once beside the columns preparation built; a request
    /// below that floor fails. Address-first, a request below its own handoff
    /// fails, and one the cycle-first floor rejects passes.
    fn register_read_write_floors<F: JoltField>(direct_entry_bytes: usize) {
        let (mut config, witness) = witness::<F>(LOG_T);
        let trace = &witness.trace.trace;
        let cycle_first = cycle_first_transition_floor::<F>(LOG_T, direct_entry_bytes);
        assert!(
            matches!(
                check::<F>(cycle_first - 1, &config, trace),
                Err(ProverError::MemoryBudgetExceeded { .. })
            ),
            "cycle-first: budget below the fourth-bind floor {cycle_first} was admitted"
        );

        config.rw_config.registers_rw_phase1_num_rounds = 0;
        config.rw_config.registers_rw_phase2_num_rounds =
            u8::try_from(REGISTER_ADDRESS_BITS).unwrap();
        let address_first = address_first_handoff_floor::<F>();
        assert!(address_first < cycle_first);
        assert!(
            matches!(
                check::<F>(address_first - 1, &config, trace),
                Err(ProverError::MemoryBudgetExceeded { .. })
            ),
            "address-first: budget below the handoff floor {address_first} was admitted"
        );
        assert!(
            check::<F>(cycle_first - 1, &config, trace).is_ok(),
            "address-first request priced as cycle-first"
        );
    }

    /// `SparseEntry<Fr, Fr>`: 32 + 3 * 8 + 2 * 32 + 1 bytes, padded to 128.
    #[test]
    fn register_read_write_floors_fr() {
        register_read_write_floors::<Fr>(128);
    }

    /// `SparseEntry<AkitaField, AkitaField>`: 16 + 3 * 8 + 2 * 16 + 1 bytes,
    /// padded to 80.
    #[cfg(feature = "akita")]
    #[test]
    fn register_read_write_floors_akita_field() {
        register_read_write_floors::<AkitaField>(80);
    }

    /// Padded trace sizes this build accepts (at least 2^12 cycles under
    /// Akita) at which the third and fourth increment binds run in place:
    /// `Polynomial`'s parallel path copies only tables of at least 2^11
    /// entries.
    const IN_PLACE_LOG_TS: &[usize] = if cfg!(feature = "akita") {
        &[12]
    } else {
        &[10, 12]
    };

    /// At these sizes the increment table keeps the second bind's `T/4`-entry
    /// allocation while the fourth bind builds the direct generation, so a
    /// request below the floor holding that observed storage must fail.
    fn rejects_below_the_retained_increment_floor<F: JoltField>(direct_entry_bytes: usize) {
        for &log_t in IN_PLACE_LOG_TS {
            let (config, witness) = witness::<F>(log_t);
            let floor = cycle_first_transition_floor::<F>(log_t, direct_entry_bytes);
            assert!(
                matches!(
                    check::<F>(floor - 1, &config, &witness.trace.trace),
                    Err(ProverError::MemoryBudgetExceeded { .. })
                ),
                "2^{log_t} cycles: budget below the floor {floor} with {} retained increment \
                 entries was admitted",
                retained_increments::<F>(1 << log_t)
            );
        }
    }

    #[test]
    fn retained_increment_floor_fr() {
        rejects_below_the_retained_increment_floor::<Fr>(128);
    }

    #[cfg(feature = "akita")]
    #[test]
    fn retained_increment_floor_akita_field() {
        rejects_below_the_retained_increment_floor::<AkitaField>(80);
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
        let trace = TraceData::from_parts(rows(LOG_T), events).unwrap();
        let floor =
            TRACE_BYTES + FIELD_EVENTS * size_of::<FieldEvent>() + stage_three_floor::<Fr>();
        assert!(matches!(
            check::<Fr>(floor - 1, &config::<Fr>(LOG_T), &trace),
            Err(ProverError::MemoryBudgetExceeded { .. })
        ));
    }

    /// A budget at the reported requirement passes and one byte less fails
    /// with the same requirement; an unbounded budget passes.
    #[test]
    fn budget_at_the_reported_requirement_passes() {
        let (config, witness) = witness::<Fr>(LOG_T);
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

    /// Stage 4 reserves the RAM value check's tables through its first bind
    /// beside the register read/write peak, and its RAM access columns once.
    /// Over `2^13` cycles, `2^12` addresses and exactly `2^13` reserved rows,
    /// `interior` lies midway between the requirement without those tables
    /// and the one with them, and `above` midway between that and a
    /// requirement charging the access columns twice. These requirements
    /// hold below the few hundred Rayon threads at which the register
    /// round-1 scratch overtakes its fourth bind.
    fn reserves_the_value_check_first_bind<F: JoltField>(interior: usize, above: usize) {
        let config = ProverConfig::from_padded_dimensions::<F>(CYCLES, 1 << 12, CYCLES).unwrap();
        let trace = TraceData::new(vec![JoltTraceRow::default(); CYCLES]);
        assert_eq!(trace.heap_bytes(), CYCLES * 64);
        assert!(
            matches!(
                check::<F>(interior, &config, &trace),
                Err(ProverError::MemoryBudgetExceeded { .. })
            ),
            "budget {interior} without the value check's first-bind tables was admitted"
        );
        assert!(
            check::<F>(above, &config, &trace).is_ok(),
            "budget {above} above the reservation was rejected"
        );
    }

    /// 7,585,936 bytes without the tables, 7,989,392 with them and 8,284,304
    /// with the access columns twice.
    #[test]
    fn value_check_first_bind_fr() {
        reserves_the_value_check_first_bind::<Fr>(7_787_664, 8_136_848);
    }

    /// 4,927,632, 5,129,360 and 5,424,272 bytes.
    #[cfg(feature = "akita")]
    #[test]
    fn value_check_first_bind_akita_field() {
        reserves_the_value_check_first_bind::<AkitaField>(5_028_496, 5_276_816);
    }

    /// Stage 2 prices the instruction claim reduction after its preparation
    /// beside the RAM read-write kernel's preparation and first bind, never
    /// the reduction's construction collector beside that bind: every member
    /// prepares before any member binds. Over default rows at exact capacity
    /// and a fixed 16-thread pool, `below` must be rejected and `above`
    /// admitted.
    #[cfg(feature = "parallel")]
    fn pairs_stage_two_phases<F: JoltField>(config: &ProverConfig, below: usize, above: usize) {
        let cycles = config.trace_length;
        let trace = TraceData::new(vec![JoltTraceRow::default(); cycles]);
        // The rows at exact capacity and no field-event capacity.
        assert_eq!(trace.heap_bytes(), 64 * cycles);
        ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap()
            .install(|| {
                assert!(
                    matches!(
                        check::<F>(below, config, &trace),
                        Err(ProverError::MemoryBudgetExceeded { .. })
                    ),
                    "budget {below} below the stage-2 reservation was admitted"
                );
                assert!(
                    check::<F>(above, config, &trace).is_ok(),
                    "budget {above} above the stage-2 reservation was rejected"
                );
            });
    }

    /// Cycle-first RAM over 2^17 cycles and 2^16 addresses: the check
    /// requires 98,110,024 bytes, stage 2 its largest envelope. 92 MiB lies
    /// below that and above the 96,126,128 bytes required without the
    /// reduction's post-preparation storage; 94 MiB lies above it and below
    /// the 119,571,608 bytes required with the reduction's construction
    /// collector beside the first bind instead.
    #[cfg(feature = "parallel")]
    #[test]
    fn stage_two_phase_pairing_cycle_first_fr() {
        let config = ProverConfig::from_padded_dimensions::<Fr>(1 << 17, 1 << 16, 1 << 17).unwrap();
        pairs_stage_two_phases::<Fr>(&config, 92 << 20, 94 << 20);
    }

    /// Address-first RAM over 2^13 cycles and 2^20 addresses, registers
    /// still cycle-first: the check requires 60,564,688 bytes. 56 MiB lies
    /// below that and above the 56,042,704 bytes required without the first
    /// bind's half-length `K`-entry table copy; 58 MiB lies above it and below
    /// the 61,395,456 bytes required with the reduction's construction
    /// collector beside the first bind.
    #[cfg(all(feature = "parallel", feature = "akita"))]
    #[test]
    fn stage_two_phase_pairing_address_first_akita_field() {
        let mut config =
            ProverConfig::from_padded_dimensions::<AkitaField>(1 << 13, 1 << 20, 1 << 13).unwrap();
        config.rw_config.ram_rw_phase1_num_rounds = 0;
        config.rw_config.ram_rw_phase2_num_rounds = 20;
        pairs_stage_two_phases::<AkitaField>(&config, 56 << 20, 58 << 20);
    }
}
