//! The explicit per-proof memory budget: an operational request beside the
//! proof shape ([`ProverConfig`]). It changes no dimension, schedule, kernel,
//! transcript or PCS choice; [`MemoryBudget::check_known_eager`] only rejects
//! a request below the optimized prover's known eager reservation, after
//! tracing and before proving starts.

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_field::JoltField;
use jolt_kernels::optimized::instruction_claim_reduction::OptimizedInstructionClaimReduction;
use jolt_kernels::optimized::instruction_input::OptimizedInstructionInput;
use jolt_kernels::optimized::ram_read_write::RamReadWriteBytes;
use jolt_kernels::optimized::ram_val_check;
use jolt_kernels::optimized::registers_claim_reduction::OptimizedRegistersClaimReduction;
use jolt_kernels::optimized::registers_read_write::OptimizedRegistersReadWrite;
use jolt_kernels::optimized::registers_val_evaluation::OptimizedRegistersValEvaluation;
use jolt_kernels::optimized::spartan_outer::OptimizedOuterRemainder;
use jolt_kernels::optimized::spartan_product::OptimizedProductRemainder;
use jolt_kernels::optimized::spartan_shift::OptimizedSpartanShift;
use jolt_kernels::optimized::{ram_access_columns_bytes, ram_value_collection_bytes};
use jolt_program::execution::TraceData;

use crate::{ProverConfig, ProverError};

/// One batch member's priced bytes under the kernel
/// [`JoltBackend::optimized`](crate::JoltBackend::optimized) installs in its
/// slot, over `$prefix` cycles before the trace's constant suffix, or `None`
/// past `usize::MAX`: the kernel's own peak unless the arm's comment says
/// otherwise. Every member of the priced batches has an arm, so a member
/// added to one of them fails to compile here until it is classified.
macro_rules! member_peak_bytes {
    ($F:ty, $log_t:ident, $prefix:ident, $registers:ident, outer_remainder) => {
        OptimizedOuterRemainder::peak_bytes::<$F>($prefix)
    };
    ($F:ty, $log_t:ident, $prefix:ident, $registers:ident, product_remainder) => {
        OptimizedProductRemainder::peak_bytes::<$F>($prefix)
    };
    ($F:ty, $log_t:ident, $prefix:ident, $registers:ident, shift) => {
        OptimizedSpartanShift::peak_bytes::<$F>($log_t)
    };
    ($F:ty, $log_t:ident, $prefix:ident, $registers:ident, instruction_input) => {
        OptimizedInstructionInput::peak_bytes::<$F>($prefix)
    };
    ($F:ty, $log_t:ident, $prefix:ident, $registers:ident, registers_claim_reduction) => {
        OptimizedRegistersClaimReduction::peak_bytes::<$F>($log_t)
    };
    ($F:ty, $log_t:ident, $prefix:ident, $registers:ident, registers_read_write) => {
        OptimizedRegistersReadWrite::peak_bytes::<$F>($registers, $prefix)
    };
    ($F:ty, $log_t:ident, $prefix:ident, $registers:ident, registers_val_evaluation) => {
        OptimizedRegistersValEvaluation::peak_bytes::<$F>($log_t, $prefix)
    };
    // Priced with RAM by phase in stage 2's envelope (`stage_reservations`).
    ($F:ty, $log_t:ident, $prefix:ident, $registers:ident, instruction_claim_reduction) => {
        Some(0)
    };
    // Priced together in stage 2's envelope from the read-write kernel's
    // preparation through its first bind (`RamReadWriteBytes`); their later
    // states are not priced.
    ($F:ty, $log_t:ident, $prefix:ident, $registers:ident, ram_read_write) => {
        Some(0)
    };
    ($F:ty, $log_t:ident, $prefix:ident, $registers:ident, ram_raf_evaluation) => {
        Some(0)
    };
    // Priced in stages 4 and 5 at its largest window, the RAM access columns
    // included (`ram_val_check::scalar_window_bytes`).
    ($F:ty, $log_t:ident, $prefix:ident, $registers:ident, ram_val_check) => {
        Some(0)
    };
    // Not priced yet: the reservation omits these members' own allocations
    // (the RAM access columns some of them hold are added per stage).
    ($F:ty, $log_t:ident, $prefix:ident, $registers:ident, field_registers_claim_reduction) => {
        Some(0)
    };
    ($F:ty, $log_t:ident, $prefix:ident, $registers:ident, ram_output_check) => {
        Some(0)
    };
    ($F:ty, $log_t:ident, $prefix:ident, $registers:ident, field_registers_read_write) => {
        Some(0)
    };
    ($F:ty, $log_t:ident, $prefix:ident, $registers:ident, instruction_read_raf) => {
        Some(0)
    };
    ($F:ty, $log_t:ident, $prefix:ident, $registers:ident, ram_ra_claim_reduction) => {
        Some(0)
    };
    ($F:ty, $log_t:ident, $prefix:ident, $registers:ident, field_registers_val_evaluation) => {
        Some(0)
    };
}

/// One stage batch's priced bytes, folded over the verifier's member list, or
/// `None` past `usize::MAX`: the sum of the members' priced bytes. Where each
/// is a kernel's own peak, the sum bounds what those members hold at once.
macro_rules! batch_peak_bytes {
    (
        $F:ty, $log_t:ident, $prefix:ident, $registers:ident;
        batch = $batch:ident,
        label = $label:literal,
        aggregates = $aggregates:tt,
        shape = $shape:ident,
        members = [$({ name: $name:ident, $($member:tt)* },)+]
    ) => {
        [$(member_peak_bytes!($F, $log_t, $prefix, $registers, $name)),+]
            .into_iter()
            .try_fold(0usize, |sum, member| sum.checked_add(member?))
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

    /// Rejects this budget when it is below the bytes reserved for a named
    /// subset of the eager allocations of an optimized proof of `trace` under
    /// `config`, which must be the canonical configuration derived from
    /// `trace`; the check does not validate it. A reservation within
    /// `usize::MAX` above the budget fails with
    /// [`ProverError::MemoryBudgetExceeded`]; one past `usize::MAX` fails
    /// every budget with [`ProverError::MemoryReservationOverflow`]. The SDK
    /// and profile callers run it after tracing and before building the
    /// witness.
    ///
    /// The reservation is the trace buffers at their reserved capacity
    /// ([`TraceData::heap_bytes`]) plus the largest stage-1–5 envelope over
    /// the trace's proof rows ([`TraceData::proof_len`]), the cycles before
    /// its constant padding suffix, which the optimized kernels store
    /// explicitly over random-access rows: the sum of the batch members'
    /// priced bytes, each the member kernel's own `peak_bytes` unless a stage
    /// term below prices it, and one RAM term in each of stages 2–5: the RAM
    /// read-write and RAF kernels from preparation through the read-write
    /// kernel's first bind in stage 2 ([`RamReadWriteBytes`]), the RAM access
    /// columns that may still be held in stage 3
    /// ([`ram_access_columns_bytes`]), and the value check's scalar tables
    /// beside the access columns at their largest in stages 4 and 5
    /// ([`ram_val_check::scalar_window_bytes`]). Stage 4 prices the register
    /// read/write kernel of the configured phase order with the `rd` indices
    /// it parks, stage 5 the value evaluation that takes them over. Summing
    /// peaks exceeds the batch's true simultaneous maximum by at most the
    /// members' transients that do not coincide; in stage 4 the register peak
    /// and the value check's largest window need not be live at once.
    ///
    /// Stage 2 pairs the instruction claim reduction, whose combined table
    /// keeps all `T` cycles, with RAM by phase. The driver prepares every
    /// member before any member binds, and a preparation frees its
    /// collector's partial vectors before it returns, while no queued
    /// background drop is assumed to have run. Stage 2 therefore reserves the
    /// larger of the reduction's construction collect
    /// ([`OptimizedInstructionClaimReduction::construction_bytes`]) beside
    /// RAM's tables through preparation and its post-preparation window
    /// ([`OptimizedInstructionClaimReduction::post_prepare_peak_bytes`])
    /// beside RAM's first bind.
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
    /// Stage 3 therefore still reserves that generation, and so does every
    /// window of the value check, whose stage-4 windows also cover the second
    /// pair of value columns it collects and drops at its second bind. Under
    /// `parallel` the check queues its last address table for a background
    /// drop at the fourth bind, and its remaining tables with the kernel at
    /// the stage's end, so stage 5 reserves the same largest window beside its
    /// own members. Stages 2 and 4 also reserve what their collection holds
    /// until it truncates ([`ram_value_collection_bytes`]). These allowances
    /// cover storage a queued drop may still hold, not observed co-liveness,
    /// so they can reject a request even when the drop has already run.
    ///
    /// Passing does not mean the proof fits, and the reservation is neither a
    /// lower bound on the memory a proof needs nor an allocator or RSS figure.
    /// Not reserved: the unpriced batch members' own tables (among them the
    /// output-check and RA claim-reduction kernels', the instruction read-RAF,
    /// and the field-register kernels, stage 2's field-inline claim reduction
    /// included); the RAM read-write and RAF kernels' later binds and the
    /// matrices, conversions and copies they create, and stages 6–8; storage
    /// queued for a background drop other than the RAM value columns and the
    /// value check's tables above, such as each other kernel released at its
    /// stage's end, the eq and split-eq tables the RAM read-write kernel
    /// retires and the register seed entries after the second cycle bind;
    /// split-eq tables and the other eq factor tables kernels build from their
    /// points (about `√T` entries each, or `√(K·T)` for the register operand
    /// claims) unless a price above names them; round challenges, round
    /// messages, streamed-collect extraction buffers and the row collects
    /// that kernels fall back to without random-access rows, which keep every
    /// cycle, or without parked `rd` indices; kernel control state such as
    /// boxes, maps, `Vec` and `Arc` headers, Rayon jobs and shared handles;
    /// uni-skip and field-inline carries, stage outputs and the recorder;
    /// preprocessing and PCS setup, commitment and opening state, and
    /// BlindFold; allocator state, retention and the macOS purge's cache-pump
    /// requests; the trace output's final memory image, I/O device and advice
    /// tape; and the tracer, which has already run.
    pub fn check_known_eager<F: JoltField>(
        self,
        config: &ProverConfig,
        trace: &TraceData,
    ) -> Result<(), ProverError<F>> {
        self.check_reservation(
            config,
            trace.proof_len().min(config.trace_length),
            trace.heap_bytes(),
        )
    }

    /// [`Self::check_known_eager`] over `prefix_cycles` cycles before the
    /// trace's constant suffix and `trace_bytes` of trace buffers.
    fn check_reservation<F: JoltField>(
        self,
        config: &ProverConfig,
        prefix_cycles: usize,
        trace_bytes: usize,
    ) -> Result<(), ProverError<F>> {
        let required = stage_reservations::<F>(config, prefix_cycles)
            .and_then(|stages| trace_bytes.checked_add(stages.into_iter().max()?));
        match required {
            Some(required) if self.bytes >= required => Ok(()),
            Some(required) => Err(ProverError::MemoryBudgetExceeded {
                budget: self.bytes,
                required,
            }),
            None => Err(ProverError::MemoryReservationOverflow { budget: self.bytes }),
        }
    }
}

/// The stage-1–5 envelopes [`MemoryBudget::check_known_eager`] reserves beside
/// the trace for `config` over `prefix_cycles` cycles before the trace's
/// constant suffix, or `None` past `usize::MAX`.
fn stage_reservations<F: JoltField>(
    config: &ProverConfig,
    prefix_cycles: usize,
) -> Option<[usize; 5]> {
    let log_t = config.trace_length.ilog2() as usize;
    let registers = config
        .rw_config
        .register_dimensions(log_t, REGISTER_ADDRESS_BITS);
    let ram = RamReadWriteBytes::new::<F>(
        config
            .rw_config
            .ram_dimensions(log_t, config.ram_K.ilog2() as usize),
        prefix_cycles,
    )?;
    let instruction_claims = OptimizedInstructionClaimReduction::construction_bytes::<F>(log_t)?
        .checked_add(ram.prepared)?
        .max(
            OptimizedInstructionClaimReduction::post_prepare_peak_bytes::<F>(log_t)?
                .checked_add(ram.first_bind)?,
        );
    let value_check = ram_val_check::scalar_window_bytes::<F>(log_t, config.ram_K, prefix_cycles)?;
    let collected = ram_value_collection_bytes(prefix_cycles)?;
    Some([
        jolt_verifier::stage1_batch_sumchecks_members!(batch_peak_bytes F, log_t, prefix_cycles, registers;)?,
        jolt_verifier::stage2_batch_sumchecks_members!(batch_peak_bytes F, log_t, prefix_cycles, registers;)?
            .checked_add(instruction_claims)?
            .checked_add(collected)?,
        jolt_verifier::stage3_sumchecks_members!(batch_peak_bytes F, log_t, prefix_cycles, registers;)?
            .checked_add(ram_access_columns_bytes(prefix_cycles, 1)?)?,
        jolt_verifier::stage4_sumchecks_members!(batch_peak_bytes F, log_t, prefix_cycles, registers;)?
            .checked_add(value_check)?
            .checked_add(collected)?,
        jolt_verifier::stage5_sumchecks_members!(batch_peak_bytes F, log_t, prefix_cycles, registers;)?
            .checked_add(value_check)?,
    ])
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::panic, reason = "test module")]
mod tests {
    #[cfg(feature = "akita")]
    use jolt_akita::AkitaField;
    use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
    use jolt_field::{Fr, JoltField};
    use jolt_kernels::optimized::{ram_val_check, ram_value_collection_bytes};
    use jolt_program::execution::TraceData;
    #[cfg(feature = "field-inline")]
    use jolt_program::execution::{FieldEvent, FieldInlineTraceData};
    use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, JoltTraceRow, RamAccess};

    use super::{stage_reservations, MemoryBudget};
    use crate::{ProverConfig, ProverError};

    const LOG_T: usize = 13;
    const CYCLES: usize = 1 << LOG_T;
    /// Cycles before the constant suffix, in the stage floors and the trace.
    const PREFIX: usize = 1 << 12;
    /// Rows the fixture's trace buffer reserves beyond its `2^LOG_T` rows, so
    /// a check that counted rows by length instead of capacity would pass
    /// requests below the trace's actual allocation.
    const RESERVED_ROWS: usize = 1 << 14;
    const TRACE_BYTES: usize = RESERVED_ROWS * size_of::<JoltTraceRow>();
    const RAM_K: usize = 64;

    fn config<F: JoltField>(log_t: usize) -> ProverConfig {
        let cycles = 1 << log_t;
        ProverConfig::from_padded_dimensions::<F>(cycles, RAM_K, cycles).unwrap()
    }

    fn stages<F: JoltField>(config: &ProverConfig) -> [usize; 5] {
        stage_reservations::<F>(config, PREFIX).unwrap()
    }

    /// `PREFIX` jump rows, then canonical padding to `2^LOG_T` rows, in a
    /// buffer reserved at `RESERVED_ROWS`.
    fn rows() -> Vec<JoltTraceRow> {
        let jump = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::JAL,
            ..Default::default()
        };
        let mut rows = Vec::with_capacity(RESERVED_ROWS);
        rows.resize(
            PREFIX,
            JoltTraceRow::new(jump, Default::default(), RamAccess::NoOp, 1).unwrap(),
        );
        rows.resize(CYCLES, JoltTraceRow::default());
        rows
    }

    fn check<F: JoltField>(
        budget: usize,
        config: &ProverConfig,
        trace: &TraceData,
    ) -> Result<(), ProverError<F>> {
        MemoryBudget::from_bytes(budget).check_known_eager::<F>(config, trace)
    }

    /// S1-01 regression, on stage 1's own reservation so that a larger stage
    /// cannot mask it: the stage-1 outer remainder allocates `Az` and `Bz`,
    /// two tables of `2 · PREFIX` field elements.
    fn stage_one_reserves_its_tables<F: JoltField>() {
        let floor = 2 * (2 * PREFIX) * size_of::<F>();
        let stage1 = stages::<F>(&config::<F>(LOG_T))[0];
        assert!(
            stage1 >= floor,
            "stage 1 reserves {stage1} bytes, below its {floor}-byte tables"
        );
    }

    #[test]
    fn s1_01_stage_one_floor_fr() {
        stage_one_reserves_its_tables::<Fr>();
    }

    #[cfg(feature = "akita")]
    #[test]
    fn s1_01_stage_one_floor_akita_field() {
        stage_one_reserves_its_tables::<AkitaField>();
    }

    /// The stage-3 state after the first bind: the instruction input's four
    /// operand tables and packed flag words at `PREFIX / 2` entries each and
    /// its four-value flag table, beside the shift's eight prefix–suffix
    /// tables and the register claim reduction's two, `2^ceil(log_t/2)`
    /// entries each. The session keeps the `u32` RAM address column, and the
    /// two `u64` value columns stage 2 queued for a background drop stay
    /// reserved as an outstanding generation, `PREFIX` entries each when the
    /// last proof row accesses RAM.
    fn stage_three_reserves_its_state<F: JoltField>() {
        let (half, prefix) = (PREFIX / 2, 1usize << LOG_T.div_ceil(2));
        let floor = (4 * half + 4 + 8 * prefix + 2 * prefix) * size_of::<F>()
            + half * size_of::<u64>()
            + PREFIX * (size_of::<u32>() + 2 * size_of::<u64>());
        let stage3 = stages::<F>(&config::<F>(LOG_T))[2];
        assert!(
            stage3 >= floor,
            "stage 3 reserves {stage3} bytes, below its {floor}-byte state"
        );
    }

    #[test]
    fn stage_three_simultaneous_state_fr() {
        stage_three_reserves_its_state::<Fr>();
    }

    #[cfg(feature = "akita")]
    #[test]
    fn stage_three_simultaneous_state_akita_field() {
        stage_three_reserves_its_state::<AkitaField>();
    }

    /// Stage 4's RAM term as the check prices it: the value check's largest
    /// window with the access columns, and their collection. With the access
    /// columns alone instead, the floors below would leave the check's scalar
    /// tables, more than the register increments' storage, as slack.
    fn stage_four_ram<F: JoltField>() -> usize {
        ram_val_check::scalar_window_bytes::<F>(LOG_T, RAM_K, PREFIX).unwrap()
            + ram_value_collection_bytes(PREFIX).unwrap()
    }

    /// The cycle-first stage-4 state while the fourth bind dereferences the
    /// indexed generation into direct entries, with every entry generation
    /// at its source bound of three entries per cycle (`rs1`, `rs2`, `rd`;
    /// binding only merges rows): the three 2-byte register index columns
    /// (`rs1`, `rs2`, and `rd` parked for stage 5), the indexed generation's
    /// values and 25-byte packed metadata, the direct entries, the read LUT
    /// at 2^16 values and the write LUT at 2^8, and the increments'
    /// `PREFIX / 4`-entry allocation, which they keep because a prefix of
    /// 1024 entries binds in place; plus stage 4's RAM term.
    fn cycle_first_transition_floor<F: JoltField>(direct_entry_bytes: usize) -> usize {
        let f = size_of::<F>();
        stage_four_ram::<F>()
            + 3 * 2 * PREFIX
            + 3 * PREFIX * (f + 25 + direct_entry_bytes)
            + ((1 << 16) + (1 << 8)) * f
            + PREFIX / 4 * f
    }

    /// The address-first stage-4 handoff to the cycle rounds: 32-byte access
    /// rows, a 2048-byte increment checkpoint per 4096 cycles and the
    /// 2048-byte final register values, the 2-byte `rd` indices parked for
    /// stage 5, 128 bound address weights and five `PREFIX`-entry cycle
    /// tables; plus stage 4's RAM term.
    fn address_first_handoff_floor<F: JoltField>() -> usize {
        let f = size_of::<F>();
        stage_four_ram::<F>()
            + 32 * PREFIX
            + PREFIX.div_ceil(4096) * 2048
            + 2048
            + 2 * PREFIX
            + 128 * f
            + 5 * PREFIX * f
    }

    /// Stage 4 reserves the register read/write state of the configured
    /// order. Cycle-first, the fourth bind holds the indexed and direct
    /// generations at once beside the columns preparation built.
    /// Address-first, stage 4 covers its own handoff and stays below the
    /// cycle-first floor.
    fn register_read_write_floors<F: JoltField>(direct_entry_bytes: usize) {
        let mut config = config::<F>(LOG_T);
        let cycle_first = cycle_first_transition_floor::<F>(direct_entry_bytes);
        let stage4 = stages::<F>(&config)[3];
        assert!(
            stage4 >= cycle_first,
            "cycle-first: stage 4 reserves {stage4} bytes, below the fourth-bind floor {cycle_first}"
        );

        config.rw_config.registers_rw_phase1_num_rounds = 0;
        config.rw_config.registers_rw_phase2_num_rounds =
            u8::try_from(REGISTER_ADDRESS_BITS).unwrap();
        let address_first = address_first_handoff_floor::<F>();
        let stage4 = stages::<F>(&config)[3];
        assert!(
            stage4 >= address_first,
            "address-first: stage 4 reserves {stage4} bytes, below the handoff floor {address_first}"
        );
        assert!(
            stage4 < cycle_first,
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

    /// Stages 4 and 5 both reserve the value check's largest window, stage 5
    /// for the tables the check queues with its kernel at stage 4's end. Over
    /// 2^20 addresses its preparation alone holds the `K`-entry address eq
    /// table and a one-entry tail beside the `u32` address column and two
    /// generations of `u64` value columns, more than either stage's other
    /// terms, so a stage without the window falls below this floor.
    fn value_check_window_floor<F: JoltField>() {
        let addresses = 1 << 20;
        let config = ProverConfig::from_padded_dimensions::<F>(CYCLES, addresses, CYCLES).unwrap();
        let floor = PREFIX * (size_of::<u32>() + 2 * 2 * size_of::<u64>())
            + (addresses + 1) * size_of::<F>();
        let [.., stage4, stage5] = stages::<F>(&config);
        assert!(
            stage4 >= floor,
            "stage 4 reserves {stage4} bytes, below the value check's {floor}-byte window"
        );
        assert!(
            stage5 >= floor,
            "stage 5 reserves {stage5} bytes, below the value check's {floor}-byte window"
        );
    }

    #[test]
    fn value_check_window_in_stages_four_and_five_fr() {
        value_check_window_floor::<Fr>();
    }

    #[cfg(feature = "akita")]
    #[test]
    fn value_check_window_in_stages_four_and_five_akita_field() {
        value_check_window_floor::<AkitaField>();
    }

    /// Stage 2 pairs the instruction claim reduction with RAM by phase: its
    /// construction collect beside RAM's tables through preparation, its
    /// post-preparation window beside RAM's first bind, never the collect
    /// beside the bind. Cycle-first over `PREFIX` proof rows and 2^13 cycles
    /// the post-preparation window leads: stage 2 reserves 2,990,136 bytes,
    /// above the 2,685,928 without the reduction's post-preparation storage
    /// and below the 4,324,328 with its construction collect beside the first
    /// bind. Over 2^16 cycles the collect leads: 13,777,800 bytes, above the
    /// 5,087,720 without the collect and below the 15,809,680 with it beside
    /// the first bind.
    #[test]
    fn stage_two_phase_pairing_cycle_first_fr() {
        let post_leads = stages::<Fr>(&config::<Fr>(13))[1];
        assert!(
            (2_800_000..4_000_000).contains(&post_leads),
            "2^13 cycles: stage 2 reserves {post_leads} bytes"
        );
        let collect_leads = stages::<Fr>(&config::<Fr>(16))[1];
        assert!(
            (13 << 20..14 << 20).contains(&collect_leads),
            "2^16 cycles: stage 2 reserves {collect_leads} bytes"
        );
    }

    /// Address-first RAM over 2^16 cycles, `PREFIX` proof rows and 64
    /// addresses: stage 2 reserves 8,912,536 bytes, above the 8,257,176 with
    /// the order priced cycle-first and the 3,814,120 without the reduction's
    /// construction collect, and below the 10,485,616 with that collect beside
    /// the first bind.
    #[cfg(feature = "akita")]
    #[test]
    fn stage_two_phase_pairing_address_first_akita_field() {
        let mut config = config::<AkitaField>(16);
        config.rw_config.ram_rw_phase1_num_rounds = 0;
        config.rw_config.ram_rw_phase2_num_rounds = 6;
        let stage2 = stages::<AkitaField>(&config)[1];
        assert!(
            (8 << 20..9 << 20).contains(&stage2),
            "address-first: stage 2 reserves {stage2} bytes"
        );
    }

    /// Field events count at their reserved capacity beside the rows.
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
        let config = config::<Fr>(LOG_T);
        let trace = TraceData::from_parts(rows(), events).unwrap();
        let required = TRACE_BYTES
            + FIELD_EVENTS * size_of::<FieldEvent>()
            + stages::<Fr>(&config).into_iter().max().unwrap();
        assert!(matches!(
            check::<Fr>(required - 1, &config, &trace),
            Err(ProverError::MemoryBudgetExceeded { required: reported, .. }) if reported == required
        ));
    }

    /// The requirement is the trace buffer at its reserved capacity plus the
    /// largest stage reservation over the trace's proof rows. A budget at it
    /// passes and one byte less fails with the same requirement; an
    /// unbounded budget passes.
    #[test]
    fn budget_at_the_reported_requirement_passes() {
        let (config, trace) = (config::<Fr>(LOG_T), TraceData::new(rows()));
        assert_eq!(trace.proof_len(), PREFIX);
        assert!(check::<Fr>(usize::MAX, &config, &trace).is_ok());
        let Err(ProverError::MemoryBudgetExceeded {
            budget: 0,
            required,
        }) = check::<Fr>(0, &config, &trace)
        else {
            panic!("an empty budget passed");
        };
        assert_eq!(
            required,
            TRACE_BYTES + stages::<Fr>(&config).into_iter().max().unwrap()
        );
        assert!(check::<Fr>(required, &config, &trace).is_ok());
        assert!(matches!(
            check::<Fr>(required - 1, &config, &trace),
            Err(ProverError::MemoryBudgetExceeded { required: same, .. }) if same == required
        ));
    }

    /// The address-space boundary, allocation-free: over `2^(usize::BITS - 7)`
    /// cycles (2^25 on wasm32) all before the constant suffix, stage 1's `Az`
    /// and `Bz` alone hold `2 · 2^(BITS - 6)` Fr elements, `2^BITS` bytes. That
    /// reservation exceeds `usize::MAX`, so even a `usize::MAX` budget fails.
    /// Stage 2's instruction claim reduction keeps all `T` cycles at any
    /// prefix, so the representable case shrinks the trace to
    /// `2^(BITS - 13)` cycles; it passes at exactly the reported requirement.
    #[test]
    fn reservation_past_the_address_space_rejects_every_budget() {
        let check = |log_t: usize, budget: usize, prefix_cycles: usize| {
            MemoryBudget::from_bytes(budget).check_reservation::<Fr>(
                &config::<Fr>(log_t),
                prefix_cycles,
                0,
            )
        };
        let log_t = usize::BITS as usize - 7;
        assert!(matches!(
            check(log_t, usize::MAX, 1 << log_t),
            Err(ProverError::MemoryReservationOverflow { budget: usize::MAX })
        ));
        let log_t = usize::BITS as usize - 13;
        let prefix_cycles = 1 << (log_t - 13);
        let Err(ProverError::MemoryBudgetExceeded { required, .. }) =
            check(log_t, 0, prefix_cycles)
        else {
            panic!("an empty budget passed");
        };
        assert!(check(log_t, required, prefix_cycles).is_ok());
    }
}
