//! Per-proof configuration, derived from the execution trace.
//!
//! The five proof-shape values are exactly the proof's wire config block
//! (`JoltProof::{trace_length, ram_K, rw_config, one_hot_config,
//! trace_polynomial_order}`) plus the Fiat-Shamir preamble inputs. Akita's
//! witness chunk profile is a setup choice carried by its verifier setup.
//! The proof-shape derivation policies must match the verifier's choices byte-for-byte.

use common::constants::{ONEHOT_CHUNK_THRESHOLD_LOG_T, REGISTER_COUNT, XLEN};
use common::jolt_device::MemoryLayout;
#[cfg(feature = "akita")]
use jolt_akita::AkitaChunkProfile;
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig, TracePolynomialOrder};
use jolt_field::JoltField;
use jolt_program::execution::{RamAccess, TraceRow};
use jolt_riscv::{CircuitFlags, JoltTraceRow};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::ProverError;

const LOOKUP_ADDRESS_BITS: usize = 2 * XLEN;
#[cfg(feature = "parallel")]
const PARALLEL_DERIVE_MIN_ROWS: usize = 1 << 16;

/// The minimum padded trace length — the compiled protocol's PCS floor
/// (legacy's `PCS::MIN_PADDED_TRACE_LENGTH`). Dory needs `T >= K^(1/D)`
/// (256); Akita's folded-only protocol cannot schedule the K=16
/// `OneHotTrace` group below 16 variables, and column arity is
/// `log_k_chunk + log_T`, so the Akita pipeline pads every trace to at
/// least 2^12 cycles.
#[cfg(not(feature = "akita"))]
const MIN_PADDED_TRACE_LENGTH: usize = 256;
#[cfg(feature = "akita")]
const MIN_PADDED_TRACE_LENGTH: usize = 1 << 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[expect(non_snake_case)]
pub struct ProverConfig {
    /// Padded trace length (a power of two, at least 256).
    pub trace_length: usize,
    /// RAM address-space size (a power of two).
    pub ram_K: usize,
    pub rw_config: JoltReadWriteConfig,
    pub one_hot_config: JoltOneHotConfig,
    /// Coefficient placement of the trace polynomials in the commitment
    /// matrix. [`ProverConfig::derive`] always picks cycle-major (legacy has
    /// no production selection logic); address-major is chosen by
    /// overwriting this field after derivation. Dory committed-program
    /// preprocessing bakes this order into its chunk commitments, so pass it
    /// to `preprocess_committed_with_order` and keep the values equal. Akita
    /// supports only cycle-major order.
    pub trace_polynomial_order: TracePolynomialOrder,
    /// Selects the Akita witness chunk profile during preprocessing. Proving
    /// rejects a profile that differs from the prepared setup. The verifier
    /// setup carries this choice; it is not part of the proof's wire config block.
    #[cfg(feature = "akita")]
    pub akita_chunk_profile: AkitaChunkProfile,
}

impl ProverConfig {
    /// Derive the proof shape from an unpadded trace: pad the length (minimum
    /// 256 so `T >= K^(1/D)`, else next power of two past the trace plus its
    /// final no-op), size RAM to the highest touched (remapped) address or the
    /// program image extent, and pick the chunking policies from `log_T`.
    #[tracing::instrument(skip_all, name = "ProverConfig::derive", fields(rows = rows.len()))]
    pub fn derive<F: JoltField>(
        rows: &[TraceRow],
        memory_layout: &MemoryLayout,
        min_bytecode_address: u64,
        program_image_len_words: usize,
        max_padded_trace_length: usize,
    ) -> Result<Self, ProverError<F>> {
        Self::derive_from_rows(
            rows,
            memory_layout,
            min_bytecode_address,
            program_image_len_words,
            max_padded_trace_length,
            |row| match row.ram_access() {
                RamAccess::Read(read) => Some(read.address),
                RamAccess::Write(write) => Some(write.address),
                RamAccess::NoOp => None,
            },
            |row| row.circuit_flags().get(CircuitFlags::Jump),
        )
    }

    #[tracing::instrument(
        skip_all,
        name = "ProverConfig::derive_compact",
        fields(rows = rows.len())
    )]
    pub fn derive_compact<F: JoltField>(
        rows: &[JoltTraceRow],
        memory_layout: &MemoryLayout,
        min_bytecode_address: u64,
        program_image_len_words: usize,
        max_padded_trace_length: usize,
    ) -> Result<Self, ProverError<F>> {
        Self::derive_from_rows(
            rows,
            memory_layout,
            min_bytecode_address,
            program_image_len_words,
            max_padded_trace_length,
            |row| (row.is_load() || row.is_store()).then(|| row.ram_address()),
            |row| row.circuit_flags().get(CircuitFlags::Jump),
        )
    }

    #[expect(non_snake_case)]
    fn derive_from_rows<F: JoltField, R: Sync>(
        rows: &[R],
        memory_layout: &MemoryLayout,
        min_bytecode_address: u64,
        program_image_len_words: usize,
        max_padded_trace_length: usize,
        ram_address: impl Fn(&R) -> Option<u64> + Sync,
        is_jump: impl Fn(&R) -> bool,
    ) -> Result<Self, ProverError<F>> {
        // The tracer stops when the PC stops changing or on a trap that emits
        // no rows, so only `j .` (or `jalr` to itself) leaves a jump last.
        if rows.last().is_some_and(|row| !is_jump(row)) {
            return Err(ProverError::TraceDoesNotEndInJump);
        }
        let trace_length = if rows.len() < MIN_PADDED_TRACE_LENGTH {
            MIN_PADDED_TRACE_LENGTH
        } else {
            (rows.len() + 1).next_power_of_two()
        };
        if trace_length > max_padded_trace_length {
            return Err(ProverError::Unsupported {
                reason: "trace exceeds the preprocessing's maximum padded trace length",
            });
        }

        #[cfg(feature = "parallel")]
        let touched = if rows.len() >= PARALLEL_DERIVE_MIN_ROWS {
            rows.par_iter()
                .filter_map(|row| remap_address(ram_address(row).unwrap_or(0), memory_layout))
                .max()
                .unwrap_or(0)
        } else {
            rows.iter()
                .filter_map(|row| remap_address(ram_address(row).unwrap_or(0), memory_layout))
                .max()
                .unwrap_or(0)
        };
        #[cfg(not(feature = "parallel"))]
        let touched = rows
            .iter()
            .filter_map(|row| remap_address(ram_address(row).unwrap_or(0), memory_layout))
            .max()
            .unwrap_or(0);
        let image_end = remap_address(min_bytecode_address, memory_layout).unwrap_or(0)
            + program_image_len_words as u64
            + 1;
        let ram_K = touched.max(image_end).next_power_of_two() as usize;

        let log_T = trace_length.ilog2() as usize;
        Ok(Self {
            trace_length,
            ram_K,
            rw_config: read_write_config(log_T, ram_K.ilog2() as usize),
            one_hot_config: one_hot_config(log_T),
            trace_polynomial_order: TracePolynomialOrder::CycleMajor,
            #[cfg(feature = "akita")]
            akita_chunk_profile: AkitaChunkProfile::Single,
        })
    }

    /// The shared commitment-embedding variable count: the one-hot main matrix
    /// (`log_k_chunk + log_T`) maxed with the advice and committed-program
    /// candidates that are actually present in this run.
    pub fn commitment_total_vars(
        &self,
        memory_layout: &MemoryLayout,
        has_trusted_advice: bool,
        has_untrusted_advice: bool,
        committed_program: Option<CommittedProgramCandidates>,
    ) -> usize {
        let mut total_vars =
            self.one_hot_config.committed_chunk_bits() + self.trace_length.ilog2() as usize;
        if has_trusted_advice {
            total_vars = total_vars.max(advice_total_vars(memory_layout.max_trusted_advice_size));
        }
        if has_untrusted_advice {
            total_vars = total_vars.max(advice_total_vars(memory_layout.max_untrusted_advice_size));
        }
        if let Some(committed) = committed_program {
            total_vars = total_vars
                .max(committed.bytecode_chunk_vars)
                .max(committed.program_image_vars);
        }
        total_vars
    }
}

/// Map a byte address into the RAM word index space: word offsets from the
/// memory layout's lowest mapped address; address 0 means "no access".
///
/// Panics on a nonzero address below the layout's lowest mapped address — a
/// malformed trace, failed loudly here (matching legacy) rather than
/// silently under-sizing `ram_K` and failing as an opaque sumcheck error.
pub fn remap_address(address: u64, memory_layout: &MemoryLayout) -> Option<u64> {
    if address == 0 {
        return None;
    }
    let lowest = memory_layout.get_lowest_address();
    assert!(address >= lowest, "Unexpected address {address}");
    Some((address - lowest) / 8)
}

/// Read-write checking phase splits: cycle variables in phase 1, address
/// variables in phase 2 (registers have a fixed 2^7 address space).
#[expect(non_snake_case)]
pub(crate) fn read_write_config(log_T: usize, ram_log_K: usize) -> JoltReadWriteConfig {
    JoltReadWriteConfig {
        ram_rw_phase1_num_rounds: log_T as u8,
        ram_rw_phase2_num_rounds: ram_log_K as u8,
        registers_rw_phase1_num_rounds: log_T as u8,
        registers_rw_phase2_num_rounds: REGISTER_COUNT.ilog2() as u8,
    }
}

/// Akita uses 4-bit committed chunks at every trace length. Other PCS modes
/// use 4-bit chunks below `log_T = 25` and 8-bit chunks above it. Virtual-RA
/// chunks remain 16 bits below that threshold and 32 bits at or above it.
#[expect(non_snake_case)]
pub(crate) fn one_hot_config(log_T: usize) -> JoltOneHotConfig {
    if log_T < ONEHOT_CHUNK_THRESHOLD_LOG_T {
        JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: (LOOKUP_ADDRESS_BITS / 8) as u8,
        }
    } else {
        JoltOneHotConfig {
            log_k_chunk: if cfg!(feature = "akita") { 4 } else { 8 },
            lookups_ra_virtual_log_k_chunk: (LOOKUP_ADDRESS_BITS / 4) as u8,
        }
    }
}

/// The committed one-hot chunk width [`one_hot_config`] selects for a
/// `2^log_T`-cycle trace. The committed preprocessing digest and the Dory
/// setup sizing read it from here so they keep describing the chunking the
/// prover actually uses.
#[cfg(not(feature = "akita"))]
#[expect(non_snake_case)]
pub(crate) fn committed_log_k_chunk(log_T: usize) -> u8 {
    one_hot_config(log_T).log_k_chunk
}

/// The committed-program precommitted candidates' variable counts, folded
/// into the shared commitment grid alongside the advice candidates.
#[derive(Clone, Copy, Debug)]
pub struct CommittedProgramCandidates {
    pub bytecode_chunk_vars: usize,
    pub program_image_vars: usize,
}

impl CommittedProgramCandidates {
    /// Read the candidates off the validated precommitted schedule: present
    /// exactly when committed-program layouts are.
    pub fn from_schedule(schedule: &jolt_verifier::stages::PrecommittedSchedule) -> Option<Self> {
        match (&schedule.bytecode, &schedule.program_image) {
            (Some(bytecode), Some(image)) => Some(Self {
                bytecode_chunk_vars: bytecode.chunk_shape().total_vars(),
                program_image_vars: image.image_shape().total_vars(),
            }),
            _ => None,
        }
    }
}

/// A word-aligned advice buffer's balanced Dory matrix variable count.
pub(crate) fn advice_total_vars(max_advice_size_bytes: u64) -> usize {
    let words = (max_advice_size_bytes / 8) as usize;
    words.next_power_of_two().max(1).ilog2() as usize
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use common::jolt_device::MemoryLayout;
    use jolt_field::Fr;
    use jolt_program::execution::TraceRow;
    use jolt_riscv::JoltInstructionKind as Kind;
    use jolt_riscv::{CapturedState, JoltInstructionRow, JoltTraceRow, NormalizedOperands};

    use super::ProverConfig;
    use crate::ProverError;

    const TEXT_BASE: u64 = 0x8000_0000;

    fn instruction(instruction_kind: Kind, operands: NormalizedOperands) -> JoltInstructionRow {
        JoltInstructionRow {
            instruction_kind,
            address: TEXT_BASE as usize,
            operands,
            ..JoltInstructionRow::default()
        }
    }

    fn derive_both(trace: &[JoltInstructionRow]) -> [Result<ProverConfig, ProverError<Fr>>; 2] {
        let rows: Vec<TraceRow> = trace
            .iter()
            .map(|&row| TraceRow::from_instruction(row).unwrap())
            .collect();
        let compact: Vec<JoltTraceRow> = trace
            .iter()
            .map(|row| JoltTraceRow::from_components(CapturedState::default(), row, 1).unwrap())
            .collect();
        let layout = MemoryLayout::default();
        [
            ProverConfig::derive(&rows, &layout, TEXT_BASE, 0, 1 << 12),
            ProverConfig::derive_compact(&compact, &layout, TEXT_BASE, 0, 1 << 12),
        ]
    }

    fn addi() -> JoltInstructionRow {
        instruction(
            Kind::ADDI,
            NormalizedOperands {
                rs1: Some(0),
                rd: Some(1),
                imm: 1,
                ..NormalizedOperands::default()
            },
        )
    }

    #[test]
    fn derive_rejects_a_trace_whose_last_row_is_not_a_jump() {
        let self_branch = instruction(
            Kind::BEQ,
            NormalizedOperands {
                rs1: Some(0),
                rs2: Some(0),
                ..NormalizedOperands::default()
            },
        );
        // A taken self-branch, and the last row before a trap that emitted none.
        for last in [self_branch, addi()] {
            for result in derive_both(&[addi(), last]) {
                assert!(
                    matches!(result, Err(ProverError::TraceDoesNotEndInJump)),
                    "{:?}: {result:?}",
                    last.instruction_kind
                );
            }
        }
    }

    #[test]
    fn derive_accepts_a_trace_ending_in_a_jump() {
        let jal = instruction(Kind::JAL, NormalizedOperands::default());
        let jalr = instruction(
            Kind::JALR,
            NormalizedOperands {
                rs1: Some(1),
                ..NormalizedOperands::default()
            },
        );
        for last in [jal, jalr] {
            for result in derive_both(&[addi(), last]) {
                assert!(result.is_ok(), "{:?}: {result:?}", last.instruction_kind);
            }
        }
    }
}
