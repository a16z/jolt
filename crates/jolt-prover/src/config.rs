//! Per-proof configuration, derived from the execution trace.
//!
//! These five values are exactly the proof's wire config block
//! (`JoltProof::{trace_length, ram_K, rw_config, one_hot_config,
//! trace_polynomial_order}`) plus the Fiat-Shamir preamble inputs. The
//! derivation policies here must match `jolt-prover-legacy`'s choices
//! byte-for-byte while it remains the parity oracle; the byte-diff harness
//! pins them.

use common::constants::{ONEHOT_CHUNK_THRESHOLD_LOG_T, REGISTER_COUNT, XLEN};
use common::jolt_device::MemoryLayout;
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig, TracePolynomialOrder};
use jolt_field::FieldCore;
use jolt_program::execution::{RamAccess, TraceRow};

use crate::ProverError;

/// The full instruction lookup key width: two `XLEN`-bit operands.
const LOOKUP_ADDRESS_BITS: usize = 2 * XLEN;

/// The proof-shape configuration for one proving run.
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
    /// overwriting this field after derivation.
    pub trace_polynomial_order: TracePolynomialOrder,
}

impl ProverConfig {
    /// Derive the proof shape from an unpadded trace: pad the length (minimum
    /// 256 so `T >= K^(1/D)`, else next power of two past the trace plus its
    /// final no-op), size RAM to the highest touched (remapped) address or the
    /// program image extent, and pick the chunking policies from `log_T`.
    #[expect(non_snake_case)]
    pub fn derive<F: FieldCore>(
        rows: &[TraceRow],
        memory_layout: &MemoryLayout,
        min_bytecode_address: u64,
        program_image_len_words: usize,
        max_padded_trace_length: usize,
    ) -> Result<Self, ProverError<F>> {
        let trace_length = if rows.len() < 256 {
            256
        } else {
            (rows.len() + 1).next_power_of_two()
        };
        if trace_length > max_padded_trace_length {
            return Err(ProverError::Unsupported {
                reason: "trace exceeds the preprocessing's maximum padded trace length",
            });
        }

        let touched = rows
            .iter()
            .filter_map(|row| {
                let address = match row.ram_access {
                    RamAccess::Read(read) => read.address,
                    RamAccess::Write(write) => write.address,
                    RamAccess::NoOp => 0,
                };
                remap_address(address, memory_layout)
            })
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
pub fn remap_address(address: u64, memory_layout: &MemoryLayout) -> Option<u64> {
    if address == 0 {
        return None;
    }
    let lowest = memory_layout.get_lowest_address();
    (address >= lowest).then(|| (address - lowest) / 8)
}

/// Read-write checking phase splits: cycle variables in phase 1, address
/// variables in phase 2 (registers have a fixed 2^7 address space).
#[expect(non_snake_case)]
fn read_write_config(log_T: usize, ram_log_K: usize) -> JoltReadWriteConfig {
    JoltReadWriteConfig {
        ram_rw_phase1_num_rounds: log_T as u8,
        ram_rw_phase2_num_rounds: ram_log_K as u8,
        registers_rw_phase1_num_rounds: log_T as u8,
        registers_rw_phase2_num_rounds: REGISTER_COUNT.ilog2() as u8,
    }
}

/// One-hot chunking policy, mirroring `jolt-prover-legacy`'s
/// `OneHotConfig::new`: below the trace-length threshold (`log_T < 25`),
/// 4-bit committed chunks and `LOG_K/8 = 16`-bit virtual-RA chunks; at or
/// above it, 8-bit committed chunks and `LOG_K/4 = 32`-bit virtual-RA chunks
/// (a branch that requires a 2^25-cycle trace and may never have run in
/// practice — kept for parity).
#[expect(non_snake_case)]
fn one_hot_config(log_T: usize) -> JoltOneHotConfig {
    if log_T < ONEHOT_CHUNK_THRESHOLD_LOG_T {
        JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: (LOOKUP_ADDRESS_BITS / 8) as u8,
        }
    } else {
        JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: (LOOKUP_ADDRESS_BITS / 4) as u8,
        }
    }
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
