//! Per-proof configuration, derived from execution dimensions.
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
use jolt_program::execution::ExecutionDimensions;

use crate::ProverError;

const LOOKUP_ADDRESS_BITS: usize = 2 * XLEN;

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
    /// matrix. [`ProverConfig::derive_from_dimensions`] always picks cycle-major (legacy has
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
    /// Derive the proof shape from execution facts without accessing trace rows.
    /// Pad the trace length past its final no-op, size RAM to cover accessed
    /// addresses and the program image, and select chunking policies.
    #[tracing::instrument(skip_all, fields(rows = dimensions.trace_length))]
    pub fn derive_from_dimensions<F: JoltField>(
        dimensions: ExecutionDimensions,
        memory_layout: &MemoryLayout,
        min_bytecode_address: u64,
        program_image_len_words: usize,
        max_padded_trace_length: usize,
    ) -> Result<Self, ProverError<F>> {
        if dimensions.trace_length != 0 && !dimensions.ends_in_jump {
            return Err(ProverError::TraceDoesNotEndInJump);
        }
        let trace_length = dimensions
            .trace_length
            .checked_add(1)
            .and_then(usize::checked_next_power_of_two)
            .ok_or(ProverError::Unsupported {
                reason: "trace length overflows the padded domain",
            })?
            .max(MIN_PADDED_TRACE_LENGTH);
        if let Some(min) = dimensions.ram_bounds.min() {
            let _ = checked_remap_address(min, memory_layout)
                .map_err(|reason| ProverError::Unsupported { reason })?;
        }
        let touched =
            checked_remap_address(dimensions.ram_bounds.max().unwrap_or(0), memory_layout)
                .map_err(|reason| ProverError::Unsupported { reason })?
                .unwrap_or(0);
        let image_end = checked_remap_address(min_bytecode_address, memory_layout)
            .map_err(|reason| ProverError::Unsupported { reason })?
            .unwrap_or(0)
            .checked_add(program_image_len_words as u64)
            .and_then(|end| end.checked_add(1))
            .ok_or(ProverError::Unsupported {
                reason: "program image extent overflows the RAM domain",
            })?;
        let ram_k = touched
            .max(image_end)
            .checked_next_power_of_two()
            .and_then(|domain| usize::try_from(domain).ok())
            .ok_or(ProverError::Unsupported {
                reason: "RAM domain does not fit the host address space",
            })?;
        Self::from_padded_dimensions(trace_length, ram_k, max_padded_trace_length)
    }

    /// Use domains already selected by an execution backend. The caller must
    /// ensure the RAM domain covers execution and the program image; this checks
    /// domain geometry and applies the same default policies as execution derivation.
    #[expect(non_snake_case)]
    pub fn from_padded_dimensions<F: JoltField>(
        trace_length: usize,
        ram_K: usize,
        max_padded_trace_length: usize,
    ) -> Result<Self, ProverError<F>> {
        if !trace_length.is_power_of_two() || trace_length < MIN_PADDED_TRACE_LENGTH {
            return Err(ProverError::Unsupported {
                reason: "invalid padded trace domain",
            });
        }
        if trace_length > max_padded_trace_length {
            return Err(ProverError::Unsupported {
                reason: "trace exceeds the preprocessing's maximum padded trace length",
            });
        }
        if !ram_K.is_power_of_two() {
            return Err(ProverError::Unsupported {
                reason: "RAM domain must be a nonzero power of two",
            });
        }
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
#[expect(
    clippy::panic,
    reason = "preserve the public helper's documented panic contract"
)]
pub fn remap_address(address: u64, memory_layout: &MemoryLayout) -> Option<u64> {
    checked_remap_address(address, memory_layout)
        .unwrap_or_else(|_| panic!("Unexpected address {address}"))
}

fn checked_remap_address(
    address: u64,
    memory_layout: &MemoryLayout,
) -> Result<Option<u64>, &'static str> {
    if address == 0 {
        return Ok(None);
    }
    address
        .checked_sub(memory_layout.get_lowest_address())
        .map(|offset| Some(offset / 8))
        .ok_or("RAM address precedes the memory layout")
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
#[expect(clippy::expect_used, reason = "test module")]
mod dimension_tests {
    use super::*;
    use common::jolt_device::MemoryConfig;
    use jolt_field::Fr;
    use jolt_program::execution::{RamAccess, RamAddressBounds, TraceRow};

    #[test]
    fn execution_summaries_preserve_address_validation() {
        use jolt_program::execution::{RamRead, RamWrite};
        use jolt_riscv::{
            CapturedState, JoltInstructionKind, JoltInstructionRow, JoltTraceRow, LoadState,
            StoreState,
        };
        let layout = MemoryLayout::new(&MemoryConfig {
            program_size: Some(1024),
            ..Default::default()
        });
        let lowest = layout.get_lowest_address();
        for length in [3, 1 << 16] {
            for addresses in [
                [0, 0, 0],
                [0, lowest, lowest + 64],
                [lowest - 1, lowest + 64, 0],
                [lowest + 64, lowest - 1, 0],
            ] {
                let (mut rows, mut compact): (Vec<_>, Vec<_>) = (0..length)
                    .map(|index| {
                        let address = addresses[index % addresses.len()];
                        let store = index % 2 == 1;
                        let instruction = JoltInstructionRow {
                            instruction_kind: if store {
                                JoltInstructionKind::SD
                            } else {
                                JoltInstructionKind::LD
                            },
                            ..Default::default()
                        };
                        let (ram, state) = if store {
                            (
                                RamAccess::Write(RamWrite {
                                    address,
                                    ..Default::default()
                                }),
                                CapturedState::Store(StoreState {
                                    ram_address: address,
                                    ..Default::default()
                                }),
                            )
                        } else {
                            (
                                RamAccess::Read(RamRead { address, value: 0 }),
                                CapturedState::Load(LoadState {
                                    ram_address: address,
                                    ..Default::default()
                                }),
                            )
                        };
                        (
                            TraceRow::new(instruction, Default::default(), ram).expect("valid row"),
                            JoltTraceRow::from_components(state, &instruction, 1)
                                .expect("valid compact row"),
                        )
                    })
                    .unzip();
                let jump = JoltInstructionRow {
                    instruction_kind: JoltInstructionKind::JAL,
                    ..Default::default()
                };
                rows.push(TraceRow::from_instruction(jump).expect("jump row"));
                compact.push(
                    JoltTraceRow::from_components(CapturedState::default(), &jump, 1)
                        .expect("compact jump"),
                );
                let dimensions = ExecutionDimensions::from_rows(&rows);
                assert_eq!(dimensions, ExecutionDimensions::from_compact(&compact));
                assert_eq!(
                    dimensions.ram_bounds.min(),
                    addresses.iter().copied().filter(|a| *a != 0).min()
                );
                assert_eq!(
                    dimensions.ram_bounds.max(),
                    addresses.iter().copied().filter(|a| *a != 0).max()
                );
                let result = ProverConfig::derive_from_dimensions::<Fr>(
                    dimensions,
                    &layout,
                    lowest,
                    0,
                    usize::MAX,
                );
                if addresses.contains(&(lowest - 1)) {
                    assert!(matches!(
                        result,
                        Err(ProverError::Unsupported {
                            reason: "RAM address precedes the memory layout"
                        })
                    ));
                } else {
                    assert!(result.is_ok(), "valid addresses: {result:?}");
                }
            }
        }
    }

    #[test]
    fn execution_facts_determine_padding_and_ram_extent() {
        let layout = MemoryLayout::new(&MemoryConfig {
            program_size: Some(1024),
            ..Default::default()
        });
        let lowest = layout.get_lowest_address();
        let derive = |trace_length, max_ram_address, image_words| {
            ProverConfig::derive_from_dimensions::<Fr>(
                ExecutionDimensions {
                    trace_length,
                    ends_in_jump: trace_length != 0,
                    ram_bounds: RamAddressBounds::try_new(max_ram_address, max_ram_address)
                        .expect("valid bounds"),
                },
                &layout,
                lowest,
                image_words,
                1 << 26,
            )
            .expect("valid execution dimensions")
        };
        let empty = derive(0, None, 0);
        assert_eq!(empty.trace_length, MIN_PADDED_TRACE_LENGTH);
        assert_eq!(empty.ram_K, 1);
        let touched = derive(MIN_PADDED_TRACE_LENGTH, Some(lowest + 8 * 513), 3);
        assert_eq!(touched.trace_length, MIN_PADDED_TRACE_LENGTH * 2);
        assert_eq!(touched.ram_K, 1024);
        let image = derive(7, None, 1024);
        assert_eq!(image.ram_K, 2048);
        assert_eq!(
            derive((1 << 25) - 1, None, 0).one_hot_config.log_k_chunk,
            if cfg!(feature = "akita") { 4 } else { 8 }
        );
        assert_eq!(derive((1 << 24) - 1, None, 0).one_hot_config.log_k_chunk, 4);
    }

    #[test]
    fn invalid_execution_and_padded_domains_are_rejected() {
        let layout = MemoryLayout::new(&MemoryConfig {
            program_size: Some(1024),
            ..Default::default()
        });
        for dimensions in [
            ExecutionDimensions {
                trace_length: usize::MAX,
                ends_in_jump: true,
                ram_bounds: RamAddressBounds::default(),
            },
            ExecutionDimensions {
                trace_length: 0,
                ends_in_jump: false,
                ram_bounds: RamAddressBounds::try_new(
                    Some(layout.get_lowest_address() - 1),
                    Some(layout.get_lowest_address() - 1),
                )
                .expect("ordered bounds, invalid for this layout"),
            },
        ] {
            assert!(ProverConfig::derive_from_dimensions::<Fr>(
                dimensions,
                &layout,
                0,
                0,
                usize::MAX
            )
            .is_err());
        }
        for (trace, ram, maximum) in [
            (0, 1, usize::MAX),
            (MIN_PADDED_TRACE_LENGTH - 1, 1, usize::MAX),
            (MIN_PADDED_TRACE_LENGTH, 0, usize::MAX),
            (MIN_PADDED_TRACE_LENGTH, 3, usize::MAX),
            (MIN_PADDED_TRACE_LENGTH, 1, MIN_PADDED_TRACE_LENGTH / 2),
        ] {
            assert!(ProverConfig::from_padded_dimensions::<Fr>(trace, ram, maximum).is_err());
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use common::jolt_device::MemoryLayout;
    use jolt_field::Fr;
    use jolt_program::execution::{ExecutionDimensions, TraceRow};
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
            ProverConfig::derive_from_dimensions(
                ExecutionDimensions::from_rows(&rows),
                &layout,
                TEXT_BASE,
                0,
                1 << 12,
            ),
            ProverConfig::derive_from_dimensions(
                ExecutionDimensions::from_compact(&compact),
                &layout,
                TEXT_BASE,
                0,
                1 << 12,
            ),
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
