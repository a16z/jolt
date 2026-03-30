//! End-to-end witness generation from execution trace.
//!
//! [`generate_witnesses`] bridges the tracer's [`Cycle`] representation to the
//! two witness forms the proving pipeline requires:
//! 1. Per-cycle R1CS variable vectors for Spartan
//! 2. Committed polynomial evaluation tables for sumcheck stages

use jolt_field::Field;
use jolt_verifier::{OneHotConfig, ProverConfig, ReadWriteConfig};
use tracer::instruction::{Cycle, RAMAccess};

use crate::bytecode::BytecodePreprocessing;
use crate::config::WitnessConfig;
use crate::cycle_data::trace_to_cycle_data;
use crate::r1cs_inputs::trace_to_witnesses;
use crate::witness::Witness;
use crate::WitnessBuilder;

/// Complete witness generation output.
///
/// Contains everything needed to run the proving pipeline:
/// - R1CS witnesses for Spartan (S1)
/// - Committed polynomial tables for sumcheck stages (S2–S7)
/// - Bytecode preprocessing for PC expansion
/// - Proof configuration computed from trace parameters
pub struct WitnessOutput<F: Field> {
    /// Per-cycle R1CS variable vectors, ready for
    /// [`interleave_witnesses`](crate::preprocessing::interleave_witnesses).
    pub cycle_witnesses: Vec<Vec<F>>,
    /// Committed polynomial evaluation tables for sumcheck stages.
    pub witness: Witness<F>,
    /// Bytecode PC map (needed by stage construction).
    pub bytecode: BytecodePreprocessing,
    /// Proof configuration derived from trace parameters.
    pub config: ProverConfig,
}

/// Generates all witnesses from an execution trace.
///
/// Processes the trace in three steps:
/// 1. Builds [`BytecodePreprocessing`] for PC expansion
/// 2. Converts each cycle to an R1CS witness vector (41 field elements)
/// 3. Extracts committed polynomial tables (Inc, RA one-hot) via
///    [`WitnessBuilder`] → [`Witness`] (via [`WitnessSink`])
///
/// The returned [`ProverConfig`] is computed from trace parameters and
/// carried inside the proof for self-contained verification.
pub fn generate_witnesses<F: Field>(trace: &[Cycle]) -> WitnessOutput<F> {
    assert!(!trace.is_empty(), "trace must not be empty");

    let bytecode = BytecodePreprocessing::new(trace);

    // Pad trace to next power of two with NoOp cycles BEFORE generating witnesses.
    // This matches jolt-core's approach: the last real cycle sees next_cycle = Some(NoOp),
    // so next_is_noop = true and PC update constraints are satisfied via DoNotUpdateUnexpPC.
    let padded_len = trace.len().next_power_of_two();
    let padded_trace: Vec<Cycle> = trace
        .iter()
        .copied()
        .chain(std::iter::repeat_n(Cycle::NoOp, padded_len - trace.len()))
        .collect();

    let cycle_witnesses = trace_to_witnesses(&padded_trace, &bytecode);

    let cycle_data = trace_to_cycle_data(&padded_trace, &bytecode);
    debug_assert_eq!(cycle_data.len(), padded_len);

    // Derive configuration from trace parameters
    let log_t = padded_len.trailing_zeros() as usize;
    let one_hot_config = OneHotConfig::new(log_t);
    let log_k_chunk = one_hot_config.log_k_chunk as usize;

    let bytecode_k = bytecode.code_size().max(1).next_power_of_two();
    let log_k_bytecode = bytecode_k.trailing_zeros() as usize;

    let ram_k = compute_ram_k(trace);
    let log_k_ram = ram_k.trailing_zeros() as usize;

    let witness_config = WitnessConfig::new(log_k_chunk, 128, log_k_bytecode, log_k_ram);

    // Build committed polynomial evaluation tables
    let mut witness = Witness::new();
    let builder = WitnessBuilder::new(witness_config);
    builder.build(&cycle_data, &mut witness);

    let config = ProverConfig {
        trace_length: padded_len,
        ram_k,
        bytecode_k,
        one_hot_config,
        rw_config: ReadWriteConfig::new(log_t, log_k_ram),
    };

    WitnessOutput {
        cycle_witnesses,
        witness,
        bytecode,
        config,
    }
}

/// Computes the RAM address space size from the trace.
///
/// Returns the smallest power-of-two that covers all RAM addresses,
/// with a minimum of 2 (log_k_ram = 1) for programs with no RAM access.
fn compute_ram_k(trace: &[Cycle]) -> usize {
    let max_addr = trace
        .iter()
        .filter_map(|c| match c.ram_access() {
            RAMAccess::Read(r) => Some(r.address),
            RAMAccess::Write(w) => Some(w.address),
            RAMAccess::NoOp => None,
        })
        .max()
        .unwrap_or(0);

    // Address space must hold [0, max_addr], so size = max_addr + 1
    (max_addr as usize + 1).max(2).next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use crate::PolynomialId;
    use tracer::instruction::{
        add::ADD,
        format::format_r::{FormatR, RegisterStateFormatR},
        RISCVCycle,
    };

    fn make_add_cycle(addr: u64, rs1: u64, rs2: u64) -> Cycle {
        Cycle::from(RISCVCycle {
            instruction: ADD {
                address: addr,
                operands: FormatR {
                    rd: 1,
                    rs1: 2,
                    rs2: 3,
                },
                ..ADD::default()
            },
            register_state: RegisterStateFormatR {
                rd: (0, rs1.wrapping_add(rs2)),
                rs1,
                rs2,
            },
            ram_access: (),
        })
    }

    #[test]
    fn generates_r1cs_witnesses() {
        let trace = vec![make_add_cycle(0x1000, 3, 4), make_add_cycle(0x1004, 10, 20)];
        let output = generate_witnesses::<Fr>(&trace);

        // Two actual cycles (padded to power of two by trace_to_witnesses...
        // actually trace_to_witnesses does NOT pad, only cycle_data does)
        assert_eq!(output.cycle_witnesses.len(), 2);
        assert_eq!(
            output.cycle_witnesses[0].len(),
            jolt_r1cs::NUM_VARS_PER_CYCLE
        );
    }

    #[test]
    fn generates_committed_polynomials() {
        let trace = vec![make_add_cycle(0x1000, 3, 4), make_add_cycle(0x1004, 10, 20)];
        let output = generate_witnesses::<Fr>(&trace);

        // RD_INC and RAM_INC (dense) should always be present
        assert!(output.witness.contains(PolynomialId::RdInc));
        assert!(output.witness.contains(PolynomialId::RamInc));
    }

    #[test]
    fn config_trace_length_is_power_of_two() {
        let trace = vec![
            make_add_cycle(0x1000, 1, 2),
            make_add_cycle(0x1004, 3, 4),
            make_add_cycle(0x1008, 5, 6),
        ];
        let output = generate_witnesses::<Fr>(&trace);

        // 3 cycles → padded to 4
        assert_eq!(output.config.trace_length, 4);
        assert!(output.config.trace_length.is_power_of_two());
    }

    #[test]
    fn ram_k_minimum_for_no_ram() {
        // ADD has no RAM access
        let trace = vec![make_add_cycle(0x1000, 1, 2)];
        let output = generate_witnesses::<Fr>(&trace);

        assert!(output.config.ram_k >= 2);
        assert!(output.config.ram_k.is_power_of_two());
    }

    #[test]
    fn bytecode_preprocessing_available() {
        let trace = vec![make_add_cycle(0x1000, 1, 2), make_add_cycle(0x1004, 3, 4)];
        let output = generate_witnesses::<Fr>(&trace);

        assert_eq!(output.bytecode.code_size(), 2);
    }
}
