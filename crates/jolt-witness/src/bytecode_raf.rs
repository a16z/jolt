//! Bytecode data for BytecodeReadRaf sumcheck computation.
//!
//! [`BytecodeData`] provides per-bytecode-entry field data and per-cycle
//! bytecode index mappings needed by the `BytecodeVal` and `EqPushforward`
//! input bindings.

use jolt_field::Field;

/// Per-bytecode-entry field data needed for Val polynomial computation.
///
/// Each entry corresponds to one row of the bytecode table (one instruction).
/// Boolean flags are stored directly; register indices are `Option<u8>` where
/// `None` means the instruction doesn't use that register operand.
#[derive(Debug, Clone)]
pub struct BytecodeEntry<F> {
    /// Instruction address as a field element.
    pub address: F,
    /// Immediate value as a field element.
    pub imm: F,
    /// Circuit flags (NUM_CIRCUIT_FLAGS = 14 booleans).
    pub circuit_flags: Vec<bool>,
    /// Destination register index (0-31), None if unused.
    pub rd: Option<u8>,
    /// Source register 1 index, None if unused.
    pub rs1: Option<u8>,
    /// Source register 2 index, None if unused.
    pub rs2: Option<u8>,
    /// Lookup table index (for Stage 5 table flags).
    pub lookup_table: Option<usize>,
    /// True if operands are interleaved (NOT a RAF instruction).
    pub is_interleaved: bool,

    // InstructionFlags booleans:
    pub is_branch: bool,
    pub left_is_rs1: bool,
    pub left_is_pc: bool,
    pub right_is_rs2: bool,
    pub right_is_imm: bool,
    pub is_noop: bool,
}

/// Bytecode data provided to the prover runtime for BytecodeReadRaf computation.
///
/// Contains per-cycle bytecode index mapping and per-bytecode-entry field data.
/// Passed to `prove()` alongside `LookupTraceData`.
#[derive(Debug, Clone)]
pub struct BytecodeData<F> {
    /// Per-cycle bytecode table index: `pc_indices[t]` is the bytecode table
    /// row that cycle `t` executes. Length = trace_length (padded).
    pub pc_indices: Vec<usize>,
    /// Per-bytecode-entry field data. Length = K (bytecode table size).
    pub entries: Vec<BytecodeEntry<F>>,
    /// Bytecode table index of the ELF entry point.
    pub entry_index: usize,
    /// Number of lookup tables (for Stage 5 table flag iteration).
    pub num_lookup_tables: usize,
}

impl<F: Field> BytecodeData<F> {
    /// Materialize the full Val polynomial for a `BytecodeVal` input binding.
    ///
    /// Encapsulates all protocol logic: gamma power computation, eq table
    /// construction, stage-specific formula dispatch, RAF modification, and
    /// overall gamma scaling. The runtime calls this as a black box.
    #[allow(clippy::too_many_arguments)]
    pub fn materialize_val(
        &self,
        challenges: &[F],
        stage: u8,
        stage_gamma_base: usize,
        stage_gamma_count: usize,
        gamma_base: usize,
        raf_gamma_power: Option<u8>,
        register_eq_challenges: &[usize],
    ) -> Vec<F> {
        let sg_base = challenges[stage_gamma_base];
        let mut gammas = Vec::with_capacity(stage_gamma_count);
        let mut pow = F::one();
        for _ in 0..stage_gamma_count {
            gammas.push(pow);
            pow *= sg_base;
        }

        let eq_r_register = if register_eq_challenges.is_empty() {
            Vec::new()
        } else {
            let point: Vec<F> = register_eq_challenges
                .iter()
                .map(|&ci| challenges[ci])
                .collect();
            jolt_poly::EqPolynomial::<F>::evals(&point, None)
        };

        let mut val = compute_val_stage(&self.entries, stage, &gammas, &eq_r_register);

        let gamma = challenges[gamma_base];
        if let Some(p) = raf_gamma_power {
            let mut raf_g = F::one();
            for _ in 0..p {
                raf_g *= gamma;
            }
            for (k, v) in val.iter_mut().enumerate() {
                *v += raf_g * F::from_u64(k as u64);
            }
        }

        let mut overall = F::one();
        for _ in 0..stage {
            overall *= gamma;
        }
        for v in &mut val {
            *v *= overall;
        }

        val
    }
}

/// Compute the Val polynomial for a given stage from bytecode entries and challenge values.
///
/// This mirrors `BytecodeReadRafSumcheckParams::compute_val_polys` from jolt-core,
/// producing byte-identical results.
fn compute_val_stage<F: Field>(
    entries: &[BytecodeEntry<F>],
    stage: u8,
    stage_gammas: &[F],
    eq_r_register: &[F],
) -> Vec<F> {
    let k = entries.len();
    let mut val = vec![F::zero(); k];

    for (idx, entry) in entries.iter().enumerate() {
        val[idx] = match stage {
            0 => {
                // Stage 1 (Spartan outer):
                // Val(k) = address + γ·imm + Σ γ^{2+f}·circuit_flags[f]
                let mut lc = entry.address;
                lc += stage_gammas[1] * entry.imm;
                for (f, &flag) in entry.circuit_flags.iter().enumerate() {
                    if flag {
                        lc += stage_gammas[2 + f];
                    }
                }
                lc
            }
            1 => {
                // Stage 2 (product virtualization):
                // Val(k) = γ₀·jump + γ₁·branch + γ₂·wlotord + γ₃·virtual
                let mut lc = F::zero();
                // CircuitFlags::Jump = index 5
                if entry.circuit_flags[5] {
                    lc += stage_gammas[0];
                }
                if entry.is_branch {
                    lc += stage_gammas[1];
                }
                // CircuitFlags::WriteLookupOutputToRD = index 6
                if entry.circuit_flags[6] {
                    lc += stage_gammas[2];
                }
                // CircuitFlags::VirtualInstruction = index 7
                if entry.circuit_flags[7] {
                    lc += stage_gammas[3];
                }
                lc
            }
            2 => {
                // Stage 3 (shift):
                // Val(k) = imm + γ·address + γ²·left_is_rs1 + γ³·left_is_pc
                //          + γ⁴·right_is_rs2 + γ⁵·right_is_imm + γ⁶·is_noop
                //          + γ⁷·virtual + γ⁸·is_first_in_seq
                let mut lc = entry.imm;
                lc += stage_gammas[1] * entry.address;
                if entry.left_is_rs1 {
                    lc += stage_gammas[2];
                }
                if entry.left_is_pc {
                    lc += stage_gammas[3];
                }
                if entry.right_is_rs2 {
                    lc += stage_gammas[4];
                }
                if entry.right_is_imm {
                    lc += stage_gammas[5];
                }
                if entry.is_noop {
                    lc += stage_gammas[6];
                }
                // CircuitFlags::VirtualInstruction = index 7
                if entry.circuit_flags[7] {
                    lc += stage_gammas[7];
                }
                // CircuitFlags::IsFirstInSequence = index 12
                if entry.circuit_flags[12] {
                    lc += stage_gammas[8];
                }
                lc
            }
            3 => {
                // Stage 4 (registers read/write):
                // Val(k) = γ₀·eq(rd,r_reg) + γ₁·eq(rs1,r_reg) + γ₂·eq(rs2,r_reg)
                let rd_eq = entry.rd.map_or(F::zero(), |r| eq_r_register[r as usize]);
                let rs1_eq = entry.rs1.map_or(F::zero(), |r| eq_r_register[r as usize]);
                let rs2_eq = entry.rs2.map_or(F::zero(), |r| eq_r_register[r as usize]);
                rd_eq * stage_gammas[0] + rs1_eq * stage_gammas[1] + rs2_eq * stage_gammas[2]
            }
            4 => {
                // Stage 5 (registers val-eval + lookups):
                // Val(k) = eq(rd,r_reg) + γ₁·raf_flag + Σ γ_{2+t}·table_flag[t]
                let mut lc = entry.rd.map_or(F::zero(), |r| eq_r_register[r as usize]);
                if !entry.is_interleaved {
                    lc += stage_gammas[1];
                }
                if let Some(table_idx) = entry.lookup_table {
                    lc += stage_gammas[2 + table_idx];
                }
                lc
            }
            _ => panic!("BytecodeVal: invalid stage {stage}"),
        };
    }

    val
}
