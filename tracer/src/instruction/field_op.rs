//! BN254 Fr native-field coprocessor instruction.
//!
//! `FieldOp` is a single-cycle R-type instruction operating on the 16-slot ×
//! 256-bit field-register file `cpu.field_regs`. Encoded as opcode `0x0B`
//! (custom-0) with funct7 `0x40`. The funct3 field selects the op:
//!
//! | funct3 | Op   | Semantics                                                    |
//! |--------|------|--------------------------------------------------------------|
//! | 0x02   | FMUL | `field_regs[frd] = field_regs[frs1] * field_regs[frs2]` (Fr) |
//! | 0x03   | FADD | `field_regs[frd] = field_regs[frs1] + field_regs[frs2]` (Fr) |
//! | 0x04   | FINV | `field_regs[frd] = field_regs[frs1]^{-1}` (Fr; 0 → 0)        |
//! | 0x05   | FSUB | `field_regs[frd] = field_regs[frs1] - field_regs[frs2]` (Fr) |
//!
//! Bit layout (R-type): `frs2 | frs1 | funct3 | frd | opcode`, with
//! `funct7 = 0x40` occupying bits [31:25]. `frd`, `frs1`, `frs2` are 5-bit
//! field-register indices (0..15 used; high bit should be 0).
//!
//! Executing `FieldOp` emits one `FieldRegEvent` per cycle (single access
//! per cycle — Invariant 2 in `specs/native-field-registers.md`). FMov
//! instructions (FMovIntToFieldLimb / FMovFieldToIntLimb, funct3 `0x06`/`0x07`)
//! live in separate files because their operand layout differs.

use ark_bn254::Fr;
use ark_ff::{BigInteger, Field, PrimeField};
use serde::{Deserialize, Serialize};

use crate::emulator::cpu::{Cpu, FieldRegEvent};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    Cycle, NormalizedInstruction, RISCVInstruction, RISCVTrace,
};

/// Native-field coprocessor opcode (custom-0).
pub const FIELD_OP_OPCODE: u32 = 0x0B;
/// BN254 Fr funct7 family selector.
pub const BN254_FR_FUNCT7: u32 = 0x40;

pub const FUNCT3_FMUL: u8 = 0x02;
pub const FUNCT3_FADD: u8 = 0x03;
pub const FUNCT3_FINV: u8 = 0x04;
pub const FUNCT3_FSUB: u8 = 0x05;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct FieldOp {
    pub address: u64,
    pub operands: FormatR,
    /// funct3 selector (0x02 = FMUL, 0x03 = FADD, 0x04 = FINV, 0x05 = FSUB).
    /// Stored separately because FormatR doesn't carry funct3.
    pub funct3: u8,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}

impl RISCVInstruction for FieldOp {
    /// Match opcode + funct7 only — funct3 selects the specific op at runtime.
    const MASK: u32 = 0xfe00007f;
    const MATCH: u32 = (BN254_FR_FUNCT7 << 25) | FIELD_OP_OPCODE;

    type Format = FormatR;
    /// FieldOp doesn't touch RAM. Events land in `cpu.field_reg_events`, not `RAMAccess`.
    type RAMAccess = ();

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(word: u32, address: u64, validate: bool, compressed: bool) -> Self {
        if validate {
            debug_assert_eq!(
                word & Self::MASK,
                Self::MATCH,
                "word: {word:x}, mask: {:x}, match: {:x}",
                Self::MASK,
                Self::MATCH
            );
        }
        Self {
            address,
            operands: FormatR::parse(word),
            funct3: ((word >> 12) & 0x7) as u8,
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: compressed,
        }
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use rand::RngCore;
        let funct3 = (rng.next_u32() & 0x7) as u8;
        // Keep funct3 in the valid FieldOp range (0x02..=0x05).
        let funct3 = 0x02 + (funct3 % 4);
        Self {
            address: rng.next_u64(),
            operands: <FormatR as InstructionFormat>::random(rng),
            funct3,
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    fn execute(&self, cpu: &mut Cpu, _: &mut Self::RAMAccess) {
        let frd = self.operands.rd as usize;
        let frs1 = self.operands.rs1 as usize;
        let frs2 = self.operands.rs2 as usize;
        debug_assert!(frd < cpu.field_regs.len(), "frd out of range");
        debug_assert!(frs1 < cpu.field_regs.len(), "frs1 out of range");
        debug_assert!(frs2 < cpu.field_regs.len(), "frs2 out of range");

        let a = fr_from_limbs(&cpu.field_regs[frs1]);
        let b = fr_from_limbs(&cpu.field_regs[frs2]);

        let result = match self.funct3 {
            FUNCT3_FMUL => a * b,
            FUNCT3_FADD => a + b,
            FUNCT3_FSUB => a - b,
            FUNCT3_FINV => a.inverse().unwrap_or_else(|| {
                // FINV(0) is undefined: the R1CS constraint `rs1 · rd = 1` is
                // unsatisfiable for rs1 = 0. The SDK's `Fr::inverse()` returns
                // `Option<Fr>` and never emits FINV on zero; reaching this
                // branch means a caller bypassed the SDK with inline asm.
                // Fail fast here rather than produce a non-provable trace.
                panic!(
                    "FINV(0) is undefined at PC=0x{:x}; use SDK \
                     `Fr::inverse() -> Option<Fr>` (see jolt-inlines/bn254-fr) \
                     instead of emitting FieldOp via inline asm",
                    self.address
                );
            }),
            other => panic!("invalid FieldOp funct3: {other:#x}"),
        };

        cpu.field_regs[frd] = fr_to_limbs(&result);
    }

    fn has_side_effects(&self) -> bool {
        // Mutates cpu.field_regs AND emits an event; treat as side-effectful
        // so optimizers can't drop it.
        true
    }
}

impl RISCVTrace for FieldOp {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // Snapshot frd/frs1/frs2 pre-execution for the FieldRegEvent.
        let frd = self.operands.rd;
        let frs1 = self.operands.rs1;
        let frs2 = self.operands.rs2;
        let frd_pre = cpu.field_regs[frd as usize];
        let frs1_val = cpu.field_regs[frs1 as usize];
        let _frs2_val = cpu.field_regs[frs2 as usize];

        let mut cycle = RISCVCycle::<Self> {
            instruction: *self,
            register_state: Default::default(),
            ram_access: (),
        };
        self.operands()
            .capture_pre_execution_state(&mut cycle.register_state, cpu);
        self.execute(cpu, &mut cycle.ram_access);
        self.operands()
            .capture_post_execution_state(&mut cycle.register_state, cpu);

        let frd_post = cpu.field_regs[frd as usize];

        if let Some(trace_vec) = trace {
            // Global cycle index: `cpu.trace_len` is the pre-burst count;
            // `trace_vec.len()` accounts for cycles emitted earlier in this
            // instruction's burst (always 0 for single-cycle FieldOp).
            let cycle_index = cpu.trace_len + trace_vec.len();

            // FieldOp writes frd (always) and reads frs1 (plus frs2 unless FINV).
            // Emit a single write event for frd — reads are derivable from the
            // FR Twist Ra one-hot + Val polys by the witness layer.
            let _ = frs1_val; // Currently unused in the event; the Ra poly encodes the slot.
            cpu.field_reg_events.push(FieldRegEvent {
                cycle_index,
                slot: frd,
                old: frd_pre,
                new: frd_post,
            });
            trace_vec.push(cycle.into());
        }
    }
}

// -------- ark_bn254::Fr ↔ natural-form `[u64; 4]` bridges --------

/// Convert natural-form little-endian u64 limbs to an Fr element.
/// Values `>= p` are reduced modulo the BN254 Fr prime.
fn fr_from_limbs(limbs: &[u64; 4]) -> Fr {
    let mut bytes = [0u8; 32];
    for (i, &limb) in limbs.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    Fr::from_le_bytes_mod_order(&bytes)
}

/// Convert Fr to natural-form little-endian u64 limbs.
fn fr_to_limbs(fr: &Fr) -> [u64; 4] {
    let bytes: Vec<u8> = fr.into_bigint().to_bytes_le();
    let mut limbs = [0u64; 4];
    debug_assert!(bytes.len() <= 32, "Fr bytes must fit in 32");
    for (i, limb) in limbs.iter_mut().enumerate() {
        let start = i * 8;
        let end = std::cmp::min(start + 8, bytes.len());
        if start < bytes.len() {
            let mut buf = [0u8; 8];
            buf[..end - start].copy_from_slice(&bytes[start..end]);
            *limb = u64::from_le_bytes(buf);
        }
    }
    limbs
}

// Bring `RISCVCycle` into scope for the trace method.
use crate::instruction::RISCVCycle;

// -------- NormalizedInstruction bridges --------
//
// The RISCVInstruction trait requires `From<NormalizedInstruction>` + `Into<NormalizedInstruction>`.
// `NormalizedInstruction` doesn't carry funct3 for R-type — we lose the op selector
// on round-trip. Flagged as a known limitation; any code path round-tripping a
// FieldOp through `NormalizedInstruction` defaults to FMUL. Survey at task #53
// on main identified the same issue there.

impl From<NormalizedInstruction> for FieldOp {
    fn from(ni: NormalizedInstruction) -> Self {
        Self {
            address: ni.address as u64,
            operands: ni.operands.into(),
            funct3: FUNCT3_FMUL, // Default; funct3 is lost on NormalizedInstruction round-trip
            virtual_sequence_remaining: ni.virtual_sequence_remaining,
            is_first_in_sequence: ni.is_first_in_sequence,
            is_compressed: ni.is_compressed,
        }
    }
}

impl From<FieldOp> for NormalizedInstruction {
    fn from(instr: FieldOp) -> NormalizedInstruction {
        NormalizedInstruction {
            address: instr.address as usize,
            operands: instr.operands.into(),
            is_compressed: instr.is_compressed,
            virtual_sequence_remaining: instr.virtual_sequence_remaining,
            is_first_in_sequence: instr.is_first_in_sequence,
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::emulator::terminal::DummyTerminal;

    /// Encode an R-type instruction: funct7 | rs2 | rs1 | funct3 | rd | opcode
    fn encode_r(opcode: u32, funct3: u32, funct7: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
        (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    }

    fn test_cpu() -> Cpu {
        Cpu::new(Box::new(DummyTerminal::default()))
    }

    #[test]
    fn decode_fmul_matches_field_op_variant() {
        // FMUL: opcode=0x0B, funct3=0x02, funct7=0x40, frd=3, frs1=1, frs2=2
        let word = encode_r(FIELD_OP_OPCODE, FUNCT3_FMUL as u32, BN254_FR_FUNCT7, 3, 1, 2);
        let instr = crate::instruction::Instruction::decode(word, 0x1000, false).unwrap();
        match instr {
            crate::instruction::Instruction::FieldOp(op) => {
                assert_eq!(op.funct3, FUNCT3_FMUL);
                assert_eq!(op.operands.rd, 3);
                assert_eq!(op.operands.rs1, 1);
                assert_eq!(op.operands.rs2, 2);
            }
            other => panic!("expected Instruction::FieldOp, got {other:?}"),
        }
    }

    #[test]
    fn fmul_executes_bn254_fr_product() {
        let mut cpu = test_cpu();
        // field_regs[1] = 5 (natural-form little-endian limbs)
        cpu.field_regs[1] = [5, 0, 0, 0];
        // field_regs[2] = 7
        cpu.field_regs[2] = [7, 0, 0, 0];

        let word = encode_r(FIELD_OP_OPCODE, FUNCT3_FMUL as u32, BN254_FR_FUNCT7, 3, 1, 2);
        let op = FieldOp::new(word, 0x1000, true, false);
        op.execute(&mut cpu, &mut ());

        // 5 * 7 = 35, which fits in a single u64 limb (no reduction).
        assert_eq!(cpu.field_regs[3], [35, 0, 0, 0]);
    }

    #[test]
    fn fadd_fsub_finv_round_trip_over_fr() {
        let mut cpu = test_cpu();
        cpu.field_regs[1] = [42, 0, 0, 0];
        cpu.field_regs[2] = [13, 0, 0, 0];

        // FADD: 42 + 13 = 55
        let add = FieldOp::new(
            encode_r(FIELD_OP_OPCODE, FUNCT3_FADD as u32, BN254_FR_FUNCT7, 3, 1, 2),
            0,
            true,
            false,
        );
        add.execute(&mut cpu, &mut ());
        assert_eq!(cpu.field_regs[3], [55, 0, 0, 0]);

        // FSUB: 42 - 13 = 29
        let sub = FieldOp::new(
            encode_r(FIELD_OP_OPCODE, FUNCT3_FSUB as u32, BN254_FR_FUNCT7, 4, 1, 2),
            0,
            true,
            false,
        );
        sub.execute(&mut cpu, &mut ());
        assert_eq!(cpu.field_regs[4], [29, 0, 0, 0]);

        // FINV(1) = 1
        cpu.field_regs[5] = [1, 0, 0, 0];
        let inv = FieldOp::new(
            encode_r(FIELD_OP_OPCODE, FUNCT3_FINV as u32, BN254_FR_FUNCT7, 6, 5, 0),
            0,
            true,
            false,
        );
        inv.execute(&mut cpu, &mut ());
        assert_eq!(cpu.field_regs[6], [1, 0, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "FINV(0) is undefined")]
    fn finv_of_zero_panics() {
        // SDK's `Fr::inverse()` returns Option<Fr> and never emits FINV(0);
        // reaching this branch implies an inline-asm caller bypassed the SDK.
        // Tracer must fail fast with an actionable message.
        let mut cpu = test_cpu();
        cpu.field_regs[5] = [0, 0, 0, 0];
        let word = encode_r(FIELD_OP_OPCODE, FUNCT3_FINV as u32, BN254_FR_FUNCT7, 6, 5, 0);
        let op = FieldOp::new(word, 0x2000, true, false);
        op.execute(&mut cpu, &mut ());
    }

    #[test]
    fn trace_emits_field_reg_event() {
        let mut cpu = test_cpu();
        cpu.field_regs[1] = [2, 0, 0, 0];
        cpu.field_regs[2] = [3, 0, 0, 0];
        let word = encode_r(FIELD_OP_OPCODE, FUNCT3_FMUL as u32, BN254_FR_FUNCT7, 5, 1, 2);
        let op = FieldOp::new(word, 0x1000, true, false);
        let mut trace_vec: Vec<Cycle> = Vec::new();
        op.trace(&mut cpu, Some(&mut trace_vec));

        assert_eq!(trace_vec.len(), 1);
        assert_eq!(cpu.field_regs[5], [6, 0, 0, 0]);
        assert_eq!(cpu.field_reg_events.len(), 1);
        let ev = cpu.field_reg_events[0];
        assert_eq!(ev.slot, 5);
        assert_eq!(ev.old, [0, 0, 0, 0]);
        assert_eq!(ev.new, [6, 0, 0, 0]);
    }
}
