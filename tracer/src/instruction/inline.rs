//! Inline instruction support for RISC-V.
//!
//! The inline system uses the RISC-V custom-0 (0x0B) and custom-1 (0x2B) opcodes
//! with the Inline-format instruction encoding. Inlines are uniquely identified by their
//! opcode, funct3, and funct7 fields.
//!
//! Inline implementations register themselves at link time via `inventory::submit!`.
//! The INLINE instruction iterates these registrations to find the matching builder.

use super::{
    format::{format_inline::FormatInline, InstructionFormat},
    Cycle, Instruction, RISCVInstruction, RISCVTrace,
};
use crate::{
    emulator::cpu::Cpu, instruction::SourceInstruction,
    utils::virtual_registers::VirtualRegisterAllocator,
};
use jolt_program::expand::{
    ExpandedInstructionSequence, ExpansionError, InlineAdmissibility, InlineExpansionBuilder,
    InlineExpansionProvider, InlineOperands,
};
use jolt_riscv::{
    InlineExtension, JoltInstructionProfile, JoltInstructionRow, SourceInlineKey,
    SourceInstructionKind, RV64IMAC_JOLT_ALL_INLINES,
};
use serde::{Deserialize, Serialize};

use std::collections::VecDeque;

pub type InlineSequenceFn = fn(
    InlineExpansionBuilder,
    InlineOperands,
) -> Result<ExpandedInstructionSequence, ExpansionError>;

pub type AdviceFn = fn(FormatInline, &mut Cpu) -> Option<VecDeque<u64>>;

/// Runtime registration for one inline opcode.
///
/// `build_sequence` is the static recipe path consumed by `jolt-program`; it
/// must not construct tracer instructions. `build_advice` is the runtime-only
/// path that may inspect `Cpu` and produces values for `VirtualAdvice` rows in
/// the same order as the recipe emitted them.
pub struct InlineRegistration {
    pub opcode: u32,
    pub funct3: u32,
    pub funct7: u32,
    pub extension: InlineExtension,
    pub name: &'static str,
    pub admissibility: InlineAdmissibility,
    pub build_sequence: InlineSequenceFn,
    pub build_advice: AdviceFn,
}

inventory::collect!(InlineRegistration);

fn find_inline(opcode: u32, funct3: u32, funct7: u32) -> &'static InlineRegistration {
    inventory::iter::<InlineRegistration>
        .into_iter()
        .find(|r| r.opcode == opcode && r.funct3 == funct3 && r.funct7 == funct7)
        .unwrap_or_else(|| {
            panic!(
                "No inline registered for opcode={opcode:#04x}, funct3={funct3:#03b}, funct7={funct7:#09b}."
            )
        })
}

fn lookup_inline(inline: SourceInlineKey) -> Result<&'static InlineRegistration, ExpansionError> {
    inventory::iter::<InlineRegistration>
        .into_iter()
        .find(|r| {
            r.opcode == inline.opcode as u32
                && r.funct3 == inline.funct3 as u32
                && r.funct7 == inline.funct7 as u32
        })
        .ok_or(ExpansionError::UnsupportedInstruction)
}

fn build_registered_sequence(
    registration: &InlineRegistration,
    instruction: &SourceInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let source = *instruction.row();
    let operands = InlineOperands::from_source_row(source)?;
    (registration.build_sequence)(InlineExpansionBuilder::new(source), operands)
}

/// Return all linked inline registration keys and names.
///
/// This is intended for diagnostics and tests; expansion itself uses exact
/// opcode/funct3/funct7 lookup through the provider.
pub fn list_registered_inlines() -> Vec<((u32, u32, u32), String)> {
    inventory::iter::<InlineRegistration>
        .into_iter()
        .map(|r| ((r.opcode, r.funct3, r.funct7), r.name.to_string()))
        .collect()
}

/// Check whether a linked inline registration exists for the encoded key.
pub fn is_inline_registered(opcode: u32, funct3: u32, funct7: u32) -> bool {
    inventory::iter::<InlineRegistration>
        .into_iter()
        .any(|r| r.opcode == opcode && r.funct3 == funct3 && r.funct7 == funct7)
}

/// Inline provider backed by tracer-owned `inventory` registrations.
///
/// This type is the bridge from `jolt-program` static bytecode expansion to the
/// inline crates linked into the tracer binary. It performs registration lookup
/// and profile gating, then delegates recipe construction to the registered
/// inline builder.
#[derive(Debug, Clone, Default)]
pub struct TracerInlineExpansionProvider;

impl TracerInlineExpansionProvider {
    pub fn new() -> Self {
        Self
    }
}

impl InlineExpansionProvider for TracerInlineExpansionProvider {
    fn expand_inline(
        &mut self,
        instruction: &SourceInstruction,
        profile: JoltInstructionProfile,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        let inline = instruction
            .row()
            .inline
            .ok_or(ExpansionError::MalformedInstruction(
                "missing inline source metadata",
            ))?;

        if !profile.supports_source(SourceInstructionKind::Inline) {
            return Err(ExpansionError::UnsupportedInstruction);
        }
        let registration = lookup_inline(inline)?;
        if !profile.supports_inline(registration.extension) {
            return Err(ExpansionError::UnsupportedInstruction);
        }
        if let InlineAdmissibility::InternalOnly { reason } = registration.admissibility {
            return Err(ExpansionError::InternalOnlyInline {
                name: registration.name,
                reason,
            });
        }

        build_registered_sequence(registration, instruction)
    }
}

#[cfg(any(feature = "test-utils", test))]
#[derive(Debug, Clone, Default)]
struct TestInlineExpansionProvider;

#[cfg(any(feature = "test-utils", test))]
impl InlineExpansionProvider for TestInlineExpansionProvider {
    fn expand_inline(
        &mut self,
        instruction: &SourceInstruction,
        profile: JoltInstructionProfile,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        let inline = instruction
            .row()
            .inline
            .ok_or(ExpansionError::MalformedInstruction(
                "missing inline source metadata",
            ))?;

        if !profile.supports_source(SourceInstructionKind::Inline) {
            return Err(ExpansionError::UnsupportedInstruction);
        }
        let registration = lookup_inline(inline)?;
        if !profile.supports_inline(registration.extension) {
            return Err(ExpansionError::UnsupportedInstruction);
        }

        build_registered_sequence(registration, instruction)
    }
}

/// RISC-V custom instruction that dispatches to a registered inline.
///
/// This is source-only: it never appears in final proving bytecode. Static
/// expansion lowers it through `TracerInlineExpansionProvider`; runtime tracing
/// executes the same materialized final-row sequence and patches advice values
/// into `VirtualAdvice` rows when the registration supplies runtime advice.
///
/// The struct is implemented manually instead of through
/// `declare_riscv_instr!` because dispatch needs the raw opcode, funct3, and
/// funct7 fields.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct INLINE {
    /// 7-bit opcode (bits 6:0 of instruction)
    pub opcode: u32,
    /// 3-bit function selector (bits 14:12 of instruction)
    pub funct3: u32,
    /// 7-bit function selector (bits 31:25 of instruction)
    pub funct7: u32,
    /// Memory address of this instruction
    pub address: u64,
    /// R-format operands (rd, rs1, rs2)
    pub operands: FormatInline,
    /// Tracks remaining virtual instructions (used by tracer)
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}

impl RISCVInstruction for INLINE {
    const MASK: u32 = 0x0000707f;
    const MATCH: u32 = 0x0000002b; // opcode=0x2B (custom-1)

    type Format = FormatInline;
    type RAMAccess = ();

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn source_kind(&self) -> jolt_riscv::SourceInstructionKind {
        jolt_riscv::SourceInstructionKind::Inline
    }

    fn new(word: u32, address: u64, _validate: bool, is_compressed: bool) -> Self {
        Self {
            opcode: word & 0x7f,
            funct3: (word >> 12) & 0x7,
            funct7: (word >> 25) & 0x7f,
            address,
            operands: FormatInline::parse(word),
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed,
        }
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use crate::instruction::format::InstructionFormat;
        use rand::RngCore;
        Self {
            opcode: rng.next_u32() & 0x7f,
            funct3: rng.next_u32() & 0x7,
            funct7: rng.next_u32() & 0x7f,
            address: rng.next_u64(),
            operands: FormatInline::random(rng),
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    fn execute(&self, cpu: &mut Cpu, ram: &mut Self::RAMAccess) {
        self.exec(cpu, ram)
    }
}

impl INLINE {
    /// Inline source instructions cannot execute as one machine step.
    ///
    /// Callers must use `trace`, which expands the source row into ordinary
    /// final instructions and executes those rows instead.
    pub fn exec(&self, _cpu: &mut Cpu, _: &mut <INLINE as RISCVInstruction>::RAMAccess) {
        panic!("Inline instructions must use trace(), not exec()");
    }

    /// Materialize this inline into tracer instructions for runtime execution.
    ///
    /// The returned instructions are produced by the same `jolt-program`
    /// recipe/materializer path used for static bytecode. Keeping runtime trace
    /// generation on that path prevents register-allocation or metadata-stamp
    /// drift between the preprocessed bytecode and executed rows.
    pub fn inline_sequence(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let _ = allocator;
        let source = Instruction::from(*self).source_instruction();
        let mut expansion_allocator = jolt_program::expand::ExpansionAllocator::new();
        let mut provider = TracerInlineExpansionProvider::new();
        jolt_program::expand::expand_instruction_with_provider(
            &source,
            &mut expansion_allocator,
            &mut provider,
            RV64IMAC_JOLT_ALL_INLINES,
        )
        .expect("jolt-program inline expansion failed")
        .into_iter()
        .map(JoltInstructionRow::from)
        .map(|instruction| {
            Instruction::try_from_jolt_instruction_row(instruction)
                .expect("jolt-program inline expansion produced an instruction unknown to tracer")
        })
        .collect()
    }

    #[cfg(any(feature = "test-utils", test))]
    fn inline_sequence_for_test(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let _ = allocator;
        let source = Instruction::from(*self).source_instruction();
        let mut expansion_allocator = jolt_program::expand::ExpansionAllocator::new();
        let mut provider = TestInlineExpansionProvider;
        jolt_program::expand::expand_instruction_with_provider(
            &source,
            &mut expansion_allocator,
            &mut provider,
            RV64IMAC_JOLT_ALL_INLINES,
        )
        .expect("jolt-program inline expansion failed")
        .into_iter()
        .map(JoltInstructionRow::from)
        .map(|instruction| {
            Instruction::try_from_jolt_instruction_row(instruction)
                .expect("jolt-program inline expansion produced an instruction unknown to tracer")
        })
        .collect()
    }

    fn trace_sequence(
        &self,
        cpu: &mut Cpu,
        trace: Option<&mut Vec<Cycle>>,
        mut sequence: Vec<Instruction>,
    ) {
        let reg = find_inline(self.opcode, self.funct3, self.funct7);
        if let Some(mut advice) = (reg.build_advice)(self.operands, cpu) {
            let mut trace = trace;
            for instr in sequence.iter_mut() {
                if let Instruction::VirtualAdvice(va) = instr {
                    va.advice = match advice.pop_front() {
                        Some(val) => val,
                        None => panic!(
                            "Inline advice for opcode={:#04x}, funct3={:#03b}, funct7={:#09b} \
                            did not provide enough values",
                            self.opcode, self.funct3, self.funct7
                        ),
                    };
                }
                instr.trace_raw(cpu, trace.as_deref_mut());
            }
            assert!(
                advice.is_empty(),
                "Inline advice for opcode={:#04x}, funct3={:#03b}, funct7={:#09b} \
                provided too many values",
                self.opcode,
                self.funct3,
                self.funct7
            );
        } else {
            let mut trace = trace;
            for instr in sequence {
                instr.trace_raw(cpu, trace.as_deref_mut());
            }
        }
    }

    #[cfg(any(feature = "test-utils", test))]
    pub fn trace_for_test(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let sequence = self.inline_sequence_for_test(&cpu.vr_allocator);
        self.trace_sequence(cpu, trace, sequence);
    }
}

impl RISCVTrace for INLINE {
    /// Trace the materialized inline sequence and populate runtime advice.
    ///
    /// Advice generation remains tracer-owned because it can inspect `Cpu`.
    /// Static recipe construction only determines where `VirtualAdvice` rows
    /// occur; this method writes the concrete advice values into those rows
    /// immediately before executing them.
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let sequence = self.inline_sequence(&cpu.vr_allocator);
        self.trace_sequence(cpu, trace, sequence);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_INLINE_WORD: u32 = 0xfc00_602b;
    const TEST_INTERNAL_INLINE_WORD: u32 = 0xfa00_602b;

    fn test_sequence(
        mut asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        asm.emit_vshift_i::<jolt_riscv::instructions::VirtualRotriw>(
            5,
            operands.rs1,
            0xffff_ffff_0000_0000,
        );
        asm.finalize()
    }

    fn test_advice(_operands: FormatInline, _cpu: &mut Cpu) -> Option<VecDeque<u64>> {
        None
    }

    inventory::submit! {
        InlineRegistration {
            opcode: 0x2b,
            funct3: 0x6,
            funct7: 0x7e,
            extension: InlineExtension::Sha2,
            name: "TEST_INLINE_PROFILE",
            admissibility: InlineAdmissibility::Public { requirements: &[] },
            build_sequence: test_sequence,
            build_advice: test_advice,
        }
    }

    inventory::submit! {
        InlineRegistration {
            opcode: 0x2b,
            funct3: 0x6,
            funct7: 0x7d,
            extension: InlineExtension::Sha2,
            name: "TEST_INTERNAL_ONLY_INLINE",
            admissibility: InlineAdmissibility::InternalOnly {
                reason: "test registration lacks verifier-visible public safety contract",
            },
            build_sequence: test_sequence,
            build_advice: test_advice,
        }
    }

    #[test]
    fn test_inline_parsing() {
        let word: u32 = 0xffffffab;
        let inline = INLINE::new(word, 0x1000, false, false);

        assert_eq!(inline.opcode, 0x2b);
        assert_eq!(inline.funct3, 0x7);
        assert_eq!(inline.funct7, 0x7f);
        assert_eq!(inline.address, 0x1000);
    }

    #[test]
    fn test_find_inline_panics_for_unregistered() {
        let result = std::panic::catch_unwind(|| find_inline(0x7F, 0x7, 0x7F));
        assert!(result.is_err());
    }

    #[test]
    fn test_is_inline_registered_returns_false_for_unregistered() {
        assert!(!is_inline_registered(0x7F, 0x7, 0x7F));
    }

    #[test]
    fn test_list_registered_inlines_returns_vec() {
        let inlines = list_registered_inlines();
        for ((opcode, funct3, funct7), _name) in &inlines {
            assert!(is_inline_registered(*opcode, *funct3, *funct7));
        }
    }

    #[test]
    fn provider_rejects_unregistered_inline() {
        let mut provider = TracerInlineExpansionProvider::new();
        let instruction = Instruction::from(INLINE::new(0xfe00_7fab, 0x8000_0000, false, false))
            .source_instruction();

        assert!(matches!(
            provider.expand_inline(&instruction, jolt_riscv::RV64IMAC_JOLT_ALL_INLINES,),
            Err(ExpansionError::UnsupportedInstruction)
        ));
    }

    #[test]
    fn provider_rejects_registered_inline_disabled_by_profile() {
        let mut provider = TracerInlineExpansionProvider::new();
        let instruction =
            Instruction::from(INLINE::new(TEST_INLINE_WORD, 0x8000_0000, false, false))
                .source_instruction();

        assert!(matches!(
            provider.expand_inline(&instruction, jolt_riscv::RV64IMAC_JOLT,),
            Err(ExpansionError::UnsupportedInstruction)
        ));
    }

    #[test]
    fn provider_rejects_registered_inline_disabled_by_source_profile() {
        let mut provider = TracerInlineExpansionProvider::new();
        let instruction =
            Instruction::from(INLINE::new(TEST_INLINE_WORD, 0x8000_0000, false, false))
                .source_instruction();
        let profile = JoltInstructionProfile {
            source_extensions: &[],
            inline_extensions: &[InlineExtension::Sha2],
        };

        assert!(matches!(
            provider.expand_inline(&instruction, profile),
            Err(ExpansionError::UnsupportedInstruction)
        ));
    }

    #[test]
    fn provider_rejects_internal_only_inline() {
        let mut provider = TracerInlineExpansionProvider::new();
        let instruction = Instruction::from(INLINE::new(
            TEST_INTERNAL_INLINE_WORD,
            0x8000_0000,
            false,
            false,
        ))
        .source_instruction();

        assert!(matches!(
            provider.expand_inline(&instruction, jolt_riscv::RV64IMAC_JOLT_ALL_INLINES,),
            Err(ExpansionError::InternalOnlyInline {
                name: "TEST_INTERNAL_ONLY_INLINE",
                reason: "test registration lacks verifier-visible public safety contract",
            })
        ));
    }

    #[test]
    fn provider_accepts_registered_inline_enabled_by_profile() {
        let mut provider = TracerInlineExpansionProvider::new();
        let instruction =
            Instruction::from(INLINE::new(TEST_INLINE_WORD, 0x8000_0000, false, false))
                .source_instruction();

        assert!(provider
            .expand_inline(&instruction, jolt_riscv::RV64IMAC_JOLT_ALL_INLINES,)
            .is_ok());
    }

    #[test]
    fn trace_uses_program_allocator_for_rd_zero_inline() {
        let inline = INLINE::new(TEST_INLINE_WORD, 0x8000_0000, false, false);
        assert_eq!(inline.operands.rs3, 0);

        let mut cpu = Cpu::new(Box::new(crate::emulator::terminal::DummyTerminal {}));
        let expected_rows: Vec<JoltInstructionRow> = inline
            .inline_sequence(&cpu.vr_allocator)
            .into_iter()
            .map(|instruction| {
                instruction
                    .try_jolt_instruction_row()
                    .expect("test inline must expand to final Jolt rows")
            })
            .collect();

        let mut trace = Vec::new();
        inline.trace(&mut cpu, Some(&mut trace));
        let actual_rows: Vec<JoltInstructionRow> = trace
            .iter()
            .map(|cycle| {
                cycle
                    .instruction()
                    .try_jolt_instruction_row()
                    .expect("test inline trace must contain final Jolt rows")
            })
            .collect();

        assert_eq!(actual_rows, expected_rows);
    }
}
