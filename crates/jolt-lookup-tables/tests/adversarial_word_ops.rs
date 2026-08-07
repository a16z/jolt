//! Adversarial differential tests for the fused RV64 word-arithmetic lookups
//! (review of PR #1750: ADDW/ADDIW/SUBW/MULW/SLLIW fused, SLLW = Pow2W+MULW).
//!
//! Each case is checked against an independent RISC-V spec oracle written from
//! the ISA manual (not from Jolt code), along three legs:
//!
//! 1. `to_lookup_output` == spec (claimed witness output)
//! 2. table ∘ index == spec (what the proof system enforces:
//!    `materialize_entry(to_lookup_index)`; `materialize_entry` ≡ MLE on the
//!    hypercube per the table's `mle_full_hypercube`/`mle_random` tests)
//! 3. index == R1CS operand formula (add: l+r, sub: l−r+2^64, mul: l·r)
//!
//! Plus a tracer CPU differential over the same corpus.
//!
//! Corpus targets the boundaries the uniform fuzz tests only hit by luck:
//! bit-31 carries/borrows, sums crossing 2^32/2^64, negative MULW products,
//! high-garbage upper words, shamt ∈ {0,...,31}, SLLW shift amounts ≥ 32,
//! and rs1 == rs2 register aliasing.

use jolt_lookup_tables::{InstructionLookupTable, LookupQuery, LookupTableKind};
use jolt_riscv::instructions::{AddW, AddiW, MulIW, MulW, SubW};
use tracer::emulator::{cpu::Cpu, terminal::DummyTerminal};
use tracer::instruction::format::format_i::{FormatI, RegisterStateFormatI};
use tracer::instruction::format::format_r::{FormatR, RegisterStateFormatR};
use tracer::instruction::{
    addiw::ADDIW, addw::ADDW, mulw::MULW, subw::SUBW, virtual_muliw::VirtualMULIW, RISCVCycle,
    RISCVTrace,
};

const XLEN: usize = 64;
const TWO_64: u128 = 1u128 << 64;

// ---------- Independent RISC-V spec oracle (RV64I/M, word ops) ----------

fn sext32(v: u32) -> u64 {
    v as i32 as i64 as u64
}
fn spec_addw(x: u64, y: u64) -> u64 {
    sext32(x.wrapping_add(y) as u32)
}
fn spec_subw(x: u64, y: u64) -> u64 {
    sext32(x.wrapping_sub(y) as u32)
}
fn spec_mulw(x: u64, y: u64) -> u64 {
    sext32(x.wrapping_mul(y) as u32)
}
fn spec_addiw(x: u64, imm12: i16) -> u64 {
    sext32(x.wrapping_add(imm12 as i64 as u64) as u32)
}
fn spec_slliw(x: u64, shamt: u32) -> u64 {
    sext32((x as u32) << shamt)
}
fn spec_sllw(x: u64, y: u64) -> u64 {
    sext32((x as u32) << (y & 31))
}

// ---------- Boundary corpus ----------

const B: &[u64] = &[
    0,
    1,
    2,
    3,
    0x3fff_ffff,
    0x4000_0000, // carry into bit 31
    0x7fff_fffe,
    0x7fff_ffff, // word-positive max
    0x8000_0000, // word sign bit
    0x8000_0001,
    0xffff_fffe,
    0xffff_ffff, // carry out of bit 31
    0x1_0000_0000,
    0x1_0000_0001,
    0x1_7fff_ffff,
    0x1_8000_0000,
    0x7fff_ffff_ffff_ffff,
    0x8000_0000_0000_0000,
    0x8000_0000_8000_0000,
    0xdead_beef_0000_0000, // high garbage, low zero
    0xdead_beef_7fff_ffff,
    0xdead_beef_8000_0000,
    0xdead_beef_ffff_ffff,
    0xffff_ffff_0000_0000,
    0xffff_ffff_7fff_ffff,
    0xffff_ffff_8000_0000,
    0xffff_ffff_ffff_fffe,
    u64::MAX,
];

const IMM12: &[i16] = &[-2048, -1366, -1, 0, 1, 2, 1365, 2046, 2047];

// ---------- Cycle constructors ----------

macro_rules! r_cycle_ctor {
    ($fn_name:ident, $instr:ident) => {
        fn $fn_name(x: u64, y: u64, aliased: bool) -> RISCVCycle<$instr> {
            let operands = if aliased {
                FormatR {
                    rd: 5,
                    rs1: 6,
                    rs2: 6,
                }
            } else {
                FormatR {
                    rd: 5,
                    rs1: 6,
                    rs2: 7,
                }
            };
            RISCVCycle {
                instruction: $instr {
                    address: 0,
                    operands,
                    virtual_sequence_remaining: None,
                    is_first_in_sequence: false,
                    is_compressed: false,
                },
                register_state: RegisterStateFormatR {
                    rd: (0, 0),
                    rs1: x,
                    rs2: y,
                },
                ram_access: Default::default(),
            }
        }
    };
}

macro_rules! i_cycle_ctor {
    ($fn_name:ident, $instr:ident) => {
        fn $fn_name(x: u64, imm: u64) -> RISCVCycle<$instr> {
            RISCVCycle {
                instruction: $instr {
                    address: 0,
                    operands: FormatI { rd: 5, rs1: 6, imm },
                    virtual_sequence_remaining: None,
                    is_first_in_sequence: false,
                    is_compressed: false,
                },
                register_state: RegisterStateFormatI { rd: (0, 0), rs1: x },
                ram_access: Default::default(),
            }
        }
    };
}

r_cycle_ctor!(addw_cycle, ADDW);
r_cycle_ctor!(subw_cycle, SUBW);
r_cycle_ctor!(mulw_cycle, MULW);
i_cycle_ctor!(addiw_cycle, ADDIW);
i_cycle_ctor!(muliw_cycle, VirtualMULIW);

// ---------- Trace differential helper ----------

fn trace_r<T>(instr: &T, x: u64, y: u64, aliased: bool) -> u64
where
    T: RISCVTrace + tracer::instruction::RISCVInstruction,
    RISCVCycle<T>: Into<tracer::instruction::Cycle>,
{
    let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
    cpu.write_register(6, x as i64);
    if !aliased {
        cpu.write_register(7, y as i64);
    }
    instr.trace(&mut cpu, None);
    cpu.x[5] as u64
}

fn trace_i<T>(instr: &T, x: u64) -> u64
where
    T: RISCVTrace + tracer::instruction::RISCVInstruction,
    RISCVCycle<T>: Into<tracer::instruction::Cycle>,
{
    let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
    cpu.write_register(6, x as i64);
    instr.trace(&mut cpu, None);
    cpu.x[5] as u64
}

// ---------- The three proof-side legs ----------

/// Look up the table associated with a fused kind via its jolt wrapper.
fn table_of<I: Clone + WrapLookup>(instr: &I) -> LookupTableKind<XLEN> {
    instr.clone().wrap_table()
}

/// Maps each tracer instruction to `<JoltWrapper>(instr).lookup_table()`.
trait WrapLookup: Sized {
    fn wrap_table(self) -> LookupTableKind<XLEN>;
}
macro_rules! impl_wrap_lookup {
    ($tracer:ident => $jolt:ident) => {
        impl WrapLookup for $tracer {
            fn wrap_table(self) -> LookupTableKind<XLEN> {
                InstructionLookupTable::<XLEN>::lookup_table(&$jolt(self)).unwrap()
            }
        }
    };
}
impl_wrap_lookup!(ADDW => AddW);
impl_wrap_lookup!(ADDIW => AddiW);
impl_wrap_lookup!(SUBW => SubW);
impl_wrap_lookup!(MULW => MulW);
impl_wrap_lookup!(VirtualMULIW => MulIW);

/// output == spec, table∘index == spec, index == R1CS formula.
fn check_legs<W>(cycle: &W, table: LookupTableKind<XLEN>, spec: u64, index_formula: u128, ctx: &str)
where
    W: LookupQuery<XLEN> + std::fmt::Debug,
{
    let out = LookupQuery::<XLEN>::to_lookup_output(cycle);
    assert_eq!(out, spec, "to_lookup_output != spec: {ctx}: {cycle:?}");

    let index = LookupQuery::<XLEN>::to_lookup_index(cycle);
    assert_eq!(
        table.materialize_entry(index),
        spec,
        "table(index) != spec: {ctx}: {cycle:?}"
    );

    assert_eq!(
        index, index_formula,
        "lookup index != R1CS operand formula: {ctx}: {cycle:?}"
    );

    let (left, right) = LookupQuery::<XLEN>::to_lookup_operands(cycle);
    assert_eq!(left, 0, "left lookup operand must be 0: {ctx}");
    assert_eq!(right, index, "index must equal right lookup operand: {ctx}");
}

#[test]
fn addw_adversarial() {
    for &x in B {
        for &y in B {
            let spec = spec_addw(x, y);
            let cycle = AddW(addw_cycle(x, y, false));
            let (l, r) = LookupQuery::<XLEN>::to_instruction_inputs(&cycle);
            check_legs(
                &cycle,
                table_of(&cycle.0.instruction),
                spec,
                l as u128 + (r as u64) as u128,
                "addw",
            );
            assert_eq!(
                trace_r(&cycle.0.instruction, x, y, false),
                spec,
                "trace addw {x:#x} {y:#x}"
            );
        }
        // rs1 == rs2 aliasing
        let spec = spec_addw(x, x);
        let cycle = AddW(addw_cycle(x, x, true));
        let (l, r) = LookupQuery::<XLEN>::to_instruction_inputs(&cycle);
        check_legs(
            &cycle,
            table_of(&cycle.0.instruction),
            spec,
            l as u128 + (r as u64) as u128,
            "addw aliased",
        );
        assert_eq!(
            trace_r(&cycle.0.instruction, x, x, true),
            spec,
            "trace addw aliased {x:#x}"
        );
    }
}

#[test]
fn subw_adversarial() {
    for &x in B {
        for &y in B {
            let spec = spec_subw(x, y);
            let cycle = SubW(subw_cycle(x, y, false));
            let (l, r) = LookupQuery::<XLEN>::to_instruction_inputs(&cycle);
            check_legs(
                &cycle,
                table_of(&cycle.0.instruction),
                spec,
                l as u128 + TWO_64 - (r as u64) as u128,
                "subw",
            );
            assert_eq!(
                trace_r(&cycle.0.instruction, x, y, false),
                spec,
                "trace subw {x:#x} {y:#x}"
            );
        }
        let spec = spec_subw(x, x);
        let cycle = SubW(subw_cycle(x, x, true));
        let (l, r) = LookupQuery::<XLEN>::to_instruction_inputs(&cycle);
        check_legs(
            &cycle,
            table_of(&cycle.0.instruction),
            spec,
            l as u128 + TWO_64 - (r as u64) as u128,
            "subw aliased",
        );
        assert_eq!(
            trace_r(&cycle.0.instruction, x, x, true),
            spec,
            "trace subw aliased {x:#x}"
        );
    }
}

#[test]
fn mulw_adversarial() {
    for &x in B {
        for &y in B {
            let spec = spec_mulw(x, y);
            let cycle = MulW(mulw_cycle(x, y, false));
            let (l, r) = LookupQuery::<XLEN>::to_instruction_inputs(&cycle);
            check_legs(
                &cycle,
                table_of(&cycle.0.instruction),
                spec,
                l as u128 * (r as u64) as u128,
                "mulw",
            );
            assert_eq!(
                trace_r(&cycle.0.instruction, x, y, false),
                spec,
                "trace mulw {x:#x} {y:#x}"
            );
        }
        let spec = spec_mulw(x, x);
        let cycle = MulW(mulw_cycle(x, x, true));
        let (l, r) = LookupQuery::<XLEN>::to_instruction_inputs(&cycle);
        check_legs(
            &cycle,
            table_of(&cycle.0.instruction),
            spec,
            l as u128 * (r as u64) as u128,
            "mulw aliased",
        );
        assert_eq!(
            trace_r(&cycle.0.instruction, x, x, true),
            spec,
            "trace mulw aliased {x:#x}"
        );
    }
}

#[test]
fn addiw_adversarial() {
    for &x in B {
        for &imm12 in IMM12 {
            let imm = imm12 as i64 as u64; // decoder produces the sign-extended pattern
            let spec = spec_addiw(x, imm12);
            let cycle = AddiW(addiw_cycle(x, imm));
            let (l, r) = LookupQuery::<XLEN>::to_instruction_inputs(&cycle);
            check_legs(
                &cycle,
                table_of(&cycle.0.instruction),
                spec,
                l as u128 + (r as u64) as u128,
                "addiw",
            );
            assert_eq!(
                trace_i(&cycle.0.instruction, x),
                spec,
                "trace addiw {x:#x} {imm12}"
            );
        }
    }
}

/// SLLIW now expands (in jolt-program) to one VirtualMULIW row with
/// imm = 1 << (shamt & 0x1f); check that composition for every shamt.
#[test]
fn slliw_via_muliw_adversarial() {
    for &x in B {
        for shamt in 0u32..32 {
            let spec = spec_slliw(x, shamt);
            // Expander semantics (crates/jolt-program/src/expand/shifts/slliw.rs):
            let imm = 1u64 << (shamt & 0x1f);
            let cycle = MulIW(muliw_cycle(x, imm));
            let (l, r) = LookupQuery::<XLEN>::to_instruction_inputs(&cycle);
            assert_eq!(
                r,
                1i128 << shamt,
                "muliw row imm != 2^shamt: {x:#x} {shamt}"
            );
            check_legs(
                &cycle,
                table_of(&cycle.0.instruction),
                spec,
                l as u128 * (r as u64) as u128,
                "slliw=muliw",
            );
            assert_eq!(
                trace_i(&cycle.0.instruction, x),
                spec,
                "trace muliw {x:#x} {shamt}"
            );
        }
    }
}

/// SLLW = VirtualPow2W (rs2 & 0x1f) followed by fused MULW: simulate the
/// two-row sequence and check the composition against the SLLW spec,
/// including shift amounts ≥ 32 and full-garbage rs2.
#[test]
fn sllw_composition_adversarial() {
    let shift_corpus: Vec<u64> = B
        .iter()
        .copied()
        .chain([31, 32, 33, 63, 64, 65, 0x1f, 0x20, 0x3f, 0x40])
        .collect();
    for &x in B {
        for &y in &shift_corpus {
            // Row 1: VirtualPow2W spec (tracer/src/instruction/virtual_pow2_w.rs): 2^(rs2 % 32)
            let v = 1u64 << (y % 32);
            // Row 2: fused MULW on (x, v)
            let spec = spec_sllw(x, y);
            let cycle = MulW(mulw_cycle(x, v, false));
            let (l, r) = LookupQuery::<XLEN>::to_instruction_inputs(&cycle);
            check_legs(
                &cycle,
                table_of(&cycle.0.instruction),
                spec,
                l as u128 * (r as u64) as u128,
                "sllw=pow2w+mulw",
            );
        }
    }
}

/// High-garbage invariance: the word result must depend only on the low 32
/// bits of each operand — perturbing upper words must not change any leg.
#[test]
fn upper_word_garbage_invariance() {
    let garbage = [0u64, 1, 0xdead_beef, 0xffff_ffff];
    for &x in B {
        for &y in B {
            let base_add = spec_addw(x, y);
            let base_sub = spec_subw(x, y);
            let base_mul = spec_mulw(x, y);
            for &gx in &garbage {
                for &gy in &garbage {
                    let xg = (x & 0xffff_ffff) | (gx << 32);
                    let yg = (y & 0xffff_ffff) | (gy << 32);
                    // Same low words ⇒ same spec result; verify all legs track it.
                    if (x & 0xffff_ffff) == (xg & 0xffff_ffff)
                        && (y & 0xffff_ffff) == (yg & 0xffff_ffff)
                    {
                        let add = AddW(addw_cycle(xg, yg, false));
                        assert_eq!(LookupQuery::<XLEN>::to_lookup_output(&add), base_add);
                        let table = table_of(&add.0.instruction);
                        assert_eq!(
                            table.materialize_entry(LookupQuery::<XLEN>::to_lookup_index(&add)),
                            base_add
                        );
                        let sub = SubW(subw_cycle(xg, yg, false));
                        assert_eq!(LookupQuery::<XLEN>::to_lookup_output(&sub), base_sub);
                        let mul = MulW(mulw_cycle(xg, yg, false));
                        assert_eq!(LookupQuery::<XLEN>::to_lookup_output(&mul), base_mul);
                        assert_eq!(
                            table_of(&mul.0.instruction)
                                .materialize_entry(LookupQuery::<XLEN>::to_lookup_index(&mul)),
                            base_mul
                        );
                    }
                }
            }
        }
    }
}
