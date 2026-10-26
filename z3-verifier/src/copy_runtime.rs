#![cfg(test)]
//! Proves the guest `memcpy` and `memmove` in `jolt-platform/src/memory.S`
//! correct for every length, address pair, overlap, and memory content.
//!
//! The test assembles memory.S with clang, keeping its `.L` labels, links it
//! with rust-lld, and decodes it with `jolt_program`. A symbolic executor runs
//! the decoded instructions over 64-bit registers and byte memory, forking at
//! each branch the path does not decide. Every loop that can iterate without
//! bound is cut at its head by an invariant ([`Loop`]); other loops unroll
//! within [`STEP_BUDGET`]. Each path segment, from a routine entry or a loop
//! head, must establish where it ends:
//!
//! * a return: `a0 == dst`, sp, gp, tp, and s0-s11 intact, and memory equal to
//!   the initial memory with `[dst, dst + len)` replaced by the initial
//!   `[src, src + len)`;
//! * a tail call into the original routine: arguments and memory untouched;
//! * a loop head: that head's invariant. On a back edge the destination cursor,
//!   which the invariant bounds, must strictly progress; any other segment
//!   reaches a later loop head, so every execution terminates.
//!
//! Every load must lie inside the source and every store inside the
//! destination, naturally aligned. Invariants state memory exactly, as a
//! window of copied bytes over the initial memory, so obligations stay
//! quantifier-free and overlap needs no separate argument: loading a source
//! byte that an earlier store overwrote breaks the memory goal.
//!
//! The encoding keeps the obligations within Z3's reach: memory is a list of
//! byte writes expanded into conditionals ([`Proof::read`]), addresses share a
//! base variable so Z3's rewriter decides most aliasing ([`Proof::new`]), and
//! loop states substitute their invariant's equations ([`Proof::loop_state`]).
//!
//! Assumptions: `dst`, `src`, and `len` are below 2^62, so no address wraps,
//! and `memcpy`'s operands are disjoint or identical.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use jolt_program::image::decode_elf;
use jolt_riscv::{SourceInstruction, SourceInstructionKind as Kind, RV64IMAC_JOLT};
use object::{File, Object, ObjectSymbol};
use z3::ast::{Array, Ast, Bool, BV};
use z3::{Model, Params, SatResult, Solver, Sort, Tactic};

use crate::Z3_RANDOM_SEED;

/// Budget for proving a segment's goals together before proving them singly.
const QUICK_TIMEOUT_MS: u32 = 5_000;
/// Budget for any single check. Goals usually take milliseconds and the
/// slowest about 20 s here; Z3 budgets are wall-clock, so this leaves room for
/// slower CI runners.
const GOAL_TIMEOUT_MS: u32 = 120_000;

const MEMORY_S: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../jolt-platform/src/memory.S");
/// Where the linker script places the original routines the wrappers fall back to.
const REAL_MEMCPY: u64 = 0x9000_0000;
const REAL_MEMMOVE: u64 = 0x9000_1000;
/// Bounds one segment; the unrolled loops (edge peeling, tails) take far fewer.
const STEP_BUDGET: u32 = 4096;

const RA: usize = 1;
const T0: usize = 5;
const T2: usize = 7;
const A0: usize = 10;
const A1: usize = 11;
const A2: usize = 12;
const A3: usize = 13;
const A4: usize = 14;
const A5: usize = 15;
const A7: usize = 17;
const T3: usize = 28;
/// sp, gp, tp, and s0-s11.
const PRESERVED: [usize; 15] = [2, 3, 4, 8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27];

/// Loop heads of each wrapper's equal-offset path, as `.L<name>_<wrapper>`.
const WRAPPER_LOOPS: [(&str, Loop); 6] = [
    ("forward_large", Loop::FwdBlocks(256)),
    ("forward_block", Loop::FwdBlocks(80)),
    ("forward_word", Loop::FwdWords),
    ("backward_large", Loop::BwdBlocks(256)),
    ("backward_block", Loop::BwdBlocks(80)),
    ("backward_word", Loop::BwdWords),
];
/// Loop heads of `__jolt_copy_misaligned`, as `.L<name>`.
const LEAF_LOOPS: [(&str, Loop); 6] = [
    ("mis_forward_bulk", Loop::MisFwd(136)),
    ("mis_forward_word", Loop::MisFwd(16)),
    ("mis_forward_tail", Loop::FwdTail),
    ("mis_backward_bulk", Loop::MisBwd(128)),
    ("mis_backward_word", Loop::MisBwd(8)),
    ("mis_backward_tail", Loop::BwdTail),
];

/// Loop-head invariants. Every head also keeps `a0 == dst`, ra and the preserved
/// registers, the cursors `a1 - src == a7 - dst` with `a7 - dst <= len`, the
/// copy direction, and memory with the copied window replaced: `[dst, a7)` for
/// forward loops, `[a7, dst + len)` for backward ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Loop {
    /// Equal-offset blocks: aligned cursors, `a2` bytes left, `a2 >= t0 == bound`.
    FwdBlocks(u64),
    BwdBlocks(u64),
    /// Equal-offset words: aligned cursors 8 to `len` bytes below the word
    /// bound `t0`, then `a2 < 8` tail bytes.
    FwdWords,
    BwdWords,
    /// Differing offsets: `a5 = a1 % 8 != 0` and `t2 = a1 - a5`, the shifts
    /// `a3 = 8 * a5` and `a4 = 64 - a3`, and the bytes of `t3`, the aligned word
    /// at `t2`, that are still to be copied equal the initial memory. Forward
    /// loops continue while `a2 >= t0 == bound - a5`, backward ones while
    /// `a2 >= t0 == a5 + bound`.
    MisFwd(u64),
    MisBwd(u64),
    /// Byte remainder of the leaf: `a2` bytes left.
    FwdTail,
    BwdTail,
}

impl Loop {
    fn direction(self) -> Direction {
        if self.forward() {
            Direction::Forward
        } else {
            Direction::Backward
        }
    }

    fn forward(self) -> bool {
        matches!(
            self,
            Loop::FwdBlocks(_) | Loop::FwdWords | Loop::MisFwd(_) | Loop::FwdTail
        )
    }
}

/// Which side of `dst` the source lies on, as far as a proof knows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    Any,
    /// `src >= dst`, as in every forward loop.
    Forward,
    /// `src < dst`, as in every backward loop.
    Backward,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Routine {
    Memcpy,
    Memmove,
}

impl Routine {
    fn symbol(self) -> &'static str {
        match self {
            Routine::Memcpy => "__wrap_memcpy",
            Routine::Memmove => "__wrap_memmove",
        }
    }

    fn fallback(self) -> u64 {
        match self {
            Routine::Memcpy => REAL_MEMCPY,
            Routine::Memmove => REAL_MEMMOVE,
        }
    }
}

struct Program {
    instructions: BTreeMap<u64, SourceInstruction>,
    symbols: BTreeMap<String, u64>,
}

impl Program {
    /// Assembled once per test process.
    fn get() -> &'static Program {
        static PROGRAM: OnceLock<Program> = OnceLock::new();
        PROGRAM.get_or_init(Program::assemble)
    }

    fn assemble() -> Self {
        let dir = std::env::temp_dir().join(format!("z3-copy-runtime-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (object, elf, script) = (
            dir.join("memory.o"),
            dir.join("memory.elf"),
            dir.join("link.ld"),
        );
        std::fs::write(
            &script,
            format!(
                "ENTRY(__wrap_memcpy)\nSECTIONS {{ . = 0x80000000; .text : {{ *(.text .text.*) }} }}\n\
                 __real_memcpy = {REAL_MEMCPY:#x};\n__real_memmove = {REAL_MEMMOVE:#x};\n"
            ),
        )
        .unwrap();
        // -Wa,-L keeps the .L labels the loop invariants are attached to.
        run(Command::new("clang")
            .args([
                "--target=riscv64-unknown-elf",
                "-march=rv64imac",
                "-mabi=lp64",
                "-Wa,-L",
                "-c",
            ])
            .arg(MEMORY_S)
            .arg("-o")
            .arg(&object));
        run(Command::new(rust_lld())
            .args(["-flavor", "gnu", "-T"])
            .arg(&script)
            .arg(&object)
            .arg("-o")
            .arg(&elf));
        let bytes = std::fs::read(&elf).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        let instructions = decode_elf(&bytes, RV64IMAC_JOLT)
            .unwrap()
            .instructions
            .into_iter()
            .map(|instruction| (instruction.row().address as u64, instruction))
            .collect();
        let file = File::parse(&*bytes).unwrap();
        let symbols = file
            .symbols()
            .filter_map(|symbol| Some((symbol.name().ok()?.to_string(), symbol.address())))
            .collect();
        Program {
            instructions,
            symbols,
        }
    }

    fn symbol(&self, name: &str) -> u64 {
        *self
            .symbols
            .get(name)
            .unwrap_or_else(|| panic!("memory.S no longer defines {name}; update the loop table"))
    }
}

fn rust_lld() -> PathBuf {
    let output = Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .unwrap();
    let rustlib =
        PathBuf::from(String::from_utf8(output.stdout).unwrap().trim()).join("lib/rustlib");
    std::fs::read_dir(&rustlib)
        .unwrap()
        .map(|entry| entry.unwrap().path().join("bin/rust-lld"))
        .find(|path| path.exists())
        .unwrap_or_else(|| panic!("no rust-lld under {}", rustlib.display()))
}

fn run(command: &mut Command) {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("cannot run {command:?}: {error}"));
    assert!(
        output.status.success(),
        "{command:?} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[derive(Clone)]
struct State {
    pc: u64,
    regs: Vec<BV>,
    mem: Memory,
    path: Vec<Bool>,
    /// Memory-safety conditions met by this path so far, by description.
    obligations: Vec<(String, Bool)>,
    steps: u32,
    stored: bool,
    /// The destination cursor when a loop-head segment began.
    start: Option<BV>,
    /// Branch PCs where this path forked, for counterexample reports.
    trail: Vec<u64>,
}

/// Byte writes over a base memory, oldest first.
#[derive(Clone)]
struct Memory {
    base: Base,
    writes: Vec<(BV, BV)>,
}

#[derive(Clone)]
enum Base {
    Initial,
    /// The initial memory with destination bytes `[lo, hi)` holding the source
    /// bytes they copy.
    Copied {
        lo: BV,
        hi: BV,
    },
}

impl Memory {
    fn initial() -> Self {
        Memory {
            base: Base::Initial,
            writes: Vec::new(),
        }
    }

    fn store(&mut self, addr: &BV, value: &BV, width: u64) {
        for i in 0..width {
            let low = 8 * i as u32;
            self.writes.push((
                (addr + i).simplify(),
                value.extract(low + 7, low).simplify(),
            ));
        }
    }
}

enum Flow {
    Next(u64),
    Branch(Bool, u64, u64),
    Jump(BV),
}

/// One call's symbols and the loop heads the executor cuts at.
struct Proof<'a> {
    program: &'a Program,
    loops: BTreeMap<u64, (String, Loop)>,
    routine: Option<Routine>,
    /// With a known direction, `src` and `dst` share one base variable and
    /// differ by a 62-bit gap, so Z3's rewriter cancels the base out of address
    /// comparisons and folds impossible distances away.
    dst: BV,
    src: BV,
    len: BV,
    mem0: Array,
    /// Register values at the call.
    entry: Vec<BV>,
    assumptions: Vec<Bool>,
}

impl<'a> Proof<'a> {
    /// `routine` is `None` for the shared leaf, which must hold for either caller.
    fn new(program: &'a Program, routine: Option<Routine>, direction: Direction) -> Self {
        let mut loops = BTreeMap::new();
        for wrapper in [Routine::Memcpy, Routine::Memmove] {
            for (name, lp) in WRAPPER_LOOPS {
                let label = format!(".L{name}_{}", wrapper.symbol());
                loops.insert(program.symbol(&label), (label, lp));
            }
        }
        for (name, lp) in LEAF_LOOPS {
            let label = format!(".L{name}");
            loops.insert(program.symbol(&label), (label, lp));
        }
        let word = |name: &str| BV::new_const(name, 64);
        // Quantities below 2^62 are zero-extended 62-bit variables, so the
        // solver gets their clear top bits, and hence no-overflow, for free.
        let small = |name: &str| BV::new_const(name, 62).zero_ext(2);
        let (dst, src) = match direction {
            Direction::Any => (small("dst"), small("src")),
            Direction::Forward => (small("dst"), small("dst") + small("gap")),
            Direction::Backward => (small("src") + small("gap"), small("src")),
        };
        let len = small("len");
        let limit = BV::from_u64(1 << 62, 64);
        let mut assumptions = vec![dst.bvult(&limit), src.bvult(&limit), len.bvult(&limit)];
        if routine == Some(Routine::Memcpy) {
            assumptions.push((&dst + &len).bvule(&src) | (&src + &len).bvule(&dst) | dst.eq(&src));
        }
        Proof {
            program,
            loops,
            routine,
            mem0: Array::new_const("mem0", &Sort::bitvector(64), &Sort::bitvector(8)),
            entry: (0..32).map(|i| word(&format!("entry_x{i}"))).collect(),
            dst,
            src,
            len,
            assumptions,
        }
    }

    fn entry_state(&self, routine: Routine) -> State {
        let mut regs = self.entry.clone();
        regs[0] = BV::from_u64(0, 64);
        regs[A0] = self.dst.clone();
        regs[A1] = self.src.clone();
        regs[A2] = self.len.clone();
        State {
            pc: self.program.symbol(routine.symbol()),
            regs,
            mem: Memory::initial(),
            path: Vec::new(),
            obligations: Vec::new(),
            steps: 0,
            stored: false,
            start: None,
            trail: Vec::new(),
        }
    }

    /// An arbitrary state satisfying the invariant at `pc`. For the leaf's
    /// shifting loops, `offset` fixes `a5` so the shifts are constants.
    fn loop_state(&self, pc: u64, offset: Option<u64>) -> State {
        let (label, lp) = &self.loops[&pc];
        let c = |v: u64| BV::from_u64(v, 64);
        let mut regs: Vec<BV> = (0..32)
            .map(|i| BV::new_const(format!("{label}_x{i}"), 64))
            .collect();
        regs[0] = c(0);
        regs[A0] = self.dst.clone();
        regs[RA] = self.entry[RA].clone();
        for register in PRESERVED {
            regs[register] = self.entry[register].clone();
        }
        // Both cursors sit `done` bytes into their buffers. Word loops split
        // the uncopied bytes into `words` up to the bound t0 and `tail` beyond
        // it; substituting these equations leaves Z3 only small inequalities.
        // The invariant bounds each of these by `len`.
        let fresh = |name: &str| BV::new_const(format!("{label}_{name}"), 62).zero_ext(2);
        let (done, words, tail) = (fresh("done"), fresh("words"), fresh("tail"));
        let done = match lp {
            Loop::FwdWords => {
                regs[T0] = &self.src + &done + &words;
                regs[A2] = &self.len - &done - &words;
                done
            }
            Loop::BwdWords => {
                regs[T0] = &self.src + &tail;
                regs[A2] = tail.clone();
                &tail + &words
            }
            _ => done,
        };
        regs[A7] = &self.dst + &done;
        regs[A1] = &self.src + &done;
        if !matches!(lp, Loop::FwdWords | Loop::BwdWords) {
            regs[A2] = if lp.forward() {
                &self.len - &done
            } else {
                done
            };
        }
        match (*lp, offset) {
            (Loop::FwdBlocks(bound) | Loop::BwdBlocks(bound), _) => regs[T0] = c(bound),
            (Loop::MisFwd(bound) | Loop::MisBwd(bound), Some(k)) => {
                regs[A5] = c(k);
                regs[A3] = c(8 * k);
                regs[A4] = c(64 - 8 * k);
                regs[T2] = &regs[A1] - k;
                regs[T0] = c(if lp.forward() { bound - k } else { bound + k });
            }
            _ => {}
        }
        let mem = Memory {
            base: self.copied(*lp, &regs[A7]),
            writes: Vec::new(),
        };
        let regs_a7 = regs[A7].clone();
        let path = self
            .invariant(*lp, &regs, offset)
            .into_iter()
            .map(|(_, holds)| holds)
            .collect();
        State {
            pc,
            regs,
            mem,
            path,
            obligations: Vec::new(),
            steps: 0,
            stored: true,
            start: Some(regs_a7),
            trail: Vec::new(),
        }
    }

    fn copied(&self, lp: Loop, cursor: &BV) -> Base {
        let (lo, hi) = if lp.forward() {
            (self.dst.clone(), cursor.clone())
        } else {
            (cursor.clone(), &self.dst + &self.len)
        };
        Base::Copied { lo, hi }
    }

    /// The byte at `addr`. Writes expand into conditionals here, rather than
    /// into Z3 array stores, so that the rewriter sees each address comparison
    /// and cancels the terms both sides share; most comparisons fold away.
    fn read(&self, mem: &Memory, addr: &BV) -> BV {
        let base = self.base_read(&mem.base, addr);
        mem.writes.iter().fold(base, |value, (at, byte)| {
            let hit = addr.eq(at).simplify();
            match hit.as_bool() {
                Some(true) => byte.clone(),
                Some(false) => value,
                None => hit.ite(byte, &value),
            }
        })
    }

    fn base_read(&self, base: &Base, addr: &BV) -> BV {
        let initial = |at: &BV| self.mem0.select(at).as_bv().unwrap();
        match base {
            Base::Initial => initial(addr),
            Base::Copied { lo, hi } => {
                let inside = (addr - lo).bvult(&(hi - lo)).simplify();
                let copied = initial(&(addr - &self.dst + &self.src).simplify());
                match inside.as_bool() {
                    Some(true) => copied,
                    Some(false) => initial(addr),
                    None => inside.ite(&copied, &initial(addr)),
                }
            }
        }
    }

    /// The zero-extended little-endian value of `width` bytes at `addr`.
    fn load(&self, mem: &Memory, addr: &BV, width: u64) -> BV {
        let byte = |i: u64| self.read(mem, &(addr + i).simplify());
        let value = (1..width).fold(byte(0), |value, i| byte(i).concat(&value));
        value.zero_ext(64 - 8 * width as u32)
    }

    /// Goals that `mem` equals `expected` at every address. When the writes hit
    /// pairwise distinct addresses, as they do throughout memory.S, each
    /// written byte and the unwritten remainder are separate goals; otherwise
    /// one goal compares the two at an unconstrained probe address. The probe
    /// is an offset from the first write, so comparing it with each write
    /// rewrites to comparing the offset with a constant.
    fn memory_goals(&self, mem: &Memory, expected: &Base) -> Vec<(String, Bool)> {
        let anchor = mem.writes.first().map_or(&self.dst, |(at, _)| at);
        let offset = BV::new_const("probe_offset", 64);
        let probe = anchor + &offset;
        // Write offsets from the anchor, when every one rewrites to a constant.
        let constant: Option<Vec<i64>> = mem
            .writes
            .iter()
            .map(|(at, _)| (at - anchor).simplify().as_u64().map(|v| v as i64))
            .collect();
        let distinct = match &constant {
            Some(offsets) => {
                let mut sorted = offsets.clone();
                sorted.sort_unstable();
                sorted.windows(2).all(|pair| pair[0] != pair[1])
            }
            None => mem.writes.iter().enumerate().all(|(i, (at, _))| {
                mem.writes[i + 1..]
                    .iter()
                    .all(|(other, _)| at.eq(other).simplify().as_bool() == Some(false))
            }),
        };
        if !distinct {
            let same = self.read(mem, &probe).eq(self.base_read(expected, &probe));
            return vec![("memory".to_string(), same)];
        }
        let mut goals: Vec<(String, Bool)> = mem
            .writes
            .iter()
            .enumerate()
            .map(|(k, (at, byte))| {
                (
                    format!("memory: store {k}"),
                    byte.eq(self.base_read(expected, at)),
                )
            })
            .collect();
        // A contiguous run of writes is missed by one range test rather than
        // one disequality per write, which Z3 handles far better.
        let unwritten = match constant {
            Some(mut offsets) if !offsets.is_empty() => {
                offsets.sort_unstable();
                let (lo, hi) = (offsets[0], offsets[offsets.len() - 1]);
                if hi.checked_sub(lo) == Some(offsets.len() as i64 - 1) {
                    (&offset - lo as u64).bvuge(BV::from_u64(offsets.len() as u64, 64))
                } else {
                    Bool::and(
                        &mem.writes
                            .iter()
                            .map(|(at, _)| probe.eq(at).not())
                            .collect::<Vec<_>>(),
                    )
                }
            }
            _ => Bool::and(
                &mem.writes
                    .iter()
                    .map(|(at, _)| probe.eq(at).not())
                    .collect::<Vec<_>>(),
            ),
        };
        let same = self
            .base_read(&mem.base, &probe)
            .eq(self.base_read(expected, &probe));
        goals.push((
            "memory: unwritten bytes".to_string(),
            unwritten.implies(&same),
        ));
        goals
    }

    /// The initial little-endian doubleword at `addr`.
    fn initial_word(&self, addr: &BV) -> BV {
        self.load(&Memory::initial(), addr, 8)
    }

    fn calling_convention(&self, regs: &[BV]) -> Vec<(String, Bool)> {
        let mut goals = vec![("a0 == dst".to_string(), regs[A0].eq(&self.dst))];
        for register in std::iter::once(RA).chain(PRESERVED) {
            goals.push((
                format!("x{register} preserved"),
                regs[register].eq(&self.entry[register]),
            ));
        }
        goals
    }

    /// The invariant at a loop head, apart from memory, which [`Self::copied`]
    /// states exactly. For the leaf's shifting loops, `offset` states it for
    /// one value of `a5`, with constant shifts.
    fn invariant(&self, lp: Loop, regs: &[BV], offset: Option<u64>) -> Vec<(String, Bool)> {
        let r = |i: usize| &regs[i];
        let c = |v: u64| BV::from_u64(v, 64);
        let done = r(A7) - &self.dst;
        let aligned = |i: usize| (r(i) & 7u64).eq(0u64);
        let mut goals = self.calling_convention(regs);
        let mut push = |name: &str, holds: Bool| goals.push((format!("{lp:?}: {name}"), holds));
        push("cursors advance together", (r(A1) - &self.src).eq(&done));
        push("destination cursor in range", done.bvule(&self.len));
        if lp.forward() {
            push("forward direction", self.src.bvuge(&self.dst));
        } else {
            push("backward direction", self.src.bvult(&self.dst));
        }
        let remaining = if lp.forward() {
            &self.len - &done
        } else {
            done.clone()
        };
        match lp {
            Loop::FwdBlocks(bound) | Loop::BwdBlocks(bound) => {
                push("a2 counts the remaining bytes", r(A2).eq(&remaining));
                push("a2 >= t0", r(A2).bvuge(c(bound)));
                push("t0 == bound", r(T0).eq(c(bound)));
                push("source cursor aligned", aligned(A1));
                push("destination cursor aligned", aligned(A7));
            }
            Loop::FwdWords => {
                let words = r(T0) - r(A1);
                push("a1 < t0", r(A1).bvult(r(T0)));
                push("whole words to t0", (&words & 7u64).eq(0u64));
                push(
                    "8 <= t0 - a1 <= len",
                    words.bvuge(c(8)) & words.bvule(&self.len),
                );
                push(
                    "a2 tail bytes after t0",
                    (r(T0) + r(A2)).eq(&(&self.src + &self.len)),
                );
                push("a2 < 8", r(A2).bvule(c(7)));
                push("source cursor aligned", aligned(A1));
                push("destination cursor aligned", aligned(A7));
            }
            Loop::BwdWords => {
                let words = r(A1) - r(T0);
                push("t0 < a1", r(T0).bvult(r(A1)));
                push("whole words to t0", (&words & 7u64).eq(0u64));
                push(
                    "8 <= a1 - t0 <= len",
                    words.bvuge(c(8)) & words.bvule(&self.len),
                );
                push("a2 tail bytes before t0", r(T0).eq(&(&self.src + r(A2))));
                push("a2 < 8", r(A2).bvule(c(7)));
                push("source cursor aligned", aligned(A1));
                push("destination cursor aligned", aligned(A7));
            }
            Loop::MisFwd(bound) | Loop::MisBwd(bound) => {
                let (a5, a3, a4) = match offset {
                    Some(k) => (c(k), c(8 * k), c(64 - 8 * k)),
                    None => (r(A5).clone(), r(A5) << 3u64, c(64) - (r(A5) << 3u64)),
                };
                push("a5 == a1 % 8", r(A5).eq(&(r(A1) & 7u64)));
                push("a5 is this case's offset", r(A5).eq(&a5));
                push("offsets differ", a5.eq(0u64).not());
                push("t2 == a1 - a5", r(T2).eq(&(r(A1) - &a5)));
                push("a3 == 8 * a5", r(A3).eq(&a3));
                push("a4 == 64 - a3", r(A4).eq(&a4));
                push("a2 counts the remaining bytes", r(A2).eq(&remaining));
                push("a2 >= t0", r(A2).bvuge(r(T0)));
                push("destination cursor aligned", aligned(A7));
                let initial = self.initial_word(r(T2));
                if lp.forward() {
                    push("t0 == bound - a5", r(T0).eq(&(c(bound) - &a5)));
                    push(
                        "t3 holds the next source bytes",
                        r(T3).bvlshr(&a3).eq(initial.bvlshr(&a3)),
                    );
                } else {
                    push("t0 == a5 + bound", r(T0).eq(&(&a5 + c(bound))));
                    push(
                        "t3 holds the next source bytes",
                        r(T3).bvshl(&a4).eq(initial.bvshl(&a4)),
                    );
                }
            }
            Loop::FwdTail | Loop::BwdTail => {
                push("a2 counts the remaining bytes", r(A2).eq(&remaining));
            }
        }
        goals
    }

    fn solver(&self, path: &[Bool], timeout_ms: u32) -> Solver {
        let mut params = Params::new();
        params.set_u32("timeout", timeout_ms);
        params.set_u32("random_seed", Z3_RANDOM_SEED);
        // Z3's SMT core after simplification: on these obligations it is one
        // to two orders of magnitude faster than the default bit-blasting
        // pipeline, which times out on the loop exits.
        let solver = ["simplify", "solve-eqs", "simplify"]
            .into_iter()
            .rev()
            .fold(Tactic::new("smt"), |rest, first| {
                Tactic::new(first).and_then(&rest)
            })
            .solver();
        solver.set_params(&params);
        for assumption in self.assumptions.iter().chain(path) {
            solver.assert(assumption);
        }
        solver
    }

    fn feasible(&self, state: &State, condition: &Bool) -> Result<bool, String> {
        let solver = self.solver(&state.path, GOAL_TIMEOUT_MS);
        solver.assert(condition);
        match solver.check() {
            SatResult::Sat => Ok(true),
            SatResult::Unsat => Ok(false),
            SatResult::Unknown => Err(format!(
                "branch feasibility at {:#x} is inconclusive",
                state.pc
            )),
        }
    }

    /// Searches for a path through `state` on which `goal` fails: `Unsat`
    /// proves the goal, `Sat` comes with the counterexample.
    fn refute(&self, state: &State, goal: &Bool, timeout_ms: u32) -> (SatResult, Option<Model>) {
        let solver = self.solver(&state.path, timeout_ms);
        solver.assert(goal.not().simplify());
        let verdict = solver.check();
        let model = (verdict == SatResult::Sat)
            .then(|| solver.get_model())
            .flatten();
        (verdict, model)
    }

    /// Proves that every path with `state`'s condition meets `goals` and the
    /// path's memory-safety obligations.
    fn prove(&self, state: &State, what: &str, goals: Vec<(String, Bool)>) -> Result<(), String> {
        let all: Vec<(String, Bool)> = state.obligations.iter().cloned().chain(goals).collect();
        self.prove_group(state, what, &all)
    }

    /// Proves `goals` as one conjunction within [`QUICK_TIMEOUT_MS`] or, when
    /// Z3 cannot settle it that fast, goal by goal within [`GOAL_TIMEOUT_MS`]:
    /// equivalent, and for the per-byte memory goals much easier for it.
    fn prove_group(
        &self,
        state: &State,
        what: &str,
        goals: &[(String, Bool)],
    ) -> Result<(), String> {
        let conjunction = Bool::and(&goals.iter().map(|(_, holds)| holds).collect::<Vec<_>>());
        match self.refute(state, &conjunction, QUICK_TIMEOUT_MS) {
            (SatResult::Unsat, _) => return Ok(()),
            (SatResult::Sat, model) => return Err(self.counterexample(state, what, goals, model)),
            (SatResult::Unknown, _) => {}
        }
        for (name, holds) in goals {
            match self.refute(state, holds, GOAL_TIMEOUT_MS) {
                (SatResult::Unsat, _) => {}
                (SatResult::Sat, model) => {
                    return Err(self.counterexample(
                        state,
                        what,
                        &[(name.clone(), holds.clone())],
                        model,
                    ))
                }
                (SatResult::Unknown, _) => {
                    return Err(format!("{what}: {name} is inconclusive (solver budget)"))
                }
            }
        }
        Ok(())
    }

    fn counterexample(
        &self,
        state: &State,
        what: &str,
        goals: &[(String, Bool)],
        model: Option<Model>,
    ) -> String {
        let Some(model) = model else {
            return format!("{what}: fails, but Z3 gave no model");
        };
        let value = |v: &BV| model.eval(v, true).and_then(|v| v.as_u64());
        let failed: Vec<&str> = goals
            .iter()
            .filter(|(_, holds)| model.eval(holds, true).and_then(|b| b.as_bool()) != Some(true))
            .map(|(name, _)| name.as_str())
            .collect();
        format!(
            "{what}: fails {failed:?} for dst = {:?}, src = {:?}, len = {:?} (forked at {:x?})",
            value(&self.dst),
            value(&self.src),
            value(&self.len),
            state.trail
        )
    }

    /// Explores every path from `start`, which is a routine entry or the loop
    /// head `origin`, to the end of its segment.
    fn explore(&self, start: State, origin: Option<u64>) -> Result<(), String> {
        let mut pending = vec![start];
        while let Some(mut state) = pending.pop() {
            loop {
                if state.pc == REAL_MEMCPY || state.pc == REAL_MEMMOVE {
                    self.fall_back(&state)?;
                    break;
                }
                if state.steps > 0 {
                    if let Some((label, lp)) = self.loops.get(&state.pc) {
                        self.arrive(&state, label, *lp, origin)?;
                        break;
                    }
                }
                state.steps += 1;
                if state.steps > STEP_BUDGET {
                    return Err(format!(
                        "path through {:x?} exceeds the step budget",
                        state.trail
                    ));
                }
                let instruction = *self.program.instructions.get(&state.pc).ok_or_else(|| {
                    format!("jumps to {:#x}, which starts no instruction", state.pc)
                })?;
                match self.execute(&mut state, &instruction)? {
                    Flow::Next(pc) => state.pc = pc,
                    Flow::Branch(condition, taken, fallthrough) => {
                        let condition = condition.simplify();
                        let (go, stay) = match condition.as_bool() {
                            Some(decided) => (decided, !decided),
                            None => (
                                self.feasible(&state, &condition)?,
                                self.feasible(&state, &condition.not())?,
                            ),
                        };
                        if go && stay {
                            let mut other = state.clone();
                            other.path.push(condition.not());
                            other.trail.push(state.pc);
                            other.pc = fallthrough;
                            pending.push(other);
                            state.path.push(condition);
                            state.trail.push(state.pc);
                        }
                        if !go && !stay {
                            break;
                        }
                        state.pc = if go { taken } else { fallthrough };
                    }
                    Flow::Jump(target) => {
                        if target == self.entry[RA] {
                            self.ret(&state)?;
                            break;
                        }
                        let mut targets = self.targets(&state, &target)?.into_iter();
                        let first = targets.next().ok_or("jump has no feasible target")?;
                        for pc in targets {
                            let mut other = state.clone();
                            other.path.push(target.eq(pc));
                            other.trail.push(state.pc);
                            other.pc = pc;
                            pending.push(other);
                        }
                        state.path.push(target.eq(first));
                        state.pc = first;
                    }
                }
            }
        }
        Ok(())
    }

    /// Every value `target` takes on `state`'s path.
    fn targets(&self, state: &State, target: &BV) -> Result<Vec<u64>, String> {
        if let Some(pc) = target.simplify().as_u64() {
            return Ok(vec![pc]);
        }
        let solver = self.solver(&state.path, GOAL_TIMEOUT_MS);
        let mut found = Vec::new();
        loop {
            match solver.check() {
                SatResult::Unsat => return Ok(found),
                SatResult::Unknown => {
                    return Err(format!("jump at {:#x} is inconclusive", state.pc))
                }
                SatResult::Sat => {
                    let pc = solver
                        .get_model()
                        .and_then(|model| model.eval(target, true))
                        .and_then(|value| value.as_u64())
                        .ok_or("jump target has no value")?;
                    if found.len() == 64 {
                        return Err(format!("jump at {:#x} has over 64 targets", state.pc));
                    }
                    found.push(pc);
                    solver.assert(target.eq(pc).not());
                }
            }
        }
    }

    fn ret(&self, state: &State) -> Result<(), String> {
        let mut goals = self.calling_convention(&state.regs);
        let all = Base::Copied {
            lo: self.dst.clone(),
            hi: &self.dst + &self.len,
        };
        goals.extend(self.memory_goals(&state.mem, &all));
        self.prove(state, &format!("return via {:x?}", state.trail), goals)
    }

    fn fall_back(&self, state: &State) -> Result<(), String> {
        if self.routine.map(Routine::fallback) != Some(state.pc) || state.stored {
            return Err(format!(
                "tail-calls {:#x} after {:x?}{}",
                state.pc,
                state.trail,
                if state.stored { " having stored" } else { "" }
            ));
        }
        let mut goals = self.calling_convention(&state.regs);
        goals.push(("a1 == src".to_string(), state.regs[A1].eq(&self.src)));
        goals.push(("a2 == len".to_string(), state.regs[A2].eq(&self.len)));
        self.prove(state, &format!("fallback via {:x?}", state.trail), goals)
    }

    fn arrive(
        &self,
        state: &State,
        label: &str,
        lp: Loop,
        origin: Option<u64>,
    ) -> Result<(), String> {
        if !matches!(lp, Loop::MisFwd(_) | Loop::MisBwd(_)) {
            return self.arrive_case(state, label, lp, origin, None);
        }
        // One obligation per source offset keeps every shift amount constant.
        for k in 1..8 {
            let case = state.regs[A5].eq(k).simplify();
            if case.as_bool() == Some(false) || !self.feasible(state, &case)? {
                continue;
            }
            let mut split = state.clone();
            split.path.push(case);
            self.arrive_case(&split, label, lp, origin, Some(k))?;
        }
        Ok(())
    }

    fn arrive_case(
        &self,
        state: &State,
        label: &str,
        lp: Loop,
        origin: Option<u64>,
        offset: Option<u64>,
    ) -> Result<(), String> {
        let mut goals = self.invariant(lp, &state.regs, offset);
        goals.extend(self.memory_goals(&state.mem, &self.copied(lp, &state.regs[A7])));
        match origin {
            Some(head) if head == state.pc => {
                let start = state
                    .start
                    .as_ref()
                    .ok_or("back edge without a loop start")?;
                let progress = if lp.forward() {
                    state.regs[A7].bvugt(start)
                } else {
                    state.regs[A7].bvult(start)
                };
                goals.push(("destination cursor progresses".to_string(), progress));
            }
            Some(head) if head > state.pc => {
                return Err(format!(
                    "{label} is reached from the later loop head {head:#x}"
                ));
            }
            _ => {}
        }
        let case = offset.map_or(String::new(), |k| format!(" with a5 = {k}"));
        self.prove(
            state,
            &format!("{label}{case} via {:x?}", state.trail),
            goals,
        )
    }

    fn execute(&self, state: &mut State, instruction: &SourceInstruction) -> Result<Flow, String> {
        let row = instruction.row();
        let pc = state.pc;
        let next = pc + if row.is_compressed { 2 } else { 4 };
        let imm = row.operands.imm as u64;
        let reg = |index: Option<u8>| state.regs[usize::from(index.unwrap_or(0))].clone();
        let (rs1, rs2) = (reg(row.operands.rs1), reg(row.operands.rs2));
        let rd = usize::from(row.operands.rd.unwrap_or(0));
        let kind = instruction.kind();
        let value = match kind {
            Kind::ADD => &rs1 + &rs2,
            Kind::SUB => &rs1 - &rs2,
            Kind::OR => &rs1 | &rs2,
            Kind::XOR => &rs1 ^ &rs2,
            Kind::SLL => rs1.bvshl(&(&rs2 & 63u64)),
            Kind::SRL => rs1.bvlshr(&(&rs2 & 63u64)),
            Kind::ADDI => &rs1 + imm,
            Kind::ANDI => &rs1 & imm,
            Kind::SLLI => rs1.bvshl(BV::from_u64(imm & 63, 64)),
            Kind::SRLI => rs1.bvlshr(BV::from_u64(imm & 63, 64)),
            Kind::AUIPC => BV::from_u64(pc.wrapping_add(imm), 64),
            Kind::LD | Kind::LWU | Kind::LHU | Kind::LBU => {
                let width = access_width(kind);
                let addr = &rs1 + imm;
                let safe = self.inside(&addr, width, &self.src);
                state
                    .obligations
                    .push((format!("{width}-byte load at {pc:#x}"), safe));
                self.load(&state.mem, &addr, width)
            }
            Kind::SD | Kind::SW | Kind::SH | Kind::SB => {
                let width = access_width(kind);
                let addr = &rs1 + imm;
                let safe = self.inside(&addr, width, &self.dst);
                state
                    .obligations
                    .push((format!("{width}-byte store at {pc:#x}"), safe));
                state.mem.store(&addr, &rs2, width);
                state.stored = true;
                return Ok(Flow::Next(next));
            }
            Kind::BEQ => return Ok(Flow::Branch(rs1.eq(&rs2), pc.wrapping_add(imm), next)),
            Kind::BNE => return Ok(Flow::Branch(rs1.eq(&rs2).not(), pc.wrapping_add(imm), next)),
            Kind::BLTU => return Ok(Flow::Branch(rs1.bvult(&rs2), pc.wrapping_add(imm), next)),
            Kind::BGEU => return Ok(Flow::Branch(rs1.bvuge(&rs2), pc.wrapping_add(imm), next)),
            Kind::JAL if rd == 0 => return Ok(Flow::Next(pc.wrapping_add(imm))),
            Kind::JALR if rd == 0 => return Ok(Flow::Jump((&rs1 + imm).simplify())),
            _ => return Err(format!("unsupported {kind:?} at {pc:#x}")),
        };
        if rd != 0 {
            state.regs[rd] = value.simplify();
        }
        Ok(Flow::Next(next))
    }

    /// `[addr, addr + width)` lies inside `[base, base + len)` and is naturally aligned.
    fn inside(&self, addr: &BV, width: u64, base: &BV) -> Bool {
        let in_range = self.len.bvuge(width) & (addr - base).bvule(&(&self.len - width));
        in_range & (addr & (width - 1)).eq(0u64)
    }
}

fn access_width(kind: Kind) -> u64 {
    match kind {
        Kind::LD | Kind::SD => 8,
        Kind::LWU | Kind::SW => 4,
        Kind::LHU | Kind::SH => 2,
        _ => 1,
    }
}

/// Every path from the wrapper's entry to a return, a fallback, or a loop head.
fn prove_entry(routine: Routine) {
    let proof = Proof::new(Program::get(), Some(routine), Direction::Any);
    if let Err(error) = proof.explore(proof.entry_state(routine), None) {
        panic!("{}: {error}", routine.symbol());
    }
}

/// Every path from each of the wrapper's equal-offset loop heads.
fn prove_loops(routine: Routine) {
    let program = Program::get();
    for (name, lp) in WRAPPER_LOOPS {
        let proof = Proof::new(program, Some(routine), lp.direction());
        let pc = program.symbol(&format!(".L{name}_{}", routine.symbol()));
        if let Err(error) = proof.explore(proof.loop_state(pc, None), Some(pc)) {
            panic!("{} from {name}: {error}", routine.symbol());
        }
    }
}

#[test]
fn test_copy_runtime_memcpy_entry() {
    prove_entry(Routine::Memcpy);
}

#[test]
fn test_copy_runtime_memcpy_loops() {
    prove_loops(Routine::Memcpy);
}

#[test]
fn test_copy_runtime_memmove_entry() {
    prove_entry(Routine::Memmove);
}

#[test]
fn test_copy_runtime_memmove_loops() {
    prove_loops(Routine::Memmove);
}

#[test]
fn test_copy_runtime_misaligned_leaf_loops() {
    let program = Program::get();
    for (name, lp) in LEAF_LOOPS {
        let proof = Proof::new(program, None, lp.direction());
        let pc = program.symbol(&format!(".L{name}"));
        let offsets: Vec<Option<u64>> = match proof.loops[&pc].1 {
            Loop::MisFwd(_) | Loop::MisBwd(_) => (1..8).map(Some).collect(),
            _ => vec![None],
        };
        for offset in offsets {
            if let Err(error) = proof.explore(proof.loop_state(pc, offset), Some(pc)) {
                panic!("__jolt_copy_misaligned from {name} (a5 = {offset:?}): {error}");
            }
        }
    }
}
