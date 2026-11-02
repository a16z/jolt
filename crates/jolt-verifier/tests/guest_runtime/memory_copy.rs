//! Symbolic check of the guest copy runtime in `jolt-platform/src/memory.S`.
//!
//! Each case runs `__wrap_memcpy` or `__wrap_memmove`, as decoded by
//! `jolt_program`, on symbolic arguments. The source and destination bases are
//! known only modulo 8 and by the sign of `dst - src`, and every loaded byte
//! carries a tag naming its load and source offset. Data may only move through
//! registers, byte-lane shifts, and ORs of disjoint lanes; every branch and
//! address must be decidable from the case, or the case fails. One case thus
//! covers every base pair in its class and every byte value, assuming no
//! address wraparound.
//!
//! A case passes when the routine tail-calls the original routine before any
//! store, or returns `dst` with sp, gp, tp, and s0-s11 intact after naturally
//! aligned loads inside the source and stores inside the destination that give
//! destination byte `j` source byte `j`. For `memmove`, every overlapping
//! distance in the class is also safe: no store aliases a source byte before
//! that byte's last load whose lane reaches a store.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use jolt_program::image::decode_elf;
use jolt_riscv::{SourceInstruction, SourceInstructionKind as Kind, RV64IMAC_JOLT};

use super::PRESERVED_REGISTERS;

/// Every length through this bound reaches each block loop's remainders with
/// several iteration counts, including the misaligned leaf above its cutoff.
const EXHAUSTIVE_LEN: u64 = 1024;
/// Longer copies only repeat loop bodies; these sample them.
const SPOT_LENS: [u64; 4] = [4095, 4097, 16_383, 16_385];

/// Entry points of the wrapped routines and the originals they fall back to.
pub struct CopyRuntime {
    pub memcpy: u64,
    pub memmove: u64,
    pub real_memcpy: u64,
    pub real_memmove: u64,
}

/// Returns one message per failing case.
pub fn check(elf: &[u8], runtime: &CopyRuntime) -> Vec<String> {
    let program: BTreeMap<u64, SourceInstruction> = match decode_elf(elf, RV64IMAC_JOLT) {
        Ok(image) => image
            .instructions
            .into_iter()
            .map(|instruction| (instruction.row().address as u64, instruction))
            .collect(),
        Err(error) => return vec![format!("guest ELF does not decode: {error:?}")],
    };
    let routines = [
        ("__wrap_memcpy", runtime.memcpy, runtime.real_memcpy, false),
        (
            "__wrap_memmove",
            runtime.memmove,
            runtime.real_memmove,
            true,
        ),
    ];
    let mut failures = Vec::new();
    for (name, entry, fallback, allows_overlap) in routines {
        for len in (0..=EXHAUSTIVE_LEN).chain(SPOT_LENS) {
            for src_low in 0..8 {
                for dst_low in 0..8 {
                    let orders: &[Order] = if src_low == dst_low {
                        &[Order::DstBelow, Order::DstAbove, Order::Same]
                    } else {
                        &[Order::DstBelow, Order::DstAbove]
                    };
                    for &order in orders {
                        let case = Case {
                            entry,
                            fallback,
                            len,
                            src_low,
                            dst_low,
                            order,
                            allows_overlap,
                        };
                        let result = case.run(&program).and_then(|exit| match exit {
                            Exit::Return(stores) => case.verify(&stores),
                            Exit::Fallback => Ok(()),
                        });
                        if let Err(error) = result {
                            failures.push(format!(
                                "{name}: len {len}, src % 8 = {src_low}, dst % 8 = {dst_low}, \
                                 {order:?}: {error}"
                            ));
                        }
                    }
                }
            }
        }
    }
    failures
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Base {
    Src,
    Dst,
}

/// Sign of `dst - src`; each class covers every distance with that sign.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    DstBelow,
    DstAbove,
    Same,
}

/// A loaded source byte: the step of its load and its offset into the source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tag {
    step: u32,
    offset: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Value {
    Const(u64),
    /// `base + offset`.
    Addr(Base, i64),
    /// Only the low three bits are known.
    Low(u64),
    /// Byte lanes, least significant first; `None` lanes are zero.
    Data([Option<Tag>; 8]),
    /// The register's unknown value on entry.
    Entry(usize),
}

struct Store {
    step: u32,
    offset: u32,
    tags: Vec<Tag>,
}

enum Exit {
    Return(Vec<Store>),
    Fallback,
}

struct Case {
    entry: u64,
    fallback: u64,
    len: u64,
    src_low: u64,
    dst_low: u64,
    order: Order,
    allows_overlap: bool,
}

impl Case {
    fn run(&self, program: &BTreeMap<u64, SourceInstruction>) -> Result<Exit, String> {
        let mut regs: [Value; 32] = std::array::from_fn(Value::Entry);
        regs[0] = Value::Const(0);
        regs[10] = Value::Addr(Base::Dst, 0);
        regs[11] = Value::Addr(Base::Src, 0);
        regs[12] = Value::Const(self.len);
        let mut stores = Vec::new();
        let mut pc = self.entry;
        for step in 1..=200 + 40 * self.len as u32 {
            let instruction = program
                .get(&pc)
                .ok_or_else(|| format!("jumps to {pc:#x}, which starts no instruction"))?;
            let row = instruction.row();
            let reg = |index: Option<u8>| regs[usize::from(index.unwrap_or(0))];
            let (rs1, rs2) = (reg(row.operands.rs1), reg(row.operands.rs2));
            let rd = usize::from(row.operands.rd.unwrap_or(0));
            let imm = row.operands.imm as u64;
            let kind = instruction.kind();
            let mut next = pc + if row.is_compressed { 2 } else { 4 };
            let result = match kind {
                Kind::LD | Kind::LWU | Kind::LHU | Kind::LBU => {
                    let width = access_width(kind);
                    let offset = self.access(Base::Src, rs1, imm, width)?;
                    let mut lanes = [None; 8];
                    for (lane, byte) in lanes[..width].iter_mut().zip(offset..) {
                        *lane = Some(Tag { step, offset: byte });
                    }
                    Some(Value::Data(lanes))
                }
                Kind::SD | Kind::SW | Kind::SH | Kind::SB => {
                    let width = access_width(kind);
                    let offset = self.access(Base::Dst, rs1, imm, width)?;
                    let Value::Data(lanes) = rs2 else {
                        return Err(format!("stores non-data {rs2:?}"));
                    };
                    let tags = lanes[..width]
                        .iter()
                        .map(|lane| lane.ok_or("stores a zero lane"))
                        .collect::<Result<_, _>>()?;
                    stores.push(Store { step, offset, tags });
                    None
                }
                Kind::ADD | Kind::SUB | Kind::OR | Kind::XOR | Kind::SLL | Kind::SRL => {
                    Some(self.alu(kind, rs1, rs2)?)
                }
                Kind::ADDI | Kind::ANDI | Kind::SLLI | Kind::SRLI => {
                    Some(self.alu(kind, rs1, Value::Const(imm))?)
                }
                Kind::AUIPC => Some(Value::Const(relative(pc, imm)?)),
                Kind::BEQ | Kind::BNE | Kind::BLTU | Kind::BGEU => {
                    let ordering = self.compare(rs1, rs2)?;
                    let taken = match kind {
                        Kind::BEQ => ordering.is_eq(),
                        Kind::BNE => ordering.is_ne(),
                        Kind::BLTU => ordering.is_lt(),
                        _ => ordering.is_ge(),
                    };
                    if taken {
                        next = relative(pc, imm)?;
                    }
                    None
                }
                Kind::JAL if rd == 0 => {
                    next = relative(pc, imm)?;
                    None
                }
                Kind::JALR if rd == 0 => match rs1 {
                    Value::Entry(1) if imm == 0 => {
                        check_registers(&regs)?;
                        return Ok(Exit::Return(stores));
                    }
                    Value::Const(base) if relative(base, imm)? == self.fallback => {
                        check_registers(&regs)?;
                        let arguments = [
                            Value::Addr(Base::Src, 0),
                            Value::Const(self.len),
                            Value::Entry(1),
                        ];
                        if [regs[11], regs[12], regs[1]] != arguments {
                            return Err("falls back with changed arguments".into());
                        }
                        if !stores.is_empty() {
                            return Err("falls back after storing".into());
                        }
                        return Ok(Exit::Fallback);
                    }
                    Value::Const(base) => {
                        next = relative(base, imm)?;
                        None
                    }
                    target => return Err(format!("jumps through {target:?}")),
                },
                _ => return Err(format!("executes unsupported {kind:?} at {pc:#x}")),
            };
            if let Some(value) = result.filter(|_| rd != 0) {
                regs[rd] = value;
            }
            pc = next;
        }
        Err("does not exit within its step budget".into())
    }

    /// Returns the offset into `base` that `address + imm` names.
    fn access(&self, base: Base, address: Value, imm: u64, width: usize) -> Result<u32, String> {
        let Value::Addr(found, start) = address else {
            return Err(format!("accesses memory through {address:?}"));
        };
        let offset = start.checked_add(imm as i64).ok_or("overflows an offset")?;
        if found != base || offset < 0 || offset + width as i64 > self.len as i64 {
            return Err(format!(
                "{width}-byte access at {found:?} + {offset} outside {base:?}"
            ));
        }
        if (self.low(base) as i64 + offset) % width as i64 != 0 {
            return Err(format!(
                "misaligned {width}-byte access at {base:?} + {offset}"
            ));
        }
        Ok(offset as u32)
    }

    fn low(&self, base: Base) -> u64 {
        match base {
            Base::Src => self.src_low,
            Base::Dst => self.dst_low,
        }
    }

    fn low_bits(&self, value: Value) -> Result<u64, String> {
        match value {
            Value::Const(bits) | Value::Low(bits) => Ok(bits & 7),
            Value::Addr(base, offset) => Ok((self.low(base) as i64 + offset) as u64 & 7),
            Value::Data(_) | Value::Entry(_) => Err(format!("needs the low bits of {value:?}")),
        }
    }

    fn compare(&self, a: Value, b: Value) -> Result<Ordering, String> {
        match (a, b) {
            (Value::Const(x), Value::Const(y)) => Ok(x.cmp(&y)),
            (Value::Addr(p, x), Value::Addr(q, y)) if p == q => Ok(x.cmp(&y)),
            (Value::Addr(p, x), Value::Addr(_, y)) if x == y => Ok(match (self.order, p) {
                (Order::Same, _) => Ordering::Equal,
                (Order::DstBelow, Base::Dst) | (Order::DstAbove, Base::Src) => Ordering::Less,
                (Order::DstBelow, Base::Src) | (Order::DstAbove, Base::Dst) => Ordering::Greater,
            }),
            _ => Err(format!("branches on {a:?} against {b:?}")),
        }
    }

    fn alu(&self, kind: Kind, a: Value, b: Value) -> Result<Value, String> {
        Ok(match (kind, a, b) {
            (Kind::ADD | Kind::ADDI, x, Value::Const(0)) | (Kind::ADD, Value::Const(0), x) => x,
            (Kind::ADD | Kind::ADDI, Value::Const(x), Value::Const(y)) => {
                Value::Const(relative(x, y)?)
            }
            (Kind::ADD | Kind::ADDI, Value::Addr(base, x), Value::Const(y))
            | (Kind::ADD, Value::Const(y), Value::Addr(base, x)) => {
                Value::Addr(base, x.checked_add(y as i64).ok_or(OVERFLOW)?)
            }
            (Kind::SUB, Value::Const(x), Value::Const(y)) => {
                Value::Const(x.checked_sub(y).ok_or(OVERFLOW)?)
            }
            (Kind::SUB, Value::Addr(base, x), Value::Const(y)) => {
                Value::Addr(base, x.checked_sub(y as i64).ok_or(OVERFLOW)?)
            }
            (Kind::SUB, Value::Addr(p, x), Value::Addr(q, y)) if p == q => {
                Value::Const(x.checked_sub(y).ok_or(OVERFLOW)? as u64)
            }
            (Kind::ANDI, Value::Const(x), Value::Const(mask)) => Value::Const(x & mask),
            (Kind::ANDI, x, Value::Const(mask)) if mask <= 7 => {
                Value::Const(self.low_bits(x)? & mask)
            }
            (Kind::ANDI, Value::Addr(base, x), Value::Const(mask)) if mask == !7 => {
                let low = self.low(base) as i64;
                Value::Addr(base, ((low + x) & !7) - low)
            }
            (Kind::OR, Value::Data(x), Value::Data(y)) => {
                let mut lanes = [None; 8];
                for (lane, pair) in lanes.iter_mut().zip(x.into_iter().zip(y)) {
                    *lane = match pair {
                        (Some(_), Some(_)) => return Err("ORs two live lanes".into()),
                        (x, y) => x.or(y),
                    };
                }
                Value::Data(lanes)
            }
            (Kind::OR, Value::Const(x), Value::Const(y)) => Value::Const(x | y),
            (Kind::XOR, Value::Const(x), Value::Const(y)) => Value::Const(x ^ y),
            (Kind::OR | Kind::XOR, Value::Const(_) | Value::Addr(..) | Value::Low(_), _)
                if matches!(b, Value::Const(_) | Value::Addr(..) | Value::Low(_)) =>
            {
                let (x, y) = (self.low_bits(a)?, self.low_bits(b)?);
                Value::Low(if kind == Kind::OR { x | y } else { x ^ y })
            }
            (Kind::SLL | Kind::SLLI | Kind::SRL | Kind::SRLI, x, Value::Const(amount)) => {
                let (left, amount) = (matches!(kind, Kind::SLL | Kind::SLLI), amount & 63);
                match x {
                    Value::Const(bits) if left => Value::Const(bits << amount),
                    Value::Const(bits) => Value::Const(bits >> amount),
                    Value::Data(lanes) if amount % 8 == 0 => {
                        let shift = (amount / 8) as usize;
                        let mut shifted = [None; 8];
                        if left {
                            shifted[shift..].copy_from_slice(&lanes[..8 - shift]);
                        } else {
                            shifted[..8 - shift].copy_from_slice(&lanes[shift..]);
                        }
                        Value::Data(shifted)
                    }
                    _ => return Err(format!("shifts {x:?} by {amount}")),
                }
            }
            _ => return Err(format!("applies {kind:?} to {a:?} and {b:?}")),
        })
    }

    fn verify(&self, stores: &[Store]) -> Result<(), String> {
        let len = self.len as usize;
        let mut first_store = vec![None; len];
        let mut last_use = vec![None; len];
        for store in stores {
            for (byte, tag) in (store.offset as usize..).zip(&store.tags) {
                if tag.offset as usize != byte {
                    return Err(format!(
                        "stores source byte {} to destination byte {byte}",
                        tag.offset
                    ));
                }
                first_store[byte] = first_store[byte].or(Some(store.step));
                last_use[byte] = last_use[byte].max(Some(tag.step));
            }
        }
        // With dst == src every store writes back the byte already there.
        if self.order == Order::Same {
            return Ok(());
        }
        if let Some(byte) = first_store.iter().position(Option::is_none) {
            return Err(format!("never stores destination byte {byte}"));
        }
        if !self.allows_overlap {
            return Ok(());
        }
        // A store to destination byte j aliases source byte j - d (DstBelow) or
        // j + d (DstAbove), for distances d >= 1 congruent to the base offsets.
        let residue = match self.order {
            Order::DstBelow => (self.src_low + 8 - self.dst_low) % 8,
            _ => (self.dst_low + 8 - self.src_low) % 8,
        };
        let min_distance = if residue == 0 { 8 } else { residue as usize };
        let mut latest = [None; 8];
        let mut running_max = |(source, used): (usize, &Option<u32>)| {
            let slot = &mut latest[source % 8];
            *slot = (*slot).max(*used);
            *slot
        };
        // reach[k] is the latest use of a source byte congruent to k (mod 8) at
        // or past k, away from the destination. The store to byte j is safe at
        // every distance iff it follows reach[j -/+ min_distance].
        let hazard = if self.order == Order::DstBelow {
            let reach: Vec<_> = last_use.iter().enumerate().map(&mut running_max).collect();
            first_store[min_distance.min(len)..]
                .iter()
                .zip(&reach)
                .position(|(store, used)| store <= used)
                .map(|byte| byte + min_distance)
        } else {
            let mut reach: Vec<_> = last_use
                .iter()
                .enumerate()
                .rev()
                .map(&mut running_max)
                .collect();
            reach.reverse();
            first_store
                .iter()
                .zip(reach.get(min_distance..).unwrap_or_default())
                .position(|(store, used)| store <= used)
        };
        let Some(byte) = hazard else {
            return Ok(());
        };
        let aliased = |distance: usize| match self.order {
            Order::DstBelow => byte.checked_sub(distance),
            _ => Some(byte + distance).filter(|&source| source < len),
        };
        let distance = (min_distance..len)
            .step_by(8)
            .find(|&distance| {
                aliased(distance).is_some_and(|source| first_store[byte] <= last_use[source])
            })
            .unwrap_or(min_distance);
        Err(format!(
            "at distance {distance}, the store to destination byte {byte} precedes the last \
             use of the source byte it overwrites"
        ))
    }
}

const OVERFLOW: &str = "wraps around the address space";

/// `value + imm`, reading the immediate as two's complement.
fn relative(value: u64, imm: u64) -> Result<u64, String> {
    value
        .checked_add_signed(imm as i64)
        .ok_or_else(|| OVERFLOW.to_string())
}

fn check_registers(regs: &[Value; 32]) -> Result<(), String> {
    if regs[10] != Value::Addr(Base::Dst, 0) {
        return Err(format!("returns {:?} instead of dst", regs[10]));
    }
    match PRESERVED_REGISTERS
        .iter()
        .map(|&register| usize::from(register))
        .find(|&register| regs[register] != Value::Entry(register))
    {
        Some(register) => Err(format!("clobbers preserved x{register}")),
        None => Ok(()),
    }
}

fn access_width(kind: Kind) -> usize {
    match kind {
        Kind::LD | Kind::SD => 8,
        Kind::LWU | Kind::SW => 4,
        Kind::LHU | Kind::SH => 2,
        _ => 1,
    }
}
