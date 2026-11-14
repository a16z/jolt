//! Keccak-f[1600] inline expansion.
//!
//! Every emitted instruction is one trace row. The π permutation renames
//! registers at expansion time. Column parities remain live through θ, then
//! their registers are reused for χ temporaries.

use crate::{
    INLINE_OPCODE, KECCAK256_ABSORB_PERMUTE_FUNCT3, KECCAK256_ABSORB_PERMUTE_NAME,
    KECCAK256_FUNCT3, KECCAK256_FUNCT7, KECCAK256_NAME, NUM_LANES, RATE_IN_U64,
};
use jolt_inlines_sdk::host::{
    ExpandedInstructionSequence, ExpansionError, InlineBuilderExt, InlineExpansionBuilder,
    InlineOp, InlineOperands, InlineRegister, Kind, NoAdvice,
    Value::{Imm, Reg},
};

#[rustfmt::skip]
pub(crate) const ROUND_CONSTANTS: [u64; 24] = [
    0x0000000000000001, 0x0000000000008082,
    0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088,
    0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b,
    0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080,
    0x0000000080000001, 0x8000000080008008,
];

#[rustfmt::skip]
pub(crate) const ROTATION_OFFSETS: [[u32; 5]; 5] = [
    [ 0, 36,  3, 41, 18],
    [ 1, 44, 10, 45,  2],
    [62,  6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39,  8, 14],
];

/// Register plan (31 virtual registers): A[25], C[5], and one scratch
/// register shared by the column correction and χ.
struct Keccak256SequenceBuilder {
    asm: InlineExpansionBuilder,
    round: u32,
    a: [InlineRegister; NUM_LANES],
    c: [InlineRegister; 5],
    scratch: InlineRegister,
    absorb_block: bool,
    operands: InlineOperands,
}

impl Keccak256SequenceBuilder {
    fn new(
        mut asm: InlineExpansionBuilder,
        operands: InlineOperands,
        absorb_block: bool,
    ) -> Result<Self, ExpansionError> {
        let a = asm.allocate_inline_array::<NUM_LANES>()?;
        let c = asm.allocate_inline_array::<5>()?;
        let scratch = asm.allocate_for_inline()?;
        Ok(Self {
            asm,
            round: 0,
            a,
            c,
            scratch,
            absorb_block,
            operands,
        })
    }

    fn build(mut self) -> Result<ExpandedInstructionSequence, ExpansionError> {
        self.load_state();
        for round in 0..24 {
            self.round = round;
            self.theta_rho_and_pi();
            self.chi();
            self.iota();
        }
        self.store_state();

        self.asm.release_many(self.a);
        self.asm.release_many(self.c);
        self.asm.release(self.scratch);
        self.asm.finalize()
    }

    fn load_state(&mut self) {
        self.asm.load_u64_range(self.operands.rs1, 0, &self.a);
        if self.absorb_block {
            let scratch = *self.scratch;
            for i in 0..RATE_IN_U64 {
                self.asm.emit_ld(
                    Kind::LD,
                    scratch,
                    self.operands.rs2,
                    i as i64 * size_of::<u64>() as i64,
                );
                self.asm.xor(Reg(*self.a[i]), Reg(scratch), *self.a[i]);
            }
        }
    }

    fn store_state(&mut self) {
        self.asm.store_u64_range(self.operands.rs1, 0, &self.a);
    }

    fn lane(&self, x: usize, y: usize) -> u8 {
        *self.a[5 * y + x]
    }

    fn theta_rho_and_pi(&mut self) {
        for x in 0..5 {
            let c = *self.c[x];
            self.asm.xor(Reg(self.lane(x, 0)), Reg(self.lane(x, 1)), c);
            for y in 2..5 {
                self.asm.xor(Reg(c), Reg(self.lane(x, y)), c);
            }
        }

        // Keep all parities live while applying one column correction at a time.
        let mut permuted = self.a;
        for (x, offsets) in ROTATION_OFFSETS.iter().enumerate() {
            let d = *self.scratch;
            let c_prev = *self.c[(x + 4) % 5];
            let c_next = *self.c[(x + 1) % 5];
            self.asm.emit_r(Kind::VirtualXORROTL1, d, c_prev, c_next);
            for (y, &rotation) in offsets.iter().enumerate() {
                let a = self.lane(x, y);
                let fused = match rotation {
                    1 => Some(Kind::VirtualXORROT63),
                    3 => Some(Kind::VirtualXORROT61),
                    62 => Some(Kind::VirtualXORROT2),
                    _ => None,
                };
                if let Some(kind) = fused {
                    self.asm.emit_r(kind, a, a, d);
                } else {
                    self.asm.xor(Reg(a), Reg(d), a);
                    if rotation != 0 {
                        self.asm.rotl64(Reg(a), rotation, a);
                    }
                }
                let (px, py) = pi_destination((x, y));
                permuted[5 * py + px] = self.a[5 * y + x];
            }
        }
        self.a = permuted;
    }

    fn chi(&mut self) {
        for y in 0..5 {
            let b: [u8; 5] = std::array::from_fn(|x| self.lane(x, y));
            let scratch = *self.scratch;
            let t3 = *self.c[0];
            let t4 = *self.c[1];

            self.asm.emit_r(Kind::ANDN, scratch, b[2], b[1]);
            self.asm.emit_r(Kind::ANDN, t3, b[0], b[4]);
            self.asm.emit_r(Kind::ANDN, t4, b[1], b[0]);
            self.asm.xor(Reg(b[0]), Reg(scratch), b[0]);

            self.asm.emit_r(Kind::ANDN, scratch, b[3], b[2]);
            self.asm.xor(Reg(b[1]), Reg(scratch), b[1]);

            self.asm.emit_r(Kind::ANDN, scratch, b[4], b[3]);
            self.asm.xor(Reg(b[2]), Reg(scratch), b[2]);
            self.asm.xor(Reg(b[3]), Reg(t3), b[3]);
            self.asm.xor(Reg(b[4]), Reg(t4), b[4]);
        }
    }

    fn iota(&mut self) {
        let first_lane = self.lane(0, 0);
        self.asm.xor(
            Reg(first_lane),
            Imm(ROUND_CONSTANTS[self.round as usize]),
            first_lane,
        );
    }
}

/// π moves lane `(x, y)` to `(y, 2x + 3y mod 5)`.
pub(crate) const fn pi_destination((x, y): (usize, usize)) -> (usize, usize) {
    (y, (2 * x + 3 * y) % 5)
}

pub struct Keccak256Permutation;

impl InlineOp for Keccak256Permutation {
    type Advice = NoAdvice;

    const OPCODE: u32 = INLINE_OPCODE;
    const FUNCT3: u32 = KECCAK256_FUNCT3;
    const FUNCT7: u32 = KECCAK256_FUNCT7;
    const NAME: &'static str = KECCAK256_NAME;

    fn build_sequence(
        asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        Keccak256SequenceBuilder::new(asm, operands, false)?.build()
    }
}

pub struct Keccak256AbsorbPermutation;

impl InlineOp for Keccak256AbsorbPermutation {
    type Advice = NoAdvice;

    const OPCODE: u32 = INLINE_OPCODE;
    const FUNCT3: u32 = KECCAK256_ABSORB_PERMUTE_FUNCT3;
    const FUNCT7: u32 = KECCAK256_FUNCT7;
    const NAME: &'static str = KECCAK256_ABSORB_PERMUTE_NAME;

    fn build_sequence(
        asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        Keccak256SequenceBuilder::new(asm, operands, true)?.build()
    }
}
