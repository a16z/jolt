//! Keccak-f[1600] inline expansion.
//!
//! Every emitted instruction is one trace row. The ρ/π step follows its
//! single 24-lane cycle in place, leaving one rotated lane in a temporary
//! register until χ consumes it. D[3] and D[4] reuse dead C registers.

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

/// Register plan (37 virtual registers): A[25], C[5], D[3], one ρ/π
/// temporary, two χ temporaries, and one scratch register.
struct Keccak256SequenceBuilder {
    asm: InlineExpansionBuilder,
    round: u32,
    a: [InlineRegister; NUM_LANES],
    c: [InlineRegister; 5],
    d: [InlineRegister; 3],
    pi_temp: InlineRegister,
    chi_temp: [InlineRegister; 2],
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
        let d = asm.allocate_inline_array::<3>()?;
        let pi_temp = asm.allocate_for_inline()?;
        let chi_temp = asm.allocate_inline_array::<2>()?;
        let scratch = asm.allocate_for_inline()?;
        Ok(Self {
            asm,
            round: 0,
            a,
            c,
            d,
            pi_temp,
            chi_temp,
            scratch,
            absorb_block,
            operands,
        })
    }

    fn build(mut self) -> Result<ExpandedInstructionSequence, ExpansionError> {
        self.load_state();
        for round in 0..24 {
            self.round = round;
            self.theta();
            self.rho_and_pi();
            self.chi();
            self.iota();
        }
        self.store_state();

        self.asm.release_many(self.a);
        self.asm.release_many(self.c);
        self.asm.release_many(self.d);
        self.asm.release(self.pi_temp);
        self.asm.release_many(self.chi_temp);
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

    fn rho_pi_lane(&self, x: usize, y: usize) -> u8 {
        if (x, y) == PI_TEMP_LANE {
            *self.pi_temp
        } else {
            self.lane(x, y)
        }
    }

    fn d_lane(&self, x: usize) -> u8 {
        match x {
            0..=2 => *self.d[x],
            3 => *self.c[1],
            4 => *self.c[2],
            _ => unreachable!("keccak D index out of range"),
        }
    }

    fn theta(&mut self) {
        for x in 0..5 {
            let c = *self.c[x];
            self.asm.xor(Reg(self.lane(x, 0)), Reg(self.lane(x, 1)), c);
            for y in 2..5 {
                self.asm.xor(Reg(c), Reg(self.lane(x, y)), c);
            }
        }

        // C[1] is last read at x=2 and C[2] at x=3, so D[3..5] can reuse them.
        for x in 0..5 {
            let d = self.d_lane(x);
            let c_prev = *self.c[(x + 4) % 5];
            let c_next = *self.c[(x + 1) % 5];
            self.asm.emit_r(Kind::VirtualXORROTL1, d, c_prev, c_next);
        }

        let a = self.lane(0, 0);
        self.asm.xor(Reg(a), Reg(self.d_lane(0)), a);
    }

    /// Walks the 24-lane π cycle backwards from `RHO_PI_FIRST_SOURCE`: each
    /// lane is rotated into the register whose own value was rotated out one
    /// step earlier, so only the first lane needs a temporary. The last
    /// source is `PI_TEMP_LANE`, whose register is left holding stale data
    /// until χ reads the lane from `pi_temp` instead.
    fn rho_and_pi(&mut self) {
        self.emit_rho_pi_lane(RHO_PI_FIRST_SOURCE, *self.pi_temp);

        let mut destination = RHO_PI_FIRST_SOURCE;
        for _ in 1..24 {
            let source = pi_source(destination);
            self.emit_rho_pi_lane(source, self.lane(destination.0, destination.1));
            destination = source;
        }
    }

    fn emit_rho_pi_lane(&mut self, source: (usize, usize), destination: u8) {
        let (x, y) = source;
        let kind = match 64 - ROTATION_OFFSETS[x][y] {
            2 => Kind::VirtualXORROT2,
            3 => Kind::VirtualXORROT3,
            8 => Kind::VirtualXORROT8,
            9 => Kind::VirtualXORROT9,
            19 => Kind::VirtualXORROT19,
            20 => Kind::VirtualXORROT20,
            21 => Kind::VirtualXORROT21,
            23 => Kind::VirtualXORROT23,
            25 => Kind::VirtualXORROT25,
            28 => Kind::VirtualXORROT28,
            36 => Kind::VirtualXORROT36,
            37 => Kind::VirtualXORROT37,
            39 => Kind::VirtualXORROT39,
            43 => Kind::VirtualXORROT43,
            44 => Kind::VirtualXORROT44,
            46 => Kind::VirtualXORROT46,
            49 => Kind::VirtualXORROT49,
            50 => Kind::VirtualXORROT50,
            54 => Kind::VirtualXORROT54,
            56 => Kind::VirtualXORROT56,
            58 => Kind::VirtualXORROT58,
            61 => Kind::VirtualXORROT61,
            62 => Kind::VirtualXORROT62,
            63 => Kind::VirtualXORROT63,
            _ => unreachable!("nonzero Keccak rho rotation"),
        };
        self.asm
            .emit_r(kind, destination, self.lane(x, y), self.d_lane(x));
    }

    fn chi(&mut self) {
        for y in 0..5 {
            let b: [u8; 5] = std::array::from_fn(|x| self.rho_pi_lane(x, y));
            let destination: [u8; 5] = std::array::from_fn(|x| self.lane(x, y));
            let scratch = *self.scratch;
            let t3 = *self.chi_temp[0];
            let t4 = *self.chi_temp[1];

            self.asm.emit_r(Kind::ANDN, scratch, b[2], b[1]);
            self.asm.emit_r(Kind::ANDN, t3, b[0], b[4]);
            self.asm.emit_r(Kind::ANDN, t4, b[1], b[0]);
            self.asm.xor(Reg(b[0]), Reg(scratch), destination[0]);

            self.asm.emit_r(Kind::ANDN, scratch, b[3], b[2]);
            self.asm.xor(Reg(b[1]), Reg(scratch), destination[1]);

            self.asm.emit_r(Kind::ANDN, scratch, b[4], b[3]);
            self.asm.xor(Reg(b[2]), Reg(scratch), destination[2]);
            self.asm.xor(Reg(b[3]), Reg(t3), destination[3]);
            self.asm.xor(Reg(b[4]), Reg(t4), destination[4]);
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

/// First lane rotated in ρ/π. Its destination register still holds an unread
/// lane at that point, so the result is parked in `pi_temp`.
const RHO_PI_FIRST_SOURCE: (usize, usize) = (1, 0);
/// The lane whose ρ/π result lives in `pi_temp` rather than its own register.
const PI_TEMP_LANE: (usize, usize) = pi_destination(RHO_PI_FIRST_SOURCE);

/// π moves lane `(x, y)` to `(y, 2x + 3y mod 5)`.
pub(crate) const fn pi_destination((x, y): (usize, usize)) -> (usize, usize) {
    (y, (2 * x + 3 * y) % 5)
}

/// Inverse of [`pi_destination`]: `x = X + 3Y mod 5` since `2^-1 = 3 mod 5`.
const fn pi_source((x, y): (usize, usize)) -> (usize, usize) {
    ((x + 3 * y) % 5, x)
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
