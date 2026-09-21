//! Keccak-f[1600] inline expansion.
//!
//! Every emitted instruction is one trace row. The ρ/π step follows its
//! single 24-lane cycle in place, leaving one rotated lane in a temporary
//! register until χ consumes it. D[3] and D[4] reuse dead C registers.

use crate::{
    INLINE_OPCODE, KECCAK256_ABSORB_PERMUTE_FUNCT3, KECCAK256_ABSORB_PERMUTE_NAME,
    KECCAK256_ABSORB_PERMUTE_UNALIGNED_FUNCT3, KECCAK256_ABSORB_PERMUTE_UNALIGNED_NAME,
    KECCAK256_FUNCT3, KECCAK256_FUNCT7, KECCAK256_INIT_ABSORB_PERMUTE_FUNCT3,
    KECCAK256_INIT_ABSORB_PERMUTE_NAME, KECCAK256_INIT_ABSORB_PERMUTE_UNALIGNED_FUNCT3,
    KECCAK256_INIT_ABSORB_PERMUTE_UNALIGNED_NAME, KECCAK256_NAME, NUM_LANES, RATE_IN_U64,
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

/// How the rate block at `rs2` enters the lanes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Absorb {
    /// `a[i] ^= block[i]` over the state loaded from `rs1`.
    IntoState,
    /// `a[0..17] = block`, `a[17..25] = 0`; `rs1` is only written.
    Init,
}

/// Alignment the sequence may assume for the block pointer in `rs2`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockAlignment {
    /// 17 aligned doubleword loads.
    Aligned,
    /// 18 loads of the containing doublewords, funnel-shifted into lanes.
    Any,
}

#[derive(Clone, Copy)]
struct Block {
    absorb: Absorb,
    alignment: BlockAlignment,
}

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
    block: Option<Block>,
    /// Lanes whose value is zero and whose register has not been written:
    /// the capacity lanes of an [`Absorb::Init`] state before round 0, which
    /// θ folds instead of materializing.
    zero_lanes: [bool; NUM_LANES],
    operands: InlineOperands,
}

impl Keccak256SequenceBuilder {
    fn new(
        mut asm: InlineExpansionBuilder,
        operands: InlineOperands,
        block: Option<Block>,
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
            block,
            zero_lanes: [false; NUM_LANES],
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
        let Some(block) = self.block else {
            self.asm.load_u64_range(self.operands.rs1, 0, &self.a);
            return;
        };
        match block.absorb {
            Absorb::IntoState => self.asm.load_u64_range(self.operands.rs1, 0, &self.a),
            Absorb::Init => {
                for zero in &mut self.zero_lanes[RATE_IN_U64..] {
                    *zero = true;
                }
            }
        }
        match block.alignment {
            BlockAlignment::Aligned => self.load_block_aligned(block.absorb),
            BlockAlignment::Any => self.load_block_unaligned(block.absorb),
        }
    }

    fn load_block_aligned(&mut self, absorb: Absorb) {
        match absorb {
            Absorb::Init => {
                self.asm
                    .load_u64_range(self.operands.rs2, 0, &self.a[..RATE_IN_U64]);
            }
            Absorb::IntoState => {
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
    }

    /// Reads the block through the 18 aligned doublewords `w[0..18]` at
    /// `rs2 & !7` that contain it. With `sh = 8 * (rs2 & 7)`, lane `i` is
    /// `w[i] >> sh | w[i + 1] << (64 - sh)`. Both shift operands are
    /// computed once: the right shift is a `VirtualSRL` by the bitmask
    /// `-(2^sh) = !0 << sh`, the left shift a `MUL` by `2^(64 - sh) mod 2^64`,
    /// which is zero for an aligned `rs2`, so the sequence is also correct
    /// there (it then reads the doubleword after the block).
    ///
    /// The C and D registers are dead until θ and serve as temporaries.
    fn load_block_unaligned(&mut self, absorb: Absorb) {
        let rs2 = self.operands.rs2;
        let [base, srl_bitmask, mul_pow2, w_even, w_odd] = self.c.map(|register| *register);
        let [shifted, carried, _] = self.d.map(|register| *register);

        self.asm.emit_i(Kind::VirtualAlignAddr, base, rs2, 0);
        // `srl_bitmask` holds `sh` until the bitmask is formed from it.
        self.asm.emit_i(Kind::ANDI, srl_bitmask, rs2, 7);
        self.asm
            .emit_i(Kind::VirtualMULI, srl_bitmask, srl_bitmask, 8);
        // 2^(64 - sh) mod 2^64 = 2 * 2^(63 - sh); 63 - sh == sh ^ 63 for sh <= 63.
        self.asm.emit_i(Kind::XORI, mul_pow2, srl_bitmask, 63);
        self.asm.emit_i(Kind::VirtualPow2, mul_pow2, mul_pow2, 0);
        self.asm.emit_r(Kind::ADD, mul_pow2, mul_pow2, mul_pow2);
        self.asm
            .emit_i(Kind::VirtualPow2, srl_bitmask, srl_bitmask, 0);
        self.asm.emit_r(Kind::SUB, srl_bitmask, 0, srl_bitmask);

        self.asm.emit_ld(Kind::LD, w_even, base, 0);
        for i in 0..RATE_IN_U64 {
            let (low, high) = if i % 2 == 0 {
                (w_even, w_odd)
            } else {
                (w_odd, w_even)
            };
            self.asm.emit_ld(
                Kind::LD,
                high,
                base,
                (i as i64 + 1) * size_of::<u64>() as i64,
            );
            self.asm.emit_r(Kind::VirtualSRL, shifted, low, srl_bitmask);
            self.asm.emit_r(Kind::MUL, carried, high, mul_pow2);
            match absorb {
                Absorb::Init => self.asm.emit_r(Kind::OR, *self.a[i], shifted, carried),
                Absorb::IntoState => {
                    self.asm.emit_r(Kind::OR, shifted, shifted, carried);
                    self.asm.xor(Reg(*self.a[i]), Reg(shifted), *self.a[i]);
                }
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

    fn d_register(&mut self, x: usize) -> &mut InlineRegister {
        match x {
            0..=2 => &mut self.d[x],
            3 => &mut self.c[1],
            4 => &mut self.c[2],
            _ => unreachable!("keccak D index out of range"),
        }
    }

    fn d_lane(&mut self, x: usize) -> u8 {
        **self.d_register(x)
    }

    fn theta(&mut self) {
        for x in 0..5 {
            let c = *self.c[x];
            // Rows 0..3 of every column are live, so at least three lanes remain.
            let live: Vec<u8> = (0..5)
                .filter(|&y| !self.zero_lanes[5 * y + x])
                .map(|y| self.lane(x, y))
                .collect();
            self.asm.xor(Reg(live[0]), Reg(live[1]), c);
            for &lane in &live[2..] {
                self.asm.xor(Reg(c), Reg(lane), c);
            }
        }

        // C[1] is last read at x=2 and C[2] at x=3, so D[3..5] can reuse them.
        for x in 0..5 {
            let d = self.d_lane(x);
            let c_prev = *self.c[(x + 4) % 5];
            let c_next = *self.c[(x + 1) % 5];
            self.asm.emit_r(Kind::VirtualXORROTL1, d, c_prev, c_next);
        }

        for x in 0..5 {
            let d = self.d_lane(x);
            // A zero lane becomes D[x] itself: the first one in the column
            // takes over D[x]'s register, the others copy it.
            let mut adopter = None;
            for y in 0..5 {
                let a = self.lane(x, y);
                if !self.zero_lanes[5 * y + x] {
                    self.asm.xor(Reg(a), Reg(d), a);
                } else if adopter.is_none() {
                    adopter = Some(5 * y + x);
                } else {
                    self.asm.xor(Reg(d), Imm(0), a);
                }
            }
            if let Some(lane) = adopter {
                self.adopt_d_register(lane, x);
            }
        }
        self.zero_lanes = [false; NUM_LANES];
    }

    /// Swaps the unwritten register of `lane` with D[x]'s: the lane now holds
    /// D[x]'s value and D[x] is recomputed into the spare register next round.
    fn adopt_d_register(&mut self, lane: usize, x: usize) {
        let mut spare = self.a[lane];
        core::mem::swap(&mut spare, self.d_register(x));
        self.a[lane] = spare;
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
        self.asm
            .rotl64(Reg(self.lane(x, y)), ROTATION_OFFSETS[x][y], destination);
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

/// Keccak-f[1600] over the 25 lanes at `rs1`, in place.
///
/// Deprecated: no in-repo caller emits this op (the SDK only ships the absorb
/// variants); it stays registered so legacy guest sequences still decode. Do
/// not emit it from new code.
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
        Keccak256SequenceBuilder::new(asm, operands, None)?.build()
    }
}

/// XORs the 136-byte block at `rs2` (8-byte aligned) into the state at `rs1`,
/// then Keccak-f[1600], in place.
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
        Keccak256SequenceBuilder::new(
            asm,
            operands,
            Some(Block {
                absorb: Absorb::IntoState,
                alignment: BlockAlignment::Aligned,
            }),
        )?
        .build()
    }
}

/// Keccak-f[1600] of the zero state XOR the 136-byte block at `rs2` (8-byte
/// aligned), written to `rs1`. The prior contents of `rs1` are not read.
pub struct Keccak256InitAbsorbPermutation;

impl InlineOp for Keccak256InitAbsorbPermutation {
    type Advice = NoAdvice;

    const OPCODE: u32 = INLINE_OPCODE;
    const FUNCT3: u32 = KECCAK256_INIT_ABSORB_PERMUTE_FUNCT3;
    const FUNCT7: u32 = KECCAK256_FUNCT7;
    const NAME: &'static str = KECCAK256_INIT_ABSORB_PERMUTE_NAME;

    fn build_sequence(
        asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        Keccak256SequenceBuilder::new(
            asm,
            operands,
            Some(Block {
                absorb: Absorb::Init,
                alignment: BlockAlignment::Aligned,
            }),
        )?
        .build()
    }
}

/// [`Keccak256AbsorbPermutation`] for a block at any alignment. The sequence
/// reads the 18 aligned doublewords `[rs2 & !7, (rs2 & !7) + 144)`: for a
/// misaligned `rs2` exactly the doublewords containing the block, for an
/// aligned `rs2` the block plus the doubleword after it (the SDK never issues
/// that case).
pub struct Keccak256AbsorbPermutationUnaligned;

impl InlineOp for Keccak256AbsorbPermutationUnaligned {
    type Advice = NoAdvice;

    const OPCODE: u32 = INLINE_OPCODE;
    const FUNCT3: u32 = KECCAK256_ABSORB_PERMUTE_UNALIGNED_FUNCT3;
    const FUNCT7: u32 = KECCAK256_FUNCT7;
    const NAME: &'static str = KECCAK256_ABSORB_PERMUTE_UNALIGNED_NAME;

    fn build_sequence(
        asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        Keccak256SequenceBuilder::new(
            asm,
            operands,
            Some(Block {
                absorb: Absorb::IntoState,
                alignment: BlockAlignment::Any,
            }),
        )?
        .build()
    }
}

/// [`Keccak256InitAbsorbPermutation`] for a block at any alignment, with the
/// memory contract of [`Keccak256AbsorbPermutationUnaligned`].
pub struct Keccak256InitAbsorbPermutationUnaligned;

impl InlineOp for Keccak256InitAbsorbPermutationUnaligned {
    type Advice = NoAdvice;

    const OPCODE: u32 = INLINE_OPCODE;
    const FUNCT3: u32 = KECCAK256_INIT_ABSORB_PERMUTE_UNALIGNED_FUNCT3;
    const FUNCT7: u32 = KECCAK256_FUNCT7;
    const NAME: &'static str = KECCAK256_INIT_ABSORB_PERMUTE_UNALIGNED_NAME;

    fn build_sequence(
        asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        Keccak256SequenceBuilder::new(
            asm,
            operands,
            Some(Block {
                absorb: Absorb::Init,
                alignment: BlockAlignment::Any,
            }),
        )?
        .build()
    }
}
