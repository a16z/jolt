//! This file contains Keccak256-specific logic to be used in the Keccak256 inline:
//! 1) Prover: Keccak256SequenceBuilder expands the inline to a list of RV instructions.
//! 2) Host: Rust reference implementation to be called by jolt-sdk.
//!
//! Keccak is a hash function that uses a sponge construction. The sponge absorbs (and permutes) data. Each permutation has 24 rounds. Then squeezes out the hash.
//! Glossary:
//!   - “Lane”  = one 64-bit word in the 5×5 state matrix (25 lanes total for Keccak256).
//!   - “Round” = single application of θ ρ π χ ι to the state.
//!   - “Rate”  = 1088 bits (136 B) that interact with the message/output.
//!   - “Capacity” = 512 bits hidden from the attacker (1600 − 1088).
//!   - “Permutation” = Keccak-f[1600] : 24 rounds, each θ→ρ→π→χ→ι.
//!
//! Keccak256 refers to the specific variant where the rate is 1088 bits and the capacity is 512 bits.
//! Keccak256 differs from SHA3-256 (not implemented here) in the padding scheme.
//!
//! # Row budget
//!
//! Every emitted instruction is one trace row, so the builder is written to
//! minimize instruction count rather than emulated-CPU time:
//! - The lane with rotation offset 0 (A[0,0]) is never copied into the B
//!   scratch state; χ reads the θ-updated A[0,0] register directly.
//! - θ-apply and ρ are fused into a single `VirtualXORROT*` row for every lane
//!   whose rotation has a fused XOR-ROT lookup table (see `xor_rot_for_amount`).
//! - The 5 D lanes reuse C registers as they die, keeping the virtual-register
//!   footprint (and the per-register reset rows appended after the sequence) low.

use crate::NUM_LANES;
use jolt_inlines_sdk::host::{
    instruction::andn::ANDN,
    ExpandedInstructionSequence, ExpansionError, InlineBuilderExt, InlineExpansionBuilder,
    InlineOp, InlineOperands, InlineRegister, NoAdvice,
    Value::{Imm, Reg},
};

/// The 24 round constants for the Keccak-f[1600] permutation.
/// These values are XORed into the state during the `iota` step of each round.
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

/// The rotation offsets for the `rho` step of the Keccak-f[1600] permutation.
/// The state is organized as a 5x5 matrix of 64-bit lanes, and `ROTATION_OFFSETS[x][y]`
/// specifies the left-rotation amount for the lane at `(x, y)`. Also known as rotation constants.
#[rustfmt::skip]
pub (crate) const ROTATION_OFFSETS: [[u32; 5]; 5] = [
    [ 0, 36,  3, 41, 18],
    [ 1, 44, 10, 45,  2],
    [62,  6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39,  8, 14],
];

/// `Keccak256SequenceBuilder` constructs the virtual instruction sequence that
/// performs the Keccak-f[1600] permutation on a 200-byte state in memory.
///
/// Register plan (58 virtual registers total):
/// - `a[0..25]`: the 25 lanes of the state array A.
/// - `b[0..24]`: the ρ/π scratch state B for every lane except (0,0). Lane
///   (0,0) has rotation offset 0, so B[0,0] is an alias of `a[0]` (χ reads the
///   θ-updated A[0,0] directly and overwrites it last within its row).
/// - `c[0..5]`: the θ column parities. D[3] and D[4] reuse `c[1]` and `c[2]`
///   once those parities are dead.
/// - `d[0..3]`: the first three θ D lanes.
/// - `scratch`: rotation temporary for the θ D computation.
struct Keccak256SequenceBuilder {
    asm: InlineExpansionBuilder,
    round: u32,
    a: [InlineRegister; NUM_LANES],
    b: [InlineRegister; NUM_LANES - 1],
    c: [InlineRegister; 5],
    d: [InlineRegister; 3],
    scratch: InlineRegister,
    operands: InlineOperands,
}

impl Keccak256SequenceBuilder {
    fn new(
        mut asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<Self, ExpansionError> {
        let a = asm.allocate_inline_array::<NUM_LANES>()?;
        let b = asm.allocate_inline_array::<{ NUM_LANES - 1 }>()?;
        let c = asm.allocate_inline_array::<5>()?;
        let d = asm.allocate_inline_array::<3>()?;
        let scratch = asm.allocate_for_inline()?;
        Ok(Keccak256SequenceBuilder {
            asm,
            round: 0,
            a,
            b,
            c,
            d,
            scratch,
            operands,
        })
    }

    fn build(mut self) -> Result<ExpandedInstructionSequence, ExpansionError> {
        // 1. Load NUM_LANES lanes (64-bit words) of state from memory into registers.
        self.load_state();

        // 2. Main loop: 24 rounds of Keccak-f permutation.
        for round in 0..24 {
            self.round = round;
            self.theta();
            self.rho_and_pi();
            self.chi();
            self.iota();
        }

        // 3. Store the final state back to memory.
        self.store_state();

        // 4. Finalize assembler and return instruction sequence.
        self.asm.release_many(self.a);
        self.asm.release_many(self.b);
        self.asm.release_many(self.c);
        self.asm.release_many(self.d);
        self.asm.release(self.scratch);
        self.asm.finalize()
    }

    /// Load the initial Keccak state from memory into virtual registers.
    /// Keccak state is NUM_LANES lanes of 64 bits each (200 bytes total).
    fn load_state(&mut self) {
        self.asm.load_u64_range(self.operands.rs1, 0, &self.a);
    }

    /// Store the final Keccak state from virtual registers back to memory.
    fn store_state(&mut self) {
        self.asm.store_u64_range(self.operands.rs1, 0, &self.a);
    }

    /// Get the register for lane (x, y) of the state matrix A.
    fn lane(&self, x: usize, y: usize) -> u8 {
        *self.a[5 * y + x]
    }

    /// Get the register for lane (x, y) of the ρ/π scratch state B.
    ///
    /// B[0,0] is an alias of A[0,0]: ρ rotates that lane by 0, so the
    /// θ-updated A[0,0] register already holds its value.
    fn b_lane(&self, x: usize, y: usize) -> u8 {
        if x == 0 && y == 0 {
            *self.a[0]
        } else {
            *self.b[5 * y + x - 1]
        }
    }

    /// Get the register holding θ's D[x].
    ///
    /// D[0..3] live in dedicated registers; D[3] and D[4] reuse the C[1] and
    /// C[2] registers, which are dead by the time they are written.
    fn d_lane(&self, x: usize) -> u8 {
        match x {
            0..=2 => *self.d[x],
            3 => *self.c[1],
            4 => *self.c[2],
            _ => unreachable!("keccak D index out of range"),
        }
    }

    // --- Keccak-f Round Functions ---

    fn theta(&mut self) {
        // --- C[x] = A[x,0] ^ A[x,1] ^ A[x,2] ^ A[x,3] ^ A[x,4] ---
        for x in 0..5 {
            let c_reg = *self.c[x];
            // c_reg = A[x,0] ^ A[x,1]
            self.asm
                .xor(Reg(self.lane(x, 0)), Reg(self.lane(x, 1)), c_reg);
            // c_reg ^= A[x,2] ^ A[x,3] ^ A[x,4]
            for y in 2..5 {
                self.asm.xor(Reg(c_reg), Reg(self.lane(x, y)), c_reg);
            }
        }

        // --- D[x] = C[x-1] ^ rotl(C[x+1], 1) ---
        //
        // Computed in order x = 0..4 so that D[3] and D[4] can safely reuse
        // the C[1] and C[2] registers: C[1] is last read at x = 2 and C[2] at
        // x = 3.
        for x in 0..5 {
            let d_reg = self.d_lane(x);
            let c_prev = *self.c[(x + 4) % 5];
            let c_next = *self.c[(x + 1) % 5];
            let temp_rot_reg = *self.scratch;

            self.asm.rotl64(Reg(c_next), 1, temp_rot_reg);
            self.asm.xor(Reg(c_prev), Reg(temp_rot_reg), d_reg);
        }

        // --- A[x,y] ^= D[x] ---
        //
        // Lanes whose ρ rotation has a fused XOR-ROT table skip this row; the
        // XOR is folded into the ρ step instead (see `rho_and_pi`).
        #[expect(clippy::needless_range_loop)] // (x, y) indexing mirrors the spec
        for x in 0..5 {
            let d_reg = self.d_lane(x);
            for y in 0..5 {
                if xor_rot_for_amount(ROTATION_OFFSETS[x][y]).is_some() {
                    continue;
                }
                let a_reg = self.lane(x, y);
                self.asm.xor(Reg(a_reg), Reg(d_reg), a_reg);
            }
        }
    }

    fn rho_and_pi(&mut self) {
        // This function combines two steps:
        // 1. Rho (ρ): Rotates each lane A[x,y] by a fixed offset.
        // 2. Pi (π): Permutes the lanes into a new configuration.
        //
        // The combined operation is: B[y, 2x+3y] = ROTL(A[x,y], offset)
        //
        // Lane (0,0) has offset 0 and B[0,0] aliases A[0,0], so it is skipped
        // entirely. Lanes whose rotation has a fused XOR-ROT lookup table
        // compute B[y, 2x+3y] = ROTL(A[x,y] ^ D[x], offset) in a single row,
        // absorbing their θ-apply XOR.
        #[expect(clippy::needless_range_loop)] // This is clearer than enumerating
        for x in 0..5 {
            for y in 0..5 {
                if x == 0 && y == 0 {
                    continue;
                }
                // Get the source lane A[x,y] and its rotation offset.
                let source_reg = self.lane(x, y);
                // We have checked that this is [x][y].
                let rotation_offset = ROTATION_OFFSETS[x][y];

                // Calculate the permuted destination coordinates in B.
                let nx = y;
                let ny = (2 * x + 3 * y) % 5;
                let dest_reg_in_b = self.b_lane(nx, ny);

                if let Some(xor_rot) = xor_rot_for_amount(rotation_offset) {
                    // Fused θ-apply + ρ: B[nx, ny] = ROTL(A[x,y] ^ D[x], offset).
                    let d_reg = self.d_lane(x);
                    xor_rot.emit(&mut self.asm, dest_reg_in_b, source_reg, d_reg);
                } else {
                    // Rotate the θ-updated A[x,y] and store the result in B[nx, ny].
                    self.asm
                        .rotl64(Reg(source_reg), rotation_offset, dest_reg_in_b);
                }
            }
        }
    }

    fn chi(&mut self) {
        // The chi step provides non-linearity. For each row, it updates each lane as:
        // A[x,y] ^= (~A[x+1,y] & A[x+2,y])
        //
        // Row 0 processes x = 0 last: B[0,0] aliases A[0,0], and B[0,0] is
        // read by the x = 3 and x = 4 lanes, so A[0,0] must be overwritten
        // only after those reads.
        for y in 0..5 {
            let x_order: [usize; 5] = if y == 0 {
                [1, 2, 3, 4, 0]
            } else {
                [0, 1, 2, 3, 4]
            };
            for x in x_order {
                // Get the registers for the three input values
                // B[x,y], B[x+1,y], B[x+2,y]
                let current = self.b_lane(x, y);
                let next = self.b_lane((x + 1) % 5, y);
                let two_next = self.b_lane((x + 2) % 5, y);

                // Scratch register for the intermediate ANDN result.
                let not_next_and_two_next = *self.scratch;

                // Get the register for the lane we are updating in the main state A.
                let dest_a_reg = self.lane(x, y);

                // Implement A[x,y] = B[x,y] ^ (~B[x+1,y] & B[x+2,y])
                // 1. not_next_and_two_next = B[x+2,y] & ~B[x+1,y] using ANDN
                self.asm
                    .emit_r::<ANDN>(not_next_and_two_next, two_next, next);
                // 2. A[x,y] = B[x,y] ^ not_next_and_two_next
                self.asm
                    .xor(Reg(current), Reg(not_next_and_two_next), dest_a_reg);
            }
        }
    }

    fn iota(&mut self) {
        // The iota step breaks symmetry by XORing a round-specific constant
        // into the first lane of the state, A[0,0].
        let round_constant = ROUND_CONSTANTS[self.round as usize];
        let first_lane_reg = self.lane(0, 0);
        self.asm
            .xor(Reg(first_lane_reg), Imm(round_constant), first_lane_reg);
    }
}

/// A fused XOR-then-rotate lookup instruction: `rd = rotr64(rs1 ^ rs2, N)`.
///
/// Emitting one of these instead of a separate XOR + ROTRI pair saves one row
/// per lane per round, which is why θ-apply is folded into ρ wherever the
/// rotation amount has a table. Every nonzero Keccak ρ rotation has one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct XorRot {
    /// Right-rotation amount (`64 - rho_left_rotation`).
    rotr: u32,
}

macro_rules! xor_rot_dispatch {
    ($self:ident, $asm:ident, $rd:ident, $rs1:ident, $rs2:ident, [$($n:literal => $instr:ident),+ $(,)?]) => {
        match $self.rotr {
            $($n => $asm.emit_r::<jolt_inlines_sdk::host::instruction::virtual_xor_rot::$instr>($rd, $rs1, $rs2),)+
            _ => unreachable!("no VirtualXORROT table for rotation {}", $self.rotr),
        }
    };
}

impl XorRot {
    fn emit(self, asm: &mut InlineExpansionBuilder, rd: u8, rs1: u8, rs2: u8) {
        xor_rot_dispatch!(
            self,
            asm,
            rd,
            rs1,
            rs2,
            [
                2 => VirtualXORROT2,
                3 => VirtualXORROT3,
                8 => VirtualXORROT8,
                9 => VirtualXORROT9,
                19 => VirtualXORROT19,
                20 => VirtualXORROT20,
                21 => VirtualXORROT21,
                23 => VirtualXORROT23,
                25 => VirtualXORROT25,
                28 => VirtualXORROT28,
                36 => VirtualXORROT36,
                37 => VirtualXORROT37,
                39 => VirtualXORROT39,
                43 => VirtualXORROT43,
                44 => VirtualXORROT44,
                46 => VirtualXORROT46,
                49 => VirtualXORROT49,
                50 => VirtualXORROT50,
                54 => VirtualXORROT54,
                56 => VirtualXORROT56,
                58 => VirtualXORROT58,
                61 => VirtualXORROT61,
                62 => VirtualXORROT62,
                63 => VirtualXORROT63,
            ]
        )
    }
}

/// Map a ρ left-rotation amount to the fused XOR-ROT instruction computing
/// `rotl64(a ^ b, amount)`, if one exists.
///
/// `rotl64(v, r)` = `rotr64(v, 64 - r)`, so a left rotation by `r` needs the
/// `VirtualXORROT{64-r}` table. All 24 nonzero ρ rotations are covered; only
/// the identity rotation of lane (0,0) has none.
fn xor_rot_for_amount(rotl_amount: u32) -> Option<XorRot> {
    match rotl_amount {
        0 => None,
        1..=63 => Some(XorRot {
            rotr: 64 - rotl_amount,
        }),
        _ => unreachable!("keccak rho rotation out of range: {rotl_amount}"),
    }
}

pub struct Keccak256Permutation;

impl InlineOp for Keccak256Permutation {
    type Advice = NoAdvice;

    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::KECCAK256_FUNCT3;
    const FUNCT7: u32 = crate::KECCAK256_FUNCT7;
    const NAME: &'static str = crate::KECCAK256_NAME;

    fn build_sequence(
        asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        Keccak256SequenceBuilder::new(asm, operands)?.build()
    }
}
