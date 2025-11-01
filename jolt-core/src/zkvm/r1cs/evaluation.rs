//! R1CS evaluation utilities: grouped Az/Bz evaluators and Lagrange-folded helpers
//!
//! This module contains the runtime evaluation semantics for the uniform R1CS
//! constraints declared in `r1cs::constraints`.
//!
//! What lives here:
//! - Typed guard/magnitude structs for each group: `AzFirstGroup`, `BzFirstGroup`,
//!   `AzSecondGroup`, `BzSecondGroup`.
//! - Typed evaluator wrappers: `R1CSFirstGroup` and `R1CSSecondGroup` with
//!   `eval_az`, `eval_bz`, `az_at_r`, `bz_at_r`, and helper fold functions for
//!   Lagrange-window accumulation.
//!
//! What does NOT live here:
//! - The definition of constraints and grouping metadata (see `r1cs::constraints`).
//!
//! When adding/removing constraints:
//! - Update `r1cs::constraints` for the new constraint and its group assignment.
//! - If guard/magnitude shapes change for either group, update the corresponding
//!   `Az*/Bz*` structs and evaluator logic here to remain consistent.

use super::inputs::R1CSCycleInputs;
use crate::field::{BarrettReduce, FMAdd, JoltField};
use crate::utils::accumulation::{Acc5U, Acc6S, Acc7S, S128Sum};
use crate::zkvm::instruction::CircuitFlags;
use ark_ff::biginteger::{S128, S160, S192, S64};

use super::constraints::UNIVARIATE_SKIP_DOMAIN_SIZE;

#[inline]
fn s64_from_diff_u64s(a: u64, b: u64) -> S64 {
    let diff = (a as i128) - (b as i128);
    debug_assert!(diff >= (i64::MIN as i128) && diff <= (i64::MAX as i128));
    S64::from_i64(diff as i64)
}

/// Boolean guards for the first group (univariate-skip base window)
#[derive(Clone, Copy, Debug)]
pub struct AzFirstGroup {
    pub ram_addr_eq_zero_if_not_load_store: bool, // !(Load || Store)
    pub ram_read_eq_ram_write_if_load: bool,      // Load
    pub ram_read_eq_rd_write_if_load: bool,       // Load
    pub rs2_eq_ram_write_if_store: bool,          // Store
    pub left_lookup_zero_unless_add_sub_mul: bool, // Add || Sub || Mul
    pub left_lookup_eq_left_input_otherwise: bool, // !(Add || Sub || Mul)
    pub assert_lookup_one: bool,                  // Assert
    pub next_unexp_pc_eq_lookup_if_should_jump: bool, // ShouldJump
    pub next_pc_eq_pc_plus_one_if_inline: bool,   // VirtualInstruction
    pub must_start_sequence_from_beginning: bool, // NextIsVirtual && !NextIsFirstInSequence
}

/// Magnitudes for the first group (kept small: bool/u64/S64)
#[derive(Clone, Copy, Debug)]
pub struct BzFirstGroup {
    pub ram_addr: u64,                               // RamAddress - 0
    pub ram_read_minus_ram_write: S64,               // RamRead - RamWrite
    pub ram_read_minus_rd_write: S64,                // RamRead - RdWrite
    pub rs2_minus_ram_write: S64,                    // Rs2 - RamWrite
    pub left_lookup: u64,                            // LeftLookup - 0
    pub left_lookup_minus_left_input: S64,           // LeftLookup - LeftInstructionInput
    pub lookup_output_minus_one: S64,                // LookupOutput - 1
    pub next_unexp_pc_minus_lookup_output: S64,      // NextUnexpandedPC - LookupOutput
    pub next_pc_minus_pc_plus_one: S64,              // NextPC - (PC + 1)
    pub do_not_update_unexpanded_pc_minus_one: bool, // 1 - DoNotUpdateUnexpandedPC
}

/// Guards for the second group (all booleans except two u8 flags)
#[derive(Clone, Copy, Debug)]
pub struct AzSecondGroup {
    pub ram_addr_eq_rs1_plus_imm_if_load_store: bool, // Load || Store
    pub right_lookup_add: bool,                       // Add
    pub right_lookup_sub: bool,                       // Sub
    pub right_lookup_eq_product_if_mul: bool,         // Mul
    pub right_lookup_eq_right_input_otherwise: bool,  // !(Add || Sub || Mul || Advice)
    pub rd_write_eq_lookup_if_write_lookup_to_rd: bool, // write_lookup_output_to_rd_addr (Rd != 0)
    pub rd_write_eq_pc_plus_const_if_write_pc_to_rd: bool, // write_pc_to_rd_addr (Rd != 0)
    pub next_unexp_pc_eq_pc_plus_imm_if_should_branch: bool, // ShouldBranch
    pub next_unexp_pc_update_otherwise: bool,         // !(Jump || ShouldBranch)
}

/// Magnitudes for the second group (mixed precision up to S160)
#[derive(Clone, Copy, Debug)]
pub struct BzSecondGroup {
    pub ram_addr_minus_rs1_plus_imm: i128, // RamAddress - (Rs1 + Imm)
    pub right_lookup_minus_add_result: S160, // RightLookup - (Left + Right)
    pub right_lookup_minus_sub_result: S160, // RightLookup - (Left - Right + 2^64)
    pub right_lookup_minus_product: S160,  // RightLookup - Product
    pub right_lookup_minus_right_input: S160, // RightLookup - RightInput
    pub rd_write_minus_lookup_output: S64, // RdWrite - LookupOutput
    pub rd_write_minus_pc_plus_const: i128, // RdWrite - (UnexpandedPC + const)
    pub next_unexp_pc_minus_pc_plus_imm: i128, // NextUnexpandedPC - (UnexpandedPC + Imm)
    pub next_unexp_pc_minus_expected: i128, // NextUnexpandedPC - (UnexpandedPC + const)
}

/// First-group evaluator wrapper with typed accessors
#[derive(Clone, Copy, Debug)]
pub struct R1CSFirstGroup<'a, F: JoltField> {
    row: &'a R1CSCycleInputs,
    _m: core::marker::PhantomData<F>,
}

impl<'a, F: JoltField> R1CSFirstGroup<'a, F> {
    #[inline]
    pub fn from_cycle_inputs(row: &'a R1CSCycleInputs) -> Self {
        Self {
            row,
            _m: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn eval_az(&self) -> AzFirstGroup {
        let flags = &self.row.flags;
        let ld = flags[CircuitFlags::Load];
        let st = flags[CircuitFlags::Store];
        let add = flags[CircuitFlags::AddOperands];
        let sub = flags[CircuitFlags::SubtractOperands];
        let mul = flags[CircuitFlags::MultiplyOperands];
        let assert_flag = flags[CircuitFlags::Assert];
        let inline_seq = flags[CircuitFlags::VirtualInstruction];

        AzFirstGroup {
            ram_addr_eq_zero_if_not_load_store: !(ld || st),
            ram_read_eq_ram_write_if_load: ld,
            ram_read_eq_rd_write_if_load: ld,
            rs2_eq_ram_write_if_store: st,
            left_lookup_zero_unless_add_sub_mul: add || sub || mul,
            left_lookup_eq_left_input_otherwise: !(add || sub || mul),
            assert_lookup_one: assert_flag,
            next_unexp_pc_eq_lookup_if_should_jump: self.row.should_jump,
            next_pc_eq_pc_plus_one_if_inline: inline_seq,
            must_start_sequence_from_beginning: self.row.next_is_virtual
                && !self.row.next_is_first_in_sequence,
        }
    }

    #[inline]
    pub fn eval_bz(&self) -> BzFirstGroup {
        BzFirstGroup {
            ram_addr: self.row.ram_addr,
            ram_read_minus_ram_write: s64_from_diff_u64s(
                self.row.ram_read_value,
                self.row.ram_write_value,
            ),
            ram_read_minus_rd_write: s64_from_diff_u64s(
                self.row.ram_read_value,
                self.row.rd_write_value,
            ),
            rs2_minus_ram_write: s64_from_diff_u64s(
                self.row.rs2_read_value,
                self.row.ram_write_value,
            ),
            left_lookup: self.row.left_lookup,
            left_lookup_minus_left_input: s64_from_diff_u64s(
                self.row.left_lookup,
                self.row.left_input,
            ),
            lookup_output_minus_one: s64_from_diff_u64s(self.row.lookup_output, 1),
            next_unexp_pc_minus_lookup_output: s64_from_diff_u64s(
                self.row.next_unexpanded_pc,
                self.row.lookup_output,
            ),
            next_pc_minus_pc_plus_one: s64_from_diff_u64s(
                self.row.next_pc,
                self.row.pc.wrapping_add(1),
            ),
            do_not_update_unexpanded_pc_minus_one: !self.row.flags
                [CircuitFlags::DoNotUpdateUnexpandedPC],
        }
    }

    #[inline]
    pub fn az_at_r(&self, w: &[F; UNIVARIATE_SKIP_DOMAIN_SIZE]) -> F {
        let az = self.eval_az();
        let mut acc: Acc5U<F> = Acc5U::default();
        acc.fmadd(&w[0], &az.ram_addr_eq_zero_if_not_load_store);
        acc.fmadd(&w[1], &az.ram_read_eq_ram_write_if_load);
        acc.fmadd(&w[2], &az.ram_read_eq_rd_write_if_load);
        acc.fmadd(&w[3], &az.rs2_eq_ram_write_if_store);
        acc.fmadd(&w[4], &az.left_lookup_zero_unless_add_sub_mul);
        acc.fmadd(&w[5], &az.left_lookup_eq_left_input_otherwise);
        acc.fmadd(&w[6], &az.assert_lookup_one);
        acc.fmadd(&w[7], &az.next_unexp_pc_eq_lookup_if_should_jump);
        acc.fmadd(&w[8], &az.next_pc_eq_pc_plus_one_if_inline);
        acc.fmadd(&w[9], &az.must_start_sequence_from_beginning);
        acc.barrett_reduce()
    }

    #[inline]
    pub fn bz_at_r(&self, w: &[F; UNIVARIATE_SKIP_DOMAIN_SIZE]) -> F {
        let bz = self.eval_bz();
        let mut acc: Acc6S<F> = Acc6S::default();
        acc.fmadd(&w[0], &bz.ram_addr);
        acc.fmadd(&w[1], &bz.ram_read_minus_ram_write);
        acc.fmadd(&w[2], &bz.ram_read_minus_rd_write);
        acc.fmadd(&w[3], &bz.rs2_minus_ram_write);
        acc.fmadd(&w[4], &bz.left_lookup);
        acc.fmadd(&w[5], &bz.left_lookup_minus_left_input);
        acc.fmadd(&w[6], &bz.lookup_output_minus_one);
        acc.fmadd(&w[7], &bz.next_unexp_pc_minus_lookup_output);
        acc.fmadd(&w[8], &bz.next_pc_minus_pc_plus_one);
        acc.fmadd(&w[9], &bz.do_not_update_unexpanded_pc_minus_one);
        acc.barrett_reduce()
    }

    #[inline]
    pub fn product_of_sums_shifted(
        &self,
        coeffs_i32: &[i32; UNIVARIATE_SKIP_DOMAIN_SIZE],
        _coeffs_s64: &[S64; UNIVARIATE_SKIP_DOMAIN_SIZE],
    ) -> S192 {
        let az = self.eval_az();
        let bz = self.eval_bz();

        let mut sum_c_az_i32: i32 = 0;
        let mut sum_bz: S128Sum = S128Sum::default();

        // 0: RamAddrEqZeroIfNotLoadStore — RamAddress == 0 when !(Load || Store)
        let c0_i32 = coeffs_i32[0];
        if az.ram_addr_eq_zero_if_not_load_store {
            sum_c_az_i32 += c0_i32;
        } else {
            let cz = S128::from_i128(c0_i32 as i128);
            sum_bz.fmadd(&cz, &bz.ram_addr);
        }

        // 1: RamReadEqRamWriteIfLoad — RamRead == RamWrite when Load
        let c1_i32 = coeffs_i32[1];
        if az.ram_read_eq_ram_write_if_load {
            sum_c_az_i32 += c1_i32;
        } else {
            let cz = S128::from_i128(c1_i32 as i128);
            sum_bz.fmadd(&cz, &bz.ram_read_minus_ram_write);
        }

        // 2: RamReadEqRdWriteIfLoad — RamRead == RdWrite when Load
        let c2_i32 = coeffs_i32[2];
        if az.ram_read_eq_rd_write_if_load {
            sum_c_az_i32 += c2_i32;
        } else {
            let cz = S128::from_i128(c2_i32 as i128);
            sum_bz.fmadd(&cz, &bz.ram_read_minus_rd_write);
        }

        // 3: Rs2EqRamWriteIfStore — Rs2 == RamWrite when Store
        let c3_i32 = coeffs_i32[3];
        if az.rs2_eq_ram_write_if_store {
            sum_c_az_i32 += c3_i32;
        } else {
            let cz = S128::from_i128(c3_i32 as i128);
            sum_bz.fmadd(&cz, &bz.rs2_minus_ram_write);
        }

        // 4: LeftLookupZeroUnlessAddSubMul — LeftLookup == 0 when (Add||Sub||Mul)
        let c4_i32 = coeffs_i32[4];
        if az.left_lookup_zero_unless_add_sub_mul {
            sum_c_az_i32 += c4_i32;
        } else {
            let cz = S128::from_i128(c4_i32 as i128);
            sum_bz.fmadd(&cz, &bz.left_lookup);
        }

        // 5: LeftLookupEqLeftInputOtherwise — LeftLookup == LeftInstructionInput when !(Add||Sub||Mul)
        let c5_i32 = coeffs_i32[5];
        if az.left_lookup_eq_left_input_otherwise {
            sum_c_az_i32 += c5_i32;
        } else {
            let cz = S128::from_i128(c5_i32 as i128);
            sum_bz.fmadd(&cz, &bz.left_lookup_minus_left_input);
        }

        // 6: AssertLookupOne — LookupOutput == 1 when Assert
        let c6_i32 = coeffs_i32[6];
        if az.assert_lookup_one {
            sum_c_az_i32 += c6_i32;
        } else {
            let cz = S128::from_i128(c6_i32 as i128);
            sum_bz.fmadd(&cz, &bz.lookup_output_minus_one);
        }

        // 7: NextUnexpPCEqLookupIfShouldJump — NextUnexpandedPC == LookupOutput when ShouldJump
        let c7_i32 = coeffs_i32[7];
        if az.next_unexp_pc_eq_lookup_if_should_jump {
            sum_c_az_i32 += c7_i32;
        } else {
            let cz = S128::from_i128(c7_i32 as i128);
            sum_bz.fmadd(&cz, &bz.next_unexp_pc_minus_lookup_output);
        }

        // 8: NextPCEqPCPlusOneIfInline — NextPC == PC + 1 when VirtualInstruction
        let c8_i32 = coeffs_i32[8];
        if az.next_pc_eq_pc_plus_one_if_inline {
            sum_c_az_i32 += c8_i32;
        } else {
            let cz = S128::from_i128(c8_i32 as i128);
            sum_bz.fmadd(&cz, &bz.next_pc_minus_pc_plus_one);
        }

        // 9: MustStartSequenceFromBeginning — DoNotUpdateUnexpandedPC == 1 when NextIsVirtual && !NextIsFirstInSequence
        let c9_i32 = coeffs_i32[9];
        if az.must_start_sequence_from_beginning {
            sum_c_az_i32 += c9_i32;
        } else {
            // Bz = DoNotUpdateUnexpandedPC - 1 = -(1 - DoNotUpdateUnexpandedPC)
            if bz.do_not_update_unexpanded_pc_minus_one {
                sum_bz.sum -= S128::from_i128(c9_i32 as i128);
            }
        }

        let sum_az_s64 = S64::from_i64(sum_c_az_i32 as i64);
        sum_az_s64.mul_trunc::<2, 3>(&sum_bz.sum)
    }

    #[cfg(test)]
    #[inline]
    pub fn debug_assert_zero_when_guarded(&self) {
        let az = self.eval_az();
        let bz = self.eval_bz();
        debug_assert!((!az.ram_addr_eq_zero_if_not_load_store) || bz.ram_addr == 0);
        debug_assert!(
            (!az.ram_read_eq_ram_write_if_load) || bz.ram_read_minus_ram_write.to_i128() == 0
        );
        debug_assert!(
            (!az.ram_read_eq_rd_write_if_load) || bz.ram_read_minus_rd_write.to_i128() == 0
        );
        debug_assert!((!az.rs2_eq_ram_write_if_store) || bz.rs2_minus_ram_write.to_i128() == 0);
        debug_assert!((!az.left_lookup_zero_unless_add_sub_mul) || bz.left_lookup == 0);
        debug_assert!(
            (!az.left_lookup_eq_left_input_otherwise)
                || bz.left_lookup_minus_left_input.to_i128() == 0
        );
        debug_assert!((!az.assert_lookup_one) || bz.lookup_output_minus_one.to_i128() == 0);
        debug_assert!(
            (!az.next_unexp_pc_eq_lookup_if_should_jump)
                || bz.next_unexp_pc_minus_lookup_output.to_i128() == 0
        );
        debug_assert!(
            (!az.next_pc_eq_pc_plus_one_if_inline) || bz.next_pc_minus_pc_plus_one.to_i128() == 0
        );
        debug_assert!(
            (!az.must_start_sequence_from_beginning) || bz.do_not_update_unexpanded_pc_minus_one
        );
    }
}

/// Second-group evaluator wrapper with typed accessors
#[derive(Clone, Copy, Debug)]
pub struct R1CSSecondGroup<'a, F: JoltField> {
    row: &'a R1CSCycleInputs,
    _m: core::marker::PhantomData<F>,
}

impl<'a, F: JoltField> R1CSSecondGroup<'a, F> {
    #[inline]
    pub fn from_cycle_inputs(row: &'a R1CSCycleInputs) -> Self {
        Self {
            row,
            _m: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn eval_az(&self) -> AzSecondGroup {
        let flags = &self.row.flags;
        let not_add_sub_mul_advice = !(flags[CircuitFlags::AddOperands]
            || flags[CircuitFlags::SubtractOperands]
            || flags[CircuitFlags::MultiplyOperands]
            || flags[CircuitFlags::Advice]);
        let next_update_otherwise = {
            let jump = flags[CircuitFlags::Jump];
            let should_branch = self.row.should_branch;
            (!jump) && (!should_branch)
        };

        AzSecondGroup {
            ram_addr_eq_rs1_plus_imm_if_load_store: (flags[CircuitFlags::Load]
                || flags[CircuitFlags::Store]),
            right_lookup_add: flags[CircuitFlags::AddOperands],
            right_lookup_sub: flags[CircuitFlags::SubtractOperands],
            right_lookup_eq_product_if_mul: flags[CircuitFlags::MultiplyOperands],
            right_lookup_eq_right_input_otherwise: not_add_sub_mul_advice,
            rd_write_eq_lookup_if_write_lookup_to_rd: self.row.write_lookup_output_to_rd_addr,
            rd_write_eq_pc_plus_const_if_write_pc_to_rd: self.row.write_pc_to_rd_addr,
            next_unexp_pc_eq_pc_plus_imm_if_should_branch: self.row.should_branch,
            next_unexp_pc_update_otherwise: next_update_otherwise,
        }
    }

    #[inline]
    pub fn eval_bz(&self) -> BzSecondGroup {
        // RamAddrEqRs1PlusImmIfLoadStore
        let expected_addr: i128 = if self.row.imm.is_positive {
            (self.row.rs1_read_value as u128 + self.row.imm.magnitude_as_u64() as u128) as i128
        } else {
            self.row.rs1_read_value as i128 - self.row.imm.magnitude_as_u64() as i128
        };
        let ram_addr_minus_rs1_plus_imm = self.row.ram_addr as i128 - expected_addr;

        // RightLookupAdd / Sub / Product / RightInput
        let right_add_expected = (self.row.left_input as i128) + self.row.right_input.to_i128();
        let right_sub_expected =
            (self.row.left_input as i128) - self.row.right_input.to_i128() + (1i128 << 64);

        let right_lookup_minus_add_result =
            S160::from(self.row.right_lookup) - S160::from(right_add_expected);
        let right_lookup_minus_sub_result =
            S160::from(self.row.right_lookup) - S160::from(right_sub_expected);
        let right_lookup_minus_product =
            S160::from(self.row.right_lookup) - S160::from(self.row.product);
        let right_lookup_minus_right_input =
            S160::from(self.row.right_lookup) - S160::from(self.row.right_input);

        // Rd write checks (fit in i64 range by construction)
        let rd_write_minus_lookup_output =
            s64_from_diff_u64s(self.row.rd_write_value, self.row.lookup_output);
        let const_term = 4 - if self.row.flags[CircuitFlags::IsCompressed] {
            2
        } else {
            0
        };
        let rd_write_minus_pc_plus_const = (self.row.rd_write_value as i128)
            - ((self.row.unexpanded_pc as i128) + (const_term as i128));

        // Next unexpanded PC checks
        let next_unexp_pc_minus_pc_plus_imm = (self.row.next_unexpanded_pc as i128)
            - (self.row.unexpanded_pc as i128 + self.row.imm.to_i128());
        let const_term_next =
            4 - if self.row.flags[CircuitFlags::DoNotUpdateUnexpandedPC] {
                4
            } else {
                0
            } - if self.row.flags[CircuitFlags::IsCompressed] {
                2
            } else {
                0
            };
        let target_next = self.row.unexpanded_pc as i128 + const_term_next;
        let next_unexp_pc_minus_expected = self.row.next_unexpanded_pc as i128 - target_next;

        BzSecondGroup {
            ram_addr_minus_rs1_plus_imm,
            right_lookup_minus_add_result,
            right_lookup_minus_sub_result,
            right_lookup_minus_product,
            right_lookup_minus_right_input,
            rd_write_minus_lookup_output,
            rd_write_minus_pc_plus_const,
            next_unexp_pc_minus_pc_plus_imm,
            next_unexp_pc_minus_expected,
        }
    }

    #[inline]
    pub fn az_at_r(&self, _w: &[F; UNIVARIATE_SKIP_DOMAIN_SIZE]) -> F {
        let w = _w;
        let az = self.eval_az();
        let mut acc: Acc5U<F> = Acc5U::default();
        acc.fmadd(&w[0], &az.ram_addr_eq_rs1_plus_imm_if_load_store);
        acc.fmadd(&w[1], &az.right_lookup_add);
        acc.fmadd(&w[2], &az.right_lookup_sub);
        acc.fmadd(&w[3], &az.right_lookup_eq_product_if_mul);
        acc.fmadd(&w[4], &az.right_lookup_eq_right_input_otherwise);
        acc.fmadd(&w[5], &az.rd_write_eq_lookup_if_write_lookup_to_rd);
        acc.fmadd(&w[6], &az.rd_write_eq_pc_plus_const_if_write_pc_to_rd);
        acc.fmadd(&w[7], &az.next_unexp_pc_eq_pc_plus_imm_if_should_branch);
        acc.fmadd(&w[8], &az.next_unexp_pc_update_otherwise);
        acc.barrett_reduce()
    }

    #[inline]
    pub fn bz_at_r(&self, _w: &[F; UNIVARIATE_SKIP_DOMAIN_SIZE]) -> F {
        let w = _w;
        let bz = self.eval_bz();
        let mut acc: Acc7S<F> = Acc7S::default();
        acc.fmadd(&w[0], &bz.ram_addr_minus_rs1_plus_imm);
        acc.fmadd(&w[1], &bz.right_lookup_minus_add_result);
        acc.fmadd(&w[2], &bz.right_lookup_minus_sub_result);
        acc.fmadd(&w[3], &bz.right_lookup_minus_product);
        acc.fmadd(&w[4], &bz.right_lookup_minus_right_input);
        acc.fmadd(&w[5], &bz.rd_write_minus_lookup_output);
        acc.fmadd(&w[6], &bz.rd_write_minus_pc_plus_const);
        acc.fmadd(&w[7], &bz.next_unexp_pc_minus_pc_plus_imm);
        acc.fmadd(&w[8], &bz.next_unexp_pc_minus_expected);
        acc.barrett_reduce()
    }

    #[inline]
    pub fn product_of_sums_shifted(
        &self,
        _coeffs_i32: &[i32; UNIVARIATE_SKIP_DOMAIN_SIZE],
        _coeffs_s64: &[S64; UNIVARIATE_SKIP_DOMAIN_SIZE],
    ) -> S192 {
        #[cfg(test)]
        self.debug_assert_zero_when_guarded();

        let coeffs_i32 = _coeffs_i32;
        let az = self.eval_az();
        let bz = self.eval_bz();

        let mut sum_c_az_i64: i64 = 0;
        let mut sum_bz_s160 = S160::from(0i128);

        // Helper to add c * bz_term when needed
        #[inline(always)]
        fn fmadd_bz(sum_bz: &mut S160, c_i64: i64, bz_term: S160) {
            let c_s160 = S160::from(c_i64 as i128);
            *sum_bz += (&c_s160) * (&bz_term);
        }

        // 0: Load||Store guard (bool), Bz = ram_addr - (rs1 + imm)
        let c0 = coeffs_i32[0] as i64;
        if az.ram_addr_eq_rs1_plus_imm_if_load_store {
            sum_c_az_i64 += c0;
        } else {
            fmadd_bz(
                &mut sum_bz_s160,
                c0,
                S160::from(bz.ram_addr_minus_rs1_plus_imm),
            );
        }

        // 1: Add, Bz = right_lookup - (left + right)
        let c1 = coeffs_i32[1] as i64;
        if az.right_lookup_add {
            sum_c_az_i64 += c1;
        } else {
            fmadd_bz(&mut sum_bz_s160, c1, bz.right_lookup_minus_add_result);
        }

        // 2: Sub, Bz = right_lookup - (left - right + 2^64)
        let c2 = coeffs_i32[2] as i64;
        if az.right_lookup_sub {
            sum_c_az_i64 += c2;
        } else {
            fmadd_bz(&mut sum_bz_s160, c2, bz.right_lookup_minus_sub_result);
        }

        // 3: Mul, Bz = right_lookup - product
        let c3 = coeffs_i32[3] as i64;
        if az.right_lookup_eq_product_if_mul {
            sum_c_az_i64 += c3;
        } else {
            fmadd_bz(&mut sum_bz_s160, c3, bz.right_lookup_minus_product);
        }

        // 4: !(Add||Sub||Mul||Advice), Bz = right_lookup - right_input
        let c4 = coeffs_i32[4] as i64;
        if az.right_lookup_eq_right_input_otherwise {
            sum_c_az_i64 += c4;
        } else {
            fmadd_bz(&mut sum_bz_s160, c4, bz.right_lookup_minus_right_input);
        }

        // 5: rd_write_eq_lookup_if_write_lookup_to_rd (u8), Bz = rd_write - lookup_output (S64)
        let c5 = coeffs_i32[5] as i64;
        let az5 = az.rd_write_eq_lookup_if_write_lookup_to_rd as i64;
        if az5 != 0 {
            sum_c_az_i64 += c5.saturating_mul(az5);
        } else {
            fmadd_bz(
                &mut sum_bz_s160,
                c5,
                S160::from(bz.rd_write_minus_lookup_output.to_i128()),
            );
        }

        // 6: rd_write_eq_pc_plus_const_if_write_pc_to_rd (u8), Bz = rd_write - (pc + const)
        let c6 = coeffs_i32[6] as i64;
        let az6 = az.rd_write_eq_pc_plus_const_if_write_pc_to_rd as i64;
        if az6 != 0 {
            sum_c_az_i64 += c6.saturating_mul(az6);
        } else {
            fmadd_bz(
                &mut sum_bz_s160,
                c6,
                S160::from(bz.rd_write_minus_pc_plus_const),
            );
        }

        // 7: ShouldBranch (bool), Bz = next_unexp_pc - (pc + imm)
        let c7 = coeffs_i32[7] as i64;
        if az.next_unexp_pc_eq_pc_plus_imm_if_should_branch {
            sum_c_az_i64 += c7;
        } else {
            fmadd_bz(
                &mut sum_bz_s160,
                c7,
                S160::from(bz.next_unexp_pc_minus_pc_plus_imm),
            );
        }

        // 8: !(Jump||ShouldBranch) (bool), Bz = next_unexp_pc - expected
        let c8 = coeffs_i32[8] as i64;
        if az.next_unexp_pc_update_otherwise {
            sum_c_az_i64 += c8;
        } else {
            fmadd_bz(
                &mut sum_bz_s160,
                c8,
                S160::from(bz.next_unexp_pc_minus_expected),
            );
        }

        let sum_bz_s192: S192 = sum_bz_s160.to_signed_bigint_nplus1::<3>();
        let sum_az_s64 = S64::from_i64(sum_c_az_i64);
        sum_az_s64.mul_trunc::<3, 3>(&sum_bz_s192)
    }

    #[cfg(test)]
    #[inline]
    pub fn debug_assert_zero_when_guarded(&self) {
        let az = self.eval_az();
        let bz = self.eval_bz();
        debug_assert!(
            (!az.ram_addr_eq_rs1_plus_imm_if_load_store) || bz.ram_addr_minus_rs1_plus_imm == 0i128
        );
        debug_assert!((!az.right_lookup_add) || bz.right_lookup_minus_add_result.is_zero());
        debug_assert!((!az.right_lookup_sub) || bz.right_lookup_minus_sub_result.is_zero());
        debug_assert!(
            (!az.right_lookup_eq_product_if_mul) || bz.right_lookup_minus_product.is_zero()
        );
        debug_assert!(
            (!az.right_lookup_eq_right_input_otherwise)
                || bz.right_lookup_minus_right_input.is_zero()
        );
        debug_assert!(
            (!az.rd_write_eq_lookup_if_write_lookup_to_rd)
                || bz.rd_write_minus_lookup_output.to_i128() == 0
        );
        debug_assert!(
            (!az.rd_write_eq_pc_plus_const_if_write_pc_to_rd)
                || bz.rd_write_minus_pc_plus_const == 0
        );
        debug_assert!(
            (!az.next_unexp_pc_eq_pc_plus_imm_if_should_branch)
                || bz.next_unexp_pc_minus_pc_plus_imm == 0
        );
        debug_assert!((!az.next_unexp_pc_update_otherwise) || bz.next_unexp_pc_minus_expected == 0);
    }
}
