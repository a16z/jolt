//! Runtime evaluators for uniform R1CS and product virtualization
//!
//! This module implements the runtime evaluation semantics for the compile-time
//! constraints declared in `r1cs::constraints`:
//!
//! - Grouped evaluators for the uniform R1CS constraints used by the
//!   univariate‑skip first round of Spartan outer sumcheck:
//!   - Typed guard/magnitude structs: `AzFirstGroup`, `BzFirstGroup`,
//!     `AzSecondGroup`, `BzSecondGroup`
//!   - Wrappers `R1CSFirstGroup` and `R1CSSecondGroup` expose `eval_az`,
//!     `eval_bz`, and window-weighted evaluators `az_at_r`, `bz_at_r`
//!   - Specialized `product_of_sums_shifted` helpers implement the folded
//!     accumulation pattern used by the first-round polynomial
//!   - Shapes (boolean vs. wider signed magnitudes) match the grouping
//!     described in `r1cs::constraints`
//! 
//! - Input claim computation (at the end of Spartan outer sumcheck):
//!   - `R1CSEval::compute_claimed_inputs` accumulates all `JoltR1CSInputs`
//!     values at a random point without materializing per-input polynomials,
//!     using split `EqPolynomial` and fixed-limb accumulators
//!
//! - Evaluation helpers for the product virtualization sumcheck:
//!   - `ProductVirtualEval::fused_left_right_at_r` computes the fused left and
//!     right factor values at the r0 window for a single cycle row
//!   - `ProductVirtualEval::compute_claimed_factors` computes z(r) for the 8
//!     de-duplicated factor polynomials consumed by Spartan outer
//!
//! What does not live here:
//! - The definition of any constraint or grouping metadata (see
//!   `r1cs::constraints` for uniform constraints, grouping constants, and the
//!   product-virtualization catalog)
//!
//! Implementation notes:
//! - Accumulator limb widths are chosen to match the value ranges of each type
//!   (bool/u8/u64/i128/S128/S160), minimizing conversions while keeping fast
//!   Barrett reductions.
//! - Test-only `assert_constraints` methods validate that Az guards imply zero
//!   Bz magnitudes for both groups.

use ark_ff::biginteger::{S128, S160, S192, S64};
use ark_std::Zero;
use rayon::prelude::*;
use strum::IntoEnumIterator;
use tracer::instruction::Cycle;

use crate::field::{BarrettReduce, FMAdd, JoltField};
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::accumulation::{Acc5U, Acc6S, Acc6U, Acc7S, Acc7U, S128Sum};
use crate::zkvm::instruction::{CircuitFlags, NUM_CIRCUIT_FLAGS};
use crate::zkvm::r1cs::inputs::ProductCycleInputs;
use crate::zkvm::JoltSharedPreprocessing;

use super::constraints::UNIVARIATE_SKIP_DOMAIN_SIZE;
use super::inputs::{JoltR1CSInputs, R1CSCycleInputs, NUM_R1CS_INPUTS};

// TODO: put this into arkworks
#[inline(always)]
fn s64_from_diff_u64s(a: u64, b: u64) -> S64 {
    if a < b {
        return S64::new([b - a], false);
    } else {
        return S64::new([a - b], true);
    }
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
    pub fn assert_constraints(&self) {
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
        self.assert_constraints();

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
    pub fn assert_constraints(&self) {
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

#[derive(Clone, Copy, Debug)]
pub struct R1CSEval;

impl R1CSEval {
    /// Compute `z(r_cycle) = Σ_t eq(r_cycle, t) * P_i(t)` for all inputs i, without
    /// materializing P_i. Returns `[P_0(r_cycle), P_1(r_cycle), ...]` in input order.
    #[tracing::instrument(skip_all)]
    pub fn compute_claimed_inputs<F: JoltField>(
        preprocessing: &JoltSharedPreprocessing,
        trace: &[Cycle],
        r_cycle: &[F::Challenge],
    ) -> [F; NUM_R1CS_INPUTS] {
        let m = r_cycle.len() / 2;
        let (r2, r1) = r_cycle.split_at(m);
        let (eq_one, eq_two) = rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));

        (0..eq_one.len())
            .into_par_iter()
            .map(|x1| {
                let eq1_val = eq_one[x1];

                // Accumulators for each input
                // If bool or u8 => 5 limbs unsigned
                // If u64 => 6 limbs unsigned
                // If i128 => 6 limbs signed
                // If S128 => 7 limbs signed
                let mut acc_left_input: Acc6U<F> = Acc6U::default();
                let mut acc_right_input: Acc6S<F> = Acc6S::default();
                let mut acc_product: Acc7S<F> = Acc7S::default();
                let mut acc_wl_left: Acc5U<F> = Acc5U::default();
                let mut acc_wp_left: Acc5U<F> = Acc5U::default();
                let mut acc_sb_right: Acc5U<F> = Acc5U::default();
                let mut acc_pc: Acc6U<F> = Acc6U::default();
                let mut acc_unexpanded_pc: Acc6U<F> = Acc6U::default();
                let mut acc_imm: Acc6S<F> = Acc6S::default();
                let mut acc_ram_address: Acc6U<F> = Acc6U::default();
                let mut acc_rs1_value: Acc6U<F> = Acc6U::default();
                let mut acc_rs2_value: Acc6U<F> = Acc6U::default();
                let mut acc_rd_write_value: Acc6U<F> = Acc6U::default();
                let mut acc_ram_read_value: Acc6U<F> = Acc6U::default();
                let mut acc_ram_write_value: Acc6U<F> = Acc6U::default();
                let mut acc_left_lookup_operand: Acc6U<F> = Acc6U::default();
                let mut acc_right_lookup_operand: Acc7U<F> = Acc7U::default();
                let mut acc_next_unexpanded_pc: Acc6U<F> = Acc6U::default();
                let mut acc_next_pc: Acc6U<F> = Acc6U::default();
                let mut acc_lookup_output: Acc6U<F> = Acc6U::default();
                let mut acc_sj_flag: Acc5U<F> = Acc5U::default();
                let mut acc_next_is_virtual: Acc5U<F> = Acc5U::default();
                let mut acc_next_is_first_in_sequence: Acc5U<F> = Acc5U::default();
                let mut acc_flags: Vec<Acc5U<F>> =
                    (0..NUM_CIRCUIT_FLAGS).map(|_| Acc5U::default()).collect();

                let eq_two_len = eq_two.len();
                for x2 in 0..eq_two_len {
                    let e_in = eq_two[x2];
                    let idx = x1 * eq_two_len + x2;
                    let row = R1CSCycleInputs::from_trace::<F>(preprocessing, trace, idx);

                    acc_left_input.fmadd(&e_in, &row.left_input);
                    acc_right_input.fmadd(&e_in, &row.right_input.to_i128());
                    acc_product.fmadd(&e_in, &row.product);

                    acc_wl_left.fmadd(&e_in, &(row.write_lookup_output_to_rd_addr as u64));
                    acc_wp_left.fmadd(&e_in, &(row.write_pc_to_rd_addr as u64));
                    acc_sb_right.fmadd(&e_in, &row.should_branch);

                    acc_pc.fmadd(&e_in, &row.pc);
                    acc_unexpanded_pc.fmadd(&e_in, &row.unexpanded_pc);
                    acc_imm.fmadd(&e_in, &row.imm.to_i128());
                    acc_ram_address.fmadd(&e_in, &row.ram_addr);
                    acc_rs1_value.fmadd(&e_in, &row.rs1_read_value);
                    acc_rs2_value.fmadd(&e_in, &row.rs2_read_value);
                    acc_rd_write_value.fmadd(&e_in, &row.rd_write_value);
                    acc_ram_read_value.fmadd(&e_in, &row.ram_read_value);
                    acc_ram_write_value.fmadd(&e_in, &row.ram_write_value);
                    acc_left_lookup_operand.fmadd(&e_in, &row.left_lookup);
                    acc_right_lookup_operand.fmadd(&e_in, &row.right_lookup);
                    acc_next_unexpanded_pc.fmadd(&e_in, &row.next_unexpanded_pc);
                    acc_next_pc.fmadd(&e_in, &row.next_pc);
                    acc_lookup_output.fmadd(&e_in, &row.lookup_output);
                    acc_sj_flag.fmadd(&e_in, &row.should_jump);
                    acc_next_is_virtual.fmadd(&e_in, &row.next_is_virtual);
                    acc_next_is_first_in_sequence.fmadd(&e_in, &row.next_is_first_in_sequence);
                    for flag in CircuitFlags::iter() {
                        acc_flags[flag as usize].fmadd(&e_in, &row.flags[flag as usize]);
                    }
                }

                let mut out_unr: [F::Unreduced<9>; NUM_R1CS_INPUTS] =
                    [F::Unreduced::<9>::zero(); NUM_R1CS_INPUTS];
                out_unr[JoltR1CSInputs::LeftInstructionInput.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_left_input.barrett_reduce());
                out_unr[JoltR1CSInputs::RightInstructionInput.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_right_input.barrett_reduce());
                out_unr[JoltR1CSInputs::Product.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_product.barrett_reduce());
                out_unr[JoltR1CSInputs::WriteLookupOutputToRD.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_wl_left.barrett_reduce());
                out_unr[JoltR1CSInputs::WritePCtoRD.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_wp_left.barrett_reduce());
                out_unr[JoltR1CSInputs::ShouldBranch.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_sb_right.barrett_reduce());
                out_unr[JoltR1CSInputs::PC.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_pc.barrett_reduce());
                out_unr[JoltR1CSInputs::UnexpandedPC.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_unexpanded_pc.barrett_reduce());
                out_unr[JoltR1CSInputs::Imm.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_imm.barrett_reduce());
                out_unr[JoltR1CSInputs::RamAddress.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_ram_address.barrett_reduce());
                out_unr[JoltR1CSInputs::Rs1Value.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_rs1_value.barrett_reduce());
                out_unr[JoltR1CSInputs::Rs2Value.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_rs2_value.barrett_reduce());
                out_unr[JoltR1CSInputs::RdWriteValue.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_rd_write_value.barrett_reduce());
                out_unr[JoltR1CSInputs::RamReadValue.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_ram_read_value.barrett_reduce());
                out_unr[JoltR1CSInputs::RamWriteValue.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_ram_write_value.barrett_reduce());
                out_unr[JoltR1CSInputs::LeftLookupOperand.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_left_lookup_operand.barrett_reduce());
                out_unr[JoltR1CSInputs::RightLookupOperand.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_right_lookup_operand.barrett_reduce());
                out_unr[JoltR1CSInputs::NextUnexpandedPC.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_next_unexpanded_pc.barrett_reduce());
                out_unr[JoltR1CSInputs::NextPC.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_next_pc.barrett_reduce());
                out_unr[JoltR1CSInputs::LookupOutput.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_lookup_output.barrett_reduce());
                out_unr[JoltR1CSInputs::ShouldJump.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_sj_flag.barrett_reduce());
                out_unr[JoltR1CSInputs::NextIsVirtual.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_next_is_virtual.barrett_reduce());
                out_unr[JoltR1CSInputs::NextIsFirstInSequence.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_next_is_first_in_sequence.barrett_reduce());
                for flag in CircuitFlags::iter() {
                    let idx = JoltR1CSInputs::OpFlags(flag).to_index();
                    let f_idx = flag as usize;
                    out_unr[idx] = eq1_val.mul_unreduced::<9>(acc_flags[f_idx].barrett_reduce());
                }
                out_unr
            })
            .reduce(
                || [F::Unreduced::<9>::zero(); NUM_R1CS_INPUTS],
                |mut acc, item| {
                    for i in 0..NUM_R1CS_INPUTS {
                        acc[i] += item[i];
                    }
                    acc
                },
            )
            .map(|unr| F::from_montgomery_reduce::<9>(unr))
    }
}

/// Struct for implementation of evaluation logic for product virtualization
#[derive(Clone, Copy, Debug)]
pub struct ProductVirtualEval;

impl ProductVirtualEval {
    /// Compute both fused left and right factors at r0 weights for a single cycle row.
    /// Expected order of weights: [Instruction, WriteLookupOutputToRD, WritePCtoRD, ShouldBranch, ShouldJump]
    #[inline]
    pub fn fused_left_right_at_r<F: JoltField>(
        row: &ProductCycleInputs,
        weights_at_r0: &[F],
    ) -> (F, F) {
        // Left: u64/u8/bool
        let mut left_acc: Acc6U<F> = Acc6U::default();
        left_acc.fmadd(&weights_at_r0[0], &row.instruction_left_input);
        left_acc.fmadd(&weights_at_r0[1], &row.is_rd_not_zero);
        left_acc.fmadd(&weights_at_r0[2], &row.is_rd_not_zero);
        left_acc.fmadd(&weights_at_r0[3], &row.should_branch_lookup_output);
        left_acc.fmadd(&weights_at_r0[4], &row.jump_flag);

        // Right: i128/bool
        let mut right_acc: Acc6S<F> = Acc6S::default();
        right_acc.fmadd(&weights_at_r0[0], &row.instruction_right_input);
        right_acc.fmadd(&weights_at_r0[1], &row.write_lookup_output_to_rd_flag);
        right_acc.fmadd(&weights_at_r0[2], &row.jump_flag);
        right_acc.fmadd(&weights_at_r0[3], &row.should_branch_flag);
        right_acc.fmadd(&weights_at_r0[4], &row.not_next_noop);

        (left_acc.barrett_reduce(), right_acc.barrett_reduce())
    }

    /// Compute z(r_cycle) for the 8 de-duplicated factor polynomials used by Product Virtualization.
    /// Order of outputs matches PRODUCT_UNIQUE_FACTOR_VIRTUALS:
    /// 0: LeftInstructionInput (u64)
    /// 1: RightInstructionInput (i128)
    /// 2: IsRdNotZero (bool)
    /// 3: OpFlags(WriteLookupOutputToRD) (bool)
    /// 4: OpFlags(Jump) (bool)
    /// 5: LookupOutput (u64)
    /// 6: InstructionFlags(Branch) (bool)
    /// 7: NextIsNoop (bool)
    #[inline]
    pub fn compute_claimed_factors<F: JoltField>(
        trace: &[tracer::instruction::Cycle],
        r_cycle: &[<F as JoltField>::Challenge],
    ) -> [F; 8] {
        let m = r_cycle.len() / 2;
        let (r2, r1) = r_cycle.split_at(m);
        let (eq_one, eq_two) = rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));

        let eq_two_len = eq_two.len();

        let totals_unr: [F::Unreduced<9>; 8] = (0..eq_one.len())
            .into_par_iter()
            .map(|x1| {
                let eq1_val = eq_one[x1];

                // Accumulators for 8 outputs
                let mut acc_left_u64: Acc6U<F> = Acc6U::default();
                let mut acc_right_i128: Acc6S<F> = Acc6S::default();
                let mut acc_rd_zero_flag: Acc5U<F> = Acc5U::default();
                let mut acc_wl_flag: Acc5U<F> = Acc5U::default();
                let mut acc_jump_flag: Acc5U<F> = Acc5U::default();
                let mut acc_lookup_output: Acc6U<F> = Acc6U::default();
                let mut acc_branch_flag: Acc5U<F> = Acc5U::default();
                let mut acc_next_is_noop: Acc5U<F> = Acc5U::default();

                for x2 in 0..eq_two_len {
                    let e_in = eq_two[x2];
                    let idx = x1 * eq_two_len + x2;
                    let row = ProductCycleInputs::from_trace::<F>(trace, idx);

                    // 0: LeftInstructionInput (u64)
                    acc_left_u64.fmadd(&e_in, &row.instruction_left_input);
                    // 1: RightInstructionInput (i128)
                    acc_right_i128.fmadd(&e_in, &row.instruction_right_input);
                    // 2: IsRdNotZero (bool)
                    acc_rd_zero_flag.fmadd(&e_in, &(row.is_rd_not_zero));
                    // 3: OpFlags(WriteLookupOutputToRD) (bool)
                    acc_wl_flag.fmadd(&e_in, &row.write_lookup_output_to_rd_flag);
                    // 4: OpFlags(Jump) (bool)
                    acc_jump_flag.fmadd(&e_in, &row.jump_flag);
                    // 5: LookupOutput (u64)
                    acc_lookup_output.fmadd(&e_in, &row.should_branch_lookup_output);
                    // 6: InstructionFlags(Branch) (bool)
                    acc_branch_flag.fmadd(&e_in, &row.should_branch_flag);
                    // 7: NextIsNoop (bool) = !not_next_noop
                    acc_next_is_noop.fmadd(&e_in, &(!row.not_next_noop));
                }

                let mut out_unr = [F::Unreduced::<9>::zero(); 8];
                out_unr[0] = eq1_val.mul_unreduced::<9>(acc_left_u64.barrett_reduce());
                out_unr[1] = eq1_val.mul_unreduced::<9>(acc_right_i128.barrett_reduce());
                out_unr[2] = eq1_val.mul_unreduced::<9>(acc_rd_zero_flag.barrett_reduce());
                out_unr[3] = eq1_val.mul_unreduced::<9>(acc_wl_flag.barrett_reduce());
                out_unr[4] = eq1_val.mul_unreduced::<9>(acc_jump_flag.barrett_reduce());
                out_unr[5] = eq1_val.mul_unreduced::<9>(acc_lookup_output.barrett_reduce());
                out_unr[6] = eq1_val.mul_unreduced::<9>(acc_branch_flag.barrett_reduce());
                out_unr[7] = eq1_val.mul_unreduced::<9>(acc_next_is_noop.barrett_reduce());
                out_unr
            })
            .reduce(
                || [F::Unreduced::<9>::zero(); 8],
                |mut acc, item| {
                    for i in 0..8 {
                        acc[i] += item[i];
                    }
                    acc
                },
            );

        core::array::from_fn(|i| F::from_montgomery_reduce::<9>(totals_unr[i]))
    }
}
