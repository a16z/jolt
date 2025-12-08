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
//!   - Specialized `extended_azbz_product` helpers implement the folded
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

use ark_ff::biginteger::{S128, S160, S192, S256, S64};
use ark_std::Zero;
use rayon::prelude::*;
use strum::IntoEnumIterator;
use tracer::instruction::Cycle;

use crate::field::{BarrettReduce, FMAdd, JoltField};
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::lagrange_poly::LagrangeHelper;
use crate::poly::opening_proof::{OpeningPoint, BIG_ENDIAN};
use crate::subprotocols::univariate_skip::uniskip_targets;
use crate::utils::{
    accumulation::{Acc5U, Acc6S, Acc6U, Acc7S, Acc7U, S128Sum, S192Sum},
    math::s64_from_diff_u64s,
};
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::instruction::{CircuitFlags, NUM_CIRCUIT_FLAGS};
use crate::zkvm::r1cs::inputs::ProductCycleInputs;

use super::constraints::{
    NUM_PRODUCT_VIRTUAL, OUTER_UNIVARIATE_SKIP_DEGREE, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
    PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
};
use super::inputs::{JoltR1CSInputs, R1CSCycleInputs, NUM_R1CS_INPUTS};

pub(crate) const UNISKIP_TARGETS: [i64; OUTER_UNIVARIATE_SKIP_DEGREE] =
    uniskip_targets::<OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE, OUTER_UNIVARIATE_SKIP_DEGREE>();

pub(crate) const BASE_LEFT: i64 = -((OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE as i64 - 1) / 2);

pub(crate) const TARGET_SHIFTS: [i64; OUTER_UNIVARIATE_SKIP_DEGREE] = {
    let mut out = [0i64; OUTER_UNIVARIATE_SKIP_DEGREE];
    let mut j: usize = 0;
    while j < OUTER_UNIVARIATE_SKIP_DEGREE {
        out[j] = UNISKIP_TARGETS[j] - BASE_LEFT;
        j += 1;
    }
    out
};

pub(crate) const COEFFS_PER_J: [[i32; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE];
    OUTER_UNIVARIATE_SKIP_DEGREE] = {
    let mut out = [[0i32; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]; OUTER_UNIVARIATE_SKIP_DEGREE];
    let mut j: usize = 0;
    while j < OUTER_UNIVARIATE_SKIP_DEGREE {
        out[j] =
            LagrangeHelper::shift_coeffs_i32::<OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE>(TARGET_SHIFTS[j]);
        j += 1;
    }
    out
};

pub(crate) const PRODUCT_VIRTUAL_UNISKIP_TARGETS: [i64; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] =
    uniskip_targets::<
        PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE,
    >();

pub(crate) const PRODUCT_VIRTUAL_BASE_LEFT: i64 =
    -((PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE as i64 - 1) / 2);

pub(crate) const PRODUCT_VIRTUAL_TARGET_SHIFTS: [i64; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] = {
    let mut out = [0i64; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE];
    let mut j: usize = 0;
    while j < PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE {
        out[j] = PRODUCT_VIRTUAL_UNISKIP_TARGETS[j] - PRODUCT_VIRTUAL_BASE_LEFT;
        j += 1;
    }
    out
};

pub(crate) const PRODUCT_VIRTUAL_COEFFS_PER_J: [[i32;
    PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE];
    PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] = {
    let mut out = [[0i32; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE];
        PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE];
    let mut j: usize = 0;
    while j < PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE {
        out[j] = LagrangeHelper::shift_coeffs_i32::<PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE>(
            PRODUCT_VIRTUAL_TARGET_SHIFTS[j],
        );
        j += 1;
    }
    out
};

/// Boolean guards for the first group (univariate-skip base window)
#[derive(Clone, Copy, Debug)]
pub struct AzFirstGroup {
    pub not_load_store: bool,      // !(Load || Store)
    pub load_a: bool,              // Load
    pub load_b: bool,              // Load
    pub store: bool,               // Store
    pub add_sub_mul: bool,         // Add || Sub || Mul
    pub not_add_sub_mul: bool,     // !(Add || Sub || Mul)
    pub assert_flag: bool,         // Assert
    pub should_jump: bool,         // ShouldJump
    pub virtual_instruction: bool, // VirtualInstruction
    pub must_start_sequence: bool, // NextIsVirtual && !NextIsFirstInSequence
}

impl AzFirstGroup {
    /// Fused multiply-add into an unreduced accumulator using Lagrange weights `w`
    /// over the univariate-skip base window. This mirrors `az_at_r_first_group`
    /// but keeps the result in an `Acc5U` accumulator without reducing.
    #[inline(always)]
    pub fn fmadd_at_r<F: JoltField>(
        &self,
        w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
        acc: &mut Acc5U<F>,
    ) {
        acc.fmadd(&w[0], &self.not_load_store);
        acc.fmadd(&w[1], &self.load_a);
        acc.fmadd(&w[2], &self.load_b);
        acc.fmadd(&w[3], &self.store);
        acc.fmadd(&w[4], &self.add_sub_mul);
        acc.fmadd(&w[5], &self.not_add_sub_mul);
        acc.fmadd(&w[6], &self.assert_flag);
        acc.fmadd(&w[7], &self.should_jump);
        acc.fmadd(&w[8], &self.virtual_instruction);
        acc.fmadd(&w[9], &self.must_start_sequence);
    }
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
    pub one_minus_do_not_update_unexpanded_pc: bool, // 1 - DoNotUpdateUnexpandedPC
}

impl BzFirstGroup {
    /// Fused multiply-add into an unreduced accumulator using Lagrange weights `w`
    /// over the univariate-skip base window. This mirrors `bz_at_r_first_group`
    /// but keeps the result in an `Acc6S` accumulator without reducing.
    #[inline(always)]
    pub fn fmadd_at_r<F: JoltField>(
        &self,
        w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
        acc: &mut Acc6S<F>,
    ) {
        acc.fmadd(&w[0], &self.ram_addr);
        acc.fmadd(&w[1], &self.ram_read_minus_ram_write);
        acc.fmadd(&w[2], &self.ram_read_minus_rd_write);
        acc.fmadd(&w[3], &self.rs2_minus_ram_write);
        acc.fmadd(&w[4], &self.left_lookup);
        acc.fmadd(&w[5], &self.left_lookup_minus_left_input);
        acc.fmadd(&w[6], &self.lookup_output_minus_one);
        acc.fmadd(&w[7], &self.next_unexp_pc_minus_lookup_output);
        acc.fmadd(&w[8], &self.next_pc_minus_pc_plus_one);
        acc.fmadd(&w[9], &self.one_minus_do_not_update_unexpanded_pc);
    }
}

/// Guards for the second group (all booleans except two u8 flags)
#[derive(Clone, Copy, Debug)]
pub struct AzSecondGroup {
    pub load_or_store: bool,          // Load || Store
    pub add: bool,                    // Add
    pub sub: bool,                    // Sub
    pub mul: bool,                    // Mul
    pub not_add_sub_mul_advice: bool, // !(Add || Sub || Mul || Advice)
    pub write_lookup_to_rd: bool,     // write_lookup_output_to_rd_addr (Rd != 0)
    pub write_pc_to_rd: bool,         // write_pc_to_rd_addr (Rd != 0)
    pub should_branch: bool,          // ShouldBranch
    pub not_jump_or_branch: bool,     // !(Jump || ShouldBranch)
}

impl AzSecondGroup {
    /// Fused multiply-add into an unreduced accumulator using Lagrange weights `w`
    /// over the univariate-skip base window. This mirrors `az_at_r_second_group`
    /// but keeps the result in an `Acc5U` accumulator without reducing.
    #[inline(always)]
    pub fn fmadd_at_r<F: JoltField>(
        &self,
        w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
        acc: &mut Acc5U<F>,
    ) {
        acc.fmadd(&w[0], &self.load_or_store);
        acc.fmadd(&w[1], &self.add);
        acc.fmadd(&w[2], &self.sub);
        acc.fmadd(&w[3], &self.mul);
        acc.fmadd(&w[4], &self.not_add_sub_mul_advice);
        acc.fmadd(&w[5], &self.write_lookup_to_rd);
        acc.fmadd(&w[6], &self.write_pc_to_rd);
        acc.fmadd(&w[7], &self.should_branch);
        acc.fmadd(&w[8], &self.not_jump_or_branch);
    }
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
    pub rd_write_minus_pc_plus_const: S64, // RdWrite - (UnexpandedPC + const)
    pub next_unexp_pc_minus_pc_plus_imm: i128, // NextUnexpandedPC - (UnexpandedPC + Imm)
    pub next_unexp_pc_minus_expected: S64, // NextUnexpandedPC - (UnexpandedPC + const)
}

impl BzSecondGroup {
    /// Fused multiply-add into an unreduced accumulator using Lagrange weights `w`
    /// over the univariate-skip base window. This mirrors `bz_at_r_second_group`
    /// but keeps the result in an `Acc7S` accumulator without reducing.
    #[inline(always)]
    pub fn fmadd_at_r<F: JoltField>(
        &self,
        w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
        acc: &mut Acc7S<F>,
    ) {
        acc.fmadd(&w[0], &self.ram_addr_minus_rs1_plus_imm);
        acc.fmadd(&w[1], &self.right_lookup_minus_add_result);
        acc.fmadd(&w[2], &self.right_lookup_minus_sub_result);
        acc.fmadd(&w[3], &self.right_lookup_minus_product);
        acc.fmadd(&w[4], &self.right_lookup_minus_right_input);
        acc.fmadd(&w[5], &self.rd_write_minus_lookup_output);
        acc.fmadd(&w[6], &self.rd_write_minus_pc_plus_const);
        acc.fmadd(&w[7], &self.next_unexp_pc_minus_pc_plus_imm);
        acc.fmadd(&w[8], &self.next_unexp_pc_minus_expected);
    }
}

/// Unified evaluator wrapper with typed accessors for both groups
#[derive(Clone, Copy, Debug)]
pub struct R1CSEval<'a, F: JoltField> {
    row: &'a R1CSCycleInputs,
    _m: core::marker::PhantomData<F>,
}

impl<'a, F: JoltField> R1CSEval<'a, F> {
    #[inline]
    pub fn from_cycle_inputs(row: &'a R1CSCycleInputs) -> Self {
        Self {
            row,
            _m: core::marker::PhantomData,
        }
    }

    // ---------- First group ----------

    #[inline]
    pub fn eval_az_first_group(&self) -> AzFirstGroup {
        let flags = &self.row.flags;
        let ld = flags[CircuitFlags::Load];
        let st = flags[CircuitFlags::Store];
        let add = flags[CircuitFlags::AddOperands];
        let sub = flags[CircuitFlags::SubtractOperands];
        let mul = flags[CircuitFlags::MultiplyOperands];
        let assert_flag = flags[CircuitFlags::Assert];
        let inline_seq = flags[CircuitFlags::VirtualInstruction];

        AzFirstGroup {
            not_load_store: !(ld || st),
            load_a: ld,
            load_b: ld,
            store: st,
            add_sub_mul: add || sub || mul,
            not_add_sub_mul: !(add || sub || mul),
            assert_flag,
            should_jump: self.row.should_jump,
            virtual_instruction: inline_seq,
            must_start_sequence: self.row.next_is_virtual && !self.row.next_is_first_in_sequence,
        }
    }

    #[inline]
    pub fn eval_bz_first_group(&self) -> BzFirstGroup {
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
            one_minus_do_not_update_unexpanded_pc: !self.row.flags
                [CircuitFlags::DoNotUpdateUnexpandedPC],
        }
    }

    #[inline]
    pub fn az_at_r_first_group(&self, w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]) -> F {
        let az = self.eval_az_first_group();
        let mut acc: Acc5U<F> = Acc5U::zero();
        acc.fmadd(&w[0], &az.not_load_store);
        acc.fmadd(&w[1], &az.load_a);
        acc.fmadd(&w[2], &az.load_b);
        acc.fmadd(&w[3], &az.store);
        acc.fmadd(&w[4], &az.add_sub_mul);
        acc.fmadd(&w[5], &az.not_add_sub_mul);
        acc.fmadd(&w[6], &az.assert_flag);
        acc.fmadd(&w[7], &az.should_jump);
        acc.fmadd(&w[8], &az.virtual_instruction);
        acc.fmadd(&w[9], &az.must_start_sequence);
        acc.barrett_reduce()
    }

    #[inline]
    pub fn bz_at_r_first_group(&self, w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]) -> F {
        let bz = self.eval_bz_first_group();
        let mut acc: Acc6S<F> = Acc6S::zero();
        acc.fmadd(&w[0], &bz.ram_addr);
        acc.fmadd(&w[1], &bz.ram_read_minus_ram_write);
        acc.fmadd(&w[2], &bz.ram_read_minus_rd_write);
        acc.fmadd(&w[3], &bz.rs2_minus_ram_write);
        acc.fmadd(&w[4], &bz.left_lookup);
        acc.fmadd(&w[5], &bz.left_lookup_minus_left_input);
        acc.fmadd(&w[6], &bz.lookup_output_minus_one);
        acc.fmadd(&w[7], &bz.next_unexp_pc_minus_lookup_output);
        acc.fmadd(&w[8], &bz.next_pc_minus_pc_plus_one);
        acc.fmadd(&w[9], &bz.one_minus_do_not_update_unexpanded_pc);
        acc.barrett_reduce()
    }

    /// Fused accumulate of first-group Az and Bz into unreduced accumulators using
    /// Lagrange weights `w`. This keeps everything in unreduced form; callers are
    /// responsible for reducing at the end.
    #[inline]
    pub fn fmadd_first_group_at_r(
        &self,
        w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
        acc_az: &mut Acc5U<F>,
        acc_bz: &mut Acc6S<F>,
    ) {
        let az = self.eval_az_first_group();
        az.fmadd_at_r(w, acc_az);
        let bz = self.eval_bz_first_group();
        bz.fmadd_at_r(w, acc_bz);
    }

    /// Product Az·Bz at the j-th extended uniskip target for the first group (uses precomputed weights).
    pub fn extended_azbz_product_first_group(&self, j: usize) -> S192 {
        let coeffs_i32: &[i32; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE] = &COEFFS_PER_J[j];
        let az = self.eval_az_first_group();
        let bz = self.eval_bz_first_group();

        let mut az_eval_i32: i32 = 0;
        let mut bz_eval_s128: S128Sum = S128Sum::zero();

        let c0_i32 = coeffs_i32[0];
        if az.not_load_store {
            az_eval_i32 += c0_i32;
        } else {
            bz_eval_s128.fmadd(&c0_i32, &bz.ram_addr);
        }

        let c1_i32 = coeffs_i32[1];
        if az.load_a {
            az_eval_i32 += c1_i32;
        } else {
            bz_eval_s128.fmadd(&c1_i32, &bz.ram_read_minus_ram_write);
        }

        let c2_i32 = coeffs_i32[2];
        if az.load_b {
            az_eval_i32 += c2_i32;
        } else {
            bz_eval_s128.fmadd(&c2_i32, &bz.ram_read_minus_rd_write);
        }

        let c3_i32 = coeffs_i32[3];
        if az.store {
            az_eval_i32 += c3_i32;
        } else {
            bz_eval_s128.fmadd(&c3_i32, &bz.rs2_minus_ram_write);
        }

        let c4_i32 = coeffs_i32[4];
        if az.add_sub_mul {
            az_eval_i32 += c4_i32;
        } else {
            bz_eval_s128.fmadd(&c4_i32, &bz.left_lookup);
        }

        let c5_i32 = coeffs_i32[5];
        if az.not_add_sub_mul {
            az_eval_i32 += c5_i32;
        } else {
            bz_eval_s128.fmadd(&c5_i32, &bz.left_lookup_minus_left_input);
        }

        let c6_i32 = coeffs_i32[6];
        if az.assert_flag {
            az_eval_i32 += c6_i32;
        } else {
            bz_eval_s128.fmadd(&c6_i32, &bz.lookup_output_minus_one);
        }

        let c7_i32 = coeffs_i32[7];
        if az.should_jump {
            az_eval_i32 += c7_i32;
        } else {
            bz_eval_s128.fmadd(&c7_i32, &bz.next_unexp_pc_minus_lookup_output);
        }

        let c8_i32 = coeffs_i32[8];
        if az.virtual_instruction {
            az_eval_i32 += c8_i32;
        } else {
            bz_eval_s128.fmadd(&c8_i32, &bz.next_pc_minus_pc_plus_one);
        }

        let c9_i32 = coeffs_i32[9];
        if az.must_start_sequence {
            az_eval_i32 += c9_i32;
        } else {
            bz_eval_s128.fmadd(&c9_i32, &bz.one_minus_do_not_update_unexpanded_pc);
        }

        let az_eval_s64 = S64::from_i64(az_eval_i32 as i64);
        az_eval_s64.mul_trunc::<2, 3>(&bz_eval_s128.sum)
    }

    #[cfg(test)]
    pub fn assert_constraints_first_group(&self) {
        let az = self.eval_az_first_group();
        let bz = self.eval_bz_first_group();
        debug_assert!((!az.not_load_store) || bz.ram_addr == 0);
        debug_assert!((!az.load_a) || bz.ram_read_minus_ram_write.to_i128() == 0);
        debug_assert!((!az.load_b) || bz.ram_read_minus_rd_write.to_i128() == 0);
        debug_assert!((!az.store) || bz.rs2_minus_ram_write.to_i128() == 0);
        debug_assert!((!az.add_sub_mul) || bz.left_lookup == 0);
        debug_assert!((!az.not_add_sub_mul) || bz.left_lookup_minus_left_input.to_i128() == 0);
        debug_assert!((!az.assert_flag) || bz.lookup_output_minus_one.to_i128() == 0);
        debug_assert!((!az.should_jump) || bz.next_unexp_pc_minus_lookup_output.to_i128() == 0);
        debug_assert!((!az.virtual_instruction) || bz.next_pc_minus_pc_plus_one.to_i128() == 0);
        debug_assert!((!az.must_start_sequence) || bz.one_minus_do_not_update_unexpanded_pc);
    }
    // ---------- Second group ----------

    #[inline]
    pub fn eval_az_second_group(&self) -> AzSecondGroup {
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
            load_or_store: (flags[CircuitFlags::Load] || flags[CircuitFlags::Store]),
            add: flags[CircuitFlags::AddOperands],
            sub: flags[CircuitFlags::SubtractOperands],
            mul: flags[CircuitFlags::MultiplyOperands],
            not_add_sub_mul_advice,
            write_lookup_to_rd: self.row.write_lookup_output_to_rd_addr,
            write_pc_to_rd: self.row.write_pc_to_rd_addr,
            should_branch: self.row.should_branch,
            not_jump_or_branch: next_update_otherwise,
        }
    }

    #[inline]
    pub fn eval_bz_second_group(&self) -> BzSecondGroup {
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
        let expected_pc_plus_const = self.row.unexpanded_pc.wrapping_add(const_term as u64);
        let rd_write_minus_pc_plus_const =
            s64_from_diff_u64s(self.row.rd_write_value, expected_pc_plus_const);

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
        let expected_next = self.row.unexpanded_pc.wrapping_add(const_term_next as u64);
        let next_unexp_pc_minus_expected =
            s64_from_diff_u64s(self.row.next_unexpanded_pc, expected_next);

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
    pub fn az_at_r_second_group(&self, _w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]) -> F {
        let w = _w;
        let az = self.eval_az_second_group();
        let mut acc: Acc5U<F> = Acc5U::zero();
        acc.fmadd(&w[0], &az.load_or_store);
        acc.fmadd(&w[1], &az.add);
        acc.fmadd(&w[2], &az.sub);
        acc.fmadd(&w[3], &az.mul);
        acc.fmadd(&w[4], &az.not_add_sub_mul_advice);
        acc.fmadd(&w[5], &az.write_lookup_to_rd);
        acc.fmadd(&w[6], &az.write_pc_to_rd);
        acc.fmadd(&w[7], &az.should_branch);
        acc.fmadd(&w[8], &az.not_jump_or_branch);
        acc.barrett_reduce()
    }

    #[inline]
    pub fn bz_at_r_second_group(&self, _w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]) -> F {
        let w = _w;
        let bz = self.eval_bz_second_group();
        let mut acc: Acc7S<F> = Acc7S::zero();
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

    /// Fused accumulate of second-group Az and Bz into unreduced accumulators
    /// using Lagrange weights `w`. This keeps everything in unreduced form; callers
    /// are responsible for reducing at the end.
    #[inline]
    pub fn fmadd_second_group_at_r(
        &self,
        w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
        acc_az: &mut Acc5U<F>,
        acc_bz: &mut Acc7S<F>,
    ) {
        let az = self.eval_az_second_group();
        az.fmadd_at_r(w, acc_az);
        let bz = self.eval_bz_second_group();
        bz.fmadd_at_r(w, acc_bz);
    }

    /// Product Az·Bz at the j-th extended uniskip target for the second group (uses precomputed weights).
    pub fn extended_azbz_product_second_group(&self, j: usize) -> S192 {
        #[cfg(test)]
        self.assert_constraints_second_group();

        let coeffs_i32: &[i32; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE] = &COEFFS_PER_J[j];
        let az = self.eval_az_second_group();
        let bz = self.eval_bz_second_group();

        let mut az_eval_i32: i32 = 0;
        let mut bz_eval_s192 = S192Sum::zero();

        let c0 = coeffs_i32[0];
        if az.load_or_store {
            az_eval_i32 += c0;
        } else {
            bz_eval_s192.fmadd(&c0, &bz.ram_addr_minus_rs1_plus_imm);
        }

        let c1 = coeffs_i32[1];
        if az.add {
            az_eval_i32 += c1;
        } else {
            bz_eval_s192.fmadd(&c1, &bz.right_lookup_minus_add_result);
        }

        let c2 = coeffs_i32[2];
        if az.sub {
            az_eval_i32 += c2;
        } else {
            bz_eval_s192.fmadd(&c2, &bz.right_lookup_minus_sub_result);
        }

        let c3 = coeffs_i32[3];
        if az.mul {
            az_eval_i32 += c3;
        } else {
            bz_eval_s192.fmadd(&c3, &bz.right_lookup_minus_product);
        }

        let c4 = coeffs_i32[4];
        if az.not_add_sub_mul_advice {
            az_eval_i32 += c4;
        } else {
            bz_eval_s192.fmadd(&c4, &bz.right_lookup_minus_right_input);
        }

        let c5 = coeffs_i32[5];
        if az.write_lookup_to_rd {
            az_eval_i32 += c5;
        } else {
            bz_eval_s192.fmadd(&c5, &bz.rd_write_minus_lookup_output);
        }

        let c6 = coeffs_i32[6];
        if az.write_pc_to_rd {
            az_eval_i32 += c6;
        } else {
            bz_eval_s192.fmadd(&c6, &bz.rd_write_minus_pc_plus_const);
        }

        let c7 = coeffs_i32[7];
        if az.should_branch {
            az_eval_i32 += c7;
        } else {
            bz_eval_s192.fmadd(&c7, &bz.next_unexp_pc_minus_pc_plus_imm);
        }

        let c8 = coeffs_i32[8];
        if az.not_jump_or_branch {
            az_eval_i32 += c8;
        } else {
            bz_eval_s192.fmadd(&c8, &bz.next_unexp_pc_minus_expected);
        }

        let az_eval_s64 = S64::from_i64(az_eval_i32 as i64);
        az_eval_s64.mul_trunc::<3, 3>(&bz_eval_s192.sum)
    }

    #[cfg(test)]
    pub fn assert_constraints_second_group(&self) {
        let az = self.eval_az_second_group();
        let bz = self.eval_bz_second_group();
        debug_assert!((!az.load_or_store) || bz.ram_addr_minus_rs1_plus_imm == 0i128);
        debug_assert!((!az.add) || bz.right_lookup_minus_add_result.is_zero());
        debug_assert!((!az.sub) || bz.right_lookup_minus_sub_result.is_zero());
        debug_assert!((!az.mul) || bz.right_lookup_minus_product.is_zero());
        debug_assert!((!az.not_add_sub_mul_advice) || bz.right_lookup_minus_right_input.is_zero());
        debug_assert!((!az.write_lookup_to_rd) || bz.rd_write_minus_lookup_output.is_zero());
        debug_assert!((!az.write_pc_to_rd) || bz.rd_write_minus_pc_plus_const.is_zero());
        debug_assert!((!az.should_branch) || bz.next_unexp_pc_minus_pc_plus_imm == 0);
        debug_assert!((!az.not_jump_or_branch) || bz.next_unexp_pc_minus_expected.is_zero());
    }

    /// Compute `z(r_cycle) = Σ_t eq(r_cycle, t) * P_i(t)` for all inputs i, without
    /// materializing P_i. Returns `[P_0(r_cycle), P_1(r_cycle), ...]` in input order.
    #[tracing::instrument(skip_all, name = "R1CSEval::compute_claimed_inputs")]
    pub fn compute_claimed_inputs(
        bytecode_preprocessing: &BytecodePreprocessing,
        trace: &[Cycle],
        r_cycle: &OpeningPoint<BIG_ENDIAN, F>,
    ) -> [F; NUM_R1CS_INPUTS] {
        let m = r_cycle.len() / 2;
        let (r2, r1) = r_cycle.split_at_r(m);
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
                let mut acc_left_input: Acc6U<F> = Acc6U::zero();
                let mut acc_right_input: Acc6S<F> = Acc6S::zero();
                let mut acc_product: Acc7S<F> = Acc7S::zero();
                let mut acc_wl_left: Acc5U<F> = Acc5U::zero();
                let mut acc_wp_left: Acc5U<F> = Acc5U::zero();
                let mut acc_sb_right: Acc5U<F> = Acc5U::zero();
                let mut acc_pc: Acc6U<F> = Acc6U::zero();
                let mut acc_unexpanded_pc: Acc6U<F> = Acc6U::zero();
                let mut acc_imm: Acc6S<F> = Acc6S::zero();
                let mut acc_ram_address: Acc6U<F> = Acc6U::zero();
                let mut acc_rs1_value: Acc6U<F> = Acc6U::zero();
                let mut acc_rs2_value: Acc6U<F> = Acc6U::zero();
                let mut acc_rd_write_value: Acc6U<F> = Acc6U::zero();
                let mut acc_ram_read_value: Acc6U<F> = Acc6U::zero();
                let mut acc_ram_write_value: Acc6U<F> = Acc6U::zero();
                let mut acc_left_lookup_operand: Acc6U<F> = Acc6U::zero();
                let mut acc_right_lookup_operand: Acc7U<F> = Acc7U::zero();
                let mut acc_next_unexpanded_pc: Acc6U<F> = Acc6U::zero();
                let mut acc_next_pc: Acc6U<F> = Acc6U::zero();
                let mut acc_lookup_output: Acc6U<F> = Acc6U::zero();
                let mut acc_sj_flag: Acc5U<F> = Acc5U::zero();
                let mut acc_next_is_virtual: Acc5U<F> = Acc5U::zero();
                let mut acc_next_is_first_in_sequence: Acc5U<F> = Acc5U::zero();
                let mut acc_flags: Vec<Acc5U<F>> =
                    (0..NUM_CIRCUIT_FLAGS).map(|_| Acc5U::zero()).collect();

                let eq_two_len = eq_two.len();
                for x2 in 0..eq_two_len {
                    let e_in = eq_two[x2];
                    let idx = x1 * eq_two_len + x2;
                    let row = R1CSCycleInputs::from_trace::<F>(bytecode_preprocessing, trace, idx);

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
        let mut left_acc: Acc6U<F> = Acc6U::zero();
        left_acc.fmadd(&weights_at_r0[0], &row.instruction_left_input);
        left_acc.fmadd(&weights_at_r0[1], &row.is_rd_not_zero);
        left_acc.fmadd(&weights_at_r0[2], &row.is_rd_not_zero);
        left_acc.fmadd(&weights_at_r0[3], &row.should_branch_lookup_output);
        left_acc.fmadd(&weights_at_r0[4], &row.jump_flag);

        // Right: i128/bool
        let mut right_acc: Acc6S<F> = Acc6S::zero();
        right_acc.fmadd(&weights_at_r0[0], &row.instruction_right_input);
        right_acc.fmadd(&weights_at_r0[1], &row.write_lookup_output_to_rd_flag);
        right_acc.fmadd(&weights_at_r0[2], &row.jump_flag);
        right_acc.fmadd(&weights_at_r0[3], &row.should_branch_flag);
        right_acc.fmadd(&weights_at_r0[4], &row.not_next_noop);

        (left_acc.barrett_reduce(), right_acc.barrett_reduce())
    }

    /// Compute the fused left·right product at the j-th extended uniskip target for product virtualization.
    /// Uses precomputed integer Lagrange coefficients over the size-5 base window and returns an S256 product.
    #[inline]
    pub fn extended_fused_product_at_j<F: JoltField>(row: &ProductCycleInputs, j: usize) -> S256 {
        let c: &[i32; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE] =
            &PRODUCT_VIRTUAL_COEFFS_PER_J[j];

        // Weighted components lifted to i128
        let mut left_w: [i128; NUM_PRODUCT_VIRTUAL] = [0; NUM_PRODUCT_VIRTUAL];
        let mut right_w: [i128; NUM_PRODUCT_VIRTUAL] = [0; NUM_PRODUCT_VIRTUAL];

        // 0: Instruction (LeftInstructionInput × RightInstructionInput)
        left_w[0] = (c[0] as i128) * (row.instruction_left_input as i128);
        right_w[0] = (c[0] as i128) * row.instruction_right_input;

        // 1: WriteLookupOutputToRD (IsRdNotZero × WriteLookupOutputToRD_flag)
        left_w[1] = if row.is_rd_not_zero { c[1] as i128 } else { 0 };
        right_w[1] = if row.write_lookup_output_to_rd_flag {
            c[1] as i128
        } else {
            0
        };

        // 2: WritePCtoRD (IsRdNotZero × Jump_flag)
        left_w[2] = if row.is_rd_not_zero { c[2] as i128 } else { 0 };
        right_w[2] = if row.jump_flag { c[2] as i128 } else { 0 };

        // 3: ShouldBranch (LookupOutput × Branch_flag)
        left_w[3] = (c[3] as i128) * (row.should_branch_lookup_output as i128);
        right_w[3] = if row.should_branch_flag {
            c[3] as i128
        } else {
            0
        };

        // 4: ShouldJump (Jump_flag × (1 − NextIsNoop))
        left_w[4] = if row.jump_flag { c[4] as i128 } else { 0 };
        right_w[4] = if row.not_next_noop { c[4] as i128 } else { 0 };

        // Fuse in i128, then multiply as S128×S128 → S256
        let mut left_sum: i128 = 0;
        let mut right_sum: i128 = 0;
        let mut i = 0;
        while i < NUM_PRODUCT_VIRTUAL {
            left_sum += left_w[i];
            right_sum += right_w[i];
            i += 1;
        }
        let left_s128 = S128::from_i128(left_sum);
        let right_s128 = S128::from_i128(right_sum);
        left_s128.mul_trunc::<2, 4>(&right_s128)
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
    #[tracing::instrument(skip_all, name = "ProductVirtualEval::compute_claimed_factors")]
    pub fn compute_claimed_factors<F: JoltField>(
        trace: &[tracer::instruction::Cycle],
        r_cycle: &OpeningPoint<BIG_ENDIAN, F>,
    ) -> [F; 8] {
        let m = r_cycle.len() / 2;
        let (r2, r1) = r_cycle.split_at_r(m);
        let (eq_one, eq_two) = rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));

        let eq_two_len = eq_two.len();

        (0..eq_one.len())
            .into_par_iter()
            .map(|x1| {
                let eq1_val = eq_one[x1];

                // Accumulators for 8 outputs
                let mut acc_left_u64: Acc6U<F> = Acc6U::zero();
                let mut acc_right_i128: Acc6S<F> = Acc6S::zero();
                let mut acc_rd_zero_flag: Acc5U<F> = Acc5U::zero();
                let mut acc_wl_flag: Acc5U<F> = Acc5U::zero();
                let mut acc_jump_flag: Acc5U<F> = Acc5U::zero();
                let mut acc_lookup_output: Acc6U<F> = Acc6U::zero();
                let mut acc_branch_flag: Acc5U<F> = Acc5U::zero();
                let mut acc_next_is_noop: Acc5U<F> = Acc5U::zero();

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
            )
            .map(|unr| F::from_montgomery_reduce::<9>(unr))
    }
}
