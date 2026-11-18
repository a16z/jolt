use jolt_field::{Accumulator as _, JoltField, WithAccumulator};

use super::InstructionInputRow;

type NativeAccumulator<F> = <F as WithAccumulator>::SignedProductAccumulator;

pub(super) struct NativePairSums<F: JoltField, const TERMS: usize> {
    right: [NativeAccumulator<F>; TERMS],
    right_high: [NativeAccumulator<F>; TERMS],
    left: [NativeAccumulator<F>; TERMS],
}

impl<F: JoltField, const TERMS: usize> NativePairSums<F, TERMS> {
    pub(super) fn new() -> Self {
        const { assert!(TERMS == 2 || TERMS == 3) };
        Self {
            right: [NativeAccumulator::<F>::default(); TERMS],
            right_high: [NativeAccumulator::<F>::default(); TERMS],
            left: [NativeAccumulator::<F>::default(); TERMS],
        }
    }

    pub(super) fn add(&mut self, even: &InstructionInputRow, odd: &InstructionInputRow, e_in: F) {
        let (is_rs1, is_rs1_m) = ext_flag(even.is_rs1.0, odd.is_rs1.0);
        let (is_pc, is_pc_m) = ext_flag(even.is_pc.0, odd.is_pc.0);
        let (is_rs2, is_rs2_m) = ext_flag(even.is_rs2.0, odd.is_rs2.0);
        let (is_imm, is_imm_m) = ext_flag(even.is_imm.0, odd.is_imm.0);
        let (is_rd, is_rd_m) = ext_flag(even.is_rd.0, odd.is_rd.0);
        let (rs1, rs1_m) = ext_u64(even.rs1_value.0, odd.rs1_value.0);
        let (upc, upc_m) = ext_u64(even.unexpanded_pc.0, odd.unexpanded_pc.0);
        let (rs2, rs2_m) = ext_u64(even.rs2_value.0, odd.rs2_value.0);
        let (rd, rd_m) = ext_u64(even.rd_write_value.0, odd.rd_write_value.0);
        let split = |imm: i128| (imm >> 64, i128::from(imm as u64));
        let (imm_even_high, imm_even_low) = split(even.imm.0);
        let (imm_odd_high, imm_odd_low) = split(odd.imm.0);
        let left = [
            i128::from(is_rs1) * rs1 + i128::from(is_pc) * upc,
            i128::from(is_rs1_m) * rs1_m + i128::from(is_pc_m) * upc_m,
        ];
        let right = [
            i128::from(is_rs2) * rs2 + i128::from(is_imm) * imm_even_low + i128::from(is_rd) * rd,
            i128::from(is_rs2_m) * rs2_m
                + i128::from(is_imm_m) * (imm_odd_low - imm_even_low)
                + i128::from(is_rd_m) * rd_m,
        ];
        let right_high = [
            i128::from(is_imm) * imm_even_high,
            i128::from(is_imm_m) * (imm_odd_high - imm_even_high),
        ];
        // Flag differences are in {-1, 0, 1}; word and split-immediate
        // differences have magnitude < 2^64. Each sum fits in i128.
        for (i, right) in right.into_iter().enumerate() {
            self.right[i].fmadd_i128(e_in, right);
            if right_high[i] != 0 {
                self.right_high[i].fmadd_i128(e_in, right_high[i]);
            }
            self.left[i].fmadd_i128(e_in, left[i]);
        }
        if TERMS == 3 {
            let left = i128::from(odd.is_rs1.0) * i128::from(odd.rs1_value.0)
                + i128::from(odd.is_pc.0) * i128::from(odd.unexpanded_pc.0);
            let right = i128::from(odd.is_rs2.0) * i128::from(odd.rs2_value.0)
                + i128::from(odd.is_imm.0) * imm_odd_low
                + i128::from(odd.is_rd.0) * i128::from(odd.rd_write_value.0);
            let right_high = i128::from(odd.is_imm.0) * imm_odd_high;
            self.right[2].fmadd_i128(e_in, right);
            if right_high != 0 {
                self.right_high[2].fmadd_i128(e_in, right_high);
            }
            self.left[2].fmadd_i128(e_in, left);
        }
    }

    pub(super) fn finish(self, e_out: F, gamma: F) -> [F; TERMS] {
        let mut out = [F::zero(); TERMS];
        for (slot, ((right, right_high), left)) in out
            .iter_mut()
            .zip(self.right.into_iter().zip(self.right_high).zip(self.left))
        {
            let right = right.reduce() + right_high.reduce().mul_pow_2(64);
            *slot = e_out * (right + gamma * left.reduce());
        }
        out
    }
}

#[inline]
fn ext_u64(even: u64, odd: u64) -> (i128, i128) {
    (i128::from(even), i128::from(odd) - i128::from(even))
}

#[inline]
fn ext_flag(even: bool, odd: bool) -> (i64, i64) {
    (i64::from(even), i64::from(odd) - i64::from(even))
}
