//! Input claim formulas for each sumcheck instance.
//!
//! Each function computes the `claimed_sum` for a sumcheck stage from prior
//! stage evaluations. Both prover and verifier call these identically — the
//! prover populates stage evals from polynomial evaluation, the verifier from
//! proof data.
//!
//! Verified against jolt-core's `SumcheckInstanceParams::input_claim()`.

use jolt_field::Field;

use super::types::*;

#[inline]
pub fn s2_ram_rw<F: Field>(s1: &SpartanVirtualEvals<F>, gamma: F) -> F {
    s1.ram_read_value.0 + gamma * s1.ram_write_value.0
}

#[inline]
pub fn s2_instruction_lookups_cr<F: Field>(s1: &SpartanVirtualEvals<F>, gamma: F) -> F {
    let g2 = gamma * gamma;
    s1.lookup_output.0
        + gamma * s1.left_operand.0
        + g2 * s1.right_operand.0
        + g2 * gamma * s1.left_instruction_input.0
        + g2 * gamma * gamma * s1.right_instruction_input.0
}

#[inline]
pub fn s3_registers_cr<F: Field>(s1: &SpartanVirtualEvals<F>, gamma: F) -> F {
    s1.rd_write_value.0 + gamma * s1.rs1_value.0 + gamma * gamma * s1.rs2_value.0
}

#[inline]
pub fn s3_instruction_input<F: Field>(s2: &S2Evals<F>, gamma: F) -> F {
    s2.right_instr_input.0 + gamma * s2.left_instr_input.0
}

#[inline]
pub fn s4_registers_rw<F: Field>(s3: &S3Evals<F>, gamma: F) -> F {
    s3.rd_write_value.0 + gamma * (s3.rs1_value.0 + gamma * s3.rs2_value.0)
}

#[inline]
pub fn s4_ram_val_check<F: Field>(s2: &S2Evals<F>, init_eval: F, gamma: F) -> F {
    (s2.ram_val.0 - init_eval) + gamma * (s2.ram_val_final.0 - init_eval)
}

/// 4 committed evals from S2/S4/S5 reduced to a single claim.
#[inline]
pub fn s6_inc_cr<F: Field>(s2: &S2Evals<F>, s4: &S4Evals<F>, s5: &S5Evals<F>, gamma: F) -> F {
    let g2 = gamma * gamma;
    s2.ram_inc.eval
        + gamma * s4.ram_inc.eval
        + g2 * s4.rd_inc.eval
        + g2 * gamma * s5.rd_inc.eval
}

/// Batches 3 claim types per RA polynomial: hw + bool + virtual.
#[inline]
pub fn s7_hamming_weight_cr<F: Field>(
    claims_hw: &[F],
    claims_bool: &[F],
    claims_virt: &[F],
    gamma_powers: &[F],
) -> F {
    debug_assert_eq!(claims_hw.len(), claims_bool.len());
    debug_assert_eq!(claims_hw.len(), claims_virt.len());
    let mut claim = F::zero();
    for i in 0..claims_hw.len() {
        claim += gamma_powers[3 * i] * claims_hw[i];
        claim += gamma_powers[3 * i + 1] * claims_bool[i];
        claim += gamma_powers[3 * i + 2] * claims_virt[i];
    }
    claim
}
