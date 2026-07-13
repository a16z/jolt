//! The Spartan product-virtualization (stage 2) kernels: the product uni-skip
//! first-round polynomial and the product-remainder batch member.
//!
//! The uni-skip row polynomial
//! `t1(Y) = Σ_j eq(τ_low, j) · left_Y(j) · right_Y(j)` — with `left_Y`/`right_Y`
//! the centered-Lagrange-weighted combinations of the three left/right factor
//! columns — is brute-forced at all five nodes of the extended centered window
//! (domain size 3). Unlike stage 1's outer uni-skip, the in-domain values do
//! not vanish: they equal the three stage-1 product claims, and the engine's
//! round-sum check pins them against the folded input claim. The transmitted
//! polynomial is `LK(τ_high, ·) × t1` (degree 6).
//!
//! The remainder member needs no composite treatment: every leaf of the
//! product-remainder `Expr` is multilinear over the cycle domain (the Lagrange
//! weights are scalars — there is no stage-1-style quadratic stream
//! coefficient), so it is a plain [`NaiveSumcheckProver`], bound `LowToHigh`.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::dimensions::PRODUCT_UNISKIP_DOMAIN_SIZE;
use jolt_claims::protocols::jolt::geometry::spartan::{
    branch_flag_product, jump_flag_product, left_instruction_input_product, lookup_output_product,
    next_is_noop_product, right_instruction_input_product, virtual_instruction_product,
    write_lookup_output_to_rd_product, SpartanProductDimensions,
};
use jolt_claims::protocols::jolt::{JoltDerivedId, SpartanProductVirtualizationPublic};
use jolt_claims::NoChallenges;
use jolt_field::Field;
use jolt_poly::lagrange::{
    centered_lagrange_evals, centered_lagrange_kernel, interpolate_to_coeffs, poly_mul,
};
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use super::views::{dense_view, eq_table};
use crate::spartan_product::{SpartanProductInstance, SpartanProductProver};
use crate::{KernelError, NaiveSumcheckProver, ProofSession, ProveSumcheck, ReferenceBackend};

impl<F: Field> SpartanProductProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        log_t: usize,
        tau_low: &[F],
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn SpartanProductInstance<F>>, KernelError<F>> {
        Ok(Box::new(SpartanProductKernel::prepare(
            log_t, tau_low, witness,
        )?))
    }
}

/// The shared product compute state: the eight cycle-indexed factor/wire
/// tables and `eq(τ_low, ·)` — everything the uni-skip polynomial and the
/// remainder member both consume.
pub struct SpartanProductKernel<F: Field> {
    log_t: usize,
    tau_low: Vec<F>,
    eq_cycle: Vec<F>,
    left_instruction_input: Vec<F>,
    lookup_output: Vec<F>,
    jump_flag: Vec<F>,
    right_instruction_input: Vec<F>,
    branch_flag: Vec<F>,
    next_is_noop: Vec<F>,
    write_lookup_output_to_rd: Vec<F>,
    virtual_instruction: Vec<F>,
}

impl<F: Field> SpartanProductKernel<F> {
    pub fn prepare(
        log_t: usize,
        tau_low: &[F],
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Self, KernelError<F>> {
        Ok(Self {
            log_t,
            tau_low: tau_low.to_vec(),
            eq_cycle: eq_table(tau_low),
            left_instruction_input: dense_view(witness, left_instruction_input_product())?,
            lookup_output: dense_view(witness, lookup_output_product())?,
            jump_flag: dense_view(witness, jump_flag_product())?,
            right_instruction_input: dense_view(witness, right_instruction_input_product())?,
            branch_flag: dense_view(witness, branch_flag_product())?,
            next_is_noop: dense_view(witness, next_is_noop_product())?,
            write_lookup_output_to_rd: dense_view(witness, write_lookup_output_to_rd_product())?,
            virtual_instruction: dense_view(witness, virtual_instruction_product())?,
        })
    }
}

impl<F: Field> SpartanProductInstance<F> for SpartanProductKernel<F> {
    fn uniskip_first_round_poly(&self, tau_high: F) -> Result<UnivariatePoly<F>, KernelError<F>> {
        let extended_size = 2 * PRODUCT_UNISKIP_DOMAIN_SIZE - 1;
        let domain_start = -((PRODUCT_UNISKIP_DOMAIN_SIZE as i64 - 1) / 2);
        let extended_start = -((extended_size as i64 - 1) / 2);
        let cycles = 1usize << self.log_t;

        let mut t1_values = vec![F::zero(); extended_size];
        for (position, value) in t1_values.iter_mut().enumerate() {
            let node = extended_start + position as i64;
            let node_field = if node >= 0 {
                F::from_u64(node as u64)
            } else {
                -F::from_u64(node.unsigned_abs())
            };
            let weights = centered_lagrange_evals::<F>(PRODUCT_UNISKIP_DOMAIN_SIZE, node_field)?;
            let mut sum = F::zero();
            for j in 0..cycles {
                let left = weights[0] * self.left_instruction_input[j]
                    + weights[1] * self.lookup_output[j]
                    + weights[2] * self.jump_flag[j];
                let right = weights[0] * self.right_instruction_input[j]
                    + weights[1] * self.branch_flag[j]
                    + weights[2] * (F::one() - self.next_is_noop[j]);
                sum += self.eq_cycle[j] * left * right;
            }
            *value = sum;
        }

        let kernel_values = centered_lagrange_evals::<F>(PRODUCT_UNISKIP_DOMAIN_SIZE, tau_high)?;
        let kernel_coefficients = interpolate_to_coeffs(domain_start, &kernel_values);
        let t1_coefficients = interpolate_to_coeffs(extended_start, &t1_values);
        Ok(UnivariatePoly::new(poly_mul(
            &kernel_coefficients,
            &t1_coefficients,
        )))
    }

    /// The remainder member: the naive prover over the expanded product form.
    /// Each `LagrangeWeight(i)` leaf is the SCALAR `L_i(r₀)` (a constant
    /// table); `TauKernel` is the `LK(τ_high, r₀)`-scaled eq-cycle table.
    fn into_remainder(
        self: Box<Self>,
        tau_high: F,
        uniskip_challenge: F,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = ProductRemainder<F>>>, KernelError<F>> {
        let this = *self;
        let cycles = 1usize << this.log_t;
        let weights = centered_lagrange_evals::<F>(PRODUCT_UNISKIP_DOMAIN_SIZE, uniskip_challenge)?;
        let scale = centered_lagrange_kernel::<F>(
            PRODUCT_UNISKIP_DOMAIN_SIZE,
            tau_high,
            uniskip_challenge,
        )?;

        let mut derived_tables = BTreeMap::new();
        let _ = derived_tables.insert(
            JoltDerivedId::from(SpartanProductVirtualizationPublic::TauKernel),
            Polynomial::new(
                this.eq_cycle
                    .iter()
                    .map(|&eq| eq * scale)
                    .collect::<Vec<F>>(),
            ),
        );
        for (index, &weight) in weights.iter().enumerate() {
            let _ = derived_tables.insert(
                JoltDerivedId::from(SpartanProductVirtualizationPublic::LagrangeWeight(index)),
                Polynomial::new(vec![weight; cycles]),
            );
        }

        let opening_tables = BTreeMap::from([
            (
                left_instruction_input_product(),
                Polynomial::new(this.left_instruction_input),
            ),
            (lookup_output_product(), Polynomial::new(this.lookup_output)),
            (jump_flag_product(), Polynomial::new(this.jump_flag)),
            (
                right_instruction_input_product(),
                Polynomial::new(this.right_instruction_input),
            ),
            (branch_flag_product(), Polynomial::new(this.branch_flag)),
            (next_is_noop_product(), Polynomial::new(this.next_is_noop)),
            (
                write_lookup_output_to_rd_product(),
                Polynomial::new(this.write_lookup_output_to_rd),
            ),
            (
                virtual_instruction_product(),
                Polynomial::new(this.virtual_instruction),
            ),
        ]);

        let relation = ProductRemainder::new(
            SpartanProductDimensions::new(this.log_t),
            uniskip_challenge,
            tau_high,
            this.tau_low,
        );
        Ok(Box::new(NaiveSumcheckProver::new(
            relation,
            &NoChallenges::default(),
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
