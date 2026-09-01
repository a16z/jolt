//! The Hamming-weight claim-reduction (stage 7) kernel: a naive member over
//! the committed chunk domain.
//!
//! The summand is
//! `Σ_i G_i(k) · (γ^{3i} + γ^{3i+1}·eq(r_addr_bool, k) + γ^{3i+2}·eq(r_addr_virt_i, k))`
//! — reducing each checked one-hot polynomial's Hamming-weight, booleanity,
//! and virtualization claims to one fresh opening. Each `G_i(k) =
//! Σ_j eq(r_cycle, j) · ra_i(k, j)` is the cycle fold of the committed one-hot
//! grid at the shared stage-6b cycle point (the booleanity opening point's
//! cycle suffix — every stage-6b member bound the same cycle challenges, so
//! all three reduced claim families live at that cycle). The eq publics are
//! one multilinear each over the chunk domain.

use std::collections::BTreeMap;

use crate::ProverInputs;
use jolt_claims::protocols::jolt::{
    HammingWeightClaimReductionPublic, JoltCommittedPolynomial, JoltDerivedId, JoltPolynomialId,
    JoltRelationId,
};
use jolt_claims::{Source, SymbolicSumcheck};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
use jolt_witness::JoltWitnessPlane;

use jolt_verifier::stages::relations::ConcreteSumcheck;

use super::views::{cycle_fold, eq_table};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel,
};

impl<F: JoltField> PrepareKernel<F, HammingWeightClaimReduction<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, HammingWeightClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = HammingWeightClaimReduction<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let r_cycle = relation.r_cycle();
        let r_address = relation.r_address();
        let virtualization_points = relation.virtualization_points();
        if r_address.len() != dimensions.log_k_chunk
            || virtualization_points.len() != dimensions.layout.total()
        {
            return Err(KernelError::InvariantViolation {
                reason: "hamming reduction reference point shapes disagree with the layout",
            });
        }
        // Digit-zero virtualization (lattice shape only): the committed
        // polynomial of every constant-activation family omits its row-zero
        // cells, so the reduced claim opens the pushforward with the
        // digit-zero row zeroed (`specs/digit-zero-virtualization.md`). RAM
        // keeps the base fully-committed treatment.
        let virtualizes_digit_zero = |id: &jolt_claims::protocols::jolt::JoltOpeningId| {
            super::lattice_shape()
                && matches!(
                    id.polynomial_id(),
                    JoltPolynomialId::Committed(
                        JoltCommittedPolynomial::InstructionRa(_)
                            | JoltCommittedPolynomial::BytecodeRa(_)
                            | JoltCommittedPolynomial::BalancedIncDigit(_)
                            | JoltCommittedPolynomial::BalancedIncCarry,
                    )
                )
        };
        let mut opening_tables = BTreeMap::new();
        for opening in dimensions
            .layout
            .openings(JoltRelationId::HammingWeightClaimReduction)
        {
            let mut table = cycle_fold(witness, opening, dimensions.log_k_chunk, r_cycle)?;
            if virtualizes_digit_zero(&opening) {
                table[0] = F::zero();
            }
            let _ = opening_tables.insert(opening, Polynomial::new(table));
        }

        let mut derived_tables = BTreeMap::from([(
            JoltDerivedId::from(HammingWeightClaimReductionPublic::EqBooleanity),
            Polynomial::new(eq_table(r_address)),
        )]);
        for (index, point) in virtualization_points.iter().enumerate() {
            let _ = derived_tables.insert(
                JoltDerivedId::from(HammingWeightClaimReductionPublic::EqVirtualization(index)),
                Polynomial::new(eq_table(point)),
            );
        }

        // The packed (lattice) shape extends the reduction with the fused-inc
        // one-hot columns, their centered little-endian decode, and the
        // digit-zero recentering publics: serve the extra tables per the
        // relation's own expression leaves (the base shape references none of
        // them, so this loop no-ops there — the kernel adapts to the
        // jolt-claims shape instead of carrying a feature).
        let k = 1u64 << dimensions.log_k_chunk;
        let eq_at_digit_zero =
            |point: &[F]| point.iter().map(|value| F::one() - *value).product::<F>();
        for term in &relation.symbolic().output_expression::<F>().terms {
            for factor in &term.factors {
                match factor {
                    Source::Opening(id) => {
                        if matches!(
                            id.polynomial_id(),
                            JoltPolynomialId::Committed(
                                JoltCommittedPolynomial::BalancedIncDigit(_)
                                    | JoltCommittedPolynomial::BalancedIncCarry,
                            )
                        ) && !opening_tables.contains_key(id)
                        {
                            let mut table =
                                cycle_fold(witness, *id, dimensions.log_k_chunk, r_cycle)?;
                            if virtualizes_digit_zero(id) {
                                table[0] = F::zero();
                            }
                            let _ = opening_tables.insert(*id, Polynomial::new(table));
                        }
                    }
                    Source::Derived(id) => {
                        if derived_tables.contains_key(id) {
                            continue;
                        }
                        let table = match id {
                            // The centered chunk-domain value `k ↦ k` for
                            // `k < K/2`, `k − K` otherwise; LowToHigh binding
                            // reproduces the verifier's `balanced_inc_value`
                            // bound evaluation.
                            JoltDerivedId::HammingWeightClaimReduction(
                                HammingWeightClaimReductionPublic::BalancedIncValueAtAddress,
                            ) => Some(
                                (0..k)
                                    .map(|row| {
                                        F::from_i128(
                                            row as i128 - if row < k / 2 { 0 } else { k as i128 },
                                        )
                                    })
                                    .collect::<Vec<_>>(),
                            ),
                            // The digit-zero recentering baselines are
                            // constant in the chunk variable: `eq(point, 0)`.
                            JoltDerivedId::HammingWeightClaimReduction(
                                HammingWeightClaimReductionPublic::EqBooleanityAtDigitZero,
                            ) => Some(vec![eq_at_digit_zero(r_address); k as usize]),
                            JoltDerivedId::HammingWeightClaimReduction(
                                HammingWeightClaimReductionPublic::EqVirtualizationAtDigitZero(
                                    index,
                                ),
                            ) => virtualization_points
                                .get(*index)
                                .map(|point| vec![eq_at_digit_zero(point); k as usize]),
                            _ => None,
                        };
                        if let Some(table) = table {
                            let _ = derived_tables.insert(*id, Polynomial::new(table));
                        }
                    }
                    Source::Challenge(_) => {}
                }
            }
        }

        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
