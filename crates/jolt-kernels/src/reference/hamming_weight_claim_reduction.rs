//! The Hamming-weight claim-reduction (stage 7) kernel: a naive member over
//! the committed chunk domain.
//!
//! In the base protocol, the summand is
//! `Σ_i G_i(k) · (γ^{3i} + γ^{3i+1}·eq(r_addr_bool, k) + γ^{3i+2}·eq(r_addr_virt_i, k))`.
//! The lattice relation keeps those three legs for RAM and uses the
//! digit-zero-recentered two-leg form for instruction and bytecode RA. Each
//! `G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j)` is the cycle fold of the committed
//! one-hot grid at the shared stage-6b cycle point (the booleanity opening point's
//! cycle suffix — every stage-6b member bound the same cycle challenges, so
//! all three reduced claim families live at that cycle). The eq publics are
//! one multilinear each over the chunk domain.

use std::collections::BTreeMap;

use crate::ProverInputs;
#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::lattice::geometry::balanced_inc_value;
use jolt_claims::protocols::jolt::{
    HammingWeightClaimReductionPublic, JoltCommittedPolynomial, JoltDerivedId, JoltPolynomialId,
    JoltRelationId,
};
use jolt_claims::{Source, SymbolicSumcheck};
use jolt_field::Field;
#[cfg(feature = "akita")]
use jolt_poly::boolean_point_msb;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
use jolt_witness::JoltWitnessPlane;

use jolt_verifier::stages::relations::ConcreteSumcheck;

use super::views::{cycle_fold, eq_table};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel,
};

impl<F: Field> PrepareKernel<F, HammingWeightClaimReduction<F>> for ReferenceBackend {
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
        let mut opening_tables = BTreeMap::new();
        for opening in dimensions
            .layout
            .openings(JoltRelationId::HammingWeightClaimReduction)
        {
            let mut table = cycle_fold(witness, opening, dimensions.log_k_chunk, r_cycle)?;
            if crate::reference::lattice_shape()
                && matches!(
                    opening.polynomial_id(),
                    JoltPolynomialId::Committed(
                        JoltCommittedPolynomial::InstructionRa(_)
                            | JoltCommittedPolynomial::BytecodeRa(_)
                    )
                )
            {
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

        #[cfg(feature = "akita")]
        {
            let table_len = 1usize << dimensions.log_k_chunk;
            let at_digit_zero = |point: &[F]| {
                point
                    .iter()
                    .fold(F::one(), |acc, coordinate| acc * (F::one() - *coordinate))
            };
            let _ = derived_tables.insert(
                JoltDerivedId::from(HammingWeightClaimReductionPublic::EqBooleanityAtDigitZero),
                Polynomial::new(vec![at_digit_zero(r_address); table_len]),
            );
            for (index, point) in virtualization_points.iter().enumerate() {
                let _ = derived_tables.insert(
                    JoltDerivedId::from(
                        HammingWeightClaimReductionPublic::EqVirtualizationAtDigitZero(index),
                    ),
                    Polynomial::new(vec![at_digit_zero(point); table_len]),
                );
            }
            let balanced_values = (0..table_len)
                .map(|lane| balanced_inc_value(&boolean_point_msb(dimensions.log_k_chunk, lane)))
                .collect();
            let _ = derived_tables.insert(
                JoltDerivedId::from(HammingWeightClaimReductionPublic::BalancedIncValueAtAddress),
                Polynomial::new(balanced_values),
            );
        }

        // The packed (lattice) shape extends the reduction with the fused-inc
        // one-hot columns and their little-endian decode: serve the extra
        // tables per the relation's own expression leaves (the base shape
        // references none of them, so this loop no-ops there — the kernel
        // adapts to the jolt-claims shape instead of carrying a feature).
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
                            if crate::reference::lattice_shape() {
                                table[0] = F::zero();
                            }
                            let _ = opening_tables.insert(*id, Polynomial::new(table));
                        }
                    }
                    Source::Derived(_) => {}
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
