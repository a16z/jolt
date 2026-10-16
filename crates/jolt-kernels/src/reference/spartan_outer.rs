//! The Spartan-outer (stage 1) kernels: the uni-skip first-round polynomial
//! and the outer-remainder sumcheck member, computed naively at harness
//! scale.
//!
//! The uni-skip first-round polynomial is brute-forced from the R1CS
//! constraint rows: the summand over `(row-node Y, stream s, cycle t)` is
//! `LK(τ_high, Y) · eq(τ_low, (t,s)) · Az(Y,s,t) · Bz(Y,s,t)`, which vanishes
//! at the 10 in-domain row nodes for a satisfying witness, so only the 9
//! extended-node evaluations are computed. The remainder member is a plain
//! [`NaiveSumcheckProver`] over the joint `(cycle ‖ stream)` domain (stream =
//! index LSB): with the relation's factored form, every derived leaf is one
//! multilinear — the `TauKernel` eq table and the per-column `Az`/`Bz`
//! weights, each linear in the stream variable — bound `LowToHigh` to match
//! the legacy prover's convention.
//!
//! Brute-force costs are `O(T · |constraints|)` and table memory
//! `O(|inputs|² · T)` — a bring-up implementation; a streaming kernel
//! replaces these internals for real trace lengths without touching the
//! `jolt-prover` stage recipe.

#[cfg(feature = "field-inline")]
use jolt_claims::protocols::composed::ComposedOpeningId;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::geometry::spartan::outer_output_openings as field_outer_output_openings;
use std::collections::BTreeMap;

#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::geometry::spartan::FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlinePolynomialId;
use jolt_claims::protocols::jolt::geometry::spartan::{outer_opening, SpartanOuterDimensions};
use jolt_claims::protocols::jolt::{JoltDerivedId, JoltOpeningId, SpartanOuterPublic};
use jolt_field::JoltField;
use jolt_poly::lagrange::{centered_lagrange_evals, centered_lagrange_kernel, poly_mul};
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_r1cs::constraint::ConstraintMatrices;
// The COMPOSED R1CS shapes (feature-aware): identical to the jolt-claims RV64-only
// constants without field-inline, the field-inline-extended row/column composition
// under `field-inline` — the shapes the composed verifier checks.
use jolt_claims::protocols::composed::r1cs::{
    spartan_outer_constraints, spartan_outer_opening_columns, spartan_outer_row_weights,
    SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
};
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
use jolt_witness::JoltWitnessOracle;
#[cfg(feature = "field-inline")]
use jolt_witness::WitnessError;

use super::views::stream_pair_lsb;
use super::views::{dense_view, replicate_stream_lsb};
use crate::uniskip::UniskipKernel;
use crate::NaiveSumcheckProver;
use crate::ProverInputs;
use crate::{KernelError, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel};
use jolt_witness::JoltWitnessPlane;
impl<F: JoltField> UniskipKernel<F, OuterRemainder<F>> for ReferenceBackend {
    // The backend-neutral `SpartanOuterUniskip::*` spans live at the stage-1
    // call boundary (`crates/jolt-prover/src/stages/stage1.rs`), so every
    // `UniskipKernel` implementation inherits them — see the taxonomy's
    // kernel-seam contract.
    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau: &[F],
        witness: &dyn JoltWitnessPlane<F>,
    ) -> Result<(), KernelError<F>> {
        session.park(SpartanOuterKernel::prepare(log_t, tau, witness)?);
        Ok(())
    }

    fn first_round_poly(
        &self,
        session: &mut ProofSession,
        _late_tau: &[F],
        _inputs: &(),
    ) -> Result<UnivariatePoly<F>, KernelError<F>> {
        session
            .state::<SpartanOuterKernel<F>>()
            .ok_or(KernelError::InvariantViolation {
                reason: "the outer uni-skip slot parked no kernel for the first-round polynomial",
            })?
            .uniskip_first_round_poly()
    }
}

/// The stage-1 remainder slot server: reclaims the [`SpartanOuterKernel`] the
/// uni-skip slot parked and binds it into the batch member.
pub struct ReferenceOuterRemainder;

impl<F: JoltField> PrepareKernel<F, OuterRemainder<F>> for ReferenceOuterRemainder {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, OuterRemainder<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = OuterRemainder<F>>>, KernelError<F>> {
        session
            .take::<SpartanOuterKernel<F>>()
            .ok_or(KernelError::InvariantViolation {
                reason: "the outer uni-skip slot parked no kernel for the remainder member",
            })?
            .into_remainder(&inputs)
    }
}

/// The shared stage-1 compute state: the ordinary R1CS input tables, the
/// per-constraint Az/Bz row-value tables, and `eq(τ_low, ·)` — everything the
/// uni-skip polynomial and the remainder member both consume.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub struct SpartanOuterKernel<F: JoltField> {
    log_t: usize,
    tau: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    matrices: ConstraintMatrices<F>,
    /// The composed opening-column selection (`spartan_outer_opening_columns`), aligned
    /// index-for-index with `input_tables`. Not contiguous under `field-inline`: the
    /// two rv64 product-factor columns sit between the ordinary inputs and the appended
    /// field-inline columns.
    columns: Vec<usize>,
    /// Cycle-indexed R1CS input tables (big-endian cycle index), in the composed
    /// opening-column order: the relation's ordinary variables, then (under `field-inline`)
    /// the five field-inline columns.
    input_tables: Vec<Vec<F>>,
    az_rows: Vec<Vec<F>>,
    bz_rows: Vec<Vec<F>>,
    /// eq(τ_low, ·) over the (cycle ∥ stream) index `j = (t << 1) | s`
    /// (τ_low[0] pairs the index MSB, so the stream bit pairs τ_low's last
    /// entry — legacy's convention).
    eq_table: Vec<F>,
}

impl<F: JoltField> SpartanOuterKernel<F> {
    /// Materialize the stage's compute state from the witness. `tau` is the
    /// stage's full challenge vector (`log_t + 2` entries).
    pub fn prepare(
        log_t: usize,
        tau: &[F],
        witness: &dyn JoltWitnessOracle<F>,
    ) -> Result<Self, KernelError<F>> {
        let dimensions = SpartanOuterDimensions::rv64(log_t);
        let input_tables = materialize_input_tables(witness, &dimensions)?;
        let matrices = spartan_outer_constraints::<F>();
        let columns = spartan_outer_opening_columns();
        let (az_rows, bz_rows) = row_value_tables(&matrices, &input_tables, &columns)?;
        let (tau_low, _) = tau.split_at(log_t + 1);
        let eq_table = EqPolynomial::new(tau_low.to_vec()).evaluations();
        Ok(Self {
            log_t,
            tau: tau.to_vec(),
            matrices,
            columns,
            input_tables,
            az_rows,
            bz_rows,
            eq_table,
        })
    }

    fn uniskip_first_round_poly(&self) -> Result<UnivariatePoly<F>, KernelError<F>> {
        let tau_high = self.tau[self.log_t + 1];
        let extended_size = 2 * SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE - 1;
        let domain_start = -((SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE as i64 - 1) / 2);
        let extended_start = -((extended_size as i64 - 1) / 2);
        let domain_end = domain_start + SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE as i64;

        let cycles = 1usize << self.log_t;
        let mut t1_values = vec![F::zero(); extended_size];
        for (position, value) in t1_values.iter_mut().enumerate() {
            let node = extended_start + position as i64;
            if node >= domain_start && node < domain_end {
                continue;
            }
            let node_field = if node >= 0 {
                F::from_u64(node as u64)
            } else {
                -F::from_u64(node.unsigned_abs())
            };
            let weights = [
                spartan_outer_row_weights(node_field, F::zero())?,
                spartan_outer_row_weights(node_field, F::one())?,
            ];
            let mut sum = F::zero();
            for t in 0..cycles {
                for (s, stream_weights) in weights.iter().enumerate() {
                    let az: F = stream_weights
                        .iter()
                        .enumerate()
                        .map(|(row, &w)| w * self.az_rows[row][t])
                        .sum();
                    let bz: F = stream_weights
                        .iter()
                        .enumerate()
                        .map(|(row, &w)| w * self.bz_rows[row][t])
                        .sum();
                    sum += self.eq_table[(t << 1) | s] * az * bz;
                }
            }
            *value = sum;
        }

        let kernel_values =
            centered_lagrange_evals::<F>(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, tau_high)?;
        let kernel_coefficients =
            jolt_poly::lagrange::interpolate_to_coeffs(domain_start, &kernel_values);
        let t1_coefficients =
            jolt_poly::lagrange::interpolate_to_coeffs(extended_start, &t1_values);
        Ok(UnivariatePoly::new(poly_mul(
            &kernel_coefficients,
            &t1_coefficients,
        )))
    }

    /// Consume the shared state into the remainder batch member, once the
    /// uni-skip round's challenge is drawn: a plain naive member over the
    /// joint `(cycle ‖ stream)` domain, `index = (t << 1) | s`. The
    /// per-stream `Az`/`Bz` linear forms are single-sourced from the same
    /// jolt-claims functions the verifier's coefficient build uses; each is
    /// linear in the stream variable, so every derived leaf materializes as
    /// one multilinear table.
    fn into_remainder(
        self,
        inputs: &ProverInputs<'_, F, OuterRemainder<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = OuterRemainder<F>>>, KernelError<F>> {
        let uniskip_challenge = inputs.relation.uniskip_challenge();
        let kernel = centered_lagrange_kernel::<F>(
            SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
            self.tau[self.log_t + 1],
            uniskip_challenge,
        )?;

        let mut az_columns: [Vec<F>; 2] = [Vec::new(), Vec::new()];
        let mut bz_columns: [Vec<F>; 2] = [Vec::new(), Vec::new()];
        let mut az_constant = [F::zero(); 2];
        let mut bz_constant = [F::zero(); 2];
        for (index, stream) in [F::zero(), F::one()].into_iter().enumerate() {
            let weights = spartan_outer_row_weights(uniskip_challenge, stream)?;
            let weighted = self.matrices.weighted_columns(&weights, &self.columns)?;
            az_columns[index] = weighted.a;
            bz_columns[index] = weighted.b;
            let constants = self
                .matrices
                .public_column_contributions(&weights, 0, F::one())?;
            az_constant[index] = constants.a;
            bz_constant[index] = constants.b;
        }

        let cycles = 1usize << self.log_t;
        let tau_kernel_table = self
            .eq_table
            .iter()
            .map(|&eq| eq * kernel)
            .collect::<Vec<F>>();

        let variable_count = self.input_tables.len();
        let mut derived_tables = BTreeMap::new();
        let _ = derived_tables.insert(
            JoltDerivedId::from(SpartanOuterPublic::TauKernel),
            Polynomial::new(tau_kernel_table),
        );
        for index in 0..variable_count {
            let _ = derived_tables.insert(
                JoltDerivedId::from(SpartanOuterPublic::AzWeight(index)),
                Polynomial::new(stream_pair_lsb(
                    [az_columns[0][index], az_columns[1][index]],
                    cycles,
                )),
            );
            let _ = derived_tables.insert(
                JoltDerivedId::from(SpartanOuterPublic::BzWeight(index)),
                Polynomial::new(stream_pair_lsb(
                    [bz_columns[0][index], bz_columns[1][index]],
                    cycles,
                )),
            );
        }
        let _ = derived_tables.insert(
            JoltDerivedId::from(SpartanOuterPublic::AzConstant),
            Polynomial::new(stream_pair_lsb(az_constant, cycles)),
        );
        let _ = derived_tables.insert(
            JoltDerivedId::from(SpartanOuterPublic::BzConstant),
            Polynomial::new(stream_pair_lsb(bz_constant, cycles)),
        );

        let dimensions = SpartanOuterDimensions::rv64(self.log_t);
        let opening_tables: BTreeMap<JoltOpeningId, Polynomial<F>> = dimensions
            .variables()
            .iter()
            .zip(&self.input_tables)
            .map(|(&variable, table)| {
                (
                    outer_opening(variable),
                    Polynomial::new(replicate_stream_lsb(table)),
                )
            })
            .collect();

        #[cfg(feature = "field-inline")]
        let opening_tables = opening_tables
            .into_iter()
            .map(|(id, table)| (ComposedOpeningId::from(id), table))
            .chain(
                field_outer_output_openings()
                    .into_iter()
                    .zip(self.input_tables.iter().skip(dimensions.variables().len()))
                    .map(|(id, table)| {
                        (
                            ComposedOpeningId::from(id),
                            Polynomial::new(replicate_stream_lsb(table)),
                        )
                    }),
            )
            .collect();

        Ok(Box::new(NaiveSumcheckProver::new(
            inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}

/// Materialize the selected R1CS input polynomials (cycle-indexed, big-endian) in the
/// composed opening-column order: the ordinary rv64 inputs in the relation's variable order,
/// then (under `field-inline`) the five field-inline columns in
/// `FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS` order — matching
/// `spartan_outer_opening_columns()` index-for-index. Fails closed when a build with
/// field-inline enabled proves a witness without the field-inline oracle.
fn materialize_input_tables<F: JoltField>(
    witness: &dyn JoltWitnessOracle<F>,
    dimensions: &SpartanOuterDimensions,
) -> Result<Vec<Vec<F>>, KernelError<F>> {
    #[cfg_attr(not(feature = "field-inline"), expect(unused_mut))]
    let mut tables = dimensions
        .variables()
        .iter()
        .map(|&variable| dense_view(witness, outer_opening(variable)))
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(feature = "field-inline")]
    {
        let field_inline =
            witness
                .field_inline()
                .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                    label: "composed Spartan outer field-inline oracle",
                }))?;
        for polynomial in FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS {
            tables.push(field_inline.oracle_table(FieldInlinePolynomialId::Virtual(polynomial))?);
        }
    }
    Ok(tables)
}

/// Per-constraint-row Az/Bz value tables over the cycle domain:
/// `az_rows[r][t] = Σ_(v,α)∈A_r α · z_t[v]` with `z_t[0] = 1` and
/// `z_t[columns[k]] = input_tables[k][t]` — the composed opening columns are
/// not contiguous under `field-inline`, so the matrix column index resolves
/// through the selection rather than by offset. A constraint referencing a
/// non-selected, non-constant column is a composition bug, surfaced here.
#[expect(
    clippy::type_complexity,
    reason = "the Az/Bz row-table pair, now fallible under the composed column selection"
)]
fn row_value_tables<F: JoltField>(
    matrices: &ConstraintMatrices<F>,
    input_tables: &[Vec<F>],
    columns: &[usize],
) -> Result<(Vec<Vec<F>>, Vec<Vec<F>>), KernelError<F>> {
    let mut column_to_table: Vec<Option<usize>> = vec![None; matrices.num_vars];
    for (position, &column) in columns.iter().enumerate() {
        *column_to_table
            .get_mut(column)
            .ok_or(KernelError::InvariantViolation {
                reason: "Spartan outer opening column exceeds the constraint variable count",
            })? = Some(position);
    }
    let cycles = input_tables.first().map_or(0, Vec::len);
    let row_values = |rows: &[Vec<(usize, F)>]| -> Result<Vec<Vec<F>>, KernelError<F>> {
        rows.iter()
            .map(|row| {
                (0..cycles)
                    .map(|t| {
                        row.iter()
                            .map(|&(variable, coefficient)| {
                                if variable == 0 {
                                    return Ok(coefficient);
                                }
                                let table =
                                    column_to_table.get(variable).copied().flatten().ok_or(
                                        KernelError::InvariantViolation {
                                            reason: "Spartan outer constraint references a \
                                                 non-opening column",
                                        },
                                    )?;
                                Ok(coefficient * input_tables[table][t])
                            })
                            .sum::<Result<F, KernelError<F>>>()
                    })
                    .collect::<Result<Vec<F>, _>>()
            })
            .collect()
    };
    Ok((row_values(&matrices.a)?, row_values(&matrices.b)?))
}

#[cfg(test)]
mod orientation_probes {
    use jolt_field::{Fr, Ring};
    use jolt_poly::{BindingOrder, EqPolynomial, Polynomial};

    /// Pin the composite orientation assumption: an `EqPolynomial` table
    /// (big-endian, `point[0]` at the index MSB) bound LowToHigh over
    /// challenges `c_0.., c_n` must land on `mle(point, reversed challenges)`.
    #[test]
    fn eq_table_low_to_high_binding_matches_reversed_mle() {
        let tau: Vec<Fr> = (0..3).map(|i| Fr::from_u64(11 + 3 * i)).collect();
        let challenges: Vec<Fr> = (0..3).map(|i| Fr::from_u64(101 + 7 * i)).collect();

        let mut table = Polynomial::new(EqPolynomial::new(tau.clone()).evaluations());
        for &challenge in &challenges {
            table.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        let bound = table.evals()[0];

        let reversed: Vec<Fr> = challenges.iter().rev().copied().collect();
        assert_eq!(bound, EqPolynomial::mle(&tau, &reversed));
    }
}
