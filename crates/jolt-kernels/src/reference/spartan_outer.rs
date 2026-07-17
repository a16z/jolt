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

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::dimensions::OUTER_UNISKIP_DOMAIN_SIZE;
use jolt_claims::protocols::jolt::geometry::spartan::{outer_opening, SpartanOuterDimensions};
use jolt_claims::protocols::jolt::{JoltDerivedId, JoltOpeningId, SpartanOuterPublic};
use jolt_claims::NoChallenges;
use jolt_field::Field;
use jolt_poly::lagrange::{centered_lagrange_evals, centered_lagrange_kernel, poly_mul};
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_r1cs::constraint::ConstraintMatrices;
use jolt_r1cs::constraints::jolt::{spartan_outer_constraints, spartan_outer_row_weights};
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use super::views::{dense_view, replicate_stream_lsb, stream_pair_lsb};
use crate::spartan_outer::{SpartanOuterInstance, SpartanOuterProver};
use crate::{KernelError, NaiveSumcheckProver, ProofSession, ReferenceBackend, SumcheckKernel};

impl<F: Field> SpartanOuterProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        log_t: usize,
        tau: &[F],
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn SpartanOuterInstance<F>>, KernelError<F>> {
        Ok(Box::new(SpartanOuterKernel::prepare(log_t, tau, witness)?))
    }
}

/// The shared stage-1 compute state: the 35 R1CS input tables, the
/// per-constraint Az/Bz row-value tables, and `eq(τ_low, ·)` — everything the
/// uni-skip polynomial and the remainder member both consume.
pub struct SpartanOuterKernel<F: Field> {
    log_t: usize,
    tau: Vec<F>,
    matrices: ConstraintMatrices<F>,
    /// Cycle-indexed R1CS input tables (big-endian cycle index), in the
    /// relation's variable order.
    input_tables: Vec<Vec<F>>,
    /// Per-constraint-row value tables over the cycle domain:
    /// `az_rows[r][t] = Σ_(v,α)∈A_r α · z_t[v]`.
    az_rows: Vec<Vec<F>>,
    bz_rows: Vec<Vec<F>>,
    /// eq(τ_low, ·) over the (cycle ∥ stream) index `j = (t << 1) | s`
    /// (τ_low[0] pairs the index MSB, so the stream bit pairs τ_low's last
    /// entry — legacy's convention).
    eq_table: Vec<F>,
}

impl<F: Field> SpartanOuterKernel<F> {
    /// Materialize the stage's compute state from the witness. `tau` is the
    /// stage's full challenge vector (`log_t + 2` entries).
    pub fn prepare(
        log_t: usize,
        tau: &[F],
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Self, KernelError<F>> {
        let dimensions = SpartanOuterDimensions::rv64(log_t);
        let input_tables = materialize_input_tables(witness, &dimensions)?;
        let matrices = spartan_outer_constraints::<F>();
        let (az_rows, bz_rows) = row_value_tables(&matrices, &input_tables);
        let (tau_low, _) = tau.split_at(log_t + 1);
        let eq_table = EqPolynomial::new(tau_low.to_vec()).evaluations();
        Ok(Self {
            log_t,
            tau: tau.to_vec(),
            matrices,
            input_tables,
            az_rows,
            bz_rows,
            eq_table,
        })
    }
}

impl<F: Field> SpartanOuterInstance<F> for SpartanOuterKernel<F> {
    /// Brute-force the uni-skip first-round polynomial. The summand's
    /// row-node polynomial
    /// `t1(Y) = Σ_(s,t) eq(τ_low, (t,s)) · Az(Y,s,t) · Bz(Y,s,t)` vanishes on
    /// the 10 in-domain nodes (each row is a satisfied
    /// `guard · (left − right) = 0`), so `t1` is interpolated over the
    /// 19-point centered domain from the 9 extended-node evaluations; the
    /// transmitted polynomial is `LK(τ_high, ·) × t1`.
    fn uniskip_first_round_poly(&self) -> Result<UnivariatePoly<F>, KernelError<F>> {
        let tau_high = self.tau[self.log_t + 1];
        let extended_size = 2 * OUTER_UNISKIP_DOMAIN_SIZE - 1;
        let domain_start = -((OUTER_UNISKIP_DOMAIN_SIZE as i64 - 1) / 2);
        let extended_start = -((extended_size as i64 - 1) / 2);
        let domain_end = domain_start + OUTER_UNISKIP_DOMAIN_SIZE as i64;

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

        let kernel_values = centered_lagrange_evals::<F>(OUTER_UNISKIP_DOMAIN_SIZE, tau_high)?;
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
    /// jolt-r1cs functions the verifier's coefficient build uses; each is
    /// linear in the stream variable, so every derived leaf materializes as
    /// one multilinear table.
    fn into_remainder(
        self: Box<Self>,
        uniskip_challenge: F,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = OuterRemainder<F>>>, KernelError<F>> {
        let this = *self;
        let kernel = centered_lagrange_kernel::<F>(
            OUTER_UNISKIP_DOMAIN_SIZE,
            this.tau[this.log_t + 1],
            uniskip_challenge,
        )?;

        let variable_count = this.input_tables.len();
        let columns: Vec<usize> = (1..=variable_count).collect();
        let mut az_columns: [Vec<F>; 2] = [Vec::new(), Vec::new()];
        let mut bz_columns: [Vec<F>; 2] = [Vec::new(), Vec::new()];
        let mut az_constant = [F::zero(); 2];
        let mut bz_constant = [F::zero(); 2];
        for (index, stream) in [F::zero(), F::one()].into_iter().enumerate() {
            let weights = spartan_outer_row_weights(uniskip_challenge, stream)?;
            let weighted = this.matrices.weighted_columns(&weights, &columns)?;
            az_columns[index] = weighted.a;
            bz_columns[index] = weighted.b;
            let constants = this
                .matrices
                .public_column_contributions(&weights, 0, F::one())?;
            az_constant[index] = constants.a;
            bz_constant[index] = constants.b;
        }

        let cycles = 1usize << this.log_t;
        let mut derived_tables = BTreeMap::new();
        let _ = derived_tables.insert(
            JoltDerivedId::from(SpartanOuterPublic::TauKernel),
            Polynomial::new(
                this.eq_table
                    .iter()
                    .map(|&eq| eq * kernel)
                    .collect::<Vec<F>>(),
            ),
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

        let dimensions = SpartanOuterDimensions::rv64(this.log_t);
        let opening_tables: BTreeMap<JoltOpeningId, Polynomial<F>> = dimensions
            .variables()
            .iter()
            .zip(&this.input_tables)
            .map(|(&variable, table)| {
                (
                    outer_opening(variable),
                    Polynomial::new(replicate_stream_lsb(table)),
                )
            })
            .collect();

        let relation = OuterRemainder::new(dimensions, this.tau, uniskip_challenge);
        Ok(Box::new(NaiveSumcheckProver::new(
            relation,
            &NoChallenges::default(),
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}

/// Materialize the 35 R1CS input polynomials (cycle-indexed, big-endian) in
/// the relation's variable order.
fn materialize_input_tables<F: Field>(
    witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    dimensions: &SpartanOuterDimensions,
) -> Result<Vec<Vec<F>>, KernelError<F>> {
    dimensions
        .variables()
        .iter()
        .map(|&variable| dense_view(witness, outer_opening(variable)))
        .collect()
}

/// Per-constraint-row Az/Bz value tables over the cycle domain:
/// `az_rows[r][t] = Σ_(v,α)∈A_r α · z_t[v]` with `z_t[0] = 1` and
/// `z_t[1 + k] = input_tables[k][t]`.
fn row_value_tables<F: Field>(
    matrices: &ConstraintMatrices<F>,
    input_tables: &[Vec<F>],
) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
    let cycles = input_tables.first().map_or(0, Vec::len);
    let row_values = |rows: &[Vec<(usize, F)>]| {
        rows.iter()
            .map(|row| {
                (0..cycles)
                    .map(|t| {
                        row.iter()
                            .map(|&(variable, coefficient)| {
                                if variable == 0 {
                                    coefficient
                                } else {
                                    coefficient * input_tables[variable - 1][t]
                                }
                            })
                            .sum()
                    })
                    .collect::<Vec<F>>()
            })
            .collect::<Vec<_>>()
    };
    (row_values(&matrices.a), row_values(&matrices.b))
}

#[cfg(test)]
mod orientation_probes {
    use jolt_field::{Fr, FromPrimitiveInt};
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
