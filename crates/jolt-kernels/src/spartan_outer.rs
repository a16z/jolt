//! The Spartan-outer (stage 1) kernels: the uni-skip first-round polynomial
//! and the outer-remainder sumcheck member, computed naively at harness
//! scale.
//!
//! The uni-skip first-round polynomial is brute-forced from the R1CS
//! constraint rows: the summand over `(row-node Y, stream s, cycle t)` is
//! `LK(τ_high, Y) · eq(τ_low, (t,s)) · Az(Y,s,t) · Bz(Y,s,t)`, which vanishes
//! at the 10 in-domain row nodes for a satisfying witness, so only the 9
//! extended-node evaluations are computed. The remainder member is composite
//! (see [`OuterRemainderKernel`]): the stream round is hand-computed — its
//! `Expr` coefficient leaves are QUADRATIC in the stream variable, so the
//! naive multilinear-table premise does not hold for that one round — and the
//! remaining cycle rounds are a plain [`NaiveSumcheckProver`] over the
//! expanded quadratic form of the 35 R1CS input openings, bound `LowToHigh`
//! to match the legacy prover's convention.
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
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::SumcheckOutputClaims;
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
use jolt_witness::protocols::jolt_vm::{jolt_opening_oracle_ref, JoltVmNamespace};
use jolt_witness::{
    MaterializationPolicy, PolynomialEncoding, RetentionHint, ViewRequirement, WitnessProvider,
};

use crate::{KernelError, NaiveSumcheckProver, ProofSession, ProveSumcheck, ReferenceBackend};

/// The stage-1 slot: factory for a prepared Spartan-outer instance.
pub trait SpartanOuterProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau: &[F],
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn SpartanOuterInstance<F>>, KernelError<F>>;
}

/// A prepared Spartan-outer instance: the uni-skip first-round polynomial,
/// then — once the uni-skip challenge is drawn — the remainder batch member.
pub trait SpartanOuterInstance<F: Field> {
    fn uniskip_first_round_poly(&self) -> Result<UnivariatePoly<F>, KernelError<F>>;

    fn into_remainder(
        self: Box<Self>,
        uniskip_challenge: F,
    ) -> Result<Box<dyn OuterRemainderInstance<F>>, KernelError<F>>;
}

/// The remainder batch member, plus typed claim extraction once fully bound.
pub trait OuterRemainderInstance<F: Field>: ProveRounds<F> {
    fn output_claims(
        &mut self,
    ) -> Result<SumcheckOutputClaims<F, OuterRemainder<F>>, KernelError<F>>;
}

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

    /// Consume the shared state into the remainder sumcheck member, once the
    /// uni-skip round's challenge is drawn.
    fn into_remainder(
        self: Box<Self>,
        uniskip_challenge: F,
    ) -> Result<Box<dyn OuterRemainderInstance<F>>, KernelError<F>> {
        let this = *self;
        let kernel = centered_lagrange_kernel::<F>(
            OUTER_UNISKIP_DOMAIN_SIZE,
            this.tau[this.log_t + 1],
            uniskip_challenge,
        )?;

        let variable_count = this.input_tables.len();
        let columns: Vec<usize> = (1..=variable_count).collect();
        let mut az_values: [Vec<F>; 2] = [Vec::new(), Vec::new()];
        let mut bz_values: [Vec<F>; 2] = [Vec::new(), Vec::new()];
        let mut az_columns: [Vec<F>; 2] = [Vec::new(), Vec::new()];
        let mut bz_columns: [Vec<F>; 2] = [Vec::new(), Vec::new()];
        let mut az_constant = [F::zero(); 2];
        let mut bz_constant = [F::zero(); 2];
        for (index, stream) in [F::zero(), F::one()].into_iter().enumerate() {
            let weights = spartan_outer_row_weights(uniskip_challenge, stream)?;
            let cycles = 1usize << this.log_t;
            let row_form = |rows: &[Vec<F>]| {
                (0..cycles)
                    .map(|t| {
                        weights
                            .iter()
                            .enumerate()
                            .map(|(row, &weight)| weight * rows[row][t])
                            .sum()
                    })
                    .collect::<Vec<F>>()
            };
            az_values[index] = row_form(&this.az_rows);
            bz_values[index] = row_form(&this.bz_rows);
            let weighted = this.matrices.weighted_columns(&weights, &columns)?;
            az_columns[index] = weighted.a;
            bz_columns[index] = weighted.b;
            let constants = this
                .matrices
                .public_column_contributions(&weights, 0, F::one())?;
            az_constant[index] = constants.a;
            bz_constant[index] = constants.b;
        }

        let tau_low = &this.tau[..=this.log_t];
        Ok(Box::new(OuterRemainderKernel {
            log_t: this.log_t,
            kernel,
            eq_cycle: EqPolynomial::new(tau_low[..this.log_t].to_vec()).evaluations(),
            tau_stream: tau_low[this.log_t],
            tau: this.tau,
            az_values,
            bz_values,
            az_columns,
            bz_columns,
            az_constant,
            bz_constant,
            input_tables: this.input_tables,
            inner: None,
        }))
    }
}

/// The remainder member: a hand-computed stream round followed by a naive
/// member over the cycle rounds.
///
/// The remainder summand is
/// `LK(τ_high, r₀) · eq(τ_low, (t,s)) · Az(s,t) · Bz(s,t)` with
/// `Az(s,t) = (1−s)·Az₀(t) + s·Az₁(t)` — degree 3 in the stream variable
/// `s`, so the expanded `Expr`'s coefficient leaves are quadratic in `s` and
/// have no faithful multilinear table. Round 0 therefore evaluates the
/// factored form directly at 4 sample points; once `c₀` binds `s`, every
/// remaining leaf IS multilinear over the cycle domain and the tail is a
/// [`NaiveSumcheckProver`] whose relation carries a shrunk cycle geometry
/// (so its round count equals the remaining rounds).
pub struct OuterRemainderKernel<F: Field> {
    log_t: usize,
    kernel: F,
    tau: Vec<F>,
    /// eq(τ_low[..log_t], ·) over the cycle domain.
    eq_cycle: Vec<F>,
    /// τ_low's last entry — the stream variable's eq weight.
    tau_stream: F,
    /// Per-stream row-form value tables `Az_s(t)`, `Bz_s(t)`.
    az_values: [Vec<F>; 2],
    bz_values: [Vec<F>; 2],
    /// Per-stream linear forms over the openings (for the bound tail).
    az_columns: [Vec<F>; 2],
    bz_columns: [Vec<F>; 2],
    az_constant: [F; 2],
    bz_constant: [F; 2],
    input_tables: Vec<Vec<F>>,
    inner: Option<NaiveSumcheckProver<F, OuterRemainder<F>>>,
}

impl<F: Field> OuterRemainderKernel<F> {
    /// The residual naive member once the stream challenge is bound: openings
    /// are the plain cycle tables; each coefficient leaf is the SCALAR
    /// `kernel · eq₁(τ_stream, c₀) · az(c₀) · bz(c₀)` times the shared
    /// eq-cycle table — multilinear per leaf, exactly the expanded `Expr`.
    fn build_inner(&mut self, stream_challenge: F) -> Result<(), KernelError<F>> {
        let interpolate = |pair: &[F; 2]| pair[0] + stream_challenge * (pair[1] - pair[0]);
        let column = |columns: &[Vec<F>; 2], index: usize| {
            interpolate(&[columns[0][index], columns[1][index]])
        };
        let scale = self.kernel
            * (self.tau_stream * stream_challenge
                + (F::one() - self.tau_stream) * (F::one() - stream_challenge));
        let az_constant = interpolate(&self.az_constant);
        let bz_constant = interpolate(&self.bz_constant);

        let dimensions = SpartanOuterDimensions::rv64(self.log_t);
        let coefficient_table = |value: F| {
            Polynomial::new(
                self.eq_cycle
                    .iter()
                    .map(|&eq| eq * value)
                    .collect::<Vec<F>>(),
            )
        };
        let variable_count = self.input_tables.len();
        let mut derived_tables = BTreeMap::new();
        for left in 0..variable_count {
            let az_left = column(&self.az_columns, left);
            for right in 0..variable_count {
                let _ = derived_tables.insert(
                    JoltDerivedId::from(SpartanOuterPublic::QuadraticCoefficient { left, right }),
                    coefficient_table(scale * az_left * column(&self.bz_columns, right)),
                );
            }
        }
        for index in 0..variable_count {
            let _ = derived_tables.insert(
                JoltDerivedId::from(SpartanOuterPublic::LinearCoefficient(index)),
                coefficient_table(
                    scale
                        * (column(&self.az_columns, index) * bz_constant
                            + az_constant * column(&self.bz_columns, index)),
                ),
            );
        }
        let _ = derived_tables.insert(
            JoltDerivedId::from(SpartanOuterPublic::ConstantCoefficient),
            coefficient_table(scale * az_constant * bz_constant),
        );

        let opening_tables: BTreeMap<JoltOpeningId, Polynomial<F>> = dimensions
            .variables()
            .iter()
            .zip(&self.input_tables)
            .map(|(&variable, table)| (outer_opening(variable), Polynomial::new(table.clone())))
            .collect();

        // The inner relation's only jobs are round count, degree, and the
        // output expression — a shrunk cycle geometry makes its round count
        // equal the remaining rounds; its tau is never evaluated.
        let inner_dimensions = SpartanOuterDimensions::rv64(self.log_t - 1);
        let inner_tau = self.tau[..=self.log_t].to_vec();
        let inner = NaiveSumcheckProver::new(
            OuterRemainder::new(inner_dimensions, inner_tau, stream_challenge),
            &NoChallenges::default(),
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?;
        self.inner = Some(inner);
        Ok(())
    }
}

impl<F: Field> OuterRemainderInstance<F> for OuterRemainderKernel<F> {
    /// The member's typed produced-opening values, once fully bound.
    fn output_claims(
        &mut self,
    ) -> Result<SumcheckOutputClaims<F, OuterRemainder<F>>, KernelError<F>> {
        self.inner
            .as_mut()
            .ok_or(KernelError::NotFullyBound {
                remaining: self.log_t + 1,
            })?
            .output_claims()
    }
}

impl<F: Field> ProveRounds<F> for OuterRemainderKernel<F> {
    fn num_rounds(&self) -> usize {
        1 + self.log_t
    }

    fn compute_message(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if round > 0 {
            return self
                .inner
                .as_mut()
                .ok_or(SumcheckError::MissingEvaluationSource {
                    kind: "outer-remainder inner naive member",
                })?
                .compute_message(round - 1, previous_claim);
        }

        // The stream round, from the factored form: degree 3 in `s`
        // (eq linear × Az linear × Bz linear), sampled at s = 0..=3.
        let cycles = 1usize << self.log_t;
        let mut evals = Vec::with_capacity(4);
        for sample in 0..4u64 {
            let s = F::from_u64(sample);
            let eq_stream = self.tau_stream * s + (F::one() - self.tau_stream) * (F::one() - s);
            let mut sum = F::zero();
            for t in 0..cycles {
                let az = self.az_values[0][t] + s * (self.az_values[1][t] - self.az_values[0][t]);
                let bz = self.bz_values[0][t] + s * (self.bz_values[1][t] - self.bz_values[0][t]);
                sum += self.eq_cycle[t] * az * bz;
            }
            evals.push(self.kernel * eq_stream * sum);
        }
        let round_sum = evals[0] + evals[1];
        if round_sum != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: round_sum,
            });
        }
        Ok(UnivariatePoly::from_evals(&evals))
    }

    fn ingest_challenge(&mut self, challenge: F, round: usize) -> Result<(), SumcheckError<F>> {
        if round == 0 {
            return self.build_inner(challenge).map_err(|_| {
                SumcheckError::MissingEvaluationSource {
                    kind: "outer-remainder inner naive member",
                }
            });
        }
        self.inner
            .as_mut()
            .ok_or(SumcheckError::MissingEvaluationSource {
                kind: "outer-remainder inner naive member",
            })?
            .ingest_challenge(challenge, round - 1)
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
        .map(|&variable| {
            let oracle = jolt_opening_oracle_ref(outer_opening(variable))?;
            let view = witness.oracle_view(ViewRequirement {
                oracle,
                encoding: PolynomialEncoding::Dense,
                materialization: MaterializationPolicy::BackendChoice,
                retention: RetentionHint::Ephemeral,
            })?;
            let values = view
                .as_slice()
                .ok_or(KernelError::Unsupported {
                    reason: "Spartan-outer input oracle view was not materialized",
                })?
                .to_vec();
            Ok(values)
        })
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
