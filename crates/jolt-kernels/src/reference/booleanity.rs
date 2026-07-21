//! The booleanity (stage 6a) address-phase kernel — a hand kernel: the
//! relation's output `Expr` is the bare staged `BooleanityAddrClaim`
//! intermediate, hiding the true summand
//!
//! `0 = Σ_{k,j} eq(r_ref_addr, k) · eq(r_ref_cycle, j) · Σ_i γ^{2i} · (ra_i(k,j)² − ra_i(k,j))`
//!
//! from the naive interpreter (a squared one-hot is not a bind of any single
//! table). The address phase binds the `log_k_chunk` chunk variables `k`.
//! Because each `ra_i(·, j)` is one-hot in `k`, the cycle dimension collapses
//! to per-chunk masses `m_i[k] = Σ_j eq(r_ref_cycle, j) · ra_i(k, j)`, and the
//! two inner terms become two tables over the `2^log_k_chunk`-point chunk
//! domain (16 in the default sub-threshold config):
//!
//! - the linear term binds `A_i[k] = m_i[k]` as a plain multilinear;
//! - the squared term binds `B_i[k]` (same initial masses) with *squared*
//!   weights — `B'[k] = (1−r)²·B[2k] + r²·B[2k+1]` — because binding squares
//!   the one-hot's accumulated eq factor.
//!
//! The round polynomial `eqₑₓₜ(X) · Σ_i γ^{2i}(Bₑₓₜ(X) − Aₑₓₜ(X))` with
//! `Bₑₓₜ(X) = (1−X)²·B[2y] + X²·B[2y+1]` is the true cubic, sampled at four
//! points. The initial `A = B` makes the input claim exactly zero.

use std::collections::BTreeMap;

use crate::ProverInputs;
use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::{BooleanityPublic, JoltDerivedId, JoltRelationId};
use jolt_field::Field;
use jolt_poly::{try_eq_mle, BindingOrder, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6a::booleanity::{
    BooleanityAddressPhase, BooleanityAddressPhaseOutputClaims,
};
use jolt_verifier::stages::stage6b::booleanity::Booleanity;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;
use jolt_witness::WitnessProvider;

use super::views::{address_fold, dense_view, eq_table};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend,
    SumcheckKernel, SumcheckKernelError,
};

impl<F: Field> PrepareKernel<F, BooleanityAddressPhase<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltVmWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BooleanityAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BooleanityAddressPhase<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let draws = relation
            .reference_draws()
            .ok_or(KernelError::InvariantViolation {
                reason: "booleanity address phase prepared before its reference draws were set",
            })?;
        Ok(Box::new(BooleanityAddressKernel::new(
            relation,
            relation.dimensions(),
            &draws.reference_address,
            &draws.reference_cycle,
            draws.gamma,
            witness,
        )?))
    }
}

pub struct BooleanityAddressKernel<F: Field> {
    rounds: usize,
    /// Per checked polynomial, its `γ^{2i}` batching weight, in the layout's
    /// canonical order.
    gamma_weights: Vec<F>,
    /// The linear-term tables (plain multilinear binding).
    linear: Vec<Polynomial<F>>,
    /// The squared-term tables (squared-weight binding); raw vectors because
    /// the bind rule is not a multilinear bind.
    squared: Vec<Vec<F>>,
    eq_address: Polynomial<F>,
    rounds_bound: usize,
}

impl<F: Field> BooleanityAddressKernel<F> {
    pub fn new(
        relation: &BooleanityAddressPhase<F>,
        dimensions: BooleanityDimensions,
        reference_address: &[F],
        reference_cycle: &[F],
        gamma: F,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Self, KernelError<F>> {
        let log_k_chunk = dimensions.log_k_chunk;
        let log_t = dimensions.log_t;
        let chunk_size = 1usize << log_k_chunk;
        let cycles = 1usize << log_t;
        if reference_address.len() != log_k_chunk || reference_cycle.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "booleanity reference point lengths disagree with the dimensions",
            });
        }

        // The carried reference points are the little-endian (reversed)
        // vectors, and the legacy prover feeds them to its eq machinery
        // verbatim — the protocol's effective reference point is the
        // bit-reversed one, and the 6b cycle phase and the verifier's
        // `derive_output_term` both follow that convention. Use them as-is.
        let eq_cycle = eq_table(reference_cycle);

        // Per-chunk masses of each checked one-hot polynomial, folded over the
        // cycle dimension by the reference-cycle eq weights.
        let mut linear = Vec::new();
        for opening in dimensions.layout.openings(JoltRelationId::Booleanity) {
            let grid = dense_view(witness, opening)?;
            if grid.len() != chunk_size << log_t {
                return Err(KernelError::TableSizeMismatch {
                    table: format!("{opening:?}"),
                    expected: chunk_size << log_t,
                    got: grid.len(),
                });
            }
            let masses: Vec<F> = (0..chunk_size)
                .map(|k| {
                    (0..cycles)
                        .map(|j| grid[(k << log_t) | j] * eq_cycle[j])
                        .sum()
                })
                .collect();
            linear.push(Polynomial::new(masses));
        }
        let squared: Vec<Vec<F>> = linear.iter().map(|table| table.evals().to_vec()).collect();

        let mut gamma_weights = Vec::with_capacity(linear.len());
        let mut weight = F::one();
        let gamma_sqr = gamma * gamma;
        for _ in 0..linear.len() {
            gamma_weights.push(weight);
            weight *= gamma_sqr;
        }

        Ok(Self {
            rounds: relation.rounds(),
            gamma_weights,
            linear,
            squared,
            eq_address: Polynomial::new(eq_table(reference_address)),
            rounds_bound: 0,
        })
    }
}

impl<F: Field> BooleanityAddressKernel<F> {
    fn bind(&mut self, challenge: F) {
        let one_minus_sqr = (F::one() - challenge) * (F::one() - challenge);
        let challenge_sqr = challenge * challenge;
        for table in &mut self.linear {
            table.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        for table in &mut self.squared {
            let half = table.len() / 2;
            for k in 0..half {
                table[k] = one_minus_sqr * table[2 * k] + challenge_sqr * table[2 * k + 1];
            }
            table.truncate(half);
        }
        self.eq_address
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.rounds_bound += 1;
    }
}

impl<F: Field> ProveRounds<F> for BooleanityAddressKernel<F> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.eq_address.evals().len() / 2;
        let mut evals = [F::zero(); 4];
        for (c, eval) in evals.iter_mut().enumerate() {
            let point = F::from_u64(c as u64);
            let point_sqr = point * point;
            let one_minus_sqr = (F::one() - point) * (F::one() - point);
            let mut sum = F::zero();
            for y in 0..half {
                let mut inner = F::zero();
                for ((weight, linear), squared) in self
                    .gamma_weights
                    .iter()
                    .zip(&self.linear)
                    .zip(&self.squared)
                {
                    let squared_ext =
                        one_minus_sqr * squared[2 * y] + point_sqr * squared[2 * y + 1];
                    let linear_ext =
                        linear.sumcheck_round_eval_with_order(y, point, BindingOrder::LowToHigh);
                    inner += *weight * (squared_ext - linear_ext);
                }
                sum += self.eq_address.sumcheck_round_eval_with_order(
                    y,
                    point,
                    BindingOrder::LowToHigh,
                ) * inner;
            }
            *eval = sum;
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

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: Field> SumcheckKernel<F> for BooleanityAddressKernel<F> {
    type Relation = BooleanityAddressPhase<F>;

    fn output_claims(
        &mut self,
    ) -> Result<BooleanityAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        if self.rounds_bound != self.num_rounds() {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.num_rounds() - self.rounds_bound,
            });
        }
        let mut inner = F::zero();
        for ((weight, linear), squared) in self
            .gamma_weights
            .iter()
            .zip(&self.linear)
            .zip(&self.squared)
        {
            inner += *weight * (squared[0] - linear.evals()[0]);
        }
        Ok(BooleanityAddressPhaseOutputClaims {
            intermediate: self.eq_address.evals()[0] * inner,
        })
    }
}

impl<F: Field> PrepareKernel<F, Booleanity<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltVmWitnessPlane<F>,
        inputs: ProverInputs<'_, F, Booleanity<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = Booleanity<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let r_address = relation.r_address();
        let reference_address = relation.reference_address();
        let reference_cycle = relation.reference_cycle();

        let mut opening_tables = BTreeMap::new();
        for opening in dimensions.layout.openings(JoltRelationId::Booleanity) {
            let _ = opening_tables.insert(
                opening,
                Polynomial::new(address_fold(witness, opening, dimensions.log_t, r_address)?),
            );
        }

        // The fixed address eq factor: both vectors pair positionally in the
        // verifier's `derive_output_term` (each side reversed, so the product
        // is the same either way).
        let address_scalar = try_eq_mle(r_address, reference_address).map_err(|_| {
            KernelError::InvariantViolation {
                reason: "booleanity address point and reference length mismatch",
            }
        })?;
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(BooleanityPublic::EqAddressCycle),
            Polynomial::new(
                eq_table(reference_cycle)
                    .into_iter()
                    .map(|eq| address_scalar * eq)
                    .collect::<Vec<_>>(),
            ),
        )]);

        Ok(Box::new(NaiveSumcheckProver::new(
            relation,
            inputs.challenges,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
