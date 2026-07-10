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

use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6a::booleanity::{
    BooleanityAddressPhase, BooleanityAddressPhaseOutputClaims,
};
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::views::{dense_view, eq_table};
use crate::{KernelError, ProofSession, ProveSumcheck, ReferenceBackend};

/// The stage-6a booleanity address-phase slot. `reference_address` and
/// `reference_cycle` are the little-endian reference points carried in
/// `Stage6aCarriedChallenges`; `gamma` is the booleanity batching challenge.
pub trait BooleanityAddressProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: BooleanityDimensions,
        reference_address: &[F],
        reference_cycle: &[F],
        gamma: F,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = BooleanityAddressPhase<F>>>, KernelError<F>>;
}

impl<F: Field> BooleanityAddressProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        dimensions: BooleanityDimensions,
        reference_address: &[F],
        reference_cycle: &[F],
        gamma: F,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = BooleanityAddressPhase<F>>>, KernelError<F>>
    {
        Ok(Box::new(BooleanityAddressKernel::new(
            dimensions,
            reference_address,
            reference_cycle,
            gamma,
            witness,
        )?))
    }
}

pub struct BooleanityAddressKernel<F: Field> {
    relation: BooleanityAddressPhase<F>,
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
            return Err(KernelError::Unsupported {
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
            relation: BooleanityAddressPhase::new(dimensions),
            gamma_weights,
            linear,
            squared,
            eq_address: Polynomial::new(eq_table(reference_address)),
            rounds_bound: 0,
        })
    }
}

impl<F: Field> ProveRounds<F> for BooleanityAddressKernel<F> {
    fn num_rounds(&self) -> usize {
        self.relation.symbolic().rounds()
    }

    fn compute_message(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
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

    fn ingest_challenge(&mut self, challenge: F, _round: usize) -> Result<(), SumcheckError<F>> {
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
        Ok(())
    }
}

impl<F: Field> ProveSumcheck<F> for BooleanityAddressKernel<F> {
    type Relation = BooleanityAddressPhase<F>;

    fn relation(&self) -> &BooleanityAddressPhase<F> {
        &self.relation
    }

    fn output_claims(&mut self) -> Result<BooleanityAddressPhaseOutputClaims<F>, KernelError<F>> {
        if self.rounds_bound != self.num_rounds() {
            return Err(KernelError::NotFullyBound {
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
