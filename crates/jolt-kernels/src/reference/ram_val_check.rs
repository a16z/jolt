//! The RAM value-check (stage 4) kernel: a naive member over the cycle
//! domain.
//!
//! The summand is `inc(j) · ra(j) · (LT(j, r_cycle) + γ)` — batching the
//! "value delta at `r_cycle`" and "final value delta" identities by `γ`. The
//! `LtCyclePlusGamma` derived leaf is ONE multilinear table
//! (`LtPolynomial::evaluations(r_cycle)[j] + γ` — γ is a drawn challenge, so
//! the table is built after `draw_challenges`), and the `ra` opening is the
//! address-bound slice of the `(K × T)` RAM `ra` grid:
//! `ra(j) = Σ_k eq(r_address, k) · RamRa(k, j)` — an opening-side fold the
//! kernel performs, including READ accesses (their `inc` is zero but their
//! `ra` weight is not).

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::ram::{ram_inc_val_check, ram_ra_val_check};
use jolt_claims::protocols::jolt::{JoltDerivedId, RamValCheckPublic};
use jolt_field::Field;
use jolt_poly::{BindingOrder, LtPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ProverInputs;
use jolt_verifier::stages::stage4::ram_val_check::{
    RamValCheck, RamValCheckChallenges, RamValCheckInputClaims, RamValCheckOutputClaims,
};
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use super::views::{address_fold, dense_view};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend,
    SumcheckKernel, SumcheckKernelError,
};

impl<F: Field> PrepareKernel<F, RamValCheck<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltVmWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamValCheck<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamValCheck<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let trace_dimensions = relation.trace_dimensions();
        // The consumed `ram_val` point is stage 2's RAM read-write opening
        // point: `r_address` (log_k vars) followed by `r_cycle` (log_t vars).
        let ram_val_point: &[F] = &inputs.points.ram_val;
        if ram_val_point.len() != relation.ram_log_k() + trace_dimensions.log_t() {
            return Err(KernelError::InvariantViolation {
                reason: "RAM value-check input point has the wrong variable count",
            });
        }
        let (r_address, r_cycle) = ram_val_point.split_at(relation.ram_log_k());
        let challenges = inputs.challenges;
        // The address-bound `ra` slice, folded from the full `(K × T)` grid.
        let ra_folded = address_fold(
            witness,
            ram_ra_val_check(),
            trace_dimensions.log_t(),
            r_address,
        )?;

        let lt_plus_gamma: Vec<F> = LtPolynomial::evaluations(r_cycle)
            .into_iter()
            .map(|lt| lt + challenges.gamma)
            .collect();

        let opening_tables = BTreeMap::from([
            (ram_ra_val_check(), Polynomial::new(ra_folded)),
            (
                ram_inc_val_check(),
                Polynomial::new(dense_view(witness, ram_inc_val_check())?),
            ),
        ]);
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(RamValCheckPublic::LtCyclePlusGamma),
            Polynomial::new(lt_plus_gamma),
        )]);

        Ok(Box::new(RamValCheckKernel {
            inner: NaiveSumcheckProver::new(
                relation,
                challenges,
                opening_tables,
                derived_tables,
                BindingOrder::LowToHigh,
            )?,
            untrusted_advice: inputs.claims.untrusted_advice,
            trusted_advice: inputs.claims.trusted_advice,
            program_image: inputs.claims.program_image,
        }))
    }
}

/// The naive kernel plus the staged `Val_init` openings. The advice /
/// program-image wire claims are not `Expr` leaves (the naive prover has no
/// tables for them), and their values equal the member's own consumed input
/// claims — the dual-role openings, see the relation's module docs — so
/// `prepare` captures them off the `ProverInputs` bundle and extraction
/// re-attaches them to the typed output claims.
struct RamValCheckKernel<F: Field> {
    inner: NaiveSumcheckProver<F, RamValCheck<F>>,
    untrusted_advice: Option<F>,
    trusted_advice: Option<F>,
    program_image: Option<F>,
}

impl<F: Field> ProveRounds<F> for RamValCheckKernel<F> {
    fn num_rounds(&self) -> usize {
        self.inner.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        self.inner.prove_round(bind, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.inner.finish_rounds(bind)
    }
}

impl<F: Field> SumcheckKernel<F> for RamValCheckKernel<F> {
    type Relation = RamValCheck<F>;

    fn output_claims(&mut self) -> Result<RamValCheckOutputClaims<F>, SumcheckKernelError<F>> {
        let mut claims = self.inner.output_claims()?;
        claims.untrusted_advice = self.untrusted_advice;
        claims.trusted_advice = self.trusted_advice;
        claims.program_image = self.program_image;
        Ok(claims)
    }

    fn validate_derived_tables(
        &self,
        relation: &RamValCheck<F>,
        input_points: &RamValCheckInputClaims<Vec<F>>,
        output_points: &RamValCheckOutputClaims<Vec<F>>,
        challenges: &RamValCheckChallenges<F>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.inner
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}
