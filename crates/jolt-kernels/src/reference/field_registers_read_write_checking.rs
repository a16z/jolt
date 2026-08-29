//! The stage-4 `FieldRegistersReadWriteChecking` kernel: a hand-rolled member
//! over the joint `(field-register ‖ cycle)` domain.
//!
//! The summand
//! `eq(r_prod, j) · (rd_wa·(rd_inc + val) + γ·rs1_ra·val + γ²·rs2_ra·val)(k, j)`
//! is the jolt registers read/write kernel's structure at the FR dimensions:
//! dense tables of size `2^(4 + log_T)` in register-major layout
//! (`index = k·2^log_T + j`, the FR witness oracle's address-major grid),
//! bound `LowToHigh`. The config-pinned FR phase split (phase 1 = all cycle
//! rounds, phase 2 = the 4 address rounds) binds the cycle variables first,
//! exactly as `FieldRegistersReadWriteDimensions::read_write_opening_point`
//! derives the `[address ‖ cycle]` opening point. The cycle-indexed `rd_inc`
//! and eq tables are tiled across the register dimension. The FieldInline id
//! family cannot ride the jolt-keyed
//! [`NaiveSumcheckProver`](crate::NaiveSumcheckProver), so the tables and the
//! expression are hand-held (the
//! [`field_registers_claim_reduction`](super::field_registers_claim_reduction)
//! pattern). The reference tier stays dense — `K = 16` makes the full grids
//! `16·T` — with the sparse ≤3-entries-per-active-cycle replay walk left to
//! an optimized tier.

#[cfg(feature = "allocative")]
use allocative::{Allocative, Key, Visitor};
use jolt_claims::protocols::field_inline::{FieldInlineDerivedId, FieldRegistersReadWritePublic};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage4::field_registers_read_write_checking::FieldRegistersReadWriteChecking;
use jolt_verifier::VerifierError;
use jolt_witness::JoltWitnessPlane;

use super::views::{eq_table, tile};
use crate::backend::{PrepareKernel, ProofSession};
use crate::kernel::{ProverInputs, SumcheckKernel};
use crate::reference::ReferenceBackend;
use crate::{KernelError, SumcheckKernelError};

impl<F: JoltField> PrepareKernel<F, FieldRegistersReadWriteChecking<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, FieldRegistersReadWriteChecking<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = FieldRegistersReadWriteChecking<F>>>,
        KernelError<F>,
    > {
        use jolt_claims::protocols::field_inline::{
            FieldInlineChallengeId, FieldInlineCommittedPolynomial, FieldInlinePolynomialId,
            FieldInlineVirtualPolynomial, FieldRegistersReadWriteChallenge,
        };
        use jolt_claims::SumcheckChallenges as _;
        use jolt_witness::WitnessError;

        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        // The FR phase split is pinned by the compile-time protocol config
        // (phase 1 = log_t, phase 2 = log_k); this kernel's binding order
        // depends on it, so a drifted config is a bug, not a capability gap.
        if dimensions.phase1_num_rounds() != dimensions.log_t()
            || dimensions.phase2_num_rounds() != dimensions.log_k()
        {
            return Err(KernelError::InvariantViolation {
                reason: "FR read-write dimensions drifted from the config-pinned phase split",
            });
        }
        let r_cycle: &[F] = &inputs.points.rd_value;
        if r_cycle.len() != dimensions.log_t() {
            return Err(KernelError::InvariantViolation {
                reason: "FR read-write upstream cycle point has the wrong variable count",
            });
        }

        let field_inline =
            witness
                .field_inline()
                .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                    label: "field-registers read-write checking field-inline oracle",
                }))?;
        let grid = |polynomial: FieldInlineVirtualPolynomial| {
            field_inline
                .oracle_table(FieldInlinePolynomialId::Virtual(polynomial))
                .map(Polynomial::new)
        };
        let rd_inc = field_inline.oracle_table(FieldInlinePolynomialId::Committed(
            FieldInlineCommittedPolynomial::FieldRdInc,
        ))?;
        let gamma = inputs
            .challenges
            .resolve_challenge(&FieldInlineChallengeId::from(
                FieldRegistersReadWriteChallenge::Gamma,
            ))
            .ok_or(KernelError::InvariantViolation {
                reason: "FR read-write checking is missing its gamma challenge",
            })?;

        let copies = 1usize << dimensions.log_k();
        Ok(Box::new(FieldRegistersReadWriteKernel {
            relation: relation.clone(),
            gamma,
            eq_cycle: Polynomial::new(tile(&eq_table(r_cycle), copies)),
            registers_val: grid(FieldInlineVirtualPolynomial::FieldRegistersVal)?,
            rs1_ra: grid(FieldInlineVirtualPolynomial::FieldRs1Ra)?,
            rs2_ra: grid(FieldInlineVirtualPolynomial::FieldRs2Ra)?,
            rd_wa: grid(FieldInlineVirtualPolynomial::FieldRdWa)?,
            rd_inc: Polynomial::new(tile(&rd_inc, copies)),
            rounds_bound: 0,
        }))
    }
}

struct FieldRegistersReadWriteKernel<F: JoltField> {
    relation: FieldRegistersReadWriteChecking<F>,
    gamma: F,
    /// `eq(r_prod, ·)` over the cycle domain, tiled across the FR address
    /// dimension (big-endian, like the jolt registers read/write kernel's
    /// `EqCycle` table).
    eq_cycle: Polynomial<F>,
    registers_val: Polynomial<F>,
    rs1_ra: Polynomial<F>,
    rs2_ra: Polynomial<F>,
    rd_wa: Polynomial<F>,
    rd_inc: Polynomial<F>,
    rounds_bound: usize,
}

// Size arithmetic rather than a derive, like the sibling kernels.
#[cfg(feature = "allocative")]
impl<F: JoltField> Allocative for FieldRegistersReadWriteKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        for (key, table) in [
            (Key::new("eq_cycle"), &self.eq_cycle),
            (Key::new("registers_val"), &self.registers_val),
            (Key::new("rs1_ra"), &self.rs1_ra),
            (Key::new("rs2_ra"), &self.rs2_ra),
            (Key::new("rd_wa"), &self.rd_wa),
            (Key::new("rd_inc"), &self.rd_inc),
        ] {
            visitor.visit_simple(key, table.len() * size_of::<F>());
        }
        visitor.exit();
    }
}

impl<F: JoltField> FieldRegistersReadWriteKernel<F> {
    fn remaining_rounds(&self) -> usize {
        self.relation.rounds() - self.rounds_bound
    }

    fn bind_tables(&mut self, challenge: F) {
        for table in [
            &mut self.eq_cycle,
            &mut self.registers_val,
            &mut self.rs1_ra,
            &mut self.rs2_ra,
            &mut self.rd_wa,
            &mut self.rd_inc,
        ] {
            table.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        self.rounds_bound += 1;
    }

    fn require_fully_bound(&self) -> Result<(), SumcheckKernelError<F>> {
        match self.remaining_rounds() {
            0 => Ok(()),
            remaining => Err(SumcheckKernelError::NotFullyBound { remaining }),
        }
    }
}

impl<F: JoltField> ProveRounds<F> for FieldRegistersReadWriteKernel<F> {
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind_tables(challenge);
        }
        let half = (1usize << self.remaining_rounds()) / 2;
        let degree = self.relation.degree();
        let order = BindingOrder::LowToHigh;
        let gamma = self.gamma;
        let mut evals = Vec::with_capacity(degree + 1);
        for sample in 0..=degree {
            let point = F::from_u64(sample as u64);
            let sum = (0..half)
                .map(|y| {
                    let ext = |table: &Polynomial<F>| {
                        table.sumcheck_round_eval_with_order(y, point, order)
                    };
                    let val = ext(&self.registers_val);
                    let access = ext(&self.rd_wa) * (ext(&self.rd_inc) + val)
                        + gamma * ext(&self.rs1_ra) * val
                        + gamma * gamma * ext(&self.rs2_ra) * val;
                    ext(&self.eq_cycle) * access
                })
                .sum::<F>();
            evals.push(sum);
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
        self.bind_tables(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for FieldRegistersReadWriteKernel<F> {
    type Relation = FieldRegistersReadWriteChecking<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, FieldRegistersReadWriteChecking<F>>,
    ) -> Result<SumcheckOutputClaims<F, FieldRegistersReadWriteChecking<F>>, SumcheckKernelError<F>>
    {
        use jolt_claims::protocols::field_inline::relations::registers::FieldRegistersReadWriteOutputClaims;

        self.require_fully_bound()?;
        Ok(FieldRegistersReadWriteOutputClaims {
            registers_val: self.registers_val.evals()[0],
            rs1_ra: self.rs1_ra.evals()[0],
            rs2_ra: self.rs2_ra.evals()[0],
            rd_wa: self.rd_wa.evals()[0],
            rd_inc: self.rd_inc.evals()[0],
        })
    }

    /// The `EqCycle` cross-check: the bound tiled eq table's final value must
    /// equal the verifier's `derive_output_term` at the bound point (the same
    /// tie-down the naive tier performs for jolt-family members).
    fn validate_derived_tables(
        &self,
        relation: &FieldRegistersReadWriteChecking<F>,
        input_points: &SumcheckInputPoints<F, FieldRegistersReadWriteChecking<F>>,
        output_points: &SumcheckOutputPoints<F, FieldRegistersReadWriteChecking<F>>,
        challenges: &ConcreteSumcheckChallenges<F, FieldRegistersReadWriteChecking<F>>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.require_fully_bound()?;
        let expected = relation.derive_output_term(
            &FieldInlineDerivedId::from(FieldRegistersReadWritePublic::EqCycle),
            input_points,
            output_points,
            challenges,
        )?;
        let got = self.eq_cycle.evals()[0];
        if got != expected {
            return Err(SumcheckKernelError::Verifier(
                VerifierError::StageClaimSumcheckFailed {
                    stage: "FieldRegistersReadWriteChecking".to_string(),
                    reason: format!(
                        "EqCycle table bound to {got:?}, but derive_output_term gives {expected:?}"
                    ),
                },
            ));
        }
        Ok(())
    }
}
