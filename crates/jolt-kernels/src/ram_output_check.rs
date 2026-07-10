//! The RAM output-check (stage 2) kernel: a hand-rolled member over the
//! address domain with a zero input claim.
//!
//! The summand is `eq(r_address, k) · mask(k) · (val_final(k) − val_io(k))`
//! with `mask` the `[io_start, io_end)` indicator and `val_io` the committed
//! public-IO words (zero outside the segments). This member is NOT naive:
//! the relation's derived leaves (`EqIoMask = eq·mask`,
//! `NegEqIoMaskValIo = −eq·mask·val_io`) are defined by `derive_output_term`
//! as PRODUCTS of multilinears — not multilinear themselves (hence the
//! relation's degree 3) — so no single table reproduces them under
//! multilinear binding. The kernel therefore evaluates the factored form
//! directly over four multilinear tables, degree 3 per round, `LowToHigh`.
//!
//! Only the default read-write config is supported (phase 1 = all cycle
//! rounds), where the relation's rounds equal `log_K`. The legacy prover's
//! leading zero-address rounds emit 1-coefficient constant polynomials whose
//! true round polynomial is zero — this kernel computes those zeros
//! literally, and the engine's batched-polynomial trim reproduces the wire
//! lengths.

use jolt_claims::protocols::jolt::geometry::ram::ram_val_final;
use jolt_claims::protocols::jolt::ReadWriteDimensions;
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_program::preprocess::PublicIoMemory;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::SumcheckOutputClaims;
use jolt_verifier::stages::stage2::ram_output_check::{RamOutputCheck, RamOutputCheckOutputClaims};
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::views::{dense_view, eq_table};
use crate::{KernelError, ProofSession, ProveSumcheck, ReferenceBackend};

/// The stage-2 RAM output-check slot.
pub trait RamOutputCheckProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: ReadWriteDimensions,
        ram_log_k: usize,
        output_address_challenges: &[F],
        public_memory: PublicIoMemory,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = RamOutputCheck<F>>>, KernelError<F>>;
}

impl<F: Field> RamOutputCheckProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        dimensions: ReadWriteDimensions,
        ram_log_k: usize,
        output_address_challenges: &[F],
        public_memory: PublicIoMemory,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = RamOutputCheck<F>>>, KernelError<F>> {
        if dimensions.output_check_rounds() != ram_log_k {
            return Err(KernelError::Unsupported {
                reason: "reference RAM output check supports only the default read-write config \
                         (phase 1 = all cycle rounds)",
            });
        }

        let addresses = 1usize << ram_log_k;
        let mut val_io = vec![F::zero(); addresses];
        for segment in &public_memory.segments {
            for (offset, &word) in segment.words.iter().enumerate() {
                let index = segment.start_index as usize + offset;
                if index < addresses {
                    val_io[index] = F::from_u64(word);
                }
            }
        }
        let mask: Vec<F> = (0..addresses)
            .map(|k| {
                let in_io_region = (k as u128) >= public_memory.io_mask_start
                    && (k as u128) < public_memory.io_mask_end;
                if in_io_region {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect();
        let val_final = dense_view(witness, ram_val_final())?;

        let relation = RamOutputCheck::new(
            dimensions,
            output_address_challenges.to_vec(),
            public_memory,
        );
        Ok(Box::new(RamOutputCheckKernel {
            relation,
            rounds: ram_log_k,
            rounds_bound: 0,
            eq_address: Polynomial::new(eq_table(output_address_challenges)),
            mask: Polynomial::new(mask),
            val_final: Polynomial::new(val_final),
            val_io: Polynomial::new(val_io),
        }))
    }
}

/// See the module docs: the factored four-table member.
struct RamOutputCheckKernel<F: Field> {
    relation: RamOutputCheck<F>,
    rounds: usize,
    rounds_bound: usize,
    eq_address: Polynomial<F>,
    mask: Polynomial<F>,
    val_final: Polynomial<F>,
    val_io: Polynomial<F>,
}

impl<F: Field> ProveRounds<F> for RamOutputCheckKernel<F> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn compute_message(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let half = (1usize << (self.rounds - self.rounds_bound)) / 2;
        let order = BindingOrder::LowToHigh;
        let mut evals = Vec::with_capacity(4);
        for t in 0..4u64 {
            let point = F::from_u64(t);
            let mut sum = F::zero();
            for y in 0..half {
                let eq = self
                    .eq_address
                    .sumcheck_round_eval_with_order(y, point, order);
                let mask = self.mask.sumcheck_round_eval_with_order(y, point, order);
                let val_final = self
                    .val_final
                    .sumcheck_round_eval_with_order(y, point, order);
                let val_io = self.val_io.sumcheck_round_eval_with_order(y, point, order);
                sum += eq * mask * (val_final - val_io);
            }
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

    fn ingest_challenge(&mut self, challenge: F, _round: usize) -> Result<(), SumcheckError<F>> {
        for table in [
            &mut self.eq_address,
            &mut self.mask,
            &mut self.val_final,
            &mut self.val_io,
        ] {
            table.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        self.rounds_bound += 1;
        Ok(())
    }
}

impl<F: Field> ProveSumcheck<F> for RamOutputCheckKernel<F> {
    type Relation = RamOutputCheck<F>;

    fn relation(&self) -> &RamOutputCheck<F> {
        &self.relation
    }

    fn output_claims(
        &mut self,
    ) -> Result<SumcheckOutputClaims<F, RamOutputCheck<F>>, KernelError<F>> {
        if self.rounds_bound != self.rounds {
            return Err(KernelError::NotFullyBound {
                remaining: self.rounds - self.rounds_bound,
            });
        }
        Ok(RamOutputCheckOutputClaims {
            val_final: self.val_final.evals()[0],
        })
    }
}
