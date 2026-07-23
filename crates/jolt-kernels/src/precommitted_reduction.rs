//! The precommitted claim-reduction family: two independent per-phase batch
//! kernels — [`CycleReductionKernel`] (stage 6b) and
//! [`AddressReductionKernel`] (stage 7) — plus the plain owned
//! [`PrecommittedReductionCarry`] the cycle kernel parks across the batch
//! boundary via [`SumcheckKernel::park_residue`] and the stage-7 slot's
//! `prepare` reclaims by move.
//!
//! Four kinds share the machinery — trusted advice, untrusted advice,
//! committed bytecode, program image — each as a plain set of impls below:
//! the phase kernels are generic over the relation, and each kind contributes
//! one `SumcheckKernel` impl per phase (the typed wire-claim assembly plus,
//! on the cycle side, the carry parking under the paired address relation's
//! key). The table BUILDERS live with the backends (the reference ones in
//! the per-kind `reference::*_claim_reduction` modules), as do the stage-7
//! `PrepareKernel` impls reclaiming the carries
//! ([`reference::precommitted_reduction`](crate::reference::precommitted_reduction)).
//! The scalar kinds resolve intermediate-vs-final through
//! [`CycleReductionKernel::scalar_claim`]; the bytecode kind's chunked shape
//! spells its resolution out in its own impl.
//!
//! The summand in both phases is `Σ_j value(j) · eq(j)` over tables permuted
//! into Dory opening-round order, bound low-to-high; `aux` tables (the
//! per-chunk bytecode polynomials) bind alongside without joining the summand
//! — their fully bound coefficients are the reduction's final per-chunk
//! openings. The member is head-aligned (batch offset 0) and participates
//! only in its schedule's active rounds; on an inactive round inside the
//! window it emits the constant `claim/2` polynomial and halves its running
//! `scale`.
//!
//! Byte-parity subtlety: on active rounds the round polynomial is built from
//! the TRUE evaluations at 0 and 2 plus the hint `s(1) = claim/scale − s(0)`
//! (legacy `UniPoly::from_evals_and_hint`). Under the batch's `2^(max−rounds)`
//! dummy-round padding the incoming claim is the padded one, so the hinted
//! `s(1)` — and through it the emitted quadratic coefficient — carries the
//! padding factor. This is the wire format legacy emits and the verifier's
//! expected-output algebra encodes; reproduce it exactly, do not "fix" it to
//! the unpadded true polynomial.

use std::marker::PhantomData;

use jolt_claims::protocols::jolt::PrecommittedClaimReduction;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::{
    BytecodeReductionCyclePhase, BytecodeReductionCyclePhaseOutputClaims,
    ProgramImageReductionCyclePhase, ProgramImageReductionCyclePhaseOutputClaims,
    TrustedAdviceCyclePhase, TrustedAdviceCyclePhaseOutputClaims, UntrustedAdviceCyclePhase,
    UntrustedAdviceCyclePhaseOutputClaims,
};
use jolt_verifier::stages::stage7::advice_address_phase::{
    TrustedAdviceAddressPhase, TrustedAdviceAddressPhaseOutputClaims, UntrustedAdviceAddressPhase,
    UntrustedAdviceAddressPhaseOutputClaims,
};
use jolt_verifier::stages::stage7::committed_reduction_address_phase::{
    BytecodeReductionAddressPhase, BytecodeReductionAddressPhaseOutputClaims,
    ProgramImageReductionAddressPhase, ProgramImageReductionAddressPhaseOutputClaims,
};

use crate::{KernelError, ProofSession, SumcheckKernel, SumcheckKernelError};

/// The bound-table state both phase kernels drive: the summand tables, the
/// aux tables riding alongside, and the running inactive-round scale.
struct PrecommittedTables<F> {
    value: Vec<F>,
    eq: Vec<F>,
    aux: Vec<Vec<F>>,
    /// `(1/2)^k` over the `k` inactive rounds ingested so far — the factor the
    /// running claim accumulated relative to the true bound product.
    scale: F,
    scale_inv: F,
    two_inv: F,
}

impl<F: Field> PrecommittedTables<F> {
    /// The round polynomial for member-local state: the constant `claim/2` on
    /// an inactive round, else the hinted `{0,1,2}` interpolation (see the
    /// module doc for why the padded claim, not the true sum, feeds `s(1)`).
    fn round_message(&self, active: bool, previous_claim: F) -> UnivariatePoly<F> {
        if !active {
            return UnivariatePoly::new(vec![previous_claim * self.two_inv]);
        }
        let half = self.value.len() / 2;
        let mut eval_0 = F::zero();
        let mut eval_2 = F::zero();
        for j in 0..half {
            let value_0 = self.value[2 * j];
            let value_1 = self.value[2 * j + 1];
            let eq_0 = self.eq[2 * j];
            let eq_1 = self.eq[2 * j + 1];
            eval_0 += value_0 * eq_0;
            eval_2 += (value_1 + value_1 - value_0) * (eq_1 + eq_1 - eq_0);
        }
        let eval_1 = previous_claim * self.scale_inv - eval_0;
        let c2 = (eval_0 - eval_1 - eval_1 + eval_2) * self.two_inv;
        let c1 = eval_1 - eval_0 - c2;
        UnivariatePoly::new(vec![eval_0 * self.scale, c1 * self.scale, c2 * self.scale])
    }

    /// Drive one head-aligned round against the phase's active-round
    /// schedule: bind the pending challenge — it belongs to the previous
    /// round; the member is head-aligned and consulted every round of its
    /// window, so `bind` is `Some` exactly when `round >= 1` — then emit this
    /// round's message.
    fn prove_round(
        &mut self,
        active_rounds: &[usize],
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> UnivariatePoly<F> {
        if let Some(challenge) = bind {
            self.bind_round(is_active(active_rounds, round - 1), challenge);
        }
        self.round_message(is_active(active_rounds, round), previous_claim)
    }

    /// Ingest the final round's challenge for the phase's schedule.
    fn finish_rounds(&mut self, active_rounds: &[usize], total_rounds: usize, bind: F) {
        self.bind_round(is_active(active_rounds, total_rounds - 1), bind);
    }

    /// Bind a round's challenge: on an active round bind every table, on an
    /// inactive one fold the halving into the running `scale` instead.
    fn bind_round(&mut self, active: bool, challenge: F) {
        if !active {
            self.scale *= self.two_inv;
            self.scale_inv += self.scale_inv;
            return;
        }
        let half = self.value.len() / 2;
        let bind = |table: &mut Vec<F>| {
            for j in 0..half {
                table[j] = table[2 * j] + challenge * (table[2 * j + 1] - table[2 * j]);
            }
            table.truncate(half);
        };
        bind(&mut self.value);
        bind(&mut self.eq);
        for table in &mut self.aux {
            bind(table);
        }
    }

    /// The intermediate claim staged at the cycle→address handoff:
    /// `Σ_i value(i) · eq(i) · scale` over the bound tables.
    fn intermediate_claim(&self) -> F {
        let product: F = self
            .value
            .iter()
            .zip(&self.eq)
            .map(|(value, eq)| *value * *eq)
            .sum();
        product * self.scale
    }

    fn require_fully_bound(&self) -> Result<(), SumcheckKernelError<F>> {
        if self.value.len() != 1 {
            return Err(SumcheckKernelError::InvariantViolation {
                reason:
                    "precommitted reduction final claim requested before the polynomial is fully bound",
            });
        }
        Ok(())
    }

    /// The fully bound value coefficient — the reduction's final opening
    /// value (the advice/program-image polynomial's own opening; for the
    /// bytecode reduction, the chunk-weighted fold the per-chunk claims sum
    /// to). Errors while any variable remains unbound.
    fn final_claim(&self) -> Result<F, SumcheckKernelError<F>> {
        self.require_fully_bound()?;
        Ok(self.value[0])
    }

    /// The fully bound `aux` coefficients — the per-chunk `BytecodeChunk(i)`
    /// opening values. Errors while any variable remains unbound.
    fn final_aux_claims(&self) -> Result<Vec<F>, SumcheckKernelError<F>> {
        self.require_fully_bound()?;
        Ok(self.aux.iter().map(|table| table[0]).collect())
    }
}

fn is_active(active_rounds: &[usize], round: usize) -> bool {
    active_rounds.binary_search(&round).is_ok()
}

/// The 6b→7 carry: the post-cycle bound tables and running scale, moved out
/// of the cycle kernel by its `park_residue` and into a fresh
/// [`AddressReductionKernel`] by stage 7's `prepare`. Plain owned data —
/// parked once, reclaimed once — keyed in the [`ProofSession`] by the
/// address-phase relation `R` it becomes.
pub struct PrecommittedReductionCarry<F, R> {
    reduction: PrecommittedClaimReduction,
    tables: PrecommittedTables<F>,
    _relation: PhantomData<fn() -> R>,
}

/// The stage-6b cycle-phase batch member for relation `R`: binds the cycle
/// window, and — per kind — assembles the typed wire claims (resolving
/// intermediate-vs-final from the schedule) and parks the 6b→7 carry.
pub struct CycleReductionKernel<F: Field, R> {
    reduction: PrecommittedClaimReduction,
    tables: PrecommittedTables<F>,
    _relation: PhantomData<fn() -> R>,
}

impl<F: Field, R> CycleReductionKernel<F, R> {
    /// Build a member from tables ALREADY permuted into Dory opening-round
    /// order (see [`lsb_permutation`] / [`permute_coefficients`]).
    pub fn new(
        reduction: PrecommittedClaimReduction,
        value: Vec<F>,
        eq: Vec<F>,
        aux: Vec<Vec<F>>,
    ) -> Result<Self, KernelError<F>> {
        let expected = 1usize << reduction.poly_opening_round_permutation_be().len();
        for (name, len) in std::iter::once(("value", value.len()))
            .chain(std::iter::once(("eq", eq.len())))
            .chain(aux.iter().map(|table| ("aux", table.len())))
        {
            if len != expected {
                return Err(KernelError::TableSizeMismatch {
                    table: format!("precommitted reduction {name}"),
                    expected,
                    got: len,
                });
            }
        }
        #[expect(
            clippy::expect_used,
            reason = "2 is invertible in any Jolt field (large-prime characteristic)"
        )]
        let two_inv = F::from_u64(2).inverse().expect("2 is invertible");
        Ok(Self {
            reduction,
            tables: PrecommittedTables {
                value,
                eq,
                aux,
                scale: F::one(),
                scale_inv: F::one(),
                two_inv,
            },
            _relation: PhantomData,
        })
    }

    fn has_address_phase(&self) -> bool {
        self.reduction.num_address_phase_rounds() > 0
    }

    /// The schedule-resolved scalar wire claim: the intermediate handoff
    /// claim when the address phase continues, else the final opening. The
    /// single source of the intermediate-vs-final resolution for the
    /// scalar-shaped kinds (advice, program image); the bytecode kind's
    /// chunked wire shape spells the same resolution out in its own
    /// `output_claims`.
    fn scalar_claim(&self) -> Result<F, SumcheckKernelError<F>> {
        if self.has_address_phase() {
            Ok(self.tables.intermediate_claim())
        } else {
            self.tables.final_claim()
        }
    }

    /// Park the post-cycle bound state under `RA`'s carry key — the shared
    /// body of the per-kind `park_residue` overrides. A cycle-completed
    /// schedule has no stage-7 member, so it parks nothing.
    fn park_carry<RA: 'static>(self, session: &mut ProofSession) {
        if !self.has_address_phase() {
            return;
        }
        session.park(PrecommittedReductionCarry::<F, RA> {
            reduction: self.reduction,
            tables: self.tables,
            _relation: PhantomData,
        });
    }
}

impl<F: Field, R> ProveRounds<F> for CycleReductionKernel<F, R> {
    fn num_rounds(&self) -> usize {
        self.reduction.cycle_phase_total_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        Ok(self.tables.prove_round(
            self.reduction.cycle_phase_rounds(),
            bind,
            round,
            previous_claim,
        ))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        let total_rounds = self.num_rounds();
        self.tables
            .finish_rounds(self.reduction.cycle_phase_rounds(), total_rounds, bind);
        Ok(())
    }
}

/// The stage-7 address-phase batch member for relation `R`: resumes binding
/// from the reclaimed 6b carry (running scale included) and — per kind —
/// extracts the final openings from the fully bound tables.
pub struct AddressReductionKernel<F: Field, R> {
    reduction: PrecommittedClaimReduction,
    tables: PrecommittedTables<F>,
    _relation: PhantomData<fn() -> R>,
}

impl<F: Field, R> AddressReductionKernel<F, R> {
    pub fn new(carry: PrecommittedReductionCarry<F, R>) -> Self {
        Self {
            reduction: carry.reduction,
            tables: carry.tables,
            _relation: PhantomData,
        }
    }
}

impl<F: Field, R> ProveRounds<F> for AddressReductionKernel<F, R> {
    fn num_rounds(&self) -> usize {
        self.reduction.address_phase_total_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        Ok(self.tables.prove_round(
            self.reduction.address_phase_rounds(),
            bind,
            round,
            previous_claim,
        ))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        let total_rounds = self.num_rounds();
        self.tables
            .finish_rounds(self.reduction.address_phase_rounds(), total_rounds, bind);
        Ok(())
    }
}

impl<F: Field> SumcheckKernel<F> for CycleReductionKernel<F, TrustedAdviceCyclePhase<F>> {
    type Relation = TrustedAdviceCyclePhase<F>;

    fn output_claims(
        &mut self,
    ) -> Result<TrustedAdviceCyclePhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(TrustedAdviceCyclePhaseOutputClaims {
            trusted: self.scalar_claim()?,
        })
    }

    fn park_residue(self: Box<Self>, session: &mut ProofSession) {
        self.park_carry::<TrustedAdviceAddressPhase<F>>(session);
    }
}

impl<F: Field> SumcheckKernel<F> for CycleReductionKernel<F, UntrustedAdviceCyclePhase<F>> {
    type Relation = UntrustedAdviceCyclePhase<F>;

    fn output_claims(
        &mut self,
    ) -> Result<UntrustedAdviceCyclePhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(UntrustedAdviceCyclePhaseOutputClaims {
            untrusted: self.scalar_claim()?,
        })
    }

    fn park_residue(self: Box<Self>, session: &mut ProofSession) {
        self.park_carry::<UntrustedAdviceAddressPhase<F>>(session);
    }
}

impl<F: Field> SumcheckKernel<F> for CycleReductionKernel<F, BytecodeReductionCyclePhase<F>> {
    type Relation = BytecodeReductionCyclePhase<F>;

    fn output_claims(
        &mut self,
    ) -> Result<BytecodeReductionCyclePhaseOutputClaims<F>, SumcheckKernelError<F>> {
        // The chunked counterpart of `scalar_claim`: an address phase stages
        // the intermediate handoff claim (chunks come later, at stage 7); a
        // cycle-only schedule ends here with the per-chunk openings.
        Ok(if self.has_address_phase() {
            BytecodeReductionCyclePhaseOutputClaims {
                intermediate: Some(self.tables.intermediate_claim()),
                chunks: Vec::new(),
            }
        } else {
            BytecodeReductionCyclePhaseOutputClaims {
                intermediate: None,
                chunks: self.tables.final_aux_claims()?,
            }
        })
    }

    fn park_residue(self: Box<Self>, session: &mut ProofSession) {
        self.park_carry::<BytecodeReductionAddressPhase<F>>(session);
    }
}

impl<F: Field> SumcheckKernel<F> for CycleReductionKernel<F, ProgramImageReductionCyclePhase<F>> {
    type Relation = ProgramImageReductionCyclePhase<F>;

    fn output_claims(
        &mut self,
    ) -> Result<ProgramImageReductionCyclePhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(ProgramImageReductionCyclePhaseOutputClaims {
            program_image: self.scalar_claim()?,
        })
    }

    fn park_residue(self: Box<Self>, session: &mut ProofSession) {
        self.park_carry::<ProgramImageReductionAddressPhase<F>>(session);
    }
}

impl<F: Field> SumcheckKernel<F> for AddressReductionKernel<F, TrustedAdviceAddressPhase<F>> {
    type Relation = TrustedAdviceAddressPhase<F>;

    fn output_claims(
        &mut self,
    ) -> Result<TrustedAdviceAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(TrustedAdviceAddressPhaseOutputClaims {
            trusted: self.tables.final_claim()?,
        })
    }
}

impl<F: Field> SumcheckKernel<F> for AddressReductionKernel<F, UntrustedAdviceAddressPhase<F>> {
    type Relation = UntrustedAdviceAddressPhase<F>;

    fn output_claims(
        &mut self,
    ) -> Result<UntrustedAdviceAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(UntrustedAdviceAddressPhaseOutputClaims {
            untrusted: self.tables.final_claim()?,
        })
    }
}

impl<F: Field> SumcheckKernel<F> for AddressReductionKernel<F, BytecodeReductionAddressPhase<F>> {
    type Relation = BytecodeReductionAddressPhase<F>;

    fn output_claims(
        &mut self,
    ) -> Result<BytecodeReductionAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(BytecodeReductionAddressPhaseOutputClaims {
            chunks: self.tables.final_aux_claims()?,
        })
    }
}

impl<F: Field> SumcheckKernel<F>
    for AddressReductionKernel<F, ProgramImageReductionAddressPhase<F>>
{
    type Relation = ProgramImageReductionAddressPhase<F>;

    fn output_claims(
        &mut self,
    ) -> Result<ProgramImageReductionAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(ProgramImageReductionAddressPhaseOutputClaims {
            program_image: self.tables.final_claim()?,
        })
    }
}

/// The LSB-index relabeling implied by the big-endian opening-round
/// permutation: variables sorted by their global opening round become the new
/// LSB order. Returns `None` when the relabeling is the identity.
pub(crate) fn lsb_permutation(poly_opening_round_permutation_be: &[usize]) -> Option<Vec<usize>> {
    let num_vars = poly_opening_round_permutation_be.len();
    let mut be_var_by_round: Vec<usize> = (0..num_vars).collect();
    be_var_by_round.sort_unstable_by_key(|&be_index| poly_opening_round_permutation_be[be_index]);

    let mut old_lsb_to_new_lsb = vec![0usize; num_vars];
    for (new_lsb, be_index) in be_var_by_round.into_iter().enumerate() {
        old_lsb_to_new_lsb[num_vars - 1 - be_index] = new_lsb;
    }
    old_lsb_to_new_lsb
        .iter()
        .enumerate()
        .any(|(old_lsb, &new_lsb)| old_lsb != new_lsb)
        .then_some(old_lsb_to_new_lsb)
}

/// Out-of-place coefficient permute: `out[new_index] = table[old_index]` where
/// each of `new_index`'s bits moves to its pre-image LSB position.
pub(crate) fn permute_coefficients<F: Copy>(table: &[F], old_lsb_to_new_lsb: &[usize]) -> Vec<F> {
    let num_vars = old_lsb_to_new_lsb.len();
    let mut new_lsb_to_old_lsb = vec![0usize; num_vars];
    for (old_lsb, &new_lsb) in old_lsb_to_new_lsb.iter().enumerate() {
        new_lsb_to_old_lsb[new_lsb] = old_lsb;
    }
    (0..table.len())
        .map(|new_index| {
            let mut old_index = 0usize;
            for (new_lsb, &old_lsb) in new_lsb_to_old_lsb.iter().enumerate() {
                old_index |= ((new_index >> new_lsb) & 1) << old_lsb;
            }
            table[old_index]
        })
        .collect()
}

/// The challenge-vector counterpart of [`permute_coefficients`]: relabel the
/// big-endian challenge positions so `eq(permuted_challenges)` indexes the
/// permuted coefficient table.
pub(crate) fn permute_challenges<F: Copy>(
    challenges_be: &[F],
    old_lsb_to_new_lsb: &[usize],
) -> Vec<F> {
    let num_vars = challenges_be.len();
    let mut permuted = challenges_be.to_vec();
    for (old_be, &challenge) in challenges_be.iter().enumerate() {
        let new_lsb = old_lsb_to_new_lsb[num_vars - 1 - old_be];
        permuted[num_vars - 1 - new_lsb] = challenge;
    }
    permuted
}

/// Permute a batch of coefficient tables into the reduction's Dory
/// opening-round order (identity-permutation short-circuit included).
pub(crate) fn permute_tables<F: Copy>(
    reduction: &PrecommittedClaimReduction,
    tables: Vec<Vec<F>>,
) -> Vec<Vec<F>> {
    match lsb_permutation(reduction.poly_opening_round_permutation_be()) {
        Some(old_lsb_to_new_lsb) => tables
            .into_iter()
            .map(|table| permute_coefficients(&table, &old_lsb_to_new_lsb))
            .collect(),
        None => tables,
    }
}
