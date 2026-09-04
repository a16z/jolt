//! The optimized field-registers claim-reduction (stage 2) kernel,
//! byte-parity twin of [`crate::reference::field_registers_claim_reduction`].
//!
//! The reference binds four dense `T`-sized tables (`eq(τ_low)` plus the
//! three FR value columns) every round. The integer sibling's
//! ([`super::registers_claim_reduction`]) prefix–suffix `u64` machinery does
//! not transfer — FR values are full field elements — but the FR columns are
//! zero off FR-active cycles, which is the stronger structure:
//!
//! - **γ-combined sparse column**: the summand
//!   `eq(τ_low, t) · (rd + γ·rs1 + γ²·rs2)(t)` is linear in ONE combined
//!   column `V = rd + γ·rs1 + γ²·rs2` (exact by distributivity), held as
//!   sparse `(row, value)` cells over the FR-active cycles only. Absent
//!   cells are hard zeros (unlike the read-write kernel's step-function
//!   `Val`), so binding a lone cell just scales it.
//! - **Gruen split-eq rounds**: `s(t) = l(t) · q(t)` with the linear factor's
//!   `q(1)` accumulated over the sparse cells and `q(0)` recovered from the
//!   running claim (the sibling tier's eval-at-1 trade) — O(active + √T) per
//!   round, and an FR-inactive trace costs only the split-eq table build.
//! - **Direct opening claims at extraction**: the three produced openings
//!   come from one split-eq walk over the retained per-cycle value triples
//!   (the sibling kernels' post-hoc extraction pattern).

use jolt_claims::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineDerivedId, FieldRegistersClaimReductionChallenge,
    FieldRegistersClaimReductionPublic,
};
use jolt_claims::SumcheckChallenges as _;
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::field_registers_claim_reduction::FieldRegistersClaimReduction;
use jolt_verifier::VerifierError;
use jolt_witness::{JoltWitnessPlane, WitnessError};

use super::field_registers_read_write::field_register_rows;
use super::support::RoundChallenges;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// One FR-active cycle's combined-column cell.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct SparseCell<F> {
    row: usize,
    value: F,
}

pub struct OptimizedFieldRegistersClaimReduction;

impl<F: JoltField> PrepareKernel<F, FieldRegistersClaimReduction<F>>
    for OptimizedFieldRegistersClaimReduction
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, FieldRegistersClaimReduction<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = FieldRegistersClaimReduction<F>>>,
        KernelError<F>,
    > {
        let relation = inputs.relation;
        let log_t = relation.rounds();
        if log_t == 0 {
            return Err(KernelError::Unsupported {
                reason: "optimized FR claim reduction requires at least one cycle round",
            });
        }
        let tau_low: &[F] = relation.tau_low();
        if tau_low.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "FR claim-reduction tau point has the wrong variable count",
            });
        }
        let cycles = 1usize << log_t;

        let field_inline =
            witness
                .field_inline()
                .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                    label: "field-registers claim-reduction field-inline oracle",
                }))?;
        let rows = field_register_rows(session, field_inline, cycles, false)?;
        let gamma = inputs
            .challenges
            .resolve_challenge(&FieldInlineChallengeId::from(
                FieldRegistersClaimReductionChallenge::Gamma,
            ))
            .ok_or(KernelError::InvariantViolation {
                reason: "field-registers claim reduction is missing its gamma challenge",
            })?;
        let gamma_sq = gamma * gamma;

        // The per-cycle `[rd, rs1, rs2]` value triples of the FR-active
        // cycles (the oracle's `FieldRdValue`/`FieldRs1Value`/`FieldRs2Value`
        // extractions: write post-value, read values, zero when absent), and
        // their γ-combination as the sparse round column.
        let mut triples: Vec<(u32, [F; 3])> = Vec::new();
        let mut cells: Vec<SparseCell<F>> = Vec::new();
        for (row, access) in rows.iter().enumerate() {
            if access.rs1.is_none() && access.rs2.is_none() && access.rd.is_none() {
                continue;
            }
            let rd = access.rd.map_or_else(F::zero, |write| write.post_value);
            let rs1 = access.rs1.map_or_else(F::zero, |read| read.value);
            let rs2 = access.rs2.map_or_else(F::zero, |read| read.value);
            triples.push((row as u32, [rd, rs1, rs2]));
            cells.push(SparseCell {
                row,
                value: rd + gamma * rs1 + gamma_sq * rs2,
            });
        }

        Ok(Box::new(FieldClaimReductionKernel {
            gruen: GruenSplitEqPolynomial::new(tau_low, BindingOrder::LowToHigh),
            cells,
            scratch: Vec::new(),
            triples,
            challenges: RoundChallenges::new(log_t),
        }))
    }
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct FieldClaimReductionKernel<F: JoltField> {
    gruen: GruenSplitEqPolynomial<F>,
    /// Sparse combined-column cells, sorted by `row`; merged on each bind.
    cells: Vec<SparseCell<F>>,
    scratch: Vec<SparseCell<F>>,
    /// The raw per-cycle value triples, retained for the opening extraction.
    triples: Vec<(u32, [F; 3])>,
    challenges: RoundChallenges<F>,
}

impl<F: JoltField> FieldClaimReductionKernel<F> {
    /// `q(1) = Σ_z E(z) · V(2z + 1)` over the remaining domain — only cells
    /// on odd rows contribute.
    fn q_at_one(&self) -> F {
        let e_in = self.gruen.e_in_current();
        let e_out = self.gruen.e_out_current();
        let in_bits = if e_in.len() <= 1 {
            0
        } else {
            e_in.len().trailing_zeros() as usize
        };
        let mask = (1usize << in_bits) - 1;
        let mut sum = F::Accumulator::default();
        for cell in &self.cells {
            if cell.row.is_multiple_of(2) {
                continue;
            }
            let z = cell.row / 2;
            let weight = if e_in.len() <= 1 {
                e_out[z]
            } else {
                e_out[z >> in_bits] * e_in[z & mask]
            };
            sum.fmadd(weight, cell.value);
        }
        sum.reduce()
    }

    fn bind(&mut self, r: F) {
        self.scratch.clear();
        self.scratch.reserve(self.cells.len());
        let mut index = 0;
        while index < self.cells.len() {
            let cell = self.cells[index];
            let pair = cell.row / 2;
            let merged = if cell.row.is_multiple_of(2) {
                if let Some(odd) = self
                    .cells
                    .get(index + 1)
                    .filter(|next| next.row == cell.row + 1)
                {
                    let value = cell.value + r * (odd.value - cell.value);
                    index += 2;
                    SparseCell { row: pair, value }
                } else {
                    index += 1;
                    SparseCell {
                        row: pair,
                        value: (F::one() - r) * cell.value,
                    }
                }
            } else {
                index += 1;
                SparseCell {
                    row: pair,
                    value: r * cell.value,
                }
            };
            self.scratch.push(merged);
        }
        core::mem::swap(&mut self.cells, &mut self.scratch);
        self.gruen.bind(r);
        self.challenges.push(r);
    }

    /// The three produced opening values at the bound cycle point: one
    /// split-eq walk over the retained triples.
    fn claimed_values(&self) -> [F; 3] {
        let reversed: Vec<F> = self.challenges.as_slice().iter().rev().copied().collect();
        let hi_bits = reversed.len() / 2;
        let (r_hi, r_lo) = reversed.split_at(hi_bits);
        let e_hi = EqPolynomial::<F>::evals(r_hi, None);
        let e_lo = EqPolynomial::<F>::evals(r_lo, None);
        let lo_bits = reversed.len() - hi_bits;
        let mask = (1usize << lo_bits) - 1;
        let mut sums: [F::Accumulator; 3] = [F::Accumulator::default(); 3];
        for &(row, values) in &self.triples {
            let row = row as usize;
            let weight = e_hi[row >> lo_bits] * e_lo[row & mask];
            for (sum, value) in sums.iter_mut().zip(values) {
                sum.fmadd(weight, value);
            }
        }
        sums.map(Accumulator::reduce)
    }
}

impl<F: JoltField> ProveRounds<F> for FieldClaimReductionKernel<F> {
    fn num_rounds(&self) -> usize {
        self.challenges.total()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        // s(t) = l(t)·q(t) with q linear: q(1) from the sparse walk, q(0)
        // recovered from `s(0) + s(1) = previous_claim` (the eval-at-1
        // trade — a dishonest input claim surfaces at the driver's
        // final-claim check). The exact degree-2 coefficient vector is the
        // l·q product, which is what the reference's 3-point interpolation
        // reconstructs.
        let q_one = self.q_at_one();
        let (l_zero, l_one) = self.gruen.current_linear_evals();
        #[expect(
            clippy::expect_used,
            reason = "l(0) = eq-prefix·(1 − r_round) vanishes only on a zero transcript challenge — \
                      the split-eq recovery precedent"
        )]
        let q_zero = (previous_claim - l_one * q_one)
            * l_zero
                .inverse()
                .expect("current eq evaluation at zero must be invertible");
        let q_slope = q_one - q_zero;
        let l_slope = l_one - l_zero;
        Ok(UnivariatePoly::new(vec![
            l_zero * q_zero,
            l_zero * q_slope + l_slope * q_zero,
            l_slope * q_slope,
        ]))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for FieldClaimReductionKernel<F> {
    type Relation = FieldRegistersClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        use jolt_claims::protocols::field_inline::relations::claim_reductions::registers::FieldRegistersClaimReductionOutputClaims;

        self.challenges.require_complete()?;
        let [rd_value, rs1_value, rs2_value] = self.claimed_values();
        Ok(FieldRegistersClaimReductionOutputClaims {
            rd_value,
            rs1_value,
            rs2_value,
        })
    }

    /// The `EqSpartan` cross-check: the fully bound Gruen scalar must equal
    /// the verifier's `derive_output_term` at the bound point (the reference
    /// kernel's tie-down on the table it materializes).
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        let expected = relation.derive_output_term(
            &FieldInlineDerivedId::from(FieldRegistersClaimReductionPublic::EqSpartan),
            input_points,
            output_points,
            challenges,
        )?;
        let got = self.gruen.current_scalar();
        if got != expected {
            return Err(SumcheckKernelError::Verifier(
                VerifierError::StageClaimSumcheckFailed {
                    stage: "FieldRegistersClaimReduction".to_string(),
                    reason: format!(
                        "bound eq scalar {got:?}, but derive_output_term gives {expected:?}"
                    ),
                },
            ));
        }
        Ok(())
    }
}

/// Byte parity against the reference kernel on register-consistent FR
/// traces, plus the FR-inactive degenerate case (an empty sparse column).
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::field_inline::relations::claim_reductions::registers::{
        FieldRegistersClaimReductionChallenges, FieldRegistersClaimReductionInputClaims,
    };
    use jolt_claims::protocols::field_inline::FieldRegistersTraceDimensions;
    use jolt_field::{Fr, Ring};
    use jolt_riscv::FieldInlineOp;

    use super::*;
    use crate::optimized::field_registers_testing::{
        inactive_fr_fixture, structured_fr_fixture, FrTraceFixture,
    };
    use crate::optimized::parity::{
        probe_input_claim, run_lockstep, run_lockstep_degenerate, synthetic_point,
    };
    use crate::ReferenceBackend;

    fn run_parity(fixture: FrTraceFixture, log_t: usize, seed: u64, expect_active: bool) {
        fixture.with_plane(log_t, |backend| {
            let relation = FieldRegistersClaimReduction::<Fr>::new(
                FieldRegistersTraceDimensions::new(log_t),
                synthetic_point(log_t, seed),
            );
            let claims = FieldRegistersClaimReductionInputClaims {
                rd_value: Fr::from_u64(0),
                rs1_value: Fr::from_u64(0),
                rs2_value: Fr::from_u64(0),
            };
            let points = FieldRegistersClaimReductionInputClaims {
                rd_value: Vec::new(),
                rs1_value: Vec::new(),
                rs2_value: Vec::new(),
            };
            let challenges = FieldRegistersClaimReductionChallenges {
                gamma: Fr::from_u64(41 + seed),
            };
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let mut session = ProofSession::default();
            let mut reference = <ReferenceBackend as PrepareKernel<
                Fr,
                FieldRegistersClaimReduction<Fr>,
            >>::prepare(
                &ReferenceBackend, &mut session, backend, inputs()
            )
            .unwrap();
            let mut optimized = OptimizedFieldRegistersClaimReduction
                .prepare(&mut session, backend, inputs())
                .unwrap();

            let claim = probe_input_claim(reference.as_mut());
            let round_challenges =
                synthetic_point(relation.rounds(), seed.wrapping_mul(0x9E37_79B9));
            if expect_active {
                assert!(claim != Fr::from_u64(0), "FR-active fixture degenerated");
                run_lockstep(
                    reference.as_mut(),
                    optimized.as_mut(),
                    claim,
                    &round_challenges,
                );
            } else {
                assert_eq!(claim, Fr::from_u64(0), "FR-inactive claim must be zero");
                run_lockstep_degenerate(
                    reference.as_mut(),
                    optimized.as_mut(),
                    claim,
                    &round_challenges,
                );
            }
            assert_eq!(
                reference.output_claims(&claims).unwrap(),
                optimized.output_claims(&claims).unwrap()
            );
            let output_points = relation
                .derive_opening_points(&round_challenges, &points)
                .unwrap();
            reference
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
            optimized
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
        });
    }

    #[test]
    fn parity_structured_even_log_t() {
        run_parity(structured_fr_fixture(16), 4, 401, true);
    }

    #[test]
    fn parity_structured_odd_log_t() {
        run_parity(structured_fr_fixture(8), 3, 409, true);
    }

    #[test]
    fn parity_single_cycle_round() {
        let mut fixture = FrTraceFixture::new();
        fixture.load_imm(15, 7);
        fixture.arithmetic(FieldInlineOp::Add, 0, 15, 15);
        run_parity(fixture, 1, 419, true);
    }

    #[test]
    fn parity_inactive_trace_is_degenerate() {
        run_parity(inactive_fr_fixture(4), 3, 421, false);
    }
}
