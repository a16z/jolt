//! The optimized registers claim-reduction (stage 3) kernel.
//!
//! Byte-parity contract: identical round polynomials and output claims to the
//! reference kernel (`reference/registers_claim_reduction.rs`), which binds
//! four dense `T`-sized tables (`eq(τ_low)` plus the three value columns)
//! every round.
//!
//! Techniques ported from
//! `jolt-prover-legacy/src/zkvm/claim_reductions/registers.rs`:
//!
//! - **Prefix–suffix P·Q decomposition** (eprint 2025/611, Appendix A) for
//!   the first half of the rounds: `Σ_j eq(τ, j)·V(j)` becomes
//!   `Σ_{x_lo} P(x_lo)·Q(x_lo)` with `P = eq(τ_lo)` and
//!   `Q(x_lo) = Σ_{x_hi} eq(τ_hi)[x_hi]·V(x_hi‖x_lo)` — two √T-sized tables
//!   per round instead of four T-sized ones. Since `eq` factors across the
//!   variable split and suffix-summing commutes with prefix partial
//!   evaluation, every P·Q round polynomial equals the dense summand's
//!   exactly.
//! - **Phase-2 regeneration**: once the prefix is bound, the three value
//!   columns are folded by `eq(r_prefix)` straight from the raw `u64` trace
//!   values into `2^(remaining)`-sized tables, and the eq table is the
//!   suffix table scaled by the bound-prefix eq factor — the exact partial
//!   binds of the dense tables, recomputed instead of carried.
//! - **Raw `u64` trace values with small-scalar deferred-reduction
//!   accumulation** (`fmadd_u64`) for the Q build and the phase-2 folds.

use jolt_claims::protocols::jolt::{
    JoltDerivedId, JoltPolynomialId, RegistersClaimReductionPublic,
};
use jolt_field::{AdditiveAccumulator, Field, RingAccumulator, SignedScalarAccumulator};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage3::registers_claim_reduction::{
    RegistersClaimReduction, RegistersClaimReductionOutputClaims,
};
use jolt_witness::witnesses::WitnessEnv;
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::collect_rows;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Per-cycle `[rd write value, rs1 value, rs2 value]`, kept as raw `u64`s so
/// the eq folds run on small-scalar fused multiply-adds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RegisterValuesRow([u64; 3]);

impl WitnessBundle for RegisterValuesRow {
    fn from_row(
        row: &jolt_witness::__private::TraceRow,
        _next: Option<&jolt_witness::__private::TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self([
            row.rd_write_value(),
            row.rs1_value(),
            row.rs2_value(),
        ]))
    }

    fn annotated_ids() -> Vec<JoltPolynomialId> {
        Vec::new()
    }
}

/// Accumulate `eq · F(value)` for a full-range `u64` on the small-scalar
/// accumulator without overflowing it: the accumulator's headroom is one
/// extra limb, which ~4 full-magnitude `field × u64` products exhaust, so the
/// value is split into u32 halves (products ≤ 2^286, headroom ≥ 2^34 terms).
/// `eq_shifted` must be `eq · 2^32`; the two fused adds sum to exactly
/// `eq · F(value)`.
#[inline]
fn fmadd_u64_split<F: Field>(
    accumulator: &mut F::SmallScalarAccumulator,
    eq: F,
    eq_shifted: F,
    value: u64,
) {
    accumulator.fmadd_u64(eq_shifted, value >> 32);
    accumulator.fmadd_u64(eq, value & 0xFFFF_FFFF);
}

pub struct OptimizedRegistersClaimReduction;

impl<F: Field> PrepareKernel<F, RegistersClaimReduction<F>> for OptimizedRegistersClaimReduction {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersClaimReduction<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let log_t = relation.rounds();
        if log_t == 0 {
            return Err(KernelError::Unsupported {
                reason: "optimized registers claim reduction requires at least one cycle round",
            });
        }
        let tau: &[F] = relation.product_uniskip_tau_low();
        if tau.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "registers claim-reduction tau point has the wrong variable count",
            });
        }
        let cycles = 1usize << log_t;
        let values: Vec<RegisterValuesRow> = collect_rows(witness, cycles)?;

        let gamma = inputs.challenges.gamma;
        let gamma_sq = gamma * gamma;

        // τ = τ_hi ‖ τ_lo (big-endian). The prefix (low, bound-first) part
        // takes the extra variable when log_t is odd, matching legacy.
        let (tau_hi, tau_lo) = tau.split_at(log_t / 2);
        let prefix_vars = tau_lo.len();
        let p = EqPolynomial::<F>::evals(tau_lo, None);
        let eq_suffix = EqPolynomial::<F>::evals(tau_hi, None);

        // Q(x_lo) = Σ_{x_hi} eq(τ_hi)[x_hi] · V(x_hi ‖ x_lo), with the three
        // value columns folded on u64 accumulators and γ-combined once.
        const BLOCK: usize = 32;
        let build_q_block = |(block_index, q_block): (usize, &mut [F])| {
            let mut folds = vec![[F::SmallScalarAccumulator::default(); 3]; q_block.len()];
            for (x_hi, &eq_hi) in eq_suffix.iter().enumerate() {
                let eq_hi_shifted = eq_hi.mul_pow_2(32);
                let base = x_hi << prefix_vars;
                for (i, fold) in folds.iter_mut().enumerate() {
                    let x_lo = block_index * BLOCK + i;
                    let row = values[base + x_lo].0;
                    fmadd_u64_split(&mut fold[0], eq_hi, eq_hi_shifted, row[0]);
                    fmadd_u64_split(&mut fold[1], eq_hi, eq_hi_shifted, row[1]);
                    fmadd_u64_split(&mut fold[2], eq_hi, eq_hi_shifted, row[2]);
                }
            }
            for (q, fold) in q_block.iter_mut().zip(folds) {
                *q = fold[0].reduce() + gamma * fold[1].reduce() + gamma_sq * fold[2].reduce();
            }
        };
        let mut q = vec![F::zero(); 1 << prefix_vars];
        #[cfg(feature = "parallel")]
        q.par_chunks_mut(BLOCK).enumerate().for_each(build_q_block);
        #[cfg(not(feature = "parallel"))]
        q.chunks_mut(BLOCK).enumerate().for_each(build_q_block);

        Ok(Box::new(ClaimReductionKernel {
            log_t,
            gamma,
            gamma_sq,
            tau: tau.to_vec(),
            values,
            phase: Phase::PrefixSuffix { p, q },
            bound_challenges: Vec::with_capacity(log_t),
        }))
    }
}

enum Phase<F> {
    /// First half of the rounds: the P·Q buffers over the prefix variables.
    PrefixSuffix { p: Vec<F>, q: Vec<F> },
    /// Remaining rounds: the eq table and the three value columns, dense.
    Dense {
        eq: Vec<F>,
        rd_write_value: Vec<F>,
        rs1_value: Vec<F>,
        rs2_value: Vec<F>,
    },
}

struct ClaimReductionKernel<F: Field> {
    log_t: usize,
    gamma: F,
    gamma_sq: F,
    /// The full `τ_low` point (big-endian) the summand's eq factor fixes.
    tau: Vec<F>,
    /// Raw per-cycle `u64` values, kept for the phase-2 regeneration.
    values: Vec<RegisterValuesRow>,
    phase: Phase<F>,
    bound_challenges: Vec<F>,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for ClaimReductionKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        use crate::backend::vec_heap_bytes;
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(allocative::Key::new("tau"), vec_heap_bytes(&self.tau));
        visitor.visit_simple(allocative::Key::new("values"), vec_heap_bytes(&self.values));
        let phase_bytes = match &self.phase {
            Phase::PrefixSuffix { p, q } => vec_heap_bytes(p) + vec_heap_bytes(q),
            Phase::Dense {
                eq,
                rd_write_value,
                rs1_value,
                rs2_value,
            } => {
                vec_heap_bytes(eq)
                    + vec_heap_bytes(rd_write_value)
                    + vec_heap_bytes(rs1_value)
                    + vec_heap_bytes(rs2_value)
            }
        };
        visitor.visit_simple(allocative::Key::new("phase"), phase_bytes);
        visitor.visit_simple(
            allocative::Key::new("bound_challenges"),
            vec_heap_bytes(&self.bound_challenges),
        );
        visitor.exit();
    }
}

impl<F: Field> ClaimReductionKernel<F> {
    fn rounds_bound(&self) -> usize {
        self.bound_challenges.len()
    }

    fn require_fully_bound(&self) -> Result<(), SumcheckKernelError<F>> {
        let remaining = self.log_t - self.rounds_bound();
        if remaining == 0 {
            Ok(())
        } else {
            Err(SumcheckKernelError::NotFullyBound { remaining })
        }
    }

    /// Regenerate the dense phase from the raw values: the three columns
    /// folded by `eq(r_prefix)` (their exact partial binds) and the suffix
    /// eq table scaled by the bound-prefix eq factor.
    fn transition_to_dense(&mut self) {
        let bound = self.rounds_bound();
        let r_prefix: Vec<F> = self.bound_challenges.iter().rev().copied().collect();
        let eq_prefix = EqPolynomial::<F>::evals(&r_prefix, None);
        let eq_prefix_shifted: Vec<F> = eq_prefix.iter().map(|eq| eq.mul_pow_2(32)).collect();
        let chunk = eq_prefix.len();
        let remaining = 1usize << (self.log_t - bound);

        let fold_chunk = |rows: &[RegisterValuesRow]| -> [F; 3] {
            let mut fold = [F::SmallScalarAccumulator::default(); 3];
            for (row, (&eq, &eq_shifted)) in rows
                .iter()
                .zip(eq_prefix.iter().zip(eq_prefix_shifted.iter()))
            {
                fmadd_u64_split(&mut fold[0], eq, eq_shifted, row.0[0]);
                fmadd_u64_split(&mut fold[1], eq, eq_shifted, row.0[1]);
                fmadd_u64_split(&mut fold[2], eq, eq_shifted, row.0[2]);
            }
            fold.map(F::SmallScalarAccumulator::reduce)
        };
        #[cfg(feature = "parallel")]
        let folds: Vec<[F; 3]> = self.values.par_chunks(chunk).map(fold_chunk).collect();
        #[cfg(not(feature = "parallel"))]
        let folds: Vec<[F; 3]> = self.values.chunks(chunk).map(fold_chunk).collect();
        debug_assert_eq!(folds.len(), remaining);

        // The raw values only feed this regeneration; free them now.
        self.values = Vec::new();

        let (tau_hi, tau_lo) = self.tau.split_at(self.log_t / 2);
        let eq_prefix_eval = EqPolynomial::<F>::mle(&r_prefix, tau_lo);
        self.phase = Phase::Dense {
            eq: EqPolynomial::<F>::evals(tau_hi, Some(eq_prefix_eval)),
            rd_write_value: folds.iter().map(|fold| fold[0]).collect(),
            rs1_value: folds.iter().map(|fold| fold[1]).collect(),
            rs2_value: folds.iter().map(|fold| fold[2]).collect(),
        };
    }

    fn bind(&mut self, r: F) {
        self.bound_challenges.push(r);
        // Last prefix variable: regenerate the dense phase from the raw
        // values instead of binding the exhausted P·Q.
        if matches!(&self.phase, Phase::PrefixSuffix { p, .. } if p.len() == 2) {
            self.transition_to_dense();
            return;
        }
        match &mut self.phase {
            Phase::PrefixSuffix { p, q } => {
                for table in [p, q] {
                    let half = table.len() / 2;
                    for y in 0..half {
                        let lo = table[2 * y];
                        table[y] = lo + r * (table[2 * y + 1] - lo);
                    }
                    table.truncate(half);
                }
            }
            Phase::Dense {
                eq,
                rd_write_value,
                rs1_value,
                rs2_value,
            } => {
                for table in [eq, rd_write_value, rs1_value, rs2_value] {
                    let half = table.len() / 2;
                    for y in 0..half {
                        let lo = table[2 * y];
                        table[y] = lo + r * (table[2 * y + 1] - lo);
                    }
                    table.truncate(half);
                }
            }
        }
    }
}

impl<F: Field> ProveRounds<F> for ClaimReductionKernel<F> {
    fn num_rounds(&self) -> usize {
        self.log_t
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

        // Degree-2 member: evals at t = 0 and t = 2; s(1) from the hint.
        let evals: [F; 2] = match &self.phase {
            Phase::PrefixSuffix { p, q } => {
                let mut acc = [F::Accumulator::default(); 2];
                for y in 0..p.len() / 2 {
                    let (p_0, p_1) = (p[2 * y], p[2 * y + 1]);
                    let (q_0, q_1) = (q[2 * y], q[2 * y + 1]);
                    acc[0].fmadd(p_0, q_0);
                    acc[1].fmadd(p_1 + p_1 - p_0, q_1 + q_1 - q_0);
                }
                acc.map(F::Accumulator::reduce)
            }
            Phase::Dense {
                eq,
                rd_write_value,
                rs1_value,
                rs2_value,
            } => {
                let mut acc = [F::Accumulator::default(); 2];
                for y in 0..eq.len() / 2 {
                    let pair = |table: &[F]| (table[2 * y], table[2 * y + 1]);
                    let (eq_0, eq_1) = pair(eq);
                    let (rd_0, rd_1) = pair(rd_write_value);
                    let (rs1_0, rs1_1) = pair(rs1_value);
                    let (rs2_0, rs2_1) = pair(rs2_value);
                    acc[0].fmadd(eq_0, rd_0 + self.gamma * rs1_0 + self.gamma_sq * rs2_0);
                    acc[1].fmadd(
                        eq_1 + eq_1 - eq_0,
                        (rd_1 + rd_1 - rd_0)
                            + self.gamma * (rs1_1 + rs1_1 - rs1_0)
                            + self.gamma_sq * (rs2_1 + rs2_1 - rs2_0),
                    );
                }
                acc.map(F::Accumulator::reduce)
            }
        };

        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: Field> SumcheckKernel<F> for ClaimReductionKernel<F> {
    type Relation = RegistersClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RegistersClaimReductionOutputClaims<F>, SumcheckKernelError<F>> {
        self.require_fully_bound()?;
        let Phase::Dense {
            rd_write_value,
            rs1_value,
            rs2_value,
            ..
        } = &self.phase
        else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "claim reduction must finish in the dense phase",
            });
        };
        Ok(RegistersClaimReductionOutputClaims {
            rd_write_value: rd_write_value[0],
            rs1_value: rs1_value[0],
            rs2_value: rs2_value[0],
        })
    }

    /// Pin the regenerated eq table to the verifier's scalar path: its fully
    /// bound value must equal `derive_output_term(EqSpartan)`.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.require_fully_bound()?;
        let Phase::Dense { eq, .. } = &self.phase else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "claim reduction must finish in the dense phase",
            });
        };
        let id = JoltDerivedId::from(RegistersClaimReductionPublic::EqSpartan);
        let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
        if eq[0] != expected {
            return Err(SumcheckKernelError::DerivedTableDrift {
                id,
                expected,
                got: eq[0],
            });
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::Polynomial;
    use jolt_verifier::stages::stage3::registers_claim_reduction::{
        RegistersClaimReduction, RegistersClaimReductionChallenges,
        RegistersClaimReductionInputClaims,
    };
    use jolt_witness::JoltWitnessOracle;

    use super::super::registers_read_write::test_support::{
        assert_kernel_parity, assert_nontrivial, challenge_sequence, structured_fixture,
        TraceFixture,
    };
    use super::OptimizedRegistersClaimReduction;

    fn run_parity(fixture: TraceFixture, log_t: usize, seed: u64) {
        fixture.with_plane(log_t, |backend| {
            let tau = challenge_sequence(log_t, seed ^ 0x7E7E);
            let relation =
                RegistersClaimReduction::<Fr>::new(TraceDimensions::new(log_t), tau.clone());
            let evaluate = |polynomial: JoltVirtualPolynomial| {
                let table = JoltWitnessOracle::<Fr>::oracle_table(
                    backend,
                    JoltPolynomialId::Virtual(polynomial),
                )
                .unwrap();
                Polynomial::new(table).evaluate(&tau)
            };
            let gamma = Fr::from_u64(0x0DDB_A11C_0FFE_E123);
            let claims = RegistersClaimReductionInputClaims {
                rd_write_value: evaluate(JoltVirtualPolynomial::RdWriteValue),
                rs1_value: evaluate(JoltVirtualPolynomial::Rs1Value),
                rs2_value: evaluate(JoltVirtualPolynomial::Rs2Value),
            };
            let input_claim =
                claims.rd_write_value + gamma * claims.rs1_value + gamma * gamma * claims.rs2_value;
            assert_nontrivial(input_claim);
            let round_challenges = challenge_sequence(log_t, seed);
            assert_kernel_parity(
                &OptimizedRegistersClaimReduction,
                backend,
                &relation,
                &claims,
                &RegistersClaimReductionInputClaims::default(),
                &RegistersClaimReductionChallenges { gamma },
                input_claim,
                &round_challenges,
            );
        });
    }

    #[test]
    fn parity_structured_odd_log_t() {
        run_parity(structured_fixture(8), 3, 101);
    }

    #[test]
    fn parity_structured_even_log_t() {
        run_parity(structured_fixture(16), 4, 103);
    }

    #[test]
    fn parity_minimal_single_round() {
        // log_t = 1: the P·Q phase covers the single round and the dense
        // phase materializes inside `finish_rounds`.
        let mut fixture = TraceFixture::new();
        fixture.op(Some(6), Some(2), Some(3));
        fixture.op(None, Some(6), Some(6));
        run_parity(fixture, 1, 107);
    }

    #[test]
    fn parity_padded_trace() {
        let mut fixture = TraceFixture::new();
        fixture.op(Some(8), Some(1), None);
        fixture.op(Some(9), Some(8), Some(8));
        fixture.op(None, Some(9), Some(8));
        run_parity(fixture, 2, 109);
    }
}
