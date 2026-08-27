//! The optimized RAM RA claim-reduction (stage 5) kernel.
//!
//! Port of legacy `zkvm/claim_reductions/ram_ra.rs`. The summand
//! `(eq(r_cycle_raf, j) + γ·eq(r_cycle_rw, j) + γ²·eq(r_cycle_val, j)) · ra(r_address, j)`
//! has per-term tensor structure `eq_lo(j_lo)·eq_hi(j_hi)·H(j)` with
//! `H(j) = eq(r_address)[addresses[j]]` served off the session-shared RAM
//! access columns, so no round ever touches a dense `T`-sized table:
//!
//! - **Prefix rounds** (`log_T/2`, low-to-high): the round loop runs over
//!   `P_x[j_lo] = eq(r_x_lo)[j_lo]` and
//!   `Q_x[j_lo] = Σ_{j_hi} H[j]·eq(r_x_hi)[j_hi]` — six `O(√T)` tables, with
//!   `Q` built in one `O(T)` columns pass. Binding low variables commutes
//!   with summing out the high ones, so the messages equal the reference's
//!   dense-table rounds exactly.
//! - **Suffix rounds**: at the phase switch one more `O(T)` pass gathers
//!   `H'[j_hi] = Σ_{j_lo} eq(r_prefix)[j_lo]·H[j]` (the partial evaluation of
//!   `ra` at the prefix challenges); the bound prefix eq factors collapse to
//!   the scalars `eq(r_x_lo, r_prefix)` folded into the term coefficients.
//!
//! Round messages sample `t ∈ {0, 2}` with the engine hint supplying `s(1)`.

use std::sync::Arc;

use jolt_claims::protocols::jolt::{JoltDerivedId, RamRaClaimReductionPublic};
use jolt_field::JoltField;
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
#[cfg(feature = "parallel")]
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage5::ram_ra_claim_reduction::{
    RamRaClaimReduction, RamRaClaimReductionOutputClaims,
};
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::ram_trace::{RamAccessColumns, NO_ACCESS};
use super::support::{bind_pairs, pin_derived_term, RoundProgress};
use super::OptimizedBackend;
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// The three consumed claims (RAF, read-write, val-check), in γ-power order.
const TERMS: usize = 3;

impl<F: JoltField> PrepareKernel<F, RamRaClaimReduction<F>> for OptimizedBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamRaClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamRaClaimReduction<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let log_t = relation.trace_dimensions().log_t();
        let ram_log_k = relation.ram_log_k();
        let input_points = inputs.points;
        let expected_len = ram_log_k + log_t;
        for point in [
            input_points.raf(),
            input_points.read_write(),
            input_points.val_check(),
        ] {
            if point.len() != expected_len {
                return Err(KernelError::InvariantViolation {
                    reason: "RAM RA claim-reduction input point has the wrong variable count",
                });
            }
        }
        // The shared address prefix (the relation's `derive_opening_points`
        // hard-checks that all three inputs agree on it).
        let r_address = &input_points.read_write()[..ram_log_k];

        let columns = RamAccessColumns::shared(session, witness, log_t)?;
        columns.validate_addresses(1usize << ram_log_k)?;
        let eq_address = eq_table(r_address);

        let gamma = inputs.challenges.gamma;
        let gamma_powers = [F::one(), gamma, gamma * gamma];

        let prefix_bits = log_t / 2;
        let suffix_bits = log_t - prefix_bits;
        let cycle_points: [&[F]; TERMS] = [
            &input_points.raf()[ram_log_k..],
            &input_points.read_write()[ram_log_k..],
            &input_points.val_check()[ram_log_k..],
        ];
        let eq_hi = cycle_points.map(|r| eq_table(&r[..suffix_bits]));

        let phase = if prefix_bits == 0 {
            Phase::Suffix {
                h: columns.fold_addresses(&eq_address),
                eq_hi,
                scales: [F::one(); TERMS],
            }
        } else {
            Phase::Prefix {
                p: cycle_points.map(|r| eq_table(&r[suffix_bits..])),
                q: build_q_tables(&columns, &eq_address, &eq_hi, prefix_bits),
                eq_hi,
                columns,
                eq_address,
                r_cycle_lo: cycle_points.map(|r| r[suffix_bits..].to_vec()),
                challenges: Vec::with_capacity(prefix_bits),
            }
        };

        Ok(Box::new(RaReductionKernel {
            progress: RoundProgress::new(log_t),
            prefix_bits,
            gamma_powers,
            phase,
        }))
    }
}

/// `Q_x[c_lo] = Σ_{c_hi} eq(r_address)[addresses[c_hi‖c_lo]] · eq_hi_x[c_hi]`
/// for the three cycle points, in one pass over the access columns.
fn build_q_tables<F: JoltField>(
    columns: &RamAccessColumns,
    eq_address: &[F],
    eq_hi: &[Vec<F>; TERMS],
    prefix_bits: usize,
) -> [Vec<F>; TERMS] {
    let prefix_size = 1usize << prefix_bits;
    let fill = |q: &mut [Vec<F>; TERMS], base: usize, chunk: &[u32]| {
        for (i, &address) in chunk.iter().enumerate() {
            if address == NO_ACCESS {
                continue;
            }
            let c = base + i;
            let c_lo = c & (prefix_size - 1);
            let c_hi = c >> prefix_bits;
            let h_c = eq_address[address as usize];
            for x in 0..TERMS {
                q[x][c_lo] += h_c * eq_hi[x][c_hi];
            }
        }
    };

    #[cfg(feature = "parallel")]
    {
        const CHUNK: usize = 1 << 14;
        columns
            .addresses
            .par_chunks(CHUNK)
            .enumerate()
            .fold(
                || core::array::from_fn(|_| unsafe_allocate_zero_vec(prefix_size)),
                |mut q, (chunk_index, chunk)| {
                    fill(&mut q, chunk_index * CHUNK, chunk);
                    q
                },
            )
            .reduce(
                || core::array::from_fn(|_| unsafe_allocate_zero_vec(prefix_size)),
                |mut acc, q| {
                    for (acc_x, q_x) in acc.iter_mut().zip(&q) {
                        for (a, v) in acc_x.iter_mut().zip(q_x) {
                            *a += *v;
                        }
                    }
                    acc
                },
            )
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut q = core::array::from_fn(|_| vec![F::zero(); prefix_size]);
        fill(&mut q, 0, &columns.addresses);
        q
    }
}

/// `H'[c_hi] = Σ_{c_lo} eq(r_address)[addresses[c_hi‖c_lo]] · eq_prefix[c_lo]`
/// — the partial evaluation of the address-folded `ra` at the prefix
/// challenges, regathered from the access columns.
fn gather_h_prime<F: JoltField>(
    columns: &RamAccessColumns,
    eq_address: &[F],
    eq_prefix: &[F],
    prefix_bits: usize,
    suffix_bits: usize,
) -> Vec<F> {
    let prefix_size = 1usize << prefix_bits;
    let suffix_size = 1usize << suffix_bits;
    let fill = |h: &mut Vec<F>, base: usize, chunk: &[u32]| {
        for (i, &address) in chunk.iter().enumerate() {
            if address == NO_ACCESS {
                continue;
            }
            let c = base + i;
            h[c >> prefix_bits] += eq_address[address as usize] * eq_prefix[c & (prefix_size - 1)];
        }
    };

    #[cfg(feature = "parallel")]
    {
        const CHUNK: usize = 1 << 14;
        columns
            .addresses
            .par_chunks(CHUNK)
            .enumerate()
            .fold(
                || unsafe_allocate_zero_vec(suffix_size),
                |mut h, (chunk_index, chunk)| {
                    fill(&mut h, chunk_index * CHUNK, chunk);
                    h
                },
            )
            .reduce(
                || unsafe_allocate_zero_vec(suffix_size),
                |mut acc, h| {
                    for (a, v) in acc.iter_mut().zip(&h) {
                        *a += *v;
                    }
                    acc
                },
            )
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut h = vec![F::zero(); suffix_size];
        fill(&mut h, 0, &columns.addresses);
        h
    }
}

#[expect(
    clippy::large_enum_variant,
    reason = "one kernel object per proof; boxing buys nothing"
)]
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
enum Phase<F: JoltField> {
    /// Rounds over the low (prefix) cycle variables: six `O(√T)` tables. The
    /// suffix eq tables and the transition inputs (columns, address eq,
    /// low-half cycle points, collected challenges) ride along.
    Prefix {
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalar_rows))]
        p: [Vec<F>; TERMS],
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalar_rows))]
        q: [Vec<F>; TERMS],
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalar_rows))]
        eq_hi: [Vec<F>; TERMS],
        columns: Arc<RamAccessColumns>,
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        eq_address: Vec<F>,
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalar_rows))]
        r_cycle_lo: [Vec<F>; TERMS],
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        challenges: Vec<F>,
    },
    /// Rounds over the high (suffix) cycle variables after the regather.
    /// `scales[x] = eq(r_x_lo, r_prefix)` — the bound prefix eq factors.
    Suffix {
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        h: Vec<F>,
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalar_rows))]
        eq_hi: [Vec<F>; TERMS],
        #[cfg_attr(feature = "allocative", allocative(skip))]
        scales: [F; TERMS],
    },
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct RaReductionKernel<F: JoltField> {
    progress: RoundProgress,
    prefix_bits: usize,
    /// `[1, γ, γ²]` — the consumed-claim batching coefficients.
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gamma_powers: [F; TERMS],
    phase: Phase<F>,
}
impl<F: JoltField> RaReductionKernel<F> {
    fn bind(&mut self, r: F) {
        self.progress.advance();
        match &mut self.phase {
            Phase::Prefix {
                p, q, challenges, ..
            } => {
                for table in p.iter_mut().chain(q.iter_mut()) {
                    bind_pairs(table, r);
                }
                challenges.push(r);
                if p[0].len() == 1 {
                    self.transition_to_suffix();
                }
            }
            Phase::Suffix { h, eq_hi, .. } => {
                bind_pairs(h, r);
                for table in eq_hi.iter_mut() {
                    bind_pairs(table, r);
                }
            }
        }
    }

    /// Swap the prefix state for the suffix state: regather `H'` from the
    /// access columns at the collected prefix challenges and collapse the
    /// bound prefix eq factors into scalars.
    fn transition_to_suffix(&mut self) {
        let placeholder = Phase::Suffix {
            h: Vec::new(),
            eq_hi: [Vec::new(), Vec::new(), Vec::new()],
            scales: [F::one(); TERMS],
        };
        let Phase::Prefix {
            eq_hi,
            columns,
            eq_address,
            r_cycle_lo,
            challenges,
            ..
        } = core::mem::replace(&mut self.phase, placeholder)
        else {
            debug_assert!(false, "transition called outside the prefix phase");
            return;
        };
        // Low-to-high challenges reversed give the big-endian prefix point.
        let r_prefix: Vec<F> = challenges.iter().rev().copied().collect();
        let eq_prefix = eq_table(&r_prefix);
        let h = gather_h_prime(
            &columns,
            &eq_address,
            &eq_prefix,
            self.prefix_bits,
            self.progress.total() - self.prefix_bits,
        );
        let scales = core::array::from_fn(|x| EqPolynomial::<F>::mle(&r_cycle_lo[x], &r_prefix));
        self.phase = Phase::Suffix { h, eq_hi, scales };
    }

    /// `[s(0), s(2)]` of the current round polynomial; `s(1)` comes from the
    /// engine hint.
    fn message_evals(&self) -> [F; 2] {
        match &self.phase {
            Phase::Prefix { p, q, .. } => {
                let mut evals = [F::zero(); 2];
                for x in 0..TERMS {
                    let (p_x, q_x) = (&p[x], &q[x]);
                    let mut sum = [F::zero(); 2];
                    for y in 0..p_x.len() / 2 {
                        let (p_0, p_1) = (p_x[2 * y], p_x[2 * y + 1]);
                        let (q_0, q_1) = (q_x[2 * y], q_x[2 * y + 1]);
                        sum[0] += p_0 * q_0;
                        sum[1] += (p_1 + p_1 - p_0) * (q_1 + q_1 - q_0);
                    }
                    evals[0] += self.gamma_powers[x] * sum[0];
                    evals[1] += self.gamma_powers[x] * sum[1];
                }
                evals
            }
            Phase::Suffix { h, eq_hi, scales } => {
                let coeff: [F; TERMS] = core::array::from_fn(|x| self.gamma_powers[x] * scales[x]);
                let mut evals = [F::zero(); 2];
                for y in 0..h.len() / 2 {
                    let (h_0, h_1) = (h[2 * y], h[2 * y + 1]);
                    let mut eq_0 = F::zero();
                    let mut eq_2 = F::zero();
                    for x in 0..TERMS {
                        let (e_0, e_1) = (eq_hi[x][2 * y], eq_hi[x][2 * y + 1]);
                        eq_0 += coeff[x] * e_0;
                        eq_2 += coeff[x] * (e_1 + e_1 - e_0);
                    }
                    evals[0] += h_0 * eq_0;
                    evals[1] += (h_1 + h_1 - h_0) * eq_2;
                }
                evals
            }
        }
    }
}

impl<F: JoltField> ProveRounds<F> for RaReductionKernel<F> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
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
        Ok(UnivariatePoly::from_evals_and_hint(
            previous_claim,
            &self.message_evals(),
        ))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for RaReductionKernel<F> {
    type Relation = RamRaClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RamRaClaimReductionOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let Phase::Suffix { h, .. } = &self.phase else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "RAM RA claim-reduction fully bound but still in the prefix phase",
            });
        };
        Ok(RamRaClaimReductionOutputClaims { ram_ra: h[0] })
    }

    /// Pin the factored eq tables to the verifier's scalar path: for each
    /// cycle point, `scale_x · eq_hi_x` fully bound must equal
    /// `derive_output_term` at the bound point.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let Phase::Suffix { eq_hi, scales, .. } = &self.phase else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "RAM RA claim-reduction fully bound but still in the prefix phase",
            });
        };
        let ids = [
            RamRaClaimReductionPublic::EqCycleRaf,
            RamRaClaimReductionPublic::EqCycleReadWrite,
            RamRaClaimReductionPublic::EqCycleValCheck,
        ];
        for (x, public_id) in ids.into_iter().enumerate() {
            let id = JoltDerivedId::from(public_id);
            let got = scales[x] * eq_hi[x][0];
            pin_derived_term(relation, id, input_points, output_points, challenges, got)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module: fail loudly")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::geometry::ram::ram_ra_claim_reduction;
    use jolt_claims::protocols::jolt::relations::ram::{
        RamRaClaimReductionChallenges, RamRaClaimReductionInputClaims,
    };
    use jolt_field::{Fr, Ring};

    use super::super::testing::{
        assert_parity, random_scalars, with_ram_fixture, FixtureShape, RamOp,
    };
    use super::*;
    use crate::reference::views::address_fold;
    use crate::ReferenceBackend;

    fn run_parity(shape: FixtureShape, ops: Vec<RamOp>, seed: u64) {
        with_ram_fixture(shape, ops, |witness| {
            let r_address = random_scalars(shape.log_k(), seed);
            let r_cycle_raf = random_scalars(shape.log_t, seed ^ 0x11);
            let r_cycle_rw = random_scalars(shape.log_t, seed ^ 0x22);
            let r_cycle_val = random_scalars(shape.log_t, seed ^ 0x33);
            let gamma = random_scalars(1, seed ^ 0x44)[0];

            let relation =
                RamRaClaimReduction::<Fr>::new(TraceDimensions::new(shape.log_t), shape.log_k());
            let claims = RamRaClaimReductionInputClaims {
                raf: Fr::from_u64(0),
                read_write: Fr::from_u64(0),
                val_check: Fr::from_u64(0),
            };
            let points = RamRaClaimReductionInputClaims::<Vec<Fr>> {
                raf: [r_address.clone(), r_cycle_raf.clone()].concat(),
                read_write: [r_address.clone(), r_cycle_rw.clone()].concat(),
                val_check: [r_address.clone(), r_cycle_val.clone()].concat(),
            };
            let challenges = RamRaClaimReductionChallenges { gamma };

            let mut reference_session = ProofSession::default();
            let reference = PrepareKernel::<Fr, _>::prepare(
                &ReferenceBackend,
                &mut reference_session,
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .unwrap();
            let mut session = ProofSession::default();
            let optimized = PrepareKernel::<Fr, _>::prepare(
                &OptimizedBackend,
                &mut session,
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .unwrap();

            // The independently folded true input claim:
            // `Σ_j (eq_raf(j) + γ·eq_rw(j) + γ²·eq_val(j)) · ra_folded(j)`.
            let ra_folded =
                address_fold::<Fr>(witness, ram_ra_claim_reduction(), shape.log_t, &r_address)
                    .unwrap();
            let eq_raf = eq_table::<Fr>(&r_cycle_raf);
            let eq_rw = eq_table::<Fr>(&r_cycle_rw);
            let eq_val = eq_table::<Fr>(&r_cycle_val);
            let input_claim = (0..1usize << shape.log_t)
                .map(|j| (eq_raf[j] + gamma * eq_rw[j] + gamma * gamma * eq_val[j]) * ra_folded[j])
                .sum();

            assert_parity(
                reference,
                optimized,
                input_claim,
                &ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
                seed ^ 0x55,
            );
        });
    }

    fn mixed_ops() -> Vec<RamOp> {
        vec![
            RamOp::Write { word: 2, post: 7 },
            RamOp::Read { word: 2 },
            RamOp::Write { word: 5, post: 1 },
            RamOp::None,
            RamOp::Read { word: 5 },
            RamOp::Write { word: 2, post: 3 },
        ]
    }

    #[test]
    fn matches_reference_on_mixed_traffic() {
        run_parity(FixtureShape { log_t: 4, ram_k: 8 }, mixed_ops(), 53);
    }

    #[test]
    fn matches_reference_on_odd_log_t() {
        let mut ops = mixed_ops();
        ops.extend([
            RamOp::Read { word: 2 },
            RamOp::Write { word: 3, post: 9 },
            RamOp::None,
            RamOp::Read { word: 3 },
        ]);
        run_parity(FixtureShape { log_t: 5, ram_k: 8 }, ops, 59);
    }

    #[test]
    fn matches_reference_on_single_round() {
        // `log_T = 1` has no prefix rounds: the kernel starts in the suffix
        // phase off the plain address fold.
        run_parity(
            FixtureShape { log_t: 1, ram_k: 8 },
            vec![RamOp::Write { word: 1, post: 4 }],
            61,
        );
    }
}
