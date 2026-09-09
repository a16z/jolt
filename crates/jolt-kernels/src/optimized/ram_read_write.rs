//! The optimized RAM read/write-checking (stage 2) kernel: the legacy
//! `RamReadWriteCheckingProver` ported onto the kernel seam.
//!
//! Same summand, variable order, and binding order as the reference kernel —
//! `eq(τ_low, j) · ra(k,j) · (val(k,j) + γ·(val(k,j) + inc(j)))` over the
//! joint `(address ‖ cycle)` domain, bound low-to-high — but the `(K × T)`
//! `ra`/`val` grids are never materialized:
//!
//! - **Sparse read-write matrix**: one entry per RAM access;
//!   `prev_val`/`next_val` checkpoints recover every implicit coefficient
//!   (see `rw_matrix`).
//! - **Gruen split-eq + Dao–Thaler factoring** for the `log_T` cycle rounds:
//!   the eq factor stays in `O(√T)` tables and each cubic round message is
//!   reconstructed from the quadratic factor's `[q(0), q_∞]` plus the
//!   running claim ([`GruenSplitEqPolynomial::gruen_poly_deg_3`]).
//! - **Hint-based quadratic address rounds** on the address-major matrix
//!   against the bound `val_init` column (`s(1)` recovered from the claim).
//!
//! `val_init` is reconstructed from the trace and the `RamValFinal` oracle
//! (the witness plane does not expose the initial RAM state): an accessed
//! address's initial value is its first access's pre-value, an untouched
//! address's final value IS its initial value. Honest-prover data path; with
//! hint-anchored round messages a divergent witness surfaces at the driver's
//! final-claim check rather than a per-round check.
//!
//! Only the default read-write config (phase 1 = all cycle rounds) is
//! supported, like the reference kernel. The legacy phase-2/phase-3 split of
//! the address rounds is a data-structure choice with no effect on the round
//! polynomials, so this kernel binds all `log_K` address rounds sparse.

use jolt_claims::protocols::jolt::geometry::ram::ram_inc;
use jolt_claims::protocols::jolt::{
    JoltDerivedId, JoltPolynomialId, JoltVirtualPolynomial, RamReadWritePublic,
};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputClaims,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::ram_read_write_checking::{
    RamReadWriteChecking, RamReadWriteOutputClaims,
};
use jolt_witness::JoltWitnessPlane;

use super::ram_trace::RamAccessColumns;
use super::rw_matrix::{
    round0_bind, round0_quadratic_coefficients, AddressMajorMatrix, CycleMajorMatrix,
};
use super::support::pin_derived_term_if_derived;
use super::OptimizedBackend;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Raw columns, cycle matrix, address matrix, then bound values.
/// The first bind creates the first sparse entry vector at half size.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum Phase<F: JoltField> {
    Round0 {
        columns: RamAccessColumns,
        gruen: GruenSplitEqPolynomial<F>,
    },
    Cycle {
        matrix: CycleMajorMatrix<F>,
        gruen: GruenSplitEqPolynomial<F>,
    },
    Address {
        matrix: AddressMajorMatrix<F>,
        /// The cycle-eq factor fully bound by phase 1: a length-1 table.
        merged_eq: Polynomial<F>,
    },
    Done {
        merged_eq: Polynomial<F>,
        #[cfg_attr(feature = "allocative", allocative(skip))]
        final_ra: F,
        #[cfg_attr(feature = "allocative", allocative(skip))]
        final_val: F,
    },
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct RamReadWriteKernel<F: JoltField> {
    phase: Option<Phase<F>>,
    /// The committed per-cycle increment column, bound alongside phase 1;
    /// a scalar once every cycle variable is bound.
    inc: Polynomial<F>,
    /// The initial-RAM column over addresses, bound alongside phase 2.
    val_init: Polynomial<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gamma: F,
    log_t: usize,
    log_k: usize,
}

impl<F: JoltField> Phase<F> {
    /// The error for a bind or round message arriving outside its phase.
    fn error() -> SumcheckError<F> {
        SumcheckError::MissingEvaluationSource {
            kind: "RAM read-write phase state",
        }
    }
}

/// Cycle bind that triggers the late allocator purge.
const LATE_PURGE_CYCLE_ROUNDS: usize = 6;

impl<F: JoltField> RamReadWriteKernel<F> {
    /// Bind the challenge of `round` (0-indexed over the member's window),
    /// advancing the phase machine at the boundaries.
    fn ingest(&mut self, r: F, round: usize) -> Result<(), SumcheckError<F>> {
        if round < self.log_t {
            if matches!(self.phase, Some(Phase::Round0 { .. })) {
                // Create the first matrix already bound at half size.
                let Some(Phase::Round0 { columns, mut gruen }) = self.phase.take() else {
                    return Err(Phase::error());
                };
                let matrix = round0_bind(&columns, r);
                drop(columns);
                gruen.bind(r);
                self.phase = Some(Phase::Cycle { matrix, gruen });
            } else {
                let Some(Phase::Cycle { matrix, gruen }) = &mut self.phase else {
                    return Err(Phase::error());
                };
                matrix.bind(r);
                gruen.bind(r);
            }
            // Avoid a fresh half-size increment table each round.
            let _ = self.inc.bind_low_to_high_in_place(r);
            if round == self.log_t - 1 {
                let Some(Phase::Cycle { matrix, gruen }) = self.phase.take() else {
                    return Err(Phase::error());
                };
                self.phase = Some(Phase::Address {
                    matrix: matrix.into_address_major(),
                    merged_eq: gruen.merge(),
                });
                if self.log_k == 0 {
                    self.finalize()?;
                }
            }
            // Purge after raw columns, late bind tails, and the cycle matrix.
            if round == 0 || round == LATE_PURGE_CYCLE_ROUNDS || round == self.log_t - 1 {
                crate::mem::purge_retained_memory(self.log_t);
            }
        } else {
            let Some(Phase::Address { matrix, .. }) = &mut self.phase else {
                return Err(Phase::error());
            };
            matrix.bind(r, &mut self.val_init);
            if round == self.log_t + self.log_k - 1 {
                self.finalize()?;
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), SumcheckError<F>> {
        let Some(Phase::Address { matrix, merged_eq }) = self.phase.take() else {
            return Err(Phase::error());
        };
        let (final_ra, final_val) = matrix.final_values(&self.val_init);
        self.phase = Some(Phase::Done {
            merged_eq,
            final_ra,
            final_val,
        });
        Ok(())
    }

    /// Phase-1 cubic round message via Gruen: the quadratic factor's
    /// `[q(0), q_∞]` over the sparse matrix, lifted by the current linear eq
    /// factor and the running claim.
    fn cycle_round_message(
        &self,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let (gruen, [q_0, q_infty]) = match &self.phase {
            Some(Phase::Round0 { columns, gruen }) => {
                let e_in = gruen.e_in_current();
                let e_out = gruen.e_out_current();
                let in_bits = e_in.len().trailing_zeros() as usize;
                let in_mask = e_in.len() - 1;
                (
                    gruen,
                    round0_quadratic_coefficients(
                        columns,
                        |pair| e_out[pair >> in_bits] * e_in[pair & in_mask],
                        &self.inc,
                        self.gamma,
                    ),
                )
            }
            Some(Phase::Cycle { matrix, gruen }) => {
                let e_in = gruen.e_in_current();
                let e_out = gruen.e_out_current();
                let in_bits = e_in.len().trailing_zeros() as usize;
                let in_mask = e_in.len() - 1;
                (
                    gruen,
                    matrix.quadratic_coefficients(
                        |pair| e_out[pair >> in_bits] * e_in[pair & in_mask],
                        &self.inc,
                        self.gamma,
                    ),
                )
            }
            _ => return Err(Phase::error()),
        };
        Ok(gruen.gruen_poly_deg_3(q_0, q_infty, previous_claim))
    }

    /// Phase-2 quadratic round message: `[s(0), s(2)]` over the sparse
    /// matrix (cycle-bound `eq`/`inc` scalars), `s(1)` from the claim.
    fn address_round_message(
        &self,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let Some(Phase::Address { matrix, merged_eq }) = &self.phase else {
            return Err(Phase::error());
        };
        let evals = matrix.address_round_evals(&self.val_init, &self.inc, merged_eq, self.gamma);
        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }
}

impl<F: JoltField> ProveRounds<F> for RamReadWriteKernel<F> {
    fn num_rounds(&self) -> usize {
        self.log_t + self.log_k
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.ingest(challenge, round - 1)?;
        }
        if round < self.log_t {
            self.cycle_round_message(previous_claim)
        } else {
            self.address_round_message(previous_claim)
        }
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.ingest(bind, self.num_rounds() - 1)
    }
}

impl<F: JoltField> SumcheckKernel<F> for RamReadWriteKernel<F> {
    type Relation = RamReadWriteChecking<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        let Some(Phase::Done {
            final_ra,
            final_val,
            ..
        }) = &self.phase
        else {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.num_rounds(),
            });
        };
        Ok(RamReadWriteOutputClaims {
            val: *final_val,
            ra: *final_ra,
            inc: self.inc.evals()[0],
        })
    }

    /// The hand-maintained cycle-eq factor must equal the verifier's
    /// `EqCycle` scalar at the bound point — the same cross-check the naive
    /// tier runs on its tiled eq table.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        let Some(Phase::Done { merged_eq, .. }) = &self.phase else {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.num_rounds(),
            });
        };
        let id = JoltDerivedId::from(RamReadWritePublic::EqCycle);
        pin_derived_term_if_derived(
            relation,
            id,
            input_points,
            output_points,
            challenges,
            merged_eq.evals()[0],
        )
    }
}

impl<F: JoltField> PrepareKernel<F, RamReadWriteChecking<F>> for OptimizedBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamReadWriteChecking<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamReadWriteChecking<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let log_t = dimensions.log_t();
        let log_k = relation.ram_log_k();
        let tau_low = relation.product_tau_low();
        if dimensions.phase1_num_rounds() != log_t {
            return Err(KernelError::Unsupported {
                reason: "optimized RAM read-write checking supports only the default \
                         read-write config (phase 1 = all cycle rounds)",
            });
        }
        if log_t == 0 || dimensions.log_k() != log_k || tau_low.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "RAM read-write checking geometry is inconsistent",
            });
        }
        // Sparse matrix indices are u32.
        if log_t > 32 || log_k > 32 {
            return Err(KernelError::Unsupported {
                reason: "optimized RAM read-write checking packs indices as u32 \
                         (log_T, log_K ≤ 32)",
            });
        }

        let columns = RamAccessColumns::collect_full(session, witness, log_t)?;
        super::ram_trace::validate_addresses(&columns.addresses, 1usize << log_k)?;

        let inc = Polynomial::new(witness.oracle_table(ram_inc().polynomial_id())?);
        let val_final = witness.oracle_table(JoltPolynomialId::Virtual(
            JoltVirtualPolynomial::RamValFinal,
        ))?;
        if inc.len() != 1usize << log_t || val_final.len() != 1usize << log_k {
            return Err(KernelError::InvariantViolation {
                reason: "RAM read-write witness tables disagree with the relation geometry",
            });
        }
        let val_init = Polynomial::new(columns.reconstruct_val_init(val_final));

        Ok(Box::new(RamReadWriteKernel {
            phase: Some(Phase::Round0 {
                columns,
                gruen: GruenSplitEqPolynomial::new(tau_low, BindingOrder::LowToHigh),
            }),
            inc,
            val_init,
            gamma: inputs.challenges.gamma,
            log_t,
            log_k,
        }))
    }
}

#[cfg(all(feature = "test-utils", feature = "metal", target_os = "macos"))]
#[derive(Clone, Debug)]
pub(crate) struct OptimizedRamReadWriteEvalResult {
    pub(crate) round_polynomials: Vec<Vec<jolt_field::Prime128OffsetA7F7>>,
    pub(crate) final_claim: jolt_field::Prime128OffsetA7F7,
    pub(crate) output_claims: Vec<jolt_field::Prime128OffsetA7F7>,
}

#[cfg(all(feature = "test-utils", feature = "metal", target_os = "macos"))]
#[derive(Clone, Debug)]
pub(crate) struct OptimizedRamReadWriteEvalSample {
    pub(crate) result: OptimizedRamReadWriteEvalResult,
    pub(crate) member_wall: std::time::Duration,
    pub(crate) prepare_wall: std::time::Duration,
    pub(crate) rounds_wall: std::time::Duration,
    pub(crate) finish_wall: std::time::Duration,
    pub(crate) output_wall: std::time::Duration,
}

/// The CPU oracle's inputs: the same relation geometry and challenges the
/// Metal arm consumes. The kernel collects its own RAM columns from the
/// witness, so the oracle never sees the device-side record model.
#[cfg(all(feature = "test-utils", feature = "metal", target_os = "macos"))]
pub(crate) struct OptimizedRamReadWriteEvalInputs<'a> {
    pub(crate) witness: &'a dyn JoltWitnessPlane<jolt_field::Prime128OffsetA7F7>,
    pub(crate) log_t: usize,
    pub(crate) log_k: usize,
    pub(crate) tau_low: &'a [jolt_field::Prime128OffsetA7F7],
    pub(crate) gamma: jolt_field::Prime128OffsetA7F7,
    pub(crate) input_values:
        &'a jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteInputClaims<
            jolt_field::Prime128OffsetA7F7,
        >,
    pub(crate) input_claim: jolt_field::Prime128OffsetA7F7,
    pub(crate) challenges: &'a [jolt_field::Prime128OffsetA7F7],
}

#[cfg(all(feature = "test-utils", feature = "metal", target_os = "macos"))]
pub(crate) fn run_optimized_ram_read_write_eval(
    inputs: OptimizedRamReadWriteEvalInputs<'_>,
) -> Result<OptimizedRamReadWriteEvalSample, String> {
    use std::time::Instant;

    use jolt_claims::protocols::jolt::geometry::dimensions::ReadWriteDimensions;
    use jolt_claims::OutputClaims as _;
    use jolt_verifier::stages::relations::ConcreteSumcheck as _;
    use jolt_verifier::stages::stage2::ram_read_write_checking::{
        RamReadWriteChallenges, RamReadWriteInputClaims,
    };

    type F = jolt_field::Prime128OffsetA7F7;

    if inputs.tau_low.len() != inputs.log_t
        || inputs.challenges.len() != inputs.log_t + inputs.log_k
    {
        return Err("RAM read-write evaluator challenge geometry is invalid".to_owned());
    }
    let dimensions =
        ReadWriteDimensions::new(inputs.log_t, inputs.log_k, inputs.log_t, inputs.log_k);
    let relation = RamReadWriteChecking::new(dimensions, inputs.log_k, inputs.tau_low.to_vec());
    let challenge_values = RamReadWriteChallenges {
        gamma: inputs.gamma,
    };
    let input_points = RamReadWriteInputClaims::<Vec<F>>::default();

    let member_started = Instant::now();
    let prepare_started = Instant::now();
    let mut session = ProofSession::default();
    let mut kernel = OptimizedBackend
        .prepare(
            &mut session,
            inputs.witness,
            ProverInputs {
                relation: &relation,
                claims: inputs.input_values,
                points: &input_points,
                challenges: &challenge_values,
            },
        )
        .map_err(|error| error.to_string())?;
    let prepare_wall = prepare_started.elapsed();

    let rounds_started = Instant::now();
    let mut bind = None;
    let mut previous_claim = inputs.input_claim;
    let mut round_polynomials = Vec::with_capacity(inputs.challenges.len());
    for (round, &challenge) in inputs.challenges.iter().enumerate() {
        let polynomial = kernel
            .prove_round(bind, round, previous_claim)
            .map_err(|error| error.to_string())?;
        previous_claim = polynomial.evaluate(challenge);
        round_polynomials.push(polynomial.coefficients().to_vec());
        bind = Some(challenge);
    }
    let rounds_wall = rounds_started.elapsed();

    let final_challenge = inputs
        .challenges
        .last()
        .copied()
        .ok_or_else(|| "RAM read-write evaluator has no terminal challenge".to_owned())?;
    let finish_started = Instant::now();
    kernel
        .finish_rounds(final_challenge)
        .map_err(|error| error.to_string())?;
    let finish_wall = finish_started.elapsed();

    let output_started = Instant::now();
    let output_points = relation
        .derive_opening_points(inputs.challenges, &input_points)
        .map_err(|error| error.to_string())?;
    let output_claims = kernel
        .output_claims(inputs.input_values)
        .map_err(|error| error.to_string())?;
    kernel
        .validate_derived_tables(&relation, &input_points, &output_points, &challenge_values)
        .map_err(|error| error.to_string())?;
    let output_claims = output_claims.opening_values();
    let output_wall = output_started.elapsed();

    Ok(OptimizedRamReadWriteEvalSample {
        result: OptimizedRamReadWriteEvalResult {
            round_polynomials,
            final_claim: previous_claim,
            output_claims,
        },
        member_wall: member_started.elapsed(),
        prepare_wall,
        rounds_wall,
        finish_wall,
        output_wall,
    })
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::ReadWriteDimensions;
    use jolt_claims::protocols::jolt::geometry::ram::{ram_ra, ram_val};
    use jolt_field::{Fr, Ring};
    use jolt_poly::EqPolynomial;
    use jolt_verifier::stages::stage2::ram_read_write_checking::{
        RamReadWriteChallenges, RamReadWriteInputClaims,
    };

    use super::super::testing::{
        assert_parity, random_scalars, with_ram_fixture, with_ram_fixture_init, FixtureShape, RamOp,
    };
    use super::*;
    use crate::ReferenceBackend;

    /// The independently computed true input claim:
    /// `Σ_{k,j} eq(τ_low, j) · ra(k,j) · (val(k,j) + γ·(val(k,j) + inc(j)))`
    /// over the dense oracle grids.
    fn dense_input_claim(
        witness: &dyn JoltWitnessPlane<Fr>,
        tau_low: &[Fr],
        gamma: Fr,
        ram_k: usize,
    ) -> Fr {
        let cycles = 1usize << tau_low.len();
        let eq = EqPolynomial::new(tau_low.to_vec()).evaluations();
        let ra: Vec<Fr> = witness.oracle_table(ram_ra().polynomial_id()).unwrap();
        let val: Vec<Fr> = witness.oracle_table(ram_val().polynomial_id()).unwrap();
        let inc: Vec<Fr> = witness.oracle_table(ram_inc().polynomial_id()).unwrap();
        let mut claim = Fr::from_u64(0);
        for k in 0..ram_k {
            for j in 0..cycles {
                let index = (k << tau_low.len()) | j;
                claim += eq[j] * ra[index] * (val[index] + gamma * (val[index] + inc[j]));
            }
        }
        claim
    }

    fn run_parity(shape: FixtureShape, ops: Vec<RamOp>) {
        run_parity_init(shape, Vec::new(), ops);
    }

    fn run_parity_init(shape: FixtureShape, init_words: Vec<u64>, ops: Vec<RamOp>) {
        with_ram_fixture_init(shape, init_words, ops, |witness| {
            let tau_low = random_scalars(shape.log_t, 17);
            let gamma = random_scalars(1, 23)[0];
            let relation = RamReadWriteChecking::<Fr>::new(
                ReadWriteDimensions::new(shape.log_t, shape.log_k(), shape.log_t, shape.log_k()),
                shape.log_k(),
                tau_low.clone(),
            );
            let claims = RamReadWriteInputClaims {
                ram_read_value: Fr::from_u64(0),
                ram_write_value: Fr::from_u64(0),
            };
            let points = RamReadWriteInputClaims::<Vec<Fr>>::default();
            let challenges = RamReadWriteChallenges { gamma };

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

            let input_claim = dense_input_claim(witness, &tau_low, gamma, shape.ram_k);
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
                71,
            );
        });
    }

    #[test]
    fn matches_reference_on_mixed_traffic() {
        run_parity(
            FixtureShape {
                log_t: 4,
                ram_k: 16,
            },
            vec![
                RamOp::Write { word: 3, post: 5 },
                RamOp::Read { word: 3 },
                RamOp::Write { word: 3, post: 9 },
                RamOp::Read { word: 7 },
                RamOp::None,
                RamOp::Write { word: 4, post: 2 },
                RamOp::Read { word: 3 },
                RamOp::Write { word: 7, post: 6 },
                RamOp::Read { word: 4 },
                RamOp::Write { word: 12, post: 1 },
            ],
        );
    }

    #[test]
    fn matches_reference_on_sparse_traffic() {
        // Long no-access gaps and a single hot address: exercises the
        // implicit-entry checkpoint paths on both matrix orientations.
        run_parity(
            FixtureShape { log_t: 5, ram_k: 8 },
            vec![
                RamOp::None,
                RamOp::None,
                RamOp::Write { word: 5, post: 11 },
                RamOp::None,
                RamOp::None,
                RamOp::None,
                RamOp::Read { word: 5 },
                RamOp::None,
                RamOp::None,
                RamOp::None,
                RamOp::None,
                RamOp::Write { word: 5, post: 3 },
            ],
        );
    }

    #[test]
    fn matches_reference_without_ram_traffic() {
        run_parity(FixtureShape { log_t: 3, ram_k: 4 }, vec![RamOp::None; 3]);
    }

    /// Nonzero `val_init` with reads BEFORE the first write: the optimized
    /// `val_init` reconstruction must recover a read-first word's initial
    /// value from its first access's pre-value, a never-accessed nonzero
    /// word's from the final state, and stay in parity with the reference
    /// val grid through both phases.
    #[test]
    fn matches_reference_on_read_before_write_with_nonzero_val_init() {
        run_parity_init(
            FixtureShape {
                log_t: 4,
                ram_k: 16,
            },
            // Words 2..5 start at 7, 5, 11; word 3 is never accessed.
            vec![7, 5, 11],
            vec![
                RamOp::Read { word: 2 },
                RamOp::Write { word: 2, post: 9 },
                RamOp::Read { word: 2 },
                RamOp::Read { word: 4 },
                RamOp::None,
                RamOp::Write { word: 6, post: 3 },
                RamOp::Read { word: 6 },
            ],
        );
    }

    /// A non-default phase split (phase 1 shorter than the cycle rounds) is
    /// rejected as `Unsupported` instead of misproving.
    #[test]
    fn rejects_non_default_phase_split() {
        let shape = FixtureShape { log_t: 3, ram_k: 4 };
        with_ram_fixture(shape, vec![RamOp::None; 3], |witness| {
            let tau_low = random_scalars(shape.log_t, 17);
            let relation = RamReadWriteChecking::<Fr>::new(
                ReadWriteDimensions::new(
                    shape.log_t,
                    shape.log_k(),
                    shape.log_t - 1,
                    shape.log_k() + 1,
                ),
                shape.log_k(),
                tau_low,
            );
            let claims = RamReadWriteInputClaims {
                ram_read_value: Fr::from_u64(0),
                ram_write_value: Fr::from_u64(0),
            };
            let points = RamReadWriteInputClaims::<Vec<Fr>>::default();
            let challenges = RamReadWriteChallenges {
                gamma: random_scalars(1, 23)[0],
            };
            let result = PrepareKernel::<Fr, _>::prepare(
                &OptimizedBackend,
                &mut ProofSession::default(),
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            );
            assert!(matches!(
                result.map(|_| ()),
                Err(KernelError::Unsupported { .. })
            ));
        });
    }
}
