//! The optimized RAM RA virtualization (stage 6b) kernel.
//!
//! The summand is `eq(r_cycle, j) · Π_i ra_i(chunk_i, j)` over the cycle
//! domain. The reference kernel materializes every committed one-hot
//! `(2^w × T)` RAM RA grid through `oracle_table` and address-folds it; this
//! kernel is the RAM twin of
//! [`instruction_ra_virtualization`](super::instruction_ra_virtualization):
//!
//! - **Point-mass address folding**: each committed `RamRa` chunk is a point
//!   mass at that chunk of the cycle's remapped word address (cold on
//!   no-access cycles), so the address-folded value is a single eq-table
//!   lookup — `ra_i(r_chunk_i, j) = eq_table_i[chunk_i(address_j)]`, zero
//!   when the cycle makes no access. `T` lookups per chunk, no grid.
//! - **Session-carried access columns**: the per-cycle addresses come from
//!   `RamAccessColumns::shared`: one typed trace walk shared with the
//!   whole optimized RAM family across the proof session.
//! - **Gruen split-eq factoring**: `eq(r_cycle, ·)` is never materialized or
//!   bound; each round emits `s(t) = ℓ(t) · Σ_y E(y) · Π_i ra_i(t, y)` at
//!   the naive prover's `t = 0..=degree` sample points through the same
//!   `from_evals` constructor, so round polynomials and output claims are
//!   byte-identical (field arithmetic is exact under any regrouping).

use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_claims::protocols::jolt::relations::ram::RamRaVirtualizationOutputClaims;
use jolt_claims::protocols::jolt::{JoltDerivedId, RamRaVirtualizationPublic};
use jolt_field::JoltField;
use std::sync::Arc;

use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6b::ram_ra_virtualization::RamRaVirtualization;
use jolt_witness::JoltWitnessPlane;

use super::lazy_ra::{ChunkIndexSource, LazyFoldedRa};
use super::ram_trace::{RamAccessColumns, NO_ACCESS};
use super::support::{pin_derived_term, GruenRoundMessage, RoundProgress};
use super::OptimizedBackend;
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

impl<F: JoltField> PrepareKernel<F, RamRaVirtualization<F>> for OptimizedBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamRaVirtualization<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamRaVirtualization<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let log_t = dimensions.log_t();
        let ram_reduced_address = relation.ram_reduced_address();
        let committed_chunk_bits = relation.committed_chunk_bits();
        let chunks = committed_address_chunks(ram_reduced_address, committed_chunk_bits);
        let num_committed = dimensions.num_committed_ra_polys();
        if chunks.len() != num_committed {
            return Err(KernelError::InvariantViolation {
                reason: "RAM address chunk count disagrees with the committed RA count",
            });
        }
        if committed_chunk_bits == 0 || committed_chunk_bits > 32 {
            return Err(KernelError::Unsupported {
                reason: "committed RAM RA chunk width outside the supported one-hot range",
            });
        }
        let ram_reduced_cycle = relation.ram_reduced_cycle();
        if ram_reduced_cycle.len() != log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "RAM RA reduced cycle point".to_owned(),
                expected: log_t,
                got: ram_reduced_cycle.len(),
            });
        }

        let columns = RamAccessColumns::shared(session, witness, log_t)?;
        // This is the RAM family's last consumer: remove the session's copy
        // so the columns free at the lazy fold's materialization instead of
        // living to the end of the proof.
        let _ = session.take::<Arc<RamAccessColumns>>();
        columns.validate_addresses(1usize << ram_reduced_address.len())?;

        // One eq table per committed chunk point (each `2^w` entries); the
        // point-mass fold stays lazy — one table lookup per accessed cycle —
        // instead of materializing `N × T` dense selectors up front.
        let chunk_tables: Vec<Vec<F>> = chunks.iter().map(|chunk| eq_table(chunk)).collect();
        let folded_ra = LazyFoldedRa::new(
            chunk_tables,
            RamAddressChunks {
                columns,
                num_committed,
                committed_chunk_bits,
            },
        );

        Ok(Box::new(RamRaVirtualizationKernel {
            progress: RoundProgress::new(log_t),
            folded_ra,
            gruen: GruenSplitEqPolynomial::new(ram_reduced_cycle, BindingOrder::LowToHigh),
        }))
    }
}

/// Lazy-RA index source: chunk `i` of the per-cycle remapped RAM address,
/// cold on no-access cycles, off the session-shared access columns.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct RamAddressChunks {
    columns: Arc<RamAccessColumns>,
    num_committed: usize,
    committed_chunk_bits: usize,
}

impl ChunkIndexSource for RamAddressChunks {
    fn num_polys(&self) -> usize {
        self.num_committed
    }

    fn cycles(&self) -> usize {
        self.columns.addresses.len()
    }

    #[inline]
    fn index(&self, i: usize, j: usize) -> Option<usize> {
        let address = self.columns.addresses[j];
        if address == NO_ACCESS {
            return None;
        }
        let shift = (self.num_committed - 1 - i) * self.committed_chunk_bits;
        let mask = (1u128 << self.committed_chunk_bits) - 1;
        Some(((u128::from(address) >> shift) & mask) as usize)
    }
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct RamRaVirtualizationKernel<F: JoltField> {
    progress: RoundProgress,
    /// Address-folded committed RA selectors, one per committed chunk:
    /// `folded[i][j] = eq(r_chunk_i, chunk_i(address_j))`, 0 on no-access
    /// cycles, served lazily off the shared columns for the first four
    /// binds instead of `N × T` dense.
    folded_ra: LazyFoldedRa<F, RamAddressChunks>,
    gruen: GruenSplitEqPolynomial<F>,
}

impl<F: JoltField> RamRaVirtualizationKernel<F> {
    /// `s(t) = ℓ(t) · q(t)` at the naive prover's sample points, with
    /// `q(t) = Σ_y E(y) · Π_i ra_i(t, y)`.
    fn message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        // The relation degree: one eq factor plus the committed-RA product.
        let num_committed = self.folded_ra.num_polys();
        let points = num_committed + 2;

        let mut q_evals = self.gruen.par_fold_out_in(
            || {
                (
                    vec![F::zero(); points],
                    vec![F::zero(); num_committed],
                    vec![F::zero(); num_committed],
                )
            },
            |(acc, evals, steps), row, _x_in, e_in| {
                if num_committed == 0 {
                    for value in acc.iter_mut() {
                        *value += e_in;
                    }
                    return;
                }
                for position in 0..num_committed {
                    let (lo, hi) = self.folded_ra.lo_hi(position, row);
                    evals[position] = lo;
                    steps[position] = hi - lo;
                }
                for value in acc.iter_mut() {
                    let mut product = evals[0];
                    for eval in &evals[1..] {
                        product *= *eval;
                    }
                    *value += e_in * product;
                    for (eval, step) in evals.iter_mut().zip(steps.iter()) {
                        *eval += *step;
                    }
                }
            },
            |_x_out, e_out, (mut acc, _, _)| {
                for value in &mut acc {
                    *value *= e_out;
                }
                acc
            },
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(&b) {
                    *a += *b;
                }
                a
            },
        );

        self.gruen
            .checked_round_poly(&mut q_evals, previous_claim, round)
    }

    fn bind(&mut self, challenge: F) {
        self.gruen.bind(challenge);
        self.folded_ra.bind(challenge);
        self.progress.advance();
    }
}

impl<F: JoltField> ProveRounds<F> for RamRaVirtualizationKernel<F> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
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
        self.message(round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for RamRaVirtualizationKernel<F> {
    type Relation = RamRaVirtualization<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RamRaVirtualizationOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        Ok(RamRaVirtualizationOutputClaims {
            ram_ra: self.folded_ra.final_values(),
        })
    }

    /// The Gruen scalar after full binding is the bound `EqCycle` value; pin
    /// it to the verifier's `derive_output_term`, exactly as the naive tier's
    /// materialized eq table is pinned.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let id = JoltDerivedId::from(RamRaVirtualizationPublic::EqCycle);
        pin_derived_term(
            relation,
            id,
            input_points,
            output_points,
            challenges,
            self.gruen.current_scalar(),
        )
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::ram::{
        committed_ram_ra, RamRaVirtualizationDimensions,
    };
    use jolt_claims::protocols::jolt::relations::ram::RamRaVirtualizationInputClaims;
    use jolt_claims::NoChallenges;
    use jolt_field::{Fr, Ring};
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::VerifierError;

    use super::super::testing::{
        assert_parity, random_scalars, with_ram_fixture, FixtureShape, RamOp,
    };
    use super::*;
    use crate::reference::views::address_fold;
    use crate::ReferenceBackend;

    /// The fixture's one-hot chunk width (`JoltOneHotConfig.log_k_chunk`).
    const CHUNK_BITS: usize = 4;

    fn run_parity(shape: FixtureShape, ops: Vec<RamOp>, seed: u64) {
        run_parity_with(shape, ops, seed, |reference, optimized, claim, inputs| {
            assert_parity(reference, optimized, claim, inputs, seed);
        });
    }

    type KernelBox = Box<dyn SumcheckKernel<Fr, Relation = RamRaVirtualization<Fr>>>;

    fn run_parity_with(
        shape: FixtureShape,
        ops: Vec<RamOp>,
        seed: u64,
        finish: impl FnOnce(KernelBox, KernelBox, Fr, &ProverInputs<'_, Fr, RamRaVirtualization<Fr>>),
    ) {
        with_ram_fixture(shape, ops, |witness| {
            let log_k = shape.log_k();
            let num_committed = log_k.div_ceil(CHUNK_BITS);
            let ram_reduced_address = random_scalars(log_k, seed ^ 0xA0DE);
            let ram_reduced_cycle = random_scalars(shape.log_t, seed ^ 0xC1C1);
            let relation = RamRaVirtualization::<Fr>::new(
                RamRaVirtualizationDimensions::new(shape.log_t, num_committed),
                ram_reduced_address.clone(),
                ram_reduced_cycle.clone(),
                CHUNK_BITS,
            );

            // The honest reduced claim: the eq-weighted sum of the committed
            // chunk products, straight off the oracle grids.
            let chunks = committed_address_chunks(&ram_reduced_address, CHUNK_BITS);
            let folded: Vec<Vec<Fr>> = chunks
                .iter()
                .enumerate()
                .map(|(index, chunk)| {
                    address_fold(witness, committed_ram_ra(index), shape.log_t, chunk).unwrap()
                })
                .collect();
            let eq_cycle = eq_table(&ram_reduced_cycle);
            let input_claim: Fr = (0..1usize << shape.log_t)
                .map(|j| {
                    folded
                        .iter()
                        .fold(eq_cycle[j], |product, table| product * table[j])
                })
                .sum();
            assert_ne!(input_claim, Fr::from_u64(0), "degenerate fixture");

            let claims = RamRaVirtualizationInputClaims {
                ram_ra_reduced: input_claim,
            };
            let points = RamRaVirtualizationInputClaims::<Vec<Fr>>::default();
            let challenges = NoChallenges::default();
            let inputs = ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

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
            // Pre-warm the session so the kernel exercises the shared-columns
            // reclaim path (the real pipeline parks them in stage 2).
            let mut session = ProofSession::default();
            let _ = RamAccessColumns::shared::<Fr>(&mut session, witness, shape.log_t).unwrap();
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

            finish(reference, optimized, input_claim, &inputs);
        });
    }

    fn mixed_ops() -> Vec<RamOp> {
        vec![
            RamOp::Write { word: 3, post: 5 },
            RamOp::Read { word: 3 },
            RamOp::None,
            RamOp::Write { word: 9, post: 4 },
            RamOp::Read { word: 9 },
            RamOp::None,
            RamOp::Write { word: 3, post: 8 },
            RamOp::Read { word: 15 },
        ]
    }

    #[test]
    fn parity_single_committed_chunk() {
        run_parity(
            FixtureShape {
                log_t: 4,
                ram_k: 16,
            },
            mixed_ops(),
            431,
        );
    }

    #[test]
    fn parity_two_committed_chunks() {
        // log_k = 8 with 4-bit chunks: two committed RA polynomials, hot
        // words on both sides of the chunk boundary.
        run_parity(
            FixtureShape {
                log_t: 4,
                ram_k: 256,
            },
            vec![
                RamOp::Write { word: 3, post: 5 },
                RamOp::Write { word: 200, post: 7 },
                RamOp::Read { word: 200 },
                RamOp::None,
                RamOp::Read { word: 3 },
                RamOp::Write { word: 63, post: 2 },
            ],
            433,
        );
    }

    #[test]
    fn zero_committed_chunks_prove_in_parity_and_fail_closed() {
        let seed = 443;
        run_parity_with(
            FixtureShape { log_t: 3, ram_k: 1 },
            vec![RamOp::None; 3],
            seed,
            |mut reference, mut optimized, input_claim, inputs| {
                let challenges = super::super::testing::drive_parity_rounds(
                    reference.as_mut(),
                    optimized.as_mut(),
                    input_claim,
                    inputs,
                    seed,
                );
                let output_points = inputs
                    .relation
                    .derive_opening_points(&challenges, inputs.points)
                    .unwrap();
                for kernel in [reference, optimized] {
                    let error = kernel
                        .validate_derived_tables(
                            inputs.relation,
                            inputs.points,
                            &output_points,
                            inputs.challenges,
                        )
                        .unwrap_err();
                    assert!(matches!(
                        error,
                        SumcheckKernelError::Verifier(
                            VerifierError::StageClaimPublicInputFailed { .. }
                        )
                    ));
                }
            },
        );
    }

    #[test]
    fn parity_padded_chunk_count() {
        // log_k = 6 with 4-bit chunks: `committed_address_chunks` front-pads
        // the reduced address, so chunk 0 spans only two real address bits.
        run_parity(
            FixtureShape {
                log_t: 3,
                ram_k: 64,
            },
            vec![
                RamOp::Write { word: 33, post: 9 },
                RamOp::Read { word: 33 },
                RamOp::Write { word: 5, post: 1 },
            ],
            439,
        );
    }
}
