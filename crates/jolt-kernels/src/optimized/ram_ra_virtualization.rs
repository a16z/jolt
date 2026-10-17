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
//! - **Gruen split-eq factoring**: `eq(r_cycle, ·)` is never materialized or
//!   bound; each round emits `s(t) = ℓ(t) · Σ_y E(y) · Π_i ra_i(t, y)` at
//!   a minimal product grid, recovering the missing value from the running
//!   claim. Round polynomials and output claims are
//!   byte-identical (field arithmetic is exact under any regrouping).

use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_claims::protocols::jolt::relations::ram::RamRaVirtualizationOutputClaims;
use jolt_claims::protocols::jolt::{JoltDerivedId, RamRaVirtualizationPublic};
use jolt_field::{Accumulator, JoltField};
use std::ops::Range;

use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6b::ram_ra_virtualization::RamRaVirtualization;
use jolt_witness::{JoltWitnessPlane, TraceRows, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::lazy_ra::{ChunkIndexSource, LazyFoldedRa, MATERIALIZE_BYTES};
use super::ram_trace::RamAddressBundle;
use super::support::{
    accumulate_product_grid, pin_derived_term, product_grid_scratch_len, task_ranges,
    GruenRoundMessage, RoundProgress, MAX_GRID_FACTORS, VISIT_ROWS,
};
use super::OptimizedBackend;
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

impl<F: JoltField> PrepareKernel<F, RamRaVirtualization<F>> for OptimizedBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
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
        if num_committed > MAX_GRID_FACTORS {
            return Err(KernelError::Unsupported {
                reason: "more committed RAM RA chunks than the product grid supports",
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

        let rows = witness.rows().ok_or(WitnessError::UnavailableView {
            label: "stage-6b RAM address rows",
        })?;
        if rows.cycles() != 1 << log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "stage-6b RAM address rows".to_owned(),
                expected: 1 << log_t,
                got: rows.cycles(),
            });
        }
        // One bounded pass checks every access against the RAM domain.
        let ram_k = 1u64 << ram_reduced_address.len();
        let in_domain = |range: &Range<usize>| -> Result<bool, WitnessError> {
            let mut valid = true;
            rows.visit_bundles(
                range.clone(),
                VISIT_ROWS,
                |_, bundles: &[RamAddressBundle]| {
                    valid &= bundles
                        .iter()
                        .all(|row| row.address.0.is_none_or(|address| address < ram_k));
                    Ok(())
                },
            )?;
            Ok(valid)
        };
        let ranges = task_ranges(rows.cycles(), 1);
        #[cfg(feature = "parallel")]
        let valid = ranges
            .par_iter()
            .map(in_domain)
            .try_reduce(|| true, |left, right| Ok(left && right))?;
        #[cfg(not(feature = "parallel"))]
        let valid = ranges.iter().try_fold(true, |all_valid, range| {
            Ok::<_, WitnessError>(in_domain(range)? && all_valid)
        })?;
        if !valid {
            return Err(KernelError::InvariantViolation {
                reason: "RAM access address remapped beyond ram_K",
            });
        }

        // One eq table per committed chunk point (each `2^w` entries); the
        // point-mass fold stays lazy — one table lookup per accessed cycle —
        // instead of materializing `N × T` dense selectors up front.
        let chunk_tables: Vec<Vec<F>> = chunks.iter().map(|chunk| eq_table(chunk)).collect();
        let folded_ra = LazyFoldedRa::new(
            chunk_tables,
            RamAddressChunks {
                rows,
                num_committed,
                committed_chunk_bits,
            },
            MATERIALIZE_BYTES,
        );

        Ok(Box::new(RamRaVirtualizationKernel {
            progress: RoundProgress::new(log_t),
            folded_ra,
            gruen: GruenSplitEqPolynomial::new(ram_reduced_cycle, BindingOrder::LowToHigh),
        }))
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct RamAddressChunks {
    /// The witness plane owns these rows; it reports them itself.
    #[cfg_attr(feature = "allocative", allocative(skip))]
    rows: TraceRows,
    num_committed: usize,
    committed_chunk_bits: usize,
}

impl ChunkIndexSource for RamAddressChunks {
    type Row = RamAddressBundle;

    fn num_columns(&self) -> usize {
        self.num_committed
    }

    fn cycles(&self) -> usize {
        self.rows.cycles()
    }

    #[inline]
    fn index(&self, i: usize, row: &RamAddressBundle) -> Option<usize> {
        let shift = (self.num_committed - 1 - i) * self.committed_chunk_bits;
        let mask = (1u128 << self.committed_chunk_bits) - 1;
        row.address
            .0
            .map(|address| ((u128::from(address) >> shift) & mask) as usize)
    }

    fn visit_blocks(
        &self,
        range: Range<usize>,
        block: usize,
        visit: impl FnMut(usize, &[Self::Row]) -> Result<(), WitnessError>,
    ) -> Result<(), WitnessError> {
        self.rows.visit_bundles(range, block, visit)
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct RamRaVirtualizationKernel<F: JoltField> {
    progress: RoundProgress,
    /// Address-folded committed RA selectors, one per committed chunk:
    /// `folded[i][j] = eq(r_chunk_i, chunk_i(address_j))`, 0 on no-access
    /// cycles, served lazily from the trace rows until the byte-budgeted
    /// materialization instead of `N × T` dense.
    folded_ra: LazyFoldedRa<F, RamAddressChunks>,
    gruen: GruenSplitEqPolynomial<F>,
}

impl<F: JoltField> RamRaVirtualizationKernel<F> {
    /// `s(t) = ℓ(t) · q(t)` with `q(t) = Σ_y E(y) · Π_i ra_i(t, y)`,
    /// evaluated on the grid `[1, …, N−1, ∞]` through
    /// [`accumulate_product_grid`]: `e_in` rides in the first factor, so the
    /// products accumulate unreduced across each inner block, and
    /// [`GruenRoundMessage::checked_toom`] recovers `q(0)` from the running
    /// claim, evaluating it directly only when the linear factor vanishes
    /// at zero.
    fn message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let num_committed = self.folded_ra.num_polys();
        if num_committed < 2 {
            return self.message_few_factors(round, previous_claim);
        }
        let folded_ra = &self.folded_ra;

        struct Scratch<F: JoltField> {
            lanes: Vec<F::Accumulator>,
            pairs: Vec<(F, F)>,
            evals: Vec<F>,
            steps: Vec<F>,
            grid: Vec<F>,
        }

        let block_lanes = folded_ra.try_fold_out_in(
            &self.gruen,
            || Scratch {
                lanes: vec![F::Accumulator::default(); num_committed],
                pairs: vec![(F::zero(), F::zero()); num_committed],
                evals: vec![F::zero(); num_committed],
                steps: vec![F::zero(); num_committed],
                grid: vec![F::zero(); product_grid_scratch_len(num_committed)],
            },
            |scratch, block, row, _x_in, e_in| {
                block.lo_hi_all(row, &mut scratch.pairs);
                for ((&(lo, hi), eval), step) in scratch
                    .pairs
                    .iter()
                    .zip(scratch.evals.iter_mut())
                    .zip(scratch.steps.iter_mut())
                {
                    *eval = hi;
                    *step = hi - lo;
                }
                scratch.evals[0] *= e_in;
                scratch.steps[0] *= e_in;
                accumulate_product_grid(
                    &scratch.evals,
                    &scratch.steps,
                    &mut scratch.lanes,
                    &mut scratch.grid,
                );
            },
            |_x_out, e_out, scratch| {
                let mut out = vec![F::Accumulator::default(); num_committed];
                for (out, lane) in out.iter_mut().zip(scratch.lanes) {
                    out.fmadd(e_out, lane.reduce());
                }
                out
            },
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(b) {
                    a.merge(b);
                }
                a
            },
        )?;

        let q_evals: Vec<F> = block_lanes.into_iter().map(|lane| lane.reduce()).collect();
        // A failed fallback pass is returned; its placeholder never reaches a message.
        let mut failure = None;
        let message = self
            .gruen
            .checked_toom(&q_evals, previous_claim, round, || {
                let q_at_zero = folded_ra.try_fold_out_in(
                    &self.gruen,
                    || {
                        (
                            vec![(F::zero(), F::zero()); num_committed],
                            F::Accumulator::default(),
                        )
                    },
                    |(pairs, sum), block, row, _, weight| {
                        block.lo_hi_all(row, pairs);
                        let value = pairs
                            .iter()
                            .fold(F::one(), |product, pair| product * pair.0);
                        sum.fmadd(weight, value);
                    },
                    |_, weight, (_, sum)| weight * sum.reduce(),
                    |a, b| a + b,
                );
                q_at_zero.unwrap_or_else(|error| {
                    failure = Some(error);
                    F::zero()
                })
            });
        failure.map_or(message, Err)
    }

    /// Fewer than two committed chunks: the grid needs `q(1)` among its
    /// samples, so sample the summand explicitly at `t = 0, …, N + 1`.
    fn message_few_factors(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let num_committed = self.folded_ra.num_polys();
        let points = num_committed + 2;

        let mut q_evals = self.folded_ra.try_fold_out_in(
            &self.gruen,
            || vec![F::zero(); points],
            |acc, block, row, _x_in, e_in| {
                if num_committed == 0 {
                    for value in acc.iter_mut() {
                        *value += e_in;
                    }
                    return;
                }
                let (lo, hi) = block.lo_hi(0, row);
                let step = hi - lo;
                let mut eval = lo;
                for value in acc.iter_mut() {
                    *value += e_in * eval;
                    eval += step;
                }
            },
            |_x_out, e_out, mut acc| {
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
        )?;

        self.gruen
            .checked_round_poly(&mut q_evals, previous_claim, round)
    }

    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        self.gruen.bind(challenge);
        self.folded_ra.bind(challenge)?;
        self.progress.advance();
        Ok(())
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
            self.bind(challenge)?;
        }
        self.message(round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
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
            ram_ra: self.folded_ra.final_values()?,
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
        pin_derived_term(
            relation,
            JoltDerivedId::from(RamRaVirtualizationPublic::EqCycle),
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
    use jolt_witness::TraceBackend;

    use super::super::parity::ExceptionalEq;
    use super::super::testing::{
        assert_parity, ram_fixture, random_scalars, with_ram_fixture, FixtureShape, RamOp,
    };
    use super::*;
    use crate::reference::views::address_fold;
    use crate::ReferenceBackend;

    const CHUNK_BITS: usize = 4;

    fn run_parity(shape: FixtureShape, ops: Vec<RamOp>, seed: u64) {
        run_parity_with(
            shape,
            ops,
            seed,
            None,
            |reference, optimized, claim, inputs| {
                assert_parity(reference, optimized, claim, inputs, seed);
            },
        );
    }

    type KernelBox = Box<dyn SumcheckKernel<Fr, Relation = RamRaVirtualization<Fr>>>;

    fn run_parity_with(
        shape: FixtureShape,
        ops: Vec<RamOp>,
        seed: u64,
        exceptional: Option<ExceptionalEq>,
        finish: impl FnOnce(KernelBox, KernelBox, Fr, &ProverInputs<'_, Fr, RamRaVirtualization<Fr>>),
    ) {
        with_ram_fixture(shape, ops, |witness| {
            let log_k = shape.log_k();
            let num_committed = log_k.div_ceil(CHUNK_BITS);
            let ram_reduced_address = random_scalars(log_k, seed ^ 0xA0DE);
            let ram_reduced_cycle = exceptional.map_or_else(
                || random_scalars(shape.log_t, seed ^ 0xC1C1),
                |case| case.point(shape.log_t, random_scalars(1, seed)[0]),
            );
            let relation = RamRaVirtualization::<Fr>::new(
                RamRaVirtualizationDimensions::new(shape.log_t, num_committed),
                ram_reduced_address.clone(),
                ram_reduced_cycle.clone(),
                CHUNK_BITS,
            );

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
            if exceptional.is_none() {
                assert_ne!(input_claim, Fr::from_u64(0), "degenerate fixture");
            }

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
            let optimized = PrepareKernel::<Fr, _>::prepare(
                &OptimizedBackend,
                &mut ProofSession::default(),
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
    fn parity_four_committed_chunks_past_lazy_materialization() {
        run_parity(
            FixtureShape {
                log_t: 6,
                ram_k: 1 << 16,
            },
            vec![
                RamOp::Write { word: 3, post: 5 },
                RamOp::Write {
                    word: 0xabcd,
                    post: 7,
                },
                RamOp::Read { word: 0xabcd },
                RamOp::None,
                RamOp::Read { word: 3 },
                RamOp::Write {
                    word: 0x1234,
                    post: 2,
                },
            ],
            439,
        );
    }

    #[test]
    fn parity_exceptional_eq_in_lazy_and_dense_virtualization() {
        for ram_k in [256, 1 << 16] {
            for case in ExceptionalEq::ALL {
                run_parity_with(
                    FixtureShape { log_t: 6, ram_k },
                    mixed_ops(),
                    449,
                    Some(case),
                    |reference, optimized, claim, inputs| {
                        assert_parity(reference, optimized, claim, inputs, 449);
                    },
                );
            }
        }
    }

    #[test]
    fn zero_committed_chunks_prove_in_parity_and_fail_closed() {
        let seed = 443;
        run_parity_with(
            FixtureShape { log_t: 3, ram_k: 1 },
            vec![RamOp::None; 3],
            seed,
            None,
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
                    assert!(
                        matches!(
                            &error,
                            SumcheckKernelError::Verifier(
                                VerifierError::StageClaimPublicInputFailed { .. }
                            )
                        ),
                        "expected the fail-closed public-input error, got {error:?}"
                    );
                }
            },
        );
    }

    /// Missing rows, rows of another cycle domain and an access beyond
    /// `ram_K` fail `prepare` typed.
    #[test]
    fn prepare_failures_are_typed() {
        let shape = FixtureShape {
            log_t: 3,
            ram_k: 16,
        };
        let ops = vec![RamOp::Write { word: 9, post: 4 }, RamOp::Read { word: 9 }];
        let resident = ram_fixture(shape, Vec::new(), ops.clone());
        let prepare = |witness: &TraceBackend, log_k: usize| {
            let relation = RamRaVirtualization::<Fr>::new(
                RamRaVirtualizationDimensions::new(shape.log_t, 1),
                random_scalars(log_k, 5),
                random_scalars(shape.log_t, 7),
                CHUNK_BITS,
            );
            let inputs = ProverInputs {
                relation: &relation,
                claims: &Default::default(),
                points: &Default::default(),
                challenges: &NoChallenges::default(),
            };
            PrepareKernel::<Fr, _>::prepare(
                &OptimizedBackend,
                &mut ProofSession::default(),
                witness,
                inputs,
            )
            .map(|_| ())
        };
        // A configured domain shorter than the trace leaves no valid handle.
        let mut shortened = ram_fixture(shape, Vec::new(), ops.clone());
        shortened.config.log_t = 1;
        assert!(matches!(
            prepare(&shortened, 4),
            Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
        ));
        let mut widened = ram_fixture(shape, Vec::new(), ops);
        widened.config.log_t = shape.log_t + 1;
        assert!(matches!(
            prepare(&widened, 4),
            Err(KernelError::TableSizeMismatch {
                expected: 8,
                got: 16,
                ..
            })
        ));
        assert!(matches!(
            prepare(&resident, 2),
            Err(KernelError::InvariantViolation { .. })
        ));
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
