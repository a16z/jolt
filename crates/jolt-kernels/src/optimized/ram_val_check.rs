//! Optimized RAM value check (stage 4).
//!
//! Summand: `inc(j) · ra(j) · (LT(j, r_cycle) + γ)`.
//! [`LazyFoldedRa`] keeps `ra` lazy until dense storage is cheaper or four binds
//! have completed; [`SplitLt`] stores
//! `LT + γ` in ~√T space. [`IncColumn`] derives `inc` from raw trace values
//! for two rounds, then retains an explicit prefix with a zero suffix.

use jolt_claims::protocols::jolt::{JoltDerivedId, RamValCheckPublic};
use jolt_field::JoltField;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage4::ram_val_check::{RamValCheck, RamValCheckOutputClaims};
use jolt_witness::JoltWitnessPlane;
use std::sync::Arc;

use super::lazy_ra::{ChunkIndexSource, LazyBind, LazyFoldedRa, DEFAULT_MAX_WIDTH};
use super::ram_trace::{ram_access_columns_bytes, RamAccessColumns, RamAddresses, NO_ACCESS};
use super::support::{
    bind_prefix_pairs, bind_prefix_pairs_buffer_len, bound_pair, map_indices, pin_derived_term,
    triple_product_round_evals, RoundProgress, SplitLt,
};
use super::OptimizedBackend;
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct RamAddressIndices {
    addresses: Arc<RamAddresses>,
}

impl ChunkIndexSource for RamAddressIndices {
    fn constant_suffix_start(&self) -> usize {
        self.addresses.prefix_len()
    }

    fn num_columns(&self) -> usize {
        1
    }

    fn cycles(&self) -> usize {
        self.addresses.len()
    }

    #[inline]
    fn index(&self, _i: usize, j: usize) -> Option<usize> {
        let address = self.addresses[j];
        (address != NO_ACCESS).then_some(address as usize)
    }
}

impl<F: JoltField> PrepareKernel<F, RamValCheck<F>> for OptimizedBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamValCheck<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamValCheck<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let log_t = relation.trace_dimensions().log_t();
        let ram_log_k = relation.ram_log_k();
        let ram_val_point: &[F] = &inputs.points.ram_val;
        if ram_val_point.len() != ram_log_k + log_t {
            return Err(KernelError::InvariantViolation {
                reason: "RAM value-check input point has the wrong variable count",
            });
        }
        let (r_address, r_cycle) = ram_val_point.split_at(ram_log_k);

        // Reuse stage 2 addresses; collect only the short-lived values.
        let columns = RamAccessColumns::collect_full(session, witness, log_t)?;
        super::ram_trace::validate_addresses(&columns.addresses, 1usize << ram_log_k)?;
        let addresses = Arc::clone(&columns.addresses);

        Ok(Box::new(RamValCheckKernel {
            progress: RoundProgress::new(log_t),
            inc: IncColumn::Raw(columns),
            ra: LazyFoldedRa::new(vec![eq_table(r_address)], RamAddressIndices { addresses }),
            lt: SplitLt::new_plus_constant(r_cycle, inputs.challenges.gamma),
        }))
    }
}

/// Bytes the value check's scalar tables and the RAM access columns request
/// at their largest over `2^log_t` cycles, `ram_k` addresses and the
/// `prefix_cycles` cycles before the trace's constant suffix, or `None` past
/// `usize::MAX`. The bound replays the kernel's binds. The `ra` address table
/// doubles through the lazy branch schedule ([`LazyBind`]), each new table
/// beside the old one, until `ra` materializes at most
/// `⌈prefix_cycles / 2w⌉` values at the switch's width `w`; each later bind
/// may collect their bound half beside them, whose allocation is counted at
/// its materialized length. The second bind builds the
/// `⌈prefix_cycles / 4⌉`-entry increment table beside the check's own value
/// columns, then drops them, and later binds halve it in place or beside it
/// (`bind_prefix_pairs_buffer_len`). Every window counts the shared address
/// column and stage 2's value columns, which may still await their background
/// drop ([`ram_access_columns_bytes`]), the LT tables at their largest
/// (`SplitLt::peak_bytes`), and the branch tables queued for a background drop
/// at the switch, which no window assumes has run. Not counted: the access
/// columns' collection (the eager check reserves it separately), transients
/// inside `jolt-poly` constructors, the boxed kernel, `Vec` and `Arc` headers,
/// round messages, Rayon jobs and the macOS purge's cache-pump requests.
pub fn scalar_window_bytes<F: JoltField>(
    log_t: usize,
    ram_k: usize,
    prefix_cycles: usize,
) -> Option<usize> {
    /// `ra` between binds: the lazy branch table at `width` branches, or the
    /// queued branch tables beside the materialized values.
    enum Ra {
        Lazy {
            width: usize,
            table: usize,
        },
        Materialized {
            queued: usize,
            len: usize,
            capacity: usize,
        },
    }
    impl Ra {
        fn held(&self) -> Option<usize> {
            match *self {
                Self::Lazy { table, .. } => Some(table),
                Self::Materialized {
                    queued, capacity, ..
                } => queued.checked_add(capacity),
            }
        }
    }

    let cycles = 1usize << log_t;
    let mut peak = 0usize;
    // `generations` of access columns beside `entries` scalars.
    let mut window = |generations: usize, entries: usize| -> Option<()> {
        let bytes = ram_access_columns_bytes(prefix_cycles, generations)?
            .checked_add(entries.checked_mul(size_of::<F>())?)?;
        peak = peak.max(bytes);
        Some(())
    };
    // The address eq table and the one-entry tail beside both generations.
    window(2, ram_k.checked_add(1)?)?;
    let mut ra = Ra::Lazy {
        width: 1,
        table: ram_k,
    };
    // The increment table's `(len, capacity)` once the second bind builds it.
    let mut increments: Option<(usize, usize)> = None;
    for bind in 1..=log_t {
        let generations = if bind == 1 { 2 } else { 1 };
        let increments_held = match increments {
            None if bind == 2 => {
                let len = prefix_cycles.div_ceil(4);
                window(2, len.checked_add(ra.held()?)?)?;
                increments = Some((len, len));
                len
            }
            None => 0,
            Some((len, capacity)) => {
                let buffer = bind_prefix_pairs_buffer_len(len);
                window(1, capacity.checked_add(buffer)?.checked_add(ra.held()?)?)?;
                let capacity = if buffer > 0 {
                    len.div_ceil(2)
                } else {
                    capacity
                };
                increments = Some((len.div_ceil(2), capacity));
                capacity
            }
        };
        ra = match ra {
            Ra::Lazy { width, table } => {
                let doubled = table.checked_mul(2)?;
                let values = prefix_cycles.div_ceil(2 * width);
                match LazyBind::new(width, DEFAULT_MAX_WIDTH, 1, cycles, [table].into_iter()) {
                    LazyBind::Double => {
                        window(
                            generations,
                            increments_held.checked_add(table.checked_add(doubled)?)?,
                        )?;
                        Ra::Lazy {
                            width: 2 * width,
                            table: doubled,
                        }
                    }
                    LazyBind::DoubleThenMaterialize => {
                        let tables = table
                            .checked_add(doubled)?
                            .max(doubled.checked_add(values)?);
                        window(generations, increments_held.checked_add(tables)?)?;
                        Ra::Materialized {
                            queued: doubled,
                            len: values,
                            capacity: values,
                        }
                    }
                    LazyBind::FusedMaterialize => {
                        window(
                            generations,
                            increments_held.checked_add(table.checked_add(values)?)?,
                        )?;
                        Ra::Materialized {
                            queued: table,
                            len: values,
                            capacity: values,
                        }
                    }
                }
            }
            Ra::Materialized {
                queued,
                len,
                capacity,
            } => {
                let held = queued.checked_add(capacity)?.checked_add(len.div_ceil(2))?;
                window(generations, increments_held.checked_add(held)?)?;
                Ra::Materialized {
                    queued,
                    len: len.div_ceil(2),
                    capacity,
                }
            }
        };
    }
    peak.checked_add(SplitLt::<F>::peak_bytes(log_t)?)
}

/// Raw trace values for two rounds; an explicit prefix with a zero suffix afterward.
/// Reads have `post == pre`, and no-ops are zero, matching `RamInc`.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum IncColumn<F: JoltField> {
    Raw(RamAccessColumns),
    RawBound {
        columns: RamAccessColumns,
        #[cfg_attr(feature = "allocative", allocative(skip))]
        r1: F,
    },
    Bound(Vec<F>),
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct RamValCheckKernel<F: JoltField> {
    progress: RoundProgress,
    inc: IncColumn<F>,
    ra: LazyFoldedRa<F, RamAddressIndices>,
    lt: SplitLt<F>,
}

impl<F: JoltField> RamValCheckKernel<F> {
    fn bind(&mut self, challenge: F) {
        let mut freed_columns = false;
        if let IncColumn::Bound(inc) = &mut self.inc {
            bind_prefix_pairs(inc, challenge, F::zero());
        } else {
            let placeholder = IncColumn::Raw(RamAccessColumns {
                addresses: Arc::new(Vec::new().into()),
                pre_values: Vec::new(),
                post_values: Vec::new(),
            });
            self.inc = match std::mem::replace(&mut self.inc, placeholder) {
                IncColumn::Raw(columns) => IncColumn::RawBound {
                    columns,
                    r1: challenge,
                },
                IncColumn::RawBound { columns, r1 } => {
                    freed_columns = true;
                    IncColumn::Bound(map_indices(columns.value_prefix_len().div_ceil(4), |y| {
                        let first = |j| bound_pair(|k| columns.increment(k), r1, j);
                        bound_pair(first, challenge, y)
                    }))
                }
                IncColumn::Bound(_) => unreachable!("bound inc binds in place above"),
            };
        }
        self.ra.bind(challenge);
        self.lt.bind(challenge);
        self.progress.advance();
        // Return freed `T`-sized columns before the stage boundary.
        if freed_columns {
            crate::mem::purge_retained_memory(self.progress.total());
        }
    }
}

impl<F: JoltField> ProveRounds<F> for RamValCheckKernel<F> {
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

        // Each arm keeps representation branches outside the inner loop.
        let ra = |y: usize| self.ra.lo_hi(0, y);
        let lt = |y: usize| self.lt.pair(y);
        let evals = match &self.inc {
            IncColumn::Raw(columns) => triple_product_round_evals(
                columns.value_prefix_len().div_ceil(2),
                |y| (columns.increment(2 * y), columns.increment(2 * y + 1)),
                ra,
                lt,
            ),
            IncColumn::RawBound { columns, r1 } => triple_product_round_evals(
                columns.value_prefix_len().div_ceil(4),
                |y| {
                    let inc = |j| columns.increment(j);
                    (bound_pair(inc, *r1, 2 * y), bound_pair(inc, *r1, 2 * y + 1))
                },
                ra,
                lt,
            ),
            IncColumn::Bound(inc) => triple_product_round_evals(
                inc.len().div_ceil(2),
                |y| (inc[2 * y], inc.get(2 * y + 1).copied().unwrap_or(F::zero())),
                ra,
                lt,
            ),
        };

        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for RamValCheckKernel<F> {
    type Relation = RamValCheck<F>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RamValCheckOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        // Advice outputs echo their input claims; they are not bound here.
        Ok(RamValCheckOutputClaims {
            untrusted_advice: inputs.untrusted_advice,
            trusted_advice: inputs.trusted_advice,
            program_image: inputs.program_image,
            ram_ra: self.ra.final_values()[0],
            ram_inc: match &self.inc {
                IncColumn::Raw(columns) => columns.increment(0),
                IncColumn::RawBound { columns, r1 } => bound_pair(|j| columns.increment(j), *r1, 0),
                IncColumn::Bound(inc) => inc.first().copied().unwrap_or(F::zero()),
            },
        })
    }

    /// Check the bound split-LT value against the verifier's scalar path.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let id = JoltDerivedId::from(RamValCheckPublic::LtCyclePlusGamma);
        pin_derived_term(
            relation,
            id,
            input_points,
            output_points,
            challenges,
            self.lt.final_value(),
        )
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module: fail loudly")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::geometry::ram::{
        ram_inc_val_check, ram_ra_val_check, RamValCheckInit,
    };
    use jolt_claims::protocols::jolt::relations::ram::{
        RamValCheckChallenges, RamValCheckInputClaims,
    };
    use jolt_field::{Fr, Ring};
    use jolt_poly::LtPolynomial;

    use super::super::testing::{
        assert_parity, random_scalars, with_ram_fixture, FixtureShape, RamOp,
    };
    use super::*;
    use crate::reference::views::address_fold;
    use crate::ReferenceBackend;

    fn run_parity(shape: FixtureShape, ops: Vec<RamOp>, seed: u64) {
        with_ram_fixture(shape, ops, |witness| {
            let r_address = random_scalars(shape.log_k(), seed);
            let r_cycle = random_scalars(shape.log_t, seed ^ 0x11);
            let gamma = random_scalars(1, seed ^ 0x22)[0];
            let relation = RamValCheck::<Fr>::new(
                TraceDimensions::new(shape.log_t),
                shape.log_k(),
                RamValCheckInit::full(Fr::from_u64(0)),
            );
            let claims = RamValCheckInputClaims {
                ram_val: Fr::from_u64(0),
                ram_val_final: Fr::from_u64(0),
                untrusted_advice: None,
                trusted_advice: None,
                program_image: None,
            };
            let points = RamValCheckInputClaims::<Vec<Fr>> {
                ram_val: [r_address.clone(), r_cycle.clone()].concat(),
                ram_val_final: r_address.clone(),
                untrusted_advice: None,
                trusted_advice: None,
                program_image: None,
            };
            let challenges = RamValCheckChallenges { gamma };

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

            let ra_folded =
                address_fold::<Fr>(witness, ram_ra_val_check(), shape.log_t, &r_address).unwrap();
            let inc: Vec<Fr> = witness
                .oracle_table(ram_inc_val_check().polynomial_id())
                .unwrap();
            let lt = LtPolynomial::evaluations(&r_cycle);
            let input_claim = (0..1usize << shape.log_t)
                .map(|j| inc[j] * ra_folded[j] * (lt[j] + gamma))
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
                seed ^ 0x33,
            );
        });
    }

    fn mixed_ops() -> Vec<RamOp> {
        vec![
            RamOp::Write { word: 3, post: 5 },
            RamOp::Read { word: 3 },
            RamOp::None,
            RamOp::Write { word: 9, post: 4 },
            RamOp::Read { word: 9 },
            RamOp::Write { word: 3, post: 8 },
            RamOp::Read { word: 15 },
        ]
    }

    #[test]
    fn matches_reference_on_mixed_traffic() {
        run_parity(
            FixtureShape {
                log_t: 4,
                ram_k: 16,
            },
            mixed_ops(),
            31,
        );
    }

    #[test]
    fn matches_reference_on_odd_log_t() {
        // Five rounds: the lazy `ra` crosses its dense materialization and
        // the split LT collapses its lo tables mid-protocol.
        let mut ops = mixed_ops();
        ops.extend([
            RamOp::Write { word: 7, post: 2 },
            RamOp::Read { word: 7 },
            RamOp::None,
            RamOp::Read { word: 3 },
        ]);
        run_parity(
            FixtureShape {
                log_t: 5,
                ram_k: 16,
            },
            ops,
            37,
        );
    }

    #[test]
    fn zero_suffix_boundaries_match_reference() {
        let shape = FixtureShape {
            log_t: 5,
            ram_k: 16,
        };
        for prefix in [0, 1, 2, 3, 6, 7, 8, 14, 15, 16, 30, 31] {
            let ops = mixed_ops().into_iter().cycle().take(prefix).collect();
            run_parity(shape, ops, 401 + prefix as u64);
        }
        run_parity(FixtureShape { log_t: 5, ram_k: 1 }, vec![RamOp::None], 499);
    }

    #[test]
    fn matches_reference_on_single_round() {
        run_parity(
            FixtureShape { log_t: 1, ram_k: 8 },
            vec![RamOp::Write { word: 1, post: 4 }],
            41,
        );
    }

    /// Over 2^13 cycles, 64 addresses and 4096 proof rows, `ra` doubles at
    /// the first three binds and materializes 256 values at the fourth. The
    /// largest window opens the second bind: its 1024-entry increment table
    /// beside the 128-entry `ra` table, the `u32` address column and two
    /// generations of `u64` value columns, 4096 · 36 + 1152 · 32 = 184,320
    /// bytes. The windows before it hold at most 153,600 bytes (the first
    /// bind's 64- and 128-entry tables beside both generations), and those
    /// after it at most 143,360 (the fifth bind's). The LT tables add
    /// 64 + 3 · 128 entries, 14,336 bytes.
    #[test]
    fn scalar_windows_peak_at_the_second_bind() {
        assert_eq!(
            scalar_window_bytes::<Fr>(13, 64, 4096),
            Some(184_320 + 14_336)
        );
    }
}
