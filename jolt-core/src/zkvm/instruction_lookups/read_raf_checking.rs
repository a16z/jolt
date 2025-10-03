use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::constants::XLEN;
use rayon::prelude::*;
use std::{cell::RefCell, rc::Rc};
use strum::{EnumCount, IntoEnumIterator};
use tracer::instruction::Cycle;

use super::{LOG_K, LOG_M, M, PHASES};

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::{IdentityPolynomial, OperandPolynomial, OperandSide},
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        prefix_suffix::{Prefix, PrefixRegistry, PrefixSuffixDecomposition},
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    utils::{
        expanding_table::ExpandingTable, lookup_bits::LookupBits, math::Math,
        thread::unsafe_allocate_zero_vec,
    },
    zkvm::{
        dag::state_manager::StateManager,
        instruction::{InstructionFlags, InstructionLookup, InterleavedBitsMarker, LookupQuery},
        lookup_table::{
            prefixes::{PrefixCheckpoint, PrefixEval, Prefixes},
            LookupTables,
        },
        witness::VirtualPolynomial,
    },
};

use itertools::Itertools;
use rayon::iter::IndexedParallelIterator;

const DEGREE: usize = 3;

#[derive(Allocative)]
struct ReadRafProverState<F: JoltField> {
    ra_acc: Option<Vec<F>>,
    ra: Option<MultilinearPolynomial<F>>,
    r: Vec<F::Challenge>,

    lookup_indices: Vec<LookupBits>,
    lookup_indices_by_table: Vec<Vec<(usize, LookupBits)>>,
    lookup_indices_uninterleave: Vec<(usize, LookupBits)>,
    lookup_indices_identity: Vec<(usize, LookupBits)>,
    is_interleaved_operands: Vec<bool>,
    #[allocative(skip)]
    lookup_tables: Vec<Option<LookupTables<XLEN>>>,

    prefix_checkpoints: Vec<PrefixCheckpoint<F>>,
    suffix_polys: Vec<Vec<DensePolynomial<F>>>,
    v: ExpandingTable<F>,
    u_evals: Vec<F>,
    eq_r_cycle: MultilinearPolynomial<F>,

    prefix_registry: PrefixRegistry<F>,
    right_operand_ps: PrefixSuffixDecomposition<F, 2>,
    left_operand_ps: PrefixSuffixDecomposition<F, 2>,
    identity_ps: PrefixSuffixDecomposition<F, 2>,

    combined_val_polynomial: Option<MultilinearPolynomial<F>>,
}

#[derive(Allocative)]
pub struct ReadRafSumcheck<F: JoltField> {
    gamma: F,
    gamma_squared: F,
    prover_state: Option<ReadRafProverState<F>>,

    rv_claim: F,
    raf_claim: F,
    log_T: usize,
}

impl<'a, F: JoltField> ReadRafSumcheck<F> {
    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheck::new_prover")]
    pub fn new_prover(
        sm: &'a mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        eq_r_cycle: Vec<F>,
    ) -> Self {
        let trace = sm.get_prover_data().1;
        let log_T = trace.len().log_2();
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut ps = ReadRafProverState::new(trace, eq_r_cycle);
        ps.init_phase(0);
        let (_, rv_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );
        let (_, left_operand_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::SpartanOuter,
        );
        let (_, right_operand_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::SpartanOuter,
        );

        Self {
            gamma,
            gamma_squared: gamma.square(),
            prover_state: Some(ps),
            rv_claim,
            raf_claim: left_operand_claim + gamma * right_operand_claim,
            log_T,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let log_T = sm.get_verifier_data().2.log_2();
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let (_, rv_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );
        let (_, left_operand_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::SpartanOuter,
        );
        let (_, right_operand_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::SpartanOuter,
        );

        Self {
            gamma,
            gamma_squared: gamma.square(),
            prover_state: None,
            rv_claim,
            raf_claim: left_operand_claim + gamma * right_operand_claim,
            log_T,
        }
    }
}

impl<'a, F: JoltField> ReadRafProverState<F> {
    #[tracing::instrument(skip_all, name = "InstructionReadRafProverState::new")]
    fn new(trace: &'a [Cycle], eq_r_cycle: Vec<F>) -> Self {
        let log_T = trace.len().log_2();
        let right_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Right);
        let left_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Left);
        let identity_poly = IdentityPolynomial::new(LOG_K);
        let right_operand_ps =
            PrefixSuffixDecomposition::new(Box::new(right_operand_poly), LOG_M, LOG_K);
        let left_operand_ps =
            PrefixSuffixDecomposition::new(Box::new(left_operand_poly), LOG_M, LOG_K);
        let identity_ps = PrefixSuffixDecomposition::new(Box::new(identity_poly), LOG_M, LOG_K);

        // Heuristic: number of chunks = next_power_of_two(num_threads) * 4
        let threads = rayon::current_num_threads();
        let target_chunks = threads.next_power_of_two().saturating_mul(4);
        let chunk_size = std::cmp::max(1, trace.len().div_ceil(target_chunks));
        let num_tables = LookupTables::<XLEN>::COUNT;

        struct ChunkAgg<const XLEN: usize> {
            base: usize,
            lookup_indices: Vec<LookupBits>,
            uninterleave: Vec<(usize, LookupBits)>,
            identity: Vec<(usize, LookupBits)>,
            by_table: Vec<Vec<(usize, LookupBits)>>,
            flags: Vec<bool>,
            tables: Vec<Option<LookupTables<XLEN>>>,
        }

        let chunk_aggs: Vec<ChunkAgg<XLEN>> = trace
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let base = chunk_idx * chunk_size;
                let chunk_len = chunk.len();
                let mut lookup_indices = Vec::with_capacity(chunk_len);
                let mut flags = Vec::with_capacity(chunk_len);
                let mut tables = Vec::with_capacity(chunk_len);

                let mut uninterleave = Vec::with_capacity(chunk_len / 2 + 1);
                let mut identity = Vec::with_capacity(chunk_len / 2 + 1);
                let mut by_table = (0..num_tables)
                    .map(|_| Vec::with_capacity(chunk_len / num_tables + 1))
                    .collect::<Vec<_>>();

                for (off, cycle) in chunk.iter().enumerate() {
                    let idx = base + off;
                    let bits = LookupBits::new(LookupQuery::<XLEN>::to_lookup_index(cycle), LOG_K);
                    let is_interleaved = cycle
                        .instruction()
                        .circuit_flags()
                        .is_interleaved_operands();
                    let table = cycle.lookup_table();

                    if is_interleaved {
                        uninterleave.push((idx, bits));
                    } else {
                        identity.push((idx, bits));
                    }

                    if let Some(t) = table {
                        let t_idx = LookupTables::<XLEN>::enum_index(&t);
                        by_table[t_idx].push((idx, bits));
                    }

                    lookup_indices.push(bits);
                    flags.push(is_interleaved);
                    tables.push(table);
                }

                ChunkAgg {
                    base,
                    lookup_indices,
                    uninterleave,
                    identity,
                    by_table,
                    flags,
                    tables,
                }
            })
            .collect();

        let total_len = trace.len();
        let total_uninterleave: usize = chunk_aggs.iter().map(|a| a.uninterleave.len()).sum();
        let total_identity: usize = chunk_aggs.iter().map(|a| a.identity.len()).sum();
        let mut total_by_table = vec![0usize; num_tables];
        for agg in &chunk_aggs {
            for t in 0..num_tables {
                total_by_table[t] += agg.by_table[t].len();
            }
        }

        let mut lookup_indices = Vec::with_capacity(total_len);
        let mut is_interleaved_operands = Vec::with_capacity(total_len);
        let mut lookup_tables = Vec::with_capacity(total_len);
        let mut lookup_indices_uninterleave = Vec::with_capacity(total_uninterleave);
        let mut lookup_indices_identity = Vec::with_capacity(total_identity);
        let mut lookup_indices_by_table = (0..num_tables)
            .map(|t| Vec::with_capacity(total_by_table[t]))
            .collect::<Vec<_>>();

        for agg in chunk_aggs.into_iter().sorted_by_key(|a| a.base) {
            lookup_indices.extend(agg.lookup_indices);
            lookup_indices_uninterleave.extend(agg.uninterleave);
            lookup_indices_identity.extend(agg.identity);
            for t in 0..num_tables {
                lookup_indices_by_table[t].extend(agg.by_table[t].iter().copied());
            }
            is_interleaved_operands.extend(agg.flags);
            lookup_tables.extend(agg.tables);
        }

        let suffix_polys: Vec<Vec<DensePolynomial<F>>> = LookupTables::<XLEN>::iter()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|table| {
                table
                    .suffixes()
                    .par_iter()
                    .map(|_| DensePolynomial::default()) // Will be properly initialized in `init_phase`
                    .collect()
            })
            .collect();

        ReadRafProverState {
            r: Vec::with_capacity(log_T + LOG_K),
            ra_acc: None,
            ra: None,
            lookup_tables,
            lookup_indices,
            lookup_indices_by_table,
            lookup_indices_uninterleave,
            lookup_indices_identity,
            is_interleaved_operands,
            prefix_checkpoints: vec![None.into(); Prefixes::COUNT],
            suffix_polys,
            v: ExpandingTable::new(M),
            u_evals: eq_r_cycle.clone(),
            eq_r_cycle: MultilinearPolynomial::from(eq_r_cycle),
            prefix_registry: PrefixRegistry::new(),
            right_operand_ps,
            left_operand_ps,
            identity_ps,
            combined_val_polynomial: None,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for ReadRafSumcheck<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        LOG_K + self.log_T
    }

    fn input_claim(&self) -> F {
        self.rv_claim + self.gamma * self.raf_claim
    }

    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        let ps = self.prover_state.as_mut().unwrap();
        if round < LOG_K {
            // Phase 1: First log(K) rounds
            self.compute_prefix_suffix_prover_message(round).to_vec()
        } else {
            if ps.ra.is_none() {
                let ra_acc = ps.ra_acc.take().unwrap();
                ps.ra = Some(MultilinearPolynomial::from(ra_acc));
            }

            (0..ps.eq_r_cycle.len() / 2)
                .into_par_iter()
                .map(|i| {
                    let eq_evals = ps
                        .eq_r_cycle
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                    let ra_evals = ps
                        .ra
                        .as_ref()
                        .unwrap()
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                    let val_evals = ps
                        .combined_val_polynomial
                        .as_ref()
                        .unwrap()
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);

                    std::array::from_fn(|i| eq_evals[i] * ra_evals[i] * val_evals[i])
                })
                .reduce(
                    || [F::zero(); DEGREE],
                    |mut running, new| {
                        for j in 0..DEGREE {
                            running[j] += new[j];
                        }
                        running
                    },
                )
                .to_vec()
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        let ps = self.prover_state.as_mut().unwrap();
        ps.r.push(r_j);
        if round < LOG_K {
            rayon::scope(|s| {
                s.spawn(|_| {
                    ps.suffix_polys.par_iter_mut().for_each(|polys| {
                        polys
                            .par_iter_mut()
                            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow))
                    });
                });
                s.spawn(|_| ps.identity_ps.bind(r_j));
                s.spawn(|_| ps.right_operand_ps.bind(r_j));
                s.spawn(|_| ps.left_operand_ps.bind(r_j));
                s.spawn(|_| ps.v.update(r_j));
            });
            {
                if ps.r.len().is_multiple_of(2) {
                    Prefixes::update_checkpoints::<XLEN, F, F::Challenge>(
                        &mut ps.prefix_checkpoints,
                        ps.r[ps.r.len() - 2],
                        ps.r[ps.r.len() - 1],
                        round,
                    );
                }
            }

            // check if this is the last round in the phase
            if (round + 1).is_multiple_of(LOG_M) {
                let phase = round / LOG_M;
                ps.cache_phase(phase);
                // if not last phase, init next phase
                if phase != PHASES - 1 {
                    ps.init_phase(phase + 1);
                }
            }

            if (round + 1) == LOG_K {
                ps.init_log_t_rounds(self.gamma, self.gamma_squared);
            }
        } else {
            // log(T) rounds

            [
                ps.ra.as_mut().unwrap(),
                &mut ps.eq_r_cycle,
                ps.combined_val_polynomial.as_mut().unwrap(),
            ]
            .par_iter_mut()
            .for_each(|poly| {
                poly.bind_parallel(r_j, BindingOrder::HighToLow);
            });
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let (r_address_prime, r_cycle_prime) = r.split_at(LOG_K);
        let left_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Left).evaluate(r_address_prime);
        let right_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Right).evaluate(r_address_prime);
        let identity_poly_eval = IdentityPolynomial::<F>::new(LOG_K).evaluate(r_address_prime);
        let val_evals: Vec<_> = LookupTables::<XLEN>::iter()
            .map(|table| table.evaluate_mle::<F, F::Challenge>(r_address_prime))
            .collect();

        let accumulator = accumulator.as_ref().unwrap();

        let r_cycle = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r;
        let eq_eval_cycle = EqPolynomial::<F>::mle(&r_cycle, r_cycle_prime);

        let ra_claim = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRa,
                SumcheckId::InstructionReadRaf,
            )
            .1;

        let table_flag_claims: Vec<F> = (0..LookupTables::<XLEN>::COUNT)
            .map(|i| {
                let accumulator = accumulator.borrow();
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::LookupTableFlag(i),
                        SumcheckId::InstructionReadRaf,
                    )
                    .1
            })
            .collect();

        let accumulator = accumulator.borrow();
        let raf_flag_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRafFlag,
                SumcheckId::InstructionReadRaf,
            )
            .1;

        let rv_val_claim = val_evals
            .into_iter()
            .zip(table_flag_claims)
            .map(|(claim, val)| claim * val)
            .sum::<F>();

        let val_eval = rv_val_claim
            + (F::one() - raf_flag_claim)
                * (self.gamma * left_operand_eval + self.gamma_squared * right_operand_eval)
            + raf_flag_claim * self.gamma_squared * identity_poly_eval;
        eq_eval_cycle * ra_claim * val_eval
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        r_sumcheck: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self.prover_state.as_ref().unwrap();
        let (_r_address, r_cycle) = r_sumcheck.clone().split_at(LOG_K);
        let eq_r_cycle_prime = EqPolynomial::<F>::evals(&r_cycle.r);

        let flag_claims = ps
            .lookup_indices_by_table
            .par_iter()
            .map(|table_lookups| {
                table_lookups
                    .iter()
                    .map(|(j, _)| eq_r_cycle_prime[*j])
                    .sum::<F>()
            })
            .collect::<Vec<F>>();
        flag_claims.into_iter().enumerate().for_each(|(i, claim)| {
            accumulator.borrow_mut().append_virtual(
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
                r_cycle.clone(),
                claim,
            );
        });

        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
            r_sumcheck,
            ps.ra.as_ref().unwrap().final_sumcheck_claim(),
        );
        let raf_flag_claim = ps
            .lookup_indices_identity
            .par_iter()
            .map(|(j, _)| eq_r_cycle_prime[*j])
            .sum::<F>();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
            r_cycle.clone(),
            raf_flag_claim,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        r_sumcheck: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let (_r_address, r_cycle) = r_sumcheck.split_at(LOG_K);

        (0..LookupTables::<XLEN>::COUNT).for_each(|i| {
            accumulator.borrow_mut().append_virtual(
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
                r_cycle.clone(),
            );
        });

        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
            r_cycle.clone(),
        );

        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
            r_sumcheck,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

impl<F: JoltField> ReadRafProverState<F> {
    /// To be called in the beginning of each phase, before any binding
    #[tracing::instrument(skip_all, name = "InstructionReadRafProverState::init_phase")]
    fn init_phase(&mut self, phase: usize) {
        // Condensation
        if phase != 0 {
            let span = tracing::span!(tracing::Level::INFO, "Update u_evals");
            let _guard = span.enter();
            self.lookup_indices
                .par_iter()
                .zip(self.u_evals.par_iter_mut())
                .for_each(|(k, u)| {
                    let (prefix, _) = k.split((PHASES - phase) * LOG_M);
                    let k_bound: usize = prefix % M;
                    *u *= self.v[k_bound];
                });
        }

        rayon::scope(|s| {
            // Single pass over lookup_indices_uninterleave for both operands
            s.spawn(|_| {
                PrefixSuffixDecomposition::init_Q_dual(
                    &mut self.left_operand_ps,
                    &mut self.right_operand_ps,
                    &self.u_evals,
                    &self.lookup_indices_uninterleave,
                )
            });
            s.spawn(|_| {
                self.identity_ps
                    .init_Q(&self.u_evals, &self.lookup_indices_identity)
            });
        });

        self.init_suffix_polys(phase);

        self.identity_ps.init_P(&mut self.prefix_registry);
        self.right_operand_ps.init_P(&mut self.prefix_registry);
        self.left_operand_ps.init_P(&mut self.prefix_registry);

        self.v.reset(F::one());
    }

    #[tracing::instrument(skip_all, name = "InstructionReadRafProverState::init_suffix_polys")]
    fn init_suffix_polys(&mut self, phase: usize) {
        let num_chunks = rayon::current_num_threads().next_power_of_two();
        let chunk_size = (self.lookup_indices.len() / num_chunks).max(1);

        let new_suffix_polys: Vec<_> = LookupTables::<XLEN>::iter()
            .collect::<Vec<_>>()
            .par_iter()
            .zip(self.lookup_indices_by_table.par_iter())
            .map(|(table, lookup_indices)| {
                let suffixes = table.suffixes();
                lookup_indices
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let mut chunk_result: Vec<Vec<F>> =
                            vec![unsafe_allocate_zero_vec(M); suffixes.len()];

                        for (j, k) in chunk {
                            let (prefix_bits, suffix_bits) = k.split((PHASES - 1 - phase) * LOG_M);
                            for (suffix, result) in suffixes.iter().zip(chunk_result.iter_mut()) {
                                let t = suffix.suffix_mle::<XLEN>(suffix_bits);
                                if t != 0 {
                                    let u = self.u_evals[*j];
                                    result[prefix_bits % M] += u.mul_u64(t);
                                }
                            }
                        }

                        chunk_result
                    })
                    .reduce(
                        || vec![unsafe_allocate_zero_vec(M); suffixes.len()],
                        |mut acc, new| {
                            for (acc_i, new_i) in acc.iter_mut().zip(new.iter()) {
                                for (acc_coeff, new_coeff) in acc_i.iter_mut().zip(new_i.iter()) {
                                    *acc_coeff += *new_coeff;
                                }
                            }
                            acc
                        },
                    )
            })
            .collect();

        // Replace existing suffix polynomials
        self.suffix_polys
            .iter_mut()
            .zip(new_suffix_polys.into_iter())
            .for_each(|(old, new)| {
                old.iter_mut()
                    .zip(new.into_iter())
                    .for_each(|(poly, mut coeffs)| {
                        *poly = DensePolynomial::new(std::mem::take(&mut coeffs));
                    });
            });
    }

    /// To be called at the end of each phase, after binding is done
    #[tracing::instrument(skip_all, name = "InstructionReadRafProverState::cache_phase")]
    fn cache_phase(&mut self, phase: usize) {
        if let Some(ra_acc) = self.ra_acc.as_mut() {
            ra_acc
                .par_iter_mut()
                .zip(self.lookup_indices.par_iter())
                .for_each(|(ra, k)| {
                    let (prefix, _) = k.split((PHASES - 1 - phase) * LOG_M);
                    let k_bound: usize = prefix % M;
                    *ra *= self.v[k_bound]
                });
        } else {
            let ra = self
                .lookup_indices
                .par_iter()
                .map(|k| {
                    let (prefix, _) = k.split((PHASES - 1 - phase) * LOG_M);
                    let k_bound: usize = prefix % M;
                    self.v[k_bound]
                })
                .collect::<Vec<F>>();
            self.ra_acc = Some(ra);
        }

        self.prefix_registry.update_checkpoints();
    }

    /// To be called before the last log(T) rounds
    #[tracing::instrument(skip_all, name = "InstructionReadRafProverState::init_log_t_rounds")]
    fn init_log_t_rounds(&mut self, gamma: F, gamma_squared: F) {
        let prefixes: Vec<PrefixEval<F>> = std::mem::take(&mut self.prefix_checkpoints)
            .into_iter()
            .map(|checkpoint| checkpoint.unwrap())
            .collect();
        let mut combined_val_poly: Vec<F> = unsafe_allocate_zero_vec(self.lookup_indices.len());
        combined_val_poly
            .par_iter_mut()
            .zip(std::mem::take(&mut self.lookup_tables))
            .zip(std::mem::take(&mut self.is_interleaved_operands))
            .for_each(|((val, table), is_interleaved_operands)| {
                if let Some(table) = table {
                    let suffixes: Vec<_> = table
                        .suffixes()
                        .iter()
                        .map(|suffix| F::from_u64(suffix.suffix_mle::<XLEN>(LookupBits::new(0, 0))))
                        .collect();
                    *val += table.combine(&prefixes, &suffixes);
                }

                if is_interleaved_operands {
                    *val += gamma * self.prefix_registry.checkpoints[Prefix::LeftOperand].unwrap()
                        + gamma_squared
                            * self.prefix_registry.checkpoints[Prefix::RightOperand].unwrap();
                } else {
                    *val +=
                        gamma_squared * self.prefix_registry.checkpoints[Prefix::Identity].unwrap();
                }
            });
        self.combined_val_polynomial = Some(MultilinearPolynomial::from(combined_val_poly));
    }
}

impl<F: JoltField> ReadRafSumcheck<F> {
    fn compute_prefix_suffix_prover_message(&self, round: usize) -> [F; 2] {
        let mut read_checking = [F::zero(), F::zero()];
        let mut raf = [F::zero(), F::zero()];

        rayon::join(
            || {
                read_checking = self.prover_msg_read_checking(round);
            },
            || {
                raf = self.prover_msg_raf();
            },
        );

        [read_checking[0] + raf[0], read_checking[1] + raf[1]]
    }

    fn prover_msg_raf(&self) -> [F; 2] {
        let ps = self.prover_state.as_ref().unwrap();
        let len = ps.identity_ps.Q_len();
        let (left_0, left_2, right_0, right_2) = (0..len / 2)
            .into_par_iter()
            .map(|b| {
                let (i0, i2) = ps.identity_ps.sumcheck_evals(b);
                let (r0, r2) = ps.right_operand_ps.sumcheck_evals(b);
                let (l0, l2) = ps.left_operand_ps.sumcheck_evals(b);
                (l0, l2, i0 + r0, i2 + r2)
            })
            .reduce(
                || (F::zero(), F::zero(), F::zero(), F::zero()),
                |running, new| {
                    (
                        running.0 + new.0,
                        running.1 + new.1,
                        running.2 + new.2,
                        running.3 + new.3,
                    )
                },
            );
        [
            self.gamma * left_0 + self.gamma_squared * right_0,
            self.gamma * left_2 + self.gamma_squared * right_2,
        ]
    }

    fn prover_msg_read_checking(&self, j: usize) -> [F; 2] {
        let ps = self.prover_state.as_ref().unwrap();
        let lookup_tables: Vec<_> = LookupTables::<XLEN>::iter().collect();

        let len = ps.suffix_polys[0][0].len();
        let log_len = len.log_2();

        let r_x = if j % 2 == 1 {
            ps.r.last().copied()
        } else {
            None
        };

        let (eval_0, eval_2_left, eval_2_right) = (0..len / 2)
            .into_par_iter()
            .flat_map_iter(|b| {
                let b = LookupBits::new(b as u128, log_len - 1);
                let prefixes_c0: Vec<_> = Prefixes::iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<XLEN, F, F::Challenge>(
                            &ps.prefix_checkpoints,
                            r_x,
                            0,
                            b,
                            j,
                        )
                    })
                    .collect();
                let prefixes_c2: Vec<_> = Prefixes::iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<XLEN, F, F::Challenge>(
                            &ps.prefix_checkpoints,
                            r_x,
                            2,
                            b,
                            j,
                        )
                    })
                    .collect();
                lookup_tables
                    .iter()
                    .zip(ps.suffix_polys.iter())
                    .map(move |(table, suffixes)| {
                        let suffixes_left: Vec<_> =
                            suffixes.iter().map(|suffix| suffix[b.into()]).collect();
                        let suffixes_right: Vec<_> = suffixes
                            .iter()
                            .map(|suffix| suffix[usize::from(b) + len / 2])
                            .collect();
                        (
                            table.combine(&prefixes_c0, &suffixes_left),
                            table.combine(&prefixes_c2, &suffixes_left),
                            table.combine(&prefixes_c2, &suffixes_right),
                        )
                    })
            })
            .reduce(
                || (F::zero(), F::zero(), F::zero()),
                |running, new| (running.0 + new.0, running.1 + new.1, running.2 + new.2),
            );
        [eval_0, eval_2_right + eval_2_right - eval_2_left]
    }
}

/// Computes the bit-length of the suffix, for the current (`j`th) round
/// of sumcheck.
pub fn current_suffix_len(j: usize) -> usize {
    LOG_K - (j / LOG_M + 1) * LOG_M
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subprotocols::sumcheck::BatchedSumcheck;
    use crate::transcripts::Blake2bTranscript;
    use crate::{
        poly::commitment::mock::MockCommitScheme,
        zkvm::{
            bytecode::BytecodePreprocessing, ram::RAMPreprocessing, JoltProverPreprocessing,
            JoltSharedPreprocessing, JoltVerifierPreprocessing,
        },
    };
    use ark_bn254::Fr;
    use ark_std::Zero;
    use common::jolt_device::MemoryLayout;
    use rand::{rngs::StdRng, RngCore, SeedableRng};
    use strum::IntoEnumIterator;
    use tracer::emulator::memory::Memory;
    use tracer::instruction::Cycle;
    use tracer::JoltDevice;

    const LOG_T: usize = 8;
    const T: usize = 1 << LOG_T;

    fn random_instruction(rng: &mut StdRng, instruction: &Option<Cycle>) -> Cycle {
        let instruction = instruction.unwrap_or_else(|| {
            let index = rng.next_u64() as usize % Cycle::COUNT;
            Cycle::iter()
                .enumerate()
                .filter(|(i, _)| *i == index)
                .map(|(_, x)| x)
                .next()
                .unwrap()
        });

        match instruction {
            Cycle::ADD(cycle) => cycle.random(rng).into(),
            Cycle::ADDI(cycle) => cycle.random(rng).into(),
            Cycle::AND(cycle) => cycle.random(rng).into(),
            Cycle::ANDN(cycle) => cycle.random(rng).into(),
            Cycle::ANDI(cycle) => cycle.random(rng).into(),
            Cycle::AUIPC(cycle) => cycle.random(rng).into(),
            Cycle::BEQ(cycle) => cycle.random(rng).into(),
            Cycle::BGE(cycle) => cycle.random(rng).into(),
            Cycle::BGEU(cycle) => cycle.random(rng).into(),
            Cycle::BLT(cycle) => cycle.random(rng).into(),
            Cycle::BLTU(cycle) => cycle.random(rng).into(),
            Cycle::BNE(cycle) => cycle.random(rng).into(),
            Cycle::FENCE(cycle) => cycle.random(rng).into(),
            Cycle::JAL(cycle) => cycle.random(rng).into(),
            Cycle::JALR(cycle) => cycle.random(rng).into(),
            Cycle::LUI(cycle) => cycle.random(rng).into(),
            Cycle::LD(cycle) => cycle.random(rng).into(),
            Cycle::MUL(cycle) => cycle.random(rng).into(),
            Cycle::MULHU(cycle) => cycle.random(rng).into(),
            Cycle::OR(cycle) => cycle.random(rng).into(),
            Cycle::ORI(cycle) => cycle.random(rng).into(),
            Cycle::SLT(cycle) => cycle.random(rng).into(),
            Cycle::SLTI(cycle) => cycle.random(rng).into(),
            Cycle::SLTIU(cycle) => cycle.random(rng).into(),
            Cycle::SLTU(cycle) => cycle.random(rng).into(),
            Cycle::SUB(cycle) => cycle.random(rng).into(),
            Cycle::SD(cycle) => cycle.random(rng).into(),
            Cycle::XOR(cycle) => cycle.random(rng).into(),
            Cycle::XORI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAdvice(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertEQ(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertHalfwordAlignment(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertWordAlignment(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertLTE(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertValidDiv0(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertValidUnsignedRemainder(cycle) => cycle.random(rng).into(),
            Cycle::VirtualMove(cycle) => cycle.random(rng).into(),
            Cycle::VirtualMovsign(cycle) => cycle.random(rng).into(),
            Cycle::VirtualMULI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualPow2(cycle) => cycle.random(rng).into(),
            Cycle::VirtualPow2I(cycle) => cycle.random(rng).into(),
            Cycle::VirtualPow2W(cycle) => cycle.random(rng).into(),
            Cycle::VirtualPow2IW(cycle) => cycle.random(rng).into(),
            Cycle::VirtualShiftRightBitmask(cycle) => cycle.random(rng).into(),
            Cycle::VirtualShiftRightBitmaskI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSRA(cycle) => cycle.random(rng).into(),
            Cycle::VirtualRev8W(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSRAI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSRL(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSRLI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualZeroExtendWord(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSignExtendWord(cycle) => cycle.random(rng).into(),
            Cycle::VirtualROTRI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualROTRIW(cycle) => cycle.random(rng).into(),
            Cycle::VirtualChangeDivisor(cycle) => cycle.random(rng).into(),
            Cycle::VirtualChangeDivisorW(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertMulUNoOverflow(cycle) => cycle.random(rng).into(),
            _ => Cycle::NoOp,
        }
    }

    fn test_read_raf_sumcheck(instruction: Option<Cycle>) {
        let mut rng = StdRng::seed_from_u64(12345);

        let trace: Vec<_> = (0..T)
            .map(|_| random_instruction(&mut rng, &instruction))
            .collect();
        let bytecode = vec![];
        let bytecode_preprocessing = BytecodePreprocessing::preprocess(bytecode);
        let memory_layout = MemoryLayout::default();
        let shared_preprocessing = JoltSharedPreprocessing {
            bytecode: bytecode_preprocessing,
            ram: RAMPreprocessing::preprocess(vec![]),
            memory_layout: memory_layout.clone(),
        };
        let prover_preprocessing: JoltProverPreprocessing<Fr, MockCommitScheme<Fr>> =
            JoltProverPreprocessing {
                generators: (),
                shared: shared_preprocessing.clone(),
            };

        let verifier_preprocessing: JoltVerifierPreprocessing<Fr, MockCommitScheme<Fr>> =
            JoltVerifierPreprocessing {
                generators: (),
                shared: shared_preprocessing,
            };
        let program_io = JoltDevice {
            memory_layout,
            inputs: vec![],
            outputs: vec![],
            panic: false,
        };
        let final_memory_state = Memory::default();

        let mut prover_sm = StateManager::<'_, Fr, Blake2bTranscript, _>::new_prover(
            &prover_preprocessing,
            trace.clone(),
            program_io.clone(),
            final_memory_state,
        );
        let mut verifier_sm = StateManager::<'_, Fr, Blake2bTranscript, _>::new_verifier(
            &verifier_preprocessing,
            program_io,
            trace.len(),
            1 << 8,
            prover_sm.twist_sumcheck_switch_index,
        );

        let r_cycle: Vec<<Fr as JoltField>::Challenge> = prover_sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<Fr>(LOG_T);
        let _r_cycle: Vec<<Fr as JoltField>::Challenge> = verifier_sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<Fr>(LOG_T);
        let eq_r_cycle = EqPolynomial::evals(&r_cycle);

        let mut rv_claim = Fr::zero();
        let mut left_operand_claim = Fr::zero();
        let mut right_operand_claim = Fr::zero();

        for (i, cycle) in trace.iter().enumerate() {
            let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
            let table: Option<LookupTables<XLEN>> = cycle.lookup_table();
            if let Some(table) = table {
                rv_claim +=
                    JoltField::mul_u64(&eq_r_cycle[i], table.materialize_entry(lookup_index));
            }
            let (lo, ro) = LookupQuery::<XLEN>::to_lookup_operands(cycle);
            left_operand_claim += JoltField::mul_u64(&eq_r_cycle[i], lo);
            right_operand_claim += JoltField::mul_u128(&eq_r_cycle[i], ro);
        }

        let prover_accumulator = prover_sm.get_prover_accumulator();
        prover_accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(r_cycle.clone()),
            rv_claim,
        );
        prover_accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(r_cycle.clone()),
            left_operand_claim,
        );
        prover_accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(r_cycle.clone()),
            right_operand_claim,
        );

        let mut prover_sumcheck = ReadRafSumcheck::new_prover(&mut prover_sm, eq_r_cycle);

        let mut prover_transcript_ref = prover_sm.transcript.borrow_mut();

        let (proof, r_sumcheck) = BatchedSumcheck::prove(
            vec![&mut prover_sumcheck],
            Some(prover_accumulator.clone()),
            &mut *prover_transcript_ref,
        );
        drop(prover_transcript_ref);

        // Take claims
        let prover_acc_borrow = prover_accumulator.borrow();
        let verifier_accumulator = verifier_sm.get_verifier_accumulator();
        let mut verifier_acc_borrow = verifier_accumulator.borrow_mut();

        for (key, (_, value)) in prover_acc_borrow.evaluation_openings().iter() {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier_acc_borrow
                .openings_mut()
                .insert(*key, (empty_point, *value));
        }
        drop(prover_acc_borrow);
        drop(verifier_acc_borrow);

        verifier_accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(r_cycle.clone()),
        );
        verifier_accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(r_cycle.clone()),
        );
        verifier_accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(r_cycle.clone()),
        );

        let mut verifier_sumcheck = ReadRafSumcheck::new_verifier(&mut verifier_sm);

        let r_sumcheck_verif = BatchedSumcheck::verify(
            &proof,
            vec![&mut verifier_sumcheck],
            Some(verifier_accumulator.clone()),
            &mut *verifier_sm.transcript.borrow_mut(),
        )
        .unwrap();

        assert_eq!(r_sumcheck, r_sumcheck_verif);
    }

    #[test]
    fn test_random_instructions() {
        test_read_raf_sumcheck(None);
    }

    #[test]
    fn test_add() {
        test_read_raf_sumcheck(Some(Cycle::ADD(Default::default())));
    }

    #[test]
    fn test_addi() {
        test_read_raf_sumcheck(Some(Cycle::ADDI(Default::default())));
    }

    #[test]
    fn test_and() {
        test_read_raf_sumcheck(Some(Cycle::AND(Default::default())));
    }

    #[test]
    fn test_andn() {
        test_read_raf_sumcheck(Some(Cycle::ANDN(Default::default())));
    }

    #[test]
    fn test_andi() {
        test_read_raf_sumcheck(Some(Cycle::ANDI(Default::default())));
    }

    #[test]
    fn test_auipc() {
        test_read_raf_sumcheck(Some(Cycle::AUIPC(Default::default())));
    }

    #[test]
    fn test_beq() {
        test_read_raf_sumcheck(Some(Cycle::BEQ(Default::default())));
    }

    #[test]
    fn test_bge() {
        test_read_raf_sumcheck(Some(Cycle::BGE(Default::default())));
    }

    #[test]
    fn test_bgeu() {
        test_read_raf_sumcheck(Some(Cycle::BGEU(Default::default())));
    }

    #[test]
    fn test_blt() {
        test_read_raf_sumcheck(Some(Cycle::BLT(Default::default())));
    }

    #[test]
    fn test_bltu() {
        test_read_raf_sumcheck(Some(Cycle::BLTU(Default::default())));
    }

    #[test]
    fn test_bne() {
        test_read_raf_sumcheck(Some(Cycle::BNE(Default::default())));
    }

    #[test]
    fn test_fence() {
        test_read_raf_sumcheck(Some(Cycle::FENCE(Default::default())));
    }

    #[test]
    fn test_jal() {
        test_read_raf_sumcheck(Some(Cycle::JAL(Default::default())));
    }

    #[test]
    fn test_jalr() {
        test_read_raf_sumcheck(Some(Cycle::JALR(Default::default())));
    }

    #[test]
    fn test_lui() {
        test_read_raf_sumcheck(Some(Cycle::LUI(Default::default())));
    }

    #[test]
    fn test_ld() {
        test_read_raf_sumcheck(Some(Cycle::LD(Default::default())));
    }

    #[test]
    fn test_mul() {
        test_read_raf_sumcheck(Some(Cycle::MUL(Default::default())));
    }

    #[test]
    fn test_mulhu() {
        test_read_raf_sumcheck(Some(Cycle::MULHU(Default::default())));
    }

    #[test]
    fn test_or() {
        test_read_raf_sumcheck(Some(Cycle::OR(Default::default())));
    }

    #[test]
    fn test_ori() {
        test_read_raf_sumcheck(Some(Cycle::ORI(Default::default())));
    }

    #[test]
    fn test_slt() {
        test_read_raf_sumcheck(Some(Cycle::SLT(Default::default())));
    }

    #[test]
    fn test_slti() {
        test_read_raf_sumcheck(Some(Cycle::SLTI(Default::default())));
    }

    #[test]
    fn test_sltiu() {
        test_read_raf_sumcheck(Some(Cycle::SLTIU(Default::default())));
    }

    #[test]
    fn test_sltu() {
        test_read_raf_sumcheck(Some(Cycle::SLTU(Default::default())));
    }

    #[test]
    fn test_sub() {
        test_read_raf_sumcheck(Some(Cycle::SUB(Default::default())));
    }

    #[test]
    fn test_sd() {
        test_read_raf_sumcheck(Some(Cycle::SD(Default::default())));
    }

    #[test]
    fn test_xor() {
        test_read_raf_sumcheck(Some(Cycle::XOR(Default::default())));
    }

    #[test]
    fn test_xori() {
        test_read_raf_sumcheck(Some(Cycle::XORI(Default::default())));
    }

    #[test]
    fn test_advice() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAdvice(Default::default())));
    }

    #[test]
    fn test_asserteq() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertEQ(Default::default())));
    }

    #[test]
    fn test_asserthalfwordalignment() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertHalfwordAlignment(
            Default::default(),
        )));
    }

    #[test]
    fn test_assertwordalignment() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertWordAlignment(Default::default())));
    }

    #[test]
    fn test_assertlte() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertLTE(Default::default())));
    }

    #[test]
    fn test_assertvaliddiv0() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertValidDiv0(Default::default())));
    }

    #[test]
    fn test_assertvalidunsignedremainder() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertValidUnsignedRemainder(
            Default::default(),
        )));
    }

    #[test]
    fn test_move() {
        test_read_raf_sumcheck(Some(Cycle::VirtualMove(Default::default())));
    }

    #[test]
    fn test_movsign() {
        test_read_raf_sumcheck(Some(Cycle::VirtualMovsign(Default::default())));
    }

    #[test]
    fn test_muli() {
        test_read_raf_sumcheck(Some(Cycle::VirtualMULI(Default::default())));
    }

    #[test]
    fn test_pow2() {
        test_read_raf_sumcheck(Some(Cycle::VirtualPow2(Default::default())));
    }

    #[test]
    fn test_pow2i() {
        test_read_raf_sumcheck(Some(Cycle::VirtualPow2I(Default::default())));
    }

    #[test]
    fn test_pow2w() {
        test_read_raf_sumcheck(Some(Cycle::VirtualPow2W(Default::default())));
    }

    #[test]
    fn test_pow2iw() {
        test_read_raf_sumcheck(Some(Cycle::VirtualPow2IW(Default::default())));
    }

    #[test]
    fn test_shiftrightbitmask() {
        test_read_raf_sumcheck(Some(Cycle::VirtualShiftRightBitmask(Default::default())));
    }

    #[test]
    fn test_shiftrightbitmaski() {
        test_read_raf_sumcheck(Some(Cycle::VirtualShiftRightBitmaskI(Default::default())));
    }

    #[test]
    fn test_virtualrotri() {
        test_read_raf_sumcheck(Some(Cycle::VirtualROTRI(Default::default())));
    }

    #[test]
    fn test_virtualrotriw() {
        test_read_raf_sumcheck(Some(Cycle::VirtualROTRIW(Default::default())));
    }

    #[test]
    fn test_virtualsra() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSRA(Default::default())));
    }

    #[test]
    fn test_virtualsrai() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSRAI(Default::default())));
    }

    #[test]
    fn test_virtualrev8w() {
        test_read_raf_sumcheck(Some(Cycle::VirtualRev8W(Default::default())));
    }

    #[test]
    fn test_virtualsrl() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSRL(Default::default())));
    }

    #[test]
    fn test_virtualsrli() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSRLI(Default::default())));
    }

    #[test]
    fn test_virtualextend() {
        test_read_raf_sumcheck(Some(Cycle::VirtualZeroExtendWord(Default::default())));
    }

    #[test]
    fn test_virtualsignextend() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSignExtendWord(Default::default())));
    }

    #[test]
    fn test_virtualchangedivisor() {
        test_read_raf_sumcheck(Some(Cycle::VirtualChangeDivisor(Default::default())));
    }

    #[test]
    fn test_virtualchangedivisorw() {
        test_read_raf_sumcheck(Some(Cycle::VirtualChangeDivisorW(Default::default())));
    }

    #[test]
    fn test_virtualassertmulnooverflow() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertMulUNoOverflow(Default::default())));
    }
}
