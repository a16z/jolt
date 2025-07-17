use rayon::prelude::*;
use std::{cell::RefCell, rc::Rc};
use strum::{EnumCount, IntoEnumIterator};
use tracer::instruction::RV32IMCycle;

use super::{D, K_CHUNK, LOG_K, LOG_K_CHUNK, LOG_M, M, PHASES, RA_PER_LOG_M, WORD_SIZE};

use crate::{
    dag::{stage::StagedSumcheck, state_manager::StateManager},
    field::JoltField,
    jolt::{
        instruction::{InstructionFlags, InstructionLookup, InterleavedBitsMarker, LookupQuery},
        lookup_table::{
            prefixes::{PrefixCheckpoint, PrefixEval, Prefixes},
            LookupTables,
        },
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::{IdentityPolynomial, OperandPolynomial, OperandSide},
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningPoint, OpeningsKeys, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        prefix_suffix::{Prefix, PrefixRegistry, PrefixSuffixDecomposition},
    },
    r1cs::inputs::JoltR1CSInputs,
    subprotocols::{
        sparse_dense_shout::{compute_prefix_suffix_prover_message, ExpandingTable, LookupBits},
        sumcheck::{BatchableSumcheckInstance, CacheSumcheckOpenings},
    },
    utils::{
        math::Math,
        thread::{unsafe_allocate_zero_vec, unsafe_zero_slice},
        transcript::Transcript,
    },
};

const DEGREE: usize = D + 2;

struct ReadRafProverState<F: JoltField> {
    ra: Vec<MultilinearPolynomial<F>>,
    r: Vec<F>,

    lookup_indices: Vec<LookupBits>,
    lookup_indices_by_table: Vec<Vec<(usize, LookupBits)>>,
    lookup_indices_uninterleave: Vec<(usize, LookupBits)>,
    lookup_indices_identity: Vec<(usize, LookupBits)>,
    is_interleaved_operands: Vec<bool>,
    lookup_tables: Vec<Option<LookupTables<WORD_SIZE>>>,

    prefix_checkpoints: Vec<PrefixCheckpoint<F>>,
    suffix_polys: Vec<Vec<DensePolynomial<F>>>,
    v: [ExpandingTable<F>; RA_PER_LOG_M],
    u_evals: Vec<F>,
    eq_r_cycle: MultilinearPolynomial<F>,

    prefix_registry: PrefixRegistry<F>,
    right_operand_ps: PrefixSuffixDecomposition<F, 2>,
    left_operand_ps: PrefixSuffixDecomposition<F, 2>,
    identity_ps: PrefixSuffixDecomposition<F, 2>,

    combined_val_polynomial: Option<MultilinearPolynomial<F>>,

    /// For the opening proofs
    unbound_ra_polys: Vec<MultilinearPolynomial<F>>,
}

pub struct ReadRafSumcheck<F: JoltField> {
    gamma: F,
    gamma_squared: F,
    prover_state: Option<ReadRafProverState<F>>,
    prod_ra_claims: Option<F>,
    table_flag_claims: Option<Vec<F>>,
    raf_flag_claim: Option<F>,

    r_cycle: Vec<F>,
    rv_claim: F,
    raf_claim: F,
    log_T: usize,
}

impl<'a, F: JoltField> ReadRafSumcheck<F> {
    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheck::new_prover")]
    pub fn new_prover(
        sm: &'a mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        eq_r_cycle: Vec<F>,
        unbound_ra_polys: Vec<MultilinearPolynomial<F>>,
    ) -> Self {
        let trace = sm.get_prover_data().1;
        let log_T = trace.len().log_2();
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut ps = ReadRafProverState::new(trace, eq_r_cycle, unbound_ra_polys);
        ps.init_phase(0);
        let r_cycle = sm
            .get_opening_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::LookupOutput))
            .unwrap()
            .r
            .clone();
        Self {
            gamma,
            gamma_squared: gamma.square(),
            prover_state: Some(ps),
            prod_ra_claims: None,
            table_flag_claims: None,
            raf_flag_claim: None,
            r_cycle,
            rv_claim: sm.get_spartan_z(JoltR1CSInputs::LookupOutput),
            raf_claim: sm.get_spartan_z(JoltR1CSInputs::LeftLookupOperand)
                + gamma * sm.get_spartan_z(JoltR1CSInputs::RightLookupOperand),
            log_T,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let log_T = sm.get_verifier_data().2.log_2();
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let prod_ra_claims: F = (0..D)
            .map(|i| sm.get_opening(OpeningsKeys::InstructionRa(i)))
            .product();
        let table_flag_claims: Vec<F> = (0..LookupTables::<WORD_SIZE>::COUNT)
            .map(|i| sm.get_opening(OpeningsKeys::LookupTableFlag(i)))
            .collect();
        let raf_flag_claim = sm.get_opening(OpeningsKeys::InstructionRafFlag);
        let r_cycle = sm
            .get_opening_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::LookupOutput))
            .unwrap()
            .r
            .clone();
        Self {
            gamma,
            gamma_squared: gamma * gamma,
            prover_state: None,
            prod_ra_claims: Some(prod_ra_claims),
            table_flag_claims: Some(table_flag_claims),
            raf_flag_claim: Some(raf_flag_claim),
            r_cycle,
            rv_claim: sm.get_spartan_z(JoltR1CSInputs::LookupOutput),
            raf_claim: sm.get_spartan_z(JoltR1CSInputs::LeftLookupOperand)
                + gamma * sm.get_spartan_z(JoltR1CSInputs::RightLookupOperand),
            log_T,
        }
    }
}

impl<'a, F: JoltField> ReadRafProverState<F> {
    fn new(
        trace: &'a [RV32IMCycle],
        eq_r_cycle: Vec<F>,
        unbound_ra_polys: Vec<MultilinearPolynomial<F>>,
    ) -> Self {
        let log_T = trace.len().log_2();
        let right_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Right);
        let left_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Left);
        let identity_poly = IdentityPolynomial::new(LOG_K);
        let right_operand_ps =
            PrefixSuffixDecomposition::new(Box::new(right_operand_poly), LOG_M, LOG_K);
        let left_operand_ps =
            PrefixSuffixDecomposition::new(Box::new(left_operand_poly), LOG_M, LOG_K);
        let identity_ps = PrefixSuffixDecomposition::new(Box::new(identity_poly), LOG_M, LOG_K);

        // TODO: This was probably already calculated in Spartan, maybe we should just get it.
        let lookup_indices: Vec<_> = trace
            .par_iter()
            .map(|cycle| LookupBits::new(LookupQuery::<WORD_SIZE>::to_lookup_index(cycle), LOG_K))
            .collect();
        let lookup_indices_by_table: Vec<_> = LookupTables::<WORD_SIZE>::iter()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|table| {
                let table_lookups: Vec<_> = trace
                    .iter()
                    .zip(lookup_indices.iter().cloned())
                    .enumerate()
                    .filter_map(|(j, (cycle, k))| match cycle.lookup_table() {
                        Some(lookup) => {
                            if LookupTables::<WORD_SIZE>::enum_index(&lookup)
                                == LookupTables::enum_index(table)
                            {
                                Some((j, k))
                            } else {
                                None
                            }
                        }
                        None => None,
                    })
                    .collect();
                table_lookups
            })
            .collect();
        let (lookup_indices_uninterleave, lookup_indices_identity): (Vec<_>, Vec<_>) =
            lookup_indices
                .par_iter()
                .cloned()
                .enumerate()
                .zip(trace.par_iter())
                .partition_map(|((idx, item), cycle)| {
                    if cycle
                        .instruction()
                        .circuit_flags()
                        .is_interleaved_operands()
                    {
                        itertools::Either::Left((idx, item))
                    } else {
                        itertools::Either::Right((idx, item))
                    }
                });

        let (is_interleaved_operands, lookup_tables): (Vec<_>, Vec<_>) = trace
            .par_iter()
            .map(|cycle| {
                (
                    cycle
                        .instruction()
                        .circuit_flags()
                        .is_interleaved_operands(),
                    cycle.instruction().lookup_table(),
                )
            })
            .collect();
        let suffix_polys: Vec<Vec<DensePolynomial<F>>> = LookupTables::<WORD_SIZE>::iter()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|table| {
                table
                    .suffixes()
                    .par_iter()
                    .map(|_| DensePolynomial::new(unsafe_allocate_zero_vec(M)))
                    .collect()
            })
            .collect();
        ReadRafProverState {
            r: Vec::with_capacity(log_T + LOG_K),
            ra: Vec::with_capacity(D),
            lookup_tables,
            lookup_indices,
            lookup_indices_by_table,
            lookup_indices_uninterleave,
            lookup_indices_identity,
            is_interleaved_operands,
            prefix_checkpoints: vec![None.into(); Prefixes::COUNT],
            suffix_polys,
            v: std::array::from_fn(|_| ExpandingTable::new(K_CHUNK)),
            u_evals: eq_r_cycle.clone(),
            eq_r_cycle: MultilinearPolynomial::from(eq_r_cycle),
            prefix_registry: PrefixRegistry::new(),
            right_operand_ps,
            left_operand_ps,
            identity_ps,
            combined_val_polynomial: None,
            unbound_ra_polys,
        }
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for ReadRafSumcheck<F> {
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
    fn compute_prover_message(&mut self, round: usize) -> Vec<F> {
        let ps = self.prover_state.as_ref().unwrap();
        if round < LOG_K {
            // Phase 1: First log(K) rounds
            compute_prefix_suffix_prover_message::<WORD_SIZE, F>(
                &ps.prefix_checkpoints,
                &ps.suffix_polys,
                &ps.identity_ps,
                &ps.right_operand_ps,
                &ps.left_operand_ps,
                self.gamma,
                &ps.r,
                round,
            )
            .to_vec()
        } else {
            (0..ps.eq_r_cycle.len() / 2)
                .into_par_iter()
                .map(|i| {
                    let eq_evals = ps
                        .eq_r_cycle
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                    let eq_ra_evals = ps
                        .ra
                        .iter()
                        .map(|ra| ra.sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow))
                        .fold(eq_evals, |mut running, new| {
                            for j in 0..DEGREE {
                                running[j] *= new[j];
                            }
                            running
                        });
                    let val_evals = ps
                        .combined_val_polynomial
                        .as_ref()
                        .unwrap()
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);

                    std::array::from_fn(|i| eq_ra_evals[i] * val_evals[i])
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
    fn bind(&mut self, r_j: F, round: usize) {
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
                s.spawn(|_| ps.v[(round % LOG_M) / LOG_K_CHUNK].update(r_j));
            });
            {
                if ps.r.len().is_multiple_of(2) {
                    Prefixes::update_checkpoints::<WORD_SIZE, F>(
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
            ps.ra
                .par_iter_mut()
                .chain([ps.combined_val_polynomial.as_mut().unwrap()].into_par_iter())
                .chain([&mut ps.eq_r_cycle].into_par_iter())
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
        }
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let (r_address_prime, r_cycle_prime) = r.split_at(LOG_K);
        let left_operand_eval =
            OperandPolynomial::new(LOG_K, OperandSide::Left).evaluate(r_address_prime);
        let right_operand_eval =
            OperandPolynomial::new(LOG_K, OperandSide::Right).evaluate(r_address_prime);
        let identity_poly_eval = IdentityPolynomial::new(LOG_K).evaluate(r_address_prime);
        let val_evals: Vec<_> = LookupTables::<WORD_SIZE>::iter()
            .map(|table| table.evaluate_mle(r_address_prime))
            .collect();
        let eq_eval_cycle = EqPolynomial::mle(&self.r_cycle, r_cycle_prime);

        let rv_val_claim = val_evals
            .into_iter()
            .zip(self.table_flag_claims.as_ref().unwrap())
            .map(|(claim, val)| claim * val)
            .sum::<F>();
        let raf_flag_claim = self.raf_flag_claim.unwrap();
        let ra_claims = self.prod_ra_claims.unwrap();

        let val_eval = rv_val_claim
            + (F::one() - raf_flag_claim)
                * (self.gamma * left_operand_eval + self.gamma_squared * right_operand_eval)
            + raf_flag_claim * self.gamma_squared * identity_poly_eval;
        eq_eval_cycle * ra_claims * val_eval
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> CacheSumcheckOpenings<F, PCS>
    for ReadRafSumcheck<F>
{
    fn cache_openings_prover(
        &mut self,
        accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self.prover_state.as_mut().unwrap();
        let r_cycle_prime = &opening_point.r[LOG_K..];
        let eq_r_cycle_prime = EqPolynomial::evals(r_cycle_prime);

        let flag_claims = std::mem::take(&mut ps.lookup_indices_by_table)
            .into_par_iter()
            .map(|table_lookups| {
                table_lookups
                    .into_iter()
                    .map(|(j, _)| eq_r_cycle_prime[j])
                    .sum::<F>()
            })
            .collect::<Vec<F>>();
        let ra_claims = ps
            .ra
            .iter()
            .map(|ra| ra.final_sumcheck_claim())
            .collect::<Vec<F>>();
        let ra_keys = (0..D).map(OpeningsKeys::InstructionRa).collect::<Vec<_>>();

        let accumulator = accumulator.expect("accumulator is needed");
        flag_claims.into_iter().enumerate().for_each(|(i, claim)| {
            accumulator.borrow_mut().append_virtual(
                OpeningsKeys::LookupTableFlag(i),
                OpeningPoint::new(r_cycle_prime.to_vec()),
                claim,
            );
        });
        ps.unbound_ra_polys
            .iter_mut()
            .zip(ra_claims)
            .zip(ra_keys)
            .enumerate()
            .for_each(|(i, ((ra, claim), key))| {
                accumulator.borrow_mut().append_sparse(
                    vec![std::mem::take(ra)],
                    opening_point.r[LOG_K_CHUNK * i..LOG_K_CHUNK * (i + 1)].to_vec(),
                    opening_point.r[LOG_K..].to_vec(),
                    vec![claim],
                    Some(vec![key]),
                );
            });
        let raf_flag_claim = ps
            .lookup_indices_identity
            .par_iter()
            .map(|(j, _)| eq_r_cycle_prime[*j])
            .sum::<F>();
        accumulator.borrow_mut().append_virtual(
            OpeningsKeys::InstructionRafFlag,
            OpeningPoint::new(r_cycle_prime.to_vec()),
            raf_flag_claim,
        );
    }

    fn cache_openings_verifier(
        &mut self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F, PCS>>>>,
        mut r_sumcheck: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let accumulator = accumulator.expect("accumulator is needed");
        (0..D).for_each(|i| {
            accumulator
                .borrow_mut()
                // TODO: this point is incorrect
                .populate_claim_opening(OpeningsKeys::InstructionRa(i), r_sumcheck.clone())
        });
        let r_cycle_prime = r_sumcheck.split_off(LOG_K_CHUNK);
        accumulator
            .borrow_mut()
            .populate_claim_opening(OpeningsKeys::InstructionRafFlag, r_cycle_prime.clone());
        (0..LookupTables::<WORD_SIZE>::COUNT).for_each(|i| {
            accumulator
                .borrow_mut()
                .populate_claim_opening(OpeningsKeys::LookupTableFlag(i), r_cycle_prime.clone())
        })
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS> for ReadRafSumcheck<F> {}

impl<F: JoltField> ReadRafProverState<F> {
    /// To be called in the beginning of each phase, before any binding
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

                    *u *= (0..RA_PER_LOG_M)
                        .rev()
                        .map(|i| (k_bound >> (LOG_K_CHUNK * i)) % K_CHUNK)
                        .enumerate()
                        .map(|(i, idx)| self.v[i][idx])
                        .product();
                });
        }

        rayon::scope(|s| {
            s.spawn(|_| {
                LookupTables::<WORD_SIZE>::iter()
                    .collect::<Vec<_>>()
                    .par_iter()
                    .zip(self.suffix_polys.par_iter_mut())
                    .zip(self.lookup_indices_by_table.par_iter())
                    .for_each(|((table, polys), lookup_indices)| {
                        table
                            .suffixes()
                            .par_iter()
                            .zip(polys.par_iter_mut())
                            .for_each(|(suffix, poly)| {
                                if phase != 0 {
                                    // Reset polynomial
                                    poly.len = M;
                                    poly.num_vars = poly.len.log_2();
                                    unsafe_zero_slice(&mut poly.Z);
                                }

                                for (j, k) in lookup_indices.iter() {
                                    let (prefix_bits, suffix_bits) =
                                        k.split((PHASES - 1 - phase) * LOG_M);
                                    let t = suffix.suffix_mle::<WORD_SIZE>(suffix_bits);
                                    if t != 0 {
                                        let u = self.u_evals[*j];
                                        poly.Z[prefix_bits % M] += u.mul_u64(t as u64);
                                    }
                                }
                            });
                    });
            });
            s.spawn(|_| {
                self.right_operand_ps
                    .init_Q(&self.u_evals, self.lookup_indices_uninterleave.iter())
            });
            s.spawn(|_| {
                self.left_operand_ps
                    .init_Q(&self.u_evals, self.lookup_indices_uninterleave.iter())
            });
            s.spawn(|_| {
                self.identity_ps
                    .init_Q(&self.u_evals, self.lookup_indices_identity.iter())
            });
        });
        self.identity_ps.init_P(&mut self.prefix_registry);
        self.right_operand_ps.init_P(&mut self.prefix_registry);
        self.left_operand_ps.init_P(&mut self.prefix_registry);

        self.v.par_iter_mut().for_each(|v| v.reset(F::one()));
    }

    /// To be called at the end of each phase, after binding is done
    fn cache_phase(&mut self, phase: usize) {
        self.v
            .par_iter()
            .enumerate()
            .map(|(i, v)| {
                let ra_i: Vec<F> = self
                    .lookup_indices
                    .par_iter()
                    .map(|k| {
                        let (prefix, _) = k.split((PHASES - 1 - phase) * LOG_M);
                        let k_bound: usize =
                            ((prefix % M) >> (LOG_K_CHUNK * (RA_PER_LOG_M - 1 - i))) % K_CHUNK;
                        v[k_bound]
                    })
                    .collect();
                MultilinearPolynomial::from(ra_i)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|ra| {
                self.ra.push(ra);
            });
        self.prefix_registry.update_checkpoints();
    }

    /// To be called before the first log(T) rounds
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
                        .map(|suffix| {
                            F::from_u32(suffix.suffix_mle::<WORD_SIZE>(LookupBits::new(0, 0)))
                        })
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
