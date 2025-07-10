use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::{Arc, Mutex},
};
use strum::{EnumCount, IntoEnumIterator};
use tracer::instruction::RV32IMCycle;

use crate::{
    field::JoltField,
    jolt::{
        instruction::{
            CircuitFlags, InstructionFlags, InstructionLookup, InterleavedBitsMarker, LookupQuery,
        },
        lookup_table::{
            prefixes::{PrefixCheckpoint, PrefixEval, Prefixes},
            LookupTables,
        },
        vm::{
            state_manager::{Openings, OpeningsKeys, StateManager},
            JoltCommitments, JoltProverPreprocessing,
        },
        witness::CommittedPolynomials,
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::{Endianness, IdentityPolynomial, OperandPolynomial, OperandSide},
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
        prefix_suffix::{Prefix, PrefixRegistry, PrefixSuffixDecomposition},
    },
    r1cs::inputs::JoltR1CSInputs,
    subprotocols::{
        sparse_dense_shout::{
            compute_sumcheck_prover_message, prove_sparse_dense_shout, verify_sparse_dense_shout,
            ExpandingTable, LookupBits,
        },
        sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    },
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::{unsafe_allocate_zero_vec, unsafe_zero_slice},
        transcript::Transcript,
    },
};

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct LookupsProof<const WORD_SIZE: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    read_checking_proof: ReadCheckingProof<F, ProofTranscript>,
    booleanity_proof: BooleanityProof<F, ProofTranscript>,
    hamming_weight_proof: HammingWeightProof<F, ProofTranscript>,
    log_T: usize,
    _marker: PhantomData<PCS>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ReadCheckingProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    rv_claim: F,
    ra_claims: [F; 4],
    add_sub_mul_flag_claim: F,
    flag_claims: Vec<F>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct BooleanityProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claims: [F; 4],
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct HammingWeightProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claims: [F; 4],
}

impl<const WORD_SIZE: usize, F, PCS, ProofTranscript>
    LookupsProof<WORD_SIZE, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    pub fn generate_witness(_preprocessing: (), _lookups: &[LookupTables<WORD_SIZE>]) {}

    #[tracing::instrument(skip_all, name = "LookupsProof::prove")]
    pub fn prove(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[RV32IMCycle],
        opening_accumulator: &mut ProverOpeningAccumulator<F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let log_T = trace.len().log_2();
        let r_cycle: Vec<F> = transcript.challenge_vector(log_T);
        let (
            read_checking_sumcheck,
            rv_claim,
            ra_claims,
            add_sub_mul_flag_claim,
            flag_claims,
            eq_r_cycle,
        ) = prove_sparse_dense_shout::<WORD_SIZE, _, _>(trace, &r_cycle, transcript);
        let read_checking_proof = ReadCheckingProof {
            sumcheck_proof: read_checking_sumcheck,
            rv_claim,
            ra_claims,
            add_sub_mul_flag_claim,
            flag_claims,
        };

        let r_address = transcript.challenge_vector(16);
        let F = compute_ra_evals(trace, &eq_r_cycle);
        let mut sm = StateManager {
            transcript,
            prover_accumulator: Some(Arc::new(Mutex::new(opening_accumulator))),
            verifier_accumulator: None,
            openings: Arc::new(Mutex::new(HashMap::new())),
            r_address: r_address.clone(),
        };
        sm.openings.lock().unwrap().insert(
            OpeningsKeys::SpartanZ(JoltR1CSInputs::Imm),
            (r_cycle.clone(), F::zero()),
        );

        let mut booleanity = BooleanitySumcheck::new(&mut sm, trace, &eq_r_cycle, F.clone());
        let (booleanity_proof, _) = booleanity.prove_single(sm.transcript);
        // let ra_claims = (0..4)
        //     .map(|i| sm.openings(OpeningsKeys::InstructionRa(i)))
        //     .collect::<Vec<F>>()
        //     .try_into()
        //     .unwrap();

        // TODO(moodlezoup): Openings
        let booleanity_proof = BooleanityProof {
            sumcheck_proof: booleanity_proof,
            ra_claims: booleanity.ra_claims.unwrap(),
        };

        let mut hamming_weight = HammingWeightSumcheck::new(&mut sm, F);
        let (hamming_weight_sumcheck, r_hamming_weight) =
            hamming_weight.prove_single(sm.transcript);
        // let ra_claims = (0..4)
        //     .map(|i| sm.openings(OpeningsKeys::InstructionRa(i)))
        //     .collect::<Vec<F>>()
        //     .try_into()
        //     .unwrap();

        // TODO(moodlezoup): Openings
        let hamming_weight_proof = HammingWeightProof {
            sumcheck_proof: hamming_weight_sumcheck,
            ra_claims: hamming_weight.ra_claims.unwrap(),
        };

        let unbound_ra_polys = vec![
            CommittedPolynomials::InstructionRa(0).generate_witness(preprocessing, trace),
            CommittedPolynomials::InstructionRa(1).generate_witness(preprocessing, trace),
            CommittedPolynomials::InstructionRa(2).generate_witness(preprocessing, trace),
            CommittedPolynomials::InstructionRa(3).generate_witness(preprocessing, trace),
        ];

        let r_hamming_weight_rev = r_hamming_weight.iter().copied().rev().collect::<Vec<_>>();

        opening_accumulator.append_sparse(
            unbound_ra_polys,
            r_hamming_weight_rev,
            r_cycle,
            hamming_weight.ra_claims.unwrap().to_vec(),
        );

        Self {
            read_checking_proof,
            booleanity_proof,
            hamming_weight_proof,
            log_T,
            _marker: PhantomData,
        }
    }

    pub fn verify(
        &self,
        commitments: &JoltCommitments<F, PCS>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let r_cycle: Vec<F> = transcript.challenge_vector(self.log_T);
        verify_sparse_dense_shout::<WORD_SIZE, _, _>(
            &self.read_checking_proof.sumcheck_proof,
            self.log_T,
            r_cycle.clone(),
            self.read_checking_proof.rv_claim,
            self.read_checking_proof.ra_claims,
            self.read_checking_proof.add_sub_mul_flag_claim,
            &self.read_checking_proof.flag_claims,
            transcript,
        )?;

        let r_address: Vec<F> = transcript.challenge_vector(16);
        let mut sm = StateManager {
            transcript,
            prover_accumulator: None,
            verifier_accumulator: Some(Arc::new(Mutex::new(opening_accumulator))),
            openings: Arc::new(Mutex::new(HashMap::new())),
            r_address: r_address.clone(),
        };
        sm.openings.lock().unwrap().insert(
            OpeningsKeys::SpartanZ(JoltR1CSInputs::Imm),
            (r_cycle.clone(), F::zero()),
        );
        for i in 0..4 {
            sm.openings.lock().unwrap().insert(
                OpeningsKeys::InstructionRa(i),
                (Vec::new(), self.booleanity_proof.ra_claims[i]),
            );
        }

        let booleanity = BooleanitySumcheck::new_verifier(&mut sm);
        let _r_booleanity =
            booleanity.verify_single(&self.booleanity_proof.sumcheck_proof, sm.transcript)?;

        for i in 0..4 {
            sm.openings.lock().unwrap().insert(
                OpeningsKeys::InstructionRa(i),
                (Vec::new(), self.hamming_weight_proof.ra_claims[i]),
            );
        }
        let hamming_weight = HammingWeightSumcheck::new_verifier(&mut sm);
        let r_hamming_weight = hamming_weight
            .verify_single(&self.hamming_weight_proof.sumcheck_proof, sm.transcript)
            .unwrap();

        let r_hamming_weight: Vec<_> = r_hamming_weight.iter().copied().rev().collect();
        for i in 0..4 {
            opening_accumulator.append(
                &[&commitments.commitments[CommittedPolynomials::InstructionRa(i).to_index()]],
                [r_hamming_weight.as_slice(), r_cycle.as_slice()].concat(),
                &[self.hamming_weight_proof.ra_claims[i]],
                transcript,
            );
        }

        Ok(())
    }
}
const WORD_SIZE: usize = 32;
const LOG_K: usize = WORD_SIZE * 2;
const PHASES: usize = 4;
const LOG_M: usize = LOG_K / PHASES;
const M: usize = 1 << LOG_M;
const D: usize = 4;
const LOG_K_CHUNK: usize = LOG_K / D;
const K_CHUNK: usize = 1 << LOG_K_CHUNK;

struct ReadRafProverState<'a, F: JoltField> {
    trace: &'a [RV32IMCycle],
    ra: Vec<MultilinearPolynomial<F>>,
    r: Vec<F>,

    lookup_tables: Vec<LookupTables<WORD_SIZE>>,
    lookup_indices: Vec<LookupBits>,
    lookup_indices_by_table: Vec<Vec<(usize, LookupBits)>>,
    lookup_indices_uninterleave: Vec<(usize, LookupBits)>,
    lookup_indices_identity: Vec<(usize, LookupBits)>,

    prefix_checkpoints: Vec<PrefixCheckpoint<F>>,
    suffix_polys: Vec<Vec<DensePolynomial<F>>>,
    v: ExpandingTable<F>,
    u_evals: Vec<F>,
    eq_r_cycle_evals: Vec<F>,
    eq_r_cycle: MultilinearPolynomial<F>,

    prefix_registry: PrefixRegistry<F>,
    right_operand_ps: PrefixSuffixDecomposition<F, 2>,
    left_operand_ps: PrefixSuffixDecomposition<F, 2>,
    identity_ps: PrefixSuffixDecomposition<F, 2>,

    combined_val_polynomial: Option<MultilinearPolynomial<F>>,
}

pub struct ReadRafSumcheck<'a, F: JoltField> {
    gamma: F,
    gamma_squared: F,
    prover_state: Option<ReadRafProverState<'a, F>>,
    r_cycle: Vec<F>,
    rv_claim: F,
    raf_claim: F,
    openings: Arc<Mutex<Openings<F>>>,
    log_T: usize,
}

impl<'a, F: JoltField> ReadRafSumcheck<'a, F> {
    pub fn new<T: Transcript>(
        sm: &mut StateManager<F, impl CommitmentScheme<T, Field = F>, T>,
        trace: &'a [RV32IMCycle],
        eq_r_cycle: &[F],
    ) -> Self {
        let log_T = trace.len().log_2();
        let right_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Left);
        let left_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Right);
        let identity_poly = IdentityPolynomial::new_with_endianness(LOG_K, Endianness::Big);
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
        let lookup_tables: Vec<_> = LookupTables::<WORD_SIZE>::iter().collect();
        let lookup_indices_by_table: Vec<_> = lookup_tables
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
        let suffix_polys: Vec<Vec<DensePolynomial<F>>> = lookup_tables
            .par_iter()
            .map(|table| {
                table
                    .suffixes()
                    .par_iter()
                    .map(|_| DensePolynomial::new(unsafe_allocate_zero_vec(M)))
                    .collect()
            })
            .collect();
        let gamma: F = sm.transcript.challenge_scalar();
        let mut s = Self {
            gamma,
            gamma_squared: gamma * gamma,
            prover_state: Some(ReadRafProverState {
                trace,
                r: Vec::with_capacity(log_T + LOG_K),
                ra: Vec::with_capacity(4),
                lookup_tables,
                lookup_indices,
                lookup_indices_by_table,
                lookup_indices_uninterleave,
                lookup_indices_identity,
                prefix_checkpoints: vec![None.into(); Prefixes::COUNT],
                suffix_polys,
                v: ExpandingTable::new(M),
                u_evals: eq_r_cycle.to_vec(),
                eq_r_cycle_evals: eq_r_cycle.to_vec(),
                eq_r_cycle: MultilinearPolynomial::from(eq_r_cycle.to_vec()),
                prefix_registry: PrefixRegistry::new(),
                right_operand_ps,
                left_operand_ps,
                identity_ps,
                combined_val_polynomial: None,
            }),
            r_cycle: sm.r_cycle(),
            rv_claim: sm.z(JoltR1CSInputs::LookupOutput),
            raf_claim: sm.z(JoltR1CSInputs::LeftLookupOperand)
                + gamma * sm.z(JoltR1CSInputs::RightLookupOperand),
            log_T,
            openings: sm.openings.clone(),
        };
        s.init_phase(0);
        s
    }

    pub fn new_verifier<T: Transcript>(
        sm: &mut StateManager<F, impl CommitmentScheme<T, Field = F>, T>,
    ) -> Self {
        let log_T = sm.r_cycle().len();
        let gamma: F = sm.transcript.challenge_scalar();
        Self {
            gamma,
            gamma_squared: gamma * gamma,
            prover_state: None,
            r_cycle: sm.r_cycle(),
            rv_claim: sm.z(JoltR1CSInputs::LookupOutput),
            raf_claim: sm.z(JoltR1CSInputs::LeftLookupOperand)
                + gamma * sm.z(JoltR1CSInputs::RightLookupOperand),
            log_T,
            openings: sm.openings.clone(),
        }
    }
}

impl<'a, F: JoltField, T: Transcript> BatchableSumcheckInstance<F, T> for ReadRafSumcheck<'a, F> {
    fn degree(&self) -> usize {
        D + 2
    }

    fn num_rounds(&self) -> usize {
        LOG_K + self.log_T
    }

    fn input_claim(&self) -> F {
        self.rv_claim + self.gamma * self.raf_claim
    }

    fn compute_prover_message(&self, round: usize) -> Vec<F> {
        let ps = self.prover_state.as_ref().unwrap();
        if round < LOG_K {
            // Phase 1: First log(K) rounds
            compute_sumcheck_prover_message::<WORD_SIZE, F>(
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
                    let eq_evals = ps.eq_r_cycle.sumcheck_evals(i, 6, BindingOrder::HighToLow);
                    let ra_0_evals = ps.ra[0].sumcheck_evals(i, 6, BindingOrder::HighToLow);
                    let ra_1_evals = ps.ra[1].sumcheck_evals(i, 6, BindingOrder::HighToLow);
                    let ra_2_evals = ps.ra[2].sumcheck_evals(i, 6, BindingOrder::HighToLow);
                    let ra_3_evals = ps.ra[3].sumcheck_evals(i, 6, BindingOrder::HighToLow);
                    let val_evals = ps.combined_val_polynomial.as_ref().unwrap().sumcheck_evals(
                        i,
                        6,
                        BindingOrder::HighToLow,
                    );

                    std::array::from_fn(|i| {
                        eq_evals[i]
                            * ra_0_evals[i]
                            * ra_1_evals[i]
                            * ra_2_evals[i]
                            * ra_3_evals[i]
                            * val_evals[i]
                    })
                })
                .reduce(
                    || [F::zero(); 6],
                    |running, new| {
                        [
                            running[0] + new[0],
                            running[1] + new[1],
                            running[2] + new[2],
                            running[3] + new[3],
                            running[4] + new[4],
                            running[5] + new[5],
                        ]
                    },
                )
                .to_vec()
        }
    }

    fn bind(&mut self, r_j: F, round: usize) {
        let ps = self.prover_state.as_mut().unwrap();
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
                self.cache_phase(phase);
                // if not last phase, init next phase
                if phase != PHASES - 1 {
                    self.init_phase(phase + 1);
                }
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

    fn cache_openings(&mut self) {
        let ps = self.prover_state.as_mut().unwrap();
        let r_cycle_prime = &ps.r[ps.r.len() - self.log_T..];
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

        let mut openings = self.openings.lock().unwrap();
        flag_claims.into_iter().enumerate().for_each(|(i, claim)| {
            openings.insert(
                OpeningsKeys::InstructionTypeFlag(i),
                (r_cycle_prime.to_vec(), claim),
            );
        });
        ra_claims.into_iter().enumerate().for_each(|(i, claim)| {
            openings.insert(
                OpeningsKeys::InstructionRa(i),
                (r_cycle_prime.to_vec(), claim),
            );
        })
        // raf flag claim should not need to be cached since its derived from the SpartanZ
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let (r_address_prime, r_cycle_prime) = r.split_at(LOG_K);
        let left_operand_eval =
            OperandPolynomial::new(LOG_K, OperandSide::Left).evaluate(r_address_prime);
        let right_operand_eval =
            OperandPolynomial::new(LOG_K, OperandSide::Right).evaluate(r_address_prime);
        let identity_poly_eval = IdentityPolynomial::new_with_endianness(LOG_K, Endianness::Big)
            .evaluate(r_address_prime);
        let val_evals: Vec<_> = LookupTables::<WORD_SIZE>::iter()
            .map(|table| table.evaluate_mle(r_address_prime))
            .collect();
        let eq_eval_cycle = EqPolynomial::mle(&self.r_cycle, r_cycle_prime);

        let openings = self.openings.lock().unwrap();
        let rv_val_claim = (0..LookupTables::<WORD_SIZE>::COUNT)
            .map(|i| openings[&OpeningsKeys::InstructionTypeFlag(i)].1)
            .zip(val_evals.iter())
            .map(|(claim, val)| claim * val)
            .sum::<F>();
        let raf_flag_claim = openings[JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands)]
            + openings[JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands)]
            + openings[JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands)]
            + openings[JoltR1CSInputs::OpFlags(CircuitFlags::Advice)];
        let ra_claims = (0..D)
            .map(|i| openings[&OpeningsKeys::InstructionRa(i)].1)
            .product::<F>();
        drop(openings);

        let val_eval = rv_val_claim
            + (F::one() - raf_flag_claim)
                * (self.gamma * left_operand_eval + self.gamma_squared * right_operand_eval)
            + raf_flag_claim * self.gamma_squared * identity_poly_eval;
        eq_eval_cycle * ra_claims * val_eval
    }
}

impl<'a, F: JoltField> ReadRafSumcheck<'a, F> {
    /// To be called in the beginning of each phase, before any binding
    fn init_phase(&mut self, phase: usize) {
        let ps = self.prover_state.as_mut().unwrap();
        // Condensation
        if phase != 0 {
            let span = tracing::span!(tracing::Level::INFO, "Update u_evals");
            let _guard = span.enter();
            ps.lookup_indices
                .par_iter()
                .zip(ps.u_evals.par_iter_mut())
                .for_each(|(k, u)| {
                    let (prefix, _) = k.split((4 - phase) * LOG_M);
                    let k_bound: usize = prefix % M;
                    *u *= ps.v[k_bound];
                });
        }

        rayon::scope(|s| {
            s.spawn(|_| {
                ps.lookup_tables
                    .par_iter()
                    .zip(ps.suffix_polys.par_iter_mut())
                    .zip(ps.lookup_indices_by_table.par_iter())
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
                                    let (prefix_bits, suffix_bits) = k.split((3 - phase) * LOG_M);
                                    let t = suffix.suffix_mle::<WORD_SIZE>(suffix_bits);
                                    if t != 0 {
                                        let u = ps.u_evals[*j];
                                        poly.Z[prefix_bits % M] += u.mul_u64(t as u64);
                                    }
                                }
                            });
                    });
            });
            s.spawn(|_| {
                ps.right_operand_ps
                    .init_Q(&ps.u_evals, ps.lookup_indices_uninterleave.iter())
            });
            s.spawn(|_| {
                ps.left_operand_ps
                    .init_Q(&ps.u_evals, ps.lookup_indices_uninterleave.iter())
            });
            s.spawn(|_| {
                ps.identity_ps
                    .init_Q(&ps.u_evals, ps.lookup_indices_identity.iter())
            });
        });
        ps.identity_ps.init_P(&mut ps.prefix_registry);
        ps.right_operand_ps.init_P(&mut ps.prefix_registry);
        ps.left_operand_ps.init_P(&mut ps.prefix_registry);

        ps.v.reset(F::one());
    }

    /// To be called at the end of each phase, after binding is done
    fn cache_phase(&mut self, phase: usize) {
        let ps = self.prover_state.as_mut().unwrap();
        let ra_i: Vec<F> = ps
            .lookup_indices
            .par_iter()
            .map(|k| {
                let (prefix, _) = k.split((3 - phase) * LOG_M);
                let k_bound: usize = prefix % M;
                ps.v[k_bound]
            })
            .collect();
        ps.ra.push(MultilinearPolynomial::from(ra_i));
        ps.prefix_registry.update_checkpoints();
    }

    /// To be called before the first log(T) rounds
    fn init_log_t_rounds(&mut self) {
        let ps = self.prover_state.as_mut().unwrap();
        let prefixes: Vec<PrefixEval<F>> = std::mem::take(&mut ps.prefix_checkpoints)
            .into_iter()
            .map(|checkpoint| checkpoint.unwrap())
            .collect();
        let mut combined_val_poly: Vec<F> = unsafe_allocate_zero_vec(self.log_T.pow2());
        combined_val_poly
            .par_iter_mut()
            .zip(ps.trace.par_iter())
            .for_each(|(val, step)| {
                let table: Option<LookupTables<WORD_SIZE>> = step.lookup_table();
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

                if step.instruction().circuit_flags().is_interleaved_operands() {
                    *val += self.gamma
                        * ps.prefix_registry.checkpoints[Prefix::LeftOperand].unwrap()
                        + self.gamma_squared
                            * ps.prefix_registry.checkpoints[Prefix::RightOperand].unwrap();
                } else {
                    *val += self.gamma_squared
                        * ps.prefix_registry.checkpoints[Prefix::Identity].unwrap();
                }
            });
        ps.combined_val_polynomial = Some(MultilinearPolynomial::from(combined_val_poly));
    }
}

struct BooleanityProverState<F: JoltField> {
    B: MultilinearPolynomial<F>,
    D: MultilinearPolynomial<F>,
    G: [Vec<F>; D],
    H_indices: [Vec<usize>; D],
    H: Option<[MultilinearPolynomial<F>; D]>,
    F: Vec<F>,
    eq_r_r: Option<F>,
    eq_km_c: [[F; 3]; 2],
    eq_km_c_squared: [[F; 3]; 2],
}

pub struct BooleanitySumcheck<F: JoltField> {
    /// Precomputed powers of gamma - batching chgallenge
    gamma: [F; D],
    prover_state: Option<BooleanityProverState<F>>,
    r_address: Vec<F>,
    r_cycle: Vec<F>,
    log_T: usize,
    ra_claims: Option<[F; D]>,
}

impl<F: JoltField> BooleanitySumcheck<F> {
    pub fn new<T: Transcript>(
        sm: &mut StateManager<F, impl CommitmentScheme<T, Field = F>, T>,
        trace: &[RV32IMCycle],
        eq_r_cycle: &[F],
        G: [Vec<F>; D],
    ) -> Self {
        const DEGREE: usize = 3;
        let gamma: F = sm.transcript.challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        let B = MultilinearPolynomial::from(EqPolynomial::evals(&sm.r_address()));
        let mut F: Vec<F> = unsafe_allocate_zero_vec(K_CHUNK);
        F[0] = F::one();

        let H_indices: [Vec<usize>; D] = std::array::from_fn(|i| {
            trace
                .par_iter()
                .map(|cycle| {
                    let lookup_index = LookupQuery::<32>::to_lookup_index(cycle);
                    ((lookup_index >> (LOG_K_CHUNK * (3 - i))) % K_CHUNK as u64) as usize
                })
                .collect()
        });

        // EQ(k_m, c) for k_m \in {0, 1} and c \in {0, 2, 3}
        let eq_km_c: [[F; DEGREE]; 2] = [
            [
                F::one(),        // eq(0, 0) = 0 * 0 + (1 - 0) * (1 - 0)
                F::from_i64(-1), // eq(0, 2) = 0 * 2 + (1 - 0) * (1 - 2)
                F::from_i64(-2), // eq(0, 3) = 0 * 3 + (1 - 0) * (1 - 3)
            ],
            [
                F::zero(),     // eq(1, 0) = 1 * 0 + (1 - 1) * (1 - 0)
                F::from_u8(2), // eq(1, 2) = 1 * 2 + (1 - 1) * (1 - 2)
                F::from_u8(3), // eq(1, 3) = 1 * 3 + (1 - 1) * (1 - 3)
            ],
        ];
        // EQ(k_m, c)^2 for k_m \in {0, 1} and c \in {0, 2, 3}
        let eq_km_c_squared: [[F; DEGREE]; 2] = [
            [F::one(), F::one(), F::from_u8(4)],
            [F::zero(), F::from_u8(4), F::from_u8(9)],
        ];
        Self {
            gamma: gamma_powers,
            prover_state: Some(BooleanityProverState {
                B,
                D: MultilinearPolynomial::from(eq_r_cycle.to_vec()),
                G,
                H_indices,
                H: None,
                F,
                eq_r_r: None,
                eq_km_c,
                eq_km_c_squared,
            }),
            r_address: sm.r_address(),
            r_cycle: sm.r_cycle(),
            ra_claims: None,
            log_T: trace.len().log_2(),
        }
    }

    pub fn new_verifier<T: Transcript>(
        sm: &mut StateManager<F, impl CommitmentScheme<T, Field = F>, T>,
    ) -> Self {
        let log_T = sm.r_cycle().len();
        let gamma: F = sm.transcript.challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        Self {
            gamma: gamma_powers,
            prover_state: None,
            r_address: sm.r_address(),
            r_cycle: sm.r_cycle(),
            log_T,
            ra_claims: Some(
                (0..D)
                    .map(|i| sm.openings(OpeningsKeys::InstructionRa(i)))
                    .collect::<Vec<F>>()
                    .try_into()
                    .unwrap(),
            ),
        }
    }
}

impl<F: JoltField, T: Transcript> BatchableSumcheckInstance<F, T> for BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        LOG_K_CHUNK + self.log_T
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn compute_prover_message(&self, round: usize) -> Vec<F> {
        if round < LOG_K_CHUNK {
            // Phase 1: First log(K_CHUNK) rounds
            self.compute_phase1_message(round)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message()
        }
    }

    fn bind(&mut self, r_j: F, round: usize) {
        let prover = self.prover_state.as_mut().unwrap();

        if round < LOG_K_CHUNK {
            // Phase 1: Bind B and update F
            prover.B.bind_parallel(r_j, BindingOrder::LowToHigh);
            // Update F for this round (see Equation 55)
            let (F_left, F_right) = prover.F.split_at_mut(1 << round);
            F_left
                .par_iter_mut()
                .zip(F_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r_j;
                    *x -= *y;
                });
            if round == LOG_K_CHUNK - 1 {
                prover.H = Some(std::array::from_fn(|i| {
                    let coeffs: Vec<F> = std::mem::take(&mut prover.H_indices[i])
                        .into_par_iter()
                        .map(|j| prover.F[j])
                        .collect();
                    MultilinearPolynomial::from(coeffs)
                }));
                prover.eq_r_r = Some(prover.B.final_sumcheck_claim());
            }
        } else {
            // Phase 2: Bind D and H
            prover
                .H
                .as_mut()
                .unwrap()
                .into_par_iter()
                .chain(rayon::iter::once(&mut prover.D))
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn cache_openings(&mut self) {
        self.ra_claims = Some(
            self.prover_state
                .as_ref()
                .unwrap()
                .H
                .as_ref()
                .unwrap()
                .iter()
                .map(|ra| ra.final_sumcheck_claim())
                .collect::<Vec<F>>()
                .try_into()
                .unwrap(),
        );
    }

    fn expected_output_claim(&self, r_prime: &[F]) -> F {
        EqPolynomial::mle(
            r_prime,
            &self
                .r_address
                .iter()
                .cloned()
                .rev()
                .chain(self.r_cycle.iter().cloned().rev())
                .collect::<Vec<F>>(),
        ) * self
            .ra_claims
            .as_ref()
            .unwrap()
            .iter()
            .zip(self.gamma.iter())
            .map(|(&ra, &gamma)| (ra.square() - ra) * gamma)
            .sum::<F>()
    }
}

impl<F: JoltField> BooleanitySumcheck<F> {
    fn compute_phase1_message(&self, round: usize) -> Vec<F> {
        let p = self.prover_state.as_ref().unwrap();
        let m = round + 1;
        const DEGREE: usize = 3;

        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        let univariate_poly_evals: [F; 3] = (0..p.B.len() / 2)
            .into_par_iter()
            .map(|k_prime| {
                let B_evals = p.B.sumcheck_evals(k_prime, DEGREE, BindingOrder::LowToHigh);

                let inner_sum = (0..1 << m)
                    .into_par_iter()
                    .map(|k| {
                        // Since we're binding variables from low to high, k_m is the high bit
                        let k_m = k >> (m - 1);
                        // We then index into F using (k_{m-1}, ..., k_1)
                        let F_k = p.F[k % (1 << (m - 1))];
                        // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                        let k_G = (k_prime << m) + k;
                        let G_times_F =
                            p.G.iter()
                                .zip(self.gamma.iter())
                                .map(|(g, gamma)| g[k_G] * gamma)
                                .sum::<F>()
                                * F_k;
                        // For c \in {0, 2, 3} compute:
                        //    G[k] * (F[k_1, ...., k_{m-1}, c]^2 - F[k_1, ...., k_{m-1}, c])
                        //    = G_times_F * (eq(k_m, c)^2 * F[k_1, ...., k_{m-1}] - eq(k_m, c))
                        [
                            G_times_F * (p.eq_km_c_squared[k_m][0] * F_k - p.eq_km_c[k_m][0]),
                            G_times_F * (p.eq_km_c_squared[k_m][1] * F_k - p.eq_km_c[k_m][1]),
                            G_times_F * (p.eq_km_c_squared[k_m][2] * F_k - p.eq_km_c[k_m][2]),
                        ]
                    })
                    .reduce(
                        || [F::zero(); 3],
                        |running, new| {
                            [
                                running[0] + new[0],
                                running[1] + new[1],
                                running[2] + new[2],
                            ]
                        },
                    );

                [
                    B_evals[0] * inner_sum[0],
                    B_evals[1] * inner_sum[1],
                    B_evals[2] * inner_sum[2],
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        univariate_poly_evals.to_vec()
    }

    fn compute_phase2_message(&self) -> Vec<F> {
        let p = self.prover_state.as_ref().unwrap();
        const DEGREE: usize = 3;
        let univariate_poly_evals: [F; 3] = (0..p.D.len() / 2)
            .into_par_iter()
            .map(|i| {
                let D_evals = p.D.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let H = p.H.as_ref().unwrap();
                let H_evals = [
                    H[0].sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh),
                    H[1].sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh),
                    H[2].sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh),
                    H[3].sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh),
                ];

                let mut evals = [
                    H_evals[0][0].square() - H_evals[0][0],
                    H_evals[0][1].square() - H_evals[0][1],
                    H_evals[0][2].square() - H_evals[0][2],
                ];

                evals[0] += self.gamma[1] * (H_evals[1][0].square() - H_evals[1][0]);
                evals[1] += self.gamma[1] * (H_evals[1][1].square() - H_evals[1][1]);
                evals[2] += self.gamma[1] * (H_evals[1][2].square() - H_evals[1][2]);

                evals[0] += self.gamma[2] * (H_evals[2][0].square() - H_evals[2][0]);
                evals[1] += self.gamma[2] * (H_evals[2][1].square() - H_evals[2][1]);
                evals[2] += self.gamma[2] * (H_evals[2][2].square() - H_evals[2][2]);

                evals[0] += self.gamma[3] * (H_evals[3][0].square() - H_evals[3][0]);
                evals[1] += self.gamma[3] * (H_evals[3][1].square() - H_evals[3][1]);
                evals[2] += self.gamma[3] * (H_evals[3][2].square() - H_evals[3][2]);

                [
                    D_evals[0] * evals[0],
                    D_evals[1] * evals[1],
                    D_evals[2] * evals[2],
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        vec![
            p.eq_r_r.unwrap() * univariate_poly_evals[0],
            p.eq_r_r.unwrap() * univariate_poly_evals[1],
            p.eq_r_r.unwrap() * univariate_poly_evals[2],
        ]
    }
}

struct HammingProverState<F: JoltField> {
    /// ra_i polynomials
    ra: [MultilinearPolynomial<F>; D],
}

pub struct HammingWeightSumcheck<F: JoltField> {
    /// Precomputed powers of gamma - batching chgallenge
    gamma: [F; D],
    prover_state: Option<HammingProverState<F>>,
    ra_claims: Option<[F; D]>,
}

impl<F: JoltField> HammingWeightSumcheck<F> {
    pub fn new<T: Transcript>(
        sm: &mut StateManager<F, impl CommitmentScheme<T, Field = F>, T>,
        F: [Vec<F>; D],
    ) -> Self {
        let gamma: F = sm.transcript.challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let ra = F
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        Self {
            gamma: gamma_powers,
            prover_state: Some(HammingProverState { ra }),
            ra_claims: None,
        }
    }

    pub fn new_verifier<T: Transcript>(
        sm: &mut StateManager<F, impl CommitmentScheme<T, Field = F>, T>,
    ) -> Self {
        let gamma: F = sm.transcript.challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        Self {
            gamma: gamma_powers,
            prover_state: None,
            ra_claims: Some(
                (0..D)
                    .map(|i| sm.openings(OpeningsKeys::InstructionRa(i)))
                    .collect::<Vec<F>>()
                    .try_into()
                    .unwrap(),
            ),
        }
    }
}

impl<F: JoltField, T: Transcript> BatchableSumcheckInstance<F, T> for HammingWeightSumcheck<F> {
    fn degree(&self) -> usize {
        1
    }

    fn num_rounds(&self) -> usize {
        LOG_K_CHUNK
    }

    fn input_claim(&self) -> F {
        self.gamma.iter().sum()
    }

    fn compute_prover_message(&self, _round: usize) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();
        vec![prover_state
            .ra
            .iter()
            .zip(self.gamma.iter())
            .map(|(ra, gamma)| {
                (0..ra.len() / 2)
                    .into_par_iter()
                    .map(|i| ra.get_bound_coeff(2 * i))
                    .sum::<F>()
                    * gamma
            })
            .sum()]
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        self.prover_state
            .as_mut()
            .unwrap()
            .ra
            .par_iter_mut()
            .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh))
    }

    fn cache_openings(&mut self) {
        debug_assert!(self.ra_claims.is_none());
        self.ra_claims = Some(
            self.prover_state
                .as_ref()
                .unwrap()
                .ra
                .iter()
                .map(|ra| ra.final_sumcheck_claim())
                .collect::<Vec<F>>()
                .try_into()
                .unwrap(),
        );
    }

    fn expected_output_claim(&self, _r: &[F]) -> F {
        self.ra_claims
            .as_ref()
            .unwrap()
            .iter()
            .zip(self.gamma.iter())
            .map(|(&ra, &gamma)| ra * gamma)
            .sum()
    }
}

impl<const WORD_SIZE: usize, F, PCS, T> LookupsProof<WORD_SIZE, F, PCS, T>
where
    F: JoltField,
    PCS: CommitmentScheme<T, Field = F>,
    T: Transcript,
{
    pub fn prove_phase_2_sumchecks<'a>(
        sm: &mut StateManager<F, PCS, T>,
        trace: &'a [RV32IMCycle],
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, T> + 'a>> {
        let eq_r_cycle = EqPolynomial::evals(&sm.r_cycle());
        let ra_evals = compute_ra_evals(trace, &eq_r_cycle);

        let read_raf = Box::new(ReadRafSumcheck::new(sm, trace, &eq_r_cycle));
        let booleanity = Box::new(BooleanitySumcheck::new(
            sm,
            trace,
            &eq_r_cycle,
            ra_evals.clone(),
        ));
        let hamming_weight = Box::new(HammingWeightSumcheck::new(sm, ra_evals));

        vec![read_raf, booleanity, hamming_weight]
    }

    pub fn verify_phase_2_sumchecks(
        state_manager: &mut StateManager<F, impl CommitmentScheme<T, Field = F>, T>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, T>>> {
        let read_raf = Box::new(ReadRafSumcheck::new_verifier(state_manager));
        let booleanity = Box::new(BooleanitySumcheck::new_verifier(state_manager));
        let hamming_weight = Box::new(HammingWeightSumcheck::new_verifier(state_manager));

        vec![read_raf, booleanity, hamming_weight]
    }
}

#[inline(always)]
fn compute_ra_evals<F: JoltField>(trace: &[RV32IMCycle], eq_r_cycle: &[F]) -> [Vec<F>; 4] {
    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);

    trace
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, trace_chunk)| {
            let mut result = [
                unsafe_allocate_zero_vec(K_CHUNK),
                unsafe_allocate_zero_vec(K_CHUNK),
                unsafe_allocate_zero_vec(K_CHUNK),
                unsafe_allocate_zero_vec(K_CHUNK),
            ];
            let mut j = chunk_index * chunk_size;
            for cycle in trace_chunk {
                let mut lookup_index = LookupQuery::<32>::to_lookup_index(cycle);
                let k = lookup_index % K_CHUNK as u64;
                result[3][k as usize] += eq_r_cycle[j];

                lookup_index >>= LOG_K_CHUNK;
                let k = lookup_index % K_CHUNK as u64;
                result[2][k as usize] += eq_r_cycle[j];

                lookup_index >>= LOG_K_CHUNK;
                let k = lookup_index % K_CHUNK as u64;
                result[1][k as usize] += eq_r_cycle[j];

                lookup_index >>= LOG_K_CHUNK;
                let k = lookup_index % K_CHUNK as u64;
                result[0][k as usize] += eq_r_cycle[j];
                j += 1;
            }
            result
        })
        .reduce(
            || {
                [
                    unsafe_allocate_zero_vec(K_CHUNK),
                    unsafe_allocate_zero_vec(K_CHUNK),
                    unsafe_allocate_zero_vec(K_CHUNK),
                    unsafe_allocate_zero_vec(K_CHUNK),
                ]
            },
            |mut running, new| {
                running
                    .par_iter_mut()
                    .zip(new.into_par_iter())
                    .for_each(|(x, y)| {
                        x.par_iter_mut()
                            .zip(y.into_par_iter())
                            .for_each(|(x, y)| *x += y)
                    });
                running
            },
        )
}
