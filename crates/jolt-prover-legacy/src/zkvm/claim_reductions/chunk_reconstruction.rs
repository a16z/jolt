//! Unsigned-inc chunk reconstruction sumcheck (lattice/packed mode).
//!
//! One `(cycle ‖ address)`-round reduction over the one-hot chunk columns and
//! the msb column, γ-batching four duties (see the jolt-claims
//! `UnsignedIncChunkReconstruction` relation): per-row hamming weight, the
//! booleanity-point reductions of the chunk and msb openings, and the shifted
//! decode `Σ place·symbol + 2^64·msb = 2^64 + FusedInc` anchored at the
//! `IncVirtualization` cycle point.
//!
//! Cycle rounds bind first (the low bits of the `(symbol ‖ cycle)` cell
//! order). Because every leg is *linear* in the chunk columns, the address
//! hypercube sums collapse exactly during the cycle phase:
//!
//! ```text
//! Σ_k summand(k, j) = eq_inc(j)·Q(j) + eq_bool_cyc(j)·P(j)
//! Q(j) = Σ_i γ^{2i}  +  γ^{2N+1}·(2^64 + FusedInc(j))
//! P(j) = Σ_i γ^{2i+1}·eq_bool_addr[symbol_i(j)]  +  γ^{2N}·msb(j)
//! ```
//!
//! (`Σ_k chunk_i(k,·) ≡ 1` and `Σ_k id(k)·chunk_i(k,·) = symbol_i(·)` are
//! multilinear identities of one-hot columns.) The address phase then binds
//! the cycle-bound chunk columns `H_i(k) = chunk_i(k, ρ_cycle)` — an `O(T)`
//! scatter of the bound eq table by symbol — against the booleanity-address
//! eq table and the identity polynomial.

use allocative::Allocative;

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
#[cfg(feature = "prover")]
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
#[cfg(feature = "prover")]
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, SumcheckId, BIG_ENDIAN, LITTLE_ENDIAN,
};
#[cfg(feature = "prover")]
use crate::poly::unipoly::UniPoly;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint};
#[cfg(feature = "prover")]
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;
use crate::transcripts::Transcript;
use crate::zkvm::packed_witness::UNSIGNED_INC_BITS;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

const DEGREE_BOUND: usize = 2;

#[derive(Allocative, Clone)]
pub struct ChunkReconstructionSumcheckParams<F: JoltField> {
    /// γ^0 .. γ^{2N+1}.
    pub gamma_powers: Vec<F>,
    pub chunk_count: usize,
    pub log_k_chunk: usize,
    pub log_t: usize,
    /// The full booleanity opening point (`r_address ‖ r_cycle`).
    pub r_booleanity: OpeningPoint<BIG_ENDIAN, F>,
    /// The `IncVirtualization` cycle point (where the `FusedInc` claim lives).
    pub r_inc_cycle: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> ChunkReconstructionSumcheckParams<F> {
    pub fn new(
        log_k_chunk: usize,
        log_t: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let chunk_count = UNSIGNED_INC_BITS / log_k_chunk;
        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = Vec::with_capacity(2 * chunk_count + 2);
        let mut power = F::one();
        for _ in 0..2 * chunk_count + 2 {
            gamma_powers.push(power);
            power *= gamma;
        }

        let (r_booleanity, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::UnsignedIncChunk(0),
            SumcheckId::Booleanity,
        );
        let (r_inc_cycle, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::FusedInc,
            SumcheckId::IncVirtualization,
        );

        Self {
            gamma_powers,
            chunk_count,
            log_k_chunk,
            log_t,
            r_booleanity,
            r_inc_cycle,
        }
    }

    fn place_value(&self, index: usize) -> F {
        F::from_u128(1u128 << (self.log_k_chunk * index))
    }

    /// `Σ_i γ^{2i}` — the constant hamming contribution.
    fn hamming_scalar(&self) -> F {
        (0..self.chunk_count)
            .map(|index| self.gamma_powers[2 * index])
            .sum()
    }

    fn msb_reduction_scale(&self) -> F {
        self.gamma_powers[2 * self.chunk_count]
    }

    fn decode_scale(&self) -> F {
        self.gamma_powers[2 * self.chunk_count + 1]
    }

    fn booleanity_address(&self) -> &[F::Challenge] {
        &self.r_booleanity.r[..self.log_k_chunk]
    }

    fn booleanity_cycle(&self) -> &[F::Challenge] {
        &self.r_booleanity.r[self.log_k_chunk..]
    }

    /// The identity MLE `Σ_l 2^{b-1-l}·r[l]` (msb-first) at a bound
    /// big-endian address point — the verifier-side closed form, kept for
    /// the round-loop test (live verification is jolt-verifier's
    /// UnsignedIncChunkReconstruction relation; the e2e pins the bit-order
    /// convention).
    #[cfg(test)]
    fn identity_mle(&self, r_address: &[F::Challenge]) -> F {
        r_address
            .iter()
            .enumerate()
            .map(|(bit, value)| {
                F::from_u64(1 << (self.log_k_chunk - 1 - bit)) * Into::<F>::into(*value)
            })
            .sum()
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ChunkReconstructionSumcheckParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let mut claim = self.hamming_scalar();
        for index in 0..self.chunk_count {
            let (_, chunk_claim) = accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::UnsignedIncChunk(index),
                SumcheckId::Booleanity,
            );
            claim += self.gamma_powers[2 * index + 1] * chunk_claim;
        }
        let (_, msb_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::UnsignedIncMsb,
            SumcheckId::Booleanity,
        );
        claim += self.msb_reduction_scale() * msb_claim;
        let (_, fused_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::FusedInc,
            SumcheckId::IncVirtualization,
        );
        claim + self.decode_scale() * (fused_claim + F::from_u128(1u128 << UNSIGNED_INC_BITS))
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.log_t + self.log_k_chunk
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        unimplemented!("zk x lattice is rejected fail-closed; ChunkReconstruction carries no BlindFold plumbing")
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        unimplemented!("zk x lattice is rejected fail-closed; ChunkReconstruction carries no BlindFold plumbing")
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        unimplemented!("zk x lattice is rejected fail-closed; ChunkReconstruction carries no BlindFold plumbing")
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, _sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        unimplemented!("zk x lattice is rejected fail-closed; ChunkReconstruction carries no BlindFold plumbing")
    }
}

#[cfg(feature = "prover")]
#[derive(Allocative)]
#[expect(
    clippy::large_enum_variant,
    reason = "one phase exists at a time; boxing would add indirection in the round hot path"
)]
enum ChunkReconstructionPhase<F: JoltField> {
    /// The `log_t` cycle rounds: the address hypercube sums are collapsed
    /// into the `P`/`Q` streams (see the module doc).
    Cycle {
        eq_inc: MultilinearPolynomial<F>,
        eq_bool_cycle: MultilinearPolynomial<F>,
        q: MultilinearPolynomial<F>,
        p: MultilinearPolynomial<F>,
        msb: MultilinearPolynomial<F>,
        /// Per-chunk one-hot symbols, kept for the address-phase scatter.
        #[allocative(skip)]
        symbols: Vec<Vec<u8>>,
        cycle_challenges: Vec<F::Challenge>,
    },
    /// The `log_k_chunk` address rounds over the cycle-bound chunk columns.
    Address {
        h: Vec<MultilinearPolynomial<F>>,
        eq_bool_address: MultilinearPolynomial<F>,
        identity: MultilinearPolynomial<F>,
        /// `eq(r_inc, ρ_cycle)`, `eq(r_bool_cycle, ρ_cycle)`, `msb(ρ_cycle)`.
        e_inc: F,
        e_bool_cycle: F,
        msb_bound: F,
        cycle_challenges: Vec<F::Challenge>,
    },
}

#[cfg(feature = "prover")]
#[derive(Allocative)]
pub struct ChunkReconstructionSumcheckProver<F: JoltField> {
    phase: ChunkReconstructionPhase<F>,
    pub params: ChunkReconstructionSumcheckParams<F>,
}

#[cfg(feature = "prover")]
impl<F: JoltField> ChunkReconstructionSumcheckProver<F> {
    /// `symbols[i][j]` is chunk `i`'s hot symbol at cycle `j`; `msb[j]` the
    /// msb bit; `fused[j]` the signed fused delta.
    #[tracing::instrument(skip_all, name = "ChunkReconstructionSumcheckProver::initialize")]
    pub fn initialize(
        params: ChunkReconstructionSumcheckParams<F>,
        symbols: Vec<Vec<u8>>,
        msb: Vec<u8>,
        fused: Vec<i128>,
    ) -> Self {
        use rayon::prelude::*;

        debug_assert_eq!(symbols.len(), params.chunk_count);
        let eq_inc_evals = EqPolynomial::<F>::evals(&params.r_inc_cycle.r);
        let eq_bool_cycle_evals = EqPolynomial::<F>::evals(params.booleanity_cycle());
        let eq_bool_address_evals = EqPolynomial::<F>::evals(params.booleanity_address());

        let hamming_scalar = params.hamming_scalar();
        let decode_scale = params.decode_scale();
        let msb_scale = params.msb_reduction_scale();
        let shift = F::from_u128(1u128 << UNSIGNED_INC_BITS);
        let q: Vec<F> = fused
            .par_iter()
            .map(|delta| hamming_scalar + decode_scale * (shift + F::from_i128(*delta)))
            .collect();
        let p: Vec<F> = (0..msb.len())
            .into_par_iter()
            .map(|j| {
                let mut value = msb_scale * F::from_u64(msb[j] as u64);
                for (index, symbol_column) in symbols.iter().enumerate() {
                    value += params.gamma_powers[2 * index + 1]
                        * eq_bool_address_evals[symbol_column[j] as usize];
                }
                value
            })
            .collect();

        Self {
            phase: ChunkReconstructionPhase::Cycle {
                eq_inc: eq_inc_evals.into(),
                eq_bool_cycle: eq_bool_cycle_evals.into(),
                q: q.into(),
                p: p.into(),
                msb: msb.into(),
                symbols,
                cycle_challenges: Vec::new(),
            },
            params,
        }
    }

    fn transition_to_address_phase(&mut self, r_j: F::Challenge) {
        let ChunkReconstructionPhase::Cycle {
            eq_inc,
            eq_bool_cycle,
            msb,
            symbols,
            cycle_challenges,
            ..
        } = &mut self.phase
        else {
            panic!("transition requires the cycle phase");
        };
        let mut cycle_challenges = std::mem::take(cycle_challenges);
        cycle_challenges.push(r_j);
        eq_inc.bind_parallel(r_j, BindingOrder::LowToHigh);
        eq_bool_cycle.bind_parallel(r_j, BindingOrder::LowToHigh);
        msb.bind_parallel(r_j, BindingOrder::LowToHigh);
        let e_inc = eq_inc.final_sumcheck_claim();
        let e_bool_cycle = eq_bool_cycle.final_sumcheck_claim();
        let msb_bound = msb.final_sumcheck_claim();

        // `H_i(k) = chunk_i(k, ρ_cycle)`: scatter the bound cycle eq table by
        // each column's hot symbols.
        let rho: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(cycle_challenges.clone()).match_endianness();
        let eq_rho = EqPolynomial::<F>::evals(&rho.r);
        let k_chunk = 1usize << self.params.log_k_chunk;
        let h: Vec<MultilinearPolynomial<F>> = symbols
            .iter()
            .map(|symbol_column| {
                let mut column = vec![F::zero(); k_chunk];
                for (j, symbol) in symbol_column.iter().enumerate() {
                    column[*symbol as usize] += eq_rho[j];
                }
                column.into()
            })
            .collect();

        let eq_bool_address = EqPolynomial::<F>::evals(self.params.booleanity_address());
        let identity: Vec<F> = (0..k_chunk).map(|k| F::from_u64(k as u64)).collect();

        self.phase = ChunkReconstructionPhase::Address {
            h,
            eq_bool_address: eq_bool_address.into(),
            identity: identity.into(),
            e_inc,
            e_bool_cycle,
            msb_bound,
            cycle_challenges,
        };
    }
}

#[cfg(feature = "prover")]
impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for ChunkReconstructionSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "ChunkReconstructionSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        use rayon::prelude::*;

        match &self.phase {
            ChunkReconstructionPhase::Cycle {
                eq_inc,
                eq_bool_cycle,
                q,
                p,
                ..
            } => {
                let half_n = eq_inc.len() / 2;
                let evals = (0..half_n)
                    .into_par_iter()
                    .fold(
                        || [F::zero(); DEGREE_BOUND],
                        |mut acc, j| {
                            let eq_inc = eq_inc
                                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                            let q =
                                q.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                            let eq_bool = eq_bool_cycle
                                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                            let p =
                                p.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                            for k in 0..DEGREE_BOUND {
                                acc[k] += eq_inc[k] * q[k] + eq_bool[k] * p[k];
                            }
                            acc
                        },
                    )
                    .reduce(
                        || [F::zero(); DEGREE_BOUND],
                        |mut a, b| {
                            for k in 0..DEGREE_BOUND {
                                a[k] += b[k];
                            }
                            a
                        },
                    );
                UniPoly::from_evals_and_hint(previous_claim, &evals)
            }
            ChunkReconstructionPhase::Address {
                h,
                eq_bool_address,
                identity,
                e_inc,
                e_bool_cycle,
                msb_bound,
                ..
            } => {
                let params = &self.params;
                let msb_weight = *msb_bound
                    * (params.msb_reduction_scale() * *e_bool_cycle
                        + params.decode_scale()
                            * F::from_u128(1u128 << UNSIGNED_INC_BITS)
                            * *e_inc);
                let half_n = eq_bool_address.len() / 2;
                let evals = (0..half_n)
                    .into_par_iter()
                    .fold(
                        || [F::zero(); DEGREE_BOUND],
                        |mut acc, k| {
                            let eq_bool_addr = eq_bool_address
                                .sumcheck_evals_array::<DEGREE_BOUND>(k, BindingOrder::LowToHigh);
                            let id = identity
                                .sumcheck_evals_array::<DEGREE_BOUND>(k, BindingOrder::LowToHigh);
                            for (index, h_i) in h.iter().enumerate() {
                                let h_evals = h_i.sumcheck_evals_array::<DEGREE_BOUND>(
                                    k,
                                    BindingOrder::LowToHigh,
                                );
                                let gamma_even = params.gamma_powers[2 * index];
                                let gamma_odd = params.gamma_powers[2 * index + 1];
                                let place = params.place_value(index);
                                for t in 0..DEGREE_BOUND {
                                    acc[t] += h_evals[t]
                                        * (gamma_even * *e_inc
                                            + gamma_odd * *e_bool_cycle * eq_bool_addr[t]
                                            + params.decode_scale() * place * *e_inc * id[t]);
                                }
                            }
                            for t in 0..DEGREE_BOUND {
                                acc[t] += msb_weight * eq_bool_addr[t];
                            }
                            acc
                        },
                    )
                    .reduce(
                        || [F::zero(); DEGREE_BOUND],
                        |mut a, b| {
                            for k in 0..DEGREE_BOUND {
                                a[k] += b[k];
                            }
                            a
                        },
                    );
                UniPoly::from_evals_and_hint(previous_claim, &evals)
            }
        }
    }

    #[tracing::instrument(skip_all, name = "ChunkReconstructionSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        match &mut self.phase {
            ChunkReconstructionPhase::Cycle {
                eq_inc,
                eq_bool_cycle,
                q,
                p,
                msb,
                cycle_challenges,
                ..
            } => {
                if eq_inc.len() == 2 {
                    self.transition_to_address_phase(r_j);
                    return;
                }
                cycle_challenges.push(r_j);
                eq_inc.bind_parallel(r_j, BindingOrder::LowToHigh);
                eq_bool_cycle.bind_parallel(r_j, BindingOrder::LowToHigh);
                q.bind_parallel(r_j, BindingOrder::LowToHigh);
                p.bind_parallel(r_j, BindingOrder::LowToHigh);
                msb.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
            ChunkReconstructionPhase::Address {
                h,
                eq_bool_address,
                identity,
                ..
            } => {
                for h_i in h.iter_mut() {
                    h_i.bind_parallel(r_j, BindingOrder::LowToHigh);
                }
                eq_bool_address.bind_parallel(r_j, BindingOrder::LowToHigh);
                identity.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let ChunkReconstructionPhase::Address { h, msb_bound, .. } = &self.phase else {
            panic!("Should finish sumcheck in the address phase");
        };

        let opening_point = SumcheckInstanceProver::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);
        let (r_address, r_cycle) = opening_point.r.split_at(self.params.log_k_chunk);

        let polynomials = (0..self.params.chunk_count)
            .map(CommittedPolynomial::UnsignedIncChunk)
            .collect::<Vec<_>>();
        let claims = h
            .iter()
            .map(MultilinearPolynomial::final_sumcheck_claim)
            .collect::<Vec<_>>();
        accumulator.append_sparse(
            polynomials,
            SumcheckId::UnsignedIncChunkReconstruction,
            r_address.to_vec(),
            r_cycle.to_vec(),
            claims,
        );
        accumulator.append_dense(
            CommittedPolynomial::UnsignedIncMsb,
            SumcheckId::UnsignedIncChunkReconstruction,
            r_cycle.to_vec(),
            *msb_bound,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

#[cfg(all(test, feature = "prover"))]
mod tests {
    use super::*;
    use crate::transcripts::Blake2bTranscript;
    use ark_bn254::Fr;

    type Challenge = <Fr as JoltField>::Challenge;

    const LOG_T: usize = 3;
    const T: usize = 1 << LOG_T;
    const LOG_K_CHUNK: usize = 8;
    const CHUNKS: usize = 8;

    fn cycle_point(seed: u64, len: usize) -> Vec<Challenge> {
        (0..len)
            .map(|i| Challenge::from((seed + 5 * i as u64 + 2) as u128))
            .collect()
    }

    /// Drives the full two-phase round loop over a synthetic fused trace and
    /// checks every message folds the previous claim and the final claim
    /// equals the verifier's closed form over the cached openings.
    #[test]
    fn round_loop_reduces_to_the_reconstruction_openings() {
        let fused: Vec<i128> = vec![5, -7, 0, (1 << 63) - 1, -(1 << 63), 123, -456, 0];
        let symbols: Vec<Vec<u8>> = (0..CHUNKS)
            .map(|index| {
                fused
                    .iter()
                    .map(|delta| {
                        let unsigned = (delta + (1i128 << 64)) as u128;
                        ((unsigned >> (LOG_K_CHUNK * index)) & 0xff) as u8
                    })
                    .collect()
            })
            .collect();
        let msb: Vec<u8> = fused
            .iter()
            .map(|delta| (((delta + (1i128 << 64)) as u128) >> 64) as u8)
            .collect();

        // Booleanity openings at (r_bool_addr ‖ r_bool_cycle): evaluate each
        // one-hot chunk column and the msb column directly.
        let r_bool_address = cycle_point(3, LOG_K_CHUNK);
        let r_bool_cycle = cycle_point(17, LOG_T);
        let r_inc_cycle = cycle_point(29, LOG_T);
        let eq_bool_address = EqPolynomial::<Fr>::evals(&r_bool_address);
        let eq_bool_cycle = EqPolynomial::<Fr>::evals(&r_bool_cycle);
        let eq_inc = EqPolynomial::<Fr>::evals(&r_inc_cycle);

        let chunk_bool_claim = |index: usize| -> Fr {
            (0..T)
                .map(|j| eq_bool_address[symbols[index][j] as usize] * eq_bool_cycle[j])
                .sum()
        };
        let msb_bool_claim: Fr = (0..T)
            .map(|j| Fr::from_u64(msb[j] as u64) * eq_bool_cycle[j])
            .sum();
        let fused_claim: Fr = (0..T).map(|j| Fr::from_i128(fused[j]) * eq_inc[j]).sum();

        let mut accumulator = ProverOpeningAccumulator::<Fr>::new(LOG_T);
        let bool_point = [r_bool_address.clone(), r_bool_cycle.clone()].concat();
        accumulator.append_sparse(
            (0..CHUNKS)
                .map(CommittedPolynomial::UnsignedIncChunk)
                .collect(),
            SumcheckId::Booleanity,
            r_bool_address.clone(),
            r_bool_cycle.clone(),
            (0..CHUNKS).map(chunk_bool_claim).collect(),
        );
        accumulator.append_dense(
            CommittedPolynomial::UnsignedIncMsb,
            SumcheckId::Booleanity,
            r_bool_cycle.clone(),
            msb_bool_claim,
        );
        accumulator.append_virtual(
            VirtualPolynomial::FusedInc,
            SumcheckId::IncVirtualization,
            OpeningPoint::new(r_inc_cycle.clone()),
            fused_claim,
        );
        assert_eq!(bool_point.len(), LOG_K_CHUNK + LOG_T);

        let mut transcript = Blake2bTranscript::new(b"chunk-reconstruction-test");
        let params = ChunkReconstructionSumcheckParams::<Fr>::new(
            LOG_K_CHUNK,
            LOG_T,
            &accumulator,
            &mut transcript,
        );
        let mut prover = ChunkReconstructionSumcheckProver::initialize(
            params.clone(),
            symbols.clone(),
            msb.clone(),
            fused.clone(),
        );

        let mut claim = params.input_claim(&accumulator);
        let mut challenges = Vec::new();
        for round in 0..params.num_rounds() {
            let message = SumcheckInstanceProver::<Fr, Blake2bTranscript>::compute_message(
                &mut prover,
                round,
                claim,
            );
            assert_eq!(
                message.eval_at_zero() + message.eval_at_one(),
                claim,
                "round {round} message must fold the previous claim"
            );
            let r_j = Challenge::from((200 + 9 * round) as u128);
            claim = message.evaluate(&r_j);
            challenges.push(r_j);
            SumcheckInstanceProver::<Fr, Blake2bTranscript>::ingest_challenge(
                &mut prover,
                r_j,
                round,
            );
        }

        SumcheckInstanceProver::<Fr, Blake2bTranscript>::cache_openings(
            &prover,
            &mut accumulator,
            &challenges,
        );
        // The verifier's closed form over the cached openings.
        let opening_point = params.normalize_opening_point(&challenges);
        let (r_address_bound, r_cycle_bound) = opening_point.r.split_at(LOG_K_CHUNK);
        let eq_bool_addr: Fr = EqPolynomial::mle(r_address_bound, params.booleanity_address());
        let eq_bc: Fr = EqPolynomial::mle(r_cycle_bound, params.booleanity_cycle());
        let eq_ic: Fr = EqPolynomial::mle(r_cycle_bound, &params.r_inc_cycle.r);
        let identity = params.identity_mle(r_address_bound);
        let mut expected = Fr::from_u64(0);
        for index in 0..CHUNKS {
            let (_, chunk_claim) = accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::UnsignedIncChunk(index),
                SumcheckId::UnsignedIncChunkReconstruction,
            );
            expected += chunk_claim
                * (params.gamma_powers[2 * index] * eq_ic
                    + params.gamma_powers[2 * index + 1] * eq_bool_addr * eq_bc
                    + params.decode_scale() * params.place_value(index) * identity * eq_ic);
        }
        let (_, msb_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::UnsignedIncMsb,
            SumcheckId::UnsignedIncChunkReconstruction,
        );
        expected += msb_claim
            * (params.msb_reduction_scale() * eq_bool_addr * eq_bc
                + params.decode_scale() * Fr::from_u128(1u128 << 64) * eq_bool_addr * eq_ic);
        assert_eq!(claim, expected, "final claim must equal the closed form");

        // The cached chunk openings must decode: evaluate a chunk column
        // directly at the produced point and compare.
        let eq_address = EqPolynomial::<Fr>::evals(r_address_bound);
        let eq_cycle = EqPolynomial::<Fr>::evals(r_cycle_bound);
        let (_, chunk0_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::UnsignedIncChunk(0),
            SumcheckId::UnsignedIncChunkReconstruction,
        );
        let direct: Fr = (0..T)
            .map(|j| eq_address[symbols[0][j] as usize] * eq_cycle[j])
            .sum();
        assert_eq!(chunk0_claim, direct);
    }
}
