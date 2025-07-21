mod lookup_bits;
#[cfg(feature = "prover")]
pub mod prover;
#[cfg(feature = "prover")]
pub mod sparse_dense;

pub use lookup_bits::*;
#[cfg(feature = "prover")]
pub use prover::*;
use rayon::prelude::*;
use std::ops::Index;
use strum::IntoEnumIterator;

use crate::jolt::lookup_table::LookupTables;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::identity_poly::{Endianness, IdentityPolynomial, OperandPolynomial, OperandSide};
use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::subprotocols::sumcheck::{
    BatchableSumcheckVerifierInstance, BatchedSumcheck, SumcheckInstanceProof,
};
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{errors::ProofVerifyError, math::Math, transcript::Transcript},
};

pub struct ShoutProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    core_piop_claims: ShoutSumcheckClaims<F>,
    ra_claim_prime: F,
}

#[allow(dead_code)]
struct ShoutProverState<F: JoltField> {
    K: usize,
    rv_claim: F,
    z: F,
    ra: MultilinearPolynomial<F>,
    val: MultilinearPolynomial<F>,
}

#[derive(Clone)]
struct ShoutSumcheckClaims<F: JoltField> {
    ra_claim: F,
    rv_claim: F,
}

struct ShoutVerifierState<F: JoltField> {
    K: usize,
    z: F,
    val: MultilinearPolynomial<F>,
}

struct ShoutSumcheck<F: JoltField> {
    verifier_state: Option<ShoutVerifierState<F>>,
    prover_state: Option<ShoutProverState<F>>,
    claims: Option<ShoutSumcheckClaims<F>>,
}

impl<F: JoltField> ShoutVerifierState<F> {
    fn initialize<ProofTranscript: Transcript>(
        lookup_table: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let K = lookup_table.len();
        let z: F = transcript.challenge_scalar();
        let val = MultilinearPolynomial::from(lookup_table);
        Self { K, z, val }
    }
}

struct BooleanityVerifierState<F: JoltField> {
    r_address: Vec<F>,
    r_cycle: Vec<F>,
}

impl<F: JoltField> BooleanityVerifierState<F> {
    fn initialize<ProofTranscript: Transcript>(
        r_cycle: &[F],
        K: usize,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let r_cycle: Vec<_> = r_cycle.iter().copied().rev().collect();
        let r_address: Vec<F> = transcript
            .challenge_vector(K.log_2())
            .into_iter()
            .rev()
            .collect();

        Self { r_cycle, r_address }
    }
}

impl<F: JoltField, ProofTranscript: Transcript>
    BatchableSumcheckVerifierInstance<F, ProofTranscript> for ShoutSumcheck<F>
{
    #[inline(always)]
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        if self.prover_state.is_some() {
            self.prover_state.as_ref().unwrap().K.log_2()
        } else if self.verifier_state.is_some() {
            self.verifier_state.as_ref().unwrap().K.log_2()
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        if self.prover_state.is_some() {
            let ShoutProverState { rv_claim, z, .. } = self.prover_state.as_ref().unwrap();
            // Linear combination of the core PIOP claim and the Hamming weight claim (which is 1)
            *rv_claim + z
        } else if self.verifier_state.is_some() {
            let ShoutVerifierState { z, .. } = self.verifier_state.as_ref().unwrap();
            let ShoutSumcheckClaims { rv_claim, .. } = self.claims.as_ref().unwrap();
            // Linear combination of the core PIOP claim and the Hamming weight claim (which is 1)
            *rv_claim + z
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let ShoutVerifierState { z, val, .. } = self.verifier_state.as_ref().unwrap();
        let ShoutSumcheckClaims { ra_claim, .. } = self.claims.as_ref().unwrap();

        let r_address: Vec<F> = r.iter().rev().copied().collect();
        *ra_claim * (*z + val.evaluate(&r_address))
    }
}

#[allow(dead_code)]
struct BooleanityProverState<F: JoltField> {
    read_addresses: Vec<usize>,
    K: usize,
    T: usize,
    B: GruenSplitEqPolynomial<F>,
    #[cfg(test)]
    old_B: MultilinearPolynomial<F>,
    F: Vec<F>,
    G: Vec<F>,
    D: MultilinearPolynomial<F>,
    /// Initialized after first log(K) rounds of sumcheck
    H: Option<MultilinearPolynomial<F>>,
}

struct BooleanitySumcheck<F: JoltField> {
    verifier_state: Option<BooleanityVerifierState<F>>,
    prover_state: Option<BooleanityProverState<F>>,
    ra_claim: Option<F>,
}
impl<F: JoltField, ProofTranscript: Transcript> ShoutProof<F, ProofTranscript> {
    pub fn verify(
        &self,
        lookup_table: Vec<F>,
        r_cycle: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let K = lookup_table.len();

        let core_piop_verifier_state = ShoutVerifierState::initialize(lookup_table, transcript);
        let booleanity_verifier_state = BooleanityVerifierState::initialize(r_cycle, K, transcript);

        let core_piop_sumcheck = ShoutSumcheck {
            prover_state: None,
            verifier_state: Some(core_piop_verifier_state),
            claims: Some(self.core_piop_claims.clone()),
        };

        let booleanity_sumcheck = BooleanitySumcheck {
            prover_state: None,
            verifier_state: Some(booleanity_verifier_state),
            ra_claim: Some(self.ra_claim_prime),
        };

        let _r_sumcheck = BatchedSumcheck::verify(
            &self.sumcheck_proof,
            vec![&core_piop_sumcheck, &booleanity_sumcheck],
            transcript,
        )?;

        // TODO: Reduce 2 ra claims to 1 (Section 4.5.2 of Proofs, Arguments, and Zero-Knowledge)
        // TODO: Append to opening proof accumulator

        Ok(())
    }
}

/// Table containing the evaluations `EQ(x_1, ..., x_j, r_1, ..., r_j)`,
/// built up incrementally as we receive random challenges `r_j` over the
/// course of sumcheck.
#[derive(Clone, Debug)]
pub struct ExpandingTable<F: JoltField> {
    len: usize,
    values: Vec<F>,
    scratch_space: Vec<F>,
}

impl<F: JoltField> ExpandingTable<F> {
    /// Initializes an `ExpandingTable` with the given `capacity`.
    #[tracing::instrument(skip_all, name = "ExpandingTable::new")]
    pub fn new(capacity: usize) -> Self {
        let (values, scratch_space) = rayon::join(
            || unsafe_allocate_zero_vec(capacity),
            || unsafe_allocate_zero_vec(capacity),
        );
        Self {
            len: 0,
            values,
            scratch_space,
        }
    }

    /// Resets this table to be length 1, containing only the given `value`.
    pub fn reset(&mut self, value: F) {
        self.values[0] = value;
        self.len = 1;
    }

    /// Updates this table (expanding it by a factor of 2) to incorporate
    /// the new random challenge `r_j`.
    #[tracing::instrument(skip_all, name = "ExpandingTable::update")]
    pub fn update(&mut self, r_j: F) {
        self.values[..self.len]
            .par_iter()
            .zip(self.scratch_space.par_chunks_mut(2))
            .for_each(|(&v_i, dest)| {
                let eval_1 = r_j * v_i;
                dest[0] = v_i - eval_1;
                dest[1] = eval_1;
            });
        std::mem::swap(&mut self.values, &mut self.scratch_space);
        self.len *= 2;
    }
}

impl<F: JoltField> Index<usize> for ExpandingTable<F> {
    type Output = F;

    fn index(&self, index: usize) -> &F {
        assert!(index < self.len);
        &self.values[index]
    }
}

pub fn verify_sparse_dense_shout<
    const WORD_SIZE: usize,
    F: JoltField,
    ProofTranscript: Transcript,
>(
    proof: &SumcheckInstanceProof<F, ProofTranscript>,
    log_T: usize,
    r_cycle: Vec<F>,
    rv_claim: F,
    ra_claims: [F; 4],
    is_add_mul_sub_flag_claim: F,
    flag_claims: &[F],
    transcript: &mut ProofTranscript,
) -> Result<(), ProofVerifyError> {
    let log_K = 2 * WORD_SIZE;
    let first_log_K_rounds = SumcheckInstanceProof::new(proof.compressed_polys[..log_K].to_vec());
    let last_log_T_rounds = SumcheckInstanceProof::new(proof.compressed_polys[log_K..].to_vec());

    let gamma: F = transcript.challenge_scalar();
    let gamma_squared = gamma.square();

    // The first log(K) rounds' univariate polynomials are degree 2
    let (sumcheck_claim, r_address) = first_log_K_rounds.verify(rv_claim, log_K, 2, transcript)?;
    // The last log(T) rounds' univariate polynomials are degree 6
    let (sumcheck_claim, r_cycle_prime) =
        last_log_T_rounds.verify(sumcheck_claim, log_T, 6, transcript)?;

    let val_evals: Vec<_> = LookupTables::<WORD_SIZE>::iter()
        .map(|table| table.evaluate_mle(&r_address))
        .collect();
    let eq_eval_cycle = EqPolynomial::mle(&r_cycle, &r_cycle_prime);

    let rv_val_claim = flag_claims
        .iter()
        .zip(val_evals.iter())
        .map(|(flag, val)| *flag * val)
        .sum::<F>();

    let right_operand_eval = OperandPolynomial::new(log_K, OperandSide::Right).evaluate(&r_address);
    let left_operand_eval = OperandPolynomial::new(log_K, OperandSide::Left).evaluate(&r_address);
    let identity_poly_eval =
        IdentityPolynomial::new_with_endianness(log_K, Endianness::Big).evaluate(&r_address);

    let val_claim = rv_val_claim
        + (F::one() - is_add_mul_sub_flag_claim)
            * (gamma * right_operand_eval + gamma_squared * left_operand_eval)
        + gamma_squared * is_add_mul_sub_flag_claim * identity_poly_eval;

    assert_eq!(
        eq_eval_cycle * ra_claims.iter().product::<F>() * val_claim,
        sumcheck_claim,
        "Read-checking sumcheck failed"
    );

    Ok(())
}

impl<F: JoltField, ProofTranscript: Transcript>
    BatchableSumcheckVerifierInstance<F, ProofTranscript> for BooleanitySumcheck<F>
{
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        if self.prover_state.is_some() {
            let BooleanityProverState { K, T, .. } = self.prover_state.as_ref().unwrap();
            K.log_2() + T.log_2()
        } else if self.verifier_state.is_some() {
            let BooleanityVerifierState { r_cycle, r_address } =
                self.verifier_state.as_ref().unwrap();
            r_address.len() + r_cycle.len()
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let BooleanityVerifierState { r_address, r_cycle } = self.verifier_state.as_ref().unwrap();
        let (r_address_prime, r_cycle_prime) = r.split_at(r_address.len());
        let ra_claim = self.ra_claim.unwrap();

        EqPolynomial::mle(r_address, r_address_prime)
            * EqPolynomial::mle(r_cycle, r_cycle_prime)
            * (ra_claim.square() - ra_claim)
    }
}
