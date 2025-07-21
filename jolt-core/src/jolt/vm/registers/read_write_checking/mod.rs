use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::subprotocols::sumcheck::{BatchableSumcheckVerifierInstance, SumcheckInstanceProof};
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::transcript::Transcript;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::REGISTER_COUNT;
use fixedbitset::FixedBitSet;
use tracer::instruction::RV32IMCycle;

#[cfg(feature = "prover")]
mod prover;

const K: usize = REGISTER_COUNT as usize;

/// A collection of vectors that are used in each of the first log(T / num_chunks)
/// rounds of sumcheck. There is one `DataBuffers` struct per thread/chunk, reused
/// across all log(T / num_chunks) rounds.
struct DataBuffers<F: JoltField> {
    /// Contains
    ///     Val(k, j', 0, ..., 0)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i)
    val_j_0: [F; K],
    /// `val_j_r[0]` contains
    ///     Val(k, j'', 0, r_i, ..., r_1)
    /// `val_j_r[1]` contains
    ///     Val(k, j'', 1, r_i, ..., r_1)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i)
    val_j_r: [[F; K]; 2],
    /// `ra[0]` contains
    ///     ra(k, j'', 0, r_i, ..., r_1)
    /// `ra[1]` contains
    ///     ra(k, j'', 1, r_i, ..., r_1)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i),
    rs1_ra: [[F; K]; 2],
    rs2_ra: [[F; K]; 2],
    /// `wa[0]` contains
    ///     wa(k, j'', 0, r_i, ..., r_1)
    /// `wa[1]` contains
    ///     wa(k, j'', 1, r_i, ..., r_1)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i),
    /// where j'' are the higher (log(T) - i - 1) bits of j'
    rd_wa: [[F; K]; 2],
    dirty_indices: FixedBitSet,
}

struct ReadWriteCheckingProverState<F: JoltField> {
    trace: Vec<RV32IMCycle>,
    chunk_size: usize,
    val_checkpoints: Vec<F>,
    data_buffers: Vec<DataBuffers<F>>,
    I: Vec<Vec<(usize, usize, F, F)>>,
    A: Vec<F>,
    eq_r_prime: MultilinearPolynomial<F>,
    gruens_eq_r_prime: GruenSplitEqPolynomial<F>,
    inc_cycle: MultilinearPolynomial<F>,
    // The following polynomials are instantiated after
    // the first phase
    rs1_ra: Option<MultilinearPolynomial<F>>,
    rs2_ra: Option<MultilinearPolynomial<F>>,
    rd_wa: Option<MultilinearPolynomial<F>>,
    val: Option<MultilinearPolynomial<F>>,
}
struct ReadWriteCheckingVerifierState<F: JoltField> {
    r_prime: Vec<F>,
    sumcheck_switch_index: usize,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone, Default)]
pub struct ReadWriteSumcheckClaims<F: JoltField> {
    pub val_claim: F,
    rs1_ra_claim: F,
    rs2_ra_claim: F,
    rd_wa_claim: F,
    inc_claim: F,
}

pub struct RegistersReadWriteChecking<F: JoltField> {
    T: usize,
    z: F,
    z_squared: F,
    prover_state: Option<ReadWriteCheckingProverState<F>>,
    verifier_state: Option<ReadWriteCheckingVerifierState<F>>,
    claims: Option<ReadWriteSumcheckClaims<F>>,
    // TODO(moodlezoup): Wire these claims in from Spartan
    rs1_rv_claim: F,
    rs2_rv_claim: F,
    rd_wv_claim: F,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RegistersReadWriteCheckingProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    sumcheck_switch_index: usize,
    pub claims: ReadWriteSumcheckClaims<F>,
    // TODO(moodlezoup): Wire these claims in from Spartan
    rs1_rv_claim: F,
    rs2_rv_claim: F,
    rd_wv_claim: F,
}

impl<F: JoltField> RegistersReadWriteChecking<F> {
    pub fn verify<ProofTranscript: Transcript>(
        proof: &RegistersReadWriteCheckingProof<F, ProofTranscript>,
        r_prime: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(Vec<F>, Vec<F>), ProofVerifyError> {
        let sumcheck_instance = Self::new_verifier(proof, r_prime, transcript);
        let r_sumcheck = sumcheck_instance.verify_single(&proof.sumcheck_proof, transcript)?;
        let sumcheck_switch_index = proof.sumcheck_switch_index;
        let T = 1 << r_prime.len();

        // The high-order cycle variables are bound after the switch
        let mut r_cycle = r_sumcheck[sumcheck_switch_index..T.log_2()].to_vec();
        // First `sumcheck_switch_index` rounds bind cycle variables from low to high
        r_cycle.extend(r_sumcheck[..sumcheck_switch_index].iter().rev());
        let r_address = r_sumcheck[T.log_2()..].to_vec();

        Ok((r_address, r_cycle))
    }

    fn new_verifier<ProofTranscript: Transcript>(
        proof: &RegistersReadWriteCheckingProof<F, ProofTranscript>,
        r_prime: &[F],
        transcript: &mut ProofTranscript,
    ) -> Self {
        let T = 1 << r_prime.len();
        let z = transcript.challenge_scalar();

        let verifier_state = ReadWriteCheckingVerifierState {
            sumcheck_switch_index: proof.sumcheck_switch_index,
            r_prime: r_prime.to_vec(),
        };

        Self {
            T,
            z,
            z_squared: z.square(),
            prover_state: None,
            verifier_state: Some(verifier_state),
            claims: Some(proof.claims.clone()),
            rs1_rv_claim: proof.rs1_rv_claim,
            rs2_rv_claim: proof.rs2_rv_claim,
            rd_wv_claim: proof.rd_wv_claim,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript>
    BatchableSumcheckVerifierInstance<F, ProofTranscript> for RegistersReadWriteChecking<F>
{
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        K.log_2() + self.T.log_2()
    }

    fn input_claim(&self) -> F {
        self.rd_wv_claim + self.z * self.rs1_rv_claim + self.z_squared * self.rs2_rv_claim
    }
    fn expected_output_claim(&self, r: &[F]) -> F {
        let ReadWriteCheckingVerifierState {
            sumcheck_switch_index,
            r_prime,
            ..
        } = self.verifier_state.as_ref().unwrap();

        // The high-order cycle variables are bound after the switch
        let mut r_cycle = r[*sumcheck_switch_index..self.T.log_2()].to_vec();
        // First `sumcheck_switch_index` rounds bind cycle variables from low to high
        r_cycle.extend(r[..*sumcheck_switch_index].iter().rev());

        // eq(r', r_cycle)
        let eq_eval_cycle = EqPolynomial::mle(r_prime, &r_cycle);

        let claims = self.claims.as_ref().unwrap();
        eq_eval_cycle
            * (claims.rd_wa_claim * (claims.inc_claim + claims.val_claim)
                + self.z * claims.rs1_ra_claim * claims.val_claim
                + self.z_squared * claims.rs2_ra_claim * claims.val_claim)
    }
}
