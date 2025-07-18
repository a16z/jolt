use crate::field::JoltField;
use crate::subprotocols::sumcheck::{BatchableSumcheckVerifierInstance, SumcheckInstanceProof};
use crate::utils::transcript::Transcript;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::{JoltDevice, MemoryLayout};

#[cfg(feature = "prover")]
mod prover;

use crate::poly::multilinear_polynomial::{MultilinearPolynomial};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;

use tracer::instruction::RV32IMCycle;
use crate::poly::eq_poly::EqPolynomial;

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone, Default)]
pub struct ReadWriteSumcheckClaims<F: JoltField> {
    pub val_claim: F,
    ra_claim: F,
    inc_claim: F,
}

pub struct RamReadWriteChecking<F: JoltField> {
    K: usize,
    T: usize,
    z: F,
    prover_state: Option<ReadWriteCheckingProverState<F>>,
    verifier_state: Option<ReadWriteCheckingVerifierState<F>>,
    claims: Option<ReadWriteSumcheckClaims<F>>,
    memory_layout: MemoryLayout,
    // TODO(moodlezoup): Wire these claims in from Spartan
    rv_claim: F,
    wv_claim: F,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RamReadWriteCheckingProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    sumcheck_switch_index: usize,
    pub claims: ReadWriteSumcheckClaims<F>,
    // TODO(moodlezoup): Wire these claims in from Spartan
    rv_claim: F,
    wv_claim: F,
}

struct ReadWriteCheckingVerifierState<F: JoltField> {
    r_prime: Vec<F>,
    sumcheck_switch_index: usize,
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
    ra: Option<MultilinearPolynomial<F>>,
    val: Option<MultilinearPolynomial<F>>,
}

/// A collection of vectors that are used in each of the first log(T / num_chunks)
/// rounds of sumcheck. There is one `DataBuffers` struct per thread/chunk, reused
/// across all log(T / num_chunks) rounds.
struct DataBuffers<F: JoltField> {
    /// Contains
    ///     Val(k, j', 0, ..., 0)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i)
    val_j_0: Vec<F>,
    /// `val_j_r[0]` contains
    ///     Val(k, j'', 0, r_i, ..., r_1)
    /// `val_j_r[1]` contains
    ///     Val(k, j'', 1, r_i, ..., r_1)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i)
    val_j_r: [Vec<F>; 2],
    /// `ra[0]` contains
    ///     ra(k, j'', 0, r_i, ..., r_1)
    /// `ra[1]` contains
    ///     ra(k, j'', 1, r_i, ..., r_1)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i),
    ra: [Vec<F>; 2],
    dirty_indices: Vec<usize>,
}

impl<F: JoltField> RamReadWriteChecking<F> {
    pub fn verify<ProofTranscript: Transcript>(
        proof: &RamReadWriteCheckingProof<F, ProofTranscript>,
        program_io: &JoltDevice,
        K: usize,
        r_prime: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(Vec<F>, Vec<F>), ProofVerifyError> {
        let sumcheck_instance = Self::new_verifier(proof, program_io, K, r_prime, transcript);
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
        proof: &RamReadWriteCheckingProof<F, ProofTranscript>,
        program_io: &JoltDevice,
        K: usize,
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
            K,
            T,
            z,
            prover_state: None,
            verifier_state: Some(verifier_state),
            claims: Some(proof.claims.clone()),
            memory_layout: program_io.memory_layout.clone(),
            rv_claim: proof.rv_claim,
            wv_claim: proof.wv_claim,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckVerifierInstance<F, ProofTranscript>
for RamReadWriteChecking<F>
{
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2() + self.T.log_2()
    }

    fn input_claim(&self) -> F {
        self.rv_claim + self.z * self.wv_claim
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
            * claims.ra_claim
            * (claims.val_claim + self.z * (claims.val_claim + claims.inc_claim))
    }
}
