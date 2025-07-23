#[cfg(feature = "prover")]
mod prover;

use crate::field::JoltField;
use crate::jolt::vm::ram::remap_address;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
use crate::poly::program_io_polynomial::ProgramIOPolynomial;
use crate::poly::range_mask_polynomial::RangeMaskPolynomial;
use crate::subprotocols::shout::ExpandingTable;
use crate::subprotocols::sumcheck::{BatchableSumcheckVerifierInstance, SumcheckInstanceProof};
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::transcript::Transcript;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::RAM_START_ADDRESS;
use common::jolt_device::JoltDevice;

struct OutputSumcheckProverState<F: JoltField> {
    /// Val(k, 0)
    val_init: MultilinearPolynomial<F>,
    /// The MLE of the final RAM state
    val_final: MultilinearPolynomial<F>,
    /// Val_io(k) = Val_final(k) if k is in the "IO" region of memory,
    /// and 0 otherwise.
    /// Equivalently, Val_io(k) = Val(k, T) * io_mask(k) for
    /// k \in {0, 1}^log(K)
    val_io: MultilinearPolynomial<F>,
    /// EQ(k, r_address)
    eq_poly: MultilinearPolynomial<F>,
    /// io_mask(k) serves as a "mask" for the IO region of memory,
    /// i.e. io_mask(k) = 1 if k is in the "IO" region of memory,
    /// and 0 otherwise.
    io_mask: MultilinearPolynomial<F>,
    /// Updated to contain the table of evaluations
    /// EQ(x_1, ..., x_k, r_1, ..., r_k), where r_i is the
    /// random challenge for the i'th round of sumcheck.
    eq_table: ExpandingTable<F>,
}

struct OutputSumcheckVerifierState<F: JoltField> {
    r_address: Vec<F>,
    program_io: JoltDevice,
}

impl<F: JoltField> OutputSumcheckVerifierState<F> {
    fn initialize(r_address: &[F], program_io: &JoltDevice) -> Self {
        Self {
            r_address: r_address.to_vec(),
            program_io: program_io.clone(),
        }
    }
}

/// Proves that the final RAM state is consistent with the claimed
/// program output.
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct OutputProof<F: JoltField, ProofTranscript: Transcript> {
    output_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    val_final_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// Claimed evaluation Val_final(r_address) output by `OutputSumcheck`,
    /// proven using `ValFinalSumcheck`
    val_final_claim: F,
    /// Claimed evaluations Inc(r_cycle) and wa(r_cycle) output by `ValFinalSumcheck`
    output_claims: ValFinalSumcheckClaims<F>,
}

/// Sumcheck for the zero-check
///   0 = \sum_k eq(r_address, k) * io_range(k) * (Val_final(k) - Val_io(k))
/// In plain English: the final memory state (Val_final) should be consistent with
/// the expected program outputs (Val_io) at the indices where the program
/// inputs/outputs are stored (io_range).
pub struct OutputSumcheck<F: JoltField> {
    K: usize,
    T: usize,
    verifier_state: Option<OutputSumcheckVerifierState<F>>,
    prover_state: Option<OutputSumcheckProverState<F>>,
    /// Claimed evaluation Val_final(r_address) output by `OutputSumcheck`,
    /// proven using `ValFinalSumcheck`
    val_final_claim: Option<F>,
}

impl<F: JoltField> OutputSumcheck<F> {
    pub fn verify<ProofTranscript: Transcript>(
        program_io: &JoltDevice,
        val_init: MultilinearPolynomial<F>,
        r_address: &[F],
        T: usize,
        proof: &OutputProof<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let K = r_address.len().pow2();
        let output_sumcheck_verifier_state = OutputSumcheckVerifierState {
            program_io: program_io.clone(),
            r_address: r_address.to_vec(),
        };

        let output_sumcheck = OutputSumcheck {
            K,
            T,
            verifier_state: Some(output_sumcheck_verifier_state),
            prover_state: None,
            val_final_claim: Some(proof.val_final_claim),
        };

        let r_address_prime =
            output_sumcheck.verify_single(&proof.output_sumcheck_proof, transcript)?;

        let val_final_sumcheck = ValFinalSumcheck {
            T,
            prover_state: None,
            val_init_eval: val_init.evaluate(&r_address_prime),
            val_final_claim: output_sumcheck.val_final_claim.unwrap(),
            output_claims: Some(proof.output_claims.clone()),
        };
        let _r_cycle_prime =
            val_final_sumcheck.verify_single(&proof.val_final_sumcheck_proof, transcript)?;

        Ok(())
    }
}

impl<F: JoltField, ProofTranscript: Transcript>
    BatchableSumcheckVerifierInstance<F, ProofTranscript> for OutputSumcheck<F>
{
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let OutputSumcheckVerifierState {
            r_address,
            program_io,
        } = self.verifier_state.as_ref().unwrap();
        let val_final_claim = self.val_final_claim.as_ref().unwrap();

        let r_address_prime = &r[..r_address.len()];

        let io_mask = RangeMaskPolynomial::new(
            remap_address(
                program_io.memory_layout.input_start,
                &program_io.memory_layout,
            ),
            remap_address(RAM_START_ADDRESS, &program_io.memory_layout),
        );
        let val_io = ProgramIOPolynomial::new(program_io);

        let eq_eval = EqPolynomial::mle(r_address, r_address_prime);
        let io_mask_eval = io_mask.evaluate_mle(r_address_prime);
        let val_io_eval = val_io.evaluate(r_address_prime);

        // Recall that the sumcheck expression is:
        //   0 = \sum_k eq(r_address, k) * io_range(k) * (Val_final(k) - Val_io(k))
        eq_eval * io_mask_eval * (*val_final_claim - val_io_eval)
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone, Default)]
pub struct ValFinalSumcheckClaims<F: JoltField> {
    inc_claim: F,
    wa_claim: F,
}

struct ValFinalSumcheckProverState<F: JoltField> {
    inc: MultilinearPolynomial<F>,
    wa: MultilinearPolynomial<F>,
}

/// This sumcheck virtualizes Val_final(k) as:
/// Val_final(k) = Val_init(k) + \sum_k Inc(j) * wa(k, j)
///   or equivalently:
/// Val_final(k) - Val_init(k) = \sum_k Inc(j) * wa(k, j)
/// We feed the output claim Val_final(r_address) from `OutputSumcheck`
/// into this sumcheck, which reduces it to claims about `Inc` and `wa`.
/// Note that the verifier is assumed to be able to evaluate Val_init
/// on its own.
pub struct ValFinalSumcheck<F: JoltField> {
    T: usize,
    prover_state: Option<ValFinalSumcheckProverState<F>>,
    val_init_eval: F,
    val_final_claim: F,
    output_claims: Option<ValFinalSumcheckClaims<F>>,
}

impl<F: JoltField, ProofTranscript: Transcript>
    BatchableSumcheckVerifierInstance<F, ProofTranscript> for ValFinalSumcheck<F>
{
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.T.log_2()
    }

    fn input_claim(&self) -> F {
        self.val_final_claim - self.val_init_eval
    }
    fn expected_output_claim(&self, _: &[F]) -> F {
        let ValFinalSumcheckClaims {
            inc_claim,
            wa_claim,
        } = self.output_claims.as_ref().unwrap();
        *inc_claim * wa_claim
    }
}
