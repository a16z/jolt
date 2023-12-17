use ark_ec::CurveGroup;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use merlin::Transcript;

use crate::{
    poly::{dense_mlpoly::{DensePolynomial, PolyCommitment, PolyCommitmentGens, PolyEvalProof}},
    utils::{
        errors::ProofVerifyError,
        math::Math,
        random::RandomTape,
        transcript::{AppendToTranscript, ProofTranscript},
    },
};

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct CombinedTableCommitment<G: CurveGroup> {
    pub joint_commitment: PolyCommitment<G>,
}

impl<G: CurveGroup> CombinedTableCommitment<G> {
    pub fn new(joint_commitment: PolyCommitment<G>) -> Self {
        Self { joint_commitment }
    }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct CombinedTableEvalProof<G: CurveGroup> {
    joint_proof: PolyEvalProof<G>,
}

impl<G: CurveGroup> CombinedTableEvalProof<G> {
    fn prove_single(
        joint_poly: Vec<&DensePolynomial<G::ScalarField>>,
        r: &[G::ScalarField],
        evals: &[G::ScalarField],
        gens: &PolyCommitmentGens<G>,
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> PolyEvalProof<G> {
        todo!("[sam] make work");
        // assert_eq!(joint_poly.get_num_vars(), r.len() + evals.len().log_2());

        // // append the claimed evaluations to transcript
        // <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"evals_ops_val", evals);

        // // n-to-1 reduction
        // let (r_joint, eval_joint) = {
        //     let challenges = <Transcript as ProofTranscript<G>>::challenge_vector(
        //         transcript,
        //         b"challenge_combine_n_to_one",
        //         evals.len().log_2(),
        //     );

        //     let mut poly_evals = DensePolynomial::new(evals.to_vec());
        //     for i in (0..challenges.len()).rev() {
        //         poly_evals.bound_poly_var_bot(&challenges[i]);
        //     }
        //     assert_eq!(poly_evals.len(), 1);
        //     let joint_claim_eval = poly_evals[0];
        //     let mut r_joint = challenges;
        //     r_joint.extend(r);

        //     debug_assert_eq!(joint_poly.evaluate(&r_joint), joint_claim_eval);
        //     (r_joint, joint_claim_eval)
        // };
        // // decommit the joint polynomial at r_joint
        // <Transcript as ProofTranscript<G>>::append_scalar(
        //     transcript,
        //     b"joint_claim_eval",
        //     &eval_joint,
        // );

        // let (proof_table_eval, _comm_table_eval) = PolyEvalProof::prove(
        //     joint_poly,
        //     None,
        //     &r_joint,
        //     &eval_joint,
        //     None,
        //     gens,
        //     transcript,
        //     random_tape,
        // );

        // proof_table_eval
    }

    /// evalues both polynomials at r and produces a joint proof of opening
    #[tracing::instrument(skip_all, name = "CombinedEval.prove")]
    pub fn prove(
        combined_poly: Vec<&DensePolynomial<G::ScalarField>>,
        eval_ops_val_vec: &[G::ScalarField],
        r: &[G::ScalarField],
        gens: &PolyCommitmentGens<G>,
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> Self {
        todo!("[sam]: make this function work");
        // <Transcript as ProofTranscript<G>>::append_protocol_name(
        //     transcript,
        //     CombinedTableEvalProof::<G>::protocol_name(),
        // );

        // let evals = {
        //     let mut evals: Vec<G::ScalarField> = eval_ops_val_vec.to_vec();
        //     evals.resize(evals.len().next_power_of_two(), G::ScalarField::zero());
        //     evals.to_vec()
        // };
        // let joint_proof = CombinedTableEvalProof::<G>::prove_single(
        //     combined_poly,
        //     r,
        //     &evals,
        //     gens,
        //     transcript,
        //     random_tape,
        // );

        // CombinedTableEvalProof { joint_proof }
    }

    fn verify_single(
        proof: &PolyEvalProof<G>,
        comm: &PolyCommitment<G>,
        r: &[G::ScalarField],
        evals: &[G::ScalarField],
        gens: &PolyCommitmentGens<G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        // append the claimed evaluations to transcript
        <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"evals_ops_val", evals);

        // n-to-1 reduction
        let challenges = <Transcript as ProofTranscript<G>>::challenge_vector(
            transcript,
            b"challenge_combine_n_to_one",
            evals.len().log_2(),
        );
        let mut poly_evals = DensePolynomial::new(evals.to_vec());
        for i in (0..challenges.len()).rev() {
            poly_evals.bound_poly_var_bot(&challenges[i]);
        }
        assert_eq!(poly_evals.len(), 1);
        let joint_claim_eval = poly_evals[0];
        let mut r_joint = challenges;
        r_joint.extend(r);

        // decommit the joint polynomial at r_joint
        <Transcript as ProofTranscript<G>>::append_scalar(
            transcript,
            b"joint_claim_eval",
            &joint_claim_eval,
        );

        proof.verify_plain(gens, transcript, &r_joint, &joint_claim_eval, comm)
    }

    // verify evaluations of both polynomials at r
    pub fn verify(
        &self,
        r: &[G::ScalarField],
        evals: &[G::ScalarField],
        gens: &PolyCommitmentGens<G>,
        comm: &CombinedTableCommitment<G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        <Transcript as ProofTranscript<G>>::append_protocol_name(
            transcript,
            CombinedTableEvalProof::<G>::protocol_name(),
        );
        let mut evals = evals.to_owned();
        evals.resize(evals.len().next_power_of_two(), G::ScalarField::zero());

        CombinedTableEvalProof::<G>::verify_single(
            &self.joint_proof,
            &comm.joint_commitment,
            r,
            &evals,
            gens,
            transcript,
        )
    }

    fn protocol_name() -> &'static [u8] {
        b"Lasso CombinedTableEvalProof"
    }
}

impl<G: CurveGroup> AppendToTranscript<G> for CombinedTableCommitment<G> {
    fn append_to_transcript<T: ProofTranscript<G>>(
        &self,
        label: &'static [u8],
        transcript: &mut T,
    ) {
        transcript.append_message(
            b"subtable_evals_commitment",
            b"begin_subtable_evals_commitment",
        );
        self.joint_commitment
            .append_to_transcript(label, transcript);
        transcript.append_message(
            b"subtable_evals_commitment",
            b"end_subtable_evals_commitment",
        );
    }
}
