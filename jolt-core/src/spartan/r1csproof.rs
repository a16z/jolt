#![allow(clippy::too_many_arguments)]
use std::marker::PhantomData;

use crate::{
    field::JoltField,
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial},
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{math::Math, transcript::Transcript},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::r1csinstance::R1CSInstance;

#[derive(Debug)]
pub struct R1CSProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    // comm_vars: PolyCommitment,
    outer_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    // claims_phase2: (
    //     CompressedGroup,
    //     CompressedGroup,
    //     CompressedGroup,
    //     CompressedGroup,
    // ),
    inner_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    // comm_vars_at_ry: CompressedGroup,
    _marker: PhantomData<ProofTranscript>,
}

// #[derive(Serialize, Deserialize)]
// pub struct R1CSSumcheckGens {
//     gens_1: MultiCommitGens,
//     gens_3: MultiCommitGens,
//     gens_4: MultiCommitGens,
// }

// // TODO: fix passing gens_1_ref
// impl R1CSSumcheckGens {
//     pub fn new(label: &'static [u8], gens_1_ref: &MultiCommitGens) -> Self {
//         let gens_1 = gens_1_ref.clone();
//         let gens_3 = MultiCommitGens::new(3, label);
//         let gens_4 = MultiCommitGens::new(4, label);

//         R1CSSumcheckGens {
//             gens_1,
//             gens_3,
//             gens_4,
//         }
//     }
// }

// #[derive(Serialize, Deserialize)]
// pub struct R1CSGens {
//     gens_sc: R1CSSumcheckGens,
//     gens_pc: PolyCommitmentGens,
// }

// impl R1CSGens {
//     pub fn new(label: &'static [u8], _num_cons: usize, num_vars: usize) -> Self {
//         let num_poly_vars = num_vars.log_2();
//         let gens_pc = PolyCommitmentGens::new(num_poly_vars, label);
//         let gens_sc = R1CSSumcheckGens::new(label, &gens_pc.gens.gens_1);
//         R1CSGens { gens_sc, gens_pc }
//     }
// }

impl<F: JoltField, ProofTranscript: Transcript> R1CSProof<F, ProofTranscript> {
    fn protocol_name() -> &'static [u8] {
        b"R1CS proof"
    }

    pub fn prove(
        inst: &R1CSInstance<F>,
        vars: Vec<F>,
        input: &[F],
        // gens: &R1CSGens,
        transcript: &mut ProofTranscript,
    ) -> (R1CSProof<F, ProofTranscript>, Vec<F>, Vec<F>) {
        // we currently require the number of |inputs| + 1 to be at most number of vars
        assert!(input.len() < vars.len());

        // let (poly_vars, comm_vars, blinds_vars) = {
        //     // create a multilinear polynomial using the supplied assignment for variables
        //     let poly_vars = DensePolynomial::new(vars.clone());

        //     // produce a commitment to the satisfying assignment
        //     // let (comm_vars, blinds_vars) = poly_vars.commit(&gens.gens_pc, Some(random_tape));

        //     // add the commitment to the prover's transcript
        //     comm_vars.append_to_transcript(b"poly_commitment", transcript);
        //     (poly_vars, comm_vars, blinds_vars)
        // };

        // append input to variables to create a single vector z
        let z = {
            let num_inputs = input.len();
            let num_vars = vars.len();
            let mut z = vars;
            z.extend(&vec![F::one()]); // add constant term in z
            z.extend(input);
            z.extend(&vec![F::zero(); num_vars - num_inputs - 1]); // we will pad with zeros
            DensePolynomial::new(z)
        };

        // derive the verifier's challenge tau
        let (num_rounds_x, num_rounds_y) = (inst.get_num_cons().log_2(), z.len().log_2());
        let tau = transcript.challenge_vector(num_rounds_x);

        // compute the initial evaluation table for R(\tau, x)

        let eq_tau = DensePolynomial::new(EqPolynomial::evals(&tau));

        let (az, bz, cz) = inst.multiply_vec(inst.get_num_cons(), z.len(), &z.Z);
        let comb_func = |polys: &[F]| -> F { polys[0] * (polys[1] * polys[2] - polys[3]) };

        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
            SumcheckInstanceProof::prove_arbitrary(
                &F::zero(), // claim is zero
                num_rounds_x,
                &mut [eq_tau.clone(), az, bz, cz].to_vec(),
                comb_func,
                3,
                transcript,
            );

        //TODO:- Do we need to reverse?
        // let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();

        ProofTranscript::append_scalars(transcript, &outer_sumcheck_claims);

        // claims from the end of sum-check
        // claim_Az is the (scalar) value v_A = \sum_y A(r_x, y) * z(r_x) where r_x is the sumcheck randomness
        let (claim_Az, claim_Bz, claim_Cz): (F, F, F) = (
            outer_sumcheck_claims[0],
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
        );

        let r_inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let r_inner_sumcheck_RLC_square = r_inner_sumcheck_RLC * r_inner_sumcheck_RLC;
        let claim_inner_joint =
            claim_Az + r_inner_sumcheck_RLC * claim_Bz + r_inner_sumcheck_RLC_square * claim_Cz;

        let poly_ABC = {
            // compute the initial evaluation table for R(\tau, x)
            let (evals_A, evals_B, evals_C) =
                inst.compute_eval_table_sparse(inst.get_num_cons(), z.len(), eq_tau.evals_ref());

            assert_eq!(evals_A.len(), evals_B.len());
            assert_eq!(evals_A.len(), evals_C.len());
            DensePolynomial::new(
                (0..evals_A.len())
                    .into_par_iter()
                    .map(|i| {
                        evals_A[i]
                            + r_inner_sumcheck_RLC * evals_B[i]
                            + r_inner_sumcheck_RLC_square * evals_C[i]
                    })
                    .collect::<Vec<F>>(),
            )
        };

        let comb_func = |polys: &[F]| -> F { polys[0] * polys[1] };
        let (inner_sumcheck_proof, inner_sumcheck_r, _claims_inner) =
            SumcheckInstanceProof::prove_arbitrary(
                &claim_inner_joint,
                num_rounds_y,
                &mut [poly_ABC, z].to_vec(),
                comb_func,
                2,
                transcript,
            );

        // let eval_vars_at_ry = poly_vars.evaluate(&ry[1..]);

        // prove the final step of sum-check #2
        // let blind_eval_Z_at_ry = (Scalar::one() - ry[0]) * blind_eval;
        // let blind_expected_claim_postsc2 = claims_phase2[1] * blind_eval_Z_at_ry;
        // let claim_post_phase2 = claims_phase2[0] * claims_phase2[1];

        (
            R1CSProof {
                // comm_vars,
                outer_sumcheck_proof,
                // claims_phase2: (
                //     comm_Az_claim,
                //     comm_Bz_claim,
                //     comm_Cz_claim,
                //     comm_prod_Az_Bz_claims,
                // ),
                inner_sumcheck_proof,
                _marker: PhantomData,
            },
            outer_sumcheck_r,
            inner_sumcheck_r,
        )
    }

    // pub fn verify(
    //     &self,
    //     num_vars: usize,
    //     num_cons: usize,
    //     input: &[Scalar],
    //     evals: &(Scalar, Scalar, Scalar),
    //     transcript: &mut Transcript,
    //     gens: &R1CSGens,
    // ) -> Result<(Vec<Scalar>, Vec<Scalar>), ProofVerifyError> {
    //     transcript.append_protocol_name(R1CSProof::protocol_name());

    //     input.append_to_transcript(b"input", transcript);

    //     let n = num_vars;
    //     // add the commitment to the verifier's transcript
    //     self.comm_vars
    //         .append_to_transcript(b"poly_commitment", transcript);

    //     let (num_rounds_x, num_rounds_y) = (num_cons.log_2(), (2 * num_vars).log_2());

    //     // derive the verifier's challenge tau
    //     let tau = transcript.challenge_vector(b"challenge_tau", num_rounds_x);

    //     // verify the first sum-check instance
    //     let claim_phase1 = Scalar::zero()
    //         .commit(&Scalar::zero(), &gens.gens_sc.gens_1)
    //         .compress();
    //     let (comm_claim_post_phase1, rx) = self.sc_proof_phase1.verify(
    //         &claim_phase1,
    //         num_rounds_x,
    //         3,
    //         &gens.gens_sc.gens_1,
    //         &gens.gens_sc.gens_4,
    //         transcript,
    //     )?;
    //     // perform the intermediate sum-check test with claimed Az, Bz, and Cz
    //     let (comm_Az_claim, comm_Bz_claim, comm_Cz_claim, comm_prod_Az_Bz_claims) =
    //         &self.claims_phase2;
    //     let (pok_Cz_claim, proof_prod) = &self.pok_claims_phase2;

    //     pok_Cz_claim.verify(&gens.gens_sc.gens_1, transcript, comm_Cz_claim)?;
    //     proof_prod.verify(
    //         &gens.gens_sc.gens_1,
    //         transcript,
    //         comm_Az_claim,
    //         comm_Bz_claim,
    //         comm_prod_Az_Bz_claims,
    //     )?;

    //     comm_Az_claim.append_to_transcript(b"comm_Az_claim", transcript);
    //     comm_Bz_claim.append_to_transcript(b"comm_Bz_claim", transcript);
    //     comm_Cz_claim.append_to_transcript(b"comm_Cz_claim", transcript);
    //     comm_prod_Az_Bz_claims.append_to_transcript(b"comm_prod_Az_Bz_claims", transcript);

    //     let taus_bound_rx: Scalar = (0..rx.len())
    //         .map(|i| rx[i] * tau[i] + (Scalar::one() - rx[i]) * (Scalar::one() - tau[i]))
    //         .product();
    //     let expected_claim_post_phase1 = (taus_bound_rx
    //         * (comm_prod_Az_Bz_claims.decompress().unwrap() - comm_Cz_claim.decompress().unwrap()))
    //     .compress();

    //     // verify proof that expected_claim_post_phase1 == claim_post_phase1
    //     self.proof_eq_sc_phase1.verify(
    //         &gens.gens_sc.gens_1,
    //         transcript,
    //         &expected_claim_post_phase1,
    //         &comm_claim_post_phase1,
    //     )?;

    //     // derive three public challenges and then derive a joint claim
    //     let r_A = transcript.challenge_scalar(b"challenege_Az");
    //     let r_B = transcript.challenge_scalar(b"challenege_Bz");
    //     let r_C = transcript.challenge_scalar(b"challenege_Cz");

    //     // r_A * comm_Az_claim + r_B * comm_Bz_claim + r_C * comm_Cz_claim;
    //     let comm_claim_phase2 = GroupElement::vartime_multiscalar_mul(
    //         iter::once(&r_A)
    //             .chain(iter::once(&r_B))
    //             .chain(iter::once(&r_C)),
    //         iter::once(&comm_Az_claim)
    //             .chain(iter::once(&comm_Bz_claim))
    //             .chain(iter::once(&comm_Cz_claim))
    //             .map(|pt| pt.decompress().unwrap())
    //             .collect::<Vec<GroupElement>>(),
    //     )
    //     .compress();

    //     // verify the joint claim with a sum-check protocol
    //     let (comm_claim_post_phase2, ry) = self.sc_proof_phase2.verify(
    //         &comm_claim_phase2,
    //         num_rounds_y,
    //         2,
    //         &gens.gens_sc.gens_1,
    //         &gens.gens_sc.gens_3,
    //         transcript,
    //     )?;

    //     // verify Z(ry) proof against the initial commitment
    //     self.proof_eval_vars_at_ry.verify(
    //         &gens.gens_pc,
    //         transcript,
    //         &ry[1..],
    //         &self.comm_vars_at_ry,
    //         &self.comm_vars,
    //     )?;

    //     let poly_input_eval = {
    //         // constant term
    //         let mut input_as_sparse_poly_entries = vec![SparsePolyEntry::new(0, Scalar::one())];
    //         //remaining inputs
    //         input_as_sparse_poly_entries.extend(
    //             (0..input.len())
    //                 .map(|i| SparsePolyEntry::new(i + 1, input[i]))
    //                 .collect::<Vec<SparsePolyEntry>>(),
    //         );
    //         SparsePolynomial::new(n.log_2(), input_as_sparse_poly_entries).evaluate(&ry[1..])
    //     };

    //     // compute commitment to eval_Z_at_ry = (Scalar::one() - ry[0]) * self.eval_vars_at_ry + ry[0] * poly_input_eval
    //     let comm_eval_Z_at_ry = GroupElement::vartime_multiscalar_mul(
    //         iter::once(Scalar::one() - ry[0]).chain(iter::once(ry[0])),
    //         iter::once(&self.comm_vars_at_ry.decompress().unwrap()).chain(iter::once(
    //             &poly_input_eval.commit(&Scalar::zero(), &gens.gens_pc.gens.gens_1),
    //         )),
    //     );

    //     // perform the final check in the second sum-check protocol
    //     let (eval_A_r, eval_B_r, eval_C_r) = evals;
    //     let expected_claim_post_phase2 =
    //         ((r_A * eval_A_r + r_B * eval_B_r + r_C * eval_C_r) * comm_eval_Z_at_ry).compress();
    //     // verify proof that expected_claim_post_phase1 == claim_post_phase1
    //     self.proof_eq_sc_phase2.verify(
    //         &gens.gens_sc.gens_1,
    //         transcript,
    //         &expected_claim_post_phase2,
    //         &comm_claim_post_phase2,
    //     )?;

    //     Ok((rx, ry))
    // }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use rand::rngs::OsRng;

//     fn produce_tiny_r1cs() -> (R1CSInstance, Vec<Scalar>, Vec<Scalar>) {
//         // three constraints over five variables Z1, Z2, Z3, Z4, and Z5
//         // rounded to the nearest power of two
//         let num_cons = 128;
//         let num_vars = 256;
//         let num_inputs = 2;

//         // encode the above constraints into three matrices
//         let mut A: Vec<(usize, usize, Scalar)> = Vec::new();
//         let mut B: Vec<(usize, usize, Scalar)> = Vec::new();
//         let mut C: Vec<(usize, usize, Scalar)> = Vec::new();

//         let one = Scalar::one();
//         // constraint 0 entries
//         // (Z1 + Z2) * I0 - Z3 = 0;
//         A.push((0, 0, one));
//         A.push((0, 1, one));
//         B.push((0, num_vars + 1, one));
//         C.push((0, 2, one));

//         // constraint 1 entries
//         // (Z1 + I1) * (Z3) - Z4 = 0
//         A.push((1, 0, one));
//         A.push((1, num_vars + 2, one));
//         B.push((1, 2, one));
//         C.push((1, 3, one));
//         // constraint 3 entries
//         // Z5 * 1 - 0 = 0
//         A.push((2, 4, one));
//         B.push((2, num_vars, one));

//         let inst = R1CSInstance::new(num_cons, num_vars, num_inputs, &A, &B, &C);

//         // compute a satisfying assignment
//         let mut csprng: OsRng = OsRng;
//         let i0 = Scalar::random(&mut csprng);
//         let i1 = Scalar::random(&mut csprng);
//         let z1 = Scalar::random(&mut csprng);
//         let z2 = Scalar::random(&mut csprng);
//         let z3 = (z1 + z2) * i0; // constraint 1: (Z1 + Z2) * I0 - Z3 = 0;
//         let z4 = (z1 + i1) * z3; // constraint 2: (Z1 + I1) * (Z3) - Z4 = 0
//         let z5 = Scalar::zero(); //constraint 3

//         let mut vars = vec![Scalar::zero(); num_vars];
//         vars[0] = z1;
//         vars[1] = z2;
//         vars[2] = z3;
//         vars[3] = z4;
//         vars[4] = z5;

//         let mut input = vec![Scalar::zero(); num_inputs];
//         input[0] = i0;
//         input[1] = i1;

//         (inst, vars, input)
//     }

//     #[test]
//     fn test_tiny_r1cs() {
//         let (inst, vars, input) = tests::produce_tiny_r1cs();
//         let is_sat = inst.is_sat(&vars, &input);
//         assert!(is_sat);
//     }

//     #[test]
//     fn test_synthetic_r1cs() {
//         let (inst, vars, input) = R1CSInstance::produce_synthetic_r1cs(1024, 1024, 10);
//         let is_sat = inst.is_sat(&vars, &input);
//         assert!(is_sat);
//     }

//     #[test]
//     pub fn check_r1cs_proof() {
//         let num_vars = 1024;
//         let num_cons = num_vars;
//         let num_inputs = 10;
//         let (inst, vars, input) =
//             R1CSInstance::produce_synthetic_r1cs(num_cons, num_vars, num_inputs);

//         let gens = R1CSGens::new(b"test-m", num_cons, num_vars);

//         let mut random_tape = RandomTape::new(b"proof");
//         let mut prover_transcript = Transcript::new(b"example");
//         let (proof, rx, ry) = R1CSProof::prove(
//             &inst,
//             vars,
//             &input,
//             &gens,
//             &mut prover_transcript,
//             &mut random_tape,
//         );

//         let inst_evals = inst.evaluate(&rx, &ry);

//         let mut verifier_transcript = Transcript::new(b"example");
//         assert!(proof
//             .verify(
//                 inst.get_num_vars(),
//                 inst.get_num_cons(),
//                 &input,
//                 &inst_evals,
//                 &mut verifier_transcript,
//                 &gens,
//             )
//             .is_ok());
//     }
// }
