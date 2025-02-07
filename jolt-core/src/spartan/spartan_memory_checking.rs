use crate::field::{JoltField, OptimizedMul};
use crate::jolt::vm::{JoltCommitments, JoltPolynomials, JoltStuff};
use crate::lasso::memory_checking::{
    ExogenousOpenings, Initializable, NoExogenousOpenings, StructuredPolynomialData,
    VerifierComputedOpening,
};
use crate::poly::commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use crate::poly::opening_proof::{
    ProverOpeningAccumulator, ReducedOpeningProof, VerifierOpeningAccumulator,
};
use crate::r1cs::inputs;
use crate::r1cs::spartan::SpartanError;
use crate::subprotocols::grand_product::{
    BatchedDenseGrandProduct, BatchedGrandProduct, BatchedGrandProductLayer,
    BatchedGrandProductProof,
};
use crate::subprotocols::sparse_grand_product::ToggledBatchedGrandProduct;
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::Transcript;
use crate::{
    lasso::memory_checking::{
        MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes,
    },
    poly::{
        dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, identity_poly::IdentityPolynomial,
    },
    utils::errors::ProofVerifyError,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::interleave;
use rayon::prelude::*;
#[cfg(test)]
use std::collections::HashSet;
use std::ops;

use super::sparse_mlpoly::SparseMatPolynomial;
use super::{Assignment, Instance};

#[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct SpartanStuff<T: CanonicalSerialize + CanonicalDeserialize + Sync> {
    read_cts_rows: Vec<T>,
    read_cts_cols: Vec<T>,
    final_cts_rows: Vec<T>,
    final_cts_cols: Vec<T>,
    rows: Vec<T>,
    cols: Vec<T>,
    vals: Vec<T>,
    e_rx: Vec<T>,
    e_ry: Vec<T>,
    eq_rx: T,
    eq_ry: T,
    witness: T,
    // identity: VerifierComputedOpening<T>,
}

impl<T: CanonicalSerialize + CanonicalDeserialize + Sync> StructuredPolynomialData<T>
    for SpartanStuff<T>
{
    fn read_write_values(&self) -> Vec<&T> {
        todo!()
        // self.read_cts_rows
        //     .iter()
        //     .chain(self.read_cts_cols.iter())
        //     .collect()
    }
    fn init_final_values(&self) -> Vec<&T> {
        todo!()
    }
    fn init_final_values_mut(&mut self) -> Vec<&mut T> {
        todo!()
    }
    fn read_write_values_mut(&mut self) -> Vec<&mut T> {
        todo!()
        // self.read_cts_rows
        //     .iter_mut()
        //     .chain(self.read_cts_cols.iter_mut())
        //     .collect()
    }
}

pub type SpartanPolynomials<F: JoltField> = SpartanStuff<DensePolynomial<F>>;

pub type SpartanOpenings<F: JoltField> = SpartanStuff<F>;

pub type SpartanCommitments<PCS: CommitmentScheme<ProofTranscript>, ProofTranscript: Transcript> =
    SpartanStuff<PCS::Commitment>;

impl<F: JoltField, T: CanonicalSerialize + CanonicalDeserialize + Default>
    Initializable<T, SpartanPreprocessing<F>> for SpartanStuff<T>
{
}

#[derive(Clone)]
pub struct SpartanPreprocessing<F: JoltField> {
    inst: Instance<F>,
    vars: Assignment<F>,
    inputs: Assignment<F>,
}

impl<F: JoltField> SpartanPreprocessing<F> {
    #[tracing::instrument(skip_all, name = "Spartan::preprocess")]
    pub fn preprocess(circuit_file: &str) -> Self {
        //TODO(Ashish):- if circuit_file exist then construct A,B,C from that otherwise run a test case.
        // let r1cs = parse_circuit(circuit_file);

        // let matrices = r1cs.to_sparse_matrices();

        // SpartanPreprocessing {
        //     inst: Instance::new(matrices),
        //     vars: Assignment::new(r1cs.num_vars),
        //     inputs: Assignment::new(r1cs.num_inputs)
        // }
        todo!()
    }
}

// pub type ReadSpartanMemoryOpenings<F> = [F; 3];
// impl<F: JoltField> ExogenousOpenings<F> for ReadSpartanMemoryOpenings<F> {
//     fn openings(&self) -> Vec<&F> {
//         self.iter().collect()
//     }

//     fn openings_mut(&mut self) -> Vec<&mut F> {
//         self.iter_mut().collect()
//     }

//     fn exogenous_data<T: CanonicalSerialize + CanonicalDeserialize + Sync>(
//         polys_or_commitments: &JoltStuff<T>,
//     ) -> Vec<&T> {
//         vec![
//             &polys_or_commitments.read_write_memory.t_read_rd,
//             &polys_or_commitments.read_write_memory.t_read_rs1,
//             &polys_or_commitments.read_write_memory.t_read_rs2,
//             &polys_or_commitments.read_write_memory.t_read_ram,
//         ]
//     }
// }

impl<F, PCS, ProofTranscript> MemoryCheckingProver<F, PCS, ProofTranscript>
    for SpartanProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type Polynomials = SpartanPolynomials<F>;
    type Openings = SpartanOpenings<F>;

    type Preprocessing = SpartanPreprocessing<F>;

    type Commitments = SpartanCommitments<PCS, ProofTranscript>;
    // type ExogenousOpenings = >; //TODO:- Check if exogenous is required or not.
    type MemoryTuple = (F, F, F); //TODO(Ritwik):- Change this if required.

    fn fingerprint(inputs: &(F, F, F), gamma: &F, tau: &F) -> F {
        let (a, v, t) = *inputs;
        t * gamma.square() + v * *gamma + a - *tau
    }

    #[tracing::instrument(skip_all, name = "SpartanPolynomials::compute_leaves")]
    fn compute_leaves(
        preprocessing: &SpartanPreprocessing<F>,
        polynomials: &Self::Polynomials,
        _: &JoltPolynomials<F>,
        gamma: &F,
        tau: &F,
    ) -> ((Vec<F>, usize), (Vec<F>, usize)) {
        let read_write_batch_size = 12;
        let init_final_batch_size = 8;
        //TODO(Ritwik):-

        //Assuming the sparsity of all the matrices are the same, their read_ts count will be the same for rows and columns.
        let n_reads = polynomials.rows[0].len();
        let read_cts_rows = &polynomials.read_cts_rows;
        let read_cts_cols = &polynomials.read_cts_cols;
        let final_cts_rows = &polynomials.final_cts_rows;
        let final_cts_cols = &polynomials.final_cts_cols;
        let rows = &polynomials.rows;
        let cols = &polynomials.cols;
        let e_rx = &polynomials.e_rx;
        let e_ry = &polynomials.e_ry;
        let eq_rx = &polynomials.eq_rx;
        let eq_ry = &polynomials.eq_ry;

        //Interleaved A_row_reads B_row_reads C_row_reads A_col_reads B_col_reads C_col_reads

        let read_leaves: Vec<F> = (0..3)
            .into_par_iter()
            .flat_map(|i| {
                (0..n_reads).into_par_iter().map(move |j| {
                    Self::fingerprint(&(rows[i][j], e_rx[i][j], read_cts_rows[i][j]), gamma, tau)
                })
            })
            .chain((0..3).into_par_iter().flat_map(|i| {
                (0..n_reads).into_par_iter().map(move |j| {
                    Self::fingerprint(&(cols[i][j], e_ry[i][j], read_cts_cols[i][j]), gamma, tau)
                })
            }))
            .collect();

        let write_leaves: Vec<F> = (0..3)
            .into_par_iter()
            .flat_map(|i| {
                (0..n_reads).into_par_iter().map(move |j| {
                    Self::fingerprint(
                        &(rows[i][j], e_rx[i][j], read_cts_rows[i][j] + F::one()),
                        gamma,
                        tau,
                    )
                })
            })
            .chain((0..3).into_par_iter().flat_map(|i| {
                (0..n_reads).into_par_iter().map(move |j| {
                    Self::fingerprint(
                        &(cols[i][j], e_ry[i][j], read_cts_cols[i][j] + F::one()),
                        gamma,
                        tau,
                    )
                })
            }))
            .collect();

        let init_leaves: Vec<F> = (0..eq_rx.len())
            .into_par_iter()
            .map(|i| {
                Self::fingerprint(
                    &(F::from_u64(i as u64).unwrap(), eq_rx[i], F::zero()),
                    gamma,
                    tau,
                )
            })
            .chain((0..eq_rx.len()).into_par_iter().map(|i| {
                Self::fingerprint(
                    &(F::from_u64(i as u64).unwrap(), eq_ry[i], F::zero()),
                    gamma,
                    tau,
                )
            }))
            .collect();

        let final_leaves: Vec<F> = (0..3)
            .into_par_iter()
            .flat_map(|i| {
                (0..eq_rx.len()).into_par_iter().map(move |j| {
                    Self::fingerprint(
                        &(
                            F::from_u64(j as u64).unwrap(),
                            eq_rx[j],
                            final_cts_rows[i][j],
                        ),
                        gamma,
                        tau,
                    )
                })
            })
            .chain((0..3).into_par_iter().flat_map(|i| {
                (0..eq_ry.len()).into_par_iter().map(move |j| {
                    Self::fingerprint(
                        &(
                            F::from_u64(j as u64).unwrap(),
                            eq_ry[j],
                            final_cts_cols[i][j],
                        ),
                        gamma,
                        tau,
                    )
                })
            }))
            .collect();
        //Length of reads and thus writes in this case should be equal to the length of vals, which is the length of non-zero values in the sparse matrix being opened.

        let (read_write_leaves, init_final_leaves) = Self::interleave(
            &preprocessing,
            &read_leaves,
            &write_leaves,
            &init_leaves,
            &final_leaves,
        );

        (
            (read_write_leaves, read_write_batch_size),
            (init_final_leaves, init_final_batch_size),
        )
    }

    fn interleave<T: Copy + Clone>(
        preprocessing: &SpartanPreprocessing<F>,
        read_values: &Vec<T>,
        write_values: &Vec<T>,
        init_values: &Vec<T>,
        final_values: &Vec<T>,
    ) -> (Vec<T>, Vec<T>) {
        let read_write_values = interleave(read_values, write_values).cloned().collect();

        //eq_rx init, A_rx_final, B_rx_final, C_rx_final, eq_ry init, A_ry_final, B_ry_final, C_ry_final
        let mut init_final_values: Vec<T> = init_values
            .iter()
            .zip(final_values.chunks(3))
            .flat_map(|(init, final_vals)| [*init, final_vals[0], final_vals[1], final_vals[2]])
            .collect();

        (read_write_values, init_final_values)
    }

    fn uninterleave_hashes(
        preprocessing: &SpartanPreprocessing<F>,
        read_write_hashes: Vec<F>,
        init_final_hashes: Vec<F>,
    ) -> MultisetHashes<F> {
        let mut read_hashes = Vec::with_capacity(6);
        let mut write_hashes = Vec::with_capacity(6);
        for i in 0..6 {
            read_hashes.push(read_write_hashes[2 * i]);
            write_hashes.push(read_write_hashes[2 * i + 1]);
        }

        let mut init_hashes = Vec::with_capacity(2);
        let mut final_hashes = Vec::with_capacity(6);
        for i in 0..2 {
            init_hashes.push(init_final_hashes[4 * i]);
            final_hashes.push(init_final_hashes[4 * i + 1]);
            final_hashes.push(init_final_hashes[4 * i + 2]);
            final_hashes.push(init_final_hashes[4 * i + 3]);
        }

        MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes,
            final_hashes,
        }
    }

    fn check_multiset_equality(
        preprocessing: &SpartanPreprocessing<F>,
        multiset_hashes: &MultisetHashes<F>,
    ) {
        let num_memories = multiset_hashes.read_hashes.len();
        assert_eq!(multiset_hashes.final_hashes.len(), 6);
        assert_eq!(multiset_hashes.write_hashes.len(), 6);
        assert_eq!(multiset_hashes.init_hashes.len(), 2);

        (0..3).into_iter().for_each(|i| {
            let read_hash = multiset_hashes.read_hashes[i];
            let write_hash = multiset_hashes.write_hashes[i];
            let init_hash = multiset_hashes.init_hashes[0]; //row_init hash
            let final_hash = multiset_hashes.final_hashes[i];
            assert_eq!(
                init_hash * write_hash,
                final_hash * read_hash,
                "Multiset hashes don't match"
            );
        });
        (0..3).into_iter().for_each(|i| {
            let read_hash = multiset_hashes.read_hashes[3 + i];
            let write_hash = multiset_hashes.write_hashes[3 + i];
            let init_hash = multiset_hashes.init_hashes[1]; //col_init hash
            let final_hash = multiset_hashes.final_hashes[3 + i];
            assert_eq!(
                init_hash * write_hash,
                final_hash * read_hash,
                "Multiset hashes don't match"
            );
        });
    }

    fn protocol_name() -> &'static [u8] {
        b"Spartan Validity Proof"
    }
}

impl<F, PCS, ProofTranscript> MemoryCheckingVerifier<F, PCS, ProofTranscript>
    for SpartanProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    fn compute_verifier_openings(
        openings: &mut Self::Openings,
        prerocessing: &SpartanPreprocessing<F>,
        _: &[F],
        _: &[F],
    ) {
        //TODO(Ritwik):-
        //This will require changes in SpartanOpenings.
    }

    fn read_tuples(
        _: &SpartanPreprocessing<F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        //TODO(Ritwik):-

        todo!()
    }

    fn write_tuples(
        _: &SpartanPreprocessing<F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        //TODO(Ritwik):-

        todo!()
    }

    fn init_tuples(
        _: &SpartanPreprocessing<F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        //TODO(Ritwik):-

        todo!()
    }

    fn final_tuples(
        _: &SpartanPreprocessing<F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        //TODO(Ritwik):-

        todo!()
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SpartanProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub(crate) outer_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) inner_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) outer_sumcheck_claims: (F, F, F),

    pub(crate) memory_checking:
        MemoryCheckingProof<F, PCS, SpartanOpenings<F>, NoExogenousOpenings, ProofTranscript>,
    pub(crate) opening_proof: ReducedOpeningProof<F, PCS, ProofTranscript>,
}

impl<F, PCS, ProofTranscript> SpartanProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "SpartanProof::generate_witness")]
    pub fn generate_witness(preprocessing: &SpartanPreprocessing<F>) {
        let r1cs_instance = &preprocessing.inst.inst;
        let matrices = r1cs_instance.get_matrices();
        SparseMatPolynomial::multi_sparse_to_dense_rep(&matrices);
        //TODO(Ashish):- Return Polynomials and commmitments.
    }

    #[tracing::instrument(skip_all, name = "SpartanProof::prove")]
    pub fn prove<'a>(
        pcs_setup: &PCS::Setup,
        polynomials: &mut SpartanPolynomials<F>,
        commitments: &mut SpartanCommitments<PCS, ProofTranscript>,
        preprocessing: &SpartanPreprocessing<F>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let mut opening_accumulator: ProverOpeningAccumulator<F, ProofTranscript> =
            ProverOpeningAccumulator::new();

        let num_inputs = preprocessing.inputs.assignment.len();
        let num_vars = preprocessing.vars.assignment.len();

        // we currently require the number of |inputs| + 1 to be at most number of vars
        assert!(num_inputs < num_vars);

        //TODO(Ashish):- Commit Witness

        // append input to variables to create a single vector z
        let z = {
            let mut z = preprocessing.vars.assignment.clone();
            z.extend(&vec![F::one()]); // add constant term in z
            z.extend(preprocessing.inputs.assignment.clone());
            z.extend(&vec![F::zero(); num_vars - num_inputs - 1]); // we will pad with zeros
            DensePolynomial::new(z)
        };

        // derive the verifier's challenge tau
        let (num_rounds_x, num_rounds_y) = (
            preprocessing.inst.inst.get_num_cons().log_2(),
            z.len().log_2(),
        );
        let tau = transcript.challenge_vector(num_rounds_x);

        let eq_tau = DensePolynomial::new(EqPolynomial::evals(&tau));

        let (az, bz, cz) = preprocessing.inst.inst.multiply_vec(
            preprocessing.inst.inst.get_num_cons(),
            z.len(),
            &z.Z,
        );
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

        //TODO(Ashish):- Do we need to do reverse?
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
            let (evals_A, evals_B, evals_C) = preprocessing.inst.inst.compute_eval_table_sparse(
                preprocessing.inst.inst.get_num_cons(),
                z.len(),
                eq_tau.evals_ref(),
            );

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

        //TODO(Ritwik):- Add spark sum check;
        //TODO(Ritwik):- Change e_rx, e_ry poly appropriately.
        polynomials.e_rx = core::array::from_fn::<DensePolynomial<F>, 1, _>(|_| {
            DensePolynomial::new(vec![F::zero()])
        })
        .to_vec();
        polynomials.e_ry = core::array::from_fn::<DensePolynomial<F>, 1, _>(|_| {
            DensePolynomial::new(vec![F::zero()])
        })
        .to_vec();

        commitments.e_rx = PCS::batch_commit_polys(&polynomials.e_rx, pcs_setup, BatchType::Big);
        commitments.e_ry = PCS::batch_commit_polys(&polynomials.e_rx, pcs_setup, BatchType::Big);

        let memory_checking = Self::prove_memory_checking(
            pcs_setup,
            preprocessing,
            &polynomials,
            &JoltPolynomials::default(),
            &mut opening_accumulator,
            transcript,
        );

        let opening_proof = opening_accumulator.reduce_and_prove::<PCS>(pcs_setup, transcript);
        SpartanProof {
            outer_sumcheck_proof,
            inner_sumcheck_proof,
            outer_sumcheck_claims: (
                outer_sumcheck_claims[0],
                outer_sumcheck_claims[1],
                outer_sumcheck_claims[2],
            ),
            memory_checking,
            opening_proof,
        }
    }

    pub fn verify(
        pcs_setup: &PCS::Setup,
        preprocessing: &SpartanPreprocessing<F>,
        commitments: &SpartanCommitments<PCS, ProofTranscript>,
        num_vars: usize,
        num_cons: usize,
        input: &[F],
        proof: SpartanProof<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        // input.append_to_transcript(b"input", transcript);
        let mut opening_accumulator: VerifierOpeningAccumulator<F, PCS, ProofTranscript> =
            VerifierOpeningAccumulator::new();

        let n = num_vars;
        // add the commitment to the verifier's transcript
        // self.comm_vars
        //     .append_to_transcript(b"poly_commitment", transcript);

        let (num_rounds_x, num_rounds_y) = (num_cons.log_2(), (2 * num_vars).log_2());

        // derive the verifier's challenge tau
        let tau = transcript.challenge_vector(num_rounds_x);

        // verify the first sum-check instance

        let (claim_outer_final, r_x) = proof
            .outer_sumcheck_proof
            .verify(F::zero(), num_rounds_x, 3, transcript)
            .map_err(|e| e)?;

        let (claim_Az, claim_Bz, claim_Cz) = proof.outer_sumcheck_claims;

        let taus_bound_rx = EqPolynomial::new(tau).evaluate(&r_x);
        let claim_outer_final_expected = taus_bound_rx * (claim_Az * claim_Bz - claim_Cz);
        if claim_outer_final != claim_outer_final_expected {
            return Err(ProofVerifyError::SpartanError(
                "Invalid Outer Sumcheck Claim".to_string(),
            ));
        }

        transcript.append_scalars(
            [
                proof.outer_sumcheck_claims.0,
                proof.outer_sumcheck_claims.1,
                proof.outer_sumcheck_claims.2,
            ]
            .as_slice(),
        );

        let r_inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let claim_inner_joint = proof.outer_sumcheck_claims.0
            + r_inner_sumcheck_RLC * proof.outer_sumcheck_claims.1
            + r_inner_sumcheck_RLC * r_inner_sumcheck_RLC * proof.outer_sumcheck_claims.2;

        let (claim_inner_final, inner_sumcheck_r) = proof
            .inner_sumcheck_proof
            .verify(claim_inner_joint, num_rounds_y, 2, transcript)
            .map_err(|e| e)?;

        //TODO(Ritwik):- Add Spark sum check verification.

        Self::verify_memory_checking(
            preprocessing,
            pcs_setup,
            proof.memory_checking,
            &commitments,
            &JoltCommitments::<PCS, ProofTranscript>::default(),
            &mut opening_accumulator,
            transcript,
        )?;

        // Batch-verify all openings
        opening_accumulator.reduce_and_verify(pcs_setup, &proof.opening_proof, transcript)?;
        Ok(())
    }

    /// Computes the shape of all commitments.
    pub fn commitment_shapes(max_trace_length: usize) -> Vec<CommitShape> {
        //TODO(Ashish):- Check if this need to be changed.
        let max_trace_length = max_trace_length.next_power_of_two();

        vec![CommitShape::new(max_trace_length, BatchType::Big)]
    }

    fn protocol_name() -> &'static [u8] {
        b"Spartan Proof"
    }
}
