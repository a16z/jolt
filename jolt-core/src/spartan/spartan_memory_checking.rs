use super::sparse_mlpoly::SparseMatPolynomial;
use super::{Assignment, Instance};
use crate::field::{JoltField, OptimizedMul};
use crate::jolt::vm::{JoltCommitments, JoltPolynomials};
use crate::lasso::memory_checking::{
    Initializable, NoExogenousOpenings, StructuredPolynomialData, VerifierComputedOpening,
};
use crate::poly::commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use crate::poly::opening_proof::{
    ProverOpeningAccumulator, ReducedOpeningProof, VerifierOpeningAccumulator,
};
use crate::spartan::r1csinstance::R1CSInstance;
use crate::spartan::sparse_mlpoly::CircuitConfig;
use crate::spartan::sparse_mlpoly::SparseMatEntry;
use crate::subprotocols::grand_product::BatchedGrandProduct;
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::utils::math::Math;
use crate::utils::transcript::{AppendToTranscript, Transcript};
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
use itertools::{chain, interleave, Itertools};
use rayon::prelude::*;
use std::array;
use std::fs::File;

#[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct SpartanStuff<T: CanonicalSerialize + CanonicalDeserialize + Sync> {
    witness: T,
    read_cts_rows: Vec<T>,
    read_cts_cols: Vec<T>,
    final_cts_rows: Vec<T>,
    final_cts_cols: Vec<T>,
    rows: Vec<T>,
    cols: Vec<T>,
    vals: Vec<T>,
    e_rx: Vec<T>,
    e_ry: Vec<T>,
    eq_rx: VerifierComputedOpening<T>,
    eq_ry: VerifierComputedOpening<T>,
    identity: VerifierComputedOpening<T>,
}

impl<T: CanonicalSerialize + CanonicalDeserialize + Sync> StructuredPolynomialData<T>
    for SpartanStuff<T>
{
    fn read_write_values(&self) -> Vec<&T> {
        self.read_cts_rows
            .iter()
            .chain(self.read_cts_cols.iter())
            .chain(self.rows.iter())
            .chain(self.cols.iter())
            .chain(self.e_rx.iter())
            .chain(self.e_ry.iter())
            .collect()
    }

    fn init_final_values(&self) -> Vec<&T> {
        self.final_cts_rows
            .iter()
            .chain(self.final_cts_cols.iter())
            .collect()
    }

    fn init_final_values_mut(&mut self) -> Vec<&mut T> {
        self.final_cts_rows
            .iter_mut()
            .chain(self.final_cts_cols.iter_mut())
            .collect()
    }

    fn read_write_values_mut(&mut self) -> Vec<&mut T> {
        self.read_cts_rows
            .iter_mut()
            .chain(self.read_cts_cols.iter_mut())
            .chain(self.rows.iter_mut())
            .chain(self.cols.iter_mut())
            .chain(self.e_rx.iter_mut())
            .chain(self.e_ry.iter_mut())
            .collect()
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
    rx_ry: Option<Vec<F>>,
}

impl<F: JoltField> SpartanPreprocessing<F> {
    #[tracing::instrument(skip_all, name = "Spartan::preprocess")]
    // pub fn preprocess(circuit_file: &str) -> Self {
    pub fn preprocess() -> Self {
        // let file = File::open(circuit_file);

        // if file.is_err() {
        //     let reader = std::io::BufReader::new(file.unwrap());
        //     let config: CircuitConfig = serde_json::from_reader(reader).unwrap();

        //     let mut sparse_entries = vec![Vec::new(); 3];

        //     // Reading JSON file
        //     for (row, constraint) in config.constraints.iter().enumerate() {
        //         for (j, dict) in constraint.iter().enumerate() {
        //             for (key, value) in dict {
        //                 let col = key.parse::<usize>().unwrap();
        //                 let val = value.as_bytes();

        //                 sparse_entries[j].push(SparseMatEntry::new(
        //                     row,
        //                     col as usize,
        //                     F::from_bytes(val),
        //                 ));
        //             }
        //         }
        //     }

        //     let num_cons = sparse_entries[0].len();
        //     let num_vars = 10; //TODO(Ashish):- fix num_vars.
        //     let num_inputs = 0; //TODO(Ashish):- fix num_inputs.
        //     let num_poly_vars_x = num_cons.log_2();
        //     let num_poly_vars_y = (2 * num_vars).log_2();

        //     let poly_A = SparseMatPolynomial::new(
        //         num_poly_vars_x,
        //         num_poly_vars_y,
        //         sparse_entries[0].clone(),
        //     );
        //     let poly_B = SparseMatPolynomial::new(
        //         num_poly_vars_x,
        //         num_poly_vars_y,
        //         sparse_entries[1].clone(),
        //     );
        //     let poly_C = SparseMatPolynomial::new(
        //         num_poly_vars_x,
        //         num_poly_vars_y,
        //         sparse_entries[2].clone(),
        //     );
        //     let inst = R1CSInstance::new(num_cons, num_vars, num_inputs, poly_A, poly_B, poly_C);

        //     //TODO():- Implement
        //     SpartanPreprocessing {
        //         inst: Instance { inst },
        //         vars: todo!(),
        //         inputs: todo!(),
        //     }
        // } else {
        let num_vars = (2_usize).pow(10 as u32);
        let num_cons = num_vars;
        let num_inputs = 10;
        let (inst, vars, inputs) =
            Instance::<F>::produce_synthetic_r1cs(num_cons, num_vars, num_inputs);
        SpartanPreprocessing {
            inst,
            vars,
            inputs,
            rx_ry: None,
        }
        // }
    }
}

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
    type MemoryTuple = (F, F, F);

    fn fingerprint(inputs: &(F, F, F), gamma: &F, tau: &F) -> F {
        let (a, v, t) = *inputs;
        t * gamma.square() + v * *gamma + a - *tau
    }

    #[tracing::instrument(skip_all, name = "SpartanPolynomials::compute_leaves")]
    fn compute_leaves(
        _: &SpartanPreprocessing<F>,
        polynomials: &Self::Polynomials,
        _: &JoltPolynomials<F>,
        gamma: &F,
        tau: &F,
    ) -> (
        <Self::ReadWriteGrandProduct as BatchedGrandProduct<F, PCS, ProofTranscript>>::Leaves,
        <Self::InitFinalGrandProduct as BatchedGrandProduct<F, PCS, ProofTranscript>>::Leaves,
    ) {
        let read_write_batch_size = 12;
        let init_final_batch_size = 8;

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
        let eq_rx = match &polynomials.eq_rx {
            Some(eq) => eq,
            None => panic!(),
        };
        let eq_ry = match &polynomials.eq_ry {
            Some(eq) => eq,
            None => panic!(),
        };

        (0..read_cts_rows.len()).for_each(|idx| {
            assert_eq!(read_cts_rows[idx].len(), n_reads);
            assert_eq!(read_cts_cols[idx].len(), n_reads);
            assert_eq!(rows[idx].len(), n_reads);
            assert_eq!(cols[idx].len(), n_reads);
            assert_eq!(e_rx[idx].len(), n_reads);
            assert_eq!(e_ry[idx].len(), n_reads);
        });

        //Interleaved A_row_reads B_row_reads C_row_reads A_col_reads B_col_reads C_col_reads
        let read_row: Vec<F> = (0..3)
            .into_par_iter()
            .flat_map(|i| {
                (0..n_reads).into_par_iter().map(move |j| {
                    Self::fingerprint(&(rows[i][j], e_rx[i][j], read_cts_rows[i][j]), gamma, tau)
                })
            })
            .collect();

        let read_col: Vec<F> = (0..3)
            .into_par_iter()
            .flat_map(|i| {
                (0..n_reads).into_par_iter().map(move |j| {
                    Self::fingerprint(&(cols[i][j], e_ry[i][j], read_cts_cols[i][j]), gamma, tau)
                })
            })
            .collect();

        //Write tuples are just read tuples with the timestamps incremented by one.
        let write_row: Vec<F> = read_row
            .par_iter()
            .map(|leaf| *leaf + gamma.square())
            .collect();

        let write_col: Vec<F> = read_col
            .par_iter()
            .map(|leaf| *leaf + gamma.square())
            .collect();
        let init_row: Vec<F> = (0..eq_rx.len())
            .into_par_iter()
            .map(|i| {
                Self::fingerprint(
                    &(F::from_u64(i as u64).unwrap(), eq_rx[i], F::zero()),
                    gamma,
                    tau,
                )
            })
            .collect();

        let init_col: Vec<F> = (0..eq_rx.len())
            .into_par_iter()
            .map(|i| {
                Self::fingerprint(
                    &(F::from_u64(i as u64).unwrap(), eq_ry[i], F::zero()),
                    gamma,
                    tau,
                )
            })
            .collect();

        let final_row: Vec<F> = (0..3)
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
            .collect();
        let final_col: Vec<F> = (0..3)
            .into_par_iter()
            .flat_map(|i| {
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
            })
            .collect();
        //Length of reads and thus writes in this case should be equal to the length of vals, which is the length of non-zero values in the sparse matrix being opened.
        let read_write_leaves = vec![read_row, write_row, read_col, write_col].concat();
        let init_final_leaves = vec![init_row, final_row, init_col, final_col].concat();

        (
            (read_write_leaves, read_write_batch_size),
            (init_final_leaves, init_final_batch_size),
        )
    }

    fn interleave<T: Copy + Clone>(
        _: &SpartanPreprocessing<F>,
        read_values: &Vec<T>,
        write_values: &Vec<T>,
        init_values: &Vec<T>,
        final_values: &Vec<T>,
    ) -> (Vec<T>, Vec<T>) {
        let read_write_values = interleave(read_values, write_values).cloned().collect();

        //eq_rx init, A_rx_final, B_rx_final, C_rx_final, eq_ry init, A_ry_final, B_ry_final, C_ry_final
        let init_final_values: Vec<T> = init_values
            .iter()
            .zip(final_values.chunks(3))
            .flat_map(|(init, final_vals)| [*init, final_vals[0], final_vals[1], final_vals[2]])
            .collect();

        (read_write_values, init_final_values)
    }

    fn uninterleave_hashes(
        _: &SpartanPreprocessing<F>,
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

    fn check_multiset_equality(_: &SpartanPreprocessing<F>, multiset_hashes: &MultisetHashes<F>) {
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
        preprocessing: &SpartanPreprocessing<F>,
        _: &[F],
        r_init_final: &[F],
    ) {
        let binding = preprocessing.rx_ry.clone().unwrap();
        let (rx, ry) = binding.split_at(binding.len() / 2);

        let eq_rx = EqPolynomial::new(rx.to_vec());
        let eq_ry = EqPolynomial::new(ry.to_vec());

        openings.eq_rx = Some(eq_rx.evaluate(r_init_final));
        openings.eq_ry = Some(eq_ry.evaluate(r_init_final));
        openings.identity =
            Some(IdentityPolynomial::new(r_init_final.len()).evaluate(r_init_final));
    }

    fn read_tuples(
        _: &SpartanPreprocessing<F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let read_cts_rows_opening = &openings.read_cts_rows;
        let read_cts_cols_opening = &openings.read_cts_cols;
        let rows_opening = &openings.rows;
        let cols_opening = &openings.cols;
        let e_rx_opening = &openings.e_rx;
        let e_ry_opening = &openings.e_ry;

        let mut read_tuples = Vec::new();

        for i in 0..3 {
            read_tuples.push((rows_opening[i], e_rx_opening[i], read_cts_rows_opening[i]))
        }
        for i in 0..3 {
            read_tuples.push((cols_opening[i], e_ry_opening[i], read_cts_cols_opening[i]))
        }

        read_tuples
    }

    fn write_tuples(
        _: &SpartanPreprocessing<F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let read_cts_rows_opening = &openings.read_cts_rows;
        let read_cts_cols_opening = &openings.read_cts_cols;
        let rows_opening = &openings.rows;
        let cols_opening = &openings.cols;
        let e_rx_opening = &openings.e_rx;
        let e_ry_opening = &openings.e_ry;

        let mut write_tuples = Vec::new();

        for i in 0..3 {
            write_tuples.push((
                rows_opening[i],
                e_rx_opening[i],
                read_cts_rows_opening[i] + F::one(),
            ))
        }
        for i in 0..3 {
            write_tuples.push((
                cols_opening[i],
                e_ry_opening[i],
                read_cts_cols_opening[i] + F::one(),
            ))
        }

        write_tuples
    }

    fn init_tuples(
        _: &SpartanPreprocessing<F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![
            (
                openings
                    .identity
                    .expect("Expected identity polynomial evaluation"),
                openings.eq_rx.expect("Expected eq polynomial evaluation"),
                F::zero(),
            ),
            (
                openings
                    .identity
                    .expect("Expected identity polynomial evaluation"),
                openings.eq_ry.expect("Expected eq polynomial evaluation"),
                F::zero(),
            ),
        ]
    }

    fn final_tuples(
        _: &SpartanPreprocessing<F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let mut final_tuples = Vec::new();

        for i in 0..3 {
            final_tuples.push((
                openings
                    .identity
                    .expect("Expected identity polynomial evaluation"),
                openings.e_rx[i],
                openings.final_cts_rows[i],
            ))
        }
        for i in 0..3 {
            final_tuples.push((
                openings
                    .identity
                    .expect("Expected identity polynomial evaluation"),
                openings.e_ry[i],
                openings.final_cts_cols[i],
            ))
        }
        final_tuples
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
    pub(crate) spark_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) outer_sumcheck_claims: (F, F, F),
    pub(crate) inner_sumcheck_claims: (F, F, F, F),
    pub(crate) spark_sumcheck_claims: [F; 9],
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
    pub fn generate_witness(
        preprocessing: &SpartanPreprocessing<F>,
        pcs_setup: &PCS::Setup,
    ) -> (
        SpartanPolynomials<F>,
        SpartanCommitments<PCS, ProofTranscript>,
    ) {
        let r1cs_instance = &preprocessing.inst.inst;
        let matrices = r1cs_instance.get_matrices();
        let (read_cts_rows, read_cts_cols, final_cts_rows, final_cts_cols, rows, cols, vals) =
            SparseMatPolynomial::multi_sparse_to_dense_rep(&matrices);
        let polys = SpartanPolynomials {
            witness: DensePolynomial::default(),
            read_cts_rows,
            read_cts_cols,
            final_cts_rows,
            final_cts_cols,
            vals,
            rows,
            cols,
            e_rx: std::iter::repeat_with(|| DensePolynomial::default())
                .take(3)
                .collect(),
            e_ry: std::iter::repeat_with(|| DensePolynomial::default())
                .take(3)
                .collect(),
            eq_rx: None,
            eq_ry: None,
            identity: None,
        };
        let commits = SpartanCommitments::<PCS, ProofTranscript> {
            witness: PCS::Commitment::default(),
            read_cts_rows: PCS::batch_commit_polys(&polys.read_cts_rows, pcs_setup, BatchType::Big),
            read_cts_cols: PCS::batch_commit_polys(&polys.read_cts_cols, pcs_setup, BatchType::Big),
            final_cts_rows: PCS::batch_commit_polys(
                &polys.final_cts_rows,
                pcs_setup,
                BatchType::Big,
            ),
            final_cts_cols: PCS::batch_commit_polys(
                &polys.final_cts_cols,
                pcs_setup,
                BatchType::Big,
            ),
            vals: PCS::batch_commit_polys(&polys.vals, pcs_setup, BatchType::Big),
            rows: PCS::batch_commit_polys(&polys.rows, pcs_setup, BatchType::Big),
            cols: PCS::batch_commit_polys(&polys.cols, pcs_setup, BatchType::Big),
            e_rx: std::iter::repeat_with(|| PCS::Commitment::default())
                .take(3)
                .collect(),
            e_ry: std::iter::repeat_with(|| PCS::Commitment::default())
                .take(3)
                .collect(),
            eq_rx: None,
            eq_ry: None,
            identity: None,
        };
        (polys, commits)
    }

    #[tracing::instrument(skip_all, name = "SpartanProof::prove")]
    pub fn prove<'a>(
        pcs_setup: &PCS::Setup,
        polynomials: &mut SpartanPolynomials<F>,
        commitments: &mut SpartanCommitments<PCS, ProofTranscript>,
        preprocessing: &mut SpartanPreprocessing<F>,
    ) -> Self {
        let mut transcript = ProofTranscript::new(b"Spartan transcript");
        //TODO(Ashish):- Reseed with commitments.

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

        println!("num of x rounds {:?}", num_rounds_x);
        println!("num of y rounds {:?}", num_rounds_y);
        println!("az size is {:?}", az.len());

        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
            SumcheckInstanceProof::prove_arbitrary(
                &F::zero(), // claim is zero
                num_rounds_x,
                &mut [eq_tau.clone(), az, bz, cz].to_vec(),
                comb_func,
                3,
                &mut transcript,
            );

        //TODO(Ashish):- Do we need to do reverse?
        // let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();

        transcript.append_scalars(&outer_sumcheck_claims);

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
        println!("poly_ABC size is {:?}", poly_ABC.len());
        let comb_func = |polys: &[F]| -> F { polys[0] * polys[1] };
        let (inner_sumcheck_proof, inner_sumcheck_r, _claims_inner) =
            SumcheckInstanceProof::prove_arbitrary(
                &claim_inner_joint,
                num_rounds_y,
                &mut [poly_ABC, z].to_vec(),
                comb_func,
                2,
                &mut transcript,
            );

        let (Ar, Br, Cr) = preprocessing
            .inst
            .inst
            .evaluate(&outer_sumcheck_r, &inner_sumcheck_r);

        let eval_vars_at_ry = DensePolynomial::new(preprocessing.vars.assignment.clone())
            .evaluate(&inner_sumcheck_r[1..]);

        transcript.append_scalars(&[Ar, Br, Cr, eval_vars_at_ry]);

        //TODO: Add inner sum check openings to accumulator

        let eq_rx = EqPolynomial::evals(&outer_sumcheck_r);
        let eq_ry = EqPolynomial::evals(&inner_sumcheck_r);

        let num_spark_sumcheck_rounds = outer_sumcheck_r.len();

        preprocessing.rx_ry = Some([outer_sumcheck_r, inner_sumcheck_r].concat());

        polynomials.e_rx = polynomials
            .rows
            .clone()
            .into_par_iter()
            .map(|row| {
                DensePolynomial::new(
                    row.Z
                        .iter()
                        .map(|entry| eq_rx[F::to_u64(entry).unwrap() as usize])
                        .collect_vec(),
                )
            })
            .collect();

        polynomials.e_ry = polynomials
            .cols
            .clone()
            .into_par_iter()
            .map(|col| {
                DensePolynomial::new(
                    col.Z
                        .iter()
                        .map(|entry| eq_ry[F::to_u64(entry).unwrap() as usize])
                        .collect_vec(),
                )
            })
            .collect();

        polynomials.eq_rx = Some(DensePolynomial::new(eq_rx));
        polynomials.eq_ry = Some(DensePolynomial::new(eq_ry));

        commitments.e_rx = PCS::batch_commit_polys(&polynomials.e_rx, pcs_setup, BatchType::Big);
        commitments.e_ry = PCS::batch_commit_polys(&polynomials.e_ry, pcs_setup, BatchType::Big);

        //Appending commitments to the transcript
        for i in 0..3 {
            commitments.e_rx[i].append_to_transcript(&mut transcript);
            commitments.e_ry[i].append_to_transcript(&mut transcript);
        }

        //batching scalar for the spark sum check
        let batching_scalar = transcript.challenge_scalar_powers(3);

        //Flattened vec of polynomials required for spark.
        let mut spark_polys = polynomials
            .e_rx
            .iter()
            .chain(polynomials.e_ry.iter())
            .chain(polynomials.vals.iter())
            .cloned()
            .collect_vec();

        let spark_func = |polys: &[F]| -> F {
            (0..3).fold(F::zero(), |acc, idx| {
                acc + (polys[idx] * polys[idx + 3] * polys[idx + 6])
                    .mul_01_optimized(batching_scalar[idx])
            })
        };

        let (spark_sumcheck_proof, spark_r, spark_claims) = SumcheckInstanceProof::prove_arbitrary(
            &F::zero(), //passsing zero since it is not required.
            num_spark_sumcheck_rounds,
            &mut spark_polys,
            spark_func,
            3,
            &mut transcript,
        );

        transcript.append_scalars(&spark_claims);

        let spark_claim_refs: Vec<&F> = spark_claims.iter().map(|claim| claim).collect();
        let spark_eq = DensePolynomial::new(EqPolynomial::evals(&spark_r));

        opening_accumulator.append(
            &polynomials
                .e_rx
                .iter()
                .chain(polynomials.e_ry.iter())
                .chain(polynomials.vals.iter())
                .collect_vec(),
            spark_eq,
            spark_r,
            &spark_claim_refs,
            &mut transcript,
        );

        let memory_checking = Self::prove_memory_checking(
            pcs_setup,
            preprocessing,
            &polynomials,
            &JoltPolynomials::default(),
            &mut opening_accumulator,
            &mut transcript,
        );

        let opening_proof = opening_accumulator.reduce_and_prove::<PCS>(pcs_setup, &mut transcript);

        SpartanProof {
            outer_sumcheck_proof,
            inner_sumcheck_proof,
            spark_sumcheck_proof,
            outer_sumcheck_claims: (
                outer_sumcheck_claims[0],
                outer_sumcheck_claims[1],
                outer_sumcheck_claims[2],
            ),
            inner_sumcheck_claims: (Ar, Br, Cr, eval_vars_at_ry),
            spark_sumcheck_claims: array::from_fn(|i| spark_claims[i]),
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
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = ProofTranscript::new(b"Spartan transcript");

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
            .verify(F::zero(), num_rounds_x, 3, &mut transcript)
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
            .verify(claim_inner_joint, num_rounds_y, 2, &mut transcript)
            .map_err(|e| e)?;

        let num_spark_sumcheck_rounds = r_x.len();

        assert_eq!(
            preprocessing.rx_ry.clone().unwrap(),
            [r_x, inner_sumcheck_r].concat(),
        );

        let (claim_A, claim_B, claim_C, claim_Z) = proof.inner_sumcheck_claims;

        let claim_inner_final_expected = claim_Z
            * (claim_A + r_inner_sumcheck_RLC * claim_B + r_inner_sumcheck_RLC.square() * claim_C);

        if claim_inner_final != claim_inner_final_expected {
            return Err(ProofVerifyError::SpartanError(
                "Invalid Inner Sumcheck Claim".to_string(),
            ));
        }

        transcript.append_scalars(
            [
                proof.inner_sumcheck_claims.0,
                proof.inner_sumcheck_claims.1,
                proof.inner_sumcheck_claims.2,
                proof.inner_sumcheck_claims.3,
            ]
            .as_slice(),
        );

        for i in 0..3 {
            commitments.e_rx[i].append_to_transcript(&mut transcript);
            commitments.e_ry[i].append_to_transcript(&mut transcript);
        }

        let batching_scalar = transcript.challenge_scalar_powers(3);
        let spark_claim = claim_A + batching_scalar[1] * claim_B + batching_scalar[2] * claim_C;

        let (claim_spark_final, spark_sumcheck_r) = proof
            .spark_sumcheck_proof
            .verify(spark_claim, num_spark_sumcheck_rounds, 3, &mut transcript)
            .map_err(|e| e)?;

        let expected_spark_claim = (0..3).fold(F::zero(), |acc, idx| {
            acc + (proof.spark_sumcheck_claims[idx]
                * proof.spark_sumcheck_claims[idx + 3]
                * proof.spark_sumcheck_claims[idx + 6])
                .mul_01_optimized(batching_scalar[idx])
        });

        if claim_spark_final != expected_spark_claim {
            return Err(ProofVerifyError::SpartanError(
                "Invalid Spark Sumcheck Claim".to_string(),
            ));
        }

        let spark_commitment_refs: Vec<&<PCS as CommitmentScheme<ProofTranscript>>::Commitment> =
            chain![
                commitments.vals.iter().map(|reference| reference),
                commitments.e_rx.iter().map(|reference| reference),
                commitments.e_ry.iter().map(|reference| reference)
            ]
            .collect();

        let spark_claims_refs: Vec<&F> = proof
            .spark_sumcheck_claims
            .iter()
            .map(|reference| reference)
            .collect();
        opening_accumulator.append(
            &spark_commitment_refs,
            spark_sumcheck_r,
            &spark_claims_refs,
            &mut transcript,
        );

        Self::verify_memory_checking(
            preprocessing,
            pcs_setup,
            proof.memory_checking,
            &commitments,
            &JoltCommitments::<PCS, ProofTranscript>::default(),
            &mut opening_accumulator,
            &mut transcript,
        )?;

        // Batch-verify all openings
        opening_accumulator.reduce_and_verify(pcs_setup, &proof.opening_proof, &mut transcript)?;
        Ok(())
    }

    /// Computes the shape of all commitments.
    pub fn commitment_shapes(max_trace_length: usize) -> Vec<CommitShape> {
        let max_trace_length = max_trace_length.next_power_of_two();

        vec![CommitShape::new(max_trace_length, BatchType::Big)]
    }

    fn protocol_name() -> &'static [u8] {
        b"Spartan Proof"
    }
}
#[cfg(test)]
mod tests {
    use crate::{poly::commitment::hyperkzg::HyperKZG, utils::transcript::KeccakTranscript};

    use super::*;
    use ark_bn254::{Bn254, Fr};

    pub type ProofTranscript = KeccakTranscript;
    pub type PCS = HyperKZG<Bn254, ProofTranscript>;
    #[test]
    fn spartan() {
        let mut preprocessing = SpartanPreprocessing::<Fr>::preprocess();
        let commitment_shapes = SpartanProof::<Fr, PCS, ProofTranscript>::commitment_shapes(
            preprocessing.inputs.assignment.len() + preprocessing.vars.assignment.len(),
        );
        let pcs_setup = PCS::setup(&commitment_shapes);
        let (mut spartan_polynomials, mut spartan_commitments) =
            SpartanProof::<Fr, PCS, ProofTranscript>::generate_witness(&preprocessing, &pcs_setup);

        SpartanProof::<Fr, PCS, ProofTranscript>::prove(
            &pcs_setup,
            &mut spartan_polynomials,
            &mut spartan_commitments,
            &mut preprocessing,
        );
    }
}
