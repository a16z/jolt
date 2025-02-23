use super::sparse_mlpoly::SparseMatPolynomial;
use super::Instance;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use crate::r1cs::special_polys::SparsePolynomial;
use crate::spartan::r1csinstance::R1CSInstance;
use crate::spartan::sparse_mlpoly::CircuitConfig;
use crate::spartan::sparse_mlpoly::SparseMatEntry;
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::utils::math::Math;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use crate::{
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial},
    utils::errors::ProofVerifyError,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::fs::File;

// #[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
// pub struct SpartanStuff<T: CanonicalSerialize + CanonicalDeserialize + Sync> {
//     pub(crate) witness: T,
//     pub(crate) read_cts_rows: Vec<T>,
//     pub(crate) read_cts_cols: Vec<T>,
//     pub(crate) final_cts_rows: Vec<T>,
//     pub(crate) final_cts_cols: Vec<T>,
//     pub(crate) rows: Vec<T>,
//     pub(crate) cols: Vec<T>,
//     pub(crate) vals: Vec<T>,
//     pub(crate) e_rx: Vec<T>,
//     pub(crate) e_ry: Vec<T>,
//     pub(crate) eq_rx: VerifierComputedOpening<T>,
//     pub(crate) eq_ry: VerifierComputedOpening<T>,
//     pub(crate) identity: VerifierComputedOpening<T>,
// }

// impl<T: CanonicalSerialize + CanonicalDeserialize + Sync> StructuredPolynomialData<T>
//     for SpartanStuff<T>
// {
//     fn read_write_values(&self) -> Vec<&T> {
//         self.read_cts_rows
//             .iter()
//             .chain(self.read_cts_cols.iter())
//             .chain(self.rows.iter())
//             .chain(self.cols.iter())
//             .chain(self.e_rx.iter())
//             .chain(self.e_ry.iter())
//             .collect()
//     }

//     fn init_final_values(&self) -> Vec<&T> {
//         self.final_cts_rows
//             .iter()
//             .chain(self.final_cts_cols.iter())
//             .collect()
//     }

//     fn init_final_values_mut(&mut self) -> Vec<&mut T> {
//         self.final_cts_rows
//             .iter_mut()
//             .chain(self.final_cts_cols.iter_mut())
//             .collect()
//     }

//     fn read_write_values_mut(&mut self) -> Vec<&mut T> {
//         self.read_cts_rows
//             .iter_mut()
//             .chain(self.read_cts_cols.iter_mut())
//             .chain(self.rows.iter_mut())
//             .chain(self.cols.iter_mut())
//             .chain(self.e_rx.iter_mut())
//             .chain(self.e_ry.iter_mut())
//             .collect()
//     }
// }

// pub type SpartanPolynomials<F: JoltField> = SpartanStuff<DensePolynomial<F>>;

// pub type SpartanOpenings<F: JoltField> = SpartanStuff<F>;

// pub type SpartanCommitments<PCS: CommitmentScheme<ProofTranscript>, ProofTranscript: Transcript> =
//     SpartanStuff<PCS::Commitment>;

// impl<F: JoltField, T: CanonicalSerialize + CanonicalDeserialize + Default>
//     Initializable<T, SpartanPreprocessing<F>> for SpartanStuff<T>
// {
//     fn initialize(_: &SpartanPreprocessing<F>) -> Self {
//         Self {
//             witness: T::default(),
//             read_cts_rows: std::iter::repeat_with(|| T::default()).take(3).collect(),
//             read_cts_cols: std::iter::repeat_with(|| T::default()).take(3).collect(),
//             final_cts_rows: std::iter::repeat_with(|| T::default()).take(3).collect(),
//             final_cts_cols: std::iter::repeat_with(|| T::default()).take(3).collect(),
//             rows: std::iter::repeat_with(|| T::default()).take(3).collect(),
//             cols: std::iter::repeat_with(|| T::default()).take(3).collect(),
//             vals: std::iter::repeat_with(|| T::default()).take(3).collect(),
//             e_rx: std::iter::repeat_with(|| T::default()).take(3).collect(),
//             e_ry: std::iter::repeat_with(|| T::default()).take(3).collect(),
//             eq_rx: None,
//             eq_ry: None,
//             identity: None,
//         }
//     }
// }

#[derive(Clone)]
pub struct SpartanPreprocessing<F: JoltField> {
    pub(crate) inst: Instance<F>,
    pub(crate) vars: Vec<F>,
    pub(crate) inputs: Vec<F>,
}

impl<F: JoltField> SpartanPreprocessing<F> {
    #[tracing::instrument(skip_all, name = "Spartan::preprocess")]
    pub fn preprocess(
        constraints_file: Option<&str>,
        witness_file: Option<&str>,
        num_inputs: usize,
    ) -> Self {
        match constraints_file {
            Some(constraints_file) => {
                println!("Preprocess started");
                let file = File::open(witness_file.expect("Path doesn't exist"))
                    .expect("Witness file not found");
                let reader = std::io::BufReader::new(file);
                let witness: Vec<String> = serde_json::from_reader(reader).unwrap();
                let mut z = Vec::new();
                for value in witness {
                    let val: BigUint = value.parse().unwrap();
                    let mut bytes = val.to_bytes_le();
                    bytes.resize(32, 0u8);
                    let val = F::from_bytes(&bytes);
                    z.push(val);
                }
                println!("Witness reading done");

                let num_vars = z.len() - num_inputs - 1;
                assert!(num_inputs < num_vars);

                let append_zeroes = num_vars.next_power_of_two() - num_inputs - 1;
                let size_z = num_vars.next_power_of_two() + append_zeroes + num_inputs + 1;

                let vars = [
                    z[num_inputs + 1..].to_vec(),
                    vec![F::zero(); num_vars.next_power_of_two() - num_vars],
                ]
                .concat();
                let inputs = z[1..num_inputs + 1].to_vec();

                let file = File::open(constraints_file).expect("Constraints file not found");
                let reader = std::io::BufReader::new(file);
                let config: CircuitConfig = serde_json::from_reader(reader).unwrap();

                let mut sparse_entries = vec![Vec::new(); 3];

                // Reading JSON file
                let num_cons = config.constraints.len().next_power_of_two();
                for (row, constraint) in config.constraints.iter().enumerate() {
                    for (j, dict) in constraint.iter().enumerate() {
                        for (key, value) in dict {
                            let col = key.parse::<usize>().unwrap();
                            let val: BigUint = value.parse().unwrap();
                            let mut bytes = val.to_bytes_le();
                            bytes.resize(32, 0u8);
                            let val = F::from_bytes(&bytes);

                            let col = if col > num_inputs {
                                col + append_zeroes
                            } else {
                                col
                            };
                            sparse_entries[j].push(SparseMatEntry::new(row, col as usize, val));
                        }
                    }
                }
                println!("Matrix reading done");

                let num_vars = size_z - num_inputs - 1 - append_zeroes;
                let num_poly_vars_x = num_cons.next_power_of_two().log_2();
                let num_poly_vars_y = size_z.next_power_of_two().log_2();

                let poly_A = SparseMatPolynomial::new(
                    num_poly_vars_x,
                    num_poly_vars_y,
                    sparse_entries[0].clone(),
                );
                let poly_B = SparseMatPolynomial::new(
                    num_poly_vars_x,
                    num_poly_vars_y,
                    sparse_entries[1].clone(),
                );
                let poly_C = SparseMatPolynomial::new(
                    num_poly_vars_x,
                    num_poly_vars_y,
                    sparse_entries[2].clone(),
                );
                let inst =
                    R1CSInstance::new(num_cons, num_vars, num_inputs, poly_A, poly_B, poly_C);
                assert!(inst.is_sat(&inputs, &vars));
                SpartanPreprocessing {
                    inst: Instance { inst },
                    vars,
                    inputs,
                    // rx_ry: None,
                }
            }
            None => {
                let num_vars = (2_usize).pow(5 as u32);
                let num_cons = num_vars;
                let num_inputs = 10;
                let (inst, vars, inputs) =
                    Instance::<F>::produce_synthetic_r1cs(num_cons, num_vars, num_inputs);

                SpartanPreprocessing {
                    inst,
                    vars,
                    inputs,
                    // rx_ry: None,
                }
            }
        }
    }
}

// impl<F, PCS, ProofTranscript> MemoryCheckingProver<F, PCS, ProofTranscript>
//     for SpartanProof<F, PCS, ProofTranscript>
// where
//     F: JoltField,
//     PCS: CommitmentScheme<ProofTranscript, Field = F>,
//     ProofTranscript: Transcript,
// {
//     type Polynomials = SpartanPolynomials<F>;
//     type Openings = SpartanOpenings<F>;

//     type Preprocessing = SpartanPreprocessing<F>;

//     type Commitments = SpartanCommitments<PCS, ProofTranscript>;
//     type MemoryTuple = (F, F, F);

//     fn fingerprint(inputs: &(F, F, F), gamma: &F, tau: &F) -> F {
//         let (a, v, t) = *inputs;
//         t * gamma.square() + v * *gamma + a - *tau
//     }

//     #[tracing::instrument(skip_all, name = "SpartanPolynomials::compute_leaves")]
//     fn compute_leaves(
//         _: &SpartanPreprocessing<F>,
//         polynomials: &Self::Polynomials,
//         _: &JoltPolynomials<F>,
//         gamma: &F,
//         tau: &F,
//     ) -> (
//         <Self::ReadWriteGrandProduct as BatchedGrandProduct<F, PCS, ProofTranscript>>::Leaves,
//         <Self::InitFinalGrandProduct as BatchedGrandProduct<F, PCS, ProofTranscript>>::Leaves,
//     ) {
//         //Assuming the sparsity of all the matrices are the same, their read_ts count will be the same for rows and columns.
//         let n_reads = polynomials.rows[0].len();
//         let read_cts_rows = &polynomials.read_cts_rows;
//         let read_cts_cols = &polynomials.read_cts_cols;
//         let final_cts_rows = &polynomials.final_cts_rows;
//         let final_cts_cols = &polynomials.final_cts_cols;
//         let rows = &polynomials.rows;
//         let cols = &polynomials.cols;
//         let e_rx = &polynomials.e_rx;
//         let e_ry = &polynomials.e_ry;
//         let eq_rx = match &polynomials.eq_rx {
//             Some(eq) => eq,
//             None => panic!(),
//         };
//         let eq_ry = match &polynomials.eq_ry {
//             Some(eq) => eq,
//             None => panic!(),
//         };

//         (0..read_cts_rows.len()).for_each(|idx| {
//             assert_eq!(read_cts_rows[idx].len(), n_reads);
//             assert_eq!(read_cts_cols[idx].len(), n_reads);
//             assert_eq!(rows[idx].len(), n_reads);
//             assert_eq!(cols[idx].len(), n_reads);
//             assert_eq!(e_rx[idx].len(), n_reads);
//             assert_eq!(e_ry[idx].len(), n_reads);
//         });

//         let gamma_squared = gamma.square();

//         //Interleaved A_row_reads B_row_reads C_row_reads A_col_reads B_col_reads C_col_reads
//         let read_write_row: Vec<F> = (0..3)
//             .into_par_iter()
//             .flat_map(|i| {
//                 let read_fingerprints: Vec<F> = (0..n_reads)
//                     .map(|j| {
//                         let a = &rows[i][j];
//                         let v = &e_rx[i][j];
//                         let t = &read_cts_rows[i][j];
//                         mul_0_1_optimized(t, &gamma_squared) + mul_0_1_optimized(v, gamma) + *a
//                             - *tau
//                     })
//                     .collect();
//                 let write_fingerprints = read_fingerprints
//                     .iter()
//                     .map(|read_fingerprint| *read_fingerprint + gamma_squared)
//                     .collect();
//                 [read_fingerprints, write_fingerprints].concat()
//             })
//             .collect();

//         let read_write_col: Vec<F> = (0..3)
//             .into_par_iter()
//             .flat_map(|i| {
//                 let read_fingerprints: Vec<F> = (0..n_reads)
//                     .into_par_iter()
//                     .map(|j| {
//                         let a = &cols[i][j];
//                         let v = &e_ry[i][j];
//                         let t = &read_cts_cols[i][j];
//                         mul_0_1_optimized(t, &gamma_squared) + mul_0_1_optimized(v, gamma) + *a
//                             - *tau
//                     })
//                     .collect();
//                 let write_fingerprints = read_fingerprints
//                     .iter()
//                     .map(|read_fingerprint| *read_fingerprint + gamma_squared)
//                     .collect();
//                 [read_fingerprints, write_fingerprints].concat()
//             })
//             .collect();

//         let init_final_row: Vec<F> = {
//             let init_fingerprints: Vec<F> = (0..eq_rx.len())
//                 .into_par_iter()
//                 .map(|i| {
//                     let a = &F::from_u64(i as u64).unwrap();
//                     let v = &eq_rx[i];
//                     mul_0_1_optimized(v, gamma) + *a - *tau
//                 })
//                 .collect();
//             let final_fingerprits: Vec<F> = (0..3)
//                 .into_par_iter()
//                 .flat_map(|i| {
//                     init_fingerprints
//                         .iter()
//                         .enumerate()
//                         .map(|(j, init_fingerprint)| {
//                             let t = &final_cts_rows[i][j];
//                             *init_fingerprint + mul_0_1_optimized(t, &gamma_squared)
//                         })
//                         .collect::<Vec<F>>()
//                 })
//                 .collect();
//             [init_fingerprints, final_fingerprits].concat()
//         };

//         let init_final_col: Vec<F> = {
//             let init_fingerprints: Vec<F> = (0..eq_ry.len())
//                 .into_par_iter()
//                 .map(|i| {
//                     let a = &F::from_u64(i as u64).unwrap();
//                     let v = &eq_ry[i];
//                     mul_0_1_optimized(v, gamma) + *a - *tau
//                 })
//                 .collect();
//             let final_fingerprits: Vec<F> = (0..3)
//                 .into_par_iter()
//                 .flat_map(|i| {
//                     init_fingerprints
//                         .iter()
//                         .enumerate()
//                         .map(|(j, init_fingerprint)| {
//                             let t = &final_cts_cols[i][j];
//                             *init_fingerprint + mul_0_1_optimized(t, &gamma_squared)
//                         })
//                         .collect::<Vec<F>>()
//                 })
//                 .collect();
//             [init_fingerprints, final_fingerprits].concat()
//         };

//         let read_write_leaves = vec![read_write_row, read_write_col].concat();
//         let init_final_leaves = vec![init_final_row, init_final_col].concat();

//         ((read_write_leaves, 12), (init_final_leaves, 8))
//     }

//     fn interleave<T: Copy + Clone>(
//         _: &SpartanPreprocessing<F>,
//         read_values: &Vec<T>,
//         write_values: &Vec<T>,
//         init_values: &Vec<T>,
//         final_values: &Vec<T>,
//     ) -> (Vec<T>, Vec<T>) {
//         let read_write_values = interleave(read_values, write_values).cloned().collect();

//         let init_final_values: Vec<T> = init_values
//             .iter()
//             .zip(final_values.chunks(3))
//             .flat_map(|(init, final_vals)| [*init, final_vals[0], final_vals[1], final_vals[2]])
//             .collect();

//         (read_write_values, init_final_values)
//     }

//     fn uninterleave_hashes(
//         _: &SpartanPreprocessing<F>,
//         read_write_hashes: Vec<F>,
//         init_final_hashes: Vec<F>,
//     ) -> MultisetHashes<F> {
//         let mut read_hashes = Vec::with_capacity(6);
//         let mut write_hashes = Vec::with_capacity(6);

//         for i in 0..6 {
//             read_hashes.push(read_write_hashes[2 * i]);
//             write_hashes.push(read_write_hashes[2 * i + 1]);
//         }

//         let mut init_hashes = Vec::with_capacity(2);
//         let mut final_hashes = Vec::with_capacity(6);
//         for i in 0..2 {
//             init_hashes.push(init_final_hashes[4 * i]);
//             final_hashes.push(init_final_hashes[4 * i + 1]);
//             final_hashes.push(init_final_hashes[4 * i + 2]);
//             final_hashes.push(init_final_hashes[4 * i + 3]);
//         }

//         MultisetHashes {
//             read_hashes,
//             write_hashes,
//             init_hashes,
//             final_hashes,
//         }
//     }

//     fn check_multiset_equality(_: &SpartanPreprocessing<F>, multiset_hashes: &MultisetHashes<F>) {
//         assert_eq!(multiset_hashes.final_hashes.len(), 6);
//         assert_eq!(multiset_hashes.write_hashes.len(), 6);
//         assert_eq!(multiset_hashes.init_hashes.len(), 2);

//         (0..3).into_iter().for_each(|i| {
//             let read_hash = multiset_hashes.read_hashes[i];
//             let write_hash = multiset_hashes.write_hashes[i];
//             let init_hash = multiset_hashes.init_hashes[0]; //row_init hash
//             let final_hash = multiset_hashes.final_hashes[i];
//             assert_eq!(
//                 init_hash * write_hash,
//                 final_hash * read_hash,
//                 "Multiset hashes don't match"
//             );
//         });
//         (0..3).into_iter().for_each(|i| {
//             let read_hash = multiset_hashes.read_hashes[3 + i];
//             let write_hash = multiset_hashes.write_hashes[3 + i];
//             let init_hash = multiset_hashes.init_hashes[1]; //col_init hash
//             let final_hash = multiset_hashes.final_hashes[3 + i];
//             assert_eq!(
//                 init_hash * write_hash,
//                 final_hash * read_hash,
//                 "Multiset hashes don't match"
//             );
//         });
//     }

//     fn protocol_name() -> &'static [u8] {
//         b"Spartan Validity Proof"
//     }
// }

// impl<F, PCS, ProofTranscript> MemoryCheckingVerifier<F, PCS, ProofTranscript>
//     for SpartanProof<F, PCS, ProofTranscript>
// where
//     F: JoltField,
//     PCS: CommitmentScheme<ProofTranscript, Field = F>,
//     ProofTranscript: Transcript,
// {
//     fn compute_verifier_openings(
//         openings: &mut Self::Openings,
//         preprocessing: &SpartanPreprocessing<F>,
//         _: &[F],
//         r_init_final: &[F],
//     ) {
//         unimplemented!("")
//         // let binding = preprocessing.rx_ry.clone().unwrap();
//         // let (rx, ry) = binding.split_at(binding.len() / 2);

//         // let eq_rx = EqPolynomial::new(rx.to_vec());
//         // let eq_ry = EqPolynomial::new(ry.to_vec());

//         // openings.eq_rx = Some(eq_rx.evaluate(r_init_final));
//         // openings.eq_ry = Some(eq_ry.evaluate(r_init_final));
//         // openings.identity =
//         //     Some(IdentityPolynomial::new(r_init_final.len()).evaluate(r_init_final));
//     }

//     fn read_tuples(
//         _: &SpartanPreprocessing<F>,
//         openings: &Self::Openings,
//         _: &NoExogenousOpenings,
//     ) -> Vec<Self::MemoryTuple> {
//         let read_cts_rows_opening = &openings.read_cts_rows;
//         let read_cts_cols_opening = &openings.read_cts_cols;
//         let rows_opening = &openings.rows;
//         let cols_opening = &openings.cols;
//         let e_rx_opening = &openings.e_rx;
//         let e_ry_opening = &openings.e_ry;

//         let mut read_tuples = Vec::new();

//         for i in 0..3 {
//             read_tuples.push((rows_opening[i], e_rx_opening[i], read_cts_rows_opening[i]))
//         }
//         for i in 0..3 {
//             read_tuples.push((cols_opening[i], e_ry_opening[i], read_cts_cols_opening[i]))
//         }

//         read_tuples
//     }

//     fn write_tuples(
//         _: &SpartanPreprocessing<F>,
//         openings: &Self::Openings,
//         _: &NoExogenousOpenings,
//     ) -> Vec<Self::MemoryTuple> {
//         let read_cts_rows_opening = &openings.read_cts_rows;
//         let read_cts_cols_opening = &openings.read_cts_cols;
//         let rows_opening = &openings.rows;
//         let cols_opening = &openings.cols;
//         let e_rx_opening = &openings.e_rx;
//         let e_ry_opening = &openings.e_ry;

//         let mut write_tuples = Vec::new();

//         for i in 0..3 {
//             write_tuples.push((
//                 rows_opening[i],
//                 e_rx_opening[i],
//                 read_cts_rows_opening[i] + F::one(),
//             ))
//         }
//         for i in 0..3 {
//             write_tuples.push((
//                 cols_opening[i],
//                 e_ry_opening[i],
//                 read_cts_cols_opening[i] + F::one(),
//             ))
//         }

//         write_tuples
//     }

//     fn init_tuples(
//         _: &SpartanPreprocessing<F>,
//         openings: &Self::Openings,
//         _: &NoExogenousOpenings,
//     ) -> Vec<Self::MemoryTuple> {
//         vec![
//             (
//                 openings
//                     .identity
//                     .expect("Expected identity polynomial evaluation"),
//                 openings.eq_rx.expect("Expected eq polynomial evaluation"),
//                 F::zero(),
//             ),
//             (
//                 openings
//                     .identity
//                     .expect("Expected identity polynomial evaluation"),
//                 openings.eq_ry.expect("Expected eq polynomial evaluation"),
//                 F::zero(),
//             ),
//         ]
//     }

//     fn final_tuples(
//         _: &SpartanPreprocessing<F>,
//         openings: &Self::Openings,
//         _: &NoExogenousOpenings,
//     ) -> Vec<Self::MemoryTuple> {
//         let mut final_tuples = Vec::new();

//         for i in 0..3 {
//             final_tuples.push((
//                 openings
//                     .identity
//                     .expect("Expected identity polynomial evaluation"),
//                 openings.eq_rx.expect("Expected eq polynomial evaluation"),
//                 openings.final_cts_rows[i],
//             ))
//         }
//         for i in 0..3 {
//             final_tuples.push((
//                 openings
//                     .identity
//                     .expect("Expected identity polynomial evaluation"),
//                 openings.eq_ry.expect("Expected eq polynomial evaluation"),
//                 openings.final_cts_cols[i],
//             ))
//         }
//         final_tuples
//     }
// }

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
    pub(crate) inner_sumcheck_claims: (F, F, F, F),
    pub(crate) witness_commit: PCS::Commitment,
    pub(crate) pcs_proof: PCS::Proof,
    pub(crate) pi_eval: F,
}

impl<F, PCS, ProofTranscript> SpartanProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    // #[tracing::instrument(skip_all, name = "SpartanProof::generate_witness")]
    // pub fn generate_witness(
    //     preprocessing: &SpartanPreprocessing<F>,
    //     pcs_setup: &PCS::Setup,
    // ) -> (
    //     SpartanPolynomials<F>,
    //     SpartanCommitments<PCS, ProofTranscript>,
    // ) {
    //     let r1cs_instance = &preprocessing.inst.inst;
    //     let matrices = r1cs_instance.get_matrices();
    //     let (read_cts_rows, read_cts_cols, final_cts_rows, final_cts_cols, rows, cols, vals) =
    //         SparseMatPolynomial::multi_sparse_to_dense_rep(&matrices);
    //     let polys = SpartanPolynomials {
    //         witness: DensePolynomial::default(),
    //         read_cts_rows,
    //         read_cts_cols,
    //         final_cts_rows,
    //         final_cts_cols,
    //         vals,
    //         rows,
    //         cols,
    //         e_rx: std::iter::repeat_with(|| DensePolynomial::default())
    //             .take(3)
    //             .collect(),
    //         e_ry: std::iter::repeat_with(|| DensePolynomial::default())
    //             .take(3)
    //             .collect(),
    //         eq_rx: None,
    //         eq_ry: None,
    //         identity: None,
    //     };
    //     let commits = SpartanCommitments::<PCS, ProofTranscript> {
    //         witness: PCS::Commitment::default(),
    //         read_cts_rows: PCS::batch_commit_polys(&polys.read_cts_rows, pcs_setup, BatchType::Big),
    //         read_cts_cols: PCS::batch_commit_polys(&polys.read_cts_cols, pcs_setup, BatchType::Big),
    //         final_cts_rows: PCS::batch_commit_polys(
    //             &polys.final_cts_rows,
    //             pcs_setup,
    //             BatchType::Big,
    //         ),
    //         final_cts_cols: PCS::batch_commit_polys(
    //             &polys.final_cts_cols,
    //             pcs_setup,
    //             BatchType::Big,
    //         ),
    //         vals: PCS::batch_commit_polys(&polys.vals, pcs_setup, BatchType::Big),
    //         rows: PCS::batch_commit_polys(&polys.rows, pcs_setup, BatchType::Big),
    //         cols: PCS::batch_commit_polys(&polys.cols, pcs_setup, BatchType::Big),
    //         e_rx: std::iter::repeat_with(|| PCS::Commitment::default())
    //             .take(3)
    //             .collect(),
    //         e_ry: std::iter::repeat_with(|| PCS::Commitment::default())
    //             .take(3)
    //             .collect(),
    //         eq_rx: None,
    //         eq_ry: None,
    //         identity: None,
    //     };
    //     (polys, commits)
    // }

    #[tracing::instrument(skip_all, name = "SpartanProof::prove")]
    pub fn prove<'a>(pcs_setup: &PCS::Setup, preprocessing: &SpartanPreprocessing<F>) -> Self {
        let protocol_name = Self::protocol_name();
        let mut transcript = ProofTranscript::new(protocol_name);

        let num_inputs = preprocessing.inputs.len();
        let num_vars = preprocessing.vars.len();

        // we currently require the number of |inputs| + 1 to be at most number of vars
        assert!(num_inputs < num_vars);

        let append_zeroes = num_vars - num_inputs - 1;

        // append input to variables to create a single vector z
        let z = {
            let mut z = vec![F::one()]; // add constant term in z
            z.extend(&preprocessing.inputs);
            z.extend(&vec![F::zero(); append_zeroes]); // we will pad with zeros
            z.extend(&preprocessing.vars);
            DensePolynomial::new(z)
        };
        let var_poly = DensePolynomial::new(preprocessing.vars.clone());

        let witness_commit = PCS::commit(&var_poly, pcs_setup);
        witness_commit.append_to_transcript(&mut transcript);

        // derive the verifier's challenge tau
        let (num_rounds_x, num_rounds_y) = (
            (preprocessing.inst.inst.get_num_cons()).log_2(),
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
                &mut transcript,
            );

        transcript.append_scalars(&outer_sumcheck_claims[1..]);

        // claims from the end of sum-check
        // claim_Az is the (scalar) value v_A = \sum_y A(r_x, y) * z(r_x) where r_x is the sumcheck randomness
        let (claim_Az, claim_Bz, claim_Cz): (F, F, F) = (
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
            outer_sumcheck_claims[3],
        );

        let r_inner_sumcheck_RLC: F = transcript.challenge_scalar();

        let r_inner_sumcheck_RLC_square = r_inner_sumcheck_RLC * r_inner_sumcheck_RLC;
        let claim_inner_joint =
            claim_Az + r_inner_sumcheck_RLC * claim_Bz + r_inner_sumcheck_RLC_square * claim_Cz;

        let poly_ABC = {
            let eq_tau = DensePolynomial::new(EqPolynomial::evals(&outer_sumcheck_r));
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
                &mut transcript,
            );

        let (Ar, Br, Cr) = preprocessing
            .inst
            .inst
            .evaluate(&outer_sumcheck_r, &inner_sumcheck_r);

        let eval_vars_at_ry = var_poly.evaluate(&inner_sumcheck_r[1..]);

        transcript.append_scalars(&[Ar, Br, Cr, eval_vars_at_ry]);
        let pcs_proof = PCS::prove(
            pcs_setup,
            &var_poly,
            &inner_sumcheck_r[1..],
            &mut transcript,
        );
        let pi_eval = {
            // constant term
            let mut input_as_sparse_poly_entries = vec![(F::one(), 0)];
            //remaining inputs
            input_as_sparse_poly_entries.extend(
                (0..preprocessing.inputs.len())
                    .map(|i| (preprocessing.inputs[i], i + 1))
                    .collect::<Vec<(F, usize)>>(),
            );
            SparsePolynomial::new(num_vars.log_2(), input_as_sparse_poly_entries)
                .evaluate(&inner_sumcheck_r[1..])
        };

        SpartanProof {
            outer_sumcheck_proof,
            inner_sumcheck_proof,
            outer_sumcheck_claims: (
                outer_sumcheck_claims[1],
                outer_sumcheck_claims[2],
                outer_sumcheck_claims[3],
            ),
            inner_sumcheck_claims: (Ar, Br, Cr, eval_vars_at_ry),
            witness_commit,
            pcs_proof,
            pi_eval,
        }
    }

    pub fn verify(
        pcs_setup: &PCS::Setup,
        preprocessing: &SpartanPreprocessing<F>,
        proof: &SpartanProof<F, PCS, ProofTranscript>,
    ) -> Result<(), ProofVerifyError> {
        let num_vars = preprocessing.vars.len();
        let protocol_name = Self::protocol_name();
        let mut transcript = ProofTranscript::new(protocol_name);

        proof.witness_commit.append_to_transcript(&mut transcript);

        let (num_rounds_x, num_rounds_y) = (
            (preprocessing.inst.inst.get_num_cons()).log_2(),
            (2 * num_vars).log_2(),
        );
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

        let poly_input_eval = {
            // constant term
            let mut input_as_sparse_poly_entries = vec![(F::one(), 0)];
            //remaining inputs
            input_as_sparse_poly_entries.extend(
                (0..preprocessing.inputs.len())
                    .map(|i| (preprocessing.inputs[i], i + 1))
                    .collect::<Vec<(F, usize)>>(),
            );
            SparsePolynomial::new(num_vars.log_2(), input_as_sparse_poly_entries)
                .evaluate(&inner_sumcheck_r[1..])
        };

        let (claim_A, claim_B, claim_C, claim_w) = proof.inner_sumcheck_claims;

        let eval_Z_at_ry =
            (F::one() - inner_sumcheck_r[0]) * poly_input_eval + inner_sumcheck_r[0] * claim_w;

        let claim_inner_final_expected = eval_Z_at_ry
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
        PCS::verify(
            &proof.pcs_proof,
            pcs_setup,
            &mut transcript,
            &inner_sumcheck_r[1..],
            &claim_w,
            &proof.witness_commit,
        )?;
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
pub mod tests {
    use super::*;
    use crate::{
        poly::commitment::hyperkzg::HyperKZG, utils::poseidon_transcript::PoseidonTranscript,
    };
    type Fr = ark_bn254::Fr;
    type Fq = ark_bn254::Fq;
    pub type ProofTranscript = PoseidonTranscript<Fr, Fq>;
    pub type PCS = HyperKZG<ark_bn254::Bn254, ProofTranscript>;

    #[test]
    fn spartan() {
        let mut preprocessing = SpartanPreprocessing::<Fr>::preprocess(None, None, 9);
        let commitment_shapes = SpartanProof::<Fr, PCS, ProofTranscript>::commitment_shapes(
            preprocessing.inputs.len() + preprocessing.vars.len(),
        );
        let pcs_setup = PCS::setup(&commitment_shapes);
        let proof = SpartanProof::<Fr, PCS, ProofTranscript>::prove(&pcs_setup, &mut preprocessing);

        SpartanProof::<Fr, PCS, ProofTranscript>::verify(&pcs_setup, &preprocessing, &proof)
            .unwrap();
    }
}
