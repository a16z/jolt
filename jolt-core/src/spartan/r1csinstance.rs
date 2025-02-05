use super::sparse_mlpoly::SparseMatPolynomial;
use crate::{
    field::JoltField,
    poly::dense_mlpoly::DensePolynomial,
    spartan::sparse_mlpoly::SparseMatEntry,
    utils::{math::Math, transcript::Transcript},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]

pub struct R1CSInstance<F: JoltField> {
    num_cons: usize,
    num_vars: usize,
    num_inputs: usize,
    A: SparseMatPolynomial<F>,
    B: SparseMatPolynomial<F>,
    C: SparseMatPolynomial<F>,
}

// #[derive(Serialize, Deserialize)]
// pub struct R1CSCommitmentGens {
//     gens: SparseMatPolyCommitmentGens,
// }

// impl R1CSCommitmentGens {
//     pub fn new(
//         label: &'static [u8],
//         num_cons: usize,
//         num_vars: usize,
//         num_inputs: usize,
//         num_nz_entries: usize,
//     ) -> R1CSCommitmentGens {
//         assert!(num_inputs < num_vars);
//         let num_poly_vars_x = num_cons.log_2();
//         let num_poly_vars_y = (2 * num_vars).log_2();
//         let gens = SparseMatPolyCommitmentGens::new(
//             label,
//             num_poly_vars_x,
//             num_poly_vars_y,
//             num_nz_entries,
//             3,
//         );
//         R1CSCommitmentGens { gens }
//     }
// }

// #[derive(Debug, Serialize, Deserialize)]
// pub struct R1CSCommitment {
//     num_cons: usize,
//     num_vars: usize,
//     num_inputs: usize,
//     comm: SparseMatPolyCommitment,
// }

// impl AppendToTranscript for R1CSCommitment {
//     fn append_to_transcript(&self, _label: &'static [u8], transcript: &mut Transcript) {
//         transcript.append_u64(b"num_cons", self.num_cons as u64);
//         transcript.append_u64(b"num_vars", self.num_vars as u64);
//         transcript.append_u64(b"num_inputs", self.num_inputs as u64);
//         self.comm.append_to_transcript(b"comm", transcript);
//     }
// }

// #[derive(Serialize, Deserialize)]
// pub struct R1CSDecommitment {
//     dense: MultiSparseMatPolynomialAsDense,
// }

// impl R1CSCommitment {
//     pub fn get_num_cons(&self) -> usize {
//         self.num_cons
//     }

//     pub fn get_num_vars(&self) -> usize {
//         self.num_vars
//     }

//     pub fn get_num_inputs(&self) -> usize {
//         self.num_inputs
//     }
// }

impl<F: JoltField> R1CSInstance<F> {
    pub fn new(
        num_cons: usize,
        num_vars: usize,
        num_inputs: usize,
        A: &[(usize, usize, F)],
        B: &[(usize, usize, F)],
        C: &[(usize, usize, F)],
    ) -> R1CSInstance<F> {
        // check that num_cons is a power of 2
        assert_eq!(num_cons.next_power_of_two(), num_cons);

        // check that num_vars is a power of 2
        assert_eq!(num_vars.next_power_of_two(), num_vars);

        // check that number_inputs + 1 <= num_vars
        assert!(num_inputs < num_vars);

        // no errors, so create polynomials
        let num_poly_vars_x = num_cons.log_2();
        let num_poly_vars_y = (2 * num_vars).log_2();

        let mat_A = A
            .iter()
            .map(|(row, col, val)| SparseMatEntry::new(*row, *col, *val))
            .collect::<Vec<SparseMatEntry<F>>>();
        let mat_B = B
            .iter()
            .map(|(row, col, val)| SparseMatEntry::new(*row, *col, *val))
            .collect::<Vec<SparseMatEntry<F>>>();
        let mat_C = C
            .iter()
            .map(|(row, col, val)| SparseMatEntry::new(*row, *col, *val))
            .collect::<Vec<SparseMatEntry<F>>>();

        let poly_A = SparseMatPolynomial::new(num_poly_vars_x, num_poly_vars_y, mat_A);
        let poly_B = SparseMatPolynomial::new(num_poly_vars_x, num_poly_vars_y, mat_B);
        let poly_C = SparseMatPolynomial::new(num_poly_vars_x, num_poly_vars_y, mat_C);

        Self {
            num_cons,
            num_vars,
            num_inputs,
            A: poly_A,
            B: poly_B,
            C: poly_C,
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn get_num_cons(&self) -> usize {
        self.num_cons
    }

    pub fn get_num_inputs(&self) -> usize {
        self.num_inputs
    }
    pub fn get_matrices(&self) -> [&SparseMatPolynomial<F>; 3] {
        [&self.A, &self.B, &self.C]
    }
    // pub fn get_digest(&self) -> Vec<u8> {
    //     let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    //     bincode::serialize_into(&mut encoder, &self).unwrap();
    //     encoder.finish().unwrap()
    // }

    // pub fn produce_synthetic_r1cs(
    //     num_cons: usize,
    //     num_vars: usize,
    //     num_inputs: usize,
    // ) -> (R1CSInstance, Vec<Scalar>, Vec<Scalar>) {
    //     Timer::print(&format!("number_of_constraints {num_cons}"));
    //     Timer::print(&format!("number_of_variables {num_vars}"));
    //     Timer::print(&format!("number_of_inputs {num_inputs}"));

    //     let mut csprng: OsRng = OsRng;

    //     // assert num_cons and num_vars are power of 2
    //     assert_eq!((num_cons.log_2()).pow2(), num_cons);
    //     assert_eq!((num_vars.log_2()).pow2(), num_vars);

    //     // num_inputs + 1 <= num_vars
    //     assert!(num_inputs < num_vars);

    //     // z is organized as [vars,1,io]
    //     let size_z = num_vars + num_inputs + 1;

    //     // produce a random satisfying assignment
    //     let Z = {
    //         let mut Z: Vec<Scalar> = (0..size_z)
    //             .map(|_i| Scalar::random(&mut csprng))
    //             .collect::<Vec<Scalar>>();
    //         Z[num_vars] = Scalar::one(); // set the constant term to 1
    //         Z
    //     };

    //     // three sparse matrices
    //     let mut A: Vec<SparseMatEntry> = Vec::new();
    //     let mut B: Vec<SparseMatEntry> = Vec::new();
    //     let mut C: Vec<SparseMatEntry> = Vec::new();
    //     let one = Scalar::one();
    //     for i in 0..num_cons {
    //         let A_idx = i % size_z;
    //         let B_idx = (i + 2) % size_z;
    //         A.push(SparseMatEntry::new(i, A_idx, one));
    //         B.push(SparseMatEntry::new(i, B_idx, one));
    //         let AB_val = Z[A_idx] * Z[B_idx];

    //         let C_idx = (i + 3) % size_z;
    //         let C_val = Z[C_idx];

    //         if C_val == Scalar::zero() {
    //             C.push(SparseMatEntry::new(i, num_vars, AB_val));
    //         } else {
    //             C.push(SparseMatEntry::new(
    //                 i,
    //                 C_idx,
    //                 AB_val * C_val.invert().unwrap(),
    //             ));
    //         }
    //     }

    //     Timer::print(&format!("number_non-zero_entries_A {}", A.len()));
    //     Timer::print(&format!("number_non-zero_entries_B {}", B.len()));
    //     Timer::print(&format!("number_non-zero_entries_C {}", C.len()));

    //     let num_poly_vars_x = num_cons.log_2();
    //     let num_poly_vars_y = (2 * num_vars).log_2();
    //     let poly_A = SparseMatPolynomial::new(num_poly_vars_x, num_poly_vars_y, A);
    //     let poly_B = SparseMatPolynomial::new(num_poly_vars_x, num_poly_vars_y, B);
    //     let poly_C = SparseMatPolynomial::new(num_poly_vars_x, num_poly_vars_y, C);

    //     let inst = R1CSInstance {
    //         num_cons,
    //         num_vars,
    //         num_inputs,
    //         A: poly_A,
    //         B: poly_B,
    //         C: poly_C,
    //     };

    //     assert!(inst.is_sat(&Z[..num_vars], &Z[num_vars + 1..]));

    //     (inst, Z[..num_vars].to_vec(), Z[num_vars + 1..].to_vec())
    // }

    // pub fn is_sat(&self, vars: &[Scalar], input: &[Scalar]) -> bool {
    //     assert_eq!(vars.len(), self.num_vars);
    //     assert_eq!(input.len(), self.num_inputs);

    //     let z = {
    //         let mut z = vars.to_vec();
    //         z.extend(&vec![Scalar::one()]);
    //         z.extend(input);
    //         z
    //     };

    //     // verify if Az * Bz - Cz = [0...]
    //     let Az = self
    //         .A
    //         .multiply_vec(self.num_cons, self.num_vars + self.num_inputs + 1, &z);
    //     let Bz = self
    //         .B
    //         .multiply_vec(self.num_cons, self.num_vars + self.num_inputs + 1, &z);
    //     let Cz = self
    //         .C
    //         .multiply_vec(self.num_cons, self.num_vars + self.num_inputs + 1, &z);

    //     assert_eq!(Az.len(), self.num_cons);
    //     assert_eq!(Bz.len(), self.num_cons);
    //     assert_eq!(Cz.len(), self.num_cons);
    //     (0..self.num_cons).all(|i| Az[i] * Bz[i] == Cz[i])
    // }

    pub fn multiply_vec(
        &self,
        num_rows: usize,
        num_cols: usize,
        z: &[F],
    ) -> (DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>) {
        assert_eq!(num_rows, self.num_cons);
        assert_eq!(z.len(), num_cols);
        assert!(num_cols > self.num_vars);
        (
            DensePolynomial::new(self.A.multiply_vec(num_rows, num_cols, z)),
            DensePolynomial::new(self.B.multiply_vec(num_rows, num_cols, z)),
            DensePolynomial::new(self.C.multiply_vec(num_rows, num_cols, z)),
        )
    }

    pub fn compute_eval_table_sparse(
        &self,
        num_rows: usize,
        num_cols: usize,
        evals: &[F],
    ) -> (Vec<F>, Vec<F>, Vec<F>) {
        assert_eq!(num_rows, self.num_cons);
        assert!(num_cols > self.num_vars);

        let evals_A = self.A.compute_eval_table_sparse(evals, num_rows, num_cols);
        let evals_B = self.B.compute_eval_table_sparse(evals, num_rows, num_cols);
        let evals_C = self.C.compute_eval_table_sparse(evals, num_rows, num_cols);

        (evals_A, evals_B, evals_C)
    }

    pub fn evaluate(&self, rx: &[F], ry: &[F]) -> (F, F, F) {
        let evals = SparseMatPolynomial::multi_evaluate(&[&self.A, &self.B, &self.C], rx, ry);
        (evals[0], evals[1], evals[2])
    }

    // pub fn commit(&self, gens: &R1CSCommitmentGens) -> (R1CSCommitment, R1CSDecommitment) {
    //     let (comm, dense) =
    //         SparseMatPolynomial::multi_commit(&[&self.A, &self.B, &self.C], &gens.gens);
    //     let r1cs_comm = R1CSCommitment {
    //         num_cons: self.num_cons,
    //         num_vars: self.num_vars,
    //         num_inputs: self.num_inputs,
    //         comm,
    //     };

    //     let r1cs_decomm = R1CSDecommitment { dense };

    //     (r1cs_comm, r1cs_decomm)
    // }
}

// #[derive(Debug, Serialize, Deserialize)]
// pub struct R1CSEvalProof<F, ProofTranscript>
// where
//     F: JoltField,
//     ProofTranscript: Transcript,
// {
//     proof: SparseMatPolyEvalProof<F, ProofTranscript>,
// }

// impl<F: JoltField, ProofTranscript: Transcript> R1CSEvalProof<F, ProofTranscript> {
//     pub fn prove(
//         // decomm: &R1CSDecommitment,
//         rx: &[F], // point at which the polynomial is evaluated
//         ry: &[F],
//         evals: &(F, F, F),
//         // gens: &R1CSCommitmentGens,
//         transcript: &mut ProofTranscript,
//     ) -> R1CSEvalProof<F, ProofTranscript> {
//         let proof = SparseMatPolyEvalProof::prove(
//             // &decomm.dense,
//             rx,
//             ry,
//             &[evals.0, evals.1, evals.2],
//             // &gens.gens,
//             transcript,
//         );
//         R1CSEvalProof { proof }
//     }

// pub fn verify(
//     &self,
//     comm: &R1CSCommitment,
//     rx: &[Scalar], // point at which the R1CS matrix polynomials are evaluated
//     ry: &[Scalar],
//     evals: &(Scalar, Scalar, Scalar),
//     gens: &R1CSCommitmentGens,
//     transcript: &mut Transcript,
// ) -> Result<(), ProofVerifyError> {
//     self.proof.verify(
//         &comm.comm,
//         rx,
//         ry,
//         &[evals.0, evals.1, evals.2],
//         &gens.gens,
//         transcript,
//     )
// }
// }
