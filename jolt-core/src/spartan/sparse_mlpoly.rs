#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
use crate::{
    field::JoltField,
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial},
    utils::math::Math,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMatEntry<F: JoltField> {
    pub row: usize,
    pub col: usize,
    pub val: F,
}

impl<F: JoltField> SparseMatEntry<F> {
    pub fn new(row: usize, col: usize, val: F) -> Self {
        SparseMatEntry { row, col, val }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMatPolynomial<F: JoltField> {
    num_vars_x: usize,
    num_vars_y: usize,
    pub M: Vec<SparseMatEntry<F>>,
}

// #[derive(Serialize, Deserialize)]
// pub struct MultiSparseMatPolynomialAsDense {
//     batch_size: usize,
//     val: Vec<DensePolynomial>,
//     row: AddrTimestamps,
//     col: AddrTimestamps,
//     comb_ops: DensePolynomial,
//     comb_mem: DensePolynomial,
// }

// #[derive(Serialize, Deserialize)]
// pub struct SparseMatPolyCommitmentGens {
//     gens_ops: PolyCommitmentGens,
//     gens_mem: PolyCommitmentGens,
//     gens_derefs: PolyCommitmentGens,
// }

// impl SparseMatPolyCommitmentGens {
//     pub fn new(
//         label: &'static [u8],
//         num_vars_x: usize,
//         num_vars_y: usize,
//         num_nz_entries: usize,
//         batch_size: usize,
//     ) -> SparseMatPolyCommitmentGens {
//         let num_vars_ops = num_nz_entries.next_power_of_two().log_2()
//             + (batch_size * 5).next_power_of_two().log_2();
//         let num_vars_mem = if num_vars_x > num_vars_y {
//             num_vars_x
//         } else {
//             num_vars_y
//         } + 1;
//         let num_vars_derefs = num_nz_entries.next_power_of_two().log_2()
//             + (batch_size * 2).next_power_of_two().log_2();

//         let gens_ops = PolyCommitmentGens::new(num_vars_ops, label);
//         let gens_mem = PolyCommitmentGens::new(num_vars_mem, label);
//         let gens_derefs = PolyCommitmentGens::new(num_vars_derefs, label);
//         SparseMatPolyCommitmentGens {
//             gens_ops,
//             gens_mem,
//             gens_derefs,
//         }
//     }
// }

// #[derive(Debug, Serialize, Deserialize)]
// pub struct SparseMatPolyCommitment {
//     batch_size: usize,
//     num_ops: usize,
//     num_mem_cells: usize,
//     comm_comb_ops: PolyCommitment,
//     comm_comb_mem: PolyCommitment,
// }

// impl AppendToTranscript for SparseMatPolyCommitment {
//     fn append_to_transcript(&self, _label: &'static [u8], transcript: &mut Transcript) {
//         transcript.append_u64(b"batch_size", self.batch_size as u64);
//         transcript.append_u64(b"num_ops", self.num_ops as u64);
//         transcript.append_u64(b"num_mem_cells", self.num_mem_cells as u64);
//         self.comm_comb_ops
//             .append_to_transcript(b"comm_comb_ops", transcript);
//         self.comm_comb_mem
//             .append_to_transcript(b"comm_comb_mem", transcript);
//     }
// }

impl<F: JoltField> SparseMatPolynomial<F> {
    pub fn new(num_vars_x: usize, num_vars_y: usize, M: Vec<SparseMatEntry<F>>) -> Self {
        Self {
            num_vars_x,
            num_vars_y,
            M,
        }
    }

    pub fn get_num_nz_entries(&self) -> usize {
        self.M.len().next_power_of_two()
    }

    fn sparse_to_dense_vecs(&self, N: usize) -> (Vec<usize>, Vec<usize>, Vec<F>) {
        assert!(N >= self.get_num_nz_entries());
        let mut ops_row: Vec<usize> = vec![0; N];
        let mut ops_col: Vec<usize> = vec![0; N];
        let mut val: Vec<F> = vec![F::zero(); N];

        for i in 0..self.M.len() {
            ops_row[i] = self.M[i].row;
            ops_col[i] = self.M[i].col;
            val[i] = self.M[i].val;
        }
        (ops_row, ops_col, val)
    }

    pub fn multi_sparse_to_dense_rep(sparse_polys: &[&SparseMatPolynomial<F>]) {
        assert!(!sparse_polys.is_empty());
        for i in 1..sparse_polys.len() {
            assert_eq!(sparse_polys[i].num_vars_x, sparse_polys[0].num_vars_x);
            assert_eq!(sparse_polys[i].num_vars_y, sparse_polys[0].num_vars_y);
        }

        let N = sparse_polys
            .iter()
            .map(|sparse_poly| sparse_poly.get_num_nz_entries())
            .max()
            .unwrap();

        let mut ops_row_vec: Vec<Vec<usize>> = Vec::new();
        let mut ops_col_vec: Vec<Vec<usize>> = Vec::new();
        let mut val_vec: Vec<DensePolynomial<F>> = Vec::new();
        for poly in sparse_polys {
            let (ops_row, ops_col, val) = poly.sparse_to_dense_vecs(N);
            ops_row_vec.push(ops_row);
            ops_col_vec.push(ops_col);
            val_vec.push(DensePolynomial::new(val));
        }

        let any_poly = &sparse_polys[0];

        let num_mem_cells = if any_poly.num_vars_x > any_poly.num_vars_y {
            any_poly.num_vars_x.pow2()
        } else {
            any_poly.num_vars_y.pow2()
        };

        compute_ts::<F>(num_mem_cells, N, ops_row_vec);
        compute_ts::<F>(num_mem_cells, N, ops_col_vec);

        // combine polynomials into a single polynomial for commitment purposes
        // let comb_ops = DensePolynomial::merge(
        //     row.ops_addr
        //         .iter()
        //         .chain(row.read_ts.iter())
        //         .chain(col.ops_addr.iter())
        //         .chain(col.read_ts.iter())
        //         .chain(val_vec.iter()),
        // );
        // let mut comb_mem = row.audit_ts.clone();
        // comb_mem.extend(&col.audit_ts);

        // MultiSparseMatPolynomialAsDense {
        //     batch_size: sparse_polys.len(),
        //     row,
        //     col,
        //     val: val_vec,
        //     comb_ops,
        //     comb_mem,
        // }
    }

    fn evaluate_with_tables(&self, eval_table_rx: &[F], eval_table_ry: &[F]) -> F {
        assert_eq!(self.num_vars_x.pow2(), eval_table_rx.len());
        assert_eq!(self.num_vars_y.pow2(), eval_table_ry.len());

        self.M
            .iter()
            .map(|SparseMatEntry { row, col, val }| eval_table_rx[*row] * eval_table_ry[*col] * val)
            .sum()
    }

    pub fn multi_evaluate(polys: &[&SparseMatPolynomial<F>], rx: &[F], ry: &[F]) -> Vec<F> {
        let eval_table_rx = EqPolynomial::evals(rx);
        let eval_table_ry = EqPolynomial::evals(ry);

        polys
            .iter()
            .map(|poly| poly.evaluate_with_tables(&eval_table_rx, &eval_table_ry))
            .collect::<Vec<F>>()
    }

    pub fn multiply_vec(&self, num_rows: usize, num_cols: usize, z: &[F]) -> Vec<F> {
        assert_eq!(z.len(), num_cols);

        self.M.iter().fold(
            vec![F::zero(); num_rows],
            |mut Mz, SparseMatEntry { row, col, val }| {
                Mz[*row] += *val * z[*col];
                Mz
            },
        )
    }

    pub fn compute_eval_table_sparse(&self, rx: &[F], num_rows: usize, num_cols: usize) -> Vec<F> {
        assert_eq!(rx.len(), num_rows);

        self.M.iter().fold(
            vec![F::zero(); num_cols],
            |mut M_evals, SparseMatEntry { row, col, val }| {
                M_evals[*col] += rx[*row] * val;
                M_evals
            },
        )
    }

    // pub fn multi_commit(
    //     sparse_polys: &[&SparseMatPolynomial],
    //     gens: &SparseMatPolyCommitmentGens,
    // ) -> (SparseMatPolyCommitment, MultiSparseMatPolynomialAsDense) {
    //     let batch_size = sparse_polys.len();
    //     let dense = SparseMatPolynomial::multi_sparse_to_dense_rep(sparse_polys);

    //     let (comm_comb_ops, _blinds_comb_ops) = dense.comb_ops.commit(&gens.gens_ops, None);
    //     let (comm_comb_mem, _blinds_comb_mem) = dense.comb_mem.commit(&gens.gens_mem, None);

    //     (
    //         SparseMatPolyCommitment {
    //             batch_size,
    //             num_mem_cells: dense.row.audit_ts.len(),
    //             num_ops: dense.row.read_ts[0].len(),
    //             comm_comb_ops,
    //             comm_comb_mem,
    //         },
    //         dense,
    //     )
    // }
}

pub fn compute_ts<F: JoltField>(
    num_cells: usize,
    num_ops: usize,
    ops_addr: Vec<Vec<usize>>,
) -> (
    Vec<DensePolynomial<F>>,
    Vec<DensePolynomial<F>>,
    Vec<DensePolynomial<F>>,
) {
    for item in ops_addr.iter() {
        debug_assert_eq!(item.len(), num_ops);
    }

    let mut final_ts = vec![vec![0usize; num_cells]; ops_addr.len()];
    let mut ops_addr_vec: Vec<DensePolynomial<F>> = Vec::new();
    let mut read_ts_vec: Vec<DensePolynomial<F>> = Vec::new();
    for (idx, ops_addr_inst) in ops_addr.iter().enumerate() {
        let mut read_ts = vec![0usize; num_ops];

        // since read timestamps are trustworthy, we can simply increment the r-ts to obtain a w-ts
        // this is sufficient to ensure that the write-set, consisting of (addr, val, ts) tuples, is a set
        for i in 0..num_ops {
            let addr = ops_addr_inst[i];
            debug_assert!(addr < num_cells);

            read_ts[i] = final_ts[idx][addr];
            final_ts[idx][addr] = read_ts[i] + 1;
        }

        ops_addr_vec.push(DensePolynomial::from_usize(ops_addr_inst));
        read_ts_vec.push(DensePolynomial::from_usize(&read_ts));
    }
    (
        read_ts_vec,
        final_ts
            .iter()
            .map(|fts| DensePolynomial::from_usize(&fts))
            .collect(),
        ops_addr_vec,
    )
}
