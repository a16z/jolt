#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
use std::{collections::HashMap, fs::File};

use crate::{
    field::JoltField,
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial},
    utils::math::Math,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Result;

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
#[derive(Debug, Deserialize)]
pub struct CircuitConfig {
    pub constraints: Vec<Vec<HashMap<String, String>>>, // List of lists of HashMaps
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMatPolynomial<F: JoltField> {
    num_vars_x: usize,
    num_vars_y: usize,
    pub M: Vec<SparseMatEntry<F>>,
}

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

    pub fn multi_sparse_to_dense_rep(
        sparse_polys: &[&SparseMatPolynomial<F>],
    ) -> (
        Vec<DensePolynomial<F>>,
        Vec<DensePolynomial<F>>,
        Vec<DensePolynomial<F>>,
        Vec<DensePolynomial<F>>,
        Vec<DensePolynomial<F>>,
        Vec<DensePolynomial<F>>,
        Vec<DensePolynomial<F>>,
    ) {
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

        let (read_ts_rows, final_ts_rows, rows) = compute_ts::<F>(num_mem_cells, N, ops_row_vec);
        let (read_ts_cols, final_ts_cols, cols) = compute_ts::<F>(num_mem_cells, N, ops_col_vec);
        (
            read_ts_rows,
            read_ts_cols,
            final_ts_rows,
            final_ts_cols,
            rows,
            cols,
            val_vec,
        )
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
        // assert_eq!(z.len(), num_cols);
        self.M.iter().fold(
            vec![F::zero(); z.len()],
            |mut Mz, SparseMatEntry { row, col, val }| {
                Mz[*row] += *val * z[*col];
                Mz
            },
        )
    }

    pub fn compute_eval_table_sparse(&self, rx: &[F], num_rows: usize, num_cols: usize) -> Vec<F> {
        // assert_eq!(rx.len(), num_rows);

        self.M.iter().fold(
            vec![F::zero(); num_cols],
            |mut M_evals, SparseMatEntry { row, col, val }| {
                M_evals[*col] += rx[*row] * val;
                M_evals
            },
        )
    }
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
            debug_assert!(
                addr < num_cells,
                "addr is {}, num_cells {}",
                addr,
                num_cells
            );

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
