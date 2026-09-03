//! Packed HyperKZG commitments of 0/1 columns through the shared-base
//! bit-column kernel: `⌈columns/k⌉` polynomials of `rows·k` entries,
//! `P_g[j·rows + row] = column_{g·k+j}[row]`.

use jolt_crypto::ec::bn254::bit_columns::g1_bit_columns_msm;
use jolt_crypto::Bn254;
use jolt_hyperkzg::{HyperKZGCommitment, HyperKZGProverSetup};
use jolt_limb_bench::pack;
use rayon::prelude::*;

pub fn commit_bit_columns(
    setup: &HyperKZGProverSetup<Bn254>,
    rows: usize,
    k: usize,
    columns: &[Vec<u8>],
) -> Vec<HyperKZGCommitment<Bn254>> {
    let packed: Vec<Vec<u8>> = (0..pack::groups(columns.len(), k))
        .into_par_iter()
        .map(|g| {
            let mut poly = vec![0u8; rows * k];
            for j in 0..k {
                if let Some(column) = columns.get(g * k + j) {
                    poly[j * rows..(j + 1) * rows].copy_from_slice(column);
                }
            }
            poly
        })
        .collect();
    let refs: Vec<&[u8]> = packed.iter().map(Vec::as_slice).collect();
    g1_bit_columns_msm(&setup.g1_powers()[..rows * k], &refs)
        .into_iter()
        .map(HyperKZGCommitment::new)
        .collect()
}
