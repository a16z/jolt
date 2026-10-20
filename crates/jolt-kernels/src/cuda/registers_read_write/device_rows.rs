use jolt_field::{Fr, Ring};

use crate::cuda::common::read_write_matrix::CoeffTables;

pub(crate) fn register_coeff_tables(gamma: Fr) -> CoeffTables {
    let gamma_sq = gamma * gamma;
    CoeffTables {
        values: [
            vec![Fr::from_u64(0), gamma, gamma_sq, gamma + gamma_sq],
            vec![Fr::from_u64(0), Fr::from_u64(1)],
        ],
    }
}
