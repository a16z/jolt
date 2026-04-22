//! R1CS column identifiers for polynomial materialization.
//!
//! Names the derived polynomials the prover can compute from constraint
//! matrices and the per-cycle witness vector.

use serde::{Deserialize, Serialize};

/// Which column of the R1CS relation `Az ∘ Bz = Cz`.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum R1csColumn {
    Az,
    Bz,
    Cz,
    CombinedRow,
    /// A single witness variable column: `witness[c * V_pad + var_index]`
    /// for each cycle `c`. Used to evaluate individual R1CS input polynomials
    /// at the sumcheck challenge point.
    Variable(usize),
}
