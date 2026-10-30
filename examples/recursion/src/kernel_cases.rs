use serde::{Deserialize, Serialize};

/// Inputs and native software results transported to the compiled kernel checks.
#[derive(Serialize, Deserialize)]
pub(crate) struct KernelCase<F> {
    pub rows: Vec<Vec<F>>,
    pub shared: Vec<F>,
    pub weights: Vec<F>,
    pub products: Vec<[F; 4]>,
    pub dots: Vec<F>,
    pub weighted: F,
    pub signed: F,
    pub product_sums: [F; 5],
    pub squares: Vec<F>,
    pub inverses: Vec<Option<F>>,
}
