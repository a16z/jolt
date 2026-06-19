use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;

/// combined[k] = Σ_i eq(ry_row, i) · flat[i*cols + k]
pub fn combined_row<F: JoltField>(flat: &[F], cols: usize, ry_row: &[F]) -> Vec<F> {
    let R = 1usize << ry_row.len();
    let eq_row: Vec<F> = EqPolynomial::evals(ry_row);

    let mut combined = vec![F::zero(); cols];
    for i in 0..R {
        let w: F = eq_row[i];
        if w.is_zero() {
            continue;
        }
        let base = i * cols;
        for k in 0..cols {
            if base + k < flat.len() {
                combined[k] += w * flat[base + k];
            }
        }
    }
    combined
}

/// eval = Σ_k combined_row[k] · eq(ry_col, k)
pub fn evaluate<F: JoltField>(combined_row: &[F], ry_col: &[F]) -> F {
    let eq_col: Vec<F> = EqPolynomial::evals(ry_col);
    combined_row
        .iter()
        .zip(eq_col.iter())
        .map(|(c, e)| *c * *e)
        .sum()
}

/// combined_blinding = Σ_i eq(ry_row, i) · row_blindings[i]
pub fn combined_blinding<F: JoltField>(row_blindings: &[F], ry_row: &[F]) -> F {
    let eq_row: Vec<F> = EqPolynomial::evals(ry_row);
    row_blindings
        .iter()
        .zip(eq_row.iter())
        .map(|(b, e)| *b * *e)
        .sum()
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyraxOpeningProof<F: JoltField> {
    pub combined_row: Vec<F>,
    pub combined_blinding: F,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::UniformRand;
    use rand::thread_rng;

    #[test]
    fn test_free_functions_consistency() {
        let mut rng = thread_rng();
        let cols: usize = 4;
        let rows: usize = 4;
        let flat: Vec<Fr> = (0..rows * cols).map(|_| Fr::rand(&mut rng)).collect();
        let ry_row: Vec<Fr> = (0..rows.ilog2() as usize)
            .map(|_| Fr::rand(&mut rng))
            .collect();
        let ry_col: Vec<Fr> = (0..cols.ilog2() as usize)
            .map(|_| Fr::rand(&mut rng))
            .collect();

        let cr = combined_row(&flat, cols, &ry_row);
        let eval = evaluate(&cr, &ry_col);

        let blindings: Vec<Fr> = (0..rows).map(|_| Fr::rand(&mut rng)).collect();
        let cb = combined_blinding(&blindings, &ry_row);

        use crate::poly::eq_poly::EqPolynomial;
        let mut point = ry_row.clone();
        point.extend_from_slice(&ry_col);
        let eq_full: Vec<Fr> = EqPolynomial::evals(&point);
        let direct_eval: Fr = flat.iter().zip(eq_full.iter()).map(|(f, e)| *f * *e).sum();
        assert_eq!(eval, direct_eval);

        let eq_row: Vec<Fr> = EqPolynomial::evals(&ry_row);
        let direct_cb: Fr = blindings
            .iter()
            .zip(eq_row.iter())
            .map(|(b, e)| *b * *e)
            .sum();
        assert_eq!(cb, direct_cb);
    }
}
