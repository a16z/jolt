use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid};

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::serialization::{
    deserialize_bounded_vec, serialize_vec_with_len, serialized_vec_with_len_size,
    MAX_BLINDFOLD_VECTOR_LEN,
};

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

#[derive(Clone, Debug)]
pub struct HyraxOpeningProof<F: JoltField> {
    pub combined_row: Vec<F>,
    pub combined_blinding: F,
}

impl<F: JoltField> CanonicalSerialize for HyraxOpeningProof<F> {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        serialize_vec_with_len(&self.combined_row, &mut writer, compress)?;
        self.combined_blinding
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        serialized_vec_with_len_size(&self.combined_row, compress)
            + self.combined_blinding.serialized_size(compress)
    }
}

impl<F: JoltField> ark_serialize::Valid for HyraxOpeningProof<F> {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.combined_row.check()?;
        self.combined_blinding.check()
    }
}

impl<F: JoltField> CanonicalDeserialize for HyraxOpeningProof<F> {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let proof = Self {
            combined_row: deserialize_bounded_vec(
                &mut reader,
                compress,
                validate,
                MAX_BLINDFOLD_VECTOR_LEN,
            )?,
            combined_blinding: F::deserialize_with_mode(&mut reader, compress, validate)?,
        };
        if validate == ark_serialize::Validate::Yes {
            proof.check()?;
        }
        Ok(proof)
    }
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
