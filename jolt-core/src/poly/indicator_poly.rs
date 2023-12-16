use super::dense_mlpoly::{PolyCommitment, PolyCommitmentGens};
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::math::Math;
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::prelude::ParallelSlice;

pub struct IndicatorPolynomial {
    pub num_vars: usize,
    pub bitvector: Vec<bool>,
}

impl IndicatorPolynomial {
    pub fn evaluate<F: PrimeField>(&self, r: &[F]) -> F {
        let chis = EqPolynomial::new(r.to_vec()).evals();
        self.bitvector
            .par_iter()
            .enumerate()
            .filter_map(|(i, &bit)| if bit { Some(chis[i]) } else { None })
            .sum::<F>()
    }

    pub fn evaluate_at_chi<F: PrimeField>(&self, chis: &Vec<F>) -> F {
        self.bitvector
            .par_iter()
            .enumerate()
            .filter_map(|(i, &bit)| if bit { Some(chis[i]) } else { None })
            .sum::<F>()
    }

    pub fn commit<F: PrimeField, G: CurveGroup<ScalarField = F>>(
        &self,
        gens: &PolyCommitmentGens<G>,
    ) -> PolyCommitment<G> {
        let column_size = (self.num_vars - self.num_vars / 2).pow2();
        let gens = CurveGroup::normalize_batch(&gens.gens.gens_n.G);

        let C: Vec<G> = self
            .bitvector
            .par_chunks_exact(column_size)
            .map(|column_bits| {
                column_bits
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &bit)| if bit { Some(gens[i]) } else { None })
                    .sum()
            })
            .collect();

        PolyCommitment { C }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::dense_mlpoly::DensePolynomial;
    use ark_curve25519::{EdwardsProjective as G1Projective, Fr};
    use ark_std::{rand::RngCore, test_rng};

    #[test]
    fn reference_test() {
        let mut rng = test_rng();

        let num_vars: usize = 20;
        let m: usize = 1 << num_vars;
        let mut indicator_bitvector: Vec<usize> = vec![0usize; m];

        for i in 0..m {
            if rng.next_u32() as usize % 10 == 0 {
                indicator_bitvector[i] = 1;
            }
        }

        let normal_poly: DensePolynomial<Fr> = DensePolynomial::from_usize(&indicator_bitvector);
        let indicator_poly: IndicatorPolynomial = IndicatorPolynomial {
            num_vars,
            bitvector: indicator_bitvector.iter().map(|&x| x == 1).collect(),
        };

        let gens: PolyCommitmentGens<G1Projective> =
            PolyCommitmentGens::new(num_vars, b"test_gens");

        let r = vec![Fr::from(rng.next_u64()); num_vars];
        assert_eq!(normal_poly.evaluate(&r), indicator_poly.evaluate(&r));
        assert_eq!(
            normal_poly.commit(&gens, None).0,
            indicator_poly.commit(&gens)
        );
    }
}
