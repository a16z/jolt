use super::dense_mlpoly::{PolyCommitment, PolyCommitmentGens};
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::math::Math;
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use bitvec::prelude::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

pub struct IndicatorPolynomial {
    pub num_vars: usize,
    pub column_bitvectors: Vec<BitVec>,
}

impl IndicatorPolynomial {
    pub fn evaluate<F: PrimeField>(&self, r: &[F]) -> F {
        let chis = EqPolynomial::new(r.to_vec()).evals();
        let column_size = (self.num_vars - self.num_vars / 2).pow2();
        self.column_bitvectors
            .par_iter()
            .enumerate()
            .flat_map(|(i, bitvector)| {
                bitvector
                    .iter_ones()
                    .map(|j| chis[i * column_size + j])
                    .collect::<Vec<F>>()
            })
            .sum::<F>()
    }

    pub fn evaluate_at_chi<F: PrimeField>(&self, chis: &Vec<F>) -> F {
        let column_size = (self.num_vars - self.num_vars / 2).pow2();
        self.column_bitvectors
            .par_iter()
            .enumerate()
            .flat_map(|(i, bitvector)| {
                bitvector
                    .iter_ones()
                    .map(|j| chis[i * column_size + j])
                    .collect::<Vec<F>>()
            })
            .sum::<F>()
    }

    pub fn commit<F: PrimeField, G: CurveGroup<ScalarField = F>>(
        &self,
        gens: &PolyCommitmentGens<G>,
    ) -> PolyCommitment<G> {
        let gens = CurveGroup::normalize_batch(&gens.gens.gens_n.G);

        let C: Vec<G> = self
            .column_bitvectors
            .par_iter()
            .map(|bitvector| bitvector.iter_ones().map(|i| gens[i]).sum())
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
        let num_columns = 1 << (num_vars / 2);
        let column_size = 1 << (num_vars - num_vars / 2);

        let mut flat_bitvector: Vec<usize> = vec![0usize; m];
        let mut column_bitvectors: Vec<BitVec> = vec![bitvec![0; column_size]; num_columns];

        for i in 0..m {
            if rng.next_u32() as usize % 10 == 0 {
                flat_bitvector[i] = 1;
                column_bitvectors[i / num_columns].set(i % column_size, true);
            }
        }

        let normal_poly: DensePolynomial<Fr> = DensePolynomial::from_usize(&flat_bitvector);
        let indicator_poly: IndicatorPolynomial = IndicatorPolynomial {
            num_vars,
            column_bitvectors,
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
