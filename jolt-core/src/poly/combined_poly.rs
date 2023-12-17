use ark_ff::PrimeField;

use crate::utils::{compute_dotproduct, math::Math};

use super::dense_mlpoly::DensePolynomial;

pub struct CombinedPoly<'a, F> {
    polys: Vec<&'a DensePolynomial<F>>,
    len: usize,
    num_vars: usize,
}

// TODO(sragss): Implictly padded to the next power of 2 with zeros
impl<'a, F: PrimeField> CombinedPoly<'a, F> {
    pub fn new(polys: Vec<&'a DensePolynomial<F>>) -> Self {
        let len: usize = polys.iter().map(|poly| poly.len()).sum();
        let len: usize = len.next_power_of_two();
        let num_vars = len.log_2();

        CombinedPoly {
            polys,
            len,
            num_vars,
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn evaluate(&self, r: &[F]) -> F {
        assert_eq!(r.len(), self.get_num_vars());

        let chis = EqPolynomial::new(r.to_vec()).evals();
        assert_eq!(chis.len(), self.Z.len());

        compute_dotproduct(self.evals_iter(), &chis)
    }

    pub fn combined_commit<G>(
        &self,
        label: &'static [u8],
    ) -> (PolyCommitmentGens<G>, CombinedTableCommitment<G>)
    where
        G: CurveGroup<ScalarField = F>,
    {
        // let generators = PolyCommitmentGens::new(self.num_vars, label);
        // let (joint_commitment, _) = self.commit(&generators, None);
        // (generators, CombinedTableCommitment::new(joint_commitment))
        todo!("combined_commit")
    }

    fn evals_iter(&self) -> impl Iterator<Item = &F> {
        self.polys.iter()
            .flat_map(|poly| poly.evals_ref())
    }
}