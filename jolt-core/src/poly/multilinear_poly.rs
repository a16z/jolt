use ark_ff::PrimeField;

pub trait MultilinearPolynomial<F: PrimeField> {
    fn get_num_vars(&self) -> usize;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn split(&self, idx: usize) -> (Self, Self);
    fn split_evals(&self, idx: usize) -> (impl Iterator<Item = F>, impl Iterator<Item = F>);
    fn bound_poly_var_top(&mut self, r: &F);
    fn evaluate(&self, r: &[F]) -> F;
}