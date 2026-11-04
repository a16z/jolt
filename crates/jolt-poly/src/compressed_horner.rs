use super::CompressedPoly;
use jolt_field::Field;

impl<F: Field> CompressedPoly<F> {
    /// Evaluates at `point` with Horner's rule, reconstructing the linear
    /// coefficient from the sumcheck hint `f(0) + f(1)`.
    ///
    /// # Panics
    /// Panics if the polynomial is empty ([`is_empty`](Self::is_empty)).
    #[inline]
    pub fn eval_from_hint(&self, hint: &F, point: &F) -> F {
        let linear_term = self.recover_linear_term(*hint);
        let c0 = self.coeffs_except_linear_term[0];
        let mut higher = self.coeffs_except_linear_term[1..].iter().rev();
        let Some(&leading) = higher.next() else {
            return c0 + *point * linear_term;
        };
        let mut value = leading;
        for &coefficient in higher {
            value = value * *point + coefficient;
        }
        (value * *point + linear_term) * *point + c0
    }
}
