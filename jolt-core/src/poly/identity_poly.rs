use crate::field::JoltField;

use crate::utils::math::Math;

pub struct IdentityPolynomial {
    size_point: usize,
}

impl IdentityPolynomial {
    pub fn new(size_point: usize) -> Self {
        IdentityPolynomial { size_point }
    }

    pub fn evaluate<F: JoltField>(&self, r: &[F]) -> F {
        let len = r.len();
        assert_eq!(len, self.size_point);
        (0..len)
            .map(|i| F::from_u64((len - i - 1).pow2() as u64).unwrap() * r[i])
            .sum()
    }
}
