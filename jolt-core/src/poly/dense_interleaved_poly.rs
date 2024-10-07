use crate::field::JoltField;
use rayon::prelude::*;

#[cfg(test)]
use super::dense_mlpoly::DensePolynomial;

#[derive(Default, Debug)]
pub struct DenseInterleavedPolynomial<F: JoltField> {
    // num_vars: usize,
    gap: usize, // TODO(moodlezoup): Gap or scratch space approach?
    coeffs: Vec<F>,
}

impl<F: JoltField> DenseInterleavedPolynomial<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        assert!(coeffs.len() % 2 == 0);
        Self {
            // num_vars: coeffs.len().next_power_of_two().log_2(),
            gap: 1,
            coeffs,
        }
    }

    pub fn len(&self) -> usize {
        self.coeffs.len().next_power_of_two() / self.gap
    }

    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.coeffs.iter().step_by(self.gap)
    }

    #[cfg(test)]
    pub fn interleave(left: &DensePolynomial<F>, right: &DensePolynomial<F>) -> Self {
        assert_eq!(left.len(), right.len());
        let mut interleaved = vec![];
        for i in 0..left.len() {
            interleaved.push(left[i]);
            interleaved.push(right[i]);
        }
        Self::new(interleaved)
    }

    #[cfg(test)]
    pub fn uninterleave(&self) -> (DensePolynomial<F>, DensePolynomial<F>) {
        let left_poly = DensePolynomial::new_padded(
            self.coeffs
                .clone()
                .into_iter()
                .step_by(2 * self.gap)
                .collect(),
        );
        let right_poly = DensePolynomial::new_padded(
            self.coeffs
                .clone()
                .into_iter()
                .skip(self.gap)
                .step_by(2 * self.gap)
                .collect(),
        );
        (left_poly, right_poly)
    }

    pub fn bind(&mut self, r: F) {
        self.coeffs.par_chunks_mut(4 * self.gap).for_each(|chunk| {
            // Left
            chunk[0] = chunk[0] + r * (chunk[2 * self.gap] - chunk[0]);
            // Right
            chunk[2 * self.gap] = chunk[self.gap] + r * (chunk[3 * self.gap] - chunk[self.gap]);
        });

        self.gap *= 2;
    }
}

impl<F: JoltField> Clone for DenseInterleavedPolynomial<F> {
    fn clone(&self) -> Self {
        Self::new(self.coeffs.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;

    #[test]
    fn interleave_uninterleave() {
        let mut rng = test_rng();
        const NUM_VARS: usize = 10;
        let left: Vec<_> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(1 << NUM_VARS)
            .collect();
        let left = DensePolynomial::new(left);
        let right: Vec<_> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(1 << NUM_VARS)
            .collect();
        let right = DensePolynomial::new(right);

        let interleaved = DenseInterleavedPolynomial::interleave(&left, &right);

        assert_eq!(interleaved.uninterleave(), (left, right));
    }

    #[test]
    fn uninterleave_interleave() {
        let mut rng = test_rng();
        const NUM_VARS: usize = 10;
        let coeffs: Vec<_> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(2 * (1 << NUM_VARS))
            .collect();
        let interleaved = DenseInterleavedPolynomial::new(coeffs);
        let (left, right) = interleaved.uninterleave();

        assert_eq!(
            interleaved.iter().collect::<Vec<_>>(),
            DenseInterleavedPolynomial::interleave(&left, &right)
                .iter()
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn bind() {
        let mut rng = test_rng();
        const NUM_VARS: usize = 10;
        let left: Vec<_> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(1 << NUM_VARS)
            .collect();
        let mut left = DensePolynomial::new(left);
        let right: Vec<_> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(1 << NUM_VARS)
            .collect();
        let mut right = DensePolynomial::new(right);

        let mut interleaved = DenseInterleavedPolynomial::interleave(&left, &right);

        let r = Fr::random(&mut rng);
        interleaved.bind(r);
        left.bound_poly_var_bot(&r);
        right.bound_poly_var_bot(&r);

        assert_eq!(
            interleaved.iter().collect::<Vec<_>>(),
            DenseInterleavedPolynomial::interleave(&left, &right)
                .iter()
                .collect::<Vec<_>>()
        );
    }
}
