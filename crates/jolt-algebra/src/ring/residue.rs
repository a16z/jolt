//! Linear-time coefficient functionals for reduced negacyclic relations.

use akita_error::AkitaError;
use jolt_field::{ExtField, Field};

/// Reusable evaluation data for reduced negacyclic residue kernels at one
/// point and dimension.
///
/// Preparing the powers once matters when a compiler evaluates many public
/// multipliers at the same point, as the quotient-free prover does for every
/// setup column in one role.
#[derive(Clone, Debug)]
pub struct ResidueKernelPoint<E: Field> {
    alpha: E,
    powers: Vec<E>,
    modulus_evaluation: E,
}

impl<E: Field> ResidueKernelPoint<E> {
    /// Prepare `1, alpha, ..., alpha^(d-1)` and `alpha^d + 1`.
    pub fn new(alpha: E, dimension: usize) -> Result<Self, AkitaError> {
        if dimension == 0 || !dimension.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "residue-kernel dimension must be a nonzero power of two".into(),
            ));
        }
        let mut powers = Vec::new();
        powers
            .try_reserve_exact(dimension)
            .map_err(|_| AkitaError::InvalidInput("residue-kernel allocation failed".into()))?;
        let mut power = E::one();
        for _ in 0..dimension {
            powers.push(power);
            power *= alpha;
        }
        Ok(Self {
            alpha,
            powers,
            modulus_evaluation: power + E::one(),
        })
    }

    /// Ring dimension owned by this prepared point.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.powers.len()
    }

    /// Prepared powers `1, alpha, ..., alpha^(d-1)`.
    #[must_use]
    pub fn powers(&self) -> &[E] {
        &self.powers
    }

    /// Evaluate every reduced shift of one base-field multiplier.
    pub fn kernel<F>(&self, coefficients: &[F]) -> Result<Vec<E>, AkitaError>
    where
        F: Field,
        E: ExtField<F>,
    {
        self.kernel_with(coefficients, E::mul_base)
    }

    /// Evaluate every reduced shift of extension-field coefficient weights.
    pub fn field_kernel(&self, weights: &[E]) -> Result<Vec<E>, AkitaError> {
        self.kernel_with(weights, |value, weight| value * weight)
    }

    /// Write the extension-field residue kernel into an existing exact-size
    /// destination.
    pub fn field_kernel_into(&self, weights: &[E], output: &mut [E]) -> Result<(), AkitaError> {
        self.kernel_with_into(weights, output, |value, weight| value * weight)
    }

    #[inline]
    fn kernel_with<W: Copy>(
        &self,
        weights: &[W],
        multiply: impl Fn(E, W) -> E,
    ) -> Result<Vec<E>, AkitaError> {
        let mut kernel = Vec::new();
        kernel
            .try_reserve_exact(self.dimension())
            .map_err(|_| AkitaError::InvalidInput("residue-kernel allocation failed".into()))?;
        kernel.resize(self.dimension(), E::zero());
        self.kernel_with_into(weights, &mut kernel, multiply)?;
        Ok(kernel)
    }

    #[inline]
    fn kernel_with_into<W: Copy>(
        &self,
        weights: &[W],
        output: &mut [E],
        multiply: impl Fn(E, W) -> E,
    ) -> Result<(), AkitaError> {
        if weights.len() != self.dimension() || output.len() != self.dimension() {
            return Err(AkitaError::InvalidInput(
                "residue-kernel buffers do not match the prepared dimension".into(),
            ));
        }
        let mut current = weights
            .iter()
            .copied()
            .zip(&self.powers)
            .fold(E::zero(), |sum, (weight, &power)| {
                sum + multiply(power, weight)
            });
        let (first, tail) = output
            .split_first_mut()
            .ok_or_else(|| AkitaError::InvalidInput("residue-kernel output is empty".into()))?;
        *first = current;
        for (slot, wrap_weight) in tail
            .iter_mut()
            .zip(weights.iter().copied().rev().take(self.dimension() - 1))
        {
            current = self.alpha * current - multiply(self.modulus_evaluation, wrap_weight);
            *slot = current;
        }
        Ok(())
    }

    /// Evaluate every reduced shift of one sparse extension-field multiplier.
    pub fn sparse_kernel(
        &self,
        terms: impl IntoIterator<Item = (usize, E)>,
    ) -> Result<Vec<E>, AkitaError> {
        let mut terms = terms.into_iter().collect::<Vec<_>>();
        terms.sort_unstable_by_key(|(position, _)| *position);
        for (index, &(position, weight)) in terms.iter().enumerate() {
            if position >= self.dimension() {
                return Err(AkitaError::InvalidInput(
                    "sparse residue-kernel position is out of range".into(),
                ));
            }
            if weight.is_zero() {
                return Err(AkitaError::InvalidInput(
                    "sparse residue-kernel weights must be nonzero".into(),
                ));
            }
            if index != 0 && terms[index - 1].0 == position {
                return Err(AkitaError::InvalidInput(
                    "sparse residue-kernel positions must be unique".into(),
                ));
            }
        }

        let mut current = terms.iter().fold(E::zero(), |sum, &(position, weight)| {
            sum + weight * self.powers[position]
        });
        let mut kernel = Vec::new();
        kernel
            .try_reserve_exact(self.dimension())
            .map_err(|_| AkitaError::InvalidInput("residue-kernel allocation failed".into()))?;
        kernel.push(current);
        let mut reverse_term_index = terms.len();
        for wrap_position in (1..self.dimension()).rev() {
            let wrap_weight =
                if reverse_term_index != 0 && terms[reverse_term_index - 1].0 == wrap_position {
                    reverse_term_index -= 1;
                    terms[reverse_term_index].1
                } else {
                    E::zero()
                };
            current = self.alpha * current - self.modulus_evaluation * wrap_weight;
            kernel.push(current);
        }
        Ok(kernel)
    }
}

/// Evaluate every reduced shift of one public multiplier at `alpha`.
///
/// For `A(X) = sum_k coefficients[k] X^k` in a ring of dimension `d`, the
/// returned entry `j` is
///
/// `kappa[j] = (A(X) X^j mod (X^d + 1))(alpha)`.
///
/// The implementation uses the signed-wrap recurrence and never divides by
/// `alpha^d + 1`, so roots of the polynomial modulus are valid evaluation
/// points.
///
/// # Errors
///
/// Returns an error unless the coefficient count is a nonzero power of two,
/// or if the output allocation cannot be reserved.
pub fn residue_kernel<F, E>(coefficients: &[F], alpha: E) -> Result<Vec<E>, AkitaError>
where
    F: Field,
    E: Field + ExtField<F>,
{
    ResidueKernelPoint::new(alpha, coefficients.len())?.kernel(coefficients)
}

/// Prepare terminal weights for one exact physical equality window.
///
/// `equality_weights[j]` must be the checked value `eq(point, start + j)` for
/// the physical native coefficient window being evaluated. The returned entry
/// `k` is the multilinear contraction of the `k`-th signed negacyclic shift:
///
/// `H[k] = sum_j equality_weights[j] * sign(k + j) * alpha^((k + j) mod d)`.
///
/// Windows with different physical starts must be prepared separately, even
/// when their native dimensions agree.
///
/// # Errors
///
/// Returns an error unless the equality-window length is a nonzero power of
/// two, or if the output allocation cannot be reserved.
pub fn terminal_residue_kernel<E>(equality_weights: &[E], alpha: E) -> Result<Vec<E>, AkitaError>
where
    E: Field,
{
    ResidueKernelPoint::new(alpha, equality_weights.len())?.field_kernel(equality_weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offset_eq::OffsetEqWindow;
    use crate::ring::eval::eval_ring_at_pows;
    use crate::ring::{scalar_powers, CyclotomicRing};
    use jolt_field::{
        Ext2, ExtField, FpExt4, One, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59, Ring,
        Zero,
    };
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn quadratic_kernel<E: Field>(coefficients: &[E], alpha: E) -> Vec<E> {
        let dimension = coefficients.len();
        let powers = scalar_powers(alpha, dimension);
        (0..dimension)
            .map(|shift| {
                coefficients
                    .iter()
                    .enumerate()
                    .fold(E::zero(), |sum, (coefficient, &value)| {
                        let exponent = coefficient + shift;
                        let term = value * powers[exponent % dimension];
                        if exponent < dimension {
                            sum + term
                        } else {
                            sum - term
                        }
                    })
            })
            .collect()
    }

    fn check_product_identity<F, E, const D: usize>(seed: u64)
    where
        F: Field,
        E: Field + ExtField<F>,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let multiplier = CyclotomicRing::<F, D>::random(&mut rng);
        let witness = CyclotomicRing::<F, D>::random(&mut rng);
        let alpha = E::random(&mut rng);
        let powers = scalar_powers(alpha, D);
        let kernel = residue_kernel(multiplier.coefficients(), alpha).expect("residue kernel");
        let lifted_coefficients = multiplier
            .coefficients()
            .iter()
            .copied()
            .map(E::lift_base)
            .collect::<Vec<_>>();

        assert_eq!(
            kernel,
            quadratic_kernel(&lifted_coefficients, alpha),
            "linear and quadratic kernels differ at dimension {D}"
        );
        let product_evaluation = eval_ring_at_pows(&(multiplier * witness), &powers);
        let kernel_evaluation = kernel
            .iter()
            .zip(witness.coefficients())
            .fold(E::zero(), |sum, (&weight, &coefficient)| {
                sum + weight.mul_base(coefficient)
            });
        assert_eq!(product_evaluation, kernel_evaluation);
    }

    macro_rules! check_admitted_dimensions {
        ($base:ty, $extension:ty, $seed:expr) => {{
            check_product_identity::<$base, $extension, 64>($seed);
            check_product_identity::<$base, $extension, 128>($seed + 1);
            check_product_identity::<$base, $extension, 256>($seed + 2);
            check_product_identity::<$base, $extension, 512>($seed + 3);
            check_product_identity::<$base, $extension, 1024>($seed + 4);
            check_product_identity::<$base, $extension, 2048>($seed + 5);
        }};
    }

    #[test]
    fn residue_recurrence_matches_quadratic_and_ring_product_oracles() {
        for seed in 1..=8 {
            check_product_identity::<Prime32Offset99, FpExt4<Prime32Offset99>, 1>(seed);
            check_product_identity::<Prime32Offset99, FpExt4<Prime32Offset99>, 2>(seed + 10);
            check_product_identity::<Prime32Offset99, FpExt4<Prime32Offset99>, 4>(seed + 20);
            check_product_identity::<Prime32Offset99, FpExt4<Prime32Offset99>, 8>(seed + 30);
            check_product_identity::<Prime64Offset59, Ext2<Prime64Offset59>, 16>(seed + 40);
            check_product_identity::<Prime64Offset59, Ext2<Prime64Offset59>, 32>(seed + 50);
            check_product_identity::<Prime128OffsetA7F7, Prime128OffsetA7F7, 64>(seed + 60);
            check_product_identity::<Prime128OffsetA7F7, Prime128OffsetA7F7, 128>(seed + 70);
        }

        // This is the complete commitment-dimension admission surface. Keep
        // it synchronized with `akita_params::SUPPORTED_COMMITMENT_RING_DIMS`.
        check_admitted_dimensions!(Prime32Offset99, FpExt4<Prime32Offset99>, 101);
        check_admitted_dimensions!(Prime64Offset59, Ext2<Prime64Offset59>, 201);
        check_admitted_dimensions!(Prime128OffsetA7F7, Prime128OffsetA7F7, 301);
    }

    #[test]
    fn fp64_terminal_kernel_uses_the_configured_extension_field() {
        type E = Ext2<Prime64Offset59>;
        let mut rng = StdRng::seed_from_u64(401);
        let weights = (0..64).map(|_| E::random(&mut rng)).collect::<Vec<_>>();
        let alpha = E::random(&mut rng);
        assert_eq!(
            terminal_residue_kernel(&weights, alpha).unwrap(),
            quadratic_kernel(&weights, alpha)
        );
    }

    #[test]
    fn terminal_recurrence_consumes_exact_offset_equality_windows() {
        type E = FpExt4<Prime32Offset99>;

        for seed in 1..=8 {
            let mut rng = StdRng::seed_from_u64(seed);
            let point = (0..8).map(|_| E::random(&mut rng)).collect::<Vec<_>>();
            let equality = OffsetEqWindow::new(&point).expect("equality window");
            let alpha = E::random(&mut rng);
            for (start, dimension) in [(0, 8), (3, 8), (64, 16), (127, 16)] {
                let mut weights = vec![E::zero(); dimension];
                equality
                    .fill_interval(start, &mut weights)
                    .expect("checked physical equality interval");
                assert_eq!(
                    terminal_residue_kernel(&weights, alpha).expect("terminal kernel"),
                    quadratic_kernel(&weights, alpha),
                    "terminal kernel differs at seed {seed}, start {start}, dimension {dimension}"
                );
            }

            let mut first = vec![E::zero(); 8];
            let mut second = vec![E::zero(); 8];
            equality.fill_interval(0, &mut first).unwrap();
            equality.fill_interval(3, &mut second).unwrap();
            assert_ne!(
                terminal_residue_kernel(&first, alpha).unwrap(),
                terminal_residue_kernel(&second, alpha).unwrap(),
                "different physical starts must not share one terminal kernel"
            );
        }
    }

    #[test]
    fn base_multiplier_matches_literal_oracle_at_genuine_extension_point() {
        use crate::offset_eq::eq_eval_at_index;
        use crate::ring::eval_flat_ring_at_pows_fast;
        use jolt_field::{Fp32, FpExt2, NegOneNr};
        type B = Fp32<251>;
        type X = FpExt2<B, NegOneNr>;

        let extension = |lo, hi| X::from_base_slice(&[B::from_u64(lo), B::from_u64(hi)]);
        let point = [
            extension(3, 5),
            extension(7, 11),
            extension(13, 17),
            extension(19, 23),
        ];
        let alpha = extension(29, 31);
        let multiplier = (0..8)
            .map(|index| B::from_u64(37 + index as u64))
            .collect::<Vec<_>>();
        let mut equality = vec![X::zero(); 8];
        OffsetEqWindow::new(&point)
            .unwrap()
            .fill_interval(5, &mut equality)
            .unwrap();
        let functional = terminal_residue_kernel(&equality, alpha).unwrap();
        let powers = scalar_powers(alpha, 8);
        let expected = (0..8).fold(X::zero(), |evaluation, witness_coefficient| {
            let residue = multiplier.iter().enumerate().fold(
                X::zero(),
                |sum, (multiplier_coefficient, &coefficient)| {
                    let exponent = multiplier_coefficient + witness_coefficient;
                    let product = powers[exponent % 8].mul_base(coefficient);
                    if exponent < 8 {
                        sum + product
                    } else {
                        sum - product
                    }
                },
            );
            evaluation + eq_eval_at_index(&point, 5 + witness_coefficient) * residue
        });
        assert_eq!(
            eval_flat_ring_at_pows_fast(&multiplier, &functional),
            expected
        );
    }

    #[test]
    fn modulus_roots_do_not_require_division() {
        type F = Prime128OffsetA7F7;
        let alpha = crate::fft::primitive_nth_root::<F>(4);
        let coefficients = [F::from_u64(37), F::from_u64(13)];
        let weights = [F::from_u64(41), F::from_u64(19)];

        assert_eq!(alpha.square() + F::one(), F::zero());
        assert_eq!(
            residue_kernel(&coefficients, alpha).unwrap(),
            quadratic_kernel(&coefficients, alpha)
        );
        assert_eq!(
            terminal_residue_kernel(&weights, alpha).unwrap(),
            quadratic_kernel(&weights, alpha)
        );
    }

    #[test]
    fn malformed_dimensions_are_rejected_before_output_allocation() {
        type F = Prime32Offset99;
        for coefficients in [&[][..], &[F::one(); 3][..]] {
            assert!(residue_kernel::<F, F>(coefficients, F::from_u64(7)).is_err());
            assert!(terminal_residue_kernel(coefficients, F::from_u64(7)).is_err());
        }
    }

    #[test]
    fn sparse_recurrence_matches_dense_recurrence() {
        type F = Prime128OffsetA7F7;
        let alpha = F::from_u64(7);
        let mut dense = vec![F::zero(); 64];
        let terms = [(0, F::from_u64(3)), (17, -F::one()), (63, F::from_u64(2))];
        for &(position, coefficient) in &terms {
            dense[position] = coefficient;
        }

        assert_eq!(
            ResidueKernelPoint::new(alpha, 64)
                .unwrap()
                .sparse_kernel(terms)
                .unwrap(),
            residue_kernel::<F, F>(&dense, alpha).unwrap()
        );
    }

    #[test]
    fn prepared_point_matches_one_shot_base_and_field_kernels() {
        type F = Prime32Offset99;
        type E = FpExt4<F>;
        let mut rng = StdRng::seed_from_u64(509);
        let coefficients = (0..64).map(|_| F::random(&mut rng)).collect::<Vec<_>>();
        let weights = (0..64).map(|_| E::random(&mut rng)).collect::<Vec<_>>();
        let alpha = E::random(&mut rng);
        let point = ResidueKernelPoint::new(alpha, 64).unwrap();

        assert_eq!(
            point.kernel(&coefficients).unwrap(),
            residue_kernel(&coefficients, alpha).unwrap()
        );
        let expected = terminal_residue_kernel(&weights, alpha).unwrap();
        let mut output = vec![E::zero(); 64];
        point.field_kernel_into(&weights, &mut output).unwrap();
        assert_eq!(point.field_kernel(&weights).unwrap(), expected);
        assert_eq!(output, expected);
    }

    #[test]
    fn sparse_recurrence_rejects_malformed_terms() {
        type F = Prime128OffsetA7F7;
        let alpha = F::from_u64(7);
        assert!(ResidueKernelPoint::new(alpha, 3).is_err());
        let point = ResidueKernelPoint::new(alpha, 4).unwrap();
        assert!(point.sparse_kernel([(4, F::one())]).is_err());
        assert!(point
            .sparse_kernel([(1, F::one()), (1, -F::one())])
            .is_err());
        assert!(point.sparse_kernel([(1, F::zero())]).is_err());
    }
}
