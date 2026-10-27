use super::Fp128;
use crate::One;

/// Recognize every entry, including the constant coefficient, before using
/// Horner evaluation. Other callers of weighted dots may supply arbitrary powers.
pub(super) fn ratio<const P: u128>(powers: &[Fp128<P>]) -> Option<Fp128<P>> {
    let (&first, rest) = powers.split_first()?;
    let &alpha = rest.first()?;
    if first != Fp128::one() {
        return None;
    }
    let mut previous = alpha;
    for &power in rest.iter().skip(1) {
        if previous * alpha != power {
            return None;
        }
        previous = power;
    }
    Some(alpha)
}

#[cfg(test)]
mod tests {
    use super::ratio;
    use crate::{One, Prime128OffsetA7F7 as F, Ring, Zero};

    #[test]
    fn recognizes_geometric_powers_including_degenerate_challenges() {
        for alpha in [F::zero(), F::one(), -F::one(), F::from_u64(17)] {
            let powers: Vec<F> = (0..64).map(|i| alpha.pow_u128(i)).collect();
            assert_eq!(ratio(&powers), Some(alpha));
        }
    }

    #[test]
    fn rejects_each_changed_power_and_nonunit_constant() {
        let powers: Vec<F> = (0..64).map(|i| F::from_u64(17).pow_u128(i)).collect();
        for index in 0..powers.len() {
            let mut changed = powers.clone();
            changed[index] += F::one();
            assert_eq!(ratio(&changed), None);
        }
        let empty: [F; 0] = [];
        assert_eq!(ratio(&empty), None);
        assert_eq!(ratio(&[F::one()]), None);
    }
}
