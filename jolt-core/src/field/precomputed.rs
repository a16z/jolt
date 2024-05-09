use binius_field::{BinaryField, BinaryField128b, BinaryField128bPolyval};
use once_cell::sync::Lazy;

pub const LOG_TABLE_SIZE: usize = 16;
pub const TABLE_SIZE: usize = 1 << LOG_TABLE_SIZE;

// BinaryField128b
pub static PRECOMPUTED_LOW_128B: Lazy<[BinaryField128b; TABLE_SIZE]> = Lazy::new(|| {
    compute_powers::<BinaryField128b, TABLE_SIZE>(BinaryField128b::MULTIPLICATIVE_GENERATOR)
});
pub static PRECOMPUTED_HIGH_128B: Lazy<[BinaryField128b; TABLE_SIZE]> = Lazy::new(|| {
    compute_powers_starting_from::<BinaryField128b, TABLE_SIZE>(
        BinaryField128b::MULTIPLICATIVE_GENERATOR,
        16,
    )
});

// BinaryField128bPolyval
pub static PRECOMPUTED_LOW_128B_POLYVAL: Lazy<[BinaryField128bPolyval; TABLE_SIZE]> =
    Lazy::new(|| {
        compute_powers::<BinaryField128bPolyval, TABLE_SIZE>(
            BinaryField128bPolyval::MULTIPLICATIVE_GENERATOR,
        )
    });
pub static PRECOMPUTED_HIGH_128B_POLYVAL: Lazy<[BinaryField128bPolyval; TABLE_SIZE]> =
    Lazy::new(|| {
        compute_powers_starting_from::<BinaryField128bPolyval, TABLE_SIZE>(
            BinaryField128bPolyval::MULTIPLICATIVE_GENERATOR,
            16,
        )
    });

/// Computes `N` sequential powers of base^{[0, ... N]}
const fn compute_powers<F: binius_field::BinaryField, const N: usize>(base: F) -> [F; N] {
    let mut powers = [F::ZERO; N];
    powers[0] = F::ONE;
    powers[1] = base;
    let mut i = 2;
    while i < N {
        powers[i] = powers[i - 1] * base;
        i += 1;
    }
    powers
}

/// Computes `N` sequential powers of base^{2^starting_power}: {base^{2^starting_power}}^{[1, ... N]}
const fn compute_powers_starting_from<F: binius_field::BinaryField, const N: usize>(
    base: F,
    starting_power: usize,
) -> [F; N] {
    let mut starting = base;
    let mut count = 0;
    // // Repeated squaring
    while count < starting_power {
        starting = starting.mul(starting);
        count += 1;
    }
    compute_powers::<F, N>(starting)
}

#[cfg(test)]
mod tests {
    use super::*;
    use binius_field::{BinaryField, Field};

    #[test]
    fn test_compute_powers() {
        let base = BinaryField128b::MULTIPLICATIVE_GENERATOR;
        let powers = compute_powers::<BinaryField128b, 5>(base);
        let expected_powers = [
            BinaryField128b::ONE,
            base,
            base * base,
            base * base * base,
            base * base * base * base,
        ];
        assert_eq!(powers, expected_powers);
    }

    #[test]
    fn test_compute_powers_starting_from() {
        let base = BinaryField128b::MULTIPLICATIVE_GENERATOR;
        let powers = compute_powers_starting_from::<BinaryField128b, 5>(base, 2);
        let expected_starting_base = base * base * base * base;
        let expected_powers = [
            BinaryField128b::ONE,
            expected_starting_base,
            expected_starting_base * expected_starting_base,
            expected_starting_base * expected_starting_base * expected_starting_base,
            expected_starting_base
                * expected_starting_base
                * expected_starting_base
                * expected_starting_base,
        ];
        assert_eq!(powers, expected_powers);
    }
}
