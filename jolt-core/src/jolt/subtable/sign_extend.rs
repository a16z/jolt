use crate::field::JoltField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;

#[derive(Default, Debug)]
pub struct SignExtendSubtable<F: JoltField, const WIDTH: usize> {
    _field: PhantomData<F>,
}

impl<F: JoltField, const WIDTH: usize> SignExtendSubtable<F, WIDTH> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField, const WIDTH: usize> LassoSubtable<F> for SignExtendSubtable<F, WIDTH> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        // table[x] = x[b - WIDTH] * (2^{WIDTH} - 1)
        // Take the WIDTH-th bit of the input (counting from the LSB), then multiply by (2^{WIDTH} - 1)
        // Requires `log2(M) >= WIDTH`
        debug_assert!(WIDTH <= log2(M) as usize);
        let mut entries = Vec::with_capacity(M);

        // The sign-extension will be the same width as the value being extended
        // –– this is convenient for how this subtable is used in LB and LH.
        let ones: u64 = (1 << WIDTH) - 1;

        for idx in 0..M {
            let sign_bit = ((idx >> (WIDTH - 1)) & 1) as u64;
            let sign_extension = sign_bit * ones;
            let row = sign_extension as u32;
            entries.push(row);
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // 2 ^ {WIDTH - 1} * x_{b - WIDTH}
        debug_assert!(point.len() >= WIDTH);

        let sign_bit = point[point.len() - WIDTH];
        let ones: u64 = (1 << WIDTH) - 1;
        sign_bit * F::from_u64(ones)
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField,
        jolt::subtable::{sign_extend::SignExtendSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        sign_extend_8_materialize_mle_parity,
        SignExtendSubtable<Fr, 8>,
        Fr,
        1 << 16
    );

    subtable_materialize_mle_parity_test!(
        sign_extend_16_materialize_mle_parity,
        SignExtendSubtable<Fr, 16>,
        Fr,
        1 << 16
    );
}
