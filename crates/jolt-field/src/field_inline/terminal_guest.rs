    /// Canonical Fp128 weights and signed integer coefficients. Only the
    /// magnitude enters register B; subtraction carries the sign. The unit
    /// uses existing load, multiply and add/subtract instructions.
    #[cfg(feature = "terminal-contraction")]
    pub fn signed_i16_dot(
        weights: &[[u64; 2]],
        coefficients: impl IntoIterator<Item = i16>,
    ) -> [u64; 2] {
        ACC_LENT.store(true, Ordering::Relaxed);
        emit::acc_zero();
        for (weight, coefficient) in weights.iter().zip(coefficients) {
            let magnitude = [u64::from(coefficient.unsigned_abs())];
            load(REG_A, weight);
            load(REG_B, &magnitude);
            emit::mul_out();
            if coefficient < 0 {
                emit::acc_sub_out();
            } else {
                emit::acc_add_out();
            }
        }
        let result = read_out(REG_ACC);
        ACC_LENT.store(false, Ordering::Relaxed);
        result
    }

