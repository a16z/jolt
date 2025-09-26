// Small Value Optimization (SVO) helpers for Spartan first sum-check

use crate::field::JoltField;

// Accumulation primitives for SVO (moved from zkvm/r1cs/types.rs)
pub mod accum {
    use crate::field::{FmaddTrunc, FromLimbs, FromS224, JoltField};
    use ark_ff::biginteger::{I8OrI96, S160, S224};

    /// Final unreduced product after multiplying by a 256-bit field element (512-bit unsigned)
    pub type UnreducedProduct<F> = <F as JoltField>::Unreduced<8>;

    /// Fused multiply-add into unreduced accumulators.
    #[inline(always)]
    pub fn fmadd_unreduced<F: JoltField>(
        pos_acc: &mut UnreducedProduct<F>,
        neg_acc: &mut UnreducedProduct<F>,
        field: &F,
        product: S224,
    ) {
        let field_bigint = field.as_unreduced_ref();
        if !product.is_zero() {
            let mag = F::Unreduced::<4>::from_s224(product);
            let acc = if product.is_positive() {
                pos_acc
            } else {
                neg_acc
            };
            field_bigint.fmadd_trunc::<4, 8>(&mag, acc);
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SignedUnreducedAccum<F: JoltField> {
        pub pos: UnreducedProduct<F>,
        pub neg: UnreducedProduct<F>,
    }

    impl<F: JoltField> Default for SignedUnreducedAccum<F> {
        fn default() -> Self {
            Self {
                pos: UnreducedProduct::<F>::default(),
                neg: UnreducedProduct::<F>::default(),
            }
        }
    }

    impl<F: JoltField> SignedUnreducedAccum<F> {
        #[inline(always)]
        pub fn new() -> Self {
            Self::default()
        }

        #[inline(always)]
        pub fn clear(&mut self) {
            self.pos = UnreducedProduct::<F>::default();
            self.neg = UnreducedProduct::<F>::default();
        }

        /// fmadd with an `I8OrI96` (signed, up to 2 limbs)
        #[inline(always)]
        pub fn fmadd_az(&mut self, field: &F, az: I8OrI96) {
            let field_bigint = field.as_unreduced_ref();
            let v = az.to_i128();
            if v != 0 {
                let abs = v.unsigned_abs();
                let mag = F::Unreduced::<2>::from(abs);
                let acc = if v >= 0 { &mut self.pos } else { &mut self.neg };
                field_bigint.fmadd_trunc::<2, 8>(&mag, acc);
            }
        }

        /// fmadd with a `S160` (signed, up to 3 limbs)
        #[inline(always)]
        pub fn fmadd_bz(&mut self, field: &F, bz: S160) {
            let field_bigint = field.as_unreduced_ref();
            if !bz.is_zero() {
                let lo = bz.magnitude_lo();
                let hi = bz.magnitude_hi() as u64;
                let mag = F::Unreduced::from_limbs([lo[0], lo[1], hi]);
                let acc = if bz.is_positive() {
                    &mut self.pos
                } else {
                    &mut self.neg
                };
                field_bigint.fmadd_trunc::<3, 8>(&mag, acc);
            }
        }

        /// fmadd with an Az×Bz product value (1..=4 limbs)
        #[inline(always)]
        pub fn fmadd_prod(&mut self, field: &F, product: S224) {
            fmadd_unreduced(&mut self.pos, &mut self.neg, field, product)
        }

        /// Reduce accumulated value to a field element (pos - neg)
        #[inline(always)]
        pub fn reduce_to_field(&self) -> F {
            F::from_montgomery_reduce(self.pos) - F::from_montgomery_reduce(self.neg)
        }
    }

    /// Local helper to convert `S160` to field without using `.to_field()`
    /// Used for testing purpose only
    #[cfg(test)]
    pub fn s160_to_field<F: JoltField>(bz: &S160) -> F {
        if bz.is_zero() {
            return F::zero();
        }
        let lo = bz.magnitude_lo();
        let hi = bz.magnitude_hi() as u64;
        let r64 = F::from_u128(1u128 << 64);
        let r128 = r64 * r64;
        let acc = F::from_u64(lo[0]) + r64.mul_u64(lo[1]) + r128.mul_u64(hi);
        if bz.is_positive() {
            acc
        } else {
            -acc
        }
    }
}

pub mod svo_helpers {
    use super::accum::{fmadd_unreduced, UnreducedProduct};
    use super::*;
    use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
    use crate::poly::unipoly::CompressedUniPoly;
    use crate::subprotocols::sumcheck::process_eq_sumcheck_round;
    use crate::transcripts::Transcript;
    use ark_ff::biginteger::{I8OrI96, S160};

    // SVOEvalPoint enum definition
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum SVOEvalPoint {
        Zero,
        One,
        Infinity,
    }

    #[inline]
    pub const fn pow(base: usize, exp: usize) -> usize {
        let mut res = 1;
        let mut i = 0;
        while i < exp {
            res *= base;
            i += 1;
        }
        res
    }

    pub const fn num_non_trivial_ternary_points(num_svo_rounds: usize) -> usize {
        // Returns 3^num_svo_rounds - 2^num_svo_rounds
        // This is equivalent to num_non_binary_points
        num_non_binary_points(num_svo_rounds)
    }

    pub const fn total_num_accums(num_svo_rounds: usize) -> usize {
        // Compute the sum \sum_{i=1}^{num_svo_rounds} (3^i - 2^i)
        // Note: original loop was 1 to num_svo_rounds inclusive.
        // num_non_binary_points(i) is 3^i - 2^i
        let mut sum = 0;
        let mut i = 1;
        while i <= num_svo_rounds {
            // Original was i <= num_svo_rounds
            sum += num_non_binary_points(i);
            i += 1;
        }
        sum
    }

    pub const fn num_accums_eval_zero(num_svo_rounds: usize) -> usize {
        // Returns \sum_{i=0}^{num_svo_rounds - 1} (3^i - 2^i)
        let mut sum = 0;
        let mut i = 0;
        while i < num_svo_rounds {
            sum += num_non_binary_points(i);
            i += 1;
        }
        sum
    }

    pub const fn num_accums_eval_infty(num_svo_rounds: usize) -> usize {
        // Returns \sum_{i=0}^{num_svo_rounds - 1} 3^i
        let mut sum = 0;
        let mut i = 0;
        while i < num_svo_rounds {
            sum += pow(3, i);
            i += 1;
        }
        sum
    }

    pub const fn num_non_binary_points(n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        pow(3, n) - pow(2, n)
    }

    pub const fn k_to_y_ext_msb<const N: usize>(
        k_ternary: usize,
    ) -> ([SVOEvalPoint; N], bool /*is_binary*/) {
        let mut coords = [SVOEvalPoint::Zero; N];
        let mut temp_k = k_ternary;
        let mut is_binary_flag = true;

        if N == 0 {
            return (coords, is_binary_flag);
        }

        let mut i_rev = 0;
        while i_rev < N {
            let i_dim_msb = N - 1 - i_rev;
            let digit = temp_k % 3;
            coords[i_dim_msb] = if digit == 0 {
                SVOEvalPoint::Zero
            } else if digit == 1 {
                SVOEvalPoint::One
            } else {
                is_binary_flag = false;
                SVOEvalPoint::Infinity
            };
            temp_k /= 3;
            i_rev += 1;
        }
        (coords, is_binary_flag)
    }

    pub const fn build_y_ext_code_map<const N: usize, const M: usize>() -> [[SVOEvalPoint; N]; M] {
        let mut map = [[SVOEvalPoint::Zero; N]; M];
        let mut current_map_idx = 0;

        if N == 0 {
            return map;
        }

        let num_total_ternary_points = pow(3, N);
        let mut k_ternary_idx = 0;
        while k_ternary_idx < num_total_ternary_points {
            let (coords_msb, is_binary) = k_to_y_ext_msb::<N>(k_ternary_idx);
            if !is_binary && current_map_idx < M {
                map[current_map_idx] = coords_msb;
                current_map_idx += 1;
            }
            k_ternary_idx += 1;
        }
        map
    }

    /// Returns the list of global ternary indices `k` for all non-binary points,
    /// in the same deterministic MSB-first order as `build_y_ext_code_map`.
    /// Size `M` must be exactly `num_non_binary_points(N)`.
    pub const fn build_non_binary_k_list<const N: usize, const M: usize>() -> [usize; M] {
        let mut out = [0usize; M];
        let mut out_idx = 0usize;

        if N == 0 {
            return out;
        }

        let num_total_ternary_points = pow(3, N);
        let mut k = 0usize;
        while k < num_total_ternary_points {
            // Check if any ternary digit (base-3) equals 2 (Infinity)
            let mut temp_k = k;
            let mut is_non_binary = false;
            let mut i = 0usize;
            while i < N {
                if temp_k % 3 == 2 {
                    is_non_binary = true;
                    break;
                }
                temp_k /= 3;
                i += 1;
            }

            if is_non_binary && out_idx < M {
                out[out_idx] = k;
                out_idx += 1;
            }

            k += 1;
        }
        out
    }

    pub const fn v_coords_to_base3_idx(v_coords_lsb_y_ext_order: &[SVOEvalPoint]) -> usize {
        let mut idx = 0;
        let mut current_power_of_3 = 1;
        let mut i = 0;
        while i < v_coords_lsb_y_ext_order.len() {
            let point_val = match v_coords_lsb_y_ext_order[i] {
                SVOEvalPoint::Zero => 0,
                SVOEvalPoint::One => 1,
                SVOEvalPoint::Infinity => 2,
            };
            idx += point_val * current_power_of_3;
            if i < v_coords_lsb_y_ext_order.len() - 1 {
                current_power_of_3 *= 3;
            }
            i += 1;
        }
        idx
    }

    pub const fn v_coords_has_infinity(v_coords_lsb_y_ext_order: &[SVOEvalPoint]) -> bool {
        let mut i = 0;
        while i < v_coords_lsb_y_ext_order.len() {
            if matches!(v_coords_lsb_y_ext_order[i], SVOEvalPoint::Infinity) {
                return true;
            }
            i += 1;
        }
        false
    }

    pub const fn v_coords_to_non_binary_base3_idx(
        v_coords_lsb_y_ext_order: &[SVOEvalPoint],
    ) -> usize {
        let num_v_vars = v_coords_lsb_y_ext_order.len();
        if num_v_vars == 0 {
            return 0;
        }

        let mut index_count = 0;
        let target_v_config_ternary_value = v_coords_to_base3_idx(v_coords_lsb_y_ext_order);

        let max_k_v = pow(3, num_v_vars);
        let mut k_v_ternary = 0;
        while k_v_ternary < max_k_v {
            if k_v_ternary == target_v_config_ternary_value {
                break;
            }

            let mut current_k_v_has_inf = false;
            let mut temp_k = k_v_ternary;
            let mut var_idx = 0;
            while var_idx < num_v_vars {
                if temp_k % 3 == 2 {
                    current_k_v_has_inf = true;
                    break;
                }
                temp_k /= 3;
                var_idx += 1;
            }

            if current_k_v_has_inf {
                index_count += 1;
            }
            k_v_ternary += 1;
        }
        index_count
    }

    pub const fn precompute_accumulator_offsets<const N: usize>() -> ([usize; N], [usize; N]) {
        let mut offsets_infty = [0; N];
        let mut offsets_zero = [0; N];
        if N == 0 {
            return (offsets_infty, offsets_zero);
        }

        let mut current_sum_infty = 0;
        let mut current_sum_zero = 0;
        let mut s_p = 0;
        while s_p < N {
            offsets_infty[s_p] = current_sum_infty;
            offsets_zero[s_p] = current_sum_zero;

            current_sum_infty += pow(3, s_p);
            current_sum_zero += pow(3, s_p) - pow(2, s_p);
            s_p += 1;
        }
        (offsets_infty, offsets_zero)
    }

    #[inline]
    fn get_v_config_digits(mut k_global: usize, num_vars: usize) -> Vec<usize> {
        if num_vars == 0 {
            return vec![];
        }
        let mut digits = vec![0; num_vars];
        let mut i = num_vars;
        while i > 0 {
            // Fill from LSB of digits vec, which corresponds to LSB of k_global
            digits[i - 1] = k_global % 3;
            k_global /= 3;
            i -= 1;
        }
        digits
    }

    #[inline]
    fn is_v_config_non_binary(v_config: &[usize]) -> bool {
        v_config.contains(&2)
    }

    /// Converts MSB-first SVOEvalPoint coordinates to their global ternary index k.
    /// k = sum_{i=0}^{N-1} val(coords_msb[i]) * 3^(N-1-i)
    pub const fn svo_coords_msb_to_k_ternary_idx<const N: usize>(
        coords_msb: &[SVOEvalPoint; N],
    ) -> usize {
        if N == 0 {
            return 0;
        }
        let mut k_val = 0;
        let mut i_msb_dim = 0; // iterates from 0 (MSB) to N-1 (LSB)
        while i_msb_dim < N {
            let digit_val = match coords_msb[i_msb_dim] {
                SVOEvalPoint::Zero => 0,
                SVOEvalPoint::One => 1,
                SVOEvalPoint::Infinity => 2,
            };
            let power_of_3_for_dim = pow(3, N - 1 - i_msb_dim);
            k_val += digit_val * power_of_3_for_dim;
            i_msb_dim += 1;
        }
        k_val
    }

    /// Generic version for arbitrary NUM_SVO_ROUNDS
    /// NOTE: testing shows about 2x slowdown over hard-coded versions for NUM_SVO_ROUNDS = 1, 2, 3
    /// We keep this for reference and testing
    #[inline]
    pub fn compute_and_update_tA_inplace_const<
        const NUM_SVO_ROUNDS: usize,
        const M_NON_BINARY_POINTS_CONST: usize, // num_non_binary_points(NUM_SVO_ROUNDS)
        const NUM_TERNARY_POINTS_CONST: usize,  // pow(3, NUM_SVO_ROUNDS)
        F: JoltField,
    >(
        binary_az_evals_input: &[I8OrI96], // 2^N Az at binary points
        binary_bz_evals_input: &[S160],    // 2^N Bz at binary points
        e_in_val: &F,
        tA_pos_acc: &mut [UnreducedProduct<F>],
        tA_neg_acc: &mut [UnreducedProduct<F>],
    ) {
        if NUM_SVO_ROUNDS == 0 {
            debug_assert!(binary_az_evals_input.is_empty());
            debug_assert!(binary_bz_evals_input.is_empty());
            debug_assert!(tA_pos_acc.is_empty());
            debug_assert!(tA_neg_acc.is_empty());
            return;
        }

        // Sanity: consts must match the chosen NUM_SVO_ROUNDS
        debug_assert_eq!(
            M_NON_BINARY_POINTS_CONST,
            num_non_binary_points(NUM_SVO_ROUNDS)
        );
        debug_assert_eq!(NUM_TERNARY_POINTS_CONST, pow(3, NUM_SVO_ROUNDS));

        let num_binary_points = 1usize << NUM_SVO_ROUNDS;
        debug_assert_eq!(binary_az_evals_input.len(), num_binary_points);
        debug_assert_eq!(binary_bz_evals_input.len(), num_binary_points);
        debug_assert_eq!(tA_pos_acc.len(), M_NON_BINARY_POINTS_CONST);
        debug_assert_eq!(tA_neg_acc.len(), M_NON_BINARY_POINTS_CONST);

        // Precompute ternary point info (const-evaluable, but bind as local)
        let ternary_point_info_table: [TernaryPointInfo<NUM_SVO_ROUNDS>; NUM_TERNARY_POINTS_CONST] =
            precompute_ternary_point_infos::<NUM_SVO_ROUNDS, NUM_TERNARY_POINTS_CONST>();

        // Memo tables (const-size arrays)
        let mut memoized_az_evals: [Option<I8OrI96>; NUM_TERNARY_POINTS_CONST] =
            [None; NUM_TERNARY_POINTS_CONST];
        let mut memoized_bz_evals: [Option<S160>; NUM_TERNARY_POINTS_CONST] =
            [None; NUM_TERNARY_POINTS_CONST];

        // Recursive helper: Az extension at ternary index k with memoization
        #[inline]
        fn get_az_ext_const<const N: usize, const NUM_TERN: usize>(
            k: usize,
            bin: &[I8OrI96],
            memo: &mut [Option<I8OrI96>; NUM_TERN],
            info: &[TernaryPointInfo<N>; NUM_TERN],
        ) -> I8OrI96 {
            if let Some(v) = memo[k] {
                return v;
            }
            let out = if info[k].is_binary {
                bin[info[k].binary_eval_idx]
            } else {
                let e1 = get_az_ext_const::<N, NUM_TERN>(info[k].k_val_at_one, bin, memo, info);
                let e0 = get_az_ext_const::<N, NUM_TERN>(info[k].k_val_at_zero, bin, memo, info);
                e1 - e0
            };
            memo[k] = Some(out);
            out
        }

        // Recursive helper: Bz extension at ternary index k with memoization
        #[inline]
        fn get_bz_ext_const<const N: usize, const NUM_TERN: usize>(
            k: usize,
            bin: &[S160],
            memo: &mut [Option<S160>; NUM_TERN],
            info: &[TernaryPointInfo<N>; NUM_TERN],
        ) -> S160 {
            if let Some(v) = memo[k] {
                return v;
            }
            let out = if info[k].is_binary {
                bin[info[k].binary_eval_idx]
            } else {
                let e1 = get_bz_ext_const::<N, NUM_TERN>(info[k].k_val_at_one, bin, memo, info);
                let e0 = get_bz_ext_const::<N, NUM_TERN>(info[k].k_val_at_zero, bin, memo, info);
                e1 - e0
            };
            memo[k] = Some(out);
            out
        }

        // Iterate non-binary points via their global k indices in deterministic MSB-first order
        let non_binary_k_list: [usize; M_NON_BINARY_POINTS_CONST] =
            build_non_binary_k_list::<NUM_SVO_ROUNDS, M_NON_BINARY_POINTS_CONST>();

        for i_temp_tA in 0..M_NON_BINARY_POINTS_CONST {
            let k_target = non_binary_k_list[i_temp_tA];

            debug_assert!(k_target < NUM_TERNARY_POINTS_CONST);
            debug_assert!(!ternary_point_info_table[k_target].is_binary);

            let az_ext = get_az_ext_const::<NUM_SVO_ROUNDS, NUM_TERNARY_POINTS_CONST>(
                k_target,
                binary_az_evals_input,
                &mut memoized_az_evals,
                &ternary_point_info_table,
            );

            // `az_ext` is likely to be zero
            if az_ext.is_zero() {
                continue;
            }

            let bz_ext = get_bz_ext_const::<NUM_SVO_ROUNDS, NUM_TERNARY_POINTS_CONST>(
                k_target,
                binary_bz_evals_input,
                &mut memoized_bz_evals,
                &ternary_point_info_table,
            );

            // No need for this - `bz_ext` is less likely to be zero than `az_ext`
            // if bz_ext.is_zero() { continue; }

            let prod = az_ext * bz_ext;
            fmadd_unreduced::<F>(
                &mut tA_pos_acc[i_temp_tA],
                &mut tA_neg_acc[i_temp_tA],
                e_in_val,
                prod,
            );
        }
    }

    /// Specialized small-value kernel when `NUM_SVO_ROUNDS == 2`.
    /// Updates 5 non-binary accumulators corresponding to:
    /// (0,I), (1,I), (I,0), (I,1), (I,I) in this order.
    #[inline]
    pub fn compute_and_update_tA_inplace_2<F: JoltField>(
        binary_az_evals: &[I8OrI96],
        binary_bz_evals: &[S160],
        e_in_val: &F,
        tA_pos_acc: &mut [UnreducedProduct<F>],
        tA_neg_acc: &mut [UnreducedProduct<F>],
    ) {
        debug_assert!(binary_az_evals.len() == 4);
        debug_assert!(binary_bz_evals.len() == 4);
        debug_assert!(tA_pos_acc.len() == 5 && tA_neg_acc.len() == 5);

        // Binary evaluations (Y0,Y1) -> index Y0*2 + Y1
        let az00 = binary_az_evals[0];
        let bz00 = binary_bz_evals[0];
        let az01 = binary_az_evals[1];
        let bz01 = binary_bz_evals[1];
        let az10 = binary_az_evals[2];
        let bz10 = binary_bz_evals[2];
        let az11 = binary_az_evals[3];
        let bz11 = binary_bz_evals[3];

        // Precompute first-order diffs
        let az_0i = az01 - az00;
        let bz_0i = bz01 - bz00;
        let az_1i = az11 - az10;
        let bz_1i = bz11 - bz10;
        let az_i0 = az10 - az00;
        let bz_i0 = bz10 - bz00;
        let az_i1 = az11 - az01;
        let bz_i1 = bz11 - bz01;

        // 1. (0,I) -> tA[0]
        if !az_0i.is_zero() && !bz_0i.is_zero() {
            let prod = az_0i * bz_0i;
            fmadd_unreduced::<F>(&mut tA_pos_acc[0], &mut tA_neg_acc[0], e_in_val, prod);
        }

        // 2. (1,I) -> tA[1]
        if !az_1i.is_zero() && !bz_1i.is_zero() {
            let prod = az_1i * bz_1i;
            fmadd_unreduced::<F>(&mut tA_pos_acc[1], &mut tA_neg_acc[1], e_in_val, prod);
        }

        // 3. (I,0) -> tA[2]
        if !az_i0.is_zero() && !bz_i0.is_zero() {
            let prod = az_i0 * bz_i0;
            fmadd_unreduced::<F>(&mut tA_pos_acc[2], &mut tA_neg_acc[2], e_in_val, prod);
        }

        // 4. (I,1) -> tA[3]
        if !az_i1.is_zero() && !bz_i1.is_zero() {
            let prod = az_i1 * bz_i1;
            fmadd_unreduced::<F>(&mut tA_pos_acc[3], &mut tA_neg_acc[3], e_in_val, prod);
        }

        // 5. (I,I) -> tA[4]
        let az_ii = az_1i - az_0i;
        let bz_ii = bz_1i - bz_0i;
        if !az_ii.is_zero() && !bz_ii.is_zero() {
            let prod = az_ii * bz_ii;
            fmadd_unreduced::<F>(&mut tA_pos_acc[4], &mut tA_neg_acc[4], e_in_val, prod);
        }
    }

    /// Specialized small-value kernel when `NUM_SVO_ROUNDS == 3`.
    /// Updates 19 non-binary accumulators in the same order as the field version.
    #[inline]
    pub fn compute_and_update_tA_inplace_3<F: JoltField>(
        binary_az_evals: &[I8OrI96],
        binary_bz_evals: &[S160],
        e_in_val: &F,
        tA_pos_acc: &mut [UnreducedProduct<F>],
        tA_neg_acc: &mut [UnreducedProduct<F>],
    ) {
        debug_assert!(binary_az_evals.len() == 8);
        debug_assert!(binary_bz_evals.len() == 8);
        debug_assert!(tA_pos_acc.len() == 19 && tA_neg_acc.len() == 19);

        // Binary evaluations (Y0,Y1,Y2) -> index Y0*4 + Y1*2 + Y2
        let az000 = binary_az_evals[0];
        let bz000 = binary_bz_evals[0];
        let az001 = binary_az_evals[1];
        let bz001 = binary_bz_evals[1];
        let az010 = binary_az_evals[2];
        let bz010 = binary_bz_evals[2];
        let az011 = binary_az_evals[3];
        let bz011 = binary_bz_evals[3];
        let az100 = binary_az_evals[4];
        let bz100 = binary_bz_evals[4];
        let az101 = binary_az_evals[5];
        let bz101 = binary_bz_evals[5];
        let az110 = binary_az_evals[6];
        let bz110 = binary_bz_evals[6];
        let az111 = binary_az_evals[7];
        let bz111 = binary_bz_evals[7];

        // Precompute diffs used multiple times
        let az_00i = az001 - az000;
        let bz_00i = bz001 - bz000;
        if !az_00i.is_zero() && !bz_00i.is_zero() {
            let prod = az_00i * bz_00i;
            fmadd_unreduced::<F>(&mut tA_pos_acc[0], &mut tA_neg_acc[0], e_in_val, prod);
        }

        let az_01i = az011 - az010;
        let bz_01i = bz011 - bz010;
        if !az_01i.is_zero() && !bz_01i.is_zero() {
            let prod = az_01i * bz_01i;
            fmadd_unreduced::<F>(&mut tA_pos_acc[1], &mut tA_neg_acc[1], e_in_val, prod);
        }

        let az_0i0 = az010 - az000;
        let bz_0i0 = bz010 - bz000;
        if !az_0i0.is_zero() && !bz_0i0.is_zero() {
            let prod = az_0i0 * bz_0i0;
            fmadd_unreduced::<F>(&mut tA_pos_acc[2], &mut tA_neg_acc[2], e_in_val, prod);
        }

        let az_0i1 = az011 - az001;
        let bz_0i1 = bz011 - bz001;
        if !az_0i1.is_zero() && !bz_0i1.is_zero() {
            let prod = az_0i1 * bz_0i1;
            fmadd_unreduced::<F>(&mut tA_pos_acc[3], &mut tA_neg_acc[3], e_in_val, prod);
        }

        let az_0ii = az_01i - az_00i;
        let bz_0ii = bz_01i - bz_00i;
        if !az_0ii.is_zero() && !bz_0ii.is_zero() {
            let prod = az_0ii * bz_0ii;
            fmadd_unreduced::<F>(&mut tA_pos_acc[4], &mut tA_neg_acc[4], e_in_val, prod);
        }

        let az_10i = az101 - az100;
        let bz_10i = bz101 - bz100;
        if !az_10i.is_zero() && !bz_10i.is_zero() {
            let prod = az_10i * bz_10i;
            fmadd_unreduced::<F>(&mut tA_pos_acc[5], &mut tA_neg_acc[5], e_in_val, prod);
        }

        let az_11i = az111 - az110;
        let bz_11i = bz111 - bz110;
        if !az_11i.is_zero() && !bz_11i.is_zero() {
            let prod = az_11i * bz_11i;
            fmadd_unreduced::<F>(&mut tA_pos_acc[6], &mut tA_neg_acc[6], e_in_val, prod);
        }

        let az_1i0 = az110 - az100;
        let bz_1i0 = bz110 - bz100;
        if !az_1i0.is_zero() && !bz_1i0.is_zero() {
            let prod = az_1i0 * bz_1i0;
            fmadd_unreduced::<F>(&mut tA_pos_acc[7], &mut tA_neg_acc[7], e_in_val, prod);
        }

        let az_1i1 = az111 - az101;
        let bz_1i1 = bz111 - bz101;
        if !az_1i1.is_zero() && !bz_1i1.is_zero() {
            let prod = az_1i1 * bz_1i1;
            fmadd_unreduced::<F>(&mut tA_pos_acc[8], &mut tA_neg_acc[8], e_in_val, prod);
        }

        let az_1ii = az_11i - az_10i;
        let bz_1ii = bz_11i - bz_10i;
        if !az_1ii.is_zero() && !bz_1ii.is_zero() {
            let prod = az_1ii * bz_1ii;
            fmadd_unreduced::<F>(&mut tA_pos_acc[9], &mut tA_neg_acc[9], e_in_val, prod);
        }

        let az_i00 = az100 - az000;
        let bz_i00 = bz100 - bz000;
        if !az_i00.is_zero() && !bz_i00.is_zero() {
            let prod = az_i00 * bz_i00;
            fmadd_unreduced::<F>(&mut tA_pos_acc[10], &mut tA_neg_acc[10], e_in_val, prod);
        }

        let az_i01 = az101 - az001;
        let bz_i01 = bz101 - bz001;
        if !az_i01.is_zero() && !bz_i01.is_zero() {
            let prod = az_i01 * bz_i01;
            fmadd_unreduced::<F>(&mut tA_pos_acc[11], &mut tA_neg_acc[11], e_in_val, prod);
        }

        let az_i0i = az_i01 - az_i00;
        let bz_i0i = bz_i01 - bz_i00;
        if !az_i0i.is_zero() && !bz_i0i.is_zero() {
            let prod = az_i0i * bz_i0i;
            fmadd_unreduced::<F>(&mut tA_pos_acc[12], &mut tA_neg_acc[12], e_in_val, prod);
        }

        let az_i10 = az110 - az010;
        let bz_i10 = bz110 - bz010;
        if !az_i10.is_zero() && !bz_i10.is_zero() {
            let prod = az_i10 * bz_i10;
            fmadd_unreduced::<F>(&mut tA_pos_acc[13], &mut tA_neg_acc[13], e_in_val, prod);
        }

        let az_i11 = az111 - az011;
        let bz_i11 = bz111 - bz011;
        if !az_i11.is_zero() && !bz_i11.is_zero() {
            let prod = az_i11 * bz_i11;
            fmadd_unreduced::<F>(&mut tA_pos_acc[14], &mut tA_neg_acc[14], e_in_val, prod);
        }

        let az_i1i = az_i11 - az_i10;
        let bz_i1i = bz_i11 - bz_i10;
        if !az_i1i.is_zero() && !bz_i1i.is_zero() {
            let prod = az_i1i * bz_i1i;
            fmadd_unreduced::<F>(&mut tA_pos_acc[15], &mut tA_neg_acc[15], e_in_val, prod);
        }

        let az_ii0 = az_1i0 - az_0i0;
        let bz_ii0 = bz_1i0 - bz_0i0;
        if !az_ii0.is_zero() && !bz_ii0.is_zero() {
            let prod = az_ii0 * bz_ii0;
            fmadd_unreduced::<F>(&mut tA_pos_acc[16], &mut tA_neg_acc[16], e_in_val, prod);
        }

        let az_ii1 = az_1i1 - az_0i1;
        let bz_ii1 = bz_1i1 - bz_0i1;
        if !az_ii1.is_zero() && !bz_ii1.is_zero() {
            let prod = az_ii1 * bz_ii1;
            fmadd_unreduced::<F>(&mut tA_pos_acc[17], &mut tA_neg_acc[17], e_in_val, prod);
        }

        let az_iii = az_1ii - az_0ii;
        let bz_iii = bz_1ii - bz_0ii;
        if !az_iii.is_zero() && !bz_iii.is_zero() {
            let prod = az_iii * bz_iii;
            fmadd_unreduced::<F>(&mut tA_pos_acc[18], &mut tA_neg_acc[18], e_in_val, prod);
        }
    }

    /// Dispatch wrapper for the typed small-value kernel using const generics.
    /// Supports NUM_SVO_ROUNDS in [0..=5].
    #[inline]
    pub fn compute_and_update_tA_inplace<const NUM_SVO_ROUNDS: usize, F: JoltField>(
        binary_az_evals: &[I8OrI96],
        binary_bz_evals: &[S160],
        e_in_val: &F,
        tA_pos_acc: &mut [UnreducedProduct<F>],
        tA_neg_acc: &mut [UnreducedProduct<F>],
    ) {
        match NUM_SVO_ROUNDS {
            0 => {
                debug_assert!(binary_az_evals.is_empty());
                debug_assert!(binary_bz_evals.is_empty());
                debug_assert!(tA_pos_acc.is_empty());
                debug_assert!(tA_neg_acc.is_empty());
            }
            1 => compute_and_update_tA_inplace_const::<1, 1, 3, F>(
                binary_az_evals,
                binary_bz_evals,
                e_in_val,
                tA_pos_acc,
                tA_neg_acc,
            ),
            2 => compute_and_update_tA_inplace_2::<F>(
                binary_az_evals,
                binary_bz_evals,
                e_in_val,
                tA_pos_acc,
                tA_neg_acc,
            ),
            3 => compute_and_update_tA_inplace_3::<F>(
                binary_az_evals,
                binary_bz_evals,
                e_in_val,
                tA_pos_acc,
                tA_neg_acc,
            ),
            4 => compute_and_update_tA_inplace_const::<4, 65, 81, F>(
                binary_az_evals,
                binary_bz_evals,
                e_in_val,
                tA_pos_acc,
                tA_neg_acc,
            ),
            5 => compute_and_update_tA_inplace_const::<5, 211, 243, F>(
                binary_az_evals,
                binary_bz_evals,
                e_in_val,
                tA_pos_acc,
                tA_neg_acc,
            ),
            _ => panic!("Unsupported NUM_SVO_ROUNDS: {NUM_SVO_ROUNDS}"),
        }
    }

    /// Specialized small-value kernel when `NUM_SVO_ROUNDS == 1`.
    /// Updates a single non-binary accumulator corresponding to (u=∞).
    #[inline]
    pub fn compute_and_update_tA_inplace_1<F: JoltField>(
        binary_az_evals: &[I8OrI96],
        binary_bz_evals: &[S160],
        e_in_val: &F,
        tA_pos_acc: &mut [UnreducedProduct<F>],
        tA_neg_acc: &mut [UnreducedProduct<F>],
    ) {
        debug_assert!(binary_az_evals.len() == 2);
        debug_assert!(binary_bz_evals.len() == 2);
        debug_assert!(tA_pos_acc.len() == 1);
        debug_assert!(tA_neg_acc.len() == 1);

        let az_I = binary_az_evals[1] - binary_az_evals[0];
        if !az_I.is_zero() {
            let bz_I = binary_bz_evals[1] - binary_bz_evals[0];
            if !bz_I.is_zero() {
                let prod = az_I * bz_I;
                fmadd_unreduced::<F>(&mut tA_pos_acc[0], &mut tA_neg_acc[0], e_in_val, prod);
            }
        }
    }

    /// Generic version for distributing tA to svo accumulators
    pub fn distribute_tA_to_svo_accumulators_generic<const NUM_SVO_ROUNDS: usize, F: JoltField>(
        tA_accums: &[F],
        x_out_val: usize,
        E_out_vec: &[Vec<F>],
        accums_zero: &mut [F],
        accums_infty: &mut [F],
    ) {
        match NUM_SVO_ROUNDS {
            0 => {
                // No SVO rounds to process, tA_accums should be empty
                debug_assert!(
                    tA_accums.is_empty(),
                    "tA_accums should be empty for 0 SVO rounds"
                );
            }
            1 => distribute_tA_to_svo_accumulators_1(
                tA_accums,
                x_out_val,
                E_out_vec,
                accums_zero,
                accums_infty,
            ),
            2 => distribute_tA_to_svo_accumulators_2(
                tA_accums,
                x_out_val,
                E_out_vec,
                accums_zero,
                accums_infty,
            ),
            3 => distribute_tA_to_svo_accumulators_3(
                tA_accums,
                x_out_val,
                E_out_vec,
                accums_zero,
                accums_infty,
            ),
            4 => {
                // 81 - 16 = 65 non-binary points
                distribute_tA_to_svo_accumulators::<4, 65, F>(
                    tA_accums,
                    x_out_val,
                    E_out_vec,
                    accums_zero,
                    accums_infty,
                )
            }
            5 => {
                // 243 - 32 = 211 non-binary points
                distribute_tA_to_svo_accumulators::<5, 211, F>(
                    tA_accums,
                    x_out_val,
                    E_out_vec,
                    accums_zero,
                    accums_infty,
                )
            }
            _ => unreachable!("You should not try this many rounds of SVO"),
        }
    }

    /// Hardcoded version for `num_svo_rounds == 1`
    /// We have only one non-binary point (Y0=I), mapping to accum_1(I)
    #[inline]
    pub fn distribute_tA_to_svo_accumulators_1<F: JoltField>(
        tA_accums: &[F],
        x_out_val: usize,
        E_out_vec: &[Vec<F>],
        _accums_zero: &mut [F],
        accums_infty: &mut [F],
    ) {
        debug_assert!(_accums_zero.is_empty());
        debug_assert!(accums_infty.len() == 1);
        debug_assert!(tA_accums.len() == 1);

        accums_infty[0] += E_out_vec[0][x_out_val] * tA_accums[0];
    }

    /// Hardcoded version for `num_svo_rounds == 2`
    /// We have 5 non-binary points with their corresponding mappings: (recall, LSB is rightmost)
    /// tA_accums indices:
    ///   [0]: Y_ext = (0,I)
    ///   [1]: Y_ext = (1,I)
    ///   [2]: Y_ext = (I,0)
    ///   [3]: Y_ext = (I,1)
    ///   [4]: Y_ext = (I,I)
    ///
    /// Target flat accumulators (NUM_SVO_ROUNDS = 2):
    ///   accums_zero (len 1): [ A_1(0,I) ]
    ///   accums_infty (len 4): [ A_0(empty,I), A_1(I,0), A_1(I,1), A_1(I,I) ]
    /// Note that A_0(empty,I) should receive contributions from (0,I) and (1,I).
    #[inline]
    pub fn distribute_tA_to_svo_accumulators_2<F: JoltField>(
        tA_accums: &[F],
        x_out_val: usize,
        E_out_vec: &[Vec<F>],
        accums_zero: &mut [F],
        accums_infty: &mut [F],
    ) {
        debug_assert!(tA_accums.len() == 5);
        debug_assert!(accums_zero.len() == 1);
        debug_assert!(accums_infty.len() == 4);
        debug_assert!(E_out_vec.len() >= 2);

        let E0_y0 = E_out_vec[0][x_out_val << 1];
        let E0_y1 = E_out_vec[0][(x_out_val << 1) | 1];
        let E1_yempty = E_out_vec[1][x_out_val];

        // Y_ext = (0,I) -> tA_accums[0]
        // Contributes to A_0(empty,I) (i.e. accums_infty[0]) via E_out_0[x_out_val | 0] and
        // A_1(0,I) (i.e. accums_zero[0]) via E_out_1[x_out_val]

        accums_infty[0] += E0_y0 * tA_accums[0];

        accums_zero[0] += E1_yempty * tA_accums[0];

        // Y_ext = (1,I) -> tA_accums[1]
        // Contributes to A_0(empty,I) (i.e. accums_infty[0]) via E_out_0[x_out_val | 1]
        // (no term A_1(1,I) as it is not needed i.e. it's an eval at 1)
        accums_infty[0] += E0_y1 * tA_accums[1];

        // Y_ext = (I,0) -> tA_accums[2]
        // Contributes to A_1(I,0) (i.e. accums_infty[1]) via E_out_1[x_out_val]
        accums_infty[1] += E1_yempty * tA_accums[2];

        // Y_ext = (I,1) -> tA_accums[3]
        // Contributes to A_1(I,1) (i.e. accums_infty[2]) via E_out_1[x_out_val]
        accums_infty[2] += E1_yempty * tA_accums[3];

        // Y_ext = (I,I) -> tA_accums[4]
        // Contributes to A_1(I,I) (i.e. accums_infty[3]) via E_out_1[x_out_val]
        accums_infty[3] += E1_yempty * tA_accums[4];
    }

    /// Hardcoded version for `num_svo_rounds == 3`
    ///
    #[inline]
    pub fn distribute_tA_to_svo_accumulators_3<F: JoltField>(
        tA_accums: &[F],
        x_out_val: usize,
        E_out_vec: &[Vec<F>],
        accums_zero: &mut [F],
        accums_infty: &mut [F],
    ) {
        debug_assert!(tA_accums.len() == 19);
        debug_assert!(accums_zero.len() == 6);
        debug_assert!(accums_infty.len() == 13);
        debug_assert!(E_out_vec.len() >= 3);

        use SVOEvalPoint::{Infinity, One, Zero};

        // Accumulator slots (conceptual LSB-first paper round s_p)
        // Nomenclature: A_{s_p}(v_{s_p-1},...,v_0, u_s_p)
        // accums_infty slots
        const ACCUM_IDX_A0_I: usize = 0; // A_0(I)
        const ACCUM_IDX_A1_0_I: usize = 1; // A_1(v0=0, I)
        const ACCUM_IDX_A1_1_I: usize = 2; // A_1(v0=1, I)
        const ACCUM_IDX_A1_I_I: usize = 3; // A_1(v0=I, I)
        const ACCUM_IDX_A2_00_I: usize = 4; // A_2(v1=0,v0=0,I)
        const ACCUM_IDX_A2_01_I: usize = 5; // A_2(v1=0,v0=1,I)
        const ACCUM_IDX_A2_0I_I: usize = 6; // A_2(v1=0,v0=I,I)
        const ACCUM_IDX_A2_10_I: usize = 7; // A_2(v1=1,v0=0,I)
        const ACCUM_IDX_A2_11_I: usize = 8; // A_2(v1=1,v0=1,I)
        const ACCUM_IDX_A2_1I_I: usize = 9; // A_2(v1=1,v0=I,I)
        const ACCUM_IDX_A2_I0_I: usize = 10; // A_2(v1=I,v0=0,I)
        const ACCUM_IDX_A2_I1_I: usize = 11; // A_2(v1=I,v0=1,I)
        const ACCUM_IDX_A2_II_I: usize = 12; // A_2(v1=I,v0=I,I)

        // accums_zero slots
        const ACCUM_IDX_A1_I_0: usize = 0; // A_1(v0=I, 0)
        const ACCUM_IDX_A2_0I_0: usize = 1; // A_2(v1=0,v0=I,0)
        const ACCUM_IDX_A2_1I_0: usize = 2; // A_2(v1=1,v0=I,0)
        const ACCUM_IDX_A2_I0_0: usize = 3; // A_2(v1=I,v0=0,0)
        const ACCUM_IDX_A2_I1_0: usize = 4; // A_2(v1=I,v0=1,0)
        const ACCUM_IDX_A2_II_0: usize = 5; // A_2(v1=I,v0=I,0)

        // E_out_vec[s_code] interpretation (s_code is MSB-first index for Y_c variables):
        // E_out_vec[0] (s_code=0 in E_out_vec): Y2_c is u_eff for E. y_suffix_eff=(Y0_c, Y1_c). Index (x_out << 2) | (Y0_c_bit << 1) | Y1_c_bit
        let e0_suf00 = E_out_vec[0][x_out_val << 2]; // y_suffix_eff=(Y0_c=0, Y1_c=0)
        let e0_suf01 = E_out_vec[0][(x_out_val << 2) | 0b01]; // y_suffix_eff=(Y0_c=0, Y1_c=1)
        let e0_suf10 = E_out_vec[0][(x_out_val << 2) | 0b10]; // y_suffix_eff=(Y0_c=1, Y1_c=0)
        let e0_suf11 = E_out_vec[0][(x_out_val << 2) | 0b11]; // y_suffix_eff=(Y0_c=1, Y1_c=1)

        // E_out_vec[1] (s_code=1 in E_out_vec): Y1_c is u_eff for E. y_suffix_eff=(Y0_c). Index (x_out << 1) | Y0_c_bit
        let e1_suf0 = E_out_vec[1][x_out_val << 1]; // y_suffix_eff=(Y0_c=0)
        let e1_suf1 = E_out_vec[1][(x_out_val << 1) | 1]; // y_suffix_eff=(Y0_c=1)

        // E_out_vec[2] (s_code=2 in E_out_vec): Y0_c is u_eff for E. y_suffix_eff=(). Index x_out_val
        let e2_sufempty = E_out_vec[2][x_out_val];

        // Y_EXT_CODE_MAP[tA_idx] gives (Y0_c, Y1_c, Y2_c) for tA_accums[tA_idx]
        // Y0_c is MSB, Y2_c is LSB for code variables. This order matches compute_and_update_tA_inplace_3.
        const Y_EXT_CODE_MAP: [(SVOEvalPoint, SVOEvalPoint, SVOEvalPoint); 19] = [
            (Zero, Zero, Infinity),
            (Zero, One, Infinity),
            (Zero, Infinity, Zero),
            (Zero, Infinity, One),
            (Zero, Infinity, Infinity),
            (One, Zero, Infinity),
            (One, One, Infinity),
            (One, Infinity, Zero),
            (One, Infinity, One),
            (One, Infinity, Infinity),
            (Infinity, Zero, Zero),
            (Infinity, Zero, One),
            (Infinity, Zero, Infinity),
            (Infinity, One, Zero),
            (Infinity, One, One),
            (Infinity, One, Infinity),
            (Infinity, Infinity, Zero),
            (Infinity, Infinity, One),
            (Infinity, Infinity, Infinity),
        ];

        for i in 0..19 {
            let current_tA = tA_accums[i];

            let (y0_c, y1_c, y2_c) = Y_EXT_CODE_MAP[i]; // (MSB, Mid, LSB) from code's perspective

            // --- Contributions to Paper Round s_p=0 Accumulators (u = Y2_c) ---
            // E-factor uses E_out_vec[0] (where Y2_c is u_eff), E_suffix is (Y0_c, Y1_c)
            if y2_c == Infinity {
                // u = I
                if y0_c != Infinity && y1_c != Infinity {
                    // Suffix (Y0_c,Y1_c) for E_out_vec[0] must be binary
                    let e_val = match (y0_c, y1_c) {
                        (Zero, Zero) => e0_suf00,
                        (Zero, One) => e0_suf01,
                        (One, Zero) => e0_suf10,
                        (One, One) => e0_suf11,
                        _ => unreachable!(), // Should be covered by binary check above
                    };
                    accums_infty[ACCUM_IDX_A0_I] += current_tA * e_val;
                }
            }
            // No A_0(0) slots defined in the provided consts for accums_zero.

            // --- Contributions to Paper Round s_p=1 Accumulators (u = Y1_c, v0 = Y2_c) ---
            // E-factor uses E_out_vec[1] (where Y1_c is u_eff), E_suffix is (Y0_c)
            if y0_c != Infinity {
                // Suffix Y0_c for E_out_vec[1] must be binary
                let e1_val = if y0_c == Zero { e1_suf0 } else { e1_suf1 }; // y0_c is One or Zero here

                if y1_c == Infinity {
                    // u = I
                    match y2_c {
                        // v0 = Y2_c
                        Zero => {
                            accums_infty[ACCUM_IDX_A1_0_I] += current_tA * e1_val;
                        }
                        One => {
                            accums_infty[ACCUM_IDX_A1_1_I] += current_tA * e1_val;
                        }
                        Infinity => {
                            accums_infty[ACCUM_IDX_A1_I_I] += current_tA * e1_val;
                        }
                    }
                } else if y1_c == Zero {
                    // u = 0
                    if y2_c == Infinity {
                        // v0 = I, for A_1(I,0)
                        accums_zero[ACCUM_IDX_A1_I_0] += current_tA * e1_val;
                    }
                }
            }

            // --- Contributions to Paper Round s_p=2 Accumulators (u = Y0_c, v = (Y1_c,Y2_c)) ---
            // E-factor uses E_out_vec[2] (where Y0_c is u_eff), E_suffix is empty
            let e2_val = e2_sufempty;
            if y0_c == Infinity {
                // u = I
                match (y1_c, y2_c) {
                    // v = (Y1_c, Y2_c)
                    (Zero, Zero) => {
                        accums_infty[ACCUM_IDX_A2_00_I] += current_tA * e2_val;
                    }
                    (Zero, One) => {
                        accums_infty[ACCUM_IDX_A2_01_I] += current_tA * e2_val;
                    }
                    (Zero, Infinity) => {
                        accums_infty[ACCUM_IDX_A2_0I_I] += current_tA * e2_val;
                    }
                    (One, Zero) => {
                        accums_infty[ACCUM_IDX_A2_10_I] += current_tA * e2_val;
                    }
                    (One, One) => {
                        accums_infty[ACCUM_IDX_A2_11_I] += current_tA * e2_val;
                    }
                    (One, Infinity) => {
                        accums_infty[ACCUM_IDX_A2_1I_I] += current_tA * e2_val;
                    }
                    (Infinity, Zero) => {
                        accums_infty[ACCUM_IDX_A2_I0_I] += current_tA * e2_val;
                    }
                    (Infinity, One) => {
                        accums_infty[ACCUM_IDX_A2_I1_I] += current_tA * e2_val;
                    }
                    (Infinity, Infinity) => {
                        accums_infty[ACCUM_IDX_A2_II_I] += current_tA * e2_val;
                    }
                }
            } else if y0_c == Zero {
                // u = 0
                match (y1_c, y2_c) {
                    // v = (Y1_c, Y2_c)
                    // Only specific v configs for A_2(v,0) are stored, based on ACCUM_IDX constants
                    (Zero, Infinity) => {
                        accums_zero[ACCUM_IDX_A2_0I_0] += current_tA * e2_val;
                    }
                    (One, Infinity) => {
                        accums_zero[ACCUM_IDX_A2_1I_0] += current_tA * e2_val;
                    }
                    (Infinity, Zero) => {
                        accums_zero[ACCUM_IDX_A2_I0_0] += current_tA * e2_val;
                    }
                    (Infinity, One) => {
                        accums_zero[ACCUM_IDX_A2_I1_0] += current_tA * e2_val;
                    }
                    (Infinity, Infinity) => {
                        accums_zero[ACCUM_IDX_A2_II_0] += current_tA * e2_val;
                    }
                    _ => {} // Other v configs (e.g. fully binary, or other Infinity patterns) for u=0 are not stored.
                }
            }
        }
    }

    // Distributes the accumulated tA values (sum over x_in) for a single x_out_val
    // to the appropriate SVO round accumulators.
    // This is the generic version that works for any number of SVO rounds.
    // We keep around the hard-coded versions for intuition
    #[inline]
    pub fn distribute_tA_to_svo_accumulators<
        const NUM_SVO_ROUNDS: usize,
        const M_NON_BINARY_POINTS: usize,
        F: JoltField,
    >(
        tA_accums: &[F],
        x_out_val: usize,
        E_out_vec: &[Vec<F>],
        accums_zero: &mut [F],
        accums_infty: &mut [F],
    ) {
        if NUM_SVO_ROUNDS == 0 {
            debug_assert!(tA_accums.is_empty(), "tA_accums should be empty for N=0");
            debug_assert!(
                accums_zero.is_empty(),
                "accums_zero should be empty for N=0"
            );
            debug_assert!(
                accums_infty.is_empty(),
                "accums_infty should be empty for N=0"
            );
        }

        // Assert that the provided M_NON_BINARY_POINTS is correct.
        debug_assert_eq!(
            M_NON_BINARY_POINTS,
            num_non_binary_points(NUM_SVO_ROUNDS),
            "M_NON_BINARY_POINTS mismatch with calculated value"
        );

        let y_ext_code_map: [[SVOEvalPoint; NUM_SVO_ROUNDS]; M_NON_BINARY_POINTS] =
            build_y_ext_code_map::<NUM_SVO_ROUNDS, M_NON_BINARY_POINTS>();

        let round_offsets_tuple = precompute_accumulator_offsets::<NUM_SVO_ROUNDS>();
        let round_offsets_infty: [usize; NUM_SVO_ROUNDS] = round_offsets_tuple.0;
        let round_offsets_zero: [usize; NUM_SVO_ROUNDS] = round_offsets_tuple.1;

        debug_assert_eq!(
            tA_accums.len(),
            M_NON_BINARY_POINTS,
            "tA_accums length mismatch with expected non-binary points"
        );

        for tA_idx in 0..M_NON_BINARY_POINTS {
            let current_tA_val = tA_accums[tA_idx];
            if current_tA_val.is_zero() {
                continue;
            }

            let y_ext_coords_msb: &[SVOEvalPoint; NUM_SVO_ROUNDS] = &y_ext_code_map[tA_idx];

            for s_p in 0..NUM_SVO_ROUNDS {
                let num_suffix_vars_for_E = if NUM_SVO_ROUNDS > s_p + 1 {
                    NUM_SVO_ROUNDS - 1 - s_p
                } else {
                    0
                };

                let mut e_suffix_bin_idx = 0;
                let mut e_suffix_is_binary = true;
                if num_suffix_vars_for_E > 0 {
                    for i_suffix_msb in 0..num_suffix_vars_for_E {
                        let coord_val_for_suffix = y_ext_coords_msb[i_suffix_msb];
                        e_suffix_bin_idx <<= 1;
                        match coord_val_for_suffix {
                            SVOEvalPoint::Zero => {}
                            SVOEvalPoint::One => {
                                e_suffix_bin_idx |= 1;
                            }
                            SVOEvalPoint::Infinity => {
                                e_suffix_is_binary = false;
                                break;
                            }
                        }
                    }
                }

                let e_factor: F;
                if e_suffix_is_binary && E_out_vec.len() > s_p && !E_out_vec[s_p].is_empty() {
                    let e_vec_target_idx = (x_out_val << num_suffix_vars_for_E) | e_suffix_bin_idx;
                    if e_vec_target_idx < E_out_vec[s_p].len() {
                        e_factor = E_out_vec[s_p][e_vec_target_idx];
                    } else {
                        e_factor = F::zero();
                    }
                } else {
                    e_factor = F::zero();
                }

                if e_factor.is_zero() {
                    continue;
                }

                let u_A = y_ext_coords_msb[NUM_SVO_ROUNDS - 1 - s_p];

                let mut v_A_coords_lsb_buffer = [SVOEvalPoint::Zero; NUM_SVO_ROUNDS];
                if s_p > 0 {
                    for i_v_lsb in 0..s_p {
                        v_A_coords_lsb_buffer[i_v_lsb] =
                            y_ext_coords_msb[NUM_SVO_ROUNDS - 1 - i_v_lsb];
                    }
                }
                let v_A_slice_lsb_order = &v_A_coords_lsb_buffer[0..s_p];

                match u_A {
                    SVOEvalPoint::Infinity => {
                        let base_offset = round_offsets_infty[s_p];
                        let idx_within_block = v_coords_to_base3_idx(v_A_slice_lsb_order);
                        let final_idx = base_offset + idx_within_block;
                        if final_idx < accums_infty.len() {
                            accums_infty[final_idx] += current_tA_val * e_factor;
                        }
                    }
                    SVOEvalPoint::Zero => {
                        if s_p == 0 {
                            continue;
                        }

                        if v_coords_has_infinity(v_A_slice_lsb_order) {
                            let base_offset = round_offsets_zero[s_p];
                            let num_slots_in_block_A_sp_zero = pow(3, s_p) - pow(2, s_p);

                            if num_slots_in_block_A_sp_zero > 0 {
                                let idx_within_block =
                                    v_coords_to_non_binary_base3_idx(v_A_slice_lsb_order);
                                let final_idx = base_offset + idx_within_block;
                                if final_idx < accums_zero.len()
                                    && idx_within_block < num_slots_in_block_A_sp_zero
                                {
                                    accums_zero[final_idx] += current_tA_val * e_factor;
                                }
                            }
                        }
                    }
                    SVOEvalPoint::One => {}
                }
            }
        }
    }

    /// Process the first few sum-check rounds using small value optimization (SVO)
    /// We take in the pre-computed accumulator values, and use them to compute the quadratic
    /// evaluations (and thus cubic polynomials) for the first few sum-check rounds.
    pub fn process_svo_sumcheck_rounds<
        const NUM_SVO_ROUNDS: usize,
        F: JoltField,
        ProofTranscript: Transcript,
    >(
        accums_zero: &[F],
        accums_infty: &[F],
        r_challenges: &mut Vec<F>,
        round_polys: &mut Vec<CompressedUniPoly<F>>,
        claim: &mut F,
        transcript: &mut ProofTranscript,
        eq_poly: &mut GruenSplitEqPolynomial<F>,
    ) {
        // Assert lengths of accumulator slices based on NUM_SVO_ROUNDS
        let expected_accums_zero_len = num_accums_eval_zero(NUM_SVO_ROUNDS);
        let expected_accums_infty_len = num_accums_eval_infty(NUM_SVO_ROUNDS);
        assert_eq!(
            accums_zero.len(),
            expected_accums_zero_len,
            "accums_zero length mismatch"
        );
        assert_eq!(
            accums_infty.len(),
            expected_accums_infty_len,
            "accums_infty length mismatch"
        );

        let mut lagrange_coeffs: Vec<F> = vec![F::one()];
        let mut current_acc_zero_offset = 0;
        let mut current_acc_infty_offset = 0;

        for i in 0..NUM_SVO_ROUNDS {
            let mut quadratic_eval_0 = F::zero();
            let mut quadratic_eval_infty = F::zero();

            let num_vars_in_v_config = i; // v_config is (v_0, ..., v_{i-1})
            let num_lagrange_coeffs_for_round = pow(3, num_vars_in_v_config);

            // Compute quadratic_eval_infty
            let num_accs_infty_curr_round = pow(3, num_vars_in_v_config);
            if num_accs_infty_curr_round > 0
                && current_acc_infty_offset + num_accs_infty_curr_round <= accums_infty.len()
            {
                let accums_infty_slice = &accums_infty[current_acc_infty_offset
                    ..current_acc_infty_offset + num_accs_infty_curr_round];
                for k in 0..num_lagrange_coeffs_for_round {
                    if k < accums_infty_slice.len() && k < lagrange_coeffs.len() {
                        quadratic_eval_infty += accums_infty_slice[k] * lagrange_coeffs[k];
                    }
                }
            }
            current_acc_infty_offset += num_accs_infty_curr_round;

            // Compute quadratic_eval_0
            let num_accs_zero_curr_round = if num_vars_in_v_config == 0 {
                0 // 3^0 - 2^0 = 0
            } else {
                pow(3, num_vars_in_v_config) - pow(2, num_vars_in_v_config)
            };

            if num_accs_zero_curr_round > 0
                && current_acc_zero_offset + num_accs_zero_curr_round <= accums_zero.len()
            {
                let accums_zero_slice = &accums_zero
                    [current_acc_zero_offset..current_acc_zero_offset + num_accs_zero_curr_round];
                let mut non_binary_v_config_counter = 0;
                for k_global in 0..num_lagrange_coeffs_for_round {
                    let v_config = get_v_config_digits(k_global, num_vars_in_v_config);
                    if is_v_config_non_binary(&v_config)
                        && non_binary_v_config_counter < accums_zero_slice.len()
                        && k_global < lagrange_coeffs.len()
                    {
                        quadratic_eval_0 += accums_zero_slice[non_binary_v_config_counter]
                            * lagrange_coeffs[k_global];
                        non_binary_v_config_counter += 1;
                    }
                }
                current_acc_zero_offset += num_accs_zero_curr_round;
            }

            let r_i = process_eq_sumcheck_round(
                (quadratic_eval_0, quadratic_eval_infty),
                eq_poly,
                round_polys,
                r_challenges,
                claim,
                transcript,
            );

            let lagrange_coeffs_r_i = [F::one() - r_i, r_i, r_i * (r_i - F::one())];

            if i < NUM_SVO_ROUNDS.saturating_sub(1) {
                lagrange_coeffs = lagrange_coeffs_r_i
                    .iter()
                    .flat_map(|lagrange_coeff| {
                        lagrange_coeffs
                            .iter()
                            .map(move |coeff| *lagrange_coeff * *coeff)
                    })
                    .collect();
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct TernaryPointInfo<const N: usize> {
        /// True if this point (y_0, ..., y_{N-1}) has all y_i in {0, 1}.
        pub is_binary: bool,
        /// If is_binary is true, this is the index into the binary_az_evals_input.
        /// If is_binary is false, this field is unused (can be 0).
        pub binary_eval_idx: usize,
        /// If is_binary is false, this is the k_ternary_idx corresponding to
        /// replacing the *most significant* Infinity with a 1.
        /// If is_binary is true, this field is unused (can be 0).
        pub k_val_at_one: usize,
        /// If is_binary is false, this is the k_ternary_idx corresponding to
        /// replacing the *most significant* Infinity with a 0.
        /// If is_binary is true, this field is unused (can be 0).
        pub k_val_at_zero: usize,
    }

    impl<const N: usize> TernaryPointInfo<N> {
        pub const fn default_val() -> Self {
            Self {
                is_binary: false,
                binary_eval_idx: 0,
                k_val_at_one: 0,
                k_val_at_zero: 0,
            }
        }
    }

    /// Converts a k_ternary_idx (0 to 3^N - 1) to its base-3 digits, MSB-first.
    /// Example: N=3, k=5 (012_base3). Returns [0,1,2].
    /// k=26 (222_base3). Returns [2,2,2].
    pub const fn get_msb_ternary_digits<const N: usize>(k: usize) -> [u8; N] {
        if N == 0 {
            return [0u8; N];
        }
        let mut digits = [0u8; N];
        let mut temp_k = k;
        let mut i_rev = 0;
        while i_rev < N {
            let i_dim_msb = N - 1 - i_rev; // Current dimension index (MSB is 0, LSB is N-1)
            digits[i_dim_msb] = (temp_k % 3) as u8;
            temp_k /= 3;
            i_rev += 1;
        }
        digits
    }

    pub const fn precompute_ternary_point_infos<
        const N: usize,
        const NUM_TERNARY_POINTS_VAL: usize,
    >() -> [TernaryPointInfo<N>; NUM_TERNARY_POINTS_VAL] {
        if N == 0 {
            // NUM_TERNARY_POINTS_VAL should be 1 in this case (pow(3,0)=1)
            // An empty array cannot be returned if NUM_TERNARY_POINTS_VAL is > 0.
            // The caller should handle N=0 as a special case if an empty array is truly desired.
            // For now, if N=0 but NUM_TERNARY_POINTS_VAL = 1, we provide one default entry.
            if NUM_TERNARY_POINTS_VAL == 1 {
                let mut infos = [TernaryPointInfo::<N>::default_val(); NUM_TERNARY_POINTS_VAL];
                // For N=0, point k=0 is binary, index 0.
                // infos[0] = TernaryPointInfo { is_binary: true, binary_eval_idx: 0, k_val_at_one: 0, k_val_at_zero: 0 };
                // However, the existing code for N=0 for compute_and_update_tA_inplace simply returns.
                // Let's keep it simple and consistent with that.
                // If N=0, NUM_TERNARY_POINTS_VAL will be 1.
                // The point k=0 is binary, with binary_eval_idx = 0.
                // The actual logic in compute_and_update_tA_inplace will handle N=0 separately.
                // So this precomputation for N=0 might not even be used directly if compute_and_update_tA_inplace returns early.
                // For safety, and if it were used, the single point k=0 is binary.
                let mut point_info = TernaryPointInfo::<N>::default_val();
                point_info.is_binary = true;
                point_info.binary_eval_idx = 0;
                infos[0] = point_info;
                return infos;
            } else {
                // This state implies an inconsistency (e.g. N=0 but NUM_TERNARY_POINTS_VAL != 1)
                // Const panics are not stable yet, so we return a default array.
                // This should be caught by debug_asserts in the calling function.
                return [TernaryPointInfo::<N>::default_val(); NUM_TERNARY_POINTS_VAL];
            }
        }

        let mut infos = [TernaryPointInfo::<N>::default_val(); NUM_TERNARY_POINTS_VAL];
        let mut k_ternary_idx = 0;
        while k_ternary_idx < NUM_TERNARY_POINTS_VAL {
            let current_coords_base3_msb = get_msb_ternary_digits::<N>(k_ternary_idx);
            let mut is_binary_point_flag = true;
            let mut first_inf_dim_msb_idx: Option<usize> = None; // Dimension index (0 to N-1, MSB-first)

            let mut i_dim_msb = 0;
            while i_dim_msb < N {
                if current_coords_base3_msb[i_dim_msb] == 2 {
                    // 2 represents Infinity
                    is_binary_point_flag = false;
                    if first_inf_dim_msb_idx.is_none() {
                        first_inf_dim_msb_idx = Some(i_dim_msb);
                    }
                    // We only care about the *most significant* (smallest index) infinity for the reduction rule.
                    // So we can break once the first one is found.
                    break;
                }
                i_dim_msb += 1;
            }

            if is_binary_point_flag {
                let mut binary_idx = 0;
                let mut dim_idx_msb = 0;
                while dim_idx_msb < N {
                    binary_idx <<= 1;
                    if current_coords_base3_msb[dim_idx_msb] == 1 {
                        binary_idx |= 1;
                    }
                    dim_idx_msb += 1;
                }
                infos[k_ternary_idx] = TernaryPointInfo {
                    is_binary: true,
                    binary_eval_idx: binary_idx,
                    k_val_at_one: 0,  // Unused
                    k_val_at_zero: 0, // Unused
                };
            } else {
                // This is a non-binary point.
                // j_inf_dim is the index of the *most significant* dimension that is Infinity.
                // Example: N=3, point (Inf, 0, Inf) = (2,0,2)_base3. k_ternary_idx = 2*9 + 0*3 + 2*1 = 20.
                // current_coords_base3_msb = [2,0,2].
                // first_inf_dim_msb_idx will be Some(0).
                let j_inf_dim = first_inf_dim_msb_idx.unwrap(); // Safe due to !is_binary_point_flag

                // Power of 3 for the dimension j_inf_dim (0-indexed from MSB)
                // N=3, j_inf_dim=0 (MSB y_0), power = 3^(3-1-0) = 3^2 = 9
                // N=3, j_inf_dim=1 (Mid y_1), power = 3^(3-1-1) = 3^1 = 3
                // N=3, j_inf_dim=2 (LSB y_2), power = 3^(3-1-2) = 3^0 = 1
                let inf_dim_power_of_3 = pow(3, N - 1 - j_inf_dim);

                // k_ternary_idx has a '2' at dimension j_inf_dim (MSB-numbering).
                // k_at_1_idx corresponds to changing this '2' to a '1'.
                // k_at_0_idx corresponds to changing this '2' to a '0'.
                let k_at_1_val = k_ternary_idx - inf_dim_power_of_3;
                let k_at_0_val = k_ternary_idx - 2 * inf_dim_power_of_3;

                infos[k_ternary_idx] = TernaryPointInfo {
                    is_binary: false,
                    binary_eval_idx: 0, // Unused
                    k_val_at_one: k_at_1_val,
                    k_val_at_zero: k_at_0_val,
                };
            }
            k_ternary_idx += 1;
        }
        infos
    }
}

#[cfg(test)]
mod tests {
    use super::accum::UnreducedProduct;
    use super::svo_helpers::{
        compute_and_update_tA_inplace, compute_and_update_tA_inplace_1,
        compute_and_update_tA_inplace_2, compute_and_update_tA_inplace_3,
        compute_and_update_tA_inplace_const,
    };
    use crate::poly::eq_poly::EqPolynomial;
    use ark_bn254::Fr;
    use ark_ff::biginteger::{I8OrI96, S160};
    use ark_ff::UniformRand;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    fn random_az_value<R: Rng>(rng: &mut R) -> I8OrI96 {
        match rng.gen_range(0..5) {
            0 => I8OrI96::from_i8(rng.gen()),
            1 => I8OrI96::from_i8(0), // zero
            2 => I8OrI96::from_i8(1), // one
            3 => I8OrI96::from_i128(rng.gen::<i64>() as i128),
            4 => {
                // Bounded 90-bit magnitude to ensure it always fits in I8OrI96,
                // and give headroom so differences during extension remain within 96 bits.
                const BITS: u32 = 90;
                let mask: u128 = if BITS == 128 {
                    u128::MAX
                } else {
                    (1u128 << BITS) - 1
                };
                let mag = (rng.gen::<u128>() & mask) as i128;
                let val = if rng.gen::<bool>() { mag } else { -mag };
                I8OrI96::from_i128(val)
            }
            _ => unreachable!(),
        }
    }

    fn random_bz_value<R: Rng>(rng: &mut R) -> S160 {
        match rng.gen_range(0..4) {
            0 => S160::from(0i128),
            1 => S160::from(1i128),
            2 => S160::from(rng.gen::<i64>() as i128),
            3 => {
                // Bounded 156-bit magnitude to avoid overflow when summing up to 8 terms
                // during ternary extension (N<=3 => 2^N <= 8).
                // Use 120-bit cap to stay safely within S160 even after up to 8-term sums.
                const BITS: u32 = 120;
                let mask: u128 = (1u128 << BITS) - 1;
                let mag = (rng.gen::<u128>() & mask) as i128;
                let val = if rng.gen::<bool>() { mag } else { -mag };
                S160::from(val)
            }
            _ => unreachable!(),
        }
    }

    /// Consistency check: hardcoded vs generic small value compute and update tA in place
    fn run_svo_hardcoded_vs_generic_consistency_check<const NUM_SVO_ROUNDS: usize>(
        rng: &mut ChaCha20Rng,
    ) {
        let num_vars = NUM_SVO_ROUNDS;
        if num_vars == 0 {
            return;
        }
        let num_non_trivial = 3_usize.pow(num_vars as u32) - 2_usize.pow(num_vars as u32);

        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(rng)).collect();
        let e_in_val: Vec<Fr> = EqPolynomial::evals(&r);

        let num_binary_points = 1 << num_vars;

        // Create random binary evaluations
        let binary_az_vals: Vec<I8OrI96> = (0..num_binary_points)
            .map(|_| random_az_value(rng))
            .collect();
        let binary_bz_vals: Vec<S160> = (0..num_binary_points)
            .map(|_| random_bz_value(rng))
            .collect();

        // Generic small value path (produces Montgomery-form Fr elements after reduction)
        let mut ta_pos_acc_generic = vec![UnreducedProduct::<Fr>::default(); num_non_trivial];
        let mut ta_neg_acc_generic = vec![UnreducedProduct::<Fr>::default(); num_non_trivial];
        compute_and_update_tA_inplace::<NUM_SVO_ROUNDS, Fr>(
            &binary_az_vals,
            &binary_bz_vals,
            &e_in_val[0], // Use first element
            &mut ta_pos_acc_generic,
            &mut ta_neg_acc_generic,
        );

        // Hardcoded small value path - call the hardcoded versions directly
        let mut ta_pos_acc_hardcoded = vec![UnreducedProduct::<Fr>::default(); num_non_trivial];
        let mut ta_neg_acc_hardcoded = vec![UnreducedProduct::<Fr>::default(); num_non_trivial];

        // Call the hardcoded versions directly based on NUM_SVO_ROUNDS
        match NUM_SVO_ROUNDS {
            1 => compute_and_update_tA_inplace_1::<Fr>(
                &binary_az_vals,
                &binary_bz_vals,
                &e_in_val[0],
                &mut ta_pos_acc_hardcoded,
                &mut ta_neg_acc_hardcoded,
            ),
            2 => compute_and_update_tA_inplace_2::<Fr>(
                &binary_az_vals,
                &binary_bz_vals,
                &e_in_val[0],
                &mut ta_pos_acc_hardcoded,
                &mut ta_neg_acc_hardcoded,
            ),
            3 => compute_and_update_tA_inplace_3::<Fr>(
                &binary_az_vals,
                &binary_bz_vals,
                &e_in_val[0],
                &mut ta_pos_acc_hardcoded,
                &mut ta_neg_acc_hardcoded,
            ),
            4 => {
                // For NUM_SVO_ROUNDS == 4, use the const generic version
                compute_and_update_tA_inplace_const::<4, 65, 81, Fr>(
                    &binary_az_vals,
                    &binary_bz_vals,
                    &e_in_val[0],
                    &mut ta_pos_acc_hardcoded,
                    &mut ta_neg_acc_hardcoded,
                );
            }
            5 => {
                // For NUM_SVO_ROUNDS == 5, use the const generic version
                compute_and_update_tA_inplace_const::<5, 211, 243, Fr>(
                    &binary_az_vals,
                    &binary_bz_vals,
                    &e_in_val[0],
                    &mut ta_pos_acc_hardcoded,
                    &mut ta_neg_acc_hardcoded,
                );
            }
            _ => {
                panic!(
                    "Unsupported NUM_SVO_ROUNDS for hardcoded consistency check: {NUM_SVO_ROUNDS}"
                );
            }
        }

        // Compare results
        for i in 0..num_non_trivial {
            let generic_pos = Fr::montgomery_reduce_2n::<8>(ta_pos_acc_generic[i]);
            let generic_neg = Fr::montgomery_reduce_2n::<8>(ta_neg_acc_generic[i]);
            let generic_result = generic_pos - generic_neg;

            let hardcoded_pos = Fr::montgomery_reduce_2n::<8>(ta_pos_acc_hardcoded[i]);
            let hardcoded_neg = Fr::montgomery_reduce_2n::<8>(ta_neg_acc_hardcoded[i]);
            let hardcoded_result = hardcoded_pos - hardcoded_neg;

            assert_eq!(
                generic_result, hardcoded_result,
                "Hardcoded vs Generic small value mismatch for NUM_SVO_ROUNDS={NUM_SVO_ROUNDS} at index {i}"
            );
        }
    }

    #[test]
    fn test_svo_hardcoded_vs_generic_consistency() {
        let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
        run_svo_hardcoded_vs_generic_consistency_check::<1>(&mut rng);
        run_svo_hardcoded_vs_generic_consistency_check::<2>(&mut rng);
        run_svo_hardcoded_vs_generic_consistency_check::<3>(&mut rng);
        run_svo_hardcoded_vs_generic_consistency_check::<4>(&mut rng);
        run_svo_hardcoded_vs_generic_consistency_check::<5>(&mut rng);
    }
}
