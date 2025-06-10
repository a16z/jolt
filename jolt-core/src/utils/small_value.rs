// Small Value Optimization (SVO) helpers for Spartan first sum-check

use crate::field::{JoltField, OptimizedMulI128};

/// Number of rounds to use for small value optimization.
/// Testing & estimation shows that 3 rounds is the best tradeoff
/// It may be 4 rounds when we switch to streaming
pub const NUM_SVO_ROUNDS: usize = 3;

pub mod svo_helpers {
    use super::*;
    use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
    use crate::poly::unipoly::CompressedUniPoly;
    use crate::subprotocols::sumcheck::process_eq_sumcheck_round;
    use crate::utils::transcript::Transcript;

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

    pub fn compute_and_update_tA_inplace_generic<const NUM_SVO_ROUNDS: usize, F: JoltField>(
        binary_az_evals: &[i128],
        binary_bz_evals: &[i128],
        e_in_val: &F,
        temp_tA: &mut [F],
    ) {
        match NUM_SVO_ROUNDS {
            1 => {
                compute_and_update_tA_inplace_1(binary_az_evals, binary_bz_evals, e_in_val, temp_tA)
            }
            2 => {
                compute_and_update_tA_inplace_2(binary_az_evals, binary_bz_evals, e_in_val, temp_tA)
            }
            3 => {
                compute_and_update_tA_inplace_3(binary_az_evals, binary_bz_evals, e_in_val, temp_tA)
            }
            // 81 - 16 = 65 non-binary points
            4 => compute_and_update_tA_inplace::<4, 65, 81, F>(
                binary_az_evals,
                binary_bz_evals,
                e_in_val,
                temp_tA,
            ),
            _ => {
                panic!("Unsupported number of SVO rounds: {NUM_SVO_ROUNDS}");
            }
        }
    }

    /// Special case when `num_svo_rounds == 1`
    /// In this case, we know that there is 3 - 2 = 1 temp_tA accumulator,
    /// corresponding to temp_tA(infty), which should be updated via
    /// temp_tA[infty] += e_in_val * (az[1] - az[0]) * (bz[1] - bz[0])
    #[inline]
    pub fn compute_and_update_tA_inplace_1<F: JoltField>(
        binary_az_evals: &[i128],
        binary_bz_evals: &[i128],
        e_in_val: &F,
        temp_tA: &mut [F],
    ) {
        debug_assert!(binary_az_evals.len() == 2);
        debug_assert!(binary_bz_evals.len() == 2);
        debug_assert!(temp_tA.len() == 1);
        let az_I = binary_az_evals[1] - binary_az_evals[0];
        if az_I != 0 {
            let bz_I = binary_bz_evals[1] - binary_bz_evals[0];
            if bz_I != 0 {
                let product_i128 = az_I
                    .checked_mul(bz_I)
                    .expect("Az_I*Bz_I product overflow i128");
                temp_tA[0] += e_in_val.mul_i128(product_i128);
            }
        }
    }

    /// Special case when `num_svo_rounds == 2`
    /// In this case, we know that there are 4 binary evals of Az and Bz,
    /// corresponding to (0,0) => 0, (0,1) => 1, (1,0) => 2, (1,1) => 3. (i.e. big endian, TODO: double-check this)
    ///
    /// There are 9 - 4 = 5 temp_tA accumulators, with the following logic (in this order 0 => 4 indexing):
    /// temp_tA[0,∞] += e_in_val * (az[0,1] - az[0,0]) * (bz[0,1] - bz[0,0])
    /// temp_tA[1,∞] += e_in_val * (az[1,1] - az[1,0]) * (bz[1,1] - bz[1,0])
    /// temp_tA[∞,0] += e_in_val * (az[1,0] - az[0,0]) * (bz[1,0] - bz[0,0])
    /// temp_tA[∞,1] += e_in_val * (az[1,1] - az[0,1]) * (bz[1,1] - bz[0,1])
    /// temp_tA[∞,∞] += e_in_val * (az[1,∞] - az[0,∞]) * (bz[1,∞] - bz[0,∞])
    #[inline]
    pub fn compute_and_update_tA_inplace_2<F: JoltField>(
        binary_az_evals: &[i128],
        binary_bz_evals: &[i128],
        e_in_val: &F,
        temp_tA: &mut [F],
    ) {
        debug_assert!(binary_az_evals.len() == 4);
        debug_assert!(binary_bz_evals.len() == 4);
        debug_assert!(temp_tA.len() == 5);
        // Binary evaluations: (Y0,Y1) -> index Y0*2 + Y1
        let az00 = binary_az_evals[0];
        let bz00 = binary_bz_evals[0]; // Y0=0, Y1=0
        let az01 = binary_az_evals[1];
        let bz01 = binary_bz_evals[1]; // Y0=0, Y1=1
        let az10 = binary_az_evals[2];
        let bz10 = binary_bz_evals[2]; // Y0=1, Y1=0
        let az11 = binary_az_evals[3];
        let bz11 = binary_bz_evals[3]; // Y0=1, Y1=1

        // Extended evaluations (points with at least one I)
        // temp_tA indices follow the order: (0,I), (1,I), (I,0), (I,1), (I,I)

        // 1. Point (0,I) -> temp_tA[0]
        let az_0I = az01 - az00;
        let bz_0I = bz01 - bz00;
        if az_0I != 0 && bz_0I != 0 {
            let prod_0I = az_0I
                .checked_mul(bz_0I)
                .expect("Product overflow for (0,I)");
            temp_tA[0] += e_in_val.mul_i128(prod_0I);
        }

        // 2. Point (1,I) -> temp_tA[1]
        let az_1I = az11 - az10;
        let bz_1I = bz11 - bz10;
        if az_1I != 0 && bz_1I != 0 {
            let prod_1I = az_1I
                .checked_mul(bz_1I)
                .expect("Product overflow for (1,I)");
            temp_tA[1] += e_in_val.mul_i128(prod_1I);
        }

        // 3. Point (I,0) -> temp_tA[2]
        let az_I0 = az10 - az00;
        if az_I0 != 0 {
            let bz_I0 = bz10 - bz00;
            if bz_I0 != 0 {
                let prod_I0 = az_I0
                    .checked_mul(bz_I0)
                    .expect("Product overflow for (I,0)");
                temp_tA[2] += e_in_val.mul_i128(prod_I0);
            }
        }

        // 4. Point (I,1) -> temp_tA[3]
        let az_I1 = az11 - az01;
        if az_I1 != 0 {
            let bz_I1 = bz11 - bz01;
            if bz_I1 != 0 {
                let prod_I1 = az_I1
                    .checked_mul(bz_I1)
                    .expect("Product overflow for (I,1)");
                temp_tA[3] += e_in_val.mul_i128(prod_I1);
            }
        }

        // 5. Point (I,I) -> temp_tA[4]
        let az_II = az_1I - az_0I;
        if az_II != 0 {
            let bz_II = bz_1I - bz_0I;
            if bz_II != 0 {
                let prod_II = az_II
                    .checked_mul(bz_II)
                    .expect("Product overflow for (I,I)");
                // At this point, pretty unlikely for prod_II to be 1
                temp_tA[4] += e_in_val.mul_i128(prod_II);
            }
        }
    }

    /// Special case when `num_svo_rounds == 3`
    /// In this case, we know that there are 8 binary evals of Az and Bz,
    /// corresponding to (0,0,0) => 0, (0,0,1) => 1, (0,1,0) => 2, (0,1,1) => 3, (1,0,0) => 4,
    /// (1,0,1) => 5, (1,1,0) => 6, (1,1,1) => 7.
    ///
    /// There are 27 - 8 = 19 temp_tA accumulators, with the following logic:
    /// (listed in increasing order, when considering ∞ as 2 and going from MSB to LSB)
    ///
    /// temp_tA[0,0,∞] += e_in_val * (az[0,0,1] - az[0,0,0]) * (bz[0,0,1] - bz[0,0,0])
    /// temp_tA[0,1,∞] += e_in_val * (az[0,1,1] - az[0,1,0]) * (bz[0,1,1] - bz[0,1,0])
    /// temp_tA[0,∞,0] += e_in_val * (az[0,1,0] - az[0,0,0]) * (bz[0,1,0] - bz[0,0,0])
    /// temp_tA[0,∞,1] += e_in_val * (az[0,1,1] - az[0,0,1]) * (bz[0,1,1] - bz[0,0,1])
    /// temp_tA[0,∞,∞] += e_in_val * (az[0,1,∞] - az[0,0,∞]) * (bz[0,1,∞] - bz[0,0,∞])
    ///
    /// temp_tA[1,0,∞] += e_in_val * (az[1,0,1] - az[1,0,0]) * (bz[1,0,1] - bz[1,0,0])
    /// temp_tA[1,1,∞] += e_in_val * (az[1,1,1] - az[1,1,0]) * (bz[1,1,1] - bz[1,1,0])
    /// temp_tA[1,∞,0] += e_in_val * (az[1,1,0] - az[1,0,0]) * (bz[1,1,0] - bz[1,0,0])
    /// temp_tA[1,∞,1] += e_in_val * (az[1,1,0] - az[1,0,0]) * (bz[1,1,0] - bz[1,0,0])
    /// temp_tA[1,∞,∞] += e_in_val * (az[1,1,∞] - az[1,0,∞]) * (bz[1,1,∞] - bz[1,0,∞])
    ///
    /// temp_tA[∞,0,0] += e_in_val * (az[1,0,0] - az[0,0,0]) * (bz[1,0,0] - bz[0,0,0])
    /// temp_tA[∞,0,1] += e_in_val * (az[1,0,1] - az[0,0,1]) * (bz[1,0,1] - bz[0,0,1])
    /// temp_tA[∞,0,∞] += e_in_val * (az[1,0,∞] - az[0,0,∞]) * (bz[1,0,∞] - bz[0,0,∞])
    /// temp_tA[∞,1,0] += e_in_val * (az[1,1,0] - az[0,1,0]) * (bz[1,1,0] - bz[0,1,0])
    /// temp_tA[∞,1,1] += e_in_val * (az[1,1,1] - az[0,1,1]) * (bz[1,1,1] - bz[0,1,1])
    /// temp_tA[∞,1,∞] += e_in_val * (az[1,1,∞] - az[0,1,∞]) * (bz[1,1,∞] - bz[0,1,∞])
    /// temp_tA[∞,∞,0] += e_in_val * (az[1,∞,0] - az[0,∞,0]) * (bz[1,∞,0] - bz[0,∞,0])
    /// temp_tA[∞,∞,1] += e_in_val * (az[1,∞,1] - az[0,∞,1]) * (bz[1,∞,1] - bz[0,∞,1])
    /// temp_tA[∞,∞,∞] += e_in_val * (az[1,∞,∞] - az[0,∞,∞]) * (bz[1,∞,∞] - bz[0,∞,∞])
    ///
    #[inline]
    pub fn compute_and_update_tA_inplace_3<F: JoltField>(
        binary_az_evals: &[i128],
        binary_bz_evals: &[i128],
        e_in_val: &F,
        temp_tA: &mut [F],
    ) {
        debug_assert!(binary_az_evals.len() == 8);
        debug_assert!(binary_bz_evals.len() == 8);
        debug_assert!(temp_tA.len() == 19);

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

        // Populate temp_tA in lexicographical MSB-first order
        // Z=0, O=1, I=2 for Y_i

        // Point (0,0,I) -> temp_tA[0]
        let az_00I = az001 - az000;
        let bz_00I = bz001 - bz000;
        if az_00I != 0 && bz_00I != 0 {
            let prod = az_00I.checked_mul(bz_00I).expect("Prod overflow");
            // Test `mul_i128_1_optimized`
            temp_tA[0] += e_in_val.mul_i128_1_optimized(prod);
        }

        // Point (0,1,I) -> temp_tA[1]
        let az_01I = az011 - az010;
        let bz_01I = bz011 - bz010;
        if az_01I != 0 && bz_01I != 0 {
            let prod = az_01I.checked_mul(bz_01I).expect("Prod overflow");
            temp_tA[1] += e_in_val.mul_i128(prod);
        }

        // Point (0,I,0) -> temp_tA[2]
        let az_0I0 = az010 - az000;
        let bz_0I0 = bz010 - bz000;
        if az_0I0 != 0 && bz_0I0 != 0 {
            let prod = az_0I0.checked_mul(bz_0I0).expect("Prod overflow");
            temp_tA[2] += e_in_val.mul_i128(prod);
        }

        // Point (0,I,1) -> temp_tA[3]
        let az_0I1 = az011 - az001;
        let bz_0I1 = bz011 - bz001;
        if az_0I1 != 0 && bz_0I1 != 0 {
            let prod = az_0I1.checked_mul(bz_0I1).expect("Prod overflow");
            temp_tA[3] += e_in_val.mul_i128(prod);
        }

        // Point (0,I,I) -> temp_tA[4]
        let az_0II = az_01I - az_00I;
        let bz_0II = bz_01I - bz_00I; // Need to compute this outside for III term
        if az_0II != 0 && bz_0II != 0 {
            let prod = az_0II.checked_mul(bz_0II).expect("Prod overflow");
            temp_tA[4] += e_in_val.mul_i128(prod);
        }

        // Point (1,0,I) -> temp_tA[5]
        let az_10I = az101 - az100;
        let bz_10I = bz101 - bz100;
        if az_10I != 0 && bz_10I != 0 {
            let prod = az_10I.checked_mul(bz_10I).expect("Prod overflow");
            temp_tA[5] += e_in_val.mul_i128(prod);
        }

        // Point (1,1,I) -> temp_tA[6]
        let az_11I = az111 - az110;
        let bz_11I = bz111 - bz110;
        if az_11I != 0 && bz_11I != 0 {
            let prod = az_11I.checked_mul(bz_11I).expect("Prod overflow");
            temp_tA[6] += e_in_val.mul_i128(prod);
        }

        // Point (1,I,0) -> temp_tA[7]
        let az_1I0 = az110 - az100;
        let bz_1I0 = bz110 - bz100;
        if az_1I0 != 0 && bz_1I0 != 0 {
            let prod = az_1I0.checked_mul(bz_1I0).expect("Prod overflow");
            temp_tA[7] += e_in_val.mul_i128(prod);
        }

        // Point (1,I,1) -> temp_tA[8]
        let az_1I1 = az111 - az101;
        let bz_1I1 = bz111 - bz101;
        if az_1I1 != 0 && bz_1I1 != 0 {
            let prod = az_1I1.checked_mul(bz_1I1).expect("Prod overflow");
            temp_tA[8] += e_in_val.mul_i128(prod);
        }

        // Point (1,I,I) -> temp_tA[9]
        let az_1II = az_11I - az_10I;
        let bz_1II = bz_11I - bz_10I; // Need to compute this outside for III term
        if az_1II != 0 && bz_1II != 0 {
            let prod = az_1II.checked_mul(bz_1II).expect("Prod overflow");
            temp_tA[9] += e_in_val.mul_i128(prod);
        }

        // Point (I,0,0) -> temp_tA[10]
        let az_I00 = az100 - az000;
        let bz_I00 = bz100 - bz000;
        if az_I00 != 0 && bz_I00 != 0 {
            let prod = az_I00.checked_mul(bz_I00).expect("Prod overflow");
            temp_tA[10] += e_in_val.mul_i128(prod);
        }

        // Point (I,0,1) -> temp_tA[11]
        let az_I01 = az101 - az001;
        let bz_I01 = bz101 - bz001;
        if az_I01 != 0 && bz_I01 != 0 {
            let prod = az_I01.checked_mul(bz_I01).expect("Prod overflow");
            temp_tA[11] += e_in_val.mul_i128(prod);
        }

        // Point (I,0,I) -> temp_tA[12]
        let az_I0I = az_I01 - az_I00; // Uses precomputed az_I01, az_I00
        if az_I0I != 0 {
            let bz_I0I = bz_I01 - bz_I00; // Uses precomputed bz_I01, bz_I00
            if bz_I0I != 0 {
                let prod = az_I0I.checked_mul(bz_I0I).expect("Prod overflow");
                temp_tA[12] += e_in_val.mul_i128(prod);
            }
        }

        // Point (I,1,0) -> temp_tA[13]
        let az_I10 = az110 - az010;
        let bz_I10 = bz110 - bz010;
        if az_I10 != 0 && bz_I10 != 0 {
            let prod = az_I10.checked_mul(bz_I10).expect("Prod overflow");
            temp_tA[13] += e_in_val.mul_i128(prod);
        }

        // Point (I,1,1) -> temp_tA[14]
        let az_I11 = az111 - az011;
        let bz_I11 = bz111 - bz011;
        if az_I11 != 0 && bz_I11 != 0 {
            let prod = az_I11.checked_mul(bz_I11).expect("Prod overflow");
            temp_tA[14] += e_in_val.mul_i128(prod);
        }

        // Point (I,1,I) -> temp_tA[15]
        let az_I1I = az_I11 - az_I10; // Uses precomputed az_I11, az_I10
        if az_I1I != 0 {
            let bz_I1I = bz_I11 - bz_I10; // Uses precomputed bz_I11, bz_I10
            if bz_I1I != 0 {
                let prod = az_I1I.checked_mul(bz_I1I).expect("Prod overflow");
                temp_tA[15] += e_in_val.mul_i128(prod);
            }
        }

        // Point (I,I,0) -> temp_tA[16]
        let az_II0 = az_1I0 - az_0I0; // Uses precomputed az_1I0, az_0I0
        if az_II0 != 0 {
            let bz_II0 = bz_1I0 - bz_0I0; // Uses precomputed bz_1I0, bz_0I0
            if bz_II0 != 0 {
                let prod = az_II0.checked_mul(bz_II0).expect("Prod overflow");
                temp_tA[16] += e_in_val.mul_i128(prod);
            }
        }

        // Point (I,I,1) -> temp_tA[17]
        let az_II1 = az_1I1 - az_0I1; // Uses precomputed az_1I1, az_0I1
        if az_II1 != 0 {
            let bz_II1 = bz_1I1 - bz_0I1; // Uses precomputed bz_1I1, bz_0I1
            if bz_II1 != 0 {
                let prod = az_II1.checked_mul(bz_II1).expect("Prod overflow");
                temp_tA[17] += e_in_val.mul_i128(prod);
            }
        }

        // Point (I,I,I) -> temp_tA[18]
        // az_III depends on az_1II and az_0II.
        // az_1II was computed for temp_tA[9]
        // az_0II was computed for temp_tA[4]
        let az_III = az_1II - az_0II;
        if az_III != 0 {
            // bz_III depends on bz_1II and bz_0II.
            // bz_1II was computed for temp_tA[9]
            // bz_0II was computed for temp_tA[4]
            let bz_III = bz_1II - bz_0II;
            if bz_III != 0 {
                let prod = az_III.checked_mul(bz_III).expect("Prod overflow");
                temp_tA[18] += e_in_val.mul_i128(prod);
            }
        }
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

    /// Recursive helper to compute extended polynomial evaluations.
    /// Uses memoization to store intermediate results.
    fn get_extended_eval<const N: usize, const NUM_TERN_PTS: usize>(
        k_target: usize,
        binary_evals_input: &[i128],
        memoized_evals: &mut [Option<i128>; NUM_TERN_PTS], // Using array for memoization table
        ternary_point_info_table: &[TernaryPointInfo<N>; NUM_TERN_PTS],
    ) -> i128 {
        if let Some(val) = memoized_evals[k_target] {
            return val;
        }

        let point_info = ternary_point_info_table[k_target]; // This is a copy, which is fine for this small struct
        let result = if point_info.is_binary {
            binary_evals_input[point_info.binary_eval_idx]
        } else {
            // These recursive calls must use the same memoization table and info table
            let eval_at_1 = get_extended_eval(
                point_info.k_val_at_one,
                binary_evals_input,
                memoized_evals,
                ternary_point_info_table,
            );
            let eval_at_0 = get_extended_eval(
                point_info.k_val_at_zero,
                binary_evals_input,
                memoized_evals,
                ternary_point_info_table,
            );
            eval_at_1 - eval_at_0
        };

        memoized_evals[k_target] = Some(result);
        result
    }

    /// Performs in-place multilinear extension on ternary evaluation vectors
    /// and updates the temporary accumulator `temp_tA` with the product contribution.
    /// This is the generic version with no bound on num_svo_rounds.
    /// TODO: optimize this further
    #[inline]
    pub fn compute_and_update_tA_inplace<
        const NUM_SVO_ROUNDS: usize,
        const M_NON_BINARY_POINTS_CONST: usize, // num_non_binary_points(NUM_SVO_ROUNDS)
        const NUM_TERNARY_POINTS_CONST: usize,  // pow(3, NUM_SVO_ROUNDS)
        F: JoltField,
    >(
        binary_az_evals_input: &[i128], // Source of 2^N binary evals for Az
        binary_bz_evals_input: &[i128], // Source of 2^N binary evals for Bz
        e_in_val: &F,
        temp_tA: &mut [F], // Target for M_NON_BINARY_POINTS_CONST non-binary extended products
    ) {
        if NUM_SVO_ROUNDS == 0 {
            debug_assert!(
                temp_tA.is_empty(),
                "temp_tA should be empty for 0 SVO rounds"
            );
            // M_NON_BINARY_POINTS_CONST would be 0, NUM_TERNARY_POINTS_CONST would be 1.
            // The loops below would not run.
            return;
        }

        // Verify consistency of const generic arguments with NUM_SVO_ROUNDS
        debug_assert_eq!(
            M_NON_BINARY_POINTS_CONST,
            num_non_binary_points(NUM_SVO_ROUNDS),
            "M_NON_BINARY_POINTS_CONST mismatch"
        );
        debug_assert_eq!(
            NUM_TERNARY_POINTS_CONST,
            pow(3, NUM_SVO_ROUNDS),
            "NUM_TERNARY_POINTS_CONST mismatch"
        );

        let num_binary_points = 1 << NUM_SVO_ROUNDS;
        // temp_tA length is M_NON_BINARY_POINTS_CONST

        debug_assert_eq!(
            binary_az_evals_input.len(),
            num_binary_points,
            "binary_az_evals_input length mismatch"
        );
        debug_assert_eq!(
            binary_bz_evals_input.len(),
            num_binary_points,
            "binary_bz_evals_input length mismatch"
        );
        debug_assert_eq!(
            temp_tA.len(),
            M_NON_BINARY_POINTS_CONST, // temp_tA stores only non-binary point contributions
            "temp_tA length mismatch with M_NON_BINARY_POINTS_CONST"
        );

        // This table contains info for ALL 3^N points, used by the recursive helper.
        let ternary_point_info_table: [TernaryPointInfo<NUM_SVO_ROUNDS>; NUM_TERNARY_POINTS_CONST] =
            precompute_ternary_point_infos::<NUM_SVO_ROUNDS, NUM_TERNARY_POINTS_CONST>();

        // Memoization tables for all 3^N points.
        // Initialize with a value that signifies "not computed yet".
        // Using array for memoization tables.
        let mut memoized_az_evals: [Option<i128>; NUM_TERNARY_POINTS_CONST] =
            [None; NUM_TERNARY_POINTS_CONST];
        let mut memoized_bz_evals: [Option<i128>; NUM_TERNARY_POINTS_CONST] =
            [None; NUM_TERNARY_POINTS_CONST];

        // Get the map of non-binary points. The order here dictates temp_tA indexing.
        let y_ext_code_map_non_binary: [[SVOEvalPoint; NUM_SVO_ROUNDS]; M_NON_BINARY_POINTS_CONST] =
            build_y_ext_code_map::<NUM_SVO_ROUNDS, M_NON_BINARY_POINTS_CONST>();

        for i_temp_tA in 0..M_NON_BINARY_POINTS_CONST {
            let current_y_ext_coords_msb: &[SVOEvalPoint; NUM_SVO_ROUNDS] =
                &y_ext_code_map_non_binary[i_temp_tA];

            // Convert the MSB coordinates of the current non-binary point to its global k_ternary_idx
            let k_target_idx =
                svo_coords_msb_to_k_ternary_idx::<NUM_SVO_ROUNDS>(current_y_ext_coords_msb);

            // Sanity check: the k_target_idx must be non-binary.
            // The ternary_point_info_table can be indexed by k_target_idx.
            debug_assert!(
                k_target_idx < NUM_TERNARY_POINTS_CONST,
                "k_target_idx out of bounds for ternary_point_info_table"
            );
            debug_assert!(
                !ternary_point_info_table[k_target_idx].is_binary,
                "Target for temp_tA[{i_temp_tA}] (k={k_target_idx}) is unexpectedly binary.",
            );

            let az_val = get_extended_eval::<NUM_SVO_ROUNDS, NUM_TERNARY_POINTS_CONST>(
                k_target_idx,
                binary_az_evals_input,
                &mut memoized_az_evals,
                &ternary_point_info_table,
            );

            if az_val != 0 {
                let bz_val = get_extended_eval::<NUM_SVO_ROUNDS, NUM_TERNARY_POINTS_CONST>(
                    k_target_idx,
                    binary_bz_evals_input,
                    &mut memoized_bz_evals,
                    &ternary_point_info_table,
                );

                if bz_val != 0 {
                    let product = az_val
                        .checked_mul(bz_val)
                        .expect("Extended Az*Bz product overflow i128");
                    // Note: temp_tA is indexed by `i_temp_tA` which is the iteration order
                    // of non-binary points from build_y_ext_code_map.
                    temp_tA[i_temp_tA] += e_in_val.mul_i128(product);
                }
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
            return;
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
    use super::svo_helpers::*;
    use ark_bn254::Fr as TestField;
    use ark_ff::Zero;

    fn test_consistency_for_num_rounds(num_svo_rounds: usize) {
        let num_binary_points = 1 << num_svo_rounds;
        let num_temp_tA = num_non_trivial_ternary_points(num_svo_rounds);

        // Test with non-zero patterned data
        let binary_az_evals: Vec<i128> = (0..num_binary_points)
            .map(|i| (i + 1) as i128 * 2)
            .collect();
        let binary_bz_evals: Vec<i128> = (0..num_binary_points)
            .map(|i| (i + 2) as i128 * 3)
            .collect();
        let e_in_val = TestField::from(10u64);

        let mut temp_tA_new = vec![TestField::zero(); num_temp_tA];
        let mut temp_tA_old = vec![TestField::zero(); num_temp_tA];

        // Call the new general function (the one being tested)
        match num_svo_rounds {
            1 => compute_and_update_tA_inplace::<1, 1, 3, TestField>(
                &binary_az_evals,
                &binary_bz_evals,
                &e_in_val,
                &mut temp_tA_new,
            ),
            2 => compute_and_update_tA_inplace::<2, 5, 9, TestField>(
                &binary_az_evals,
                &binary_bz_evals,
                &e_in_val,
                &mut temp_tA_new,
            ),
            3 => compute_and_update_tA_inplace::<3, 19, 27, TestField>(
                &binary_az_evals,
                &binary_bz_evals,
                &e_in_val,
                &mut temp_tA_new,
            ),
            _ => panic!("Unsupported NUM_SVO_ROUNDS for this consistency test structure"),
        }

        // Call the old hardcoded function
        match num_svo_rounds {
            1 => compute_and_update_tA_inplace_1(
                &binary_az_evals,
                &binary_bz_evals,
                &e_in_val,
                &mut temp_tA_old,
            ),
            2 => compute_and_update_tA_inplace_2(
                &binary_az_evals,
                &binary_bz_evals,
                &e_in_val,
                &mut temp_tA_old,
            ),
            3 => compute_and_update_tA_inplace_3(
                &binary_az_evals,
                &binary_bz_evals,
                &e_in_val,
                &mut temp_tA_old,
            ),
            _ => panic!("Unsupported NUM_SVO_ROUNDS for this consistency test structure"),
        }
        assert_eq!(
            temp_tA_new, temp_tA_old,
            "Mismatch for NUM_SVO_ROUNDS = {} with patterned data",
            num_svo_rounds
        );

        // Test with all zeros for binary evals
        let binary_az_evals_zeros: Vec<i128> = vec![0; num_binary_points];
        let binary_bz_evals_zeros: Vec<i128> = vec![0; num_binary_points];
        let mut temp_tA_new_zeros = vec![TestField::zero(); num_temp_tA];
        let mut temp_tA_old_zeros = vec![TestField::zero(); num_temp_tA];

        match num_svo_rounds {
            1 => compute_and_update_tA_inplace::<1, 1, 3, TestField>(
                &binary_az_evals_zeros,
                &binary_bz_evals_zeros,
                &e_in_val,
                &mut temp_tA_new_zeros,
            ),
            2 => compute_and_update_tA_inplace::<2, 5, 9, TestField>(
                &binary_az_evals_zeros,
                &binary_bz_evals_zeros,
                &e_in_val,
                &mut temp_tA_new_zeros,
            ),
            3 => compute_and_update_tA_inplace::<3, 19, 27, TestField>(
                &binary_az_evals_zeros,
                &binary_bz_evals_zeros,
                &e_in_val,
                &mut temp_tA_new_zeros,
            ),
            _ => panic!("Unsupported NUM_SVO_ROUNDS for this consistency test structure"),
        }

        match num_svo_rounds {
            1 => compute_and_update_tA_inplace_1(
                &binary_az_evals_zeros,
                &binary_bz_evals_zeros,
                &e_in_val,
                &mut temp_tA_old_zeros,
            ),
            2 => compute_and_update_tA_inplace_2(
                &binary_az_evals_zeros,
                &binary_bz_evals_zeros,
                &e_in_val,
                &mut temp_tA_old_zeros,
            ),
            3 => compute_and_update_tA_inplace_3(
                &binary_az_evals_zeros,
                &binary_bz_evals_zeros,
                &e_in_val,
                &mut temp_tA_old_zeros,
            ),
            _ => panic!("Unsupported NUM_SVO_ROUNDS for this consistency test structure"),
        }
        assert_eq!(
            temp_tA_new_zeros, temp_tA_old_zeros,
            "Mismatch for NUM_SVO_ROUNDS = {} with zero data",
            num_svo_rounds
        );

        // Test with e_in_val = 0
        let e_in_val_zero = TestField::zero();
        let mut temp_tA_new_zero_e = vec![TestField::zero(); num_temp_tA];
        let mut temp_tA_old_zero_e = vec![TestField::zero(); num_temp_tA];

        match num_svo_rounds {
            1 => compute_and_update_tA_inplace::<1, 1, 3, TestField>(
                &binary_az_evals,
                &binary_bz_evals,
                &e_in_val_zero,
                &mut temp_tA_new_zero_e,
            ),
            2 => compute_and_update_tA_inplace::<2, 5, 9, TestField>(
                &binary_az_evals,
                &binary_bz_evals,
                &e_in_val_zero,
                &mut temp_tA_new_zero_e,
            ),
            3 => compute_and_update_tA_inplace::<3, 19, 27, TestField>(
                &binary_az_evals,
                &binary_bz_evals,
                &e_in_val_zero,
                &mut temp_tA_new_zero_e,
            ),
            _ => panic!("Unsupported NUM_SVO_ROUNDS for this consistency test structure"),
        }

        match num_svo_rounds {
            1 => compute_and_update_tA_inplace_1(
                &binary_az_evals,
                &binary_bz_evals,
                &e_in_val_zero,
                &mut temp_tA_old_zero_e,
            ),
            2 => compute_and_update_tA_inplace_2(
                &binary_az_evals,
                &binary_bz_evals,
                &e_in_val_zero,
                &mut temp_tA_old_zero_e,
            ),
            3 => compute_and_update_tA_inplace_3(
                &binary_az_evals,
                &binary_bz_evals,
                &e_in_val_zero,
                &mut temp_tA_old_zero_e,
            ),
            _ => panic!("Unsupported NUM_SVO_ROUNDS for this consistency test structure"),
        }
        assert_eq!(
            temp_tA_new_zero_e, temp_tA_old_zero_e,
            "Mismatch for NUM_SVO_ROUNDS = {} with zero e_in_val",
            num_svo_rounds
        );
        for val in temp_tA_new_zero_e {
            // Confirm they are all zero
            assert!(
                val.is_zero(),
                "temp_tA should be all zeros if e_in_val is zero"
            );
        }
    }

    #[test]
    fn test_compute_and_update_tA_inplace_vs_hardcoded_1() {
        test_consistency_for_num_rounds(1);
    }

    #[test]
    fn test_compute_and_update_tA_inplace_vs_hardcoded_2() {
        test_consistency_for_num_rounds(2);
    }

    #[test]
    fn test_compute_and_update_tA_inplace_vs_hardcoded_3() {
        test_consistency_for_num_rounds(3);
    }

    // --- Tests for distribute_tA_to_svo_accumulators ---

    fn test_distribute_consistency(num_svo_rounds: usize) {
        let num_tA = num_non_trivial_ternary_points(num_svo_rounds);
        let tA_accums: Vec<TestField> = (0..num_tA)
            .map(|i| TestField::from((i + 1) as u64 * 100))
            .collect();

        let x_out_val = 1; // Arbitrary x_out_val for testing E_out_vec indexing

        let mut E_out_vec: Vec<Vec<TestField>> = Vec::with_capacity(num_svo_rounds);
        for s_e in 0..num_svo_rounds {
            let num_suffix_vars = if num_svo_rounds > s_e + 1 {
                num_svo_rounds - (s_e + 1)
            } else {
                0
            };
            let num_e_entries_for_x_out = 1 << num_suffix_vars;
            let total_e_entries = (x_out_val + 10) * num_e_entries_for_x_out;
            let mut e_s: Vec<TestField> = Vec::with_capacity(total_e_entries);
            for i in 0..total_e_entries {
                e_s.push(TestField::from((s_e + 1) as u64 * 10 + (i + 1) as u64));
            }
            E_out_vec.push(e_s);
        }

        let num_zero = num_accums_eval_zero(num_svo_rounds);
        let num_infty = num_accums_eval_infty(num_svo_rounds);

        let mut accums_zero_new = vec![TestField::zero(); num_zero];
        let mut accums_infty_new = vec![TestField::zero(); num_infty];
        let mut accums_zero_old = vec![TestField::zero(); num_zero];
        let mut accums_infty_old = vec![TestField::zero(); num_infty];

        match num_svo_rounds {
            1 => {
                distribute_tA_to_svo_accumulators::<1, 1, TestField>(
                    &tA_accums,
                    x_out_val,
                    &E_out_vec,
                    &mut accums_zero_new,
                    &mut accums_infty_new,
                );
                distribute_tA_to_svo_accumulators_1(
                    &tA_accums,
                    x_out_val,
                    &E_out_vec,
                    &mut accums_zero_old,
                    &mut accums_infty_old,
                );
            }
            2 => {
                distribute_tA_to_svo_accumulators::<2, 5, TestField>(
                    &tA_accums,
                    x_out_val,
                    &E_out_vec,
                    &mut accums_zero_new,
                    &mut accums_infty_new,
                );
                distribute_tA_to_svo_accumulators_2(
                    &tA_accums,
                    x_out_val,
                    &E_out_vec,
                    &mut accums_zero_old,
                    &mut accums_infty_old,
                );
            }
            3 => {
                distribute_tA_to_svo_accumulators::<3, 19, TestField>(
                    &tA_accums,
                    x_out_val,
                    &E_out_vec,
                    &mut accums_zero_new,
                    &mut accums_infty_new,
                );
                distribute_tA_to_svo_accumulators_3(
                    &tA_accums,
                    x_out_val,
                    &E_out_vec,
                    &mut accums_zero_old,
                    &mut accums_infty_old,
                );
            }
            _ => panic!("Unsupported NUM_SVO_ROUNDS for distribute consistency test"),
        }

        assert_eq!(
            accums_zero_new, accums_zero_old,
            "accums_zero mismatch for N={}",
            num_svo_rounds
        );
        assert_eq!(
            accums_infty_new, accums_infty_old,
            "accums_infty mismatch for N={}",
            num_svo_rounds
        );
    }

    #[test]
    fn test_distribute_tA_vs_hardcoded_1() {
        test_distribute_consistency(1);
    }

    #[test]
    fn test_distribute_tA_vs_hardcoded_2() {
        test_distribute_consistency(2);
    }

    #[test]
    fn test_distribute_tA_vs_hardcoded_3() {
        test_distribute_consistency(3);
    }
}
