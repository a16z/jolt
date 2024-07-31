// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

import {MODULUS, Fr, FrLib} from "./Fr.sol";
import "forge-std/console.sol";

/// Allows calculation of the R1CS step matrix
library R1CSMatrix {
    using FrLib for Fr;

    // These constants being changed may break the logic in some areas as the assumption of their size or bit composition is built in
    uint256 constant COL_BITS = 7;
    uint256 constant ROW_BITS = 7;
    uint256 constant SEGMENT_LENGTH = 89;

    /// Evaluates the z mle for the outer sumcheck
    /// @param r The r we are evaluating on
    /// @param segmentEvals The segment evals from the proof, should be length 89
    function eval_z_mle(Fr[] memory r, uint256[] memory segmentEvals) internal pure returns (Fr) {
        require(segmentEvals.length == SEGMENT_LENGTH, "Incorrect segment length");
        // We calculate the evals based on a length 7 slice
        Fr[] memory r_var_eqs = eq_poly_evals(r, 1, ROW_BITS);
        Fr sum = Fr.wrap(0);
        for (uint256 i = 0; i < SEGMENT_LENGTH; i++) {
            sum = sum + r_var_eqs[i] * FrLib.from(segmentEvals[i]);
        }
        // In the rust we do let const_poly = SparsePolynomial::new(self.num_vars_total().log_2(), vec![(F::one(), 0)]); let eval_const = const_poly.evaluate(r_rest);
        // But because we presume self.num_vars_total().log_2() = 7 and the vec only has one element we just have to run compute chis on the bit vec of zero and r
        // Which simplifies to this for loop:
        Fr prod = Fr.wrap(1);
        for (uint256 i = 0; i < r.length - 1; i++) {
            // Every bit of the bit decomp of zero is zero and so we just have to do this on r rest
            prod = prod * (Fr.wrap(1) - r[1 + i]);
        }

        return ((Fr.wrap(1) - r[0]) * sum + r[0] * prod);
    }

    /// Evaluates our stepwise low memory representation of the r1cs using a constant fixed matrix of constraints.
    /// NOTE - Changing the R1CS in jolt's rust will break this function and new constant and offset calculations must
    ///        be calculated for the A, B and C functions
    /// @param r Our typechecked field element array which is log the size of the jolt trace
    /// @param row_bits The log of the number of rows of our global r1cs
    /// @param col_bits The log of the number of cols of our global r1cs
    // @param total_cols The total number of columns in our r1cs used for our col bit vector.
    function evaluate_r1cs_matrix_mles(Fr[] memory r, uint256 row_bits, uint256 col_bits, uint256)
        internal
        pure
        returns (Fr, Fr, Fr)
    {
        uint256 constraint_row_bits = ROW_BITS;
        uint256 constraint_col_bits = COL_BITS;
        //uint256 step_bits = row_bits - constraint_row_bits;

        // Do an eq pol eval on the parts of r which are not in the first 7 bits of the row and col sub vector
        Fr eq_rx_ry_step = eq_poly_evaluate(
            r,
            constraint_row_bits,
            row_bits - constraint_row_bits,
            r,
            row_bits + constraint_col_bits + 1,
            col_bits - constraint_col_bits
        );
        Fr[] memory eq_poly_row = eq_poly_evals(r, 0, constraint_row_bits);
        // Creates a 256 field element allocation but doesn't need it?
        Fr[] memory eq_poly_col = eq_poly_evals(r, row_bits, constraint_col_bits + 1);

        // Does the eval of the constraint row const via a bit vec
        Fr col_eq_constant = Fr.wrap(0);
        // Fr eq_plus_one_eval = eq_plus_one(r, constraint_row_bits, r, row_bits + constraint_col_bits + 1, step_bits);

        // Fr non_uni_constraint_index = Fr.wrap(0);

        return (
            A(eq_poly_row, eq_poly_col, eq_rx_ry_step, col_eq_constant),
            B(eq_poly_row, eq_poly_col, eq_rx_ry_step, col_eq_constant),
            C(eq_poly_row, eq_poly_col, eq_rx_ry_step, col_eq_constant)
        );
    }

    // The same as rust's EqPolynomial(eq).evaluate(r)
    function eq_poly_evaluate(
        Fr[] memory eq,
        uint256 eq_start,
        uint256 eq_len,
        Fr[] memory r,
        uint256 r_start,
        uint256 r_len
    ) internal pure returns (Fr) {
        assert(r_len == eq_len);
        Fr ret = Fr.wrap(1);
        for (uint256 i = 0; i < r_len; i++) {
            ret = ret
                * (eq[eq_start + i] * r[r_start + i] + (Fr.wrap(1) - r[r_start + i]) * (Fr.wrap(1) - eq[eq_start + i]));
        }
        return (ret);
    }

    // The same as rust's EqPolynomial(eq).evaluate(r) but with r defined as the bits of an r_in_bits
    function eval_poly_evaluate_bitvec(Fr[] memory eq, uint256 eq_start, uint256 eq_len, uint256 r_in_bits)
        internal
        pure
        returns (Fr)
    {
        Fr ret = Fr.wrap(1);
        uint256 extracted = r_in_bits;
        for (uint256 i = 0; i < eq_len; i++) {
            Fr bit = Fr.wrap(extracted & 1);
            extracted = extracted >> 1;
            ret = ret * (eq[eq_start + i] * bit + (Fr.wrap(1) - bit) * (Fr.wrap(1) - eq[eq_start + i]));
        }
        return (ret);
    }

    // Transcribed version of this function from rust, which has not be evaluated for efficiency
    function eq_poly_evals(Fr[] memory eq, uint256 start, uint256 eq_len) internal pure returns (Fr[] memory) {
        uint256 ell = 2 ** eq_len;
        Fr[] memory evals = new Fr[](ell);
        for (uint256 i = 0; i < ell; i++) {
            evals[i] = Fr.wrap(1);
        }
        uint256 size = 1;
        for (uint256 j = 0; j < eq_len; j++) {
            size *= 2;
            for (uint256 i = size - 1;; i = i - 2) {
                Fr scalar = evals[i / 2];
                evals[i] = scalar * eq[start + j];
                evals[i - 1] = scalar - evals[i];
                if (i <= 1) {
                    break;
                }
            }
        }
        return evals;
    }

    function eq_plus_one(Fr[] memory x, uint256 x_start, Fr[] memory y, uint256 y_start, uint256 length)
        internal
        pure
        returns (Fr)
    {
        // Firstly we evaluate the upper bit products
        Fr running_upper_product = Fr.wrap(1);
        Fr[] memory upper = new Fr[](length);
        for (uint256 i = length - 1;; i = i - 1) {
            Fr new_prod =
                x[x_start + i] * y[y_start + i] * (Fr.wrap(1) - x[x_start + i]) * (Fr.wrap(1) - y[y_start + i]);
            running_upper_product = running_upper_product * new_prod;
            upper[i] = running_upper_product;
            if (i == 0) {
                break;
            }
        }

        // Now we can compute the lower bit product and current bit product by just working in order
        Fr running_lower_product = Fr.wrap(1);
        Fr sum = Fr.wrap(0);
        for (uint256 i = 0; i < length; i++) {
            Fr new_prod = x[x_start + length - 1 - i] * (Fr.wrap(1) - y[y_start + length - 1 - i]);
            running_lower_product = running_lower_product * new_prod;
            Fr current_bit = y[y_start + length - 1 - i] * (Fr.wrap(1) - x[x_start + length - 1 - i]);
            sum = sum + running_lower_product * upper[i] * current_bit;
        }

        return (sum);
    }

    /// Computes the sparse MLE evaluation at the step matrix of the r1cs#
    /// @param row The relevant row values
    /// @param col The relevant row values
    /// @param eq_rx_ry_step The const for the var steps
    /// @param col_eq_constant The const for the constant evals
    function A(Fr[] memory row, Fr[] memory col, Fr eq_rx_ry_step, Fr col_eq_constant) internal pure returns (Fr) {
        Fr running = Fr.wrap(0);
        Fr rv = Fr.wrap(0);
        rv = rv + row[0] * col[34];
        rv = rv + row[1] * col[35];
        rv = rv + row[2] * col[36];
        rv = rv + row[3] * col[37];
        rv = rv + row[4] * col[38];
        rv = rv + row[5] * col[39];
        rv = rv + row[6] * col[40];
        rv = rv + row[7] * col[41];
        rv = rv + row[8] * col[42];
        rv = rv + row[9] * col[43];
        rv = rv + row[10] * col[44];
        rv = rv + row[11] * col[45];
        rv = rv + row[12] * col[46];
        rv = rv + row[13] * col[47];
        rv = rv + row[14] * col[48];
        rv = rv + row[15] * col[49];
        rv = rv + row[16] * col[50];
        rv = rv + row[17] * col[51];
        rv = rv + row[18] * col[52];
        rv = rv + row[19] * col[53];
        rv = rv + row[20] * col[54];
        rv = rv + row[21] * col[55];
        rv = rv + row[22] * col[56];
        rv = rv + row[23] * col[57];
        rv = rv + row[24] * col[58];
        rv = rv + row[25] * col[59];
        rv = rv + row[26] * col[60];
        rv = rv + row[27] * col[61];
        rv = rv + row[28] * col[62];
        rv = rv + row[29] * col[63];
        rv = rv + row[30] * col[64];
        rv = rv + row[31] * col[65];
        rv = rv + row[32] * col[66];
        rv = rv + row[33] * col[67];
        rv = rv + row[34] * col[68];
        rv = rv + row[35] * col[69];
        rv = rv + row[36] * col[70];
        rv = rv + row[37] * col[71];
        rv = rv + row[38] * col[0];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[38] * col[1];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[39] * col[3];
        rv = rv + Fr.wrap(0x2000000000) * row[39] * col[34];
        rv = rv + Fr.wrap(0x1000000000) * row[39] * col[35];
        rv = rv + Fr.wrap(0x0800000000) * row[39] * col[36];
        rv = rv + Fr.wrap(0x0400000000) * row[39] * col[37];
        rv = rv + Fr.wrap(0x0200000000) * row[39] * col[38];
        rv = rv + Fr.wrap(0x0100000000) * row[39] * col[39];
        rv = rv + Fr.wrap(0x0080000000) * row[39] * col[40];
        rv = rv + Fr.wrap(0x0040000000) * row[39] * col[41];
        rv = rv + Fr.wrap(0x0020000000) * row[39] * col[42];
        rv = rv + Fr.wrap(0x0010000000) * row[39] * col[43];
        rv = rv + Fr.wrap(0x0008000000) * row[39] * col[44];
        rv = rv + Fr.wrap(0x0004000000) * row[39] * col[45];
        rv = rv + Fr.wrap(0x0002000000) * row[39] * col[46];
        rv = rv + Fr.wrap(0x0001000000) * row[39] * col[47];
        rv = rv + Fr.wrap(0x0000800000) * row[39] * col[48];
        rv = rv + Fr.wrap(0x0000400000) * row[39] * col[49];
        rv = rv + Fr.wrap(0x0000200000) * row[39] * col[50];
        rv = rv + Fr.wrap(0x0000100000) * row[39] * col[51];
        rv = rv + Fr.wrap(0x0000080000) * row[39] * col[52];
        rv = rv + Fr.wrap(0x0000040000) * row[39] * col[53];
        rv = rv + Fr.wrap(0x0000020000) * row[39] * col[54];
        rv = rv + Fr.wrap(0x0000010000) * row[39] * col[55];
        rv = rv + Fr.wrap(0x0000008000) * row[39] * col[56];
        rv = rv + Fr.wrap(0x0000004000) * row[39] * col[57];
        rv = rv + Fr.wrap(0x0000002000) * row[39] * col[58];
        rv = rv + Fr.wrap(0x0000001000) * row[39] * col[59];
        rv = rv + Fr.wrap(0x0000000800) * row[39] * col[60];
        rv = rv + Fr.wrap(0x0000000400) * row[39] * col[61];
        rv = rv + Fr.wrap(0x0000000200) * row[39] * col[62];
        rv = rv + Fr.wrap(0x0000000100) * row[39] * col[63];
        rv = rv + Fr.wrap(0x0000000080) * row[39] * col[64];
        rv = rv + Fr.wrap(0x0000000040) * row[39] * col[65];
        rv = rv + Fr.wrap(0x0000000020) * row[39] * col[66];
        rv = rv + Fr.wrap(0x0000000010) * row[39] * col[67];
        rv = rv + Fr.wrap(0x0000000008) * row[39] * col[68];
        rv = rv + Fr.wrap(0x0000000004) * row[39] * col[69];
        rv = rv + Fr.wrap(0x0000000002) * row[39] * col[70];
        rv = rv + row[39] * col[71];
        rv = rv + row[40] * col[34];
        rv = rv + row[41] * col[35];
        rv = rv + row[42] * col[41];
        rv = rv + row[43] * col[36];
        rv = rv + row[43] * col[37];
        rv = rv + row[44] * col[36];
        rv = rv + row[45] * col[36];
        rv = rv + row[46] * col[36];
        rv = rv + row[47] * col[36];
        rv = rv + row[48] * col[17];
        rv = rv + Fr.wrap(0x0000000100) * row[48] * col[18];
        rv = rv + Fr.wrap(0x0000010000) * row[48] * col[19];
        rv = rv + Fr.wrap(0x0001000000) * row[48] * col[20];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[48] * col[75];
        rv = rv + row[49] * col[37];
        rv = rv + Fr.wrap(0x0000000000000000000000000000000000000000000000000001000000000000) * row[50] * col[29];
        rv = rv + Fr.wrap(0x0100000000) * row[50] * col[30];
        rv = rv + Fr.wrap(0x0000010000) * row[50] * col[31];
        rv = rv + row[50] * col[32];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[50] * col[76];
        rv = rv + row[51] * col[45];
        rv = rv + row[52] * col[46];
        rv = rv + row[53] * col[36];
        rv = rv + row[54] * col[37];
        rv = rv + Fr.wrap(0x0001000000) * row[55] * col[21];
        rv = rv + Fr.wrap(0x0000010000) * row[55] * col[22];
        rv = rv + Fr.wrap(0x0000000100) * row[55] * col[23];
        rv = rv + row[55] * col[24];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[55] * col[77];
        rv = rv + Fr.wrap(0x0001000000) * row[56] * col[25];
        rv = rv + Fr.wrap(0x0000010000) * row[56] * col[26];
        rv = rv + Fr.wrap(0x0000000100) * row[56] * col[27];
        rv = rv + row[56] * col[28];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[56] * col[78];
        rv = rv + row[57] * col[42];
        rv = rv + row[58] * col[42];
        rv = rv + row[59] * col[61];
        rv = rv + row[59] * col[62];
        rv = rv + row[59] * col[63];
        rv = rv + row[60] * col[42];
        rv = rv + row[61] * col[61];
        rv = rv + row[61] * col[62];
        rv = rv + row[61] * col[63];
        rv = rv + row[62] * col[42];
        rv = rv + row[63] * col[61];
        rv = rv + row[63] * col[62];
        rv = rv + row[63] * col[63];
        rv = rv + row[64] * col[42];
        rv = rv + row[65] * col[61];
        rv = rv + row[65] * col[62];
        rv = rv + row[65] * col[63];
        rv = rv + row[66] * col[42];
        rv = rv + row[67] * col[6];
        rv = rv + row[68] * col[83];
        rv = rv + row[69] * col[6];
        rv = rv + row[70] * col[84];
        rv = rv + row[71] * col[39];
        rv = rv + row[72] * col[38];
        rv = rv + row[73] * col[85];
        running = running + rv * eq_rx_ry_step;
        // then we do constant col
        Fr rc = Fr.wrap(0);
        running = running + rc * col_eq_constant;
        return (running);
    }

    /// Computes the sparse MLE evaluation at the step matrix of the r1cs#
    /// @param row The relevant row values
    /// @param col The relevant row values
    /// @param eq_rx_ry_step The const for the var steps
    /// @param col_eq_constant The const for the constant evals
    function B(Fr[] memory row, Fr[] memory col, Fr eq_rx_ry_step, Fr col_eq_constant) internal pure returns (Fr) {
        Fr running = Fr.wrap(0);
        Fr rv = Fr.wrap(0);
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[0] * col[34];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[1] * col[35];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[2] * col[36];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[3] * col[37];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[4] * col[38];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[5] * col[39];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[6] * col[40];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[7] * col[41];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[8] * col[42];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[9] * col[43];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[10] * col[44];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[11] * col[45];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[12] * col[46];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[13] * col[47];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[14] * col[48];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[15] * col[49];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[16] * col[50];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[17] * col[51];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[18] * col[52];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[19] * col[53];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[20] * col[54];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[21] * col[55];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[22] * col[56];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[23] * col[57];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[24] * col[58];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[25] * col[59];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[26] * col[60];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[27] * col[61];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[28] * col[62];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[29] * col[63];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[30] * col[64];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[31] * col[65];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[32] * col[66];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[33] * col[67];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[34] * col[68];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[35] * col[69];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[36] * col[70];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[37] * col[71];
        rv = rv + Fr.wrap(0x0000000004) * row[40] * col[0];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[40] * col[9];
        rv = rv + row[41] * col[7];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[41] * col[10];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[43] * col[8];
        rv = rv + row[43] * col[9];
        rv = rv + row[43] * col[74];
        rv = rv + row[44] * col[12];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[44] * col[17];
        rv = rv + row[45] * col[13];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[45] * col[18];
        rv = rv + row[46] * col[14];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[46] * col[19];
        rv = rv + row[47] * col[15];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[47] * col[20];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[49] * col[33];
        rv = rv + row[49] * col[75];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[51] * col[72];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[51] * col[73];
        rv = rv + row[51] * col[76];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[52] * col[72];
        rv = rv + row[52] * col[73];
        rv = rv + row[52] * col[76];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[53] * col[75];
        rv = rv + row[53] * col[76];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[54] * col[10];
        rv = rv + row[54] * col[76];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[57] * col[72];
        rv = rv + row[57] * col[77];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[58] * col[73];
        rv = rv + row[58] * col[78];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[59] * col[25];
        rv = rv + row[59] * col[28];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffff01) * row[60] * col[21];
        rv = rv + row[60] * col[29];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[60] * col[79];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[61] * col[26];
        rv = rv + row[61] * col[28];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffff01) * row[62] * col[22];
        rv = rv + row[62] * col[30];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[62] * col[80];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[63] * col[27];
        rv = rv + row[63] * col[28];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffff01) * row[64] * col[23];
        rv = rv + row[64] * col[31];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[64] * col[81];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffff01) * row[66] * col[24];
        rv = rv + row[66] * col[32];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[66] * col[82];
        rv = rv + row[67] * col[40];
        rv = rv + row[68] * col[16];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[68] * col[33];
        rv = rv + row[69] * col[38];
        rv = rv + row[70] * col[0];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[70] * col[16];
        rv = rv + row[71] * col[33];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593effffffd) * row[72] * col[0];
        rv = rv + row[72] * col[33];
        rv = rv + Fr.wrap(0x0000000004) * row[73] * col[0];
        rv = rv + row[73] * col[74];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[73] * col[86];
        running = running + rv * eq_rx_ry_step;
        // then we do constant col
        Fr rc = Fr.wrap(0);
        rc = rc + row[0];
        rc = rc + row[1];
        rc = rc + row[2];
        rc = rc + row[3];
        rc = rc + row[4];
        rc = rc + row[5];
        rc = rc + row[6];
        rc = rc + row[7];
        rc = rc + row[8];
        rc = rc + row[9];
        rc = rc + row[10];
        rc = rc + row[11];
        rc = rc + row[12];
        rc = rc + row[13];
        rc = rc + row[14];
        rc = rc + row[15];
        rc = rc + row[16];
        rc = rc + row[17];
        rc = rc + row[18];
        rc = rc + row[19];
        rc = rc + row[20];
        rc = rc + row[21];
        rc = rc + row[22];
        rc = rc + row[23];
        rc = rc + row[24];
        rc = rc + row[25];
        rc = rc + row[26];
        rc = rc + row[27];
        rc = rc + row[28];
        rc = rc + row[29];
        rc = rc + row[30];
        rc = rc + row[31];
        rc = rc + row[32];
        rc = rc + row[33];
        rc = rc + row[34];
        rc = rc + row[35];
        rc = rc + row[36];
        rc = rc + row[37];
        rc = rc + row[38];
        rc = rc + row[39];
        rc = rc + Fr.wrap(0x007ffffffc) * row[40];
        rc = rc + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f592f0000001) * row[42];
        rc = rc + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f59370004001) * row[43];
        rc = rc + row[48];
        rc = rc + row[50];
        rc = rc + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f592f0000001) * row[52];
        rc = rc + row[55];
        rc = rc + row[56];
        rc = rc + Fr.wrap(0x007ffffffc) * row[70];
        rc = rc + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f59370000001) * row[72];
        rc = rc + Fr.wrap(0x0080000000) * row[73];
        running = running + rc * col_eq_constant;
        return (running);
    }

    /// Computes the sparse MLE evaluation at the step matrix of the r1cs#
    /// @param row The relevant row values
    /// @param col The relevant row values
    /// @param eq_rx_ry_step The const for the var steps
    /// @param col_eq_constant The const for the constant evals
    function C(Fr[] memory row, Fr[] memory col, Fr eq_rx_ry_step, Fr col_eq_constant) internal pure returns (Fr) {
        Fr running = Fr.wrap(0);
        Fr rv = Fr.wrap(0);
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[40] * col[9];
        rv = rv + row[40] * col[72];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[41] * col[10];
        rv = rv + row[41] * col[73];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[42] * col[7];
        rv = rv + row[42] * col[74];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[59] * col[25];
        rv = rv + row[59] * col[79];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[61] * col[26];
        rv = rv + row[61] * col[80];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[63] * col[27];
        rv = rv + row[63] * col[81];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[65] * col[28];
        rv = rv + row[65] * col[82];
        rv = rv + row[67] * col[83];
        rv = rv + row[69] * col[84];
        rv = rv + row[71] * col[85];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593effffffd) * row[72] * col[0];
        rv = rv + row[72] * col[86];
        rv = rv + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000) * row[73] * col[86];
        rv = rv + row[73] * col[87];
        running = running + rv * eq_rx_ry_step;
        // then we do constant col
        Fr rc = Fr.wrap(0);
        rc = rc + Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f5936ffffffd) * row[72];
        running = running + rc * col_eq_constant;

        return (running);
    }
}
