pragma circom 2.2.1;
include "./spartan_buses.circom";
include "./../../../fields/non_native/utils.circom";
include "./../utils.circom";
include "./../jolt1_buses.circom";

template VerifyR1CS(num_commitments, num_steps, num_cons_total, num_vars, num_rows) {
   
    var num_rounds_x = log2(num_cons_total);
    var num_var_next_power2 =  NextPowerOf2(num_vars);
    var num_cols = 2 * num_steps * num_var_next_power2;
    var num_rounds_y = log2(num_cols); 

    input R1CSProof(num_rounds_x, num_rounds_y, num_commitments) r1cs_proof;
    input Transcript() transcript;
    output Transcript() up_transcript;

    Transcript() int_transcript[6];   
    
    signal tau[num_rounds_x];
    signal claim_outer_final, claim_inner_final;
    signal r_x[num_rounds_x], inner_sumcheck_r[num_rounds_y];
    signal r_inner_sumcheck_RLC;


    (int_transcript[0], tau) <== ChallengeVector(num_rounds_x)(transcript);
    (int_transcript[1], claim_outer_final, r_x) <== SumCheck(num_rounds_x, 3)(0, r1cs_proof.proof.outer_sumcheck_proof, int_transcript[0]);
   
    signal rev_r_x[num_rounds_x] <== ReverseVector(num_rounds_x)(r_x);

    signal taus_bound_rx <== EvaluateEq(num_rounds_x)(tau, rev_r_x);
    signal claim_Az_Bz <== (r1cs_proof.proof.outer_sumcheck_claims[0] *  r1cs_proof.proof.outer_sumcheck_claims[1]);
    signal claim_Az_Bz_negCz <== (claim_Az_Bz -  r1cs_proof.proof.outer_sumcheck_claims[2]);
    signal claim_outer_final_expected <== (taus_bound_rx *  claim_Az_Bz_negCz);
    
    claim_outer_final_expected === claim_outer_final;

    int_transcript[2] <== AppendScalars(3)(r1cs_proof.proof.outer_sumcheck_claims, int_transcript[1]);

    (int_transcript[3], r_inner_sumcheck_RLC) <== ChallengeScalar()(int_transcript[2]);

     signal claim_inner_joint <== EvalUniPoly(2)(r1cs_proof.proof.outer_sumcheck_claims, r_inner_sumcheck_RLC);
    (int_transcript[4], claim_inner_final, inner_sumcheck_r) <== SumCheck(num_rounds_y, 2)(claim_inner_joint, r1cs_proof.proof.inner_sumcheck_proof, int_transcript[3]);

     var num_var_next_power2_log2 = log2(num_var_next_power2);
     var n_prefix = num_var_next_power2_log2 + 1;
     signal eval_Z <== EvaluateZMle(num_rounds_y, num_commitments, num_var_next_power2_log2)(inner_sumcheck_r, r1cs_proof.proof.claimed_witness_evals);
   
    signal r[num_rounds_x + num_rounds_y] <== Concatenate(num_rounds_x, num_rounds_y)(rev_r_x, inner_sumcheck_r);
    signal eval_a_b_c[3] <== EvaluateR1CSMatrixMles(num_rounds_x, num_rounds_y, num_cons_total, num_steps, num_rows, num_var_next_power2_log2, num_cols)(r);

    signal left_expected <== EvalUniPoly(2)(eval_a_b_c, r_inner_sumcheck_RLC);
    signal claim_inner_final_expected <== (left_expected *  eval_Z);

    claim_inner_final === claim_inner_final_expected;

    signal r_y_point[num_rounds_y - n_prefix] <== TruncateVec(n_prefix, num_rounds_y, num_rounds_y)(inner_sumcheck_r);
    output VerifierOpening(num_rounds_y - n_prefix) r1cs_opening;
    (up_transcript, r1cs_opening) <== OpeningAccumulator(num_commitments, num_rounds_y - n_prefix)(r_y_point, r1cs_proof.proof.claimed_witness_evals, int_transcript[4]);
}

template Concatenate(size1, size2) {
    input signal vec1[size1];
    input signal vec2[size2];
    output signal out[size1 + size2];
    
    for (var i = 0; i < size1; i++){
        out[i] <== vec1[i];
    }

    for (var i = 0; i < size2; i++){
        out[i + size1] <== vec2[i];
    }
}

template EvaluateZMle(r_len, num_evals, var_bits) {    
    input signal r[r_len];
    input signal segment_evals[num_evals];
    output signal eval_Z;

    // Z can be computed in two halves, [Variables, (constant) 1, 0 , ...] indexed by the first bit.
    signal r_const <== r[0];
    signal r_rest[r_len - 1] <== TruncateVec(1, r_len, r_len)(r);
    signal r_var[var_bits] <== TruncateVec(0, var_bits, r_len - 1)(r_rest);
    

    signal r_var_eq[1 << var_bits] <== Evals(var_bits)(r_var);
   

    signal tr_r_var_eq[num_evals] <== TruncateVec(0, num_evals, 1 << var_bits)(r_var_eq);

    signal eval_variables <== ComputeDotProduct(num_evals)(segment_evals, tr_r_var_eq);
   
  
    signal eval_const <== EvaluateConstPoly(r_len - 1)(r_rest);
    signal temp1 <== (1 -  r_const); 
    signal temp2 <== (temp1 *  eval_variables); 
    signal temp3 <== (r_const *  eval_const); 
  
    eval_Z <== (temp2 + temp3);
}


template EvaluateConstPoly(size){
    input signal r[size];
    output signal eval;

    signal int_eval[size];
    int_eval[0] <==  1 - r[0];

    for (var i = 1; i < size; i++){
        int_eval[i] <== (int_eval[i - 1]  * (1 - r[i]));
    }
    eval <== int_eval[size - 1];
}





template EvaluateR1CSMatrixMles(total_rows_bits, total_cols_bits, num_cons_total, num_steps, num_rows, uniform_cols_bits, num_cols){
    var r_size = total_rows_bits + total_cols_bits;

    input signal r[r_size];
    output signal eval_a_b_c[3];

    var steps_bits = log2(num_steps);
    var temp1 = NextPowerOf2(num_rows + 1);
    var constraint_rows_bits = log2(temp1);

 
    signal r_row[total_rows_bits], r_col[total_cols_bits];
    (r_row, r_col) <== SplitAt(total_rows_bits, r_size)(r); 
   
    signal r_row_constr[constraint_rows_bits], r_row_step[total_rows_bits - constraint_rows_bits];
    (r_row_constr, r_row_step) <== SplitAt(constraint_rows_bits, total_rows_bits)(r_row); 

    signal r_col_var[uniform_cols_bits + 1], r_col_step[total_cols_bits - uniform_cols_bits - 1];
    (r_col_var, r_col_step) <== SplitAt(uniform_cols_bits + 1, total_cols_bits)(r_col); 
    
    signal eq_rx_ry_step <== EvaluateEq(total_rows_bits - constraint_rows_bits)(r_row_step, r_col_step);
 

    signal eq_rx_constr[1 << constraint_rows_bits] <== Evals(constraint_rows_bits)(r_row_constr);
    signal eq_ry_var[1 << (uniform_cols_bits + 1)] <== Evals(uniform_cols_bits + 1)(r_col_var);
    
    signal constant_column[total_cols_bits] <== IndexToFieldBitvector(num_cols / 2, total_cols_bits)();
    signal col_eq_constant <== EvaluateEq(total_cols_bits)(r_col, constant_column);



    var len = 2;
    signal int_a_mle[len + 1];
    signal int_b_mle[len + 1];

    int_a_mle[0] <== ComputeUniformMatrixMleA(1 << constraint_rows_bits, 1 << (uniform_cols_bits + 1))(eq_rx_ry_step,eq_rx_constr,eq_ry_var);
    int_b_mle[0] <== ComputeUniformMatrixMleB(1 << constraint_rows_bits, 1 << (uniform_cols_bits + 1))(eq_rx_ry_step,col_eq_constant, eq_rx_constr,eq_ry_var);
    eval_a_b_c[2] <== ComputeUniformMatrixMleC(1 << constraint_rows_bits, 1 << (uniform_cols_bits + 1))(eq_rx_ry_step,col_eq_constant, eq_rx_constr,eq_ry_var);

    signal eq_step_offset_1 <== EqPlusOne(total_rows_bits - constraint_rows_bits)(r_row_step, r_col_step);
    
    signal non_uni_a[len];
    signal non_uni_b[len];
    signal non_uni_constraint_index[len][constraint_rows_bits];
    signal row_constr_eq_non_uni[len];
    signal int_mul[2 * len];

    for (var i = 0; i < len; i++) {
        non_uni_a[i] <== ComputeNonUniformEq(i, 1 << constraint_rows_bits, 1 << (uniform_cols_bits + 1))
                                            (eq_rx_ry_step, eq_step_offset_1, eq_rx_constr, eq_ry_var, col_eq_constant);
                                         
        non_uni_b[i] <== ComputeNonUniformCondition(i, 1 << constraint_rows_bits, 1 << (uniform_cols_bits + 1))
                                                    (eq_rx_ry_step, eq_step_offset_1,eq_rx_constr,eq_ry_var,col_eq_constant);

        non_uni_constraint_index[i] <== IndexToFieldBitvector(num_rows + i, constraint_rows_bits)();

        row_constr_eq_non_uni[i] <== EvaluateEq(constraint_rows_bits)(r_row_constr, non_uni_constraint_index[i]);

        row_constr_eq_non_uni[i] === eq_rx_constr[num_rows + i];

        int_mul[2 * i] <== (non_uni_a[i] *  row_constr_eq_non_uni[i] );
        int_mul[2 * i + 1] <== (non_uni_b[i] *  row_constr_eq_non_uni[i]);

        int_a_mle[i + 1] <== (int_a_mle[i] +  int_mul[2 * i]);
        int_b_mle[i + 1] <== (int_b_mle[i] +  int_mul[2 * i + 1]);
    }
    eval_a_b_c[0] <== int_a_mle[len];
    eval_a_b_c[1] <== int_b_mle[len];
}


/* This MLE is 1 if y = x + 1 for x in the range [0... 2^l-2].
That is, it ignores the case where x is all 1s, outputting 0.
Assumes x and y are provided big-endian. */
template EqPlusOne(l) {
    input signal x[l];          
    input signal y[l];          
    output signal eval;      

    signal lower_bits_product[l][l + 1];  
    signal kth_bit_product[l];    
    signal higher_bits_product[l][l]; 
    signal int_sum[l + 1], int_mul[2 * l];            

    signal lower_bits_product_int_sub[l][l][1];
    signal lower_bits_product_int_mul[l][l][1];
    signal kth_bit_product_int_sub[l];
    signal higher_bits_product_int_sub[l][l][3]; 
    signal higher_bits_product_int_mul[l][l][2];
    signal one <== 1;
  

    int_sum[0] <== 0;

    for (var k = 0; k < l; k++) {
        
        lower_bits_product[k][0] <== one;
        for (var i = 0; i < k; i++) {
            lower_bits_product_int_sub[k][i][0] <== (one - y[l - 1 - i]);
            lower_bits_product_int_mul[k][i][0] <== (x[l - 1 - i]  *  lower_bits_product_int_sub[k][i][0]);
            lower_bits_product[k][i + 1] <== (lower_bits_product[k][i] *  lower_bits_product_int_mul[k][i][0]);
        }

        kth_bit_product_int_sub[k] <== (one -   x[l - 1 - k]);
        kth_bit_product[k] <== (kth_bit_product_int_sub[k] *  y[l - 1 - k]);

        higher_bits_product[k][k] <== one;

        for (var i = k + 1; i < l; i++) {
            higher_bits_product_int_sub[k][i][0] <== (one -  x[l - 1 - i]);
            higher_bits_product_int_sub[k][i][1] <== (one -  y[l - 1 - i]);
            higher_bits_product_int_mul[k][i][0] <== (higher_bits_product_int_sub[k][i][0]*  higher_bits_product_int_sub[k][i][1]);
            higher_bits_product_int_mul[k][i][1] <== (x[l - 1 - i] *  y[l - 1 - i]);
            higher_bits_product_int_sub[k][i][2] <== (higher_bits_product_int_mul[k][i][0] + higher_bits_product_int_mul[k][i][1]);
            higher_bits_product[k][i] <== (higher_bits_product[k][i - 1] *  higher_bits_product_int_sub[k][i][2]);
        }
        
        int_mul[2 * k] <== (kth_bit_product[k] *  higher_bits_product[k][l - 1]);
        int_mul[2 * k + 1] <==  (lower_bits_product[k][k] *   int_mul[2 * k]);
        int_sum[k + 1] <== (int_sum[k] +  int_mul[2 * k + 1]);
    }

    eval <== int_sum[l];
}





function UniformR1CSVarsA(outer_idx) { 
    var arrays[118][3] =  [[0, 38,  1], [1, 39,  1], [2, 40,  1], [3, 41,  1], [4, 42,  1], [5, 43,  1], [6, 44,  1], [7, 45,  1], [8, 46,  1], [9, 47, 1], [10, 48, 1], [11, 49,  1], [12, 50,  1], [13, 51,  1], [14, 52,  1], [15, 53,  1], [16, 54,  1], [17, 55,  1], [18, 56,  1], [19, 57,  1], [20, 58,  1], [21, 59,  1], [22, 60,  1], [23, 61,  1], 
    [24, 62,  1], [25, 63,  1], [26, 27,  1], [27, 28,  1], [28, 29,  1], [29, 30,  1], [30, 31,  1], [31, 32,  1], [32, 33,  1], [33, 34,  1], [34, 35,  1], [35, 36,  1], [36, 37,  1], [37, 2,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [37, 27,  68719476736], [37, 28, 34359738368], [37, 29, 17179869184], 
    [37, 30,  8589934592], [37, 31, 4294967296], [37, 32, 2147483648], [37, 33,  1073741824], [37, 34,  536870912], [37, 35,  268435456], [37, 36, 134217728], [37, 37,  67108864], [37, 38, 33554432], [37, 39, 16777216], [37, 40, 8388608], [37, 41,  4194304], [37, 42,  2097152], [37, 43, 1048576], [37, 44, 524288], [37, 45, 262144], [37, 46,  131072], 
    [37, 47, 65536], [37, 48,  32768], [37, 49, 16384], [37, 50,  8192], [37, 51,  4096], [37, 52, 2048], [37, 53,  1024], [37, 54, 512], [37, 55,  256], [37, 56,  128], [37, 57, 64], [37, 58, 32], [37, 59,  16], [37, 60, 8], [37, 61,  4], [37, 62,  2], [37, 63,  1], [38, 27,  1], [39, 28,  1], [40, 29,  1], [40, 30,  1], [41, 29,  1], [42, 29,  1], [43, 30,  1], 
    [44, 38,  1], [44, 62,  1], [44, 63,  1], [45, 39,  1], [46, 64,  1], [47, 53,  1], [47, 54,  1], [47, 55,  1], [48, 52,  1], [48, 57,  1], [49, 36,  1], [50, 34,  1], [51, 34,  1], [52, 49,  1], [52, 50,  1], [52, 51,  1], [53, 34,  1], [54, 49, 1], [54, 50, 1], [54, 51, 1], [55, 34, 1], [56, 49, 1], [56, 50, 1], [56, 51, 1], 
    [57, 34,1], [58, 49,1], [58, 50,1], [58, 51,1], [59, 34,1], [60, 5,1], [61, 71,1], [62, 5,1], [63, 72,1], [64, 31,  1], [65, 32,  1], [66, 74,  1]];
    var result[3] = arrays[outer_idx];
    return result;
}




function UniformR1CSVarsB(outer_idx) { 
    var arrays[116][3] = [[0, 38,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [1, 39,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [2, 40,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [3, 41,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [4, 42,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [5, 43,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [6, 44,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [7, 45,  21888242871839275222246405745257275088548364400416034343698204186575808495616], 
    [8, 46,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [9, 47,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [10, 48,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [11, 49,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [12, 50,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [13, 51,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [14, 52,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [15, 53,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [16, 54,  21888242871839275222246405745257275088548364400416034343698204186575808495616],
     [17, 55,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [18, 56,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [19, 57,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [20, 58,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [21, 59,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [22, 60,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [23, 61,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [24, 62,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [25, 63,  21888242871839275222246405745257275088548364400416034343698204186575808495616],
      [26, 27,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [27, 28,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [28, 29,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [29, 30,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [30, 31,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [31, 32,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [32, 33,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [33, 34,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [34, 35,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [35, 36,  21888242871839275222246405745257275088548364400416034343698204186575808495616], 
      [36, 37,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [38, 1,  4], [38, 8,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [39, 6,  1], [39, 9,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [40, 6,  1], [40, 7,  21888242871839275222246405745257275088548364400416034343698204186575808495613], [40, 8,  1], [41, 11,  1], [41, 13,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [42, 11,  1], [42, 12,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [43, 9,  1], [43, 13,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [44, 14,  281474976710656], [44, 15,  4294967296], [44, 16,  65536], [44, 17,  1], [44, 64,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [44, 65,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [45, 14,  281474976710656], [45, 15,  4294967296], [45, 16,  65536], [45, 17,  1], [45, 64,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [45, 65,  1], [46, 65,  1], [47, 14,  281474976710656], [47, 15,  4294967296], [47, 16,  65536], [47, 17,  1], [47, 66,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [48, 14,  281474976710656], [48, 15,  4294967296], [48, 16,  65536], [48, 17,  1], [48, 64,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [49, 18,  1], [50, 19,  16777216], [50, 20,  65536], [50, 21,  256], [50, 22,  1], [50, 64,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [51, 23,  16777216], [51, 24,  65536], [51, 25,  256], [51, 26,  1], [51, 65,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [52, 23,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [52, 26,  1], [53, 14,  1], [53, 19,  21888242871839275222246405745257275088548364400416034343698204186575808495361], [53, 67,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [54, 24,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [54, 26,  1], [55, 15,  1], [55, 20,  21888242871839275222246405745257275088548364400416034343698204186575808495361], [55, 68,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [56, 25,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [56, 26,  1], [57, 16,  1], [57, 21,  21888242871839275222246405745257275088548364400416034343698204186575808495361], [57, 69,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [58, 26,  0], [59, 17,  1], [59, 22,  21888242871839275222246405745257275088548364400416034343698204186575808495361], [59, 70,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [60, 33,  1], [61, 12,  1], [61, 18,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [62, 31,  1], [63, 1,  4], [63, 12,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [64, 1,  21888242871839275222246405745257275088548364400416034343698204186575808495613], [64, 18,  1], [64, 37,  4], [65, 18,  1], [66, 1,  4], [66, 6,  1], [66, 73,  21888242871839275222246405745257275088548364400416034343698204186575808495616]];
   
    var result[3] = arrays[outer_idx];
    return result;

}

function UniformR1CSVarsC(outer_idx) { 
    var arrays[21][3] = [[38, 8,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [38, 64,  1], [39, 9,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [39, 65,  1], [46, 66,  1], [52, 23,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [52, 67,  1], [54, 24,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [54, 68,  1], [56, 25,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [56, 69,  1], [58, 26,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [58, 70,  1], [60, 71,  1], [62, 72,  1], [64, 1,  21888242871839275222246405745257275088548364400416034343698204186575808495613], [64, 37,  4], [64, 73,  1], [65, 74,  1], [66, 73,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [66, 75,  1]];
    var result[3] = arrays[outer_idx];
    return result;
}

function UniformR1CSConstB(outer_idx) { 
     var arrays[45][2] = 
     [[0,  1], [1,  1], [2,  1], [3,  1], [4,  1], [5,  1], [6,  1], [7,  1], [8,  1], [9,  1], [10,  1], [11,  1], [12,  1], [13,  1], [14,  1], [15,  1], [16,  1], [17,  1], [18,  1], [19,  1], [20,  1], [21,  1], [22,  1], [23,  1], [24,  1],
     [25,  1], [26,  1], [27,  1], [28,  1], [29,  1], [30,  1], [31,  1], [32,  1], [33,  1], [34,  1], [35,  1], [36,  1], [37,  1], [38,  2147483644], [40,  21888242871839275222246405745257275088548364400416034343698204186573661028353], 
     [45,   21888242871839275222246405745257275088548364400416034343698204186571513528321], [49,  21888242871839275222246405745257275088548364400416034343698204186575808495616], [63, 2147483648], [64,  21888242871839275222246405745257275088548364400416034343698204186573661011969],
     [66, 2147483648]];
      var result[2] = arrays[outer_idx];
     return result;
}

function UniformR1CSConstC(outer_idx) { 
     var arrays[1][2] = 
     [[64,  21888242871839275222246405745257275088548364400416034343698204186573661011965]]; 
     var result[2] = arrays[outer_idx];
     return result;
}


template ComputeUniformMatrixMleA(eq_rx_constr_len, eq_ry_var_len){
    input  signal  eq_rx_ry_step;
    input  signal  eq_rx_constr[eq_rx_constr_len];
    input  signal  eq_ry_var[eq_ry_var_len] ;
    output  signal   full_mle_evaluation;

    var index = 118;
    var matrix[3];

    signal sum[index +1];
    sum[0] <== 0;

    for (var i = 0; i < index; i++) {
        matrix = UniformR1CSVarsA(i);
        sum[i + 1] <== sum[i] +  matrix[2] * eq_rx_constr[matrix[0]] * eq_ry_var[matrix[1]];
    }
    full_mle_evaluation <== sum[index] *  eq_rx_ry_step;

} 

template ComputeUniformMatrixMleB(eq_rx_constr_len, eq_ry_var_len){
    input  signal  eq_rx_ry_step;
    input  signal  col_eq_constant;
    input  signal  eq_rx_constr[ eq_rx_constr_len];
    input  signal  eq_ry_var[eq_ry_var_len ] ;
    output signal   result;
    var index = 116;
    var matrix[3];

    signal temp_sum[index][2];
    signal sum[index + 1];

    sum[0] <== 0;
    for (var i = 0; i < index; i++) {
      matrix = UniformR1CSVarsB(i);
      sum[i + 1] <== sum[i] + matrix[2] * eq_rx_constr[matrix[0]] * eq_ry_var[matrix[1]];
    }


    signal  full_mle_evaluation[2];
    full_mle_evaluation[0]  <== (sum[index] *  eq_rx_ry_step);

    index = 45;
    signal consts_sum[index +1];
    var consts_matrix[2];

    consts_sum[0] <== 0;

    for (var i = 0; i < index; i++) {
      consts_matrix = UniformR1CSConstB(i);
      consts_sum[i + 1] <== consts_sum[i] + eq_rx_constr[consts_matrix[0]] * consts_matrix[1];
    }

    full_mle_evaluation[1] <== (col_eq_constant *  consts_sum[index]);

    result <== full_mle_evaluation[1]  +  full_mle_evaluation[0];

}  

template ComputeUniformMatrixMleC(eq_rx_constr_len, eq_ry_var_len){
    input signal  eq_rx_ry_step;
    input signal  col_eq_constant;
    input signal  eq_rx_constr[eq_rx_constr_len];
    input signal  eq_ry_var[eq_ry_var_len ] ;
    output signal   result;

    var index = 21;
    signal val[index];
    var matrix[3];

    signal temp_sum[index][2];
    signal sum[index +1];
    sum[0] <== 0;

    for (var i = 0; i < index; i++) {
      matrix = UniformR1CSVarsC(i);
      sum[i + 1] <== sum[i] + matrix[2] * eq_rx_constr[matrix[0]] * eq_ry_var[matrix[1]];
    }

    signal full_mle_evaluation[2];
    full_mle_evaluation[0]  <== (sum[index] *  eq_rx_ry_step);

    index = 1;
    signal consts_sum[index +1];
    var consts_matrix[2];

    consts_sum[0] <== 0;

    for (var i = 0; i < index; i++) {
        consts_matrix= UniformR1CSConstC(i);
        consts_sum[i +1] <== consts_sum[i] + eq_rx_constr[consts_matrix[0]] * consts_matrix[1];
    }
    full_mle_evaluation[1] <== (col_eq_constant *  consts_sum[index]);
    result <== full_mle_evaluation[1]   +  full_mle_evaluation[0];
}  


template ComputeNonUniformCondition(idx, eq_rx_constr_len, eq_ry_var_len){
    input signal  eq_rx_ry_step;
    input signal  eq_step_offset_1;
    input signal  eq_rx_constr[ eq_rx_constr_len];
    input signal  eq_ry_var[eq_ry_var_len];
    input signal  col_eq_constant;
    output signal result;   
    var offset_var[3];
    offset_var =  NonUniformOffsetVarsCondition(idx);
    signal coeff;
    coeff  <==   offset_var[2];

    signal temp_result[6];
    temp_result[0] <==  (eq_ry_var[offset_var[0]] * eq_rx_ry_step);
    temp_result[1] <==  (eq_ry_var[offset_var[0]] * eq_step_offset_1);
    signal one <== 1;
  
    signal offset_0;
    offset_0 <== offset_var[1];

    temp_result[2] <== (one - offset_0);
    temp_result[3] <== (temp_result[2] * temp_result[0] );
    temp_result[4] <== (offset_0 *  temp_result[1] );
    temp_result[5] <== (temp_result[3]+ temp_result[4]);
    result <==  (temp_result[5] *  coeff);
}

template ComputeNonUniformEq(idx, eq_rx_constr_len, eq_ry_var_len){
    input signal  eq_rx_ry_step;
    input signal  eq_step_offset_1;
    input signal  eq_rx_constr[eq_rx_constr_len];
    input signal  eq_ry_var[eq_ry_var_len];
    input signal  col_eq_constant;
    output signal result;
    var offset_var[3];

    signal temp_sum[2][3];
    signal final_sum[3];
    final_sum[0] <== 0;
    signal coeff[2];
    for (var i = 0; i < 2; i++){
        offset_var =  NonUniformOffsetVarsEq(idx, i);
        coeff[i]  <==   offset_var[2];
        temp_sum[i][0] <==  (coeff[i] * eq_ry_var[offset_var[0]]);
        temp_sum[i][1] <==  (eq_rx_ry_step * temp_sum[i][0]);
        temp_sum[i][2] <==  (eq_step_offset_1 * temp_sum[i][0]);
        final_sum[i + 1] <==   final_sum[i]  + (1 -  offset_var[1]) *    temp_sum[i][1]  +  offset_var[1] *  temp_sum[i][2];
    }

    signal constant;
    var constant_var =  NonUniformConstantEq(idx);
    constant <== constant_var;
  
    var temp_result = (col_eq_constant *  constant);

    result <== (temp_result +  final_sum[2]);
}

function NonUniformOffsetVarsCondition(outer_idx) { 
    var arrays[2][3] = [[1, 1,  1], [35, 0,  1]];
    var result[3] = arrays[outer_idx];
    return result;

}

function NonUniformOffsetVarsEq(outer_idx ,inner_idx) { 
    var arrays[2][2][3] = [[[75, 0,  1], [1, 1,  21888242871839275222246405745257275088548364400416034343698204186575808495613]], [[0, 1,  1], [0, 0,  21888242871839275222246405745257275088548364400416034343698204186575808495616]]];
    var result[3] = arrays[outer_idx][inner_idx];
    return result;

}

function NonUniformConstantEq(outer_idx) { 
    var arrays[2] = [ 21888242871839275222246405745257275088548364400416034343698204186573661011969 , 21888242871839275222246405745257275088548364400416034343698204186575808495616];
    var result = arrays[outer_idx];
    return result;

}

