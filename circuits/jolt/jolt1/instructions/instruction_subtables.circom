pragma circom 2.2.1;
include "./../../../fields/non_native/utils.circom";
include "./../utils.circom";


template evaluate_mle_and (point_len) {
    input signal point[point_len];
    output signal result;

    var b = point_len / 2;
    signal x[b];
    signal y[b];
    (x, y) <== SplitAt(b, point_len)(point);

    signal res[b+1];
    res[0] <== 0;
 
    signal t1[b];
    signal t2[b];
    for (var i = 0; i < b; i++) {
        t1[i] <==  (res[i] + res[i]);
        t2[i] <==  (x[i] *  y[i]);
        res[i + 1] <== (t1[i] +  t2[i]);
    }

    result <== res[b];
}

template evaluate_mle_div_by_zero(point_len) {
    input signal point[point_len];
    output signal result;

    var b = point_len / 2;
    signal x[b];
    signal y[b];
    (x, y) <== SplitAt(b, point_len)(point);

    signal res[b+1];
    res[0] <== 1;
 
    signal t1[b];
    signal t2[b];
    for (var i = 0; i < b; i++) {
        t1[i] <== (res[0] - x[i]);
        t2[i] <== (t1[i] * y[i]);        
        res[i + 1] <== (res[i] * t2[i]);
    }
    result <== res[b];
}

template evaluate_mle_eq_abs (point_len) {
    input signal point[point_len];
    output signal result;

    var b = point_len / 2;
    signal x[b];
    signal y[b];
    (x, y) <== SplitAt(b, point_len)(point);

    signal res[b];
    res[0] <== 1;
    
    signal t1[b-1];
    signal t2[b-1];
    signal t3[b-1];
    signal t4[b-1];
    signal t5[b-1];
    for (var i = 0; i < b - 1; i++) {
        t1[i] <== (x[i+1] + y[i+1]);
        t2[i] <== (1 - t1[i]);
        t3[i] <== (x[i+1]* y[i+1]);
        t4[i] <== (t3[i]+ t3[i]);
        t5[i] <== (t2[i] + t4[i]);
        res[i + 1] <== (res[i] *  t5[i]); 
    } 

    result <== res[b - 1];
}

template evaluate_mle_eq (point_len) {
    input signal point[point_len];
    output signal result;

    var b = point_len / 2;
    signal x[b];
    signal y[b];
    (x, y) <== SplitAt(b, point_len)(point);

    signal res[b+1];
    res[0] <== 1;
 
    signal t1[b];
    signal t2[b];
    signal t3[b];
    signal t4[b];
    signal t5[b];
    for (var i = 0; i < b; i++) {
        t1[i] <==  (x[i] + y[i]);
        t2[i] <==  (res[0] - t1[i]);
        t3[i] <==  (x[i] * y[i]);
        t4[i] <==  (t3[i] + t3[i]);
        t5[i] <==  (t2[i] +  t4[i]);
        res[i + 1] <== (res[i] *  t5[i]); 
    }

    result <== res[b];
}

template evaluate_mle_identiy(point_len) {
    input signal point[point_len];
    output signal result;

    signal res[point_len + 1];
    res[0] <== 0;
  
    signal t1[point_len];
    for (var i = 0; i < point_len; i++) {
        t1[i] <== (res[i] + res[i]);
        res[i + 1] <== (t1[i] +  point[i]);
    }

    result <== res[point_len];
}

template evaluate_mle_left_is_zero (point_len) {
    input signal point[point_len];
    output signal result;

    var b = point_len / 2;
    signal x[b];
    signal y[b];
    (x, y) <== SplitAt(b, point_len)(point);

    signal res[b+1];
    res[0] <== 1;
  

    signal t1[b];
    for (var i = 0; i < b; i++) {
        t1[i] <== (res[0] - x[i]);
        res[i + 1] <==  (res[i] *  t1[i]);
    }

    result <== res[b];
}

template evaluate_mle_left_msb(point_len) {
    input signal point[point_len];
    output signal result;

    var b = point_len / 2;
    signal x[b];
    signal y[b];
    (x, y) <== SplitAt(b, point_len)(point);

    result <== x[0];
}

template evaluate_mle_low_bit (point_len, OFFSET) {
    input signal point[point_len];
    output signal result;

    result <== point[point_len - 1 - OFFSET];
}

template evaluate_mle_lt_abs(point_len){
    input signal point[point_len];
    output signal result;

    var b = point_len / 2;
    signal x[b];
    signal y[b];
    (x, y) <== SplitAt(b, point_len)(point);

    signal res[b];
    res[0] <== 0;
 
    signal eq_term[b];
    eq_term[0] <== 1;
    

    signal t1[b-1];
    signal t2[b-1];
    signal t3[b-1];
    signal t4[b-1];
    signal t5[b-1];
    signal t6[b-1];
    signal t7[b-1];
    signal t8[b-1];
    for (var i = 0; i < b - 1; i++) {
        t1[i] <== (1 - x[i+1]);
        t2[i] <== (t1[i] * y[i+1]);
        t3[i] <== (t2[i] * eq_term[i]);
        res[i + 1] <== (res[i] + t3[i]);

        t4[i] <== (x[i+1] + y[i+1]);
        t5[i] <== (1 -  t4[i]);
        t6[i] <== (x[i+1] * y[i+1]);
        t7[i] <== (t6[i] + t6[i]);
        t8[i] <== (t5[i] + t7[i]);
        eq_term[i + 1] <== (eq_term[i] *  t8[i]);
    }

    result <== res[b-1];
}

template evaluate_mle_ltu(point_len) {
    input signal point[point_len];
    output signal result;

    var b = point_len / 2;
    signal x[b];
    signal y[b];
    (x, y) <== SplitAt(b, point_len)(point);

    signal res[b+1];
    res[0] <== 0;
    

    signal eq_term[b+1];
    eq_term[0] <== 1;
  
    signal t1[b];
    signal t2[b];
    signal t3[b];
    signal t4[b];
    signal t5[b];
    signal t6[b];
    signal t7[b];
    signal t8[b];
    for (var i = 0; i < b; i++) {
        t1[i] <== (1 - x[i]);
        t2[i] <== (t1[i] * y[i]);
        t3[i] <== (t2[i] * eq_term[i]);
        res[i + 1] <== (res[i] + t3[i]);

        t4[i] <== (x[i] + y[i]);
        t5[i] <== (1 - t4[i]);
        t6[i] <== (x[i] * y[i]);
        t7[i] <== (t6[i] + t6[i]);
        t8[i] <== (t5[i] + t7[i]);
        eq_term[i + 1] <== eq_term[i] * t8[i];
    }

    result <== res[b];
}

template evaluate_mle_or (point_len) {
    input signal point[point_len];
    output signal result;

    var b = point_len / 2;
    signal x[b];
    signal y[b];
    (x, y) <== SplitAt(b, point_len)(point);

    signal res[b+1];
    res[0] <== 0;
   

    signal t1[b];
    signal t2[b];
    signal t3[b];
    signal t4[b];
    for (var i = 0; i < b; i++) {
        t1[i] <==  (res[i] + res[i]);
        t2[i] <==  (x[i]+  y[i]);
        t3[i] <== (x[i] * y[i]);
        t4[i] <== (t2[i] -  t3[i]);
        res[i + 1] <== (t1[i] + t4[i]);
    }

    result <== res[b];
}

template evaluate_mle_right_is_zero(point_len) {
    input signal point[point_len];
    output signal result;

    var b = point_len / 2;
    signal x[b];
    signal y[b];
    (x, y) <== SplitAt(b, point_len)(point);

    signal res[b+1];
    res[0] <== 1;

    signal t1[b];
    for (var i = 0; i < b; i++) {
        t1[i] <==  (res[0] - y[i]);
        res[i + 1] <==  (res[i] * t1[i]);
    }

    result <== res[b];
}

template evaluate_mle_right_msb(point_len) {
    input signal point[point_len];
    output signal result;

    var b = point_len / 2;
    signal x[b];
    signal y[b];
    (x, y) <== SplitAt(b, point_len)(point);

   result <== y[0];
}

template evaluate_mle_sign_extend(point_len, WIDTH) {
    input signal point[point_len];
    output signal result;

    signal sign_bit <== point[point_len - WIDTH];

    var temp = (1 << WIDTH) - 1;
    
    signal ones;
    // Works if WIDTH <= 126. WIDTH = 8 in test case.
    ones <== temp;
  
    result <== (sign_bit * ones);
}

template evaluate_mle_truncate_overflow(point_len, WORD_SIZE) {
    input signal point[point_len];
    output signal result;

    var cutoff = WORD_SIZE % point_len;

    signal res[cutoff + 1];
    res[0] <== 0;
  

    signal t1[cutoff];
    for (var i = 0; i < cutoff; i++) {
        t1[i] <==(res[i] + res[i]);
        res[i + 1] <==  (t1[i] + point[point_len - cutoff - 1 + i]);
    }

    result <== res[cutoff];
}

template evaluate_mle_sll(point_len, eq_vec_len, CHUNK_INDEX, WORD_SIZE) {
    input signal point[point_len];
    input signal eq_vec[eq_vec_len];
    output signal result;

    var b = point_len / 2;
    signal x[b];
    signal y[b];
    (x, y) <== SplitAt (b, point_len) (point);

    var log_WORD_SIZE = log2 (WORD_SIZE);

    var min_1 = 0;
    if (WORD_SIZE < (1 << b)) {
        min_1 = WORD_SIZE;
    } else {
        min_1 = 1 << b;
    }

    signal res[min_1+1];
    res[0] <== 0;

    var min_2 = 0;
    if (log_WORD_SIZE < b) {
        min_2 = log_WORD_SIZE;
    } else {
        min_2 =  b;
    }

    

    signal acc[min_1][b + 1];    
    signal t6[min_1][b];
    signal t7[min_1];
    signal t8[min_1];
    signal two_power[min_1];
    for (var k = 0; k < min_1; k++) {

        var m;
        if (k + b * (CHUNK_INDEX + 1) > WORD_SIZE) {
            if (b < (k + b * (CHUNK_INDEX + 1)) - WORD_SIZE) {
                m = b;
            } else {
                m = (k + b * (CHUNK_INDEX + 1)) - WORD_SIZE;
            }
        } else {
            m = 0;
        }
        var m_prime = b - m;

        acc[k][0] <== 0;
     

        for (var j = 0; j < m_prime; j++) {
            t6[k][j] <==  (acc[k][j] + acc[k][j]);
            acc[k][j + 1] <== (t6[k][j] + x[b - m_prime + j]);
        }

        two_power[k] <-- 1 << k;
 
        t7[k] <== (two_power[k] * acc[k][m_prime]);
        t8[k] <== (eq_vec[k] * t7[k]); 
        res[k + 1] <== (res[k] + t8[k]);
    }

    result <== res[min_1];
}

template evaluate_mle_srl (point_len, eq_vec_len, CHUNK_INDEX, WORD_SIZE) {
    input signal point[point_len];
    input signal eq_vec[eq_vec_len];
    output signal result;

    var b = point_len / 2;
    signal x[b];
    signal y[b];
    (x, y) <== SplitAt (b, point_len) (point);

    var log_WORD_SIZE = log2(WORD_SIZE);

    var min_1 = 0;
    if (WORD_SIZE < 1 << b) {
        min_1 = WORD_SIZE;
    } else {
        min_1 = 1 << b;
    }
    signal res[min_1+1];
    res[0] <== 0;

    var min_2 = 0;
    if (log_WORD_SIZE < b) {
        min_2 = log_WORD_SIZE;
    } else {
        min_2 =  b;
    }

    signal acc[min_1][b + 1];     
    signal t6[min_1][b];
    signal t7[min_1];
    signal t8[min_1];
    signal two_power[min_1];
    for (var k = 0; k < min_1; k++) {
        
        var m;
        if (k > b * CHUNK_INDEX) {
            if (b < k - b * CHUNK_INDEX) {
                m = b;
            } else {
                m = k - b * CHUNK_INDEX;
            }
        } else {
            m = 0;
        }
        var chunk_length;
        if (b < WORD_SIZE - b * CHUNK_INDEX) {
            chunk_length = b;
        } else {
            chunk_length = WORD_SIZE - b * CHUNK_INDEX;
        }

        acc[k][0] <== 0;
      
        for (var j = 1; j <= chunk_length - m; j++) {
            t6[k][j-1] <==  (acc[k][j-1] + acc[k][j-1]);
            acc[k][j] <==  (t6[k][j-1] + x[b - chunk_length + j - 1]);
        }

        two_power[k] <== 1 << (b * CHUNK_INDEX - k + m);
     
        t7[k] <== (two_power[k] * acc[k][chunk_length - m]);
        t8[k] <== (eq_vec[k] * t7[k]);
        res[k + 1] <== (res[k] + t8[k]);
    }

    result <== res[min_1];
}

template evaluate_mle_sra_sign (point_len, eq_vec_len, WORD_SIZE) {
    input signal point[point_len];
    input signal eq_vec[eq_vec_len];
    output signal result;

    var b = point_len / 2;
    signal x[b];
    signal y[b];
    (x, y) <== SplitAt (b, point_len) (point);

    var log_WORD_SIZE = log2 (WORD_SIZE);
    var sign_index = (WORD_SIZE - 1) % b;
    signal x_sign <== x[b - 1 - sign_index];

    var min_1 = 0;

    if (WORD_SIZE < 1 << b) {
        min_1 = WORD_SIZE;
    } else {
        min_1 = 1 << b;
    }
    signal res[min_1+1];
    res[0] <== 0;


    var min_2 = 0;
    if (log_WORD_SIZE < b) {
        min_2 = log_WORD_SIZE;
    } else {
        min_2 =  b;
    }

    signal t7[min_1];
    signal t8[min_1];
    signal diff_two_powers[min_1];

    for (var k = 0; k < min_1; k++) {

        diff_two_powers[k] <== (1 << (WORD_SIZE)) - (1 << (WORD_SIZE - k));
     
        t7[k] <==  (diff_two_powers[k] * x_sign);
        t8[k] <==  (eq_vec[k] * t7[k]);
        res[k + 1] <==  (res[k] + t8[k]);
    }

    result <== res[min_1];
}

template evaluate_mle_xor (point_len) {
    input signal point[point_len];
    output signal result;

    var b = point_len / 2;
    signal x[b];
    signal y[b];
    (x, y) <== SplitAt (b, point_len) (point);

    signal res[b+1];
    res[0] <== 0;

    signal t1[b];
    signal t2[b];
    signal t3[b];
    signal t4[b];
    signal t5[b];
    for (var i = 0; i < b; i++) {
        t1[i] <==  (res[i] + res[i]);
        t2[i] <==  (x[i] + y[i]);
        t3[i] <==  (x[i] * y[i]);
        t4[i] <==  (t3[i] + t3[i]);
        t5[i] <==  (t2[i] -t4[i]);
        res[i + 1] <==  (t1[i] + t5[i]);
    }

    result <== res[b];
}

template evaluate_mle (len) {
    signal point[len];
    output signal out;

    signal one;
    one <== 1;


    for (var i = 0; i < len; i++) {
        point[i] <== one;
    }

    out <== evaluate_mle_xor (len)(point);    
}

// component main = evaluate_mle (16);




