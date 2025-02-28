pragma circom 2.2.1;
include "./../../../fields/non_native/utils.circom";
include "./../utils.circom";




template CombineLookupsAdd(C, M) {
    signal input vals[C];
    signal output result;

    var log_M = log2(M);
    result <== ConcatenateLookups(C, log_M)(vals);
}

template CombineLookupsAnd(C, M) {
    signal input vals[C];
    signal output result;

    var log_M_by_2 = log2(M) / 2;
    result <== ConcatenateLookups(C, log_M_by_2)(vals);
}

template CombineLookupsOr(C, M){
    signal input vals[C];
    signal output result;

    var log_M_by_2 = log2(M) / 2;
    result <== ConcatenateLookups(C, log_M_by_2) (vals);
}

template CombineLookupsXor(C, M) {
    signal input vals[C];
    signal output result;

    var log_M_by_2 = log2(M) / 2;
    result <== ConcatenateLookups(C, log_M_by_2) (vals);
}

template CombineLookupsBeq(C, M) {
    signal input vals[C];
    signal output result;

    result <== Product(C)(vals);
}

template CombineLookupsBge(C, M){
    signal input vals[2 * C + 1];
    signal output result;

    signal temp_prod;
    
    temp_prod <== CombineLookupsSlt(C, M)(vals);
    result <== 1 - temp_prod;
}

template CombineLookupsBgeu(C, M){
    signal input vals[2 * C - 1];
    signal output result;

    signal temp_prod;
    
    temp_prod <== CombineLookupsSltu(C, M)(vals);

    result <== 1 - temp_prod;
}

template CombineLookupsBne(C, M) {
    signal input vals[C];
    signal output result;

    signal temp_prod;

    temp_prod <== Product(C)(vals);
    result <== 1 - temp_prod;
}

template CombineLookupsSlt(C, M){
    signal input vals[2 * C +1];
    signal output result;

    signal left_msb[1];
    signal right_msb[1];
    signal ltu[C-1];
    signal eq[C-2];
    signal lt_abs[1];
    signal eq_abs[1];

    component sliceValues = SliceValuesSlt(C, M);
    sliceValues.vals <== vals;
    left_msb <== sliceValues.left_msb;
    right_msb <== sliceValues.right_msb;
    ltu <== sliceValues.ltu;
    eq <== sliceValues.eq;
    lt_abs <== sliceValues.lt_abs;
    eq_abs <== sliceValues.eq_abs;


    signal ltu_sum[C];
    ltu_sum[0] <== lt_abs[0];

    signal eq_prod[C-1];
    eq_prod[0] <== eq_abs[0];

    signal temp_vec[C-2];

    for (var i = 0; i < C-2; i++) {
        temp_vec[i] <== ltu[i] * eq_prod[i];
        ltu_sum[i+1] <== ltu_sum[i] + temp_vec[i];
        eq_prod[i+1] <== eq_prod[i] * eq[i];
    }


    signal ltu_sum_1 <== ltu[C-2] * eq_prod[C-2];
    ltu_sum[C-1] <== ltu_sum[C-2] + ltu_sum_1;
 
    signal  temp_result[8];
    temp_result[0] <== 1 - right_msb[0];
    temp_result[1] <== left_msb[0] * temp_result[0];
    temp_result[2] <== left_msb[0] * right_msb[0];
    temp_result[3] <== 1 - left_msb[0];
    temp_result[4] <== 1 - right_msb[0];
    temp_result[5] <== temp_result[3] * temp_result[4];
    temp_result[6] <== temp_result[2] + temp_result[5];
    temp_result[7] <== temp_result[6] * ltu_sum[C-1];
    result <== temp_result[1] + temp_result[7];

}

template CombineLookupsSltu(C, M){
    signal input vals[2 * C - 1];
    signal output result;

    signal ltu[C];
    signal eq[C-1];

    component sliceValues = SliceValuesSltu(C, M);
    sliceValues.vals <== vals;
    ltu <== sliceValues.ltu;
    eq <== sliceValues.eq;

   signal  sum[C];
   sum[0] <== 0;

   signal  eq_prod[C];
   eq_prod[0] <== 1;


    signal temp_vec[C-1];
    for (var i = 0; i < C-1; i++) {
       temp_vec[i] <== ltu[i] * eq_prod[i];
       sum[i+1] <== sum[i] + temp_vec[i];
       eq_prod[i+1] <== eq_prod[i] * eq[i];
    }

    result <== sum[C-1] + ltu[C-1] * eq_prod[C-1];
     

}

template CombineLookupsSll(C, M) {
    signal input vals[C];
    signal output result;

    var log_M_by_2 = log2(M) / 2;
    result <==  ConcatenateLookups(C, log_M_by_2) (vals);
}

template CombineLookupsMulhu(C, M) {
    signal input vals[C];
    signal output result;

    var log_M = log2(M);
    result <== ConcatenateLookups(C, log_M) (vals);
}

template CombineLookupsMulu(C, M) {
    signal input vals[C];
    signal output result;

    var log_M = log2(M);
    result <== ConcatenateLookups(C, log_M) (vals);
}

template SliceValuesSlt(C, M) {
    signal input vals[2 *C +1];
    var offset = 0;


    signal output left_msb[1];
    left_msb[0] <== vals[offset];
    offset += 1;


    signal output right_msb[1];
    right_msb[0] <== vals[offset];
    offset += 1;

    signal output ltu[C-1];
    for (var i = 0; i < C-1; i++) {
        ltu[i] <== vals[offset + i];
    }
    offset += C-1;

    signal output eq[C-2];
    for (var i = 0; i < C-2; i++) {
        eq[i] <== vals[offset + i];
    }
    offset += C-2;

    signal output lt_abs[1];
    lt_abs[0] <== vals[offset];
    offset += 1;

    signal output eq_abs[1];
    eq_abs[0] <== vals[offset];
    offset += 1;


}


template SliceValuesSltu(C, M) {
    signal input vals[2 *C - 1];
    var offset = 0;

    signal output ltu[C];
    for (var i = 0; i < C; i++) {
        ltu[i] <== vals[offset + i];
    }
    offset += C;

    signal output eq[C-1];
    for (var i = 0; i < C-1; i++) {
        eq[i] <== vals[offset + i];
    }
    offset += C-1;

}


template CombineLookupsSra(C, M) {
    signal input vals[C + 1];
    signal output result;

    result <== Sum(C + 1)(vals);
}

template CombineLookupsSrl(C, M) {
    signal input vals[C];
    signal output result;
    
    result <== Sum(C)(vals);
}

template CombineLookupsVirtualMovsign(C, M, WORD_SIZE) {
    signal input vals[C];
    signal output result;

    var repeat = WORD_SIZE / 16;
    signal val_vec[repeat]; //correct it
    for (var i = 0; i < repeat; i++) {
        val_vec[i] <== vals[0];
    }
    var log_M = log2(M);
    result <==  ConcatenateLookups(repeat, log_M) (val_vec);
}

template CombineLookupsSub(C, M) {
    signal input vals[C];
    signal output result;

    var log_M = log2(M);
    result <==  ConcatenateLookups(C, log_M)(vals);
}

template CombineLookupsVirtualAdvice(C, M) {
    signal input vals[C];
    signal output result;

    var log_M = log2(M);
    result <== ConcatenateLookups(C, log_M) (vals);
}


template CombineLookupsMul(C, M) {
    signal input vals[C];
    signal output result;

    var log_M = log2(M);
    result <== ConcatenateLookups(C, log_M) (vals);
}

template CombineLookupsAlignedMemoryAccess(C, ALIGN) {
    signal input vals[C];
    signal output result;
  
    signal one_mius_lowest_bit;
    signal one_minus_second_lowest_bit;

    if (ALIGN == 2) {
        assert(C == 1);

        signal lowest_bit <== vals[0];
        result <== 1 - lowest_bit;

    } else if (ALIGN == 4) {
        assert(C == 2);
        signal lowest_bit <== vals[0];
        signal second_lowest_bit <== vals[1];
        one_mius_lowest_bit <== 1 - lowest_bit;
        one_minus_second_lowest_bit <== 1 - second_lowest_bit;
        result <== one_minus_second_lowest_bit * one_mius_lowest_bit; 
    } 
}

template CombineLookupsVirtualAssertLte(C, M) {
    signal input vals[2 * C ];
    signal output result;

    signal ltu[C];
    signal eq[C];

    signal vals_by_subtable[2][C];
    vals_by_subtable <== SliceValues(C, M)(vals);

    ltu <== vals_by_subtable[0];
    eq  <== vals_by_subtable[1];

    signal ltu_sum[C+1];
    ltu_sum[0] <== 0;

    signal eq_prod[C+1];
    signal temp_vec[C];

    eq_prod[0] <== 0;
    for (var i = 0; i < C; i++) {
        temp_vec[i] <== ltu[i] * eq_prod[i];
        ltu_sum[i+1] <== ltu_sum[i] + temp_vec[i];
        eq_prod[i+1] <== eq_prod[i] * eq[i];
    }

    result <== ltu_sum[C] + eq_prod[C];
}

template SliceValues( C, M) {
    signal input vals[2 * C];
    signal output slices[2][C];

    var offset = 0;

    var indices_len = C; // Assuming each subtable has M elements
     for (var i = 0; i < 2; i++) {  
         for (var j = 0; j < C; j++) {  
           slices[i][j] <== vals[offset + j];
         }
        offset += indices_len;
    }
}


template CombineLookupsVirtualAssertValidDiv0(C, M) {
    signal input vals[2 * C ];
    signal output result;

    signal vals_by_subtable[2][C] <== SliceValues(C, M)(vals);

    signal divisor_is_zero <== Product(C)(vals_by_subtable[0]);

    signal is_valid_div_by_zero <== Product(C)(vals_by_subtable[1]);

    signal  temp_result <== divisor_is_zero + is_valid_div_by_zero;

    result <== 1 - temp_result;

}



template CombineLookupsVirtualAssertValidSignedRemainder(C, M) {
    signal input vals[4 * C + 2];
    signal output result;

    signal left_msb[1];
    signal right_msb[1];
    signal eq[C - 1];
    signal ltu[C - 1];
    signal eq_abs[1];
    signal lt_abs[1];
    signal slices[2][C];

    component sliceValues = SliceValuesValidSignedRemainder(C, M);
    sliceValues.vals <== vals;

    left_msb <== sliceValues.left_msb;
    right_msb <== sliceValues.right_msb;
    eq <== sliceValues.eq;
    ltu <== sliceValues.ltu;
    eq_abs <== sliceValues.eq_abs;
    lt_abs <== sliceValues.lt_abs;

    signal remainder_is_zero <== Product(C)(sliceValues.slices[0]);
    signal divisor_is_zero <== Product(C)(sliceValues.slices[1]);

    signal  ltu_sum[C];
    ltu_sum[0] <== lt_abs[0];

    signal  eq_prod[C];
    eq_prod[0] <== eq_abs[0];

    signal  temp_vec[C-1];
    for (var i = 0; i < C-1; i++) {
        temp_vec[i] <== ltu[i] * eq_prod[i];
        ltu_sum[i+1] <== ltu_sum[i] + temp_vec[i];
        eq_prod[i+1] <== eq_prod[i] * eq[i];
    }

    signal temp_result[11];
    temp_result[0] <== 1 - left_msb[0];
    temp_result[1] <== temp_result[0] - right_msb[0];
    temp_result[2] <== temp_result[1] * ltu_sum[C-1];
    temp_result[3] <== left_msb[0] * right_msb[0];
    temp_result[4] <== 1 - eq_prod[C-1];
    temp_result[5] <== temp_result[3] * temp_result[4];
    temp_result[6] <== 1 - left_msb[0];
    temp_result[7] <== temp_result[6] * right_msb[0];
    temp_result[8] <== temp_result[7] * remainder_is_zero;
    temp_result[9] <== divisor_is_zero + temp_result[8];
    temp_result[10] <== temp_result[2] + temp_result[5];
    result <== temp_result[10] + temp_result[9];
}


template SliceValuesValidSignedRemainder(C, M) {
    signal input vals[4 * C + 2];

    var offset = 0;

    signal output left_msb[1];
    left_msb[0] <== vals[offset];
    offset += 1;

    signal output right_msb[1];
    right_msb[0] <== vals[offset];
    offset += 1;

    signal output eq[C-1];
    for (var i = 0; i < C-1; i++) {
        eq[i] <== vals[offset + i];
    }
    offset += C-1;

    signal output ltu[C-1];
    for (var i = 0; i < C-1; i++) {
        ltu[i] <== vals[offset + i];
    }
    offset += C-1;

    signal output eq_abs[1];
    eq_abs[0] <== vals[offset];
    offset += 1;

    signal output lt_abs[1];
    lt_abs[0] <== vals[offset];
    offset += 1;

    signal output slices[2][C];
    for (var i = 0; i < 2; i++) {  
        for (var j = 0; j < C; j++) {  
            slices[i][j] <== vals[offset + j];
        }
        offset += C;
    }
}





template CombineLookupsVirtualAssertValidUnsignedRemainder(C, M) {
    signal input vals[3 * C - 1 ];
    signal output result;

    signal ltu[C];
    signal eq[C-1];
  
    component sliceValues = SliceValuesValidUnsignedRemainder(C, M);

    sliceValues.vals <== vals;
    ltu <== sliceValues.ltu;
    eq <== sliceValues.eq;

    signal divisor_is_zero <== Product(C)(sliceValues.slice2);

    signal  sum[C];
    sum[0] <== 0;

    signal  eq_prod[C];
    eq_prod[0] <== 1;


    signal  temp_vec[C-1];
    for (var i = 0; i < C-1; i++) {
        temp_vec[i] <== ltu[i] * eq_prod[i];
        sum[i+1] <== sum[i] + temp_vec[i];
        eq_prod[i+1] <== eq_prod[i] * eq[i];
    }

    signal temp_result[2];
    temp_result[0] <== eq_prod[C-1] * ltu[C-1];
    temp_result[1] <== temp_result[0] + sum[C-1];
    result <== temp_result[1] + divisor_is_zero;

}


template SliceValuesValidUnsignedRemainder(C, M) {
    signal input vals[3 * C -1];

    var offset = 0;

    signal output ltu[C];
    for (var i = 0; i < C; i++) {
        ltu[i] <== vals[offset + i];
    }
    offset += C;


    signal output eq[C-1];
    for (var i = 0; i < C-1; i++) {
        eq[i] <== vals[offset + i];
    }
    offset += C-1;

    signal output slice2[C];
    for (var i = 0; i < C; i++) {
        slice2[i] <== vals[offset + i];
    }
    offset += C;

}


template CombineLookupsVirtualMove(C, M) {
    signal input vals[C];
    signal output result;

    var log_M = log2(M);
    result <==  ConcatenateLookups(C, log_M) (vals);
}



template ConcatenateLookups(C, operand_bits)  {
    signal input vals[C];
    signal output result;

    signal sum[C + 1];
    sum[0] <== 0;
   
    signal weight[C];
    signal t1[C];

    for (var i = 0; i < C; i++) {
        weight[i] <-- 1 << (i * operand_bits);
        t1[i] <== weight[i] * vals[C - i - 1];
        sum[i + 1] <== sum[i] + t1[i];
    }
    
    result <== sum[C];
}














