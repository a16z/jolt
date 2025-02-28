pragma circom 2.2.1;

// include "./../../groups/bn254_g1.circom";
include "./../../fields/non_native/non_native_over_bn_scalar.circom";

template EvaluateEq(n) {
    input Fq() r[n];      
    input Fq() rx[n];     
    output Fq() result;   

    Fq() terms[n];        
    Fq() int_mul[2 * n];
    Fq() int_neg[2 * n];
    Fq() int_add[n];
    
    Fq() one; 
    one.limbs  <== [1, 0, 0];


    for (var i = 0; i < n; i++) {
        int_neg[2 * i] <== NonNativeSub()(one, r[i]);
        int_neg[2 * i + 1] <== NonNativeSub()(one, rx[i]);
        int_mul[2 * i] <== NonNativeMul()(r[i], rx[i]);
        int_mul[2 * i + 1] <== NonNativeMul()(int_neg[2 * i], int_neg[2 * i + 1]);
        int_add[i] <== NonNativeAdd()(int_mul[2 * i], int_mul[2 * i + 1]);
    }

    Fq() int_prod[n + 1];

    int_prod[0] <== one;  
    int_prod[1] <== int_add[0];

    for (var i = 2; i < n + 1; i++) {
        int_prod[i] <== NonNativeMul()(int_prod[i - 1], int_add[i - 1]);
    }
    
    result <== int_prod[n];
}

template EvalUniPoly(degree){
    input Fq() poly[degree + 1]; 
    input Fq() random_point;
    output Fq() eval;

    Fq() mul[degree]; 
    Fq() add[degree - 1]; 

    mul[0] <== NonNativeMul()(poly[degree], random_point);  
    
    for (var i = degree - 1; i > 0; i--)
    {    
        add[degree - i - 1] <== NonNativeAdd()(mul[degree - i - 1], poly[i]);
        
        mul[degree - i] <== NonNativeMul()(add[degree - i - 1], random_point);
    }
    eval <== NonNativeAdd()(mul[degree - 1], poly[0]);
}

template Pad(size){
    input Fq() vec[size];
    var next_power_of_two = NextPowerOf2(size);
    output Fq() padded_vec[next_power_of_two];
    Fq() zero; 
    zero.limbs  <== [0, 0, 0];

    if (size == next_power_of_two){
        padded_vec <== vec;
    }
    else{
        for (var i = 0; i < size; i++){
            padded_vec[i] <== vec[i];
        }
        for (var i = size; i < next_power_of_two; i++){
            padded_vec[i] <== zero;
        }
    }
}

template Evals(ell) {
    input Fq() r[ell];   
    var pow_2 = 1 << ell;       
    output Fq() evals[pow_2]; 

    Fq() temp[ell + 1][pow_2]; 

    Fq() one; 
    one.limbs  <== [1, 0, 0];

    temp[0][0] <== one; 

    var size = 1;
    for (var j = 0; j < ell; j++) {
        size *= 2; 
        for (var i = 0; i < size; i += 2) {
            var half_i = i / 2;
            temp[j + 1][i] <== NonNativeMul()(temp[j][half_i], r[j]);               
            temp[j + 1][i + 1] <== NonNativeSub()(temp[j][half_i], temp[j + 1][i]); 
        }
    }

    for (var i = 0; i < pow_2; i++) {
        evals[i] <== temp[ell][pow_2 - i - 1];
    }
}

template EvalsNew(ell, len) {
    input Fq() r[ell];   
    var pow_2 = 1 << ell;       
    output Fq() evals[pow_2]; 

    Fq() temp[ell + 1][pow_2]; 

    Fq() one; 
    one.limbs  <== [1, 0, 0];
  
    Fq() zero; 
    zero.limbs  <== [0, 0, 0];

    temp[0][0] <== one; 

    var size = 1;
    for (var j = 0; j < ell; j++) {
        size *= 2; 
        for (var i = 0; i < size; i += 2) {
            var half_i = i / 2;
            if (half_i < len) {
                temp[j + 1][i] <== NonNativeMul()(temp[j][half_i], r[j]);   
            } else {
                temp[j + 1][i] <== zero;
            }
            if (i + 1 < len) {
                temp[j + 1][i + 1] <== NonNativeSub()(temp[j][half_i], temp[j + 1][i]);    
            } else {
                temp[j + 1][i + 1] <== zero;
            }               
        }
    }

    for (var i = 0; i < pow_2; i++) {
        evals[i] <== temp[ell][pow_2 - i - 1];
    }
}

template ComputeDotProduct(size){
    input Fq() a[size];
    input Fq() b[size];
    output Fq() dot_product;

    Fq() mul[size]; 
    Fq() add[size]; 

    Fq() one;
    one.limbs[0] <== 1;
    one.limbs[1] <== 0;
    one.limbs[2] <== 0;

    for (var i = 0; i < size; i++)
    {    
        log("i = ", i);
        mul[i] <== NonNativeMul()(a[i], b[i]);
    }

    add[0] <== mul[0];
    for (var i = 0; i < size - 1; i++)
    {    
        add[i + 1] <== NonNativeAdd()(mul[i + 1], add[i]);
    }

    dot_product <== add[size - 1];
}

template ComputeDotProductNew(size){
    input Fq() a[size];
    input Fq() b[size];
    output Fq() dot_product;

    signal mul[size][6]; 
    signal add[size][6]; 

    for (var i = 0; i < size; i++) {    
        mul[i] <== NonNativeMulWithoutReduction()(a[i], b[i]);
    }

    add[0] <== mul[0];
    for (var i = 0; i < size - 1; i++)
    {    
        add[i + 1] <== NonNativeAdd6Limbs()(mul[i + 1], add[i]);
    }

    dot_product <== NonNativeModulo6Limbs()(add[size - 1]);
}

template SplitAt(mid, size){
    assert(mid <= size);

    input Fq() vec[size];
    output Fq() low[mid];
    output Fq() hi[size - mid];

    for (var i = 0; i < mid; i++){
        low[i] <== vec[i];
    }
    for (var i = mid; i < size; i++){
        hi[i - mid] <== vec[i];
    }
}

template Product(C)  {
    Fq() input vals[C];
    Fq() output result;
    Fq() prod[C];
    Fq() one; 
  
    if (C == 0) {
        one.limbs <== [1, 0, 0];
        result <== one;
    }
    else
    {
        prod[0] <== vals[0];
        for (var i = 1; i < C; i++) {
            prod[i] <== NonNativeMul()(prod[i - 1], vals[i]);
        }
        result <== prod[C - 1];
    }
}

template Sum(C)  {
    Fq() input vals[C];
    Fq() output result;
    Fq() sum[C];
    Fq() zero; 
    
    if (C == 0) {
        zero.limbs <== [0, 0, 0];
        result <== zero;
    }
    else{
        sum[0] <== vals[0];
        for (var i = 1; i < C; i++) {
            sum[i] <== NonNativeAdd()(sum[i - 1], vals[i]);
        }
    result <== sum[C - 1];
    }
}

template IndexToFieldBitvector(value, num_bits) {
    output Fq() bitvector[num_bits]; 
    signal bits[num_bits]; 

    for (var i = 0; i < num_bits; i++) {
        bits[i] <-- (value >> i) & 1;
        bits[i] * (bits[i] - 1) === 0;
        bitvector[num_bits - 1 - i].limbs <== [bits[i], 0, 0];
    }
}

template TruncateVec(start_index, end_index, size){
    // assert(start_index < end_index);

    input Fq() vec[size];
    output Fq() trun_vec[end_index - start_index];

    for (var i = start_index; i < end_index; i++){
        trun_vec[i - start_index] <== vec[i];
    }
}

template CombineCommitments(num_commitments){
    input G1Projective() commitments[num_commitments];
    input Fq() coeffs[num_commitments];
    output G1Projective() combine_commitment;

    G1Projective() int_sum[num_commitments + 1];
    (int_sum[0].x, int_sum[0].y, int_sum[0].z) <== (0, 1, 0);

    G1Projective() temp[num_commitments];

    for (var i = 0; i < num_commitments; i++) {        
        temp[i] <== G1Mul()(commitments[i], coeffs[i]); 
        int_sum[i + 1] <== G1Add()(int_sum[i], temp[i]);
    }
    combine_commitment <== int_sum[num_commitments];
}

template EvaluateIdentityPoly(size){
    input Fq() r[size];
    output Fq() eval;

    Fq() mul[size - 1]; 
    Fq() add[size - 2]; 
    Fq() two; 

    two.limbs <== [2, 0, 0];


    mul[0] <== NonNativeMul()(r[0], two);  
    
    for (var i = 1; i < size - 1; i++)
    {    
        add[i - 1] <== NonNativeAdd()(mul[i - 1], r[i]);
        
        mul[i] <== NonNativeMul()(add[i - 1], two);
    }

    eval <== NonNativeAdd()(mul[size - 2], r[size - 1]);
}
 
template NativeEvals(ell) {
    input signal r[ell];   
    var pow_2 = 1 << ell;       
    output signal evals[pow_2]; 

    signal temp[ell + 1][pow_2]; 

    temp[0][0] <== 1; 

    var size = 1;
    for (var j = 0; j < ell; j++) {
        size *= 2; 
        for (var i = 0; i < size; i += 2) {
            var half_i = i / 2;
            temp[j + 1][i] <== temp[j][half_i] * r[j];               
            temp[j + 1][i + 1] <== temp[j][half_i] - temp[j + 1][i]; 
        }
    }

    for (var i = 0; i < pow_2; i++) {
        evals[i] <== temp[ell][pow_2 - i - 1];
    }
}

template NativeComputeDotProduct(size){
    input signal a[size];
    input signal b[size];
    output signal dot_product;

    signal mul[size]; 
    signal add[size]; 

    for (var i = 0; i < size; i++)
    {    
        mul[i] <== a[i] * b[i];
    }

    add[0] <== mul[0];
    for (var i = 0; i < size - 1; i++)
    {    
        add[i + 1] <== mul[i + 1] + add[i];
    }

    dot_product <== add[size - 1];
}