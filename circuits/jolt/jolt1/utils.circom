pragma circom 2.2.1;
bus UniPoly(degree) {
    signal coeffs[degree + 1];
}

template ReverseVector(n) {
    input signal in[n];  
    output signal out[n]; 

    // Reverse the input vector
    for (var i = 0; i < n; i++) {
        out[i] <== in[n - 1 - i];
    }
}

//Circuit to compute f(r)
template EvalUniPoly(degree){
    input signal poly[degree + 1]; 
    input signal random_point;
    output signal eval;

    signal mul[degree]; 
    signal add[degree - 1]; 

    mul[0] <== poly[degree] * random_point;  
    
    for (var i = degree - 1; i > 0; i--)
    {    
        add[degree - i - 1] <== mul[degree - i - 1] + poly[i];
        
        mul[degree - i] <== add[degree - i - 1] * random_point;
    }
    eval <== mul[degree - 1] + poly[0];
}

template Evals(ell) {
    input signal r[ell];   
    var pow_2 = 1 << ell;       
    output signal evals[pow_2]; 

    // Temporary signals for intermediate steps
    signal temp[ell + 1][pow_2]; 

    temp[0][0] <== 1; 

    // Iteratively compute evaluations
    var size = 1;
    for (var j = 0; j < ell; j++) {
        size *= 2; 
        for (var i = 0; i < size; i += 2) {
            var half_i = i / 2;
            temp[j + 1][i] <== temp[j][half_i] * r[j];               
            temp[j + 1][i + 1] <== temp[j][half_i] - temp[j + 1][i]; 
        }
    }

    // Final output is in the last temp row
    for (var i = 0; i < pow_2; i++) {
        evals[i] <== temp[ell][pow_2 - i - 1];
    }
}

template ComputeDotProduct(size){
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

template Pad(size){
    input signal vec[size];
    var next_power_of_two = NextPowerOf2(size);
    output signal padded_vec[next_power_of_two];

    if (size == next_power_of_two){
        padded_vec <== vec;
    }
    else{
        for (var i = 0; i < size; i++){
            padded_vec[i] <== vec[i];
        }
        for (var i = size; i < next_power_of_two; i++){
            padded_vec[i] <== 0;
        }
    }
}
template Product(C)  {
    signal input vals[C];
    signal output result;
    signal prod[C];
  
    if (C == 0) {
        result <== 1;
    }
    else
    {
        prod[0] <== vals[0];
        for (var i = 1; i < C; i++) {
            prod[i] <== prod[i - 1] * vals[i];
        }
        result <== prod[C - 1];
    }
}

template Product2(C)  {
   signal input vals[C];
   signal output result;
   signal prod[C];

   if (C == 0) {
       result <== 1;
   }
   else
   {
       prod[0] <== (1 - vals[0]);
       for (var i = 1; i < C; i++) {
           prod[i] <== prod[i - 1] * (1 - vals[i]);
       }
       result <== prod[C - 1];
   }
}



template Sum(C)  {
    signal input vals[C];
    signal output result;
    signal sum[C]; 
    
    if (C == 0) {
        result <== 0;
    }
    else{
        sum[0] <== vals[0];
        for (var i = 1; i < C; i++) {
            sum[i] <== sum[i - 1] + vals[i];
        }
    result <== sum[C - 1];
    }
}

template EvaluateEq(n) {
    input signal r[n];      
    input signal rx[n];     
    output signal result;   

    signal terms[n];        
    signal int_mul[2 * n];
    signal int_neg[2 * n];
    signal int_add[n];

    for (var i = 0; i < n; i++) {
        int_neg[2 * i] <== 1 - r[i];
        int_neg[2 * i + 1] <== 1 - rx[i];
        int_mul[2 * i] <== r[i] * rx[i];
        int_mul[2 * i + 1] <== int_neg[2 * i] * int_neg[2 * i + 1];
        int_add[i] <== int_mul[2 * i] + int_mul[2 * i + 1];
    }

    signal int_prod[n + 1];

    // Compute the product of terms
    int_prod[0] <== 1;  
    int_prod[1] <== int_add[0];

    for (var i = 2; i < n + 1; i++) {
        int_prod[i] <== int_prod[i - 1] * int_add[i - 1];
    }
    
    result <== int_prod[n];
}

template SplitAt(mid, size){
    assert(mid <= size);

    input signal vec[size];
    output signal low[mid];
    output signal hi[size - mid];

    for (var i = 0; i < mid; i++){
        low[i] <== vec[i];
    }
    for (var i = mid; i < size; i++){
        hi[i - mid] <== vec[i];
    }
}

//TODO(Bhargav):- Verify if we need constraint to check sum(bits) = value.
template IndexToFieldBitvector(value, num_bits) {
    output signal bitvector[num_bits]; 
    signal bits[num_bits]; 

    for (var i = 0; i < num_bits; i++) {
        bits[i] <-- (value >> i) & 1;
        bits[i] * (bits[i] - 1) === 0;
        bitvector[num_bits - 1 - i] <== bits[i];
    }
}

template TruncateVec(start_index, end_index, size){
    // assert(start_index < end_index);
    input signal vec[size];
    output signal trun_vec[end_index - start_index];

    for (var i = start_index; i < end_index; i++){
        trun_vec[i - start_index] <== vec[i];
    }
}

template OpeningAccumulator(num_commitments, opening_point_len){
    // input HyperKZGCommitment() commitments[num_commitments];
    input signal opening_point[opening_point_len];
    input signal claims[num_commitments];
    input Transcript() transcript;
    output Transcript() up_transcript;
    output VerifierOpening(opening_point_len) r1cs_verifier_opening;
    signal rho_powers[num_commitments];
    
    signal rho ;
    (up_transcript, rho) <== ChallengeScalar()(transcript);

    rho_powers[0] <== 1;
    for (var i = 1; i < num_commitments; i++) {
        rho_powers[i] <==  rho_powers[i-1] * rho;
    }
    
    r1cs_verifier_opening.rho <== rho;

    r1cs_verifier_opening.claim <== ComputeDotProduct(num_commitments)(rho_powers, claims);

    r1cs_verifier_opening.opening_point <== opening_point;
}


template EvaluateIdentityPoly(size){
    input signal r[size];
    output signal eval;

    signal mul[size - 1]; 
    signal add[size - 2]; 
    signal two; 

    two <== 2;
    mul[0] <== (r[0] *  two);  
    
    for (var i = 1; i < size - 1; i++)
    {    
        add[i - 1] <== (mul[i - 1] + r[i]);
        
        mul[i] <== (add[i - 1] *  two);
    }

    eval <== (mul[size - 2] +  r[size - 1]);
}
 