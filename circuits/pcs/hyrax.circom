pragma circom 2.2.1;

include "./../groups/grumpkin_g1.circom";

template HyraxVerifier(num_vars){
    input HyraxCommitment(num_vars) commit;
    input EvalProof(num_vars) proof;
    input PedersenGenerators(num_vars) setup;
    input Fq() evaluation;
    input Fq() eval_point[num_vars];

    var right_num_vars = num_vars >> 1; //rows
    var left_num_vars = num_vars - right_num_vars; //cols

    Fq() r_vars[right_num_vars];
    Fq() l_vars[left_num_vars];

    (l_vars, r_vars) <== SplitAt(left_num_vars, num_vars)(eval_point);

    var L_size = 1 << left_num_vars;
    var R_size = 1 << right_num_vars;

    Fq() L[L_size];
    Fq() R[R_size];

    (L, R) <== (Evals(left_num_vars)(l_vars), Evals(right_num_vars)(r_vars));
    
    signal L_elem <== L[0].limbs[0] + (1 << 125) * L[0].limbs[1] + (1 << 250) * L[0].limbs[2];
    signal R_elem <== R[0].limbs[0] + (1 << 125) * R[0].limbs[1] + (1 << 250) * R[0].limbs[2];

    Fq() eval <== ComputeDotProduct(R_size)(R, proof.tau);
    
    NonNativeEquality()(evaluation, eval);

    G1Projective() claimed_commitment[R_size];
   
    claimed_commitment[0] <== G1Mul()(setup.gens[0], proof.tau[0]);
  
    for (var i = 1; i < R_size; i++){
        claimed_commitment[i] <== G1Add()(claimed_commitment[i-1], G1Mul()(setup.gens[i], proof.tau[i]));
    }

    G1Projective() derived_commitment[L_size];
    
    derived_commitment[0] <== G1Mul()(commit.row_commitments[0], L[0]);

    for (var i = 1; i < L_size; i++){
        derived_commitment[i] <== G1Add()(derived_commitment[i-1], G1Mul()(commit.row_commitments[i], L[i]));
    }

    signal expected_x <== claimed_commitment[R_size - 1].z * derived_commitment[L_size - 1].x;
    signal actual_x <== claimed_commitment[R_size - 1].x * derived_commitment[L_size - 1].z;
    
    signal expected_y <== claimed_commitment[R_size - 1].z * derived_commitment[L_size - 1].y ;
    signal actual_y <== claimed_commitment[R_size - 1].y * derived_commitment[L_size - 1].z;
    
    expected_x === actual_x;
    expected_y === actual_y;

}


template VerifyEval(num_vars){
    input HyraxCommitment(num_vars) commit;
    input EvalProof(num_vars) proof;
    input PedersenGenerators(num_vars) setup;
    input Fq() evaluation;
    input Fq() eval_point[num_vars];

    var right_num_vars = num_vars >> 1; //rows
    var left_num_vars = num_vars - right_num_vars; //cols

 
    Fq() r_vars[right_num_vars];
    Fq() l_vars[left_num_vars];

    (l_vars, r_vars) <== SplitAt(left_num_vars, num_vars)(eval_point);


    var L_size = 1 << left_num_vars;
    var R_size = 1 << right_num_vars;

    Fq() L[L_size];
    Fq() R[R_size];

    (L, R) <== (Evals(left_num_vars)(l_vars), Evals(right_num_vars)(r_vars));
    signal L_elem <== L[0].limbs[0] + (1 << 125) * L[0].limbs[1] + (1 << 250) * L[0].limbs[2];
    signal R_elem <== R[0].limbs[0] + (1 << 125) * R[0].limbs[1] + (1 << 250) * R[0].limbs[2];

    Fq() eval <== ComputeDotProductNew(R_size)(R, proof.tau);
    
    NonNativeEquality()(evaluation, eval);

    G1Projective() claimed_commitment[R_size];

   
    claimed_commitment[0] <== G1Mul()(setup.gens[0], proof.tau[0]);
  
    for (var i = 1; i < R_size; i++){
        claimed_commitment[i] <== G1Add()(claimed_commitment[i-1], G1Mul()(setup.gens[i], proof.tau[i]));
    }

    G1Projective() derived_commitment[L_size];
    
    derived_commitment[0] <== G1Mul()(commit.row_commitments[0], L[0]);
    G1Affine() t1 <== G1ToAffine()(derived_commitment[0]);
  
    G1Affine() t2 <== G1ToAffine()(commit.row_commitments[0]);

    for (var i = 1; i < L_size; i++){
        derived_commitment[i] <== G1Add()(derived_commitment[i-1], G1Mul()(commit.row_commitments[i], L[i]));
    }


    signal expected_x <== claimed_commitment[R_size - 1].z * derived_commitment[L_size - 1].x;
    signal  actual_x <==
        claimed_commitment[R_size - 1].x * derived_commitment[L_size - 1].z;
    
    expected_x === actual_x;

   signal expected_y <==  claimed_commitment[R_size - 1].z * derived_commitment[L_size - 1].y ;
   signal  actual_y <==
        claimed_commitment[R_size - 1].y * derived_commitment[L_size - 1].z;
    expected_y === actual_y;

}

//eval proof 
bus EvalProof(num_vars) {
    Fq() tau[2 ** (num_vars >> 1)];
}

bus PedersenGenerators(num_vars) {
    G1Projective() gens[2 ** (num_vars >> 1)];
}

bus HyraxCommitment(num_vars) {
    G1Projective() row_commitments[2 ** (num_vars - (num_vars >> 1))];
}