
pragma circom 2.2.1;
include "./hyperkzg_utils.circom";
include "./pairing.circom";
include "./../groups/utils.circom";  
include "./../transcript/bn254_transcript.circom";
include "./../spartan/spartan_hyperkzg/utils.circom";

template HyperKzgVerifier(ell){
    input HyperKZGVerifierKey() vk;
    input HyperKZGCommitment() C;
    input Fq() point[ell];   
    input HyperKZGProof(ell) pi; 
    input Fq() P_of_x;   
    input Transcript() transcript;

    Fq() q_powers[ell];
    Fq() d_0;
    Fq() Y[ell + 1];
    Transcript() int_transcript[6];
    Fq() r;

    int_transcript[0] <== AppendPoints(ell - 1)(pi.com, transcript);
    (int_transcript[1], r) <== ChallengeScalar()(int_transcript[0]);

    G1Projective() commitments[ell];
    commitments[0] <== toProjective()(C.commitment);
    for (var i = 0; i < ell - 1; i++){
        commitments[i + 1] <== toProjective()(pi.com[i]);
    }

    Fq() ypos[ell] <== pi.v[0];
    Fq() yneg[ell] <== pi.v[1];

    for (var i = 0; i < ell; i++){
        Y[i] <== pi.v[2][i];
    }
    Y[ell] <== P_of_x;
    
    Fq() one; 
    one.limbs <-- [1, 0, 0];

    Fq() temp[8 * ell];
    Fq() two_times_r <== NonNativeAdd()(r, r);
    for (var i = 0; i < ell; i++){
        temp[8 * i] <== NonNativeMul()(two_times_r, Y[i + 1]);          // LHS
        temp[8 * i + 1] <== NonNativeSub()(one, point[ell - i -1]);
        temp[8 * i + 2] <== NonNativeMul()(r, temp[8 * i + 1]);
        temp[8 * i + 3] <== NonNativeAdd()(ypos[i], yneg[i]);           
        temp[8 * i + 4] <== NonNativeMul()(temp[8 * i + 2], temp[8 * i + 3]); // RHS part 1
        temp[8 * i + 5] <== NonNativeSub()(ypos[i], yneg[i]);           
        temp[8 * i + 6] <== NonNativeMul()(point[ell - i - 1], temp[8 * i + 5]); // RHS part 2
        temp[8 * i + 7] <== NonNativeAdd()(temp[8 * i + 4], temp[8 * i + 6]);  // RHS

        NonNativeEquality()(temp[8 * i], temp[8 * i + 7]);

    }

    Fq() flat_v[3 * ell]  <== flatten(3, ell)(pi.v);
    int_transcript[2] <== AppendScalars(3 * ell)(flat_v, int_transcript[1]);
    (int_transcript[3], q_powers) <== ChallengeScalarPowers(ell)(int_transcript[2]);//TODO(Ashish):- Can call ChallengeScalar

    int_transcript[4] <== AppendPoints(3)(pi.w, int_transcript[3]);
    (int_transcript[5], d_0) <== ChallengeScalar()(int_transcript[4]);
    Fq() d_1 <== NonNativeMul()(d_0 , d_0);


    Fq() eval1 <== EvalUniPoly(ell - 1)(pi.v[0], q_powers[1]);
    Fq() eval2 <== EvalUniPoly(ell - 1)(pi.v[1], q_powers[1]);
    Fq() eval3 <== EvalUniPoly(ell - 1)(pi.v[2], q_powers[1]);
    
    Fq() reduced_eval1 <== NonNativeModulo()(eval1);
    Fq() reduced_eval2 <== NonNativeModulo()(eval2);
    Fq() reduced_eval3 <== NonNativeModulo()(eval3);


    Fq() eval4 <== NonNativeMul()(eval2, d_0);
    Fq() eval5 <== NonNativeMul()(eval3, d_1);
    Fq() eval6 <== NonNativeAdd()(eval4, eval5);
    Fq() v_temp <== NonNativeAdd()(eval1, eval6);
    Fq() v <== NonNativeModulo()(v_temp);
    G1Projective() projective_g1 <== toProjective()(vk.kzg_vk.g1);

    G1Projective() V <== G1Mul()(projective_g1, v);

    Fq() d_0_d_1 <== NonNativeAdd()(d_0, d_1);    
    Fq() q_power_multiplierunreduced <== NonNativeAdd()(one, d_0_d_1); 
    Fq() q_power_multiplier <== NonNativeModulo()(q_power_multiplierunreduced); 

    G1Projective() int_sum[ell + 1];    
    (int_sum[0].x, int_sum[0].y, int_sum[0].z) <== (0, 1, 0);

    G1Projective() temp_point[ell];
    for (var i = 0; i < ell; i++) {
        temp_point[i] <== G1Mul()(commitments[i], q_powers[i]); //TODO(Anuj):- Can skip first iteration since q_powers[0] = 1;
        int_sum[i + 1] <== G1Add()(int_sum[i], temp_point[i]);
    }

    G1Projective() projective_w_0 <== toProjective()(pi.w[0]);
    G1Projective() projective_w_1 <== toProjective()(pi.w[1]);
    G1Projective() projective_w_2 <== toProjective()(pi.w[2]);

    G1Projective() B <== G1Mul()(int_sum[ell], q_power_multiplier);
    G1Projective() temp11 <== G1Mul()(projective_w_2, d_0);     // d_0 * w[2]
    G1Projective() temp12 <== G1Mul()(temp11, r);               // r * d_0 * w[2]
    G1Projective() temp13 <== G1Negative()(projective_w_1);     // - w[1]; 
    G1Projective() temp14 <== G1Add()(temp12, temp13);          // - w[1] + r * d_0 * w[2]; 
    G1Projective() temp15 <== G1Mul()(temp14, d_0);             // d_0 * (- w[1] + r * d_0 * w[2]); 
    G1Projective() temp16 <== G1Add()(temp15, projective_w_0);  // w[0] + d_0 * (- w[1] +  r * d_0 * w[2]); 
    G1Projective() w_dash <== G1Mul()(temp16, r);               // r * (w[0] + d_0 * (- w[1] + r * d_0 * w[2])); 
    G1Projective() temp17 <== G1Add()(w_dash, B);     
    G1Projective() negative_V <== G1Negative()(V);
    G1Projective() L <== G1Add()(temp17, negative_V); 

    G1Projective() temp18 <== G1Add()(temp11, projective_w_1);   //   w[1] +  d_0 * w[2]
    G1Projective() temp19 <== G1Mul()(temp18, d_0);       //  d_0 * w[1] +  d_0 * d_0 * w[2]
    G1Projective() R <== G1Add()(projective_w_0, temp19);
    G1Projective() R_neg <== G1Negative()(R);

    Fp12() p1 <== MillerLoop()(vk.kzg_vk.beta_g2, R_neg);
    Fp12() p2 <== MillerLoop()(vk.kzg_vk.g2, L);
    Fp12() temp20 <== Fp12mul()(p1, p2);
    Fp12() multi_pairing <== FinalExp()(temp20);

    Fp2() zero_2, one_2;
    (zero_2.x, zero_2.y) <== (0,0);
    (one_2.x, one_2.y) <== (1,0);
    Fp6() zero_6, one_6;
    (zero_6.x, zero_6.y, zero_6.z) <== (zero_2, zero_2, zero_2);
    (one_6.x, one_6.y, one_6.z) <== (one_2, zero_2, zero_2);
    Fp12() one_12;
    (one_12.x, one_12.y) <== (one_6,zero_6);
    multi_pairing === one_12;

}     

template CombineCommitments(num_commitments){
    input G1Projective() commitments[num_commitments];
    input Fq() coeffs;
    output G1Projective() combine_commitment;

    G1Projective() int_sum[num_commitments];
    int_sum[0] <== G1Mul()(commitments[num_commitments - 1], coeffs);

    G1Projective() temp[num_commitments-2];

    for (var i = 0; i < num_commitments - 2; i++) {        
        temp[i] <== G1Add()(int_sum[i] , commitments[num_commitments - 2 - i]); 
        int_sum[i + 1] <==  G1Mul()(temp[i], coeffs);
    }

    int_sum[num_commitments - 1] <== G1Add()(int_sum[num_commitments - 2] , commitments[0]);

    combine_commitment <== int_sum[num_commitments - 1];
}


//This works when each inner vec is of same size.
template flatten(size1, size2) {
    input Fq() vec[size1][size2];
    output Fq() flat_vec[size1 * size2];
    for (var i = 0; i < size1; i++){
        for (var j = 0; j < size2; j++){
            flat_vec[j + i * size2] <-- vec[i][j];
        }
    }
}

bus HyperKZGVerifierKey {
    KZGVerifierKey() kzg_vk;
}

bus KZGVerifierKey {
    G1Affine() g1;
    G2Affine() g2;
    G2Affine() beta_g2;
}


bus HyperKZGProof(ell) {
    G1Affine() com[ell - 1];
    G1Affine() w[3];
    Fq() v[3][ell];
}
