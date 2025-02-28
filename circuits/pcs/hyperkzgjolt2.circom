include "./pairing.circom";
include "utils.circom";

template HyperKzgVerifierJolt2(ell){
    input HyperKZGVerifierKey() vk;
    input HyperKZGProof(ell) pi; 
    input HyperKZGCommitment() C; 

    input Fq() r;
    input Fq() d_0;
    input Fq() v;
    input Fq() q_power;

    G1Projective() commitments[ell];
    commitments[0] <== toProjectiveNew()(C.commitment);
   
    for (var i = 0; i < ell - 1; i++){
        commitments[i + 1] <== toProjectiveNew()(pi.com[i]);
    }

    Fq() d_1 <== NonNativeMul()(d_0  , d_0);

    Fq() one;
    one.limbs <== [1, 0, 0];


    G1Projective() projective_g1 <== toProjectiveNew()(vk.kzg_vk.g1);
    G1Projective() V <== G1Mul()(projective_g1, v);
    
    Fq() d_0_d_1 <== NonNativeAdd()(d_0, d_1);    
    Fq() q_power_multiplierunreduced <== NonNativeAdd()(one, d_0_d_1); 
    Fq() q_power_multiplier <== NonNativeModulo()(q_power_multiplierunreduced); 


    G1Projective() int_sum <== CombineCommitments(ell)(commitments, q_power);


    G1Projective() projective_w_0 <== toProjectiveNew()(pi.w[0]);
    G1Projective() projective_w_1 <== toProjectiveNew()(pi.w[1]);
    G1Projective() projective_w_2 <== toProjectiveNew()(pi.w[2]);

    G1Projective() B <== G1Mul()(int_sum, q_power_multiplier);
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

