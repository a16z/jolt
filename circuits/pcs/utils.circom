// template CombineCommitments(num_commitments){
//     input G1Projective() commitments[num_commitments];
//     input Fq() coeffs;
//     output G1Projective() combine_commitment;

//     G1Projective() int_sum[num_commitments];
//     int_sum[0] <== G1Mul()(commitments[num_commitments - 1], coeffs);

//     G1Projective() temp[num_commitments-2];

//     for (var i = 0; i < num_commitments - 2; i++) {        
//         temp[i] <== G1Add()(int_sum[i] , commitments[num_commitments - 2 - i]); 
//         int_sum[i + 1] <==  G1Mul()(temp[i], coeffs);
//     }

//     int_sum[num_commitments - 1] <== G1Add()(int_sum[num_commitments - 2] , commitments[0]);

//     combine_commitment <== int_sum[num_commitments - 1];
// }

// template ConvertToProjective(num_commitments){
//     input G1Affine() commitments[num_commitments];
//     output G1Projective() projective_commitments[num_commitments];
    
//     for (var i = 0; i < num_commitments; i++) {
//         projective_commitments[i] <== toProjective()(commitments[i]);
//     }
// }

// bus HyperKZGVerifierKey {
//     KZGVerifierKey() kzg_vk;
// }

// bus KZGVerifierKey {
//     G1Affine() g1;
//     G2Affine() g2;
//     G2Affine() beta_g2;
// }

// bus HyperKZGCommitment {
//     G1Affine() commitment;
// }

// bus HyperKZGProof(ell) {
//     G1Affine() com[ell - 1];
//     G1Affine() w[3];
//     Fq() v[3][ell];
// }