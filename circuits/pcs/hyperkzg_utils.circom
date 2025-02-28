pragma circom 2.2.1;

bus HyperKZGCommitment {
    G1Affine() commitment;
}

// template ConvertToProjective(num_commitments){
//     input G1Affine() commitments[num_commitments];
//     output G1Projective() projective_commitments[num_commitments];
    
//     for (var i = 0; i < num_commitments; i++) {
//         projective_commitments[i] <== toProjective()(commitments[i]);
//     }
// }

bus HyperKZGVerifierKeyNN() {
    KZGVerifierKeyNN() kzg_vk;
}

bus KZGVerifierKeyNN() {
    G1AffineNN() g1;
    G2AffineNN() g2;
    G2AffineNN() beta_g2;
}

bus G1AffineNN() {
    Fq() x;
    Fq() y;
}

bus G2AffineNN() {
    Fp2NN() x;
    Fp2NN() y;
}

bus Fp2NN() {
    Fq() x;
    Fq() y;
}