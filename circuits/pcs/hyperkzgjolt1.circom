pragma circom 2.2.1;

// bus Fp() {
//     signal limbs[3];
// }

// bus G1Affine {
//     Fp() x;
//     Fp() y;
// }


bus HyperKZGCommitment {
    G1Affine() commitment;
}

bus HyperKZGProof(ell) {
    G1Affine() com[ell - 1];
    G1Affine() w[3];
    signal v[3][ell];
}

template flatten(size1, size2) {
    input signal vec[size1][size2];
    output signal flat_vec[size1 * size2];
    for (var i = 0; i < size1; i++){
        for (var j = 0; j < size2; j++){
            flat_vec[j + i * size2] <-- vec[i][j];
        }
    }
}

template HyperKzgVerifierJolt1(ell){

    input signal point[ell];   
    input HyperKZGProof(ell) pi; 
    input signal P_of_x;   
    input Transcript() transcript;

    signal q_powers[ell];
    signal d_0;
    signal Y[ell + 1];
    Transcript() int_transcript[6];
    signal r;

    output HyperKzgVerifierAdvice()  hyperkzg_verifier_advice;
    

    int_transcript[0] <== AppendPoints(ell - 1)(pi.com, transcript);
    (int_transcript[1], r) <== ChallengeScalar()(int_transcript[0]);

    //TODO(Ashish):- add check of r = 0 || C = identity;
    signal ypos[ell] <== pi.v[0];
    signal yneg[ell] <== pi.v[1];

    for (var i = 0; i < ell; i++){
        Y[i] <== pi.v[2][i];
    }
    Y[ell] <== P_of_x;

  
    signal temp[8 * ell];
    signal two_times_r <== r + r ;
    for (var i = 0; i < ell; i++){
        temp[8 * i] <== (two_times_r * Y[i + 1]);          // LHS
        temp[8 * i + 1] <== (1 - point[ell - i -1]);
        temp[8 * i + 2] <== (r *  temp[8 * i + 1]);
        temp[8 * i + 3] <== (ypos[i] + yneg[i]);           
        temp[8 * i + 4] <== (temp[8 * i + 2] * temp[8 * i + 3]); 
        temp[8 * i + 5] <== (ypos[i] - yneg[i]);           
        temp[8 * i + 6] <== (point[ell - i - 1] * temp[8 * i + 5]); // RHS part 2
        temp[8 * i + 7] <== (temp[8 * i + 4] + temp[8 * i + 6]);  // RHS
        log("i = ", i, "temp[8*i] = ", temp[8 * i], "temp[8*i+7] = ", temp[8 * i + 7]);
        temp[8 * i] ===  temp[8 * i + 7];
    }

    signal flat_v[3 * ell]  <== flatten(3, ell)(pi.v);
    int_transcript[2] <== AppendScalars(3 * ell)(flat_v, int_transcript[1]);
    (int_transcript[3], q_powers) <== ChallengeScalarPowers(ell)(int_transcript[2]);//TODO(Ashish):- Can call ChallengeScalar

    int_transcript[4] <== AppendPoints(3)(pi.w, int_transcript[3]);
    (int_transcript[5], d_0) <== ChallengeScalar()(int_transcript[4]);

    signal d_1 <== d_0  * d_0;
    signal eval1 <== EvalUniPoly(ell - 1)(pi.v[0], q_powers[1]);
    signal eval2 <== EvalUniPoly(ell - 1)(pi.v[1], q_powers[1]);
    signal eval3 <== EvalUniPoly(ell - 1)(pi.v[2], q_powers[1]);
    

    signal eval4 <== (eval2 * d_0);
    signal eval5 <== (eval3 * d_1);
    signal eval6 <== (eval4 + eval5);
    signal  v <== (eval1 + eval6);


    hyperkzg_verifier_advice.r <== r;
    hyperkzg_verifier_advice.d_0 <== d_0;
    hyperkzg_verifier_advice.v <== v;
    hyperkzg_verifier_advice.q_power <== q_powers[1];

}
