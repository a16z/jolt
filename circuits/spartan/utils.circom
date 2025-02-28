pragma circom 2.2.1;
include "./../fields/non_native/utils.circom";
include "./../node_modules/circomlib/circuits/bitify.circom";


bus PostponedEval(point_len) {
    Fq() point[point_len];
    Fq() eval;
}

bus UniPoly(degree) {
    Fq() coeffs[degree + 1];
}

bus PrimarySumcheckOpenings(NUM_MEMORIES, NUM_INSTRUCTIONS) {
   
    Fq() E_poly_openings[NUM_MEMORIES];

    Fq() flag_openings[NUM_INSTRUCTIONS];

    Fq() lookup_outputs_opening;
}

bus SumcheckInstanceProof(degree, rounds) {
    UniPoly(degree) uni_polys[rounds];
}

bus PrimarySumcheck(degree, rounds, NUM_MEMORIES, NUM_INSTRUCTIONS) {
    SumcheckInstanceProof(degree, rounds) sumcheck_proof;
    PrimarySumcheckOpenings(NUM_MEMORIES, NUM_INSTRUCTIONS) openings;
}

function modulus_bits(field) {
    
    var p_bits[254] = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1];
    var q_bits[254] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1];
    var mod_bits[2][254] = [q_bits, p_bits];
    
    return mod_bits[field];
}

template FqRangeCheck(field) {
    input Fq() a;

    signal a0[125];
    signal a1[125];
    signal a2[4];

    a0 <== Num2Bits(125)(a.limbs[0]);
    a1 <== Num2Bits(125)(a.limbs[1]);
    a2 <== Num2Bits(4)(a.limbs[2]);

    signal r0[126], r1[126], r2[5];

    (r0[0], r1[0], r2[0]) <== (0, 0, 0);

    for(var i = 0; i < 125; i++) {
        r0[i+1] <== r0[i] + a0[i] * (1 << i);
        r1[i+1] <== r1[i] + a1[i] * (1 << i);
    }

    for(var i = 0; i < 4; i++) {
        r2[i+1] <== r2[i] + a2[i] * (1 << i);
    }

    r0[125] === a.limbs[0];
    r1[125] === a.limbs[1];
    r2[4] === a.limbs[2];

    signal A[254];
    for(var i = 0; i < 125; i++) {
        A[i] <== a0[i];
        A[125+i] <== a1[i];
    }
    
    for(var i = 0; i < 4; i++) {
        A[250 + i] <== a2[i];
    }

    signal lt[254];
    var q[254] = modulus_bits(field);
    
    lt[0] <== (1 - A[0]) * q[0];

    for(var i = 1; i < 254; i++) {
        lt[i] <== (1 - A[i]) * q[i] + (A[i] * q[i] + (1 - A[i]) * (1 - q[i])) * lt[i - 1];
    }

    lt[253] === 1;
}