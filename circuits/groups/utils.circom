pragma circom 2.2.1;
include "./../fields/field/fp2.circom";


bus G1Affine {
    signal x;
    signal y;
}


bus G2Affine {
    Fp2() x;
    Fp2() y;
}

bus G1Projective() {
    signal x;
    signal y;
    signal z;
}

bus G2Projective() {
    Fp2() x;
    Fp2() y;
    Fp2() z;
}

template ScalarToBits(){
    input Fq() op;
    output signal bits[254];
    signal bits_limb_0[125] <== Num2Bits(125)(op.limbs[0]);
    signal bits_limb_1[125] <== Num2Bits(125)(op.limbs[1]);
    signal bits_limb_2[4] <== Num2Bits(4)(op.limbs[2]);
    for (var i = 0; i < 125; i++) {
        bits[i] <== bits_limb_0[i];
        bits[125 + i] <== bits_limb_1[i];
    }
    for (var i = 0; i < 4; i++) {
        bits[250 + i] <== bits_limb_2[i];
    }
}

template toProjective(){
    input G1Affine() affine_point; 
    output G1Projective() projective_point; 

    projective_point.x <== affine_point.x;
    projective_point.y <== affine_point.y;
    projective_point.z <== 1;
}

template G1Negative() {
    input G1Projective() point;
    output G1Projective() neg_point;
    neg_point.x <== point.x;
    neg_point.y <== -point.y;
    neg_point.z <== point.z;
}

template toProjectiveNew(){
    input G1Affine() affine_point; 
    output G1Projective() projective_point; 

    var n = 256;
    signal x_bits[n] <== Num2Bits(n)(affine_point.x);
    signal y_bits[n] <== Num2Bits(n)(affine_point.y);

    signal x_bits_comp[n];
    signal y_bits_comp[n];

    for (var i = 0; i < n; i++) {
        x_bits_comp[i] <== 1 - x_bits[i];
        y_bits_comp[i] <== 1 - y_bits[i];
    }

    signal x_prod[n];
    signal y_prod[n];
    x_prod[0] <==  x_bits_comp[0];
    y_prod[0] <==  y_bits_comp[0];

    for (var i = 1; i < n; i++) {
        x_prod[i] <== x_prod[i - 1] * x_bits_comp[i];
        y_prod[i] <== y_prod[i - 1] * y_bits_comp[i];
    }

    signal identity_indicator <== x_prod[n -1] * y_prod[n - 1];
    projective_point.x <== (1 - identity_indicator) * (affine_point.x);
    projective_point.y <== (1 - identity_indicator) * (affine_point.y) + identity_indicator;
    projective_point.z <== (1 - identity_indicator);
}

template G1ToAffine(){
    input G1Projective() op1;
    output G1Affine() out;

    var n = 256;

    signal op1_z_inv <-- op1.z!=0 ? 1/op1.z : 0;

    signal inv_check <== op1_z_inv * op1.z;

    signal z_bits[n] <== Num2Bits(n)(op1.z);

    signal z_bits_comp[n];

    for (var i = 0; i < n; i++) {
        z_bits_comp[i] <== 1 - z_bits[i];
    }

    signal z_prod[n];

    z_prod[0] <==  z_bits_comp[0];

    for (var i = 1; i < n; i++) {
        z_prod[i] <== z_prod[i - 1] * z_bits_comp[i];
    }

    // z_prod[n-1] is identity_indicator
    z_prod[n-1] + (1 - z_prod[n-1]) * inv_check === 1;

    out.x <== op1.x * op1_z_inv;
    out.y <== op1.y * op1_z_inv;
}