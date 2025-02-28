pragma circom 2.2.1;
include "./utils.circom";
include "./../fields/non_native/utils.circom";
include "./../node_modules/circomlib/circuits/bitify.circom";

template G1Double() {
    input G1Projective() op1;
    output G1Projective() out;

    signal b <== 3;
    signal b3 <== 3 * b;

    signal g0 <== op1.y * op1.y;
    signal z3 <== 8 * g0;

    signal g1 <== op1.y * op1.z;
    signal g2 <== op1.z * op1.z;
    signal g3 <== b3 * g2;

    signal x3 <== g3 * z3;
    signal y3 <== g0 + g3;
    out.z <== g1 * z3;

    signal t1 <== 2 * g3;
    signal t2 <== t1 + g3;
    signal t0 <== g0 - t2;

    out.y <== x3 + t0 * y3;
    signal r1 <== op1.x * op1.y;
    out.x <== 2 * t0 * r1;
}

template G1Add() {
    input G1Projective() op1;
    input G1Projective() op2;
    output G1Projective() out;

    signal b <== 3;
    signal b3 <== 3 * b;

    signal t0 <== op1.x * op2.x;
    signal t1 <== op1.y * op2.y;
    signal t2 <== op1.z * op2.z;
    signal t3 <== op1.x + op1.y;
    signal t4 <== op2.x + op2.y;
    signal t5 <== t3 * t4;
    signal t6 <== t0 + t1;
    signal t7 <== t5 - t6;
    signal t8 <== op1.y + op1.z;
    signal t9 <== op2.y + op2.z;
    signal t10 <== t8 * t9;
    signal t11 <== t1 + t2;
    signal t12 <== t10 - t11;
    signal t13 <== op1.x + op1.z;
    signal t14 <== op2.x + op2.z;
    signal t15 <== t13 * t14;
    signal t16 <== t0 + t2;
    signal t17 <== t15 - t16;
    signal t18 <== t0 + t0;
    signal t19 <== t18 + t0;
    signal t20 <== b3 * t2;
    signal t21 <== t1 + t20;
    signal t22 <== t1 - t20;
    signal t23 <== b3 * t17;
    signal t24 <== t12 * t23;
    signal t25 <== t7 * t22;
    signal t26 <== t25 - t24;
    signal t27 <== t23 * t19;
    signal t28 <== t22 * t21;
    signal t29 <== t28 + t27;
    signal t30 <== t19 * t7;
    signal t31 <== t21 * t12;
    signal t32 <== t31 + t30;

    out.x <== t26;
    out.y <== t29;
    out.z <== t32;
}

template G1Mul() {
    input G1Projective() op1;
    input Fq() op2;
    output G1Projective() out;
    var n = 254;

    signal bits[n] <== ScalarToBits()(op2);

    G1Projective() intermediateProducts[n + 1][3];
    G1Projective() condProduct[n][2];
    
    // Initialise result to identity
    G1Projective() identity;
    (identity.x, identity.y, identity.z) <== (0, 1, 0);
    intermediateProducts[0][0] <== identity;
   
    for(var i = 0; i < n; i++) {
        // Double
        intermediateProducts[i][1] <== G1Double()(intermediateProducts[i][0]);
        // Add
        intermediateProducts[i][2] <== G1Add()(intermediateProducts[i][1], op1);
        // Conditionally select double or double-and-add
        (condProduct[i][0].x, condProduct[i][0].y, condProduct[i][0].z) <== (
            (1 - bits[n-1-i]) * intermediateProducts[i][1].x,
            (1 - bits[n-1-i]) * intermediateProducts[i][1].y,
            (1 - bits[n-1-i]) * intermediateProducts[i][1].z
        );
        (condProduct[i][1].x, condProduct[i][1].y, condProduct[i][1].z) <== (
            bits[n-1-i] * intermediateProducts[i][2].x,
            bits[n-1-i] * intermediateProducts[i][2].y,
            bits[n-1-i] * intermediateProducts[i][2].z
        );
        // Add conditional products
        intermediateProducts[i+1][0].x <== condProduct[i][0].x + condProduct[i][1].x;
        intermediateProducts[i+1][0].y <== condProduct[i][0].y + condProduct[i][1].y;
        intermediateProducts[i+1][0].z <== condProduct[i][0].z + condProduct[i][1].z;
    }
    out <== intermediateProducts[n][0];
} 

