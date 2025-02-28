pragma circom 2.2.1;

include "./../fields/field/fp2.circom";
include "./utils.circom";
include "./../fields/non_native/utils.circom";
include "./../node_modules/circomlib/circuits/bitify.circom";

template G2Double() {
    input G2Projective() op1;
    output G2Projective() out;

    var b_arr[2] = [
        19485874751759354771024239261021720505790618469301721065564631296452457478373,
        266929791119991161246907387137283842545076965332900288569378510910307636690
    ];

    Fp2() u, b, three;
    (u.x, u.y) <== (9, 1);
    (b.x, b.y) <== (b_arr[0], b_arr[1]);
    (three.x, three.y) <== (3, 0);

    Fp2() b3 <== Fp2mul()(three, b);
    Fp2() g0 <== Fp2mul()(op1.y, op1.y);

    Fp2() eight;
    (eight.x, eight.y) <== (8, 0);
    Fp2() z3 <== Fp2mul()(eight, g0);

    Fp2() g1 <== Fp2mul()(op1.y, op1.z);
    Fp2() g2 <== Fp2mul()(op1.z, op1.z);
    Fp2() g3 <== Fp2mul()(b3, g2);

    Fp2() x3 <== Fp2mul()(g3, z3);
    Fp2() y3 <== Fp2add()(g0, g3);
    out.z <== Fp2mul()(g1, z3);
    
    Fp2() two;
    (two.x, two.y) <== (2, 0);
    Fp2() t1 <== Fp2mul()(two, g3);
    Fp2() t2 <== Fp2add()(t1, g3);
    Fp2() t0 <== Fp2sub()(g0, t2);
    Fp2() t3 <== Fp2mul()(y3, t0);

    out.y <== Fp2add()(t3, x3);
    Fp2() r1 <== Fp2mul()(op1.x, op1.y);
    Fp2() r2 <== Fp2mul()(t0, r1);
    out.x <== Fp2mul()(two, r2);
}

template G2Add() {
    input G2Projective() op1;
    input G2Projective() op2;
    output G2Projective() out;

    var b_arr[2] = [
        19485874751759354771024239261021720505790618469301721065564631296452457478373,
        266929791119991161246907387137283842545076965332900288569378510910307636690
    ];

    Fp2() u, b, three;
    (u.x, u.y) <== (9, 1);
    (b.x, b.y) <== (b_arr[0], b_arr[1]);
    (three.x, three.y) <== (3, 0);
    Fp2() b3 <== Fp2mul()(three, b);

    Fp2() t0 <== Fp2mul ()(op1.x, op2.x);
    Fp2() t1 <== Fp2mul ()(op1.y, op2.y);
    Fp2() t2 <== Fp2mul ()(op1.z, op2.z);
    Fp2() t3 <== Fp2add ()(op1.x, op1.y);
    Fp2() t4 <== Fp2add ()(op2.x, op2.y);
    Fp2() t5 <== Fp2mul ()(t3, t4);
    Fp2() t6 <== Fp2add ()(t0, t1);
    Fp2() t7 <== Fp2sub ()(t5, t6);
    Fp2() t8 <== Fp2add ()(op1.y, op1.z);
    Fp2() t9 <== Fp2add ()(op2.y, op2.z);
    Fp2() t10 <== Fp2mul ()(t8, t9);
    Fp2() t11 <== Fp2add ()(t1, t2);
    Fp2() t12 <== Fp2sub ()(t10, t11);
    Fp2() t13 <== Fp2add ()(op1.x, op1.z);
    Fp2() t14 <== Fp2add ()(op2.x, op2.z);
    Fp2() t15 <== Fp2mul ()(t13, t14);
    Fp2() t16 <== Fp2add ()(t0, t2);
    Fp2() t17 <== Fp2sub ()(t15, t16);
    Fp2() t18 <== Fp2add ()(t0, t0);
    Fp2() t19 <== Fp2add ()(t18, t0);
    Fp2() t20 <== Fp2mul ()(b3, t2);
    Fp2() t21 <== Fp2add ()(t1, t20);
    Fp2() t22 <== Fp2sub ()(t1, t20);
    Fp2() t23 <== Fp2mul ()(b3, t17);
    Fp2() t24 <== Fp2mul ()(t12, t23);
    Fp2() t25 <== Fp2mul ()(t7, t22);
    Fp2() t26 <== Fp2sub ()(t25, t24);
    Fp2() t27 <== Fp2mul ()(t23, t19);
    Fp2() t28 <== Fp2mul ()(t22, t21);
    Fp2() t29 <== Fp2add ()(t27, t28);
    Fp2() t30 <== Fp2mul ()(t19, t7);
    Fp2() t31 <== Fp2mul ()(t21, t12);
    Fp2() t32 <== Fp2add ()(t31, t30);

    out.x <== t26;
    out.y <== t29;
    out.z <== t32;
}

template G2Mul() {
    input G2Projective() op1;
    input Fq() op2;
    output G2Projective() out;

    var n = 254;
    signal bits[n] <== ScalarToBits()(op2);

    G2Projective() intermediateProducts[n + 1][3];
    G2Projective() condProduct[n][2];

    // Initialise result to identity
    Fp2() zero_2, one_2;
    (zero_2.x, zero_2.y) <== (0, 0);
    (one_2.x, one_2.y) <== (1, 0);
    G2Projective() identity;
    (identity.x, identity.y, identity.z) <== (zero_2, one_2, zero_2);
    intermediateProducts[0][0] <== identity;
    
    for(var i = 0; i < n; i++) {
        // Double
        intermediateProducts[i][1] <== G2Double()(intermediateProducts[i][0]);

        // Add
        intermediateProducts[i][2] <== G2Add()(intermediateProducts[i][1], op1);

        // Conditionally select double or double-and-add
        (condProduct[i][0].x, condProduct[i][0].y, condProduct[i][0].z) <== (
            Fp2mulbyfp()((1 - bits[n-1-i]), intermediateProducts[i][1].x),
            Fp2mulbyfp()((1 - bits[n-1-i]), intermediateProducts[i][1].y),
            Fp2mulbyfp()((1 - bits[n-1-i]), intermediateProducts[i][1].z)
        );
        (condProduct[i][1].x, condProduct[i][1].y, condProduct[i][1].z) <== (
            Fp2mulbyfp()(bits[n-1-i], intermediateProducts[i][2].x),
            Fp2mulbyfp()(bits[n-1-i], intermediateProducts[i][2].y),
            Fp2mulbyfp()(bits[n-1-i], intermediateProducts[i][2].z)
        );

        // Add conditional products
        intermediateProducts[i+1][0].x <== Fp2add()(condProduct[i][0].x, condProduct[i][1].x);
        intermediateProducts[i+1][0].y <== Fp2add()(condProduct[i][0].y, condProduct[i][1].y);
        intermediateProducts[i+1][0].z <== Fp2add()(condProduct[i][0].z, condProduct[i][1].z);
    }

    out <== intermediateProducts[n][0];
}
template G2toProjective(){
    input G2Affine() affine_point; 
    output G2Projective() projective_point; 

    projective_point.x <== affine_point.x;
    projective_point.y <== affine_point.y;
    projective_point.z.x <== 1;
    projective_point.z.y <== 0;
}

// component main = G2Mul();