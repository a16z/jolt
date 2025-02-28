pragma circom 2.2.1;
include "./fp2.circom";

bus Fp4() {
    Fp2() x;
    Fp2() y;
}

template Fp4Square() {
    input Fp4() op1;
    output Fp2() out0, out1;

    Fp2() t0 <== Fp2mul()(op1.x, op1.x);
    Fp2() t1 <== Fp2mul()(op1.y, op1.y);

    Fp2() shi;
    (shi.x, shi.y) <== (9, 1);
    out0 <== Fp2muladd()(t1, shi, t0);

    Fp2() c1 <== Fp2add()(op1.x, op1.y);
    Fp2() c2 <== Fp2mul()(c1, c1);
    Fp2() c3 <== Fp2sub()(c2, t0);
    out1 <== Fp2sub()(c3, t1);
}