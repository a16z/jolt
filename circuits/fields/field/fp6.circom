pragma circom 2.2.1;
include "./fp2.circom";

bus Fp6() {
    Fp2() x;
    Fp2() y;
    Fp2() z;
}

template Fp6add() {
    input Fp6() op1;
    input Fp6() op2;
    output Fp6() out;

    (out.x, out.y, out.z) <== (Fp2add()(op1.x, op2.x), Fp2add()(op1.y, op2.y), Fp2add()(op1.z, op2.z));
}

template Fp6sub() {
    input Fp6() op1;
    input Fp6() op2;
    output Fp6() out;

    (out.x, out.y, out.z) <== (Fp2sub()(op1.x, op2.x), Fp2sub()(op1.y, op2.y), Fp2sub()(op1.z, op2.z));
}

template Fp6neg() {
    input Fp6() op1;
    output Fp6() out;

    out.x <== Fp2mulbyfp()(-1, op1.x);
    out.y <== Fp2mulbyfp()(-1, op1.y);
    out.z <== Fp2mulbyfp()(-1, op1.z);
}

template Fp6mul() {
    input Fp6() op1;
    input Fp6() op2;
    output Fp6() out;

    Fp2() shi;
    (shi.x, shi.y) <== (9, 1);
    Fp2() t1 <== Fp2mul()(op1.y, op2.z);
    Fp2() t2 <== Fp2muladd()(op1.z, op2.y, t1);
    Fp2() t3 <== Fp2mul()(op1.x, op2.x);
    out.x <== Fp2muladd()(shi, t2, t3);
    Fp2() t5 <== Fp2mul()(op1.x, op2.y);
    Fp2() t6 <== Fp2muladd()(op1.y, op2.x, t5);
    Fp2() t7 <== Fp2mul()(op1.z, op2.z);
    out.y <== Fp2muladd()(shi, t7, t6);
    Fp2() t9 <== Fp2mul()(op1.x, op2.z);
    Fp2() t10 <== Fp2muladd()(op1.y, op2.y, t9);
    out.z <== Fp2muladd()(op1.z, op2.x, t10);
}

function inverse_6(p0, p1, q0, q1, r0, r1) {
    // t0 = a0^2, t1 = a1^2, t2 = a2^2
    var t0[2] = [p0 * p0 - p1 * p1, 2 * p0 * p1];
    var t1[2] = [q0 * q0 - q1 * q1, 2 * q0 * q1];
    var t2[2] = [r0 * r0 - r1 * r1, 2 * r0 * r1];

    // t3 = a0.a1, t4 = a0.a2, t5 = a1.a2 ?? a2.a3
    var t3[2] = [p0 * q0 - p1 * q1, p0 * q1 + p1 * q0];
    var t4[2] = [p0 * r0 - p1 * r1, p0 * r1 + p1 * r0];
    var t5[2] = [q0 * r0 - q1 * r1, q0 * r1 + q1 * r0];

    // c0 = t0 - shi.t5
    // c1 = shi.t2 - t3
    // c2 = t1 - t4. Paper has a typo here.
    var c0[2] = [t0[0] - 9 * t5[0] + t5[1], t0[1] - 9 * t5[1] - t5[0]];
    var c1[2] = [9 * t2[0] - t2[1] - t3[0], 9 * t2[1] + t2[0] - t3[1]];
    // var c2[2] = [t1[0] * t4[0] - t1[1] * t4[1], t1[0] * t4[1] + t1[1] * t4[0]];
    var c2[2] = [t1[0] - t4[0], t1[1] - t4[1]];

    // t6 = a0.c0
    // t6 += shi.a2.c1
    var t6[2] = [p0 * c0[0] - p1 * c0[1], p0 * c0[1] + p1 * c0[0]];
    var a2c1[2] = [r0 * c1[0] - r1 * c1[1], r0 * c1[1] + r1 * c1[0]];
    t6 = [9 * a2c1[0] - a2c1[1] + t6[0], 9 * a2c1[1] + a2c1[0] + t6[1]];

    // t6 += shi.a1.c2
    var a1c2[2] = [q0 * c2[0] - q1 * c2[1], q0 * c2[1] + q1 * c2[0]];
    t6 = [9 * a1c2[0] - a1c2[1] + t6[0], 9 * a1c2[1] + a1c2[0] + t6[1]];

    // t6 = inv(t6)
    var temp = t6[0] * t6[0] + t6[1] * t6[1];
    t6 = [t6[0] / temp, -t6[1] / temp];

    // c0 *= t6, c1 *= t6, c2 *= t6
    var d0[2] = [c0[0] * t6[0] - c0[1] * t6[1], c0[0] * t6[1] + c0[1] * t6[0]];
    var d1[2] = [c1[0] * t6[0] - c1[1] * t6[1], c1[0] * t6[1] + c1[1] * t6[0]];
    var d2[2] = [c2[0] * t6[0] - c2[1] * t6[1], c2[0] * t6[1] + c2[1] * t6[0]];
    
    return [[d0[0], d0[1]], [d1[0], d1[1]], [d2[0], d2[1]]];
}

template Fp6inv() {
    Fp6() input op1;
    Fp6() output out;

    Fp2() zero_2, one_2;
    (zero_2.x, zero_2.y) <== (0, 0);
    (one_2.x, one_2.y) <== (1, 0);

    Fp6() one_6;
    (one_6.x, one_6.y, one_6.z) <== (one_2, zero_2, zero_2);

    var res[3][2] = inverse_6(op1.x.x, op1.x.y, op1.y.x, op1.y.y, op1.z.x, op1.z.y);
    (out.x.x, out.x.y) <-- (res[0][0], res[0][1]);
    (out.y.x, out.y.y) <-- (res[1][0], res[1][1]);
    (out.z.x, out.z.y) <-- (res[2][0], res[2][1]);

    Fp6() mult <== Fp6mul()(op1, out);
    mult === one_6;
}
// component main = Fp6add();

// component main = Fp6inv();

// component main = Fp6neg();
