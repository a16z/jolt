pragma circom 2.2.1;
include "./fp4.circom";
include "./fp6.circom";

bus Fp12() {
    Fp6() x;
    Fp6() y;
}

template Fp12add() {
    input Fp12() op1;
    input Fp12() op2;
    output Fp12() out;

    (out.x, out.y) <== (Fp6add()(op1.x, op2.x), Fp6add()(op1.y, op2.y));
}

template Fp12mul() {
    input Fp12() op1;
    input Fp12() op2;
    output Fp12() out;

    Fp6() t0 <== Fp6mul()(op1.x, op2.x);
    Fp6() t1 <== Fp6mul()(op1.y, op2.y);

    Fp2() zero_2, one_2;
    (zero_2.x, zero_2.y) <== (0, 0);
    (one_2.x, one_2.y) <== (1, 0);
    Fp6() v;
    (v.x, v.y, v.z) <== (zero_2, one_2, zero_2);

    Fp6() t3 <== Fp6mul()(t1, v);
    out.x <== Fp6add()(t0, t3);
    Fp6() t4 <== Fp6mul()(op1.x, op2.y);
    Fp6() t5 <== Fp6mul()(op1.y, op2.x);
    out.y <== Fp6add()(t4, t5);
}

template Fp12mulbyfp() {
    signal input op1;
    input Fp12() op2;
    output Fp12() out;

    (out.x.x.x, out.x.y.x, out.x.z.x) <== (op1 * op2.x.x.x, op1 * op2.x.y.x, op1 * op2.x.z.x);
    (out.x.x.y, out.x.y.y, out.x.z.y) <== (op1 * op2.x.x.y, op1 * op2.x.y.y, op1 * op2.x.z.y);

    (out.y.x.x, out.y.y.x, out.y.z.x) <== (op1 * op2.y.x.x, op1 * op2.y.y.x, op1 * op2.y.z.x);
    (out.y.x.y, out.y.y.y, out.y.z.y) <== (op1 * op2.y.x.y, op1 * op2.y.y.y, op1 * op2.y.z.y);
}

template Fp12conjugate() {
    input Fp12() op1;
    output Fp12() out;

    out.x <== op1.x;
    out.y <== Fp6neg()(op1.y);
}

template Fp12square() {
    input Fp12() op1;
    output Fp12() out;

    Fp4() a0, a1, a2;
    (a0.x, a0.y) <== (op1.x.x, op1.y.y);
    (a1.x, a1.y) <== (op1.y.x, op1.x.z);
    (a2.x, a2.y) <== (op1.x.y, op1.y.z);

    Fp2() t00, t11, t12, t01, t02, t10, aux, shi;
    (shi.x, shi.y) <== (9, 1);

    (t00, t11) <== Fp4Square()(a0);
    (t01, t12) <== Fp4Square()(a1);         // Paper has typo here.
    (t02, aux) <== Fp4Square()(a2);

    t10 <== Fp2mul()(aux, shi);

    Fp2() m0 <== Fp2mulbyfp()(3, t00);
    Fp2() m1 <== Fp2mulbyfp()(-2, op1.x.x);
    Fp2() c00 <== Fp2add()(m0, m1);

    Fp2() m2 <== Fp2mulbyfp()(3, t01);
    Fp2() m3 <== Fp2mulbyfp()(-2, op1.x.y);
    Fp2() c01 <== Fp2add()(m2, m3);

    Fp2() m4 <== Fp2mulbyfp()(3, t02);
    Fp2() m5 <== Fp2mulbyfp()(-2, op1.x.z);
    Fp2() c02 <== Fp2add()(m4, m5);

    Fp2() m6 <== Fp2mulbyfp()(3, t10);
    Fp2() m7 <== Fp2mulbyfp()(2, op1.y.x);
    Fp2() c10 <== Fp2add()(m6, m7);

    Fp2() m8 <== Fp2mulbyfp()(3, t11);
    Fp2() m9 <== Fp2mulbyfp()(2, op1.y.y);
    Fp2() c11 <== Fp2add()(m8, m9);

    Fp2() m10 <== Fp2mulbyfp()(3, t12);
    Fp2() m11 <== Fp2mulbyfp()(2, op1.y.z);
    Fp2() c12 <== Fp2add()(m10, m11);

    Fp6() c0, c1;
    (c0.x, c0.y, c0.z) <== (c00, c01, c02);
    (c1.x, c1.y, c1.z) <== (c10, c11, c12);

    (out.x, out.y) <== (c0, c1);
    
}

// Fails
function inverse_12(
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11
) {
    // t0 = a0 * a0;
    var c4[2] = [2 * (x0 * x2 - x1 * x3), 2 * (x0 * x3 + x1 * x2)];
    var c5[2] = [x4 * x4 - x5 * x5, 2 * x4 * x5];
    var c1[2] = [9 * c5[0] - c5[1] + c4[0], c5[0] + 9 * c5[1] + c4[1]];
    var c2[2] = [c4[0] - c5[0], c4[1] - c5[1]];
    var c3[2] = [x0 * x0 - x1 * x1, 2 * x0 * x1];
    
    c4 = [x0 - x2 + x4, x1 - x3 + x5];
    var c40 = c4[0] * c4[0] - c4[1] * c4[1];
    var c41 = 2 * c4[0] * c4[1];
    c4 = [c40, c41];
    var c0[2] = [9 * c5[0] - c5[1] + c3[0], c5[0] + 9 * c5[1] + c3[1]];
    c2 = [c2[0] + c4[0] + c5[0] - c3[0], c2[1] + c4[1] + c5[1] - c3[1]];

    var t0[3][2] = [c0, c1, c2];

    // t1 = a1 * a1;
    c4 = [2 * (x6 * x8 - x7 * x9), 2 * (x6 * x9 + x7 * x8)];
    c5 = [x10 * x10 - x11 * x11, 2 * x10 * x11];
    c1 = [9 * c5[0] - c5[1] + c4[0], c5[0] + 9 * c5[1] + c4[1]];
    c2 = [c4[0] - c5[0], c4[1] - c5[1]];
    c3 = [x6 * x6 - x7 * x7, 2 * x6 * x7];
    
    c4 = [x6 - x8 + x10, x7 - x9 + x11];
    c40 = c4[0] * c4[0] - c4[1] * c4[1];
    c41 = 2 * c4[0] * c4[1];
    c4 = [c40, c41];
    c0 = [9 * c5[0] - c5[1] + c3[0], c5[0] + 9 * c5[1] + c3[1]];
    c2 = [c2[0] + c4[0] + c5[0] - c3[0], c2[1] + c4[1] + c5[1] - c3[1]];

    var t1[3][2] = [c0, c1, c2];

    //t0 = t0 - gamma * t1;
    var gamma_t1[3][2] = [[9 * t1[2][0] - t1[2][1], 9 * t1[2][1] + t1[2][0]], t1[0], t1[1]];
    t0 = [[t0[0][0] - gamma_t1[0][0], t0[0][1] - gamma_t1[0][1]],
            [t0[1][0] - gamma_t1[1][0], t0[1][1] - gamma_t1[1][1]],
            [t0[2][0] - gamma_t1[2][0], t0[2][1] - gamma_t1[2][1]]];

    // t1 = t1inv;
    var m0[2] = [t1[0][0] * t1[0][0] - t1[0][1] * t1[0][1], 2 * t1[0][0] * t1[0][1]];
    var m1[2] = [t1[1][0] * t1[1][0] - t1[1][1] * t1[1][1], 2 * t1[1][0] * t1[1][1]];
    var m2[2] = [t1[2][0] * t1[2][0] - t1[2][1] * t1[2][1], 2 * t1[2][0] * t1[2][1]];

    var m3[2] = [t1[0][0] * t1[1][0] - t1[0][1] * t1[1][1], t1[0][1] * t1[1][0] + t1[0][0] * t1[1][1]];
    var m4[2] = [t1[0][0] * t1[2][0] - t1[0][1] * t1[2][1], t1[0][1] * t1[2][0] + t1[0][0] * t1[2][1]];
    var m5[2] = [t1[1][0] * t1[2][0] - t1[1][1] * t1[2][1], t1[1][1] * t1[2][0] + t1[1][0] * t1[2][1]];

    c0 = [m0[0] - (9 * m5[0] - m5[1]), m0[1] - (9 * m5[1] + m5[0])];
    c1 = [9 * m2[0] - m2[1] - m3[0], 9 * m2[1] + m2[0] - m3[1]];
    c2 = [m1[0] * m4[0] - m1[1] * m4[1], m1[0] * m4[1] + m1[1] * m4[0]];

    var m6[2] = [t1[0][0] * c0[0] - t1[0][1] * c0[1], t1[0][1] * c0[0] + t1[0][0] * c0[1]];
    
    var a2c1[2] = [t1[2][0] * c1[0] - t1[2][1] * c1[1], t1[2][0] * c1[1] + t1[2][1] * c1[0]];
    var shi_a2c1[2] = [9 * a2c1[0] - a2c1[1], 9 * a2c1[1] + a2c1[0]];
    m6 = [m6[0] + shi_a2c1[0], m6[1] + shi_a2c1[1]];

    var a1c2[2] = [t1[1][0] * c2[0] - t1[1][1] * c2[1], t1[1][0] * c2[1] + t1[1][1] * c2[0]];
    var shi_a1c2[2] = [9 * a1c2[0] - a1c2[1], 9 * a1c2[1] + a1c2[0]];
    m6 = [m6[0] + shi_a1c2[0], m6[1] + shi_a1c2[1]];

    var denom = (m6[0] * m6[0] + m6[1] * m6[1]);
    m6 = [m6[0] / denom, -m6[1] / denom];

    var c00 = c0[0] * m6[0] - c0[1] * m6[1];
    var c01 = c0[0] * m6[1] + c0[1] * m6[0];
    c0 = [c00, c01];
    var c10 = c1[0] * m6[0] - c1[1] * m6[1];
    var c11 = c1[0] * m6[1] + c1[1] * m6[0];
    c1 = [c10, c11];
    var c20 = c2[0] * m6[0] - c2[1] * m6[1];
    var c21 = c2[0] * m6[1] + c2[1] * m6[0];
    c2 = [c20, c21];

    t1 = [c0, c1, c2];


    // c0 = a0 * t1;
    var u0[2] = [t1[0][0] * x0 - t1[0][1] * x1, t1[0][1] * x0 + t1[0][0] * x1];
    var u1[2] = [t1[1][0] * x2 - t1[1][1] * x3, t1[1][1] * x2 + t1[1][0] * x3];
    var u2[2] = [t1[2][0] * x4 - t1[0][1] * x5, t1[2][1] * x4 + t1[0][0] * x5];

    var a1_a2[2] = [t1[1][0] + t1[2][0], t1[1][1] + t1[2][1]];
    var b1_b2[2] = [x2 + x4, x3 + x5];

    var a0_a1[2] = [t1[0][0] + t1[1][0], t1[0][1] + t1[1][1]];
    var b0_b1[2] = [x0 + x2, x1 + x3];

    var a0_a2[2] = [t1[0][0] + t1[2][0], t1[0][1] + t1[2][1]];
    var b0_b2[2] = [x0 + x4, x1 + x5];

    var d0[2] = [a1_a2[0] * b1_b2[0] - a1_a2[1] * b1_b2[1] - u1[0] - u2[0],
                 a1_a2[0] * b1_b2[1] + a1_a2[1] * b1_b2[0] - u1[1] - u2[1]];
    var d1[2] = [a0_a1[0] * b0_b1[0] - a0_a1[1] * b0_b1[1] - u0[0] - u1[0], 
                a0_a1[0] * b0_b1[1] + a0_a1[1] * b0_b1[0] - u0[1] - u1[1]];
    var d2[2] = [a0_a2[0] * b0_b2[0] - a0_a2[1] * b0_b2[1] - u0[0] - u2[0] + u1[0],
                 a0_a2[0] * b0_b2[1] + a0_a2[1] * b0_b2[0] - u0[1] - u2[1] + u1[1]];

    var d00 = 9 * d0[0] - d0[1] + u0[0];
    var d01 = 9 * d0[1] + d0[0] + u0[1];
    d0 = [d00, d01];
    d1 = [d1[0] + 9 * u2[0] - u2[1], d1[1] + 9 * u2[1] + u2[0]];

    var e0[3][2] = [d0, d1, d2];

    // c1 = -a1 * t1;
    u0 = [t1[0][0] * x6 - t1[0][1] * x7, t1[0][1] * x6 + t1[0][0] * x7];
    u1 = [t1[1][0] * x8 - t1[1][1] * x9, t1[1][1] * x8 + t1[1][0] * x9];
    u2 = [t1[2][0] * x10 - t1[0][1] * x11, t1[2][1] * x10 + t1[0][0] * x11];

    a1_a2 = [t1[1][0] + t1[2][0], t1[1][1] + t1[2][1]];
    b1_b2 = [x8 + x10, x9 + x11];

    a0_a1 = [t1[0][0] + t1[1][0], t1[0][1] + t1[1][1]];
    b0_b1 = [x6 + x8, x7 + x9];

    a0_a2 = [t1[0][0] + t1[2][0], t1[0][1] + t1[2][1]];
    b0_b2 = [x6 + x10, x7 + x11];

    d0 = [a1_a2[0] * b1_b2[0] - a1_a2[1] * b1_b2[1] - u1[0] - u2[0],
                 a1_a2[0] * b1_b2[1] + a1_a2[1] * b1_b2[0] - u1[1] - u2[1]];
    d1 = [a0_a1[0] * b0_b1[0] - a0_a1[1] * b0_b1[1] - u0[0] - u1[0], 
                a0_a1[0] * b0_b1[1] + a0_a1[1] * b0_b1[0] - u0[1] - u1[1]];
    d2 = [-(a0_a2[0] * b0_b2[0] - a0_a2[1] * b0_b2[1] - u0[0] - u2[0] + u1[0]),
                 -(a0_a2[0] * b0_b2[1] + a0_a2[1] * b0_b2[0] - u0[1] - u2[1] + u1[1])];

    d00 = -(9 * d0[0] - d0[1] + u0[0]);
    d01 = -(9 * d0[1] + d0[0] + u0[1]);
    d0 = [d00, d01];
    d1 = [-(d1[0] + 9 * u2[0] - u2[1]), -(d1[1] + 9 * u2[1] + u2[0])];

    var e1[3][2] = [d0, d1, d2];

    return [e0, e1];
}

template Fp12inv() {
    input Fp12() op1;
    output Fp12() out;

    Fp2() zero_2, one_2;
    (zero_2.x, zero_2.y) <== (0, 0);
    (one_2.x, one_2.y) <== (1, 0);

    Fp6() zero_6, one_6, v;
    (zero_6.x, zero_6.y, zero_6.z) <== (zero_2, zero_2, zero_2);
    (one_6.x, one_6.y, one_6.z) <== (one_2, zero_2, zero_2);
    (v.x, v.y, v.z) <== (zero_2, one_2, zero_2);

    Fp12() one_12;
    (one_12.x, one_12.y) <== (one_6, zero_6);

    Fp6() t0 <== Fp6mul ()(op1.x, op1.x);
    Fp6() t1 <== Fp6mul ()(op1.y, op1.y);
    Fp6() t2 <== Fp6mul ()(v, t1);
    Fp6() t3 <== Fp6sub ()(t0, t2);
    Fp6() t4 <== Fp6inv ()(t3);
    Fp6() c0 <== Fp6mul ()(op1.x, t4);
    Fp6() c1 <== Fp6mul ()(op1.y, t4);

    out.x <== c0;
    out.y <== Fp6sub ()(zero_6, c1);
}

// Currently not in use
// Fails 
template Fp12inv_better() {
    input Fp12() op1;
    output Fp12() out;

    Fp2() zero_2, one_2;
    (zero_2.x, zero_2.y) <== (0, 0);
    (one_2.x, one_2.y) <== (1, 0);

    Fp6() zero_6, one_6;
    (zero_6.x, zero_6.y, zero_6.z) <== (zero_2, zero_2, zero_2);
    (one_6.x, one_6.y, one_6.z) <== (one_2, zero_2, zero_2);

    Fp12() one_12;
    (one_12.x, one_12.y) <== (one_6, zero_6);

    var res[2][3][2] = inverse_12(
        op1.x.x.x,
        op1.x.x.y,
        op1.x.y.x,
        op1.x.y.y,
        op1.x.z.x,
        op1.x.z.y,
        op1.y.x.x,
        op1.y.x.y,
        op1.y.y.x,
        op1.y.y.y,
        op1.y.z.x,
        op1.y.z.y
    );    

    (out.x.x.x, out.x.x.y) <-- (res[0][0][0], res[0][0][1]);
    (out.x.y.x, out.x.y.y) <-- (res[0][1][0], res[0][1][1]);
    (out.x.z.x, out.x.z.y) <-- (res[0][2][0], res[0][2][1]);
    
    (out.y.x.x, out.y.x.y) <-- (res[1][0][0], res[1][0][1]);
    (out.y.y.x, out.y.y.y) <-- (res[1][1][0], res[1][1][1]);
    (out.y.z.x, out.y.z.y) <-- (res[1][2][0], res[1][2][1]);

    Fp12() mult <== Fp12mul()(op1, out);
    mult === one_12;
}

// Currently not in use
template Fp12sq() {
    input Fp12() op1;
    output Fp12() out;

    Fp6() a0sq <== Fp6mul()(op1.x, op1.x);
    Fp6() a1sq <== Fp6mul()(op1.y, op1.y);
    Fp6() a0a1 <== Fp6mul()(op1.x, op1.y);

    Fp2() zero_2;
    (zero_2.x, zero_2.y) <== (0, 0);
    Fp2() one_2;
    (one_2.x, one_2.y) <== (1, 0);
    Fp6() v;
    (v.x, v.y, v.z) <== (zero_2, one_2, zero_2);

    Fp6() v_a1sq <== Fp6mul()(a1sq, v);
    out.x <== Fp6add()(a0sq, v_a1sq);
    out.y <== Fp6add()(a0a1, a0a1);
}

template Fp12exp() {
    signal input op1;
    input Fp12() op2;
    output Fp12() out;

    Fp2() zero_2;
    (zero_2.x, zero_2.y) <== (0, 0);
    Fp2() one_2;
    (one_2.x, one_2.y) <== (1, 0);
    Fp6() zero_6;
    (zero_6.x, zero_6.y, zero_6.z) <== (zero_2, zero_2, zero_2);
    Fp6() one_6;
    (one_6.x, one_6.y, one_6.z) <== (one_2, zero_2, zero_2);
    
    var n = 256;
    Fp12() f[n+1][3];
    Fp12() condProduct[n][2];
    (f[0][0].x, f[0][0].y) <== (one_6, zero_6);

    signal bits[n] <== Num2Bits(n)(op1);

    for(var i = 0; i < n; i++) {
        // Square
        // f[i][1] <== Fp12sq()(f[i][0]);
        f[i][1] <== Fp12square()(f[i][0]);

        // Multiply
        f[i][2] <== Fp12mul()(f[i][1], op2);

        // Conditionally select
        condProduct[i][0] <== Fp12mulbyfp()(1 - bits[n-1-i], f[i][1]);
        condProduct[i][1] <== Fp12mulbyfp()(bits[n-1-i], f[i][2]);

        // Add conditional products
        f[i+1][0] <== Fp12add()(condProduct[i][0], condProduct[i][1]);
    }

    out <== f[n][0];
}
// // component main = Fp12exp();
