pragma circom 2.2.1;
include "./../../node_modules/circomlib/circuits/bitify.circom";

bus Fp2() {
    signal x;
    signal y;
}

template Fp2add() {
    input Fp2() op1;
    input Fp2() op2;
    output Fp2() out;

    out.x <== op1.x + op2.x;
    out.y <== op1.y + op2.y;
}

template Fp2sub() {
    input Fp2() op1;
    input Fp2() op2;
    output Fp2() out;

    out.x <== op1.x - op2.x;
    out.y <== op1.y - op2.y;
}

template Fp2mul() {
    input Fp2() op1;
    input Fp2() op2;
    output Fp2() out;

    signal a1b1 <== op1.y * op2.y; 
    signal a1b0 <== op1.y * op2.x;

    out.x <== op1.x * op2.x - a1b1;
    out.y <== op1.x * op2.y + a1b0;
}

template Fp2muladd() {
    input Fp2() op1;
    input Fp2() op2;
    input Fp2() op3;
    output Fp2() out;

    signal a1b1 <== op1.y * op2.y;
    signal a1b0 <== op1.y * op2.x;

    out.x <== op1.x * op2.x - a1b1 + op3.x;
    out.y <== op1.x * op2.y + a1b0 + op3.y;   
}

template Fp2mulbyfp() {
    signal input op1;
    input Fp2() op2;
    output Fp2() out;

    out.x <== op2.x * op1;
    out.y <== op2.y * op1;
}

function inverse_2(x, y) {
    var denom = x*x + y*y;
    var p = x/denom;
    var q = -y/denom;
    return [p, q];
}

template Fp2inv() {
    input Fp2() op1;
    output Fp2() out;

    Fp2() one_2;
    (one_2.x, one_2.y) <== (1, 0);

    var res[2] = inverse_2(op1.x, op1.y);
    (out.x, out.y) <-- (res[0], res[1]);

    Fp2() mult <== Fp2mul()(op1, out);
    mult === one_2;
}

template Fp2conjugate() {
    input Fp2() op1;
    output Fp2() out;

    (out.x, out.y) <== (op1.x, -op1.y);
}

template Fp2exp() {
    signal input op1;
    input Fp2() op2;
    output Fp2() out;

    var n = 256;
    //TODO(Anuj):- Fix this op2 is of the type Fq() but Num2Bits takes signal 
    signal bits[n] <== Num2Bits(n)(op1);

    Fp2() results[n+1][3];
    Fp2() condProduct[n][2];
    (results[0][0].x, results[0][0].y) <== (1, 0);

    for(var i = 0; i < n; i++) {
        // Square
        results[i][1] <== Fp2mul()(results[i][0], results[i][0]);

        // Multiply
        results[i][2] <== Fp2mul()(op2, results[i][1]);

        // Conditionally select
        condProduct[i][0] <== Fp2mulbyfp()(1 - bits[n-i-1], results[i][1]);
        condProduct[i][1] <== Fp2mulbyfp()(bits[n-i-1], results[i][2]);

        // Add conditional products
        results[i+1][0] <== Fp2add()(condProduct[i][0], condProduct[i][1]);
    }

    out <== results[n][0];
}

// component main = Fp2add();
// component main = Fp2inv();

// component main = Fp2exp();

// component main = Fp2mul();

// component main = Fp2mulbyfp();
