pragma circom 2.2.1;

//Scalar field element represented by three limbs
bus Fq{
    signal limbs[3];
}

template Num2LoHi(num_lo_bits, num_hi_bits) {
    signal input in;
    signal output lo;
    signal output hi;

    signal lo_bits[num_lo_bits];
    signal hi_bits[num_hi_bits];
    signal int_lo[num_lo_bits + 1];
    signal int_hi[num_hi_bits + 1];

    int_lo[0] <== 0;
    var pow2_0 = 1;

    for (var i = 0; i < num_lo_bits; i++) {
        lo_bits[i] <-- (in >> i) & 1;
        lo_bits[i] * (lo_bits[i] - 1) === 0;
        int_lo[i + 1] <== int_lo[i] + lo_bits[i] * pow2_0;
        pow2_0 = pow2_0 + pow2_0;
    }

    int_hi[0] <== 0;
    var pow2_1 = 1;

    for (var i = 0; i < num_hi_bits; i++) {
        hi_bits[i] <-- (in >> (i + num_lo_bits)) & 1;
        hi_bits[i] * (hi_bits[i] - 1) === 0;
        int_hi[i + 1] <== int_hi[i] + hi_bits[i] * pow2_1;
        pow2_1 = pow2_1 + pow2_1;
    }
    lo <== int_lo[num_lo_bits];
    hi <== int_hi[num_hi_bits];
    
    lo + pow2_0 * hi === in;
}

function log2(n) {
    var result = 0;
    while (n > 1) {
        n = n >> 1; 
        result++;
    }
    return result;
}

function NextPowerOf2(n) {
    var power = 1;
    while (power < n) {
        power = power << 1;
    }
    return power;
}

function CeilLog2(n) {
    var r = NextPowerOf2(n);
    var result = log2(r);

      return result;
}
