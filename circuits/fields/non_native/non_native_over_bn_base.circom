pragma circom 2.2.1;
// Change the path whenever use the common utils and utiltiy_buses
include "./utils.circom";
include "./../bigints/bigint.circom";

// We have different getScalarModulus(), so there are two files for non native operations

// The most significant limbs of a and b can have at most 100 bits.
// Tested. 391 constraints.
template NonNativeAdd(){
    input Fq() op1, op2;
    output Fq() out;

    component bigAdd = BigAddNoCarry (125, 3);
    for (var i = 0; i < 3; i++) {
        bigAdd.a[i] <== op1.limbs[i];
        bigAdd.b[i] <== op2.limbs[i];
    }
    for (var i = 0; i < 3; i++) {
        out.limbs[i] <== bigAdd.out[i];
    }
}

template NonNativeAdd6Limbs(){
    input signal op1[6], op2[6];
    output signal out[6];

    component bigAdd = BigAddNoCarry (125, 6);
    for (var i = 0; i < 6; i++) {
        bigAdd.a[i] <== op1[i];
        bigAdd.b[i] <== op2[i];
    }
    for (var i = 0; i < 6; i++) {
        out[i] <== bigAdd.out[i];
    }
}

template NonNativeModulo6Limbs(){
    input signal op[6];
    output Fq() out;

    Fq() q;
    var scalar_modulus[4] = getScalarModulus();
    q.limbs[0] <== scalar_modulus[1];
    q.limbs[1] <== scalar_modulus[2];
    q.limbs[2] <== scalar_modulus[3];

    component bigMod = BigMod (125, 3);
    for (var i = 0; i < 3; i++) {
        bigMod.a[i] <== op[i];
        bigMod.a[i + 3] <== op[i + 3];
        bigMod.b[i] <== q.limbs[i];
    }
    for (var i = 0; i < 3; i++) {
        out.limbs[i] <== bigMod.mod[i];
    }
}

// Tested. 791 constraints.
template NonNativeSub() {
    input Fq() op1, op2;
    output Fq() out;

    Fq() q;
    var scalar_modulus[4] = getScalarModulus();
    q.limbs[0] <== scalar_modulus[1];
    q.limbs[1] <== scalar_modulus[2];
    q.limbs[2] <== scalar_modulus[3];

    component bigSub = BigSubModP (125, 3);
    for (var i = 0; i < 3; i++) {
        bigSub.a[i] <== op1.limbs[i];
        bigSub.b[i] <== op2.limbs[i];
        bigSub.p[i] <== q.limbs[i];
    }
    for (var i = 0; i < 3; i++) {
        out.limbs[i] <== bigSub.out[i];
    }
}

// Tested. 391 constraints.
template NonNativeAdditiveInverse() {
    input Fq() op;
    output Fq() out;

    Fq() zero;
    zero.limbs[0] <== 0;
    zero.limbs[1] <== 0;
    zero.limbs[2] <== 0;

    Fq() q;
    var scalar_modulus[4] = getScalarModulus();
    q.limbs[0] <== scalar_modulus[1];
    q.limbs[1] <== scalar_modulus[2];
    q.limbs[2] <== scalar_modulus[3];

    component bigSub = BigSubModP (125, 3);
    for (var i = 0; i < 3; i++) {
        bigSub.a[i] <== zero.limbs[i];
        bigSub.b[i] <== op.limbs[i];
        bigSub.p[i] <== q.limbs[i];
    }
    for (var i = 0; i < 3; i++) {
        out.limbs[i] <== bigSub.out[i];
    }
}

// Tested. 5666 constraints.
template NonNativeMul(){
    input Fq() op1, op2;
    output Fq() out;

    Fq() q;
    var scalar_modulus[4] = getScalarModulus();
    q.limbs[0] <== scalar_modulus[1];
    q.limbs[1] <== scalar_modulus[2];
    q.limbs[2] <== scalar_modulus[3];

    component bigMul = BigMultModP (125, 3);
    for (var i = 0; i < 3; i++) {
        bigMul.a[i] <== op1.limbs[i];
        bigMul.b[i] <== op2.limbs[i];
        bigMul.p[i] <== q.limbs[i];
    }
    for (var i = 0; i < 3; i++) {
        out.limbs[i] <== bigMul.out[i];
    }
}

template NonNativeMulOld(){
    input Fq() op1, op2;
    
    output Fq() out;

    signal preFold[3], limbMul[5], low[6], hi[4], temp[4], gamma[3][3];
    signal gammaLo[3][3], gammaHi[3][2];

    //Compute (a0, a1, a2) * (b0, b1, b2)
    limbMul[0] <== op1.limbs[0] * op2.limbs[0];
    temp[0] <== op1.limbs[0] * op2.limbs[1];
    limbMul[1] <== temp[0] + (op1.limbs[1] * op2.limbs[0]);
    temp[1] <== op1.limbs[0] * op2.limbs[2];
    temp[2] <== temp[1] + (op1.limbs[1] * op2.limbs[1]);
    limbMul[2] <==  temp[2]  + (op1.limbs[2] * op2.limbs[0]);
    temp[3] <== op1.limbs[1] * op2.limbs[2];
    limbMul[3] <== temp[3] + (op1.limbs[2] * op2.limbs[1]);
    limbMul[4] <== op1.limbs[2] * op2.limbs[2];


    (low[0], hi[0]) <== Num2LoHi(125, 125)(limbMul[0]);
    (low[1], hi[1]) <== Num2LoHi(125, 126)(limbMul[1] + hi[0]);
    (low[2], hi[2]) <== Num2LoHi(125, 127)(limbMul[2] + hi[1]);
    (low[3], hi[3]) <== Num2LoHi(125, 126)(limbMul[3] + hi[2]);
    (low[4], low[5]) <== Num2LoHi(125, 125)(limbMul[4] + hi[3]);


    for (var j = 0; j < 3; j++)
    {
        var twoTimesJ = 2 * j;
        var k[3] = TwoPow125Mod(j + 3);
        gamma[j][0] <== k[0] * low[j + 3];
        gamma[j][1] <== k[1] * low[j + 3];
        gamma[j][2] <== k[2] * low[j + 3];

        (gammaLo[j][0], gammaHi[j][0]) <== Num2LoHi(125, 125)(gamma[j][0]);
        (gammaLo[j][1], gammaHi[j][1]) <== Num2LoHi(125, 125)(gammaHi[j][0] + gamma[j][1]);
        gammaLo[j][2] <== gammaHi[j][1] + gamma[j][2];
    }

    for (var i = 0; i < 3; i++)
    {
        preFold[i] <== low[i] + gammaLo[0][i] + gammaLo[1][i] + gammaLo[2][i];
    }


    signal postFold[3] <== FinalFold(133)(preFold);

    for (var i = 0; i < 3; i++)
    {
        out.limbs[i] <== postFold[i];
    }
}

template FinalFold(nBits){
    assert(nBits > 50);
    signal input a[3];
    signal output fold[3];
    signal t0, t1, intA[2], loIntA[3], hiIntA[2];
    
    var b[3] = TwoPow300Limb();

    (t0, t1) <== Num2LoHi(50, nBits - 50)(a[2]);

    intA[0] <== a[0] + t1 * b[0];
    intA[1] <== a[1] + t1 * b[1];

    (loIntA[0], hiIntA[0]) <== Num2LoHi(125, nBits - 50)(intA[0]);
    signal IntAPlusHiIntA <== intA[1] + hiIntA[0];

    (loIntA[1], hiIntA[1]) <== Num2LoHi(125, nBits - 50)(IntAPlusHiIntA);
    loIntA[2] <== t0 + hiIntA[1];
    
    for (var i = 0; i < 3; i++)
    {
        fold[i] <== loIntA[i];
    }
}

function TwoPow300Limb(){
    return [19519013210766711215589865100324740550, 9357004037415048792450643750180786195, 0];
}

function TwoPow125Mod(select){
    if (select == 3) {
        return [38553441778134509641527191661516408448, 30986793088188334059812209232102964857, 7];
    }
    else if (select == 4) {
        return [25978960821266683100703934416343392791, 32975171814276302499345656289751347705, 1];
    }
    else {
        return [6192014213090135202558608763185540141, 7774807645944629514010873931778462840, 11];
    } 
}





template NonNativeMulWithoutReduction(){
    input Fq() op1, op2;
    output signal out[6];

    component bigMul = BigMult (125, 3);
    for (var i = 0; i < 3; i++) {
        bigMul.a[i] <== op1.limbs[i];
        bigMul.b[i] <== op2.limbs[i];
    }
    for (var i = 0; i < 6; i++) {
        out[i] <== bigMul.out[i];
    }
}

template Num2Limbs(l1, l2, l3) {
    signal input in;
    signal output limbs[3];

    signal l1_bits[l1];
    signal l2_bits[l2];
    signal l3_bits[l3];
    signal int_l1[l1 + 1];
    signal int_l2[l2 + 1];
    signal int_l3[l3 + 1];

    int_l1[0] <== 0;
    var pow2_0 = 1;

    for (var i = 0; i < l1; i++) {
        l1_bits[i] <-- (in >> i) & 1;
        l1_bits[i] * (l1_bits[i] - 1) === 0;
        int_l1[i + 1] <== int_l1[i] + l1_bits[i] * pow2_0;
        pow2_0 = pow2_0 + pow2_0;
    }
    
    int_l2[0] <== 0;
    var pow2_1 = 1;

    for (var i = 0; i < l2; i++) {
        l2_bits[i] <-- (in >> (i + l1)) & 1;
        l2_bits[i] * (l2_bits[i] - 1) === 0;
        int_l2[i + 1] <== int_l2[i] + l2_bits[i] * pow2_1;
        pow2_1 = pow2_1 + pow2_1;
    }

    var pow2_2 = 1;
    int_l3[0] <== 0;

     for (var i = 0; i < l3; i++) {
        l3_bits[i] <-- (in >> (i + l1 + l2)) & 1;
        l3_bits[i] * (l3_bits[i] - 1) === 0;
        int_l3[i + 1] <== int_l3[i] + l3_bits[i] * pow2_2;
        pow2_2 = pow2_2 + pow2_2;
    }

    limbs[0] <== int_l1[l1];
    limbs[1] <== int_l2[l2];
    limbs[2] <== int_l3[l3];
    limbs[0] + pow2_0 * limbs[1] + pow2_0 *  pow2_1 * limbs[2] === in;
    
}

template NonNativeModuloFp(){
    signal input op;
    output Fq() out;

    signal limbs[3] <== Num2Limbs(125, 125, 4)(op);
    Fq() op1;
    op1.limbs <== limbs;
    out <== NonNativeModulo()(op1);

}
// Tested. 4255 constraints.
template NonNativeModulo() {
    input Fq() op;
    output Fq() out;

    Fq() q;
    var scalar_modulus[4] = getScalarModulus();
    q.limbs[0] <== scalar_modulus[1];
    q.limbs[1] <== scalar_modulus[2];
    q.limbs[2] <== scalar_modulus[3];

    component bigMod = BigMod (125, 3);
    for (var i = 0; i < 3; i++) {
        bigMod.a[i] <== op.limbs[i];
        bigMod.a[i + 3] <== 0;
        bigMod.b[i] <== q.limbs[i];
    }
    for (var i = 0; i < 3; i++) {
        out.limbs[i] <== bigMod.mod[i];
    }
}

// Tested. 8510 constraints. 
template NonNativeEquality() {
    input Fq() op1, op2;

    Fq() reduced_op1, reduced_op2;
    Fq() q;
    var scalar_modulus[4] = getScalarModulus();
    q.limbs[0] <== scalar_modulus[1];
    q.limbs[1] <== scalar_modulus[2];
    q.limbs[2] <== scalar_modulus[3];

    component bigMod1 = BigMod (125, 3);
    for (var i = 0; i < 3; i++) {
        bigMod1.a[i] <== op1.limbs[i];
        bigMod1.a[i + 3] <== 0;
        bigMod1.b[i] <== q.limbs[i];
    }
    for (var i = 0; i < 3; i++) {
        reduced_op1.limbs[i] <== bigMod1.mod[i];
    }

    component bigMod2 = BigMod (125, 3);
    for (var i = 0; i < 3; i++) {
        bigMod2.a[i] <== op2.limbs[i];
        bigMod2.a[i + 3] <== 0;
        bigMod2.b[i] <== q.limbs[i];
    }
    for (var i = 0; i < 3; i++) {
        reduced_op2.limbs[i] <== bigMod2.mod[i];
    }

    for (var i = 0; i < 3; i++) {
        reduced_op1.limbs[i] === reduced_op2.limbs[i];
    }
}

template NonNativeEqualityBothReduced() {
    input Fq() op1, op2;

    for (var i = 0; i < 3; i++) {
        op1.limbs[i] === op2.limbs[i];
    }
}

// Tested. 4255 constraints. 
template NonNativeEqualityReducedRHS() {
    input Fq() op1, op2;

    Fq() reduced_op1, reduced_op2;
    Fq() q;
    var scalar_modulus[4] = getScalarModulus();
    q.limbs[0] <== scalar_modulus[1];
    q.limbs[1] <== scalar_modulus[2];
    q.limbs[2] <== scalar_modulus[3];

    component bigMod1 = BigMod (125, 3);
    for (var i = 0; i < 3; i++) {
        bigMod1.a[i] <== op1.limbs[i];
        bigMod1.a[i + 3] <== 0;
        bigMod1.b[i] <== q.limbs[i];
    }
    for (var i = 0; i < 3; i++) {
        reduced_op1.limbs[i] <== bigMod1.mod[i];
    }

    for (var i = 0; i < 3; i++) {
        reduced_op1.limbs[i] === op2.limbs[i];
    }
}

// Tested. 4255 constraints.
template NonNativeEqualityReducedLHS() {
    input Fq() op1, op2;

    Fq() reduced_op2;
    Fq() q;
    var scalar_modulus[4] = getScalarModulus();
    // q <== scalar_modulus[0];
    q.limbs[0] <== scalar_modulus[1];
    q.limbs[1] <== scalar_modulus[2];
    q.limbs[2] <== scalar_modulus[3];

    component bigMod2 = BigMod (125, 3);
    for (var i = 0; i < 3; i++) {
        bigMod2.a[i] <== op2.limbs[i];
        bigMod2.a[i + 3] <== 0;
        bigMod2.b[i] <== q.limbs[i];
    }
    for (var i = 0; i < 3; i++) {
        reduced_op2.limbs[i] <== bigMod2.mod[i];
    }

    for (var i = 0; i < 3; i++) {
        op1.limbs[i] === reduced_op2.limbs[i];
    }
}

// BN Scalar modulus
function getScalarModulus() {
    var val = 21888242871839275222246405745257275088548364400416034343698204186575808495617;
    var low = 10903342367192220456583066779700428801;
    var mid = 4166566524057721139834548734155997929;
    var hi = 12;

    return [val, low, mid, hi];
}
 
// component main = NonNativeModulo6Limbs();