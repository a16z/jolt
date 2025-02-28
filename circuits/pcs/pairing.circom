pragma circom 2.2.1;
include "./../groups/bn254_g2.circom";
include "./../groups/bn254_g1.circom";
include "./../fields/field/fp12.circom";


template Fpinv(){
    signal input in;
    signal output inv;

    var inverse = 1/in;
    inv <-- inverse;
    inv * in === 1;

}

template G2ToAffine(){
    input G2Projective() op1;
    output G2Affine() out;
    Fp2() op1_z_inv <== Fp2inv()(op1.z);
    out.x <== Fp2mul()(op1.x, op1_z_inv);
    out.y <== Fp2mul()(op1.y, op1_z_inv);
}

template FinalExp() {
    Fp12() input f;
    Fp12() output out;
    
    // Easy part from High-Speed Software Implementation of the Optimal Ate Pairing over Barreto-Naehrig Curves.

    Fp12() f1 <== Fp12conjugate()(f);
    Fp12() f2 <== Fp12inv()(f);
    Fp12() f3 <== Fp12mul()(f1, f2);

    Fp12() f4 <== frobenius2()(f3);
    Fp12() r <== Fp12mul()(f4, f3);

    // Hard part replicating arkwork's implementation
    Fp12() y0 <== ExpByNegX()(r);
    Fp12() y1 <== Fp12square()(y0);
    Fp12() y2 <== Fp12square()(y1);
    Fp12() y3 <== Fp12mul()(y2, y1);
    Fp12() y4 <== ExpByNegX()(y3);
    Fp12() y5 <== Fp12square()(y4);
    Fp12() y6 <== ExpByNegX()(y5);
    Fp12() y3_cyclo_inv <== Fp12conjugate()(y3);  // replace y3 by y3_cyclo_inv
    Fp12() y6_cyclo_inv <== Fp12conjugate()(y6);  // replace y6 by y6_cyclo_inv
    Fp12() y7 <== Fp12mul()(y6_cyclo_inv, y4);
    Fp12() y8 <== Fp12mul()(y7, y3_cyclo_inv);
    Fp12() y9 <== Fp12mul()(y8, y1);
    Fp12() y10 <== Fp12mul()(y8, y4);
    Fp12() y11 <== Fp12mul()(y10, r);
    Fp12() y12 <== frobenius1()(y9);   // check i Frobenius correct
    Fp12() y13 <== Fp12mul()(y12, y11);
    Fp12() y8_frobenius <== frobenius2()(y8);  // replace y8 by y8_frobenius
    Fp12() y14 <== Fp12mul()(y8_frobenius, y13);
    Fp12() r_cyclo_inv <== Fp12conjugate()(r); // replace r by r_cyclo_inv
    Fp12() y15 <== Fp12mul()(r_cyclo_inv, y9);
    Fp12() y15_frobenius <== frobenius3()(y15); // replace y15 by y15_frobenius
    out <== Fp12mul()(y15_frobenius, y14);
}

template ExpByNegX() {
    input Fp12() f;
    output Fp12() out;

    signal X <== 4965661367192848881;
    Fp12() t <== Fp12exp()(X, f);               // f = f.cyclotomic_exp(P::X);
    out <== Fp12conjugate()(t);                 // f.cyclotomic_inverse_in_place();
}

template gamma1() {
    Fp2() output gamma1[6];

    Fp2() shi;
    (shi.x, shi.y) <== (9, 1);
    signal constExp <== 3648040478639879203707734290876212514782718526216303943781506315774204368097;

    Fp2() shi_c <== Fp2exp()(constExp, shi);
    (gamma1[0].x, gamma1[0].y) <== (1, 0);

    for(var i = 1; i <= 5; i++) {
        gamma1[i] <== Fp2mul()(gamma1[i - 1], shi_c);
    }
}

template gamma2() {
    Fp2() input gamma1[6];
    Fp2() output gamma2[6];

    Fp2() gamma1Conjugate[6];
    for(var i = 1; i <= 5; i++) {
        gamma1Conjugate[i] <== Fp2conjugate()(gamma1[i]);
        gamma2[i] <== Fp2mul()(gamma1[i], gamma1Conjugate[i]);
    }
}

template gamma3() {
    Fp2() input gamma1[6];
    Fp2() input gamma2[6];
    Fp2() output gamma3[6];

    for(var i = 1; i <= 5; i++) {
        gamma3[i] <== Fp2mul()(gamma1[i], gamma2[i]);
    }
}

template frobenius1() {
    Fp12() input op1;
    Fp12() output out;

    Fp2() t[7], u[7];
    t[1] <== Fp2conjugate()(op1.x.x);
    t[2] <== Fp2conjugate()(op1.y.x);
    t[3] <== Fp2conjugate()(op1.x.y);
    t[4] <== Fp2conjugate()(op1.y.y);
    t[5] <== Fp2conjugate()(op1.x.z);
    t[6] <== Fp2conjugate()(op1.y.z);

    Fp2() gamma1[6] <== gamma1()();

    for(var i = 2; i <= 6; i++) {
        u[i] <== Fp2mul()(t[i], gamma1[i - 1]);
    }

    Fp6() c0, c1;
    (c0.x, c0.y, c0.z) <== (t[1], u[3], u[5]);
    (c1.x, c1.y, c1.z) <== (u[2], u[4], u[6]);

    (out.x, out.y) <== (c0, c1);
}

template frobenius2() {
    Fp12() input op1;
    Fp12() output out;

    Fp2() gamma1[6] <== gamma1()();
    Fp2() gamma2[6] <== gamma2()(gamma1);

    Fp2() u2 <== Fp2mul()(op1.y.x, gamma2[1]);
    Fp2() u3 <== Fp2mul()(op1.x.y, gamma2[2]);
    Fp2() u4 <== Fp2mul()(op1.y.y, gamma2[3]);
    Fp2() u5 <== Fp2mul()(op1.x.z, gamma2[4]);
    Fp2() u6 <== Fp2mul()(op1.y.z, gamma2[5]);

    Fp6() c0, c1;
    (c0.x, c0.y, c0.z) <== (op1.x.x, u3, u5);
    (c1.x, c1.y, c1.z) <== (u2, u4, u6);

    (out.x, out.y) <== (c0, c1);
}

template frobenius3() {
    Fp12() input op1;
    Fp12() output out;

    Fp2() t[7], u[7];
    t[1] <== Fp2conjugate()(op1.x.x);
    t[2] <== Fp2conjugate()(op1.y.x);
    t[3] <== Fp2conjugate()(op1.x.y);
    t[4] <== Fp2conjugate()(op1.y.y);
    t[5] <== Fp2conjugate()(op1.x.z);
    t[6] <== Fp2conjugate()(op1.y.z);

    Fp2() gamma1[6] <== gamma1()();
    Fp2() gamma2[6] <== gamma2()(gamma1);
    Fp2() gamma3[6] <== gamma3()(gamma1, gamma2);
    
    for(var i = 2; i <= 6; i++) {
        u[i] <== Fp2mul()(t[i], gamma3[i - 1]);
    }

    Fp6() c0, c1;
    (c0.x, c0.y, c0.z) <== (t[1], u[3], u[5]);
    (c1.x, c1.y, c1.z) <== (u[2], u[4], u[6]);

    (out.x, out.y) <== (c0, c1);
}



template MillerLoop() {
    input G2Affine() Q;
    input G1Projective() P;
    output Fp12() out;

    var n = 64;

    var bits[n] = [
        0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0,
        0, -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0, -1, 0,
        -1, 0, 0, 0, 1, 0, 1];

    G1Affine() p <== G1ToAffine()(P);

    Fp6() ell_coeff[2 * n + 2] <== EllCoeffs()(Q);

    Fp2() zero_2, one_2;
    (zero_2.x, zero_2.y) <== (0, 0);
    (one_2.x, one_2.y) <== (1, 0);

    Fp6() zero_6, one_6;
    (zero_6.x, zero_6.y, zero_6.z) <== (zero_2, zero_2, zero_2);
    (one_6.x, one_6.y, one_6.z) <== (one_2, zero_2, zero_2);
    
    Fp12() f[3 * n + 3];
    (f[0].x, f[0].y) <== (one_6, zero_6);

    for(var i = 0; i < n; i++) {
        f[3 * i + 1] <== Fp12mul()(f[3 * i], f[3 * i]);
        f[3 * i + 2] <== Ell()(f[3 * i + 1], ell_coeff[2 * i], p);

        if(bits[n - i - 1] == 1) {
            f[3 * i + 3] <== Ell()(f[3 * i + 2], ell_coeff[2 * i + 1], p);
        } else if (bits[n - i - 1] == -1) {
            f[3 * i + 3] <== Ell()(f[3 * i + 2], ell_coeff[2 * i + 1], p);
        }
        else {
            f[3 * i + 3] <== f[3 * i + 2];
        }
    }
    
    f[3 * n + 1] <== Ell()(f[3 * n], ell_coeff[2 * n], p);
    f[3 * n + 2] <== Ell()(f[3 * n + 1], ell_coeff[2 * n + 1], p);
    
    out <== f[3 * n + 2];
}

template Ell() {
    input Fp12() f;
    input Fp6() coeff;
    input G1Affine() p;
    output Fp12() updated_f;

    // TwistTpye::D
    Fp2() c0 <== Fp2mulbyfp()(p.y, coeff.x);
    Fp2() c1 <== Fp2mulbyfp()(p.x, coeff.y);
    updated_f <== MulBy034()(f, c0, c1, coeff.z);
}

template MulBy034() {
    input Fp12() f;                        
    input Fp2() c0, c3, c4;
    output Fp12() updated_f;

    Fp6() a;
    a.x <== Fp2mul()(f.x.x, c0);
    a.y <== Fp2mul()(f.x.y, c0);
    a.z <== Fp2mul()(f.x.z, c0);           

    Fp6() b <== MulBy01()(f.y, c3, c4);     

    Fp2() c0_new <== Fp2add()(c0, c3);    
    // c4 = c1;
    Fp6() temp_e <== Fp6add()(f.x, f.y);    
    Fp6() e <== MulBy01()(temp_e, c0_new, c4);

    updated_f.y <== Fp6sub()(e, Fp6add()(a, b)); 

    Fp6() temp <== MulFp6ByNonresidue()(b); 
   
    updated_f.x <== Fp6add()(temp, a); 
}

template MulBy01() {
    // Inputs
    input Fp6()  op1;    
    input Fp2() c0, c1;  // The Fp2 coefficients to multiply by

    // Outputs
    Fp6() output out;

    Fp2() a_a <== Fp2mul()(op1.x, c0);
    Fp2() b_b <== Fp2mul()(op1.y, c1);

    Fp2() tmp1 <== Fp2add()(op1.y, op1.z);
    Fp2() tmp2 <== Fp2mul()(tmp1, c1);
    Fp2() tmp3 <== Fp2sub()(tmp2, b_b);
    Fp2() tmp4 <== MulFp2ByNonresidue()(tmp3);
    Fp2() t1 <== Fp2add()(tmp4, a_a);

    Fp2() tmp5 <== Fp2add()(op1.x, op1.z);
    Fp2() tmp6 <== Fp2mul()(tmp5, c0);
    Fp2() tmp7 <== Fp2sub()(tmp6, a_a);
    Fp2() t3 <== Fp2add()(tmp7, b_b);

    Fp2() tmp8 <== Fp2add()(c0, c1);
    Fp2() tmp9 <== Fp2add()(op1.x, op1.y);
    Fp2() tmp10 <== Fp2mul()(tmp8, tmp9);
    Fp2() tmp11 <== Fp2sub()(tmp10, a_a);
    Fp2() t2 <== Fp2sub()(tmp11, b_b);

    out.x <== t1;
    out.y <== t2;
    out.z <== t3;
}

template MulFp6ByNonresidue() {
    input Fp6() in;
    output Fp6() out;

    out.y <== in.x;                                 
    out.x <== MulFp2ByNonresidue()(in.z);          
    out.z <== in.y;
}

template MulFp2ByNonresidue() {
    input Fp2() in;
    output Fp2() out;

    Fp2() NONRESIDUE;
    NONRESIDUE.x <== 9;
    NONRESIDUE.y <== 1;

    out <== Fp2mul()(in, NONRESIDUE);
}

template LineDouble(){
    G2Projective() input R;
    input signal two_inv;

    G2Projective() output R_Double;
    Fp6() output ell_coeff; 
    

    Fp2() a_first1 <== Fp2mul()(R.x, R.y);
    Fp2() a <== Fp2mulbyfp()(two_inv, a_first1);

    Fp2() b <== Fp2mul()(R.y, R.y);
    Fp2() c <== Fp2mul()(R.z, R.z);

    Fp2() c_double <== Fp2add()(c, c);
    Fp2() c2_plus_c <== Fp2add()(c_double, c);

    Fp2() COEFF_B;
    COEFF_B.x <-- 19485874751759354771024239261021720505790618469301721065564631296452457478373;
    COEFF_B.y <-- 266929791119991161246907387137283842545076965332900288569378510910307636690;

    Fp2() e <== Fp2mul()(COEFF_B, c2_plus_c);
    Fp2() e_double <== Fp2add()(e, e);
    Fp2() f <== Fp2add()(e_double, e);

    Fp2() g_part1 <== Fp2add()(b, f);
    Fp2() g <== Fp2mulbyfp()(two_inv, g_part1);

    Fp2() h_part1 <== Fp2add()(R.y, R.z);
    Fp2() h_part2 <== Fp2mul()(h_part1, h_part1);
    Fp2() b_plus_c <== Fp2add()(b, c);
    Fp2() h <== Fp2sub()(h_part2, b_plus_c);

    Fp2() i <== Fp2sub()(e, b);
    Fp2() j <== Fp2mul()(R.x, R.x);
    Fp2() e_square <== Fp2mul()(e, e);

    Fp2() b_minus_f <== Fp2sub()(b, f);
    R_Double.x <== Fp2mul()(a, b_minus_f);

    Fp2() g_square <== Fp2mul()(g, g);
    Fp2() e_square_double <== Fp2add()(e_square, e_square);
    Fp2() l <== Fp2add()(e_square_double, e_square);
    R_Double.y <== Fp2sub()(g_square, l);

    R_Double.z <== Fp2mul()(b, h);
    
    ell_coeff.x.x <== - h.x;
    ell_coeff.x.y <== - h.y;
    ell_coeff.y <== Fp2add()(Fp2add()(j, j), j);
   
    ell_coeff.z <== i; 

}

template LineAddition() {
    G2Projective() input R;
    G2Affine() input Q;

    G2Projective() output R_New;
    Fp6() output ell_coeff;

    Fp2() Qy_Rz <== Fp2mul()(Q.y, R.z);
    Fp2() theta <== Fp2sub()(R.y, Qy_Rz);

    Fp2() Qx_Rz <== Fp2mul()(Q.x, R.z);
    Fp2() lambda <== Fp2sub()(R.x, Qx_Rz);

    Fp2() c <== Fp2mul()(theta, theta);
    Fp2() d <== Fp2mul()(lambda, lambda);
    Fp2() e <== Fp2mul()(lambda, d);
    Fp2() f <== Fp2mul()(R.z, c);
    Fp2() g <== Fp2mul()(R.x, d);
    Fp2() g_double <== Fp2add()(g, g); 

    Fp2() h_temp <== Fp2add()(e, f);
    Fp2() h <== Fp2sub()(h_temp, g_double);

    R_New.x <== Fp2mul()(lambda, h);
    Fp2() g_minus_h <== Fp2sub()(g, h);
    Fp2() e_Ry <== Fp2mul()(e, R.y);
    Fp2() theta_g_minus_h <== Fp2mul()(theta, g_minus_h);
    R_New.y <== Fp2sub()(theta_g_minus_h, e_Ry);
    R_New.z <== Fp2mul()(R.z, e);

    Fp2() theta_Qx <== Fp2mul()(theta, Q.x);
    Fp2() lambda_Qy <== Fp2mul()(lambda, Q.y);
    Fp2() j <== Fp2sub()(theta_Qx, lambda_Qy);

    ell_coeff.x <== lambda;
    ell_coeff.y.x <== -theta.x;
    ell_coeff.y.y <== -theta.y;
    ell_coeff.z <== j;
}

template EllCoeffs(){
    var n = 64;
    input G2Affine() Q;

    output Fp6() ell_coeff[2 * n + 2];
    G2Projective() R[2 * n + 2];

    signal two_inv <== Fpinv()(2);


    R[0] <== G2toProjective()(Q);

    G2Affine() neg_Q;
    neg_Q.x <== Q.x;
    neg_Q.y.x <== -Q.y.x;
    neg_Q.y.y <== -Q.y.y;


    var bits[n] = [
        0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0,
        0, -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0, -1, 0,
        -1, 0, 0, 0, 1, 0, 1
    ];
    
    for (var i = 0; i < n; i++){
       
       (R[2 * i + 1], ell_coeff[2 * i]) <== LineDouble()(R[2 * i], two_inv);
       
       if (bits[n - i - 1] == 1){
            (R[2 * i + 2], ell_coeff[2 * i + 1]) <== LineAddition()(R[2 * i + 1], Q);
       }
       else if (bits[n - i - 1] == -1){
            (R[2 * i + 2], ell_coeff[2 * i + 1]) <== LineAddition()(R[2 * i + 1], neg_Q);
       }
       else{
            (R[2 * i + 2], ell_coeff[2 * i + 1]) <==  (R[2 * i + 1], ell_coeff[2 * i]);
       }
    }
    
    G2Affine() Q1 <== MulByChar()(Q);
    G2Affine() Q2 <== MulByChar()(Q1);
    
    G2Affine() Q2_neg; 
    Q2_neg.x <== Q2.x;
    Q2_neg.y.x <== -Q2.y.x;
    Q2_neg.y.y <== -Q2.y.y;

    (R[2 * n + 1], ell_coeff[2 * n]) <== LineAddition()(R[2 * n], Q1);
    (_, ell_coeff[2 * n + 1]) <== LineAddition()(R[2 * n + 1], Q2_neg);

}

template MulByChar() {
    input G2Affine() in;
    output G2Affine() out;
    
    Fp2() TWIST_MUL_BY_Q_X;
    TWIST_MUL_BY_Q_X.x <== 21575463638280843010398324269430826099269044274347216827212613867836435027261;
    TWIST_MUL_BY_Q_X.y <== 10307601595873709700152284273816112264069230130616436755625194854815875713954;
  
    Fp2() TWIST_MUL_BY_Q_Y;
    TWIST_MUL_BY_Q_Y.x <== 2821565182194536844548159561693502659359617185244120367078079554186484126554;
    TWIST_MUL_BY_Q_Y.y <== 3505843767911556378687030309984248845540243509899259641013678093033130930403;
   
    Fp2() t1 <== Fp2conjugate()(in.x);            // Frobenius of in.x
    out.x <== Fp2mul()(t1, TWIST_MUL_BY_Q_X);
   
    Fp2() t2 <== Fp2conjugate()(in.y);            // Frobenius of in.y
    out.y <== Fp2mul()(t2, TWIST_MUL_BY_Q_Y);
}

template Pairing() {
    G2Affine() input Q;
    G1Projective() input P;
    Fp12() output out;

    Fp12() miller_output <== MillerLoop()(Q, P);
    out <== FinalExp()(miller_output);
}

