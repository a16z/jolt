pragma circom 2.2.1;
include "./../../../fields/non_native/non_native_over_bn_scalar.circom";
include "../../utils.circom";
include "./../../../fields/non_native/utils.circom";
include "./../utils.circom";
include "./../../../transcript/grumpkin_transcript.circom";

template NonNativeSumCheck(rounds, degree) {
    input Fq() initialClaim ;
    input SumcheckInstanceProof(degree, rounds) sumcheck_proof;
    input Transcript() transcript ;
    output Transcript() up_transcript;
    output Fq() finalClaim ; 
    output Fq() randomPoints[rounds];  

    Transcript() int_transcript[rounds + 1];
    Fq() r;

    component eval_at_01[rounds];
    component eval_at_r[rounds];
    
    Fq() claim[rounds + 1]; 
    claim[0] <== initialClaim;

    Transcript()  temp_transcript[rounds];
 
    int_transcript[0] <== transcript;
    
    for (var i = 0; i < rounds; i++)
    {  
        eval_at_01[i] = eval_at_01(degree);
        eval_at_01[i].poly <== sumcheck_proof.uni_polys[i].coeffs;
        log("i is ", i);

        NonNativeEquality()(claim[i], eval_at_01[i].eval);
        
        // reseeding with sumcheck_proof.uni_polys[i].coeffs
        temp_transcript[i] <== AppendScalars(degree + 1)(sumcheck_proof.uni_polys[i].coeffs, int_transcript[i]);
        (int_transcript[i + 1], randomPoints[i]) <== ChallengeScalar()(temp_transcript[i]);
        
        eval_at_r[i] = EvalUniPoly(degree);
        eval_at_r[i].poly <== sumcheck_proof.uni_polys[i].coeffs;
        eval_at_r[i].random_point <== randomPoints[i];
 
       claim[i + 1] <== eval_at_r[i].eval;
    } 
    finalClaim <== claim[rounds] ;
    up_transcript <== int_transcript[rounds];
}
 
 

//Circuit to compute f(0) + f(1)
template eval_at_01(degree){
    input Fq()  poly[degree + 1];
    output Fq()  eval;

    Fq() add[degree + 1];  

    add[0] <== poly[0];
     for (var i = 1; i < degree + 1; i++ )
    {           
        add[i] <== NonNativeAdd()(add[i - 1], poly[i]);
    }
     eval <== NonNativeAdd()(add[degree], poly[0]);

}

// component main = NonNativeSumCheck(3, 1); 