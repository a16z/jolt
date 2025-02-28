pragma circom 2.2.1;
include "./../utils.circom";
include "./../../../transcript/transcript.circom";

template SumCheck(rounds, degree) {
    input signal initialClaim ;
    input SumcheckInstanceProof(degree, rounds) sumcheck_proof;
    input Transcript() transcript ;
    output Transcript() up_transcript;
    output signal finalClaim ; 
    output signal randomPoints[rounds];  

    Transcript() int_transcript[rounds + 1];
    signal r;

    component eval_at_01[rounds];
    component eval_at_r[rounds];
    
    signal claim[rounds + 1]; 
    claim[0] <== initialClaim;

    Transcript()  temp_transcript[rounds];
 
    int_transcript[0] <== transcript;
    

    for (var i = 0; i < rounds; i++)
    {  
        eval_at_01[i] = eval_at_01(degree);
        eval_at_01[i].poly <== sumcheck_proof.uni_polys[i].coeffs;
        
        claim[i] === eval_at_01[i].eval;
        
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
    input signal  poly[degree + 1];
    output signal  eval;

    signal add[degree + 1];  

    add[0] <== poly[0];

    for (var i = 1; i < degree + 1; i++ ){           
        add[i] <== (add[i - 1] + poly[i]);
    }

    eval <== (add[degree] +  poly[0]);
}

bus PrimarySumcheckOpenings(NUM_MEMORIES, NUM_INSTRUCTIONS) {
    /// Evaluations of the E_i polynomials at the opening point. Vector is of length NUM_MEMORIES.
    signal E_poly_openings[NUM_MEMORIES];

    /// Evaluations of the flag polynomials at the opening point. Vector is of length NUM_INSTRUCTIONS.
    signal flag_openings[NUM_INSTRUCTIONS];

    /// Evaluation of the lookup_outputs polynomial at the opening point.
    signal lookup_outputs_opening;
}

bus SumcheckInstanceProof(degree, rounds) {
    UniPoly(degree) uni_polys[rounds];
}

bus PrimarySumcheck(degree, rounds, NUM_MEMORIES, NUM_INSTRUCTIONS) {
    SumcheckInstanceProof(degree, rounds) sumcheck_proof;
    //signal num_rounds;
    PrimarySumcheckOpenings(NUM_MEMORIES, NUM_INSTRUCTIONS) openings;
}

//  component main = SumCheck(3, 1); 