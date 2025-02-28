pragma circom 2.2.1;
include "../sum_check/sumcheck.circom";
include "./../../../pcs/hyperkzgjolt1.circom";

template ReduceAndVerify(byte_code_read_write_opening_len, byte_code_init_final_opening_len,
                        inst_primary_sum_check_opening_len, inst_read_write_opening_len, inst_init_final_opening_len,
                        memory_checking_read_write_opening_len, memory_checking_init_final_opening_len,
                        output_sum_check_opening_len, timestamp_validity_opening_len,
                        r1cs_opening_len,
                        num_sumcheck_rounds){

    var num_openings = 10;

    input Transcript() transcript;
    input ReducedOpeningProof(num_sumcheck_rounds, num_openings) reduced_opening_proof;

    input VerifierOpening(byte_code_read_write_opening_len) byte_code_read_write_openings;
    input VerifierOpening(byte_code_init_final_opening_len) byte_code_init_final_openings;
    input VerifierOpening(inst_primary_sum_check_opening_len) inst_primary_sum_check_openings;
    input VerifierOpening(inst_read_write_opening_len) inst_read_write_openings;
    input VerifierOpening(inst_init_final_opening_len) inst_init_final_openings;
    input VerifierOpening(memory_checking_read_write_opening_len) memory_checking_read_write_openings;
    input VerifierOpening(memory_checking_init_final_opening_len) memory_checking_init_final_openings;
    input VerifierOpening(output_sum_check_opening_len) output_sum_check_openings;
    input VerifierOpening(timestamp_validity_opening_len) timestamp_validity_openings;
    input VerifierOpening(r1cs_opening_len) r1cs_openings;

    Transcript() int_transcript[4];

    signal gamma_powers[num_openings];

    
   signal temp_rho[num_openings];
   signal int_claims_rho[num_openings];
   signal temp_gamma[num_openings];
   signal int_claims_gamma[num_openings];

   signal rho_powers[num_openings],  sumcheck_claim, r_sumcheck[num_sumcheck_rounds];
    
    (int_transcript[0], rho_powers) <== ChallengeScalarPowers(num_openings)(transcript);
    
    (sumcheck_claim, r_sumcheck, int_transcript[1]) <== VerifyBatchOpeningReduction(byte_code_read_write_opening_len, byte_code_init_final_opening_len,
                                                        inst_primary_sum_check_opening_len, inst_read_write_opening_len, inst_init_final_opening_len,
                                                        memory_checking_read_write_opening_len, memory_checking_init_final_opening_len,
                                                        output_sum_check_opening_len, timestamp_validity_opening_len,
                                                        r1cs_opening_len, num_sumcheck_rounds, num_openings)
                                                        (rho_powers, reduced_opening_proof.sumcheck_proof, int_transcript[0],
                                                        byte_code_read_write_openings, byte_code_init_final_openings,
                                                         inst_primary_sum_check_openings, inst_read_write_openings, inst_init_final_openings,
                                                        memory_checking_read_write_openings, memory_checking_init_final_openings, output_sum_check_openings, timestamp_validity_openings, r1cs_openings);
    
    // Compute random linear combination of the claims, accounting for the fact that the
    // polynomials may be of different sizes
   signal bc_rw_r_lo[num_sumcheck_rounds - byte_code_read_write_opening_len];
   signal bc_rw_r_hi[byte_code_read_write_opening_len];
    (bc_rw_r_lo, bc_rw_r_hi) <== SplitAt(num_sumcheck_rounds - byte_code_read_write_opening_len, num_sumcheck_rounds)(r_sumcheck);
   signal bc_rw_eq_eval <== EvaluateEq(byte_code_read_write_opening_len)(bc_rw_r_hi, byte_code_read_write_openings.opening_point);
    temp_rho[0] <== (bc_rw_eq_eval *  reduced_opening_proof.sumcheck_claims[0]);
    int_claims_rho[0] <== (temp_rho[0] * rho_powers[0]);

   signal bc_if_r_lo[num_sumcheck_rounds - byte_code_init_final_opening_len];
   signal bc_if_r_hi[byte_code_init_final_opening_len];
    (bc_if_r_lo, bc_if_r_hi) <== SplitAt(num_sumcheck_rounds - byte_code_init_final_opening_len, num_sumcheck_rounds)(r_sumcheck);
   signal bc_if_eq_eval <== EvaluateEq(byte_code_init_final_opening_len)(bc_if_r_hi, byte_code_init_final_openings.opening_point);
    temp_rho[1] <== (bc_if_eq_eval * reduced_opening_proof.sumcheck_claims[1]);
    int_claims_rho[1] <==(temp_rho[1] * rho_powers[1]);

   signal inst_psc_r_lo[num_sumcheck_rounds - inst_primary_sum_check_opening_len];
   signal inst_psc_r_hi[inst_primary_sum_check_opening_len];
    (inst_psc_r_lo, inst_psc_r_hi) <== SplitAt(num_sumcheck_rounds - inst_primary_sum_check_opening_len, num_sumcheck_rounds)(r_sumcheck);
   signal inst_psc_eq_eval <== EvaluateEq(inst_primary_sum_check_opening_len)(inst_psc_r_hi, inst_primary_sum_check_openings.opening_point);
    temp_rho[2] <== (inst_psc_eq_eval *  reduced_opening_proof.sumcheck_claims[2]);
    int_claims_rho[2] <== (temp_rho[2] * rho_powers[2]);

   signal inst_rw_r_lo[num_sumcheck_rounds - inst_read_write_opening_len];
   signal inst_rw_r_hi[inst_read_write_opening_len];
    (inst_rw_r_lo, inst_rw_r_hi) <== SplitAt(num_sumcheck_rounds - inst_read_write_opening_len, num_sumcheck_rounds)(r_sumcheck);
    signal inst_rw_eq_eval <== EvaluateEq(inst_read_write_opening_len)(inst_rw_r_hi, inst_read_write_openings.opening_point);
    temp_rho[3] <== (inst_rw_eq_eval * reduced_opening_proof.sumcheck_claims[3]);
    int_claims_rho[3] <== (temp_rho[3] * rho_powers[3]);

   signal inst_if_r_lo[num_sumcheck_rounds - inst_init_final_opening_len];

   signal inst_if_r_hi[inst_init_final_opening_len];
    (inst_if_r_lo, inst_if_r_hi) <== SplitAt(num_sumcheck_rounds - inst_init_final_opening_len, num_sumcheck_rounds)(r_sumcheck);
   signal inst_if_eq_eval <== EvaluateEq(inst_init_final_opening_len)(inst_if_r_hi, inst_init_final_openings.opening_point);
    temp_rho[4] <== (inst_if_eq_eval * reduced_opening_proof.sumcheck_claims[4]);
    int_claims_rho[4] <== (temp_rho[4] * rho_powers[4]);

   signal mc_rw_r_lo[num_sumcheck_rounds - memory_checking_read_write_opening_len];
   signal mc_rw_r_hi[memory_checking_read_write_opening_len];
    (mc_rw_r_lo, mc_rw_r_hi) <== SplitAt(num_sumcheck_rounds - memory_checking_read_write_opening_len, num_sumcheck_rounds)(r_sumcheck);
   signal mc_rw_eq_eval <== EvaluateEq(memory_checking_read_write_opening_len)(mc_rw_r_hi, memory_checking_read_write_openings.opening_point);
    temp_rho[5] <== (mc_rw_eq_eval * reduced_opening_proof.sumcheck_claims[5]);
    int_claims_rho[5] <== (temp_rho[5] * rho_powers[5]);

   signal mc_if_r_lo[num_sumcheck_rounds - memory_checking_init_final_opening_len];
   signal mc_if_r_hi[memory_checking_init_final_opening_len];
    (mc_if_r_lo, mc_if_r_hi) <== SplitAt(num_sumcheck_rounds - memory_checking_init_final_opening_len, num_sumcheck_rounds)(r_sumcheck);
   signal mc_if_eq_eval <== EvaluateEq(memory_checking_init_final_opening_len)(mc_if_r_hi, memory_checking_init_final_openings.opening_point);
    temp_rho[6] <== (mc_if_eq_eval * reduced_opening_proof.sumcheck_claims[6]);
    int_claims_rho[6] <==(temp_rho[6] * rho_powers[6]);

   signal osc_r_lo[num_sumcheck_rounds - output_sum_check_opening_len];
   signal osc_r_hi[output_sum_check_opening_len];
    (osc_r_lo, osc_r_hi) <== SplitAt(num_sumcheck_rounds - output_sum_check_opening_len, num_sumcheck_rounds)(r_sumcheck);
   signal osc_eq_eval <== EvaluateEq(output_sum_check_opening_len)(osc_r_hi, output_sum_check_openings.opening_point);
    temp_rho[7] <== (osc_eq_eval * reduced_opening_proof.sumcheck_claims[7]);
    int_claims_rho[7] <== (temp_rho[7] * rho_powers[7]);

   signal tsv_r_lo[num_sumcheck_rounds - timestamp_validity_opening_len];
   signal tsv_r_hi[timestamp_validity_opening_len];
    (tsv_r_lo, tsv_r_hi) <== SplitAt(num_sumcheck_rounds - timestamp_validity_opening_len, num_sumcheck_rounds)(r_sumcheck);
   signal tsv_eq_eval <== EvaluateEq(timestamp_validity_opening_len)(tsv_r_hi, timestamp_validity_openings.opening_point);
    temp_rho[8] <== (tsv_eq_eval * reduced_opening_proof.sumcheck_claims[8]);
    int_claims_rho[8] <== (temp_rho[8] * rho_powers[8]);

   signal r1cs_r_lo[num_sumcheck_rounds - r1cs_opening_len];
   signal r1cs_r_hi[r1cs_opening_len];
    (r1cs_r_lo, r1cs_r_hi) <== SplitAt(num_sumcheck_rounds - r1cs_opening_len, num_sumcheck_rounds)(r_sumcheck);
   signal r1cs_eq_eval <== EvaluateEq(r1cs_opening_len)(r1cs_r_hi, r1cs_openings.opening_point);
    temp_rho[9] <== (r1cs_eq_eval * reduced_opening_proof.sumcheck_claims[9]);
    int_claims_rho[9] <== (temp_rho[9] * rho_powers[9]);
    
   signal expected_sumcheck_claim <== Sum(num_openings)(int_claims_rho);

    expected_sumcheck_claim === sumcheck_claim;

    int_transcript[2] <== AppendScalars(num_openings)(reduced_opening_proof.sumcheck_claims, int_transcript[1]);
    
    (int_transcript[3], gamma_powers) <== ChallengeScalarPowers(num_openings)(int_transcript[2]); 
    
    output signal gamma <== gamma_powers[1];


    signal bc_rw_le <== Product2(num_sumcheck_rounds - byte_code_read_write_opening_len)(bc_rw_r_lo);
    temp_gamma[0] <== (bc_rw_le * reduced_opening_proof.sumcheck_claims[0]);
    int_claims_gamma[0] <== (temp_gamma[0]* gamma_powers[0]);

    signal bc_if_le <== Product2(num_sumcheck_rounds - byte_code_init_final_opening_len)(bc_if_r_lo);
    temp_gamma[1] <== (bc_if_le * reduced_opening_proof.sumcheck_claims[1]);
    int_claims_gamma[1] <== (temp_gamma[1] * gamma_powers[1]);

    signal inst_psc_le <== Product2(num_sumcheck_rounds - inst_primary_sum_check_opening_len)(inst_psc_r_lo);
    temp_gamma[2] <== (inst_psc_le * reduced_opening_proof.sumcheck_claims[2]);
    int_claims_gamma[2] <== (temp_gamma[2] * gamma_powers[2]);

    signal inst_rw_le <== Product2(num_sumcheck_rounds - inst_read_write_opening_len)(inst_rw_r_lo);
    temp_gamma[3] <== (inst_rw_le * reduced_opening_proof.sumcheck_claims[3]);
    int_claims_gamma[3] <== (temp_gamma[3] * gamma_powers[3]);

    signal inst_if_le <== Product2(num_sumcheck_rounds - inst_init_final_opening_len)(inst_if_r_lo);
    temp_gamma[4] <== (inst_if_le * reduced_opening_proof.sumcheck_claims[4]);
    int_claims_gamma[4] <== (temp_gamma[4] * gamma_powers[4]);

    signal mc_rw_le <== Product2(num_sumcheck_rounds - memory_checking_read_write_opening_len)(mc_rw_r_lo);
    temp_gamma[5] <== (mc_rw_le * reduced_opening_proof.sumcheck_claims[5]);
    int_claims_gamma[5] <== (temp_gamma[5] * gamma_powers[5]);

    signal mc_if_le <== Product2(num_sumcheck_rounds - memory_checking_init_final_opening_len)(mc_if_r_lo);
    temp_gamma[6] <== (mc_if_le * reduced_opening_proof.sumcheck_claims[6]);
    int_claims_gamma[6] <== (temp_gamma[6] * gamma_powers[6]);

    signal osc_le <== Product2(num_sumcheck_rounds - output_sum_check_opening_len)(osc_r_lo);
    temp_gamma[7] <== (osc_le * reduced_opening_proof.sumcheck_claims[7]);
    int_claims_gamma[7] <== (temp_gamma[7] * gamma_powers[7]);

    signal tsv_le <== Product2(num_sumcheck_rounds - timestamp_validity_opening_len)(tsv_r_lo);
    temp_gamma[8] <== (tsv_le *  reduced_opening_proof.sumcheck_claims[8]);
    int_claims_gamma[8] <== (temp_gamma[8] * gamma_powers[8]);

    signal r1cs_le <== Product2(num_sumcheck_rounds - r1cs_opening_len)(r1cs_r_lo);
    temp_gamma[9] <== (r1cs_le * reduced_opening_proof.sumcheck_claims[9]);
    int_claims_gamma[9] <== (temp_gamma[9] * gamma_powers[9]);
    signal joint_claim <== Sum(num_openings)(int_claims_gamma);

    output HyperKzgVerifierAdvice()  hyperkzg_verifier_advice;
    log("joint_claim", joint_claim);
    for (var i = 0; i < num_sumcheck_rounds - 1; i++){
        log("r_sumcheckm", r_sumcheck[i]);
        log("com.x limb 0",reduced_opening_proof.joint_opening_proof.com[i].x.limbs[0]);
        log("com.x limb 1",reduced_opening_proof.joint_opening_proof.com[i].x.limbs[1]);
        log("com.x limb 2",reduced_opening_proof.joint_opening_proof.com[i].x.limbs[2]);
        log("com.y limb 0",reduced_opening_proof.joint_opening_proof.com[i].y.limbs[0]);
        log("com.y limb 1",reduced_opening_proof.joint_opening_proof.com[i].y.limbs[1]);
        log("com.y limb 2",reduced_opening_proof.joint_opening_proof.com[i].y.limbs[2]);
    }
    log("w.x limb 0",reduced_opening_proof.joint_opening_proof.w[0].x.limbs[0]);
    log("w.x limb 1",reduced_opening_proof.joint_opening_proof.w[0].x.limbs[1]);
    log("w.x limb 2",reduced_opening_proof.joint_opening_proof.w[0].x.limbs[2]);
    log("w.y limb 0",reduced_opening_proof.joint_opening_proof.w[0].y.limbs[0]);
    log("w.y limb 1",reduced_opening_proof.joint_opening_proof.w[0].y.limbs[1]);
    log("w.y limb 2",reduced_opening_proof.joint_opening_proof.w[0].y.limbs[2]);

    // log("w.x",reduced_opening_proof.joint_opening_proof.w[1].x);
    // log("w.y",reduced_opening_proof.joint_opening_proof.w[1].y);
    
    // log("w.x",reduced_opening_proof.joint_opening_proof.w[2].x);
    // log("w.y",reduced_opening_proof.joint_opening_proof.w[2].y);
    
    hyperkzg_verifier_advice <== HyperKzgVerifierJolt1(num_sumcheck_rounds)(r_sumcheck, reduced_opening_proof.joint_opening_proof, joint_claim, int_transcript[3]);

}


template VerifyBatchOpeningReduction(byte_code_read_write_opening_len, byte_code_init_final_opening_len,
                                    inst_primary_sum_check_opening_len, inst_read_write_opening_len, inst_init_final_opening_len,
                                    memory_checking_read_write_opening_len, memory_checking_init_final_opening_len,
                                    output_sum_check_opening_len, timestamp_validity_opening_len,
                                    r1cs_opening_len, num_sumcheck_rounds, num_openings
                                    ){

    input signal rho_powers[num_openings];    
    input SumcheckInstanceProof(2, num_sumcheck_rounds) sumcheck_proof;
    input Transcript() transcript;

    input VerifierOpening(byte_code_read_write_opening_len) byte_code_read_write_openings;
    input VerifierOpening(byte_code_init_final_opening_len) byte_code_init_final_openings;
    input VerifierOpening(inst_primary_sum_check_opening_len) inst_primary_sum_check_openings;
    input VerifierOpening(inst_read_write_opening_len) inst_read_write_openings;
    input VerifierOpening(inst_init_final_opening_len) inst_init_final_openings;
    input VerifierOpening(memory_checking_read_write_opening_len) memory_checking_read_write_openings;
    input VerifierOpening(memory_checking_init_final_opening_len) memory_checking_init_final_openings;
    input VerifierOpening(output_sum_check_opening_len) output_sum_check_openings;
    input VerifierOpening(timestamp_validity_opening_len) timestamp_validity_openings;
    input VerifierOpening(r1cs_opening_len) r1cs_openings;

  
    output signal sumcheck_claim;
    output signal r_sumcheck[num_sumcheck_rounds];
    output Transcript() up_transcript;

   signal two_powers[num_openings];
    var fp_two_powers[num_openings];
   signal temp[num_openings];
   signal int_claims[num_openings];

    fp_two_powers[0] = 1 << (num_sumcheck_rounds - byte_code_read_write_opening_len);
    two_powers[0] <-- fp_two_powers[0];

    temp[0] <== (two_powers[0] * rho_powers[0]);
    int_claims[0] <== (temp[0] *  byte_code_read_write_openings.claim);

    fp_two_powers[1] = 1 << (num_sumcheck_rounds - byte_code_init_final_opening_len);
    two_powers[1] <-- fp_two_powers[1];

    temp[1] <== (two_powers[1] * rho_powers[1]);
    int_claims[1] <== (temp[1] * byte_code_init_final_openings.claim);

    fp_two_powers[2] = 1 << (num_sumcheck_rounds - inst_primary_sum_check_opening_len);
    two_powers[2] <-- fp_two_powers[2];

    temp[2] <== (two_powers[2] * rho_powers[2]);
    int_claims[2] <== (temp[2] * inst_primary_sum_check_openings.claim);

    fp_two_powers[3] = 1 << (num_sumcheck_rounds - inst_read_write_opening_len);
    two_powers[3] <-- fp_two_powers[3];
    
    temp[3] <== (two_powers[3] * rho_powers[3]);
    int_claims[3] <== (temp[3] * inst_read_write_openings.claim);

    fp_two_powers[4] = 1 << (num_sumcheck_rounds - inst_init_final_opening_len);
    two_powers[4] <-- fp_two_powers[4];
   
    temp[4] <== (two_powers[4] * rho_powers[4]);
    int_claims[4] <== (temp[4] * inst_init_final_openings.claim);

    fp_two_powers[5] = 1 << (num_sumcheck_rounds - memory_checking_read_write_opening_len);
    two_powers[5] <--fp_two_powers[5];
    
    temp[5] <== (two_powers[5] * rho_powers[5]);
    int_claims[5] <== (temp[5] * memory_checking_read_write_openings.claim);

    fp_two_powers[6] = 1 << (num_sumcheck_rounds - memory_checking_init_final_opening_len);
    two_powers[6] <-- fp_two_powers[6];

    temp[6] <== (two_powers[6]* rho_powers[6]);
    int_claims[6] <== (temp[6]* memory_checking_init_final_openings.claim);

    fp_two_powers[7] = 1 << (num_sumcheck_rounds - output_sum_check_opening_len);
    two_powers[7] <-- fp_two_powers[7];

    temp[7] <== (two_powers[7]* rho_powers[7]);
    int_claims[7] <== (temp[7]* output_sum_check_openings.claim);

    fp_two_powers[8] = 1 << (num_sumcheck_rounds - timestamp_validity_opening_len);
    two_powers[8] <-- fp_two_powers[8];
  
    temp[8] <== (two_powers[8] * rho_powers[8]);
    int_claims[8] <== (temp[8] * timestamp_validity_openings.claim);

    fp_two_powers[9] = 1 << (num_sumcheck_rounds - r1cs_opening_len);
    two_powers[9] <-- fp_two_powers[9];
    
    temp[9] <== (two_powers[9] * rho_powers[9]);
    int_claims[9] <== (temp[9] * r1cs_openings.claim);

   signal combined_claim <== Sum(num_openings)(int_claims);  
    (up_transcript, sumcheck_claim, r_sumcheck) <== SumCheck(num_sumcheck_rounds, 2)(combined_claim, sumcheck_proof, transcript); 
}
