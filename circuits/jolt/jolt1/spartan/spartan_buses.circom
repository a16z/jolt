pragma circom 2.2.1;
include "./../../../fields/non_native/non_native_over_bn_base.circom";

bus R1CSProof(outer_num_rounds, inner_num_rounds, num_spartan_witness_evals){
    UniformSpartanProof(outer_num_rounds, inner_num_rounds, num_spartan_witness_evals) proof;
}

bus SparseConstraints(num_non_zero_non_const_coeff, num_non_zero_const_coeff) {
    
    signal row[num_non_zero_non_const_coeff];
    signal col[num_non_zero_non_const_coeff];
    signal val[num_non_zero_non_const_coeff];

    
    signal uniform_row_index[num_non_zero_const_coeff];
    signal coeff[num_non_zero_const_coeff];
}

bus UniformSpartanProof(outer_num_rounds, inner_num_rounds, num_spartan_witness_evals) {
    SumcheckInstanceProof(3, outer_num_rounds) outer_sumcheck_proof; 
    
    signal outer_sumcheck_claims[3];
    
    SumcheckInstanceProof(2, inner_num_rounds) inner_sumcheck_proof;
    
    signal claimed_witness_evals[num_spartan_witness_evals];
}