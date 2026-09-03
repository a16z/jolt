fn tamper_suite(
    proof: &WrapperProof,
    _wire_phase_groups: [usize; 5],
    _k: usize,
    rejected: impl Fn(&WrapperProof) -> bool,
) {
    let original = proof.clone();
    let tamper = |edit: &dyn Fn(&mut WrapperProof)| {
        let mut candidate = original.clone();
        edit(&mut candidate);
        assert!(rejected(&candidate));
    };
    for challenge in 0..original.public_challenges.len() {
        tamper(&|candidate| candidate.public_challenges[challenge][0] ^= 1);
    }
    for commitment in 0..original.commitments.len() {
        tamper(&|candidate| {
            candidate.commitments[commitment] = Commitment::new(original.opening.com[0]);
        });
    }
    for stage in 0..original.stages.len() {
        for round in 0..original.stages[stage]
            .round_polynomials
            .round_polynomials
            .len()
        {
            for coefficient in 0..original.stages[stage].round_polynomials.round_polynomials[round]
                .coeffs_except_linear_term()
                .len()
            {
                tamper(&|candidate| {
                    let polynomial =
                        &mut candidate.stages[stage].round_polynomials.round_polynomials[round];
                    let mut coefficients = polynomial.coeffs_except_linear_term().to_vec();
                    coefficients[coefficient] += Fr::one();
                    *polynomial = CompressedPoly::new(coefficients);
                });
            }
        }
        if let Some(committed) = &original.stages[stage].committed_rounds {
            for round in 0..committed.round_commitments.len() {
                tamper(&|candidate| {
                    candidate.stages[stage]
                        .committed_rounds
                        .as_mut()
                        .expect("committed stage")
                        .round_commitments[round] += Bn254::g1_generator();
                });
            }
            for round in 0..committed.round_claims.len() {
                tamper(&|candidate| {
                    candidate.stages[stage]
                        .committed_rounds
                        .as_mut()
                        .expect("committed stage")
                        .round_claims[round] += Fr::one();
                });
            }
            tamper(&|candidate| {
                candidate.stages[stage]
                    .committed_rounds
                    .as_mut()
                    .expect("committed stage")
                    .sum_at_zero += Fr::one();
            });
            if committed.opening.is_some() {
                tamper(&|candidate| {
                    candidate.stages[stage]
                        .committed_rounds
                        .as_mut()
                        .expect("committed stage")
                        .opening
                        .as_mut()
                        .expect("stage opening")
                        .shifted_commitment += Bn254::g1_generator();
                });
                tamper(&|candidate| {
                    candidate.stages[stage]
                        .committed_rounds
                        .as_mut()
                        .expect("committed stage")
                        .opening
                        .as_mut()
                        .expect("stage opening")
                        .quotient_commitment += Bn254::g1_generator();
                });
                tamper(&|candidate| {
                    candidate.stages[stage]
                        .committed_rounds
                        .as_mut()
                        .expect("committed stage")
                        .opening
                        .as_mut()
                        .expect("stage opening")
                        .evaluation_witness += Bn254::g1_generator();
                });
            }
        }
    }
    if original.round_opening.is_some() {
        tamper(&|candidate| {
            candidate
                .round_opening
                .as_mut()
                .expect("shared round opening")
                .shifted_commitment += Bn254::g1_generator();
        });
        tamper(&|candidate| {
            candidate
                .round_opening
                .as_mut()
                .expect("shared round opening")
                .quotient_commitment += Bn254::g1_generator();
        });
        tamper(&|candidate| {
            candidate
                .round_opening
                .as_mut()
                .expect("shared round opening")
                .evaluation_witness += Bn254::g1_generator();
        });
    }
    for stage in 0..original.stage_claims.len() {
        for claim in 0..original.stage_claims[stage].len() {
            tamper(&|candidate| candidate.stage_claims[stage][claim] += Fr::one());
        }
    }
    for evaluation in 0..original.term_evaluations.len() {
        tamper(&|candidate| candidate.term_evaluations[evaluation] += Fr::one());
    }
    for claim in 0..original.reduced_claims.len() {
        tamper(&|candidate| candidate.reduced_claims[claim] += Fr::one());
    }
    for commitment in 0..original.opening.com.len() {
        tamper(&|candidate| candidate.opening.com[commitment] += Bn254::g1_generator());
    }
    tamper(&|candidate| candidate.opening.w += Bn254::g1_generator());
    for row in 0..original.opening.v.len() {
        for evaluation in 0..original.opening.v[row].len() {
            tamper(&|candidate| candidate.opening.v[row][evaluation] += Fr::one());
        }
    }
    tamper(&|candidate| candidate.opening.p0_at_r_squared += Fr::one());
}

fn report(
    proof: &WrapperProof,
    wire_phase_groups: [usize; 5],
    term_count: usize,
    cost: VerifierCost,
    statement_fields: usize,
    times: &[(&str, u128)],
    uptime: &[u8],
) {
    let stage_a = committed_stage_bytes(&proof.stages[0]);
    let term_stage = committed_stage_bytes(&proof.stages[1]);
    let shared = 96 * usize::from(proof.round_opening.is_some());
    let ell = 32 * proof.term_evaluations.len();
    let stage_b = clear_stage_bytes(&proof.stages[2]);
    let reduced = 32 * proof.reduced_claims.len();
    let commitment_bytes = wire_phase_groups.map(|groups| 32 * groups);
    assert_eq!(
        wire_phase_groups.iter().sum::<usize>(),
        proof.commitments.len()
    );
    let opening = proof.payload_bytes()
        - commitment_bytes.iter().sum::<usize>()
        - stage_a
        - term_stage
        - shared
        - ell
        - stage_b
        - reduced;
    let serialized = encode_to_vec(proof, standard()).expect("serialize wrapper");
    assert_eq!(serialized.len(), proof.bincode_bytes());
    println!("uptime={}", String::from_utf8_lossy(uptime).trim());
    let phases = times
        .iter()
        .map(|(name, ms)| format!("{name}={ms}"))
        .collect::<Vec<_>>()
        .join(" ");
    println!("phases_ms {phases}");
    println!(
        "bytes phase1a={} phase1b={} phase2a={} phase2b={} phase2c={} stage_a={stage_a} term={term_stage} shared_bdfg={shared} ell={ell} stage_b={stage_b} reduced={reduced} hyperkzg={opening} proof={} bincode={} statement={}",
        commitment_bytes[0],
        commitment_bytes[1],
        commitment_bytes[2],
        commitment_bytes[3],
        commitment_bytes[4],
        proof.payload_bytes(),
        proof.bincode_bytes(),
        32 * statement_fields,
    );
    println!(
        "terms={term_count} term_rounds={}",
        proof.stages[1]
            .committed_rounds
            .as_ref()
            .expect("term stage")
            .round_commitments
            .len()
    );
    println!("cost={cost:?} gas={}", estimated_gas(cost, proof));
}

fn committed_stage_bytes(stage: &StageProof) -> usize {
    let committed = stage.committed_rounds.as_ref().expect("committed stage");
    32 * (committed.round_commitments.len() + committed.round_claims.len() + 1)
}

fn clear_stage_bytes(stage: &StageProof) -> usize {
    32 * stage
        .round_polynomials
        .round_polynomials
        .iter()
        .map(|round| round.coeffs_except_linear_term().len())
        .sum::<usize>()
}

fn estimated_gas(cost: VerifierCost, proof: &WrapperProof) -> usize {
    let proof_g1 = proof.commitments.len()
        + proof
            .stages
            .iter()
            .filter_map(|stage| stage.committed_rounds.as_ref())
            .map(|stage| stage.round_commitments.len() + 3 * usize::from(stage.opening.is_some()))
            .sum::<usize>()
        + 3 * usize::from(proof.round_opening.is_some())
        + proof.opening.com.len()
        + 1;
    let calldata = proof.payload_bytes() + 32 * proof_g1 + 7 * 32;
    21_000
        + 16 * calldata
        + 7_700 * cost.ec_mul
        + 20 * cost.fr_mul
        + batched_inversion_gas(cost.fr_inv)
        + 100 * cost.keccak
        + 2 * 114_700
        + 183_400
}

fn batched_inversion_gas(inversions: usize) -> usize {
    if inversions == 0 {
        return 0;
    }
    let multiplication_complexity = 32usize.div_ceil(8).pow(2);
    let iteration_count = 253;
    let modexp = (multiplication_complexity * iteration_count / 3).max(200);
    modexp + 3 * (inversions - 1) * 20
}
