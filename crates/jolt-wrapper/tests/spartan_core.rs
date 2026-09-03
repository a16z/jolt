#![expect(
    clippy::expect_used,
    reason = "test fixtures fail immediately on setup or proof errors"
)]

use bincode::config::standard;
use bincode::serde::encode_to_vec;
use jolt_crypto::Bn254;
use jolt_field::{Fr, Ring};
use jolt_hyperkzg::{HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_poly::CompressedPoly;
use jolt_r1cs::ConstraintMatrices;
use jolt_wrapper::spartan::{prove_spartan, verify_spartan};
use jolt_wrapper::stream::Commitment;

fn instance(log_size: usize, public_count: usize) -> (ConstraintMatrices<Fr>, Vec<Fr>, Vec<Fr>) {
    let size = 1usize << log_size;
    let public: Vec<Fr> = (0..public_count)
        .map(|index| Fr::from_u64(index as u64 + 3))
        .collect();
    let quarter = size / 4;
    let mut witness = vec![Fr::from_u64(0); size];
    for index in 0..quarter {
        let a_value = Fr::from_u64(index as u64 + 2);
        let b_value = Fr::from_u64(index as u64 * 3 + 5);
        let public_value = public.get(index).copied().unwrap_or(Fr::from_u64(0));
        witness[index] = a_value;
        witness[quarter + index] = b_value;
        witness[2 * quarter + index] = (a_value + public_value) * b_value;
        witness[3 * quarter + index] = Fr::from_u64(index as u64 * 11 + 1);
    }
    let witness_start = 1 + public_count;
    let mut a = Vec::with_capacity(size);
    let mut b = Vec::with_capacity(size);
    let mut c = Vec::with_capacity(size);
    for row in 0..size {
        let mut mixed = row as u64;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        let index = (mixed ^ (mixed >> 31)) as usize & (quarter - 1);
        let mut linear = vec![(witness_start + index, Fr::from_u64(1))];
        if index < public_count {
            linear.push((1 + index, Fr::from_u64(1)));
        }
        a.push(linear.clone());
        b.push(vec![(witness_start + quarter + index, Fr::from_u64(1))]);
        c.push(vec![(witness_start + 2 * quarter + index, Fr::from_u64(1))]);
    }
    (
        ConstraintMatrices::new(size, witness_start + size, a, b, c),
        public,
        witness,
    )
}

#[test]
fn spartan_round_trip_and_tampers() {
    let key_digest = [42; 32];
    let (r1cs, public, witness) = instance(12, 50);
    let setup = HyperKZGScheme::<Bn254>::setup_from_secret(
        Fr::from_u64(7),
        witness.len(),
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let verifier_setup = HyperKZGVerifierSetup::from(&setup);
    let proof = prove_spartan(&key_digest, &r1cs, &public, &witness, &setup).expect("prove");
    verify_spartan(&key_digest, &r1cs, &public, &proof, &verifier_setup).expect("verify");

    let wrong_key_digest = [43; 32];
    assert!(verify_spartan(&wrong_key_digest, &r1cs, &public, &proof, &verifier_setup).is_err());
    let mut other_r1cs = r1cs.clone();
    for row in other_r1cs.a.iter_mut().chain(&mut other_r1cs.c) {
        for (_, coefficient) in row {
            *coefficient *= Fr::from_u64(2);
        }
    }
    let other_proof = prove_spartan(&key_digest, &other_r1cs, &public, &witness, &setup)
        .expect("prove other key");
    assert!(verify_spartan(&key_digest, &r1cs, &public, &other_proof, &verifier_setup).is_err());

    let mut unsatisfied = witness.clone();
    unsatisfied[0] += Fr::from_u64(1);
    assert!(prove_spartan(&key_digest, &r1cs, &public, &unsatisfied, &setup).is_err());

    let mut wrong_public = public.clone();
    wrong_public[0] += Fr::from_u64(1);
    assert!(verify_spartan(&key_digest, &r1cs, &wrong_public, &proof, &verifier_setup).is_err());

    let mut inner_round_tamper = proof.clone();
    let round = &mut inner_round_tamper.stages[1]
        .round_polynomials
        .round_polynomials[0];
    let mut coefficients = round.coeffs_except_linear_term().to_vec();
    coefficients[0] += Fr::from_u64(1);
    *round = CompressedPoly::new(coefficients);
    assert!(verify_spartan(
        &key_digest,
        &r1cs,
        &public,
        &inner_round_tamper,
        &verifier_setup
    )
    .is_err());

    let mut round_tamper = proof.clone();
    let round = &mut round_tamper.stages[0].round_polynomials.round_polynomials[0];
    let mut coefficients = round.coeffs_except_linear_term().to_vec();
    coefficients[0] += Fr::from_u64(1);
    *round = CompressedPoly::new(coefficients);
    assert!(verify_spartan(&key_digest, &r1cs, &public, &round_tamper, &verifier_setup).is_err());

    let mut witness_eval_tamper = proof.clone();
    witness_eval_tamper.reduced_claims[3] += Fr::from_u64(1);
    assert!(verify_spartan(
        &key_digest,
        &r1cs,
        &public,
        &witness_eval_tamper,
        &verifier_setup
    )
    .is_err());

    let mut commitment_tamper = proof.clone();
    commitment_tamper.commitments[0] = Commitment::new(proof.opening.com[0]);
    assert!(verify_spartan(
        &key_digest,
        &r1cs,
        &public,
        &commitment_tamper,
        &verifier_setup
    )
    .is_err());

    let mut claim_tamper = proof.clone();
    claim_tamper.reduced_claims[0] += Fr::from_u64(1);
    assert!(verify_spartan(&key_digest, &r1cs, &public, &claim_tamper, &verifier_setup).is_err());

    let mut stage_claim_tamper = proof.clone();
    stage_claim_tamper.stage_claims[1][0] += Fr::from_u64(1);
    assert!(verify_spartan(
        &key_digest,
        &r1cs,
        &public,
        &stage_claim_tamper,
        &verifier_setup
    )
    .is_err());

    let mut opening_tamper = proof.clone();
    opening_tamper.opening.v[0][0] += Fr::from_u64(1);
    assert!(verify_spartan(
        &key_digest,
        &r1cs,
        &public,
        &opening_tamper,
        &verifier_setup
    )
    .is_err());

    let encoded = encode_to_vec(&proof, standard()).expect("serialize proof");
    assert_eq!(proof.payload_bytes(), 3_680);
    assert_eq!(encoded.len(), 3_731);
    assert_eq!(encoded.len(), proof.bincode_bytes());
}
