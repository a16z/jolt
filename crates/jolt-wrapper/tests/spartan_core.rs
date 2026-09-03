#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    reason = "test fixtures fail immediately on setup or proof errors"
)]

use bincode::config::standard;
use bincode::serde::encode_to_vec;
use jolt_crypto::Bn254;
use jolt_field::{Fr, Ring};
use jolt_hyperkzg::{HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_poly::{CompressedPoly, MultilinearPoly};
use jolt_r1cs::ConstraintMatrices;
use jolt_transcript::{Blake3Transcript, Transcript};
use jolt_wrapper::hash_table::{Decoder, Event, RecordingTranscript};
use jolt_wrapper::spartan::{
    prove_spartan, verify_spartan, ChallengeDecoder, PublicChallenge, SharedWitnessColumn,
    SpartanPublicInputStatement, SpartanPublicInputs,
};
use jolt_wrapper::stream::{Column, Commitment, PackingLayout};

fn instance(log_size: usize, public: Vec<Fr>) -> (ConstraintMatrices<Fr>, Vec<Fr>, Vec<Fr>) {
    let size = 1usize << log_size;
    let public_count = public.len();
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

fn recorded_challenges() -> Vec<PublicChallenge> {
    drop(RecordingTranscript::<Blake3Transcript>::take_log());
    let mut transcript = RecordingTranscript::<Blake3Transcript>::new(b"spartan-public-inputs");
    for index in 0..28 {
        transcript.append_bytes(&(index as u64).to_be_bytes());
        if index < 16 {
            let _challenge = transcript.challenge();
        } else {
            let _challenge = transcript.challenge_scalar();
        }
    }
    RecordingTranscript::<Blake3Transcript>::take_log()
        .into_iter()
        .filter_map(|record| match record.event {
            Event::Squeeze { decoder, value } => Some(PublicChallenge {
                value,
                decoder: match decoder {
                    Decoder::Challenge125 => ChallengeDecoder::Challenge125,
                    Decoder::Scalar128 => ChallengeDecoder::Scalar128,
                },
            }),
            Event::Start { .. } | Event::Append { .. } => None,
        })
        .collect()
}

#[test]
fn shared_witness_uses_the_common_row_point_prefix() {
    let witness: Vec<Fr> = (0..6_760)
        .map(|index| Fr::from_u64(index as u64 + 1))
        .collect();
    let common_rows = 1 << 18;
    let shared = SharedWitnessColumn::new(&witness, common_rows).expect("embed witness");
    assert_eq!(
        shared.inner_member(),
        jolt_wrapper::stream::StageMemberSpec {
            rounds: 13,
            degree: 2,
            offset: 0,
        }
    );
    let common_point: Vec<Fr> = (0..18)
        .map(|index| Fr::from_u64(index as u64 + 2))
        .collect();
    let inner_point = shared.inner_point(&common_point).expect("inner point");
    let mut padded_witness = witness;
    padded_witness.resize(1 << 13, Fr::from_u64(0));
    let expected = padded_witness.as_slice().evaluate(inner_point);
    let Column::Fr(evaluations) = shared.into_column() else {
        panic!("shared witness must be a field column");
    };
    assert_eq!(evaluations.len(), common_rows);
    assert_eq!(evaluations.as_slice().evaluate(&common_point), expected);

    assert_eq!(
        PackingLayout::new(common_rows, 254, 16)
            .unwrap()
            .group_count,
        16
    );
    assert_eq!(
        PackingLayout::new(common_rows, 255, 16)
            .unwrap()
            .group_count,
        16
    );
}

#[test]
fn spartan_round_trip_and_tampers() {
    let key_digest = [42; 32];
    let challenges = recorded_challenges();
    assert_eq!(challenges.len(), 28);
    let mut public: Vec<Fr> = (0..22)
        .map(|index| Fr::from_u64(index as u64 + 3))
        .collect();
    public.extend(challenges.iter().map(|challenge| challenge.value));
    let (r1cs, public, witness) = instance(12, public);
    let setup = HyperKZGScheme::<Bn254>::setup_from_secret(
        Fr::from_u64(7),
        witness.len(),
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let verifier_setup = HyperKZGVerifierSetup::from(&setup);
    let known = &public[..22];
    let decoders: Vec<ChallengeDecoder> = challenges
        .iter()
        .map(|challenge| challenge.decoder)
        .collect();
    let public_request = SpartanPublicInputs {
        known,
        challenges: &challenges,
    };
    let public_statement = SpartanPublicInputStatement {
        known,
        challenge_decoders: &decoders,
    };
    let proof = prove_spartan(&key_digest, &r1cs, public_request, &witness, &setup).expect("prove");
    assert_eq!(proof.public_challenges.len(), 28);
    verify_spartan(
        &key_digest,
        &r1cs,
        public_statement,
        &proof,
        &verifier_setup,
    )
    .expect("verify");

    let wrong_key_digest = [43; 32];
    assert!(verify_spartan(
        &wrong_key_digest,
        &r1cs,
        public_statement,
        &proof,
        &verifier_setup
    )
    .is_err());
    let mut other_r1cs = r1cs.clone();
    for row in other_r1cs.a.iter_mut().chain(&mut other_r1cs.c) {
        for (_, coefficient) in row {
            *coefficient *= Fr::from_u64(2);
        }
    }
    let other_proof = prove_spartan(&key_digest, &other_r1cs, public_request, &witness, &setup)
        .expect("prove other key");
    assert!(verify_spartan(
        &key_digest,
        &r1cs,
        public_statement,
        &other_proof,
        &verifier_setup
    )
    .is_err());

    let mut unsatisfied = witness.clone();
    unsatisfied[0] += Fr::from_u64(1);
    assert!(prove_spartan(&key_digest, &r1cs, public_request, &unsatisfied, &setup).is_err());

    let mut wrong_public = known.to_vec();
    wrong_public[0] += Fr::from_u64(1);
    let wrong_public_statement = SpartanPublicInputStatement {
        known: &wrong_public,
        challenge_decoders: &decoders,
    };
    assert!(verify_spartan(
        &key_digest,
        &r1cs,
        wrong_public_statement,
        &proof,
        &verifier_setup
    )
    .is_err());

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
        public_statement,
        &inner_round_tamper,
        &verifier_setup
    )
    .is_err());

    let mut round_tamper = proof.clone();
    let round = &mut round_tamper.stages[0].round_polynomials.round_polynomials[0];
    let mut coefficients = round.coeffs_except_linear_term().to_vec();
    coefficients[0] += Fr::from_u64(1);
    *round = CompressedPoly::new(coefficients);
    assert!(verify_spartan(
        &key_digest,
        &r1cs,
        public_statement,
        &round_tamper,
        &verifier_setup
    )
    .is_err());

    let mut witness_eval_tamper = proof.clone();
    witness_eval_tamper.reduced_claims[3] += Fr::from_u64(1);
    assert!(verify_spartan(
        &key_digest,
        &r1cs,
        public_statement,
        &witness_eval_tamper,
        &verifier_setup
    )
    .is_err());

    let mut commitment_tamper = proof.clone();
    commitment_tamper.commitments[0] = Commitment::new(proof.opening.com[0]);
    assert!(verify_spartan(
        &key_digest,
        &r1cs,
        public_statement,
        &commitment_tamper,
        &verifier_setup
    )
    .is_err());

    let mut claim_tamper = proof.clone();
    claim_tamper.reduced_claims[0] += Fr::from_u64(1);
    assert!(verify_spartan(
        &key_digest,
        &r1cs,
        public_statement,
        &claim_tamper,
        &verifier_setup
    )
    .is_err());

    let mut stage_claim_tamper = proof.clone();
    stage_claim_tamper.stage_claims[1][0] += Fr::from_u64(1);
    assert!(verify_spartan(
        &key_digest,
        &r1cs,
        public_statement,
        &stage_claim_tamper,
        &verifier_setup
    )
    .is_err());

    let mut opening_tamper = proof.clone();
    opening_tamper.opening.v[0][0] += Fr::from_u64(1);
    assert!(verify_spartan(
        &key_digest,
        &r1cs,
        public_statement,
        &opening_tamper,
        &verifier_setup
    )
    .is_err());

    let mut packed_challenge_tamper = proof.clone();
    packed_challenge_tamper.public_challenges[0][0] ^= 1;
    assert!(verify_spartan(
        &key_digest,
        &r1cs,
        public_statement,
        &packed_challenge_tamper,
        &verifier_setup
    )
    .is_err());

    for high_bits in 1..8 {
        let mut noncanonical = proof.clone();
        noncanonical.public_challenges[0][15] |= high_bits << 5;
        assert!(verify_spartan(
            &key_digest,
            &r1cs,
            public_statement,
            &noncanonical,
            &verifier_setup
        )
        .is_err());
    }

    let encoded = encode_to_vec(&proof, standard()).expect("serialize proof");
    assert_eq!(proof.payload_bytes(), 3_776);
    assert_eq!(encoded.len(), 3_827);
    assert_eq!(encoded.len(), proof.bincode_bytes());
}
