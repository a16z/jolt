#![expect(clippy::expect_used, reason = "integration tests may fail by panic")]

use jolt_field::{CanonicalBytes, FixedByteSize, Fq, Fr, FromPrimitiveInt};
use jolt_r1cs::{FqVar, LinearCombination, R1csBuilder, Variable};
use jolt_sumcheck::{
    allocate_sumcheck_r1cs_layout, append_sumcheck_r1cs_constraints, SumcheckR1csRound,
    SumcheckR1csRoundLayout, SumcheckStatement,
};
use jolt_transcript::r1cs::{
    PoseidonR1csTranscript, R1csJoltByteTranscript, R1csJoltTranscript, R1csTranscript,
};
use jolt_wrapper::{verify_r1cs_witness, WrapperProtocol, WrapperProtocolBuilder};

#[derive(Clone, Debug)]
struct ComposedProtocol {
    protocol: WrapperProtocol<Fr>,
    tamper_targets: Vec<(&'static str, usize)>,
}

#[derive(Clone, Debug)]
struct VariableChallengeRound {
    degree: usize,
    challenge: LinearCombination<Fr>,
}

impl SumcheckR1csRound<Fr> for VariableChallengeRound {
    fn degree(&self) -> usize {
        self.degree
    }

    fn challenge(&self) -> LinearCombination<Fr> {
        self.challenge.clone()
    }
}

#[test]
fn composed_wrapper_transcript_nonnative_and_sumcheck_constraints_accept() {
    let composed = build_composed_protocol();

    verify_r1cs_witness(&composed.protocol).expect("composed R1CS witness should verify");
}

#[test]
fn composed_wrapper_transcript_nonnative_and_sumcheck_constraints_reject_tampering() {
    let composed = build_composed_protocol();
    verify_r1cs_witness(&composed.protocol).expect("baseline witness should verify");

    for (label, variable_index) in composed.tamper_targets {
        let mut tampered = composed.protocol.clone();
        tampered.witness[variable_index] += Fr::from_u64(1);
        assert!(
            verify_r1cs_witness(&tampered).is_err(),
            "{label} variable {variable_index} accepted after tampering"
        );
    }
}

fn build_composed_protocol() -> ComposedProtocol {
    let mut protocol =
        WrapperProtocolBuilder::<Fr, PoseidonR1csTranscript>::new(b"R1csComposition");

    let public_scalar = protocol.alloc_public_scalar(Fr::from_u64(42));
    let witness_scalar = protocol.alloc_witness_scalar(Fr::from_u64(99));
    let bytes = [0x72, 0x31, 0x63, 0x73].map(|byte| protocol.alloc_witness_byte(byte));
    let mut tamper_targets = vec![
        ("public scalar", variable(&public_scalar).index()),
        ("witness scalar", variable(&witness_scalar).index()),
        ("byte witness", variable(&bytes[0]).index()),
    ];

    protocol
        .transcript
        .append_scalar(&mut protocol.builder, b"public", public_scalar.clone());
    protocol
        .transcript
        .append_scalar(&mut protocol.builder, b"witness", witness_scalar);
    protocol
        .transcript
        .append_bytes(&mut protocol.builder, b"bytes", &bytes);
    let challenge = protocol.transcript.challenge_scalar(&mut protocol.builder);
    tamper_targets.push(("transcript challenge", variable(&challenge).index()));

    append_nonnative_relation(&mut protocol.builder, &challenge, &mut tamper_targets);
    append_sumcheck_relation(&mut protocol.builder, &challenge, &mut tamper_targets);

    ComposedProtocol {
        protocol: protocol.finish().expect("all witness variables assigned"),
        tamper_targets,
    }
}

fn append_nonnative_relation(
    builder: &mut R1csBuilder<Fr>,
    challenge: &jolt_r1cs::AssignedScalar<Fr>,
    tamper_targets: &mut Vec<(&'static str, usize)>,
) {
    let fq_challenge = FqVar::from_fr(builder, challenge);
    let seed = FqVar::alloc(builder, Fq::from_u64(17));
    let scale = FqVar::alloc(builder, Fq::from_u64(23));
    let offset = FqVar::alloc(builder, Fq::from_u64(11));

    let computed = fq_challenge
        .add(builder, &seed)
        .mul(builder, &scale)
        .sub(builder, &offset);
    let expected_value =
        (fr_to_fq(challenge.value) + Fq::from_u64(17)) * Fq::from_u64(23) - Fq::from_u64(11);
    let expected = FqVar::alloc(builder, expected_value);

    computed.assert_equal(builder, &expected);
    tamper_targets.push((
        "non-native input limb",
        variable(&fq_challenge.limbs()[0]).index(),
    ));
    tamper_targets.push((
        "non-native output limb",
        variable(&computed.limbs()[0]).index(),
    ));
    tamper_targets.push((
        "non-native expected limb",
        variable(&expected.limbs()[0]).index(),
    ));
}

fn append_sumcheck_relation(
    builder: &mut R1csBuilder<Fr>,
    challenge: &jolt_r1cs::AssignedScalar<Fr>,
    tamper_targets: &mut Vec<(&'static str, usize)>,
) {
    let statement = SumcheckStatement::new(1, 2);
    let rounds = [VariableChallengeRound {
        degree: 2,
        challenge: challenge.lc.clone(),
    }];

    let layout = allocate_sumcheck_r1cs_layout(builder, statement, &rounds)
        .expect("sumcheck layout should allocate");
    tamper_targets.push(("sumcheck input claim", layout.input_claim.index()));
    tamper_targets.push((
        "sumcheck coefficient",
        layout.rounds[0].coefficients[1].index(),
    ));
    tamper_targets.push(("sumcheck output claim", layout.output_claim.index()));
    assign_sumcheck_witness(
        builder,
        &layout.rounds[0],
        layout.input_claim,
        challenge.value,
    );
    append_sumcheck_r1cs_constraints(builder, statement, &rounds, &layout)
        .expect("sumcheck constraints should build");
}

fn assign_sumcheck_witness(
    builder: &mut R1csBuilder<Fr>,
    round: &SumcheckR1csRoundLayout,
    input_claim: Variable,
    challenge: Fr,
) {
    let c0 = Fr::from_u64(3);
    let c1 = Fr::from_u64(4);
    let c2 = Fr::from_u64(5);
    let output_claim = c0 + c1 * challenge + c2 * challenge * challenge;

    assign(builder, input_claim, c0 + (c0 + c1 + c2));
    assign(builder, round.coefficients[0], c0);
    assign(builder, round.coefficients[1], c1);
    assign(builder, round.coefficients[2], c2);
    assign(builder, round.claim_out, output_claim);
}

fn assign(builder: &mut R1csBuilder<Fr>, variable: Variable, value: Fr) {
    builder
        .assign(variable, value)
        .expect("sumcheck witness variable should be assignable");
}

fn fr_to_fq(value: Fr) -> Fq {
    let mut bytes = [0u8; Fr::NUM_BYTES];
    value.to_bytes_le(&mut bytes);
    Fq::from_le_bytes_mod_order(&bytes)
}

fn variable(value: &jolt_r1cs::AssignedScalar<Fr>) -> Variable {
    assert_eq!(value.lc.terms.len(), 1);
    let (variable, coefficient) = value
        .lc
        .terms
        .first()
        .copied()
        .expect("assigned scalar should be backed by one variable");
    assert_eq!(coefficient, Fr::from_u64(1));
    variable
}
