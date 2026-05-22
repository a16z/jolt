#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "tests may panic on assertion failures"
)]

use jolt_field::{Field, Fr, FromPrimitiveInt};
use jolt_r1cs::{AssignedScalar, LinearCombination, Variable};
use jolt_transcript::r1cs::{
    PoseidonR1csTranscript, R1csJoltByteTranscript, R1csJoltTranscript, R1csTranscript,
};
use jolt_wrapper::{verify_r1cs_witness, Error, WrapperProtocol, WrapperProtocolBuilder};

type Transcript = PoseidonR1csTranscript;

#[derive(Clone, Debug, PartialEq, Eq)]
struct FixtureLayout {
    scalar: Variable,
    bytes: Vec<Variable>,
    challenge: Variable,
}

#[test]
fn fixture_protocol_witness_satisfies_r1cs() {
    let (protocol, _) = build_protocol();

    verify_r1cs_witness(&protocol).expect("valid wrapper fixture witness satisfies R1CS");
}

#[test]
fn fixture_protocol_rejects_tampered_scalar() {
    let (mut protocol, layout) = build_protocol();
    protocol.witness[layout.scalar.index()] += Fr::from_u64(1);

    assert!(matches!(
        verify_r1cs_witness(&protocol),
        Err(Error::UnsatisfiedConstraint { .. })
    ));
}

#[test]
fn fixture_protocol_rejects_tampered_challenge() {
    let (mut protocol, layout) = build_protocol();
    protocol.witness[layout.challenge.index()] += Fr::from_u64(1);

    assert!(matches!(
        verify_r1cs_witness(&protocol),
        Err(Error::UnsatisfiedConstraint { .. })
    ));
}

#[test]
fn fixture_protocol_rejects_tampered_public_input() {
    let (mut protocol, _) = build_protocol();
    protocol.public_inputs[0] += Fr::from_u64(1);

    assert!(matches!(
        verify_r1cs_witness(&protocol),
        Err(Error::PublicInputMismatch { .. })
    ));
}

#[test]
fn fixture_protocol_rejects_tampered_byte() {
    let (mut protocol, layout) = build_protocol();
    protocol.witness[layout.bytes[0].index()] += Fr::from_u64(1);

    assert!(matches!(
        verify_r1cs_witness(&protocol),
        Err(Error::UnsatisfiedConstraint { .. })
    ));
}

fn build_protocol() -> (WrapperProtocol<Fr>, FixtureLayout) {
    let scalar_value = Fr::from_u64(17);
    let byte_values = vec![3, 1, 4, 1, 5, 9];
    let public_output = fixture_public_output(scalar_value, &byte_values);
    let mut builder = WrapperProtocolBuilder::<Fr, Transcript>::new(b"JoltWrapperFixture");

    let scalar = builder.alloc_witness_scalar(scalar_value);
    let scalar_variable = single_variable(&scalar);
    let bytes = byte_values
        .iter()
        .map(|&byte| builder.alloc_witness_byte(byte))
        .collect::<Vec<_>>();
    let byte_variables = bytes.iter().map(single_variable).collect::<Vec<_>>();

    builder
        .transcript
        .append_bytes(&mut builder.builder, b"fixture_bytes", &bytes);
    builder
        .transcript
        .append_scalar(&mut builder.builder, b"fixture_scalar", scalar.clone());
    let challenge = builder.transcript.challenge_scalar(&mut builder.builder);
    let challenge_variable = single_variable(&challenge);
    let output = builder.alloc_public_scalar(public_output);
    let byte_sum = byte_sum(&bytes);
    let product = builder
        .builder
        .multiply(challenge.lc.clone(), scalar.lc.clone());
    builder
        .builder
        .assert_equal(product + byte_sum.lc, output.lc.clone());

    let protocol = builder.finish().expect("fixture protocol builds");
    let layout = FixtureLayout {
        scalar: scalar_variable,
        bytes: byte_variables,
        challenge: challenge_variable,
    };
    (protocol, layout)
}

fn fixture_public_output(scalar: Fr, bytes: &[u8]) -> Fr {
    let mut probe = WrapperProtocolBuilder::<Fr, Transcript>::new(b"JoltWrapperFixture");
    let scalar = probe.alloc_witness_scalar(scalar);
    let bytes = bytes
        .iter()
        .map(|&byte| probe.alloc_witness_byte(byte))
        .collect::<Vec<_>>();
    let byte_sum = byte_sum(&bytes);
    probe
        .transcript
        .append_bytes(&mut probe.builder, b"fixture_bytes", &bytes);
    probe
        .transcript
        .append_scalar(&mut probe.builder, b"fixture_scalar", scalar.clone());
    let challenge = probe.transcript.challenge_scalar(&mut probe.builder);
    challenge.value * scalar.value + byte_sum.value
}

fn byte_sum<F: Field>(bytes: &[AssignedScalar<F>]) -> AssignedScalar<F> {
    let mut value = F::zero();
    let mut lc = LinearCombination::zero();
    for byte in bytes {
        value += byte.value;
        lc = lc + byte.lc.clone();
    }
    AssignedScalar::new(value, lc)
}

fn single_variable(scalar: &AssignedScalar<Fr>) -> Variable {
    let [(variable, coefficient)] = scalar.lc.terms.as_slice() else {
        panic!("fixture scalar should be backed by one variable");
    };
    assert_eq!(*coefficient, Fr::from_u64(1));
    *variable
}
