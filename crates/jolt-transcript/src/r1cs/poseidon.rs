//! Jolt Poseidon proof transcript gadget for BN254.

use std::sync::OnceLock;

use jolt_field::{Fr, FromPrimitiveInt};
use jolt_r1cs::{AssignedScalar, LinearCombination, R1csBuilder};
use light_poseidon::parameters::bn254_x5;

use super::{R1csAlgebraicTranscript, R1csByteTranscript, R1csTranscript};

const POSEIDON_INPUTS: usize = 3;
const POSEIDON_WIDTH: usize = POSEIDON_INPUTS + 1;

#[derive(Clone, Debug)]
struct PoseidonR1csParameters {
    ark: Vec<Fr>,
    mds: Vec<Vec<Fr>>,
    full_rounds: usize,
    partial_rounds: usize,
    width: usize,
}

/// Poseidon Fiat-Shamir transcript encoded as R1CS constraints.
///
/// This mirrors `jolt-core`'s `transcript-poseidon` proof transcript: every
/// raw absorb and challenge hashes `(state, n_rounds, payload)`, scalar payloads
/// are absorbed as BN254 field elements, and challenges are full field elements.
#[derive(Clone, Debug)]
pub struct PoseidonR1csTranscript {
    state: AssignedScalar<Fr>,
    round: u64,
}

impl PoseidonR1csTranscript {
    /// Returns the current assigned transcript state.
    pub fn state(&self) -> &AssignedScalar<Fr> {
        &self.state
    }
}

impl R1csTranscript<Fr> for PoseidonR1csTranscript {
    type Challenge = AssignedScalar<Fr>;

    fn new(builder: &mut R1csBuilder<Fr>, label: &'static [u8]) -> Self {
        let label = label_scalar(label);
        let state = poseidon_permutation(
            builder,
            [
                AssignedScalar::constant(label),
                AssignedScalar::constant(zero()),
                AssignedScalar::constant(zero()),
            ],
        );
        Self { state, round: 0 }
    }

    fn challenge_scalar(&mut self, builder: &mut R1csBuilder<Fr>) -> AssignedScalar<Fr> {
        let challenge = poseidon_permutation(
            builder,
            [
                self.state.clone(),
                AssignedScalar::constant(round_tag(self.round)),
                AssignedScalar::constant(zero()),
            ],
        );
        self.state = challenge.clone();
        self.round += 1;
        challenge
    }
}

impl R1csAlgebraicTranscript<Fr> for PoseidonR1csTranscript {
    fn absorb_scalar(&mut self, builder: &mut R1csBuilder<Fr>, value: AssignedScalar<Fr>) {
        self.state = poseidon_permutation(
            builder,
            [
                self.state.clone(),
                AssignedScalar::constant(round_tag(self.round)),
                value,
            ],
        );
        self.round += 1;
    }

    fn absorb_u64(&mut self, builder: &mut R1csBuilder<Fr>, value: u64) {
        self.absorb_constant_scalar(builder, Fr::from_u64(value));
    }

    fn absorb_label(&mut self, builder: &mut R1csBuilder<Fr>, label: &'static [u8]) {
        self.absorb_constant_scalar(builder, label_scalar(label));
    }

    fn absorb_label_with_len(
        &mut self,
        builder: &mut R1csBuilder<Fr>,
        label: &'static [u8],
        len: u64,
    ) {
        self.absorb_constant_scalar(builder, label_with_len_scalar(label, len));
    }
}

impl R1csByteTranscript<Fr> for PoseidonR1csTranscript {
    type Byte = AssignedScalar<Fr>;

    fn absorb_bytes(&mut self, builder: &mut R1csBuilder<Fr>, bytes: &[Self::Byte]) {
        let mut chunks = bytes.chunks(32);
        let first = chunks
            .next()
            .map_or_else(|| AssignedScalar::constant(zero()), pack_bytes);
        let mut current = poseidon_permutation(
            builder,
            [
                self.state.clone(),
                AssignedScalar::constant(round_tag(self.round)),
                first,
            ],
        );

        for chunk in chunks {
            let chunk = pack_bytes(chunk);
            current =
                poseidon_permutation(builder, [current, AssignedScalar::constant(zero()), chunk]);
        }

        self.state = current;
        self.round += 1;
    }

    fn absorb_constant_bytes(&mut self, builder: &mut R1csBuilder<Fr>, bytes: &'static [u8]) {
        let mut chunks = bytes.chunks(32);
        let first = chunks.next().map_or_else(zero, bytes_scalar);
        let mut current = poseidon_permutation(
            builder,
            [
                self.state.clone(),
                AssignedScalar::constant(round_tag(self.round)),
                AssignedScalar::constant(first),
            ],
        );

        for chunk in chunks {
            current = poseidon_permutation(
                builder,
                [
                    current,
                    AssignedScalar::constant(zero()),
                    AssignedScalar::constant(bytes_scalar(chunk)),
                ],
            );
        }

        self.state = current;
        self.round += 1;
    }
}

fn label_scalar(label: &[u8]) -> Fr {
    assert!(
        label.len() <= 32,
        "label must fit in one Jolt transcript word"
    );
    Fr::from_le_bytes_mod_order(label)
}

fn bytes_scalar(bytes: &[u8]) -> Fr {
    assert!(
        bytes.len() <= 32,
        "Poseidon byte chunks must fit in one BN254 scalar"
    );
    Fr::from_le_bytes_mod_order(bytes)
}

fn label_with_len_scalar(label: &[u8], len: u64) -> Fr {
    assert!(
        label.len() <= 24,
        "label must leave 8 bytes for the Jolt transcript length word"
    );
    let mut packed = [0u8; 32];
    packed[..label.len()].copy_from_slice(label);
    packed[24..32].copy_from_slice(&len.to_be_bytes());
    Fr::from_le_bytes_mod_order(&packed)
}

fn round_tag(round: u64) -> Fr {
    Fr::from_u64(round)
}

fn zero() -> Fr {
    Fr::from_u64(0)
}

fn pack_bytes(bytes: &[AssignedScalar<Fr>]) -> AssignedScalar<Fr> {
    assert!(
        bytes.len() <= 32,
        "Poseidon byte chunks must fit in one BN254 scalar"
    );
    let mut value = zero();
    let mut lc = LinearCombination::zero();
    let mut coefficient = Fr::from_u64(1);
    let radix = Fr::from_u64(256);
    for byte in bytes {
        value += byte.value * coefficient;
        lc = lc + byte.lc.clone().scale(coefficient);
        coefficient *= radix;
    }
    AssignedScalar::new(value, lc)
}

#[cfg(test)]
fn assigned_bytes(builder: &mut R1csBuilder<Fr>, bytes: &[u8]) -> Vec<AssignedScalar<Fr>> {
    bytes
        .iter()
        .map(|byte| AssignedScalar::alloc(builder, Fr::from_u64(u64::from(*byte))))
        .collect()
}

fn poseidon_permutation(
    builder: &mut R1csBuilder<Fr>,
    inputs: [AssignedScalar<Fr>; POSEIDON_INPUTS],
) -> AssignedScalar<Fr> {
    let params = poseidon_parameters();
    let mut state = Vec::with_capacity(params.width);
    state.push(AssignedScalar::constant(zero()));
    state.extend(inputs);

    let all_rounds = params.full_rounds + params.partial_rounds;
    let half_rounds = params.full_rounds / 2;

    for round in 0..half_rounds {
        apply_ark(&mut state, round, params);
        apply_sbox_full(builder, &mut state);
        apply_mds(builder, &mut state, params);
    }

    for round in half_rounds..half_rounds + params.partial_rounds {
        apply_ark(&mut state, round, params);
        apply_sbox_partial(builder, &mut state);
        apply_mds(builder, &mut state, params);
    }

    for round in half_rounds + params.partial_rounds..all_rounds {
        apply_ark(&mut state, round, params);
        apply_sbox_full(builder, &mut state);
        apply_mds(builder, &mut state, params);
    }

    state[0].clone()
}

#[cfg(test)]
fn poseidon_hash(inputs: [Fr; POSEIDON_INPUTS]) -> Fr {
    let params = poseidon_parameters();
    let mut state = Vec::with_capacity(params.width);
    state.push(zero());
    state.extend(inputs);

    let all_rounds = params.full_rounds + params.partial_rounds;
    let half_rounds = params.full_rounds / 2;

    for round in 0..half_rounds {
        apply_ark_values(&mut state, round, params);
        apply_sbox_full_values(&mut state);
        apply_mds_values(&mut state, params);
    }

    for round in half_rounds..half_rounds + params.partial_rounds {
        apply_ark_values(&mut state, round, params);
        state[0] = pow5_value(state[0]);
        apply_mds_values(&mut state, params);
    }

    for round in half_rounds + params.partial_rounds..all_rounds {
        apply_ark_values(&mut state, round, params);
        apply_sbox_full_values(&mut state);
        apply_mds_values(&mut state, params);
    }

    state[0]
}

fn apply_ark(state: &mut [AssignedScalar<Fr>], round: usize, params: &PoseidonR1csParameters) {
    for (index, assigned) in state.iter_mut().enumerate() {
        let constant = params.ark[round * params.width + index];
        assigned.value += constant;
        assigned.lc = assigned.lc.clone() + LinearCombination::constant(constant);
    }
}

#[cfg(test)]
fn apply_ark_values(state: &mut [Fr], round: usize, params: &PoseidonR1csParameters) {
    for (index, value) in state.iter_mut().enumerate() {
        *value += params.ark[round * params.width + index];
    }
}

fn apply_sbox_full(builder: &mut R1csBuilder<Fr>, state: &mut [AssignedScalar<Fr>]) {
    for assigned in state.iter_mut() {
        *assigned = pow5(builder, assigned.clone());
    }
}

#[cfg(test)]
fn apply_sbox_full_values(state: &mut [Fr]) {
    for value in state {
        *value = pow5_value(*value);
    }
}

fn apply_sbox_partial(builder: &mut R1csBuilder<Fr>, state: &mut [AssignedScalar<Fr>]) {
    state[0] = pow5(builder, state[0].clone());
}

fn apply_mds(
    builder: &mut R1csBuilder<Fr>,
    state: &mut Vec<AssignedScalar<Fr>>,
    params: &PoseidonR1csParameters,
) {
    let previous = state.clone();
    state.clear();
    for row in 0..params.width {
        let mut value = zero();
        let mut lc = LinearCombination::zero();
        for (assigned, &coefficient) in previous.iter().zip(&params.mds[row]) {
            value += assigned.value * coefficient;
            lc = lc + assigned.lc.clone().scale(coefficient);
        }
        let output = AssignedScalar::alloc(builder, value);
        builder.assert_equal(output.lc.clone(), lc);
        state.push(output);
    }
}

#[cfg(test)]
fn apply_mds_values(state: &mut Vec<Fr>, params: &PoseidonR1csParameters) {
    let previous = state.clone();
    state.clear();
    for row in 0..params.width {
        let mut value = zero();
        for (input, &coefficient) in previous.iter().zip(&params.mds[row]) {
            value += *input * coefficient;
        }
        state.push(value);
    }
}

fn pow5(builder: &mut R1csBuilder<Fr>, value: AssignedScalar<Fr>) -> AssignedScalar<Fr> {
    let square = multiply(builder, &value, &value);
    let fourth = multiply(builder, &square, &square);
    multiply(builder, &fourth, &value)
}

#[cfg(test)]
fn pow5_value(value: Fr) -> Fr {
    let square = value * value;
    let fourth = square * square;
    fourth * value
}

fn multiply(
    builder: &mut R1csBuilder<Fr>,
    lhs: &AssignedScalar<Fr>,
    rhs: &AssignedScalar<Fr>,
) -> AssignedScalar<Fr> {
    AssignedScalar::new(
        lhs.value * rhs.value,
        builder.multiply(lhs.lc.clone(), rhs.lc.clone()),
    )
}

fn poseidon_parameters() -> &'static PoseidonR1csParameters {
    static PARAMS: OnceLock<PoseidonR1csParameters> = OnceLock::new();
    PARAMS.get_or_init(load_poseidon_parameters)
}

#[expect(
    clippy::expect_used,
    reason = "constant width-4 BN254 Poseidon parameters are generated by light-poseidon"
)]
fn load_poseidon_parameters() -> PoseidonR1csParameters {
    let params = bn254_x5::get_poseidon_parameters::<ark_bn254::Fr>(POSEIDON_WIDTH as u8)
        .expect("valid width-4 BN254 Poseidon parameters");
    PoseidonR1csParameters {
        ark: params.ark.into_iter().map(Fr::from).collect(),
        mds: params
            .mds
            .into_iter()
            .map(|row| row.into_iter().map(Fr::from).collect())
            .collect(),
        full_rounds: params.full_rounds,
        partial_rounds: params.partial_rounds,
        width: params.width,
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, clippy::panic, reason = "tests may fail loudly")]
mod tests {
    use super::*;
    use crate::r1cs::{R1csJoltByteTranscript, R1csJoltTranscript};
    use jolt_r1cs::Variable;
    use light_poseidon::{Poseidon, PoseidonHasher};

    #[derive(Clone, Copy, Debug)]
    struct NativeTranscript {
        state: Fr,
        round: u64,
    }

    impl NativeTranscript {
        fn new(label: &'static [u8]) -> Self {
            Self {
                state: poseidon_hash([label_scalar(label), zero(), zero()]),
                round: 0,
            }
        }

        fn absorb_scalar(&mut self, value: Fr) {
            self.state = poseidon_hash([self.state, round_tag(self.round), value]);
            self.round += 1;
        }

        fn absorb_bytes(&mut self, bytes: &[u8]) {
            let mut chunks = bytes.chunks(32);
            let first = chunks.next().map_or_else(zero, bytes_scalar);
            let mut current = poseidon_hash([self.state, round_tag(self.round), first]);
            for chunk in chunks {
                current = poseidon_hash([current, zero(), bytes_scalar(chunk)]);
            }
            self.state = current;
            self.round += 1;
        }

        fn absorb_label(&mut self, label: &'static [u8]) {
            self.absorb_scalar(label_scalar(label));
        }

        fn absorb_u64(&mut self, value: u64) {
            self.absorb_scalar(Fr::from_u64(value));
        }

        fn challenge_scalar(&mut self) -> Fr {
            self.state = poseidon_hash([self.state, round_tag(self.round), zero()]);
            self.round += 1;
            self.state
        }

        fn append_scalar(&mut self, label: &'static [u8], value: Fr) {
            self.absorb_label(label);
            self.absorb_scalar(value);
        }

        fn append_scalars(&mut self, label: &'static [u8], values: &[Fr]) {
            self.absorb_scalar(label_with_len_scalar(label, values.len() as u64));
            for value in values {
                self.absorb_scalar(*value);
            }
        }

        fn append_bytes(&mut self, label: &'static [u8], bytes: &[u8]) {
            self.absorb_scalar(label_with_len_scalar(label, bytes.len() as u64));
            self.absorb_bytes(bytes);
        }
    }

    #[test]
    fn native_permutation_matches_light_poseidon() {
        let inputs = [Fr::from_u64(7), Fr::from_u64(11), Fr::from_u64(19)];

        assert_eq!(poseidon_hash(inputs), light_poseidon_hash(inputs));
    }

    #[test]
    fn poseidon_gadget_matches_native_permutation() {
        let inputs = [Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
        let mut builder = R1csBuilder::new();
        let assigned_inputs = inputs.map(|input| AssignedScalar::alloc(&mut builder, input));

        let output = poseidon_permutation(&mut builder, assigned_inputs);

        assert_eq!(output.value, poseidon_hash(inputs));
        let witness = builder.witness().expect("witness is assigned");
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());
        assert!((500..=560).contains(&matrices.num_constraints));
    }

    #[test]
    fn jolt_poseidon_transcript_matches_native_sequence() {
        let mut builder = R1csBuilder::new();
        let mut r1cs = PoseidonR1csTranscript::new(&mut builder, b"Jolt");
        let mut native = NativeTranscript::new(b"Jolt");

        r1cs.append_label(&mut builder, b"sumcheck");
        native.absorb_label(b"sumcheck");

        r1cs.append_u64(&mut builder, b"round", 9);
        native.absorb_label(b"round");
        native.absorb_u64(9);

        let scalar = AssignedScalar::alloc(&mut builder, Fr::from_u64(42));
        r1cs.append_scalar(&mut builder, b"sumcheck_claim", scalar);
        native.append_scalar(b"sumcheck_claim", Fr::from_u64(42));

        let scalars = [
            AssignedScalar::alloc(&mut builder, Fr::from_u64(3)),
            AssignedScalar::alloc(&mut builder, Fr::from_u64(5)),
            AssignedScalar::alloc(&mut builder, Fr::from_u64(8)),
        ];
        r1cs.append_scalars(&mut builder, b"sumcheck_poly", &scalars);
        native.append_scalars(
            b"sumcheck_poly",
            &[Fr::from_u64(3), Fr::from_u64(5), Fr::from_u64(8)],
        );

        let byte_payload = [0xabu8; 45];
        let assigned_payload = assigned_bytes(&mut builder, &byte_payload);
        r1cs.append_bytes(&mut builder, b"inputs", &assigned_payload);
        native.append_bytes(b"inputs", &byte_payload);

        r1cs.append_constant_bytes(&mut builder, b"preprocessing_digest", &[0xcdu8; 32]);
        native.append_bytes(b"preprocessing_digest", &[0xcdu8; 32]);

        let first = r1cs.challenge_scalar(&mut builder);
        let native_first = native.challenge_scalar();
        assert_eq!(first.value, native_first);

        r1cs.absorb_constant_scalar(&mut builder, Fr::from_u64(100));
        native.absorb_scalar(Fr::from_u64(100));

        let second = r1cs.challenge_scalar(&mut builder);
        let native_second = native.challenge_scalar();
        assert_eq!(second.value, native_second);
        assert_eq!(r1cs.state().value, native.state);

        let witness = builder.witness().expect("witness is assigned");
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }

    #[test]
    fn tampered_challenge_witness_fails_constraints() {
        let mut builder = R1csBuilder::new();
        let mut transcript = PoseidonR1csTranscript::new(&mut builder, b"Jolt");
        transcript.absorb_constant_scalar(&mut builder, Fr::from_u64(42));
        let challenge = transcript.challenge_scalar(&mut builder);

        let mut witness = builder.witness().expect("witness is assigned");
        let variable = challenge_variable(&challenge);
        witness[variable.index()] += Fr::from_u64(1);

        assert!(builder.into_matrices().check_witness(&witness).is_err());
    }

    fn light_poseidon_hash(inputs: [Fr; POSEIDON_INPUTS]) -> Fr {
        let mut poseidon =
            Poseidon::<ark_bn254::Fr>::new_circom(POSEIDON_INPUTS).expect("Poseidon init");
        let inputs = inputs.map(ark_bn254::Fr::from);
        poseidon.hash(&inputs).expect("Poseidon hash").into()
    }

    fn challenge_variable(challenge: &AssignedScalar<Fr>) -> Variable {
        let [(variable, coefficient)] = challenge.lc.terms.as_slice() else {
            panic!("challenge should be represented by one allocated variable");
        };
        assert_eq!(*coefficient, Fr::from_u64(1));
        *variable
    }
}
