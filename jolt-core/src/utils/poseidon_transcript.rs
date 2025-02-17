use std::marker::PhantomData;

use super::transcript::Transcript;
use crate::field::JoltField;
use ark_crypto_primitives::sponge::{
    poseidon::{get_poseidon_parameters, PoseidonDefaultConfigEntry, PoseidonSponge},
    Absorb, CryptographicSponge, DuplexSpongeMode, FieldBasedCryptographicSponge,
};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{AdditiveGroup, BigInteger, PrimeField};
use ark_serialize::CanonicalSerialize;
use num_bigint::BigUint;

/// Represents the current state of the protocol's Fiat-Shamir transcript.
#[derive(Clone)]
pub struct PoseidonTranscript<J: PrimeField, K: PrimeField> {
    pub state: PoseidonSponge<K>,
    /// We append an ordinal to each invocation of the hash
    n_rounds: u32,
    #[cfg(test)]
    /// A complete history of the transcript's `state`; used for testing.
    state_history: Vec<K>,
    #[cfg(test)]
    /// For a proof to be valid, the verifier's `state_history` should always match
    /// the prover's. In testing, the Jolt verifier may be provided the prover's
    /// `state_history` so that we can detect any deviations and the backtrace can
    /// tell us where it happened.
    expected_state_history: Option<Vec<K>>,
    _marker_j: PhantomData<J>,
}

impl<J: PrimeField, K: PrimeField> PoseidonTranscript<J, K> {
    /// Gives the hasher object with the running seed and index added
    /// To load hash you must call finalize, after appending u8 vectors
    pub fn new() -> PoseidonSponge<K> {
        let parameters =
            get_poseidon_parameters::<K>(4, PoseidonDefaultConfigEntry::new(4, 5, 8, 56, 0))
                .unwrap();
        let state = vec![K::zero(); parameters.rate + parameters.capacity];
        let mode = DuplexSpongeMode::Absorbing {
            next_absorb_index: 0,
        };

        PoseidonSponge {
            parameters: parameters.clone(),
            state,
            mode,
        }
    }

    fn absorb(&mut self, input: &impl Absorb) {
        let elems = input.to_sponge_field_elements_as_vec::<K>();
        if elems.is_empty() {
            return;
        }

        match self.state.mode {
            DuplexSpongeMode::Absorbing { next_absorb_index } => {
                let mut absorb_index = next_absorb_index;
                if absorb_index == self.state.parameters.rate {
                    self.state.permute();
                    absorb_index = 0;
                }
                self.state.absorb_internal(absorb_index, elems.as_slice());
            }
            DuplexSpongeMode::Squeezing {
                next_squeeze_index: _,
            } => {
                panic!("DupleSpongeMode can't be Squeezing in case of absorb")
            }
        };
    }

    fn squeeze_bytes(&mut self, num_bytes: usize) -> Vec<u8> {
        let usable_bytes = ((K::MODULUS_BIT_SIZE - 1) / 8) as usize;

        let num_elements = (num_bytes + usable_bytes - 1) / usable_bytes;
        let src_elements = self.state.squeeze_native_field_elements(num_elements);

        let mut bytes: Vec<u8> = Vec::with_capacity(usable_bytes * num_elements);
        for elem in &src_elements {
            let elem_bytes = elem.into_bigint().to_bytes_le();
            bytes.extend_from_slice(&elem_bytes[..usable_bytes]);
        }

        bytes.truncate(num_bytes);
        bytes
    }

    fn squeeze_bits(&mut self, num_bits: usize) -> Vec<bool> {
        let usable_bits = (K::MODULUS_BIT_SIZE - 1) as usize;

        let num_elements = (num_bits + usable_bits - 1) / usable_bits;
        let src_elements = self.state.squeeze_native_field_elements(num_elements);

        let mut bits: Vec<bool> = Vec::with_capacity(usable_bits * num_elements);
        for elem in &src_elements {
            let elem_bits = elem.into_bigint().to_bits_le();
            bits.extend_from_slice(&elem_bits[..usable_bits]);
        }

        bits.truncate(num_bits);
        bits
    }

    fn squeeze_field_element(&mut self) -> K {
        self.state.squeeze_native_field_elements(1)[0]
    }

    fn update_state(&mut self, new_state: K) {
        self.state.state =
            vec![K::zero(); self.state.parameters.rate + self.state.parameters.capacity];
        self.state.state[self.state.parameters.capacity] = new_state;
        self.state.mode = DuplexSpongeMode::Absorbing {
            next_absorb_index: 1,
        };
        self.n_rounds += 1;
        #[cfg(test)]
        {
            if let Some(expected_state_history) = &self.expected_state_history {
                assert!(
                    new_state == expected_state_history[self.n_rounds as usize],
                    "Fiat-Shamir transcript mismatch"
                );
            }
            self.state_history.push(new_state);
        }
    }
}

//TODO:- Convert label into scalar element
impl<J: PrimeField, K: PrimeField> Transcript for PoseidonTranscript<J, K> {
    fn new(label: &'static [u8]) -> Self {
        let mut hasher = Self::new();
        hasher.absorb(&label);
        let new_state = hasher.squeeze_native_field_elements(1)[0];
        hasher.state = vec![K::zero(); hasher.parameters.rate + hasher.parameters.capacity];
        hasher.state[hasher.parameters.capacity] = new_state;
        hasher.mode = DuplexSpongeMode::Absorbing {
            next_absorb_index: 1,
        };
        Self {
            state: hasher.clone(),
            n_rounds: 0,
            #[cfg(test)]
            state_history: vec![new_state],
            #[cfg(test)]
            expected_state_history: None,
            _marker_j: PhantomData,
        }
    }

    #[cfg(test)]
    fn compare_to(&mut self, other: Self) {
        self.expected_state_history = Some(other.state_history);
    }

    //TODO:-
    fn append_message(&mut self, _msg: &'static [u8]) {
        panic!("In circom code we don't append messages")
        // assert!(msg.len() < 32);
        // let scalar = <ark_bn254::Fq as ark_ff::PrimeField>::from_le_bytes_mod_order(&msg);
        // let n_rounds = self.n_rounds;
        // self.absorb(&n_rounds);
        // self.absorb(&scalar);
        // let new_state = self.squeeze_field_element();
        // self.update_state(new_state);
    }

    //TODO:- Convert bytes into scalar
    fn append_bytes(&mut self, bytes: &[u8]) {
        let n_rounds = self.n_rounds;
        self.absorb(&n_rounds);
        self.absorb(&bytes);
        let new_state = self.squeeze_field_element();
        self.update_state(new_state);
    }

    fn append_u64(&mut self, x: u64) {
        let n_rounds = self.n_rounds;
        self.absorb(&n_rounds);
        self.absorb(&x);
        let new_state = self.squeeze_field_element();
        self.update_state(new_state);
    }

    fn append_scalar<F: JoltField>(&mut self, scalar: &F) {
        if J::MODULUS.to_string() == K::MODULUS.to_string() {
            let mut buf = vec![];
            scalar.serialize_uncompressed(&mut buf).unwrap();
            let wrapped_scalar = ark_bn254::Fr::from_le_bytes_mod_order(&buf);

            let to_absorb = [
                ark_bn254::Fr::from_bigint(self.n_rounds.into()).unwrap(),
                wrapped_scalar,
            ]
            .to_vec();
            self.absorb(&to_absorb);
        } else if J::MODULUS.to_string() < K::MODULUS.to_string() {
            let mut buf = vec![];
            scalar.serialize_uncompressed(&mut buf).unwrap();
            let wrapped_scalar = ark_bn254::Fq::from_le_bytes_mod_order(&buf);

            let to_absorb = [
                ark_bn254::Fq::from_bigint(self.n_rounds.into()).unwrap(),
                wrapped_scalar,
            ]
            .to_vec();
            self.absorb(&to_absorb);
        } else if J::MODULUS.to_string() > K::MODULUS.to_string() {
            let mut buf = vec![];
            scalar.serialize_uncompressed(&mut buf).unwrap();
            let wrapped_scalar_lo = ark_grumpkin::Fq::from_le_bytes_mod_order(&buf[0..16]);
            let wrapped_scalar_hi = ark_grumpkin::Fq::from_le_bytes_mod_order(&buf[16..32]);

            let to_absorb = [
                ark_grumpkin::Fq::from_bigint(self.n_rounds.into()).unwrap(),
                wrapped_scalar_lo,
                wrapped_scalar_hi,
            ]
            .to_vec();
            self.absorb(&to_absorb);
        }
        let new_state = self.squeeze_field_element();
        self.update_state(new_state);
    }

    fn append_scalars<F: JoltField>(&mut self, scalars: &[F]) {
        // self.append_message(b"begin_append_vector");
        for item in scalars.iter() {
            self.append_scalar(item);
        }
        // self.append_message(b"end_append_vector");
    }

    fn append_point<G: CurveGroup>(&mut self, point: &G) {
        if J::MODULUS.to_string() == K::MODULUS.to_string() {
            if point.is_zero() {
                let to_absorb = [
                    <ark_bn254::Fr as ark_ff::PrimeField>::from_bigint(self.n_rounds.into())
                        .unwrap(),
                    <ark_bn254::Fr as ark_ff::Zero>::zero(),
                    <ark_bn254::Fr as ark_ff::Zero>::zero(),
                    <ark_bn254::Fr as ark_ff::Zero>::zero(),
                    <ark_bn254::Fr as ark_ff::Zero>::zero(),
                    <ark_bn254::Fr as ark_ff::Zero>::zero(),
                    <ark_bn254::Fr as ark_ff::Zero>::zero(),
                ]
                .to_vec();
                self.absorb(&to_absorb);

                let new_state = self.squeeze_field_element();
                self.update_state(new_state);
                return;
            }
            // If we add the point at infinity then we hash over a region of zeros
            let aff = point.into_affine();

            let x = aff.x().unwrap();
            let y = aff.y().unwrap();
            let mut x_bytes = vec![];
            let mut y_bytes = vec![];
            x.serialize_compressed(&mut x_bytes).unwrap();
            y.serialize_compressed(&mut y_bytes).unwrap();

            let x_limbs = three_limb_repr(&x_bytes);
            let y_limbs = three_limb_repr(&x_bytes);
            let mut to_absorb = vec![<ark_bn254::Fr>::from_bigint(self.n_rounds.into()).unwrap()];
            let limbs: Vec<_> = x_limbs.iter().chain(y_limbs.iter()).collect();
            to_absorb.extend(limbs);
            self.absorb(&to_absorb);
        } else if J::MODULUS.to_string() < K::MODULUS.to_string() {
            if point.is_zero() {
                let to_absorb = [
                    <ark_bn254::Fq as ark_ff::PrimeField>::from_bigint(self.n_rounds.into())
                        .unwrap(),
                    <ark_bn254::Fq as ark_ff::Zero>::zero(),
                    <ark_bn254::Fq as ark_ff::Zero>::zero(),
                ]
                .to_vec();
                self.absorb(&to_absorb);

                let new_state = self.squeeze_field_element();
                self.update_state(new_state);
                return;
            }
            // If we add the point at infinity then we hash over a region of zeros
            let aff = point.into_affine();
            let mut x_bytes = vec![];
            let mut y_bytes = vec![];
            let x = aff.x().unwrap();
            x.serialize_compressed(&mut x_bytes).unwrap();
            let y = aff.y().unwrap();
            y.serialize_compressed(&mut y_bytes).unwrap();
            let to_absorb = [
                <ark_bn254::Fq as ark_ff::PrimeField>::from_bigint(self.n_rounds.into()).unwrap(),
                <ark_bn254::Fq as ark_ff::PrimeField>::from_le_bytes_mod_order(&x_bytes),
                <ark_bn254::Fq as ark_ff::PrimeField>::from_le_bytes_mod_order(&y_bytes),
            ]
            .to_vec();
            self.absorb(&to_absorb);
        } else if J::MODULUS.to_string() > K::MODULUS.to_string() {
            if point.is_zero() {
                let to_absorb = [
                    <ark_grumpkin::Fq as ark_ff::PrimeField>::from_bigint(self.n_rounds.into())
                        .unwrap(),
                    <ark_grumpkin::Fq as ark_ff::Zero>::zero(),
                    <ark_grumpkin::Fq as ark_ff::Zero>::zero(),
                ]
                .to_vec();
                self.absorb(&to_absorb);

                let new_state = self.squeeze_field_element();
                self.update_state(new_state);
                return;
            }
            // If we add the point at infinity then we hash over a region of zeros
            let aff = point.into_affine();
            let mut x_bytes = vec![];
            let mut y_bytes = vec![];
            let x = aff.x().unwrap();
            x.serialize_compressed(&mut x_bytes).unwrap();
            let y = aff.y().unwrap();
            y.serialize_compressed(&mut y_bytes).unwrap();
            let to_absorb = [
                <ark_grumpkin::Fq as ark_ff::PrimeField>::from_bigint(self.n_rounds.into())
                    .unwrap(),
                <ark_grumpkin::Fq as ark_ff::PrimeField>::from_le_bytes_mod_order(&x_bytes),
                <ark_grumpkin::Fq as ark_ff::PrimeField>::from_le_bytes_mod_order(&y_bytes),
            ]
            .to_vec();
            self.absorb(&to_absorb);
        }
        let new_state = self.squeeze_field_element();

        self.update_state(new_state);
    }

    fn append_points<G: CurveGroup>(&mut self, points: &[G]) {
        // self.append_message(b"begin_append_vector");
        for item in points.iter() {
            self.append_point(item);
        }
        // self.append_message(b"end_append_vector");
    }

    fn challenge_scalar<F: JoltField>(&mut self) -> F {
        let n_rounds = self.n_rounds;
        self.absorb(&n_rounds);
        let new_state = self.squeeze_field_element();
        self.update_state(new_state);
        F::from_bytes(&new_state.into_bigint().to_bytes_le())
    }

    fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        (0..len)
            .map(|_| self.challenge_scalar())
            .collect::<Vec<F>>()
    }

    fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let q: F = self.challenge_scalar();
        let mut q_powers = vec![F::one(); len];
        for i in 1..len {
            q_powers[i] = q_powers[i - 1] * q;
        }
        q_powers
    }
}

fn three_limb_repr(bytes: &Vec<u8>) -> Vec<ark_bn254::Fr> {
    let mut limbs = [ark_bn254::Fr::ZERO; 3];
    let elem = <ark_bn254::Fq as ark_ff::PrimeField>::from_le_bytes_mod_order(&bytes);

    let mask = BigUint::from((1u128 << 125) - 1);
    limbs[0] = <ark_bn254::Fr as ark_ff::PrimeField>::from_le_bytes_mod_order(
        &(BigUint::from(elem.into_bigint()) & mask.clone()).to_bytes_le(),
    );
    limbs[1] = <ark_bn254::Fr as ark_ff::PrimeField>::from_le_bytes_mod_order(
        &BigUint::from(BigUint::from(elem.into_bigint()) >> 125 & mask.clone()).to_bytes_le(),
    );
    limbs[2] = <ark_bn254::Fr as ark_ff::PrimeField>::from_le_bytes_mod_order(
        &BigUint::from((BigUint::from(elem.into_bigint()) >> 250) & mask.clone()).to_bytes_le(),
    );

    limbs.to_vec()
}
