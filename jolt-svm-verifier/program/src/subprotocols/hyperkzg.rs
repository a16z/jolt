use ark_ec::pairing::Pairing;
use ark_ec::AffineRepr;
use ark_ff::{BigInteger, PrimeField};
use ark_serialize::CanonicalSerialize;
use ark_std::{One, Zero};
use jolt_types::field;
use jolt_types::poly::commitment::hyperkzg::{
    HyperKZGCommitment, HyperKZGProof, HyperKZGVerifierKey,
};
use jolt_types::utils::errors::ProofVerifyError;
use jolt_types::utils::transcript::ProofTranscript;

use crate::utils::convert_endianness_64;
use solana_program::alt_bn128::prelude::{
    alt_bn128_addition, alt_bn128_multiplication, alt_bn128_pairing,
};

/// A method to verify purported evaluations of a batch of polynomials
pub fn verify_hyperkzg<P: Pairing>(
    vk: &HyperKZGVerifierKey<P>,
    C: &HyperKZGCommitment<P>,
    point: &[P::ScalarField],
    P_of_x: &P::ScalarField,
    pi: &HyperKZGProof<P>,
    transcript: &mut ProofTranscript,
) -> Result<(), ProofVerifyError>
where
    <P as Pairing>::ScalarField: field::JoltField,
{
    let x = point.to_vec();
    let y = P_of_x;

    let ell = x.len();

    let mut com = pi.com.clone();

    // we do not need to add x to the transcript, because in our context x was
    // obtained from the transcript
    transcript.append_points(&com.iter().map(|g| g.into_group()).collect::<Vec<P::G1>>());
    let r: <P as Pairing>::ScalarField = transcript.challenge_scalar();

    if r == P::ScalarField::zero() || C.0 == P::G1Affine::zero() {
        return Err(ProofVerifyError::InternalError);
    }
    com.insert(0, C.0); // set com_0 = C, shifts other commitments to the right

    let u = vec![r, -r, r * r];

    // Setup vectors (Y, ypos, yneg) from pi.v
    let v = &pi.v;
    if v.len() != 3 {
        return Err(ProofVerifyError::InternalError);
    }
    if v[0].len() != ell || v[1].len() != ell || v[2].len() != ell {
        return Err(ProofVerifyError::InternalError);
    }
    let ypos = &v[0];
    let yneg = &v[1];
    let mut Y = v[2].to_vec();
    Y.push(*y);

    // Check consistency of (Y, ypos, yneg)
    let two = P::ScalarField::from(2u64);
    for i in 0..ell {
        println!("Y[{}]: {:?}", i, two * r * Y[i + 1]);
        println!(
            "Y[{}]: {:?}",
            i,
            r * (P::ScalarField::one() - x[ell - i - 1]) * (ypos[i] + yneg[i])
                + x[ell - i - 1] * (ypos[i] - yneg[i])
        );

        if two * r * Y[i + 1]
            != r * (P::ScalarField::one() - x[ell - i - 1]) * (ypos[i] + yneg[i])
                + x[ell - i - 1] * (ypos[i] - yneg[i])
        {
            return Err(ProofVerifyError::InternalError);
        }
        // Note that we don't make any checks about Y[0] here, but our batching
        // check below requires it
    }

    // Check commitments to (Y, ypos, yneg) are valid
    if !kzg_verify_batch_single(vk, &com, &pi.w, &u, &pi.v, transcript) {
        return Err(ProofVerifyError::InternalError);
    }

    Ok(())
}

// vk is hashed in transcript already, so we do not add it here
fn kzg_verify_batch_single<P: Pairing>(
    vk: &HyperKZGVerifierKey<P>,
    C: &[P::G1Affine],
    W: &[P::G1Affine],
    u: &[P::ScalarField],
    v: &[Vec<P::ScalarField>],
    transcript: &mut ProofTranscript,
) -> bool
where
    <P as Pairing>::ScalarField: field::JoltField,
{
    let k = C.len();
    let t = u.len();

    // 25k CU savings by doing this, but it doesn't work for some reason
    // v.iter().flat_map(|x| x.iter()).for_each(|x| transcript.append_scalar(x));
    transcript.append_scalars(&v.iter().flatten().cloned().collect::<Vec<P::ScalarField>>());
    let q_powers: Vec<P::ScalarField> = transcript.challenge_scalar_powers(k);

    transcript.append_points(&W.iter().map(|g| g.into_group()).collect::<Vec<P::G1>>());
    let d_0: P::ScalarField = transcript.challenge_scalar();
    let d_1 = d_0 * d_0;

    assert_eq!(t, 3);
    assert_eq!(W.len(), 3);

    // Compute the batched openings
    // compute B(u_i) = v[i][0] + q*v[i][1] + ... + q^(t-1) * v[i][t-1]
    let B_u = v
        .into_iter()
        .map(|v_i| {
            v_i.into_iter()
                .zip(q_powers.iter())
                .map(|(a, b)| *a * *b)
                .sum()
        })
        .collect::<Vec<P::ScalarField>>();

    // Finally we do a MSM to get the value of the left hand side
    // NOTE - This is gas inefficient and grows with log of the proof size so we might want
    //        to move to a pippenger window algo with much smaller MSMs which we might save gas on.
    // Our first value is the C[0].
    let mut c_0 = vec![];
    C[0].serialize_uncompressed(&mut c_0).unwrap();
    let c_0 = convert_endianness_64(&c_0);

    let q_powers_0 = q_powers[0].into_bigint().to_bytes_be();
    let mut L_bytes = alt_bn128_multiplication(&[c_0, q_powers_0].concat()).unwrap();

    // Now do a running sum over the points in com
    for (i, point) in C.iter().enumerate() {
        let mut temp_bytes = vec![];
        point.serialize_uncompressed(&mut temp_bytes).unwrap();
        let temp_bytes = convert_endianness_64(&temp_bytes);

        let q_powers_i = q_powers[i / 2 + 1].into_bigint().to_bytes_be();
        let temp = alt_bn128_multiplication(&[temp_bytes, q_powers_i].concat()).unwrap();
        L_bytes = alt_bn128_addition(&[L_bytes, temp].concat()).unwrap();
    }

    // Next add in the W dot product U
    let mut temp_bytes = Vec::<u8>::with_capacity(64);
    W[0].serialize_uncompressed(&mut temp_bytes).unwrap();
    let W_0_bytes = convert_endianness_64(&temp_bytes);
    let temp =
        alt_bn128_multiplication(&[W_0_bytes.clone(), u[0].into_bigint().to_bytes_be()].concat())
            .unwrap();
    L_bytes = alt_bn128_addition(&[L_bytes, temp].concat()).unwrap();

    temp_bytes.clear();
    W[1].serialize_uncompressed(&mut temp_bytes).unwrap();
    let W_1_bytes = convert_endianness_64(&temp_bytes);
    let temp = alt_bn128_multiplication(
        &[W_1_bytes.clone(), (u[1] * d_0).into_bigint().to_bytes_be()].concat(),
    )
    .unwrap();
    L_bytes = alt_bn128_addition(&[L_bytes, temp].concat()).unwrap();

    temp_bytes.clear();
    W[2].serialize_uncompressed(&mut temp_bytes).unwrap();
    let W_2_bytes = convert_endianness_64(&temp_bytes);
    let temp = alt_bn128_multiplication(
        &[W_2_bytes.clone(), (u[2] * d_1).into_bigint().to_bytes_be()].concat(),
    )
    .unwrap();
    L_bytes = alt_bn128_addition(&[L_bytes, temp].concat()).unwrap();

    //-(B_u[0] + d_0 * B_u[1] + d_1 * B_u[2])
    let b_u = -(B_u[0] + d_0 * B_u[1] + d_1 * B_u[2]);

    // Add in to the msm b_u Vk_g1
    temp_bytes.clear();
    vk.kzg_vk
        .g1
        .serialize_uncompressed(&mut temp_bytes)
        .unwrap();
    let temp_bytes = convert_endianness_64(&temp_bytes);
    let temp =
        alt_bn128_multiplication(&[temp_bytes, b_u.into_bigint().to_bytes_be()].concat()).unwrap();
    L_bytes = alt_bn128_addition(&[L_bytes, temp].concat()).unwrap();

    // W[0] + W[1] * d_0 + W[2] * d_1;
    let mut R_bytes = W_0_bytes;
    let temp =
        alt_bn128_multiplication(&[W_1_bytes, d_0.into_bigint().to_bytes_be()].concat()).unwrap();
    R_bytes = alt_bn128_addition(&[R_bytes, temp].concat()).unwrap();
    let temp =
        alt_bn128_multiplication(&[W_2_bytes, d_1.into_bigint().to_bytes_be()].concat()).unwrap();
    R_bytes = alt_bn128_addition(&[R_bytes, temp].concat()).unwrap();

    let pairing_res = alt_bn128_pairing(&[L_bytes, R_bytes].concat());
    println!("Pairing result: {:?}", pairing_res);
    pairing_res.is_ok()
}
