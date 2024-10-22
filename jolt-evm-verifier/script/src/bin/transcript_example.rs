use jolt_core::{
    field::JoltField,
    utils::transcript::{KeccakTranscript, Transcript},
};

use ark_ff::{BigInteger, PrimeField};

use alloy_primitives::{hex, FixedBytes, U256};
use alloy_sol_types::{sol, SolType};
use ark_bn254::{Fr, G1Projective};
use ark_ec::CurveGroup;
use ark_std::rand::Rng;
use ark_std::test_rng;
use ark_std::UniformRand;

use ark_serialize::CanonicalSerialize;

fn main() {
    sol!(struct TranscriptValues {
        uint64[] usizes;
        uint256[] scalars;
        uint256[][] scalarArrays;
        uint256[] points;
        uint256[][] pointArrays;
        bytes32[][] bytes_examples;
        uint256[] expectedScalarResponses;
        uint256[][] expectedVectorResponses;
    });

    // We write two elements of each kind to the rust transcript, then we pull a scalar and
    // array challenge. We feed the data about the writes and the reads back into sol and then
    // we do the same thing in solidity. Checking that the reads are the same. Because of how
    // hashes work this should give good coverage.

    let mut rng = test_rng();
    let mut scalar_responses = Vec::<Fr>::new();
    let mut vector_responses = Vec::<Vec<Fr>>::new();

    let usizes: Vec<u64> = vec![rng.gen(), rng.gen()];
    let mut transcript = KeccakTranscript::new(b"test_transcript");

    transcript.append_u64(usizes[0]);
    transcript.append_u64(usizes[1]);

    scalar_responses.push(transcript.challenge_scalar());
    vector_responses.push(transcript.challenge_vector(4));

    let scalars = [Fr::random(&mut rng), Fr::random(&mut rng)];
    transcript.append_scalar(&scalars[0]);
    transcript.append_scalar(&scalars[1]);
    let mut serialized_0 = vec![];
    let mut serialized_1 = vec![];
    scalars[0]
        .serialize_uncompressed(&mut serialized_0)
        .unwrap();
    scalars[1]
        .serialize_uncompressed(&mut serialized_1)
        .unwrap();

    let encoded_scalars = vec![
        U256::from_le_slice(&serialized_0),
        U256::from_le_slice(&serialized_1),
    ];

    scalar_responses.push(transcript.challenge_scalar());
    vector_responses.push(transcript.challenge_vector(4));

    let vectors = vec![
        vec![Fr::random(&mut rng), Fr::random(&mut rng)],
        vec![Fr::random(&mut rng), Fr::random(&mut rng)],
    ];
    transcript.append_scalars(&vectors[0]);
    transcript.append_scalars(&vectors[1]);
    scalar_responses.push(transcript.challenge_scalar());
    vector_responses.push(transcript.challenge_vector(4));
    let encoded_vectors: Vec<Vec<U256>> = vectors
        .into_iter()
        .map(|x| {
            x.into_iter()
                .map(|c| U256::from_be_slice(c.into_bigint().to_bytes_be().as_slice()))
                .collect()
        })
        .collect();

    let points = vec![G1Projective::rand(&mut rng), G1Projective::rand(&mut rng)];
    transcript.append_point::<G1Projective>(&points[0]);
    transcript.append_point::<G1Projective>(&points[1]);
    scalar_responses.push(transcript.challenge_scalar());
    vector_responses.push(transcript.challenge_vector(4));
    let mut encoded_points = Vec::<U256>::new();
    for point in points {
        let point_aff = point.into_affine();
        encoded_points.push(U256::from_be_slice(
            &point_aff.x.into_bigint().to_bytes_be(),
        ));
        encoded_points.push(U256::from_be_slice(
            &point_aff.y.into_bigint().to_bytes_be(),
        ));
    }

    let point_vectors = vec![
        vec![G1Projective::rand(&mut rng), G1Projective::rand(&mut rng)],
        vec![G1Projective::rand(&mut rng), G1Projective::rand(&mut rng)],
    ];
    transcript.append_points::<G1Projective>(&point_vectors[0]);
    transcript.append_points::<G1Projective>(&point_vectors[1]);
    scalar_responses.push(transcript.challenge_scalar());
    vector_responses.push(transcript.challenge_vector(4));

    let mut encoded_point_vector = Vec::<Vec<U256>>::new();
    for point_vec in point_vectors {
        let mut encoded_point = Vec::<U256>::new();
        for point in point_vec {
            let point_aff = point.into_affine();
            encoded_point.push(U256::from_be_slice(
                &point_aff.x.into_bigint().to_bytes_be(),
            ));
            encoded_point.push(U256::from_be_slice(
                &point_aff.y.into_bigint().to_bytes_be(),
            ));
        }
        encoded_point_vector.push(encoded_point)
    }

    let byte_vectors = vec![
        vec![
            rng.gen::<[u8; 32]>(),
            rng.gen::<[u8; 32]>(),
            rng.gen::<[u8; 32]>(),
        ],
        vec![
            rng.gen::<[u8; 32]>(),
            rng.gen::<[u8; 32]>(),
            rng.gen::<[u8; 32]>(),
        ],
    ];
    let mut encoded_bytes_vector = Vec::<Vec<FixedBytes<32>>>::new();
    for bytes in byte_vectors.clone() {
        let mut encoded_bytes = Vec::<FixedBytes<32>>::new();
        for &byte32 in bytes.iter() {
            encoded_bytes.push(FixedBytes::<32>::new(byte32));
        }
        encoded_bytes_vector.push(encoded_bytes)
    }

    transcript.append_bytes(
        &(byte_vectors[0]
            .clone()
            .into_iter()
            .flat_map(|x| x.into_iter())
            .collect::<Vec<u8>>()),
    );
    transcript.append_bytes(
        &(byte_vectors[1]
            .clone()
            .into_iter()
            .flat_map(|x| x.into_iter())
            .collect::<Vec<u8>>()),
    );
    scalar_responses.push(transcript.challenge_scalar());
    vector_responses.push(transcript.challenge_vector(4));

    let encoded_scalar_responses = scalar_responses
        .iter()
        .map(|c| U256::from_be_slice(c.into_bigint().to_bytes_be().as_slice()))
        .collect();
    let encoded_vector_responses = vector_responses
        .into_iter()
        .map(|x| {
            x.into_iter()
                .map(|c| U256::from_be_slice(c.into_bigint().to_bytes_be().as_slice()))
                .collect()
        })
        .collect();

    let pre_encoded: TranscriptValues = TranscriptValues::from((
        usizes,
        encoded_scalars,
        encoded_vectors,
        encoded_points,
        encoded_point_vector,
        encoded_bytes_vector,
        encoded_scalar_responses,
        encoded_vector_responses,
    ));

    print!(
        "{}",
        hex::encode(TranscriptValues::abi_encode(&pre_encoded))
    );
}
