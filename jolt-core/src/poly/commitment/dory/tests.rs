use ark_bn254::Bn254;
use ark_std::UniformRand;

use crate::{
    poly::commitment::dory::scalar::Witness,
    utils::transcript::{KeccakTranscript, Transcript},
};

use super::{commit, reduce, G1Vec, G2Vec, PublicParams, ScalarProof, G1, G2};

#[test]
fn test_scalar_product_proof() {
    let mut rng = ark_std::test_rng();
    let public_params = PublicParams::<Bn254>::new(&mut rng, 1).unwrap();
    let PublicParams::Single(single_param) = &public_params else {
        panic!()
    };

    let g1v = vec![G1::<Bn254>::rand(&mut rng)];
    let g2v = vec![G2::<Bn254>::rand(&mut rng)];
    let witness = Witness {
        v1: g1v.into(),
        v2: g2v.into(),
    };
    let commitment = commit(witness.clone(), &public_params).unwrap();

    let proof = ScalarProof::new(witness);

    assert!(proof.verify(single_param, &commitment).unwrap());
}

#[test]
fn test_dory_reduce() {
    let mut rng = ark_std::test_rng();
    let n = 8;
    let g1v = G1Vec::<Bn254>::random(&mut rng, n);
    let g2v = G2Vec::random(&mut rng, n);

    let params = PublicParams::generate_public_params(&mut rng, n).unwrap();

    let witness = Witness { v1: g1v, v2: g2v };
    let commitment = commit(witness.clone(), &params[0]).unwrap();
    let mut transcript = KeccakTranscript::new(&[]);

    let proof = reduce::reduce(&mut transcript, &params, witness, commitment).unwrap();

    assert_eq!(proof.from_prover_1.len(), 3);
    assert_eq!(proof.from_prover_2.len(), 3);

    assert_eq!(params[0].g1v().len(), 8);
    assert_eq!(params[1].g1v().len(), 4);
    assert_eq!(params[2].g1v().len(), 2);
    assert_eq!(params[3].g1v().len(), 1);

    let mut prev = n;
    for param in &params[..params.len() - 1] {
        let PublicParams::Multi { gamma_1_prime, .. } = param else {
            panic!()
        };
        prev /= 2;
        assert_eq!(gamma_1_prime.len(), prev);
    }
    assert!(matches!(params[3], PublicParams::Single(_)));

    let mut transcript = KeccakTranscript::new(&[]);
    assert!(proof.verify(&mut transcript, &params, commitment).unwrap());
}
