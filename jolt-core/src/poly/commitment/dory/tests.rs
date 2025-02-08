use ark_bn254::Bn254;
use ark_std::UniformRand;

use crate::poly::commitment::dory::scalar::Witness;

use super::{commit, reduce, G1Vec, G2Vec, PublicParams, ScalarProof, G1, G2};

#[test]
fn test_scalar_product_proof() {
    let mut rng = ark_std::test_rng();
    let public_params = PublicParams::<Bn254>::new(&mut rng, 1).unwrap();

    let g1v = vec![G1::<Bn254>::rand(&mut rng)];
    let g2v = vec![G2::<Bn254>::rand(&mut rng)];
    let witness = Witness {
        v1: g1v.into(),
        v2: g2v.into(),
    };
    let commitment = commit(witness.clone(), &public_params).unwrap();

    let proof = ScalarProof::new(witness);
    assert!(proof.verify(&public_params, &commitment).unwrap());
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

    let proof = reduce::reduce(&params, witness, commitment).unwrap();

    assert_eq!(proof.from_prover_1.len(), 3);
    assert_eq!(proof.from_prover_2.len(), 3);

    assert_eq!(params[0].g1v.len(), 8);
    assert_eq!(params[1].g1v.len(), 4);
    assert_eq!(params[2].g1v.len(), 2);
    assert_eq!(params[3].g1v.len(), 1);

    assert_eq!(params[0].reduce_pp.as_ref().unwrap().gamma_1_prime.len(), 4);
    assert_eq!(params[1].reduce_pp.as_ref().unwrap().gamma_1_prime.len(), 2);
    assert_eq!(params[2].reduce_pp.as_ref().unwrap().gamma_1_prime.len(), 1);
    assert!(params[3].reduce_pp.is_none());
    assert!(proof.verify(&params, commitment).unwrap());
}
