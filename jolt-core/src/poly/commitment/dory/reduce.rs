#![allow(dead_code)]

use std::ops::{Add, Mul};

use ark_ec::pairing::Pairing;
use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use sha3::{Digest, Sha3_256};

use super::{
    params::ReducePublicParams,
    vec_operations::{mul_gt, InnerProd},
    Commitment, Error, G1Vec, G2Vec, Gt, PublicParams, ScalarProof, Witness, Zr, G1, G2,
};

/// Proof
#[derive(Clone, CanonicalDeserialize, CanonicalSerialize)]
pub struct DoryProof<Curve: Pairing> {
    pub from_prover_1: Vec<ReduceProverStep1Elements<Curve>>,
    pub from_prover_2: Vec<ReduceProverStep2Elements<Curve>>,
    pub final_proof: ScalarProof<Curve>,
}

impl<Curve: Pairing> DoryProof<Curve> {
    fn verify_recursive(
        public_params: &[PublicParams<Curve>],
        commitment: Commitment<Curve>,
        from_prover_1: &[ReduceProverStep1Elements<Curve>],
        from_prover_2: &[ReduceProverStep2Elements<Curve>],
        final_proof: &ScalarProof<Curve>,
    ) -> Result<bool, Error> {
        match public_params {
            [] => Err(Error::EmptyPublicParams),
            [params] => final_proof.verify(params, &commitment),
            [param1, public_params_rest @ ..] => {
                let digest = param1.digest(None)?.to_vec();

                let PublicParams { x, reduce_pp, .. } = param1;
                let ReducePublicParams {
                    delta_1r,
                    delta_1l,
                    delta_2r,
                    delta_2l,
                    ..
                } = reduce_pp.as_ref().expect("gv1 is greater than 1");

                match (from_prover_1, from_prover_2) {
                    (
                        [ReduceProverStep1Elements {
                            d1l, d1r, d2l, d2r, ..
                        }, from_prover_1_rest @ ..],
                        [ReduceProverStep2Elements {
                            c_plus, c_minus, ..
                        }, from_prover_2_rest @ ..],
                    ) => {
                        let Commitment { c, d1, d2 } = commitment;

                        let step_1_element = ReduceProverStep1Elements {
                            pp_digest: digest,
                            d1l: *d1l,
                            d1r: *d1r,
                            d2l: *d2l,
                            d2r: *d2r,
                            c,
                            d1,
                            d2,
                        };

                        let (betha, step_1_digest) = step_1_element.ro()?;

                        let step_2_element = ReduceProverStep2Elements {
                            step_1_digest,
                            c_plus: *c_plus,
                            c_minus: *c_minus,
                        };

                        let alpha = step_2_element.ro()?;
                        let inverse_alpha = alpha.inverse().ok_or(Error::ZrZero)?;
                        let inverse_betha = betha.inverse().ok_or(Error::ZrZero)?;

                        let c_prime = mul_gt(&[
                            c,
                            *x,
                            d2 * betha,
                            d1 * inverse_betha,
                            *c_plus * alpha,
                            *c_minus * inverse_alpha,
                        ])
                        .expect("slice is not empty");

                        let d1_prime = mul_gt(&[
                            *d1l * alpha,
                            *d1r,
                            *delta_1l * alpha * betha,
                            *delta_1r * betha,
                        ])
                        .expect("slice is not empty");

                        let d2_prime = mul_gt(&[
                            *d2l * inverse_alpha,
                            *d2r,
                            *delta_2l * inverse_alpha * inverse_betha,
                            *delta_2r * inverse_betha,
                        ])
                        .expect("slice is not empty");

                        let next_commitment = Commitment {
                            c: c_prime,
                            d1: d1_prime,
                            d2: d2_prime,
                        };

                        Self::verify_recursive(
                            public_params_rest,
                            next_commitment,
                            from_prover_1_rest,
                            from_prover_2_rest,
                            final_proof,
                        )
                    }
                    _ => todo!(),
                }
            }
        }
    }

    pub fn verify(
        &self,
        public_params: &[PublicParams<Curve>],
        commitment: Commitment<Curve>,
    ) -> Result<bool, Error>
    where
        Gt<Curve>: Mul<Zr<Curve>, Output = Gt<Curve>>,
        G1<Curve>: Mul<Zr<Curve>, Output = G1<Curve>>,
        G2<Curve>: Mul<Zr<Curve>, Output = G2<Curve>>,
    {
        Self::verify_recursive(
            public_params,
            commitment,
            &self.from_prover_1,
            &self.from_prover_2,
            &self.final_proof,
        )
    }
}

#[derive(Clone, CanonicalDeserialize, CanonicalSerialize)]
pub struct ReduceProverStep1Elements<Curve: Pairing> {
    pp_digest: Vec<u8>,
    d1l: Gt<Curve>,
    d1r: Gt<Curve>,
    d2l: Gt<Curve>,
    d2r: Gt<Curve>,
    c: Gt<Curve>,
    d1: Gt<Curve>,
    d2: Gt<Curve>,
}

impl<Curve: Pairing> ReduceProverStep1Elements<Curve> {
    pub fn ro(&self) -> Result<(Zr<Curve>, Vec<u8>), Error> {
        let mut hasher = Sha3_256::new();
        self.serialize_uncompressed(&mut hasher)?;
        let digest = hasher.finalize();
        Ok((
            Zr::<Curve>::from_be_bytes_mod_order(&digest),
            digest.to_vec(),
        ))
    }
}

#[derive(Clone, CanonicalDeserialize, CanonicalSerialize)]
pub struct ReduceProverStep2Elements<Curve: Pairing> {
    step_1_digest: Vec<u8>,
    c_plus: Gt<Curve>,
    c_minus: Gt<Curve>,
}

impl<Curve: Pairing> ReduceProverStep2Elements<Curve> {
    pub fn ro(&self) -> Result<Zr<Curve>, Error> {
        let mut hasher = Sha3_256::new();
        self.serialize_uncompressed(&mut hasher)?;
        let digest = hasher.finalize();
        Ok(Zr::<Curve>::from_be_bytes_mod_order(&digest))
    }
}

pub fn reduce<Curve: Pairing>(
    params: &[PublicParams<Curve>],
    witness: Witness<Curve>,
    Commitment { c, d1, d2 }: Commitment<Curve>,
) -> Result<DoryProof<Curve>, Error>
where
    G1Vec<Curve>: Add<G1Vec<Curve>, Output = G1Vec<Curve>>,
    G2Vec<Curve>: Add<G2Vec<Curve>, Output = G2Vec<Curve>>,
{
    match params {
        [] => unimplemented!(),
        [param1, rest_param @ ..] => {
            let digest = param1.digest(None)?;

            let PublicParams {
                g1v,
                g2v,
                x,
                reduce_pp,
                ..
            } = param1;

            let ReducePublicParams {
                delta_1r,
                delta_1l,
                delta_2r,
                delta_2l,
                gamma_1_prime,
                gamma_2_prime,
            } = reduce_pp.as_ref().unwrap();

            let m = g1v.len() / 2;

            // P:
            let v1l: G1Vec<Curve> = (&witness.v1[..m]).into();
            let v1r: G1Vec<Curve> = (&witness.v1[m..]).into();
            let v2l = (&witness.v2[..m]).into();
            let v2r = (&witness.v2[m..]).into();

            // P --> V:
            let d1l = v1l.inner_prod(gamma_2_prime)?;
            let d1r = v1r.inner_prod(gamma_2_prime)?;
            let d2l = gamma_1_prime.inner_prod(&v2l)?;
            let d2r = gamma_1_prime.inner_prod(&v2r)?;

            let step_1_element = ReduceProverStep1Elements {
                pp_digest: digest,
                d1l,
                d1r,
                d2l,
                d2r,
                c,
                d1,
                d2,
            };

            let (betha, step_1_digest) = step_1_element.ro()?;
            let inverse_betha = betha.inverse().unwrap();

            // P:
            let v1 = witness.v1 + (g1v * betha);
            let v2 = witness.v2 + (g2v * inverse_betha);

            let v1l: G1Vec<Curve> = v1[..m].to_vec().into();
            let v1r: G1Vec<Curve> = v1[m..].to_vec().into();
            let v2l = v2[..m].to_vec().into();
            let v2r = v2[m..].to_vec().into();

            // P --> V:
            let c_plus = v1l.inner_prod(&v2r)?;
            let c_minus = v1r.inner_prod(&v2l)?;

            let step_2_element = ReduceProverStep2Elements {
                step_1_digest,
                c_plus,
                c_minus,
            };
            let alpha = step_2_element.ro()?;
            let inverse_alpha = alpha.inverse().unwrap();

            let v1_prime = v1l * alpha + v1r;
            let v2_prime = v2l * inverse_alpha + v2r;

            let next_witness = Witness {
                v1: v1_prime,
                v2: v2_prime,
            };

            if m == 1 {
                return Ok(DoryProof {
                    from_prover_1: vec![step_1_element],
                    from_prover_2: vec![step_2_element],
                    final_proof: ScalarProof::new(next_witness),
                });
            }

            let c_prime = mul_gt(&[
                c,
                *x,
                d2 * betha,
                d1 * inverse_betha,
                c_plus * alpha,
                c_minus * inverse_alpha,
            ])
            .unwrap();

            let d1_prime = mul_gt(&[
                d1l * alpha,
                d1r,
                *delta_1l * alpha * betha,
                *delta_1r * betha,
            ])
            .unwrap();

            let d2_prime = mul_gt(&[
                d2l * inverse_alpha,
                d2r,
                *delta_2l * inverse_alpha * inverse_betha,
                *delta_2r * inverse_betha,
            ])
            .unwrap();

            let next_commitment = Commitment {
                c: c_prime,
                d1: d1_prime,
                d2: d2_prime,
            };

            let DoryProof {
                from_prover_1: step_1_elements,
                from_prover_2: step_2_elements,
                final_proof: scalar_product_proof,
            } = reduce(rest_param, next_witness, next_commitment)?;

            let mut from_prover_1 = vec![step_1_element];
            from_prover_1.extend(step_1_elements);
            let mut from_prover_2 = vec![step_2_element];
            from_prover_2.extend(step_2_elements);

            Ok(DoryProof {
                from_prover_1,
                from_prover_2,
                final_proof: scalar_product_proof,
            })
        }
    }
}
