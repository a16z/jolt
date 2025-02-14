use std::ops::Mul;

use ark_ec::pairing::Pairing;
use ark_ff::{Field, UniformRand};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::thread_rng;

use super::{
    vec_operations::{e, mul_gt, InnerProd},
    Error, G1Vec, G2Vec, Gt, PublicParams, Zr, G1, G2,
};

/// Witness over set Zr
#[derive(Clone)]
pub struct Witness<Curve: Pairing> {
    pub v1: G1Vec<Curve>,
    pub v2: G2Vec<Curve>,
}

impl<P: Pairing> Witness<P> {
    pub fn new(params: &PublicParams<P>, poly: &[Zr<P>]) -> Self {
        let v1 = params
            .g1v
            .iter()
            .zip(poly.iter())
            .map(|(a, b)| *a * *b)
            .collect::<Vec<G1<P>>>();

        let v2 = params
            .g2v
            .iter()
            .zip(poly.iter())
            .map(|(a, b)| *a * *b)
            .collect::<Vec<G2<P>>>();
        let v1 = v1.into();
        let v2 = v2.into();

        Self { v1, v2 }
    }
}

#[derive(Clone, Copy, CanonicalSerialize, CanonicalDeserialize, Debug, Default, PartialEq, Eq)]
pub struct Commitment<Curve: Pairing> {
    pub c: Gt<Curve>,
    pub d1: Gt<Curve>,
    pub d2: Gt<Curve>,
}

pub fn commit<Curve: Pairing>(
    Witness { v1, v2 }: Witness<Curve>,
    public_params: &PublicParams<Curve>,
) -> Result<Commitment<Curve>, Error> {
    let d1 = v1.inner_prod(&public_params.g2v)?;
    let d2 = public_params.g1v.inner_prod(&v2)?;
    let c = v1.inner_prod(&v2)?;

    let commitment = Commitment { d1, d2, c };
    Ok(commitment)
}

#[derive(Clone, CanonicalDeserialize, CanonicalSerialize)]
pub struct ScalarProof<Curve: Pairing> {
    //pp: &'a PublicParams<Curve>,
    e1: G1Vec<Curve>,
    e2: G2Vec<Curve>,
}

impl<Curve: Pairing> ScalarProof<Curve> {
    pub fn new(witness: Witness<Curve>) -> Self {
        Self {
            //pp: public_params,
            e1: witness.v1,
            e2: witness.v2,
        }
    }

    pub fn verify(
        &self,
        pp: &PublicParams<Curve>,
        Commitment { c, d1, d2 }: &Commitment<Curve>,
    ) -> Result<bool, Error>
    where
        for<'c> &'c G1Vec<Curve>: Mul<Zr<Curve>, Output = G1Vec<Curve>>,
        G1<Curve>: Mul<Zr<Curve>, Output = G1<Curve>>,
        G2<Curve>: Mul<Zr<Curve>, Output = G2<Curve>>,
        Gt<Curve>: Mul<Zr<Curve>, Output = Gt<Curve>>,
    {
        let mut rng = thread_rng();
        let d: Zr<Curve> = Zr::<Curve>::rand(&mut rng);
        let d_inv = d.inverse().ok_or(Error::CouldntInvertD)?;

        let g1 = G1Vec::<Curve>::from(&[self.e1[0], pp.g1v[0] * d]).sum();

        let g2 = G2Vec::<Curve>::from(&[self.e2[0], pp.g2v[0] * d_inv]).sum();
        let left_eq = e(g1, g2);

        let right_eq = mul_gt(&[pp.x, *c, *d2 * d, *d1 * d_inv]).expect("has more than one item");

        Ok(left_eq == right_eq)
    }
}
