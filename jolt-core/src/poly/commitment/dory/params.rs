use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;
use ark_serialize::CanonicalSerialize;
use ark_std::rand::Rng;
use sha3::{Digest, Sha3_256};

use super::{vec_operations::InnerProd, Error, G1Vec, G2Vec, Gt, G1, G2};

#[derive(Clone)]
pub struct PublicParams<P: Pairing> {
    pub g1v: G1Vec<P>,
    pub g2v: G2Vec<P>,

    pub x: Gt<P>,

    pub reduce_pp: Option<ReducePublicParams<P>>,
}

#[derive(Clone)]
pub struct ReducePublicParams<P: Pairing> {
    pub gamma_1_prime: G1Vec<P>,
    pub gamma_2_prime: G2Vec<P>,

    pub delta_1r: Gt<P>,
    pub delta_1l: Gt<P>,
    pub delta_2r: Gt<P>,
    pub delta_2l: Gt<P>,
}

impl<Curve: Pairing> PublicParams<Curve> {
    pub fn new(rng: &mut impl Rng, n: usize) -> Result<Self, Error>
    where
        G1<Curve>: UniformRand,
        G2<Curve>: UniformRand,
    {
        let g1v = G1Vec::random(rng, n);
        let g2v = G2Vec::random(rng, n);
        let x = g1v.inner_prod(&g2v)?;
        let reduce_pp = ReducePublicParams::new(rng, &g1v, &g2v)?;
        let value = Self {
            g1v,
            g2v,
            reduce_pp,
            x,
        };
        Ok(value)
    }

    pub fn new_derived(&self, rng: &mut impl Rng, n: usize) -> Result<Self, Error>
    where
        G1<Curve>: UniformRand,
        G2<Curve>: UniformRand,
    {
        if self.g1v.len() != 2 * n || self.g2v.len() != 2 * n {
            return Err(Error::LengthNotTwice);
        }
        let Some(reduce_pp) = &self.reduce_pp else {
            return Err(Error::ReduceParamsNotInitialized);
        };
        let g1v = reduce_pp.gamma_1_prime.clone();
        let g2v = reduce_pp.gamma_2_prime.clone();

        let reduce_pp = ReducePublicParams::new(rng, &g1v, &g2v)?;
        let x = g1v.inner_prod(&g2v)?;

        let value = Self {
            g1v,
            g2v,
            reduce_pp,
            x,
        };
        Ok(value)
    }

    pub fn digest(&self, prev: Option<&[u8]>) -> Result<Vec<u8>, Error> {
        let mut hasher = Sha3_256::new();
        if let Some(prev) = prev {
            hasher.update(prev);
        }

        if let Some(reduce_pp) = &self.reduce_pp {
            hasher.update(reduce_pp.digest()?);
        }
        self.x
            .serialize_uncompressed(&mut hasher)
            .expect("Serialization failed");

        self.g1v.serialize_uncompressed(&mut hasher)?;
        self.g2v.serialize_uncompressed(&mut hasher)?;

        Ok(hasher.finalize().to_vec())
    }

    pub fn generate_public_params(rng: &mut impl Rng, mut n: usize) -> Result<Vec<Self>, Error>
    where
        G1<Curve>: UniformRand,
        G2<Curve>: UniformRand,
    {
        let mut res = Vec::new();
        let mut params = Self::new(rng, n)?;
        while n > 0 {
            res.push(params);
            if n / 2 == 0 {
                break;
            }
            n /= 2;
            params = res.last().expect("just pushed").new_derived(rng, n)?;
        }
        Ok(res)
    }
}

impl<Curve: Pairing> ReducePublicParams<Curve> {
    pub fn new(
        rng: &mut impl Rng,
        g1v: &[G1<Curve>],
        g2v: &[G2<Curve>],
    ) -> Result<Option<Self>, Error>
    where
        G1<Curve>: UniformRand,
        G2<Curve>: UniformRand,
    {
        assert_eq!(g1v.len(), g2v.len());
        if g1v.len() == 1 {
            return Ok(None);
        }
        let m = g1v.len() / 2;
        let gamma_1l: G1Vec<Curve> = (&g1v[..m]).into();
        let gamma_1r: G1Vec<Curve> = (&g1v[m..]).into();

        let gamma_2l = (&g2v[..m]).into();
        let gamma_2r = (&g2v[m..]).into();

        let gamma_1_prime = G1Vec::random(rng, m);
        let gamma_2_prime = G2Vec::random(rng, m);

        let delta_1l = gamma_1l.inner_prod(&gamma_2_prime)?;
        let delta_1r = gamma_1r.inner_prod(&gamma_2_prime)?;
        let delta_2l = gamma_1_prime.inner_prod(&gamma_2l)?;
        let delta_2r = gamma_1_prime.inner_prod(&gamma_2r)?;
        Ok(Some(Self {
            gamma_1_prime,
            gamma_2_prime,
            delta_1r,
            delta_1l,
            delta_2r,
            delta_2l,
        }))
    }

    pub fn digest(&self) -> Result<Vec<u8>, Error> {
        let mut hasher = Sha3_256::new();

        self.gamma_1_prime.serialize_uncompressed(&mut hasher)?;
        self.gamma_2_prime.serialize_uncompressed(&mut hasher)?;
        self.delta_1r.serialize_uncompressed(&mut hasher)?;
        self.delta_1l.serialize_uncompressed(&mut hasher)?;
        self.delta_2r.serialize_uncompressed(&mut hasher)?;
        self.delta_2l.serialize_uncompressed(&mut hasher)?;

        Ok(hasher.finalize().to_vec())
    }
}
