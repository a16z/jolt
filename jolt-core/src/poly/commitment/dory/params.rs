use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;
use ark_std::rand::Rng;

use super::{vec_operations::InnerProd, Error, G1Vec, G2Vec, Gt, G1, G2};

#[derive(Clone)]
pub struct SingleParam<P: Pairing> {
    pub g1: G1<P>,
    pub g2: G2<P>,
    pub x: Gt<P>,
}

#[derive(Clone)]
pub enum PublicParams<P: Pairing> {
    Single(SingleParam<P>),

    Multi {
        g1v: G1Vec<P>,
        g2v: G2Vec<P>,
        x: Gt<P>,

        gamma_1_prime: G1Vec<P>,
        gamma_2_prime: G2Vec<P>,

        delta_1r: Gt<P>,
        delta_1l: Gt<P>,
        delta_2r: Gt<P>,
        delta_2l: Gt<P>,
    },
}

impl<Curve: Pairing> PublicParams<Curve> {
    pub fn g1v(&self) -> Vec<G1<Curve>> {
        match self {
            PublicParams::Single(SingleParam { g1, .. }) => vec![*g1],
            PublicParams::Multi { g1v, .. } => g1v.to_vec(),
        }
    }

    pub fn g2v(&self) -> Vec<G2<Curve>> {
        match self {
            PublicParams::Single(SingleParam { g2, .. }) => vec![*g2],
            PublicParams::Multi { g2v, .. } => g2v.to_vec(),
        }
    }

    pub fn x(&self) -> &Gt<Curve> {
        match self {
            PublicParams::Single(SingleParam { x, .. }) | PublicParams::Multi { x, .. } => x,
        }
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
            params = res.last().expect("just pushed").new_derived(rng)?;
        }
        Ok(res)
    }

    pub fn new(rng: &mut impl Rng, n: usize) -> Result<Self, Error>
    where
        G1<Curve>: UniformRand,
        G2<Curve>: UniformRand,
    {
        let g1v = G1Vec::random(rng, n);
        let g2v = G2Vec::random(rng, n);
        Self::params_with_provided_g(rng, g1v, g2v)
    }

    fn params_with_provided_g(
        rng: &mut impl Rng,
        g1v: G1Vec<Curve>,
        g2v: G2Vec<Curve>,
    ) -> Result<Self, Error> {
        let x = g1v.inner_prod(&g2v)?;
        // if there's a single element, return a single param
        if let ([g1], [g2]) = (&*g1v, &*g2v) {
            Ok(Self::Single(SingleParam {
                g1: *g1,
                g2: *g2,
                x,
            }))
        // else, prepare gamma and delta public params
        } else {
            let m = g1v.len() / 2;
            let gamma_1l: G1Vec<Curve> = (&g1v[..m]).into();
            let gamma_1r: G1Vec<Curve> = (&g1v[m..]).into();

            let gamma_2l: G2Vec<Curve> = (&g2v[..m]).into();
            let gamma_2r: G2Vec<Curve> = (&g2v[m..]).into();

            let gamma_1_prime = G1Vec::random(rng, m);
            let gamma_2_prime = G2Vec::random(rng, m);

            let delta_1l = gamma_1l.inner_prod(&gamma_2_prime)?;
            let delta_1r = gamma_1r.inner_prod(&gamma_2_prime)?;
            let delta_2l = gamma_1_prime.inner_prod(&gamma_2l)?;
            let delta_2r = gamma_1_prime.inner_prod(&gamma_2r)?;

            Ok(Self::Multi {
                g1v,
                g2v,
                x,
                gamma_1_prime,
                gamma_2_prime,
                delta_1r,
                delta_1l,
                delta_2r,
                delta_2l,
            })
        }
    }

    fn new_derived(&self, rng: &mut impl Rng) -> Result<Self, Error>
    where
        G1<Curve>: UniformRand,
        G2<Curve>: UniformRand,
    {
        let Self::Multi {
            gamma_1_prime,
            gamma_2_prime,
            ..
        } = self
        else {
            panic!()
        };

        let g1v = gamma_1_prime.clone();
        let g2v = gamma_2_prime.clone();
        Self::params_with_provided_g(rng, g1v, g2v)
    }
}
