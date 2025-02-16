use ark_ec::pairing::Pairing;
use ark_std::rand::Rng;

use super::{error::GType, vec_operations::InnerProd, Error, G1Vec, G2Vec, Gt, G1, G2};

#[derive(Clone)]
pub struct SingleParam<P>
where
    P: Pairing,
{
    /// random g1 generator
    ///
    /// only known by the **prover**
    pub g1: G1<P>,
    /// random g2 generator
    ///
    /// only known by the **prover**
    pub g2: G2<P>,
    /// commitment of <g1, g2> (inner product)
    ///
    /// known by the **verifier**
    pub c_g: Gt<P>,
}

#[derive(Clone)]
pub enum PublicParams<P>
where
    P: Pairing,
{
    Single(SingleParam<P>),

    Multi {
        /// random vec of generators of g1
        ///
        /// only known by the **prover**
        g1v: G1Vec<P>,
        /// random vec of generators of g2
        ///
        /// only known by the **prover**
        g2v: G2Vec<P>,

        /// random vec of generators of g1 that contains half of len(g1v) and it's used to
        /// calculate deltas
        ///
        /// only known by the **prover**
        gamma_1: G1Vec<P>,
        /// random vec of generators of g2 that contains half of len(g2v) and it's used to
        /// calculate deltas
        ///
        /// only known by the **prover**
        gamma_2: G2Vec<P>,

        /// commitment of <g1v, g2v> (inner product)
        ///
        /// known by the **verifier**
        c_g: Gt<P>,

        /// commitment of <g1v[..n/2], gamma_2> (inner product)
        ///
        /// known by the **verifier**
        delta_1l: Gt<P>,
        /// commitment of <g1v[n/2..], gamma_2> (inner product)
        ///
        /// known by the **verifier**
        delta_1r: Gt<P>,

        /// commitment of <gamma_1, g2v[..n/2]> (inner product)
        ///
        /// known by the **verifier**
        delta_2l: Gt<P>,
        /// commitment of <gamma_1, g2v[n/2..]> (inner product)
        ///
        /// known by the **verifier**
        delta_2r: Gt<P>,
    },
}

impl<Curve> PublicParams<Curve>
where
    Curve: Pairing,
{
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

    pub fn generate_public_params(rng: &mut impl Rng, mut n: usize) -> Result<Vec<Self>, Error> {
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

    pub fn new(rng: &mut impl Rng, n: usize) -> Result<Self, Error> {
        let g1v = G1Vec::random(rng, n);
        let g2v = G2Vec::random(rng, n);
        Self::params_with_provided_g(rng, g1v, g2v)
    }

    fn params_with_provided_g(
        rng: &mut impl Rng,
        g1v: G1Vec<Curve>,
        g2v: G2Vec<Curve>,
    ) -> Result<Self, Error> {
        let c_g = g1v.inner_prod(&g2v)?;
        match (&*g1v, &*g2v) {
            // if there's a single element, return a single param
            ([g1], [g2]) => Ok(Self::Single(SingleParam {
                g1: *g1,
                g2: *g2,
                c_g,
            })),
            // else, prepare gamma and delta public params
            (a, b) if !a.is_empty() & !b.is_empty() && a.len() == b.len() => {
                let m = g1v.len() / 2;

                let g1l: G1Vec<Curve> = (&g1v[..m]).into();
                let g1r: G1Vec<Curve> = (&g1v[m..]).into();

                let g2l: G2Vec<Curve> = (&g2v[..m]).into();
                let g2r: G2Vec<Curve> = (&g2v[m..]).into();

                let gamma_1 = G1Vec::random(rng, m);
                let gamma_2 = G2Vec::random(rng, m);

                let delta_1l = g1l.inner_prod(&gamma_2)?;
                let delta_1r = g1r.inner_prod(&gamma_2)?;
                let delta_2l = gamma_1.inner_prod(&g2l)?;
                let delta_2r = gamma_1.inner_prod(&g2r)?;

                Ok(Self::Multi {
                    g1v,
                    g2v,
                    c_g,
                    gamma_1,
                    gamma_2,
                    delta_1r,
                    delta_1l,
                    delta_2r,
                    delta_2l,
                })
            }
            ([], _) => Err(Error::EmptyVector(GType::G1)),
            (_, []) => Err(Error::EmptyVector(GType::G2)),
            (_, _) => Err(Error::LengthMismatch),
        }
    }

    fn new_derived(&self, rng: &mut impl Rng) -> Result<Self, Error> {
        let Self::Multi {
            gamma_1, gamma_2, ..
        } = self
        else {
            return Err(Error::DerivedFromSingle);
        };

        let g1v = gamma_1.clone();
        let g2v = gamma_2.clone();
        Self::params_with_provided_g(rng, g1v, g2v)
    }
}
