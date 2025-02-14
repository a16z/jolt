use std::ops::{Add, Deref, Mul};

use ark_ec::{pairing::Pairing, Group};
use ark_ff::UniformRand;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::Rng;

use super::{
    error::{Error, GType},
    Gt, Zr, G1, G2,
};

pub fn e<Curve: Pairing>(g1: G1<Curve>, g2: G2<Curve>) -> Gt<Curve> {
    Curve::pairing(g1, g2)
}

pub fn mul_gt<Curve: Pairing>(gts: &[Gt<Curve>]) -> Option<Gt<Curve>> {
    gts.iter().fold(None, |prev, curr| match prev {
        Some(prev) => Some(curr + prev),
        None => Some(*curr),
    })
}

pub trait InnerProd {
    type G2;
    type Gt;
    fn inner_prod(&self, g2v: &Self::G2) -> Result<Self::Gt, Error>;
}

impl<Curve: Pairing> InnerProd for G1Vec<Curve> {
    type G2 = G2Vec<Curve>;

    type Gt = Gt<Curve>;

    fn inner_prod(&self, g2v: &Self::G2) -> Result<Self::Gt, Error> {
        match (self.as_ref(), g2v.as_ref()) {
            ([], _) => Err(Error::EmptyVector(GType::G1)),
            (_, []) => Err(Error::EmptyVector(GType::G2)),
            (a, b) if a.len() != b.len() => Err(Error::LengthMismatch),
            ([g1], [g2]) => Ok(e(*g1, *g2)),
            (a, b) => Ok(Curve::multi_pairing(a, b)),
        }
    }
}

// G1
#[derive(Clone, CanonicalDeserialize, CanonicalSerialize)]
pub struct G1Vec<Curve: Pairing>(Vec<G1<Curve>>);

impl<Curve: Pairing> G1Vec<Curve> {
    pub fn sum(&self) -> G1<Curve> {
        self.iter().sum()
    }

    pub fn random(rng: &mut impl Rng, n: usize) -> Self
    where
        G1<Curve>: UniformRand,
    {
        Self(
            (0..n)
                .map(|_| {
                    let random_scalar = Zr::<Curve>::rand(rng);
                    G1::<Curve>::generator() * random_scalar
                })
                .collect(),
        )
    }
}

impl<Curve: Pairing> Deref for G1Vec<Curve> {
    type Target = [G1<Curve>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Curve> Add for G1Vec<Curve>
where
    Curve: Pairing,
    for<'b> &'b G1<Curve>: Add<&'b G1<Curve>, Output = G1<Curve>>,
{
    type Output = G1Vec<Curve>;

    fn add(self, rhs: Self) -> Self::Output {
        Self(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(val1, val2)| val1 + val2)
                .collect(),
        )
    }
}

impl<Curve: Pairing> Mul<Zr<Curve>> for G1Vec<Curve>
where
    G1<Curve>: Copy,
    G1<Curve>: Mul<Zr<Curve>, Output = G1<Curve>>,
{
    type Output = G1Vec<Curve>;

    fn mul(self, rhs: Zr<Curve>) -> Self::Output {
        G1Vec(self.0.iter().map(|val| *val * rhs).collect())
    }
}

impl<Curve: Pairing> Mul<Zr<Curve>> for &G1Vec<Curve>
where
    G1<Curve>: Copy,
    G1<Curve>: Mul<Zr<Curve>, Output = G1<Curve>>,
{
    type Output = G1Vec<Curve>;

    fn mul(self, rhs: Zr<Curve>) -> Self::Output {
        G1Vec(self.iter().map(|val| *val * rhs).collect())
    }
}

impl<Curve: Pairing> From<&[G1<Curve>]> for G1Vec<Curve> {
    fn from(value: &[G1<Curve>]) -> Self {
        Self(value.into())
    }
}

impl<Curve: Pairing, const N: usize> From<&[G1<Curve>; N]> for G1Vec<Curve> {
    fn from(value: &[G1<Curve>; N]) -> Self {
        Self((*value).into())
    }
}

impl<Curve: Pairing> From<Vec<G1<Curve>>> for G1Vec<Curve> {
    fn from(value: Vec<G1<Curve>>) -> Self {
        Self(value)
    }
}

// G2

#[derive(Clone, CanonicalDeserialize, CanonicalSerialize)]
pub struct G2Vec<Curve: Pairing>(Vec<G2<Curve>>);

impl<Curve: Pairing> G2Vec<Curve> {
    pub fn sum(&self) -> G2<Curve> {
        self.iter().sum()
    }

    pub fn random(rng: &mut impl Rng, n: usize) -> Self
    where
        G2<Curve>: UniformRand,
    {
        Self(
            (0..n)
                .map(|_| {
                    let random_scalar = Zr::<Curve>::rand(rng);
                    G2::<Curve>::generator() * random_scalar
                })
                .collect(),
        )
    }
}

impl<Curve: Pairing> Deref for G2Vec<Curve> {
    type Target = [G2<Curve>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Curve: Pairing> From<&[G2<Curve>]> for G2Vec<Curve> {
    fn from(value: &[G2<Curve>]) -> Self {
        Self(value.into())
    }
}

impl<Curve: Pairing, const N: usize> From<&[G2<Curve>; N]> for G2Vec<Curve> {
    fn from(value: &[G2<Curve>; N]) -> Self {
        Self((*value).into())
    }
}

impl<Curve: Pairing> From<Vec<G2<Curve>>> for G2Vec<Curve> {
    fn from(value: Vec<G2<Curve>>) -> Self {
        Self(value)
    }
}

impl<Curve> Add for G2Vec<Curve>
where
    Curve: Pairing,
{
    type Output = G2Vec<Curve>;

    fn add(self, rhs: Self) -> Self::Output {
        G2Vec(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(val1, val2)| *val1 + *val2)
                .collect(),
        )
    }
}

impl<Curve> Mul<Zr<Curve>> for G2Vec<Curve>
where
    Curve: Pairing,
    G2<Curve>: Copy + Mul<Zr<Curve>, Output = G2<Curve>>,
{
    type Output = G2Vec<Curve>;

    fn mul(self, rhs: Zr<Curve>) -> Self::Output {
        G2Vec(self.0.iter().map(|val| *val * rhs).collect())
    }
}

impl<Curve> Mul<Zr<Curve>> for &G2Vec<Curve>
where
    Curve: Pairing,
    G2<Curve>: Copy,
    G2<Curve>: Mul<Zr<Curve>, Output = G2<Curve>>,
{
    type Output = G2Vec<Curve>;

    fn mul(self, rhs: Zr<Curve>) -> Self::Output {
        G2Vec(self.0.iter().map(|val| *val * rhs).collect())
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Bn254;
    use ark_ff::UniformRand;

    use super::InnerProd;

    use super::{
        super::{G1Vec, G1, G2},
        e, mul_gt,
    };

    #[test]
    fn test_inner_prod() {
        let mut rng = ark_std::test_rng();
        let g1a = G1::<Bn254>::rand(&mut rng);
        let g1b = G1::<Bn254>::rand(&mut rng);
        let g1c = G1::<Bn254>::rand(&mut rng);

        let g2a = G2::<Bn254>::rand(&mut rng);
        let g2b = G2::<Bn254>::rand(&mut rng);
        let g2c = G2::<Bn254>::rand(&mut rng);

        let expected = mul_gt(&[e(g1a, g2a), e(g1b, g2b), e(g1c, g2c)]).unwrap();

        let g1v = &[g1a, g1b, g1c];
        let g1v: G1Vec<Bn254> = g1v.into();
        let g2v = &[g2a, g2b, g2c];
        let g2v = g2v.into();

        let actual = g1v.inner_prod(&g2v).unwrap();

        assert_eq!(expected, actual);
    }
}
