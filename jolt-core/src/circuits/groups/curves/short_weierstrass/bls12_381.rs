use crate::circuits::groups::curves::short_weierstrass::ProjectiveVar;
use ark_bls12_381::{g1, Fq, Fr};
use ark_r1cs_std::fields::nonnative::NonNativeFieldVar;

pub type FBaseVar = NonNativeFieldVar<Fq, Fr>;

pub type G1Var = ProjectiveVar<g1::Config, Fr, FBaseVar>;
