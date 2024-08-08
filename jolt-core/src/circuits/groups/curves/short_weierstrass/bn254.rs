use crate::circuits::groups::curves::short_weierstrass::ProjectiveVar;
use ark_bn254::{Bn254, Fq, Fr};
use ark_r1cs_std::fields::nonnative::NonNativeFieldVar;

pub type FBaseVar = NonNativeFieldVar<Fq, Fr>;

pub type G1Var = ProjectiveVar<ark_bn254::g1::Config, Fr, FBaseVar>;
