use std::fmt;

use ark_bn254::Fq as Fp;
use ark_bn254::Fr as Scalar;
use ark_ec::AdditiveGroup;
use ark_ff::Field;
use ark_ff::PrimeField;
use num_bigint::BigUint;

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fqq {
    pub element: Scalar,
    pub limbs: [Fp; 3],
}

pub fn convert_from_3_limbs(limbs: Vec<Fp>) -> Scalar {
    let r = Scalar::from(BigUint::from(limbs[0].into_bigint()))
        + Scalar::from(2u8).pow([(125) as u64, 0, 0, 0]) * Scalar::from(limbs[1].into_bigint())
        + Scalar::from(2u8).pow([(250) as u64, 0, 0, 0]) * Scalar::from(limbs[2].into_bigint());
    r
}

pub fn convert_to_3_limbs(r: Scalar) -> [Fp; 3] {
    let mut limbs = [Fp::ZERO; 3];

    let mask = BigUint::from((1u128 << 125) - 1);

    limbs[0] = Fp::from(BigUint::from(r.into_bigint()) & mask.clone());

    limbs[1] = Fp::from((BigUint::from(r.into_bigint()) >> 125) & mask.clone());

    limbs[2] = Fp::from((BigUint::from(r.into_bigint()) >> 250) & mask.clone());

    limbs
}

pub fn convert_to_fqq(r: &Scalar) -> Fqq {
    Fqq {
        element: *r,
        limbs: convert_to_3_limbs(*r),
    }
}

pub fn convert_vec_to_fqq(r: &Vec<Scalar>) -> Vec<Fqq> {
    r.iter().map(|val| convert_to_fqq(val)).collect()
}
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct FqLimb {
    pub limbs: [Fp; 3],
}

impl fmt::Debug for Fqq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                              "element": "{}",
                              "limbs": ["{}", "{}", "{}"]
                              
                              }}"#,
            self.element,
            &self.limbs[0],
            &self.limbs[1].to_string(),
            &self.limbs[2].to_string()
        )
    }
}

impl fmt::Debug for FqLimb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"[[
            "{}", "{}", "{}"
            ]]"#,
            self.limbs[0], self.limbs[1], self.limbs[2]
        )
    }
}
