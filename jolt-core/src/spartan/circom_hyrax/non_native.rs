use ark_ff::AdditiveGroup;
use ark_ff::Field;
use ark_ff::PrimeField;
use ark_grumpkin::{Fq as Fp, Fr as Scalar};
use num_bigint::BigUint;
use std::fmt;

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fqq {
    pub element: Scalar,
    pub limbs: [Fp; 3],
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
