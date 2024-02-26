use ark_bn254::Fr as ArkFr;
use ark_ff::{fields::PrimeField as ArkPrimeField, BigInteger};

use ff::PrimeField as GenericPrimeField;
use halo2curves::serde::SerdeObject;
use spartan2::provider::bn256_grumpkin::bn256::Scalar as Spartan2Fr;

pub fn ark_to_spartan<ArkF: ArkPrimeField>(ark: ArkF) -> Spartan2Fr {
    let bigint: <ArkF as ArkPrimeField>::BigInt = ark.into_bigint();
    let bytes = bigint.to_bytes_le();
    let mut array = [0u64; 4];
    for (i, chunk) in bytes.chunks(8).enumerate() {
        array[i] = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    Spartan2Fr::from_raw(array)
}

pub fn spartan_to_ark(bell: Spartan2Fr) -> ArkFr {
    let bytes = bell.to_repr();
    ArkFr::from_le_bytes_mod_order(&bytes)
}

pub fn ark_to_ff<FF: GenericPrimeField<Repr = [u8; 32]>, AF: ArkPrimeField>(ark: AF) -> FF {
    let repr = ark.into_bigint();
    let bytes: Vec<u8> = repr.to_bytes_le();
    let bytes: [u8; 32] = bytes.try_into().unwrap();

    GenericPrimeField::from_repr(bytes).unwrap()
}

pub fn ff_to_ark<FF: GenericPrimeField<Repr = [u8; 32]>, AF: ArkPrimeField>(ff: FF) -> AF {
    let bytes = ff.to_repr();
    AF::from_le_bytes_mod_order(&bytes)
}

pub fn ff_to_ruint<FF: GenericPrimeField<Repr = [u8; 32]>>(ff: FF) -> ruint::aliases::U256 {
    let bytes  = ff.to_repr();
    let bytes = bytes.as_ref();
    let bi: [u8; 32] = bytes.try_into().unwrap();
    ruint::aliases::U256::from_le_bytes(bi)
}

pub fn ruint_to_ff<FF: GenericPrimeField<Repr = [u8; 32]>>(ruint: ruint::aliases::U256) -> FF {
    let bytes: [u8; 32] = ruint.to_le_bytes().try_into().expect("should be 256 bits");
    FF::from_repr(bytes).unwrap()
}

pub fn ff_to_ruints<FF: GenericPrimeField<Repr = [u8; 32]>>(ff: Vec<FF>) -> Vec<ruint::aliases::U256> {
    ff.into_iter().map(|f| ff_to_ruint(f)).collect()
}

pub fn ark_to_spartan_unsafe<AF: ArkPrimeField, FF: GenericPrimeField<Repr = [u8; 32]>>(ark: AF) -> FF {
    assert_eq!(std::mem::size_of::<AF>(), 32);
    assert_eq!(std::mem::size_of::<FF>(), 32);
    let ff: FF;
    unsafe {
        let inner = access_ark_private(&ark);
        ff = std::mem::transmute_copy(&inner);
    }
    ff
}

pub fn spartan_to_ark_unsafe<FF: GenericPrimeField<Repr = [u8; 32]>, AF: ArkPrimeField>(ff: FF) -> AF {
    assert_eq!(std::mem::size_of::<FF>(), 32);
    assert_eq!(std::mem::size_of::<AF>(), 32);
    let ark: AF;
    unsafe {
        let inner = access_spartan_private(&ff);
        ark = std::mem::transmute_copy(&inner);
    }
    ark
}


unsafe fn access_ark_private<AF: ArkPrimeField>(value: &AF) -> [u8; 32] {
    let ptr = value as *const AF as *const [u8; 32];
    *ptr
}

unsafe fn access_spartan_private<FF: GenericPrimeField<Repr = [u8; 32]>>(value: &FF) -> [u8; 32] {
    let ptr = value as *const FF as *const [u8; 32];
    *ptr
}

#[cfg(test)]
mod tests {
    use ark_std::rand;

    use super::*;

    fn random_spartan() -> Spartan2Fr {
        let rand_vec: Vec<u64> = std::iter::repeat_with(|| rand::random::<u64>()).take(4).collect();
        let random: [u64; 4] = rand_vec.try_into().unwrap();
        Spartan2Fr::from_raw(random)
    }

    fn random_ark() -> ArkFr {
        let random: Vec<u8> = std::iter::repeat_with(|| rand::random::<u8>()).take(32).collect();
        ArkFr::from_be_bytes_mod_order(&random)
    }

    #[test]
    fn test_ark_to_bell() {
        let ark_value = ArkFr::from(5);
        let spartan_value = Spartan2Fr::from(5);
        assert_eq!(ark_to_spartan(ark_value), spartan_value);

        let mut ark_value = ArkFr::from(13);
        ark_value *= ArkFr::from(101);
        let mut spartan_value = Spartan2Fr::from(13);
        spartan_value *= Spartan2Fr::from(101);
        assert_eq!(ark_to_spartan(ark_value), spartan_value);
    }

    #[test]
    fn test_bell_to_ark() {
        let spartan_value = Spartan2Fr::from(5);
        let ark_value = ArkFr::from(5);
        assert_eq!(spartan_to_ark(spartan_value), ark_value);

        let mut spartan_value = Spartan2Fr::from(13);
        spartan_value *= Spartan2Fr::from(101);
        let mut ark_value = ArkFr::from(13);
        ark_value *= ArkFr::from(101);
        assert_eq!(spartan_to_ark(spartan_value), ark_value);
    }

    #[test]
    fn test_round_trip_conversion() {
        for _ in 0..10 {
            let ark_value = random_ark();
            let new_ark_value = spartan_to_ark(ark_to_spartan(ark_value));
            assert_eq!(ark_value, new_ark_value);
        }

        for _ in 0..10 {
            let spartan_value = random_spartan();
            let new_spartan_value = ark_to_spartan(spartan_to_ark(spartan_value));

            assert_eq!(spartan_value, new_spartan_value);
        }
    }

    #[test]
    fn test_generic_ark_to_ff() {
        for _ in 0..10 {
            let random_u64 = rand::random::<u64>();
            let ark_value = ArkFr::from(random_u64);
            let spartan_value = Spartan2Fr::from(random_u64);
            let new_spartan_value = ark_to_ff::<Spartan2Fr, ArkFr>(ark_value);

            assert_eq!(spartan_value, new_spartan_value);
        }
    }

    #[test]
    fn test_round_trip_conversion_generic() {
        for _ in 0..10 {
            let ark_value = random_ark();
            let new_ark_value =
                ff_to_ark::<Spartan2Fr, ArkFr>(ark_to_ff::<Spartan2Fr, ArkFr>(ark_value));
            assert_eq!(ark_value, new_ark_value);
        }
        for _ in 0..10 {
            let spartan_value = random_spartan();
            let new_spartan_value =
                ark_to_ff::<Spartan2Fr, ArkFr>(ff_to_ark::<Spartan2Fr, ArkFr>(spartan_value));

            assert_eq!(spartan_value, new_spartan_value);
        }
    }

    #[test]
    fn unsafe_roundtrip_test() {
        for _ in 0..100_000 {
            let spartan_value = random_spartan();
            let new_spartan_value = ark_to_spartan_unsafe(spartan_to_ark_unsafe::<Spartan2Fr, ArkFr>(spartan_value));
            assert_eq!(spartan_value, new_spartan_value);
        }

        let ark_value = ArkFr::from(-1);
        let new_ark_value = spartan_to_ark_unsafe(ark_to_spartan_unsafe::<ArkFr, Spartan2Fr>(ark_value));
        assert_eq!(ark_value, new_ark_value);

        for _ in 0..100_000 {
            let ark_value = random_ark();
            let new_ark_value = spartan_to_ark_unsafe(ark_to_spartan_unsafe::<ArkFr, Spartan2Fr>(ark_value));
            assert_eq!(ark_value, new_ark_value);
        }
    }

}
