use ark_bn254::Fr as ArkFr;
use ark_ff::{fields::PrimeField as ArkPrimeField, BigInteger};



use ff::PrimeField as GenericPrimeField;
use spartan2::provider::bn256_grumpkin::bn256::Base as Spartan2Fr;

pub fn ark_to_spartan(ark: ArkFr) -> Spartan2Fr {
    let bigint = ark.into_bigint();
    Spartan2Fr::from_raw(bigint.0)
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

#[cfg(test)]
mod tests {
    use ark_std::{rand, UniformRand};

    use super::*;

    fn random_spartan() -> Spartan2Fr {
        let random = [rand::random::<u64>(); 4];
        Spartan2Fr::from_raw(random)
    }

    fn random_ark() -> ArkFr {
        let random = [rand::random::<u8>(); 32];
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
}
