//! Admitted one-hot family shapes, independent of schedule catalog inventories.

use akita_pcs::AkitaError;

use crate::configs::AkitaChunkProfile;

pub const AKITA_ONE_HOT_K16: usize = 16;
pub const AKITA_ONE_HOT_K256: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OneHotFamilyContract {
    K16(AkitaChunkProfile),
    K256(AkitaChunkProfile),
}

impl OneHotFamilyContract {
    pub fn from_parts(k: usize, profile: AkitaChunkProfile) -> Result<Self, AkitaError> {
        match k {
            AKITA_ONE_HOT_K16 => Ok(Self::K16(profile)),
            AKITA_ONE_HOT_K256 => Ok(Self::K256(profile)),
            _ => Err(AkitaError::InvalidSetup(format!(
                "unsupported one-hot K={k}"
            ))),
        }
    }

    pub fn num_vars_range(self, num_polys: usize) -> (usize, usize) {
        let (base_min, max, profile) = match self {
            Self::K16(profile) => (12, 40, profile),
            Self::K256(profile) => (12, 43, profile),
        };
        // Two-polynomial rows have more live blocks for the two chunked levels.
        let min = match (profile.chunks, profile.levels, num_polys) {
            (4, 2, 2) => base_min,
            (4, 2, _) | (8, 2, 2) => base_min + 1,
            (8, 2, _) => base_min + 2,
            // The catalog audit checks each fold against the requested geometry.
            _ => base_min,
        };
        (min, max)
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "frozen admission fixtures use supported families"
)]
mod tests {
    use super::*;

    #[test]
    fn profile_bounds_follow_the_admitted_family_contract() {
        for (profile, one_poly_min, two_poly_min) in [
            (AkitaChunkProfile::new(1, 0), 12, 12),
            (AkitaChunkProfile::new(2, 2), 12, 12),
            (AkitaChunkProfile::new(4, 2), 13, 12),
            (AkitaChunkProfile::new(8, 2), 14, 13),
        ] {
            for (k, max) in [(16, 40), (256, 43)] {
                let contract = OneHotFamilyContract::from_parts(k, profile).unwrap();
                assert_eq!(contract.num_vars_range(1), (one_poly_min, max));
                assert_eq!(contract.num_vars_range(2), (two_poly_min, max));
            }
        }
        assert!(OneHotFamilyContract::from_parts(64, AkitaChunkProfile::new(1, 0)).is_err());
    }
}
