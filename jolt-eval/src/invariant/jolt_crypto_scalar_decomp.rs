//! Scalar decomposition reconstruction invariants for `jolt-crypto`.
//!
//! The 2D and 4D GLV decompositions are correct iff the `scalar_mul` path
//! (which uses them) matches an independent MSM-based reference. Arkworks'
//! Pippenger MSM does not go through the GLV decomposition tables, so
//! `G.scalar_mul(&s) == Bn254G1::msm(&[G], &[s])` is an independent check
//! that catches any divergence in `decompose_scalar_{2d,4d}`.

use arbitrary::{Arbitrary, Unstructured};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use jolt_crypto::{Bn254, Bn254G1, Bn254G2, JoltGroup};
use jolt_field::{Field, Fr};

use super::{CheckError, Invariant, InvariantViolation};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ScalarDecompInput {
    pub seed: u64,
}

impl<'a> Arbitrary<'a> for ScalarDecompInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self {
            seed: u.arbitrary()?,
        })
    }
}

#[jolt_eval_macros::invariant(Test, Fuzz)]
#[derive(Default)]
pub struct JoltCryptoScalarDecompInvariant;

impl Invariant for JoltCryptoScalarDecompInvariant {
    type Setup = ();
    type Input = ScalarDecompInput;

    fn name(&self) -> &str {
        "jolt_crypto_scalar_decomp_reconstructs"
    }

    fn description(&self) -> String {
        "2D and 4D GLV decompositions must reconstruct the scalar exactly. \
         Verified by comparing G.scalar_mul (GLV path) against G::msm (Pippenger path). \
         A decomposition bug in decompose_scalar_{2d,4d} causes these to diverge."
            .to_string()
    }

    fn setup(&self) {}

    fn check(&self, _setup: &(), input: ScalarDecompInput) -> Result<(), CheckError> {
        let mut rng = ChaCha20Rng::seed_from_u64(input.seed);
        let scalar = Fr::random(&mut rng);

        // 2D decomposition: G1 scalar_mul uses decompose_scalar_2d.
        let g1 = Bn254::random_g1(&mut rng);
        let glv = g1.scalar_mul(&scalar);
        let via_msm = Bn254G1::msm(&[g1], &[scalar]);
        if glv != via_msm {
            return Err(CheckError::Violation(InvariantViolation::with_details(
                "G1 scalar_mul ≠ G1 msm (2D GLV decomposition bug)",
                format!("seed={}", input.seed),
            )));
        }

        // 4D decomposition: G2 scalar_mul uses decompose_scalar_4d.
        let g2 = Bn254::g2_generator().scalar_mul(&Fr::random(&mut rng));
        let glv = g2.scalar_mul(&scalar);
        let via_msm = Bn254G2::msm(&[g2], &[scalar]);
        if glv != via_msm {
            return Err(CheckError::Violation(InvariantViolation::with_details(
                "G2 scalar_mul ≠ G2 msm (4D GLV decomposition bug)",
                format!("seed={}", input.seed),
            )));
        }

        Ok(())
    }

    fn seed_corpus(&self) -> Vec<ScalarDecompInput> {
        (0..12)
            .map(|i| ScalarDecompInput {
                seed: i as u64 + 1000,
            })
            .collect()
    }
}
