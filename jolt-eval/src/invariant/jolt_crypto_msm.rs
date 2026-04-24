//! MSM-vs-naive invariants for `jolt-crypto` (G1, G2, GT).

use arbitrary::{Arbitrary, Unstructured};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use jolt_crypto::{Bn254, Bn254G1, Bn254G2, Bn254GT, JoltGroup, PairingGroup};
use jolt_field::{Field, Fr};

use super::{CheckError, Invariant, InvariantViolation};

const MAX_LEN: usize = 256;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct MsmInput {
    pub seed: u64,
    pub len: u16,
}

impl<'a> Arbitrary<'a> for MsmInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self {
            seed: u.arbitrary()?,
            len: u.int_in_range(0u16..=MAX_LEN as u16)?,
        })
    }
}

fn random_fr(rng: &mut ChaCha20Rng) -> Fr {
    Fr::random(rng)
}

fn naive_msm<G: JoltGroup>(bases: &[G], scalars: &[Fr]) -> G {
    let mut acc = G::identity();
    for (b, s) in bases.iter().zip(scalars.iter()) {
        acc += b.scalar_mul(s);
    }
    acc
}

#[jolt_eval_macros::invariant(Test, Fuzz)]
#[derive(Default)]
pub struct JoltCryptoMsmInvariant;

impl Invariant for JoltCryptoMsmInvariant {
    type Setup = ();
    type Input = MsmInput;

    fn name(&self) -> &str {
        "jolt_crypto_msm_matches_naive"
    }

    fn description(&self) -> String {
        "Bn254G1::msm / Bn254G2::msm / Bn254GT::msm must equal the naive \
         Σ bases[i].scalar_mul(&scalars[i]) for random inputs of length 0..=256."
            .to_string()
    }

    fn setup(&self) {}

    fn check(&self, _setup: &(), input: MsmInput) -> Result<(), CheckError> {
        let n = input.len as usize;
        let mut rng = ChaCha20Rng::seed_from_u64(input.seed);

        let scalars: Vec<Fr> = (0..n).map(|_| random_fr(&mut rng)).collect();

        // G1
        let g1_bases: Vec<Bn254G1> = (0..n).map(|_| Bn254::random_g1(&mut rng)).collect();
        let msm_g1 = Bn254G1::msm(&g1_bases, &scalars);
        let naive_g1 = naive_msm(&g1_bases, &scalars);
        if msm_g1 != naive_g1 {
            return Err(CheckError::Violation(InvariantViolation::with_details(
                "G1 MSM mismatch",
                format!("len={n}, seed={}", input.seed),
            )));
        }

        // G2 — derive bases from generator to avoid needing a random_g2 helper.
        let g2 = Bn254::g2_generator();
        let g2_bases: Vec<Bn254G2> = (0..n)
            .map(|_| g2.scalar_mul(&random_fr(&mut rng)))
            .collect();
        let msm_g2 = Bn254G2::msm(&g2_bases, &scalars);
        let naive_g2 = naive_msm(&g2_bases, &scalars);
        if msm_g2 != naive_g2 {
            return Err(CheckError::Violation(InvariantViolation::with_details(
                "G2 MSM mismatch",
                format!("len={n}, seed={}", input.seed),
            )));
        }

        // GT — cap to 8 for CPU budget; GT pow is slow and a win in opt (3)
        // would not change correctness for short MSMs.
        let gt_len = n.min(8);
        let gt_base = Bn254::pairing(&Bn254::g1_generator(), &g2);
        let gt_bases: Vec<Bn254GT> = (0..gt_len)
            .map(|i| gt_base.scalar_mul(&Fr::from_u64(i as u64 + 1)))
            .collect();
        let gt_scalars: Vec<Fr> = scalars.iter().take(gt_len).copied().collect();
        let msm_gt = Bn254GT::msm(&gt_bases, &gt_scalars);
        let naive_gt = naive_msm(&gt_bases, &gt_scalars);
        if msm_gt != naive_gt {
            return Err(CheckError::Violation(InvariantViolation::with_details(
                "GT MSM mismatch",
                format!("len={gt_len}, seed={}", input.seed),
            )));
        }

        Ok(())
    }

    fn seed_corpus(&self) -> Vec<MsmInput> {
        [0u16, 1, 2, 3, 15, 16, 17, 63, 64, 65, 255]
            .into_iter()
            .enumerate()
            .map(|(i, len)| MsmInput {
                seed: i as u64 + 1,
                len,
            })
            .collect()
    }
}
