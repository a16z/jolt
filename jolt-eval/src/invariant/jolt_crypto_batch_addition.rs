//! `batch_g1_additions_multi` vs naive-sum invariant.

use std::collections::HashSet;

use arbitrary::{Arbitrary, Unstructured};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use jolt_crypto::ec::bn254::batch_addition::batch_g1_additions_multi;
use jolt_crypto::{Bn254, Bn254G1, JoltGroup};

use super::{CheckError, Invariant, InvariantViolation};

const BASE_SIZE: usize = 64;
const MAX_SETS: usize = 8;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct BatchAdditionInput {
    pub seed: u64,
    pub num_sets: u8,
}

impl<'a> Arbitrary<'a> for BatchAdditionInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self {
            seed: u.arbitrary()?,
            num_sets: u.int_in_range(0u8..=MAX_SETS as u8)?,
        })
    }
}

#[jolt_eval_macros::invariant(Test, Fuzz)]
#[derive(Default)]
pub struct JoltCryptoBatchAdditionInvariant;

impl Invariant for JoltCryptoBatchAdditionInvariant {
    type Setup = ();
    type Input = BatchAdditionInput;

    fn name(&self) -> &str {
        "jolt_crypto_batch_addition_matches_naive"
    }

    fn description(&self) -> String {
        "batch_g1_additions_multi(&bases, &indices_sets)[k] must equal the \
         naive Σ_{i ∈ indices_sets[k]} bases[i] for any inputs with \
         distinct x-coordinates per pair."
            .to_string()
    }

    fn setup(&self) {}

    fn check(&self, _setup: &(), input: BatchAdditionInput) -> Result<(), CheckError> {
        let num_sets = input.num_sets as usize;
        let mut rng = ChaCha20Rng::seed_from_u64(input.seed);

        let bases: Vec<Bn254G1> = (0..BASE_SIZE).map(|_| Bn254::random_g1(&mut rng)).collect();

        // Draw unique indices per set to avoid the distinct-x precondition.
        let indices_sets: Vec<Vec<usize>> = (0..num_sets)
            .map(|_| {
                let size = (rng.next_u32() as usize) % 20 + 1;
                let mut picked: HashSet<usize> = HashSet::new();
                while picked.len() < size {
                    let _ = picked.insert((rng.next_u32() as usize) % BASE_SIZE);
                }
                picked.into_iter().collect()
            })
            .collect();

        let results = batch_g1_additions_multi(&bases, &indices_sets);
        if results.len() != indices_sets.len() {
            return Err(CheckError::Violation(InvariantViolation::with_details(
                "batch addition result count mismatch",
                format!("want {}, got {}", indices_sets.len(), results.len()),
            )));
        }

        for (k, (indices, got)) in indices_sets.iter().zip(results.iter()).enumerate() {
            let mut expected = Bn254G1::identity();
            for &i in indices {
                expected = expected + bases[i];
            }
            if *got != expected {
                return Err(CheckError::Violation(InvariantViolation::with_details(
                    "batch addition sum mismatch",
                    format!("set={k}, |set|={}", indices.len()),
                )));
            }
        }

        Ok(())
    }

    fn seed_corpus(&self) -> Vec<BatchAdditionInput> {
        [0u8, 1, 2, 3, 4, 5, 8]
            .into_iter()
            .enumerate()
            .map(|(i, n)| BatchAdditionInput {
                seed: i as u64 + 200,
                num_sets: n,
            })
            .collect()
    }
}
