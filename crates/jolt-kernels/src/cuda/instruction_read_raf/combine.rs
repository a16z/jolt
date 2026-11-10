use jolt_field::{Fr, One, Zero};
use jolt_lookup_tables::tables::prefixes::{PrefixEval, Prefixes, ALL_PREFIXES};
use jolt_lookup_tables::tables::LookupTableKind;

use crate::cuda::common::error::CudaError;

pub const RISCV_XLEN: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CombineTerm {
    pub scale: Fr,
    pub prefix: Option<Prefixes>,
    pub suffix: usize,
}

/// Recover the bilinear prefix/suffix coefficients from the canonical table
/// formulas. These small descriptors are uploaded once per address phase.
pub fn combine_terms(table: LookupTableKind<RISCV_XLEN>) -> Result<Vec<CombineTerm>, CudaError> {
    let mut prefixes = vec![PrefixEval::from(Fr::zero()); ALL_PREFIXES.len()];
    let mut suffixes = vec![Fr::zero(); table.suffixes().len()];
    if !table.combine(&prefixes, &suffixes).is_zero() {
        return Err(CudaError::InvariantViolation {
            reason: "a lookup combination has a suffix-independent term",
        });
    }
    let mut terms = Vec::new();
    for suffix in 0..suffixes.len() {
        suffixes[suffix] = Fr::one();
        let standalone = table.combine(&prefixes, &suffixes);
        if !standalone.is_zero() {
            terms.push(CombineTerm {
                scale: standalone,
                prefix: None,
                suffix,
            });
        }
        for &prefix in ALL_PREFIXES {
            prefixes[prefix as usize] = Fr::one().into();
            let scale = table.combine(&prefixes, &suffixes) - standalone;
            prefixes[prefix as usize] = Fr::zero().into();
            if !scale.is_zero() {
                terms.push(CombineTerm {
                    scale,
                    prefix: Some(prefix),
                    suffix,
                });
            }
        }
        suffixes[suffix] = Fr::zero();
    }
    Ok(terms)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use super::ALL_PREFIXES;
    use jolt_field::{Fr, Ring};
    use jolt_lookup_tables::tables::prefixes::PrefixEval;
    use jolt_lookup_tables::tables::LookupTableKind;
    use strum::IntoEnumIterator;

    use super::{combine_terms, CombineTerm, RISCV_XLEN};
    use crate::cuda::common::testing::fr;

    const NUM_PREFIXES: usize = ALL_PREFIXES.len();

    fn present_tables() -> Vec<LookupTableKind<RISCV_XLEN>> {
        <LookupTableKind<RISCV_XLEN> as IntoEnumIterator>::iter().collect()
    }

    fn host_from_terms(terms: &[CombineTerm], prefixes: &[Fr], suffixes: &[Fr]) -> Fr {
        terms.iter().fold(Fr::from_u64(0), |acc, term| {
            let prefix = term
                .prefix
                .map_or(Fr::from_u64(1), |prefix| prefixes[prefix as usize]);
            acc + term.scale * prefix * suffixes[term.suffix]
        })
    }

    #[test]
    fn combine_terms_reproduce_every_table_combine() {
        let mut prefixes: Vec<Fr> = (0..NUM_PREFIXES).map(|i| fr(i as u64 + 11)).collect();
        prefixes[0] = Fr::from_u64(0);
        let wrapped: Vec<PrefixEval<Fr>> = prefixes.iter().copied().map(PrefixEval::from).collect();

        for table in present_tables() {
            let suffix_kinds = table.suffixes();
            let suffix_values: Vec<Fr> = (0..suffix_kinds.len())
                .map(|i| fr(i as u64 + 101))
                .collect();
            let expected = table.combine::<Fr>(&wrapped, &suffix_values);

            let terms = combine_terms(table).expect("combine terms");
            let got = host_from_terms(&terms, &prefixes, &suffix_values);
            assert_eq!(
                got, expected,
                "term list for {table:?} does not reproduce combine"
            );
        }
    }
}
