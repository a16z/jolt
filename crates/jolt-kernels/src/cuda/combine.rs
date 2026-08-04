#![expect(
    dead_code,
    reason = "implementation target: the device address phase wires this once its kernels land"
)]

use cudarc::driver::PushKernelArg;
use jolt_lookup_tables::tables::prefixes::Prefixes;
use jolt_lookup_tables::tables::suffixes::Suffixes;
use jolt_lookup_tables::tables::LookupTableKind;

use super::context::CudaKernelContext;
use super::device::DeviceFrVec;
use super::error::CudaError;

pub const RISCV_XLEN: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scale {
    One,
    NegOne,
    TwoPowXlen,
    XlenOnes,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CombineTerm {
    pub scale: Scale,
    pub prefix: Option<Prefixes>,
    pub suffix: usize,
}

pub fn combine_terms(table: LookupTableKind<RISCV_XLEN>) -> Result<Vec<CombineTerm>, CudaError> {
    use LookupTableKind as K;
    use Prefixes as P;
    use Scale::{NegOne, One as S1, TwoPowXlen, XlenOnes};

    let slot = |kind: Suffixes| -> usize {
        table
            .suffixes()
            .iter()
            .position(|candidate| *candidate == kind)
            .unwrap_or(usize::MAX)
    };
    let one = slot(Suffixes::One);

    let term = |scale: Scale, prefix: Option<Prefixes>, suffix: usize| CombineTerm {
        scale,
        prefix,
        suffix,
    };

    let terms = match table {
        K::And(_) => vec![
            term(S1, Some(P::And), one),
            term(S1, None, slot(Suffixes::And)),
        ],
        K::Andn(_) => vec![
            term(S1, Some(P::Andn), one),
            term(S1, None, slot(Suffixes::AndNot)),
        ],
        K::Or(_) => vec![
            term(S1, Some(P::Or), one),
            term(S1, None, slot(Suffixes::Or)),
        ],
        K::Xor(_) => vec![
            term(S1, Some(P::Xor), one),
            term(S1, None, slot(Suffixes::Xor)),
        ],
        K::Equal(_) => vec![term(S1, Some(P::Eq), slot(Suffixes::Eq))],
        K::NotEqual(_) => vec![
            term(S1, None, one),
            term(NegOne, Some(P::Eq), slot(Suffixes::Eq)),
        ],
        K::HalfwordAlignment(_) => vec![
            term(S1, None, one),
            term(NegOne, Some(P::Lsb), slot(Suffixes::Lsb)),
        ],
        K::WordAlignment(_) => vec![term(S1, Some(P::TwoLsb), slot(Suffixes::TwoLsb))],
        K::LowerHalfWord(_) => vec![
            term(S1, Some(P::LowerHalfWord), one),
            term(S1, None, slot(Suffixes::LowerHalfWord)),
        ],
        K::MulUNoOverflow(_) => vec![term(
            S1,
            Some(P::OverflowBitsZero),
            slot(Suffixes::OverflowBitsZero),
        )],
        K::Pow2(_) => vec![term(S1, Some(P::Pow2), slot(Suffixes::Pow2))],
        K::Pow2W(_) => vec![term(S1, Some(P::Pow2W), slot(Suffixes::Pow2W))],
        K::RangeCheck(_) => vec![
            term(S1, Some(P::LowerWord), one),
            term(S1, None, slot(Suffixes::LowerWord)),
        ],
        K::RangeCheckAligned(_) => vec![
            term(S1, Some(P::LowerWord), one),
            term(S1, None, slot(Suffixes::LowerWord)),
            term(NegOne, Some(P::Lsb), slot(Suffixes::Lsb)),
        ],
        K::ShiftRightBitmask(_) => vec![
            term(TwoPowXlen, None, one),
            term(NegOne, Some(P::Pow2), slot(Suffixes::Pow2)),
        ],
        K::SignMask(_) => vec![term(XlenOnes, Some(P::LeftOperandMsb), one)],
        K::SignedGreaterThanEqual(_) => vec![
            term(S1, None, one),
            term(S1, Some(P::RightOperandMsb), one),
            term(NegOne, Some(P::LeftOperandMsb), one),
            term(NegOne, Some(P::LessThan), one),
            term(NegOne, Some(P::Eq), slot(Suffixes::LessThan)),
        ],
        K::SignedLessThan(_) => vec![
            term(S1, Some(P::LeftOperandMsb), one),
            term(NegOne, Some(P::RightOperandMsb), one),
            term(S1, Some(P::LessThan), one),
            term(S1, Some(P::Eq), slot(Suffixes::LessThan)),
        ],
        K::UnsignedGreaterThanEqual(_) => vec![
            term(S1, None, one),
            term(NegOne, Some(P::LessThan), one),
            term(NegOne, Some(P::Eq), slot(Suffixes::LessThan)),
        ],
        K::UnsignedLessThan(_) => vec![
            term(S1, Some(P::LessThan), one),
            term(S1, Some(P::Eq), slot(Suffixes::LessThan)),
        ],
        K::UnsignedLessThanEqual(_) => vec![
            term(S1, Some(P::LessThan), one),
            term(S1, Some(P::Eq), slot(Suffixes::LessThan)),
            term(S1, Some(P::Eq), slot(Suffixes::Eq)),
        ],
        K::SignExtendHalfWord(_) => vec![
            term(S1, Some(P::LowerHalfWord), one),
            term(S1, None, slot(Suffixes::LowerHalfWord)),
            term(
                S1,
                Some(P::SignExtensionUpperHalf),
                slot(Suffixes::SignExtensionUpperHalf),
            ),
        ],
        K::UpperWord(_) => vec![
            term(S1, Some(P::UpperWord), one),
            term(S1, None, slot(Suffixes::UpperWord)),
        ],
        K::ValidDiv0(_) => vec![
            term(S1, None, one),
            term(
                NegOne,
                Some(P::LeftOperandIsZero),
                slot(Suffixes::LeftOperandIsZero),
            ),
            term(S1, Some(P::DivByZero), slot(Suffixes::DivByZero)),
        ],
        K::ValidUnsignedRemainder(_) => vec![
            term(
                S1,
                Some(P::RightOperandIsZero),
                slot(Suffixes::RightOperandIsZero),
            ),
            term(S1, Some(P::LessThan), one),
            term(S1, Some(P::Eq), slot(Suffixes::LessThan)),
        ],
        K::VirtualChangeDivisor(_) => vec![
            term(S1, Some(P::RightOperand), one),
            term(S1, None, slot(Suffixes::RightOperand)),
            term(S1, Some(P::ChangeDivisor), slot(Suffixes::ChangeDivisor)),
        ],
        K::VirtualChangeDivisorW(_) => vec![
            term(S1, Some(P::RightOperandW), one),
            term(S1, None, slot(Suffixes::RightOperandW)),
            term(S1, Some(P::ChangeDivisorW), slot(Suffixes::ChangeDivisorW)),
            term(
                S1,
                Some(P::SignExtensionRightOperand),
                slot(Suffixes::SignExtensionRightOperand),
            ),
        ],
        K::VirtualRev8W(_) => vec![
            term(S1, Some(P::Rev8W), one),
            term(S1, None, slot(Suffixes::Rev8W)),
        ],
        K::VirtualSRL(_) => vec![
            term(S1, Some(P::RightShift), slot(Suffixes::RightShiftHelper)),
            term(S1, None, slot(Suffixes::RightShift)),
        ],
        K::VirtualSRA(_) => vec![
            term(S1, Some(P::RightShift), slot(Suffixes::RightShiftHelper)),
            term(S1, None, slot(Suffixes::RightShift)),
            term(S1, Some(P::LeftOperandMsb), slot(Suffixes::SignExtension)),
            term(S1, Some(P::SignExtension), one),
        ],
        K::VirtualROTR(_) => vec![
            term(S1, Some(P::RightShift), slot(Suffixes::RightShiftHelper)),
            term(S1, None, slot(Suffixes::RightShift)),
            term(S1, Some(P::LeftShiftHelper), slot(Suffixes::LeftShift)),
            term(S1, Some(P::LeftShift), one),
        ],
        K::VirtualROTRW(_) => vec![
            term(S1, Some(P::RightShiftW), slot(Suffixes::RightShiftWHelper)),
            term(S1, None, slot(Suffixes::RightShiftW)),
            term(S1, Some(P::LeftShiftWHelper), slot(Suffixes::LeftShiftW)),
            term(S1, Some(P::LeftShiftW), one),
        ],
        K::VirtualXORROT16(_) => xor_rot_terms(P::XorRot16, one, slot(Suffixes::XorRot16)),
        K::VirtualXORROT24(_) => xor_rot_terms(P::XorRot24, one, slot(Suffixes::XorRot24)),
        K::VirtualXORROT32(_) => xor_rot_terms(P::XorRot32, one, slot(Suffixes::XorRot32)),
        K::VirtualXORROT63(_) => xor_rot_terms(P::XorRot63, one, slot(Suffixes::XorRot63)),
        K::VirtualXORROTW7(_) => xor_rot_terms(P::XorRotW7, one, slot(Suffixes::XorRotW7)),
        K::VirtualXORROTW8(_) => xor_rot_terms(P::XorRotW8, one, slot(Suffixes::XorRotW8)),
        K::VirtualXORROTW12(_) => xor_rot_terms(P::XorRotW12, one, slot(Suffixes::XorRotW12)),
        K::VirtualXORROTW16(_) => xor_rot_terms(P::XorRotW16, one, slot(Suffixes::XorRotW16)),
    };

    if terms.iter().any(|term| term.suffix == usize::MAX) {
        return Err(CudaError::InvariantViolation {
            reason: "a combine term references a suffix the table does not declare",
        });
    }
    Ok(terms)
}

fn xor_rot_terms(prefix: Prefixes, one: usize, rot: usize) -> Vec<CombineTerm> {
    vec![
        CombineTerm {
            scale: Scale::One,
            prefix: Some(prefix),
            suffix: one,
        },
        CombineTerm {
            scale: Scale::One,
            prefix: None,
            suffix: rot,
        },
    ]
}

pub fn combine_batch(
    context: &CudaKernelContext,
    terms: &[CombineTerm],
    prefixes: &DeviceFrVec,
    suffixes: &[&DeviceFrVec],
    count: usize,
) -> Result<DeviceFrVec, CudaError> {
    let mut out = context.alloc(count)?;
    if count == 0 || terms.is_empty() {
        return Ok(out);
    }
    for term in terms {
        if term.suffix >= suffixes.len() {
            return Err(CudaError::LengthMismatch {
                expected: suffixes.len(),
                got: term.suffix,
            });
        }
    }
    for column in suffixes {
        if column.len() != count {
            return Err(CudaError::LengthMismatch {
                expected: count,
                got: column.len(),
            });
        }
    }

    let scales: Vec<u32> = terms.iter().map(|term| term.scale as u32).collect();
    let prefix_ids: Vec<u32> = terms
        .iter()
        .map(|term| term.prefix.map_or(NO_PREFIX, |prefix| prefix as u32))
        .collect();
    let slots: Vec<u32> = terms.iter().map(|term| term.suffix as u32).collect();

    let device_scales = context.upload_u32_slice(&scales)?;
    let device_prefix_ids = context.upload_u32_slice(&prefix_ids)?;
    let device_slots = context.upload_u32_slice(&slots)?;
    let pointers = context.device_pointers(suffixes)?;
    let term_count = CudaKernelContext::count_of(terms.len())?;
    let n = CudaKernelContext::count_of(count)?;

    let mut builder = context.stream().launch_builder(context.cmb_combine());
    let _ = builder.arg(&device_scales);
    let _ = builder.arg(&device_prefix_ids);
    let _ = builder.arg(&device_slots);
    let _ = builder.arg(&term_count);
    let _ = builder.arg(prefixes.limbs());
    let _ = builder.arg(&pointers);
    let _ = builder.arg(out.limbs_mut());
    let _ = builder.arg(&n);
    // SAFETY: thread `i < n` reads, for each of `term_count` terms,
    // `suffix_columns[slots[t]][i]` — every slot is `< suffixes.len()` and every
    // column holds exactly `count` elements, both checked above — plus
    // `prefixes[prefix_ids[t]]` when that id is not `NO_PREFIX` (ids come from
    // the `Prefixes` enum, bounded by the 46-element buffer). Writes only
    // `out[i]` of `count`; `out` is a distinct allocation. The three term arrays
    // hold `term_count` u32s each.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(n)) }?;
    context.stream().synchronize()?;
    Ok(out)
}

const NO_PREFIX: u32 = u32::MAX;

pub(super) fn suffix_index(suffix: Suffixes) -> usize {
    suffix as usize
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_lookup_tables::tables::prefixes::PrefixEval;
    use jolt_lookup_tables::tables::LookupTableKind;
    use proptest::prelude::*;

    use super::super::context::shared_context;
    use super::super::testing::fr;
    use super::{combine_batch, combine_terms, CombineTerm, Scale, RISCV_XLEN};

    const NUM_PREFIXES: usize = 46;

    fn present_tables() -> Vec<LookupTableKind<RISCV_XLEN>> {
        <LookupTableKind<RISCV_XLEN> as strum::IntoEnumIterator>::iter().collect()
    }

    fn scale_value(scale: Scale) -> Fr {
        match scale {
            Scale::One => Fr::from_u64(1),
            Scale::NegOne => Fr::from_u64(0) - Fr::from_u64(1),
            Scale::TwoPowXlen => Fr::from_u128(1u128 << RISCV_XLEN),
            Scale::XlenOnes => Fr::from_u64(u64::MAX),
        }
    }

    fn host_from_terms(terms: &[CombineTerm], prefixes: &[Fr], suffixes: &[Fr]) -> Fr {
        terms.iter().fold(Fr::from_u64(0), |acc, term| {
            let prefix = term
                .prefix
                .map_or(Fr::from_u64(1), |prefix| prefixes[prefix as usize]);
            acc + scale_value(term.scale) * prefix * suffixes[term.suffix]
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

    proptest! {
        #[test]
        fn combine_batch_matches_the_term_sum(
            seed in 1u64..100_000,
            table_index in 0usize..40,
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let tables = present_tables();
            let Some(&table) = tables.get(table_index) else { return Ok(()); };
            let terms = combine_terms(table).expect("combine terms");

            let count = 256usize;
            let prefixes: Vec<Fr> = (0..NUM_PREFIXES).map(|i| fr(seed + i as u64)).collect();
            let suffix_columns: Vec<Vec<Fr>> = (0..table.suffixes().len())
                .map(|slot| {
                    (0..count).map(|j| fr(seed + (slot * count + j) as u64 + 7)).collect()
                })
                .collect();

            let device_prefixes = context.upload(&prefixes).expect("upload prefixes");
            let device_suffixes: Vec<_> = suffix_columns
                .iter()
                .map(|column| context.upload(column).expect("upload suffix column"))
                .collect();
            let handles: Vec<_> = device_suffixes.iter().collect();

            let got = combine_batch(context, &terms, &device_prefixes, &handles, count)
                .expect("device combine_batch")
                .to_host()
                .expect("download");

            let expected: Vec<Fr> = (0..count)
                .map(|j| {
                    let row: Vec<Fr> =
                        suffix_columns.iter().map(|column| column[j]).collect();
                    host_from_terms(&terms, &prefixes, &row)
                })
                .collect();
            prop_assert_eq!(got, expected, "combine_batch diverged for {:?}", table);
        }
    }
}
