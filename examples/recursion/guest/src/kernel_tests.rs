#[cfg(not(feature = "akita"))]
use jolt::VerifierField as F;
#[cfg(feature = "akita")]
use jolt_akita::AkitaField as F;
use jolt_field::Field;
#[cfg(feature = "akita")]
use jolt_field::{PseudoMersenne, Ring};
use jolt_sdk as jolt;

#[path = "../../src/kernel_cases.rs"]
mod kernel_cases;
use kernel_cases::KernelCase;

fn check_products<const K: usize>(case: &KernelCase<F>) {
    let terms: Vec<[F; K]> = case
        .products
        .iter()
        .map(|term| core::array::from_fn(|i| term[i]))
        .collect();
    assert_eq!(F::sum_of_products(&terms), case.product_sums[K]);
}

provable_with_config! {
fn check_field_kernels(bytes: &[u8]) -> u32 {
    let (cases, consumed): (Vec<KernelCase<F>>, _) =
        bincode::serde::decode_from_slice(bytes, bincode::config::standard()).unwrap();
    assert_eq!(consumed, bytes.len());
    assert!(!cases.is_empty());
    for case in cases {
        for (row, expected) in case.rows.iter().zip(&case.dots) {
            assert_eq!(F::dot_product(row, &case.shared), *expected);
        }
        #[cfg(feature = "akita")]
        {
            let rows: Vec<&[F]> = case.rows.iter().map(Vec::as_slice).collect();
            let mut dots = vec![F::from_u64(0); rows.len()];
            // These must dispatch to the riscv64 kernels, never silently fall back.
            assert!(F::inline_dot_rows(&rows, &case.shared, &mut dots));
            assert_eq!(dots, case.dots);
            assert_eq!(F::inline_weighted_dot(&rows, &case.weights, &case.shared), Some(case.weighted));
        }
        let signed = F::signed_sum(case.shared.iter().enumerate().map(|(i, value)| {
            // While signed_sum lends its accumulator, nested folds must use
            // software without changing the outer register-resident sum.
            assert_eq!(F::dot_product(&case.weights, &case.dots), case.weighted);
            assert_eq!(F::sum_of_products(&case.products), case.product_sums[4]);
            let nested = F::signed_sum(case.shared.iter().enumerate().map(|(j, term)| (term, j % 2 == 0)));
            assert_eq!(nested, case.signed);
            (value, i % 2 == 0)
        }));
        assert_eq!(signed, case.signed);
        check_products::<0>(&case);
        check_products::<1>(&case);
        check_products::<2>(&case);
        check_products::<3>(&case);
        check_products::<4>(&case);
        for ((value, square), inverse) in case.shared.iter().zip(&case.squares).zip(&case.inverses) {
            assert_eq!(*value * *value, *square);
            assert_eq!(value.inverse(), *inverse);
        }
    }
    1
}
}
