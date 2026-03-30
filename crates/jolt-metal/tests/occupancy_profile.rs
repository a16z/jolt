#![cfg(target_os = "macos")]
#![allow(clippy::print_stdout)]

use jolt_compiler::{Factor, Formula, ProductTerm};
use jolt_field::Fr;
use jolt_metal::MetalBackend;

fn product_sum_formula(d: usize, p: usize) -> Formula {
    let terms: Vec<_> = (0..p)
        .map(|g| ProductTerm {
            coefficient: 1,
            factors: (0..d).map(|j| Factor::Input((g * d + j) as u32)).collect(),
        })
        .collect();
    Formula::from_terms(terms)
}

fn eq_product_formula() -> Formula {
    Formula::from_terms(vec![ProductTerm {
        coefficient: 1,
        factors: vec![Factor::Input(0), Factor::Input(1)],
    }])
}

fn hamming_booleanity_formula() -> Formula {
    Formula::from_terms(vec![
        ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(1)],
        },
        ProductTerm {
            coefficient: -1,
            factors: vec![Factor::Input(0), Factor::Input(1)],
        },
    ])
}

#[test]
fn print_kernel_occupancy() {
    let backend = MetalBackend::new();

    let formulas: Vec<(&str, Formula)> = vec![
        ("ProductSum D=4 P=1", product_sum_formula(4, 1)),
        ("ProductSum D=8 P=1", product_sum_formula(8, 1)),
        ("EqProduct", eq_product_formula()),
        ("HammingBooleanity", hamming_booleanity_formula()),
    ];

    println!();
    println!("=== Metal Kernel Occupancy Profile (CompileMode::Performance) ===");
    println!();

    for (name, formula) in &formulas {
        let kernel = backend.compile_kernel::<Fr>(formula);
        println!("--- {name} ---");

        for occ in kernel.occupancy() {
            println!(
                "  {:<16}  max_threads_per_tg={:<5}  simd_width={}",
                occ.name, occ.max_threads_per_threadgroup, occ.thread_execution_width,
            );
        }

        println!();
    }
}
