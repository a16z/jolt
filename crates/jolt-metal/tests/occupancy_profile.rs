#![cfg(target_os = "macos")]
#![allow(clippy::print_stdout)]

use jolt_field::Fr;
use jolt_ir::{KernelDescriptor, KernelShape};
use jolt_metal::MetalBackend;

#[test]
fn print_kernel_occupancy() {
    let backend = MetalBackend::new();

    let descriptors: Vec<(&str, KernelDescriptor)> = vec![
        (
            "ProductSum D=4 P=1",
            KernelDescriptor {
                shape: KernelShape::ProductSum {
                    num_inputs_per_product: 4,
                    num_products: 1,
                },
                degree: 4,
                tensor_split: None,
            },
        ),
        (
            "ProductSum D=8 P=1",
            KernelDescriptor {
                shape: KernelShape::ProductSum {
                    num_inputs_per_product: 8,
                    num_products: 1,
                },
                degree: 8,
                tensor_split: None,
            },
        ),
        (
            "EqProduct",
            KernelDescriptor {
                shape: KernelShape::EqProduct,
                degree: 2,
                tensor_split: None,
            },
        ),
        (
            "HammingBooleanity",
            KernelDescriptor {
                shape: KernelShape::HammingBooleanity,
                degree: 3,
                tensor_split: None,
            },
        ),
    ];

    println!();
    println!("=== Metal Kernel Occupancy Profile (CompileMode::Performance) ===");
    println!();

    for (name, desc) in &descriptors {
        let kernel = backend.compile_kernel::<Fr>(desc);
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
