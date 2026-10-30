use super::kernel_cases::KernelCase;
use super::{GuestField as F, GuestProgram};
use jolt_field::{CanonicalEncoding, Field, Ring};

fn next_field(seed: &mut u64, index: usize) -> F {
    match index % 8 {
        0 => F::from_u64(0),
        1 => F::from_u64(1),
        2 => -F::from_u64(1),
        _ => {
            let mut bytes = [0; 32];
            for chunk in bytes.chunks_exact_mut(8) {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                chunk.copy_from_slice(&seed.to_le_bytes());
            }
            F::from_bytes_le_reduced(&bytes)
        }
    }
}

fn product_sum<const K: usize>(terms: &[[F; 4]]) -> F {
    let factors: Vec<[F; K]> = terms.iter()
        .map(|term| core::array::from_fn(|i| term[i]))
        .collect();
    F::sum_of_products(&factors)
}

pub(super) fn run() {
    let mut cases = Vec::new();
    for mut seed in [0x1234_5678_9abc_def0, 0xfedc_ba98_7654_3210] {
        // Every one-to-five-row block tail, with and without a preceding block;
        // every zero-to-three-element tail, including empty and multi-block dots.
        for row_count in 1..=10 {
            for width in 0..12 {
                let rows: Vec<Vec<F>> = (0..row_count).map(|row| {
                    (0..width).map(|column| next_field(&mut seed, row + column)).collect()
                }).collect();
                let shared: Vec<F> = (0..width).map(|i| next_field(&mut seed, i)).collect();
                let weights: Vec<F> = (0..row_count).map(|i| next_field(&mut seed, i)).collect();
                let products: Vec<[F; 4]> = (0..width).map(|i| {
                    core::array::from_fn(|j| next_field(&mut seed, i + j))
                }).collect();
                let dots: Vec<F> = rows.iter().map(|row| F::dot_product(row, &shared)).collect();
                let weighted = F::dot_product(&weights, &dots);
                let signed = F::signed_sum(shared.iter().enumerate().map(|(i, value)| (value, i % 2 == 0)));
                let product_sums = [
                    product_sum::<0>(&products), product_sum::<1>(&products),
                    product_sum::<2>(&products), product_sum::<3>(&products),
                    product_sum::<4>(&products),
                ];
                let squares = shared.iter().map(|value| *value * *value).collect();
                let inverses = shared.iter().map(Field::inverse).collect();
                cases.push(KernelCase { rows, shared, weights, products, dots, weighted,
                    signed, product_sums, squares, inverses });
            }
        }
    }
    let encoded = bincode::serde::encode_to_vec(&cases, bincode::config::standard()).unwrap();
    let input = postcard::to_stdvec(&encoded.as_slice()).unwrap();
    let config = GuestProgram::Fibonacci.get_memory_config(true, input.len());
    let output_dir = super::get_guest_src_dir();
    super::generate_provable_macro(
        GuestProgram::Fibonacci, true, &config, &output_dir, super::RecordLimits::default(),
    );
    super::clear_embedded_bytes(&output_dir, &[]);
    let mut program = super::configured_recursion_program(config);
    program.set_func("check_field_kernels");
    program.add_guest_feature("kernel-tests");
    program.build("/tmp/jolt-guest-targets");
    let (rows, device) = program.execute_with_output(&input, &[], &[]);
    assert!(!device.panic, "field kernel differential guest panicked");
    assert_eq!(super::decode_verifier_output(&device.outputs), 1);
    tracing::info!("{} field kernel cases passed in {rows} rows", cases.len());
}
