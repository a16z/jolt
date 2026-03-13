#![no_main]
use jolt_field::{Field, Fr};
use jolt_ir::ExprBuilder;
use libfuzzer_sys::fuzz_target;

// Builds a random expression and verifies that `to_sum_of_products` produces
// a semantically equivalent result for fixed evaluation points.
fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let b = ExprBuilder::new();
    let num_openings = 3u32;
    let num_challenges = 2u32;

    let mut handles = Vec::with_capacity(16);
    for i in 0..num_openings {
        handles.push(b.opening(i));
    }
    for i in 0..num_challenges {
        handles.push(b.challenge(i));
    }

    for &byte in data {
        if handles.is_empty() {
            break;
        }
        let op = byte % 5;
        let idx = (byte / 5) as usize;

        match op {
            0 => {
                let a = handles[idx % handles.len()];
                let bh = handles[(idx.wrapping_add(1)) % handles.len()];
                handles.push(a + bh);
            }
            1 => {
                let a = handles[idx % handles.len()];
                let bh = handles[(idx.wrapping_add(1)) % handles.len()];
                handles.push(a - bh);
            }
            2 => {
                let a = handles[idx % handles.len()];
                let bh = handles[(idx.wrapping_add(1)) % handles.len()];
                handles.push(a * bh);
            }
            3 => {
                let a = handles[idx % handles.len()];
                handles.push(-a);
            }
            _ => {
                handles.push(b.constant((byte as i128) - 128));
            }
        }

        if handles.len() > 128 {
            break;
        }
    }

    if let Some(&root) = handles.last() {
        let expr = b.build(root);
        let openings: Vec<Fr> = (0..num_openings).map(|i| Fr::from_u64(i as u64 + 7)).collect();
        let challenges: Vec<Fr> =
            (0..num_challenges).map(|i| Fr::from_u64(i as u64 + 42)).collect();

        let direct: Fr = expr.evaluate(&openings, &challenges);
        let sop = expr.to_sum_of_products();
        let via_sop: Fr = sop.evaluate(&openings, &challenges);
        assert_eq!(direct, via_sop, "SoP normalization changed semantics");
    }
});
