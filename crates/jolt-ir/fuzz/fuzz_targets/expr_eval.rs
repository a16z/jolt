#![no_main]
use jolt_field::{Field, Fr};
use jolt_ir::ExprBuilder;
use libfuzzer_sys::fuzz_target;

// Builds an expression from a byte stream and checks that evaluation does not
// panic. Each byte selects an operation; operands are drawn from a small pool
// of live handles.
fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let b = ExprBuilder::new();
    let num_openings = 3u32;
    let num_challenges = 2u32;

    // Seed the handle pool with variables
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
        let op = byte >> 5; // top 3 bits → operation
        let idx = (byte & 0x1F) as usize; // bottom 5 bits → operand index

        match op {
            0 => {
                // Add two handles
                let a = handles[idx % handles.len()];
                let bh = handles[(idx.wrapping_add(1)) % handles.len()];
                handles.push(a + bh);
            }
            1 => {
                // Sub two handles
                let a = handles[idx % handles.len()];
                let bh = handles[(idx.wrapping_add(1)) % handles.len()];
                handles.push(a - bh);
            }
            2 => {
                // Mul two handles
                let a = handles[idx % handles.len()];
                let bh = handles[(idx.wrapping_add(1)) % handles.len()];
                handles.push(a * bh);
            }
            3 => {
                // Neg
                let a = handles[idx % handles.len()];
                handles.push(-a);
            }
            4 => {
                // Constant
                let val = (byte as i128) - 64;
                handles.push(b.constant(val));
            }
            _ => {
                // Integer literal mul
                let a = handles[idx % handles.len()];
                let val = (byte as i128) - 128;
                handles.push(a * val);
            }
        }

        // Cap pool size to avoid OOM
        if handles.len() > 256 {
            break;
        }
    }

    if let Some(&root) = handles.last() {
        let expr = b.build(root);
        let openings: Vec<Fr> = (0..num_openings).map(|i| Fr::from_u64(i as u64 + 1)).collect();
        let challenges: Vec<Fr> =
            (0..num_challenges).map(|i| Fr::from_u64(i as u64 + 100)).collect();
        let _: Fr = expr.evaluate(&openings, &challenges);
    }
});
