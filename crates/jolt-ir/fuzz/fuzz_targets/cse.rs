#![no_main]
use jolt_field::{Field, Fr};
use jolt_ir::ExprBuilder;
use libfuzzer_sys::fuzz_target;

// Builds an expression with shared subexpressions and verifies CSE
// preserves evaluation semantics.
fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let b = ExprBuilder::new();
    let num_openings = 3u32;

    let mut handles = Vec::with_capacity(16);
    for i in 0..num_openings {
        handles.push(b.opening(i));
    }

    // Deliberately reuse handles to create shared subexpressions
    for &byte in data {
        if handles.is_empty() {
            break;
        }
        let op = byte % 4;
        let idx1 = (byte / 4) as usize % handles.len();
        let idx2 = (byte / 16) as usize % handles.len();

        match op {
            0 => handles.push(handles[idx1] + handles[idx2]),
            1 => handles.push(handles[idx1] - handles[idx2]),
            2 => handles.push(handles[idx1] * handles[idx2]),
            _ => handles.push(-handles[idx1]),
        }

        if handles.len() > 128 {
            break;
        }
    }

    if let Some(&root) = handles.last() {
        let expr = b.build(root);
        let openings: Vec<Fr> = (0..num_openings).map(|i| Fr::from_u64(i as u64 + 3)).collect();

        let original: Fr = expr.evaluate(&openings, &[]);
        let optimized = expr.eliminate_common_subexpressions();
        let cse_result: Fr = optimized.evaluate(&openings, &[]);
        assert_eq!(original, cse_result, "CSE changed semantics");
        assert!(optimized.len() <= expr.len(), "CSE increased expression size");
    }
});
