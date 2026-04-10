//! Direct Blake2b hash computation test to debug op #112 divergence.
#![allow(non_snake_case, clippy::print_stderr)]

use blake2::{Blake2b, Digest};
use blake2::digest::consts::U32;

type Blake2b256 = Blake2b<U32>;

fn hex(b: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        let _ = write!(s, "{byte:02x}");
    }
    s
}

fn from_hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i+2], 16).unwrap())
        .collect()
}

fn try_hash(state: &[u8], n_rounds: u32, data: &[u8]) -> [u8; 32] {
    let mut rb = [0u8; 32];
    rb[28..].copy_from_slice(&n_rounds.to_be_bytes());
    Blake2b256::new()
        .chain_update(state)
        .chain_update(rb)
        .chain_update(data)
        .finalize()
        .into()
}

#[test]
fn hash_op112_direct() {
    let state = from_hex("b14c4cefd0d915ad188a656b0c782a0fada66db863a197777c63e87a0e549c79");
    let core_expected = from_hex("1bcf4b272ac94a8556251ed31264ef5b25965512c54b998262dfb40af02e203e");
    let zkvm_expected = from_hex("f625bb6055493fcbaa0d6ff1d8087f1a949ad9380875353c70b42aa650d4b1a7");

    // What jolt-zkvm appends: LabelWithCount("uniskip_poly", 55)
    let data = from_hex("756e69736b69705f706f6c790000000000000000000000000000000000000037");

    let hash = try_hash(&state, 112, &data);
    eprintln!("Direct hash (n=112, data=LabelWithCount): {}", hex(&hash));
    eprintln!("  matches jolt-zkvm: {}", hash[..] == zkvm_expected[..]);
    eprintln!("  matches jolt-core: {}", hash[..] == core_expected[..]);

    // === Scan for what n_rounds jolt-core used with the same data ===
    eprintln!("\n--- Scan: n_rounds with LabelWithCount data ---");
    let mut found = false;
    for n in 0u32..2000 {
        if try_hash(&state, n, &data)[..] == core_expected[..] {
            eprintln!("FOUND: jolt-core used n_rounds={n} with LabelWithCount data");
            found = true;
            break;
        }
    }
    if !found {
        eprintln!("NOT FOUND: no n_rounds in 0..2000 matches core with LabelWithCount data");
    }

    // === Scan for what coeff count jolt-core used ===
    eprintln!("\n--- Scan: different coeff counts with n_rounds=112 ---");
    found = false;
    for count in 0u64..200 {
        let mut alt = [0u8; 32];
        alt[..12].copy_from_slice(b"uniskip_poly");
        alt[24..32].copy_from_slice(&count.to_be_bytes());
        if try_hash(&state, 112, &alt)[..] == core_expected[..] {
            eprintln!("FOUND: jolt-core used count={count} with n_rounds=112");
            found = true;
            break;
        }
    }
    if !found {
        eprintln!("NOT FOUND: no count in 0..200 matches core with n_rounds=112");
    }

    // === Scan: raw label (no count, 32 bytes zero-padded) ===
    eprintln!("\n--- Scan: raw label (no count) at various n_rounds ---");
    let mut label_only = [0u8; 32];
    label_only[..12].copy_from_slice(b"uniskip_poly");
    found = false;
    for n in 0u32..2000 {
        if try_hash(&state, n, &label_only)[..] == core_expected[..] {
            eprintln!("FOUND: jolt-core used raw label (no count) with n_rounds={n}");
            found = true;
            break;
        }
    }
    if !found {
        eprintln!("NOT FOUND: raw label (no count) doesn't match core for any n_rounds 0..2000");
    }

    // === Scan: maybe it's a challenge (no data), not an append ===
    eprintln!("\n--- Scan: challenge (no data) at various n_rounds ---");
    found = false;
    for n in 0u32..2000 {
        let mut rb = [0u8; 32];
        rb[28..].copy_from_slice(&n.to_be_bytes());
        let h: [u8; 32] = Blake2b256::new()
            .chain_update(&state)
            .chain_update(rb)
            .finalize()
            .into();
        if h[..] == core_expected[..] {
            eprintln!("FOUND: jolt-core did a challenge (no data) at n_rounds={n}");
            found = true;
            break;
        }
    }
    if !found {
        eprintln!("NOT FOUND: no-data challenge doesn't match core for any n_rounds 0..2000");
    }

    // === Scan: different labels at n_rounds=112 ===
    eprintln!("\n--- Scan: common labels at n_rounds=112 ---");
    let labels: &[&[u8]] = &[
        b"sumcheck_poly", b"sumcheck_claim", b"opening_claim",
        b"commitment", b"untrusted_advice", b"Jolt",
    ];
    for label in labels {
        let mut buf = [0u8; 32];
        let len = label.len().min(32);
        buf[..len].copy_from_slice(&label[..len]);
        if try_hash(&state, 112, &buf)[..] == core_expected[..] {
            eprintln!("FOUND: jolt-core used label {:?} (no count) at n_rounds=112",
                std::str::from_utf8(label).unwrap_or("???"));
        }
        // Also try with count=55
        buf[24..32].copy_from_slice(&55u64.to_be_bytes());
        if try_hash(&state, 112, &buf)[..] == core_expected[..] {
            eprintln!("FOUND: jolt-core used label {:?} with count=55 at n_rounds=112",
                std::str::from_utf8(label).unwrap_or("???"));
        }
    }

    // === Nuclear option: try ALL 32-byte scalar data at n_rounds=112 ===
    // This would find if it's a scalar append. Try a few known patterns.
    eprintln!("\n--- Scan: zero scalar at n_rounds=112 ---");
    let zero_scalar = [0u8; 32];
    if try_hash(&state, 112, &zero_scalar)[..] == core_expected[..] {
        eprintln!("FOUND: jolt-core appended a zero scalar at n_rounds=112");
    } else {
        eprintln!("NOT FOUND: zero scalar doesn't match");
    }
}
