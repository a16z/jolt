#![cfg_attr(feature = "guest", no_std)]

use jolt::{end_cycle_tracking, start_cycle_tracking};

// Reference implementations
#[allow(unused_imports)]
use blake2::Digest as _;
#[allow(unused_imports)]
use sha2::Digest as _;
#[allow(unused_imports)]
use sha3::Digest as _;

// Inline implementations
use jolt_inlines_blake2 as blake2_inline;
use jolt_inlines_blake3 as blake3_inline;
use jolt_inlines_keccak256 as keccak_inline;
use jolt_inlines_sha2 as sha2_inline;

/// Test sizes for SHA256, Keccak256, Blake2b (up to 2049B)
const SIZES: &[usize] = &[
    32, // baseline
    55, 56, // SHA256: max 1 block (55) / min 2 blocks (56), 9-byte padding
    63, 64, 65, // block boundary (64B)
    127, 128, 129, // Blake2b block boundary (128B)
    135, 136, 137, // Keccak rate boundary (136B)
    255, 256, 257, // 256B boundary
    511, 512, 513, // 512B boundary
    1023, 1024, 1025, 2047, 2048, 2049,
];

/// Blake3 sizes (limited to 64B, single block only)
const BLAKE3_SIZES: &[usize] = &[
    32, // baseline
    55, 56, // near block boundary
    63, 64, // max supported (64B block)
];

#[jolt::provable(
    max_output_size = 4096,
    memory_size = 33554432,
    stack_size = 10485760,
    max_trace_length = 50000000
)]
fn hashbench() -> [u8; 32] {
    bench_sha256();
    bench_keccak();
    bench_blake2b();
    bench_blake3();
    bench_blake3_keyed64();
    [0; 32]
}

fn fill(buf: &mut [u8], seed: u32) {
    let (mut s, a, c) = (seed, 1664525u32, 1013904223u32);
    for b in buf {
        s = s.wrapping_mul(a).wrapping_add(c);
        *b = (s ^ (s >> 16)) as u8;
    }
}

macro_rules! bench {
    ($name:expr, $code:expr) => {{
        start_cycle_tracking($name);
        let r = $code;
        end_cycle_tracking($name);
        r
    }};
}

/// Verify: ref == aligned_digest == stream == unaligned_digest
macro_rules! verify_all {
    ($name:expr, $ref_r:expr, $aligned:expr, $stream:expr, $unaligned:expr) => {
        assert_eq!($ref_r, $aligned, concat!($name, " ref!=aligned"));
        assert_eq!($aligned, $stream, concat!($name, " aligned!=stream"));
        assert_eq!($stream, $unaligned, concat!($name, " stream!=unaligned"));
    };
}

// ============ SHA256 ============
fn bench_sha256() {
    let mut buf = [0u8; 2049];
    let mut ubuf = [0u8; 2050];

    for (i, &sz) in SIZES.iter().enumerate() {
        fill(&mut buf[..sz], i as u32);
        ubuf[1..sz + 1].copy_from_slice(&buf[..sz]);

        let ref_r: [u8; 32] = bench!("sha256_ref", sha2::Sha256::digest(&buf[..sz]).into());
        let aligned = bench!("sha256_a", sha2_inline::Sha256::digest(&buf[..sz]));
        let stream = bench!("sha256_s", {
            let mut h = sha2_inline::Sha256::new();
            h.update(&buf[..sz]);
            h.finalize()
        });
        let unaligned = bench!("sha256_u", sha2_inline::Sha256::digest(&ubuf[1..sz + 1]));

        verify_all!("SHA256", ref_r, aligned, stream, unaligned);
    }
}

// ============ Keccak256 ============
fn bench_keccak() {
    let mut buf = [0u8; 2049];
    let mut ubuf = [0u8; 2050];

    for (i, &sz) in SIZES.iter().enumerate() {
        fill(&mut buf[..sz], 100 + i as u32);
        ubuf[1..sz + 1].copy_from_slice(&buf[..sz]);

        let ref_r: [u8; 32] = bench!("keccak_ref", sha3::Keccak256::digest(&buf[..sz]).into());
        let aligned = bench!("keccak_a", keccak_inline::Keccak256::digest(&buf[..sz]));
        let stream = bench!("keccak_s", {
            let mut h = keccak_inline::Keccak256::new();
            h.update(&buf[..sz]);
            h.finalize()
        });
        let unaligned = bench!(
            "keccak_u",
            keccak_inline::Keccak256::digest(&ubuf[1..sz + 1])
        );

        verify_all!("Keccak", ref_r, aligned, stream, unaligned);
    }
}

// ============ Blake2b ============
fn bench_blake2b() {
    let mut buf = [0u8; 2049];
    let mut ubuf = [0u8; 2050];

    for (i, &sz) in SIZES.iter().enumerate() {
        fill(&mut buf[..sz], 200 + i as u32);
        ubuf[1..sz + 1].copy_from_slice(&buf[..sz]);

        let ref_r: [u8; 64] = bench!("blake2b_ref", blake2::Blake2b512::digest(&buf[..sz]).into());
        let aligned = bench!("blake2b_a", blake2_inline::Blake2b::digest(&buf[..sz]));
        let stream = bench!("blake2b_s", {
            let mut h = blake2_inline::Blake2b::new();
            h.update(&buf[..sz]);
            h.finalize()
        });
        let unaligned = bench!(
            "blake2b_u",
            blake2_inline::Blake2b::digest(&ubuf[1..sz + 1])
        );

        verify_all!("Blake2b", ref_r, aligned, stream, unaligned);
    }
}

// ============ Blake3 ============
fn bench_blake3() {
    let mut buf = [0u8; 64];
    let mut ubuf = [0u8; 65];

    for (i, &sz) in BLAKE3_SIZES.iter().enumerate() {
        fill(&mut buf[..sz], 300 + i as u32);
        ubuf[1..sz + 1].copy_from_slice(&buf[..sz]);

        let ref_r: [u8; 32] = bench!("blake3_ref", *blake3::hash(&buf[..sz]).as_bytes());
        let aligned = bench!("blake3_a", blake3_inline::Blake3::digest(&buf[..sz]));
        let stream = bench!("blake3_s", {
            let mut h = blake3_inline::Blake3::new();
            h.update(&buf[..sz]);
            h.finalize()
        });
        let unaligned = bench!("blake3_u", blake3_inline::Blake3::digest(&ubuf[1..sz + 1]));

        verify_all!("Blake3", ref_r, aligned, stream, unaligned);
    }
}

// ============ Blake3 keyed64 ============
fn bench_blake3_keyed64() {
    use blake3_inline::{blake3_keyed64, AlignedHash32, BLAKE3_IV};

    let mut left = AlignedHash32::new([0u8; 32]);
    let mut right = AlignedHash32::new([0u8; 32]);
    fill(&mut left.0, 401);
    fill(&mut right.0, 402);

    // Reference input: left || right
    let mut input = [0u8; 64];
    input[..32].copy_from_slice(&left.0);
    input[32..].copy_from_slice(&right.0);

    let ref_r: [u8; 32] = bench!("blake3_k64_ref", {
        *blake3::keyed_hash(&BLAKE3_IV.0, &input).as_bytes()
    });

    let mut iv = BLAKE3_IV;
    bench!("blake3_k64", blake3_keyed64(&left, &right, &mut iv));

    assert_eq!(ref_r, iv.0, "Blake3 keyed64 mismatch");
}
