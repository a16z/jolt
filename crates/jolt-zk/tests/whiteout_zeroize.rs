//! The Whiteout prover's heap secrets are zeroed before they are freed.
//!
//! A global allocator scans every block it frees, while armed, for a 16-byte
//! needle: a marker the randomness or pads fed in carry verbatim, or a
//! pattern derived from it. A buffer that holds it and is freed unwiped is
//! found. Each test first frees an unwiped buffer of the same data to show
//! the scan finds it.
//! Stack copies and registers are outside what this can see.

#![expect(clippy::unwrap_used, reason = "tests should fail loudly")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::Mutex;

use jolt_algebra::{CanonicalEncoding, CyclotomicRing, Prime128Offset6A23801F};
use jolt_crypto::lattice::{
    derive_public_matrix_prefix, sample_box_digits, PublicMatrixSeed, CHALLENGE_RING_DIM,
};
use jolt_crypto::sample::{GaussianSampler, GaussianWidth};
use jolt_zk::masking::PadLayout;
use jolt_zk::whiteout::{LatticePadCommitment, PadPlacement};
use rand_core::{CryptoRng, Error, RngCore};

type F = Prime128Offset6A23801F;

/// Little-endian, top byte below `0xff` so that it is a canonical element
/// of `F` whose in-memory limbs are these bytes.
const MARKER: [u8; 16] = [
    0x5a, 0x3c, 0x96, 0xe1, 0x0f, 0x7b, 0xd2, 0x4c, 0xa5, 0x17, 0x68, 0xb3, 0xc9, 0x2e, 0x81, 0x7e,
];

struct Scanning;

static ARMED: AtomicBool = AtomicBool::new(false);
static FOUND: AtomicUsize = AtomicUsize::new(0);
static NEEDLE: [AtomicU8; 16] = [const { AtomicU8::new(0) }; 16];
/// Serializes the tests when they share a process.
static SERIAL: Mutex<()> = Mutex::new(());

// SAFETY: forwards every call to `System` unchanged; `dealloc` only reads
// the block it is handed, before freeing it.
unsafe impl GlobalAlloc for Scanning {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract is `System`'s.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if ARMED.load(Ordering::SeqCst) {
            // SAFETY: `ptr` is a live block of `layout.size()` bytes until
            // the `System.dealloc` below.
            let block = unsafe { std::slice::from_raw_parts(ptr, layout.size()) };
            let needle = NEEDLE.each_ref().map(|byte| byte.load(Ordering::SeqCst));
            if block.windows(needle.len()).any(|window| window == needle) {
                let _ = FOUND.fetch_add(1, Ordering::SeqCst);
            }
        }
        // SAFETY: the caller's contract is `System`'s.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Scanning = Scanning;

/// The number of blocks freed with `needle` in them while `f` runs.
fn marked_frees(needle: [u8; 16], f: impl FnOnce()) -> usize {
    for (slot, byte) in NEEDLE.iter().zip(needle) {
        slot.store(byte, Ordering::SeqCst);
    }
    FOUND.store(0, Ordering::SeqCst);
    ARMED.store(true, Ordering::SeqCst);
    f();
    ARMED.store(false, Ordering::SeqCst);
    FOUND.load(Ordering::SeqCst)
}

/// Repeats the marker: every 16-byte-aligned draw is the marker.
struct MarkerRng;

impl RngCore for MarkerRng {
    fn next_u32(&mut self) -> u32 {
        u32::from_le_bytes(MARKER[..4].try_into().unwrap())
    }

    fn next_u64(&mut self) -> u64 {
        u64::from_le_bytes(MARKER[..8].try_into().unwrap())
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for (byte, marker) in dest.iter_mut().zip(MARKER.iter().cycle()) {
            *byte = *marker;
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for MarkerRng {}

/// The chunk seeds of the parallel Gaussian fill.
#[test]
fn gaussian_chunk_seeds_are_zeroed() {
    let _serial = SERIAL.lock().unwrap();
    let sampler = GaussianSampler::new(GaussianWidth::from_multiplier(1 << 20).unwrap());
    let mut out = vec![0i64; 3 << 14];
    assert_eq!(
        marked_frees(MARKER, || {
            let mut seeds = vec![[0u8; 32]; 3];
            for seed in &mut seeds {
                MarkerRng.fill_bytes(seed);
            }
            drop(black_box(seeds));
        }),
        1
    );
    assert_eq!(
        marked_frees(MARKER, || sampler.fill_parallel(&mut MarkerRng, &mut out)),
        0
    );
}

/// The hiding randomness digits. Radix-256 digits are their bytes minus
/// 128, so their memory is the marker with every top bit flipped.
#[test]
fn box_digits_are_zeroed() {
    let _serial = SERIAL.lock().unwrap();
    let digits = MARKER.map(|byte| byte ^ 0x80);
    assert_eq!(
        marked_frees(digits, || {
            let mut bytes = vec![0u8; 1024];
            MarkerRng.fill_bytes(&mut bytes);
            for byte in &mut bytes {
                *byte ^= 0x80;
            }
            drop(black_box(bytes));
        }),
        1
    );
    assert_eq!(
        marked_frees(digits, || drop(
            sample_box_digits(&mut MarkerRng, 1024, 8).unwrap()
        )),
        0
    );
}

/// The copies of the pads the lattice pad commitment lays out before
/// decomposing them.
#[test]
fn pad_commitment_copies_are_zeroed() {
    let _serial = SERIAL.lock().unwrap();
    const D: usize = 64;
    let marker = F::from_u128_checked(u128::from_le_bytes(MARKER)).unwrap();
    let layout = PadLayout::new(5, 1).unwrap();
    let pads = vec![marker; layout.num_pads];
    let stream: Vec<F> =
        derive_public_matrix_prefix(1 << 16, &PublicMatrixSeed::shake256_paged_v1([21; 32]))
            .unwrap();
    let commitment =
        LatticePadCommitment::<F, D>::new(PadPlacement::new(&layout).unwrap(), &stream, 2, 2)
            .unwrap();
    let g = CyclotomicRing::<F, CHALLENGE_RING_DIM>::zero();
    let randomness = vec![0i8; 2 * D];

    assert_eq!(marked_frees(MARKER, || drop(black_box(pads.clone()))), 1);
    assert_eq!(
        marked_frees(MARKER, || drop(
            commitment.commit(&pads, g, &randomness).unwrap()
        )),
        0
    );
}
