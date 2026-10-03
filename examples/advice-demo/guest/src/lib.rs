use jolt::{end_cycle_tracking, start_cycle_tracking, AdviceTapeIO, JoltPod};

/// Factors u8 n into two u8 factors (a, b)
/// With a <= b such that a * b = n
/// Uses runtime advice to compute the factors outside the proof
/// And then ingest them via the advice tape
/// If the number is prime (or zero), returns (1, n)
/// Unoptimized, but that is fine because this runs outside the proof
#[jolt::advice]
fn factor_u8(n: u8) -> jolt::UntrustedAdvice<(u8, u8)> {
    let mut a = 1u8;
    let mut b = n;
    for i in 2..=n {
        if n % i == 0 {
            a = i;
            b = n / i;
            break;
        }
    }
    (a, b)
}

fn verify_composite_u8(n: u8) {
    let adv = factor_u8(n);
    let (a, b) = *adv;
    // CRITICAL: Verify that the advice is correct!
    // Here we demonstrate both check_advice_eq! and check_advice!
    // With custom error messages (removed when compiled on guest but useful for debugging)
    jolt::check_advice_eq!(
        (a as u16) * (b as u16),
        n as u16,
        "incorrect factors for u8"
    );
    jolt::check_advice!(1 < a && a <= b && b < n, "factors out of range for u8");
}

#[jolt::advice]
fn factor_u16(n: u16) -> jolt::UntrustedAdvice<[u16; 2]> {
    let mut a = 1u16;
    let mut b = n;
    for i in 2..=n {
        if n % i == 0 {
            a = i;
            b = n / i;
            break;
        }
    }
    [a, b]
}

fn verify_composite_u16(n: u16) {
    let adv = factor_u16(n);
    let [a, b] = *adv;
    jolt::check_advice_eq!((a as u32) * (b as u32), n as u32);
    jolt::check_advice!(1 < a && a <= b && b < n);
}

#[jolt::advice]
fn factor_u32(n: u32) -> jolt::UntrustedAdvice<[u32; 2]> {
    let mut a = 1u32;
    let mut b = n;
    for i in 2..=n {
        if n % i == 0 {
            a = i;
            b = n / i;
            break;
        }
    }
    [a, b]
}

fn verify_composite_u32(n: u32) {
    let adv = factor_u32(n);
    let [a, b] = *adv;
    jolt::check_advice_eq!((a as u64) * (b as u64), n as u64);
    jolt::check_advice!(1 < a && a <= b && b < n);
}

#[jolt::advice]
fn factor_u64(n: u64) -> jolt::UntrustedAdvice<[u64; 2]> {
    let mut a = 1u64;
    let mut b = n;
    for i in 2..=n {
        if n % i == 0 {
            a = i;
            b = n / i;
            break;
        }
    }
    [a, b]
}

fn verify_composite_u64(n: u64) {
    let adv = factor_u64(n);
    let [a, b] = *adv;
    // CRITICAL: Verify that the advice is correct!
    // note that jolt::check_advice_eq! doesn't work for u128, so we use jolt::check_advice! here
    jolt::check_advice!((a as u128) * (b as u128) == (n as u128) && 1 < a && a <= b && b < n);
}

#[jolt::advice]
fn subset_index(a: &[usize], b: &[usize]) -> jolt::UntrustedAdvice<Vec<usize>> {
    let mut indices = Vec::new();
    for &item in a.iter() {
        let mut found = false;
        for (i, &b_item) in b.iter().enumerate() {
            if item == b_item {
                indices.push(i);
                found = true;
                break;
            }
        }
        if !found {
            indices = Vec::new();
            break;
        }
    }
    indices
}

fn verify_subset(a: &[usize], b: &[usize]) {
    let adv = subset_index(a, b);
    let indices = &*adv;
    jolt::check_advice!(indices.len() == a.len());
    for (i, &item) in a.iter().enumerate() {
        let index = indices[i];
        jolt::check_advice!(index < b.len() && b[index] == item);
    }
}

struct Frobnitz {
    x: u8,
    y: u64,
    z: Vec<u16>,
}

impl AdviceTapeIO for Frobnitz {
    fn write_to_advice_tape(&self) {
        self.x.write_to_advice_tape();
        self.y.write_to_advice_tape();
        self.z.write_to_advice_tape();
    }
    fn new_from_advice_tape() -> Self {
        Frobnitz {
            x: u8::new_from_advice_tape(),
            y: u64::new_from_advice_tape(),
            z: Vec::<u16>::new_from_advice_tape(),
        }
    }
}

#[jolt::advice]
fn frobnitz_advice() -> jolt::UntrustedAdvice<Frobnitz> {
    Frobnitz {
        x: 42,
        y: 9999,
        z: vec![1, 2, 3, 4, 5],
    }
}

fn verify_frobnitz() {
    let adv = frobnitz_advice();
    let frob = &*adv;
    jolt::check_advice_eq!(frob.x as u64, 42u64);
    jolt::check_advice_eq!(frob.y, 9999u64);
    jolt::check_advice_eq!(frob.z.len() as u64, 5u64);
    for (i, &val) in frob.z.iter().enumerate() {
        jolt::check_advice_eq!(val as u64, (i as u64) + 1u64);
    }
}

use bytemuck_derive::{Pod, Zeroable};
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct Point {
    x: u32,
    y: u32,
}
impl JoltPod for Point {}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct Triangle {
    p1: Point,
    p2: Point,
    p3: Point,
}
impl JoltPod for Triangle {}

#[jolt::advice]
fn triangle_from_area(area: u32) -> jolt::UntrustedAdvice<Triangle> {
    let target = 2 * area;
    let mut x = 1;
    let mut y = target;
    for i in 1..=target {
        if target % i == 0 {
            x = i;
            y = target / i;
            break;
        }
    }
    Triangle {
        p1: Point { x: 0, y: 0 },
        p2: Point { x, y: 0 },
        p3: Point { x: 0, y },
    }
}

fn verify_triangle_from_area(area: u32) {
    let adv = triangle_from_area(area);
    let double = (adv.p1.x as i64 * (adv.p2.y as i64 - adv.p3.y as i64)
        + adv.p2.x as i64 * (adv.p3.y as i64 - adv.p1.y as i64)
        + adv.p3.x as i64 * (adv.p1.y as i64 - adv.p2.y as i64))
        .abs();
    jolt::check_advice_eq!(double as u32, 2 * area);
}

#[jolt::provable]
fn advice_demo(n: u8, a: Vec<usize>, b: Vec<usize>) {
    start_cycle_tracking("verify composite u8");
    verify_composite_u8(n);
    end_cycle_tracking("verify composite u8");

    start_cycle_tracking("verify composite u16");
    verify_composite_u16(n as u16);
    end_cycle_tracking("verify composite u16");

    start_cycle_tracking("verify composite u32");
    verify_composite_u32(n as u32);
    end_cycle_tracking("verify composite u32");

    start_cycle_tracking("verify composite u64");
    verify_composite_u64(n as u64);
    end_cycle_tracking("verify composite u64");

    start_cycle_tracking("verify subset");
    verify_subset(&a, &b);
    end_cycle_tracking("verify subset");

    start_cycle_tracking("verify frobnitz");
    verify_frobnitz();
    end_cycle_tracking("verify frobnitz");

    start_cycle_tracking("verify triangle from area");
    verify_triangle_from_area(n as u32);
    end_cycle_tracking("verify triangle from area");
}
