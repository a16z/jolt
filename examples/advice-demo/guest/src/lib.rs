// This file demonstrates how to use Jolt's advice system in various ways.
// It showcases different types of advice functions, verification of advice,
// and the use of custom structs with both manual and automatic AdviceTapeIO derivations.
// Most of the functions below are silly examples meant purely for demonstration.

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

/// Verifies that a u8 is composite by obtaining its factors (a, b) via advice
/// And checking a * b = n and 1 < a <= b < n
fn verify_composite_u8(n: u8) {
    // Get the factors from the advice tape
    let adv = factor_u8(n);
    // Extract the value from the UntrustedAdvice wrapper using Deref
    let (a, b) = *adv;
    // CRITICAL: Verify that the advice is correct!
    jolt::check_advice!((a as u16) * (b as u16) == (n as u16) && 1 < a && a <= b && b < n);
}

/// Similar functions for u16, just for demonstration
/// This time return an array instead of a tuple, demonstrating both styles
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
    // Get the factors from the advice tape
    let adv = factor_u16(n);
    // Extract the value from the UntrustedAdvice wrapper using Deref
    let [a, b] = *adv;
    // CRITICAL: Verify that the advice is correct!
    jolt::check_advice!((a as u32) * (b as u32) == (n as u32) && 1 < a && a <= b && b < n);
}

/// Similar function for u32, just for demonstration
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
    // Get the factors from the advice tape
    let adv = factor_u32(n);
    // Extract the value from the UntrustedAdvice wrapper using Deref
    let [a, b] = *adv;
    // CRITICAL: Verify that the advice is correct!
    jolt::check_advice!((a as u64) * (b as u64) == (n as u64) && 1 < a && a <= b && b < n);
}

/// Similar function for u64, just for demonstration
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
    // Get the factors from the advice tape
    let adv = factor_u64(n);
    // Extract the value from the UntrustedAdvice wrapper using Deref
    let [a, b] = *adv;
    // CRITICAL: Verify that the advice is correct!
    jolt::check_advice!((a as u128) * (b as u128) == (n as u128) && 1 < a && a <= b && b < n);
}

/// Function to help prove that a is a subset of b using advice
/// Provides a Vec<usize> of indices in b where each element of a can be found
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
            // If any item in a is not found in b, return an empty vector
            indices = Vec::new();
            break;
        }
    }
    indices
}

/// Function to verify that the elements of a are all contained in b
/// using a call to an advice function
fn verify_subset(a: &[usize], b: &[usize]) {
    // Get the indices from the advice tape
    let adv = subset_index(a, b);
    let indices = &*adv;
    // CRITICAL: Verify that indices length matches a length
    jolt::check_advice!(indices.len() == a.len());
    // CRITICAL: Verify that each element in a is found in b at the provided indices
    for (i, &item) in a.iter().enumerate() {
        let index = indices[i];
        jolt::check_advice!(index < b.len() && b[index] == item);
    }
}

/// Custom struct
struct Frobnitz {
    x: u8,
    y: u64,
    z: Vec<u16>,
}

/// Implementation of AdviceTapeIO for Frobnitz
/// This is required for custom structs to be used with advice functions
/// Such an implementation can be very mechanical, as demonstrated here
impl AdviceTapeIO for Frobnitz {
    // Specify serialization to advice tape
    fn write_to_advice_tape(&self) {
        self.x.write_to_advice_tape();
        self.y.write_to_advice_tape();
        self.z.write_to_advice_tape();
    }
    // Specify deserialization from advice tape
    fn new_from_advice_tape() -> Self {
        Frobnitz {
            x: u8::new_from_advice_tape(),
            y: u64::new_from_advice_tape(),
            z: Vec::<u16>::new_from_advice_tape(),
        }
    }
}

/// Advice function to provide a Frobnitz struct
#[jolt::advice]
fn frobnitz_advice() -> jolt::UntrustedAdvice<Frobnitz> {
    Frobnitz {
        x: 42,
        y: 9999,
        z: vec![1, 2, 3, 4, 5],
    }
}

/// Verify that this is a real Frobnitz
fn verify_frobnitz() {
    let adv = frobnitz_advice();
    let frob = &*adv;
    // CRITICAL: dummy checks to make sure the frobnitz is well formed
    jolt::check_advice!(frob.x == 42);
    jolt::check_advice!(frob.y == 9999);
    jolt::check_advice!(frob.z.len() == 5);
    for (i, &val) in frob.z.iter().enumerate() {
        jolt::check_advice!(val == (i as u16) + 1);
    }
}

use bytemuck_derive::{Pod, Zeroable};
/// Custom pod structs for demonstration
/// This struct demonstrates how to automatically derive AdviceTapeIO via bytemuck and JoltPod
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

/// Given a desired area, find a triangle with integer coordinates and that area
/// Lazy approach: find a right triangle with legs (x, y) such that area = (x * y) / 2
/// Returns the triangle as advice
#[jolt::advice]
fn triangle_from_area(area: u32) -> jolt::UntrustedAdvice<Triangle> {
    let target = 2 * area;
    // Find some factorization of target
    let mut x = 1;
    let mut y = target;
    for i in 1..=target {
        if target % i == 0 {
            x = i;
            y = target / i;
            break; // take first factor pair
        }
    }
    Triangle {
        p1: Point { x: 0, y: 0 },
        p2: Point { x, y: 0 },
        p3: Point { x: 0, y },
    }
}

/// Get a triangle with the specified area via advice
/// Verify that the triangle provided by advice has the correct area
fn verify_triangle_from_area(area: u32) {
    let adv = triangle_from_area(area);
    // CRITICAL: Verify that the triangle has the correct area using the shoelace formula
    let double = (adv.p1.x as i64 * (adv.p2.y as i64 - adv.p3.y as i64)
        + adv.p2.x as i64 * (adv.p3.y as i64 - adv.p1.y as i64)
        + adv.p3.x as i64 * (adv.p1.y as i64 - adv.p2.y as i64))
        .abs();
    jolt::check_advice!(double as u32 == 2 * area);
}

/// Entrypoint for the advice demonstration.
#[jolt::provable]
fn advice_demo(n: u8, a: Vec<usize>, b: Vec<usize>) {
    // Exercise all advice functions for pairs and arrays
    // with different sized entries
    // via a simple composite number test
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

    // Demonstrate advice applied to Vec<usize> for a subset check
    start_cycle_tracking("verify subset");
    verify_subset(&a, &b);
    end_cycle_tracking("verify subset");

    // Demonstrate advice applied to custom struct
    start_cycle_tracking("verify frobnitz");
    verify_frobnitz();
    end_cycle_tracking("verify frobnitz");

    // Demonstrate advice applied to custom structs (nested) with auto-derived AdviceTapeIO
    start_cycle_tracking("verify triangle from area");
    verify_triangle_from_area(n as u32); // area = n
    end_cycle_tracking("verify triangle from area");
}
