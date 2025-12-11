#![cfg_attr(feature = "guest", no_std)]
use jolt::{end_cycle_tracking, start_cycle_tracking};

#[jolt::provable(memory_size = 32768, max_trace_length = 65536)]
fn fib(n: u32) -> u128 {
    let mut a: u128 = 0;
    let mut b: u128 = 1;
    let mut sum: u128;

    start_cycle_tracking("fib_loop"); // Use `start_cycle_tracking("{name}")` to start a cycle span

    for _ in 1..n {
        sum = a + b;
        a = b;
        b = sum;
    }
    end_cycle_tracking("fib_loop"); // Use `end_cycle_tracking("{name}")` to end a cycle span
    b
}

#[jolt::provable(memory_size = 32768, max_trace_length = 65536)]
fn dummy(
    n: u128,
    _1: jolt::TrustedAdvice<[u8; 16]>,
    _2: jolt::TrustedAdvice<[u8; 16]>,
    _3: jolt::TrustedAdvice<[u8; 16]>,
    _4: jolt::TrustedAdvice<[u8; 16]>,
    _5: jolt::TrustedAdvice<[u8; 16]>,
    _6: jolt::TrustedAdvice<[u8; 16]>,
    _7: jolt::TrustedAdvice<[u8; 16]>,
    _8: jolt::TrustedAdvice<[u8; 16]>,
    _9: jolt::TrustedAdvice<[u8; 16]>,
    _10: jolt::TrustedAdvice<[u8; 16]>,
    _11: jolt::TrustedAdvice<[u8; 16]>,
    _12: jolt::TrustedAdvice<[u8; 16]>,
    _13: jolt::TrustedAdvice<[u8; 16]>,
    _14: jolt::TrustedAdvice<[u8; 16]>,
    _15: jolt::TrustedAdvice<[u8; 16]>,
    _16: jolt::TrustedAdvice<[u8; 16]>,
    _17: jolt::TrustedAdvice<[u8; 16]>,
    _18: jolt::TrustedAdvice<[u8; 16]>,
    _19: jolt::TrustedAdvice<[u8; 16]>,
    _20: jolt::TrustedAdvice<[u8; 16]>,
    _21: jolt::TrustedAdvice<[u8; 16]>,
    _22: jolt::TrustedAdvice<[u8; 16]>,
    _23: jolt::TrustedAdvice<[u8; 16]>,
    _24: jolt::TrustedAdvice<[u8; 16]>,
    _25: jolt::TrustedAdvice<[u8; 16]>,
    _26: jolt::TrustedAdvice<[u8; 16]>,
    _27: jolt::TrustedAdvice<[u8; 16]>,
    _28: jolt::TrustedAdvice<[u8; 16]>,
    _29: jolt::TrustedAdvice<[u8; 16]>,
    _30: jolt::TrustedAdvice<[u8; 16]>,
    _31: jolt::TrustedAdvice<[u8; 16]>,
    _32: jolt::TrustedAdvice<[u8; 16]>,
    _33: jolt::TrustedAdvice<[u8; 16]>,
    _34: jolt::TrustedAdvice<[u8; 16]>,
    _35: jolt::TrustedAdvice<[u8; 16]>,
    _36: jolt::TrustedAdvice<[u8; 16]>,
    _37: jolt::TrustedAdvice<[u8; 16]>,
    _38: jolt::TrustedAdvice<[u8; 16]>,
    _39: jolt::TrustedAdvice<[u8; 16]>,
    _40: jolt::TrustedAdvice<[u8; 16]>,
    _41: jolt::TrustedAdvice<[u8; 16]>,
    _42: jolt::TrustedAdvice<[u8; 16]>,
    _43: jolt::TrustedAdvice<[u8; 16]>,
    _44: jolt::TrustedAdvice<[u8; 16]>,
    _45: jolt::TrustedAdvice<[u8; 16]>,
    _46: jolt::TrustedAdvice<[u8; 16]>,
    _47: jolt::TrustedAdvice<[u8; 16]>,
    _48: jolt::TrustedAdvice<[u8; 16]>,
    _49: jolt::TrustedAdvice<[u8; 16]>,
    _50: jolt::TrustedAdvice<[u8; 16]>,
    _51: jolt::TrustedAdvice<[u8; 16]>,
    _52: jolt::TrustedAdvice<[u8; 16]>,
    _53: jolt::TrustedAdvice<[u8; 16]>,
    _54: jolt::TrustedAdvice<[u8; 16]>,
    _55: jolt::TrustedAdvice<[u8; 16]>,
    _56: jolt::TrustedAdvice<[u8; 16]>,
    _57: jolt::TrustedAdvice<[u8; 16]>,
    _58: jolt::TrustedAdvice<[u8; 16]>,
    _59: jolt::TrustedAdvice<[u8; 16]>,
    _60: jolt::TrustedAdvice<[u8; 16]>,
    _61: jolt::TrustedAdvice<[u8; 16]>,
    _62: jolt::TrustedAdvice<[u8; 16]>,
    _63: jolt::TrustedAdvice<[u8; 16]>,
    _64: jolt::TrustedAdvice<[u8; 16]>,
    _65: jolt::TrustedAdvice<[u8; 16]>,
    _66: jolt::TrustedAdvice<[u8; 16]>,
    _67: jolt::TrustedAdvice<[u8; 16]>,
    _68: jolt::TrustedAdvice<[u8; 16]>,
    _69: jolt::TrustedAdvice<[u8; 16]>,
    _70: jolt::TrustedAdvice<[u8; 16]>,
    _71: jolt::TrustedAdvice<[u8; 16]>,
    _72: jolt::TrustedAdvice<[u8; 16]>,
    _73: jolt::TrustedAdvice<[u8; 16]>,
    _74: jolt::TrustedAdvice<[u8; 16]>,
    _75: jolt::TrustedAdvice<[u8; 16]>,
    _76: jolt::TrustedAdvice<[u8; 16]>,
    _77: jolt::TrustedAdvice<[u8; 16]>,
    _78: jolt::TrustedAdvice<[u8; 16]>,
    _79: jolt::TrustedAdvice<[u8; 16]>,
    _80: jolt::TrustedAdvice<[u8; 16]>,
    _81: jolt::TrustedAdvice<[u8; 16]>,
    _82: jolt::TrustedAdvice<[u8; 16]>,
    _83: jolt::TrustedAdvice<[u8; 16]>,
    _84: jolt::TrustedAdvice<[u8; 16]>,
    _85: jolt::TrustedAdvice<[u8; 16]>,
    _86: jolt::TrustedAdvice<[u8; 16]>,
    _87: jolt::TrustedAdvice<[u8; 16]>,
    _88: jolt::TrustedAdvice<[u8; 16]>,
    _89: jolt::TrustedAdvice<[u8; 16]>,
    _90: jolt::TrustedAdvice<[u8; 16]>,
    _91: jolt::TrustedAdvice<[u8; 16]>,
    _92: jolt::TrustedAdvice<[u8; 16]>,
    _93: jolt::TrustedAdvice<[u8; 16]>,
    _94: jolt::TrustedAdvice<[u8; 16]>,
    _95: jolt::TrustedAdvice<[u8; 16]>,
    _96: jolt::TrustedAdvice<[u8; 16]>,
    _97: jolt::TrustedAdvice<[u8; 16]>,
    _98: jolt::TrustedAdvice<[u8; 16]>,
    _99: jolt::TrustedAdvice<[u8; 16]>,
    _100: jolt::TrustedAdvice<[u8; 16]>,
    _101: jolt::TrustedAdvice<[u8; 16]>,
    _102: jolt::TrustedAdvice<[u8; 16]>,
    _103: jolt::TrustedAdvice<[u8; 16]>,
    _104: jolt::TrustedAdvice<[u8; 16]>,
    _105: jolt::TrustedAdvice<[u8; 16]>,
    _106: jolt::TrustedAdvice<[u8; 16]>,
    _107: jolt::TrustedAdvice<[u8; 16]>,
    _108: jolt::TrustedAdvice<[u8; 16]>,
    _109: jolt::TrustedAdvice<[u8; 16]>,
    _110: jolt::TrustedAdvice<[u8; 16]>,
    _111: jolt::TrustedAdvice<[u8; 16]>,
    _112: jolt::TrustedAdvice<[u8; 16]>,
    _113: jolt::TrustedAdvice<[u8; 16]>,
    _114: jolt::TrustedAdvice<[u8; 16]>,
    _115: jolt::TrustedAdvice<[u8; 16]>,
    _116: jolt::TrustedAdvice<[u8; 16]>,
    _117: jolt::TrustedAdvice<[u8; 16]>,
    _118: jolt::TrustedAdvice<[u8; 16]>,
    _119: jolt::TrustedAdvice<[u8; 16]>,
    _120: jolt::TrustedAdvice<[u8; 16]>,
    _121: jolt::TrustedAdvice<[u8; 16]>,
    _122: jolt::TrustedAdvice<[u8; 16]>,
    _123: jolt::TrustedAdvice<[u8; 16]>,
    _124: jolt::TrustedAdvice<[u8; 16]>,
    _125: jolt::TrustedAdvice<[u8; 16]>,
    _126: jolt::TrustedAdvice<[u8; 16]>,
    _127: jolt::TrustedAdvice<[u8; 16]>,
    _128: jolt::TrustedAdvice<[u8; 16]>,
) -> u128 {
    // let mut a: u128 = 0;
    // let mut b: u128 = 1;
    // let mut sum: u128;
    n * 2
    // start_cycle_tracking("fib_loop"); // Use `start_cycle_tracking("{name}")` to start a cycle span

    // for _ in 1..n {
    //     sum = a + b;
    //     a = b;
    //     b = sum;
    // }
    // end_cycle_tracking("fib_loop"); // Use `end_cycle_tracking("{name}")` to end a cycle span
    // b
}
