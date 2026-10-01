#![cfg_attr(feature = "guest", no_std, no_main)]
#[cfg(feature = "guest")]
use ntt_guest as _;

#[cfg(not(feature = "guest"))]
fn main() {}
