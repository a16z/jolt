#![cfg_attr(feature = "guest", no_std)]
#![no_main]

use cozy_chess::Board;

#[jolt::provable]
fn fib(n: u32) -> u128 {
    let board = Board::default();
    let _enemy_pieces = board.colors(!board.side_to_move());
    let mut a: u128 = 0;
    let mut b: u128 = 1;
    let mut sum: u128;
    for _ in 1..n {
        sum = a + b;
        a = b;
        b = sum;
    }

    b
}

