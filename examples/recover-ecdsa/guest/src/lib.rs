#![cfg_attr(feature = "guest", no_std)]
extern crate alloc;
use jolt::{end_cycle_tracking, start_cycle_tracking};

use secp256k1::{
    ecdsa::{RecoverableSignature, RecoveryId},
    Message, PublicKey,
};

#[jolt::provable(stack_size = 8388608, heap_size = 16777216, max_trace_length = 1048576)]
fn recover(sig: &[u8], msg: [u8; 32]) -> PublicKey {
    start_cycle_tracking("recover");
    let sig = RecoverableSignature::from_compact(
        &sig[0..64],
        RecoveryId::try_from(sig[64] as i32).unwrap(),
    )
    .unwrap();
    let public = sig.recover_ecdsa(Message::from_digest(msg)).unwrap();
    end_cycle_tracking("recover");

    public
}
