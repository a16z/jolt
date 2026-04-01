#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    jolt_eval::invariant::synthesis::fuzz::fuzz_invariant("serialization_roundtrip", data);
});
