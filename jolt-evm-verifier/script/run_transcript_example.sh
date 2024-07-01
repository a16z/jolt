#!/usr/bin/env bash

cargo build --manifest-path script/Cargo.toml --release --quiet 2> /dev/null 

./script/target/release/transcript_example ${1:-proofs}
