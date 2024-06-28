#!/usr/bin/env bash

cargo build --manifest-path script/Cargo.toml --release --quiet 2> /dev/null 

./script/target/release/script ${1:-proofs}
