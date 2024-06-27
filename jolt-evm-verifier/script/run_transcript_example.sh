#!/usr/bin/env bash

cargo build --manifest-path script/Cargo.toml --release --quiet

./script/target/release/transcript_example ${1:-proofs}
