#!/usr/bin/env bash

cargo build --manifest-path script/Cargo.toml --release --quiet

./script/target/release/script ${1:-proofs}
