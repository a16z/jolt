#!/bin/bash
CIRCUIT_DIR=$1
OUT_DIR=$2

mkdir -p $OUT_DIR
circom ${CIRCUIT_DIR} --r1cs --wasm --sym --c --output ${OUT_DIR}
