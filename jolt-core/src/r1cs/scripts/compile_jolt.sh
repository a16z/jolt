#!/bin/bash
CIRCUIT_PATH=$1
OUT_DIR=$2

mkdir -p $OUT_DIR
circom ${CIRCUIT_PATH} --r1cs --wasm --sym --c --output ${OUT_DIR}
