#!/bin/bash

cd src/r1cs/circuits
mkdir -p jolt
circom jolt.circom --r1cs --wasm --sym --c --output jolt
circom jolt_duplicated.circom --r1cs --wasm --sym --c --output jolt
cd -