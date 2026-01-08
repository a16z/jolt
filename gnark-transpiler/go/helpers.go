package jolt_verifier

import "math/big"

// bigInt creates a *big.Int from a string, for constants too large for int64
func bigInt(s string) *big.Int {
	n, _ := new(big.Int).SetString(s, 10)
	return n
}
