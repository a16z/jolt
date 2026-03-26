// Package poseidon implements the Poseidon hash function with circom-compatible parameters
// for BN254. This is adapted from succinctlabs/gnark-plonky2-verifier/poseidon but without
// the Goldilocks dependency.
//
// Parameters: t=4 (width 4), 8 full rounds, 56 partial rounds, S-box x^5
// Constants match light-poseidon circom.
package poseidon

import (
	"math/big"

	"github.com/consensys/gnark/frontend"
)

const BN254_FULL_ROUNDS int = 8
const BN254_PARTIAL_ROUNDS int = 56
const BN254_SPONGE_WIDTH int = 4

type BN254State = [BN254_SPONGE_WIDTH]frontend.Variable

type BN254Chip struct {
	api frontend.API
}

func NewBN254Chip(api frontend.API) *BN254Chip {
	return &BN254Chip{api: api}
}

// Hash computes Poseidon hash of 3 field elements.
// API: poseidon.Hash(api, state, n_rounds, data)
func Hash(api frontend.API, in1, in2, in3 frontend.Variable) frontend.Variable {
	chip := NewBN254Chip(api)

	// Initialize state: [0, in1, in2, in3]
	state := BN254State{
		frontend.Variable(0),
		in1,
		in2,
		in3,
	}

	result := chip.Poseidon(state)
	return result[0]
}

func (c *BN254Chip) Poseidon(state BN254State) BN254State {
	state = c.ark(state, 0)
	state = c.fullRounds(state, true)
	state = c.partialRounds(state)
	state = c.fullRounds(state, false)
	return state
}

func (c *BN254Chip) fullRounds(state BN254State, isFirst bool) BN254State {
	for i := 0; i < BN254_FULL_ROUNDS/2-1; i++ {
		state = c.exp5state(state)
		if isFirst {
			state = c.ark(state, (i+1)*BN254_SPONGE_WIDTH)
		} else {
			state = c.ark(state, (BN254_FULL_ROUNDS/2+1)*BN254_SPONGE_WIDTH+BN254_PARTIAL_ROUNDS+i*BN254_SPONGE_WIDTH)
		}
		state = c.mix(state, mMatrix)
	}

	state = c.exp5state(state)
	if isFirst {
		state = c.ark(state, (BN254_FULL_ROUNDS/2)*BN254_SPONGE_WIDTH)
		state = c.mix(state, pMatrix)
	} else {
		state = c.mix(state, mMatrix)
	}

	return state
}

func (c *BN254Chip) partialRounds(state BN254State) BN254State {
	for i := 0; i < BN254_PARTIAL_ROUNDS; i++ {
		state[0] = c.exp5(state[0])
		state[0] = c.api.Add(state[0], cConstants[(BN254_FULL_ROUNDS/2+1)*BN254_SPONGE_WIDTH+i])

		newState0 := frontend.Variable(0)
		for j := 0; j < BN254_SPONGE_WIDTH; j++ {
			newState0 = c.api.MulAcc(newState0, sConstants[(BN254_SPONGE_WIDTH*2-1)*i+j], state[j])
		}

		for k := 1; k < BN254_SPONGE_WIDTH; k++ {
			state[k] = c.api.MulAcc(state[k], state[0], sConstants[(BN254_SPONGE_WIDTH*2-1)*i+BN254_SPONGE_WIDTH+k-1])
		}
		state[0] = newState0
	}

	return state
}

func (c *BN254Chip) ark(state BN254State, it int) BN254State {
	var result BN254State

	for i := 0; i < len(state); i++ {
		result[i] = c.api.Add(state[i], cConstants[it+i])
	}

	return result
}

func (c *BN254Chip) exp5(x frontend.Variable) frontend.Variable {
	x2 := c.api.Mul(x, x)
	x4 := c.api.Mul(x2, x2)
	return c.api.Mul(x4, x)
}

func (c *BN254Chip) exp5state(state BN254State) BN254State {
	for i := 0; i < BN254_SPONGE_WIDTH; i++ {
		state[i] = c.exp5(state[i])
	}
	return state
}

func (c *BN254Chip) mix(state_ BN254State, constantMatrix [][]*big.Int) BN254State {
	var result BN254State

	for i := 0; i < BN254_SPONGE_WIDTH; i++ {
		result[i] = frontend.Variable(0)
	}

	for i := 0; i < BN254_SPONGE_WIDTH; i++ {
		for j := 0; j < BN254_SPONGE_WIDTH; j++ {
			result[i] = c.api.MulAcc(result[i], constantMatrix[j][i], state_[j])
		}
	}

	return result
}
