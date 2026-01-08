package jolt_verifier

import (
	"encoding/json"
	"fmt"
	"math/big"
	"os"
	"reflect"
)

// WitnessData represents the JSON structure of witness_data.json
type WitnessData struct {
	Preamble struct {
		MaxInputSize  uint64 `json:"max_input_size"`
		MaxOutputSize uint64 `json:"max_output_size"`
		MemorySize    uint64 `json:"memory_size"`
		InputChunk0   string `json:"input_chunk0"`
		OutputChunk0  string `json:"output_chunk0"`
		Panic         uint64 `json:"panic"`
		RamK          uint64 `json:"ram_k"`
		TraceLength   uint64 `json:"trace_length"`
	} `json:"preamble"`
	Commitments    [][]string `json:"commitments"`
	UniSkipCoeffs  []string   `json:"uni_skip_coeffs"`
	SumcheckPolys  [][]string `json:"sumcheck_polys"`
	R1csInputEvals []string   `json:"r1cs_input_evals,omitempty"`
}

// LoadWitnessData loads witness data from JSON file
func LoadWitnessData(filepath string) (*WitnessData, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read witness file: %w", err)
	}

	var witness WitnessData
	if err := json.Unmarshal(data, &witness); err != nil {
		return nil, fmt.Errorf("failed to parse witness JSON: %w", err)
	}

	return &witness, nil
}

// LoadStage1Assignment creates a Stage1Circuit assignment from witness data
func LoadStage1Assignment(witnessPath string) (*Stage1Circuit, error) {
	witness, err := LoadWitnessData(witnessPath)
	if err != nil {
		return nil, err
	}

	assignment := &Stage1Circuit{}
	v := reflect.ValueOf(assignment).Elem()

	// Load commitments (41 commitments × 12 chunks each)
	for i, commitment := range witness.Commitments {
		for j, chunk := range commitment {
			fieldName := fmt.Sprintf("Commitment%dChunk%d", i, j)
			field := v.FieldByName(fieldName)
			if field.IsValid() && field.CanSet() {
				val, ok := new(big.Int).SetString(chunk, 10)
				if !ok {
					return nil, fmt.Errorf("invalid commitment value: %s", chunk)
				}
				field.Set(reflect.ValueOf(val))
			}
		}
	}

	// Load uni-skip coefficients (28 coefficients)
	for i, coeff := range witness.UniSkipCoeffs {
		fieldName := fmt.Sprintf("UniSkipCoeff%d", i)
		field := v.FieldByName(fieldName)
		if field.IsValid() && field.CanSet() {
			val, ok := new(big.Int).SetString(coeff, 10)
			if !ok {
				return nil, fmt.Errorf("invalid uni-skip coeff value: %s", coeff)
			}
			field.Set(reflect.ValueOf(val))
		}
	}

	// Load sumcheck polynomials (11 rounds × 3 coefficients each)
	for round, poly := range witness.SumcheckPolys {
		for coeff, value := range poly {
			fieldName := fmt.Sprintf("SumcheckR%dC%d", round, coeff)
			field := v.FieldByName(fieldName)
			if field.IsValid() && field.CanSet() {
				val, ok := new(big.Int).SetString(value, 10)
				if !ok {
					return nil, fmt.Errorf("invalid sumcheck coeff value: %s", value)
				}
				field.Set(reflect.ValueOf(val))
			}
		}
	}

	// Load R1CS input evals (36 values) if present
	if witness.R1csInputEvals != nil {
		for i, value := range witness.R1csInputEvals {
			fieldName := fmt.Sprintf("R1csInput%d", i)
			field := v.FieldByName(fieldName)
			if field.IsValid() && field.CanSet() {
				val, ok := new(big.Int).SetString(value, 10)
				if !ok {
					return nil, fmt.Errorf("invalid r1cs input value: %s", value)
				}
				field.Set(reflect.ValueOf(val))
			}
		}
	}

	return assignment, nil
}
