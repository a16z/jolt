package jolt_verifier

import (
	"github.com/consensys/gnark/frontend"
	"jolt_verifier/poseidon"
)

type Stage1DebugCircuit struct {
	Commitment0Chunk0 frontend.Variable `gnark:",public"`
	Commitment0Chunk1 frontend.Variable `gnark:",public"`
	Commitment0Chunk2 frontend.Variable `gnark:",public"`
	Commitment0Chunk3 frontend.Variable `gnark:",public"`
	Commitment0Chunk4 frontend.Variable `gnark:",public"`
	Commitment0Chunk5 frontend.Variable `gnark:",public"`
	Commitment0Chunk6 frontend.Variable `gnark:",public"`
	Commitment0Chunk7 frontend.Variable `gnark:",public"`
	Commitment0Chunk8 frontend.Variable `gnark:",public"`
	Commitment0Chunk9 frontend.Variable `gnark:",public"`
	Commitment0Chunk10 frontend.Variable `gnark:",public"`
	Commitment0Chunk11 frontend.Variable `gnark:",public"`
	Commitment1Chunk0 frontend.Variable `gnark:",public"`
	Commitment1Chunk1 frontend.Variable `gnark:",public"`
	Commitment1Chunk2 frontend.Variable `gnark:",public"`
	Commitment1Chunk3 frontend.Variable `gnark:",public"`
	Commitment1Chunk4 frontend.Variable `gnark:",public"`
	Commitment1Chunk5 frontend.Variable `gnark:",public"`
	Commitment1Chunk6 frontend.Variable `gnark:",public"`
	Commitment1Chunk7 frontend.Variable `gnark:",public"`
	Commitment1Chunk8 frontend.Variable `gnark:",public"`
	Commitment1Chunk9 frontend.Variable `gnark:",public"`
	Commitment1Chunk10 frontend.Variable `gnark:",public"`
	Commitment1Chunk11 frontend.Variable `gnark:",public"`
	Commitment2Chunk0 frontend.Variable `gnark:",public"`
	Commitment2Chunk1 frontend.Variable `gnark:",public"`
	Commitment2Chunk2 frontend.Variable `gnark:",public"`
	Commitment2Chunk3 frontend.Variable `gnark:",public"`
	Commitment2Chunk4 frontend.Variable `gnark:",public"`
	Commitment2Chunk5 frontend.Variable `gnark:",public"`
	Commitment2Chunk6 frontend.Variable `gnark:",public"`
	Commitment2Chunk7 frontend.Variable `gnark:",public"`
	Commitment2Chunk8 frontend.Variable `gnark:",public"`
	Commitment2Chunk9 frontend.Variable `gnark:",public"`
	Commitment2Chunk10 frontend.Variable `gnark:",public"`
	Commitment2Chunk11 frontend.Variable `gnark:",public"`
	Commitment3Chunk0 frontend.Variable `gnark:",public"`
	Commitment3Chunk1 frontend.Variable `gnark:",public"`
	Commitment3Chunk2 frontend.Variable `gnark:",public"`
	Commitment3Chunk3 frontend.Variable `gnark:",public"`
	Commitment3Chunk4 frontend.Variable `gnark:",public"`
	Commitment3Chunk5 frontend.Variable `gnark:",public"`
	Commitment3Chunk6 frontend.Variable `gnark:",public"`
	Commitment3Chunk7 frontend.Variable `gnark:",public"`
	Commitment3Chunk8 frontend.Variable `gnark:",public"`
	Commitment3Chunk9 frontend.Variable `gnark:",public"`
	Commitment3Chunk10 frontend.Variable `gnark:",public"`
	Commitment3Chunk11 frontend.Variable `gnark:",public"`
	Commitment4Chunk0 frontend.Variable `gnark:",public"`
	Commitment4Chunk1 frontend.Variable `gnark:",public"`
	Commitment4Chunk2 frontend.Variable `gnark:",public"`
	Commitment4Chunk3 frontend.Variable `gnark:",public"`
	Commitment4Chunk4 frontend.Variable `gnark:",public"`
	Commitment4Chunk5 frontend.Variable `gnark:",public"`
	Commitment4Chunk6 frontend.Variable `gnark:",public"`
	Commitment4Chunk7 frontend.Variable `gnark:",public"`
	Commitment4Chunk8 frontend.Variable `gnark:",public"`
	Commitment4Chunk9 frontend.Variable `gnark:",public"`
	Commitment4Chunk10 frontend.Variable `gnark:",public"`
	Commitment4Chunk11 frontend.Variable `gnark:",public"`
	Commitment5Chunk0 frontend.Variable `gnark:",public"`
	Commitment5Chunk1 frontend.Variable `gnark:",public"`
	Commitment5Chunk2 frontend.Variable `gnark:",public"`
	Commitment5Chunk3 frontend.Variable `gnark:",public"`
	Commitment5Chunk4 frontend.Variable `gnark:",public"`
	Commitment5Chunk5 frontend.Variable `gnark:",public"`
	Commitment5Chunk6 frontend.Variable `gnark:",public"`
	Commitment5Chunk7 frontend.Variable `gnark:",public"`
	Commitment5Chunk8 frontend.Variable `gnark:",public"`
	Commitment5Chunk9 frontend.Variable `gnark:",public"`
	Commitment5Chunk10 frontend.Variable `gnark:",public"`
	Commitment5Chunk11 frontend.Variable `gnark:",public"`
	Commitment6Chunk0 frontend.Variable `gnark:",public"`
	Commitment6Chunk1 frontend.Variable `gnark:",public"`
	Commitment6Chunk2 frontend.Variable `gnark:",public"`
	Commitment6Chunk3 frontend.Variable `gnark:",public"`
	Commitment6Chunk4 frontend.Variable `gnark:",public"`
	Commitment6Chunk5 frontend.Variable `gnark:",public"`
	Commitment6Chunk6 frontend.Variable `gnark:",public"`
	Commitment6Chunk7 frontend.Variable `gnark:",public"`
	Commitment6Chunk8 frontend.Variable `gnark:",public"`
	Commitment6Chunk9 frontend.Variable `gnark:",public"`
	Commitment6Chunk10 frontend.Variable `gnark:",public"`
	Commitment6Chunk11 frontend.Variable `gnark:",public"`
	Commitment7Chunk0 frontend.Variable `gnark:",public"`
	Commitment7Chunk1 frontend.Variable `gnark:",public"`
	Commitment7Chunk2 frontend.Variable `gnark:",public"`
	Commitment7Chunk3 frontend.Variable `gnark:",public"`
	Commitment7Chunk4 frontend.Variable `gnark:",public"`
	Commitment7Chunk5 frontend.Variable `gnark:",public"`
	Commitment7Chunk6 frontend.Variable `gnark:",public"`
	Commitment7Chunk7 frontend.Variable `gnark:",public"`
	Commitment7Chunk8 frontend.Variable `gnark:",public"`
	Commitment7Chunk9 frontend.Variable `gnark:",public"`
	Commitment7Chunk10 frontend.Variable `gnark:",public"`
	Commitment7Chunk11 frontend.Variable `gnark:",public"`
	Commitment8Chunk0 frontend.Variable `gnark:",public"`
	Commitment8Chunk1 frontend.Variable `gnark:",public"`
	Commitment8Chunk2 frontend.Variable `gnark:",public"`
	Commitment8Chunk3 frontend.Variable `gnark:",public"`
	Commitment8Chunk4 frontend.Variable `gnark:",public"`
	Commitment8Chunk5 frontend.Variable `gnark:",public"`
	Commitment8Chunk6 frontend.Variable `gnark:",public"`
	Commitment8Chunk7 frontend.Variable `gnark:",public"`
	Commitment8Chunk8 frontend.Variable `gnark:",public"`
	Commitment8Chunk9 frontend.Variable `gnark:",public"`
	Commitment8Chunk10 frontend.Variable `gnark:",public"`
	Commitment8Chunk11 frontend.Variable `gnark:",public"`
	Commitment9Chunk0 frontend.Variable `gnark:",public"`
	Commitment9Chunk1 frontend.Variable `gnark:",public"`
	Commitment9Chunk2 frontend.Variable `gnark:",public"`
	Commitment9Chunk3 frontend.Variable `gnark:",public"`
	Commitment9Chunk4 frontend.Variable `gnark:",public"`
	Commitment9Chunk5 frontend.Variable `gnark:",public"`
	Commitment9Chunk6 frontend.Variable `gnark:",public"`
	Commitment9Chunk7 frontend.Variable `gnark:",public"`
	Commitment9Chunk8 frontend.Variable `gnark:",public"`
	Commitment9Chunk9 frontend.Variable `gnark:",public"`
	Commitment9Chunk10 frontend.Variable `gnark:",public"`
	Commitment9Chunk11 frontend.Variable `gnark:",public"`
	Commitment10Chunk0 frontend.Variable `gnark:",public"`
	Commitment10Chunk1 frontend.Variable `gnark:",public"`
	Commitment10Chunk2 frontend.Variable `gnark:",public"`
	Commitment10Chunk3 frontend.Variable `gnark:",public"`
	Commitment10Chunk4 frontend.Variable `gnark:",public"`
	Commitment10Chunk5 frontend.Variable `gnark:",public"`
	Commitment10Chunk6 frontend.Variable `gnark:",public"`
	Commitment10Chunk7 frontend.Variable `gnark:",public"`
	Commitment10Chunk8 frontend.Variable `gnark:",public"`
	Commitment10Chunk9 frontend.Variable `gnark:",public"`
	Commitment10Chunk10 frontend.Variable `gnark:",public"`
	Commitment10Chunk11 frontend.Variable `gnark:",public"`
	Commitment11Chunk0 frontend.Variable `gnark:",public"`
	Commitment11Chunk1 frontend.Variable `gnark:",public"`
	Commitment11Chunk2 frontend.Variable `gnark:",public"`
	Commitment11Chunk3 frontend.Variable `gnark:",public"`
	Commitment11Chunk4 frontend.Variable `gnark:",public"`
	Commitment11Chunk5 frontend.Variable `gnark:",public"`
	Commitment11Chunk6 frontend.Variable `gnark:",public"`
	Commitment11Chunk7 frontend.Variable `gnark:",public"`
	Commitment11Chunk8 frontend.Variable `gnark:",public"`
	Commitment11Chunk9 frontend.Variable `gnark:",public"`
	Commitment11Chunk10 frontend.Variable `gnark:",public"`
	Commitment11Chunk11 frontend.Variable `gnark:",public"`
	Commitment12Chunk0 frontend.Variable `gnark:",public"`
	Commitment12Chunk1 frontend.Variable `gnark:",public"`
	Commitment12Chunk2 frontend.Variable `gnark:",public"`
	Commitment12Chunk3 frontend.Variable `gnark:",public"`
	Commitment12Chunk4 frontend.Variable `gnark:",public"`
	Commitment12Chunk5 frontend.Variable `gnark:",public"`
	Commitment12Chunk6 frontend.Variable `gnark:",public"`
	Commitment12Chunk7 frontend.Variable `gnark:",public"`
	Commitment12Chunk8 frontend.Variable `gnark:",public"`
	Commitment12Chunk9 frontend.Variable `gnark:",public"`
	Commitment12Chunk10 frontend.Variable `gnark:",public"`
	Commitment12Chunk11 frontend.Variable `gnark:",public"`
	Commitment13Chunk0 frontend.Variable `gnark:",public"`
	Commitment13Chunk1 frontend.Variable `gnark:",public"`
	Commitment13Chunk2 frontend.Variable `gnark:",public"`
	Commitment13Chunk3 frontend.Variable `gnark:",public"`
	Commitment13Chunk4 frontend.Variable `gnark:",public"`
	Commitment13Chunk5 frontend.Variable `gnark:",public"`
	Commitment13Chunk6 frontend.Variable `gnark:",public"`
	Commitment13Chunk7 frontend.Variable `gnark:",public"`
	Commitment13Chunk8 frontend.Variable `gnark:",public"`
	Commitment13Chunk9 frontend.Variable `gnark:",public"`
	Commitment13Chunk10 frontend.Variable `gnark:",public"`
	Commitment13Chunk11 frontend.Variable `gnark:",public"`
	Commitment14Chunk0 frontend.Variable `gnark:",public"`
	Commitment14Chunk1 frontend.Variable `gnark:",public"`
	Commitment14Chunk2 frontend.Variable `gnark:",public"`
	Commitment14Chunk3 frontend.Variable `gnark:",public"`
	Commitment14Chunk4 frontend.Variable `gnark:",public"`
	Commitment14Chunk5 frontend.Variable `gnark:",public"`
	Commitment14Chunk6 frontend.Variable `gnark:",public"`
	Commitment14Chunk7 frontend.Variable `gnark:",public"`
	Commitment14Chunk8 frontend.Variable `gnark:",public"`
	Commitment14Chunk9 frontend.Variable `gnark:",public"`
	Commitment14Chunk10 frontend.Variable `gnark:",public"`
	Commitment14Chunk11 frontend.Variable `gnark:",public"`
	Commitment15Chunk0 frontend.Variable `gnark:",public"`
	Commitment15Chunk1 frontend.Variable `gnark:",public"`
	Commitment15Chunk2 frontend.Variable `gnark:",public"`
	Commitment15Chunk3 frontend.Variable `gnark:",public"`
	Commitment15Chunk4 frontend.Variable `gnark:",public"`
	Commitment15Chunk5 frontend.Variable `gnark:",public"`
	Commitment15Chunk6 frontend.Variable `gnark:",public"`
	Commitment15Chunk7 frontend.Variable `gnark:",public"`
	Commitment15Chunk8 frontend.Variable `gnark:",public"`
	Commitment15Chunk9 frontend.Variable `gnark:",public"`
	Commitment15Chunk10 frontend.Variable `gnark:",public"`
	Commitment15Chunk11 frontend.Variable `gnark:",public"`
	Commitment16Chunk0 frontend.Variable `gnark:",public"`
	Commitment16Chunk1 frontend.Variable `gnark:",public"`
	Commitment16Chunk2 frontend.Variable `gnark:",public"`
	Commitment16Chunk3 frontend.Variable `gnark:",public"`
	Commitment16Chunk4 frontend.Variable `gnark:",public"`
	Commitment16Chunk5 frontend.Variable `gnark:",public"`
	Commitment16Chunk6 frontend.Variable `gnark:",public"`
	Commitment16Chunk7 frontend.Variable `gnark:",public"`
	Commitment16Chunk8 frontend.Variable `gnark:",public"`
	Commitment16Chunk9 frontend.Variable `gnark:",public"`
	Commitment16Chunk10 frontend.Variable `gnark:",public"`
	Commitment16Chunk11 frontend.Variable `gnark:",public"`
	Commitment17Chunk0 frontend.Variable `gnark:",public"`
	Commitment17Chunk1 frontend.Variable `gnark:",public"`
	Commitment17Chunk2 frontend.Variable `gnark:",public"`
	Commitment17Chunk3 frontend.Variable `gnark:",public"`
	Commitment17Chunk4 frontend.Variable `gnark:",public"`
	Commitment17Chunk5 frontend.Variable `gnark:",public"`
	Commitment17Chunk6 frontend.Variable `gnark:",public"`
	Commitment17Chunk7 frontend.Variable `gnark:",public"`
	Commitment17Chunk8 frontend.Variable `gnark:",public"`
	Commitment17Chunk9 frontend.Variable `gnark:",public"`
	Commitment17Chunk10 frontend.Variable `gnark:",public"`
	Commitment17Chunk11 frontend.Variable `gnark:",public"`
	Commitment18Chunk0 frontend.Variable `gnark:",public"`
	Commitment18Chunk1 frontend.Variable `gnark:",public"`
	Commitment18Chunk2 frontend.Variable `gnark:",public"`
	Commitment18Chunk3 frontend.Variable `gnark:",public"`
	Commitment18Chunk4 frontend.Variable `gnark:",public"`
	Commitment18Chunk5 frontend.Variable `gnark:",public"`
	Commitment18Chunk6 frontend.Variable `gnark:",public"`
	Commitment18Chunk7 frontend.Variable `gnark:",public"`
	Commitment18Chunk8 frontend.Variable `gnark:",public"`
	Commitment18Chunk9 frontend.Variable `gnark:",public"`
	Commitment18Chunk10 frontend.Variable `gnark:",public"`
	Commitment18Chunk11 frontend.Variable `gnark:",public"`
	Commitment19Chunk0 frontend.Variable `gnark:",public"`
	Commitment19Chunk1 frontend.Variable `gnark:",public"`
	Commitment19Chunk2 frontend.Variable `gnark:",public"`
	Commitment19Chunk3 frontend.Variable `gnark:",public"`
	Commitment19Chunk4 frontend.Variable `gnark:",public"`
	Commitment19Chunk5 frontend.Variable `gnark:",public"`
	Commitment19Chunk6 frontend.Variable `gnark:",public"`
	Commitment19Chunk7 frontend.Variable `gnark:",public"`
	Commitment19Chunk8 frontend.Variable `gnark:",public"`
	Commitment19Chunk9 frontend.Variable `gnark:",public"`
	Commitment19Chunk10 frontend.Variable `gnark:",public"`
	Commitment19Chunk11 frontend.Variable `gnark:",public"`
	Commitment20Chunk0 frontend.Variable `gnark:",public"`
	Commitment20Chunk1 frontend.Variable `gnark:",public"`
	Commitment20Chunk2 frontend.Variable `gnark:",public"`
	Commitment20Chunk3 frontend.Variable `gnark:",public"`
	Commitment20Chunk4 frontend.Variable `gnark:",public"`
	Commitment20Chunk5 frontend.Variable `gnark:",public"`
	Commitment20Chunk6 frontend.Variable `gnark:",public"`
	Commitment20Chunk7 frontend.Variable `gnark:",public"`
	Commitment20Chunk8 frontend.Variable `gnark:",public"`
	Commitment20Chunk9 frontend.Variable `gnark:",public"`
	Commitment20Chunk10 frontend.Variable `gnark:",public"`
	Commitment20Chunk11 frontend.Variable `gnark:",public"`
	Commitment21Chunk0 frontend.Variable `gnark:",public"`
	Commitment21Chunk1 frontend.Variable `gnark:",public"`
	Commitment21Chunk2 frontend.Variable `gnark:",public"`
	Commitment21Chunk3 frontend.Variable `gnark:",public"`
	Commitment21Chunk4 frontend.Variable `gnark:",public"`
	Commitment21Chunk5 frontend.Variable `gnark:",public"`
	Commitment21Chunk6 frontend.Variable `gnark:",public"`
	Commitment21Chunk7 frontend.Variable `gnark:",public"`
	Commitment21Chunk8 frontend.Variable `gnark:",public"`
	Commitment21Chunk9 frontend.Variable `gnark:",public"`
	Commitment21Chunk10 frontend.Variable `gnark:",public"`
	Commitment21Chunk11 frontend.Variable `gnark:",public"`
	Commitment22Chunk0 frontend.Variable `gnark:",public"`
	Commitment22Chunk1 frontend.Variable `gnark:",public"`
	Commitment22Chunk2 frontend.Variable `gnark:",public"`
	Commitment22Chunk3 frontend.Variable `gnark:",public"`
	Commitment22Chunk4 frontend.Variable `gnark:",public"`
	Commitment22Chunk5 frontend.Variable `gnark:",public"`
	Commitment22Chunk6 frontend.Variable `gnark:",public"`
	Commitment22Chunk7 frontend.Variable `gnark:",public"`
	Commitment22Chunk8 frontend.Variable `gnark:",public"`
	Commitment22Chunk9 frontend.Variable `gnark:",public"`
	Commitment22Chunk10 frontend.Variable `gnark:",public"`
	Commitment22Chunk11 frontend.Variable `gnark:",public"`
	Commitment23Chunk0 frontend.Variable `gnark:",public"`
	Commitment23Chunk1 frontend.Variable `gnark:",public"`
	Commitment23Chunk2 frontend.Variable `gnark:",public"`
	Commitment23Chunk3 frontend.Variable `gnark:",public"`
	Commitment23Chunk4 frontend.Variable `gnark:",public"`
	Commitment23Chunk5 frontend.Variable `gnark:",public"`
	Commitment23Chunk6 frontend.Variable `gnark:",public"`
	Commitment23Chunk7 frontend.Variable `gnark:",public"`
	Commitment23Chunk8 frontend.Variable `gnark:",public"`
	Commitment23Chunk9 frontend.Variable `gnark:",public"`
	Commitment23Chunk10 frontend.Variable `gnark:",public"`
	Commitment23Chunk11 frontend.Variable `gnark:",public"`
	Commitment24Chunk0 frontend.Variable `gnark:",public"`
	Commitment24Chunk1 frontend.Variable `gnark:",public"`
	Commitment24Chunk2 frontend.Variable `gnark:",public"`
	Commitment24Chunk3 frontend.Variable `gnark:",public"`
	Commitment24Chunk4 frontend.Variable `gnark:",public"`
	Commitment24Chunk5 frontend.Variable `gnark:",public"`
	Commitment24Chunk6 frontend.Variable `gnark:",public"`
	Commitment24Chunk7 frontend.Variable `gnark:",public"`
	Commitment24Chunk8 frontend.Variable `gnark:",public"`
	Commitment24Chunk9 frontend.Variable `gnark:",public"`
	Commitment24Chunk10 frontend.Variable `gnark:",public"`
	Commitment24Chunk11 frontend.Variable `gnark:",public"`
	Commitment25Chunk0 frontend.Variable `gnark:",public"`
	Commitment25Chunk1 frontend.Variable `gnark:",public"`
	Commitment25Chunk2 frontend.Variable `gnark:",public"`
	Commitment25Chunk3 frontend.Variable `gnark:",public"`
	Commitment25Chunk4 frontend.Variable `gnark:",public"`
	Commitment25Chunk5 frontend.Variable `gnark:",public"`
	Commitment25Chunk6 frontend.Variable `gnark:",public"`
	Commitment25Chunk7 frontend.Variable `gnark:",public"`
	Commitment25Chunk8 frontend.Variable `gnark:",public"`
	Commitment25Chunk9 frontend.Variable `gnark:",public"`
	Commitment25Chunk10 frontend.Variable `gnark:",public"`
	Commitment25Chunk11 frontend.Variable `gnark:",public"`
	Commitment26Chunk0 frontend.Variable `gnark:",public"`
	Commitment26Chunk1 frontend.Variable `gnark:",public"`
	Commitment26Chunk2 frontend.Variable `gnark:",public"`
	Commitment26Chunk3 frontend.Variable `gnark:",public"`
	Commitment26Chunk4 frontend.Variable `gnark:",public"`
	Commitment26Chunk5 frontend.Variable `gnark:",public"`
	Commitment26Chunk6 frontend.Variable `gnark:",public"`
	Commitment26Chunk7 frontend.Variable `gnark:",public"`
	Commitment26Chunk8 frontend.Variable `gnark:",public"`
	Commitment26Chunk9 frontend.Variable `gnark:",public"`
	Commitment26Chunk10 frontend.Variable `gnark:",public"`
	Commitment26Chunk11 frontend.Variable `gnark:",public"`
	Commitment27Chunk0 frontend.Variable `gnark:",public"`
	Commitment27Chunk1 frontend.Variable `gnark:",public"`
	Commitment27Chunk2 frontend.Variable `gnark:",public"`
	Commitment27Chunk3 frontend.Variable `gnark:",public"`
	Commitment27Chunk4 frontend.Variable `gnark:",public"`
	Commitment27Chunk5 frontend.Variable `gnark:",public"`
	Commitment27Chunk6 frontend.Variable `gnark:",public"`
	Commitment27Chunk7 frontend.Variable `gnark:",public"`
	Commitment27Chunk8 frontend.Variable `gnark:",public"`
	Commitment27Chunk9 frontend.Variable `gnark:",public"`
	Commitment27Chunk10 frontend.Variable `gnark:",public"`
	Commitment27Chunk11 frontend.Variable `gnark:",public"`
	Commitment28Chunk0 frontend.Variable `gnark:",public"`
	Commitment28Chunk1 frontend.Variable `gnark:",public"`
	Commitment28Chunk2 frontend.Variable `gnark:",public"`
	Commitment28Chunk3 frontend.Variable `gnark:",public"`
	Commitment28Chunk4 frontend.Variable `gnark:",public"`
	Commitment28Chunk5 frontend.Variable `gnark:",public"`
	Commitment28Chunk6 frontend.Variable `gnark:",public"`
	Commitment28Chunk7 frontend.Variable `gnark:",public"`
	Commitment28Chunk8 frontend.Variable `gnark:",public"`
	Commitment28Chunk9 frontend.Variable `gnark:",public"`
	Commitment28Chunk10 frontend.Variable `gnark:",public"`
	Commitment28Chunk11 frontend.Variable `gnark:",public"`
	Commitment29Chunk0 frontend.Variable `gnark:",public"`
	Commitment29Chunk1 frontend.Variable `gnark:",public"`
	Commitment29Chunk2 frontend.Variable `gnark:",public"`
	Commitment29Chunk3 frontend.Variable `gnark:",public"`
	Commitment29Chunk4 frontend.Variable `gnark:",public"`
	Commitment29Chunk5 frontend.Variable `gnark:",public"`
	Commitment29Chunk6 frontend.Variable `gnark:",public"`
	Commitment29Chunk7 frontend.Variable `gnark:",public"`
	Commitment29Chunk8 frontend.Variable `gnark:",public"`
	Commitment29Chunk9 frontend.Variable `gnark:",public"`
	Commitment29Chunk10 frontend.Variable `gnark:",public"`
	Commitment29Chunk11 frontend.Variable `gnark:",public"`
	Commitment30Chunk0 frontend.Variable `gnark:",public"`
	Commitment30Chunk1 frontend.Variable `gnark:",public"`
	Commitment30Chunk2 frontend.Variable `gnark:",public"`
	Commitment30Chunk3 frontend.Variable `gnark:",public"`
	Commitment30Chunk4 frontend.Variable `gnark:",public"`
	Commitment30Chunk5 frontend.Variable `gnark:",public"`
	Commitment30Chunk6 frontend.Variable `gnark:",public"`
	Commitment30Chunk7 frontend.Variable `gnark:",public"`
	Commitment30Chunk8 frontend.Variable `gnark:",public"`
	Commitment30Chunk9 frontend.Variable `gnark:",public"`
	Commitment30Chunk10 frontend.Variable `gnark:",public"`
	Commitment30Chunk11 frontend.Variable `gnark:",public"`
	Commitment31Chunk0 frontend.Variable `gnark:",public"`
	Commitment31Chunk1 frontend.Variable `gnark:",public"`
	Commitment31Chunk2 frontend.Variable `gnark:",public"`
	Commitment31Chunk3 frontend.Variable `gnark:",public"`
	Commitment31Chunk4 frontend.Variable `gnark:",public"`
	Commitment31Chunk5 frontend.Variable `gnark:",public"`
	Commitment31Chunk6 frontend.Variable `gnark:",public"`
	Commitment31Chunk7 frontend.Variable `gnark:",public"`
	Commitment31Chunk8 frontend.Variable `gnark:",public"`
	Commitment31Chunk9 frontend.Variable `gnark:",public"`
	Commitment31Chunk10 frontend.Variable `gnark:",public"`
	Commitment31Chunk11 frontend.Variable `gnark:",public"`
	Commitment32Chunk0 frontend.Variable `gnark:",public"`
	Commitment32Chunk1 frontend.Variable `gnark:",public"`
	Commitment32Chunk2 frontend.Variable `gnark:",public"`
	Commitment32Chunk3 frontend.Variable `gnark:",public"`
	Commitment32Chunk4 frontend.Variable `gnark:",public"`
	Commitment32Chunk5 frontend.Variable `gnark:",public"`
	Commitment32Chunk6 frontend.Variable `gnark:",public"`
	Commitment32Chunk7 frontend.Variable `gnark:",public"`
	Commitment32Chunk8 frontend.Variable `gnark:",public"`
	Commitment32Chunk9 frontend.Variable `gnark:",public"`
	Commitment32Chunk10 frontend.Variable `gnark:",public"`
	Commitment32Chunk11 frontend.Variable `gnark:",public"`
	Commitment33Chunk0 frontend.Variable `gnark:",public"`
	Commitment33Chunk1 frontend.Variable `gnark:",public"`
	Commitment33Chunk2 frontend.Variable `gnark:",public"`
	Commitment33Chunk3 frontend.Variable `gnark:",public"`
	Commitment33Chunk4 frontend.Variable `gnark:",public"`
	Commitment33Chunk5 frontend.Variable `gnark:",public"`
	Commitment33Chunk6 frontend.Variable `gnark:",public"`
	Commitment33Chunk7 frontend.Variable `gnark:",public"`
	Commitment33Chunk8 frontend.Variable `gnark:",public"`
	Commitment33Chunk9 frontend.Variable `gnark:",public"`
	Commitment33Chunk10 frontend.Variable `gnark:",public"`
	Commitment33Chunk11 frontend.Variable `gnark:",public"`
	Commitment34Chunk0 frontend.Variable `gnark:",public"`
	Commitment34Chunk1 frontend.Variable `gnark:",public"`
	Commitment34Chunk2 frontend.Variable `gnark:",public"`
	Commitment34Chunk3 frontend.Variable `gnark:",public"`
	Commitment34Chunk4 frontend.Variable `gnark:",public"`
	Commitment34Chunk5 frontend.Variable `gnark:",public"`
	Commitment34Chunk6 frontend.Variable `gnark:",public"`
	Commitment34Chunk7 frontend.Variable `gnark:",public"`
	Commitment34Chunk8 frontend.Variable `gnark:",public"`
	Commitment34Chunk9 frontend.Variable `gnark:",public"`
	Commitment34Chunk10 frontend.Variable `gnark:",public"`
	Commitment34Chunk11 frontend.Variable `gnark:",public"`
	Commitment35Chunk0 frontend.Variable `gnark:",public"`
	Commitment35Chunk1 frontend.Variable `gnark:",public"`
	Commitment35Chunk2 frontend.Variable `gnark:",public"`
	Commitment35Chunk3 frontend.Variable `gnark:",public"`
	Commitment35Chunk4 frontend.Variable `gnark:",public"`
	Commitment35Chunk5 frontend.Variable `gnark:",public"`
	Commitment35Chunk6 frontend.Variable `gnark:",public"`
	Commitment35Chunk7 frontend.Variable `gnark:",public"`
	Commitment35Chunk8 frontend.Variable `gnark:",public"`
	Commitment35Chunk9 frontend.Variable `gnark:",public"`
	Commitment35Chunk10 frontend.Variable `gnark:",public"`
	Commitment35Chunk11 frontend.Variable `gnark:",public"`
	Commitment36Chunk0 frontend.Variable `gnark:",public"`
	Commitment36Chunk1 frontend.Variable `gnark:",public"`
	Commitment36Chunk2 frontend.Variable `gnark:",public"`
	Commitment36Chunk3 frontend.Variable `gnark:",public"`
	Commitment36Chunk4 frontend.Variable `gnark:",public"`
	Commitment36Chunk5 frontend.Variable `gnark:",public"`
	Commitment36Chunk6 frontend.Variable `gnark:",public"`
	Commitment36Chunk7 frontend.Variable `gnark:",public"`
	Commitment36Chunk8 frontend.Variable `gnark:",public"`
	Commitment36Chunk9 frontend.Variable `gnark:",public"`
	Commitment36Chunk10 frontend.Variable `gnark:",public"`
	Commitment36Chunk11 frontend.Variable `gnark:",public"`
	Commitment37Chunk0 frontend.Variable `gnark:",public"`
	Commitment37Chunk1 frontend.Variable `gnark:",public"`
	Commitment37Chunk2 frontend.Variable `gnark:",public"`
	Commitment37Chunk3 frontend.Variable `gnark:",public"`
	Commitment37Chunk4 frontend.Variable `gnark:",public"`
	Commitment37Chunk5 frontend.Variable `gnark:",public"`
	Commitment37Chunk6 frontend.Variable `gnark:",public"`
	Commitment37Chunk7 frontend.Variable `gnark:",public"`
	Commitment37Chunk8 frontend.Variable `gnark:",public"`
	Commitment37Chunk9 frontend.Variable `gnark:",public"`
	Commitment37Chunk10 frontend.Variable `gnark:",public"`
	Commitment37Chunk11 frontend.Variable `gnark:",public"`
	Commitment38Chunk0 frontend.Variable `gnark:",public"`
	Commitment38Chunk1 frontend.Variable `gnark:",public"`
	Commitment38Chunk2 frontend.Variable `gnark:",public"`
	Commitment38Chunk3 frontend.Variable `gnark:",public"`
	Commitment38Chunk4 frontend.Variable `gnark:",public"`
	Commitment38Chunk5 frontend.Variable `gnark:",public"`
	Commitment38Chunk6 frontend.Variable `gnark:",public"`
	Commitment38Chunk7 frontend.Variable `gnark:",public"`
	Commitment38Chunk8 frontend.Variable `gnark:",public"`
	Commitment38Chunk9 frontend.Variable `gnark:",public"`
	Commitment38Chunk10 frontend.Variable `gnark:",public"`
	Commitment38Chunk11 frontend.Variable `gnark:",public"`
	Commitment39Chunk0 frontend.Variable `gnark:",public"`
	Commitment39Chunk1 frontend.Variable `gnark:",public"`
	Commitment39Chunk2 frontend.Variable `gnark:",public"`
	Commitment39Chunk3 frontend.Variable `gnark:",public"`
	Commitment39Chunk4 frontend.Variable `gnark:",public"`
	Commitment39Chunk5 frontend.Variable `gnark:",public"`
	Commitment39Chunk6 frontend.Variable `gnark:",public"`
	Commitment39Chunk7 frontend.Variable `gnark:",public"`
	Commitment39Chunk8 frontend.Variable `gnark:",public"`
	Commitment39Chunk9 frontend.Variable `gnark:",public"`
	Commitment39Chunk10 frontend.Variable `gnark:",public"`
	Commitment39Chunk11 frontend.Variable `gnark:",public"`
	Commitment40Chunk0 frontend.Variable `gnark:",public"`
	Commitment40Chunk1 frontend.Variable `gnark:",public"`
	Commitment40Chunk2 frontend.Variable `gnark:",public"`
	Commitment40Chunk3 frontend.Variable `gnark:",public"`
	Commitment40Chunk4 frontend.Variable `gnark:",public"`
	Commitment40Chunk5 frontend.Variable `gnark:",public"`
	Commitment40Chunk6 frontend.Variable `gnark:",public"`
	Commitment40Chunk7 frontend.Variable `gnark:",public"`
	Commitment40Chunk8 frontend.Variable `gnark:",public"`
	Commitment40Chunk9 frontend.Variable `gnark:",public"`
	Commitment40Chunk10 frontend.Variable `gnark:",public"`
	Commitment40Chunk11 frontend.Variable `gnark:",public"`
	UniSkipCoeff0 frontend.Variable `gnark:",public"`
	UniSkipCoeff1 frontend.Variable `gnark:",public"`
	UniSkipCoeff2 frontend.Variable `gnark:",public"`
	UniSkipCoeff3 frontend.Variable `gnark:",public"`
	UniSkipCoeff4 frontend.Variable `gnark:",public"`
	UniSkipCoeff5 frontend.Variable `gnark:",public"`
	UniSkipCoeff6 frontend.Variable `gnark:",public"`
	UniSkipCoeff7 frontend.Variable `gnark:",public"`
	UniSkipCoeff8 frontend.Variable `gnark:",public"`
	UniSkipCoeff9 frontend.Variable `gnark:",public"`
	UniSkipCoeff10 frontend.Variable `gnark:",public"`
	UniSkipCoeff11 frontend.Variable `gnark:",public"`
	UniSkipCoeff12 frontend.Variable `gnark:",public"`
	UniSkipCoeff13 frontend.Variable `gnark:",public"`
	UniSkipCoeff14 frontend.Variable `gnark:",public"`
	UniSkipCoeff15 frontend.Variable `gnark:",public"`
	UniSkipCoeff16 frontend.Variable `gnark:",public"`
	UniSkipCoeff17 frontend.Variable `gnark:",public"`
	UniSkipCoeff18 frontend.Variable `gnark:",public"`
	UniSkipCoeff19 frontend.Variable `gnark:",public"`
	UniSkipCoeff20 frontend.Variable `gnark:",public"`
	UniSkipCoeff21 frontend.Variable `gnark:",public"`
	UniSkipCoeff22 frontend.Variable `gnark:",public"`
	UniSkipCoeff23 frontend.Variable `gnark:",public"`
	UniSkipCoeff24 frontend.Variable `gnark:",public"`
	UniSkipCoeff25 frontend.Variable `gnark:",public"`
	UniSkipCoeff26 frontend.Variable `gnark:",public"`
	UniSkipCoeff27 frontend.Variable `gnark:",public"`
	SumcheckR0C0 frontend.Variable `gnark:",public"`
	SumcheckR0C1 frontend.Variable `gnark:",public"`
	SumcheckR0C2 frontend.Variable `gnark:",public"`
	SumcheckR1C0 frontend.Variable `gnark:",public"`
	SumcheckR1C1 frontend.Variable `gnark:",public"`
	SumcheckR1C2 frontend.Variable `gnark:",public"`
	SumcheckR2C0 frontend.Variable `gnark:",public"`
	SumcheckR2C1 frontend.Variable `gnark:",public"`
	SumcheckR2C2 frontend.Variable `gnark:",public"`
	SumcheckR3C0 frontend.Variable `gnark:",public"`
	SumcheckR3C1 frontend.Variable `gnark:",public"`
	SumcheckR3C2 frontend.Variable `gnark:",public"`
	SumcheckR4C0 frontend.Variable `gnark:",public"`
	SumcheckR4C1 frontend.Variable `gnark:",public"`
	SumcheckR4C2 frontend.Variable `gnark:",public"`
	SumcheckR5C0 frontend.Variable `gnark:",public"`
	SumcheckR5C1 frontend.Variable `gnark:",public"`
	SumcheckR5C2 frontend.Variable `gnark:",public"`
	SumcheckR6C0 frontend.Variable `gnark:",public"`
	SumcheckR6C1 frontend.Variable `gnark:",public"`
	SumcheckR6C2 frontend.Variable `gnark:",public"`
	SumcheckR7C0 frontend.Variable `gnark:",public"`
	SumcheckR7C1 frontend.Variable `gnark:",public"`
	SumcheckR7C2 frontend.Variable `gnark:",public"`
	SumcheckR8C0 frontend.Variable `gnark:",public"`
	SumcheckR8C1 frontend.Variable `gnark:",public"`
	SumcheckR8C2 frontend.Variable `gnark:",public"`
	SumcheckR9C0 frontend.Variable `gnark:",public"`
	SumcheckR9C1 frontend.Variable `gnark:",public"`
	SumcheckR9C2 frontend.Variable `gnark:",public"`
	SumcheckR10C0 frontend.Variable `gnark:",public"`
	SumcheckR10C1 frontend.Variable `gnark:",public"`
	SumcheckR10C2 frontend.Variable `gnark:",public"`
	R1csInput0 frontend.Variable `gnark:",public"`
	R1csInput1 frontend.Variable `gnark:",public"`
	R1csInput2 frontend.Variable `gnark:",public"`
	R1csInput3 frontend.Variable `gnark:",public"`
	R1csInput4 frontend.Variable `gnark:",public"`
	R1csInput5 frontend.Variable `gnark:",public"`
	R1csInput6 frontend.Variable `gnark:",public"`
	R1csInput7 frontend.Variable `gnark:",public"`
	R1csInput8 frontend.Variable `gnark:",public"`
	R1csInput9 frontend.Variable `gnark:",public"`
	R1csInput10 frontend.Variable `gnark:",public"`
	R1csInput11 frontend.Variable `gnark:",public"`
	R1csInput12 frontend.Variable `gnark:",public"`
	R1csInput13 frontend.Variable `gnark:",public"`
	R1csInput14 frontend.Variable `gnark:",public"`
	R1csInput15 frontend.Variable `gnark:",public"`
	R1csInput16 frontend.Variable `gnark:",public"`
	R1csInput17 frontend.Variable `gnark:",public"`
	R1csInput18 frontend.Variable `gnark:",public"`
	R1csInput19 frontend.Variable `gnark:",public"`
	R1csInput20 frontend.Variable `gnark:",public"`
	R1csInput21 frontend.Variable `gnark:",public"`
	R1csInput22 frontend.Variable `gnark:",public"`
	R1csInput23 frontend.Variable `gnark:",public"`
	R1csInput24 frontend.Variable `gnark:",public"`
	R1csInput25 frontend.Variable `gnark:",public"`
	R1csInput26 frontend.Variable `gnark:",public"`
	R1csInput27 frontend.Variable `gnark:",public"`
	R1csInput28 frontend.Variable `gnark:",public"`
	R1csInput29 frontend.Variable `gnark:",public"`
	R1csInput30 frontend.Variable `gnark:",public"`
	R1csInput31 frontend.Variable `gnark:",public"`
	R1csInput32 frontend.Variable `gnark:",public"`
	R1csInput33 frontend.Variable `gnark:",public"`
	R1csInput34 frontend.Variable `gnark:",public"`
	R1csInput35 frontend.Variable `gnark:",public"`
}

func (circuit *Stage1DebugCircuit) Define(api frontend.API) error {
	// Memoized subexpressions
	cse_0 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, 1953263434, 0, 0), 0, poseidon.AppendU64Transform(api, 4096)), 1, poseidon.AppendU64Transform(api, 4096)), 2, poseidon.AppendU64Transform(api, 32768)), 3, 50), 4, 201625236193), 5, poseidon.AppendU64Transform(api, 0)), 6, poseidon.AppendU64Transform(api, 8192)), 7, poseidon.AppendU64Transform(api, 1024)), 8, circuit.Commitment0Chunk0), 0, circuit.Commitment0Chunk1), 0, circuit.Commitment0Chunk2), 0, circuit.Commitment0Chunk3), 0, circuit.Commitment0Chunk4), 0, circuit.Commitment0Chunk5), 0, circuit.Commitment0Chunk6), 0, circuit.Commitment0Chunk7), 0, circuit.Commitment0Chunk8), 0, circuit.Commitment0Chunk9), 0, circuit.Commitment0Chunk10), 0, circuit.Commitment0Chunk11), 9, circuit.Commitment1Chunk0), 0, circuit.Commitment1Chunk1), 0, circuit.Commitment1Chunk2), 0, circuit.Commitment1Chunk3), 0, circuit.Commitment1Chunk4), 0, circuit.Commitment1Chunk5), 0, circuit.Commitment1Chunk6), 0, circuit.Commitment1Chunk7), 0, circuit.Commitment1Chunk8), 0, circuit.Commitment1Chunk9), 0, circuit.Commitment1Chunk10), 0, circuit.Commitment1Chunk11), 10, circuit.Commitment2Chunk0), 0, circuit.Commitment2Chunk1), 0, circuit.Commitment2Chunk2), 0, circuit.Commitment2Chunk3), 0, circuit.Commitment2Chunk4), 0, circuit.Commitment2Chunk5), 0, circuit.Commitment2Chunk6), 0, circuit.Commitment2Chunk7), 0, circuit.Commitment2Chunk8), 0, circuit.Commitment2Chunk9), 0, circuit.Commitment2Chunk10), 0, circuit.Commitment2Chunk11), 11, circuit.Commitment3Chunk0), 0, circuit.Commitment3Chunk1), 0, circuit.Commitment3Chunk2), 0, circuit.Commitment3Chunk3), 0, circuit.Commitment3Chunk4), 0, circuit.Commitment3Chunk5), 0, circuit.Commitment3Chunk6), 0, circuit.Commitment3Chunk7), 0, circuit.Commitment3Chunk8), 0, circuit.Commitment3Chunk9), 0, circuit.Commitment3Chunk10), 0, circuit.Commitment3Chunk11), 12, circuit.Commitment4Chunk0), 0, circuit.Commitment4Chunk1), 0, circuit.Commitment4Chunk2), 0, circuit.Commitment4Chunk3), 0, circuit.Commitment4Chunk4), 0, circuit.Commitment4Chunk5), 0, circuit.Commitment4Chunk6), 0, circuit.Commitment4Chunk7), 0, circuit.Commitment4Chunk8), 0, circuit.Commitment4Chunk9), 0, circuit.Commitment4Chunk10), 0, circuit.Commitment4Chunk11), 13, circuit.Commitment5Chunk0), 0, circuit.Commitment5Chunk1), 0, circuit.Commitment5Chunk2), 0, circuit.Commitment5Chunk3), 0, circuit.Commitment5Chunk4), 0, circuit.Commitment5Chunk5), 0, circuit.Commitment5Chunk6), 0, circuit.Commitment5Chunk7), 0, circuit.Commitment5Chunk8), 0, circuit.Commitment5Chunk9), 0, circuit.Commitment5Chunk10), 0, circuit.Commitment5Chunk11), 14, circuit.Commitment6Chunk0), 0, circuit.Commitment6Chunk1), 0, circuit.Commitment6Chunk2), 0, circuit.Commitment6Chunk3), 0, circuit.Commitment6Chunk4), 0, circuit.Commitment6Chunk5), 0, circuit.Commitment6Chunk6), 0, circuit.Commitment6Chunk7), 0, circuit.Commitment6Chunk8), 0, circuit.Commitment6Chunk9), 0, circuit.Commitment6Chunk10), 0, circuit.Commitment6Chunk11), 15, circuit.Commitment7Chunk0), 0, circuit.Commitment7Chunk1), 0, circuit.Commitment7Chunk2), 0, circuit.Commitment7Chunk3), 0, circuit.Commitment7Chunk4), 0, circuit.Commitment7Chunk5), 0, circuit.Commitment7Chunk6), 0, circuit.Commitment7Chunk7), 0, circuit.Commitment7Chunk8), 0, circuit.Commitment7Chunk9), 0, circuit.Commitment7Chunk10), 0, circuit.Commitment7Chunk11), 16, circuit.Commitment8Chunk0), 0, circuit.Commitment8Chunk1), 0, circuit.Commitment8Chunk2), 0, circuit.Commitment8Chunk3), 0, circuit.Commitment8Chunk4), 0, circuit.Commitment8Chunk5), 0, circuit.Commitment8Chunk6), 0, circuit.Commitment8Chunk7), 0, circuit.Commitment8Chunk8), 0, circuit.Commitment8Chunk9), 0, circuit.Commitment8Chunk10), 0, circuit.Commitment8Chunk11), 17, circuit.Commitment9Chunk0), 0, circuit.Commitment9Chunk1), 0, circuit.Commitment9Chunk2), 0, circuit.Commitment9Chunk3), 0, circuit.Commitment9Chunk4), 0, circuit.Commitment9Chunk5), 0, circuit.Commitment9Chunk6), 0, circuit.Commitment9Chunk7), 0, circuit.Commitment9Chunk8), 0, circuit.Commitment9Chunk9), 0, circuit.Commitment9Chunk10), 0, circuit.Commitment9Chunk11), 18, circuit.Commitment10Chunk0), 0, circuit.Commitment10Chunk1), 0, circuit.Commitment10Chunk2), 0, circuit.Commitment10Chunk3), 0, circuit.Commitment10Chunk4), 0, circuit.Commitment10Chunk5), 0, circuit.Commitment10Chunk6), 0, circuit.Commitment10Chunk7), 0, circuit.Commitment10Chunk8), 0, circuit.Commitment10Chunk9), 0, circuit.Commitment10Chunk10), 0, circuit.Commitment10Chunk11), 19, circuit.Commitment11Chunk0), 0, circuit.Commitment11Chunk1), 0, circuit.Commitment11Chunk2), 0, circuit.Commitment11Chunk3), 0, circuit.Commitment11Chunk4), 0, circuit.Commitment11Chunk5), 0, circuit.Commitment11Chunk6), 0, circuit.Commitment11Chunk7), 0, circuit.Commitment11Chunk8), 0, circuit.Commitment11Chunk9), 0, circuit.Commitment11Chunk10), 0, circuit.Commitment11Chunk11), 20, circuit.Commitment12Chunk0), 0, circuit.Commitment12Chunk1), 0, circuit.Commitment12Chunk2), 0, circuit.Commitment12Chunk3), 0, circuit.Commitment12Chunk4), 0, circuit.Commitment12Chunk5), 0, circuit.Commitment12Chunk6), 0, circuit.Commitment12Chunk7), 0, circuit.Commitment12Chunk8), 0, circuit.Commitment12Chunk9), 0, circuit.Commitment12Chunk10), 0, circuit.Commitment12Chunk11), 21, circuit.Commitment13Chunk0), 0, circuit.Commitment13Chunk1), 0, circuit.Commitment13Chunk2), 0, circuit.Commitment13Chunk3), 0, circuit.Commitment13Chunk4), 0, circuit.Commitment13Chunk5), 0, circuit.Commitment13Chunk6), 0, circuit.Commitment13Chunk7), 0, circuit.Commitment13Chunk8), 0, circuit.Commitment13Chunk9), 0, circuit.Commitment13Chunk10), 0, circuit.Commitment13Chunk11), 22, circuit.Commitment14Chunk0), 0, circuit.Commitment14Chunk1), 0, circuit.Commitment14Chunk2), 0, circuit.Commitment14Chunk3), 0, circuit.Commitment14Chunk4), 0, circuit.Commitment14Chunk5), 0, circuit.Commitment14Chunk6), 0, circuit.Commitment14Chunk7), 0, circuit.Commitment14Chunk8), 0, circuit.Commitment14Chunk9), 0, circuit.Commitment14Chunk10), 0, circuit.Commitment14Chunk11), 23, circuit.Commitment15Chunk0), 0, circuit.Commitment15Chunk1), 0, circuit.Commitment15Chunk2), 0, circuit.Commitment15Chunk3), 0, circuit.Commitment15Chunk4), 0, circuit.Commitment15Chunk5), 0, circuit.Commitment15Chunk6), 0, circuit.Commitment15Chunk7), 0, circuit.Commitment15Chunk8), 0, circuit.Commitment15Chunk9), 0, circuit.Commitment15Chunk10), 0, circuit.Commitment15Chunk11), 24, circuit.Commitment16Chunk0), 0, circuit.Commitment16Chunk1), 0, circuit.Commitment16Chunk2), 0, circuit.Commitment16Chunk3), 0, circuit.Commitment16Chunk4), 0, circuit.Commitment16Chunk5), 0, circuit.Commitment16Chunk6), 0, circuit.Commitment16Chunk7), 0, circuit.Commitment16Chunk8), 0, circuit.Commitment16Chunk9), 0, circuit.Commitment16Chunk10), 0, circuit.Commitment16Chunk11), 25, circuit.Commitment17Chunk0), 0, circuit.Commitment17Chunk1), 0, circuit.Commitment17Chunk2), 0, circuit.Commitment17Chunk3), 0, circuit.Commitment17Chunk4), 0, circuit.Commitment17Chunk5), 0, circuit.Commitment17Chunk6), 0, circuit.Commitment17Chunk7), 0, circuit.Commitment17Chunk8), 0, circuit.Commitment17Chunk9), 0, circuit.Commitment17Chunk10), 0, circuit.Commitment17Chunk11), 26, circuit.Commitment18Chunk0), 0, circuit.Commitment18Chunk1), 0, circuit.Commitment18Chunk2), 0, circuit.Commitment18Chunk3), 0, circuit.Commitment18Chunk4), 0, circuit.Commitment18Chunk5), 0, circuit.Commitment18Chunk6), 0, circuit.Commitment18Chunk7), 0, circuit.Commitment18Chunk8), 0, circuit.Commitment18Chunk9), 0, circuit.Commitment18Chunk10), 0, circuit.Commitment18Chunk11), 27, circuit.Commitment19Chunk0), 0, circuit.Commitment19Chunk1), 0, circuit.Commitment19Chunk2), 0, circuit.Commitment19Chunk3), 0, circuit.Commitment19Chunk4), 0, circuit.Commitment19Chunk5), 0, circuit.Commitment19Chunk6), 0, circuit.Commitment19Chunk7), 0, circuit.Commitment19Chunk8), 0, circuit.Commitment19Chunk9), 0, circuit.Commitment19Chunk10), 0, circuit.Commitment19Chunk11), 28, circuit.Commitment20Chunk0), 0, circuit.Commitment20Chunk1), 0, circuit.Commitment20Chunk2), 0, circuit.Commitment20Chunk3), 0, circuit.Commitment20Chunk4), 0, circuit.Commitment20Chunk5), 0, circuit.Commitment20Chunk6), 0, circuit.Commitment20Chunk7), 0, circuit.Commitment20Chunk8), 0, circuit.Commitment20Chunk9), 0, circuit.Commitment20Chunk10), 0, circuit.Commitment20Chunk11), 29, circuit.Commitment21Chunk0), 0, circuit.Commitment21Chunk1), 0, circuit.Commitment21Chunk2), 0, circuit.Commitment21Chunk3), 0, circuit.Commitment21Chunk4), 0, circuit.Commitment21Chunk5), 0, circuit.Commitment21Chunk6), 0, circuit.Commitment21Chunk7), 0, circuit.Commitment21Chunk8), 0, circuit.Commitment21Chunk9), 0, circuit.Commitment21Chunk10), 0, circuit.Commitment21Chunk11), 30, circuit.Commitment22Chunk0), 0, circuit.Commitment22Chunk1), 0, circuit.Commitment22Chunk2), 0, circuit.Commitment22Chunk3), 0, circuit.Commitment22Chunk4), 0, circuit.Commitment22Chunk5), 0, circuit.Commitment22Chunk6), 0, circuit.Commitment22Chunk7), 0, circuit.Commitment22Chunk8), 0, circuit.Commitment22Chunk9), 0, circuit.Commitment22Chunk10), 0, circuit.Commitment22Chunk11), 31, circuit.Commitment23Chunk0), 0, circuit.Commitment23Chunk1), 0, circuit.Commitment23Chunk2), 0, circuit.Commitment23Chunk3), 0, circuit.Commitment23Chunk4), 0, circuit.Commitment23Chunk5), 0, circuit.Commitment23Chunk6), 0, circuit.Commitment23Chunk7), 0, circuit.Commitment23Chunk8), 0, circuit.Commitment23Chunk9), 0, circuit.Commitment23Chunk10), 0, circuit.Commitment23Chunk11), 32, circuit.Commitment24Chunk0), 0, circuit.Commitment24Chunk1), 0, circuit.Commitment24Chunk2), 0, circuit.Commitment24Chunk3), 0, circuit.Commitment24Chunk4), 0, circuit.Commitment24Chunk5), 0, circuit.Commitment24Chunk6), 0, circuit.Commitment24Chunk7), 0, circuit.Commitment24Chunk8), 0, circuit.Commitment24Chunk9), 0, circuit.Commitment24Chunk10), 0, circuit.Commitment24Chunk11), 33, circuit.Commitment25Chunk0), 0, circuit.Commitment25Chunk1), 0, circuit.Commitment25Chunk2), 0, circuit.Commitment25Chunk3), 0, circuit.Commitment25Chunk4), 0, circuit.Commitment25Chunk5), 0, circuit.Commitment25Chunk6), 0, circuit.Commitment25Chunk7), 0, circuit.Commitment25Chunk8), 0, circuit.Commitment25Chunk9), 0, circuit.Commitment25Chunk10), 0, circuit.Commitment25Chunk11), 34, circuit.Commitment26Chunk0), 0, circuit.Commitment26Chunk1), 0, circuit.Commitment26Chunk2), 0, circuit.Commitment26Chunk3), 0, circuit.Commitment26Chunk4), 0, circuit.Commitment26Chunk5), 0, circuit.Commitment26Chunk6), 0, circuit.Commitment26Chunk7), 0, circuit.Commitment26Chunk8), 0, circuit.Commitment26Chunk9), 0, circuit.Commitment26Chunk10), 0, circuit.Commitment26Chunk11), 35, circuit.Commitment27Chunk0), 0, circuit.Commitment27Chunk1), 0, circuit.Commitment27Chunk2), 0, circuit.Commitment27Chunk3), 0, circuit.Commitment27Chunk4), 0, circuit.Commitment27Chunk5), 0, circuit.Commitment27Chunk6), 0, circuit.Commitment27Chunk7), 0, circuit.Commitment27Chunk8), 0, circuit.Commitment27Chunk9), 0, circuit.Commitment27Chunk10), 0, circuit.Commitment27Chunk11), 36, circuit.Commitment28Chunk0), 0, circuit.Commitment28Chunk1), 0, circuit.Commitment28Chunk2), 0, circuit.Commitment28Chunk3), 0, circuit.Commitment28Chunk4), 0, circuit.Commitment28Chunk5), 0, circuit.Commitment28Chunk6), 0, circuit.Commitment28Chunk7), 0, circuit.Commitment28Chunk8), 0, circuit.Commitment28Chunk9), 0, circuit.Commitment28Chunk10), 0, circuit.Commitment28Chunk11), 37, circuit.Commitment29Chunk0), 0, circuit.Commitment29Chunk1), 0, circuit.Commitment29Chunk2), 0, circuit.Commitment29Chunk3), 0, circuit.Commitment29Chunk4), 0, circuit.Commitment29Chunk5), 0, circuit.Commitment29Chunk6), 0, circuit.Commitment29Chunk7), 0, circuit.Commitment29Chunk8), 0, circuit.Commitment29Chunk9), 0, circuit.Commitment29Chunk10), 0, circuit.Commitment29Chunk11), 38, circuit.Commitment30Chunk0), 0, circuit.Commitment30Chunk1), 0, circuit.Commitment30Chunk2), 0, circuit.Commitment30Chunk3), 0, circuit.Commitment30Chunk4), 0, circuit.Commitment30Chunk5), 0, circuit.Commitment30Chunk6), 0, circuit.Commitment30Chunk7), 0, circuit.Commitment30Chunk8), 0, circuit.Commitment30Chunk9), 0, circuit.Commitment30Chunk10), 0, circuit.Commitment30Chunk11), 39, circuit.Commitment31Chunk0), 0, circuit.Commitment31Chunk1), 0, circuit.Commitment31Chunk2), 0, circuit.Commitment31Chunk3), 0, circuit.Commitment31Chunk4), 0, circuit.Commitment31Chunk5), 0, circuit.Commitment31Chunk6), 0, circuit.Commitment31Chunk7), 0, circuit.Commitment31Chunk8), 0, circuit.Commitment31Chunk9), 0, circuit.Commitment31Chunk10), 0, circuit.Commitment31Chunk11), 40, circuit.Commitment32Chunk0), 0, circuit.Commitment32Chunk1), 0, circuit.Commitment32Chunk2), 0, circuit.Commitment32Chunk3), 0, circuit.Commitment32Chunk4), 0, circuit.Commitment32Chunk5), 0, circuit.Commitment32Chunk6), 0, circuit.Commitment32Chunk7), 0, circuit.Commitment32Chunk8), 0, circuit.Commitment32Chunk9), 0, circuit.Commitment32Chunk10), 0, circuit.Commitment32Chunk11), 41, circuit.Commitment33Chunk0), 0, circuit.Commitment33Chunk1), 0, circuit.Commitment33Chunk2), 0, circuit.Commitment33Chunk3), 0, circuit.Commitment33Chunk4), 0, circuit.Commitment33Chunk5), 0, circuit.Commitment33Chunk6), 0, circuit.Commitment33Chunk7), 0, circuit.Commitment33Chunk8), 0, circuit.Commitment33Chunk9), 0, circuit.Commitment33Chunk10), 0, circuit.Commitment33Chunk11), 42, circuit.Commitment34Chunk0), 0, circuit.Commitment34Chunk1), 0, circuit.Commitment34Chunk2), 0, circuit.Commitment34Chunk3), 0, circuit.Commitment34Chunk4), 0, circuit.Commitment34Chunk5), 0, circuit.Commitment34Chunk6), 0, circuit.Commitment34Chunk7), 0, circuit.Commitment34Chunk8), 0, circuit.Commitment34Chunk9), 0, circuit.Commitment34Chunk10), 0, circuit.Commitment34Chunk11), 43, circuit.Commitment35Chunk0), 0, circuit.Commitment35Chunk1), 0, circuit.Commitment35Chunk2), 0, circuit.Commitment35Chunk3), 0, circuit.Commitment35Chunk4), 0, circuit.Commitment35Chunk5), 0, circuit.Commitment35Chunk6), 0, circuit.Commitment35Chunk7), 0, circuit.Commitment35Chunk8), 0, circuit.Commitment35Chunk9), 0, circuit.Commitment35Chunk10), 0, circuit.Commitment35Chunk11), 44, circuit.Commitment36Chunk0), 0, circuit.Commitment36Chunk1), 0, circuit.Commitment36Chunk2), 0, circuit.Commitment36Chunk3), 0, circuit.Commitment36Chunk4), 0, circuit.Commitment36Chunk5), 0, circuit.Commitment36Chunk6), 0, circuit.Commitment36Chunk7), 0, circuit.Commitment36Chunk8), 0, circuit.Commitment36Chunk9), 0, circuit.Commitment36Chunk10), 0, circuit.Commitment36Chunk11), 45, circuit.Commitment37Chunk0), 0, circuit.Commitment37Chunk1), 0, circuit.Commitment37Chunk2), 0, circuit.Commitment37Chunk3), 0, circuit.Commitment37Chunk4), 0, circuit.Commitment37Chunk5), 0, circuit.Commitment37Chunk6), 0, circuit.Commitment37Chunk7), 0, circuit.Commitment37Chunk8), 0, circuit.Commitment37Chunk9), 0, circuit.Commitment37Chunk10), 0, circuit.Commitment37Chunk11), 46, circuit.Commitment38Chunk0), 0, circuit.Commitment38Chunk1), 0, circuit.Commitment38Chunk2), 0, circuit.Commitment38Chunk3), 0, circuit.Commitment38Chunk4), 0, circuit.Commitment38Chunk5), 0, circuit.Commitment38Chunk6), 0, circuit.Commitment38Chunk7), 0, circuit.Commitment38Chunk8), 0, circuit.Commitment38Chunk9), 0, circuit.Commitment38Chunk10), 0, circuit.Commitment38Chunk11), 47, circuit.Commitment39Chunk0), 0, circuit.Commitment39Chunk1), 0, circuit.Commitment39Chunk2), 0, circuit.Commitment39Chunk3), 0, circuit.Commitment39Chunk4), 0, circuit.Commitment39Chunk5), 0, circuit.Commitment39Chunk6), 0, circuit.Commitment39Chunk7), 0, circuit.Commitment39Chunk8), 0, circuit.Commitment39Chunk9), 0, circuit.Commitment39Chunk10), 0, circuit.Commitment39Chunk11), 48, circuit.Commitment40Chunk0), 0, circuit.Commitment40Chunk1), 0, circuit.Commitment40Chunk2), 0, circuit.Commitment40Chunk3), 0, circuit.Commitment40Chunk4), 0, circuit.Commitment40Chunk5), 0, circuit.Commitment40Chunk6), 0, circuit.Commitment40Chunk7), 0, circuit.Commitment40Chunk8), 0, circuit.Commitment40Chunk9), 0, circuit.Commitment40Chunk10), 0, circuit.Commitment40Chunk11), 49, 0)
	cse_1 := poseidon.Hash(api, cse_0, 50, 0)
	cse_2 := poseidon.Hash(api, cse_1, 51, 0)
	cse_3 := poseidon.Hash(api, cse_2, 52, 0)
	cse_4 := poseidon.Hash(api, cse_3, 53, 0)
	cse_5 := poseidon.Hash(api, cse_4, 54, 0)
	cse_6 := poseidon.Hash(api, cse_5, 55, 0)
	cse_7 := poseidon.Hash(api, cse_6, 56, 0)
	cse_8 := poseidon.Hash(api, cse_7, 57, 0)
	cse_9 := poseidon.Hash(api, cse_8, 58, 0)
	cse_10 := poseidon.Hash(api, cse_9, 59, 0)
	cse_11 := poseidon.Hash(api, cse_10, 60, 0)
	cse_12 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_11, 61, bigInt("693065686773592458709161276463075796193455407009757267193429")), 62, poseidon.ByteReverse(api, circuit.UniSkipCoeff0)), 63, poseidon.ByteReverse(api, circuit.UniSkipCoeff1)), 64, poseidon.ByteReverse(api, circuit.UniSkipCoeff2)), 65, poseidon.ByteReverse(api, circuit.UniSkipCoeff3)), 66, poseidon.ByteReverse(api, circuit.UniSkipCoeff4)), 67, poseidon.ByteReverse(api, circuit.UniSkipCoeff5)), 68, poseidon.ByteReverse(api, circuit.UniSkipCoeff6)), 69, poseidon.ByteReverse(api, circuit.UniSkipCoeff7)), 70, poseidon.ByteReverse(api, circuit.UniSkipCoeff8)), 71, poseidon.ByteReverse(api, circuit.UniSkipCoeff9)), 72, poseidon.ByteReverse(api, circuit.UniSkipCoeff10)), 73, poseidon.ByteReverse(api, circuit.UniSkipCoeff11)), 74, poseidon.ByteReverse(api, circuit.UniSkipCoeff12)), 75, poseidon.ByteReverse(api, circuit.UniSkipCoeff13)), 76, poseidon.ByteReverse(api, circuit.UniSkipCoeff14)), 77, poseidon.ByteReverse(api, circuit.UniSkipCoeff15)), 78, poseidon.ByteReverse(api, circuit.UniSkipCoeff16)), 79, poseidon.ByteReverse(api, circuit.UniSkipCoeff17)), 80, poseidon.ByteReverse(api, circuit.UniSkipCoeff18)), 81, poseidon.ByteReverse(api, circuit.UniSkipCoeff19)), 82, poseidon.ByteReverse(api, circuit.UniSkipCoeff20)), 83, poseidon.ByteReverse(api, circuit.UniSkipCoeff21)), 84, poseidon.ByteReverse(api, circuit.UniSkipCoeff22)), 85, poseidon.ByteReverse(api, circuit.UniSkipCoeff23)), 86, poseidon.ByteReverse(api, circuit.UniSkipCoeff24)), 87, poseidon.ByteReverse(api, circuit.UniSkipCoeff25)), 88, poseidon.ByteReverse(api, circuit.UniSkipCoeff26)), 89, poseidon.ByteReverse(api, circuit.UniSkipCoeff27)), 90, bigInt("9619401173246373414507010453289387209824226095986339413")), 91, 0)
	cse_13 := poseidon.Truncate128Reverse(api, cse_12)
	cse_14 := api.Mul(cse_13, cse_13)
	cse_15 := api.Mul(cse_14, cse_13)
	cse_16 := api.Mul(cse_15, cse_13)
	cse_17 := api.Mul(cse_16, cse_13)
	cse_18 := api.Mul(cse_17, cse_13)
	cse_19 := api.Mul(cse_18, cse_13)
	cse_20 := api.Mul(cse_19, cse_13)
	cse_21 := api.Mul(cse_20, cse_13)
	cse_22 := api.Mul(cse_21, cse_13)
	cse_23 := api.Mul(cse_22, cse_13)
	cse_24 := api.Mul(cse_23, cse_13)
	cse_25 := api.Mul(cse_24, cse_13)
	cse_26 := api.Mul(cse_25, cse_13)
	cse_27 := api.Mul(cse_26, cse_13)
	cse_28 := api.Mul(cse_27, cse_13)
	cse_29 := api.Mul(cse_28, cse_13)
	cse_30 := api.Mul(cse_29, cse_13)
	cse_31 := api.Mul(cse_30, cse_13)
	cse_32 := api.Mul(cse_31, cse_13)
	cse_33 := api.Mul(cse_32, cse_13)
	cse_34 := api.Mul(cse_33, cse_13)
	cse_35 := api.Mul(cse_34, cse_13)
	cse_36 := api.Mul(cse_35, cse_13)
	cse_37 := api.Mul(cse_36, cse_13)
	cse_38 := api.Mul(cse_37, cse_13)
	cse_39 := api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(circuit.UniSkipCoeff0, api.Mul(circuit.UniSkipCoeff1, cse_13)), api.Mul(circuit.UniSkipCoeff2, cse_14)), api.Mul(circuit.UniSkipCoeff3, cse_15)), api.Mul(circuit.UniSkipCoeff4, cse_16)), api.Mul(circuit.UniSkipCoeff5, cse_17)), api.Mul(circuit.UniSkipCoeff6, cse_18)), api.Mul(circuit.UniSkipCoeff7, cse_19)), api.Mul(circuit.UniSkipCoeff8, cse_20)), api.Mul(circuit.UniSkipCoeff9, cse_21)), api.Mul(circuit.UniSkipCoeff10, cse_22)), api.Mul(circuit.UniSkipCoeff11, cse_23)), api.Mul(circuit.UniSkipCoeff12, cse_24)), api.Mul(circuit.UniSkipCoeff13, cse_25)), api.Mul(circuit.UniSkipCoeff14, cse_26)), api.Mul(circuit.UniSkipCoeff15, cse_27)), api.Mul(circuit.UniSkipCoeff16, cse_28)), api.Mul(circuit.UniSkipCoeff17, cse_29)), api.Mul(circuit.UniSkipCoeff18, cse_30)), api.Mul(circuit.UniSkipCoeff19, cse_31)), api.Mul(circuit.UniSkipCoeff20, cse_32)), api.Mul(circuit.UniSkipCoeff21, cse_33)), api.Mul(circuit.UniSkipCoeff22, cse_34)), api.Mul(circuit.UniSkipCoeff23, cse_35)), api.Mul(circuit.UniSkipCoeff24, cse_36)), api.Mul(circuit.UniSkipCoeff25, cse_37)), api.Mul(circuit.UniSkipCoeff26, cse_38)), api.Mul(circuit.UniSkipCoeff27, api.Mul(cse_38, cse_13)))
	cse_40 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_12, 92, poseidon.ByteReverse(api, cse_39)), 93, poseidon.ByteReverse(api, cse_39)), 94, 0)
	cse_41 := poseidon.Truncate128(api, cse_40)
	cse_42 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_40, 95, bigInt("8747718800733414012499765325397")), 96, poseidon.ByteReverse(api, circuit.SumcheckR0C0)), 97, poseidon.ByteReverse(api, circuit.SumcheckR0C1)), 98, poseidon.ByteReverse(api, circuit.SumcheckR0C2)), 99, bigInt("121413912275379154240237141")), 100, 0)
	cse_43 := poseidon.Truncate128Reverse(api, cse_42)
	cse_44 := api.Mul(cse_43, cse_43)
	cse_45 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_42, 101, bigInt("8747718800733414012499765325397")), 102, poseidon.ByteReverse(api, circuit.SumcheckR1C0)), 103, poseidon.ByteReverse(api, circuit.SumcheckR1C1)), 104, poseidon.ByteReverse(api, circuit.SumcheckR1C2)), 105, bigInt("121413912275379154240237141")), 106, 0)
	cse_46 := poseidon.Truncate128Reverse(api, cse_45)
	cse_47 := api.Mul(cse_46, cse_46)
	cse_48 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_45, 107, bigInt("8747718800733414012499765325397")), 108, poseidon.ByteReverse(api, circuit.SumcheckR2C0)), 109, poseidon.ByteReverse(api, circuit.SumcheckR2C1)), 110, poseidon.ByteReverse(api, circuit.SumcheckR2C2)), 111, bigInt("121413912275379154240237141")), 112, 0)
	cse_49 := poseidon.Truncate128Reverse(api, cse_48)
	cse_50 := api.Mul(cse_49, cse_49)
	cse_51 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_48, 113, bigInt("8747718800733414012499765325397")), 114, poseidon.ByteReverse(api, circuit.SumcheckR3C0)), 115, poseidon.ByteReverse(api, circuit.SumcheckR3C1)), 116, poseidon.ByteReverse(api, circuit.SumcheckR3C2)), 117, bigInt("121413912275379154240237141")), 118, 0)
	cse_52 := poseidon.Truncate128Reverse(api, cse_51)
	cse_53 := api.Mul(cse_52, cse_52)
	cse_54 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_51, 119, bigInt("8747718800733414012499765325397")), 120, poseidon.ByteReverse(api, circuit.SumcheckR4C0)), 121, poseidon.ByteReverse(api, circuit.SumcheckR4C1)), 122, poseidon.ByteReverse(api, circuit.SumcheckR4C2)), 123, bigInt("121413912275379154240237141")), 124, 0)
	cse_55 := poseidon.Truncate128Reverse(api, cse_54)
	cse_56 := api.Mul(cse_55, cse_55)
	cse_57 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_54, 125, bigInt("8747718800733414012499765325397")), 126, poseidon.ByteReverse(api, circuit.SumcheckR5C0)), 127, poseidon.ByteReverse(api, circuit.SumcheckR5C1)), 128, poseidon.ByteReverse(api, circuit.SumcheckR5C2)), 129, bigInt("121413912275379154240237141")), 130, 0)
	cse_58 := poseidon.Truncate128Reverse(api, cse_57)
	cse_59 := api.Mul(cse_58, cse_58)
	cse_60 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_57, 131, bigInt("8747718800733414012499765325397")), 132, poseidon.ByteReverse(api, circuit.SumcheckR6C0)), 133, poseidon.ByteReverse(api, circuit.SumcheckR6C1)), 134, poseidon.ByteReverse(api, circuit.SumcheckR6C2)), 135, bigInt("121413912275379154240237141")), 136, 0)
	cse_61 := poseidon.Truncate128Reverse(api, cse_60)
	cse_62 := api.Mul(cse_61, cse_61)
	cse_63 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_60, 137, bigInt("8747718800733414012499765325397")), 138, poseidon.ByteReverse(api, circuit.SumcheckR7C0)), 139, poseidon.ByteReverse(api, circuit.SumcheckR7C1)), 140, poseidon.ByteReverse(api, circuit.SumcheckR7C2)), 141, bigInt("121413912275379154240237141")), 142, 0)
	cse_64 := poseidon.Truncate128Reverse(api, cse_63)
	cse_65 := api.Mul(cse_64, cse_64)
	cse_66 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_63, 143, bigInt("8747718800733414012499765325397")), 144, poseidon.ByteReverse(api, circuit.SumcheckR8C0)), 145, poseidon.ByteReverse(api, circuit.SumcheckR8C1)), 146, poseidon.ByteReverse(api, circuit.SumcheckR8C2)), 147, bigInt("121413912275379154240237141")), 148, 0)
	cse_67 := poseidon.Truncate128Reverse(api, cse_66)
	cse_68 := api.Mul(cse_67, cse_67)
	cse_69 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_66, 149, bigInt("8747718800733414012499765325397")), 150, poseidon.ByteReverse(api, circuit.SumcheckR9C0)), 151, poseidon.ByteReverse(api, circuit.SumcheckR9C1)), 152, poseidon.ByteReverse(api, circuit.SumcheckR9C2)), 153, bigInt("121413912275379154240237141")), 154, 0)
	cse_70 := poseidon.Truncate128Reverse(api, cse_69)
	cse_71 := api.Mul(cse_70, cse_70)
	cse_72 := poseidon.Truncate128Reverse(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_69, 155, bigInt("8747718800733414012499765325397")), 156, poseidon.ByteReverse(api, circuit.SumcheckR10C0)), 157, poseidon.ByteReverse(api, circuit.SumcheckR10C1)), 158, poseidon.ByteReverse(api, circuit.SumcheckR10C2)), 159, bigInt("121413912275379154240237141")), 160, 0))
	cse_73 := api.Mul(cse_72, cse_72)
	cse_74 := api.Mul(1, 362880)
	cse_75 := api.Mul(cse_74, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808455297"))
	cse_76 := api.Mul(cse_75, 10080)
	cse_77 := api.Mul(cse_76, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808491297"))
	cse_78 := api.Mul(cse_77, 2880)
	cse_79 := api.Mul(cse_78, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808492737"))
	cse_80 := api.Mul(cse_79, 4320)
	cse_81 := api.Mul(cse_80, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808485537"))
	cse_82 := api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808132737"))
	cse_83 := api.Mul(cse_82, 40320)
	cse_84 := api.Mul(cse_83, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808485537"))
	cse_85 := api.Mul(cse_84, 4320)
	cse_86 := api.Mul(cse_85, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808492737"))
	cse_87 := api.Mul(cse_86, 2880)
	cse_88 := api.Mul(cse_87, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808491297"))
	cse_89 := api.Mul(cse_88, 10080)
	cse_90 := api.Mul(cse_89, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808455297"))
	cse_91 := api.Inverse(api.Mul(cse_90, 362880))
	cse_92 := api.Mul(api.Mul(1, api.Mul(cse_81, 40320)), cse_91)
	cse_93 := api.Sub(poseidon.Truncate128Reverse(api, cse_11), bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495613"))
	cse_94 := api.Sub(cse_93, 1)
	cse_95 := api.Sub(cse_94, 1)
	cse_96 := api.Sub(cse_95, 1)
	cse_97 := api.Sub(cse_96, 1)
	cse_98 := api.Sub(cse_97, 1)
	cse_99 := api.Sub(cse_98, 1)
	cse_100 := api.Sub(cse_99, 1)
	cse_101 := api.Sub(cse_100, 1)
	cse_102 := api.Sub(cse_101, 1)
	cse_103 := api.Mul(1, cse_102)
	cse_104 := api.Mul(cse_103, cse_101)
	cse_105 := api.Mul(cse_104, cse_100)
	cse_106 := api.Mul(cse_105, cse_99)
	cse_107 := api.Mul(cse_106, cse_98)
	cse_108 := api.Mul(cse_107, cse_97)
	cse_109 := api.Mul(cse_108, cse_96)
	cse_110 := api.Mul(cse_109, cse_95)
	cse_111 := api.Mul(1, cse_93)
	cse_112 := api.Mul(cse_111, cse_94)
	cse_113 := api.Mul(cse_112, cse_95)
	cse_114 := api.Mul(cse_113, cse_96)
	cse_115 := api.Mul(cse_114, cse_97)
	cse_116 := api.Mul(cse_115, cse_98)
	cse_117 := api.Mul(cse_116, cse_99)
	cse_118 := api.Mul(cse_117, cse_100)
	cse_119 := api.Mul(cse_118, cse_101)
	cse_120 := api.Inverse(api.Mul(cse_119, cse_102))
	cse_121 := api.Mul(cse_92, api.Mul(api.Mul(1, api.Mul(cse_110, cse_94)), cse_120))
	cse_122 := api.Sub(cse_13, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495613"))
	cse_123 := api.Sub(cse_122, 1)
	cse_124 := api.Sub(cse_123, 1)
	cse_125 := api.Sub(cse_124, 1)
	cse_126 := api.Sub(cse_125, 1)
	cse_127 := api.Sub(cse_126, 1)
	cse_128 := api.Sub(cse_127, 1)
	cse_129 := api.Sub(cse_128, 1)
	cse_130 := api.Sub(cse_129, 1)
	cse_131 := api.Sub(cse_130, 1)
	cse_132 := api.Mul(1, cse_131)
	cse_133 := api.Mul(cse_132, cse_130)
	cse_134 := api.Mul(cse_133, cse_129)
	cse_135 := api.Mul(cse_134, cse_128)
	cse_136 := api.Mul(cse_135, cse_127)
	cse_137 := api.Mul(cse_136, cse_126)
	cse_138 := api.Mul(cse_137, cse_125)
	cse_139 := api.Mul(cse_138, cse_124)
	cse_140 := api.Mul(1, cse_122)
	cse_141 := api.Mul(cse_140, cse_123)
	cse_142 := api.Mul(cse_141, cse_124)
	cse_143 := api.Mul(cse_142, cse_125)
	cse_144 := api.Mul(cse_143, cse_126)
	cse_145 := api.Mul(cse_144, cse_127)
	cse_146 := api.Mul(cse_145, cse_128)
	cse_147 := api.Mul(cse_146, cse_129)
	cse_148 := api.Mul(cse_147, cse_130)
	cse_149 := api.Inverse(api.Mul(cse_148, cse_131))
	cse_150 := api.Mul(cse_92, api.Mul(api.Mul(1, api.Mul(cse_139, cse_123)), cse_149))
	cse_151 := api.Mul(api.Mul(cse_82, cse_81), cse_91)
	cse_152 := api.Mul(cse_151, api.Mul(api.Mul(cse_111, cse_110), cse_120))
	cse_153 := api.Mul(cse_151, api.Mul(api.Mul(cse_140, cse_139), cse_149))
	cse_154 := api.Mul(api.Mul(cse_83, cse_80), cse_91)
	cse_155 := api.Mul(cse_154, api.Mul(api.Mul(cse_112, cse_109), cse_120))
	cse_156 := api.Mul(cse_154, api.Mul(api.Mul(cse_141, cse_138), cse_149))
	cse_157 := api.Mul(api.Mul(cse_84, cse_79), cse_91)
	cse_158 := api.Mul(cse_157, api.Mul(api.Mul(cse_113, cse_108), cse_120))
	cse_159 := api.Mul(cse_157, api.Mul(api.Mul(cse_142, cse_137), cse_149))
	cse_160 := api.Mul(api.Mul(cse_85, cse_78), cse_91)
	cse_161 := api.Mul(cse_160, api.Mul(api.Mul(cse_114, cse_107), cse_120))
	cse_162 := api.Mul(cse_160, api.Mul(api.Mul(cse_143, cse_136), cse_149))
	cse_163 := api.Mul(api.Mul(cse_86, cse_77), cse_91)
	cse_164 := api.Mul(cse_163, api.Mul(api.Mul(cse_115, cse_106), cse_120))
	cse_165 := api.Mul(cse_163, api.Mul(api.Mul(cse_144, cse_135), cse_149))
	cse_166 := api.Mul(api.Mul(cse_87, cse_76), cse_91)
	cse_167 := api.Mul(cse_166, api.Mul(api.Mul(cse_116, cse_105), cse_120))
	cse_168 := api.Mul(cse_166, api.Mul(api.Mul(cse_145, cse_134), cse_149))
	cse_169 := api.Mul(api.Mul(cse_88, cse_75), cse_91)
	cse_170 := api.Mul(cse_169, api.Mul(api.Mul(cse_117, cse_104), cse_120))
	cse_171 := api.Mul(cse_169, api.Mul(api.Mul(cse_146, cse_133), cse_149))
	cse_172 := api.Mul(api.Mul(cse_89, cse_74), cse_91)
	cse_173 := api.Mul(cse_172, api.Mul(api.Mul(cse_118, cse_103), cse_120))
	cse_174 := api.Mul(cse_172, api.Mul(api.Mul(cse_147, cse_132), cse_149))
	cse_175 := api.Mul(api.Mul(cse_90, 1), cse_91)
	cse_176 := api.Mul(cse_175, api.Mul(api.Mul(cse_119, 1), cse_120))
	cse_177 := api.Mul(cse_175, api.Mul(api.Mul(cse_148, 1), cse_149))
	cse_178 := poseidon.Truncate128Reverse(api, cse_0)
	cse_179 := poseidon.Truncate128Reverse(api, cse_1)
	cse_180 := poseidon.Truncate128Reverse(api, cse_2)
	cse_181 := poseidon.Truncate128Reverse(api, cse_3)
	cse_182 := poseidon.Truncate128Reverse(api, cse_4)
	cse_183 := poseidon.Truncate128Reverse(api, cse_5)
	cse_184 := poseidon.Truncate128Reverse(api, cse_6)
	cse_185 := poseidon.Truncate128Reverse(api, cse_7)
	cse_186 := poseidon.Truncate128Reverse(api, cse_8)
	cse_187 := poseidon.Truncate128Reverse(api, cse_9)
	cse_188 := poseidon.Truncate128Reverse(api, cse_10)
	cse_189 := api.Mul(1, 362880)
	cse_190 := api.Mul(cse_189, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808455297"))
	cse_191 := api.Mul(cse_190, 10080)
	cse_192 := api.Mul(cse_191, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808491297"))
	cse_193 := api.Mul(cse_192, 2880)
	cse_194 := api.Mul(cse_193, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808492737"))
	cse_195 := api.Mul(cse_194, 4320)
	cse_196 := api.Mul(cse_195, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808485537"))
	cse_197 := api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808132737"))
	cse_198 := api.Mul(cse_197, 40320)
	cse_199 := api.Mul(cse_198, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808485537"))
	cse_200 := api.Mul(cse_199, 4320)
	cse_201 := api.Mul(cse_200, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808492737"))
	cse_202 := api.Mul(cse_201, 2880)
	cse_203 := api.Mul(cse_202, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808491297"))
	cse_204 := api.Mul(cse_203, 10080)
	cse_205 := api.Mul(cse_204, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808455297"))
	cse_206 := api.Inverse(api.Mul(cse_205, 362880))
	cse_207 := api.Sub(cse_13, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495613"))
	cse_208 := api.Sub(cse_207, 1)
	cse_209 := api.Sub(cse_208, 1)
	cse_210 := api.Sub(cse_209, 1)
	cse_211 := api.Sub(cse_210, 1)
	cse_212 := api.Sub(cse_211, 1)
	cse_213 := api.Sub(cse_212, 1)
	cse_214 := api.Sub(cse_213, 1)
	cse_215 := api.Sub(cse_214, 1)
	cse_216 := api.Sub(cse_215, 1)
	cse_217 := api.Mul(1, cse_216)
	cse_218 := api.Mul(cse_217, cse_215)
	cse_219 := api.Mul(cse_218, cse_214)
	cse_220 := api.Mul(cse_219, cse_213)
	cse_221 := api.Mul(cse_220, cse_212)
	cse_222 := api.Mul(cse_221, cse_211)
	cse_223 := api.Mul(cse_222, cse_210)
	cse_224 := api.Mul(cse_223, cse_209)
	cse_225 := api.Mul(1, cse_207)
	cse_226 := api.Mul(cse_225, cse_208)
	cse_227 := api.Mul(cse_226, cse_209)
	cse_228 := api.Mul(cse_227, cse_210)
	cse_229 := api.Mul(cse_228, cse_211)
	cse_230 := api.Mul(cse_229, cse_212)
	cse_231 := api.Mul(cse_230, cse_213)
	cse_232 := api.Mul(cse_231, cse_214)
	cse_233 := api.Mul(cse_232, cse_215)
	cse_234 := api.Inverse(api.Mul(cse_233, cse_216))
	cse_235 := api.Mul(api.Mul(api.Mul(1, api.Mul(cse_196, 40320)), cse_206), api.Mul(api.Mul(1, api.Mul(cse_224, cse_208)), cse_234))
	cse_236 := api.Mul(api.Mul(api.Mul(cse_197, cse_196), cse_206), api.Mul(api.Mul(cse_225, cse_224), cse_234))
	cse_237 := api.Mul(api.Mul(api.Mul(cse_198, cse_195), cse_206), api.Mul(api.Mul(cse_226, cse_223), cse_234))
	cse_238 := api.Mul(api.Mul(api.Mul(cse_199, cse_194), cse_206), api.Mul(api.Mul(cse_227, cse_222), cse_234))
	cse_239 := api.Mul(api.Mul(api.Mul(cse_200, cse_193), cse_206), api.Mul(api.Mul(cse_228, cse_221), cse_234))
	cse_240 := api.Mul(api.Mul(api.Mul(cse_201, cse_192), cse_206), api.Mul(api.Mul(cse_229, cse_220), cse_234))
	cse_241 := api.Mul(api.Mul(api.Mul(cse_202, cse_191), cse_206), api.Mul(api.Mul(cse_230, cse_219), cse_234))
	cse_242 := api.Mul(api.Mul(api.Mul(cse_203, cse_190), cse_206), api.Mul(api.Mul(cse_231, cse_218), cse_234))
	cse_243 := api.Mul(api.Mul(api.Mul(cse_204, cse_189), cse_206), api.Mul(api.Mul(cse_232, cse_217), cse_234))
	cse_244 := api.Mul(api.Mul(api.Mul(cse_205, 1), cse_206), api.Mul(api.Mul(cse_233, 1), cse_234))
	cse_245 := api.Inverse(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, cse_235), cse_236), cse_237), cse_238), cse_239), cse_240), cse_241), cse_242), cse_243), cse_244))
	cse_246 := api.Mul(cse_235, cse_245)
	cse_247 := api.Mul(cse_236, cse_245)
	cse_248 := api.Mul(cse_237, cse_245)
	cse_249 := api.Mul(cse_238, cse_245)
	cse_250 := api.Mul(cse_239, cse_245)
	cse_251 := api.Mul(cse_240, cse_245)
	cse_252 := api.Mul(cse_241, cse_245)
	cse_253 := api.Mul(cse_242, cse_245)
	cse_254 := api.Mul(cse_243, cse_245)
	cse_255 := api.Mul(cse_244, cse_245)
	cse_256 := api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, api.Mul(cse_246, api.Add(api.Add(api.Mul(circuit.R1csInput26, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")), api.Mul(circuit.R1csInput27, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(1, 1)))), api.Mul(cse_247, api.Mul(circuit.R1csInput26, 1))), api.Mul(cse_248, api.Mul(circuit.R1csInput26, 1))), api.Mul(cse_249, api.Mul(circuit.R1csInput27, 1))), api.Mul(cse_250, api.Add(api.Add(api.Mul(circuit.R1csInput23, 1), api.Mul(circuit.R1csInput24, 1)), api.Mul(circuit.R1csInput25, 1)))), api.Mul(cse_251, api.Add(api.Add(api.Add(api.Mul(circuit.R1csInput23, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")), api.Mul(circuit.R1csInput24, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.R1csInput25, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(1, 1)))), api.Mul(cse_252, api.Mul(circuit.R1csInput31, 1))), api.Mul(cse_253, api.Mul(circuit.R1csInput22, 1))), api.Mul(cse_254, api.Mul(circuit.R1csInput30, 1))), api.Mul(cse_255, api.Add(api.Mul(circuit.R1csInput19, 1), api.Mul(circuit.R1csInput20, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")))))
	cse_257 := api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, api.Mul(cse_246, api.Mul(circuit.R1csInput9, 1))), api.Mul(cse_247, api.Add(api.Mul(circuit.R1csInput13, 1), api.Mul(circuit.R1csInput14, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_248, api.Add(api.Mul(circuit.R1csInput13, 1), api.Mul(circuit.R1csInput12, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_249, api.Add(api.Mul(circuit.R1csInput11, 1), api.Mul(circuit.R1csInput14, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_250, api.Mul(circuit.R1csInput15, 1))), api.Mul(cse_251, api.Add(api.Mul(circuit.R1csInput15, 1), api.Mul(circuit.R1csInput0, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_252, api.Add(api.Mul(circuit.R1csInput21, 1), api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_253, api.Add(api.Mul(circuit.R1csInput17, 1), api.Mul(circuit.R1csInput21, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_254, api.Add(api.Add(api.Mul(circuit.R1csInput18, 1), api.Mul(circuit.R1csInput6, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_255, api.Add(api.Mul(circuit.R1csInput32, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")), api.Mul(1, 1))))

	// === DEBUG PRINTS ===
	api.Println("=== GO CIRCUIT DEBUG ===")
	api.Println("cse_11 (state after taus):", cse_11)
	api.Println("cse_13 (r0 challenge):", cse_13)
	api.Println("cse_39 (claim_after_uni_skip):", cse_39)
	api.Println("cse_41 (batching_coeff):", cse_41)
	api.Println("cse_43 (sumcheck_challenge[0]):", cse_43)
	api.Println("cse_46 (sumcheck_challenge[1]):", cse_46)
	api.Println("cse_72 (sumcheck_challenge[10]):", cse_72)
	api.Println("cse_178 (tau[0]):", cse_178)
	api.Println("cse_188 (tau[10]):", cse_188)

	// power_sum_check
	PowerSumCheck := api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, api.Mul(circuit.UniSkipCoeff0, 10)), api.Mul(circuit.UniSkipCoeff1, 5)), api.Mul(circuit.UniSkipCoeff2, 85)), api.Mul(circuit.UniSkipCoeff3, 125)), api.Mul(circuit.UniSkipCoeff4, 1333)), api.Mul(circuit.UniSkipCoeff5, 3125)), api.Mul(circuit.UniSkipCoeff6, 25405)), api.Mul(circuit.UniSkipCoeff7, 78125)), api.Mul(circuit.UniSkipCoeff8, 535333)), api.Mul(circuit.UniSkipCoeff9, 1953125)), api.Mul(circuit.UniSkipCoeff10, 11982925)), api.Mul(circuit.UniSkipCoeff11, 48828125)), api.Mul(circuit.UniSkipCoeff12, 278766133)), api.Mul(circuit.UniSkipCoeff13, 1220703125)), api.Mul(circuit.UniSkipCoeff14, 6649985245)), api.Mul(circuit.UniSkipCoeff15, 30517578125)), api.Mul(circuit.UniSkipCoeff16, 161264049733)), api.Mul(circuit.UniSkipCoeff17, 762939453125)), api.Mul(circuit.UniSkipCoeff18, 3952911584365)), api.Mul(circuit.UniSkipCoeff19, 19073486328125)), api.Mul(circuit.UniSkipCoeff20, 97573430562133)), api.Mul(circuit.UniSkipCoeff21, 476837158203125)), api.Mul(circuit.UniSkipCoeff22, 2419432933612285)), api.Mul(circuit.UniSkipCoeff23, 11920928955078125)), api.Mul(circuit.UniSkipCoeff24, 60168159621439333)), api.Mul(circuit.UniSkipCoeff25, 298023223876953125)), api.Mul(circuit.UniSkipCoeff26, 1499128402505381005)), api.Mul(circuit.UniSkipCoeff27, 7450580596923828125))
	api.Println("PowerSumCheck:", PowerSumCheck)
	api.AssertIsEqual(PowerSumCheck, 0)

	// final_check
	FinalCheck := api.Sub(api.Add(api.Add(api.Add(circuit.SumcheckR10C0, api.Mul(api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.SumcheckR9C0, api.Mul(api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.SumcheckR8C0, api.Mul(api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.SumcheckR7C0, api.Mul(api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.SumcheckR6C0, api.Mul(api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.SumcheckR5C0, api.Mul(api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.SumcheckR4C0, api.Mul(api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.SumcheckR3C0, api.Mul(api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.SumcheckR2C0, api.Mul(api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.SumcheckR1C0, api.Mul(api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.SumcheckR0C0, api.Mul(api.Sub(api.Sub(api.Sub(api.Sub(api.Mul(cse_39, cse_41), circuit.SumcheckR0C0), circuit.SumcheckR0C0), circuit.SumcheckR0C1), circuit.SumcheckR0C2), cse_43)), api.Mul(circuit.SumcheckR0C1, cse_44)), api.Mul(circuit.SumcheckR0C2, api.Mul(cse_44, cse_43))), circuit.SumcheckR1C0), circuit.SumcheckR1C0), circuit.SumcheckR1C1), circuit.SumcheckR1C2), cse_46)), api.Mul(circuit.SumcheckR1C1, cse_47)), api.Mul(circuit.SumcheckR1C2, api.Mul(cse_47, cse_46))), circuit.SumcheckR2C0), circuit.SumcheckR2C0), circuit.SumcheckR2C1), circuit.SumcheckR2C2), cse_49)), api.Mul(circuit.SumcheckR2C1, cse_50)), api.Mul(circuit.SumcheckR2C2, api.Mul(cse_50, cse_49))), circuit.SumcheckR3C0), circuit.SumcheckR3C0), circuit.SumcheckR3C1), circuit.SumcheckR3C2), cse_52)), api.Mul(circuit.SumcheckR3C1, cse_53)), api.Mul(circuit.SumcheckR3C2, api.Mul(cse_53, cse_52))), circuit.SumcheckR4C0), circuit.SumcheckR4C0), circuit.SumcheckR4C1), circuit.SumcheckR4C2), cse_55)), api.Mul(circuit.SumcheckR4C1, cse_56)), api.Mul(circuit.SumcheckR4C2, api.Mul(cse_56, cse_55))), circuit.SumcheckR5C0), circuit.SumcheckR5C0), circuit.SumcheckR5C1), circuit.SumcheckR5C2), cse_58)), api.Mul(circuit.SumcheckR5C1, cse_59)), api.Mul(circuit.SumcheckR5C2, api.Mul(cse_59, cse_58))), circuit.SumcheckR6C0), circuit.SumcheckR6C0), circuit.SumcheckR6C1), circuit.SumcheckR6C2), cse_61)), api.Mul(circuit.SumcheckR6C1, cse_62)), api.Mul(circuit.SumcheckR6C2, api.Mul(cse_62, cse_61))), circuit.SumcheckR7C0), circuit.SumcheckR7C0), circuit.SumcheckR7C1), circuit.SumcheckR7C2), cse_64)), api.Mul(circuit.SumcheckR7C1, cse_65)), api.Mul(circuit.SumcheckR7C2, api.Mul(cse_65, cse_64))), circuit.SumcheckR8C0), circuit.SumcheckR8C0), circuit.SumcheckR8C1), circuit.SumcheckR8C2), cse_67)), api.Mul(circuit.SumcheckR8C1, cse_68)), api.Mul(circuit.SumcheckR8C2, api.Mul(cse_68, cse_67))), circuit.SumcheckR9C0), circuit.SumcheckR9C0), circuit.SumcheckR9C1), circuit.SumcheckR9C2), cse_70)), api.Mul(circuit.SumcheckR9C1, cse_71)), api.Mul(circuit.SumcheckR9C2, api.Mul(cse_71, cse_70))), circuit.SumcheckR10C0), circuit.SumcheckR10C0), circuit.SumcheckR10C1), circuit.SumcheckR10C2), cse_72)), api.Mul(circuit.SumcheckR10C1, cse_73)), api.Mul(circuit.SumcheckR10C2, api.Mul(cse_73, cse_72))), api.Mul(api.Mul(api.Mul(api.Mul(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, api.Mul(cse_121, cse_150)), api.Mul(cse_152, cse_153)), api.Mul(cse_155, cse_156)), api.Mul(cse_158, cse_159)), api.Mul(cse_161, cse_162)), api.Mul(cse_164, cse_165)), api.Mul(cse_167, cse_168)), api.Mul(cse_170, cse_171)), api.Mul(cse_173, cse_174)), api.Mul(cse_176, cse_177)), api.Inverse(api.Mul(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, cse_121), cse_152), cse_155), cse_158), cse_161), cse_164), cse_167), cse_170), cse_173), cse_176), api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, cse_150), cse_153), cse_156), cse_159), cse_162), cse_165), cse_168), cse_171), cse_174), cse_177)))), api.Mul(api.Mul(api.Mul(api.Mul(1, api.Add(api.Mul(cse_178, cse_72), api.Mul(api.Sub(1, cse_178), api.Sub(1, cse_72)))), api.Mul(1, api.Add(api.Mul(cse_179, cse_70), api.Mul(api.Sub(1, cse_179), api.Sub(1, cse_70))))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_180, cse_67), api.Mul(api.Sub(1, cse_180), api.Sub(1, cse_67)))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_181, cse_64), api.Mul(api.Sub(1, cse_181), api.Sub(1, cse_64)))), api.Mul(1, api.Add(api.Mul(cse_182, cse_61), api.Mul(api.Sub(1, cse_182), api.Sub(1, cse_61))))))), api.Mul(api.Mul(api.Mul(1, api.Add(api.Mul(cse_183, cse_58), api.Mul(api.Sub(1, cse_183), api.Sub(1, cse_58)))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_184, cse_55), api.Mul(api.Sub(1, cse_184), api.Sub(1, cse_55)))), api.Mul(1, api.Add(api.Mul(cse_185, cse_52), api.Mul(api.Sub(1, cse_185), api.Sub(1, cse_52)))))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_186, cse_49), api.Mul(api.Sub(1, cse_186), api.Sub(1, cse_49)))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_187, cse_46), api.Mul(api.Sub(1, cse_187), api.Sub(1, cse_46)))), api.Mul(1, api.Add(api.Mul(cse_188, cse_43), api.Mul(api.Sub(1, cse_188), api.Sub(1, cse_43))))))))), api.Mul(api.Add(cse_256, api.Mul(cse_43, api.Sub(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, api.Mul(cse_246, api.Add(api.Mul(circuit.R1csInput26, 1), api.Mul(circuit.R1csInput27, 1)))), api.Mul(cse_247, api.Mul(circuit.R1csInput23, 1))), api.Mul(cse_248, api.Mul(circuit.R1csInput24, 1))), api.Mul(cse_249, api.Mul(circuit.R1csInput25, 1))), api.Mul(cse_250, api.Add(api.Add(api.Add(api.Add(api.Mul(circuit.R1csInput23, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")), api.Mul(circuit.R1csInput24, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.R1csInput25, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.R1csInput33, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(1, 1)))), api.Mul(cse_251, api.Mul(circuit.R1csInput3, 1))), api.Mul(cse_252, api.Mul(circuit.R1csInput4, 1))), api.Mul(cse_253, api.Mul(circuit.R1csInput5, 1))), api.Mul(cse_254, api.Add(api.Add(api.Mul(circuit.R1csInput5, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")), api.Mul(circuit.R1csInput28, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(1, 1)))), cse_256))), api.Add(cse_257, api.Mul(cse_43, api.Sub(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, api.Mul(cse_246, api.Add(api.Add(api.Mul(circuit.R1csInput9, 1), api.Mul(circuit.R1csInput10, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.R1csInput8, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_247, api.Add(api.Add(api.Mul(circuit.R1csInput16, 1), api.Mul(circuit.R1csInput0, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.R1csInput1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_248, api.Add(api.Add(api.Add(api.Mul(circuit.R1csInput16, 1), api.Mul(circuit.R1csInput0, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.R1csInput1, 1)), api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343679757442502098944001"))))), api.Mul(cse_249, api.Add(api.Mul(circuit.R1csInput16, 1), api.Mul(circuit.R1csInput2, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_250, api.Add(api.Mul(circuit.R1csInput16, 1), api.Mul(circuit.R1csInput1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_251, api.Add(api.Mul(circuit.R1csInput12, 1), api.Mul(circuit.R1csInput21, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_252, api.Add(api.Add(api.Add(api.Mul(circuit.R1csInput12, 1), api.Mul(circuit.R1csInput7, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.R1csInput34, 2)), api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495613"))))), api.Mul(cse_253, api.Add(api.Add(api.Mul(circuit.R1csInput17, 1), api.Mul(circuit.R1csInput7, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.R1csInput8, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_254, api.Add(api.Add(api.Add(api.Add(api.Mul(circuit.R1csInput17, 1), api.Mul(circuit.R1csInput7, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.R1csInput32, 4)), api.Mul(circuit.R1csInput34, 2)), api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495613"))))), cse_257))))), cse_41))
	api.Println("FinalCheck:", FinalCheck)
	api.AssertIsEqual(FinalCheck, 0)

	return nil
}
