package jolt_verifier

import (
	"github.com/consensys/gnark/frontend"
	"jolt_verifier/poseidon"
)

type JoltStages16Circuit struct {
	Commitment_0_0 frontend.Variable `gnark:",public"`
	Commitment_0_1 frontend.Variable `gnark:",public"`
	Commitment_0_2 frontend.Variable `gnark:",public"`
	Commitment_0_3 frontend.Variable `gnark:",public"`
	Commitment_0_4 frontend.Variable `gnark:",public"`
	Commitment_0_5 frontend.Variable `gnark:",public"`
	Commitment_0_6 frontend.Variable `gnark:",public"`
	Commitment_0_7 frontend.Variable `gnark:",public"`
	Commitment_0_8 frontend.Variable `gnark:",public"`
	Commitment_0_9 frontend.Variable `gnark:",public"`
	Commitment_0_10 frontend.Variable `gnark:",public"`
	Commitment_0_11 frontend.Variable `gnark:",public"`
	Commitment_1_0 frontend.Variable `gnark:",public"`
	Commitment_1_1 frontend.Variable `gnark:",public"`
	Commitment_1_2 frontend.Variable `gnark:",public"`
	Commitment_1_3 frontend.Variable `gnark:",public"`
	Commitment_1_4 frontend.Variable `gnark:",public"`
	Commitment_1_5 frontend.Variable `gnark:",public"`
	Commitment_1_6 frontend.Variable `gnark:",public"`
	Commitment_1_7 frontend.Variable `gnark:",public"`
	Commitment_1_8 frontend.Variable `gnark:",public"`
	Commitment_1_9 frontend.Variable `gnark:",public"`
	Commitment_1_10 frontend.Variable `gnark:",public"`
	Commitment_1_11 frontend.Variable `gnark:",public"`
	Commitment_2_0 frontend.Variable `gnark:",public"`
	Commitment_2_1 frontend.Variable `gnark:",public"`
	Commitment_2_2 frontend.Variable `gnark:",public"`
	Commitment_2_3 frontend.Variable `gnark:",public"`
	Commitment_2_4 frontend.Variable `gnark:",public"`
	Commitment_2_5 frontend.Variable `gnark:",public"`
	Commitment_2_6 frontend.Variable `gnark:",public"`
	Commitment_2_7 frontend.Variable `gnark:",public"`
	Commitment_2_8 frontend.Variable `gnark:",public"`
	Commitment_2_9 frontend.Variable `gnark:",public"`
	Commitment_2_10 frontend.Variable `gnark:",public"`
	Commitment_2_11 frontend.Variable `gnark:",public"`
	Commitment_3_0 frontend.Variable `gnark:",public"`
	Commitment_3_1 frontend.Variable `gnark:",public"`
	Commitment_3_2 frontend.Variable `gnark:",public"`
	Commitment_3_3 frontend.Variable `gnark:",public"`
	Commitment_3_4 frontend.Variable `gnark:",public"`
	Commitment_3_5 frontend.Variable `gnark:",public"`
	Commitment_3_6 frontend.Variable `gnark:",public"`
	Commitment_3_7 frontend.Variable `gnark:",public"`
	Commitment_3_8 frontend.Variable `gnark:",public"`
	Commitment_3_9 frontend.Variable `gnark:",public"`
	Commitment_3_10 frontend.Variable `gnark:",public"`
	Commitment_3_11 frontend.Variable `gnark:",public"`
	Commitment_4_0 frontend.Variable `gnark:",public"`
	Commitment_4_1 frontend.Variable `gnark:",public"`
	Commitment_4_2 frontend.Variable `gnark:",public"`
	Commitment_4_3 frontend.Variable `gnark:",public"`
	Commitment_4_4 frontend.Variable `gnark:",public"`
	Commitment_4_5 frontend.Variable `gnark:",public"`
	Commitment_4_6 frontend.Variable `gnark:",public"`
	Commitment_4_7 frontend.Variable `gnark:",public"`
	Commitment_4_8 frontend.Variable `gnark:",public"`
	Commitment_4_9 frontend.Variable `gnark:",public"`
	Commitment_4_10 frontend.Variable `gnark:",public"`
	Commitment_4_11 frontend.Variable `gnark:",public"`
	Commitment_5_0 frontend.Variable `gnark:",public"`
	Commitment_5_1 frontend.Variable `gnark:",public"`
	Commitment_5_2 frontend.Variable `gnark:",public"`
	Commitment_5_3 frontend.Variable `gnark:",public"`
	Commitment_5_4 frontend.Variable `gnark:",public"`
	Commitment_5_5 frontend.Variable `gnark:",public"`
	Commitment_5_6 frontend.Variable `gnark:",public"`
	Commitment_5_7 frontend.Variable `gnark:",public"`
	Commitment_5_8 frontend.Variable `gnark:",public"`
	Commitment_5_9 frontend.Variable `gnark:",public"`
	Commitment_5_10 frontend.Variable `gnark:",public"`
	Commitment_5_11 frontend.Variable `gnark:",public"`
	Commitment_6_0 frontend.Variable `gnark:",public"`
	Commitment_6_1 frontend.Variable `gnark:",public"`
	Commitment_6_2 frontend.Variable `gnark:",public"`
	Commitment_6_3 frontend.Variable `gnark:",public"`
	Commitment_6_4 frontend.Variable `gnark:",public"`
	Commitment_6_5 frontend.Variable `gnark:",public"`
	Commitment_6_6 frontend.Variable `gnark:",public"`
	Commitment_6_7 frontend.Variable `gnark:",public"`
	Commitment_6_8 frontend.Variable `gnark:",public"`
	Commitment_6_9 frontend.Variable `gnark:",public"`
	Commitment_6_10 frontend.Variable `gnark:",public"`
	Commitment_6_11 frontend.Variable `gnark:",public"`
	Commitment_7_0 frontend.Variable `gnark:",public"`
	Commitment_7_1 frontend.Variable `gnark:",public"`
	Commitment_7_2 frontend.Variable `gnark:",public"`
	Commitment_7_3 frontend.Variable `gnark:",public"`
	Commitment_7_4 frontend.Variable `gnark:",public"`
	Commitment_7_5 frontend.Variable `gnark:",public"`
	Commitment_7_6 frontend.Variable `gnark:",public"`
	Commitment_7_7 frontend.Variable `gnark:",public"`
	Commitment_7_8 frontend.Variable `gnark:",public"`
	Commitment_7_9 frontend.Variable `gnark:",public"`
	Commitment_7_10 frontend.Variable `gnark:",public"`
	Commitment_7_11 frontend.Variable `gnark:",public"`
	Commitment_8_0 frontend.Variable `gnark:",public"`
	Commitment_8_1 frontend.Variable `gnark:",public"`
	Commitment_8_2 frontend.Variable `gnark:",public"`
	Commitment_8_3 frontend.Variable `gnark:",public"`
	Commitment_8_4 frontend.Variable `gnark:",public"`
	Commitment_8_5 frontend.Variable `gnark:",public"`
	Commitment_8_6 frontend.Variable `gnark:",public"`
	Commitment_8_7 frontend.Variable `gnark:",public"`
	Commitment_8_8 frontend.Variable `gnark:",public"`
	Commitment_8_9 frontend.Variable `gnark:",public"`
	Commitment_8_10 frontend.Variable `gnark:",public"`
	Commitment_8_11 frontend.Variable `gnark:",public"`
	Commitment_9_0 frontend.Variable `gnark:",public"`
	Commitment_9_1 frontend.Variable `gnark:",public"`
	Commitment_9_2 frontend.Variable `gnark:",public"`
	Commitment_9_3 frontend.Variable `gnark:",public"`
	Commitment_9_4 frontend.Variable `gnark:",public"`
	Commitment_9_5 frontend.Variable `gnark:",public"`
	Commitment_9_6 frontend.Variable `gnark:",public"`
	Commitment_9_7 frontend.Variable `gnark:",public"`
	Commitment_9_8 frontend.Variable `gnark:",public"`
	Commitment_9_9 frontend.Variable `gnark:",public"`
	Commitment_9_10 frontend.Variable `gnark:",public"`
	Commitment_9_11 frontend.Variable `gnark:",public"`
	Commitment_10_0 frontend.Variable `gnark:",public"`
	Commitment_10_1 frontend.Variable `gnark:",public"`
	Commitment_10_2 frontend.Variable `gnark:",public"`
	Commitment_10_3 frontend.Variable `gnark:",public"`
	Commitment_10_4 frontend.Variable `gnark:",public"`
	Commitment_10_5 frontend.Variable `gnark:",public"`
	Commitment_10_6 frontend.Variable `gnark:",public"`
	Commitment_10_7 frontend.Variable `gnark:",public"`
	Commitment_10_8 frontend.Variable `gnark:",public"`
	Commitment_10_9 frontend.Variable `gnark:",public"`
	Commitment_10_10 frontend.Variable `gnark:",public"`
	Commitment_10_11 frontend.Variable `gnark:",public"`
	Commitment_11_0 frontend.Variable `gnark:",public"`
	Commitment_11_1 frontend.Variable `gnark:",public"`
	Commitment_11_2 frontend.Variable `gnark:",public"`
	Commitment_11_3 frontend.Variable `gnark:",public"`
	Commitment_11_4 frontend.Variable `gnark:",public"`
	Commitment_11_5 frontend.Variable `gnark:",public"`
	Commitment_11_6 frontend.Variable `gnark:",public"`
	Commitment_11_7 frontend.Variable `gnark:",public"`
	Commitment_11_8 frontend.Variable `gnark:",public"`
	Commitment_11_9 frontend.Variable `gnark:",public"`
	Commitment_11_10 frontend.Variable `gnark:",public"`
	Commitment_11_11 frontend.Variable `gnark:",public"`
	Commitment_12_0 frontend.Variable `gnark:",public"`
	Commitment_12_1 frontend.Variable `gnark:",public"`
	Commitment_12_2 frontend.Variable `gnark:",public"`
	Commitment_12_3 frontend.Variable `gnark:",public"`
	Commitment_12_4 frontend.Variable `gnark:",public"`
	Commitment_12_5 frontend.Variable `gnark:",public"`
	Commitment_12_6 frontend.Variable `gnark:",public"`
	Commitment_12_7 frontend.Variable `gnark:",public"`
	Commitment_12_8 frontend.Variable `gnark:",public"`
	Commitment_12_9 frontend.Variable `gnark:",public"`
	Commitment_12_10 frontend.Variable `gnark:",public"`
	Commitment_12_11 frontend.Variable `gnark:",public"`
	Commitment_13_0 frontend.Variable `gnark:",public"`
	Commitment_13_1 frontend.Variable `gnark:",public"`
	Commitment_13_2 frontend.Variable `gnark:",public"`
	Commitment_13_3 frontend.Variable `gnark:",public"`
	Commitment_13_4 frontend.Variable `gnark:",public"`
	Commitment_13_5 frontend.Variable `gnark:",public"`
	Commitment_13_6 frontend.Variable `gnark:",public"`
	Commitment_13_7 frontend.Variable `gnark:",public"`
	Commitment_13_8 frontend.Variable `gnark:",public"`
	Commitment_13_9 frontend.Variable `gnark:",public"`
	Commitment_13_10 frontend.Variable `gnark:",public"`
	Commitment_13_11 frontend.Variable `gnark:",public"`
	Commitment_14_0 frontend.Variable `gnark:",public"`
	Commitment_14_1 frontend.Variable `gnark:",public"`
	Commitment_14_2 frontend.Variable `gnark:",public"`
	Commitment_14_3 frontend.Variable `gnark:",public"`
	Commitment_14_4 frontend.Variable `gnark:",public"`
	Commitment_14_5 frontend.Variable `gnark:",public"`
	Commitment_14_6 frontend.Variable `gnark:",public"`
	Commitment_14_7 frontend.Variable `gnark:",public"`
	Commitment_14_8 frontend.Variable `gnark:",public"`
	Commitment_14_9 frontend.Variable `gnark:",public"`
	Commitment_14_10 frontend.Variable `gnark:",public"`
	Commitment_14_11 frontend.Variable `gnark:",public"`
	Commitment_15_0 frontend.Variable `gnark:",public"`
	Commitment_15_1 frontend.Variable `gnark:",public"`
	Commitment_15_2 frontend.Variable `gnark:",public"`
	Commitment_15_3 frontend.Variable `gnark:",public"`
	Commitment_15_4 frontend.Variable `gnark:",public"`
	Commitment_15_5 frontend.Variable `gnark:",public"`
	Commitment_15_6 frontend.Variable `gnark:",public"`
	Commitment_15_7 frontend.Variable `gnark:",public"`
	Commitment_15_8 frontend.Variable `gnark:",public"`
	Commitment_15_9 frontend.Variable `gnark:",public"`
	Commitment_15_10 frontend.Variable `gnark:",public"`
	Commitment_15_11 frontend.Variable `gnark:",public"`
	Commitment_16_0 frontend.Variable `gnark:",public"`
	Commitment_16_1 frontend.Variable `gnark:",public"`
	Commitment_16_2 frontend.Variable `gnark:",public"`
	Commitment_16_3 frontend.Variable `gnark:",public"`
	Commitment_16_4 frontend.Variable `gnark:",public"`
	Commitment_16_5 frontend.Variable `gnark:",public"`
	Commitment_16_6 frontend.Variable `gnark:",public"`
	Commitment_16_7 frontend.Variable `gnark:",public"`
	Commitment_16_8 frontend.Variable `gnark:",public"`
	Commitment_16_9 frontend.Variable `gnark:",public"`
	Commitment_16_10 frontend.Variable `gnark:",public"`
	Commitment_16_11 frontend.Variable `gnark:",public"`
	Commitment_17_0 frontend.Variable `gnark:",public"`
	Commitment_17_1 frontend.Variable `gnark:",public"`
	Commitment_17_2 frontend.Variable `gnark:",public"`
	Commitment_17_3 frontend.Variable `gnark:",public"`
	Commitment_17_4 frontend.Variable `gnark:",public"`
	Commitment_17_5 frontend.Variable `gnark:",public"`
	Commitment_17_6 frontend.Variable `gnark:",public"`
	Commitment_17_7 frontend.Variable `gnark:",public"`
	Commitment_17_8 frontend.Variable `gnark:",public"`
	Commitment_17_9 frontend.Variable `gnark:",public"`
	Commitment_17_10 frontend.Variable `gnark:",public"`
	Commitment_17_11 frontend.Variable `gnark:",public"`
	Commitment_18_0 frontend.Variable `gnark:",public"`
	Commitment_18_1 frontend.Variable `gnark:",public"`
	Commitment_18_2 frontend.Variable `gnark:",public"`
	Commitment_18_3 frontend.Variable `gnark:",public"`
	Commitment_18_4 frontend.Variable `gnark:",public"`
	Commitment_18_5 frontend.Variable `gnark:",public"`
	Commitment_18_6 frontend.Variable `gnark:",public"`
	Commitment_18_7 frontend.Variable `gnark:",public"`
	Commitment_18_8 frontend.Variable `gnark:",public"`
	Commitment_18_9 frontend.Variable `gnark:",public"`
	Commitment_18_10 frontend.Variable `gnark:",public"`
	Commitment_18_11 frontend.Variable `gnark:",public"`
	Commitment_19_0 frontend.Variable `gnark:",public"`
	Commitment_19_1 frontend.Variable `gnark:",public"`
	Commitment_19_2 frontend.Variable `gnark:",public"`
	Commitment_19_3 frontend.Variable `gnark:",public"`
	Commitment_19_4 frontend.Variable `gnark:",public"`
	Commitment_19_5 frontend.Variable `gnark:",public"`
	Commitment_19_6 frontend.Variable `gnark:",public"`
	Commitment_19_7 frontend.Variable `gnark:",public"`
	Commitment_19_8 frontend.Variable `gnark:",public"`
	Commitment_19_9 frontend.Variable `gnark:",public"`
	Commitment_19_10 frontend.Variable `gnark:",public"`
	Commitment_19_11 frontend.Variable `gnark:",public"`
	Commitment_20_0 frontend.Variable `gnark:",public"`
	Commitment_20_1 frontend.Variable `gnark:",public"`
	Commitment_20_2 frontend.Variable `gnark:",public"`
	Commitment_20_3 frontend.Variable `gnark:",public"`
	Commitment_20_4 frontend.Variable `gnark:",public"`
	Commitment_20_5 frontend.Variable `gnark:",public"`
	Commitment_20_6 frontend.Variable `gnark:",public"`
	Commitment_20_7 frontend.Variable `gnark:",public"`
	Commitment_20_8 frontend.Variable `gnark:",public"`
	Commitment_20_9 frontend.Variable `gnark:",public"`
	Commitment_20_10 frontend.Variable `gnark:",public"`
	Commitment_20_11 frontend.Variable `gnark:",public"`
	Commitment_21_0 frontend.Variable `gnark:",public"`
	Commitment_21_1 frontend.Variable `gnark:",public"`
	Commitment_21_2 frontend.Variable `gnark:",public"`
	Commitment_21_3 frontend.Variable `gnark:",public"`
	Commitment_21_4 frontend.Variable `gnark:",public"`
	Commitment_21_5 frontend.Variable `gnark:",public"`
	Commitment_21_6 frontend.Variable `gnark:",public"`
	Commitment_21_7 frontend.Variable `gnark:",public"`
	Commitment_21_8 frontend.Variable `gnark:",public"`
	Commitment_21_9 frontend.Variable `gnark:",public"`
	Commitment_21_10 frontend.Variable `gnark:",public"`
	Commitment_21_11 frontend.Variable `gnark:",public"`
	Commitment_22_0 frontend.Variable `gnark:",public"`
	Commitment_22_1 frontend.Variable `gnark:",public"`
	Commitment_22_2 frontend.Variable `gnark:",public"`
	Commitment_22_3 frontend.Variable `gnark:",public"`
	Commitment_22_4 frontend.Variable `gnark:",public"`
	Commitment_22_5 frontend.Variable `gnark:",public"`
	Commitment_22_6 frontend.Variable `gnark:",public"`
	Commitment_22_7 frontend.Variable `gnark:",public"`
	Commitment_22_8 frontend.Variable `gnark:",public"`
	Commitment_22_9 frontend.Variable `gnark:",public"`
	Commitment_22_10 frontend.Variable `gnark:",public"`
	Commitment_22_11 frontend.Variable `gnark:",public"`
	Commitment_23_0 frontend.Variable `gnark:",public"`
	Commitment_23_1 frontend.Variable `gnark:",public"`
	Commitment_23_2 frontend.Variable `gnark:",public"`
	Commitment_23_3 frontend.Variable `gnark:",public"`
	Commitment_23_4 frontend.Variable `gnark:",public"`
	Commitment_23_5 frontend.Variable `gnark:",public"`
	Commitment_23_6 frontend.Variable `gnark:",public"`
	Commitment_23_7 frontend.Variable `gnark:",public"`
	Commitment_23_8 frontend.Variable `gnark:",public"`
	Commitment_23_9 frontend.Variable `gnark:",public"`
	Commitment_23_10 frontend.Variable `gnark:",public"`
	Commitment_23_11 frontend.Variable `gnark:",public"`
	Commitment_24_0 frontend.Variable `gnark:",public"`
	Commitment_24_1 frontend.Variable `gnark:",public"`
	Commitment_24_2 frontend.Variable `gnark:",public"`
	Commitment_24_3 frontend.Variable `gnark:",public"`
	Commitment_24_4 frontend.Variable `gnark:",public"`
	Commitment_24_5 frontend.Variable `gnark:",public"`
	Commitment_24_6 frontend.Variable `gnark:",public"`
	Commitment_24_7 frontend.Variable `gnark:",public"`
	Commitment_24_8 frontend.Variable `gnark:",public"`
	Commitment_24_9 frontend.Variable `gnark:",public"`
	Commitment_24_10 frontend.Variable `gnark:",public"`
	Commitment_24_11 frontend.Variable `gnark:",public"`
	Commitment_25_0 frontend.Variable `gnark:",public"`
	Commitment_25_1 frontend.Variable `gnark:",public"`
	Commitment_25_2 frontend.Variable `gnark:",public"`
	Commitment_25_3 frontend.Variable `gnark:",public"`
	Commitment_25_4 frontend.Variable `gnark:",public"`
	Commitment_25_5 frontend.Variable `gnark:",public"`
	Commitment_25_6 frontend.Variable `gnark:",public"`
	Commitment_25_7 frontend.Variable `gnark:",public"`
	Commitment_25_8 frontend.Variable `gnark:",public"`
	Commitment_25_9 frontend.Variable `gnark:",public"`
	Commitment_25_10 frontend.Variable `gnark:",public"`
	Commitment_25_11 frontend.Variable `gnark:",public"`
	Commitment_26_0 frontend.Variable `gnark:",public"`
	Commitment_26_1 frontend.Variable `gnark:",public"`
	Commitment_26_2 frontend.Variable `gnark:",public"`
	Commitment_26_3 frontend.Variable `gnark:",public"`
	Commitment_26_4 frontend.Variable `gnark:",public"`
	Commitment_26_5 frontend.Variable `gnark:",public"`
	Commitment_26_6 frontend.Variable `gnark:",public"`
	Commitment_26_7 frontend.Variable `gnark:",public"`
	Commitment_26_8 frontend.Variable `gnark:",public"`
	Commitment_26_9 frontend.Variable `gnark:",public"`
	Commitment_26_10 frontend.Variable `gnark:",public"`
	Commitment_26_11 frontend.Variable `gnark:",public"`
	Commitment_27_0 frontend.Variable `gnark:",public"`
	Commitment_27_1 frontend.Variable `gnark:",public"`
	Commitment_27_2 frontend.Variable `gnark:",public"`
	Commitment_27_3 frontend.Variable `gnark:",public"`
	Commitment_27_4 frontend.Variable `gnark:",public"`
	Commitment_27_5 frontend.Variable `gnark:",public"`
	Commitment_27_6 frontend.Variable `gnark:",public"`
	Commitment_27_7 frontend.Variable `gnark:",public"`
	Commitment_27_8 frontend.Variable `gnark:",public"`
	Commitment_27_9 frontend.Variable `gnark:",public"`
	Commitment_27_10 frontend.Variable `gnark:",public"`
	Commitment_27_11 frontend.Variable `gnark:",public"`
	Commitment_28_0 frontend.Variable `gnark:",public"`
	Commitment_28_1 frontend.Variable `gnark:",public"`
	Commitment_28_2 frontend.Variable `gnark:",public"`
	Commitment_28_3 frontend.Variable `gnark:",public"`
	Commitment_28_4 frontend.Variable `gnark:",public"`
	Commitment_28_5 frontend.Variable `gnark:",public"`
	Commitment_28_6 frontend.Variable `gnark:",public"`
	Commitment_28_7 frontend.Variable `gnark:",public"`
	Commitment_28_8 frontend.Variable `gnark:",public"`
	Commitment_28_9 frontend.Variable `gnark:",public"`
	Commitment_28_10 frontend.Variable `gnark:",public"`
	Commitment_28_11 frontend.Variable `gnark:",public"`
	Commitment_29_0 frontend.Variable `gnark:",public"`
	Commitment_29_1 frontend.Variable `gnark:",public"`
	Commitment_29_2 frontend.Variable `gnark:",public"`
	Commitment_29_3 frontend.Variable `gnark:",public"`
	Commitment_29_4 frontend.Variable `gnark:",public"`
	Commitment_29_5 frontend.Variable `gnark:",public"`
	Commitment_29_6 frontend.Variable `gnark:",public"`
	Commitment_29_7 frontend.Variable `gnark:",public"`
	Commitment_29_8 frontend.Variable `gnark:",public"`
	Commitment_29_9 frontend.Variable `gnark:",public"`
	Commitment_29_10 frontend.Variable `gnark:",public"`
	Commitment_29_11 frontend.Variable `gnark:",public"`
	Commitment_30_0 frontend.Variable `gnark:",public"`
	Commitment_30_1 frontend.Variable `gnark:",public"`
	Commitment_30_2 frontend.Variable `gnark:",public"`
	Commitment_30_3 frontend.Variable `gnark:",public"`
	Commitment_30_4 frontend.Variable `gnark:",public"`
	Commitment_30_5 frontend.Variable `gnark:",public"`
	Commitment_30_6 frontend.Variable `gnark:",public"`
	Commitment_30_7 frontend.Variable `gnark:",public"`
	Commitment_30_8 frontend.Variable `gnark:",public"`
	Commitment_30_9 frontend.Variable `gnark:",public"`
	Commitment_30_10 frontend.Variable `gnark:",public"`
	Commitment_30_11 frontend.Variable `gnark:",public"`
	Commitment_31_0 frontend.Variable `gnark:",public"`
	Commitment_31_1 frontend.Variable `gnark:",public"`
	Commitment_31_2 frontend.Variable `gnark:",public"`
	Commitment_31_3 frontend.Variable `gnark:",public"`
	Commitment_31_4 frontend.Variable `gnark:",public"`
	Commitment_31_5 frontend.Variable `gnark:",public"`
	Commitment_31_6 frontend.Variable `gnark:",public"`
	Commitment_31_7 frontend.Variable `gnark:",public"`
	Commitment_31_8 frontend.Variable `gnark:",public"`
	Commitment_31_9 frontend.Variable `gnark:",public"`
	Commitment_31_10 frontend.Variable `gnark:",public"`
	Commitment_31_11 frontend.Variable `gnark:",public"`
	Commitment_32_0 frontend.Variable `gnark:",public"`
	Commitment_32_1 frontend.Variable `gnark:",public"`
	Commitment_32_2 frontend.Variable `gnark:",public"`
	Commitment_32_3 frontend.Variable `gnark:",public"`
	Commitment_32_4 frontend.Variable `gnark:",public"`
	Commitment_32_5 frontend.Variable `gnark:",public"`
	Commitment_32_6 frontend.Variable `gnark:",public"`
	Commitment_32_7 frontend.Variable `gnark:",public"`
	Commitment_32_8 frontend.Variable `gnark:",public"`
	Commitment_32_9 frontend.Variable `gnark:",public"`
	Commitment_32_10 frontend.Variable `gnark:",public"`
	Commitment_32_11 frontend.Variable `gnark:",public"`
	Commitment_33_0 frontend.Variable `gnark:",public"`
	Commitment_33_1 frontend.Variable `gnark:",public"`
	Commitment_33_2 frontend.Variable `gnark:",public"`
	Commitment_33_3 frontend.Variable `gnark:",public"`
	Commitment_33_4 frontend.Variable `gnark:",public"`
	Commitment_33_5 frontend.Variable `gnark:",public"`
	Commitment_33_6 frontend.Variable `gnark:",public"`
	Commitment_33_7 frontend.Variable `gnark:",public"`
	Commitment_33_8 frontend.Variable `gnark:",public"`
	Commitment_33_9 frontend.Variable `gnark:",public"`
	Commitment_33_10 frontend.Variable `gnark:",public"`
	Commitment_33_11 frontend.Variable `gnark:",public"`
	Commitment_34_0 frontend.Variable `gnark:",public"`
	Commitment_34_1 frontend.Variable `gnark:",public"`
	Commitment_34_2 frontend.Variable `gnark:",public"`
	Commitment_34_3 frontend.Variable `gnark:",public"`
	Commitment_34_4 frontend.Variable `gnark:",public"`
	Commitment_34_5 frontend.Variable `gnark:",public"`
	Commitment_34_6 frontend.Variable `gnark:",public"`
	Commitment_34_7 frontend.Variable `gnark:",public"`
	Commitment_34_8 frontend.Variable `gnark:",public"`
	Commitment_34_9 frontend.Variable `gnark:",public"`
	Commitment_34_10 frontend.Variable `gnark:",public"`
	Commitment_34_11 frontend.Variable `gnark:",public"`
	Commitment_35_0 frontend.Variable `gnark:",public"`
	Commitment_35_1 frontend.Variable `gnark:",public"`
	Commitment_35_2 frontend.Variable `gnark:",public"`
	Commitment_35_3 frontend.Variable `gnark:",public"`
	Commitment_35_4 frontend.Variable `gnark:",public"`
	Commitment_35_5 frontend.Variable `gnark:",public"`
	Commitment_35_6 frontend.Variable `gnark:",public"`
	Commitment_35_7 frontend.Variable `gnark:",public"`
	Commitment_35_8 frontend.Variable `gnark:",public"`
	Commitment_35_9 frontend.Variable `gnark:",public"`
	Commitment_35_10 frontend.Variable `gnark:",public"`
	Commitment_35_11 frontend.Variable `gnark:",public"`
	Commitment_36_0 frontend.Variable `gnark:",public"`
	Commitment_36_1 frontend.Variable `gnark:",public"`
	Commitment_36_2 frontend.Variable `gnark:",public"`
	Commitment_36_3 frontend.Variable `gnark:",public"`
	Commitment_36_4 frontend.Variable `gnark:",public"`
	Commitment_36_5 frontend.Variable `gnark:",public"`
	Commitment_36_6 frontend.Variable `gnark:",public"`
	Commitment_36_7 frontend.Variable `gnark:",public"`
	Commitment_36_8 frontend.Variable `gnark:",public"`
	Commitment_36_9 frontend.Variable `gnark:",public"`
	Commitment_36_10 frontend.Variable `gnark:",public"`
	Commitment_36_11 frontend.Variable `gnark:",public"`
	Commitment_37_0 frontend.Variable `gnark:",public"`
	Commitment_37_1 frontend.Variable `gnark:",public"`
	Commitment_37_2 frontend.Variable `gnark:",public"`
	Commitment_37_3 frontend.Variable `gnark:",public"`
	Commitment_37_4 frontend.Variable `gnark:",public"`
	Commitment_37_5 frontend.Variable `gnark:",public"`
	Commitment_37_6 frontend.Variable `gnark:",public"`
	Commitment_37_7 frontend.Variable `gnark:",public"`
	Commitment_37_8 frontend.Variable `gnark:",public"`
	Commitment_37_9 frontend.Variable `gnark:",public"`
	Commitment_37_10 frontend.Variable `gnark:",public"`
	Commitment_37_11 frontend.Variable `gnark:",public"`
	Commitment_38_0 frontend.Variable `gnark:",public"`
	Commitment_38_1 frontend.Variable `gnark:",public"`
	Commitment_38_2 frontend.Variable `gnark:",public"`
	Commitment_38_3 frontend.Variable `gnark:",public"`
	Commitment_38_4 frontend.Variable `gnark:",public"`
	Commitment_38_5 frontend.Variable `gnark:",public"`
	Commitment_38_6 frontend.Variable `gnark:",public"`
	Commitment_38_7 frontend.Variable `gnark:",public"`
	Commitment_38_8 frontend.Variable `gnark:",public"`
	Commitment_38_9 frontend.Variable `gnark:",public"`
	Commitment_38_10 frontend.Variable `gnark:",public"`
	Commitment_38_11 frontend.Variable `gnark:",public"`
	Commitment_39_0 frontend.Variable `gnark:",public"`
	Commitment_39_1 frontend.Variable `gnark:",public"`
	Commitment_39_2 frontend.Variable `gnark:",public"`
	Commitment_39_3 frontend.Variable `gnark:",public"`
	Commitment_39_4 frontend.Variable `gnark:",public"`
	Commitment_39_5 frontend.Variable `gnark:",public"`
	Commitment_39_6 frontend.Variable `gnark:",public"`
	Commitment_39_7 frontend.Variable `gnark:",public"`
	Commitment_39_8 frontend.Variable `gnark:",public"`
	Commitment_39_9 frontend.Variable `gnark:",public"`
	Commitment_39_10 frontend.Variable `gnark:",public"`
	Commitment_39_11 frontend.Variable `gnark:",public"`
	Commitment_40_0 frontend.Variable `gnark:",public"`
	Commitment_40_1 frontend.Variable `gnark:",public"`
	Commitment_40_2 frontend.Variable `gnark:",public"`
	Commitment_40_3 frontend.Variable `gnark:",public"`
	Commitment_40_4 frontend.Variable `gnark:",public"`
	Commitment_40_5 frontend.Variable `gnark:",public"`
	Commitment_40_6 frontend.Variable `gnark:",public"`
	Commitment_40_7 frontend.Variable `gnark:",public"`
	Commitment_40_8 frontend.Variable `gnark:",public"`
	Commitment_40_9 frontend.Variable `gnark:",public"`
	Commitment_40_10 frontend.Variable `gnark:",public"`
	Commitment_40_11 frontend.Variable `gnark:",public"`
	Claim_Virtual_PC_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_UnexpandedPC_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_NextPC_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_NextUnexpandedPC_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_NextIsVirtual_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_NextIsFirstInSequence_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_LeftLookupOperand_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_RightLookupOperand_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_LeftInstructionInput_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_RightInstructionInput_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_Product_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_ShouldJump_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_ShouldBranch_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_WritePCtoRD_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_WriteLookupOutputToRD_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_Imm_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_Rs1Value_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_Rs2Value_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_RdWriteValue_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_LookupOutput_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_RamAddress_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_RamReadValue_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_RamWriteValue_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_UnivariateSkip_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_AddOperands_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_SubtractOperands_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_MultiplyOperands_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_Load_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_Store_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_Jump_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_VirtualInstruction_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_Assert_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_DoNotUpdateUnexpandedPC_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_Advice_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_IsCompressed_SpartanOuter frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_0 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_1 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_2 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_3 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_4 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_5 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_6 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_7 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_8 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_9 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_10 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_11 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_12 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_13 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_14 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_15 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_16 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_17 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_18 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_19 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_20 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_21 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_22 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_23 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_24 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_25 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_26 frontend.Variable `gnark:",public"`
	Stage1_Uni_Skip_Coeff_27 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R0_0 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R0_1 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R0_2 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R1_0 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R1_1 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R1_2 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R2_0 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R2_1 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R2_2 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R3_0 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R3_1 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R3_2 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R4_0 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R4_1 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R4_2 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R5_0 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R5_1 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R5_2 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R6_0 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R6_1 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R6_2 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R7_0 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R7_1 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R7_2 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R8_0 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R8_1 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R8_2 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R9_0 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R9_1 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R9_2 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R10_0 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R10_1 frontend.Variable `gnark:",public"`
	Stage1_Sumcheck_R10_2 frontend.Variable `gnark:",public"`
}

func (circuit *JoltStages16Circuit) Define(api frontend.API) error {
	// Memoized subexpressions (CSE)
	cse_0 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, 1953263434, 0, 0), 0, poseidon.AppendU64Transform(api, 4096)), 1, poseidon.AppendU64Transform(api, 4096)), 2, poseidon.AppendU64Transform(api, 32768)), 3, 50), 4, 201625236193), 5, poseidon.AppendU64Transform(api, 0)), 6, poseidon.AppendU64Transform(api, 8192)), 7, poseidon.AppendU64Transform(api, 1024)), 8, circuit.Commitment_0_0), 0, circuit.Commitment_0_1), 0, circuit.Commitment_0_2), 0, circuit.Commitment_0_3), 0, circuit.Commitment_0_4), 0, circuit.Commitment_0_5), 0, circuit.Commitment_0_6), 0, circuit.Commitment_0_7), 0, circuit.Commitment_0_8), 0, circuit.Commitment_0_9), 0, circuit.Commitment_0_10), 0, circuit.Commitment_0_11), 9, circuit.Commitment_1_0), 0, circuit.Commitment_1_1), 0, circuit.Commitment_1_2), 0, circuit.Commitment_1_3), 0, circuit.Commitment_1_4), 0, circuit.Commitment_1_5), 0, circuit.Commitment_1_6), 0, circuit.Commitment_1_7), 0, circuit.Commitment_1_8), 0, circuit.Commitment_1_9), 0, circuit.Commitment_1_10), 0, circuit.Commitment_1_11), 10, circuit.Commitment_2_0), 0, circuit.Commitment_2_1), 0, circuit.Commitment_2_2), 0, circuit.Commitment_2_3), 0, circuit.Commitment_2_4), 0, circuit.Commitment_2_5), 0, circuit.Commitment_2_6), 0, circuit.Commitment_2_7), 0, circuit.Commitment_2_8), 0, circuit.Commitment_2_9), 0, circuit.Commitment_2_10), 0, circuit.Commitment_2_11), 11, circuit.Commitment_3_0), 0, circuit.Commitment_3_1), 0, circuit.Commitment_3_2), 0, circuit.Commitment_3_3), 0, circuit.Commitment_3_4), 0, circuit.Commitment_3_5), 0, circuit.Commitment_3_6), 0, circuit.Commitment_3_7), 0, circuit.Commitment_3_8), 0, circuit.Commitment_3_9), 0, circuit.Commitment_3_10), 0, circuit.Commitment_3_11), 12, circuit.Commitment_4_0), 0, circuit.Commitment_4_1), 0, circuit.Commitment_4_2), 0, circuit.Commitment_4_3), 0, circuit.Commitment_4_4), 0, circuit.Commitment_4_5), 0, circuit.Commitment_4_6), 0, circuit.Commitment_4_7), 0, circuit.Commitment_4_8), 0, circuit.Commitment_4_9), 0, circuit.Commitment_4_10), 0, circuit.Commitment_4_11), 13, circuit.Commitment_5_0), 0, circuit.Commitment_5_1), 0, circuit.Commitment_5_2), 0, circuit.Commitment_5_3), 0, circuit.Commitment_5_4), 0, circuit.Commitment_5_5), 0, circuit.Commitment_5_6), 0, circuit.Commitment_5_7), 0, circuit.Commitment_5_8), 0, circuit.Commitment_5_9), 0, circuit.Commitment_5_10), 0, circuit.Commitment_5_11), 14, circuit.Commitment_6_0), 0, circuit.Commitment_6_1), 0, circuit.Commitment_6_2), 0, circuit.Commitment_6_3), 0, circuit.Commitment_6_4), 0, circuit.Commitment_6_5), 0, circuit.Commitment_6_6), 0, circuit.Commitment_6_7), 0, circuit.Commitment_6_8), 0, circuit.Commitment_6_9), 0, circuit.Commitment_6_10), 0, circuit.Commitment_6_11), 15, circuit.Commitment_7_0), 0, circuit.Commitment_7_1), 0, circuit.Commitment_7_2), 0, circuit.Commitment_7_3), 0, circuit.Commitment_7_4), 0, circuit.Commitment_7_5), 0, circuit.Commitment_7_6), 0, circuit.Commitment_7_7), 0, circuit.Commitment_7_8), 0, circuit.Commitment_7_9), 0, circuit.Commitment_7_10), 0, circuit.Commitment_7_11), 16, circuit.Commitment_8_0), 0, circuit.Commitment_8_1), 0, circuit.Commitment_8_2), 0, circuit.Commitment_8_3), 0, circuit.Commitment_8_4), 0, circuit.Commitment_8_5), 0, circuit.Commitment_8_6), 0, circuit.Commitment_8_7), 0, circuit.Commitment_8_8), 0, circuit.Commitment_8_9), 0, circuit.Commitment_8_10), 0, circuit.Commitment_8_11), 17, circuit.Commitment_9_0), 0, circuit.Commitment_9_1), 0, circuit.Commitment_9_2), 0, circuit.Commitment_9_3), 0, circuit.Commitment_9_4), 0, circuit.Commitment_9_5), 0, circuit.Commitment_9_6), 0, circuit.Commitment_9_7), 0, circuit.Commitment_9_8), 0, circuit.Commitment_9_9), 0, circuit.Commitment_9_10), 0, circuit.Commitment_9_11), 18, circuit.Commitment_10_0), 0, circuit.Commitment_10_1), 0, circuit.Commitment_10_2), 0, circuit.Commitment_10_3), 0, circuit.Commitment_10_4), 0, circuit.Commitment_10_5), 0, circuit.Commitment_10_6), 0, circuit.Commitment_10_7), 0, circuit.Commitment_10_8), 0, circuit.Commitment_10_9), 0, circuit.Commitment_10_10), 0, circuit.Commitment_10_11), 19, circuit.Commitment_11_0), 0, circuit.Commitment_11_1), 0, circuit.Commitment_11_2), 0, circuit.Commitment_11_3), 0, circuit.Commitment_11_4), 0, circuit.Commitment_11_5), 0, circuit.Commitment_11_6), 0, circuit.Commitment_11_7), 0, circuit.Commitment_11_8), 0, circuit.Commitment_11_9), 0, circuit.Commitment_11_10), 0, circuit.Commitment_11_11), 20, circuit.Commitment_12_0), 0, circuit.Commitment_12_1), 0, circuit.Commitment_12_2), 0, circuit.Commitment_12_3), 0, circuit.Commitment_12_4), 0, circuit.Commitment_12_5), 0, circuit.Commitment_12_6), 0, circuit.Commitment_12_7), 0, circuit.Commitment_12_8), 0, circuit.Commitment_12_9), 0, circuit.Commitment_12_10), 0, circuit.Commitment_12_11), 21, circuit.Commitment_13_0), 0, circuit.Commitment_13_1), 0, circuit.Commitment_13_2), 0, circuit.Commitment_13_3), 0, circuit.Commitment_13_4), 0, circuit.Commitment_13_5), 0, circuit.Commitment_13_6), 0, circuit.Commitment_13_7), 0, circuit.Commitment_13_8), 0, circuit.Commitment_13_9), 0, circuit.Commitment_13_10), 0, circuit.Commitment_13_11), 22, circuit.Commitment_14_0), 0, circuit.Commitment_14_1), 0, circuit.Commitment_14_2), 0, circuit.Commitment_14_3), 0, circuit.Commitment_14_4), 0, circuit.Commitment_14_5), 0, circuit.Commitment_14_6), 0, circuit.Commitment_14_7), 0, circuit.Commitment_14_8), 0, circuit.Commitment_14_9), 0, circuit.Commitment_14_10), 0, circuit.Commitment_14_11), 23, circuit.Commitment_15_0), 0, circuit.Commitment_15_1), 0, circuit.Commitment_15_2), 0, circuit.Commitment_15_3), 0, circuit.Commitment_15_4), 0, circuit.Commitment_15_5), 0, circuit.Commitment_15_6), 0, circuit.Commitment_15_7), 0, circuit.Commitment_15_8), 0, circuit.Commitment_15_9), 0, circuit.Commitment_15_10), 0, circuit.Commitment_15_11), 24, circuit.Commitment_16_0), 0, circuit.Commitment_16_1), 0, circuit.Commitment_16_2), 0, circuit.Commitment_16_3), 0, circuit.Commitment_16_4), 0, circuit.Commitment_16_5), 0, circuit.Commitment_16_6), 0, circuit.Commitment_16_7), 0, circuit.Commitment_16_8), 0, circuit.Commitment_16_9), 0, circuit.Commitment_16_10), 0, circuit.Commitment_16_11), 25, circuit.Commitment_17_0), 0, circuit.Commitment_17_1), 0, circuit.Commitment_17_2), 0, circuit.Commitment_17_3), 0, circuit.Commitment_17_4), 0, circuit.Commitment_17_5), 0, circuit.Commitment_17_6), 0, circuit.Commitment_17_7), 0, circuit.Commitment_17_8), 0, circuit.Commitment_17_9), 0, circuit.Commitment_17_10), 0, circuit.Commitment_17_11), 26, circuit.Commitment_18_0), 0, circuit.Commitment_18_1), 0, circuit.Commitment_18_2), 0, circuit.Commitment_18_3), 0, circuit.Commitment_18_4), 0, circuit.Commitment_18_5), 0, circuit.Commitment_18_6), 0, circuit.Commitment_18_7), 0, circuit.Commitment_18_8), 0, circuit.Commitment_18_9), 0, circuit.Commitment_18_10), 0, circuit.Commitment_18_11), 27, circuit.Commitment_19_0), 0, circuit.Commitment_19_1), 0, circuit.Commitment_19_2), 0, circuit.Commitment_19_3), 0, circuit.Commitment_19_4), 0, circuit.Commitment_19_5), 0, circuit.Commitment_19_6), 0, circuit.Commitment_19_7), 0, circuit.Commitment_19_8), 0, circuit.Commitment_19_9), 0, circuit.Commitment_19_10), 0, circuit.Commitment_19_11), 28, circuit.Commitment_20_0), 0, circuit.Commitment_20_1), 0, circuit.Commitment_20_2), 0, circuit.Commitment_20_3), 0, circuit.Commitment_20_4), 0, circuit.Commitment_20_5), 0, circuit.Commitment_20_6), 0, circuit.Commitment_20_7), 0, circuit.Commitment_20_8), 0, circuit.Commitment_20_9), 0, circuit.Commitment_20_10), 0, circuit.Commitment_20_11), 29, circuit.Commitment_21_0), 0, circuit.Commitment_21_1), 0, circuit.Commitment_21_2), 0, circuit.Commitment_21_3), 0, circuit.Commitment_21_4), 0, circuit.Commitment_21_5), 0, circuit.Commitment_21_6), 0, circuit.Commitment_21_7), 0, circuit.Commitment_21_8), 0, circuit.Commitment_21_9), 0, circuit.Commitment_21_10), 0, circuit.Commitment_21_11), 30, circuit.Commitment_22_0), 0, circuit.Commitment_22_1), 0, circuit.Commitment_22_2), 0, circuit.Commitment_22_3), 0, circuit.Commitment_22_4), 0, circuit.Commitment_22_5), 0, circuit.Commitment_22_6), 0, circuit.Commitment_22_7), 0, circuit.Commitment_22_8), 0, circuit.Commitment_22_9), 0, circuit.Commitment_22_10), 0, circuit.Commitment_22_11), 31, circuit.Commitment_23_0), 0, circuit.Commitment_23_1), 0, circuit.Commitment_23_2), 0, circuit.Commitment_23_3), 0, circuit.Commitment_23_4), 0, circuit.Commitment_23_5), 0, circuit.Commitment_23_6), 0, circuit.Commitment_23_7), 0, circuit.Commitment_23_8), 0, circuit.Commitment_23_9), 0, circuit.Commitment_23_10), 0, circuit.Commitment_23_11), 32, circuit.Commitment_24_0), 0, circuit.Commitment_24_1), 0, circuit.Commitment_24_2), 0, circuit.Commitment_24_3), 0, circuit.Commitment_24_4), 0, circuit.Commitment_24_5), 0, circuit.Commitment_24_6), 0, circuit.Commitment_24_7), 0, circuit.Commitment_24_8), 0, circuit.Commitment_24_9), 0, circuit.Commitment_24_10), 0, circuit.Commitment_24_11), 33, circuit.Commitment_25_0), 0, circuit.Commitment_25_1), 0, circuit.Commitment_25_2), 0, circuit.Commitment_25_3), 0, circuit.Commitment_25_4), 0, circuit.Commitment_25_5), 0, circuit.Commitment_25_6), 0, circuit.Commitment_25_7), 0, circuit.Commitment_25_8), 0, circuit.Commitment_25_9), 0, circuit.Commitment_25_10), 0, circuit.Commitment_25_11), 34, circuit.Commitment_26_0), 0, circuit.Commitment_26_1), 0, circuit.Commitment_26_2), 0, circuit.Commitment_26_3), 0, circuit.Commitment_26_4), 0, circuit.Commitment_26_5), 0, circuit.Commitment_26_6), 0, circuit.Commitment_26_7), 0, circuit.Commitment_26_8), 0, circuit.Commitment_26_9), 0, circuit.Commitment_26_10), 0, circuit.Commitment_26_11), 35, circuit.Commitment_27_0), 0, circuit.Commitment_27_1), 0, circuit.Commitment_27_2), 0, circuit.Commitment_27_3), 0, circuit.Commitment_27_4), 0, circuit.Commitment_27_5), 0, circuit.Commitment_27_6), 0, circuit.Commitment_27_7), 0, circuit.Commitment_27_8), 0, circuit.Commitment_27_9), 0, circuit.Commitment_27_10), 0, circuit.Commitment_27_11), 36, circuit.Commitment_28_0), 0, circuit.Commitment_28_1), 0, circuit.Commitment_28_2), 0, circuit.Commitment_28_3), 0, circuit.Commitment_28_4), 0, circuit.Commitment_28_5), 0, circuit.Commitment_28_6), 0, circuit.Commitment_28_7), 0, circuit.Commitment_28_8), 0, circuit.Commitment_28_9), 0, circuit.Commitment_28_10), 0, circuit.Commitment_28_11), 37, circuit.Commitment_29_0), 0, circuit.Commitment_29_1), 0, circuit.Commitment_29_2), 0, circuit.Commitment_29_3), 0, circuit.Commitment_29_4), 0, circuit.Commitment_29_5), 0, circuit.Commitment_29_6), 0, circuit.Commitment_29_7), 0, circuit.Commitment_29_8), 0, circuit.Commitment_29_9), 0, circuit.Commitment_29_10), 0, circuit.Commitment_29_11), 38, circuit.Commitment_30_0), 0, circuit.Commitment_30_1), 0, circuit.Commitment_30_2), 0, circuit.Commitment_30_3), 0, circuit.Commitment_30_4), 0, circuit.Commitment_30_5), 0, circuit.Commitment_30_6), 0, circuit.Commitment_30_7), 0, circuit.Commitment_30_8), 0, circuit.Commitment_30_9), 0, circuit.Commitment_30_10), 0, circuit.Commitment_30_11), 39, circuit.Commitment_31_0), 0, circuit.Commitment_31_1), 0, circuit.Commitment_31_2), 0, circuit.Commitment_31_3), 0, circuit.Commitment_31_4), 0, circuit.Commitment_31_5), 0, circuit.Commitment_31_6), 0, circuit.Commitment_31_7), 0, circuit.Commitment_31_8), 0, circuit.Commitment_31_9), 0, circuit.Commitment_31_10), 0, circuit.Commitment_31_11), 40, circuit.Commitment_32_0), 0, circuit.Commitment_32_1), 0, circuit.Commitment_32_2), 0, circuit.Commitment_32_3), 0, circuit.Commitment_32_4), 0, circuit.Commitment_32_5), 0, circuit.Commitment_32_6), 0, circuit.Commitment_32_7), 0, circuit.Commitment_32_8), 0, circuit.Commitment_32_9), 0, circuit.Commitment_32_10), 0, circuit.Commitment_32_11), 41, circuit.Commitment_33_0), 0, circuit.Commitment_33_1), 0, circuit.Commitment_33_2), 0, circuit.Commitment_33_3), 0, circuit.Commitment_33_4), 0, circuit.Commitment_33_5), 0, circuit.Commitment_33_6), 0, circuit.Commitment_33_7), 0, circuit.Commitment_33_8), 0, circuit.Commitment_33_9), 0, circuit.Commitment_33_10), 0, circuit.Commitment_33_11), 42, circuit.Commitment_34_0), 0, circuit.Commitment_34_1), 0, circuit.Commitment_34_2), 0, circuit.Commitment_34_3), 0, circuit.Commitment_34_4), 0, circuit.Commitment_34_5), 0, circuit.Commitment_34_6), 0, circuit.Commitment_34_7), 0, circuit.Commitment_34_8), 0, circuit.Commitment_34_9), 0, circuit.Commitment_34_10), 0, circuit.Commitment_34_11), 43, circuit.Commitment_35_0), 0, circuit.Commitment_35_1), 0, circuit.Commitment_35_2), 0, circuit.Commitment_35_3), 0, circuit.Commitment_35_4), 0, circuit.Commitment_35_5), 0, circuit.Commitment_35_6), 0, circuit.Commitment_35_7), 0, circuit.Commitment_35_8), 0, circuit.Commitment_35_9), 0, circuit.Commitment_35_10), 0, circuit.Commitment_35_11), 44, circuit.Commitment_36_0), 0, circuit.Commitment_36_1), 0, circuit.Commitment_36_2), 0, circuit.Commitment_36_3), 0, circuit.Commitment_36_4), 0, circuit.Commitment_36_5), 0, circuit.Commitment_36_6), 0, circuit.Commitment_36_7), 0, circuit.Commitment_36_8), 0, circuit.Commitment_36_9), 0, circuit.Commitment_36_10), 0, circuit.Commitment_36_11), 45, circuit.Commitment_37_0), 0, circuit.Commitment_37_1), 0, circuit.Commitment_37_2), 0, circuit.Commitment_37_3), 0, circuit.Commitment_37_4), 0, circuit.Commitment_37_5), 0, circuit.Commitment_37_6), 0, circuit.Commitment_37_7), 0, circuit.Commitment_37_8), 0, circuit.Commitment_37_9), 0, circuit.Commitment_37_10), 0, circuit.Commitment_37_11), 46, circuit.Commitment_38_0), 0, circuit.Commitment_38_1), 0, circuit.Commitment_38_2), 0, circuit.Commitment_38_3), 0, circuit.Commitment_38_4), 0, circuit.Commitment_38_5), 0, circuit.Commitment_38_6), 0, circuit.Commitment_38_7), 0, circuit.Commitment_38_8), 0, circuit.Commitment_38_9), 0, circuit.Commitment_38_10), 0, circuit.Commitment_38_11), 47, circuit.Commitment_39_0), 0, circuit.Commitment_39_1), 0, circuit.Commitment_39_2), 0, circuit.Commitment_39_3), 0, circuit.Commitment_39_4), 0, circuit.Commitment_39_5), 0, circuit.Commitment_39_6), 0, circuit.Commitment_39_7), 0, circuit.Commitment_39_8), 0, circuit.Commitment_39_9), 0, circuit.Commitment_39_10), 0, circuit.Commitment_39_11), 48, circuit.Commitment_40_0), 0, circuit.Commitment_40_1), 0, circuit.Commitment_40_2), 0, circuit.Commitment_40_3), 0, circuit.Commitment_40_4), 0, circuit.Commitment_40_5), 0, circuit.Commitment_40_6), 0, circuit.Commitment_40_7), 0, circuit.Commitment_40_8), 0, circuit.Commitment_40_9), 0, circuit.Commitment_40_10), 0, circuit.Commitment_40_11), 49, 0)
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
	cse_12 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_11, 61, bigInt("693065686773592458709161276463075796193455407009757267193429")), 62, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_0)), 63, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_1)), 64, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_2)), 65, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_3)), 66, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_4)), 67, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_5)), 68, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_6)), 69, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_7)), 70, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_8)), 71, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_9)), 72, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_10)), 73, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_11)), 74, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_12)), 75, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_13)), 76, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_14)), 77, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_15)), 78, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_16)), 79, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_17)), 80, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_18)), 81, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_19)), 82, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_20)), 83, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_21)), 84, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_22)), 85, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_23)), 86, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_24)), 87, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_25)), 88, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_26)), 89, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_27)), 90, bigInt("9619401173246373414507010453289387209824226095986339413")), 91, 0)
	cse_13 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_12, 92, poseidon.ByteReverse(api, circuit.Claim_Virtual_UnivariateSkip_SpartanOuter)), 93, poseidon.ByteReverse(api, circuit.Claim_Virtual_UnivariateSkip_SpartanOuter)), 94, 0)
	cse_14 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_13, 95, bigInt("8747718800733414012499765325397")), 96, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R0_0)), 97, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R0_1)), 98, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R0_2)), 99, bigInt("121413912275379154240237141")), 100, 0)
	cse_15 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_14, 101, bigInt("8747718800733414012499765325397")), 102, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R1_0)), 103, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R1_1)), 104, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R1_2)), 105, bigInt("121413912275379154240237141")), 106, 0)
	cse_16 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_15, 107, bigInt("8747718800733414012499765325397")), 108, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R2_0)), 109, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R2_1)), 110, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R2_2)), 111, bigInt("121413912275379154240237141")), 112, 0)
	cse_17 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_16, 113, bigInt("8747718800733414012499765325397")), 114, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R3_0)), 115, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R3_1)), 116, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R3_2)), 117, bigInt("121413912275379154240237141")), 118, 0)
	cse_18 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_17, 119, bigInt("8747718800733414012499765325397")), 120, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R4_0)), 121, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R4_1)), 122, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R4_2)), 123, bigInt("121413912275379154240237141")), 124, 0)
	cse_19 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_18, 125, bigInt("8747718800733414012499765325397")), 126, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R5_0)), 127, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R5_1)), 128, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R5_2)), 129, bigInt("121413912275379154240237141")), 130, 0)
	cse_20 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_19, 131, bigInt("8747718800733414012499765325397")), 132, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R6_0)), 133, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R6_1)), 134, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R6_2)), 135, bigInt("121413912275379154240237141")), 136, 0)
	cse_21 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_20, 137, bigInt("8747718800733414012499765325397")), 138, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R7_0)), 139, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R7_1)), 140, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R7_2)), 141, bigInt("121413912275379154240237141")), 142, 0)
	cse_22 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_21, 143, bigInt("8747718800733414012499765325397")), 144, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R8_0)), 145, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R8_1)), 146, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R8_2)), 147, bigInt("121413912275379154240237141")), 148, 0)
	cse_23 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_22, 149, bigInt("8747718800733414012499765325397")), 150, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R9_0)), 151, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R9_1)), 152, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R9_2)), 153, bigInt("121413912275379154240237141")), 154, 0)
	cse_24 := poseidon.Truncate128Reverse(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_23, 155, bigInt("8747718800733414012499765325397")), 156, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R10_0)), 157, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R10_1)), 158, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R10_2)), 159, bigInt("121413912275379154240237141")), 160, 0))
	cse_25 := poseidon.Truncate128Reverse(api, cse_23)
	cse_26 := poseidon.Truncate128Reverse(api, cse_22)
	cse_27 := poseidon.Truncate128Reverse(api, cse_21)
	cse_28 := poseidon.Truncate128Reverse(api, cse_20)
	cse_29 := poseidon.Truncate128Reverse(api, cse_19)
	cse_30 := poseidon.Truncate128Reverse(api, cse_18)
	cse_31 := poseidon.Truncate128Reverse(api, cse_17)
	cse_32 := poseidon.Truncate128Reverse(api, cse_16)
	cse_33 := poseidon.Truncate128Reverse(api, cse_15)
	cse_34 := poseidon.Truncate128Reverse(api, cse_14)
	cse_35 := poseidon.Truncate128(api, cse_13)
	cse_36 := api.Mul(cse_34, cse_34)
	cse_37 := api.Mul(cse_33, cse_33)
	cse_38 := api.Mul(cse_32, cse_32)
	cse_39 := api.Mul(cse_31, cse_31)
	cse_40 := api.Mul(cse_30, cse_30)
	cse_41 := api.Mul(cse_29, cse_29)
	cse_42 := api.Mul(cse_28, cse_28)
	cse_43 := api.Mul(cse_27, cse_27)
	cse_44 := api.Mul(cse_26, cse_26)
	cse_45 := api.Mul(cse_25, cse_25)
	cse_46 := api.Mul(cse_24, cse_24)
	cse_47 := api.Mul(1, 362880)
	cse_48 := api.Mul(cse_47, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808455297"))
	cse_49 := api.Mul(cse_48, 10080)
	cse_50 := api.Mul(cse_49, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808491297"))
	cse_51 := api.Mul(cse_50, 2880)
	cse_52 := api.Mul(cse_51, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808492737"))
	cse_53 := api.Mul(cse_52, 4320)
	cse_54 := api.Mul(cse_53, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808485537"))
	cse_55 := api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808132737"))
	cse_56 := api.Mul(cse_55, 40320)
	cse_57 := api.Mul(cse_56, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808485537"))
	cse_58 := api.Mul(cse_57, 4320)
	cse_59 := api.Mul(cse_58, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808492737"))
	cse_60 := api.Mul(cse_59, 2880)
	cse_61 := api.Mul(cse_60, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808491297"))
	cse_62 := api.Mul(cse_61, 10080)
	cse_63 := api.Mul(cse_62, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808455297"))
	cse_64 := api.Inverse(api.Mul(cse_63, 362880))
	cse_65 := api.Mul(api.Mul(1, api.Mul(cse_54, 40320)), cse_64)
	cse_66 := api.Sub(poseidon.Truncate128Reverse(api, cse_11), bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495613"))
	cse_67 := api.Sub(cse_66, 1)
	cse_68 := api.Sub(cse_67, 1)
	cse_69 := api.Sub(cse_68, 1)
	cse_70 := api.Sub(cse_69, 1)
	cse_71 := api.Sub(cse_70, 1)
	cse_72 := api.Sub(cse_71, 1)
	cse_73 := api.Sub(cse_72, 1)
	cse_74 := api.Sub(cse_73, 1)
	cse_75 := api.Sub(cse_74, 1)
	cse_76 := api.Mul(1, cse_75)
	cse_77 := api.Mul(cse_76, cse_74)
	cse_78 := api.Mul(cse_77, cse_73)
	cse_79 := api.Mul(cse_78, cse_72)
	cse_80 := api.Mul(cse_79, cse_71)
	cse_81 := api.Mul(cse_80, cse_70)
	cse_82 := api.Mul(cse_81, cse_69)
	cse_83 := api.Mul(cse_82, cse_68)
	cse_84 := api.Mul(1, cse_66)
	cse_85 := api.Mul(cse_84, cse_67)
	cse_86 := api.Mul(cse_85, cse_68)
	cse_87 := api.Mul(cse_86, cse_69)
	cse_88 := api.Mul(cse_87, cse_70)
	cse_89 := api.Mul(cse_88, cse_71)
	cse_90 := api.Mul(cse_89, cse_72)
	cse_91 := api.Mul(cse_90, cse_73)
	cse_92 := api.Mul(cse_91, cse_74)
	cse_93 := api.Inverse(api.Mul(cse_92, cse_75))
	cse_94 := api.Mul(cse_65, api.Mul(api.Mul(1, api.Mul(cse_83, cse_67)), cse_93))
	cse_95 := poseidon.Truncate128Reverse(api, cse_12)
	cse_96 := api.Sub(cse_95, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495613"))
	cse_97 := api.Sub(cse_96, 1)
	cse_98 := api.Sub(cse_97, 1)
	cse_99 := api.Sub(cse_98, 1)
	cse_100 := api.Sub(cse_99, 1)
	cse_101 := api.Sub(cse_100, 1)
	cse_102 := api.Sub(cse_101, 1)
	cse_103 := api.Sub(cse_102, 1)
	cse_104 := api.Sub(cse_103, 1)
	cse_105 := api.Sub(cse_104, 1)
	cse_106 := api.Mul(1, cse_105)
	cse_107 := api.Mul(cse_106, cse_104)
	cse_108 := api.Mul(cse_107, cse_103)
	cse_109 := api.Mul(cse_108, cse_102)
	cse_110 := api.Mul(cse_109, cse_101)
	cse_111 := api.Mul(cse_110, cse_100)
	cse_112 := api.Mul(cse_111, cse_99)
	cse_113 := api.Mul(cse_112, cse_98)
	cse_114 := api.Mul(1, cse_96)
	cse_115 := api.Mul(cse_114, cse_97)
	cse_116 := api.Mul(cse_115, cse_98)
	cse_117 := api.Mul(cse_116, cse_99)
	cse_118 := api.Mul(cse_117, cse_100)
	cse_119 := api.Mul(cse_118, cse_101)
	cse_120 := api.Mul(cse_119, cse_102)
	cse_121 := api.Mul(cse_120, cse_103)
	cse_122 := api.Mul(cse_121, cse_104)
	cse_123 := api.Inverse(api.Mul(cse_122, cse_105))
	cse_124 := api.Mul(cse_65, api.Mul(api.Mul(1, api.Mul(cse_113, cse_97)), cse_123))
	cse_125 := api.Mul(api.Mul(cse_55, cse_54), cse_64)
	cse_126 := api.Mul(cse_125, api.Mul(api.Mul(cse_84, cse_83), cse_93))
	cse_127 := api.Mul(cse_125, api.Mul(api.Mul(cse_114, cse_113), cse_123))
	cse_128 := api.Mul(api.Mul(cse_56, cse_53), cse_64)
	cse_129 := api.Mul(cse_128, api.Mul(api.Mul(cse_85, cse_82), cse_93))
	cse_130 := api.Mul(cse_128, api.Mul(api.Mul(cse_115, cse_112), cse_123))
	cse_131 := api.Mul(api.Mul(cse_57, cse_52), cse_64)
	cse_132 := api.Mul(cse_131, api.Mul(api.Mul(cse_86, cse_81), cse_93))
	cse_133 := api.Mul(cse_131, api.Mul(api.Mul(cse_116, cse_111), cse_123))
	cse_134 := api.Mul(api.Mul(cse_58, cse_51), cse_64)
	cse_135 := api.Mul(cse_134, api.Mul(api.Mul(cse_87, cse_80), cse_93))
	cse_136 := api.Mul(cse_134, api.Mul(api.Mul(cse_117, cse_110), cse_123))
	cse_137 := api.Mul(api.Mul(cse_59, cse_50), cse_64)
	cse_138 := api.Mul(cse_137, api.Mul(api.Mul(cse_88, cse_79), cse_93))
	cse_139 := api.Mul(cse_137, api.Mul(api.Mul(cse_118, cse_109), cse_123))
	cse_140 := api.Mul(api.Mul(cse_60, cse_49), cse_64)
	cse_141 := api.Mul(cse_140, api.Mul(api.Mul(cse_89, cse_78), cse_93))
	cse_142 := api.Mul(cse_140, api.Mul(api.Mul(cse_119, cse_108), cse_123))
	cse_143 := api.Mul(api.Mul(cse_61, cse_48), cse_64)
	cse_144 := api.Mul(cse_143, api.Mul(api.Mul(cse_90, cse_77), cse_93))
	cse_145 := api.Mul(cse_143, api.Mul(api.Mul(cse_120, cse_107), cse_123))
	cse_146 := api.Mul(api.Mul(cse_62, cse_47), cse_64)
	cse_147 := api.Mul(cse_146, api.Mul(api.Mul(cse_91, cse_76), cse_93))
	cse_148 := api.Mul(cse_146, api.Mul(api.Mul(cse_121, cse_106), cse_123))
	cse_149 := api.Mul(api.Mul(cse_63, 1), cse_64)
	cse_150 := api.Mul(cse_149, api.Mul(api.Mul(cse_92, 1), cse_93))
	cse_151 := api.Mul(cse_149, api.Mul(api.Mul(cse_122, 1), cse_123))
	cse_152 := poseidon.Truncate128Reverse(api, cse_0)
	cse_153 := poseidon.Truncate128Reverse(api, cse_1)
	cse_154 := poseidon.Truncate128Reverse(api, cse_2)
	cse_155 := poseidon.Truncate128Reverse(api, cse_3)
	cse_156 := poseidon.Truncate128Reverse(api, cse_4)
	cse_157 := poseidon.Truncate128Reverse(api, cse_5)
	cse_158 := poseidon.Truncate128Reverse(api, cse_6)
	cse_159 := poseidon.Truncate128Reverse(api, cse_7)
	cse_160 := poseidon.Truncate128Reverse(api, cse_8)
	cse_161 := poseidon.Truncate128Reverse(api, cse_9)
	cse_162 := poseidon.Truncate128Reverse(api, cse_10)
	cse_163 := api.Mul(1, 362880)
	cse_164 := api.Mul(cse_163, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808455297"))
	cse_165 := api.Mul(cse_164, 10080)
	cse_166 := api.Mul(cse_165, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808491297"))
	cse_167 := api.Mul(cse_166, 2880)
	cse_168 := api.Mul(cse_167, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808492737"))
	cse_169 := api.Mul(cse_168, 4320)
	cse_170 := api.Mul(cse_169, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808485537"))
	cse_171 := api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808132737"))
	cse_172 := api.Mul(cse_171, 40320)
	cse_173 := api.Mul(cse_172, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808485537"))
	cse_174 := api.Mul(cse_173, 4320)
	cse_175 := api.Mul(cse_174, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808492737"))
	cse_176 := api.Mul(cse_175, 2880)
	cse_177 := api.Mul(cse_176, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808491297"))
	cse_178 := api.Mul(cse_177, 10080)
	cse_179 := api.Mul(cse_178, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808455297"))
	cse_180 := api.Inverse(api.Mul(cse_179, 362880))
	cse_181 := api.Sub(cse_95, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495613"))
	cse_182 := api.Sub(cse_181, 1)
	cse_183 := api.Sub(cse_182, 1)
	cse_184 := api.Sub(cse_183, 1)
	cse_185 := api.Sub(cse_184, 1)
	cse_186 := api.Sub(cse_185, 1)
	cse_187 := api.Sub(cse_186, 1)
	cse_188 := api.Sub(cse_187, 1)
	cse_189 := api.Sub(cse_188, 1)
	cse_190 := api.Sub(cse_189, 1)
	cse_191 := api.Mul(1, cse_190)
	cse_192 := api.Mul(cse_191, cse_189)
	cse_193 := api.Mul(cse_192, cse_188)
	cse_194 := api.Mul(cse_193, cse_187)
	cse_195 := api.Mul(cse_194, cse_186)
	cse_196 := api.Mul(cse_195, cse_185)
	cse_197 := api.Mul(cse_196, cse_184)
	cse_198 := api.Mul(cse_197, cse_183)
	cse_199 := api.Mul(1, cse_181)
	cse_200 := api.Mul(cse_199, cse_182)
	cse_201 := api.Mul(cse_200, cse_183)
	cse_202 := api.Mul(cse_201, cse_184)
	cse_203 := api.Mul(cse_202, cse_185)
	cse_204 := api.Mul(cse_203, cse_186)
	cse_205 := api.Mul(cse_204, cse_187)
	cse_206 := api.Mul(cse_205, cse_188)
	cse_207 := api.Mul(cse_206, cse_189)
	cse_208 := api.Inverse(api.Mul(cse_207, cse_190))
	cse_209 := api.Mul(api.Mul(api.Mul(1, api.Mul(cse_170, 40320)), cse_180), api.Mul(api.Mul(1, api.Mul(cse_198, cse_182)), cse_208))
	cse_210 := api.Mul(api.Mul(api.Mul(cse_171, cse_170), cse_180), api.Mul(api.Mul(cse_199, cse_198), cse_208))
	cse_211 := api.Mul(api.Mul(api.Mul(cse_172, cse_169), cse_180), api.Mul(api.Mul(cse_200, cse_197), cse_208))
	cse_212 := api.Mul(api.Mul(api.Mul(cse_173, cse_168), cse_180), api.Mul(api.Mul(cse_201, cse_196), cse_208))
	cse_213 := api.Mul(api.Mul(api.Mul(cse_174, cse_167), cse_180), api.Mul(api.Mul(cse_202, cse_195), cse_208))
	cse_214 := api.Mul(api.Mul(api.Mul(cse_175, cse_166), cse_180), api.Mul(api.Mul(cse_203, cse_194), cse_208))
	cse_215 := api.Mul(api.Mul(api.Mul(cse_176, cse_165), cse_180), api.Mul(api.Mul(cse_204, cse_193), cse_208))
	cse_216 := api.Mul(api.Mul(api.Mul(cse_177, cse_164), cse_180), api.Mul(api.Mul(cse_205, cse_192), cse_208))
	cse_217 := api.Mul(api.Mul(api.Mul(cse_178, cse_163), cse_180), api.Mul(api.Mul(cse_206, cse_191), cse_208))
	cse_218 := api.Mul(api.Mul(api.Mul(cse_179, 1), cse_180), api.Mul(api.Mul(cse_207, 1), cse_208))
	cse_219 := api.Inverse(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, cse_209), cse_210), cse_211), cse_212), cse_213), cse_214), cse_215), cse_216), cse_217), cse_218))
	cse_220 := api.Mul(cse_209, cse_219)
	cse_221 := api.Mul(cse_210, cse_219)
	cse_222 := api.Mul(cse_211, cse_219)
	cse_223 := api.Mul(cse_212, cse_219)
	cse_224 := api.Mul(cse_213, cse_219)
	cse_225 := api.Mul(cse_214, cse_219)
	cse_226 := api.Mul(cse_215, cse_219)
	cse_227 := api.Mul(cse_216, cse_219)
	cse_228 := api.Mul(cse_217, cse_219)
	cse_229 := api.Mul(cse_218, cse_219)
	cse_230 := api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, api.Mul(cse_220, api.Add(api.Add(api.Mul(circuit.Claim_Virtual_OpFlags_Load_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")), api.Mul(circuit.Claim_Virtual_OpFlags_Store_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(1, 1)))), api.Mul(cse_221, api.Mul(circuit.Claim_Virtual_OpFlags_Load_SpartanOuter, 1))), api.Mul(cse_222, api.Mul(circuit.Claim_Virtual_OpFlags_Load_SpartanOuter, 1))), api.Mul(cse_223, api.Mul(circuit.Claim_Virtual_OpFlags_Store_SpartanOuter, 1))), api.Mul(cse_224, api.Add(api.Add(api.Mul(circuit.Claim_Virtual_OpFlags_AddOperands_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_OpFlags_SubtractOperands_SpartanOuter, 1)), api.Mul(circuit.Claim_Virtual_OpFlags_MultiplyOperands_SpartanOuter, 1)))), api.Mul(cse_225, api.Add(api.Add(api.Add(api.Mul(circuit.Claim_Virtual_OpFlags_AddOperands_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")), api.Mul(circuit.Claim_Virtual_OpFlags_SubtractOperands_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_OpFlags_MultiplyOperands_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(1, 1)))), api.Mul(cse_226, api.Mul(circuit.Claim_Virtual_OpFlags_Assert_SpartanOuter, 1))), api.Mul(cse_227, api.Mul(circuit.Claim_Virtual_ShouldJump_SpartanOuter, 1))), api.Mul(cse_228, api.Mul(circuit.Claim_Virtual_OpFlags_VirtualInstruction_SpartanOuter, 1))), api.Mul(cse_229, api.Add(api.Mul(circuit.Claim_Virtual_NextIsVirtual_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_NextIsFirstInSequence_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")))))
	cse_231 := api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, api.Mul(cse_220, api.Mul(circuit.Claim_Virtual_RamAddress_SpartanOuter, 1))), api.Mul(cse_221, api.Add(api.Mul(circuit.Claim_Virtual_RamReadValue_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_RamWriteValue_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_222, api.Add(api.Mul(circuit.Claim_Virtual_RamReadValue_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_RdWriteValue_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_223, api.Add(api.Mul(circuit.Claim_Virtual_Rs2Value_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_RamWriteValue_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_224, api.Mul(circuit.Claim_Virtual_LeftLookupOperand_SpartanOuter, 1))), api.Mul(cse_225, api.Add(api.Mul(circuit.Claim_Virtual_LeftLookupOperand_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_LeftInstructionInput_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_226, api.Add(api.Mul(circuit.Claim_Virtual_LookupOutput_SpartanOuter, 1), api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_227, api.Add(api.Mul(circuit.Claim_Virtual_NextUnexpandedPC_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_LookupOutput_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_228, api.Add(api.Add(api.Mul(circuit.Claim_Virtual_NextPC_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_PC_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_229, api.Add(api.Mul(circuit.Claim_Virtual_OpFlags_DoNotUpdateUnexpandedPC_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")), api.Mul(1, 1))))

	// Verification assertions (each must equal 0)
	a0 := api.Sub(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, api.Mul(circuit.Stage1_Uni_Skip_Coeff_0, 10)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_1, 5)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_2, 85)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_3, 125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_4, 1333)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_5, 3125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_6, 25405)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_7, 78125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_8, 535333)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_9, 1953125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_10, 11982925)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_11, 48828125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_12, 278766133)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_13, 1220703125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_14, 6649985245)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_15, 30517578125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_16, 161264049733)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_17, 762939453125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_18, 3952911584365)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_19, 19073486328125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_20, 97573430562133)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_21, 476837158203125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_22, 2419432933612285)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_23, 11920928955078125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_24, 60168159621439333)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_25, 298023223876953125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_26, 1499128402505381005)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_27, 7450580596923828125)), 0)
	api.AssertIsEqual(a0, 0)
	a1 := api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R10_0, api.Mul(cse_24, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R9_0, api.Mul(cse_25, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R8_0, api.Mul(cse_26, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R7_0, api.Mul(cse_27, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R6_0, api.Mul(cse_28, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R5_0, api.Mul(cse_29, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R4_0, api.Mul(cse_30, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R3_0, api.Mul(cse_31, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R2_0, api.Mul(cse_32, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R1_0, api.Mul(cse_33, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R0_0, api.Mul(cse_34, api.Sub(api.Sub(api.Sub(api.Sub(api.Mul(api.Mul(circuit.Claim_Virtual_UnivariateSkip_SpartanOuter, 1), cse_35), circuit.Stage1_Sumcheck_R0_0), circuit.Stage1_Sumcheck_R0_0), circuit.Stage1_Sumcheck_R0_1), circuit.Stage1_Sumcheck_R0_2))), api.Mul(circuit.Stage1_Sumcheck_R0_1, cse_36)), api.Mul(circuit.Stage1_Sumcheck_R0_2, api.Mul(cse_36, cse_34))), circuit.Stage1_Sumcheck_R1_0), circuit.Stage1_Sumcheck_R1_0), circuit.Stage1_Sumcheck_R1_1), circuit.Stage1_Sumcheck_R1_2))), api.Mul(circuit.Stage1_Sumcheck_R1_1, cse_37)), api.Mul(circuit.Stage1_Sumcheck_R1_2, api.Mul(cse_37, cse_33))), circuit.Stage1_Sumcheck_R2_0), circuit.Stage1_Sumcheck_R2_0), circuit.Stage1_Sumcheck_R2_1), circuit.Stage1_Sumcheck_R2_2))), api.Mul(circuit.Stage1_Sumcheck_R2_1, cse_38)), api.Mul(circuit.Stage1_Sumcheck_R2_2, api.Mul(cse_38, cse_32))), circuit.Stage1_Sumcheck_R3_0), circuit.Stage1_Sumcheck_R3_0), circuit.Stage1_Sumcheck_R3_1), circuit.Stage1_Sumcheck_R3_2))), api.Mul(circuit.Stage1_Sumcheck_R3_1, cse_39)), api.Mul(circuit.Stage1_Sumcheck_R3_2, api.Mul(cse_39, cse_31))), circuit.Stage1_Sumcheck_R4_0), circuit.Stage1_Sumcheck_R4_0), circuit.Stage1_Sumcheck_R4_1), circuit.Stage1_Sumcheck_R4_2))), api.Mul(circuit.Stage1_Sumcheck_R4_1, cse_40)), api.Mul(circuit.Stage1_Sumcheck_R4_2, api.Mul(cse_40, cse_30))), circuit.Stage1_Sumcheck_R5_0), circuit.Stage1_Sumcheck_R5_0), circuit.Stage1_Sumcheck_R5_1), circuit.Stage1_Sumcheck_R5_2))), api.Mul(circuit.Stage1_Sumcheck_R5_1, cse_41)), api.Mul(circuit.Stage1_Sumcheck_R5_2, api.Mul(cse_41, cse_29))), circuit.Stage1_Sumcheck_R6_0), circuit.Stage1_Sumcheck_R6_0), circuit.Stage1_Sumcheck_R6_1), circuit.Stage1_Sumcheck_R6_2))), api.Mul(circuit.Stage1_Sumcheck_R6_1, cse_42)), api.Mul(circuit.Stage1_Sumcheck_R6_2, api.Mul(cse_42, cse_28))), circuit.Stage1_Sumcheck_R7_0), circuit.Stage1_Sumcheck_R7_0), circuit.Stage1_Sumcheck_R7_1), circuit.Stage1_Sumcheck_R7_2))), api.Mul(circuit.Stage1_Sumcheck_R7_1, cse_43)), api.Mul(circuit.Stage1_Sumcheck_R7_2, api.Mul(cse_43, cse_27))), circuit.Stage1_Sumcheck_R8_0), circuit.Stage1_Sumcheck_R8_0), circuit.Stage1_Sumcheck_R8_1), circuit.Stage1_Sumcheck_R8_2))), api.Mul(circuit.Stage1_Sumcheck_R8_1, cse_44)), api.Mul(circuit.Stage1_Sumcheck_R8_2, api.Mul(cse_44, cse_26))), circuit.Stage1_Sumcheck_R9_0), circuit.Stage1_Sumcheck_R9_0), circuit.Stage1_Sumcheck_R9_1), circuit.Stage1_Sumcheck_R9_2))), api.Mul(circuit.Stage1_Sumcheck_R9_1, cse_45)), api.Mul(circuit.Stage1_Sumcheck_R9_2, api.Mul(cse_45, cse_25))), circuit.Stage1_Sumcheck_R10_0), circuit.Stage1_Sumcheck_R10_0), circuit.Stage1_Sumcheck_R10_1), circuit.Stage1_Sumcheck_R10_2))), api.Mul(circuit.Stage1_Sumcheck_R10_1, cse_46)), api.Mul(circuit.Stage1_Sumcheck_R10_2, api.Mul(cse_46, cse_24))), api.Mul(api.Mul(api.Mul(api.Mul(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, api.Mul(cse_94, cse_124)), api.Mul(cse_126, cse_127)), api.Mul(cse_129, cse_130)), api.Mul(cse_132, cse_133)), api.Mul(cse_135, cse_136)), api.Mul(cse_138, cse_139)), api.Mul(cse_141, cse_142)), api.Mul(cse_144, cse_145)), api.Mul(cse_147, cse_148)), api.Mul(cse_150, cse_151)), api.Inverse(api.Mul(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, cse_94), cse_126), cse_129), cse_132), cse_135), cse_138), cse_141), cse_144), cse_147), cse_150), api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, cse_124), cse_127), cse_130), cse_133), cse_136), cse_139), cse_142), cse_145), cse_148), cse_151)))), api.Mul(api.Mul(api.Mul(api.Mul(1, api.Add(api.Mul(cse_152, cse_24), api.Mul(api.Sub(1, cse_152), api.Sub(1, cse_24)))), api.Mul(1, api.Add(api.Mul(cse_153, cse_25), api.Mul(api.Sub(1, cse_153), api.Sub(1, cse_25))))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_154, cse_26), api.Mul(api.Sub(1, cse_154), api.Sub(1, cse_26)))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_155, cse_27), api.Mul(api.Sub(1, cse_155), api.Sub(1, cse_27)))), api.Mul(1, api.Add(api.Mul(cse_156, cse_28), api.Mul(api.Sub(1, cse_156), api.Sub(1, cse_28))))))), api.Mul(api.Mul(api.Mul(1, api.Add(api.Mul(cse_157, cse_29), api.Mul(api.Sub(1, cse_157), api.Sub(1, cse_29)))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_158, cse_30), api.Mul(api.Sub(1, cse_158), api.Sub(1, cse_30)))), api.Mul(1, api.Add(api.Mul(cse_159, cse_31), api.Mul(api.Sub(1, cse_159), api.Sub(1, cse_31)))))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_160, cse_32), api.Mul(api.Sub(1, cse_160), api.Sub(1, cse_32)))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_161, cse_33), api.Mul(api.Sub(1, cse_161), api.Sub(1, cse_33)))), api.Mul(1, api.Add(api.Mul(cse_162, cse_34), api.Mul(api.Sub(1, cse_162), api.Sub(1, cse_34))))))))), api.Mul(api.Add(cse_230, api.Mul(cse_34, api.Sub(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, api.Mul(cse_220, api.Add(api.Mul(circuit.Claim_Virtual_OpFlags_Load_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_OpFlags_Store_SpartanOuter, 1)))), api.Mul(cse_221, api.Mul(circuit.Claim_Virtual_OpFlags_AddOperands_SpartanOuter, 1))), api.Mul(cse_222, api.Mul(circuit.Claim_Virtual_OpFlags_SubtractOperands_SpartanOuter, 1))), api.Mul(cse_223, api.Mul(circuit.Claim_Virtual_OpFlags_MultiplyOperands_SpartanOuter, 1))), api.Mul(cse_224, api.Add(api.Add(api.Add(api.Add(api.Mul(circuit.Claim_Virtual_OpFlags_AddOperands_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")), api.Mul(circuit.Claim_Virtual_OpFlags_SubtractOperands_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_OpFlags_MultiplyOperands_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_OpFlags_Advice_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(1, 1)))), api.Mul(cse_225, api.Mul(circuit.Claim_Virtual_WriteLookupOutputToRD_SpartanOuter, 1))), api.Mul(cse_226, api.Mul(circuit.Claim_Virtual_WritePCtoRD_SpartanOuter, 1))), api.Mul(cse_227, api.Mul(circuit.Claim_Virtual_ShouldBranch_SpartanOuter, 1))), api.Mul(cse_228, api.Add(api.Add(api.Mul(circuit.Claim_Virtual_ShouldBranch_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")), api.Mul(circuit.Claim_Virtual_OpFlags_Jump_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(1, 1)))), cse_230))), api.Add(cse_231, api.Mul(cse_34, api.Sub(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(0, api.Mul(cse_220, api.Add(api.Add(api.Mul(circuit.Claim_Virtual_RamAddress_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_Rs1Value_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_Imm_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_221, api.Add(api.Add(api.Mul(circuit.Claim_Virtual_RightLookupOperand_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_LeftInstructionInput_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_RightInstructionInput_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_222, api.Add(api.Add(api.Add(api.Mul(circuit.Claim_Virtual_RightLookupOperand_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_LeftInstructionInput_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_RightInstructionInput_SpartanOuter, 1)), api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343679757442502098944001"))))), api.Mul(cse_223, api.Add(api.Mul(circuit.Claim_Virtual_RightLookupOperand_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_Product_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_224, api.Add(api.Mul(circuit.Claim_Virtual_RightLookupOperand_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_RightInstructionInput_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_225, api.Add(api.Mul(circuit.Claim_Virtual_RdWriteValue_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_LookupOutput_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_226, api.Add(api.Add(api.Add(api.Mul(circuit.Claim_Virtual_RdWriteValue_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_UnexpandedPC_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_OpFlags_IsCompressed_SpartanOuter, 2)), api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495613"))))), api.Mul(cse_227, api.Add(api.Add(api.Mul(circuit.Claim_Virtual_NextUnexpandedPC_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_UnexpandedPC_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_Imm_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_228, api.Add(api.Add(api.Add(api.Add(api.Mul(circuit.Claim_Virtual_NextUnexpandedPC_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_UnexpandedPC_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_OpFlags_DoNotUpdateUnexpandedPC_SpartanOuter, 4)), api.Mul(circuit.Claim_Virtual_OpFlags_IsCompressed_SpartanOuter, 2)), api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495613"))))), cse_231))))), cse_35))
	api.AssertIsEqual(a1, 0)

	return nil
}
