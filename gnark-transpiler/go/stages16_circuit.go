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
	Claim_Committed_RamInc_RamReadWriteChecking frontend.Variable `gnark:",public"`
	Claim_Virtual_PC_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_UnexpandedPC_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_NextPC_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_NextUnexpandedPC_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_NextIsNoop_ProductVirtualization frontend.Variable `gnark:",public"`
	Claim_Virtual_NextIsVirtual_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_NextIsFirstInSequence_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_LeftLookupOperand_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_LeftLookupOperand_InstructionClaimReduction frontend.Variable `gnark:",public"`
	Claim_Virtual_RightLookupOperand_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_RightLookupOperand_InstructionClaimReduction frontend.Variable `gnark:",public"`
	Claim_Virtual_LeftInstructionInput_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_LeftInstructionInput_ProductVirtualization frontend.Variable `gnark:",public"`
	Claim_Virtual_RightInstructionInput_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_RightInstructionInput_ProductVirtualization frontend.Variable `gnark:",public"`
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
	Claim_Virtual_LookupOutput_ProductVirtualization frontend.Variable `gnark:",public"`
	Claim_Virtual_LookupOutput_InstructionClaimReduction frontend.Variable `gnark:",public"`
	Claim_Virtual_RamAddress_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_RamRa_RamReadWriteChecking frontend.Variable `gnark:",public"`
	Claim_Virtual_RamRa_RamRafEvaluation frontend.Variable `gnark:",public"`
	Claim_Virtual_RamReadValue_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_RamWriteValue_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_RamVal_RamReadWriteChecking frontend.Variable `gnark:",public"`
	Claim_Virtual_RamValFinal_RamOutputCheck frontend.Variable `gnark:",public"`
	Claim_Virtual_UnivariateSkip_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_UnivariateSkip_ProductVirtualization frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_AddOperands_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_SubtractOperands_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_MultiplyOperands_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_Load_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_Store_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_Jump_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_Jump_ProductVirtualization frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_WriteLookupOutputToRD_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_WriteLookupOutputToRD_ProductVirtualization frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_VirtualInstruction_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_Assert_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_DoNotUpdateUnexpandedPC_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_Advice_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_IsCompressed_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_OpFlags_IsFirstInSequence_SpartanOuter frontend.Variable `gnark:",public"`
	Claim_Virtual_InstructionFlags_Branch_ProductVirtualization frontend.Variable `gnark:",public"`
	Claim_Virtual_InstructionFlags_IsRdNotZero_ProductVirtualization frontend.Variable `gnark:",public"`
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
	Stage2_Uni_Skip_Coeff_0 frontend.Variable `gnark:",public"`
	Stage2_Uni_Skip_Coeff_1 frontend.Variable `gnark:",public"`
	Stage2_Uni_Skip_Coeff_2 frontend.Variable `gnark:",public"`
	Stage2_Uni_Skip_Coeff_3 frontend.Variable `gnark:",public"`
	Stage2_Uni_Skip_Coeff_4 frontend.Variable `gnark:",public"`
	Stage2_Uni_Skip_Coeff_5 frontend.Variable `gnark:",public"`
	Stage2_Uni_Skip_Coeff_6 frontend.Variable `gnark:",public"`
	Stage2_Uni_Skip_Coeff_7 frontend.Variable `gnark:",public"`
	Stage2_Uni_Skip_Coeff_8 frontend.Variable `gnark:",public"`
	Stage2_Uni_Skip_Coeff_9 frontend.Variable `gnark:",public"`
	Stage2_Uni_Skip_Coeff_10 frontend.Variable `gnark:",public"`
	Stage2_Uni_Skip_Coeff_11 frontend.Variable `gnark:",public"`
	Stage2_Uni_Skip_Coeff_12 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R0_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R0_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R0_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R1_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R1_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R1_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R2_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R2_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R2_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R3_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R3_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R3_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R4_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R4_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R4_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R5_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R5_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R5_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R6_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R6_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R6_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R7_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R7_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R7_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R8_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R8_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R8_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R9_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R9_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R9_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R10_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R10_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R10_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R11_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R11_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R11_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R12_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R12_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R12_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R13_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R13_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R13_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R14_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R14_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R14_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R15_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R15_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R15_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R16_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R16_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R16_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R17_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R17_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R17_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R18_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R18_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R18_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R19_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R19_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R19_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R20_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R20_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R20_2 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R21_0 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R21_1 frontend.Variable `gnark:",public"`
	Stage2_Sumcheck_R21_2 frontend.Variable `gnark:",public"`
}

func (circuit *JoltStages16Circuit) Define(api frontend.API) error {
	// Memoized subexpressions (CSE)
	// CSE bindings for assertion 1
	cse_1_0 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, 1953263434, 0, 0), 0, poseidon.AppendU64Transform(api, 4096)), 1, poseidon.AppendU64Transform(api, 4096)), 2, poseidon.AppendU64Transform(api, 32768)), 3, 10), 4, 55), 5, poseidon.AppendU64Transform(api, 0)), 6, poseidon.AppendU64Transform(api, 8192)), 7, poseidon.AppendU64Transform(api, 512)), 8, circuit.Commitment_0_0), 0, circuit.Commitment_0_1), 0, circuit.Commitment_0_2), 0, circuit.Commitment_0_3), 0, circuit.Commitment_0_4), 0, circuit.Commitment_0_5), 0, circuit.Commitment_0_6), 0, circuit.Commitment_0_7), 0, circuit.Commitment_0_8), 0, circuit.Commitment_0_9), 0, circuit.Commitment_0_10), 0, circuit.Commitment_0_11), 9, circuit.Commitment_1_0), 0, circuit.Commitment_1_1), 0, circuit.Commitment_1_2), 0, circuit.Commitment_1_3), 0, circuit.Commitment_1_4), 0, circuit.Commitment_1_5), 0, circuit.Commitment_1_6), 0, circuit.Commitment_1_7), 0, circuit.Commitment_1_8), 0, circuit.Commitment_1_9), 0, circuit.Commitment_1_10), 0, circuit.Commitment_1_11), 10, circuit.Commitment_2_0), 0, circuit.Commitment_2_1), 0, circuit.Commitment_2_2), 0, circuit.Commitment_2_3), 0, circuit.Commitment_2_4), 0, circuit.Commitment_2_5), 0, circuit.Commitment_2_6), 0, circuit.Commitment_2_7), 0, circuit.Commitment_2_8), 0, circuit.Commitment_2_9), 0, circuit.Commitment_2_10), 0, circuit.Commitment_2_11), 11, circuit.Commitment_3_0), 0, circuit.Commitment_3_1), 0, circuit.Commitment_3_2), 0, circuit.Commitment_3_3), 0, circuit.Commitment_3_4), 0, circuit.Commitment_3_5), 0, circuit.Commitment_3_6), 0, circuit.Commitment_3_7), 0, circuit.Commitment_3_8), 0, circuit.Commitment_3_9), 0, circuit.Commitment_3_10), 0, circuit.Commitment_3_11), 12, circuit.Commitment_4_0), 0, circuit.Commitment_4_1), 0, circuit.Commitment_4_2), 0, circuit.Commitment_4_3), 0, circuit.Commitment_4_4), 0, circuit.Commitment_4_5), 0, circuit.Commitment_4_6), 0, circuit.Commitment_4_7), 0, circuit.Commitment_4_8), 0, circuit.Commitment_4_9), 0, circuit.Commitment_4_10), 0, circuit.Commitment_4_11), 13, circuit.Commitment_5_0), 0, circuit.Commitment_5_1), 0, circuit.Commitment_5_2), 0, circuit.Commitment_5_3), 0, circuit.Commitment_5_4), 0, circuit.Commitment_5_5), 0, circuit.Commitment_5_6), 0, circuit.Commitment_5_7), 0, circuit.Commitment_5_8), 0, circuit.Commitment_5_9), 0, circuit.Commitment_5_10), 0, circuit.Commitment_5_11), 14, circuit.Commitment_6_0), 0, circuit.Commitment_6_1), 0, circuit.Commitment_6_2), 0, circuit.Commitment_6_3), 0, circuit.Commitment_6_4), 0, circuit.Commitment_6_5), 0, circuit.Commitment_6_6), 0, circuit.Commitment_6_7), 0, circuit.Commitment_6_8), 0, circuit.Commitment_6_9), 0, circuit.Commitment_6_10), 0, circuit.Commitment_6_11), 15, circuit.Commitment_7_0), 0, circuit.Commitment_7_1), 0, circuit.Commitment_7_2), 0, circuit.Commitment_7_3), 0, circuit.Commitment_7_4), 0, circuit.Commitment_7_5), 0, circuit.Commitment_7_6), 0, circuit.Commitment_7_7), 0, circuit.Commitment_7_8), 0, circuit.Commitment_7_9), 0, circuit.Commitment_7_10), 0, circuit.Commitment_7_11), 16, circuit.Commitment_8_0), 0, circuit.Commitment_8_1), 0, circuit.Commitment_8_2), 0, circuit.Commitment_8_3), 0, circuit.Commitment_8_4), 0, circuit.Commitment_8_5), 0, circuit.Commitment_8_6), 0, circuit.Commitment_8_7), 0, circuit.Commitment_8_8), 0, circuit.Commitment_8_9), 0, circuit.Commitment_8_10), 0, circuit.Commitment_8_11), 17, circuit.Commitment_9_0), 0, circuit.Commitment_9_1), 0, circuit.Commitment_9_2), 0, circuit.Commitment_9_3), 0, circuit.Commitment_9_4), 0, circuit.Commitment_9_5), 0, circuit.Commitment_9_6), 0, circuit.Commitment_9_7), 0, circuit.Commitment_9_8), 0, circuit.Commitment_9_9), 0, circuit.Commitment_9_10), 0, circuit.Commitment_9_11), 18, circuit.Commitment_10_0), 0, circuit.Commitment_10_1), 0, circuit.Commitment_10_2), 0, circuit.Commitment_10_3), 0, circuit.Commitment_10_4), 0, circuit.Commitment_10_5), 0, circuit.Commitment_10_6), 0, circuit.Commitment_10_7), 0, circuit.Commitment_10_8), 0, circuit.Commitment_10_9), 0, circuit.Commitment_10_10), 0, circuit.Commitment_10_11), 19, circuit.Commitment_11_0), 0, circuit.Commitment_11_1), 0, circuit.Commitment_11_2), 0, circuit.Commitment_11_3), 0, circuit.Commitment_11_4), 0, circuit.Commitment_11_5), 0, circuit.Commitment_11_6), 0, circuit.Commitment_11_7), 0, circuit.Commitment_11_8), 0, circuit.Commitment_11_9), 0, circuit.Commitment_11_10), 0, circuit.Commitment_11_11), 20, circuit.Commitment_12_0), 0, circuit.Commitment_12_1), 0, circuit.Commitment_12_2), 0, circuit.Commitment_12_3), 0, circuit.Commitment_12_4), 0, circuit.Commitment_12_5), 0, circuit.Commitment_12_6), 0, circuit.Commitment_12_7), 0, circuit.Commitment_12_8), 0, circuit.Commitment_12_9), 0, circuit.Commitment_12_10), 0, circuit.Commitment_12_11), 21, circuit.Commitment_13_0), 0, circuit.Commitment_13_1), 0, circuit.Commitment_13_2), 0, circuit.Commitment_13_3), 0, circuit.Commitment_13_4), 0, circuit.Commitment_13_5), 0, circuit.Commitment_13_6), 0, circuit.Commitment_13_7), 0, circuit.Commitment_13_8), 0, circuit.Commitment_13_9), 0, circuit.Commitment_13_10), 0, circuit.Commitment_13_11), 22, circuit.Commitment_14_0), 0, circuit.Commitment_14_1), 0, circuit.Commitment_14_2), 0, circuit.Commitment_14_3), 0, circuit.Commitment_14_4), 0, circuit.Commitment_14_5), 0, circuit.Commitment_14_6), 0, circuit.Commitment_14_7), 0, circuit.Commitment_14_8), 0, circuit.Commitment_14_9), 0, circuit.Commitment_14_10), 0, circuit.Commitment_14_11), 23, circuit.Commitment_15_0), 0, circuit.Commitment_15_1), 0, circuit.Commitment_15_2), 0, circuit.Commitment_15_3), 0, circuit.Commitment_15_4), 0, circuit.Commitment_15_5), 0, circuit.Commitment_15_6), 0, circuit.Commitment_15_7), 0, circuit.Commitment_15_8), 0, circuit.Commitment_15_9), 0, circuit.Commitment_15_10), 0, circuit.Commitment_15_11), 24, circuit.Commitment_16_0), 0, circuit.Commitment_16_1), 0, circuit.Commitment_16_2), 0, circuit.Commitment_16_3), 0, circuit.Commitment_16_4), 0, circuit.Commitment_16_5), 0, circuit.Commitment_16_6), 0, circuit.Commitment_16_7), 0, circuit.Commitment_16_8), 0, circuit.Commitment_16_9), 0, circuit.Commitment_16_10), 0, circuit.Commitment_16_11), 25, circuit.Commitment_17_0), 0, circuit.Commitment_17_1), 0, circuit.Commitment_17_2), 0, circuit.Commitment_17_3), 0, circuit.Commitment_17_4), 0, circuit.Commitment_17_5), 0, circuit.Commitment_17_6), 0, circuit.Commitment_17_7), 0, circuit.Commitment_17_8), 0, circuit.Commitment_17_9), 0, circuit.Commitment_17_10), 0, circuit.Commitment_17_11), 26, circuit.Commitment_18_0), 0, circuit.Commitment_18_1), 0, circuit.Commitment_18_2), 0, circuit.Commitment_18_3), 0, circuit.Commitment_18_4), 0, circuit.Commitment_18_5), 0, circuit.Commitment_18_6), 0, circuit.Commitment_18_7), 0, circuit.Commitment_18_8), 0, circuit.Commitment_18_9), 0, circuit.Commitment_18_10), 0, circuit.Commitment_18_11), 27, circuit.Commitment_19_0), 0, circuit.Commitment_19_1), 0, circuit.Commitment_19_2), 0, circuit.Commitment_19_3), 0, circuit.Commitment_19_4), 0, circuit.Commitment_19_5), 0, circuit.Commitment_19_6), 0, circuit.Commitment_19_7), 0, circuit.Commitment_19_8), 0, circuit.Commitment_19_9), 0, circuit.Commitment_19_10), 0, circuit.Commitment_19_11), 28, circuit.Commitment_20_0), 0, circuit.Commitment_20_1), 0, circuit.Commitment_20_2), 0, circuit.Commitment_20_3), 0, circuit.Commitment_20_4), 0, circuit.Commitment_20_5), 0, circuit.Commitment_20_6), 0, circuit.Commitment_20_7), 0, circuit.Commitment_20_8), 0, circuit.Commitment_20_9), 0, circuit.Commitment_20_10), 0, circuit.Commitment_20_11), 29, circuit.Commitment_21_0), 0, circuit.Commitment_21_1), 0, circuit.Commitment_21_2), 0, circuit.Commitment_21_3), 0, circuit.Commitment_21_4), 0, circuit.Commitment_21_5), 0, circuit.Commitment_21_6), 0, circuit.Commitment_21_7), 0, circuit.Commitment_21_8), 0, circuit.Commitment_21_9), 0, circuit.Commitment_21_10), 0, circuit.Commitment_21_11), 30, circuit.Commitment_22_0), 0, circuit.Commitment_22_1), 0, circuit.Commitment_22_2), 0, circuit.Commitment_22_3), 0, circuit.Commitment_22_4), 0, circuit.Commitment_22_5), 0, circuit.Commitment_22_6), 0, circuit.Commitment_22_7), 0, circuit.Commitment_22_8), 0, circuit.Commitment_22_9), 0, circuit.Commitment_22_10), 0, circuit.Commitment_22_11), 31, circuit.Commitment_23_0), 0, circuit.Commitment_23_1), 0, circuit.Commitment_23_2), 0, circuit.Commitment_23_3), 0, circuit.Commitment_23_4), 0, circuit.Commitment_23_5), 0, circuit.Commitment_23_6), 0, circuit.Commitment_23_7), 0, circuit.Commitment_23_8), 0, circuit.Commitment_23_9), 0, circuit.Commitment_23_10), 0, circuit.Commitment_23_11), 32, circuit.Commitment_24_0), 0, circuit.Commitment_24_1), 0, circuit.Commitment_24_2), 0, circuit.Commitment_24_3), 0, circuit.Commitment_24_4), 0, circuit.Commitment_24_5), 0, circuit.Commitment_24_6), 0, circuit.Commitment_24_7), 0, circuit.Commitment_24_8), 0, circuit.Commitment_24_9), 0, circuit.Commitment_24_10), 0, circuit.Commitment_24_11), 33, circuit.Commitment_25_0), 0, circuit.Commitment_25_1), 0, circuit.Commitment_25_2), 0, circuit.Commitment_25_3), 0, circuit.Commitment_25_4), 0, circuit.Commitment_25_5), 0, circuit.Commitment_25_6), 0, circuit.Commitment_25_7), 0, circuit.Commitment_25_8), 0, circuit.Commitment_25_9), 0, circuit.Commitment_25_10), 0, circuit.Commitment_25_11), 34, circuit.Commitment_26_0), 0, circuit.Commitment_26_1), 0, circuit.Commitment_26_2), 0, circuit.Commitment_26_3), 0, circuit.Commitment_26_4), 0, circuit.Commitment_26_5), 0, circuit.Commitment_26_6), 0, circuit.Commitment_26_7), 0, circuit.Commitment_26_8), 0, circuit.Commitment_26_9), 0, circuit.Commitment_26_10), 0, circuit.Commitment_26_11), 35, circuit.Commitment_27_0), 0, circuit.Commitment_27_1), 0, circuit.Commitment_27_2), 0, circuit.Commitment_27_3), 0, circuit.Commitment_27_4), 0, circuit.Commitment_27_5), 0, circuit.Commitment_27_6), 0, circuit.Commitment_27_7), 0, circuit.Commitment_27_8), 0, circuit.Commitment_27_9), 0, circuit.Commitment_27_10), 0, circuit.Commitment_27_11), 36, circuit.Commitment_28_0), 0, circuit.Commitment_28_1), 0, circuit.Commitment_28_2), 0, circuit.Commitment_28_3), 0, circuit.Commitment_28_4), 0, circuit.Commitment_28_5), 0, circuit.Commitment_28_6), 0, circuit.Commitment_28_7), 0, circuit.Commitment_28_8), 0, circuit.Commitment_28_9), 0, circuit.Commitment_28_10), 0, circuit.Commitment_28_11), 37, circuit.Commitment_29_0), 0, circuit.Commitment_29_1), 0, circuit.Commitment_29_2), 0, circuit.Commitment_29_3), 0, circuit.Commitment_29_4), 0, circuit.Commitment_29_5), 0, circuit.Commitment_29_6), 0, circuit.Commitment_29_7), 0, circuit.Commitment_29_8), 0, circuit.Commitment_29_9), 0, circuit.Commitment_29_10), 0, circuit.Commitment_29_11), 38, circuit.Commitment_30_0), 0, circuit.Commitment_30_1), 0, circuit.Commitment_30_2), 0, circuit.Commitment_30_3), 0, circuit.Commitment_30_4), 0, circuit.Commitment_30_5), 0, circuit.Commitment_30_6), 0, circuit.Commitment_30_7), 0, circuit.Commitment_30_8), 0, circuit.Commitment_30_9), 0, circuit.Commitment_30_10), 0, circuit.Commitment_30_11), 39, circuit.Commitment_31_0), 0, circuit.Commitment_31_1), 0, circuit.Commitment_31_2), 0, circuit.Commitment_31_3), 0, circuit.Commitment_31_4), 0, circuit.Commitment_31_5), 0, circuit.Commitment_31_6), 0, circuit.Commitment_31_7), 0, circuit.Commitment_31_8), 0, circuit.Commitment_31_9), 0, circuit.Commitment_31_10), 0, circuit.Commitment_31_11), 40, circuit.Commitment_32_0), 0, circuit.Commitment_32_1), 0, circuit.Commitment_32_2), 0, circuit.Commitment_32_3), 0, circuit.Commitment_32_4), 0, circuit.Commitment_32_5), 0, circuit.Commitment_32_6), 0, circuit.Commitment_32_7), 0, circuit.Commitment_32_8), 0, circuit.Commitment_32_9), 0, circuit.Commitment_32_10), 0, circuit.Commitment_32_11), 41, circuit.Commitment_33_0), 0, circuit.Commitment_33_1), 0, circuit.Commitment_33_2), 0, circuit.Commitment_33_3), 0, circuit.Commitment_33_4), 0, circuit.Commitment_33_5), 0, circuit.Commitment_33_6), 0, circuit.Commitment_33_7), 0, circuit.Commitment_33_8), 0, circuit.Commitment_33_9), 0, circuit.Commitment_33_10), 0, circuit.Commitment_33_11), 42, circuit.Commitment_34_0), 0, circuit.Commitment_34_1), 0, circuit.Commitment_34_2), 0, circuit.Commitment_34_3), 0, circuit.Commitment_34_4), 0, circuit.Commitment_34_5), 0, circuit.Commitment_34_6), 0, circuit.Commitment_34_7), 0, circuit.Commitment_34_8), 0, circuit.Commitment_34_9), 0, circuit.Commitment_34_10), 0, circuit.Commitment_34_11), 43, circuit.Commitment_35_0), 0, circuit.Commitment_35_1), 0, circuit.Commitment_35_2), 0, circuit.Commitment_35_3), 0, circuit.Commitment_35_4), 0, circuit.Commitment_35_5), 0, circuit.Commitment_35_6), 0, circuit.Commitment_35_7), 0, circuit.Commitment_35_8), 0, circuit.Commitment_35_9), 0, circuit.Commitment_35_10), 0, circuit.Commitment_35_11), 44, circuit.Commitment_36_0), 0, circuit.Commitment_36_1), 0, circuit.Commitment_36_2), 0, circuit.Commitment_36_3), 0, circuit.Commitment_36_4), 0, circuit.Commitment_36_5), 0, circuit.Commitment_36_6), 0, circuit.Commitment_36_7), 0, circuit.Commitment_36_8), 0, circuit.Commitment_36_9), 0, circuit.Commitment_36_10), 0, circuit.Commitment_36_11), 45, circuit.Commitment_37_0), 0, circuit.Commitment_37_1), 0, circuit.Commitment_37_2), 0, circuit.Commitment_37_3), 0, circuit.Commitment_37_4), 0, circuit.Commitment_37_5), 0, circuit.Commitment_37_6), 0, circuit.Commitment_37_7), 0, circuit.Commitment_37_8), 0, circuit.Commitment_37_9), 0, circuit.Commitment_37_10), 0, circuit.Commitment_37_11), 46, circuit.Commitment_38_0), 0, circuit.Commitment_38_1), 0, circuit.Commitment_38_2), 0, circuit.Commitment_38_3), 0, circuit.Commitment_38_4), 0, circuit.Commitment_38_5), 0, circuit.Commitment_38_6), 0, circuit.Commitment_38_7), 0, circuit.Commitment_38_8), 0, circuit.Commitment_38_9), 0, circuit.Commitment_38_10), 0, circuit.Commitment_38_11), 47, circuit.Commitment_39_0), 0, circuit.Commitment_39_1), 0, circuit.Commitment_39_2), 0, circuit.Commitment_39_3), 0, circuit.Commitment_39_4), 0, circuit.Commitment_39_5), 0, circuit.Commitment_39_6), 0, circuit.Commitment_39_7), 0, circuit.Commitment_39_8), 0, circuit.Commitment_39_9), 0, circuit.Commitment_39_10), 0, circuit.Commitment_39_11), 48, circuit.Commitment_40_0), 0, circuit.Commitment_40_1), 0, circuit.Commitment_40_2), 0, circuit.Commitment_40_3), 0, circuit.Commitment_40_4), 0, circuit.Commitment_40_5), 0, circuit.Commitment_40_6), 0, circuit.Commitment_40_7), 0, circuit.Commitment_40_8), 0, circuit.Commitment_40_9), 0, circuit.Commitment_40_10), 0, circuit.Commitment_40_11), 49, 0)
	cse_1_1 := poseidon.Hash(api, cse_1_0, 50, 0)
	cse_1_2 := poseidon.Hash(api, cse_1_1, 51, 0)
	cse_1_3 := poseidon.Hash(api, cse_1_2, 52, 0)
	cse_1_4 := poseidon.Hash(api, cse_1_3, 53, 0)
	cse_1_5 := poseidon.Hash(api, cse_1_4, 54, 0)
	cse_1_6 := poseidon.Hash(api, cse_1_5, 55, 0)
	cse_1_7 := poseidon.Hash(api, cse_1_6, 56, 0)
	cse_1_8 := poseidon.Hash(api, cse_1_7, 57, 0)
	cse_1_9 := poseidon.Hash(api, cse_1_8, 58, 0)
	cse_1_10 := poseidon.Hash(api, cse_1_9, 59, 0)
	cse_1_11 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_1_10, 60, bigInt("693065686773592458709161276463075796193455407009757267193429")), 61, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_0)), 62, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_1)), 63, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_2)), 64, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_3)), 65, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_4)), 66, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_5)), 67, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_6)), 68, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_7)), 69, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_8)), 70, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_9)), 71, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_10)), 72, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_11)), 73, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_12)), 74, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_13)), 75, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_14)), 76, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_15)), 77, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_16)), 78, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_17)), 79, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_18)), 80, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_19)), 81, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_20)), 82, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_21)), 83, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_22)), 84, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_23)), 85, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_24)), 86, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_25)), 87, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_26)), 88, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_27)), 89, bigInt("9619401173246373414507010453289387209824226095986339413")), 90, 0)
	cse_1_12 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_1_11, 91, poseidon.ByteReverse(api, circuit.Claim_Virtual_UnivariateSkip_SpartanOuter)), 92, poseidon.ByteReverse(api, circuit.Claim_Virtual_UnivariateSkip_SpartanOuter)), 93, 0)
	cse_1_13 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_1_12, 94, bigInt("8747718800733414012499765325397")), 95, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R0_0)), 96, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R0_1)), 97, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R0_2)), 98, bigInt("121413912275379154240237141")), 99, 0)
	cse_1_14 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_1_13, 100, bigInt("8747718800733414012499765325397")), 101, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R1_0)), 102, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R1_1)), 103, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R1_2)), 104, bigInt("121413912275379154240237141")), 105, 0)
	cse_1_15 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_1_14, 106, bigInt("8747718800733414012499765325397")), 107, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R2_0)), 108, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R2_1)), 109, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R2_2)), 110, bigInt("121413912275379154240237141")), 111, 0)
	cse_1_16 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_1_15, 112, bigInt("8747718800733414012499765325397")), 113, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R3_0)), 114, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R3_1)), 115, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R3_2)), 116, bigInt("121413912275379154240237141")), 117, 0)
	cse_1_17 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_1_16, 118, bigInt("8747718800733414012499765325397")), 119, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R4_0)), 120, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R4_1)), 121, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R4_2)), 122, bigInt("121413912275379154240237141")), 123, 0)
	cse_1_18 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_1_17, 124, bigInt("8747718800733414012499765325397")), 125, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R5_0)), 126, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R5_1)), 127, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R5_2)), 128, bigInt("121413912275379154240237141")), 129, 0)
	cse_1_19 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_1_18, 130, bigInt("8747718800733414012499765325397")), 131, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R6_0)), 132, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R6_1)), 133, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R6_2)), 134, bigInt("121413912275379154240237141")), 135, 0)
	cse_1_20 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_1_19, 136, bigInt("8747718800733414012499765325397")), 137, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R7_0)), 138, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R7_1)), 139, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R7_2)), 140, bigInt("121413912275379154240237141")), 141, 0)
	cse_1_21 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_1_20, 142, bigInt("8747718800733414012499765325397")), 143, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R8_0)), 144, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R8_1)), 145, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R8_2)), 146, bigInt("121413912275379154240237141")), 147, 0)
	cse_1_22 := poseidon.Truncate128Reverse(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_1_21, 148, bigInt("8747718800733414012499765325397")), 149, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R9_0)), 150, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R9_1)), 151, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R9_2)), 152, bigInt("121413912275379154240237141")), 153, 0))
	cse_1_23 := poseidon.Truncate128Reverse(api, cse_1_21)
	cse_1_24 := poseidon.Truncate128Reverse(api, cse_1_20)
	cse_1_25 := poseidon.Truncate128Reverse(api, cse_1_19)
	cse_1_26 := poseidon.Truncate128Reverse(api, cse_1_18)
	cse_1_27 := poseidon.Truncate128Reverse(api, cse_1_17)
	cse_1_28 := poseidon.Truncate128Reverse(api, cse_1_16)
	cse_1_29 := poseidon.Truncate128Reverse(api, cse_1_15)
	cse_1_30 := poseidon.Truncate128Reverse(api, cse_1_14)
	cse_1_31 := poseidon.Truncate128Reverse(api, cse_1_13)
	cse_1_32 := poseidon.Truncate128(api, cse_1_12)
	cse_1_33 := api.Mul(cse_1_31, cse_1_31)
	cse_1_34 := api.Mul(cse_1_30, cse_1_30)
	cse_1_35 := api.Mul(cse_1_29, cse_1_29)
	cse_1_36 := api.Mul(cse_1_28, cse_1_28)
	cse_1_37 := api.Mul(cse_1_27, cse_1_27)
	cse_1_38 := api.Mul(cse_1_26, cse_1_26)
	cse_1_39 := api.Mul(cse_1_25, cse_1_25)
	cse_1_40 := api.Mul(cse_1_24, cse_1_24)
	cse_1_41 := api.Mul(cse_1_23, cse_1_23)
	cse_1_42 := api.Mul(cse_1_22, cse_1_22)
	cse_1_43 := api.Mul(1, 362880)
	cse_1_44 := api.Mul(cse_1_43, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808455297"))
	cse_1_45 := api.Mul(cse_1_44, 10080)
	cse_1_46 := api.Mul(cse_1_45, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808491297"))
	cse_1_47 := api.Mul(cse_1_46, 2880)
	cse_1_48 := api.Mul(cse_1_47, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808492737"))
	cse_1_49 := api.Mul(cse_1_48, 4320)
	cse_1_50 := api.Mul(cse_1_49, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808485537"))
	cse_1_51 := api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808132737"))
	cse_1_52 := api.Mul(cse_1_51, 40320)
	cse_1_53 := api.Mul(cse_1_52, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808485537"))
	cse_1_54 := api.Mul(cse_1_53, 4320)
	cse_1_55 := api.Mul(cse_1_54, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808492737"))
	cse_1_56 := api.Mul(cse_1_55, 2880)
	cse_1_57 := api.Mul(cse_1_56, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808491297"))
	cse_1_58 := api.Mul(cse_1_57, 10080)
	cse_1_59 := api.Mul(cse_1_58, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808455297"))
	cse_1_60 := api.Inverse(api.Mul(cse_1_59, 362880))
	cse_1_61 := api.Mul(api.Mul(1, api.Mul(cse_1_50, 40320)), cse_1_60)
	cse_1_62 := api.Sub(poseidon.Truncate128Reverse(api, cse_1_10), bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495613"))
	cse_1_63 := api.Sub(cse_1_62, 1)
	cse_1_64 := api.Sub(cse_1_63, 1)
	cse_1_65 := api.Sub(cse_1_64, 1)
	cse_1_66 := api.Sub(cse_1_65, 1)
	cse_1_67 := api.Sub(cse_1_66, 1)
	cse_1_68 := api.Sub(cse_1_67, 1)
	cse_1_69 := api.Sub(cse_1_68, 1)
	cse_1_70 := api.Sub(cse_1_69, 1)
	cse_1_71 := api.Sub(cse_1_70, 1)
	cse_1_72 := api.Mul(1, cse_1_71)
	cse_1_73 := api.Mul(cse_1_72, cse_1_70)
	cse_1_74 := api.Mul(cse_1_73, cse_1_69)
	cse_1_75 := api.Mul(cse_1_74, cse_1_68)
	cse_1_76 := api.Mul(cse_1_75, cse_1_67)
	cse_1_77 := api.Mul(cse_1_76, cse_1_66)
	cse_1_78 := api.Mul(cse_1_77, cse_1_65)
	cse_1_79 := api.Mul(cse_1_78, cse_1_64)
	cse_1_80 := api.Mul(1, cse_1_62)
	cse_1_81 := api.Mul(cse_1_80, cse_1_63)
	cse_1_82 := api.Mul(cse_1_81, cse_1_64)
	cse_1_83 := api.Mul(cse_1_82, cse_1_65)
	cse_1_84 := api.Mul(cse_1_83, cse_1_66)
	cse_1_85 := api.Mul(cse_1_84, cse_1_67)
	cse_1_86 := api.Mul(cse_1_85, cse_1_68)
	cse_1_87 := api.Mul(cse_1_86, cse_1_69)
	cse_1_88 := api.Mul(cse_1_87, cse_1_70)
	cse_1_89 := api.Inverse(api.Mul(cse_1_88, cse_1_71))
	cse_1_90 := api.Mul(cse_1_61, api.Mul(api.Mul(1, api.Mul(cse_1_79, cse_1_63)), cse_1_89))
	cse_1_91 := poseidon.Truncate128Reverse(api, cse_1_11)
	cse_1_92 := api.Sub(cse_1_91, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495613"))
	cse_1_93 := api.Sub(cse_1_92, 1)
	cse_1_94 := api.Sub(cse_1_93, 1)
	cse_1_95 := api.Sub(cse_1_94, 1)
	cse_1_96 := api.Sub(cse_1_95, 1)
	cse_1_97 := api.Sub(cse_1_96, 1)
	cse_1_98 := api.Sub(cse_1_97, 1)
	cse_1_99 := api.Sub(cse_1_98, 1)
	cse_1_100 := api.Sub(cse_1_99, 1)
	cse_1_101 := api.Sub(cse_1_100, 1)
	cse_1_102 := api.Mul(1, cse_1_101)
	cse_1_103 := api.Mul(cse_1_102, cse_1_100)
	cse_1_104 := api.Mul(cse_1_103, cse_1_99)
	cse_1_105 := api.Mul(cse_1_104, cse_1_98)
	cse_1_106 := api.Mul(cse_1_105, cse_1_97)
	cse_1_107 := api.Mul(cse_1_106, cse_1_96)
	cse_1_108 := api.Mul(cse_1_107, cse_1_95)
	cse_1_109 := api.Mul(cse_1_108, cse_1_94)
	cse_1_110 := api.Mul(1, cse_1_92)
	cse_1_111 := api.Mul(cse_1_110, cse_1_93)
	cse_1_112 := api.Mul(cse_1_111, cse_1_94)
	cse_1_113 := api.Mul(cse_1_112, cse_1_95)
	cse_1_114 := api.Mul(cse_1_113, cse_1_96)
	cse_1_115 := api.Mul(cse_1_114, cse_1_97)
	cse_1_116 := api.Mul(cse_1_115, cse_1_98)
	cse_1_117 := api.Mul(cse_1_116, cse_1_99)
	cse_1_118 := api.Mul(cse_1_117, cse_1_100)
	cse_1_119 := api.Inverse(api.Mul(cse_1_118, cse_1_101))
	cse_1_120 := api.Mul(cse_1_61, api.Mul(api.Mul(1, api.Mul(cse_1_109, cse_1_93)), cse_1_119))
	cse_1_121 := api.Mul(api.Mul(cse_1_51, cse_1_50), cse_1_60)
	cse_1_122 := api.Mul(cse_1_121, api.Mul(api.Mul(cse_1_80, cse_1_79), cse_1_89))
	cse_1_123 := api.Mul(cse_1_121, api.Mul(api.Mul(cse_1_110, cse_1_109), cse_1_119))
	cse_1_124 := api.Mul(api.Mul(cse_1_52, cse_1_49), cse_1_60)
	cse_1_125 := api.Mul(cse_1_124, api.Mul(api.Mul(cse_1_81, cse_1_78), cse_1_89))
	cse_1_126 := api.Mul(cse_1_124, api.Mul(api.Mul(cse_1_111, cse_1_108), cse_1_119))
	cse_1_127 := api.Mul(api.Mul(cse_1_53, cse_1_48), cse_1_60)
	cse_1_128 := api.Mul(cse_1_127, api.Mul(api.Mul(cse_1_82, cse_1_77), cse_1_89))
	cse_1_129 := api.Mul(cse_1_127, api.Mul(api.Mul(cse_1_112, cse_1_107), cse_1_119))
	cse_1_130 := api.Mul(api.Mul(cse_1_54, cse_1_47), cse_1_60)
	cse_1_131 := api.Mul(cse_1_130, api.Mul(api.Mul(cse_1_83, cse_1_76), cse_1_89))
	cse_1_132 := api.Mul(cse_1_130, api.Mul(api.Mul(cse_1_113, cse_1_106), cse_1_119))
	cse_1_133 := api.Mul(api.Mul(cse_1_55, cse_1_46), cse_1_60)
	cse_1_134 := api.Mul(cse_1_133, api.Mul(api.Mul(cse_1_84, cse_1_75), cse_1_89))
	cse_1_135 := api.Mul(cse_1_133, api.Mul(api.Mul(cse_1_114, cse_1_105), cse_1_119))
	cse_1_136 := api.Mul(api.Mul(cse_1_56, cse_1_45), cse_1_60)
	cse_1_137 := api.Mul(cse_1_136, api.Mul(api.Mul(cse_1_85, cse_1_74), cse_1_89))
	cse_1_138 := api.Mul(cse_1_136, api.Mul(api.Mul(cse_1_115, cse_1_104), cse_1_119))
	cse_1_139 := api.Mul(api.Mul(cse_1_57, cse_1_44), cse_1_60)
	cse_1_140 := api.Mul(cse_1_139, api.Mul(api.Mul(cse_1_86, cse_1_73), cse_1_89))
	cse_1_141 := api.Mul(cse_1_139, api.Mul(api.Mul(cse_1_116, cse_1_103), cse_1_119))
	cse_1_142 := api.Mul(api.Mul(cse_1_58, cse_1_43), cse_1_60)
	cse_1_143 := api.Mul(cse_1_142, api.Mul(api.Mul(cse_1_87, cse_1_72), cse_1_89))
	cse_1_144 := api.Mul(cse_1_142, api.Mul(api.Mul(cse_1_117, cse_1_102), cse_1_119))
	cse_1_145 := api.Mul(api.Mul(cse_1_59, 1), cse_1_60)
	cse_1_146 := api.Mul(cse_1_145, api.Mul(api.Mul(cse_1_88, 1), cse_1_89))
	cse_1_147 := api.Mul(cse_1_145, api.Mul(api.Mul(cse_1_118, 1), cse_1_119))
	cse_1_148 := poseidon.Truncate128Reverse(api, cse_1_0)
	cse_1_149 := poseidon.Truncate128Reverse(api, cse_1_1)
	cse_1_150 := poseidon.Truncate128Reverse(api, cse_1_2)
	cse_1_151 := poseidon.Truncate128Reverse(api, cse_1_3)
	cse_1_152 := poseidon.Truncate128Reverse(api, cse_1_4)
	cse_1_153 := poseidon.Truncate128Reverse(api, cse_1_5)
	cse_1_154 := poseidon.Truncate128Reverse(api, cse_1_6)
	cse_1_155 := poseidon.Truncate128Reverse(api, cse_1_7)
	cse_1_156 := poseidon.Truncate128Reverse(api, cse_1_8)
	cse_1_157 := poseidon.Truncate128Reverse(api, cse_1_9)
	cse_1_158 := api.Mul(1, 362880)
	cse_1_159 := api.Mul(cse_1_158, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808455297"))
	cse_1_160 := api.Mul(cse_1_159, 10080)
	cse_1_161 := api.Mul(cse_1_160, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808491297"))
	cse_1_162 := api.Mul(cse_1_161, 2880)
	cse_1_163 := api.Mul(cse_1_162, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808492737"))
	cse_1_164 := api.Mul(cse_1_163, 4320)
	cse_1_165 := api.Mul(cse_1_164, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808485537"))
	cse_1_166 := api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808132737"))
	cse_1_167 := api.Mul(cse_1_166, 40320)
	cse_1_168 := api.Mul(cse_1_167, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808485537"))
	cse_1_169 := api.Mul(cse_1_168, 4320)
	cse_1_170 := api.Mul(cse_1_169, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808492737"))
	cse_1_171 := api.Mul(cse_1_170, 2880)
	cse_1_172 := api.Mul(cse_1_171, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808491297"))
	cse_1_173 := api.Mul(cse_1_172, 10080)
	cse_1_174 := api.Mul(cse_1_173, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808455297"))
	cse_1_175 := api.Inverse(api.Mul(cse_1_174, 362880))
	cse_1_176 := api.Sub(cse_1_91, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495613"))
	cse_1_177 := api.Sub(cse_1_176, 1)
	cse_1_178 := api.Sub(cse_1_177, 1)
	cse_1_179 := api.Sub(cse_1_178, 1)
	cse_1_180 := api.Sub(cse_1_179, 1)
	cse_1_181 := api.Sub(cse_1_180, 1)
	cse_1_182 := api.Sub(cse_1_181, 1)
	cse_1_183 := api.Sub(cse_1_182, 1)
	cse_1_184 := api.Sub(cse_1_183, 1)
	cse_1_185 := api.Sub(cse_1_184, 1)
	cse_1_186 := api.Mul(1, cse_1_185)
	cse_1_187 := api.Mul(cse_1_186, cse_1_184)
	cse_1_188 := api.Mul(cse_1_187, cse_1_183)
	cse_1_189 := api.Mul(cse_1_188, cse_1_182)
	cse_1_190 := api.Mul(cse_1_189, cse_1_181)
	cse_1_191 := api.Mul(cse_1_190, cse_1_180)
	cse_1_192 := api.Mul(cse_1_191, cse_1_179)
	cse_1_193 := api.Mul(cse_1_192, cse_1_178)
	cse_1_194 := api.Mul(1, cse_1_176)
	cse_1_195 := api.Mul(cse_1_194, cse_1_177)
	cse_1_196 := api.Mul(cse_1_195, cse_1_178)
	cse_1_197 := api.Mul(cse_1_196, cse_1_179)
	cse_1_198 := api.Mul(cse_1_197, cse_1_180)
	cse_1_199 := api.Mul(cse_1_198, cse_1_181)
	cse_1_200 := api.Mul(cse_1_199, cse_1_182)
	cse_1_201 := api.Mul(cse_1_200, cse_1_183)
	cse_1_202 := api.Mul(cse_1_201, cse_1_184)
	cse_1_203 := api.Inverse(api.Mul(cse_1_202, cse_1_185))
	cse_1_204 := api.Mul(api.Mul(api.Mul(1, api.Mul(cse_1_165, 40320)), cse_1_175), api.Mul(api.Mul(1, api.Mul(cse_1_193, cse_1_177)), cse_1_203))
	cse_1_205 := api.Mul(api.Mul(api.Mul(cse_1_166, cse_1_165), cse_1_175), api.Mul(api.Mul(cse_1_194, cse_1_193), cse_1_203))
	cse_1_206 := api.Mul(api.Mul(api.Mul(cse_1_167, cse_1_164), cse_1_175), api.Mul(api.Mul(cse_1_195, cse_1_192), cse_1_203))
	cse_1_207 := api.Mul(api.Mul(api.Mul(cse_1_168, cse_1_163), cse_1_175), api.Mul(api.Mul(cse_1_196, cse_1_191), cse_1_203))
	cse_1_208 := api.Mul(api.Mul(api.Mul(cse_1_169, cse_1_162), cse_1_175), api.Mul(api.Mul(cse_1_197, cse_1_190), cse_1_203))
	cse_1_209 := api.Mul(api.Mul(api.Mul(cse_1_170, cse_1_161), cse_1_175), api.Mul(api.Mul(cse_1_198, cse_1_189), cse_1_203))
	cse_1_210 := api.Mul(api.Mul(api.Mul(cse_1_171, cse_1_160), cse_1_175), api.Mul(api.Mul(cse_1_199, cse_1_188), cse_1_203))
	cse_1_211 := api.Mul(api.Mul(api.Mul(cse_1_172, cse_1_159), cse_1_175), api.Mul(api.Mul(cse_1_200, cse_1_187), cse_1_203))
	cse_1_212 := api.Mul(api.Mul(api.Mul(cse_1_173, cse_1_158), cse_1_175), api.Mul(api.Mul(cse_1_201, cse_1_186), cse_1_203))
	cse_1_213 := api.Mul(api.Mul(api.Mul(cse_1_174, 1), cse_1_175), api.Mul(api.Mul(cse_1_202, 1), cse_1_203))
	cse_1_214 := api.Inverse(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(cse_1_204, cse_1_205), cse_1_206), cse_1_207), cse_1_208), cse_1_209), cse_1_210), cse_1_211), cse_1_212), cse_1_213))
	cse_1_215 := api.Mul(cse_1_204, cse_1_214)
	cse_1_216 := api.Mul(cse_1_205, cse_1_214)
	cse_1_217 := api.Mul(cse_1_206, cse_1_214)
	cse_1_218 := api.Mul(cse_1_207, cse_1_214)
	cse_1_219 := api.Mul(cse_1_208, cse_1_214)
	cse_1_220 := api.Mul(cse_1_209, cse_1_214)
	cse_1_221 := api.Mul(cse_1_210, cse_1_214)
	cse_1_222 := api.Mul(cse_1_211, cse_1_214)
	cse_1_223 := api.Mul(cse_1_212, cse_1_214)
	cse_1_224 := api.Mul(cse_1_213, cse_1_214)
	cse_1_225 := api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Mul(cse_1_215, api.Add(api.Add(api.Mul(circuit.Claim_Virtual_OpFlags_Load_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")), api.Mul(circuit.Claim_Virtual_OpFlags_Store_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(1, 1))), api.Mul(cse_1_216, api.Mul(circuit.Claim_Virtual_OpFlags_Load_SpartanOuter, 1))), api.Mul(cse_1_217, api.Mul(circuit.Claim_Virtual_OpFlags_Load_SpartanOuter, 1))), api.Mul(cse_1_218, api.Mul(circuit.Claim_Virtual_OpFlags_Store_SpartanOuter, 1))), api.Mul(cse_1_219, api.Add(api.Add(api.Mul(circuit.Claim_Virtual_OpFlags_AddOperands_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_OpFlags_SubtractOperands_SpartanOuter, 1)), api.Mul(circuit.Claim_Virtual_OpFlags_MultiplyOperands_SpartanOuter, 1)))), api.Mul(cse_1_220, api.Add(api.Add(api.Add(api.Mul(circuit.Claim_Virtual_OpFlags_AddOperands_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")), api.Mul(circuit.Claim_Virtual_OpFlags_SubtractOperands_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_OpFlags_MultiplyOperands_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(1, 1)))), api.Mul(cse_1_221, api.Mul(circuit.Claim_Virtual_OpFlags_Assert_SpartanOuter, 1))), api.Mul(cse_1_222, api.Mul(circuit.Claim_Virtual_ShouldJump_SpartanOuter, 1))), api.Mul(cse_1_223, api.Mul(circuit.Claim_Virtual_OpFlags_VirtualInstruction_SpartanOuter, 1))), api.Mul(cse_1_224, api.Add(api.Mul(circuit.Claim_Virtual_NextIsVirtual_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_NextIsFirstInSequence_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")))))
	cse_1_226 := api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Mul(cse_1_215, api.Mul(circuit.Claim_Virtual_RamAddress_SpartanOuter, 1)), api.Mul(cse_1_216, api.Add(api.Mul(circuit.Claim_Virtual_RamReadValue_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_RamWriteValue_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_1_217, api.Add(api.Mul(circuit.Claim_Virtual_RamReadValue_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_RdWriteValue_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_1_218, api.Add(api.Mul(circuit.Claim_Virtual_Rs2Value_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_RamWriteValue_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_1_219, api.Mul(circuit.Claim_Virtual_LeftLookupOperand_SpartanOuter, 1))), api.Mul(cse_1_220, api.Add(api.Mul(circuit.Claim_Virtual_LeftLookupOperand_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_LeftInstructionInput_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_1_221, api.Add(api.Mul(circuit.Claim_Virtual_LookupOutput_SpartanOuter, 1), api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_1_222, api.Add(api.Mul(circuit.Claim_Virtual_NextUnexpandedPC_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_LookupOutput_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_1_223, api.Add(api.Add(api.Mul(circuit.Claim_Virtual_NextPC_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_PC_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_1_224, api.Add(api.Mul(circuit.Claim_Virtual_OpFlags_DoNotUpdateUnexpandedPC_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")), api.Mul(1, 1))))
	// CSE bindings for assertion 2
	cse_2_0 := api.Mul(1, 24)
	cse_2_1 := api.Mul(cse_2_0, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495611"))
	cse_2_2 := api.Mul(cse_2_1, 4)
	cse_2_3 := api.Mul(1, 24)
	cse_2_4 := api.Mul(cse_2_3, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495611"))
	cse_2_5 := api.Mul(cse_2_4, 4)
	cse_2_6 := api.Mul(cse_2_5, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495611"))
	cse_2_7 := api.Inverse(api.Mul(cse_2_6, 24))
	cse_2_8 := api.Sub(poseidon.Truncate128Reverse(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, 1953263434, 0, 0), 0, poseidon.AppendU64Transform(api, 4096)), 1, poseidon.AppendU64Transform(api, 4096)), 2, poseidon.AppendU64Transform(api, 32768)), 3, 10), 4, 55), 5, poseidon.AppendU64Transform(api, 0)), 6, poseidon.AppendU64Transform(api, 8192)), 7, poseidon.AppendU64Transform(api, 512)), 8, circuit.Commitment_0_0), 0, circuit.Commitment_0_1), 0, circuit.Commitment_0_2), 0, circuit.Commitment_0_3), 0, circuit.Commitment_0_4), 0, circuit.Commitment_0_5), 0, circuit.Commitment_0_6), 0, circuit.Commitment_0_7), 0, circuit.Commitment_0_8), 0, circuit.Commitment_0_9), 0, circuit.Commitment_0_10), 0, circuit.Commitment_0_11), 9, circuit.Commitment_1_0), 0, circuit.Commitment_1_1), 0, circuit.Commitment_1_2), 0, circuit.Commitment_1_3), 0, circuit.Commitment_1_4), 0, circuit.Commitment_1_5), 0, circuit.Commitment_1_6), 0, circuit.Commitment_1_7), 0, circuit.Commitment_1_8), 0, circuit.Commitment_1_9), 0, circuit.Commitment_1_10), 0, circuit.Commitment_1_11), 10, circuit.Commitment_2_0), 0, circuit.Commitment_2_1), 0, circuit.Commitment_2_2), 0, circuit.Commitment_2_3), 0, circuit.Commitment_2_4), 0, circuit.Commitment_2_5), 0, circuit.Commitment_2_6), 0, circuit.Commitment_2_7), 0, circuit.Commitment_2_8), 0, circuit.Commitment_2_9), 0, circuit.Commitment_2_10), 0, circuit.Commitment_2_11), 11, circuit.Commitment_3_0), 0, circuit.Commitment_3_1), 0, circuit.Commitment_3_2), 0, circuit.Commitment_3_3), 0, circuit.Commitment_3_4), 0, circuit.Commitment_3_5), 0, circuit.Commitment_3_6), 0, circuit.Commitment_3_7), 0, circuit.Commitment_3_8), 0, circuit.Commitment_3_9), 0, circuit.Commitment_3_10), 0, circuit.Commitment_3_11), 12, circuit.Commitment_4_0), 0, circuit.Commitment_4_1), 0, circuit.Commitment_4_2), 0, circuit.Commitment_4_3), 0, circuit.Commitment_4_4), 0, circuit.Commitment_4_5), 0, circuit.Commitment_4_6), 0, circuit.Commitment_4_7), 0, circuit.Commitment_4_8), 0, circuit.Commitment_4_9), 0, circuit.Commitment_4_10), 0, circuit.Commitment_4_11), 13, circuit.Commitment_5_0), 0, circuit.Commitment_5_1), 0, circuit.Commitment_5_2), 0, circuit.Commitment_5_3), 0, circuit.Commitment_5_4), 0, circuit.Commitment_5_5), 0, circuit.Commitment_5_6), 0, circuit.Commitment_5_7), 0, circuit.Commitment_5_8), 0, circuit.Commitment_5_9), 0, circuit.Commitment_5_10), 0, circuit.Commitment_5_11), 14, circuit.Commitment_6_0), 0, circuit.Commitment_6_1), 0, circuit.Commitment_6_2), 0, circuit.Commitment_6_3), 0, circuit.Commitment_6_4), 0, circuit.Commitment_6_5), 0, circuit.Commitment_6_6), 0, circuit.Commitment_6_7), 0, circuit.Commitment_6_8), 0, circuit.Commitment_6_9), 0, circuit.Commitment_6_10), 0, circuit.Commitment_6_11), 15, circuit.Commitment_7_0), 0, circuit.Commitment_7_1), 0, circuit.Commitment_7_2), 0, circuit.Commitment_7_3), 0, circuit.Commitment_7_4), 0, circuit.Commitment_7_5), 0, circuit.Commitment_7_6), 0, circuit.Commitment_7_7), 0, circuit.Commitment_7_8), 0, circuit.Commitment_7_9), 0, circuit.Commitment_7_10), 0, circuit.Commitment_7_11), 16, circuit.Commitment_8_0), 0, circuit.Commitment_8_1), 0, circuit.Commitment_8_2), 0, circuit.Commitment_8_3), 0, circuit.Commitment_8_4), 0, circuit.Commitment_8_5), 0, circuit.Commitment_8_6), 0, circuit.Commitment_8_7), 0, circuit.Commitment_8_8), 0, circuit.Commitment_8_9), 0, circuit.Commitment_8_10), 0, circuit.Commitment_8_11), 17, circuit.Commitment_9_0), 0, circuit.Commitment_9_1), 0, circuit.Commitment_9_2), 0, circuit.Commitment_9_3), 0, circuit.Commitment_9_4), 0, circuit.Commitment_9_5), 0, circuit.Commitment_9_6), 0, circuit.Commitment_9_7), 0, circuit.Commitment_9_8), 0, circuit.Commitment_9_9), 0, circuit.Commitment_9_10), 0, circuit.Commitment_9_11), 18, circuit.Commitment_10_0), 0, circuit.Commitment_10_1), 0, circuit.Commitment_10_2), 0, circuit.Commitment_10_3), 0, circuit.Commitment_10_4), 0, circuit.Commitment_10_5), 0, circuit.Commitment_10_6), 0, circuit.Commitment_10_7), 0, circuit.Commitment_10_8), 0, circuit.Commitment_10_9), 0, circuit.Commitment_10_10), 0, circuit.Commitment_10_11), 19, circuit.Commitment_11_0), 0, circuit.Commitment_11_1), 0, circuit.Commitment_11_2), 0, circuit.Commitment_11_3), 0, circuit.Commitment_11_4), 0, circuit.Commitment_11_5), 0, circuit.Commitment_11_6), 0, circuit.Commitment_11_7), 0, circuit.Commitment_11_8), 0, circuit.Commitment_11_9), 0, circuit.Commitment_11_10), 0, circuit.Commitment_11_11), 20, circuit.Commitment_12_0), 0, circuit.Commitment_12_1), 0, circuit.Commitment_12_2), 0, circuit.Commitment_12_3), 0, circuit.Commitment_12_4), 0, circuit.Commitment_12_5), 0, circuit.Commitment_12_6), 0, circuit.Commitment_12_7), 0, circuit.Commitment_12_8), 0, circuit.Commitment_12_9), 0, circuit.Commitment_12_10), 0, circuit.Commitment_12_11), 21, circuit.Commitment_13_0), 0, circuit.Commitment_13_1), 0, circuit.Commitment_13_2), 0, circuit.Commitment_13_3), 0, circuit.Commitment_13_4), 0, circuit.Commitment_13_5), 0, circuit.Commitment_13_6), 0, circuit.Commitment_13_7), 0, circuit.Commitment_13_8), 0, circuit.Commitment_13_9), 0, circuit.Commitment_13_10), 0, circuit.Commitment_13_11), 22, circuit.Commitment_14_0), 0, circuit.Commitment_14_1), 0, circuit.Commitment_14_2), 0, circuit.Commitment_14_3), 0, circuit.Commitment_14_4), 0, circuit.Commitment_14_5), 0, circuit.Commitment_14_6), 0, circuit.Commitment_14_7), 0, circuit.Commitment_14_8), 0, circuit.Commitment_14_9), 0, circuit.Commitment_14_10), 0, circuit.Commitment_14_11), 23, circuit.Commitment_15_0), 0, circuit.Commitment_15_1), 0, circuit.Commitment_15_2), 0, circuit.Commitment_15_3), 0, circuit.Commitment_15_4), 0, circuit.Commitment_15_5), 0, circuit.Commitment_15_6), 0, circuit.Commitment_15_7), 0, circuit.Commitment_15_8), 0, circuit.Commitment_15_9), 0, circuit.Commitment_15_10), 0, circuit.Commitment_15_11), 24, circuit.Commitment_16_0), 0, circuit.Commitment_16_1), 0, circuit.Commitment_16_2), 0, circuit.Commitment_16_3), 0, circuit.Commitment_16_4), 0, circuit.Commitment_16_5), 0, circuit.Commitment_16_6), 0, circuit.Commitment_16_7), 0, circuit.Commitment_16_8), 0, circuit.Commitment_16_9), 0, circuit.Commitment_16_10), 0, circuit.Commitment_16_11), 25, circuit.Commitment_17_0), 0, circuit.Commitment_17_1), 0, circuit.Commitment_17_2), 0, circuit.Commitment_17_3), 0, circuit.Commitment_17_4), 0, circuit.Commitment_17_5), 0, circuit.Commitment_17_6), 0, circuit.Commitment_17_7), 0, circuit.Commitment_17_8), 0, circuit.Commitment_17_9), 0, circuit.Commitment_17_10), 0, circuit.Commitment_17_11), 26, circuit.Commitment_18_0), 0, circuit.Commitment_18_1), 0, circuit.Commitment_18_2), 0, circuit.Commitment_18_3), 0, circuit.Commitment_18_4), 0, circuit.Commitment_18_5), 0, circuit.Commitment_18_6), 0, circuit.Commitment_18_7), 0, circuit.Commitment_18_8), 0, circuit.Commitment_18_9), 0, circuit.Commitment_18_10), 0, circuit.Commitment_18_11), 27, circuit.Commitment_19_0), 0, circuit.Commitment_19_1), 0, circuit.Commitment_19_2), 0, circuit.Commitment_19_3), 0, circuit.Commitment_19_4), 0, circuit.Commitment_19_5), 0, circuit.Commitment_19_6), 0, circuit.Commitment_19_7), 0, circuit.Commitment_19_8), 0, circuit.Commitment_19_9), 0, circuit.Commitment_19_10), 0, circuit.Commitment_19_11), 28, circuit.Commitment_20_0), 0, circuit.Commitment_20_1), 0, circuit.Commitment_20_2), 0, circuit.Commitment_20_3), 0, circuit.Commitment_20_4), 0, circuit.Commitment_20_5), 0, circuit.Commitment_20_6), 0, circuit.Commitment_20_7), 0, circuit.Commitment_20_8), 0, circuit.Commitment_20_9), 0, circuit.Commitment_20_10), 0, circuit.Commitment_20_11), 29, circuit.Commitment_21_0), 0, circuit.Commitment_21_1), 0, circuit.Commitment_21_2), 0, circuit.Commitment_21_3), 0, circuit.Commitment_21_4), 0, circuit.Commitment_21_5), 0, circuit.Commitment_21_6), 0, circuit.Commitment_21_7), 0, circuit.Commitment_21_8), 0, circuit.Commitment_21_9), 0, circuit.Commitment_21_10), 0, circuit.Commitment_21_11), 30, circuit.Commitment_22_0), 0, circuit.Commitment_22_1), 0, circuit.Commitment_22_2), 0, circuit.Commitment_22_3), 0, circuit.Commitment_22_4), 0, circuit.Commitment_22_5), 0, circuit.Commitment_22_6), 0, circuit.Commitment_22_7), 0, circuit.Commitment_22_8), 0, circuit.Commitment_22_9), 0, circuit.Commitment_22_10), 0, circuit.Commitment_22_11), 31, circuit.Commitment_23_0), 0, circuit.Commitment_23_1), 0, circuit.Commitment_23_2), 0, circuit.Commitment_23_3), 0, circuit.Commitment_23_4), 0, circuit.Commitment_23_5), 0, circuit.Commitment_23_6), 0, circuit.Commitment_23_7), 0, circuit.Commitment_23_8), 0, circuit.Commitment_23_9), 0, circuit.Commitment_23_10), 0, circuit.Commitment_23_11), 32, circuit.Commitment_24_0), 0, circuit.Commitment_24_1), 0, circuit.Commitment_24_2), 0, circuit.Commitment_24_3), 0, circuit.Commitment_24_4), 0, circuit.Commitment_24_5), 0, circuit.Commitment_24_6), 0, circuit.Commitment_24_7), 0, circuit.Commitment_24_8), 0, circuit.Commitment_24_9), 0, circuit.Commitment_24_10), 0, circuit.Commitment_24_11), 33, circuit.Commitment_25_0), 0, circuit.Commitment_25_1), 0, circuit.Commitment_25_2), 0, circuit.Commitment_25_3), 0, circuit.Commitment_25_4), 0, circuit.Commitment_25_5), 0, circuit.Commitment_25_6), 0, circuit.Commitment_25_7), 0, circuit.Commitment_25_8), 0, circuit.Commitment_25_9), 0, circuit.Commitment_25_10), 0, circuit.Commitment_25_11), 34, circuit.Commitment_26_0), 0, circuit.Commitment_26_1), 0, circuit.Commitment_26_2), 0, circuit.Commitment_26_3), 0, circuit.Commitment_26_4), 0, circuit.Commitment_26_5), 0, circuit.Commitment_26_6), 0, circuit.Commitment_26_7), 0, circuit.Commitment_26_8), 0, circuit.Commitment_26_9), 0, circuit.Commitment_26_10), 0, circuit.Commitment_26_11), 35, circuit.Commitment_27_0), 0, circuit.Commitment_27_1), 0, circuit.Commitment_27_2), 0, circuit.Commitment_27_3), 0, circuit.Commitment_27_4), 0, circuit.Commitment_27_5), 0, circuit.Commitment_27_6), 0, circuit.Commitment_27_7), 0, circuit.Commitment_27_8), 0, circuit.Commitment_27_9), 0, circuit.Commitment_27_10), 0, circuit.Commitment_27_11), 36, circuit.Commitment_28_0), 0, circuit.Commitment_28_1), 0, circuit.Commitment_28_2), 0, circuit.Commitment_28_3), 0, circuit.Commitment_28_4), 0, circuit.Commitment_28_5), 0, circuit.Commitment_28_6), 0, circuit.Commitment_28_7), 0, circuit.Commitment_28_8), 0, circuit.Commitment_28_9), 0, circuit.Commitment_28_10), 0, circuit.Commitment_28_11), 37, circuit.Commitment_29_0), 0, circuit.Commitment_29_1), 0, circuit.Commitment_29_2), 0, circuit.Commitment_29_3), 0, circuit.Commitment_29_4), 0, circuit.Commitment_29_5), 0, circuit.Commitment_29_6), 0, circuit.Commitment_29_7), 0, circuit.Commitment_29_8), 0, circuit.Commitment_29_9), 0, circuit.Commitment_29_10), 0, circuit.Commitment_29_11), 38, circuit.Commitment_30_0), 0, circuit.Commitment_30_1), 0, circuit.Commitment_30_2), 0, circuit.Commitment_30_3), 0, circuit.Commitment_30_4), 0, circuit.Commitment_30_5), 0, circuit.Commitment_30_6), 0, circuit.Commitment_30_7), 0, circuit.Commitment_30_8), 0, circuit.Commitment_30_9), 0, circuit.Commitment_30_10), 0, circuit.Commitment_30_11), 39, circuit.Commitment_31_0), 0, circuit.Commitment_31_1), 0, circuit.Commitment_31_2), 0, circuit.Commitment_31_3), 0, circuit.Commitment_31_4), 0, circuit.Commitment_31_5), 0, circuit.Commitment_31_6), 0, circuit.Commitment_31_7), 0, circuit.Commitment_31_8), 0, circuit.Commitment_31_9), 0, circuit.Commitment_31_10), 0, circuit.Commitment_31_11), 40, circuit.Commitment_32_0), 0, circuit.Commitment_32_1), 0, circuit.Commitment_32_2), 0, circuit.Commitment_32_3), 0, circuit.Commitment_32_4), 0, circuit.Commitment_32_5), 0, circuit.Commitment_32_6), 0, circuit.Commitment_32_7), 0, circuit.Commitment_32_8), 0, circuit.Commitment_32_9), 0, circuit.Commitment_32_10), 0, circuit.Commitment_32_11), 41, circuit.Commitment_33_0), 0, circuit.Commitment_33_1), 0, circuit.Commitment_33_2), 0, circuit.Commitment_33_3), 0, circuit.Commitment_33_4), 0, circuit.Commitment_33_5), 0, circuit.Commitment_33_6), 0, circuit.Commitment_33_7), 0, circuit.Commitment_33_8), 0, circuit.Commitment_33_9), 0, circuit.Commitment_33_10), 0, circuit.Commitment_33_11), 42, circuit.Commitment_34_0), 0, circuit.Commitment_34_1), 0, circuit.Commitment_34_2), 0, circuit.Commitment_34_3), 0, circuit.Commitment_34_4), 0, circuit.Commitment_34_5), 0, circuit.Commitment_34_6), 0, circuit.Commitment_34_7), 0, circuit.Commitment_34_8), 0, circuit.Commitment_34_9), 0, circuit.Commitment_34_10), 0, circuit.Commitment_34_11), 43, circuit.Commitment_35_0), 0, circuit.Commitment_35_1), 0, circuit.Commitment_35_2), 0, circuit.Commitment_35_3), 0, circuit.Commitment_35_4), 0, circuit.Commitment_35_5), 0, circuit.Commitment_35_6), 0, circuit.Commitment_35_7), 0, circuit.Commitment_35_8), 0, circuit.Commitment_35_9), 0, circuit.Commitment_35_10), 0, circuit.Commitment_35_11), 44, circuit.Commitment_36_0), 0, circuit.Commitment_36_1), 0, circuit.Commitment_36_2), 0, circuit.Commitment_36_3), 0, circuit.Commitment_36_4), 0, circuit.Commitment_36_5), 0, circuit.Commitment_36_6), 0, circuit.Commitment_36_7), 0, circuit.Commitment_36_8), 0, circuit.Commitment_36_9), 0, circuit.Commitment_36_10), 0, circuit.Commitment_36_11), 45, circuit.Commitment_37_0), 0, circuit.Commitment_37_1), 0, circuit.Commitment_37_2), 0, circuit.Commitment_37_3), 0, circuit.Commitment_37_4), 0, circuit.Commitment_37_5), 0, circuit.Commitment_37_6), 0, circuit.Commitment_37_7), 0, circuit.Commitment_37_8), 0, circuit.Commitment_37_9), 0, circuit.Commitment_37_10), 0, circuit.Commitment_37_11), 46, circuit.Commitment_38_0), 0, circuit.Commitment_38_1), 0, circuit.Commitment_38_2), 0, circuit.Commitment_38_3), 0, circuit.Commitment_38_4), 0, circuit.Commitment_38_5), 0, circuit.Commitment_38_6), 0, circuit.Commitment_38_7), 0, circuit.Commitment_38_8), 0, circuit.Commitment_38_9), 0, circuit.Commitment_38_10), 0, circuit.Commitment_38_11), 47, circuit.Commitment_39_0), 0, circuit.Commitment_39_1), 0, circuit.Commitment_39_2), 0, circuit.Commitment_39_3), 0, circuit.Commitment_39_4), 0, circuit.Commitment_39_5), 0, circuit.Commitment_39_6), 0, circuit.Commitment_39_7), 0, circuit.Commitment_39_8), 0, circuit.Commitment_39_9), 0, circuit.Commitment_39_10), 0, circuit.Commitment_39_11), 48, circuit.Commitment_40_0), 0, circuit.Commitment_40_1), 0, circuit.Commitment_40_2), 0, circuit.Commitment_40_3), 0, circuit.Commitment_40_4), 0, circuit.Commitment_40_5), 0, circuit.Commitment_40_6), 0, circuit.Commitment_40_7), 0, circuit.Commitment_40_8), 0, circuit.Commitment_40_9), 0, circuit.Commitment_40_10), 0, circuit.Commitment_40_11), 49, 0), 50, 0), 51, 0), 52, 0), 53, 0), 54, 0), 55, 0), 56, 0), 57, 0), 58, 0), 59, 0), 60, bigInt("693065686773592458709161276463075796193455407009757267193429")), 61, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_0)), 62, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_1)), 63, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_2)), 64, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_3)), 65, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_4)), 66, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_5)), 67, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_6)), 68, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_7)), 69, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_8)), 70, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_9)), 71, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_10)), 72, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_11)), 73, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_12)), 74, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_13)), 75, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_14)), 76, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_15)), 77, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_16)), 78, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_17)), 79, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_18)), 80, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_19)), 81, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_20)), 82, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_21)), 83, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_22)), 84, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_23)), 85, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_24)), 86, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_25)), 87, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_26)), 88, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_27)), 89, bigInt("9619401173246373414507010453289387209824226095986339413")), 90, 0), 91, poseidon.ByteReverse(api, circuit.Claim_Virtual_UnivariateSkip_SpartanOuter)), 92, poseidon.ByteReverse(api, circuit.Claim_Virtual_UnivariateSkip_SpartanOuter)), 93, 0), 94, bigInt("8747718800733414012499765325397")), 95, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R0_0)), 96, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R0_1)), 97, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R0_2)), 98, bigInt("121413912275379154240237141")), 99, 0), 100, bigInt("8747718800733414012499765325397")), 101, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R1_0)), 102, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R1_1)), 103, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R1_2)), 104, bigInt("121413912275379154240237141")), 105, 0), 106, bigInt("8747718800733414012499765325397")), 107, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R2_0)), 108, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R2_1)), 109, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R2_2)), 110, bigInt("121413912275379154240237141")), 111, 0), 112, bigInt("8747718800733414012499765325397")), 113, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R3_0)), 114, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R3_1)), 115, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R3_2)), 116, bigInt("121413912275379154240237141")), 117, 0), 118, bigInt("8747718800733414012499765325397")), 119, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R4_0)), 120, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R4_1)), 121, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R4_2)), 122, bigInt("121413912275379154240237141")), 123, 0), 124, bigInt("8747718800733414012499765325397")), 125, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R5_0)), 126, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R5_1)), 127, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R5_2)), 128, bigInt("121413912275379154240237141")), 129, 0), 130, bigInt("8747718800733414012499765325397")), 131, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R6_0)), 132, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R6_1)), 133, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R6_2)), 134, bigInt("121413912275379154240237141")), 135, 0), 136, bigInt("8747718800733414012499765325397")), 137, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R7_0)), 138, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R7_1)), 139, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R7_2)), 140, bigInt("121413912275379154240237141")), 141, 0), 142, bigInt("8747718800733414012499765325397")), 143, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R8_0)), 144, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R8_1)), 145, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R8_2)), 146, bigInt("121413912275379154240237141")), 147, 0), 148, bigInt("8747718800733414012499765325397")), 149, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R9_0)), 150, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R9_1)), 151, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R9_2)), 152, bigInt("121413912275379154240237141")), 153, 0), 154, poseidon.ByteReverse(api, circuit.Claim_Virtual_LeftInstructionInput_SpartanOuter)), 155, poseidon.ByteReverse(api, circuit.Claim_Virtual_RightInstructionInput_SpartanOuter)), 156, poseidon.ByteReverse(api, circuit.Claim_Virtual_Product_SpartanOuter)), 157, poseidon.ByteReverse(api, circuit.Claim_Virtual_WriteLookupOutputToRD_SpartanOuter)), 158, poseidon.ByteReverse(api, circuit.Claim_Virtual_WritePCtoRD_SpartanOuter)), 159, poseidon.ByteReverse(api, circuit.Claim_Virtual_ShouldBranch_SpartanOuter)), 160, poseidon.ByteReverse(api, circuit.Claim_Virtual_PC_SpartanOuter)), 161, poseidon.ByteReverse(api, circuit.Claim_Virtual_UnexpandedPC_SpartanOuter)), 162, poseidon.ByteReverse(api, circuit.Claim_Virtual_Imm_SpartanOuter)), 163, poseidon.ByteReverse(api, circuit.Claim_Virtual_RamAddress_SpartanOuter)), 164, poseidon.ByteReverse(api, circuit.Claim_Virtual_Rs1Value_SpartanOuter)), 165, poseidon.ByteReverse(api, circuit.Claim_Virtual_Rs2Value_SpartanOuter)), 166, poseidon.ByteReverse(api, circuit.Claim_Virtual_RdWriteValue_SpartanOuter)), 167, poseidon.ByteReverse(api, circuit.Claim_Virtual_RamReadValue_SpartanOuter)), 168, poseidon.ByteReverse(api, circuit.Claim_Virtual_RamWriteValue_SpartanOuter)), 169, poseidon.ByteReverse(api, circuit.Claim_Virtual_LeftLookupOperand_SpartanOuter)), 170, poseidon.ByteReverse(api, circuit.Claim_Virtual_RightLookupOperand_SpartanOuter)), 171, poseidon.ByteReverse(api, circuit.Claim_Virtual_NextUnexpandedPC_SpartanOuter)), 172, poseidon.ByteReverse(api, circuit.Claim_Virtual_NextPC_SpartanOuter)), 173, poseidon.ByteReverse(api, circuit.Claim_Virtual_NextIsVirtual_SpartanOuter)), 174, poseidon.ByteReverse(api, circuit.Claim_Virtual_NextIsFirstInSequence_SpartanOuter)), 175, poseidon.ByteReverse(api, circuit.Claim_Virtual_LookupOutput_SpartanOuter)), 176, poseidon.ByteReverse(api, circuit.Claim_Virtual_ShouldJump_SpartanOuter)), 177, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_AddOperands_SpartanOuter)), 178, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_SubtractOperands_SpartanOuter)), 179, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_MultiplyOperands_SpartanOuter)), 180, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_Load_SpartanOuter)), 181, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_Store_SpartanOuter)), 182, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_Jump_SpartanOuter)), 183, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_WriteLookupOutputToRD_SpartanOuter)), 184, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_VirtualInstruction_SpartanOuter)), 185, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_Assert_SpartanOuter)), 186, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_DoNotUpdateUnexpandedPC_SpartanOuter)), 187, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_Advice_SpartanOuter)), 188, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_IsCompressed_SpartanOuter)), 189, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_IsFirstInSequence_SpartanOuter)), 190, 0)), bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495615"))
	cse_2_9 := api.Sub(cse_2_8, 1)
	cse_2_10 := api.Sub(cse_2_9, 1)
	cse_2_11 := api.Sub(cse_2_10, 1)
	cse_2_12 := api.Sub(cse_2_11, 1)
	cse_2_13 := api.Mul(1, cse_2_12)
	cse_2_14 := api.Mul(cse_2_13, cse_2_11)
	cse_2_15 := api.Mul(cse_2_14, cse_2_10)
	cse_2_16 := api.Mul(1, cse_2_8)
	cse_2_17 := api.Mul(cse_2_16, cse_2_9)
	cse_2_18 := api.Mul(cse_2_17, cse_2_10)
	cse_2_19 := api.Mul(cse_2_18, cse_2_11)
	cse_2_20 := api.Inverse(api.Mul(cse_2_19, cse_2_12))
	cse_2_21 := api.Mul(api.Mul(api.Mul(1, api.Mul(cse_2_2, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495611"))), cse_2_7), api.Mul(api.Mul(1, api.Mul(cse_2_15, cse_2_9)), cse_2_20))
	cse_2_22 := api.Mul(api.Mul(api.Mul(cse_2_3, cse_2_2), cse_2_7), api.Mul(api.Mul(cse_2_16, cse_2_15), cse_2_20))
	cse_2_23 := api.Mul(api.Mul(api.Mul(cse_2_4, cse_2_1), cse_2_7), api.Mul(api.Mul(cse_2_17, cse_2_14), cse_2_20))
	cse_2_24 := api.Mul(api.Mul(api.Mul(cse_2_5, cse_2_0), cse_2_7), api.Mul(api.Mul(cse_2_18, cse_2_13), cse_2_20))
	cse_2_25 := api.Mul(api.Mul(api.Mul(cse_2_6, 1), cse_2_7), api.Mul(api.Mul(cse_2_19, 1), cse_2_20))
	cse_2_26 := api.Inverse(api.Add(api.Add(api.Add(api.Add(cse_2_21, cse_2_22), cse_2_23), cse_2_24), cse_2_25))
	// CSE bindings for assertion 3
	cse_3_0 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, 1953263434, 0, 0), 0, poseidon.AppendU64Transform(api, 4096)), 1, poseidon.AppendU64Transform(api, 4096)), 2, poseidon.AppendU64Transform(api, 32768)), 3, 10), 4, 55), 5, poseidon.AppendU64Transform(api, 0)), 6, poseidon.AppendU64Transform(api, 8192)), 7, poseidon.AppendU64Transform(api, 512)), 8, circuit.Commitment_0_0), 0, circuit.Commitment_0_1), 0, circuit.Commitment_0_2), 0, circuit.Commitment_0_3), 0, circuit.Commitment_0_4), 0, circuit.Commitment_0_5), 0, circuit.Commitment_0_6), 0, circuit.Commitment_0_7), 0, circuit.Commitment_0_8), 0, circuit.Commitment_0_9), 0, circuit.Commitment_0_10), 0, circuit.Commitment_0_11), 9, circuit.Commitment_1_0), 0, circuit.Commitment_1_1), 0, circuit.Commitment_1_2), 0, circuit.Commitment_1_3), 0, circuit.Commitment_1_4), 0, circuit.Commitment_1_5), 0, circuit.Commitment_1_6), 0, circuit.Commitment_1_7), 0, circuit.Commitment_1_8), 0, circuit.Commitment_1_9), 0, circuit.Commitment_1_10), 0, circuit.Commitment_1_11), 10, circuit.Commitment_2_0), 0, circuit.Commitment_2_1), 0, circuit.Commitment_2_2), 0, circuit.Commitment_2_3), 0, circuit.Commitment_2_4), 0, circuit.Commitment_2_5), 0, circuit.Commitment_2_6), 0, circuit.Commitment_2_7), 0, circuit.Commitment_2_8), 0, circuit.Commitment_2_9), 0, circuit.Commitment_2_10), 0, circuit.Commitment_2_11), 11, circuit.Commitment_3_0), 0, circuit.Commitment_3_1), 0, circuit.Commitment_3_2), 0, circuit.Commitment_3_3), 0, circuit.Commitment_3_4), 0, circuit.Commitment_3_5), 0, circuit.Commitment_3_6), 0, circuit.Commitment_3_7), 0, circuit.Commitment_3_8), 0, circuit.Commitment_3_9), 0, circuit.Commitment_3_10), 0, circuit.Commitment_3_11), 12, circuit.Commitment_4_0), 0, circuit.Commitment_4_1), 0, circuit.Commitment_4_2), 0, circuit.Commitment_4_3), 0, circuit.Commitment_4_4), 0, circuit.Commitment_4_5), 0, circuit.Commitment_4_6), 0, circuit.Commitment_4_7), 0, circuit.Commitment_4_8), 0, circuit.Commitment_4_9), 0, circuit.Commitment_4_10), 0, circuit.Commitment_4_11), 13, circuit.Commitment_5_0), 0, circuit.Commitment_5_1), 0, circuit.Commitment_5_2), 0, circuit.Commitment_5_3), 0, circuit.Commitment_5_4), 0, circuit.Commitment_5_5), 0, circuit.Commitment_5_6), 0, circuit.Commitment_5_7), 0, circuit.Commitment_5_8), 0, circuit.Commitment_5_9), 0, circuit.Commitment_5_10), 0, circuit.Commitment_5_11), 14, circuit.Commitment_6_0), 0, circuit.Commitment_6_1), 0, circuit.Commitment_6_2), 0, circuit.Commitment_6_3), 0, circuit.Commitment_6_4), 0, circuit.Commitment_6_5), 0, circuit.Commitment_6_6), 0, circuit.Commitment_6_7), 0, circuit.Commitment_6_8), 0, circuit.Commitment_6_9), 0, circuit.Commitment_6_10), 0, circuit.Commitment_6_11), 15, circuit.Commitment_7_0), 0, circuit.Commitment_7_1), 0, circuit.Commitment_7_2), 0, circuit.Commitment_7_3), 0, circuit.Commitment_7_4), 0, circuit.Commitment_7_5), 0, circuit.Commitment_7_6), 0, circuit.Commitment_7_7), 0, circuit.Commitment_7_8), 0, circuit.Commitment_7_9), 0, circuit.Commitment_7_10), 0, circuit.Commitment_7_11), 16, circuit.Commitment_8_0), 0, circuit.Commitment_8_1), 0, circuit.Commitment_8_2), 0, circuit.Commitment_8_3), 0, circuit.Commitment_8_4), 0, circuit.Commitment_8_5), 0, circuit.Commitment_8_6), 0, circuit.Commitment_8_7), 0, circuit.Commitment_8_8), 0, circuit.Commitment_8_9), 0, circuit.Commitment_8_10), 0, circuit.Commitment_8_11), 17, circuit.Commitment_9_0), 0, circuit.Commitment_9_1), 0, circuit.Commitment_9_2), 0, circuit.Commitment_9_3), 0, circuit.Commitment_9_4), 0, circuit.Commitment_9_5), 0, circuit.Commitment_9_6), 0, circuit.Commitment_9_7), 0, circuit.Commitment_9_8), 0, circuit.Commitment_9_9), 0, circuit.Commitment_9_10), 0, circuit.Commitment_9_11), 18, circuit.Commitment_10_0), 0, circuit.Commitment_10_1), 0, circuit.Commitment_10_2), 0, circuit.Commitment_10_3), 0, circuit.Commitment_10_4), 0, circuit.Commitment_10_5), 0, circuit.Commitment_10_6), 0, circuit.Commitment_10_7), 0, circuit.Commitment_10_8), 0, circuit.Commitment_10_9), 0, circuit.Commitment_10_10), 0, circuit.Commitment_10_11), 19, circuit.Commitment_11_0), 0, circuit.Commitment_11_1), 0, circuit.Commitment_11_2), 0, circuit.Commitment_11_3), 0, circuit.Commitment_11_4), 0, circuit.Commitment_11_5), 0, circuit.Commitment_11_6), 0, circuit.Commitment_11_7), 0, circuit.Commitment_11_8), 0, circuit.Commitment_11_9), 0, circuit.Commitment_11_10), 0, circuit.Commitment_11_11), 20, circuit.Commitment_12_0), 0, circuit.Commitment_12_1), 0, circuit.Commitment_12_2), 0, circuit.Commitment_12_3), 0, circuit.Commitment_12_4), 0, circuit.Commitment_12_5), 0, circuit.Commitment_12_6), 0, circuit.Commitment_12_7), 0, circuit.Commitment_12_8), 0, circuit.Commitment_12_9), 0, circuit.Commitment_12_10), 0, circuit.Commitment_12_11), 21, circuit.Commitment_13_0), 0, circuit.Commitment_13_1), 0, circuit.Commitment_13_2), 0, circuit.Commitment_13_3), 0, circuit.Commitment_13_4), 0, circuit.Commitment_13_5), 0, circuit.Commitment_13_6), 0, circuit.Commitment_13_7), 0, circuit.Commitment_13_8), 0, circuit.Commitment_13_9), 0, circuit.Commitment_13_10), 0, circuit.Commitment_13_11), 22, circuit.Commitment_14_0), 0, circuit.Commitment_14_1), 0, circuit.Commitment_14_2), 0, circuit.Commitment_14_3), 0, circuit.Commitment_14_4), 0, circuit.Commitment_14_5), 0, circuit.Commitment_14_6), 0, circuit.Commitment_14_7), 0, circuit.Commitment_14_8), 0, circuit.Commitment_14_9), 0, circuit.Commitment_14_10), 0, circuit.Commitment_14_11), 23, circuit.Commitment_15_0), 0, circuit.Commitment_15_1), 0, circuit.Commitment_15_2), 0, circuit.Commitment_15_3), 0, circuit.Commitment_15_4), 0, circuit.Commitment_15_5), 0, circuit.Commitment_15_6), 0, circuit.Commitment_15_7), 0, circuit.Commitment_15_8), 0, circuit.Commitment_15_9), 0, circuit.Commitment_15_10), 0, circuit.Commitment_15_11), 24, circuit.Commitment_16_0), 0, circuit.Commitment_16_1), 0, circuit.Commitment_16_2), 0, circuit.Commitment_16_3), 0, circuit.Commitment_16_4), 0, circuit.Commitment_16_5), 0, circuit.Commitment_16_6), 0, circuit.Commitment_16_7), 0, circuit.Commitment_16_8), 0, circuit.Commitment_16_9), 0, circuit.Commitment_16_10), 0, circuit.Commitment_16_11), 25, circuit.Commitment_17_0), 0, circuit.Commitment_17_1), 0, circuit.Commitment_17_2), 0, circuit.Commitment_17_3), 0, circuit.Commitment_17_4), 0, circuit.Commitment_17_5), 0, circuit.Commitment_17_6), 0, circuit.Commitment_17_7), 0, circuit.Commitment_17_8), 0, circuit.Commitment_17_9), 0, circuit.Commitment_17_10), 0, circuit.Commitment_17_11), 26, circuit.Commitment_18_0), 0, circuit.Commitment_18_1), 0, circuit.Commitment_18_2), 0, circuit.Commitment_18_3), 0, circuit.Commitment_18_4), 0, circuit.Commitment_18_5), 0, circuit.Commitment_18_6), 0, circuit.Commitment_18_7), 0, circuit.Commitment_18_8), 0, circuit.Commitment_18_9), 0, circuit.Commitment_18_10), 0, circuit.Commitment_18_11), 27, circuit.Commitment_19_0), 0, circuit.Commitment_19_1), 0, circuit.Commitment_19_2), 0, circuit.Commitment_19_3), 0, circuit.Commitment_19_4), 0, circuit.Commitment_19_5), 0, circuit.Commitment_19_6), 0, circuit.Commitment_19_7), 0, circuit.Commitment_19_8), 0, circuit.Commitment_19_9), 0, circuit.Commitment_19_10), 0, circuit.Commitment_19_11), 28, circuit.Commitment_20_0), 0, circuit.Commitment_20_1), 0, circuit.Commitment_20_2), 0, circuit.Commitment_20_3), 0, circuit.Commitment_20_4), 0, circuit.Commitment_20_5), 0, circuit.Commitment_20_6), 0, circuit.Commitment_20_7), 0, circuit.Commitment_20_8), 0, circuit.Commitment_20_9), 0, circuit.Commitment_20_10), 0, circuit.Commitment_20_11), 29, circuit.Commitment_21_0), 0, circuit.Commitment_21_1), 0, circuit.Commitment_21_2), 0, circuit.Commitment_21_3), 0, circuit.Commitment_21_4), 0, circuit.Commitment_21_5), 0, circuit.Commitment_21_6), 0, circuit.Commitment_21_7), 0, circuit.Commitment_21_8), 0, circuit.Commitment_21_9), 0, circuit.Commitment_21_10), 0, circuit.Commitment_21_11), 30, circuit.Commitment_22_0), 0, circuit.Commitment_22_1), 0, circuit.Commitment_22_2), 0, circuit.Commitment_22_3), 0, circuit.Commitment_22_4), 0, circuit.Commitment_22_5), 0, circuit.Commitment_22_6), 0, circuit.Commitment_22_7), 0, circuit.Commitment_22_8), 0, circuit.Commitment_22_9), 0, circuit.Commitment_22_10), 0, circuit.Commitment_22_11), 31, circuit.Commitment_23_0), 0, circuit.Commitment_23_1), 0, circuit.Commitment_23_2), 0, circuit.Commitment_23_3), 0, circuit.Commitment_23_4), 0, circuit.Commitment_23_5), 0, circuit.Commitment_23_6), 0, circuit.Commitment_23_7), 0, circuit.Commitment_23_8), 0, circuit.Commitment_23_9), 0, circuit.Commitment_23_10), 0, circuit.Commitment_23_11), 32, circuit.Commitment_24_0), 0, circuit.Commitment_24_1), 0, circuit.Commitment_24_2), 0, circuit.Commitment_24_3), 0, circuit.Commitment_24_4), 0, circuit.Commitment_24_5), 0, circuit.Commitment_24_6), 0, circuit.Commitment_24_7), 0, circuit.Commitment_24_8), 0, circuit.Commitment_24_9), 0, circuit.Commitment_24_10), 0, circuit.Commitment_24_11), 33, circuit.Commitment_25_0), 0, circuit.Commitment_25_1), 0, circuit.Commitment_25_2), 0, circuit.Commitment_25_3), 0, circuit.Commitment_25_4), 0, circuit.Commitment_25_5), 0, circuit.Commitment_25_6), 0, circuit.Commitment_25_7), 0, circuit.Commitment_25_8), 0, circuit.Commitment_25_9), 0, circuit.Commitment_25_10), 0, circuit.Commitment_25_11), 34, circuit.Commitment_26_0), 0, circuit.Commitment_26_1), 0, circuit.Commitment_26_2), 0, circuit.Commitment_26_3), 0, circuit.Commitment_26_4), 0, circuit.Commitment_26_5), 0, circuit.Commitment_26_6), 0, circuit.Commitment_26_7), 0, circuit.Commitment_26_8), 0, circuit.Commitment_26_9), 0, circuit.Commitment_26_10), 0, circuit.Commitment_26_11), 35, circuit.Commitment_27_0), 0, circuit.Commitment_27_1), 0, circuit.Commitment_27_2), 0, circuit.Commitment_27_3), 0, circuit.Commitment_27_4), 0, circuit.Commitment_27_5), 0, circuit.Commitment_27_6), 0, circuit.Commitment_27_7), 0, circuit.Commitment_27_8), 0, circuit.Commitment_27_9), 0, circuit.Commitment_27_10), 0, circuit.Commitment_27_11), 36, circuit.Commitment_28_0), 0, circuit.Commitment_28_1), 0, circuit.Commitment_28_2), 0, circuit.Commitment_28_3), 0, circuit.Commitment_28_4), 0, circuit.Commitment_28_5), 0, circuit.Commitment_28_6), 0, circuit.Commitment_28_7), 0, circuit.Commitment_28_8), 0, circuit.Commitment_28_9), 0, circuit.Commitment_28_10), 0, circuit.Commitment_28_11), 37, circuit.Commitment_29_0), 0, circuit.Commitment_29_1), 0, circuit.Commitment_29_2), 0, circuit.Commitment_29_3), 0, circuit.Commitment_29_4), 0, circuit.Commitment_29_5), 0, circuit.Commitment_29_6), 0, circuit.Commitment_29_7), 0, circuit.Commitment_29_8), 0, circuit.Commitment_29_9), 0, circuit.Commitment_29_10), 0, circuit.Commitment_29_11), 38, circuit.Commitment_30_0), 0, circuit.Commitment_30_1), 0, circuit.Commitment_30_2), 0, circuit.Commitment_30_3), 0, circuit.Commitment_30_4), 0, circuit.Commitment_30_5), 0, circuit.Commitment_30_6), 0, circuit.Commitment_30_7), 0, circuit.Commitment_30_8), 0, circuit.Commitment_30_9), 0, circuit.Commitment_30_10), 0, circuit.Commitment_30_11), 39, circuit.Commitment_31_0), 0, circuit.Commitment_31_1), 0, circuit.Commitment_31_2), 0, circuit.Commitment_31_3), 0, circuit.Commitment_31_4), 0, circuit.Commitment_31_5), 0, circuit.Commitment_31_6), 0, circuit.Commitment_31_7), 0, circuit.Commitment_31_8), 0, circuit.Commitment_31_9), 0, circuit.Commitment_31_10), 0, circuit.Commitment_31_11), 40, circuit.Commitment_32_0), 0, circuit.Commitment_32_1), 0, circuit.Commitment_32_2), 0, circuit.Commitment_32_3), 0, circuit.Commitment_32_4), 0, circuit.Commitment_32_5), 0, circuit.Commitment_32_6), 0, circuit.Commitment_32_7), 0, circuit.Commitment_32_8), 0, circuit.Commitment_32_9), 0, circuit.Commitment_32_10), 0, circuit.Commitment_32_11), 41, circuit.Commitment_33_0), 0, circuit.Commitment_33_1), 0, circuit.Commitment_33_2), 0, circuit.Commitment_33_3), 0, circuit.Commitment_33_4), 0, circuit.Commitment_33_5), 0, circuit.Commitment_33_6), 0, circuit.Commitment_33_7), 0, circuit.Commitment_33_8), 0, circuit.Commitment_33_9), 0, circuit.Commitment_33_10), 0, circuit.Commitment_33_11), 42, circuit.Commitment_34_0), 0, circuit.Commitment_34_1), 0, circuit.Commitment_34_2), 0, circuit.Commitment_34_3), 0, circuit.Commitment_34_4), 0, circuit.Commitment_34_5), 0, circuit.Commitment_34_6), 0, circuit.Commitment_34_7), 0, circuit.Commitment_34_8), 0, circuit.Commitment_34_9), 0, circuit.Commitment_34_10), 0, circuit.Commitment_34_11), 43, circuit.Commitment_35_0), 0, circuit.Commitment_35_1), 0, circuit.Commitment_35_2), 0, circuit.Commitment_35_3), 0, circuit.Commitment_35_4), 0, circuit.Commitment_35_5), 0, circuit.Commitment_35_6), 0, circuit.Commitment_35_7), 0, circuit.Commitment_35_8), 0, circuit.Commitment_35_9), 0, circuit.Commitment_35_10), 0, circuit.Commitment_35_11), 44, circuit.Commitment_36_0), 0, circuit.Commitment_36_1), 0, circuit.Commitment_36_2), 0, circuit.Commitment_36_3), 0, circuit.Commitment_36_4), 0, circuit.Commitment_36_5), 0, circuit.Commitment_36_6), 0, circuit.Commitment_36_7), 0, circuit.Commitment_36_8), 0, circuit.Commitment_36_9), 0, circuit.Commitment_36_10), 0, circuit.Commitment_36_11), 45, circuit.Commitment_37_0), 0, circuit.Commitment_37_1), 0, circuit.Commitment_37_2), 0, circuit.Commitment_37_3), 0, circuit.Commitment_37_4), 0, circuit.Commitment_37_5), 0, circuit.Commitment_37_6), 0, circuit.Commitment_37_7), 0, circuit.Commitment_37_8), 0, circuit.Commitment_37_9), 0, circuit.Commitment_37_10), 0, circuit.Commitment_37_11), 46, circuit.Commitment_38_0), 0, circuit.Commitment_38_1), 0, circuit.Commitment_38_2), 0, circuit.Commitment_38_3), 0, circuit.Commitment_38_4), 0, circuit.Commitment_38_5), 0, circuit.Commitment_38_6), 0, circuit.Commitment_38_7), 0, circuit.Commitment_38_8), 0, circuit.Commitment_38_9), 0, circuit.Commitment_38_10), 0, circuit.Commitment_38_11), 47, circuit.Commitment_39_0), 0, circuit.Commitment_39_1), 0, circuit.Commitment_39_2), 0, circuit.Commitment_39_3), 0, circuit.Commitment_39_4), 0, circuit.Commitment_39_5), 0, circuit.Commitment_39_6), 0, circuit.Commitment_39_7), 0, circuit.Commitment_39_8), 0, circuit.Commitment_39_9), 0, circuit.Commitment_39_10), 0, circuit.Commitment_39_11), 48, circuit.Commitment_40_0), 0, circuit.Commitment_40_1), 0, circuit.Commitment_40_2), 0, circuit.Commitment_40_3), 0, circuit.Commitment_40_4), 0, circuit.Commitment_40_5), 0, circuit.Commitment_40_6), 0, circuit.Commitment_40_7), 0, circuit.Commitment_40_8), 0, circuit.Commitment_40_9), 0, circuit.Commitment_40_10), 0, circuit.Commitment_40_11), 49, 0), 50, 0), 51, 0), 52, 0), 53, 0), 54, 0), 55, 0), 56, 0), 57, 0), 58, 0), 59, 0), 60, bigInt("693065686773592458709161276463075796193455407009757267193429")), 61, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_0)), 62, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_1)), 63, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_2)), 64, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_3)), 65, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_4)), 66, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_5)), 67, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_6)), 68, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_7)), 69, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_8)), 70, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_9)), 71, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_10)), 72, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_11)), 73, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_12)), 74, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_13)), 75, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_14)), 76, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_15)), 77, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_16)), 78, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_17)), 79, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_18)), 80, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_19)), 81, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_20)), 82, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_21)), 83, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_22)), 84, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_23)), 85, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_24)), 86, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_25)), 87, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_26)), 88, poseidon.ByteReverse(api, circuit.Stage1_Uni_Skip_Coeff_27)), 89, bigInt("9619401173246373414507010453289387209824226095986339413")), 90, 0), 91, poseidon.ByteReverse(api, circuit.Claim_Virtual_UnivariateSkip_SpartanOuter)), 92, poseidon.ByteReverse(api, circuit.Claim_Virtual_UnivariateSkip_SpartanOuter)), 93, 0), 94, bigInt("8747718800733414012499765325397")), 95, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R0_0)), 96, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R0_1)), 97, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R0_2)), 98, bigInt("121413912275379154240237141")), 99, 0), 100, bigInt("8747718800733414012499765325397")), 101, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R1_0)), 102, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R1_1)), 103, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R1_2)), 104, bigInt("121413912275379154240237141")), 105, 0)
	cse_3_1 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_0, 106, bigInt("8747718800733414012499765325397")), 107, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R2_0)), 108, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R2_1)), 109, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R2_2)), 110, bigInt("121413912275379154240237141")), 111, 0)
	cse_3_2 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_1, 112, bigInt("8747718800733414012499765325397")), 113, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R3_0)), 114, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R3_1)), 115, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R3_2)), 116, bigInt("121413912275379154240237141")), 117, 0)
	cse_3_3 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_2, 118, bigInt("8747718800733414012499765325397")), 119, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R4_0)), 120, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R4_1)), 121, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R4_2)), 122, bigInt("121413912275379154240237141")), 123, 0)
	cse_3_4 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_3, 124, bigInt("8747718800733414012499765325397")), 125, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R5_0)), 126, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R5_1)), 127, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R5_2)), 128, bigInt("121413912275379154240237141")), 129, 0)
	cse_3_5 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_4, 130, bigInt("8747718800733414012499765325397")), 131, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R6_0)), 132, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R6_1)), 133, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R6_2)), 134, bigInt("121413912275379154240237141")), 135, 0)
	cse_3_6 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_5, 136, bigInt("8747718800733414012499765325397")), 137, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R7_0)), 138, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R7_1)), 139, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R7_2)), 140, bigInt("121413912275379154240237141")), 141, 0)
	cse_3_7 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_6, 142, bigInt("8747718800733414012499765325397")), 143, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R8_0)), 144, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R8_1)), 145, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R8_2)), 146, bigInt("121413912275379154240237141")), 147, 0)
	cse_3_8 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_7, 148, bigInt("8747718800733414012499765325397")), 149, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R9_0)), 150, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R9_1)), 151, poseidon.ByteReverse(api, circuit.Stage1_Sumcheck_R9_2)), 152, bigInt("121413912275379154240237141")), 153, 0)
	cse_3_9 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_8, 154, poseidon.ByteReverse(api, circuit.Claim_Virtual_LeftInstructionInput_SpartanOuter)), 155, poseidon.ByteReverse(api, circuit.Claim_Virtual_RightInstructionInput_SpartanOuter)), 156, poseidon.ByteReverse(api, circuit.Claim_Virtual_Product_SpartanOuter)), 157, poseidon.ByteReverse(api, circuit.Claim_Virtual_WriteLookupOutputToRD_SpartanOuter)), 158, poseidon.ByteReverse(api, circuit.Claim_Virtual_WritePCtoRD_SpartanOuter)), 159, poseidon.ByteReverse(api, circuit.Claim_Virtual_ShouldBranch_SpartanOuter)), 160, poseidon.ByteReverse(api, circuit.Claim_Virtual_PC_SpartanOuter)), 161, poseidon.ByteReverse(api, circuit.Claim_Virtual_UnexpandedPC_SpartanOuter)), 162, poseidon.ByteReverse(api, circuit.Claim_Virtual_Imm_SpartanOuter)), 163, poseidon.ByteReverse(api, circuit.Claim_Virtual_RamAddress_SpartanOuter)), 164, poseidon.ByteReverse(api, circuit.Claim_Virtual_Rs1Value_SpartanOuter)), 165, poseidon.ByteReverse(api, circuit.Claim_Virtual_Rs2Value_SpartanOuter)), 166, poseidon.ByteReverse(api, circuit.Claim_Virtual_RdWriteValue_SpartanOuter)), 167, poseidon.ByteReverse(api, circuit.Claim_Virtual_RamReadValue_SpartanOuter)), 168, poseidon.ByteReverse(api, circuit.Claim_Virtual_RamWriteValue_SpartanOuter)), 169, poseidon.ByteReverse(api, circuit.Claim_Virtual_LeftLookupOperand_SpartanOuter)), 170, poseidon.ByteReverse(api, circuit.Claim_Virtual_RightLookupOperand_SpartanOuter)), 171, poseidon.ByteReverse(api, circuit.Claim_Virtual_NextUnexpandedPC_SpartanOuter)), 172, poseidon.ByteReverse(api, circuit.Claim_Virtual_NextPC_SpartanOuter)), 173, poseidon.ByteReverse(api, circuit.Claim_Virtual_NextIsVirtual_SpartanOuter)), 174, poseidon.ByteReverse(api, circuit.Claim_Virtual_NextIsFirstInSequence_SpartanOuter)), 175, poseidon.ByteReverse(api, circuit.Claim_Virtual_LookupOutput_SpartanOuter)), 176, poseidon.ByteReverse(api, circuit.Claim_Virtual_ShouldJump_SpartanOuter)), 177, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_AddOperands_SpartanOuter)), 178, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_SubtractOperands_SpartanOuter)), 179, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_MultiplyOperands_SpartanOuter)), 180, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_Load_SpartanOuter)), 181, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_Store_SpartanOuter)), 182, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_Jump_SpartanOuter)), 183, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_WriteLookupOutputToRD_SpartanOuter)), 184, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_VirtualInstruction_SpartanOuter)), 185, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_Assert_SpartanOuter)), 186, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_DoNotUpdateUnexpandedPC_SpartanOuter)), 187, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_Advice_SpartanOuter)), 188, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_IsCompressed_SpartanOuter)), 189, poseidon.ByteReverse(api, circuit.Claim_Virtual_OpFlags_IsFirstInSequence_SpartanOuter)), 190, 0)
	cse_3_10 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_9, 191, bigInt("693065686773592458709161276463075796193455407009757267193429")), 192, poseidon.ByteReverse(api, circuit.Stage2_Uni_Skip_Coeff_0)), 193, poseidon.ByteReverse(api, circuit.Stage2_Uni_Skip_Coeff_1)), 194, poseidon.ByteReverse(api, circuit.Stage2_Uni_Skip_Coeff_2)), 195, poseidon.ByteReverse(api, circuit.Stage2_Uni_Skip_Coeff_3)), 196, poseidon.ByteReverse(api, circuit.Stage2_Uni_Skip_Coeff_4)), 197, poseidon.ByteReverse(api, circuit.Stage2_Uni_Skip_Coeff_5)), 198, poseidon.ByteReverse(api, circuit.Stage2_Uni_Skip_Coeff_6)), 199, poseidon.ByteReverse(api, circuit.Stage2_Uni_Skip_Coeff_7)), 200, poseidon.ByteReverse(api, circuit.Stage2_Uni_Skip_Coeff_8)), 201, poseidon.ByteReverse(api, circuit.Stage2_Uni_Skip_Coeff_9)), 202, poseidon.ByteReverse(api, circuit.Stage2_Uni_Skip_Coeff_10)), 203, poseidon.ByteReverse(api, circuit.Stage2_Uni_Skip_Coeff_11)), 204, poseidon.ByteReverse(api, circuit.Stage2_Uni_Skip_Coeff_12)), 205, bigInt("9619401173246373414507010453289387209824226095986339413")), 206, 0)
	cse_3_11 := poseidon.Hash(api, poseidon.Hash(api, cse_3_10, 207, poseidon.ByteReverse(api, circuit.Claim_Virtual_UnivariateSkip_ProductVirtualization)), 208, 0)
	cse_3_12 := poseidon.Hash(api, cse_3_11, 209, 0)
	cse_3_13 := poseidon.Hash(api, cse_3_12, 210, 0)
	cse_3_14 := poseidon.Hash(api, cse_3_13, 211, 0)
	cse_3_15 := poseidon.Hash(api, cse_3_14, 212, 0)
	cse_3_16 := poseidon.Hash(api, cse_3_15, 213, 0)
	cse_3_17 := poseidon.Hash(api, cse_3_16, 214, 0)
	cse_3_18 := poseidon.Hash(api, cse_3_17, 215, 0)
	cse_3_19 := poseidon.Hash(api, cse_3_18, 216, 0)
	cse_3_20 := poseidon.Hash(api, cse_3_19, 217, 0)
	cse_3_21 := poseidon.Hash(api, cse_3_20, 218, 0)
	cse_3_22 := poseidon.Hash(api, cse_3_21, 219, 0)
	cse_3_23 := poseidon.Hash(api, cse_3_22, 220, 0)
	cse_3_24 := poseidon.Hash(api, cse_3_23, 221, 0)
	cse_3_25 := poseidon.Hash(api, cse_3_24, 222, 0)
	cse_3_26 := poseidon.Truncate128(api, cse_3_11)
	cse_3_27 := poseidon.Truncate128(api, cse_3_25)
	cse_3_28 := api.Mul(cse_3_27, cse_3_27)
	cse_3_29 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_25, 223, poseidon.ByteReverse(api, circuit.Claim_Virtual_UnivariateSkip_ProductVirtualization)), 224, poseidon.ByteReverse(api, circuit.Claim_Virtual_RamAddress_SpartanOuter)), 225, poseidon.ByteReverse(api, api.Add(circuit.Claim_Virtual_RamReadValue_SpartanOuter, api.Mul(cse_3_26, circuit.Claim_Virtual_RamWriteValue_SpartanOuter)))), 226, poseidon.ByteReverse(api, 0)), 227, poseidon.ByteReverse(api, api.Add(api.Add(circuit.Claim_Virtual_LookupOutput_SpartanOuter, api.Mul(cse_3_27, circuit.Claim_Virtual_LeftLookupOperand_SpartanOuter)), api.Mul(cse_3_28, circuit.Claim_Virtual_RightLookupOperand_SpartanOuter)))), 228, 0)
	cse_3_30 := poseidon.Hash(api, cse_3_29, 229, 0)
	cse_3_31 := poseidon.Hash(api, cse_3_30, 230, 0)
	cse_3_32 := poseidon.Hash(api, cse_3_31, 231, 0)
	cse_3_33 := poseidon.Hash(api, cse_3_32, 232, 0)
	cse_3_34 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_33, 233, bigInt("8747718800733414012499765325397")), 234, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R0_0)), 235, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R0_1)), 236, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R0_2)), 237, bigInt("121413912275379154240237141")), 238, 0)
	cse_3_35 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_34, 239, bigInt("8747718800733414012499765325397")), 240, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R1_0)), 241, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R1_1)), 242, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R1_2)), 243, bigInt("121413912275379154240237141")), 244, 0)
	cse_3_36 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_35, 245, bigInt("8747718800733414012499765325397")), 246, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R2_0)), 247, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R2_1)), 248, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R2_2)), 249, bigInt("121413912275379154240237141")), 250, 0)
	cse_3_37 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_36, 251, bigInt("8747718800733414012499765325397")), 252, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R3_0)), 253, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R3_1)), 254, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R3_2)), 255, bigInt("121413912275379154240237141")), 256, 0)
	cse_3_38 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_37, 257, bigInt("8747718800733414012499765325397")), 258, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R4_0)), 259, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R4_1)), 260, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R4_2)), 261, bigInt("121413912275379154240237141")), 262, 0)
	cse_3_39 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_38, 263, bigInt("8747718800733414012499765325397")), 264, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R5_0)), 265, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R5_1)), 266, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R5_2)), 267, bigInt("121413912275379154240237141")), 268, 0)
	cse_3_40 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_39, 269, bigInt("8747718800733414012499765325397")), 270, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R6_0)), 271, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R6_1)), 272, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R6_2)), 273, bigInt("121413912275379154240237141")), 274, 0)
	cse_3_41 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_40, 275, bigInt("8747718800733414012499765325397")), 276, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R7_0)), 277, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R7_1)), 278, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R7_2)), 279, bigInt("121413912275379154240237141")), 280, 0)
	cse_3_42 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_41, 281, bigInt("8747718800733414012499765325397")), 282, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R8_0)), 283, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R8_1)), 284, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R8_2)), 285, bigInt("121413912275379154240237141")), 286, 0)
	cse_3_43 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_42, 287, bigInt("8747718800733414012499765325397")), 288, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R9_0)), 289, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R9_1)), 290, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R9_2)), 291, bigInt("121413912275379154240237141")), 292, 0)
	cse_3_44 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_43, 293, bigInt("8747718800733414012499765325397")), 294, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R10_0)), 295, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R10_1)), 296, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R10_2)), 297, bigInt("121413912275379154240237141")), 298, 0)
	cse_3_45 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_44, 299, bigInt("8747718800733414012499765325397")), 300, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R11_0)), 301, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R11_1)), 302, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R11_2)), 303, bigInt("121413912275379154240237141")), 304, 0)
	cse_3_46 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_45, 305, bigInt("8747718800733414012499765325397")), 306, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R12_0)), 307, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R12_1)), 308, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R12_2)), 309, bigInt("121413912275379154240237141")), 310, 0)
	cse_3_47 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_46, 311, bigInt("8747718800733414012499765325397")), 312, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R13_0)), 313, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R13_1)), 314, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R13_2)), 315, bigInt("121413912275379154240237141")), 316, 0)
	cse_3_48 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_47, 317, bigInt("8747718800733414012499765325397")), 318, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R14_0)), 319, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R14_1)), 320, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R14_2)), 321, bigInt("121413912275379154240237141")), 322, 0)
	cse_3_49 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_48, 323, bigInt("8747718800733414012499765325397")), 324, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R15_0)), 325, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R15_1)), 326, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R15_2)), 327, bigInt("121413912275379154240237141")), 328, 0)
	cse_3_50 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_49, 329, bigInt("8747718800733414012499765325397")), 330, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R16_0)), 331, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R16_1)), 332, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R16_2)), 333, bigInt("121413912275379154240237141")), 334, 0)
	cse_3_51 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_50, 335, bigInt("8747718800733414012499765325397")), 336, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R17_0)), 337, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R17_1)), 338, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R17_2)), 339, bigInt("121413912275379154240237141")), 340, 0)
	cse_3_52 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_51, 341, bigInt("8747718800733414012499765325397")), 342, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R18_0)), 343, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R18_1)), 344, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R18_2)), 345, bigInt("121413912275379154240237141")), 346, 0)
	cse_3_53 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_52, 347, bigInt("8747718800733414012499765325397")), 348, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R19_0)), 349, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R19_1)), 350, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R19_2)), 351, bigInt("121413912275379154240237141")), 352, 0)
	cse_3_54 := poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_53, 353, bigInt("8747718800733414012499765325397")), 354, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R20_0)), 355, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R20_1)), 356, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R20_2)), 357, bigInt("121413912275379154240237141")), 358, 0)
	cse_3_55 := poseidon.Truncate128Reverse(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, cse_3_54, 359, bigInt("8747718800733414012499765325397")), 360, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R21_0)), 361, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R21_1)), 362, poseidon.ByteReverse(api, circuit.Stage2_Sumcheck_R21_2)), 363, bigInt("121413912275379154240237141")), 364, 0))
	cse_3_56 := poseidon.Truncate128Reverse(api, cse_3_54)
	cse_3_57 := poseidon.Truncate128Reverse(api, cse_3_53)
	cse_3_58 := poseidon.Truncate128Reverse(api, cse_3_52)
	cse_3_59 := poseidon.Truncate128Reverse(api, cse_3_51)
	cse_3_60 := poseidon.Truncate128Reverse(api, cse_3_50)
	cse_3_61 := poseidon.Truncate128Reverse(api, cse_3_49)
	cse_3_62 := poseidon.Truncate128Reverse(api, cse_3_48)
	cse_3_63 := poseidon.Truncate128Reverse(api, cse_3_47)
	cse_3_64 := poseidon.Truncate128Reverse(api, cse_3_46)
	cse_3_65 := poseidon.Truncate128Reverse(api, cse_3_45)
	cse_3_66 := poseidon.Truncate128Reverse(api, cse_3_44)
	cse_3_67 := poseidon.Truncate128Reverse(api, cse_3_43)
	cse_3_68 := poseidon.Truncate128Reverse(api, cse_3_42)
	cse_3_69 := poseidon.Truncate128Reverse(api, cse_3_41)
	cse_3_70 := poseidon.Truncate128Reverse(api, cse_3_40)
	cse_3_71 := poseidon.Truncate128Reverse(api, cse_3_39)
	cse_3_72 := poseidon.Truncate128Reverse(api, cse_3_38)
	cse_3_73 := poseidon.Truncate128Reverse(api, cse_3_37)
	cse_3_74 := poseidon.Truncate128Reverse(api, cse_3_36)
	cse_3_75 := poseidon.Truncate128Reverse(api, cse_3_35)
	cse_3_76 := poseidon.Truncate128Reverse(api, cse_3_34)
	cse_3_77 := poseidon.Truncate128(api, cse_3_29)
	cse_3_78 := poseidon.Truncate128(api, cse_3_30)
	cse_3_79 := poseidon.Truncate128(api, cse_3_31)
	cse_3_80 := poseidon.Truncate128(api, cse_3_33)
	cse_3_81 := api.Mul(cse_3_76, cse_3_76)
	cse_3_82 := api.Mul(cse_3_75, cse_3_75)
	cse_3_83 := api.Mul(cse_3_74, cse_3_74)
	cse_3_84 := api.Mul(cse_3_73, cse_3_73)
	cse_3_85 := api.Mul(cse_3_72, cse_3_72)
	cse_3_86 := api.Mul(cse_3_71, cse_3_71)
	cse_3_87 := api.Mul(cse_3_70, cse_3_70)
	cse_3_88 := api.Mul(cse_3_69, cse_3_69)
	cse_3_89 := api.Mul(cse_3_68, cse_3_68)
	cse_3_90 := api.Mul(cse_3_67, cse_3_67)
	cse_3_91 := api.Mul(cse_3_66, cse_3_66)
	cse_3_92 := api.Mul(cse_3_65, cse_3_65)
	cse_3_93 := api.Mul(cse_3_64, cse_3_64)
	cse_3_94 := api.Mul(cse_3_63, cse_3_63)
	cse_3_95 := api.Mul(cse_3_62, cse_3_62)
	cse_3_96 := api.Mul(cse_3_61, cse_3_61)
	cse_3_97 := api.Mul(cse_3_60, cse_3_60)
	cse_3_98 := api.Mul(cse_3_59, cse_3_59)
	cse_3_99 := api.Mul(cse_3_58, cse_3_58)
	cse_3_100 := api.Mul(cse_3_57, cse_3_57)
	cse_3_101 := api.Mul(cse_3_56, cse_3_56)
	cse_3_102 := api.Mul(cse_3_55, cse_3_55)
	cse_3_103 := api.Mul(1, 24)
	cse_3_104 := api.Mul(cse_3_103, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495611"))
	cse_3_105 := api.Mul(cse_3_104, 4)
	cse_3_106 := api.Mul(1, 24)
	cse_3_107 := api.Mul(cse_3_106, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495611"))
	cse_3_108 := api.Mul(cse_3_107, 4)
	cse_3_109 := api.Mul(cse_3_108, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495611"))
	cse_3_110 := api.Inverse(api.Mul(cse_3_109, 24))
	cse_3_111 := api.Mul(api.Mul(1, api.Mul(cse_3_105, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495611"))), cse_3_110)
	cse_3_112 := api.Sub(poseidon.Truncate128Reverse(api, cse_3_9), bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495615"))
	cse_3_113 := api.Sub(cse_3_112, 1)
	cse_3_114 := api.Sub(cse_3_113, 1)
	cse_3_115 := api.Sub(cse_3_114, 1)
	cse_3_116 := api.Sub(cse_3_115, 1)
	cse_3_117 := api.Mul(1, cse_3_116)
	cse_3_118 := api.Mul(cse_3_117, cse_3_115)
	cse_3_119 := api.Mul(cse_3_118, cse_3_114)
	cse_3_120 := api.Mul(1, cse_3_112)
	cse_3_121 := api.Mul(cse_3_120, cse_3_113)
	cse_3_122 := api.Mul(cse_3_121, cse_3_114)
	cse_3_123 := api.Mul(cse_3_122, cse_3_115)
	cse_3_124 := api.Inverse(api.Mul(cse_3_123, cse_3_116))
	cse_3_125 := api.Mul(cse_3_111, api.Mul(api.Mul(1, api.Mul(cse_3_119, cse_3_113)), cse_3_124))
	cse_3_126 := poseidon.Truncate128Reverse(api, cse_3_10)
	cse_3_127 := api.Sub(cse_3_126, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495615"))
	cse_3_128 := api.Sub(cse_3_127, 1)
	cse_3_129 := api.Sub(cse_3_128, 1)
	cse_3_130 := api.Sub(cse_3_129, 1)
	cse_3_131 := api.Sub(cse_3_130, 1)
	cse_3_132 := api.Mul(1, cse_3_131)
	cse_3_133 := api.Mul(cse_3_132, cse_3_130)
	cse_3_134 := api.Mul(cse_3_133, cse_3_129)
	cse_3_135 := api.Mul(1, cse_3_127)
	cse_3_136 := api.Mul(cse_3_135, cse_3_128)
	cse_3_137 := api.Mul(cse_3_136, cse_3_129)
	cse_3_138 := api.Mul(cse_3_137, cse_3_130)
	cse_3_139 := api.Inverse(api.Mul(cse_3_138, cse_3_131))
	cse_3_140 := api.Mul(cse_3_111, api.Mul(api.Mul(1, api.Mul(cse_3_134, cse_3_128)), cse_3_139))
	cse_3_141 := api.Mul(api.Mul(cse_3_106, cse_3_105), cse_3_110)
	cse_3_142 := api.Mul(cse_3_141, api.Mul(api.Mul(cse_3_120, cse_3_119), cse_3_124))
	cse_3_143 := api.Mul(cse_3_141, api.Mul(api.Mul(cse_3_135, cse_3_134), cse_3_139))
	cse_3_144 := api.Mul(api.Mul(cse_3_107, cse_3_104), cse_3_110)
	cse_3_145 := api.Mul(cse_3_144, api.Mul(api.Mul(cse_3_121, cse_3_118), cse_3_124))
	cse_3_146 := api.Mul(cse_3_144, api.Mul(api.Mul(cse_3_136, cse_3_133), cse_3_139))
	cse_3_147 := api.Mul(api.Mul(cse_3_108, cse_3_103), cse_3_110)
	cse_3_148 := api.Mul(cse_3_147, api.Mul(api.Mul(cse_3_122, cse_3_117), cse_3_124))
	cse_3_149 := api.Mul(cse_3_147, api.Mul(api.Mul(cse_3_137, cse_3_132), cse_3_139))
	cse_3_150 := api.Mul(api.Mul(cse_3_109, 1), cse_3_110)
	cse_3_151 := api.Mul(cse_3_150, api.Mul(api.Mul(cse_3_123, 1), cse_3_124))
	cse_3_152 := api.Mul(cse_3_150, api.Mul(api.Mul(cse_3_138, 1), cse_3_139))
	cse_3_153 := poseidon.Truncate128Reverse(api, cse_3_8)
	cse_3_154 := poseidon.Truncate128Reverse(api, cse_3_7)
	cse_3_155 := poseidon.Truncate128Reverse(api, cse_3_6)
	cse_3_156 := poseidon.Truncate128Reverse(api, cse_3_5)
	cse_3_157 := poseidon.Truncate128Reverse(api, cse_3_4)
	cse_3_158 := poseidon.Truncate128Reverse(api, cse_3_3)
	cse_3_159 := poseidon.Truncate128Reverse(api, cse_3_2)
	cse_3_160 := poseidon.Truncate128Reverse(api, cse_3_1)
	cse_3_161 := poseidon.Truncate128Reverse(api, cse_3_0)
	cse_3_162 := api.Mul(1, 24)
	cse_3_163 := api.Mul(cse_3_162, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495611"))
	cse_3_164 := api.Mul(cse_3_163, 4)
	cse_3_165 := api.Mul(1, 24)
	cse_3_166 := api.Mul(cse_3_165, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495611"))
	cse_3_167 := api.Mul(cse_3_166, 4)
	cse_3_168 := api.Mul(cse_3_167, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495611"))
	cse_3_169 := api.Inverse(api.Mul(cse_3_168, 24))
	cse_3_170 := api.Sub(cse_3_126, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495615"))
	cse_3_171 := api.Sub(cse_3_170, 1)
	cse_3_172 := api.Sub(cse_3_171, 1)
	cse_3_173 := api.Sub(cse_3_172, 1)
	cse_3_174 := api.Sub(cse_3_173, 1)
	cse_3_175 := api.Mul(1, cse_3_174)
	cse_3_176 := api.Mul(cse_3_175, cse_3_173)
	cse_3_177 := api.Mul(cse_3_176, cse_3_172)
	cse_3_178 := api.Mul(1, cse_3_170)
	cse_3_179 := api.Mul(cse_3_178, cse_3_171)
	cse_3_180 := api.Mul(cse_3_179, cse_3_172)
	cse_3_181 := api.Mul(cse_3_180, cse_3_173)
	cse_3_182 := api.Inverse(api.Mul(cse_3_181, cse_3_174))
	cse_3_183 := api.Mul(api.Mul(api.Mul(1, api.Mul(cse_3_164, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495611"))), cse_3_169), api.Mul(api.Mul(1, api.Mul(cse_3_177, cse_3_171)), cse_3_182))
	cse_3_184 := api.Mul(api.Mul(api.Mul(cse_3_165, cse_3_164), cse_3_169), api.Mul(api.Mul(cse_3_178, cse_3_177), cse_3_182))
	cse_3_185 := api.Mul(api.Mul(api.Mul(cse_3_166, cse_3_163), cse_3_169), api.Mul(api.Mul(cse_3_179, cse_3_176), cse_3_182))
	cse_3_186 := api.Mul(api.Mul(api.Mul(cse_3_167, cse_3_162), cse_3_169), api.Mul(api.Mul(cse_3_180, cse_3_175), cse_3_182))
	cse_3_187 := api.Mul(api.Mul(api.Mul(cse_3_168, 1), cse_3_169), api.Mul(api.Mul(cse_3_181, 1), cse_3_182))
	cse_3_188 := api.Inverse(api.Add(api.Add(api.Add(api.Add(cse_3_183, cse_3_184), cse_3_185), cse_3_186), cse_3_187))
	cse_3_189 := api.Mul(cse_3_183, cse_3_188)
	cse_3_190 := api.Mul(cse_3_184, cse_3_188)
	cse_3_191 := api.Mul(cse_3_185, cse_3_188)
	cse_3_192 := api.Mul(cse_3_186, cse_3_188)
	cse_3_193 := api.Mul(cse_3_187, cse_3_188)
	cse_3_194 := poseidon.Truncate128Reverse(api, cse_3_12)
	cse_3_195 := poseidon.Truncate128Reverse(api, cse_3_13)
	cse_3_196 := poseidon.Truncate128Reverse(api, cse_3_14)
	cse_3_197 := poseidon.Truncate128Reverse(api, cse_3_15)
	cse_3_198 := poseidon.Truncate128Reverse(api, cse_3_16)
	cse_3_199 := poseidon.Truncate128Reverse(api, cse_3_17)
	cse_3_200 := poseidon.Truncate128Reverse(api, cse_3_18)
	cse_3_201 := poseidon.Truncate128Reverse(api, cse_3_19)
	cse_3_202 := poseidon.Truncate128Reverse(api, cse_3_20)
	cse_3_203 := poseidon.Truncate128Reverse(api, cse_3_21)
	cse_3_204 := poseidon.Truncate128Reverse(api, cse_3_22)
	cse_3_205 := poseidon.Truncate128Reverse(api, cse_3_23)
	cse_3_206 := poseidon.Truncate128Reverse(api, cse_3_24)
	cse_3_207 := api.Sub(1, api.Mul(1, cse_3_62))
	cse_3_208 := api.Sub(cse_3_207, api.Mul(cse_3_207, cse_3_63))
	cse_3_209 := api.Sub(cse_3_208, api.Mul(cse_3_208, cse_3_64))
	cse_3_210 := api.Sub(cse_3_209, api.Mul(cse_3_209, cse_3_65))
	cse_3_211 := api.Sub(cse_3_210, api.Mul(cse_3_210, cse_3_66))
	cse_3_212 := api.Mul(cse_3_211, cse_3_67)
	cse_3_213 := api.Sub(cse_3_211, cse_3_212)
	cse_3_214 := api.Mul(1, cse_3_56)
	cse_3_215 := api.Mul(api.Sub(1, cse_3_214), cse_3_57)
	cse_3_216 := api.Mul(cse_3_215, cse_3_58)
	cse_3_217 := api.Sub(cse_3_215, cse_3_216)
	cse_3_218 := api.Sub(cse_3_217, api.Mul(cse_3_217, cse_3_59))
	cse_3_219 := api.Sub(cse_3_218, api.Mul(cse_3_218, cse_3_60))
	cse_3_220 := api.Sub(cse_3_216, api.Mul(cse_3_216, cse_3_59))
	cse_3_221 := api.Sub(cse_3_220, api.Mul(cse_3_220, cse_3_60))
	cse_3_222 := api.Sub(cse_3_214, api.Mul(cse_3_214, cse_3_57))
	cse_3_223 := api.Sub(cse_3_222, api.Mul(cse_3_222, cse_3_58))
	cse_3_224 := api.Sub(cse_3_223, api.Mul(cse_3_223, cse_3_59))
	cse_3_225 := api.Sub(cse_3_224, api.Mul(cse_3_224, cse_3_60))

	// Verification assertions (each must equal 0)
	a0 := api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Mul(circuit.Stage1_Uni_Skip_Coeff_0, 10), api.Mul(circuit.Stage1_Uni_Skip_Coeff_1, 5)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_2, 85)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_3, 125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_4, 1333)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_5, 3125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_6, 25405)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_7, 78125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_8, 535333)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_9, 1953125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_10, 11982925)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_11, 48828125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_12, 278766133)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_13, 1220703125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_14, 6649985245)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_15, 30517578125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_16, 161264049733)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_17, 762939453125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_18, 3952911584365)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_19, 19073486328125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_20, 97573430562133)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_21, 476837158203125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_22, 2419432933612285)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_23, 11920928955078125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_24, 60168159621439333)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_25, 298023223876953125)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_26, 1499128402505381005)), api.Mul(circuit.Stage1_Uni_Skip_Coeff_27, 7450580596923828125))
	api.AssertIsEqual(a0, 0)
	a1 := api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R9_0, api.Mul(cse_1_22, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R8_0, api.Mul(cse_1_23, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R7_0, api.Mul(cse_1_24, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R6_0, api.Mul(cse_1_25, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R5_0, api.Mul(cse_1_26, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R4_0, api.Mul(cse_1_27, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R3_0, api.Mul(cse_1_28, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R2_0, api.Mul(cse_1_29, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R1_0, api.Mul(cse_1_30, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage1_Sumcheck_R0_0, api.Mul(cse_1_31, api.Sub(api.Sub(api.Sub(api.Sub(api.Mul(api.Mul(circuit.Claim_Virtual_UnivariateSkip_SpartanOuter, 1), cse_1_32), circuit.Stage1_Sumcheck_R0_0), circuit.Stage1_Sumcheck_R0_0), circuit.Stage1_Sumcheck_R0_1), circuit.Stage1_Sumcheck_R0_2))), api.Mul(circuit.Stage1_Sumcheck_R0_1, cse_1_33)), api.Mul(circuit.Stage1_Sumcheck_R0_2, api.Mul(cse_1_33, cse_1_31))), circuit.Stage1_Sumcheck_R1_0), circuit.Stage1_Sumcheck_R1_0), circuit.Stage1_Sumcheck_R1_1), circuit.Stage1_Sumcheck_R1_2))), api.Mul(circuit.Stage1_Sumcheck_R1_1, cse_1_34)), api.Mul(circuit.Stage1_Sumcheck_R1_2, api.Mul(cse_1_34, cse_1_30))), circuit.Stage1_Sumcheck_R2_0), circuit.Stage1_Sumcheck_R2_0), circuit.Stage1_Sumcheck_R2_1), circuit.Stage1_Sumcheck_R2_2))), api.Mul(circuit.Stage1_Sumcheck_R2_1, cse_1_35)), api.Mul(circuit.Stage1_Sumcheck_R2_2, api.Mul(cse_1_35, cse_1_29))), circuit.Stage1_Sumcheck_R3_0), circuit.Stage1_Sumcheck_R3_0), circuit.Stage1_Sumcheck_R3_1), circuit.Stage1_Sumcheck_R3_2))), api.Mul(circuit.Stage1_Sumcheck_R3_1, cse_1_36)), api.Mul(circuit.Stage1_Sumcheck_R3_2, api.Mul(cse_1_36, cse_1_28))), circuit.Stage1_Sumcheck_R4_0), circuit.Stage1_Sumcheck_R4_0), circuit.Stage1_Sumcheck_R4_1), circuit.Stage1_Sumcheck_R4_2))), api.Mul(circuit.Stage1_Sumcheck_R4_1, cse_1_37)), api.Mul(circuit.Stage1_Sumcheck_R4_2, api.Mul(cse_1_37, cse_1_27))), circuit.Stage1_Sumcheck_R5_0), circuit.Stage1_Sumcheck_R5_0), circuit.Stage1_Sumcheck_R5_1), circuit.Stage1_Sumcheck_R5_2))), api.Mul(circuit.Stage1_Sumcheck_R5_1, cse_1_38)), api.Mul(circuit.Stage1_Sumcheck_R5_2, api.Mul(cse_1_38, cse_1_26))), circuit.Stage1_Sumcheck_R6_0), circuit.Stage1_Sumcheck_R6_0), circuit.Stage1_Sumcheck_R6_1), circuit.Stage1_Sumcheck_R6_2))), api.Mul(circuit.Stage1_Sumcheck_R6_1, cse_1_39)), api.Mul(circuit.Stage1_Sumcheck_R6_2, api.Mul(cse_1_39, cse_1_25))), circuit.Stage1_Sumcheck_R7_0), circuit.Stage1_Sumcheck_R7_0), circuit.Stage1_Sumcheck_R7_1), circuit.Stage1_Sumcheck_R7_2))), api.Mul(circuit.Stage1_Sumcheck_R7_1, cse_1_40)), api.Mul(circuit.Stage1_Sumcheck_R7_2, api.Mul(cse_1_40, cse_1_24))), circuit.Stage1_Sumcheck_R8_0), circuit.Stage1_Sumcheck_R8_0), circuit.Stage1_Sumcheck_R8_1), circuit.Stage1_Sumcheck_R8_2))), api.Mul(circuit.Stage1_Sumcheck_R8_1, cse_1_41)), api.Mul(circuit.Stage1_Sumcheck_R8_2, api.Mul(cse_1_41, cse_1_23))), circuit.Stage1_Sumcheck_R9_0), circuit.Stage1_Sumcheck_R9_0), circuit.Stage1_Sumcheck_R9_1), circuit.Stage1_Sumcheck_R9_2))), api.Mul(circuit.Stage1_Sumcheck_R9_1, cse_1_42)), api.Mul(circuit.Stage1_Sumcheck_R9_2, api.Mul(cse_1_42, cse_1_22))), api.Mul(api.Mul(api.Mul(api.Mul(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Mul(cse_1_90, cse_1_120), api.Mul(cse_1_122, cse_1_123)), api.Mul(cse_1_125, cse_1_126)), api.Mul(cse_1_128, cse_1_129)), api.Mul(cse_1_131, cse_1_132)), api.Mul(cse_1_134, cse_1_135)), api.Mul(cse_1_137, cse_1_138)), api.Mul(cse_1_140, cse_1_141)), api.Mul(cse_1_143, cse_1_144)), api.Mul(cse_1_146, cse_1_147)), api.Inverse(api.Mul(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(cse_1_90, cse_1_122), cse_1_125), cse_1_128), cse_1_131), cse_1_134), cse_1_137), cse_1_140), cse_1_143), cse_1_146), api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(cse_1_120, cse_1_123), cse_1_126), cse_1_129), cse_1_132), cse_1_135), cse_1_138), cse_1_141), cse_1_144), cse_1_147)))), api.Mul(api.Mul(api.Mul(api.Mul(1, api.Add(api.Mul(cse_1_148, cse_1_22), api.Mul(api.Sub(1, cse_1_148), api.Sub(1, cse_1_22)))), api.Mul(1, api.Add(api.Mul(cse_1_149, cse_1_23), api.Mul(api.Sub(1, cse_1_149), api.Sub(1, cse_1_23))))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_1_150, cse_1_24), api.Mul(api.Sub(1, cse_1_150), api.Sub(1, cse_1_24)))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_1_151, cse_1_25), api.Mul(api.Sub(1, cse_1_151), api.Sub(1, cse_1_25)))), api.Mul(1, api.Add(api.Mul(cse_1_152, cse_1_26), api.Mul(api.Sub(1, cse_1_152), api.Sub(1, cse_1_26))))))), api.Mul(api.Mul(api.Mul(1, api.Add(api.Mul(cse_1_153, cse_1_27), api.Mul(api.Sub(1, cse_1_153), api.Sub(1, cse_1_27)))), api.Mul(1, api.Add(api.Mul(cse_1_154, cse_1_28), api.Mul(api.Sub(1, cse_1_154), api.Sub(1, cse_1_28))))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_1_155, cse_1_29), api.Mul(api.Sub(1, cse_1_155), api.Sub(1, cse_1_29)))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_1_156, cse_1_30), api.Mul(api.Sub(1, cse_1_156), api.Sub(1, cse_1_30)))), api.Mul(1, api.Add(api.Mul(cse_1_157, cse_1_31), api.Mul(api.Sub(1, cse_1_157), api.Sub(1, cse_1_31))))))))), api.Mul(api.Add(cse_1_225, api.Mul(cse_1_31, api.Sub(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Mul(cse_1_215, api.Add(api.Mul(circuit.Claim_Virtual_OpFlags_Load_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_OpFlags_Store_SpartanOuter, 1))), api.Mul(cse_1_216, api.Mul(circuit.Claim_Virtual_OpFlags_AddOperands_SpartanOuter, 1))), api.Mul(cse_1_217, api.Mul(circuit.Claim_Virtual_OpFlags_SubtractOperands_SpartanOuter, 1))), api.Mul(cse_1_218, api.Mul(circuit.Claim_Virtual_OpFlags_MultiplyOperands_SpartanOuter, 1))), api.Mul(cse_1_219, api.Add(api.Add(api.Add(api.Add(api.Mul(circuit.Claim_Virtual_OpFlags_AddOperands_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")), api.Mul(circuit.Claim_Virtual_OpFlags_SubtractOperands_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_OpFlags_MultiplyOperands_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_OpFlags_Advice_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(1, 1)))), api.Mul(cse_1_220, api.Mul(circuit.Claim_Virtual_WriteLookupOutputToRD_SpartanOuter, 1))), api.Mul(cse_1_221, api.Mul(circuit.Claim_Virtual_WritePCtoRD_SpartanOuter, 1))), api.Mul(cse_1_222, api.Mul(circuit.Claim_Virtual_ShouldBranch_SpartanOuter, 1))), api.Mul(cse_1_223, api.Add(api.Add(api.Mul(circuit.Claim_Virtual_ShouldBranch_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")), api.Mul(circuit.Claim_Virtual_OpFlags_Jump_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(1, 1)))), cse_1_225))), api.Add(cse_1_226, api.Mul(cse_1_31, api.Sub(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Mul(cse_1_215, api.Add(api.Add(api.Mul(circuit.Claim_Virtual_RamAddress_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_Rs1Value_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_Imm_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616")))), api.Mul(cse_1_216, api.Add(api.Add(api.Mul(circuit.Claim_Virtual_RightLookupOperand_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_LeftInstructionInput_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_RightInstructionInput_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_1_217, api.Add(api.Add(api.Add(api.Mul(circuit.Claim_Virtual_RightLookupOperand_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_LeftInstructionInput_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_RightInstructionInput_SpartanOuter, 1)), api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343679757442502098944001"))))), api.Mul(cse_1_218, api.Add(api.Mul(circuit.Claim_Virtual_RightLookupOperand_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_Product_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_1_219, api.Add(api.Mul(circuit.Claim_Virtual_RightLookupOperand_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_RightInstructionInput_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_1_220, api.Add(api.Mul(circuit.Claim_Virtual_RdWriteValue_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_LookupOutput_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_1_221, api.Add(api.Add(api.Add(api.Mul(circuit.Claim_Virtual_RdWriteValue_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_UnexpandedPC_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_OpFlags_IsCompressed_SpartanOuter, 2)), api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495613"))))), api.Mul(cse_1_222, api.Add(api.Add(api.Mul(circuit.Claim_Virtual_NextUnexpandedPC_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_UnexpandedPC_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_Imm_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))))), api.Mul(cse_1_223, api.Add(api.Add(api.Add(api.Add(api.Mul(circuit.Claim_Virtual_NextUnexpandedPC_SpartanOuter, 1), api.Mul(circuit.Claim_Virtual_UnexpandedPC_SpartanOuter, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495616"))), api.Mul(circuit.Claim_Virtual_OpFlags_DoNotUpdateUnexpandedPC_SpartanOuter, 4)), api.Mul(circuit.Claim_Virtual_OpFlags_IsCompressed_SpartanOuter, 2)), api.Mul(1, bigInt("21888242871839275222246405745257275088548364400416034343698204186575808495613"))))), cse_1_226))))), cse_1_32))
	api.AssertIsEqual(a1, 0)
	a2 := api.Sub(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Mul(circuit.Stage2_Uni_Skip_Coeff_0, 5), api.Mul(circuit.Stage2_Uni_Skip_Coeff_2, 10)), api.Mul(circuit.Stage2_Uni_Skip_Coeff_4, 34)), api.Mul(circuit.Stage2_Uni_Skip_Coeff_6, 130)), api.Mul(circuit.Stage2_Uni_Skip_Coeff_8, 514)), api.Mul(circuit.Stage2_Uni_Skip_Coeff_10, 2050)), api.Mul(circuit.Stage2_Uni_Skip_Coeff_12, 8194)), api.Add(api.Add(api.Add(api.Add(api.Mul(api.Mul(cse_2_21, cse_2_26), circuit.Claim_Virtual_Product_SpartanOuter), api.Mul(api.Mul(cse_2_22, cse_2_26), circuit.Claim_Virtual_WriteLookupOutputToRD_SpartanOuter)), api.Mul(api.Mul(cse_2_23, cse_2_26), circuit.Claim_Virtual_WritePCtoRD_SpartanOuter)), api.Mul(api.Mul(cse_2_24, cse_2_26), circuit.Claim_Virtual_ShouldBranch_SpartanOuter)), api.Mul(api.Mul(cse_2_25, cse_2_26), circuit.Claim_Virtual_ShouldJump_SpartanOuter)))
	api.AssertIsEqual(a2, 0)
	a3 := api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R21_0, api.Mul(cse_3_55, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R20_0, api.Mul(cse_3_56, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R19_0, api.Mul(cse_3_57, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R18_0, api.Mul(cse_3_58, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R17_0, api.Mul(cse_3_59, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R16_0, api.Mul(cse_3_60, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R15_0, api.Mul(cse_3_61, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R14_0, api.Mul(cse_3_62, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R13_0, api.Mul(cse_3_63, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R12_0, api.Mul(cse_3_64, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R11_0, api.Mul(cse_3_65, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R10_0, api.Mul(cse_3_66, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R9_0, api.Mul(cse_3_67, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R8_0, api.Mul(cse_3_68, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R7_0, api.Mul(cse_3_69, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R6_0, api.Mul(cse_3_70, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R5_0, api.Mul(cse_3_71, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R4_0, api.Mul(cse_3_72, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R3_0, api.Mul(cse_3_73, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R2_0, api.Mul(cse_3_74, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R1_0, api.Mul(cse_3_75, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(circuit.Stage2_Sumcheck_R0_0, api.Mul(cse_3_76, api.Sub(api.Sub(api.Sub(api.Sub(api.Add(api.Add(api.Add(api.Mul(api.Mul(circuit.Claim_Virtual_UnivariateSkip_ProductVirtualization, 8192), cse_3_77), api.Mul(api.Mul(circuit.Claim_Virtual_RamAddress_SpartanOuter, 512), cse_3_78)), api.Mul(api.Mul(api.Add(circuit.Claim_Virtual_RamReadValue_SpartanOuter, api.Mul(cse_3_26, circuit.Claim_Virtual_RamWriteValue_SpartanOuter)), 1), cse_3_79)), api.Mul(api.Mul(api.Add(api.Add(circuit.Claim_Virtual_LookupOutput_SpartanOuter, api.Mul(cse_3_27, circuit.Claim_Virtual_LeftLookupOperand_SpartanOuter)), api.Mul(cse_3_28, circuit.Claim_Virtual_RightLookupOperand_SpartanOuter)), 8192), cse_3_80)), circuit.Stage2_Sumcheck_R0_0), circuit.Stage2_Sumcheck_R0_0), circuit.Stage2_Sumcheck_R0_1), circuit.Stage2_Sumcheck_R0_2))), api.Mul(circuit.Stage2_Sumcheck_R0_1, cse_3_81)), api.Mul(circuit.Stage2_Sumcheck_R0_2, api.Mul(cse_3_81, cse_3_76))), circuit.Stage2_Sumcheck_R1_0), circuit.Stage2_Sumcheck_R1_0), circuit.Stage2_Sumcheck_R1_1), circuit.Stage2_Sumcheck_R1_2))), api.Mul(circuit.Stage2_Sumcheck_R1_1, cse_3_82)), api.Mul(circuit.Stage2_Sumcheck_R1_2, api.Mul(cse_3_82, cse_3_75))), circuit.Stage2_Sumcheck_R2_0), circuit.Stage2_Sumcheck_R2_0), circuit.Stage2_Sumcheck_R2_1), circuit.Stage2_Sumcheck_R2_2))), api.Mul(circuit.Stage2_Sumcheck_R2_1, cse_3_83)), api.Mul(circuit.Stage2_Sumcheck_R2_2, api.Mul(cse_3_83, cse_3_74))), circuit.Stage2_Sumcheck_R3_0), circuit.Stage2_Sumcheck_R3_0), circuit.Stage2_Sumcheck_R3_1), circuit.Stage2_Sumcheck_R3_2))), api.Mul(circuit.Stage2_Sumcheck_R3_1, cse_3_84)), api.Mul(circuit.Stage2_Sumcheck_R3_2, api.Mul(cse_3_84, cse_3_73))), circuit.Stage2_Sumcheck_R4_0), circuit.Stage2_Sumcheck_R4_0), circuit.Stage2_Sumcheck_R4_1), circuit.Stage2_Sumcheck_R4_2))), api.Mul(circuit.Stage2_Sumcheck_R4_1, cse_3_85)), api.Mul(circuit.Stage2_Sumcheck_R4_2, api.Mul(cse_3_85, cse_3_72))), circuit.Stage2_Sumcheck_R5_0), circuit.Stage2_Sumcheck_R5_0), circuit.Stage2_Sumcheck_R5_1), circuit.Stage2_Sumcheck_R5_2))), api.Mul(circuit.Stage2_Sumcheck_R5_1, cse_3_86)), api.Mul(circuit.Stage2_Sumcheck_R5_2, api.Mul(cse_3_86, cse_3_71))), circuit.Stage2_Sumcheck_R6_0), circuit.Stage2_Sumcheck_R6_0), circuit.Stage2_Sumcheck_R6_1), circuit.Stage2_Sumcheck_R6_2))), api.Mul(circuit.Stage2_Sumcheck_R6_1, cse_3_87)), api.Mul(circuit.Stage2_Sumcheck_R6_2, api.Mul(cse_3_87, cse_3_70))), circuit.Stage2_Sumcheck_R7_0), circuit.Stage2_Sumcheck_R7_0), circuit.Stage2_Sumcheck_R7_1), circuit.Stage2_Sumcheck_R7_2))), api.Mul(circuit.Stage2_Sumcheck_R7_1, cse_3_88)), api.Mul(circuit.Stage2_Sumcheck_R7_2, api.Mul(cse_3_88, cse_3_69))), circuit.Stage2_Sumcheck_R8_0), circuit.Stage2_Sumcheck_R8_0), circuit.Stage2_Sumcheck_R8_1), circuit.Stage2_Sumcheck_R8_2))), api.Mul(circuit.Stage2_Sumcheck_R8_1, cse_3_89)), api.Mul(circuit.Stage2_Sumcheck_R8_2, api.Mul(cse_3_89, cse_3_68))), circuit.Stage2_Sumcheck_R9_0), circuit.Stage2_Sumcheck_R9_0), circuit.Stage2_Sumcheck_R9_1), circuit.Stage2_Sumcheck_R9_2))), api.Mul(circuit.Stage2_Sumcheck_R9_1, cse_3_90)), api.Mul(circuit.Stage2_Sumcheck_R9_2, api.Mul(cse_3_90, cse_3_67))), circuit.Stage2_Sumcheck_R10_0), circuit.Stage2_Sumcheck_R10_0), circuit.Stage2_Sumcheck_R10_1), circuit.Stage2_Sumcheck_R10_2))), api.Mul(circuit.Stage2_Sumcheck_R10_1, cse_3_91)), api.Mul(circuit.Stage2_Sumcheck_R10_2, api.Mul(cse_3_91, cse_3_66))), circuit.Stage2_Sumcheck_R11_0), circuit.Stage2_Sumcheck_R11_0), circuit.Stage2_Sumcheck_R11_1), circuit.Stage2_Sumcheck_R11_2))), api.Mul(circuit.Stage2_Sumcheck_R11_1, cse_3_92)), api.Mul(circuit.Stage2_Sumcheck_R11_2, api.Mul(cse_3_92, cse_3_65))), circuit.Stage2_Sumcheck_R12_0), circuit.Stage2_Sumcheck_R12_0), circuit.Stage2_Sumcheck_R12_1), circuit.Stage2_Sumcheck_R12_2))), api.Mul(circuit.Stage2_Sumcheck_R12_1, cse_3_93)), api.Mul(circuit.Stage2_Sumcheck_R12_2, api.Mul(cse_3_93, cse_3_64))), circuit.Stage2_Sumcheck_R13_0), circuit.Stage2_Sumcheck_R13_0), circuit.Stage2_Sumcheck_R13_1), circuit.Stage2_Sumcheck_R13_2))), api.Mul(circuit.Stage2_Sumcheck_R13_1, cse_3_94)), api.Mul(circuit.Stage2_Sumcheck_R13_2, api.Mul(cse_3_94, cse_3_63))), circuit.Stage2_Sumcheck_R14_0), circuit.Stage2_Sumcheck_R14_0), circuit.Stage2_Sumcheck_R14_1), circuit.Stage2_Sumcheck_R14_2))), api.Mul(circuit.Stage2_Sumcheck_R14_1, cse_3_95)), api.Mul(circuit.Stage2_Sumcheck_R14_2, api.Mul(cse_3_95, cse_3_62))), circuit.Stage2_Sumcheck_R15_0), circuit.Stage2_Sumcheck_R15_0), circuit.Stage2_Sumcheck_R15_1), circuit.Stage2_Sumcheck_R15_2))), api.Mul(circuit.Stage2_Sumcheck_R15_1, cse_3_96)), api.Mul(circuit.Stage2_Sumcheck_R15_2, api.Mul(cse_3_96, cse_3_61))), circuit.Stage2_Sumcheck_R16_0), circuit.Stage2_Sumcheck_R16_0), circuit.Stage2_Sumcheck_R16_1), circuit.Stage2_Sumcheck_R16_2))), api.Mul(circuit.Stage2_Sumcheck_R16_1, cse_3_97)), api.Mul(circuit.Stage2_Sumcheck_R16_2, api.Mul(cse_3_97, cse_3_60))), circuit.Stage2_Sumcheck_R17_0), circuit.Stage2_Sumcheck_R17_0), circuit.Stage2_Sumcheck_R17_1), circuit.Stage2_Sumcheck_R17_2))), api.Mul(circuit.Stage2_Sumcheck_R17_1, cse_3_98)), api.Mul(circuit.Stage2_Sumcheck_R17_2, api.Mul(cse_3_98, cse_3_59))), circuit.Stage2_Sumcheck_R18_0), circuit.Stage2_Sumcheck_R18_0), circuit.Stage2_Sumcheck_R18_1), circuit.Stage2_Sumcheck_R18_2))), api.Mul(circuit.Stage2_Sumcheck_R18_1, cse_3_99)), api.Mul(circuit.Stage2_Sumcheck_R18_2, api.Mul(cse_3_99, cse_3_58))), circuit.Stage2_Sumcheck_R19_0), circuit.Stage2_Sumcheck_R19_0), circuit.Stage2_Sumcheck_R19_1), circuit.Stage2_Sumcheck_R19_2))), api.Mul(circuit.Stage2_Sumcheck_R19_1, cse_3_100)), api.Mul(circuit.Stage2_Sumcheck_R19_2, api.Mul(cse_3_100, cse_3_57))), circuit.Stage2_Sumcheck_R20_0), circuit.Stage2_Sumcheck_R20_0), circuit.Stage2_Sumcheck_R20_1), circuit.Stage2_Sumcheck_R20_2))), api.Mul(circuit.Stage2_Sumcheck_R20_1, cse_3_101)), api.Mul(circuit.Stage2_Sumcheck_R20_2, api.Mul(cse_3_101, cse_3_56))), circuit.Stage2_Sumcheck_R21_0), circuit.Stage2_Sumcheck_R21_0), circuit.Stage2_Sumcheck_R21_1), circuit.Stage2_Sumcheck_R21_2))), api.Mul(circuit.Stage2_Sumcheck_R21_1, cse_3_102)), api.Mul(circuit.Stage2_Sumcheck_R21_2, api.Mul(cse_3_102, cse_3_55))), api.Add(api.Add(api.Add(api.Add(api.Mul(api.Mul(api.Mul(api.Mul(api.Mul(api.Add(api.Add(api.Add(api.Add(api.Mul(cse_3_125, cse_3_140), api.Mul(cse_3_142, cse_3_143)), api.Mul(cse_3_145, cse_3_146)), api.Mul(cse_3_148, cse_3_149)), api.Mul(cse_3_151, cse_3_152)), api.Inverse(api.Mul(api.Add(api.Add(api.Add(api.Add(cse_3_125, cse_3_142), cse_3_145), cse_3_148), cse_3_151), api.Add(api.Add(api.Add(api.Add(cse_3_140, cse_3_143), cse_3_146), cse_3_149), cse_3_152)))), api.Mul(api.Mul(api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_153, cse_3_55), api.Mul(api.Sub(1, cse_3_153), api.Sub(1, cse_3_55)))), api.Mul(1, api.Add(api.Mul(cse_3_154, cse_3_56), api.Mul(api.Sub(1, cse_3_154), api.Sub(1, cse_3_56))))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_155, cse_3_57), api.Mul(api.Sub(1, cse_3_155), api.Sub(1, cse_3_57)))), api.Mul(1, api.Add(api.Mul(cse_3_156, cse_3_58), api.Mul(api.Sub(1, cse_3_156), api.Sub(1, cse_3_58)))))), api.Mul(api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_157, cse_3_59), api.Mul(api.Sub(1, cse_3_157), api.Sub(1, cse_3_59)))), api.Mul(1, api.Add(api.Mul(cse_3_158, cse_3_60), api.Mul(api.Sub(1, cse_3_158), api.Sub(1, cse_3_60))))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_159, cse_3_61), api.Mul(api.Sub(1, cse_3_159), api.Sub(1, cse_3_61)))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_160, cse_3_62), api.Mul(api.Sub(1, cse_3_160), api.Sub(1, cse_3_62)))), api.Mul(1, api.Add(api.Mul(cse_3_161, cse_3_63), api.Mul(api.Sub(1, cse_3_161), api.Sub(1, cse_3_63))))))))), api.Add(api.Add(api.Add(api.Add(api.Mul(cse_3_189, circuit.Claim_Virtual_LeftInstructionInput_ProductVirtualization), api.Mul(cse_3_190, circuit.Claim_Virtual_InstructionFlags_IsRdNotZero_ProductVirtualization)), api.Mul(cse_3_191, circuit.Claim_Virtual_InstructionFlags_IsRdNotZero_ProductVirtualization)), api.Mul(cse_3_192, circuit.Claim_Virtual_LookupOutput_ProductVirtualization)), api.Mul(cse_3_193, circuit.Claim_Virtual_OpFlags_Jump_ProductVirtualization))), api.Add(api.Add(api.Add(api.Add(api.Mul(cse_3_189, circuit.Claim_Virtual_RightInstructionInput_ProductVirtualization), api.Mul(cse_3_190, circuit.Claim_Virtual_OpFlags_WriteLookupOutputToRD_ProductVirtualization)), api.Mul(cse_3_191, circuit.Claim_Virtual_OpFlags_Jump_ProductVirtualization)), api.Mul(cse_3_192, circuit.Claim_Virtual_InstructionFlags_Branch_ProductVirtualization)), api.Mul(cse_3_193, api.Sub(1, circuit.Claim_Virtual_NextIsNoop_ProductVirtualization)))), cse_3_77), api.Mul(api.Mul(api.Add(api.Mul(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Add(api.Mul(cse_3_55, 4096), api.Mul(cse_3_56, 2048)), api.Mul(cse_3_57, 1024)), api.Mul(cse_3_58, 512)), api.Mul(cse_3_59, 256)), api.Mul(cse_3_60, 128)), api.Mul(cse_3_61, 64)), api.Mul(cse_3_62, 32)), api.Mul(cse_3_63, 16)), api.Mul(cse_3_64, 8)), api.Mul(cse_3_65, 4)), api.Mul(cse_3_66, 2)), api.Mul(cse_3_67, 1)), 8), 2147450880), circuit.Claim_Virtual_RamRa_RamRafEvaluation), cse_3_78)), api.Mul(api.Mul(api.Mul(api.Mul(api.Mul(api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_153, cse_3_68), api.Mul(api.Sub(1, cse_3_153), api.Sub(1, cse_3_68)))), api.Mul(1, api.Add(api.Mul(cse_3_154, cse_3_69), api.Mul(api.Sub(1, cse_3_154), api.Sub(1, cse_3_69))))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_155, cse_3_70), api.Mul(api.Sub(1, cse_3_155), api.Sub(1, cse_3_70)))), api.Mul(1, api.Add(api.Mul(cse_3_156, cse_3_71), api.Mul(api.Sub(1, cse_3_156), api.Sub(1, cse_3_71)))))), api.Mul(api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_157, cse_3_72), api.Mul(api.Sub(1, cse_3_157), api.Sub(1, cse_3_72)))), api.Mul(1, api.Add(api.Mul(cse_3_158, cse_3_73), api.Mul(api.Sub(1, cse_3_158), api.Sub(1, cse_3_73))))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_159, cse_3_74), api.Mul(api.Sub(1, cse_3_159), api.Sub(1, cse_3_74)))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_160, cse_3_75), api.Mul(api.Sub(1, cse_3_160), api.Sub(1, cse_3_75)))), api.Mul(1, api.Add(api.Mul(cse_3_161, cse_3_76), api.Mul(api.Sub(1, cse_3_161), api.Sub(1, cse_3_76)))))))), circuit.Claim_Virtual_RamRa_RamReadWriteChecking), api.Add(circuit.Claim_Virtual_RamVal_RamReadWriteChecking, api.Mul(cse_3_26, api.Add(circuit.Claim_Virtual_RamVal_RamReadWriteChecking, circuit.Claim_Committed_RamInc_RamReadWriteChecking)))), cse_3_79)), api.Mul(api.Mul(api.Mul(api.Mul(api.Mul(api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_194, cse_3_55), api.Mul(api.Sub(1, cse_3_194), api.Sub(1, cse_3_55)))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_195, cse_3_56), api.Mul(api.Sub(1, cse_3_195), api.Sub(1, cse_3_56)))), api.Mul(1, api.Add(api.Mul(cse_3_196, cse_3_57), api.Mul(api.Sub(1, cse_3_196), api.Sub(1, cse_3_57)))))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_197, cse_3_58), api.Mul(api.Sub(1, cse_3_197), api.Sub(1, cse_3_58)))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_198, cse_3_59), api.Mul(api.Sub(1, cse_3_198), api.Sub(1, cse_3_59)))), api.Mul(1, api.Add(api.Mul(cse_3_199, cse_3_60), api.Mul(api.Sub(1, cse_3_199), api.Sub(1, cse_3_60))))))), api.Mul(api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_200, cse_3_61), api.Mul(api.Sub(1, cse_3_200), api.Sub(1, cse_3_61)))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_201, cse_3_62), api.Mul(api.Sub(1, cse_3_201), api.Sub(1, cse_3_62)))), api.Mul(1, api.Add(api.Mul(cse_3_202, cse_3_63), api.Mul(api.Sub(1, cse_3_202), api.Sub(1, cse_3_63)))))), api.Mul(api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_203, cse_3_64), api.Mul(api.Sub(1, cse_3_203), api.Sub(1, cse_3_64)))), api.Mul(1, api.Add(api.Mul(cse_3_204, cse_3_65), api.Mul(api.Sub(1, cse_3_204), api.Sub(1, cse_3_65))))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_205, cse_3_66), api.Mul(api.Sub(1, cse_3_205), api.Sub(1, cse_3_66)))), api.Mul(1, api.Add(api.Mul(cse_3_206, cse_3_67), api.Mul(api.Sub(1, cse_3_206), api.Sub(1, cse_3_67)))))))), api.Sub(api.Mul(1, api.Sub(1, cse_3_55)), api.Mul(api.Mul(api.Mul(1, api.Sub(1, cse_3_55)), api.Sub(1, cse_3_56)), api.Sub(1, cse_3_57)))), api.Sub(circuit.Claim_Virtual_RamValFinal_RamOutputCheck, api.Mul(api.Add(api.Add(api.Mul(api.Mul(cse_3_213, 10), api.Sub(cse_3_219, api.Mul(cse_3_219, cse_3_61))), api.Mul(api.Mul(cse_3_213, 55), api.Sub(cse_3_221, api.Mul(cse_3_221, cse_3_61)))), api.Mul(api.Mul(cse_3_212, 1), api.Sub(cse_3_225, api.Mul(cse_3_225, cse_3_61)))), api.Sub(1, cse_3_55)))), poseidon.Truncate128(api, cse_3_32))), api.Mul(api.Mul(api.Mul(api.Mul(api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_55, cse_3_153), api.Mul(api.Sub(1, cse_3_55), api.Sub(1, cse_3_153)))), api.Mul(1, api.Add(api.Mul(cse_3_56, cse_3_154), api.Mul(api.Sub(1, cse_3_56), api.Sub(1, cse_3_154))))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_57, cse_3_155), api.Mul(api.Sub(1, cse_3_57), api.Sub(1, cse_3_155)))), api.Mul(1, api.Add(api.Mul(cse_3_58, cse_3_156), api.Mul(api.Sub(1, cse_3_58), api.Sub(1, cse_3_156)))))), api.Mul(api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_59, cse_3_157), api.Mul(api.Sub(1, cse_3_59), api.Sub(1, cse_3_157)))), api.Mul(1, api.Add(api.Mul(cse_3_60, cse_3_158), api.Mul(api.Sub(1, cse_3_60), api.Sub(1, cse_3_158))))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_61, cse_3_159), api.Mul(api.Sub(1, cse_3_61), api.Sub(1, cse_3_159)))), api.Mul(api.Mul(1, api.Add(api.Mul(cse_3_62, cse_3_160), api.Mul(api.Sub(1, cse_3_62), api.Sub(1, cse_3_160)))), api.Mul(1, api.Add(api.Mul(cse_3_63, cse_3_161), api.Mul(api.Sub(1, cse_3_63), api.Sub(1, cse_3_161)))))))), api.Add(api.Add(circuit.Claim_Virtual_LookupOutput_InstructionClaimReduction, api.Mul(cse_3_27, circuit.Claim_Virtual_LeftLookupOperand_InstructionClaimReduction)), api.Mul(cse_3_28, circuit.Claim_Virtual_RightLookupOperand_InstructionClaimReduction))), cse_3_80)))
	api.AssertIsEqual(a3, 0)

	return nil
}
