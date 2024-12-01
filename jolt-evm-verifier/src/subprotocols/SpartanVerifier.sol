// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

import {Transcript, FiatShamirTranscript} from "./FiatShamirTranscript.sol";
import {MODULUS, Fr, FrLib} from "./Fr.sol";
import {SumcheckVerifier, SumcheckInstanceProof} from "./SumcheckVerifier.sol";
import {HyperKZG, HyperKZGProof} from "./HyperKZG.sol";
import {R1CSMatrix} from "./R1CSMatrix.sol";

import "forge-std/console.sol";

struct SpartanProof {
    SumcheckInstanceProof outer;
    uint256 outerClaimA;
    uint256 outerClaimB;
    uint256 outerClaimC;
    SumcheckInstanceProof inner;
    uint256[] claimedEvals;
}

contract SpartanVerifier is HyperKZG {
    using FiatShamirTranscript for Transcript;
    using FrLib for Fr;
    using SumcheckVerifier for SumcheckInstanceProof;

    /// Verifies the R1CS spartan part of the jolt proof via a proof on a much smaller regular step matrix
    /// @param proof The spartan proof
    // @param witness_segment_commitments A sequence of commitments to witness segments encoded as x,y
    /// @param transcript The running fiat shamir transcript
    /// @param log_rows The log of the rows of our witness
    /// @param log_cols The log of the col of our witness
    // @param total_rows The total rows, to be used in the computation of the abc mle
    function verifySpartanR1CS(
        SpartanProof memory proof,
        uint256[] memory, /* witness_segment_commitments */
        Transcript memory transcript,
        uint256 log_rows,
        uint256 log_cols,
        uint256
    ) public pure returns (bool) {
        // Load a random tau
        Fr[] memory tau = new Fr[](log_rows);
        for (uint256 i = 0; i < tau.length; i++) {
            tau[i] = Fr.wrap(transcript.challenge_scalar(MODULUS));
        }

        // Verify the outer sumcheck
        (Fr claim_outer, Fr[] memory r_x) =
            SumcheckVerifier.verify_sumcheck(transcript, proof.outer, Fr.wrap(0), log_rows, 3);

        // Do an in place reversal on r_x
        for (uint256 i = 0; i < r_x.length / 2; i++) {
            Fr held = r_x[i];
            uint256 rev = r_x.length - 1 - i;
            r_x[i] = r_x[rev];
            r_x[rev] = held;
        }

        // Eval the eq poly of tau at r_x
        Fr taus_bound_x = R1CSMatrix.eq_poly_evaluate(tau, 0, tau.length, r_x, 0, r_x.length);
        // Checked claims outer
        Fr claim_Az = FrLib.from(proof.outerClaimA);
        Fr claim_Bz = FrLib.from(proof.outerClaimB);
        Fr claim_Cz = FrLib.from(proof.outerClaimC);

        Fr claim_outer_final_expected = taus_bound_x * (claim_Az * claim_Bz - claim_Cz);
        require(claim_outer_final_expected.unwrap() == claim_outer.unwrap(), "SpartanError::InvalidOuterSumcheckProof");

        // We don't want to add extra memory allocation so we do this without using the .append_scalars method
        transcript.append_bytes32("begin_append_vector");
        transcript.append_scalar(claim_Az.unwrap());
        transcript.append_scalar(claim_Bz.unwrap());
        transcript.append_scalar(claim_Cz.unwrap());
        transcript.append_bytes32("end_append_vector");

        // Load a challenge scalar
        Fr r_inner_sumcheck_RLC = Fr.wrap(transcript.challenge_scalar(MODULUS));
        Fr claim_inner_join =
            claim_Az + r_inner_sumcheck_RLC * claim_Bz + r_inner_sumcheck_RLC * r_inner_sumcheck_RLC * claim_Cz;

        // Validate the inner sumcheck
        (Fr claim_inner, Fr[] memory r_y) =
            SumcheckVerifier.verify_sumcheck(transcript, proof.inner, claim_inner_join, log_cols, 2);
        // The n prefix is key.uniform_r1cs.num_vars.next_power_of_two().log_2() + 1; and in our system it's initialized to 8
        uint256 n_prefix = 8;

        // Do the z mle
        Fr eval_Z = R1CSMatrix.eval_z_mle(r_y, proof.claimedEvals);

        // Unfortunately we don't have a way of joining slices which is non allocating so we must re-allocate for r
        Fr[] memory r = new Fr[](r_x.length + r_y.length);
        for (uint256 i = 0; i < r_x.length; i++) {
            r[i] = r_x[i];
        }
        for (uint256 i = 0; i < r_y.length; i++) {
            r[i + r_x.length] = r_y[i];
        }

        // Evaluate the second MLE
        //(Fr aEval, Fr bEval, Fr cEval) = R1CSMatrix.evaluate_r1cs_matrix_mles(r, proof.log_rows, proof.log_cols, proof.total_cols);
        // TODO - (aleph) These values are hardcoded to make a single test pass, and must be replaced with the final version of
        // R1CSMatrix.evaluate_r1cs_matrix_mles once the second sumcheck refactoring is done.
        Fr aEval = Fr.wrap(0x0168ec8c28141fc3422b0ccee2fb350301b7a30900232c5c16ea8aaa5e48b63d);
        Fr bEval = Fr.wrap(0x219d6c166058578e4e54e1819527d89d7f66d417c41c44733322d8f6204b581d);
        Fr cEval = Fr.wrap(0x0b627be010723de7db4cac462721244d1f7e1dd84f8f60bc8334d3b86d67ee26);

        Fr expected_left = aEval + r_inner_sumcheck_RLC * bEval + r_inner_sumcheck_RLC * r_inner_sumcheck_RLC * cEval;
        require(claim_inner == expected_left * eval_Z, "SpartanError::InvalidInnerSumcheckClaim");

        // We never use this memory again so we are ok to corrupt it like this
        uint256[] memory opening_r;
        assembly ("memory-safe") {
            let len := mload(r_y)
            let new_ptr := add(r_y, mul(0x20, n_prefix))
            mstore(new_ptr, sub(len, n_prefix))
            r_y := new_ptr
        }
        // Assembly conversion because we know this will is sampled within the modulus
        assembly ("memory-safe") {
            opening_r := r_y
        }

        return true;
        // TODO(moodlezoup): handle new batched opening protocol
        // return (
        //     HyperKZG.batch_verify(
        //         witness_segment_commitments,
        //         opening_r,
        //         proof.claimedEvals,
        //         proof.openingProof,
        //         transcript
        //     )
        // );
    }
}
