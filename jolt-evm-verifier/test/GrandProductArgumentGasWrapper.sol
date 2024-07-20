// SPDX-License-Identifier: MIT
pragma solidity ^0.8.21;

import {Jolt} from "../src/reference/JoltTypes.sol";
import {Fr, FrLib, sub, MODULUS} from "../src/reference/Fr.sol";
import {Transcript, FiatShamirTranscript} from "../src/subprotocols/FiatShamirTranscript.sol";
import {UniPoly, UniPolyLib} from "../src/reference/UniPoly.sol";
import {GrandProductArgument} from "../src/reference/JoltVerifier.sol";
import {TestBase} from "./base/TestBase.sol";

import "forge-std/console.sol";

error GrandProductArgumentFailed();
error SumcheckFailed();

contract GrandProductArgumentGasWrapper is TestBase {
    using FiatShamirTranscript for Transcript;

    bool private enableLogging = vm.envOr("BENCHMARK_LOGGING", false);

    function conditionalLog(string memory message) internal view {
        if (enableLogging) {
            console.log(message);
        }
    }

    function conditionalLog(string memory message, uint256 value) internal view {
        if (enableLogging) {
            console.log(message, value);
        }
    }

    function verifySumcheckLayer(
        Jolt.BatchedGrandProductLayerProof memory layer,
        Transcript memory transcript,
        Fr claim,
        uint256 degree_bound,
        uint256 num_rounds
    ) public pure returns (Fr, Fr[] memory) {
        return GrandProductArgument.verifySumcheckLayer(layer, transcript, claim, degree_bound, num_rounds);
    }

    function buildEqEval(Fr[] memory rGrandProduct, Fr[] memory rSumcheck) public pure returns (Fr eqEval) {
        return GrandProductArgument.buildEqEval(rGrandProduct, rSumcheck);
    }

    function verifySumcheckClaim(
        Jolt.BatchedGrandProductLayerProof[] memory layerProofs,
        uint256 layerIndex,
        Fr[] memory coeffs,
        Fr sumcheckClaim,
        Fr eqEval,
        Fr[] memory claims,
        Fr[] memory rGrandProduct,
        Transcript memory transcript
    ) public pure returns (Fr[] memory newClaims, Fr[] memory newRGrandProduct) {
        return GrandProductArgument.verifySumcheckClaim(
            layerProofs, layerIndex, coeffs, sumcheckClaim, eqEval, claims, rGrandProduct, transcript
        );
    }

    function verify(Jolt.BatchedGrandProductProof memory proof, Fr[] memory claims, Transcript memory transcript)
        external
        view
        returns (Fr[] memory)
    {
        Fr[] memory rGrandProduct = new Fr[](0);
        uint256 gasUsed = gasleft();

        for (uint256 i = 0; i < proof.layers.length; i++) {
            conditionalLog("[grand-product] round:", i);
            //get coeffs
            uint256[] memory loaded = transcript.challenge_scalars(claims.length, MODULUS);
            Fr[] memory coeffs;
            // TODO - hard convert should be removed when the transcript gets better Fr native typed support.
            assembly {
                coeffs := loaded
            }

            //create a joined claim
            Fr joined_claim = Fr.wrap(0);
            for (uint256 k = 0; k < claims.length; k++) {
                joined_claim = joined_claim + (claims[k] * coeffs[k]);
            }

            if (
                claims.length != proof.layers[i].leftClaims.length
                    || claims.length != proof.layers[i].rightClaims.length
            ) {
                revert GrandProductArgumentFailed();
            }

            gasUsed = gasleft();
            // verify sumcheck and get rSumcheck
            (Fr sumcheckClaim, Fr[] memory rSumcheck) =
                verifySumcheckLayer(proof.layers[i], transcript, joined_claim, 3, i);
            gasUsed -= gasleft();
            conditionalLog("[grand-product] verifySumcheckLayer gas usage:", gasUsed);

            if (rSumcheck.length != rGrandProduct.length) {
                revert GrandProductArgumentFailed();
            }

            // Append the right and left claims to the transcript
            for (uint256 l = 0; l < proof.layers[l].leftClaims.length; l++) {
                transcript.append_scalar(Fr.unwrap(proof.layers[i].leftClaims[l]));
                transcript.append_scalar(Fr.unwrap(proof.layers[i].rightClaims[l]));
            }

            gasUsed = gasleft();
            Fr eqEval = buildEqEval(rGrandProduct, rSumcheck);
            gasUsed -= gasleft();
            conditionalLog("[grand-product] buildEqEval gas usage:", gasUsed);

            rGrandProduct = new Fr[](rSumcheck.length);
            for (uint256 l = 0; l < rSumcheck.length; l++) {
                rGrandProduct[l] = rSumcheck[rSumcheck.length - 1 - l];
            }

            gasUsed = gasleft();
            (claims, rGrandProduct) =
                verifySumcheckClaim(proof.layers, i, coeffs, sumcheckClaim, eqEval, claims, rGrandProduct, transcript);
            gasUsed -= gasleft();
            conditionalLog("[grand-product] verifySumcheckClaim gas usage:", gasUsed);
        }

        return rGrandProduct;
    }
}
