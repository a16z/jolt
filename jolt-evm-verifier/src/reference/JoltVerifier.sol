// SPDX-License-Identifier: MIT

pragma solidity >=0.8.21;

import {Transcript, FiatShamirTranscript} from "../subprotocols/FiatShamirTranscript.sol";
import {Fr, FrLib, sub, MODULUS} from "./Fr.sol";
import {Jolt} from "./JoltTypes.sol";
import {UniPoly, UniPolyLib} from "./UniPoly.sol";

import "forge-std/console.sol";

error GrandProductArgumentFailed();
error SumcheckFailed();

library GrandProductArgument {
    using FiatShamirTranscript for Transcript;

    function verifySumcheckLayer(
        Jolt.BatchedGrandProductLayerProof memory layer,
        Transcript memory transcript,
        Fr claim,
        uint256 degree_bound,
        uint256 num_rounds
    ) internal pure returns (Fr, Fr[] memory) {
        if (layer.sumcheck_univariate_coeffs.length != num_rounds) {
            revert SumcheckFailed();
        }

        Fr e = claim;
        Fr[] memory r = new Fr[](num_rounds);

        for (uint256 i = 0; i < num_rounds; i++) {
            UniPoly memory poly = UniPolyLib.decompress(layer.sumcheck_univariate_coeffs[i], e);

            //verify degree bound
            if (layer.sumcheck_univariate_coeffs[i].length != degree_bound) {
                revert SumcheckFailed();
            }

            // check if G_k(0) + G_k(1) = e
            if (UniPolyLib.evalAtZero(poly) + UniPolyLib.evalAtOne(poly) != e) {
                revert SumcheckFailed();
            }

            // TODO - We can move this into the transcript lib
            transcript.append_bytes32("UniPoly_begin");
            for (uint256 j = 0; j < poly.coeffs.length; j++) {
                transcript.append_scalar(Fr.unwrap(poly.coeffs[j]));
            }
            transcript.append_bytes32("UniPoly_end");

            // Sample random from the transcript
            Fr r_i = Fr.wrap(transcript.challenge_scalar(MODULUS));
            r[i] = r_i;

            e = UniPolyLib.evaluate(poly, r_i);
        }
        return (e, r);
    }

    function buildEqEval(Fr[] memory rGrandProduct, Fr[] memory rSumcheck) internal pure returns (Fr eqEval) {
        eqEval = FrLib.from(1); // Start with the multiplicative identity

        for (uint256 i = 0; i < rGrandProduct.length; i++) {
            Fr rGp = rGrandProduct[i];
            Fr rSc = rSumcheck[rSumcheck.length - 1 - i]; // Reverse order for rSumcheck

            // Calculate: rGp * rSc + (1 - rGp) * (1 - rSc)
            Fr term = rGp * rSc + (sub(Fr.wrap(1), rGp)) * sub(Fr.wrap(1), rSc);

            // Multiply the result
            eqEval = eqEval * term;
        }

        return eqEval;
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
    ) internal pure returns (Fr[] memory newClaims, Fr[] memory newRGrandProduct) {
        Jolt.BatchedGrandProductLayerProof memory layerProof = layerProofs[layerIndex];

        Fr expectedSumcheckClaim = Fr.wrap(0);

        for (uint256 i = 0; i < claims.length; i++) {
            expectedSumcheckClaim =
                expectedSumcheckClaim + coeffs[i] * layerProof.leftClaims[i] * layerProof.rightClaims[i] * eqEval;
        }

        require(expectedSumcheckClaim == sumcheckClaim, "Sumcheck claim mismatch");

        // produce a random challenge to condense two claims into a single claim
        Fr rLayer = Fr.wrap(transcript.challenge_scalar(MODULUS));

        newClaims = new Fr[](claims.length);
        for (uint256 i = 0; i < claims.length; i++) {
            newClaims[i] = layerProof.leftClaims[i] + rLayer * (layerProof.rightClaims[i] - layerProof.leftClaims[i]);
        }

        newRGrandProduct = new Fr[](rGrandProduct.length + 1);
        for (uint256 i = 0; i < rGrandProduct.length; i++) {
            newRGrandProduct[i] = rGrandProduct[i];
        }
        newRGrandProduct[rGrandProduct.length] = rLayer;
        return (newClaims, newRGrandProduct);
    }

    function verify(
        Jolt.BatchedGrandProductProof memory proof,
        Fr[] memory claims,
        Transcript memory transcript
    ) public pure returns (Fr[] memory) {
        Fr[] memory rGrandProduct = new Fr[](0);

        for (uint256 i = 0; i < proof.layers.length; i++) {
            //get coeffs
            uint256[] memory loaded = transcript.challenge_scalars(claims.length, MODULUS);
            Fr[] memory coeffs;
            // TODO - This hard convert should be removed when the transcript gets better Fr native typed support.
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

            // verify sumcheck and get rSumcheck
            (Fr sumcheckClaim, Fr[] memory rSumcheck) =
                verifySumcheckLayer(proof.layers[i], transcript, joined_claim, 3, i);

            if (rSumcheck.length != rGrandProduct.length) {
                revert GrandProductArgumentFailed();
            }

            // Append the right and left claims to the transcript
            for (uint256 l = 0; l < proof.layers[l].leftClaims.length; l++) {
                transcript.append_scalar(Fr.unwrap(proof.layers[i].leftClaims[l]));
                transcript.append_scalar(Fr.unwrap(proof.layers[i].rightClaims[l]));
            }

            Fr eqEval = buildEqEval(rGrandProduct, rSumcheck);

            rGrandProduct = new Fr[](rSumcheck.length);
            for (uint256 l = 0; l < rSumcheck.length; l++) {
                rGrandProduct[l] = rSumcheck[rSumcheck.length - 1 - l];
            }

            (claims, rGrandProduct) =
                verifySumcheckClaim(proof.layers, i, coeffs, sumcheckClaim, eqEval, claims, rGrandProduct, transcript);
        }

        return rGrandProduct;
    }
}
