// SPDX-License-Identifier: MIT

pragma solidity >=0.8.21;

import {Transcript, FiatShamirTranscript} from "./FiatShamirTranscript.sol";
import {Fr, FrLib, sub, MODULUS} from "./Fr.sol";
import {SumcheckInstanceProof, SumcheckVerifier} from "./SumcheckVerifier.sol";

error GrandProductArgumentFailed();
error SumcheckFailed();

struct GKRLayer {
    SumcheckInstanceProof sumcheck;
    uint256[] leftClaims;
    uint256[] rightClaims;
}

struct GrandProductProof {
    GKRLayer[] layers;
}

library GrandProductVerifier {
    using FiatShamirTranscript for Transcript;

    function evalEqMLE(Fr[] memory rGrandProduct, Fr[] memory rSumcheck) internal pure returns (Fr eqEval) {
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
        GKRLayer memory layerProof,
        Fr[] memory coeffs,
        Fr sumcheckClaim,
        Fr eqEval,
        Fr[] memory claims,
        Fr[] memory rGrandProduct,
        Transcript memory transcript
    ) internal pure returns (Fr[] memory newClaims, Fr[] memory newRGrandProduct) {
        Fr expectedSumcheckClaim = Fr.wrap(0);

        for (uint256 i = 0; i < claims.length; i++) {
            expectedSumcheckClaim = expectedSumcheckClaim
                + coeffs[i] * FrLib.from(layerProof.leftClaims[i]) * FrLib.from(layerProof.rightClaims[i]) * eqEval;
        }

        require(expectedSumcheckClaim == sumcheckClaim, "Sumcheck claim mismatch");

        // produce a random challenge to condense two claims into a single claim
        Fr rLayer = Fr.wrap(transcript.challenge_scalar(MODULUS));

        newClaims = new Fr[](claims.length);
        for (uint256 i = 0; i < claims.length; i++) {
            Fr checked_left = FrLib.from(layerProof.leftClaims[i]);
            newClaims[i] = checked_left + rLayer * (FrLib.from(layerProof.rightClaims[i]) - checked_left);
        }

        newRGrandProduct = new Fr[](rGrandProduct.length + 1);
        for (uint256 i = 0; i < rGrandProduct.length; i++) {
            newRGrandProduct[i] = rGrandProduct[i];
        }
        newRGrandProduct[rGrandProduct.length] = rLayer;
        return (newClaims, newRGrandProduct);
    }

    function verifyGrandProduct(GrandProductProof memory proof, Fr[] memory claims, Transcript memory transcript)
        external
        pure
        returns (Fr[] memory)
    {
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
                SumcheckVerifier.verify_sumcheck(transcript, proof.layers[i].sumcheck, joined_claim, i, 3);

            if (rSumcheck.length != rGrandProduct.length) {
                revert GrandProductArgumentFailed();
            }

            // Append the right and left claims to the transcript
            for (uint256 l = 0; l < proof.layers[l].leftClaims.length; l++) {
                transcript.append_scalar(proof.layers[i].leftClaims[l] % MODULUS);
                transcript.append_scalar(proof.layers[i].rightClaims[l] % MODULUS);
            }

            Fr eqEval = evalEqMLE(rGrandProduct, rSumcheck);

            rGrandProduct = new Fr[](rSumcheck.length);
            for (uint256 l = 0; l < rSumcheck.length; l++) {
                rGrandProduct[l] = rSumcheck[rSumcheck.length - 1 - l];
            }

            (claims, rGrandProduct) =
                verifySumcheckClaim(proof.layers[i], coeffs, sumcheckClaim, eqEval, claims, rGrandProduct, transcript);
        }

        return rGrandProduct;
    }
}
