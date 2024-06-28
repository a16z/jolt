// SPDX-License-Identifier: MIT

pragma solidity >=0.8.21;

import {IVerifier} from "../interfaces/IVerifier.sol";
import {ITranscript} from "../interfaces/ITranscript.sol";
import {Fr, FrLib, sub} from "./Fr.sol";
import {Jolt} from "./JoltTypes.sol";
import {UniPoly, UniPolyLib} from "./UniPoly.sol";

import "forge-std/console.sol";

error GrandProductArgumentFailed();
error SumcheckFailed();

contract JoltVerifier is IVerifier {

    ITranscript transcript;
    uint256 transcriptIndex;

    constructor(ITranscript _transcript) {

        transcript = _transcript;
        transcriptIndex = 0;
    }

    function verifySumcheckLayer(Jolt.BatchedGrandProductLayerProof memory layer, Fr claim, uint256 degree_bound, uint256 num_rounds) internal returns (Fr, Fr[] memory){

        if (layer.sumcheck_univariate_coeffs.length != num_rounds){
            revert SumcheckFailed();
        }

        Fr e = claim;
        Fr[] memory r = new Fr[](num_rounds);

        for (uint i=0; i < num_rounds; i++){
            UniPoly memory poly = UniPolyLib.decompress(layer.sumcheck_univariate_coeffs[i], e);

            //verify degree bound
            if (layer.sumcheck_univariate_coeffs[i].length != degree_bound){
                revert SumcheckFailed();
            }

            // check if G_k(0) + G_k(1) = e
            if (UniPolyLib.evalAtZero(poly) + UniPolyLib.evalAtOne(poly) != e){
                revert SumcheckFailed();
            }

            //TODO: append to transcript

            // evaluate at r_i
            Fr r_i = transcript.challengeScalar("challenge_nextround", transcriptIndex);
            transcriptIndex++;
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
        Fr[] memory rGrandProduct
    ) internal returns (Fr[] memory newClaims, Fr[] memory newRGrandProduct) {
        Jolt.BatchedGrandProductLayerProof memory layerProof = layerProofs[layerIndex];


        Fr expectedSumcheckClaim = Fr.wrap(0);
        
        for (uint256 i = 0; i < claims.length; i++) {
            expectedSumcheckClaim = expectedSumcheckClaim + coeffs[i] * layerProof.leftClaims[i] * layerProof.rightClaims[i] * eqEval;
        }
        
        require(expectedSumcheckClaim == sumcheckClaim, "Sumcheck claim mismatch");

        // produce a random challenge to condense two claims into a single claim
        Fr rLayer = transcript.challengeScalar("challenge_r_layer", transcriptIndex);
        transcriptIndex++;

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


    function verifyGrandProduct(Jolt.BatchedGrandProductProof memory proof, Fr[] memory claims) external returns (Fr[] memory) {


        Fr[] memory rGrandProduct = new Fr[](0);
        for (uint256 i=0; i < proof.layers.length; i++){

            //get coeffs
            Fr[] memory coeffs = new Fr[](claims.length);
            for (uint j = 0; j < claims.length; j++) {
                coeffs[j] = transcript.challengeScalar("rand_coeffs_next_layer", transcriptIndex);
                transcriptIndex++;
            }

            //create a joined claim
            Fr joined_claim = Fr.wrap(0);
            for (uint k = 0; k < claims.length; k++) {
                joined_claim = joined_claim + (claims[k] * coeffs[k]);
            }

            if (claims.length != proof.layers[i].leftClaims.length || claims.length != proof.layers[i].rightClaims.length){
                revert GrandProductArgumentFailed();
            }

            // verify sumcheck and get rSumcheck
            (Fr sumcheckClaim, Fr[] memory rSumcheck) = verifySumcheckLayer(proof.layers[i], joined_claim, 3, i);

            if (rSumcheck.length != rGrandProduct.length) {
                revert GrandProductArgumentFailed();
            }

            // TODO: append left and right claims to transcript

            Fr eqEval = buildEqEval(rGrandProduct, rSumcheck);

            rGrandProduct = new Fr[](rSumcheck.length);
            for (uint256 l=0; l < rSumcheck.length; l++) {
                rGrandProduct[l] = rSumcheck[rSumcheck.length - 1 - l];
            }

            (claims, rGrandProduct) = verifySumcheckClaim(proof.layers, i, coeffs, sumcheckClaim, eqEval, claims, rGrandProduct);
        }


        return rGrandProduct;
    }
}
