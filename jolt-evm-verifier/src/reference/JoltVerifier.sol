// SPDX-License-Identifier: MIT

pragma solidity >=0.8.21;

import {IVerifier} from "../interfaces/IVerifier.sol";
import {ITranscript} from "../interfaces/ITranscript.sol";
import {Fr, FrLib} from "./Fr.sol";
import {Jolt} from "./JoltTypes.sol";
import {UniPoly, UniPolyLib} from "./UniPoly.sol";

import "forge-std/console.sol";

error GrandProductArgumentFailed();
error SumcheckFailed();

contract JoltVerifier is IVerifier {

    ITranscript transcript;

    constructor(ITranscript _transcript) {

        transcript = _transcript;
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
            Fr r_i = transcript.challengeScalar("challenge_nextround", 18);
            r[i] = r_i;

            e = UniPolyLib.evaluate(poly, r_i);

        }
        return (e, r);
    }

    function verifyGrandProduct(Jolt.BatchedGrandProductProof memory proof, Fr[] memory claims) external returns (bool verified) {


        for (uint256 i=0; i < proof.layers.length; i++){

            if (claims.length != proof.layers[i].left_claims.length || claims.length != proof.layers[i].right_claims.length){
                revert GrandProductArgumentFailed();
            }

            // get coeffs to created a joined claim
            Fr joined_claim = Fr.wrap(0);
            for (uint j=0; j < claims.length; j++) {
                // TODO change this challengeVector
                joined_claim = joined_claim + (claims[j] * transcript.challengeScalar("rand_coeffs_next_layer", j + i*4));
            }

            // verify sumcheck and get rSumcheck
            (Fr sumcheckClaim, Fr[] memory rSumcheck) = verifySumcheckLayer(proof.layers[i], joined_claim, 3, i);


            // TODO: append left and right claims to transcript

            // TODO: eq_eval 

            // TODO: verify sumcheck claim we got earlier
        }


        return true;
    }
}
