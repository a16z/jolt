// SPDX-License-Identifier: MIT

pragma solidity >=0.8.21;

import {IVerifier} from "../interfaces/IVerifier.sol";
import {Fr, FrLib} from "./Fr.sol";
import {Jolt} from "./JoltTypes.sol";

import "forge-std/console.sol";

error GrandProductArgumentFailed();

contract JoltVerifier is IVerifier {

    //function verifySumcheckClaim(BatchedGrandProductLayerProof memory layer, Fr[] memory coeffs, )

    function verifyGrandProduct(Jolt.BatchedGrandProductProof memory proof, Fr[] memory claims) external view returns (bool verified) {

        for (uint256 i=0; i < proof.layers.length; i++){
            console.log(proof.layers[i].sumcheck_univariate_coeffs.length);

            if (claims.length != proof.layers[i].left_claims.length || claims.length != proof.layers[i].right_claims.length){
                revert GrandProductArgumentFailed();
            }
        }


        return true;
    }
}
