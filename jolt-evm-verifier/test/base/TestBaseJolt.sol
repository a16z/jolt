// SPDX-License-Identifier: MIT

pragma solidity ^0.8.21;

import {TestBase} from "./TestBase.sol";
import {IVerifier} from "../../src/interfaces/IVerifier.sol";
import {ITranscript} from "../../src/interfaces/ITranscript.sol";
import {Jolt} from "../../src/reference/JoltTypes.sol";
import {JoltVerifier} from "../../src/reference/JoltVerifier.sol";
import {JoltTranscript} from "../../src/reference/JoltTranscript.sol";
import {Fr} from "../../src/reference/Fr.sol";

import "forge-std/console.sol";

contract TestBaseJolt is TestBase {
    IVerifier public verifier;
    ITranscript public transcript;

    function testValidGrandProductProof() public {

        transcript = ITranscript(address(new JoltTranscript()));

        verifier = IVerifier(address(new JoltVerifier(transcript)));

        Jolt.BatchedGrandProductProof memory proof = getProofData();
        Fr[] memory proverClaims = getClaims();

        Fr[] memory proverRGrandProduct = getProverRGrandProduct();

        Fr[] memory verifierRGrandProduct = verifier.verifyGrandProduct(proof, proverClaims);


        for (uint256 i=0; i < verifierRGrandProduct.length; i++) {
            assertTrue(proverRGrandProduct[i] == verifierRGrandProduct[i]);
        }
    }

}
