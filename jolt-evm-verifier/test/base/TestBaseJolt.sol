// SPDX-License-Identifier: MIT

pragma solidity ^0.8.21;

import {TestBase} from "./TestBase.sol";
import {IVerifier} from "../../src/interfaces/IVerifier.sol";
import {Jolt} from "../../src/reference/JoltTypes.sol";
import {JoltVerifier} from "../../src/reference/JoltVerifier.sol";
import {Transcript, FiatShamirTranscript} from "../../src/subprotocols/FiatShamirTranscript.sol";
import {Fr} from "../../src/reference/Fr.sol";

import "forge-std/console.sol";

contract TestBaseJolt is TestBase {
    IVerifier public verifier;

    function testValidGrandProductProof() public {
        // Inits the transcript with the same string label as the rust code
        Transcript memory transcript = FiatShamirTranscript.new_transcript("test_transcript", 4);

        verifier = IVerifier(address(new JoltVerifier()));

        (Jolt.BatchedGrandProductProof memory proof, uint256[] memory claims, uint256[] memory r) = getProofData();

        Fr[] memory claims_fr;
        assembly ("memory-safe") {
            claims_fr := claims
        }

        Fr[] memory verifierRGrandProduct = verifier.verifyGrandProduct(proof, claims_fr, transcript);

        for (uint256 i = 0; i < verifierRGrandProduct.length; i++) {
            assertTrue(r[i] == Fr.unwrap(verifierRGrandProduct[i]));
        }
    }
}
