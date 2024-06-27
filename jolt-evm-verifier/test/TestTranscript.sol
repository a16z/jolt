pragma solidity ^0.8.21;

import {TestBase} from "./base/TestBase.sol";
import {FiatShamirTranscript, Transcript} from "../src/subprotocols/FiatShamirTranscript.sol";

import "forge-std/console.sol";

contract TestTranscript is TestBase {

    using FiatShamirTranscript for Transcript;

    function testTranscriptIntegration() public {
        // The values are loaded from a rust run which runs the various function analogues
        TranscriptExampleValues memory transcript = getTranscriptExample();
    }   
}